"""Tests for the BGA human-games pipeline (``ludometer/human/``).

Nothing here touches the network. The end-to-end coverage comes from
:mod:`ludometer.human.fixture`, which plays a real game in our engine and writes
it out in the JSON shape a BGA replay log has: parsing that back and replaying it
exercises the same code path a real download will, so the mapping conventions,
the scripted deals, the strict validation and the ``replay.npz`` writer are all
tested against a game whose truth we know exactly.

The negative tests matter as much as the positive one: the whole reason the
converter exists is to *reject* games it cannot reproduce, so a corrupted pick, a
corrupted deal, a permuted colour map and an unknown notification each have to be
caught rather than absorbed.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import re
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pytest

from ludometer.azul.engine import ACTION_SPACE, ENCODED_SIZE, decode_action
from ludometer.human.client import (
    AccountDisabled,
    AuthRequired,
    BgaClient,
    BgaError,
    ClientConfig,
    ReplayLimitReached,
    ReplayUnavailable,
    display_elo,
    endpoints,
    raw_elo,
    read_json_gz,
    write_json_gz,
)
from ludometer.human.convert import (
    ConversionError,
    convert_game,
    solve_color_map,
    solve_color_map_over,
)
from ludometer.human.dataset import (
    OPPONENT_VALUE_ONLY,
    GameMeta,
    build_dataset,
    one_hot_policies,
)
from ludometer.human.fetch import (
    ARENA_MODE,
    GAME_MODE_OPTION,
    CrawlPace,
    Fetcher,
    FetchState,
    PlayerRow,
    TableFilter,
    TableVerdict,
    choose_target_player,
    elo_after_map,
    extract_table_ids,
    extract_table_rows,
    option_value,
    player_elos,
    select_players,
    table_row_players,
    table_row_scores,
)
from ludometer.human.fixture import synthetic_log
from ludometer.human.parse import (
    AZUL_COLOR_MAP,
    DEFAULT_SCHEMA,
    ParseError,
    conceder_from_infos,
    iter_log_entries,
    log_packets,
    log_type_histogram,
    observed_color_ids,
    parse_gamelogs_html,
    parse_log,
    scores_from_infos,
    with_color_map,
)
from ludometer.train.replay import ReplayBuffer

TABLE_ID = 999_000_001
PLAYERS = (91843016, 91718783)


def _fixture(seed: int = 7, swap_seats: bool = False):
    return synthetic_log(
        seed=seed, player_ids=PLAYERS, table_id=TABLE_ID, swap_seats=swap_seats
    )


# --------------------------------------------------------------------- happy path
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7])
def test_a_synthetic_replay_round_trips_through_the_engine(seed: int) -> None:
    """Parse -> convert reproduces the engine game the fixture actually played."""
    game, payload, infos = _fixture(seed=seed)
    replay = parse_log(payload, TABLE_ID, PLAYERS, infos=infos)

    assert len(replay.picks) == len(game.actions)
    assert [p.action_id() for p in replay.picks] == game.actions
    assert len(replay.deals) == len(game.deals)

    converted = convert_game(replay)
    assert converted.actions.tolist() == game.actions
    assert converted.movers.tolist() == game.movers
    assert converted.scores == game.scores
    assert converted.outcome == game.outcome
    assert converted.states.shape == (len(game.actions), ENCODED_SIZE)


def test_the_first_mover_does_not_have_to_be_seat_zero() -> None:
    """``swap_seats`` puts the log's first mover in engine seat 1."""
    game, payload, infos = _fixture(seed=5, swap_seats=True)
    converted = convert_game(parse_log(payload, TABLE_ID, PLAYERS, infos=infos))
    # seats are mirrored, so scores and outcome mirror too
    assert converted.scores == tuple(reversed(game.scores))
    assert converted.outcome == -game.outcome
    assert converted.movers.tolist() == [1 - m for m in game.movers]


def test_every_position_is_the_state_the_human_moved_in() -> None:
    """The recorded state must be the one whose legal moves contain the action."""
    _game, payload, infos = _fixture(seed=11)
    converted = convert_game(parse_log(payload, TABLE_ID, PLAYERS, infos=infos))
    # the encoded "tiles left this round" feature must never be zero: a position
    # with an empty board is a round boundary, which is not a decision point
    tiles_left = converted.states[:, 173]
    assert np.all(tiles_left > 0.0)
    # and the action's source must hold at least one tile of the chosen colour
    for row, action in zip(converted.states, converted.actions, strict=True):
        source, color, _dest = decode_action(int(action))
        if source == 5:
            assert row[156 + color] > 0.0
        else:
            assert row[126 + source * 5 + color] > 0.0


# ------------------------------------------------------------------ strictness
def test_an_illegal_pick_is_rejected_not_absorbed() -> None:
    _game, payload, infos = _fixture(seed=3)
    broken = copy.deepcopy(payload)
    for packet in broken["data"]["data"]:
        entry = packet["data"][0]
        if entry["type"] == "tilesSelected":
            # swap in a tile type that is not on that factory
            wrong = 1 + (int(entry["args"]["type"]) % 5)
            entry["args"]["type"] = wrong
            for tile in entry["args"]["selectedTiles"]:
                tile["type"] = wrong
            break
    replay = parse_log(broken, TABLE_ID, PLAYERS, infos=infos)
    with pytest.raises(ConversionError, match="illegal action|log says seat"):
        convert_game(replay)


def test_a_deal_that_breaks_tile_conservation_is_rejected() -> None:
    """25 tiles of one colour cannot exist — only 20 of each are in the box.

    This is the check that a mis-parsed deal (wrong colour key, doubled list, a
    factory index collision) trips before a single move is replayed.
    """
    _game, payload, infos = _fixture(seed=4)
    broken = copy.deepcopy(payload)
    for packet in broken["data"]["data"]:
        entry = packet["data"][0]
        if entry["type"] == "factoriesFilled":
            # 1-based deal: entry 0 is the center/marker slot, entries 1..5 the
            # factories. Five factories of five blacks = 25, which cannot exist.
            entry["args"]["factories"] = [[0]] + [[1, 1, 1, 1, 1]] * 5
            break
    replay = parse_log(broken, TABLE_ID, PLAYERS, infos=infos)
    with pytest.raises(ConversionError, match="off-board"):
        convert_game(replay)


def test_a_deal_of_the_wrong_tiles_is_caught_even_when_it_conserves_tiles() -> None:
    """A deal that is *possible* but not what happened fails on the next move."""
    _game, payload, infos = _fixture(seed=4)
    broken = copy.deepcopy(payload)
    for packet in broken["data"]["data"]:
        entry = packet["data"][0]
        if entry["type"] == "factoriesFilled":
            # 1-based deal (center slot first): all 20 blacks, a legal deal that is
            # not what actually happened, so the first move is illegal.
            entry["args"]["factories"] = [[0]] + [[1, 1, 1, 1]] * 5
            break
    replay = parse_log(broken, TABLE_ID, PLAYERS, infos=infos)
    with pytest.raises(ConversionError, match="illegal action"):
        convert_game(replay)


def test_a_truncated_log_is_rejected() -> None:
    """A log that stops mid-game must not become half a game of training data."""
    _game, payload, infos = _fixture(seed=6)
    broken = copy.deepcopy(payload)
    broken["data"]["data"] = broken["data"]["data"][: len(broken["data"]["data"]) // 2]
    replay = parse_log(broken, TABLE_ID, PLAYERS, infos=infos)
    with pytest.raises(ConversionError):
        convert_game(replay)


def test_a_score_mismatch_is_rejected() -> None:
    """The reported-score check is the backstop against a plausible-but-wrong map."""
    _game, payload, infos = _fixture(seed=8)
    broken = copy.deepcopy(payload)
    for packet in broken["data"]["data"]:
        entry = packet["data"][0]
        if entry["type"] == "score":
            entry["args"]["score"] = int(entry["args"]["score"]) + 5
            break
    replay = parse_log(broken, TABLE_ID, PLAYERS, infos=infos)
    with pytest.raises(ConversionError, match="BGA reported"):
        convert_game(replay)
    # ...and the same game converts when the check is off, proving the check is
    # what rejected it and not something else
    assert len(convert_game(replay, check_scores=False)) > 0


def test_a_first_player_marker_placement_is_not_a_pick() -> None:
    """BGA emits a standalone marker-to-floor `tilesPlacedOnLine` (real, 2026-08-17).

    When a player first takes from the center, the log carries a `tilesPlacedOnLine`
    with the first-player marker in `discardedTiles`, `placedTiles` empty and `type`
    0, *before* that turn's `tilesSelected`. It is not a move — the engine assigns
    the marker itself — so the parser must skip it and reproduce the same game.
    """
    _game, payload, infos = _fixture(seed=7)
    baseline = parse_log(payload, TABLE_ID, PLAYERS, infos=infos)
    injected = copy.deepcopy(payload)
    marker_packet = {
        "channel": f"/table/t{TABLE_ID}",
        "move_id": "1",
        "data": [
            {
                "type": "tilesPlacedOnLine",
                "args": {
                    "playerId": PLAYERS[0],
                    "type": 0,
                    "line": 0,
                    "placedTiles": [],
                    "discardedTiles": [
                        {"id": 19, "type": 0, "location": "factory", "line": 0}
                    ],
                },
            }
        ],
    }
    # insert right after the opening deal, before the first real selection
    injected["data"]["data"].insert(1, marker_packet)
    after = parse_log(injected, TABLE_ID, PLAYERS, infos=infos)
    assert [p.action_id() for p in after.picks] == [
        p.action_id() for p in baseline.picks
    ]
    assert convert_game(after).scores == convert_game(baseline).scores


def _table_pkt(entry: dict, move_id: int = 99) -> dict:
    """One table-channel packet wrapping a single notification ``entry``."""
    return {"channel": f"/table/t{TABLE_ID}", "move_id": str(move_id), "data": [entry]}


def _tile_of(bga_type: int) -> dict:
    return {"id": 0, "type": bga_type, "column": 0, "line": 0, "location": "factory"}


def test_an_undone_turn_is_rewound_and_leaves_the_game_unchanged() -> None:
    """A take/place a player took back must vanish from the reconstructed moves.

    BGA emits one notification per undo step, in reverse order: ``undoSelectLine``
    (the placement, tiles back to hand) then ``undoTakeTiles`` (the take). Both were
    the dominant reason real replays failed to validate — an ignored `undoSelectLine`
    leaves the following re-placement looking like "tiles placed without a
    selection", and an ignored `undoTakeTiles` leaves a phantom floor pick. Here a
    whole bogus turn is played and then fully undone before the first real move; the
    converted game must be identical to the untouched one.
    """
    _game, payload, infos = _fixture(seed=7)
    baseline = convert_game(parse_log(payload, TABLE_ID, PLAYERS, infos=infos))
    injected = copy.deepcopy(payload)
    who = PLAYERS[0]
    undo_block = [
        _table_pkt(
            {
                "type": "tilesSelected",
                "args": {
                    "player_id": str(who),
                    "fromFactory": 1,
                    "type": 1,
                    "selectedTiles": [_tile_of(1), _tile_of(1)],
                    "discardedTiles": [],
                },
            }
        ),
        _table_pkt(
            {
                "type": "tilesPlacedOnLine",
                "args": {
                    "player_id": str(who),
                    "line": 0,
                    "placedTiles": [_tile_of(1), _tile_of(1)],
                    "discardedTiles": [],
                },
            }
        ),
        _table_pkt({"type": "undoSelectLine", "args": {"playerId": who}}),
        _table_pkt({"type": "undoTakeTiles", "args": {"playerId": who}}),
    ]
    # right after the opening deal, before the first real selection
    injected["data"]["data"][1:1] = undo_block
    replay = parse_log(injected, TABLE_ID, PLAYERS, infos=infos)
    after = convert_game(replay)
    assert list(after.actions) == list(baseline.actions)
    assert after.scores == baseline.scores
    assert after.outcome == baseline.outcome


def test_an_undone_placement_reopens_the_turn_so_a_re_placement_is_legal() -> None:
    """``undoSelectLine`` puts the tiles back in hand: the take is still standing.

    So a re-placement (the common case — a player changes only which line the tiles
    go on) must close the same open turn, not raise "tiles placed without a
    selection". Re-placing on the same line reproduces the original game exactly.
    """
    _game, payload, infos = _fixture(seed=5)
    baseline = convert_game(parse_log(payload, TABLE_ID, PLAYERS, infos=infos))
    injected = copy.deepcopy(payload)
    pkts = injected["data"]["data"]
    first_place = next(
        i for i, p in enumerate(pkts) if p["data"][0]["type"] == "tilesPlacedOnLine"
    )
    place_entry = copy.deepcopy(pkts[first_place]["data"][0])
    # undo that placement, then re-place identically
    pkts[first_place + 1 : first_place + 1] = [
        _table_pkt({"type": "undoSelectLine", "args": {"playerId": PLAYERS[0]}}),
        _table_pkt(place_entry),
    ]
    after = convert_game(parse_log(injected, TABLE_ID, PLAYERS, infos=infos))
    assert list(after.actions) == list(baseline.actions)
    assert after.scores == baseline.scores


def test_an_undo_with_nothing_to_rewind_is_a_clean_parse_error() -> None:
    """The rewind is strict: an undo that cannot be matched is refused, not ignored."""
    _game, payload, infos = _fixture(seed=7)
    injected = copy.deepcopy(payload)
    # an undoTakeTiles with no open selection (placed right after the deal)
    injected["data"]["data"].insert(
        1, _table_pkt({"type": "undoTakeTiles", "args": {"playerId": PLAYERS[0]}})
    )
    with pytest.raises(ParseError, match="no open selection to undo"):
        parse_log(injected, TABLE_ID, PLAYERS, infos=infos)


def test_a_conceded_game_ends_at_the_resignation_with_the_conceder_losing() -> None:
    """A resignation stops the game: the moves so far are kept, the conceder loses.

    Real replays carry a `playerConcedeGame` as the last move notification, and BGA
    then reports only a nominal 1-0 result, so the terminal-state and score-match
    guards cannot apply. The kept picks must be the prefix actually played, and the
    outcome must be a loss for whoever resigned regardless of the board score.
    """
    game, payload, infos = _fixture(seed=7)
    baseline = parse_log(payload, TABLE_ID, PLAYERS, infos=infos)
    injected = copy.deepcopy(payload)
    pkts = injected["data"]["data"]
    place_idx = [
        i for i, p in enumerate(pkts) if p["data"][0]["type"] == "tilesPlacedOnLine"
    ]
    cut = place_idx[3]  # resign after four completed turns, well inside round 0
    conceder = game.seat_to_player(1)  # engine seat 1 gives up
    injected["data"]["data"] = pkts[: cut + 1] + [
        _table_pkt({"type": "playerConcedeGame", "args": {"player_id": str(conceder)}}),
        # anything after the concession is ignored — the parser stops there
        _table_pkt({"type": "tilesSelected", "args": {"fromFactory": 1, "type": 1}}),
    ]
    replay = parse_log(injected, TABLE_ID, PLAYERS, infos=infos)
    assert replay.conceded_by == conceder
    assert [p.action_id() for p in replay.picks] == [
        p.action_id() for p in baseline.picks[:4]
    ]
    converted = convert_game(replay)  # must not raise on the non-terminal state
    assert len(converted) == 4
    assert converted.outcome == 1.0  # seat 1 resigned, so seat 0 wins


def test_a_concession_missing_from_the_log_is_read_from_tableinfos() -> None:
    """A rare real shape (2 of 28 concessions in the first 400-game corpus): the
    log just stops mid-game with no `playerConcedeGame` at all, and the concession
    lives only in `tableinfos` — `endgame_reason` is `normal_concede_end` and the
    conceder holds the nominal 0 in the 1-0 result (verified against all 26
    notification-carrying concessions in the same corpus).
    """
    game, payload, infos = _fixture(seed=7)
    truncated = copy.deepcopy(payload)
    pkts = truncated["data"]["data"]
    place_idx = [
        i for i, p in enumerate(pkts) if p["data"][0]["type"] == "tilesPlacedOnLine"
    ]
    truncated["data"]["data"] = [
        pkt
        for pkt in pkts[: place_idx[3] + 1]
        if pkt["data"][0]["type"] != "score"  # a real archive log has no score notif
    ]
    conceder = game.seat_to_player(1)
    conceded_infos = copy.deepcopy(infos)
    conceded_infos["data"]["result"]["endgame_reason"] = "normal_concede_end"
    for row in conceded_infos["data"]["result"]["player"]:
        row["score"] = "0" if int(row["player_id"]) == conceder else "1"
    assert conceder_from_infos(conceded_infos) == conceder
    replay = parse_log(truncated, TABLE_ID, PLAYERS, infos=conceded_infos)
    assert replay.conceded_by == conceder
    converted = convert_game(replay)  # non-terminal state + nominal score: no raise
    assert len(converted) == 4
    assert converted.outcome == 1.0  # seat 1 conceded, so seat 0 wins
    # a natural end keeps conceded_by empty: same infos shape, normal reason
    assert conceder_from_infos(infos) is None


def test_a_time_joker_notification_is_ignored() -> None:
    """`timeJokerUsed` is turn-based clock bookkeeping ("+N days thinking time"),
    seen in real logs 2026-08-18; it carries no move information."""
    _game, payload, infos = _fixture(seed=4)
    baseline = convert_game(parse_log(payload, TABLE_ID, PLAYERS, infos=infos))
    injected = copy.deepcopy(payload)
    injected["data"]["data"].insert(
        2,
        _table_pkt(
            {
                "type": "timeJokerUsed",
                "args": {"player_name": "lucky sven", "nb_days": 4},
            }
        ),
    )
    after = convert_game(parse_log(injected, TABLE_ID, PLAYERS, infos=infos))
    assert list(after.actions) == list(baseline.actions)
    assert after.scores == baseline.scores


def test_a_cosmetic_log_detail_notification_is_ignored() -> None:
    """`completeColorLogDetails` and its family are human-readable text, not moves."""
    _game, payload, infos = _fixture(seed=3)
    baseline = convert_game(parse_log(payload, TABLE_ID, PLAYERS, infos=infos))
    injected = copy.deepcopy(payload)
    injected["data"]["data"].insert(
        3,
        _table_pkt(
            {
                "type": "completeColorLogDetails",
                "args": {"player_id": str(PLAYERS[0]), "color": "red"},
            }
        ),
    )
    after = convert_game(parse_log(injected, TABLE_ID, PLAYERS, infos=infos))
    assert list(after.actions) == list(baseline.actions)
    assert after.scores == baseline.scores


def test_final_scores_fall_back_to_tableinfos() -> None:
    """A real archive log has no score notification; scores come from `tableinfos`."""
    game, payload, infos = _fixture(seed=1)
    # confirm the fixture's infos carries the authoritative scores
    reported = scores_from_infos(infos)
    assert reported == {game.seat_to_player(0): game.scores[0],
                        game.seat_to_player(1): game.scores[1]}
    # strip the log's `score` notifications, as a real log has none
    stripped = copy.deepcopy(payload)
    stripped["data"]["data"] = [
        pkt for pkt in stripped["data"]["data"] if pkt["data"][0]["type"] != "score"
    ]
    replay = parse_log(stripped, TABLE_ID, PLAYERS, infos=infos)
    assert replay.scores_by_seat() == game.scores
    assert convert_game(replay).scores == game.scores


def test_an_unknown_notification_type_stops_the_parse() -> None:
    _game, payload, infos = _fixture(seed=9)
    broken = copy.deepcopy(payload)
    # rename a score notification: renaming a move would trip the turn-pairing
    # check first, and this test is about the unknown-type guard
    for packet in reversed(broken["data"]["data"]):
        if packet["data"][0]["type"] == "score":
            packet["data"][0]["type"] = "somethingNewBgaAdded"
            break
    with pytest.raises(ParseError, match="unknown notification types"):
        parse_log(broken, TABLE_ID, PLAYERS, infos=infos)


def test_a_permuted_colour_map_is_detected_and_solvable() -> None:
    """A wrong colour map changes wall adjacency, so the score check catches it.

    This is the property that lets the next session pin the real BGA tile ids
    without a human eyeballing a replay: only the true mapping reproduces the
    score BGA reported.
    """
    _game, payload, infos = _fixture(seed=2)
    raw_ids = observed_color_ids(payload, DEFAULT_SCHEMA)
    assert raw_ids == [1, 2, 3, 4, 5], "BGA numbers Azul's five tile types 1..5"

    truth = dict(AZUL_COLOR_MAP)
    survivors = solve_color_map(payload, TABLE_ID, PLAYERS, DEFAULT_SCHEMA, infos)
    assert truth in survivors
    # the search is a real filter, not a rubber stamp
    assert len(survivors) < 120


def test_intersecting_several_games_narrows_the_colour_map() -> None:
    """One game leaves a handful of candidates; several leave far fewer."""
    tables = []
    for seed in (0, 2, 5, 11, 13):
        _g, payload, infos = _fixture(seed=seed)
        tables.append((payload, TABLE_ID + seed, PLAYERS, infos))
    single = solve_color_map(
        tables[0][0], tables[0][1], PLAYERS, DEFAULT_SCHEMA, tables[0][3]
    )
    intersected = solve_color_map_over(tables, DEFAULT_SCHEMA)
    assert dict(AZUL_COLOR_MAP) in intersected
    assert len(intersected) <= len(single)
    # with the wall-column check in play the answer is already unique
    assert intersected == [dict(AZUL_COLOR_MAP)]


def test_a_shifted_colour_map_fails_conversion() -> None:
    _game, payload, infos = _fixture(seed=2)
    shifted = {raw: (engine + 1) % 5 for raw, engine in AZUL_COLOR_MAP.items()}
    schema = with_color_map(DEFAULT_SCHEMA, shifted)
    with pytest.raises((ConversionError, ParseError)):
        convert_game(parse_log(payload, TABLE_ID, PLAYERS, schema, infos))


def test_the_parser_refuses_to_guess_an_incomplete_colour_map() -> None:
    payload = {
        "status": 1,
        "data": {
            "logs": [
                {
                    "move_id": "1",
                    "data": [
                        {
                            "type": "factoriesFilled",
                            # 1-based: center slot first, then the factories.
                            "args": {"factories": [[0], [1, 1], [2], [], [], []]},
                        }
                    ],
                },
            ]
        },
    }
    # only reachable with an unpinned map: the Azul default is already known
    unpinned = replace(DEFAULT_SCHEMA, color_map=None)
    with pytest.raises(ParseError, match="need 5"):
        parse_log(payload, TABLE_ID, PLAYERS, unpinned)


# ---------------------------------------------------------------------- dataset
def test_the_dataset_is_written_in_the_pretrain_format(tmp_path: Path) -> None:
    """``build_dataset`` output must load straight into a training ReplayBuffer."""
    games = []
    for seed in range(4):
        _g, payload, infos = _fixture(seed=seed)
        games.append(
            convert_game(parse_log(payload, TABLE_ID + seed, PLAYERS, infos=infos))
        )
    path = tmp_path / "replay.npz"
    stats = build_dataset(games, path)
    assert stats.games == 4
    assert stats.positions == sum(len(g) for g in games)

    buffer = ReplayBuffer(capacity=stats.positions)
    assert buffer.load(path) == stats.positions
    assert buffer.states.shape[1] == ENCODED_SIZE
    assert buffer.policies.shape[1] == ACTION_SPACE

    # policy targets are one-hot on the human move
    rows = buffer.policies[: stats.positions]
    assert np.allclose(rows.sum(axis=1), 1.0)
    assert set(np.unique(rows).tolist()) <= {0.0, 1.0}
    # every optional head has a real target: margins, aux and policy all unmasked
    reported = buffer.stats()
    assert reported["margin_targets"] == stats.positions
    assert reported["aux_targets"] == stats.positions
    assert reported["policy_targets"] == stats.positions
    # values are the outcome in the mover's frame, so |v| is 1 unless it is a draw
    values = buffer.values[: stats.positions]
    assert set(np.unique(values).tolist()) <= {-1.0, 0.0, 1.0}


def test_value_and_margin_are_in_the_mover_frame() -> None:
    game, payload, infos = _fixture(seed=1)
    converted = convert_game(parse_log(payload, TABLE_ID, PLAYERS, infos=infos))
    values = converted.values()
    margins = converted.margins()
    for i, seat in enumerate(converted.movers.tolist()):
        sign = 1.0 if seat == 0 else -1.0
        assert values[i] == pytest.approx(game.outcome * sign)
        assert margins[i] == pytest.approx(
            margins[0] * (1.0 if seat == converted.movers[0] else -1.0)
        )


def test_one_hot_policies_rejects_out_of_range_actions() -> None:
    with pytest.raises(ValueError, match="outside"):
        one_hot_policies(np.array([ACTION_SPACE]))


# ------------------------------------------------------------------------ client
class _FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200):
        super().__init__(body)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeOpener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.urls: list[str] = []

    def open(self, request, timeout=None):
        self.urls.append(request.full_url)
        return self.responses.pop(0)


def _client_with(*responses) -> BgaClient:
    client = BgaClient(ClientConfig(min_interval=0.0, jitter=0.0))
    client._opener = _FakeOpener(*responses)
    return client


def test_the_client_unwraps_the_bga_envelope() -> None:
    client = _client_with(
        _FakeResponse(
            json.dumps(
                {
                    "status": 1,
                    "data": {
                        "ranks": [
                            {
                                "id": "1",
                                "name": "a",
                                "ranking": "2486.16",
                                "nbr_game": "10",
                                "rank_no": "1",
                            }
                        ]
                    },
                }
            ).encode()
        )
    )
    rows = client.ranking_page(0)
    assert rows[0]["name"] == "a"
    assert "mode=elo" in client._opener.urls[0]


def test_a_806_answer_becomes_AuthRequired() -> None:
    """This is exactly what BGA returns for a session-less private endpoint."""
    body = json.dumps(
        {
            "status": "0",
            "error": "Invalid session information for this action.",
            "code": 806,
        }
    ).encode()
    client = _client_with(_FakeResponse(body))
    with pytest.raises(AuthRequired, match="806"):
        client.get_json("/archive/archive/logs.html?table=1")


def test_an_html_answer_becomes_AuthRequired() -> None:
    client = _client_with(_FakeResponse(b"<!DOCTYPE html><html>login wall</html>"))
    with pytest.raises(AuthRequired, match="HTML"):
        client.get_json("/gamestats?player=1")


def test_a_non_auth_error_stays_a_BgaError() -> None:
    body = json.dumps({"status": "0", "error": "Table not found", "code": 100}).encode()
    client = _client_with(_FakeResponse(body))
    with pytest.raises(BgaError) as exc:
        client.get_json("/archive/archive/logs.html?table=1")
    assert not isinstance(exc.value, AuthRequired)


def test_cookies_are_loaded_from_a_netscape_file(tmp_path: Path) -> None:
    jar = tmp_path / "cookies.txt"
    jar.write_text(
        "# Netscape HTTP Cookie File\n"
        ".boardgamearena.com\tTRUE\t/\tTRUE\t0\tPHPSESSID\tdeadbeef\n"
        ".boardgamearena.com\tTRUE\t/\tTRUE\t0\tTournoiEnLigneid\ttoken\n"
    )
    client = BgaClient(ClientConfig(cookies_path=jar))
    assert client.authenticated
    assert client.cookie_names() == ["PHPSESSID", "TournoiEnLigneid"]


def test_a_client_without_cookies_knows_it_is_anonymous() -> None:
    assert not BgaClient(ClientConfig()).authenticated


def test_the_elo_scale_conversion_is_the_one_the_site_uses() -> None:
    # the site shows max(0, raw - 1300), floored: BGA's #1 Azul player on
    # 2026-08-17 had raw 2486.16 and displayed 1186
    assert display_elo("2486.16") == 1186
    assert display_elo(1200) == 0
    assert raw_elo(700) == 2000.0


def test_the_daily_replay_quota_is_its_own_exception() -> None:
    """BGA sends the quota as a 200 with an error field, not as an HTTP 429.

    Getting this wrong would mean a run that silently records hundreds of
    "error" verdicts and burns the day's budget on refusals.
    """
    body = json.dumps(
        {"status": 1, "error": "You have reached a limit (replay)"}
    ).encode()
    client = _client_with(_FakeResponse(body))
    with pytest.raises(ReplayLimitReached):
        client.get_json("/archive/archive/logs.html?table=1")


def test_a_disabled_account_and_a_lost_archive_are_told_apart() -> None:
    disabled = json.dumps(
        {"status": 1, "error": "This feature is disabled for your account"}
    ).encode()
    with pytest.raises(AccountDisabled):
        _client_with(_FakeResponse(disabled)).get_json("/archive/archive/logs.html")

    lost = json.dumps(
        {"status": 1, "error": "Unfortunately the replay for this game has been lost"}
    ).encode()
    with pytest.raises(ReplayUnavailable):
        _client_with(_FakeResponse(lost)).get_json("/archive/archive/logs.html")

    new = json.dumps(
        {
            "status": "0",
            "error": "Sorry, you need to be registered more than 24 hours and have "
            "played at least 2 games to access this feature.",
        }
    ).encode()
    with pytest.raises(AuthRequired):
        _client_with(_FakeResponse(new)).get_json("/archive/archive/logs.html")


def test_the_request_token_is_scraped_from_a_page() -> None:
    """`getGames` wants it as an `X-Request-Token` header."""
    page = b"var bgaConfig = { requestToken: '02ae0224be5a2ef01b1eaf153165b5fd', };"
    client = _client_with(_FakeResponse(page))
    assert client.fetch_request_token() == "02ae0224be5a2ef01b1eaf153165b5fd"
    assert client.request_token == "02ae0224be5a2ef01b1eaf153165b5fd"


def test_the_request_token_reads_the_real_short_mixed_case_form() -> None:
    """The live token is short and mixed-case (e.g. 'gNZGNxgP5p7MOzj'), not hex.

    Verified live 2026-08-17: an earlier ``[0-9a-f]{16,128}`` regex never matched
    it, so ``request_token`` stayed ``None`` and ``getGames`` answered code 806.
    """
    page = b"bgaConfig = {\n\trequestToken: 'gNZGNxgP5p7MOzj',\n\tutm_source: '',\n};"
    client = _client_with(_FakeResponse(page))
    assert client.fetch_request_token() == "gNZGNxgP5p7MOzj"


def test_the_endpoint_table_documents_the_archive_priming_call() -> None:
    eps = endpoints()
    assert "requestTableArchive" in eps["archive_prime"]
    # robots.txt disallows /table, so a robots-clean alternative must exist
    assert eps["table_infos_alt"].startswith("/tablemanager/")
    assert "page={page}" in eps["player_tables"], "getGames pages by `page`, not offset"


def test_the_endpoint_table_documents_the_ranking_mode() -> None:
    eps = endpoints()
    assert "mode=elo" in eps["ranking"], "mode=arena is the season ladder, not all-time"
    assert "game={game}" in eps["ranking"]
    assert "logs.html" in eps["table_logs"]


# ------------------------------------------------------------------------ fetcher
class _StubClient:
    """Stands in for BgaClient: serves canned pages, counts requests."""

    def __init__(self, pages: list[list[dict]]):
        self.pages = pages
        self.requests_made = 0
        self.config = ClientConfig(min_interval=0.0, max_requests_per_day=0)

    def ranking_page(self, start: int, game: int = 1467) -> list[dict]:
        self.requests_made += 1
        index = start // 10
        return self.pages[index] if index < len(self.pages) else []


def _row(i: int) -> dict:
    return {
        "id": str(90_000_000 + i),
        "name": f"p{i}",
        "ranking": f"{2100 - i}",
        "nbr_game": "1500",
        "rank_no": str(i + 1),
    }


def test_the_ranking_fetch_is_cached_in_the_state_file(tmp_path: Path) -> None:
    client = _StubClient(
        [[_row(i) for i in range(10)], [_row(10 + i) for i in range(10)]]
    )
    fetcher = Fetcher(client=client, out_dir=tmp_path)  # type: ignore[arg-type]
    rows = fetcher.fetch_ranking(pages=3)
    assert len(rows) == 20  # third page comes back empty and stops the loop
    assert client.requests_made == 3

    # a fresh Fetcher on the same directory resumes from disk, no requests
    again = Fetcher(client=_StubClient([]), out_dir=tmp_path)  # type: ignore[arg-type]
    assert len(again.fetch_ranking(pages=3)) == 20
    assert again.client.requests_made == 0
    assert (tmp_path / "state.json").exists()


def test_the_state_file_remembers_table_verdicts(tmp_path: Path) -> None:
    state = FetchState.load(tmp_path / "state.json")
    state.record(TableVerdict(123, "skipped", "3 players"))
    state.record(TableVerdict(124, "downloaded"))
    state.save()

    reloaded = FetchState.load(tmp_path / "state.json")
    assert reloaded.verdict(123).reason == "3 players"
    assert reloaded.verdict(123).terminal
    assert reloaded.verdict(124).terminal
    assert reloaded.verdict(999) is None


def test_a_state_file_from_another_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 99}))
    with pytest.raises(ValueError, match="version"):
        FetchState.load(path)


def test_player_selection_uses_displayed_elo(tmp_path: Path) -> None:
    rows = [
        PlayerRow(1, "a", 2200.0, 900, 1, 3000),
        PlayerRow(2, "b", 2000.0, 700, 2, 50),
        PlayerRow(3, "c", 1900.0, 600, 3, 2000),
    ]
    assert [
        r.player_id for r in select_players(rows, top_n=None, min_elo_display=700)
    ] == [1, 2]
    assert [r.player_id for r in select_players(rows, top_n=None, min_games=100)] == [
        1,
        3,
    ]
    assert [r.player_id for r in select_players(rows, top_n=1)] == [1]


def test_player_rows_carry_both_elo_scales() -> None:
    row = PlayerRow.from_api(_row(0))
    assert row.elo_raw == 2100.0
    assert row.elo_display == 800


# ------------------------------------------------------------------------ filter
def _infos(**overrides) -> dict:
    data = {
        "game_id": "1467",
        "status": "finished",
        "players": {"1": {"player_elo": "2100"}, "2": {"player_elo": "2050"}},
        "options": {},
    }
    data.update(overrides)
    return {"status": 1, "data": data}


def test_the_wall_filter_uses_the_board_option() -> None:
    """Board option (id 100) pinned 2026-08-17: 1 = standard, 2/3/4 = variants."""
    # a table whose options omit the board option is skipped, not accepted blindly
    assert "missing" in TableFilter().check(_infos())
    # value 1 ("Colored side") is the standard fixed-colour wall -> accepted
    assert TableFilter().check(_infos(options={"100": {"value": "1"}})) == ""
    # 2 = "Gray side", 3/4 = "Crystal Mozaic" sides -> all rejected
    for variant in (2, 3, 4):
        assert "non-standard wall" in TableFilter().check(
            _infos(options={"100": {"value": str(variant)}})
        )


def test_the_filter_drops_three_player_and_unfinished_tables() -> None:
    flt = TableFilter(require_standard_wall=False)
    assert flt.check(_infos()) == ""
    three = _infos(players={"1": {}, "2": {}, "3": {}})
    assert "3 players" in flt.check(three)
    assert "status" in flt.check(_infos(status="play"))
    assert "wrong game" in flt.check(_infos(game_id="2220"))  # Azul Duel


def test_the_filter_can_require_a_per_seat_elo_floor() -> None:
    flt = TableFilter(require_standard_wall=False, min_player_elo_raw=2060.0)
    assert "below floor" in flt.check(_infos())
    assert (
        flt.check(
            _infos(players={"1": {"player_elo": "2100"}, "2": {"player_elo": "2200"}})
        )
        == ""
    )


def test_option_values_are_read_from_both_payload_shapes() -> None:
    """`tableinfos.options` is `{id: {name, value}}`, older payloads `{id: value}`."""
    assert option_value({"201": {"name": "Game mode", "value": "2"}}, 201) == 2
    assert option_value({"201": 2}, GAME_MODE_OPTION) == 2
    assert option_value({201: "0"}, 201) == 0
    assert option_value({}, 201) is None
    assert option_value({"201": {"name": "x"}}, 201) is None


def test_the_filter_can_require_arena_games() -> None:
    """Option 201 == 2 is BGA's ranked Arena mode."""
    flt = TableFilter(require_standard_wall=False, allowed_game_modes=(ARENA_MODE,))
    assert flt.check(_infos(options={"201": {"value": "2"}})) == ""
    assert "game mode 1" in flt.check(_infos(options={"201": {"value": "1"}}))
    assert "missing" in flt.check(_infos(options={}))


def test_the_filter_drops_unranked_tables() -> None:
    flt = TableFilter(require_standard_wall=False)
    assert "unranked" in flt.check(_infos(unranked="1"))


def test_the_wall_filter_reads_the_bare_value_shape() -> None:
    """Older ``tableinfos`` payloads use ``{id: value}`` instead of ``{id: {value}}``."""
    flt = TableFilter()
    assert flt.check(_infos(options={"100": 1})) == ""
    assert "non-standard wall" in flt.check(_infos(options={"100": 2}))
    assert "missing" in flt.check(_infos(options={"101": 1}))


# -------------------------------------------------------------------- utilities
def test_table_ids_are_extracted_from_any_plausible_shape() -> None:
    assert extract_table_ids({"data": {"tables": [{"table_id": "5"}, {"id": 6}]}}) == [
        5,
        6,
    ]
    assert extract_table_ids({"data": {"games": {"a": {"table_id": 7}}}}) == [7]
    assert extract_table_ids({"data": {}}) == []


def test_the_confirmed_envelope_shapes_are_all_accepted() -> None:
    """A real log puts the packets at ``data.data``, not ``data.logs``."""
    packet = {"channel": "/table/t1", "move_id": "1", "data": []}
    assert log_packets({"status": 1, "data": {"valid": 1, "data": [packet]}}) == [
        packet
    ]
    assert log_packets({"status": 1, "data": {"logs": [packet]}}) == [packet]
    assert log_packets([packet]) == [packet]
    with pytest.raises(ParseError, match="packet list"):
        log_packets({"status": 1, "data": {"valid": 1}})


def test_private_player_channel_packets_are_dropped() -> None:
    """`/player/pNNN` packets are one player's private hints, not game events.

    The fixture deliberately plants a bogus ``tilesSelected`` on a player channel;
    if the filter ever regresses it becomes an extra illegal pick.
    """
    game, payload, infos = _fixture(seed=1)
    channels = {p["channel"].split("/")[1] for p in log_packets(payload)}
    assert channels == {"table", "player"}, "the fixture must contain both channels"

    kept = list(iter_log_entries(payload))
    everything = list(iter_log_entries(payload, table_channel_only=False))
    assert len(everything) == len(kept) + 1

    replay = parse_log(payload, TABLE_ID, PLAYERS, infos=infos)
    assert len(replay.picks) == len(game.actions)
    assert convert_game(replay).scores == game.scores


def test_a_saved_replay_page_can_be_parsed_without_any_request() -> None:
    """`g_gamelogs` in a browser-saved replay page is a complete, ToS-safe log."""
    game, payload, infos = _fixture(seed=3)
    page = (
        "<html><head><script>\n"
        "var g_gamelogs = " + json.dumps(payload) + "\n;\n"
        "</script></head><body>replay</body></html>"
    )
    recovered = parse_gamelogs_html(page)
    converted = convert_game(parse_log(recovered, TABLE_ID, PLAYERS, infos=infos))
    assert converted.scores == game.scores
    with pytest.raises(ParseError, match="no g_gamelogs"):
        parse_gamelogs_html("<html>nothing here</html>")


def test_history_rows_are_read_from_the_dojo_wrapped_shape() -> None:
    """Community code reads `results[0].data.tables`; real HTTP gives `data.tables`."""
    row = {
        "table_id": "712345678",
        "players": "91843016,91718783",
        "scores": "78,64",
        "start": 1690000000,
    }
    for payload in (
        {"status": 1, "data": {"tables": [row]}},
        {"results": [{"data": {"tables": [row]}}]},
    ):
        assert extract_table_ids(payload) == [712345678]
        rows = extract_table_rows(payload)
        assert table_row_players(rows[0]) == [91843016, 91718783]
        assert table_row_scores(rows[0]) == [78, 64]


def test_history_rows_give_the_player_count_for_free() -> None:
    """Two players in the row = the 2-player filter without a `tableinfos` call."""
    three = {"table_id": "1", "players": "1,2,3", "scores": "10,20,30"}
    assert len(table_row_players(three)) == 3
    assert table_row_players({"table_id": "1"}) == []
    assert table_row_scores({"scores": ""}) == []


def test_raw_payloads_round_trip_through_gzip(tmp_path: Path) -> None:
    path = write_json_gz(tmp_path / "raw" / "1.json.gz", {"a": [1, 2, 3]})
    assert read_json_gz(path) == {"a": [1, 2, 3]}


def test_the_log_histogram_reports_types_and_arg_keys() -> None:
    _game, payload, _infos = _fixture(seed=0)
    histogram = log_type_histogram(payload)
    assert set(histogram) == {
        "factoriesFilled",
        "tilesSelected",
        "tilesPlacedOnLine",
        "placeTileOnWall",
        "score",
    }
    assert histogram["tilesSelected"]["arg_keys"] == [
        "discardedTiles",
        "fromFactory",
        "player_id",
        "selectedTiles",
        "type",
    ]
    assert histogram["factoriesFilled"]["count"] == len(_game.deals)


def test_the_fetcher_reads_back_what_it_stored(tmp_path: Path) -> None:
    fetcher = Fetcher(client=_StubClient([]), out_dir=tmp_path)  # type: ignore[arg-type]
    _game, payload, infos = _fixture(seed=0)
    write_json_gz(
        fetcher.raw_dir / f"{TABLE_ID}.json.gz",
        {"table_id": TABLE_ID, "infos": infos, "logs": payload},
    )
    stored = dict(fetcher.iter_raw())
    assert list(stored) == [TABLE_ID]
    replay = parse_log(stored[TABLE_ID]["logs"], TABLE_ID, PLAYERS, infos=infos)
    assert convert_game(replay).scores == _game.scores


# ==========================================================================
# Rank-ordered crawl
# ==========================================================================
class _CrawlClient:
    """Serves canned ``getGames`` / ``tableinfos`` / ``logs`` answers by URL.

    Enough of :class:`BgaClient` for the crawl: it counts requests, remembers the
    URLs it was asked for (which is how the tests prove nothing was re-requested)
    and can be told to answer BGA's replay-quota error after N replays.
    """

    def __init__(
        self,
        tables_by_player: dict[int, list[int]],
        elos: dict[int, float] | None = None,
        replay_limit_after: int | None = None,
        elo_after: dict[int, float] | None = None,
    ):
        self.config = ClientConfig(min_interval=0.0, jitter=0.0, max_requests_per_day=0)
        self.requests_made = 0
        self.urls: list[str] = []
        self.tables_by_player = tables_by_player
        self.elos = elos or {}
        self.replay_limit_after = replay_limit_after
        self.elo_after = elo_after or {}  # per-table override for the canned rows
        self.logs_served = 0
        self.request_token = None

    # the crawl calls this once before the first getGames
    def fetch_request_token(self, path: str = "") -> str | None:  # pragma: no cover
        return None

    def _owner(self, table_id: int) -> int:
        for player, tables in self.tables_by_player.items():
            if table_id in tables:
                return player
        raise AssertionError(f"unknown table {table_id}")

    def get_json(self, path: str, referer: str | None = None) -> dict:
        self.requests_made += 1
        self.urls.append(path)
        if "getGames" in path:
            player = int(re.search(r"player=(\d+)", path).group(1))
            page = int(re.search(r"page=(\d+)", path).group(1))
            chunk = self.tables_by_player.get(player, [])[(page - 1) * 10 : page * 10]
            return {
                "status": 1,
                "data": {
                    "tables": [
                        {
                            "table_id": str(t),
                            "players": f"{player},{_WEAK}",
                            # as the live row spells it (2026-08-19): the QUERIED
                            # player's raw Elo right after this game, a string
                            "elo_after": str(self.elo_after.get(t, 2400 + t % 10)),
                        }
                        for t in chunk
                    ]
                },
            }
        if "tableinfos" in path:
            table_id = int(re.search(r"id=(\d+)", path).group(1))
            owner = self._owner(table_id)
            return {
                "status": 1,
                "data": {
                    "game_id": "1467",
                    "status": "finished",
                    "players": {
                        str(owner): {
                            "id": str(owner),
                            "player_elo": str(self.elos.get(owner, 2100.0)),
                        },
                        str(_WEAK): {"id": str(_WEAK), "player_elo": "1450"},
                    },
                    "options": {},
                },
            }
        if "requestTableArchive" in path:
            return {"status": 1, "data": {}}
        if "logs.html" in path:
            self.logs_served += 1
            if (
                self.replay_limit_after is not None
                and self.logs_served > self.replay_limit_after
            ):
                raise ReplayLimitReached(f"{path}: You have reached a limit (replay)")
            return {"status": 1, "data": {"valid": 1, "data": []}}
        raise AssertionError(f"unexpected request {path}")


#: the weak opponent every canned table pairs the ranked player with
_WEAK = 55_555_555


def _ladder(count: int = 3) -> list[PlayerRow]:
    """``count`` players, rank 1 first, strongest first (as BGA orders them)."""
    return [
        PlayerRow(
            player_id=90_000_000 + i,
            name=f"elite{i}",
            elo_raw=2400.0 - 10 * i,
            elo_display=1100 - 10 * i,
            rank=i + 1,
            games_played=1500,
        )
        for i in range(count)
    ]


def _tables_for(rows: list[PlayerRow], per_player: int = 3) -> dict[int, list[int]]:
    return {
        row.player_id: [700_000_000 + row.rank * 100 + n for n in range(per_player)]
        for row in rows
    }


def _expected_order(rows: list[PlayerRow], tables: dict[int, list[int]]) -> list[int]:
    """Rank order between players, newest table first within a player.

    ``fetch_player_tables`` keeps a player's ids sorted descending on purpose: the
    highest table id is the most recent game, recent play reflects the current meta,
    and old archives are the ones BGA is likeliest to have dropped (docs §5.4).
    """
    return [t for row in rows for t in sorted(tables[row.player_id], reverse=True)]


def _crawl_fetcher(tmp_path: Path, client: _CrawlClient, **pace) -> Fetcher:
    settings = {"long_pause_every": 0, "max_tables_per_day": 0}
    settings.update(pace)
    return Fetcher(
        client=client,  # type: ignore[arg-type]
        out_dir=tmp_path,
        table_filter=TableFilter(require_standard_wall=False),
        pace=CrawlPace(**settings),
        sleeper=lambda seconds: None,
    )


def _downloaded_tables(fetcher: Fetcher) -> list[int]:
    """Table ids in the order they were downloaded, across every run in this dir.

    ``fetch.jsonl`` is append-only, so this is the whole history — which is what
    makes "resume did not re-fetch and did not skip anything" a single comparison
    against :func:`_expected_order`.
    """
    out = []
    for line in fetcher.log_path.read_text().splitlines():
        event = json.loads(line)
        if event.get("status") == "downloaded":
            out.append(int(event["table"]))
    return out


def test_the_crawl_walks_the_ladder_in_rank_order(tmp_path: Path) -> None:
    """Rank 1's games first, then rank 2 — not "whatever the history returns"."""
    rows = _ladder(3)
    tables = _tables_for(rows)
    fetcher = _crawl_fetcher(tmp_path, _CrawlClient(tables))
    report = fetcher.crawl_ranked(rows, per_player=3, history_pages=1)

    assert report.downloaded == 9
    assert report.stopped == ""
    expected = _expected_order(rows, tables)
    assert _downloaded_tables(fetcher) == expected
    # ...and the state file knows exactly where it ended up
    cursor = fetcher.state.get_cursor()
    assert (cursor.rank, cursor.table_offset, cursor.players_done) == (3, 3, 3)
    assert fetcher.state.downloads["total"] == 9
    # rank order survives a shuffled input list: the crawl sorts by rank itself
    other = _crawl_fetcher(tmp_path / "b", _CrawlClient(tables))
    other.crawl_ranked(list(reversed(rows)), per_player=3, history_pages=1)
    assert _downloaded_tables(other) == expected


def test_the_crawl_resumes_at_the_exact_table_it_stopped_on(tmp_path: Path) -> None:
    """The cursor records rank *and* offset inside that rank's table list."""
    rows = _ladder(3)
    tables = _tables_for(rows)
    first = _crawl_fetcher(tmp_path, _CrawlClient(tables))
    first.crawl_ranked(rows, per_player=3, history_pages=1, max_tables=4)
    cursor = first.state.get_cursor()
    assert (cursor.rank, cursor.player_id, cursor.table_offset) == (
        2,
        rows[1].player_id,
        1,
    )

    # a brand-new process on the same directory
    client = _CrawlClient(tables)
    second = _crawl_fetcher(tmp_path, client)
    report = second.crawl_ranked(rows, per_player=3, history_pages=1)

    assert report.downloaded == 5  # 9 total minus the 4 already done
    assert report.players == 2, "rank 1 is behind the cursor and is not revisited"
    # the append-only log across both runs is exactly the ladder order, once each
    assert _downloaded_tables(second) == _expected_order(rows, tables)
    # nothing about rank 1 was requested again — not its history, not its tables
    assert not any(str(rows[0].player_id) in url for url in client.urls)
    for table_id in tables[rows[0].player_id]:
        assert not any(str(table_id) in url for url in client.urls)


def test_history_rows_are_persisted_with_their_elo_after(tmp_path: Path) -> None:
    """Listing a player's history must keep the raw `getGames` rows: `elo_after`
    is the only per-game, time-accurate Elo BGA exposes (tableinfos has none),
    and it is the basis for Elo-weighted training."""
    rows = _ladder(2)
    tables = _tables_for(rows)
    fetcher = _crawl_fetcher(tmp_path, _CrawlClient(tables))
    fetcher.crawl_ranked(rows, per_player=3, history_pages=1)

    assert fetcher.rows_path.exists()
    elos = elo_after_map(fetcher.rows_path)
    for player, player_tables in tables.items():
        for t in player_tables:
            assert elos[t][player] == 2400 + t % 10
    # the map is keyed by the QUERIED player: the weak opponent contributed none
    assert all(_WEAK not in per_player for per_player in elos.values())
    # a corrupt line and a missing file degrade to "no data", never raise
    fetcher.rows_path.write_text('not json\n{"row": {}}\n', encoding="utf-8")
    assert elo_after_map(fetcher.rows_path) == {}
    assert elo_after_map(tmp_path / "absent.jsonl") == {}


def test_a_beginner_era_game_is_skipped_before_spending_quota(tmp_path: Path) -> None:
    """A top player's early games (low `elo_after`) must not cost replay slots:
    the crawl skips them from the history row alone — zero table requests."""
    rows = _ladder(1)
    tables = _tables_for(rows)
    weak_table = tables[rows[0].player_id][1]  # the middle of three tables
    client = _CrawlClient(tables, elo_after={weak_table: 1500.0})
    fetcher = _crawl_fetcher(tmp_path, client)
    report = fetcher.crawl_ranked(
        rows, per_player=3, history_pages=1, min_source_elo_raw=1950.0
    )
    assert report.downloaded == 2
    assert report.skipped == 1
    verdict = fetcher.state.verdict(weak_table)
    assert verdict.status == "skipped"
    assert "at game time below floor" in verdict.reason
    # the skip cost nothing: the weak table was never requested
    assert not any(str(weak_table) in url for url in client.urls)
    # and without a floor the same table would have been fetched
    other = _crawl_fetcher(tmp_path / "b", _CrawlClient(tables))
    assert other.crawl_ranked(rows, per_player=3, history_pages=1).downloaded == 3


def test_the_dataset_floor_prefers_the_elo_at_game_time(tmp_path: Path) -> None:
    """A game played before the target became elite is dropped by the floor even
    though their snapshot Elo clears it — and the fallback still works."""
    game, payload, infos = _fixture(seed=6)
    target = game.seat_to_player(0)
    meta = GameMeta(
        table_id=TABLE_ID,
        target_player_id=target,
        source_player_id=target,
        elos={target: 2486.0},  # elite TODAY...
    )
    converted = convert_game(parse_log(payload, TABLE_ID, PLAYERS, infos=infos))
    dropped = build_dataset(
        [(converted, meta)],
        tmp_path / "floor.npz",
        min_target_elo_raw=1950.0,
        elo_at_game={TABLE_ID: {target: 1500.0}},  # ...but weak back then
    )
    assert dropped.games == 0
    assert dropped.skipped_below_elo == 1
    # no per-game number -> the snapshot decides, as before
    kept = build_dataset(
        [(converted, meta)], tmp_path / "floor2.npz", min_target_elo_raw=1950.0
    )
    assert kept.games == 1


def test_the_dataset_records_the_targets_elo_at_game_time(tmp_path: Path) -> None:
    """`game_records` carries both the snapshot Elo and the per-game `elo_after`."""
    game, payload, infos = _fixture(seed=2)
    target = game.seat_to_player(0)
    meta = GameMeta(
        table_id=TABLE_ID,
        target_player_id=target,
        source_player_id=target,
        elos={target: 2222.0},
    )
    stats = build_dataset(
        [(convert_game(parse_log(payload, TABLE_ID, PLAYERS, infos=infos)), meta)],
        tmp_path / "elo.npz",
        elo_at_game={TABLE_ID: {target: 2345.0}},
    )
    (record,) = stats.game_records
    assert record["table_id"] == TABLE_ID
    assert record["target_elo_raw"] == 2222.0
    assert record["target_elo_after"] == 2345.0
    assert record["rows"] > 0


def test_a_restart_rewalk_revisits_judged_tables_for_free(tmp_path: Path) -> None:
    """`resume=False` re-walks the ladder from rank 1 to deepen earlier ranks'
    histories: already-judged tables cost no request, no pause, and are reported
    as `cached`, not re-counted as downloads."""
    rows = _ladder(3)
    tables = _tables_for(rows)
    fetcher = _crawl_fetcher(tmp_path, _CrawlClient(tables))
    first = fetcher.crawl_ranked(rows, per_player=3, history_pages=1)
    assert first.downloaded == 9

    client = _CrawlClient(tables)
    again = _crawl_fetcher(tmp_path, client)
    report = again.crawl_ranked(rows, per_player=3, history_pages=1, resume=False)
    assert report.downloaded == 0
    assert report.skipped == 0
    assert report.cached == 9
    assert report.players == 3, "every rank is revisited"
    # no table was requested again — histories are complete, verdicts terminal
    for player_tables in tables.values():
        for table_id in player_tables:
            assert not any(str(table_id) in url for url in client.urls)


def test_the_crawl_stops_cleanly_on_the_replay_limit(tmp_path: Path) -> None:
    """BGA's daily quota: stop, record why, and leave the failing table pending."""
    rows = _ladder(2)
    tables = _tables_for(rows)
    client = _CrawlClient(tables, replay_limit_after=2)
    fetcher = _crawl_fetcher(tmp_path, client)
    report = fetcher.crawl_ranked(rows, per_player=3, history_pages=1)

    assert report.stopped == "replay-limit"
    assert report.downloaded == 2
    assert client.logs_served == 3, "it must not keep asking after the refusal"
    error = fetcher.state.error
    assert error["kind"] == "replay-limit"
    assert "limit (replay)" in error["message"]
    assert error["at"]
    # the cursor points AT the table that was refused, not past it
    cursor = fetcher.state.get_cursor()
    assert (cursor.rank, cursor.table_offset) == (1, 2)
    refused = _expected_order(rows, tables)[2]
    assert fetcher.state.verdict(refused) is None

    # tomorrow's run picks that same table up again
    tomorrow = _crawl_fetcher(tmp_path, _CrawlClient(tables))
    resumed = tomorrow.crawl_ranked(rows, per_player=3, history_pages=1)
    assert resumed.downloaded == 4
    done = _downloaded_tables(tomorrow)
    assert done == _expected_order(rows, tables)
    assert done[2] == refused, "it picks up on the table BGA refused"
    assert tomorrow.state.error == {}, "a fresh run clears the stale stop reason"


def test_our_own_daily_replay_cap_stops_the_crawl(tmp_path: Path) -> None:
    """The conservative self-limit fires before BGA's unknown one does."""
    rows = _ladder(2)
    client = _CrawlClient(_tables_for(rows))
    fetcher = _crawl_fetcher(tmp_path, client, max_tables_per_day=2)
    report = fetcher.crawl_ranked(rows, per_player=3, history_pages=1)
    assert report.stopped == "daily-table-cap"
    assert report.downloaded == 2
    assert fetcher.state.error["kind"] == "daily-table-cap"
    assert fetcher.state.downloads_today() == 2


def test_the_conservative_pace_is_the_default(tmp_path: Path) -> None:
    """A default client is slow on purpose; the quota is the budget, not bandwidth."""
    pace = CrawlPace()
    assert pace.min_interval >= 6.0 and pace.jitter >= 2.0
    assert pace.max_tables_per_day <= 200
    assert pace.max_requests_per_day <= 1000
    assert pace.long_pause_every > 0 and pace.long_pause_max >= 120.0
    assert 20.0 <= pace.seconds_per_table() <= 60.0
    config = ClientConfig()
    assert config.min_interval >= 6.0
    assert config.max_requests_per_day <= 1000


def test_the_crawl_records_both_players_elos_and_the_target(tmp_path: Path) -> None:
    """Per-game Elos are the input to the elite-only filter and its Elo floor."""
    rows = _ladder(1)
    tables = _tables_for(rows, per_player=1)
    fetcher = _crawl_fetcher(
        tmp_path, _CrawlClient(tables, elos={rows[0].player_id: 2380.0})
    )
    fetcher.crawl_ranked(rows, per_player=1, history_pages=1)
    table_id = tables[rows[0].player_id][0]
    entry = fetcher.state.tables[str(table_id)]
    assert entry["status"] == "downloaded"
    assert entry["target"] == rows[0].player_id
    assert entry["rank"] == 1
    assert entry["elos"][str(rows[0].player_id)] == 2380.0
    assert entry["elos"][str(_WEAK)] == 1450.0
    # and the same block is in the raw payload, so a rebuild needs no state file
    raw = dict(fetcher.iter_raw())[table_id]
    assert raw["meta"]["target_player_id"] == rows[0].player_id
    assert raw["meta"]["rank"] == 1


def test_a_version_one_state_file_is_upgraded_not_refused(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 1, "game_id": 1467, "tables": {}}))
    state = FetchState.load(path)
    assert state.get_cursor().rank == 0
    assert state.error == {}
    state.save()
    assert json.loads(path.read_text())["version"] == 2


# ==========================================================================
# Elite-player-only learning
# ==========================================================================
def _elite_meta(target: int, target_elo: float = 2400.0, weak_elo: float = 1500.0):
    other = PLAYERS[1] if target == PLAYERS[0] else PLAYERS[0]
    return GameMeta(
        table_id=TABLE_ID,
        target_player_id=target,
        source_player_id=target,
        rank=1,
        elos={target: target_elo, other: weak_elo},
    )


def _converted(seed: int = 7):
    _game, payload, infos = _fixture(seed=seed)
    return convert_game(parse_log(payload, TABLE_ID, PLAYERS, infos=infos))


def test_seat_elos_are_read_out_of_tableinfos() -> None:
    """Both seats' Elos, whichever key the payload spells them with."""
    payload = {
        "data": {
            "players": {
                "1": {"id": "1", "player_elo": "2100"},
                "2": {"id": "2", "elo": 1450.5},
                "3": {"id": "3"},
            }
        }
    }
    assert player_elos(payload) == {1: 2100.0, 2: 1450.5}
    assert player_elos({}) == {}


def test_the_target_player_is_the_ranked_one_and_the_stronger_of_two() -> None:
    strong, weak = PLAYERS
    elos = {strong: 2400.0, weak: 1500.0}
    # only the source player is on the ladder
    assert (
        choose_target_player(PLAYERS, elos, source_player_id=weak, ranked_ids=[weak])
        == weak
    )
    # both are: the higher Elo wins, whichever list the game came from
    assert (
        choose_target_player(PLAYERS, elos, source_player_id=weak, ranked_ids=PLAYERS)
        == strong
    )
    # no ladder information at all: the source, then the Elo
    assert choose_target_player(PLAYERS, elos, source_player_id=weak) == weak
    assert choose_target_player(PLAYERS, elos) == strong
    assert choose_target_player(PLAYERS, {}) is None


def test_only_the_elite_players_turns_become_training_rows(tmp_path: Path) -> None:
    """The whole point: a game teaches us one player's decisions, not both."""
    game = _converted(seed=7)
    target = PLAYERS[0]
    target_seat = game.seat_of(target)
    assert target_seat == 0
    mine = [i for i, m in enumerate(game.movers.tolist()) if m == target_seat]
    assert 0 < len(mine) < len(game), "the fixture must contain both players' turns"

    path = tmp_path / "replay.npz"
    stats = build_dataset([(game, _elite_meta(target))], path, require_target=True)
    assert stats.positions == len(mine)
    assert stats.elite_positions == len(mine)
    assert stats.all_positions == len(game)

    buffer = ReplayBuffer(capacity=stats.positions)
    assert buffer.load(path) == len(mine)
    stored = buffer.states[: len(mine)]
    assert np.array_equal(stored, game.states[mine])
    # the policy target is that player's own move, one-hot, in the same order
    assert buffer.policies[: len(mine)].argmax(axis=1).tolist() == [
        int(game.actions[i]) for i in mine
    ]
    assert buffer.stats()["policy_targets"] == len(mine)
    # the whole file is that player's frame: one value, one margin, no mixing
    values = buffer.values[: len(mine)]
    assert np.allclose(values, game.outcome)  # target sits in seat 0
    assert np.allclose(np.sign(buffer.margins[: len(mine)]), np.sign(values))


def test_the_outcome_sign_follows_the_target_not_seat_zero(tmp_path: Path) -> None:
    """Pick the loser as the target and every value flips — that is the frame."""
    game = _converted(seed=5)
    assert abs(game.outcome) == 1.0
    rows = {}
    for target in PLAYERS:
        path = tmp_path / f"replay-{target}.npz"
        stats = build_dataset([(game, _elite_meta(target))], path, require_target=True)
        buffer = ReplayBuffer(capacity=max(1, stats.positions))
        buffer.load(path)
        rows[target] = (stats.positions, buffer.values[: stats.positions].copy())
    seat0, seat1 = rows[PLAYERS[0]], rows[PLAYERS[1]]
    assert seat0[0] + seat1[0] == len(game), "between them, every turn is covered once"
    assert np.allclose(seat0[1], game.outcome)
    assert np.allclose(seat1[1], -game.outcome)


def test_opponent_turns_can_be_kept_as_value_only_rows(tmp_path: Path) -> None:
    """The masked alternative: replay.npz's own policy_mask convention."""
    game = _converted(seed=2)
    target = PLAYERS[1]
    target_seat = game.seat_of(target)
    mine = [i for i, m in enumerate(game.movers.tolist()) if m == target_seat]
    path = tmp_path / "replay.npz"
    stats = build_dataset(
        [(game, _elite_meta(target))],
        path,
        require_target=True,
        opponent_rows=OPPONENT_VALUE_ONLY,
    )
    assert stats.positions == len(game)
    assert stats.elite_positions == len(mine)

    buffer = ReplayBuffer(capacity=stats.positions)
    buffer.load(path)
    reported = buffer.stats()
    assert reported["policy_targets"] == len(mine)
    assert reported["margin_targets"] == len(game)
    masked = buffer.policy_mask[: len(game)] == 0.0
    assert masked.sum() == len(game) - len(mine)
    # a masked row carries no policy at all, exactly as run6's cheap searches do
    assert buffer.policies[: len(game)][masked].sum() == 0.0


def test_an_elo_floor_applies_to_the_target_player(tmp_path: Path) -> None:
    """A floor on "somebody at the table" is the weak version of this filter."""
    game = _converted(seed=7)
    strong = PLAYERS[0]
    # the elite seat is 2000 raw: below a 2100 floor, even though the *table*
    # would pass a floor that only asks for one strong player
    below = build_dataset(
        [(game, _elite_meta(strong, target_elo=2000.0, weak_elo=2400.0))],
        tmp_path / "a.npz",
        min_target_elo_raw=2100.0,
        require_target=True,
    )
    assert below.skipped_below_elo == 1
    assert below.positions == 0 and below.games == 0

    above = build_dataset(
        [(game, _elite_meta(strong, target_elo=2200.0))],
        tmp_path / "b.npz",
        min_target_elo_raw=2100.0,
        require_target=True,
    )
    assert above.skipped_below_elo == 0 and above.games == 1
    assert above.positions > 0


def test_a_game_with_no_identifiable_elite_player_is_dropped(tmp_path: Path) -> None:
    game = _converted(seed=7)
    stats = build_dataset(
        [(game, GameMeta(table_id=TABLE_ID))], tmp_path / "a.npz", require_target=True
    )
    assert stats.skipped_no_target == 1 and stats.games == 0


def test_the_dataset_stats_sidecar_is_written_for_the_progress_page(
    tmp_path: Path,
) -> None:
    game = _converted(seed=7)
    path = tmp_path / "replay.npz"
    stats = build_dataset([(game, _elite_meta(PLAYERS[0]))], path, require_target=True)
    sidecar = json.loads((tmp_path / "replay.stats.json").read_text())
    assert sidecar["positions"] == stats.positions
    assert sidecar["elite_positions"] == stats.elite_positions
    assert sidecar["all_positions"] == len(game)
    assert sum(sidecar["target_outcomes"].values()) == 1


def test_the_game_meta_is_recovered_from_a_raw_payload() -> None:
    _game, payload, infos = _fixture(seed=7)
    raw = {"table_id": TABLE_ID, "infos": infos, "logs": payload}
    meta = GameMeta.from_raw(TABLE_ID, raw, ranked_ids=[PLAYERS[1]])
    # the fixture gives both seats the same Elo, so the ladder decides
    assert meta.target_player_id == PLAYERS[1]
    crawled = dict(
        raw,
        meta={
            "target_player_id": PLAYERS[0],
            "rank": 3,
            "source_player_id": PLAYERS[0],
            "elos": {str(PLAYERS[0]): 2400.0},
        },
    )
    meta = GameMeta.from_raw(TABLE_ID, crawled)
    assert (meta.target_player_id, meta.rank) == (PLAYERS[0], 3)
    assert meta.target_elo() == 2400.0


def test_the_target_elo_is_resolved_from_the_ladder_when_tableinfos_lacks_it() -> None:
    """The Elo floor's input: a real `tableinfos` carries no per-seat Elo.

    So `meta.elos` and `player_elos(infos)` are both empty on real data, and the
    only place the target's Elo lives is the ranking snapshot in `state.json`. If it
    is not resolved from there the floor reads every target as null and drops the
    whole dataset — the bug this fixes.
    """
    target, weak = PLAYERS
    # a real-shaped payload: no player_elo on either seat, empty meta.elos
    infos = {
        "data": {
            "players": {str(target): {"id": str(target)}, str(weak): {"id": str(weak)}},
            "options": {},
        }
    }
    assert player_elos(infos) == {}  # nothing to read from tableinfos
    raw = {
        "table_id": TABLE_ID,
        "infos": infos,
        "meta": {
            "rank": 1,
            "source_player_id": target,
            "source_elo_raw": 2486.16,
            "target_player_id": target,
            "elos": {},
        },
    }
    # the crawl's recorded source_elo_raw resolves the target (target == source)
    meta = GameMeta.from_raw(TABLE_ID, raw, ranked_ids=[target])
    assert meta.target_elo() == 2486.16
    # and the ladder map resolves it even with no source_elo_raw recorded
    raw2 = dict(raw, meta=dict(raw["meta"], source_elo_raw=None))
    meta2 = GameMeta.from_raw(
        TABLE_ID, raw2, ranked_ids=[target], ranked_elos={target: 2459.99}
    )
    assert meta2.target_elo() == 2459.99
    # a displayed floor of 650 is a raw floor of 1950 — a rank-1/2 target clears it
    assert meta.target_elo() >= 650 + 1300


def test_a_ladder_resolved_elo_survives_the_displayed_floor_end_to_end(
    tmp_path: Path,
) -> None:
    """The whole Part-2 path: real-shaped meta + `--min-target-elo` in display units."""
    game = _converted(seed=7)
    target = game.player_ids[0]
    infos = {"data": {"players": {str(p): {"id": str(p)} for p in game.player_ids}}}
    raw = {
        "table_id": game.table_id,
        "infos": infos,
        "meta": {
            "rank": 1,
            "source_player_id": target,
            "source_elo_raw": None,
            "target_player_id": target,
            "elos": {},
        },
    }
    meta = GameMeta.from_raw(
        game.table_id, raw, ranked_ids=[target], ranked_elos={target: 2486.16}
    )
    # displayed floor 650 -> raw 1950: the rank-1 target (raw 2486) is kept
    kept = build_dataset(
        [(game, meta)], tmp_path / "keep.npz", min_target_elo_raw=650 + 1300.0
    )
    assert kept.skipped_below_elo == 0 and kept.games == 1 and kept.positions > 0
    # displayed floor 1200 -> raw 2500: even a rank-1 target is now below it
    dropped = build_dataset(
        [(game, meta)], tmp_path / "drop.npz", min_target_elo_raw=1200 + 1300.0
    )
    assert dropped.skipped_below_elo == 1 and dropped.positions == 0


# ==========================================================================
# The harvest progress page (web/make_harvest.py)
# ==========================================================================
_VOID_TAGS = {"meta", "br", "hr", "img", "input", "link", "source", "col"}


class _StrictHtml(HTMLParser):
    """Parses the page and asserts every non-void tag is closed in order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.tags: dict[str, int] = {}
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        self.tags[tag] = self.tags.get(tag, 0) + 1
        if tag not in _VOID_TAGS:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in _VOID_TAGS:
            return
        assert self.stack, f"</{tag}> with nothing open"
        opened = self.stack.pop()
        assert opened == tag, f"</{tag}> closes <{opened}>"

    def handle_data(self, data):
        self.text.append(data)


def _harvest_module():
    path = Path(__file__).resolve().parent.parent / "web" / "make_harvest.py"
    spec = importlib.util.spec_from_file_location("make_harvest_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _render(state_dir: Path, out: Path, npz: Path | None = None) -> tuple[str, object]:
    module = _harvest_module()
    harvest, size = module.generate(state_dir, out, npz=npz)
    page = out.read_text()
    assert size == len(page)
    parser = _StrictHtml()
    parser.feed(page)
    assert not parser.stack, f"unclosed tags: {parser.stack}"
    assert parser.tags.get("html") == 1 and parser.tags.get("body") == 1
    assert "nan" not in page.lower().replace("financ", ""), "no NaN may reach the page"
    for width in re.findall(r"width:([0-9.]+)%", page):
        assert 0.0 <= float(width) <= 100.0, f"bar width {width}% is outside the track"
    return page, harvest


def _bar_widths(page: str) -> list[float]:
    return [float(w) for w in re.findall(r"width:([0-9.]+)%", page)]


def _write_harvest_state(
    tmp_path: Path,
    *,
    error: dict | None = None,
    stats: dict | None = None,
    events: list[dict] | None = None,
) -> Path:
    """A state directory shaped exactly like a mid-crawl one."""
    state = {
        "version": 2,
        "game_id": 1467,
        "started": "2026-08-17T09:00:00+00:00",
        "requests": {"total": 431},
        "downloads": {"total": 96},
        "cursor": {
            "rank": 7,
            "player_id": 91843016,
            "table_offset": 34,
            "players_done": 6,
            "updated": "2026-08-17T11:02:00+00:00",
        },
        "pace": {
            "min_interval": 6.0,
            "jitter": 4.0,
            "max_requests_per_day": 600,
            "max_tables_per_day": 120,
            "long_pause_every": 20,
        },
        "error": error or {},
        "ranking": {
            "rows": [
                {
                    "player_id": 91843016,
                    "name": "Sapperlot",
                    "elo_raw": 2486.16,
                    "elo_display": 1186,
                    "rank": 7,
                    "games_played": 1633,
                }
            ]
        },
        "players": {"91843016": {"pages_done": 4, "tables": [1, 2, 3]}},
        "tables": {
            "700000001": {"status": "downloaded", "reason": "", "rank": 7},
            "700000002": {"status": "downloaded", "reason": "", "rank": 7},
            "700000003": {
                "status": "skipped",
                "reason": "wall variant option id unknown (see docs)",
            },
            "700000004": {"status": "skipped", "reason": "3 players"},
            "700000005": {"status": "error", "reason": "Table not found (code 100)"},
        },
    }
    (tmp_path / "state.json").write_text(json.dumps(state))
    if stats is not None:
        (tmp_path / "replay.stats.json").write_text(json.dumps(stats))
    if events is not None:
        (tmp_path / "fetch.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events)
        )
    return tmp_path


def test_the_harvest_page_renders_before_anything_has_been_fetched(
    tmp_path: Path,
) -> None:
    """The first thing anyone does is open the page *before* starting the crawl."""
    page, harvest = _render(tmp_path / "state", tmp_path / "harvest.html")
    assert not harvest.has_state
    assert "Nothing collected yet" in page
    assert "cli crawl" in page, "the empty state must say how to start"
    assert "—" in page, "unknown numbers are em dashes, not zeros"
    assert set(_bar_widths(page)) == {0.0}
    assert "auto-refresh" in page and 'http-equiv="refresh"' in page
    assert "viewport" in page, "it has to be readable on a phone"
    assert "http://" not in page and "https://" not in page


def test_the_harvest_page_shows_the_rank_and_player_being_crawled(
    tmp_path: Path,
) -> None:
    directory = _write_harvest_state(
        tmp_path,
        stats={
            "games": 88,
            "positions": 2431,
            "elite_positions": 2431,
            "all_positions": 4880,
            "games_rejected": 8,
            "skipped_below_elo": 3,
            "skipped_no_target": 0,
            "elo_floor_raw": 1950.0,
            "opponent_rows": "drop",
            "target_outcomes": {"win": 60, "loss": 26, "draw": 2},
            "reasons": {"engine scored (10, 5), BGA reported (11, 5)": 8},
        },
        events=[
            {
                "ts": 1_800_000_000.0,
                "status": "downloaded",
                "table": 700000001,
                "rank": 7,
                "player": "Sapperlot",
                "target": 91843016,
                "target_elo": 2486.16,
            },
            {
                "ts": 1_800_000_030.0,
                "status": "skipped",
                "table": 700000004,
                "rank": 7,
                "player": "Sapperlot",
                "reason": "3 players",
            },
        ],
    )
    page, harvest = _render(directory, tmp_path / "harvest.html")

    assert harvest.rank == 7 and harvest.downloaded == 2
    assert "#7" in page
    assert "Sapperlot" in page and "1,186" in page  # name + displayed Elo
    assert "34" in page  # table offset inside this player's list
    # yield tallies, bucketed by reason
    assert "gray-wall variant" in page
    assert "not 2 players" in page
    assert "below Elo floor" in page
    assert "replay failed validation" in page
    # dataset progress: elite positions vs all replayed decision points
    assert "2,431" in page or "2.4k" in page
    assert "Elite-only positions" in page
    assert "60W/26L/2D" in page
    # the log tail
    assert "700000001" in page and "downloaded" in page
    # milestones
    assert "2.0k" in page and "10k" in page
    widths = _bar_widths(page)
    assert len(widths) >= 4 and max(widths) <= 100.0


def test_the_harvest_page_shows_the_replay_limit_prominently(tmp_path: Path) -> None:
    """The one state that must never be buried: BGA's daily quota."""
    directory = _write_harvest_state(
        tmp_path,
        error={
            "kind": "replay-limit",
            "message": "/archive/archive/logs.html: You have reached a limit (replay)",
            "at": "2026-08-17T11:04:00+00:00",
        },
    )
    page, _harvest = _render(directory, tmp_path / "harvest.html")
    start = page.index('class="banner banner-crit"')
    banner = page[start : page.index("</section>", start)]
    assert "replay-limit" in banner
    assert "daily replay quota is reached" in banner
    assert "You have reached a limit (replay)" in page
    assert "resumes at the exact table" in page
    # the banner comes before the panels, not after them
    assert page.index("banner") < page.index('class="panel"')


def test_the_harvest_bars_clamp_instead_of_overflowing(tmp_path: Path) -> None:
    """Passing a target must not draw a bar wider than its track."""
    module = _harvest_module()
    assert "width:100.00%" in module.bar("done", 900_000, 550_000)
    assert "width:0.00%" in module.bar("nothing", 0, 550_000)
    assert "width:0.00%" in module.bar("no target", 5, 0)  # never a division by zero
    assert "—" in module.bar("unknown", None, 100)
    for value in (float("nan"), float("inf")):
        assert "nan" not in module.bar("weird", value, 100).lower()


def test_the_harvest_page_can_count_positions_straight_out_of_the_npz(
    tmp_path: Path,
) -> None:
    """No sidecar and no numpy: the row count comes from the npy header."""
    game = _converted(seed=7)
    npz = tmp_path / "replay.npz"
    stats = build_dataset(
        [(game, _elite_meta(PLAYERS[0]))], npz, require_target=True, write_stats=False
    )
    assert not (tmp_path / "replay.stats.json").exists()
    module = _harvest_module()
    assert module.npz_rows(npz) == stats.positions
    assert module.npz_rows(tmp_path / "missing.npz") is None

    page, harvest = _render(tmp_path, tmp_path / "harvest.html", npz=npz)
    assert harvest.positions == stats.positions
    assert f"{stats.positions:,}" in page or f"{stats.positions}" in page
