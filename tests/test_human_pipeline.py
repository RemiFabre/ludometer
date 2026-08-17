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
import io
import json
from dataclasses import replace
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
from ludometer.human.dataset import build_dataset, one_hot_policies
from ludometer.human.fetch import (
    ARENA_MODE,
    GAME_MODE_OPTION,
    Fetcher,
    FetchState,
    PlayerRow,
    TableFilter,
    TableVerdict,
    extract_table_ids,
    extract_table_rows,
    option_value,
    select_players,
    table_row_players,
    table_row_scores,
)
from ludometer.human.fixture import synthetic_log
from ludometer.human.parse import (
    AZUL_COLOR_MAP,
    DEFAULT_SCHEMA,
    ParseError,
    iter_log_entries,
    log_packets,
    log_type_histogram,
    observed_color_ids,
    parse_gamelogs_html,
    parse_log,
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
            entry["args"]["factories"] = [[1, 1, 1, 1, 1]] * 5  # 25 black tiles
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
            entry["args"]["factories"] = [[1, 1, 1, 1]] * 5  # all 20 blacks, legal
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
                            "args": {"factories": [[1, 1], [2], [], [], []]},
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


def test_the_filter_is_fail_closed_about_the_wall_variant() -> None:
    """Until the option id is known, every table is skipped with a loud reason."""
    reason = TableFilter().check(_infos())
    assert "wall variant option id unknown" in reason


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


def test_the_wall_filter_accepts_once_the_option_id_is_known(monkeypatch) -> None:
    """Simulates the one-line change the next session makes after reading a log."""
    from ludometer.human import fetch as fetch_mod

    monkeypatch.setitem(fetch_mod.STANDARD_WALL_OPTION_HINTS, "option_id", 100)
    monkeypatch.setitem(fetch_mod.STANDARD_WALL_OPTION_HINTS, "standard_values", (0,))
    flt = TableFilter()
    assert flt.check(_infos(options={"100": 0})) == ""
    assert "non-standard wall" in flt.check(_infos(options={"100": 1}))
    assert "missing" in flt.check(_infos(options={"101": 0}))


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
