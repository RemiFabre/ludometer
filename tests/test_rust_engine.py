"""The Rust Azul rules (`ludometer_rs.State`) against the Python engine, exactly.

Acceptance layer 1 of docs/RUST_ENGINE.md §6: on random-play games from the same
seed and on every BGA replay, the two engines must agree on every legal-move
list, every `encode()` row (exact float equality), every `chance_key`,
`fingerprint` and `is_stochastic`, every score and outcome. The Rust engine's
`rng="python"` mode reproduces CPython's `random.Random`, so "same seed" really
does mean "same deals" — nothing is scripted in the random-play test.

Skipped when `ludometer_rs` is not built (`rust/README.md` says how).
"""

from __future__ import annotations

import gzip
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

from ludometer.azul.engine import ACTION_SPACE, AzulState, encode_action

rs = pytest.importorskip("ludometer_rs")

REPO = Path(__file__).resolve().parents[1]
BGA = REPO / "data" / "cloud" / "bga_positions.json.gz"


def assert_same(py: AzulState, rust: rs.State, where: str = "") -> None:
    """Every observable of the two positions is identical."""
    assert rust.is_terminal == py.is_terminal, where
    assert rust.current_player == py.current_player, where
    assert rust.first_player == py.first_player, where
    assert rust.round_index == py.round_index, where
    assert rust.tiles_left == py.tiles_left, where
    assert rust.scores == py.scores, where
    assert rust.factories == py.factories, where
    assert rust.center == py.center, where
    assert rust.lid == py.lid, where
    assert rust.bag == py.bag, where  # order included
    assert rust.walls == py.walls, where
    assert rust.pl_color == py.pl_color, where
    assert rust.pl_count == py.pl_count, where
    assert rust.floor == py.floor, where
    assert rust.floor_marker == py.floor_marker, where
    assert rust.marker_in_center == py.marker_in_center, where
    legal_py = py.legal_actions()
    assert rust.legal_actions() == legal_py, where
    for a in range(ACTION_SPACE):
        assert rust.is_legal(a) == py.is_legal(a), f"{where} is_legal({a})"
    for a in legal_py:
        assert rust.is_stochastic(a) == py.is_stochastic(a), f"{where} stochastic({a})"
    enc_py = py.encode()
    enc_rs = rust.encode()
    assert enc_rs.dtype == np.float32 and enc_rs.shape == (182,)
    assert np.array_equal(enc_rs, enc_py), f"{where} encode differs at {np.nonzero(enc_rs != enc_py)[0]}"
    assert rust.chance_key() == py.chance_key(), where
    assert rust.fingerprint() == py.fingerprint(), where
    assert rust.outcome() == py.outcome(), where
    assert rust.tile_census() == py.tile_census() == [20] * 5, where
    for p in (0, 1):
        assert rust.wall_summary(p) == py.wall_summary(p), where
        assert rust.completed_rows(p) == py.completed_rows(p), where
        assert rust.completed_cols(p) == py.completed_cols(p), where
        assert rust.completed_colors(p) == py.completed_colors(p), where
        assert rust.floor_occupied(p) == py.floor_occupied(p), where
        assert rust.floor_penalty(p) == py.floor_penalty(p), where


def random_play(seed: int, max_moves: int = 400) -> tuple[AzulState, rs.State, int]:
    py = AzulState.new_game(seed)
    rust = rs.State.new_game(seed, rng="python")
    picker = random.Random(seed * 7 + 1)
    moves = 0
    while not py.is_terminal and moves < max_moves:
        assert_same(py, rust, f"seed {seed} move {moves}")
        legal = py.legal_actions()
        a = legal[picker.randrange(len(legal))]
        py.apply(a)
        rust.apply(a)
        moves += 1
    assert_same(py, rust, f"seed {seed} final")
    return py, rust, moves


@pytest.mark.parametrize("seed", list(range(0, 60)))
def test_random_play_matches_python(seed: int) -> None:
    py, rust, moves = random_play(seed)
    assert py.is_terminal
    assert rust.outcome() == py.outcome()


@pytest.mark.skipif(not os.environ.get("LUDOMETER_SLOW"), reason="set LUDOMETER_SLOW=1")
def test_random_play_matches_python_ten_thousand_seeds() -> None:
    for seed in range(10_000):
        random_play(seed)


def test_new_game_fast_rng_is_a_valid_deal() -> None:
    s = rs.State.new_game(4, rng="fast")
    assert s.tiles_left == 20
    assert s.tile_census() == [20] * 5
    assert len(s.bag) == 80
    assert s.rng_kind == "fast"
    # Different seeds deal differently; the same seed deals the same.
    assert rs.State.new_game(4).chance_key() == s.chance_key()
    assert rs.State.new_game(5).chance_key() != s.chance_key()


def test_clone_is_independent() -> None:
    s = rs.State.new_game(1, rng="python")
    c = s.clone()
    c.apply(c.legal_actions()[0])
    assert s.tiles_left == 20
    assert c.tiles_left < 20


def test_apply_rejects_illegal_moves_like_python() -> None:
    py = AzulState.new_game(2)
    rust = rs.State.new_game(2, rng="python")
    for a in range(ACTION_SPACE):
        if py.is_legal(a):
            continue
        with pytest.raises(ValueError):
            py.clone().apply(a)
        with pytest.raises(ValueError):
            rust.clone().apply(a)


def near_round_end(seed: int) -> tuple[AzulState, rs.State]:
    """One tile left on the board (the tests' recipe), on both engines."""
    py = AzulState.new_game(seed)
    for factory in py.factories:
        for c in range(5):
            py.lid[c] += factory[c]
            factory[c] = 0
    for c in range(5):
        py.lid[c] += py.center[c]
        py.center[c] = 0
    py.lid[0] -= 1
    py.factories[0][0] = 1
    py.recount()
    rust = rs.State.from_dict(
        {
            "factories": py.factories,
            "center": py.center,
            "marker_in_center": py.marker_in_center,
            "bag": py.bag,
            "lid": py.lid,
            "walls": py.walls,
            "pl_color": py.pl_color,
            "pl_count": py.pl_count,
            "floor": py.floor,
            "floor_marker": py.floor_marker,
            "scores": py.scores,
            "current_player": py.current_player,
            "first_player": py.first_player,
            "round_index": py.round_index,
            "is_terminal": py.is_terminal,
            "exhausted": py.exhausted,
        },
        rng="python",
    )
    assert_same(py, rust, "near_round_end")
    return py, rust


@pytest.mark.parametrize("seed", [5, 6, 7])
def test_determinize_matches_python(seed: int) -> None:
    py, rust = near_round_end(seed)
    action = encode_action(0, 0, 5)
    assert py.is_stochastic(action) and rust.is_stochastic(action)
    keys = set()
    for det_seed in range(40):
        a = py.determinize(action, det_seed)
        b = rust.determinize(action, det_seed)
        assert_same(a, b, f"determinize seed {det_seed}")
        keys.add(b.chance_key())
    assert len(keys) > 20  # different seeds really do sample different refills


def test_from_dict_round_trips_through_to_dict() -> None:
    a = rs.State.new_game(11, rng="python")
    a.apply(a.legal_actions()[3])
    b = rs.State.from_dict(a.to_dict(), rng="python")
    assert b.to_dict() == a.to_dict()
    assert np.array_equal(a.encode(), b.encode())


# ------------------------------------------------------------------ BGA replays
def _load_bga(limit: int | None = None) -> list[dict]:
    if not BGA.exists():
        pytest.skip(f"{BGA} missing")
    with gzip.open(BGA, "rt", encoding="utf-8") as fh:
        games = json.load(fh)["games"]
    return games if limit is None else games[:limit]


def replay_both(game: dict) -> int:
    """`ludometer.cloud.label.replay_positions` on both engines, checked per step."""
    deals = game["d"]
    py = AzulState.new_game(seed=0)
    rust = rs.State.new_game(0, rng="python")
    from ludometer.human.convert import apply_deal
    from ludometer.human.parse import Deal

    def deal(i: int) -> None:
        apply_deal(py, Deal(round_index=i, factories=tuple(tuple(f) for f in deals[i])))
        rust.apply_deal(deals[i])

    deal(0)
    py.current_player = py.first_player = game["f"]
    rust.current_player = game["f"]
    rust.first_player = game["f"]
    deal_index = 1
    table = game["t"]
    for k, action in enumerate(game["a"]):
        assert_same(py, rust, f"table {table} step {k}")
        assert py.is_legal(action), f"table {table}: illegal action {action}"
        round_before = py.round_index
        py.apply(action)
        rust.apply(action)
        if py.round_index > round_before and not py.is_terminal:
            deal(deal_index)
            deal_index += 1
    assert_same(py, rust, f"table {table} final")
    if py.is_terminal:  # a few logged games were abandoned before the end
        assert list(py.scores) == list(game["s"])
        assert py.outcome() == game["o"]
    return int(py.is_terminal)


def test_bga_replays_match_python_sample() -> None:
    games = _load_bga(limit=150)
    finished = sum(replay_both(g) for g in games)
    assert finished > 100


@pytest.mark.skipif(not os.environ.get("LUDOMETER_SLOW"), reason="set LUDOMETER_SLOW=1")
def test_bga_replays_match_python_all() -> None:
    games = _load_bga()
    assert len(games) >= 3_000
    finished = sum(replay_both(g) for g in games)
    assert finished > 0.9 * len(games)
