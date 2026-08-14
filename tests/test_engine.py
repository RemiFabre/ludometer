"""Tests for the Azul rules engine (see docs/DESIGN.md).

Test helpers below take care of tile bookkeeping: every tile placed on the board
by a test is *drawn* from the bag/lid, so the 100-tile census stays exact and the
conservation assertions are meaningful.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from ludometer.azul.engine import (
    ACTION_SPACE,
    CENTER,
    ENCODED_SIZE,
    FLOOR,
    FLOOR_PENALTIES,
    AzulState,
    decode_action,
    encode_action,
    wall_col,
)

# ---------------------------------------------------------------- helpers


def draw(s: AzulState, color: int, n: int = 1) -> None:
    """Remove n tiles of `color` from the bag (falling back to the lid)."""
    for _ in range(n):
        try:
            s.bag.remove(color)
        except ValueError:
            assert s.lid[color] > 0, f"no color {color} left to draw"
            s.lid[color] -= 1


def blank_round(s: AzulState) -> None:
    """Move every factory/center tile to the lid so a test can hand-place tiles."""
    for f in s.factories:
        for c in range(5):
            s.lid[c] += f[c]
            f[c] = 0
    for c in range(5):
        s.lid[c] += s.center[c]
        s.center[c] = 0
    s.recount()


def set_factory(s: AzulState, i: int, counts: list[int]) -> None:
    for c in range(5):
        s.lid[c] += s.factories[i][c]
        s.factories[i][c] = 0
    for c, n in enumerate(counts):
        if n:
            draw(s, c, n)
            s.factories[i][c] = n
    s.recount()


def set_center(s: AzulState, counts: list[int]) -> None:
    for c in range(5):
        s.lid[c] += s.center[c]
        s.center[c] = 0
    for c, n in enumerate(counts):
        if n:
            draw(s, c, n)
            s.center[c] = n
    s.recount()


def set_line(s: AzulState, p: int, r: int, color: int, n: int) -> None:
    draw(s, color, n)
    s.pl_color[p][r] = color if n else -1
    s.pl_count[p][r] = n
    s.recount()


def set_floor(s: AzulState, p: int, color: int, n: int) -> None:
    draw(s, color, n)
    s.floor[p][color] += n


def set_wall(s: AzulState, p: int, r: int, color: int) -> None:
    draw(s, color, 1)
    s.walls[p][r * 5 + wall_col(color, r)] = 1
    s.recount()


def set_wall_cell(s: AzulState, p: int, r: int, col: int) -> int:
    """Fill wall square (r, col); returns the color that lives there."""
    color = (col - r) % 5
    set_wall(s, p, r, color)
    return color


def assert_conserved(s: AzulState) -> None:
    cen = s.tile_census()
    assert cen == [20, 20, 20, 20, 20], cen


def assert_caches_fresh(s: AzulState) -> None:
    """The incrementally maintained caches must match a from-scratch rebuild."""
    masks = [m[:] for m in s.open_mask]
    tiles_left = s.tiles_left
    s.recount()
    assert masks == s.open_mask
    assert tiles_left == s.tiles_left


def finish_round(s: AzulState, *, keeper: int = 0) -> None:
    """Clear the board with a single move by `keeper`, resolving the round."""
    blank_round(s)
    set_factory(s, 0, [4, 0, 0, 0, 0])
    s.marker_in_center = False
    s.floor_marker[keeper] = True
    s.current_player = keeper
    s.recount()
    s.apply(encode_action(0, 0, FLOOR))


# ---------------------------------------------------------------- action space


def test_action_space_size():
    assert ACTION_SPACE == 180
    assert AzulState.ACTION_SPACE == 180


def test_action_encoding_roundtrip():
    seen = set()
    for a in range(ACTION_SPACE):
        src, col, dest = decode_action(a)
        assert 0 <= src <= 5
        assert 0 <= col <= 4
        assert 0 <= dest <= 5
        assert encode_action(src, col, dest) == a
        seen.add((src, col, dest))
    assert len(seen) == ACTION_SPACE


def test_action_encoding_formula():
    for src in range(6):
        for col in range(5):
            for dest in range(6):
                assert encode_action(src, col, dest) == src * 30 + col * 6 + dest
    assert CENTER == 5
    assert FLOOR == 5


def test_wall_col_formula():
    for c in range(5):
        for r in range(5):
            assert wall_col(c, r) == (c + r) % 5
    for r in range(5):
        assert sorted(wall_col(c, r) for c in range(5)) == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------- setup


def test_new_game_setup():
    s = AzulState.new_game(seed=1)
    assert len(s.factories) == 5
    assert all(sum(f) == 4 for f in s.factories)
    assert sum(s.center) == 0
    assert s.marker_in_center is True
    assert s.scores == [0, 0]
    assert s.current_player == 0
    assert s.is_terminal is False
    assert s.outcome() is None
    assert len(s.bag) == 80
    assert s.lid == [0, 0, 0, 0, 0]
    assert all(sum(w) == 0 for w in s.walls)
    assert s.round_index == 0
    assert_conserved(s)


def test_new_game_rejects_non_two_player():
    with pytest.raises(ValueError):
        AzulState.new_game(seed=0, num_players=3)


def test_same_seed_same_game():
    def play(seed):
        s = AzulState.new_game(seed=seed)
        rng = random.Random(999)
        trace = []
        while not s.is_terminal:
            a = rng.choice(s.legal_actions())
            s.apply(a)
            trace.append((a, tuple(s.scores)))
        return trace, list(s.scores)

    t1, sc1 = play(7)
    t2, sc2 = play(7)
    t3, _ = play(8)
    assert t1 == t2
    assert sc1 == sc2
    assert t1 != t3


# ---------------------------------------------------------------- taking tiles


def test_take_from_factory_moves_rest_to_center():
    s = AzulState.new_game(seed=3)
    blank_round(s)
    set_factory(s, 0, [2, 1, 1, 0, 0])
    set_factory(s, 1, [0, 0, 0, 4, 0])  # keeps the round alive
    s.apply(encode_action(0, 0, FLOOR))
    assert s.factories[0] == [0, 0, 0, 0, 0]
    assert s.center == [0, 1, 1, 0, 0]
    assert s.floor[0] == [2, 0, 0, 0, 0]
    assert s.current_player == 1
    assert_conserved(s)


def test_take_from_center_leaves_other_colors():
    s = AzulState.new_game(seed=3)
    blank_round(s)
    set_center(s, [3, 2, 0, 0, 0])
    s.marker_in_center = False
    set_factory(s, 1, [0, 0, 0, 4, 0])
    s.apply(encode_action(CENTER, 0, 2))
    assert s.center == [0, 2, 0, 0, 0]
    assert s.pl_color[0][2] == 0
    assert s.pl_count[0][2] == 3
    assert_conserved(s)


def test_illegal_actions_rejected():
    s = AzulState.new_game(seed=3)
    blank_round(s)
    set_factory(s, 0, [4, 0, 0, 0, 0])
    set_line(s, 0, 3, 2, 1)
    with pytest.raises(ValueError):
        s.apply(encode_action(1, 0, FLOOR))  # empty factory
    with pytest.raises(ValueError):
        s.apply(encode_action(0, 1, FLOOR))  # color absent from that factory
    with pytest.raises(ValueError):
        s.apply(encode_action(CENTER, 0, FLOOR))  # center empty
    with pytest.raises(ValueError):
        s.apply(encode_action(0, 0, 3))  # row already holds another color
    with pytest.raises(ValueError):
        s.apply(999)
    with pytest.raises(ValueError):
        s.apply(-1)


def test_overflow_into_undersized_row_is_legal():
    s = AzulState.new_game(seed=3)
    blank_round(s)
    set_factory(s, 0, [4, 0, 0, 0, 0])
    set_factory(s, 1, [0, 4, 0, 0, 0])
    s.apply(encode_action(0, 0, 0))  # 4 tiles into a row of capacity 1
    assert s.pl_count[0][0] == 1
    assert s.floor[0] == [3, 0, 0, 0, 0]
    assert_conserved(s)


# ---------------------------------------------------------------- first player marker


def test_first_center_taker_gets_marker_on_floor():
    s = AzulState.new_game(seed=5)
    blank_round(s)
    set_center(s, [2, 2, 0, 0, 0])
    set_factory(s, 0, [0, 0, 4, 0, 0])
    assert s.marker_in_center
    s.apply(encode_action(CENTER, 0, 0))
    assert s.marker_in_center is False
    assert s.floor_marker == [True, False]
    # marker occupies a floor slot: 1 overflow tile + marker = 2 slots
    assert s.floor_occupied(0) == 2
    assert s.floor_penalty(0) == FLOOR_PENALTIES[0] + FLOOR_PENALTIES[1]
    # a later center take does not hand out a second marker
    s.apply(encode_action(CENTER, 1, 1))
    assert s.floor_marker == [True, False]
    assert_conserved(s)


def test_marker_holder_starts_next_round():
    s = AzulState.new_game(seed=5)
    blank_round(s)
    set_factory(s, 0, [4, 0, 0, 0, 0])
    set_center(s, [0, 2, 0, 0, 0])
    s.apply(encode_action(0, 0, FLOOR))  # P0 takes a factory
    assert s.current_player == 1
    s.apply(encode_action(CENTER, 1, FLOOR))  # P1 takes the center + marker
    # round is over; a new one has been dealt
    assert s.round_index == 1
    assert s.first_player == 1
    assert s.current_player == 1
    assert s.marker_in_center is True
    assert s.floor_marker == [False, False]
    assert_conserved(s)


def test_marker_never_taken_keeps_alternation():
    s = AzulState.new_game(seed=5)
    blank_round(s)
    set_factory(s, 0, [4, 0, 0, 0, 0])  # monochrome: nothing reaches the center
    s.apply(encode_action(0, 0, FLOOR))
    assert s.marker_in_center is True
    assert s.current_player == 1  # P1 would have moved next, so P1 starts
    assert s.first_player == 1
    assert_conserved(s)


# ---------------------------------------------------------------- pattern lines


def test_pattern_line_single_color_only():
    s = AzulState.new_game(seed=9)
    blank_round(s)
    set_line(s, 0, 3, 2, 1)
    set_factory(s, 0, [2, 0, 2, 0, 0])
    legal = set(s.legal_actions())
    assert encode_action(0, 2, 3) in legal  # same color, room left
    assert encode_action(0, 0, 3) not in legal  # different color
    assert encode_action(0, 0, FLOOR) in legal


def test_full_pattern_line_is_closed():
    s = AzulState.new_game(seed=9)
    blank_round(s)
    set_line(s, 0, 1, 2, 2)  # capacity 2 -> full
    set_factory(s, 0, [0, 0, 4, 0, 0])
    legal = set(s.legal_actions())
    assert encode_action(0, 2, 1) not in legal
    assert encode_action(0, 2, 2) in legal


def test_wall_row_color_excluded():
    s = AzulState.new_game(seed=9)
    blank_round(s)
    r, c = 2, 3
    set_wall(s, 0, r, c)
    set_factory(s, 0, [0, 0, 0, 4, 0])
    legal = set(s.legal_actions())
    assert encode_action(0, c, r) not in legal
    assert encode_action(0, c, 1) in legal
    assert encode_action(0, c, FLOOR) in legal


def test_floor_is_always_legal():
    s = AzulState.new_game(seed=9)
    blank_round(s)
    for r in range(5):
        set_wall(s, 0, r, 0)
    set_factory(s, 0, [4, 0, 0, 0, 0])
    assert s.legal_actions() == [encode_action(0, 0, FLOOR)]


def test_floor_overflow_goes_to_lid():
    s = AzulState.new_game(seed=9)
    blank_round(s)
    set_floor(s, 0, 0, 6)  # 6 slots used
    set_factory(s, 0, [0, 4, 0, 0, 0])
    set_factory(s, 1, [0, 0, 4, 0, 0])
    before_lid = list(s.lid)
    s.apply(encode_action(0, 1, FLOOR))
    assert s.floor_occupied(0) == 7
    assert s.floor[0] == [6, 1, 0, 0, 0]
    assert s.lid[1] == before_lid[1] + 3
    assert_conserved(s)


def test_marker_counts_toward_floor_capacity():
    s = AzulState.new_game(seed=9)
    blank_round(s)
    set_floor(s, 0, 4, 6)
    s.floor_marker[0] = True  # 7 slots used already
    set_factory(s, 0, [2, 0, 0, 0, 0])
    set_factory(s, 1, [0, 0, 4, 0, 0])
    before = s.lid[0]
    s.apply(encode_action(0, 0, FLOOR))
    assert s.lid[0] == before + 2  # no room at all
    assert s.floor_penalty(0) == sum(FLOOR_PENALTIES)
    assert_conserved(s)


def test_floor_penalty_table():
    s = AzulState.new_game(seed=9)
    expected = [0, -1, -2, -4, -6, -8, -11, -14, -14]
    for n in range(9):
        s.floor[0] = [0, 0, 0, 0, 0]
        s.floor[0][0] = n
        s.floor_marker[0] = False
        assert s.floor_penalty(0) == expected[n], n
    assert list(FLOOR_PENALTIES) == [-1, -1, -2, -2, -2, -3, -3]


# ---------------------------------------------------------------- round end / tiling


def test_complete_line_tiles_wall_rest_to_lid():
    s = AzulState.new_game(seed=11)
    set_line(s, 0, 2, 4, 3)  # complete (capacity 3)
    set_line(s, 0, 4, 2, 3)  # incomplete: carries over
    lid_before = list(s.lid)
    finish_round(s, keeper=1)
    assert s.walls[0][2 * 5 + wall_col(4, 2)] == 1
    assert s.pl_count[0][2] == 0
    assert s.pl_color[0][2] == -1
    assert s.lid[4] >= lid_before[4] + 2  # 3 tiles: 1 to the wall, 2 to the lid
    assert s.pl_color[0][4] == 2
    assert s.pl_count[0][4] == 3
    assert_conserved(s)


def test_scoring_isolated_tile_scores_one():
    s = AzulState.new_game(seed=11)
    set_line(s, 0, 0, 0, 1)
    finish_round(s, keeper=1)
    assert s.scores[0] == 1


def test_scoring_horizontal_and_vertical_runs():
    """A tile joining a horizontal run of 3 and a vertical run of 2 scores 5."""
    s = AzulState.new_game(seed=11)
    r, col = 2, 2
    color = (col - r) % 5
    set_wall_cell(s, 0, r, col - 1)
    set_wall_cell(s, 0, r, col + 1)
    set_wall_cell(s, 0, r - 1, col)
    set_line(s, 0, r, color, r + 1)
    finish_round(s, keeper=1)
    assert s.walls[0][r * 5 + col] == 1
    assert s.scores[0] == 5


def test_scoring_multiple_lines_accumulates_top_down():
    s = AzulState.new_game(seed=11)
    # rows 0 and 1 each get a tile in column 2: row 0 isolated (1),
    # row 1 then forms a vertical run of 2 (2). Total 3.
    c0 = (2 - 0) % 5
    c1 = (2 - 1) % 5
    set_line(s, 0, 0, c0, 1)
    set_line(s, 0, 1, c1, 2)
    finish_round(s, keeper=1)
    assert s.walls[0][0 * 5 + 2] == 1
    assert s.walls[0][1 * 5 + 2] == 1
    assert s.scores[0] == 3


def test_full_row_scores_five_for_last_tile():
    s = AzulState.new_game(seed=11)
    r = 3
    for col in range(4):
        set_wall_cell(s, 0, r, col)
    color = (4 - r) % 5
    set_line(s, 0, r, color, r + 1)
    finish_round(s, keeper=1)
    # last tile of the row: horizontal run of 5, vertical run of 1 -> 5, plus
    # the +2 completed-row bonus at game end (game ends this round).
    assert s.is_terminal
    assert s.scores[0] == 5 + 2


def test_floor_penalty_applied_and_score_floored_at_zero():
    s = AzulState.new_game(seed=11)
    s.scores[1] = 1
    set_floor(s, 1, 0, 3)  # -1 -1 -2 = -4
    finish_round(s, keeper=0)
    assert s.scores[1] == 0
    assert s.floor[1] == [0, 0, 0, 0, 0]
    assert_conserved(s)


def test_round_end_refills_and_returns_marker():
    s = AzulState.new_game(seed=11)
    finish_round(s, keeper=1)
    assert all(sum(f) == 4 for f in s.factories)
    assert sum(s.center) == 0
    assert s.marker_in_center is True
    assert s.first_player == 1
    assert s.current_player == 1
    assert s.floor_marker == [False, False]
    assert s.round_index == 1
    assert_conserved(s)


def test_bag_refill_from_lid():
    s = AzulState.new_game(seed=12)
    blank_round(s)
    # empty the bag into the lid, then hand out one monochrome factory
    for c in s.bag:
        s.lid[c] += 1
    s.bag.clear()
    set_factory(s, 0, [4, 0, 0, 0, 0])
    s.marker_in_center = False
    s.floor_marker[0] = True
    lid_total_before = sum(s.lid)
    assert lid_total_before >= 20
    s.apply(encode_action(0, 0, FLOOR))  # ends the round -> refill from the lid
    assert sum(map(sum, s.factories)) == 20
    # the lid (incl. the 4 floor tiles) went into the bag, 20 came back out
    assert len(s.bag) == lid_total_before + 4 - 20
    assert s.lid == [0, 0, 0, 0, 0]
    assert_conserved(s)


def test_bag_and_lid_empty_leaves_factories_short():
    """Artificial state (census intentionally short): play must continue."""
    s = AzulState.new_game(seed=12)
    blank_round(s)
    s.bag.clear()
    s.lid[:] = [0, 0, 0, 0, 0]
    s.factories[0][0] = 6
    s.marker_in_center = False
    s.floor_marker[0] = True
    s.recount()
    s.apply(encode_action(0, 0, FLOOR))  # 6 tiles + marker fill the floor line
    total = sum(map(sum, s.factories)) + sum(s.center)
    assert total == 6  # only the 6 recycled floor tiles were available
    assert s.is_terminal is False
    assert s.legal_actions()


def test_completely_exhausted_board_ends_the_game():
    """Artificial state: nothing left to deal -> the game has to stop."""
    s = AzulState.new_game(seed=12)
    blank_round(s)
    s.bag.clear()
    s.lid[:] = [0, 0, 0, 0, 0]
    s.factories[0][0] = 1
    s.marker_in_center = False
    s.floor_marker[0] = True
    s.recount()
    s.apply(encode_action(0, 0, 0))  # the very last tile
    assert s.is_terminal is True
    assert s.exhausted is True
    assert s.outcome() in (-1.0, 0.0, 1.0)


# ---------------------------------------------------------------- game end


def fill_row_except_last(s: AzulState, p: int, r: int) -> int:
    """Fill wall row r of player p leaving column 4 free; return its color."""
    for col in range(4):
        set_wall_cell(s, p, r, col)
    return (4 - r) % 5


def test_game_ends_when_row_completed():
    s = AzulState.new_game(seed=13)
    c = fill_row_except_last(s, 0, 1)
    set_line(s, 0, 1, c, 2)
    finish_round(s, keeper=1)
    assert s.is_terminal is True
    assert s.completed_rows(0) == 1
    assert s.outcome() is not None
    assert all(sc >= 0 for sc in s.scores)
    assert s.legal_actions() == []
    assert_conserved(s)


def test_no_end_when_row_incomplete():
    s = AzulState.new_game(seed=13)
    fill_row_except_last(s, 0, 1)
    finish_round(s, keeper=1)
    assert s.is_terminal is False


def test_column_bonus():
    s = AzulState.new_game(seed=13)
    # complete column 4 and row 0 with one placement at (0, 4)
    for r in range(1, 5):
        set_wall_cell(s, 0, r, 4)
    color = fill_row_except_last(s, 0, 0)
    set_line(s, 0, 0, color, 1)
    finish_round(s, keeper=1)
    assert s.is_terminal
    assert s.completed_rows(0) == 1
    assert s.completed_cols(0) == 1
    # placed tile: h run 5 + v run 5 = 10, plus +2 row and +7 column
    assert s.scores[0] == 10 + 2 + 7


def test_color_bonus():
    s = AzulState.new_game(seed=13)
    color = 3
    for r in range(1, 5):
        set_wall(s, 0, r, color)
    assert s.completed_colors(0) == 0
    # row 0 lacks exactly the square of `color`, so one placement completes both
    # the row (ending the game) and the color set.
    target = wall_col(color, 0)
    for col in range(5):
        if col != target:
            set_wall_cell(s, 0, 0, col)
    set_line(s, 0, 0, color, 1)
    finish_round(s, keeper=1)
    assert s.is_terminal
    assert s.completed_colors(0) == 1
    assert s.completed_rows(0) == 1
    # placed tile: h run 5, v run 1 -> 5; bonuses +2 row, +10 color
    assert s.scores[0] == 5 + 2 + 10


def test_end_bonus_arithmetic():
    s = AzulState.new_game(seed=13)
    # full wall for P0 except one square, filled from a pattern line
    for r in range(5):
        for col in range(5):
            if (r, col) != (0, 0):
                set_wall_cell(s, 0, r, col)
    set_line(s, 0, 0, 0, 1)  # wall_col(0, 0) == 0
    finish_round(s, keeper=1)
    assert s.is_terminal
    assert s.completed_rows(0) == 5
    assert s.completed_cols(0) == 5
    assert s.completed_colors(0) == 5
    # last tile: h 5 + v 5 = 10; bonuses 5*2 + 5*7 + 5*10 = 95
    assert s.scores[0] == 10 + 95


def test_outcome_values_and_tiebreak():
    s = AzulState.new_game(seed=14)
    s.is_terminal = True
    s.scores = [30, 20]
    assert s.outcome() == 1.0
    s.scores = [20, 30]
    assert s.outcome() == -1.0
    s.scores = [25, 25]
    assert s.outcome() == 0.0
    for col in range(5):
        s.walls[1][0 * 5 + col] = 1
    assert s.completed_rows(1) == 1
    assert s.outcome() == -1.0
    for col in range(5):
        s.walls[0][1 * 5 + col] = 1
        s.walls[0][2 * 5 + col] = 1
    assert s.completed_rows(0) == 2
    assert s.outcome() == 1.0


def test_no_legal_actions_only_when_terminal():
    s = AzulState.new_game(seed=15)
    rng = random.Random(0)
    while not s.is_terminal:
        legal = s.legal_actions()
        assert legal
        s.apply(rng.choice(legal))
    assert s.legal_actions() == []
    with pytest.raises(ValueError):
        s.apply(0)


# ---------------------------------------------------------------- clone


def test_clone_independence():
    s = AzulState.new_game(seed=16)
    rng = random.Random(3)
    for _ in range(6):
        s.apply(rng.choice(s.legal_actions()))
    t = s.clone()
    assert t.to_json() == s.to_json()
    tr = random.Random(4)
    while not t.is_terminal:
        t.apply(tr.choice(t.legal_actions()))
    assert s.is_terminal is False
    assert t.to_json() != s.to_json()
    assert t.factories is not s.factories
    assert t.factories[0] is not s.factories[0]
    assert t.walls[0] is not s.walls[0]
    assert t.pl_count[0] is not s.pl_count[0]
    assert t.pl_color[0] is not s.pl_color[0]
    assert t.floor[0] is not s.floor[0]
    assert t.bag is not s.bag
    assert t.lid is not s.lid
    assert t.scores is not s.scores
    assert t.floor_marker is not s.floor_marker


def test_clone_replays_identically():
    s = AzulState.new_game(seed=17)
    rng = random.Random(1)
    a = s.clone()
    b = s.clone()
    moves = []
    while not a.is_terminal:
        m = rng.choice(a.legal_actions())
        moves.append(m)
        a.apply(m)
    for m in moves:
        b.apply(m)
    assert a.scores == b.scores
    assert a.to_json() == b.to_json()


# ---------------------------------------------------------------- encode / render


def test_encoded_size_and_dtype():
    s = AzulState.new_game(seed=18)
    v = s.encode()
    assert isinstance(v, np.ndarray)
    assert v.dtype == np.float32
    assert v.shape == (ENCODED_SIZE,)
    assert np.all(np.isfinite(v))
    assert float(v.min()) >= -1.0 and float(v.max()) <= 2.0


def test_encode_is_current_player_relative():
    s = AzulState.new_game(seed=18)
    s.walls[0][0] = 1
    s.scores = [10, 0]
    v0 = s.encode()
    s.current_player = 1
    v1 = s.encode()
    assert v0[0] == 1.0  # my wall
    assert v1[0] == 0.0
    assert v1[25] == 1.0  # opponent's wall
    assert v0[124] > v0[125]
    assert v1[125] > v1[124]


def test_encode_reflects_the_board():
    s = AzulState.new_game(seed=18)
    blank_round(s)
    set_factory(s, 0, [4, 0, 0, 0, 0])
    set_center(s, [0, 2, 0, 0, 0])
    set_line(s, 0, 2, 3, 2)
    v = s.encode()
    # factory 0, color 0 count = 4/4
    assert v[126] == 1.0
    assert v[127] == 0.0
    # center color 1
    assert v[156 + 1] > 0.0
    # my pattern line row 2: one-hot color 3, fill 2/3
    row2 = v[50 + 2 * 6 : 50 + 3 * 6]
    assert list(row2[:5]) == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert row2[5] == pytest.approx(2 / 3)
    assert v[162] == 1.0  # marker still in the center


def test_encode_stays_in_shape_over_a_game():
    s = AzulState.new_game(seed=19)
    rng = random.Random(2)
    while not s.is_terminal:
        v = s.encode()
        assert v.shape == (ENCODED_SIZE,)
        assert np.all(np.isfinite(v))
        s.apply(rng.choice(s.legal_actions()))
    assert s.encode().shape == (ENCODED_SIZE,)


def test_render_text_and_json():
    s = AzulState.new_game(seed=20)
    txt = s.render_text()
    assert isinstance(txt, str)
    assert "factories" in txt.lower()
    assert len(txt.splitlines()) > 8
    d = s.to_json()
    for key in (
        "current_player",
        "factories",
        "center",
        "marker_in_center",
        "players",
        "scores",
        "bag",
        "lid",
        "is_terminal",
        "round",
        "legal_actions",
        "first_player",
        "outcome",
    ):
        assert key in d, key
    assert len(d["players"]) == 2
    for pl in d["players"]:
        assert len(pl["wall"]) == 5
        assert all(len(row) == 5 for row in pl["wall"])
        assert len(pl["pattern_lines"]) == 5
    json.dumps(d)  # must be JSON-serialisable


# ---------------------------------------------------------------- invariants / fuzz


def test_conservation_every_step():
    s = AzulState.new_game(seed=21)
    rng = random.Random(5)
    assert_conserved(s)
    while not s.is_terminal:
        s.apply(rng.choice(s.legal_actions()))
        assert_conserved(s)


def test_derived_caches_stay_fresh():
    s = AzulState.new_game(seed=22)
    rng = random.Random(6)
    assert_caches_fresh(s)
    while not s.is_terminal:
        s.apply(rng.choice(s.legal_actions()))
        assert_caches_fresh(s)


@pytest.mark.parametrize("seed", [30, 31, 32, 33, 34])
def test_legal_actions_matches_brute_force(seed):
    """legal_actions() must be exactly the ascending list of legal ids."""
    s = AzulState.new_game(seed=seed)
    rng = random.Random(seed)
    while not s.is_terminal:
        legal = s.legal_actions()
        brute = [a for a in range(ACTION_SPACE) if s.is_legal(a)]
        assert legal == brute
        s.apply(rng.choice(legal))
    assert [a for a in range(ACTION_SPACE) if s.is_legal(a)] == []


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_random_game_is_sane(seed):
    s = AzulState.new_game(seed=seed)
    rng = random.Random(seed)
    moves = 0
    while not s.is_terminal:
        s.apply(rng.choice(s.legal_actions()))
        moves += 1
        assert moves < 2000
    assert s.round_index >= 1
    assert all(sc >= 0 for sc in s.scores)
    assert s.outcome() in (-1.0, 0.0, 1.0)
    assert s.completed_rows(0) or s.completed_rows(1) or s.exhausted


def test_fuzz_many_full_games():
    seeds = random.Random(1234)
    max_moves = 0
    for _ in range(200):
        seed = seeds.randrange(1 << 30)
        s = AzulState.new_game(seed=seed)
        rng = random.Random(seed ^ 0x5EED)
        moves = 0
        while not s.is_terminal:
            legal = s.legal_actions()
            assert legal, "no legal action in a non-terminal state"
            s.apply(rng.choice(legal))
            moves += 1
            assert moves < 1000, "game did not terminate"
        max_moves = max(max_moves, moves)
        assert all(sc >= 0 for sc in s.scores)
        assert s.outcome() in (-1.0, 0.0, 1.0)
        assert s.tile_census() == [20, 20, 20, 20, 20]
        assert s.legal_actions() == []
    assert max_moves < 400
