"""Connect Four: the non-trivial half of the calibrated pair (NEXT_GAMES.md §3)."""

from __future__ import annotations

import random

import pytest

from ludometer.agents import make_agent
from ludometer.games import get_game
from ludometer.c4.engine import HEIGHT, WIDTH, Connect4State

BASELINES = ("c4:random", "c4:greedy", "c4:heuristic")


def play(state: Connect4State, *cols: int) -> Connect4State:
    for c in cols:
        state.apply(c)
    return state


def test_a_fresh_board_offers_seven_columns():
    state = Connect4State.new_game(seed=0)
    assert state.legal_actions() == list(range(7))
    assert not state.is_terminal


def test_four_in_a_column_wins():
    state = play(Connect4State.new_game(seed=0), 3, 0, 3, 0, 3, 0, 3)
    assert state.is_terminal
    assert state.outcome() == 1.0


def test_four_in_a_row_wins_for_the_second_player():
    state = play(Connect4State.new_game(seed=0), 0, 3, 0, 4, 0, 5, 6, 2)
    assert state.is_terminal
    assert state.outcome() == -1.0


def test_a_diagonal_wins():
    # X builds the / diagonal from (0,0): needs stacks 0,1,2,3
    state = play(Connect4State.new_game(seed=0), 0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3)
    assert state.is_terminal
    assert state.outcome() == 1.0


def test_a_full_column_is_illegal():
    state = play(Connect4State.new_game(seed=0), 0, 0, 0, 0, 0, 0)
    assert 0 not in state.legal_actions()
    assert not state.is_legal(0)
    with pytest.raises(ValueError):
        state.apply(0)


def test_a_full_board_is_a_draw():
    state = Connect4State.new_game(seed=0)
    rng = random.Random(4)
    while not state.is_terminal:
        state.apply(rng.choice(state.legal_actions()))
    assert state.outcome() in (1.0, -1.0, 0.0)
    assert state.ply <= WIDTH * HEIGHT


def test_the_encoding_is_from_the_movers_seat():
    state = play(Connect4State.new_game(seed=0), 3)  # X in column 3; O to move
    encoded = state.encode()
    assert encoded.shape == (Connect4State.ENCODED_SIZE,)
    cell = 3 * HEIGHT + 0  # column 3, bottom row
    assert encoded[42 + cell] == 1.0  # opponent plane, mover's view
    assert encoded[cell] == 0.0
    state.apply(3)  # O stacks on top; X to move
    encoded = state.encode()
    assert encoded[cell] == 1.0
    assert encoded[42 + 3 * HEIGHT + 1] == 1.0


def test_clone_and_search_hooks():
    state = play(Connect4State.new_game(seed=0), 3, 3, 2)
    twin = state.clone()
    twin.apply(0)
    assert twin.ply == state.ply + 1
    assert not state.is_stochastic(0)
    assert state.search_root(random.Random(0)).fingerprint() == state.fingerprint()
    assert state.wall_summary(1) == [0] * 15


def test_the_registry_serves_it():
    spec = get_game("connect4")
    assert spec.action_space == 7
    assert spec.encoded_size == Connect4State.ENCODED_SIZE
    assert spec.max_moves >= 42


def test_greedy_takes_a_win_and_blocks_a_loss():
    greedy = make_agent("c4:greedy")
    greedy.seed(0)
    # X has three in column 3: winning move is 3
    state = play(Connect4State.new_game(seed=0), 3, 0, 3, 0, 3, 1)
    assert greedy.act(state) == 3
    # O threatens column 0 (three stacked) and X has no win of his own
    state = play(Connect4State.new_game(seed=0), 3, 0, 3, 0, 5, 0)
    assert greedy.act(state) == 0


@pytest.mark.parametrize("spec", BASELINES)
def test_every_baseline_plays_only_legal_moves(spec):
    agent = make_agent(spec)
    agent.seed(1)
    for seed in range(4):
        state = Connect4State.new_game(seed=seed)
        while not state.is_terminal:
            action = agent.act(state)
            assert state.is_legal(action)
            state.apply(action)
