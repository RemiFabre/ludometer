"""Tic-tac-toe: the trivially solved calibration game (NEXT_GAMES.md §3)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from ludometer.agents import make_agent
from ludometer.games import get_game
from ludometer.ttt.engine import TicTacToeState

BASELINES = ("ttt:random", "ttt:greedy", "ttt:heuristic")


def play(state: TicTacToeState, *actions: int) -> TicTacToeState:
    for a in actions:
        state.apply(a)
    return state


def test_a_fresh_board_offers_nine_moves():
    state = TicTacToeState.new_game(seed=0)
    assert state.legal_actions() == list(range(9))
    assert state.current_player == 0
    assert not state.is_terminal


def test_three_in_a_row_wins():
    # X 0,1,2 top row; O elsewhere
    state = play(TicTacToeState.new_game(seed=0), 0, 3, 1, 4, 2)
    assert state.is_terminal
    assert state.outcome() == 1.0
    assert state.legal_actions() == []


def test_the_second_player_can_win_too():
    state = play(TicTacToeState.new_game(seed=0), 0, 6, 1, 7, 4, 8)
    assert state.is_terminal
    assert state.outcome() == -1.0


def test_a_full_board_without_a_line_is_a_draw():
    # X: 0 1 5 6 8 / O: 2 3 4 7 — no line for either
    state = play(TicTacToeState.new_game(seed=0), 0, 2, 1, 4, 5, 3, 6, 7, 8)
    assert state.is_terminal
    assert state.outcome() == 0.0


def test_occupied_cells_are_illegal():
    state = play(TicTacToeState.new_game(seed=0), 4)
    assert 4 not in state.legal_actions()
    assert not state.is_legal(4)
    with pytest.raises(ValueError):
        state.apply(4)


def test_the_encoding_is_from_the_movers_seat():
    state = play(TicTacToeState.new_game(seed=0), 4)  # X center; O to move
    encoded = state.encode()
    assert encoded.shape == (TicTacToeState.ENCODED_SIZE,)
    assert encoded[9 + 4] == 1.0  # the opponent's stone, seen by the mover
    assert encoded[4] == 0.0
    # X to move again after O plays 0: same cell flips planes
    state.apply(0)
    encoded = state.encode()
    assert encoded[4] == 1.0
    assert encoded[9 + 0] == 1.0


def test_clone_is_independent_and_search_hooks_are_trivial():
    state = play(TicTacToeState.new_game(seed=0), 4, 0)
    twin = state.clone()
    twin.apply(twin.legal_actions()[0])
    assert state.legal_actions() != twin.legal_actions() or state.is_terminal
    assert not state.is_stochastic(state.legal_actions()[0])
    view = state.search_root(random.Random(0))
    assert view.fingerprint() == state.fingerprint()
    assert state.wall_summary(0) == [0] * 15


def test_the_registry_serves_it():
    spec = get_game("tictactoe")
    assert spec.action_space == 9
    assert spec.encoded_size == TicTacToeState.ENCODED_SIZE
    assert spec.max_moves >= 9
    state = spec.new_game(3)
    assert isinstance(state, TicTacToeState)


def test_greedy_takes_a_win_and_blocks_a_loss():
    greedy = make_agent("ttt:greedy")
    greedy.seed(0)
    # X can complete 0,1,2 by playing 2
    state = play(TicTacToeState.new_game(seed=0), 0, 3, 1, 4)
    assert greedy.act(state) == 2
    # O must block X's 0,1,_ threat
    state = play(TicTacToeState.new_game(seed=0), 0, 8, 1)
    assert greedy.act(state) == 2


@pytest.mark.parametrize("spec", BASELINES)
def test_every_baseline_plays_only_legal_moves(spec):
    agent = make_agent(spec)
    agent.seed(1)
    for seed in range(5):
        state = TicTacToeState.new_game(seed=seed)
        while not state.is_terminal:
            action = agent.act(state)
            assert state.is_legal(action)
            state.apply(action)


def test_random_games_always_finish():
    rng = random.Random(0)
    outcomes = set()
    for _ in range(50):
        state = TicTacToeState.new_game(seed=0)
        while not state.is_terminal:
            state.apply(rng.choice(state.legal_actions()))
        outcomes.add(state.outcome())
    assert outcomes == {1.0, -1.0, 0.0}
