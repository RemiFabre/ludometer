"""The solver harness for the calibrated pair (NEXT_GAMES.md §3):
exact WDL values, the cached position suites, and % optimal scoring."""

from __future__ import annotations

import random

import pytest

from ludometer.c4.engine import Connect4State
from ludometer.solved.solver import c4_solve, ttt_solve
from ludometer.solved.suite import (
    build_suite,
    rebuild_state,
    score_agent,
    solve_state,
)
from ludometer.ttt.engine import TicTacToeState


def ttt_after(*cells: int) -> TicTacToeState:
    state = TicTacToeState.new_game(seed=0)
    for c in cells:
        state.apply(c)
    return state


def c4_after(*cols: int) -> Connect4State:
    state = Connect4State.new_game(seed=0)
    for c in cols:
        state.apply(c)
    return state


# -------------------------------------------------------------------- solvers
def test_tictactoe_is_a_draw_from_the_start():
    assert ttt_solve(0, 0) == 0


def test_a_mover_with_two_in_a_row_and_a_free_cell_wins():
    # X holds 0,1 with 2 free and it is X's move: 1 (X also threatens nothing else)
    state = ttt_after(0, 3, 1, 4)  # X: 0,1 / O: 3,4 — X plays 2 and wins
    assert solve_state(state) == 1


def test_a_mover_facing_a_double_threat_loses():
    # O faces X holding 0,4 with corners open after O's weak reply: classic loss.
    # Simpler exact case: X holds 0,1 AND 3,6 both open; O to move cannot block both.
    state = ttt_after(0, 4, 1, 5, 3)  # X: 0,1,3  O: 4,5 — threats at 2 and 6
    assert solve_state(state) == -1


def test_connect4_immediate_win_is_seen():
    state = c4_after(3, 0, 3, 0, 3, 0)  # X to move, three stacked in column 3
    assert solve_state(state) == 1


def test_connect4_unstoppable_double_threat_is_a_loss():
    # X has 3,4,5 on the bottom with both 2 and 6 playable; O to move loses.
    state = c4_after(3, 0, 4, 0, 5)
    assert solve_state(state) == -1


def test_connect4_solver_agrees_with_negamax_on_shallow_endgames():
    # Play deterministic near-full games and check WDL consistency: the value
    # of a position equals the best of the negated child values.
    rng = random.Random(0)
    for _ in range(5):
        state = Connect4State.new_game(seed=0)
        while state.ply < 30 and not state.is_terminal:
            state.apply(rng.choice(state.legal_actions()))
        if state.is_terminal:
            continue
        value = solve_state(state)
        children = []
        for action in state.legal_actions():
            child = state.clone()
            child.apply(action)
            if child.is_terminal:
                out = child.outcome()
                mover_view = out if state.current_player == 0 else -out
                children.append(int(mover_view))
            else:
                children.append(-solve_state(child))
        assert value == max(children)


# ---------------------------------------------------------------------- suite
def test_a_tictactoe_suite_is_reachable_solved_and_stratified():
    suite = build_suite("tictactoe", n_positions=40, seed=1)
    assert suite["game"] == "tictactoe"
    entries = suite["positions"]
    assert len(entries) >= 30
    plies = {e["ply"] for e in entries}
    assert len(plies) >= 4  # not all from the opening
    for e in entries:
        state = rebuild_state("tictactoe", e["key"])
        legal = state.legal_actions()
        assert set(map(int, e["values"])) == set(legal)
        best = max(e["values"].values())
        assert e["value"] == best
        assert sorted(e["optimal"]) == sorted(
            int(a) for a, v in e["values"].items() if v == best
        )


def test_a_connect4_suite_solves_from_the_configured_ply():
    suite = build_suite("connect4", n_positions=12, seed=2, min_ply=16)
    for e in suite["positions"]:
        assert e["ply"] >= 16
        assert e["value"] in (-1, 0, 1)


# ------------------------------------------------------------------ % optimal
class _OracleAgent:
    """Plays a value-preserving move by construction."""

    name = "oracle"

    def seed(self, n: int) -> None:
        pass

    def act(self, state) -> int:
        entry_best = max(
            (v, int(a))
            for a, v in (
                (a, -solve_state(_apply(state, a)) if not _apply(state, a).is_terminal
                 else _terminal_mover_value(state, a))
                for a in state.legal_actions()
            )
        )
        return entry_best[1]


def _apply(state, action):
    child = state.clone()
    child.apply(action)
    return child


def _terminal_mover_value(state, action):
    child = _apply(state, action)
    out = child.outcome()
    return int(out if state.current_player == 0 else -out)


def test_the_oracle_scores_100_percent_and_random_does_not():
    suite = build_suite("tictactoe", n_positions=60, seed=3)
    perfect = score_agent(_OracleAgent(), suite)
    assert perfect["pct_optimal"] == 1.0
    assert perfect["blunder_rate"] == 0.0

    from ludometer.agents import make_agent

    rand = make_agent("ttt:random")
    rand.seed(0)
    sloppy = score_agent(rand, suite)
    assert sloppy["n"] == perfect["n"]
    assert sloppy["pct_optimal"] < 1.0
    assert sloppy["blunder_rate"] > 0.0


# --------------------------------------------------------------- perfect play
def test_the_perfect_tictactoe_agent_never_loses():
    from ludometer.agents import make_agent

    perfect = make_agent("ttt:perfect")
    perfect.seed(0)
    for spec in ("ttt:random", "ttt:greedy", "ttt:perfect"):
        rival = make_agent(spec)
        rival.seed(1)
        for seed in range(6):
            for perfect_seat in (0, 1):
                state = TicTacToeState.new_game(seed=seed)
                agents = (perfect, rival) if perfect_seat == 0 else (rival, perfect)
                for a in agents:
                    a.seed(seed * 2 + perfect_seat)
                while not state.is_terminal:
                    state.apply(agents[state.current_player].act(state))
                outcome = state.outcome()
                lost = outcome == (-1.0 if perfect_seat == 0 else 1.0)
                assert not lost, f"perfect lost to {spec} (seed {seed})"
                if spec == "ttt:perfect":
                    assert outcome == 0.0  # perfect vs perfect always draws


def test_the_perfect_connect4_agent_converts_a_won_midgame():
    from ludometer.agents import make_agent

    perfect = make_agent("c4:perfect")
    perfect.seed(0)
    rival = make_agent("c4:heuristic")
    rival.seed(2)
    # a deep-enough midgame that the WDL solve is cheap (ply 14+)
    state = Connect4State.new_game(seed=0)
    roll = random.Random(7)
    while state.ply < 14 and not state.is_terminal:
        state.apply(roll.choice(state.legal_actions()))
    if state.is_terminal:
        pytest.skip("rollout ended early")
    value = solve_state(state)
    mover = state.current_player
    agents = {mover: perfect, 1 - mover: rival}
    while not state.is_terminal:
        state.apply(agents[state.current_player].act(state))
    outcome = state.outcome()
    if value == 1:
        assert outcome == (1.0 if mover == 0 else -1.0)
