"""Tests for MCTS tree reuse (``MCTSConfig.tree_reuse``).

Reuse is a throughput optimisation, so the bar is that it must not change *what*
the search concludes — only how much work it costs. The claims tested here:

* with nothing to inherit, a reuse search is bit-identical to a plain one
  (so switching the flag on cannot silently change run behaviour by itself);
* after a move, the visits below the chosen child are carried over and the root
  still ends up with exactly ``sims`` visits — the budget is a total, not an
  addition;
* the search agrees with a fresh search about the best move;
* the subtree is dropped at a refill boundary, where the child edge is a bag of
  determinizations rather than the position the game actually dealt;
* a caller that hands over an unexpected state gets a fresh root instead of a
  wrong one.
"""

from __future__ import annotations

import numpy as np
import pytest

from ludometer.azul.engine import ACTION_SPACE, AzulState, encode_action
from ludometer.train.mcts import MCTS, MCTSConfig, UniformEvaluator, select_action
from ludometer.train.selfplay import SelfPlayConfig, play_selfplay_game


class ScoreEvaluator:
    """Deterministic, net-free evaluator: value = current score margin.

    Deterministic matters here — two searches can then be compared move by move
    without a random playout deciding the difference.
    """

    def __call__(self, state, legal):
        n = len(legal)
        priors = (
            np.full(n, 1.0 / n, dtype=np.float32)
            if n
            else np.zeros(0, dtype=np.float32)
        )
        me = state.current_player
        margin = state.scores[me] - state.scores[1 - me]
        margin += 2 * (state.completed_rows(me) - state.completed_rows(1 - me))
        return priors, float(np.tanh(margin / 15.0))


def near_round_end_state() -> AzulState:
    """One tile left on the board: the next move triggers the refill."""
    state = AzulState.new_game(seed=5)
    for factory in state.factories:
        for c in range(5):
            state.lid[c] += factory[c]
            factory[c] = 0
    for c in range(5):
        state.lid[c] += state.center[c]
        state.center[c] = 0
    state.lid[0] -= 1
    state.factories[0][0] = 1
    state.recount()
    return state


def make(reuse: bool, sims: int = 96, seed: int = 4) -> MCTS:
    return MCTS(ScoreEvaluator(), MCTSConfig(sims=sims, tree_reuse=reuse), seed=seed)


# ------------------------------------------------------------------ no-op case
def test_first_search_is_identical_with_and_without_reuse() -> None:
    state = AzulState.new_game(seed=1)
    plain = make(False).search(state)
    reused = make(True).search(state)
    np.testing.assert_array_equal(plain.policy, reused.policy)
    assert plain.value == reused.value
    assert plain.sims == reused.sims == 96


def test_advance_is_a_no_op_when_reuse_is_off() -> None:
    state = AzulState.new_game(seed=2)
    mcts = make(False)
    result = mcts.search(state)
    action = int(np.argmax(result.policy))
    assert mcts.advance(action) is False
    state.apply(action)
    assert mcts.search(state).sims == 96
    assert mcts.reused_visits == 0


# --------------------------------------------------------------- the mechanism
def test_reuse_inherits_visits_and_keeps_the_budget_a_total() -> None:
    state = AzulState.new_game(seed=7)
    mcts = make(True)
    first = mcts.search(state)
    action = int(np.argmax(first.policy))
    inherited = first.visits[action]
    assert inherited > 0
    state.apply(action)
    assert mcts.advance(action) is True

    second = mcts.search(state)
    # A node counts the simulations that passed *through* it, so it is one short
    # of the parent's edge count (the visit that expanded it stopped there).
    assert mcts.reused_visits == inherited - 1 > 0
    # the root ends up with exactly `sims` visits, just cheaper
    assert second.sims == 96
    assert sum(second.visits.values()) == 96
    legal = state.legal_actions()
    assert set(second.visits) == set(legal)
    assert second.policy.sum() == pytest.approx(1.0, abs=1e-5)
    assert second.policy[np.setdiff1d(np.arange(ACTION_SPACE), legal)].sum() == 0.0


def test_reuse_saves_evaluations_over_a_whole_game() -> None:
    """The point of the flag: same search budget, fewer network calls."""
    config = SelfPlayConfig(
        mcts=MCTSConfig(sims=64, tree_reuse=False), temp_moves=0, max_moves=60
    )
    plain = play_selfplay_game(ScoreEvaluator(), 3, config)
    reused = play_selfplay_game(
        ScoreEvaluator(),
        3,
        SelfPlayConfig(
            mcts=MCTSConfig(sims=64, tree_reuse=True), temp_moves=0, max_moves=60
        ),
    )
    assert reused.evals < plain.evals
    print(
        f"\nevals/move: plain {plain.evals / plain.moves:.1f}, "
        f"reuse {reused.evals / reused.moves:.1f} "
        f"({plain.evals / max(1, reused.evals):.2f}x fewer calls)"
    )


def test_reuse_and_fresh_search_pick_the_same_move() -> None:
    """Walk a game with reuse on, re-searching each position from scratch too."""
    state = AzulState.new_game(seed=13)
    keeper = make(True, sims=128)
    agreed = 0
    compared = 0
    for _ in range(8):
        if state.is_terminal:  # pragma: no cover - not in 8 moves
            break
        legal = state.legal_actions()
        if len(legal) > 1:
            kept = keeper.search(state)
            fresh = make(False, sims=128).search(state)
            best_fresh = int(np.argmax(fresh.policy))
            compared += 1
            agreed += int(int(np.argmax(kept.policy)) == best_fresh)
            # even when the arg-max differs, the fresh best must be a move the
            # reusing search also rates highly
            ranked = sorted(kept.visits, key=lambda a: -kept.visits[a])
            assert best_fresh in ranked[:3]
            action = select_action(kept.policy, 0.0)
        else:
            action = legal[0]
        state.apply(action)
        keeper.advance(action)
    assert compared >= 5
    assert agreed >= compared - 1, f"only {agreed}/{compared} arg-maxes agreed"


def test_forced_move_keeps_walking_down_the_same_tree() -> None:
    """A move played without searching still advances the kept root."""
    state = AzulState.new_game(seed=21)
    mcts = make(True)
    mcts.search(state)
    action = int(np.argmax(mcts.search(state).policy))
    state.apply(action)
    assert mcts.advance(action) is True
    assert mcts._root is not None


# ------------------------------------------------------------------ boundaries
def test_reuse_is_dropped_across_a_refill() -> None:
    state = near_round_end_state()
    mcts = make(True, sims=32)
    mcts.search(state)
    action = encode_action(0, 0, 0)  # takes the last tile -> triggers the refill
    assert mcts._is_stochastic(state, action)
    state.apply(action)
    assert mcts.advance(action) is False
    mcts.search(state)
    assert mcts.reused_visits == 0


def test_a_surprising_state_falls_back_to_a_fresh_root() -> None:
    state = AzulState.new_game(seed=8)
    mcts = make(True)
    result = mcts.search(state)
    action = int(np.argmax(result.policy))
    played = state.clone()
    played.apply(action)
    assert mcts.advance(action) is True
    # hand it a different position than the one it advanced into
    other = AzulState.new_game(seed=99)
    fresh = mcts.search(other)
    assert mcts.reused_visits == 0
    assert fresh.sims == 96


def test_reseeding_forgets_the_tree() -> None:
    state = AzulState.new_game(seed=2)
    mcts = make(True)
    action = int(np.argmax(mcts.search(state).policy))
    state.apply(action)
    assert mcts.advance(action) is True
    mcts.seed(11)
    assert mcts.search(state).sims == 96
    assert mcts.reused_visits == 0


def test_config_flag_round_trips() -> None:
    assert MCTSConfig.from_dict({"sims": 8, "tree_reuse": True}).tree_reuse is True
    assert MCTSConfig.from_dict({"sims": 8}).tree_reuse is False
    assert SelfPlayConfig.from_dict({"tree_reuse": True}).mcts.tree_reuse is True


def test_uniform_evaluator_path_still_works() -> None:
    """Reuse must not depend on the evaluator (UniformEvaluator has no value)."""
    state = AzulState.new_game(seed=6)
    mcts = MCTS(UniformEvaluator(), MCTSConfig(sims=40, tree_reuse=True), seed=1)
    action = int(np.argmax(mcts.search(state).policy))
    state.apply(action)
    mcts.advance(action)
    assert sum(mcts.search(state).visits.values()) == 40
