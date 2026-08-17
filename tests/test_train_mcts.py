"""Tests for the PUCT search (see docs/DESIGN.md, "Training").

The load-bearing claims are: the search only ever returns legal moves, it is a
pure function of its seed, round-boundary chance edges are re-sampled (and
bounded), and — the real sanity check — search on top of a *net-free* evaluator
already crushes the random baseline. If the value backup had a sign error or the
chance handling let the search peek at the bag order, that last test is what
would catch it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ludometer.agents import make_agent
from ludometer.azul.engine import ACTION_SPACE, AzulState, encode_action
from ludometer.eval.arena import play_match
from ludometer.train.mcts import (
    MCTS,
    MCTSConfig,
    RolloutEvaluator,
    UniformEvaluator,
    select_action,
)
from ludometer.train.mcts_agent import MCTSAgent
from ludometer.train.net import NetConfig, NetEvaluator, PolicyValueNet


class RolloutAgentSpec:
    """Picklable-ish spec (used single-process): MCTS with random playouts."""

    name = "mcts-rollout"

    def __init__(self, sims: int = 32) -> None:
        self.sims = sims

    def __call__(self) -> MCTSAgent:
        return MCTSAgent(
            evaluator=RolloutEvaluator(seed=1),
            config=MCTSConfig(sims=self.sims),
            name=self.name,
        )


class UntrainedNetSpec:
    """MCTS on a freshly initialised (uniform-ish) net — no learning at all."""

    name = "mcts-untrained"

    def __init__(self, sims: int = 32) -> None:
        self.sims = sims

    def __call__(self) -> MCTSAgent:
        torch.manual_seed(0)
        torch.set_num_threads(1)
        net = PolicyValueNet(NetConfig(hidden=64, blocks=2, value_hidden=32))
        return MCTSAgent(net, sims=self.sims, seed=4, name=self.name)


def near_round_end_state() -> AzulState:
    """One tile left on the board, so the next move triggers the refill."""
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


def test_search_returns_a_distribution_over_legal_actions() -> None:
    state = AzulState.new_game(seed=1)
    mcts = MCTS(UniformEvaluator(), MCTSConfig(sims=48), seed=2)
    result = mcts.search(state)
    legal = np.asarray(state.legal_actions())
    assert result.policy.shape == (ACTION_SPACE,)
    assert result.policy.sum() == pytest.approx(1.0, abs=1e-5)
    assert result.policy[legal].sum() == pytest.approx(1.0, abs=1e-5)
    illegal = np.setdiff1d(np.arange(ACTION_SPACE), legal)
    assert result.policy[illegal].sum() == 0.0
    assert result.sims == 48
    assert sum(result.visits.values()) == 48
    assert -1.0 <= result.value <= 1.0


def test_search_does_not_mutate_the_state() -> None:
    state = AzulState.new_game(seed=9)
    before = state.render_text()
    MCTS(UniformEvaluator(), MCTSConfig(sims=32), seed=1).search(state)
    assert state.render_text() == before


def test_search_is_deterministic_given_the_seed() -> None:
    state = AzulState.new_game(seed=4)
    net = PolicyValueNet(NetConfig(hidden=32, blocks=1, value_hidden=16))
    evaluator = NetEvaluator(net)
    a = MCTS(evaluator, MCTSConfig(sims=40), seed=7, add_noise=True).search(state)
    b = MCTS(evaluator, MCTSConfig(sims=40), seed=7, add_noise=True).search(state)
    c = MCTS(evaluator, MCTSConfig(sims=40), seed=8, add_noise=True).search(state)
    np.testing.assert_array_equal(a.policy, b.policy)
    assert not np.array_equal(a.policy, c.policy)


def test_dirichlet_noise_only_applies_at_the_root() -> None:
    state = AzulState.new_game(seed=6)
    mcts = MCTS(UniformEvaluator(), MCTSConfig(sims=8), seed=3)
    root = mcts._new_node(state.clone())
    mcts._expand(root)
    flat = list(root.priors)
    mcts._apply_noise(root)
    assert sum(root.priors) == pytest.approx(1.0, abs=1e-6)
    assert root.priors != flat  # noise moved the root priors
    child = mcts._child(root, 0)
    mcts._expand(child)
    assert child.priors == pytest.approx([1.0 / len(child.legal)] * len(child.legal))


def test_round_boundary_edges_are_resampled_and_capped() -> None:
    state = near_round_end_state()
    mcts = MCTS(UniformEvaluator(), MCTSConfig(sims=4, chance_children=3), seed=1)
    root = mcts._new_node(state.clone())
    mcts._expand(root)
    action = encode_action(0, 0, 0)
    index = root.legal.index(action)
    assert mcts._is_stochastic(root.state, action)

    children = [mcts._child(root, index) for _ in range(20)]
    table = root.children[index]
    assert isinstance(table, dict)
    assert 1 < len(table) <= 3  # several refills sampled, but bounded
    assert all(child.state.round_index == state.round_index + 1 for child in children)
    # every determinization is a genuine deal: 20 tiles back on the board
    assert {child.state.tiles_left for child in children} == {20}
    # ... and they differ, i.e. we are not replaying one pre-shuffled bag
    assert len({mcts._chance_key(child.state) for child in children}) > 1


def test_a_chance_edge_q_is_the_visit_weighted_mean_of_its_determinizations() -> None:
    """``chance_backup = "mean"`` is a *description of existing behaviour*.

    There is no chance-node object and no averaging step anywhere in the search:
    the edge counters ``(N, W)`` live in the parent, and every simulation through
    a stochastic edge — whichever determinization it happens to land in — adds to
    that same pair. The claim run6 writes into its config is that this makes

        Q(edge) = sum_d N_d * Q_d / sum_d N_d

    over the sampled determinizations ``d``. So rather than change any code, watch
    the partition happen: run the simulations one at a time, record which
    determinization each one went through and how much it moved the edge, and
    check that the per-determinization pieces (a) account for *every* visit and
    *all* of the value the edge holds, and (b) recombine into the edge's Q as
    that weighted mean.
    """
    from collections import defaultdict

    state = near_round_end_state()
    mcts = MCTS(
        RolloutEvaluator(seed=5), MCTSConfig(sims=200, chance_children=4), seed=7
    )
    root = mcts._new_node(state.clone())
    mcts._expand(root)
    index = root.legal.index(encode_action(0, 0, 0))
    assert mcts._is_stochastic(root.state, root.legal[index])

    took: list = []
    plain_child = mcts._child

    def spy(node, i):  # which determinization did this simulation take?
        child = plain_child(node, i)
        if node is root and i == index:
            took.append(child)
        return child

    mcts._child = spy  # type: ignore[method-assign]
    per: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])  # [visits, value]
    for _ in range(200):
        before_n, before_w = root.visits[index], root.wins[index]
        took.clear()
        mcts._simulate(root)
        if root.visits[index] == before_n:
            continue  # this simulation went down a different root edge
        entry = per[id(took[0])]
        entry[0] += 1
        entry[1] += root.wins[index] - before_w

    assert len(per) > 1, "the edge only ever sampled one refill"
    # (a) the pieces account for the whole edge — this is the real assertion,
    # because it is what "the edge is the sum of its determinizations" means.
    assert sum(n for n, _ in per.values()) == root.visits[index]
    assert sum(w for _, w in per.values()) == pytest.approx(root.wins[index], abs=1e-9)
    # (b) ... and therefore Q is their visit-weighted mean, spelled out.
    q_per_determinization = {d: w / n for d, (n, w) in per.items()}
    total_n = sum(n for n, _ in per.values())
    weighted_mean = (
        sum(per[d][0] * q for d, q in q_per_determinization.items()) / total_n
    )
    assert root.wins[index] / root.visits[index] == pytest.approx(
        weighted_mean, abs=1e-9
    )


def test_chance_backup_only_understands_the_mean() -> None:
    """A config asking for a rule that does not exist must say so, not be ignored."""
    assert MCTSConfig.from_dict({"chance_backup": "mean"}).chance_backup == "mean"
    with pytest.raises(ValueError, match="unknown chance_backup"):
        MCTSConfig.from_dict({"chance_backup": "max"})


@pytest.mark.parametrize("children", [4, 8])
def test_more_determinizations_widen_the_sample_without_changing_the_rule(
    children: int,
) -> None:
    """run6 raises ``chance_children`` to 8; nothing else about the edge moves."""
    state = near_round_end_state()
    mcts = MCTS(
        UniformEvaluator(), MCTSConfig(sims=4, chance_children=children), seed=1
    )
    root = mcts._new_node(state.clone())
    mcts._expand(root)
    index = root.legal.index(encode_action(0, 0, 0))
    for _ in range(60):
        mcts._child(root, index)
    table = root.children[index]
    assert isinstance(table, dict)
    assert 1 < len(table) <= children
    # 8 really does sample more refills than 4 (the deal has far more outcomes)
    if children == 8:
        assert len(table) > 4
    assert {child.state.tiles_left for child in table.values()} == {20}


def test_within_round_edges_are_deterministic_and_shared() -> None:
    state = AzulState.new_game(seed=2)
    mcts = MCTS(UniformEvaluator(), MCTSConfig(sims=4), seed=1)
    root = mcts._new_node(state.clone())
    mcts._expand(root)
    assert not mcts._is_stochastic(root.state, root.legal[0])
    first = mcts._child(root, 0)
    assert mcts._child(root, 0) is first  # one child, reused


def test_terminal_outcomes_are_backed_up_with_the_right_sign() -> None:
    """A finished game must be scored for the player who actually won."""
    state = AzulState.new_game(seed=3)
    rng = np.random.default_rng(0)
    while not state.is_terminal:
        legal = state.legal_actions()
        state.apply(legal[int(rng.integers(len(legal)))])
    mcts = MCTS(UniformEvaluator(), MCTSConfig(sims=4), seed=1)
    node = mcts._new_node(state)
    assert node.terminal_v0 == state.outcome()
    with pytest.raises(ValueError):
        mcts.search(state)


@pytest.mark.parametrize("sims", [16, 32])
def test_search_beats_random(sims: int) -> None:
    """Pure-search sanity: no trained weights at all, 20 games vs RandomAgent.

    The evaluator is a uniform prior plus a single random playout, so anything
    the agent gets right comes from the tree: PUCT selection, the per-player sign
    of the value backup and the chance-node re-sampling. A sign error here shows
    up immediately as a win rate below 0.5.
    """
    match = play_match(RolloutAgentSpec(sims), "random", n_games=20, base_seed=11)
    assert match.win_rate >= 0.75, f"sims={sims}: {match.win_rate}"
    assert match.mean_score_a > match.mean_score_b


def test_untrained_net_agent_plays_legal_moves() -> None:
    """An untrained value head is arbitrary, so only legality is asserted here.

    (Strength of a *random* net is a coin flip: what the pure-search test above
    measures is the search, not the weights.)
    """
    agent = make_agent(UntrainedNetSpec(8))
    state = AzulState.new_game(seed=12)
    agent.seed(3)
    for _ in range(8):
        action = agent.act(state)
        assert state.is_legal(action)
        state.apply(action)
        if state.is_terminal:  # pragma: no cover - not in 8 moves
            break


def test_a_stalled_game_makes_the_agent_sample_again() -> None:
    """Past ``stall_rounds`` the agent randomises, which breaks arg-max loops."""
    state = AzulState.new_game(seed=21)
    steady = MCTSAgent(
        evaluator=UniformEvaluator(),
        config=MCTSConfig(sims=12),
        stall_rounds=99,
        seed=1,
    )
    assert len({steady.act(state) for _ in range(12)}) == 1
    stalled = MCTSAgent(
        evaluator=UniformEvaluator(),
        config=MCTSConfig(sims=12),
        stall_rounds=0,  # "this game has gone on far too long"
        seed=1,
    )
    assert len({stalled.act(state) for _ in range(12)}) > 1


def test_select_action_temperature() -> None:
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[7] = 0.7
    policy[9] = 0.3
    assert select_action(policy, 0.0) == 7
    import random

    rng = random.Random(0)
    draws = [select_action(policy, 1.0, rng) for _ in range(400)]
    share = draws.count(9) / len(draws)
    assert set(draws) == {7, 9}
    assert 0.2 < share < 0.4
    # a low temperature concentrates on the argmax
    rng = random.Random(0)
    hot = [select_action(policy, 0.1, rng) for _ in range(50)]
    assert hot.count(7) >= 45
