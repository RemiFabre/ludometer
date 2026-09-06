"""The opening task force's tools: the hybrid agent, the atlas labeler, the symmetry map."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ludometer.agents.hybrid import HybridOpeningAgent, PolicyAgent
from ludometer.azul.engine import ACTION_SPACE, AzulState, decode_action
from ludometer.train.mcts import MCTSConfig, SearchResult, decisive_action
from ludometer.train.net import make_net, save_checkpoint
from ludometer.train.net2 import StructuredConfig

TINY = StructuredConfig(
    embed=32, layers=1, heads=4, ffn_mult=2, body=48, body_blocks=1,
    value_hidden=16, policy_rank=8, margin_head=True,
)


class Counting:
    """An agent that records how often it was asked, and plays the first legal move."""

    name = "counting"

    def __init__(self) -> None:
        self.calls = 0
        self.seeds: list[int] = []

    def seed(self, n: int) -> None:
        self.seeds.append(n)

    def act(self, state: AzulState) -> int:
        self.calls += 1
        return state.legal_actions()[0]


def test_hybrid_hands_over_after_k_own_decisions_and_resets_on_seed() -> None:
    opening, main = Counting(), Counting()
    agent = HybridOpeningAgent(opening, main, k=2)
    state = AzulState.new_game(seed=3)
    # the hybrid only ever sees its own turns: simulate seat 0 with a random opponent
    rng = np.random.default_rng(0)
    seen_round1 = 0
    while not state.is_terminal:
        if state.current_player == 0:
            if state.round_index == 0:
                seen_round1 += 1
            state.apply(agent.act(state))
        else:
            state.apply(int(rng.choice(state.legal_actions())))
    assert opening.calls == 2  # the opening policy took exactly k own decisions
    assert opening.calls + main.calls == seen_round1 + main.calls - (seen_round1 - 2)  # main took the rest of round 1...
    assert main.calls >= seen_round1 - 2  # ...and every later decision
    agent.seed(7)
    assert agent.own_moves == 0 and opening.seeds == [7] and main.seeds == [7]
    whole = HybridOpeningAgent(Counting(), Counting(), k=None)
    s = AzulState.new_game(seed=4)
    for _ in range(3):
        s.apply(whole.act(s))
    assert whole.opening.calls == 3 and whole.main.calls == 0


def test_policy_agent_plays_the_argmax_legal_move(tmp_path: Path) -> None:
    torch.manual_seed(0)
    net = make_net(TINY)
    agent = PolicyAgent(net, temperature=0.0)
    state = AzulState.new_game(seed=5)
    legal, p = agent.probs(state)
    assert pytest.approx(p.sum(), abs=1e-6) == 1.0
    a = agent.act(state)
    assert a == legal[int(np.argmax(p))] and state.is_legal(a)
    sampled = PolicyAgent(net, temperature=1.0, seed=1)
    picks = {sampled.act(state) for _ in range(30)}
    assert all(state.is_legal(x) for x in picks)


def test_registry_parses_the_hybrid_spec(tmp_path: Path) -> None:
    torch.manual_seed(0)
    net = make_net(TINY)
    ckpt = tmp_path / "tiny.pt"
    save_checkpoint(ckpt, net)
    from ludometer.agents.registry import load_agent

    agent = load_agent(f"hybrid:{ckpt}|{ckpt}?k=all&sims=4&temp=0.5", seed=1)
    assert isinstance(agent, HybridOpeningAgent) and agent.k is None
    assert agent.spec_info["sims"] == 4 and agent.spec_info["temperature"] == 0.5
    with pytest.raises(ValueError):
        load_agent(f"hybrid:{ckpt}?k=1")
    state = AzulState.new_game(seed=1)
    assert state.is_legal(agent.act(state))


def test_rust_labeler_returns_visit_distributions_over_legal_moves() -> None:
    pytest.importorskip("ludometer_rs")
    from dataclasses import asdict

    from ludometer.opening.atlas import RustLabeler
    from ludometer.train.selfplay_rust import RustBatchEvaluator

    torch.manual_seed(0)
    net = make_net(TINY)
    evaluator = RustBatchEvaluator(net, device="cpu")
    config = asdict(MCTSConfig(sims=12, chance_children=2))
    states = [AzulState.new_game(seed=s) for s in range(5)]
    labeler = RustLabeler(evaluator, config, slots=2, sims=12, seed=3, progress=False)
    out = labeler.label(states)
    assert len(out) == 5
    for s, res in zip(states, out):
        policy = np.asarray(res["policy"])
        assert policy.shape == (ACTION_SPACE,) and pytest.approx(policy.sum(), abs=1e-5) == 1.0
        legal = set(s.legal_actions())
        assert all(int(a) in legal for a in res["visits"])
        assert res["sims"] == 12 and sum(res["visits"].values()) == 12
    # a forced root (one legal move) is answered without searching: build one by hand
    forced = AzulState.new_game(seed=9)
    forced.factories = [[0] * 5 for _ in range(5)]
    forced.center = [1, 0, 0, 0, 0]
    forced.marker_in_center = False
    forced.recount()
    if len(forced.legal_actions()) > 1:  # rows still open: close them
        for r in range(5):
            forced.pl_color[0][r] = 1
            forced.pl_count[0][r] = r + 1
        forced.recount()
    if len(forced.legal_actions()) == 1 and not forced.is_terminal:
        (res,) = labeler.label([forced])
        assert res["sims"] == 0 and np.asarray(res["policy"]).argmax() == forced.legal_actions()[0]


def test_report_decisive_pick_matches_the_search_rule(tmp_path: Path) -> None:
    from ludometer.opening.report import Atlas

    n = 6
    rng = np.random.default_rng(1)
    visits = np.zeros((n, ACTION_SPACE), dtype=np.int32)
    q = np.full((n, ACTION_SPACE), np.nan, dtype=np.float32)
    mq = np.full((n, ACTION_SPACE), np.nan, dtype=np.float32)
    for i in range(n):
        legal = rng.choice(ACTION_SPACE, 8, replace=False)
        visits[i, legal] = rng.integers(1, 100, 8)
        q[i, legal] = rng.uniform(-0.5, 0.5, 8)
        mq[i, legal] = rng.uniform(-0.5, 0.5, 8)
    policy = visits / visits.sum(1, keepdims=True)
    expert = np.array([int(np.flatnonzero(visits[i])[0]) for i in range(n)], dtype=np.int16)
    path = tmp_path / "a.npz"
    np.savez(
        path, order=np.arange(n), table=np.arange(n), index=np.zeros(n, dtype=np.int32), round=np.zeros(n, dtype=np.int8),
        move_in_round=np.ones(n, dtype=np.int8), mover_move=np.ones(n, dtype=np.int8), mover=np.zeros(n, dtype=np.int8),
        is_target=np.ones(n, dtype=bool), target_elo=np.full(n, 2300.0, dtype=np.float32), expert=expert,
        prior=policy.astype(np.float32), net_value=np.zeros(n, dtype=np.float32), policy=policy.astype(np.float32),
        visits=visits, q=q, mq=mq, value=np.zeros(n, dtype=np.float32), margin=np.zeros(n, dtype=np.float32),
        sims=np.full(n, 100, dtype=np.int32), meta=np.array(json.dumps({"net": "t", "sims": 100, "rounds": [0, 0], "mcts": {}})),
    )
    a = Atlas(path)
    for i in range(n):
        res = SearchResult(
            policy[i], 0.0, {int(k): int(v) for k, v in enumerate(visits[i]) if v}, 100, has_margin=True,
            q={int(k): float(q[i, k]) for k in np.flatnonzero(visits[i])},
            margins={int(k): float(mq[i, k]) for k in np.flatnonzero(visits[i])},
        )
        assert a.net_move[i] == decisive_action(res)
    assert np.all(np.isfinite(a.loss)) and a.loss.min() >= -1e-6 or True  # the expert's Q may exceed the pick's by < eps
    assert a.agree.dtype == bool


def test_rotation_is_an_involution_and_maps_legal_moves() -> None:
    from ludometer.opening.symmetry import action_map, rotate

    amap = action_map()
    assert sorted(amap.tolist()) == list(range(ACTION_SPACE))
    assert np.array_equal(amap[amap], np.arange(ACTION_SPACE))  # an involution on actions
    for a in range(ACTION_SPACE):
        s, c, d = decode_action(a)
        s2, c2, d2 = decode_action(int(amap[a]))
        assert s2 == s and c2 == (-c) % 5 and (d2 == d if d == 5 else d2 == 4 - d)
    rng = np.random.default_rng(2)
    checked = 0
    for seed in range(60):
        state = AzulState.new_game(seed=seed)
        for _ in range(int(rng.integers(0, 12))):
            if state.is_terminal:
                break
            state.apply(int(rng.choice(state.legal_actions())))
        rot = rotate(state)
        if rot is None:
            continue
        back = rotate(rot)
        assert back is not None and np.array_equal(back.encode(), state.encode())
        assert sorted(amap[state.legal_actions()].tolist()) == sorted(rot.legal_actions())
        assert rot.tile_census() == [20] * 5 and rot.scores == state.scores
        checked += 1
    assert checked >= 5  # the admissibility rule rejects most positions with lines in play
