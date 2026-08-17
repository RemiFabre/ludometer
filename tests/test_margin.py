"""Tests for run4's margin head and the decisive move it buys.

The complaint this answers: once a game is decided, a win/draw/loss value head is
indifferent between +1 and +40, so the AI plays whatever winning move it visited
most and looks broken. run4 adds a third head predicting
``tanh(score_diff / 20)`` and breaks ties on it.

Four things have to hold, and each has its own section below:

1. **the head exists and survives a round trip** — forward, checkpoint, ONNX;
2. **nothing without the head changes**, at all: a run3 checkpoint must load and
   play exactly as before, and its ONNX graph must keep its two outputs;
3. **the tie-break is lexicographic**: winning first, margin only among equals,
   and *only* at move-selection time — the visit counts (i.e. the policy targets)
   must be bit-identical to what a margin-blind search would have produced;
4. **old replay buffers still pretrain a new net**, with the margin loss masked
   out on the positions that have no margin to learn from.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
import torch

from ludometer.azul.engine import ACTION_SPACE, ENCODED_SIZE, AzulState
from ludometer.train.mcts import (
    MCTS,
    MCTSConfig,
    SearchResult,
    UniformEvaluator,
    decisive_action,
    margin_target,
    select_action,
    select_play_action,
)
from ludometer.train.mcts_agent import MCTSAgent
from ludometer.train.net import NetEvaluator, load_net, make_net, save_checkpoint
from ludometer.train.net2 import MARGIN_VERSION, StructuredConfig, StructuredNet
from ludometer.train.replay import ReplayBuffer, unblend_values
from ludometer.train.trainer import TrainConfig, Trainer

REPO = Path(__file__).resolve().parents[1]

# Small enough to build in milliseconds; the head is two Linears, not a mystery.
PLAIN = StructuredConfig(
    embed=32, layers=1, heads=4, body=48, body_blocks=1, value_hidden=16, policy_rank=8
)
WITH_MARGIN = StructuredConfig(**{**PLAIN.to_dict(), "margin_head": True})


def a_mid_game_state(seed: int = 21, moves: int = 30) -> AzulState:
    """A position deep enough to have real score differences on the board."""
    rng = np.random.default_rng(seed)
    state = AzulState.new_game(seed=seed)
    for _ in range(moves):
        legal = state.legal_actions()
        if state.is_terminal or not legal:  # pragma: no cover - defensive
            break
        state.apply(legal[int(rng.integers(len(legal)))])
    return state


# --------------------------------------------------------------- 1. the head
def test_config_and_version_track_each_other() -> None:
    assert PLAIN.margin_head is False
    assert PLAIN.version == 1
    assert WITH_MARGIN.version == MARGIN_VERSION == 2
    # a checkpoint that only recorded the version still rebuilds the right net
    from_version = StructuredConfig.from_dict({"arch": "structured", "version": 2})
    assert from_version.margin_head is True
    # ... and one that only recorded the flag reports the right version
    from_flag = StructuredConfig.from_dict({"arch": "structured", "margin_head": True})
    assert from_flag.version == MARGIN_VERSION
    # a run3 net_config (neither key) is unambiguously the two-head net
    assert StructuredConfig.from_dict({"arch": "structured"}).margin_head is False


def test_forward_gives_three_bounded_heads() -> None:
    torch.manual_seed(0)
    net = StructuredNet(WITH_MARGIN).eval()
    assert net.has_margin
    x = torch.randn(4, ENCODED_SIZE) * 50.0
    with torch.no_grad():
        logits, value, margin = net.forward_heads(x)
        two = net(x)
    assert logits.shape == (4, ACTION_SPACE)
    assert value.shape == margin.shape == (4,)
    assert float(margin.abs().max()) <= 1.0  # it is a tanh
    # forward() is still the old two-output contract, and answers the same thing
    assert len(two) == 2
    torch.testing.assert_close(two[0], logits)
    torch.testing.assert_close(two[1], value)


def test_a_net_without_the_head_says_so_everywhere() -> None:
    net = StructuredNet(PLAIN).eval()
    assert net.has_margin is False
    assert not [k for k in net.state_dict() if "margin" in k]
    with torch.no_grad():
        logits, value, margin = net.forward_heads(torch.zeros(2, ENCODED_SIZE))
    assert margin is None
    assert logits.shape == (2, ACTION_SPACE) and value.shape == (2,)


def test_margin_head_learns_a_score_gap() -> None:
    """The head has to be able to fit the thing it is for."""
    torch.manual_seed(3)
    net = StructuredNet(WITH_MARGIN)
    x = torch.randn(24, ENCODED_SIZE)
    target_m = torch.tanh(x[:, 0])  # some smooth function of the position
    target_v = torch.sign(x[:, 0])
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    for _ in range(250):
        _, value, margin = net.forward_heads(x)
        loss = torch.nn.functional.mse_loss(margin, target_m)
        loss = loss + torch.nn.functional.mse_loss(value, target_v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        _, _, margin = net.forward_heads(x)
    assert float(torch.nn.functional.mse_loss(margin, target_m)) < 0.05


def test_checkpoint_roundtrip_keeps_the_margin(tmp_path) -> None:
    torch.manual_seed(1)
    net = StructuredNet(WITH_MARGIN).eval()
    path = save_checkpoint(tmp_path / "ckpt.pt", net, {"games": 7})
    restored, payload = load_net(path)
    assert isinstance(restored, StructuredNet)
    assert restored.has_margin
    assert payload["net_config"]["margin_head"] is True
    assert payload["net_config"]["version"] == MARGIN_VERSION
    x = torch.randn(5, ENCODED_SIZE)
    with torch.no_grad():
        torch.testing.assert_close(
            net.forward_heads(x)[2], restored.forward_heads(x)[2]
        )


def test_numpy_state_dict_carries_the_head() -> None:
    """This is what crosses the process boundary to the self-play workers."""
    torch.manual_seed(4)
    net = StructuredNet(WITH_MARGIN).eval()
    other = StructuredNet(WITH_MARGIN).eval()
    other.load_numpy_state_dict(net.cpu_state_dict())
    x = torch.randn(3, ENCODED_SIZE)
    with torch.no_grad():
        torch.testing.assert_close(net.forward_heads(x)[2], other.forward_heads(x)[2])


def test_evaluator_reports_the_margin() -> None:
    plain = NetEvaluator(StructuredNet(PLAIN).eval())
    rich = NetEvaluator(StructuredNet(WITH_MARGIN).eval())
    state = a_mid_game_state()
    legal = state.legal_actions()
    assert plain.has_margin is False
    assert len(plain(state, legal)) == 2
    assert rich.has_margin is True
    priors, value, margin = rich(state, legal)
    assert priors.shape == (len(legal),)
    assert -1.0 <= value <= 1.0
    assert -1.0 <= margin <= 1.0


# ------------------------------------------------------- 2. nothing else moves
def test_a_run3_checkpoint_still_plays_the_visit_argmax(tmp_path) -> None:
    """Bit-identical selection: the played move IS ``argmax`` over visit counts.

    The reference is a second search with the same seed, the same config and the
    same evaluator, so if ``MCTSAgent.act`` had started consulting anything else
    the two would part company on the first position where it mattered.
    """
    torch.manual_seed(5)
    net = StructuredNet(PLAIN).eval()
    path = save_checkpoint(tmp_path / "run3.pt", net)
    agent = MCTSAgent.from_checkpoint(path, sims=48, seed=5)
    reference = MCTS(NetEvaluator(net), agent.mcts.config, seed=5, add_noise=False)

    state = AzulState.new_game(seed=99)
    played = 0
    while played < 10 and not state.is_terminal:
        legal = state.legal_actions()
        if len(legal) > 1:
            expected = int(np.argmax(reference.search(state).policy))
            assert agent.act(state) == expected
            played += 1
        state.apply(legal[0] if len(legal) == 1 else expected)
    assert played >= 5, "the test never reached a position with a choice"


def test_select_play_action_is_the_old_function_without_a_margin() -> None:
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[[3, 11, 40]] = [0.2, 0.5, 0.3]
    result = SearchResult(policy, 0.0, {3: 2, 11: 5, 40: 3}, 10)
    assert result.has_margin is False
    assert select_play_action(result) == select_action(policy, 0.0) == 11
    # and the exploration path is untouched whatever the head situation is
    import random

    for has_margin in (False, True):
        r = SearchResult(policy, 0.0, {3: 2, 11: 5, 40: 3}, 10, has_margin=has_margin)
        draws = [select_play_action(r, 1.0, random.Random(7)) for _ in range(20)]
        assert draws == [select_action(policy, 1.0, random.Random(7))] * 20


# --------------------------------------------------------- 3. decisive play
def test_decisive_action_is_lexicographic() -> None:
    """Win first, margin second — and never the other way round."""
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[[1, 2, 3]] = [0.5, 0.3, 0.2]
    visits = {1: 50, 2: 30, 3: 20}

    def result(q: dict[int, float], margins: dict[int, float]) -> SearchResult:
        return SearchResult(
            policy, 0.0, visits, 100, has_margin=True, q=q, margins=margins
        )

    # all three equally winning -> the biggest margin wins, not the most visits
    same = result({1: 0.90, 2: 0.91, 3: 0.89}, {1: 0.1, 2: 0.2, 3: 0.9})
    assert decisive_action(same, eps=0.03) == 3
    # a bigger margin that costs win probability is refused: 3 is now 0.2 of win
    # value away from the best, far outside eps, so it is not even a candidate
    costly = result({1: 0.90, 2: 0.91, 3: 0.70}, {1: 0.1, 2: 0.2, 3: 0.9})
    assert decisive_action(costly, eps=0.03) == 2
    # eps is a real knob: widen it and the same position answers differently
    assert decisive_action(costly, eps=0.30) == 3
    # ties fall back to visits, so the answer never depends on dict order
    tied = result({1: 0.9, 2: 0.9, 3: 0.9}, {1: 0.5, 2: 0.5, 3: 0.5})
    assert decisive_action(tied) == 1


def test_a_barely_visited_child_cannot_define_the_best_win_value() -> None:
    """One backup of noise must not become "the best win-Q" to tie against."""
    policy = np.zeros(ACTION_SPACE, dtype=np.float32)
    policy[[1, 2]] = [0.9, 0.1]
    result = SearchResult(
        policy,
        0.0,
        {1: 100, 2: 1},  # child 2 was visited once
        101,
        has_margin=True,
        q={1: 0.80, 2: 0.99},  # ... and that once happened to look wonderful
        margins={1: 0.1, 2: 0.9},
    )
    assert decisive_action(result, min_visit_frac=0.1) == 1
    # with the floor removed it is admitted, which is exactly what the floor is for
    assert decisive_action(result, min_visit_frac=0.0) == 2


class ScriptedEvaluator:
    """Player-0-framed evaluator: a fixed win value, a score-driven margin.

    Every leaf says "player 0 is comfortably winning" (so every root child is
    within epsilon on win value and the whole set stays in play) while the margin
    tracks the actual score difference on the board. The decisive pick therefore
    has a checkable meaning: of the moves search considers equally winning, take
    the one whose subtree ends up furthest ahead on points.
    """

    def __init__(self, has_margin: bool = True, win: float = 0.9) -> None:
        self.has_margin = has_margin
        self.win = win

    def __call__(self, state: AzulState, legal: Sequence[int]) -> tuple:
        n = len(legal)
        priors = (
            np.full(n, 1.0 / n, dtype=np.float32)
            if n
            else np.zeros(0, dtype=np.float32)
        )
        flip = 1.0 if state.current_player == 0 else -1.0
        value = self.win * flip
        if not self.has_margin:
            return priors, value
        m0 = margin_target(state.scores[0] - state.scores[1])
        return priors, value, m0 * flip


def test_the_margin_never_touches_the_search_itself() -> None:
    """Same seed, same config, one evaluator with a margin and one without:
    the visit counts — i.e. the policy targets — must come out identical."""
    state = a_mid_game_state()
    cfg = MCTSConfig(sims=200, chance_children=2)
    blind = MCTS(ScriptedEvaluator(has_margin=False), cfg, seed=11).search(state)
    rich = MCTS(ScriptedEvaluator(has_margin=True), cfg, seed=11).search(state)
    assert rich.has_margin and not blind.has_margin
    assert rich.visits == blind.visits
    np.testing.assert_array_equal(rich.policy, blind.policy)
    assert rich.value == pytest.approx(blind.value)


def test_decisive_play_picks_the_bigger_score_gap() -> None:
    """End to end: the played move maximises the margin among the equal wins."""
    cfg = MCTSConfig(sims=300, chance_children=2)
    differed = 0
    checked = 0
    for seed in (21, 34, 55, 89):
        state = a_mid_game_state(seed=seed)
        if state.is_terminal or len(state.legal_actions()) < 3:  # pragma: no cover
            continue
        result = MCTS(ScriptedEvaluator(), cfg, seed=seed).search(state)
        chosen = select_play_action(result, 0.0, eps=0.03, min_visit_frac=0.1)
        checked += 1
        # the invariant: nothing search considered equally winning wins by more
        floor = 0.1 * max(result.visits.values())
        rivals = [
            a
            for a, n in result.visits.items()
            if n >= floor and a in result.q and result.q[a] >= result.q[chosen] - 0.03
        ]
        assert result.margins[chosen] == max(result.margins[a] for a in rivals)
        if chosen != select_action(result.policy, 0.0):
            differed += 1
    assert checked >= 3
    assert differed, "the tie-break never fired; the test proves nothing"


def test_the_stall_breaker_overrides_the_lexicographic_pick() -> None:
    """A game that will not end must not be kept alive by the tie-break.

    An Azul game in which neither side ever completes a pattern line never
    terminates, and *every* deterministic pick can sustain that loop. The margin
    tie-break is the worst offender, because :func:`decisive_action` ignores the
    visit counts among equally-winning children — so an agent that "adds
    randomness" by raising the temperature would find that randomness never
    reaches the decision at all. ``select_play_action(..., stalling=True)``
    therefore takes the sampled path whatever heads the net has.
    """
    result = SearchResult(
        policy=_policy({0: 0.5, 1: 0.3, 2: 0.2}),
        value=0.9,
        visits={0: 50, 1: 30, 2: 20},
        sims=100,
        has_margin=True,
        # every move is equally winning, so the tie-break decides — and it likes
        # the least-visited one, which is exactly how a loop gets sustained
        q={0: 0.90, 1: 0.90, 2: 0.90},
        margins={0: 0.10, 1: 0.20, 2: 0.90},
    )
    assert select_play_action(result, 0.0) == 2  # the lexicographic pick
    picks = {
        select_play_action(result, 0.0, random.Random(s), stalling=True)
        for s in range(60)
    }
    assert len(picks) > 1, "the stall breaker did not randomise anything"
    assert picks <= {0, 1, 2}
    # ... and it samples the VISIT distribution, so the well-searched move is
    # still the likeliest one; this is a loop breaker, not a coin toss.
    counts = [
        select_play_action(result, 0.0, random.Random(s), stalling=True)
        for s in range(400)
    ]
    assert counts.count(0) > counts.count(2)


def test_the_stall_breaker_also_covers_a_net_without_the_head() -> None:
    result = SearchResult(
        policy=_policy({0: 0.9, 1: 0.1}), value=0.0, visits={0: 90, 1: 10}, sims=100
    )
    assert select_play_action(result, 0.0) == 0  # plain argmax, unchanged
    picks = {
        select_play_action(result, 0.0, random.Random(s), stalling=True)
        for s in range(60)
    }
    assert picks == {0, 1}


def _policy(weights: dict[int, float]) -> np.ndarray:
    out = np.zeros(ACTION_SPACE, dtype=np.float32)
    for action, p in weights.items():
        out[action] = p
    return out


def test_the_agent_hands_the_stall_flag_to_the_picker(monkeypatch) -> None:
    """The rule lives in one place; the agent's job is only to report the round."""
    from ludometer.train import mcts_agent as agent_module

    torch.manual_seed(8)
    net = StructuredNet(WITH_MARGIN).eval()
    agent = MCTSAgent(net, sims=8, seed=1, stall_rounds=3)
    seen: list[bool] = []
    real = agent_module.select_play_action

    def spy(result, temperature=0.0, rng=None, **kwargs):
        seen.append(bool(kwargs.get("stalling")))
        return real(result, temperature, rng, **kwargs)

    monkeypatch.setattr(agent_module, "select_play_action", spy)
    state = a_mid_game_state(seed=21, moves=20)
    state.round_index = 0
    agent.act(state)
    state.round_index = 9  # past stall_rounds
    agent.act(state)
    assert seen == [False, True]


def test_a_terminal_node_backs_up_its_real_final_margin() -> None:
    """Search that reaches the end of the game uses facts, not an estimate."""
    state = AzulState.new_game(seed=4)
    rng = np.random.default_rng(4)
    while not state.is_terminal:
        legal = state.legal_actions()
        state.apply(legal[int(rng.integers(len(legal)))])
    from ludometer.train.mcts import Node

    node = Node(state)
    assert node.is_terminal
    expected = margin_target(state.scores[0] - state.scores[1])
    assert node.terminal_m0 == pytest.approx(expected)
    assert node.terminal_v0 == pytest.approx(float(state.outcome()))


def test_a_margin_agent_still_plays_legal_moves(tmp_path) -> None:
    torch.manual_seed(6)
    path = save_checkpoint(tmp_path / "run4.pt", StructuredNet(WITH_MARGIN).eval())
    agent = MCTSAgent.from_checkpoint(path, sims=16, seed=2)
    assert agent.net.has_margin
    state = AzulState.new_game(seed=12)
    for _ in range(8):
        action = agent.act(state)
        assert state.is_legal(action)
        state.apply(action)
        if state.is_terminal:  # pragma: no cover - not in 8 moves
            break
    assert "margin" in agent.last_search


def test_the_gui_coach_reads_a_margin_search(tmp_path) -> None:
    """The coach borrows the opponent's search; a third head must not upset it."""
    from ludometer.gui.coach import MoveCoach

    torch.manual_seed(7)
    path = save_checkpoint(tmp_path / "run4.pt", StructuredNet(WITH_MARGIN).eval())
    agent = MCTSAgent.from_checkpoint(path, sims=16, seed=3)
    analysis = MoveCoach(agent, time_budget_s=0.05).analyse(a_mid_game_state())
    assert analysis["children"]
    assert analysis["best"]["q"] is not None


# ------------------------------------------------------ 4. buffers and training
def old_format_buffer(path: Path, n: int = 320, seed: int = 0) -> Path:
    """A replay file in run1-run3's shape: no ``margins``, no ``margin_mask``.

    Written by hand rather than by :meth:`ReplayBuffer.save`, because the whole
    point is to reproduce a file this version of the code cannot write any more.
    Values are blended exactly as run1-run3 blended them, so the same file also
    exercises ``unblend``.
    """
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    values: list[float] = []
    game = 0
    while len(states) < n:
        state = AzulState.new_game(seed=1000 + game)
        game += 1
        players: list[int] = []
        start = len(states)
        while not state.is_terminal and len(states) < n:
            legal = state.legal_actions()
            policy = np.zeros(ACTION_SPACE, dtype=np.float32)
            policy[min(legal)] = 1.0
            states.append(state.encode())
            policies.append(policy)
            players.append(state.current_player)
            state.apply(legal[int(rng.integers(len(legal)))])
        outcome = float(state.outcome() or 0.0)
        m0 = margin_target(state.scores[0] - state.scores[1])
        v0 = 0.85 * outcome + 0.15 * m0
        values.extend(v0 if p == 0 else -v0 for p in players[: len(states) - start])
    with path.open("wb") as fh:
        np.savez(
            fh,
            states=np.stack(states).astype(np.float32),
            policies=np.stack(policies).astype(np.float32),
            values=np.asarray(values, dtype=np.float32),
            meta=np.array([n, n, n, game, 0], dtype=np.int64),
        )
    return path


def test_unblend_recovers_the_outcome_and_the_margin_exactly() -> None:
    outcomes = np.array([1.0, 1.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32)
    margins = np.array([0.99, 0.05, -0.99, -0.05, 0.0, 0.4], dtype=np.float32)
    blended = (0.85 * outcomes + 0.15 * margins).astype(np.float32)
    got_o, got_m = unblend_values(blended, 0.15)
    np.testing.assert_array_equal(got_o, outcomes)
    np.testing.assert_allclose(got_m, margins, atol=1e-6)
    with pytest.raises(ValueError, match="unblend weight"):
        unblend_values(blended, 0.9)


def test_an_old_buffer_loads_with_the_margin_masked_out(tmp_path) -> None:
    path = old_format_buffer(tmp_path / "old.npz")
    buf = ReplayBuffer(capacity=1000, seed=0)
    n = buf.load(path)
    assert n == 320
    assert buf.stats()["margin_targets"] == 0
    _s, _p, _v, margins, mask = buf.sample(64)[:5]
    assert not mask.any()
    assert not margins.any()

    # ... and the same file, unblended, is a fully supervised margin dataset
    rich = ReplayBuffer(capacity=1000, seed=0)
    assert rich.load(path, unblend=0.15) == 320
    assert rich.stats()["margin_targets"] == 320
    assert set(np.unique(rich.values).tolist()) <= {-1.0, 0.0, 1.0}
    assert np.abs(rich.margins).max() > 0.0


def test_a_new_buffer_roundtrips_the_margin_columns(tmp_path) -> None:
    from ludometer.train.selfplay import SelfPlayConfig, play_selfplay_game

    record = play_selfplay_game(
        UniformEvaluator(), seed=17, config=SelfPlayConfig(mcts=MCTSConfig(sims=8))
    )
    assert record.margins.shape == record.values.shape
    # one magnitude for the whole game, opposite signs for the two seats
    assert len(set(np.abs(record.margins).round(5).tolist())) == 1

    buf = ReplayBuffer(capacity=1000, seed=0)
    buf.add_game(record)
    assert buf.stats()["margin_targets"] == len(record)
    path = buf.save(tmp_path / "new.npz")
    other = ReplayBuffer(capacity=1000, seed=0)
    other.load(path)
    np.testing.assert_array_equal(other.margins[: len(record)], record.margins)
    np.testing.assert_array_equal(
        other.margin_mask[: len(record)], np.ones(len(record))
    )


def margin_config(**overrides) -> TrainConfig:
    data = {
        "run": "margin-test",
        "device": "cpu",
        "workers": 1,
        "arch": "structured",
        "margin_head": True,
        "value_score_weight": 0.0,
        "embed": 32,
        "layers": 1,
        "heads": 2,
        "body": 64,
        "body_blocks": 1,
        "value_hidden": 16,
        "policy_rank": 8,
        "sims": 4,
        "batch_size": 64,
        "replay_capacity": 2000,
        "min_buffer": 64,
        "games_per_iter": 2,
        "total_games": 2,
        "eval_games": 0,
        "eval_at_start": False,
        "pretrain_epochs": 2,
        "pretrain_lr": 3e-3,
        "heartbeat": 0.0,
    }
    data.update(overrides)
    return TrainConfig.from_dict(data)


def test_a_margin_head_needs_a_pure_value_target() -> None:
    with pytest.raises(ValueError, match="value_score_weight"):
        margin_config(value_score_weight=0.15)
    with pytest.raises(ValueError, match="arch='structured'"):
        margin_config(arch="mlp")


def test_pretraining_on_an_old_buffer_masks_the_margin_loss(tmp_path) -> None:
    """The head gets no gradient from positions that have no margin target."""
    path = old_format_buffer(tmp_path / "old.npz")
    trainer = Trainer(margin_config(), tmp_path / "run", log=None)
    trainer.prepare()
    assert trainer.pretrain(path) > 0
    assert trainer.buffer.stats()["margin_targets"] == 0

    # the claim, stated where it can be checked exactly: a fully masked batch
    # produces a zero margin loss and a zero gradient into the head. (Comparing
    # the weights themselves would fail for an unrelated reason — Adam's weight
    # decay moves every parameter, gradient or no gradient.)
    batch = trainer.buffer.sample(64)
    loss_m = trainer._losses(*batch)[2]
    assert float(loss_m.detach()) == 0.0
    trainer.net.zero_grad(set_to_none=True)
    loss_m.backward()
    grad = trainer.net.margin_out.weight.grad
    assert grad is None or not grad.any()

    lines = [
        json.loads(line)
        for line in (tmp_path / "run" / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert lines and all(entry["loss_m"] == 0.0 for entry in lines)
    assert lines[-1]["loss_p"] < lines[0]["loss_p"], "the rest still trains"


def test_pretraining_on_an_unblended_old_buffer_trains_the_margin(tmp_path) -> None:
    path = old_format_buffer(tmp_path / "old.npz")
    trainer = Trainer(margin_config(pretrain_unblend=0.15), tmp_path / "run", log=None)
    trainer.prepare()
    assert trainer.pretrain(path) > 0
    assert trainer.buffer.stats()["margin_targets"] == 320
    batch = trainer.buffer.sample(64)
    loss_m = trainer._losses(*batch)[2]
    assert float(loss_m.detach()) > 0.0
    trainer.net.zero_grad(set_to_none=True)
    loss_m.backward()
    assert trainer.net.margin_out.weight.grad.any(), "the head is being trained"
    lines = [
        json.loads(line)
        for line in (tmp_path / "run" / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert any(entry["loss_m"] > 0.0 for entry in lines)


# --------------------------------------------------------------------- export
def test_export_appends_a_margin_output(tmp_path) -> None:
    pytest.importorskip("onnx", reason="onnx is only needed to export")
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime is only for export")
    from ludometer.export.onnx_export import export_checkpoint

    torch.manual_seed(8)
    ckpt = save_checkpoint(
        tmp_path / "ckpt-000042.pt", StructuredNet(WITH_MARGIN).eval(), {"games": 42}
    )
    meta = export_checkpoint(
        ckpt=ckpt, out_dir=tmp_path / "model", samples=12, reference=None
    )
    assert meta["has_margin"] is True
    assert meta["outputs"] == ["policy", "value", "margin"]
    assert meta["parity"]["margin_max_abs_diff"] < 1e-4

    session = ort.InferenceSession(
        str(tmp_path / "model" / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    # order matters: the page reads by name, but existing readers read by index
    assert [o.name for o in session.get_outputs()] == ["policy", "value", "margin"]
    obs = np.zeros((3, ENCODED_SIZE), dtype=np.float32)
    policy, value, margin = session.run(None, {"obs": obs})
    assert policy.shape == (3, ACTION_SPACE)
    assert value.shape == margin.shape == (3, 1)
    assert (np.abs(margin) <= 1.0).all()


def test_export_of_an_old_checkpoint_is_unchanged(tmp_path) -> None:
    pytest.importorskip("onnx", reason="onnx is only needed to export")
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime is only for export")
    from ludometer.export.onnx_export import export_checkpoint

    torch.manual_seed(9)
    ckpt = save_checkpoint(tmp_path / "ckpt-000001.pt", StructuredNet(PLAIN).eval())
    meta = export_checkpoint(
        ckpt=ckpt, out_dir=tmp_path / "model", samples=8, reference=None
    )
    assert meta["has_margin"] is False
    assert meta["outputs"] == ["policy", "value"]
    assert "margin_max_abs_diff" not in meta["parity"]
    session = ort.InferenceSession(
        str(tmp_path / "model" / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    assert [o.name for o in session.get_outputs()] == ["policy", "value"]


# ------------------------------------------------------------------- configs
def test_shipped_run4_config_loads() -> None:
    cfg = TrainConfig.load(REPO / "configs" / "run4.json")
    assert cfg.arch == "structured"
    assert cfg.margin_head is True
    assert cfg.value_score_weight == 0.0, "the value head is pure win/draw/loss now"
    assert cfg.margin_weight == 0.25
    assert cfg.decisive_eps == 0.03
    assert cfg.sims == 512 and cfg.tree_reuse is True and cfg.workers == 8
    assert cfg.total_games == 60_000
    assert cfg.eval_every_games == 512 and cfg.eval_sims == 100
    assert cfg.pretrain == "runs/run3/checkpoints/replay.npz"
    assert cfg.pretrain_unblend == 0.15
    for anchor, elo in (
        ("mcts:runs/run1/checkpoints/ckpt-024064.pt?sims=100", 2014.0),
        ("mcts:runs/run2/checkpoints/ckpt-023040.pt?sims=100", 2020.3),
    ):
        assert anchor in cfg.eval_anchors
        assert cfg.anchor_elos[anchor] == elo
    assert cfg.anchor_elos["random"] == 0.0
    net = make_net(cfg.net_config())
    assert net.has_margin
    assert 1.0e6 < net.num_params < 4.0e6
    # the placeholder the orchestrator has to fill in, and how to fill it in
    raw = json.loads((REPO / "configs" / "run4.json").read_text())
    assert "run3" in raw["_note_anchors"]
    assert "gauntlet" in raw["_note_anchors"], "say how to re-rate, not just that"


def test_the_margin_head_costs_nothing_per_position() -> None:
    """The self-play budget is what pays for 512 sims; the head must be cheap.

    Two claims, one exact and one measured. The exact one: the head is a single
    ``body -> value_hidden`` Linear plus a scalar readout, so it stays a small
    fraction of a net whose body already costs a megaparameter. The measured one:
    batch-1 CPU inference is dispatch-bound rather than FLOP-bound, so those
    parameters should not show up on the clock at all — repeated runs on an idle
    machine come out at 1.00-1.08x.

    The bound is loose (1.5x) on purpose. This box normally shares itself with a
    training run, and even timing the two nets **alternately** round by round —
    so that a load spike hits both rather than whichever one was being measured —
    leaves several percent of drift. A tight bound here would fail for reasons
    that have nothing to do with the head; 1.5x still catches the regression that
    matters, which is someone giving the margin its own trunk.
    """
    import time

    torch.set_num_threads(1)
    state = a_mid_game_state(seed=5, moves=12)
    legal = state.legal_actions()
    nets = {
        name: NetEvaluator(
            make_net(
                TrainConfig.load(REPO / "configs" / f"{name}.json").net_config()
            ).eval()
        )
        for name in ("run3", "run4")
    }
    best = dict.fromkeys(nets, float("inf"))
    for _round in range(6):
        for name, evaluator in nets.items():
            for _ in range(30):  # warm this net's caches back up
                evaluator(state, legal)
            started = time.perf_counter()
            for _ in range(150):
                evaluator(state, legal)
            best[name] = min(best[name], (time.perf_counter() - started) / 150 * 1000)
    ratio = best["run4"] / best["run3"]
    print(
        f"\nmargin head: {best['run4']:.3f} vs {best['run3']:.3f} ms/position "
        f"({ratio:.2f}x), {nets['run4'].net.num_params:,} vs "
        f"{nets['run3'].net.num_params:,} params"
    )
    assert nets["run4"].net.has_margin and not nets["run3"].net.has_margin
    extra = nets["run4"].net.num_params - nets["run3"].net.num_params
    assert extra > 0, "the head has to be in there"
    assert extra / nets["run3"].net.num_params < 0.10, (
        f"the head added {extra:,} params"
    )
    assert ratio < 1.5, f"the margin head costs {ratio:.2f}x per position"


def test_smoke4_runs_end_to_end_with_masked_pretraining(tmp_path: Path) -> None:
    """The acceptance run: margin head + decisive play + a pre-run4 buffer."""
    buffer_path = old_format_buffer(tmp_path / "replay.npz", n=256)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    proc = subprocess.run(
        [
            "nice",
            "-n",
            "19",
            sys.executable,
            "-m",
            "ludometer.train.run",
            "--config",
            str(REPO / "configs" / "smoke4.json"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--max-games",
            "8",
            "--pretrain",
            str(buffer_path),
            "--pretrain-epochs",
            "1",
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    run_dir = tmp_path / "runs" / "smoke4"
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "done", status
    assert status["games"] >= 8

    entries = [
        json.loads(line)
        for line in (run_dir / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    pretrain = [e for e in entries if e.get("phase") == "pretrain"]
    selfplay = [e for e in entries if "phase" not in e]
    assert pretrain and selfplay
    assert all(e["loss_m"] == 0.0 for e in pretrain), "the old buffer has no margins"
    assert any(e["loss_m"] > 0.0 for e in selfplay), "self-play games do have margins"

    agent = MCTSAgent.from_checkpoint(
        run_dir / "checkpoints" / "latest.pt", sims=4, seed=1
    )
    assert isinstance(agent.net, StructuredNet)
    assert agent.net.has_margin
    state = AzulState.new_game(seed=3)
    assert state.is_legal(agent.act(state))
