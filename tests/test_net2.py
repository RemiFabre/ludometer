"""Tests for the structured entity net (``ludometer/train/net2.py``).

Four things must hold or run3 cannot use it:

1. it is a drop-in for :class:`~ludometer.train.net.PolicyValueNet` — same input,
   same two outputs, same checkpoint format, and ``load_net`` /
   ``MCTSAgent.from_checkpoint`` pick the class out of the checkpoint itself
   (that polymorphism is what lets the GUI and the arena load run3 unchanged);
2. the policy never puts probability on an illegal action;
3. the factorised head really is ``<key[source, colour], query[destination]>``
   laid out in the engine's ``source * 30 + colour * 6 + dest`` order — an index
   slip here would silently score the wrong move;
4. it fits the CPU inference budget that pays for 512 sims/move.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ludometer.azul.engine import ACTION_SPACE, ENCODED_SIZE, AzulState, decode_action
from ludometer.train.benchmark import bench_inference
from ludometer.train.mcts_agent import MCTSAgent
from ludometer.train.net import (
    NetConfig,
    NetEvaluator,
    PolicyValueNet,
    load_net,
    make_net,
    masked_policy,
    net_config_from_dict,
    save_checkpoint,
)
from ludometer.train.net2 import (
    NUM_TOKENS,
    StructuredConfig,
    StructuredNet,
    token_slices,
)

# Small enough to build in milliseconds, wide enough to exercise every path.
TINY = StructuredConfig(
    embed=32,
    layers=1,
    heads=4,
    ffn_mult=2,
    body=48,
    body_blocks=1,
    value_hidden=16,
    policy_rank=8,
)
# What configs/run3.json asks for — the config the budget claim is about.
RUN3 = StructuredConfig(
    embed=96,
    layers=1,
    heads=4,
    ffn_mult=2,
    body=1024,
    body_blocks=1,
    value_hidden=128,
    policy_rank=32,
)

# ms/position on one CPU thread. The absolute number only means anything on an
# idle machine, and these tests share the box with a training run, so the bound
# scales with a reference MLP measured in the same process (see the docstring of
# ludometer.train.benchmark).
BUDGET_MS = 0.6
BUDGET_REF_MULTIPLE = 5.0


@pytest.fixture(scope="module")
def net() -> StructuredNet:
    torch.manual_seed(0)
    return StructuredNet(TINY).eval()


def sample_states(n: int = 6, seed: int = 3) -> list[AzulState]:
    rng = np.random.default_rng(seed)
    state = AzulState.new_game(seed=seed)
    out = [state.clone()]
    while len(out) < n and not state.is_terminal:
        legal = state.legal_actions()
        state.apply(legal[int(rng.integers(len(legal)))])
        if not state.is_terminal:
            out.append(state.clone())
    return out


# --------------------------------------------------------------------- layout
def test_token_layout_covers_the_board() -> None:
    slices = token_slices()
    assert sum(count for _start, count, _dims in slices.values()) == NUM_TOKENS == 22
    # 5 factories + centre, then 2 x 5 pattern rows, then the per-player summaries
    assert slices["pool"] == (0, 6, 6)
    assert slices["row"] == (6, 10, 11)
    assert slices["globals"][1] == 1


def test_entity_types_share_one_embedder(net: StructuredNet) -> None:
    """The five factories (and the ten rows) must use the *same* weights."""
    types = net.type_idx.tolist()
    assert types[:6] == [0] * 6, "all six tile sources share the pool embedder"
    assert types[6:16] == [1] * 10, "all ten pattern rows share the row embedder"
    assert net.embed_w.shape[0] == len(set(types)) == 6
    # identity comes from a per-slot bias, not from per-slot weights
    assert net.embed_b.shape == (1, NUM_TOKENS, net.config.embed)


def test_gathers_read_the_encoding_and_nothing_else(net: StructuredNet) -> None:
    idx = net.gather_idx.tolist()
    assert max(idx) == ENCODED_SIZE  # the appended zero-padding column
    assert min(idx) >= 0
    assert len(idx) == NUM_TOKENS * net.token_width


# ------------------------------------------------------------------- contract
def test_forward_shapes_and_bounded_value(net: StructuredNet) -> None:
    with torch.no_grad():
        logits, value = net(torch.randn(4, ENCODED_SIZE) * 50.0)
    assert logits.shape == (4, ACTION_SPACE)
    assert value.shape == (4,)
    assert torch.isfinite(value).all()
    assert float(value.abs().max()) <= 1.0


def test_masked_policy_never_puts_probability_on_illegal_actions(
    net: StructuredNet,
) -> None:
    for state in sample_states():
        legal = state.legal_actions()
        logits, _ = net.evaluate_batch(state.encode()[None, :])
        row = logits[0].copy()
        illegal = np.setdiff1d(np.arange(ACTION_SPACE), np.asarray(legal))
        row[illegal] += 50.0  # make the illegal moves irresistible
        probs = masked_policy(row, legal)
        assert probs[illegal].sum() == 0.0
        assert probs[np.asarray(legal)].min() > 0.0
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_evaluator_priors_align_with_legal_actions(net: StructuredNet) -> None:
    evaluator = NetEvaluator(net)
    for state in sample_states(4):
        legal = state.legal_actions()
        priors, value = evaluator(state, legal)
        assert priors.shape == (len(legal),)
        assert priors.sum() == pytest.approx(1.0, abs=1e-5)
        assert -1.0 <= value <= 1.0


def test_policy_head_is_factorised_in_engine_action_order() -> None:
    """``logit[s, c, d] == <key[s, c], query[d]> + bias[s, c]``, engine order.

    With the optional global correction switched off the head *is* the
    factorisation, so every one of the 180 logits can be checked against the
    source and destination tokens it is supposed to come from. Getting the
    ``source * 30 + colour * 6 + dest`` order wrong would leave the net scoring
    a different move than the one MCTS then plays.
    """
    torch.manual_seed(2)
    plain = StructuredNet(
        StructuredConfig(**{**TINY.to_dict(), "policy_global": False})
    ).eval()
    x = torch.randn(1, ENCODED_SIZE)
    with torch.no_grad():
        h = plain.tokens(x)
        for block in plain.trunk:
            h = block(h)
        h = plain.trunk_norm(h)
        k = plain.config.policy_rank
        src = plain.src_proj(h.narrow(1, 0, 6))
        keys = src[0, :, : 5 * k].view(6, 5, k)
        bias = src[0, :, 5 * k :]
        queries = plain.dst_proj(h.index_select(1, plain.dest_idx))[0]
        logits, _ = plain(x)
    for action in range(ACTION_SPACE):
        source, colour, dest = decode_action(action)
        expected = float(keys[source, colour] @ queries[dest] + bias[source, colour])
        assert float(logits[0, action]) == pytest.approx(expected, abs=2e-4), action


# ---------------------------------------------------------------- checkpoints
def test_config_dispatch_and_roundtrip() -> None:
    cfg = net_config_from_dict({"arch": "structured", "embed": 32, "layers": 1})
    assert isinstance(cfg, StructuredConfig)
    assert cfg.embed == 32
    assert net_config_from_dict({"hidden": 64}) == NetConfig(hidden=64)
    assert net_config_from_dict(None).to_dict()["arch"] == "mlp"
    with pytest.raises(ValueError, match="unknown net arch"):
        net_config_from_dict({"arch": "nope"})
    # unrelated training keys are ignored, like NetConfig.from_dict
    assert StructuredConfig.from_dict({"arch": "structured", "lr": 0.1}).embed == 96


def test_checkpoint_roundtrip_is_polymorphic(tmp_path, net: StructuredNet) -> None:
    path = save_checkpoint(tmp_path / "ckpt.pt", net, {"games": 11})
    restored, payload = load_net(path)
    assert isinstance(restored, StructuredNet)
    assert payload["games"] == 11
    assert restored.config == net.config
    assert payload["net_config"]["arch"] == "structured"
    x = torch.randn(5, ENCODED_SIZE)
    with torch.no_grad():
        a, b = net(x), restored(x)
    torch.testing.assert_close(a[0], b[0])
    torch.testing.assert_close(a[1], b[1])


def test_mlp_checkpoints_still_load(tmp_path) -> None:
    """run1/run2 checkpoints must keep working: no arch key means the MLP."""
    mlp = PolicyValueNet(NetConfig(hidden=32, blocks=1, value_hidden=16))
    path = save_checkpoint(tmp_path / "old.pt", mlp)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["net_config"].pop("arch")  # exactly what a pre-run3 file looks like
    torch.save(payload, path)
    restored, _ = load_net(path)
    assert isinstance(restored, PolicyValueNet)


def test_agent_from_checkpoint_plays_legal_moves(tmp_path, net: StructuredNet) -> None:
    path = save_checkpoint(tmp_path / "ckpt.pt", net)
    agent = MCTSAgent.from_checkpoint(path, sims=8, seed=1)
    state = AzulState.new_game(seed=12)
    for _ in range(6):
        action = agent.act(state)
        assert state.is_legal(action)
        state.apply(action)
        if state.is_terminal:  # pragma: no cover - not in 6 moves
            break


def test_numpy_state_dict_roundtrip(net: StructuredNet) -> None:
    """This is what crosses the process boundary to the self-play workers."""
    weights = net.cpu_state_dict()
    assert all(isinstance(v, np.ndarray) for v in weights.values())
    other = StructuredNet(net.config)
    other.load_numpy_state_dict(weights)
    x = torch.randn(3, ENCODED_SIZE)
    with torch.no_grad():
        torch.testing.assert_close(net(x)[1], other(x)[1])


# ------------------------------------------------------------------- learning
def test_value_head_learns_both_seats() -> None:
    """Same convention as the MLP: value is for the player to move."""
    torch.manual_seed(1)
    states = sample_states(8, seed=11)
    mine = np.stack([s.encode() for s in states])
    swapped = []
    for s in states:
        other = s.clone()
        other.current_player = 1 - other.current_player
        swapped.append(other.encode())
    x = torch.from_numpy(np.concatenate([mine, np.stack(swapped)]))
    y = torch.cat([torch.ones(len(mine)), -torch.ones(len(mine))])
    model = StructuredNet(TINY)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(300):
        _, value = model(x)
        loss = torch.nn.functional.mse_loss(value, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert float(loss.detach()) < 0.1
    with torch.no_grad():
        _, value = model(x)
    assert float(value[: len(mine)].min()) > 0.3
    assert float(value[len(mine) :].max()) < -0.3


# --------------------------------------------------------------------- budget
def test_cpu_inference_fits_the_self_play_budget() -> None:
    """Report the actual ms/position and hold it to the run3 budget."""
    torch.set_num_threads(1)
    result = bench_inference(StructuredNet(RUN3), rounds=6, per_round=120)
    print(
        f"\nrun3 structured net: {int(result['params']):,} params, "
        f"{result['ms']:.3f} ms/position "
        f"(reference 3x512 MLP {result['ref_ms']:.3f} ms, "
        f"{result['ratio']:.2f}x)"
    )
    assert 1.0e6 < result["params"] < 4.0e6
    bound = max(BUDGET_MS, BUDGET_REF_MULTIPLE * result["ref_ms"])
    assert result["ms"] <= bound, (
        f"{result['ms']:.3f} ms/position exceeds {bound:.3f} "
        f"(reference MLP took {result['ref_ms']:.3f} ms, so the machine is busy)"
    )


def test_make_net_builds_the_shipped_configs() -> None:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("run3", "smoke3"):
        data = json.loads((root / "configs" / f"{name}.json").read_text())
        built = make_net(data)
        assert isinstance(built, StructuredNet), name
        logits, value = built(torch.zeros(2, ENCODED_SIZE))
        assert logits.shape == (2, ACTION_SPACE)
        assert value.shape == (2,)
