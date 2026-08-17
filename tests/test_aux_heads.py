"""Tests for run6's auxiliary strategic heads (see ``ludometer/train/net2.py``).

The complaint this answers: run5's tactics are good and its long-term play is
weak, which is what a net supervised only on "who won" and "by how much" should
look like. run6 predicts, off the same trunk, which wall rows / columns / colours
**both** players will hold when the game is over — 30 sigmoids whose label is
decided several rounds after the position that carries it.

Five things have to hold, one section each:

1. **the heads exist and are what they claim** — 30 outputs, the documented
   layout, and a head that can actually fit its target;
2. **the targets are right** — computed from real final walls, in the
   player-to-move frame, checked against walls built by hand;
3. **the buffer carries them, compactly and compatibly** — 30 bits a position, a
   pre-run6 file still loads, and a pre-run6 file's rows are masked out;
4. **the loss is masked** — a position with no aux label contributes no aux
   gradient, and the head still learns from the ones that do;
5. **nothing without the heads moves, in either direction** — a run5 checkpoint
   loads and plays identically everywhere, and a run6 checkpoint loads in the
   agent, the arena and the ONNX exporter, whose existing outputs are untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ludometer.azul.engine import ACTION_SPACE, ENCODED_SIZE, AzulState, wall_col
from ludometer.eval.arena import play_game
from ludometer.train.mcts_agent import MCTSAgent, MCTSAgentSpec
from ludometer.train.net import NetEvaluator, load_net, make_net, save_checkpoint
from ludometer.train.net2 import (
    AUX_OUTPUTS,
    AUX_VERSION,
    MARGIN_VERSION,
    StructuredConfig,
    StructuredNet,
    aux_slices,
)
from ludometer.train.replay import (
    AUX_BITS,
    AUX_BYTES,
    ReplayBuffer,
    pack_aux,
    unpack_aux,
)
from ludometer.train.selfplay import SelfPlayConfig, aux_targets, play_selfplay_game
from ludometer.train.trainer import TrainConfig, Trainer

REPO = Path(__file__).resolve().parents[1]

# run5's shape in miniature: margin head, no aux heads.
RUN5_LIKE = StructuredConfig(
    embed=32,
    layers=1,
    heads=4,
    body=48,
    body_blocks=1,
    value_hidden=16,
    policy_rank=8,
    margin_head=True,
)
RUN6_LIKE = StructuredConfig(**{**RUN5_LIKE.to_dict(), "aux_heads": True})


# ------------------------------------------------------------------ 1. the heads
def test_config_and_version_track_the_new_head() -> None:
    assert RUN5_LIKE.aux_heads is False
    assert RUN5_LIKE.version == MARGIN_VERSION == 2
    assert RUN6_LIKE.aux_heads is True and RUN6_LIKE.margin_head is True
    assert RUN6_LIKE.version == AUX_VERSION == 3
    # a checkpoint that only recorded the version still rebuilds the right net
    assert StructuredConfig.from_dict({"version": 3}).aux_heads is True
    # ... and one that only recorded the flag reports the right version
    assert StructuredConfig.from_dict({"aux_heads": True}).version == AUX_VERSION
    # a run3/run4/run5 net_config is unambiguously aux-free
    assert StructuredConfig.from_dict({"arch": "structured"}).aux_heads is False
    assert StructuredConfig.from_dict({"version": 2}).aux_heads is False
    assert StructuredConfig.from_dict({"margin_head": True}).aux_heads is False


def test_forward_gives_thirty_aux_logits_without_touching_the_other_heads() -> None:
    torch.manual_seed(0)
    net = StructuredNet(RUN6_LIKE).eval()
    assert net.has_aux and net.has_margin
    x = torch.randn(4, ENCODED_SIZE) * 20.0
    with torch.no_grad():
        two = net(x)
        three = net.forward_heads(x)
        logits, value, margin, aux = net.forward_aux(x)
    assert aux.shape == (4, AUX_OUTPUTS) == (4, 30)
    # forward() is still two outputs and forward_heads() still three: that is what
    # self-play, the arena and the ONNX wrapper call.
    assert len(two) == 2 and len(three) == 3
    torch.testing.assert_close(two[0], logits)
    torch.testing.assert_close(two[1], value)
    torch.testing.assert_close(three[2], margin)
    # the aux head emits LOGITS (the sigmoid lives in the loss and the exporter)
    assert float(torch.sigmoid(aux).max()) <= 1.0
    assert float(torch.sigmoid(aux).min()) >= 0.0


def test_a_net_without_the_aux_heads_says_so_everywhere() -> None:
    net = StructuredNet(RUN5_LIKE).eval()
    assert net.has_aux is False
    assert not [k for k in net.state_dict() if k.startswith("aux")]
    with torch.no_grad():
        _logits, _value, margin, aux = net.forward_aux(torch.zeros(2, ENCODED_SIZE))
    assert aux is None and margin is not None


def test_the_aux_layout_is_the_documented_one() -> None:
    s = aux_slices()
    assert s["me_rows"] == slice(0, 5)
    assert s["me_cols"] == slice(5, 10)
    assert s["me_colors"] == slice(10, 15)
    assert s["them_rows"] == slice(15, 20)
    assert s["them_colors"] == slice(25, 30)


def test_the_aux_head_can_fit_its_target() -> None:
    """A head that cannot learn 30 correlated bits would be decoration."""
    torch.manual_seed(3)
    net = StructuredNet(RUN6_LIKE)
    x = torch.randn(24, ENCODED_SIZE)
    target = (torch.rand(24, AUX_OUTPUTS) < torch.sigmoid(x[:, :1])).float()
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    first = last = 0.0
    for step in range(200):
        _l, _v, _m, aux = net.forward_aux(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(aux, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach())
        first = first or last
    assert last < 0.3 * first, f"aux BCE went {first:.3f} -> {last:.3f}"


# --------------------------------------------------------------- 2. the targets
def a_finished_game(seed: int = 5) -> AzulState:
    rng = np.random.default_rng(seed)
    state = AzulState.new_game(seed=seed)
    while not state.is_terminal:
        legal = state.legal_actions()
        state.apply(legal[int(rng.integers(len(legal)))])
    return state


def test_wall_summary_reads_a_hand_built_wall() -> None:
    """Rows, columns and colours, spelled out on a wall we filled ourselves."""
    state = AzulState.new_game(seed=1)
    wall = [0] * 25
    wall[5:10] = [1] * 5  # row 1 complete
    for r in range(5):
        wall[r * 5 + 2] = 1  # column 2 complete
    for r in range(5):
        wall[r * 5 + wall_col(3, r)] = 1  # colour 3 (black) complete
    state.walls[0] = wall
    state.walls[1] = [0] * 25

    bits = state.wall_summary(0)
    assert len(bits) == 15
    assert bits[:5] == [0, 1, 0, 0, 0], "only row 1 is closed"
    assert bits[5:10] == [0, 0, 1, 0, 0], "only column 2 is closed"
    assert bits[10:15] == [0, 0, 0, 1, 0], "only colour 3 is closed"
    assert state.wall_summary(1) == [0] * 15
    # and it agrees with the counts the engine already computed for scoring
    assert sum(bits[:5]) == state.completed_rows(0)
    assert sum(bits[5:10]) == state.completed_cols(0)
    assert sum(bits[10:15]) == state.completed_colors(0)


def test_aux_targets_are_per_seat_and_only_two_distinct_rows() -> None:
    state = a_finished_game()
    players = [0, 1, 1, 0, 1]
    targets = aux_targets(state, players)
    assert targets.shape == (5, AUX_OUTPUTS)
    p0 = np.array(state.wall_summary(0) + state.wall_summary(1), dtype=np.uint8)
    p1 = np.array(state.wall_summary(1) + state.wall_summary(0), dtype=np.uint8)
    for i, p in enumerate(players):
        np.testing.assert_array_equal(targets[i], p0 if p == 0 else p1)
    # the two seats see the same facts with the halves swapped
    np.testing.assert_array_equal(p0[:15], p1[15:])
    assert aux_targets(state, []).shape[0] == 0


def test_a_self_play_record_carries_the_true_final_walls() -> None:
    """End to end: the label on every position is the board the game ended on."""
    config = SelfPlayConfig(value_score_weight=0.0, max_moves=200)
    config = SelfPlayConfig(**{**config.__dict__, "temp_moves": 4})
    record = play_selfplay_game(_uniform_evaluator(), 11, config)
    assert record.aux.shape == (len(record.values), AUX_OUTPUTS)
    assert set(np.unique(record.aux)) <= {0, 1}
    # a finished Azul game always closes at least one row for somebody
    assert record.aux[:, :5].sum() + record.aux[:, 15:20].sum() > 0
    # every row is one of the two seat views
    assert len({tuple(row) for row in record.aux}) <= 2


def _uniform_evaluator():
    from ludometer.train.mcts import UniformEvaluator

    return UniformEvaluator()


# ---------------------------------------------------------------- 3. the buffer
def test_aux_bits_survive_a_pack_round_trip() -> None:
    rng = np.random.default_rng(0)
    bits = (rng.random((37, AUX_BITS)) < 0.5).astype(np.uint8)
    packed = pack_aux(bits)
    assert packed.shape == (37, AUX_BYTES) == (37, 4)
    np.testing.assert_array_equal(unpack_aux(packed), bits.astype(np.float32))
    # already-packed input passes through untouched
    assert pack_aux(packed) is packed
    with pytest.raises(ValueError):
        pack_aux(bits[:, :7])


def make_block(n: int, aux: bool = True):
    rng = np.random.default_rng(1)
    states = rng.random((n, ENCODED_SIZE)).astype(np.float32)
    policies = rng.random((n, ACTION_SPACE)).astype(np.float32)
    values = rng.random(n).astype(np.float32)
    margins = rng.random(n).astype(np.float32)
    bits = (rng.random((n, AUX_BITS)) < 0.5).astype(np.uint8) if aux else None
    return states, policies, values, margins, bits


def test_the_buffer_stores_aux_and_reports_the_coverage() -> None:
    buf = ReplayBuffer(capacity=64, seed=0)
    states, policies, values, margins, bits = make_block(20)
    buf.add(states, policies, values, margins, aux=bits)
    buf.add(*make_block(10, aux=False)[:4])  # a block with no aux target
    stats = buf.stats()
    assert stats["size"] == 30
    assert stats["aux_targets"] == 20
    assert stats["policy_targets"] == 30, "policy targets default to present"
    batch = buf.sample(16)
    assert batch.aux.shape == (16, AUX_BITS)
    assert batch.aux_mask.shape == (16,)


def test_a_pre_run6_file_still_loads_with_the_aux_masked_out(tmp_path) -> None:
    """Pretraining run6 on run5's buffer is exactly this path."""
    old = ReplayBuffer(capacity=64, seed=0)
    states, policies, values, margins, _bits = make_block(20)
    old.add(states, policies, values, margins)
    path = old.save(tmp_path / "run5like.npz")
    # strip the run6 keys, so the file is byte-for-byte a pre-run6 one
    with np.load(path) as data:
        keep = {k: data[k] for k in data.files if not k.startswith(("aux", "policy_"))}
    np.savez(path, **keep)

    buf = ReplayBuffer(capacity=64, seed=0)
    assert buf.load(path) == 20
    assert buf.stats()["aux_targets"] == 0
    assert buf.stats()["margin_targets"] == 20
    # ... and every one of those positions still has a policy target
    assert buf.stats()["policy_targets"] == 20


def test_a_run6_file_round_trips(tmp_path) -> None:
    buf = ReplayBuffer(capacity=64, seed=0)
    states, policies, values, margins, bits = make_block(20)
    buf.add(states, policies, values, margins, aux=bits, policy_mask=0.0)
    path = buf.save(tmp_path / "run6.npz")
    again = ReplayBuffer(capacity=64, seed=0)
    again.load(path)
    np.testing.assert_array_equal(again.aux[:20], pack_aux(bits))
    assert again.stats()["aux_targets"] == 20
    assert again.stats()["policy_targets"] == 0
    # an older reader that only knows the first three arrays is unaffected
    with np.load(path) as data:
        assert {"states", "policies", "values"} <= set(data.files)


# ------------------------------------------------------------------ 4. the loss
def aux_config(**overrides) -> TrainConfig:
    data = {
        "run": "aux",
        "arch": "structured",
        "margin_head": True,
        "aux_heads": True,
        "value_score_weight": 0.0,
        "embed": 32,
        "layers": 1,
        "heads": 4,
        "body": 48,
        "body_blocks": 1,
        "value_hidden": 16,
        "policy_rank": 8,
        "device": "cpu",
        "batch_size": 32,
        "replay_capacity": 2000,
        "eval_games": 0,
        "eval_at_start": False,
        "heartbeat": 0.0,
    }
    data.update(overrides)
    return TrainConfig.from_dict(data)


def test_a_masked_out_position_contributes_no_aux_gradient(tmp_path) -> None:
    trainer = Trainer(aux_config(), tmp_path / "run", log=None)
    states, policies, values, margins, bits = make_block(8)
    zeros = np.zeros((8, AUX_BITS), dtype=np.float32)

    def aux_grad(aux, mask):
        trainer.net.zero_grad(set_to_none=True)
        loss_a = trainer._losses(
            states, policies, values, margins, np.ones(8), aux, mask
        )[3]
        loss_a.backward()
        head = trainer.net.aux_out
        return float(loss_a.detach()), head.weight.grad.abs().sum().item()

    loss_real, grad_real = aux_grad(
        bits.astype(np.float32), np.ones(8, dtype=np.float32)
    )
    loss_masked, grad_masked = aux_grad(
        bits.astype(np.float32), np.zeros(8, dtype=np.float32)
    )
    assert grad_real > 0.0
    assert loss_masked == 0.0 and grad_masked == 0.0
    # a fabricated all-zero target would NOT be free: masking is doing real work
    loss_fake, grad_fake = aux_grad(zeros, np.ones(8, dtype=np.float32))
    assert grad_fake > 0.0 and loss_fake != loss_real


def test_the_aux_weight_is_the_only_thing_scaling_the_term(tmp_path) -> None:
    trainer = Trainer(aux_config(aux_weight=0.1), tmp_path / "run", log=None)
    parts = [torch.tensor(2.0), torch.tensor(3.0), torch.tensor(4.0), torch.tensor(5.0)]
    total = trainer._total_loss(*parts)
    assert float(total) == pytest.approx(2.0 + 1.0 * 3.0 + 0.25 * 4.0 + 0.1 * 5.0)


def test_pretraining_a_run6_net_on_a_run5_buffer_skips_the_aux(tmp_path) -> None:
    """The launch path for run6: old data, new heads, no fabricated labels."""
    old = ReplayBuffer(capacity=512, seed=0)
    states, policies, values, margins, _ = make_block(256)
    old.add(states, policies, values, margins)
    path = old.save(tmp_path / "old.npz")

    trainer = Trainer(
        aux_config(pretrain_epochs=1, pretrain_lr=1e-3), tmp_path / "run", log=None
    )
    trainer.prepare()
    assert trainer.pretrain(path) > 0
    assert trainer.buffer.stats()["aux_targets"] == 0
    batch = trainer.buffer.sample(64)
    losses = trainer._losses(*batch)
    assert float(losses[3].detach()) == 0.0, "no aux gradient from old data"
    assert float(losses[1].detach()) > 0.0, "the value head still learns"


# ------------------------------------------------------- 5. nothing else moves
def test_a_run5_checkpoint_loads_and_plays_unchanged(tmp_path) -> None:
    torch.manual_seed(4)
    net = make_net(RUN5_LIKE)
    path = save_checkpoint(tmp_path / "run5.pt", net, {"games": 1})
    again, _payload = load_net(path)
    assert again.has_margin is True and again.has_aux is False
    assert again.config.version == MARGIN_VERSION
    state = AzulState.new_game(seed=2)
    before = NetEvaluator(net)(state, state.legal_actions())
    after = NetEvaluator(again)(state, state.legal_actions())
    assert len(before) == len(after) == 3
    np.testing.assert_allclose(before[0], after[0], atol=1e-6)


def test_a_run6_checkpoint_loads_in_the_agent_and_the_arena(tmp_path) -> None:
    torch.manual_seed(5)
    net = make_net(RUN6_LIKE)
    path = save_checkpoint(tmp_path / "run6.pt", net, {"games": 1})
    again, _payload = load_net(path)
    assert again.has_aux is True and again.has_margin is True
    assert again.config.version == AUX_VERSION

    agent = MCTSAgent.from_checkpoint(path, sims=8, seed=1)
    state = AzulState.new_game(seed=3)
    assert agent.act(state) in state.legal_actions()
    # the evaluator still sees three values, not four: the aux head never enters
    # the search (it has no meaning for a leaf's value)
    assert len(agent.evaluator(state, state.legal_actions())) == 3

    spec = MCTSAgentSpec(path=str(path), sims=4, seed=2, name="run6")
    result = play_game(spec, "greedy", seed=9)
    assert result.moves > 0 and not result.truncated


def test_the_gui_registry_can_build_a_run6_checkpoint(tmp_path) -> None:
    from ludometer.agents.registry import load_agent

    torch.manual_seed(6)
    path = save_checkpoint(tmp_path / "run6.pt", make_net(RUN6_LIKE), {"games": 1})
    agent = load_agent(f"mcts:{path}?sims=8", seed=1)
    state = AzulState.new_game(seed=4)
    assert agent.act(state) in state.legal_actions()


@pytest.mark.parametrize("cfg,expected", [(RUN5_LIKE, 3), (RUN6_LIKE, 4)])
def test_the_exporter_emits_the_aux_output_only_when_it_exists(
    tmp_path, cfg, expected
) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from ludometer.export.onnx_export import export_checkpoint, output_names

    torch.manual_seed(7)
    net = make_net(cfg)
    assert len(output_names(net)) == expected
    path = save_checkpoint(tmp_path / "ckpt.pt", net, {"games": 1})
    meta = export_checkpoint(
        ckpt=path, out_dir=tmp_path / "out", samples=6, reference=None
    )
    # policy and value keep their names AND their positions, whatever is appended
    assert meta["outputs"][:2] == ["policy", "value"]
    assert meta["outputs"][2] == "margin"
    assert meta["has_aux"] is (expected == 4)
    assert (meta["outputs"] + [None])[3] == ("wall" if expected == 4 else None)
    assert meta["parity"]["checked"] and meta["parity"]["value_max_abs_diff"] < 1e-4

    import onnxruntime as ort

    session = ort.InferenceSession(str(meta.onnx_path))
    obs = AzulState.new_game(seed=1).encode()[None, :]
    out = session.run(None, {"obs": obs})
    assert out[0].shape == (1, ACTION_SPACE) and out[1].shape == (1, 1)
    if expected == 4:
        # the graph emits probabilities, so a browser needs no sigmoid of its own
        assert out[3].shape == (1, AUX_OUTPUTS)
        assert 0.0 <= out[3].min() and out[3].max() <= 1.0
        assert "wall_max_abs_diff" in meta["parity"]


def test_the_shipped_configs_build_and_run6_asks_for_the_heads() -> None:
    for name in ("run3", "run4", "run5", "smoke3", "smoke4", "smoke5"):
        cfg = TrainConfig.load(REPO / "configs" / f"{name}.json")
        assert cfg.aux_heads is False, name
        assert make_net(cfg.net_config()).has_aux is False, name
    for name in ("run6", "smoke6"):
        cfg = TrainConfig.load(REPO / "configs" / f"{name}.json")
        cfg.validate()
        net = make_net(cfg.net_config())
        assert net.has_aux and net.has_margin, name
        assert net.config.version == AUX_VERSION
    run6 = TrainConfig.load(REPO / "configs" / "run6.json")
    assert run6.aux_weight == 0.1
    assert run6.margin_weight == 0.25
    assert run6.chance_children == 8
    assert run6.pretrain.endswith("run5/checkpoints/replay.npz")
    assert run6.pretrain_unblend == 0.0, "run5's buffer stores margins natively"
    assert run6.total_games == 60000
    # the run5 anchors are all still pinned, on the same scale
    run5 = TrainConfig.load(REPO / "configs" / "run5.json")
    assert set(run5.eval_anchors) <= set(run6.eval_anchors)
    assert run5.anchor_elos.items() <= run6.anchor_elos.items()
    # ... and the launch note tells whoever launches it to add run5's own best
    raw = json.loads((REPO / "configs" / "run6.json").read_text())
    assert "ADD RUN5'S BEST HERE AT LAUNCH" in raw["_note_anchors"]
    assert "gauntlet" in raw["_note_anchors"]
