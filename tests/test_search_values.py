"""The search's root value rides along every position, and can be mixed into
the value target (``value_search_weight``). Old files load with a zero mask."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ludometer.cloud.shards import read_shard, write_shard
from ludometer.train.mcts import MCTSConfig
from ludometer.train.net import make_net
from ludometer.train.net2 import StructuredConfig
from ludometer.train.replay import ReplayBuffer
from ludometer.train.selfplay import SelfPlayConfig, play_selfplay_game
from ludometer.train.selfplay_batched import BatchedSelfPlay
from ludometer.train.trainer import TrainConfig, Trainer

REPO = Path(__file__).resolve().parents[1]
TINY = StructuredConfig(
    embed=32,
    layers=1,
    heads=4,
    ffn_mult=2,
    body=48,
    body_blocks=1,
    value_hidden=16,
    policy_rank=8,
    margin_head=True,
)


def _cfg() -> SelfPlayConfig:
    return SelfPlayConfig(
        mcts=MCTSConfig(sims=12, tree_reuse=True, chance_children=2),
        temp_moves=4,
        max_moves=120,
        value_score_weight=0.0,
    )


def test_both_engines_record_search_values() -> None:
    torch.manual_seed(0)
    net = make_net(TINY)
    net.eval()
    from ludometer.train.net import NetEvaluator

    seq = play_selfplay_game(NetEvaluator(net), 5, _cfg())
    bat = BatchedSelfPlay(TINY, _cfg(), games=1, device="cpu")
    bat.net.load_state_dict(net.state_dict())
    bat.net.eval()
    (rec,) = bat.play(1, 5)
    for r in (seq, rec):
        assert r.search_values is not None and r.search_mask is not None
        assert len(r.search_values) == len(r) == len(r.search_mask)
        assert r.search_mask.sum() == r.decisions  # every searched move has one
        assert np.all(np.abs(r.search_values) <= 1.0)
    np.testing.assert_array_equal(seq.search_mask, rec.search_mask)
    np.testing.assert_allclose(seq.search_values, rec.search_values, atol=1e-6)


def test_buffer_round_trips_and_old_files_load_masked(tmp_path: Path) -> None:
    torch.manual_seed(1)
    rec = BatchedSelfPlay(TINY, _cfg(), games=1, device="cpu").play(1, 9)[0]
    buf = ReplayBuffer(capacity=1000, seed=0)
    buf.add_game(rec)
    assert buf.stats()["search_targets"] == rec.decisions
    buf.save(tmp_path / "b.npz")
    back = ReplayBuffer(capacity=1000, seed=0)
    back.load(tmp_path / "b.npz")
    np.testing.assert_array_equal(back.search_values[: len(rec)], rec.search_values)
    batch = back.sample(8)
    assert batch.search_values.shape == (8,) and batch.search_mask.shape == (8,)
    # a pre-2026-09 file: no search columns -> zero mask
    with np.load(tmp_path / "b.npz") as z:
        old = {k: z[k] for k in ("states", "policies", "values")}
    np.savez(tmp_path / "old.npz", **old)
    legacy = ReplayBuffer(capacity=1000, seed=0)
    legacy.load(tmp_path / "old.npz")
    assert legacy.stats()["search_targets"] == 0
    # shards carry them too
    path = write_shard(tmp_path / "s.npz", [rec], {})
    (again,), _ = read_shard(path)
    np.testing.assert_array_equal(again.search_values, rec.search_values)
    np.testing.assert_array_equal(again.search_mask, rec.search_mask)


def test_value_target_mixes_toward_search_value(tmp_path: Path) -> None:
    data = json.loads((REPO / "configs" / "smoke5.json").read_text())
    data.update(
        {"run": "sv", "device": "cpu", "eval_games": 0, "value_search_weight": 0.5}
    )
    cfg = TrainConfig.from_dict(data)
    trainer = Trainer(cfg, tmp_path / "sv", log=None)
    n = 16
    states = np.zeros((n, 182), dtype=np.float32)
    policies = np.full((n, 180), 1 / 180, dtype=np.float32)
    values = np.ones(n, dtype=np.float32)
    margins = np.zeros(n, dtype=np.float32)
    mask = np.ones(n, dtype=np.float32)
    search = -np.ones(n, dtype=np.float32)
    # with the search mask on, target = 0.5*1 + 0.5*(-1) = 0: the loss against
    # the net's output differs from the plain-outcome loss, so the mix is live
    _, lv_mixed, _, _ = trainer._losses(
        states, policies, values, margins, mask, None, None, None, search, mask
    )
    _, lv_plain, _, _ = trainer._losses(
        states,
        policies,
        values,
        margins,
        mask,
        None,
        None,
        None,
        search,
        np.zeros(n, np.float32),
    )
    with torch.no_grad():
        out = trainer.net.forward_aux(torch.from_numpy(states))[1].numpy()
    np.testing.assert_allclose(float(lv_mixed), float(np.mean(out**2)), atol=1e-5)
    np.testing.assert_allclose(
        float(lv_plain), float(np.mean((out - 1.0) ** 2)), atol=1e-5
    )
    trainer.selfplay.close()
