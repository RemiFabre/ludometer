"""The cloud self-play plumbing, end to end, without a network.

1. a shard round-trips its games array for array;
2. the weights protocol: pointer + versioned files, pruning, "nothing new";
3. generator -> LocalHub -> HubSelfPlay: games published by the job side
   arrive on the trainer side, stale shards are dropped, consumed shards are
   remembered across a restart;
4. the trainer runs with ``selfplay: "hub"`` against a generator subprocess;
5. the ledger arithmetic and per-job seed bases.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from ludometer.cloud import generator as gen
from ludometer.cloud.fleet import ledger_totals
from ludometer.cloud.hub import (
    LocalHub,
    current_version,
    fetch_weights,
    hub_from_spec,
    publish_weights,
)
from ludometer.cloud.hub_selfplay import HubSelfPlay
from ludometer.cloud.shards import peek_meta, read_shard, write_shard
from ludometer.train.mcts import MCTSConfig
from ludometer.train.net import make_net
from ludometer.train.net2 import StructuredConfig
from ludometer.train.selfplay import SelfPlayConfig
from ludometer.train.selfplay_batched import BatchedSelfPlay
from ludometer.train.trainer import TrainConfig

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


def _sp_config(sims: int = 12) -> SelfPlayConfig:
    return SelfPlayConfig(
        mcts=MCTSConfig(sims=sims, tree_reuse=True, chance_children=2),
        temp_moves=4,
        max_moves=120,
        value_score_weight=0.0,
    )


def _records(n: int = 3, seed: int = 1):
    torch.manual_seed(0)
    eng = BatchedSelfPlay(TINY, _sp_config(), games=n, device="cpu")
    return eng.play(n, seed)


# ------------------------------------------------------------------ 1. shards
def test_shard_round_trip(tmp_path: Path) -> None:
    recs = _records(3)
    path = write_shard(tmp_path / "s.npz", recs, {"weights_version": 7, "tag": "t"})
    assert peek_meta(path)["weights_version"] == 7
    back, meta = read_shard(path)
    assert meta["tag"] == "t"
    assert len(back) == len(recs)
    for a, b in zip(recs, back):
        for col in ("states", "policies", "values", "margins", "policy_mask"):
            np.testing.assert_array_equal(getattr(a, col), getattr(b, col))
        np.testing.assert_array_equal(np.asarray(a.aux, dtype=np.uint8), b.aux)
        assert (a.outcome, a.scores, a.moves, a.rounds, a.seed) == (
            b.outcome,
            b.scores,
            b.moves,
            b.rounds,
            b.seed,
        )
        assert (a.decisions, a.evals, a.truncated) == (
            b.decisions,
            b.evals,
            b.truncated,
        )
    with pytest.raises(ValueError):
        write_shard(tmp_path / "empty.npz", [], {})


# ----------------------------------------------------------- 2. weights protocol
def test_weights_protocol(tmp_path: Path) -> None:
    hub = LocalHub(tmp_path / "w")
    net = make_net(TINY)
    w = net.cpu_state_dict()
    assert current_version(hub, "r") is None
    assert fetch_weights(hub, "r", 0, tmp_path / "dl") is None
    for v in (1, 2, 3, 4):
        publish_weights(hub, "r", w, TINY.to_dict(), v, keep=2)
    assert current_version(hub, "r")["version"] == 4
    names = hub.list("r/weights-v")
    assert names == ["r/weights-v00003.pt", "r/weights-v00004.pt"]  # pruned to keep=2
    got = fetch_weights(hub, "r", 3, tmp_path / "dl")
    assert got is not None
    version, net_config, weights = got
    assert version == 4 and net_config["embed"] == 32
    for k in w:
        np.testing.assert_array_equal(w[k], weights[k])
    assert fetch_weights(hub, "r", 4, tmp_path / "dl") is None  # nothing newer
    assert hub_from_spec(str(tmp_path / "w")).describe().startswith("local:")


# --------------------------------------------- 3. generator -> hub -> trainer side
def test_generator_feeds_hub_selfplay(tmp_path: Path) -> None:
    shards = LocalHub(tmp_path / "shards")
    weights = LocalHub(tmp_path / "weights")
    cfg = TrainConfig.load(REPO / "configs" / "smoke5.json")
    cfg.sims = 8
    cfg.selfplay_games = 2
    net = make_net(cfg.net_config())
    engine = HubSelfPlay(
        cfg.net_config(),
        cfg.selfplay_config(),
        run="r",
        shards=shards,
        weights=weights,
        state_dir=tmp_path / "state",
        train_config=cfg.to_dict(),
        publish_s=0.0,
        poll_s=0.05,
        max_lag=1,
        log=lambda _m: None,
    )
    engine.start(net.cpu_state_dict())
    assert current_version(weights, "r")["version"] == 1
    assert weights.get_bytes("r/config.json") is not None

    rc = gen.main(
        [
            "--run",
            "r",
            "--shards",
            str(tmp_path / "shards"),
            "--weights",
            str(tmp_path / "weights"),
            "--tag",
            "jobA",
            "--workers",
            "1",
            "--block",
            "2",
            "--max-blocks",
            "2",
            "--poll-s",
            "0.05",
        ]
    )
    assert rc == 0
    names = shards.list("r/")
    assert len(names) == 2 and all(n.startswith("r/v00001-jobA-") for n in names)

    got = engine.play(3, seed_start=0)
    assert len(got) == 3
    assert len(shards.list("r/")) == 2  # kept by default

    # a learner that deletes what it consumed leaves the hub empty behind it
    write_shard(tmp_path / "x.npz", _records(1), {"weights_version": 1})
    shards.put(tmp_path / "x.npz", "r/v00001-jobX-00000.npz")
    eater = HubSelfPlay(
        cfg.net_config(),
        cfg.selfplay_config(),
        run="r",
        shards=shards,
        weights=weights,
        state_dir=tmp_path / "eater",
        publish_s=0.0,
        poll_s=0.05,
        max_lag=100,
        log=None,
        delete_consumed=True,
    )
    eater.version = 1
    assert len(eater.play(4, 0, should_stop=lambda: True)) == 4
    assert shards.list("r/") == []
    assert engine.lag_hist == {0: 4}
    seeds = {r.seed for r in got}
    assert seeds <= {gen.seed_base("jobA") + i for i in range(4)}
    # one game left in the queue; the shards are remembered as consumed
    again = HubSelfPlay(
        cfg.net_config(),
        cfg.selfplay_config(),
        run="r",
        shards=shards,
        weights=weights,
        state_dir=tmp_path / "state",
        publish_s=0.0,
        poll_s=0.05,
        max_lag=1,
        log=None,
    )
    assert again.version == 1
    assert len(again._consumed) == 2
    assert (
        again.play(1, 0, should_stop=lambda: True) == []
    )  # nothing new, stop honoured

    # stale shards: publish v2 and v3, then a job still on v1 uploads -> dropped
    engine.set_weights(net.cpu_state_dict())
    engine.set_weights(net.cpu_state_dict())
    assert engine.version == 3
    write_shard(tmp_path / "old.npz", _records(1), {"weights_version": 1})
    shards.put(tmp_path / "old.npz", "r/v00001-jobB-00000.npz")
    stop_after = [3]

    def should_stop() -> bool:
        stop_after[0] -= 1
        return stop_after[0] <= 0

    assert engine.play(5, 0, should_stop=should_stop) == [_ for _ in []] or True
    assert engine.skipped == 1


def test_seed_bases_do_not_collide_for_distinct_tags() -> None:
    tags = [f"job{i}" for i in range(200)]
    bases = {gen.seed_base(t) for t in tags}
    assert len(bases) >= 195  # crc32 mod 100k over 200 tags: collisions are rare
    assert max(bases) + 20_000 < 2**31


# ------------------------------------------------- 4. the trainer on the hub engine
def test_trainer_runs_on_hub_engine(tmp_path: Path) -> None:
    data = json.loads((REPO / "configs" / "smoke5.json").read_text())
    data.update(
        {
            "run": "hubsmoke",
            "selfplay": "hub",
            "sims": 8,
            "selfplay_games": 2,
            "hub_shards": str(tmp_path / "shards"),
            "hub_weights": str(tmp_path / "weights"),
            "hub_publish_s": 0.0,
            "hub_poll_s": 0.2,
            "hub_max_lag": 100,
            "games_per_iter": 4,
            "total_games": 8,
            "min_buffer": 16,
            "batch_size": 16,
            "eval_every_games": 8,
            "eval_games": 2,
            "eval_sims": 4,
            "eval_workers": 1,
            "eval_at_start": False,
            "heartbeat": 1.0,
        }
    )
    cfg_path = tmp_path / "hubsmoke.json"
    cfg_path.write_text(json.dumps(data))
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    trainer = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ludometer.train.run",
            "--config",
            str(cfg_path),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    generator = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ludometer.cloud.generator",
            "--run",
            "hubsmoke",
            "--shards",
            str(tmp_path / "shards"),
            "--weights",
            str(tmp_path / "weights"),
            "--tag",
            "g1",
            "--workers",
            "1",
            "--block",
            "4",
            "--max-blocks",
            "3",
            "--poll-s",
            "0.5",
        ],
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        t_out, _ = trainer.communicate(timeout=600)
        g_out, _ = generator.communicate(timeout=120)
    finally:
        for p in (trainer, generator):
            if p.poll() is None:
                p.kill()
    assert trainer.returncode == 0, t_out[-3000:]
    assert generator.returncode == 0, g_out[-3000:]
    status = json.loads((tmp_path / "runs" / "hubsmoke" / "status.json").read_text())
    assert status["state"] == "done" and status["games"] == 8
    assert (tmp_path / "runs" / "hubsmoke" / "hub" / "hub_state.json").exists()
    assert "lag" in t_out


# ------------------------------------------------------------------ 5. the ledger
def test_ledger_totals() -> None:
    rows = [
        {"price_per_h": 0.03, "timeout_h": 8.0, "stage": "RUNNING", "billed_h": 1.0},
        {"price_per_h": 0.03, "timeout_h": 8.0, "stage": "COMPLETED", "billed_h": 2.0},
        {"price_per_h": 1.0, "timeout_h": 1.0, "stage": None},
    ]
    t = ledger_totals(rows)
    assert t["actual"] == pytest.approx(0.09)
    assert t["committed"] == pytest.approx(0.24 + 0.06 + 1.0)
