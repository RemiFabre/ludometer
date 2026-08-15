"""Tests for replay-buffer pretraining and the run3 stack end to end.

``Trainer.pretrain`` is the warm start: a fresh net is fitted to an existing
``replay.npz`` (run2's, in production) before a single self-play game is played.
What has to be true is that it reads the buffer format ``replay.py`` writes, that
the loss actually goes down, that it says so in ``train.jsonl`` under
``"phase": "pretrain"``, and that the loaded positions are still in the buffer
afterwards so early self-play trains on something.

The last test is the acceptance run for the whole run3 stack: ``configs/smoke3.json``
(structured net + tree reuse + pretraining) driven through ``run.py`` in a
subprocess, exactly as the orchestrator will drive ``configs/run3.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ludometer.azul.engine import ACTION_SPACE, AzulState
from ludometer.train.replay import ReplayBuffer
from ludometer.train.trainer import TrainConfig, Trainer

REPO = Path(__file__).resolve().parents[1]


def synthetic_buffer(path: Path, n: int = 640, seed: int = 0) -> Path:
    """A small, *learnable* replay file in exactly ``ReplayBuffer.save``'s format.

    Positions come from real games so the encoding is realistic; the targets are
    a deterministic function of the position (a one-hot policy on the lowest
    legal action, a value read off the score margin), which is what makes "the
    loss goes down" a meaningful assertion rather than a fit to noise.
    """
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    values: list[float] = []
    game = 0
    while len(states) < n:
        state = AzulState.new_game(seed=1000 + game)
        game += 1
        while not state.is_terminal and len(states) < n:
            legal = state.legal_actions()
            policy = np.zeros(ACTION_SPACE, dtype=np.float32)
            policy[min(legal)] = 1.0
            me = state.current_player
            states.append(state.encode())
            policies.append(policy)
            values.append(np.tanh((state.scores[me] - state.scores[1 - me]) / 10.0))
            state.apply(legal[int(rng.integers(len(legal)))])
    buffer = ReplayBuffer(capacity=n)
    buffer.add(
        np.stack(states),
        np.stack(policies),
        np.asarray(values, dtype=np.float32),
    )
    buffer.games_added = game
    return buffer.save(path)


def tiny_config(tmp_path: Path, **overrides) -> TrainConfig:
    data = {
        "run": "pretrain-test",
        "device": "cpu",
        "workers": 1,
        "arch": "structured",
        "embed": 32,
        "layers": 1,
        "heads": 2,
        "body": 64,
        "body_blocks": 1,
        "value_hidden": 16,
        "policy_rank": 8,
        "sims": 4,
        "tree_reuse": True,
        "batch_size": 64,
        "replay_capacity": 2000,
        "min_buffer": 64,
        "games_per_iter": 2,
        "total_games": 2,
        "eval_games": 0,
        "eval_at_start": False,
        "pretrain_epochs": 4,
        "pretrain_lr": 3e-3,
        "heartbeat": 0.0,
    }
    data.update(overrides)
    return TrainConfig.from_dict(data)


def test_pretrain_reads_the_replay_format_and_reduces_the_loss(tmp_path: Path) -> None:
    buffer_path = synthetic_buffer(tmp_path / "replay.npz")
    run_dir = tmp_path / "run"
    trainer = Trainer(tiny_config(tmp_path), run_dir, log=None)
    trainer.prepare()
    steps = trainer.pretrain(buffer_path)
    assert steps > 0
    assert trainer.pretrain_steps == steps
    assert trainer.steps == 0, "pretraining must not advance the self-play schedule"
    assert len(trainer.buffer) == 640, "the positions stay in the buffer (warm start)"

    lines = [
        json.loads(line)
        for line in (run_dir / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert lines and all(entry["phase"] == "pretrain" for entry in lines)
    assert [entry["epoch"] for entry in lines] == [1, 2, 3, 4]
    for entry in lines:  # the schema the dashboard reads
        assert {"t", "games", "steps", "loss", "loss_p", "loss_v", "buffer", "lr"} <= (
            set(entry)
        )
    assert lines[-1]["loss"] < lines[0]["loss"], lines
    assert lines[-1]["loss_v"] < lines[0]["loss_v"]

    # counters survive into the checkpoint
    from ludometer.train.net import load_checkpoint

    payload = load_checkpoint(run_dir / "checkpoints" / "latest.pt")
    assert payload["pretrain_steps"] == steps
    assert payload["net_config"]["arch"] == "structured"


def test_pretrain_can_drop_the_buffer_again(tmp_path: Path) -> None:
    buffer_path = synthetic_buffer(tmp_path / "replay.npz", n=192)
    trainer = Trainer(
        tiny_config(tmp_path, pretrain_epochs=1, pretrain_keep_buffer=False),
        tmp_path / "run",
        log=None,
    )
    trainer.prepare()
    assert trainer.pretrain(buffer_path) > 0
    assert len(trainer.buffer) == 0


def test_pretrain_needs_a_real_file(tmp_path: Path) -> None:
    trainer = Trainer(tiny_config(tmp_path), tmp_path / "run", log=None)
    trainer.prepare()
    with pytest.raises(FileNotFoundError):
        trainer.pretrain(tmp_path / "missing.npz")


def test_config_comments_are_allowed() -> None:
    """configs/run3.json leaves instructions in `_note*` keys."""
    cfg = TrainConfig.from_dict({"run": "x", "_note": "hello", "_note_lr": "why"})
    assert cfg.run == "x"
    with pytest.raises(ValueError, match="unknown config keys"):
        TrainConfig.from_dict({"run": "x", "nope": 1})


def test_shipped_run3_config_loads() -> None:
    cfg = TrainConfig.load(REPO / "configs" / "run3.json")
    assert cfg.arch == "structured"
    assert cfg.sims == 512
    assert cfg.tree_reuse is True
    assert cfg.workers == 8
    assert cfg.total_games == 60_000
    assert cfg.replay_capacity == 500_000
    assert cfg.eval_every_games == 512
    assert cfg.eval_sims == 100
    assert cfg.anchor_elos["random"] == 0.0
    anchor = "mcts:runs/run1/checkpoints/ckpt-024064.pt?sims=100"
    assert cfg.anchor_elos[anchor] == 2014.0
    assert anchor in cfg.eval_anchors
    assert (
        1.0e6
        < sum(
            p.numel()
            for p in __import__("ludometer.train.net", fromlist=["make_net"])
            .make_net(cfg.net_config())
            .parameters()
        )
        < 4.0e6
    )
    # the placeholder the orchestrator has to fill in when run2 finishes
    raw = json.loads((REPO / "configs" / "run3.json").read_text())
    assert "run2" in raw["_note_anchors"]


def test_smoke3_runs_end_to_end_with_pretraining(tmp_path: Path) -> None:
    """The acceptance run: structured net + tree reuse + pretrain, via run.py."""
    buffer_path = synthetic_buffer(tmp_path / "replay.npz", n=256)
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
            str(REPO / "configs" / "smoke3.json"),
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
    run_dir = tmp_path / "runs" / "smoke3"
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "done", status
    assert status["games"] >= 8

    entries = [
        json.loads(line)
        for line in (run_dir / "train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert any(e.get("phase") == "pretrain" for e in entries), "no pretrain phase"
    assert any("phase" not in e for e in entries), "no self-play training happened"
    assert (run_dir / "elo.jsonl").exists()

    # the checkpoint a later run / the GUI would pick up
    from ludometer.train.mcts_agent import MCTSAgent
    from ludometer.train.net2 import StructuredNet

    agent = MCTSAgent.from_checkpoint(
        run_dir / "checkpoints" / "latest.pt", sims=4, seed=1
    )
    assert isinstance(agent.net, StructuredNet)
    state = AzulState.new_game(seed=3)
    assert state.is_legal(agent.act(state))
