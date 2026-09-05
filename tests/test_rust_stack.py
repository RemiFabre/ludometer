"""The full stack on the Rust engine (docs/RUST_ENGINE.md §6, layer 4).

* ``configs/smoke5_rust.json`` (smoke5 with ``"selfplay": "rust"``) trains end
  to end in a subprocess, resumes, and leaves the same artefacts smoke5 does;
* the hub loop (``tests/test_cloud.py``'s trainer + generator pair) runs with the
  generator on ``--engine rust``.

Both are slow (a minute or two each) and skipped when ``ludometer_rs`` is not
built (`rust/README.md` says how).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("ludometer_rs")

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "configs" / "smoke5_rust.json"


def run_trainer(args: list[str], timeout: float = 600.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ludometer.train.run", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_smoke5_rust_config_parses() -> None:
    from ludometer.train.trainer import TrainConfig

    data = json.loads(SMOKE.read_text())
    cfg = TrainConfig.from_dict(
        {k: v for k, v in data.items() if not k.startswith("_")}
    )
    assert cfg.selfplay == "rust"
    base = json.loads((REPO / "configs" / "smoke5.json").read_text())
    for k, v in base.items():
        if k in ("run", "note", "_note_engine", "selfplay"):
            continue
        assert data[k] == v, f"smoke5_rust differs from smoke5 on {k}"


def test_smoke5_rust_runs_end_to_end_and_resumes(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    args = ["--config", str(SMOKE), "--runs-dir", str(runs), "--run", "smoke5_rust"]
    first = run_trainer([*args, "--max-games", "16"])
    assert first.returncode == 0, first.stderr[-4000:]
    assert "selfplay=rust(" in first.stdout
    run_dir = runs / "smoke5_rust"
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "done" and status["error"] is None
    assert status["games"] == 16
    train = read_jsonl(run_dir / "train.jsonl")
    assert train and train[-1]["buffer"] > 0
    assert all(row["loss_m"] > 0.0 for row in train), (
        "margin head trains on rust records"
    )
    elo = read_jsonl(run_dir / "elo.jsonl")
    assert len(elo) >= 2
    assert (run_dir / "checkpoints" / "replay.npz").exists()

    second = run_trainer(["--resume", str(run_dir), "--max-games", "24"])
    assert second.returncode == 0, second.stderr[-4000:]
    status2 = json.loads((run_dir / "status.json").read_text())
    assert status2["games"] == 24 and status2["steps"] > status["steps"]


def test_hub_loop_with_the_rust_generator(tmp_path: Path) -> None:
    data = json.loads((REPO / "configs" / "smoke5.json").read_text())
    data.update(
        {
            "run": "hubrust",
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
    cfg_path = tmp_path / "hubrust.json"
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
            "hubrust",
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
            "--engine",
            "rust",
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
    assert "rust drivers" in g_out
    status = json.loads((tmp_path / "runs" / "hubrust" / "status.json").read_text())
    assert status["state"] == "done" and status["games"] == 8
