"""End-to-end trainer test: run ``configs/smoke.json`` for real, in a subprocess.

This is the test that guards the contract the dashboard depends on: the exact
``runs/<run>/`` layout and the exact JSON schemas from docs/DESIGN.md. It also
covers the two lifecycle paths that matter — a clean stop at ``--max-games`` and
a SIGINT mid-run followed by ``--resume`` continuing the counters.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

from ludometer.train.trainer import TrainConfig

REPO = Path(__file__).resolve().parents[1]
SMOKE = REPO / "configs" / "smoke.json"
STATUS_KEYS = {
    "run",
    "state",
    "started",
    "updated",
    "ended",
    "error",
    "games",
    "steps",
    "note",
}
TRAIN_KEYS = {
    "t",
    "games",
    "steps",
    "loss",
    "loss_p",
    "loss_v",
    "loss_m",  # run4's margin head; 0.0 for a net without one
    "loss_a",  # run6's final-wall heads; 0.0 for a net without them
    "buffer",
    "lr",
}
ELO_KEYS = {
    "t",
    "games",
    "positions",  # cumulative practice, both units (docs/NEXT_GAMES.md §4)
    "decisions",
    "ckpt",
    "elo",
    "elo_err",
    "vs",
    "n_games",
    "pool",
}


# ------------------------------------------------------------------- configs
def test_shipped_configs_parse() -> None:
    # Every config in the repo must load: a stale key (configs/uno_smoke.json
    # once shipped a leftover "segment_value_weight") fails at launch time.
    for path in sorted((REPO / "configs").glob("*.json")):
        if path.name.endswith("_net.json"):
            continue  # a bare net config (distill --student-config), not a run
        cfg = TrainConfig.load(path)
        cfg.validate()
        assert cfg.run
        assert cfg.sims >= 1
        assert cfg.net_config() is not None  # arch keys validated by the net config
        assert cfg.selfplay_config().mcts.sims == cfg.sims


def test_unknown_config_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown config keys"):
        TrainConfig.from_dict({"run": "x", "hiden": 512})


# --------------------------------------------------------------------- helpers
def run_trainer(args: list[str], timeout: float = 300.0) -> subprocess.CompletedProcess:
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


def check_iso(stamp: str) -> None:
    datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")  # noqa: DTZ007 - UTC by construction


def check_status(payload: dict, run: str) -> None:
    assert set(payload) == STATUS_KEYS
    assert payload["run"] == run
    assert payload["state"] in ("running", "done", "failed")
    check_iso(payload["started"])
    check_iso(payload["updated"])
    if payload["state"] != "running":
        check_iso(payload["ended"])
    assert isinstance(payload["games"], int)
    assert isinstance(payload["steps"], int)
    assert isinstance(payload["note"], str)


def check_train_lines(lines: list[dict]) -> None:
    assert lines
    for row in lines:
        assert set(row) == TRAIN_KEYS
        assert row["loss"] == pytest.approx(
            row["loss_p"] + row["loss_v"] + row["loss_m"] + row["loss_a"], abs=1e-3
        )
        assert row["loss_m"] == 0.0, "no margin head in this config"
        assert row["loss_a"] == 0.0, "no aux heads in this config"
        assert row["buffer"] > 0
        assert 0.0 < row["lr"] <= 1.0
        assert row["t"] >= 0.0
    games = [row["games"] for row in lines]
    steps = [row["steps"] for row in lines]
    assert games == sorted(games)
    assert steps == sorted(steps)


def check_elo_lines(lines: list[dict], run_games: int) -> None:
    assert lines
    for row in lines:
        assert set(row) == ELO_KEYS
        assert isinstance(row["vs"], dict)
        assert row["vs"], "no opponents recorded"
        assert all(0.0 <= v <= 1.0 for v in row["vs"].values())
        assert isinstance(row["pool"], list)
        assert any(entry.startswith("random=") for entry in row["pool"])
        assert set(row["vs"]) <= {p.split("=")[0] for p in row["pool"]}
        assert row["n_games"] > 0
        assert row["ckpt"].startswith("ckpt-")
        assert isinstance(row["elo"], float)
        assert row["elo_err"] >= 0.0
        assert row["games"] <= run_games


# ---------------------------------------------------------------- end to end
def test_smoke_run_then_resume(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    args = ["--config", str(SMOKE), "--runs-dir", str(runs), "--run", "smoke"]
    first = run_trainer([*args, "--max-games", "8"])
    assert first.returncode == 0, first.stderr[-4000:]
    run_dir = runs / "smoke"
    assert run_dir.is_dir()

    config = json.loads((run_dir / "config.json").read_text())
    assert config["run"] == "smoke"
    check_iso(config["started"])
    assert config["sims"] == 16

    status = json.loads((run_dir / "status.json").read_text())
    check_status(status, "smoke")
    assert status["state"] == "done"
    assert status["error"] is None
    assert status["games"] == 8

    train = read_jsonl(run_dir / "train.jsonl")
    check_train_lines(train)
    elo = read_jsonl(run_dir / "elo.jsonl")
    check_elo_lines(elo, status["games"])
    assert len(elo) >= 2, "expect the initial net plus one trained checkpoint"
    assert elo[0]["games"] == 0

    ckpts = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
    assert "latest.pt" in ckpts
    assert any(name.startswith("ckpt-") for name in ckpts)
    assert (run_dir / "checkpoints" / "replay.npz").exists()

    # ---- resume: counters continue, nothing is reset
    second = run_trainer(["--resume", str(run_dir), "--max-games", "16"])
    assert second.returncode == 0, second.stderr[-4000:]
    assert "resumed" in second.stdout
    status2 = json.loads((run_dir / "status.json").read_text())
    check_status(status2, "smoke")
    assert status2["games"] == 16
    assert status2["steps"] > status["steps"]
    assert status2["started"] == status["started"]  # same run, same start time

    train2 = read_jsonl(run_dir / "train.jsonl")
    check_train_lines(train2)
    assert len(train2) > len(train)
    assert train2[: len(train)] == train  # append-only
    assert train2[-1]["t"] > train[-1]["t"]  # elapsed time carried across resume
    elo2 = read_jsonl(run_dir / "elo.jsonl")
    check_elo_lines(elo2, status2["games"])
    assert len(elo2) > len(elo)


def test_sigint_stops_cleanly_and_resumes(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run_dir = runs / "irq"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "ludometer.train.run",
            "--config",
            str(SMOKE),
            "--runs-dir",
            str(runs),
            "--run",
            "irq",
            "--max-games",
            "24",
        ],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    status_path = run_dir / "status.json"
    deadline = time.monotonic() + 120
    games = 0
    try:
        while time.monotonic() < deadline:
            if status_path.exists():
                try:
                    payload = json.loads(status_path.read_text())
                except json.JSONDecodeError:  # pragma: no cover - mid-write
                    payload = {}
                games = payload.get("games", 0)
                if games >= 8:
                    break
            if proc.poll() is not None:  # pragma: no cover - died early
                break
            time.sleep(0.5)
        assert games >= 8, "trainer never reported a finished self-play batch"
        proc.send_signal(signal.SIGINT)
        out, err = proc.communicate(timeout=120)
    finally:
        if proc.poll() is None:  # pragma: no cover - defensive
            proc.kill()
            proc.communicate()
    assert proc.returncode == 0, err[-4000:]
    assert "SIGINT" in out or "SIGINT" in err

    status = json.loads(status_path.read_text())
    check_status(status, "irq")
    assert status["state"] == "done"
    assert status["error"] is None
    assert (run_dir / "checkpoints" / "latest.pt").exists()

    interrupted_games = status["games"]
    resumed = run_trainer(
        ["--resume", str(run_dir), "--max-games", str(interrupted_games + 8)]
    )
    assert resumed.returncode == 0, resumed.stderr[-4000:]
    final = json.loads(status_path.read_text())
    assert final["games"] == interrupted_games + 8
    assert final["steps"] >= status["steps"]
    assert final["started"] == status["started"]
