"""Tests for the ``best`` agent spec: "play the strongest checkpoint on disk".

Everything runs against a fake ``runs/`` tree (``LUDOMETER_RUNS_DIR``) holding
hand-written ``elo.jsonl`` files and *real* — but deliberately tiny — checkpoints,
so the resolution logic is tested without touching a live training run and
without a 512-hidden net.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ludometer.agents.registry import (
    BEST_SIMS,
    find_best_checkpoint,
    load_agent,
    runs_dir,
)
from ludometer.gui.server import best_entry, create_app
from ludometer.gui.session import GameSession
from ludometer.train.net import NetConfig, PolicyValueNet, save_checkpoint

# the smallest net the config allows: one hidden unit, no residual blocks
TINY = NetConfig(hidden=1, blocks=0, value_hidden=1)


def write_ckpt(run: Path, name: str) -> Path:
    """Save a real (tiny) checkpoint at ``run/checkpoints/<name>.pt``."""
    return save_checkpoint(run / "checkpoints" / f"{name}.pt", PolicyValueNet(TINY))


def write_elo(run: Path, records: list[dict | str]) -> None:
    """Append ``records`` (dicts, or raw strings for torn lines) to elo.jsonl."""
    run.mkdir(parents=True, exist_ok=True)
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    (run / "elo.jsonl").write_text("\n".join(lines) + "\n")


def elo_record(name: str, elo: float) -> dict:
    return {"t": 1.0, "games": 512, "ckpt": name, "elo": elo, "elo_err": 5.0}


@pytest.fixture
def fake_runs(tmp_path, monkeypatch):
    """A runs/ tree with two runs; the best *existing* ckpt is run2/ckpt-000512."""
    root = tmp_path / "runs"
    run1, run2 = root / "run1", root / "run2"
    write_elo(
        run1,
        [
            elo_record("ckpt-000000", 100.0),
            elo_record("ckpt-000512", 900.0),
            elo_record("ckpt-001024", 5000.0),  # rated but pruned from disk
        ],
    )
    write_elo(
        run2,
        [
            elo_record("ckpt-000000", 50.0),
            elo_record("ckpt-000512", 1234.5),
            '{"ckpt": "ckpt-001024", "elo": 99',  # torn line, trainer still writing
            json.dumps({"ckpt": "ckpt-001536"}),  # no elo
            json.dumps({"elo": 9999.0}),  # no ckpt
        ],
    )
    for run in (run1, run2):
        write_ckpt(run, "ckpt-000000")
        write_ckpt(run, "ckpt-000512")
    monkeypatch.setenv("LUDOMETER_RUNS_DIR", str(root))
    return root


# ------------------------------------------------------------------ resolution
def test_runs_dir_honours_the_env_override(tmp_path, monkeypatch):
    monkeypatch.delenv("LUDOMETER_RUNS_DIR", raising=False)
    assert runs_dir().name == "runs"
    assert runs_dir().parent.joinpath("pyproject.toml").is_file()
    monkeypatch.setenv("LUDOMETER_RUNS_DIR", str(tmp_path / "elsewhere"))
    assert runs_dir() == tmp_path / "elsewhere"


def test_find_best_checkpoint_picks_the_highest_rated_file_on_disk(fake_runs):
    best = find_best_checkpoint()
    assert best.run == "run2"
    assert best.ckpt == "ckpt-000512"
    assert best.elo == pytest.approx(1234.5)
    assert best.path == fake_runs / "run2" / "checkpoints" / "ckpt-000512.pt"
    assert best.path.is_file()


def test_an_explicit_root_beats_the_env_var(fake_runs, tmp_path, monkeypatch):
    monkeypatch.setenv("LUDOMETER_RUNS_DIR", str(tmp_path / "nothing"))
    assert find_best_checkpoint(fake_runs).ckpt == "ckpt-000512"
    with pytest.raises(FileNotFoundError):
        find_best_checkpoint()


def test_a_newer_stronger_checkpoint_wins_as_the_run_progresses(fake_runs):
    assert find_best_checkpoint().ckpt == "ckpt-000512"
    run3 = fake_runs / "run3"
    write_elo(run3, [elo_record("ckpt-002048", 1800.0)])
    write_ckpt(run3, "ckpt-002048")
    best = find_best_checkpoint()
    assert (best.run, best.ckpt, best.elo) == ("run3", "ckpt-002048", 1800.0)


def test_no_checkpoints_is_a_clear_error(tmp_path, monkeypatch):
    empty = tmp_path / "runs"
    monkeypatch.setenv("LUDOMETER_RUNS_DIR", str(empty))
    for _ in range(2):  # missing dir, then an empty one
        with pytest.raises(FileNotFoundError) as excinfo:
            find_best_checkpoint()
        message = str(excinfo.value)
        assert "no rated checkpoints" in message
        assert str(empty) in message
        empty.mkdir(exist_ok=True)
    # ratings exist but the .pt files are gone
    write_elo(empty / "run1", [elo_record("ckpt-000000", 100.0)])
    with pytest.raises(FileNotFoundError):
        find_best_checkpoint()
    with pytest.raises(FileNotFoundError):
        load_agent("best")


# ----------------------------------------------------------------- load_agent
def test_load_agent_best_wraps_the_checkpoint_in_an_mcts_agent(fake_runs):
    agent = load_agent("best", seed=3)
    assert agent.name == "best:ckpt-000512"
    assert agent.mcts.config.sims == BEST_SIMS == 400
    info = agent.spec_info
    assert info["kind"] == "best"
    assert info["checkpoint"] == "ckpt-000512"
    assert info["run"] == "run2"
    assert info["elo"] == pytest.approx(1234.5)
    assert info["sims"] == 400
    assert info["path"].endswith("run2/checkpoints/ckpt-000512.pt")
    assert (
        info["resolved_spec"].startswith("mcts:")
        and "sims=400" in info["resolved_spec"]
    )


def test_best_accepts_a_sims_option(fake_runs):
    for sims in (1, 4, 1200):
        agent = load_agent(f"best?sims={sims}")
        assert agent.mcts.config.sims == sims
        assert agent.spec_info["sims"] == sims
    assert load_agent("  best?sims=8  ").mcts.config.sims == 8


def test_best_rejects_bad_options(fake_runs):
    for spec in (
        "best?sims=0",
        "best?sims=-2",
        "best?sims=abc",
        "best?depth=2",
        "bestest",
    ):
        with pytest.raises(ValueError):
            load_agent(spec)


def test_mcts_specs_still_carry_their_resolved_info(fake_runs):
    path = fake_runs / "run1" / "checkpoints" / "ckpt-000000.pt"
    agent = load_agent(f"mcts:{path}?sims=3")
    assert agent.mcts.config.sims == 3
    assert agent.spec_info["kind"] == "mcts"
    assert agent.spec_info["checkpoint"] == "ckpt-000000"


# ------------------------------------------------------------------------ GUI
def test_best_entry_describes_the_resolved_checkpoint(fake_runs):
    entry = best_entry()
    assert entry["available"] is True
    assert entry["spec"] == "best"
    assert entry["label"] == "Strongest trained (auto)"
    assert entry["checkpoint"] == "ckpt-000512"
    assert entry["run"] == "run2"
    assert entry["elo"] == pytest.approx(1234.5)
    assert "ckpt-000512" in entry["detail"] and "Elo" in entry["detail"]
    assert entry["sims_choices"] == [100, 400, 1200]
    assert entry["default_sims"] == 400


def test_best_entry_is_unavailable_without_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("LUDOMETER_RUNS_DIR", str(tmp_path / "nothing"))
    entry = best_entry()
    assert entry["available"] is False
    assert "no rated checkpoints" in entry["error"]
    assert "checkpoint" not in entry


def test_agents_endpoint_defaults_to_best_when_available(fake_runs):
    app = create_app()
    app.testing = True
    data = app.test_client().get("/api/agents").get_json()
    assert data["default"] == "best"
    assert data["fallback_default"] == "heuristic"
    assert data["best"]["checkpoint"] == "ckpt-000512"
    assert set(data["baselines"]) == {"heuristic", "greedy", "random"}


def test_agents_endpoint_falls_back_to_the_heuristic(tmp_path, monkeypatch):
    monkeypatch.setenv("LUDOMETER_RUNS_DIR", str(tmp_path / "nothing"))
    app = create_app()
    app.testing = True
    data = app.test_client().get("/api/agents").get_json()
    assert data["default"] == "heuristic"
    assert data["best"]["available"] is False


def test_a_new_game_against_best_names_the_checkpoint(fake_runs):
    app = create_app()
    app.testing = True
    client = app.test_client()
    response = client.post(
        "/api/new",
        json={"opponent_spec": "best?sims=2", "human_plays_first": True, "seed": 1},
    )
    assert response.status_code == 200, response.get_json()
    data = response.get_json()
    assert data["agent_name"] == "best:ckpt-000512"
    assert data["opponent_info"]["checkpoint"] == "ckpt-000512"
    assert data["opponent_info"]["sims"] == 2
    blurb = data["opponent_blurb"]
    assert "ckpt-000512" in blurb and "+1234" in blurb and "internal ladder" in blurb
    assert any(entry["text"] == blurb for entry in data["log"])

    # ...and it actually plays: two human moves, each answered by the net
    for _ in range(2):
        legal = data["human_legal_actions"]
        assert legal
        response = client.post("/api/act", json={"action_id": legal[0]})
        assert response.status_code == 200, response.get_json()
        data = response.get_json()
        assert data["ai_moves"] and data["ai_moves"][0]["side"] == "ai"


def test_session_without_a_checkpoint_opponent_has_no_blurb():
    session = GameSession("heuristic", seed=2)
    assert session.opponent_info == {}
    assert session.opponent_blurb == ""
    assert session.snapshot()["opponent_blurb"] is None
    assert session.snapshot()["opponent_info"] is None
