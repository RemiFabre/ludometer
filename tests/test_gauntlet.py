"""Tests for the round-robin gauntlet CLI (``ludometer/eval/gauntlet.py``)."""

from __future__ import annotations

import json

import pytest

from ludometer.eval.gauntlet import GauntletSpec, cross_table, main, run_gauntlet


def test_spec_parsing() -> None:
    plain = GauntletSpec.parse("greedy")
    assert (plain.label, plain.spec) == ("greedy", "greedy")
    labelled = GauntletSpec.parse("run1=mcts:runs/run1/checkpoints/x.pt?sims=100")
    assert labelled.label == "run1"
    assert labelled.spec == "mcts:runs/run1/checkpoints/x.pt?sims=100"
    assert labelled.name == "run1"  # what arena.spec_name reports
    with pytest.raises(ValueError, match="bad agent spec"):
        GauntletSpec.parse("=greedy")


def test_spec_builds_a_named_agent() -> None:
    agent = GauntletSpec.parse("baseline=heuristic")()
    assert agent.name == "baseline"


def test_round_robin_and_cross_table() -> None:
    specs = [GauntletSpec.parse(s) for s in ("random", "greedy", "heuristic")]
    results = run_gauntlet(specs, games=2, base_seed=5, workers=1, log=None)
    assert len(results) == 3  # every unordered pair
    names = [s.label for s in specs]
    table = cross_table(results, names)
    assert "greedy" in table and "random" in table
    assert len(table.splitlines()) == len(names) + 1
    # greedy and heuristic both beat random over these two games
    by_pair = {(m.name_a, m.name_b): m for m in results}
    assert by_pair[("random", "greedy")].win_rate <= 0.5


def test_needs_two_agents() -> None:
    with pytest.raises(ValueError, match="at least two"):
        run_gauntlet([GauntletSpec.parse("greedy")], games=2, log=None)


def test_cli_writes_json(tmp_path, capsys) -> None:
    out = tmp_path / "gauntlet.json"
    code = main(
        [
            "greedy",
            "heuristic",
            "--games",
            "2",
            "--seed",
            "1",
            "--nice",
            "0",
            "--anchor",
            "greedy=100",
            "--json",
            str(out),
            "--quiet",
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload["agents"] == {"greedy": "greedy", "heuristic": "heuristic"}
    assert payload["elo"]["greedy"] == pytest.approx(100.0)
    assert payload["matches"][0]["n_games"] == 2
    printed = capsys.readouterr().out
    assert "greedy" in printed and "heuristic" in printed


def test_cli_rejects_a_bad_anchor() -> None:
    with pytest.raises(SystemExit, match="not one of the agents"):
        main(["greedy", "heuristic", "--games", "2", "--nice", "0", "--anchor", "x=1"])
