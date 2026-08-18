"""%-optimal evaluation of a run's checkpoints against a solved suite.

For solved games the headline chart is % optimal play, with Elo secondary
(NEXT_GAMES.md §3). This walks ``runs/<run>/checkpoints/ckpt-*.pt`` in game
order, plays every suite position once with the same MCTS configuration the
Elo evaluator uses, and appends one record per checkpoint to
``runs/<run>/optimal.jsonl``::

    {"ckpt": "ckpt-001024", "games": 1024, "pct_optimal": 0.84,
     "blunder_rate": 0.11, "n": 2000, "won_positions": 812, "sims": 64}

Baselines can be rated too (they become the reference lines on the chart)::

    uv run python -m ludometer.eval.optimal --run runs/ttt1
    uv run python -m ludometer.eval.optimal --game tictactoe --agents \\
        ttt:random ttt:greedy ttt:heuristic
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from ludometer.agents import make_agent
from ludometer.solved.suite import load_suite, score_agent, suite_path
from ludometer.train.mcts_agent import MCTSAgent

__all__ = ["evaluate_run", "score_spec"]


def score_spec(
    spec: Any, suite: dict[str, Any], sims: int = 64, seed: int = 0
) -> dict[str, Any]:
    """Score one agent spec (baseline name or checkpoint path) on ``suite``."""
    if isinstance(spec, (str, Path)) and str(spec).endswith(".pt"):
        agent = MCTSAgent.from_checkpoint(str(spec), sims=sims, seed=seed)
    else:
        agent = make_agent(spec)
        agent.seed(seed)
    return score_agent(agent, suite)


def evaluate_run(
    run_dir: str | Path,
    suite: dict[str, Any],
    sims: int = 64,
    log: Any = print,
) -> list[dict[str, Any]]:
    """Score every ``ckpt-*.pt`` of a run; append new rows to optimal.jsonl."""
    run_dir = Path(run_dir)
    out_path = run_dir / "optimal.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["ckpt"])
    rows = []
    checkpoints = sorted(run_dir.glob("checkpoints/ckpt-*.pt"))
    for path in checkpoints:
        name = path.stem
        if name in done:
            continue
        games = int(re.sub(r"\D", "", name) or 0)
        t0 = time.monotonic()
        score = score_spec(path, suite, sims=sims)
        row = {
            "ckpt": name,
            "games": games,
            "pct_optimal": round(score["pct_optimal"], 4),
            "blunder_rate": round(score["blunder_rate"], 4),
            "n": score["n"],
            "won_positions": score["won_positions"],
            "sims": sims,
        }
        rows.append(row)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        if log:
            log(
                f"{name}: {row['pct_optimal']:.1%} optimal, "
                f"{row['blunder_rate']:.1%} blunders "
                f"({time.monotonic() - t0:.0f}s)"
            )
    return rows


def main() -> None:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="runs/<run> to walk")
    parser.add_argument("--game", default=None, help="suite to load if no --run")
    parser.add_argument("--suite", default=None, help="explicit suite path")
    parser.add_argument("--sims", type=int, default=64)
    parser.add_argument("--agents", nargs="*", default=[], help="baseline specs")
    args = parser.parse_args()

    game = args.game
    if args.run and not game:
        config = json.loads((Path(args.run) / "config.json").read_text())
        game = (config.get("eval_game") or config["game"]).replace("_hand", "")
    suite = load_suite(args.suite or suite_path(game))
    for spec in args.agents:
        score = score_spec(spec, suite, sims=args.sims)
        print(
            f"{spec}: {score['pct_optimal']:.1%} optimal, "
            f"{score['blunder_rate']:.1%} blunders over {score['n']}"
        )
    if args.run:
        evaluate_run(args.run, suite, sims=args.sims)


if __name__ == "__main__":  # pragma: no cover
    main()
