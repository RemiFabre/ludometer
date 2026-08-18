"""Round-robin tournament between arbitrary agents, at play-time settings.

    uv run python -m ludometer.eval.gauntlet --games 40 --workers 8 \
        greedy heuristic \
        run1=mcts:runs/run1/checkpoints/ckpt-024064.pt?sims=100 \
        run2=best?sims=400 \
        run2-1s=mcts:runs/run2/checkpoints/latest.pt?think=1.0

Why this exists next to :mod:`ludometer.eval.arena` and the trainer's own Elo
pass: the trainer rates *one* checkpoint against a fixed pool at a fixed sim
count, because that is what makes an Elo-vs-compute curve comparable. A gauntlet
answers the other question — "which of these actually plays better, at the
settings a human or a tournament would use" — so every agent here is a full spec
including its search budget: ``?sims=<n>`` for a fixed sim count or
``?think=<seconds>`` for a wall-clock budget per move (the ``time_limit_s`` path
in :class:`~ludometer.train.mcts_agent.MCTSAgent`). The same checkpoint at two
budgets is simply two entries, which is how you measure what thinking longer is
worth in Elo.

Each agent is ``[label=]spec``; the label is what shows up in the cross table.
Output is a cross table of score shares plus a Bradley-Terry Elo fit
(:func:`ludometer.eval.elo.fit_elo`) with any ``--anchor NAME=ELO`` held fixed,
so gauntlet ratings can be put on the same scale as a run's curve.

It runs **niced** by default (``--nice 19``): a gauntlet is never more urgent
than the training run it is measuring.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ludometer.agents import Agent, make_agent
from ludometer.eval.arena import MatchResult, play_match
from ludometer.eval.elo import PairResult, fit_elo

__all__ = ["GauntletSpec", "cross_table", "main", "run_gauntlet"]


@dataclass(frozen=True)
class GauntletSpec:
    """Picklable ``label + spec string`` factory (one agent per worker game)."""

    label: str
    spec: str
    seed: int | None = None
    threads: int = 1

    @property
    def name(self) -> str:  # what `arena.spec_name` reports
        return self.label

    @classmethod
    def parse(cls, text: str, seed: int | None = None) -> GauntletSpec:
        """``"label=spec"`` or just ``"spec"`` (the spec is then the label)."""
        label, sep, spec = text.partition("=")
        if not sep:
            return cls(label=text, spec=text, seed=seed)
        if not label or not spec:
            raise ValueError(f"bad agent spec {text!r} (expected label=spec)")
        return cls(label=label, spec=spec, seed=seed)

    def __call__(self) -> Agent:
        with contextlib.suppress(ImportError):  # torch is optional for baselines
            import torch

            torch.set_num_threads(self.threads)
        from ludometer.agents.registry import load_agent

        try:
            agent = load_agent(self.spec, seed=self.seed)
        except ValueError:
            agent = make_agent(self.spec)
        agent.name = self.label
        return agent


def cross_table(results: list[MatchResult], names: list[str]) -> str:
    """Score share of the row agent against the column agent (draws = 0.5)."""
    share: dict[tuple[str, str], float] = {}
    played: dict[tuple[str, str], int] = {}
    for m in results:
        share[(m.name_a, m.name_b)] = m.win_rate
        share[(m.name_b, m.name_a)] = 1.0 - m.win_rate
        played[(m.name_a, m.name_b)] = m.n_games
        played[(m.name_b, m.name_a)] = m.n_games
    width = max(max((len(n) for n in names), default=6), 6)
    head = " " * (width + 2) + "".join(f"{n[:7]:>8}" for n in names) + "   total"
    lines = [head]
    for row in names:
        cells = []
        total = 0.0
        games = 0
        for col in names:
            if row == col:
                cells.append(f"{'-':>8}")
                continue
            value = share.get((row, col))
            if value is None:
                cells.append(f"{'.':>8}")
                continue
            cells.append(f"{value:8.2f}")
            total += value * played[(row, col)]
            games += played[(row, col)]
        rate = total / games if games else 0.0
        lines.append(f"{row:<{width}}  " + "".join(cells) + f"   {rate:5.2f}")
    return "\n".join(lines)


def run_gauntlet(
    specs: list[GauntletSpec],
    games: int = 20,
    base_seed: int = 0,
    workers: int = 1,
    log: Any = print,
    game: str = "azul",
) -> list[MatchResult]:
    """Every unordered pair plays ``games`` alternating-seat games."""
    if len(specs) < 2:
        raise ValueError("a gauntlet needs at least two agents")
    out: list[MatchResult] = []
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            seed = base_seed + 1_000_000 * (i * len(specs) + j)
            match = play_match(
                specs[i],
                specs[j],
                n_games=games,
                base_seed=seed,
                n_workers=workers,
                game=game,
            )
            if log is not None:
                log(
                    f"{match.name_a} vs {match.name_b}: "
                    f"{match.wins}-{match.draws}-{match.losses} "
                    f"({match.win_rate:.2f}) "
                    f"scores {match.mean_score_a:.1f}-{match.mean_score_b:.1f}"
                )
            out.append(match)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ludometer.eval.gauntlet",
        description="round-robin between agent specs, with a cross table and Elo",
    )
    parser.add_argument("agents", nargs="+", help="[label=]spec, at least two")
    parser.add_argument("--games", type=int, default=20, help="games per pairing")
    parser.add_argument("--workers", type=int, default=1, help="worker processes")
    parser.add_argument("--seed", type=int, default=0, help="base seed")
    parser.add_argument(
        "--game", default="azul", help="rules engine to play (ludometer/games.py)"
    )
    parser.add_argument(
        "--nice", type=int, default=19, help="niceness to run at (0 = leave alone)"
    )
    parser.add_argument(
        "--anchor",
        action="append",
        default=[],
        metavar="NAME=ELO",
        help="hold an agent's rating fixed (repeatable)",
    )
    parser.add_argument("--json", type=Path, help="also write the results here")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.nice:
        with contextlib.suppress(OSError):
            os.nice(args.nice)
    specs = [GauntletSpec.parse(text, seed=args.seed) for text in args.agents]
    names = [s.label for s in specs]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate agent labels: {names}")
    anchors: dict[str, float] = {}
    for item in args.anchor:
        key, _, value = item.partition("=")
        if key not in names:
            raise SystemExit(f"anchor {key!r} is not one of the agents: {names}")
        anchors[key] = float(value)

    log = None if args.quiet else print
    results = run_gauntlet(
        specs,
        games=args.games,
        base_seed=args.seed,
        workers=args.workers,
        log=log,
        game=args.game,
    )
    pairs = [PairResult(m.name_a, m.name_b, m.wins, m.draws, m.losses) for m in results]
    fit = fit_elo(pairs, anchors=anchors, error_method="fisher")
    print()
    print(cross_table(results, names))
    print()
    print(fit.table())
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "agents": {s.label: s.spec for s in specs},
                    "games": args.games,
                    "matches": [m.as_dict() for m in results],
                    "elo": fit.ratings,
                    "elo_err": fit.errors,
                    "anchors": anchors,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
