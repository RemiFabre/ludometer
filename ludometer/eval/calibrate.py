"""Calibrate the baseline ladder: Random vs Greedy vs Heuristic.

    uv run python -m ludometer.eval.calibrate [--games 300] [--workers 10]

Plays a full round robin (``--games`` games per pairing, seats alternating), fits
Bradley-Terry ratings with Random pinned at 0 Elo (docs/DESIGN.md) and prints the
cross table plus the ratings. This is the fixed anchor pool the training curves
are measured against, so its numbers should be stable across runs.
"""

from __future__ import annotations

import argparse
import json
import time

from ludometer.eval.arena import MatchResult, round_robin
from ludometer.eval.elo import EloFit, fit_elo

DEFAULT_POOL = ["random", "greedy", "heuristic"]


def cross_table(names: list[str], matches: list[MatchResult]) -> str:
    """Score share of the row agent against the column agent."""
    share: dict[tuple[str, str], float] = {}
    diff: dict[tuple[str, str], float] = {}
    n_games: dict[tuple[str, str], int] = {}
    for m in matches:
        share[m.name_a, m.name_b] = m.win_rate
        share[m.name_b, m.name_a] = 1.0 - m.win_rate
        diff[m.name_a, m.name_b] = m.mean_score_diff
        diff[m.name_b, m.name_a] = -m.mean_score_diff
        n_games[m.name_a, m.name_b] = m.n_games
        n_games[m.name_b, m.name_a] = m.n_games

    width = max(max(len(n) for n in names), 9)
    head = " " * (width + 2) + "".join(f"{n:>12}" for n in names)
    lines = ["Win rate (row vs column; draws count 1/2)", head]
    for row in names:
        cells = []
        for col in names:
            cells.append("       -    " if row == col else f"{share[row, col]:>12.3f}")
        lines.append(f"  {row:<{width}}" + "".join(cells))
    lines += ["", "Mean score difference (row minus column)", head]
    for row in names:
        cells = []
        for col in names:
            cells.append("       -    " if row == col else f"{diff[row, col]:>+12.2f}")
        lines.append(f"  {row:<{width}}" + "".join(cells))
    played = sorted({v for v in n_games.values()})
    lines.append("")
    lines.append(f"games per pairing: {played}")
    return "\n".join(lines)


def calibrate(
    pool: list[str] | None = None,
    n_games: int = 300,
    n_workers: int = 10,
    base_seed: int = 20250814,
    error_method: str = "fisher",
    n_bootstrap: int = 200,
) -> tuple[list[MatchResult], EloFit]:
    """Round robin over ``pool`` plus an Elo fit anchored on Random at 0."""
    names = list(pool or DEFAULT_POOL)
    matches = round_robin(
        names, n_games=n_games, base_seed=base_seed, n_workers=n_workers
    )
    anchors = {"random": 0.0} if "random" in names else {}
    fit = fit_elo(
        matches,
        anchors=anchors,
        error_method=error_method,
        n_bootstrap=n_bootstrap,
        seed=base_seed,
    )
    return matches, fit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=300, help="games per pairing")
    ap.add_argument("--workers", type=int, default=10, help="worker processes")
    ap.add_argument("--seed", type=int, default=20250814)
    ap.add_argument(
        "--agents", nargs="*", default=DEFAULT_POOL, help="agent names to include"
    )
    ap.add_argument(
        "--errors",
        choices=("fisher", "bootstrap", "none"),
        default="fisher",
        help="how to estimate the Elo error bars",
    )
    ap.add_argument("--bootstrap", type=int, default=200, help="bootstrap resamples")
    ap.add_argument("--json", action="store_true", help="also dump JSON")
    args = ap.parse_args()

    t0 = time.perf_counter()
    matches, fit = calibrate(
        pool=args.agents,
        n_games=args.games,
        n_workers=args.workers,
        base_seed=args.seed,
        error_method=args.errors,
        n_bootstrap=args.bootstrap,
    )
    elapsed = time.perf_counter() - t0

    print(cross_table(list(args.agents), matches))
    print()
    print("Elo (Bradley-Terry ML fit, random anchored at 0)")
    print(fit.table())
    print()
    for m in matches:
        print(
            f"  {m.name_a} vs {m.name_b}: {m.wins}-{m.draws}-{m.losses} "
            f"(W-D-L), score share {m.win_rate:.3f}, "
            f"mean score {m.mean_score_a:.1f} vs {m.mean_score_b:.1f}, "
            f"{m.mean_moves:.0f} moves/game"
        )
    total = sum(m.n_games for m in matches)
    print()
    print(
        f"{total} games in {elapsed:.1f}s "
        f"({total / elapsed:.1f} games/s, {args.workers} workers), "
        f"converged={fit.converged} in {fit.iterations} iterations"
    )
    if args.json:
        print(
            json.dumps(
                {
                    "matches": [m.as_dict() for m in matches],
                    "elo": fit.ratings,
                    "elo_err": fit.errors,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
