"""Compare two runs' Elo-vs-games curves (the pretrain A/B readout).

    uv run python -m ludometer.eval.compare_runs runs/hp1 runs/run6

Both runs rate their checkpoints against the same anchor pool during training
(``elo.jsonl``), so at equal self-play game counts the ratings are comparable.
Prints a side-by-side table plus the mean gap over the shared range.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def curve(run_dir: Path) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    path = run_dir / "elo.jsonl"
    if not path.exists():
        raise SystemExit(f"no elo.jsonl in {run_dir}")
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out[int(d["games"])] = (float(d["elo"]), float(d.get("elo_err", 0.0)))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ludometer.eval.compare_runs")
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    args = parser.parse_args(argv)
    a, b = curve(args.run_a), curve(args.run_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        print("no shared game counts yet")
        print(f"{args.run_a.name}: {sorted(a)}")
        print(f"{args.run_b.name}: {sorted(b)}")
        return 1
    print(f"{'games':>7} {args.run_a.name:>16} {args.run_b.name:>16} {'gap':>8}")
    gaps = []
    for g in shared:
        ea, err_a = a[g]
        eb, err_b = b[g]
        gaps.append(ea - eb)
        sig = "*" if abs(ea - eb) > (err_a + err_b) else " "
        print(
            f"{g:>7} {ea:>9.1f} ±{err_a:<4.0f} {eb:>9.1f} ±{err_b:<4.0f} "
            f"{ea - eb:>+7.1f}{sig}"
        )
    print(
        f"mean gap over {len(shared)} shared points: "
        f"{sum(gaps) / len(gaps):+.1f} Elo ({args.run_a.name} minus {args.run_b.name}); "
        "* = outside both error bars"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
