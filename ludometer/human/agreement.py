"""How much do our nets agree with elite human moves? (diagnostic, no training)

    uv run python -m ludometer.human.agreement \
        --npz data/human/replay.npz --stats data/human/replay.stats.json \
        run4=runs/run4/checkpoints/ckpt-037888.pt \
        run5=runs/run5/checkpoints/ckpt-006912.pt \
        run6=runs/run6/checkpoints/ckpt-009984.pt

For every stored position that carries a policy target (``policy_mask == 1`` —
in the human datasets these are exactly the elite player's moves, stored as
one-hot targets), each net's raw policy head is scored against the human's
actual move: top-1 / top-3 agreement and mean NLL (``-log softmax[move]`` over
the full 180-action space — no legality mask is stored in the npz, so the
softmax runs over all actions; trained nets concentrate their mass on legal
moves, and the random-init reference line shows what chance looks like under
the same convention).

Rows are additionally split by within-game move quartile (from the
``game_records`` spans in the stats sidecar) — early quartiles are where
strategy lives, late quartiles are forced/tactical, so a net that is strong
late and weak early is exactly the "great calculator, poor strategist" shape.

Output: a table on stdout and a JSON report (``--out``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

__all__ = ["main", "score_net"]


def _policy_rows(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(states, target action ids, keep mask over ALL rows) for policy rows."""
    with np.load(npz_path) as z:
        states = z["states"]
        policies = z["policies"]
        mask = z["policy_mask"] if "policy_mask" in z else np.ones(len(states))
        size = int(z["size"][0]) if "size" in z else len(states)
    states, policies, mask = states[:size], policies[:size], mask[:size]
    keep = mask > 0.5
    return states[keep].astype(np.float32), policies[keep].argmax(axis=1), keep


def _quartiles(stats_path: Path | None, keep: np.ndarray) -> np.ndarray:
    """Within-game quartile (0-3) for each *kept* row, zeros when unknown.

    ``game_records[].rows`` spans count every stored row (including opponent
    value-only rows), so the quartile is computed over the full row sequence and
    then filtered down by the same ``keep`` mask as the policy rows.
    """
    n_all = len(keep)
    q = np.zeros(n_all, dtype=np.int64)
    if stats_path is None or not stats_path.exists():
        return q[keep]
    records = json.loads(stats_path.read_text()).get("game_records") or []
    at = 0
    for rec in records:
        rows = int(rec["rows"])
        if rows <= 0:
            continue
        if at + rows > n_all:  # spans out of sync with the npz -> don't pretend
            return np.zeros(n_all, dtype=np.int64)[keep]
        idx = np.arange(rows)
        q[at : at + rows] = np.minimum(3, idx * 4 // rows)
        at += rows
    if at != n_all:
        return np.zeros(n_all, dtype=np.int64)[keep]
    return q[keep]


def score_net(
    ckpt: str, states: np.ndarray, targets: np.ndarray, batch: int = 4096
) -> dict:
    import torch

    from ludometer.train.net import load_net

    net, _ = load_net(ckpt)
    top1 = np.zeros(len(states), dtype=bool)
    top3 = np.zeros(len(states), dtype=bool)
    nll = np.zeros(len(states), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(states), batch):
            x = torch.from_numpy(states[i : i + batch])
            logits = net(x)[0]
            logp = torch.log_softmax(logits, dim=1).numpy()
            t = targets[i : i + batch]
            order = np.argsort(-logp, axis=1)
            top1[i : i + batch] = order[:, 0] == t
            top3[i : i + batch] = (order[:, :3] == t[:, None]).any(axis=1)
            nll[i : i + batch] = -logp[np.arange(len(t)), t]
    return {"top1": top1, "top3": top3, "nll": nll}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ludometer.human.agreement")
    parser.add_argument("nets", nargs="+", help="label=checkpoint.pt")
    parser.add_argument("--npz", type=Path, default=Path("data/human/replay.npz"))
    parser.add_argument(
        "--stats", type=Path, default=None, help="replay.stats.json for game spans"
    )
    parser.add_argument("--out", type=Path, default=None, help="JSON report path")
    args = parser.parse_args(argv)

    stats = args.stats
    if stats is None:
        guess = args.npz.with_suffix("").with_suffix("")  # replay.npz -> replay
        stats = guess.parent / (args.npz.name.replace(".npz", ".stats.json"))
        if not stats.exists():
            stats = None

    states, targets, keep = _policy_rows(args.npz)
    quart = _quartiles(stats, keep)
    print(f"{args.npz}: {len(states):,} policy rows " f"(quartile spans: {'yes' if quart.any() else 'no'})")

    report: dict = {"npz": str(args.npz), "rows": int(len(states)), "nets": {}}
    header = f"{'net':>10} {'top1':>7} {'top3':>7} {'nll':>7} " + " ".join(
        f"q{i}-top1" for i in range(4)
    )
    print(header)
    for spec in args.nets:
        label, _, ckpt = spec.rpartition("=")
        label = label or Path(ckpt).stem
        s = score_net(ckpt, states, targets)
        by_q = [float(s["top1"][quart == i].mean()) if (quart == i).any() else float("nan") for i in range(4)]
        row = {
            "checkpoint": ckpt,
            "top1": float(s["top1"].mean()),
            "top3": float(s["top3"].mean()),
            "nll": float(s["nll"].mean()),
            "top1_by_quartile": by_q,
        }
        report["nets"][label] = row
        print(
            f"{label:>10} {row['top1']:7.3f} {row['top3']:7.3f} {row['nll']:7.3f} "
            + " ".join(f"{v:7.3f}" for v in by_q)
        )
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
