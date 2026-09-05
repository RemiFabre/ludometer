"""Pull a run's shards to disk and fold them into one replay file.

    uv run python -m ludometer.cloud.corpus pull --run rlx_teacher
    uv run python -m ludometer.cloud.corpus build --run rlx_teacher \
        --out data/cloud/rlx_teacher.npz
    uv run python -m ludometer.cloud.corpus stats --run rlx_teacher

``pull`` is incremental: shards already under ``data/cloud/<run>/shards/``
are not fetched again, so it can run on a timer while the fleet plays.
``build`` writes a ``ReplayBuffer``-format ``.npz`` (newest last) that the
trainer's ``--pretrain`` path and ``finetune``/``distill`` consume unchanged,
search values included. ``stats`` reads only the shard metas.

Unlike the trainer-side hub engine this never *consumes* anything: the
teacher corpus is a dataset, not a stream, and every shard stays on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ludometer.cloud.hub import hub_from_spec
from ludometer.cloud.shards import peek_meta, read_shard
from ludometer.train.replay import ReplayBuffer

__all__ = ["build", "main", "pull", "stats"]

ROOT = Path("data/cloud")


def _dir(run: str) -> Path:
    return ROOT / run / "shards"


def pull(run: str, shards_spec: str, token: str | None = None) -> int:
    hub = hub_from_spec(shards_spec, token)
    target = _dir(run)
    target.mkdir(parents=True, exist_ok=True)
    have = {p.name for p in target.glob("*.npz")}
    names = [n for n in hub.list(f"{run}/") if n.endswith(".npz")]
    fresh = [n for n in names if Path(n).name not in have]
    for i, name in enumerate(fresh):
        hub.get(name, target / Path(name).name)
        if (i + 1) % 20 == 0:
            print(f"[corpus] {i + 1}/{len(fresh)} pulled", flush=True)
    print(f"[corpus] {run}: {len(have)} had, {len(fresh)} new, {len(names)} on the hub")
    return len(fresh)


def stats(run: str) -> dict:
    games = positions = evals = 0
    seconds = 0.0
    tags: dict[str, int] = {}
    versions: dict[int, int] = {}
    files = sorted(_dir(run).glob("*.npz"))
    for p in files:
        m = peek_meta(p)
        games += int(m.get("games", 0))
        positions += int(m.get("positions", 0))
        evals += int(m.get("evals", 0))
        seconds += float(m.get("seconds", 0.0))
        tags[m.get("tag", "?")] = tags.get(m.get("tag", "?"), 0) + int(
            m.get("games", 0)
        )
        v = int(m.get("weights_version", 0))
        versions[v] = versions.get(v, 0) + int(m.get("games", 0))
    out = {
        "shards": len(files),
        "games": games,
        "positions": positions,
        "evals": evals,
        "jobs": len(tags),
        "games_by_version": dict(sorted(versions.items())),
        "evals_per_game": round(evals / games) if games else 0,
        # block seconds are per job, so this is the per-job rate, not the fleet's
        "games_per_job_hour": round(3600 * games / seconds) if seconds else 0,
    }
    print(json.dumps(out, indent=1))
    return out


def build(runs: list[str] | str, out: Path, cap: int = 0) -> int:
    """Fold the shards of ``runs`` (in that order; the last run ends up newest,
    which is what survives a ``cap``) into one replay file."""
    runs = [runs] if isinstance(runs, str) else list(runs)
    files: list[Path] = []
    for run in runs:
        part = sorted(_dir(run).glob("*.npz"), key=lambda p: p.stat().st_mtime)
        if not part:
            raise SystemExit(f"no shards under {_dir(run)}")
        files.extend(part)
    total = 0
    for p in files:
        total += int(peek_meta(p).get("positions", 0))
    capacity = min(total, cap) if cap else total
    buf = ReplayBuffer(capacity=max(1, capacity), seed=0)
    n_games = 0
    for p in files:
        records, _meta = read_shard(p)
        for r in records:
            buf.add_game(r)
            n_games += 1
    buf.save(out)
    s = buf.stats()
    print(
        f"[corpus] wrote {out}: {len(buf):,} positions from {n_games:,} games "
        f"({s['search_targets']:,} with a search value, {s['policy_targets']:,} policy targets)"
    )
    return len(buf)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ludometer.cloud.corpus")
    sub = p.add_subparsers(dest="cmd", required=True)
    parsers = [sub.add_parser(name) for name in ("pull", "stats", "build")]
    for sp in parsers:
        sp.add_argument(
            "--run", required=True, nargs="+", help="run(s); build folds them in order"
        )
        sp.add_argument("--shards", default="RemiFabre/rl-experiment-shards")
    b = parsers[2]
    b.add_argument("--out", type=Path, required=True)
    b.add_argument(
        "--cap", type=int, default=0, help="keep only the newest N positions"
    )
    args = p.parse_args(argv)
    if args.cmd == "pull":
        for run in args.run:
            pull(run, args.shards, os.environ.get("HF_TOKEN"))
    elif args.cmd == "stats":
        for run in args.run:
            stats(run)
    else:
        build(args.run, args.out, cap=args.cap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
