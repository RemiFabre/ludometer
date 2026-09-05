"""The job side: fetch weights, play games, upload shards, poll, repeat.

    python -m ludometer.cloud.generator --run mid3 \
        --shards RemiFabre/rl-experiment-shards \
        --weights model:RemiFabre/rl-experiment-weights \
        --workers 8 --block 64 --tag j1a2b3

The run's search settings come from ``<run>/config.json`` in the weights repo
(published by :class:`~ludometer.cloud.hub_selfplay.HubSelfPlay.start`, or by
hand for a fixed-teacher corpus); ``--sims`` and friends override them. The
weights come from ``<run>/current.json``; the pointer is re-read between
blocks and a new version is broadcast to the driver processes before the
next block starts.

Seeds are unique per job and per block: ``(crc32(tag) % 100_000) * 20_000 +
block * block_size + i``, well under 2**31 for any job that plays under 20k
games. The record carries its seed, so a game is reproducible from the shard's
meta (weights version) plus that number.

The process runs until it is killed (the job's timeout). Uploads retry; if the
hub stays unreachable for ``--give-up-min`` minutes the process exits non-zero
so the job stops billing.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any

from ludometer.cloud.hub import fetch_weights, hub_from_spec
from ludometer.cloud.shards import write_shard
from ludometer.train.net import net_config_from_dict
from ludometer.train.selfplay import make_selfplay
from ludometer.train.trainer import TrainConfig

__all__ = ["main", "seed_base"]


def seed_base(tag: str) -> int:
    return (zlib.crc32(tag.encode("utf-8")) % 100_000) * 20_000


class _BlockNote:
    """Once a minute, the positions/s of the block in flight (see the pool's ticks)."""

    def __init__(self, engine: Any, t_block: float) -> None:
        self.engine = engine
        self.t_block = t_block
        self.last = time.monotonic()
        self.base = sum(getattr(engine, "worker_positions", {}).values())

    def __call__(self, done: int, total: int) -> None:
        now = time.monotonic()
        if now - self.last < 60.0:
            return
        self.last = now
        pos = sum(getattr(self.engine, "worker_positions", {}).values()) - self.base
        rate = pos / max(1e-9, now - self.t_block)
        _log(
            f"  ... {done}/{total} games done, {pos:,} positions this block, "
            f"{rate:,.0f} positions/s"
        )


def _log(msg: str) -> None:
    print(f"[rl-experiment {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ludometer.cloud.generator")
    p.add_argument("--run", required=True)
    p.add_argument("--shards", required=True, help="hub spec of the shards store")
    p.add_argument("--weights", required=True, help="hub spec of the weights store")
    p.add_argument(
        "--tag", default="", help="unique job tag (seeds); default hostname+pid"
    )
    p.add_argument(
        "--workers", type=int, default=0, help="driver processes (0 = cpu count)"
    )
    p.add_argument(
        "--games", type=int, default=0, help="concurrent games per driver (0 = config)"
    )
    p.add_argument("--block", type=int, default=64, help="games per shard")
    p.add_argument("--sims", type=int, default=0, help="override the run's sims")
    p.add_argument(
        "--max-blocks", type=int, default=0, help="stop after N blocks (0 = forever)"
    )
    p.add_argument(
        "--poll-s",
        type=float,
        default=30.0,
        help="wait between pointer checks when no weights yet",
    )
    p.add_argument("--give-up-min", type=float, default=30.0)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--half", action="store_true", help="fp16 net on the device (MPS/CUDA)"
    )
    p.add_argument(
        "--engine",
        default="batched",
        choices=("batched", "rust"),
        help="self-play engine: the Python batched driver or the Rust arena",
    )
    return p


def _load_run_config(
    weights_hub: Any, run: str, overrides: argparse.Namespace
) -> TrainConfig:
    # The trainer publishes the config in start(); a job launched a little
    # earlier waits for it the way it waits for the first weights.
    t_wait = time.monotonic()
    while True:
        raw = weights_hub.get_bytes(f"{run}/config.json")
        if raw is not None:
            break
        if time.monotonic() - t_wait > overrides.give_up_min * 60:
            raise SystemExit(f"no {run}/config.json in {weights_hub.describe()}")
        _log("waiting for config...")
        time.sleep(overrides.poll_s)
    data = json.loads(raw.decode("utf-8"))
    data.pop("started", None)
    cfg = TrainConfig.from_dict(data)
    if overrides.sims:
        cfg.sims = int(overrides.sims)
        if cfg.pcr:
            cfg.pcr = {}
    if overrides.games:
        cfg.selfplay_games = int(overrides.games)
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tag = args.tag or f"{socket.gethostname()}-{os.getpid()}"
    token = os.environ.get("HF_TOKEN")
    shards = hub_from_spec(args.shards, token)
    weights_hub = hub_from_spec(args.weights, token)
    cfg = _load_run_config(weights_hub, args.run, args)
    workers = args.workers or max(1, os.cpu_count() or 1)
    # A block is split evenly over the drivers, and a driver batches only what
    # it holds: a block under `workers x games` runs the trees under-batched
    # (the first smoke job played 16 lone games on 8 vCPUs at 1/10 the rate).
    full_block = workers * cfg.selfplay_games
    if args.block < full_block:
        _log(
            f"block {args.block} raised to {full_block} ({workers} x {cfg.selfplay_games})"
        )
        args.block = full_block
    _log(
        f"run {args.run} tag {tag}: {workers} {args.engine} drivers x {cfg.selfplay_games} games, "
        f"sims {cfg.sims}, shards -> {shards.describe()}, weights <- {weights_hub.describe()}"
    )

    # Wait for the first weights.
    version = 0
    net_config: dict[str, Any] | None = None
    weights = None
    t_wait = time.monotonic()
    with tempfile.TemporaryDirectory() as td:
        while True:
            got = fetch_weights(weights_hub, args.run, version, td)
            if got is not None:
                version, net_config, weights = got
                break
            if time.monotonic() - t_wait > args.give_up_min * 60:
                _log("no weights published; giving up")
                return 2
            _log("waiting for weights...")
            time.sleep(args.poll_s)
        assert net_config is not None and weights is not None
        nc = net_config_from_dict(net_config)
        engine = make_selfplay(
            nc,
            cfg.selfplay_config(),
            workers,
            kind=args.engine,
            games=cfg.selfplay_games,
            device=args.device,
            max_batch=cfg.selfplay_max_batch,
            half=args.half,
        )
        engine.start(weights)
        _log(f"weights v{version} loaded; playing")

        # A killed parent would leave its driver processes running (orphans
        # burning the GPU for nobody, seen on the Mac): close the pool first.
        import signal

        def _terminate(signum: int, _frame: Any) -> None:
            _log(f"signal {signum}: closing the drivers")
            engine.close()
            raise SystemExit(128 + signum)

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, _terminate)
        base = seed_base(tag)
        block = 0
        played = 0
        t0 = time.monotonic()
        last_ok = time.monotonic()
        try:
            while not args.max_blocks or block < args.max_blocks:
                seed_start = base + block * args.block
                t_block = time.monotonic()
                note = _BlockNote(engine, t_block)
                records = engine.play(args.block, seed_start, progress=note)
                dt = time.monotonic() - t_block
                if not records:
                    _log("engine returned no games; stopping")
                    return 3
                played += len(records)
                positions = sum(len(r) for r in records)
                evals = sum(r.evals for r in records)
                meta = {
                    "run": args.run,
                    "tag": tag,
                    "block": block,
                    "weights_version": version,
                    "sims": cfg.sims,
                    "games": len(records),
                    "positions": positions,
                    "evals": evals,
                    "seconds": round(dt, 1),
                    "created": time.time(),
                }
                name = f"{args.run}/v{version:05d}-{tag}-{block:05d}.npz"
                local = Path(td) / "shard.npz"
                write_shard(local, records, meta)
                try:
                    shards.put(local, name)
                    last_ok = time.monotonic()
                except Exception as exc:  # noqa: BLE001 - keep playing, maybe it comes back
                    _log(f"upload failed after retries: {exc}")
                    if time.monotonic() - last_ok > args.give_up_min * 60:
                        _log("hub unreachable for too long; exiting")
                        return 4
                _log(
                    f"block {block}: {len(records)} games, {positions} positions, "
                    f"{evals / dt:,.0f} evals/s, {60 * len(records) / dt:.1f} games/min, "
                    f"v{version}; total {played} games in {(time.monotonic() - t0) / 60:.1f} min"
                )
                block += 1
                # New weights?
                try:
                    got = fetch_weights(weights_hub, args.run, version, td)
                except Exception as exc:  # noqa: BLE001
                    _log(f"pointer check failed: {exc}")
                    got = None
                if got is not None:
                    version, _nc, weights = got
                    engine.set_weights(weights)
                    _log(f"weights v{version} broadcast")
        finally:
            engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
