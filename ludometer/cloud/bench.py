"""What a Job flavor is actually worth: net latency, Python search rate, both together.

    python -m ludometer.cloud.bench --run rlx_teacher --weights model:... [--seconds 60]

Prints, for the published weights of ``--run``:

* the container's CPU quota (``cpu.max``) against what ``nproc`` claims;
* forward-pass ms/position at batch 1 / 16 / 64 on one CPU thread, and on
  CUDA when there is one;
* the pure search rate: batched self-play of a *tiny* net (search-bound) for
  ``--seconds`` — how fast this core walks a tree;
* the real thing: one driver of batched self-play with the published net at
  the run's sims for ``--seconds`` — positions/s, and the eval/search split.

The first smoke jobs on ``cpu-upgrade`` did 150 positions/s per driver with
the 7M teacher where one Mac core does 2,666; this says which half is slow.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

__all__ = ["main"]


def _until(seconds: float):
    end = time.perf_counter() + seconds
    return lambda: time.perf_counter() > end


def _read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return "n/a"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ludometer.cloud.bench")
    p.add_argument("--run", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--games", type=int, default=16)
    p.add_argument("--sims", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument(
        "--engine",
        default="batched",
        choices=("batched", "rust", "both"),
        help="self-play driver to measure (rust = ludometer_rs arena)",
    )
    p.add_argument(
        "--half", action="store_true", help="fp16 net on cuda for the real driver"
    )
    args = p.parse_args(argv)

    import torch

    torch.set_num_threads(args.threads)
    from ludometer.cloud.hub import fetch_weights, hub_from_spec
    from ludometer.train.net import make_net, net_config_from_dict
    from ludometer.train.net2 import StructuredConfig
    from ludometer.train.selfplay import make_selfplay
    from ludometer.train.trainer import TrainConfig

    engines = ["batched", "rust"] if args.engine == "both" else [args.engine]

    print(
        f"[bench] nproc={os.cpu_count()} cpu.max={_read('/sys/fs/cgroup/cpu.max')} "
        f"cpu.cfs_quota={_read('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')}/{_read('/sys/fs/cgroup/cpu/cpu.cfs_period_us')} "
        f"torch={torch.__version__} threads={torch.get_num_threads()} cuda={torch.cuda.is_available()}",
        flush=True,
    )
    print(
        f"[bench] {_read('/proc/cpuinfo').split(chr(10))[4] if 'model name' in _read('/proc/cpuinfo') else ''}",
        flush=True,
    )

    hub = hub_from_spec(args.weights, os.environ.get("HF_TOKEN"))
    raw = hub.get_bytes(f"{args.run}/config.json")
    cfg = TrainConfig.from_dict(
        {k: v for k, v in __import__("json").loads(raw).items() if k != "started"}
    )
    if args.sims:
        cfg.sims = args.sims
    with tempfile.TemporaryDirectory() as td:
        version, net_config, weights = fetch_weights(hub, args.run, 0, td)
    net = make_net(net_config_from_dict(net_config))
    net.load_numpy_state_dict(weights)
    net.eval()
    print(
        f"[bench] {args.run} v{version}: {net.num_params:,} params, sims {cfg.sims}",
        flush=True,
    )

    # 1. forward-pass latency per batch size
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        n = net.to(dev)
        for b in (1, 16, 64, 256):
            x = torch.rand(b, 182, device=dev)
            with torch.inference_mode():
                for _ in range(3):
                    n.forward_heads(x)
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                reps = 20 if b <= 16 else 8
                for _ in range(reps):
                    out = n.forward_heads(x)
                    _ = out[0].float().sum().item()  # force the readback
                dt = (time.perf_counter() - t0) / reps
            print(
                f"[bench] forward {dev} batch {b:3d}: {1e3 * dt:7.2f} ms  ({b / dt:9,.0f} positions/s)",
                flush=True,
            )
    net.to("cpu")

    # 1b. int8 dynamic quantization of the Linear layers (x86 VNNI is the hope)
    try:
        from torch.ao.quantization import quantize_dynamic

        q = quantize_dynamic(net, {torch.nn.Linear}, dtype=torch.qint8)
        for b in (16, 64):
            x = torch.rand(b, 182)
            with torch.inference_mode():
                for _ in range(3):
                    q.forward_heads(x)
                t0 = time.perf_counter()
                for _ in range(10):
                    _ = q.forward_heads(x)[0].sum().item()
                dt = (time.perf_counter() - t0) / 10
            print(
                f"[bench] forward int8 batch {b:3d}: {1e3 * dt:7.2f} ms  ({b / dt:9,.0f} positions/s)",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001 - informative only
        print(f"[bench] int8 unavailable: {exc}", flush=True)

    # 2. pure search rate (tiny net, search-bound)
    tiny = StructuredConfig(
        embed=32,
        layers=1,
        heads=4,
        ffn_mult=2,
        body=48,
        value_hidden=16,
        policy_rank=8,
        margin_head=True,
    )
    for kind in engines:
        eng = make_selfplay(
            tiny, cfg.selfplay_config(), 1, kind=kind, games=args.games, device="cpu"
        )
        eng.start(None)
        t0 = time.perf_counter()
        eng.play(10_000, 1, should_stop=_until(min(30.0, args.seconds / 2)))
        dt = time.perf_counter() - t0
        print(
            f"[bench] search-bound {kind} (tiny net, {args.games} games): {eng.positions / dt:,.0f} positions/s  "
            f"eval {eng.eval_seconds:.1f}s search {eng.search_seconds:.1f}s",
            flush=True,
        )

    # 3. the real driver
    for dev in devices:
        for kind in engines:
            half = bool(args.half and dev == "cuda")
            eng = make_selfplay(
                net_config_from_dict(net_config),
                cfg.selfplay_config(),
                1,
                kind=kind,
                games=args.games,
                device=dev,
                half=half,
            )
            eng.set_weights(weights)
            t0 = time.perf_counter()
            eng.play(10_000, 7, should_stop=_until(args.seconds))
            dt = time.perf_counter() - t0
            print(
                f"[bench] one {kind} driver on {dev}{' fp16' if half else ''} ({args.games} games, {cfg.sims} sims): "
                f"{eng.positions / dt:,.0f} positions/s  "
                f"eval {eng.eval_seconds:.1f}s search {eng.search_seconds:.1f}s batches {eng.batches}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
