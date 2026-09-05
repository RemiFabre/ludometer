"""Measure what actually limits self-play: CPU inference and search throughput.

    uv run python -m ludometer.train.benchmark --config configs/run3.json
    uv run python -m ludometer.train.benchmark --arch structured --embed 64 --games 2

Two numbers come out of it.

``ms/position`` — one :class:`~ludometer.train.net.NetEvaluator` call on **one**
CPU thread, exactly as a self-play worker does it (encode + forward + softmax
over the legal actions). This is the hard constraint on a run's sim count: a
512-sim move costs ``512 x ms/position`` of pure inference.

Because the machine is usually busy with a training run, a single timing is
meaningless: this tool interleaves the candidate with a **reference net**
(run1's 3x512 MLP, whose in-situ cost is known) and reports the *minimum* over
many short rounds. The minimum is the least-contended sample, i.e. the closest
thing to an idle-machine number, and the candidate/reference ratio is stable
even when the absolute numbers drift by 2x.

``sims/s`` — full self-play games, with and without MCTS tree reuse, which is
what the reuse flag is worth in practice (fewer evaluator calls per move for the
same number of root visits).

``--batched`` — the run5 question instead of the run1-run4 one. Batched self-play
(:mod:`ludometer.train.selfplay_batched`) never evaluates one position at a time,
so ``ms/position`` on a CPU thread stops being the constraint; what binds is the
**round-trip latency of one batch** on the self-play device, because the driver
cannot descend again until the values are home. This mode reports that, per batch
size, and it is the number to look at before changing run5's net size.

Run it niced (``--nice``, on by default) so it never steals time from a live run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from ludometer.azul.engine import AzulState
from ludometer.train.mcts import MCTS, MCTSConfig
from ludometer.train.net import NetConfig, NetEvaluator, PolicyValueNet, make_net
from ludometer.train.selfplay import SelfPlayConfig, play_selfplay_game

__all__ = ["bench_batched", "bench_inference", "bench_selfplay", "main"]

# run1's net: 1.0M-param 3x512 MLP, the architecture that produced +2014 Elo at
# 160 sims/move with 8 workers. Anything at or below its cost is affordable.
REFERENCE = NetConfig(hidden=512, blocks=3, value_hidden=64)


def _time_evaluator(evaluator: NetEvaluator, state: AzulState, n: int) -> float:
    legal = state.legal_actions()
    start = time.perf_counter()
    for _ in range(n):
        evaluator(state, legal)
    return (time.perf_counter() - start) / n


def bench_inference(
    net: Any, rounds: int = 12, per_round: int = 200, reference: bool = True
) -> dict[str, float]:
    """Minimum ms/position over ``rounds``, interleaved with the reference net."""
    torch.set_num_threads(1)
    state = AzulState.new_game(seed=1)
    candidate = NetEvaluator(net.eval(), device="cpu")
    ref = (
        NetEvaluator(PolicyValueNet(REFERENCE).eval(), device="cpu")
        if reference
        else None
    )
    for _ in range(100):  # warm up both
        candidate(state, state.legal_actions())
        if ref is not None:
            ref(state, state.legal_actions())
    best = 1e9
    best_ref = 1e9
    for _ in range(rounds):
        best = min(best, _time_evaluator(candidate, state, per_round))
        if ref is not None:
            best_ref = min(best_ref, _time_evaluator(ref, state, per_round))
    out = {"ms": best * 1e3, "params": float(net.num_params)}
    if ref is not None:
        out["ref_ms"] = best_ref * 1e3
        out["ratio"] = best / best_ref
    return out


def bench_selfplay(
    net: Any,
    sims: int,
    games: int = 2,
    reuse: bool = True,
    seed: int = 1,
    engine: str = "python",
) -> dict[str, float]:
    """Play ``games`` self-play games and report evals/s, sims/s and s/game.

    ``engine="rust"`` runs the same loop on the Rust tree
    (:mod:`ludometer.train.mcts_rs`): the net is still evaluated one leaf at a
    time in Python, so this isolates what the tree walk itself costs.
    """
    torch.set_num_threads(1)
    evaluator = NetEvaluator(net.eval(), device="cpu")
    config = SelfPlayConfig(
        mcts=MCTSConfig(sims=sims, tree_reuse=reuse), temp_moves=6, max_moves=200
    )
    mcts_cls: Any = MCTS
    if engine == "rust":
        from ludometer.train.mcts_rs import MCTS as RustMCTS

        mcts_cls = RustMCTS
    elif engine != "python":
        raise ValueError(f"engine must be python or rust, got {engine!r}")
    start = time.perf_counter()
    evals = moves = 0
    for i in range(games):
        record = play_selfplay_game(evaluator, seed + i, config, mcts_cls=mcts_cls)
        evals += record.evals
        moves += record.moves
    elapsed = time.perf_counter() - start
    # Every searched move ends up with `sims` root visits either way (reuse tops
    # the inherited subtree up to the same budget), so "effective sims/s" is the
    # search actually delivered per second and evals/move is what reuse saves.
    return {
        "games": float(games),
        "seconds": elapsed,
        "s_per_game": elapsed / max(1, games),
        "moves": float(moves),
        "evals": float(evals),
        "evals_per_move": evals / max(1, moves),
        "evals_per_s": evals / elapsed,
        "sims_per_s": moves * sims / elapsed,
    }


def bench_batched(
    net: Any, batches: Sequence[int], device: str = "auto", rounds: int = 7
) -> dict[str, Any]:
    """Round-trip latency of one batched evaluation, per batch size.

    Deliberately *not* pipelined: the batched self-play loop submits a batch and
    then waits for it, so back-to-back dispatches would flatter the GPU by an
    order of magnitude. The minimum over ``rounds`` is the least-contended sample.
    """
    from ludometer.train.selfplay_batched import BatchEvaluator, resolve_selfplay_device

    resolved = resolve_selfplay_device(device)
    evaluator = BatchEvaluator(net.eval(), device=resolved)
    state = AzulState.new_game(seed=1)
    legal = state.legal_actions()
    out: dict[str, Any] = {"device": resolved, "params": net.num_params, "batches": {}}
    for size in batches:
        states = [state] * size
        legals = [legal] * size
        for _ in range(3):  # warm the kernels for this shape
            evaluator.evaluate(states, legals)
        best = 1e9
        for _ in range(rounds):
            start = time.perf_counter()
            evaluator.evaluate(states, legals)
            best = min(best, time.perf_counter() - start)
        out["batches"][str(size)] = {"ms": best * 1e3, "positions_per_s": size / best}
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ludometer.train.benchmark",
        description="CPU inference + self-play throughput for a net config",
    )
    parser.add_argument("--config", type=Path, help="configs/*.json to benchmark")
    parser.add_argument("--checkpoint", type=Path, help="benchmark a saved net")
    parser.add_argument("--arch", default=None, help="mlp | structured")
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--per-round", type=int, default=200)
    parser.add_argument("--sims", type=int, default=0, help="0 -> take it from config")
    parser.add_argument("--games", type=int, default=0, help="self-play games (0=skip)")
    parser.add_argument(
        "--engine",
        default="python",
        choices=("python", "rust"),
        help="search implementation for --games (rust = ludometer_rs tree)",
    )
    parser.add_argument(
        "--batched",
        action="store_true",
        help="measure batched round-trip latency (the run5 constraint)",
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,32,64,128,256",
        help="comma-separated batch sizes for --batched",
    )
    parser.add_argument(
        "--selfplay-device", default="", help="device for --batched (default: config)"
    )
    parser.add_argument("--nice", type=int, default=19, help="niceness (0 = leave)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.nice:
        with __import__("contextlib").suppress(OSError):
            os.nice(args.nice)
    data: dict[str, Any] = {}
    if args.config:
        data = json.loads(args.config.read_text(encoding="utf-8"))
    if args.arch:
        data["arch"] = args.arch
    if args.checkpoint:
        from ludometer.train.net import load_net

        net, _payload = load_net(args.checkpoint)
    else:
        net = make_net(data)
    sims = args.sims or int(data.get("sims", 512))

    out: dict[str, Any] = {
        "arch": getattr(net.config, "arch", "mlp"),
        "params": net.num_params,
        "sims": sims,
        "inference": bench_inference(net, args.rounds, args.per_round),
    }
    if args.games:
        out["engine"] = args.engine
        out["selfplay_reuse"] = bench_selfplay(
            net, sims, args.games, reuse=True, engine=args.engine
        )
        out["selfplay_plain"] = bench_selfplay(
            net, sims, args.games, reuse=False, engine=args.engine
        )
    if args.batched:
        sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
        out["batched"] = bench_batched(
            net,
            sizes,
            device=args.selfplay_device or str(data.get("selfplay_device", "auto")),
        )
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    inf = out["inference"]
    print(f"arch={out['arch']} params={out['params']:,} sims={sims}")
    print(
        f"inference: {inf['ms']:.3f} ms/position  "
        f"(reference 3x512 MLP {inf['ref_ms']:.3f} ms, {inf['ratio']:.2f}x)"
    )
    print(f"           -> {sims * inf['ms']:.0f} ms of inference per {sims}-sim move")
    for key in ("selfplay_reuse", "selfplay_plain"):
        if key in out:
            s = out[key]
            print(
                f"{key:<16} {s['s_per_game']:6.1f} s/game  "
                f"{s['evals_per_move']:6.1f} evals/move  "
                f"{s['sims_per_s']:7.1f} effective sims/s"
            )
    if "selfplay_plain" in out:
        gain = out["selfplay_plain"]["s_per_game"] / out["selfplay_reuse"]["s_per_game"]
        print(f"tree reuse speedup: {gain:.2f}x")
    if "batched" in out:
        batched = out["batched"]
        print(f"batched round trip on {batched['device']} (not pipelined):")
        for size, row in batched["batches"].items():
            print(
                f"  batch {size:>4}: {row['ms']:7.2f} ms  "
                f"{row['positions_per_s']:9,.0f} positions/s"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
