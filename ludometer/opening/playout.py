"""Play it out: from a disagreement, the expert's move vs the net's, then Porcelain vs Porcelain.

    uv run python -m ludometer.opening.playout --report docs/opening/porcelain.json \
        --games 20 --think 1.0 --workers 8 --out data/cloud/opening/playout_porcelain.json

For each listed disagreement the position is replayed, the two candidate moves
are applied (a refill move is determinized per game seed, the same seed for
both branches), and the same searching agent plays both seats to the end.
Reported per position: the mover's score share and mean point margin after the
expert's move and after the net's, over ``--games`` games each (the same seeds
on both sides, so the difference is paired). This measures the net's *own*
valuation of the two continuations with a deeper, played-out search, not the
truth; a branch the net's Q called a blunder that then scores the same as the
net's own move is a Q that was wrong.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ludometer.opening.atlas import POSITIONS

__all__ = ["main", "play_from"]

_GAMES: dict[int, Any] = {}


def _game(table: int, positions: Path):
    if not _GAMES:
        from ludometer.cloud.label import load_positions

        for g in load_positions(positions):
            _GAMES[g.table_id] = g
    return _GAMES[table]


def play_from(task: tuple[Any, ...]) -> dict[str, Any]:
    (table, index, action, seed, agent_spec, positions, max_moves) = task
    from ludometer.agents.registry import load_agent
    from ludometer.cloud.label import replay_positions
    from ludometer.eval.arena import _agent_seed

    game = _game(table, positions)
    states, _movers, _final = replay_positions(game)
    state = states[index].clone()
    mover = state.current_player
    if state.is_stochastic(action):
        state = state.determinize(action, seed)
    else:
        state.apply(action)
    agents = [load_agent(agent_spec, seed=_agent_seed(seed, 0)), load_agent(agent_spec, seed=_agent_seed(seed, 1))]
    agents[0].seed(_agent_seed(seed, 0))
    agents[1].seed(_agent_seed(seed, 1))
    moves = 0
    while not state.is_terminal and moves < max_moves:
        a = agents[state.current_player].act(state)
        state.apply(a)
        moves += 1
    outcome = state.outcome() or 0.0
    sign = 1.0 if mover == 0 else -1.0
    res = 0.5 if outcome == 0.0 else (1.0 if outcome * sign > 0 else 0.0)
    return {
        "table": table, "index": index, "action": action, "seed": seed,
        "result": res, "margin": (state.scores[mover] - state.scores[1 - mover]),
        "moves": moves, "truncated": not state.is_terminal,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ludometer.opening.playout")
    p.add_argument("--report", type=Path, required=True, help="the report's JSON (disagreements_round1)")
    p.add_argument("--key", default="disagreements_round1")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--think", type=float, default=1.0)
    p.add_argument("--agent", default="mcts:runs/porc_w-p0905-2038/checkpoints/ckpt-000000.pt?think={think}&engine=rust")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--positions", type=Path, default=POSITIONS)
    p.add_argument("--max-moves", type=int, default=400)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    spec = args.agent.format(think=args.think)
    records = json.loads(args.report.read_text())[args.key][: args.top]
    tasks = []
    for rec in records:
        for g in range(args.games):
            seed = args.seed + 1000 * rec["rank"] + g
            for which in ("expert", "net"):
                tasks.append((rec["table"], rec["index"], rec[which], seed, spec, args.positions, args.max_moves))
    print(f"[playout] {len(records)} positions x {args.games} games x 2 branches = {len(tasks)} games, {spec}, {args.workers} workers", flush=True)
    t0 = time.monotonic()
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(play_from, tasks, chunksize=1), 1):
            results.append(r)
            if i % 40 == 0:
                print(f"[playout] {i}/{len(tasks)} games, {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    per: list[dict[str, Any]] = []
    for rec in records:
        rows = {"expert": [], "net": []}
        for r in results:
            if r["table"] == rec["table"] and r["index"] == rec["index"]:
                rows["expert" if r["action"] == rec["expert"] else "net"].append(r)
        if rec["expert"] == rec["net"]:
            continue
        e = rows["expert"]
        n = rows["net"]
        # paired by seed
        by_seed_e = {r["seed"]: r for r in e}
        by_seed_n = {r["seed"]: r for r in n}
        seeds = sorted(set(by_seed_e) & set(by_seed_n))
        d_res = np.array([by_seed_n[s]["result"] - by_seed_e[s]["result"] for s in seeds])
        d_mar = np.array([by_seed_n[s]["margin"] - by_seed_e[s]["margin"] for s in seeds])
        per.append({
            **{k: rec[k] for k in ("rank", "table", "index", "round", "move", "expert_text", "net_text", "loss", "points", "q_expert", "q_net")},
            "games": len(seeds),
            "expert_score": float(np.mean([r["result"] for r in e])),
            "net_score": float(np.mean([r["result"] for r in n])),
            "expert_margin": float(np.mean([r["margin"] for r in e])),
            "net_margin": float(np.mean([r["margin"] for r in n])),
            "delta_score": float(d_res.mean()),
            "delta_score_se": float(d_res.std(ddof=1) / np.sqrt(len(d_res))) if len(d_res) > 1 else None,
            "delta_margin": float(d_mar.mean()),
            "delta_margin_se": float(d_mar.std(ddof=1) / np.sqrt(len(d_mar))) if len(d_mar) > 1 else None,
            "truncated": sum(r["truncated"] for r in e + n),
        })
    summary = {
        "agent": spec, "games_per_branch": args.games, "positions": len(per),
        "mean_delta_score": float(np.mean([r["delta_score"] for r in per])) if per else None,
        "mean_delta_margin": float(np.mean([r["delta_margin"] for r in per])) if per else None,
        "mean_q_loss": float(np.mean([r["loss"] for r in per])) if per else None,
        "mean_points_loss": float(np.mean([r["points"] for r in per])) if per else None,
        "net_better": sum(1 for r in per if r["delta_score"] > 0),
        "expert_better": sum(1 for r in per if r["delta_score"] < 0),
        "seconds": round(time.monotonic() - t0, 1),
        "per_position": per,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    print(f"[playout] net's move scores {summary['mean_delta_score']:+.3f} (share) / {summary['mean_delta_margin']:+.2f} points "
          f"better than the expert's on average over {len(per)} positions; net better in {summary['net_better']}, "
          f"expert better in {summary['expert_better']}; the search's own Q gap averaged {summary['mean_q_loss']:.3f} "
          f"({summary['mean_points_loss']:+.2f} pts). {summary['seconds'] / 60:.1f} min -> {args.out}", flush=True)
    for r in per:
        print(f"  #{r['rank']:2d} r{r['round']} m{r['move']}: Q loss {r['loss']:.3f} | played out: expert {r['expert_score']:.2f} ({r['expert_margin']:+.1f}) "
              f"net {r['net_score']:.2f} ({r['net_margin']:+.1f}) delta {r['delta_score']:+.2f}±{(r['delta_score_se'] or 0):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
