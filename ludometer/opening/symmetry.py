"""E4: how invariant is the net under the 180° board rotation?

The wall is a Latin square with ``col = (color + row) % 5``; rotating it by 180°
maps cell ``(r, col)`` to ``(4 - r, 4 - col)`` and is the same as relabelling the
colours ``c -> (-c) % 5`` (0->0, 1->4, 2->3, 3->2, 4->1) together with reversing
the rows. Adjacency (rows, columns, colour sets) is preserved, so the *wall* is
exactly symmetric; the *pattern lines* are not, because row ``r`` holds ``r + 1``
tiles and row ``4 - r`` holds ``5 - r``. So the transform is only defined on
positions whose reversed pattern lines still fit, and it is only a true
symmetry of the game when the lines are empty (move 1 of the game). This module
measures how far the net is from invariant anyway:

    uv run python -m ludometer.opening.symmetry --net porcelain=<ckpt> --rounds 0-1

For every position (rounds ``--rounds`` of the expert games) that admits the
transform: the raw policy on the original, the raw policy on the transformed
position mapped back through the inverse action map, the KL between them, and
whether the top-1 move changed. Reported separately for empty boards (an exact
symmetry, where any difference is a generalisation gap) and for the rest (an
approximate symmetry). The *draft marginal* (which tiles to take, summed over
the destination row) is reported too: that is the part the task force would
augment, if anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ludometer.azul.engine import (
    ACTION_SPACE,
    CENTER,
    FLOOR,
    NUM_COLORS,
    AzulState,
    decode_action,
    encode_action,
)
from ludometer.cloud.label import load_positions
from ludometer.opening.atlas import POSITIONS, load_targets, opening_positions, parse_rounds

__all__ = ["rotate", "action_map", "main"]


def _c(color: int) -> int:
    return (-color) % NUM_COLORS


def rotate(state: AzulState) -> AzulState | None:
    """The 180°-rotated position, or ``None`` when a pattern line would not fit."""
    s = state.clone()
    for p in range(2):
        for r in range(5):
            n = state.pl_count[p][r]
            if not n:
                continue
            # row 4-r holds 5-r tiles: the line must fit, and a full line must
            # stay full (else the rotated position has moves the original lacks)
            if n > 5 - r or (n == r + 1) != (n == 5 - r):
                return None
    s.factories = [[f[_c(c)] for c in range(NUM_COLORS)] for f in state.factories]
    s.center = [state.center[_c(c)] for c in range(NUM_COLORS)]
    s.lid = [state.lid[_c(c)] for c in range(NUM_COLORS)]
    s.bag = [ _c(c) for c in state.bag ]
    for p in range(2):
        w = state.walls[p]
        s.walls[p] = [w[(4 - r) * 5 + (4 - col)] for r in range(5) for col in range(5)]
        s.pl_color[p] = [(-1 if state.pl_color[p][4 - r] < 0 else _c(state.pl_color[p][4 - r])) for r in range(5)]
        s.pl_count[p] = [state.pl_count[p][4 - r] for r in range(5)]
        s.floor[p] = [state.floor[p][_c(c)] for c in range(NUM_COLORS)]
    s.recount()
    return s


def action_map() -> np.ndarray:
    """``m[a]`` = the action in the rotated frame that corresponds to ``a``."""
    m = np.zeros(ACTION_SPACE, dtype=np.int64)
    for a in range(ACTION_SPACE):
        src, c, d = decode_action(a)
        m[a] = encode_action(src, _c(c), d if d == FLOOR else 4 - d)
    return m


def draft_marginal(policy: np.ndarray) -> np.ndarray:
    """Sum over the destination: a (6 sources x 5 colours) table, flattened."""
    return policy.reshape(6, 5, 6).sum(axis=2).reshape(-1)


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-9
    m = p > eps
    return float((p[m] * (np.log(p[m] + eps) - np.log(q[m] + eps))).sum())


def measure(net: Any, states: list[AzulState], device: str, batch: int = 1024) -> dict[str, np.ndarray]:
    import torch

    amap = action_map()
    inv = np.argsort(amap)
    pol_o = np.zeros((len(states), ACTION_SPACE), dtype=np.float64)
    pol_r = np.zeros((len(states), ACTION_SPACE), dtype=np.float64)
    val_o = np.zeros(len(states))
    val_r = np.zeros(len(states))
    rotated = [rotate(s) for s in states]
    ok = np.array([r is not None for r in rotated])
    with torch.inference_mode():
        for start in range(0, len(states), batch):
            chunk = list(range(start, min(len(states), start + batch)))
            xo = torch.from_numpy(np.stack([states[i].encode() for i in chunk])).to(device)
            xr = torch.from_numpy(np.stack([(rotated[i] or states[i]).encode() for i in chunk])).to(device)
            lo, vo = net(xo)[:2]
            lr, vr = net(xr)[:2]
            lo = lo.float().cpu().numpy(); lr = lr.float().cpu().numpy()
            val_o[chunk] = vo.float().cpu().numpy().reshape(-1)
            val_r[chunk] = vr.float().cpu().numpy().reshape(-1)
            for j, i in enumerate(chunk):
                legal = np.asarray(states[i].legal_actions(), dtype=np.int64)
                p = np.zeros(ACTION_SPACE)
                sel = np.exp(lo[j][legal] - lo[j][legal].max()); p[legal] = sel / sel.sum()
                pol_o[i] = p
                if rotated[i] is not None:
                    legal_r = np.asarray(rotated[i].legal_actions(), dtype=np.int64)
                    q = np.zeros(ACTION_SPACE)
                    sel = np.exp(lr[j][legal_r] - lr[j][legal_r].max()); q[legal_r] = sel / sel.sum()
                    pol_r[i] = q[amap]  # mapped back: q_back[a] = q[amap[a]]
    return {"ok": ok, "pol_o": pol_o, "pol_r": pol_r, "val_o": val_o, "val_r": val_r}


def summarize(m: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    sel = mask & m["ok"]
    if not sel.any():
        return {"n": 0}
    po, pr = m["pol_o"][sel], m["pol_r"][sel]
    kl = np.array([_kl(a, b) for a, b in zip(po, pr)])
    klsym = 0.5 * (kl + np.array([_kl(b, a) for a, b in zip(po, pr)]))
    top1 = po.argmax(1) != pr.argmax(1)
    do, dr = draft_marginal(po.T).T if False else np.stack([draft_marginal(p) for p in po]), np.stack([draft_marginal(p) for p in pr])
    kl_d = np.array([_kl(a, b) for a, b in zip(do, dr)])
    top1_d = do.argmax(1) != dr.argmax(1)
    tv = 0.5 * np.abs(po - pr).sum(1)
    return {
        "n": int(sel.sum()),
        "kl_mean": float(kl.mean()), "kl_median": float(np.median(kl)), "kl_sym_mean": float(klsym.mean()),
        "tv_mean": float(tv.mean()),
        "top1_change": float(top1.mean()),
        "draft_kl_mean": float(kl_d.mean()), "draft_top1_change": float(top1_d.mean()),
        "value_abs_diff_mean": float(np.abs(m["val_o"][sel] - m["val_r"][sel]).mean()),
        "prob_of_orig_top1_after": float(np.mean([pr[i][po[i].argmax()] for i in range(len(po))])),
        "prob_of_orig_top1_before": float(po.max(1).mean()),
    }


def search_reference(ckpt: str, states: list[AzulState], rotated_ok: np.ndarray, rows: dict[str, np.ndarray],
                     sims: int, per_slice: int, device: str, slots: int) -> dict[str, dict[str, float]]:
    """The same comparison with the search's visit distribution instead of the raw prior."""
    from ludometer.opening.atlas import RustLabeler, _load_net_evaluator, _mcts_config

    evaluator = _load_net_evaluator(ckpt, device, False)
    config = _mcts_config(Path("runs/porc_w-p0905-2038/config.json"))
    amap = action_map()
    out = {}
    rng = np.random.default_rng(0)
    for name, mask in rows.items():
        if name.startswith("all"):
            continue
        idx = np.flatnonzero(mask & rotated_ok)
        if len(idx) > per_slice:
            idx = np.sort(rng.choice(idx, per_slice, replace=False))
        if len(idx) == 0:
            continue
        orig = [states[i] for i in idx]
        rot = [rotate(states[i]) for i in idx]
        labeler = RustLabeler(evaluator, config, slots=slots, sims=sims, seed=5, progress=False)
        ro = labeler.label(orig)
        rr = labeler.label(rot)
        po = np.stack([r["policy"] for r in ro]).astype(np.float64)
        pr = np.stack([r["policy"][amap] for r in rr]).astype(np.float64)
        vo = np.array([r["value"] for r in ro]); vr = np.array([r["value"] for r in rr])
        m = {"ok": np.ones(len(idx), dtype=bool), "pol_o": po, "pol_r": pr, "val_o": vo, "val_r": vr}
        out[name] = summarize(m, np.ones(len(idx), dtype=bool))
        out[name]["sims"] = sims
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ludometer.opening.symmetry")
    p.add_argument("--net", action="append", required=True, help="name=ckpt (repeatable)")
    p.add_argument("--rounds", default="0-1")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="mps")
    p.add_argument("--out", type=Path, default=Path("data/cloud/opening/symmetry.json"))
    p.add_argument("--search", type=int, default=0, help="also compare the search's policies at this many sims")
    p.add_argument("--search-max", type=int, default=1500, help="positions per slice for the search reference")
    p.add_argument("--slots", type=int, default=256)
    args = p.parse_args(argv)
    from ludometer.train.net import load_net

    games = load_positions(POSITIONS)
    if args.limit:
        games = games[: args.limit]
    targets = load_targets(games)
    positions = opening_positions(games, targets, parse_rounds(args.rounds))
    states = [q.state for q in positions]
    empty = np.array([not any(s.walls[0]) and not any(s.walls[1]) and not any(s.pl_count[0]) and not any(s.pl_count[1]) for s in states])
    walls_empty = np.array([not any(s.walls[0]) and not any(s.walls[1]) for s in states])
    rnd = np.array([q.round_index for q in positions])
    # sanity: the rotation is an involution and preserves legality counts
    r = rotate(states[0]); assert r is not None
    rr = rotate(r); assert rr is not None and rr.encode().tolist() == states[0].encode().tolist()
    assert sorted(action_map()[states[0].legal_actions()].tolist()) == sorted(r.legal_actions())
    report: dict[str, Any] = {"rounds": args.rounds, "positions": len(states), "nets": {}}
    print(f"[symmetry] {len(states):,} positions; empty boards {int(empty.sum()):,}, empty walls {int(walls_empty.sum()):,}")
    for spec in args.net:
        name, _, ckpt = spec.partition("=")
        net, _ = load_net(ckpt, device=args.device); net.eval()
        m = measure(net, states, args.device)
        rows = {
            "empty boards (the wall part is exact, the lines are not)": empty,
            "empty walls, some pattern lines (round 1)": walls_empty & ~empty & (rnd == 0),
            "round 2 (walls in play, transform admissible)": rnd == 1,
            "all admissible": np.ones(len(states), dtype=bool),
        }
        report["nets"][name] = {k: summarize(m, v) for k, v in rows.items()}
        if args.search:
            report["nets"][name]["search"] = search_reference(ckpt, states, rotated_ok=m["ok"], rows=rows, sims=args.search,
                                                              per_slice=args.search_max, device=args.device, slots=args.slots)
        print(f"\n{name}: {int(m['ok'].sum()):,} of {len(states):,} positions admit the transform")
        print(f"{'slice':>48} {'n':>6} {'KL':>7} {'medKL':>7} {'TV':>6} {'top1 chg':>9} {'draft KL':>9} {'draft top1 chg':>15} {'|dV|':>6} {'p(top1) before/after':>22}")
        for k, v in report["nets"][name].items():
            if k == "search":
                for kk, vv in v.items():
                    print(f"{'SEARCH ' + kk:>48} {vv['n']:6,} {vv['kl_mean']:7.3f} {vv['kl_median']:7.3f} {vv['tv_mean']:6.3f} {vv['top1_change']:9.1%} {vv['draft_kl_mean']:9.3f} {vv['draft_top1_change']:15.1%} {vv['value_abs_diff_mean']:6.3f} {vv['prob_of_orig_top1_before']:.2f} / {vv['prob_of_orig_top1_after']:.2f}")
                continue
            if v["n"]:
                print(f"{k:>48} {v['n']:6,} {v['kl_mean']:7.3f} {v['kl_median']:7.3f} {v['tv_mean']:6.3f} {v['top1_change']:9.1%} {v['draft_kl_mean']:9.3f} {v['draft_top1_change']:15.1%} {v['value_abs_diff_mean']:6.3f} {v['prob_of_orig_top1_before']:.2f} / {v['prob_of_orig_top1_after']:.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
