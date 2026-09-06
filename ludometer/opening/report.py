"""Read the opening atlas: agreement tables, value losses, the disagreements.

    uv run python -m ludometer.opening.report atlas data/cloud/opening/porcelain_r01_s4096.npz \
        --out docs/opening/porcelain.md --top 20

Everything is computed in the mover's frame: ``q[a]`` is the search's win-Q of
the root edge ``a`` (in [-1, 1]), so the **value loss** of the expert's move is
``q[net_move] - q[expert]`` where ``net_move`` is what the net would actually
play (:func:`ludometer.train.mcts.decisive_action`: best win-Q among the
well-visited children, biggest margin among the near-ties). The margin-Q is
``tanh(points / 20)`` so ``20 * atanh`` turns it back into points.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ludometer.azul.engine import CENTER, FLOOR, decode_action
from ludometer.cloud.label import load_positions, replay_positions
from ludometer.opening.atlas import POSITIONS, describe_move

__all__ = ["Atlas", "load_atlas", "main"]


class Atlas:
    """One label file, with the derived columns."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        with np.load(self.path) as z:
            self.meta = json.loads(str(z["meta"]))
            self.d = {k: z[k] for k in z.files if k != "meta"}
        d = self.d
        self.n = len(d["expert"])
        self._mark_dual_targets()
        self.expert = d["expert"].astype(np.int64)
        self.mcts = self.meta.get("mcts") or {}
        self.net_move = self._decisive()
        self.visit_move = d["policy"].argmax(axis=1)
        self.prior_move = d["prior"].argmax(axis=1)
        rows = np.arange(self.n)
        self.q_expert = d["q"][rows, self.expert]
        self.q_net = d["q"][rows, self.net_move]
        self.loss = self.q_net - self.q_expert  # NaN when the expert move was unvisited
        m_e = np.clip(d["mq"][rows, self.expert], -0.999, 0.999)
        m_n = np.clip(d["mq"][rows, self.net_move], -0.999, 0.999)
        self.points_loss = 20.0 * (np.arctanh(m_n) - np.arctanh(m_e))
        self.agree = self.net_move == self.expert
        self.child_loss = np.full(self.n, np.nan, dtype=np.float32)
        self.visit_share_expert = d["policy"][rows, self.expert]
        order = np.argsort(-d["policy"], axis=1)[:, :3]
        self.top3 = (order == self.expert[:, None]).any(axis=1)

    def _mark_dual_targets(self, stats: Path = Path("data/human/replay.stats.json")) -> None:
        """In a dual-target game both seats are experts (the human pipeline's
        ``--dual-target-min-elo``): count the other seat's decisions as expert too."""
        if not stats.exists():
            return
        dual = {int(r["table_id"]) for r in json.loads(stats.read_text()).get("game_records") or [] if r.get("dual")}
        if dual:
            self.d["is_target"] = self.d["is_target"] | np.isin(self.d["table"], list(dual))
            self.dual_tables = len(dual)

    def _decisive(self) -> np.ndarray:
        eps = float(self.mcts.get("decisive_eps", 0.03))
        frac = float(self.mcts.get("decisive_min_visit_frac", 0.1))
        visits = self.d["visits"]
        q = self.d["q"]
        mq = self.d["mq"]
        out = np.zeros(self.n, dtype=np.int64)
        for i in range(self.n):
            v = visits[i]
            best = v.max()
            if best <= 0:
                out[i] = int(self.d["policy"][i].argmax())
                continue
            cand = (v >= frac * best) & (v > 0) & ~np.isnan(q[i])
            if not cand.any():
                out[i] = int(v.argmax())
                continue
            qi = np.where(cand, q[i], -np.inf)
            keep = cand & (qi >= qi.max() - eps)
            idx = np.flatnonzero(keep)
            key = [(mq[i][a], v[a], -a) for a in idx]
            out[i] = int(idx[int(np.argmax([k[0] * 1e9 + k[1] * 1e3 + k[2] * 1e-3 for k in key]))]) if len(idx) > 1 else int(idx[0])
        return out

    def attach_children(self, path: Path) -> int:
        """Value loss from full searches of the two child positions (``atlas children``).

        ``child_loss[i] = V(after net's move) - V(after expert's move)`` in the mover's
        frame; NaN where a child was not searched (a round-ending move, or a row the
        pass did not cover). Returns the number of rows covered."""
        self.child_value_expert = np.full(self.n, np.nan, dtype=np.float32)
        self.child_value_net = np.full(self.n, np.nan, dtype=np.float32)
        with np.load(path) as z:
            rows = z["row"]
            which = z["which"]
            value = z["value"]
        for r, w, v in zip(rows, which, value):
            (self.child_value_expert if w == "expert" else self.child_value_net)[int(r)] = float(v)
        self.child_loss = self.child_value_net - self.child_value_expert
        self.children_path = Path(path)
        return int((~np.isnan(self.child_loss)).sum())

    def keys(self) -> np.ndarray:
        return self.d["table"] * 1000 + self.d["index"]

    def mask(self, target: bool | None = True, rounds: tuple[int, ...] | None = None, moves: tuple[int, ...] | None = None) -> np.ndarray:
        m = np.ones(self.n, dtype=bool)
        if target is not None:
            m &= self.d["is_target"] == target
        if rounds is not None:
            m &= np.isin(self.d["round"], rounds)
        if moves is not None:
            m &= np.isin(self.d["mover_move"], moves)
        return m


def load_atlas(path: Path) -> Atlas:
    return Atlas(path)


# ------------------------------------------------------------------- tables
def _row(a: Atlas, m: np.ndarray) -> dict[str, float]:
    n = int(m.sum())
    if n == 0:
        return {"n": 0}
    loss = a.loss[m]
    known = ~np.isnan(loss)
    pts = a.points_loss[m][known]
    return {
        "n": n,
        "top1": float(a.agree[m].mean()),
        "top1_visits": float((a.visit_move[m] == a.expert[m]).mean()),
        "top3": float(a.top3[m].mean()),
        "prior_top1": float((a.prior_move[m] == a.expert[m]).mean()),
        "expert_visit_share": float(a.visit_share_expert[m].mean()),
        "unvisited": float((~known).mean()),
        "loss_mean": float(loss[known].mean()) if known.any() else float("nan"),
        "loss_median": float(np.median(loss[known])) if known.any() else float("nan"),
        "loss_p90": float(np.quantile(loss[known], 0.9)) if known.any() else float("nan"),
        "loss_gt_05": float((loss[known] > 0.05).mean()) if known.any() else float("nan"),
        "loss_gt_10": float((loss[known] > 0.10).mean()) if known.any() else float("nan"),
        "loss_gt_20": float((loss[known] > 0.20).mean()) if known.any() else float("nan"),
        "points_mean": float(pts.mean()) if len(pts) else float("nan"),
        "points_median": float(np.median(pts)) if len(pts) else float("nan"),
        **_child_stats(a, m),
    }


def _child_stats(a: Atlas, m: np.ndarray) -> dict[str, float]:
    cl = a.child_loss[m]
    dis = ~a.agree[m]
    known = ~np.isnan(cl)
    if not known.any():
        return {"child_n": 0}
    # over disagreements only (agreements have no child pass): mean loss and the
    # share the deeper look still calls a clear mistake
    sel = known & dis
    return {
        "child_n": int(sel.sum()),
        "child_loss_mean": float(cl[sel].mean()) if sel.any() else float("nan"),
        "child_loss_median": float(np.median(cl[sel])) if sel.any() else float("nan"),
        "child_gt_10": float((cl[sel] > 0.10).mean()) if sel.any() else float("nan"),
        "child_expert_better": float((cl[sel] < -0.02).mean()) if sel.any() else float("nan"),
        "root_loss_on_same": float(a.loss[m][sel][~np.isnan(a.loss[m][sel])].mean()) if sel.any() else float("nan"),
    }


def _fmt_table(rows: list[tuple[str, dict[str, float]]]) -> str:
    head = "| slice | n | top-1 | top-1 (visits) | top-3 | raw prior top-1 | expert visit share | mean loss | median loss | p90 loss | loss>0.05 | loss>0.10 | loss>0.20 | mean pts | unvisited |"
    sep = "|" + "---|" * 15
    out = [head, sep]
    for name, r in rows:
        if r["n"] == 0:
            continue
        out.append(
            f"| {name} | {r['n']:,} | {r['top1']:.1%} | {r['top1_visits']:.1%} | {r['top3']:.1%} | "
            f"{r['prior_top1']:.1%} | {r['expert_visit_share']:.1%} | {r['loss_mean']:.3f} | {r['loss_median']:.3f} | "
            f"{r['loss_p90']:.3f} | {r['loss_gt_05']:.1%} | {r['loss_gt_10']:.1%} | {r['loss_gt_20']:.1%} | "
            f"{r['points_mean']:+.2f} | {r['unvisited']:.1%} |"
        )
    return "\n".join(out)


def _fmt_child_table(rows: list[tuple[str, dict[str, float]]]) -> str:
    head = "| slice | disagreements searched | root-Q loss (same rows) | child-search loss, mean | median | child loss > 0.10 | expert's child better (loss < -0.02) |"
    out = [head, "|" + "---|" * 7]
    for name, r in rows:
        if not r.get("child_n"):
            continue
        out.append(
            f"| {name} | {r['child_n']:,} | {r['root_loss_on_same']:.3f} | {r['child_loss_mean']:.3f} | "
            f"{r['child_loss_median']:.3f} | {r['child_gt_10']:.1%} | {r['child_expert_better']:.1%} |"
        )
    return "\n".join(out) if len(out) > 2 else ""


def agreement_tables(a: Atlas) -> tuple[str, dict[str, Any]]:
    parts = []
    numbers: dict[str, Any] = {}
    rounds = sorted(set(int(r) for r in a.d["round"]))
    for r in rounds:
        rows = []
        max_move = int(a.d["mover_move"][a.d["round"] == r].max())
        for k in range(1, max_move + 1):
            m = a.mask(True, (r,), (k,))
            if m.sum() < 30:
                continue
            rows.append((f"round {r + 1}, move {k}", _row(a, m)))
        rows.append((f"round {r + 1}, moves 1-3", _row(a, a.mask(True, (r,), (1, 2, 3)))))
        rows.append((f"round {r + 1}, all", _row(a, a.mask(True, (r,)))))
        rows.append((f"round {r + 1}, opponent seat", _row(a, a.mask(False, (r,)))))
        numbers[f"round{r + 1}"] = {name: row for name, row in rows}
        parts.append(f"**Round {r + 1}** (the target seat, i.e. the expert; the last row is the other seat for contrast)\n\n" + _fmt_table(rows))
        child = _fmt_child_table(rows)
        if child:
            parts.append(f"**Round {r + 1}, the disagreements re-searched from the child positions** "
                         f"(a full {a.meta['sims']:,}-sim search after the expert's move and after the net's; "
                         f"round-ending moves excluded)\n\n" + child)
    # by Elo bucket, round 1 moves 1-3
    elo = a.d["target_elo"]
    buckets = [("expert Elo < 2200", elo < 2200), ("2200-2350", (elo >= 2200) & (elo < 2350)), ("≥ 2350", elo >= 2350)]
    rows = [(name, _row(a, a.mask(True, (0,)) & b)) for name, b in buckets]
    numbers["round1_by_elo"] = {name: row for name, row in rows}
    parts.append("**Round 1 by the expert's Elo** (the source ladder's raw scale; the pipeline's floor is 1950)\n\n" + _fmt_table(rows))
    # by disagreement type
    return "\n\n".join(parts), numbers


def loss_distribution(a: Atlas, m: np.ndarray) -> tuple[str, dict[str, Any]]:
    loss = a.loss[m]
    known = ~np.isnan(loss)
    loss = loss[known]
    pts = a.points_loss[m][known]
    qs = [0.5, 0.75, 0.9, 0.95, 0.99]
    lines = ["| quantile | value loss | points |", "|---|---|---|"]
    for q in qs:
        lines.append(f"| p{int(q * 100)} | {np.quantile(loss, q):.3f} | {np.quantile(pts, q):+.2f} |")
    lines.append(f"| mean | {loss.mean():.3f} | {pts.mean():+.2f} |")
    disagree = ~a.agree[m][known]
    lines.append("")
    lines.append(f"Disagreements: {disagree.mean():.1%} of decisions. Among them the mean value loss is "
                 f"{loss[disagree].mean():.3f} ({pts[disagree].mean():+.2f} points); "
                 f"{(loss[disagree] <= 0.02).mean():.1%} are within 0.02 of the net's own pick (a coin flip to the net), "
                 f"{(loss[disagree] > 0.10).mean():.1%} are what the net calls a clear mistake (> 0.10), "
                 f"{(loss[disagree] > 0.20).mean():.1%} a blunder (> 0.20).")
    nums = {
        "n": int(known.sum()),
        "quantiles": {f"p{int(q * 100)}": float(np.quantile(loss, q)) for q in qs},
        "mean": float(loss.mean()),
        "points_mean": float(pts.mean()),
        "disagree_frac": float(disagree.mean()),
        "disagree_loss_mean": float(loss[disagree].mean()) if disagree.any() else None,
        "disagree_within_002": float((loss[disagree] <= 0.02).mean()) if disagree.any() else None,
        "disagree_gt_010": float((loss[disagree] > 0.10).mean()) if disagree.any() else None,
        "disagree_gt_020": float((loss[disagree] > 0.20).mean()) if disagree.any() else None,
    }
    return "\n".join(lines), nums


# ------------------------------------------------------------ disagreement patterns
def _pattern(expert: int, net: int) -> str:
    es, ec, ed = decode_action(expert)
    ns, nc, nd = decode_action(net)
    parts = []
    if ec != nc:
        parts.append("different colour")
    else:
        parts.append("same colour")
    if (es == CENTER) != (ns == CENTER):
        parts.append("expert center, net factory" if es == CENTER else "expert factory, net center")
    elif es != ns:
        parts.append("different factory")
    if ed == FLOOR and nd != FLOOR:
        parts.append("expert floors it, net builds")
    elif nd == FLOOR and ed != FLOOR:
        parts.append("expert builds, net floors it")
    elif ed != nd:
        parts.append(f"row {ed + 1} vs row {nd + 1}" + (" (expert lower)" if ed > nd else " (expert higher)"))
    else:
        parts.append("same row")
    return "; ".join(parts)


def pattern_table(a: Atlas, m: np.ndarray, threshold: float = 0.05) -> tuple[str, dict[str, int]]:
    sel = m & ~a.agree & ~np.isnan(a.loss) & (a.loss > threshold)
    counts: Counter[str] = Counter()
    coarse: Counter[str] = Counter()
    for i in np.flatnonzero(sel):
        p = _pattern(int(a.expert[i]), int(a.net_move[i]))
        counts[p] += 1
        es, ec, ed = decode_action(int(a.expert[i]))
        ns, nc, nd = decode_action(int(a.net_move[i]))
        if ed == FLOOR and nd != FLOOR:
            coarse["expert floors, net builds"] += 1
        elif nd == FLOOR and ed != FLOOR:
            coarse["expert builds, net floors"] += 1
        elif ec == nc and ed == nd:
            coarse["same colour and row, other source"] += 1
        elif ec == nc:
            coarse["same colour, other row" + (" (expert lower row)" if ed > nd else " (expert higher row)")] += 1
        else:
            coarse["different colour"] += 1
        if (es == CENTER) != (ns == CENTER):
            coarse["center vs factory: " + ("expert center" if es == CENTER else "net center")] += 1
    total = int(sel.sum())
    lines = [f"Disagreements with value loss > {threshold} (target seat): {total:,}. Coarse patterns (a disagreement can count in two rows):", "",
             "| pattern | count | share |", "|---|---|---|"]
    for k, v in coarse.most_common():
        lines.append(f"| {k} | {v} | {v / max(total, 1):.1%} |")
    lines += ["", "Finest patterns, top 12:", "", "| pattern | count |", "|---|---|"]
    for k, v in counts.most_common(12):
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines), dict(coarse)


# ----------------------------------------------------------- the disagreements
def top_disagreements(a: Atlas, m: np.ndarray, top: int, games: dict[int, Any]) -> tuple[str, list[dict[str, Any]]]:
    use_child = not np.all(np.isnan(a.child_loss[m]))
    sel = m & ~a.agree & ~np.isnan(a.child_loss if use_child else a.loss)
    idx = np.flatnonzero(sel)
    rank_key = a.child_loss if use_child else a.loss
    idx = idx[np.argsort(-rank_key[idx])][:top]
    out = []
    records = []
    for rank, i in enumerate(idx, 1):
        table = int(a.d["table"][i])
        pos = int(a.d["index"][i])
        game = games[table]
        states, _movers, _final = replay_positions(game)
        state = states[pos]
        e = int(a.expert[i])
        n = int(a.net_move[i])
        v = a.d["visits"][i]
        total = max(1, int(v.sum()))
        rec = {
            "rank": rank,
            "table": table,
            "index": pos,
            "round": int(a.d["round"][i]) + 1,
            "move": int(a.d["mover_move"][i]),
            "elo": float(a.d["target_elo"][i]),
            "expert": e,
            "net": n,
            "expert_text": describe_move(state, e),
            "net_text": describe_move(state, n),
            "q_expert": float(a.q_expert[i]),
            "q_net": float(a.q_net[i]),
            "loss": float(a.loss[i]),
            "points": float(a.points_loss[i]),
            "expert_visits": int(v[e]),
            "net_visits": int(v[n]),
            "root_value": float(a.d["value"][i]),
            "child_value_expert": float(a.child_value_expert[i]) if use_child else None,
            "child_value_net": float(a.child_value_net[i]) if use_child else None,
            "child_loss": float(a.child_loss[i]) if use_child else None,
            "prior_expert": float(a.d["prior"][i][e]),
            "prior_net": float(a.d["prior"][i][n]),
            "outcome_for_mover": float(game.outcome * (1 if a.d["mover"][i] == 0 else -1)),
            "final_scores": list(game.scores),
        }
        records.append(rec)
        game_no = list(games).index(table)  # an index into the compact file, not the source's id
        rec["game_no"] = game_no
        out.append(
            f"### {rank}. Round {rec['round']}, the expert's move {rec['move']} (game {game_no}, position {pos}; expert Elo {rec['elo']:.0f})\n\n"
            f"- **Expert took** {rec['expert_text']}. Net's Q for it {rec['q_expert']:+.3f}, "
            f"{rec['expert_visits'] / total:.1%} of visits, raw prior {rec['prior_expert']:.1%}.\n"
            f"- **Net wanted** {rec['net_text']}. Q {rec['q_net']:+.3f}, {rec['net_visits'] / total:.1%} of visits, "
            f"raw prior {rec['prior_net']:.1%}.\n"
            + (f"- Searched from the children: after the expert's move {rec['child_value_expert']:+.3f}, after the net's {rec['child_value_net']:+.3f}, "
               f"**child-search loss {rec['child_loss']:.3f}**.\n" if use_child else "")
            + f"- Root-edge value loss **{rec['loss']:.3f}** ({rec['points']:+.1f} points by the margin head); root value {rec['root_value']:+.3f}. "
            f"The expert {'won' if rec['outcome_for_mover'] > 0 else 'lost' if rec['outcome_for_mover'] < 0 else 'drew'} the game "
            f"{game.scores[0]}-{game.scores[1]}.\n\n"
            "```\n" + state.render_text() + "\n```\n"
        )
    return "\n".join(out), records


# --------------------------------------------------------------------- report
def write_report(a: Atlas, out: Path, top: int, positions_path: Path) -> dict[str, Any]:
    games = {g.table_id: g for g in load_positions(positions_path)}
    tables, numbers = agreement_tables(a)
    m_r1 = a.mask(True, (0,))
    m_all = a.mask(True)
    dist_r1, dist_r1_n = loss_distribution(a, m_r1)
    dist_all, dist_all_n = loss_distribution(a, m_all)
    patterns_r1, pat_n = pattern_table(a, m_r1)
    dis_r1, dis_records = top_disagreements(a, m_r1, top, games)
    dis_r2, dis_records2 = top_disagreements(a, a.mask(True, (1,)), max(5, top // 2), games)
    meta = a.meta
    md = [
        f"# Opening atlas: {meta['net']} at {meta['sims']:,} sims",
        "",
        f"*Generated from `{a.path}`: {a.n:,} positions of the expert games in rounds "
        f"{meta['rounds'][0] + 1}-{meta['rounds'][1] + 1} (both seats), {int(a.d['is_target'].sum()):,} of them the expert's "
        f"(the dataset's target seat). Search: no root noise, {meta['sims']:,} simulations per position, "
        f"c_puct {a.mcts.get('c_puct')}, {a.mcts.get('chance_children')} determinizations per refill edge. "
        f"Labelled in {meta['seconds'] / 60:.0f} min on {meta['device']}.*",
        "",
        "Columns: **top-1** = the move the net would play (decisive pick) equals the expert's; "
        "**top-1 (visits)** = most-visited move equals the expert's; **top-3** = the expert's move is among the three "
        "most visited; **raw prior top-1** = the policy head alone, no search; **expert visit share** = the fraction "
        "of the search that went into the expert's move; **loss** = Q(net's move) - Q(expert's move) in the mover's "
        "frame, win-probability units on [-1, 1] (0.10 ≈ 5 percentage points of win chance); **pts** = the same gap "
        "read from the margin head, in final-score points; **unvisited** = the search never tried the expert's move "
        "(the loss is then unknown and excluded).",
        "",
        "## 1. Agreement by move",
        "",
        tables,
        "",
        "## 2. How much the net thinks the expert lost",
        "",
        "**Round 1, expert's decisions**",
        "",
        dist_r1,
        "",
        "**Rounds 1-2, expert's decisions**",
        "",
        dist_all,
        "",
        "## 3. What the disagreements look like (round 1)",
        "",
        patterns_r1,
        "",
        f"## 4. The {top} largest round-1 disagreements",
        "",
        "Boards are printed before the move, from the engine's text renderer: each player's five pattern lines on the "
        "left (`.` empty slot), the wall on the right (`.` empty), colours B=blue Y=yellow R=red K=black T=teal, "
        "`#` on the floor = the first-player marker; `*` marks the player to move.",
        "",
        dis_r1,
        "",
        f"## 5. The largest round-2 disagreements",
        "",
        dis_r2,
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    summary = {
        "atlas": str(a.path),
        "meta": meta,
        "tables": numbers,
        "loss_round1": dist_r1_n,
        "loss_rounds12": dist_all_n,
        "patterns_round1": pat_n,
        "disagreements_round1": dis_records,
        "disagreements_round2": dis_records2,
    }
    # the JSON carries the source's table ids: it stays with the data, not the docs
    json_out = Path("data/cloud/opening") / (out.stem + ".report.json")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=1))
    summary["json"] = str(json_out)
    return summary


# ------------------------------------------------------------------ compare
def compare(paths: list[Path], target_only: bool = True) -> str:
    atlases = [Atlas(p) for p in paths]
    keysets = [set(a.keys().tolist()) for a in atlases]
    common = set.intersection(*keysets)
    lines = [f"Common positions: {len(common):,}", ""]
    base = atlases[0]
    m0 = np.isin(base.keys(), list(common))
    if target_only:
        m0 &= base.d["is_target"]
    # align others to base order
    order_keys = base.keys()[m0]
    aligned = []
    for a in atlases:
        pos = {k: i for i, k in enumerate(a.keys().tolist())}
        aligned.append(np.array([pos[k] for k in order_keys.tolist()]))
    head = "| slice | " + " | ".join(f"{a.meta['net']}@{a.meta['sims']} top-1 / mean loss / loss>0.10" for a in atlases) + " |"
    lines += [head, "|---|" + "---|" * len(atlases)]
    rounds = sorted(set(int(r) for r in base.d["round"][m0]))
    slices = []
    for r in rounds:
        for k in (1, 2, 3):
            slices.append((f"round {r + 1}, move {k}", (r,), (k,)))
        slices.append((f"round {r + 1}, all", (r,), None))
    for name, rr, kk in slices:
        sel = np.isin(base.d["round"][m0], rr)
        if kk is not None:
            sel &= np.isin(base.d["mover_move"][m0], kk)
        if sel.sum() < 20:
            continue
        cells = []
        for a, idx in zip(atlases, aligned):
            ii = idx[sel]
            loss = a.loss[ii]
            known = ~np.isnan(loss)
            cells.append(f"{a.agree[ii].mean():.1%} / {loss[known].mean():.3f} / {(loss[known] > 0.10).mean():.1%}")
        lines.append(f"| {name} ({int(sel.sum()):,}) | " + " | ".join(cells) + " |")
    # cross agreement
    if len(atlases) >= 2:
        lines.append("")
        a0, a1 = atlases[0], atlases[1]
        i0, i1 = aligned[0], aligned[1]
        both = a0.agree[i0] & a1.agree[i1]
        only0 = a0.agree[i0] & ~a1.agree[i1]
        only1 = ~a0.agree[i0] & a1.agree[i1]
        neither = ~a0.agree[i0] & ~a1.agree[i1]
        same_net = a0.net_move[i0] == a1.net_move[i1]
        lines.append(
            f"{a0.meta['net']} vs {a1.meta['net']} on the same {len(i0):,} expert decisions: both agree with the expert "
            f"{both.mean():.1%}, only {a0.meta['net']} {only0.mean():.1%}, only {a1.meta['net']} {only1.mean():.1%}, "
            f"neither {neither.mean():.1%}; the two nets pick the same move {same_net.mean():.1%} of the time, "
            f"and among the positions where neither matches the expert they still agree with each other "
            f"{same_net[neither].mean():.1%}."
        )
        # where a0 disagrees with the expert: what does a1 say about the loss?
        d0 = ~a0.agree[i0] & ~np.isnan(a0.loss[i0]) & ~np.isnan(a1.loss[i1])
        if d0.any():
            lines.append(
                f"Where {a0.meta['net']} disagrees with the expert (n={int(d0.sum()):,}), its mean loss for the expert's move is "
                f"{a0.loss[i0][d0].mean():.3f}; {a1.meta['net']}'s loss for the same expert moves is {a1.loss[i1][d0].mean():.3f}, "
                f"and {a1.meta['net']} sides with the expert in {a1.agree[i1][d0].mean():.1%} of them."
            )
            big = d0 & (a0.loss[i0] > 0.10)
            if big.any():
                lines.append(
                    f"Restricted to {a0.meta['net']}'s clear disagreements (loss > 0.10, n={int(big.sum()):,}): "
                    f"{a1.meta['net']} loss {a1.loss[i1][big].mean():.3f}, sides with the expert {a1.agree[i1][big].mean():.1%}, "
                    f"calls it > 0.10 too in {(a1.loss[i1][big] > 0.10).mean():.1%}."
                )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ludometer.opening.report")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("atlas")
    r.add_argument("npz", type=Path)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--top", type=int, default=20)
    r.add_argument("--positions", type=Path, default=POSITIONS)
    r.add_argument("--children", type=Path, default=None, help="an `atlas children` file for the same atlas")
    c = sub.add_parser("compare")
    c.add_argument("npz", type=Path, nargs="+")
    c.add_argument("--all-seats", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "atlas":
        a = Atlas(args.npz)
        if args.children is not None:
            print(f"children: {a.attach_children(args.children):,} rows covered")
        summary = write_report(a, args.out, args.top, args.positions)
        print(f"wrote {args.out} and {summary['json']}")
        r1 = summary["tables"]["round1"]
        for k, v in r1.items():
            if v["n"]:
                print(f"{k:>28}: n={v['n']:6,} top1={v['top1']:.1%} top3={v['top3']:.1%} prior={v['prior_top1']:.1%} loss={v['loss_mean']:.3f} >0.10={v['loss_gt_10']:.1%}")
        return 0
    print(compare(args.npz, target_only=not args.all_seats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
