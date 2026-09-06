#!/usr/bin/env python3
"""Regenerate web/experiments.html — the human-data experiments board.

One page, three figures, same visual language as the training monitor (this
script imports web/make_dashboard.py for its tokens, plot engine, hover layer
and CSS; the categorical palette is the dashboard's, re-validated against the
dark surface with the dataviz validator — all checks pass):

  A. the self-play curves of the pretrain A/B (hp2 vs run6 control), the
     mid-size distill+polish candidate (mid1, live) and the pure-BC arm (hp1),
     with the deployed net and the best human-fine-tuned net as reference lines;
  B. fine-tune candidates against their own base — did human data make the
     *same* net stronger? (>50% = yes);
  C. the deploy gate — every candidate against the LIVE site net, split by the
     test that matters: equal search vs equal think time (wall clock).

Stdlib only. Reads runs/<run>/elo.jsonl and data/human/gauntlet_*.json.

Usage:
    python3 web/make_experiments.py               # write once
    python3 web/make_experiments.py --watch 60    # rewrite forever
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_dashboard as md  # noqa: E402  (tokens, Plot, css, hover JS)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "web" / "experiments.html"

BLUE, ORANGE, AQUA, YELLOW = md.SERIES[0], md.SERIES[1], md.SERIES[2], md.SERIES[3]

DEPLOYED = 2360.6  # run4/ckpt-037888, the live site net (gauntlet anchor)
BEST_NET = 2445.0  # ft2/ft-004000, chained fit estimate (equal search)


# ---------------------------------------------------------------- data ----
def elo_curve(run: str):
    path = REPO / "runs" / run / "elo.jsonl"
    points = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                d = json.loads(line)
                points.append((float(d["games"]), float(d["elo"]), float(d.get("elo_err") or 0.0)))
            except (ValueError, KeyError):
                continue
    return points


def match(name: str):
    path = REPO / "data" / "human" / f"gauntlet_{name}.json"
    if not path.exists():
        return None
    try:
        m = json.loads(path.read_text())["matches"][0]
    except (ValueError, KeyError, IndexError):
        return None
    p, n = float(m["win_rate"]), int(m["n_games"])
    ci = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / n)
    return {"p": p, "n": n, "ci": ci, "w": m["wins"], "d": m["draws"], "l": m["losses"]}


# rows: (json name, label, condition) — condition drives the color.
SEARCH, CLOCK = "equal search", "equal think time"
VS_BASE = [
    ("ft-000250", "ft1-250 vs run5-best", SEARCH),
    ("ft-000500", "ft1-500 vs run5-best", SEARCH),
    ("ft-001000", "ft1-1000 vs run5-best", SEARCH),
    ("ft2000_deep", "ft1-2000 vs run5-best", SEARCH),
    ("ft2000_think", "ft1-2000 vs run5-best @1s", CLOCK),
    ("ft2-003000", "ft2-3000 vs ft1-2000", SEARCH),
    ("ft2-004000", "ft2-4000 vs ft1-2000", SEARCH),
    ("ft2-006000", "ft2-6000 vs ft1-2000", SEARCH),
]
VS_SITE = [
    ("baseline", "run5-best (no human data)", SEARCH),
    ("ft2000_vs_site", "ft1-2000 (7.0M params)", SEARCH),
    ("ft3b1000_deep", "ft3b-1000 (site-size)", SEARCH),
    ("ft3-002000", "ft3-2000 (site-size)", SEARCH),
    ("ds1-003000", "ds1-3000 (distilled, site-size)", SEARCH),
    ("ft2000_wallclock", "ft1-2000 (7.0M params)", CLOCK),
    ("ds2-008000", "ds2-8000 (distilled, 2.9M)", CLOCK),
    ("ds2-014000", "ds2-14000 (distilled, 2.9M)", CLOCK),
    ("ds2-020000", "ds2-20000 (distilled, 2.9M)", CLOCK),
    ("mid1_wallclock", "mid1-best (distill+polish, 2.9M)", CLOCK),
]


# ------------------------------------------------------------- figures ----
def curves_figure() -> str:
    series = [
        ("run6 (control: self-play pretrain)", "run6", BLUE),
        ("hp2 (26% human pretrain mix)", "hp2", ORANGE),
        ("mid1 (2.9M distill + polish, 8k games)", "mid1", AQUA),
        ("mid2 (2.9M, new corpus teacher, 20k games)", "mid2", YELLOW),
    ]
    # hp1 (the human-only pretrain arm, aborted at 1.5k games) starts at 1315 —
    # plotting it would squash the ±100-Elo band the A/B actually lives in, so
    # it rides in the data table instead.
    hp1 = elo_curve("hp1")
    present = [(label, run, color, elo_curve(run)) for label, run, color in series]
    present = [row for row in present if row[3]]
    if not present:
        return md.empty_figure("Training curves", "", "no elo.jsonl found", wide=True)
    xs = [p[0] for _, _, _, pts in present for p in pts]
    ys = [p[1] for _, _, _, pts in present for p in pts] + [DEPLOYED, BEST_NET]
    plot = md.Plot(width=1080, height=360)
    xt, yt = plot.scale(xs, ys)
    plot.frame(xt, yt, md.fmt_compact, md.fmt_compact, "self-play games", "internal Elo (100-sim gauntlet)")
    plot.hline(DEPLOYED, "deployed site net (run4) · 2361")
    plot.hline(BEST_NET, "best net overall: ft2-4000 human fine-tune · ~2445 (equal search, est.)")
    for label, _, color, pts in present:
        xy = [(x, y) for x, y, _ in pts]
        plot.errbars([(x, y, e) for x, y, e in pts], color)
        plot.line(xy, color)
        plot.dots([xy[-1]], color, radius=3.5)
        plot.register(label, color, [(x, y, f"{md.fmt_compact(x)} games", f"{y:.0f} Elo") for x, y, _ in pts])
    legend_html = md.legend([(label, color, "line") for label, _, color, _ in present])
    rows = []
    for label, _, _, pts in present:
        rows.append([label, md.fmt_compact(pts[-1][0]), f"{pts[-1][1]:.1f} ± {pts[-1][2]:.0f}", f"{max(p[1] for p in pts):.1f}"])
    if hp1:
        rows.append([
            "hp1 (human-only pretrain, aborted — below this frame)",
            md.fmt_compact(hp1[-1][0]),
            f"{hp1[-1][1]:.1f} ± {hp1[-1][2]:.0f}",
            f"{max(p[1] for p in hp1):.1f}",
        ])
    table_html = md.table(["arm", "games", "latest Elo", "best Elo"], rows)
    return md.figure(
        "Pretrain A/B and the deploy candidate",
        "hp2 = run6's exact config with 26% elite-human rows mixed into the warm start: it pays −58 Elo "
        "at game 0, catches up by ~2.3k games and its second half averages ~+55 over the control. "
        "mid1 distills the best human-fine-tuned net into a deployable 2.9M-param body, then self-play polishes it.",
        plot.svg("Elo versus self-play games for the experiment arms"),
        legend_html=legend_html,
        table_html=table_html,
        wide=True,
    )


def dot_figure(title, subtitle, rows, note="") -> str:
    data = [(label, cond, match(name)) for name, label, cond in rows]
    data = [(label, cond, m) for label, cond, m in data if m]
    if not data:
        return md.empty_figure(title, subtitle, "no gauntlet results yet")
    plot = md.Plot(width=520, height=64 + 26 * len(data), left=232, right=18, top=14, bottom=36)
    plot.xlo, plot.xhi = 0.15, 0.85
    plot.ylo, plot.yhi = -0.6, len(data) - 0.4
    for value in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        x = plot.X(value)
        cls = "ref" if abs(value - 0.5) < 1e-9 else "grid"
        plot.back.append(
            f'<line x1="{md.coord(x)}" y1="{md.coord(plot.y0)}" x2="{md.coord(x)}" '
            f'y2="{md.coord(plot.y0 + plot.ph)}" class="{cls}" />'
        )
        plot.back.append(
            f'<text x="{md.coord(x)}" y="{md.coord(plot.y0 + plot.ph + 18)}" class="tick tick-x">'
            f"{value * 100:.0f}%</text>"
        )
    hover = []
    for i, (label, cond, m) in enumerate(data):
        y = len(data) - 1 - i  # first row on top
        color = BLUE if cond == SEARCH else ORANGE
        cy = plot.Y(y)
        lo, hi = plot.X(max(plot.xlo, m["p"] - m["ci"])), plot.X(min(plot.xhi, m["p"] + m["ci"]))
        plot.marks.append(
            f'<line x1="{md.coord(lo)}" y1="{md.coord(cy)}" x2="{md.coord(hi)}" y2="{md.coord(cy)}" '
            f'stroke="{color}" stroke-width="1.5" opacity="0.45" />'
        )
        plot.dots([(m["p"], y)], color, radius=4.5)
        plot.back.append(
            f'<text x="{md.coord(plot.x0 - 10)}" y="{md.coord(cy + 3.5)}" class="tick tick-y" '
            f'text-anchor="end">{md.esc(label)}</text>'
        )
        hover.append((m["p"], y, label, f"{m['p'] * 100:.1f}% over {m['n']} games ({m['w']}-{m['d']}-{m['l']})"))
    for cond, color in ((SEARCH, BLUE), (CLOCK, ORANGE)):
        pts = [h for h, (_, c, _) in zip(hover, data) if c == cond]
        if pts:
            plot.register(cond, color, pts)
    plot.front.append(
        f'<text x="{md.coord(plot.x0 + plot.pw)}" y="{md.coord(plot.height - 4)}" class="axis-title" '
        f'text-anchor="end">win rate (whiskers: 95% CI)</text>'
    )
    legend_html = md.legend([(SEARCH + " (sims=100 each)", BLUE, "dot"), (CLOCK + " (think=1.0s each)", ORANGE, "dot")])
    table_html = md.table(
        ["candidate", "condition", "win rate", "games"],
        [[label, cond, f"{m['p'] * 100:.1f}%", str(m["n"])] for label, cond, m in data],
    )
    return md.figure(title, subtitle + (" " + note if note else ""), plot.svg(title), legend_html=legend_html, table_html=table_html)


def page() -> str:
    generated = md.stamp_text(time.time())
    figures = [
        curves_figure(),
        dot_figure(
            "Does human data strengthen the same net?",
            "Each fine-tuned checkpoint against the exact net it started from. Above the 50% line = the "
            "human data made it stronger. ft1 fine-tunes run5-best (7.0M params); ft2 continues on the "
            "dual-target buffer and beats even ft1-2000.",
            VS_BASE,
        ),
        dot_figure(
            "The deploy gate: against the live site net",
            "Blue is the flattering test (same simulation count); orange is the honest one — same wall-clock "
            "think time, which is what a visitor experiences. Nothing orange has crossed 50% yet, so the site "
            "still runs run4. mid1 (chart above) is the current attempt to change that.",
            VS_SITE,
        ),
    ]
    body = [
        '<div class="wrap">',
        "<header><h1>Human-data experiments</h1>"
        '<p class="sub">2,797 elite BGA games (ranks 1-2) + 343 humans-beat-the-bot Faïence games → '
        "90,475 policy targets · Porcelain bar: +150 wall-clock Elo vs the site net (docs/BOT_DEPLOYMENT.md) · "
        "full write-up in docs/HUMAN_GAMES.md §18 · page auto-refreshes</p></header>",
        '<div class="grid">',
        *figures,
        "</div>",
        '<footer class="foot"><span>ludometer · generated by web/make_experiments.py</span>'
        f"<span>static page, no external requests</span><span>{md.esc(generated)}</span></footer>",
        "</div>",
    ]
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8" />\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        '<meta http-equiv="refresh" content="30" />\n'
        "<title>Ludometer Human-Data Experiments</title>\n"
        f"<style>{md.styles()}</style>\n"
        "</head><body>\n" + "".join(body) + f"\n<script>{md.JS}</script>\n</body></html>\n"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", type=int, default=0, help="rewrite every N seconds")
    args = parser.parse_args(argv)
    while True:
        md.write_atomic(OUT, page())
        print(f"wrote {OUT}")
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
