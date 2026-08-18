#!/usr/bin/env python3
"""Regenerate web/compare.html — the cross-game learning-curve comparison.

This is the page NEXT_GAMES.md §4 specifies, and it enforces that section's
rules by construction:

* **Elo is never drawn on a shared axis across games** — absolute curves are
  small multiples, one panel per game, each on its own scale.
* **The headline is the normalised shape**: fraction of the run's own range
  (0 = its games-0 rating, 1 = its best checkpoint) against fraction of its
  budget, with the slope-half point marked per run.
* **The x axis is decisions**; games ride along in every hover and in the
  table. For runs whose elo.jsonl predates the native ``decisions`` field the
  conversion is a measured per-run constant (see RUN_REFS), which makes the
  two axes affine — a shape conclusion survives on both by construction.
* Azul **run1 is the primary** from-scratch reference and run2 is drawn as a
  muted context line (same game, 256 vs 160 sims), never a competitor.

Reuses the Plot/figure/style machinery of make_dashboard.py so the page looks
and behaves like the rest of the project (inline SVG, hover, no requests).

Usage:
    python3 web/make_compare.py            # write web/compare.html once
    python3 web/make_compare.py --watch 60 # rewrite forever (during runs)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_dashboard as dash  # noqa: E402  (repo-local sibling script)

REPO = Path(__file__).resolve().parent.parent

# Fixed slot per *game* (never cycled; a run inherits its game's hue).
GAME_COLOR = {
    "azul": dash.SERIES[0],  # blue
    "uno": dash.SERIES[1],  # orange
    "unoplus": dash.SERIES[2],  # aqua
    "tictactoe": dash.SERIES[3],  # yellow
    "connect4": dash.SERIES[4],  # magenta
}
CONTEXT = dash.MUTED  # run2 and other same-game context lines


@dataclass
class RunRef:
    name: str
    game: str
    label: str
    #: decisions per game for elo.jsonl rows that predate the native field.
    #: run1/run2: 52.98 positions/game measured from the checkpoints, branching
    #: is >1 essentially always so decisions ~= positions. uno1: 43.9 moves per
    #: hand (measured, checkpoints) x 0.428 searched fraction (measured with
    #: the trained net at 96 sims) = 18.8.
    decisions_per_game: float
    context: bool = False  # drawn muted, excluded from the headline claims


RUN_REFS = [
    RunRef("run1", "azul", "Azul run1", 52.98),
    RunRef("run2", "azul", "Azul run2 (256 sims)", 52.98, context=True),
    RunRef("uno1", "uno", "Uno", 18.8),
    RunRef("unoplus1", "unoplus", "Uno+ (house rules)", 40.0),
    RunRef("ttt1", "tictactoe", "Tic-tac-toe", 5.5),
    RunRef("c4_1", "connect4", "Connect Four", 30.0),
]

#: pre-rated baseline reference lines per game (from the journal's gauntlets)
BASELINE_ELOS = {
    "uno": (("uno:greedy", 572.0), ("uno:heuristic", 396.0)),
    # measured by gauntlet (200 games/pairing, random=0): perfect play is
    # ratable because the trained net draws it - +527 +/- 25, and the net at
    # 256 sims sits at +519: the ceiling is reached, not just visible.
    "tictactoe": (("perfect play", 526.6),),
}


@dataclass
class Curve:
    ref: RunRef
    points: list  # (games, decisions, elo, err)
    state: str

    @property
    def color(self) -> str:
        return CONTEXT if self.ref.context else GAME_COLOR[self.ref.game]


def load_curves() -> list[Curve]:
    curves = []
    for ref in RUN_REFS:
        run_dir = REPO / "runs" / ref.name
        rows, _ = dash.read_jsonl(run_dir / "elo.jsonl")
        points = []
        for row in rows:
            games = dash.num(row, "games")
            elo = dash.num(row, "elo")
            if games is None or elo is None:
                continue
            decisions = dash.num(row, "decisions")
            if decisions is None:
                decisions = games * ref.decisions_per_game
            points.append((games, decisions, elo, dash.num(row, "elo_err") or 0.0))
        if len(points) < 3:
            continue
        status = dash.read_json(run_dir / "status.json")
        curves.append(Curve(ref, points, str(status.get("state", ""))))
    return curves


# ---------------------------------------------------------------- statistics
def smooth(values: list[float], window: int = 5) -> list[float]:
    half = window // 2
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def normalised(curve: Curve) -> list[tuple[float, float]]:
    """(fraction of budget in decisions, fraction of own Elo range)."""
    xs = [p[1] for p in curve.points]
    ys = smooth([p[2] for p in curve.points])
    x_end = xs[-1]
    y0 = ys[0]
    y_best = max(ys)
    span = y_best - y0
    if x_end <= 0 or span <= 0:
        return []
    return [(x / x_end, (y - y0) / span) for x, y in zip(xs, ys)]


def slope_half_point(curve: Curve) -> float | None:
    """Fraction of budget where the smoothed slope first halves (and stays).

    Method (stated on the page): 5-point moving average of Elo, per-interval
    slope in Elo/decision, early slope = mean over the first quarter of the
    budget, and the report is the first point after that quarter where the
    slope stays below half of the early slope for three intervals running.
    """
    pts = normalised(curve)
    if len(pts) < 8:
        return None
    slopes = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x1 - x0 <= 0:
            continue
        slopes.append(((x0 + x1) / 2, (y1 - y0) / (x1 - x0)))
    early = [s for x, s in slopes if x <= 0.25]
    if not early:
        return None
    threshold = (sum(early) / len(early)) / 2.0
    run_length = 0
    for i, (x, s) in enumerate(slopes):
        if x <= 0.25:
            continue
        run_length = run_length + 1 if s < threshold else 0
        if run_length >= 3:
            return slopes[i - 2][0]
    return None


def truncation_r2(curve: Curve) -> dict[str, float | None]:
    """R² of a linear Elo-vs-decisions fit over growing prefixes (the §4 trap:
    every curve is straight at the start, so R² mostly measures truncation)."""
    xs = [p[1] for p in curve.points]
    ys = [p[2] for p in curve.points]
    out = {}
    for frac, key in ((0.25, "r2_25"), (0.5, "r2_50"), (1.0, "r2_full")):
        cut = max(3, int(len(xs) * frac))
        fit = dash.linfit(xs[:cut], ys[:cut])
        out[key] = fit[2] if fit else None
    # Elo against log(decisions): the "you learn fast early, slowly later"
    # hypothesis, as a number beside the straight-line fit.
    logs = [(math.log(x), y) for x, y in zip(xs, ys) if x > 0]
    fit = dash.linfit([x for x, _ in logs], [y for _, y in logs])
    out["r2_log"] = fit[2] if fit else None
    return out


# ------------------------------------------------------------------- panels
def combined_panel(curves: list[Curve]) -> str:
    """THE chart (Rémi's call): every run's absolute Elo on one shared axis.

    Each game is still anchored to its own ladder (random = 0), so the height
    of a curve reads as *how much distinguishable skill above random the game
    offers* — one curve sitting far below another is the reality of the depth
    of the game, not a scoring artifact. The remaining caveat stays in the
    subtitle: match-based rating stretches a per-hand edge, so cross-game
    heights are indicative, not exchangeable.
    """
    plot = dash.Plot(width=1080, height=420)
    xs = [p[1] for c in curves for p in c.points]
    ys = [p[2] for c in curves for p in c.points]
    xt, yt = plot.scale(xs, ys, x_count=8)
    plot.frame(
        xt,
        yt,
        dash.fmt_compact,
        lambda v: f"{v:+.0f}",
        x_title="cumulative decisions",
        y_title="Elo above its own random baseline",
    )
    entries = []
    for curve in curves:
        pts = [(p[1], p[2]) for p in curve.points]
        plot.line(pts, curve.color,
                  width=2.2 if not curve.ref.context else 1.5,
                  opacity=1.0 if not curve.ref.context else 0.7,
                  dash="4 4" if curve.ref.context else None)
        plot.errbars([(p[1], p[2], p[3]) for p in curve.points], curve.color)
        plot.register(
            curve.ref.label,
            curve.color,
            [
                (p[1], p[2],
                 f"{dash.fmt_compact(p[1])} decisions ({dash.fmt_compact(p[0])} games)",
                 f"{p[2]:+.0f} Elo")
                for p in curve.points
            ],
        )
        if not curve.ref.context:
            x_last, y_last = pts[-1]
            # a run that ends in the left sliver of the axis labels rightward
            left = (x_last - plot.xlo) < 0.2 * (plot.xhi - plot.xlo)
            plot.label(x_last, y_last, f"{curve.ref.game} {y_last:+.0f}",
                       anchor="start" if left else "end",
                       dx=6 if left else 0, dy=-10)
        entries.append((curve.ref.label, curve.color,
                        "dash" if curve.ref.context else "line"))
    svg = plot.svg("All games, absolute Elo, one axis")
    return dash.figure(
        "How much skill does each game hold?",
        "Every run on one axis, each anchored to its own ladder (random = 0). "
        "A curve that tops out lower offers less distinguishable skill above "
        "random - the depth of the game, in one picture. Caveat: rating units "
        "differ per game (Uno is rated over matches, which stretch a per-hand "
        "edge), so read heights as indicative, shapes as exact.",
        svg,
        legend_html=dash.legend(entries),
        wide=True,
    )


def shape_panel(curves: list[Curve]) -> str:
    plot = dash.Plot(width=1080, height=360)
    xt, yt = plot.scale([0.0, 1.0], [0.0, 1.05], x_count=10)
    plot.frame(
        xt,
        yt,
        lambda v: f"{v * 100:.0f}%",
        lambda v: f"{v:g}",
        x_title="fraction of the run's budget (decisions)",
        y_title="fraction of own Elo range",
    )
    entries = []
    for curve in curves:
        pts = normalised(curve)
        if not pts:
            continue
        plot.line(pts, curve.color, width=2.0 if not curve.ref.context else 1.5,
                  opacity=1.0 if not curve.ref.context else 0.75,
                  dash="4 4" if curve.ref.context else None)
        plot.register(
            curve.ref.label,
            curve.color,
            [(x, y, f"{x * 100:.0f}% of budget", f"{y:.2f} of range") for x, y in pts],
        )
        half = slope_half_point(curve)
        if half is not None and not curve.ref.context:
            y_half = min((abs(x - half), y) for x, y in pts)[1]
            plot.dots([(half, y_half)], curve.color, radius=4.5)
            plot.label(half, y_half, f"slope ½ at {half * 100:.0f}%", dy=-14)
        entries.append((curve.ref.label + (" — context" if curve.ref.context else ""), curve.color, "dash" if curve.ref.context else "line"))
    svg = plot.svg("Normalised learning-curve shapes")
    return dash.figure(
        "The shape of learning, game against game",
        "Each curve normalised to its own run: 0 = its games-0 rating, 1 = its best "
        "checkpoint. Elo numbers are never compared across games — only this shape is. "
        "The marked point is where the smoothed slope first stays below half its "
        "early value.",
        svg,
        legend_html=dash.legend(entries),
        wide=True,
    )


def elo_panel(game: str, curves: list[Curve]) -> str:
    mine = [c for c in curves if c.ref.game == game]
    if not mine:
        return ""
    plot = dash.Plot(width=520, height=300)
    xs = [p[1] for c in mine for p in c.points]
    ys = [p[2] for c in mine for p in c.points]
    for _, elo in BASELINE_ELOS.get(game, ()):  # keep reference lines in frame
        ys.append(elo)
    xt, yt = plot.scale(xs, ys)
    plot.frame(
        xt,
        yt,
        dash.fmt_compact,
        lambda v: f"{v:+.0f}",
        x_title="cumulative decisions",
        y_title="Elo (this game's ladder)",
    )
    for name, elo in BASELINE_ELOS.get(game, ()):
        plot.hline(elo, f"{name} {elo:+.0f}")
    entries = []
    for curve in mine:
        pts = [(p[1], p[2]) for p in curve.points]
        plot.line(pts, curve.color,
                  width=2.0 if not curve.ref.context else 1.5,
                  opacity=1.0 if not curve.ref.context else 0.75,
                  dash="4 4" if curve.ref.context else None)
        plot.errbars([(p[1], p[2], p[3]) for p in curve.points], curve.color)
        plot.register(
            curve.ref.label,
            curve.color,
            [
                (p[1], p[2], f"{dash.fmt_compact(p[1])} decisions ({dash.fmt_compact(p[0])} games)",
                 f"{p[2]:+.0f} Elo")
                for p in curve.points
            ],
        )
        entries.append((curve.ref.label, curve.color, "dash" if curve.ref.context else "line"))
    live = any(c.state == "running" for c in mine)
    svg = plot.svg(f"{game} Elo curve")
    return dash.figure(
        f"{game} — Elo against decisions",
        "Anchored to this game's own baseline pool (random = 0)."
        + (" A run is still training - the curve is not final." if live else ""),
        svg,
        legend_html=dash.legend(entries),
    )


def optimal_panel(curves: list[Curve]) -> str:
    """% optimal for solved games, if any run has an optimal.jsonl yet."""
    series = []
    for curve in curves:
        rows, _ = dash.read_jsonl(REPO / "runs" / curve.ref.name / "optimal.jsonl")
        pts = []
        for row in rows:
            games = dash.num(row, "games")
            pct = dash.num(row, "pct_optimal")
            if games is None or pct is None:
                continue
            pts.append((games * curve.ref.decisions_per_game, pct * 100.0, games))
        if pts:
            series.append((curve, pts))
    if not series:
        return ""
    plot = dash.Plot(width=1080, height=320)
    xs = [x for _, pts in series for x, _, _ in pts]
    ys = [y for _, pts in series for _, y, _ in pts] + [100.0]
    xt, yt = plot.scale(xs, ys)
    plot.frame(xt, yt, dash.fmt_compact, lambda v: f"{v:.0f}%",
               x_title="cumulative decisions", y_title="% optimal moves")
    plot.hline(100.0, "perfect play")
    entries = []
    for curve, pts in series:
        plot.line([(x, y) for x, y, _ in pts], curve.color)
        plot.register(curve.ref.label, curve.color,
                      [(x, y, f"{dash.fmt_compact(g)} games", f"{y:.1f}% optimal")
                       for x, y, g in pts])
        entries.append((curve.ref.label, curve.color, "line"))
    svg = plot.svg("% optimal play on the solved suites")
    return dash.figure(
        "Solved games — % of moves that preserve the exact value",
        "The headline metric for tic-tac-toe and Connect Four: measured on the fixed "
        "solved suites (data/solved/), with an exact ceiling at 100%.",
        svg,
        legend_html=dash.legend(entries),
        wide=True,
    )


def stats_table(curves: list[Curve]) -> str:
    headers = [
        "run", "game", "Elo start → best", "games", "decisions",
        "slope ½ at", "R² 25% / 50% / full", "R² log fit",
    ]
    rows = []
    for curve in curves:
        ys = smooth([p[2] for p in curve.points])
        half = slope_half_point(curve)
        r2 = truncation_r2(curve)

        def r2f(v):
            return f"{v:.3f}" if v is not None else "—"

        rows.append([
            curve.ref.label,
            curve.ref.game,
            f"{ys[0]:+.0f} → {max(ys):+.0f}",
            dash.fmt_compact(curve.points[-1][0]),
            dash.fmt_compact(curve.points[-1][1]),
            f"{half * 100:.0f}% of budget" if half is not None else "not reached",
            f"{r2f(r2['r2_25'])} / {r2f(r2['r2_50'])} / {r2f(r2['r2_full'])}",
            r2f(r2["r2_log"]),
        ])
    note = (
        "R² of a straight-line fit over the first 25%, 50% and 100% of each run "
        "— shown three ways because a high value mostly measures truncation, "
        "not design (every curve is straight at the start). The log column fits "
        "Elo against log(decisions) - fast early, slow late; where it beats the "
        "straight line, the curve is closer to a log than a line. The informative "
        "column is still where the slope dies."
    )
    return (
        '<section class="panel"><h2>The numbers behind the shapes</h2>'
        + dash.table(headers, rows)
        + f'<p class="fig-sub" style="margin-top:10px">{dash.esc(note)}</p></section>'
    )


# --------------------------------------------------------------------- page
def build_page(now: float) -> str:
    curves = load_curves()
    generated = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")  # noqa: DTZ006
    body = [
        '<div class="wrap">',
        '<header class="masthead"><div class="brand">',
        "<h1>Ludometer · cross-game comparison</h1>",
        '<p class="thesis">What does each game’s learning curve look like — '
        "and where does the slope die?</p>",
        '<a class="cta" href="dashboard.html">Live training monitor →</a></div>',
        f'<div class="masthead-meta"><span>generated {dash.esc(generated)}</span>'
        f"<span>{len(curves)} runs</span><span>auto-refresh 60s</span></div>",
        "</header>",
    ]
    if not curves:
        body.append('<p class="empty">No comparable runs found under runs/.</p>')
    else:
        body.append(f'<section class="panel"><div class="grid">{combined_panel(curves)}</div></section>')
        body.append(f'<section class="panel"><div class="grid">{shape_panel(curves)}</div></section>')
        panels = [elo_panel(g, curves) for g in ("azul", "uno", "unoplus", "tictactoe", "connect4")]
        panels = [p for p in panels if p]
        body.append(f'<section class="panel"><h2>Absolute curves, one scale per game</h2><div class="grid">{"".join(panels)}</div></section>')
        optimal = optimal_panel(curves)
        if optimal:
            body.append(f'<section class="panel"><div class="grid">{optimal}</div></section>')
        body.append(stats_table(curves))
    body.append(
        '<footer class="foot"><span>ludometer · generated by web/make_compare.py</span>'
        "<span>static page, no external requests</span>"
        f"<span>{dash.esc(generated)}</span></footer></div>"
    )
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8" />\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        '<meta http-equiv="refresh" content="60" />\n'
        "<title>Ludometer Cross-Game Comparison</title>\n"
        f"<style>{dash.styles()}</style>\n"
        "</head><body>\n" + "".join(body) + f"\n<script>{dash.JS}</script>\n</body></html>\n"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO / "web" / "compare.html"))
    parser.add_argument("--watch", type=float, default=0.0)
    args = parser.parse_args(argv)
    out = Path(args.out)
    while True:
        size = dash.write_atomic(out, build_page(time.time()))
        print(f"wrote {out} ({size:,} bytes)")
        if not args.watch:
            return 0
        time.sleep(max(args.watch, 5.0))


if __name__ == "__main__":
    raise SystemExit(main())
