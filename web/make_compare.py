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
    "tictactoe": dash.SERIES[6],  # violet - far from Uno's orange (Remi)
    "connect4": dash.SERIES[4],  # magenta
    "lostcities": dash.SERIES[5],  # green
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
    RunRef("uno1", "uno", "Uno", 18.8),
    RunRef("unoplus1", "unoplus", "Uno+ (house rules)", 40.0),
    RunRef("ttt1", "tictactoe", "Tic-tac-toe", 5.5),
    RunRef("c4_1", "connect4", "Connect Four", 30.0),
    RunRef("lc1", "lostcities", "Lost Cities", 220.0),
]

#: pre-rated baseline reference lines per game (from the journal's gauntlets)
BASELINE_ELOS = {
    "uno": (("uno:greedy", 572.0), ("uno:heuristic", 396.0)),
    # measured by gauntlet (200 games/pairing, random=0): perfect play is
    # ratable because the trained net draws it - +527 +/- 25, and the net at
    # 256 sims sits at +519: the ceiling is reached, not just visible.
    "tictactoe": (("perfect play", 526.6),),
}


GAME_NOTES = {
    "azul": {
        "tags": "perfect information · no luck after the draft · unsolved",
        "html": "Azul (Kiesling 2017): the calibrated deep reference.",
    },
    "uno": {
        "tags": "hidden hands · high luck · unsolved",
        "html": "Uno, official 2-player rules (draw only when stuck, no stacking). "
                "First to 500 points wins.",
    },
    "unoplus": {
        "tags": "hidden hands · high luck · unsolved",
        "html": "Uno+ = Uno plus four house rules. First to 500 points wins."
                "<ul style='margin:6px 0 6px 18px;padding:0'>"
                "<li>9-card deal (instead of 7).</li>"
                "<li>Drawing a card is always a legal move (up to a 15-card hand cap).</li>"
                "<li>+2 and +4 stack, like on like: instead of drawing, you may answer "
                "a +2 with your own +2 (or a +4 with a +4), passing the grown penalty "
                "back - whoever runs out of answers draws the whole stack.</li>"
                "<li>Playing a 7 swaps hands (both players then know the opposing "
                "hand exactly).</li>"
                "</ul>"
                "Note: with the default Uno rules ~64% of turns are forced; with the "
                "Uno+ rules that drops to ~19%.",
    },
    "tictactoe": {
        "tags": "perfect information · no luck · solved (draw)",
        "html": "Tic-tac-toe: a solved game used as calibration. The agent reaches "
                "perfect play very rapidly, as expected.",
    },
    "connect4": {
        "tags": "perfect information · no luck · solved (first player wins)",
        "html": "Connect Four: tic-tac-toe's character, but a much deeper game.",
    },
    "lostcities": {
        "tags": "hidden hand · high luck · unsolved · Knizia 1999",
        "html": "Lost Cities: Uno's dials on a great game's design - the control case. "
                "The rules in four lines: on your turn play one card onto one of your "
                "five colour expeditions (ascending only) or discard it, then draw from "
                "the deck or a discard pile. Starting an expedition costs 20 points; its "
                "cards earn them back. Handshake cards played first multiply the colour's "
                "result. The game ends when the deck runs out; higher total wins.",
    },
}


@dataclass
class Curve:
    ref: RunRef
    points: list  # (games, decisions, elo, err)
    state: str

    @property
    def color(self) -> str:
        return CONTEXT if self.ref.context else GAME_COLOR[self.ref.game]


def _rerate_points(run_dir: Path, rows: list) -> list | None:
    """A self-consistent post-hoc ladder, if the run has one (rerate.json).

    The trainer's own Elo pass can saturate: when a net beats every anchor
    (c4_1 hits 1.00 vs random by game 4096) the fit diverges and each frozen
    checkpoint carries the inflated number forward. A round-robin where the
    checkpoints bound each other (ludometer.eval.gauntlet --json) replaces it.
    Labels are g<games>; decisions come from the run's own elo.jsonl mapping.
    """
    data = dash.read_json(run_dir / "rerate.json")
    ratings = data.get("elo") or {}
    errors = data.get("elo_err") or {}
    dec_by_games = {}
    t_by_games = {}
    for row in rows:
        g, d = dash.num(row, "games"), dash.num(row, "decisions")
        if g is not None and d is not None:
            dec_by_games[g] = d
        if g is not None and dash.num(row, "t") is not None:
            t_by_games[g] = dash.num(row, "t")
    points = []
    for label, elo in ratings.items():
        if not label.startswith("g") or not label[1:].isdigit():
            continue
        games = int(label[1:])
        decisions = dec_by_games.get(games, games * 1.0)
        points.append(
            (
                games,
                decisions,
                float(elo),
                float(errors.get(label, 0.0)),
                t_by_games.get(games, 0.0),
            )
        )
    points.sort()
    return points if len(points) >= 3 else None


def load_curves() -> list[Curve]:
    curves = []
    for ref in RUN_REFS:
        run_dir = REPO / "runs" / ref.name
        rows, _ = dash.read_jsonl(run_dir / "elo.jsonl")
        points = _rerate_points(run_dir, rows)
        if points is None:
            points = []
            for row in rows:
                games = dash.num(row, "games")
                elo = dash.num(row, "elo")
                if games is None or elo is None:
                    continue
                decisions = dash.num(row, "decisions")
                if decisions is None:
                    decisions = games * ref.decisions_per_game
                points.append((games, decisions, elo, dash.num(row, "elo_err") or 0.0,
                               dash.num(row, "t") or 0.0))
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
def _draw_order(curves: list[Curve], key) -> list[Curve]:
    """Longest series first, shortest last — so the little curves (tic-tac-toe's
    sliver) are painted ON TOP of the big ones instead of vanishing under them."""
    return sorted(curves, key=lambda c: -key(c))


def _vline(plot: "dash.Plot", x_value: float, label: str) -> None:
    if not plot.xlo <= x_value <= plot.xhi:
        return
    x = plot.X(x_value)
    plot.back.append(
        f'<line x1="{dash.coord(x)}" y1="{dash.coord(plot.y0)}" '
        f'x2="{dash.coord(x)}" y2="{dash.coord(plot.y0 + plot.ph)}" class="ref" />'
    )
    plot.front.append(
        f'<text x="{dash.coord(x + 5)}" y="{dash.coord(plot.y0 + 12)}" '
        f'class="ref-label" text-anchor="start">{dash.esc(label)}</text>'
    )


# Boiling 1 L of water (20 -> 100 °C) is 4.186 kJ/kg·K x 80 K = 335 kJ = 93 Wh
# of heat; an electric stove delivers ~70% of what it draws, so the wall pays
# ~133 Wh. At the laptop's ~45 W that is ~3 hours of training.
KETTLE_WH = 93.0 / 0.70
LAPTOP_W = 45.0
_ANNO = dash.INK_2  # annotation ink: brighter than the grid, quieter than data


def _pot_icon(x: float, y: float) -> str:
    """A ~30px pot of water steaming on a stove, drawn in hairline strokes."""
    return (
        f'<g transform="translate({dash.coord(x)} {dash.coord(y)})" '
        f'stroke="{_ANNO}" fill="none" stroke-width="1.4" stroke-linecap="round">'
        '<path d="M4 8 q2 -4 0 -7" /><path d="M11 8 q2 -4 0 -7" />'
        '<path d="M18 8 q2 -4 0 -7" />'
        '<rect x="0" y="11" width="22" height="12" rx="1.5" />'
        '<line x1="-5" y1="14" x2="0" y2="14" /><line x1="22" y1="14" x2="27" y2="14" />'
        '<line x1="-3" y1="27" x2="25" y2="27" />'
        '<line x1="1" y1="30" x2="5" y2="30" /><line x1="9" y1="30" x2="13" y2="30" />'
        '<line x1="17" y1="30" x2="21" y2="30" />'
        "</g>"
    )


def _bulb_icon(x: float, y: float) -> str:
    """A ~22px light bulb with rays."""
    return (
        f'<g transform="translate({dash.coord(x)} {dash.coord(y)})" '
        f'stroke="{_ANNO}" fill="none" stroke-width="1.4" stroke-linecap="round">'
        '<circle cx="8" cy="8" r="6.5" />'
        '<path d="M5.5 14 v3 h5 v-3" /><line x1="6" y1="19.5" x2="10" y2="19.5" />'
        '<line x1="8" y1="-3" x2="8" y2="-6" /><line x1="16" y1="0" x2="18.5" y2="-2.5" />'
        '<line x1="0" y1="0" x2="-2.5" y2="-2.5" /><line x1="17" y1="8" x2="20.5" y2="8" />'
        '<line x1="-1" y1="8" x2="-4.5" y2="8" />'
        "</g>"
    )


def _arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    """A curved annotation arrow from (x1,y1) to (x2,y2), head at the target."""
    mx, my = (x1 + x2) / 2, min(y1, y2) - 14
    import math as _m

    angle = _m.atan2(y2 - my, x2 - mx)
    a1 = angle + _m.radians(155)
    a2 = angle - _m.radians(155)
    head = (
        f"M {dash.coord(x2)} {dash.coord(y2)} "
        f"L {dash.coord(x2 + 7 * _m.cos(a1))} {dash.coord(y2 + 7 * _m.sin(a1))} "
        f"M {dash.coord(x2)} {dash.coord(y2)} "
        f"L {dash.coord(x2 + 7 * _m.cos(a2))} {dash.coord(y2 + 7 * _m.sin(a2))}"
    )
    return (
        f'<path d="M {dash.coord(x1)} {dash.coord(y1)} Q {dash.coord(mx)} {dash.coord(my)} '
        f'{dash.coord(x2)} {dash.coord(y2)}" stroke="{_ANNO}" fill="none" stroke-width="1.3" />'
        f'<path d="{head}" stroke="{_ANNO}" fill="none" stroke-width="1.3" stroke-linecap="round" />'
    )


def _anno_text(x: float, y: float, lines: list[str], anchor: str = "start") -> str:
    spans = "".join(
        f'<text x="{dash.coord(x)}" y="{dash.coord(y + i * 13)}" class="ref-label" '
        f'text-anchor="{anchor}">{dash.esc(line)}</text>'
        for i, line in enumerate(lines)
    )
    return spans


def combined_panel(curves: list[Curve], smoothed: bool = False) -> str:
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
    for curve in _draw_order(curves, lambda c: c.points[-1][1]):
        ys_line = [p[2] for p in curve.points]
        if smoothed:
            ys_line = smooth(ys_line)
        pts = [(p[1], y) for p, y in zip(curve.points, ys_line)]
        plot.line(pts, curve.color,
                  width=2.2 if not curve.ref.context else 1.5,
                  opacity=1.0 if not curve.ref.context else 0.7,
                  dash="4 4" if curve.ref.context else None)
        if not smoothed:
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
    if smoothed:
        return dash.figure(
            "The same chart, smoothed",
            "A 5-point moving average over each curve, error bars dropped - "
            "the shapes without the eval noise. Read trends here, exact "
            "values on the raw chart above.",
            svg,
            legend_html=dash.legend(entries),
            wide=True,
        )
    return dash.figure(
        "How much skill does each game hold?",
        "Every run on one axis, each anchored to its own ladder (random = 0). "
        "A curve that tops out lower offers less distinguishable skill above "
        "random - the depth of the game, in one picture. Curves do not start at "
        "zero because the first checkpoint is the UNTRAINED net driven by the "
        "search, which already beats random play. Caveat: rating units differ "
        "per game (Uno is rated over matches, which stretch a per-hand edge), "
        "so read heights as indicative, shapes as exact.",
        svg,
        legend_html=dash.legend(entries),
        wide=True,
    )


def compute_panel(curves: list[Curve]) -> str:
    """Elo against wall-clock training time - Remi's "computational learning
    effort" axis. Every run trained on the same M-series laptop, so hours are
    literal compute (and energy: ~45 W under load, so 1 h is roughly 45 Wh).
    A decision is not the same amount of thinking in every game; an hour is.
    """
    plot = dash.Plot(width=1080, height=380)
    CUT_H = 4.2  # Remi: Uno+'s long flat tail adds nothing - trim like Azul's
    xs, ys = [], []
    for c in curves:
        for p in c.points:
            if p[4] and p[4] / 3600.0 <= CUT_H:
                xs.append(p[4] / 3600.0)
                ys.append(p[2])
    if not xs:
        return ""
    xt, yt = plot.scale(xs, ys, x_count=8)
    plot.frame(
        xt,
        yt,
        lambda v: f"{v:g}h",
        lambda v: f"{v:+.0f}",
        x_title="wall-clock training time (same laptop; ~45 Wh per hour)",
        y_title="Elo above its own random baseline",
    )
    kettle_h = KETTLE_WH / LAPTOP_W
    _vline(plot, kettle_h, "")
    if plot.xlo <= kettle_h <= plot.xhi:
        kx = plot.X(kettle_h)
        py = plot.y0 + plot.ph - 150
        plot.front.append(_pot_icon(kx + 52, py))
        plot.front.append(_arrow(kx + 48, py + 20, kx + 4, py + 4))
        plot.front.append(
            _anno_text(
                kx + 34,
                py + 48,
                [
                    "energy needed to boil one litre",
                    f"of water on the stove — {KETTLE_WH:.0f} Wh",
                ],
            )
        )
    entries = []
    for curve in _draw_order(curves, lambda c: c.points[-1][4]):
        pts = [
            (p[4] / 3600.0, p[2])
            for p in curve.points
            if p[4] and p[4] / 3600.0 <= CUT_H
        ]
        clipped = sum(1 for p in curve.points if p[4] / 3600.0 > CUT_H)
        if len(pts) < 2:
            continue
        if clipped and not curve.ref.context:
            end_t = curve.points[-1][4] / 3600.0
            plot.label(pts[-1][0], pts[-1][1],
                       f"continues flat to {end_t:.1f}h",
                       anchor="end", dy=-10)
        plot.line(pts, curve.color,
                  width=2.0 if not curve.ref.context else 1.5,
                  opacity=1.0 if not curve.ref.context else 0.7,
                  dash="4 4" if curve.ref.context else None)
        plot.register(
            curve.ref.label,
            curve.color,
            [(x, y, f"{x:.2f} h (~{x * 45:.0f} Wh)", f"{y:+.0f} Elo") for x, y in pts],
        )
        entries.append((curve.ref.label, curve.color,
                        "dash" if curve.ref.context else "line"))
    bulb = ""
    ttt = next((c for c in curves if c.ref.game == "tictactoe"), None)
    if ttt and ttt.points[-1][4]:
        t_end = ttt.points[-1][4] / 3600.0
        elo_end = ttt.points[-1][2]
        wh = t_end * LAPTOP_W
        minutes = wh / 10 * 60
        bx = plot.X(t_end) + 96
        by = plot.Y(elo_end) + 34
        plot.front.append(_bulb_icon(bx, by))
        plot.front.append(_arrow(bx - 4, by + 4, plot.X(t_end) + 4, plot.Y(elo_end) + 8))
        plot.front.append(
            _anno_text(
                bx + 30,
                by + 4,
                [
                    "tic-tac-toe is solved for",
                    f"~{minutes:.0f} minutes of a 10 W LED",
                ],
            )
        )
        bulb = (
            f" Tic-tac-toe's entire run cost ~{wh:.1f} Wh: a 10 W LED bulb "
            f"left on for {minutes:.0f} minutes solves the game."
        )
    svg = plot.svg("Elo against training hours")
    return dash.figure(
        "The same chart in computational effort",
        "x is wall-clock training time on the one laptop every run used - "
        "literal compute spent, including its evaluation pauses. A 'decision' "
        "is cheaper to think about in some games than others; an hour is an "
        "hour. The reference line is the energy to boil one litre of water on "
        "an electric stove (93 Wh of heat at ~70% efficiency): learning Azul "
        "to +2000 costs about one pot of tea, Uno+ about two." + bulb,
        svg,
        legend_html=dash.legend(entries),
        wide=True,
    )


def notes_section() -> str:
    items = []
    for g, note in GAME_NOTES.items():
        items.append(
            f'<div style="margin:10px 0"><p class="fig-sub" style="margin:0">'
            f'<strong style="color:{GAME_COLOR[g]}">{g}</strong>'
            f'<span style="opacity:.75">&nbsp; {dash.esc(note["tags"])}</span></p>'
            f'<div class="fig-sub" style="margin:2px 0 0">{note["html"]}</div></div>'
        )
    return (
        '<section class="panel"><h2>The games, and what was changed</h2>'
        + "".join(items)
        + "</section>"
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
    for c in mine:
        rerate = dash.read_json(REPO / "runs" / c.ref.name / "rerate.json")
        perfect = (rerate.get("elo") or {}).get("perfect")
        if perfect is not None:
            ys.append(float(perfect))
    xt, yt = plot.scale(xs, ys)
    plot.frame(
        xt,
        yt,
        dash.fmt_compact,
        lambda v: f"{v:+.0f}",
        x_title="cumulative decisions",
        y_title="Elo (this game's ladder)",
    )
    refs = list(BASELINE_ELOS.get(game, ()))
    for c in mine:
        rerate = dash.read_json(REPO / "runs" / c.ref.name / "rerate.json")
        perfect = (rerate.get("elo") or {}).get("perfect")
        if perfect is not None:
            refs = [(r for r in refs if r[0] != "perfect play")] and [
                r for r in refs if r[0] != "perfect play"
            ]
            refs.append(("perfect play (same fit)", float(perfect)))
    for name, elo in refs:
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
        body.append(f'<section class="panel"><div class="grid">{combined_panel(curves, smoothed=True)}</div></section>')
        body.append(notes_section())
        compute = compute_panel(curves)
        if compute:
            body.append(f'<section class="panel"><div class="grid">{compute}</div></section>')
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
