#!/usr/bin/env python3
"""Regenerate web/dashboard.html from everything under runs/.

Stdlib only (no numpy, no torch, no chart libraries): charts are inline SVG
emitted by this script. The output is a single self-contained file that works
from file:// and refreshes itself every 30 s, so a browser tab left open during
training keeps up as the trainer rewrites the logs in place.

Usage:
    python3 web/make_dashboard.py                # write once
    python3 web/make_dashboard.py --watch 20     # rewrite every 20 s, forever
    python3 web/make_dashboard.py --runs /tmp/r --out /tmp/d.html

Reads (all optional, all tolerant of truncated last lines — the trainer may be
mid-write): runs/<run>/config.json, status.json, train.jsonl, elo.jsonl, plus
NOTES_FOR_REMI.md for the journal. Schemas are defined in docs/DESIGN.md.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Design tokens. Single committed dark look: a slate-teal ground borrowed from
# Azul's teal tile, chart series from the validated dark categorical palette
# (checked with the dataviz validator against surface #151c20 — all six checks
# pass), status colours reserved for run state only.
# --------------------------------------------------------------------------
PLANE = "#0b1013"
SURFACE = "#151c20"
SURFACE_2 = "#1b2429"
INK = "#eaf0f1"
INK_2 = "#9dafb5"
MUTED = "#6d8188"
GRID = "#1f2a2f"
AXIS = "#36484f"
HAIRLINE = "rgba(234,240,241,0.10)"

SERIES = [
    "#3987e5",  # 1 blue
    "#d95926",  # 2 orange
    "#199e70",  # 3 aqua
    "#c98500",  # 4 yellow
    "#d55181",  # 5 magenta
    "#008300",  # 6 green
    "#9085e9",  # 7 violet
    "#e66767",  # 8 red
]
ST_GOOD = "#0ca30c"
ST_WARN = "#fab219"
ST_CRIT = "#d03b3b"

STALE_AFTER = 180.0  # seconds; docs call a heartbeat older than 3 min stale
MAX_SERIES = 8  # never cycle categorical hues; the tail folds away

# ==========================================================================
# Loading
# ==========================================================================


def read_json(path: Path) -> dict:
    """Parse a JSON file, returning {} if it is missing, empty or half-written."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> tuple[list[dict], int]:
    """Parse a JSONL file line by line. Returns (records, dropped_line_count).

    A trainer appending to this file can leave the last line truncated; bad
    lines are counted and skipped rather than raising.
    """
    records: list[dict] = []
    dropped = 0
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return records, dropped
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except (ValueError, RecursionError):
            dropped += 1
            continue
        if isinstance(value, dict):
            records.append(value)
        else:
            dropped += 1
    return records, dropped


def num(record: dict, key: str):
    """Fetch a finite float from a record, or None."""
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


class Run:
    """Everything the dashboard knows about one run directory."""

    def __init__(self, directory: Path, now: float):
        self.dir = directory
        self.name = directory.name
        self.config = read_json(directory / "config.json")
        self.status = read_json(directory / "status.json")
        self.train, self.train_dropped = read_jsonl(directory / "train.jsonl")
        self.elo, self.elo_dropped = read_jsonl(directory / "elo.jsonl")

        self.run_name = str(self.status.get("run") or self.config.get("run") or self.name)
        self.state = str(self.status.get("state") or "unknown").lower()
        if self.state not in ("running", "done", "failed"):
            self.state = "unknown"
        self.note = self.status.get("note")

        self.started = parse_iso(self.status.get("started") or self.config.get("started"))
        self.updated = parse_iso(self.status.get("updated"))
        self.age = None if self.updated is None else max(0.0, now - self.updated)
        self.stale = self.state == "running" and (self.age is None or self.age > STALE_AFTER)

        last_train = self.train[-1] if self.train else {}
        last_elo = self.elo[-1] if self.elo else {}
        self.games = first_num(self.status.get("games"), num(last_train, "games"), num(last_elo, "games"))
        self.steps = first_num(self.status.get("steps"), num(last_train, "steps"))
        self.elapsed = first_num(
            None if (self.started is None or self.updated is None) else self.updated - self.started,
            num(last_train, "t"),
            num(last_elo, "t"),
        )

        self.elo_points = [
            (num(r, "games"), num(r, "elo"), num(r, "elo_err") or 0.0, str(r.get("ckpt") or ""), r)
            for r in self.elo
        ]
        self.elo_points = [p for p in self.elo_points if p[0] is not None and p[1] is not None]
        self.elo_points.sort(key=lambda p: p[0])
        self.fit = linfit([p[0] for p in self.elo_points], [p[1] for p in self.elo_points])

    @property
    def latest_elo(self):
        return self.elo_points[-1][1] if self.elo_points else None

    @property
    def latest_elo_err(self):
        return self.elo_points[-1][2] if self.elo_points else None

    @property
    def has_data(self) -> bool:
        return bool(self.train or self.elo or self.status)

    @property
    def sort_key(self):
        rank = {"running": 0, "unknown": 1, "failed": 2, "done": 3}.get(self.state, 4)
        return (rank, -(self.started or 0.0), self.name)


def first_num(*candidates):
    for value in candidates:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def discover_runs(runs_dir: Path, now: float) -> list[Run]:
    if not runs_dir.is_dir():
        return []
    runs = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        run = Run(child, now)
        if run.has_data:
            runs.append(run)
    runs.sort(key=lambda r: r.sort_key)
    return runs


# ==========================================================================
# Time, numbers, statistics
# ==========================================================================


def parse_iso(value):
    """ISO-8601 -> epoch seconds. Naive timestamps are read as local time."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                stamp = datetime.strptime(text, fmt)  # noqa: DTZ007 - naive input is localized just below
                break
            except ValueError:
                continue
        else:
            return None
    if stamp.tzinfo is None:
        stamp = stamp.astimezone()
    return stamp.timestamp()


def fmt_int(value) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{round(float(value)):,}"


def fmt_compact(value) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    value = float(value)
    if abs(value) < 1e-9:
        return "0"
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= limit:
            scaled = value / limit
            digits = 0 if abs(scaled) >= 100 else 1
            return f"{scaled:.{digits}f}{suffix}"
    return f"{value:,.0f}"


def fmt_float(value, digits=2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    value = float(value)
    if abs(value) < 0.5 * 10.0 ** (-digits):
        value = 0.0  # never print "-0"
    return f"{value:,.{digits}f}"


def fmt_duration(seconds) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or seconds < 0:
        return "—"
    seconds = round(float(seconds))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def fmt_ago(seconds) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "no heartbeat"
    return f"updated {fmt_duration(seconds)} ago"


def linfit(xs: list[float], ys: list[float]):
    """Least-squares fit. Returns (slope, intercept, r2) or None if undefined."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    syy = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 if syy <= 0 else 1.0 - ss_res / syy
    if not all(math.isfinite(v) for v in (slope, intercept, r2)):
        return None
    return slope, intercept, r2


def nice_ticks(lo: float, hi: float, count: int = 5):
    """Round tick values covering [lo, hi]. Returns (ticks, axis_lo, axis_hi)."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        lo, hi = 0.0, 1.0
    if hi < lo:
        lo, hi = hi, lo
    if hi - lo < 1e-12:
        pad = max(abs(hi) * 0.1, 1.0)
        lo, hi = lo - pad, hi + pad
    raw = (hi - lo) / max(1, count)
    mag = 10.0 ** math.floor(math.log10(raw)) if raw > 0 else 1.0
    step = mag
    for multiple in (1, 2, 2.5, 5, 10):
        step = multiple * mag
        if raw <= step:
            break
    start = math.floor(lo / step) * step
    end = math.ceil(hi / step) * step
    ticks = []
    value = start
    while value <= end + step * 1e-9 and len(ticks) < 40:
        ticks.append(round(value, 10))
        value += step
    if len(ticks) < 2:
        ticks = [lo, hi]
    return ticks, ticks[0], ticks[-1]


def thin(records: list, limit: int = 400) -> list:
    """Uniformly subsample a long series, always keeping the last point."""
    if len(records) <= limit:
        return records
    stride = math.ceil(len(records) / limit)
    kept = records[::stride]
    if kept[-1] is not records[-1]:
        kept.append(records[-1])
    return kept


# ==========================================================================
# SVG plotting
# ==========================================================================


def coord(value) -> str:
    """Emit an SVG coordinate. Non-finite values can never reach the markup."""
    value = float(value)
    if not math.isfinite(value):
        return "0"
    return f"{value:.2f}".rstrip("0").rstrip(".") or "0"


def esc(text) -> str:
    return html.escape("" if text is None else str(text), quote=True)


class Plot:
    """A cartesian inline-SVG plot: hairline grid, thin marks, hover payload."""

    # Default geometry matches the *rendered* pixel width of a half-width card
    # (~520px at the 1180px page width), so SVG text lands at its nominal size
    # instead of being scaled up or down by the viewBox. Wide cards pass 1080.
    def __init__(self, width=520, height=290, left=54, right=18, top=18, bottom=40):
        self.width = width
        self.height = height
        self.x0 = left
        self.y0 = top
        self.pw = width - left - right
        self.ph = height - top - bottom
        self.back: list[str] = []
        self.marks: list[str] = []
        self.front: list[str] = []
        self.hover: list[dict] = []
        self.xlo, self.xhi = 0.0, 1.0
        self.ylo, self.yhi = 0.0, 1.0

    # -- scales ---------------------------------------------------------
    def scale(self, xs, ys, x_count=6, y_count=5, y_pad=0.08, y_zero=False):
        """Fit the axes to the data, then place round ticks *inside* that range.

        Snapping the axis out to whole tick boundaries is what leaves a chart
        with a quarter of empty plot area (an Elo axis running to -250), so the
        bounds follow the data and frame() simply drops out-of-range ticks.
        """
        xlo, xhi = float(min(xs)), float(max(xs))
        if xhi - xlo < 1e-12:
            # A single x value: one tick at that value beats a fake range whose
            # rounded labels repeat ("-1, -0, 0, 0, 1").
            only = xhi
            pad = max(abs(only) * 0.05, 1.0)
            self.xlo, self.xhi = only - pad, only + pad
            xt = [only]
        else:
            self.xlo, self.xhi = xlo, xhi
            xt, _, _ = nice_ticks(xlo, xhi, x_count)

        ylo, yhi = float(min(ys)), float(max(ys))
        span = yhi - ylo
        if span > 0:
            ylo -= span * y_pad
            yhi += span * y_pad
        elif span == 0:
            ylo, yhi = ylo - 1.0, yhi + 1.0
        if y_zero:
            ylo = min(ylo, 0.0)
        self.ylo, self.yhi = ylo, yhi
        yt, _, _ = nice_ticks(ylo, yhi, y_count)
        return xt, yt

    def X(self, value) -> float:
        span = self.xhi - self.xlo
        if span <= 0:
            return self.x0 + self.pw / 2
        return self.x0 + (float(value) - self.xlo) / span * self.pw

    def Y(self, value) -> float:
        span = self.yhi - self.ylo
        if span <= 0:
            return self.y0 + self.ph / 2
        return self.y0 + self.ph - (float(value) - self.ylo) / span * self.ph

    # -- chrome ---------------------------------------------------------
    def frame(self, xt, yt, x_fmt, y_fmt, x_title="", y_title=""):
        # A degenerate range (one distinct value) can round several ticks to the
        # same label; draw each label once.
        seen_y: set[str] = set()
        seen_x: set[str] = set()
        for value in yt:
            if value < self.ylo - 1e-9 or value > self.yhi + 1e-9:
                continue
            text = y_fmt(value)
            if text in seen_y:
                continue
            seen_y.add(text)
            y = self.Y(value)
            self.back.append(
                f'<line x1="{coord(self.x0)}" y1="{coord(y)}" x2="{coord(self.x0 + self.pw)}" '
                f'y2="{coord(y)}" class="grid" />'
            )
            self.back.append(
                f'<text x="{coord(self.x0 - 10)}" y="{coord(y + 3.5)}" class="tick tick-y">'
                f"{esc(text)}</text>"
            )
        for value in xt:
            if value < self.xlo - 1e-9 or value > self.xhi + 1e-9:
                continue
            text = x_fmt(value)
            if text in seen_x:
                continue
            seen_x.add(text)
            x = self.X(value)
            self.back.append(
                f'<text x="{coord(x)}" y="{coord(self.y0 + self.ph + 20)}" class="tick tick-x">'
                f"{esc(text)}</text>"
            )
        base = self.y0 + self.ph
        self.back.append(
            f'<line x1="{coord(self.x0)}" y1="{coord(base)}" x2="{coord(self.x0 + self.pw)}" '
            f'y2="{coord(base)}" class="axis" />'
        )
        if x_title:
            self.front.append(
                f'<text x="{coord(self.x0 + self.pw)}" y="{coord(self.height - 5)}" '
                f'class="axis-title" text-anchor="end">{esc(x_title)}</text>'
            )
        if y_title:
            # Centred on the axis, clear of the topmost tick label.
            self.front.append(
                f'<text transform="translate(12 {coord(self.y0 + self.ph / 2)}) rotate(-90)" '
                f'class="axis-title" text-anchor="middle">{esc(y_title)}</text>'
            )

    def hline(self, value, label=""):
        y = self.Y(value)
        self.back.append(
            f'<line x1="{coord(self.x0)}" y1="{coord(y)}" x2="{coord(self.x0 + self.pw)}" '
            f'y2="{coord(y)}" class="ref" />'
        )
        if label:
            # Left edge: direct end-labels live at the right.
            self.back.append(
                f'<text x="{coord(self.x0 + 4)}" y="{coord(y - 6)}" class="ref-label" '
                f'text-anchor="start">{esc(label)}</text>'
            )

    # -- marks ----------------------------------------------------------
    def line(self, points, color, dash=None, width=2.0, opacity=1.0):
        if len(points) < 2:
            return
        path = " ".join(
            ("M" if i == 0 else "L") + f"{coord(self.X(x))} {coord(self.Y(y))}"
            for i, (x, y) in enumerate(points)
        )
        attrs = f'stroke="{color}" stroke-width="{coord(width)}" opacity="{coord(opacity)}"'
        if dash:
            attrs += f' stroke-dasharray="{dash}"'
        self.marks.append(f'<path d="{path}" class="series-line" {attrs} />')

    def errbars(self, points, color):
        for x, y, err in points:
            if not err:
                continue
            self.marks.append(
                f'<line x1="{coord(self.X(x))}" y1="{coord(self.Y(y - err))}" '
                f'x2="{coord(self.X(x))}" y2="{coord(self.Y(y + err))}" '
                f'stroke="{color}" stroke-width="1" opacity="0.45" />'
            )

    def dots(self, points, color, radius=4.0):
        for x, y in points:
            self.marks.append(
                f'<circle cx="{coord(self.X(x))}" cy="{coord(self.Y(y))}" r="{coord(radius)}" '
                f'fill="{color}" stroke="{SURFACE}" stroke-width="2" />'
            )

    def label(self, x, y, text, anchor="middle", dy=-12, dx=0, cls="point-label"):
        self.front.append(
            f'<text x="{coord(self.X(x) + dx)}" y="{coord(self.Y(y) + dy)}" class="{cls}" '
            f'text-anchor="{anchor}">{esc(text)}</text>'
        )

    # -- hover ----------------------------------------------------------
    def register(self, name, color, points):
        """points: iterable of (x_value, y_value, x_label, y_label)."""
        payload = [
            [round(self.X(x), 1), round(self.Y(y), 1), xl, yl]
            for x, y, xl, yl in points
            if math.isfinite(float(x)) and math.isfinite(float(y))
        ]
        if payload:
            self.hover.append({"n": name, "c": color, "p": payload})

    # -- output ---------------------------------------------------------
    def svg(self, title) -> str:
        data = json.dumps(
            {
                "plot": [round(self.x0, 1), round(self.y0, 1), round(self.pw, 1), round(self.ph, 1)],
                "series": self.hover,
            },
            separators=(",", ":"),
        )
        body = "".join(self.back + self.marks + self.front)
        return (
            f'<svg class="chart" style="width:{coord(self.width)}px" '
            f'viewBox="0 0 {coord(self.width)} {coord(self.height)}" '
            f'preserveAspectRatio="xMidYMid meet" role="img" tabindex="0" '
            f'aria-label="{esc(title)}" data-chart="{esc(data)}">{body}'
            f'<g class="hoverlay" aria-hidden="true">'
            f'<line class="crosshair" x1="0" y1="{coord(self.y0)}" x2="0" '
            f'y2="{coord(self.y0 + self.ph)}" />'
            f'<circle class="hotdot" cx="-99" cy="-99" r="5.5" /></g></svg>'
        )


def legend(entries) -> str:
    """entries: (label, color, kind) with kind in {'line','dash','dot'}."""
    items = []
    for label, color, kind in entries:
        if kind == "dash":
            key = (
                f'<svg class="key" viewBox="0 0 18 8" aria-hidden="true"><line x1="0" y1="4" '
                f'x2="18" y2="4" stroke="{color}" stroke-width="2" stroke-dasharray="5 4" /></svg>'
            )
        elif kind == "dot":
            key = (
                f'<svg class="key" viewBox="0 0 18 8" aria-hidden="true">'
                f'<circle cx="9" cy="4" r="3.5" fill="{color}" /></svg>'
            )
        else:
            key = (
                f'<svg class="key" viewBox="0 0 18 8" aria-hidden="true"><line x1="0" y1="4" '
                f'x2="18" y2="4" stroke="{color}" stroke-width="2" /></svg>'
            )
        items.append(f'<span class="legend-item">{key}{esc(label)}</span>')
    return f'<div class="legend">{"".join(items)}</div>'


def table(headers, rows, cap=None) -> str:
    """The table-view twin every chart carries, folded away in a <details>."""
    shown = rows if cap is None or len(rows) <= cap else rows[-cap:]
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in shown)
    trimmed = (
        ""
        if len(shown) == len(rows)
        else f'<p class="table-note">Showing the last {len(shown)} of {len(rows)} rows.</p>'
    )
    return (
        '<details class="data-table"><summary>Data table</summary>'
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>{trimmed}</details>"
    )


def figure(title, subtitle, svg, legend_html="", readout="", table_html="", wide=False) -> str:
    classes = "figure" + (" figure-wide" if wide else "")
    parts = [f'<figure class="{classes}">', '<div class="fig-head"><div>']
    parts.append(f'<h3 class="fig-title">{esc(title)}</h3>')
    if subtitle:
        parts.append(f'<p class="fig-sub">{esc(subtitle)}</p>')
    parts.append("</div>")
    if readout:
        parts.append(f'<div class="readout">{readout}</div>')
    parts.append("</div>")
    if legend_html:
        parts.append(legend_html)
    parts.append(f'<div class="plot">{svg}</div>')
    if table_html:
        parts.append(table_html)
    parts.append("</figure>")
    return "".join(parts)


def empty_figure(title, subtitle, message, wide=False) -> str:
    classes = "figure" + (" figure-wide" if wide else "")
    return (
        f'<figure class="{classes}"><div class="fig-head"><div>'
        f'<h3 class="fig-title">{esc(title)}</h3>'
        f'<p class="fig-sub">{esc(subtitle)}</p></div></div>'
        f'<p class="empty">{esc(message)}</p></figure>'
    )


# ==========================================================================
# The three charts
# ==========================================================================


def elo_chart(run: Run, wide=True) -> str:
    """The money plot: Elo vs self-play games, with a least-squares fit and R²."""
    points = run.elo_points
    title = "Elo vs self-play games"
    subtitle = "Bradley-Terry rating against the fixed anchor pool. Random = 0 Elo."
    if len(points) < 2:
        return empty_figure(
            title, subtitle, "Waiting for at least two checkpoint evaluations in elo.jsonl.", wide
        )

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    errs = [p[2] for p in points]
    plot = Plot(width=1080, height=344, left=62)
    lo_band = [y - e for y, e in zip(ys, errs)]
    hi_band = [y + e for y, e in zip(ys, errs)]
    xt, yt = plot.scale(xs, lo_band + hi_band, x_count=7, y_count=6)
    plot.frame(xt, yt, fmt_compact, lambda v: fmt_float(v, 0), "self-play games", "Elo")

    fit = run.fit
    if fit:
        slope, intercept, r2 = fit
        plot.line(
            [(plot.xlo, slope * plot.xlo + intercept), (plot.xhi, slope * plot.xhi + intercept)],
            SERIES[1],
            dash="6 5",
        )
        # Identify the fit in-plot at its quiet end; the numbers live in the
        # readout above, so they are not repeated here.
        plot.label(
            plot.xlo,
            slope * plot.xlo + intercept,
            "linear fit",
            anchor="start",
            dy=21,
            dx=7,
            cls="plot-note",
        )

    plot.errbars(list(zip(xs, ys, errs)), SERIES[0])
    plot.line(list(zip(xs, ys)), SERIES[0])
    plot.dots(list(zip(xs, ys)), SERIES[0])
    last_x, last_y = xs[-1], ys[-1]
    plot.label(last_x, last_y, f"{last_y:,.0f}", anchor="end", dy=-14)
    plot.register(
        "Elo",
        SERIES[0],
        [
            (x, y, f"{fmt_int(x)} games", f"{y:,.0f} ± {e:,.0f}" + (f"  ({c})" if c else ""))
            for x, y, e, c, _ in points
        ],
    )

    readout = ""
    if fit:
        slope, _, r2 = fit
        verdict = "roughly linear" if r2 >= 0.9 else ("bending" if r2 >= 0.6 else "not linear")
        readout = (
            f'<div class="stat-inline"><span class="k">R²</span>'
            f'<span class="v">{r2:.3f}</span></div>'
            f'<div class="stat-inline"><span class="k">slope</span>'
            f'<span class="v">{fmt_float(slope * 1000, 1)}<span class="u">/1k games</span></span></div>'
            f'<div class="stat-inline"><span class="k">shape</span>'
            f'<span class="v small">{esc(verdict)}</span></div>'
        )

    rows = [
        (p[3] or "—", fmt_int(p[0]), f"{p[1]:,.1f}", f"±{p[2]:,.1f}", fmt_int(num(p[4], "n_games")))
        for p in points
    ]
    return figure(
        title,
        subtitle,
        plot.svg("Elo versus self-play games with linear fit"),
        legend([("measured Elo", SERIES[0], "dot"), ("least-squares fit", SERIES[1], "dash")]),
        readout,
        table(["checkpoint", "games", "Elo", "err", "eval games"], rows, cap=40),
        wide,
    )


def loss_chart(run: Run) -> str:
    title = "Training loss"
    subtitle = "Total, policy and value loss over self-play games."
    records = [r for r in run.train if num(r, "games") is not None]
    if len(records) < 2:
        return empty_figure(title, subtitle, "Waiting for train.jsonl to fill up.")

    records = thin(records)
    defs = [("total", "loss", SERIES[2]), ("policy", "loss_p", SERIES[0]), ("value", "loss_v", SERIES[1])]
    present = []
    for label, key, color in defs:
        pts = [(num(r, "games"), num(r, key)) for r in records]
        pts = [(x, y) for x, y in pts if x is not None and y is not None]
        if len(pts) >= 2:
            present.append((label, key, color, pts))
    if not present:
        return empty_figure(title, subtitle, "train.jsonl has no loss fields yet.")

    xs = [x for _, _, _, pts in present for x, _ in pts]
    ys = [y for _, _, _, pts in present for _, y in pts]
    plot = Plot()
    xt, yt = plot.scale(xs, ys, y_zero=True, y_pad=0.05)
    plot.frame(xt, yt, fmt_compact, lambda v: fmt_float(v, 2), "self-play games", "loss")
    for label, _, color, pts in present:
        plot.line(pts, color)
    # Direct-label the end points only where they will not collide; the legend
    # and the tooltip carry the rest.
    ends = sorted(((pts[-1][1], label, color, pts) for label, _, color, pts in present), reverse=True)
    labelled_y: list[float] = []
    for y_value, label, color, pts in ends:
        x, y = pts[-1]
        plot.dots([(x, y)], color, radius=3.5)
        py = plot.Y(y)
        if all(abs(py - other) > 15 for other in labelled_y):
            plot.label(x, y, f"{y:,.2f}", anchor="end", dy=-11)
            labelled_y.append(py)
    for label, _, color, pts in present:
        plot.register(
            label, color, [(px, py, f"{fmt_int(px)} games", f"{py:,.3f}") for px, py in pts]
        )

    last = records[-1]
    lr = num(last, "lr")
    lr_text = "—" if lr is None else f"{lr:.2e}"
    readout = (
        f'<div class="stat-inline"><span class="k">buffer</span>'
        f'<span class="v">{fmt_compact(num(last, "buffer"))}</span></div>'
        f'<div class="stat-inline"><span class="k">lr</span>'
        f'<span class="v">{esc(lr_text)}</span></div>'
    )
    rows = [
        (
            fmt_int(num(r, "games")),
            fmt_int(num(r, "steps")),
            fmt_float(num(r, "loss"), 3),
            fmt_float(num(r, "loss_p"), 3),
            fmt_float(num(r, "loss_v"), 3),
            fmt_int(num(r, "buffer")),
        )
        for r in run.train
    ]
    return figure(
        title,
        subtitle,
        plot.svg("Training loss curves"),
        legend([(label, color, "line") for label, _, color, _ in present]),
        readout,
        table(["games", "steps", "loss", "policy", "value", "buffer"], rows, cap=25),
    )


def winrate_chart(run: Run) -> str:
    title = "Win rate vs anchors"
    subtitle = "Share of evaluation games won against each fixed opponent."
    series: dict[str, list[tuple[float, float]]] = {}
    for games, _, _, _, record in run.elo_points:
        versus = record.get("vs")
        if not isinstance(versus, dict):
            continue
        for opponent in versus:
            value = num(versus, opponent)
            if value is None:
                continue
            series.setdefault(str(opponent), []).append((games, value * 100.0))
    series = {k: v for k, v in series.items() if len(v) >= 1}
    if not series:
        return empty_figure(title, subtitle, "No per-opponent results in elo.jsonl yet.")

    # Colour follows the opponent, never its rank: sort by name, keep the first
    # MAX_SERIES, and fold any tail rather than cycling hues.
    names = sorted(series, key=lambda k: (-len(series[k]), k))
    folded = names[MAX_SERIES:]
    names = names[:MAX_SERIES]

    xs = [x for name in names for x, _ in series[name]]
    plot = Plot()
    xt, yt = plot.scale(xs, [0.0, 100.0], y_count=4, y_pad=0.0)
    plot.frame(xt, yt, fmt_compact, lambda v: f"{v:,.0f}%", "self-play games", "win rate")
    plot.hline(50.0, "even")
    entries = []
    for index, name in enumerate(names):
        color = SERIES[index % len(SERIES)]
        pts = sorted(series[name])
        entries.append((name, color, "line"))
        plot.line(pts, color)
        plot.dots([pts[-1]], color, radius=3.5)
        plot.register(name, color, [(x, y, f"{fmt_int(x)} games", f"{y:,.1f}%") for x, y in pts])
    if len(names) <= 3:
        for index, name in enumerate(names):
            x, y = max(series[name])
            plot.label(x, y, f"{y:,.0f}%", anchor="end", dy=-11)

    all_names = sorted(series)
    rows = []
    for games, _, _, ckpt, record in run.elo_points:
        versus = record.get("vs") if isinstance(record.get("vs"), dict) else {}
        rows.append(
            [ckpt or "—", fmt_int(games)]
            + [
                ("—" if num(versus, n) is None else f"{num(versus, n) * 100:,.1f}%")
                for n in all_names
            ]
        )
    subtitle_full = subtitle + (
        f" Showing {len(names)} of {len(all_names)} opponents; the rest are in the table."
        if folded
        else ""
    )
    return figure(
        title,
        subtitle_full,
        plot.svg("Win rate against anchor opponents"),
        legend(entries),
        "",
        table(["checkpoint", "games"] + all_names, rows, cap=40),
    )


def overview_chart(runs: list[Run]) -> str:
    """Cross-run comparison, only worth drawing when there are several runs."""
    # Two distinct x values minimum, or the run contributes a legend entry with
    # no visible line.
    usable = [r for r in runs if len({p[0] for p in r.elo_points}) >= 2]
    if len(usable) < 2:
        return ""
    xs = [p[0] for r in usable for p in r.elo_points]
    ys = [p[1] for r in usable for p in r.elo_points]
    plot = Plot(width=1080, height=320, left=62)
    xt, yt = plot.scale(xs, ys, x_count=7)
    plot.frame(xt, yt, fmt_compact, lambda v: fmt_float(v, 0), "self-play games", "Elo")
    entries = []
    for index, run in enumerate(usable[:MAX_SERIES]):
        color = SERIES[index % len(SERIES)]
        pts = [(p[0], p[1]) for p in run.elo_points]
        entries.append((run.run_name, color, "line"))
        plot.line(pts, color)
        plot.dots([pts[-1]], color, radius=3.5)
        plot.register(
            run.run_name, color, [(x, y, f"{fmt_int(x)} games", f"{y:,.0f} Elo") for x, y in pts]
        )
    rows = [
        (
            r.run_name,
            r.state,
            fmt_int(r.games),
            fmt_float(r.latest_elo, 1),
            "—" if not r.fit else f"{r.fit[2]:.3f}",
            "—" if not r.fit else fmt_float(r.fit[0] * 1000, 1),
        )
        for r in usable
    ]
    return figure(
        "All runs",
        "Every run's Elo curve on one scale — the anchor pool is fixed, so they are comparable.",
        plot.svg("Elo curves for all runs"),
        legend(entries),
        "",
        table(["run", "state", "games", "latest Elo", "R²", "Elo / 1k games"], rows),
        wide=True,
    )


# ==========================================================================
# Status header, tiles, panels
# ==========================================================================


def state_pill(run: Run) -> str:
    glyph, label, cls = {
        "running": ("●", "running", "run"),
        "done": ("✓", "done", "done"),
        "failed": ("✕", "failed", "failed"),
    }.get(run.state, ("?", "unknown", "unknown"))
    return f'<span class="pill pill-{cls}"><span class="glyph" aria-hidden="true">{glyph}</span>{label}</span>'


def heartbeat(run: Run) -> str:
    cls = "hb hb-stale" if run.stale else "hb"
    updated_attr = "" if run.updated is None else f' data-updated="{run.updated:.0f}"'
    text = fmt_ago(run.age)
    if run.state in ("done", "failed") and run.updated is not None:
        text = f"last write {fmt_duration(run.age)} ago"
    live = ' data-live="1"' if run.state == "running" else ""
    warn = (
        '<span class="hb-warn">no heartbeat for over 3 min</span>'
        if run.stale and run.age is not None
        else ""
    )
    return f'<span class="{cls}"{updated_attr}{live}>{esc(text)}</span>{warn}'


def tile(label, value, unit="", sub="", hero=False) -> str:
    cls = "tile tile-hero" if hero else "tile"
    unit_html = f'<span class="u">{esc(unit)}</span>' if unit else ""
    sub_html = f'<div class="tile-sub">{esc(sub)}</div>' if sub else ""
    return (
        f'<div class="{cls}"><div class="tile-label">{esc(label)}</div>'
        f'<div class="tile-value">{esc(value)}{unit_html}</div>{sub_html}</div>'
    )


def config_block(run: Run) -> str:
    interesting = {k: v for k, v in run.config.items() if k not in ("run", "started")}
    if not interesting:
        return ""
    rows = []
    for key in sorted(interesting):
        value = interesting[key]
        if isinstance(value, (dict, list)):
            value = json.dumps(value, separators=(", ", ": "))
        rows.append(f'<div class="cfg-row"><dt>{esc(key)}</dt><dd>{esc(value)}</dd></div>')
    return (
        '<details class="config"><summary>Run configuration</summary>'
        f'<dl class="cfg">{"".join(rows)}</dl></details>'
    )


def integrity_note(run: Run) -> str:
    problems = []
    if run.train_dropped:
        problems.append(f"{run.train_dropped} unparsable line(s) in train.jsonl")
    if run.elo_dropped:
        problems.append(f"{run.elo_dropped} unparsable line(s) in elo.jsonl")
    if not run.status:
        problems.append("no status.json — state and heartbeat unknown")
    if not problems:
        return ""
    return f'<p class="integrity">Skipped while reading: {esc("; ".join(problems))}.</p>'


def run_panel(run: Run, primary: bool) -> str:
    elo = run.latest_elo
    err = run.latest_elo_err
    tiles = []
    if elo is None:
        tiles.append(tile("Latest Elo", "—", sub="no checkpoint rated yet", hero=primary))
    else:
        delta = ""
        if len(run.elo_points) >= 2:
            gain = elo - run.elo_points[0][1]
            delta = f"{gain:+,.0f} since first checkpoint"
        tiles.append(
            tile(
                "Latest Elo",
                f"{elo:,.0f}",
                unit=("" if not err else f"± {err:,.0f}"),
                sub=delta,
                hero=primary,
            )
        )
    tiles.append(tile("Self-play games", fmt_compact(run.games), sub=fmt_int(run.games)))
    tiles.append(tile("Optimizer steps", fmt_compact(run.steps), sub=fmt_int(run.steps)))
    tiles.append(
        tile(
            "Elapsed",
            fmt_duration(run.elapsed),
            sub=("" if run.started is None else "since " + stamp_text(run.started)),
        )
    )
    tiles.append(tile("Checkpoints rated", fmt_int(len(run.elo_points))))

    note = f'<p class="run-note">{esc(run.note)}</p>' if run.note else ""
    charts = [elo_chart(run), loss_chart(run), winrate_chart(run)]
    return (
        f'<section class="panel{"" if primary else " panel-secondary"}">'
        f'<header class="panel-head">'
        f'<div class="run-id"><span class="eyebrow">run</span>'
        f'<h2>{esc(run.run_name)}</h2></div>'
        f'<div class="panel-state">{state_pill(run)}{heartbeat(run)}</div>'
        f"</header>"
        f"{note}{integrity_note(run)}"
        f'<div class="tiles">{"".join(tiles)}</div>'
        f'<div class="grid">{charts[0]}{charts[1]}{charts[2]}</div>'
        f"{config_block(run)}"
        f"</section>"
    )


def stamp_text(epoch) -> str:
    if epoch is None:
        return "—"
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")  # noqa: DTZ006 (local clock)


# ==========================================================================
# Markdown (journal)
# ==========================================================================

_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\*\w])\*([^\*\n]+)\*(?!\*)")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def inline_md(text: str) -> str:
    """Escape, then apply code / bold / italic / link, in that order."""
    slots: list[str] = []

    def stash(match):
        slots.append(f"<code>{html.escape(match.group(1), quote=False)}</code>")
        return f"\x00{len(slots) - 1}\x00"

    text = _INLINE_CODE.sub(stash, text)
    text = html.escape(text, quote=False)
    text = _BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", text)
    text = _ITALIC.sub(lambda m: f"<em>{m.group(1)}</em>", text)
    text = _LINK.sub(
        lambda m: f'<a href="{html.escape(m.group(2), quote=True)}" rel="noreferrer">{m.group(1)}</a>',
        text,
    )
    return re.sub(r"\x00(\d+)\x00", lambda m: slots[int(m.group(1))], text)


def md_to_html(text: str) -> str:
    """A small hand-rolled subset: headings, lists, rules, code, paragraphs."""
    out: list[str] = []
    paragraph: list[str] = []
    list_kind = None
    in_code = False
    code: list[str] = []

    def flush_paragraph():
        nonlocal paragraph
        if paragraph:
            out.append(f"<p>{inline_md(' '.join(paragraph))}</p>")
            paragraph = []

    def close_list():
        nonlocal list_kind
        if list_kind:
            out.append(f"</{list_kind}>")
            list_kind = None

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip().startswith("```"):
            if in_code:
                out.append(f"<pre><code>{html.escape(chr(10).join(code), quote=False)}</code></pre>")
                code = []
                in_code = False
            else:
                flush_paragraph()
                close_list()
                in_code = True
            continue
        if in_code:
            code.append(raw)
            continue
        if not line.strip():
            flush_paragraph()
            close_list()
            continue
        if re.fullmatch(r"\s*([-*_]\s*){3,}", line):
            flush_paragraph()
            close_list()
            out.append('<hr class="md-rule" />')
            continue
        heading = re.match(r"(#{1,6})\s+(.*)", line.strip())
        if heading:
            flush_paragraph()
            close_list()
            depth = len(heading.group(1))
            if depth == 1 and not out:
                continue  # the section header already carries the document title
            level = min(6, depth + 1)  # the page itself owns <h1>
            out.append(f"<h{level}>{inline_md(heading.group(2).strip())}</h{level}>")
            continue
        bullet = re.match(r"(\s*)[-*+]\s+(.*)", line)
        ordered = re.match(r"(\s*)\d+[.)]\s+(.*)", line)
        item = bullet or ordered
        if item:
            flush_paragraph()
            kind = "ul" if bullet else "ol"
            if list_kind and list_kind != kind:
                close_list()
            if not list_kind:
                out.append(f"<{kind}>")
                list_kind = kind
            out.append(f"<li>{inline_md(item.group(2).strip())}</li>")
            continue
        if list_kind and raw.startswith(("  ", "\t")):
            # continuation of the previous bullet
            out[-1] = out[-1][: -len("</li>")] + " " + inline_md(line.strip()) + "</li>"
            continue
        close_list()
        paragraph.append(line.strip())

    if in_code and code:
        out.append(f"<pre><code>{html.escape(chr(10).join(code), quote=False)}</code></pre>")
    flush_paragraph()
    close_list()
    return "".join(out)


def journal_section(notes_path: Path) -> str:
    try:
        text = notes_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return (
            '<section class="journal"><header class="section-head">'
            '<span class="eyebrow">journal</span><h2>Notes for Remi</h2></header>'
            f'<p class="empty">{esc(str(notes_path.name))} not found.</p></section>'
        )
    return (
        '<section class="journal"><header class="section-head">'
        '<span class="eyebrow">journal</span><h2>Notes for Remi</h2>'
        f'<p class="section-sub">Read from {esc(notes_path.name)} · newest entries first</p>'
        "</header>"
        f'<div class="prose">{md_to_html(text)}</div></section>'
    )


# ==========================================================================
# Page assembly
# ==========================================================================

CSS = """
*, *::before, *::after { box-sizing: border-box; }
:root {
  color-scheme: dark;
  --plane: PLANE; --surface: SURFACE; --surface-2: SURFACE_2;
  --ink: INK; --ink-2: INK_2; --muted: MUTED;
  --grid: GRID; --axis: AXIS; --hairline: HAIRLINE;
  --good: ST_GOOD; --warn: ST_WARN; --crit: ST_CRIT; --accent: S0;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, ui-sans-serif, sans-serif;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  --serif: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
  --r: 10px;
}
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--plane); color: var(--ink);
  font-family: var(--sans); font-size: 15px; line-height: 1.55;
  padding: 0 22px 72px;
}
.wrap { max-width: 1180px; margin: 0 auto; }
a { color: var(--accent); text-underline-offset: 3px; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 3px; }

/* masthead ------------------------------------------------------------- */
.masthead { padding: 46px 0 26px; border-bottom: 1px solid var(--hairline); }
.brand { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
.brand h1 {
  font-size: 27px; margin: 0; font-weight: 600; letter-spacing: -0.015em;
}
.thesis {
  font-family: var(--serif); font-style: italic; color: var(--ink-2); font-size: 17px;
}
.masthead-meta {
  margin: 14px 0 0; display: flex; gap: 22px; flex-wrap: wrap;
  font-family: var(--mono); font-size: 11.5px; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--muted);
}
.eyebrow {
  font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--muted);
}

/* panels --------------------------------------------------------------- */
.panel { padding: 34px 0 6px; border-bottom: 1px solid var(--hairline); }
.panel-head {
  display: flex; align-items: flex-end; justify-content: space-between;
  gap: 18px; flex-wrap: wrap;
}
.run-id { display: flex; flex-direction: column; gap: 2px; }
.run-id h2 { margin: 0; font-size: 22px; font-weight: 600; letter-spacing: -0.01em; }
.panel-secondary .run-id h2 { font-size: 19px; }
.panel-state { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.pill {
  display: inline-flex; align-items: center; gap: 7px;
  font-family: var(--mono); font-size: 11.5px; letter-spacing: 0.1em;
  text-transform: uppercase; padding: 5px 11px; border-radius: 999px;
  border: 1px solid var(--hairline); background: var(--surface-2); color: var(--ink);
}
.pill .glyph { font-size: 10px; line-height: 1; }
.pill-run { color: var(--good); border-color: color-mix(in srgb, var(--good) 45%, transparent); }
.pill-run .glyph { animation: beat 2.4s ease-in-out infinite; }
.pill-done { color: var(--ink-2); }
.pill-failed { color: var(--crit); border-color: color-mix(in srgb, var(--crit) 50%, transparent); }
.pill-unknown { color: var(--muted); }
@keyframes beat { 0%, 100% { opacity: 1; } 50% { opacity: 0.28; } }
.hb { font-family: var(--mono); font-size: 12px; color: var(--muted); }
.hb-stale { color: var(--crit); }
.hb-warn {
  font-family: var(--mono); font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--crit); border: 1px solid color-mix(in srgb, var(--crit) 50%, transparent);
  padding: 3px 8px; border-radius: 999px;
}
.run-note {
  margin: 16px 0 0; font-family: var(--serif); font-size: 15.5px; color: var(--ink-2);
  border-left: 2px solid var(--axis); padding-left: 14px;
}
.integrity {
  margin: 14px 0 0; font-family: var(--mono); font-size: 12px; color: var(--warn);
}

/* tiles ---------------------------------------------------------------- */
.tiles {
  margin: 22px 0 26px; display: grid; gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(158px, 1fr));
}
.tile {
  background: var(--surface); border: 1px solid var(--hairline); border-radius: var(--r);
  padding: 14px 16px 15px; display: flex; flex-direction: column; gap: 4px;
}
.tile-label {
  font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.13em;
  text-transform: uppercase; color: var(--muted);
}
.tile-value { font-size: 25px; font-weight: 600; letter-spacing: -0.02em; line-height: 1.15; }
.tile-value .u {
  font-size: 13px; font-weight: 400; color: var(--ink-2); margin-left: 6px; letter-spacing: 0;
}
.tile-sub { font-family: var(--mono); font-size: 11.5px; color: var(--muted); }
.tile-hero { grid-column: span 2; background: var(--surface-2); }
.tile-hero .tile-value { font-size: 54px; letter-spacing: -0.03em; }
.tile-hero .tile-value .u { font-size: 17px; }

/* figures -------------------------------------------------------------- */
.grid { display: grid; gap: 16px; grid-template-columns: repeat(2, minmax(0, 1fr)); }
.figure {
  margin: 0; background: var(--surface); border: 1px solid var(--hairline);
  border-radius: var(--r); padding: 18px 18px 14px; min-width: 0;
}
.figure-wide { grid-column: 1 / -1; }
.fig-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; flex-wrap: wrap; }
.fig-title { margin: 0; font-size: 15px; font-weight: 600; letter-spacing: -0.005em; }
.fig-sub { margin: 4px 0 0; font-size: 12.5px; color: var(--muted); max-width: 62ch; }
.readout { display: flex; gap: 20px; flex-wrap: wrap; }
.stat-inline { display: flex; flex-direction: column; align-items: flex-end; gap: 1px; }
.stat-inline .k {
  font-family: var(--mono); font-size: 10px; letter-spacing: 0.13em;
  text-transform: uppercase; color: var(--muted);
}
.stat-inline .v {
  font-family: var(--mono); font-size: 16px; font-variant-numeric: tabular-nums; color: var(--ink);
}
.stat-inline .v.small { font-size: 12.5px; color: var(--ink-2); }
.stat-inline .v .u { font-size: 10.5px; color: var(--muted); margin-left: 4px; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin: 14px 0 2px; }
.legend-item {
  display: inline-flex; align-items: center; gap: 7px;
  font-family: var(--mono); font-size: 11.5px; color: var(--ink-2);
}
.key { width: 18px; height: 8px; overflow: visible; flex: none; }
.plot { position: relative; margin-top: 4px; }
/* Each chart carries its natural pixel width inline, so SVG text renders at its
   nominal size. max-width lets it shrink on narrow viewports but never blow up. */
svg.chart { display: block; max-width: 100%; height: auto; overflow: visible; }
svg.chart .grid { stroke: var(--grid); stroke-width: 1; }
svg.chart .axis { stroke: var(--axis); stroke-width: 1; }
svg.chart .ref { stroke: var(--axis); stroke-width: 1; }
svg.chart .series-line { fill: none; stroke-linejoin: round; stroke-linecap: round; }
svg.chart text { font-family: var(--mono); fill: var(--muted); }
svg.chart .tick { font-size: 10.5px; font-variant-numeric: tabular-nums; }
svg.chart .tick-y { text-anchor: end; }
svg.chart .tick-x { text-anchor: middle; }
svg.chart .axis-title { font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; }
svg.chart .ref-label { font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; }
svg.chart .point-label { font-size: 11px; fill: INK_2; font-variant-numeric: tabular-nums; }
svg.chart .plot-note { font-size: 11px; fill: INK_2; }
svg.chart .crosshair { stroke: var(--axis); stroke-width: 1; opacity: 0; }
svg.chart .hotdot { fill: none; stroke: var(--ink); stroke-width: 2; opacity: 0; }
svg.chart.hot .crosshair, svg.chart.hot .hotdot { opacity: 1; }

/* tooltip -------------------------------------------------------------- */
.tip {
  position: absolute; pointer-events: none; z-index: 20; min-width: 130px;
  background: SURFACE_2; border: 1px solid var(--hairline); border-radius: 8px;
  padding: 9px 11px; box-shadow: 0 10px 26px rgba(0,0,0,0.45);
  font-family: var(--mono); font-size: 11.5px; color: var(--ink);
  opacity: 0; transition: opacity 0.09s linear;
}
.tip.on { opacity: 1; }
.tip .tip-x { color: var(--muted); letter-spacing: 0.06em; text-transform: uppercase; font-size: 10px; }
.tip .tip-row { display: flex; align-items: center; gap: 7px; margin-top: 5px; white-space: nowrap; }
.tip .swatch { width: 8px; height: 8px; border-radius: 2px; flex: none; }
.tip .tip-name { color: var(--ink-2); }
.tip .tip-val { margin-left: auto; font-variant-numeric: tabular-nums; }

/* tables & details ----------------------------------------------------- */
details { margin-top: 14px; }
summary {
  cursor: pointer; font-family: var(--mono); font-size: 11px; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted);
}
summary:hover { color: var(--ink-2); }
.table-wrap { overflow-x: auto; margin-top: 10px; }
table { border-collapse: collapse; width: 100%; font-family: var(--mono); font-size: 11.5px; }
th, td { text-align: right; padding: 5px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--muted); font-weight: 400; letter-spacing: 0.06em; text-transform: uppercase; font-size: 10px; }
td { font-variant-numeric: tabular-nums; color: var(--ink-2); }
th:first-child, td:first-child { text-align: left; }
.table-note { font-family: var(--mono); font-size: 11px; color: var(--muted); }
.cfg { margin: 12px 0 0; display: grid; gap: 2px 18px; }
.cfg-row { display: flex; gap: 14px; border-bottom: 1px solid var(--grid); padding: 4px 0; }
.cfg dt { font-family: var(--mono); font-size: 11.5px; color: var(--muted); min-width: 190px; }
.cfg dd {
  margin: 0; font-family: var(--mono); font-size: 11.5px; color: var(--ink-2);
  overflow-wrap: anywhere;
}
.empty {
  margin: 26px 0; font-family: var(--mono); font-size: 12px; color: var(--muted);
  border: 1px dashed var(--grid); border-radius: 8px; padding: 20px; text-align: center;
}

/* journal -------------------------------------------------------------- */
.journal { padding: 40px 0 0; }
.section-head { margin-bottom: 20px; }
.section-head h2 { margin: 2px 0 0; font-size: 22px; font-weight: 600; }
.section-sub { margin: 6px 0 0; font-family: var(--mono); font-size: 11.5px; color: var(--muted); }
.prose {
  font-family: var(--serif); font-size: 16.5px; line-height: 1.68; color: var(--ink-2);
  max-width: 68ch;
}
.prose h2 {
  font-family: var(--sans); font-size: 18px; font-weight: 600; color: var(--ink);
  margin: 34px 0 10px; letter-spacing: -0.01em;
}
.prose h3 {
  font-family: var(--sans); font-size: 16px; font-weight: 600; color: var(--ink);
  margin: 30px 0 10px; letter-spacing: -0.005em;
}
.prose h2 + p, .prose h3 + p { margin-top: 0; }
.prose h4, .prose h5, .prose h6 {
  font-family: var(--mono); font-size: 12px; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--muted); margin: 24px 0 8px; font-weight: 500;
}
.prose p { margin: 0 0 14px; }
.prose ul, .prose ol { margin: 0 0 16px; padding-left: 22px; }
.prose li { margin: 0 0 7px; }
.prose strong { color: var(--ink); font-weight: 600; }
.prose code, .prose pre {
  font-family: var(--mono); font-size: 13px; background: var(--surface);
  border: 1px solid var(--hairline); border-radius: 5px;
}
.prose code { padding: 1px 5px; color: var(--ink); }
.prose pre { padding: 12px 14px; overflow-x: auto; }
.prose pre code { border: 0; padding: 0; background: none; }
.md-rule { border: 0; border-top: 1px solid var(--grid); margin: 26px 0; }

.foot {
  margin-top: 46px; padding-top: 18px; border-top: 1px solid var(--hairline);
  font-family: var(--mono); font-size: 11px; color: var(--muted);
  display: flex; gap: 20px; flex-wrap: wrap;
}

@media (max-width: 900px) {
  .grid { grid-template-columns: minmax(0, 1fr); }
  .tile-hero { grid-column: 1 / -1; }
  .tile-hero .tile-value { font-size: 44px; }
  /* The wide chart is the only one that has to shrink below 1:1 here; scale its
     in-SVG type up in user units so it lands near its nominal size again. */
  .figure-wide svg.chart .tick { font-size: 16px; }
  .figure-wide svg.chart .axis-title, .figure-wide svg.chart .ref-label { font-size: 15px; }
  .figure-wide svg.chart .point-label, .figure-wide svg.chart .plot-note { font-size: 17px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
"""

JS = """
(function () {
  'use strict';

  // Heartbeat ages tick in the browser, so an open tab still shows the truth
  // between regenerations.
  var STALE = 180;
  function dur(s) {
    s = Math.max(0, Math.round(s));
    var d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600);
    var m = Math.floor((s % 3600) / 60), x = s % 60;
    if (d) return d + 'd ' + h + 'h';
    if (h) return h + 'h ' + (m < 10 ? '0' : '') + m + 'm';
    if (m) return m + 'm ' + (x < 10 ? '0' : '') + x + 's';
    return x + 's';
  }
  function ticker() {
    var now = Date.now() / 1000;
    var nodes = document.querySelectorAll('.hb[data-updated]');
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i], age = now - parseFloat(el.getAttribute('data-updated'));
      if (age < 0) age = 0;
      var live = el.getAttribute('data-live') === '1';
      el.textContent = (live ? 'updated ' : 'last write ') + dur(age) + ' ago';
      if (live) el.classList.toggle('hb-stale', age > STALE);
    }
  }
  ticker();
  setInterval(ticker, 1000);

  // Nearest-point crosshair + tooltip for every chart. Values are also in each
  // chart's data table, so hover only ever enhances.
  var tip = document.createElement('div');
  tip.className = 'tip';
  document.body.appendChild(tip);

  function pointer(svg, evt) {
    var box = svg.getBoundingClientRect();
    var vb = svg.viewBox.baseVal;
    var scale = box.width / (vb.width || 1);
    return {
      x: (evt.clientX - box.left) / (scale || 1),
      y: (evt.clientY - box.top) / (scale || 1),
      box: box, scale: scale
    };
  }

  function show(svg, data, px, py, geom) {
    var best = null, bestD = Infinity;
    for (var s = 0; s < data.series.length; s++) {
      var pts = data.series[s].p;
      for (var i = 0; i < pts.length; i++) {
        var dx = pts[i][0] - px, dy = pts[i][1] - py;
        var d = dx * dx + dy * dy * 0.3;
        if (d < bestD) { bestD = d; best = { s: s, i: i }; }
      }
    }
    if (!best || bestD > 90 * 90) { hide(svg); return; }
    var anchor = data.series[best.s].p[best.i];
    var rows = '';
    for (var t = 0; t < data.series.length; t++) {
      var series = data.series[t];
      for (var k = 0; k < series.p.length; k++) {
        if (Math.abs(series.p[k][0] - anchor[0]) < 0.6) {
          rows += '<div class="tip-row"><span class="swatch" style="background:' + series.c +
            '"></span><span class="tip-name">' + series.n + '</span><span class="tip-val">' +
            series.p[k][3] + '</span></div>';
          break;
        }
      }
    }
    tip.innerHTML = '<div class="tip-x">' + anchor[2] + '</div>' + rows;
    tip.classList.add('on');
    svg.classList.add('hot');
    var cross = svg.querySelector('.crosshair');
    var dot = svg.querySelector('.hotdot');
    if (cross) { cross.setAttribute('x1', anchor[0]); cross.setAttribute('x2', anchor[0]); }
    if (dot) { dot.setAttribute('cx', anchor[0]); dot.setAttribute('cy', anchor[1]); }
    var host = svg.parentNode;
    var hostBox = host.getBoundingClientRect();
    var left = geom.box.left - hostBox.left + anchor[0] * geom.scale + 14;
    var top = geom.box.top - hostBox.top + anchor[1] * geom.scale - 12;
    if (left + tip.offsetWidth > hostBox.width) left -= tip.offsetWidth + 28;
    tip.style.left = Math.max(0, left) + 'px';
    tip.style.top = Math.max(0, top) + 'px';
    host.appendChild(tip);
  }

  function hide(svg) {
    tip.classList.remove('on');
    if (svg) svg.classList.remove('hot');
  }

  var charts = document.querySelectorAll('svg.chart[data-chart]');
  for (var c = 0; c < charts.length; c++) {
    (function (svg) {
      var data;
      try { data = JSON.parse(svg.getAttribute('data-chart')); } catch (e) { return; }
      if (!data || !data.series || !data.series.length) return;
      var cursor = { s: 0, i: data.series[0].p.length - 1 };
      svg.addEventListener('mousemove', function (evt) {
        var p = pointer(svg, evt);
        show(svg, data, p.x, p.y, p);
      });
      svg.addEventListener('mouseleave', function () { hide(svg); });
      svg.addEventListener('blur', function () { hide(svg); });
      svg.addEventListener('keydown', function (evt) {
        var step = evt.key === 'ArrowRight' ? 1 : (evt.key === 'ArrowLeft' ? -1 : 0);
        if (!step) return;
        evt.preventDefault();
        var pts = data.series[cursor.s].p;
        cursor.i = Math.min(pts.length - 1, Math.max(0, cursor.i + step));
        var target = pts[cursor.i];
        var box = svg.getBoundingClientRect();
        var vb = svg.viewBox.baseVal;
        show(svg, data, target[0], target[1],
             { box: box, scale: box.width / (vb.width || 1) });
      });
    })(charts[c]);
  }
})();
"""


def styles() -> str:
    css = CSS
    for token, value in (
        ("PLANE", PLANE),
        ("SURFACE_2", SURFACE_2),
        ("SURFACE", SURFACE),
        ("INK_2", INK_2),
        ("INK", INK),
        ("MUTED", MUTED),
        ("GRID", GRID),
        ("AXIS", AXIS),
        ("HAIRLINE", HAIRLINE),
        ("ST_GOOD", ST_GOOD),
        ("ST_WARN", ST_WARN),
        ("ST_CRIT", ST_CRIT),
        ("S0", SERIES[0]),
    ):
        css = css.replace(token, value)
    return css


def build_page(runs: list[Run], notes_path: Path, now: float) -> str:
    generated = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")  # noqa: DTZ006
    live = sum(1 for r in runs if r.state == "running")
    stale = sum(1 for r in runs if r.stale)

    meta = [
        f"generated {generated}",
        f"{len(runs)} run{'s' if len(runs) != 1 else ''}",
        f"{live} running",
        "auto-refresh 30s",
    ]
    if stale:
        meta.append(f"{stale} stale")

    body = [
        '<div class="wrap">',
        '<header class="masthead"><div class="brand">',
        "<h1>Ludometer</h1>",
        '<p class="thesis">Does a good game teach linearly?</p></div>',
        f'<div class="masthead-meta">{"".join(f"<span>{esc(m)}</span>" for m in meta)}</div>',
        "</header>",
    ]

    if not runs:
        body.append(
            '<p class="empty">No runs found under runs/. Start a training run and this page '
            "fills in on the next refresh.</p>"
        )
    else:
        overview = overview_chart(runs)
        if overview:
            body.append(f'<section class="panel"><div class="grid">{overview}</div></section>')
        for index, run in enumerate(runs):
            body.append(run_panel(run, primary=(index == 0)))

    body.append(journal_section(notes_path))
    body.append(
        '<footer class="foot"><span>ludometer · generated by web/make_dashboard.py</span>'
        "<span>static page, no external requests</span>"
        f"<span>{esc(generated)}</span></footer>"
    )
    body.append("</div>")

    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8" />\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        '<meta http-equiv="refresh" content="30" />\n'
        "<title>Ludometer Training Monitor</title>\n"
        f"<style>{styles()}</style>\n"
        "</head><body>\n"
        + "".join(body)
        + f"\n<script>{JS}</script>\n</body></html>\n"
    )


# ==========================================================================
# Entry point
# ==========================================================================


def generate(runs_dir: Path, notes_path: Path, out_path: Path) -> tuple[int, int]:
    now = time.time()
    runs = discover_runs(runs_dir, now)
    page = build_page(runs, notes_path, now)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(out_path)  # atomic: a refreshing tab never sees a half file
    return len(runs), len(page)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the Ludometer dashboard.")
    parser.add_argument("--runs", type=Path, default=REPO / "runs", help="directory of run dirs")
    parser.add_argument("--notes", type=Path, default=REPO / "NOTES_FOR_REMI.md", help="journal file")
    parser.add_argument("--out", type=Path, default=REPO / "web" / "dashboard.html", help="output HTML")
    parser.add_argument(
        "--watch",
        type=float,
        metavar="N",
        default=None,
        help="regenerate every N seconds, forever (Ctrl-C to stop)",
    )
    parser.add_argument("--quiet", action="store_true", help="only report errors")
    args = parser.parse_args(argv)

    def once():
        count, size = generate(args.runs, args.notes, args.out)
        if not args.quiet:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] wrote {args.out} "  # noqa: DTZ005
                f"({count} run{'s' if count != 1 else ''}, {size / 1024:.0f} kB)"
            )

    if args.watch is None:
        once()
        return 0

    interval = max(1.0, float(args.watch))
    if not args.quiet:
        print(f"watching {args.runs} every {interval:g}s — Ctrl-C to stop")
    try:
        while True:
            try:
                once()
            except Exception as exc:  # noqa: BLE001 - keep the watcher alive through any IO hiccup
                print(f"error: {exc}", file=sys.stderr)
            time.sleep(interval)
    except KeyboardInterrupt:
        if not args.quiet:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
