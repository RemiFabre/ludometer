#!/usr/bin/env python3
"""Regenerate web/harvest.html — the live progress page for the BGA human-games crawl.

Stdlib only, no imports from ``ludometer`` (this is a viewer, and it must keep
working while the fetcher is mid-write), no external requests, one self-contained
dark file that refreshes itself every 20 s. Same conventions as
``web/make_dashboard.py``: inline CSS, inline SVG if any, atomic writes, and a
``--watch N`` mode so a tab left open follows the crawl.

It reads three things, all optional and all tolerated half-written:

* ``<state>/state.json`` — the fetcher's resume point: the rank-ordered cursor
  (which rank, which player, which offset inside that player's table list), the
  ladder snapshot, every table verdict with its reason, the per-day request and
  replay counters, the pace, and the error that stopped the last run;
* ``<state>/fetch.jsonl`` — one line per table decision, which gives the log tail
  and the measured fetch rate;
* ``<state>/replay.stats.json`` — what the dataset builder actually wrote:
  positions, **elite** positions (the target player's turns) out of all decision
  points, and per-reason rejection counts. If it is missing but ``replay.npz``
  exists, the row count is read out of the npz's ``states.npy`` header with
  ``zipfile`` — no numpy needed.

Usage:
    python3 web/make_harvest.py                               # write it once
    python3 web/make_harvest.py --state data/human --watch 20  # follow the crawl
    python3 web/make_harvest.py --state /tmp/h --out /tmp/h.html

Everything renders with an empty or partial state, because the first thing anybody
does is open the page *before* starting the crawl.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Design tokens — the same committed dark look as the training dashboard
# (web/make_dashboard.py). Duplicated on purpose: this file must stand alone.
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
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"]
ST_GOOD = "#0ca30c"
ST_WARN = "#fab219"
ST_CRIT = "#d03b3b"

#: docs/HUMAN_GAMES.md §5.3: the full target is ~550k positions (~10k games), the
#: first milestone 2k games. Elite-only collection roughly halves the positions per
#: game, which is exactly why the page shows both counts side by side.
POSITION_TARGET = 550_000
GAME_MILESTONES = (2_000, 10_000)
#: assumed only when the state file has no ``pace`` block yet
ASSUMED_REPLAYS_PER_DAY = 120
REFRESH_SECONDS = 20
LOG_TAIL = 24

#: How a rejection reason (a free-text string from the fetcher or the converter)
#: is bucketed for the tally. Order matters — the first matching bucket wins.
REASON_BUCKETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "gray-wall variant",
        (
            "wall variant",
            "non-standard wall",
            "wall placement",
            "variable-wall",
            "grey wall",
            "gray wall",
        ),
    ),
    ("below Elo floor", ("below floor", "elo floor", "below the target elo", "elo")),
    ("not 2 players", ("players, need 2", "players")),
    ("unranked or wrong mode", ("unranked", "game mode")),
    ("replay unavailable", ("replay lost", "has been lost", "replay for this game")),
    ("not finished", ("status ",)),
    (
        "replay failed validation",
        (
            "illegal action",
            "census",
            "bga reported",
            "unknown notification",
            "log says seat",
            "log ran out",
            "log continues past",
            "off-board",
            "no picks",
            "no factory deals",
            "engine",
            "deal",
            "colour",
            "color",
            "parse",
        ),
    ),
)


# ==========================================================================
# Loading — every reader returns something usable for a missing/partial file
# ==========================================================================
def read_json(path: Path) -> dict:
    """Parse a JSON file, returning ``{}`` if missing, empty or half-written."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return {}
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path, limit: int = 2000) -> list[dict]:
    """Parse a JSONL log, skipping unparsable lines (the writer may be mid-write)."""
    out: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines()[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def npz_rows(path: Path, member: str = "states") -> int | None:
    """Row count of an array inside a ``.npz``, read from its ``.npy`` header.

    A ``replay.npz`` for the full target is hundreds of megabytes, and this page is
    stdlib-only, so "how many positions are in the file?" is answered by parsing 100
    bytes of header out of the zip member rather than by loading the array. The
    sidecar written by the dataset builder is preferred; this is the fallback for a
    file built before the sidecar existed.
    """
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            name = next(
                (n for n in names if n in (member, member + ".npy")),
                None,
            )
            if name is None:
                return None
            with archive.open(name) as handle:
                if handle.read(6) != b"\x93NUMPY":
                    return None
                major = handle.read(2)[0]
                width = 2 if major == 1 else 4
                length = int.from_bytes(handle.read(width), "little")
                header = handle.read(length).decode("latin-1")
        shape = ast.literal_eval(header.strip()).get("shape")
    except (OSError, ValueError, SyntaxError, KeyError, zipfile.BadZipFile):
        return None
    if not isinstance(shape, tuple) or not shape:
        return None
    try:
        return int(shape[0])
    except (TypeError, ValueError):
        return None


def safe(value: Any, default: float = 0.0) -> float:
    """A finite float, always — the page must never print ``nan`` or ``inf``."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return out


def bucket_for(reason: str) -> str:
    low = str(reason).lower()
    for name, needles in REASON_BUCKETS:
        if any(needle in low for needle in needles):
            return name
    return "other"


# ==========================================================================
# The facts, gathered once
# ==========================================================================
class Harvest:
    """Everything the page shows, read off disk and pre-digested.

    Deliberately defensive: a field is either a real number or ``None``, and
    ``None`` renders as an em dash rather than as a zero that looks like progress.
    """

    def __init__(self, state_dir: Path, npz: Path | None = None, now: float = 0.0):
        self.now = now or time.time()
        self.state_dir = Path(state_dir)
        self.state_path = (
            self.state_dir
            if self.state_dir.suffix == ".json"
            else self.state_dir / "state.json"
        )
        base = self.state_path.parent
        self.log_path = base / "fetch.jsonl"
        self.npz_path = Path(npz) if npz else base / "replay.npz"
        self.stats_path = self.npz_path.with_name(self.npz_path.stem + ".stats.json")

        self.state = read_json(self.state_path)
        self.stats = read_json(self.stats_path)
        self.events = read_jsonl(self.log_path)
        self.started = self.state.get("started") or ""

        # ---- the cursor: where in the ladder we are
        cursor = self.state.get("cursor") or {}
        self.rank = int(safe(cursor.get("rank"), 0)) or None
        self.cursor_player = int(safe(cursor.get("player_id"), 0)) or None
        self.table_offset = int(safe(cursor.get("table_offset"), 0))
        self.players_done = int(safe(cursor.get("players_done"), 0))
        self.cursor_updated = str(cursor.get("updated") or "")

        # ---- the ladder snapshot, so the cursor can be named
        self.ladder = {}
        for row in (self.state.get("ranking") or {}).get("rows") or []:
            try:
                self.ladder[int(row.get("player_id"))] = row
            except (TypeError, ValueError):
                continue
        self.ladder_size = len(self.ladder)
        row = self.ladder.get(self.cursor_player or -1) or {}
        self.player_name = str(row.get("name") or "")
        self.player_elo = row.get("elo_display")
        self.player_elo_raw = row.get("elo_raw")
        self.player_games = row.get("games_played")

        # ---- verdicts
        tables = self.state.get("tables") or {}
        self.decided = len(tables)
        self.downloaded = 0
        self.skipped = 0
        self.errored = 0
        self.reasons: dict[str, int] = {}
        for entry in tables.values():
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status", ""))
            reason = str(entry.get("reason", ""))
            if status == "downloaded":
                self.downloaded += 1
                continue
            if status == "error":
                self.errored += 1
            else:
                self.skipped += 1
            key = bucket_for(reason)
            self.reasons[key] = self.reasons.get(key, 0) + 1

        # ---- counters
        downloads = self.state.get("downloads") or {}
        requests = self.state.get("requests") or {}
        self.downloads_total = int(safe(downloads.get("total"), self.downloaded))
        self.requests_total = int(safe(requests.get("total"), 0))
        # the fetcher's per-day counters are keyed by the UTC date, so match it
        today = datetime.fromtimestamp(self.now, UTC).date().isoformat()
        self.downloads_today = int(safe(downloads.get(today), 0))
        self.requests_today = int(safe(requests.get(today), 0))

        # ---- pace (from the state file when the crawl wrote it)
        pace = self.state.get("pace") or {}
        self.pace_known = bool(pace)
        self.min_interval = safe(pace.get("min_interval"), 6.0)
        self.jitter = safe(pace.get("jitter"), 4.0)
        self.replays_per_day = int(
            safe(pace.get("max_tables_per_day"), ASSUMED_REPLAYS_PER_DAY)
        )
        self.requests_per_day = int(safe(pace.get("max_requests_per_day"), 600))

        # ---- the error that stopped the last run
        error = self.state.get("error") or {}
        self.error_kind = str(error.get("kind") or "")
        self.error_message = str(error.get("message") or "")
        self.error_at = str(error.get("at") or "")

        # ---- the dataset
        self.validated = self.stats.get("games")
        self.positions = self.stats.get("positions")
        self.elite_positions = self.stats.get("elite_positions")
        self.all_positions = self.stats.get("all_positions")
        self.failed_validation = int(safe(self.stats.get("games_rejected"), 0))
        self.below_floor = int(safe(self.stats.get("skipped_below_elo"), 0))
        self.no_target = int(safe(self.stats.get("skipped_no_target"), 0))
        self.target_outcomes = self.stats.get("target_outcomes") or {}
        self.elo_floor_raw = safe(self.stats.get("elo_floor_raw"), 0.0)
        self.opponent_rows = str(self.stats.get("opponent_rows") or "")
        detailed = self.stats.get("reasons") or {}
        for reason, count in detailed.items():
            key = bucket_for(reason)
            self.reasons[key] = self.reasons.get(key, 0) + int(safe(count, 0))
        if self.failed_validation and not detailed:
            # a sidecar without the per-reason breakdown still has to show up
            self.reasons["replay failed validation"] = (
                self.reasons.get("replay failed validation", 0) + self.failed_validation
            )
        if self.no_target:
            self.reasons["no elite player identified"] = (
                self.reasons.get("no elite player identified", 0) + self.no_target
            )
        if self.below_floor:
            self.reasons["below Elo floor"] = (
                self.reasons.get("below Elo floor", 0) + self.below_floor
            )
        if self.positions is None:
            rows = npz_rows(self.npz_path)
            if rows is not None:
                self.positions = rows

        # ---- rate, measured from the log tail
        self.rate_per_hour = self._rate()
        self.last_event = self.events[-1] if self.events else {}
        self.last_ts = safe(self.last_event.get("ts"), 0.0) or None

    # ------------------------------------------------------------------ rate
    def _rate(self, window: int = 60) -> float | None:
        """Replays per hour over the last ``window`` downloads, or ``None``."""
        stamps = [
            safe(event.get("ts"), 0.0)
            for event in self.events
            if str(event.get("status")) == "downloaded"
        ]
        stamps = [s for s in stamps if s > 0][-window:]
        if len(stamps) < 2:
            return None
        span = stamps[-1] - stamps[0]
        if span <= 0:
            return None
        return (len(stamps) - 1) / span * 3600.0

    # ------------------------------------------------------------------ derived
    @property
    def has_state(self) -> bool:
        return bool(self.state)

    @property
    def elite_share(self) -> float | None:
        if not self.all_positions or self.elite_positions is None:
            return None
        return safe(self.elite_positions) / safe(self.all_positions, 1.0)

    def positions_estimate(self) -> int:
        """Positions we have, or a projection from the games fetched so far."""
        if self.positions:
            return int(self.positions)
        if not self.downloaded:
            return 0
        # docs §5.3: ~55 decision points a game, of which the elite player's turns
        # are about half. Only used before a dataset has ever been built.
        return int(self.downloaded * 27.5)

    def eta_hours(self, games: int) -> float | None:
        """Hours of *fetching* left to reach ``games``, at the measured rate."""
        remaining = games - self.downloaded
        if remaining <= 0:
            return 0.0
        if not self.rate_per_hour:
            return None
        return remaining / self.rate_per_hour

    def eta_days(self, games: int) -> float | None:
        """Days left to reach ``games``, limited by the daily replay cap."""
        remaining = games - self.downloaded
        if remaining <= 0:
            return 0.0
        cap = self.replays_per_day
        if cap <= 0:
            return None
        return remaining / float(cap)


# ==========================================================================
# Formatting
# ==========================================================================
def esc(text: Any) -> str:
    return html.escape("" if text is None else str(text), quote=True)


def dash(value: Any, suffix: str = "") -> str:
    if value is None or value == "":
        return "—"
    return f"{value}{suffix}"


def fmt_int(value: Any) -> str:
    if value is None:
        return "—"
    return f"{int(safe(value)):,}"


def fmt_compact(value: Any) -> str:
    if value is None:
        return "—"
    number = safe(value)
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if abs(number) >= 10_000:
        return f"{number / 1000:.0f}k"
    if abs(number) >= 1000:
        return f"{number / 1000:.1f}k"
    return f"{number:,.0f}"


def fmt_hours(value: float | None) -> str:
    if value is None:
        return "—"
    hours = safe(value)
    if hours <= 0:
        return "reached"
    if hours < 1:
        return f"{hours * 60:.0f} min"
    if hours < 48:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} d"


def fmt_days(value: float | None) -> str:
    if value is None:
        return "—"
    days = safe(value)
    if days <= 0:
        return "reached"
    if days < 1:
        return "today"
    return f"{math.ceil(days):d} d"


def fmt_ago(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0.0, safe(seconds))
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h ago"
    return f"{seconds / 86400:.1f} d ago"


def clock(ts: Any) -> str:
    value = safe(ts, 0.0)
    if value <= 0:
        return "—"
    return datetime.fromtimestamp(value).strftime("%H:%M:%S")  # noqa: DTZ006


# ==========================================================================
# HTML pieces
# ==========================================================================
def tile(label: str, value: str, sub: str = "", hero: bool = False) -> str:
    cls = "tile tile-hero" if hero else "tile"
    sub_html = f'<div class="tile-sub">{esc(sub)}</div>' if sub else ""
    return (
        f'<div class="{cls}"><div class="tile-label">{esc(label)}</div>'
        f'<div class="tile-value">{esc(value)}</div>{sub_html}</div>'
    )


def bar(
    label: str,
    value: float | None,
    target: float,
    note: str = "",
    color: str = SERIES[0],
) -> str:
    """One progress bar. Always a sane width: clamped to 0-100%, never ``nan``."""
    have = safe(value, 0.0)
    goal = safe(target, 0.0)
    frac = 0.0 if goal <= 0 else max(0.0, min(1.0, have / goal))
    percent = frac * 100.0
    head = (
        f"{'—' if value is None else fmt_compact(have)} / "
        f"{'—' if goal <= 0 else fmt_compact(goal)}"
    )
    if not note:
        note = "no target yet" if goal <= 0 else f"{percent:.1f}% of target"
    return (
        '<div class="bar-row">'
        f'<div class="bar-head"><span class="bar-label">{esc(label)}</span>'
        f'<span class="bar-value">{esc(head)}</span></div>'
        '<div class="bar-track" role="img" '
        f'aria-label="{esc(label)}: {percent:.1f} percent of {fmt_compact(goal)}">'
        f'<div class="bar-fill" style="width:{percent:.2f}%;background:{color}">'
        "</div></div>"
        f'<div class="bar-note">{esc(note)}</div>'
        "</div>"
    )


def table_html(headers: list[str], rows: list[list[str]], empty: str) -> str:
    if not rows:
        return f'<p class="empty-note">{esc(empty)}</p>'
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return (
        '<div class="tbl-wrap"><table class="tbl">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def section(title: str, eyebrow: str, body: str) -> str:
    return (
        f'<section class="panel"><header class="panel-head">'
        f'<span class="eyebrow">{esc(eyebrow)}</span><h2>{esc(title)}</h2>'
        f"</header>{body}</section>"
    )


def error_banner(h: Harvest) -> str:
    """BGA's stop conditions, shown loudly — the replay quota above all."""
    if not h.error_kind:
        return ""
    known = {
        "replay-limit": (
            "BGA's daily replay quota is reached",
            (
                "This is the expected way a day ends: BGA caps how many archived "
                "games one account may open per day. The crawl stopped on it rather "
                "than hammering past it, nothing is lost, and re-running in ~24 h "
                "resumes at the exact table it stopped on."
            ),
            "crit",
        ),
        "account-disabled": (
            "BGA has disabled replay access for this account",
            "Stop here. Do not retry, and talk to BGA before doing anything else.",
            "crit",
        ),
        "auth": (
            "BGA rejected the session",
            (
                "The exported cookies are stale or incomplete. Re-export "
                "boardgamearena.com cookies from a logged-in browser."
            ),
            "crit",
        ),
        "daily-table-cap": (
            "our own daily replay cap is spent",
            (
                "This is our conservative self-limit, not BGA's. The crawl picks up "
                "again after UTC midnight."
            ),
            "warn",
        ),
        "daily-request-cap": (
            "our own daily request cap is spent",
            "Same as above: our limit, not BGA's.",
            "warn",
        ),
    }
    title, explain, level = known.get(
        h.error_kind,
        (
            f"the crawl stopped: {h.error_kind}",
            (
                "An unfamiliar stop reason — the message below is verbatim from "
                "the state file."
            ),
            "warn",
        ),
    )
    detail = (
        f'<p class="banner-detail"><code>{esc(h.error_message)}</code></p>'
        if h.error_message
        else ""
    )
    when = (
        f'<span class="banner-when">recorded {esc(h.error_at)}</span>'
        if h.error_at
        else ""
    )
    return (
        f'<section class="banner banner-{level}">'
        f'<div class="banner-head"><span class="banner-kind">{esc(h.error_kind)}</span>'
        f"{when}</div>"
        f"<h2>{esc(title)}</h2><p>{esc(explain)}</p>{detail}</section>"
    )


def position_panel(h: Harvest) -> str:
    """Where the rank-ordered crawl is right now."""
    rank_sub = ""
    if h.ladder_size:
        rank_sub = f"of {fmt_int(h.ladder_size)} in the ladder snapshot"
    player_value = h.player_name or (str(h.cursor_player) if h.cursor_player else "—")
    elo_sub = ""
    if h.player_elo is not None:
        elo_sub = f"displayed Elo {fmt_int(h.player_elo)}"
        if h.player_elo_raw is not None:
            elo_sub += f" (raw {safe(h.player_elo_raw):.0f})"
    tiles = [
        tile(
            "Rank being crawled",
            f"#{h.rank}" if h.rank else "—",
            sub=rank_sub,
            hero=True,
        ),
        tile("Player", player_value, sub=elo_sub),
        tile(
            "Tables done for this player",
            fmt_int(h.table_offset),
            sub=(
                f"{fmt_int(h.player_games)} ranked games on record"
                if h.player_games is not None
                else ""
            ),
        ),
        tile("Players finished", fmt_int(h.players_done)),
        tile(
            "Replays today",
            f"{fmt_int(h.downloads_today)} / {fmt_int(h.replays_per_day)}",
            sub=(
                "pace from state.json" if h.pace_known else "cap assumed, not yet run"
            ),
        ),
        tile(
            "Requests today",
            f"{fmt_int(h.requests_today)} / {fmt_int(h.requests_per_day)}",
            sub=f"{fmt_int(h.requests_total)} in total",
        ),
    ]
    freshness = (
        f'<p class="run-note">last fetch {esc(fmt_ago(h.now - h.last_ts))}'
        f"{esc(', cursor written ' + h.cursor_updated) if h.cursor_updated else ''}</p>"
        if h.last_ts
        else ""
    )
    return section(
        "Ladder position",
        "now",
        f'{freshness}<div class="tiles">{"".join(tiles)}</div>',
    )


def collected_panel(h: Harvest) -> str:
    rejected = h.skipped + h.errored + h.failed_validation + h.below_floor + h.no_target
    tiles = [
        tile(
            "Games fetched", fmt_int(h.downloaded), sub=f"{fmt_int(h.decided)} decided"
        ),
        tile(
            "Passed engine validation",
            fmt_int(h.validated),
            sub=(
                "replayed move for move in our engine"
                if h.validated is not None
                else "run `cli dataset` to find out"
            ),
        ),
        tile("Rejected", fmt_int(rejected), sub="fetch filters + replay validation"),
        tile(
            "Target record",
            (
                f"{fmt_int(h.target_outcomes.get('win'))}W"
                f"/{fmt_int(h.target_outcomes.get('loss'))}L"
                f"/{fmt_int(h.target_outcomes.get('draw'))}D"
                if h.target_outcomes
                else "—"
            ),
            sub="the elite player's results, not seat 0's",
        ),
    ]
    rows = [
        [esc(name), f'<span class="mono">{fmt_int(count)}</span>']
        for name, count in sorted(h.reasons.items(), key=lambda kv: -kv[1])
        if count
    ]
    tally = table_html(
        ["rejection reason", "tables"],
        rows,
        "Nothing rejected yet — the tally fills in as the crawl runs.",
    )
    return section(
        "Collected",
        "yield",
        f'<div class="tiles">{"".join(tiles)}</div>{tally}',
    )


def progress_panel(h: Harvest) -> str:
    positions = h.positions
    projected = h.positions_estimate()
    note = (
        f"{fmt_compact(projected)} projected from the games fetched so far"
        if projected
        else "nothing fetched yet"
    )
    bars = [
        bar(
            "Positions toward the full target",
            positions if positions is not None else projected,
            POSITION_TARGET,
            note=(
                f"{safe(positions) / POSITION_TARGET * 100:.2f}% of {fmt_compact(POSITION_TARGET)}"
                if positions
                else note
            ),
        ),
        bar(
            f"Games toward the {fmt_compact(GAME_MILESTONES[0])}-game milestone",
            h.downloaded,
            GAME_MILESTONES[0],
            color=SERIES[2],
        ),
        bar(
            f"Games toward the {fmt_compact(GAME_MILESTONES[1])}-game target",
            h.downloaded,
            GAME_MILESTONES[1],
            color=SERIES[3],
        ),
    ]
    share = h.elite_share
    if share is None:
        elite_bar = bar(
            "Elite-only positions vs all decision points",
            h.elite_positions,
            safe(h.all_positions, 0.0),
            note="no dataset built yet — only the elite player's turns become rows",
            color=SERIES[4],
        )
    else:
        elite_bar = bar(
            "Elite-only positions vs all decision points",
            h.elite_positions,
            h.all_positions,
            note=(
                f"{share * 100:.1f}% of replayed decisions are the target player's; "
                "the rest are validated but never trained on"
            ),
            color=SERIES[4],
        )
    floor = ""
    if h.elo_floor_raw:
        floor = (
            f'<p class="run-note">target Elo floor: raw {h.elo_floor_raw:.0f} '
            f"(displayed {max(0, int(h.elo_floor_raw - 1300))}); opponent rows: "
            f"{esc(h.opponent_rows or 'dropped')}</p>"
        )
    return section(
        "Progress",
        "targets",
        f'<div class="bars">{"".join(bars)}{elite_bar}</div>{floor}',
    )


def rate_panel(h: Harvest) -> str:
    tiles = [
        tile(
            "Fetch rate",
            (f"{safe(h.rate_per_hour):.1f}/h" if h.rate_per_hour is not None else "—"),
            sub=(
                f"pace allows one request every {h.min_interval:.0f}-"
                f"{h.min_interval + h.jitter:.0f}s"
            ),
        ),
        tile(
            f"{fmt_compact(GAME_MILESTONES[0])} games",
            fmt_days(h.eta_days(GAME_MILESTONES[0])),
            sub=f"{fmt_hours(h.eta_hours(GAME_MILESTONES[0]))} of fetching",
        ),
        tile(
            f"{fmt_compact(GAME_MILESTONES[1])} games",
            fmt_days(h.eta_days(GAME_MILESTONES[1])),
            sub=f"{fmt_hours(h.eta_hours(GAME_MILESTONES[1]))} of fetching",
        ),
        tile(
            "Days assume",
            f"{fmt_int(h.replays_per_day)}/day",
            sub="our own cap; BGA's quota may be lower and stops the day early",
        ),
    ]
    return section(
        "Rate and time to milestones",
        "arithmetic",
        f'<div class="tiles">{"".join(tiles)}</div>',
    )


def log_panel(h: Harvest) -> str:
    rows = []
    for event in reversed(h.events[-LOG_TAIL:]):
        status = str(event.get("status", ""))
        cls = {
            "downloaded": "ok",
            "skipped": "warn",
            "error": "crit",
            "stopped": "crit",
        }.get(status, "")
        target_elo = event.get("target_elo")
        rows.append(
            [
                f'<span class="mono">{esc(clock(event.get("ts")))}</span>',
                esc(dash(event.get("rank"), "") if event.get("rank") else ""),
                esc(dash(event.get("player"))),
                f'<span class="mono">{esc(dash(event.get("table")))}</span>',
                f'<span class="tag tag-{cls}">{esc(status or "?")}</span>',
                esc(
                    event.get("reason")
                    or event.get("message")
                    or (
                        f"target {event.get('target')}"
                        + (
                            f" ({safe(target_elo):.0f} raw)"
                            if target_elo is not None
                            else ""
                        )
                    )
                ),
            ]
        )
    return section(
        "Recent fetches",
        "log tail",
        table_html(
            ["time", "rank", "player", "table", "status", "detail"],
            rows,
            "No fetches logged yet. The crawl appends one line per table to "
            "fetch.jsonl and this tail follows it.",
        ),
    )


def empty_hint(h: Harvest) -> str:
    if h.has_state:
        return ""
    return (
        '<section class="banner banner-idle"><h2>Nothing collected yet</h2>'
        f"<p>No state file at <code>{esc(h.state_path)}</code>. This page is live as "
        "soon as the crawl writes one:</p>"
        "<pre>python -m ludometer.human.cli crawl \\\n"
        "  --out data/human --cookies ~/ludometer/.bga_cookies.txt \\\n"
        "  --top 200 --min-elo 650 --min-games 200</pre>"
        "<p>Every tile below is showing its empty state, not a zero measurement.</p>"
        "</section>"
    )


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
  padding: 0 18px 64px;
}
.wrap { max-width: 1080px; margin: 0 auto; }
code, .mono { font-family: var(--mono); font-size: 12.5px; }

.masthead { padding: 34px 0 20px; border-bottom: 1px solid var(--hairline); }
.brand { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; }
.brand h1 { font-size: 26px; margin: 0; font-weight: 600; letter-spacing: -0.015em; }
.thesis { font-family: var(--serif); font-style: italic; color: var(--ink-2); font-size: 16px; }
.masthead-meta {
  margin: 12px 0 0; display: flex; gap: 18px; flex-wrap: wrap;
  font-family: var(--mono); font-size: 11px; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--muted);
}
.eyebrow {
  font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--muted);
}

.panel { padding: 26px 0 6px; border-bottom: 1px solid var(--hairline); }
.panel-head { display: flex; flex-direction: column; gap: 2px; margin-bottom: 16px; }
.panel-head h2 { margin: 0; font-size: 20px; font-weight: 600; letter-spacing: -0.01em; }
.run-note { color: var(--ink-2); font-size: 13.5px; margin: 0 0 14px; }

.tiles {
  display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(158px, 1fr));
}
.tile {
  background: var(--surface); border: 1px solid var(--hairline);
  border-radius: var(--r); padding: 13px 15px 15px;
}
.tile-hero { background: var(--surface-2); }
.tile-label {
  font-family: var(--mono); font-size: 10px; letter-spacing: 0.13em;
  text-transform: uppercase; color: var(--muted);
}
.tile-value {
  font-size: 27px; font-weight: 600; letter-spacing: -0.02em; margin-top: 5px;
  overflow-wrap: anywhere;
}
.tile-hero .tile-value { font-size: 36px; color: var(--accent); }
.tile-sub { color: var(--ink-2); font-size: 12.5px; margin-top: 4px; }

.bars { display: grid; gap: 18px; }
.bar-head { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; }
.bar-label { font-size: 14px; }
.bar-value { font-family: var(--mono); font-size: 12.5px; color: var(--ink-2); }
.bar-track {
  margin-top: 7px; height: 12px; border-radius: 999px; overflow: hidden;
  background: var(--grid); border: 1px solid var(--hairline);
}
.bar-fill { height: 100%; border-radius: 999px; min-width: 0; }
.bar-note { color: var(--muted); font-size: 12px; margin-top: 5px; }

.tbl-wrap { overflow-x: auto; margin-top: 16px; }
.tbl { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.tbl th {
  text-align: left; font-family: var(--mono); font-size: 10px; font-weight: 500;
  letter-spacing: 0.13em; text-transform: uppercase; color: var(--muted);
  padding: 0 12px 7px 0; border-bottom: 1px solid var(--grid); white-space: nowrap;
}
.tbl td {
  padding: 6px 12px 6px 0; border-bottom: 1px solid var(--grid);
  color: var(--ink-2); vertical-align: top;
}
.tbl td:first-child, .tbl th:first-child { white-space: nowrap; }
.tag {
  display: inline-block; padding: 1px 8px; border-radius: 999px;
  font-family: var(--mono); font-size: 10.5px; text-transform: uppercase;
  letter-spacing: 0.08em; border: 1px solid var(--hairline); color: var(--ink-2);
}
.tag-ok { color: var(--good); border-color: color-mix(in srgb, var(--good) 45%, transparent); }
.tag-warn { color: var(--warn); border-color: color-mix(in srgb, var(--warn) 45%, transparent); }
.tag-crit { color: var(--crit); border-color: color-mix(in srgb, var(--crit) 45%, transparent); }
.empty-note { color: var(--muted); font-size: 13.5px; margin: 14px 0 0; }

.banner {
  margin: 22px 0 0; padding: 18px 20px; border-radius: var(--r);
  background: var(--surface); border: 1px solid var(--hairline);
}
.banner h2 { margin: 6px 0 8px; font-size: 20px; }
.banner p { margin: 0 0 8px; color: var(--ink-2); font-size: 14px; }
.banner pre {
  font-family: var(--mono); font-size: 12.5px; background: var(--plane);
  border: 1px solid var(--hairline); border-radius: 7px; padding: 11px 13px;
  overflow-x: auto; color: var(--ink);
}
.banner-crit { border-color: color-mix(in srgb, var(--crit) 60%, transparent); background: color-mix(in srgb, var(--crit) 10%, var(--surface)); }
.banner-warn { border-color: color-mix(in srgb, var(--warn) 55%, transparent); background: color-mix(in srgb, var(--warn) 8%, var(--surface)); }
.banner-idle { border-style: dashed; }
.banner-head {
  display: flex; gap: 14px; flex-wrap: wrap; align-items: baseline;
  font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.13em;
  text-transform: uppercase; color: var(--muted);
}
.banner-kind { color: var(--crit); }
.banner-detail code { color: var(--ink-2); overflow-wrap: anywhere; }

.foot {
  margin-top: 38px; padding-top: 16px; border-top: 1px solid var(--hairline);
  font-family: var(--mono); font-size: 11px; color: var(--muted);
  display: flex; gap: 18px; flex-wrap: wrap;
}

@media (max-width: 560px) {
  body { padding: 0 13px 48px; font-size: 14.5px; }
  .tiles { grid-template-columns: repeat(auto-fit, minmax(132px, 1fr)); }
  .tile-value { font-size: 23px; }
  .tile-hero .tile-value { font-size: 30px; }
  .brand h1 { font-size: 22px; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
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


def build_page(h: Harvest) -> str:
    generated = datetime.fromtimestamp(h.now).strftime("%Y-%m-%d %H:%M:%S")  # noqa: DTZ006
    meta = [
        f"generated {generated}",
        f"auto-refresh {REFRESH_SECONDS}s",
        f"state {h.state_path.name if h.has_state else 'missing'}",
        f"{fmt_int(h.downloaded)} games fetched",
    ]
    if h.error_kind:
        meta.append(f"stopped: {h.error_kind}")
    body = [
        '<div class="wrap">',
        '<header class="masthead"><div class="brand">',
        "<h1>Harvest</h1>",
        '<p class="thesis">Human games, one elite player at a time.</p></div>',
        f'<div class="masthead-meta">{"".join(f"<span>{esc(m)}</span>" for m in meta)}</div>',
        "</header>",
        error_banner(h),
        empty_hint(h),
        position_panel(h),
        collected_panel(h),
        progress_panel(h),
        rate_panel(h),
        log_panel(h),
        (
            '<footer class="foot">'
            "<span>ludometer · generated by web/make_harvest.py</span>"
            "<span>static page, no external requests</span>"
            f"<span>{esc(generated)}</span></footer>"
        ),
        "</div>",
    ]
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8" />\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}" />\n'
        "<title>Ludometer Harvest Monitor</title>\n"
        f"<style>{styles()}</style>\n"
        "</head><body>\n" + "".join(body) + "\n</body></html>\n"
    )


# ==========================================================================
# Entry point
# ==========================================================================
def write_atomic(path: Path, page: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(path)  # atomic: a refreshing tab never sees a half file
    return len(page)


def generate(
    state_dir: Path, out_path: Path, npz: Path | None = None, now: float = 0.0
) -> tuple[Harvest, int]:
    harvest = Harvest(state_dir, npz=npz, now=now or time.time())
    return harvest, write_atomic(out_path, build_page(harvest))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the Ludometer harvest (BGA collection) monitor."
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=REPO / "data" / "human",
        help="fetcher working directory (or a state.json path)",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="dataset path (default: replay.npz beside the state file)",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO / "web" / "harvest.html", help="output HTML"
    )
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
        harvest, size = generate(args.state, args.out, args.npz)
        if not args.quiet:
            state = (
                "no state file"
                if not harvest.has_state
                else (f"rank {harvest.rank or '—'}, {harvest.downloaded} games")
            )
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] wrote {args.out} "  # noqa: DTZ005
                f"({state}, {size / 1024:.0f} kB)"
            )

    if args.watch is None:
        once()
        return 0
    interval = max(1.0, float(args.watch))
    if not args.quiet:
        print(f"watching {args.state} every {interval:g}s — Ctrl-C to stop")
    try:
        while True:
            try:
                once()
            except Exception as exc:  # noqa: BLE001 - keep the watcher alive
                print(f"error: {exc}", file=sys.stderr)
            time.sleep(interval)
    except KeyboardInterrupt:
        if not args.quiet:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
