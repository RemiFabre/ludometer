"""Resumable, rate-limited fetcher for BGA Azul games.

Three stages, each one restartable and each one recorded in a single JSON state
file so that a run killed halfway costs nothing:

1. :meth:`Fetcher.fetch_ranking` — the **all-time** ladder (public, no cookies).
2. :meth:`Fetcher.fetch_player_tables` — one player's finished Azul tables
   (needs a session).
3. :meth:`Fetcher.fetch_table` — one table's metadata + move log (needs a
   session). This is the only stage whose volume is large, and it is the one the
   filters below exist to keep small.

:meth:`Fetcher.crawl_ranked` drives 2 and 3 **in ladder order**: rank 1's games
first, then rank 2, and so on, with a cursor in the state file
(:class:`CrawlCursor`) that names the rank, the player and the offset inside that
player's table list, so a restart continues at the exact table it stopped on. The
pace lives in :class:`CrawlPace` and is deliberately slow (§ "Pace" below).

The **state file** (``<out>/state.json``, see :class:`FetchState`) is the resume
point *and* the audit trail: it records every table we have decided about,
including the ones we deliberately skipped and why, so a rerun never re-requests
a game it already rejected. Raw payloads are written next to it, gzipped, one
file per table, and are never rewritten once present.

Nothing here is concurrent and nothing here is fast; see
:class:`~ludometer.human.client.ClientConfig` for the budget.
"""

from __future__ import annotations

import datetime as _dt
import json
import random
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ludometer.human.client import (
    AZUL_GAME_ID,
    AccountDisabled,
    AuthRequired,
    BgaClient,
    BgaError,
    ReplayLimitReached,
    ReplayUnavailable,
    display_elo,
    endpoints,
    read_json_gz,
    write_json_gz,
)

__all__ = [
    "ARENA_MODE",
    "GAME_MODE_OPTION",
    "MAX_REPLAY_FETCHES_PER_DAY",
    "MAX_REQUESTS_PER_DAY",
    "REPLAY_JITTER",
    "REPLAY_MIN_INTERVAL",
    "STANDARD_WALL_OPTION_HINTS",
    "CrawlCursor",
    "CrawlPace",
    "CrawlReport",
    "FetchState",
    "Fetcher",
    "PlayerRow",
    "TableFilter",
    "TableVerdict",
    "choose_target_player",
    "extract_table_ids",
    "extract_table_rows",
    "option_value",
    "player_elos",
    "select_players",
    "table_row_players",
    "table_row_scores",
]

#: Version 2 adds ``cursor`` (the rank-ordered crawl's resume point),
#: ``downloads`` (the per-day replay counter the quota is spent on) and ``error``
#: (why the last run stopped). A version-1 file loads and is upgraded in place —
#: the three new sections simply start empty.
STATE_VERSION = 2
SUPPORTED_STATE_VERSIONS = (1, 2)

# ---------------------------------------------------------------------- pace
# The binding constraint is BGA's **undocumented per-account daily replay quota**
# (docs/HUMAN_GAMES.md §5.1), which arrives as a 200 whose JSON says "You have
# reached a limit (replay)". Since we neither know the number nor rotate accounts
# to dodge it, the defaults below are chosen to sit *well* under any plausible
# value and to look like someone reviewing their own games rather than a crawler:
#
# * one replay fetched every 6-10 s (``REPLAY_MIN_INTERVAL`` + up to
#   ``REPLAY_JITTER``, uniform) — a table costs 2-3 requests, so ~20-30 s per
#   game, ~2 games/minute at the very most;
# * ``MAX_REPLAY_FETCHES_PER_DAY`` tables per day, i.e. ~50 minutes of traffic,
#   and a hard stop after that even if the process keeps running;
# * ``MAX_REQUESTS_PER_DAY`` for *all* requests (histories included);
# * a randomized long pause every :data:`LONG_PAUSE_EVERY` tables, so the traffic
#   has gaps in it instead of being a metronome for an hour.
#
# Raise them only with a measured quota in hand.
REPLAY_MIN_INTERVAL = 6.0
REPLAY_JITTER = 4.0
MAX_REPLAY_FETCHES_PER_DAY = 120
MAX_REQUESTS_PER_DAY = 600
LONG_PAUSE_EVERY = 20
LONG_PAUSE_MIN = 90.0
LONG_PAUSE_MAX = 300.0

#: Where Azul's wall variant lives in `tableinfos.options` — the filter Remi asked
#: for ("the gray board where wall placement is free must be excluded").
#:
#: **Verified live 2026-08-17** against several of rank-1's finished 2-player tables
#: (`/table/table/tableinfos.html?id=<id>`). The option is id **100, name "Board"**,
#: an enum whose values are::
#:
#:     1 = "Colored side"          <- the STANDARD fixed colour wall (default)
#:     2 = "Gray side"             <- the grey/variable wall (tile goes in any column)
#:     3 = "Crystal Mozaic: Side 1"  <- a different board entirely
#:     4 = "Crystal Mozaic: Side 2"  <- a different board entirely
#:
#: so ONLY value ``1`` is the board our engine models; ``2``/``3``/``4`` are all
#: rejected. This matches (and refines) the old `media.majorvariant` hypothesis:
#: there are four board sides, not two, and the standard one is value 1.
#:
#: A **separate** variant to be aware of is option **110, "Special Factories (Azul
#: Master Chocolatier variant)"** (1 = Disabled = standard, 2 = Enabled): it adds
#: special factories and is not something our engine implements. The wall filter
#: here does not cover it; a Special-Factories game would instead be caught
#: downstream by the tile-census / score checks in `convert_game` (docs §3.1), but
#: see `SPECIAL_FACTORIES_OPTION` below if a request-saving pre-filter is wanted.
#:
#: The wall-column identity in `convert.check_wall_placements` (docs §3.1) remains
#: the real guarantee — this option filter only saves the `logs` request on a
#: variant table. A grey-wall game cannot reach the dataset regardless.
STANDARD_WALL_OPTION_HINTS = {
    "option_id": 100,  # verified 2026-08-17: option "Board"
    "standard_values": (1,),  # 1 = "Colored side" = standard fixed colour wall
    "variant_name_patterns": (
        "variable",
        "variant",
        "grey",
        "gray",
        "free",
        "crystal",
        "mozaic",
    ),
}

#: BGA's framework-wide option ids, the same for every game: 200 = game speed,
#: **201 = game mode** (0 normal, 1 friendly/training, 2 Arena), 204 = thinking
#: time. Game-*specific* options start at 100, which is where Azul's wall variant
#: lives. Confirmed across several community projects.
GAME_SPEED_OPTION = 200
GAME_MODE_OPTION = 201
THINKING_TIME_OPTION = 204
#: `option_value(options, GAME_MODE_OPTION) == ARENA_MODE` is "this was a ranked
#: Arena game", the strongest available "both players were trying" signal.
ARENA_MODE = 2
#: Azul-specific option 110 ("Special Factories", the Master Chocolatier variant):
#: 1 = Disabled (standard), 2 = Enabled. Not modelled by our engine. Verified live
#: 2026-08-17 alongside the wall option; `TableFilter` does not yet gate on it (the
#: converter's census/score checks catch it), but it is named here for a future
#: request-saving pre-filter.
SPECIAL_FACTORIES_OPTION = 110
SPECIAL_FACTORIES_DISABLED = 1


def option_value(options: dict[str, Any], option_id: int) -> int | None:
    """One game option's integer value, whichever shape ``tableinfos`` used.

    BGA's ``options`` map is documented as ``{id: {"name": ..., "value": ...}}`` but
    older payloads (and the lobby's own preference strings) use ``{id: value}``.
    Both are read here; anything unparseable returns ``None``, which every caller
    treats as "unknown", never as "fine".
    """
    raw = options.get(str(option_id), options.get(option_id))
    if isinstance(raw, dict):
        raw = raw.get("value", raw.get("val"))
    if raw is None or isinstance(raw, (dict, list)):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class PlayerRow:
    """One row of the all-time ladder, normalised."""

    player_id: int
    name: str
    elo_raw: float
    elo_display: int
    rank: int
    games_played: int

    @classmethod
    def from_api(cls, row: dict[str, Any]) -> PlayerRow:
        return cls(
            player_id=int(row["id"]),
            name=str(row.get("name", "")),
            elo_raw=float(row["ranking"]),
            elo_display=display_elo(row["ranking"]),
            rank=int(row.get("rank_no") or 0),
            games_played=int(row.get("nbr_game") or 0),
        )


@dataclass(frozen=True)
class CrawlPace:
    """How fast :meth:`Fetcher.crawl_ranked` may go. Deliberately slow.

    The defaults are the module constants above: one replay every 6-10 s,
    ``max_tables_per_day`` replays a day, ``max_requests_per_day`` requests a day,
    and a randomized long pause every ``long_pause_every`` tables. They are data,
    not literals in the loop, so the CLI can print them and a future session can
    raise them once the real quota is measured — not before.
    """

    min_interval: float = REPLAY_MIN_INTERVAL
    jitter: float = REPLAY_JITTER
    max_requests_per_day: int = MAX_REQUESTS_PER_DAY
    max_tables_per_day: int = MAX_REPLAY_FETCHES_PER_DAY
    long_pause_every: int = LONG_PAUSE_EVERY
    long_pause_min: float = LONG_PAUSE_MIN
    long_pause_max: float = LONG_PAUSE_MAX

    def seconds_per_table(self, requests_per_table: float = 3.0) -> float:
        """Wall-clock seconds one accepted table costs at this pace, on average."""
        gap = self.min_interval + self.jitter / 2.0
        extra = 0.0
        if self.long_pause_every > 0:
            extra = (
                (self.long_pause_min + self.long_pause_max)
                / 2.0
                / (self.long_pause_every)
            )
        return gap * float(requests_per_table) + extra

    def describe(self) -> str:
        return (
            f"one request every {self.min_interval:.0f}-"
            f"{self.min_interval + self.jitter:.0f}s, "
            f"<={self.max_tables_per_day} replays/day, "
            f"<={self.max_requests_per_day} requests/day, "
            f"a {self.long_pause_min:.0f}-{self.long_pause_max:.0f}s pause every "
            f"{self.long_pause_every} tables"
        )


@dataclass(frozen=True)
class CrawlCursor:
    """Where the rank-ordered crawl is, precisely enough to resume on it.

    ``rank`` is the ladder rank **currently being processed** (1 = the all-time
    number one, ``0`` = nothing started yet), ``player_id`` is that rank's player,
    and ``table_offset`` is how many of that player's tables we have already
    decided about — so a restart re-enters at ``tables[table_offset]`` of rank
    ``rank`` and never re-requests a table it already judged. ``players_done``
    counts fully processed players, for the progress page.
    """

    rank: int = 0
    player_id: int = 0
    table_offset: int = 0
    players_done: int = 0
    updated: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, Any] | None) -> CrawlCursor:
        row = row or {}
        return cls(
            rank=int(row.get("rank") or 0),
            player_id=int(row.get("player_id") or 0),
            table_offset=int(row.get("table_offset") or 0),
            players_done=int(row.get("players_done") or 0),
            updated=str(row.get("updated") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "player_id": self.player_id,
            "table_offset": self.table_offset,
            "players_done": self.players_done,
            "updated": self.updated
            or _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        }


@dataclass
class CrawlReport:
    """What one :meth:`Fetcher.crawl_ranked` call did — printed by the CLI."""

    downloaded: int = 0
    skipped: int = 0
    errors: int = 0
    #: tables whose verdict was already terminal in the state file — revisited for
    #: free (no request, no pause) by a ``resume=False`` re-walk of the ladder.
    cached: int = 0
    players: int = 0
    #: ``""`` when the crawl ran out of work; otherwise why it stopped:
    #: ``"replay-limit"`` (BGA's daily quota), ``"account-disabled"``,
    #: ``"auth"``, ``"daily-table-cap"``, ``"daily-request-cap"`` or ``"limit"``.
    stopped: str = ""
    message: str = ""
    cursor: CrawlCursor = field(default_factory=CrawlCursor)


@dataclass(frozen=True)
class TableVerdict:
    """Why a table was kept or dropped — stored verbatim in the state file."""

    table_id: int
    status: str  # "downloaded" | "skipped" | "error"
    reason: str = ""

    @property
    def terminal(self) -> bool:
        """Never ask BGA about this table again."""
        return self.status in ("downloaded", "skipped")


@dataclass
class TableFilter:
    """The "is this game usable?" rules, applied to ``tableinfos`` metadata.

    Fail-closed by design: :meth:`check` returns a reason string (i.e. rejects)
    whenever it cannot *prove* a table is a standard-wall 2-player finished game.
    A dataset that silently absorbed a few hundred variable-wall games would be
    much more expensive to notice than one that is 5% smaller than it could be.
    """

    players: int = 2
    require_standard_wall: bool = True
    min_player_elo_raw: float = 0.0
    require_finished: bool = True
    #: When set, only keep tables whose game mode (option 201) is one of these.
    #: ``2`` is Arena — BGA's competitive ranked mode, and Azul's arena is
    #: 2-player. ``1`` is friendly/training and is exactly the sort of game a strong
    #: player does not try hard in, so it is worth excluding when the field is there.
    allowed_game_modes: tuple[int, ...] | None = None

    def check(self, infos: dict[str, Any]) -> str:
        """Return ``""`` to accept, or a short reason to skip."""
        data = infos.get("data", infos)
        if int(data.get("game_id", 0) or 0) not in (0, AZUL_GAME_ID):
            return f"wrong game {data.get('game_id')}"
        seats = data.get("players") or {}
        if self.players and len(seats) != self.players:
            return f"{len(seats)} players"
        status = str(data.get("status", ""))
        if self.require_finished and status not in ("finished", "archive"):
            return f"status {status!r}"
        if str(data.get("unranked") or "0") == "1":
            return "unranked table"
        if self.min_player_elo_raw:
            # NB: BGA's tableinfos does NOT carry per-seat Elo (confirmed in
            # docs/HUMAN_GAMES.md App. C). So a seat with no Elo field is not
            # evidence of a weak player and must not reject the table — the
            # target (elite) player's quality is guaranteed by ladder selection,
            # and the real Elo floor is applied at dataset-build time against the
            # known ladder Elo. Only reject when an Elo is actually present and
            # below the floor.
            for seat in seats.values():
                raw = seat.get("player_elo")
                if raw is None:
                    raw = seat.get("elo")
                if raw is None:
                    raw = seat.get("rank")
                if raw is None:
                    continue
                if float(raw or 0) < self.min_player_elo_raw:
                    return f"seat elo {raw} below floor"
        options = data.get("options") or {}
        if self.allowed_game_modes is not None:
            mode = option_value(options, GAME_MODE_OPTION)
            if mode is None:
                return "game mode option missing from tableinfos"
            if mode not in self.allowed_game_modes:
                return f"game mode {mode} not in {self.allowed_game_modes}"
        if self.require_standard_wall:
            verdict = self.wall_verdict(options)
            if verdict:
                return verdict
        return ""

    def wall_verdict(self, options: dict[str, Any]) -> str:
        """``""`` if the options prove a standard wall, else the reason to skip.

        The option id is not confirmed yet (see :data:`STANDARD_WALL_OPTION_HINTS`),
        so this currently rejects everything with an explicit, greppable reason
        rather than pretending to know. Filling the id in is one line, and step 4 of
        the plan in docs/HUMAN_GAMES.md §9.
        """
        option_id = STANDARD_WALL_OPTION_HINTS["option_id"]
        if option_id is None:
            return "wall variant option id unknown (see docs/HUMAN_GAMES.md)"
        value = option_value(options, int(option_id))
        if value is None:
            return "wall variant option missing from tableinfos"
        if value in STANDARD_WALL_OPTION_HINTS["standard_values"]:
            return ""
        return f"non-standard wall (option {option_id}={value})"


@dataclass
class FetchState:
    """The resume point. One JSON file, rewritten atomically after every step.

    Layout::

        {
          "version": 2,
          "game_id": 1467,
          "started": "2026-08-17T10:00:00",
          "requests": {"total": 812, "2026-08-17": 812},
          "downloads": {"total": 96, "2026-08-17": 96},
          "cursor": {"rank": 7, "player_id": 91843016, "table_offset": 34,
                     "players_done": 6, "updated": "2026-08-17T11:02:00"},
          "error": {"kind": "replay-limit", "message": "...", "at": "..."},
          "ranking": {"fetched": "2026-08-17T10:05:00",
                      "rows": [ {player_id, name, elo_raw, elo_display,
                                 rank, games_played}, ... ]},
          "players": {"91843016": {"pages_done": 3, "complete": true,
                                   "tables": [712345678, ...]}},
          "tables":  {"712345678": {"status": "downloaded", "reason": "",
                                    "rank": 1, "target": 91843016,
                                    "elos": {"91843016": 2486.2, ...}}}
        }

    Resuming = load it and skip. ``players[pid]["pages_done"]`` is the number of
    history pages already consumed, so a player is resumed mid-history;
    ``cursor`` is where the rank-ordered crawl was (rank, player, offset inside
    that player's list); a table with a terminal verdict is never requested again.
    ``downloads`` counts the requests that spend BGA's daily replay quota, and
    ``error`` records why the last run stopped — the harvest page reads both.
    """

    path: Path
    game_id: int = AZUL_GAME_ID
    version: int = STATE_VERSION
    started: str = ""
    requests: dict[str, int] = field(default_factory=dict)
    downloads: dict[str, int] = field(default_factory=dict)
    ranking: dict[str, Any] = field(default_factory=dict)
    players: dict[str, dict[str, Any]] = field(default_factory=dict)
    tables: dict[str, dict[str, Any]] = field(default_factory=dict)
    cursor: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] = field(default_factory=dict)
    #: the :class:`CrawlPace` the last run used, so the progress page can show it
    #: and turn "games still wanted" into days without guessing the caps
    pace: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, game_id: int = AZUL_GAME_ID) -> FetchState:
        path = Path(path)
        if not path.exists():
            return cls(
                path=path,
                game_id=game_id,
                started=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            )
        payload = json.loads(path.read_text())
        if int(payload.get("version", 0)) not in SUPPORTED_STATE_VERSIONS:
            raise ValueError(
                f"state file {path} has version {payload.get('version')}, "
                f"this code reads {SUPPORTED_STATE_VERSIONS} and writes "
                f"{STATE_VERSION}"
            )
        return cls(
            path=path,
            game_id=int(payload.get("game_id", game_id)),
            version=STATE_VERSION,
            started=payload.get("started", ""),
            requests=dict(payload.get("requests", {})),
            downloads=dict(payload.get("downloads", {})),
            ranking=dict(payload.get("ranking", {})),
            players=dict(payload.get("players", {})),
            tables=dict(payload.get("tables", {})),
            cursor=dict(payload.get("cursor", {})),
            error=dict(payload.get("error", {})),
            pace=dict(payload.get("pace", {})),
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": STATE_VERSION,
            "game_id": self.game_id,
            "started": self.started,
            "requests": self.requests,
            "downloads": self.downloads,
            "cursor": self.cursor,
            "error": self.error,
            "pace": self.pace,
            "ranking": self.ranking,
            "players": self.players,
            "tables": self.tables,
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(self.path)

    # ---------------------------------------------------------------- accessors
    def note_requests(self, total: int) -> None:
        today = _dt.datetime.now(_dt.UTC).date().isoformat()
        done_before = self.requests.get("total", 0)
        self.requests["total"] = total
        self.requests[today] = self.requests.get(today, 0) + max(0, total - done_before)

    def requests_today(self) -> int:
        return self.requests.get(_dt.datetime.now(_dt.UTC).date().isoformat(), 0)

    def note_download(self) -> None:
        """Count one replay actually fetched — the thing BGA's quota counts."""
        today = _dt.datetime.now(_dt.UTC).date().isoformat()
        self.downloads["total"] = self.downloads.get("total", 0) + 1
        self.downloads[today] = self.downloads.get(today, 0) + 1

    def downloads_today(self) -> int:
        return self.downloads.get(_dt.datetime.now(_dt.UTC).date().isoformat(), 0)

    def get_cursor(self) -> CrawlCursor:
        return CrawlCursor.from_dict(self.cursor)

    def set_cursor(self, cursor: CrawlCursor) -> None:
        self.cursor = cursor.to_dict()

    def set_error(self, kind: str, message: str) -> None:
        """Record why a run stopped. ``kind`` is a short machine tag."""
        self.error = {
            "kind": kind,
            "message": message,
            "at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        }

    def clear_error(self) -> None:
        self.error = {}

    def verdict(self, table_id: int) -> TableVerdict | None:
        row = self.tables.get(str(table_id))
        if row is None:
            return None
        return TableVerdict(table_id, row.get("status", ""), row.get("reason", ""))

    def record(self, verdict: TableVerdict, **extra: Any) -> None:
        """Store a table's verdict, plus any audit fields worth keeping.

        ``extra`` is where the crawl records the ladder ``rank`` it came from, the
        ``target`` player (the elite one whose decisions we will learn from) and
        both seats' ``elos`` — so the dataset builder can apply an Elo floor to the
        *target* without re-reading every raw payload, and the progress page can
        tally rejections by reason without a single request.
        """
        row: dict[str, Any] = {"status": verdict.status, "reason": verdict.reason}
        row.update({k: v for k, v in extra.items() if v is not None})
        self.tables[str(verdict.table_id)] = row

    def ranking_rows(self) -> list[PlayerRow]:
        return [PlayerRow(**row) for row in self.ranking.get("rows", [])]


def select_players(
    rows: Iterable[PlayerRow],
    top_n: int | None = 100,
    min_elo_display: int | None = None,
    min_games: int = 0,
) -> list[PlayerRow]:
    """Apply the dataset's player thresholds to a ladder snapshot.

    ``min_elo_display`` is in **website** units (raw minus 1300), because that is
    what a human reads off the leaderboard. Both filters may be combined; the
    result keeps ladder order.
    """
    kept = [r for r in rows if r.games_played >= min_games]
    if min_elo_display is not None:
        kept = [r for r in kept if r.elo_display >= min_elo_display]
    if top_n:
        kept = kept[:top_n]
    return kept


def player_elos(infos: dict[str, Any]) -> dict[int, float]:
    """Raw Elo per seated player id, read out of a ``tableinfos`` payload.

    BGA spells the number ``player_elo`` in most payloads and ``elo`` / ``rank`` in
    others, and it is the **raw** ~1500-centred value (subtract 1300 for the number
    the website shows). Missing or unparseable entries are simply absent from the
    result, and every caller treats "absent" as unknown rather than as zero.
    """
    data = infos.get("data", infos) if isinstance(infos, dict) else {}
    seats = (data or {}).get("players") or {}
    if isinstance(seats, list):
        seats = {str(seat.get("id", index)): seat for index, seat in enumerate(seats)}
    out: dict[int, float] = {}
    for key, seat in seats.items():
        raw: Any = None
        if isinstance(seat, dict):
            for name in ("player_elo", "elo", "rank", "player_rank"):
                if seat.get(name) not in (None, ""):
                    raw = seat[name]
                    break
            key = seat.get("id", key)
        try:
            player_id = int(key)
        except (TypeError, ValueError):
            continue
        if raw is None:
            continue
        try:
            out[player_id] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


def choose_target_player(
    player_ids: Sequence[int],
    elos: dict[int, float] | None = None,
    source_player_id: int | None = None,
    ranked_ids: Iterable[int] = (),
) -> int | None:
    """Which player's decisions we learn from — the **elite** one.

    Many BGA games pair a top-200 player with someone far weaker, and imitating
    the weaker player's moves is worse than not training on the game at all. So one
    game contributes only one player's turns, chosen in this order:

    1. if both seats are in the ladder selection (``ranked_ids``), the higher raw
       Elo of the two;
    2. if exactly one seat is, that player — normally ``source_player_id``, the
       player whose game list this table came from;
    3. failing any ladder information, ``source_player_id`` if it is seated;
    4. failing that, the higher Elo reported by ``tableinfos``.

    Returns ``None`` when nothing distinguishes the seats, which the dataset
    builder treats as "no target" and counts rather than guessing.
    """
    ids = [int(p) for p in player_ids]
    elos = {int(k): float(v) for k, v in (elos or {}).items()}
    known = {int(p) for p in ranked_ids}
    ranked = [p for p in ids if p in known]

    def by_elo(candidates: list[int]) -> int:
        return max(candidates, key=lambda p: (elos.get(p, float("-inf")), -p))

    if len(ranked) > 1:
        return by_elo(ranked)
    if len(ranked) == 1:
        return ranked[0]
    if source_player_id is not None and int(source_player_id) in ids:
        return int(source_player_id)
    rated = [p for p in ids if p in elos]
    if rated:
        return by_elo(rated)
    return None


@dataclass
class Fetcher:
    """Drives :class:`BgaClient` and keeps :class:`FetchState` honest."""

    client: BgaClient
    out_dir: Path
    state: FetchState = field(init=False)
    table_filter: TableFilter = field(default_factory=TableFilter)
    game_id: int = AZUL_GAME_ID
    pace: CrawlPace = field(default_factory=CrawlPace)
    #: injected so the tests can run the crawl without sleeping through its pauses
    sleeper: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.state = FetchState.load(self.out_dir / "state.json", self.game_id)
        self._rng = random.Random(0xC0FFEE)

    # ------------------------------------------------------------------ helpers
    @property
    def raw_dir(self) -> Path:
        return self.out_dir / "raw"

    @property
    def log_path(self) -> Path:
        """Append-only JSONL of every table decision — the page's log tail."""
        return self.out_dir / "fetch.jsonl"

    @property
    def rows_path(self) -> Path:
        """Append-only JSONL of the raw ``getGames`` history rows, verbatim.

        The history row is the ONLY place BGA reports a player's Elo **at the
        time of a game** (``elo_after``) — ``tableinfos`` has no per-seat Elo
        (docs §12.3) and the ladder snapshot is today's number, not the
        game-day's. Discarding the rows (the pre-2026-08-19 behaviour) threw
        that away, so every listed page now appends its rows here; duplicates
        (re-listed pages, tables shared by two ranked players' histories) are
        the reader's problem, by design — this writer must never lose data.
        """
        return self.out_dir / "table_rows.jsonl"

    def _log_rows(
        self, player_id: int, page: int, rows: list[dict[str, Any]]
    ) -> None:
        """Append one JSONL line per history row. Never raises: it is only a log."""
        if not rows:
            return
        at = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")
        try:
            self.rows_path.parent.mkdir(parents=True, exist_ok=True)
            with self.rows_path.open("a", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(
                        json.dumps(
                            {"player": int(player_id), "page": int(page),
                             "at": at, "row": row},
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
        except OSError:  # pragma: no cover - a full disk must not stop a crawl
            pass

    def _flush(self) -> None:
        self.state.note_requests(self.client.requests_made)
        self.state.save()

    def _budget_left(self) -> bool:
        cap = self.client.config.max_requests_per_day
        return not cap or self.state.requests_today() < cap

    def _replay_budget_left(self) -> bool:
        """Our own conservative cap on replays per day, well under BGA's quota."""
        cap = self.pace.max_tables_per_day
        return not cap or self.state.downloads_today() < cap

    def _log_event(self, **event: Any) -> None:
        """One line of JSON per table decision. Never raises: it is only a log."""
        event.setdefault("at", _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"))
        event.setdefault("ts", time.time())
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, separators=(",", ":")) + "\n")
        except OSError:  # pragma: no cover - a full disk must not stop a crawl
            pass

    # ------------------------------------------------------------------ ranking
    def fetch_ranking(self, pages: int = 10, force: bool = False) -> list[PlayerRow]:
        """Fetch ``pages`` x 10 rows of the all-time ladder (public endpoint).

        Cached in the state file: a second call returns the stored snapshot unless
        ``force``. 10 pages = the top 100, which is 10 requests.
        """
        if self.state.ranking.get("rows") and not force:
            return self.state.ranking_rows()
        rows: list[PlayerRow] = []
        for page in range(int(pages)):
            if not self._budget_left():
                break
            batch = self.client.ranking_page(page * 10, game=self.game_id)
            if not batch:
                break
            rows.extend(PlayerRow.from_api(row) for row in batch)
            self._flush()
        self.state.ranking = {
            "fetched": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "rows": [asdict(r) for r in rows],
        }
        self._flush()
        return rows

    # ---------------------------------------------------------- player history
    def fetch_player_tables(
        self, player_id: int, max_pages: int = 5, page_size: int = 10
    ) -> list[int]:
        """Table ids of one player's finished Azul games. **Needs a session.**

        Resumes from ``pages_done``. The response shape is BGA's usual
        ``{"status":1,"data":{"tables":[...]}}`` with rows carrying ``table_id``,
        ``players`` and ``scores``; :func:`extract_table_ids` stays tolerant about
        the layout because we could not observe a real payload anonymously
        (see docs/HUMAN_GAMES.md §8). Pagination is by **1-based ``page``**, not by
        row offset, and the rows themselves are cached in the state file so the
        cheap filters can run without re-requesting anything.
        """
        key = str(int(player_id))
        entry = self.state.players.setdefault(key, {"pages_done": 0, "tables": []})
        if entry.get("complete"):
            return [int(t) for t in entry["tables"]]
        template = endpoints()["player_tables"]
        while entry["pages_done"] < max_pages and self._budget_left():
            page = int(entry["pages_done"]) + 1
            path = template.format(player=int(player_id), game=self.game_id, page=page)
            payload = self.client.get_json(path)
            ids = extract_table_ids(payload)
            self._log_rows(player_id, page, extract_table_rows(payload))
            known = {int(t) for t in entry["tables"]}
            entry["tables"] = sorted(known | set(ids), reverse=True)
            entry["pages_done"] = int(entry["pages_done"]) + 1
            if len(ids) < page_size:
                entry["complete"] = True
                self._flush()
                break
            self._flush()
        return [int(t) for t in entry["tables"]]

    # ------------------------------------------------------------------- tables
    def fetch_table(
        self, table_id: int, meta: dict[str, Any] | None = None
    ) -> TableVerdict:
        """Metadata + move log for one table, filtered, cached, resumable.

        Two requests per accepted table (``tableinfos`` then ``logs``), one per
        rejected one. The verdict is stored either way, so re-running the same
        player list is free.

        ``meta`` is the crawl's context for this table — the ladder ``rank`` and the
        ``source_player_id`` whose history it came from. It is merged with both
        seats' Elos and the resulting **target player** into the raw payload
        (``payload["meta"]``) and into the state file, which is what lets the
        dataset builder learn from the elite seat only and apply an Elo floor to it.
        """
        table_id = int(table_id)
        cached = self.state.verdict(table_id)
        if cached is not None and cached.terminal:
            return cached
        meta = dict(meta or {})
        eps = endpoints()
        try:
            infos = self.client.get_json(eps["table_infos"].format(table=table_id))
            elos = player_elos(infos)
            seats = [int(p) for p in elos] or [
                int(p)
                for p in ((infos.get("data") or infos).get("players") or {})
                if str(p).isdigit()
            ]
            meta["elos"] = {str(pid): elo for pid, elo in elos.items()}
            meta["target_player_id"] = choose_target_player(
                seats,
                elos,
                source_player_id=meta.get("source_player_id"),
                ranked_ids=meta.get("ranked_ids") or (),
            )
            meta.pop("ranked_ids", None)
            reason = self.table_filter.check(infos)
            if reason:
                verdict = TableVerdict(table_id, "skipped", reason)
                self.state.record(verdict, **_audit_fields(meta))
                self._flush()
                self._log_event(
                    table=table_id, status="skipped", reason=reason, **_log_fields(meta)
                )
                return verdict
            # BGA wants the archive requested before it will serve the log; three
            # independent projects do this and one notes it is "seemingly required".
            # Failures here are not fatal — try the log anyway.
            try:
                self.client.get_json(eps["archive_prime"].format(table=table_id))
            except (ReplayLimitReached, AccountDisabled, AuthRequired):
                raise
            except BgaError:
                pass
            logs = self.client.get_json(eps["table_logs"].format(table=table_id))
        except (AuthRequired, ReplayLimitReached, AccountDisabled):
            # Not this table's fault and not retryable: let the caller stop the run.
            raise
        except ReplayUnavailable as exc:
            verdict = TableVerdict(table_id, "skipped", str(exc))
            self.state.record(verdict, **_audit_fields(meta))
            self._flush()
            self._log_event(
                table=table_id,
                status="skipped",
                reason="replay lost",
                **_log_fields(meta),
            )
            return verdict
        except BgaError as exc:
            verdict = TableVerdict(table_id, "error", str(exc))
            self.state.record(verdict, **_audit_fields(meta))
            self._flush()
            self._log_event(
                table=table_id, status="error", reason=str(exc), **_log_fields(meta)
            )
            return verdict
        write_json_gz(
            self.raw_dir / f"{table_id}.json.gz",
            {"table_id": table_id, "infos": infos, "logs": logs, "meta": meta},
        )
        verdict = TableVerdict(table_id, "downloaded")
        self.state.record(verdict, **_audit_fields(meta))
        self.state.note_download()
        self._flush()
        self._log_event(table=table_id, status="downloaded", **_log_fields(meta))
        return verdict

    # ------------------------------------------------------- rank-ordered crawl
    def crawl_ranked(
        self,
        players: Sequence[PlayerRow],
        per_player: int = 120,
        history_pages: int = 12,
        max_tables: int = 0,
        resume: bool = True,
        min_source_elo_raw: float = 0.0,
    ) -> CrawlReport:
        """Walk the ladder **in rank order**, fetching each player's games.

        Rank 1's tables first, then rank 2, and so on. The cursor in the state file
        is updated after every single table (rank, player, offset inside that
        player's list), so a restart re-enters at the table it stopped on and no
        request is ever spent twice — the resume property the whole design exists
        for, since the budget is a daily quota rather than bandwidth.

        Stops cleanly, with the reason recorded in the state file, on:

        * :class:`~ludometer.human.client.ReplayLimitReached` — BGA's daily replay
          quota. We do **not** keep asking after it: the run ends, the page says so,
          and tomorrow's run resumes for free;
        * :class:`~ludometer.human.client.AccountDisabled` /
          :class:`~ludometer.human.client.AuthRequired` — nothing good comes of
          retrying either;
        * our own conservative daily caps (:class:`CrawlPace`).

        ``resume=False`` starts the walk at rank 1 again (nothing is re-*fetched* —
        table verdicts are still terminal — but every player is revisited, which is
        what you want after raising ``per_player``).
        """
        rows = sorted(
            (r for r in players), key=lambda r: (r.rank or 10**9, r.player_id)
        )
        ranked_ids = [r.player_id for r in rows]
        cursor = self.state.get_cursor() if resume else CrawlCursor()
        report = CrawlReport(cursor=cursor)
        self.state.pace = asdict(self.pace)
        self.state.clear_error()
        self.state.save()
        try:
            for row in rows:
                if resume and cursor.rank and row.rank < cursor.rank:
                    continue  # already done in an earlier run
                offset = (
                    cursor.table_offset
                    if (row.rank == cursor.rank and row.player_id == cursor.player_id)
                    else 0
                )
                cursor = CrawlCursor(
                    rank=row.rank,
                    player_id=row.player_id,
                    table_offset=offset,
                    players_done=cursor.players_done,
                )
                self.state.set_cursor(cursor)
                self._flush()
                report.players += 1
                tables = self.fetch_player_tables(
                    row.player_id, max_pages=history_pages
                )[:per_player]
                # `elo_after` floor: the history rows we just listed carry the
                # source player's raw Elo AT THE TIME of each game, so a game
                # played while they were still weak can be skipped BEFORE any
                # request — a top player's beginner era must not spend replay
                # quota (measured: 7% of rank 1's history is below the 650
                # floor). Reloaded per player: this player's rows were only
                # appended by the listing call above.
                at_game_elos: dict[int, dict[int, float]] = (
                    elo_after_map(self.rows_path) if min_source_elo_raw else {}
                )
                for index in range(offset, len(tables)):
                    # A terminal verdict is free to revisit: no request is made and
                    # neither the daily budgets nor the long pause should count it.
                    # This is what makes a `resume=False` re-walk of the whole
                    # ladder (to deepen earlier ranks' histories) cost nothing for
                    # the tables already judged.
                    prior = self.state.verdict(tables[index])
                    if prior is not None and prior.terminal:
                        report.cached += 1
                        cursor = CrawlCursor(
                            rank=row.rank,
                            player_id=row.player_id,
                            table_offset=index + 1,
                            players_done=cursor.players_done,
                        )
                        self.state.set_cursor(cursor)
                        continue
                    at_game = at_game_elos.get(int(tables[index]), {}).get(
                        int(row.player_id)
                    )
                    if at_game is not None and at_game < min_source_elo_raw:
                        reason = (
                            f"source elo {at_game:.0f} at game time below "
                            f"floor {min_source_elo_raw:.0f}"
                        )
                        verdict = TableVerdict(int(tables[index]), "skipped", reason)
                        self.state.record(
                            verdict, rank=row.rank, player=row.player_id
                        )
                        report.skipped += 1
                        cursor = CrawlCursor(
                            rank=row.rank,
                            player_id=row.player_id,
                            table_offset=index + 1,
                            players_done=cursor.players_done,
                        )
                        self.state.set_cursor(cursor)
                        self._flush()
                        self._log_event(
                            table=int(tables[index]),
                            status="skipped",
                            reason=reason,
                            rank=row.rank,
                        )
                        continue
                    if not self._budget_left():
                        return self._stop(
                            report,
                            cursor,
                            "daily-request-cap",
                            f"our own {self.client.config.max_requests_per_day} "
                            "requests/day cap is spent",
                        )
                    if not self._replay_budget_left():
                        return self._stop(
                            report,
                            cursor,
                            "daily-table-cap",
                            f"our own {self.pace.max_tables_per_day} replays/day cap "
                            "is spent (BGA's own quota is stricter and unknown)",
                        )
                    verdict = self.fetch_table(
                        tables[index],
                        meta={
                            "rank": row.rank,
                            "source_player_id": row.player_id,
                            "source_name": row.name,
                            "source_elo_raw": row.elo_raw,
                            "ranked_ids": ranked_ids,
                        },
                    )
                    report.downloaded += verdict.status == "downloaded"
                    report.skipped += verdict.status == "skipped"
                    report.errors += verdict.status == "error"
                    cursor = CrawlCursor(
                        rank=row.rank,
                        player_id=row.player_id,
                        table_offset=index + 1,
                        players_done=cursor.players_done,
                    )
                    self.state.set_cursor(cursor)
                    self._flush()
                    if max_tables and report.downloaded >= max_tables:
                        return self._stop(
                            report, cursor, "limit", f"asked for {max_tables} tables"
                        )
                    self._pause(report.downloaded)
                cursor = CrawlCursor(
                    rank=row.rank,
                    player_id=row.player_id,
                    table_offset=len(tables),
                    players_done=cursor.players_done + 1,
                )
                self.state.set_cursor(cursor)
                self._flush()
        except ReplayLimitReached as exc:
            return self._stop(report, cursor, "replay-limit", str(exc))
        except AccountDisabled as exc:
            return self._stop(report, cursor, "account-disabled", str(exc))
        except AuthRequired as exc:
            return self._stop(report, cursor, "auth", str(exc))
        report.cursor = cursor
        return report

    def _stop(
        self, report: CrawlReport, cursor: CrawlCursor, kind: str, message: str
    ) -> CrawlReport:
        """Record why the crawl stopped and hand the report back to the caller."""
        report.stopped = kind
        report.message = message
        report.cursor = cursor
        self.state.set_cursor(cursor)
        if kind != "limit":
            self.state.set_error(kind, message)
        self._flush()
        self._log_event(status="stopped", reason=kind, message=message)
        return report

    def _pause(self, tables_done: int) -> None:
        """The randomized long break, every ``pace.long_pause_every`` tables."""
        every = self.pace.long_pause_every
        if not every or not tables_done or tables_done % every:
            return
        low, high = self.pace.long_pause_min, self.pace.long_pause_max
        self.sleeper(low + self._rng.random() * max(0.0, high - low))

    def iter_raw(self) -> Iterator[tuple[int, dict[str, Any]]]:
        """Every downloaded payload, oldest file first — the converter's input."""
        for path in sorted(self.raw_dir.glob("*.json.gz")):
            table_id = int(path.name.split(".")[0])
            yield table_id, read_json_gz(path)


def _audit_fields(meta: dict[str, Any]) -> dict[str, Any]:
    """The parts of a table's crawl context worth keeping in the state file.

    Both seats' Elos are kept (not just the target's) because that is what lets a
    later dataset build apply an Elo floor to the **target** player, and what lets
    the progress page say how lopsided the games were, without re-reading a single
    raw payload or spending a request.
    """
    return {
        "rank": meta.get("rank"),
        "player": meta.get("source_player_id"),
        "target": meta.get("target_player_id"),
        "elos": meta.get("elos") or None,
    }


def _log_fields(meta: dict[str, Any]) -> dict[str, Any]:
    """The short form of the same thing, for the JSONL log tail."""
    target = meta.get("target_player_id")
    elos = meta.get("elos") or {}
    return {
        "rank": meta.get("rank"),
        "player": meta.get("source_name") or meta.get("source_player_id"),
        "target": target,
        "target_elo": elos.get(str(target)) if target is not None else None,
    }


def extract_table_ids(payload: dict[str, Any]) -> list[int]:
    """Pull table ids out of a ``getGames`` payload without assuming its shape.

    BGA has renamed these fields before, and this recon never saw an authenticated
    response, so we accept any list of dicts under ``data`` and take the first key
    that looks like a table id. Any real payload with ``table_id``/``id`` works;
    anything else returns nothing rather than garbage.
    """
    ids: list[int] = []
    for row in extract_table_rows(payload):
        for key in ("table_id", "id", "table"):
            if key in row:
                try:
                    ids.append(int(row[key]))
                except (TypeError, ValueError):
                    pass
                break
    return ids


def elo_after_map(path: str | Path) -> dict[int, dict[int, float]]:
    """``table_rows.jsonl`` -> ``{table_id: {player_id: elo_after_raw}}``.

    ``elo_after`` is the **queried** player's raw Elo right after that game — the
    only per-game, time-accurate Elo BGA exposes (verified live 2026-08-19:
    ``{"table_id": "897979436", ..., "elo_after": "2488", "elo_win": "2"}``).
    A table shared by two ranked players' histories appears once per history,
    each contributing its own player's number. Later lines win (re-listed pages
    refresh nothing — the value is historical — but keep the newest anyway).
    Unparseable lines and blank ``elo_after`` values are skipped, never fatal.
    """
    out: dict[int, dict[int, float]] = {}
    path = Path(path)
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
            row = event.get("row") or {}
            table_id = int(row.get("table_id") or row.get("id"))
            player = int(event["player"])
            elo = float(row["elo_after"])
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
        out.setdefault(table_id, {})[player] = elo
    return out


def extract_table_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """The history rows themselves, wherever the envelope keeps them.

    A real ``getGames`` row (from working community code) looks like::

        {"table_id": "712345678", "players": "91843016,91718783",
         "scores": "78,64", "start": 1690000000, "concede": 0, "arena_win": null}

    ``players`` and ``scores`` are **comma-joined strings**, and their presence is
    what could let us skip the per-table ``tableinfos`` call for the player-count
    filter and the score check — see :func:`table_row_players` and docs §5.2.
    Community code also reads the payload as ``results[0].data.tables``, which is a
    dojo sync-XHR artefact, so that spelling is accepted too.
    """
    seen: Any = payload
    if (
        isinstance(seen, dict)
        and isinstance(seen.get("results"), list)
        and seen["results"]
    ):
        seen = seen["results"][0]
    if isinstance(seen, dict):
        seen = seen.get("data", seen)
    rows: list[Any] = []
    if isinstance(seen, dict):
        for key in ("tables", "games", "rows"):
            value = seen.get(key)
            if isinstance(value, list):
                rows = value
                break
            if isinstance(value, dict):
                rows = list(value.values())
                break
    elif isinstance(seen, list):
        rows = seen
    return [row for row in rows if isinstance(row, dict)]


def table_row_players(row: dict[str, Any]) -> list[int]:
    """Player ids from a history row's comma-joined ``players`` field.

    ``len(...) == 2`` is the cheap 2-player filter: it costs no request at all,
    where the same answer from ``tableinfos`` costs one per table.
    """
    raw = row.get("players")
    if raw is None:
        return []
    parts = str(raw).split(",") if not isinstance(raw, list) else raw
    out: list[int] = []
    for part in parts:
        try:
            out.append(int(str(part).strip()))
        except (TypeError, ValueError):
            continue
    return out


def table_row_scores(row: dict[str, Any]) -> list[int]:
    """Final scores from a history row's comma-joined ``scores`` field.

    Same order as :func:`table_row_players`. Useful as an independent cross-check
    of the score the log reports — if the two disagree, distrust the parse.
    """
    raw = row.get("scores")
    if raw is None:
        return []
    parts = str(raw).split(",") if not isinstance(raw, list) else raw
    out: list[int] = []
    for part in parts:
        try:
            out.append(int(str(part).strip()))
        except (TypeError, ValueError):
            continue
    return out
