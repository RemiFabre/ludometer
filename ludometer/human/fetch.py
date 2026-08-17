"""Resumable, rate-limited fetcher for BGA Azul games.

Three stages, each one restartable and each one recorded in a single JSON state
file so that a run killed halfway costs nothing:

1. :meth:`Fetcher.fetch_ranking` — the **all-time** ladder (public, no cookies).
2. :meth:`Fetcher.fetch_player_tables` — one player's finished Azul tables
   (needs a session).
3. :meth:`Fetcher.fetch_table` — one table's metadata + move log (needs a
   session). This is the only stage whose volume is large, and it is the one the
   filters below exist to keep small.

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
from collections.abc import Iterable, Iterator
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
    "STANDARD_WALL_OPTION_HINTS",
    "FetchState",
    "Fetcher",
    "PlayerRow",
    "TableFilter",
    "TableVerdict",
    "extract_table_ids",
    "extract_table_rows",
    "option_value",
    "select_players",
    "table_row_players",
    "table_row_scores",
]

STATE_VERSION = 1

#: Where Azul's wall variant lives in `tableinfos.options` — the filter Remi asked
#: for ("the gray board where wall placement is free must be excluded").
#:
#: **Hypothesis, not yet verified.** Azul's player board is double-sided: the fixed
#: colour wall, and the grey wall where a tile may go in any column of its row. BGA
#: models that as a *major variant*, and Azul's public game metadata indeed lists
#: exactly two of them (`media.majorvariant` has keys "1" and "2" — those keys are
#: the option's values). BGA's convention puts the major variant at option id 100,
#: so `options["100"] == 1` is almost certainly the standard wall and `== 2` the
#: grey one. Confirming it costs one authenticated call to
#: `/gamelist/gamelist/gameOptions.html?game=1467`, which this recon could not make
#: (it answers 806 anonymously) — see docs/HUMAN_GAMES.md §3 and §9 step 4.
#:
#: Until `option_id` is filled in, :class:`TableFilter` is **fail-closed**: every
#: table is skipped with a greppable reason rather than accepted on a guess. Note
#: that this filter only saves requests — a grey-wall game cannot reach the dataset
#: anyway, because its extra "choose a column" notification is an unknown type to
#: the parser and its scores would not match ours (docs §3.2).
STANDARD_WALL_OPTION_HINTS = {
    "option_id": None,  # set to 100 once verified
    "standard_values": (1,),  # the value(s) meaning "standard fixed colour wall"
    "variant_name_patterns": ("variable", "variant", "grey", "gray", "free"),
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
            for seat in seats.values():
                elo = seat.get("player_elo") or seat.get("elo") or seat.get("rank") or 0
                if float(elo or 0) < self.min_player_elo_raw:
                    return f"seat elo {elo} below floor"
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
          "version": 1,
          "game_id": 1467,
          "started": "2026-08-17T10:00:00",
          "requests": {"total": 812, "2026-08-17": 812},
          "ranking": {"fetched": "2026-08-17T10:05:00",
                      "rows": [ {player_id, name, elo_raw, elo_display,
                                 rank, games_played}, ... ]},
          "players": {"91843016": {"pages_done": 3, "complete": true,
                                   "tables": [712345678, ...]}},
          "tables":  {"712345678": {"status": "downloaded", "reason": ""}}
        }

    Resuming = load it and skip. ``players[pid]["pages_done"]`` is the number of
    history pages already consumed, so a player is resumed mid-history; a table
    with a terminal verdict is never requested again.
    """

    path: Path
    game_id: int = AZUL_GAME_ID
    version: int = STATE_VERSION
    started: str = ""
    requests: dict[str, int] = field(default_factory=dict)
    ranking: dict[str, Any] = field(default_factory=dict)
    players: dict[str, dict[str, Any]] = field(default_factory=dict)
    tables: dict[str, dict[str, Any]] = field(default_factory=dict)

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
        if int(payload.get("version", 0)) != STATE_VERSION:
            raise ValueError(
                f"state file {path} has version {payload.get('version')}, "
                f"this code writes {STATE_VERSION}"
            )
        return cls(
            path=path,
            game_id=int(payload.get("game_id", game_id)),
            version=STATE_VERSION,
            started=payload.get("started", ""),
            requests=dict(payload.get("requests", {})),
            ranking=dict(payload.get("ranking", {})),
            players=dict(payload.get("players", {})),
            tables=dict(payload.get("tables", {})),
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": STATE_VERSION,
            "game_id": self.game_id,
            "started": self.started,
            "requests": self.requests,
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

    def verdict(self, table_id: int) -> TableVerdict | None:
        row = self.tables.get(str(table_id))
        if row is None:
            return None
        return TableVerdict(table_id, row.get("status", ""), row.get("reason", ""))

    def record(self, verdict: TableVerdict) -> None:
        self.tables[str(verdict.table_id)] = {
            "status": verdict.status,
            "reason": verdict.reason,
        }

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


@dataclass
class Fetcher:
    """Drives :class:`BgaClient` and keeps :class:`FetchState` honest."""

    client: BgaClient
    out_dir: Path
    state: FetchState = field(init=False)
    table_filter: TableFilter = field(default_factory=TableFilter)
    game_id: int = AZUL_GAME_ID

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.state = FetchState.load(self.out_dir / "state.json", self.game_id)

    # ------------------------------------------------------------------ helpers
    @property
    def raw_dir(self) -> Path:
        return self.out_dir / "raw"

    def _flush(self) -> None:
        self.state.note_requests(self.client.requests_made)
        self.state.save()

    def _budget_left(self) -> bool:
        cap = self.client.config.max_requests_per_day
        return not cap or self.state.requests_today() < cap

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
    def fetch_table(self, table_id: int) -> TableVerdict:
        """Metadata + move log for one table, filtered, cached, resumable.

        Two requests per accepted table (``tableinfos`` then ``logs``), one per
        rejected one. The verdict is stored either way, so re-running the same
        player list is free.
        """
        table_id = int(table_id)
        cached = self.state.verdict(table_id)
        if cached is not None and cached.terminal:
            return cached
        eps = endpoints()
        try:
            infos = self.client.get_json(eps["table_infos"].format(table=table_id))
            reason = self.table_filter.check(infos)
            if reason:
                verdict = TableVerdict(table_id, "skipped", reason)
                self.state.record(verdict)
                self._flush()
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
            self.state.record(verdict)
            self._flush()
            return verdict
        except BgaError as exc:
            verdict = TableVerdict(table_id, "error", str(exc))
            self.state.record(verdict)
            self._flush()
            return verdict
        write_json_gz(
            self.raw_dir / f"{table_id}.json.gz",
            {"table_id": table_id, "infos": infos, "logs": logs},
        )
        verdict = TableVerdict(table_id, "downloaded")
        self.state.record(verdict)
        self._flush()
        return verdict

    def iter_raw(self) -> Iterator[tuple[int, dict[str, Any]]]:
        """Every downloaded payload, oldest file first — the converter's input."""
        for path in sorted(self.raw_dir.glob("*.json.gz")):
            table_id = int(path.name.split(".")[0])
            yield table_id, read_json_gz(path)


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
