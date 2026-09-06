"""BGA replay log JSON -> :class:`ReplayGame` (an ordered, engine-flavoured game).

What a BGA replay actually is
-----------------------------
``GET /archive/archive/logs.html?table=<id>&translated=true`` answers the
framework's own notification stream for a finished table — the same packets the
browser client replayed to draw the game::

    {"status": 1, "data": {"players": [...], "logs": [
        {"channel": "/table/t181130958", "table_id": "181130958",
         "packet_id": "2", "packet_type": "resend", "move_id": "2",
         "time": "1624036533",
         "data": [ {"uid": "60ccd4b5c9dea", "type": "<notification name>",
                    "log": "${player_name} takes ...", "args": { ... }} ]},
        ...]}}

A replay page's embedded ``g_gamelogs`` global carries the same thing one level
deeper (``data.data``); :func:`log_packets` accepts both, and
:func:`parse_gamelogs_html` reads the page.

So a replay is **structured JSON, not HTML and not a rendered move list**: each
entry carries the notification ``type`` and its machine ``args``. The ``log``
string is only the human sentence and we never parse it.

Three framework facts that shape this module:

* packets are per **channel**. ``/table/tNNN`` is the public game; ``/player/pNNN``
  carries one player's private UI hints. :func:`iter_log_entries` keeps only the
  table channel;
* the deal is **observable** (the client has to be told which tiles appeared), so
  :mod:`ludometer.human.convert` scripts our engine's chance events from the log
  rather than drawing its own;
* every game names its own notifications. Azul's are known — see
  :class:`LogSchema` — but they are known **second-hand**, from a working
  third-party Azul parser rather than from a log we read ourselves, so each field
  is overridable data and :func:`log_type_histogram` prints what a real log
  actually contains. ``docs/HUMAN_GAMES.md`` §4.4 is the confirmation recipe.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ludometer.azul.engine import CENTER, FLOOR, NUM_COLORS, NUM_FACTORIES, NUM_ROWS

__all__ = [
    "AZUL_COLOR_MAP",
    "DEFAULT_SCHEMA",
    "Deal",
    "LogSchema",
    "ParseError",
    "Pick",
    "ReplayGame",
    "WallPlacement",
    "iter_log_entries",
    "log_packets",
    "log_type_histogram",
    "observed_color_ids",
    "parse_gamelogs_html",
    "parse_log",
    "with_color_map",
]

#: BGA Azul tile ``type`` -> our engine's colour index.
#:
#: BGA numbers Azul's tiles ``0`` = first-player marker, ``1`` = Black,
#: ``2`` = Cyan, ``3`` = Blue, ``4`` = Yellow, ``5`` = Red (documented by the
#: ``bga-assistant`` extension, which extracted the tile images per type). Our
#: engine is ``0`` = blue, ``1`` = yellow, ``2`` = red, ``3`` = black, ``4`` = teal
#: (:data:`ludometer.azul.engine.COLOR_NAMES`), so:
#:
#: ===========  ===============  =============
#: BGA ``type``  colour           engine index
#: ===========  ===============  =============
#: 1            Black            3
#: 2            Cyan / teal      4
#: 3            Blue             0
#: 4            Yellow           1
#: 5            Red              2
#: ===========  ===============  =============
#:
#: This is the one mapping a mistake in would be invisible to the eye and fatal to
#: the dataset, so it is **verified per game, mechanically**: the wall column of
#: colour ``c`` in row ``r`` is ``(c + r) % 5``, and the log tells us the column of
#: every tile it places, so a wrong permutation contradicts the log within one
#: round (:func:`ludometer.human.convert.check_wall_placements`) and changes the
#: final score (:func:`ludometer.human.convert.solve_color_map`).
AZUL_COLOR_MAP = {1: 3, 2: 4, 3: 0, 4: 1, 5: 2}

#: BGA's tile ``type`` for the first-player marker. It sits in floor-line tile
#: lists and must never be counted as a coloured tile.
MARKER_TILE_TYPE = 0


class ParseError(ValueError):
    """The log did not match the schema — always names the offending entry."""


@dataclass(frozen=True)
class Pick:
    """One human turn, already in our engine's coordinates.

    ``source`` 0..4 factory / 5 center, ``color`` 0..4, ``dest`` 0..4 pattern row
    / 5 floor — i.e. ``encode_action(source, color, dest)`` is the action id.

    An Azul turn is **two** notifications in the log (``tilesSelected`` then
    ``tilesPlacedOnLine``); :func:`parse_log` pairs them into one of these.
    """

    player_id: int
    source: int
    color: int
    dest: int
    move_id: int = 0
    count: int = 0  # tiles taken, when the log says; 0 = unknown

    def action_id(self) -> int:
        return self.source * 30 + self.color * 6 + self.dest


@dataclass(frozen=True)
class Deal:
    """The tiles that appeared on the factories at the start of one round.

    ``factories`` is ``NUM_FACTORIES`` lists of ``NUM_COLORS`` counts, the same
    layout as ``AzulState.factories``. A short deal (end of bag) is allowed and is
    exactly why we script the refill instead of drawing our own.
    ``remaining`` is BGA's own post-deal bag count when the log reports it
    (``factoriesFilled.args.remainingTiles``), a free cross-check.
    """

    round_index: int
    factories: tuple[tuple[int, ...], ...]
    remaining: int | None = None

    def total(self) -> int:
        return sum(sum(f) for f in self.factories)


@dataclass(frozen=True)
class WallPlacement:
    """One tile moved to the wall at the end of a round.

    The log gives the ``column`` BGA put it in, which is what makes the standard
    wall verifiable: on the fixed wall ``column == (color + row) % 5`` always, and
    in the grey "variable wall" variant it need not — so these records both confirm
    the colour map and identify variant games
    (:func:`ludometer.human.convert.check_wall_placements`).
    """

    player_id: int
    color: int
    row: int
    column: int


@dataclass
class ReplayGame:
    """One parsed BGA table, ready for :func:`ludometer.human.convert.convert_game`."""

    table_id: int
    player_ids: tuple[int, ...]  # seat order = our engine's player 0, 1
    picks: tuple[Pick, ...] = ()
    deals: tuple[Deal, ...] = ()
    wall_placements: tuple[WallPlacement, ...] = ()
    first_player: int | None = None  # BGA player id holding the marker in round 0
    final_scores: dict[int, int] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    #: BGA player id who resigned, or ``None`` for a game that ran to its natural
    #: end. When set, the game stopped at the concession: the picks are those played
    #: up to it and :func:`ludometer.human.convert.convert_game` scores it as a loss
    #: for this player rather than requiring a terminal engine state.
    conceded_by: int | None = None

    def seat_of(self, player_id: int) -> int:
        try:
            return self.player_ids.index(int(player_id))
        except ValueError as exc:  # pragma: no cover - defensive
            raise ParseError(
                f"player {player_id} is not seated at table {self.table_id}"
            ) from exc

    def scores_by_seat(self) -> tuple[int, int] | None:
        if len(self.player_ids) != 2 or len(self.final_scores) != 2:
            return None
        return (
            int(self.final_scores[self.player_ids[0]]),
            int(self.final_scores[self.player_ids[1]]),
        )


@dataclass(frozen=True)
class LogSchema:
    """How to read one game's notifications. Everything game-specific lives here.

    The Azul defaults come from a **working third-party Azul log parser** (the
    ``bga-assistant`` extension, which ships a fixture and design notes), so the
    notification names, the tile object and the tile-type numbering are grounded
    rather than guessed. What remains unconfirmed is flagged per field below and
    listed in ``docs/HUMAN_GAMES.md`` §8.

    Confirmed Azul notifications and their arguments::

        factoriesFilled   {factories: Tile[][], remainingTiles: int}
        tilesSelected     {type, selectedTiles[], discardedTiles[], fromFactory}
        tilesPlacedOnLine {placedTiles[], discardedTiles[], line}
        placeTileOnWall   {completeLines: {pid: {placedTile, discardedTiles[],
                                                pointsDetail}}}
        emptyFloorLine    {floorLines: {pid: {tiles[], points}}}   # [] when empty!
        firstPlayerToken  {...}

    A ``Tile`` is ``{"id", "type", "column", "line", "location"}`` with
    ``location`` one of ``"factory_N"``, ``"wall"``, ``"discard"``, ``"floor"``, and
    ``type`` the colour — ``0`` meaning the first-player marker, not a colour.
    """

    #: "took tiles": carries the colour and the source display.
    select_types: tuple[str, ...] = ("tilesSelected", "tilesTaken", "takeTiles")
    #: "put them somewhere": carries the destination pattern line.
    place_types: tuple[str, ...] = (
        "tilesPlacedOnLine",
        "tilesPlaced",
        "placeTiles",
        "tilePlaced",
    )
    #: start-of-round factory fill.
    deal_types: tuple[str, ...] = (
        "factoriesFilled",
        "newRound",
        "fillFactories",
        "tilesToFactories",
    )
    #: round-end wall tiling; the placed tile carries ``line`` and ``column``.
    wall_types: tuple[str, ...] = ("placeTileOnWall",)
    #: round-end floor clearing.
    floor_clear_types: tuple[str, ...] = ("emptyFloorLine",)
    score_types: tuple[str, ...] = ("score", "scoreUpdate", "playerScore", "finalScore")
    marker_types: tuple[str, ...] = (
        "firstPlayerToken",
        "takeFirstPlayer",
        "firstPlayer",
    )
    #: "take that back": a player rewound a move before confirming it. BGA lets a
    #: turn be undone in two steps, and emits one notification per step **in reverse
    #: order** — confirmed across the real corpus (2026-08-17):
    #:
    #: * ``undoSelectLine`` undoes the *placement* (``tilesPlacedOnLine``): the tiles
    #:   go back into the hand, so the turn re-opens as a pending selection. It is
    #:   always immediately preceded by a ``tilesPlacedOnLine``.
    #: * ``undoTakeTiles`` undoes the *take* (``tilesSelected``): the open selection
    #:   is cancelled entirely. It is always immediately preceded either by a
    #:   ``tilesSelected`` (undo of an un-placed take) or by an ``undoSelectLine``
    #:   (undo of a fully-placed turn, placement first then take).
    #:
    #: :func:`parse_log` must *rewind* these, not ignore them: the reconstructed move
    #: sequence has to match what finally happened at the table, or the engine replay
    #: desynchronises and every following move looks illegal.
    undo_place_types: tuple[str, ...] = ("undoSelectLine", "undoPlaceTiles")
    undo_take_types: tuple[str, ...] = ("undoTakeTiles", "undoSelect")
    #: A player resigned. The game ends at that notification (confirmed: it is always
    #: the last move notification in the log); the moves played up to it are kept and
    #: the outcome is the resignation result — the conceder loses. Handled by
    #: :func:`ludometer.human.convert.convert_game`, which reads
    #: :attr:`ReplayGame.conceded_by`.
    concede_types: tuple[str, ...] = (
        "playerConcedeGame",
        "playerGiveUp",
        "concede",
    )
    #: Framework-level notifications that every BGA game emits and that carry no
    #: move information (the first four are confirmed present in real logs).
    #: Anything NOT listed here and not recognised above is a fatal parse error, on
    #: purpose: a silently dropped notification is a silently wrong game.
    ignore_types: tuple[str, ...] = (
        "gameStateChange",
        "gameStateMultipleActiveUpdate",
        "leaveGameState",
        "updateReflexionTime",
        "updateMoves",
        "message",
        "simpleNote",
        "simpleNode",
        "wakeupPlayers",
        "yourturnack",
        "tableWindowShow",
        "tableWindowClose",
        "history_history",
        "resend",
        # Confirmed present in a real Azul archive log (2026-08-17) and carrying no
        # move information — they are the human-readable "…completed a line/column…"
        # text lines that mirror a real notification, the last-round banner, and the
        # end-of-game per-tile score breakdown. The authoritative final score is read
        # from `tableinfos` (see `scores_from_infos`), not from `endScore`, whose
        # `points` are the *final-round* increment, not a total.
        "placeTileOnWallTextLogDetails",
        "emptyFloorLineTextLogDetails",
        "completeLineLogDetails",
        "completeColumnLogDetails",
        # `completeColorLogDetails` is the same family (a "…completed a colour…"
        # human-readable log line, present in a handful of real games) and carries
        # no move information.
        "completeColorLogDetails",
        "lastRound",
        "endScore",
        # Turn-based (holiday) clock bookkeeping — "${player_name} uses a holiday
        # time joker (+${nb_days} days thinking time)". Args are player_name/nb_days
        # only (verified on tables 779195006 and 781529784); no move information.
        "timeJokerUsed",
    )
    arg_aliases: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            # `firstPlayerToken` uses `playerId`, most others `player_id`.
            "player": ("player_id", "playerId", "player", "pid"),
            "source": ("fromFactory", "factory", "factory_id", "from", "source"),
            "color": ("type", "color", "color_id", "tile_type", "tile"),
            "dest": ("line", "row", "to", "dest", "pattern_line", "target"),
            "score": ("score", "player_score", "points", "total"),
            "factories": ("factories", "tiles", "displays", "content"),
            "remaining": ("remainingTiles", "remaining", "bag"),
            "selected": ("selectedTiles", "tiles", "placedTiles"),
            "placed": ("placedTiles", "tiles"),
            "complete_lines": ("completeLines", "lines"),
            "floor_lines": ("floorLines", "floors"),
            "placed_tile": ("placedTile", "tile"),
        }
    )
    #: BGA tile ``type`` -> engine colour. Defaults to :data:`AZUL_COLOR_MAP`;
    #: ``None`` means "infer from the ids present", which only works when all five
    #: colours appear.
    color_map: dict[int, int] | None = field(
        default_factory=lambda: dict(AZUL_COLOR_MAP)
    )
    #: **Confirmed live 2026-08-17**: the log numbers the center pile ``0`` in
    #: ``tilesSelected.fromFactory`` (the five real factories are ``1``..``5``), and
    #: ``factoriesFilled.args.factories`` is the *same* 1-based list — index 0 is the
    #: center/"deck" slot (it holds only the first-player marker at fill time) and
    #: indices 1..5 are the factories. So ``center_values`` includes ``0``; anything
    #: ``>= NUM_FACTORIES`` after de-basing is also treated as the center for
    #: robustness, and the negative/sentinel escapes are kept.
    center_values: tuple[int, ...] = (0, -1, 99)
    #: **Confirmed live 2026-08-17**: pattern lines are 1..5 with ``0`` the floor, so
    #: this floor set and ``lines_one_based`` below are both correct. Applies to
    #: ``tilesPlacedOnLine.line`` and to the wall placement's ``line``.
    floor_values: tuple[int, ...] = (0, -1, 6, 9)
    #: **Confirmed live 2026-08-17**: ``True`` — the log's pattern lines are 1..5.
    lines_one_based: bool = True
    #: **Confirmed live 2026-08-17**: ``True`` — factory ids (``fromFactory`` and the
    #: ``factories`` array index) are 1..5, with 0 = center. The earlier guess of
    #: 0-based ``factory_0`` strings was wrong: the real tile ``location`` is just
    #: ``"factory"`` and the numbering lives in ``fromFactory`` / the array index.
    factories_one_based: bool = True
    #: **Confirmed live 2026-08-17**: ``True`` — the wall placement's ``column`` (and
    #: the deal tiles' ``column``) are 1..5. ``convert.wall_col`` returns a 0-based
    #: column, so the wall column must be de-based by one before comparison.
    columns_one_based: bool = True
    #: Marker tile ``type``, excluded from every colour count.
    marker_tile_type: int = MARKER_TILE_TYPE

    def arg(self, args: dict[str, Any], logical: str) -> Any:
        for key in self.arg_aliases.get(logical, ()):
            if key in args:
                return args[key]
        return None


DEFAULT_SCHEMA = LogSchema()


def with_color_map(schema: LogSchema, color_map: dict[int, int]) -> LogSchema:
    """A copy of ``schema`` pinned to one colour mapping (used by the solvers)."""
    return replace(schema, color_map=dict(color_map))


# --------------------------------------------------------------------- envelopes
def log_packets(payload: Any) -> list[dict[str, Any]]:
    """The packet list, whichever envelope it arrived in.

    The live endpoint returns it at ``data.logs``; a replay page's ``g_gamelogs``
    global nests one level deeper at ``data.data``. Both are accepted, along with a
    bare list, because the fetcher stores whatever BGA sent and tests build the
    inner list directly.
    """
    seen: Any = payload
    for _ in range(3):
        if isinstance(seen, dict):
            for key in ("logs", "packets"):
                if isinstance(seen.get(key), list):
                    return list(seen[key])
            if "data" in seen:
                seen = seen["data"]
                continue
        break
    if isinstance(seen, list):
        return list(seen)
    raise ParseError(
        "log payload holds no packet list (looked for data.logs / data.data)"
    )


def iter_log_entries(
    payload: Any, table_channel_only: bool = True
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(move_id, entry)`` for every notification, in packet order.

    ``table_channel_only`` drops packets on a ``/player/pNNN`` channel. Those are
    real and they are in the log, but they are one player's **private** UI hints
    rather than public game events — including them would both confuse the parser
    and quietly pull one player's private view into the dataset.
    """
    for packet in log_packets(payload):
        if not isinstance(packet, dict):
            continue
        channel = str(packet.get("channel", ""))
        if table_channel_only and channel and not channel.startswith("/table/"):
            continue
        try:
            move_id = int(packet.get("move_id") or 0)
        except (TypeError, ValueError):
            move_id = 0
        entries = packet.get("data") or []
        if isinstance(entries, dict):
            entries = list(entries.values())
        for entry in entries:
            if isinstance(entry, dict):
                yield move_id, entry


def parse_gamelogs_html(text: str) -> dict[str, Any]:
    """Pull the log JSON out of a **saved replay page** (``g_gamelogs = {...};``).

    BGA's replay page embeds the whole log in a global, so a page saved from a
    browser is a complete replay obtained with no request of ours at all — the one
    route to a real Azul log that involves no automated access whatsoever (Remi
    opens a replay, saves the page, hands over the file). Prior art:
    ``BGAtoFreeboard/main.py`` uses the same regex, ``DavidEGx/Hive-bga2bs`` reads
    the same global in-page, and BGA's own client does
    ``g_gamelogs = g_gamelogs.data.data``.
    """
    match = re.search(r"g_gamelogs\s*=\s*(\{.*?\})\s*;", text, re.DOTALL)
    if not match:
        raise ParseError("no g_gamelogs blob in this page")
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ParseError(f"g_gamelogs is not valid JSON: {exc}") from exc


def log_type_histogram(payload: Any) -> dict[str, dict[str, Any]]:
    """``{type: {"count": n, "arg_keys": [...], "example": {...}}}``.

    The first thing to run on a freshly fetched real log: it prints exactly which
    notification names and arg keys the game uses, which is all :class:`LogSchema`
    needs. Exposed on the CLI as ``inspect``.
    """
    out: dict[str, dict[str, Any]] = {}
    for _move_id, entry in iter_log_entries(payload):
        name = str(entry.get("type", "?"))
        args = entry.get("args") or {}
        row = out.setdefault(name, {"count": 0, "arg_keys": set(), "example": args})
        row["count"] += 1
        if isinstance(args, dict):
            row["arg_keys"].update(args.keys())
    for row in out.values():
        row["arg_keys"] = sorted(row["arg_keys"])
    return out


# ------------------------------------------------------------------- primitives
def _as_int(value: Any, what: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ParseError(f"{what}: expected an int, got {value!r}") from exc


def _tile_types(tiles: Any, schema: LogSchema, what: str) -> list[int]:
    """Raw ``type`` of every **coloured** tile in a tile list (marker excluded)."""
    if tiles is None:
        return []
    if isinstance(tiles, dict):
        tiles = list(tiles.values())
    if not isinstance(tiles, Sequence) or isinstance(tiles, str):
        raise ParseError(f"{what}: expected a list of tiles, got {tiles!r}")
    out: list[int] = []
    for tile in tiles:
        raw = tile.get("type") if isinstance(tile, dict) else tile
        value = _as_int(raw, f"{what} tile type")
        if value != schema.marker_tile_type:
            out.append(value)
    return out


def _map_color(raw: Any, color_map: dict[int, int]) -> int:
    key = _as_int(raw, "colour")
    if key in color_map:
        return color_map[key]
    raise ParseError(f"tile type {raw!r} is not in the colour map {color_map}")


def _map_source(raw: Any, schema: LogSchema) -> int:
    """``fromFactory`` -> engine source (0..4 displays, 5 = center)."""
    if isinstance(raw, str) and not raw.lstrip("-").isdigit():
        text = raw.lower()
        if "center" in text or "centre" in text:
            return CENTER
        match = re.search(r"(-?\d+)", text)
        if not match:
            raise ParseError(f"cannot read a source from {raw!r}")
        raw = match.group(1)
    value = _as_int(raw, "pick source")
    if value in schema.center_values:
        return CENTER
    index = value - 1 if schema.factories_one_based else value
    if index >= NUM_FACTORIES:
        # 0-based displays are 0..4 for two players, so anything above is the
        # center pile — the same convention our own engine uses (CENTER == 5).
        return CENTER
    if index < 0:
        raise ParseError(f"pick source {raw!r} maps to {index}")
    return index


def _map_line(raw: Any, schema: LogSchema, what: str = "destination") -> int:
    """``line`` -> engine destination (0..4 pattern rows, 5 = floor)."""
    value = _as_int(raw, what)
    if value in schema.floor_values:
        return FLOOR
    row = value - 1 if schema.lines_one_based else value
    if not 0 <= row < NUM_ROWS:
        raise ParseError(f"{what} {raw!r} maps outside 0..{NUM_ROWS - 1}")
    return row


def _counts(types: Sequence[int], color_map: dict[int, int]) -> list[int]:
    counts = [0] * NUM_COLORS
    for raw in types:
        counts[_map_color(raw, color_map)] += 1
    return counts


def _parse_factories(
    value: Any, schema: LogSchema, colors: list[int]
) -> list[list[int]]:
    """Normalise ``factoriesFilled.args.factories`` into 5 lists of raw tile types.

    Accepts the confirmed shape (a list per factory of tile objects) as well as a
    flat list of ``{fromFactory, type}`` records and plain colour-id lists, because
    only the first is confirmed and the others cost nothing to tolerate.
    """
    per_factory: list[list[int]] = [[] for _ in range(NUM_FACTORIES)]
    if isinstance(value, dict):
        value = [value[k] for k in sorted(value, key=str)]
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ParseError(f"deal argument is not a list: {value!r}")
    if value and isinstance(value[0], dict) and "type" in value[0]:
        # flat list of tiles, each naming its own factory
        for tile in value:
            index = _map_source(schema.arg(tile, "source"), schema)
            if index >= NUM_FACTORIES:
                raise ParseError("a deal cannot place tiles in the center")
            for raw in _tile_types([tile], schema, "deal"):
                per_factory[index].append(raw)
                colors.append(raw)
        return per_factory
    # The real (2026-08-17) shape is 1-based: entry 0 is the center/"deck" slot
    # (it carries only the first-player marker at fill time), entries 1..5 are the
    # factories — the same numbering `fromFactory` uses. Route each array index
    # through `_map_source` (which knows `factories_one_based` + `center_values`),
    # drop whatever lands on the center, and reject a genuinely oversized deal
    # (a 3-/4-player table) rather than silently truncating it.
    real_factories = 0
    for index, tiles in enumerate(value):
        dest = _map_source(index, schema)
        if dest == CENTER:
            continue
        real_factories += 1
        if dest >= NUM_FACTORIES or real_factories > NUM_FACTORIES:
            raise ParseError(
                f"deal lists {len(value)} slots, more than the 2-player game's "
                f"{NUM_FACTORIES} factories plus a center (a 3-/4-player table)"
            )
        for raw in _tile_types(tiles, schema, "deal"):
            per_factory[dest].append(raw)
            colors.append(raw)
    return per_factory


def _infer_color_map(raw_colors: Sequence[int]) -> dict[int, int]:
    """Map observed tile ids onto 0..4 in ascending order (last resort)."""
    ids = sorted(set(raw_colors))
    if len(ids) != NUM_COLORS:
        raise ParseError(
            f"log mentions {len(ids)} tile colours {ids}, need {NUM_COLORS} to infer a "
            "mapping — pin LogSchema.color_map (see AZUL_COLOR_MAP)"
        )
    return {cid: i for i, cid in enumerate(ids)}


def observed_color_ids(payload: Any, schema: LogSchema = DEFAULT_SCHEMA) -> list[int]:
    """Every distinct **raw** BGA tile type the log mentions, ascending.

    Used by :func:`ludometer.human.convert.solve_color_map`, which permutes the
    mapping's *values* while keeping its keys (the ids BGA actually sent).
    """
    ids: set[int] = set()
    for _move_id, entry in iter_log_entries(payload):
        name = str(entry.get("type", ""))
        args = entry.get("args") or {}
        if not isinstance(args, dict):
            continue
        if name in schema.select_types:
            raw = schema.arg(args, "color")
            if raw is not None and not isinstance(raw, (list, dict)):
                value = _as_int(raw, "colour")
                if value != schema.marker_tile_type:
                    ids.add(value)
            ids.update(_tile_types(schema.arg(args, "selected"), schema, "selected"))
        elif name in schema.deal_types:
            value = schema.arg(args, "factories")
            if value is not None:
                collected: list[int] = []
                _parse_factories(value, schema, collected)
                ids.update(collected)
    return sorted(ids)


def scores_from_infos(infos: dict[str, Any] | None) -> dict[int, int]:
    """BGA's authoritative final scores, keyed by player id, from ``tableinfos``.

    A real Azul archive log does **not** carry a cumulative-score notification —
    ``endScore`` reports only the final-round *increment* per player. The reported
    final scores instead live in the table metadata, in two redundant places
    (confirmed live 2026-08-17)::

        data.result.player[]          -> {"player_id": "...", "score": "72", ...}
        data.gameResult.rankedTeams[] -> {"players": [{"id": ..., "score": 72}]}

    Either is read here; the map is empty when neither is present (the synthetic
    fixture uses log ``score`` notifications instead, and those still take
    priority — see :func:`parse_log`).
    """
    data = (infos or {}).get("data", infos or {})
    if not isinstance(data, dict):
        return {}
    out: dict[int, int] = {}
    result = data.get("result")
    if isinstance(result, dict):
        players = result.get("player")
        if isinstance(players, list):
            for p in players:
                if isinstance(p, dict) and p.get("player_id") and p.get("score") is not None:
                    try:
                        out[int(p["player_id"])] = int(p["score"])
                    except (TypeError, ValueError):
                        pass
    if not out:
        game_result = data.get("gameResult")
        teams = game_result.get("rankedTeams") if isinstance(game_result, dict) else None
        if isinstance(teams, list):
            for team in teams:
                for p in (team.get("players") or []) if isinstance(team, dict) else []:
                    if isinstance(p, dict) and p.get("id") is not None and p.get("score") is not None:
                        try:
                            out[int(p["id"])] = int(p["score"])
                        except (TypeError, ValueError):
                            pass
    return out


def conceder_from_infos(infos: dict[str, Any] | None) -> int | None:
    """The conceding player id, from ``tableinfos``, for logs with no concede notification.

    A few real concessions (2 of 28 in the first 400-game corpus) carry **no**
    ``playerConcedeGame`` notification at all — the log just stops mid-game. The
    concession is still visible in the table metadata: ``data.result.endgame_reason``
    is ``"normal_concede_end"`` and BGA reports a nominal 1-0, with the **conceder
    holding the 0** (verified against all 26 notification-carrying concessions in
    the same corpus: the zero-score player is the conceder in every one). Returns
    the conceder's player id only when the reason matches and exactly one player
    has score 0; ``None`` otherwise.
    """
    data = (infos or {}).get("data", infos or {})
    if not isinstance(data, dict):
        return None
    result = data.get("result")
    if not isinstance(result, dict) or result.get("endgame_reason") != "normal_concede_end":
        return None
    zeros = []
    players = result.get("player")
    if isinstance(players, list):
        for p in players:
            if isinstance(p, dict) and p.get("player_id") and str(p.get("score")) == "0":
                try:
                    zeros.append(int(p["player_id"]))
                except (TypeError, ValueError):
                    return None
    return zeros[0] if len(zeros) == 1 else None


# ------------------------------------------------------------------------ parse
def parse_log(
    payload: Any,
    table_id: int,
    player_ids: Sequence[int],
    schema: LogSchema = DEFAULT_SCHEMA,
    infos: dict[str, Any] | None = None,
) -> ReplayGame:
    """Parse one table's log into a :class:`ReplayGame`.

    Raises :class:`ParseError` on anything it does not understand — an unknown
    notification type that is not in ``schema.ignore_types`` included. That
    strictness is the point: a silently dropped notification is a silently wrong
    game, and the converter's checks might not catch it.

    Turn pairing: a ``tilesSelected`` opens a turn and the following
    ``tilesPlacedOnLine`` closes it. If a turn is never closed (the next select or
    a round boundary arrives first) the tiles went to the floor line, which is the
    one destination that may not need its own notification.
    """
    raw_colors: list[int] = []
    picks: list[Pick] = []
    deals: list[Deal] = []
    walls: list[WallPlacement] = []
    scores: dict[int, int] = {}
    first_player: int | None = None
    conceded_by: int | None = None
    unknown: dict[str, int] = {}
    pending: dict[str, Any] | None = None
    warnings: list[str] = []

    def flush(move_id: int, dest: int, line_player: int | None = None) -> None:
        """Close the open turn with ``dest``."""
        nonlocal pending
        if pending is None:
            return
        player = pending["player"] if pending["player"] is not None else line_player
        if player is None:
            raise ParseError(f"table {table_id}: a turn has no player id")
        picks.append(
            Pick(
                player_id=int(player),
                source=int(pending["source"]),
                color=int(pending["color"]),
                dest=dest,
                move_id=move_id,
                count=int(pending["count"]),
            )
        )
        pending = None

    for move_id, entry in iter_log_entries(payload):
        name = str(entry.get("type", ""))
        args = entry.get("args") or {}
        if not isinstance(args, dict):
            args = {}

        if name in schema.select_types:
            if pending is not None:  # never closed: the tiles went to the floor
                flush(move_id, FLOOR)
            selected = _tile_types(schema.arg(args, "selected"), schema, "selected")
            raw = schema.arg(args, "color")
            raw_color = (
                _as_int(raw, "colour")
                if raw is not None and not isinstance(raw, (list, dict))
                else (selected[0] if selected else None)
            )
            if raw_color is None:
                raise ParseError(f"table {table_id}: a selection names no colour")
            raw_colors.append(raw_color)
            raw_colors.extend(selected)
            player = schema.arg(args, "player")
            pending = {
                "player": None if player is None else _as_int(player, "pick player"),
                "source": _map_source(schema.arg(args, "source"), schema),
                "color": raw_color,
                "count": len(selected),
            }
        elif name in schema.place_types:
            if pending is None:
                # A `tilesPlacedOnLine` with no open selection and no *coloured*
                # tiles is the first-player-marker going to the floor: BGA emits a
                # standalone placement (marker in `discardedTiles`, `placedTiles`
                # empty, `type` 0) right before the `tilesSelected` of the turn that
                # takes from the center. It is not a pick — our engine assigns the
                # marker to whoever first takes from the center on its own (docs
                # §4.2 row 7) — so skip it. A placement that *does* carry coloured
                # tiles with no selection is a genuinely dropped turn: still fatal.
                placed = _tile_types(schema.arg(args, "placed"), schema, "placed")
                if not placed:
                    continue
                raise ParseError(
                    f"table {table_id} move {move_id}: tiles placed without a selection"
                )
            player = schema.arg(args, "player")
            flush(
                move_id,
                _map_line(schema.arg(args, "dest"), schema),
                None if player is None else _as_int(player, "place player"),
            )
        elif name in schema.deal_types:
            if pending is not None:
                flush(move_id, FLOOR)
            value = schema.arg(args, "factories")
            if value is not None:
                remaining = schema.arg(args, "remaining")
                # Raw tile types for now; they become per-colour counts in the
                # second pass below, once the colour map is settled.
                deals.append(
                    Deal(
                        round_index=len(deals),
                        factories=tuple(
                            tuple(tiles)
                            for tiles in _parse_factories(value, schema, raw_colors)
                        ),
                        remaining=None
                        if remaining is None
                        else _as_int(remaining, "remainingTiles"),
                    )
                )
        elif name in schema.wall_types:
            if pending is not None:
                flush(move_id, FLOOR)
            walls.extend(_wall_placements(args, schema, raw_colors))
        elif name in schema.floor_clear_types:
            if pending is not None:
                flush(move_id, FLOOR)
        elif name in schema.score_types:
            player = schema.arg(args, "player")
            score = schema.arg(args, "score")
            if player is not None and score is not None:
                scores[_as_int(player, "score player")] = _as_int(score, "score")
        elif name in schema.marker_types:
            player = schema.arg(args, "player")
            if player is not None and first_player is None:
                first_player = _as_int(player, "marker player")
        elif name in schema.undo_place_types:
            # Undo a placement: the tiles come back into the hand, so the last
            # committed turn re-opens as a pending selection (a re-placement, or an
            # `undoTakeTiles`, follows). Confirmed always preceded by a placement.
            if pending is not None:
                # a placement was undone while another selection is still open —
                # impossible in BGA's own flow; refuse rather than corrupt state
                raise ParseError(
                    f"table {table_id} move {move_id}: {name} with a selection "
                    "still open"
                )
            if not picks:
                raise ParseError(
                    f"table {table_id} move {move_id}: {name} with no placed turn "
                    "to undo"
                )
            last = picks.pop()
            pending = {
                "player": last.player_id,
                "source": last.source,
                "color": last.color,
                "count": last.count,
            }
        elif name in schema.undo_take_types:
            # Undo a take: whatever selection is currently open is cancelled. It is
            # always preceded by the `tilesSelected` it cancels, or by the
            # `undoSelectLine` that just re-opened the turn, so a selection must be
            # pending here.
            if pending is None:
                raise ParseError(
                    f"table {table_id} move {move_id}: {name} with no open "
                    "selection to undo"
                )
            pending = None
        elif name in schema.concede_types:
            # A resignation ends the game where it stands. Drop any half-made turn
            # (the resigning player never completed it) and stop reading moves — the
            # concession is the last move notification in every real game.
            player = schema.arg(args, "player")
            if player is not None:
                conceded_by = _as_int(player, "concede player")
            pending = None
            break
        elif name in schema.ignore_types:
            continue
        else:
            unknown[name] = unknown.get(name, 0) + 1

    if unknown:
        raise ParseError(
            f"table {table_id}: unknown notification types {sorted(unknown)}; run "
            "`python -m ludometer.human.cli inspect <raw.json.gz>` and extend "
            "LogSchema (docs/HUMAN_GAMES.md §4.4)"
        )
    if conceded_by is None:
        # A concession with no `playerConcedeGame` in the log at all (rare but
        # real): the metadata still records it, and the half-made turn — if any —
        # is dropped exactly as the notification path does.
        conceded_by = conceder_from_infos(infos)
        if conceded_by is not None:
            pending = None
            warnings.append("concession read from tableinfos (no concede notification)")
    if pending is not None:
        flush(0, FLOOR)
        warnings.append("the last turn had no placement notification")

    # Log `score` notifications win when present (the fixture uses them, and the
    # score-mismatch guard depends on reading them); real archive logs carry none,
    # so fall back to the authoritative final scores in `tableinfos`.
    if len(scores) < 2:
        scores = scores_from_infos(infos) or scores

    color_map = schema.color_map or _infer_color_map(raw_colors)
    options = dict(((infos or {}).get("data", infos or {})).get("options", {}) or {})
    return ReplayGame(
        table_id=int(table_id),
        player_ids=tuple(int(p) for p in player_ids),
        picks=tuple(
            Pick(
                player_id=p.player_id,
                source=p.source,
                color=_map_color(p.color, color_map),
                dest=p.dest,
                move_id=p.move_id,
                count=p.count,
            )
            for p in picks
        ),
        deals=tuple(
            Deal(
                round_index=d.round_index,
                factories=tuple(tuple(_counts(f, color_map)) for f in d.factories),
                remaining=d.remaining,
            )
            for d in deals
        ),
        wall_placements=tuple(
            WallPlacement(
                player_id=w.player_id,
                color=_map_color(w.color, color_map),
                row=w.row,
                column=w.column,
            )
            for w in walls
        ),
        first_player=first_player,
        final_scores=scores,
        options=options,
        warnings=tuple(warnings),
        conceded_by=conceded_by,
    )


def _wall_placements(
    args: dict[str, Any], schema: LogSchema, raw_colors: list[int]
) -> list[WallPlacement]:
    """Read ``placeTileOnWall.args.completeLines`` into raw-colour placements."""
    lines = schema.arg(args, "complete_lines")
    if not isinstance(lines, dict):
        return []  # `[]` when nobody completed a line
    out: list[WallPlacement] = []
    for player, record in lines.items():
        if not isinstance(record, dict):
            continue
        tile = schema.arg(record, "placed_tile")
        if not isinstance(tile, dict):
            continue
        raw = _as_int(tile.get("type"), "wall tile type")
        if raw == schema.marker_tile_type:
            continue
        raw_colors.append(raw)
        column = _as_int(tile.get("column"), "wall column")
        if schema.columns_one_based:
            column -= 1  # BGA columns are 1..5; wall_col() is 0-based
        out.append(
            WallPlacement(
                player_id=_as_int(player, "wall player"),
                color=raw,
                row=_map_line(tile.get("line"), schema, "wall row"),
                column=column,
            )
        )
    return out
