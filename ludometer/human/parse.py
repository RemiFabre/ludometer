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
    #: **Unconfirmed**: how the log names the center pile in ``fromFactory``. With
    #: 0-based displays (``factory_0`` ... ``factory_4`` for two players) the
    #: natural encoding is the next index up, which is why any source ``>=
    #: NUM_FACTORIES`` is read as the center; these explicit values are the escape
    #: hatch for a negative or sentinel encoding.
    center_values: tuple[int, ...] = (-1, 99)
    #: **Unconfirmed**: how the log names the floor line in ``line``. With 1-based
    #: pattern lines, 0 is the floor; a wrong guess here makes moves illegal or
    #: scores mismatch, so the converter will catch it — see docs §8.
    floor_values: tuple[int, ...] = (0, -1, 6, 9)
    #: **Unconfirmed**: ``True`` when the log's pattern lines are 1..5 rather than
    #: 0..4. Applies to ``tilesPlacedOnLine.line`` and to the wall placement's
    #: ``line``.
    lines_one_based: bool = True
    #: ``True`` when factory ids in the log are 1..5 rather than 0..4. The
    #: ``location: "factory_0"`` strings say 0-based, so this defaults to False.
    factories_one_based: bool = False
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
    for index, tiles in enumerate(value):
        if index >= NUM_FACTORIES:
            raise ParseError(
                f"deal lists {len(value)} factories, the 2-player game has "
                f"{NUM_FACTORIES} (a 3- or 4-player table has more)"
            )
        for raw in _tile_types(tiles, schema, "deal"):
            per_factory[index].append(raw)
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
    if pending is not None:
        flush(0, FLOOR)
        warnings.append("the last turn had no placement notification")

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
        out.append(
            WallPlacement(
                player_id=_as_int(player, "wall player"),
                color=raw,
                row=_map_line(tile.get("line"), schema, "wall row"),
                column=_as_int(tile.get("column"), "wall column"),
            )
        )
    return out
