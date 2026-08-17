"""Build a fake BGA replay log out of a game our own engine plays.

This is the round-trip harness for the whole parse -> convert -> dataset chain
without touching the network: play a real game in the engine, write it out in the
JSON shape ``/archive/archive/logs.html`` returns (as documented in
:mod:`ludometer.human.parse`), then feed that back through the parser and require
the converter to reproduce the *same* moves, the same scores and the same outcome.

It writes the notifications Azul really uses — ``factoriesFilled``,
``tilesSelected`` + ``tilesPlacedOnLine`` per turn, ``placeTileOnWall`` at each
round end — with real tile objects (``{id, type, column, line, location}``) and
BGA's tile-type numbering (:data:`~ludometer.human.parse.AZUL_COLOR_MAP`), so it
exercises the same code path a downloaded log will:

* factories are **0-based** (``location: "factory_0"``), and source
  ``NUM_FACTORIES`` (5, past the last display) means the center;
* pattern lines are **1-based** in ``line``, ``0`` meaning the floor line;
* tile ``type`` is ``1`` Black, ``2`` Cyan, ``3`` Blue, ``4`` Yellow, ``5`` Red,
  and ``0`` is the first-player marker;
* each wall placement carries the fixed wall's ``column``, i.e.
  ``(colour + row) % 5``.

The last three are confirmed from a working third-party Azul parser; the center
and floor encodings are the parts still to verify (``docs/HUMAN_GAMES.md`` §8).
The point of the fixture is that the *code* does not care: change
:class:`~ludometer.human.parse.LogSchema` and the same tests still pass.
"""

from __future__ import annotations

import random
from typing import Any

from ludometer.azul.engine import (
    CENTER,
    FLOOR,
    NUM_COLORS,
    NUM_FACTORIES,
    AzulState,
    decode_action,
)
from ludometer.human.parse import AZUL_COLOR_MAP

__all__ = ["SyntheticGame", "synthetic_log"]


class SyntheticGame:
    """A finished engine game plus its fake-log rendering."""

    def __init__(
        self, seed: int, player_ids: tuple[int, int], swap_seats: bool = False
    ):
        self.seed = int(seed)
        self.player_ids = tuple(int(p) for p in player_ids)
        #: when True, engine seat 0 is ``player_ids[1]`` — exercises the converter's
        #: "whoever moved first is not necessarily seat 0" path.
        self.swap_seats = bool(swap_seats)
        self.actions: list[int] = []
        self.movers: list[int] = []
        self.deals: list[list[list[int]]] = []
        #: per round end, the ``(seat, colour, row, column)`` tiles the wall-tiling
        #: step placed — what a real log reports as ``placeTileOnWall``.
        self.wall_events: list[list[tuple[int, int, int, int]]] = []
        self.scores: tuple[int, int] = (0, 0)
        self.outcome: float = 0.0
        self._play()

    # ------------------------------------------------------------------ playing
    def _snapshot_deal(self, state: AzulState) -> None:
        """Record the current factories as a per-factory list of colour ids."""
        self.deals.append(
            [
                [c for c in range(NUM_COLORS) for _ in range(factory[c])]
                for factory in state.factories
            ]
        )

    def _play(self) -> None:
        rng = random.Random(self.seed ^ 0x5EED)
        state = AzulState.new_game(seed=self.seed)
        self._snapshot_deal(state)
        while not state.is_terminal:
            legal = state.legal_actions()
            if not legal:  # pragma: no cover - only if the engine is broken
                raise RuntimeError("no legal actions in a non-terminal state")
            action = rng.choice(legal)
            self.actions.append(action)
            self.movers.append(state.current_player)
            round_before = state.round_index
            walls_before = [wall[:] for wall in state.walls]
            state.apply(action)
            ended = state.round_index > round_before or state.is_terminal
            if ended:
                # Which tiles the wall-tiling step just placed, read off the wall
                # diff: cell r*5+col holds colour (col - r) % 5 on the fixed wall.
                placed: list[tuple[int, int, int, int]] = []
                for seat in range(2):
                    after = state.walls[seat]
                    for index in range(25):
                        if after[index] and not walls_before[seat][index]:
                            row, col = divmod(index, 5)
                            placed.append((seat, (col - row) % NUM_COLORS, row, col))
                self.wall_events.append(placed)
            if state.round_index > round_before and not state.is_terminal:
                self._snapshot_deal(state)
        self.scores = (int(state.scores[0]), int(state.scores[1]))
        self.outcome = float(state.outcome() or 0.0)

    # ------------------------------------------------------------------ log JSON
    def seat_to_player(self, seat: int) -> int:
        order = tuple(reversed(self.player_ids)) if self.swap_seats else self.player_ids
        return order[seat]

    def log_payload(self, table_id: int = 999_000_001) -> dict[str, Any]:
        """The fake ``logs.html`` envelope for this game."""
        packets: list[dict[str, Any]] = []
        move_id = 1

        def packet(entry: dict[str, Any], channel: str | None = None) -> None:
            nonlocal move_id
            packets.append(
                {
                    "channel": channel or f"/table/t{table_id}",
                    "table_id": str(table_id),
                    "packet_id": str(len(packets) + 1),
                    "packet_type": "resend",
                    "move_id": str(move_id),
                    "time": 1_700_000_000 + move_id,
                    "data": [entry],
                }
            )
            move_id += 1

        deals = iter(self.deals)
        wall_events = iter(self.wall_events)
        packet(_deal_entry(next(deals), 0))
        round_index = 0
        state = AzulState.new_game(seed=self.seed)  # re-walk to know where rounds end
        for action, seat in zip(self.actions, self.movers, strict=True):
            source, color, dest = decode_action(action)
            player = str(self.seat_to_player(seat))
            taken = (
                state.center[color]
                if source == CENTER
                else state.factories[source][color]
            )
            # A turn is two notifications: took tiles, then placed them.
            packet(
                {
                    "uid": f"u{move_id}",
                    "type": "tilesSelected",
                    "log": "${player_name} takes ${number} ${color} tiles",
                    "args": {
                        "player_id": player,
                        "fromFactory": NUM_FACTORIES if source == CENTER else source,
                        "type": _bga_type(color),
                        "selectedTiles": [
                            _tile(color, f"factory_{source}") for _ in range(taken)
                        ],
                        "discardedTiles": [],
                    },
                }
            )
            packet(
                {
                    "uid": f"u{move_id}",
                    "type": "tilesPlacedOnLine",
                    "log": "${player_name} places tiles",
                    "args": {
                        "player_id": player,
                        "line": 0 if dest == FLOOR else dest + 1,
                        "placedTiles": [_tile(color, "line") for _ in range(taken)],
                        "discardedTiles": [],
                    },
                }
            )
            round_before = state.round_index
            state.apply(action)
            if state.round_index > round_before or state.is_terminal:
                for wall_seat, wall_color, row, column in next(wall_events, []):
                    packet(
                        {
                            "uid": f"u{move_id}",
                            "type": "placeTileOnWall",
                            "log": "",
                            "args": {
                                "completeLines": {
                                    str(self.seat_to_player(wall_seat)): {
                                        "placedTile": {
                                            "id": 1,
                                            "type": _bga_type(wall_color),
                                            "column": column,
                                            "line": row + 1,
                                            "location": "wall",
                                        },
                                        "discardedTiles": [],
                                        "pointsDetail": {},
                                    }
                                }
                            },
                        }
                    )
            if state.round_index > round_before and not state.is_terminal:
                round_index += 1
                packet(_deal_entry(next(deals), round_index))
        for seat in (0, 1):
            packet(
                {
                    "uid": f"u{move_id}",
                    "type": "score",
                    "log": "${player_name} scores ${score}",
                    "args": {
                        "player_id": str(self.seat_to_player(seat)),
                        "score": self.scores[seat],
                    },
                }
            )
        # One packet on a PRIVATE player channel, which a real log also contains:
        # per-player UI hints, plus (as here) a duplicate of a move notification that
        # only that client was told about. `iter_log_entries` must drop it — if it
        # ever stops doing so, this fixture produces an extra bogus pick and the
        # round-trip tests fail, which is exactly the alarm we want.
        packet(
            {
                "uid": "private",
                "type": "updateMoves",
                "log": "",
                "args": {"player_id": str(self.seat_to_player(0)), "possibleMoves": {}},
            },
            channel=f"/player/p{self.seat_to_player(0)}",
        )
        # Confirmed envelope: the packet list lives at data.data, with a `valid` flag.
        return {"status": 1, "data": {"valid": 1, "data": packets}}

    def infos_payload(self, table_id: int = 999_000_001) -> dict[str, Any]:
        """A fake ``tableinfos`` payload shaped like BGA's, for the filter tests."""
        return {
            "status": 1,
            "data": {
                "id": str(table_id),
                "game_id": "1467",
                "status": "finished",
                "players": {
                    str(pid): {
                        "id": str(pid),
                        "player_elo": "2050",
                        "score": str(score),
                    }
                    for pid, score in zip(
                        (self.seat_to_player(0), self.seat_to_player(1)),
                        self.scores,
                        strict=True,
                    )
                },
                "options": {},
            },
        }


#: engine colour -> BGA tile ``type``, the inverse of
#: :data:`~ludometer.human.parse.AZUL_COLOR_MAP`.
_ENGINE_TO_BGA = {engine: bga for bga, engine in AZUL_COLOR_MAP.items()}


def _bga_type(color: int) -> int:
    return _ENGINE_TO_BGA[int(color)]


def _tile(color: int, location: str, tile_id: int = 0) -> dict[str, Any]:
    """A BGA tile object as the real log carries them."""
    return {
        "id": tile_id,
        "type": _bga_type(color),
        "column": 0,
        "line": 0,
        "location": location,
    }


def _deal_entry(factories: list[list[int]], round_index: int) -> dict[str, Any]:
    return {
        "uid": f"deal{round_index}",
        "type": "factoriesFilled",
        "log": "",
        "args": {
            "remainingTiles": None,
            "factories": [
                [_tile(color, f"factory_{index}") for color in factory]
                for index, factory in enumerate(factories)
            ],
        },
    }


def synthetic_log(
    seed: int = 7,
    player_ids: tuple[int, int] = (91843016, 91718783),
    table_id: int = 999_000_001,
    swap_seats: bool = False,
) -> tuple[SyntheticGame, dict[str, Any], dict[str, Any]]:
    """``(game, log_payload, infos_payload)`` for one synthetic table."""
    game = SyntheticGame(seed, player_ids, swap_seats=swap_seats)
    return game, game.log_payload(table_id), game.infos_payload(table_id)
