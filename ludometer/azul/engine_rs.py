"""`AzulState` on the Rust rules: an API-compatible wrapper over ``ludometer_rs.State``.

Selected by ``LUDOMETER_ENGINE=rust`` (see :func:`ludometer.games.get_game`) or
built directly; the default stays the Python engine. Same method names and
return types as :class:`ludometer.azul.engine.AzulState`, and the same values —
``tests/test_rust_engine.py`` checks every observable on 10,000 random games and
every BGA replay.

One difference: attributes like ``factories`` or ``walls`` are *copies* read
out of the Rust struct, so ``state.factories[0][0] = 1`` edits a copy. Edit a
position through :meth:`to_dict` / :meth:`from_dict` instead (the tests do).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ludometer.azul.engine import (
    ACTION_SPACE,
    COLOR_CHARS,
    COLOR_NAMES,
    ENCODED_SIZE,
    NUM_COLORS,
    NUM_ROWS,
)

__all__ = ["AzulState", "available", "to_rust"]

_STATE_KEYS = (
    "factories",
    "center",
    "marker_in_center",
    "bag",
    "lid",
    "walls",
    "pl_color",
    "pl_count",
    "floor",
    "floor_marker",
    "scores",
    "current_player",
    "first_player",
    "round_index",
    "is_terminal",
    "exhausted",
)


def available() -> bool:
    try:
        import ludometer_rs  # noqa: F401
    except ImportError:
        return False
    return True


def _rs() -> Any:
    import ludometer_rs

    return ludometer_rs


def to_rust(state: Any, rng: str = "fast") -> Any:
    """A ``ludometer_rs.State`` for any Azul state object (Python, wrapper or Rust).

    The RNG stream is not carried over from a Python state: only the search
    ever consumes a state's generator (through determinizations, which reseed),
    so this is exact for search roots.
    """
    rs = _rs()
    if isinstance(state, rs.State):
        return state
    inner = getattr(state, "_rs", None)
    if inner is not None:
        return inner
    return rs.State.from_dict({k: getattr(state, k) for k in _STATE_KEYS}, rng=rng)


class AzulState:
    """Azul on the Rust engine, with the Python engine's interface."""

    ACTION_SPACE: int = ACTION_SPACE
    ENCODED_SIZE: int = ENCODED_SIZE
    num_players: int = 2

    __slots__ = ("_rs",)

    def __init__(self, inner: Any) -> None:
        self._rs = inner

    # ------------------------------------------------------------------ setup
    @classmethod
    def new_game(cls, seed: int, num_players: int = 2, rng: str = "fast") -> AzulState:
        if num_players != 2:
            raise ValueError("only the 2-player game is implemented")
        return cls(_rs().State.new_game(int(seed), rng=rng))

    @classmethod
    def from_dict(cls, data: dict[str, Any], rng: str = "fast") -> AzulState:
        return cls(_rs().State.from_dict(data, rng=rng))

    @classmethod
    def from_python(cls, state: Any, rng: str = "fast") -> AzulState:
        return cls(to_rust(state, rng=rng))

    def to_dict(self) -> dict[str, Any]:
        return self._rs.to_dict()

    def clone(self) -> AzulState:
        return AzulState(self._rs.clone())

    def recount(self) -> None:
        self._rs.recount()

    # ------------------------------------------------------------ legal moves
    def legal_actions(self) -> list[int]:
        return self._rs.legal_actions()

    def is_legal(self, action_id: int) -> bool:
        return self._rs.is_legal(int(action_id))

    def apply(self, action_id: int) -> None:
        self._rs.apply(int(action_id))

    # ------------------------------------------------------------ inspection
    def floor_occupied(self, player: int) -> int:
        return self._rs.floor_occupied(player)

    def floor_penalty(self, player: int) -> int:
        return self._rs.floor_penalty(player)

    def completed_rows(self, player: int) -> int:
        return self._rs.completed_rows(player)

    def completed_cols(self, player: int) -> int:
        return self._rs.completed_cols(player)

    def completed_colors(self, player: int) -> int:
        return self._rs.completed_colors(player)

    def wall_summary(self, player: int) -> list[int]:
        return self._rs.wall_summary(player)

    def outcome(self) -> float | None:
        return self._rs.outcome()

    def bag_counts(self) -> list[int]:
        return self._rs.bag_counts()

    def tile_census(self) -> list[int]:
        return self._rs.tile_census()

    # ---------------------------------------------------- search integration
    def is_stochastic(self, action_id: int) -> bool:
        return self._rs.is_stochastic(int(action_id))

    def determinize(self, action_id: int, seed: int) -> AzulState:
        return AzulState(self._rs.determinize(int(action_id), int(seed)))

    def chance_key(self) -> bytes:
        return self._rs.chance_key()

    def fingerprint(self) -> tuple[Any, ...]:
        return self._rs.fingerprint()

    def search_root(self, rng: Any) -> AzulState:
        return self.clone()

    def apply_deal(self, factories: list[list[int]]) -> None:
        """The BGA hook: replace the refill just made with the observed deal."""
        self._rs.apply_deal([list(f) for f in factories])

    # --------------------------------------------------------------- encoding
    def encode(self) -> np.ndarray:
        return self._rs.encode()

    # ------------------------------------------------------------- attributes
    @property
    def rng_kind(self) -> str:
        return self._rs.rng_kind

    @property
    def current_player(self) -> int:
        return self._rs.current_player

    @current_player.setter
    def current_player(self, value: int) -> None:
        self._rs.current_player = int(value)

    @property
    def first_player(self) -> int:
        return self._rs.first_player

    @first_player.setter
    def first_player(self, value: int) -> None:
        self._rs.first_player = int(value)

    @property
    def round_index(self) -> int:
        return self._rs.round_index

    @property
    def tiles_left(self) -> int:
        return self._rs.tiles_left

    @property
    def is_terminal(self) -> bool:
        return self._rs.is_terminal

    @property
    def exhausted(self) -> bool:
        return self._rs.exhausted

    @property
    def marker_in_center(self) -> bool:
        return self._rs.marker_in_center

    @property
    def scores(self) -> list[int]:
        return self._rs.scores

    @property
    def factories(self) -> list[list[int]]:
        return self._rs.factories

    @property
    def center(self) -> list[int]:
        return self._rs.center

    @property
    def lid(self) -> list[int]:
        return self._rs.lid

    @property
    def bag(self) -> list[int]:
        return self._rs.bag

    @property
    def walls(self) -> list[list[int]]:
        return self._rs.walls

    @property
    def pl_color(self) -> list[list[int]]:
        return self._rs.pl_color

    @property
    def pl_count(self) -> list[list[int]]:
        return self._rs.pl_count

    @property
    def floor(self) -> list[list[int]]:
        return self._rs.floor

    @property
    def floor_marker(self) -> list[bool]:
        return self._rs.floor_marker

    # ---------------------------------------------------------------- display
    def render_text(self) -> str:
        s = self._rs
        lines = [
            f"round {s.round_index}  to move: P{s.current_player}  "
            f"first player: P{s.first_player}  scores: {s.scores[0]}-{s.scores[1]}"
            + ("  [GAME OVER]" if s.is_terminal else "")
        ]
        lines.append("Factories:")
        for i, f in enumerate(s.factories):
            tiles = "".join(COLOR_CHARS[c] * f[c] for c in range(NUM_COLORS))
            lines.append(f"  {i}: {tiles if tiles else '-'}")
        cen = "".join(COLOR_CHARS[c] * s.center[c] for c in range(NUM_COLORS))
        marker = " [1st]" if s.marker_in_center else ""
        lines.append(f"  center: {cen if cen else '-'}{marker}")
        bag = s.bag_counts()
        lines.append(
            "  bag: "
            + " ".join(f"{COLOR_CHARS[c]}{bag[c]}" for c in range(NUM_COLORS))
            + "   lid: "
            + " ".join(f"{COLOR_CHARS[c]}{s.lid[c]}" for c in range(NUM_COLORS))
        )
        walls, plc_all, pln_all, floors = s.walls, s.pl_color, s.pl_count, s.floor
        for p in range(2):
            mark = "*" if p == s.current_player else " "
            lines.append(f"{mark}P{p}  score {s.scores[p]}")
            wall, plc, pln = walls[p], plc_all[p], pln_all[p]
            for r in range(NUM_ROWS):
                cap = r + 1
                n = pln[r]
                filled = COLOR_CHARS[plc[r]] * n if n else ""
                line = ("." * (cap - n) + filled).rjust(5)
                row = "".join(
                    COLOR_CHARS[(col - r) % NUM_COLORS] if wall[r * 5 + col] else "."
                    for col in range(5)
                )
                lines.append(f"    {line} | {row}")
            fl = floors[p]
            floor = "".join(COLOR_CHARS[c] * fl[c] for c in range(NUM_COLORS))
            if s.floor_marker[p]:
                floor += "#"
            lines.append(f"    floor: {floor if floor else '-'} ({s.floor_penalty(p)})")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        s = self._rs
        walls, plc, pln, floors, fm = (
            s.walls,
            s.pl_color,
            s.pl_count,
            s.floor,
            s.floor_marker,
        )
        return {
            "round": s.round_index,
            "current_player": s.current_player,
            "first_player": s.first_player,
            "factories": s.factories,
            "center": s.center,
            "marker_in_center": s.marker_in_center,
            "bag": s.bag_counts(),
            "lid": s.lid,
            "tiles_left": s.tiles_left,
            "scores": s.scores,
            "is_terminal": s.is_terminal,
            "exhausted": s.exhausted,
            "outcome": s.outcome(),
            "legal_actions": s.legal_actions(),
            "color_names": list(COLOR_NAMES),
            "players": [
                {
                    "score": s.scores[p],
                    "wall": [walls[p][r * 5 : r * 5 + 5] for r in range(NUM_ROWS)],
                    "pattern_lines": [
                        {"capacity": r + 1, "color": plc[p][r], "count": pln[p][r]}
                        for r in range(NUM_ROWS)
                    ],
                    "floor": floors[p],
                    "floor_marker": fm[p],
                    "floor_penalty": s.floor_penalty(p),
                    "completed_rows": s.completed_rows(p),
                    "completed_cols": s.completed_cols(p),
                    "completed_colors": s.completed_colors(p),
                }
                for p in range(2)
            ],
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        s = self._rs
        return (
            f"<AzulState[rust] round={s.round_index} player={s.current_player} "
            f"scores={s.scores} terminal={s.is_terminal}>"
        )
