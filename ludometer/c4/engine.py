"""Connect Four rules engine — 7 columns, 7x6 board, perfect information.

Bitboard layout is the standard one (John Tromp's): each column owns 7 bits
(6 playable + a sentinel), bit ``col * 7 + row`` with row 0 at the bottom.
``position`` always holds the stones of the player *to move* and ``mask`` all
stones, which makes :meth:`apply` two bit operations and lets the alpha-beta
solver in :mod:`ludometer.solved` share the exact representation.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

__all__ = [
    "ACTION_SPACE",
    "ENCODED_SIZE",
    "HEIGHT",
    "WIDTH",
    "Connect4State",
    "connect4_won",
]

WIDTH = 7
HEIGHT = 6
ACTION_SPACE = WIDTH
# two 42-cell planes: the mover's stones, then the opponent's
ENCODED_SIZE = 2 * WIDTH * HEIGHT

_AUX_BITS = 15
_BOTTOM = tuple(1 << (col * (HEIGHT + 1)) for col in range(WIDTH))
_TOP = tuple(1 << (col * (HEIGHT + 1) + HEIGHT - 1) for col in range(WIDTH))
_FULL = sum(
    1 << (col * (HEIGHT + 1) + row) for col in range(WIDTH) for row in range(HEIGHT)
)


def connect4_won(stones: int) -> bool:
    """Does this bitboard contain four in a row (any direction)?"""
    for shift in (1, HEIGHT + 1, HEIGHT, HEIGHT + 2):  # | — \ /
        m = stones & (stones >> shift)
        if m & (m >> (2 * shift)):
            return True
    return False


class Connect4State:
    """Mutated in place by :meth:`apply`. ``position`` is the mover's stones."""

    ACTION_SPACE: int = ACTION_SPACE
    ENCODED_SIZE: int = ENCODED_SIZE
    num_players: int = 2

    __slots__ = ("finished", "mask", "ply", "position", "scores", "winner")

    def __init__(self) -> None:
        self.position = 0
        self.mask = 0
        self.ply = 0
        self.finished = False
        self.winner: int | None = None
        self.scores: list[int] = [0, 0]

    @classmethod
    def new_game(cls, seed: int = 0, num_players: int = 2) -> Connect4State:
        """``seed`` is accepted for the shared interface; nothing here is random."""
        if num_players != 2:
            raise ValueError("Connect Four is a 2-player game")
        return cls()

    def clone(self) -> Connect4State:
        other = Connect4State.__new__(Connect4State)
        other.position = self.position
        other.mask = self.mask
        other.ply = self.ply
        other.finished = self.finished
        other.winner = self.winner
        other.scores = list(self.scores)
        return other

    # ------------------------------------------------------------------ rules
    @property
    def is_terminal(self) -> bool:
        return self.finished

    @property
    def current_player(self) -> int:
        return self.ply & 1

    @property
    def round_index(self) -> int:
        return self.ply

    def legal_actions(self) -> list[int]:
        if self.finished:
            return []
        return [c for c in range(WIDTH) if not self.mask & _TOP[c]]

    def is_legal(self, action_id: int) -> bool:
        return (
            not self.finished
            and 0 <= action_id < WIDTH
            and not self.mask & _TOP[action_id]
        )

    def apply(self, action_id: int) -> None:
        if not self.is_legal(action_id):
            raise ValueError(f"illegal action {action_id}")
        new_mask = self.mask | (self.mask + _BOTTOM[action_id])
        moved = self.position | (new_mask ^ self.mask)
        if connect4_won(moved):
            self.finished = True
            self.winner = self.current_player
            self.scores[self.winner] = 1
        elif new_mask == _FULL:
            self.finished = True
        self.position = moved ^ new_mask  # the next mover's stones
        self.mask = new_mask
        self.ply += 1

    def outcome(self) -> float | None:
        if not self.finished:
            return None
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == 0 else -1.0

    # ---------------------------------------------------- search integration
    def is_stochastic(self, action_id: int) -> bool:
        return False

    def determinize(self, action_id: int, seed: int) -> Connect4State:
        raise RuntimeError("Connect Four has no chance events")  # pragma: no cover

    def chance_key(self) -> bytes:
        return self.position.to_bytes(8, "little") + self.mask.to_bytes(8, "little")

    def fingerprint(self) -> tuple[Any, ...]:
        return (self.position, self.mask, self.ply)

    def search_root(self, rng: random.Random) -> Connect4State:
        return self.clone()

    def wall_summary(self, player: int) -> list[int]:
        return [0] * _AUX_BITS

    # --------------------------------------------------------------- encoding
    def encode(self) -> np.ndarray:
        out = np.zeros(ENCODED_SIZE, dtype=np.float32)
        mine = self.position
        theirs = self.position ^ self.mask
        cell = 0
        for col in range(WIDTH):
            base = col * (HEIGHT + 1)
            for row in range(HEIGHT):
                bit = 1 << (base + row)
                if mine & bit:
                    out[cell] = 1.0
                elif theirs & bit:
                    out[42 + cell] = 1.0
                cell += 1
        return out

    # ------------------------------------------------------------- reporting
    def render_text(self) -> str:
        first, second = (
            (self.position, self.position ^ self.mask)
            if self.ply % 2 == 0
            else (self.position ^ self.mask, self.position)
        )
        rows = []
        for row in range(HEIGHT - 1, -1, -1):
            line = []
            for col in range(WIDTH):
                bit = 1 << (col * (HEIGHT + 1) + row)
                line.append("X" if first & bit else "O" if second & bit else ".")
            rows.append(" ".join(line))
        return "\n".join(rows)

    def to_json(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "mask": self.mask,
            "ply": self.ply,
            "finished": self.finished,
            "winner": self.winner,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Connect4State ply={self.ply}>"
