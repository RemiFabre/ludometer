"""Tic-tac-toe rules engine — 9 actions, perfect information, no chance.

The point of this game in the study is *calibration*: it is solved, tiny and
obviously shallow, so the ludometer must report it as such (NEXT_GAMES.md §3).
The engine keeps the exact duck-typed surface the trainer expects; every
chance hook is the trivial one because nothing here is ever random.

Cells are ``0..8``, row-major:

    0 1 2
    3 4 5
    6 7 8
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

__all__ = ["ACTION_SPACE", "ENCODED_SIZE", "WIN_MASKS", "TicTacToeState"]

ACTION_SPACE = 9
# two 9-cell planes: the mover's stones, then the opponent's
ENCODED_SIZE = 18

WIN_MASKS = tuple(
    sum(1 << c for c in line)
    for line in (
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
        (0, 4, 8), (2, 4, 6),  # diagonals
    )
)
FULL = (1 << 9) - 1

_AUX_BITS = 15


def _won(stones: int) -> bool:
    return any(stones & m == m for m in WIN_MASKS)


class TicTacToeState:
    """Mutated in place by :meth:`apply`. ``me`` is always the player to move."""

    ACTION_SPACE: int = ACTION_SPACE
    ENCODED_SIZE: int = ENCODED_SIZE
    num_players: int = 2

    __slots__ = ("finished", "me", "ply", "scores", "them", "winner")

    def __init__(self) -> None:
        self.me = 0  # bitboard of the player to move
        self.them = 0
        self.ply = 0
        self.finished = False
        self.winner: int | None = None
        self.scores: list[int] = [0, 0]

    @classmethod
    def new_game(cls, seed: int = 0, num_players: int = 2) -> TicTacToeState:
        """``seed`` is accepted for the shared interface; nothing here is random."""
        if num_players != 2:
            raise ValueError("tic-tac-toe is a 2-player game")
        return cls()

    def clone(self) -> TicTacToeState:
        other = TicTacToeState.__new__(TicTacToeState)
        other.me = self.me
        other.them = self.them
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
        taken = self.me | self.them
        return [c for c in range(9) if not taken >> c & 1]

    def is_legal(self, action_id: int) -> bool:
        return (
            not self.finished
            and 0 <= action_id < 9
            and not (self.me | self.them) >> action_id & 1
        )

    def apply(self, action_id: int) -> None:
        if not self.is_legal(action_id):
            raise ValueError(f"illegal action {action_id}")
        stones = self.me | 1 << action_id
        if _won(stones):
            self.finished = True
            self.winner = self.current_player
            self.scores[self.winner] = 1
        elif stones | self.them == FULL:
            self.finished = True
        self.me, self.them = self.them, stones
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

    def determinize(self, action_id: int, seed: int) -> TicTacToeState:
        raise RuntimeError("tic-tac-toe has no chance events")  # pragma: no cover

    def chance_key(self) -> bytes:
        return self.me.to_bytes(2, "little") + self.them.to_bytes(2, "little")

    def fingerprint(self) -> tuple[Any, ...]:
        return (self.me, self.them, self.ply)

    def search_root(self, rng: random.Random) -> TicTacToeState:
        return self.clone()

    def wall_summary(self, player: int) -> list[int]:
        return [0] * _AUX_BITS

    # --------------------------------------------------------------- encoding
    def encode(self) -> np.ndarray:
        out = np.zeros(ENCODED_SIZE, dtype=np.float32)
        for c in range(9):
            if self.me >> c & 1:
                out[c] = 1.0
            elif self.them >> c & 1:
                out[9 + c] = 1.0
        return out

    # ------------------------------------------------------------- reporting
    def render_text(self) -> str:
        # X always moved first; whose stones are in `me` depends on the ply
        x, o = (self.me, self.them) if self.ply % 2 == 0 else (self.them, self.me)
        rows = []
        for r in range(3):
            rows.append(
                " ".join(
                    "X" if x >> (3 * r + c) & 1 else "O" if o >> (3 * r + c) & 1 else "."
                    for c in range(3)
                )
            )
        return "\n".join(rows)

    def to_json(self) -> dict[str, Any]:
        return {
            "me": self.me,
            "them": self.them,
            "ply": self.ply,
            "finished": self.finished,
            "winner": self.winner,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<TicTacToeState ply={self.ply}>"
