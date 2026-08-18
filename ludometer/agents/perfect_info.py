"""Baselines for the solved perfect-information games (tic-tac-toe, Connect 4).

The greedy rung is game-generic: take an immediate win, otherwise avoid any
move that hands the opponent an immediate win (which subsumes "block their
threat" — every non-blocking move leaves it standing). The heuristic rung adds
the one piece of positional lore each game is known for: fork creation for
tic-tac-toe, centre preference for Connect Four.
"""

from __future__ import annotations

import random

from ludometer.agents.base import Agent
from ludometer.c4.engine import Connect4State
from ludometer.ttt.engine import WIN_MASKS, TicTacToeState

__all__ = [
    "C4GreedyAgent",
    "C4HeuristicAgent",
    "C4RandomAgent",
    "TTTGreedyAgent",
    "TTTHeuristicAgent",
    "TTTRandomAgent",
]


class _TwoPlyAgent(Agent):
    """Win now if possible; never volunteer an immediate loss; then prefer()."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def seed(self, n: int) -> None:
        self.rng.seed(n)

    def act(self, state) -> int:
        legal = state.legal_actions()
        me = state.current_player
        wins, safe = [], []
        for action in legal:
            child = state.clone()
            child.apply(action)
            outcome = child.outcome()
            if outcome is not None and outcome != 0.0 and (outcome > 0) == (me == 0):
                wins.append(action)
                continue
            if child.is_terminal or not self._opponent_can_win(child):
                safe.append(action)
        if wins:
            return self.prefer(state, wins)
        return self.prefer(state, safe or legal)

    @staticmethod
    def _opponent_can_win(state) -> bool:
        opp = state.current_player
        for reply in state.legal_actions():
            child = state.clone()
            child.apply(reply)
            outcome = child.outcome()
            if outcome is not None and outcome != 0.0 and (outcome > 0) == (opp == 0):
                return True
        return False

    def prefer(self, state, options: list[int]) -> int:
        return self.rng.choice(options)


# ---------------------------------------------------------------- tic-tac-toe
class TTTRandomAgent(Agent):
    name = "ttt:random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def seed(self, n: int) -> None:
        self.rng.seed(n)

    def act(self, state: TicTacToeState) -> int:
        return self.rng.choice(state.legal_actions())


class TTTGreedyAgent(_TwoPlyAgent):
    name = "ttt:greedy"


class TTTHeuristicAgent(_TwoPlyAgent):
    """Greedy plus fork detection, then centre, then corners."""

    name = "ttt:heuristic"

    def prefer(self, state: TicTacToeState, options: list[int]) -> int:
        def threats(cell: int) -> int:
            mine = state.me | 1 << cell
            free = ~(mine | state.them)
            return sum(
                1
                for m in WIN_MASKS
                if not m & state.them and bin(m & mine).count("1") == 2 and m & free
            )

        best = max(threats(c) for c in options)
        pool = [c for c in options if threats(c) == best]
        for pick in (4, 0, 2, 6, 8):
            if pick in pool:
                return pick
        return self.rng.choice(pool)


# --------------------------------------------------------------- Connect Four
_CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)


class C4RandomAgent(Agent):
    name = "c4:random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def seed(self, n: int) -> None:
        self.rng.seed(n)

    def act(self, state: Connect4State) -> int:
        return self.rng.choice(state.legal_actions())


class C4GreedyAgent(_TwoPlyAgent):
    name = "c4:greedy"


class C4HeuristicAgent(_TwoPlyAgent):
    """Greedy with the classic centre-out column preference."""

    name = "c4:heuristic"

    def prefer(self, state: Connect4State, options: list[int]) -> int:
        for col in _CENTER_ORDER:
            if col in options:
                return col
        return self.rng.choice(options)  # pragma: no cover - options never empty
