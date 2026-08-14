"""Greedy 1-ply baseline: maximise your own immediate placement value.

For every legal action it clones the state, applies the action and scores the
result with :func:`~ludometer.agents.features.immediate_value` — banked score
plus what the player's complete pattern lines are already worth, minus floor
damage. Because ``apply`` resolves the round boundary itself, the last move of a
round is evaluated on the *tiled* board, which is exactly what a 1-ply greedy
player wants to see.

No lookahead beyond that: no opponent modelling, no future-round planning.
"""

from __future__ import annotations

import random

from ludometer.agents.base import Agent
from ludometer.agents.features import immediate_value
from ludometer.azul.engine import AzulState

__all__ = ["GreedyAgent"]


class GreedyAgent(Agent):
    """1-ply immediate-score maximiser; ties are broken uniformly at random."""

    name = "greedy"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def seed(self, n: int) -> None:
        self.rng.seed(n)

    def act(self, state: AzulState) -> int:
        actions = state.legal_actions()
        if not actions:
            raise ValueError("no legal actions (terminal state?)")
        me = state.current_player
        best: list[int] = []
        best_value = -1e18
        for action in actions:
            child = state.clone()
            child.apply(action)
            value = immediate_value(child, me)
            if child.is_terminal:
                # a finished game is worth what it is worth: settle it on score
                value += 100.0 * (child.scores[me] - child.scores[1 - me] > 0)
            if value > best_value:
                best_value = value
                best = [action]
            elif value == best_value:
                best.append(action)
        if len(best) == 1:
            return best[0]
        return best[self.rng.randrange(len(best))]
