"""Uniform random baseline — the Elo anchor at 0 (see docs/DESIGN.md)."""

from __future__ import annotations

import random

from ludometer.agents.base import Agent
from ludometer.azul.engine import AzulState

__all__ = ["RandomAgent"]


class RandomAgent(Agent):
    """Picks uniformly among the legal actions."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def seed(self, n: int) -> None:
        self.rng.seed(n)

    def act(self, state: AzulState) -> int:
        actions = state.legal_actions()
        if not actions:
            raise ValueError("no legal actions (terminal state?)")
        return actions[self.rng.randrange(len(actions))]
