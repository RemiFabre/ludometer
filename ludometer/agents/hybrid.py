"""An agent that plays one policy for the first moves of round 1 and hands over.

The task force's E2 (``docs/STRATEGY_TASKFORCE.md``): an imitation net trained
on the experts' round-1 decisions chooses the opening, a strong searching agent
plays everything after. Nothing in the strong agent is retrained, so a gain or
a loss against the plain strong agent is attributable to the opening choice.

Spec (``ludometer.agents.registry``)::

    hybrid:<opening ckpt>|<main ckpt>?k=3&think=1.0&engine=rust[&temp=0]

``k`` counts *this agent's own* decisions in round 1 (``k=all`` = the whole
first round); the opening policy is the raw policy head, argmax over the legal
moves (``temp>0`` samples it instead). ``think``/``sims``/``engine`` go to the
main agent. The counter resets on :meth:`seed`, which the arena calls before
every game, and also whenever the position shows a new game (round 0, empty
walls, no move yet: that is only true at the very start).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ludometer.agents.base import Agent
from ludometer.azul.engine import AzulState

__all__ = ["HybridOpeningAgent", "PolicyAgent"]


class PolicyAgent(Agent):
    """The raw policy head: argmax (or a temperature sample) over the legal moves."""

    def __init__(self, net: Any, temperature: float = 0.0, seed: int | None = None, name: str = "policy") -> None:
        import torch

        self.torch = torch
        self.net = net
        self.net.eval()
        self.temperature = float(temperature)
        self.name = name
        self.rng = np.random.default_rng(seed)

    def seed(self, n: int) -> None:
        self.rng = np.random.default_rng(int(n))

    def probs(self, state: AzulState) -> tuple[list[int], np.ndarray]:
        legal = state.legal_actions()
        x = self.torch.from_numpy(state.encode()[None, :])
        with self.torch.inference_mode():
            logits = self.net(x)[0][0].float().numpy()
        sel = logits[np.asarray(legal, dtype=np.int64)].astype(np.float64)
        sel = np.exp(sel - sel.max())
        return legal, sel / sel.sum()

    def act(self, state: AzulState) -> int:
        legal, p = self.probs(state)
        if not legal:
            raise ValueError("no legal actions (terminal state?)")
        if self.temperature <= 0.0:
            return int(legal[int(np.argmax(p))])
        q = np.power(p, 1.0 / self.temperature)
        q /= q.sum()
        return int(legal[int(self.rng.choice(len(legal), p=q))])


class HybridOpeningAgent(Agent):
    """``opening`` for the first ``k`` own decisions of round 1, ``main`` after."""

    def __init__(self, opening: Agent, main: Agent, k: int | None = 3, name: str = "hybrid") -> None:
        self.opening = opening
        self.main = main
        self.k = k  # None = the whole first round
        self.name = name
        self.own_moves = 0
        self.handed_over = 0  # decisions the opening policy took, for the record

    def seed(self, n: int) -> None:
        self.own_moves = 0
        self.opening.seed(int(n))
        self.main.seed(int(n))

    def _opening_turn(self, state: AzulState) -> bool:
        if state.round_index != 0:
            return False
        if self.k is None:
            return True
        return self.own_moves < self.k

    def act(self, state: AzulState) -> int:
        use_opening = self._opening_turn(state)
        if state.round_index == 0:
            self.own_moves += 1
        else:
            self.own_moves = 0  # a later round: the counter is irrelevant until the next game
        if use_opening:
            self.handed_over += 1
            return self.opening.act(state)
        return self.main.act(state)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<HybridOpeningAgent k={self.k} opening={self.opening!r} main={self.main!r}>"
