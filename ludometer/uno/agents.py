"""Hand-written Uno baselines — the fixed rungs of the Elo ladder.

Registered as ``uno:random`` / ``uno:greedy`` / ``uno:heuristic`` (see
:mod:`ludometer.agents`), so a spec string says which game it belongs to.
"""

from __future__ import annotations

import random

from ludometer.agents.base import Agent
from ludometer.uno.engine import (
    CARD_POINTS,
    DRAW,
    DRAW_TWO,
    NUM_COLORS,
    NUM_RANKS,
    REVERSE,
    SKIP,
    WILD,
    WILD4,
    UnoState,
)

__all__ = ["UnoGreedyAgent", "UnoHeuristicAgent", "UnoRandomAgent"]


def _best_color(state: UnoState, player: int) -> int:
    """The color this player holds most of (ties go to the lower color)."""
    hand = state.hands[player]
    counts = [sum(hand[c * NUM_RANKS : (c + 1) * NUM_RANKS]) for c in range(NUM_COLORS)]
    return max(range(NUM_COLORS), key=lambda c: (counts[c], -c))


def _card_of(action: int) -> int:
    """The card an action plays (``WILD``/``WILD4`` for the two wild blocks)."""
    if action >= 56:
        return WILD4
    if action >= 52:
        return WILD
    return action


class UnoRandomAgent(Agent):
    """Uniform over legal actions."""

    name = "uno:random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def seed(self, n: int) -> None:
        self.rng.seed(n)

    def act(self, state: UnoState) -> int:
        return self.rng.choice(state.legal_actions())


class UnoGreedyAgent(Agent):
    """Dump the most expensive card you can, keeping wilds for last."""

    name = "uno:greedy"

    def act(self, state: UnoState) -> int:
        legal = state.legal_actions()
        if legal == [DRAW]:
            return DRAW
        color = _best_color(state, state.current_player)
        best, best_key = legal[0], None
        for action in legal:
            card = _card_of(action)
            wild = card >= WILD
            # non-wild first, then most points, then the color we hold most of
            key = (
                0 if wild else 1,
                CARD_POINTS[card],
                1 if wild and action % 4 == color else 0,
            )
            if best_key is None or key > best_key:
                best, best_key = action, key
        return best


class UnoHeuristicAgent(Agent):
    """Tempo and hand shape: deny the opponent when they are close to out,
    dump expensive cards early, and hold wilds until they are worth spending.
    """

    name = "uno:heuristic"

    def __init__(self, deny: float = 30.0, wild_cost: float = 35.0) -> None:
        self.deny = deny
        self.wild_cost = wild_cost

    def act(self, state: UnoState) -> int:
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        me = state.current_player
        mine = state.hand_size[me]
        theirs = state.hand_size[1 - me]
        want = _best_color(state, me)
        best, best_score = legal[0], -1e30
        for action in legal:
            if action == DRAW:
                score = -1e6
            else:
                card = _card_of(action)
                score = float(CARD_POINTS[card])  # get the expensive cards out
                if mine == 1:
                    score += 1e6  # this action empties the hand and wins it
                if card >= WILD:
                    score -= self.wild_cost  # a wild is always playable later
                    if action % 4 == want:
                        score += 10.0
                    if card == WILD4 and theirs <= 3:
                        score += self.deny  # +4 lands hardest when they are low
                else:
                    rank = card % NUM_RANKS
                    if rank in (SKIP, REVERSE, DRAW_TWO) and theirs <= 3:
                        score += self.deny
                    if card // NUM_RANKS == want:
                        score += 8.0  # staying in our own color keeps options
            if score > best_score:
                best, best_score = action, score
        return best
