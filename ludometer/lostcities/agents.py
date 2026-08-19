"""Hand-written Lost Cities baselines — the fixed rungs of its Elo ladder."""

from __future__ import annotations

import random

from ludometer.agents.base import Agent
from ludometer.lostcities.engine import (
    DISCARD_BASE,
    DRAW_DECK,
    DRAW_PILE,
    NUM_CARDS,
    NUM_COLORS,
    NUM_RANKS,
    LostCitiesState,
    card_points,
)

__all__ = ["LCGreedyAgent", "LCHeuristicAgent", "LCRandomAgent"]


class LCRandomAgent(Agent):
    name = "lc:random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def seed(self, n: int) -> None:
        self.rng.seed(n)

    def act(self, state: LostCitiesState) -> int:
        return self.rng.choice(state.legal_actions())


def _color_value(state: LostCitiesState, player: int, color: int) -> int:
    """Points this player's hand could still add to that color."""
    hand = state.hands[player]
    top, _ = state._pile_state(player, color)
    return sum(
        card_points(color * NUM_RANKS + r) * hand[color * NUM_RANKS + r]
        for r in range(NUM_RANKS)
        if r > top
    )


class LCGreedyAgent(Agent):
    """Plays the lowest playable card of the color it is richest in; discards
    its most useless card otherwise; always draws from the deck."""

    name = "lc:greedy"

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def seed(self, n: int) -> None:
        self.rng.seed(n)

    def act(self, state: LostCitiesState) -> int:
        legal = state.legal_actions()
        if state.phase == 1:
            return DRAW_DECK if DRAW_DECK in legal else self.rng.choice(legal)
        me = state.current_player
        plays = [a for a in legal if a < DISCARD_BASE]
        if plays:
            # richest color first, then the lowest rank so the pile stays open
            def key(action: int) -> tuple:
                color, rank = divmod(action, NUM_RANKS)
                return (-_color_value(state, me, color), rank)

            best = min(plays, key=key)
            color = best // NUM_RANKS
            if _color_value(state, me, color) >= 10 or state.expeditions[me][color]:
                return best
        discards = [a for a in legal if a >= DISCARD_BASE]
        return min(
            discards,
            key=lambda a: card_points(a - DISCARD_BASE)
            + 5 * bool(state.expeditions[me][(a - DISCARD_BASE) // NUM_RANKS]),
        )


class LCHeuristicAgent(LCGreedyAgent):
    """Greedy plus: commits to at most three colors, keeps handshakes only
    when the color is rich, and picks up a discard that fits a committed
    expedition instead of drawing blind."""

    name = "lc:heuristic"

    def act(self, state: LostCitiesState) -> int:
        legal = state.legal_actions()
        me = state.current_player
        if state.phase == 1:
            for color in range(NUM_COLORS):
                action = DRAW_PILE + color
                if action not in legal or not state.discards[color]:
                    continue
                card = state.discards[color][-1]
                if state.expeditions[me][color] and state._may_play(me, card):
                    return action
            return DRAW_DECK if DRAW_DECK in legal else self.rng.choice(legal)

        committed = [
            c
            for c in range(NUM_COLORS)
            if state.expeditions[me][c] or _color_value(state, me, c) >= 15
        ][:3]
        plays = [
            a
            for a in legal
            if a < DISCARD_BASE
            and a // NUM_RANKS in committed
            and (
                a % NUM_RANKS != 0  # a handshake only on a rich color
                or _color_value(state, me, a // NUM_RANKS) >= 15
            )
        ]
        if plays:
            return min(plays, key=lambda a: a % NUM_RANKS)
        discards = [
            a
            for a in legal
            if a >= DISCARD_BASE and (a - DISCARD_BASE) // NUM_RANKS not in committed
        ] or [a for a in legal if a >= DISCARD_BASE]
        return min(discards, key=lambda a: card_points(a - DISCARD_BASE))
