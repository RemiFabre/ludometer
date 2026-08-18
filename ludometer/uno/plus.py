"""Uno+ — plain Uno with four house rules that add decisions (NEXT_GAMES.md §2).

Same deck, same 61 actions, same match-to-500 shell as :mod:`.engine`. What
changes is *who gets to choose*:

* **R1 — draw is always legal.** ``DRAW`` sits beside every playable card, and
  it is atomic: take one card, your turn ends (you do not get to play it).
  Illegal once your hand holds :data:`DRAW_CAP` cards or when deck and discard
  are both exhausted — if you are also out of plays, ``DRAW`` degrades to the
  base engine's dead pass so a blocked hand still terminates.
* **R2 — stacking.** A ``+2`` answers a ``+2`` and a ``+4`` answers a ``+4``
  (like on like only). The victim chooses: stack (the penalty accumulates and
  crosses back) or ``DRAW`` (take the whole stack and lose the turn). While
  ``pending_draw > 0`` nothing else is legal.
* **R3 — the 7-swap.** Playing a 7 swaps the hands, then the turn passes as
  for any number card. Going out with a 7 wins first — no swap. The swap
  creates *knowledge*: each player then knows the opponent's hand exactly, and
  that knowledge decays as cards are played. ``known[p]`` counts what player
  ``p`` knows of the opponent's hand; :meth:`UnoPlusState.search_root` deals
  those cards to the opponent before determinizing the remainder — without
  this the search throws away the very information the rule exists to create.
* **R4 — the opening hand is 9 cards.**

The encoding is the base observation plus ``pending_draw / 4``, a 3-way
``pending_kind`` one-hot and the 54-slot ``known`` vector, so the two nets
differ only where the games differ.
"""

from __future__ import annotations

import random

import numpy as np

from ludometer.uno.engine import (
    DECK_COUNTS,
    DRAW,
    DRAW_TWO,
    ENCODED_SIZE,
    NUM_CARDS,
    NUM_RANKS,
    WILD,
    WILD4,
    UnoState,
)

__all__ = ["DRAW_CAP", "PLUS_ENCODED_SIZE", "PLUS_HAND_SIZE", "UnoPlusState"]

PLUS_HAND_SIZE = 9
DRAW_CAP = 15  # a hand at the cap may not voluntarily draw (stacks still land)
SEVEN = 7

# base observation | pending_draw/4 | pending_kind one-hot (none/+2/+4) | known
PLUS_ENCODED_SIZE = ENCODED_SIZE + 1 + 3 + NUM_CARDS
_KIND_SLOT = {0: 0, 2: 1, 4: 2}


class UnoPlusState(UnoState):
    """A 2-player Uno+ match. Mutated in place by :meth:`apply`."""

    ENCODED_SIZE: int = PLUS_ENCODED_SIZE
    DEAL_SIZE: int = PLUS_HAND_SIZE

    __slots__ = ("known", "pending_draw", "pending_kind")

    def __init__(self) -> None:
        self.pending_draw = 0
        self.pending_kind = 0
        self.known: list[list[int]] = [[0] * NUM_CARDS, [0] * NUM_CARDS]
        super().__init__()

    def _deal(self) -> None:
        super()._deal()
        self.pending_draw = 0
        self.pending_kind = 0
        self.known = [[0] * NUM_CARDS, [0] * NUM_CARDS]

    def clone(self) -> UnoPlusState:
        other = UnoPlusState.__new__(UnoPlusState)
        other.hands = [list(self.hands[0]), list(self.hands[1])]
        other.hand_size = list(self.hand_size)
        other.deck = list(self.deck)
        other.discard = list(self.discard)
        other.discard_counts = list(self.discard_counts)
        other.current_color = self.current_color
        other.current_player = self.current_player
        other.first_player = self.first_player
        other.scores = list(self.scores)
        other.hand_index = self.hand_index
        other.hand_limit = self.hand_limit
        other.segment_values = list(self.segment_values)
        other.finished = self.finished
        other._dead_passes = self._dead_passes
        other._horizon = self._horizon
        other.pending_draw = self.pending_draw
        other.pending_kind = self.pending_kind
        other.known = [list(self.known[0]), list(self.known[1])]
        rng = random.Random.__new__(random.Random)
        rng.setstate(self.rng.getstate())
        other.rng = rng
        return other

    # ------------------------------------------------------------------ rules
    def _stack_answers(self) -> list[int]:
        """Actions that answer the pending stack (like on like only)."""
        hand = self.hands[self.current_player]
        if self.pending_kind == 2:
            return [c * NUM_RANKS + DRAW_TWO for c in range(4) if hand[c * NUM_RANKS + DRAW_TWO]]
        return list(range(56, 60)) if hand[WILD4] else []

    def _may_draw(self) -> bool:
        """Is a voluntary draw available (R1's two guards)?"""
        if self.hand_size[self.current_player] >= DRAW_CAP:
            return False
        return bool(self.deck) or len(self.discard) > 1

    def legal_actions(self) -> list[int]:
        if self.finished:
            return []
        if self.pending_draw:
            out = self._stack_answers()
            out.append(DRAW)  # take the stack; always available
            return sorted(out)
        base = super().legal_actions()
        plays = [] if base == [DRAW] else base
        if self._may_draw():
            plays = [*plays, DRAW]
        return plays or [DRAW]  # out of plays AND barred from drawing: a pass

    def apply(self, action_id: int) -> None:
        if self.finished:
            raise ValueError("game is over")
        player = self.current_player
        opponent = 1 - player

        if self.pending_draw and action_id == DRAW:
            # take the whole accumulated stack and lose the turn
            self._draw(player, self.pending_draw)
            self.pending_draw = 0
            self.pending_kind = 0
            self._dead_passes = 0
            self.current_player = opponent
            return

        if action_id == DRAW:
            if not self._may_draw() and self._has_playable(player):
                raise ValueError("draw is barred (cap/exhausted) and plays exist")
            if self._may_draw() and self._draw(player, 1):
                self._dead_passes = 0
            else:
                # capped or exhausted with nothing to play: the dead pass that
                # keeps a blocked hand terminating (two in a row end it)
                self._dead_passes += 1
                if self._dead_passes >= 2:
                    self._end_hand(None)
                    return
            self.current_player = opponent  # atomic: drawing ends the turn
            return

        if action_id >= 56:
            card, new_color = WILD4, action_id - 56
        elif action_id >= 52:
            card, new_color = WILD, action_id - 52
        elif 0 <= action_id < 52:
            card, new_color = action_id, action_id // NUM_RANKS
        else:
            raise ValueError(f"illegal action {action_id}")
        if self.pending_draw and not (
            (self.pending_kind == 2 and card < WILD and card % NUM_RANKS == DRAW_TWO)
            or (self.pending_kind == 4 and card == WILD4)
        ):
            raise ValueError(f"only a +{self.pending_kind} answers a +{self.pending_kind}")

        hand = self.hands[player]
        if not hand[card]:
            raise ValueError(f"action {action_id}: card {card} not in hand")
        top = self.discard[-1]
        if (
            not self.pending_draw
            and card < WILD
            and card // NUM_RANKS != self.current_color
            and not (top < WILD and card % NUM_RANKS == top % NUM_RANKS)
        ):
            raise ValueError(f"action {action_id}: matches neither color nor rank")
        hand[card] -= 1
        self.hand_size[player] -= 1
        self.discard.append(card)
        self.discard_counts[card] += 1
        self.current_color = new_color
        self._dead_passes = 0
        if self.known[opponent][card]:  # the opponent saw a card they knew leave
            self.known[opponent][card] -= 1

        rank = card % NUM_RANKS if card < WILD else -1
        went_out = self.hand_size[player] == 0

        if card == WILD4 or rank == DRAW_TWO:
            add = 4 if card == WILD4 else 2
            if went_out:
                # official scoring: the victim still draws before being counted
                self._draw(opponent, self.pending_draw + add)
                self.pending_draw = 0
                self.pending_kind = 0
                self._end_hand(player)
                return
            self.pending_draw += add
            self.pending_kind = add
            self.current_player = opponent  # they answer or take
            return

        if went_out:
            self._end_hand(player)
            return

        if rank == SEVEN:
            # swap hands; both players now know the opposing hand exactly
            self.hands[player], self.hands[opponent] = (
                self.hands[opponent],
                self.hands[player],
            )
            self.hand_size[player], self.hand_size[opponent] = (
                self.hand_size[opponent],
                self.hand_size[player],
            )
            self.known[player] = list(self.hands[opponent])
            self.known[opponent] = list(self.hands[player])
            self.current_player = opponent
            return

        if rank in (10, 11):  # skip / reverse: keep the turn (2-player)
            return
        self.current_player = opponent

    # ---------------------------------------------------- search integration
    def is_stochastic(self, action_id: int) -> bool:
        """A draw (single, stack or forced-pass attempt) consumes randomness,
        and so does going out (the opponent may draw, and a match redeals)."""
        return action_id == DRAW or self.hand_size[self.current_player] == 1

    def chance_key(self) -> bytes:
        parts = list(self.hands[0])
        parts.extend(self.hands[1])
        parts.append(self.discard[-1])
        parts.append(self.current_color)
        parts.append(self.current_player)
        parts.append(min(len(self.deck), 255))
        parts.append(min(self.pending_draw, 255))
        parts.append(self.pending_kind)
        parts.extend(self.known[0])
        parts.extend(self.known[1])
        return bytes(parts)

    def search_root(self, rng: random.Random) -> UnoPlusState:
        """PIMC root that respects the 7-swap's knowledge: the cards this
        player *knows* the opponent holds are dealt to them first, and only
        the remainder is determinized. Horizon: the current hand (see base)."""
        child = self.clone()
        child.hand_limit = child.hand_index + 1
        child._horizon = True
        me = self.current_player
        known = self.known[me]
        unseen: list[int] = []
        for card in range(NUM_CARDS):
            n = (
                DECK_COUNTS[card]
                - self.hands[me][card]
                - self.discard_counts[card]
                - known[card]
            )
            unseen.extend([card] * max(n, 0))
        rng.shuffle(unseen)
        k = max(self.hand_size[1 - me] - sum(known), 0)
        opp = list(known)
        for card in unseen[:k]:
            opp[card] += 1
        child.hands[1 - me] = opp
        child.hand_size[1 - me] = self.hand_size[1 - me]
        child.deck = unseen[k:]
        return child

    # --------------------------------------------------------------- encoding
    def encode(self) -> np.ndarray:
        out = np.zeros(PLUS_ENCODED_SIZE, dtype=np.float32)
        out[:ENCODED_SIZE] = super().encode()
        base = ENCODED_SIZE
        out[base] = self.pending_draw / 4.0
        out[base + 1 + _KIND_SLOT[self.pending_kind]] = 1.0
        known = self.known[self.current_player]
        for card in range(NUM_CARDS):
            out[base + 4 + card] = known[card] * 0.5
        return out

    # ------------------------------------------------------------- reporting
    def to_json(self) -> dict:
        payload = super().to_json()
        payload["pending_draw"] = self.pending_draw
        payload["pending_kind"] = self.pending_kind
        payload["known"] = [list(self.known[0]), list(self.known[1])]
        return payload
