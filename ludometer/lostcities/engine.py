"""Lost Cities rules engine — 2 players, one deck, hidden hands.

Why this game is in the study: it moves the *same* dials as Uno (hidden hand,
high luck) and is universally rated excellent. If Uno's learning curve
flattens where this one keeps climbing, the flatness was Uno's design, not a
property of lucky card games (NEXT_GAMES.md §5).

Cards are ``color * 10 + rank`` with colors 0..4 and rank 0 the handshake
(three copies per color) while ranks 1..9 are the number cards 2..10 (one copy
each) — 50 distinct ids, 60 physical cards.

A turn is two phases by the rulebook: **place** (play a card onto your own
expedition of its color, ascending, handshakes strictly first — or discard it
onto its color's pile), then **draw** (the deck, or the top of any discard
pile except the one you just discarded to). The game ends the moment the last
deck card is drawn. Scoring per started expedition: card sum − 20, times
(1 + handshakes), plus 20 if it holds 8+ cards.

Hidden information is handled exactly as in Uno: PIMC determinization at the
search root, with one extra piece of *public* knowledge tracked per observer —
cards the opponent picked up from a discard pile are known to be in their hand
(``known``), and the determinizer deals those first, decrementing as they are
played (the same trap and fix as Uno+'s 7-swap).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

__all__ = [
    "ACTION_SPACE",
    "DECK_COUNTS",
    "DRAW_DECK",
    "ENCODED_SIZE",
    "HAND_SIZE",
    "NUM_CARDS",
    "NUM_COLORS",
    "LostCitiesState",
    "card_id",
    "score_expedition",
]

NUM_COLORS = 5
NUM_RANKS = 10  # 0 = handshake, 1..9 = the number cards 2..10
NUM_CARDS = NUM_COLORS * NUM_RANKS
HAND_SIZE = 8

#: actions: 0..49 play card, 50..99 discard card, 100 draw deck, 101..105 pile
DISCARD_BASE = 50
DRAW_DECK = 100
DRAW_PILE = 101
ACTION_SPACE = 106

DECK_COUNTS = tuple(3 if c % NUM_RANKS == 0 else 1 for c in range(NUM_CARDS))
FULL_DECK = tuple(c for c in range(NUM_CARDS) for _ in range(DECK_COUNTS[c]))

_AUX_BITS = 15

COLOR_NAMES = ("red", "green", "blue", "white", "yellow")


def card_id(color: int, rank: int) -> int:
    return color * NUM_RANKS + rank


def card_points(card: int) -> int:
    """Handshakes are worth nothing by themselves; number rank r is r+1."""
    rank = card % NUM_RANKS
    return 0 if rank == 0 else rank + 1


def score_expedition(pile: list[int]) -> int:
    """Rulebook scoring for one expedition (a list of card ids, played order)."""
    if not pile:
        return 0
    handshakes = sum(1 for c in pile if c % NUM_RANKS == 0)
    total = (sum(card_points(c) for c in pile) - 20) * (1 + handshakes)
    if len(pile) >= 8:
        total += 20
    return total


# encode() layout: hand | my expeditions 5x5 | their expeditions 5x5 |
# discard census | per-pile top rank + handshake flag | 3 scalars
ENCODED_SIZE = NUM_CARDS + 25 + 25 + NUM_CARDS + 2 * NUM_COLORS + 3


class LostCitiesState:
    """A 2-player Lost Cities game. Mutated in place by :meth:`apply`."""

    ACTION_SPACE: int = ACTION_SPACE
    ENCODED_SIZE: int = ENCODED_SIZE
    num_players: int = 2
    TREE_REUSE_OK: bool = False  # determinized roots never match (see Uno)

    __slots__ = (
        "current_player",
        "deck",
        "discards",
        "expeditions",
        "finished",
        "hand_size",
        "hands",
        "just_discarded",
        "known",
        "phase",
        "rng",
        "scores",
    )

    def __init__(self) -> None:
        self.hands: list[list[int]] = [[0] * NUM_CARDS, [0] * NUM_CARDS]
        self.hand_size = [0, 0]
        self.deck: list[int] = []
        self.discards: list[list[int]] = [[] for _ in range(NUM_COLORS)]
        self.expeditions: list[list[list[int]]] = [
            [[] for _ in range(NUM_COLORS)] for _ in (0, 1)
        ]
        self.current_player = 0
        self.phase = 0  # 0 = place, 1 = draw
        self.just_discarded = -1  # color discarded to this turn, else -1
        #: known[p][c]: copies of card c that player p KNOWS the opponent holds
        #: (picked up from a discard pile in public view)
        self.known: list[list[int]] = [[0] * NUM_CARDS, [0] * NUM_CARDS]
        self.scores = [0, 0]
        self.finished = False
        self.rng = random.Random()

    # ------------------------------------------------------------------ setup
    @classmethod
    def new_game(cls, seed: int, num_players: int = 2) -> LostCitiesState:
        if num_players != 2:
            raise ValueError("only the 2-player game is implemented")
        state = cls()
        state.rng = random.Random(seed)
        deck = list(FULL_DECK)
        state.rng.shuffle(deck)
        for player in (0, 1):
            for _ in range(HAND_SIZE):
                state.hands[player][deck.pop()] += 1
            state.hand_size[player] = HAND_SIZE
        state.deck = deck
        return state

    def clone(self) -> LostCitiesState:
        other = LostCitiesState.__new__(LostCitiesState)
        other.hands = [list(self.hands[0]), list(self.hands[1])]
        other.hand_size = list(self.hand_size)
        other.deck = list(self.deck)
        other.discards = [list(p) for p in self.discards]
        other.expeditions = [
            [list(p) for p in self.expeditions[0]],
            [list(p) for p in self.expeditions[1]],
        ]
        other.current_player = self.current_player
        other.phase = self.phase
        other.just_discarded = self.just_discarded
        other.known = [list(self.known[0]), list(self.known[1])]
        other.scores = list(self.scores)
        other.finished = self.finished
        rng = random.Random.__new__(random.Random)
        rng.setstate(self.rng.getstate())
        other.rng = rng
        return other

    # ------------------------------------------------------------------ rules
    @property
    def is_terminal(self) -> bool:
        return self.finished

    @property
    def round_index(self) -> int:
        """Deck cards consumed — monotonic, which is all stall detection needs."""
        return 60 - 2 * HAND_SIZE - len(self.deck)

    def _pile_state(self, player: int, color: int) -> tuple[int, bool]:
        """(highest number rank in the expedition, does it hold any number)."""
        top = 0
        has_number = False
        for card in self.expeditions[player][color]:
            rank = card % NUM_RANKS
            if rank:
                has_number = True
                top = max(top, rank)
        return top, has_number

    def _may_play(self, player: int, card: int) -> bool:
        color, rank = divmod(card, NUM_RANKS)
        top, has_number = self._pile_state(player, color)
        if rank == 0:
            return not has_number  # handshakes strictly before numbers
        return rank > top

    def legal_actions(self) -> list[int]:
        if self.finished:
            return []
        player = self.current_player
        if self.phase == 0:
            out = []
            hand = self.hands[player]
            for card in range(NUM_CARDS):
                if not hand[card]:
                    continue
                if self._may_play(player, card):
                    out.append(card)
                out.append(DISCARD_BASE + card)
            out.sort()
            return out
        out = [DRAW_DECK] if self.deck else []
        for color in range(NUM_COLORS):
            if self.discards[color] and color != self.just_discarded:
                out.append(DRAW_PILE + color)
        return out

    def is_legal(self, action_id: int) -> bool:
        return action_id in self.legal_actions()

    def apply(self, action_id: int) -> None:
        if self.finished:
            raise ValueError("game is over")
        player = self.current_player
        opponent = 1 - player
        if self.phase == 0:
            card = action_id if action_id < DISCARD_BASE else action_id - DISCARD_BASE
            if not 0 <= card < NUM_CARDS or not self.hands[player][card]:
                raise ValueError(f"action {action_id}: card {card} not in hand")
            if action_id < DISCARD_BASE and not self._may_play(player, card):
                raise ValueError(f"action {action_id}: expedition must ascend")
            self.hands[player][card] -= 1
            self.hand_size[player] -= 1
            if self.known[opponent][card]:
                self.known[opponent][card] -= 1
            if action_id < DISCARD_BASE:
                self.expeditions[player][card // NUM_RANKS].append(card)
                self.just_discarded = -1
            else:
                self.discards[card // NUM_RANKS].append(card)
                self.just_discarded = card // NUM_RANKS
            self.phase = 1
            return

        if action_id == DRAW_DECK:
            if not self.deck:
                raise ValueError("the deck is empty")
            card = self.deck.pop()
            self.hands[player][card] += 1
            self.hand_size[player] += 1
            if not self.deck:
                self._finish()
                return
        elif DRAW_PILE <= action_id < DRAW_PILE + NUM_COLORS:
            color = action_id - DRAW_PILE
            if not self.discards[color] or color == self.just_discarded:
                raise ValueError(f"cannot draw from pile {color}")
            card = self.discards[color].pop()
            self.hands[player][card] += 1
            self.hand_size[player] += 1
            self.known[opponent][card] += 1  # picked up in public view
        else:
            raise ValueError(f"illegal action {action_id}")
        self.phase = 0
        self.just_discarded = -1
        self.current_player = opponent

    def _finish(self) -> None:
        for player in (0, 1):
            self.scores[player] = sum(
                score_expedition(pile) for pile in self.expeditions[player]
            )
        self.finished = True

    def outcome(self) -> float | None:
        if not self.finished:
            return None
        if self.scores[0] > self.scores[1]:
            return 1.0
        if self.scores[1] > self.scores[0]:
            return -1.0
        return 0.0

    # ---------------------------------------------------- search integration
    def is_stochastic(self, action_id: int) -> bool:
        return action_id == DRAW_DECK

    def determinize(self, action_id: int, seed: int) -> LostCitiesState:
        child = self.clone()
        child.rng.seed(seed)
        child.rng.shuffle(child.deck)
        child.apply(action_id)
        return child

    def chance_key(self) -> bytes:
        parts = list(self.hands[0])
        parts.extend(self.hands[1])
        for player in (0, 1):
            for color in range(NUM_COLORS):
                top, has_number = self._pile_state(player, color)
                pile = self.expeditions[player][color]
                parts.extend((top, int(has_number), len(pile) % 256))
        for color in range(NUM_COLORS):
            pile = self.discards[color]
            parts.append(pile[-1] + 1 if pile else 0)
            parts.append(len(pile) % 256)
        parts.append(self.current_player)
        parts.append(self.phase)
        parts.append(self.just_discarded + 1)
        parts.append(min(len(self.deck), 255))
        return bytes(parts)

    def fingerprint(self) -> tuple[Any, ...]:
        return (self.current_player, self.phase, len(self.deck), self.chance_key())

    def search_root(self, rng: random.Random) -> LostCitiesState:
        """PIMC: redeal the opponent's hand and the deck from the unseen cards,
        after first dealing them every card they publicly picked up (known)."""
        child = self.clone()
        me = self.current_player
        known = self.known[me]
        visible = [0] * NUM_CARDS
        for card in range(NUM_CARDS):
            visible[card] += self.hands[me][card]
        for pile in self.discards:
            for card in pile:
                visible[card] += 1
        for player in (0, 1):
            for pile in self.expeditions[player]:
                for card in pile:
                    visible[card] += 1
        unseen: list[int] = []
        for card in range(NUM_CARDS):
            n = DECK_COUNTS[card] - visible[card] - known[card]
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

    def wall_summary(self, player: int) -> list[int]:
        return [0] * _AUX_BITS

    # --------------------------------------------------------------- encoding
    def encode(self) -> np.ndarray:
        out = np.zeros(ENCODED_SIZE, dtype=np.float32)
        me = self.current_player
        them = 1 - me
        hand = self.hands[me]
        for card in range(NUM_CARDS):
            out[card] = hand[card] / 3.0
        base = NUM_CARDS
        for who, player in ((0, me), (1, them)):
            for color in range(NUM_COLORS):
                pile = self.expeditions[player][color]
                top, has_number = self._pile_state(player, color)
                handshakes = sum(1 for c in pile if c % NUM_RANKS == 0)
                at = base + 25 * who + 5 * color
                out[at] = 1.0 if pile else 0.0
                out[at + 1] = top / 9.0
                out[at + 2] = handshakes / 3.0
                out[at + 3] = sum(card_points(c) for c in pile) / 54.0
                out[at + 4] = len(pile) / 12.0
        base += 50
        for pile in self.discards:
            for card in pile:
                out[base + card] += 1 / 3.0
        base += NUM_CARDS
        for color in range(NUM_COLORS):
            pile = self.discards[color]
            if pile:
                out[base + color] = (pile[-1] % NUM_RANKS) / 9.0
                out[base + NUM_COLORS + color] = 1.0 if pile[-1] % NUM_RANKS == 0 else 0.5
        base += 2 * NUM_COLORS
        out[base] = len(self.deck) / 44.0
        out[base + 1] = float(self.phase)
        out[base + 2] = self.hand_size[me] / 8.0
        return out

    # ------------------------------------------------------------- reporting
    def render_text(self) -> str:
        def exp(player: int) -> str:
            return " | ".join(
                f"{COLOR_NAMES[c][0]}:" + ",".join(str(x % NUM_RANKS) for x in pile)
                for c, pile in enumerate(self.expeditions[player])
                if pile
            )

        return "\n".join(
            [
                f"deck {len(self.deck)}  P{self.current_player} to "
                + ("place" if self.phase == 0 else "draw"),
                f"P0: {exp(0)}",
                f"P1: {exp(1)}",
                "discards: "
                + " | ".join(
                    f"{COLOR_NAMES[c][0]}({len(p)})" for c, p in enumerate(self.discards)
                ),
            ]
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "hands": [list(self.hands[0]), list(self.hands[1])],
            "deck": len(self.deck),
            "discards": [list(p) for p in self.discards],
            "expeditions": [
                [list(p) for p in self.expeditions[0]],
                [list(p) for p in self.expeditions[1]],
            ],
            "phase": self.phase,
            "current_player": self.current_player,
            "finished": self.finished,
            "scores": list(self.scores),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<LostCitiesState deck={len(self.deck)} p{self.current_player}>"
