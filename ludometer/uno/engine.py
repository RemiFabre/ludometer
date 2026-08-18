"""Uno rules engine — 2 players, match to 500 points.

Same shape as :mod:`ludometer.azul.engine` so the trainer, MCTS, arena and
replay buffer drive it unchanged: pure Python + numpy, all chance in an internal
``random.Random``, deterministic given the ``new_game`` seed.

Unlike Azul this game has **hidden information** (the opponent's hand and the
deck order). The search handles it by determinizing at the root — see
:meth:`UnoState.search_root` — and every draw stays a chance node, so the tree
cannot memorise the deck order it was handed.

Cards are integers 0..53: a colored card is ``color * 13 + rank`` with ranks
0-9 = numbers, 10 = skip, 11 = reverse, 12 = draw two; 52 = wild, 53 = wild
draw four. Colors are 0..3 = red, yellow, green, blue.

Actions (61):

``0..51``   play that colored card
``52..55``  play a wild, declaring color ``a - 52``
``56..59``  play a wild draw four, declaring color ``a - 56``
``60``      draw

Rules as implemented (2-player, one deliberate simplification each):

* Reverse acts as a skip, draw two and draw four skip the opponent — so the
  player who played keeps the turn. Official.
* No stacking of draw two / draw four, and no challenge: a draw four is
  playable at any time.
* Draw is legal only when nothing else is; you draw one card and, if it can be
  played, you must play it. (Official lets you keep it — this keeps the action
  space at 61 and every hand strictly progressing.)
* The starting card of a hand is re-flipped until it is a number card, so no
  action card ever resolves against an empty board state.
* A hand ends when someone empties their hand: they score the sum of the
  opponent's remaining cards (numbers face value, action cards 20, wilds 50).
  First to :data:`TARGET_SCORE` wins the match.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

# Allocate a `random.Random` without seeding it (see UnoState.clone).
_NEW_RANDOM = random.Random.__new__

__all__ = [
    "ACTION_SPACE",
    "CARD_NAMES",
    "COLOR_NAMES",
    "DRAW",
    "ENCODED_SIZE",
    "NUM_CARDS",
    "TARGET_SCORE",
    "UnoState",
    "card_points",
]

NUM_COLORS = 4
NUM_RANKS = 13  # 0-9, skip, reverse, draw two
NUM_CARDS = 54  # 52 colored + wild + wild draw four
WILD = 52
WILD4 = 53
SKIP, REVERSE, DRAW_TWO = 10, 11, 12

DRAW = 60
ACTION_SPACE = 61
HAND_SIZE = 7
TARGET_SCORE = 500
MAX_HANDS = 60  # backstop only; 300 random matches peaked at 33 hands

COLOR_NAMES = ("red", "yellow", "green", "blue")
RANK_NAMES = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "S", "R", "+2")


def card_points(card: int) -> int:
    """Scoring value of one card (numbers face value, actions 20, wilds 50)."""
    if card >= WILD:
        return 50
    rank = card % NUM_RANKS
    return rank if rank <= 9 else 20


#: how many copies of each card id the 108-card deck holds
DECK_COUNTS = tuple(
    4 if c >= WILD else (1 if c % NUM_RANKS == 0 else 2) for c in range(NUM_CARDS)
)
FULL_DECK = tuple(c for c in range(NUM_CARDS) for _ in range(DECK_COUNTS[c]))
CARD_POINTS = tuple(card_points(c) for c in range(NUM_CARDS))
CARD_NAMES = tuple(
    "wild"
    if c == WILD
    else "wild+4"
    if c == WILD4
    else f"{COLOR_NAMES[c // NUM_RANKS][0]}{RANK_NAMES[c % NUM_RANKS]}"
    for c in range(NUM_CARDS)
)

# encode() layout: hand counts | discard census | top one-hot | color one-hot |
# 3 scalars | 2 flags. Match scores and the hand number are deliberately absent:
# the net trains on single hands and is rated over matches, and a feature it
# never varied during training would be out of distribution at rating time.
ENCODED_SIZE = NUM_CARDS * 3 + NUM_COLORS + 5

_AUX_BITS = 15  # per player; the buffer's aux slot is 2 x 15 (unused for Uno)


class UnoState:
    """A 2-player Uno match. Mutated in place by :meth:`apply`."""

    ACTION_SPACE: int = ACTION_SPACE
    ENCODED_SIZE: int = ENCODED_SIZE
    DEAL_SIZE: int = HAND_SIZE  # opening hand; Uno+ deals 9 (see uno/plus.py)
    num_players: int = 2  # interface parity with AzulState (generic drivers)
    # Tree reuse can never fire here: a kept subtree's root is a determinized
    # world whose fingerprint (it embeds both hands) never matches the real
    # state, so the knob would be a silent no-op. TrainConfig rejects it.
    TREE_REUSE_OK: bool = False

    __slots__ = (
        "_dead_passes",
        "_horizon",
        "current_color",
        "current_player",
        "deck",
        "discard",
        "discard_counts",
        "finished",
        "first_player",
        "hand_index",
        "hand_limit",
        "hand_size",
        "hands",
        "rng",
        "scores",
        "segment_values",
    )

    def __init__(self) -> None:
        self.hands: list[list[int]] = []
        self.hand_size: list[int] = [0, 0]
        self.deck: list[int] = []
        self.discard: list[int] = []
        self.discard_counts: list[int] = [0] * NUM_CARDS
        self.current_color = 0
        self.current_player = 0
        self.first_player = 0
        self.scores: list[int] = [0, 0]
        self.hand_index = 0
        self.hand_limit = MAX_HANDS
        #: outcome of each finished hand, from player 0's point of view
        self.segment_values: list[float] = []
        self.finished = False
        self._dead_passes = 0
        #: True on search_root clones: this world's hand_limit is a search
        #: horizon, so a limit exit is decided by the hand, never the score.
        self._horizon = False
        self.rng = random.Random()

    # ------------------------------------------------------------------ setup
    @classmethod
    def new_game(
        cls, seed: int, num_players: int = 2, hand_limit: int = MAX_HANDS
    ) -> UnoState:
        """A match to :data:`TARGET_SCORE`, or ``hand_limit=1`` for a single hand.

        One hand is the *training* episode and the match is the *rating* unit —
        see the note on the value function in :meth:`_end_hand`.
        """
        if num_players != 2:
            raise ValueError("only the 2-player match is implemented")
        state = cls()
        state.hand_limit = max(1, int(hand_limit))
        state.rng = random.Random(seed)
        state._deal()
        return state

    def _deal(self) -> None:
        """Fresh 108-card deck, 7 cards each, a number card face up."""
        deck = list(FULL_DECK)
        self.rng.shuffle(deck)
        self.hands = [[0] * NUM_CARDS, [0] * NUM_CARDS]
        self.hand_size = [0, 0]
        for p in (0, 1):
            for _ in range(self.DEAL_SIZE):
                card = deck.pop()
                self.hands[p][card] += 1
            self.hand_size[p] = self.DEAL_SIZE
        start = deck.pop()
        while start >= WILD or start % NUM_RANKS > 9:
            deck.insert(0, start)  # back to the bottom, then flip again
            start = deck.pop()
        self.deck = deck
        self.discard = [start]
        self.discard_counts = [0] * NUM_CARDS
        self.discard_counts[start] += 1
        self.current_color = start // NUM_RANKS
        self.current_player = self.first_player
        self._dead_passes = 0

    def clone(self) -> UnoState:
        other = UnoState.__new__(UnoState)
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
        rng = _NEW_RANDOM(random.Random)
        rng.setstate(self.rng.getstate())
        other.rng = rng
        return other

    # ------------------------------------------------------------------ rules
    @property
    def is_terminal(self) -> bool:
        return self.finished

    @property
    def round_index(self) -> int:
        """Hands played so far — what Azul calls a round (stall detection)."""
        return self.hand_index

    def legal_actions(self) -> list[int]:
        if self.finished:
            return []
        hand = self.hands[self.current_player]
        color = self.current_color
        top = self.discard[-1]
        out: list[int] = []
        base = color * NUM_RANKS
        for r in range(NUM_RANKS):
            if hand[base + r]:
                out.append(base + r)
        if top < WILD:
            rank = top % NUM_RANKS
            for c in range(NUM_COLORS):
                if c != color and hand[c * NUM_RANKS + rank]:
                    out.append(c * NUM_RANKS + rank)
        if hand[WILD]:
            out.extend((52, 53, 54, 55))
        if hand[WILD4]:
            out.extend((56, 57, 58, 59))
        if not out:
            return [DRAW]
        out.sort()
        return out

    def is_legal(self, action_id: int) -> bool:
        return action_id in self.legal_actions()

    def _has_playable(self, player: int) -> bool:
        hand = self.hands[player]
        if hand[WILD] or hand[WILD4]:
            return True
        base = self.current_color * NUM_RANKS
        if any(hand[base + r] for r in range(NUM_RANKS)):
            return True
        top = self.discard[-1]
        if top < WILD:
            rank = top % NUM_RANKS
            return any(hand[c * NUM_RANKS + rank] for c in range(NUM_COLORS))
        return False

    def _refill_deck(self) -> None:
        """Shuffle everything below the top card back into the deck."""
        if len(self.discard) < 2:
            return
        top = self.discard[-1]
        recycled = self.discard[:-1]
        self.rng.shuffle(recycled)
        self.deck = recycled
        self.discard = [top]
        self.discard_counts = [0] * NUM_CARDS
        self.discard_counts[top] += 1

    def _draw(self, player: int, n: int) -> int:
        """Draw up to ``n`` cards; returns how many were actually drawn."""
        drawn = 0
        for _ in range(n):
            if not self.deck:
                self._refill_deck()
                if not self.deck:
                    break
            card = self.deck.pop()
            self.hands[player][card] += 1
            self.hand_size[player] += 1
            drawn += 1
        return drawn

    def apply(self, action_id: int) -> None:
        if self.finished:
            raise ValueError("game is over")
        player = self.current_player
        opponent = 1 - player

        if action_id == DRAW:
            if self._has_playable(player):
                raise ValueError("draw is only legal when nothing is playable")
            if self._draw(player, 1) == 0:
                self._dead_passes += 1
                if self._dead_passes >= 2:  # nobody can move and nothing to draw
                    self._end_hand(None)
                    return
                self.current_player = opponent
                return
            self._dead_passes = 0
            if not self._has_playable(player):
                self.current_player = opponent
            return

        if action_id >= 56:
            card, new_color = WILD4, action_id - 56
        elif action_id >= 52:
            card, new_color = WILD, action_id - 52
        elif 0 <= action_id < 52:
            card, new_color = action_id, action_id // NUM_RANKS
        else:
            raise ValueError(f"illegal action {action_id}")

        hand = self.hands[player]
        if not hand[card]:
            raise ValueError(f"action {action_id}: no {CARD_NAMES[card]} in hand")
        top = self.discard[-1]
        if card < WILD and card // NUM_RANKS != self.current_color and not (
            top < WILD and card % NUM_RANKS == top % NUM_RANKS
        ):
            raise ValueError(
                f"action {action_id}: {CARD_NAMES[card]} matches neither the "
                f"color nor the rank of {CARD_NAMES[top]}"
            )
        hand[card] -= 1
        self.hand_size[player] -= 1
        self.discard.append(card)
        self.discard_counts[card] += 1
        self.current_color = new_color
        self._dead_passes = 0

        keeps_turn = False
        if card == WILD4:
            self._draw(opponent, 4)
            keeps_turn = True
        elif card < WILD:
            rank = card % NUM_RANKS
            if rank == DRAW_TWO:
                self._draw(opponent, 2)
                keeps_turn = True
            elif rank in (SKIP, REVERSE):
                keeps_turn = True

        if self.hand_size[player] == 0:
            self._end_hand(player)
        elif not keeps_turn:
            self.current_player = opponent

    def _end_hand(self, winner: int | None) -> None:
        """Score the hand, then either deal the next one or end the game.

        ``hand_limit`` exists because of the value function. A match is ~20
        near-independent hands, so a *match* label is nearly noise for a move
        played in hand 3 — but a per-hand label is not a value function either:
        it resets at every hand boundary, so a search bootstrapping through one
        sees winning a hand as a fall from +1 to 0 and avoids doing it. Training
        therefore plays one hand per episode (the label is then the episode's own
        result, which is exactly what the search backs up), while the arena rates
        checkpoints over the full match.
        """
        self.segment_values.append(
            0.0 if winner is None else (1.0 if winner == 0 else -1.0)
        )
        if winner is not None:
            loser = 1 - winner
            hand = self.hands[loser]
            self.scores[winner] += sum(
                CARD_POINTS[c] * n for c, n in enumerate(hand) if n
            )
            if self.scores[winner] >= TARGET_SCORE:
                self.finished = True
                return
        # `hand_index` stays the 0-based index of the last finished hand on
        # EVERY terminal path, so hands-played is hand_index + 1 however the
        # game ends; it only advances when the next hand is actually dealt.
        if self.hand_index + 1 >= self.hand_limit:
            self.finished = True
            return
        self.hand_index += 1
        self.first_player = 1 - self.first_player
        self._deal()

    def outcome(self) -> float | None:
        """+1 player 0 wins, -1 player 1 wins, 0 a tie, None while unfinished."""
        if not self.finished:
            return None
        if self.scores[0] >= TARGET_SCORE and self.scores[1] < TARGET_SCORE:
            return 1.0
        if self.scores[1] >= TARGET_SCORE and self.scores[0] < TARGET_SCORE:
            return -1.0
        # Nobody reached the target, so the game ended on `hand_limit`. For a
        # single-hand training episode or a search horizon the hand just played
        # is the whole result — pricing a horizon by the carried match score is
        # exactly the value-function trap in NEXT_GAMES.md §1. A *rating* match
        # truncated at the MAX_HANDS backstop, though, belongs to whoever
        # leads on points, like every card game scores an interrupted session.
        if self.hand_limit > 1 and not self._horizon:
            if self.scores[0] != self.scores[1]:
                return 1.0 if self.scores[0] > self.scores[1] else -1.0
            return 0.0
        return self.segment_values[-1] if self.segment_values else 0.0

    # ---------------------------------------------------- search integration
    def is_stochastic(self, action_id: int) -> bool:
        """Does applying ``action_id`` consume randomness (a draw or a redeal)?"""
        if action_id == DRAW or action_id >= 56:
            return True
        if action_id < 52 and action_id % NUM_RANKS == DRAW_TWO:
            return True
        return self.hand_size[self.current_player] == 1  # last card -> new hand

    def determinize(self, action_id: int, seed: int) -> UnoState:
        """Clone with a fresh deck order, then apply ``action_id``."""
        child = self.clone()
        child.rng.seed(seed)
        child.rng.shuffle(child.deck)
        child.apply(action_id)
        return child

    def chance_key(self) -> bytes:
        """Identity of a position after a chance event (both hands + the board)."""
        parts = list(self.hands[0])
        parts.extend(self.hands[1])
        parts.append(self.discard[-1])
        parts.append(self.current_color)
        parts.append(self.current_player)
        parts.append(min(len(self.deck), 255))
        return bytes(parts)

    def fingerprint(self) -> tuple[Any, ...]:
        return (
            self.current_player,
            self.hand_index,
            self.scores[0],
            self.scores[1],
            self.chance_key(),
        )

    def search_root(self, rng: random.Random) -> UnoState:
        """A determinized clone to search: the opponent's hand and the deck are
        redealt at random from the cards this player has not seen.

        Everything the mover legitimately knows — their own hand, the discard
        pile, the opponent's card count, both scores — is preserved, so the
        sampled world is consistent with their observation (PIMC).
        """
        child = self.clone()
        # The search's horizon is the end of the CURRENT hand. Hands are all but
        # independent — only the score carries over, and the encoding does not
        # even show it — so "win this hand" is what the net was trained to
        # predict and the only value the tree can bootstrap consistently. Without
        # this, a search inside a match sees going out as a fall from +1 to the
        # ~0 of a freshly dealt hand and quietly avoids winning (measured: 41%
        # of hands won, 0% of matches).
        child.hand_limit = child.hand_index + 1
        child._horizon = True
        me = self.current_player
        unseen: list[int] = []
        for card in range(NUM_CARDS):
            n = DECK_COUNTS[card] - self.hands[me][card] - self.discard_counts[card]
            unseen.extend([card] * n)
        rng.shuffle(unseen)
        k = self.hand_size[1 - me]
        opp = [0] * NUM_CARDS
        for card in unseen[:k]:
            opp[card] += 1
        child.hands[1 - me] = opp
        child.hand_size[1 - me] = k
        child.deck = unseen[k:]
        return child

    def wall_summary(self, player: int) -> list[int]:
        """Auxiliary training target — Uno has none; the head is off (see net2)."""
        return [0] * _AUX_BITS

    # --------------------------------------------------------------- encoding
    def encode(self) -> np.ndarray:
        """Observation of the player to move, ``ENCODED_SIZE`` floats.

        Deliberately *not* the full state: the opponent's cards are summarised
        by their count, so a net evaluating a determinized world cannot read the
        determinization off its own input.
        """
        out = np.zeros(ENCODED_SIZE, dtype=np.float32)
        me = self.current_player
        them = 1 - me
        hand = self.hands[me]
        for card in range(NUM_CARDS):
            out[card] = hand[card] * 0.5
            out[NUM_CARDS + card] = self.discard_counts[card] * 0.25
        out[NUM_CARDS * 2 + self.discard[-1]] = 1.0
        base = NUM_CARDS * 3
        out[base + self.current_color] = 1.0
        base += NUM_COLORS
        out[base] = self.hand_size[me] / 20.0
        out[base + 1] = self.hand_size[them] / 20.0
        out[base + 2] = len(self.deck) / 108.0
        out[base + 3] = 1.0 if self.hand_size[me] == 1 else 0.0
        out[base + 4] = 1.0 if self.hand_size[them] == 1 else 0.0
        return out

    # ------------------------------------------------------------- reporting
    def render_text(self) -> str:
        def cards(p: int) -> str:
            return " ".join(
                CARD_NAMES[c] * 1 if n == 1 else f"{CARD_NAMES[c]}x{n}"
                for c, n in enumerate(self.hands[p])
                if n
            )

        head = (
            f"hand {self.hand_index + 1}  scores {self.scores[0]}-{self.scores[1]}"
            f"  to move: P{self.current_player}"
        )
        board = (
            f"top {CARD_NAMES[self.discard[-1]]} "
            f"color {COLOR_NAMES[self.current_color]}  deck {len(self.deck)}"
        )
        return "\n".join(
            [
                head,
                board,
                f"P0 ({self.hand_size[0]}): {cards(0)}",
                f"P1 ({self.hand_size[1]}): {cards(1)}",
            ]
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "hands": [list(self.hands[0]), list(self.hands[1])],
            "hand_size": list(self.hand_size),
            "deck": len(self.deck),
            "top": self.discard[-1],
            "color": self.current_color,
            "scores": list(self.scores),
            "hand_index": self.hand_index,
            "current_player": self.current_player,
            "finished": self.finished,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<UnoState hand={self.hand_index} scores={self.scores} "
            f"p{self.current_player} deck={len(self.deck)}>"
        )
