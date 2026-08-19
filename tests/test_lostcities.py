"""Lost Cities (Knizia) — the scientific control for Uno (NEXT_GAMES.md §5).

Same dials as Uno (hidden hand, high luck), universally rated excellent. If
Uno's curve flattens where this one keeps climbing, the flatness was Uno's.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from ludometer.agents import make_agent
from ludometer.games import get_game
from ludometer.lostcities.engine import (
    ACTION_SPACE,
    DECK_COUNTS,
    DRAW_DECK,
    ENCODED_SIZE,
    HAND_SIZE,
    NUM_CARDS,
    LostCitiesState,
    card_id,
    score_expedition,
)

BASELINES = ("lc:random", "lc:greedy", "lc:heuristic")


def census(state: LostCitiesState) -> list[int]:
    counts = [0] * NUM_CARDS
    for player in (0, 1):
        for card, n in enumerate(state.hands[player]):
            counts[card] += n
        for pile in state.expeditions[player]:
            for card in pile:
                counts[card] += 1
    for pile in state.discards:
        for card in pile:
            counts[card] += 1
    for card in state.deck:
        counts[card] += 1
    return counts


def play_out(state: LostCitiesState, rng: random.Random, limit: int = 400) -> int:
    moves = 0
    while not state.is_terminal and moves < limit:
        legal = state.legal_actions()
        assert legal, state.render_text()
        assert census(state) == list(DECK_COUNTS)
        state.apply(rng.choice(legal))
        moves += 1
    return moves


# ---------------------------------------------------------------- the basics
def test_the_deck_is_the_real_sixty_card_deck():
    assert sum(DECK_COUNTS) == 60
    assert DECK_COUNTS[card_id(0, 0)] == 3  # three handshakes per color
    assert DECK_COUNTS[card_id(2, 5)] == 1  # one copy of each number
    assert ACTION_SPACE == 106


def test_the_deal_is_eight_cards_each():
    state = LostCitiesState.new_game(seed=1)
    assert state.hand_size == [HAND_SIZE, HAND_SIZE]
    assert len(state.deck) == 60 - 2 * HAND_SIZE
    assert census(state) == list(DECK_COUNTS)


def test_scoring_matches_the_rulebook():
    # started expedition costs 20; handshakes multiply; 8 cards earn +20
    assert score_expedition([]) == 0
    assert score_expedition([card_id(0, 5)]) == 6 - 20  # the 6 card
    # 2..10 all played, no handshake: (54 - 20) = 34, +20 for 8+ cards
    full = [card_id(0, r) for r in range(1, 10)]
    assert score_expedition(full) == 54 - 20 + 20
    # one handshake then the 10: (10 - 20) * 2 = -20
    assert score_expedition([card_id(0, 0), card_id(0, 9)]) == -20
    # three handshakes alone: (0 - 20) * 4 = -80
    assert score_expedition([card_id(0, 0)] * 3) == -80


def test_a_turn_is_place_then_draw():
    state = LostCitiesState.new_game(seed=2)
    player = state.current_player
    legal = state.legal_actions()
    assert all(a < 100 for a in legal)  # place phase: plays and discards only
    state.apply(legal[0])
    assert state.current_player == player  # still my turn: the draw phase
    draw_legal = state.legal_actions()
    assert all(a >= 100 for a in draw_legal)
    state.apply(draw_legal[0])
    assert state.current_player == 1 - player
    assert state.hand_size[player] == HAND_SIZE


def test_expeditions_only_ascend():
    state = LostCitiesState.new_game(seed=3)
    player = state.current_player
    hand = [0] * NUM_CARDS
    hand[card_id(1, 5)] = 1  # the yellow 6
    hand[card_id(1, 3)] = 1  # the yellow 4
    hand[card_id(1, 0)] = 1  # a yellow handshake
    state.hands[player] = hand
    state.hand_size[player] = 3
    state.apply(card_id(1, 5))  # play the 6
    state.apply(DRAW_DECK)
    state.apply(state.legal_actions()[0])  # opponent places
    state.apply(DRAW_DECK)
    legal = state.legal_actions()
    assert card_id(1, 3) not in legal  # the 4 can no longer be played
    assert 50 + card_id(1, 3) in legal  # but it can be discarded
    assert card_id(1, 0) not in legal  # handshakes only before numbers


def test_you_may_not_redraw_the_card_you_just_discarded():
    state = LostCitiesState.new_game(seed=4)
    player = state.current_player
    discard = next(a for a in state.legal_actions() if a >= 50)
    color = (discard - 50) // 10
    state.apply(discard)
    legal = state.legal_actions()
    assert DRAW_DECK in legal
    assert 101 + color not in legal  # taking it straight back is a non-turn
    # but another pile with cards in it may be drawn from
    for c in range(5):
        if c != color and state.discards[c]:
            assert 101 + c in legal


def test_the_game_ends_when_the_deck_runs_out():
    rng = random.Random(0)
    for seed in range(8):
        state = LostCitiesState.new_game(seed=seed)
        moves = play_out(state, rng)
        assert state.is_terminal, f"seed {seed}: {moves} moves"
        assert len(state.deck) == 0
        assert state.outcome() in (1.0, 0.0, -1.0)
        # final scores match rescoring the expeditions from scratch
        for p in (0, 1):
            total = sum(score_expedition(pile) for pile in state.expeditions[p])
            assert state.scores[p] == total


# ------------------------------------------------ hidden information / search
def test_search_root_keeps_what_the_mover_can_see():
    state = LostCitiesState.new_game(seed=5)
    rng = random.Random(1)
    for _ in range(12):
        state.apply(rng.choice(state.legal_actions()))
    me = state.current_player
    view = state.search_root(random.Random(0))
    assert view.hands[me] == state.hands[me]
    assert view.discards == state.discards
    assert view.expeditions == state.expeditions
    assert view.hand_size == state.hand_size
    assert len(view.deck) == len(state.deck)
    assert census(view) == list(DECK_COUNTS)


def test_search_root_reshuffles_the_hidden_cards():
    state = LostCitiesState.new_game(seed=6)
    them = 1 - state.current_player
    hands = {tuple(state.search_root(random.Random(i)).hands[them]) for i in range(20)}
    assert len(hands) > 1


def test_drawing_from_the_deck_is_a_chance_event():
    state = LostCitiesState.new_game(seed=7)
    state.apply(state.legal_actions()[0])
    assert state.is_stochastic(DRAW_DECK)
    keys = {state.determinize(DRAW_DECK, s).chance_key() for s in range(20)}
    assert len(keys) > 1


def test_drawing_from_a_discard_pile_is_deterministic():
    state = LostCitiesState.new_game(seed=8)
    discard = next(a for a in state.legal_actions() if a >= 50)
    state.apply(discard)
    # give another pile a card so a deterministic draw exists
    other = ((discard - 50) // 10 + 1) % 5
    if state.discards[other]:
        assert not state.is_stochastic(101 + other)


def test_the_encoding_hides_the_opponents_hand():
    state = LostCitiesState.new_game(seed=9)
    encoded = state.encode()
    assert encoded.shape == (ENCODED_SIZE,)
    other = state.clone()
    them = 1 - state.current_player
    other.hands[them] = [0] * NUM_CARDS
    other.hands[them][card_id(4, 9)] = other.hand_size[them]
    assert np.array_equal(encoded, other.encode())


# ---------------------------------------------------------------- integration
def test_the_registry_serves_it():
    spec = get_game("lostcities")
    assert spec.action_space == ACTION_SPACE
    assert spec.encoded_size == ENCODED_SIZE
    assert isinstance(spec.new_game(1), LostCitiesState)


@pytest.mark.parametrize("spec", BASELINES)
def test_every_baseline_plays_only_legal_moves(spec):
    agent = make_agent(spec)
    agent.seed(1)
    state = LostCitiesState.new_game(seed=10)
    while not state.is_terminal:
        action = agent.act(state)
        assert state.is_legal(action)
        state.apply(action)
    assert census(state) == list(DECK_COUNTS)


def test_the_heuristic_beats_random():
    from ludometer.eval.arena import play_match

    match = play_match("lc:heuristic", "lc:random", n_games=30, base_seed=3,
                       game="lostcities")
    assert match.win_rate > 0.7, match.as_dict()
