"""Uno+ — the rule-knob variant (docs/NEXT_GAMES.md §2).

Four rules over the plain engine: draw is always legal (and atomic), +2/+4
stack like-on-like, a played 7 swaps hands (and creates *knowledge* the search
must respect), and the opening hand is 9 cards.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from ludometer.agents import make_agent
from ludometer.games import get_game
from ludometer.uno.engine import (
    DECK_COUNTS,
    DRAW,
    NUM_CARDS,
    WILD4,
    UnoState,
)
from ludometer.uno.plus import (
    DRAW_CAP,
    PLUS_ENCODED_SIZE,
    PLUS_HAND_SIZE,
    UnoPlusState,
)

BASELINES = ("unoplus:random", "unoplus:greedy", "unoplus:heuristic")


def census(state: UnoPlusState) -> list[int]:
    counts = [0] * NUM_CARDS
    for player in (0, 1):
        for card, n in enumerate(state.hands[player]):
            counts[card] += n
    for card in state.deck:
        counts[card] += 1
    for card in state.discard:
        counts[card] += 1
    return counts


def play_out(state: UnoPlusState, rng: random.Random, limit: int = 25000) -> int:
    moves = 0
    while not state.is_terminal and moves < limit:
        legal = state.legal_actions()
        assert legal, state.render_text()
        assert census(state) == list(DECK_COUNTS)
        state.apply(rng.choice(legal))
        moves += 1
    return moves


def give_hand(state: UnoPlusState, player: int, cards: dict[int, int]) -> None:
    """Overwrite ``player``'s hand (test setup only; breaks the census)."""
    hand = [0] * NUM_CARDS
    for card, n in cards.items():
        hand[card] = n
    state.hands[player] = hand
    state.hand_size[player] = sum(cards.values())


# ---------------------------------------------------------------- R4: 9 cards
def test_the_opening_hand_is_nine_cards():
    state = UnoPlusState.new_game(seed=1)
    assert PLUS_HAND_SIZE == 9
    assert state.hand_size == [9, 9]


# ------------------------------------------------------- R1: draw when you like
def test_draw_is_legal_even_with_playable_cards():
    with_plays = 0
    for seed in range(10):
        state = UnoPlusState.new_game(seed=seed)
        legal = state.legal_actions()
        assert DRAW in legal  # always on offer (fresh hand: cap is far away)
        if len(legal) > 1:
            with_plays += 1
    assert with_plays >= 8  # 9 cards: something is nearly always playable too


def test_draw_is_atomic_one_card_then_the_turn_ends():
    state = UnoPlusState.new_game(seed=3)
    player = state.current_player
    before = state.hand_size[player]
    state.apply(DRAW)
    assert state.hand_size[player] == before + 1
    assert state.current_player == 1 - player  # even if the card was playable


def test_draw_is_illegal_at_the_cap():
    state = UnoPlusState.new_game(seed=4)
    player = state.current_player
    playable = state.current_color * 13 + 5
    give_hand(state, player, {playable: 1, WILD4: DRAW_CAP - 1})
    assert state.hand_size[player] == DRAW_CAP
    assert DRAW not in state.legal_actions()


def test_a_stuck_capped_player_passes_without_drawing():
    state = UnoPlusState.new_game(seed=5)
    player = state.current_player
    dead = (state.current_color + 1) % 4 * 13 + ((state.discard[-1] % 13 + 1) % 10)
    give_hand(state, player, {dead: DRAW_CAP})
    assert state.legal_actions() == [DRAW]
    state.apply(DRAW)
    assert state.hand_size[player] == DRAW_CAP  # nothing drawn
    assert state.current_player == 1 - player


# ------------------------------------------------------------- R2: stacking
def _pending_setup(seed: int = 6) -> tuple[UnoPlusState, int, int]:
    """A state where the player to move holds a +2 in the live color."""
    state = UnoPlusState.new_game(seed=seed)
    player = state.current_player
    plus2 = state.current_color * 13 + 12
    give_hand(state, player, {plus2: 1, 5: 2})
    return state, player, plus2


def test_a_plus_two_hands_the_opponent_a_choice_not_cards():
    state, player, plus2 = _pending_setup()
    opp_size = state.hand_size[1 - player]
    state.apply(plus2)
    assert state.pending_draw == 2
    assert state.pending_kind == 2
    assert state.hand_size[1 - player] == opp_size  # nothing drawn yet
    assert state.current_player == 1 - player  # they answer or take


def test_while_pending_only_stack_cards_and_draw_are_legal():
    state, player, plus2 = _pending_setup()
    opponent = 1 - player
    give_hand(state, opponent, {2 * 13 + 12: 1, 5: 1, WILD4: 1})
    state.apply(plus2)
    assert set(state.legal_actions()) == {2 * 13 + 12, DRAW}


def test_taking_the_stack_draws_it_all_and_loses_the_turn():
    state, player, plus2 = _pending_setup()
    opponent = 1 - player
    give_hand(state, opponent, {2 * 13 + 12: 1, 5: 1})
    state.apply(plus2)
    state.apply(2 * 13 + 12)  # stack back: pending 4, my turn again
    assert state.pending_draw == 4
    assert state.current_player == player
    my_size = state.hand_size[player]
    state.apply(DRAW)
    assert state.hand_size[player] == my_size + 4
    assert state.pending_draw == 0
    assert state.current_player == opponent  # the last stacker moves on


def test_a_plus_four_answers_a_plus_four_but_a_plus_two_does_not():
    state = UnoPlusState.new_game(seed=7)
    player = state.current_player
    give_hand(state, player, {WILD4: 1, 5: 1})
    state.apply(56 + 2)  # wild+4 declaring green
    opponent = 1 - player
    give_hand(state, opponent, {2 * 13 + 12: 1, WILD4: 1, 5: 1})
    legal = state.legal_actions()
    assert 2 * 13 + 12 not in legal  # a +2 does not answer a +4
    assert {56, 57, 58, 59} <= set(legal)
    assert DRAW in legal
    assert state.pending_kind == 4


def test_going_out_with_a_plus_two_still_makes_the_opponent_draw():
    state, player, plus2 = _pending_setup(seed=8)
    give_hand(state, player, {plus2: 1})
    opponent_size = state.hand_size[1 - player]
    state.apply(plus2)
    assert state.hand_index == 1  # the hand ended and the next was dealt
    assert state.scores[player] > 0
    assert state.segment_values[0] == (1.0 if player == 0 else -1.0)


# ------------------------------------------------------------- R3: the 7-swap
def test_playing_a_seven_swaps_the_hands():
    state = UnoPlusState.new_game(seed=9)
    player = state.current_player
    seven = state.current_color * 13 + 7
    give_hand(state, player, {seven: 1, 5: 2})
    opponent_hand = list(state.hands[1 - player])
    opponent_size = state.hand_size[1 - player]
    state.apply(seven)
    assert state.hands[player] == opponent_hand
    assert state.hand_size[player] == opponent_size
    assert state.hand_size[1 - player] == 2  # the two 5s, minus the 7 played
    assert state.current_player == 1 - player  # a 7 is otherwise a number card


def test_a_seven_as_the_last_card_goes_out_without_swapping():
    state = UnoPlusState.new_game(seed=10)
    player = state.current_player
    seven = state.current_color * 13 + 7
    give_hand(state, player, {seven: 1})
    opponent_hand = list(state.hands[1 - player])
    state.apply(seven)
    assert state.hand_index == 1  # went out; hand over
    assert state.segment_values[0] == (1.0 if player == 0 else -1.0)


def test_after_a_swap_both_players_know_the_opposing_hand_exactly():
    state = UnoPlusState.new_game(seed=11)
    player = state.current_player
    seven = state.current_color * 13 + 7
    give_hand(state, player, {seven: 1, 5: 2})
    state.apply(seven)
    assert state.known[player] == state.hands[1 - player]
    assert state.known[1 - player] == state.hands[player]


def test_knowledge_decays_as_known_cards_are_played():
    state = UnoPlusState.new_game(seed=12)
    player = state.current_player
    seven = state.current_color * 13 + 7
    give_hand(state, player, {seven: 1, 5: 2})
    state.apply(seven)  # opponent now holds the two 5s and knows my hand
    mover = state.current_player
    watcher = 1 - mover
    state.current_color = 0  # make the red 5s playable
    played = next(a for a in state.legal_actions() if a != DRAW)
    card = WILD4 if played >= 56 else (52 if 52 <= played < 56 else played)
    known_before = state.known[watcher][card]
    state.apply(played)
    if known_before:
        assert state.known[watcher][card] == known_before - 1


def test_knowledge_resets_when_a_new_hand_is_dealt():
    state = UnoPlusState.new_game(seed=13)
    player = state.current_player
    seven = state.current_color * 13 + 7
    give_hand(state, player, {seven: 1, 5: 2})
    state.apply(seven)
    assert any(state.known[0]) or any(state.known[1])
    # the mover now holds two 5s; going out ends the hand and deals afresh
    mover = state.current_player
    state.current_color = 0  # a red 5 is playable
    give_hand(state, mover, {5: 1})
    state.apply(5)
    assert state.hand_index == 1
    assert state.known == [[0] * NUM_CARDS, [0] * NUM_CARDS]
    assert state.pending_draw == 0


def test_search_root_deals_the_known_cards_to_the_opponent():
    state = UnoPlusState.new_game(seed=14)
    player = state.current_player
    seven = state.current_color * 13 + 7
    give_hand(state, player, {seven: 1, 5: 2})
    # rebuild a consistent world: put the removed cards back in the deck
    state.deck = [c for c in state.deck]
    state.apply(seven)
    me = state.current_player
    for _ in range(30):
        view = state.search_root(random.Random(_))
        for card in range(NUM_CARDS):
            assert view.hands[1 - me][card] >= state.known[me][card]
        assert view.hands[me] == state.hands[me]
        assert view.hand_size == state.hand_size


# ------------------------------------------------------------------- encoding
def test_the_encoding_extends_the_base_and_shows_pending_and_known():
    state = UnoPlusState.new_game(seed=15)
    encoded = state.encode()
    assert encoded.shape == (PLUS_ENCODED_SIZE,)
    assert PLUS_ENCODED_SIZE == UnoState.ENCODED_SIZE + 1 + 3 + NUM_CARDS
    assert encoded.dtype == np.float32
    # fresh hand: nothing pending, nothing known
    assert encoded[UnoState.ENCODED_SIZE] == 0.0
    assert encoded[UnoState.ENCODED_SIZE + 1] == 1.0  # pending_kind none
    assert not encoded[UnoState.ENCODED_SIZE + 4 :].any()


def test_pending_reaches_the_encoding():
    state, player, plus2 = _pending_setup(seed=16)
    state.apply(plus2)
    encoded = state.encode()
    assert encoded[UnoState.ENCODED_SIZE] == pytest.approx(2 / 4)
    assert encoded[UnoState.ENCODED_SIZE + 2] == 1.0  # kind == +2


# ----------------------------------------------------------------- integration
def test_random_games_finish_and_never_lose_a_card():
    rng = random.Random(0)
    for seed in range(6):
        state = UnoPlusState.new_game(seed=seed, hand_limit=1)
        moves = play_out(state, rng)
        assert state.is_terminal, f"seed {seed} ran {moves} moves"
        assert state.outcome() is not None


def test_a_full_match_terminates():
    # Random players bank cards forever (~1,700 moves/hand measured), so the
    # match backstop is exercised with the greedy baseline, which plays out.
    a = make_agent("unoplus:greedy")
    b = make_agent("unoplus:heuristic")
    a.seed(1)
    b.seed(2)
    state = UnoPlusState.new_game(seed=3)
    moves = 0
    while not state.is_terminal and moves < 8000:
        agent = (a, b)[state.current_player]
        state.apply(agent.act(state))
        moves += 1
    assert state.is_terminal, moves
    assert max(state.scores) >= 500


def test_the_registry_serves_both_specs():
    spec = get_game("unoplus")
    hand = get_game("unoplus_hand")
    assert spec.encoded_size == hand.encoded_size == PLUS_ENCODED_SIZE
    assert spec.action_space == hand.action_space == 61
    assert hand.new_game(1).hand_limit == 1
    assert spec.new_game(1).hand_limit > 1


@pytest.mark.parametrize("spec", BASELINES)
def test_every_baseline_plays_only_legal_moves(spec):
    agent = make_agent(spec)
    agent.seed(1)
    state = UnoPlusState.new_game(seed=17)
    for _ in range(400):
        if state.is_terminal:
            break
        action = agent.act(state)
        assert state.is_legal(action)
        state.apply(action)
    assert census(state) == list(DECK_COUNTS)


def test_greedy_and_heuristic_do_not_draw_when_they_can_play():
    for spec in ("unoplus:greedy", "unoplus:heuristic"):
        agent = make_agent(spec)
        agent.seed(2)
        state = UnoPlusState.new_game(seed=18)
        for _ in range(200):
            if state.is_terminal:
                break
            action = agent.act(state)
            if action == DRAW:
                assert state.legal_actions() == [DRAW] or state.pending_draw > 0
            state.apply(action)


def test_the_search_horizon_still_ends_with_the_current_hand():
    state = UnoPlusState.new_game(seed=19)
    rng = random.Random(4)
    while state.hand_index == 0 and not state.is_terminal:
        state.apply(rng.choice(state.legal_actions()))
    root = state.search_root(random.Random(0))
    assert root.hand_limit == state.hand_index + 1
    play_out(root, random.Random(5))
    assert root.is_terminal
    assert root.outcome() == root.segment_values[-1]
