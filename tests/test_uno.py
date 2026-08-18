"""Uno engine, baselines and the framework seams that let it share the trainer."""

from __future__ import annotations

import random

import numpy as np
import pytest

from ludometer.agents import make_agent
from ludometer.eval.arena import play_game, play_match
from ludometer.games import get_game
from ludometer.uno.engine import (
    ACTION_SPACE,
    DECK_COUNTS,
    DRAW,
    ENCODED_SIZE,
    NUM_CARDS,
    TARGET_SCORE,
    WILD,
    WILD4,
    UnoState,
    card_points,
)

BASELINES = ("uno:random", "uno:greedy", "uno:heuristic")


def census(state: UnoState) -> list[int]:
    """Every card in the game, wherever it is."""
    counts = [0] * NUM_CARDS
    for player in (0, 1):
        for card, n in enumerate(state.hands[player]):
            counts[card] += n
    for card in state.deck:
        counts[card] += 1
    for card in state.discard:
        counts[card] += 1
    return counts


def play_out(state: UnoState, rng: random.Random, limit: int = 4000) -> int:
    moves = 0
    while not state.is_terminal and moves < limit:
        legal = state.legal_actions()
        assert legal, state.render_text()
        assert census(state) == list(DECK_COUNTS)
        state.apply(rng.choice(legal))
        moves += 1
    return moves


# ------------------------------------------------------------------- the rules
def test_the_deck_is_the_real_108_card_deck():
    assert sum(DECK_COUNTS) == 108
    assert DECK_COUNTS[WILD] == DECK_COUNTS[WILD4] == 4
    assert card_points(0) == 0 and card_points(9) == 9
    assert card_points(12) == 20 and card_points(WILD4) == 50


def test_random_matches_finish_and_never_lose_a_card():
    rng = random.Random(0)
    for seed in range(12):
        state = UnoState.new_game(seed=seed)
        moves = play_out(state, rng)
        assert state.is_terminal, f"seed {seed} ran {moves} moves"
        assert max(state.scores) >= TARGET_SCORE
        assert state.outcome() in (1.0, -1.0)
        assert len(state.segment_values) == state.hand_index + 1


def test_only_matching_cards_are_legal():
    state = UnoState.new_game(seed=4)
    top = state.discard[-1]
    for action in state.legal_actions():
        if action == DRAW or action >= 52:
            continue  # a draw, or a wild, which is always playable
        assert action % 13 == top % 13 or action // 13 == state.current_color


def test_drawing_is_the_only_move_when_nothing_matches():
    state = UnoState.new_game(seed=7)
    state.hands[state.current_player] = [0] * NUM_CARDS
    state.hands[state.current_player][(state.current_color + 1) % 4 * 13 + 5] = 1
    state.hand_size[state.current_player] = 1
    if state.discard[-1] % 13 != 5:
        assert state.legal_actions() == [DRAW]


def test_an_emptied_hand_scores_the_opponents_cards():
    state = UnoState.new_game(seed=11)
    player = state.current_player
    card = state.current_color * 13 + 5  # a plain number in the live color
    state.hands[player] = [0] * NUM_CARDS
    state.hands[player][card] = 1
    state.hand_size[player] = 1
    opponent = [0] * NUM_CARDS
    opponent[WILD4] = 1  # 50 points
    opponent[9] = 1  # a red 9
    state.hands[1 - player] = opponent
    state.hand_size[1 - player] = 2
    state.apply(card)
    assert state.scores[player] == 59
    assert state.segment_values[0] == (1.0 if player == 0 else -1.0)
    assert state.hand_index == 1  # a fresh hand was dealt


def test_one_hand_is_its_own_game():
    """The training episode: it ends when somebody goes out, and they won."""
    rng = random.Random(2)
    for seed in range(12):
        state = UnoState.new_game(seed=seed, hand_limit=1)
        play_out(state, rng)
        assert state.is_terminal
        assert state.hand_index == 0  # the one hand played, 0-indexed
        assert state.outcome() == state.segment_values[-1]
        assert state.outcome() in (1.0, -1.0)


def test_going_out_is_the_end_of_a_training_episode_not_a_reset():
    """The bug this rule exists to prevent: a value that resets mid-episode."""
    state = UnoState.new_game(seed=11, hand_limit=1)
    player = state.current_player
    card = state.current_color * 13 + 5
    state.hands[player] = [0] * NUM_CARDS
    state.hands[player][card] = 1
    state.hand_size[player] = 1
    state.hands[1 - player] = [0] * NUM_CARDS
    state.hands[1 - player][WILD4] = 1
    state.hand_size[1 - player] = 1
    state.apply(card)
    assert state.is_terminal
    assert state.outcome() == (1.0 if player == 0 else -1.0)


def test_a_clone_is_independent():
    state = UnoState.new_game(seed=3)
    twin = state.clone()
    play_out(twin, random.Random(1), limit=50)
    assert census(state) == list(DECK_COUNTS)
    assert state.hand_index == 0


# ------------------------------------------------ hidden information / search
def test_search_root_keeps_what_the_mover_can_see():
    state = UnoState.new_game(seed=5)
    for _ in range(9):
        state.apply(state.legal_actions()[0])
    me = state.current_player
    view = state.search_root(random.Random(0))
    assert view.hands[me] == state.hands[me]
    assert view.discard == state.discard
    assert view.hand_size == state.hand_size
    assert len(view.deck) == len(state.deck)
    assert census(view) == list(DECK_COUNTS)


def test_search_root_actually_reshuffles_the_hidden_cards():
    state = UnoState.new_game(seed=5)
    them = 1 - state.current_player
    draws = {tuple(state.search_root(random.Random(i)).hands[them]) for i in range(20)}
    assert len(draws) > 1


def test_a_draw_is_a_chance_event_with_several_outcomes():
    state = UnoState.new_game(seed=8)
    state.hands[state.current_player] = [0] * NUM_CARDS
    state.hand_size[state.current_player] = 0
    assert state.legal_actions() == [DRAW]
    assert state.is_stochastic(DRAW)
    keys = {state.determinize(DRAW, seed).chance_key() for seed in range(30)}
    assert len(keys) > 1


def test_playing_a_plain_number_is_deterministic():
    state = UnoState.new_game(seed=6)
    plain = [a for a in state.legal_actions() if a < 52 and a % 13 <= 9]
    if plain and state.hand_size[state.current_player] > 1:
        assert not state.is_stochastic(plain[0])


def test_the_encoding_ignores_the_match_score():
    """One net plays hands and matches, so nothing match-only may reach it."""
    state = UnoState.new_game(seed=2)
    other = state.clone()
    other.scores = [480, 30]
    other.hand_index = 9
    assert np.array_equal(state.encode(), other.encode())


def test_the_encoding_hides_the_opponents_cards():
    state = UnoState.new_game(seed=2)
    encoded = state.encode()
    assert encoded.shape == (ENCODED_SIZE,)
    assert encoded.dtype == np.float32
    other = state.clone()
    them = 1 - state.current_player
    other.hands[them] = [0] * NUM_CARDS
    other.hands[them][WILD4] = other.hand_size[them]
    assert np.array_equal(encoded, other.encode())


# ---------------------------------------------------------------- integration
def test_the_game_registry_agrees_with_the_engine():
    spec = get_game("uno")
    assert (spec.encoded_size, spec.action_space) == (ENCODED_SIZE, ACTION_SPACE)
    assert spec.state_cls is UnoState
    # the training spec is the same engine, same widths, one hand long
    hand = get_game("uno_hand")
    assert hand.encoded_size == spec.encoded_size
    assert hand.action_space == spec.action_space
    assert hand.new_game(1).hand_limit == 1
    assert spec.new_game(1).hand_limit > 1
    assert get_game(None).name == "azul"
    with pytest.raises(ValueError):
        get_game("chess")


@pytest.mark.parametrize("spec", BASELINES)
def test_every_baseline_plays_only_legal_moves(spec):
    agent = make_agent(spec)
    agent.seed(1)
    state = UnoState.new_game(seed=13)
    for _ in range(300):
        if state.is_terminal:
            break
        action = agent.act(state)
        assert state.is_legal(action)
        state.apply(action)
    assert census(state) == list(DECK_COUNTS)


def test_the_arena_plays_uno_when_asked():
    result = play_game("uno:greedy", "uno:random", seed=21, game="uno")
    assert not result.truncated
    assert max(result.a_score, result.b_score) >= TARGET_SCORE
    assert result.result in (0.0, 1.0)


def test_the_heuristic_beats_random_over_a_match():
    match = play_match(
        "uno:heuristic", "uno:random", n_games=30, base_seed=5, game="uno"
    )
    assert match.win_rate > 0.6, match.as_dict()


def test_self_play_plays_a_whole_hand_as_one_episode():
    from ludometer.train.mcts import MCTSConfig, UniformEvaluator
    from ludometer.train.selfplay import SelfPlayConfig, play_selfplay_game

    config = SelfPlayConfig(
        game="uno_hand", mcts=MCTSConfig(sims=4), max_moves=400, value_score_weight=0.0
    )
    record = play_selfplay_game(UniformEvaluator(), seed=2, config=config)
    assert not record.truncated
    assert record.states.shape[1] == ENCODED_SIZE
    assert record.policies.shape[1] == ACTION_SPACE
    # one episode, one label: every position carries its own seat's view of it
    assert set(np.abs(record.values).round(3).tolist()) == {1.0}


def test_illegal_applies_raise_like_azul():
    state = UnoState.new_game(seed=4)
    player = state.current_player
    hand = state.hands[player]
    # a held colored card that matches neither the color nor the top rank
    color = state.current_color
    rank = state.discard[-1] % 13
    bad = next(
        (
            c
            for c in range(52)
            if hand[c] and c // 13 != color and c % 13 != rank
        ),
        None,
    )
    if bad is not None:
        with pytest.raises(ValueError):
            state.clone().apply(bad)
    if state.legal_actions() != [DRAW]:
        with pytest.raises(ValueError):
            state.clone().apply(DRAW)


def test_the_hand_limit_backstop_goes_to_the_score_leader():
    """A rating match truncated at the limit belongs to whoever leads on
    points, even if the other player won the very last hand."""
    state = UnoState.new_game(seed=11, hand_limit=2)
    player = state.current_player
    card = state.current_color * 13 + 5
    state.hands[player] = [0] * NUM_CARDS
    state.hands[player][card] = 1
    state.hand_size[player] = 1
    state.apply(card)  # hand 1: `player` banks the points
    leader = player
    assert state.scores[leader] > 0 and not state.is_terminal
    # hand 2: the OTHER player goes out against an empty-ish hand (few points)
    other = 1 - leader
    state.current_player = other
    card2 = state.current_color * 13 + 5
    state.hands[other] = [0] * NUM_CARDS
    state.hands[other][card2] = 1
    state.hand_size[other] = 1
    state.hands[leader] = [0] * NUM_CARDS
    state.hands[leader][3] = 1  # a red 3: three points on the table
    state.hand_size[leader] = 1
    state.apply(card2)
    assert state.is_terminal
    if state.scores[leader] > state.scores[other]:
        assert state.outcome() == (1.0 if leader == 0 else -1.0)


def test_a_search_horizon_is_never_decided_by_the_carried_score():
    """The paid-for trap (NEXT_GAMES.md par.1): inside a match the truncated
    search world is decided by the CURRENT hand, never the match score."""
    state = UnoState.new_game(seed=3)
    state.scores = [0, 200] if state.current_player == 0 else [200, 0]
    root = state.search_root(random.Random(0))
    mover = root.current_player
    card = root.current_color * 13 + 5
    root.hands[mover] = [0] * NUM_CARDS
    root.hands[mover][card] = 1
    root.hand_size[mover] = 1
    root.apply(card)
    assert root.is_terminal
    assert root.outcome() == (1.0 if mover == 0 else -1.0)


def test_both_terminal_paths_report_the_same_hand_count():
    """hand_index counts finished hands the same way however the game ends."""
    one = UnoState.new_game(seed=6, hand_limit=1)
    play_out(one, random.Random(1))
    assert one.hand_index == 0  # one hand played, index of the last hand
    assert len(one.segment_values) == 1
    match = UnoState.new_game(seed=6)
    play_out(match, random.Random(1))
    assert len(match.segment_values) == match.hand_index + 1


def test_the_search_horizon_ends_with_the_current_hand():
    """Inside a match, a search must not bootstrap across a hand boundary."""
    state = UnoState.new_game(seed=5)
    for _ in range(60):
        if state.is_terminal or state.hand_index > 0:
            break
        state.apply(state.legal_actions()[0])
    root = state.search_root(random.Random(0))
    assert root.hand_limit == state.hand_index + 1
    # play the horizon out: the game ends with the hand, and the hand decides
    play_out(root, random.Random(3))
    assert root.is_terminal
    assert root.hand_index == state.hand_index  # the horizon hand, 0-indexed
    assert root.outcome() == root.segment_values[-1]
