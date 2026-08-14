"""Tests for the arena (see docs/DESIGN.md, "Evaluation").

Determinism is the contract that matters: an Elo curve is only comparable across
checkpoints if a match is a pure function of its seeds.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from ludometer.agents import Agent, GreedyAgent, RandomAgent
from ludometer.azul.engine import AzulState
from ludometer.eval.arena import (
    GameResult,
    play_game,
    play_match,
    round_robin,
)


class FirstAction(Agent):
    """Deterministic, seed-independent: always the lowest legal action id."""

    name = "first"

    def act(self, state: AzulState) -> int:
        return min(state.legal_actions())


class IllegalAgent(Agent):
    name = "illegal"

    def act(self, state: AzulState) -> int:
        return 179 if 179 not in state.legal_actions() else 0


# ------------------------------------------------------------------- one game
def test_play_game_is_deterministic():
    a = play_game("greedy", "random", seed=7)
    b = play_game("greedy", "random", seed=7)
    assert asdict(a) == asdict(b)
    assert a.moves > 20
    assert a.rounds >= 2
    assert a.result in (0.0, 0.5, 1.0)
    assert a.score_diff == a.a_score - a.b_score


def test_play_game_seed_changes_the_game():
    games = {play_game("random", "random", seed=s).seed for s in range(5)}
    assert games == set(range(5))
    scores = [
        (g.a_score, g.b_score)
        for g in (play_game("random", "random", seed=s) for s in range(8))
    ]
    assert len(set(scores)) > 1


def test_seats_swap_with_a_first():
    """Same deal, swapped seats: the results are mirror images of each other."""
    first = play_game(FirstAction(), "greedy", seed=3, a_first=True)
    second = play_game("greedy", FirstAction(), seed=3, a_first=False)
    assert first.a_score == second.b_score
    assert first.b_score == second.a_score
    assert first.result == pytest.approx(1.0 - second.result)


def test_play_game_accepts_instances_and_specs():
    by_spec = play_game("greedy", "random", seed=11)
    by_instance = play_game(GreedyAgent(), RandomAgent(), seed=11)
    assert asdict(by_spec) == asdict(by_instance)


def test_illegal_action_is_rejected():
    with pytest.raises(ValueError, match="illegal action"):
        play_game(IllegalAgent(), "random", seed=1)


# ---------------------------------------------------------------------- match
def test_play_match_alternates_seats_and_pairs_deals():
    match = play_match("greedy", "random", n_games=6, base_seed=100, keep_games=True)
    assert [g.a_first for g in match.games] == [True, False] * 3
    assert [g.seed for g in match.games] == [100, 100, 101, 101, 102, 102]
    assert match.wins + match.draws + match.losses == 6
    assert match.n_games == 6


def test_play_match_is_deterministic():
    kwargs = {"n_games": 8, "base_seed": 42}
    first = play_match("greedy", "random", **kwargs)
    second = play_match("greedy", "random", **kwargs)
    assert first.as_dict() == second.as_dict()
    other = play_match("greedy", "random", n_games=8, base_seed=43)
    assert other.as_dict() != first.as_dict()


def test_play_match_multiprocessing_matches_single_process():
    serial = play_match("greedy", "random", n_games=6, base_seed=9, n_workers=1)
    parallel = play_match("greedy", "random", n_games=6, base_seed=9, n_workers=3)
    assert serial.as_dict() == parallel.as_dict()


def test_match_orientation_is_symmetric():
    forward = play_match("greedy", "random", n_games=8, base_seed=77)
    backward = play_match("random", "greedy", n_games=8, base_seed=77)
    assert forward.wins == backward.losses
    assert forward.losses == backward.wins
    assert forward.draws == backward.draws
    assert forward.mean_score_diff == pytest.approx(-backward.mean_score_diff)


def test_match_rates():
    match = play_match("greedy", "random", n_games=4, base_seed=1)
    expected = (match.wins + 0.5 * match.draws) / match.n_games
    assert match.win_rate == pytest.approx(expected)
    assert match.decisive_win_rate == pytest.approx(match.wins / match.n_games)
    assert match.mean_score_a - match.mean_score_b == pytest.approx(
        match.mean_score_diff
    )
    assert match.name_a == "greedy" and match.name_b == "random"


def test_empty_match_is_harmless():
    match = play_match("greedy", "random", n_games=0)
    assert match.n_games == 0 and match.win_rate == 0.0
    with pytest.raises(ValueError):
        play_match("greedy", "random", n_games=-1)


def test_game_result_is_picklable():
    import pickle

    result = play_game("greedy", "random", seed=2)
    assert pickle.loads(pickle.dumps(result)) == result
    assert isinstance(result, GameResult)


# ----------------------------------------------------------------- round robin
def test_round_robin_covers_every_pair():
    matches = round_robin(["random", "greedy", "heuristic"], n_games=2, base_seed=0)
    pairs = {(m.name_a, m.name_b) for m in matches}
    assert pairs == {
        ("random", "greedy"),
        ("random", "heuristic"),
        ("greedy", "heuristic"),
    }
    assert all(m.n_games == 2 for m in matches)
    # independent seed blocks per pairing
    assert len({m.base_seed for m in matches}) == 3
