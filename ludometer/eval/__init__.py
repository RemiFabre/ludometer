"""Evaluation: arena matches and Bradley-Terry Elo fitting (see docs/DESIGN.md)."""

from ludometer.eval.arena import (
    GameResult,
    MatchResult,
    play_game,
    play_match,
    round_robin,
)
from ludometer.eval.elo import EloFit, PairResult, expected_score, fit_elo

__all__ = [
    "EloFit",
    "GameResult",
    "MatchResult",
    "PairResult",
    "expected_score",
    "fit_elo",
    "play_game",
    "play_match",
    "round_robin",
]
