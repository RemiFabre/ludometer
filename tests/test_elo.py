"""Tests for the Bradley-Terry Elo fit (see docs/DESIGN.md, "Evaluation")."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ludometer.eval.elo import (
    PairResult,
    expected_score,
    fit_elo,
    winrate_to_elo,
)

TRUE_ELO = {"random": 0.0, "weak": 120.0, "mid": 350.0, "strong": 610.0}


def synthetic_results(
    true_elo: dict[str, float], n_games: int = 800, seed: int = 0
) -> list[PairResult]:
    """Sample every pairing from the true Bradley-Terry model."""
    rng = np.random.default_rng(seed)
    names = list(true_elo)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            p = expected_score(true_elo[a], true_elo[b])
            wins = int(rng.binomial(n_games, p))
            out.append(PairResult(a, b, float(wins), 0.0, n_games - wins))
    return out


# ------------------------------------------------------------------- recovery
def test_recovers_synthetic_ratings():
    results = synthetic_results(TRUE_ELO, n_games=1200, seed=1)
    fit = fit_elo(results, anchors={"random": 0.0})
    assert fit.converged
    for name, true in TRUE_ELO.items():
        assert fit.ratings[name] == pytest.approx(true, abs=40.0), fit.table()
    # error bars should cover the truth and be of a sensible size
    for name, true in TRUE_ELO.items():
        if name == "random":
            continue
        assert 0.0 < fit.errors[name] < 60.0
        assert abs(fit.ratings[name] - true) < 3.5 * fit.errors[name]


def test_recovery_is_stable_across_samples():
    fits = [
        fit_elo(synthetic_results(TRUE_ELO, n_games=1000, seed=s), {"random": 0.0})
        for s in range(4)
    ]
    for name, true in TRUE_ELO.items():
        mean = sum(f.ratings[name] for f in fits) / len(fits)
        assert mean == pytest.approx(true, abs=30.0)
    order = sorted(fits[0].ratings, key=lambda n: fits[0].ratings[n])
    assert order == ["random", "weak", "mid", "strong"]


def test_more_games_shrink_the_error():
    small = fit_elo(synthetic_results(TRUE_ELO, n_games=100, seed=3), {"random": 0.0})
    large = fit_elo(synthetic_results(TRUE_ELO, n_games=2000, seed=3), {"random": 0.0})
    for name in TRUE_ELO:
        if name == "random":
            continue
        assert large.errors[name] < small.errors[name]


def test_draws_count_as_half_wins():
    """Two draws must be worth exactly one win plus one loss."""
    decisive = fit_elo([PairResult("a", "random", 30.0, 0.0, 10.0)], {"random": 0.0})
    drawish = fit_elo([PairResult("a", "random", 25.0, 10.0, 5.0)], {"random": 0.0})
    assert drawish.ratings["a"] == pytest.approx(decisive.ratings["a"], abs=1e-6)
    assert drawish.errors["a"] == pytest.approx(decisive.errors["a"], abs=1e-6)
    # all-draw results mean equal strength
    even = fit_elo(
        [PairResult("a", "b", 0.0, 200.0, 0.0), PairResult("b", "c", 0.0, 200.0, 0.0)],
        anchors={"a": 0.0},
    )
    assert even.ratings["b"] == pytest.approx(0.0, abs=1.0)
    assert even.ratings["c"] == pytest.approx(0.0, abs=1.0)


# --------------------------------------------------------------------- anchors
def test_anchor_stays_fixed():
    results = synthetic_results(TRUE_ELO, n_games=400, seed=7)
    fit = fit_elo(results, anchors={"random": 0.0})
    assert fit.ratings["random"] == 0.0
    assert fit.errors["random"] == 0.0

    shifted = fit_elo(results, anchors={"random": 1500.0})
    assert shifted.ratings["random"] == 1500.0
    for name in TRUE_ELO:
        assert shifted.ratings[name] == pytest.approx(
            fit.ratings[name] + 1500.0, abs=1e-3
        )


def test_multiple_anchors_are_all_held():
    results = synthetic_results(TRUE_ELO, n_games=600, seed=11)
    anchors = {"random": 0.0, "mid": 350.0}
    fit = fit_elo(results, anchors=anchors)
    assert fit.ratings["random"] == 0.0
    assert fit.ratings["mid"] == 350.0
    assert fit.ratings["strong"] == pytest.approx(610.0, abs=60.0)
    assert fit.anchors == anchors


def test_all_anchored_is_a_no_op():
    fit = fit_elo(
        [PairResult("a", "b", 10.0, 0.0, 5.0)], anchors={"a": 100.0, "b": -100.0}
    )
    assert fit.ratings == {"a": 100.0, "b": -100.0}
    assert fit.converged


def test_no_anchor_centres_on_zero():
    results = synthetic_results(TRUE_ELO, n_games=800, seed=13)
    fit = fit_elo(results, anchors=None)
    mean = sum(fit.ratings.values()) / len(fit.ratings)
    assert mean == pytest.approx(0.0, abs=1e-6)
    offset = sum(TRUE_ELO.values()) / len(TRUE_ELO)
    for name, true in TRUE_ELO.items():
        assert fit.ratings[name] == pytest.approx(true - offset, abs=45.0)


def test_unknown_anchor_is_an_error():
    with pytest.raises(ValueError, match="anchors not present"):
        fit_elo([PairResult("a", "b", 1.0, 0.0, 1.0)], anchors={"ghost": 0.0})


# ------------------------------------------------------------------ edge cases
def test_perfect_score_stays_finite():
    fit = fit_elo(
        [PairResult("winner", "random", 300.0, 0.0, 0.0)], anchors={"random": 0.0}
    )
    assert fit.ratings["random"] == 0.0
    assert 500.0 < fit.ratings["winner"] < 3000.0
    assert math.isfinite(fit.errors["winner"])


def test_input_coercion_forms_agree():
    reference = fit_elo(
        [PairResult("a", "random", 30.0, 4.0, 6.0)], anchors={"random": 0.0}
    )
    tuples = fit_elo([("a", "random", 30.0, 4.0, 6.0)], anchors={"random": 0.0})
    dicts = fit_elo(
        [{"a": "a", "b": "random", "wins": 30, "draws": 4, "losses": 6}],
        anchors={"random": 0.0},
    )
    triples = fit_elo([("a", "random", 32.0, 8.0)], anchors={"random": 0.0})
    assert tuples.ratings == pytest.approx(reference.ratings)
    assert dicts.ratings == pytest.approx(reference.ratings)
    assert triples.ratings["a"] == pytest.approx(reference.ratings["a"], abs=1e-6)
    with pytest.raises(ValueError):
        fit_elo([("a", "b", 1.0)], anchors={})
    with pytest.raises(TypeError):
        fit_elo([object()], anchors={})
    with pytest.raises(ValueError, match="self-play"):
        fit_elo([PairResult("a", "a", 1.0, 0.0, 1.0)], anchors={})


def test_match_results_are_accepted():
    from ludometer.eval.arena import play_match

    matches = [play_match("greedy", "random", n_games=4, base_seed=0)]
    fit = fit_elo(matches, anchors={"random": 0.0})
    assert fit.ratings["greedy"] > 0.0
    assert set(fit.players) == {"greedy", "random"}
    assert fit.n_games["greedy"] == 4 + 0.5  # includes the prior draws


def test_empty_input():
    fit = fit_elo([], anchors=None)
    assert fit.ratings == {} and fit.players == []


def test_table_renders():
    fit = fit_elo(synthetic_results(TRUE_ELO, n_games=100, seed=2), {"random": 0.0})
    text = fit.table()
    assert "(anchor)" in text
    assert text.splitlines()[0].startswith("strong")
    assert fit["strong"] == fit.ratings["strong"]


# -------------------------------------------------------------------- bootstrap
def test_bootstrap_errors_agree_with_fisher():
    results = synthetic_results(TRUE_ELO, n_games=500, seed=17)
    fisher = fit_elo(results, {"random": 0.0}, error_method="fisher")
    boot = fit_elo(
        results, {"random": 0.0}, error_method="bootstrap", n_bootstrap=60, seed=4
    )
    assert boot.ratings == pytest.approx(fisher.ratings)
    for name in TRUE_ELO:
        if name == "random":
            continue
        assert boot.errors[name] == pytest.approx(fisher.errors[name], rel=0.5)


def test_unknown_error_method():
    with pytest.raises(ValueError, match="unknown error_method"):
        fit_elo(
            [PairResult("a", "random", 5.0, 0.0, 5.0)],
            anchors={"random": 0.0},
            error_method="magic",
        )


# --------------------------------------------------------------------- helpers
def test_expected_score_and_inverse():
    assert expected_score(0.0, 0.0) == pytest.approx(0.5)
    assert expected_score(400.0, 0.0) == pytest.approx(10.0 / 11.0)
    for elo in (-500.0, -75.0, 0.0, 120.0, 800.0):
        assert winrate_to_elo(expected_score(elo, 0.0)) == pytest.approx(elo, abs=1e-6)
    assert winrate_to_elo(1.0) > 1000.0
    assert winrate_to_elo(0.0) < -1000.0
