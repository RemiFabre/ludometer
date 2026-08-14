"""Bradley-Terry maximum-likelihood Elo fit with fixed anchors (pure numpy).

The model is the usual logistic one: player ``i`` beats player ``j`` with
probability ``1 / (1 + 10 ** (-(r_i - r_j) / 400))``, and a draw counts as half a
win for both sides. Ratings are fitted by Newton's method on the log-likelihood
with *anchor* players held fixed (docs/DESIGN.md: Random is pinned at 0 Elo), so
every checkpoint's rating lives on the same scale across a whole run.

Results are arbitrary pairwise aggregates — any set of pairings works, the graph
just has to connect each free player to an anchor (directly or not).

Errors come either from the Fisher information (default, cheap) or from a
parametric bootstrap over games.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "EloFit",
    "PairResult",
    "expected_score",
    "fit_elo",
    "winrate_to_elo",
]

SCALE = 400.0 / math.log(10.0)  # Elo points per logit


@dataclass(frozen=True)
class PairResult:
    """``a`` and ``b`` played ``wins + draws + losses`` games (a's viewpoint)."""

    a: str
    b: str
    wins: float
    draws: float = 0.0
    losses: float = 0.0

    @property
    def n_games(self) -> float:
        return self.wins + self.draws + self.losses


@dataclass
class EloFit:
    """Result of :func:`fit_elo`."""

    ratings: dict[str, float]
    errors: dict[str, float]
    players: list[str]
    anchors: dict[str, float]
    log_likelihood: float
    iterations: int
    converged: bool
    n_games: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, player: str) -> float:
        return self.ratings[player]

    def table(self) -> str:
        """Ratings sorted from strongest to weakest, one per line."""
        rows = sorted(self.ratings.items(), key=lambda kv: -kv[1])
        width = max((len(n) for n in self.ratings), default=6)
        out = []
        for name, elo in rows:
            tag = " (anchor)" if name in self.anchors else ""
            out.append(
                f"{name:<{width}}  {elo:+8.1f} +/- {self.errors.get(name, 0.0):5.1f}"
                f"  {self.n_games.get(name, 0.0):7.0f} games{tag}"
            )
        return "\n".join(out)


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score (win + half draw) of A against B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def winrate_to_elo(win_rate: float, clip: float = 1e-3) -> float:
    """Elo difference implied by a score share (clipped away from 0 and 1)."""
    p = min(max(win_rate, clip), 1.0 - clip)
    return 400.0 * math.log10(p / (1.0 - p))


# ------------------------------------------------------------------ input glue
def _coerce(result: Any) -> PairResult:
    if isinstance(result, PairResult):
        return result
    if isinstance(result, Mapping):
        a = result.get("a", result.get("name_a"))
        b = result.get("b", result.get("name_b"))
        return PairResult(
            str(a),
            str(b),
            float(result.get("wins", 0.0)),
            float(result.get("draws", 0.0)),
            float(result.get("losses", 0.0)),
        )
    if isinstance(result, Sequence) and not isinstance(result, str):
        if len(result) == 4:
            a, b, wins, losses = result
            return PairResult(str(a), str(b), float(wins), 0.0, float(losses))
        if len(result) == 5:
            a, b, wins, draws, losses = result
            return PairResult(str(a), str(b), float(wins), float(draws), float(losses))
        raise ValueError(f"expected 4 or 5 fields, got {result!r}")
    # duck-typed MatchResult
    for attr in ("name_a", "name_b", "wins", "draws", "losses"):
        if not hasattr(result, attr):
            raise TypeError(f"cannot read a pairwise result from {result!r}")
    return PairResult(
        str(result.name_a),
        str(result.name_b),
        float(result.wins),
        float(result.draws),
        float(result.losses),
    )


def _build_matrices(results: Iterable[Any]) -> tuple[list[str], np.ndarray]:
    """Return the player list and ``W[i, j]`` = score of i against j."""
    pairs = [_coerce(r) for r in results]
    names: list[str] = []
    seen: set[str] = set()
    for p in pairs:
        for name in (p.a, p.b):
            if name not in seen:
                seen.add(name)
                names.append(name)
    names.sort()
    index = {name: i for i, name in enumerate(names)}
    n = len(names)
    scores = np.zeros((n, n), dtype=np.float64)
    for p in pairs:
        i, j = index[p.a], index[p.b]
        if i == j:
            raise ValueError(f"self-play result for {p.a!r} is not usable")
        scores[i, j] += p.wins + 0.5 * p.draws
        scores[j, i] += p.losses + 0.5 * p.draws
    return names, scores


# ------------------------------------------------------------------------- fit
def _log_likelihood(ratings: np.ndarray, scores: np.ndarray) -> float:
    diff = (ratings[:, None] - ratings[None, :]) / SCALE
    # log p_ij = -log(1 + exp(-diff)) computed stably
    log_p = -np.logaddexp(0.0, -diff)
    mask = scores > 0.0
    return float(np.sum(scores[mask] * log_p[mask]))


def _grad_hess(
    ratings: np.ndarray, scores: np.ndarray, games: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    diff = (ratings[:, None] - ratings[None, :]) / SCALE
    p = 1.0 / (1.0 + np.exp(-diff))
    grad = np.sum(scores - games * p, axis=1) / SCALE
    wt = games * p * (1.0 - p) / (SCALE * SCALE)
    hess = wt.copy()
    np.fill_diagonal(hess, 0.0)
    diag = -np.sum(wt, axis=1)
    np.fill_diagonal(hess, diag)
    return grad, hess


def fit_elo(
    results: Iterable[Any],
    anchors: Mapping[str, float] | None = None,
    prior_draws: float = 0.5,
    max_iter: int = 200,
    tol: float = 1e-8,
    error_method: str = "fisher",
    n_bootstrap: int = 200,
    seed: int = 0,
) -> EloFit:
    """Fit Bradley-Terry ratings on the Elo scale.

    Args:
        results: pairwise aggregates. Each item may be a :class:`PairResult`, a
            ``(a, b, wins, draws, losses)`` tuple, a dict with those keys (or
            ``name_a``/``name_b``), or any object exposing
            ``name_a/name_b/wins/draws/losses`` (e.g. a
            :class:`~ludometer.eval.arena.MatchResult`).
        anchors: ratings held fixed, e.g. ``{"random": 0.0}``. Anchors must appear
            in ``results`` (a typo would otherwise pass silently). When empty, the
            fit is centred on mean zero instead (the model is shift-invariant).
        prior_draws: virtual drawn games added to every pairing that played. Keeps
            100%-score pairings finite; 0.5 is negligible past a few dozen games.
        error_method: ``"fisher"`` (inverse observed information),
            ``"bootstrap"`` (refit ``n_bootstrap`` resampled result sets) or
            ``"none"`` (skip, all errors reported as 0).

    Returns:
        An :class:`EloFit` with ratings, one-sigma errors and diagnostics.
    """
    anchors = dict(anchors or {})
    names, scores = _build_matrices(results)
    unknown = set(anchors) - set(names)
    if unknown:
        raise ValueError(f"anchors not present in the results: {sorted(unknown)}")
    n = len(names)
    if n == 0:
        return EloFit({}, {}, [], anchors, 0.0, 0, True, {})

    games = scores + scores.T
    if prior_draws > 0.0:
        played = games > 0.0
        half = 0.5 * prior_draws
        scores = scores + half * played
        games = games + prior_draws * played

    free = np.array([i for i, nm in enumerate(names) if nm not in anchors], dtype=int)

    ratings = np.zeros(n, dtype=np.float64)
    for name, value in anchors.items():
        ratings[names.index(name)] = float(value)
    # warm start: each free player at the anchor mean shifted by their score share
    anchor_mean = float(np.mean([anchors[nm] for nm in anchors])) if anchors else 0.0
    for i in free:
        played = games[i].sum()
        if played > 0.0:
            share = scores[i].sum() / played
            ratings[i] = anchor_mean + winrate_to_elo(share, clip=0.05)

    converged = False
    iterations = 0
    ll = _log_likelihood(ratings, scores)
    if free.size:
        for iterations in range(1, max_iter + 1):
            grad, hess = _grad_hess(ratings, scores, games)
            g = grad[free]
            h = hess[np.ix_(free, free)]
            a = -h + 1e-9 * np.eye(free.size)
            if not anchors:
                # pin the mean: the likelihood is invariant to a global shift
                a = a + np.ones((free.size, free.size)) / free.size
                g = g - g.mean()
            try:
                step = np.linalg.solve(a, g)
            except np.linalg.LinAlgError:  # pragma: no cover - degenerate input
                step = np.linalg.lstsq(a, g, rcond=None)[0]
            # damped Newton: shrink until the likelihood improves
            scale = 1.0
            for _ in range(40):
                trial = ratings.copy()
                trial[free] += scale * step
                if not anchors:
                    trial[free] -= trial[free].mean()
                trial_ll = _log_likelihood(trial, scores)
                if trial_ll >= ll - 1e-12:
                    break
                scale *= 0.5
            else:  # pragma: no cover - no improving step found
                break
            moved = float(np.max(np.abs(scale * step)))
            ratings, ll = trial, trial_ll
            if moved < tol * max(1.0, float(np.max(np.abs(ratings)))) or moved < 1e-10:
                converged = True
                break
    else:
        converged = True

    errors = {nm: 0.0 for nm in names}
    if free.size and error_method != "none":
        if error_method == "fisher":
            errs = _fisher_errors(ratings, games, free)
        elif error_method == "bootstrap":
            errs = _bootstrap_errors(
                names,
                scores,
                games,
                anchors,
                prior_draws=0.0,
                n_bootstrap=n_bootstrap,
                seed=seed,
                max_iter=max_iter,
                tol=tol,
            )
        else:
            raise ValueError(f"unknown error_method {error_method!r}")
        for k, i in enumerate(free):
            errors[names[i]] = float(errs[k])

    n_games = {names[i]: float(games[i].sum()) for i in range(n)}
    return EloFit(
        ratings={names[i]: float(ratings[i]) for i in range(n)},
        errors=errors,
        players=names,
        anchors=anchors,
        log_likelihood=float(ll),
        iterations=iterations,
        converged=converged,
        n_games=n_games,
    )


def _fisher_errors(
    ratings: np.ndarray, games: np.ndarray, free: np.ndarray
) -> np.ndarray:
    diff = (ratings[:, None] - ratings[None, :]) / SCALE
    p = 1.0 / (1.0 + np.exp(-diff))
    wt = games * p * (1.0 - p) / (SCALE * SCALE)
    info = -wt.copy()
    np.fill_diagonal(info, 0.0)
    np.fill_diagonal(info, np.sum(wt, axis=1))
    sub = info[np.ix_(free, free)] + 1e-12 * np.eye(free.size)
    try:
        cov = np.linalg.inv(sub)
    except np.linalg.LinAlgError:  # pragma: no cover - disconnected graph
        cov = np.linalg.pinv(sub)
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


def _bootstrap_errors(
    names: list[str],
    scores: np.ndarray,
    games: np.ndarray,
    anchors: Mapping[str, float],
    prior_draws: float,
    n_bootstrap: int,
    seed: int,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    """Parametric bootstrap: resample each pairing's games from its own rate."""
    rng = np.random.default_rng(seed)
    n = len(names)
    pairs = [
        (i, j, games[i, j], scores[i, j] / games[i, j])
        for i in range(n)
        for j in range(i + 1, n)
        if games[i, j] > 0.0
    ]
    free_names = [nm for nm in names if nm not in anchors]
    samples = np.zeros((n_bootstrap, len(free_names)))
    for b in range(n_bootstrap):
        results = []
        for i, j, total, rate in pairs:
            k = int(np.rint(total))
            wins = float(rng.binomial(k, min(max(rate, 0.0), 1.0)))
            results.append(PairResult(names[i], names[j], wins, 0.0, k - wins))
        fit = fit_elo(
            results,
            anchors=anchors,
            prior_draws=max(prior_draws, 0.5),
            max_iter=max_iter,
            tol=tol,
            error_method="none",
        )
        samples[b] = [fit.ratings[nm] for nm in free_names]
    return samples.std(axis=0, ddof=1)
