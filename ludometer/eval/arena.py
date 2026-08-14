"""Arena: play deterministic games and matches between two agents.

Everything is a pure function of the seeds: ``play_game(a, b, seed)`` always
produces the same game, and ``play_match`` derives its per-game seeds from
``base_seed`` so a match is reproducible with or without multiprocessing.

Matches alternate who moves first. Game ``2k`` and game ``2k+1`` share the deal
seed ``base_seed + k`` and swap seats, so both agents see the same bag order in
each pair (a cheap variance reduction).

Agents may be passed as instances or, for multiprocessing, as *specs* — see
:func:`ludometer.agents.make_agent`. Specs are what crosses the process
boundary, so each worker builds its own agent objects.

The pool uses the ``spawn`` start method (safe next to torch/MPS), so a script
calling ``play_match(..., n_workers>1)`` must guard its entry point with
``if __name__ == "__main__":``.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any

from ludometer.agents import AgentSpec, make_agent, spec_name
from ludometer.azul.engine import AzulState

__all__ = [
    "GameResult",
    "MatchResult",
    "play_game",
    "play_match",
    "round_robin",
]

MAX_MOVES = 2000  # safety net; real games take well under 200 moves


@dataclass(frozen=True)
class GameResult:
    """One finished game, always reported from agent A's point of view."""

    seed: int
    a_first: bool
    a_score: int
    b_score: int
    result: float  # 1.0 = A won, 0.5 = draw, 0.0 = B won
    moves: int
    rounds: int

    @property
    def score_diff(self) -> int:
        return self.a_score - self.b_score


@dataclass
class MatchResult:
    """Aggregate of ``n_games`` games, from agent A's point of view."""

    name_a: str
    name_b: str
    n_games: int
    wins: int
    draws: int
    losses: int
    mean_score_diff: float
    mean_score_a: float
    mean_score_b: float
    mean_moves: float
    base_seed: int
    games: list[GameResult] = field(default_factory=list, repr=False)

    @property
    def win_rate(self) -> float:
        """Score share: wins + half the draws, over the number of games."""
        if self.n_games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.n_games

    @property
    def decisive_win_rate(self) -> float:
        """Fraction of *games won outright* (draws count as losses)."""
        return self.wins / self.n_games if self.n_games else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "a": self.name_a,
            "b": self.name_b,
            "n_games": self.n_games,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "mean_score_diff": self.mean_score_diff,
            "mean_score_a": self.mean_score_a,
            "mean_score_b": self.mean_score_b,
            "mean_moves": self.mean_moves,
            "base_seed": self.base_seed,
        }


def _agent_seed(seed: int, slot: int) -> int:
    """Per-game agent seed; distinct per seat and decorrelated from the deal."""
    return (seed * 2_654_435_761 + slot * 40_503 + 12_345) % (1 << 31)


def play_game(
    agent_a: AgentSpec,
    agent_b: AgentSpec,
    seed: int,
    a_first: bool = True,
) -> GameResult:
    """Play one game. ``a_first`` puts agent A in seat 0 (moves first).

    Agents are reseeded from ``seed`` before the game, so the result only depends
    on ``(agent specs, seed, a_first)``.
    """
    a = make_agent(agent_a)
    b = make_agent(agent_b)
    a.seed(_agent_seed(seed, 0 if a_first else 1))
    b.seed(_agent_seed(seed, 1 if a_first else 0))

    state = AzulState.new_game(seed=seed)
    players = (a, b) if a_first else (b, a)
    moves = 0
    while not state.is_terminal:
        if moves >= MAX_MOVES:  # pragma: no cover - defensive
            raise RuntimeError(f"game did not terminate in {MAX_MOVES} moves")
        agent = players[state.current_player]
        action = agent.act(state)
        if not state.is_legal(action):
            raise ValueError(
                f"{agent.name} returned illegal action {action} "
                f"(seed={seed}, move={moves})"
            )
        state.apply(action)
        moves += 1

    seat_a = 0 if a_first else 1
    a_score = state.scores[seat_a]
    b_score = state.scores[1 - seat_a]
    outcome = state.outcome() or 0.0  # +1 => player 0 won
    if outcome == 0.0:
        result = 0.5
    else:
        a_won = (outcome > 0.0) == (seat_a == 0)
        result = 1.0 if a_won else 0.0
    return GameResult(
        seed=seed,
        a_first=a_first,
        a_score=a_score,
        b_score=b_score,
        result=result,
        moves=moves,
        rounds=state.round_index + 1,
    )


def _play_game_task(task: tuple[AgentSpec, AgentSpec, int, bool]) -> GameResult:
    """Top-level worker so :func:`play_match` can use a process pool."""
    agent_a, agent_b, seed, a_first = task
    return play_game(agent_a, agent_b, seed, a_first)


def play_match(
    agent_a: AgentSpec,
    agent_b: AgentSpec,
    n_games: int = 100,
    base_seed: int = 0,
    n_workers: int = 1,
    keep_games: bool = False,
) -> MatchResult:
    """Play ``n_games`` alternating-seat games and aggregate them.

    ``n_workers > 1`` spreads the games over a process pool; the aggregate is
    identical to the single-process one (the seeds, not the scheduling, decide
    the games). Agent specs must be picklable in that case.
    """
    if n_games < 0:
        raise ValueError("n_games must be >= 0")
    tasks = [(agent_a, agent_b, base_seed + i // 2, i % 2 == 0) for i in range(n_games)]
    if n_workers > 1 and n_games > 1:
        chunksize = max(1, n_games // (n_workers * 4))
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = list(pool.imap(_play_game_task, tasks, chunksize=chunksize))
    else:
        results = [_play_game_task(t) for t in tasks]

    wins = sum(1 for r in results if r.result == 1.0)
    draws = sum(1 for r in results if r.result == 0.5)
    losses = len(results) - wins - draws
    n = len(results) or 1
    return MatchResult(
        name_a=spec_name(agent_a),
        name_b=spec_name(agent_b),
        n_games=len(results),
        wins=wins,
        draws=draws,
        losses=losses,
        mean_score_diff=sum(r.score_diff for r in results) / n,
        mean_score_a=sum(r.a_score for r in results) / n,
        mean_score_b=sum(r.b_score for r in results) / n,
        mean_moves=sum(r.moves for r in results) / n,
        base_seed=base_seed,
        games=results if keep_games else [],
    )


def round_robin(
    specs: list[AgentSpec],
    n_games: int = 100,
    base_seed: int = 0,
    n_workers: int = 1,
) -> list[MatchResult]:
    """Every unordered pair plays ``n_games``; returns one MatchResult per pair.

    Each pairing gets its own seed block so pairings are independent.
    """
    out: list[MatchResult] = []
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            seed = base_seed + 1_000_000 * (i * len(specs) + j)
            out.append(
                play_match(
                    specs[i],
                    specs[j],
                    n_games=n_games,
                    base_seed=seed,
                    n_workers=n_workers,
                )
            )
    return out
