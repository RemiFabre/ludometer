"""Position suites for %-optimal evaluation (NEXT_GAMES.md §3).

A suite is a one-off artifact: sample reachable positions from real games
(stratified by ply so the opening is not over-represented), solve every legal
move exactly, cache to ``data/solved/<game>_suite.json``. Evaluating an agent
is then a dict lookup per position:

* **% optimal** — the chosen move preserves the game-theoretic value;
* **blunder rate** — the chosen move turns a won position into a non-won one.

Build from the CLI (once)::

    uv run python -m ludometer.solved.suite --game tictactoe --n 2000
    uv run python -m ludometer.solved.suite --game connect4 --n 2000 --min-ply 8

The sampling games are played by the game's own baseline mix (random, greedy,
heuristic in all pairings) — the trained runs this suite will judge do not
exist yet when it is built, which is also why it must be built only once.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ludometer.agents import make_agent
from ludometer.c4.engine import Connect4State
from ludometer.games import get_game
from ludometer.solved.solver import c4_solve, ttt_solve
from ludometer.ttt.engine import TicTacToeState

__all__ = [
    "build_suite",
    "load_suite",
    "rebuild_state",
    "score_agent",
    "solve_state",
    "suite_path",
]

_MIN_PLY = {"tictactoe": 0, "connect4": 8}


def solve_state(state: Any) -> int:
    """Exact WDL for the player to move (dispatch on the engine type)."""
    if isinstance(state, TicTacToeState):
        return ttt_solve(state.me, state.them)
    if isinstance(state, Connect4State):
        return c4_solve(state.position, state.mask)
    raise TypeError(f"no solver for {type(state).__name__}")


def _key(state: Any) -> list[int]:
    if isinstance(state, TicTacToeState):
        return [state.me, state.them, state.ply]
    return [state.position, state.mask, state.ply]


def rebuild_state(game: str, key: list[int]) -> Any:
    state = get_game(game).new_game(0)
    if isinstance(state, TicTacToeState):
        state.me, state.them, state.ply = key
    else:
        state.position, state.mask, state.ply = key
    return state


def _mover_value_after(state: Any, action: int) -> int:
    child = state.clone()
    child.apply(action)
    if child.is_terminal:
        outcome = child.outcome()
        return int(outcome if state.current_player == 0 else -outcome)
    return -solve_state(child)


def _sample_positions(game: str, want: int, seed: int, min_ply: int) -> list[Any]:
    """Reachable, non-terminal, deduplicated positions from baseline games."""
    spec = get_game(game)
    names = list(spec.baselines)
    pairs = [(a, b) for a in names for b in names]
    by_key: dict[tuple[int, ...], Any] = {}
    game_seed = seed
    # oversample, then stratify: ~8x the ask, capped for safety
    while len(by_key) < want * 8 and game_seed < seed + want * 40:
        a, b = pairs[game_seed % len(pairs)]
        agents = (make_agent(a), make_agent(b))
        for i, agent in enumerate(agents):
            agent.seed(game_seed * 2 + i)
        state = spec.new_game(game_seed)
        game_seed += 1
        while not state.is_terminal:
            if state.ply >= min_ply:
                by_key.setdefault(tuple(_key(state)), state.clone())
            state.apply(agents[state.current_player].act(state))
    return list(by_key.values())


def build_suite(
    game: str,
    n_positions: int,
    seed: int = 0,
    min_ply: int | None = None,
    log: Any = None,
) -> dict[str, Any]:
    if min_ply is None:
        min_ply = _MIN_PLY[game]
    t0 = time.monotonic()
    pool = _sample_positions(game, n_positions, seed, min_ply)
    # stratify: round-robin over ply buckets so no depth dominates
    buckets: dict[int, list[Any]] = {}
    for state in pool:
        buckets.setdefault(state.ply, []).append(state)
    picked: list[Any] = []
    while len(picked) < n_positions and any(buckets.values()):
        for ply in sorted(buckets):
            if buckets[ply] and len(picked) < n_positions:
                picked.append(buckets[ply].pop(0))
    # Solve deepest-first: deep positions are cheap and their sub-results fill
    # the shared transposition table before the expensive shallow solves need it.
    picked.sort(key=lambda s: -s.ply)
    entries = []
    for i, state in enumerate(picked):
        values = {str(a): _mover_value_after(state, a) for a in state.legal_actions()}
        best = max(values.values())
        entries.append(
            {
                "key": _key(state),
                "ply": state.ply,
                "value": best,
                "values": values,
                "optimal": sorted(int(a) for a, v in values.items() if v == best),
            }
        )
        if log and (i + 1) % 25 == 0:
            log(f"solved {i + 1}/{len(picked)} ({time.monotonic() - t0:.0f}s)")
    return {
        "game": game,
        "seed": seed,
        "min_ply": min_ply,
        "positions": entries,
        "built_seconds": round(time.monotonic() - t0, 1),
    }


def suite_path(game: str) -> Path:
    return Path("data/solved") / f"{game}_suite.json"


def load_suite(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def score_agent(agent: Any, suite: dict[str, Any]) -> dict[str, Any]:
    """% optimal and blunder rate of ``agent`` over ``suite`` (one pass)."""
    game = suite["game"]
    n = optimal = blunders = wins = 0
    for entry in suite["positions"]:
        state = rebuild_state(game, entry["key"])
        action = agent.act(state)
        value = entry["values"].get(str(action))
        if value is None:  # an illegal choice is maximally wrong
            value = -2
        n += 1
        if value == entry["value"]:
            optimal += 1
        if entry["value"] == 1:
            wins += 1
            if value < 1:
                blunders += 1
    return {
        "n": n,
        "pct_optimal": optimal / n if n else 0.0,
        "blunder_rate": blunders / wins if wins else 0.0,
        "won_positions": wins,
    }


def main() -> None:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True, choices=("tictactoe", "connect4"))
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--min-ply", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    suite = build_suite(args.game, args.n, args.seed, args.min_ply, log=print)
    out = Path(args.out) if args.out else suite_path(args.game)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(suite), encoding="utf-8")
    print(f"wrote {len(suite['positions'])} solved positions to {out} "
          f"in {suite['built_seconds']}s")


if __name__ == "__main__":  # pragma: no cover
    main()
