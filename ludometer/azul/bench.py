"""Throughput benchmark: full random Azul games per second, single core.

    uv run python -m ludometer.azul.bench [--seconds 3] [--seed 0] [--encode]

Target (docs/DESIGN.md): >= 2000 games/sec on one core.
"""

from __future__ import annotations

import argparse
import random
import time

from ludometer.azul.engine import AzulState


def run(
    seconds: float = 3.0, seed: int = 0, with_encode: bool = False
) -> dict[str, float]:
    rng = random.Random(seed)
    randrange = rng.randrange
    choice = rng.choice
    games = 0
    moves = 0
    t0 = time.perf_counter()
    deadline = t0 + seconds
    while True:
        state = AzulState.new_game(seed=randrange(1 << 30))
        apply_ = state.apply
        legal = state.legal_actions
        while not state.is_terminal:
            apply_(choice(legal()))
            moves += 1
            if with_encode:
                state.encode()
        games += 1
        if time.perf_counter() >= deadline:
            break
    elapsed = time.perf_counter() - t0
    return {
        "elapsed": elapsed,
        "games": games,
        "moves": moves,
        "games_per_sec": games / elapsed,
        "moves_per_sec": moves / elapsed,
        "moves_per_game": moves / games,
        "us_per_move": 1e6 * elapsed / moves,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--encode",
        action="store_true",
        help="also call encode() after every move (measures observation cost)",
    )
    args = ap.parse_args()

    # short warm-up so the first-call overheads do not skew a short run
    run(seconds=0.2, seed=args.seed + 1, with_encode=args.encode)
    r = run(seconds=args.seconds, seed=args.seed, with_encode=args.encode)

    print(f"random self-play{' + encode()' if args.encode else ''}")
    print(f"  elapsed        {r['elapsed']:.2f} s")
    print(f"  games          {r['games']:.0f}")
    print(f"  moves/game     {r['moves_per_game']:.1f}")
    print(f"  us/move        {r['us_per_move']:.2f}")
    print(f"  moves/sec      {r['moves_per_sec']:,.0f}")
    print(f"  GAMES/SEC      {r['games_per_sec']:,.0f}")
    target = 2000
    verdict = "OK" if r["games_per_sec"] >= target else "BELOW TARGET"
    print(f"  target {target} games/sec: {verdict}")


if __name__ == "__main__":
    main()
