#!/usr/bin/env python
"""Record seeded Python games so the JavaScript engine can be proved against them.

``web/player/js/engine.js`` is a hand port of ``ludometer/azul/engine.py``. A port
is only worth anything if it is checked, so this script plays N seeded random
games with the *Python* engine and writes, for every single move:

* ``state``   — the full ``to_json()`` snapshot before the move;
* ``legal``   — ``legal_actions()`` **in engine order**, not as a set;
* ``action``  — the action that was played;
* ``enc``     — ``encode()`` rounded to 6 decimals (integers stay integers);
* ``scores``  — the running score pair;

plus, per game, the terminal scores/outcome and the RNG's shuffle log.

The shuffle log is the trick that makes the comparison possible at all: JS cannot
reproduce CPython's Mersenne Twister, so instead of the seed we record the *bag
ordering* every ``rng.shuffle`` produced. ``ScriptedRng`` in engine.js replays
them, both engines then deal identical tiles, and every difference that remains
is a genuine rules difference.

Output is gzipped JSON (``web/player/test/fixtures/games.json.gz``) — a few MB of
plain text, a few hundred KB on disk, and ``node:zlib`` reads it with no
dependencies.

Run it niced; it is single-threaded and takes a few seconds::

    nice -n 15 uv run python scripts/dump_fixtures.py --games 30
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import ludometer.azul.engine as _engine
from ludometer.azul.engine import AzulState

DEFAULT_OUT = _REPO_ROOT / "web" / "player" / "test" / "fixtures" / "games.json.gz"
MAX_MOVES = 400


class _RecordingRandom:
    """Wraps a ``random.Random`` and logs the result of every ``shuffle``."""

    def __init__(self, inner: random.Random, log: list[list[int]]) -> None:
        self._inner = inner
        self._log = log

    def shuffle(self, x: list[int]) -> None:
        self._inner.shuffle(x)
        self._log.append(list(x))

    def __getattr__(self, name: str) -> Any:  # seed/getstate/... pass straight through
        return getattr(self._inner, name)


def _new_game_recorded(seed: int) -> tuple[AzulState, list[list[int]]]:
    """``AzulState.new_game`` with every ``shuffle`` logged, deal included.

    ``new_game`` builds its own ``random.Random`` *and* consumes the first deal
    before returning, so wrapping ``state.rng`` afterwards would miss the opening
    shuffle. Swapping the engine module's ``random`` for a shim that hands back a
    recording RNG catches it — the shim is restored immediately after.
    """
    log: list[list[int]] = []
    real = _engine.random

    class _Shim:
        @staticmethod
        def Random(s: int) -> _RecordingRandom:
            return _RecordingRandom(real.Random(s), log)

    _engine.random = _Shim  # type: ignore[assignment]
    try:
        state = AzulState.new_game(seed=seed)
    finally:
        _engine.random = real
    return state, log


def _compact(value: float) -> float | int:
    """6-decimal round, but keep whole numbers as ints (most of the vector is 0)."""
    r = round(float(value), 6)
    return int(r) if r == int(r) else r


def play_game(seed: int, move_seed: int) -> dict[str, Any]:
    """One seeded uniform-random game, fully recorded."""
    state, shuffles = _new_game_recorded(seed)
    picker = random.Random(move_seed)
    moves: list[dict[str, Any]] = []
    while not state.is_terminal and len(moves) < MAX_MOVES:
        legal = state.legal_actions()
        if not legal:  # pragma: no cover - defensive
            break
        action = legal[picker.randrange(len(legal))]
        moves.append(
            {
                "state": state.to_json(),
                "legal": legal,
                "action": action,
                "enc": [_compact(x) for x in state.encode()],
                "scores": list(state.scores),
            }
        )
        state.apply(action)

    return {
        "seed": seed,
        "move_seed": move_seed,
        "shuffles": shuffles,
        "moves": moves,
        "final_state": state.to_json(),
        "final_enc": [_compact(x) for x in state.encode()],
        "scores": list(state.scores),
        "outcome": state.outcome(),
        "is_terminal": state.is_terminal,
        "exhausted": state.exhausted,
        "rounds": state.round_index + 1,
        "census": state.tile_census(),
    }


# ------------------------------------------------------------- handcrafted cases
#
# Random play never visits some of the engine's branches: the "nobody took the
# first-player marker" fallback needs all five factories to be monochrome (about
# 1e-11 per round), and a bag that runs dry mid-refill needs the lid to be nearly
# empty too. A mutation of either survives 30 random games untouched, so these
# positions are built by hand instead — the same door ``recount()`` exists for.


def _blank_setup() -> dict[str, Any]:
    return {
        "bag": [],
        "lid": [0] * 5,
        "factories": [[0] * 5 for _ in range(5)],
        "center": [0] * 5,
        "marker_in_center": True,
        "walls": [[0] * 25 for _ in range(2)],
        "pl_color": [[-1] * 5 for _ in range(2)],
        "pl_count": [[0] * 5 for _ in range(2)],
        "floor": [[0] * 5 for _ in range(2)],
        "floor_marker": [False, False],
        "scores": [0, 0],
        "current_player": 0,
        "first_player": 0,
        "round_index": 0,
    }


def _state_from_setup(setup: dict[str, Any]) -> tuple[AzulState, list[list[int]]]:
    """Apply a field dump on top of a fresh game, then ``recount()``."""
    state, log = _new_game_recorded(0)
    log.clear()  # the deal that follows is scripted, not shuffled
    state.bag = list(setup["bag"])
    state.lid = list(setup["lid"])
    state.factories = [list(f) for f in setup["factories"]]
    state.center = list(setup["center"])
    state.marker_in_center = setup["marker_in_center"]
    state.walls = [list(w) for w in setup["walls"]]
    state.pl_color = [list(x) for x in setup["pl_color"]]
    state.pl_count = [list(x) for x in setup["pl_count"]]
    state.floor = [list(f) for f in setup["floor"]]
    state.floor_marker = list(setup["floor_marker"])
    state.scores = list(setup["scores"])
    state.current_player = setup["current_player"]
    state.first_player = setup["first_player"]
    state.round_index = setup["round_index"]
    state.is_terminal = False
    state.exhausted = False
    state.recount()
    return state, log


def _wall_index(color: int, row: int) -> int:
    return row * 5 + (color + row) % 5


def handcrafted_cases() -> list[tuple[str, dict[str, Any]]]:
    """Named field dumps for the branches random play does not reach."""
    cases: list[tuple[str, dict[str, Any]]] = []

    # 1. every factory monochrome and the center empty: the round can end with
    #    the marker still in the middle, so `holder is None` and the fallback
    #    "whoever did not move last starts" decides.
    s = _blank_setup()
    for i in range(5):
        s["factories"][i][i] = 4
    s["bag"] = [c for c in range(5) for _ in range(16)]
    cases.append(("monochrome-round-end", s))

    # 2. a bag and lid that cannot fill the next round: `_refill` gives up and
    #    the game ends as `exhausted`.
    s = _blank_setup()
    for i in range(5):
        s["factories"][i][i % 5] = 4
    s["bag"] = [0, 1, 2]  # three tiles for twenty slots, and an empty lid
    s["round_index"] = 4
    cases.append(("bag-runs-dry", s))

    # 3. a floor already full plus a big take: the spill has to reach the lid,
    #    and the round's penalty must clamp the score at zero rather than go
    #    negative.
    s = _blank_setup()
    s["factories"][0][2] = 4
    s["factories"][1][3] = 4
    s["center"] = [0, 0, 6, 0, 0]
    s["marker_in_center"] = True
    s["floor"] = [[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
    s["scores"] = [3, 40]
    s["bag"] = [c for c in range(5) for _ in range(12)]
    cases.append(("floor-overflow-and-score-clamp", s))

    # 4. a wall one tile short of a row, a column and a colour: the end-of-game
    #    bonuses (2 / 7 / 10) all fire in the same scoring pass.
    s = _blank_setup()
    wall = s["walls"][0]
    for color in range(5):
        for row in range(5):
            wall[_wall_index(color, row)] = 1
    wall[_wall_index(0, 0)] = 0  # leave one square open, row 0
    s["pl_color"][0][0] = 0
    s["pl_count"][0][0] = 0
    s["factories"][0][0] = 4
    for i in range(1, 5):
        s["factories"][i][i] = 4
    s["scores"] = [55, 20]
    s["bag"] = [c for c in range(5) for _ in range(4)]
    cases.append(("wall-completion-bonuses", s))

    # 5. both players holding wide-open boards with a center that already has the
    #    marker taken — exercises the `holder` path where the marker is on a
    #    floor line at round end.
    s = _blank_setup()
    s["factories"][0] = [2, 2, 0, 0, 0]
    s["factories"][1] = [0, 0, 2, 2, 0]
    s["factories"][2] = [1, 1, 1, 1, 0]
    s["factories"][3] = [0, 0, 0, 2, 2]
    s["factories"][4] = [4, 0, 0, 0, 0]
    s["center"] = [1, 0, 1, 0, 2]
    s["marker_in_center"] = False
    s["floor_marker"] = [False, True]
    s["floor"] = [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
    s["pl_count"] = [[1, 1, 0, 0, 0], [0, 0, 2, 0, 0]]
    s["pl_color"] = [[0, 1, -1, -1, -1], [-1, -1, 2, -1, -1]]
    s["current_player"] = 1
    s["first_player"] = 1
    s["round_index"] = 2
    s["scores"] = [17, 21]
    s["bag"] = [c for c in range(5) for _ in range(10)]
    s["lid"] = [2, 2, 2, 2, 2]
    cases.append(("marker-on-a-floor-line", s))

    return cases


def play_case(name: str, setup: dict[str, Any], move_seed: int) -> dict[str, Any]:
    """Play a handcrafted position out to the end, recorded like a normal game."""
    state, shuffles = _state_from_setup(setup)
    picker = random.Random(move_seed)
    moves: list[dict[str, Any]] = []
    while not state.is_terminal and len(moves) < MAX_MOVES:
        legal = state.legal_actions()
        if not legal:  # pragma: no cover - defensive
            break
        action = legal[picker.randrange(len(legal))]
        moves.append(
            {
                "state": state.to_json(),
                "legal": legal,
                "action": action,
                "enc": [_compact(x) for x in state.encode()],
                "scores": list(state.scores),
            }
        )
        state.apply(action)
    return {
        "name": name,
        "setup": setup,
        "move_seed": move_seed,
        "shuffles": shuffles,
        "moves": moves,
        "final_state": state.to_json(),
        "final_enc": [_compact(x) for x in state.encode()],
        "scores": list(state.scores),
        "outcome": state.outcome(),
        "is_terminal": state.is_terminal,
        "exhausted": state.exhausted,
        "rounds": state.round_index + 1,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260815, help="master seed")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    master = random.Random(args.seed)
    games = []
    for i in range(args.games):
        game = play_game(master.randrange(1 << 30), master.randrange(1 << 30))
        assert game["census"] == [20] * 5, f"game {i} lost tiles: {game['census']}"
        games.append(game)

    cases = [
        play_case(name, setup, master.randrange(1 << 30))
        for name, setup in handcrafted_cases()
    ]
    moves = sum(len(g["moves"]) for g in games)
    case_moves = sum(len(c["moves"]) for c in cases)
    payload = {
        "format": 1,
        "generator": "scripts/dump_fixtures.py",
        "master_seed": args.seed,
        "games": games,
        "cases": cases,
        "totals": {
            "games": len(games),
            "moves": moves,
            "cases": len(cases),
            "case_moves": case_moves,
            "terminal": sum(1 for g in games if g["is_terminal"]),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, separators=(",", ":")).encode()
    with gzip.open(out, "wb", compresslevel=9) as fh:
        fh.write(blob)
    print(
        f"{len(games)} games ({moves} moves) + {len(cases)} handcrafted cases "
        f"({case_moves} moves) -> {out} "
        f"({len(blob) / 1e6:.1f} MB json, {out.stat().st_size / 1e6:.2f} MB gzipped)"
    )
    for case in cases:
        print(
            f"  case {case['name']}: {len(case['moves'])} moves, "
            f"scores {case['scores']}, outcome {case['outcome']}, "
            f"exhausted {case['exhausted']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
