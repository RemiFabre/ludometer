"""Convert Faïence site games (faience-game/1 jsonl) into training rows.

    uv run python -m ludometer.human.faience \
        --games data/faience/games --npz data/faience/human_windraw.npz \
        --results win,draw

The Faïence player (web/player) runs a JS port of our engine, so a record is
deterministic: ``seed`` reproduces every shuffle, ``moves[].action`` are already
our 180-space action ids, and the ingest Space only accepts games that replay
(web/ingest/verify.js). This module re-verifies in the *Python* engine anyway —
``AzulState.new_game(seed)`` must deal what the record's ``deals`` say (any
mismatch falls back to :func:`ludometer.human.convert.apply_deal`, i.e. the deal
is scripted rather than drawn), every move must be legal, and the final scores
must equal ``final.scores`` exactly, or the game is rejected.

Rows follow the same conventions as :mod:`ludometer.human.dataset`: the human is
the *target* player (their moves are one-hot policy targets), the agent's rows
are kept value-only (``policy_mask = 0``), and value/margin are in the mover's
frame. ``--results win,draw`` keeps only games where the human beat or drew the
deployed net — with anonymous visitors, the result against a known-strength
agent is the only strength signal there is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ludometer.azul.engine import AzulState, NUM_COLORS, TILES_PER_COLOR
from ludometer.human.convert import ConversionError, HumanGame, apply_deal
from ludometer.human.dataset import OPPONENT_VALUE_ONLY, add_game
from ludometer.human.parse import Deal
from ludometer.train.replay import ReplayBuffer

__all__ = ["convert_record", "iter_records", "main"]


def iter_records(games_dir: Path):
    for path in sorted(games_dir.rglob("*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)


def _factories_match(state: AzulState, factories) -> bool:
    return [list(f) for f in state.factories] == [list(f) for f in factories]


def convert_record(record: dict) -> HumanGame:
    """Replay one faience-game/1 record; raises ConversionError if it won't."""
    if record.get("format") != "faience-game/1":
        raise ConversionError(f"unknown format {record.get('format')!r}")
    final = record.get("final") or {}
    if not final.get("finished"):
        raise ConversionError("game not finished")
    deals = record.get("deals") or []
    if not deals:
        raise ConversionError("no deals recorded")

    state = AzulState.new_game(seed=int(record["seed"]))
    if not _factories_match(state, deals[0]["factories"]):
        apply_deal(state, Deal(0, tuple(tuple(f) for f in deals[0]["factories"])))
    deal_index = 1

    states: list[np.ndarray] = []
    actions: list[int] = []
    movers: list[int] = []
    for move in record["moves"]:
        if state.is_terminal:
            raise ConversionError(f"moves continue past game end (ply {move['ply']})")
        seat = int(move["player"])
        if seat != state.current_player:
            raise ConversionError(
                f"ply {move['ply']}: record says seat {seat}, engine says "
                f"{state.current_player}"
            )
        action = int(move["action"])
        if not state.is_legal(action):
            raise ConversionError(f"ply {move['ply']}: illegal action {action}")
        states.append(state.encode())
        actions.append(action)
        movers.append(seat)
        round_before = state.round_index
        state.apply(action)
        if state.round_index > round_before and not state.is_terminal:
            if deal_index < len(deals) and not _factories_match(
                state, deals[deal_index]["factories"]
            ):
                apply_deal(
                    state,
                    Deal(
                        deal_index,
                        tuple(tuple(f) for f in deals[deal_index]["factories"]),
                    ),
                )
            deal_index += 1

    if not state.is_terminal:
        raise ConversionError(f"record ran out after {len(actions)} moves")
    census = state.tile_census()
    if census != [TILES_PER_COLOR] * NUM_COLORS:
        raise ConversionError(f"final tile census {census}")
    engine_scores = (int(state.scores[0]), int(state.scores[1]))
    reported = final.get("scores")
    if reported is not None and tuple(int(s) for s in reported) != engine_scores:
        raise ConversionError(
            f"engine scored {engine_scores}, record says {tuple(reported)}"
        )

    walls = [state.wall_summary(0), state.wall_summary(1)]
    aux_by_seat = [np.array(walls[p] + walls[1 - p], dtype=np.uint8) for p in (0, 1)]
    return HumanGame(
        table_id=int(record["seed"]),  # no table id; the seed is the stable handle
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        movers=np.asarray(movers, dtype=np.int64),
        aux=np.stack([aux_by_seat[m] for m in movers]),
        scores=engine_scores,
        outcome=float(state.outcome() or 0.0),
        rounds=int(state.round_index) + 1,
        player_ids=(0, 1),
    )


def human_result(record: dict, game: HumanGame) -> str:
    seat = int(record["human_seat"])
    h, a = game.scores[seat], game.scores[1 - seat]
    return "win" if h > a else "loss" if h < a else "draw"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ludometer.human.faience")
    parser.add_argument("--games", type=Path, default=Path("data/faience/games"))
    parser.add_argument("--npz", type=Path, default=None, help="write rows here")
    parser.add_argument(
        "--results",
        default="win,draw",
        help="comma list of human results to keep for the npz (win/draw/loss)",
    )
    args = parser.parse_args(argv)

    keep = {r.strip() for r in args.results.split(",") if r.strip()}
    tallies = {"win": 0, "draw": 0, "loss": 0}
    rejects: list[str] = []
    kept_games: list[tuple[dict, HumanGame]] = []
    scripted = replayed = 0
    for record in iter_records(args.games):
        try:
            game = convert_record(record)
        except (ConversionError, KeyError, TypeError, ValueError) as exc:
            rejects.append(str(exc))
            continue
        replayed += 1
        result = human_result(record, game)
        tallies[result] += 1
        if result in keep:
            kept_games.append((record, game))
    print(
        f"replayed {replayed} games exactly ({len(rejects)} rejected); "
        f"human record vs the site net: {tallies}"
    )
    for reason in rejects[:5]:
        print(f"  reject: {reason}")

    if args.npz:
        buffer = ReplayBuffer(capacity=max(1, sum(len(g) for _, g in kept_games)))
        rows = policy_rows = 0
        for record, game in kept_games:
            w, p = add_game(
                buffer,
                game,
                target_seat=int(record["human_seat"]),
                opponent_rows=OPPONENT_VALUE_ONLY,
            )
            rows += w
            policy_rows += p
        args.npz.parent.mkdir(parents=True, exist_ok=True)
        buffer.save(args.npz)
        print(
            f"wrote {args.npz}: {rows} rows from {len(kept_games)} "
            f"{sorted(keep)} games, {policy_rows} human policy targets"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
