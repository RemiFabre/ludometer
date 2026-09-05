"""Search-improved targets on human positions: the teacher labels BGA games.

Two halves, one file.

**Local, once per corpus refresh** — ``export``: every raw BGA payload under
``data/human/raw`` is parsed and replayed (the same validation
:func:`ludometer.human.convert.convert_game` does; a game it rejects is
skipped), and what survives is written as a *compact* JSON: the deals, the
action ids, the first seat, the outcome. A few hundred kilobytes for thousands
of games, small enough to ship with the source bundle. Nothing under
``data/human`` is written to.

**In the job** — ``run``: the positions are replayed in the engine (deals are
scripted, so the replay is exact), every decision point of *both* players is
searched by the published weights at ``--sims`` with no root noise, and the
result is a shard per block whose rows look exactly like self-play rows: the
policy target is the visit distribution, the value is the game's outcome, the
search value is the root estimate, margins and final walls come from the game.
So a human game contributes ~55 positions the self-play distribution would
rarely visit, each labelled by the strongest search we can afford. The student
learns *what the teacher would play there*, not what the human played — the
human's role is to have reached the position.

``--part k/n`` splits the corpus across jobs; ``--tag`` keeps seeds apart.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ludometer.azul.engine import AzulState
from ludometer.train.mcts import MCTS, MCTSConfig
from ludometer.train.selfplay import GameRecord, margin_targets

__all__ = [
    "BatchLabeler",
    "PositionGame",
    "export_positions",
    "main",
    "replay_positions",
]


# --------------------------------------------------------------- compact games
@dataclass
class PositionGame:
    """One human game as the engine needs it to replay: deals + actions."""

    table_id: int
    first_seat: int
    deals: list[list[list[int]]]  # per round: NUM_FACTORIES x NUM_COLORS counts
    actions: list[int]
    outcome: float  # +1 seat 0 won, -1 seat 1 won, 0 draw
    scores: tuple[int, int]

    def to_json(self) -> dict[str, Any]:
        return {
            "t": self.table_id,
            "f": self.first_seat,
            "d": self.deals,
            "a": self.actions,
            "o": self.outcome,
            "s": list(self.scores),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> PositionGame:
        return cls(
            table_id=int(d["t"]),
            first_seat=int(d["f"]),
            deals=[[list(map(int, f)) for f in deal] for deal in d["d"]],
            actions=[int(a) for a in d["a"]],
            outcome=float(d["o"]),
            scores=(int(d["s"][0]), int(d["s"][1])),
        )


def _deal_obj(counts: list[list[int]], round_index: int) -> Any:
    from ludometer.human.parse import Deal

    return Deal(round_index=round_index, factories=tuple(tuple(f) for f in counts))


def replay_positions(
    game: PositionGame,
) -> tuple[list[AzulState], list[int], AzulState]:
    """``(states before each action, mover per action, final state)``.

    Mirrors the loop in :func:`ludometer.human.convert.convert_game` (kept
    separate so this package never edits the human pipeline's files).
    """
    from ludometer.human.convert import apply_deal

    state = AzulState.new_game(seed=0)
    apply_deal(state, _deal_obj(game.deals[0], 0))
    state.current_player = game.first_seat
    state.first_player = game.first_seat
    states: list[AzulState] = []
    movers: list[int] = []
    deal_index = 1
    for action in game.actions:
        if state.is_terminal:
            raise ValueError(f"table {game.table_id}: actions past the end")
        if not state.is_legal(action):
            raise ValueError(f"table {game.table_id}: illegal action {action}")
        states.append(state.clone())
        movers.append(state.current_player)
        round_before = state.round_index
        state.apply(action)
        if state.round_index > round_before and not state.is_terminal:
            if deal_index >= len(game.deals):
                raise ValueError(f"table {game.table_id}: out of deals")
            apply_deal(state, _deal_obj(game.deals[deal_index], deal_index))
            deal_index += 1
    return states, movers, state


def export_positions(raw_dir: Path, out: Path, limit: int = 0) -> int:
    """Parse + validate every raw BGA payload; write the compact JSON (gzip)."""
    from ludometer.human.client import read_json_gz
    from ludometer.human.convert import ConversionError, convert_game
    from ludometer.human.parse import ParseError, parse_log

    games: list[dict[str, Any]] = []
    rejected = 0
    paths = sorted(raw_dir.glob("*.json.gz"))
    if limit:
        paths = paths[:limit]
    for path in paths:
        table_id = int(path.name.split(".")[0])
        payload = read_json_gz(path)
        infos = payload.get("infos") or {}
        seats = ((infos.get("data") or {}).get("players") or {}).keys()
        try:
            replay = parse_log(
                payload.get("logs") or {},
                table_id,
                [int(s) for s in seats],
                infos=infos,
            )
            human = convert_game(replay)  # the validation
        except (ConversionError, ParseError):
            rejected += 1
            continue
        game = PositionGame(
            table_id=table_id,
            first_seat=int(human.movers[0]),
            deals=[[list(f) for f in d.factories] for d in replay.deals],
            actions=[int(a) for a in human.actions],
            outcome=float(human.outcome),
            scores=human.scores,
        )
        # Round-trip check: our replay must reach the same encoded positions.
        try:
            states, _movers, _final = replay_positions(game)
        except ValueError:
            rejected += 1
            continue
        if len(states) != len(human.states) or not np.allclose(
            np.stack([s.encode() for s in states]), human.states
        ):
            rejected += 1
            continue
        games.append(game.to_json())
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as fh:
        json.dump({"format": 1, "games": games}, fh)
    print(f"[label] exported {len(games)} games ({rejected} rejected) -> {out}")
    return len(games)


def load_positions(path: Path) -> list[PositionGame]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    return [PositionGame.from_json(d) for d in data["games"]]


# ----------------------------------------------------------------- the labeler
class BatchLabeler:
    """Search many independent positions concurrently, one forward pass each round.

    The same leaf-request pump :class:`~ludometer.train.selfplay_batched.BatchedSelfPlay`
    uses, minus the game: every slot is one position, searched with no root noise,
    and returns ``(policy, value, margin)`` when its budget is spent.
    """

    def __init__(
        self, evaluator: Any, config: MCTSConfig, slots: int, sims: int, seed: int
    ) -> None:
        self.evaluator = evaluator
        self.config = config
        self.slots = max(1, int(slots))
        self.sims = int(sims)
        self.seed = int(seed)
        self.evals = 0

    def label(self, states: list[AzulState]) -> list[tuple[np.ndarray, float, float]]:
        out: list[tuple[np.ndarray, float, float] | None] = [None] * len(states)
        active: list[tuple[int, MCTS]] = []
        next_index = 0
        while next_index < len(states) or active:
            while len(active) < self.slots and next_index < len(states):
                i = next_index
                next_index += 1
                mcts = MCTS(
                    self.evaluator,
                    self.config,
                    seed=(self.seed * 1_000_003 + i) & 0x7FFFFFFF,
                    add_noise=False,
                )
                mcts.start_search(states[i], add_noise=False, sims=self.sims)
                active.append((i, mcts))
            requests = []
            pending = []
            for i, mcts in active:
                got = mcts.leaf_requests(1)
                pending.append(len(got))
                requests.extend(got)
            if requests:
                results = self.evaluator.evaluate(
                    [r.node.state for r in requests], [r.node.legal for r in requests]
                )
                self.evals += len(requests)
                at = 0
                for (i, mcts), n in zip(active, pending):
                    if n:
                        mcts.apply_leaves(results[at : at + n])
                        at += n
            still: list[tuple[int, MCTS]] = []
            for i, mcts in active:
                if mcts.search_done():
                    res = mcts.finish_search()
                    out[i] = (res.policy, float(res.value), float(res.margin))
                else:
                    still.append((i, mcts))
            active = still
        return [o for o in out if o is not None]


def label_game(labeler: BatchLabeler, game: PositionGame) -> GameRecord:
    states, movers, final = replay_positions(game)
    encoded = np.stack([s.encode() for s in states]).astype(np.float32)
    labels = labeler.label(states)
    policies = np.stack([p for p, _v, _m in labels]).astype(np.float32)
    search_values = np.array([v for _p, v, _m in labels], dtype=np.float32)
    signs = np.array([1.0 if m == 0 else -1.0 for m in movers], dtype=np.float32)
    values = (game.outcome * signs).astype(np.float32)
    score_diff = game.scores[0] - game.scores[1]
    walls = [final.wall_summary(0), final.wall_summary(1)]
    aux = np.stack([np.array(walls[m] + walls[1 - m], dtype=np.uint8) for m in movers])
    return GameRecord(
        states=encoded,
        policies=policies,
        values=values,
        margins=margin_targets(score_diff, movers),
        aux=aux,
        policy_mask=np.ones(len(states), dtype=np.float32),
        outcome=game.outcome,
        scores=game.scores,
        moves=len(states),
        rounds=int(final.round_index) + 1,
        seed=game.table_id,
        decisions=len(states),
        evals=0,
        duration=0.0,
        truncated=not final.is_terminal,
        search_values=search_values,
        search_mask=np.ones(len(states), dtype=np.float32),
    )


# ---------------------------------------------------------------- job workers
def _worker(args: tuple[Any, ...]) -> list[GameRecord]:  # pragma: no cover - subprocess
    weights_path, net_config, mcts_config, sims, slots, seed, games_json = args
    import torch

    torch.set_num_threads(1)
    from ludometer.train.net import make_net, net_config_from_dict
    from ludometer.train.selfplay_batched import BatchEvaluator

    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    net = make_net(net_config_from_dict(net_config))
    net.load_numpy_state_dict({k: v.numpy() for k, v in payload["weights"].items()})
    evaluator = BatchEvaluator(net, device="cpu")
    labeler = BatchLabeler(evaluator, mcts_config, slots=slots, sims=sims, seed=seed)
    out = []
    for d in games_json:
        try:
            out.append(label_game(labeler, PositionGame.from_json(d)))
        except ValueError as exc:
            print(f"[label] skipped: {exc}", flush=True)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ludometer.cloud.label")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--raw", type=Path, default=Path("data/human/raw"))
    e.add_argument("--out", type=Path, default=Path("data/cloud/bga_positions.json.gz"))
    e.add_argument("--limit", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--positions", type=Path, required=True)
    r.add_argument("--run", required=True, help="weights run to label with")
    r.add_argument("--out-run", required=True, help="shard prefix, e.g. rlx_bga")
    r.add_argument("--shards", required=True)
    r.add_argument("--weights", required=True)
    r.add_argument("--tag", default="")
    r.add_argument("--part", default="0/1", help="k/n: this job's slice of the games")
    r.add_argument("--workers", type=int, default=0)
    r.add_argument(
        "--games",
        type=int,
        default=32,
        help="positions searched concurrently per worker",
    )
    r.add_argument("--sims", type=int, default=0)
    r.add_argument("--block", type=int, default=16, help="games per shard")
    return p


def _run(args: argparse.Namespace) -> int:
    import multiprocessing as mp

    from ludometer.cloud.hub import fetch_weights, hub_from_spec
    from ludometer.cloud.shards import write_shard
    from ludometer.train.trainer import TrainConfig

    token = os.environ.get("HF_TOKEN")
    shards = hub_from_spec(args.shards, token)
    weights_hub = hub_from_spec(args.weights, token)
    raw = weights_hub.get_bytes(f"{args.run}/config.json")
    if raw is None:
        raise SystemExit(f"no {args.run}/config.json")
    cfg = TrainConfig.from_dict(
        {k: v for k, v in json.loads(raw).items() if k != "started"}
    )
    sims = args.sims or cfg.sims
    mcts_config = cfg.selfplay_config().mcts
    tag = args.tag or f"label-{os.getpid()}"
    k, n = (int(x) for x in args.part.split("/"))
    games = load_positions(args.positions)[k::n]
    workers = args.workers or max(1, os.cpu_count() or 1)
    print(
        f"[rl-experiment] label {len(games)} games (part {k}/{n}) with {args.run} at {sims} sims, "
        f"{workers} workers x {args.games} slots",
        flush=True,
    )
    with tempfile.TemporaryDirectory() as td:
        got = fetch_weights(weights_hub, args.run, 0, td)
        if got is None:
            raise SystemExit("no weights published")
        version, net_config, _w = got
        weights_path = str(Path(td) / f"weights-v{version:05d}.pt")
        ctx = mp.get_context("spawn")
        block = max(1, args.block)
        chunks = [games[i : i + block] for i in range(0, len(games), block)]
        t0 = time.monotonic()
        done = 0
        with ctx.Pool(workers) as pool:
            jobs = (
                (
                    weights_path,
                    net_config,
                    mcts_config,
                    sims,
                    args.games,
                    7919 * (i + 1),
                    [g.to_json() for g in chunk],
                )
                for i, chunk in enumerate(chunks)
            )
            for i, records in enumerate(pool.imap(_worker, jobs)):
                if not records:
                    continue
                positions = sum(len(r) for r in records)
                meta = {
                    "run": args.out_run,
                    "source": "bga",
                    "labelled_by": args.run,
                    "tag": tag,
                    "block": i,
                    "weights_version": version,
                    "sims": sims,
                    "games": len(records),
                    "positions": positions,
                    "evals": 0,
                    "seconds": round(time.monotonic() - t0, 1),
                    "created": time.time(),
                }
                local = Path(td) / "shard.npz"
                write_shard(local, records, meta)
                shards.put(local, f"{args.out_run}/v{version:05d}-{tag}-{i:05d}.npz")
                done += len(records)
                print(
                    f"[rl-experiment] block {i}: {len(records)} games, {positions} positions; "
                    f"{done}/{len(games)} games in {(time.monotonic() - t0) / 60:.1f} min",
                    flush=True,
                )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "export":
        export_positions(args.raw, args.out, args.limit)
        return 0
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
