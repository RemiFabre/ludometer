"""The opening atlas: a net's search on every expert opening decision.

    uv run python -m ludometer.opening.atlas label \
        --net porcelain=runs/porc_w-p0905-2038/checkpoints/ckpt-000000.pt \
        --rounds 0-1 --sims 4096 --out data/cloud/opening/porcelain_r01_s4096.npz

For every position of the expert games (``data/cloud/bga_positions.json.gz``,
replayed exactly) whose ``round_index`` is in ``--rounds``, the net searches
with no root noise at ``--sims`` on the Rust tree (many positions at once, one
forward pass per round of leaves, the Mac GPU or an L4). What is kept per
position: who moved (seat, whether it is the dataset's *target* seat and its
Elo), the move index inside the round, the expert's action, the net's raw prior,
the visit distribution, the per-action win-Q and margin-Q of the root's edges,
and the root value. Nothing here writes under ``data/human``: the target seat
comes from the crawl's ``meta`` block in the raw payloads (read only) and the
Elo from ``replay.stats.json``.

``report`` turns one or more label files into the tables and the disagreement
list the task force asked for (``docs/STRATEGY_TASKFORCE.md`` §E1).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ludometer.azul.engine import (
    ACTION_SPACE,
    CENTER,
    COLOR_NAMES,
    FLOOR,
    AzulState,
    decode_action,
)
from ludometer.cloud.label import PositionGame, load_positions, replay_positions

__all__ = [
    "Position",
    "RustLabeler",
    "describe_move",
    "load_targets",
    "opening_positions",
    "main",
]

POSITIONS = Path("data/cloud/bga_positions.json.gz")
RAW_DIR = Path("data/human/raw")
STATS = Path("data/human/replay.stats.json")
TARGETS_CACHE = Path("data/cloud/bga_targets.json")


# ------------------------------------------------------------------ the target
def load_targets(
    games: list[PositionGame],
    raw_dir: Path = RAW_DIR,
    stats_path: Path = STATS,
    cache: Path | None = TARGETS_CACHE,
) -> dict[int, dict[str, Any]]:
    """``table_id -> {"seat": 0|1|None, "elo": raw Elo|None}`` for the target player.

    The seat order of a compact game is the order of ``infos.data.players`` in the
    raw payload (that is what ``export_positions`` passed to the parser), and the
    target is the crawl's ``meta.target_player_id``. Read-only on ``data/human``.
    """
    if cache is not None and cache.exists():
        data = json.loads(cache.read_text())
        out = {int(k): v for k, v in data.items()}
        if all(g.table_id in out for g in games):
            return out
    elos: dict[int, float | None] = {}
    if stats_path.exists():
        for rec in json.loads(stats_path.read_text()).get("game_records") or []:
            elos[int(rec["table_id"])] = rec.get("target_elo_raw")
    from ludometer.human.client import read_json_gz

    out = {}
    for g in games:
        path = raw_dir / f"{g.table_id}.json.gz"
        seat = None
        if path.exists():
            payload = read_json_gz(path)
            infos = payload.get("infos") or {}
            players = [
                str(p)
                for p in ((infos.get("data") or infos).get("players") or {})
                if str(p).isdigit()
            ]
            target = (payload.get("meta") or {}).get("target_player_id")
            if target is not None and str(target) in players:
                seat = players.index(str(target))
        out[g.table_id] = {"seat": seat, "elo": elos.get(g.table_id)}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({str(k): v for k, v in out.items()}))
    return out


# --------------------------------------------------------------- the positions
@dataclass
class Position:
    table_id: int
    index: int  # position index inside the game (0-based)
    round_index: int
    move_in_round: int  # 1-based, over both players
    mover_move: int  # 1-based, this mover's k-th decision of the round
    mover: int
    is_target: bool
    target_elo: float | None
    expert: int
    state: AzulState

    def key(self) -> tuple[int, int]:
        return (self.table_id, self.index)


def parse_rounds(text: str) -> tuple[int, int]:
    lo, _, hi = text.partition("-")
    lo_i = int(lo)
    hi_i = int(hi) if hi else lo_i
    if hi_i < lo_i:
        raise ValueError(f"bad round range {text!r}")
    return lo_i, hi_i


def opening_positions(
    games: list[PositionGame],
    targets: dict[int, dict[str, Any]],
    rounds: tuple[int, int] = (0, 1),
    max_move: int = 0,
) -> list[Position]:
    """Every decision of both seats in ``rounds`` (``max_move`` caps the mover's
    per-round move index; 0 = all)."""
    out: list[Position] = []
    lo, hi = rounds
    for g in games:
        try:
            states, movers, _final = replay_positions(g)
        except ValueError:
            continue
        tgt = targets.get(g.table_id) or {}
        seat = tgt.get("seat")
        elo = tgt.get("elo")
        per_round: dict[int, int] = {}
        per_mover: dict[tuple[int, int], int] = {}
        for i, (state, mover) in enumerate(zip(states, movers)):
            r = state.round_index
            if r > hi:
                break
            per_round[r] = per_round.get(r, 0) + 1
            per_mover[(r, mover)] = per_mover.get((r, mover), 0) + 1
            if r < lo:
                continue
            k = per_mover[(r, mover)]
            if max_move and k > max_move:
                continue
            out.append(
                Position(
                    table_id=g.table_id,
                    index=i,
                    round_index=r,
                    move_in_round=per_round[r],
                    mover_move=k,
                    mover=mover,
                    is_target=(seat is not None and mover == seat),
                    target_elo=elo,
                    expert=int(g.actions[i]),
                    state=state,
                )
            )
    return out


# ----------------------------------------------------------------- the labeler
class RustLabeler:
    """Search many positions at once on the Rust tree; one forward pass per round.

    ``evaluator`` is a :class:`~ludometer.train.selfplay_rust.RustBatchEvaluator`
    (raw logits in, the tree softmaxes over the legal actions itself).
    """

    def __init__(
        self,
        evaluator: Any,
        config: dict[str, Any],
        slots: int = 256,
        sims: int = 4096,
        seed: int = 1,
        progress: bool = True,
    ) -> None:
        import ludometer_rs

        self._rs = ludometer_rs
        self.evaluator = evaluator
        self.config = dict(config)
        self.config["sims"] = int(sims)
        self.config["tree_reuse"] = False
        self.slots = max(1, int(slots))
        self.sims = int(sims)
        self.seed = int(seed)
        self.has_margin = bool(getattr(evaluator, "has_margin", False))
        self.evals = 0
        self.progress = progress

    def _tree(self, i: int) -> Any:
        return self._rs.Tree(
            self.config,
            has_margin=self.has_margin,
            seed=(self.seed * 1_000_003 + i) & 0x7FFFFFFF,
            add_noise=False,
            rng="fast",
        )

    def label(self, states: list[AzulState]) -> list[dict[str, Any]]:
        from ludometer.azul.engine_rs import to_rust

        out: list[dict[str, Any] | None] = [None] * len(states)
        active: list[tuple[int, Any]] = []
        next_index = 0
        t0 = time.monotonic()
        done = 0
        last = t0
        while next_index < len(states) or active:
            while len(active) < self.slots and next_index < len(states):
                i = next_index
                next_index += 1
                tree = self._tree(i)
                tree.start_search(to_rust(states[i]), add_noise=False, sims=self.sims)
                if tree.search_done():  # forced root: one legal move
                    out[i] = tree.finish_search()
                    done += 1
                else:
                    active.append((i, tree))
            if not active:
                continue
            obs_parts = []
            counts = []
            for _i, tree in active:
                obs, _legal = tree.leaf_requests(0)
                obs_parts.append(obs)
                counts.append(len(obs))
            obs_all = np.concatenate(obs_parts) if len(obs_parts) > 1 else obs_parts[0]
            logits, values, margins = self.evaluator.forward(obs_all)
            self.evals += len(obs_all)
            at = 0
            still: list[tuple[int, Any]] = []
            for (i, tree), n in zip(active, counts):
                tree.apply_logits(
                    np.ascontiguousarray(logits[at : at + n]),
                    np.ascontiguousarray(values[at : at + n]),
                    None if margins is None else np.ascontiguousarray(margins[at : at + n]),
                )
                at += n
                if tree.search_done():
                    out[i] = tree.finish_search()
                    done += 1
                else:
                    still.append((i, tree))
            active = still
            now = time.monotonic()
            if self.progress and now - last > 30:
                last = now
                rate = self.evals / max(1e-9, now - t0)
                print(
                    f"[atlas] {done}/{len(states)} positions, {self.evals:,} evals, "
                    f"{rate:,.0f}/s, {(now - t0) / 60:.1f} min",
                    flush=True,
                )
        return [o for o in out if o is not None]


# ---------------------------------------------------------------- description
ROW_WORDS = ("the 1-tile row", "the 2-tile row", "the 3-tile row", "the 4-tile row", "the 5-tile row")


def describe_move(state: AzulState, action: int) -> str:
    """One line a player understands: what was taken, from where, to where."""
    source, color, dest = decode_action(action)
    me = state.current_player
    if source == CENTER:
        n = state.center[color]
        where = "the center"
        marker = state.marker_in_center
    else:
        n = state.factories[source][color]
        where = f"factory {source}"  # the board render numbers factories from 0
        marker = False
    name = COLOR_NAMES[color]
    text = f"{n} {name} from {where}"
    if dest == FLOOR:
        text += " straight to the floor"
    else:
        have = state.pl_count[me][dest]
        cap = dest + 1
        fits = min(n, cap - have)
        over = n - fits
        text += f" to {ROW_WORDS[dest]}"
        if have:
            text += f" (already {have}/{cap})"
        if fits == cap - have:
            text += ", completing it"
        if over > 0:
            text += f", {over} to the floor"
    if marker:
        text += " (takes the first-player marker)"
    return text


# ------------------------------------------------------------------------- CLI
def _load_net_evaluator(ckpt: str, device: str, half: bool, weights_hub: str = "") -> Any:
    """``ckpt`` is a checkpoint path, or ``hub:<run>`` for a run published on
    ``weights_hub`` (the fleet's weights repo, as in ``ludometer.cloud.label``)."""
    from ludometer.train.selfplay_rust import RustBatchEvaluator

    if ckpt.startswith("hub:"):
        import os
        import tempfile

        from ludometer.cloud.hub import fetch_weights, hub_from_spec
        from ludometer.train.net import make_net, net_config_from_dict

        run = ckpt[len("hub:") :]
        hub = hub_from_spec(weights_hub, os.environ.get("HF_TOKEN"))
        with tempfile.TemporaryDirectory() as td:
            got = fetch_weights(hub, run, 0, td)
            if got is None:
                raise SystemExit(f"no weights published for {run} on {weights_hub}")
            version, net_config, weights = got
        net = make_net(net_config_from_dict(net_config))
        net.load_numpy_state_dict(weights)
        print(f"[atlas] weights {run} v{version} from {weights_hub}", flush=True)
    else:
        from ludometer.train.net import load_net

        net, _payload = load_net(ckpt, device="cpu")
    return RustBatchEvaluator(net, device=device, half=half)


def _mcts_config(path: Path | None) -> dict[str, Any]:
    from dataclasses import asdict

    from ludometer.train.mcts import MCTSConfig

    if path is None:
        return asdict(MCTSConfig())
    from ludometer.train.trainer import TrainConfig

    cfg = TrainConfig.from_dict(
        {k: v for k, v in json.loads(path.read_text()).items() if k != "started"}
    )
    return asdict(cfg.selfplay_config().mcts)


def _raw_priors(evaluator: Any, states: list[AzulState], batch: int = 1024) -> np.ndarray:
    """Softmax of the net's raw policy over the legal actions, per position."""
    out = np.zeros((len(states), ACTION_SPACE), dtype=np.float32)
    values = np.zeros(len(states), dtype=np.float32)
    for start in range(0, len(states), batch):
        chunk = states[start : start + batch]
        obs = np.stack([s.encode() for s in chunk]).astype(np.float32)
        logits, vals, _m = evaluator.forward(obs)
        for j, s in enumerate(chunk):
            legal = np.asarray(s.legal_actions(), dtype=np.int64)
            sel = logits[j][legal].astype(np.float64)
            sel = np.exp(sel - sel.max())
            out[start + j, legal] = sel / sel.sum()
        values[start : start + len(chunk)] = vals
    return out, values


def cmd_label(args: argparse.Namespace) -> int:
    name, _, ckpt = args.net.partition("=")
    if not ckpt:
        ckpt, name = name, Path(name).stem
    games = load_positions(args.positions)
    if args.limit:
        games = games[: args.limit]
    targets = load_targets(games)
    rounds = parse_rounds(args.rounds)
    positions = opening_positions(games, targets, rounds, args.max_move)
    if args.only is not None:
        wanted = set()
        with np.load(args.only) as z:
            for t, i in zip(z["table"], z["index"]):
                wanted.add((int(t), int(i)))
        positions = [p for p in positions if p.key() in wanted]
    order = np.arange(len(positions))
    if args.part != "0/1":
        k, n_parts = (int(x) for x in args.part.split("/"))
        positions = positions[k::n_parts]
        order = order[k::n_parts]
    if args.workers > 1 and args.part == "0/1":
        return _label_with_workers(args, len(positions))
    print(
        f"[atlas] {len(games)} games, rounds {rounds[0]}-{rounds[1]}"
        + (f", moves <= {args.max_move}" if args.max_move else "")
        + f": {len(positions)} positions ({sum(p.is_target for p in positions)} target-seat)",
        flush=True,
    )
    evaluator = _load_net_evaluator(ckpt, args.device, args.half, args.weights)
    config = _mcts_config(args.mcts_config)
    print(f"[atlas] net {name} ({ckpt}) on {args.device}{' fp16' if args.half else ''}; "
          f"{args.sims} sims, {args.slots} slots; c_puct {config['c_puct']}, "
          f"chance_children {config['chance_children']}", flush=True)
    states = [p.state for p in positions]
    priors, net_values = _raw_priors(evaluator, states)
    labeler = RustLabeler(evaluator, config, slots=args.slots, sims=args.sims, seed=args.seed)
    t0 = time.monotonic()
    results = labeler.label(states)
    dt = time.monotonic() - t0
    print(f"[atlas] searched {len(results)} positions in {dt / 60:.1f} min "
          f"({labeler.evals:,} evals, {labeler.evals / max(dt, 1e-9):,.0f}/s)", flush=True)
    n = len(positions)
    policy = np.zeros((n, ACTION_SPACE), dtype=np.float32)
    visits = np.zeros((n, ACTION_SPACE), dtype=np.int32)
    q = np.full((n, ACTION_SPACE), np.nan, dtype=np.float32)
    mq = np.full((n, ACTION_SPACE), np.nan, dtype=np.float32)
    value = np.zeros(n, dtype=np.float32)
    margin = np.zeros(n, dtype=np.float32)
    sims_done = np.zeros(n, dtype=np.int32)
    for i, res in enumerate(results):
        policy[i] = res["policy"]
        for a, c in res["visits"].items():
            visits[i, int(a)] = int(c)
        for a, v in res["q"].items():
            q[i, int(a)] = float(v)
        for a, v in res["margins"].items():
            mq[i, int(a)] = float(v)
        value[i] = float(res["value"])
        margin[i] = float(res["margin"])
        sims_done[i] = int(res["sims"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_labels(out, positions, order, priors, net_values, policy, visits, q, mq, value, margin, sims_done, meta={
        "net": name,
        "checkpoint": ckpt,
        "sims": args.sims,
        "rounds": list(rounds),
        "max_move": args.max_move,
        "device": args.device,
        "half": bool(args.half),
        "mcts": labeler.config,
        "seconds": round(dt, 1),
        "evals": labeler.evals,
        "created": time.time(),
    })
    print(f"[atlas] wrote {out}", flush=True)
    if args.upload:
        _upload(out, args.upload)
    return 0


def _write_labels(out, positions, order, priors, net_values, policy, visits, q, mq, value, margin, sims_done, meta):
    np.savez_compressed(
        out,
        order=np.asarray(order, dtype=np.int64),
        table=np.array([p.table_id for p in positions], dtype=np.int64),
        index=np.array([p.index for p in positions], dtype=np.int32),
        round=np.array([p.round_index for p in positions], dtype=np.int8),
        move_in_round=np.array([p.move_in_round for p in positions], dtype=np.int8),
        mover_move=np.array([p.mover_move for p in positions], dtype=np.int8),
        mover=np.array([p.mover for p in positions], dtype=np.int8),
        is_target=np.array([p.is_target for p in positions], dtype=bool),
        target_elo=np.array(
            [np.nan if p.target_elo is None else p.target_elo for p in positions],
            dtype=np.float32,
        ),
        expert=np.array([p.expert for p in positions], dtype=np.int16),
        prior=priors,
        net_value=net_values,
        policy=policy,
        visits=visits,
        q=q,
        mq=mq,
        value=value,
        margin=margin,
        sims=sims_done,
        meta=np.array(json.dumps(meta)),
    )


def _upload(out: Path, hub_spec: str) -> None:
    import os

    from ludometer.cloud.hub import hub_from_spec

    hub = hub_from_spec(hub_spec, os.environ.get("HF_TOKEN"))
    remote = f"opening/{out.name}"
    hub.put(out, remote)
    print(f"[atlas] uploaded {remote} -> {hub.describe()}", flush=True)


def _label_with_workers(args: argparse.Namespace, n_positions: int) -> int:
    """Run ``--workers`` copies of this command on interleaved parts, merge them."""
    import subprocess

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = int(args.workers)
    argv = sys.argv[1:]
    procs = []
    parts = []
    for k in range(n):
        part_out = out.with_name(f"{out.stem}.part{k}{out.suffix}")
        parts.append(part_out)
        cmd = [sys.executable, "-m", "ludometer.opening.atlas", *argv]
        # replace --out / --workers / --upload, add --part (argv[0] is the subcommand)
        clean: list[str] = []
        skip = False
        for a in cmd[3:]:
            if skip:
                skip = False
                continue
            if a in ("--out", "--workers", "--upload", "--part"):
                skip = True
                continue
            clean.append(a)
        cmd = cmd[:3] + clean + ["--part", f"{k}/{n}", "--out", str(part_out), "--workers", "1", "--seed", str(args.seed + k)]
        print(f"[atlas] worker {k}/{n}: {' '.join(cmd[3:])}", flush=True)
        procs.append(subprocess.Popen(cmd))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"a worker failed: exit codes {codes}")
    merge_parts(parts, out)
    for part in parts:
        part.unlink()
    print(f"[atlas] wrote {out} ({n_positions} positions from {n} workers)", flush=True)
    if args.upload:
        _upload(out, args.upload)
    return 0


def merge_parts(parts: list[Path], out: Path) -> None:
    arrays: dict[str, list[np.ndarray]] = {}
    metas = []
    for part in parts:
        with np.load(part) as z:
            for key in z.files:
                if key == "meta":
                    metas.append(json.loads(str(z["meta"])))
                else:
                    arrays.setdefault(key, []).append(z[key])
    merged = {k: np.concatenate(v) for k, v in arrays.items()}
    idx = np.argsort(merged["order"], kind="stable")
    merged = {k: v[idx] for k, v in merged.items()}
    meta = dict(metas[0])
    meta["seconds"] = max(m["seconds"] for m in metas)
    meta["evals"] = sum(m["evals"] for m in metas)
    meta["workers"] = len(parts)
    np.savez_compressed(out, meta=np.array(json.dumps(meta)), **merged)


def cmd_children(args: argparse.Namespace) -> int:
    """A full search from the child positions of an atlas' disagreements.

    The root's per-edge Q of a move the search barely visited is one or two
    backups of noise; the value the task force asks for is "the net's value
    after the expert's move versus after its own best move", so each child is
    searched on its own at ``--sims``. A move that ends the round is a chance
    node (the deal is unknown): its child is not searched and the root Q stays.
    """
    from ludometer.opening.report import Atlas

    name, _, ckpt = args.net.partition("=")
    if not ckpt:
        ckpt, name = name, Path(name).stem
    a = Atlas(args.atlas)
    sel = ~a.agree
    if args.target_only:
        sel &= a.d["is_target"]
    if args.rounds:
        lo, hi = parse_rounds(args.rounds)
        sel &= (a.d["round"] >= lo) & (a.d["round"] <= hi)
    idx = np.flatnonzero(sel)
    if args.part != "0/1":
        k, n_parts = (int(x) for x in args.part.split("/"))
        idx = idx[k::n_parts]
    if args.workers > 1 and args.part == "0/1":
        return _label_with_workers(args, len(idx))
    games = {g.table_id: g for g in load_positions(args.positions)}
    print(f"[atlas] children of {len(idx)} disagreements from {args.atlas}", flush=True)
    keys = []  # (row, which, mover, sign) per searched child
    states: list[AzulState] = []
    terminal: dict[tuple[int, str], float] = {}
    stochastic = 0
    cache: dict[int, list[AzulState]] = {}
    for i in idx:
        t = int(a.d["table"][i])
        if t not in cache:
            cache[t] = replay_positions(games[t])[0]
        root = cache[t][int(a.d["index"][i])]
        mover = root.current_player
        for which, action in (("expert", int(a.expert[i])), ("net", int(a.net_move[i]))):
            if root.is_stochastic(action):
                stochastic += 1
                continue
            child = root.clone()
            child.apply(action)
            if child.is_terminal:
                out = child.outcome() or 0.0
                terminal[(int(i), which)] = out if mover == 0 else -out
                continue
            keys.append((int(i), which, mover, 1.0 if child.current_player == mover else -1.0))
            states.append(child)
    print(f"[atlas] {len(states)} child searches, {len(terminal)} terminal, {stochastic} round-ending moves left to the root Q", flush=True)
    evaluator = _load_net_evaluator(ckpt, args.device, args.half, args.weights)
    config = _mcts_config(args.mcts_config)
    labeler = RustLabeler(evaluator, config, slots=args.slots, sims=args.sims, seed=args.seed)
    t0 = time.monotonic()
    results = labeler.label(states)
    dt = time.monotonic() - t0
    print(f"[atlas] searched {len(results)} children in {dt / 60:.1f} min ({labeler.evals:,} evals)", flush=True)
    rows = [k[0] for k in keys] + [k[0] for k in terminal]
    which = [k[1] for k in keys] + [k[1] for k in terminal]
    value = [float(r["value"]) * k[3] for r, k in zip(results, keys)] + list(terminal.values())
    margin = [float(r["margin"]) * k[3] for r, k in zip(results, keys)] + [0.0] * len(terminal)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        order=np.array(rows, dtype=np.int64) * 2 + np.array([0 if w == "expert" else 1 for w in which]),
        row=np.array(rows, dtype=np.int64),
        which=np.array(which),
        value=np.array(value, dtype=np.float32),
        margin=np.array(margin, dtype=np.float32),
        meta=np.array(json.dumps({"net": name, "checkpoint": ckpt, "sims": args.sims, "atlas": str(args.atlas),
                                  "mcts": labeler.config, "seconds": round(dt, 1), "evals": labeler.evals})),
    )
    print(f"[atlas] wrote {out}", flush=True)
    if args.upload:
        _upload(out, args.upload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ludometer.opening.atlas")
    sub = p.add_subparsers(dest="cmd", required=True)
    lab = sub.add_parser("label")
    lab.add_argument("--net", required=True, help="name=checkpoint.pt")
    lab.add_argument("--positions", type=Path, default=POSITIONS)
    lab.add_argument("--rounds", default="0-1", help="round_index range, e.g. 0 or 0-1")
    lab.add_argument("--max-move", type=int, default=0, help="the mover's k-th move of the round, 0 = all")
    lab.add_argument("--only", type=Path, default=None, help="restrict to the (table,index) keys of this label file")
    lab.add_argument("--sims", type=int, default=4096)
    lab.add_argument("--slots", type=int, default=256)
    lab.add_argument("--device", default="mps")
    lab.add_argument("--half", action="store_true")
    lab.add_argument("--seed", type=int, default=1)
    lab.add_argument("--limit", type=int, default=0, help="first N games only (smoke)")
    lab.add_argument(
        "--mcts-config",
        type=Path,
        default=Path("runs/porc_w-p0905-2038/config.json"),
        help="run config whose search settings to use (None = MCTSConfig defaults)",
    )
    lab.add_argument("--part", default="0/1", help="k/n: label this interleaved slice only")
    lab.add_argument("--workers", type=int, default=1, help="processes sharing the device")
    lab.add_argument("--weights", default="model:RemiFabre/rl-experiment-weights", help="hub for --net name=hub:<run>")
    lab.add_argument("--upload", default="", help="hub spec to put the result on (under opening/)")
    lab.add_argument("--out", required=True)
    ch = sub.add_parser("children")
    ch.add_argument("--atlas", type=Path, required=True)
    ch.add_argument("--net", required=True, help="name=checkpoint.pt (should be the atlas' net)")
    ch.add_argument("--positions", type=Path, default=POSITIONS)
    ch.add_argument("--rounds", default="", help="restrict to these rounds (default: the atlas' rows)")
    ch.add_argument("--target-only", action="store_true")
    ch.add_argument("--sims", type=int, default=4096)
    ch.add_argument("--slots", type=int, default=256)
    ch.add_argument("--device", default="mps")
    ch.add_argument("--half", action="store_true")
    ch.add_argument("--seed", type=int, default=11)
    ch.add_argument("--mcts-config", type=Path, default=Path("runs/porc_w-p0905-2038/config.json"))
    ch.add_argument("--part", default="0/1")
    ch.add_argument("--workers", type=int, default=1)
    ch.add_argument("--weights", default="model:RemiFabre/rl-experiment-weights")
    ch.add_argument("--upload", default="")
    ch.add_argument("--out", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "label":
        return cmd_label(args)
    if args.cmd == "children":
        return cmd_children(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
