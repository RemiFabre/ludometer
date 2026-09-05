"""The training loop: self-play -> optimize -> checkpoint + Elo, forever.

One iteration is

1. ``games_per_iter`` self-play games on the worker pool (CPU), streamed back;
2. ``K`` optimizer steps on the training device (MPS here) over minibatches
   sampled uniformly from the replay buffer, where ``K`` is chosen so every new
   position is replayed ~``steps_per_position`` times;
3. every ``eval_every_games`` games: a checkpoint plus an Elo evaluation against
   the fixed anchor pool and up to ``eval_frozen`` earlier checkpoints.

Everything observable is written to ``runs/<run>/`` exactly as specified in
docs/DESIGN.md (``config.json``, ``status.json``, ``train.jsonl``, ``elo.jsonl``,
``checkpoints/``). ``status.json`` is rewritten atomically at least every
``heartbeat`` seconds, including in the middle of a long self-play batch or eval.

The run is fully resumable: ``checkpoints/latest.pt`` holds the net, the optimizer
state and the counters, ``checkpoints/replay.npz`` holds the buffer.
"""

from __future__ import annotations

import json
import math
import os
import signal
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ludometer.eval.arena import play_match
from ludometer.eval.elo import PairResult, fit_elo
from ludometer.games import DEFAULT_GAME, get_game
from ludometer.train.mcts import MAX_GAME_MOVES, STALL_ROUNDS, MCTSConfig
from ludometer.train.mcts_agent import MCTSAgentSpec
from ludometer.train.net import (
    load_checkpoint,
    make_net,
    net_config_from_dict,
    save_checkpoint,
)
from ludometer.train.replay import ReplayBuffer, unpack_aux
from ludometer.train.selfplay import SelfPlayConfig, make_selfplay

__all__ = ["TrainConfig", "Trainer", "log_line", "resolve_device", "utc_now"]

DEFAULT_ANCHORS = ("random", "greedy", "heuristic")


# --------------------------------------------------------------------- helpers
def log_line(message: str) -> None:
    """Print and flush — a redirected log must be tailable while the run works."""
    print(message, flush=True)


def utc_now() -> str:
    """UTC ISO-8601 with an explicit offset, e.g. ``2026-08-14T15:04:05Z``."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_device(name: str = "auto") -> str:
    if name and name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():  # pragma: no cover - not this Mac
        return "cuda"
    return "cpu"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ---------------------------------------------------------------------- config
@dataclass
class TrainConfig:
    """Flat, JSON-serialisable hyperparameters (``configs/*.json``)."""

    run: str = "run"
    # which rules engine this run trains on (ludometer/games.py). Absent from
    # every run1-run6 config, so they all keep meaning Azul.
    game: str = DEFAULT_GAME
    # The game checkpoints are RATED on, when it differs from the one they train
    # on. Uno trains on single hands and is rated over matches to 500 (same
    # engine, same encoding); empty means "the same game".
    eval_game: str = ""
    seed: int = 20260814
    device: str = "auto"
    workers: int = 8
    note: str = ""

    # self-play engine: "workers" is run1-run4's (one position per forward pass,
    # `workers` CPU processes) and stays the default so every old config is
    # untouched; "batched" is run5's (see ludometer.train.selfplay_batched) —
    # `selfplay_games` concurrent trees per driver process, every tree's leaf in
    # one forward pass on `selfplay_device`, `workers` driver processes;
    # "rust" is the same engine with the rules and the tree walk in Rust
    # (ludometer.train.selfplay_rust, needs the ludometer_rs extension built).
    selfplay: str = "workers"
    selfplay_games: int = 64  # concurrent games per driver process
    selfplay_device: str = "auto"  # auto|mps|cpu|cuda — inference for self-play
    selfplay_max_batch: int = 0  # cap on rows per forward pass (0 = no cap)
    # Within-tree batching. 1 keeps each tree's search bit-identical to the
    # sequential one; above 1 costs search quality (see mcts.py) and is ramped.
    search_batch: int = 1
    search_batch_ramp: int = 16
    # "hub" (ludometer.cloud): games are played by a fleet of Hugging Face Jobs
    # and pulled from a shards store; weights are published to a weights store.
    # Specs are hub paths (`owner/name`, `model:owner/name`) or local dirs.
    hub_shards: str = ""
    hub_weights: str = ""
    hub_run: str = ""  # "" -> the run name
    hub_publish_s: float = 180.0  # min seconds between weight publishes
    hub_poll_s: float = 15.0  # seconds between shard polls when idle
    hub_max_lag: int = (
        3  # shards from weights older than this many versions are dropped
    )
    hub_delete_consumed: bool = False  # remove a shard from the hub once it is read
    search_min_batch: int = 1
    virtual_loss: float = 1.0

    # net: "mlp" (run1/run2) uses hidden/blocks; "structured" (net2.py) uses the
    # embed/layers/... block below. Both share `value_hidden`.
    arch: str = "mlp"
    hidden: int = 512
    blocks: int = 3
    value_hidden: int = 64
    embed: int = 96
    layers: int = 1
    heads: int = 4
    ffn_mult: int = 2
    body: int = 1024
    body_blocks: int = 1
    policy_rank: int = 32
    policy_global: bool = True
    # run4: a third head predicting tanh(final score diff / 20). The value head
    # then goes back to pure win/draw/loss, which is why `margin_head` and a
    # non-zero `value_score_weight` are mutually exclusive (see validate()).
    margin_head: bool = False
    # run6: 30 sigmoids predicting both players' FINAL wall sets (see net2.py).
    # Long-horizon supervision — the label is decided rounds after the position.
    aux_heads: bool = False

    # self-play search
    sims: int = 160
    c_puct: float = 1.4
    dirichlet_alpha_scale: float = 10.0
    dirichlet_eps: float = 0.25
    chance_children: int = 4
    chance_backup: str = "mean"  # how a stochastic edge combines its draws
    tree_reuse: bool = False
    # Playout-cap randomization: {"full_sims": .., "cheap_sims": .., "full_prob": ..}
    # Empty (the default, and every pre-run6 config) runs `sims` on every move.
    pcr: dict[str, float] = field(default_factory=dict)
    temp_moves: int = 12
    temperature: float = 1.0
    stall_rounds: int = STALL_ROUNDS
    max_game_moves: int = MAX_GAME_MOVES
    value_score_weight: float = 0.15
    # play-time tie-break among equally-winning root children (margin nets only)
    decisive_eps: float = 0.03
    decisive_min_visit_frac: float = 0.1

    # loop
    games_per_iter: int = 64
    total_games: int = 200_000
    min_buffer: int = 2_000
    batch_size: int = 256
    steps_per_position: float = 1.5
    steps_per_iter: int = 0  # 0 -> derive from steps_per_position
    chunk_steps: int = 25  # optimizer steps between heartbeats

    # optimizer
    lr: float = 1e-3
    lr_min: float = 1e-4
    lr_schedule: str = "cosine"  # "cosine" | "constant"
    lr_total_steps: int = 0  # 0 -> derive from total_games
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    # 2026-09-05: the value target is (1-w) * outcome + w * search root value on
    # positions that carry one (search_mask), the plain outcome elsewhere. The
    # search value is a lower-variance, slightly biased estimate; every study
    # that tried it (Willemsen et al. 2020/22 on Connect Four and Breakthrough,
    # Lc0's q_ratio) learned faster with a mix. 0 keeps the historical target.
    value_search_weight: float = 0.0
    margin_weight: float = 0.25  # weight of the margin MSE (margin_head only)
    aux_weight: float = 0.1  # weight of the final-wall BCE (aux_heads only)
    grad_clip: float = 1.0

    # replay
    replay_capacity: int = 300_000
    save_buffer: bool = True

    # pretraining on an existing replay buffer (warm start; see Trainer.pretrain)
    pretrain: str = ""  # path to a replay.npz, "" = off
    pretrain_epochs: int = 0
    pretrain_lr: float = 0.0  # 0 -> use `lr`
    # 2026-09-05: cosine-decay the pretraining rate to this floor over the whole
    # pretraining budget (0 = the historical constant rate). A big corpus at a
    # constant 1e-3 leaves the net at the top of its noise ball.
    pretrain_lr_min: float = 0.0
    pretrain_keep_buffer: bool = True  # keep the loaded positions for self-play
    # The `value_score_weight` the pretraining buffer was written with, so its
    # blended value can be split back into (outcome, margin). 0 = leave it alone
    # and mask the margin loss on those positions. See replay.unblend_values.
    pretrain_unblend: float = 0.0

    # evaluation
    eval_every_games: int = 512
    eval_games: int = 40
    eval_sims: int = 100
    eval_workers: int = 8
    eval_anchors: list[str] = field(default_factory=lambda: list(DEFAULT_ANCHORS))
    eval_frozen: int = 2
    anchor_elos: dict[str, float] = field(default_factory=lambda: {"random": 0.0})
    eval_at_start: bool = True

    heartbeat: float = 20.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        # JSON has no comments: any key starting with "_" is one (configs/run3.json
        # uses `_note` to leave instructions for whoever launches the run).
        comments = {k for k in data if k.startswith("_")}
        unknown = sorted(set(data) - known - comments - {"started"})
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        payload = {k: v for k, v in data.items() if k in known}
        cfg = cls(**payload)
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> TrainConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        if self.games_per_iter < 1:
            raise ValueError("games_per_iter must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.lr_schedule not in ("cosine", "constant"):
            raise ValueError(f"unknown lr_schedule {self.lr_schedule!r}")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be >= batch_size")
        if self.pretrain and self.pretrain_epochs < 1:
            raise ValueError("pretrain needs pretrain_epochs >= 1")
        if not 0.0 <= self.value_search_weight <= 1.0:
            raise ValueError("value_search_weight must be in [0, 1]")
        # Both game names must resolve NOW: a typo in eval_game would otherwise
        # only surface inside an eval worker process, hours into the run.
        spec = get_game(self.game)
        if self.eval_game:
            get_game(self.eval_game)
        if self.tree_reuse and not getattr(spec.state_cls, "TREE_REUSE_OK", True):
            raise ValueError(
                f"tree_reuse is a silent no-op for {self.game}: a kept subtree "
                "is a determinization the reuse guard always rejects"
            )
        if self.margin_head and self.game.startswith("uno"):
            raise ValueError(
                "the margin target tanh(diff/20) is Azul's scale; Uno hand "
                "scores saturate it (tanh(50/20)=0.99) - retune MARGIN_SCALE first"
            )
        if self.selfplay not in ("workers", "batched", "rust", "hub"):
            raise ValueError(
                f"unknown selfplay engine {self.selfplay!r} (workers | batched | rust | hub)"
            )
        if self.selfplay == "hub" and not (self.hub_shards and self.hub_weights):
            raise ValueError("selfplay='hub' needs hub_shards and hub_weights")
        if self.selfplay_games < 1:
            raise ValueError("selfplay_games must be >= 1")
        if self.margin_head:
            if self.arch != "structured":
                raise ValueError("margin_head needs arch='structured' (net2.py)")
            if self.value_score_weight:
                raise ValueError(
                    "with margin_head the value target is the pure outcome: "
                    "set value_score_weight to 0 (the margin has its own head)"
                )
        if self.aux_heads and self.arch != "structured":
            raise ValueError("aux_heads needs arch='structured' (net2.py)")
        self._validate_pcr()
        self.net_config()  # architecture keys are validated by the net configs

    def _validate_pcr(self) -> None:
        """Playout-cap randomization is all-or-nothing and must be a real split."""
        if not self.pcr:
            return
        unknown = sorted(set(self.pcr) - {"full_sims", "cheap_sims", "full_prob"})
        if unknown:
            raise ValueError(f"unknown pcr keys: {unknown}")
        full = int(self.pcr.get("full_sims", 0))
        cheap = int(self.pcr.get("cheap_sims", 0))
        prob = float(self.pcr.get("full_prob", 0.0))
        if full < 1 or cheap < 1:
            raise ValueError("pcr needs full_sims >= 1 and cheap_sims >= 1")
        if cheap > full:
            raise ValueError("pcr cheap_sims must not exceed full_sims")
        if not 0.0 < prob <= 1.0:
            raise ValueError("pcr full_prob must be in (0, 1]")
        if self.sims != full:
            raise ValueError(
                f"pcr full_sims ({full}) must equal sims ({self.sims}): the full "
                "search IS the configured search, and anything that reads `sims` "
                "(the log line, the tree-reuse budget) has to agree with it"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------ sub-configs
    def net_config(self) -> Any:
        """``NetConfig`` or ``StructuredConfig``, per :attr:`arch`."""
        data = self.to_dict()
        spec = get_game(self.game)
        data.setdefault("input_size", spec.encoded_size)
        data.setdefault("action_space", spec.action_space)
        return net_config_from_dict(data)

    def selfplay_config(self) -> SelfPlayConfig:
        return SelfPlayConfig(
            game=self.game,
            mcts=MCTSConfig(
                sims=self.sims,
                c_puct=self.c_puct,
                dirichlet_alpha_scale=self.dirichlet_alpha_scale,
                dirichlet_eps=self.dirichlet_eps,
                chance_children=self.chance_children,
                chance_backup=self.chance_backup,
                tree_reuse=self.tree_reuse,
                decisive_eps=self.decisive_eps,
                decisive_min_visit_frac=self.decisive_min_visit_frac,
                search_batch=self.search_batch,
                search_batch_ramp=self.search_batch_ramp,
                search_min_batch=self.search_min_batch,
                virtual_loss=self.virtual_loss,
            ),
            temp_moves=self.temp_moves,
            temperature=self.temperature,
            stall_rounds=self.stall_rounds,
            max_moves=self.max_game_moves,
            value_score_weight=self.value_score_weight,
            pcr_full_sims=int(self.pcr.get("full_sims", 0)),
            pcr_cheap_sims=int(self.pcr.get("cheap_sims", 0)),
            pcr_full_prob=float(self.pcr.get("full_prob", 0.0)),
        )


# --------------------------------------------------------------------- trainer
class Trainer:
    """Owns a run directory and drives the self-play/optimize/eval loop."""

    def __init__(
        self,
        config: TrainConfig,
        run_dir: str | os.PathLike[str],
        resume: bool = False,
        log: Callable[[str], None] | None = log_line,
    ) -> None:
        self.config = config
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.resume = resume
        self._log = log or (lambda _msg: None)
        self.device = resolve_device(config.device)

        self.games = 0
        self.steps = 0
        self.iteration = 0
        self.elapsed = 0.0  # seconds of run time before this process started
        self.rated: list[dict[str, Any]] = []
        self.last_eval_games = -1
        self.positions = 0
        self.decisions = 0  # cumulative searched moves (docs/NEXT_GAMES.md §4)
        self.started = utc_now()
        self._t0 = time.monotonic()
        self._last_status = 0.0
        self._stop = False
        self._stop_reason = ""
        self._state = "running"
        self._ended: str | None = None
        self._error: str | None = None
        self._note: str = config.note

        self.pretrain_steps = 0
        self.net = make_net(config.net_config()).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        spec = get_game(config.game)
        self.buffer = ReplayBuffer(
            capacity=config.replay_capacity,
            seed=config.seed ^ 0x5EED,
            input_size=spec.encoded_size,
            action_space=spec.action_space,
        )
        if config.selfplay == "hub":
            from ludometer.cloud.hub_selfplay import HubSelfPlay  # lazy: hub deps

            self.selfplay = HubSelfPlay(
                config.net_config(),
                config.selfplay_config(),
                run=config.hub_run or config.run,
                shards=config.hub_shards,
                weights=config.hub_weights,
                state_dir=self.run_dir / "hub",
                train_config=config.to_dict(),
                publish_s=config.hub_publish_s,
                poll_s=config.hub_poll_s,
                max_lag=config.hub_max_lag,
                token=os.environ.get("HF_TOKEN"),
                log=self._log,
                delete_consumed=config.hub_delete_consumed,
            )
        else:
            self.selfplay = make_selfplay(
                config.net_config(),
                config.selfplay_config(),
                config.workers,
                kind=config.selfplay,
                games=config.selfplay_games,
                device=config.selfplay_device,
                max_batch=config.selfplay_max_batch,
            )

    # -------------------------------------------------------------- properties
    @property
    def t(self) -> float:
        """Seconds since the run started (across resumes)."""
        return self.elapsed + (time.monotonic() - self._t0)

    def _ckpt_name(self, games: int | None = None) -> str:
        return f"ckpt-{self.games if games is None else games:06d}"

    # ------------------------------------------------------------------- setup
    def prepare(self) -> None:
        """Create/adopt the run directory and load any resumable state."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        latest = self.ckpt_dir / "latest.pt"
        if self.resume and latest.exists():
            self._load_state(latest)
            self._log(
                f"resumed {self.config.run} at {self.games} games / {self.steps} steps"
            )
        else:
            self.resume = False
        config_path = self.run_dir / "config.json"
        payload = self.config.to_dict()
        payload["run"] = self.config.run
        payload["started"] = self.started
        payload["device"] = self.device
        write_json_atomic(config_path, payload)
        self._write_status(force=True)

    def _load_state(self, path: Path) -> None:
        payload = load_checkpoint(path)
        self.net.load_state_dict(payload["state_dict"])
        self.net.to(self.device)
        if "optimizer" in payload:
            try:
                self.optimizer.load_state_dict(payload["optimizer"])
            except ValueError as exc:  # pragma: no cover - config changed
                self._log(f"could not restore optimizer state: {exc}")
        self.games = int(payload.get("games", 0))
        self.steps = int(payload.get("steps", 0))
        self.pretrain_steps = int(payload.get("pretrain_steps", 0))
        self.iteration = int(payload.get("iteration", 0))
        self.elapsed = float(payload.get("elapsed", 0.0))
        self.positions = int(payload.get("positions", 0))
        self.decisions = int(payload.get("decisions", 0))
        self.rated = list(payload.get("rated", []))
        self.last_eval_games = int(payload.get("last_eval_games", -1))
        self.started = str(payload.get("started", self.started))
        buffer_path = self.ckpt_dir / "replay.npz"
        if buffer_path.exists():
            n = self.buffer.load(buffer_path)
            self._log(f"restored replay buffer: {n} positions")

    # --------------------------------------------------------------- reporting
    def _write_status(self, note: str | None = None, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_status < self.config.heartbeat:
            return
        self._last_status = now
        if note is not None:
            self._note = note
        payload = {
            "run": self.config.run,
            "state": self._state,
            "started": self.started,
            "updated": utc_now(),
            "ended": self._ended if self._state != "running" else None,
            "error": self._error,
            "games": self.games,
            "steps": self.steps,
            "note": self._note,
        }
        write_json_atomic(self.run_dir / "status.json", payload)

    def heartbeat(self, note: str | None = None) -> None:
        self._write_status(note=note)

    # ------------------------------------------------------------------ signals
    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._on_signal)
            except ValueError:  # pragma: no cover - not the main thread
                pass

    def _on_signal(self, signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name
        if self._stop:  # pragma: no cover - second signal: give up immediately
            raise KeyboardInterrupt(name)
        self._stop = True
        self._stop_reason = name
        self._log(f"\n{name} received: finishing the current chunk and shutting down")

    def should_stop(self) -> bool:
        return self._stop

    # --------------------------------------------------------------------- run
    def run(self, max_games: int | None = None) -> int:
        """Run until ``max_games`` (or ``total_games``); returns an exit code."""
        cfg = self.config
        target = cfg.total_games if max_games is None else int(max_games)
        self.prepare()
        self.install_signal_handlers()
        code = 0
        try:
            shape = (
                f"{cfg.layers}L{cfg.embed}+{cfg.body_blocks}x{cfg.body}"
                if cfg.arch == "structured"
                else f"{cfg.blocks}x{cfg.hidden}"
            )
            if cfg.selfplay == "hub":
                engine = f"hub({cfg.hub_shards} <- fleet, weights -> {cfg.hub_weights})"
            elif cfg.selfplay in ("batched", "rust"):
                engine = (
                    f"{cfg.selfplay}({cfg.selfplay_games}x{cfg.workers} on "
                    f"{getattr(self.selfplay, 'device', cfg.selfplay_device)})"
                )
            else:
                engine = f"workers({cfg.workers})"
            self._log(
                f"run {cfg.run}: device={self.device} selfplay={engine} "
                f"arch={cfg.arch} net={shape} params={self.net.num_params:,} "
                f"target={target} games"
            )
            # Warm start before the workers get their first weights, so the very
            # first self-play games (and the games=0 Elo point) already use it.
            if cfg.pretrain and not self.resume and self.pretrain_steps == 0:
                self.pretrain(cfg.pretrain)
            self.selfplay.start(self.net.cpu_state_dict())
            if not self.resume and cfg.eval_at_start and self.games == 0:
                self._checkpoint_and_eval()
            while self.games < target and not self._stop:
                self._iteration(target)
            self._state = "done"
            self._stop_reason = self._stop_reason or "target reached"
        except BaseException as exc:  # noqa: BLE001 - the run must report failure
            self._state = "failed"
            self._error = f"{type(exc).__name__}: {exc}"
            code = 1
            self._log(f"run failed: {self._error}")
            import traceback

            traceback.print_exc()
        finally:
            self._ended = utc_now()
            try:
                self._save_state(final=True)
            except Exception as exc:  # noqa: BLE001 - pragma: no cover
                self._log(f"could not write the final checkpoint: {exc}")
            note = (
                f"stopped ({self._stop_reason})"
                if self._state == "done"
                else f"failed: {self._error}"
            )
            self._write_status(note=note, force=True)
            self.selfplay.close()
            self._log(
                f"{self._state}: {self.games} games, {self.steps} steps, "
                f"{self.t / 60:.1f} min"
            )
        return code

    def _iteration(self, target: int) -> None:
        cfg = self.config
        self.iteration += 1
        n_games = min(cfg.games_per_iter, target - self.games)

        # ---------------------------------------------------- 1. self-play
        t_play = time.monotonic()
        base_seed = cfg.seed * 1_000_003 + self.games
        prefix = f"iter {self.iteration}"

        def progress(done: int, total: int) -> None:
            self.heartbeat(f"{prefix}: self-play {done}/{total} games")

        records = self.selfplay.play(
            n_games,
            base_seed,
            progress=progress,
            should_stop=self.should_stop,
        )
        play_time = time.monotonic() - t_play
        added = 0
        for record in records:
            added += self.buffer.add_game(record)
        self.games += len(records)
        self.positions += added
        self.decisions += sum(r.decisions for r in records)
        self.heartbeat(f"{prefix}: {len(records)} games played")

        # ---------------------------------------------------- 2. optimize
        steps = self._steps_for(added)
        loss_p = loss_v = loss_m = loss_a = 0.0
        done_steps = 0
        t_train = time.monotonic()
        if len(self.buffer) >= max(cfg.min_buffer, cfg.batch_size) and steps > 0:
            loss_p, loss_v, loss_m, loss_a, done_steps = self._train(steps, prefix)
        train_time = time.monotonic() - t_train

        moves = sum(r.moves for r in records)
        truncated = sum(1 for r in records if r.truncated)
        games_per_min = 60.0 * len(records) / play_time if play_time > 0 else 0.0
        total_loss = loss_p + loss_v + loss_m + loss_a
        if done_steps:
            append_jsonl(
                self.run_dir / "train.jsonl",
                {
                    "t": round(self.t, 1),
                    "games": self.games,
                    "steps": self.steps,
                    "loss": round(total_loss, 4),
                    "loss_p": round(loss_p, 4),
                    "loss_v": round(loss_v, 4),
                    "loss_m": round(loss_m, 4),
                    "loss_a": round(loss_a, 4),
                    "buffer": len(self.buffer),
                    "lr": self._lr_at(self.steps),
                },
            )
        self._log(
            f"{prefix}: {len(records)} games ({games_per_min:.1f} games/min, "
            f"{moves} moves"
            + (f", {truncated} truncated" if truncated else "")
            + f") | {done_steps} steps in {train_time:.1f}s | "
            f"loss {total_loss:.4f} "
            f"(p {loss_p:.4f} v {loss_v:.4f}"
            + (f" m {loss_m:.4f}" if self.net.has_margin else "")
            + (f" a {loss_a:.4f}" if self.net.has_aux else "")
            + f") | buffer {len(self.buffer)} | games {self.games}"
            + (
                f" | lag {dict(sorted(self.selfplay.lag_hist.items()))} skipped {self.selfplay.skipped}"
                if hasattr(self.selfplay, "lag_hist")
                else ""
            )
        )

        # ---------------------------------------------------- 3. checkpoint
        if self.games - max(self.last_eval_games, 0) >= cfg.eval_every_games:
            self._checkpoint_and_eval()
        else:
            self._save_state()
        if not self._stop:
            self.selfplay.set_weights(self.net.cpu_state_dict())

    # ------------------------------------------------------------- pretraining
    def pretrain(self, path: str | os.PathLike[str], epochs: int | None = None) -> int:
        """Fit the fresh net to an existing replay buffer before any self-play.

        This is the warm start: run2 spent a night filling ``replay.npz`` with
        (state, visit distribution, value) triples, and a new architecture can
        learn most of what that data has to say in minutes of supervised
        training — policy cross-entropy against the stored MCTS visit counts plus
        value MSE, i.e. exactly the loss :meth:`_train` uses, only over shuffled
        full passes instead of random minibatches.

        The positions are loaded straight into the run's own replay buffer (it is
        capacity-bounded, so a bigger file keeps its newest positions), and by
        default they stay there: early self-play then trains on a mix of its own
        fresh games and the inherited ones instead of on a nearly empty buffer.

        A pre-run4 buffer has no margin column, so those positions come in with a
        zero ``margin_mask`` and the margin head simply gets no gradient from
        them — a run4 net can still be pretrained on run1/run2/run3 data. Setting
        ``pretrain_unblend`` to the weight that produced the file recovers the
        margin exactly instead (see :func:`ludometer.train.replay.unblend_values`),
        which is what ``configs/run4.json`` does with run3's 0.15.

        Returns the number of optimizer steps taken. These do **not** advance
        ``self.steps``: the self-play learning-rate schedule is meant to start at
        its peak when self-play starts.
        """
        cfg = self.config
        epochs = cfg.pretrain_epochs if epochs is None else int(epochs)
        if epochs < 1:
            return 0
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"pretrain buffer not found: {target}")
        self._write_status(note=f"pretrain: loading {target}", force=True)
        n = self.buffer.load(target, unblend=cfg.pretrain_unblend)
        stats = self.buffer.stats()
        covered = []
        if self.net.has_margin:
            covered.append(f"{stats['margin_targets']:,} with a margin target")
        if self.net.has_aux:
            covered.append(f"{stats['aux_targets']:,} with final-wall targets")
        self._log(
            f"pretrain: loaded {n:,} positions from {target}"
            + (f" ({'; '.join(covered)})" if covered else "")
        )
        if n < cfg.batch_size:
            raise ValueError(f"pretrain buffer has only {n} positions")

        lr = cfg.pretrain_lr or cfg.lr
        net = self.net
        rng = np.random.default_rng(cfg.seed ^ 0xB00C)
        steps_per_epoch = n // cfg.batch_size
        total_steps = max(1, steps_per_epoch * epochs)

        def lr_at(step: int) -> float:
            if cfg.pretrain_lr_min <= 0.0:
                return lr
            frac = min(1.0, step / total_steps)
            return cfg.pretrain_lr_min + 0.5 * (lr - cfg.pretrain_lr_min) * (
                1.0 + math.cos(math.pi * frac)
            )

        t_start = time.monotonic()
        for epoch in range(1, epochs + 1):
            net.train()
            order = rng.permutation(n)
            sum_p = sum_v = sum_m = sum_a = 0.0
            for i in range(steps_per_epoch):
                idx = np.sort(order[i * cfg.batch_size : (i + 1) * cfg.batch_size])
                loss_p, loss_v, loss_m, loss_a = self._losses(
                    self.buffer.states[idx],
                    self.buffer.policies[idx],
                    self.buffer.values[idx],
                    self.buffer.margins[idx],
                    self.buffer.margin_mask[idx],
                    unpack_aux(self.buffer.aux[idx]),
                    self.buffer.aux_mask[idx],
                    self.buffer.policy_mask[idx],
                    self.buffer.search_values[idx],
                    self.buffer.search_mask[idx],
                )
                loss = self._total_loss(loss_p, loss_v, loss_m, loss_a)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr_at(self.pretrain_steps)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                self.optimizer.step()
                sum_p += float(loss_p.detach())
                sum_v += float(loss_v.detach())
                sum_m += float(loss_m.detach())
                sum_a += float(loss_a.detach())
                self.pretrain_steps += 1
                if (i + 1) % max(1, cfg.chunk_steps) == 0:
                    self.heartbeat(
                        f"pretrain: epoch {epoch}/{epochs} "
                        f"step {i + 1}/{steps_per_epoch}"
                    )
                if self._stop:
                    break
            net.eval()
            done = max(1, steps_per_epoch)
            append_jsonl(
                self.run_dir / "train.jsonl",
                {
                    "phase": "pretrain",
                    "t": round(self.t, 1),
                    "games": self.games,
                    "steps": self.steps,
                    "epoch": epoch,
                    "pretrain_steps": self.pretrain_steps,
                    "loss": round((sum_p + sum_v + sum_m + sum_a) / done, 4),
                    "loss_p": round(sum_p / done, 4),
                    "loss_v": round(sum_v / done, 4),
                    "loss_m": round(sum_m / done, 4),
                    "loss_a": round(sum_a / done, 4),
                    "buffer": len(self.buffer),
                    "lr": lr,
                },
            )
            self._log(
                f"pretrain epoch {epoch}/{epochs}: "
                f"loss {(sum_p + sum_v + sum_m + sum_a) / done:.4f} "
                f"(p {sum_p / done:.4f} v {sum_v / done:.4f}"
                + (f" m {sum_m / done:.4f}" if net.has_margin else "")
                + (f" a {sum_a / done:.4f}" if net.has_aux else "")
                + f") in {time.monotonic() - t_start:.0f}s"
            )
            if self._stop:
                break
        if not cfg.pretrain_keep_buffer:
            spec = get_game(cfg.game)
            self.buffer = ReplayBuffer(
                capacity=cfg.replay_capacity,
                seed=cfg.seed ^ 0x5EED,
                input_size=spec.encoded_size,
                action_space=spec.action_space,
            )
        # the self-play schedule starts fresh: reset the LR the optimizer holds
        for group in self.optimizer.param_groups:
            group["lr"] = self._lr_at(self.steps)
        self._save_state()
        self._write_status(
            note=f"pretrained {self.pretrain_steps} steps on {n} positions", force=True
        )
        return self.pretrain_steps

    # ---------------------------------------------------------------- training
    def _steps_for(self, positions_added: int) -> int:
        cfg = self.config
        if cfg.steps_per_iter > 0:
            return int(cfg.steps_per_iter)
        return round(positions_added * cfg.steps_per_position / cfg.batch_size)

    def _lr_at(self, step: int) -> float:
        cfg = self.config
        if cfg.lr_schedule == "constant":
            return cfg.lr
        total = cfg.lr_total_steps
        if total <= 0:
            # derive from the game budget: same steps/game as the current setting
            per_game = max(
                cfg.steps_per_iter / cfg.games_per_iter
                if cfg.steps_per_iter > 0
                else 55.0 * cfg.steps_per_position / cfg.batch_size,
                1e-6,
            )
            total = max(1, int(cfg.total_games * per_game))
        frac = min(max(step / total, 0.0), 1.0)
        return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (
            1.0 + math.cos(math.pi * frac)
        )

    def _losses(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        margins: np.ndarray,
        mask: np.ndarray,
        aux: np.ndarray | None = None,
        aux_mask: np.ndarray | None = None,
        policy_mask: np.ndarray | None = None,
        search_values: np.ndarray | None = None,
        search_mask: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(policy CE, value MSE, margin MSE, aux BCE)`` for one minibatch.

        Three of the four are **masked** means, and they are masked for the same
        reason: a position may legitimately not carry that target.

        * the margin, since run4: positions inherited from a pre-run4 replay
          buffer have none, so they contribute nothing instead of pulling the head
          towards a fabricated zero;
        * the aux heads, since run6: same story for a pre-run6 buffer, which is
          exactly what pretraining run6 on run5's data is;
        * the **policy**, since run6: a position searched cheaply under
          playout-cap randomization has a zeroed policy row. A zero row already
          contributes no gradient (the CE is a dot product with it), but it would
          still divide the batch mean, so the loss — and with it the effective
          learning rate on the policy head — would shrink with the cheap fraction.
          Dividing by the number of *real* targets keeps the policy gradient the
          same size it would be in a run without cheap searches.

        With no head, or no masked-in position in the batch, a term is exactly 0
        and carries no gradient.
        """
        device = self.device
        x = torch.from_numpy(states).to(device, non_blocking=True)
        target_p = torch.from_numpy(policies).to(device, non_blocking=True)
        target_v = torch.from_numpy(values).to(device, non_blocking=True)
        w_search = self.config.value_search_weight
        if w_search > 0.0 and search_values is not None and search_mask is not None:
            s_v = torch.from_numpy(search_values).to(device, non_blocking=True)
            s_w = torch.from_numpy(search_mask).to(device, non_blocking=True) * w_search
            target_v = (1.0 - s_w) * target_v + s_w * s_v
        logits, value, margin, aux_logits = self.net.forward_aux(x)
        logp = torch.log_softmax(logits, dim=-1)
        per_row_p = -(target_p * logp).sum(dim=1)
        if policy_mask is None:
            loss_p = per_row_p.mean()
        else:
            p_w = torch.from_numpy(policy_mask).to(device, non_blocking=True)
            loss_p = (per_row_p * p_w).sum() / p_w.sum().clamp(min=1.0)
        loss_v = torch.nn.functional.mse_loss(value, target_v)
        zero = torch.zeros((), device=device)
        loss_m = zero
        if margin is not None:
            target_m = torch.from_numpy(margins).to(device, non_blocking=True)
            weights = torch.from_numpy(mask).to(device, non_blocking=True)
            loss_m = (
                (margin - target_m).square() * weights
            ).sum() / weights.sum().clamp(min=1.0)
        loss_a = zero
        if aux_logits is not None and aux is not None:
            target_a = torch.from_numpy(aux).to(device, non_blocking=True)
            a_w = torch.from_numpy(
                np.ones(len(aux), dtype=np.float32) if aux_mask is None else aux_mask
            ).to(device, non_blocking=True)
            # BCE per output, averaged over the 30 questions, masked over rows.
            per_row_a = torch.nn.functional.binary_cross_entropy_with_logits(
                aux_logits, target_a, reduction="none"
            ).mean(dim=1)
            loss_a = (per_row_a * a_w).sum() / a_w.sum().clamp(min=1.0)
        return loss_p, loss_v, loss_m, loss_a

    def _total_loss(
        self,
        loss_p: torch.Tensor,
        loss_v: torch.Tensor,
        loss_m: torch.Tensor,
        loss_a: torch.Tensor,
    ) -> torch.Tensor:
        """The one weighted sum both the self-play loop and pretraining minimise."""
        cfg = self.config
        return (
            loss_p
            + cfg.value_weight * loss_v
            + cfg.margin_weight * loss_m
            + cfg.aux_weight * loss_a
        )

    def _train(self, steps: int, prefix: str) -> tuple[float, float, float, float, int]:
        cfg = self.config
        device = self.device
        net = self.net
        net.train()
        sums = [torch.zeros((), device=device) for _ in range(4)]
        done = 0
        chunk = max(1, cfg.chunk_steps)
        while done < steps:
            todo = min(chunk, steps - done)
            for _ in range(todo):
                batch = self.buffer.sample(cfg.batch_size)
                losses = self._losses(*batch)
                loss = self._total_loss(*losses)
                lr = self._lr_at(self.steps)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                self.optimizer.step()
                for i, term in enumerate(losses):
                    sums[i] += term.detach()
                self.steps += 1
            done += todo
            self.heartbeat(f"{prefix}: training {done}/{steps} steps")
            if self._stop:
                break
        net.eval()
        if done == 0:  # pragma: no cover - defensive
            return 0.0, 0.0, 0.0, 0.0, 0
        return (*(float(total) / done for total in sums), done)

    # ------------------------------------------------------------- checkpoints
    def _save_state(self, final: bool = False) -> Path:
        extra = {
            "games": self.games,
            "steps": self.steps,
            "pretrain_steps": self.pretrain_steps,
            "iteration": self.iteration,
            "elapsed": self.t,
            "positions": self.positions,
            "decisions": self.decisions,
            "started": self.started,
            "rated": self.rated,
            "last_eval_games": self.last_eval_games,
            "config": self.config.to_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        path = save_checkpoint(self.ckpt_dir / "latest.pt", self.net, extra)
        if final and self.config.save_buffer and len(self.buffer):
            self.buffer.save(self.ckpt_dir / "replay.npz")
        return path

    def _checkpoint_and_eval(self) -> dict[str, Any] | None:
        name = self._ckpt_name()
        path = save_checkpoint(
            self.ckpt_dir / f"{name}.pt",
            self.net,
            {"games": self.games, "steps": self.steps, "config": self.config.to_dict()},
        )
        self.last_eval_games = self.games
        self._save_state()
        if self.config.save_buffer and len(self.buffer):
            self.heartbeat(f"saving replay buffer ({len(self.buffer)} positions)")
            self.buffer.save(self.ckpt_dir / "replay.npz")
        return self._evaluate(name, path)

    # ---------------------------------------------------------------- Elo eval
    def _frozen_opponents(self) -> list[dict[str, Any]]:
        """Up to ``eval_frozen`` earlier rated checkpoints: newest and strongest."""
        available = [r for r in self.rated if Path(r["path"]).exists()]
        if not available or self.config.eval_frozen <= 0:
            return []
        picks: list[dict[str, Any]] = [available[-1]]
        strongest = max(available, key=lambda r: r["elo"])
        if strongest["name"] != picks[0]["name"]:
            picks.append(strongest)
        return picks[: self.config.eval_frozen]

    def _evaluate(self, name: str, path: Path) -> dict[str, Any] | None:
        cfg = self.config
        if cfg.eval_games <= 0:
            return None
        t_start = time.monotonic()
        candidate = MCTSAgentSpec(
            path=str(path),
            sims=cfg.eval_sims,
            seed=cfg.seed,
            name=name,
            stall_rounds=cfg.stall_rounds,
        )
        frozen = self._frozen_opponents()
        opponents: list[tuple[str, Any]] = [(a, a) for a in cfg.eval_anchors]
        for entry in frozen:
            opponents.append(
                (
                    entry["name"],
                    MCTSAgentSpec(
                        path=entry["path"],
                        sims=cfg.eval_sims,
                        seed=cfg.seed + 1,
                        name=entry["name"],
                        stall_rounds=cfg.stall_rounds,
                    ),
                )
            )

        results: list[PairResult] = []
        versus: dict[str, float] = {}
        total_games = 0
        stalled = 0  # eval games that hit the arena backstop (see eval/arena.py)
        base_seed = cfg.seed * 7_919 + self.games
        for i, (opp_name, spec) in enumerate(opponents):
            self.heartbeat(f"eval {name}: vs {opp_name} ({i + 1}/{len(opponents)})")
            match = play_match(
                candidate,
                spec,
                n_games=cfg.eval_games,
                base_seed=base_seed + 100_000 * i,
                n_workers=cfg.eval_workers,
                game=cfg.eval_game or cfg.game,
            )
            results.append(
                PairResult(name, opp_name, match.wins, match.draws, match.losses)
            )
            versus[opp_name] = round(match.win_rate, 3)
            total_games += match.n_games
            stalled += match.truncated

        # Random stays pinned at 0 and previously published checkpoints keep their
        # rating, so every point of the curve lives on the same scale.
        anchors = {k: float(v) for k, v in cfg.anchor_elos.items() if k in versus}
        for entry in frozen:
            anchors[entry["name"]] = float(entry["elo"])
        fit = fit_elo(results, anchors=anchors, error_method="fisher")
        elo = float(fit.ratings[name])
        elo_err = float(fit.errors.get(name, 0.0))
        pool = [
            (f"{opp}={anchors[opp]:.1f}" if opp in anchors else opp) for opp in versus
        ]
        record = {
            "t": round(self.t, 1),
            "games": self.games,
            # cumulative practice in the two cross-game units (NEXT_GAMES.md §4):
            # every position generated, and only the searched (non-forced) moves.
            "positions": self.positions,
            "decisions": self.decisions,
            "ckpt": name,
            "elo": round(elo, 1),
            "elo_err": round(elo_err, 1),
            "vs": versus,
            "n_games": total_games,
            "pool": pool,
        }
        if stalled:
            # Never expected (a real game is ~54 moves): worth seeing in the log
            # rather than quietly absorbed into the draw column.
            record["truncated"] = stalled
        append_jsonl(self.run_dir / "elo.jsonl", record)
        self.rated.append(
            {
                "name": name,
                "path": str(path),
                "elo": round(elo, 1),
                "games": self.games,
                "t": round(self.t, 1),
            }
        )
        self._save_state()
        self._log(
            f"eval {name}: elo {elo:+.1f} +/- {elo_err:.1f} over {total_games} games "
            f"in {time.monotonic() - t_start:.0f}s | "
            + (f"{stalled} TRUNCATED | " if stalled else "")
            + " ".join(f"{k} {v:.2f}" for k, v in versus.items())
        )
        self._write_status(note=f"rated {name}: {elo:+.0f} Elo", force=True)
        return record
