"""Self-play on the Rust engine: `make_selfplay(kind="rust")`.

The same shape as :mod:`ludometer.train.selfplay_batched` — ``games`` concurrent
trees per driver process, every tree's leaf in one forward pass — with the
rules, the tree walk and the game loop in Rust (``ludometer_rs.Arena``, see
``rust/`` and ``docs/RUST_ENGINE.md``). Python keeps exactly two jobs: run the net
on the ``[B, 182]`` rows the arena hands out, and turn the finished games into
:class:`~ludometer.train.selfplay.GameRecord` objects for the replay buffer and
the shard writer. The pool interface (``start``, ``set_weights``, ``play``,
``close``), the streamed records, the ``("progress", worker_id, positions)``
ticks and the contiguous seed blocks are the batched engine's, so the trainer
and the fleet generator cannot tell the two apart.

Equivalence: ``tests/test_rust_arena.py`` shows the arena reproduces the
batched engine's games array for array given identical evaluations (root noise
off); with noise on the two engines sample from the same distribution with
different random streams (docs/RUST_ENGINE.md §6, layer 3).
"""

from __future__ import annotations

import signal
import time
from collections.abc import Callable
from typing import Any, Self

import numpy as np

from ludometer.train.mcts import MCTSConfig
from ludometer.train.net import make_net
from ludometer.train.selfplay import GameRecord, SelfPlayConfig, SelfPlayPool
from ludometer.train.selfplay_batched import resolve_selfplay_device

__all__ = [
    "RustBatchEvaluator",
    "RustSelfPlay",
    "RustSelfPlayPool",
    "flat_selfplay_config",
    "record_from_dict",
    "rust_available",
]


def rust_available() -> bool:
    try:
        import ludometer_rs  # noqa: F401
    except ImportError:
        return False
    return True


def _require_rust() -> Any:
    try:
        import ludometer_rs
    except ImportError as exc:  # pragma: no cover - depends on the build
        raise ImportError(
            "ludometer_rs is not built: see rust/README.md "
            "(uv pip install maturin; maturin develop --release -m rust/ludometer-engine/Cargo.toml)"
        ) from exc
    return ludometer_rs


def flat_selfplay_config(config: SelfPlayConfig) -> dict[str, Any]:
    """The flat dict ``ludometer_rs.Arena`` reads: MCTS keys + the game-loop knobs."""
    if config.game not in ("azul",):
        raise ValueError(f"the rust engine plays azul only, not {config.game!r}")
    out: dict[str, Any] = {
        k: getattr(config.mcts, k) for k in MCTSConfig.__dataclass_fields__
    }
    for k in (
        "temp_moves",
        "temperature",
        "stall_rounds",
        "max_moves",
        "value_score_weight",
        "pcr_full_sims",
        "pcr_cheap_sims",
        "pcr_full_prob",
    ):
        out[k] = getattr(config, k)
    return out


def record_from_dict(d: dict[str, Any]) -> GameRecord:
    """A :class:`GameRecord` from what ``Arena.drain()`` returns."""
    return GameRecord(
        states=d["states"],
        policies=d["policies"],
        values=d["values"],
        margins=d["margins"],
        aux=d["aux"],
        policy_mask=d["policy_mask"],
        outcome=float(d["outcome"]),
        scores=(int(d["scores"][0]), int(d["scores"][1])),
        moves=int(d["moves"]),
        rounds=int(d["rounds"]),
        seed=int(d["seed"]),
        decisions=int(d["decisions"]),
        evals=int(d["evals"]),
        duration=float(d["duration"]),
        truncated=bool(d["truncated"]),
        search_values=d["search_values"],
        search_mask=d["search_mask"],
    )


# --------------------------------------------------------------------- evaluator
class RustBatchEvaluator:
    """One forward pass over ``[B, 182]`` rows -> ``(logits, values, margins)``.

    :class:`~ludometer.train.selfplay_batched.BatchEvaluator` minus the encoding
    and the softmax: the arena encodes the rows itself and softmaxes the logits
    over each leaf's legal actions in Rust (float32, numpy's order of
    operations). ``margins`` is ``None`` for a net without the head.
    """

    def __init__(
        self, net: Any, device: str = "cpu", max_batch: int = 0, half: bool = False
    ) -> None:
        import torch

        self.torch = torch
        self.net = net
        self.device = torch.device(device)
        self.half = bool(half)
        self.net.to(self.device)
        if self.half:
            self.net.half()
        self.net.eval()
        self.has_margin = bool(getattr(net, "has_margin", False))
        self.max_batch = max(0, int(max_batch))
        self.calls = 0
        self.rows = 0

    def forward(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        n = len(obs)
        chunk = self.max_batch or n
        parts: list[np.ndarray] = []
        torch = self.torch
        with torch.inference_mode():
            for start in range(0, n, chunk):
                block = np.ascontiguousarray(obs[start : start + chunk])
                x = torch.from_numpy(block).to(self.device)
                if self.half:
                    x = x.half()
                logits, value, margin = self.net.forward_heads(x)
                # One readback (a full sync on MPS), not three.
                tail = (
                    value.unsqueeze(1)
                    if margin is None
                    else torch.stack((value, margin), dim=1)
                )
                parts.append(
                    torch.cat((logits, tail), dim=1)
                    .to("cpu", torch.float32, copy=True)
                    .numpy()
                )
                self.calls += 1
                self.rows += len(block)
        packed = parts[0] if len(parts) == 1 else np.concatenate(parts)
        action_space = packed.shape[1] - (2 if self.has_margin else 1)
        logits = np.ascontiguousarray(packed[:, :action_space])
        values = np.ascontiguousarray(packed[:, action_space])
        margins = (
            np.ascontiguousarray(packed[:, action_space + 1])
            if self.has_margin
            else None
        )
        return logits, values, margins

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        if self.half:
            self.net.float()
        self.net.load_numpy_state_dict(weights)
        self.net.to(self.device)
        if self.half:
            self.net.half()
        self.net.eval()


# ------------------------------------------------------------------- the engine
class RustSelfPlay:
    """``games`` self-play games in one process on the Rust arena.

    Same API as :class:`~ludometer.train.selfplay.SelfPlayPool` and
    :class:`~ludometer.train.selfplay_batched.BatchedSelfPlay`.
    """

    def __init__(
        self,
        net_config: Any,
        config: SelfPlayConfig,
        games: int = 64,
        device: str = "auto",
        max_batch: int = 0,
        leaf_cap: int = 0,
        tick: float = 2.0,
        half: bool = False,
        rng: str = "fast",
    ) -> None:
        rs = _require_rust()
        self.tick = float(tick)
        self.net_config = net_config
        self.config = config
        self.games = max(1, int(games))
        self.device = resolve_selfplay_device(device)
        self.max_batch = int(max_batch)
        self.leaf_cap = int(leaf_cap) or max(1, config.mcts.search_batch)
        self.workers = 1
        self.rng = rng
        self.net = make_net(net_config)
        self.net.eval()
        self.evaluator = RustBatchEvaluator(
            self.net, device=self.device, max_batch=self.max_batch, half=half
        )
        self.arena = rs.Arena(
            flat_selfplay_config(config),
            has_margin=self.evaluator.has_margin,
            games=self.games,
            rng=rng,
            leaf_cap=self.leaf_cap,
        )
        self.batches = 0
        self.positions = 0
        self.eval_seconds = 0.0
        self.search_seconds = 0.0

    # ------------------------------------------------------------------ setup
    def start(self, weights: dict[str, np.ndarray] | None = None) -> None:
        if weights is not None:
            self.set_weights(weights)

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        self.evaluator.set_weights(weights)

    def close(self) -> None:
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------- play
    def play(
        self,
        n_games: int,
        seed_start: int,
        progress: Any = None,
        should_stop: Any = None,
        on_record: Callable[[GameRecord], None] | None = None,
    ) -> list[GameRecord]:
        """Play ``n_games`` (seeds ``seed_start ...``); records as they finish."""
        if n_games <= 0:
            return []
        arena = self.arena
        arena.begin(int(n_games), int(seed_start))
        out: list[GameRecord] = []
        next_tick = time.monotonic() + self.tick

        def take() -> bool:
            for d in arena.drain():
                record = record_from_dict(d)
                out.append(record)
                if on_record is not None:
                    on_record(record)
                if progress is not None:
                    progress(len(out), n_games)
                if should_stop is not None and should_stop():
                    arena.set_stop()
                    return True
            return False

        while not arena.finished():
            now = time.monotonic()
            if now >= next_tick:
                next_tick = now + self.tick
                if progress is not None:
                    progress(len(out), n_games)
                if should_stop is not None and should_stop():
                    arena.set_stop()
                    break
            t_search = time.perf_counter()
            obs = arena.gather()
            t_eval = time.perf_counter()
            self.search_seconds += t_eval - t_search
            if take() or arena.finished():
                break
            if not len(obs):  # pragma: no cover - every slot just finished
                continue
            logits, values, margins = self.evaluator.forward(obs)
            t_back = time.perf_counter()
            self.eval_seconds += t_back - t_eval
            arena.apply_logits(logits, values, margins)
            self.search_seconds += time.perf_counter() - t_back
            self.batches += 1
            self.positions += len(obs)
        take()
        return out


# ------------------------------------------------------------------- processes
def _rust_worker_loop(
    worker_id: int,
    net_config: Any,
    config: SelfPlayConfig,
    cmd_q: Any,
    result_q: Any,
    games: int,
    device: str,
    max_batch: int,
    half: bool = False,
) -> None:  # pragma: no cover - runs in a child process
    """Worker entry point: one Rust arena + one net, fed blocks of seeds."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import torch

    torch.set_num_threads(1)
    engine = RustSelfPlay(
        net_config, config, games=games, device=device, max_batch=max_batch, half=half
    )
    while True:
        message = cmd_q.get()
        kind = message[0]
        if kind == "stop":
            break
        if kind == "weights":
            engine.set_weights(message[1])
            continue
        if kind == "play":
            seeds = list(message[1])
            try:
                engine.play(
                    len(seeds),
                    seeds[0],
                    on_record=lambda rec: result_q.put(rec),
                    progress=lambda _d, _t: result_q.put(
                        ("progress", worker_id, engine.positions)
                    ),
                )
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                result_q.put(("error", worker_id, f"{type(exc).__name__}: {exc}"))


class RustSelfPlayPool(SelfPlayPool):
    """``workers`` Rust drivers, each running ``games`` concurrent trees."""

    worker_main = staticmethod(_rust_worker_loop)

    def __init__(
        self,
        net_config: Any,
        config: SelfPlayConfig,
        workers: int = 4,
        games: int = 32,
        device: str = "auto",
        max_batch: int = 0,
        poll: float = 1.0,
        half: bool = False,
    ) -> None:
        _require_rust()
        flat_selfplay_config(config)  # fail early on a non-Azul game
        super().__init__(net_config, config, workers=workers, poll=poll)
        self.games = max(1, int(games))
        self.device = device
        self.worker_extra = (self.games, device, int(max_batch), bool(half))

    def _dispatch(self, n_games: int, seed_start: int) -> None:
        """One contiguous block of seeds per worker (the batch needs them all)."""
        base, extra = divmod(n_games, self.workers)
        at = seed_start
        for wid in range(self.workers):
            count = base + (1 if wid < extra else 0)
            if count <= 0:
                continue
            self._cmd_qs[wid].put(("play", list(range(at, at + count))))
            at += count
