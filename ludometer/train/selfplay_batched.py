"""Batched self-play: many games in one process, one forward pass for all of them.

Why
---
run1-run4 self-play (:mod:`ludometer.train.selfplay`) evaluates **one position per
forward pass**, on CPU, in each of 8 worker processes. That is the cheapest thing
a net can do per unit of hardware: on this Mac one position through run4's 1.8M
net costs ~0.66 ms on a contended CPU thread, while 128 positions through the
*same* net cost 1.57 ms *in total* on the GPU — 80,000 positions/s against 1,500.
The browser player hit exactly this wall and the fix is the same one: stop asking
for one position at a time.

Two levels of batching, and they are very different
---------------------------------------------------
**Across games** — the big win, and free. ``games`` independent trees are searched
concurrently; each contributes the leaf it currently wants, all of them go into
one tensor, one forward pass answers all of them. Every individual tree still
runs a strictly sequential search — it never has more than one position in flight
— so with ``search_batch = 1`` the trajectories are *bit-identical* to what the
sequential engine would have produced from the same seed. ``tests/
test_selfplay_batched.py`` asserts that game for game.

**Within one tree** (``search_batch > 1``) — quantity bought with a little
quality. Several descents are made before one forward pass, each laying a virtual
loss so the next is pushed elsewhere (see :mod:`ludometer.train.mcts`). It is off
by default; when on, the batch is ramped with the tree
(``search_batch_ramp``), because the browser measured a *flat* batch of 64 losing
3-17 to a batch-1 search at equal simulations. With 64-128 concurrent games there
is already a full tensor to fill without it, which is why run5 leaves it at 1.

What is preserved
-----------------
Everything the sequential path does, because it is the same code doing it: the
margin head (leaf evaluations return three values, the search backs the margin up
and :func:`~ludometer.train.mcts.select_play_action` breaks ties on it), tree
reuse (a per-tree property; it composes with cross-game batching without either
side knowing about the other), the temperature schedule, the stall breaker, the
``max_moves`` backstop, chance-node determinization, and run6's playout-cap
randomization (the full/cheap draw comes from a per-game RNG seeded off the game
seed and is taken at the same point in the move loop as the sequential path, so
the two engines schedule an identical game). A game's RNG streams are
its own (``AzulState.new_game(seed)`` and one :class:`~ludometer.train.mcts.MCTS`
seeded from the same number), so a game's trajectory does not depend on which
other games happened to share its batches — only on the evaluations it gets back.

The one compromise worth stating plainly: **"the evaluations it gets back" are not
bit-stable across batch shapes, and that is true on the CPU too.** A backend picks
different kernels and different blocking for different tensor sizes, so the same
position evaluated alone and in a batch of 128 can differ in the last float32
digits. Measured on this Mac's CPU with a small net (``tests/
test_selfplay_batched.py`` pins it): exactly 0 at batch 1, ~1e-8 by batch 8, ~4e-8
by batch 40. Nothing about a *value* cares about 1e-8 — but PUCT is a *ranking*,
so once in a long while such a difference flips which child a descent picks and
two trajectories part company from there.

So the honest statement of the guarantee, which is what the tests assert:

* **exact** at ``games = 1``, where every forward pass carries one position and
  the arithmetic is the single-position path's, bit for bit;
* **exact per game given identical evaluations** — the search bookkeeping itself
  introduces nothing, which is the property that actually matters and the one
  ``test_pumped_search_is_identical_to_the_blocking_one`` pins directly;
* **statistical** for a real ``games = 128`` run, on CPU and on MPS alike: a
  game's trajectory can depend on how many other games happened to share its
  batches. That is the same trade the trainer already makes (its own optimizer
  steps run on MPS), and it costs nothing in training — every trajectory is a
  legitimate game of the same distribution.

Scaling past one process
------------------------
One driver process is the unit this module defines, and it is what
``workers <= 1`` builds. But the descents themselves are pure Python — ~90 us per
simulation, of which the net is now a small part — so a *single* driver cannot
saturate the GPU: it is CPU-bound long before it gets there.
:class:`BatchedSelfPlayPool` therefore runs ``workers`` driver processes, each
with its own ``games`` trees and its own MPS queue, which is how the engine
actually beats 8 sequential CPU workers rather than merely matching them.
"""

from __future__ import annotations

import contextlib
import gc
import signal
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

from ludometer.azul.engine import AzulState
from ludometer.games import get_game
from ludometer.train.mcts import MCTS, LeafRequest, select_play_action
from ludometer.train.net import make_net
from ludometer.train.selfplay import (
    GameRecord,
    SelfPlayConfig,
    SelfPlayPool,
    aux_targets,
    margin_targets,
    pcr_rng,
    pcr_sims,
    value_target,
)

__all__ = [
    "BatchEvaluator",
    "BatchedSelfPlay",
    "BatchedSelfPlayPool",
    "resolve_selfplay_device",
]


@contextlib.contextmanager
def _relaxed_gc():
    """Stop the cyclic collector from re-scanning the live trees, measured.

    ``games`` concurrent 512-simulation trees keep on the order of a hundred
    thousand :class:`~ludometer.train.mcts.Node` objects (and their cloned
    states, which are lists of lists) alive at once. CPython's generation-2
    collection walks *every* one of them, and it fires roughly every 70k
    allocations by default — which a batched driver reaches in seconds. Measured
    on this Mac with 128 concurrent games: **2,803 positions/s with the default
    thresholds, 4,466 with the collector off, 4,255 with generations 1 and 2
    pushed out**. The last is what this does, because it keeps generation 0
    reclaiming ordinary cyclic garbage instead of betting the whole process on
    nothing anywhere making a cycle.

    The trees themselves need no cycle collection at all: edges point from parent
    to child and nothing points back, so plain reference counting frees a whole
    discarded subtree the moment its root is dropped.
    """
    before = gc.get_threshold()
    gc.set_threshold(before[0], 1000, 1000)
    try:
        yield
    finally:
        gc.set_threshold(*before)
        gc.collect()


def resolve_selfplay_device(name: str = "auto") -> str:
    """``"auto"`` means MPS when this Mac has it, CPU everywhere else."""
    if name and name != "auto":
        return name
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():  # pragma: no cover - not this Mac
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------- evaluator
class BatchEvaluator:
    """Evaluate a list of positions in one forward pass.

    The single-position :class:`~ludometer.train.net.NetEvaluator` contract, only
    plural: in go ``(state, legal)`` pairs, out come ``(priors, value)`` — or
    ``(priors, value, margin)`` from a margin-head net — one per input, in order.

    ``max_batch`` (0 = unlimited) splits an over-large call into several passes,
    which is the knob that keeps MPS memory bounded when many games all want a
    leaf at once.
    """

    def __init__(self, net: Any, device: str = "cpu", max_batch: int = 0) -> None:
        import torch

        self.torch = torch
        self.net = net
        self.device = torch.device(device)
        self.net.to(self.device)
        self.net.eval()
        self.has_margin = bool(getattr(net, "has_margin", False))
        self.max_batch = max(0, int(max_batch))
        self.input_size = int(net.config.input_size)
        self.calls = 0  # forward passes
        self.rows = 0  # positions pushed through them
        self._buf = np.zeros((0, self.input_size), dtype=np.float32)

    def _staging(self, n: int) -> np.ndarray:
        """A reusable ``(n, input_size)`` float32 block (grown, never shrunk)."""
        if len(self._buf) < n:
            self._buf = np.zeros((n, self.input_size), dtype=np.float32)
        return self._buf[:n]

    def evaluate(
        self, states: Sequence[AzulState], legals: Sequence[Sequence[int]]
    ) -> list[tuple[np.ndarray, float] | tuple[np.ndarray, float, float]]:
        n = len(states)
        if n == 0:
            return []
        rows = self._staging(n)
        for i, state in enumerate(states):
            rows[i] = state.encode()
        chunk = self.max_batch or n
        parts: list[np.ndarray] = []
        torch = self.torch
        with torch.inference_mode():
            for start in range(0, n, chunk):
                block = rows[start : start + chunk]
                x = torch.from_numpy(block).to(self.device)
                logits, value, margin = self.net.forward_heads(x)
                # One readback, not three. Every device->host copy is a full
                # synchronisation on MPS (~1-2 ms of pure latency each, and this
                # loop cannot descend again until the values are home), so the
                # three head outputs are glued into one [B, 182] tensor on the
                # device and fetched in a single transfer.
                if margin is None:
                    tail = value.unsqueeze(1)
                else:
                    tail = torch.stack((value, margin), dim=1)
                parts.append(
                    torch.cat((logits, tail), dim=1).to("cpu", copy=True).numpy()
                )
                self.calls += 1
                self.rows += len(block)
        packed = parts[0] if len(parts) == 1 else np.concatenate(parts)
        action_space = packed.shape[1] - (2 if self.has_margin else 1)
        all_logits = packed[:, :action_space]
        all_values = packed[:, action_space]
        all_margins = packed[:, action_space + 1] if self.has_margin else None
        out: list[Any] = []
        for i in range(n):
            legal = legals[i]
            value = float(all_values[i])
            margin = 0.0 if all_margins is None else float(all_margins[i])
            if not len(legal):  # pragma: no cover - a searched node always has one
                priors = np.zeros(0, dtype=np.float32)
            else:
                # Softmax over the legal logits only — identical to masking the
                # rest to -inf first, and this is the hot loop.
                sel = all_logits[i][np.asarray(legal, dtype=np.int64)]
                sel = sel - sel.max()
                np.exp(sel, out=sel)
                priors = sel / sel.sum()
            out.append((priors, value, margin) if self.has_margin else (priors, value))
        return out

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        self.net.load_numpy_state_dict(weights)
        self.net.to(self.device)
        self.net.eval()


# -------------------------------------------------------------------- game slots
@dataclass
class _Slot:
    """One concurrent game: its state, its tree and the trajectory so far."""

    seed: int
    state: AzulState
    mcts: MCTS
    started: float
    states: list[np.ndarray] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)
    players: list[int] = field(default_factory=list)
    policy_mask: list[float] = field(default_factory=list)
    move: int = 0
    searching: bool = False
    pending: int = 0  # leaves handed to the current forward pass
    full: bool = True  # is the search in flight the deep one? (see `pcr`)
    schedule: Any = None  # per-game RNG for the playout-cap draw


class BatchedSelfPlay:
    """``games`` self-play games in one process, batched into one evaluator.

    Same API as :class:`~ludometer.train.selfplay.SelfPlayPool` — ``start``,
    ``set_weights``, ``play``, ``close`` — so the trainer cannot tell the two
    apart.
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
    ) -> None:
        self.tick = float(tick)  # seconds between progress/should_stop polls
        self.net_config = net_config
        self.config = config
        self.games = max(1, int(games))
        self.device = resolve_selfplay_device(device)
        self.max_batch = int(max_batch)
        # Per-tree ceiling on one gather, so a single game with a big
        # `search_batch` cannot crowd the others out of the tensor.
        self.leaf_cap = int(leaf_cap) or max(1, config.mcts.search_batch)
        self.workers = 1
        self.net = make_net(net_config)
        self.net.eval()
        self.evaluator = BatchEvaluator(
            self.net, device=self.device, max_batch=self.max_batch
        )
        self.batches = 0
        self.positions = 0
        # Split of the wall clock, because which side of it is binding is the
        # whole design question here (the descents are Python, the forward
        # passes are the GPU) and it changes with net size and `games`.
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
        """Play ``n_games`` (seeds ``seed_start ...``) and collect the records.

        Games start as slots free up, so the batch stays full until the last few
        games drain; ``progress(done, total)`` fires per finished game and
        ``should_stop()`` abandons the games still in flight (they are simply
        dropped — a half-played game has no target to learn from).
        """
        if n_games <= 0:
            return []
        with _relaxed_gc():
            return self._play(n_games, seed_start, progress, should_stop, on_record)

    def _play(
        self,
        n_games: int,
        seed_start: int,
        progress: Any,
        should_stop: Any,
        on_record: Callable[[GameRecord], None] | None,
    ) -> list[GameRecord]:
        out: list[GameRecord] = []
        slots: list[_Slot] = []
        started = 0
        stop = False
        # With G concurrent games nothing finishes for the first minute or two,
        # so the caller's heartbeat cannot ride on completions alone.
        next_tick = time.monotonic() + self.tick
        while len(out) < n_games and not stop:
            now = time.monotonic()
            if now >= next_tick:
                next_tick = now + self.tick
                if progress is not None:
                    progress(len(out), n_games)
                if should_stop is not None and should_stop():
                    break
            while len(slots) < self.games and started < n_games:
                slots.append(self._new_slot(seed_start + started))
                started += 1
            live: list[_Slot] = []
            for slot in slots:
                record = self._pump(slot)
                if record is None:
                    live.append(slot)
                    continue
                out.append(record)
                if on_record is not None:
                    on_record(record)
                if progress is not None:
                    progress(len(out), n_games)
                if should_stop is not None and should_stop():
                    stop = True
            slots = live
            if stop or len(out) >= n_games:
                break
            if not slots:
                # Every slot finished at once and there are games left to start:
                # go round again so the refill at the top can seat them.
                if started >= n_games:  # pragma: no cover - defensive
                    break
                continue
            t_search = time.perf_counter()
            requests: list[LeafRequest] = []
            for slot in slots:
                gathered = slot.mcts.leaf_requests(self.leaf_cap)
                slot.pending = len(gathered)
                requests.extend(gathered)
            if not requests:  # pragma: no cover - defensive: every slot idle
                continue
            t_eval = time.perf_counter()
            self.search_seconds += t_eval - t_search
            results = self.evaluator.evaluate(
                [r.node.state for r in requests], [r.node.legal for r in requests]
            )
            t_back = time.perf_counter()
            self.eval_seconds += t_back - t_eval
            self.batches += 1
            self.positions += len(requests)
            at = 0
            for slot in slots:
                if slot.pending:
                    slot.mcts.apply_leaves(results[at : at + slot.pending])
                    at += slot.pending
            self.search_seconds += time.perf_counter() - t_back
        return out

    # ------------------------------------------------------------------ guts
    def _new_slot(self, seed: int) -> _Slot:
        """A fresh game with its own RNGs — the seeds the sequential path uses."""
        return _Slot(
            seed=int(seed),
            state=get_game(self.config.game).new_game(seed),
            mcts=MCTS(
                self.evaluator,
                self.config.mcts,
                seed=(seed * 2 + 1) & 0x7FFFFFFF,
                add_noise=True,
            ),
            started=time.perf_counter(),
            schedule=pcr_rng(seed),
        )

    def _pump(self, slot: _Slot) -> GameRecord | None:
        """Push a game as far as it can go without the net; ``None`` if it needs it.

        The move loop of :func:`~ludometer.train.selfplay.play_selfplay_game`, cut
        open at the one point where it would have blocked on an evaluation.
        """
        config = self.config
        while True:
            if slot.searching:
                if not slot.mcts.search_done():
                    return None  # waiting for the current batch
                self._play_searched_move(slot)
                slot.searching = False
                continue
            state = slot.state
            if state.is_terminal or slot.move >= config.max_moves:
                return self._finish(slot)
            legal = state.legal_actions()
            slot.states.append(state.encode())
            slot.players.append(state.current_player)
            if len(legal) == 1:
                policy = np.zeros(state.ACTION_SPACE, dtype=np.float32)
                policy[legal[0]] = 1.0
                slot.policies.append(policy)
                slot.policy_mask.append(1.0)
                state.apply(legal[0])
                slot.mcts.advance(legal[0])
                slot.move += 1
                continue
            # Same draw, same stream, same point in the move loop as the
            # sequential path (ludometer.train.selfplay.play_selfplay_game).
            sims, slot.full = pcr_sims(config, slot.schedule)
            slot.mcts.start_search(state, add_noise=slot.full, sims=sims)
            slot.searching = True
            if slot.mcts.search_done():  # a reused root already at the budget
                continue
            return None

    def _play_searched_move(self, slot: _Slot) -> None:
        """Close the search, record the target, play the move."""
        config = self.config
        result = slot.mcts.finish_search()
        slot.policies.append(
            result.policy
            if slot.full
            else np.zeros(slot.state.ACTION_SPACE, dtype=np.float32)
        )
        slot.policy_mask.append(1.0 if slot.full else 0.0)
        # Two deterministic policies can keep a game going forever (nobody ever
        # completes a pattern line): past `stall_rounds` we sample again.
        stalling = slot.state.round_index >= config.stall_rounds
        explore = slot.move < config.temp_moves or stalling
        action = select_play_action(
            result,
            config.temperature if explore else 0.0,
            slot.mcts.rng,
            eps=config.mcts.decisive_eps,
            min_visit_frac=config.mcts.decisive_min_visit_frac,
            stalling=stalling,
        )
        slot.state.apply(action)
        slot.mcts.advance(action)
        slot.move += 1

    def _finish(self, slot: _Slot) -> GameRecord:
        state = slot.state
        truncated = not state.is_terminal
        outcome = float(state.outcome() or 0.0)  # a truncated game counts as a draw
        score_diff = state.scores[0] - state.scores[1]
        v0 = value_target(outcome, score_diff, self.config)
        values = np.array(
            [v0 if p == 0 else -v0 for p in slot.players], dtype=np.float32
        )
        return GameRecord(
            states=np.asarray(slot.states, dtype=np.float32),
            policies=np.asarray(slot.policies, dtype=np.float32),
            values=values,
            margins=margin_targets(score_diff, slot.players),
            aux=aux_targets(state, slot.players),
            policy_mask=np.asarray(slot.policy_mask, dtype=np.float32),
            outcome=outcome,
            scores=(int(state.scores[0]), int(state.scores[1])),
            moves=slot.move,
            rounds=state.round_index + 1,
            seed=slot.seed,
            evals=slot.mcts.evals,
            duration=time.perf_counter() - slot.started,
            truncated=truncated,
        )


# ------------------------------------------------------------------- processes
def _batched_worker_loop(
    worker_id: int,
    net_config: Any,
    config: SelfPlayConfig,
    cmd_q: Any,
    result_q: Any,
    games: int,
    device: str,
    max_batch: int,
) -> None:  # pragma: no cover - runs in a child process
    """Worker entry point: one batched engine, fed blocks of seeds."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent orchestrates shutdown
    import torch

    # The descents are the CPU cost here and they are single-threaded Python; the
    # forward passes go to the GPU. Letting torch spin up 6 CPU threads per driver
    # process would only make the drivers fight each other.
    torch.set_num_threads(1)
    engine = BatchedSelfPlay(
        net_config, config, games=games, device=device, max_batch=max_batch
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
                # Seeds are contiguous per worker, so one `play` call covers the
                # whole block and every record is streamed back as it finishes.
                engine.play(
                    len(seeds), seeds[0], on_record=lambda rec: result_q.put(rec)
                )
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                result_q.put(("error", worker_id, f"{type(exc).__name__}: {exc}"))


class BatchedSelfPlayPool(SelfPlayPool):
    """``workers`` batched drivers, each running ``games`` concurrent trees.

    The pool plumbing (persistent processes, weight broadcast, streamed records,
    heartbeat-friendly polling) is the one
    :class:`~ludometer.train.selfplay.SelfPlayPool` already had; only the worker
    body and the seed hand-out differ. Seeds go out in **contiguous blocks**, one
    block per worker, because a batched driver wants a whole block at once —
    round-robin single seeds would give each driver one game at a time and
    there would be nothing to batch.
    """

    worker_main = staticmethod(_batched_worker_loop)

    def __init__(
        self,
        net_config: Any,
        config: SelfPlayConfig,
        workers: int = 4,
        games: int = 32,
        device: str = "auto",
        max_batch: int = 0,
        poll: float = 1.0,
    ) -> None:
        super().__init__(net_config, config, workers=workers, poll=poll)
        self.games = max(1, int(games))
        self.device = device
        self.worker_extra = (self.games, device, int(max_batch))

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
