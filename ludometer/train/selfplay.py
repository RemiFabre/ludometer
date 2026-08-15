"""Self-play: play games with MCTS + net and return training trajectories.

A game produces one :class:`GameRecord` — parallel numpy arrays of encoded
states, MCTS visit policies and value targets (game outcome seen from the player
to move). Move selection samples from the visit counts for the first
``temp_moves`` moves and takes the arg-max afterwards; the *targets* are always
the raw visit distribution.

The value target is the win/draw/loss outcome blended with a little of the final
score margin: ``(1 - w) * outcome + w * tanh(score_diff / SCORE_SCALE)`` with
``w = value_score_weight``. ``w = 0`` gives the textbook AlphaZero target; a small
positive ``w`` mattered in run1-run3 because a weak early policy produces piles of
0-0 games, and a pure win/loss target on a drawn game carries no gradient at all
(the margin term keeps the sign of the winner, it only grades the size).

**run4 sets ``w = 0``**: the score margin has its own head and its own target
(:func:`margin_targets`, ``tanh(score_diff / 20)`` per seat), so the value head goes
back to being a pure win/draw/loss estimate and the two questions stop fighting
over one output. Every record carries the margin array whatever ``w`` is — it costs
one tanh per game — so a buffer written by this module is always usable by a
margin-head net.

Parallelism: ``N`` ``spawn``-ed worker processes, each with its own CPU copy of
the net and ``torch.set_num_threads(1)`` (8 single-threaded workers beat 1
multi-threaded one for these tiny MLPs). The parent drives them over queues:

    parent -> worker  ("weights", {name: numpy array})   once per sync
    parent -> worker  ("play", [seed, ...])              one batch of games
    worker -> parent  GameRecord                          streamed, one per game

Streaming the records back one at a time is what lets the trainer keep its
``status.json`` heartbeat alive during a long self-play batch.
"""

from __future__ import annotations

import contextlib
import math
import multiprocessing as mp
import queue
import signal
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

from ludometer.azul.engine import ACTION_SPACE, AzulState
from ludometer.train.mcts import (
    MARGIN_SCALE,
    MAX_GAME_MOVES,
    MCTS,
    STALL_ROUNDS,
    MCTSConfig,
    margin_target,
    select_play_action,
)
from ludometer.train.net import NetEvaluator, make_net

# Points that saturate the score margin — one constant for the legacy blended
# value and for run4's margin head, so the two never drift apart.
SCORE_SCALE = MARGIN_SCALE

__all__ = [
    "GameRecord",
    "InlineSelfPlay",
    "SelfPlayConfig",
    "SelfPlayPool",
    "make_selfplay",
    "margin_targets",
    "play_selfplay_game",
    "value_target",
]


@dataclass
class GameRecord:
    """One self-play game: trajectory plus a few diagnostics."""

    states: np.ndarray  # (T, 182) float32
    policies: np.ndarray  # (T, 180) float32
    values: np.ndarray  # (T,) float32, player-to-move perspective
    margins: np.ndarray  # (T,) float32, tanh(score diff / 20), same perspective
    outcome: float  # +1 player 0 won, -1 player 1 won, 0 draw
    scores: tuple[int, int]
    moves: int
    rounds: int
    seed: int
    evals: int = 0
    duration: float = 0.0
    truncated: bool = False  # hit max_moves without finishing (scored as a draw)

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class SelfPlayConfig:
    """Everything a worker needs to play a game (picklable)."""

    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    temp_moves: int = 12
    temperature: float = 1.0
    stall_rounds: int = STALL_ROUNDS
    max_moves: int = MAX_GAME_MOVES
    value_score_weight: float = 0.15

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SelfPlayConfig:
        data = data or {}
        return cls(
            mcts=MCTSConfig.from_dict(data),
            temp_moves=int(data.get("temp_moves", 12)),
            temperature=float(data.get("temperature", 1.0)),
            stall_rounds=int(data.get("stall_rounds", STALL_ROUNDS)),
            max_moves=int(data.get("max_game_moves", MAX_GAME_MOVES)),
            value_score_weight=float(data.get("value_score_weight", 0.15)),
        )


def value_target(outcome: float, score_diff: int, config: SelfPlayConfig) -> float:
    """Player-0 value target: the outcome, nudged by the final score margin."""
    w = config.value_score_weight
    if w <= 0.0:
        return outcome
    margin = math.tanh(score_diff / SCORE_SCALE)
    return (1.0 - w) * outcome + w * margin


def margin_targets(score_diff: int, players: Sequence[int]) -> np.ndarray:
    """Per-position margin targets: ``tanh(diff / 20)``, flipped per seat."""
    m0 = margin_target(score_diff)
    return np.array([m0 if p == 0 else -m0 for p in players], dtype=np.float32)


def play_selfplay_game(evaluator: Any, seed: int, config: SelfPlayConfig) -> GameRecord:
    """Play one full game against itself; never mutates anything shared."""
    started = time.perf_counter()
    state = AzulState.new_game(seed=seed)
    mcts = MCTS(
        evaluator, config.mcts, seed=(seed * 2 + 1) & 0x7FFFFFFF, add_noise=True
    )
    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    players: list[int] = []
    move = 0
    while not state.is_terminal and move < config.max_moves:
        legal = state.legal_actions()
        states.append(state.encode())
        players.append(state.current_player)
        if len(legal) == 1:
            policy = np.zeros(ACTION_SPACE, dtype=np.float32)
            policy[legal[0]] = 1.0
            action = legal[0]
        else:
            result = mcts.search(state)
            policy = result.policy
            # Two deterministic policies can keep a game going forever (nobody
            # ever completes a pattern line, so no wall tile is ever placed):
            # past `stall_rounds` we sample again, which breaks the loop.
            explore = (
                move < config.temp_moves or state.round_index >= config.stall_rounds
            )
            # Temperature 0 with a margin-head net is the decisive pick: same
            # visit counts (so `policy`, the training target, is untouched), but
            # the move played is the biggest-margin one among the equally winning
            # ones. See ludometer.train.mcts, "Decisive play".
            action = select_play_action(
                result,
                config.temperature if explore else 0.0,
                mcts.rng,
                eps=config.mcts.decisive_eps,
                min_visit_frac=config.mcts.decisive_min_visit_frac,
            )
        policies.append(policy)
        state.apply(action)
        # Keep the chosen child's subtree for the next move (no-op unless
        # `mcts.tree_reuse` is on). One search drives both seats here, so one
        # `advance` per played move is exactly one ply down the tree.
        mcts.advance(action)
        move += 1

    truncated = not state.is_terminal
    outcome = float(state.outcome() or 0.0)  # a truncated game counts as a draw
    score_diff = state.scores[0] - state.scores[1]
    v0 = value_target(outcome, score_diff, config)
    values = np.array([v0 if p == 0 else -v0 for p in players], dtype=np.float32)
    return GameRecord(
        states=np.asarray(states, dtype=np.float32),
        policies=np.asarray(policies, dtype=np.float32),
        values=values,
        margins=margin_targets(score_diff, players),
        outcome=outcome,
        scores=(int(state.scores[0]), int(state.scores[1])),
        moves=move,
        rounds=state.round_index + 1,
        seed=int(seed),
        evals=mcts.evals,
        duration=time.perf_counter() - started,
        truncated=truncated,
    )


# ------------------------------------------------------------------ in-process
class InlineSelfPlay:
    """Single-process self-play with the same API as :class:`SelfPlayPool`."""

    def __init__(self, net_config: Any, config: SelfPlayConfig) -> None:
        self.net = make_net(net_config)
        self.net.eval()
        self.evaluator = NetEvaluator(self.net, device="cpu")
        self.config = config
        self.workers = 1

    def start(self, weights: dict[str, np.ndarray] | None = None) -> None:
        if weights is not None:
            self.set_weights(weights)

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        self.net.load_numpy_state_dict(weights)
        self.net.eval()

    def play(
        self,
        n_games: int,
        seed_start: int,
        progress: Any = None,
        should_stop: Any = None,
    ) -> list[GameRecord]:
        out: list[GameRecord] = []
        for i in range(n_games):
            if should_stop is not None and should_stop():
                break
            out.append(play_selfplay_game(self.evaluator, seed_start + i, self.config))
            if progress is not None:
                progress(len(out), n_games)
        return out

    def close(self) -> None:
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ------------------------------------------------------------------- processes
def _worker_loop(
    worker_id: int,
    net_config: Any,
    config: SelfPlayConfig,
    cmd_q: Any,
    result_q: Any,
) -> None:  # pragma: no cover - runs in a child process
    """Worker entry point: build a CPU net, then serve commands until stopped."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent orchestrates shutdown
    import torch

    torch.set_num_threads(1)
    net = make_net(net_config)
    net.eval()
    evaluator = NetEvaluator(net, device="cpu")
    while True:
        message = cmd_q.get()
        kind = message[0]
        if kind == "stop":
            break
        if kind == "weights":
            net.load_numpy_state_dict(message[1])
            net.eval()
            continue
        if kind == "play":
            for seed in message[1]:
                try:
                    record = play_selfplay_game(evaluator, seed, config)
                except Exception as exc:  # noqa: BLE001 - reported to the parent
                    result_q.put(("error", worker_id, f"{type(exc).__name__}: {exc}"))
                    continue
                result_q.put(record)


class SelfPlayPool:
    """A pool of persistent self-play workers with refreshable weights."""

    def __init__(
        self,
        net_config: Any,
        config: SelfPlayConfig,
        workers: int = 8,
        poll: float = 1.0,
    ) -> None:
        self.net_config = net_config
        self.config = config
        self.workers = max(1, int(workers))
        self.poll = poll
        self._ctx = mp.get_context("spawn")
        self._cmd_qs: list[Any] = []
        self._result_q: Any = None
        self._procs: list[Any] = []

    # ------------------------------------------------------------------ setup
    def start(self, weights: dict[str, np.ndarray] | None = None) -> None:
        if self._procs:
            raise RuntimeError("pool already started")
        self._result_q = self._ctx.Queue()
        for wid in range(self.workers):
            cmd_q = self._ctx.Queue()
            proc = self._ctx.Process(
                target=_worker_loop,
                args=(wid, self.net_config, self.config, cmd_q, self._result_q),
                daemon=True,
                name=f"selfplay-{wid}",
            )
            proc.start()
            self._cmd_qs.append(cmd_q)
            self._procs.append(proc)
        if weights is not None:
            self.set_weights(weights)

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Broadcast new weights; each worker applies them before its next game."""
        for cmd_q in self._cmd_qs:
            cmd_q.put(("weights", weights))

    # ------------------------------------------------------------------- play
    def play(
        self,
        n_games: int,
        seed_start: int,
        progress: Any = None,
        should_stop: Any = None,
    ) -> list[GameRecord]:
        """Play ``n_games`` (seeds ``seed_start ...``) and collect the records.

        ``progress(done, total)`` is called after every finished game and on every
        idle poll tick, which is how the trainer heartbeats through a long batch.
        ``should_stop()`` aborts early and returns the games finished so far.
        """
        if not self._procs:
            raise RuntimeError("pool not started")
        if n_games <= 0:
            return []
        for i in range(n_games):
            self._cmd_qs[i % self.workers].put(("play", [seed_start + i]))
        out: list[GameRecord] = []
        while len(out) < n_games:
            try:
                item = self._result_q.get(timeout=self.poll)
            except queue.Empty:
                self._check_alive()
                if progress is not None:
                    progress(len(out), n_games)
                if should_stop is not None and should_stop():
                    break
                continue
            if isinstance(item, tuple):  # ("error", worker_id, detail)
                raise RuntimeError(  # noqa: TRY004 - a report, not a type problem
                    f"self-play worker {item[1]} failed: {item[2]}"
                )
            out.append(item)
            if progress is not None:
                progress(len(out), n_games)
            if should_stop is not None and should_stop():
                break
        return out

    def _check_alive(self) -> None:
        for proc in self._procs:
            if proc.exitcode is not None:
                raise RuntimeError(
                    f"self-play worker {proc.name} exited with {proc.exitcode}"
                )

    # ---------------------------------------------------------------- shutdown
    def close(self, timeout: float = 5.0) -> None:
        for cmd_q in self._cmd_qs:
            with contextlib.suppress(OSError, ValueError):  # queue already closed
                cmd_q.put(("stop",))
        deadline = time.monotonic() + timeout
        # drain the result queue so the children can exit cleanly
        while self._result_q is not None and time.monotonic() < deadline:
            try:
                self._result_q.get_nowait()
            except queue.Empty:
                break
            except (OSError, ValueError):  # pragma: no cover - queue closed
                break
        for proc in self._procs:
            proc.join(timeout=max(0.1, deadline - time.monotonic()))
            if proc.is_alive():  # pragma: no cover - stubborn worker
                proc.terminate()
                proc.join(timeout=1.0)
        self._procs = []
        self._cmd_qs = []
        self._result_q = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def make_selfplay(
    net_config: Any, config: SelfPlayConfig, workers: int
) -> SelfPlayPool | InlineSelfPlay:
    """``workers <= 1`` runs in-process (tests, debugging); otherwise a pool."""
    if workers <= 1:
        return InlineSelfPlay(net_config, config)
    return SelfPlayPool(net_config, config, workers=workers)
