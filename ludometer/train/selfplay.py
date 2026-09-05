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

**run6 adds two more columns**, on the same "always recorded, cost nothing"
principle:

* ``aux`` — the 30 bits of :meth:`~ludometer.azul.engine.AzulState.wall_summary`
  for both players' final walls, flipped per seat like the margin
  (:func:`aux_targets`). This is the long-horizon label the strategic heads in
  :mod:`ludometer.train.net2` learn from, and it is free: it is read once per
  game off the finished board;
* ``policy_mask`` — 1 where the visit distribution in ``policies`` came from a
  full search, 0 where it did not.

Playout-cap randomization (run6, ``pcr``)
-----------------------------------------
KataGo's trick, and the reason run6 can afford 1024-simulation policy targets.
Each move independently draws "full" with probability ``pcr_full_prob``:

* a **full** move searches ``pcr_full_sims`` simulations with root Dirichlet noise
  and records its visit distribution as a policy target (mask 1);
* a **cheap** move searches ``pcr_cheap_sims``, with **no root noise**, and enters
  the buffer with a zeroed policy and mask 0. Its value, margin and aux labels are
  every bit as good as a full move's — those come from the end of the game, not
  from the search — so the position is not wasted, only its policy is.

The expected cost per move is ``p * full + (1 - p) * cheap`` (run6: 0.25 * 1024 +
0.75 * 256 = 448, *below* run5's flat 512), so the game volume is not paid for out
of the policy target's depth. The draw uses a per-game RNG of its own, seeded from
the game seed, so a game's schedule is reproducible and independent both of the
search's RNG stream (a run with ``pcr`` off is bit-identical to run5) and of which
other games shared its batches.

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
import random
import signal
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np

from ludometer.azul.engine import AzulState
from ludometer.games import DEFAULT_GAME, get_game
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
    "PCR_RNG_SALT",
    "GameRecord",
    "InlineSelfPlay",
    "SelfPlayConfig",
    "SelfPlayPool",
    "aux_targets",
    "make_selfplay",
    "margin_targets",
    "pcr_rng",
    "pcr_sims",
    "play_selfplay_game",
    "value_target",
]

#: mixed into the game seed for the playout-cap draw, so the schedule has its own
#: RNG stream and turning `pcr` on does not shift the search's or the sampler's.
PCR_RNG_SALT = 0x9E3779B9


@dataclass
class GameRecord:
    """One self-play game: trajectory plus a few diagnostics."""

    states: np.ndarray  # (T, 182) float32
    policies: np.ndarray  # (T, 180) float32
    values: np.ndarray  # (T,) float32, player-to-move perspective
    margins: np.ndarray  # (T,) float32, tanh(score diff / 20), same perspective
    aux: np.ndarray  # (T, 30) uint8, final-wall bits, same perspective
    policy_mask: np.ndarray  # (T,) float32, 1 where `policies` is a real target
    outcome: float  # +1 player 0 won, -1 player 1 won, 0 draw
    scores: tuple[int, int]
    moves: int
    rounds: int
    seed: int
    #: moves with more than one legal action — the cross-game unit of practice
    #: (docs/NEXT_GAMES.md §4). ~60% of Uno turns are forced; Azul has ~none.
    decisions: int = 0
    evals: int = 0
    duration: float = 0.0
    truncated: bool = False  # hit max_moves without finishing (scored as a draw)
    #: The search's own root value estimate per position (player-to-move frame),
    #: and 1 where a real search produced it (forced moves and cheap PCR searches
    #: carry 0). A second value target next to the game outcome: less noisy,
    #: slightly biased — see ``TrainConfig.value_search_weight``.
    search_values: np.ndarray | None = None  # (T,) float32
    search_mask: np.ndarray | None = None  # (T,) float32

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class SelfPlayConfig:
    """Everything a worker needs to play a game (picklable)."""

    game: str = DEFAULT_GAME
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    temp_moves: int = 12
    temperature: float = 1.0
    stall_rounds: int = STALL_ROUNDS
    max_moves: int = MAX_GAME_MOVES
    value_score_weight: float = 0.15
    # Playout-cap randomization (see the module docstring). `pcr_full_prob <= 0`
    # is off, which is what every pre-run6 config says by omission: the search
    # then runs `mcts.sims` on every move, with noise, exactly as before.
    pcr_full_sims: int = 0
    pcr_cheap_sims: int = 0
    pcr_full_prob: float = 0.0

    @property
    def pcr(self) -> bool:
        """Is playout-cap randomization on for this run?"""
        return self.pcr_full_prob > 0.0 and self.pcr_cheap_sims > 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SelfPlayConfig:
        data = data or {}
        pcr = dict(data.get("pcr") or {})
        return cls(
            game=str(data.get("game", DEFAULT_GAME)),
            mcts=MCTSConfig.from_dict(data),
            temp_moves=int(data.get("temp_moves", 12)),
            temperature=float(data.get("temperature", 1.0)),
            stall_rounds=int(data.get("stall_rounds", STALL_ROUNDS)),
            max_moves=int(data.get("max_game_moves", MAX_GAME_MOVES)),
            value_score_weight=float(data.get("value_score_weight", 0.15)),
            pcr_full_sims=int(pcr.get("full_sims", 0)),
            pcr_cheap_sims=int(pcr.get("cheap_sims", 0)),
            pcr_full_prob=float(pcr.get("full_prob", 0.0)),
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


def aux_targets(state: AzulState, players: Sequence[int]) -> np.ndarray:
    """Per-position final-wall bits ``(T, 30)`` in the player-to-move frame.

    ``state`` is the *finished* game. Row layout is
    ``[me rows 5 | me cols 5 | me colours 5 | them rows 5 | them cols 5 | them
    colours 5]``, "me" being whoever is to move in that position — the same
    convention :meth:`~ludometer.azul.engine.AzulState.encode` uses, so the net
    never has to work out which wall it is looking at.
    """
    walls = [state.wall_summary(0), state.wall_summary(1)]
    seat = [
        np.array(walls[p] + walls[1 - p], dtype=np.uint8) for p in (0, 1)
    ]  # only two distinct rows exist
    if not len(players):
        return np.zeros((0, len(seat[0])), dtype=np.uint8)
    return np.stack([seat[p] for p in players])


def pcr_rng(seed: int) -> random.Random:
    """The playout-cap draw's own RNG for a game — see :data:`PCR_RNG_SALT`."""
    return random.Random((int(seed) * 2 + 1) ^ PCR_RNG_SALT)


def pcr_sims(config: SelfPlayConfig, rng: random.Random) -> tuple[int | None, bool]:
    """``(simulation budget or None, is this a full search)`` for one move."""
    if not config.pcr:
        return None, True
    full = rng.random() < config.pcr_full_prob
    if full:
        return (config.pcr_full_sims or config.mcts.sims), True
    return config.pcr_cheap_sims, False


def play_selfplay_game(evaluator: Any, seed: int, config: SelfPlayConfig) -> GameRecord:
    """Play one full game against itself; never mutates anything shared."""
    started = time.perf_counter()
    state = get_game(config.game).new_game(seed)
    mcts = MCTS(
        evaluator, config.mcts, seed=(seed * 2 + 1) & 0x7FFFFFFF, add_noise=True
    )
    states: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    players: list[int] = []
    policy_mask: list[float] = []
    search_values: list[float] = []
    search_mask: list[float] = []
    schedule = pcr_rng(seed)
    move = 0
    decisions = 0
    while not state.is_terminal and move < config.max_moves:
        legal = state.legal_actions()
        states.append(state.encode())
        players.append(state.current_player)
        if len(legal) == 1:
            policy = np.zeros(state.ACTION_SPACE, dtype=np.float32)
            policy[legal[0]] = 1.0
            policy_mask.append(1.0)
            search_values.append(0.0)
            search_mask.append(0.0)
            action = legal[0]
        else:
            decisions += 1
            # Playout-cap randomization: budget and root noise are drawn per move
            # and a cheap move's visit distribution is not a training target.
            sims, full = pcr_sims(config, schedule)
            result = mcts.search(state, add_noise=full, sims=sims)
            policy = (
                result.policy
                if full
                else np.zeros(state.ACTION_SPACE, dtype=np.float32)
            )
            policy_mask.append(1.0 if full else 0.0)
            search_values.append(float(result.value) if full else 0.0)
            search_mask.append(1.0 if full else 0.0)
            # Two deterministic policies can keep a game going forever (nobody
            # ever completes a pattern line, so no wall tile is ever placed):
            # past `stall_rounds` we sample again, which breaks the loop.
            stalling = state.round_index >= config.stall_rounds
            explore = move < config.temp_moves or stalling
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
                stalling=stalling,
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
        aux=aux_targets(state, players),
        policy_mask=np.asarray(policy_mask, dtype=np.float32),
        outcome=outcome,
        scores=(int(state.scores[0]), int(state.scores[1])),
        moves=move,
        rounds=state.round_index + 1,
        seed=int(seed),
        decisions=decisions,
        evals=mcts.evals,
        duration=time.perf_counter() - started,
        truncated=truncated,
        search_values=np.asarray(search_values, dtype=np.float32),
        search_mask=np.asarray(search_mask, dtype=np.float32),
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

    #: entry point each worker process runs; overridden by the batched pool
    worker_main: Any = staticmethod(_worker_loop)
    #: extra positional arguments appended to the worker's argument list
    worker_extra: tuple[Any, ...] = ()

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
        #: positions evaluated so far by each worker in the current/last block
        #: (batched workers report it on their tick; sequential ones never do)
        self.worker_positions: dict[int, int] = {}

    # ------------------------------------------------------------------ setup
    def start(self, weights: dict[str, np.ndarray] | None = None) -> None:
        if self._procs:
            raise RuntimeError("pool already started")
        self._result_q = self._ctx.Queue()
        for wid in range(self.workers):
            cmd_q = self._ctx.Queue()
            proc = self._ctx.Process(
                target=self.worker_main,
                args=(
                    wid,
                    self.net_config,
                    self.config,
                    cmd_q,
                    self._result_q,
                    *self.worker_extra,
                ),
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
        self._dispatch(n_games, seed_start)
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
            if isinstance(item, tuple):
                if item[0] == "progress":  # ("progress", worker_id, positions)
                    self.worker_positions[int(item[1])] = int(item[2])
                    if progress is not None:
                        progress(len(out), n_games)
                    continue
                raise RuntimeError(f"self-play worker {item[1]} failed: {item[2]}")
            out.append(item)
            if progress is not None:
                progress(len(out), n_games)
            if should_stop is not None and should_stop():
                break
        return out

    def _dispatch(self, n_games: int, seed_start: int) -> None:
        """Hand the seeds out, one game per message, round robin."""
        for i in range(n_games):
            self._cmd_qs[i % self.workers].put(("play", [seed_start + i]))

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
    net_config: Any,
    config: SelfPlayConfig,
    workers: int,
    kind: str = "workers",
    games: int = 64,
    device: str = "auto",
    max_batch: int = 0,
    half: bool = False,
) -> Any:
    """The self-play engine a run asked for.

    ``kind="workers"`` is the run1-run4 path and the default for every old
    config: one game at a time per process, ``workers <= 1`` running in-process.
    ``kind="batched"`` is run5's — ``games`` concurrent trees per driver process,
    all of their leaves evaluated in one forward pass on ``device``.
    """
    if kind == "batched":
        from ludometer.train.selfplay_batched import (  # lazy: torch/MPS at import
            BatchedSelfPlay,
            BatchedSelfPlayPool,
        )

        if workers <= 1:
            return BatchedSelfPlay(
                net_config,
                config,
                games=games,
                device=device,
                max_batch=max_batch,
                half=half,
            )
        return BatchedSelfPlayPool(
            net_config,
            config,
            workers=workers,
            games=games,
            device=device,
            max_batch=max_batch,
            half=half,
        )
    if kind != "workers":
        raise ValueError(f"unknown selfplay engine {kind!r} (workers | batched)")
    if workers <= 1:
        return InlineSelfPlay(net_config, config)
    return SelfPlayPool(net_config, config, workers=workers)
