"""PUCT MCTS over cloned Azul states (see docs/DESIGN.md, "Training").

Search shape
------------
Standard AlphaZero PUCT: edge statistics ``(N, W, P)`` live in the parent, a
simulation walks down by maximising ``Q + c_puct * P * sqrt(Nparent) / (1 + N)``,
expands one leaf, evaluates it with the net and backs the value up.

Perspectives. Azul does **not** strictly alternate: the holder of the
first-player marker starts the next round, so the same player can move twice in a
row across a round boundary. All values are therefore propagated in *player 0's*
frame (``v0``) and converted per node with ``node.player``, which is read from
``state.current_player`` — nothing assumes alternation.

Chance nodes (round refills)
----------------------------
Inside a round Azul is deterministic; the only chance event is the refill that
happens when a move empties the last factory/center tile. Such an edge is a
chance node with astronomically many outcomes, so it is handled by **re-sampled
determinizations**:

* an edge is stochastic iff the move takes the last tiles of the round
  (``pool[color] == state.tiles_left``);
* each traversal of a stochastic edge clones the parent, reseeds the clone's RNG
  from a per-search counter, **reshuffles the clone's bag** and applies the move.
  Reshuffling matters twice over: the engine pre-shuffles its bag, so a plain
  clone would let the search peek at the exact next deal (bag *contents* are
  public, their order is not), and it makes each traversal an independent draw
  from the true refill distribution;
* the resulting child is stored in a per-edge dict keyed by the post-refill
  factory/center contents (that key determines bag and lid counts too, so it
  identifies the position exactly). Distinct outcomes are capped at
  ``chance_children``; past the cap a traversal reuses one of the stored
  determinizations uniformly at random, which keeps the subtree deep enough to be
  useful while the edge's ``Q`` stays an average over sampled refills.

Tree reuse (opt-in: ``MCTSConfig.tree_reuse``)
---------------------------------------------
With ``tree_reuse`` off (the default, and what run1/run2 did) every ``search``
starts from a fresh root. With it on, the caller plays a move and calls
:meth:`MCTS.advance`; the chosen child then becomes the next search root, so the
work already spent below it is kept:

* only a **deterministic** edge is followed. If the move triggered a refill the
  edge is a chance node (a dict of determinizations, none of which is *the*
  position the game actually dealt), so reuse is dropped — that is the safe
  boundary and the reason the flag never lets the search inherit a bag order it
  is not entitled to;
* a cheap fingerprint of the reused node's state is compared with the state the
  next ``search`` is given; anything unexpected falls back to a fresh root, so a
  caller that forgets an ``advance`` loses speed, never correctness;
* the budget is a **total**: the reused root already has ``N`` visits, so only
  ``sims - N`` new simulations are run. Same tree size per move, fewer network
  evaluations — that is where the throughput gain comes from;
* **Dirichlet noise is re-mixed at the new root**, and that is exact rather than
  approximate: noise is only ever applied to a root's own priors, so the reused
  child's priors are still the raw network priors and mixing fresh noise into
  them is identical to what a fresh search would have done. Nothing has to be
  un-mixed. Visit counts inherited from below the old root were produced under
  the old root's noise only through *which* child was explored, which is the
  intended effect of noise, not a bias in the target.

Only self-play uses it: one :class:`MCTS` drives both seats there, so advancing
one ply per move is exactly right. A tournament agent sees the position again
only after the opponent has replied, which would need a second descent, so
:class:`~ludometer.train.mcts_agent.MCTSAgent` deliberately leaves it off.

Decisive play (opt-in: a net with a margin head)
------------------------------------------------
A win/draw/loss value head is *indifferent* between winning by one point and
winning by forty, so once a game is decided the search plays whichever winning
move it happens to have visited most — technically correct, and it looks broken.

When the evaluator advertises ``has_margin`` (see
:class:`~ludometer.train.net.NetEvaluator`) every simulation backs up a **pair**:
the usual win value ``v`` and a score margin ``m = tanh(score_diff / 20)``, both in
player 0's frame and both converted per node. Terminal nodes contribute the real
final margin, not an estimate.

Two things stay untouched, on purpose:

* **PUCT is still driven by the win value alone.** The margin never steers the
  tree, so search effort is never spent buying points at the cost of a win, and
  the visit counts — i.e. the *policy targets* — are exactly what they would have
  been without the head;
* **the visit distribution is the policy target, unchanged.** The lexicographic
  pick below only decides which move is *played*. In late self-play that means a
  position can be labelled with a visit distribution that does not peak on the
  played move; that is intended (the target still says what search believed, the
  move says what a human would expect) and it is why the head does not need its
  own exploration machinery.

:func:`select_play_action` is the pick used whenever a move is played for real
(temperature-0 self-play moves, ``MCTSAgent.act``, the arena, the GUI): among the
root children that are *adequately visited* (at least ``decisive_min_visit_frac``
of the best child's visits) it takes the best win-Q, keeps everyone within
``decisive_eps`` of it, and plays the one with the highest margin-Q. Winning stays
lexicographically first; the margin is only ever a tie-break. Without a margin
head the function is exactly ``select_action(result.policy, ...)``, so run1/run2/
run3 checkpoints keep bit-identical behaviour.

Time budget (opt-in, GUI only)
------------------------------
``search(state, time_limit_s=...)`` keeps simulating until the wall clock runs
out, checking it every :data:`TIME_CHECK_EVERY` simulations, with
``config.sims`` acting as the upper bound. The default is ``None``: training and
the arena keep running exactly ``config.sims`` simulations, so results stay
reproducible.

Batched search (run5: :mod:`ludometer.train.selfplay_batched`)
--------------------------------------------------------------
:meth:`MCTS.search` owns its loop: it *calls* the evaluator, one position at a
time. That is exactly what makes a GPU useless to it — a Metal dispatch costs the
same ~1.7 ms whether it carries 1 position or 128 — so run5 turns the loop inside
out. The same tree, the same PUCT, the same chance handling, driven by four
methods the caller pumps instead of one method that blocks:

    mcts.start_search(state)
    while not mcts.search_done():
        leaves = mcts.leaf_requests(max_leaves)   # descend, collect
        mcts.apply_leaves(evaluate(leaves))       # one forward pass, backed up
    result = mcts.finish_search()

The caller can therefore interleave *many* trees and put every tree's leaves in
one tensor. See :mod:`ludometer.train.selfplay_batched` for the driver.

Two levels of batching compose, and they are not equally safe:

* **across games** — G independent trees contribute one leaf each per pass. Every
  tree still runs a strictly sequential search, so the search is *bit-identical*
  to :meth:`search` (the test suite asserts exactly that). This is the free win.
* **within one tree** (``search_batch > 1``) — several descents before one pass,
  each laying a **virtual loss** (``virtual_loss``, in the value's own [-1, 1]
  units) on the edges it walks so the next descent is pushed elsewhere; the loss
  is taken off again when the real value is backed up, which makes the
  bookkeeping exact rather than approximate. This one costs search quality, so it
  is off by default and ramped when on: the browser player measured a **flat**
  batch of 64 losing 3-17 to a batch-1 search at equal simulations, because with
  an empty tree virtual loss shoves all 64 descents down 64 different branches.
  The damage is a function of batch / tree, so the batch never exceeds
  ``root.n_visits / search_batch_ramp`` (16, the browser's rule) and starts at
  ``search_min_batch``.

Tree reuse composes with cross-game batching untouched: reuse is a property of
one tree between two moves, batching is a property of many trees within one
move, and neither reads the other's state.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ludometer.azul.engine import ACTION_SPACE, CENTER, AzulState

__all__ = [
    "MARGIN_SCALE",
    "MAX_GAME_MOVES",
    "MCTS",
    "STALL_ROUNDS",
    "TIME_CHECK_EVERY",
    "LeafRequest",
    "MCTSConfig",
    "Node",
    "RolloutEvaluator",
    "SearchResult",
    "UniformEvaluator",
    "decisive_action",
    "margin_target",
    "select_action",
    "select_play_action",
]

# Points that saturate the margin target: tanh(diff / 20) is ~0.76 at 20 points
# and ~0.96 at 40, so ordinary Azul margins land on the informative part of the
# curve instead of pinning the head at +/-1.
MARGIN_SCALE = 20.0


def margin_target(score_diff: float) -> float:
    """``tanh(score_diff / 20)`` — the margin head's target and MCTS's backup."""
    return math.tanh(score_diff / MARGIN_SCALE)


# A game where neither side ever completes a pattern line never terminates (no
# wall tile is placed, so no row is ever completed and tiles just cycle through
# the lid). Two arg-max players can fall into exactly that loop, so past
# `STALL_ROUNDS` rounds the caller re-introduces sampling, and `MAX_GAME_MOVES`
# is the hard backstop (a truncated self-play game is scored as a draw).
STALL_ROUNDS = 16
MAX_GAME_MOVES = 400

# Simulations between two wall-clock checks when a time budget is in force. A
# handful of sims is a fraction of a millisecond of over-run, and the check
# itself never shows up in a profile at this granularity.
TIME_CHECK_EVERY = 8

# An evaluator maps (state, legal actions) to (priors aligned with `legal`,
# value in [-1, 1] for the player to move) — or, with a margin head, to a
# 3-tuple that appends the margin, also in [-1, 1] and also for the player to
# move. An evaluator that returns 3-tuples sets `has_margin = True` on itself.
Evaluator = Callable[[AzulState, Sequence[int]], "tuple[np.ndarray, float]"]


@dataclass(frozen=True)
class MCTSConfig:
    """Search hyperparameters (all config-driven; see ``configs/*.json``)."""

    sims: int = 160
    c_puct: float = 1.4
    dirichlet_alpha_scale: float = 10.0  # alpha = scale / len(legal)
    dirichlet_eps: float = 0.25
    chance_children: int = 4
    fpu: float = 0.0  # Q assumed for an unvisited edge (parent's frame)
    tree_reuse: bool = False  # keep the chosen child's subtree between moves
    # Decisive play (margin-head nets only; see the module docstring). A child is
    # a candidate if it has at least `decisive_min_visit_frac` of the best child's
    # visits, and it stays one if its win-Q is within `decisive_eps` of the best
    # candidate's. 0.03 of a [-1, 1] value is well inside the noise of a 512-sim
    # root, so the tie-break only ever fires between moves search calls equal.
    decisive_eps: float = 0.03
    decisive_min_visit_frac: float = 0.1
    # Within-tree batching (batched self-play only; see the module docstring).
    # search_batch = 1 is one leaf per forward pass, i.e. a search that is
    # bit-identical to the sequential one — which is why it is the default.
    search_batch: int = 1  # ceiling on leaves gathered per forward pass
    search_batch_ramp: int = 16  # ...and never more than root visits / this
    search_min_batch: int = 1  # ...but never fewer than this
    virtual_loss: float = 1.0  # discouragement on an edge with a pending descent

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> MCTSConfig:
        data = data or {}
        keys = (
            "sims",
            "c_puct",
            "dirichlet_alpha_scale",
            "dirichlet_eps",
            "chance_children",
            "fpu",
            "tree_reuse",
            "decisive_eps",
            "decisive_min_visit_frac",
            "search_batch",
            "search_batch_ramp",
            "search_min_batch",
            "virtual_loss",
        )
        kwargs: dict[str, Any] = {}
        for key in keys:
            if key in data:
                kwargs[key] = data[key]
        cfg = cls(**kwargs)
        if cfg.sims < 1:
            raise ValueError("sims must be >= 1")
        if cfg.chance_children < 1:
            raise ValueError("chance_children must be >= 1")
        if cfg.search_batch < 1 or cfg.search_min_batch < 1:
            raise ValueError("search_batch and search_min_batch must be >= 1")
        if cfg.search_batch_ramp < 1:
            raise ValueError("search_batch_ramp must be >= 1")
        return cfg


class Node:
    """A search node: one concrete state plus its outgoing edge statistics."""

    __slots__ = (
        "children",
        "expanded",
        "legal",
        "margins",
        "n_visits",
        "player",
        "priors",
        "state",
        "terminal_m0",
        "terminal_v0",
        "visits",
        "wins",
    )

    def __init__(self, state: AzulState) -> None:
        self.state = state
        self.player = state.current_player
        self.legal: list[int] = state.legal_actions()
        self.priors: list[float] = []
        self.visits: list[int] = []
        self.wins: list[float] = []  # total value in THIS node's player frame
        self.margins: list[float] = []  # total margin, same frame (run4 nets)
        self.children: list[Any] = []  # Node | dict[bytes, Node] | None
        self.expanded = False
        self.n_visits = 0
        self.terminal_v0 = 0.0
        self.terminal_m0 = 0.0
        if state.is_terminal:
            self.terminal_v0 = float(state.outcome() or 0.0)
            # A finished game knows its margin exactly, so the tie-break is fed
            # facts wherever the search actually reached the end.
            self.terminal_m0 = margin_target(state.scores[0] - state.scores[1])

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal

    def init_edges(self, priors: np.ndarray) -> None:
        n = len(self.legal)
        self.priors = [float(p) for p in priors] if n else []
        self.visits = [0] * n
        self.wins = [0.0] * n
        self.margins = [0.0] * n
        self.children = [None] * n
        self.expanded = True


class LeafRequest:
    """One position waiting for the network, plus the descents that want it.

    ``paths`` holds every descent that stopped at this node in the current
    gather — two descents can reach the same unexpanded leaf, and both are real
    simulations that must both be backed up with the same value. ``is_root`` marks
    the root's own evaluation, which has no path to back up at all.
    """

    __slots__ = ("is_root", "node", "paths")

    def __init__(self, node: Node, is_root: bool = False) -> None:
        self.node = node
        self.paths: list[list[tuple[Node, int]]] = []
        self.is_root = is_root

    @property
    def state(self) -> AzulState:
        return self.node.state

    @property
    def legal(self) -> list[int]:
        return self.node.legal


class _BatchSearch:
    """Bookkeeping for one in-flight :meth:`MCTS.start_search`."""

    __slots__ = (
        "by_node",
        "cap",
        "done",
        "forced",
        "need_root",
        "noise",
        "queue",
        "root",
        "root_margin",
        "root_value",
        "started",
    )

    def __init__(self, root: Node, noise: bool, cap: int, started: float) -> None:
        self.root = root
        self.noise = noise
        self.cap = cap
        self.started = started
        self.done = 0
        self.need_root = not root.expanded
        self.forced = False
        self.queue: list[LeafRequest] = []
        self.by_node: dict[Node, LeafRequest] = {}
        self.root_value = 0.0
        self.root_margin = 0.0


@dataclass(frozen=True)
class SearchResult:
    """What one :meth:`MCTS.search` call produced."""

    policy: np.ndarray  # length 180, visit distribution, 0 on illegal actions
    value: float  # root value estimate for the player to move
    visits: dict[int, int]  # action -> visit count
    sims: int
    elapsed_s: float = 0.0  # wall clock spent in the simulation loop
    # Margin-head nets only (see the module docstring); empty otherwise.
    has_margin: bool = False
    q: dict[int, float] = field(default_factory=dict)  # action -> win-Q, visited
    margins: dict[int, float] = field(default_factory=dict)  # action -> margin-Q
    margin: float = 0.0  # root margin estimate for the player to move


# --------------------------------------------------------------- evaluators
class UniformEvaluator:
    """Uniform priors, zero value — a net-free baseline for tests."""

    def __call__(
        self, state: AzulState, legal: Sequence[int]
    ) -> tuple[np.ndarray, float]:
        n = len(legal)
        if n == 0:
            return np.zeros(0, dtype=np.float32), 0.0
        return np.full(n, 1.0 / n, dtype=np.float32), 0.0


class RolloutEvaluator:
    """Uniform priors, value from one uniform-random playout (classic MCTS).

    Not used in training (AlphaZero replaces rollouts by the value head) but it
    gives MCTS a real signal without a trained net, which is exactly what the
    pure-search sanity test needs.
    """

    def __init__(self, seed: int = 0, max_moves: int = 400) -> None:
        self.rng = random.Random(seed)
        self.max_moves = max_moves

    def __call__(
        self, state: AzulState, legal: Sequence[int]
    ) -> tuple[np.ndarray, float]:
        n = len(legal)
        priors = (
            np.full(n, 1.0 / n, dtype=np.float32)
            if n
            else np.zeros(0, dtype=np.float32)
        )
        sim = state.clone()
        sim.rng.seed(self.rng.getrandbits(31))
        randrange = self.rng.randrange
        moves = 0
        while not sim.is_terminal and moves < self.max_moves:
            actions = sim.legal_actions()
            if not actions:  # pragma: no cover - defensive
                break
            sim.apply(actions[randrange(len(actions))])
            moves += 1
        # value for the player to move in `state`, from player 0's frame
        v0 = float(sim.outcome() or 0.0)
        return priors, v0 if state.current_player == 0 else -v0


# ---------------------------------------------------------------------- search
class MCTS:
    """PUCT search. One instance per playing agent (it owns its RNGs)."""

    def __init__(
        self,
        evaluator: Evaluator,
        config: MCTSConfig | None = None,
        seed: int = 0,
        add_noise: bool = False,
    ) -> None:
        self.evaluator = evaluator
        self.config = config or MCTSConfig()
        self.add_noise = add_noise
        # Set once: a margin-head evaluator returns 3-tuples, and the whole
        # margin backup (an extra float per edge) is skipped when it does not.
        self.has_margin = bool(getattr(evaluator, "has_margin", False))
        self.seed(seed)
        self.nodes_created = 0
        self.evals = 0

    def seed(self, n: int) -> None:
        self._seed = int(n) & 0x7FFFFFFF
        self._np_rng = np.random.default_rng(self._seed)
        self._rng = random.Random(self._seed)
        self._counter = 0
        self.reset_tree()

    # ------------------------------------------------------------- tree reuse
    def reset_tree(self) -> None:
        """Forget the kept subtree (a new game, or a caller that lost track)."""
        self._root: Node | None = None
        self._reuse_root: Node | None = None
        self._reuse_fp: tuple[Any, ...] | None = None
        self.reused_visits = 0  # visits inherited by the last search
        self._search: _BatchSearch | None = None  # in-flight batched search

    @staticmethod
    def _fingerprint(state: AzulState) -> tuple[Any, ...]:
        """Cheap near-unique identity of a position (guards a stale reuse)."""
        return (
            state.current_player,
            state.round_index,
            state.tiles_left,
            state.marker_in_center,
            tuple(state.scores),
            tuple(state.pl_count[0]),
            tuple(state.pl_count[1]),
            tuple(state.pl_color[0]),
            tuple(state.pl_color[1]),
            sum(state.walls[0]),
            sum(state.walls[1]),
            MCTS._chance_key(state),
        )

    def advance(self, action: int) -> bool:
        """Follow ``action`` from the current root; keep its subtree if we can.

        Call it once per move actually played. Returns whether a subtree
        survived — ``False`` at a refill boundary (the edge is a chance node),
        for an edge the search never expanded, or with reuse switched off.
        """
        node = self._root
        self._root = self._reuse_root = None
        self._reuse_fp = None
        if node is None or not self.config.tree_reuse or not node.expanded:
            return False
        try:
            index = node.legal.index(action)
        except ValueError:  # pragma: no cover - defensive
            return False
        child = node.children[index]
        if type(child) is not Node:  # chance table, or never expanded
            return False
        self._root = self._reuse_root = child
        self._reuse_fp = self._fingerprint(child.state)
        return True

    @property
    def rng(self) -> random.Random:
        """The search's own Python RNG (also used for move sampling)."""
        return self._rng

    # ------------------------------------------------------------------ public
    def search(
        self,
        state: AzulState,
        add_noise: bool | None = None,
        time_limit_s: float | None = None,
    ) -> SearchResult:
        """Run simulations from ``state`` (never mutated).

        ``time_limit_s`` (seconds, ``None`` = off) turns ``config.sims`` into an
        upper bound and keeps simulating until the budget is spent. Training
        never passes it, so training behaviour is unchanged.
        """
        noise = self.add_noise if add_noise is None else add_noise
        started = time.perf_counter()
        root = self._reuse_for(state)
        self.reused_visits = root.n_visits if root is not None else 0
        value = 0.0
        margin = 0.0
        if root is None:
            root = self._new_node(state.clone())
            if root.is_terminal:
                raise ValueError("cannot search a terminal state")
            value, margin = self._expand(root)
        self._root = root
        if len(root.legal) == 1:
            policy = np.zeros(ACTION_SPACE, dtype=np.float32)
            policy[root.legal[0]] = 1.0
            if root.n_visits:
                value = sum(root.wins) / root.n_visits
                margin = sum(root.margins) / root.n_visits
            return SearchResult(
                policy,
                float(value),
                {root.legal[0]: 1},
                0,
                0.0,
                has_margin=self.has_margin,
                margin=float(margin),
            )
        if noise:
            self._apply_noise(root)
        # A reused root keeps its visits, so the budget is a total: the tree ends
        # up the same size as a fresh search would have made it, for fewer evals.
        cap = max(0, self.config.sims - root.n_visits)
        budget = None if time_limit_s is None else float(time_limit_s)
        if budget is None or budget <= 0.0:
            for _ in range(cap):
                self._simulate(root)
        else:
            deadline = started + budget
            done = 0
            while done < cap:
                chunk = min(TIME_CHECK_EVERY, cap - done)
                for _ in range(chunk):
                    self._simulate(root)
                done += chunk
                if time.perf_counter() >= deadline:
                    break
        return self._result(root, value, margin, time.perf_counter() - started)

    def _result(
        self, root: Node, value: float, margin: float, elapsed: float
    ) -> SearchResult:
        """Read the root's edge statistics out as a :class:`SearchResult`.

        ``value``/``margin`` are the root's own network estimate and are only used
        for a root that was never simulated from.
        """
        total = root.n_visits
        policy = np.zeros(ACTION_SPACE, dtype=np.float32)
        visits: dict[int, int] = {}
        q: dict[int, float] = {}
        margins: dict[int, float] = {}
        want_margin = self.has_margin
        for i, action in enumerate(root.legal):
            n = root.visits[i]
            visits[action] = n
            if n and total:
                policy[action] = n / total
            if want_margin and n:
                q[action] = root.wins[i] / n
                margins[action] = root.margins[i] / n
        root_value = (sum(root.wins) / total) if total else value
        root_margin = (sum(root.margins) / total) if total else margin
        return SearchResult(
            policy,
            float(root_value),
            visits,
            total,
            float(elapsed),
            has_margin=want_margin,
            q=q,
            margins=margins,
            margin=float(root_margin),
        )

    # ------------------------------------------------------- batched interface
    def start_search(self, state: AzulState, add_noise: bool | None = None) -> None:
        """Open a search the caller will drive (see the module docstring).

        Does the same setup :meth:`search` does — reuse the kept subtree or clone
        a fresh root, work out the simulation budget — but stops before the first
        evaluation instead of calling the evaluator itself.
        """
        if self._search is not None:  # pragma: no cover - defensive
            raise RuntimeError("a batched search is already in progress")
        noise = self.add_noise if add_noise is None else add_noise
        started = time.perf_counter()
        root = self._reuse_for(state)
        self.reused_visits = root.n_visits if root is not None else 0
        if root is None:
            root = self._new_node(state.clone())
            if root.is_terminal:
                raise ValueError("cannot search a terminal state")
        self._root = root
        self._search = _BatchSearch(
            root, noise, max(0, self.config.sims - root.n_visits), started
        )
        if not self._search.need_root:
            self._open_root()

    def _open_root(self) -> None:
        """Root is expanded: settle the forced case, then mix in the noise.

        Same order as :meth:`search`, which returns on a forced root *before* it
        touches the priors — so a one-move root never consumes Dirichlet noise in
        either path, and the two RNG streams stay in step.
        """
        st = self._search
        assert st is not None
        if len(st.root.legal) == 1:
            st.forced = True
            return
        if st.noise:
            self._apply_noise(st.root)

    def search_done(self) -> bool:
        """Has the open search spent its budget? (False with none open.)"""
        st = self._search
        if st is None or st.need_root:
            return False
        return st.forced or st.done >= st.cap

    def leaf_requests(self, max_leaves: int = 0) -> list[LeafRequest]:
        """Descend until at least one position needs the net; return them.

        ``max_leaves`` (0 = no extra cap) bounds how many descents this call may
        make, which is how the driver keeps one game from monopolising a batch.
        The returned list is owned by the search: pass its evaluations straight
        back to :meth:`apply_leaves`.
        """
        st = self._search
        if st is None:  # pragma: no cover - defensive
            raise RuntimeError("no batched search in progress")
        if st.queue:  # already gathered, still waiting for its evaluations
            return st.queue
        if st.need_root:
            st.queue.append(LeafRequest(st.root, is_root=True))
            return st.queue
        cfg = self.config
        floor = min(cfg.search_min_batch, cfg.search_batch)
        while not st.queue and not self.search_done():
            # As big as the tree can afford, never bigger than the caller wants:
            # a flat batch on a small tree is what made the browser's first
            # attempt weaker than no batching at all.
            want = max(
                floor, min(cfg.search_batch, st.root.n_visits // cfg.search_batch_ramp)
            )
            want = min(want, st.cap - st.done)
            if max_leaves > 0:
                want = min(want, max_leaves)
            want = max(1, want)
            for _ in range(want):
                self._collect(st)
            st.done += want
        return st.queue

    def apply_leaves(
        self,
        results: Sequence[tuple[np.ndarray, float] | tuple[np.ndarray, float, float]],
    ) -> None:
        """Expand every gathered leaf with its evaluation and back the values up."""
        st = self._search
        if st is None:  # pragma: no cover - defensive
            raise RuntimeError("no batched search in progress")
        queue = st.queue
        if len(results) != len(queue):
            raise ValueError(f"expected {len(queue)} evaluations, got {len(results)}")
        for request, out in zip(queue, results):
            node = request.node
            if len(out) == 3:
                priors, value, margin = out  # type: ignore[misc]
            else:
                priors, value = out  # type: ignore[misc]
                margin = 0.0
            node.init_edges(priors)
            self.evals += 1
            if request.is_root:
                st.root_value = float(value)
                st.root_margin = float(margin)
                continue
            flip = 1.0 if node.player == 0 else -1.0
            v0 = float(value) * flip
            m0 = float(margin) * flip
            for path in request.paths:
                self._backup(path, v0, m0)
        st.queue = []
        st.by_node.clear()
        if st.need_root:
            st.need_root = False
            self._open_root()

    def finish_search(self) -> SearchResult:
        """Close the open search and report it, exactly like :meth:`search`."""
        st = self._search
        if st is None:  # pragma: no cover - defensive
            raise RuntimeError("no batched search in progress")
        self._search = None
        root = st.root
        if st.forced:
            policy = np.zeros(ACTION_SPACE, dtype=np.float32)
            policy[root.legal[0]] = 1.0
            value, margin = st.root_value, st.root_margin
            if root.n_visits:
                value = sum(root.wins) / root.n_visits
                margin = sum(root.margins) / root.n_visits
            return SearchResult(
                policy,
                float(value),
                {root.legal[0]: 1},
                0,
                0.0,
                has_margin=self.has_margin,
                margin=float(margin),
            )
        return self._result(
            root, st.root_value, st.root_margin, time.perf_counter() - st.started
        )

    def _collect(self, st: _BatchSearch) -> None:
        """One descent, laying virtual loss; ends at a terminal or a new leaf."""
        vl = self.config.virtual_loss
        node = st.root
        path: list[tuple[Node, int]] = []
        while True:
            if node.is_terminal:
                self._backup(path, node.terminal_v0, node.terminal_m0)
                return
            if not node.expanded:
                request = st.by_node.get(node)
                if request is None:
                    request = LeafRequest(node)
                    st.by_node[node] = request
                    st.queue.append(request)
                request.paths.append(path)
                return
            index = self._select(node)
            # The visit is taken now and the loss assumed now; `_backup` gives the
            # assumed loss back and credits the real value, so a finished batch
            # leaves the tree exactly where the same evaluations would have left
            # it one at a time.
            node.visits[index] += 1
            node.n_visits += 1
            node.wins[index] -= vl
            path.append((node, index))
            node = self._child(node, index)

    def _backup(self, path: list[tuple[Node, int]], v0: float, m0: float) -> None:
        """Undo the virtual loss along ``path`` and credit the real result."""
        vl = self.config.virtual_loss
        margin = self.has_margin
        for parent, index in path:
            flip = 1.0 if parent.player == 0 else -1.0
            parent.wins[index] += vl + v0 * flip
            if margin:
                parent.margins[index] += m0 * flip

    # ------------------------------------------------------------------ guts
    def _reuse_for(self, state: AzulState) -> Node | None:
        """The kept subtree if it really is ``state``'s node, else ``None``."""
        root = self._reuse_root
        self._reuse_root = None
        fingerprint = self._reuse_fp
        self._reuse_fp = None
        if root is None or not self.config.tree_reuse:
            return None
        if root.is_terminal or not root.expanded or not root.legal:
            return None
        if fingerprint != self._fingerprint(state):  # pragma: no cover - defensive
            return None
        return root

    def _new_node(self, state: AzulState) -> Node:
        self.nodes_created += 1
        return Node(state)

    def _expand(self, node: Node) -> tuple[float, float]:
        """Evaluate ``node`` and initialise its edges; returns ``(value, margin)``.

        ``margin`` is 0.0 for an evaluator without a margin head, and nothing
        downstream reads it in that case.
        """
        out = self.evaluator(node.state, node.legal)
        self.evals += 1
        if len(out) == 3:
            priors, value, margin = out  # type: ignore[misc]
        else:
            priors, value = out  # type: ignore[misc]
            margin = 0.0
        node.init_edges(priors)
        return float(value), float(margin)

    def _apply_noise(self, root: Node) -> None:
        n = len(root.legal)
        if n < 2:
            return
        cfg = self.config
        alpha = max(cfg.dirichlet_alpha_scale / n, 1e-3)
        noise = self._np_rng.dirichlet([alpha] * n)
        eps = cfg.dirichlet_eps
        root.priors = [
            (1.0 - eps) * p + eps * float(x) for p, x in zip(root.priors, noise)
        ]

    def _select(self, node: Node) -> int:
        cfg = self.config
        priors = node.priors
        visits = node.visits
        wins = node.wins
        scale = cfg.c_puct * math.sqrt(node.n_visits + 1)
        fpu = cfg.fpu
        best = -1e30
        best_i = 0
        for i, n in enumerate(visits):
            q = wins[i] / n if n else fpu
            score = q + scale * priors[i] / (1 + n)
            if score > best:
                best = score
                best_i = i
        return best_i

    def _is_stochastic(self, state: AzulState, action: int) -> bool:
        """True iff ``action`` empties the board and therefore triggers a refill."""
        source, rest = divmod(action, 30)
        color = rest // 6
        pool = state.center if source == CENTER else state.factories[source]
        return pool[color] == state.tiles_left

    def _determinize(self, state: AzulState, action: int) -> AzulState:
        """Clone with a fresh bag order, then apply ``action`` (one refill draw)."""
        child = state.clone()
        self._counter += 1
        child.rng.seed(
            (self._seed * 1_000_003 + self._counter * 2_654_435_761) & 0x7FFFFFFF
        )
        child.rng.shuffle(child.bag)
        child.apply(action)
        return child

    @staticmethod
    def _chance_key(state: AzulState) -> bytes:
        """Identity of a post-refill position: factory + center contents."""
        parts: list[int] = []
        for factory in state.factories:
            parts.extend(factory)
        parts.extend(state.center)
        return bytes(parts)

    def _child(self, node: Node, index: int) -> Node:
        entry = node.children[index]
        if type(entry) is Node:
            return entry
        action = node.legal[index]
        if entry is None and not self._is_stochastic(node.state, action):
            child = self._new_node(self._apply_plain(node.state, action))
            node.children[index] = child
            return child
        table: dict[bytes, Node] = {} if entry is None else entry
        node.children[index] = table
        if len(table) >= self.config.chance_children:
            keys = list(table)
            return table[keys[self._rng.randrange(len(keys))]]
        state = self._determinize(node.state, action)
        key = self._chance_key(state)
        child = table.get(key)
        if child is None:
            child = self._new_node(state)
            table[key] = child
        return child

    @staticmethod
    def _apply_plain(state: AzulState, action: int) -> AzulState:
        child = state.clone()
        child.apply(action)
        return child

    def _simulate(self, root: Node) -> None:
        node = root
        path: list[tuple[Node, int]] = []
        while True:
            if node.is_terminal:
                v0 = node.terminal_v0
                m0 = node.terminal_m0
                break
            if not node.expanded:
                value, margin = self._expand(node)
                flip = 1.0 if node.player == 0 else -1.0
                v0 = value * flip
                m0 = margin * flip
                break
            index = self._select(node)
            path.append((node, index))
            node = self._child(node, index)
        if self.has_margin:
            for parent, index in path:
                parent.visits[index] += 1
                parent.n_visits += 1
                flip = 1.0 if parent.player == 0 else -1.0
                parent.wins[index] += v0 * flip
                parent.margins[index] += m0 * flip
            return
        for parent, index in path:
            parent.visits[index] += 1
            parent.n_visits += 1
            parent.wins[index] += v0 if parent.player == 0 else -v0


def select_action(
    policy: np.ndarray,
    temperature: float = 0.0,
    rng: random.Random | np.random.Generator | None = None,
) -> int:
    """Pick an action from a visit distribution.

    ``temperature <= 0`` is argmax; otherwise the distribution is raised to
    ``1 / temperature``, renormalised and sampled.
    """
    if temperature <= 0.0:
        return int(np.argmax(policy))
    probs = np.asarray(policy, dtype=np.float64)
    if temperature != 1.0:
        with np.errstate(divide="ignore"):
            probs = np.power(probs, 1.0 / temperature)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0.0:  # pragma: no cover - defensive
        return int(np.argmax(policy))
    probs = probs / total
    if isinstance(rng, np.random.Generator):
        return int(rng.choice(len(probs), p=probs))
    draw = (rng.random() if rng is not None else random.random()) * 1.0
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if draw <= acc:
            return i
    return int(np.argmax(probs))  # pragma: no cover - float rounding


def decisive_action(
    result: SearchResult,
    eps: float = 0.03,
    min_visit_frac: float = 0.1,
) -> int:
    """Best win-Q first, biggest margin second — the run4 play-time pick.

    Winning is lexicographically prior: the candidate set is *only* the root
    children whose win-Q is within ``eps`` of the best, so the margin can never
    trade a win away. ``min_visit_frac`` keeps a barely-visited child, whose Q is
    one or two backups of noise, from defining "the best".

    Ties (equal margin) fall back to visits and then to the lowest action index,
    so the choice is deterministic and reproducible across processes.
    """
    if not result.has_margin or not result.q:
        return int(np.argmax(result.policy))
    best_visits = max(result.visits.values(), default=0)
    if best_visits <= 0:  # pragma: no cover - defensive: nothing was searched
        return int(np.argmax(result.policy))
    floor = min_visit_frac * best_visits
    candidates = [a for a, n in result.visits.items() if n >= floor and a in result.q]
    if not candidates:  # pragma: no cover - defensive
        return int(np.argmax(result.policy))
    best_q = max(result.q[a] for a in candidates)
    keep = [a for a in candidates if result.q[a] >= best_q - eps]
    return max(keep, key=lambda a: (result.margins[a], result.visits[a], -a))


def select_play_action(
    result: SearchResult,
    temperature: float = 0.0,
    rng: random.Random | np.random.Generator | None = None,
    eps: float = 0.03,
    min_visit_frac: float = 0.1,
) -> int:
    """The move to actually play: sampled when exploring, decisive when not.

    ``temperature > 0`` is the exploration path and is untouched — the visit
    distribution is sampled exactly as before. At temperature 0 a margin-head
    search uses :func:`decisive_action`; anything else is the historical
    ``argmax`` over visits, which is what makes run1/run2/run3 checkpoints play
    bit-identically to before this head existed.
    """
    if temperature > 0.0 or not result.has_margin:
        return select_action(result.policy, temperature, rng)
    return decisive_action(result, eps=eps, min_visit_frac=min_visit_frac)
