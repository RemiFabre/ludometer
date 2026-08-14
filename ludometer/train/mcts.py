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

Tree reuse between moves is not implemented (optional per the design): every
``search`` call starts from a fresh root.

Time budget (opt-in, GUI only)
------------------------------
``search(state, time_limit_s=...)`` keeps simulating until the wall clock runs
out, checking it every :data:`TIME_CHECK_EVERY` simulations, with
``config.sims`` acting as the upper bound. The default is ``None``: training and
the arena keep running exactly ``config.sims`` simulations, so results stay
reproducible.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ludometer.azul.engine import ACTION_SPACE, CENTER, AzulState

__all__ = [
    "MAX_GAME_MOVES",
    "MCTS",
    "STALL_ROUNDS",
    "TIME_CHECK_EVERY",
    "MCTSConfig",
    "Node",
    "RolloutEvaluator",
    "SearchResult",
    "UniformEvaluator",
    "select_action",
]

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
# value in [-1, 1] for the player to move).
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
        return cfg


class Node:
    """A search node: one concrete state plus its outgoing edge statistics."""

    __slots__ = (
        "children",
        "expanded",
        "legal",
        "n_visits",
        "player",
        "priors",
        "state",
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
        self.children: list[Any] = []  # Node | dict[bytes, Node] | None
        self.expanded = False
        self.n_visits = 0
        self.terminal_v0 = 0.0
        if state.is_terminal:
            self.terminal_v0 = float(state.outcome() or 0.0)

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal

    def init_edges(self, priors: np.ndarray) -> None:
        n = len(self.legal)
        self.priors = [float(p) for p in priors] if n else []
        self.visits = [0] * n
        self.wins = [0.0] * n
        self.children = [None] * n
        self.expanded = True


@dataclass(frozen=True)
class SearchResult:
    """What one :meth:`MCTS.search` call produced."""

    policy: np.ndarray  # length 180, visit distribution, 0 on illegal actions
    value: float  # root value estimate for the player to move
    visits: dict[int, int]  # action -> visit count
    sims: int
    elapsed_s: float = 0.0  # wall clock spent in the simulation loop


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
        self.seed(seed)
        self.nodes_created = 0
        self.evals = 0

    def seed(self, n: int) -> None:
        self._seed = int(n) & 0x7FFFFFFF
        self._np_rng = np.random.default_rng(self._seed)
        self._rng = random.Random(self._seed)
        self._counter = 0

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
        root = self._new_node(state.clone())
        if root.is_terminal:
            raise ValueError("cannot search a terminal state")
        started = time.perf_counter()
        value = self._expand(root)
        if len(root.legal) == 1:
            policy = np.zeros(ACTION_SPACE, dtype=np.float32)
            policy[root.legal[0]] = 1.0
            return SearchResult(policy, value, {root.legal[0]: 1}, 0, 0.0)
        if noise:
            self._apply_noise(root)
        cap = self.config.sims
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
        elapsed = time.perf_counter() - started

        total = root.n_visits
        policy = np.zeros(ACTION_SPACE, dtype=np.float32)
        visits: dict[int, int] = {}
        for i, action in enumerate(root.legal):
            n = root.visits[i]
            visits[action] = n
            if n and total:
                policy[action] = n / total
        root_value = (sum(root.wins) / total) if total else value
        return SearchResult(policy, float(root_value), visits, total, float(elapsed))

    # ------------------------------------------------------------------ guts
    def _new_node(self, state: AzulState) -> Node:
        self.nodes_created += 1
        return Node(state)

    def _expand(self, node: Node) -> float:
        """Evaluate ``node`` and initialise its edges; returns its value."""
        priors, value = self.evaluator(node.state, node.legal)
        self.evals += 1
        node.init_edges(priors)
        return float(value)

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
                break
            if not node.expanded:
                value = self._expand(node)
                v0 = value if node.player == 0 else -value
                break
            index = self._select(node)
            path.append((node, index))
            node = self._child(node, index)
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
