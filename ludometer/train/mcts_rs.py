"""`MCTS` on the Rust tree: an API-compatible wrapper over ``ludometer_rs.Tree``.

Same constructor, same ``search`` / ``advance`` / ``seed`` / leaf protocol as
:class:`ludometer.train.mcts.MCTS`, same :class:`~ludometer.train.mcts.SearchResult`
out. The evaluator is still a Python callable ``(state, legal) -> (priors,
value[, margin])``; it receives a ``ludometer_rs.State`` (which has ``encode()``
and every attribute the evaluators read). Given identical evaluations the two
searches are identical (``tests/test_rust_mcts.py``); root Dirichlet noise draws
from the Rust generator, so noisy searches match only in distribution.

Selected by :class:`~ludometer.train.mcts_agent.MCTSAgent` with
``engine="rust"`` (agent specs: ``mcts:<ckpt>?sims=400&engine=rust``) or by
``LUDOMETER_ENGINE=rust``.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

from ludometer.train.mcts import Evaluator, MCTSConfig, SearchResult

__all__ = ["MCTS", "LeafRequest", "default_engine"]


def default_engine() -> str:
    """``"rust"`` when ``LUDOMETER_ENGINE=rust`` is set and the module is built."""
    want = os.environ.get("LUDOMETER_ENGINE", "").strip().lower()
    if want == "rust":
        try:
            import ludometer_rs  # noqa: F401
        except ImportError:
            return "python"
        return "rust"
    return "python"


class _Rng:
    """The tree's own generator with ``random.Random``'s two methods."""

    __slots__ = ("_tree",)

    def __init__(self, tree: Any) -> None:
        self._tree = tree

    def random(self) -> float:
        return self._tree.rng_random()

    def randrange(self, n: int) -> int:
        return self._tree.rng_randrange(int(n))


class LeafRequest:
    """One pending leaf: its state and its legal list (see ``MCTS.leaf_requests``)."""

    __slots__ = ("legal", "state")

    def __init__(self, state: Any, legal: list[int]) -> None:
        self.state = state
        self.legal = legal

    @property
    def node(self) -> LeafRequest:  # the batched driver reads `r.node.state`
        return self


class MCTS:
    """PUCT search on the Rust tree; one instance per playing agent."""

    def __init__(
        self,
        evaluator: Evaluator,
        config: MCTSConfig | None = None,
        seed: int = 0,
        add_noise: bool = False,
        rng: str = "fast",
    ) -> None:
        import ludometer_rs

        from ludometer.azul.engine_rs import to_rust

        self._rs = ludometer_rs
        self._to_rust = to_rust
        self.evaluator = evaluator
        self._config = config or MCTSConfig()
        self.add_noise = add_noise
        self.has_margin = bool(getattr(evaluator, "has_margin", False))
        self._tree = ludometer_rs.Tree(
            asdict(self._config),
            has_margin=self.has_margin,
            seed=int(seed) & 0x7FFFFFFF,
            add_noise=add_noise,
            rng=rng,
        )
        self._rng = _Rng(self._tree)
        self._queue: list[LeafRequest] = []

    # ---------------------------------------------------------------- config
    @property
    def config(self) -> MCTSConfig:
        return self._config

    @config.setter
    def config(self, value: MCTSConfig) -> None:
        self._config = value
        self._tree.config = asdict(value)

    @property
    def rng(self) -> _Rng:
        return self._rng

    @property
    def evals(self) -> int:
        return self._tree.evals

    @evals.setter
    def evals(self, value: int) -> None:
        self._tree.evals = int(value)

    @property
    def nodes_created(self) -> int:
        return self._tree.nodes_created

    @property
    def reused_visits(self) -> int:
        return self._tree.reused_visits

    def seed(self, n: int) -> None:
        self._tree.seed(int(n))

    def reset_tree(self) -> None:
        self._tree.reset_tree()
        self._queue = []

    def advance(self, action: int) -> bool:
        return self._tree.advance(int(action))

    # ---------------------------------------------------------------- search
    def _evaluate(self, state: Any, legal: Sequence[int]) -> tuple[Any, float, float]:
        out = self.evaluator(state, legal)
        if len(out) == 3:
            return out  # type: ignore[return-value]
        priors, value = out  # type: ignore[misc]
        return priors, value, 0.0

    def search(
        self,
        state: Any,
        add_noise: bool | None = None,
        time_limit_s: float | None = None,
        sims: int | None = None,
    ) -> SearchResult:
        root = self._to_rust(state)
        out = self._tree.search(
            root,
            self._evaluate,
            add_noise=add_noise,
            time_limit_s=time_limit_s,
            sims=sims,
        )
        return _result(out)

    # ---------------------------------------------------- batched interface
    def start_search(
        self, state: Any, add_noise: bool | None = None, sims: int | None = None
    ) -> None:
        self._queue = []
        self._tree.start_search(self._to_rust(state), add_noise=add_noise, sims=sims)

    def search_done(self) -> bool:
        return self._tree.search_done()

    def leaf_requests(self, max_leaves: int = 0) -> list[LeafRequest]:
        if self._queue:
            return self._queue
        _obs, legal = self._tree.leaf_requests(int(max_leaves))
        states = self._tree.leaf_states()
        self._queue = [LeafRequest(s, l) for s, l in zip(states, legal)]
        return self._queue

    def apply_leaves(self, results: Sequence[tuple[Any, ...]]) -> None:
        if len(results) != len(self._queue):
            raise ValueError(
                f"expected {len(self._queue)} evaluations, got {len(results)}"
            )
        priors = [np.ascontiguousarray(r[0], dtype=np.float32) for r in results]
        values = [float(r[1]) for r in results]
        margins = [float(r[2]) if len(r) == 3 else 0.0 for r in results]
        self._tree.apply_leaves(priors, values, margins)
        self._queue = []

    def finish_search(self) -> SearchResult:
        self._queue = []
        return _result(self._tree.finish_search())


def _result(d: dict[str, Any]) -> SearchResult:
    return SearchResult(
        policy=d["policy"],
        value=float(d["value"]),
        visits={int(a): int(n) for a, n in d["visits"].items()},
        sims=int(d["sims"]),
        elapsed_s=float(d["elapsed_s"]),
        has_margin=bool(d["has_margin"]),
        q={int(a): float(v) for a, v in d["q"].items()},
        margins={int(a): float(v) for a, v in d["margins"].items()},
        margin=float(d["margin"]),
    )
