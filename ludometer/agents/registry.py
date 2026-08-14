"""Agent specs -> instances, shared by the GUI, arena, and eval.

Spec strings:
    "random" | "greedy" | "heuristic"
    "mcts:<checkpoint_path>?sims=<n>"   (neural agent; requires ludometer.train)

``load_agent`` raises ``ValueError`` for anything it cannot parse and lets the
underlying error through for a checkpoint that exists but cannot be loaded, so
callers (the GUI) can show the reason verbatim.
"""

from __future__ import annotations

from ludometer.agents.greedy import GreedyAgent
from ludometer.agents.heuristic import HeuristicAgent
from ludometer.agents.random_agent import RandomAgent

__all__ = ["BASELINES", "DEFAULT_SIMS", "load_agent"]

BASELINES = ("random", "greedy", "heuristic")
DEFAULT_SIMS = 200


def load_agent(spec: str, seed: int | None = None):
    """Build the agent described by ``spec`` (see the module docstring)."""
    if not isinstance(spec, str):
        raise TypeError(f"agent spec must be a string, got {type(spec).__name__}")
    spec = spec.strip()
    # the baselines take `seed` as a plain int; None means "system entropy"
    kwargs = {} if seed is None else {"seed": int(seed)}
    if spec == "random":
        return RandomAgent(**kwargs)
    if spec == "greedy":
        return GreedyAgent(**kwargs)
    if spec == "heuristic":
        return HeuristicAgent(**kwargs)
    if spec.startswith("mcts:"):
        rest = spec[len("mcts:") :]
        path, _, query = rest.partition("?")
        if not path:
            raise ValueError("mcts spec needs a checkpoint path: mcts:<path>?sims=<n>")
        sims = DEFAULT_SIMS
        for part in query.split("&"):
            if not part:
                continue
            key, _, value = part.partition("=")
            if key != "sims":
                raise ValueError(
                    f"unknown mcts option {key!r} (only sims= is supported)"
                )
            try:
                sims = int(value)
            except ValueError:
                raise ValueError(f"sims must be an integer, got {value!r}") from None
            if sims < 1:
                raise ValueError(f"sims must be >= 1, got {sims}")
        from ludometer.train.mcts_agent import MCTSAgent  # lazy: needs torch

        return MCTSAgent.from_checkpoint(path, sims=sims, seed=seed)
    raise ValueError(
        f"unknown agent spec: {spec!r}; expected one of {', '.join(BASELINES)} "
        "or mcts:<checkpoint>?sims=<n>"
    )
