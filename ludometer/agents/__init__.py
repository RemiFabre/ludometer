"""Agents: the common interface plus the hand-written baselines.

``AGENT_REGISTRY`` maps a short name to a constructor so that agents can be
passed *by spec* (a string, a ``(name, kwargs)`` pair or a callable) across
process boundaries — see :mod:`ludometer.eval.arena`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Union

from ludometer.agents.base import Agent
from ludometer.agents.greedy import GreedyAgent
from ludometer.agents.heuristic import HeuristicAgent
from ludometer.agents.random_agent import RandomAgent
from ludometer.uno.agents import (
    UnoGreedyAgent,
    UnoHeuristicAgent,
    UnoPlusGreedyAgent,
    UnoPlusHeuristicAgent,
    UnoPlusRandomAgent,
    UnoRandomAgent,
)

__all__ = [
    "AGENT_REGISTRY",
    "Agent",
    "AgentSpec",
    "GreedyAgent",
    "HeuristicAgent",
    "RandomAgent",
    "UnoGreedyAgent",
    "UnoHeuristicAgent",
    "UnoRandomAgent",
    "make_agent",
    "spec_name",
]

AGENT_REGISTRY: dict[str, Callable[..., Agent]] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "heuristic": HeuristicAgent,
    # Uno's ladder; specs are game-qualified so one registry serves both games.
    "uno:random": UnoRandomAgent,
    "uno:greedy": UnoGreedyAgent,
    "uno:heuristic": UnoHeuristicAgent,
    "unoplus:random": UnoPlusRandomAgent,
    "unoplus:greedy": UnoPlusGreedyAgent,
    "unoplus:heuristic": UnoPlusHeuristicAgent,
}

# A spec is anything :func:`make_agent` understands.
AgentSpec = Union[  # noqa: UP007 - kept explicit for readability
    str,
    Agent,
    Callable[[], Agent],
    tuple[str, Mapping[str, Any]],
]


def make_agent(spec: AgentSpec) -> Agent:
    """Build an agent from a spec.

    Accepted forms: ``"greedy"``, ``("heuristic", {"floor": 2.0})``, any
    zero-argument callable returning an :class:`Agent` (e.g. a class), or an
    already-built :class:`Agent` (returned unchanged).
    """
    if isinstance(spec, Agent):
        return spec
    if isinstance(spec, str):
        try:
            factory = AGENT_REGISTRY[spec]
        except KeyError:
            # richer spec strings ("mcts:<ckpt>?sims=n", "best") live in the registry
            from ludometer.agents.registry import load_agent

            try:
                return load_agent(spec)
            except ValueError:
                raise KeyError(
                    f"unknown agent {spec!r}; known: {sorted(AGENT_REGISTRY)} "
                    "or a registry spec (mcts:<ckpt>?sims=n, best)"
                ) from None
        return factory()
    if isinstance(spec, tuple):
        name, kwargs = spec
        return AGENT_REGISTRY[name](**dict(kwargs))
    if callable(spec):
        return spec()
    raise TypeError(f"cannot build an agent from {spec!r}")


def spec_name(spec: AgentSpec) -> str:
    """Human-readable name for a spec, without building it when avoidable."""
    if isinstance(spec, Agent):
        return spec.name
    if isinstance(spec, str):
        return spec
    if isinstance(spec, tuple):
        return str(spec[0])
    return getattr(spec, "name", getattr(spec, "__name__", repr(spec)))
