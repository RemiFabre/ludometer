"""Agent interface shared by every Ludometer player (see docs/DESIGN.md).

An agent is a tiny object with a name and an ``act`` method mapping a state to a
legal ``action_id``. Agents must never mutate the state they are handed; use
``state.clone()`` for lookahead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ludometer.azul.engine import AzulState

__all__ = ["Agent"]


class Agent(ABC):
    """Base class: subclasses set :attr:`name` and implement :meth:`act`."""

    name: str = "agent"

    @abstractmethod
    def act(self, state: AzulState) -> int:
        """Return a legal action id for ``state.current_player``."""

    def seed(self, n: int) -> None:  # optional: deterministic agents need nothing
        """Reseed any internal randomness. Deterministic agents ignore this."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name}>"
