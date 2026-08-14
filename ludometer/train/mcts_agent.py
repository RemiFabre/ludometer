"""The neural agent: PUCT MCTS driven by a :class:`PolicyValueNet` checkpoint.

This is what the arena, the Elo evaluator and the GUI play against, so the
constructor signature is part of the public API:

    MCTSAgent.from_checkpoint(path, sims=..., seed=...)

For multiprocessing (``ludometer.eval.arena``) pass a :class:`MCTSAgentSpec`
instead of an instance: it is a small picklable factory that each worker turns
into an agent, reusing a per-process cache so a 40-game match loads the
checkpoint once instead of once per game.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ludometer.agents.base import Agent
from ludometer.azul.engine import AzulState
from ludometer.train.mcts import MCTS, STALL_ROUNDS, MCTSConfig, select_action
from ludometer.train.net import NetEvaluator, PolicyValueNet, load_net

__all__ = ["MCTSAgent", "MCTSAgentSpec"]

# (path, device) -> net, so repeated games in one worker share the weights.
_NET_CACHE: dict[tuple[str, str], PolicyValueNet] = {}


def _cached_net(path: str | os.PathLike[str], device: str) -> PolicyValueNet:
    key = (str(Path(path).resolve()), device)
    net = _NET_CACHE.get(key)
    if net is None:
        net, _ = load_net(path, device=device)
        _NET_CACHE[key] = net
    return net


class MCTSAgent(Agent):
    """Plays the arg-max (or sampled) visit count of a PUCT search."""

    def __init__(
        self,
        net: PolicyValueNet | None = None,
        sims: int = 200,
        seed: int | None = None,
        device: str = "cpu",
        c_puct: float = 1.4,
        temperature: float = 0.0,
        add_noise: bool = False,
        name: str = "mcts",
        evaluator: Any = None,
        config: MCTSConfig | None = None,
        stall_rounds: int = STALL_ROUNDS,
    ) -> None:
        if net is None and evaluator is None:
            raise ValueError("MCTSAgent needs either a net or an evaluator")
        self.name = name
        self.net = net
        self.temperature = float(temperature)
        self.stall_rounds = int(stall_rounds)
        cfg = config or MCTSConfig(sims=sims, c_puct=c_puct)
        self.evaluator = evaluator or NetEvaluator(net, device=device)
        self.mcts = MCTS(
            self.evaluator, cfg, seed=0 if seed is None else seed, add_noise=add_noise
        )
        self._rng_seed = 0 if seed is None else seed

    # ------------------------------------------------------------------ build
    @classmethod
    def from_checkpoint(
        cls,
        path: str | os.PathLike[str],
        sims: int = 200,
        seed: int | None = None,
        device: str = "cpu",
        threads: int | None = None,
        **kwargs: Any,
    ) -> MCTSAgent:
        """Load ``path`` (a ``save_checkpoint`` payload) and wrap it in an agent.

        ``threads`` is only honoured when given: process-wide thread counts are
        the caller's business (arena workers pass 1, the GUI leaves it alone).
        """
        if threads:
            torch.set_num_threads(threads)
        net = _cached_net(path, device)
        name = kwargs.pop("name", None) or f"mcts:{Path(path).stem}"
        return cls(net, sims=sims, seed=seed, device=device, name=name, **kwargs)

    # ------------------------------------------------------------------- play
    def seed(self, n: int) -> None:
        self._rng_seed = int(n)
        self.mcts.seed(n)

    def act(self, state: AzulState) -> int:
        legal = state.legal_actions()
        if not legal:
            raise ValueError("no legal actions (terminal state?)")
        if len(legal) == 1:
            return legal[0]
        result = self.mcts.search(state)
        temperature = self.temperature
        if state.round_index >= self.stall_rounds:
            # Two arg-max players can loop forever (see mcts.STALL_ROUNDS):
            # a pathologically long game gets randomised so that it terminates.
            temperature = max(temperature, 1.0)
        return select_action(result.policy, temperature, self.mcts.rng)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<MCTSAgent {self.name} sims={self.mcts.config.sims}>"


@dataclass(frozen=True)
class MCTSAgentSpec:
    """Picklable agent spec: ``make_agent(spec)`` calls it in the worker."""

    path: str
    sims: int = 100
    seed: int | None = None
    device: str = "cpu"
    name: str = "mcts"
    c_puct: float = 1.4
    temperature: float = 0.0
    stall_rounds: int = STALL_ROUNDS
    threads: int = 1  # one match game per process: keep torch single-threaded

    def __call__(self) -> MCTSAgent:
        return MCTSAgent.from_checkpoint(
            self.path,
            sims=self.sims,
            seed=self.seed,
            threads=self.threads,
            stall_rounds=self.stall_rounds,
            device=self.device,
            name=self.name,
            c_puct=self.c_puct,
            temperature=self.temperature,
        )
