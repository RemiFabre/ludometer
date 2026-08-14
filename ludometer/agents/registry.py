"""Agent specs -> instances, shared by the GUI, arena, and eval.

Spec strings:
    "random" | "greedy" | "heuristic"
    "mcts:<checkpoint_path>?sims=<n>"   (neural agent; requires ludometer.train)
"""

from __future__ import annotations

from ludometer.agents.greedy import GreedyAgent
from ludometer.agents.heuristic import HeuristicAgent
from ludometer.agents.random_agent import RandomAgent


def load_agent(spec: str, seed: int | None = None):
    if spec == "random":
        return RandomAgent(seed=seed)
    if spec == "greedy":
        return GreedyAgent(seed=seed)
    if spec == "heuristic":
        return HeuristicAgent(seed=seed)
    if spec.startswith("mcts:"):
        rest = spec[len("mcts:") :]
        path, _, query = rest.partition("?")
        sims = 200
        for part in query.split("&"):
            if part.startswith("sims="):
                sims = int(part[len("sims=") :])
        from ludometer.train.mcts_agent import MCTSAgent  # lazy: needs torch

        return MCTSAgent.from_checkpoint(path, sims=sims, seed=seed)
    raise ValueError(f"unknown agent spec: {spec!r}")
