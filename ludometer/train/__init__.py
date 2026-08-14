"""Training: AlphaZero-style self-play, MCTS, replay buffer and the run loop.

Import order matters for the multiprocessing workers: everything here is safe to
import in a freshly ``spawn``-ed process (no side effects, no device selection at
import time).
"""

from __future__ import annotations

__all__ = [
    "MCTS",
    "MCTSAgent",
    "MCTSConfig",
    "NetConfig",
    "PolicyValueNet",
    "ReplayBuffer",
    "TrainConfig",
    "Trainer",
]


def __getattr__(name: str):  # pragma: no cover - lazy re-exports (torch is heavy)
    if name in ("NetConfig", "PolicyValueNet"):
        from ludometer.train import net as _net

        return getattr(_net, name)
    if name in ("MCTS", "MCTSConfig"):
        from ludometer.train import mcts as _mcts

        return getattr(_mcts, name)
    if name == "MCTSAgent":
        from ludometer.train.mcts_agent import MCTSAgent

        return MCTSAgent
    if name == "ReplayBuffer":
        from ludometer.train.replay import ReplayBuffer

        return ReplayBuffer
    if name in ("TrainConfig", "Trainer"):
        from ludometer.train import trainer as _trainer

        return getattr(_trainer, name)
    raise AttributeError(name)
