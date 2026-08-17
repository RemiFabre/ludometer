"""Turn validated human games into a ``replay.npz`` that ``--pretrain`` can read.

The output is byte-for-byte the format
:class:`ludometer.train.replay.ReplayBuffer` writes, because it *is* that class
doing the writing — importing it costs nothing (that module is numpy-only) and
guarantees we never drift from the format the trainer expects. Per row:

===============  ====================================================================
``states``       the position before the human move, mover's frame (182 floats)
``policies``     **one-hot on the move the human played** (180 floats)
``values``       the game outcome in the mover's frame, +1 / 0 / -1
``margins``      ``tanh(final score diff / 20)``, mover's frame, mask 1
``aux``          the 30 final-wall bits for both walls, mover's frame, mask 1
``policy_mask``  1 — every human move is a real policy target
===============  ====================================================================

Two things to know about the policy target. It is a *hard* one-hot, not a visit
distribution, so its gradient is a plain cross-entropy towards "what a strong
human did"; that is the standard imitation signal and it is the reason a
human-pretrained net starts with sane move preferences instead of noise. And it is
inevitably noisier than an MCTS target — humans blunder, and Azul's floor-line
sacrifices look like blunders until several rounds later — which is why the Elo
floor in the fetcher matters more here than dataset size does.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ludometer.azul.engine import ACTION_SPACE, ENCODED_SIZE
from ludometer.human.convert import HumanGame
from ludometer.train.replay import ReplayBuffer

__all__ = ["DatasetStats", "add_game", "build_dataset"]


@dataclass
class DatasetStats:
    """What went into the file — printed by the CLI and asserted by the tests."""

    games: int = 0
    positions: int = 0
    tables: list[int] = field(default_factory=list)
    outcomes: dict[str, int] = field(
        default_factory=lambda: {"p0": 0, "p1": 0, "draw": 0}
    )

    def note(self, game: HumanGame) -> None:
        self.games += 1
        self.positions += len(game)
        self.tables.append(game.table_id)
        key = "draw" if game.outcome == 0 else ("p0" if game.outcome > 0 else "p1")
        self.outcomes[key] += 1


def one_hot_policies(actions: np.ndarray) -> np.ndarray:
    """``(T,)`` action ids -> ``(T, 180)`` one-hot float32 policy targets."""
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    if actions.size and (actions.min() < 0 or actions.max() >= ACTION_SPACE):
        raise ValueError(f"action id outside 0..{ACTION_SPACE - 1}")
    policies = np.zeros((len(actions), ACTION_SPACE), dtype=np.float32)
    policies[np.arange(len(actions)), actions] = 1.0
    return policies


def add_game(buffer: ReplayBuffer, game: HumanGame) -> int:
    """Append one converted game's rows to ``buffer``. Returns the row count."""
    if game.states.shape[1:] != (ENCODED_SIZE,):
        raise ValueError(
            f"states have shape {game.states.shape}, expected (T, {ENCODED_SIZE})"
        )
    buffer.games_added += 1
    return buffer.add(
        game.states,
        one_hot_policies(game.actions),
        game.values(),
        game.margins(),
        margin_mask=1.0,
        aux=game.aux,
        aux_mask=1.0,
        policy_mask=1.0,
    )


def build_dataset(
    games: Iterable[HumanGame],
    path: str | Path,
    capacity: int | None = None,
    seed: int = 0,
) -> DatasetStats:
    """Write every game in ``games`` to a ``replay.npz`` at ``path``.

    ``capacity`` defaults to the number of rows the games actually produce, so the
    file holds all of them and nothing is silently dropped by the ring; pass an
    explicit capacity to cap the dataset (the newest rows win, as in training).
    """
    materialised = list(games)
    total = sum(len(g) for g in materialised)
    buffer = ReplayBuffer(capacity=max(1, capacity or total), seed=seed)
    stats = DatasetStats()
    for game in materialised:
        add_game(buffer, game)
        stats.note(game)
    buffer.save(Path(path))
    return stats
