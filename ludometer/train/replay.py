"""Uniform replay buffer: a fixed-capacity numpy ring of training positions.

One position is ``(encoded state [182], visit policy [180], value target)`` where
the value target is the game outcome **from the point of view of the player to
move** in that position (matching the value head's convention).

The buffer is part of the resumable state: :meth:`save` writes an ``.npz`` next to
the checkpoints and :meth:`load` restores contents, write position and counters.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ludometer.azul.engine import ACTION_SPACE, ENCODED_SIZE

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ludometer.train.selfplay import GameRecord

__all__ = ["ReplayBuffer"]


class ReplayBuffer:
    """Ring buffer with uniform sampling."""

    def __init__(
        self,
        capacity: int = 300_000,
        input_size: int = ENCODED_SIZE,
        action_space: int = ACTION_SPACE,
        seed: int = 0,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self.input_size = int(input_size)
        self.action_space = int(action_space)
        self.states = np.zeros((self.capacity, self.input_size), dtype=np.float32)
        self.policies = np.zeros((self.capacity, self.action_space), dtype=np.float32)
        self.values = np.zeros(self.capacity, dtype=np.float32)
        self.size = 0
        self.position = 0
        self.total_added = 0
        self.games_added = 0
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return self.size

    # -------------------------------------------------------------------- add
    def add(self, states: np.ndarray, policies: np.ndarray, values: np.ndarray) -> int:
        """Append a block of positions, overwriting the oldest ones when full."""
        states = np.asarray(states, dtype=np.float32).reshape(-1, self.input_size)
        policies = np.asarray(policies, dtype=np.float32).reshape(-1, self.action_space)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        n = len(states)
        if n != len(policies) or n != len(values):
            raise ValueError("states, policies and values must have equal length")
        if n == 0:
            return 0
        if n >= self.capacity:  # only the tail fits
            states = states[-self.capacity :]
            policies = policies[-self.capacity :]
            values = values[-self.capacity :]
            n = self.capacity
        end = self.position + n
        if end <= self.capacity:
            self.states[self.position : end] = states
            self.policies[self.position : end] = policies
            self.values[self.position : end] = values
        else:
            first = self.capacity - self.position
            self.states[self.position :] = states[:first]
            self.policies[self.position :] = policies[:first]
            self.values[self.position :] = values[:first]
            rest = n - first
            self.states[:rest] = states[first:]
            self.policies[:rest] = policies[first:]
            self.values[:rest] = values[first:]
        self.position = end % self.capacity
        self.size = min(self.capacity, self.size + n)
        self.total_added += n
        return n

    def add_game(self, record: GameRecord) -> int:
        self.games_added += 1
        return self.add(record.states, record.policies, record.values)

    # ----------------------------------------------------------------- sample
    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Uniform sample with replacement; raises if the buffer is empty."""
        if self.size == 0:
            raise ValueError("cannot sample an empty buffer")
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return self.states[idx], self.policies[idx], self.values[idx]

    # -------------------------------------------------------------- persistence
    def save(self, path: str | os.PathLike[str]) -> Path:
        """Atomically dump the filled part of the ring (uncompressed npz)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".npz.tmp")
        order = self._ordered_indices()
        # a partially filled ring is contiguous: slice it instead of copying
        take = (
            (lambda a: a[: self.size])
            if self.size < self.capacity
            else (lambda a: a[order])
        )
        with tmp.open("wb") as fh:
            np.savez(
                fh,
                states=take(self.states),
                policies=take(self.policies),
                values=take(self.values),
                meta=np.array(
                    [
                        self.capacity,
                        self.size,
                        self.total_added,
                        self.games_added,
                        self.seed,
                    ],
                    dtype=np.int64,
                ),
            )
        os.replace(tmp, target)
        return target

    def _ordered_indices(self) -> np.ndarray:
        """Indices oldest-to-newest, so a reload keeps the ring order."""
        if self.size < self.capacity:
            return np.arange(self.size)
        return np.concatenate(
            [np.arange(self.position, self.capacity), np.arange(self.position)]
        )

    def load(self, path: str | os.PathLike[str]) -> int:
        """Refill from :meth:`save` output; keeps this buffer's capacity."""
        with np.load(path) as data:
            states = data["states"]
            policies = data["policies"]
            values = data["values"]
            meta = data["meta"] if "meta" in data.files else None
        self.size = 0
        self.position = 0
        self.total_added = 0
        self.games_added = 0
        self.add(states, policies, values)
        if meta is not None and len(meta) >= 5:
            self.total_added = int(meta[2])
            self.games_added = int(meta[3])
        return self.size

    def stats(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "capacity": self.capacity,
            "total_added": self.total_added,
            "games_added": self.games_added,
        }
