"""Uniform replay buffer: a fixed-capacity numpy ring of training positions.

One position is ``(encoded state [182], visit policy [180], value target, margin
target, margin mask)`` where the value target is the game outcome **from the point
of view of the player to move** in that position (matching the value head's
convention) and the margin target is ``tanh(final score diff / 20)`` in the same
frame.

The margin columns are run4's addition and are **optional data**, which is what
``margin_mask`` is for: a position loaded from a run1-run3 ``replay.npz`` has no
margin to learn from, so its mask is 0 and the margin loss simply skips it (see
:meth:`ludometer.train.trainer.Trainer.pretrain`). Old files therefore load
unchanged, and files this module writes stay readable by anything that only asks
for ``states``/``policies``/``values``.

There is one exception worth knowing about: run1-run3 wrote a *blended* value,
``0.85 * outcome + 0.15 * tanh(diff / 20)``. Those three bands do not overlap, so
:meth:`load` can be asked (``unblend=0.15``) to split that number back into the
exact outcome and the exact margin — which turns run3's 500k-position buffer into
real supervision for the new head instead of 500k masked-out rows.

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

__all__ = ["ReplayBuffer", "unblend_values"]


def unblend_values(values: np.ndarray, weight: float) -> tuple[np.ndarray, np.ndarray]:
    """Split ``(1 - w) * outcome + w * margin`` back into ``(outcome, margin)``.

    This is exact, not an approximation, because the three outcome bands cannot
    overlap: with ``w = 0.15`` a win lands in ``[0.85, 1.0]``, a loss in
    ``[-1.0, -0.85]`` and a draw in ``(-0.15, 0.15)`` (the winner of an Azul game
    always has the higher score, or the same score and more completed rows, so a
    win's margin term is never negative). Reading the outcome off the sign with a
    dead band therefore recovers it exactly, and the margin follows by algebra —
    it was ``tanh(diff / 20)`` when it went in, which is precisely the run4 margin
    target. Verified against runs/run3: 486k wins/losses, 13.5k draws, nothing in
    between.
    """
    v = np.asarray(values, dtype=np.float32)
    w = float(weight)
    if not 0.0 < w < 0.5:
        raise ValueError(f"unblend weight must be in (0, 0.5), got {w}")
    cut = 0.5  # the midpoint of the draw band's edge (w) and the win band's (1 - w)
    outcome = np.where(v > cut, 1.0, np.where(v < -cut, -1.0, 0.0)).astype(np.float32)
    margin = np.clip((v - (1.0 - w) * outcome) / w, -1.0, 1.0).astype(np.float32)
    return outcome, margin


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
        self.margins = np.zeros(self.capacity, dtype=np.float32)
        # 1.0 where `margins` is a real target, 0.0 where it is a placeholder.
        self.margin_mask = np.zeros(self.capacity, dtype=np.float32)
        self.size = 0
        self.position = 0
        self.total_added = 0
        self.games_added = 0
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return self.size

    # -------------------------------------------------------------------- add
    def add(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        margins: np.ndarray | None = None,
        margin_mask: np.ndarray | float | None = None,
    ) -> int:
        """Append a block of positions, overwriting the oldest ones when full.

        ``margins`` may be omitted (older callers, and buffers restored from a
        pre-run4 file): the block is then stored with a zero margin and a zero
        mask, i.e. "no margin target here", and the margin loss ignores it.
        """
        states = np.asarray(states, dtype=np.float32).reshape(-1, self.input_size)
        policies = np.asarray(policies, dtype=np.float32).reshape(-1, self.action_space)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        n = len(states)
        if n != len(policies) or n != len(values):
            raise ValueError("states, policies and values must have equal length")
        if margins is None:
            margins = np.zeros(n, dtype=np.float32)
            mask = np.zeros(n, dtype=np.float32)
        else:
            margins = np.asarray(margins, dtype=np.float32).reshape(-1)
            if len(margins) != n:
                raise ValueError("margins must have the same length as states")
            if margin_mask is None:
                mask = np.ones(n, dtype=np.float32)
            elif np.isscalar(margin_mask):
                mask = np.full(n, float(margin_mask), dtype=np.float32)
            else:
                mask = np.asarray(margin_mask, dtype=np.float32).reshape(-1)
                if len(mask) != n:
                    raise ValueError("margin_mask must have the same length as states")
        if n == 0:
            return 0
        if n >= self.capacity:  # only the tail fits
            states = states[-self.capacity :]
            policies = policies[-self.capacity :]
            values = values[-self.capacity :]
            margins = margins[-self.capacity :]
            mask = mask[-self.capacity :]
            n = self.capacity
        end = self.position + n
        if end <= self.capacity:
            self.states[self.position : end] = states
            self.policies[self.position : end] = policies
            self.values[self.position : end] = values
            self.margins[self.position : end] = margins
            self.margin_mask[self.position : end] = mask
        else:
            first = self.capacity - self.position
            self.states[self.position :] = states[:first]
            self.policies[self.position :] = policies[:first]
            self.values[self.position :] = values[:first]
            self.margins[self.position :] = margins[:first]
            self.margin_mask[self.position :] = mask[:first]
            rest = n - first
            self.states[:rest] = states[first:]
            self.policies[:rest] = policies[first:]
            self.values[:rest] = values[first:]
            self.margins[:rest] = margins[first:]
            self.margin_mask[:rest] = mask[first:]
        self.position = end % self.capacity
        self.size = min(self.capacity, self.size + n)
        self.total_added += n
        return n

    def add_game(self, record: GameRecord) -> int:
        self.games_added += 1
        margins = getattr(record, "margins", None)
        return self.add(record.states, record.policies, record.values, margins)

    # ----------------------------------------------------------------- sample
    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Uniform sample with replacement; raises if the buffer is empty.

        Returns ``(states, policies, values, margins, margin_mask)``.
        """
        if self.size == 0:
            raise ValueError("cannot sample an empty buffer")
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return (
            self.states[idx],
            self.policies[idx],
            self.values[idx],
            self.margins[idx],
            self.margin_mask[idx],
        )

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
                # extra arrays, never fewer: a reader that only knows about the
                # first three keys (an older checkout, the dashboard) is fine.
                margins=take(self.margins),
                margin_mask=take(self.margin_mask),
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

    def load(self, path: str | os.PathLike[str], unblend: float = 0.0) -> int:
        """Refill from :meth:`save` output; keeps this buffer's capacity.

        ``unblend`` is the ``value_score_weight`` the file was *written* with. It
        is 0 (off) for anything this version writes; pass run3's 0.15 to recover
        the pure outcome and the exact margin from that run's blended value —
        see :func:`unblend_values` for why the split is exact and not a guess.
        A file that already carries a ``margins`` array is never unblended.
        """
        with np.load(path) as data:
            states = data["states"]
            policies = data["policies"]
            values = data["values"]
            margins = data["margins"] if "margins" in data.files else None
            mask = data["margin_mask"] if "margin_mask" in data.files else None
            meta = data["meta"] if "meta" in data.files else None
        if margins is None and unblend > 0.0:
            values, margins = unblend_values(values, unblend)
            mask = None  # every recovered row is a real target
        self.size = 0
        self.position = 0
        self.total_added = 0
        self.games_added = 0
        self.add(states, policies, values, margins, mask)
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
            # how many of the stored positions can train the margin head
            "margin_targets": int(self.margin_mask[: self.size].sum()),
        }
