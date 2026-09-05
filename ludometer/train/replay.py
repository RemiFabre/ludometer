"""Uniform replay buffer: a fixed-capacity numpy ring of training positions.

One position is ``(encoded state [182], visit policy [180], value target, margin
target, margin mask, final-wall bits, aux mask, policy mask)`` where the value
target is the game outcome **from the point of view of the player to move** in
that position (matching the value head's convention) and the margin target is
``tanh(final score diff / 20)`` in the same frame.

Everything past the first three columns is **optional data**, and every optional
column has a mask saying whether it is real:

* ``margins`` / ``margin_mask`` — run4. A position loaded from a run1-run3
  ``replay.npz`` has no margin to learn from, so its mask is 0 and the margin loss
  skips it (see :meth:`ludometer.train.trainer.Trainer.pretrain`);
* ``aux`` / ``aux_mask`` — run6. The 30 bits of
  :meth:`~ludometer.azul.engine.AzulState.wall_summary` for both players' *final*
  walls, in the player-to-move frame, stored **packed**: ``np.packbits`` makes it
  4 bytes per position instead of 120, which is 2 MB rather than 60 MB over a
  500k-position buffer. Anything written before run6 loads with a zero mask;
* ``policy_mask`` — run6's playout-cap randomization
  (:mod:`ludometer.train.selfplay_batched`). A position searched cheaply has a
  perfectly good value/margin/aux label and a *shallow* visit distribution, so it
  is stored with a zeroed policy and a 0 mask and contributes no policy gradient.
  Anything written before run6 loads with a mask of **1** — those positions all
  carry a real policy target.

Old files therefore load unchanged, and files this module writes stay readable by
anything that only asks for ``states``/``policies``/``values``.

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
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from ludometer.azul.engine import ACTION_SPACE, ENCODED_SIZE

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ludometer.train.selfplay import GameRecord

__all__ = [
    "AUX_BITS",
    "AUX_BYTES",
    "Batch",
    "ReplayBuffer",
    "pack_aux",
    "unblend_values",
    "unpack_aux",
]

#: width of the auxiliary final-wall target (see net2.AUX_OUTPUTS) and of its
#: packed form. Spelled out here rather than imported so the buffer keeps its
#: "numpy only, no torch" property.
AUX_BITS = 30
AUX_BYTES = (AUX_BITS + 7) // 8


def pack_aux(bits: np.ndarray) -> np.ndarray:
    """``(n, 30)`` 0/1 -> ``(n, 4)`` uint8; already-packed input passes through."""
    arr = np.asarray(bits)
    if arr.ndim != 2:
        raise ValueError(f"aux targets must be 2-D, got shape {arr.shape}")
    if arr.shape[1] == AUX_BYTES and arr.dtype == np.uint8:
        return arr
    if arr.shape[1] != AUX_BITS:
        raise ValueError(f"aux targets must have {AUX_BITS} columns, got {arr.shape}")
    return np.packbits(arr.astype(np.uint8), axis=1)


def unpack_aux(packed: np.ndarray) -> np.ndarray:
    """``(n, 4)`` uint8 -> ``(n, 30)`` float32 0/1 (the BCE target)."""
    arr = np.asarray(packed, dtype=np.uint8).reshape(-1, AUX_BYTES)
    return np.unpackbits(arr, axis=1)[:, :AUX_BITS].astype(np.float32)


class Batch(NamedTuple):
    """One training minibatch — what :meth:`ReplayBuffer.sample` hands back."""

    states: np.ndarray  # (B, 182) float32
    policies: np.ndarray  # (B, 180) float32
    values: np.ndarray  # (B,) float32
    margins: np.ndarray  # (B,) float32
    margin_mask: np.ndarray  # (B,) float32
    aux: np.ndarray  # (B, 30) float32, unpacked
    aux_mask: np.ndarray  # (B,) float32
    policy_mask: np.ndarray  # (B,) float32
    search_values: np.ndarray  # (B,) float32, the search's root value estimate
    search_mask: np.ndarray  # (B,) float32, 1 where `search_values` is real


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
        # run6: packed final-wall bits and their own mask, same convention.
        self.aux = np.zeros((self.capacity, AUX_BYTES), dtype=np.uint8)
        self.aux_mask = np.zeros(self.capacity, dtype=np.float32)
        # run6: 0.0 for a position whose search was the cheap one, so `policies`
        # holds no target. Defaults to 1.0 because every older position has one.
        self.policy_mask = np.ones(self.capacity, dtype=np.float32)
        # 2026-09-05: the search's root value per position, its own mask. Files
        # written before it load with a zero mask, so a run that mixes it into
        # the value target falls back to the outcome on those rows.
        self.search_values = np.zeros(self.capacity, dtype=np.float32)
        self.search_mask = np.zeros(self.capacity, dtype=np.float32)
        self.size = 0
        self.position = 0
        self.total_added = 0
        self.games_added = 0
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return self.size

    # -------------------------------------------------------------------- add
    def _column(
        self, value: np.ndarray | float | None, n: int, default: float, name: str
    ) -> np.ndarray:
        """A length-``n`` float32 mask from an array, a scalar or ``None``."""
        if value is None:
            return np.full(n, default, dtype=np.float32)
        if np.isscalar(value):
            return np.full(n, float(value), dtype=np.float32)  # type: ignore[arg-type]
        out = np.asarray(value, dtype=np.float32).reshape(-1)
        if len(out) != n:
            raise ValueError(f"{name} must have the same length as states")
        return out

    def add(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        margins: np.ndarray | None = None,
        margin_mask: np.ndarray | float | None = None,
        aux: np.ndarray | None = None,
        aux_mask: np.ndarray | float | None = None,
        policy_mask: np.ndarray | float | None = None,
        search_values: np.ndarray | None = None,
        search_mask: np.ndarray | float | None = None,
    ) -> int:
        """Append a block of positions, overwriting the oldest ones when full.

        ``margins`` and ``aux`` may be omitted (older callers, and buffers
        restored from a pre-run4 / pre-run6 file): the block is then stored with a
        zero target and a zero mask, i.e. "no target here", and that head's loss
        ignores it. ``policy_mask`` defaults the other way, to 1: a caller that
        does not mention it is a caller from before cheap searches existed, and
        every position it adds carries a real visit distribution.

        ``aux`` is accepted either unpacked ``(n, 30)`` 0/1 or already packed
        ``(n, 4)`` uint8.
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
            margins = self._column(margins, n, 0.0, "margins")
            mask = self._column(margin_mask, n, 1.0, "margin_mask")
        if aux is None:
            aux_bits = np.zeros((n, AUX_BYTES), dtype=np.uint8)
            a_mask = np.zeros(n, dtype=np.float32)
        else:
            aux_bits = pack_aux(aux)
            if len(aux_bits) != n:
                raise ValueError("aux must have the same length as states")
            a_mask = self._column(aux_mask, n, 1.0, "aux_mask")
        p_mask = self._column(policy_mask, n, 1.0, "policy_mask")
        if search_values is None:
            s_vals = np.zeros(n, dtype=np.float32)
            s_mask = np.zeros(n, dtype=np.float32)
        else:
            s_vals = self._column(search_values, n, 0.0, "search_values")
            s_mask = self._column(search_mask, n, 1.0, "search_mask")
        if n == 0:
            return 0
        blocks = (
            (self.states, states),
            (self.policies, policies),
            (self.values, values),
            (self.margins, margins),
            (self.margin_mask, mask),
            (self.aux, aux_bits),
            (self.aux_mask, a_mask),
            (self.policy_mask, p_mask),
            (self.search_values, s_vals),
            (self.search_mask, s_mask),
        )
        if n >= self.capacity:  # only the tail fits
            blocks = tuple((dest, src[-self.capacity :]) for dest, src in blocks)
            n = self.capacity
        end = self.position + n
        if end <= self.capacity:
            for dest, src in blocks:
                dest[self.position : end] = src
        else:
            first = self.capacity - self.position
            for dest, src in blocks:
                dest[self.position :] = src[:first]
                dest[: n - first] = src[first:]
        self.position = end % self.capacity
        self.size = min(self.capacity, self.size + n)
        self.total_added += n
        return n

    def add_game(self, record: GameRecord) -> int:
        self.games_added += 1
        return self.add(
            record.states,
            record.policies,
            record.values,
            getattr(record, "margins", None),
            aux=getattr(record, "aux", None),
            policy_mask=getattr(record, "policy_mask", None),
            search_values=getattr(record, "search_values", None),
            search_mask=getattr(record, "search_mask", None),
        )

    # ----------------------------------------------------------------- sample
    def sample(self, batch_size: int) -> Batch:
        """Uniform sample with replacement; raises if the buffer is empty.

        The aux column is unpacked here rather than stored unpacked: 256 rows of
        ``np.unpackbits`` is microseconds against a minibatch that then crosses to
        the GPU, and the buffer stays 30x smaller in RAM and on disk.
        """
        if self.size == 0:
            raise ValueError("cannot sample an empty buffer")
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return Batch(
            self.states[idx],
            self.policies[idx],
            self.values[idx],
            self.margins[idx],
            self.margin_mask[idx],
            unpack_aux(self.aux[idx]),
            self.aux_mask[idx],
            self.policy_mask[idx],
            self.search_values[idx],
            self.search_mask[idx],
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
                aux=take(self.aux),  # packed: 4 bytes a position
                aux_mask=take(self.aux_mask),
                policy_mask=take(self.policy_mask),
                search_values=take(self.search_values),
                search_mask=take(self.search_mask),
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
            aux = data["aux"] if "aux" in data.files else None
            aux_mask = data["aux_mask"] if "aux_mask" in data.files else None
            # A file without the key predates cheap searches: every row has a
            # policy target, which is what `add`'s default already says.
            p_mask = data["policy_mask"] if "policy_mask" in data.files else None
            s_vals = data["search_values"] if "search_values" in data.files else None
            s_mask = data["search_mask"] if "search_mask" in data.files else None
            meta = data["meta"] if "meta" in data.files else None
        if margins is None and unblend > 0.0:
            values, margins = unblend_values(values, unblend)
            mask = None  # every recovered row is a real target
        self.size = 0
        self.position = 0
        self.total_added = 0
        self.games_added = 0
        self.add(
            states,
            policies,
            values,
            margins,
            mask,
            aux=aux,
            aux_mask=aux_mask,
            policy_mask=p_mask,
            search_values=s_vals,
            search_mask=s_mask,
        )
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
            # how many of the stored positions can train each optional head
            "margin_targets": int(self.margin_mask[: self.size].sum()),
            "aux_targets": int(self.aux_mask[: self.size].sum()),
            "policy_targets": int(self.policy_mask[: self.size].sum()),
            "search_targets": int(self.search_mask[: self.size].sum()),
        }
