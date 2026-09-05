"""A block of self-play games as one file: ``GameRecord`` <-> ``.npz``.

The trainer ingests games one at a time (``ReplayBuffer.add_game``) and reads
per-game diagnostics (``moves``, ``decisions``, ``truncated``), so a shard keeps
the game boundaries: the position columns are concatenated, and a second set of
per-game columns says where each game starts and what it was. ``read_shard``
returns exactly the records ``write_shard`` was given, array for array.

``meta`` is a free JSON dict (weights version, sims, job tag) stored as one
string; :func:`peek_meta` reads it without touching the big arrays.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ludometer.train.selfplay import GameRecord

__all__ = ["peek_meta", "read_shard", "write_shard"]

FORMAT = 1


def write_shard(
    path: str | os.PathLike[str], records: list[GameRecord], meta: dict[str, Any]
) -> Path:
    """Write ``records`` (at least one) and ``meta`` atomically, compressed."""
    if not records:
        raise ValueError("a shard needs at least one game")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    lengths = np.array([len(r) for r in records], dtype=np.int64)
    aux_width = max((r.aux.shape[1] if r.aux.ndim == 2 else 0) for r in records)
    with tmp.open("wb") as fh:
        np.savez_compressed(
            fh,
            format=np.array([FORMAT], dtype=np.int64),
            meta=np.array(json.dumps(meta)),
            lengths=lengths,
            states=np.concatenate([r.states for r in records]).astype(np.float32),
            policies=np.concatenate([r.policies for r in records]).astype(np.float32),
            values=np.concatenate([r.values for r in records]).astype(np.float32),
            margins=np.concatenate([r.margins for r in records]).astype(np.float32),
            aux=np.concatenate(
                [
                    np.asarray(r.aux, dtype=np.uint8).reshape(len(r), aux_width)
                    for r in records
                ]
            ),
            policy_mask=np.concatenate([r.policy_mask for r in records]).astype(
                np.float32
            ),
            search_values=np.concatenate(
                [
                    r.search_values
                    if r.search_values is not None
                    else np.zeros(len(r), dtype=np.float32)
                    for r in records
                ]
            ).astype(np.float32),
            search_mask=np.concatenate(
                [
                    r.search_mask
                    if r.search_mask is not None
                    else np.zeros(len(r), dtype=np.float32)
                    for r in records
                ]
            ).astype(np.float32),
            outcome=np.array([r.outcome for r in records], dtype=np.float32),
            scores=np.array([r.scores for r in records], dtype=np.int64).reshape(-1, 2),
            moves=np.array([r.moves for r in records], dtype=np.int64),
            rounds=np.array([r.rounds for r in records], dtype=np.int64),
            seed=np.array([r.seed for r in records], dtype=np.int64),
            decisions=np.array([r.decisions for r in records], dtype=np.int64),
            evals=np.array([r.evals for r in records], dtype=np.int64),
            duration=np.array([r.duration for r in records], dtype=np.float64),
            truncated=np.array([r.truncated for r in records], dtype=np.bool_),
        )
    os.replace(tmp, target)
    return target


def peek_meta(path: str | os.PathLike[str]) -> dict[str, Any]:
    with np.load(path) as z:
        return json.loads(str(z["meta"]))


def read_shard(path: str | os.PathLike[str]) -> tuple[list[GameRecord], dict[str, Any]]:
    with np.load(path) as z:
        fmt = int(z["format"][0])
        if fmt != FORMAT:
            raise ValueError(f"shard format {fmt} is not {FORMAT}: {path}")
        meta = json.loads(str(z["meta"]))
        lengths = z["lengths"]
        cols = {
            k: z[k]
            for k in (
                "states",
                "policies",
                "values",
                "margins",
                "aux",
                "policy_mask",
                "search_values",
                "search_mask",
            )
        }
        per_game = {
            k: z[k]
            for k in (
                "outcome",
                "scores",
                "moves",
                "rounds",
                "seed",
                "decisions",
                "evals",
                "duration",
                "truncated",
            )
        }
    records: list[GameRecord] = []
    at = 0
    for i, n in enumerate(lengths):
        sl = slice(at, at + int(n))
        records.append(
            GameRecord(
                states=cols["states"][sl],
                policies=cols["policies"][sl],
                values=cols["values"][sl],
                margins=cols["margins"][sl],
                aux=cols["aux"][sl],
                policy_mask=cols["policy_mask"][sl],
                search_values=cols["search_values"][sl],
                search_mask=cols["search_mask"][sl],
                outcome=float(per_game["outcome"][i]),
                scores=(int(per_game["scores"][i][0]), int(per_game["scores"][i][1])),
                moves=int(per_game["moves"][i]),
                rounds=int(per_game["rounds"][i]),
                seed=int(per_game["seed"][i]),
                decisions=int(per_game["decisions"][i]),
                evals=int(per_game["evals"][i]),
                duration=float(per_game["duration"][i]),
                truncated=bool(per_game["truncated"][i]),
            )
        )
        at += int(n)
    return records, meta
