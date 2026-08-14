"""Agent specs -> instances, shared by the GUI, arena, and eval.

Spec strings:
    "random" | "greedy" | "heuristic"
    "mcts:<checkpoint_path>?sims=<n>"   (neural agent; requires ludometer.train)
    "best?sims=<n>"                     (highest-Elo checkpoint on disk)

``best`` resolves at load time by scanning ``runs/*/elo.jsonl`` for the
highest-rated checkpoint whose ``.pt`` file still exists, so "play the strongest
model" keeps working while a run is training: every new game picks up the newest
strongest checkpoint. The resolved choice is attached to the agent as
``agent.spec_info`` (kind/path/run/checkpoint/elo/sims) so the GUI can tell the
human who they are facing.

``load_agent`` raises ``ValueError`` for anything it cannot parse and lets the
underlying error through for a checkpoint that exists but cannot be loaded, so
callers (the GUI) can show the reason verbatim.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NamedTuple

from ludometer.agents.greedy import GreedyAgent
from ludometer.agents.heuristic import HeuristicAgent
from ludometer.agents.random_agent import RandomAgent

__all__ = [
    "BASELINES",
    "BEST_SIMS",
    "BEST_SPEC",
    "DEFAULT_SIMS",
    "BestCheckpoint",
    "find_best_checkpoint",
    "load_agent",
    "runs_dir",
]

BASELINES = ("random", "greedy", "heuristic")
DEFAULT_SIMS = 200
BEST_SPEC = "best"
BEST_SIMS = 400  # a human-vs-AI game wants strength; the arena wants throughput
RUNS_ENV = "LUDOMETER_RUNS_DIR"

# repo layout: <root>/ludometer/agents/registry.py and <root>/runs/
_REPO_ROOT = Path(__file__).resolve().parents[2]


class BestCheckpoint(NamedTuple):
    """The highest-Elo checkpoint found on disk."""

    path: Path
    elo: float
    run: str
    ckpt: str


def runs_dir() -> Path:
    """Where runs live: ``$LUDOMETER_RUNS_DIR`` if set, else ``<repo>/runs``."""
    override = os.environ.get(RUNS_ENV)
    return Path(override).expanduser() if override else _REPO_ROOT / "runs"


def find_best_checkpoint(root: str | os.PathLike[str] | None = None) -> BestCheckpoint:
    """Best rated checkpoint under ``root`` (default :func:`runs_dir`).

    Scans every ``<root>/*/elo.jsonl``, keeps the highest ``elo`` whose
    ``checkpoints/<ckpt>.pt`` exists, and returns ``(path, elo, run, ckpt)``.
    Unparsable or half-written JSONL lines are skipped — the trainer appends to
    these files while we read them. Raises ``FileNotFoundError`` when nothing
    usable is there yet.
    """
    base = Path(root).expanduser() if root is not None else runs_dir()
    best: BestCheckpoint | None = None
    logs = sorted(base.glob("*/elo.jsonl")) if base.is_dir() else []
    for log in logs:
        run_dir = log.parent
        try:
            lines = log.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue  # a torn last line while the trainer appends
            if not isinstance(record, dict):
                continue
            name = record.get("ckpt")
            elo = record.get("elo")
            if not isinstance(name, str) or not name:
                continue
            if isinstance(elo, bool) or not isinstance(elo, (int, float)):
                continue
            if best is not None and float(elo) <= best.elo:
                continue
            path = run_dir / "checkpoints" / f"{name}.pt"
            if not path.is_file():
                continue
            best = BestCheckpoint(path, float(elo), run_dir.name, name)
    if best is None:
        raise FileNotFoundError(
            f"no rated checkpoints found under {base} — train a run first "
            "(uv run ludometer-train ...) or pick a baseline opponent; "
            "'best' needs a runs/<run>/elo.jsonl entry whose "
            "checkpoints/<ckpt>.pt is still on disk"
        )
    return best


def _parse_sims(query: str, default: int) -> int:
    """Read ``sims=<n>`` out of a spec query string."""
    sims = default
    for part in query.split("&"):
        if not part:
            continue
        key, _, value = part.partition("=")
        if key != "sims":
            raise ValueError(f"unknown option {key!r} (only sims= is supported)")
        try:
            sims = int(value)
        except ValueError:
            raise ValueError(f"sims must be an integer, got {value!r}") from None
        if sims < 1:
            raise ValueError(f"sims must be >= 1, got {sims}")
    return sims


def load_agent(spec: str, seed: int | None = None):
    """Build the agent described by ``spec`` (see the module docstring)."""
    if not isinstance(spec, str):
        raise TypeError(f"agent spec must be a string, got {type(spec).__name__}")
    spec = spec.strip()
    # the baselines take `seed` as a plain int; None means "system entropy"
    kwargs = {} if seed is None else {"seed": int(seed)}
    if spec == "random":
        return RandomAgent(**kwargs)
    if spec == "greedy":
        return GreedyAgent(**kwargs)
    if spec == "heuristic":
        return HeuristicAgent(**kwargs)
    if spec == BEST_SPEC or spec.startswith(BEST_SPEC + "?"):
        sims = _parse_sims(spec.partition("?")[2], BEST_SIMS)
        best = find_best_checkpoint()
        from ludometer.train.mcts_agent import MCTSAgent  # lazy: needs torch

        agent = MCTSAgent.from_checkpoint(
            best.path, sims=sims, seed=seed, name=f"best:{best.ckpt}"
        )
        agent.spec_info = {
            "kind": "best",
            "spec": spec,
            "resolved_spec": f"mcts:{best.path}?sims={sims}",
            "path": str(best.path),
            "run": best.run,
            "checkpoint": best.ckpt,
            "elo": best.elo,
            "sims": sims,
        }
        return agent
    if spec.startswith("mcts:"):
        rest = spec[len("mcts:") :]
        path, _, query = rest.partition("?")
        if not path:
            raise ValueError("mcts spec needs a checkpoint path: mcts:<path>?sims=<n>")
        sims = _parse_sims(query, DEFAULT_SIMS)
        from ludometer.train.mcts_agent import MCTSAgent  # lazy: needs torch

        agent = MCTSAgent.from_checkpoint(path, sims=sims, seed=seed)
        agent.spec_info = {
            "kind": "mcts",
            "spec": spec,
            "resolved_spec": spec,
            "path": path,
            "checkpoint": Path(path).stem,
            "sims": sims,
        }
        return agent
    raise ValueError(
        f"unknown agent spec: {spec!r}; expected one of {', '.join(BASELINES)}, "
        f"{BEST_SPEC}[?sims=<n>] or mcts:<checkpoint>?sims=<n>"
    )
