"""Entry point for a training run.

    uv run python -m ludometer.train.run --config configs/run1.json
    uv run python -m ludometer.train.run --resume runs/run1

Extra knobs (all optional): ``--max-games`` stops at an absolute game count (handy
for smoke tests and for extending a finished run), ``--run NAME`` overrides the
run name from the config, ``--runs-dir DIR`` moves the run root,
``--device/--workers`` override the config for a one-off run, and
``--pretrain runs/<run>/checkpoints/replay.npz`` warm-starts a fresh net on an
earlier run's replay buffer before any self-play happens.

The process is ``spawn``-safe: the self-play pool and the arena both fork fresh
interpreters, so all of the work happens under ``if __name__ == "__main__"``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ludometer.train.trainer import TrainConfig, Trainer, log_line

__all__ = ["main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ludometer.train.run", description="AlphaZero-style Azul trainer"
    )
    parser.add_argument("--config", type=Path, help="path to a configs/*.json file")
    parser.add_argument(
        "--resume", type=Path, help="run directory to continue (runs/<name>)"
    )
    parser.add_argument("--run", help="override the run name")
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="root for run directories"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        help="stop at this total number of self-play games (absolute, not extra)",
    )
    parser.add_argument(
        "--pretrain",
        type=Path,
        help="replay.npz to warm-start from before self-play (see Trainer.pretrain)",
    )
    parser.add_argument(
        "--pretrain-epochs", type=int, help="passes over the pretraining buffer"
    )
    parser.add_argument("--device", help="training device (auto|mps|cpu|cuda)")
    parser.add_argument("--workers", type=int, help="self-play worker processes")
    parser.add_argument("--seed", type=int, help="override the config seed")
    parser.add_argument("--note", help="free-form note stored in status.json")
    parser.add_argument(
        "--quiet", action="store_true", help="do not print progress lines"
    )
    return parser


def _load_config(args: argparse.Namespace) -> tuple[TrainConfig, Path, bool]:
    resume = args.resume is not None
    if resume:
        run_dir = Path(args.resume)
        config_path = Path(args.config) if args.config else run_dir / "config.json"
        if not config_path.exists():
            raise SystemExit(f"no config to resume from: {config_path}")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        data.pop("started", None)
        cfg = TrainConfig.from_dict(data)
    else:
        if not args.config:
            raise SystemExit("--config is required (or --resume a run directory)")
        cfg = TrainConfig.load(args.config)
        run_dir = Path(args.runs_dir) / (args.run or cfg.run)

    if args.run:
        cfg.run = args.run
    elif resume:
        cfg.run = run_dir.name
    if args.device:
        cfg.device = args.device
    if args.workers is not None:
        cfg.workers = args.workers
    if args.seed is not None:
        cfg.seed = args.seed
    if args.note:
        cfg.note = args.note
    if args.pretrain is not None:
        cfg.pretrain = str(args.pretrain)
        if not cfg.pretrain_epochs:
            cfg.pretrain_epochs = 1
    if args.pretrain_epochs is not None:
        cfg.pretrain_epochs = args.pretrain_epochs
    cfg.validate()
    return cfg, run_dir, resume


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg, run_dir, resume = _load_config(args)
    trainer = Trainer(cfg, run_dir, resume=resume, log=None if args.quiet else log_line)
    return trainer.run(max_games=args.max_games)


if __name__ == "__main__":
    sys.exit(main())
