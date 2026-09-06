"""Distill a big teacher checkpoint into a small student architecture.

    uv run python -m ludometer.train.distill \
        --teacher runs/ft1/checkpoints/ft-002000.pt \
        --student-init runs/run4/checkpoints/ckpt-037888.pt \
        --states runs/run5/checkpoints/replay.npz data/human/replay_full.npz \
        --out runs/ds1 --save-at 1000,2000,3000

Why: the human-fine-tuned run5 net (ft2000) is the strongest policy we have at
equal search, but at 4x the parameters it *loses* at equal think time — the
site's budget. Distillation trains the deployed-size net to imitate the big
net's outputs (soft policy via KL, value/margin via MSE) over a large state
collection, which transfers quality without the compute cost. The student
initializes from the currently deployed checkpoint so distillation only has to
move it toward the teacher, not rebuild it.

Teacher labels are computed once up front (``--teacher-device``, default mps,
falling back to cpu) and cached in memory; the student then trains on cpu by
default so a concurrent MPS training run keeps its device.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ludometer.train.net import load_net, save_checkpoint

__all__ = ["main"]


def _load_states(paths: list[Path], cap_per_file: int) -> np.ndarray:
    parts = []
    for path in paths:
        with np.load(path) as z:
            states = z["states"]
            n = len(states)
            take = min(n, cap_per_file) if cap_per_file else n
            # newest rows are at the end for our ring-buffer files
            parts.append(np.asarray(states[n - take :], dtype=np.float32))
    return np.concatenate(parts)


def _teacher_labels(teacher, states: np.ndarray, device: str, batch: int = 2048):
    n = len(states)
    pol = np.empty((n, 180), dtype=np.float16)
    val = np.empty(n, dtype=np.float32)
    mar = np.zeros(n, dtype=np.float32)
    has_margin = getattr(teacher, "has_margin", False)
    with torch.no_grad():
        for i in range(0, n, batch):
            x = torch.from_numpy(states[i : i + batch]).to(device)
            out = teacher.forward_aux(x)
            logits, value, margin = out[0], out[1], out[2]
            pol[i : i + batch] = (
                torch.softmax(logits, dim=1).to("cpu", torch.float16).numpy()
            )
            val[i : i + batch] = value.float().cpu().numpy()
            if margin is not None and has_margin:
                mar[i : i + batch] = margin.float().cpu().numpy()
    return pol, val, mar


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ludometer.train.distill")
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument(
        "--student-init", type=Path, default=None, help="checkpoint to start from"
    )
    parser.add_argument(
        "--student-config",
        type=Path,
        default=None,
        help="net_config JSON for a fresh student (alternative to --student-init)",
    )
    parser.add_argument("--states", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-at", default="1000,2000,3000")
    parser.add_argument(
        "--cap-per-file", type=int, default=250_000, help="newest N states per file"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--teacher-device", default="mps")
    parser.add_argument("--device", default="cpu", help="student training device")
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args(argv)

    t_dev = args.teacher_device
    if t_dev == "mps" and not torch.backends.mps.is_available():
        t_dev = "cpu"
    teacher, _ = load_net(args.teacher, device=t_dev)
    states = _load_states(list(args.states), args.cap_per_file)
    print(f"teacher {args.teacher} on {t_dev}; {len(states):,} states")
    t0 = time.monotonic()
    pol, val, mar = _teacher_labels(teacher, states, t_dev)
    print(f"teacher labels in {time.monotonic() - t0:.0f}s")
    del teacher

    if args.student_init is not None:
        student, _ = load_net(args.student_init, device=args.device)
    elif args.student_config is not None:
        from ludometer.train.net import make_net

        student = make_net(json.loads(args.student_config.read_text()))
        student.to(args.device)
    else:
        raise SystemExit("pass --student-init or --student-config")
    student.train()
    has_margin = getattr(student, "has_margin", False)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    save_at = sorted(int(s) for s in args.save_at.split(",") if s.strip())
    ckpt_dir = args.out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "distill.jsonl"
    sums = np.zeros(3)
    last = 0
    t0 = time.monotonic()
    for step in range(1, save_at[-1] + 1):
        idx = rng.integers(0, len(states), size=args.batch_size)
        x = torch.from_numpy(states[idx]).to(args.device)
        target_p = torch.from_numpy(pol[idx].astype(np.float32)).to(args.device)
        target_v = torch.from_numpy(val[idx]).to(args.device)
        out = student.forward_aux(x)
        logits, value, margin = out[0], out[1], out[2]
        logp = torch.log_softmax(logits, dim=1)
        # KL(teacher || student) up to the teacher-entropy constant
        loss_p = -(target_p * logp).sum(dim=1).mean()
        loss_v = torch.nn.functional.mse_loss(value, target_v)
        loss_m = torch.zeros((), device=args.device)
        if margin is not None and has_margin:
            target_m = torch.from_numpy(mar[idx]).to(args.device)
            loss_m = torch.nn.functional.mse_loss(margin, target_m)
        loss = loss_p + args.value_weight * loss_v + args.margin_weight * loss_m
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()
        sums += [float(loss_p.detach()), float(loss_v.detach()), float(loss_m.detach())]
        if step in save_at:
            name = f"ds-{step:06d}"
            student.eval()
            save_checkpoint(
                ckpt_dir / f"{name}.pt",
                student,
                extra={
                    "run": args.out.name,
                    "checkpoint": name,
                    "distilled_from": str(args.teacher),
                    "student_init": str(args.student_init or args.student_config),
                },
            )
            student.train()
            done = step - last
            entry = {
                "step": step,
                "loss_p": round(float(sums[0]) / done, 4),
                "loss_v": round(float(sums[1]) / done, 4),
                "loss_m": round(float(sums[2]) / done, 4),
                "t": round(time.monotonic() - t0, 1),
            }
            with open(log_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
            print(f"saved {name}: {entry}")
            sums[:] = 0
            last = step
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
