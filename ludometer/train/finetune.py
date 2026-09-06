"""Fine-tune an existing checkpoint on human games, with self-play rehearsal.

    uv run python -m ludometer.train.finetune \
        --base runs/run5/checkpoints/ckpt-006912.pt \
        --human data/human/replay_full.npz data/faience/human_windraw.npz \
        --rehearsal runs/run5/checkpoints/replay.npz \
        --out runs/ft1 --save-at 250,500,1000,2000

Why this exists next to ``Trainer.pretrain``: pretraining warm-starts a *fresh*
net before self-play; this instead nudges an already-strong net toward elite
human move preferences without forgetting its self-play knowledge. Each batch
mixes human rows with rehearsal rows (default 1:3): the human fraction is a
massive oversampling of a ~25k-row corpus, and the rehearsal fraction — the
net's own recent self-play positions — anchors everything the human data has
no opinion about.

The loss is the trainer's own composition (policy CE masked by ``policy_mask``,
value MSE, masked margin MSE, masked aux BCE) with the run5/run6 weights, at a
learning rate well under the training peak. Checkpoints are saved at the
requested step counts in the normal format, so ``mcts:runs/ft1/checkpoints/...``
agent specs and the gauntlet load them unchanged.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ludometer.train.net import load_net, save_checkpoint
from ludometer.train.replay import ReplayBuffer

__all__ = ["main"]


def _load_many(paths: list[Path], seed: int) -> ReplayBuffer:
    """Concatenate several replay.npz files into one buffer.

    ``ReplayBuffer.load`` keeps the destination's capacity, so the row counts
    are read from the files first and each file is staged through a buffer big
    enough to hold it whole.
    """
    rows = []
    for path in paths:
        with np.load(path) as d:
            n = len(d["states"])
        rows.append(n)
    total = max(1, sum(rows))
    out = ReplayBuffer(capacity=total, seed=seed)
    for path, n_file in zip(paths, rows):
        staging = ReplayBuffer(capacity=max(1, n_file), seed=seed)
        staging.load(path)
        n = len(staging)
        out.add(
            staging.states[:n],
            staging.policies[:n],
            staging.values[:n],
            staging.margins[:n],
            margin_mask=staging.margin_mask[:n],
            aux=staging.aux[:n],
            aux_mask=staging.aux_mask[:n],
            policy_mask=staging.policy_mask[:n],
        )
    return out


def _losses(net, batch, device, aux_capable: bool):
    x = torch.from_numpy(batch[0]).to(device)
    target_p = torch.from_numpy(batch[1]).to(device)
    target_v = torch.from_numpy(batch[2]).to(device)
    logits, value, margin, aux_logits = net.forward_aux(x)
    logp = torch.log_softmax(logits, dim=-1)
    per_row_p = -(target_p * logp).sum(dim=1)
    p_w = torch.from_numpy(batch[7]).to(device)
    loss_p = (per_row_p * p_w).sum() / p_w.sum().clamp(min=1.0)
    loss_v = torch.nn.functional.mse_loss(value, target_v)
    zero = torch.zeros((), device=device)
    loss_m = zero
    if margin is not None:
        target_m = torch.from_numpy(batch[3]).to(device)
        m_w = torch.from_numpy(batch[4]).to(device)
        loss_m = ((margin - target_m).square() * m_w).sum() / m_w.sum().clamp(min=1.0)
    loss_a = zero
    if aux_logits is not None and aux_capable:
        target_a = torch.from_numpy(batch[5]).to(device)
        a_w = torch.from_numpy(batch[6]).to(device)
        per_row_a = torch.nn.functional.binary_cross_entropy_with_logits(
            aux_logits, target_a, reduction="none"
        ).mean(dim=1)
        loss_a = (per_row_a * a_w).sum() / a_w.sum().clamp(min=1.0)
    return loss_p, loss_v, loss_m, loss_a


def _mixed_batch(human: ReplayBuffer, rehearsal: ReplayBuffer, n_h: int, n_r: int):
    a, b = human.sample(n_h), rehearsal.sample(n_r)
    return tuple(np.concatenate([xa, xb]) for xa, xb in zip(a, b))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ludometer.train.finetune")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--human", type=Path, nargs="+", required=True)
    parser.add_argument("--rehearsal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="run dir (runs/ft1)")
    parser.add_argument("--save-at", default="250,500,1000,2000")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--human-frac", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args(argv)

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    net, payload = load_net(args.base, device=device)
    net.train()
    human = _load_many(list(args.human), args.seed)
    rehearsal = _load_many([args.rehearsal], args.seed + 1)
    n_h = max(1, round(args.batch_size * args.human_frac))
    n_r = args.batch_size - n_h
    print(
        f"base {args.base} on {device}; human rows {len(human):,} "
        f"({int(human.policy_mask[: len(human)].sum()):,} policy), "
        f"rehearsal rows {len(rehearsal):,}; batch {n_h}+{n_r}, lr {args.lr}"
    )

    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    save_at = sorted(int(s) for s in args.save_at.split(",") if s.strip())
    ckpt_dir = args.out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "finetune.jsonl"
    t0 = time.monotonic()
    sums = np.zeros(4)
    last_log = 0
    for step in range(1, save_at[-1] + 1):
        batch = _mixed_batch(human, rehearsal, n_h, n_r)
        loss_p, loss_v, loss_m, loss_a = _losses(net, batch, device, net.has_aux)
        loss = (
            loss_p
            + args.value_weight * loss_v
            + args.margin_weight * loss_m
            + args.aux_weight * loss_a
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optimizer.step()
        sums += [
            float(loss_p.detach()),
            float(loss_v.detach()),
            float(loss_m.detach()),
            float(loss_a.detach()),
        ]
        if step in save_at:
            name = f"ft-{step:06d}"
            net.eval()
            save_checkpoint(
                ckpt_dir / f"{name}.pt",
                net,
                extra={
                    "run": args.out.name,
                    "checkpoint": name,
                    "finetuned_from": str(args.base),
                    "human": [str(p) for p in args.human],
                    "human_frac": args.human_frac,
                    "lr": args.lr,
                },
            )
            net.train()
            done = step - last_log
            entry = {
                "step": step,
                "loss_p": round(float(sums[0]) / done, 4),
                "loss_v": round(float(sums[1]) / done, 4),
                "loss_m": round(float(sums[2]) / done, 4),
                "loss_a": round(float(sums[3]) / done, 4),
                "t": round(time.monotonic() - t0, 1),
            }
            with open(log_path, "a") as fh:
                fh.write(json.dumps(entry) + "\n")
            print(f"saved {name}: {entry}")
            sums[:] = 0
            last_log = step
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
