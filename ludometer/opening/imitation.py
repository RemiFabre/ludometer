"""An opening policy from the experts alone: behaviour cloning on round-1 rows.

    uv run python -m ludometer.opening.imitation --rounds 0 --out runs/imit_r1 \
        --baseline porcelain=runs/porc_w-p0905-2038/checkpoints/ckpt-000000.pt

Rows come from the human pipeline's ``data/human/replay.npz`` (read only): every
stored row there is a target-seat decision with a one-hot policy target, and
the encoded state carries the round index (``OFF_ROUND``), so "round 1" is a
row filter. The last 10% of rows (in game order, i.e. the newest games) are held
out; the checkpoint kept is the epoch with the best held-out top-1. ``--init``
starts from an existing checkpoint (Porcelain) instead of a fresh small net,
which is the other natural opening policy: the strong net's prior, bent toward
the experts on the opening rows only.

Loss: policy cross-entropy over the full action space (the convention of
``ludometer.human.agreement``: no legality mask is stored, the net learns to
concentrate on legal moves and the agent masks at play time) plus a small value
MSE on the game outcome so the checkpoint's value head is not garbage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from ludometer.azul.engine import OFF_ROUND
from ludometer.train.net import load_net, make_net, save_checkpoint
from ludometer.train.net2 import StructuredConfig

__all__ = ["main", "load_rows"]


def load_rows(path: Path, rounds: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as z:
        n = int(z["meta"][1]) if "meta" in z else len(z["states"])
        states = z["states"][:n]
        policies = z["policies"][:n]
        values = z["values"][:n]
        mask = z["policy_mask"][:n] if "policy_mask" in z else np.ones(n)
    rnd = np.rint(states[:, OFF_ROUND] * 10).astype(int)
    keep = (mask > 0.5) & (rnd >= rounds[0]) & (rnd <= rounds[1])
    return states[keep].astype(np.float32), policies[keep].argmax(axis=1).astype(np.int64), values[keep].astype(np.float32)


def score(net: torch.nn.Module, states: np.ndarray, targets: np.ndarray, device: str, batch: int = 2048) -> dict[str, float]:
    net.eval()
    top1 = top3 = 0
    nll = 0.0
    with torch.inference_mode():
        for i in range(0, len(states), batch):
            x = torch.from_numpy(states[i : i + batch]).to(device)
            logits = net(x)[0].float()
            logp = torch.log_softmax(logits, dim=1).cpu().numpy()
            t = targets[i : i + batch]
            order = np.argsort(-logp, axis=1)
            top1 += int((order[:, 0] == t).sum())
            top3 += int((order[:, :3] == t[:, None]).any(axis=1).sum())
            nll += float(-logp[np.arange(len(t)), t].sum())
    n = len(states)
    return {"top1": top1 / n, "top3": top3 / n, "nll": nll / n}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ludometer.opening.imitation")
    p.add_argument("--npz", type=Path, default=Path("data/human/replay.npz"))
    p.add_argument("--rounds", default="0")
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--init", type=Path, default=None, help="start from this checkpoint")
    p.add_argument("--embed", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--body", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--value-weight", type=float, default=0.25)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=20260906)
    p.add_argument("--baseline", action="append", default=[], help="name=ckpt: raw-prior agreement on the same rows")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    lo, _, hi = args.rounds.partition("-")
    rounds = (int(lo), int(hi) if hi else int(lo))
    states, targets, values = load_rows(args.npz, rounds)
    n = len(states)
    cut = int(n * (1.0 - args.holdout))
    tr = slice(0, cut)
    ho = slice(cut, n)
    device = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    print(f"[imitation] rounds {rounds}: {n:,} expert decisions, {cut:,} train / {n - cut:,} held out; device {device}", flush=True)
    torch.manual_seed(args.seed)
    if args.init is not None:
        net, _payload = load_net(args.init, device=device)
        print(f"[imitation] init from {args.init} ({sum(q.numel() for q in net.parameters()):,} params)", flush=True)
    else:
        cfg = StructuredConfig(embed=args.embed, layers=args.layers, heads=4, ffn_mult=2, body=args.body,
                               body_blocks=1, value_hidden=128, policy_rank=40, policy_global=True, margin_head=True)
        net = make_net(cfg).to(device)
        print(f"[imitation] fresh net {sum(q.numel() for q in net.parameters()):,} params", flush=True)
    report: dict = {"rounds": list(rounds), "rows": n, "train": cut, "holdout": n - cut, "baselines": {}, "epochs": []}
    for spec in args.baseline:
        name, _, ckpt = spec.partition("=")
        b, _ = load_net(ckpt, device=device)
        report["baselines"][name] = {"train": score(b, states[tr], targets[tr], device), "holdout": score(b, states[ho], targets[ho], device)}
        h = report["baselines"][name]["holdout"]
        print(f"[imitation] baseline {name}: held-out top1 {h['top1']:.3f} top3 {h['top3']:.3f} nll {h['nll']:.3f}", flush=True)
        del b
    before = score(net, states[ho], targets[ho], device)
    print(f"[imitation] before training: held-out top1 {before['top1']:.3f} top3 {before['top3']:.3f} nll {before['nll']:.3f}", flush=True)
    report["before"] = before

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, cut // args.batch_size)
    total = steps_per_epoch * args.epochs
    rng = np.random.default_rng(args.seed)
    best = None
    ckpt_dir = args.out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    step = 0
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = rng.permutation(cut)
        sum_p = sum_v = 0.0
        for i in range(steps_per_epoch):
            idx = np.sort(order[i * args.batch_size : (i + 1) * args.batch_size])
            lr = 0.5 * args.lr * (1.0 + math.cos(math.pi * step / total))
            for g in opt.param_groups:
                g["lr"] = lr
            x = torch.from_numpy(states[idx]).to(device)
            t = torch.from_numpy(targets[idx]).to(device)
            v = torch.from_numpy(values[idx]).to(device)
            out = net(x)
            logits, value = out[0], out[1]
            loss_p = torch.nn.functional.cross_entropy(logits.float(), t)
            loss_v = torch.nn.functional.mse_loss(value.float().reshape(-1), v)
            loss = loss_p + args.value_weight * loss_v
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sum_p += float(loss_p.detach())
            sum_v += float(loss_v.detach())
            step += 1
        h = score(net, states[ho], targets[ho], device)
        entry = {"epoch": epoch, "loss_p": sum_p / steps_per_epoch, "loss_v": sum_v / steps_per_epoch, **{f"holdout_{k}": v for k, v in h.items()}, "t": round(time.monotonic() - t0, 1)}
        report["epochs"].append(entry)
        flag = ""
        if best is None or h["top1"] > best["top1"]:
            best = {**h, "epoch": epoch}
            net.eval()
            save_checkpoint(ckpt_dir / "best.pt", net, extra={"run": args.out.name, "checkpoint": "best", "epoch": epoch,
                                                               "imitation": {"rounds": list(rounds), "rows": n, "init": str(args.init) if args.init else None}})
            flag = " *"
        print(f"[imitation] epoch {epoch:3d}: train CE {entry['loss_p']:.3f}  held-out top1 {h['top1']:.3f} top3 {h['top3']:.3f} nll {h['nll']:.3f}{flag}", flush=True)
    net.eval()
    save_checkpoint(ckpt_dir / "last.pt", net, extra={"run": args.out.name, "checkpoint": "last", "epoch": args.epochs})
    report["best"] = best
    (args.out / "imitation.json").write_text(json.dumps(report, indent=1))
    print(f"[imitation] best epoch {best['epoch']}: held-out top1 {best['top1']:.3f} top3 {best['top3']:.3f} nll {best['nll']:.3f} -> {ckpt_dir / 'best.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
