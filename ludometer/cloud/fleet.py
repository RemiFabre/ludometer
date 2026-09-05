"""Launch, list and cancel generator jobs; keep the spend ledger honest.

    uv run python -m ludometer.cloud.fleet launch --n 4 --flavor cpu-upgrade \
        --timeout 8h --run mid3 --purpose "phase A teacher corpus"
    uv run python -m ludometer.cloud.fleet ps
    uv run python -m ludometer.cloud.fleet ledger
    uv run python -m ludometer.cloud.fleet cancel --all

Every launch appends a line to ``runs/cloud/ledger.jsonl`` (id, namespace,
flavor, $/hour, timeout, purpose). ``ledger`` refreshes each job's status from
the Hub and prints actual cost (billed minutes x price) plus the *committed*
cost of what is still running (its full timeout — the worst case). ``launch``
refuses when committed + requested would exceed the cap.

The job's command is one bootstrap script: install numpy + CPU torch +
huggingface_hub into a plain ``python:3.12`` image, download the source
bundle, run the generator. The only secret is ``HF_TOKEN``. The name every
colleague sees is ``rl-experiment``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["BOOTSTRAP", "PRICES", "launch", "ledger_totals", "main"]

LEDGER = Path("runs/cloud/ledger.jsonl")
NAMESPACE = "pollen-robotics"
CAP_USD = 90.0
IMAGE = "python:3.12-slim"
LABEL = "rl-experiment"

# $/hour, from `hf jobs hardware` on 2026-09-05.
PRICES = {
    "cpu-basic": 0.01,
    "cpu-upgrade": 0.03,
    "cpu-xl": 1.00,
    "cpu-performance": 1.90,
    "t4-small": 0.40,
    "t4-medium": 0.60,
    "l4x1": 0.80,
    "a10g-small": 1.00,
    "a10g-large": 1.50,
}

# vCPUs per flavor: `nproc` inside a job reports the HOST's cores (64 on the
# first smoke job), so the generator must be told how many drivers to run.
VCPUS = {
    "cpu-basic": 2,
    "cpu-upgrade": 8,
    "cpu-xl": 16,
    "cpu-performance": 32,
    "t4-small": 4,
    "t4-medium": 8,
    "l4x1": 8,
    "a10g-small": 4,
    "a10g-large": 12,
}

BOOTSTRAP = r"""
set -euo pipefail
echo "[rl-experiment] bootstrap on $(nproc) cpus"
pip install -q --no-cache-dir numpy huggingface_hub >/dev/null
pip install -q --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu >/dev/null
python - <<'PY'
import os
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["RLX_SRC_REPO"], os.environ["RLX_BUNDLE"], repo_type="dataset", local_dir="/work/src")
print("[rl-experiment] bundle", p)
PY
mkdir -p /work/tree && tar -xzf "/work/src/$RLX_BUNDLE" -C /work/tree
cd /work/tree
export PYTHONPATH=/work/tree OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec python -m ludometer.cloud.generator --run "$RLX_RUN" --shards "$RLX_SHARDS" \
  --weights "$RLX_WEIGHTS" --tag "$RLX_TAG" --workers "$RLX_WORKERS" --block "$RLX_BLOCK" $RLX_EXTRA
"""


def _timeout_hours(timeout: str) -> float:
    unit = timeout[-1]
    n = float(timeout[:-1])
    return {"h": n, "m": n / 60, "s": n / 3600}[unit]


def _read_ledger() -> list[dict[str, Any]]:
    if not LEDGER.exists():
        return []
    return [
        json.loads(l)
        for l in LEDGER.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def _write_ledger(rows: list[dict[str, Any]]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, LEDGER)


def ledger_totals(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Actual (billed so far) and committed (worst case) dollars."""
    actual = 0.0
    committed = 0.0
    for r in rows:
        price = float(r["price_per_h"])
        hours = r.get("billed_h")
        if hours is not None:
            actual += price * float(hours)
        if r.get("stage") in (None, "SCHEDULING", "RUNNING", "STARTING", "PENDING"):
            committed += price * float(r["timeout_h"])
        else:
            committed += price * float(hours or 0.0)
    return {"actual": round(actual, 3), "committed": round(committed, 3)}


def _api(token: str | None = None):
    from huggingface_hub import HfApi

    return HfApi(token=token or os.environ.get("HF_TOKEN"))


def refresh(
    rows: list[dict[str, Any]], namespace: str = NAMESPACE
) -> list[dict[str, Any]]:
    api = _api()
    for r in rows:
        if r.get("stage") in ("COMPLETED", "ERROR", "CANCELED", "DELETED"):
            continue
        try:
            info = api.inspect_job(
                job_id=r["id"], namespace=r.get("namespace", namespace)
            )
        except Exception as exc:  # noqa: BLE001
            r["note"] = f"inspect failed: {exc}"
            continue
        r["stage"] = info.status.stage
        start = info.started_at or info.created_at
        end = info.finished_at
        if start is not None:
            end_t = end or dt.datetime.now(dt.UTC)
            r["billed_h"] = round(max(0.0, (end_t - start).total_seconds()) / 3600, 4)
        if end is not None:
            r["ended"] = end.isoformat()
    return rows


def launch(
    n: int,
    flavor: str,
    timeout: str,
    run: str,
    shards: str,
    weights: str,
    src_repo: str,
    bundle: str,
    purpose: str,
    workers: int = 0,
    block: int = 64,
    extra: str = "",
    namespace: str = NAMESPACE,
    cap: float = CAP_USD,
    dry_run: bool = False,
) -> list[str]:
    if flavor not in PRICES:
        raise SystemExit(f"unknown flavor {flavor!r} (add its price to PRICES first)")
    price = PRICES[flavor]
    hours = _timeout_hours(timeout)
    rows = refresh(_read_ledger()) if LEDGER.exists() else []
    totals = ledger_totals(rows)
    ask = n * price * hours
    if totals["committed"] + ask > cap:
        raise SystemExit(
            f"refusing: committed ${totals['committed']:.2f} + requested ${ask:.2f} "
            f"> cap ${cap:.2f}"
        )
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN must be set")
    api = _api(token)
    print(
        f"[fleet] {n} x {flavor} ({price:.2f} $/h) x {timeout} = ${ask:.2f} worst case; "
        f"ledger committed ${totals['committed']:.2f}, actual ${totals['actual']:.2f}"
    )
    ids: list[str] = []
    for _ in range(n):
        tag = secrets.token_hex(3)
        env = {
            "RLX_SRC_REPO": src_repo,
            "RLX_BUNDLE": bundle,
            "RLX_RUN": run,
            "RLX_SHARDS": shards,
            "RLX_WEIGHTS": weights,
            "RLX_TAG": tag,
            "RLX_WORKERS": str(workers or VCPUS.get(flavor, 0)),
            "RLX_BLOCK": str(block),
            "RLX_EXTRA": extra,
        }
        if dry_run:
            print(f"[fleet] dry run: {flavor} {timeout} tag {tag} env {env}")
            continue
        job = api.run_job(
            image=IMAGE,
            command=["bash", "-c", BOOTSTRAP],
            env=env,
            secrets={"HF_TOKEN": token},
            flavor=flavor,
            timeout=timeout,
            namespace=namespace,
            labels={"project": LABEL, "run": run},
        )
        row = {
            "id": job.id,
            "namespace": namespace,
            "flavor": flavor,
            "price_per_h": price,
            "timeout_h": hours,
            "launched": dt.datetime.now(dt.UTC).isoformat(),
            "run": run,
            "tag": tag,
            "purpose": purpose,
            "url": getattr(job, "url", None),
            "stage": "SCHEDULING",
        }
        rows.append(row)
        _write_ledger(rows)
        ids.append(job.id)
        print(f"[fleet] launched {job.id} tag {tag} {row['url'] or ''}")
        time.sleep(1.0)
    return ids


def cmd_ps(args: argparse.Namespace) -> int:
    rows = refresh(_read_ledger())
    _write_ledger(rows)
    live = [
        r
        for r in rows
        if r.get("stage") not in ("COMPLETED", "ERROR", "CANCELED", "DELETED")
    ]
    for r in rows if args.all else live:
        print(
            f"{r['id']}  {r.get('stage', '?'):10s} {r['flavor']:12s} run={r['run']} "
            f"tag={r.get('tag')} billed={r.get('billed_h', 0):.2f}h  {r['purpose']}"
        )
    t = ledger_totals(rows)
    print(
        f"[fleet] {len(live)} live; actual ${t['actual']:.2f}, committed ${t['committed']:.2f}"
    )
    # Anything of ours the ledger does not know about?
    try:
        api = _api()
        for info in api.list_jobs(namespace=args.namespace):
            if (info.labels or {}).get("project") == LABEL and info.id not in {
                r["id"] for r in rows
            }:
                print(f"[fleet] NOT IN LEDGER: {info.id} {info.status.stage}")
    except Exception as exc:  # noqa: BLE001
        print(f"[fleet] list_jobs failed: {exc}")
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    rows = refresh(_read_ledger())
    api = _api()
    for r in rows:
        if r.get("stage") in ("COMPLETED", "ERROR", "CANCELED", "DELETED"):
            continue
        if not args.all and r["id"] not in args.ids and r.get("run") != args.run:
            continue
        try:
            api.cancel_job(job_id=r["id"], namespace=r.get("namespace", args.namespace))
            r["stage"] = "CANCELED"
            print(f"[fleet] cancelled {r['id']}")
        except Exception as exc:  # noqa: BLE001
            print(f"[fleet] cancel {r['id']} failed: {exc}")
    rows = refresh(rows)
    _write_ledger(rows)
    return 0


def cmd_ledger(_args: argparse.Namespace) -> int:
    rows = refresh(_read_ledger())
    _write_ledger(rows)
    t = ledger_totals(rows)
    by_purpose: dict[str, float] = {}
    for r in rows:
        by_purpose[r["purpose"]] = by_purpose.get(r["purpose"], 0.0) + r[
            "price_per_h"
        ] * float(r.get("billed_h") or 0.0)
    for k, v in sorted(by_purpose.items(), key=lambda kv: -kv[1]):
        print(f"  ${v:6.2f}  {k}")
    print(
        f"[fleet] {len(rows)} jobs; actual ${t['actual']:.2f}; committed ${t['committed']:.2f}; cap ${CAP_USD:.0f}"
    )
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    bundle = args.bundle
    if not bundle:
        from ludometer.cloud.bundle import upload_bundle

        bundle = upload_bundle(args.src_repo)
        print(f"[fleet] bundle {bundle} -> {args.src_repo}")
    launch(
        n=args.n,
        flavor=args.flavor,
        timeout=args.timeout,
        run=args.run,
        shards=args.shards,
        weights=args.weights,
        src_repo=args.src_repo,
        bundle=bundle,
        purpose=args.purpose,
        workers=args.workers,
        block=args.block,
        extra=args.extra,
        namespace=args.namespace,
        dry_run=args.dry_run,
    )
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    api = _api()
    lines = list(
        api.fetch_job_logs(job_id=args.id, namespace=args.namespace, follow=False)
    )
    for line in lines[-args.tail :]:
        print(line)
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    """A checkpoint + config as a run's current weights (fixed-teacher corpora)."""
    from ludometer.cloud.hub import current_version, hub_from_spec, publish_weights
    from ludometer.train.net import load_net
    from ludometer.train.trainer import TrainConfig

    cfg = TrainConfig.load(args.config)
    run = args.run or cfg.hub_run or cfg.run
    hub = hub_from_spec(args.weights, os.environ.get("HF_TOKEN"))
    net, payload = load_net(args.ckpt)
    pointer = current_version(hub, run)
    version = (int(pointer["version"]) + 1) if pointer else 1
    hub.put_bytes(
        json.dumps(cfg.to_dict(), indent=1).encode("utf-8"), f"{run}/config.json"
    )
    publish_weights(
        hub,
        run,
        net.cpu_state_dict(),
        payload["net_config"],
        version,
        extra={"run": run, "ckpt": str(args.ckpt), "params": net.num_params},
    )
    print(
        f"[fleet] published {args.ckpt} ({net.num_params:,} params) as {run} v{version} "
        f"to {hub.describe()}; sims {cfg.sims}, {cfg.selfplay_games} games/driver"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ludometer.cloud.fleet")
    p.add_argument("--namespace", default=NAMESPACE)
    sub = p.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("launch")
    l.add_argument("--n", type=int, default=1)
    l.add_argument("--flavor", default="cpu-upgrade")
    l.add_argument("--timeout", default="8h")
    l.add_argument("--run", required=True)
    l.add_argument("--shards", default="RemiFabre/rl-experiment-shards")
    l.add_argument("--weights", default="model:RemiFabre/rl-experiment-weights")
    l.add_argument("--src-repo", default="RemiFabre/rl-experiment-src")
    l.add_argument(
        "--bundle", default="", help="existing bundle name (default: build+upload)"
    )
    l.add_argument("--purpose", required=True)
    l.add_argument("--workers", type=int, default=0)
    l.add_argument("--block", type=int, default=64)
    l.add_argument(
        "--extra", default="", help="extra generator args, e.g. '--sims 1024'"
    )
    l.add_argument("--dry-run", action="store_true")
    l.set_defaults(func=cmd_launch)
    ps = sub.add_parser("ps")
    ps.add_argument("--all", action="store_true")
    ps.set_defaults(func=cmd_ps)
    c = sub.add_parser("cancel")
    c.add_argument("ids", nargs="*")
    c.add_argument("--all", action="store_true")
    c.add_argument("--run", default=None)
    c.set_defaults(func=cmd_cancel)
    lg = sub.add_parser("ledger")
    lg.set_defaults(func=cmd_ledger)
    pb = sub.add_parser("publish")
    pb.add_argument("--ckpt", required=True)
    pb.add_argument("--config", required=True, help="train config (search keys)")
    pb.add_argument("--run", default="")
    pb.add_argument("--weights", default="model:RemiFabre/rl-experiment-weights")
    pb.set_defaults(func=cmd_publish)
    lo = sub.add_parser("logs")
    lo.add_argument("id")
    lo.add_argument("--tail", type=int, default=60)
    lo.set_defaults(func=cmd_logs)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
