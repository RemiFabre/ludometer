"""Export a :class:`~ludometer.train.net.PolicyValueNet` checkpoint to ONNX.

The browser player (``web/player/``) has no torch: it feeds the very same 182-float
observation produced by :meth:`AzulState.encode` to onnxruntime-web. This module
is the bridge — it writes

    <out>/model.onnx        policy+value graph, float32
    <out>/model_meta.json   which run/checkpoint/Elo it came from

The exported graph is deliberately boring: one input ``obs`` of shape
``[batch, 182]`` (batch is dynamic, the player uses 1) and two outputs,
``policy`` ``[batch, 180]`` raw logits and ``value`` ``[batch, 1]`` in [-1, 1]
from the point of view of the player to move. Masking to the legal actions and
the softmax stay on the JavaScript side, exactly as :class:`NetEvaluator` does
them in Python — the graph itself knows nothing about legality.

A run4 checkpoint (margin head, see :mod:`ludometer.train.net2`) adds a **third**
output, ``margin`` ``[batch, 1]``, also in [-1, 1] and also from the player to
move's point of view: ``tanh(score difference / 20)``. It is appended, never
inserted, and only when the checkpoint has the head — so ``policy`` and ``value``
keep their names *and* their positions, a run3 export is byte-for-byte what it
always was, and the deployed page (which reads outputs by name) keeps working
whichever kind of checkpoint is published.

A run6 checkpoint appends a **fourth**, ``wall`` ``[batch, 30]``: the probability
that each of the two players ends the game holding each wall row, column and
colour (see :func:`ludometer.train.net2.aux_slices` for the layout). Same rules —
appended last, present only when the checkpoint has the heads — and the graph
emits **probabilities**, not logits, because a browser reading "row 3: 0.82" needs
no knowledge of the training loss. The player does not have to consume it; it is
exported because it is the most explainable thing the net knows ("it is playing
for the blue colour bonus") and it costs 8 KB of graph.

Parity is checked before the file is accepted: ~100 observations taken from real
random games are run through both torch and onnxruntime and the maximum absolute
difference must stay under ``--tol`` (1e-4 by default). onnxruntime is optional
(it is not a project dependency); without it the check is skipped loudly and the
node-side test ``web/player/test/`` remains the gate.

CLI::

    uv run python -m ludometer.export.onnx_export                  # best checkpoint
    uv run python -m ludometer.export.onnx_export --ckpt runs/run2/checkpoints/x.pt
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from ludometer.agents.registry import find_best_checkpoint
from ludometer.azul.engine import ACTION_SPACE, ENCODED_SIZE, AzulState
from ludometer.train.net import PolicyValueNet, load_net

__all__ = [
    "AUX_OUTPUT",
    "DEFAULT_OUT_DIR",
    "MARGIN_OUTPUT",
    "OUTPUT_NAMES",
    "ExportResult",
    "ExportWrapper",
    "collect_observations",
    "export_checkpoint",
    "main",
    "output_names",
    "verify_parity",
]

# repo layout: <root>/ludometer/export/onnx_export.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = _REPO_ROOT / "web" / "player" / "model"
# torch's own answers on the parity samples, so the *browser* runtime
# (onnxruntime-web, not onnxruntime-python) can be held to them too
DEFAULT_REFERENCE = (
    _REPO_ROOT / "web" / "player" / "test" / "fixtures" / "torch_reference.json.gz"
)

OPSET = 17  # widely supported by onnxruntime-web 1.x; the net only needs matmul/norm


OUTPUT_NAMES = ("policy", "value")
MARGIN_OUTPUT = "margin"
AUX_OUTPUT = "wall"


class ExportWrapper(nn.Module):
    """``PolicyValueNet`` with a web-friendly signature: value stays 2-D.

    :meth:`PolicyValueNet.forward` squeezes the value to ``[B]``; keeping it at
    ``[B, 1]`` means the JS side reads ``value.data[0]`` whatever the batch is.
    The margin, when the net has one, is shaped and appended the same way, and the
    aux heads follow it as ``[B, 30]`` **probabilities** (the sigmoid the training
    loss keeps inside itself is applied here, once, in the graph).
    """

    def __init__(self, net: PolicyValueNet) -> None:
        super().__init__()
        self.net = net
        self.has_margin = bool(getattr(net, "has_margin", False))
        self.has_aux = bool(getattr(net, "has_aux", False))

    def forward(self, obs: Tensor) -> tuple[Tensor, ...]:
        if not self.has_margin and not self.has_aux:
            logits, value = self.net(obs)
            return logits, value.reshape(-1, 1)
        logits, value, margin, aux = self.net.forward_aux(obs)
        out: tuple[Tensor, ...] = (logits, value.reshape(-1, 1))
        if margin is not None:
            out = (*out, margin.reshape(-1, 1))
        if aux is not None:
            out = (*out, torch.sigmoid(aux))
        return out


def output_names(net: Any) -> list[str]:
    """Graph outputs for ``net``, in order (new heads are always appended)."""
    names = list(OUTPUT_NAMES)
    if getattr(net, "has_margin", False):
        names.append(MARGIN_OUTPUT)
    if getattr(net, "has_aux", False):
        names.append(AUX_OUTPUT)
    return names


# --------------------------------------------------------------------- fixtures
def collect_observations(count: int, seed: int = 12345) -> np.ndarray:
    """``count`` encoded positions taken from real random games (not noise).

    Random floats would exercise the graph but not the input distribution the
    player actually feeds it; playing games instead means the parity check runs
    on genuine encodings, including the round-boundary ones.
    """
    import random

    rng = random.Random(seed)
    rows: list[np.ndarray] = []
    game = 0
    while len(rows) < count:
        state = AzulState.new_game(seed=rng.randrange(1 << 30))
        game += 1
        moves = 0
        while not state.is_terminal and moves < 400 and len(rows) < count:
            rows.append(state.encode())
            legal = state.legal_actions()
            if not legal:  # pragma: no cover - defensive
                break
            state.apply(legal[rng.randrange(len(legal))])
            moves += 1
    return np.stack(rows[:count]).astype(np.float32)


# ----------------------------------------------------------------------- parity
def verify_parity(
    onnx_path: Path, net: PolicyValueNet, samples: np.ndarray, tol: float = 1e-4
) -> dict[str, Any]:
    """Run ``samples`` through torch and onnxruntime; report the worst gap.

    Returns ``{"checked": bool, "n": int, "policy_max_abs_diff": float,
    "value_max_abs_diff": float, "tol": float}``. Raises ``AssertionError`` when
    onnxruntime is available and the gap exceeds ``tol``.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {
            "checked": False,
            "reason": "onnxruntime not installed (pip install onnxruntime)",
            "n": int(samples.shape[0]),
            "tol": tol,
        }

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    net.eval()
    with torch.inference_mode():
        t_logits, t_value, t_margin, t_aux = net.forward_aux(torch.from_numpy(samples))
    ref_logits = t_logits.numpy()
    ref_value = t_value.reshape(-1, 1).numpy()
    ref_margin = None if t_margin is None else t_margin.reshape(-1, 1).numpy()
    ref_aux = None if t_aux is None else torch.sigmoid(t_aux).numpy()

    p_diff = 0.0
    v_diff = 0.0
    m_diff = 0.0
    a_diff = 0.0
    # one row at a time: batch 1 is what the browser runs, so that is what we check
    for i in range(samples.shape[0]):
        out = session.run(None, {"obs": samples[i : i + 1]})
        p_diff = max(p_diff, float(np.abs(out[0] - ref_logits[i : i + 1]).max()))
        v_diff = max(v_diff, float(np.abs(out[1] - ref_value[i : i + 1]).max()))
        at = 2
        if ref_margin is not None:
            m_diff = max(m_diff, float(np.abs(out[at] - ref_margin[i : i + 1]).max()))
            at += 1
        if ref_aux is not None:
            a_diff = max(a_diff, float(np.abs(out[at] - ref_aux[i : i + 1]).max()))
    worst = max(p_diff, v_diff, m_diff, a_diff)
    if worst > tol:
        raise AssertionError(
            f"ONNX/torch parity failed: max |diff| = {worst:.3e} > {tol:.1e} "
            f"(policy {p_diff:.3e}, value {v_diff:.3e}, margin {m_diff:.3e}, "
            f"wall {a_diff:.3e})"
        )
    report = {
        "checked": True,
        "n": int(samples.shape[0]),
        "policy_max_abs_diff": p_diff,
        "value_max_abs_diff": v_diff,
        "tol": tol,
    }
    if ref_margin is not None:
        report["margin_max_abs_diff"] = m_diff
    if ref_aux is not None:
        report["wall_max_abs_diff"] = a_diff
    return report


def write_torch_reference(
    path: Path, net: PolicyValueNet, samples: np.ndarray, meta: dict[str, Any]
) -> Path:
    """Dump ``(obs, torch policy, torch value)`` for the node-side parity test.

    :func:`verify_parity` compares torch against *onnxruntime-python*. The browser
    runs onnxruntime-**web**, a different build with its own kernels, so the same
    100 positions are written out here and ``web/player/test/parity.test.mjs``
    holds that runtime to the same 1e-4 — the check that actually covers what a
    visitor executes.
    """
    net.eval()
    with torch.inference_mode():
        logits, value, margin, aux = net.forward_aux(torch.from_numpy(samples))
    payload = {
        "checkpoint": meta.get("checkpoint"),
        "onnx_sha256": meta.get("onnx_sha256"),
        "tol": meta.get("parity", {}).get("tol", 1e-4),
        "obs": samples.astype(np.float32).tolist(),
        "policy": logits.numpy().astype(np.float32).tolist(),
        "value": value.reshape(-1).numpy().astype(np.float32).tolist(),
    }
    if margin is not None:
        payload["margin"] = margin.reshape(-1).numpy().astype(np.float32).tolist()
    if aux is not None:
        payload["wall"] = torch.sigmoid(aux).numpy().astype(np.float32).tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    import gzip

    with gzip.open(path, "wt", compresslevel=9) as fh:
        json.dump(payload, fh)
    return path


# ----------------------------------------------------------------------- export
class ExportResult(dict):
    """The written ``model_meta.json`` payload, plus the paths that were written."""

    @property
    def onnx_path(self) -> Path:
        return Path(self["_onnx_path"])

    @property
    def meta_path(self) -> Path:
        return Path(self["_meta_path"])


def export_checkpoint(
    ckpt: str | os.PathLike[str] | None = None,
    out_dir: str | os.PathLike[str] = DEFAULT_OUT_DIR,
    samples: int = 100,
    tol: float = 1e-4,
    elo: float | None = None,
    run: str | None = None,
    reference: str | os.PathLike[str] | None = DEFAULT_REFERENCE,
) -> ExportResult:
    """Export ``ckpt`` (default: the best rated checkpoint on disk) to ``out_dir``.

    Writes ``model.onnx`` and ``model_meta.json`` and returns the metadata. The
    ONNX file is only moved into place after the parity check passes (when
    onnxruntime is installed), so a bad export never overwrites a good one.
    """
    if ckpt is None:
        best = find_best_checkpoint()
        ckpt_path, elo, run, name = best.path, best.elo, best.run, best.ckpt
    else:
        ckpt_path = Path(ckpt)
        name = ckpt_path.stem
        if run is None:
            # runs/<run>/checkpoints/<name>.pt
            parents = ckpt_path.resolve().parents
            run = parents[1].name if len(parents) > 1 else ""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    net, payload = load_net(ckpt_path, device="cpu")
    wrapper = ExportWrapper(net).eval()
    dummy = torch.zeros(1, net.config.input_size, dtype=torch.float32)

    names = output_names(net)
    axes: dict[str, dict[int, str]] = {"obs": {0: "batch"}}
    for output in names:
        axes[output] = {0: "batch"}

    tmp = out / "model.onnx.tmp"
    # dynamo=False on purpose: the TorchScript exporter is deprecated but it emits
    # the plain Gemm/LayerNormalization graph onnxruntime-web is happiest with,
    # and this net has no control flow for the new exporter to earn its keep on.
    # Its deprecation notice is not news, so it is not printed on every deploy.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(tmp),
            input_names=["obs"],
            output_names=names,
            dynamic_axes=axes,
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )

    obs = collect_observations(samples)
    parity = verify_parity(tmp, net, obs, tol=tol)

    target = out / "model.onnx"
    os.replace(tmp, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()

    meta: dict[str, Any] = {
        "run": run or "",
        "checkpoint": name,
        "elo": None if elo is None else round(float(elo), 1),
        "games": payload.get("games"),
        "exported_at": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_size": int(net.config.input_size),
        "action_space": int(net.config.action_space),
        "net_config": net.config.to_dict(),
        "num_params": int(net.num_params),
        "outputs": names,
        "has_margin": bool(getattr(net, "has_margin", False)),
        "has_aux": bool(getattr(net, "has_aux", False)),
        "opset": OPSET,
        "onnx_bytes": target.stat().st_size,
        "onnx_sha256": digest,
        "parity": parity,
        "source_checkpoint": str(ckpt_path),
    }
    assert meta["input_size"] == ENCODED_SIZE, "encoding size drifted from the engine"
    assert meta["action_space"] == ACTION_SPACE, "action space drifted from the engine"

    if reference is not None:
        meta["torch_reference"] = str(
            write_torch_reference(Path(reference), net, obs, meta)
        )

    meta_path = out / "model_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    result = ExportResult(meta)
    result["_onnx_path"] = str(target)
    result["_meta_path"] = str(meta_path)
    return result


# -------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument(
        "--ckpt", default=None, help="checkpoint .pt (default: best rated on disk)"
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT_DIR), help="output directory")
    parser.add_argument(
        "--samples", type=int, default=100, help="positions used for the parity check"
    )
    parser.add_argument("--tol", type=float, default=1e-4, help="max |torch - onnx|")
    parser.add_argument("--elo", type=float, default=None, help="override the Elo tag")
    args = parser.parse_args(argv)

    meta = export_checkpoint(
        ckpt=args.ckpt,
        out_dir=args.out,
        samples=args.samples,
        tol=args.tol,
        elo=args.elo,
    )
    size_mb = meta["onnx_bytes"] / 1e6
    print(
        f"exported {meta['run']}/{meta['checkpoint']} "
        f"(elo {meta['elo']}, {meta['num_params']:,} params) "
        f"-> {meta['_onnx_path']}  [{size_mb:.1f} MB]"
    )
    parity = meta["parity"]
    if parity.get("checked"):
        print(
            f"parity ok on {parity['n']} positions: "
            f"policy {parity['policy_max_abs_diff']:.2e}, "
            f"value {parity['value_max_abs_diff']:.2e} (tol {parity['tol']:.0e})"
        )
    else:
        print(f"PARITY NOT CHECKED — {parity.get('reason')}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
