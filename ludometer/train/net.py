"""Policy + value network for Azul (see docs/DESIGN.md, "Training").

An MLP with residual blocks on the fixed 182-dim encoding produced by
:meth:`ludometer.azul.engine.AzulState.encode`:

    182 -> Linear -> LayerNorm -> ReLU          (stem)
    `blocks` x  x + ReLU(LayerNorm(Linear(x)))  (residual, width `hidden`)
    -> policy head: 180 logits (masked to the legal actions before softmax)
    -> value head:  Linear -> ReLU -> Linear -> tanh, in [-1, 1]

The value is always **from the point of view of the player to move** in the
encoded state (the encoding itself is current-player relative), so a value of
+1 means "the player about to move wins".

Everything is float32. The training device is picked by the trainer (MPS on this
Mac); self-play workers always run on CPU with one thread each, which is why
:class:`NetEvaluator` is written to keep per-call overhead low (inference mode,
preallocated input buffer, softmax over the legal actions only).

This module also owns the *architecture registry*: a checkpoint stores
``net_config["arch"]`` and :func:`make_net` turns that back into the right class
(``"mlp"`` here, ``"structured"`` in :mod:`ludometer.train.net2`). Everything
downstream — ``load_net``, ``MCTSAgent.from_checkpoint``, the arena, the GUI —
therefore works with any architecture without a single change.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from ludometer.azul.engine import ACTION_SPACE, ENCODED_SIZE, AzulState

__all__ = [
    "CHECKPOINT_FORMAT",
    "BaseNet",
    "NetConfig",
    "NetEvaluator",
    "PolicyValueNet",
    "load_checkpoint",
    "load_net",
    "make_net",
    "masked_log_softmax",
    "masked_policy",
    "net_config_from_dict",
    "save_checkpoint",
]

CHECKPOINT_FORMAT = 1
DEFAULT_ARCH = "mlp"


@dataclass(frozen=True)
class NetConfig:
    """Shape of a :class:`PolicyValueNet`; stored inside every checkpoint."""

    input_size: int = ENCODED_SIZE
    hidden: int = 512
    blocks: int = 3
    value_hidden: int = 64
    action_space: int = ACTION_SPACE

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> NetConfig:
        """Build from a dict, ignoring unrelated keys (configs are shared)."""
        fields = {"input_size", "hidden", "blocks", "value_hidden", "action_space"}
        known = {k: int(v) for k, v in (data or {}).items() if k in fields}
        cfg = replace(cls(), **known)
        if cfg.hidden < 1 or cfg.blocks < 0 or cfg.value_hidden < 1:
            raise ValueError(f"invalid net config: {cfg}")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        # "arch" is what makes a checkpoint self-describing: `make_net` reads it
        # back and picks the class, so a run3 checkpoint loads with zero changes
        # anywhere (GUI, arena, registry). Older checkpoints simply lack the key
        # and default to the MLP.
        return {
            "arch": DEFAULT_ARCH,
            "input_size": self.input_size,
            "hidden": self.hidden,
            "blocks": self.blocks,
            "value_hidden": self.value_hidden,
            "action_space": self.action_space,
        }


class BaseNet(nn.Module):
    """Shared plumbing for every architecture (see :func:`make_net`).

    A net is anything that maps ``[B, input_size]`` to
    ``(policy logits [B, action_space], value [B] in [-1, 1])`` and carries a
    ``config`` whose ``to_dict()`` records its ``arch``.
    """

    config: Any

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.inference_mode()
    def evaluate_batch(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Numpy in, numpy out: ``(logits [B, 180], values [B])`` on this device."""
        device = next(self.parameters()).device
        x = torch.as_tensor(states, dtype=torch.float32, device=device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        logits, value = self(x)
        return (
            logits.detach().to("cpu", copy=True).numpy(),
            value.detach().to("cpu", copy=True).numpy(),
        )

    def cpu_state_dict(self) -> dict[str, np.ndarray]:
        """Weights as numpy arrays — what crosses the process boundary."""
        return {
            k: v.detach().to("cpu", copy=True).numpy()
            for k, v in self.state_dict().items()
        }

    def load_numpy_state_dict(self, weights: dict[str, np.ndarray]) -> None:
        self.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})


class _ResidualBlock(nn.Module):
    """``x + ReLU(LayerNorm(Linear(x)))`` — identity path stays untouched."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: Tensor) -> Tensor:
        return x + torch.relu(self.norm(self.fc(x)))


class PolicyValueNet(BaseNet):
    """Joint policy (180 logits) and value (tanh) network."""

    def __init__(self, config: NetConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config if isinstance(config, NetConfig) else NetConfig.from_dict(config)
        self.config = cfg
        self.stem = nn.Linear(cfg.input_size, cfg.hidden)
        self.stem_norm = nn.LayerNorm(cfg.hidden)
        self.blocks = nn.ModuleList(
            _ResidualBlock(cfg.hidden) for _ in range(cfg.blocks)
        )
        self.policy_head = nn.Linear(cfg.hidden, cfg.action_space)
        self.value_fc = nn.Linear(cfg.hidden, cfg.value_hidden)
        self.value_head = nn.Linear(cfg.value_hidden, 1)
        self._init_weights()
        self.float()

    # ------------------------------------------------------------------ setup
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        # Start near a uniform policy and a neutral value: the first self-play
        # batch is then driven by search, not by an arbitrary random prior.
        nn.init.normal_(self.policy_head.weight, std=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.normal_(self.value_head.weight, std=0.01)
        nn.init.zeros_(self.value_head.bias)

    # ----------------------------------------------------------------- forward
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """``(policy_logits [B, 180], value [B])`` — value in [-1, 1]."""
        h = torch.relu(self.stem_norm(self.stem(x)))
        for block in self.blocks:
            h = block(h)
        logits = self.policy_head(h)
        value = torch.tanh(self.value_head(torch.relu(self.value_fc(h)))).squeeze(-1)
        return logits, value


# ------------------------------------------------------------------- factories
def net_config_from_dict(data: dict[str, Any] | Any | None) -> Any:
    """Config object for ``data``, dispatching on ``data["arch"]``.

    Accepts an already-built config (returned unchanged), so callers can pass
    whatever they have. Unknown keys are ignored: the same flat ``configs/*.json``
    dict feeds both the trainer and the net.
    """
    if data is None or isinstance(data, dict):
        arch = (data or {}).get("arch", DEFAULT_ARCH)
        if arch in (DEFAULT_ARCH, "mlp", ""):
            return NetConfig.from_dict(data)
        if arch == "structured":
            from ludometer.train.net2 import StructuredConfig  # lazy: avoids a cycle

            return StructuredConfig.from_dict(data)
        raise ValueError(f"unknown net arch {arch!r} (expected 'mlp' or 'structured')")
    return data  # already a NetConfig / StructuredConfig


def make_net(config: Any) -> BaseNet:
    """Build the net described by ``config`` (dict or config object)."""
    cfg = net_config_from_dict(config)
    if isinstance(cfg, NetConfig):
        return PolicyValueNet(cfg)
    from ludometer.train.net2 import StructuredConfig, StructuredNet  # lazy

    if isinstance(cfg, StructuredConfig):
        return StructuredNet(cfg)
    raise TypeError(f"cannot build a net from {config!r}")  # pragma: no cover


# --------------------------------------------------------------------- masking
def masked_policy(logits: np.ndarray, legal: Sequence[int]) -> np.ndarray:
    """Full-length (180) probability vector that is exactly 0 off ``legal``.

    Softmax is taken over the legal logits only, which is mathematically the same
    as setting the illegal ones to ``-inf`` first but cheaper.
    """
    out = np.zeros(logits.shape[-1], dtype=np.float32)
    if len(legal) == 0:
        return out
    idx = np.asarray(legal, dtype=np.int64)
    sel = logits[idx].astype(np.float32)
    sel -= sel.max()
    np.exp(sel, out=sel)
    out[idx] = sel / sel.sum()
    return out


def masked_log_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    """Log-softmax over the entries where ``mask`` is true; ``-inf`` elsewhere.

    ``mask`` is a bool tensor broadcastable to ``logits``. Illegal entries come
    back as ``-inf``, so ``exp()`` of the result is exactly zero there.
    """
    neg_inf = torch.finfo(logits.dtype).min
    masked = torch.where(mask, logits, torch.full_like(logits, neg_inf))
    out = torch.log_softmax(masked, dim=-1)
    return torch.where(mask, out, torch.full_like(out, float("-inf")))


# ----------------------------------------------------------------- checkpoints
def save_checkpoint(
    path: str | os.PathLike[str],
    net: BaseNet,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Atomically write ``net`` (plus optional ``extra`` payload) to ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "net_config": net.config.to_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
    }
    if extra:
        payload.update(extra)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)
    return target


def load_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a checkpoint payload (weights stay on CPU)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - pragma: no cover, looser payloads
        return torch.load(path, map_location="cpu", weights_only=False)


def load_net(
    path: str | os.PathLike[str], device: str = "cpu"
) -> tuple[BaseNet, dict[str, Any]]:
    """Rebuild the net stored in ``path``; returns ``(net, payload)``.

    The architecture comes from the checkpoint's own ``net_config["arch"]``, so
    every consumer (arena, Elo eval, GUI, registry) loads any run's checkpoints
    without knowing which net produced them.
    """
    payload = load_checkpoint(path)
    net = make_net(payload.get("net_config"))
    net.load_state_dict(payload["state_dict"])
    net.to(device)
    net.eval()
    return net, payload


# ------------------------------------------------------------------ evaluators
class NetEvaluator:
    """Leaf evaluator for MCTS: ``(state, legal) -> (priors, value)``.

    ``priors`` is a numpy array aligned with the ``legal`` list (softmax over the
    legal logits) and ``value`` is a float in [-1, 1] for the player to move.
    """

    def __init__(self, net: BaseNet, device: str = "cpu") -> None:
        self.net = net
        self.device = torch.device(device)
        self.net.to(self.device)
        self.net.eval()
        self._buf = torch.zeros(
            1, net.config.input_size, dtype=torch.float32, device=self.device
        )

    @torch.inference_mode()
    def __call__(
        self, state: AzulState, legal: Sequence[int]
    ) -> tuple[np.ndarray, float]:
        self._buf[0].copy_(torch.from_numpy(state.encode()))
        logits, value = self.net(self._buf)
        v = float(value[0])
        if not legal:
            return np.zeros(0, dtype=np.float32), v
        row = logits[0].to("cpu", copy=True).numpy()
        sel = row[np.asarray(legal, dtype=np.int64)]
        sel = sel - sel.max()
        np.exp(sel, out=sel)
        return sel / sel.sum(), v

    def full_policy(self, state: AzulState, legal: Sequence[int]) -> np.ndarray:
        """180-long masked probability vector (debugging / tests)."""
        priors, _ = self(state, legal)
        out = np.zeros(self.net.config.action_space, dtype=np.float32)
        if len(legal):
            out[np.asarray(legal, dtype=np.int64)] = priors
        return out
