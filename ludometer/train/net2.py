"""Structured policy/value network — entities instead of a flat vector.

Why
---
:class:`ludometer.train.net.PolicyValueNet` is an MLP over the flat 182-dim
encoding, so it has to *learn* that features 126..131 and 131..136 are two
interchangeable factory displays, and that "colour ``c`` into row ``r``" is the
same operation whatever ``r`` is. Every such symmetry costs capacity and data.

This net slices the very same encoding (via the ``OFF_*`` constants in
:mod:`ludometer.azul.engine` — the encoding is unchanged, so old buffers and the
GUI keep working) into **22 entity tokens**, embeds each *type* with a small
weight-shared MLP, mixes them with self-attention and reads two heads off the
result:

    tokens                         count  raw dims  shared embedder
    ------------------------------ -----  --------  ---------------------------
    pool (5 factories + centre)        6         6  one MLP for all six sources
    pattern row (2 players x 5)       10        11  one MLP for all ten rows
    wall sets (2 players)              2         4  one MLP for both players
    floor (2 players)                  2         7  one MLP for both floors
    supply (bag + lid)                 1        10
    globals (scores, round, ...)       1         7

A pattern-row token carries the row's colour one-hot, its fill fraction **and
the matching wall row** — the three things that decide whether a colour may go
there and what it would score — so the "can I still use this row" question is
answerable inside one token.

Weight sharing is the point: the factory embedder sees 5 (well, 6) examples per
position instead of 1, and the row embedder 10. Identity is not lost, because a
learned per-slot bias (``slot_bias``, one vector per token position) is added
after the shared embedding: the net still knows *which* factory or row it is
looking at, it just does not need a private parameter block for each.

Trunk: **self-attention**, 1-2 pre-LN layers, chosen over gated pooling because
the decisions this game needs are relational — "is the black tile on factory 3
useful *for my row 2*, and would taking it hand the centre to my opponent" is a
pairwise question between two tokens. Gated pooling can weight tokens but cannot
compare them. With 22 tokens the attention matrix is 22x22, i.e. free compared
to the projections.

Policy head (180 = source x colour x destination): **factorised**. Each of the
6 post-trunk source tokens emits a per-colour key ``A[s, c] in R^k``; each of the
6 destination tokens (my 5 pattern rows + my floor) emits a query
``B[d] in R^k``; the logit is ``<A[s, c], B[d]>`` plus a per-(source, colour)
bias, plus (optionally) a full 180-wide correction read off the pooled trunk
vector. So "take blue from factory 2" and "put it in row 3" are scored by the
tokens that actually own those facts, and the 180 logits share statistical
strength instead of being 180 independent output units.

Value head: readout vector (globals token, which has attended over the whole
board, concatenated with the mean token) -> residual MLP body -> tanh, in
[-1, 1] for the player to move (same convention as ``net.py``).

Budget
------
Self-play workers run this single-threaded on CPU, one position at a time, so
what matters is not FLOPs but *time per call*: keep the op count low (the six
gathers are one ``index_select``; the per-type embedders are one ``Linear``
each; attention uses the fused SDPA kernel). See
``ludometer.train.benchmark`` for the measurement; the run3 config is chosen to
sit under 0.45 ms/position.

Checkpoints are the same format as ``net.py``: ``StructuredConfig.to_dict()``
records ``"arch": "structured"``, and :func:`ludometer.train.net.make_net`
dispatches on it, so ``load_net`` / ``MCTSAgent.from_checkpoint`` / the GUI need
no changes at all.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ludometer.azul.engine import (
    ACTION_SPACE,
    ENCODED_SIZE,
    NUM_COLORS,
    NUM_FACTORIES,
    NUM_ROWS,
    OFF_BAG,
    OFF_CENTER,
    OFF_CENTER_TOTAL,
    OFF_FACTORIES,
    OFF_FACTORY_FLAGS,
    OFF_I_START,
    OFF_LID,
    OFF_MARKER_CENTER,
    OFF_MY_FLOOR,
    OFF_MY_LINES,
    OFF_MY_SETS,
    OFF_MY_WALL,
    OFF_OP_FLOOR,
    OFF_OP_LINES,
    OFF_OP_SETS,
    OFF_OP_WALL,
    OFF_ROUND,
    OFF_SCORES,
    OFF_TILES_LEFT,
)
from ludometer.train.net import BaseNet

__all__ = [
    "DEST_TOKENS",
    "NUM_TOKENS",
    "StructuredConfig",
    "StructuredNet",
    "token_slices",
]

ARCH = "structured"

NUM_SOURCES = NUM_FACTORIES + 1  # 5 factories + centre
NUM_DESTS = NUM_ROWS + 1  # 5 pattern lines + floor
FLOOR_FEATURES = 7  # 5 colour counts, occupancy, marker flag


# ------------------------------------------------------------------ token specs
def _pool_indices() -> list[list[int]]:
    """One token per tile source: 5 colour counts + "how full" (6 dims)."""
    out = []
    for i in range(NUM_FACTORIES):
        base = OFF_FACTORIES + i * NUM_COLORS
        out.append([base + c for c in range(NUM_COLORS)] + [OFF_FACTORY_FLAGS + i])
    out.append([OFF_CENTER + c for c in range(NUM_COLORS)] + [OFF_CENTER_TOTAL])
    return out


def _row_indices() -> list[list[int]]:
    """One token per pattern line: colour one-hot + fill + its wall row (11)."""
    out = []
    for lines, wall in ((OFF_MY_LINES, OFF_MY_WALL), (OFF_OP_LINES, OFF_OP_WALL)):
        for r in range(NUM_ROWS):
            base = lines + r * 6
            out.append(
                [base + c for c in range(NUM_COLORS)]
                + [base + 5]
                + [wall + r * 5 + col for col in range(NUM_COLORS)]
            )
    return out


def _wall_indices() -> list[list[int]]:
    """One token per wall: completed rows / columns / colours + score (4).

    The 25 wall cells themselves are *not* repeated here: every row token already
    carries its own wall row, so this token only has to add what a single row
    cannot see — the column and colour sets — plus the score they are worth.
    """
    return [
        [OFF_MY_SETS, OFF_MY_SETS + 1, OFF_MY_SETS + 2, OFF_SCORES],
        [OFF_OP_SETS, OFF_OP_SETS + 1, OFF_OP_SETS + 2, OFF_SCORES + 1],
    ]


def _floor_indices() -> list[list[int]]:
    return [
        [OFF_MY_FLOOR + i for i in range(FLOOR_FEATURES)],
        [OFF_OP_FLOOR + i for i in range(FLOOR_FEATURES)],
    ]


def _supply_indices() -> list[list[int]]:
    """Bag and lid counts — the public part of what is still to come."""
    return [
        [OFF_BAG + c for c in range(NUM_COLORS)]
        + [OFF_LID + c for c in range(NUM_COLORS)]
    ]


def _global_indices() -> list[list[int]]:
    return [
        [
            OFF_SCORES,
            OFF_SCORES + 1,
            OFF_TILES_LEFT,
            OFF_I_START,
            OFF_ROUND,
            OFF_MARKER_CENTER,
            OFF_CENTER_TOTAL,
        ]
    ]


# (name, index table) in token order — the order defines the token layout.
TOKEN_SPECS: tuple[tuple[str, list[list[int]]], ...] = (
    ("pool", _pool_indices()),
    ("row", _row_indices()),
    ("wall", _wall_indices()),
    ("floor", _floor_indices()),
    ("supply", _supply_indices()),
    ("globals", _global_indices()),
)

NUM_TOKENS = sum(len(idx) for _name, idx in TOKEN_SPECS)  # 22

# Token slots, in order: 0..5 sources, 6..10 my rows, 11..15 their rows,
# 16 my wall, 17 their wall, 18 my floor, 19 their floor, 20 supply, 21 globals.
SRC_TOKENS = tuple(range(NUM_SOURCES))
DEST_TOKENS = (
    *range(NUM_SOURCES, NUM_SOURCES + NUM_ROWS),
    NUM_SOURCES + 2 * NUM_ROWS + 2,
)


def token_slices() -> dict[str, tuple[int, int, int]]:
    """``{type: (first slot, token count, raw feature dims)}`` (docs / tests)."""
    out: dict[str, tuple[int, int, int]] = {}
    start = 0
    for name, idx in TOKEN_SPECS:
        out[name] = (start, len(idx), len(idx[0]))
        start += len(idx)
    return out


# ---------------------------------------------------------------------- config
@dataclass(frozen=True)
class StructuredConfig:
    """Shape of a :class:`StructuredNet`; stored inside every checkpoint."""

    arch: str = ARCH
    input_size: int = ENCODED_SIZE
    action_space: int = ACTION_SPACE
    # Defaults are configs/run3.json's measured operating point: ~1.7M params
    # at ~0.2 ms/position on one idle CPU thread (see ludometer.train.benchmark).
    embed: int = 96  # token width
    layers: int = 1  # self-attention layers over the 22 tokens
    heads: int = 4
    ffn_mult: int = 2  # attention-block feed-forward expansion
    body: int = 1024  # width of the pooled readout trunk
    body_blocks: int = 1  # residual blocks on top of it
    value_hidden: int = 128
    policy_rank: int = 32  # k in <A[s, c], B[d]>
    policy_global: bool = True  # add a 180-wide correction from the body

    _INT_FIELDS = (
        "input_size",
        "action_space",
        "embed",
        "layers",
        "heads",
        "ffn_mult",
        "body",
        "body_blocks",
        "value_hidden",
        "policy_rank",
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> StructuredConfig:
        """Build from a dict, ignoring unrelated keys (configs are shared)."""
        data = data or {}
        known: dict[str, Any] = {
            k: int(v) for k, v in data.items() if k in cls._INT_FIELDS
        }
        if "policy_global" in data:
            known["policy_global"] = bool(data["policy_global"])
        cfg = replace(cls(), **known)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.embed < 1 or self.embed % max(1, self.heads):
            raise ValueError(f"embed must be a positive multiple of heads: {self}")
        if self.layers < 0 or self.body < 1 or self.body_blocks < 0:
            raise ValueError(f"invalid structured net config: {self}")
        if self.value_hidden < 1 or self.policy_rank < 1 or self.ffn_mult < 1:
            raise ValueError(f"invalid structured net config: {self}")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# ---------------------------------------------------------------------- modules
class _AttentionBlock(nn.Module):
    """Pre-LN transformer encoder layer over the entity tokens."""

    def __init__(self, dim: int, heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn_mult * dim)
        self.fc2 = nn.Linear(ffn_mult * dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        h = self.qkv(self.norm1(x))
        q, k, v = h.view(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(attn.transpose(1, 2).reshape(b, t, d))
        return x + self.fc2(torch.relu(self.fc1(self.norm2(x))))


class _ResidualBlock(nn.Module):
    """``x + ReLU(LayerNorm(Linear(x)))`` — same block as ``net.py``."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: Tensor) -> Tensor:
        return x + torch.relu(self.norm(self.fc(x)))


class StructuredNet(BaseNet):
    """Entity-token policy/value net (see the module docstring)."""

    def __init__(self, config: StructuredConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = (
            config
            if isinstance(config, StructuredConfig)
            else StructuredConfig.from_dict(config)
        )
        cfg.validate()
        self.config = cfg
        dim = cfg.embed

        # --- one gather for every token of every type.
        #
        # Batch-1 CPU inference is dominated by *op dispatch*, not by FLOPs, so
        # the six per-type embedders are folded into a single batched matmul:
        # every token's raw features are padded to `width` with a zero column
        # appended to the input (index `input_size`), and the per-type weight
        # matrix is expanded to one matrix per token slot. Weight *sharing*
        # (all factories, all rows) is preserved exactly — `type_idx` is what
        # ties the slots to their type — but the whole embedding layer costs a
        # handful of ops instead of ~20.
        width = max(len(table[0]) for _name, table in TOKEN_SPECS)
        flat: list[int] = []
        types: list[int] = []
        for type_id, (_name, table) in enumerate(TOKEN_SPECS):
            for row in table:
                flat.extend(row + [cfg.input_size] * (width - len(row)))
                types.append(type_id)
        self.token_width = width
        self.register_buffer("gather_idx", torch.tensor(flat, dtype=torch.long))
        self.register_buffer("type_idx", torch.tensor(types, dtype=torch.long))
        self.register_buffer("zero_pad", torch.zeros(1, 1))
        self.register_buffer(
            "dest_idx", torch.tensor(DEST_TOKENS, dtype=torch.long), persistent=False
        )

        # --- entity embedder: weights shared per type, bias learned per slot
        # (the bias is what tells factory 3 from factory 4, and my row 2 from
        # theirs, without giving either its own weight matrix).
        n_types = len(TOKEN_SPECS)
        self.embed_w = nn.Parameter(torch.zeros(n_types, width, dim))
        self.embed_b = nn.Parameter(torch.zeros(1, NUM_TOKENS, dim))
        # layer 2 of the entity MLP, shared by every type: one op for 22 tokens
        # (no LayerNorm here — the first attention block starts with one)
        self.mix = nn.Linear(dim, dim)

        self.trunk = nn.ModuleList(
            _AttentionBlock(dim, cfg.heads, cfg.ffn_mult) for _ in range(cfg.layers)
        )
        self.trunk_norm = nn.LayerNorm(dim)

        # --- readout: the globals token (which has attended over the whole
        # board, i.e. a CLS token) concatenated with the mean over all tokens
        self.body_in = nn.Linear(2 * dim, cfg.body)
        self.body_norm = nn.LayerNorm(cfg.body)
        self.body = nn.ModuleList(
            _ResidualBlock(cfg.body) for _ in range(cfg.body_blocks)
        )

        # --- heads
        k = cfg.policy_rank
        self.src_proj = nn.Linear(dim, NUM_COLORS * k + NUM_COLORS)
        self.dst_proj = nn.Linear(dim, k)
        self.policy_global = (
            nn.Linear(cfg.body, cfg.action_space) if cfg.policy_global else None
        )
        self.value_fc = nn.Linear(cfg.body, cfg.value_hidden)
        self.value_head = nn.Linear(cfg.value_hidden, 1)
        self._init_weights()
        self.float()

    # ------------------------------------------------------------------ setup
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        nn.init.kaiming_uniform_(self.embed_w, nonlinearity="relu")
        nn.init.zeros_(self.embed_b)
        # Start near a uniform policy and a neutral value, exactly like net.py:
        # the first self-play batch is then driven by search, not by noise.
        for module in (self.src_proj, self.dst_proj, self.value_head):
            nn.init.normal_(module.weight, std=0.01)
            nn.init.zeros_(module.bias)
        if self.policy_global is not None:
            nn.init.zeros_(self.policy_global.weight)
            nn.init.zeros_(self.policy_global.bias)

    # ---------------------------------------------------------------- forward
    def tokens(self, x: Tensor) -> Tensor:
        """``[B, 182] -> [B, 22, embed]`` entity embeddings (before the trunk)."""
        b = x.shape[0]
        padded = torch.cat([x, self.zero_pad.expand(b, 1)], dim=1)
        raw = padded.index_select(1, self.gather_idx).view(
            b, NUM_TOKENS, self.token_width
        )
        weight = self.embed_w.index_select(0, self.type_idx)  # [22, width, embed]
        h = torch.einsum("btw,twd->btd", raw, weight) + self.embed_b
        return self.mix(torch.relu(h))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """``(policy_logits [B, 180], value [B])`` — value in [-1, 1]."""
        b = x.shape[0]
        h = self.tokens(x)
        for block in self.trunk:
            h = block(h)
        h = self.trunk_norm(h)

        # readout: globals token (attends over everything) + mean of all tokens
        pooled = torch.cat([h.narrow(1, NUM_TOKENS - 1, 1).view(b, -1), h.mean(1)], -1)
        g = torch.relu(self.body_norm(self.body_in(pooled)))
        for block in self.body:
            g = block(g)

        k = self.config.policy_rank
        src = self.src_proj(h.narrow(1, 0, NUM_SOURCES))
        # [B, 6 sources * 5 colours, k] x [B, k, 6 destinations] -> the 180 logits
        # in exactly the engine's source*30 + colour*6 + dest order.
        keys = src.narrow(2, 0, NUM_COLORS * k).reshape(b, NUM_SOURCES * NUM_COLORS, k)
        queries = self.dst_proj(h.index_select(1, self.dest_idx))
        logits = torch.baddbmm(
            src.narrow(2, NUM_COLORS * k, NUM_COLORS).reshape(
                b, NUM_SOURCES * NUM_COLORS, 1
            ),
            keys,
            queries.transpose(1, 2),
        )
        logits = logits.reshape(b, self.config.action_space)
        if self.policy_global is not None:
            logits = logits + self.policy_global(g)

        value = torch.tanh(self.value_head(torch.relu(self.value_fc(g)))).squeeze(-1)
        return logits, value
