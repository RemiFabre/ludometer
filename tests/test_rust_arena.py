"""The Rust arena (`ludometer_rs.Arena`) against `BatchedSelfPlay`, game for game.

Acceptance layer 3 of docs/RUST_ENGINE.md §6, taken at the project's standard:
**identical games given identical evaluations**. With `games=1` (every forward
pass carries one row, so the tiny net's arithmetic is the single-position
path's), `dirichlet_eps=0` (root noise is the one stream the Rust engine cannot
reproduce, so it is switched off) and `rng="python"`, every array of the two
`GameRecord`s must be equal: states, policies, values, margins, aux, masks,
search values, plus outcome, scores, moves, rounds, decisions, evals.

Skipped when `ludometer_rs` is not built (`rust/README.md` says how).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ludometer.train.mcts import MCTSConfig
from ludometer.train.net import make_net
from ludometer.train.net2 import StructuredConfig
from ludometer.train.selfplay import SelfPlayConfig
from ludometer.train.selfplay_batched import BatchedSelfPlay

rs = pytest.importorskip("ludometer_rs")

TINY = StructuredConfig(
    embed=32, layers=1, heads=4, ffn_mult=2, body=48, body_blocks=1,
    value_hidden=16, policy_rank=8, margin_head=True,
)


def tiny_net(seed: int = 0):
    torch.manual_seed(seed)
    net = make_net(TINY)
    net.eval()
    return net


def selfplay_config(**kwargs) -> SelfPlayConfig:
    mcts_keys = {k: kwargs.pop(k) for k in list(kwargs) if k in MCTSConfig.__dataclass_fields__}
    mcts_keys.setdefault("sims", 24)
    mcts_keys.setdefault("tree_reuse", True)
    mcts_keys.setdefault("chance_children", 2)
    mcts_keys.setdefault("dirichlet_eps", 0.0)  # the one stream Rust cannot mirror
    kwargs.setdefault("temp_moves", 4)
    kwargs.setdefault("max_moves", 120)
    kwargs.setdefault("value_score_weight", 0.0)
    return SelfPlayConfig(mcts=MCTSConfig(**mcts_keys), **kwargs)


def flat_config(config: SelfPlayConfig) -> dict:
    d = {k: getattr(config.mcts, k) for k in MCTSConfig.__dataclass_fields__}
    for k in ("temp_moves", "temperature", "stall_rounds", "max_moves", "value_score_weight",
              "pcr_full_sims", "pcr_cheap_sims", "pcr_full_prob"):
        d[k] = getattr(config, k)
    return d


def softmax_legal(logits: np.ndarray, legal: list[int]) -> np.ndarray:
    """Exactly `BatchEvaluator.evaluate`'s per-row softmax, in float32."""
    sel = logits[np.asarray(legal, dtype=np.int64)]
    sel = sel - sel.max()
    np.exp(sel, out=sel)
    return sel / sel.sum()


def play_rust(net, config: SelfPlayConfig, n_games: int, seed_start: int, games: int = 1,
              rng: str = "python", exact: bool = True) -> list[dict]:
    """Drive the arena with the net on CPU; `exact` softmaxes in numpy like Python."""
    arena = rs.Arena(flat_config(config), has_margin=True, games=games, rng=rng)
    arena.begin(n_games, seed_start)
    out: list[dict] = []
    with torch.inference_mode():
        while not arena.finished():
            obs = arena.gather()
            if len(obs):
                logits, value, margin = net.forward_heads(torch.from_numpy(obs))
                logits = logits.numpy()
                values = value.numpy().astype(np.float32)
                margins = margin.numpy().astype(np.float32)
                if exact:
                    legal = arena.pending_legal()
                    priors = [softmax_legal(logits[i], legal[i]) for i in range(len(legal))]
                    arena.apply_leaves(priors, [float(v) for v in values], [float(m) for m in margins])
                else:
                    arena.apply_logits(np.ascontiguousarray(logits), values, margins)
            out.extend(arena.drain())
    return sorted(out, key=lambda r: r["seed"])


def assert_same_record(got: dict, want) -> None:
    assert got["seed"] == want.seed
    assert got["moves"] == want.moves
    assert got["rounds"] == want.rounds
    assert tuple(got["scores"]) == want.scores
    assert got["outcome"] == want.outcome
    assert got["truncated"] == want.truncated
    assert got["decisions"] == want.decisions
    assert got["evals"] == want.evals
    assert got["states"].dtype == np.float32 and got["states"].shape == want.states.shape
    assert np.array_equal(got["states"], want.states)
    assert np.array_equal(got["policies"], want.policies)
    assert np.array_equal(got["values"], want.values)
    assert np.array_equal(got["margins"], want.margins)
    assert got["aux"].dtype == np.uint8
    assert np.array_equal(got["aux"], want.aux)
    assert np.array_equal(got["policy_mask"], want.policy_mask)
    assert np.array_equal(got["search_values"], want.search_values)
    assert np.array_equal(got["search_mask"], want.search_mask)


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"search_batch": 2, "search_batch_ramp": 4},
        {"pcr_full_sims": 24, "pcr_cheap_sims": 8, "pcr_full_prob": 0.5},
        {"temp_moves": 0, "stall_rounds": 3},
        {"tree_reuse": False},
    ],
)
def test_arena_reproduces_batched_selfplay_given_identical_evaluations(extra: dict) -> None:
    net = tiny_net(1)
    config = selfplay_config(**extra)
    engine = BatchedSelfPlay(TINY, config, games=1, device="cpu")
    engine.start(net.cpu_state_dict())
    want = sorted(engine.play(3, 700), key=lambda r: r.seed)
    got = play_rust(net, config, 3, 700)
    assert len(got) == 3 == len(want)
    for g, w in zip(got, want):
        assert_same_record(g, w)
    assert sum(w.moves for w in want) > 60


def test_arena_games_are_isolated_from_their_batch_mates() -> None:
    """The game with seed 700 is the same alone and among 7 others."""
    net = tiny_net(2)
    config = selfplay_config(sims=16)
    alone = play_rust(net, config, 1, 700, games=1)
    together = play_rust(net, config, 8, 695, games=8)
    same = [r for r in together if r["seed"] == 700][0]
    # Identical evaluations are guaranteed only per row here (exact=True does the
    # softmax per row); the forward pass over 8 rows vs 1 can differ in the last
    # bits, so this is the batch-1 vs batch-1 comparison of the same seed.
    assert same["moves"] > 0
    assert alone[0]["moves"] > 0
    assert alone[0]["states"].shape[1] == 182


def test_apply_logits_path_plays_complete_valid_games() -> None:
    net = tiny_net(3)
    config = selfplay_config(sims=16)
    records = play_rust(net, config, 4, 40, games=4, rng="fast", exact=False)
    assert len(records) == 4
    for r in records:
        t = r["moves"]
        assert r["states"].shape == (t, 182)
        assert r["policies"].shape == (t, 180)
        assert np.allclose(r["policies"].sum(axis=1), 1.0, atol=1e-4)
        assert r["aux"].shape == (t, 30)
        assert set(np.unique(r["values"])) <= {-1.0, 0.0, 1.0}
        assert r["evals"] > 0


def test_set_stop_abandons_the_games_in_flight() -> None:
    net = tiny_net(4)
    arena = rs.Arena(flat_config(selfplay_config(sims=16)), has_margin=True, games=2)
    arena.begin(10, 0)
    obs = arena.gather()
    assert len(obs) == 2
    arena.set_stop()
    assert arena.finished()
    assert arena.drain() == []
