"""Teacher labels on human positions: replay is exact, the search is the search."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import torch

from ludometer.azul.engine import AzulState
from ludometer.cloud.label import (
    BatchLabeler,
    PositionGame,
    label_game,
    load_positions,
    replay_positions,
)
from ludometer.cloud.shards import read_shard, write_shard
from ludometer.train.mcts import MCTS, MCTSConfig
from ludometer.train.net import NetEvaluator, make_net
from ludometer.train.net2 import StructuredConfig
from ludometer.train.selfplay_batched import BatchEvaluator

TINY = StructuredConfig(
    embed=32,
    layers=1,
    heads=4,
    ffn_mult=2,
    body=48,
    body_blocks=1,
    value_hidden=16,
    policy_rank=8,
    margin_head=True,
)


def _random_game(seed: int) -> tuple[PositionGame, list[np.ndarray]]:
    """A random-play game, recorded the way a BGA log would be: deals + actions."""
    rng = np.random.default_rng(seed)
    state = AzulState.new_game(seed)
    deals = [[list(f) for f in state.factories]]
    first = state.current_player
    actions: list[int] = []
    encoded: list[np.ndarray] = []
    while not state.is_terminal and len(actions) < 400:
        legal = state.legal_actions()
        a = int(rng.choice(legal))
        encoded.append(state.encode())
        actions.append(a)
        r = state.round_index
        state.apply(a)
        if state.round_index > r and not state.is_terminal:
            deals.append([list(f) for f in state.factories])
    game = PositionGame(
        table_id=seed,
        first_seat=first,
        deals=deals,
        actions=actions,
        outcome=float(state.outcome() or 0.0),
        scores=(int(state.scores[0]), int(state.scores[1])),
    )
    return game, encoded


def test_replay_reaches_the_recorded_positions(tmp_path: Path) -> None:
    game, encoded = _random_game(3)
    states, movers, final = replay_positions(game)
    assert len(states) == len(encoded) == len(movers)
    np.testing.assert_array_equal(
        np.stack([s.encode() for s in states]), np.stack(encoded)
    )
    assert (
        final.is_terminal
        and (int(final.scores[0]), int(final.scores[1])) == game.scores
    )
    # the compact file round-trips
    with gzip.open(tmp_path / "p.json.gz", "wt") as fh:
        json.dump({"format": 1, "games": [game.to_json()]}, fh)
    (back,) = load_positions(tmp_path / "p.json.gz")
    assert back == game


def test_labeler_matches_a_plain_search() -> None:
    torch.manual_seed(0)
    net = make_net(TINY)
    net.eval()
    evaluator = BatchEvaluator(net, device="cpu")
    cfg = MCTSConfig(sims=16, tree_reuse=False, chance_children=2)
    game, _ = _random_game(4)
    states, _m, _f = replay_positions(game)
    picks = states[:3]
    labeler = BatchLabeler(evaluator, cfg, slots=1, sims=16, seed=11)
    got = labeler.label(picks)
    single = NetEvaluator(net)
    for i, (policy, value, _margin) in enumerate(got):
        ref = MCTS(
            single, cfg, seed=(11 * 1_000_003 + i) & 0x7FFFFFFF, add_noise=False
        )
        res = ref.search(picks[i], add_noise=False, sims=16)
        np.testing.assert_array_equal(policy, res.policy)
        assert value == float(res.value)
    # concurrent slots give the same policies (each tree is its own search)
    wide = BatchLabeler(evaluator, cfg, slots=3, sims=16, seed=11).label(picks)
    for (p1, _v1, _m1), (p2, _v2, _m2) in zip(got, wide):
        np.testing.assert_allclose(p1, p2, atol=1e-6)


def test_label_game_yields_a_full_record(tmp_path: Path) -> None:
    torch.manual_seed(0)
    net = make_net(TINY)
    net.eval()
    labeler = BatchLabeler(
        BatchEvaluator(net, device="cpu"),
        MCTSConfig(sims=8, chance_children=2),
        slots=8,
        sims=8,
        seed=1,
    )
    game, encoded = _random_game(5)
    rec = label_game(labeler, game)
    assert len(rec) == len(encoded) and rec.decisions == len(encoded)
    np.testing.assert_array_equal(rec.states, np.stack(encoded))
    np.testing.assert_allclose(rec.policies.sum(axis=1), 1.0, atol=1e-5)
    assert rec.search_mask.sum() == len(encoded) and rec.policy_mask.sum() == len(
        encoded
    )
    assert set(np.unique(rec.values)) <= {-1.0, 0.0, 1.0}
    assert rec.aux.shape == (len(encoded), 30)
    (back,), meta = read_shard(
        write_shard(tmp_path / "s.npz", [rec], {"source": "bga"})
    )
    assert meta["source"] == "bga" and back.seed == game.table_id
