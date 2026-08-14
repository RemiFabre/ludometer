"""Tests for the replay buffer and the self-play trajectory format.

The buffer is part of the resumable state, so the round trip through disk has to
preserve contents *in ring order* plus the counters the trainer reports.
"""

from __future__ import annotations

import numpy as np
import pytest

from ludometer.azul.engine import ACTION_SPACE, ENCODED_SIZE
from ludometer.train.mcts import MCTSConfig, UniformEvaluator
from ludometer.train.net import NetConfig
from ludometer.train.replay import ReplayBuffer
from ludometer.train.selfplay import (
    GameRecord,
    InlineSelfPlay,
    SelfPlayConfig,
    play_selfplay_game,
    value_target,
)


def make_block(n: int, start: float = 0.0) -> tuple[np.ndarray, ...]:
    states = np.full((n, ENCODED_SIZE), 0.0, dtype=np.float32)
    states[:, 0] = np.arange(start, start + n)
    policies = np.zeros((n, ACTION_SPACE), dtype=np.float32)
    policies[:, 0] = 1.0
    values = np.arange(start, start + n, dtype=np.float32)
    return states, policies, values


def test_add_and_sample() -> None:
    buf = ReplayBuffer(capacity=100, seed=1)
    assert len(buf) == 0
    with pytest.raises(ValueError):
        buf.sample(4)
    buf.add(*make_block(30))
    assert len(buf) == 30
    assert buf.total_added == 30
    states, policies, values = buf.sample(16)
    assert states.shape == (16, ENCODED_SIZE)
    assert policies.shape == (16, ACTION_SPACE)
    assert values.shape == (16,)
    assert set(np.unique(values)).issubset(set(range(30)))


def test_ring_overwrites_the_oldest_positions() -> None:
    buf = ReplayBuffer(capacity=10, seed=0)
    buf.add(*make_block(8, start=0))
    buf.add(*make_block(6, start=8))
    assert len(buf) == 10
    assert buf.total_added == 14
    # the 10 newest (4..13) survived, the first four were overwritten
    assert sorted(buf.values.tolist()) == list(range(4, 14))


def test_block_larger_than_capacity_keeps_the_tail() -> None:
    buf = ReplayBuffer(capacity=5, seed=0)
    buf.add(*make_block(12))
    assert len(buf) == 5
    assert sorted(buf.values.tolist()) == [7, 8, 9, 10, 11]


def test_save_load_roundtrip_preserves_ring_order_and_counters(tmp_path) -> None:
    buf = ReplayBuffer(capacity=10, seed=3)
    buf.add(*make_block(8, start=0))
    buf.add(*make_block(6, start=8))  # wraps
    buf.games_added = 4
    path = buf.save(tmp_path / "replay.npz")
    assert path.exists()

    other = ReplayBuffer(capacity=10, seed=3)
    n = other.load(path)
    assert n == len(buf) == 10
    assert other.total_added == buf.total_added
    assert other.games_added == 4
    # oldest-to-newest ordering means the next write overwrites the oldest again
    np.testing.assert_array_equal(other.values, np.arange(4, 14, dtype=np.float32))
    other.add(*make_block(1, start=99))
    assert 4.0 not in other.values.tolist()
    assert 99.0 in other.values.tolist()


def test_load_into_a_bigger_buffer(tmp_path) -> None:
    small = ReplayBuffer(capacity=6, seed=0)
    small.add(*make_block(6))
    path = small.save(tmp_path / "b.npz")
    big = ReplayBuffer(capacity=50, seed=0)
    assert big.load(path) == 6
    assert big.position == 6
    assert big.stats()["capacity"] == 50


def test_game_record_shapes_and_value_targets() -> None:
    config = SelfPlayConfig(mcts=MCTSConfig(sims=8), temp_moves=4)
    record = play_selfplay_game(UniformEvaluator(), seed=17, config=config)
    assert isinstance(record, GameRecord)
    n = len(record)
    assert record.states.shape == (n, ENCODED_SIZE)
    assert record.policies.shape == (n, ACTION_SPACE)
    assert record.values.shape == (n,)
    assert n == record.moves
    assert record.policies.sum(axis=1) == pytest.approx(np.ones(n), abs=1e-5)
    assert record.outcome in (-1.0, 0.0, 1.0)
    # one magnitude for the whole game, opposite signs for the two seats
    assert len(set(np.abs(record.values).round(5).tolist())) == 1
    assert np.all(np.sign(record.values[record.values != 0]) != 0)

    buf = ReplayBuffer(capacity=1000, seed=0)
    assert buf.add_game(record) == n
    assert buf.games_added == 1
    assert len(buf) == n


def test_value_target_blends_the_outcome_with_the_score_margin() -> None:
    pure = SelfPlayConfig(value_score_weight=0.0)
    blended = SelfPlayConfig(value_score_weight=0.15)
    assert value_target(1.0, 30, pure) == 1.0
    assert value_target(0.0, 0, blended) == 0.0
    # the winner keeps the sign; the margin only grades the magnitude
    big = value_target(1.0, 40, blended)
    small = value_target(1.0, 1, blended)
    assert 0.8 < small < big <= 1.0
    assert value_target(-1.0, -40, blended) == pytest.approx(-big)
    # a 0-0 draw is flat, but a drawn-by-tiebreak game with a margin is not
    assert value_target(0.0, 6, blended) > 0.0


def test_a_stalling_game_is_truncated_and_scored_as_a_draw() -> None:
    """Deterministic policies can loop forever; the move cap ends the game."""
    config = SelfPlayConfig(mcts=MCTSConfig(sims=4), temp_moves=0, max_moves=6)
    record = play_selfplay_game(UniformEvaluator(), seed=17, config=config)
    assert record.truncated
    assert record.moves == 6
    assert len(record) == 6
    assert record.outcome == 0.0
    assert not record.values.any()  # a draw is worth zero to both seats


def test_inline_selfplay_matches_the_pool_api() -> None:
    inline = InlineSelfPlay(
        NetConfig(hidden=16, blocks=1, value_hidden=8),
        SelfPlayConfig(mcts=MCTSConfig(sims=4), temp_moves=2),
    )
    inline.start(inline.net.cpu_state_dict())
    seen: list[tuple[int, int]] = []
    records = inline.play(2, seed_start=5, progress=lambda d, t: seen.append((d, t)))
    inline.close()
    assert len(records) == 2
    assert [r.seed for r in records] == [5, 6]
    assert seen == [(1, 2), (2, 2)]
