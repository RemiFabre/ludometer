"""Tests for run6's playout-cap randomization (KataGo's trick).

The bargain: each self-play **move** independently draws a deep search (whose
visit distribution becomes a policy target) with probability ``full_prob``, or a
cheap one (which does not). A cheap move's position still enters the buffer — its
value, margin and final-wall labels come from the end of the game, not from the
search — but with a zeroed policy and ``policy_mask = 0``. Expected simulations
per move fall *below* run5's flat 512 while the surviving targets are twice as
deep.

Four things have to hold, one section each:

1. **the schedule is what it says** — about ``full_prob`` of the searched moves
   are full ones, the draw is per move and reproducible from the game seed, and
   the two engines draw the same schedule for the same game;
2. **cheap moves carry no policy target, and full ones do** — in the record, in
   the buffer and after a save/load round trip;
3. **the masked rows contribute exactly zero policy gradient**, and the loss the
   unmasked rows produce is the same size it would be without them;
4. **nothing without ``pcr`` moves** — every older config runs one search per
   move with noise, bit-identically — and ``smoke6`` runs end to end.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from ludometer.azul.engine import ACTION_SPACE, ENCODED_SIZE, AzulState
from ludometer.train.mcts import MCTS, MCTSConfig, UniformEvaluator
from ludometer.train.net import make_net
from ludometer.train.net2 import StructuredConfig
from ludometer.train.replay import ReplayBuffer
from ludometer.train.selfplay import (
    SelfPlayConfig,
    pcr_rng,
    pcr_sims,
    play_selfplay_game,
)
from ludometer.train.selfplay_batched import BatchedSelfPlay
from ludometer.train.trainer import TrainConfig, Trainer

REPO = Path(__file__).resolve().parents[1]

TINY = StructuredConfig(
    embed=32,
    layers=1,
    heads=4,
    body=48,
    body_blocks=1,
    value_hidden=16,
    policy_rank=8,
    margin_head=True,
    aux_heads=True,
)


def pcr_config(full=16, cheap=4, prob=0.25, **kwargs) -> SelfPlayConfig:
    mcts_keys = {
        k: kwargs.pop(k)
        for k in list(kwargs)
        if k in MCTSConfig.__dataclass_fields__  # type: ignore[attr-defined]
    }
    mcts_keys.setdefault("sims", full)
    mcts_keys.setdefault("tree_reuse", False)
    mcts_keys.setdefault("chance_children", 2)
    kwargs.setdefault("temp_moves", 4)
    kwargs.setdefault("max_moves", 200)
    kwargs.setdefault("value_score_weight", 0.0)
    return SelfPlayConfig(
        mcts=MCTSConfig(**mcts_keys),
        pcr_full_sims=full,
        pcr_cheap_sims=cheap,
        pcr_full_prob=prob,
        **kwargs,
    )


# ------------------------------------------------------------- 1. the schedule
def test_pcr_is_off_unless_it_is_asked_for() -> None:
    plain = SelfPlayConfig()
    assert plain.pcr is False
    sims, full = pcr_sims(plain, pcr_rng(1))
    assert sims is None and full is True, "no override, and noise as before"
    # a config with a probability but no cheap budget is still off, not broken
    assert SelfPlayConfig(pcr_full_prob=0.5).pcr is False


@pytest.mark.parametrize("prob", [0.25, 0.5])
def test_the_draw_hits_full_prob_of_the_time(prob: float) -> None:
    config = pcr_config(prob=prob)
    rng = pcr_rng(7)
    draws = [pcr_sims(config, rng) for _ in range(20_000)]
    full_rate = sum(1 for _sims, full in draws if full) / len(draws)
    assert full_rate == pytest.approx(prob, abs=0.02)
    # and the budget always matches the kind of search
    assert {(s, f) for s, f in draws} == {(16, True), (4, False)}


def test_the_schedule_is_a_function_of_the_game_seed_alone() -> None:
    def schedule(seed: int) -> list:
        config, rng = pcr_config(), pcr_rng(seed)
        return [pcr_sims(config, rng) for _ in range(50)]

    a, b, c = schedule(31), schedule(31), schedule(32)
    assert a == b, "same seed, same schedule"
    assert a != c, "different games do not share a schedule"


def test_the_schedule_stream_is_its_own(monkeypatch) -> None:
    """Turning pcr on must not shift the search's or the sampler's RNG."""
    plain = pcr_config(prob=0.0)
    plain = SelfPlayConfig(**{**plain.__dict__, "pcr_full_prob": 0.0})
    a = play_selfplay_game(UniformEvaluator(), 5, plain)
    b = play_selfplay_game(UniformEvaluator(), 5, plain)
    np.testing.assert_array_equal(a.states, b.states)
    assert a.policy_mask.min() == 1.0, "pcr off: every move is a real target"


def test_a_pcr_game_records_both_kinds_of_move() -> None:
    record = play_selfplay_game(UniformEvaluator(), 21, pcr_config(prob=0.5))
    mask = record.policy_mask
    assert mask.shape == (len(record.values),)
    assert set(np.unique(mask)) == {0.0, 1.0}
    # a masked-out row is a zeroed policy, not a stale one
    assert record.policies[mask == 0.0].sum() == 0.0
    # ... and a kept row is a real visit distribution
    kept = record.policies[mask == 1.0]
    np.testing.assert_allclose(kept.sum(axis=1), 1.0, atol=1e-5)


def test_the_batched_engine_draws_the_same_schedule_as_the_sequential_one() -> None:
    """Same game, same seed, same full/cheap pattern — the equivalence extends."""
    config = pcr_config(prob=0.5)
    torch.manual_seed(2)
    net = make_net(TINY)
    net.eval()
    engine = BatchedSelfPlay(TINY, config, games=3, device="cpu")
    engine.start(net.cpu_state_dict())
    batched = {r.seed: r for r in engine.play(3, 400)}

    from ludometer.train.net import NetEvaluator

    evaluator = NetEvaluator(net, device="cpu")
    for seed, want in batched.items():
        got = play_selfplay_game(evaluator, seed, config)
        np.testing.assert_array_equal(got.policy_mask, want.policy_mask)
        np.testing.assert_allclose(got.policies, want.policies, atol=1e-6)
        np.testing.assert_array_equal(got.aux, want.aux)
        assert got.moves == want.moves


def test_a_cheap_search_really_is_cheaper() -> None:
    """The budget override reaches the search, not just the bookkeeping."""
    state = AzulState.new_game(seed=8)
    mcts = MCTS(UniformEvaluator(), MCTSConfig(sims=64), seed=1)
    deep = mcts.search(state)
    mcts.reset_tree()
    cheap = mcts.search(state, sims=8)
    assert deep.sims == 64 and cheap.sims == 8
    assert sum(cheap.visits.values()) == 8


# ------------------------------------------------------------ 2. what is stored
def test_the_buffer_keeps_the_policy_mask_through_a_save(tmp_path) -> None:
    record = play_selfplay_game(UniformEvaluator(), 21, pcr_config(prob=0.5))
    buf = ReplayBuffer(capacity=1000, seed=0)
    buf.add_game(record)
    n_full = int(record.policy_mask.sum())
    assert buf.stats()["policy_targets"] == n_full
    assert 0 < n_full < len(record.policy_mask), "the game had both kinds of move"

    path = buf.save(tmp_path / "pcr.npz")
    again = ReplayBuffer(capacity=1000, seed=0)
    again.load(path)
    assert again.stats()["policy_targets"] == n_full
    np.testing.assert_array_equal(
        again.policy_mask[: len(record.policy_mask)], record.policy_mask
    )


# -------------------------------------------------------------- 3. the loss math
def masked_config(**overrides) -> TrainConfig:
    data = {
        "run": "pcr",
        "arch": "structured",
        "margin_head": True,
        "aux_heads": True,
        "value_score_weight": 0.0,
        "embed": 32,
        "layers": 1,
        "heads": 4,
        "body": 48,
        "body_blocks": 1,
        "value_hidden": 16,
        "policy_rank": 8,
        "device": "cpu",
        "batch_size": 32,
        "replay_capacity": 2000,
        "eval_games": 0,
        "eval_at_start": False,
        "heartbeat": 0.0,
    }
    data.update(overrides)
    return TrainConfig.from_dict(data)


def _block(n: int, seed: int = 0, action: int = 0):
    """A batch whose policy target is a sharp one-hot, so the CE is informative."""
    rng = np.random.default_rng(seed)
    states = rng.random((n, ENCODED_SIZE)).astype(np.float32)
    policies = np.zeros((n, ACTION_SPACE), dtype=np.float32)
    policies[:, action] = 1.0
    values = rng.standard_normal(n).astype(np.float32)
    margins = rng.standard_normal(n).astype(np.float32)
    return states, policies, values, margins


def test_masked_rows_contribute_exactly_zero_policy_gradient(tmp_path) -> None:
    """The load-bearing claim: a cheap position is invisible to the policy head.

    Take a batch of 8 real targets. Append 8 more rows with *different* states and
    *different* policies but ``policy_mask = 0``. Both the loss and every gradient
    must come out identical to the 8-row batch — if the mask were merely a zeroed
    target row, the loss would be halved by the batch mean and the gradients with
    it.
    """
    trainer = Trainer(masked_config(), tmp_path / "run", log=None)
    states, policies, values, margins = _block(8, seed=1, action=0)
    # different positions AND a target that pulls the head somewhere else
    extra_s, extra_p, extra_v, extra_m = _block(8, seed=2, action=97)

    def run(s, p, v, m, mask):
        trainer.net.zero_grad(set_to_none=True)
        loss_p = trainer._losses(
            s, p, v, m, np.ones(len(s), np.float32), policy_mask=mask
        )[0]
        loss_p.backward()
        grads = torch.cat(
            [q.grad.reshape(-1) for q in trainer.net.parameters() if q.grad is not None]
        )
        return float(loss_p.detach()), grads

    base_loss, base_grads = run(
        states, policies, values, margins, np.ones(8, np.float32)
    )
    both_loss, both_grads = run(
        np.concatenate([states, extra_s]),
        np.concatenate([policies, extra_p]),
        np.concatenate([values, extra_v]),
        np.concatenate([margins, extra_m]),
        np.concatenate([np.ones(8, np.float32), np.zeros(8, np.float32)]),
    )
    assert both_loss == pytest.approx(base_loss, rel=1e-5)
    torch.testing.assert_close(both_grads, base_grads, atol=1e-6, rtol=1e-4)

    # and the contrast: without the mask those 8 rows DO change the answer
    unmasked_loss, unmasked_grads = run(
        np.concatenate([states, extra_s]),
        np.concatenate([policies, extra_p]),
        np.concatenate([values, extra_v]),
        np.concatenate([margins, extra_m]),
        np.ones(16, np.float32),
    )
    assert unmasked_loss != pytest.approx(base_loss, rel=1e-3)
    assert not torch.allclose(unmasked_grads, base_grads, atol=1e-6)


def test_an_all_cheap_batch_is_zero_policy_loss_and_still_trains(tmp_path) -> None:
    trainer = Trainer(masked_config(), tmp_path / "run", log=None)
    states, policies, values, margins = _block(8, seed=3)
    loss_p, loss_v, loss_m, _loss_a = trainer._losses(
        states,
        policies,
        values,
        margins,
        np.ones(8, np.float32),
        policy_mask=np.zeros(8, np.float32),
    )
    assert float(loss_p.detach()) == 0.0
    assert float(loss_v.detach()) > 0.0, "the value head learns from a cheap move"
    assert float(loss_m.detach()) > 0.0, "so does the margin head"


# ------------------------------------------------------------- 4. nothing moved
def test_every_older_config_runs_one_search_per_move() -> None:
    for name in ("run1", "run2", "run3", "run4", "run5", "smoke", "smoke3", "smoke5"):
        cfg = TrainConfig.load(REPO / "configs" / f"{name}.json")
        assert cfg.pcr == {}, name
        assert cfg.selfplay_config().pcr is False, name


def test_run6_asks_for_the_split_and_the_trainer_checks_it() -> None:
    cfg = TrainConfig.load(REPO / "configs" / "run6.json")
    cfg.validate()
    assert cfg.pcr == {"full_sims": 1024, "cheap_sims": 256, "full_prob": 0.25}
    assert cfg.sims == 1024, "sims must be the full search"
    sp = cfg.selfplay_config()
    assert sp.pcr and sp.pcr_full_sims == 1024 and sp.pcr_cheap_sims == 256
    # expected simulations per move stay below run5's flat 512
    run5_sims = TrainConfig.load(REPO / "configs" / "run5.json").sims
    expected = (
        sp.pcr_full_prob * sp.pcr_full_sims + (1 - sp.pcr_full_prob) * sp.pcr_cheap_sims
    )
    assert expected == 448.0
    assert expected < run5_sims, "pcr must not cost game volume"


@pytest.mark.parametrize(
    "bad,match",
    [
        ({"full_sims": 1024, "cheap_sims": 0, "full_prob": 0.25}, "cheap_sims >= 1"),
        ({"full_sims": 128, "cheap_sims": 256, "full_prob": 0.25}, "not exceed"),
        ({"full_sims": 1024, "cheap_sims": 256, "full_prob": 0.0}, "full_prob"),
        ({"full_sims": 1024, "cheap_sims": 256, "full_prob": 1.5}, "full_prob"),
        ({"fool_sims": 1024}, "unknown pcr keys"),
    ],
)
def test_a_broken_pcr_block_is_rejected(bad, match) -> None:
    with pytest.raises(ValueError, match=match):
        TrainConfig.from_dict({"run": "x", "sims": 1024, "pcr": bad}).validate()


def test_pcr_full_sims_must_equal_sims() -> None:
    with pytest.raises(ValueError, match="must equal sims"):
        TrainConfig.from_dict(
            {
                "run": "x",
                "sims": 512,
                "pcr": {"full_sims": 1024, "cheap_sims": 256, "full_prob": 0.25},
            }
        ).validate()


def test_smoke6_runs_end_to_end() -> None:
    """The acceptance run for the whole run6 stack, in a real subprocess."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp) / "runs"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO)
        proc = subprocess.run(
            [
                "nice",
                "-n",
                "19",
                sys.executable,
                "-m",
                "ludometer.train.run",
                "--config",
                str(REPO / "configs" / "smoke6.json"),
                "--runs-dir",
                str(runs),
                "--max-games",
                "16",
            ],
            cwd=REPO,
            env=env,
            capture_output=True,
            check=False,
            text=True,
            timeout=900,
        )
        assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
        assert "selfplay=batched" in proc.stdout
        run_dir = runs / "smoke6"
        status = json.loads((run_dir / "status.json").read_text())
        assert status["state"] == "done", status
        assert status["games"] >= 16
        entries = [
            json.loads(line)
            for line in (run_dir / "train.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert entries, "the run produced no training steps"
        assert any(e["loss_m"] > 0.0 for e in entries), "margin targets are present"
        assert any(e["loss_a"] > 0.0 for e in entries), "aux targets are present"
        assert any(e["loss_p"] > 0.0 for e in entries), "policy targets survived pcr"
        assert (run_dir / "elo.jsonl").read_text().splitlines()

        # the saved buffer is the run6 format, with both masks doing their job
        buf = ReplayBuffer(capacity=20000, seed=0)
        buf.load(run_dir / "checkpoints" / "replay.npz")
        stats = buf.stats()
        assert stats["aux_targets"] == stats["size"], "every position has a wall label"
        assert 0 < stats["policy_targets"] < stats["size"], "pcr split the positions"
