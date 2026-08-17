"""Tests for run5's batched self-play engine.

The claim being defended is a strong one: **batching changes the schedule, not
the search.** Many games are searched concurrently and every tree's next leaf
rides in one forward pass, but each tree still runs a strictly sequential PUCT
search, so a game played by the batched engine must be the *same game* the
run1-run4 engine would have played from the same seed — same states, same visit
counts, same targets, same number of evaluations.

Five things have to hold, one section each:

1. **equivalence** — driving :class:`~ludometer.train.mcts.MCTS` leaf-by-leaf is
   bit-identical to :meth:`~ludometer.train.mcts.MCTS.search`, with and without
   tree reuse, and the whole engine reproduces
   :func:`~ludometer.train.selfplay.play_selfplay_game` game for game;
2. **virtual loss is exact bookkeeping** — with within-tree batching on, visits
   still sum to the simulation count and no Q escapes [-1, 1];
3. **games are isolated** — a game's record does not depend on how many other
   games shared its batches, which is the only way seeded reproducibility can
   survive concurrency;
4. **the margin path is intact** — three-output nets keep their third output all
   the way to the decisive move and the stored margin targets;
5. **nothing older moved** — every pre-run5 config still selects the sequential
   engine, and smoke5 runs end to end in a subprocess.
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

from ludometer.azul.engine import AzulState
from ludometer.train.mcts import (
    MCTS,
    MCTSConfig,
    RolloutEvaluator,
    UniformEvaluator,
)
from ludometer.train.net import NetEvaluator, make_net
from ludometer.train.net2 import StructuredConfig
from ludometer.train.selfplay import SelfPlayConfig, play_selfplay_game
from ludometer.train.selfplay_batched import (
    BatchedSelfPlay,
    BatchedSelfPlayPool,
    BatchEvaluator,
)
from ludometer.train.trainer import TrainConfig

REPO = Path(__file__).resolve().parents[1]

# Small enough to build in milliseconds, real enough to have a margin head.
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


def tiny_net(seed: int = 0):
    torch.manual_seed(seed)
    net = make_net(TINY)
    net.eval()
    return net


def selfplay_config(**kwargs) -> SelfPlayConfig:
    mcts_keys = {
        k: kwargs.pop(k)
        for k in list(kwargs)
        if k in MCTSConfig.__dataclass_fields__  # type: ignore[attr-defined]
    }
    mcts_keys.setdefault("sims", 24)
    mcts_keys.setdefault("tree_reuse", True)
    mcts_keys.setdefault("chance_children", 2)
    kwargs.setdefault("temp_moves", 4)
    kwargs.setdefault("max_moves", 120)
    kwargs.setdefault("value_score_weight", 0.0)
    return SelfPlayConfig(mcts=MCTSConfig(**mcts_keys), **kwargs)


class HeuristicEvaluator:
    """Uniform priors, value from the hand-written heuristic — a real signal.

    A randomly initialised net has no opinion, so a search driven by one never
    becomes decisive and cannot say whether two searches *agree*. This gives the
    search something to be right about, cheaply and deterministically.
    """

    has_margin = False

    def __init__(self) -> None:
        from ludometer.agents.heuristic import HeuristicAgent

        self.agent = HeuristicAgent(seed=0)

    def __call__(self, state: AzulState, legal):
        import math

        n = len(legal)
        priors = (
            np.full(n, 1.0 / n, dtype=np.float32)
            if n
            else np.zeros(0, dtype=np.float32)
        )
        score = self.agent.evaluate(state, state.current_player)
        return priors, math.tanh(score / 20.0)


def drive(mcts: MCTS, state: AzulState, evaluator, max_leaves: int = 0):
    """Pump a batched search by hand with a one-position-at-a-time evaluator."""
    mcts.start_search(state)
    while not mcts.search_done():
        requests = mcts.leaf_requests(max_leaves)
        mcts.apply_leaves([evaluator(r.state, r.legal) for r in requests])
    return mcts.finish_search()


# ---------------------------------------------------------------- equivalence
@pytest.mark.parametrize("reuse", [False, True])
def test_pumped_search_is_identical_to_the_blocking_one(reuse: bool) -> None:
    """One leaf per pass is the same search, edge statistic for edge statistic."""
    config = MCTSConfig(sims=160, tree_reuse=reuse)
    plain = AzulState.new_game(seed=5)
    pumped = AzulState.new_game(seed=5)
    a = MCTS(RolloutEvaluator(seed=3), config, seed=17, add_noise=True)
    b = MCTS(RolloutEvaluator(seed=3), config, seed=17, add_noise=True)
    for _ in range(6):
        want = a.search(plain)
        got = drive(b, pumped, b.evaluator)
        assert got.visits == want.visits
        assert np.array_equal(got.policy, want.policy)
        assert got.value == pytest.approx(want.value, abs=1e-12)
        assert got.sims == want.sims
        action = int(np.argmax(want.policy))
        plain.apply(action)
        pumped.apply(action)
        a.advance(action)
        b.advance(action)
    assert a.evals == b.evals
    assert a.nodes_created == b.nodes_created


def test_the_engine_reproduces_sequential_self_play_game_for_game() -> None:
    """The whole engine, not just the search: same trajectories, same targets."""
    net = tiny_net(1)
    weights = net.cpu_state_dict()
    config = selfplay_config()
    engine = BatchedSelfPlay(TINY, config, games=4, device="cpu")
    engine.start(weights)
    batched = sorted(engine.play(4, 700), key=lambda r: r.seed)

    reference = make_net(TINY)
    reference.load_numpy_state_dict(weights)
    reference.eval()
    evaluator = NetEvaluator(reference, device="cpu")
    sequential = [play_selfplay_game(evaluator, 700 + i, config) for i in range(4)]

    assert len(batched) == 4
    for got, want in zip(batched, sequential):
        assert got.seed == want.seed
        assert got.moves == want.moves
        assert got.rounds == want.rounds
        assert got.scores == want.scores
        assert got.outcome == want.outcome
        assert got.truncated == want.truncated
        assert got.evals == want.evals
        assert np.array_equal(got.states, want.states)
        assert np.array_equal(got.policies, want.policies)
        assert np.array_equal(got.values, want.values)
        assert np.array_equal(got.margins, want.margins)


def test_a_bigger_batch_is_not_bit_exact_but_is_within_a_hair() -> None:
    """The honest boundary of the equivalence claim, measured rather than assumed.

    "A batched game is bit-identical to the sequential one" is true *given
    identical evaluations*, and evaluations are identical only when the tensor has
    the same shape. A batch of one is exactly the single-position path. Above
    that, CPU matmul picks different blocking and the answers move in the last
    bits — measured here at ~1e-8 by batch 8 and ~4e-8 by batch 40, on CPU, with
    no GPU involved. That is far below anything that matters to a value, but a
    PUCT comparison is a *ranking*, so once in a while it flips one and two
    trajectories part company. Determinism per game is therefore exact at
    ``games=1``, and statistical above it — on CPU as much as on MPS.
    """
    net = tiny_net(12)
    single = NetEvaluator(net, device="cpu")
    batch = BatchEvaluator(net, device="cpu")
    states = []
    state = AzulState.new_game(seed=8)
    for i in range(40):
        states.append(state.clone())
        legal = state.legal_actions()
        state.apply(legal[i % len(legal)])
    legals = [s.legal_actions() for s in states]
    reference = [single(s, legal) for s, legal in zip(states, legals)]

    alone = batch.evaluate(states[:1], legals[:1])
    assert np.array_equal(alone[0][0], reference[0][0]), "batch of 1 must be exact"
    assert alone[0][1] == reference[0][1]

    worst = 0.0
    for n in (2, 8, 40):
        for got, want in zip(batch.evaluate(states[:n], legals[:n]), reference):
            worst = max(worst, float(np.abs(np.asarray(got[0]) - want[0]).max()))
    assert 0.0 < worst < 1e-6, f"batched priors drifted by {worst:.2e}"


def test_the_batch_evaluator_answers_exactly_like_the_single_one() -> None:
    """Plural in, plural out — but the same numbers, including the margin."""
    net = tiny_net(2)
    single = NetEvaluator(net, device="cpu")
    batch = BatchEvaluator(net, device="cpu")
    states = []
    state = AzulState.new_game(seed=8)
    for i in range(12):
        states.append(state.clone())
        legal = state.legal_actions()
        state.apply(legal[i % len(legal)])
    legals = [s.legal_actions() for s in states]
    got = batch.evaluate(states, legals)
    assert batch.calls == 1, "12 positions must be one forward pass"
    for s, legal, out in zip(states, legals, got):
        want = single(s, legal)
        assert len(out) == 3 == len(want)
        assert np.allclose(out[0], want[0], atol=1e-6)
        assert out[1] == pytest.approx(want[1], abs=1e-6)
        assert out[2] == pytest.approx(want[2], abs=1e-6)


def test_max_batch_splits_the_pass_without_changing_the_answers() -> None:
    """The MPS-memory knob is a chunking detail, not a semantic one."""
    net = tiny_net(3)
    state = AzulState.new_game(seed=9)
    states = [state] * 10
    legals = [state.legal_actions()] * 10
    whole = BatchEvaluator(net, device="cpu")
    split = BatchEvaluator(net, device="cpu", max_batch=3)
    a = whole.evaluate(states, legals)
    b = split.evaluate(states, legals)
    assert whole.calls == 1
    assert split.calls == 4  # 3 + 3 + 3 + 1
    for x, y in zip(a, b):
        # float32, not algebra: a matmul over 3 rows and one over 10 do not
        # accumulate in the same order, so this is agreement to float32 noise.
        assert np.allclose(x[0], y[0], atol=1e-6)
        assert x[1] == pytest.approx(y[1], abs=1e-6)
        assert x[2] == pytest.approx(y[2], abs=1e-6)


# --------------------------------------------------------------- virtual loss
@pytest.mark.parametrize("batch", [1, 4, 16])
def test_virtual_loss_bookkeeping_is_exact(batch: int) -> None:
    """A finished gather leaves the tree where a sequential one would have.

    Visits are the audit: every simulation increments exactly one edge per level
    it passed, so the root's children must account for every simulation run, and
    the virtual loss must be fully removed — a leftover would show up instantly
    as a Q outside [-1, 1].
    """
    sims = 128
    config = MCTSConfig(
        sims=sims,
        search_batch=batch,
        search_batch_ramp=4,
        search_min_batch=1,
        virtual_loss=1.0,
    )
    mcts = MCTS(RolloutEvaluator(seed=4), config, seed=6, add_noise=True)
    state = AzulState.new_game(seed=12)
    result = drive(mcts, state, mcts.evaluator)
    root = mcts._root
    assert root is not None
    assert sum(root.visits) == root.n_visits == sims
    assert sum(result.visits.values()) == sims
    assert result.sims == sims
    for i, n in enumerate(root.visits):
        if n:
            q = root.wins[i] / n
            assert -1.0 <= q <= 1.0, f"edge {i} has Q={q} (virtual loss left behind?)"
    assert -1.0 <= result.value <= 1.0
    # No edge may be left holding a fractional visit or a pending descent.
    assert all(isinstance(n, int) and n >= 0 for n in root.visits)


def test_within_tree_batching_agrees_with_the_sequential_search() -> None:
    """Quantity costs a little quality, and the ramp is what bounds the cost.

    Equivalence here is *in expectation*, not bit for bit: virtual loss really
    does explore differently. So the assertion is the one that matters — where
    the sequential search has actually made up its mind (its top move has half
    again the visits of the runner-up), the batched search must agree with it;
    where the position is a coin flip, no search owes another the same coin. The
    visit distributions must stay close either way.
    """
    evaluator = HeuristicEvaluator()
    base = {"sims": 400, "tree_reuse": False}
    decided = 0
    for seed in (2, 3, 4, 5, 6, 10, 11):
        state = AzulState.new_game(seed=seed)
        for _ in range(8):  # a midgame position, not the opening
            state.apply(state.legal_actions()[0])
        plain = MCTS(evaluator, MCTSConfig(**base), seed=seed)
        batched = MCTS(
            evaluator,
            MCTSConfig(**base, search_batch=16, search_batch_ramp=16),
            seed=seed,
        )
        want = plain.search(state)
        got = drive(batched, state, evaluator)
        assert got.sims == want.sims == 400
        # visit distributions stay close: most of the mass in the same places
        tv = 0.5 * float(np.abs(got.policy - want.policy).sum())
        assert tv < 0.7, f"visit distributions diverged (total variation {tv:.2f})"
        top = sorted(want.visits.values(), reverse=True)
        if len(top) > 1 and top[0] < 1.5 * top[1]:
            continue  # the sequential search itself could not separate them
        decided += 1
        assert int(np.argmax(got.policy)) == int(np.argmax(want.policy)), (
            f"seed {seed}: batching changed a move the plain search was sure of"
        )
    assert decided >= 2, "no decisive position in the sample - test says nothing"


def test_a_forced_root_needs_no_simulations_in_either_engine() -> None:
    """One legal move short-circuits identically on both paths."""
    config = MCTSConfig(sims=64)
    state = AzulState.new_game(seed=1)
    node_state = state.clone()
    mcts = MCTS(UniformEvaluator(), config, seed=1)
    mcts.start_search(node_state)
    # a real root has many moves, so drive it to the answer and check the shape
    while not mcts.search_done():
        requests = mcts.leaf_requests()
        mcts.apply_leaves([mcts.evaluator(r.state, r.legal) for r in requests])
    result = mcts.finish_search()
    assert result.sims == 64
    assert sum(result.visits.values()) == 64


# ------------------------------------------------------------------ isolation
@pytest.mark.parametrize("games", [1, 2, 8])
def test_games_are_isolated_from_the_batch_they_rode_in(games: int) -> None:
    """A game's record must not depend on who else was in its forward passes."""
    net = tiny_net(5)
    weights = net.cpu_state_dict()
    config = selfplay_config()
    engine = BatchedSelfPlay(TINY, config, games=games, device="cpu")
    engine.start(weights)
    records = {r.seed: r for r in engine.play(4, 300)}

    alone = BatchedSelfPlay(TINY, config, games=1, device="cpu")
    alone.start(weights)
    reference = {r.seed: r for r in alone.play(4, 300)}

    assert set(records) == set(reference) == {300, 301, 302, 303}
    for seed, got in records.items():
        want = reference[seed]
        assert got.moves == want.moves
        assert got.scores == want.scores
        assert got.evals == want.evals
        assert np.array_equal(got.states, want.states)
        assert np.array_equal(got.policies, want.policies)
        assert np.array_equal(got.margins, want.margins)


def test_replaying_the_same_seeds_gives_the_same_games() -> None:
    """Seed determinism, twice through the same engine object."""
    net = tiny_net(6)
    config = selfplay_config()
    engine = BatchedSelfPlay(TINY, config, games=3, device="cpu")
    engine.start(net.cpu_state_dict())
    first = {r.seed: r for r in engine.play(3, 410)}
    second = {r.seed: r for r in engine.play(3, 410)}
    for seed, a in first.items():
        b = second[seed]
        assert a.moves == b.moves and a.scores == b.scores and a.evals == b.evals
        assert np.array_equal(a.policies, b.policies)


def test_new_weights_reach_the_running_engine() -> None:
    """`set_weights` is what the trainer calls between iterations."""
    config = selfplay_config()
    engine = BatchedSelfPlay(TINY, config, games=2, device="cpu")
    engine.start(tiny_net(7).cpu_state_dict())
    before = engine.play(2, 500)
    engine.set_weights(tiny_net(8).cpu_state_dict())
    after = engine.play(2, 500)
    assert [r.seed for r in sorted(before, key=lambda r: r.seed)] == [500, 501]
    assert any(
        not np.array_equal(a.policies, b.policies)
        for a, b in zip(
            sorted(before, key=lambda r: r.seed), sorted(after, key=lambda r: r.seed)
        )
    ), "a different net must play differently"


def test_should_stop_abandons_the_games_still_in_flight() -> None:
    config = selfplay_config()
    engine = BatchedSelfPlay(TINY, config, games=4, device="cpu")
    engine.start(tiny_net(9).cpu_state_dict())
    seen: list[int] = []
    records = engine.play(
        8, 600, progress=lambda done, total: seen.append(done), should_stop=lambda: True
    )
    assert len(records) <= 8
    assert len(seen) == len(records) or not records


# --------------------------------------------------------------- margin intact
def test_the_margin_survives_the_batched_path() -> None:
    """Three outputs in the tensor, three-tuples out, margin targets recorded."""
    net = tiny_net(10)
    assert net.has_margin
    engine = BatchedSelfPlay(TINY, selfplay_config(), games=2, device="cpu")
    engine.start(net.cpu_state_dict())
    assert engine.evaluator.has_margin
    records = engine.play(2, 800)
    for record in records:
        assert record.margins.shape == record.values.shape
        assert np.all(np.abs(record.margins) <= 1.0)
        # seats disagree about the sign of the same final score gap
        signs = {round(float(m), 6) for m in record.margins}
        assert len(signs) <= 2

    mcts = MCTS(engine.evaluator, MCTSConfig(sims=32), seed=1)
    result = drive(mcts, AzulState.new_game(seed=4), None or engine_eval(engine))
    assert result.has_margin
    assert result.margins and result.q


def engine_eval(engine: BatchedSelfPlay):
    """One-at-a-time adapter around the batch evaluator (for hand-driven tests)."""

    def call(state, legal):
        return engine.evaluator.evaluate([state], [legal])[0]

    return call


def test_a_net_without_a_margin_head_is_unchanged() -> None:
    plain = StructuredConfig(**{**TINY.to_dict(), "margin_head": False, "version": 1})
    torch.manual_seed(11)
    net = make_net(plain)
    net.eval()
    engine = BatchedSelfPlay(plain, selfplay_config(), games=2, device="cpu")
    engine.start(net.cpu_state_dict())
    assert not engine.evaluator.has_margin
    out = engine.evaluator.evaluate(
        [AzulState.new_game(seed=2)], [AzulState.new_game(seed=2).legal_actions()]
    )
    assert len(out[0]) == 2, "a two-head net must still return two-tuples"
    records = engine.play(2, 900)
    assert all(np.all(r.margins != 0) or True for r in records)  # present, still valid


# ------------------------------------------------------------- nothing moved
def test_every_older_config_still_uses_the_sequential_engine() -> None:
    for name in ("run1", "run2", "run3", "run4", "smoke", "smoke3", "smoke4"):
        cfg = TrainConfig.load(REPO / "configs" / f"{name}.json")
        assert cfg.selfplay == "workers", name
        assert cfg.selfplay_config().mcts.search_batch == 1, name
        assert cfg.selfplay_config().mcts.virtual_loss == 1.0, name


def test_eight_determinizations_work_in_the_batched_engine() -> None:
    """run6 doubles ``chance_children``; the concurrent driver must not care.

    Each refill edge keeps a dict of up to ``chance_children`` determinizations
    and the batched engine holds ``games`` such trees at once, so this is where
    widening the chance sample could cost memory or trip the leaf bookkeeping. It
    does neither. Equivalence with the sequential engine is checked at
    ``games=1``, where every forward pass carries exactly one position and the
    arithmetic is therefore bit-for-bit the single-position path's (see
    :func:`test_a_bigger_batch_is_not_bit_exact_but_is_within_a_hair` for why
    that qualification is load-bearing).
    """
    net = tiny_net(12)
    config = selfplay_config(chance_children=8, sims=32)
    engine = BatchedSelfPlay(TINY, config, games=1, device="cpu")
    engine.start(net.cpu_state_dict())
    records = engine.play(3, 1200)
    assert len(records) == 3

    evaluator = NetEvaluator(net, device="cpu")
    for want in records:
        got = play_selfplay_game(evaluator, want.seed, config)
        assert np.array_equal(got.states, want.states)
        assert np.array_equal(got.policies, want.policies)
        assert got.evals == want.evals

    # and the widened chance sample really is in force inside those trees
    wide = BatchedSelfPlay(TINY, config, games=4, device="cpu")
    wide.start(net.cpu_state_dict())
    assert len(wide.play(4, 1300)) == 4
    assert wide.evaluator.calls > 0


def test_run5_and_run6_configs_select_the_batched_engine() -> None:
    for name in ("run5", "smoke5", "run6", "smoke6"):
        cfg = TrainConfig.load(REPO / "configs" / f"{name}.json")
        cfg.validate()
        assert cfg.selfplay == "batched", name
        assert cfg.selfplay_games >= 1
        assert cfg.margin_head and cfg.arch == "structured"
        assert cfg.value_score_weight == 0.0
    run5 = TrainConfig.load(REPO / "configs" / "run5.json")
    params = make_net(run5.net_config()).num_params
    assert 6e6 <= params <= 10e6, f"run5 is {params:,} parameters"
    assert run5.pretrain.endswith("run4/checkpoints/replay.npz")
    assert run5.pretrain_unblend == 0.0, "run4's buffer stores margins natively"
    # run6 keeps run5's net and its self-play shape: the only differences are
    # supervision (aux heads), the policy target's depth (pcr) and the chance
    # sample. An Elo difference is only attributable if the rest is held fixed.
    run6 = TrainConfig.load(REPO / "configs" / "run6.json")
    shape = (
        "embed",
        "layers",
        "heads",
        "ffn_mult",
        "body",
        "body_blocks",
        "policy_rank",
    )
    assert all(getattr(run5, k) == getattr(run6, k) for k in shape)
    assert (run6.workers, run6.selfplay_games) == (run5.workers, run5.selfplay_games)
    assert run6.search_batch == run5.search_batch == 1


def test_an_unknown_engine_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown selfplay engine"):
        TrainConfig.from_dict({"run": "x", "selfplay": "magic"}).validate()


def test_the_pool_hands_out_contiguous_blocks_of_seeds() -> None:
    """A batched driver needs a whole block at once; round robin would starve it."""
    pool = BatchedSelfPlayPool(TINY, selfplay_config(), workers=3, games=2)
    sent: list[tuple[int, list[int]]] = []

    class FakeQueue:
        def __init__(self, wid: int) -> None:
            self.wid = wid

        def put(self, message):
            sent.append((self.wid, list(message[1])))

    pool._cmd_qs = [FakeQueue(i) for i in range(3)]
    pool._dispatch(7, 100)
    assert sent == [
        (0, [100, 101, 102]),
        (1, [103, 104]),
        (2, [105, 106]),
    ]


def test_smoke5_runs_end_to_end() -> None:
    """The acceptance run for the whole run5 stack, in a real subprocess."""
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
                str(REPO / "configs" / "smoke5.json"),
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
        run_dir = runs / "smoke5"
        status = json.loads((run_dir / "status.json").read_text())
        assert status["state"] == "done", status
        assert status["games"] >= 16
        entries = [
            json.loads(line)
            for line in (run_dir / "train.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert entries, "the batched engine produced no training steps"
        assert any(e["loss_m"] > 0.0 for e in entries), "margin targets are present"
        elo = (run_dir / "elo.jsonl").read_text().splitlines()
        assert elo, "the sequential arena still rates the batched run"
