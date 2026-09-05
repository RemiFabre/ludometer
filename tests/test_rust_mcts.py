"""The Rust PUCT tree (`ludometer_rs.Tree`) against `ludometer.train.mcts.MCTS`.

Acceptance layer 2 of docs/RUST_ENGINE.md §6: given identical evaluations and no
root noise, the two searches must produce identical visit counts, root values,
policies, win-Q and margin-Q — on isolated positions, across whole games with
tree reuse, through the leaf protocol, and with within-tree batching (virtual
loss). The Rust tree's `rng="python"` mode reproduces CPython's `random.Random`,
so the chance-table pick past `chance_children` and the determinization seeds
match too; only Dirichlet noise is statistical (and is off here).

Skipped when `ludometer_rs` is not built (`rust/README.md` says how).
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from ludometer.azul.engine import AzulState
from ludometer.train.mcts import MCTS, MCTSConfig, UniformEvaluator, select_action

rs = pytest.importorskip("ludometer_rs")


# ------------------------------------------------------------------ evaluators
class ScoreEvaluator:
    """Deterministic, net-free: value = tanh(current score margin / 15)."""

    has_margin = False

    def __call__(self, state, legal):
        n = len(legal)
        priors = np.full(n, 1.0 / n, dtype=np.float32) if n else np.zeros(0, np.float32)
        me = state.current_player
        margin = state.scores[me] - state.scores[1 - me]
        margin += 2 * (state.completed_rows(me) - state.completed_rows(1 - me))
        return priors, float(np.tanh(margin / 15.0))


class MarginEvaluator(ScoreEvaluator):
    """Same, with a third output so the margin path is exercised."""

    has_margin = True

    def __call__(self, state, legal):
        priors, value = super().__call__(state, legal)
        me = state.current_player
        return (
            priors,
            value,
            math.tanh((state.scores[me] - state.scores[1 - me]) / 20.0),
        )


class SkewedEvaluator:
    """Non-uniform priors (a fixed hash of the action), so PUCT has something to rank."""

    has_margin = False

    def __call__(self, state, legal):
        raw = np.array([((a * 2654435761) % 97) + 1 for a in legal], dtype=np.float64)
        priors = (raw / raw.sum()).astype(np.float32)
        me = state.current_player
        return priors, float(np.tanh((state.scores[me] - state.scores[1 - me]) / 10.0))


# --------------------------------------------------------------------- helpers
def twin_positions(seed: int, n: int) -> list[tuple[AzulState, rs.State]]:
    """`n` positions along one random-play game, on both engines."""
    py = AzulState.new_game(seed)
    rust = rs.State.new_game(seed, rng="python")
    picker = random.Random(seed)
    out = []
    while len(out) < n and not py.is_terminal:
        legal = py.legal_actions()
        if len(legal) > 1:
            out.append((py.clone(), rust.clone()))
        a = legal[picker.randrange(len(legal))]
        py.apply(a)
        rust.apply(a)
    return out


def cfg_dict(config: MCTSConfig) -> dict:
    return {k: getattr(config, k) for k in MCTSConfig.__dataclass_fields__}


def assert_same_result(got: dict, want, where: str = "") -> None:
    assert got["visits"] == want.visits, where
    assert np.array_equal(got["policy"], want.policy), where
    assert got["value"] == pytest.approx(want.value, abs=1e-12), where
    assert got["sims"] == want.sims, where
    assert got["has_margin"] == want.has_margin, where
    if want.has_margin:
        assert got["q"].keys() == want.q.keys(), where
        for a in want.q:
            assert got["q"][a] == pytest.approx(want.q[a], abs=1e-12), where
            assert got["margins"][a] == pytest.approx(want.margins[a], abs=1e-12), where
        assert got["margin"] == pytest.approx(want.margin, abs=1e-12), where


# ------------------------------------------------------------ isolated searches
@pytest.mark.parametrize(
    "evaluator,chance_children",
    [
        (UniformEvaluator(), 1),
        (ScoreEvaluator(), 1),
        (ScoreEvaluator(), 4),
        (MarginEvaluator(), 2),
        (SkewedEvaluator(), 4),
    ],
)
def test_search_matches_python_on_isolated_positions(
    evaluator, chance_children
) -> None:
    config = MCTSConfig(sims=96, chance_children=chance_children)
    checked = 0
    for seed in range(4):
        for k, (py, rust) in enumerate(twin_positions(seed, 12)):
            ref = MCTS(evaluator, config, seed=seed * 31 + k, add_noise=False)
            tree = rs.Tree(
                cfg_dict(config),
                has_margin=getattr(evaluator, "has_margin", False),
                seed=seed * 31 + k,
                rng="python",
            )
            want = ref.search(py)
            got = tree.search(rust, evaluator)
            assert_same_result(got, want, f"seed {seed} pos {k}")
            assert tree.evals == ref.evals
            assert tree.nodes_created == ref.nodes_created
            checked += 1
    assert checked >= 40


def test_fast_rng_search_is_a_valid_search_too() -> None:
    py, rust = twin_positions(3, 1)[0]
    tree = rs.Tree({"sims": 64, "chance_children": 4}, seed=5, rng="fast")
    got = tree.search(rust, ScoreEvaluator())
    assert sum(got["visits"].values()) == 64
    assert got["policy"].sum() == pytest.approx(1.0, abs=1e-5)
    legal = set(py.legal_actions())
    assert set(a for a, n in got["visits"].items() if n) <= legal


# -------------------------------------------------------- whole games, reuse
@pytest.mark.parametrize("reuse", [False, True])
def test_search_matches_python_across_a_game_with_tree_reuse(reuse: bool) -> None:
    evaluator = MarginEvaluator()
    config = MCTSConfig(sims=80, tree_reuse=reuse, chance_children=3)
    py = AzulState.new_game(21)
    rust = rs.State.new_game(21, rng="python")
    ref = MCTS(evaluator, config, seed=9, add_noise=False)
    tree = rs.Tree(cfg_dict(config), has_margin=True, seed=9, rng="python")
    moves = 0
    reused = 0
    while not py.is_terminal and moves < 60:
        legal = py.legal_actions()
        if len(legal) == 1:
            action = legal[0]
        else:
            want = ref.search(py)
            got = tree.search(rust, evaluator)
            assert_same_result(got, want, f"move {moves}")
            assert tree.reused_visits == ref.reused_visits
            reused += tree.reused_visits > 0
            action = int(np.argmax(want.policy))
        py.apply(action)
        rust.apply(action)
        assert ref.advance(action) == tree.advance(action)
        moves += 1
    assert tree.evals == ref.evals
    assert tree.nodes_created == ref.nodes_created
    if reuse:
        assert reused > 10


# ------------------------------------------------------------- leaf protocol
def drive_python(mcts: MCTS, state, evaluator, max_leaves: int = 0):
    mcts.start_search(state)
    while not mcts.search_done():
        requests = mcts.leaf_requests(max_leaves)
        mcts.apply_leaves([evaluator(r.state, r.legal) for r in requests])
    return mcts.finish_search()


def drive_rust(tree: rs.Tree, state, evaluator, max_leaves: int = 0) -> dict:
    tree.start_search(state)
    while not tree.search_done():
        obs, legal = tree.leaf_requests(max_leaves)
        states = tree.leaf_states()
        assert obs.shape == (len(states), 182)
        outs = [evaluator(s, l) for s, l in zip(states, legal)]
        for s, row in zip(states, obs):
            assert np.array_equal(s.encode(), row)
        priors = [np.asarray(o[0], dtype=np.float32) for o in outs]
        values = [float(o[1]) for o in outs]
        margins = [float(o[2]) for o in outs] if len(outs[0]) == 3 else None
        tree.apply_leaves(priors, values, margins)
    return tree.finish_search()


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("batch", [1, 4])
def test_leaf_protocol_matches_python_pumped_search(reuse: bool, batch: int) -> None:
    evaluator = MarginEvaluator()
    config = MCTSConfig(
        sims=120,
        tree_reuse=reuse,
        search_batch=batch,
        search_batch_ramp=4,
        virtual_loss=1.0,
    )
    py = AzulState.new_game(5)
    rust = rs.State.new_game(5, rng="python")
    ref = MCTS(evaluator, config, seed=17, add_noise=False)
    tree = rs.Tree(cfg_dict(config), has_margin=True, seed=17, rng="python")
    for _ in range(8):
        want = drive_python(ref, py, evaluator)
        got = drive_rust(tree, rust, evaluator)
        assert_same_result(got, want)
        action = int(np.argmax(want.policy))
        py.apply(action)
        rust.apply(action)
        ref.advance(action)
        tree.advance(action)
    assert tree.evals == ref.evals
    assert tree.nodes_created == ref.nodes_created


def test_leaf_protocol_equals_blocking_search_in_rust() -> None:
    evaluator = ScoreEvaluator()
    config = {"sims": 100, "chance_children": 2}
    _, rust = twin_positions(8, 1)[0]
    a = rs.Tree(config, seed=3, rng="fast")
    b = rs.Tree(config, seed=3, rng="fast")
    want = a.search(rust, evaluator)
    got = drive_rust(b, rust, evaluator)
    assert got["visits"] == want["visits"]
    assert np.array_equal(got["policy"], want["policy"])
    assert got["value"] == want["value"]


def test_apply_logits_softmaxes_over_the_legal_actions() -> None:
    _, rust = twin_positions(2, 1)[0]
    tree = rs.Tree({"sims": 8}, seed=1)
    tree.start_search(rust)
    obs, legal = tree.leaf_requests()
    n = len(legal)
    logits = np.random.default_rng(0).standard_normal((n, 180)).astype(np.float32)
    values = np.zeros(n, dtype=np.float32)
    tree.apply_logits(logits, values)
    # The root's priors are now softmax(logits[legal]); a search of 8 more sims runs.
    while not tree.search_done():
        obs, legal = tree.leaf_requests()
        n = len(legal)
        tree.apply_logits(np.zeros((n, 180), np.float32), np.zeros(n, np.float32))
    res = tree.finish_search()
    assert res["sims"] == 8


def test_forced_root_returns_without_simulating() -> None:
    """One legal move: policy one-hot, sims 0, no evaluation consumed."""
    py = AzulState.new_game(1)
    # Build a position with a single legal action: one tile on the board and a
    # player whose pattern lines are all closed for that colour except the floor.
    for factory in py.factories:
        for c in range(5):
            py.lid[c] += factory[c]
            factory[c] = 0
    for c in range(5):
        py.lid[c] += py.center[c]
        py.center[c] = 0
    py.lid[0] -= 1
    py.factories[0][0] = 1
    for r in range(5):
        py.pl_color[0][r] = 1
        py.pl_count[0][r] = r + 1
    py.recount()
    assert py.legal_actions() == [5]  # factory 0, blue, floor
    rust = rs.State.from_dict(
        {
            k: getattr(py, k)
            for k in (
                "factories",
                "center",
                "marker_in_center",
                "bag",
                "lid",
                "walls",
                "pl_color",
                "pl_count",
                "floor",
                "floor_marker",
                "scores",
                "current_player",
                "first_player",
                "round_index",
                "is_terminal",
                "exhausted",
            )
        },
        rng="python",
    )
    ref = MCTS(UniformEvaluator(), MCTSConfig(sims=20), seed=1)
    tree = rs.Tree({"sims": 20}, seed=1, rng="python")
    want = ref.search(py)
    got = tree.search(rust, UniformEvaluator())
    assert_same_result(got, want)
    assert got["sims"] == 0 and got["visits"] == {5: 1}


# ---------------------------------------------------------- time budget, noise
def test_time_budget_caps_the_simulations() -> None:
    _, rust = twin_positions(4, 1)[0]
    tree = rs.Tree({"sims": 100_000}, seed=2)
    got = tree.search(rust, UniformEvaluator(), time_limit_s=0.05)
    assert 8 < got["sims"] < 100_000
    assert got["elapsed_s"] >= 0.05


def test_noise_changes_the_policy_and_only_at_a_real_root() -> None:
    _, rust = twin_positions(6, 1)[0]
    plain = rs.Tree({"sims": 40}, seed=2, add_noise=False).search(
        rust, UniformEvaluator()
    )
    noisy = rs.Tree({"sims": 40}, seed=2, add_noise=True).search(
        rust, UniformEvaluator()
    )
    assert not np.array_equal(plain["policy"], noisy["policy"])
    assert noisy["policy"].sum() == pytest.approx(1.0, abs=1e-5)
    again = rs.Tree({"sims": 40}, seed=2, add_noise=True).search(
        rust, UniformEvaluator()
    )
    assert np.array_equal(noisy["policy"], again["policy"])  # seeded


# --------------------------------------------------------- driver-side helpers
def test_numpy_sum_is_bit_exact() -> None:
    rng = np.random.default_rng(1)
    for n in list(range(1, 40)) + [
        63,
        64,
        65,
        127,
        128,
        129,
        180,
        255,
        256,
        257,
        300,
        1000,
    ]:
        a = rng.random(n) * rng.integers(1, 1000)
        assert rs.numpy_sum(list(a)) == float(a.sum()), n


def test_tree_rng_matches_python_random() -> None:
    tree = rs.Tree({}, seed=12345, rng="python")
    ref = random.Random(12345)
    assert tree.rng_random() == ref.random()
    assert [tree.rng_randrange(7) for _ in range(20)] == [
        ref.randrange(7) for _ in range(20)
    ]
    assert tree.rng_random() == ref.random()


def test_select_action_matches_python_sampling() -> None:
    policy = np.zeros(180, dtype=np.float32)
    rng = np.random.default_rng(3)
    idx = rng.choice(180, 30, replace=False)
    policy[idx] = rng.random(30).astype(np.float32)
    policy /= policy.sum()
    for temperature in (1.0, 0.7, 1.5):
        tree = rs.Tree({}, seed=99, rng="python")
        ref = random.Random(99)
        got = [tree.select_action(policy, temperature) for _ in range(300)]
        want = [select_action(policy, temperature, ref) for _ in range(300)]
        assert got == want, temperature
    tree = rs.Tree({}, seed=1)
    assert tree.select_action(policy, 0.0) == int(np.argmax(policy))
