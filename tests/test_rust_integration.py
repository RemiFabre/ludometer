"""The Rust engine through the Python seams: registry, agent, self-play pool.

* ``LUDOMETER_ENGINE=rust`` makes ``get_game("azul")`` hand out the wrapper, and
  the wrapper behaves like the Python engine where the arena and agents look;
* ``MCTSAgent(engine="rust")`` and the ``?engine=rust`` spec option search on the
  Rust tree with the ordinary ``NetEvaluator`` and play legal, decisive moves;
* ``make_selfplay(kind="rust")`` returns engines with the pool interface that
  stream valid ``GameRecord``s, in-process and across worker processes.

Skipped when ``ludometer_rs`` is not built (`rust/README.md` says how).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ludometer.azul.engine import AzulState
from ludometer.eval.arena import play_match
from ludometer.train.mcts import MCTSConfig
from ludometer.train.mcts_agent import MCTSAgent
from ludometer.train.net import NetConfig, PolicyValueNet, make_net
from ludometer.train.net2 import StructuredConfig
from ludometer.train.selfplay import GameRecord, SelfPlayConfig, make_selfplay

rs = pytest.importorskip("ludometer_rs")
engine_rs = pytest.importorskip("ludometer.azul.engine_rs")

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


# ----------------------------------------------------------------- the wrapper
def test_wrapper_matches_python_engine_on_a_random_game() -> None:
    py = AzulState.new_game(3)
    rust = engine_rs.AzulState.new_game(3, rng="python")
    rng = np.random.default_rng(0)
    while not py.is_terminal:
        assert rust.legal_actions() == py.legal_actions()
        assert np.array_equal(rust.encode(), py.encode())
        assert rust.scores == py.scores and rust.current_player == py.current_player
        assert rust.render_text() == py.render_text()
        assert rust.to_json() == py.to_json()
        legal = py.legal_actions()
        a = legal[rng.integers(len(legal))]
        py.apply(a)
        rust.apply(a)
    assert rust.is_terminal and rust.outcome() == py.outcome()
    assert rust.clone().to_dict() == rust.to_dict()


def test_wrapper_is_a_search_root_for_both_trees() -> None:
    from ludometer.train.mcts import MCTS, UniformEvaluator
    from ludometer.train.mcts_rs import MCTS as RustMCTS

    state = engine_rs.AzulState.new_game(4)
    a = MCTS(UniformEvaluator(), MCTSConfig(sims=32), seed=1).search(state)
    b = RustMCTS(UniformEvaluator(), MCTSConfig(sims=32), seed=1).search(state)
    assert a.visits == b.visits
    assert np.array_equal(a.policy, b.policy)


def test_env_var_selects_the_rust_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    from ludometer.games import get_game

    monkeypatch.delenv("LUDOMETER_ENGINE", raising=False)
    assert get_game("azul").state_cls is AzulState
    monkeypatch.setenv("LUDOMETER_ENGINE", "rust")
    spec = get_game("azul")
    assert spec.state_cls is engine_rs.AzulState
    assert spec.encoded_size == 182 and spec.action_space == 180
    state = spec.new_game(7)
    assert isinstance(state, engine_rs.AzulState)
    assert get_game("uno").state_cls is not engine_rs.AzulState


def test_baseline_agents_play_a_match_on_the_rust_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LUDOMETER_ENGINE", "rust")
    result = play_match("heuristic", "greedy", n_games=2, base_seed=1)
    assert result.wins + result.draws + result.losses == 2 == result.n_games


# --------------------------------------------------------------------- the agent
def test_mcts_agent_on_the_rust_tree_plays_legal_decisive_moves() -> None:
    torch.manual_seed(0)
    net = PolicyValueNet(NetConfig(hidden=32, blocks=1, value_hidden=16))
    py_agent = MCTSAgent(net, sims=24, seed=3, engine="python")
    rs_agent = MCTSAgent(net, sims=24, seed=3, engine="rust")
    assert rs_agent.engine == "rust"
    state = AzulState.new_game(11)
    for _ in range(6):
        a = py_agent.act(state)
        b = rs_agent.act(state)
        assert a in state.legal_actions() and b in state.legal_actions()
        assert a == b, "same net, one leaf per pass, no noise: identical picks"
        assert rs_agent.last_search["sims"] == py_agent.last_search["sims"] == 24
        state.apply(a)


def test_mcts_agent_time_budget_on_the_rust_tree() -> None:
    torch.manual_seed(0)
    net = PolicyValueNet(NetConfig(hidden=32, blocks=1, value_hidden=16))
    agent = MCTSAgent(net, sims=10, seed=3, engine="rust", time_limit_s=0.05)
    assert agent.mcts.config.sims >= 20_000  # the cap was raised for the clock
    state = AzulState.new_game(2)
    agent.act(state)
    assert 8 < agent.last_search["sims"] < 20_000


def test_spec_option_engine_rust(tmp_path) -> None:
    from ludometer.agents.registry import load_agent
    from ludometer.train.net import save_checkpoint

    torch.manual_seed(0)
    net = PolicyValueNet(NetConfig(hidden=32, blocks=1, value_hidden=16))
    path = tmp_path / "ckpt-000001.pt"
    save_checkpoint(path, net, {"games": 1})
    agent = load_agent(f"mcts:{path}?sims=8&engine=rust")
    assert agent.engine == "rust"
    assert agent.spec_info["engine"] == "rust"
    assert load_agent(f"mcts:{path}?sims=8").engine == "python"
    with pytest.raises(ValueError):
        load_agent(f"mcts:{path}?engine=cobol")


# ------------------------------------------------------------------- self-play
def selfplay_config() -> SelfPlayConfig:
    return SelfPlayConfig(
        mcts=MCTSConfig(sims=16, tree_reuse=True, chance_children=2),
        temp_moves=4,
        max_moves=120,
        value_score_weight=0.0,
    )


def check_record(r: GameRecord) -> None:
    t = len(r)
    assert r.states.shape == (t, 182) and r.states.dtype == np.float32
    assert r.policies.shape == (t, 180)
    assert r.values.shape == (t,) and r.margins.shape == (t,)
    assert r.aux.shape == (t, 30) and r.aux.dtype == np.uint8
    assert r.policy_mask.shape == (t,)
    assert r.search_values is not None and r.search_values.shape == (t,)
    assert r.moves == t and r.decisions <= t and r.evals > 0
    assert r.outcome in (-1.0, 0.0, 1.0)


def test_make_selfplay_rust_in_process_streams_records() -> None:
    engine = make_selfplay(
        TINY, selfplay_config(), workers=1, kind="rust", games=4, device="cpu"
    )
    net = make_net(TINY)
    engine.start(net.cpu_state_dict())
    seen: list[GameRecord] = []
    ticks: list[tuple[int, int]] = []
    records = engine.play(
        5, 300, progress=lambda d, t: ticks.append((d, t)), on_record=seen.append
    )
    assert len(records) == 5 == len(seen)
    assert sorted(r.seed for r in records) == list(range(300, 305))
    for r in records:
        check_record(r)
    assert ticks and ticks[-1] == (5, 5)
    assert engine.positions > 0 and engine.batches > 0
    engine.close()


def test_make_selfplay_rust_should_stop_abandons_games() -> None:
    engine = make_selfplay(
        TINY, selfplay_config(), workers=1, kind="rust", games=2, device="cpu"
    )
    engine.start(make_net(TINY).cpu_state_dict())
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    engine.tick = 0.0
    records = engine.play(50, 0, should_stop=stop)
    assert len(records) < 50


def test_make_selfplay_rust_pool_streams_records_and_progress() -> None:
    pool = make_selfplay(
        TINY, selfplay_config(), workers=2, kind="rust", games=2, device="cpu"
    )
    pool.start(make_net(TINY).cpu_state_dict())
    try:
        records = pool.play(4, 900, progress=lambda d, t: None)
        assert len(records) == 4
        assert sorted(r.seed for r in records) == [900, 901, 902, 903]
        for r in records:
            check_record(r)
        assert set(pool.worker_positions) <= {0, 1}
    finally:
        pool.close()


def test_make_selfplay_rust_rejects_other_games() -> None:
    with pytest.raises(ValueError):
        make_selfplay(
            TINY,
            SelfPlayConfig(game="uno"),
            workers=1,
            kind="rust",
            games=2,
            device="cpu",
        )
