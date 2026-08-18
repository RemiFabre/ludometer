"""Tests for the baseline agents (see docs/DESIGN.md, "Agents").

The fuzz tests are the important ones: an agent that returns an illegal action
poisons every match and every Elo number downstream.
"""

from __future__ import annotations

import random
import time

import pytest

from ludometer.agents import (
    AGENT_REGISTRY,
    Agent,
    GreedyAgent,
    HeuristicAgent,
    RandomAgent,
    make_agent,
    spec_name,
)
from ludometer.agents.features import (
    board_color_counts,
    immediate_value,
    tile_score,
    tiling_gain,
    virtual_wall,
    wall_progress,
)
from ludometer.azul.engine import AzulState
from ludometer.eval.arena import play_match

ALL_SPECS = ["random", "greedy", "heuristic"]


def play_out(agent: Agent, seed: int, opponent: Agent | None = None) -> AzulState:
    """Run a full game with `agent` on both seats (or vs `opponent`), checking legality."""
    state = AzulState.new_game(seed=seed)
    players = [agent, opponent or agent]
    moves = 0
    while not state.is_terminal:
        actor = players[state.current_player]
        action = actor.act(state)
        assert isinstance(action, int)
        assert state.is_legal(action), (
            f"{actor.name} returned illegal action {action} at move {moves}"
        )
        assert action in state.legal_actions()
        state.apply(action)
        moves += 1
        assert moves < 500
    assert state.tile_census() == [20] * 5
    return state


# ------------------------------------------------------------------- interface
def test_registry_covers_the_baselines():
    assert set(AGENT_REGISTRY) == {
        "random",
        "greedy",
        "heuristic",
        "uno:random",
        "uno:greedy",
        "uno:heuristic",
        "unoplus:random",
        "unoplus:greedy",
        "unoplus:heuristic",
    }
    for name, factory in AGENT_REGISTRY.items():
        agent = factory()
        assert isinstance(agent, Agent)
        assert agent.name == name


def test_abstract_base_needs_act():
    with pytest.raises(TypeError):
        Agent()  # type: ignore[abstract]

    class Silly(Agent):
        name = "silly"

        def act(self, state):
            return state.legal_actions()[0]

    agent = Silly()
    agent.seed(3)  # optional, must not explode
    state = AzulState.new_game(seed=0)
    assert state.is_legal(agent.act(state))


def test_make_agent_forms():
    assert make_agent("greedy").name == "greedy"
    assert make_agent(GreedyAgent).name == "greedy"
    instance = RandomAgent(seed=7)
    assert make_agent(instance) is instance
    tuned = make_agent(("heuristic", {"floor": 3.0}))
    assert isinstance(tuned, HeuristicAgent)
    assert tuned.w["floor"] == 3.0
    with pytest.raises(KeyError):
        make_agent("nope")
    with pytest.raises(TypeError):
        make_agent(42)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        HeuristicAgent(bogus_weight=1.0)


def test_spec_name_forms():
    assert spec_name("greedy") == "greedy"
    assert spec_name(("heuristic", {})) == "heuristic"
    assert spec_name(GreedyAgent) == "greedy"
    assert spec_name(RandomAgent(seed=1)) == "random"


# ------------------------------------------------------------------- fuzz play
@pytest.mark.parametrize("spec", ALL_SPECS)
def test_agent_only_plays_legal_actions(spec):
    agent = make_agent(spec)
    for seed in range(4):
        agent.seed(seed)
        state = play_out(agent, seed)
        assert state.is_terminal
        assert state.outcome() in (-1.0, 0.0, 1.0)


@pytest.mark.parametrize("spec", ALL_SPECS)
def test_agent_survives_fuzzed_positions(spec):
    """Drop each agent into positions reached by random play, mid-round included."""
    agent = make_agent(spec)
    agent.seed(1234)
    rng = random.Random(99)
    for seed in range(6):
        state = AzulState.new_game(seed=seed)
        while not state.is_terminal:
            action = agent.act(state)
            assert state.is_legal(action)
            # take a random action instead, to wander into odd positions
            state.apply(rng.choice(state.legal_actions()))


@pytest.mark.parametrize("spec", ALL_SPECS)
def test_agent_does_not_mutate_the_state(spec):
    agent = make_agent(spec)
    agent.seed(5)
    state = AzulState.new_game(seed=11)
    for _ in range(12):
        before = state.to_json()
        agent.act(state)
        assert state.to_json() == before
        state.apply(state.legal_actions()[0])
        if state.is_terminal:
            break


@pytest.mark.parametrize("spec", ALL_SPECS)
def test_agent_is_deterministic_given_a_seed(spec):
    def moves(seed: int) -> list[int]:
        agent = make_agent(spec)
        agent.seed(seed)
        state = AzulState.new_game(seed=3)
        out = []
        while not state.is_terminal:
            action = agent.act(state)
            out.append(action)
            state.apply(action)
        return out

    assert moves(42) == moves(42)


def test_random_agent_reseeding_changes_play():
    a, b = RandomAgent(seed=1), RandomAgent(seed=2)
    state = AzulState.new_game(seed=0)
    picks_a = [a.act(state) for _ in range(20)]
    picks_b = [b.act(state) for _ in range(20)]
    assert picks_a != picks_b
    a.seed(2)
    assert [a.act(state) for _ in range(20)] == picks_b


def test_agent_refuses_terminal_state():
    state = AzulState.new_game(seed=0)
    while not state.is_terminal:
        state.apply(state.legal_actions()[0])
    for spec in ALL_SPECS:
        with pytest.raises(ValueError):
            make_agent(spec).act(state)


# -------------------------------------------------------------------- features
def test_tile_score_runs():
    wall = [0] * 25
    wall[0] = 1
    assert tile_score(wall, 0, 0) == 1  # lone tile
    wall[1] = 1
    assert tile_score(wall, 0, 1) == 2  # horizontal run of 2
    wall[5] = 1
    assert tile_score(wall, 1, 0) == 2  # vertical run of 2
    wall[6] = 1
    assert tile_score(wall, 1, 1) == 4  # 2 horizontal + 2 vertical


def test_virtual_wall_matches_the_engine():
    """The features' hypothetical tiling must agree with a real round end."""
    checked = 0
    for seed in range(6):
        state = AzulState.new_game(seed=seed)
        rng = random.Random(seed)
        while not state.is_terminal:
            # the mover cannot touch the *other* player's pending tiling, so the
            # features computed before the move must predict their new score
            other = 1 - state.current_player
            before = state.scores[other]
            pending = tiling_gain(state, other)
            penalty = state.floor_penalty(other)
            round_before = state.round_index
            state.apply(rng.choice(state.legal_actions()))
            if state.round_index != round_before and not state.is_terminal:
                assert state.scores[other] == max(0, before + pending + penalty)
                checked += 1
    assert checked > 5


def test_immediate_value_prefers_completed_lines():
    state = AzulState.new_game(seed=0)
    base = immediate_value(state, 0)
    assert base == 0.0
    # first row takes one tile: filling it is worth a point at the next tiling
    child = state.clone()
    color = next(c for c in range(5) if child.factories[0][c])
    child.factories[0] = [0] * 5
    child.factories[0][color] = 1
    child.recount()
    child.apply(color * 6 + 0)  # factory 0, that color, pattern row 0
    assert immediate_value(child, 0) == 1.0
    # the same tile dumped on the floor costs a point instead
    other = state.clone()
    other.factories[0] = [0] * 5
    other.factories[0][color] = 1
    other.recount()
    other.apply(color * 6 + 5)
    assert immediate_value(other, 0) == -1.0


def test_board_color_counts_and_wall_progress():
    state = AzulState.new_game(seed=4)
    counts, best = board_color_counts(state)
    assert sum(counts) == state.tiles_left
    assert 1 <= best <= 4
    wall = [0] * 25
    wall[0] = wall[1] = wall[5] = 1
    rows, cols, colors = wall_progress(wall)
    assert rows == [2, 1, 0, 0, 0]
    assert cols == [2, 1, 0, 0, 0]
    assert sum(colors) == 3
    gain, copied = virtual_wall(state, 0)
    assert gain == 0 and copied == [0] * 25


# ------------------------------------------------------------ strength & speed
def test_greedy_crushes_random():
    match = play_match("greedy", "random", n_games=30, base_seed=5)
    assert match.decisive_win_rate >= 0.8, match.as_dict()
    assert match.mean_score_diff > 20.0


def test_heuristic_beats_greedy():
    match = play_match("heuristic", "greedy", n_games=30, base_seed=6)
    assert match.decisive_win_rate >= 0.6, match.as_dict()
    assert match.mean_score_diff > 0.0


@pytest.mark.parametrize("spec", ALL_SPECS)
def test_move_time_budget(spec):
    """Loose upper bound: the heuristic must stay in the sub-millisecond range."""
    agent = make_agent(spec)
    agent.seed(0)
    state = AzulState.new_game(seed=17)
    moves = 0
    t0 = time.perf_counter()
    while not state.is_terminal:
        state.apply(agent.act(state))
        moves += 1
    ms_per_move = 1000.0 * (time.perf_counter() - t0) / moves
    assert ms_per_move < 5.0, f"{spec}: {ms_per_move:.2f} ms/move"
