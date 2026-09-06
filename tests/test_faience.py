"""Round-trip tests for ludometer.human.faience (no network, no real data).

The converter's contract: a faience-game/1 record replays in the Python engine
move for move, deal for deal, and lands on the recorded final scores — or it is
rejected. The fixture here is the engine itself: play a game with a seeded
policy, serialize it in the record shape the site writes, and the converter
must reproduce it exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from ludometer.azul.engine import AzulState
from ludometer.human.convert import ConversionError
from ludometer.human.faience import convert_record, human_result


def synthetic_record(seed: int, human_seat: int = 0) -> dict:
    """Play a full engine game and write it out as a faience-game/1 record."""
    rng = np.random.default_rng(seed)
    state = AzulState.new_game(seed=seed)
    deals = [
        {
            "round": 0,
            "factories": [list(f) for f in state.factories],
            "bag": list(state.bag_counts()),
            "lid": list(state.lid),
        }
    ]
    moves = []
    ply = 0
    while not state.is_terminal:
        ply += 1
        action = int(rng.choice(state.legal_actions()))
        moves.append({"ply": ply, "player": int(state.current_player), "action": action})
        round_before = state.round_index
        state.apply(action)
        if state.round_index > round_before and not state.is_terminal:
            deals.append(
                {
                    "round": int(state.round_index),
                    "factories": [list(f) for f in state.factories],
                    "bag": list(state.bag_counts()),
                    "lid": list(state.lid),
                }
            )
    return {
        "format": "faience-game/1",
        "seed": seed,
        "human_seat": human_seat,
        "human_first": human_seat == 0,
        "moves": moves,
        "deals": deals,
        "final": {
            "finished": True,
            "scores": [int(state.scores[0]), int(state.scores[1])],
            "outcome": "human",
            "rounds": int(state.round_index) + 1,
            "exhausted": False,
        },
    }


def test_a_synthetic_game_round_trips() -> None:
    record = synthetic_record(seed=1234)
    game = convert_record(record)
    assert len(game) == len(record["moves"])
    assert game.scores == tuple(record["final"]["scores"])
    assert list(game.actions) == [m["action"] for m in record["moves"]]
    assert list(game.movers) == [m["player"] for m in record["moves"]]
    assert human_result(record, game) in ("win", "loss", "draw")


def test_the_result_is_read_from_the_human_seat() -> None:
    record = synthetic_record(seed=1234, human_seat=0)
    game = convert_record(record)
    flipped = dict(record, human_seat=1)
    a, b = human_result(record, game), human_result(flipped, game)
    if game.scores[0] != game.scores[1]:
        assert {a, b} == {"win", "loss"}
    else:
        assert a == b == "draw"


def test_an_unfinished_game_is_rejected() -> None:
    record = synthetic_record(seed=99)
    record["final"]["finished"] = False
    with pytest.raises(ConversionError, match="not finished"):
        convert_record(record)


def test_a_wrong_final_score_is_rejected() -> None:
    record = synthetic_record(seed=99)
    record["final"]["scores"][0] += 1
    with pytest.raises(ConversionError, match="engine scored"):
        convert_record(record)


def test_a_truncated_move_list_is_rejected() -> None:
    record = synthetic_record(seed=7)
    record["moves"] = record["moves"][:-3]
    with pytest.raises(ConversionError, match="ran out"):
        convert_record(record)


def test_an_illegal_move_is_rejected() -> None:
    record = synthetic_record(seed=7)
    state = AzulState.new_game(seed=7)
    legal = set(state.legal_actions())
    bad = next(a for a in range(180) if a not in legal)
    record["moves"][0]["action"] = bad
    with pytest.raises(ConversionError, match="illegal"):
        convert_record(record)
