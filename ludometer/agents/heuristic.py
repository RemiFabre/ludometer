"""Hand-tuned 1-ply agent — the strong non-learning anchor of the Elo ladder.

Same search as :class:`~ludometer.agents.greedy.GreedyAgent` (clone + apply every
legal action) but a much richer evaluation of the resulting position:

* banked score plus the points the player's complete pattern lines already earn;
* floor damage, weighted above face value (floor tiles also waste a turn);
* partial pattern lines valued by *where* they will land on the wall (adjacency
  potential) and discounted by how much of the needed color is still on the board
  this round (supply matching);
* progress toward the end bonuses (rows +2, columns +7, colors +10), quadratic so
  that finishing a nearly-complete line/column/color is worth more than starting
  a new one;
* the first-player marker, worth slightly more than the floor slot it costs;
* simple denial: a resulting position that leaves the opponent a big monochrome
  take is discounted.

Weights were tuned by self-play against :class:`GreedyAgent`; see
``ludometer/eval/calibrate.py`` for the measurement harness.
"""

from __future__ import annotations

import random
from typing import Any

from ludometer.agents.base import Agent
from ludometer.agents.features import (
    board_color_counts,
    tile_score,
    virtual_wall,
    wall_progress,
)
from ludometer.azul.engine import NUM_ROWS, AzulState

__all__ = ["DEFAULT_WEIGHTS", "HeuristicAgent"]

DEFAULT_WEIGHTS: dict[str, float] = {
    "win": 200.0,  # terminal states dominate everything else
    "op_score": 0.6,  # weight on the opponent's banked score
    "floor": 1.6,  # multiplier on the (negative) floor penalty
    "marker": 1.4,  # holding the first-player marker
    "partial": 0.85,  # partial pattern lines, scaled by wall value and supply
    "supply_floor": 0.35,  # value kept when the needed color is out of supply
    "row": 0.35,  # quadratic progress toward complete rows (+2 each)
    "col": 0.55,  # ... columns (+7 each)
    "color": 0.30,  # ... colors (+10 each)
    "deny": 0.45,  # discount for leaving a big monochrome take
    "waste": 0.30,  # penalty per tile parked in an unfinishable line
}


class HeuristicAgent(Agent):
    """1-ply search with a hand-tuned positional evaluation."""

    name = "heuristic"

    def __init__(self, seed: int = 0, **weights: float) -> None:
        self.rng = random.Random(seed)
        self.w = dict(DEFAULT_WEIGHTS)
        unknown = set(weights) - set(self.w)
        if unknown:
            raise ValueError(f"unknown weights: {sorted(unknown)}")
        self.w.update(weights)

    def seed(self, n: int) -> None:
        self.rng.seed(n)

    # ------------------------------------------------------------------ search
    def act(self, state: AzulState) -> int:
        actions = state.legal_actions()
        if not actions:
            raise ValueError("no legal actions (terminal state?)")
        me = state.current_player
        best: list[int] = []
        best_value = -1e18
        for action in actions:
            child = state.clone()
            child.apply(action)
            value = self.evaluate(child, me)
            if value > best_value:
                best_value = value
                best = [action]
            elif value == best_value:
                best.append(action)
        if len(best) == 1:
            return best[0]
        return best[self.rng.randrange(len(best))]

    # -------------------------------------------------------------- evaluation
    def evaluate(self, state: AzulState, me: int) -> float:
        """Score ``state`` from ``me``'s point of view (higher is better)."""
        w = self.w
        op = 1 - me
        if state.is_terminal:
            outcome = state.outcome() or 0.0
            sign = outcome if me == 0 else -outcome
            return w["win"] * sign + (state.scores[me] - state.scores[op])

        gain, wall = virtual_wall(state, me)
        value = float(state.scores[me]) + gain
        value -= w["op_score"] * state.scores[op]
        value += w["floor"] * state.floor_penalty(me)
        if state.floor_marker[me]:
            value += w["marker"]

        counts, best_mono = board_color_counts(state)
        plc = state.pl_color[me]
        pln = state.pl_count[me]
        supply_floor = w["supply_floor"]
        for r in range(NUM_ROWS):
            n = pln[r]
            cap = r + 1
            if not n or n == cap:
                continue
            color = plc[r]
            need = cap - n
            col = (color + r) % 5
            idx = r * 5 + col
            wall[idx] = 1
            here = tile_score(wall, r, col)
            wall[idx] = 0
            avail = counts[color]
            if avail >= need:
                feasible = 1.0
            else:
                feasible = avail / need
                # tiles stuck in a line nobody can finish are dead weight
                value -= w["waste"] * n * (1.0 - feasible)
            value += (
                w["partial"]
                * here
                * (n / cap)
                * (supply_floor + (1.0 - supply_floor) * feasible)
            )

        rows, cols, colors = wall_progress(wall)
        row_w, col_w, color_w = w["row"], w["col"], w["color"]
        for i in range(5):
            value += row_w * rows[i] * rows[i]
            value += col_w * cols[i] * cols[i]
            value += color_w * colors[i] * colors[i]

        if state.current_player == op:
            value -= w["deny"] * best_mono
        return value

    def config(self) -> dict[str, Any]:  # pragma: no cover - introspection aid
        return dict(self.w)
