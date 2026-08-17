"""Replay a parsed BGA game inside our own engine, or reject it.

The rule for this stage is: **a game becomes training data only if our engine
reproduces it exactly.** Every human turn must be a legal action in the state our
engine believes it is in, the tiles must be conserved after every scripted deal,
the game must end when BGA says it ended, and the final scores must match the ones
BGA reported. Anything else raises :class:`ConversionError` and the game is
dropped. That is cheap (we have thousands of candidate games and need no
particular one) and it is the only defence against a subtly wrong mapping — an
off-by-one factory index or a permuted colour would otherwise produce a dataset
that trains a net on plausible-looking nonsense.

Scripting the chance events
---------------------------
Our engine owns its own bag and draws refills from it (that is what makes self-play
reproducible). A human game had a different bag, so :func:`apply_deal` overwrites
``factories`` with the deal the log reports and re-derives ``bag``/``lid`` around
it, then calls ``recount()``. This touches only public attributes — the engine's
source stays untouched — and every deal is followed by a
``tile_census() == [20] * 5`` assertion, so a mis-parsed deal cannot survive.

The seat convention: ``ReplayGame.player_ids[0]`` is engine player 0. Who moves
first is taken from the log (the first pick), not assumed.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ludometer.azul.engine import (
    NUM_COLORS,
    NUM_FACTORIES,
    TILES_PER_COLOR,
    AzulState,
    wall_col,
)
from ludometer.human.parse import (
    DEFAULT_SCHEMA,
    Deal,
    LogSchema,
    ParseError,
    ReplayGame,
    observed_color_ids,
    parse_log,
    with_color_map,
)

__all__ = [
    "SCORE_SCALE",
    "ConversionError",
    "HumanGame",
    "apply_deal",
    "check_wall_placements",
    "convert_game",
    "solve_color_map",
    "solve_color_map_over",
]

#: Same constant self-play uses for the margin target, so the two datasets are
#: interchangeable (``ludometer.train.selfplay.SCORE_SCALE``). Spelled out rather
#: than imported: this package must not pull torch in through that module.
SCORE_SCALE = 20.0


class ConversionError(ValueError):
    """The engine could not reproduce this game — it must be discarded."""


@dataclass
class HumanGame:
    """One validated human game as training rows.

    ``states[i]`` is the encoded position **before** move ``i`` from the point of
    view of the player to move, and ``actions[i]`` is the action that player
    actually chose: the policy target is a one-hot on it. ``movers[i]`` is that
    player's seat, which is what turns a game-level outcome into a per-row label.
    """

    table_id: int
    states: np.ndarray  # (T, 182) float32
    actions: np.ndarray  # (T,) int64
    movers: np.ndarray  # (T,) int64, 0 or 1
    aux: np.ndarray  # (T, 30) uint8, final-wall bits in the mover's frame
    scores: tuple[int, int]
    outcome: float  # +1 seat 0 won, -1 seat 1 won, 0 draw
    rounds: int

    def __len__(self) -> int:
        return len(self.actions)

    def values(self) -> np.ndarray:
        """Outcome per row, in the mover's frame (the value head's convention)."""
        signs = np.where(self.movers == 0, 1.0, -1.0)
        return (self.outcome * signs).astype(np.float32)

    def margins(self) -> np.ndarray:
        """``tanh(score diff / 20)`` per row, in the mover's frame."""
        diff = math.tanh((self.scores[0] - self.scores[1]) / SCORE_SCALE)
        signs = np.where(self.movers == 0, 1.0, -1.0)
        return (diff * signs).astype(np.float32)


def _bag_counts_from(state: AzulState) -> list[int]:
    return state.bag_counts()


def apply_deal(state: AzulState, deal: Deal) -> None:
    """Replace the engine's random refill with the deal the log reports.

    The engine has just dealt ``state.factories`` out of its own bag. We want
    ``deal.factories`` instead, so:

    1. the tiles that exist "off board" are ``bag + lid + the engine's deal``;
    2. the observed deal is taken out of that pool and written to the factories;
    3. whatever is left goes back. It stays split between bag and lid when the
       engine's own draw did not exhaust the bag (so the public bag/lid features
       keep their meaning); if the observed deal needs more of a colour than the
       bag can supply, the real game must have reshuffled the lid in, and we
       merge the two — the same thing ``_refill`` does.

    Raises :class:`ConversionError` when the deal is impossible (more tiles of a
    colour than exist), which is the check that catches a mis-parsed deal.
    """
    target = [list(f) for f in deal.factories]
    if len(target) != NUM_FACTORIES:
        raise ConversionError(
            f"deal has {len(target)} factories, engine has {NUM_FACTORIES}"
        )
    dealt = [0] * NUM_COLORS
    for factory in target:
        if len(factory) != NUM_COLORS:
            raise ConversionError(f"deal factory has {len(factory)} colors")
        for color, count in enumerate(factory):
            if count < 0:
                raise ConversionError("deal has a negative tile count")
            dealt[color] += count

    bag = _bag_counts_from(state)
    lid = list(state.lid)
    engine_deal = [sum(f[c] for f in state.factories) for c in range(NUM_COLORS)]
    pool = [bag[c] + lid[c] + engine_deal[c] for c in range(NUM_COLORS)]
    remaining = [pool[c] - dealt[c] for c in range(NUM_COLORS)]
    if any(r < 0 for r in remaining):
        raise ConversionError(
            f"round {deal.round_index}: deal wants {dealt} but only {pool} are off-board"
        )

    new_bag = [remaining[c] - lid[c] for c in range(NUM_COLORS)]
    if any(n < 0 for n in new_bag):  # the real game reshuffled the lid into the bag
        new_bag, lid = remaining, [0] * NUM_COLORS

    state.factories = target
    state.bag = [c for c in range(NUM_COLORS) for _ in range(new_bag[c])]
    state.lid = lid
    state.recount()
    census = state.tile_census()
    if census != [TILES_PER_COLOR] * NUM_COLORS:
        raise ConversionError(
            f"round {deal.round_index}: tile census {census} after deal"
        )


def check_wall_placements(game: ReplayGame) -> str:
    """``""`` if every wall tile sits where the **fixed** wall puts it, else why not.

    On Azul's standard wall the column of colour ``c`` in row ``r`` is
    ``(c + r) % 5`` (:func:`ludometer.azul.engine.wall_col`), and the log reports
    the column of every tile the wall-tiling step places. So this single comparison
    does three jobs:

    * it **identifies grey "variable wall" games** — the variant Remi wants
      excluded. There, placement inside the row is the player's choice, so the
      columns will not follow the formula;
    * it **verifies the colour map** independently of the score;
    * it catches a wrong ``lines_one_based`` guess, because shifting every row by
      one shifts every expected column by one.

    A handful of games failing this while the rest pass means those games are the
    variant (correct rejection). *Everything* failing means the schema is wrong —
    fix ``LogSchema``, do not relax the check.
    """
    for placement in game.wall_placements:
        expected = wall_col(placement.color, placement.row)
        if placement.column != expected:
            return (
                f"wall placement colour {placement.color} row {placement.row} is in "
                f"column {placement.column}, the fixed wall puts it in {expected} "
                "(variable-wall variant, or a wrong colour/line mapping)"
            )
    return ""


def convert_game(
    game: ReplayGame,
    check_scores: bool = True,
    require_terminal: bool = True,
    check_wall: bool = True,
) -> HumanGame:
    """Replay ``game`` in the engine and return its training rows.

    ``check_scores`` compares our engine's final scores with the ones BGA
    reported; leave it on. It is the single most valuable validation we have,
    because it is sensitive to exactly the mistakes a mapping bug makes: a wrong
    wall column changes adjacency bonuses, so a permuted colour map or a swapped
    seat shows up as a score mismatch rather than as an illegal move.
    """
    if len(game.player_ids) != 2:
        raise ConversionError(
            f"table {game.table_id}: {len(game.player_ids)} players, need 2"
        )
    if not game.picks:
        raise ConversionError(f"table {game.table_id}: no picks in the log")
    if not game.deals:
        raise ConversionError(f"table {game.table_id}: no factory deals in the log")
    if check_wall:
        reason = check_wall_placements(game)
        if reason:
            raise ConversionError(f"table {game.table_id}: {reason}")

    state = AzulState.new_game(seed=0)
    apply_deal(state, game.deals[0])
    first_seat = game.seat_of(game.picks[0].player_id)
    state.current_player = first_seat
    state.first_player = first_seat

    states: list[np.ndarray] = []
    actions: list[int] = []
    movers: list[int] = []
    deal_index = 1

    for pick in game.picks:
        if state.is_terminal:
            raise ConversionError(
                f"table {game.table_id}: log continues past the engine's game end "
                f"(move {pick.move_id})"
            )
        seat = game.seat_of(pick.player_id)
        if seat != state.current_player:
            raise ConversionError(
                f"table {game.table_id} move {pick.move_id}: log says seat {seat} moves, "
                f"engine says seat {state.current_player}"
            )
        action = pick.action_id()
        if not state.is_legal(action):
            raise ConversionError(
                f"table {game.table_id} move {pick.move_id}: illegal action {action} "
                f"(source {pick.source}, color {pick.color}, dest {pick.dest})"
            )
        states.append(state.encode())
        actions.append(action)
        movers.append(seat)
        round_before = state.round_index
        state.apply(action)
        if state.round_index > round_before and not state.is_terminal:
            if deal_index >= len(game.deals):
                raise ConversionError(
                    f"table {game.table_id}: engine started round {state.round_index} "
                    f"but the log only holds {len(game.deals)} deals"
                )
            apply_deal(state, game.deals[deal_index])
            deal_index += 1

    if require_terminal and not state.is_terminal:
        raise ConversionError(
            f"table {game.table_id}: log ran out after {len(actions)} moves, "
            "engine game is not over"
        )
    census = state.tile_census()
    if census != [TILES_PER_COLOR] * NUM_COLORS:
        raise ConversionError(f"table {game.table_id}: final tile census {census}")

    engine_scores = (int(state.scores[0]), int(state.scores[1]))
    reported = game.scores_by_seat()
    if check_scores and reported is not None and reported != engine_scores:
        raise ConversionError(
            f"table {game.table_id}: engine scored {engine_scores}, BGA reported {reported}"
        )

    walls = [state.wall_summary(0), state.wall_summary(1)]
    aux_by_seat = [np.array(walls[p] + walls[1 - p], dtype=np.uint8) for p in (0, 1)]
    return HumanGame(
        table_id=game.table_id,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        movers=np.asarray(movers, dtype=np.int64),
        aux=np.stack([aux_by_seat[m] for m in movers]),
        scores=engine_scores,
        outcome=float(state.outcome() or 0.0),
        rounds=int(state.round_index) + 1,
    )


def solve_color_map(
    payload: dict[str, Any],
    table_id: int,
    player_ids: tuple[int, ...],
    schema: LogSchema,
    infos: dict[str, Any] | None = None,
) -> list[dict[int, int]]:
    """Brute-force the BGA-tile-id -> engine-colour mapping against one real game.

    Run this once on a handful of real logs to pin ``LogSchema.color_map``; the
    answer is then a constant. It works because the wall's column for colour ``c``
    in row ``r`` is ``(c + r) % 5``: relabelling colours rotates the wall's
    columns, which breaks horizontal adjacency at the wrap-around and therefore
    changes the final score. Only mappings that both replay legally *and* match
    BGA's reported score survive, and across two or three games only the true one
    does.

    Returns every surviving mapping (usually one). An empty list means the schema
    is wrong somewhere else — factory indexing or the pick argument keys.
    """
    survivors: list[dict[int, int]] = []
    raw_ids = observed_color_ids(payload, schema)
    if len(raw_ids) != NUM_COLORS:
        raise ConversionError(
            f"table {table_id}: log mentions {len(raw_ids)} tile ids ({raw_ids}), need "
            f"{NUM_COLORS} before the mapping can be solved"
        )
    for permutation in itertools.permutations(range(NUM_COLORS)):
        candidate = {raw: permutation[i] for i, raw in enumerate(raw_ids)}
        try:
            game = parse_log(
                payload, table_id, player_ids, with_color_map(schema, candidate), infos
            )
            convert_game(game, check_scores=True)
        except (ConversionError, ParseError):
            continue
        survivors.append(candidate)
    return survivors


def solve_color_map_over(
    tables: Iterable[
        tuple[dict[str, Any], int, tuple[int, ...], dict[str, Any] | None]
    ],
    schema: LogSchema = DEFAULT_SCHEMA,
) -> list[dict[int, int]]:
    """Intersect :func:`solve_color_map` over several games.

    One game is not enough: a low-scoring game has few wall adjacencies, so
    several colour permutations reproduce its score by luck (measured on synthetic
    random-play games: 6 to 24 of the 120 survive a single game). The true mapping
    survives *every* game, so intersecting a handful collapses the set — use five
    or so real, high-scoring games and expect exactly one survivor.
    """
    surviving: list[dict[int, int]] | None = None
    for payload, table_id, player_ids, infos in tables:
        found = solve_color_map(payload, table_id, player_ids, schema, infos)
        keys = {tuple(sorted(m.items())) for m in found}
        if surviving is None:
            surviving = found
        else:
            surviving = [m for m in surviving if tuple(sorted(m.items())) in keys]
        if not surviving:
            break
    return surviving or []
