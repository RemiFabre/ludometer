"""Plain-language descriptions of what the engine just did.

:meth:`AzulState.apply` mutates in place and resolves the round boundary itself,
so nothing in the engine reports *which* tiles moved to the wall or how many
points each one earned. This module reconstructs that story from a snapshot taken
**before** the move, using exactly the engine's rules (``docs/DESIGN.md``):

* :func:`describe_action` — "took 3 red from factory 2 -> row 4 (1 to floor)";
* :func:`round_report` — per-player wall tiling, points per tile, floor penalty;
* :func:`final_report` — end bonuses broken down, and the winner.

Everything returned is JSON-serialisable and uses 0-based indices, plus
``*_label`` strings that are already 1-based for humans.
"""

from __future__ import annotations

from typing import Any

from ludometer.azul.engine import (
    CENTER,
    COL_BONUS,
    COLOR_BONUS,
    COLOR_NAMES,
    CUM_PENALTY,
    FLOOR,
    FLOOR_PENALTIES,
    FLOOR_SLOTS,
    NUM_COLORS,
    NUM_ROWS,
    ROW_BONUS,
    WALL_IDX,
    AzulState,
    decode_action,
)

__all__ = [
    "after_placement",
    "describe_action",
    "final_report",
    "round_report",
    "source_label",
]


def source_label(source: int) -> str:
    """``0..4`` -> "factory 1".."factory 5", ``5`` -> "the center"."""
    return "the center" if source == CENTER else f"factory {source + 1}"


def dest_label(dest: int) -> str:
    return "the floor line" if dest == FLOOR else f"row {dest + 1}"


def describe_action(state: AzulState, action_id: int) -> dict[str, Any]:
    """Describe ``action_id`` **as it would play out** in ``state`` (no mutation)."""
    source, color, dest = decode_action(action_id)
    player = state.current_player
    pool = state.center if source == CENTER else state.factories[source]
    count = pool[color]
    took_marker = source == CENTER and state.marker_in_center

    if dest == FLOOR:
        placed, overflow = 0, count
    else:
        room = dest + 1 - state.pl_count[player][dest]
        placed = min(count, room)
        overflow = count - placed

    # how much of the overflow actually fits on the floor (rest goes to the lid)
    occupied = state.floor_occupied(player) + (1 if took_marker else 0)
    to_floor = max(0, min(overflow, FLOOR_SLOTS - occupied))
    to_lid = overflow - to_floor

    text = (
        f"took {count} {COLOR_NAMES[color]} from {source_label(source)}"
        f" → {dest_label(dest)}"
    )
    if dest != FLOOR and overflow:
        text += f" ({overflow} to the floor)"
    if took_marker:
        text += " + first-player marker"

    return {
        "action_id": action_id,
        "player": player,
        "source": source,
        "color": color,
        "dest": dest,
        "count": count,
        "placed": placed,
        "overflow": overflow,
        "to_floor": to_floor,
        "to_lid": to_lid,
        "took_marker": took_marker,
        "color_name": COLOR_NAMES[color],
        "source_label": source_label(source),
        "dest_label": dest_label(dest),
        "text": text,
    }


def _run_lengths(wall: list[int], row: int, col: int) -> tuple[int, int]:
    base = row * 5
    h = 1
    i = col - 1
    while i >= 0 and wall[base + i]:
        h += 1
        i -= 1
    i = col + 1
    while i < 5 and wall[base + i]:
        h += 1
        i += 1
    v = 1
    i = row - 1
    while i >= 0 and wall[i * 5 + col]:
        v += 1
        i -= 1
    i = row + 1
    while i < 5 and wall[i * 5 + col]:
        v += 1
        i += 1
    return h, v


def _tile_player(before: AzulState, player: int) -> dict[str, Any]:
    """Replay the engine's wall-tiling for one player on a snapshot."""
    wall = before.walls[player][:]
    colors = before.pl_color[player]
    counts = before.pl_count[player]
    tiles: list[dict[str, Any]] = []
    gain = 0
    for row in range(NUM_ROWS):
        if counts[row] != row + 1:
            continue
        color = colors[row]
        idx = WALL_IDX[color * 5 + row]
        wall[idx] = 1
        col = idx - row * 5
        h, v = _run_lengths(wall, row, col)
        points = (h if h > 1 else 0) + (v if v > 1 else 0) if (h > 1 or v > 1) else 1
        gain += points
        tiles.append(
            {
                "row": row,
                "col": col,
                "color": color,
                "color_name": COLOR_NAMES[color],
                "points": points,
                "h_run": h,
                "v_run": v,
                "discarded": row,  # leftover tiles of that pattern line
            }
        )

    floor = before.floor[player]
    floor_tiles = [c for c in range(NUM_COLORS) for _ in range(floor[c])]
    occupied = before.floor_occupied(player)
    penalty = CUM_PENALTY[min(FLOOR_SLOTS, occupied)]
    score_before = before.scores[player]
    return {
        "seat": player,
        "tiles": tiles,
        "tiling_points": gain,
        "carried_rows": [
            {"row": r, "color": colors[r], "count": counts[r]}
            for r in range(NUM_ROWS)
            if counts[r] and counts[r] != r + 1
        ],
        "floor": {
            "tiles": floor_tiles,
            "marker": before.floor_marker[player],
            "occupied": occupied,
            "penalty": penalty,
            "slot_penalties": list(FLOOR_PENALTIES[: min(FLOOR_SLOTS, occupied)]),
        },
        "score_before": score_before,
        "score_after": max(0, score_before + gain + penalty),
        "delta": max(0, score_before + gain + penalty) - score_before,
    }


def after_placement(before: AzulState, move: dict[str, Any]) -> AzulState:
    """``before`` with ``move``'s tiles placed, i.e. the state the engine tiles from.

    ``AzulState.apply`` places the tiles *and* resolves the round in one step, so
    the position the wall-tiling actually starts from is never observable. Only
    the mover's own pattern lines and floor change, and :func:`describe_action`
    already worked out how the tiles split, so replaying that much is enough for
    the report (the emptied factory/center play no part in scoring).
    """
    state = before.clone()
    player = move["player"]
    if move["took_marker"]:
        state.floor_marker[player] = True
    dest = move["dest"]
    if dest != FLOOR:
        state.pl_color[player][dest] = move["color"]
        state.pl_count[player][dest] += move["placed"]
    if move["to_floor"]:
        state.floor[player][move["color"]] += move["to_floor"]
    return state


def round_report(before: AzulState, move: dict[str, Any]) -> dict[str, Any]:
    """Wall-tiling result of the round ended by ``move`` played in ``before``."""
    state = after_placement(before, move)
    return {
        "round": state.round_index,
        "players": [_tile_player(state, p) for p in range(state.num_players)],
    }


def _bonus_breakdown(state: AzulState, player: int) -> dict[str, Any]:
    rows = state.completed_rows(player)
    cols = state.completed_cols(player)
    colors = state.completed_colors(player)
    row_pts = ROW_BONUS * rows
    col_pts = COL_BONUS * cols
    color_pts = COLOR_BONUS * colors
    total = row_pts + col_pts + color_pts
    return {
        "seat": player,
        "rows": rows,
        "row_points": row_pts,
        "cols": cols,
        "col_points": col_pts,
        "colors": colors,
        "color_points": color_pts,
        "total": total,
        "score_before_bonus": state.scores[player] - total,
        "final_score": state.scores[player],
    }


def final_report(state: AzulState, human_seat: int) -> dict[str, Any] | None:
    """End-of-game summary: bonuses per player, winner, why the game stopped."""
    if not state.is_terminal:
        return None
    outcome = state.outcome() or 0.0
    if outcome == 0.0:
        winner: int | None = None
    else:
        winner = 0 if outcome > 0 else 1
    if winner is None:
        headline = "A draw — same score, same completed rows."
    elif winner == human_seat:
        headline = "You win!"
    else:
        headline = "The AI wins."
    return {
        "winner": winner,
        "winner_side": None
        if winner is None
        else ("human" if winner == human_seat else "ai"),
        "headline": headline,
        "scores": state.scores[:],
        "bonuses": [_bonus_breakdown(state, p) for p in range(state.num_players)],
        "exhausted": state.exhausted,
        "rounds_played": state.round_index + 1,
    }
