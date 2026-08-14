"""Cheap state features shared by the hand-written agents.

Everything here is read-only with respect to the state it is given and works on
plain Python lists so it stays fast enough for a 1-ply search over ~60 actions
(target: well under 1 ms per move).
"""

from __future__ import annotations

from ludometer.azul.engine import (
    NUM_COLORS,
    NUM_ROWS,
    WALL_IDX,
    AzulState,
)

__all__ = [
    "board_color_counts",
    "immediate_value",
    "tile_score",
    "tiling_gain",
    "virtual_wall",
    "wall_progress",
]


def tile_score(wall: list[int], row: int, col: int) -> int:
    """Azul score for a tile at (row, col) of ``wall`` (the cell must be set)."""
    row_base = row * 5
    h = 1
    i = col - 1
    while i >= 0 and wall[row_base + i]:
        h += 1
        i -= 1
    i = col + 1
    while i < 5 and wall[row_base + i]:
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
    if h > 1 or v > 1:
        return (h if h > 1 else 0) + (v if v > 1 else 0)
    return 1


def virtual_wall(state: AzulState, player: int) -> tuple[int, list[int]]:
    """Tile every *complete* pattern line of ``player`` on a copy of the wall.

    Returns ``(score_gain, wall_copy)`` — exactly what the engine would award at
    round end for the lines that are full right now (floor penalties excluded).
    Rows are resolved top to bottom, like the engine does.
    """
    wall = state.walls[player][:]
    plc = state.pl_color[player]
    pln = state.pl_count[player]
    gain = 0
    for r in range(NUM_ROWS):
        if pln[r] != r + 1:
            continue
        idx = WALL_IDX[plc[r] * 5 + r]
        wall[idx] = 1
        gain += tile_score(wall, r, idx - r * 5)
    return gain, wall


def tiling_gain(state: AzulState, player: int) -> int:
    """Points ``player`` would bank right now from their complete pattern lines."""
    return virtual_wall(state, player)[0]


def immediate_value(state: AzulState, player: int) -> float:
    """Greedy value: banked score + pending tiling gain - floor damage.

    Works both before and after a round boundary: once the engine has tiled, the
    gain and the penalty are already inside ``scores`` and the pending terms are
    zero, so the quantity is comparable across the boundary.
    """
    return (
        state.scores[player] + tiling_gain(state, player) + state.floor_penalty(player)
    )


def board_color_counts(state: AzulState) -> tuple[list[int], int]:
    """Per-color tiles available on factories+center, and the biggest single take.

    The second value is the largest monochrome group sitting in one pool — what
    an opponent could scoop in a single turn.
    """
    counts = [0] * NUM_COLORS
    best = 0
    for pool in state.factories:
        for c in range(NUM_COLORS):
            n = pool[c]
            if n:
                counts[c] += n
                best = max(best, n)
    cen = state.center
    for c in range(NUM_COLORS):
        n = cen[c]
        if n:
            counts[c] += n
            best = max(best, n)
    return counts, best


def wall_progress(wall: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Tiles per row, per column and per color on ``wall``."""
    rows = [0] * 5
    cols = [0] * 5
    colors = [0] * NUM_COLORS
    for r in range(5):
        base = r * 5
        for col in range(5):
            if wall[base + col]:
                rows[r] += 1
                cols[col] += 1
                colors[(col - r) % NUM_COLORS] += 1
    return rows, cols, colors
