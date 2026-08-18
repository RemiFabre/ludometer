"""Exact win/draw/loss solvers for tic-tac-toe and Connect Four.

Both return the game-theoretic value **for the player to move**: ``1`` win,
``0`` draw, ``-1`` loss. Tic-tac-toe is a memoised full search (5478 reachable
positions). Connect Four is a WDL negamax over the engine's own bitboards with
alpha-beta, a transposition table shared across calls, mirror normalisation
and the two classic prunings (an immediate win ends the node; a single enemy
winning square forces the reply, two lose outright). The perfect player never
joins the Elo anchor pool — see the brief for why.
"""

from __future__ import annotations

from functools import lru_cache

from ludometer.c4.engine import HEIGHT, WIDTH
from ludometer.ttt.engine import FULL as TTT_FULL
from ludometer.ttt.engine import WIN_MASKS

__all__ = ["c4_solve", "c4_solver_reset", "c4_solver_stats", "ttt_solve"]

# ------------------------------------------------------------------ tic-tac-toe
@lru_cache(maxsize=None)
def ttt_solve(me: int, them: int) -> int:
    """Value for the player to move holding ``me`` against ``them``."""
    if any(them & m == m for m in WIN_MASKS):
        return -1  # the opponent's last move completed a line
    board = me | them
    if board == TTT_FULL:
        return 0
    best = -1
    for cell in range(9):
        bit = 1 << cell
        if board & bit:
            continue
        mine = me | bit
        if any(mine & m == m for m in WIN_MASKS):
            return 1
        value = -ttt_solve(them, mine)
        if value > best:
            best = value
            if best == 1:
                return 1
    return best


# ----------------------------------------------------------------- Connect Four
H1 = HEIGHT + 1
BOTTOM = tuple(1 << (col * H1) for col in range(WIDTH))
COLUMN = tuple(((1 << HEIGHT) - 1) << (col * H1) for col in range(WIDTH))
BOARD_MASK = sum(COLUMN)
BOTTOM_ALL = sum(BOTTOM)
CENTER_ORDER = (3, 2, 4, 1, 5, 0, 6)

# entries are packed as (flag + 1) * 4 + (value + 1): two small ints, no tuple
_TT: dict[int, int] = {}
_TT_LIMIT = 8_000_000  # packed ints: ~50 bytes/entry, ~400 MB at the cap
_nodes = 0


def c4_solver_stats() -> dict[str, int]:
    return {"tt_entries": len(_TT), "nodes": _nodes}


def c4_solver_reset() -> None:
    """Drop the shared transposition table (frees a few hundred MB after a build)."""
    _TT.clear()


def _winning_spots(pos: int, mask: int) -> int:
    """Empty squares that would complete four-in-a-row for ``pos``."""
    r = (pos << 1) & (pos << 2) & (pos << 3)  # vertical
    for delta in (HEIGHT, H1, HEIGHT + 2):  # \ , — , /
        p = (pos << delta) & (pos << 2 * delta)
        r |= p & (pos << 3 * delta)
        r |= p & (pos >> delta)
        p = (pos >> delta) & (pos >> 2 * delta)
        r |= p & (pos >> 3 * delta)
        r |= p & (pos << delta)
    return r & (BOARD_MASK ^ mask)


def _mirror(board: int) -> int:
    out = 0
    for col in range(WIDTH):
        column = (board >> (col * H1)) & ((1 << H1) - 1)
        out |= column << ((WIDTH - 1 - col) * H1)
    return out


def c4_solve(position: int, mask: int) -> int:
    """WDL for the side to move; the opponent must not already have won."""
    return _negamax(position, mask, -1, 1)


def _negamax(pos: int, mask: int, alpha: int, beta: int) -> int:
    global _nodes
    _nodes += 1
    if mask == BOARD_MASK:
        return 0
    playable = (mask + BOTTOM_ALL) & BOARD_MASK
    if _winning_spots(pos, mask) & playable:
        return 1
    opp = pos ^ mask
    opp_wins = _winning_spots(opp, mask) & playable
    if opp_wins:
        if opp_wins & (opp_wins - 1):
            return -1  # two winning squares: only one can be blocked
        forced = opp_wins
        moves = [next(c for c in range(WIDTH) if COLUMN[c] & forced)]
    else:
        moves = [c for c in CENTER_ORDER if playable & COLUMN[c]]

    key = (pos << 49) | mask
    mirrored = (_mirror(pos) << 49) | _mirror(mask)
    if mirrored < key:
        key = mirrored
    entry = _TT.get(key)
    if entry is not None:
        flag, value = entry // 4 - 1, entry % 4 - 1
        if flag == 0:
            return value
        if flag < 0:  # upper bound
            if value <= alpha:
                return value
            beta = min(beta, value)
        else:  # lower bound
            if value >= beta:
                return value
            alpha = max(alpha, value)
        if alpha >= beta:
            return value

    orig_alpha = alpha
    best = -1
    for col in moves:
        added = (mask + BOTTOM[col]) & COLUMN[col]
        value = -_negamax(pos ^ mask, mask | added, -beta, -alpha)
        if value > best:
            best = value
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    if len(_TT) < _TT_LIMIT:
        flag = -1 if best <= orig_alpha else (1 if best >= beta else 0)
        _TT[key] = (flag + 1) * 4 + (best + 1)
    return best
