"""Azul rules engine — official 2-player rules (see docs/DESIGN.md).

Pure Python + numpy (numpy is only used by :meth:`AzulState.encode`), no torch.
Fully deterministic given the ``new_game`` seed: all chance lives in an internal
``random.Random`` and is only consumed at round refill.

Board / state layout (all plain Python lists, cheap to copy):

``factories``      5 lists of 5 ints — tile counts per color per factory display
``center``         5 ints — tile counts per color in the center
``marker_in_center``  bool — is the first-player marker still in the center
``bag``            list of color ints, shuffled; tiles are drawn with ``pop()``
``lid``            5 ints — discard counts per color
``walls[p]``       25 ints (0/1), row-major; square (r, col) is index ``r * 5 + col``
``pl_color[p]``    5 ints — color of each pattern line, -1 when empty
``pl_count[p]``    5 ints — tiles in each pattern line (row r has capacity r + 1)
``floor[p]``       5 ints — tile counts per color on the floor line
``floor_marker[p]``   bool — does this player hold the first-player marker
``scores[p]``      int
``first_player``   player who starts the current round (marker holder)
``tiles_left``     tiles still on factories + center (round ends when it hits 0)

Colors are 0..4 = blue, yellow, red, black, teal.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

# Allocate a `random.Random` without seeding it (see AzulState.clone).
_NEW_RANDOM = random.Random.__new__

__all__ = [
    "ACTION_SPACE",
    "CENTER",
    "COLOR_NAMES",
    "ENCODED_SIZE",
    "FACTORY_SIZE",
    "FLOOR",
    "FLOOR_PENALTIES",
    "NUM_COLORS",
    "NUM_FACTORIES",
    "TILES_PER_COLOR",
    "AzulState",
    "decode_action",
    "encode_action",
    "wall_col",
]

NUM_COLORS = 5
TILES_PER_COLOR = 20
NUM_TILES = NUM_COLORS * TILES_PER_COLOR
NUM_FACTORIES = 5  # 2-player count
FACTORY_SIZE = 4
NUM_ROWS = 5
CENTER = 5  # action `source` value for the center
FLOOR = 5  # action `dest` value for the floor line
ACTION_SPACE = 180  # source (6) * color (5) * dest (6)

FLOOR_PENALTIES = (-1, -1, -2, -2, -2, -3, -3)
FLOOR_SLOTS = len(FLOOR_PENALTIES)
# cumulative penalty for n occupied floor slots (index 0..7)
_CUM_PENALTY = [0]
for _p in FLOOR_PENALTIES:
    _CUM_PENALTY.append(_CUM_PENALTY[-1] + _p)
CUM_PENALTY = tuple(_CUM_PENALTY)

COLOR_NAMES = ("blue", "yellow", "red", "black", "teal")
COLOR_CHARS = "BYRKT"

ROW_BONUS = 2
COL_BONUS = 7
COLOR_BONUS = 10

# flat wall lookup: WALL_IDX[color * 5 + row] -> index into the 25-cell wall
WALL_IDX = tuple(
    (r * 5 + (c + r) % 5) for c in range(NUM_COLORS) for r in range(NUM_ROWS)
)

# --- lookup tables for the hot path -----------------------------------------
#
# A player's placement options are summarised by one 5-bit mask per color
# ("open_mask"): bit r is set when that color may still go into pattern line r.
# _ACTION_TABLE[source][color][mask] is then the ready-made tuple of action ids
# (rows in ascending order, floor last), so legal_actions() is just a few
# C-level list extends.
_ACTION_TABLE = tuple(
    tuple(
        tuple(
            tuple(
                [src * 30 + c * 6 + r for r in range(NUM_ROWS) if (mask >> r) & 1]
                + [src * 30 + c * 6 + FLOOR]
            )
            for mask in range(1 << NUM_ROWS)
        )
        for c in range(NUM_COLORS)
    )
    for src in range(6)
)
_ALL_ROWS = (1 << NUM_ROWS) - 1
# _AND_KEEP[color][row]: AND-masks closing `row` for every color but `color`
_AND_KEEP = tuple(
    tuple(
        tuple(
            _ALL_ROWS if c2 == c else _ALL_ROWS ^ (1 << r) for c2 in range(NUM_COLORS)
        )
        for r in range(NUM_ROWS)
    )
    for c in range(NUM_COLORS)
)
# _AND_CLOSE[row]: AND-masks closing `row` for every color (line full / tiled)
_AND_CLOSE = tuple(
    tuple(_ALL_ROWS ^ (1 << r) for _ in range(NUM_COLORS)) for r in range(NUM_ROWS)
)


def wall_col(color: int, row: int) -> int:
    """Wall column of `color` in `row` on the standard fixed wall."""
    return (color + row) % NUM_COLORS


def encode_action(source: int, color: int, dest: int) -> int:
    """`source` 0..4 factories / 5 center, `color` 0..4, `dest` 0..4 rows / 5 floor."""
    return source * 30 + color * 6 + dest


def decode_action(action_id: int) -> tuple[int, int, int]:
    """Inverse of :func:`encode_action`; returns ``(source, color, dest)``."""
    source, rest = divmod(action_id, 30)
    color, dest = divmod(rest, 6)
    return source, color, dest


# --------------------------------------------------------------------- encoding
#
# encode() layout (float32, current player = "me", opponent = "them"):
#
#   [  0:  25)  my wall, row-major 5x5 (0/1)
#   [ 25:  50)  their wall
#   [ 50:  80)  my pattern lines: 5 rows x (5 color one-hot, fill = count/(r+1))
#   [ 80: 110)  their pattern lines, same layout
#   [110: 117)  my floor: 5 color counts /7, occupied slots /7, marker flag
#   [117: 124)  their floor, same layout
#   [124: 126)  scores /100 (mine, theirs)
#   [126: 151)  factories: 5 x 5 color counts /4
#   [151: 156)  per-factory non-empty flag
#   [156: 161)  center color counts /10
#   [161: 162)  center total /20
#   [162: 163)  first-player marker still in the center (0/1)
#   [163: 168)  bag color counts /20            (public information)
#   [168: 173)  lid color counts /20            (public information)
#   [173: 174)  tiles left on the board this round /20  (round position)
#   [174: 175)  I hold the first-player marker (i.e. I start the next round)
#   [175: 176)  round index /10
#   [176: 179)  my completed rows /5, columns /5, colors /5
#   [179: 182)  their completed rows /5, columns /5, colors /5
#
OFF_MY_WALL = 0
OFF_OP_WALL = 25
OFF_MY_LINES = 50
OFF_OP_LINES = 80
OFF_MY_FLOOR = 110
OFF_OP_FLOOR = 117
OFF_SCORES = 124
OFF_FACTORIES = 126
OFF_FACTORY_FLAGS = 151
OFF_CENTER = 156
OFF_CENTER_TOTAL = 161
OFF_MARKER_CENTER = 162
OFF_BAG = 163
OFF_LID = 168
OFF_TILES_LEFT = 173
OFF_I_START = 174
OFF_ROUND = 175
OFF_MY_SETS = 176
OFF_OP_SETS = 179
ENCODED_SIZE = 182


class AzulState:
    """Mutable Azul game state for exactly two players."""

    ACTION_SPACE: int = ACTION_SPACE
    ENCODED_SIZE: int = ENCODED_SIZE

    __slots__ = (
        "bag",
        "center",
        "current_player",
        "exhausted",
        "factories",
        "first_player",
        "floor",
        "floor_marker",
        "is_terminal",
        "lid",
        "marker_in_center",
        "num_players",
        "open_mask",
        "pl_color",
        "pl_count",
        "rng",
        "round_index",
        "scores",
        "tiles_left",
        "walls",
    )

    # ------------------------------------------------------------------ setup
    @classmethod
    def new_game(cls, seed: int, num_players: int = 2) -> AzulState:
        if num_players != 2:
            raise ValueError("only the 2-player game is implemented")
        self = cls.__new__(cls)
        self.num_players = num_players
        self.rng = random.Random(seed)
        self.bag = [c for c in range(NUM_COLORS) for _ in range(TILES_PER_COLOR)]
        self.rng.shuffle(self.bag)
        self.lid = [0] * NUM_COLORS
        self.factories = [[0] * NUM_COLORS for _ in range(NUM_FACTORIES)]
        self.center = [0] * NUM_COLORS
        self.marker_in_center = True
        self.walls = [[0] * 25 for _ in range(num_players)]
        self.pl_color = [[-1] * NUM_ROWS for _ in range(num_players)]
        self.pl_count = [[0] * NUM_ROWS for _ in range(num_players)]
        self.floor = [[0] * NUM_COLORS for _ in range(num_players)]
        self.floor_marker = [False] * num_players
        self.open_mask = [[_ALL_ROWS] * NUM_COLORS for _ in range(num_players)]
        self.scores = [0] * num_players
        self.current_player = 0
        self.first_player = 0
        self.round_index = 0
        self.is_terminal = False
        self.exhausted = False
        self.tiles_left = 0
        self._refill()
        return self

    def clone(self) -> AzulState:
        """Deep-enough copy: every mutable container is duplicated."""
        other = AzulState.__new__(AzulState)
        other.num_players = self.num_players
        # `random.Random()` seeds itself from the OS *twice* (once in the C
        # `__new__`, once in `__init__`), which costs ~19 us and is thrown away
        # by the `setstate` on the next line. Calling `__new__` directly skips
        # both seedings; `setstate` then installs the parent's Mersenne state
        # and its `gauss_next`, so the clone's stream is bit-identical to what
        # the old two-step produced. Measured 30.1 us -> 11.0 us per clone, and
        # MCTS clones a state on every node it creates.
        other.rng = _NEW_RANDOM(random.Random)
        other.rng.setstate(self.rng.getstate())
        other.bag = self.bag[:]
        other.lid = self.lid[:]
        other.factories = [f[:] for f in self.factories]
        other.center = self.center[:]
        other.marker_in_center = self.marker_in_center
        other.walls = [w[:] for w in self.walls]
        other.pl_color = [x[:] for x in self.pl_color]
        other.pl_count = [x[:] for x in self.pl_count]
        other.floor = [f[:] for f in self.floor]
        other.floor_marker = self.floor_marker[:]
        other.open_mask = [m[:] for m in self.open_mask]
        other.scores = self.scores[:]
        other.current_player = self.current_player
        other.first_player = self.first_player
        other.round_index = self.round_index
        other.is_terminal = self.is_terminal
        other.exhausted = self.exhausted
        other.tiles_left = self.tiles_left
        return other

    def recount(self) -> None:
        """Rebuild derived caches (``tiles_left`` and the placement masks).

        ``apply`` keeps them up to date by itself; call this after editing
        ``factories`` / ``center`` / ``pl_*`` / ``walls`` by hand (tests, GUI).
        """
        total = (
            self.center[0]
            + self.center[1]
            + self.center[2]
            + self.center[3]
            + self.center[4]
        )
        for f in self.factories:
            total += f[0] + f[1] + f[2] + f[3] + f[4]
        self.tiles_left = total
        for p in range(self.num_players):
            self._rebuild_mask(p)

    def _rebuild_mask(self, player: int) -> None:
        wall = self.walls[player]
        plc = self.pl_color[player]
        pln = self.pl_count[player]
        masks = self.open_mask[player]
        for c in range(NUM_COLORS):
            base = c * 5
            m = 0
            for r in range(NUM_ROWS):
                n = pln[r]
                if n <= r and (n == 0 or plc[r] == c) and not wall[WALL_IDX[base + r]]:
                    m |= 1 << r
            masks[c] = m

    # ------------------------------------------------------------ legal moves
    def legal_actions(self) -> list[int]:
        """Legal action ids for :attr:`current_player` (empty iff terminal)."""
        if self.is_terminal:
            return []
        masks = self.open_mask[self.current_player]
        m0, m1, m2, m3, m4 = masks
        out: list[int] = []
        for src, pool in enumerate(self.factories):
            table = _ACTION_TABLE[src]
            if pool[0]:
                out += table[0][m0]
            if pool[1]:
                out += table[1][m1]
            if pool[2]:
                out += table[2][m2]
            if pool[3]:
                out += table[3][m3]
            if pool[4]:
                out += table[4][m4]
        pool = self.center
        table = _ACTION_TABLE[CENTER]
        if pool[0]:
            out += table[0][m0]
        if pool[1]:
            out += table[1][m1]
        if pool[2]:
            out += table[2][m2]
        if pool[3]:
            out += table[3][m3]
        if pool[4]:
            out += table[4][m4]
        return out

    def is_legal(self, action_id: int) -> bool:
        if self.is_terminal or not 0 <= action_id < ACTION_SPACE:
            return False
        src, color, dest = decode_action(action_id)
        pool = self.center if src == CENTER else self.factories[src]
        if pool[color] == 0:
            return False
        if dest == FLOOR:
            return True
        p = self.current_player
        n = self.pl_count[p][dest]
        if n > dest:  # line already full
            return False
        if n and self.pl_color[p][dest] != color:
            return False
        return not self.walls[p][WALL_IDX[color * 5 + dest]]

    # ------------------------------------------------------------------ moves
    def apply(self, action_id: int) -> None:
        """Play `action_id`, then resolve round end / refill / game end as needed."""
        if self.is_terminal:
            raise ValueError("game is over")
        if not 0 <= action_id < ACTION_SPACE:
            raise ValueError(f"action {action_id} out of range")
        src, rest = divmod(action_id, 30)
        color, dest = divmod(rest, 6)

        p = self.current_player
        pool = self.center if src == CENTER else self.factories[src]
        count = pool[color]
        if count == 0:
            raise ValueError(f"no color {color} at source {src}")

        # validate the destination before mutating anything
        if dest != FLOOR:
            held = self.pl_count[p][dest]
            if held > dest:
                raise ValueError(f"pattern line {dest} is full")
            if held and self.pl_color[p][dest] != color:
                raise ValueError(f"pattern line {dest} holds another color")
            if self.walls[p][WALL_IDX[color * 5 + dest]]:
                raise ValueError(f"color {color} already on wall row {dest}")

        # --- take the tiles
        pool[color] = 0
        if src == CENTER:
            if self.marker_in_center:
                self.marker_in_center = False
                self.floor_marker[p] = True
        else:
            cen = self.center
            for c in range(NUM_COLORS):
                n = pool[c]
                if n:
                    cen[c] += n
                    pool[c] = 0
        self.tiles_left -= count

        # --- place them
        if dest != FLOOR:
            pln = self.pl_count[p]
            room = dest + 1 - pln[dest]
            self.pl_color[p][dest] = color
            if count < room:
                pln[dest] += count
                overflow = 0
                keep = _AND_KEEP[color][dest]
            else:
                pln[dest] = dest + 1
                overflow = count - room
                keep = _AND_CLOSE[dest]  # the line is now full
            masks = self.open_mask[p]
            masks[0] &= keep[0]
            masks[1] &= keep[1]
            masks[2] &= keep[2]
            masks[3] &= keep[3]
            masks[4] &= keep[4]
        else:
            overflow = count

        if overflow:
            fl = self.floor[p]
            occupied = fl[0] + fl[1] + fl[2] + fl[3] + fl[4]
            if self.floor_marker[p]:
                occupied += 1
            room = FLOOR_SLOTS - occupied
            if overflow <= room:
                fl[color] += overflow
            elif room > 0:
                fl[color] += room
                self.lid[color] += overflow - room
            else:
                self.lid[color] += overflow

        # --- round / game transitions
        if self.tiles_left:
            self.current_player = 1 - p
        else:
            self._end_round(p)

    # ------------------------------------------------------------ round logic
    def _end_round(self, last_mover: int) -> None:
        """Wall-tiling, scoring, floor penalties, then refill or finish the game."""
        lid = self.lid
        for q in range(self.num_players):
            wall = self.walls[q]
            plc = self.pl_color[q]
            pln = self.pl_count[q]
            gain = 0
            for r in range(NUM_ROWS):
                if pln[r] != r + 1:
                    continue
                c = plc[r]
                idx = WALL_IDX[c * 5 + r]
                wall[idx] = 1
                row_base = r * 5
                col = idx - row_base
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
                i = r - 1
                while i >= 0 and wall[i * 5 + col]:
                    v += 1
                    i -= 1
                i = r + 1
                while i < 5 and wall[i * 5 + col]:
                    v += 1
                    i += 1
                if h > 1 or v > 1:
                    gain += (h if h > 1 else 0) + (v if v > 1 else 0)
                else:
                    gain += 1
                lid[c] += r  # the r leftover tiles of the line
                plc[r] = -1
                pln[r] = 0

            fl = self.floor[q]
            occupied = fl[0] + fl[1] + fl[2] + fl[3] + fl[4]
            if self.floor_marker[q]:
                occupied += 1
            gain += CUM_PENALTY[min(FLOOR_SLOTS, occupied)]
            for c in range(NUM_COLORS):
                n = fl[c]
                if n:
                    lid[c] += n
                    fl[c] = 0
            total = self.scores[q] + gain
            self.scores[q] = max(0, total)
            self._rebuild_mask(q)

        # who starts next round: the marker holder (marker goes back to the center)
        holder = None
        for q in range(self.num_players):
            if self.floor_marker[q]:
                self.floor_marker[q] = False
                holder = q
        if holder is None:
            # Nobody ever took from the center this round (possible when every
            # factory is monochrome): the marker stays in the center and normal
            # alternation decides who starts.
            holder = 1 - last_mover
        self.first_player = holder
        self.marker_in_center = True
        self.current_player = holder

        if self._any_row_complete():
            self._finish()
            return

        self.round_index += 1
        self._refill()
        if self.tiles_left == 0:
            # No tiles anywhere: cannot deal another round, stop the game.
            self.exhausted = True
            self._finish()

    def _any_row_complete(self) -> bool:
        for wall in self.walls:
            for base in (0, 5, 10, 15, 20):
                if (
                    wall[base]
                    and wall[base + 1]
                    and wall[base + 2]
                    and wall[base + 3]
                    and wall[base + 4]
                ):
                    return True
        return False

    def _finish(self) -> None:
        for q in range(self.num_players):
            self.scores[q] += (
                ROW_BONUS * self.completed_rows(q)
                + COL_BONUS * self.completed_cols(q)
                + COLOR_BONUS * self.completed_colors(q)
            )
        self.is_terminal = True

    def _refill(self) -> None:
        bag = self.bag
        lid = self.lid
        rng = self.rng
        total = 0
        for f in self.factories:
            for _ in range(FACTORY_SIZE):
                if not bag:
                    for c in range(NUM_COLORS):
                        n = lid[c]
                        if n:
                            bag.extend([c] * n)
                            lid[c] = 0
                    if not bag:
                        self.tiles_left = total
                        return
                    rng.shuffle(bag)
                f[bag.pop()] += 1
                total += 1
        self.tiles_left = total

    # ------------------------------------------------------------ inspection
    def floor_occupied(self, player: int) -> int:
        """Floor slots in use, marker included (may exceed the 7 scoring slots)."""
        fl = self.floor[player]
        return sum(fl) + (1 if self.floor_marker[player] else 0)

    def floor_penalty(self, player: int) -> int:
        occupied = self.floor_occupied(player)
        return CUM_PENALTY[min(FLOOR_SLOTS, occupied)]

    def completed_rows(self, player: int) -> int:
        wall = self.walls[player]
        return sum(1 for base in (0, 5, 10, 15, 20) if all(wall[base : base + 5]))

    def completed_cols(self, player: int) -> int:
        wall = self.walls[player]
        return sum(1 for col in range(5) if all(wall[col::5]))

    def completed_colors(self, player: int) -> int:
        wall = self.walls[player]
        done = 0
        for c in range(NUM_COLORS):
            base = c * 5
            if all(wall[WALL_IDX[base + r]] for r in range(NUM_ROWS)):
                done += 1
        return done

    def outcome(self) -> float | None:
        """+1 if player 0 wins, -1 if player 1 wins, 0 for a draw, None if unfinished."""
        if not self.is_terminal:
            return None
        s0, s1 = self.scores[0], self.scores[1]
        if s0 != s1:
            return 1.0 if s0 > s1 else -1.0
        r0, r1 = self.completed_rows(0), self.completed_rows(1)
        if r0 != r1:
            return 1.0 if r0 > r1 else -1.0
        return 0.0

    def bag_counts(self) -> list[int]:
        counts = [0] * NUM_COLORS
        for c in self.bag:
            counts[c] += 1
        return counts

    def tile_census(self) -> list[int]:
        """Count all tiles, wherever they are. Must always be ``[20] * 5``."""
        counts = self.bag_counts()
        for c in range(NUM_COLORS):
            counts[c] += self.lid[c]
            counts[c] += self.center[c]
            for f in self.factories:
                counts[c] += f[c]
            for p in range(self.num_players):
                counts[c] += self.floor[p][c]
        for p in range(self.num_players):
            plc = self.pl_color[p]
            pln = self.pl_count[p]
            for r in range(NUM_ROWS):
                if pln[r]:
                    counts[plc[r]] += pln[r]
            wall = self.walls[p]
            for r in range(NUM_ROWS):
                for col in range(5):
                    if wall[r * 5 + col]:
                        counts[(col - r) % NUM_COLORS] += 1
        return counts

    # --------------------------------------------------------------- encoding
    def encode(self) -> np.ndarray:
        """Fixed-size float32 observation from the current player's perspective.

        See the module-level "encode() layout" comment for the exact field map;
        :data:`ENCODED_SIZE` is the vector length (182).
        """
        v = np.zeros(ENCODED_SIZE, dtype=np.float32)
        me = self.current_player
        op = 1 - me

        v[OFF_MY_WALL : OFF_MY_WALL + 25] = self.walls[me]
        v[OFF_OP_WALL : OFF_OP_WALL + 25] = self.walls[op]

        for off, p in ((OFF_MY_LINES, me), (OFF_OP_LINES, op)):
            plc = self.pl_color[p]
            pln = self.pl_count[p]
            for r in range(NUM_ROWS):
                n = pln[r]
                if n:
                    base = off + r * 6
                    v[base + plc[r]] = 1.0
                    v[base + 5] = n / (r + 1)

        for off, p in ((OFF_MY_FLOOR, me), (OFF_OP_FLOOR, op)):
            fl = self.floor[p]
            for c in range(NUM_COLORS):
                if fl[c]:
                    v[off + c] = fl[c] / FLOOR_SLOTS
            v[off + 5] = min(self.floor_occupied(p), FLOOR_SLOTS) / FLOOR_SLOTS
            v[off + 6] = 1.0 if self.floor_marker[p] else 0.0

        v[OFF_SCORES] = self.scores[me] / 100.0
        v[OFF_SCORES + 1] = self.scores[op] / 100.0

        for i, f in enumerate(self.factories):
            base = OFF_FACTORIES + i * 5
            total = 0
            for c in range(NUM_COLORS):
                n = f[c]
                if n:
                    v[base + c] = n / FACTORY_SIZE
                    total += n
            if total:
                v[OFF_FACTORY_FLAGS + i] = 1.0

        cen_total = 0
        for c in range(NUM_COLORS):
            n = self.center[c]
            if n:
                v[OFF_CENTER + c] = n / 10.0
                cen_total += n
        v[OFF_CENTER_TOTAL] = cen_total / 20.0
        v[OFF_MARKER_CENTER] = 1.0 if self.marker_in_center else 0.0

        bag = self.bag_counts()
        for c in range(NUM_COLORS):
            v[OFF_BAG + c] = bag[c] / TILES_PER_COLOR
            v[OFF_LID + c] = self.lid[c] / TILES_PER_COLOR

        v[OFF_TILES_LEFT] = self.tiles_left / 20.0
        v[OFF_I_START] = (
            1.0 if (self.floor_marker[me] or self.first_player == me) else 0.0
        )
        v[OFF_ROUND] = min(self.round_index, 10) / 10.0

        for off, p in ((OFF_MY_SETS, me), (OFF_OP_SETS, op)):
            v[off] = self.completed_rows(p) / 5.0
            v[off + 1] = self.completed_cols(p) / 5.0
            v[off + 2] = self.completed_colors(p) / 5.0
        return v

    # ---------------------------------------------------------------- display
    def render_text(self) -> str:
        lines = [
            f"round {self.round_index}  "
            f"to move: P{self.current_player}  "
            f"first player: P{self.first_player}  "
            f"scores: {self.scores[0]}-{self.scores[1]}"
            + ("  [GAME OVER]" if self.is_terminal else "")
        ]
        lines.append("Factories:")
        for i, f in enumerate(self.factories):
            tiles = "".join(COLOR_CHARS[c] * f[c] for c in range(NUM_COLORS))
            lines.append(f"  {i}: {tiles if tiles else '-'}")
        cen = "".join(COLOR_CHARS[c] * self.center[c] for c in range(NUM_COLORS))
        marker = " [1st]" if self.marker_in_center else ""
        lines.append(f"  center: {cen if cen else '-'}{marker}")
        bag = self.bag_counts()
        lines.append(
            "  bag: "
            + " ".join(f"{COLOR_CHARS[c]}{bag[c]}" for c in range(NUM_COLORS))
            + "   lid: "
            + " ".join(f"{COLOR_CHARS[c]}{self.lid[c]}" for c in range(NUM_COLORS))
        )
        for p in range(self.num_players):
            mark = "*" if p == self.current_player else " "
            lines.append(f"{mark}P{p}  score {self.scores[p]}")
            wall = self.walls[p]
            plc = self.pl_color[p]
            pln = self.pl_count[p]
            for r in range(NUM_ROWS):
                cap = r + 1
                n = pln[r]
                filled = COLOR_CHARS[plc[r]] * n if n else ""
                line = ("." * (cap - n) + filled).rjust(5)
                row = "".join(
                    COLOR_CHARS[(col - r) % NUM_COLORS] if wall[r * 5 + col] else "."
                    for col in range(5)
                )
                lines.append(f"    {line} | {row}")
            fl = self.floor[p]
            floor = "".join(COLOR_CHARS[c] * fl[c] for c in range(NUM_COLORS))
            if self.floor_marker[p]:
                floor += "#"  # first-player marker
            lines.append(
                f"    floor: {floor if floor else '-'} ({self.floor_penalty(p)})"
            )
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        """Full state as plain JSON-serialisable data (for the GUI)."""
        return {
            "round": self.round_index,
            "current_player": self.current_player,
            "first_player": self.first_player,
            "factories": [f[:] for f in self.factories],
            "center": self.center[:],
            "marker_in_center": self.marker_in_center,
            "bag": self.bag_counts(),
            "lid": self.lid[:],
            "tiles_left": self.tiles_left,
            "scores": self.scores[:],
            "is_terminal": self.is_terminal,
            "exhausted": self.exhausted,
            "outcome": self.outcome(),
            "legal_actions": self.legal_actions(),
            "color_names": list(COLOR_NAMES),
            "players": [
                {
                    "score": self.scores[p],
                    "wall": [self.walls[p][r * 5 : r * 5 + 5] for r in range(NUM_ROWS)],
                    "pattern_lines": [
                        {
                            "capacity": r + 1,
                            "color": self.pl_color[p][r],
                            "count": self.pl_count[p][r],
                        }
                        for r in range(NUM_ROWS)
                    ],
                    "floor": self.floor[p][:],
                    "floor_marker": self.floor_marker[p],
                    "floor_penalty": self.floor_penalty(p),
                    "completed_rows": self.completed_rows(p),
                    "completed_cols": self.completed_cols(p),
                    "completed_colors": self.completed_colors(p),
                }
                for p in range(self.num_players)
            ],
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<AzulState round={self.round_index} player={self.current_player} "
            f"scores={self.scores} terminal={self.is_terminal}>"
        )
