//! Azul rules, official 2-player game: a line-for-line twin of
//! `ludometer/azul/engine.py`. Read that module's docstring for the layout;
//! every method here keeps its Python name and its Python semantics, down to
//! the order tiles are popped from the bag and the order lid tiles are put back.
//!
//! The one structural difference: a [`State`] is a small `Copy` struct (no heap,
//! ~250 bytes) so the search clones it for free. The bag keeps its *order*
//! (`bag[..bag_len]`, popped from the end like `list.pop()`), because the
//! Python engine pre-shuffles its bag and a determinization must reshuffle the
//! same list to sample the same refill.

use crate::rng::{Rng, RngKind};

pub const NUM_COLORS: usize = 5;
pub const TILES_PER_COLOR: u8 = 20;
pub const NUM_TILES: usize = 100;
pub const NUM_FACTORIES: usize = 5;
pub const FACTORY_SIZE: usize = 4;
pub const NUM_ROWS: usize = 5;
pub const CENTER: usize = 5;
pub const FLOOR: usize = 5;
pub const ACTION_SPACE: usize = 180;
pub const ENCODED_SIZE: usize = 182;
pub const FLOOR_PENALTIES: [i32; 7] = [-1, -1, -2, -2, -2, -3, -3];
pub const FLOOR_SLOTS: usize = 7;
/// Cumulative penalty for `n` occupied floor slots, `n` in `0..=7`.
pub const CUM_PENALTY: [i32; 8] = [0, -1, -2, -4, -6, -8, -11, -14];
pub const ROW_BONUS: i32 = 2;
pub const COL_BONUS: i32 = 7;
pub const COLOR_BONUS: i32 = 10;

const ALL_ROWS: u8 = 0b1_1111;

// encode() offsets, copied from engine.py.
pub const OFF_MY_WALL: usize = 0;
pub const OFF_OP_WALL: usize = 25;
pub const OFF_MY_LINES: usize = 50;
pub const OFF_OP_LINES: usize = 80;
pub const OFF_MY_FLOOR: usize = 110;
pub const OFF_OP_FLOOR: usize = 117;
pub const OFF_SCORES: usize = 124;
pub const OFF_FACTORIES: usize = 126;
pub const OFF_FACTORY_FLAGS: usize = 151;
pub const OFF_CENTER: usize = 156;
pub const OFF_CENTER_TOTAL: usize = 161;
pub const OFF_MARKER_CENTER: usize = 162;
pub const OFF_BAG: usize = 163;
pub const OFF_LID: usize = 168;
pub const OFF_TILES_LEFT: usize = 173;
pub const OFF_I_START: usize = 174;
pub const OFF_ROUND: usize = 175;
pub const OFF_MY_SETS: usize = 176;
pub const OFF_OP_SETS: usize = 179;

/// Wall column of `color` in `row` on the fixed wall.
#[inline]
pub const fn wall_col(color: usize, row: usize) -> usize {
    (color + row) % NUM_COLORS
}

/// `WALL_IDX[color * 5 + row]`: index into the 25-cell wall.
#[inline]
const fn wall_idx(color: usize, row: usize) -> usize {
    row * 5 + wall_col(color, row)
}

#[inline]
pub const fn encode_action(source: usize, color: usize, dest: usize) -> u8 {
    (source * 30 + color * 6 + dest) as u8
}

#[inline]
pub const fn decode_action(action: u8) -> (usize, usize, usize) {
    let a = action as usize;
    (a / 30, (a % 30) / 6, a % 6)
}

/// A complete 2-player Azul position. `Copy`; no heap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State {
    pub factories: [[u8; NUM_COLORS]; NUM_FACTORIES],
    pub center: [u8; NUM_COLORS],
    pub lid: [u8; NUM_COLORS],
    /// Bag order; tiles are drawn from the end (`bag[bag_len - 1]`).
    pub bag: [u8; NUM_TILES],
    pub bag_len: u8,
    /// Wall bits per player: bit `r * 5 + col`.
    pub walls: [u32; 2],
    /// Colour of each pattern line, -1 when empty.
    pub pl_color: [[i8; NUM_ROWS]; 2],
    pub pl_count: [[u8; NUM_ROWS]; 2],
    pub floor: [[u8; NUM_COLORS]; 2],
    pub floor_marker: [bool; 2],
    /// `open_mask[p][c]`: bit `r` set when colour `c` may still go into row `r`.
    pub open_mask: [[u8; NUM_COLORS]; 2],
    pub scores: [i32; 2],
    pub marker_in_center: bool,
    pub current_player: u8,
    pub first_player: u8,
    pub round_index: u16,
    pub tiles_left: u8,
    pub is_terminal: bool,
    pub exhausted: bool,
    pub rng: Rng,
}

pub type ActionList = Vec<u8>;

impl State {
    // ------------------------------------------------------------------ setup
    /// `AzulState.new_game(seed)`; `kind` picks the generator (Python = exact twin).
    pub fn new_game(seed: u64, kind: RngKind) -> State {
        let mut bag = [0u8; NUM_TILES];
        for c in 0..NUM_COLORS {
            for i in 0..TILES_PER_COLOR as usize {
                bag[c * 20 + i] = c as u8;
            }
        }
        let mut rng = Rng::new(kind, seed);
        rng.shuffle(&mut bag);
        let mut s = State {
            factories: [[0; NUM_COLORS]; NUM_FACTORIES],
            center: [0; NUM_COLORS],
            lid: [0; NUM_COLORS],
            bag,
            bag_len: NUM_TILES as u8,
            walls: [0; 2],
            pl_color: [[-1; NUM_ROWS]; 2],
            pl_count: [[0; NUM_ROWS]; 2],
            floor: [[0; NUM_COLORS]; 2],
            floor_marker: [false; 2],
            open_mask: [[ALL_ROWS; NUM_COLORS]; 2],
            scores: [0; 2],
            marker_in_center: true,
            current_player: 0,
            first_player: 0,
            round_index: 0,
            tiles_left: 0,
            is_terminal: false,
            exhausted: false,
            rng,
        };
        s.refill();
        s
    }

    /// An empty board with no tiles anywhere (for `from_parts`); callers fill it.
    pub fn blank(kind: RngKind) -> State {
        State {
            factories: [[0; NUM_COLORS]; NUM_FACTORIES],
            center: [0; NUM_COLORS],
            lid: [0; NUM_COLORS],
            bag: [0; NUM_TILES],
            bag_len: 0,
            walls: [0; 2],
            pl_color: [[-1; NUM_ROWS]; 2],
            pl_count: [[0; NUM_ROWS]; 2],
            floor: [[0; NUM_COLORS]; 2],
            floor_marker: [false; 2],
            open_mask: [[ALL_ROWS; NUM_COLORS]; 2],
            scores: [0; 2],
            marker_in_center: true,
            current_player: 0,
            first_player: 0,
            round_index: 0,
            tiles_left: 0,
            is_terminal: false,
            exhausted: false,
            rng: Rng::new(kind, 0),
        }
    }

    #[inline]
    pub fn bag_slice(&self) -> &[u8] {
        &self.bag[..self.bag_len as usize]
    }

    /// Replace the bag with `tiles` in this order (the last one is drawn first).
    pub fn set_bag(&mut self, tiles: &[u8]) {
        self.bag_len = tiles.len() as u8;
        self.bag[..tiles.len()].copy_from_slice(tiles);
    }

    /// Rebuild `tiles_left` and the placement masks after a hand edit.
    pub fn recount(&mut self) {
        let mut total: u32 = self.center.iter().map(|&x| x as u32).sum();
        for f in &self.factories {
            total += f.iter().map(|&x| x as u32).sum::<u32>();
        }
        self.tiles_left = total as u8;
        for p in 0..2 {
            self.rebuild_mask(p);
        }
    }

    fn rebuild_mask(&mut self, p: usize) {
        let wall = self.walls[p];
        for c in 0..NUM_COLORS {
            let mut m = 0u8;
            for r in 0..NUM_ROWS {
                let n = self.pl_count[p][r] as usize;
                if n <= r
                    && (n == 0 || self.pl_color[p][r] == c as i8)
                    && wall & (1 << wall_idx(c, r)) == 0
                {
                    m |= 1 << r;
                }
            }
            self.open_mask[p][c] = m;
        }
    }

    // ------------------------------------------------------------ legal moves
    /// Legal action ids for the player to move, in the Python engine's order
    /// (factories 0..4 then the center; colours ascending; rows ascending, floor last).
    pub fn legal_actions_into(&self, out: &mut ActionList) {
        out.clear();
        if self.is_terminal {
            return;
        }
        let masks = &self.open_mask[self.current_player as usize];
        for src in 0..=NUM_FACTORIES {
            let pool = if src == CENTER { &self.center } else { &self.factories[src] };
            for c in 0..NUM_COLORS {
                if pool[c] == 0 {
                    continue;
                }
                let base = (src * 30 + c * 6) as u8;
                let m = masks[c];
                for r in 0..NUM_ROWS {
                    if (m >> r) & 1 == 1 {
                        out.push(base + r as u8);
                    }
                }
                out.push(base + FLOOR as u8);
            }
        }
    }

    pub fn legal_actions(&self) -> ActionList {
        let mut out = Vec::with_capacity(64);
        self.legal_actions_into(&mut out);
        out
    }

    pub fn is_legal(&self, action: i64) -> bool {
        if self.is_terminal || !(0..ACTION_SPACE as i64).contains(&action) {
            return false;
        }
        let (src, color, dest) = decode_action(action as u8);
        let pool = if src == CENTER { &self.center } else { &self.factories[src] };
        if pool[color] == 0 {
            return false;
        }
        if dest == FLOOR {
            return true;
        }
        let p = self.current_player as usize;
        let n = self.pl_count[p][dest] as usize;
        if n > dest {
            return false;
        }
        if n != 0 && self.pl_color[p][dest] != color as i8 {
            return false;
        }
        self.walls[p] & (1 << wall_idx(color, dest)) == 0
    }

    // ------------------------------------------------------------------ moves
    /// Play `action`, then resolve round end / refill / game end as needed.
    pub fn apply(&mut self, action: u8) -> Result<(), &'static str> {
        if self.is_terminal {
            return Err("game is over");
        }
        if action as usize >= ACTION_SPACE {
            return Err("action out of range");
        }
        let (src, color, dest) = decode_action(action);
        let p = self.current_player as usize;
        let count = if src == CENTER { self.center[color] } else { self.factories[src][color] };
        if count == 0 {
            return Err("no tiles of that color at that source");
        }
        if dest != FLOOR {
            let held = self.pl_count[p][dest] as usize;
            if held > dest {
                return Err("pattern line is full");
            }
            if held != 0 && self.pl_color[p][dest] != color as i8 {
                return Err("pattern line holds another color");
            }
            if self.walls[p] & (1 << wall_idx(color, dest)) != 0 {
                return Err("color already on that wall row");
            }
        }

        // --- take the tiles
        if src == CENTER {
            self.center[color] = 0;
            if self.marker_in_center {
                self.marker_in_center = false;
                self.floor_marker[p] = true;
            }
        } else {
            let pool = &mut self.factories[src];
            pool[color] = 0;
            for c in 0..NUM_COLORS {
                let n = pool[c];
                if n != 0 {
                    self.center[c] += n;
                    pool[c] = 0;
                }
            }
        }
        self.tiles_left -= count;

        // --- place them
        let overflow: u8;
        if dest != FLOOR {
            let room = dest as u8 + 1 - self.pl_count[p][dest];
            self.pl_color[p][dest] = color as i8;
            let masks = &mut self.open_mask[p];
            if count < room {
                self.pl_count[p][dest] += count;
                overflow = 0;
                // _AND_KEEP[color][dest]: close `dest` for every colour but `color`
                let close = !(1u8 << dest) & ALL_ROWS;
                for c2 in 0..NUM_COLORS {
                    if c2 != color {
                        masks[c2] &= close;
                    }
                }
            } else {
                self.pl_count[p][dest] = dest as u8 + 1;
                overflow = count - room;
                let close = !(1u8 << dest) & ALL_ROWS;
                for m in masks.iter_mut() {
                    *m &= close;
                }
            }
        } else {
            overflow = count;
        }

        if overflow != 0 {
            let fl = &mut self.floor[p];
            let mut occupied = fl.iter().map(|&x| x as usize).sum::<usize>();
            if self.floor_marker[p] {
                occupied += 1;
            }
            let room = FLOOR_SLOTS.saturating_sub(occupied) as u8;
            if overflow <= room {
                fl[color] += overflow;
            } else if room > 0 {
                fl[color] += room;
                self.lid[color] += overflow - room;
            } else {
                self.lid[color] += overflow;
            }
        }

        // --- round / game transitions
        if self.tiles_left != 0 {
            self.current_player = 1 - p as u8;
        } else {
            self.end_round(p);
        }
        Ok(())
    }

    // ------------------------------------------------------------ round logic
    fn end_round(&mut self, last_mover: usize) {
        for q in 0..2 {
            let mut wall = self.walls[q];
            let mut gain: i32 = 0;
            for r in 0..NUM_ROWS {
                if self.pl_count[q][r] as usize != r + 1 {
                    continue;
                }
                let c = self.pl_color[q][r] as usize;
                let idx = wall_idx(c, r);
                wall |= 1 << idx;
                let col = idx - r * 5;
                let row_bits = (wall >> (r * 5)) & ALL_ROWS as u32;
                let mut h = 1;
                let mut i = col as i32 - 1;
                while i >= 0 && (row_bits >> i) & 1 == 1 {
                    h += 1;
                    i -= 1;
                }
                let mut i = col + 1;
                while i < 5 && (row_bits >> i) & 1 == 1 {
                    h += 1;
                    i += 1;
                }
                let mut v = 1;
                let mut i = r as i32 - 1;
                while i >= 0 && (wall >> (i as usize * 5 + col)) & 1 == 1 {
                    v += 1;
                    i -= 1;
                }
                let mut i = r + 1;
                while i < 5 && (wall >> (i * 5 + col)) & 1 == 1 {
                    v += 1;
                    i += 1;
                }
                if h > 1 || v > 1 {
                    gain += (if h > 1 { h } else { 0 }) + (if v > 1 { v } else { 0 });
                } else {
                    gain += 1;
                }
                self.lid[c] += r as u8; // the r leftover tiles of the line
                self.pl_color[q][r] = -1;
                self.pl_count[q][r] = 0;
            }
            self.walls[q] = wall;

            let fl = &mut self.floor[q];
            let mut occupied = fl.iter().map(|&x| x as usize).sum::<usize>();
            if self.floor_marker[q] {
                occupied += 1;
            }
            gain += CUM_PENALTY[occupied.min(FLOOR_SLOTS)];
            for c in 0..NUM_COLORS {
                let n = fl[c];
                if n != 0 {
                    self.lid[c] += n;
                    fl[c] = 0;
                }
            }
            let total = self.scores[q] + gain;
            self.scores[q] = total.max(0);
            self.rebuild_mask(q);
        }

        // who starts next round: the marker holder (marker goes back to the center)
        let mut holder: Option<usize> = None;
        for q in 0..2 {
            if self.floor_marker[q] {
                self.floor_marker[q] = false;
                holder = Some(q);
            }
        }
        let holder = holder.unwrap_or(1 - last_mover);
        self.first_player = holder as u8;
        self.marker_in_center = true;
        self.current_player = holder as u8;

        if self.any_row_complete() {
            self.finish();
            return;
        }

        self.round_index += 1;
        self.refill();
        if self.tiles_left == 0 {
            self.exhausted = true;
            self.finish();
        }
    }

    fn any_row_complete(&self) -> bool {
        for wall in self.walls {
            for r in 0..NUM_ROWS {
                if (wall >> (r * 5)) & ALL_ROWS as u32 == ALL_ROWS as u32 {
                    return true;
                }
            }
        }
        false
    }

    fn finish(&mut self) {
        for q in 0..2 {
            self.scores[q] += ROW_BONUS * self.completed_rows(q) as i32
                + COL_BONUS * self.completed_cols(q) as i32
                + COLOR_BONUS * self.completed_colors(q) as i32;
        }
        self.is_terminal = true;
    }

    fn refill(&mut self) {
        let mut total = 0u8;
        for f in 0..NUM_FACTORIES {
            for _ in 0..FACTORY_SIZE {
                if self.bag_len == 0 {
                    // Python: bag.extend([c] * n) in colour order, then shuffle.
                    let mut n_total = 0usize;
                    for c in 0..NUM_COLORS {
                        let n = self.lid[c] as usize;
                        for _ in 0..n {
                            self.bag[n_total] = c as u8;
                            n_total += 1;
                        }
                        self.lid[c] = 0;
                    }
                    self.bag_len = n_total as u8;
                    if self.bag_len == 0 {
                        self.tiles_left = total;
                        return;
                    }
                    let len = self.bag_len as usize;
                    self.rng.shuffle(&mut self.bag[..len]);
                }
                self.bag_len -= 1;
                let c = self.bag[self.bag_len as usize] as usize;
                self.factories[f][c] += 1;
                total += 1;
            }
        }
        self.tiles_left = total;
    }

    // ------------------------------------------------------------ inspection
    pub fn floor_occupied(&self, p: usize) -> usize {
        self.floor[p].iter().map(|&x| x as usize).sum::<usize>() + self.floor_marker[p] as usize
    }

    pub fn floor_penalty(&self, p: usize) -> i32 {
        CUM_PENALTY[self.floor_occupied(p).min(FLOOR_SLOTS)]
    }

    pub fn completed_rows(&self, p: usize) -> u32 {
        let wall = self.walls[p];
        (0..NUM_ROWS)
            .filter(|&r| (wall >> (r * 5)) & ALL_ROWS as u32 == ALL_ROWS as u32)
            .count() as u32
    }

    pub fn completed_cols(&self, p: usize) -> u32 {
        let wall = self.walls[p];
        (0..5)
            .filter(|&col| (0..NUM_ROWS).all(|r| (wall >> (r * 5 + col)) & 1 == 1))
            .count() as u32
    }

    pub fn completed_colors(&self, p: usize) -> u32 {
        let wall = self.walls[p];
        (0..NUM_COLORS)
            .filter(|&c| (0..NUM_ROWS).all(|r| (wall >> wall_idx(c, r)) & 1 == 1))
            .count() as u32
    }

    /// 15 bits: 5 rows, 5 columns, 5 colours (see `AzulState.wall_summary`).
    pub fn wall_summary(&self, p: usize) -> [u8; 15] {
        let wall = self.walls[p];
        let mut out = [0u8; 15];
        for r in 0..NUM_ROWS {
            out[r] = ((wall >> (r * 5)) & ALL_ROWS as u32 == ALL_ROWS as u32) as u8;
        }
        for col in 0..5 {
            out[5 + col] = (0..NUM_ROWS).all(|r| (wall >> (r * 5 + col)) & 1 == 1) as u8;
        }
        for c in 0..NUM_COLORS {
            out[10 + c] = (0..NUM_ROWS).all(|r| (wall >> wall_idx(c, r)) & 1 == 1) as u8;
        }
        out
    }

    /// +1 if player 0 wins, -1 if player 1 wins, 0 for a draw, None if unfinished.
    pub fn outcome(&self) -> Option<f32> {
        if !self.is_terminal {
            return None;
        }
        let (s0, s1) = (self.scores[0], self.scores[1]);
        if s0 != s1 {
            return Some(if s0 > s1 { 1.0 } else { -1.0 });
        }
        let (r0, r1) = (self.completed_rows(0), self.completed_rows(1));
        if r0 != r1 {
            return Some(if r0 > r1 { 1.0 } else { -1.0 });
        }
        Some(0.0)
    }

    pub fn bag_counts(&self) -> [u8; NUM_COLORS] {
        let mut counts = [0u8; NUM_COLORS];
        for &c in self.bag_slice() {
            counts[c as usize] += 1;
        }
        counts
    }

    /// Count all tiles wherever they are. Must always be `[20; 5]`.
    pub fn tile_census(&self) -> [u32; NUM_COLORS] {
        let mut counts = [0u32; NUM_COLORS];
        for (c, &n) in self.bag_counts().iter().enumerate() {
            counts[c] += n as u32;
        }
        for c in 0..NUM_COLORS {
            counts[c] += self.lid[c] as u32 + self.center[c] as u32;
            for f in &self.factories {
                counts[c] += f[c] as u32;
            }
            for p in 0..2 {
                counts[c] += self.floor[p][c] as u32;
            }
        }
        for p in 0..2 {
            for r in 0..NUM_ROWS {
                let n = self.pl_count[p][r];
                if n != 0 {
                    counts[self.pl_color[p][r] as usize] += n as u32;
                }
            }
            let wall = self.walls[p];
            for r in 0..NUM_ROWS {
                for col in 0..5 {
                    if (wall >> (r * 5 + col)) & 1 == 1 {
                        counts[(col + NUM_COLORS - r) % NUM_COLORS] += 1;
                    }
                }
            }
        }
        counts
    }

    #[inline]
    pub fn census_ok(&self) -> bool {
        self.tile_census() == [TILES_PER_COLOR as u32; NUM_COLORS]
    }

    // ---------------------------------------------------- search integration
    /// True iff `action` empties the board and therefore triggers a refill.
    #[inline]
    pub fn is_stochastic(&self, action: u8) -> bool {
        let (src, color, _) = decode_action(action);
        let n = if src == CENTER { self.center[color] } else { self.factories[src][color] };
        n == self.tiles_left
    }

    /// Clone with a fresh bag order (from `seed`), then apply `action`.
    pub fn determinize(&self, action: u8, seed: u64) -> State {
        let mut child = *self;
        child.rng.reseed(seed);
        let len = child.bag_len as usize;
        child.rng.shuffle(&mut child.bag[..len]);
        child.apply(action).expect("determinize: illegal action");
        child
    }

    /// Identity of a post-refill position: factory + center contents.
    pub fn chance_key(&self) -> [u8; 30] {
        let mut out = [0u8; 30];
        for f in 0..NUM_FACTORIES {
            out[f * 5..f * 5 + 5].copy_from_slice(&self.factories[f]);
        }
        out[25..30].copy_from_slice(&self.center);
        out
    }

    /// Cheap near-unique identity of a position (guards a stale tree reuse):
    /// the same fields as `AzulState.fingerprint`, packed.
    pub fn fingerprint(&self) -> Fingerprint {
        let mut pl_color = [[0u8; 5]; 2];
        for p in 0..2 {
            for r in 0..5 {
                pl_color[p][r] = self.pl_color[p][r] as u8;
            }
        }
        Fingerprint {
            current_player: self.current_player,
            round_index: self.round_index,
            tiles_left: self.tiles_left,
            marker_in_center: self.marker_in_center,
            scores: self.scores,
            pl_count: self.pl_count,
            pl_color,
            wall_bits: [self.walls[0].count_ones() as u8, self.walls[1].count_ones() as u8],
            chance_key: self.chance_key(),
        }
    }

    /// Replace the refill the engine just made with the observed deal
    /// (`ludometer.human.convert.apply_deal`). The bag becomes sorted by colour.
    pub fn apply_deal(&mut self, target: &[[u8; NUM_COLORS]; NUM_FACTORIES]) -> Result<(), String> {
        let mut dealt = [0i32; NUM_COLORS];
        for f in target {
            for c in 0..NUM_COLORS {
                dealt[c] += f[c] as i32;
            }
        }
        let bag = self.bag_counts();
        let mut lid = self.lid;
        let mut engine_deal = [0i32; NUM_COLORS];
        for f in &self.factories {
            for c in 0..NUM_COLORS {
                engine_deal[c] += f[c] as i32;
            }
        }
        let mut remaining = [0i32; NUM_COLORS];
        for c in 0..NUM_COLORS {
            let pool = bag[c] as i32 + lid[c] as i32 + engine_deal[c];
            remaining[c] = pool - dealt[c];
            if remaining[c] < 0 {
                return Err(format!(
                    "round {}: deal wants {:?} but only {} of colour {} are off-board",
                    self.round_index, dealt, pool, c
                ));
            }
        }
        let mut new_bag = [0i32; NUM_COLORS];
        let mut merge = false;
        for c in 0..NUM_COLORS {
            new_bag[c] = remaining[c] - lid[c] as i32;
            if new_bag[c] < 0 {
                merge = true;
            }
        }
        if merge {
            new_bag = remaining;
            lid = [0; NUM_COLORS];
        }
        self.factories = *target;
        let mut n = 0usize;
        for c in 0..NUM_COLORS {
            for _ in 0..new_bag[c] {
                self.bag[n] = c as u8;
                n += 1;
            }
        }
        self.bag_len = n as u8;
        self.lid = lid;
        self.recount();
        if !self.census_ok() {
            return Err(format!(
                "round {}: tile census {:?} after deal",
                self.round_index,
                self.tile_census()
            ));
        }
        Ok(())
    }

    // --------------------------------------------------------------- encoding
    /// The 182-float observation from the player to move's perspective. Every
    /// value is computed in f64 then rounded to f32, exactly as numpy does when
    /// a Python float is stored into a float32 array.
    pub fn encode(&self, v: &mut [f32; ENCODED_SIZE]) {
        #[inline]
        fn f(n: f64, d: f64) -> f32 {
            (n / d) as f32
        }
        *v = [0.0; ENCODED_SIZE];
        let me = self.current_player as usize;
        let op = 1 - me;
        for (off, p) in [(OFF_MY_WALL, me), (OFF_OP_WALL, op)] {
            let wall = self.walls[p];
            for i in 0..25 {
                v[off + i] = ((wall >> i) & 1) as f32;
            }
        }
        for (off, p) in [(OFF_MY_LINES, me), (OFF_OP_LINES, op)] {
            for r in 0..NUM_ROWS {
                let n = self.pl_count[p][r];
                if n != 0 {
                    let base = off + r * 6;
                    v[base + self.pl_color[p][r] as usize] = 1.0;
                    v[base + 5] = f(n as f64, (r + 1) as f64);
                }
            }
        }
        for (off, p) in [(OFF_MY_FLOOR, me), (OFF_OP_FLOOR, op)] {
            for c in 0..NUM_COLORS {
                let n = self.floor[p][c];
                if n != 0 {
                    v[off + c] = f(n as f64, FLOOR_SLOTS as f64);
                }
            }
            v[off + 5] = f(self.floor_occupied(p).min(FLOOR_SLOTS) as f64, FLOOR_SLOTS as f64);
            v[off + 6] = if self.floor_marker[p] { 1.0 } else { 0.0 };
        }
        v[OFF_SCORES] = f(self.scores[me] as f64, 100.0);
        v[OFF_SCORES + 1] = f(self.scores[op] as f64, 100.0);
        for (i, fac) in self.factories.iter().enumerate() {
            let base = OFF_FACTORIES + i * 5;
            let mut total = 0u32;
            for c in 0..NUM_COLORS {
                let n = fac[c];
                if n != 0 {
                    v[base + c] = f(n as f64, FACTORY_SIZE as f64);
                    total += n as u32;
                }
            }
            if total != 0 {
                v[OFF_FACTORY_FLAGS + i] = 1.0;
            }
        }
        let mut cen_total = 0u32;
        for c in 0..NUM_COLORS {
            let n = self.center[c];
            if n != 0 {
                v[OFF_CENTER + c] = f(n as f64, 10.0);
                cen_total += n as u32;
            }
        }
        v[OFF_CENTER_TOTAL] = f(cen_total as f64, 20.0);
        v[OFF_MARKER_CENTER] = if self.marker_in_center { 1.0 } else { 0.0 };
        let bag = self.bag_counts();
        for c in 0..NUM_COLORS {
            v[OFF_BAG + c] = f(bag[c] as f64, TILES_PER_COLOR as f64);
            v[OFF_LID + c] = f(self.lid[c] as f64, TILES_PER_COLOR as f64);
        }
        v[OFF_TILES_LEFT] = f(self.tiles_left as f64, 20.0);
        v[OFF_I_START] = if self.floor_marker[me] || self.first_player as usize == me { 1.0 } else { 0.0 };
        v[OFF_ROUND] = f(self.round_index.min(10) as f64, 10.0);
        for (off, p) in [(OFF_MY_SETS, me), (OFF_OP_SETS, op)] {
            v[off] = f(self.completed_rows(p) as f64, 5.0);
            v[off + 1] = f(self.completed_cols(p) as f64, 5.0);
            v[off + 2] = f(self.completed_colors(p) as f64, 5.0);
        }
    }

    pub fn encoded(&self) -> [f32; ENCODED_SIZE] {
        let mut v = [0.0f32; ENCODED_SIZE];
        self.encode(&mut v);
        v
    }

    /// `walls[p]` as 25 ints, row-major (the Python layout).
    pub fn wall_cells(&self, p: usize) -> [u8; 25] {
        let mut out = [0u8; 25];
        for i in 0..25 {
            out[i] = ((self.walls[p] >> i) & 1) as u8;
        }
        out
    }

    pub fn set_wall_cells(&mut self, p: usize, cells: &[u8; 25]) {
        let mut w = 0u32;
        for (i, &c) in cells.iter().enumerate() {
            if c != 0 {
                w |= 1 << i;
            }
        }
        self.walls[p] = w;
    }
}

/// The fields of `AzulState.fingerprint()`, packed into one comparable value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fingerprint {
    pub current_player: u8,
    pub round_index: u16,
    pub tiles_left: u8,
    pub marker_in_center: bool,
    pub scores: [i32; 2],
    pub pl_count: [[u8; 5]; 2],
    pub pl_color: [[u8; 5]; 2],
    pub wall_bits: [u8; 2],
    pub chance_key: [u8; 30],
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_game(seed: u64, kind: RngKind) -> State {
        let mut s = State::new_game(seed, kind);
        let mut g = crate::rng::SplitMix64::new(seed ^ 77);
        let mut legal = Vec::new();
        let mut moves = 0;
        while !s.is_terminal && moves < 400 {
            s.legal_actions_into(&mut legal);
            assert!(!legal.is_empty());
            for &a in &legal {
                assert!(s.is_legal(a as i64), "legal list disagrees with is_legal");
            }
            let a = legal[g.below(legal.len() as u64) as usize];
            s.apply(a).unwrap();
            assert!(s.census_ok(), "census broken after move {moves}");
            moves += 1;
        }
        s
    }

    #[test]
    fn new_game_deals_twenty_tiles() {
        let s = State::new_game(1, RngKind::Fast);
        assert_eq!(s.tiles_left, 20);
        assert_eq!(s.bag_len, 80);
        assert!(s.census_ok());
        assert_eq!(s.legal_actions().len() > 10, true);
    }

    #[test]
    fn random_games_terminate_and_conserve_tiles() {
        for seed in 0..200 {
            let s = random_game(seed, RngKind::Fast);
            assert!(s.is_terminal, "seed {seed} did not finish in 400 moves");
            assert!(s.outcome().is_some());
            assert!(s.scores[0] >= 0 && s.scores[1] >= 0);
        }
        for seed in 0..20 {
            random_game(seed, RngKind::Python);
        }
    }

    #[test]
    fn wall_scoring_counts_adjacent_lines() {
        // Place a tile at (2, 2) with neighbours left and above: h = 2, v = 2.
        let mut s = State::blank(RngKind::Fast);
        let mut cells = [0u8; 25];
        cells[2 * 5 + 1] = 1; // (2,1)
        cells[1 * 5 + 2] = 1; // (1,2)
        s.set_wall_cells(0, &cells);
        // Row 2 colour whose column is 2: (c + 2) % 5 == 2 -> c = 0 (blue).
        s.pl_color[0][2] = 0;
        s.pl_count[0][2] = 3;
        // Put the 3 line tiles + 2 wall tiles into the census by faking a bag.
        s.end_round(1);
        assert_eq!(s.scores[0], 4);
        // The two leftover blue tiles went to the lid, then (the bag being
        // empty) straight back through the lid into the next round's deal.
        assert_eq!(s.lid[0], 0);
        assert_eq!(s.tiles_left, 2);
    }

    #[test]
    fn floor_overflow_goes_to_the_lid() {
        let mut s = State::new_game(3, RngKind::Fast);
        // Fill player 0's floor with 7 tiles by hand, then drop 3 more.
        s.floor[0] = [7, 0, 0, 0, 0];
        // Take colour with count n from a factory to the floor.
        let legal = s.legal_actions();
        let a = *legal.iter().find(|&&a| decode_action(a).2 == FLOOR).unwrap();
        let (src, color, _) = decode_action(a);
        let n = s.factories[src][color];
        let lid_before = s.lid[color];
        s.apply(a).unwrap();
        assert_eq!(s.lid[color], lid_before + n);
    }

    #[test]
    fn determinize_is_a_different_deal_but_a_valid_position() {
        let mut s = State::new_game(9, RngKind::Fast);
        // Empty the board down to one tile so the next move refills.
        for f in 0..NUM_FACTORIES {
            for c in 0..NUM_COLORS {
                s.lid[c] += s.factories[f][c];
                s.factories[f][c] = 0;
            }
        }
        for c in 0..NUM_COLORS {
            s.lid[c] += s.center[c];
            s.center[c] = 0;
        }
        s.lid[0] -= 1;
        s.factories[0][0] = 1;
        s.recount();
        let a = encode_action(0, 0, FLOOR);
        assert!(s.is_stochastic(a));
        let d1 = s.determinize(a, 1);
        let d2 = s.determinize(a, 2);
        assert!(d1.census_ok() && d2.census_ok());
        assert_eq!(d1.round_index, 1);
        assert_ne!(d1.chance_key(), d2.chance_key());
    }
}
