/* Azul rules engine — a line-by-line port of ludometer/azul/engine.py.
 *
 * Official 2-player rules (docs/DESIGN.md). Same action encoding
 * (`action_id = source * 30 + color * 6 + dest`) and, crucially, the same
 * 182-float observation layout, because the ONNX net in ../model/ was trained on
 * exactly those numbers: a single misplaced offset would leave the search
 * evaluating garbage while still looking like it works.
 *
 * The port is proven, not asserted: scripts/dump_fixtures.py plays seeded games
 * with the Python engine and records every state, legal-action list, encoding and
 * outcome; test/engine.test.mjs replays them here and demands an exact match.
 *
 * Chance. The Python engine keeps all randomness in one `random.Random` that is
 * only consumed at a round refill. Here the RNG is an injected object with
 * `shuffle` / `clone`, which is what lets three callers coexist: the page uses a
 * seeded PRNG, MCTS reshuffles a clone's bag to determinize a refill, and the
 * fixture test replays Python's own shuffles so the two engines see identical
 * deals (JS cannot reproduce the Mersenne Twister, and it does not need to —
 * only the rules are under test).
 *
 * Colors are 0..4 = blue, yellow, red, black, teal.
 */

export const NUM_COLORS = 5;
export const TILES_PER_COLOR = 20;
export const NUM_FACTORIES = 5; // 2-player count
export const FACTORY_SIZE = 4;
export const NUM_ROWS = 5;
export const CENTER = 5; // action `source` value for the center
export const FLOOR = 5; // action `dest` value for the floor line
export const ACTION_SPACE = 180; // source (6) * color (5) * dest (6)
export const ENCODED_SIZE = 182;

export const FLOOR_PENALTIES = [-1, -1, -2, -2, -2, -3, -3];
export const FLOOR_SLOTS = FLOOR_PENALTIES.length;
export const CUM_PENALTY = (() => {
  const out = [0];
  for (const p of FLOOR_PENALTIES) out.push(out[out.length - 1] + p);
  return out;
})();

export const COLOR_NAMES = ["blue", "yellow", "red", "black", "teal"];

export const ROW_BONUS = 2;
export const COL_BONUS = 7;
export const COLOR_BONUS = 10;

/** Flat wall lookup: WALL_IDX[color * 5 + row] -> index into the 25-cell wall. */
export const WALL_IDX = (() => {
  const out = new Int32Array(NUM_COLORS * NUM_ROWS);
  for (let c = 0; c < NUM_COLORS; c++) {
    for (let r = 0; r < NUM_ROWS; r++) out[c * 5 + r] = r * 5 + ((c + r) % 5);
  }
  return out;
})();

export function wallCol(color, row) {
  return (color + row) % NUM_COLORS;
}

export function encodeAction(source, color, dest) {
  return source * 30 + color * 6 + dest;
}

export function decodeAction(actionId) {
  const source = Math.floor(actionId / 30);
  const rest = actionId - source * 30;
  const color = Math.floor(rest / 6);
  return [source, color, rest - color * 6];
}

/* ------------------------------------------------------------ lookup tables */
/* _ACTION_TABLE[source][color][open_mask] -> the ready-made action ids (pattern
 * rows in ascending order, floor last), mirroring the Python hot path so the
 * legal-action *order* matches too, not just the set. */
const _ALL_ROWS = (1 << NUM_ROWS) - 1;
const _ACTION_TABLE = [];
for (let src = 0; src < 6; src++) {
  const perSource = [];
  for (let c = 0; c < NUM_COLORS; c++) {
    const perColor = [];
    for (let mask = 0; mask < 1 << NUM_ROWS; mask++) {
      const ids = [];
      for (let r = 0; r < NUM_ROWS; r++) if ((mask >> r) & 1) ids.push(src * 30 + c * 6 + r);
      ids.push(src * 30 + c * 6 + FLOOR);
      perColor.push(ids);
    }
    perSource.push(perColor);
  }
  _ACTION_TABLE.push(perSource);
}
/* _AND_KEEP[color][row]: AND-masks closing `row` for every color but `color`. */
const _AND_KEEP = [];
for (let c = 0; c < NUM_COLORS; c++) {
  const perColor = [];
  for (let r = 0; r < NUM_ROWS; r++) {
    const masks = [];
    for (let c2 = 0; c2 < NUM_COLORS; c2++) masks.push(c2 === c ? _ALL_ROWS : _ALL_ROWS ^ (1 << r));
    perColor.push(masks);
  }
  _AND_KEEP.push(perColor);
}
/* _AND_CLOSE[row]: closes `row` for every color (the line is full / tiled). */
const _AND_CLOSE = [];
for (let r = 0; r < NUM_ROWS; r++) {
  const masks = [];
  for (let c = 0; c < NUM_COLORS; c++) masks.push(_ALL_ROWS ^ (1 << r));
  _AND_CLOSE.push(masks);
}

/* --------------------------------------------------------------------- RNG */
/**
 * Small seeded PRNG (mulberry32) with the bits the engine needs.
 *
 * It is *not* Python's Mersenne Twister — the two engines deal different tiles
 * from the same seed, which is fine: reproducibility is per-engine, and the
 * fixture test hands the JS side Python's recorded shuffles instead.
 */
export class Rng {
  constructor(seed = 0) {
    this.seed(seed);
  }

  seed(n) {
    this.state = (Number(n) >>> 0) || 0x9e3779b9;
    return this;
  }

  /** uint32 */
  next() {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  }

  /** float in [0, 1) */
  random() {
    return this.next() / 4294967296;
  }

  /** int in [0, n) */
  randrange(n) {
    return this.next() % n;
  }

  /** Fisher-Yates, in place (same loop shape as random.shuffle). */
  shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.randrange(i + 1);
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return arr;
  }

  clone() {
    const other = new Rng(0);
    other.state = this.state;
    return other;
  }
}

/**
 * An RNG that replays a recorded list of shuffle *results* (fixtures only).
 *
 * `shuffle(bag)` overwrites the bag with the next recorded ordering, so the JS
 * engine deals exactly what Python dealt without owning Python's RNG.
 */
export class ScriptedRng {
  constructor(shuffles) {
    this.shuffles = shuffles;
    this.index = 0;
  }

  seed() {
    return this;
  }

  shuffle(arr) {
    if (this.index >= this.shuffles.length) {
      throw new Error(`ScriptedRng ran out of recorded shuffles (${this.index})`);
    }
    const next = this.shuffles[this.index++];
    if (next.length !== arr.length) {
      throw new Error(`recorded shuffle #${this.index - 1} has ${next.length} tiles, bag has ${arr.length}`);
    }
    for (let i = 0; i < next.length; i++) arr[i] = next[i];
    return arr;
  }

  clone() {
    const other = new ScriptedRng(this.shuffles);
    other.index = this.index;
    return other;
  }
}

/* ------------------------------------------------------------------- state */
export class AzulState {
  /** Deal a new 2-player game. `rng` defaults to a fresh seeded `Rng`. */
  static newGame(seed = 0, rng = null) {
    const self = new AzulState();
    self.numPlayers = 2;
    self.rng = rng || new Rng(seed);
    self.bag = [];
    for (let c = 0; c < NUM_COLORS; c++) {
      for (let i = 0; i < TILES_PER_COLOR; i++) self.bag.push(c);
    }
    self.rng.shuffle(self.bag);
    self.lid = [0, 0, 0, 0, 0];
    self.factories = [];
    for (let i = 0; i < NUM_FACTORIES; i++) self.factories.push([0, 0, 0, 0, 0]);
    self.center = [0, 0, 0, 0, 0];
    self.markerInCenter = true;
    self.walls = [new Array(25).fill(0), new Array(25).fill(0)];
    self.plColor = [new Array(NUM_ROWS).fill(-1), new Array(NUM_ROWS).fill(-1)];
    self.plCount = [new Array(NUM_ROWS).fill(0), new Array(NUM_ROWS).fill(0)];
    self.floor = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]];
    self.floorMarker = [false, false];
    self.openMask = [new Array(NUM_COLORS).fill(_ALL_ROWS), new Array(NUM_COLORS).fill(_ALL_ROWS)];
    self.scores = [0, 0];
    self.currentPlayer = 0;
    self.firstPlayer = 0;
    self.roundIndex = 0;
    self.isTerminal = false;
    self.exhausted = false;
    self.tilesLeft = 0;
    self._refill();
    return self;
  }

  /**
   * Build a state from an explicit field dump (fixtures and tests only).
   *
   * The Python engine lets tests hand-edit `factories` / `center` / `pl_*` /
   * `walls` and then call `recount()`; this is the same door, so a fixture can
   * put the engine in a position random play would essentially never reach
   * (an all-monochrome round end, a bag that runs dry, a wall one tile short).
   */
  static fromSetup(setup, rng = null) {
    const self = new AzulState();
    self.numPlayers = 2;
    self.rng = rng || new Rng(0);
    self.bag = setup.bag.slice();
    self.lid = setup.lid.slice();
    self.factories = setup.factories.map((f) => f.slice());
    self.center = setup.center.slice();
    self.markerInCenter = setup.marker_in_center;
    self.walls = setup.walls.map((w) => w.slice());
    self.plColor = setup.pl_color.map((x) => x.slice());
    self.plCount = setup.pl_count.map((x) => x.slice());
    self.floor = setup.floor.map((f) => f.slice());
    self.floorMarker = setup.floor_marker.slice();
    self.openMask = [new Array(NUM_COLORS).fill(_ALL_ROWS), new Array(NUM_COLORS).fill(_ALL_ROWS)];
    self.scores = setup.scores.slice();
    self.currentPlayer = setup.current_player;
    self.firstPlayer = setup.first_player;
    self.roundIndex = setup.round_index;
    self.isTerminal = Boolean(setup.is_terminal);
    self.exhausted = Boolean(setup.exhausted);
    self.tilesLeft = 0;
    self.recount();
    return self;
  }

  /** The inverse of `fromSetup`: a structured-clone-safe field dump. */
  toSetup() {
    return {
      bag: this.bag.slice(),
      lid: this.lid.slice(),
      factories: this.factories.map((f) => f.slice()),
      center: this.center.slice(),
      marker_in_center: this.markerInCenter,
      walls: this.walls.map((w) => w.slice()),
      pl_color: this.plColor.map((x) => x.slice()),
      pl_count: this.plCount.map((x) => x.slice()),
      floor: this.floor.map((f) => f.slice()),
      floor_marker: this.floorMarker.slice(),
      scores: this.scores.slice(),
      current_player: this.currentPlayer,
      first_player: this.firstPlayer,
      round_index: this.roundIndex,
      is_terminal: this.isTerminal,
      exhausted: this.exhausted,
    };
  }

  /** Rebuild the derived caches (`tilesLeft` and the placement masks). */
  recount() {
    let total = this.center[0] + this.center[1] + this.center[2] + this.center[3] + this.center[4];
    for (const f of this.factories) total += f[0] + f[1] + f[2] + f[3] + f[4];
    this.tilesLeft = total;
    for (let p = 0; p < this.numPlayers; p++) this._rebuildMask(p);
  }

  /** Deep-enough copy: every mutable container is duplicated. */
  clone() {
    const other = new AzulState();
    other.numPlayers = this.numPlayers;
    other.rng = this.rng.clone();
    other.bag = this.bag.slice();
    other.lid = this.lid.slice();
    other.factories = this.factories.map((f) => f.slice());
    other.center = this.center.slice();
    other.markerInCenter = this.markerInCenter;
    other.walls = this.walls.map((w) => w.slice());
    other.plColor = this.plColor.map((x) => x.slice());
    other.plCount = this.plCount.map((x) => x.slice());
    other.floor = this.floor.map((f) => f.slice());
    other.floorMarker = this.floorMarker.slice();
    other.openMask = this.openMask.map((m) => m.slice());
    other.scores = this.scores.slice();
    other.currentPlayer = this.currentPlayer;
    other.firstPlayer = this.firstPlayer;
    other.roundIndex = this.roundIndex;
    other.isTerminal = this.isTerminal;
    other.exhausted = this.exhausted;
    other.tilesLeft = this.tilesLeft;
    return other;
  }

  _rebuildMask(player) {
    const wall = this.walls[player];
    const plc = this.plColor[player];
    const pln = this.plCount[player];
    const masks = this.openMask[player];
    for (let c = 0; c < NUM_COLORS; c++) {
      const base = c * 5;
      let m = 0;
      for (let r = 0; r < NUM_ROWS; r++) {
        const n = pln[r];
        if (n <= r && (n === 0 || plc[r] === c) && !wall[WALL_IDX[base + r]]) m |= 1 << r;
      }
      masks[c] = m;
    }
  }

  /* ---------------------------------------------------------- legal moves */
  legalActions() {
    if (this.isTerminal) return [];
    const masks = this.openMask[this.currentPlayer];
    const out = [];
    for (let src = 0; src < NUM_FACTORIES; src++) {
      const pool = this.factories[src];
      const table = _ACTION_TABLE[src];
      for (let c = 0; c < NUM_COLORS; c++) {
        if (pool[c]) {
          const ids = table[c][masks[c]];
          for (let i = 0; i < ids.length; i++) out.push(ids[i]);
        }
      }
    }
    const pool = this.center;
    const table = _ACTION_TABLE[CENTER];
    for (let c = 0; c < NUM_COLORS; c++) {
      if (pool[c]) {
        const ids = table[c][masks[c]];
        for (let i = 0; i < ids.length; i++) out.push(ids[i]);
      }
    }
    return out;
  }

  isLegal(actionId) {
    if (this.isTerminal || !(actionId >= 0 && actionId < ACTION_SPACE)) return false;
    const [src, color, dest] = decodeAction(actionId);
    const pool = src === CENTER ? this.center : this.factories[src];
    if (pool[color] === 0) return false;
    if (dest === FLOOR) return true;
    const p = this.currentPlayer;
    const n = this.plCount[p][dest];
    if (n > dest) return false;
    if (n && this.plColor[p][dest] !== color) return false;
    return !this.walls[p][WALL_IDX[color * 5 + dest]];
  }

  /* ---------------------------------------------------------------- moves */
  /** Play `actionId`, then resolve round end / refill / game end as needed. */
  apply(actionId) {
    if (this.isTerminal) throw new Error("game is over");
    if (!(actionId >= 0 && actionId < ACTION_SPACE)) throw new Error(`action ${actionId} out of range`);
    const src = Math.floor(actionId / 30);
    const rest = actionId - src * 30;
    const color = Math.floor(rest / 6);
    const dest = rest - color * 6;

    const p = this.currentPlayer;
    const pool = src === CENTER ? this.center : this.factories[src];
    const count = pool[color];
    if (count === 0) throw new Error(`no color ${color} at source ${src}`);

    if (dest !== FLOOR) {
      const held = this.plCount[p][dest];
      if (held > dest) throw new Error(`pattern line ${dest} is full`);
      if (held && this.plColor[p][dest] !== color) throw new Error(`pattern line ${dest} holds another color`);
      if (this.walls[p][WALL_IDX[color * 5 + dest]]) throw new Error(`color ${color} already on wall row ${dest}`);
    }

    // --- take the tiles
    pool[color] = 0;
    if (src === CENTER) {
      if (this.markerInCenter) {
        this.markerInCenter = false;
        this.floorMarker[p] = true;
      }
    } else {
      const cen = this.center;
      for (let c = 0; c < NUM_COLORS; c++) {
        const n = pool[c];
        if (n) {
          cen[c] += n;
          pool[c] = 0;
        }
      }
    }
    this.tilesLeft -= count;

    // --- place them
    let overflow;
    if (dest !== FLOOR) {
      const pln = this.plCount[p];
      const room = dest + 1 - pln[dest];
      this.plColor[p][dest] = color;
      let keep;
      if (count < room) {
        pln[dest] += count;
        overflow = 0;
        keep = _AND_KEEP[color][dest];
      } else {
        pln[dest] = dest + 1;
        overflow = count - room;
        keep = _AND_CLOSE[dest];
      }
      const masks = this.openMask[p];
      masks[0] &= keep[0];
      masks[1] &= keep[1];
      masks[2] &= keep[2];
      masks[3] &= keep[3];
      masks[4] &= keep[4];
    } else {
      overflow = count;
    }

    if (overflow) {
      const fl = this.floor[p];
      let occupied = fl[0] + fl[1] + fl[2] + fl[3] + fl[4];
      if (this.floorMarker[p]) occupied += 1;
      const room = FLOOR_SLOTS - occupied;
      if (overflow <= room) {
        fl[color] += overflow;
      } else if (room > 0) {
        fl[color] += room;
        this.lid[color] += overflow - room;
      } else {
        this.lid[color] += overflow;
      }
    }

    // --- round / game transitions
    if (this.tilesLeft) this.currentPlayer = 1 - p;
    else this._endRound(p);
  }

  /* --------------------------------------------------------- round logic */
  _endRound(lastMover) {
    const lid = this.lid;
    for (let q = 0; q < this.numPlayers; q++) {
      const wall = this.walls[q];
      const plc = this.plColor[q];
      const pln = this.plCount[q];
      let gain = 0;
      for (let r = 0; r < NUM_ROWS; r++) {
        if (pln[r] !== r + 1) continue;
        const c = plc[r];
        const idx = WALL_IDX[c * 5 + r];
        wall[idx] = 1;
        const rowBase = r * 5;
        const col = idx - rowBase;
        let h = 1;
        for (let i = col - 1; i >= 0 && wall[rowBase + i]; i--) h += 1;
        for (let i = col + 1; i < 5 && wall[rowBase + i]; i++) h += 1;
        let v = 1;
        for (let i = r - 1; i >= 0 && wall[i * 5 + col]; i--) v += 1;
        for (let i = r + 1; i < 5 && wall[i * 5 + col]; i++) v += 1;
        if (h > 1 || v > 1) gain += (h > 1 ? h : 0) + (v > 1 ? v : 0);
        else gain += 1;
        lid[c] += r; // the r leftover tiles of the line
        plc[r] = -1;
        pln[r] = 0;
      }

      const fl = this.floor[q];
      let occupied = fl[0] + fl[1] + fl[2] + fl[3] + fl[4];
      if (this.floorMarker[q]) occupied += 1;
      gain += CUM_PENALTY[Math.min(FLOOR_SLOTS, occupied)];
      for (let c = 0; c < NUM_COLORS; c++) {
        const n = fl[c];
        if (n) {
          lid[c] += n;
          fl[c] = 0;
        }
      }
      const total = this.scores[q] + gain;
      this.scores[q] = Math.max(0, total);
      this._rebuildMask(q);
    }

    // who starts next round: the marker holder (marker goes back to the center)
    let holder = null;
    for (let q = 0; q < this.numPlayers; q++) {
      if (this.floorMarker[q]) {
        this.floorMarker[q] = false;
        holder = q;
      }
    }
    if (holder === null) holder = 1 - lastMover;
    this.firstPlayer = holder;
    this.markerInCenter = true;
    this.currentPlayer = holder;

    if (this._anyRowComplete()) {
      this._finish();
      return;
    }

    this.roundIndex += 1;
    this._refill();
    if (this.tilesLeft === 0) {
      // No tiles anywhere: cannot deal another round, stop the game.
      this.exhausted = true;
      this._finish();
    }
  }

  _anyRowComplete() {
    for (const wall of this.walls) {
      for (const base of [0, 5, 10, 15, 20]) {
        if (wall[base] && wall[base + 1] && wall[base + 2] && wall[base + 3] && wall[base + 4]) return true;
      }
    }
    return false;
  }

  _finish() {
    for (let q = 0; q < this.numPlayers; q++) {
      this.scores[q] +=
        ROW_BONUS * this.completedRows(q) + COL_BONUS * this.completedCols(q) + COLOR_BONUS * this.completedColors(q);
    }
    this.isTerminal = true;
  }

  _refill() {
    const bag = this.bag;
    const lid = this.lid;
    let total = 0;
    for (const f of this.factories) {
      for (let k = 0; k < FACTORY_SIZE; k++) {
        if (!bag.length) {
          for (let c = 0; c < NUM_COLORS; c++) {
            const n = lid[c];
            if (n) {
              for (let i = 0; i < n; i++) bag.push(c);
              lid[c] = 0;
            }
          }
          if (!bag.length) {
            this.tilesLeft = total;
            return;
          }
          this.rng.shuffle(bag);
        }
        f[bag.pop()] += 1;
        total += 1;
      }
    }
    this.tilesLeft = total;
  }

  /* ---------------------------------------------------------- inspection */
  floorOccupied(player) {
    const fl = this.floor[player];
    return fl[0] + fl[1] + fl[2] + fl[3] + fl[4] + (this.floorMarker[player] ? 1 : 0);
  }

  floorPenalty(player) {
    return CUM_PENALTY[Math.min(FLOOR_SLOTS, this.floorOccupied(player))];
  }

  completedRows(player) {
    const wall = this.walls[player];
    let n = 0;
    for (const base of [0, 5, 10, 15, 20]) {
      if (wall[base] && wall[base + 1] && wall[base + 2] && wall[base + 3] && wall[base + 4]) n += 1;
    }
    return n;
  }

  completedCols(player) {
    const wall = this.walls[player];
    let n = 0;
    for (let col = 0; col < 5; col++) {
      if (wall[col] && wall[col + 5] && wall[col + 10] && wall[col + 15] && wall[col + 20]) n += 1;
    }
    return n;
  }

  completedColors(player) {
    const wall = this.walls[player];
    let done = 0;
    for (let c = 0; c < NUM_COLORS; c++) {
      const base = c * 5;
      let all = true;
      for (let r = 0; r < NUM_ROWS; r++) {
        if (!wall[WALL_IDX[base + r]]) {
          all = false;
          break;
        }
      }
      if (all) done += 1;
    }
    return done;
  }

  /** +1 if player 0 wins, -1 if player 1 wins, 0 for a draw, null if unfinished. */
  outcome() {
    if (!this.isTerminal) return null;
    const s0 = this.scores[0];
    const s1 = this.scores[1];
    if (s0 !== s1) return s0 > s1 ? 1.0 : -1.0;
    const r0 = this.completedRows(0);
    const r1 = this.completedRows(1);
    if (r0 !== r1) return r0 > r1 ? 1.0 : -1.0;
    return 0.0;
  }

  bagCounts() {
    const counts = [0, 0, 0, 0, 0];
    for (const c of this.bag) counts[c] += 1;
    return counts;
  }

  /* ------------------------------------------------------------ encoding */
  /**
   * Fixed-size float32 observation from the current player's perspective.
   *
   * The offsets below are the OFF_* constants of ludometer/azul/engine.py and
   * must never drift from them — the exported net reads this vector verbatim.
   *
   *   [  0:  25)  my wall, row-major 5x5           [126: 151)  factories /4
   *   [ 25:  50)  their wall                       [151: 156)  factory non-empty
   *   [ 50:  80)  my pattern lines (one-hot, fill)  [156: 161)  center counts /10
   *   [ 80: 110)  their pattern lines               [161: 162)  center total /20
   *   [110: 117)  my floor (counts /7, slots, mark) [162: 163)  marker in center
   *   [117: 124)  their floor                       [163: 168)  bag counts /20
   *   [124: 126)  scores /100                       [168: 173)  lid counts /20
   *                                                 [173: 174)  tiles left /20
   *                                                 [174: 175)  I start next round
   *                                                 [175: 176)  round /10
   *                                                 [176: 179)  my rows/cols/colors /5
   *                                                 [179: 182)  theirs
   */
  encode(out = null) {
    const v = out || new Float32Array(ENCODED_SIZE);
    if (out) v.fill(0);
    const me = this.currentPlayer;
    const op = 1 - me;

    const myWall = this.walls[me];
    const opWall = this.walls[op];
    for (let i = 0; i < 25; i++) {
      v[i] = myWall[i];
      v[25 + i] = opWall[i];
    }

    for (const [off, p] of [[50, me], [80, op]]) {
      const plc = this.plColor[p];
      const pln = this.plCount[p];
      for (let r = 0; r < NUM_ROWS; r++) {
        const n = pln[r];
        if (n) {
          const base = off + r * 6;
          v[base + plc[r]] = 1.0;
          v[base + 5] = n / (r + 1);
        }
      }
    }

    for (const [off, p] of [[110, me], [117, op]]) {
      const fl = this.floor[p];
      for (let c = 0; c < NUM_COLORS; c++) {
        if (fl[c]) v[off + c] = fl[c] / FLOOR_SLOTS;
      }
      v[off + 5] = Math.min(this.floorOccupied(p), FLOOR_SLOTS) / FLOOR_SLOTS;
      v[off + 6] = this.floorMarker[p] ? 1.0 : 0.0;
    }

    v[124] = this.scores[me] / 100.0;
    v[125] = this.scores[op] / 100.0;

    for (let i = 0; i < NUM_FACTORIES; i++) {
      const f = this.factories[i];
      const base = 126 + i * 5;
      let total = 0;
      for (let c = 0; c < NUM_COLORS; c++) {
        const n = f[c];
        if (n) {
          v[base + c] = n / FACTORY_SIZE;
          total += n;
        }
      }
      if (total) v[151 + i] = 1.0;
    }

    let cenTotal = 0;
    for (let c = 0; c < NUM_COLORS; c++) {
      const n = this.center[c];
      if (n) {
        v[156 + c] = n / 10.0;
        cenTotal += n;
      }
    }
    v[161] = cenTotal / 20.0;
    v[162] = this.markerInCenter ? 1.0 : 0.0;

    const bag = this.bagCounts();
    for (let c = 0; c < NUM_COLORS; c++) {
      v[163 + c] = bag[c] / TILES_PER_COLOR;
      v[168 + c] = this.lid[c] / TILES_PER_COLOR;
    }

    v[173] = this.tilesLeft / 20.0;
    v[174] = this.floorMarker[me] || this.firstPlayer === me ? 1.0 : 0.0;
    v[175] = Math.min(this.roundIndex, 10) / 10.0;

    for (const [off, p] of [[176, me], [179, op]]) {
      v[off] = this.completedRows(p) / 5.0;
      v[off + 1] = this.completedCols(p) / 5.0;
      v[off + 2] = this.completedColors(p) / 5.0;
    }
    return v;
  }

  /* -------------------------------------------------------------- display */
  /** Full state as plain data — the shape the old server's /api/state sent. */
  toJSON() {
    const players = [];
    for (let p = 0; p < this.numPlayers; p++) {
      const wall = [];
      for (let r = 0; r < NUM_ROWS; r++) wall.push(this.walls[p].slice(r * 5, r * 5 + 5));
      const patternLines = [];
      for (let r = 0; r < NUM_ROWS; r++) {
        patternLines.push({ capacity: r + 1, color: this.plColor[p][r], count: this.plCount[p][r] });
      }
      players.push({
        score: this.scores[p],
        wall,
        pattern_lines: patternLines,
        floor: this.floor[p].slice(),
        floor_marker: this.floorMarker[p],
        floor_penalty: this.floorPenalty(p),
        completed_rows: this.completedRows(p),
        completed_cols: this.completedCols(p),
        completed_colors: this.completedColors(p),
      });
    }
    return {
      round: this.roundIndex,
      current_player: this.currentPlayer,
      first_player: this.firstPlayer,
      factories: this.factories.map((f) => f.slice()),
      center: this.center.slice(),
      marker_in_center: this.markerInCenter,
      bag: this.bagCounts(),
      lid: this.lid.slice(),
      tiles_left: this.tilesLeft,
      scores: this.scores.slice(),
      is_terminal: this.isTerminal,
      exhausted: this.exhausted,
      outcome: this.outcome(),
      legal_actions: this.legalActions(),
      color_names: COLOR_NAMES.slice(),
      players,
    };
  }
}
