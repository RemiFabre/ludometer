/* PUCT MCTS over cloned Azul states — a port of ludometer/train/mcts.py.
 *
 * Same search as the trained agent plays with, down to the constants: edge stats
 * `(N, W, P)` live in the parent, a simulation descends by maximising
 * `Q + c_puct * P * sqrt(Nparent + 1) / (1 + N)`, expands one leaf, evaluates it
 * with the net and backs the value up.
 *
 * Perspectives. Azul does not strictly alternate — the holder of the first-player
 * marker starts the next round, so the same player can move twice in a row across
 * a round boundary. Values are therefore propagated in *player 0's* frame (`v0`)
 * and converted per node from `node.player`; nothing here assumes alternation.
 *
 * Chance nodes. Inside a round Azul is deterministic; the only chance event is the
 * refill triggered by the move that takes the last tiles. Those edges are handled
 * exactly as Python does — by re-sampled determinizations: clone the parent,
 * reseed and **reshuffle the clone's bag**, apply the move, and file the result in
 * a per-edge table keyed by the post-refill factory/center contents. Reshuffling
 * is what stops the search peeking at the pre-shuffled deal (bag *counts* are
 * public, the order is not). Past `chanceChildren` distinct outcomes a traversal
 * reuses a stored one at random, so the subtree stays deep while the edge's Q
 * remains an average over sampled refills.
 *
 * The one real difference from Python: evaluation is asynchronous, because
 * onnxruntime-web's `run` is. Each simulation therefore awaits at most one
 * inference (batch of 1), and the loop yields to the event loop periodically so a
 * worker can still answer messages. Dirichlet noise is not ported — it only ever
 * applies to self-play training, never to a game against a human.
 *
 * Batching (`config.batch > 1`). A profile of a 5 s think said 92 % of it is
 * spent inside one `session.run` per simulation — the search itself, cloning and
 * legal-move generation included, is 2 %. The only way to buy more positions is
 * therefore to put more of them in each forward pass, which is what `_collect`
 * does: descend `batch` times, each descent laying down a **virtual loss** on the
 * edges it walks so the next one is pushed elsewhere, evaluate the whole leaf set
 * in one dispatch, then back the real values up and take the virtual loss off
 * again. Because the virtual loss is added and removed in the same units it is
 * exact bookkeeping: a finished batch leaves the tree in the state the same
 * sequence of leaf evaluations would have left it in sequentially. `batch = 1`
 * takes the original code path, untouched.
 *
 * The margin head (`config.margin`). If the loaded net has a third output the
 * search also averages it up the tree, in player 0's frame like the value, and
 * the *root* move is then chosen lexicographically: among the moves whose win-Q
 * is within `marginEpsilon` of the best, play the one with the largest expected
 * score gap. A two-output net has no margin at all and keeps the old rule
 * (most-visited child), bit for bit.
 */

import { ACTION_SPACE, CENTER, Rng } from "./engine.js";

/** Matches configs/run2.json, which is what the exported checkpoint trained with. */
export const DEFAULT_CONFIG = {
  // An upper bound only: with a time budget the clock is meant to bind first, so
  // this sits well above what the fastest machine reaches in the longest budget
  // (~1k sims/s per second of budget in the opening, several times that once the
  // tree starts hitting terminal positions).
  sims: 200000,
  cPuct: 1.4,
  chanceChildren: 4,
  fpu: 0.0, // Q assumed for an unvisited edge, in the parent's frame
  /* Leaves gathered per forward pass. 1 is the original one-at-a-time search;
   * the worker raises it to what the active backend actually likes (measured:
   * 16 for WASM, 64 for WebGPU). This is a *ceiling* — see `batchRamp`. */
  batch: 1,
  /* The batch may not exceed the tree divided by this.
   *
   * Measured the hard way. A flat batch of 64 from the first simulation loses
   * 3–17 to the same search at batch 1 over the same number of simulations:
   * with nothing in the tree yet, virtual loss pushes all 64 descents down 64
   * different early branches, and a 800-simulation search never recovers from
   * spending its first eighth that way. The damage is entirely a function of
   * batch ÷ tree, so the batch starts small and grows with the tree, and by the
   * time it is at its ceiling the tree is 16× bigger than it. */
  batchRamp: 16,
  /* …but never below this, because on a GPU a batch of one is not a smaller
   * step, it is a wasted 4 ms dispatch. Raised with `batch` by the worker. */
  minBatch: 1,
  /* Discouragement applied to an edge already on a pending descent, in the same
   * [-1, 1] units as the value. 1.0 = "assume it loses" is the usual choice and
   * is what keeps a batch from collecting the same leaf `batch` times. */
  virtualLoss: 1.0,
  /* Root moves whose win-Q is within this of the best are considered equally
   * good, and the margin head breaks the tie. Ignored without a margin head. */
  marginEpsilon: 0.03,
  /* A child must have at least this share of the most-visited child's visits to
   * be a candidate for that tie-break — a one-visit edge can carry a Q of +1 by
   * luck, and that must not be allowed to define "the best". */
  marginMinVisitShare: 0.1,
};

/* Two arg-max players can loop forever in a game where no pattern line is ever
 * completed (see mcts.py STALL_ROUNDS): past this many rounds the caller
 * re-introduces sampling, and MAX_GAME_MOVES is the hard backstop. */
export const STALL_ROUNDS = 16;
export const MAX_GAME_MOVES = 400;

/** How often the search hands the event loop back (keeps a worker responsive). */
const YIELD_EVERY = 64;

const nowMs = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
const yieldToLoop = () => new Promise((r) => setTimeout(r, 0));

class Node {
  constructor(state) {
    this.state = state;
    this.player = state.currentPlayer;
    this.legal = state.legalActions();
    this.priors = null;
    this.visits = null;
    this.wins = null;
    this.margins = null; // parallel to `wins`, only allocated with a margin head
    this.children = null; // Node | Map<string, Node> | null, per edge
    this.expanded = false;
    this.pending = false; // already queued for evaluation in the current batch
    this.nVisits = 0;
    this.terminalV0 = state.isTerminal ? state.outcome() || 0 : 0;
    // The margin of a finished game is the real score gap, in player 0's frame,
    // on the same scale the head is trained on (points).
    this.terminalMargin0 = state.isTerminal ? state.scores[0] - state.scores[1] : 0;
  }

  get isTerminal() {
    return this.state.isTerminal;
  }

  initEdges(priors, withMargin = false) {
    const n = this.legal.length;
    this.priors = priors;
    this.visits = new Int32Array(n);
    this.wins = new Float64Array(n);
    if (withMargin) this.margins = new Float64Array(n);
    this.children = new Array(n).fill(null);
    this.expanded = true;
  }
}

export class MCTS {
  /**
   * @param {{evaluate(state, legal): Promise<{priors, value}>}} evaluator
   * @param {object} config  overrides for DEFAULT_CONFIG
   * @param {import("./engine.js").Rng} rng  the search's own RNG
   */
  constructor(evaluator, config = {}, rng = null) {
    this.evaluator = evaluator;
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.rng = rng || new Rng(1);
    this.counter = 0;
    this.nodesCreated = 0;
    this.evals = 0;
    this.batches = 0;
    // A margin head is a property of the loaded net, so it is read off the
    // evaluator rather than configured — `config.margin` can still force it off.
    this.hasMargin =
      config.margin === false ? false : !!(evaluator && evaluator.hasMargin);
  }

  /**
   * Run simulations from `state` (never mutated) and return the visit policy.
   *
   * @param {AzulState} state
   * @param {object} opts
   * @param {number} opts.timeLimitS  wall-clock budget; 0/undefined means "run `sims`"
   * @param {function} opts.onProgress  called with {sims, elapsedS} every so often
   * @param {function} opts.shouldStop  polled; return true to abandon the search
   * @returns {Promise<{policy, value, visits, sims, elapsedS, best}>}
   */
  async search(state, opts = {}) {
    const { timeLimitS = 0, onProgress = null, shouldStop = null } = opts;
    const root = this._newNode(state.clone());
    // kept so a caller can read the root's edge statistics back out afterwards —
    // that table is all coach mode is (see rootChildren)
    this._root = root;
    if (root.isTerminal) throw new Error("cannot search a terminal state");
    const started = nowMs();

    const value = await this._expand(root);
    if (root.legal.length === 1) {
      const policy = new Float32Array(ACTION_SPACE);
      policy[root.legal[0]] = 1;
      return {
        policy,
        value,
        visits: new Map([[root.legal[0], 1]]),
        sims: 0,
        elapsedS: 0,
        best: root.legal[0],
        forced: true,
      };
    }

    const cap = this.config.sims;
    const batch = Math.max(1, this.config.batch | 0);
    const minBatch = Math.max(1, Math.min(batch, this.config.minBatch | 0));
    const ramp = Math.max(1, this.config.batchRamp | 0);
    const deadline = timeLimitS > 0 ? started + timeLimitS * 1000 : Infinity;
    let done = 0;
    let sinceYield = 0;
    while (done < cap) {
      // As big as the tree can afford, never bigger than the backend wants.
      const want = Math.max(minBatch, Math.min(batch, Math.floor(root.nVisits / ramp)));
      const step =
        want === 1
          ? ((await this._simulate(root)), 1)
          : await this._simulateBatch(root, Math.min(want, cap - done));
      // A batch that collected nothing (everything below the root is pending or
      // terminal) would spin; stop rather than burn the clock.
      if (step <= 0) break;
      done += step;
      sinceYield += step;
      if (sinceYield >= YIELD_EVERY) {
        sinceYield = 0;
        if (onProgress) onProgress({ sims: done, elapsedS: (nowMs() - started) / 1000 });
        await yieldToLoop();
        if (shouldStop && shouldStop()) break;
      }
      if (nowMs() >= deadline) break;
    }
    const elapsedS = (nowMs() - started) / 1000;

    const total = root.nVisits;
    const policy = new Float32Array(ACTION_SPACE);
    const visits = new Map();
    let bestN = -1;
    let winsSum = 0;
    for (let i = 0; i < root.legal.length; i++) {
      const action = root.legal[i];
      const n = root.visits[i];
      visits.set(action, n);
      winsSum += root.wins[i];
      if (n && total) policy[action] = n / total;
      if (n > bestN) bestN = n;
    }
    return {
      policy,
      value: total ? winsSum / total : value,
      visits,
      sims: total,
      elapsedS,
      best: this._bestRootAction(root),
      forced: false,
    };
  }

  /**
   * The move to play from the finished tree.
   *
   * Without a margin head this is the rule the agent has always used and the one
   * the Python trainer uses: the most-visited root child. With one, visits still
   * decide which moves are *candidates* — a move the search barely looked at is
   * not evidence of anything — but among the candidates whose win-Q is within
   * `marginEpsilon` of the best, the largest expected score gap wins. That is
   * what turns "wins by one point" into "wins by twenty" without ever trading a
   * win away for points.
   */
  _bestRootAction(root) {
    let bestN = -1;
    let bestI = 0;
    for (let i = 0; i < root.legal.length; i++) {
      if (root.visits[i] > bestN) {
        bestN = root.visits[i];
        bestI = i;
      }
    }
    if (!this.hasMargin || !root.margins || bestN <= 0) return root.legal[bestI];

    const floor = Math.max(1, bestN * this.config.marginMinVisitShare);
    let bestQ = -Infinity;
    for (let i = 0; i < root.legal.length; i++) {
      const n = root.visits[i];
      if (n >= floor) bestQ = Math.max(bestQ, root.wins[i] / n);
    }
    if (!Number.isFinite(bestQ)) return root.legal[bestI];

    let pickI = bestI;
    let pickMargin = -Infinity;
    let pickVisits = -1;
    for (let i = 0; i < root.legal.length; i++) {
      const n = root.visits[i];
      if (n < floor) continue;
      if (root.wins[i] / n < bestQ - this.config.marginEpsilon) continue;
      const margin = root.margins[i] / n;
      // ties on margin fall back to the visit count, so the rule stays a
      // refinement of the old one rather than a different search
      if (margin > pickMargin || (margin === pickMargin && n > pickVisits)) {
        pickMargin = margin;
        pickVisits = n;
        pickI = i;
      }
    }
    return root.legal[pickI];
  }

  /**
   * The last search's root edges — `{action, visits, q, prior}` per legal move.
   *
   * A port of `RootStatsMCTS.root_children()` in ludometer/gui/coach.py, and read
   * for exactly the same reason: `q` is the search's own value estimate for that
   * move, in the root player's frame. An edge the search never visited has no `q`
   * at all and reports `null` — callers must not invent one.
   */
  rootChildren() {
    const root = this._root;
    if (!root || !root.expanded) return [];
    const out = [];
    for (let i = 0; i < root.legal.length; i++) {
      const visits = root.visits[i];
      out.push({
        action: root.legal[i],
        visits,
        q: visits ? root.wins[i] / visits : null,
        // expected score gap, in the root player's frame; null without a margin head
        margin: visits && root.margins ? root.margins[i] / visits : null,
        prior: root.priors ? root.priors[i] : 0,
      });
    }
    return out;
  }

  /* --------------------------------------------------------------- internals */
  _newNode(state) {
    this.nodesCreated += 1;
    return new Node(state);
  }

  async _expand(node) {
    const { priors, value, margin } = await this.evaluator.evaluate(node.state, node.legal);
    this.evals += 1;
    node.initEdges(priors, this.hasMargin);
    node.evalMargin = this.hasMargin && margin != null ? margin : 0;
    return value;
  }

  _select(node) {
    const { cPuct, fpu } = this.config;
    const priors = node.priors;
    const visits = node.visits;
    const wins = node.wins;
    const scale = cPuct * Math.sqrt(node.nVisits + 1);
    let best = -1e30;
    let bestI = 0;
    for (let i = 0; i < visits.length; i++) {
      const n = visits[i];
      const q = n ? wins[i] / n : fpu;
      const score = q + (scale * priors[i]) / (1 + n);
      if (score > best) {
        best = score;
        bestI = i;
      }
    }
    return bestI;
  }

  /** True iff `action` empties the board and therefore triggers a refill. */
  _isStochastic(state, action) {
    const source = Math.floor(action / 30);
    const color = Math.floor((action - source * 30) / 6);
    const pool = source === CENTER ? state.center : state.factories[source];
    return pool[color] === state.tilesLeft;
  }

  /** Clone with a fresh bag order, then apply `action` — one refill draw. */
  _determinize(state, action) {
    const child = state.clone();
    this.counter += 1;
    child.rng.seed((Math.imul(this.counter, 2654435761) ^ 0x9e3779b9) >>> 0);
    child.rng.shuffle(child.bag);
    child.apply(action);
    return child;
  }

  /** Identity of a post-refill position: factory + center contents. */
  static _chanceKey(state) {
    const parts = [];
    for (const f of state.factories) parts.push(f.join(""));
    parts.push(state.center.join(""));
    return parts.join("|");
  }

  _child(node, index) {
    const entry = node.children[index];
    if (entry instanceof Node) return entry;
    const action = node.legal[index];
    if (entry === null && !this._isStochastic(node.state, action)) {
      const next = node.state.clone();
      next.apply(action);
      const child = this._newNode(next);
      node.children[index] = child;
      return child;
    }
    const table = entry === null ? new Map() : entry;
    node.children[index] = table;
    if (table.size >= this.config.chanceChildren) {
      const keys = Array.from(table.keys());
      return table.get(keys[this.rng.randrange(keys.length)]);
    }
    const state = this._determinize(node.state, action);
    const key = MCTS._chanceKey(state);
    let child = table.get(key);
    if (child === undefined) {
      child = this._newNode(state);
      table.set(key, child);
    }
    return child;
  }

  async _simulate(root) {
    let node = root;
    const path = [];
    let v0;
    let m0 = 0;
    for (;;) {
      if (node.isTerminal) {
        v0 = node.terminalV0;
        m0 = node.terminalMargin0;
        break;
      }
      if (!node.expanded) {
        const value = await this._expand(node);
        v0 = node.player === 0 ? value : -value;
        m0 = node.player === 0 ? node.evalMargin : -node.evalMargin;
        break;
      }
      const index = this._select(node);
      path.push([node, index]);
      node = this._child(node, index);
    }
    for (const [parent, index] of path) {
      parent.visits[index] += 1;
      parent.nVisits += 1;
      const sign = parent.player === 0 ? 1 : -1;
      parent.wins[index] += sign * v0;
      if (parent.margins) parent.margins[index] += sign * m0;
    }
  }

  /* ------------------------------------------------------------- batched search */

  /**
   * One descent, laying virtual loss as it goes.
   *
   * Ends either at a terminal node — backed up immediately, since no net is
   * needed — or at an unexpanded one, which is filed in `queue` for the coming
   * forward pass. A leaf that two descents both reach keeps one queue entry and
   * two paths: both are real simulations and both get the same value back.
   */
  _collect(root, queue, byNode) {
    const vl = this.config.virtualLoss;
    let node = root;
    const path = [];
    for (;;) {
      if (node.isTerminal) {
        this._backup(path, node.terminalV0, node.terminalMargin0);
        return;
      }
      if (!node.expanded) {
        let entry = byNode.get(node);
        if (!entry) {
          entry = { node, paths: [] };
          byNode.set(node, entry);
          queue.push(entry);
          node.pending = true;
        }
        entry.paths.push(path);
        return;
      }
      const index = this._select(node);
      // The visit is taken now and the loss assumed now; `_backup` puts the
      // assumed loss back and adds the real value, so the finished tree is the
      // one a sequential search would have built.
      node.visits[index] += 1;
      node.nVisits += 1;
      node.wins[index] -= vl;
      path.push([node, index]);
      node = this._child(node, index);
    }
  }

  /** Undo the virtual loss along `path` and credit the real result. */
  _backup(path, v0, m0) {
    const vl = this.config.virtualLoss;
    for (let k = 0; k < path.length; k++) {
      const parent = path[k][0];
      const index = path[k][1];
      const sign = parent.player === 0 ? 1 : -1;
      parent.wins[index] += vl + sign * v0;
      if (parent.margins) parent.margins[index] += sign * m0;
    }
  }

  /** Gather up to `want` leaves, evaluate them in one pass, back them all up. */
  async _simulateBatch(root, want) {
    const queue = [];
    const byNode = new Map();
    for (let i = 0; i < want; i++) this._collect(root, queue, byNode);
    if (!queue.length) return want; // every descent hit a terminal node

    const results = await this.evaluator.evaluateBatch(
      queue.map((e) => e.node.state),
      queue.map((e) => e.node.legal)
    );
    this.evals += queue.length;
    this.batches += 1;
    for (let i = 0; i < queue.length; i++) {
      const { node, paths } = queue[i];
      const { priors, value, margin } = results[i];
      node.initEdges(priors, this.hasMargin);
      node.evalMargin = this.hasMargin && margin != null ? margin : 0;
      node.pending = false;
      const sign = node.player === 0 ? 1 : -1;
      const v0 = sign * value;
      const m0 = sign * node.evalMargin;
      for (let p = 0; p < paths.length; p++) this._backup(paths[p], v0, m0);
    }
    return want;
  }
}

/**
 * Pick an action from a visit distribution.
 *
 * `temperature <= 0` is arg-max; otherwise the distribution is raised to
 * `1 / temperature`, renormalised and sampled — the stall-breaker the Python
 * agent uses once a game has dragged past STALL_ROUNDS rounds.
 */
export function selectAction(policy, temperature = 0, rng = null) {
  if (temperature <= 0) {
    let best = 0;
    let bestP = -Infinity;
    for (let i = 0; i < policy.length; i++) {
      if (policy[i] > bestP) {
        bestP = policy[i];
        best = i;
      }
    }
    return best;
  }
  const probs = new Float64Array(policy.length);
  let total = 0;
  for (let i = 0; i < policy.length; i++) {
    const p = policy[i] > 0 ? Math.pow(policy[i], 1 / temperature) : 0;
    probs[i] = p;
    total += p;
  }
  if (!(total > 0) || !Number.isFinite(total)) return selectAction(policy, 0);
  const draw = (rng ? rng.random() : Math.random()) * total;
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i];
    if (draw <= acc) return i;
  }
  return selectAction(policy, 0);
}

/** The playing agent: search, then play the arg-max visit count. */
export class MCTSAgent {
  constructor(evaluator, config = {}, rng = null) {
    this.mcts = new MCTS(evaluator, config, rng);
    this.rng = this.mcts.rng;
    this.lastSearch = {};
  }

  /** @returns {Promise<number>} the chosen action id */
  async act(state, opts = {}) {
    const legal = state.legalActions();
    if (!legal.length) throw new Error("no legal actions (terminal state?)");
    if (legal.length === 1) {
      this.lastSearch = { sims: 0, elapsedS: 0, forced: true };
      return legal[0];
    }
    const result = await this.mcts.search(state, opts);
    this.lastSearch = {
      sims: result.sims,
      elapsedS: result.elapsedS,
      value: result.value,
      forced: result.forced,
      nodes: this.mcts.nodesCreated,
    };
    // a pathologically long game gets randomised so that it terminates
    const temperature = state.roundIndex >= STALL_ROUNDS ? 1 : 0;
    if (temperature > 0) return selectAction(result.policy, temperature, this.rng);
    return result.best;
  }
}
