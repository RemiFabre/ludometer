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
    this.children = null; // Node | Map<string, Node> | null, per edge
    this.expanded = false;
    this.nVisits = 0;
    this.terminalV0 = state.isTerminal ? state.outcome() || 0 : 0;
  }

  get isTerminal() {
    return this.state.isTerminal;
  }

  initEdges(priors) {
    const n = this.legal.length;
    this.priors = priors;
    this.visits = new Int32Array(n);
    this.wins = new Float64Array(n);
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
    const deadline = timeLimitS > 0 ? started + timeLimitS * 1000 : Infinity;
    let done = 0;
    while (done < cap) {
      await this._simulate(root);
      done += 1;
      if (done % YIELD_EVERY === 0) {
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
    let best = root.legal[0];
    let bestN = -1;
    let winsSum = 0;
    for (let i = 0; i < root.legal.length; i++) {
      const action = root.legal[i];
      const n = root.visits[i];
      visits.set(action, n);
      winsSum += root.wins[i];
      if (n && total) policy[action] = n / total;
      if (n > bestN) {
        bestN = n;
        best = action;
      }
    }
    return {
      policy,
      value: total ? winsSum / total : value,
      visits,
      sims: total,
      elapsedS,
      best,
      forced: false,
    };
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
    const { priors, value } = await this.evaluator.evaluate(node.state, node.legal);
    this.evals += 1;
    node.initEdges(priors);
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
    for (;;) {
      if (node.isTerminal) {
        v0 = node.terminalV0;
        break;
      }
      if (!node.expanded) {
        const value = await this._expand(node);
        v0 = node.player === 0 ? value : -value;
        break;
      }
      const index = this._select(node);
      path.push([node, index]);
      node = this._child(node, index);
    }
    for (const [parent, index] of path) {
      parent.visits[index] += 1;
      parent.nVisits += 1;
      parent.wins[index] += parent.player === 0 ? v0 : -v0;
    }
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
