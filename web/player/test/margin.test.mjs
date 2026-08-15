/* The margin head: detected, carried, and used — before any model has one.
 *
 * A later export will add a third ONNX output, `margin`, predicting the score
 * gap rather than just who wins. The player must pick that up on its own: no
 * flag, no redeploy of the JavaScript, just a new model.onnx. This test holds
 * both halves of that promise:
 *
 *   1. **Detection and plumbing**, through the real vendored runtime, against
 *      two hand-built graphs in fixtures/ (see make_toy_onnx.py). They are not
 *      nets — they slice the observation vector straight into their outputs — so
 *      the expected answer for any input is exact, and the batched read-out can
 *      be checked row by row rather than approximately.
 *
 *   2. **Selection**, with a stub evaluator: among root moves that win equally
 *      often, the search must play the one that wins by more — and a two-output
 *      net must still get the old most-visited-child rule, unchanged.
 *
 *   node web/player/test/margin.test.mjs
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { ACTION_SPACE, AzulState, ENCODED_SIZE, Rng } from "../js/engine.js";
import { MCTS } from "../js/mcts.js";
import { OnnxEvaluator } from "../js/net.js";
import * as ort from "../vendor/onnxruntime-web/ort.wasm.bundle.min.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");

let failures = 0;
function check(name, ok, detail = "") {
  console.log(`  ${ok ? "ok  " : "FAIL"} ${name}${detail ? " — " + detail : ""}`);
  if (!ok) failures += 1;
}
const close = (a, b, tol = 1e-5) => Math.abs(a - b) <= tol;

/** A stand-in for AzulState: all the evaluator ever asks of it is `encode`. */
class FakeState {
  constructor(vector) {
    this.vector = vector;
  }
  encode(out) {
    out.set(this.vector);
  }
}

function obsWith(policyAt, value, margin) {
  const v = new Float32Array(ENCODED_SIZE);
  for (const [action, logit] of Object.entries(policyAt)) v[Number(action)] = logit;
  v[ACTION_SPACE] = value;
  v[ACTION_SPACE + 1] = margin;
  return v;
}

/* ------------------------------------------------- 1. detection and plumbing */
async function plumbing() {
  console.log("the runtime, on two hand-built graphs:");
  const two = await OnnxEvaluator.create(
    ort,
    readFileSync(join(HERE, "fixtures", "toy_two_output.onnx"))
  );
  check("a two-output graph reports no margin", two.hasMargin === false, two.outputNames.join(","));
  const plain = await two.evaluate(new FakeState(obsWith({ 0: 1, 1: 0 }, 0.25, 9)), [0, 1]);
  check("value still read from a two-output graph", close(plain.value, 0.25), `value ${plain.value}`);
  check("margin is null, never invented", plain.margin === null, String(plain.margin));

  const three = await OnnxEvaluator.create(
    ort,
    readFileSync(join(HERE, "fixtures", "toy_margin.onnx"))
  );
  check("a three-output graph reports a margin", three.hasMargin === true, three.outputNames.join(","));

  const one = await three.evaluate(new FakeState(obsWith({ 5: 2, 7: 0 }, -0.5, 12.5)), [5, 7]);
  check("value read from the right output", close(one.value, -0.5), `value ${one.value}`);
  check("margin read from the right output", close(one.margin, 12.5), `margin ${one.margin}`);
  // logits 2 and 0 over two legal moves: softmax is e^2 / (e^2 + 1)
  const expected = Math.exp(2) / (Math.exp(2) + 1);
  check("priors still softmax over the legal moves only", close(one.priors[0], expected, 1e-5),
    `${one.priors[0].toFixed(6)} vs ${expected.toFixed(6)}`);

  // A batch has to hand each row back to its own caller: same graph, three
  // different positions, three different answers, in order.
  const states = [
    new FakeState(obsWith({ 0: 1 }, 0.1, 1)),
    new FakeState(obsWith({ 0: 1 }, -0.2, -20)),
    new FakeState(obsWith({ 0: 1 }, 0.9, 33.5)),
  ];
  const batch = await three.evaluateBatch(states, [[0, 1], [0, 1], [0, 1]]);
  check("batched values are row-aligned",
    close(batch[0].value, 0.1) && close(batch[1].value, -0.2) && close(batch[2].value, 0.9),
    batch.map((b) => b.value.toFixed(2)).join(" "));
  check("batched margins are row-aligned",
    close(batch[0].margin, 1) && close(batch[1].margin, -20) && close(batch[2].margin, 33.5),
    batch.map((b) => b.margin.toFixed(1)).join(" "));
  // the same three positions one at a time must give the same answers
  const single = [];
  for (let i = 0; i < states.length; i++) single.push(await three.evaluate(states[i], [0, 1]));
  check("a batch answers exactly what one-at-a-time answers",
    single.every((s, i) => close(s.value, batch[i].value) && close(s.margin, batch[i].margin) &&
      close(s.priors[0], batch[i].priors[0])));

  const twoBatch = await two.evaluateBatch(states.slice(0, 2), [[0, 1], [0, 1]]);
  check("a batched two-output graph still reports no margin",
    twoBatch.every((r) => r.margin === null));
}

/* -------------------------------------------------------------- 2. selection */

/**
 * A net with an opinion, keyed by the position's own hash.
 *
 * `valueOf`/`marginOf` are given the state so a test can make one branch of the
 * real game tree "win narrowly" and another "win by a lot" without needing a
 * trained model to happen to think so.
 */
class ScriptedEvaluator {
  constructor({ hasMargin, valueOf, marginOf }) {
    this.hasMargin = hasMargin;
    this.valueOf = valueOf;
    this.marginOf = marginOf;
    this.calls = 0;
  }
  async evaluate(state, legal) {
    this.calls += 1;
    const n = legal.length;
    return {
      priors: new Float32Array(n).fill(n ? 1 / n : 0),
      value: this.valueOf(state),
      margin: this.hasMargin ? this.marginOf(state) : null,
    };
  }
  async evaluateBatch(states, legals) {
    const out = [];
    for (let i = 0; i < states.length; i++) out.push(await this.evaluate(states[i], legals[i]));
    return out;
  }
}

/**
 * The smallest game that can pose the question: three moves, then alternation.
 *
 * A real Azul position cannot be used here — nothing in a descendant state says
 * which root move led to it, and that is exactly what the script has to key on.
 * This carries `first` for the whole subtree, so every leaf under root move *m*
 * can be given the same value and the same margin, and the tie the margin rule
 * is supposed to break is a genuine one rather than search noise.
 *
 * It implements only what MCTS asks of a state; `tilesLeft = -1` keeps every
 * edge deterministic, so no determinization machinery is involved.
 */
class ToyState {
  constructor() {
    this.currentPlayer = 0;
    this.first = null;
    this.depth = 0;
    this.scores = [0, 0];
    this.factories = [[0, 0, 0, 0, 0]];
    this.center = [0, 0, 0, 0, 0];
    this.tilesLeft = -1;
    this.isTerminal = false;
  }
  legalActions() {
    return [0, 1, 2];
  }
  clone() {
    const s = new ToyState();
    s.currentPlayer = this.currentPlayer;
    s.first = this.first;
    s.depth = this.depth;
    return s;
  }
  apply(action) {
    if (this.first === null) this.first = action;
    this.depth += 1;
    this.currentPlayer = 1 - this.currentPlayer;
  }
  outcome() {
    return 0;
  }
}

/**
 * A root whose three moves are, to the value head, indistinguishable.
 *
 * All are worth exactly +0.20 to the root player; the only thing that separates
 * them is the margin head, which likes move 1. The old rule cannot see that and
 * plays whichever the visit counts happen to favour; the new rule must play 1.
 */
async function selection() {
  console.log("\nroot selection:");
  const state = new ToyState();
  const favoured = 1;

  // Keyed on the *root* move, so the whole subtree agrees and every root edge
  // settles at the same Q — a real tie, not one search's noise.
  const sign = (s) => (s.currentPlayer === 0 ? 1 : -1);
  const valueOf = (s) => sign(s) * 0.2;
  const marginOf = (s) => sign(s) * (s.first === favoured ? 30 : 2);
  const budget = { sims: 3000, batch: 1 };

  const withMargin = new MCTS(
    new ScriptedEvaluator({ hasMargin: true, valueOf, marginOf }),
    budget,
    new Rng(5)
  );
  const a = await withMargin.search(state.clone(), {});
  check("the margin head breaks a tie in win-Q", a.best === favoured,
    `played ${a.best}, wanted ${favoured}`);
  const kids = withMargin.rootChildren().filter((c) => c.visits);
  check("root children report a margin", kids.every((c) => c.margin !== null));
  check("the tie really was a tie in Q",
    Math.max(...kids.map((c) => c.q)) - Math.min(...kids.map((c) => c.q)) < 0.03,
    `spread ${(Math.max(...kids.map((c) => c.q)) - Math.min(...kids.map((c) => c.q))).toFixed(4)}`);

  const noMargin = new MCTS(
    new ScriptedEvaluator({ hasMargin: false, valueOf, marginOf }),
    budget,
    new Rng(5)
  );
  const b = await noMargin.search(state.clone(), {});
  let bestN = -1;
  let mostVisited = null;
  for (const c of noMargin.rootChildren()) {
    if (c.visits > bestN) {
      bestN = c.visits;
      mostVisited = c.action;
    }
  }
  check("without a margin head the old rule is untouched", b.best === mostVisited,
    `played ${b.best}, most visited ${mostVisited}`);
  check("without a margin head no margin is stored",
    noMargin.rootChildren().every((c) => c.margin === null));

  // A win is never traded for points: a move that is clearly worse in Q must
  // not be chosen however big its margin.
  const lopsided = new MCTS(
    new ScriptedEvaluator({
      hasMargin: true,
      // the favoured move loses badly, but promises a huge score gap
      valueOf: (s) => sign(s) * (s.first === favoured ? -0.9 : 0.4),
      marginOf: (s) => sign(s) * (s.first === favoured ? 99 : 1),
    }),
    budget,
    new Rng(5)
  );
  const c = await lopsided.search(state.clone(), {});
  check("a losing move is not bought with points", c.best !== favoured, `played ${c.best}`);
}

/* --------------------------------------------------- 3. batching is bookkeeping */

/**
 * Virtual loss has to be exact, not approximate.
 *
 * Every simulation must show up as one visit and one real value — no more, no
 * less — whatever the batch size. If the assumed loss were ever left behind,
 * `wins` would drift and Q would be quietly wrong, which is the kind of bug that
 * looks like "the AI got weaker" and nothing else.
 */
async function batching() {
  console.log("\nbatched search bookkeeping:");
  const state = AzulState.newGame(11, new Rng(11));
  for (const batch of [1, 4, 32]) {
    const mcts = new MCTS(
      new ScriptedEvaluator({ hasMargin: true, valueOf: () => 0.15, marginOf: () => 3 }),
      { batch },
      new Rng(2)
    );
    const res = await mcts.search(state.clone(), { timeLimitS: 0.5 });
    const kids = mcts.rootChildren();
    const visits = kids.reduce((a, c) => a + c.visits, 0);
    check(`batch ${batch}: visits add up to the simulation count`, visits === res.sims,
      `${visits} vs ${res.sims}`);
    // every leaf is worth +0.15 to whoever is to move there, so no root Q can
    // possibly be outside [-1, 1] — a leaked virtual loss shows up here at once
    check(`batch ${batch}: every root Q stays inside [-1, 1]`,
      kids.every((c) => c.q === null || (c.q >= -1 && c.q <= 1)),
      kids.map((c) => (c.q === null ? "-" : c.q.toFixed(2))).slice(0, 4).join(" "));
    check(`batch ${batch}: the search actually ran`, res.sims > 50, `${res.sims} sims`);
  }

  // The ramp is not a nicety: a batch of 64 laid on an empty tree cost 3–17 in
  // a 20-game match at equal simulation counts. So a small search must never
  // *see* a batch of 64, whatever the ceiling says.
  const small = new MCTS(
    new ScriptedEvaluator({ hasMargin: false, valueOf: () => 0, marginOf: () => 0 }),
    { batch: 64, batchRamp: 16, sims: 160 },
    new Rng(2)
  );
  const res = await small.search(state.clone(), {});
  const biggest = Math.ceil(res.sims / Math.max(1, small.batches));
  check("the batch stays small while the tree is small",
    biggest <= 1 + Math.ceil(res.sims / 16),
    `average batch ${biggest} over ${small.batches} passes of ${res.sims} sims`);
}

async function main() {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmBinary = readFileSync(
    join(ROOT, "vendor", "onnxruntime-web", "ort-wasm-simd-threaded.wasm")
  ).buffer;
  ort.env.logLevel = "error";

  await plumbing();
  await selection();
  await batching();

  if (failures) {
    console.error(`\nFAIL — ${failures} check(s) failed`);
    return 1;
  }
  console.log("\nmargin ok: detected when present, ignored when absent, and never worth a loss");
  return 0;
}

process.exit(await main());
