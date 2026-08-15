/* The AI's thread.
 *
 * Everything expensive happens here: loading the 13 MB ONNX graph, and running
 * PUCT search against it. The page keeps the game state and only ships the
 * position over (`toSetup()` is structured-clone-safe), so the UI thread never
 * blocks — the board stays interactive and the "thinking" clock keeps ticking
 * while the search runs.
 *
 * Protocol (all messages carry an `id` that is echoed back):
 *   -> {type:"init", ortUrl, wasmUrl, modelUrl}
 *                                            <- {type:"loading"} ... {type:"ready"} | {type:"error"}
 *   -> {type:"search", setup, budgetS}       <- {type:"progress"} ... {type:"result"}
 *   -> {type:"policy", setup}                <- {type:"result"}   (hint: no search)
 *   -> {type:"rate", setup, actionId, budgetS}
 *                                            <- {type:"progress"} ... {type:"result"} (coach)
 *   -> {type:"cancel"}
 */

import { AzulState, Rng } from "./engine.js";
import { MCTS, STALL_ROUNDS, selectAction } from "./mcts.js";
import { OnnxEvaluator } from "./net.js";
import { describeAction } from "./report.js";

let ort = null;
let evaluator = null;
let cancelled = false;
const rng = new Rng((Date.now() ^ 0x5eed) >>> 0);

/**
 * Download the model, reporting bytes as they arrive.
 *
 * It is a 13 MB file, which on a phone is several seconds of nothing happening;
 * streaming the body lets the page show a real percentage instead of a spinner
 * that might as well be broken.
 */
async function fetchWithProgress(url, id) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`model fetch failed: ${response.status} ${response.statusText}`);
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    self.postMessage({ type: "loading", id, received, total });
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

async function init(msg) {
  ort = await import(msg.ortUrl);
  // SharedArrayBuffer needs cross-origin isolation, which GitHub Pages does not
  // send; one thread is also plenty for a batch-of-one MLP.
  ort.env.wasm.numThreads = 1;
  // Object form on purpose: a bare string prefix makes onnxruntime look for the
  // Emscripten glue .mjs next to the .wasm, and the bundled build has it inlined.
  ort.env.wasm.wasmPaths = { wasm: msg.wasmUrl };
  ort.env.logLevel = "error";
  const bytes = await fetchWithProgress(msg.modelUrl, msg.id);
  evaluator = await OnnxEvaluator.create(ort, bytes);
  // one throwaway evaluation so the first real move is not paying for warm-up
  const warm = AzulState.newGame(1, new Rng(1));
  await evaluator.evaluate(warm, warm.legalActions());
  return { bytes: bytes.length };
}

async function search(msg) {
  const state = AzulState.fromSetup(msg.setup, new Rng(rng.next()));
  const legal = state.legalActions();
  if (!legal.length) throw new Error("no legal actions in the position sent to the worker");
  if (legal.length === 1) {
    return { action: legal[0], search: { sims: 0, elapsedS: 0, forced: true } };
  }

  const mcts = new MCTS(evaluator, {}, new Rng(rng.next()));
  cancelled = false;
  const result = await mcts.search(state, {
    timeLimitS: msg.budgetS,
    shouldStop: () => cancelled,
    onProgress: ({ sims, elapsedS }) => {
      self.postMessage({ type: "progress", id: msg.id, sims, elapsedS });
    },
  });
  // A game that drags past STALL_ROUNDS gets randomised so that it terminates
  // (two arg-max players can otherwise loop forever — see mcts.js).
  const action =
    state.roundIndex >= STALL_ROUNDS ? selectAction(result.policy, 1, mcts.rng) : result.best;
  const top = [...result.visits.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);
  return {
    action,
    search: {
      sims: result.sims,
      elapsedS: result.elapsedS,
      value: result.value,
      nodes: mcts.nodesCreated,
      forced: false,
      top,
    },
  };
}

/**
 * Coach mode: rate one of *your* moves with the AI's own search.
 *
 * A port of ludometer/gui/coach.py, definition for definition. The same PUCT
 * search the opponent plays with is run on your position and the root's edge
 * statistics are read back out:
 *
 *     delta = Q(the move you played) − max over explored children Q
 *
 * `Q` is in the root player's frame — yours — on the net's [-1, 1] scale, so
 * 0.00 means "the move the AI would have played" and −0.06 means the search
 * values yours six hundredths of a win worse. A move the search never visited
 * has no `Q` and is reported `unrated` rather than given an invented number.
 */
async function rate(msg) {
  const state = AzulState.fromSetup(msg.setup, new Rng(rng.next()));
  const action = msg.actionId;
  const legal = state.legalActions();
  const base = { budgetS: msg.budgetS, legal: legal.length, sims: 0, elapsedS: 0 };
  if (legal.indexOf(action) === -1) {
    return { coach: { ...base, unrated: true, reason: "that move is not legal" } };
  }
  if (legal.length === 1) {
    return { coach: { ...base, delta: 0, forced: true } };
  }

  const mcts = new MCTS(evaluator, {}, new Rng(rng.next()));
  const result = await mcts.search(state, {
    timeLimitS: msg.budgetS,
    onProgress: ({ sims, elapsedS }) => {
      self.postMessage({ type: "progress", id: msg.id, sims, elapsedS });
    },
  });
  base.sims = result.sims;
  base.elapsedS = result.elapsedS;

  const explored = mcts.rootChildren().filter((c) => c.visits && c.q !== null);
  if (!explored.length) {
    return {
      coach: { ...base, unrated: true, reason: "the search had no time to explore this position" },
    };
  }
  const best = explored.reduce((a, b) => (b.q > a.q ? b : a));
  const mine = explored.find((c) => c.action === action);
  if (!mine) {
    return {
      coach: { ...base, unrated: true, reason: "the search never explored this move" },
    };
  }
  return {
    coach: {
      ...base,
      // Q is already in your frame, so this can only be <= 0; clamp the float
      // noise away rather than showing "+0.00"
      delta: Math.min(0, mine.q - best.q),
      your_q: mine.q,
      best_q: best.q,
      visits: mine.visits,
      best_visits: best.visits,
      best_text: describeAction(state, best.action).text,
      explored: explored.length,
    },
  };
}

/** The policy head's own pick, no search — what the "Suggest a move" button asks. */
async function policy(msg) {
  const state = AzulState.fromSetup(msg.setup, new Rng(rng.next()));
  const legal = state.legalActions();
  if (!legal.length) throw new Error("no legal actions in the position sent to the worker");
  const { priors, value } = await evaluator.evaluate(state, legal);
  let best = 0;
  for (let i = 1; i < priors.length; i++) if (priors[i] > priors[best]) best = i;
  return { action: legal[best], search: { sims: 0, elapsedS: 0, value, prior: priors[best], forced: false } };
}

self.onmessage = async (event) => {
  const msg = event.data;
  if (msg.type === "cancel") {
    cancelled = true;
    return;
  }
  try {
    let payload;
    if (msg.type === "init") payload = await init(msg);
    else if (msg.type === "search") payload = await search(msg);
    else if (msg.type === "policy") payload = await policy(msg);
    else if (msg.type === "rate") payload = await rate(msg);
    else throw new Error(`unknown message type ${msg.type}`);
    self.postMessage({ type: msg.type === "init" ? "ready" : "result", id: msg.id, ...payload });
  } catch (err) {
    self.postMessage({ type: "error", id: msg.id, message: String((err && err.message) || err) });
  }
};
