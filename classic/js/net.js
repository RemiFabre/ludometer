/* The policy+value net, as seen from JavaScript.
 *
 * A port of ludometer/train/net.py's `NetEvaluator`: encode the position, run the
 * exported ONNX graph, softmax the logits **over the legal actions only** (the
 * same thing as masking the rest to -inf, and cheaper), and hand back priors
 * aligned with the caller's `legal` list plus a value in [-1, 1] for the player
 * to move.
 *
 * The graph is ../model/model.onnx, exported by ludometer/export/onnx_export.py
 * and checked against torch on 100 real positions before it is written, so the
 * only thing that can go wrong on this side is the input vector — which is what
 * the engine fixtures pin down.
 *
 * onnxruntime-web is passed in rather than imported, so this module stays usable
 * from a worker, from node, and from a test with a stub evaluator.
 *
 * Two things beyond a single forward pass live here, both measured rather than
 * assumed (see NOTES_FOR_REMI.md):
 *
 *  - **Batching.** One position at a time is the worst case for every backend:
 *    on this laptop the WASM runtime does 2.1 k positions/s at batch 1 and
 *    4.2 k at batch 32, and the WebGPU runtime does 250/s at batch 1 (a GPU
 *    dispatch costs ~4 ms whatever is in it) but 16 k/s at batch 64. So
 *    `evaluateBatch` is the primitive and `evaluate` is the batch-of-one
 *    special case; the search feeds it through virtual loss.
 *
 *  - **The margin head.** Older exports have two outputs (`policy`, `value`);
 *    a later one may add `margin`, a prediction of the score gap. Which
 *    outputs exist is read off the session, never assumed, so the same player
 *    code runs both.
 */

import { ENCODED_SIZE } from "./engine.js";

/** The name of the optional third output: a signed score-gap prediction. */
export const MARGIN_OUTPUT = "margin";

export class OnnxEvaluator {
  constructor(session, ort, backend = "wasm") {
    this.session = session;
    this.ort = ort;
    this.backend = backend;
    this.buffer = new Float32Array(ENCODED_SIZE);
    this.batchBuffer = null; // grown on demand by evaluateBatch
    this.calls = 0;
    this.positions = 0;
    // Feature-detected, never assumed: a run3-era graph has policy+value, a
    // later one may also have margin. `outputNames` is what the runtime loaded.
    const names = session.outputNames || [];
    this.hasMargin = names.indexOf(MARGIN_OUTPUT) !== -1;
    this.outputNames = Array.from(names);
  }

  /**
   * @param {object} ort  the onnxruntime-web module namespace
   * @param {string|ArrayBuffer|Uint8Array} model  URL or raw bytes of model.onnx
   * @param {object} opts  {executionProviders, backend}
   */
  static async create(ort, model, opts = {}) {
    const backend = opts.backend || "wasm";
    const session = await ort.InferenceSession.create(model, {
      executionProviders: opts.executionProviders || [backend],
      graphOptimizationLevel: "all",
    });
    return new OnnxEvaluator(session, ort, backend);
  }

  /** `(state, legal) -> {priors, value, margin}`; priors are aligned with `legal`. */
  async evaluate(state, legal) {
    state.encode(this.buffer);
    const input = new this.ort.Tensor("float32", this.buffer, [1, ENCODED_SIZE]);
    const out = await this.session.run({ obs: input });
    this.calls += 1;
    this.positions += 1;
    return this._read(out, 0, legal);
  }

  /**
   * Evaluate `n` positions in one graph run.
   *
   * `states[i]` is scored against `legals[i]`; the result array is aligned with
   * both. This is the primitive the batched search calls — one dispatch for the
   * whole leaf set instead of one per leaf.
   */
  async evaluateBatch(states, legals) {
    const n = states.length;
    if (n === 0) return [];
    if (n === 1) return [await this.evaluate(states[0], legals[0])];
    if (!this.batchBuffer || this.batchBuffer.length < n * ENCODED_SIZE) {
      this.batchBuffer = new Float32Array(n * ENCODED_SIZE);
    }
    // A fresh view per run: ort keeps a reference to the tensor's data, so the
    // sub-arrays must cover exactly the rows this run submits.
    const data = this.batchBuffer.subarray(0, n * ENCODED_SIZE);
    for (let i = 0; i < n; i++) {
      states[i].encode(data.subarray(i * ENCODED_SIZE, (i + 1) * ENCODED_SIZE));
    }
    const input = new this.ort.Tensor("float32", data, [n, ENCODED_SIZE]);
    const out = await this.session.run({ obs: input });
    this.calls += 1;
    this.positions += n;
    const results = new Array(n);
    for (let i = 0; i < n; i++) results[i] = this._read(out, i, legals[i]);
    return results;
  }

  /** Row `row` of a (possibly batched) output, softmaxed over `legal`. */
  _read(out, row, legal) {
    const value = out.value.data[row];
    const margin = this.hasMargin ? out[MARGIN_OUTPUT].data[row] : null;
    const n = legal.length;
    if (n === 0) return { priors: new Float32Array(0), value, margin };

    const logits = out.policy.data;
    // policy is [batch, ACTION_SPACE]; the last dim is the row stride.
    const dims = out.policy.dims;
    const stride = dims && dims.length ? dims[dims.length - 1] : logits.length;
    const base = row * stride;
    const priors = new Float32Array(n);
    let max = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = logits[base + legal[i]];
      priors[i] = v;
      if (v > max) max = v;
    }
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const e = Math.exp(priors[i] - max);
      priors[i] = e;
      sum += e;
    }
    for (let i = 0; i < n; i++) priors[i] /= sum;
    return { priors, value, margin };
  }
}

/** Uniform priors, zero value — lets the search run with no net at all (tests). */
export class UniformEvaluator {
  constructor() {
    this.calls = 0;
    this.positions = 0;
    this.hasMargin = false;
    this.backend = "none";
  }

  async evaluate(_state, legal) {
    this.calls += 1;
    this.positions += 1;
    const n = legal.length;
    return { priors: new Float32Array(n).fill(n ? 1 / n : 0), value: 0, margin: null };
  }

  async evaluateBatch(states, legals) {
    this.calls += 1;
    this.positions += states.length;
    return legals.map((legal) => ({
      priors: new Float32Array(legal.length).fill(legal.length ? 1 / legal.length : 0),
      value: 0,
      margin: null,
    }));
  }
}

/* ------------------------------------------------------------------- backends */

/**
 * What the two vendored onnxruntime-web builds are, and when each one is used.
 *
 * The default build is the pure-WASM one the player has always shipped. The
 * `jspi` build adds onnxruntime's native **WebGPU** execution provider; it is
 * only downloaded when the browser can actually use it, so a browser without
 * WebGPU pays nothing for its existence.
 *
 * Two things have to be true for the WebGPU build to load at all:
 *   - `navigator.gpu` — the browser has WebGPU;
 *   - `WebAssembly.Suspending` — the browser has JSPI (stack switching), which
 *     is how onnxruntime's WebGPU EP suspends the WASM call while the GPU
 *     works. The alternative build (asyncify) needs no JSPI but is 2.5 MB
 *     larger over the wire, which is not worth it for a fallback path.
 *
 * Anything else — Safari today, Firefox today, an old Chrome, a machine with no
 * GPU adapter — takes the WASM path, which is exactly what shipped before.
 */
export const BACKENDS = {
  wasm: {
    module: "../vendor/onnxruntime-web/ort.wasm.bundle.min.mjs",
    wasm: "../vendor/onnxruntime-web/ort-wasm-simd-threaded.wasm",
    ep: "wasm",
  },
  webgpu: {
    module: "../vendor/onnxruntime-web/ort.jspi.bundle.min.mjs",
    wasm: "../vendor/onnxruntime-web/ort-wasm-simd-threaded.jspi.wasm",
    ep: "webgpu",
  },
};

/**
 * Is the WebGPU path worth trying in this environment?
 *
 * Deliberately cheap and synchronous — `requestAdapter()` is the real test and
 * it happens when the session is created, where a failure falls back anyway.
 */
export function webgpuLikely(scope = globalThis) {
  return !!(
    scope.navigator &&
    scope.navigator.gpu &&
    typeof scope.WebAssembly !== "undefined" &&
    typeof scope.WebAssembly.Suspending === "function"
  );
}
