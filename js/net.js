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
 */

import { ENCODED_SIZE } from "./engine.js";

export class OnnxEvaluator {
  constructor(session, ort) {
    this.session = session;
    this.ort = ort;
    this.buffer = new Float32Array(ENCODED_SIZE);
    this.calls = 0;
  }

  /**
   * @param {object} ort  the onnxruntime-web module namespace
   * @param {string|ArrayBuffer|Uint8Array} model  URL or raw bytes of model.onnx
   */
  static async create(ort, model) {
    const session = await ort.InferenceSession.create(model, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return new OnnxEvaluator(session, ort);
  }

  /** `(state, legal) -> {priors, value}`; priors are aligned with `legal`. */
  async evaluate(state, legal) {
    state.encode(this.buffer);
    const input = new this.ort.Tensor("float32", this.buffer, [1, ENCODED_SIZE]);
    const out = await this.session.run({ obs: input });
    this.calls += 1;
    const value = out.value.data[0];
    const n = legal.length;
    if (n === 0) return { priors: new Float32Array(0), value };

    const logits = out.policy.data;
    const priors = new Float32Array(n);
    let max = -Infinity;
    for (let i = 0; i < n; i++) {
      const v = logits[legal[i]];
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
    return { priors, value };
  }
}

/** Uniform priors, zero value — lets the search run with no net at all (tests). */
export class UniformEvaluator {
  constructor() {
    this.calls = 0;
  }

  async evaluate(_state, legal) {
    this.calls += 1;
    const n = legal.length;
    return { priors: new Float32Array(n).fill(n ? 1 / n : 0), value: 0 };
  }
}
