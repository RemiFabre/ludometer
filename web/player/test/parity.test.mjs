/* Hold the *browser's* runtime to the torch reference.
 *
 * ludometer/export/onnx_export.py already checks the exported graph against torch
 * — but through onnxruntime-python, which is a different build from the
 * onnxruntime-web WASM the page actually runs. This test closes that gap: the
 * exporter writes torch's own answers on 100 real positions to
 * fixtures/torch_reference.json.gz, and here they are replayed through the exact
 * vendored WASM runtime a visitor executes, one position at a time (batch of 1,
 * as in the player).
 *
 *   node web/player/test/parity.test.mjs
 */
import { gunzipSync } from "node:zlib";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createHash } from "node:crypto";

import * as ort from "../vendor/onnxruntime-web/ort.wasm.bundle.min.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");

async function main() {
  ort.env.wasm.numThreads = 1;
  // node's fetch cannot read file:// URLs, so hand the runtime the bytes
  // directly; the page instead points wasmPaths at the served .wasm.
  ort.env.wasm.wasmBinary = readFileSync(join(ROOT, "vendor", "onnxruntime-web", "ort-wasm-simd-threaded.wasm")).buffer;
  ort.env.logLevel = "error";

  const ref = JSON.parse(gunzipSync(readFileSync(join(HERE, "fixtures", "torch_reference.json.gz"))).toString("utf8"));
  const modelBytes = readFileSync(join(ROOT, "model", "model.onnx"));
  const digest = createHash("sha256").update(modelBytes).digest("hex");
  if (ref.onnx_sha256 && ref.onnx_sha256 !== digest) {
    console.error("FAIL — model.onnx does not match the reference it was exported with");
    console.error(`  model  ${digest}`);
    console.error(`  refers ${ref.onnx_sha256}`);
    console.error("  re-run: uv run python -m ludometer.export.onnx_export");
    return 1;
  }

  const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
  const tol = ref.tol || 1e-4;
  let policyDiff = 0;
  let valueDiff = 0;
  const n = ref.obs.length;
  for (let i = 0; i < n; i++) {
    const input = new ort.Tensor("float32", Float32Array.from(ref.obs[i]), [1, ref.obs[i].length]);
    const out = await session.run({ obs: input });
    const policy = out.policy.data;
    for (let k = 0; k < policy.length; k++) {
      policyDiff = Math.max(policyDiff, Math.abs(policy[k] - ref.policy[i][k]));
    }
    valueDiff = Math.max(valueDiff, Math.abs(out.value.data[0] - ref.value[i]));
  }

  const worst = Math.max(policyDiff, valueDiff);
  console.log(
    `onnxruntime-web vs torch on ${n} positions (${ref.checkpoint}): ` +
      `policy max |diff| ${policyDiff.toExponential(2)}, value ${valueDiff.toExponential(2)}, tol ${tol.toExponential(0)}`
  );
  if (worst > tol) {
    console.error(`FAIL — worst gap ${worst.toExponential(2)} exceeds ${tol}`);
    return 1;
  }
  console.log("parity ok: the browser runtime reproduces the trained net");
  return 0;
}

process.exit(await main());
