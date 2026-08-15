/* Hold the *GPU* backend to the same torch reference the WASM one answers to.
 *
 * parity.test.mjs replays 100 real positions through the vendored WASM runtime
 * under node. The WebGPU runtime cannot be tested that way — it needs a browser
 * with an adapter — so this test serves web/player/, opens it in Chrome, loads
 * model.onnx on the WebGPU execution provider and replays the same fixture
 * there, one position at a time and again as a batch.
 *
 * The tolerance is looser than the WASM test's, deliberately: a GPU runs the
 * same fp32 graph with different kernels and a different summation order, so
 * agreement to ~1e-3 is the honest bar. What would matter is a *policy* that
 * ranks moves differently, so that is checked separately and exactly.
 *
 * No GPU, no WebGPU, no Chrome — the test says so and passes. It is a guard on
 * a path that only some visitors take, not a requirement on the machine.
 *
 *   node web/player/test/webgpu.test.mjs
 */
import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { readFile, mkdtemp, rm } from "node:fs/promises";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { fileURLToPath } from "node:url";
import { dirname, join, normalize, extname } from "node:path";
import { tmpdir } from "node:os";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");
const CHROME =
  process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".gz": "application/gzip",
};

/* The probe page is generated rather than committed: it is apparatus, and the
 * deployed site must not carry a file whose only purpose is this test. */
const PROBE = `<!doctype html><meta charset="utf-8"><title>webgpu parity</title><body><pre id=o></pre>
<script type="module">
import { OnnxEvaluator, BACKENDS, webgpuLikely } from "/js/net.js";
window.__probe = async () => {
  if (!webgpuLikely()) return { skip: "no navigator.gpu or no JSPI in this browser" };
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return { skip: "navigator.gpu is present but there is no adapter" };
  const ort = await import(BACKENDS.webgpu.module);
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = { wasm: BACKENDS.webgpu.wasm };
  ort.env.logLevel = "error";
  const bytes = new Uint8Array(await (await fetch("/model/model.onnx")).arrayBuffer());
  const ev = await OnnxEvaluator.create(ort, bytes, { backend: "webgpu", executionProviders: ["webgpu"] });
  const ref = window.__ref;
  const state = (row) => ({ encode: (out) => out.set(Float32Array.from(row)) });
  const legal = ref.legal;
  let vDiff = 0, pDiff = 0, rankDiff = 0;
  const singles = [];
  for (let i = 0; i < ref.obs.length; i++) {
    const r = await ev.evaluate(state(ref.obs[i]), legal);
    singles.push(r);
    vDiff = Math.max(vDiff, Math.abs(r.value - ref.value[i]));
    let bestJs = 0, bestRef = 0;
    for (let k = 0; k < legal.length; k++) {
      pDiff = Math.max(pDiff, Math.abs(Math.log(Math.max(r.priors[k], 1e-12)) - Math.log(Math.max(ref.priors[i][k], 1e-12))));
      if (r.priors[k] > r.priors[bestJs]) bestJs = k;
      if (ref.priors[i][k] > ref.priors[i][bestRef]) bestRef = k;
    }
    if (bestJs !== bestRef) rankDiff += 1;
  }
  // and again as one batch, which is how the search actually calls it
  const batch = await ev.evaluateBatch(ref.obs.map((row) => state(row)), ref.obs.map(() => legal));
  let batchDiff = 0;
  for (let i = 0; i < batch.length; i++) {
    batchDiff = Math.max(batchDiff, Math.abs(batch[i].value - singles[i].value));
    for (let k = 0; k < legal.length; k++) {
      batchDiff = Math.max(batchDiff, Math.abs(batch[i].priors[k] - singles[i].priors[k]));
    }
  }
  return { n: ref.obs.length, vDiff, pDiff, rankDiff, batchDiff, adapter: adapter.info ? adapter.info.vendor : "?" };
};
window.__ready = true;
</script></body>`;

function serve() {
  const server = createServer(async (req, res) => {
    const path = decodeURIComponent(req.url.split("?")[0]);
    if (path === "/__probe.html") {
      res.writeHead(200, { "content-type": MIME[".html"] }).end(PROBE);
      return;
    }
    const file = normalize(join(ROOT, path));
    if (!file.startsWith(ROOT)) {
      res.writeHead(403).end();
      return;
    }
    try {
      const body = await readFile(file);
      res.writeHead(200, { "content-type": MIME[extname(file)] || "application/octet-stream" });
      res.end(body);
    } catch {
      res.writeHead(404).end("not found");
    }
  });
  return new Promise((r) => server.listen(0, "127.0.0.1", () => r(server)));
}

class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.waiting = new Map();
    ws.addEventListener("message", (event) => {
      const msg = JSON.parse(event.data);
      if (msg.id && this.waiting.has(msg.id)) {
        const { resolve, reject } = this.waiting.get(msg.id);
        this.waiting.delete(msg.id);
        if (msg.error) reject(new Error(msg.error.message));
        else resolve(msg.result);
      }
    });
  }
  static async connect(url) {
    const ws = new WebSocket(url);
    await new Promise((resolve, reject) => {
      ws.addEventListener("open", resolve, { once: true });
      ws.addEventListener("error", () => reject(new Error("CDP connect failed")), { once: true });
    });
    return new Cdp(ws);
  }
  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.waiting.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
  async eval(expression) {
    const r = await this.send("Runtime.evaluate", {
      expression: `(async () => { ${expression} })()`,
      awaitPromise: true,
      returnByValue: true,
    });
    if (r.exceptionDetails) {
      throw new Error(
        "page threw: " +
          (r.exceptionDetails.exception?.description || r.exceptionDetails.text)
      );
    }
    return r.result.value;
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(label, fn, ms) {
  const deadline = Date.now() + ms;
  for (;;) {
    let v = null;
    try {
      v = await fn();
    } catch {}
    if (v) return v;
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${label}`);
    await sleep(250);
  }
}

/** The reference, trimmed: 100 × 182 floats survive a CDP round trip, barely. */
function reference() {
  const ref = JSON.parse(
    gunzipSync(readFileSync(join(HERE, "fixtures", "torch_reference.json.gz"))).toString("utf8")
  );
  // A fixed legal set keeps the comparison about the net, not about masking;
  // the softmax over it is what the search consumes, so that is what is checked.
  const legal = [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 179];
  const priors = ref.policy.map((logits) => {
    let max = -Infinity;
    for (const a of legal) max = Math.max(max, logits[a]);
    const e = legal.map((a) => Math.exp(logits[a] - max));
    const sum = e.reduce((x, y) => x + y, 0);
    return e.map((x) => x / sum);
  });
  const take = Math.min(40, ref.obs.length); // enough to catch a broken kernel
  return {
    obs: ref.obs.slice(0, take),
    value: ref.value.slice(0, take),
    priors: priors.slice(0, take),
    legal,
    checkpoint: ref.checkpoint,
  };
}

async function main() {
  const ref = reference();
  const server = await serve();
  const url = `http://127.0.0.1:${server.address().port}/__probe.html`;
  const profile = await mkdtemp(join(tmpdir(), "ludometer-webgpu-"));
  const chrome = spawn(
    CHROME,
    [
      "--headless=new",
      "--no-first-run",
      "--no-default-browser-check",
      "--remote-debugging-port=0",
      `--user-data-dir=${profile}`,
      "about:blank",
    ],
    { stdio: ["ignore", "pipe", "pipe"] }
  );
  let devtools = null;
  chrome.stderr.on("data", (chunk) => {
    const m = String(chunk).match(/DevTools listening on (ws:\/\/\S+)/);
    if (m) devtools = m[1];
  });
  const cleanup = async () => {
    chrome.kill();
    server.close();
    await rm(profile, { recursive: true, force: true }).catch(() => {});
  };

  try {
    await until("chrome to start", async () => devtools, 20000);
    const browser = await Cdp.connect(devtools);
    const { targetId } = await browser.send("Target.createTarget", { url: "about:blank" });
    const page = await Cdp.connect(
      devtools.replace(/\/devtools\/browser\/.*/, `/devtools/page/${targetId}`)
    );
    await page.send("Runtime.enable");
    await page.send("Page.enable");
    await page.send("Page.navigate", { url });
    await until("the probe page", () => page.eval("return window.__ready === true;"), 60000);
    await page.send("Runtime.evaluate", {
      expression: `window.__ref = ${JSON.stringify(ref)};`,
      returnByValue: true,
    });
    const out = await page.eval("return await window.__probe();");
    if (out.skip) {
      console.log(`webgpu parity skipped — ${out.skip}`);
      return 0;
    }
    console.log(
      `webgpu vs torch on ${out.n} positions (${ref.checkpoint}, adapter ${out.adapter}): ` +
        `value max |diff| ${out.vDiff.toExponential(2)}, log-prior ${out.pDiff.toExponential(2)}, ` +
        `batch vs single ${out.batchDiff.toExponential(2)}, top-move disagreements ${out.rankDiff}`
    );
    let bad = 0;
    if (out.vDiff > 2e-3) {
      console.error(`FAIL — value gap ${out.vDiff} exceeds 2e-3`);
      bad += 1;
    }
    if (out.pDiff > 5e-2) {
      console.error(`FAIL — log-prior gap ${out.pDiff} exceeds 5e-2`);
      bad += 1;
    }
    if (out.rankDiff > 0) {
      console.error(`FAIL — the GPU ranks a different move first on ${out.rankDiff} position(s)`);
      bad += 1;
    }
    if (out.batchDiff > 1e-3) {
      console.error(`FAIL — a batched run disagrees with one-at-a-time by ${out.batchDiff}`);
      bad += 1;
    }
    if (bad) return 1;
    console.log("webgpu parity ok: the GPU backend plays the same net as the CPU one");
    return 0;
  } catch (err) {
    console.error("FAIL —", err.message);
    return 1;
  } finally {
    await cleanup();
  }
}

process.exit(await main());
