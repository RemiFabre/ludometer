#!/usr/bin/env node
/* Record a whole game of the browser player as frames, for a social clip.
 *
 *   node scripts/game_gif.mjs --out /tmp/frames [--fps 8] [--scale 2] [--think 0] [--speed 1]
 *
 * Serves web/player/ locally, opens it in headless Chrome, lets the strongest
 * net play BOTH seats (the human's moves come from "Suggest a move" + "Play
 * this move", so the game is Porcelain against itself), and saves a PNG every
 * 1/fps seconds together with frames.json: per frame the time, whether the
 * scoring panel is showing, whether the game is over, and the tile count on
 * the table; plus the screen rectangles of the two boards and the middle, so
 * a cut or a zoom can be placed on a board without guessing pixels.
 * scripts/game_gif_render.sh turns the frames into the clips.
 */
import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { readFile, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, normalize, extname } from "node:path";
import { tmpdir } from "node:os";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, "..");
const CHROME = process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const flag = (name, fallback) => {
  const i = process.argv.indexOf("--" + name);
  return i >= 0 && process.argv[i + 1] !== undefined ? process.argv[i + 1] : fallback;
};
const OUT = flag("out", "/tmp/game-frames");
const FPS = Number(flag("fps", 8));
const SCALE = Number(flag("scale", 2));
const THINK = flag("think", "0");
const SPEED = flag("speed", "1");
const WIDTH = Number(flag("width", 1500));
const HEIGHT = Number(flag("height", 1180));
const MAX_S = Number(flag("max-s", 420));

const MIME = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css",
  ".json": "application/json", ".onnx": "application/octet-stream", ".wasm": "application/wasm",
  ".png": "image/png", ".svg": "image/svg+xml", ".gz": "application/gzip", ".woff2": "font/woff2" };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function serve(root) {
  const server = createServer(async (req, res) => {
    let path = decodeURIComponent(req.url.split("?")[0]);
    if (path.endsWith("/")) path += "index.html";
    const file = normalize(join(root, path));
    if (!file.startsWith(root)) return void res.writeHead(403).end();
    try {
      const body = await readFile(file);
      res.writeHead(200, { "content-type": MIME[extname(file)] || "application/octet-stream", "content-length": body.length });
      res.end(body);
    } catch { res.writeHead(404).end("not found"); }
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

class Cdp {
  constructor(ws) { this.ws = ws; this.id = 0; this.waiting = new Map();
    ws.addEventListener("message", (ev) => { const m = JSON.parse(ev.data); if (m.id && this.waiting.has(m.id)) { const { resolve, reject } = this.waiting.get(m.id); this.waiting.delete(m.id); m.error ? reject(new Error(m.error.message)) : resolve(m.result); } }); }
  static async connect(url) { const ws = new WebSocket(url); await new Promise((res, rej) => { ws.addEventListener("open", res, { once: true }); ws.addEventListener("error", () => rej(new Error("CDP connect failed")), { once: true }); }); return new Cdp(ws); }
  send(method, params = {}) { const id = ++this.id; return new Promise((resolve, reject) => { this.waiting.set(id, { resolve, reject }); this.ws.send(JSON.stringify({ id, method, params })); }); }
  async eval(expression) { const r = await this.send("Runtime.evaluate", { expression: `(async () => { ${expression} })()`, awaitPromise: true, returnByValue: true }); if (r.exceptionDetails) throw new Error("page threw: " + (r.exceptionDetails.exception?.description || r.exceptionDetails.text)); return r.result.value; }
}

async function launchChrome() {
  const profile = await mkdtemp(join(tmpdir(), "ludometer-gif-"));
  const chrome = spawn(CHROME, ["--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
    "--remote-debugging-port=0", `--user-data-dir=${profile}`, `--window-size=${WIDTH},${HEIGHT}`, "about:blank"], { stdio: ["ignore", "pipe", "pipe"] });
  let url = null;
  chrome.stderr.on("data", (c) => { const m = String(c).match(/DevTools listening on (ws:\/\/\S+)/); if (m) url = m[1]; });
  const t0 = Date.now(); while (!url && Date.now() - t0 < 20000) await sleep(100);
  if (!url) throw new Error("chrome did not start");
  const browser = await Cdp.connect(url);
  const { targetId } = await browser.send("Target.createTarget", { url: "about:blank" });
  const page = await Cdp.connect(url.replace(/\/devtools\/browser\/.*/, `/devtools/page/${targetId}`));
  await page.send("Page.enable"); await page.send("Runtime.enable");
  await page.send("Emulation.setDeviceMetricsOverride", { width: WIDTH, height: HEIGHT, deviceScaleFactor: SCALE, mobile: false });
  return { page, cleanup: async () => { chrome.kill(); await rm(profile, { recursive: true, force: true }).catch(() => {}); } };
}

const STATE = `return {
  locked: document.body.classList.contains("locked"),
  ready: document.getElementById("engine-bar").classList.contains("ready"),
  scoring: !document.getElementById("scoring").hidden,
  final: !!document.querySelector("#scoring.final"),
  confirm: !!document.getElementById("confirm-bar") && !document.getElementById("confirm-bar").hidden,
  counts: (document.getElementById("counts") || {}).textContent || "",
  tiles: document.querySelectorAll("#middle .tile").length,
  scores: [...document.querySelectorAll(".score-value, .scoreboard .score")].map((e) => e.textContent).slice(0, 2),
};`;
const RECTS = `const r = (id) => { const e = document.querySelector(id); if (!e) return null; const b = e.getBoundingClientRect(); return { x: b.x, y: b.y, w: b.width, h: b.height }; };
  return { human: r("#board-human"), ai: r("#board-ai"), middle: r("#middle"), scoring: r("#scoring"), status: r("#status"), log: r("#log") };`;

async function main() {
  await mkdir(OUT, { recursive: true });
  const server = await serve(join(REPO, "web", "player"));
  const { page, cleanup } = await launchChrome();
  const frames = [];
  let n = 0;
  const shot = async (state) => {
    const { data } = await page.send("Page.captureScreenshot", { format: "png" });
    const name = `${String(n).padStart(5, "0")}.png`;
    await writeFile(join(OUT, name), Buffer.from(data, "base64"));
    frames.push({ i: n, t: (Date.now() - t0) / 1000, ...state });
    n += 1;
  };
  try {
    await page.send("Page.navigate", { url: `http://127.0.0.1:${server.address().port}/index.html` });
    for (let i = 0; i < 2400; i++) { if ((await page.eval(STATE)).ready) break; await sleep(100); }
    await page.eval(`const s = document.getElementById("think"); s.value = "${THINK}"; s.dispatchEvent(new Event("change", { bubbles: true })); return s.value;`);
    await page.eval(`const b = document.querySelector('.speed[data-speed="${SPEED}"]'); if (b) b.click(); return true;`);
    await page.eval(`const box = document.getElementById("coach"); if (box && box.checked) { box.checked = false; box.dispatchEvent(new Event("change")); } return true;`);
    // no "Validate this move" bar in the clip: confirm mode off, and dismiss the news line
    await page.eval(`const c = document.querySelector('.flag[data-confirm="false"]'); if (c) c.click(); const x = document.getElementById("news-close"); if (x) x.click(); return true;`);
    await page.eval(`document.getElementById("deal").click(); return true;`);
    await sleep(800);
    const rects = await page.eval(RECTS);
    await writeFile(join(OUT, "rects.json"), JSON.stringify({ scale: SCALE, width: WIDTH, height: HEIGHT, ...rects }, null, 1));
    var t0 = Date.now();
    const period = 1000 / FPS;
    let next = Date.now();
    let waitingHint = false;
    let lastTiles = -1;
    while (Date.now() - t0 < MAX_S * 1000) {
      const state = await page.eval(STATE);
      if (Date.now() >= next) { await shot(state); next += period; }
      if (state.final) { for (let k = 0; k < FPS * 3; k++) { await sleep(period); await shot(await page.eval(STATE)); } break; }
      if (!state.locked) {
        // the suggestion picks the tiles and opens one row; the human still
        // has to click that row, then "Play this move"
        const step = await page.eval(`
          const bar = document.getElementById("confirm-bar");
          if (bar && !bar.hidden) { document.getElementById("confirm").click(); return "confirmed"; }
          const row = document.querySelector("#board-human .line.open") || document.querySelector("#board-human .floor.open");
          if (row) { row.click(); return "row"; }
          return "none";`);
        if (step === "confirmed") waitingHint = false;
        else if (step === "none" && !waitingHint) { await page.eval(`document.getElementById("hint").click(); return true;`); waitingHint = true; }
        else if (step === "none" && waitingHint && state.tiles !== lastTiles) waitingHint = false;
        lastTiles = state.tiles;
      }
      await sleep(Math.max(5, Math.min(period / 2, next - Date.now())));
    }
    await writeFile(join(OUT, "frames.json"), JSON.stringify(frames));
    const scoringStarts = frames.filter((f, i) => f.scoring && !(frames[i - 1] || {}).scoring).map((f) => f.i);
    console.log(`${n} frames at ${FPS} fps over ${((Date.now() - t0) / 1000).toFixed(0)} s -> ${OUT}; scoring starts at frames ${scoringStarts.join(", ")}; final: ${frames.at(-1)?.final}`);
  } finally {
    await cleanup();
    server.close();
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
