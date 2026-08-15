/* Drive both Azul tables in headless Chrome, over the DevTools protocol.
 *
 * There are two of them — the local Flask GUI (web/play/) and the hosted player
 * (web/player/) — and they share ui/, so a regression in the kit shows up in
 * both. This test opens each one for real and checks the things that a unit
 * test cannot see: that tiles actually move, that the settings panel governs
 * how fast, that ← and → walk the game, that the log is drawn as tiles newest
 * first, and that a filled square and an empty one do not look alike.
 *
 * ── why this file exists ─────────────────────────────────────────────────────
 * The GUI shipped for weeks with animations that worked in every test and never
 * once on the author's machine. `animate.js` returned early under
 * `prefers-reduced-motion`, board.css forced `transition-duration: .001ms`, and
 * macOS ships "Reduce motion" switched on for a lot of people — while the test
 * harness set `no-preference` before it looked. So every motion check here runs
 * **twice**, once in each state, and both must pass.
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *   node web/play/test/gui.test.mjs                  # both pages
 *   node web/play/test/gui.test.mjs --only play      # just the local GUI
 *   node web/play/test/gui.test.mjs --only player    # just the hosted player
 *   node web/play/test/gui.test.mjs --shots DIR      # also save screenshots
 *
 * The local GUI half starts `ludometer-gui` itself, niced, on a spare port.
 */
import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { readFile, mkdtemp, rm, writeFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, normalize, extname } from "node:path";
import { tmpdir } from "node:os";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, "..", "..", "..");
const CHROME =
  process.env.CHROME_PATH || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

function flag(name, fallback = null) {
  const i = process.argv.indexOf("--" + name);
  return i >= 0 && process.argv[i + 1] !== undefined ? process.argv[i + 1] : fallback;
}
const ONLY = flag("only");
const SHOTS = flag("shots");

const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8", ".wasm": "application/wasm",
  ".onnx": "application/octet-stream", ".md": "text/plain; charset=utf-8",
};

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function until(label, fn, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = await fn();
    if (value) return value;
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${label} (${timeoutMs}ms)`);
    await sleep(200);
  }
}

function serve(root) {
  const server = createServer(async (req, res) => {
    let path = decodeURIComponent(req.url.split("?")[0]);
    if (path.endsWith("/")) path += "index.html";
    const file = normalize(join(root, path));
    if (!file.startsWith(root)) return void res.writeHead(403).end();
    try {
      const body = await readFile(file);
      res.writeHead(200, {
        "content-type": MIME[extname(file)] || "application/octet-stream",
        "content-length": body.length,
      });
      res.end(body);
    } catch {
      res.writeHead(404).end("not found");
    }
  });
  return new Promise((r) => server.listen(0, "127.0.0.1", () => r(server)));
}

/* ------------------------------------------------------------------------- CDP */
class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.waiting = new Map();
    this.listeners = [];
    ws.addEventListener("message", (event) => {
      const msg = JSON.parse(event.data);
      if (msg.id && this.waiting.has(msg.id)) {
        const { resolve, reject } = this.waiting.get(msg.id);
        this.waiting.delete(msg.id);
        if (msg.error) reject(new Error(msg.error.message));
        else resolve(msg.result);
      } else if (msg.method) this.listeners.forEach((fn) => fn(msg));
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
  on(fn) { this.listeners.push(fn); }
  async eval(expression) {
    const result = await this.send("Runtime.evaluate", {
      expression: `(async () => { ${expression} })()`,
      awaitPromise: true,
      returnByValue: true,
    });
    if (result.exceptionDetails) {
      const d = result.exceptionDetails;
      throw new Error("page threw: " + (d.exception?.description || d.text));
    }
    return result.result.value;
  }
  async shot(path) {
    const { data } = await this.send("Page.captureScreenshot", {
      format: "png", captureBeyondViewport: true,
    });
    await writeFile(path, Buffer.from(data, "base64"));
  }
}

async function launchChrome() {
  const profile = await mkdtemp(join(tmpdir(), "ludometer-chrome-"));
  const chrome = spawn(
    CHROME,
    ["--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
     "--remote-debugging-port=0", `--user-data-dir=${profile}`,
     "--window-size=1500,1180", "about:blank"],
    { stdio: ["ignore", "pipe", "pipe"] }
  );
  let devtoolsUrl = null;
  chrome.stderr.on("data", (chunk) => {
    const m = String(chunk).match(/DevTools listening on (ws:\/\/\S+)/);
    if (m) devtoolsUrl = m[1];
  });
  await until("chrome to start", async () => devtoolsUrl, 20000);
  const browser = await Cdp.connect(devtoolsUrl);
  const { targetId } = await browser.send("Target.createTarget", { url: "about:blank" });
  const page = await Cdp.connect(
    devtoolsUrl.replace(/\/devtools\/browser\/.*/, `/devtools/page/${targetId}`)
  );
  const errors = [];
  page.on((msg) => {
    if (msg.method === "Runtime.consoleAPICalled" && msg.params.type === "error") {
      errors.push("console.error: " + (msg.params.args || []).map((a) => a.value ?? a.description ?? "").join(" "));
    } else if (msg.method === "Runtime.exceptionThrown") {
      const d = msg.params.exceptionDetails;
      errors.push("uncaught: " + (d.exception?.description || d.text));
    } else if (msg.method === "Log.entryAdded" && msg.params.entry.level === "error") {
      // On a cold start the page asks /api/state before a game exists and deals
      // one on the 409. Chrome logs every non-2xx response whether or not the
      // page handled it; that one is the page working as designed.
      if (!/409/.test(msg.params.entry.text)) errors.push("log: " + msg.params.entry.text);
    }
  });
  await page.send("Runtime.enable");
  await page.send("Log.enable");
  await page.send("Page.enable");
  await page.send("Emulation.setDeviceMetricsOverride", {
    width: 1500, height: 1180, deviceScaleFactor: 1, mobile: false,
  });
  return {
    page,
    errors,
    cleanup: async () => {
      chrome.kill();
      await rm(profile, { recursive: true, force: true }).catch(() => {});
    },
  };
}

const setMotion = (page, value) =>
  page.send("Emulation.setEmulatedMedia", {
    features: [{ name: "prefers-reduced-motion", value }],
  });

/* ---------------------------------------------------------------- page scripts */
/* Watch the flight layer: a straight-line flight is a clone appended to it, so
 * counting clones is counting the animation, and their computed transition
 * duration is how long a tile is actually in the air. `azul:animated` fires once
 * per move the turn sequence plays out, which is how "the AI's second move
 * across a round boundary is animated too" gets checked rather than assumed. */
const INSTRUMENT = `
  window.__anim = { flights: 0, durations: [], moves: [] };
  new MutationObserver((records) => {
    records.forEach((r) => r.addedNodes.forEach((n) => {
      window.__anim.flights += 1;
      const d = getComputedStyle(n).transitionDuration;
      window.__anim.durations.push(Math.round(parseFloat(d) * 1000));
    }));
  }).observe(document.getElementById("fly"), { childList: true });
  // score pop-ups live in their own layer, counted apart from tile flights;
  // a page without the layer keeps __pops undefined so its check skips
  const popLayer = document.getElementById("pops");
  if (popLayer) {
    window.__pops = 0;
    new MutationObserver((records) => {
      records.forEach((r) => { window.__pops += r.addedNodes.length; });
    }).observe(popLayer, { childList: true });
  }
  document.addEventListener("azul:animated", (e) => {
    window.__anim.moves.push({
      ply: e.detail.ply, side: e.detail.side, at: window.__anim.flights,
    });
  });
  return true;
`;

const RESET = `window.__anim.flights = 0; window.__anim.durations = []; window.__anim.moves = []; return true;`;

const PLAY_ONE = `
  if (document.body.classList.contains("locked")) return "busy";
  const tile = document.querySelector("#middle button.tile:not([disabled])");
  if (!tile) return "wait";
  tile.click();
  const row = document.querySelector("#board-human .line.open") ||
              document.querySelector("#board-human .floor.open");
  if (!row) { document.getElementById("cancel").click(); return "blocked"; }
  row.click();
  // confirm mode (on by default on pages that have it): press "Play this move"
  const bar = document.getElementById("confirm-bar");
  if (bar && !bar.hidden) document.getElementById("confirm").click();
  return "played";
`;

/** Play `n` human moves, waiting for each turn to finish. */
async function playMoves(page, n, timeoutMs = 120000) {
  let played = 0;
  const deadline = Date.now() + timeoutMs;
  while (played < n && Date.now() < deadline) {
    const done = await page.eval('return !!document.querySelector("#scoring.final");');
    if (done) break;
    const outcome = await page.eval(PLAY_ONE);
    if (outcome === "played") {
      played += 1;
      await until("the turn to finish", () =>
        page.eval('return !document.body.classList.contains("locked");'), 90000);
    } else {
      await sleep(250);
    }
  }
  return played;
}

const setSpeed = (page, value) => page.eval(`
  const b = document.querySelector('.speed[data-speed="${value}"]');
  if (!b) throw new Error("no ${value}x speed button");
  b.click();
  return document.body.dataset.anim;
`);

/* ------------------------------------------------------------------ assertions */
const median = (xs) => {
  const s = xs.slice().sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
};

/**
 * Tiles must move at this speed, in this reduced-motion state.
 *
 * `expectedMs` is what one flight should last at 1×; the check is loose (±35%)
 * because it is measuring a rendered transition, not arithmetic, but it is
 * tight enough to tell 460 ms from 230 ms and either from zero.
 */
async function checkFlights(page, label, expectedMs, errors) {
  await page.eval(RESET);
  const played = await playMoves(page, 1);
  if (!played) return errors.push(`${label}: could not play a move`);
  const anim = await page.eval("return window.__anim;");
  if (expectedMs === 0) {
    if (anim.flights) errors.push(`${label}: animation is off but ${anim.flights} tiles flew`);
    if (!anim.moves.length) errors.push(`${label}: the turn sequence skipped the move`);
    return;
  }
  if (!anim.flights) {
    return errors.push(`${label}: NO TILES FLEW — the flight layer stayed empty`);
  }
  const got = median(anim.durations);
  const low = expectedMs * 0.65;
  const high = expectedMs * 1.35;
  if (got < low || got > high) {
    errors.push(`${label}: flights lasted ${got}ms, expected about ${expectedMs}ms`);
  }
  console.log(`    ${label}: ${anim.flights} tiles flew, ${got}ms each`);
}

/** Filled and empty squares must not look alike — see ui/THEMING.md. */
async function checkContrast(page, label, errors) {
  const read = await page.eval(`
    const px = (s) => {
      const m = (s || "").match(/[\\d.]+/g) || [0, 0, 0];
      return [Number(m[0]) / 255, Number(m[1]) / 255, Number(m[2]) / 255];
    };
    const chroma = (c) => Math.max(...c) - Math.min(...c);
    const luma = (c) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    // split a box-shadow list on the commas that separate shadows, not the ones
    // inside rgb(...) — a raised square has at least one shadow that is not inset
    const layers = (s) => (s || "").split(/,(?![^(]*\\))/);
    const look = (el) => {
      const cs = getComputedStyle(el);
      const c = px(cs.backgroundColor);
      return {
        chroma: chroma(c), luma: luma(c),
        shadow: cs.boxShadow,
        inset: layers(cs.boxShadow).some((p) => /inset/.test(p)),
        outer: layers(cs.boxShadow).some((p) => !/inset/.test(p) && /rgb/.test(p)),
      };
    };
    const out = { tiles: {}, cells: {}, slots: [], panel: null };
    for (let c = 0; c < 5; c++) {
      const t = document.querySelector('.board .tile[data-color="' + c + '"]') ||
                document.querySelector('#middle .tile[data-color="' + c + '"]');
      const e = document.querySelector('.board .cell.empty[data-color="' + c + '"]');
      if (t) out.tiles[c] = look(t);
      if (e) out.cells[c] = look(e);
    }
    out.slots = [...document.querySelectorAll(".board .slot")].slice(0, 4).map(look);
    out.panel = look(document.querySelector("#board-human"));
    // the wall's colour hint: the outlined diamond inside an empty square must
    // be saturated, thick and near-opaque — visible colour on a neutral well
    out.motifs = {};
    for (let c = 0; c < 5; c++) {
      const m = document.querySelector('.board .cell.empty[data-color="' + c + '"]');
      if (!m) continue;
      const cs = getComputedStyle(m, "::after");
      out.motifs[c] = {
        chroma: chroma(px(cs.borderTopColor)),
        width: parseFloat(cs.borderTopWidth) || 0,
        opacity: parseFloat(cs.opacity) || 0,
      };
    }
    return out;
  `);

  const empties = [...Object.values(read.cells), ...read.slots];
  if (empties.length < 5) errors.push(`${label}: found only ${empties.length} empty squares`);
  empties.forEach((e, i) => {
    // an empty square carries no glaze: that is the whole rule of the palette
    if (e.chroma > 0.15) {
      errors.push(`${label}: empty square #${i} is tinted (chroma ${e.chroma.toFixed(2)} > 0.15)`);
    }
    if (!e.inset) errors.push(`${label}: empty square #${i} is not recessed (no inset shadow)`);
    if (read.panel && read.panel.luma - e.luma < 0.1) {
      errors.push(`${label}: empty square #${i} does not sit below the board surface`);
    }
  });

  Object.entries(read.tiles).forEach(([c, t]) => {
    const chromatic = c !== "3"; // charcoal is a glaze that happens to be grey
    if (chromatic && t.chroma < 0.3) {
      errors.push(`${label}: tile ${c} is washed out (chroma ${t.chroma.toFixed(2)} < 0.30)`);
    }
    if (!t.outer) errors.push(`${label}: tile ${c} is not raised (no outer shadow)`);
    const e = read.cells[c];
    if (!e) return;
    const dc = Math.abs(t.chroma - e.chroma);
    const dl = Math.abs(t.luma - e.luma);
    if (dc < 0.3 && dl < 0.3) {
      errors.push(
        `${label}: a filled and an empty ${c} square are too close ` +
          `(Δchroma ${dc.toFixed(2)}, Δluma ${dl.toFixed(2)})`
      );
    }
  });
  Object.entries(read.motifs).forEach(([c, m]) => {
    const chromatic = c !== "3"; // charcoal's motif is a grey by design
    if (chromatic && m.chroma < 0.25) {
      errors.push(`${label}: wall motif ${c} carries no colour (chroma ${m.chroma.toFixed(2)} < 0.25)`);
    }
    if (m.width < 2) errors.push(`${label}: wall motif ${c} is too thin (${m.width}px < 2px)`);
    if (m.opacity < 0.8) errors.push(`${label}: wall motif ${c} is too faint (opacity ${m.opacity} < 0.8)`);
  });
  const worst = Object.entries(read.tiles)
    .filter(([c]) => read.cells[c])
    .map(([c, t]) => Math.max(Math.abs(t.chroma - read.cells[c].chroma), Math.abs(t.luma - read.cells[c].luma)))
    .reduce((a, b) => Math.min(a, b), 9);
  console.log(`    ${label}: weakest filled/empty separation ${worst.toFixed(2)} (need 0.30)`);
}

/** One theme file: changing the skin must move every glaze. */
async function checkTheme(page, label, errors) {
  const result = await page.eval(`
    const tile = () => getComputedStyle(document.querySelector('.board .tile[data-color="0"]') ||
                                        document.querySelector('#middle .tile[data-color="0"]')).backgroundColor;
    const linen = () => getComputedStyle(document.body).backgroundColor;
    const before = { tile: tile(), linen: linen() };
    document.documentElement.dataset.skin = "dusk";
    const after = { tile: tile(), linen: linen() };
    delete document.documentElement.dataset.skin;
    const back = { tile: tile(), linen: linen() };
    return { before, after, back };
  `);
  if (result.before.tile === result.after.tile) {
    errors.push(`${label}: the "dusk" skin did not change the tiles — colours are not centralised`);
  }
  if (result.before.linen === result.after.linen) {
    errors.push(`${label}: the "dusk" skin did not change the table ground`);
  }
  if (result.before.tile !== result.back.tile) {
    errors.push(`${label}: removing the skin did not restore the default palette`);
  }
}

/** The log: tiles drawn as tiles, newest at the top, uniform entries. */
async function checkLog(page, label, errors) {
  const log = await page.eval(`
    const entries = [...document.querySelectorAll("#log .log-entry")];
    const moves = entries.filter((e) => e.dataset.kind === "move");
    return {
      entries: entries.length,
      moves: moves.length,
      glyphs: document.querySelectorAll("#log .log-entry .glyph").length,
      colourWords: moves.filter((e) => /blue|yellow|red|black|teal/i.test(e.textContent)).length,
      plies: moves.map((e) => Number(e.dataset.ply)),
      places: moves.slice(0, 2).map((e) => [...e.querySelectorAll(".log-where")].map((n) => n.textContent).join(" -> ")),
      // uniformity: every entry drawn the same, including the newest
      distinctStyles: new Set(entries.map((e) => {
        const cs = getComputedStyle(e);
        return [cs.fontWeight, cs.fontSize, cs.backgroundColor, cs.color].join("|");
      })).size,
    };
  `);
  if (log.moves < 2) return errors.push(`${label}: the move log did not fill in`);
  if (!log.glyphs) errors.push(`${label}: the log has no tile glyphs — it is still words`);
  if (log.colourWords) errors.push(`${label}: ${log.colourWords} log entries still spell a colour out`);
  const descending = log.plies.every((p, i) => i === 0 || log.plies[i - 1] > p);
  if (!descending) errors.push(`${label}: the log is not newest-first (plies ${log.plies.join(",")})`);
  if (log.distinctStyles !== 1) {
    errors.push(`${label}: log entries are not drawn uniformly (${log.distinctStyles} styles)`);
  }
  console.log(`    ${label}: ${log.moves} moves, ${log.glyphs} glyphs, newest first (${log.places[0]})`);
}

/**
 * Coach verdicts survive the reversed log.
 *
 * The chip is attached to the entry it is about, not to a position in the list,
 * so drawing newest-first must not move it onto someone else's move. Skipped
 * where the opponent has no search to borrow (the scripted baselines).
 */
async function checkCoach(page, label, errors) {
  const available = await page.eval(`
    const box = document.getElementById("coach");
    if (box.disabled) return false;
    box.checked = true;
    box.dispatchEvent(new Event("change"));
    return true;
  `);
  if (!available) {
    console.log(`    ${label}: coach mode unavailable against this opponent — skipped`);
    return;
  }
  await playMoves(page, 1);
  const chips = await page.eval(`
    const withChip = [...document.querySelectorAll("#log .log-entry")]
      .filter((e) => e.querySelector(".coach-chip"));
    return withChip.map((e) => ({
      kind: e.dataset.kind,
      side: e.dataset.side,
      grade: e.querySelector(".coach-chip").dataset.grade,
      glyphs: e.querySelectorAll(".glyph").length,
      text: e.querySelector(".coach-delta").textContent,
    }));
  `);
  await page.eval(`
    const box = document.getElementById("coach");
    box.checked = false;
    box.dispatchEvent(new Event("change"));
    return true;
  `);
  if (!chips.length) return errors.push(`${label}: coach mode produced no verdict`);
  chips.forEach((c) => {
    if (c.kind !== "move" || c.side !== "human") {
      errors.push(`${label}: a coach chip landed on a ${c.side} ${c.kind} entry`);
    }
    if (!c.grade) errors.push(`${label}: a coach chip has no grade`);
    if (!c.glyphs) errors.push(`${label}: a rated move lost its tile glyphs`);
  });
  console.log(`    ${label}: ${chips.length} coach chip(s), all on your own moves (${chips[0].text})`);
}

/** ← and → walk the game; the live position is untouched while you look. */
async function checkHistory(page, label, errors) {
  const key = (k) => page.eval(`
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "${k}", bubbles: true, cancelable: true }));
    return true;
  `);
  const snap = () => page.eval(`return {
    count: document.querySelector(".nav-count").textContent,
    headline: document.querySelector("#status .status-headline").textContent,
    viewing: document.body.classList.contains("viewing"),
    pickable: document.querySelectorAll("#middle button.tile:not([disabled])").length,
    openRows: document.querySelectorAll("#board-human .line.open").length,
    board: document.querySelector("#board-human .board-grid").textContent,
    logEntries: document.querySelectorAll("#log .log-entry").length,
    liveShown: !document.querySelector(".nav-live").hidden,
  };`);

  const live = await snap();
  await key("ArrowLeft");
  await key("ArrowLeft");
  await sleep(120);
  const past = await snap();

  if (!past.viewing) return errors.push(`${label}: ← did not enter the history`);
  if (!/viewing move \d+ of \d+/i.test(past.headline)) {
    errors.push(`${label}: no "viewing move N of M" indicator, got ${JSON.stringify(past.headline)}`);
  }
  if (past.board === live.board) errors.push(`${label}: the board did not change on ←`);
  if (past.pickable) errors.push(`${label}: tiles are still pickable while browsing the history`);
  if (past.openRows) errors.push(`${label}: rows are still playable while browsing the history`);
  if (past.logEntries >= live.logEntries) {
    errors.push(`${label}: the log was not rewound with the board`);
  }
  if (!past.liveShown) errors.push(`${label}: no way back to the live game is offered`);

  await key("ArrowRight");
  await sleep(120);
  const stepped = await snap();
  if (!stepped.viewing) errors.push(`${label}: → jumped straight out of the history`);
  if (stepped.count === past.count) errors.push(`${label}: → did not advance a move`);

  await key("End");
  await sleep(150);
  const back = await snap();
  if (back.viewing) errors.push(`${label}: End did not return to the live game`);
  if (back.board !== live.board) errors.push(`${label}: the live position changed while browsing`);
  if (back.pickable !== live.pickable) errors.push(`${label}: play did not resume after browsing`);
  console.log(`    ${label}: walked back to ${past.count}, stepped to ${stepped.count}, resumed`);
}

/** Every move the game played must have been animated — both of a double AI move. */
async function checkEveryMoveAnimated(page, label, errors) {
  const result = await page.eval(`
    const anim = window.__anim;
    const animated = anim.moves.map((m) => m.ply).sort((a, b) => a - b);
    // the counters were reset mid-game, so only judge the moves played since
    const from = animated.length ? animated[0] : 0;
    const logged = [...document.querySelectorAll('#log .log-entry[data-kind="move"]')]
      .map((e) => Number(e.dataset.ply)).filter((p) => p >= from).sort((a, b) => a - b);
    const rounds = document.querySelectorAll('#log .log-entry[data-kind="round"]').length;
    // consecutive AI plies with nothing of yours between them = a double move
    const sides = {};
    anim.moves.forEach((m) => { sides[m.ply] = m.side; });
    let doubles = 0;
    animated.forEach((p) => { if (sides[p] === "ai" && sides[p - 1] === "ai") doubles += 1; });
    const silent = anim.moves.filter((m, i) => (i ? m.at - anim.moves[i - 1].at : m.at) === 0);
    return { logged, animated, rounds, doubles, silent: silent.map((m) => m.ply) };
  `);
  const missing = result.logged.filter((p) => !result.animated.includes(p));
  if (missing.length) {
    errors.push(`${label}: ${missing.length} move(s) never animated — plies ${missing.join(",")}`);
  }
  if (result.silent.length) {
    errors.push(`${label}: ${result.silent.length} move(s) produced no tile flights — plies ${result.silent.join(",")}`);
  }
  if (!result.rounds) errors.push(`${label}: the game never reached a round boundary`);
  console.log(
    `    ${label}: ${result.animated.length}/${result.logged.length} moves animated, ` +
      `${result.rounds} round boundary/ies, ${result.doubles} double AI move(s)`
  );
}

/* The v4 behaviours, on pages that have them (the hosted player; the local GUI
 * has no hand tray, pop layer or in-page coach queue yet — it skips cleanly). */

/** Picking a colour must act the selection out at once — and reversibly. */
async function checkHeldPreview(page, label, errors) {
  if (!(await page.eval('return !!document.getElementById("hand");'))) {
    console.log(`    ${label}: no hand tray on this page — skipped`);
    return;
  }
  await setSpeed(page, 1);
  await until("your turn", () => page.eval(`
    return !document.body.classList.contains("locked") &&
      !!document.querySelector("#middle button.tile:not([disabled])");`), 60000);
  await page.eval(`
    window.__anim.flights = 0;
    document.querySelector("#middle button.tile:not([disabled])").click();
    return true;
  `);
  await sleep(900); // let the transient flight land and its clones remove themselves
  const out = await page.eval(`
    return {
      flights: window.__anim.flights,
      hand: !document.getElementById("hand").hidden,
      tiles: document.querySelectorAll("#hand-tiles .tile").length,
      ghosts: document.querySelectorAll("#fly .fly-tile").length,
    };
  `);
  if (!out.flights) errors.push(`${label}: picking a colour flew nothing`);
  if (!out.hand || !out.tiles) errors.push(`${label}: the hand tray did not fill on pick`);
  if (out.ghosts) errors.push(`${label}: flight clones outlived the flight — a fixed clone scrolls apart from the table`);
  await page.eval(`
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true }));
    return true;
  `);
  await sleep(120);
  const back = await page.eval(`
    return {
      hand: !document.getElementById("hand").hidden,
      hidden: [...document.querySelectorAll("#middle .tile")]
        .filter((t) => t.style.visibility === "hidden").length,
      pickable: document.querySelectorAll("#middle button.tile:not([disabled])").length,
    };
  `);
  if (back.hand) errors.push(`${label}: Escape left the hand tray open`);
  if (back.hidden) errors.push(`${label}: Escape left dish tiles hidden`);
  if (!back.pickable) errors.push(`${label}: Escape did not restore the dishes`);
  console.log(`    ${label}: pick flew ${out.flights} tiles into the tray (${out.tiles} held); Escape reset it`);
}

/** With coach mode on, the move must land at once — the verdict follows. */
async function checkCoachImmediate(page, label, errors) {
  if (!(await page.eval('return !!document.getElementById("pops");'))) return;
  const available = await page.eval(`
    const box = document.getElementById("coach");
    if (box.disabled) return false;
    box.checked = true;
    box.dispatchEvent(new Event("change"));
    return true;
  `);
  if (!available) return;
  const before = await page.eval(
    'return document.querySelectorAll(\'#log .log-entry[data-kind="move"]\').length;'
  );
  let outcome = "wait";
  for (let i = 0; i < 20 && outcome !== "played"; i++) {
    outcome = await page.eval(PLAY_ONE);
    if (outcome !== "played") await sleep(250);
  }
  if (outcome !== "played") {
    errors.push(`${label}: could not play a coached move (${outcome})`);
    return;
  }
  const t0 = Date.now();
  let landed = null;
  try {
    await until("the coached move to land", () => page.eval(
      `return document.querySelectorAll('#log .log-entry[data-kind="move"]').length > ${before};`
    ), 5000);
    landed = Date.now() - t0;
  } catch {
    errors.push(`${label}: the coached move never landed`);
  }
  // the old flow rated first (2s search) and only then moved; the new one must
  // put the move on the table in well under a second of search time
  if (landed !== null && landed > 1600) {
    errors.push(`${label}: with coach on, the move took ${landed}ms to land`);
  }
  const grade = await page.eval(`
    const chip = document.querySelector("#log .coach-chip");
    return chip ? chip.dataset.grade : null;
  `);
  if (!grade) errors.push(`${label}: the coached move has no chip at all`);
  await until("the turn to finish", () =>
    page.eval('return !document.body.classList.contains("locked");'), 90000);
  try {
    await until("the coach verdict", () => page.eval(`
      const chip = document.querySelector("#log .coach-chip");
      return chip && chip.dataset.grade !== "pending";`), 30000);
  } catch {
    errors.push(`${label}: the coach verdict never filled in`);
  }
  await page.eval(`
    const box = document.getElementById("coach");
    box.checked = false;
    box.dispatchEvent(new Event("change"));
    return true;
  `);
  console.log(`    ${label}: coached move landed in ${landed}ms, verdict filled in behind it`);
}

/** Round scoring must have popped its numbers where they were earned. */
async function checkScorePops(page, label, errors) {
  const pops = await page.eval("return window.__pops === undefined ? null : window.__pops;");
  if (pops === null) return;
  const rounds = await page.eval(
    'return document.querySelectorAll(\'#log .log-entry[data-kind="round"]\').length;'
  );
  if (rounds && !pops) {
    errors.push(`${label}: ${rounds} round(s) scored but no score pop-up ever appeared`);
  }
  console.log(`    ${label}: ${pops} score pop-up(s) over ${rounds} scored round(s)`);
}

/** Stepping through the history must animate the step. */
async function checkNavAnimation(page, label, errors) {
  if (!(await page.eval('return !!document.getElementById("pops");'))) return;
  await setSpeed(page, 2);
  const key = (k) => page.eval(`
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "${k}", bubbles: true, cancelable: true }));
    return true;
  `);
  await page.eval("window.__anim.flights = 0; return true;");
  // a step that crosses a round boundary redraws without flying (deliberate),
  // so allow a few steps in each direction before concluding nothing animates
  await key("ArrowLeft");
  const entered = await page.eval(
    'return document.body.classList.contains("viewing");'
  );
  await sleep(420);
  let back = await page.eval("return window.__anim.flights;");
  for (let i = 0; i < 3 && !back; i++) {
    await key("ArrowLeft");
    await sleep(420);
    back = await page.eval("return window.__anim.flights;");
  }
  let forward = back;
  for (let i = 0; i < 4 && forward <= back; i++) {
    const browsing = await page.eval('return document.body.classList.contains("viewing");');
    if (!browsing) break;
    await key("ArrowRight");
    await sleep(420);
    forward = await page.eval("return window.__anim.flights;");
  }
  await key("End");
  await sleep(450);
  const stranded = await page.eval(`
    return [...document.querySelectorAll(".board .tile, #middle .tile")]
      .filter((t) => t.style.visibility === "hidden").length;
  `);
  if (!entered) errors.push(`${label}: ← did not enter the history`);
  if (!back) errors.push(`${label}: stepping back animated nothing`);
  if (forward <= back) errors.push(`${label}: stepping forward animated nothing`);
  if (stranded) errors.push(`${label}: a history step left ${stranded} tile(s) hidden`);
  console.log(`    ${label}: history steps flew ${forward} tiles (${back} back, ${forward - back} forward)`);
}

/** Confirm mode: a clicked row draws the position the move would leave — real
 * tiles, the new ones glowing, a banner asking for the word. Play commits;
 * Cancel goes all the way back to before the pick. */
async function checkConfirmMove(page, label, errors) {
  if (!(await page.eval('return !!document.getElementById("confirm-bar");'))) {
    console.log(`    ${label}: no confirm mode on this page — skipped`);
    return;
  }
  await setSpeed(page, 1);
  await until("your turn", () => page.eval(`
    return !document.body.classList.contains("locked") &&
      !!document.querySelector("#middle button.tile:not([disabled])");`), 60000);
  const placed = await page.eval(`
    document.querySelector("#middle button.tile:not([disabled])").click();
    const row = document.querySelector("#board-human .line.open") ||
                document.querySelector("#board-human .floor.open");
    if (!row) return null;
    const logBefore = document.querySelectorAll('#log .log-entry[data-kind="move"]').length;
    row.click();
    return {
      logBefore,
      barShown: !document.getElementById("confirm-bar").hidden,
      glowing: document.querySelectorAll("#board-human .proposed").length,
      openRows: document.querySelectorAll("#board-human .line.open, #board-human .floor.open").length,
      locked: document.body.classList.contains("locked"),
    };
  `);
  if (!placed) return errors.push(`${label}: no open row to place a move on`);
  if (!placed.barShown) errors.push(`${label}: placing a move raised no confirm banner`);
  if (!placed.glowing) errors.push(`${label}: the placed tiles do not glow`);
  if (placed.openRows) errors.push(`${label}: rows are still clickable under a placed move`);
  if (placed.locked) errors.push(`${label}: the move committed without confirmation`);
  await sleep(900); // let the placement flight land and its clones go
  const shown = await page.eval(`return {
    log: document.querySelectorAll('#log .log-entry[data-kind="move"]').length,
    ghosts: document.querySelectorAll("#fly .fly-tile").length,
    hidden: [...document.querySelectorAll("#board-human .tile, #middle .tile")]
      .filter((t) => t.style.visibility === "hidden").length,
  };`);
  if (shown.log > placed.logBefore) errors.push(`${label}: a placed move reached the log early`);
  if (shown.ghosts) errors.push(`${label}: a placed move left clones parked over the board`);
  if (shown.hidden) errors.push(`${label}: a placed move left real tiles hidden`);
  // Cancel: all the way back, as if nothing had been picked
  await page.eval('document.getElementById("confirm-cancel").click(); return true;');
  await sleep(200);
  const back = await page.eval(`return {
    barShown: !document.getElementById("confirm-bar").hidden,
    glowing: document.querySelectorAll("#board-human .proposed").length,
    openRows: document.querySelectorAll("#board-human .line.open, #board-human .floor.open").length,
    hand: !document.getElementById("hand").hidden,
    pickable: document.querySelectorAll("#middle button.tile:not([disabled])").length,
    log: document.querySelectorAll('#log .log-entry[data-kind="move"]').length,
  };`);
  if (back.barShown) errors.push(`${label}: Cancel left the banner up`);
  if (back.glowing) errors.push(`${label}: Cancel left tiles glowing`);
  if (back.openRows || back.hand) errors.push(`${label}: Cancel did not go all the way back`);
  if (!back.pickable) errors.push(`${label}: Cancel did not restore the dishes`);
  if (back.log > placed.logBefore) errors.push(`${label}: Cancel still played the move`);
  // place again, and this time play it
  const committed = await page.eval(`
    document.querySelector("#middle button.tile:not([disabled])").click();
    const row = document.querySelector("#board-human .line.open") ||
                document.querySelector("#board-human .floor.open");
    if (!row) return false;
    row.click();
    document.getElementById("confirm").click();
    return document.body.classList.contains("locked");
  `);
  if (!committed) errors.push(`${label}: confirming a placed move did not play it`);
  await until("the confirmed turn to finish", () =>
    page.eval('return !document.body.classList.contains("locked");'), 90000);
  const final = await page.eval(
    'return document.querySelectorAll(\'#log .log-entry[data-kind="move"]\').length;'
  );
  if (final <= placed.logBefore) errors.push(`${label}: the confirmed move never reached the log`);
  // and with the switch off, a row click commits at once
  await page.eval(`
    document.querySelector('.flag[data-confirm="false"]').click();
    return true;
  `);
  await until("your turn", () => page.eval(`
    return !document.body.classList.contains("locked") &&
      !!document.querySelector("#middle button.tile:not([disabled])");`), 60000);
  const direct = await page.eval(`
    document.querySelector("#middle button.tile:not([disabled])").click();
    const row = document.querySelector("#board-human .line.open") ||
                document.querySelector("#board-human .floor.open");
    if (!row) return null;
    row.click();
    return {
      locked: document.body.classList.contains("locked"),
      barShown: !document.getElementById("confirm-bar").hidden,
    };
  `);
  if (direct && !direct.locked) errors.push(`${label}: with confirm off, the row click did not play`);
  if (direct && direct.barShown) errors.push(`${label}: with confirm off, the banner appeared`);
  await until("the direct turn to finish", () =>
    page.eval('return !document.body.classList.contains("locked");'), 90000);
  await page.eval('document.querySelector(\'.flag[data-confirm="true"]\').click(); return true;');
  console.log(`    ${label}: place → cancel-all → place → confirm behaved; off-switch commits at once`);
}

/* ------------------------------------------------------------------- the pages */
async function checkPage({ name, url, deal, errors, shots }) {
  console.log(`\n== ${name} — ${url}`);
  const { page, errors: pageErrors, cleanup } = await launchChrome();
  try {
    await page.send("Page.navigate", { url });
    await deal(page);
    await page.eval(INSTRUMENT);

    // 1. Motion, in both reduced-motion states. This is the regression that
    //    matters: on a machine with "Reduce motion" on, tiles used to teleport.
    for (const motion of ["reduce", "no-preference"]) {
      await setMotion(page, motion);
      await setSpeed(page, 1);
      await checkFlights(page, `${name} · OS motion=${motion} · 1x`, 460, errors);
    }

    // 2. The setting, not the OS, decides how fast — and whether at all.
    await setMotion(page, "reduce");
    await setSpeed(page, 2);
    await checkFlights(page, `${name} · 2x`, 230, errors);
    await setSpeed(page, 0.5);
    await checkFlights(page, `${name} · 0.5x`, 920, errors);
    await setSpeed(page, 0);
    await checkFlights(page, `${name} · off`, 0, errors);

    const stored = await page.eval(`return {
      value: window.localStorage.getItem("ludometer.anim.speed"),
      body: document.body.dataset.anim,
      panel: !document.querySelector(".settings-panel").hidden,
      presets: [...document.querySelectorAll(".speed")].map((b) => b.textContent),
      pressed: document.querySelector('.speed[aria-pressed="true"]').textContent,
    };`);
    if (stored.value !== "0") errors.push(`${name}: the speed was not persisted (${stored.value})`);
    if (stored.body !== "off") errors.push(`${name}: body[data-anim] is ${stored.body}, expected off`);
    if (stored.presets.join(",") !== "Off,0.5×,1×,2×") {
      errors.push(`${name}: unexpected speed presets ${stored.presets.join(",")}`);
    }
    if (stored.pressed !== "Off") errors.push(`${name}: the panel does not show the chosen speed`);
    await setSpeed(page, 1);

    // 3. The look, and the theme behind it.
    await checkContrast(page, name, errors);
    await checkTheme(page, name, errors);

    // 4. A stretch of game at speed, then the log and the navigator.
    await setSpeed(page, 2);
    await page.eval(RESET);
    const played = await playMoves(page, 14, 240000);
    console.log(`    ${name}: played ${played} moves`);
    await checkEveryMoveAnimated(page, name, errors);
    await checkLog(page, name, errors);
    await checkCoach(page, name, errors);
    await checkHistory(page, name, errors);

    // 4b. The v4 behaviours: the acted-out selection, the coach that never
    //     blocks a move, score pop-ups, and animated history steps.
    await checkScorePops(page, name, errors);
    await checkNavAnimation(page, name, errors);
    await checkHeldPreview(page, name, errors);
    await checkConfirmMove(page, name, errors);
    await checkCoachImmediate(page, name, errors);

    // 5. Nothing may cover the board, settings panel included.
    const overlays = await page.eval(`
      return [...document.querySelectorAll(".overlay, .sheet, [role='dialog'], [aria-modal='true']")]
        .map((n) => n.tagName + "." + (n.className || "").toString().split(" ")[0]);
    `);
    if (overlays.length) errors.push(`${name}: pop-ups found — ${overlays.join(", ")}`);

    if (shots) {
      await setSpeed(page, 1);
      await page.eval(`document.querySelector(".gear").click(); return true;`);
      await sleep(400);
      await page.shot(join(shots, `${name}-after.png`));
      console.log(`    ${name}: screenshot -> ${join(shots, `${name}-after.png`)}`);
    }

    pageErrors.forEach((e) => errors.push(`${name}: ${e}`));
  } finally {
    await cleanup();
  }
}

/* ------------------------------------------------------------------------ main */
async function main() {
  const errors = [];
  const shots = SHOTS ? (await mkdir(SHOTS, { recursive: true }), SHOTS) : null;

  if (ONLY !== "player") {
    const port = 8700 + Math.floor(Math.random() * 90);
    const gui = spawn(
      "nice",
      ["-n", "15", "uv", "run", "ludometer-gui", "--no-browser", "--port", String(port)],
      { cwd: REPO, stdio: ["ignore", "pipe", "pipe"] }
    );
    try {
      await until("the Flask GUI", async () => {
        try {
          const r = await fetch(`http://127.0.0.1:${port}/`);
          return r.ok;
        } catch { return false; }
      }, 60000);
      await checkPage({
        name: "play",
        url: `http://127.0.0.1:${port}/`,
        errors,
        shots,
        deal: async (page) => {
          await until("the board", () =>
            page.eval('return document.querySelectorAll("#middle .factory").length === 5;'), 30000);
          await page.eval(`
            const o = document.getElementById("opponent");
            if ([...o.options].some((x) => x.value === "heuristic")) o.value = "heuristic";
            o.dispatchEvent(new Event("change"));
            document.getElementById("seed").value = "31337";
            document.getElementById("setup").dispatchEvent(new Event("submit", { cancelable: true }));
            return true;
          `);
          await until("a fresh deal", () =>
            page.eval('return document.getElementById("matchup").textContent.includes("seed 31337");'), 30000);
        },
      });
    } finally {
      gui.kill();
    }
  }

  if (ONLY !== "play") {
    const server = await serve(join(REPO, "web", "player"));
    try {
      await checkPage({
        name: "player",
        url: `http://127.0.0.1:${server.address().port}/index.html`,
        errors,
        shots,
        deal: async (page) => {
          await until("the net to load", () =>
            page.eval('return document.getElementById("engine-bar").classList.contains("ready");'), 240000);
          await page.eval(`
            const think = document.getElementById("think");
            think.value = "0";
            think.dispatchEvent(new Event("change"));
            document.getElementById("seed").value = "31337";
            document.getElementById("setup").dispatchEvent(new Event("submit", { cancelable: true }));
            return true;
          `);
          await until("a fresh deal", () =>
            page.eval('return document.getElementById("matchup").textContent.includes("seed 31337");'), 30000);
        },
      });
    } finally {
      server.close();
    }
  }

  if (errors.length) {
    console.error(`\nFAIL — ${errors.length} problem(s):`);
    errors.forEach((e) => console.error("  " + e));
    return 1;
  }
  console.log("\nboth tables animate in both reduced-motion states, the settings panel governs the pace,");
  console.log("every move is animated, the log is pictographic and newest-first, and history navigation works.");
  return 0;
}

process.exit(await main());
