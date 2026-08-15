/* Drive the real page in headless Chrome, over the DevTools protocol.
 *
 * The node tests exercise the engine, the net and the search as modules. This one
 * exercises the *page*: it serves web/player/ over http, opens it in headless
 * Chrome, waits for the worker to report the net ready, then plays a move the way
 * a person would — click a colour, click a row — and waits for the AI to answer.
 *
 * What it asserts:
 *   - no console errors and no uncaught exceptions, start to finish;
 *   - the board actually rendered (5 factories, both player boards, a legal move
 *     available);
 *   - a human move is accepted and the AI replies within its budget;
 *   - the AI's own "searched N positions in T s" line, which is the honest
 *     in-browser rate and is printed at the end.
 *
 * No puppeteer: CDP is a WebSocket and node has one built in.
 *
 *   node web/player/test/browser.test.mjs [--budget 5]
 *   node web/player/test/browser.test.mjs --live      # the deployed GitHub Pages site
 */
import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { readFile, mkdtemp, rm } from "node:fs/promises";
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
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".onnx": "application/octet-stream",
  ".md": "text/plain; charset=utf-8",
};

function arg(name, fallback) {
  const i = process.argv.indexOf("--" + name);
  return i >= 0 && process.argv[i + 1] !== undefined ? Number(process.argv[i + 1]) : fallback;
}

const BUDGET = arg("budget", 5);
const LIVE = process.argv.includes("--live") ? "https://remifabre.github.io/ludometer/index.html" : null;

/* ------------------------------------------------------------- tiny http server */
function serve(root) {
  const server = createServer(async (req, res) => {
    let path = decodeURIComponent(req.url.split("?")[0]);
    if (path.endsWith("/")) path += "index.html";
    const file = normalize(join(root, path));
    if (!file.startsWith(root)) {
      res.writeHead(403).end();
      return;
    }
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
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
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
      } else if (msg.method) {
        this.listeners.forEach((fn) => fn(msg));
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

  on(fn) {
    this.listeners.push(fn);
  }

  /** Evaluate an async expression in the page and return its JSON value. */
  async eval(expression) {
    const result = await this.send("Runtime.evaluate", {
      expression: `(async () => { ${expression} })()`,
      awaitPromise: true,
      returnByValue: true,
    });
    if (result.exceptionDetails) {
      throw new Error("page threw: " + JSON.stringify(result.exceptionDetails.exception?.description || result.exceptionDetails.text));
    }
    return result.result.value;
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function until(label, fn, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const value = await fn();
    if (value) return value;
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${label} (${timeoutMs}ms)`);
    await sleep(250);
  }
}

/* ------------------------------------------------------------------------ main */
async function main() {
  // --live points the same checks at the deployed site, so "it works locally" and
  // "it works on GitHub Pages" are the same test rather than two stories.
  const server = LIVE ? null : await serve(ROOT);
  const url = LIVE || `http://127.0.0.1:${server.address().port}/index.html`;
  const profile = await mkdtemp(join(tmpdir(), "ludometer-chrome-"));
  const chrome = spawn(
    CHROME,
    [
      "--headless=new",
      "--disable-gpu",
      "--no-first-run",
      "--no-default-browser-check",
      "--remote-debugging-port=0",
      `--user-data-dir=${profile}`,
      "about:blank",
    ],
    { stdio: ["ignore", "pipe", "pipe"] }
  );

  let devtoolsUrl = null;
  chrome.stderr.on("data", (chunk) => {
    const match = String(chunk).match(/DevTools listening on (ws:\/\/\S+)/);
    if (match) devtoolsUrl = match[1];
  });

  const cleanup = async () => {
    chrome.kill();
    if (server) server.close();
    await rm(profile, { recursive: true, force: true }).catch(() => {});
  };

  try {
    await until("chrome to start", async () => devtoolsUrl, 20000);
    const browser = await Cdp.connect(devtoolsUrl);
    const { targetId } = await browser.send("Target.createTarget", { url: "about:blank" });
    const page = await Cdp.connect(devtoolsUrl.replace(/\/devtools\/browser\/.*/, `/devtools/page/${targetId}`));

    const errors = [];
    const consoleLines = [];
    page.on((msg) => {
      if (msg.method === "Runtime.consoleAPICalled") {
        const text = (msg.params.args || []).map((a) => a.value ?? a.description ?? "").join(" ");
        consoleLines.push(`${msg.params.type}: ${text}`);
        if (msg.params.type === "error") errors.push("console.error: " + text);
      } else if (msg.method === "Runtime.exceptionThrown") {
        const d = msg.params.exceptionDetails;
        errors.push("uncaught: " + (d.exception?.description || d.text));
      } else if (msg.method === "Log.entryAdded") {
        const entry = msg.params.entry;
        consoleLines.push(`log(${entry.level}): ${entry.text}`);
        if (entry.level === "error") errors.push("log: " + entry.text + " " + (entry.url || ""));
      }
    });

    await page.send("Runtime.enable");
    await page.send("Log.enable");
    await page.send("Page.enable");
    await page.send("Page.navigate", { url });

    console.log(LIVE ? `testing the live site ${url}` : `serving ${ROOT} on ${url}`);
    await until("the net to load", () => page.eval('return document.getElementById("engine-bar").classList.contains("ready");'), 180000);
    const engineText = await page.eval('return document.getElementById("engine-text").textContent;');
    console.log("engine bar:", engineText);

    // the board should be dealt automatically once the net is ready
    await until("the board", () => page.eval('return document.querySelectorAll("#factories .factory").length === 5;'), 20000);
    const shape = await page.eval(`return {
      factories: document.querySelectorAll("#factories .factory").length,
      tiles: document.querySelectorAll("#factories button.tile").length,
      humanRows: document.querySelectorAll("#board-human .wall-row").length,
      aiRows: document.querySelectorAll("#board-ai .wall-row").length,
      title: document.title,
    };`);
    console.log("board:", JSON.stringify(shape));
    if (shape.factories !== 5 || shape.humanRows !== 5 || shape.aiRows !== 5 || shape.tiles < 10) {
      errors.push("the board did not render as expected: " + JSON.stringify(shape));
    }

    // Phone check: at 390 CSS px nothing may push the page sideways, and the
    // controls must stay tappable. A board you have to scroll horizontally is a
    // board you cannot play on a phone.
    await page.send("Emulation.setDeviceMetricsOverride", {
      width: 390, height: 844, deviceScaleFactor: 2, mobile: true,
    });
    await sleep(600);
    const phone = await page.eval(`return {
      scrollWidth: document.documentElement.scrollWidth,
      innerWidth: window.innerWidth,
      overflowing: [...document.querySelectorAll("body *")]
        .filter((n) => n.getBoundingClientRect().right > window.innerWidth + 1)
        .slice(0, 3)
        .map((n) => n.tagName + "." + (n.className || "").toString().split(" ")[0]),
      // hidden controls have no size; a checkbox's real target is its label
      smallTargets: [...document.querySelectorAll(".btn, select, input")]
        .filter((n) => n.offsetParent !== null)
        .map((n) => (n.type === "checkbox" ? n.closest("label") || n : n))
        .filter((n) => n.getBoundingClientRect().height < 32)
        .map((n) => n.id || n.className || n.tagName),
    };`);
    console.log("at 390px:", JSON.stringify(phone));
    if (phone.scrollWidth > phone.innerWidth + 1) {
      errors.push(`the page scrolls sideways on a phone (${phone.scrollWidth} > ${phone.innerWidth}): ${phone.overflowing.join(", ")}`);
    }
    if (phone.smallTargets.length) {
      errors.push("controls below a 32px touch target: " + phone.smallTargets.join(", "));
    }
    await page.send("Emulation.clearDeviceMetricsOverride");
    await sleep(300);

    // set the thinking budget, deal a fresh game, then play like a person would
    await page.eval(`
      const think = document.getElementById("think");
      think.value = "${BUDGET}";
      think.dispatchEvent(new Event("change"));
      document.getElementById("seed").value = "424242";
      document.getElementById("setup").dispatchEvent(new Event("submit", {cancelable: true}));
      return true;
    `);
    await until("a fresh deal", () => page.eval('return document.getElementById("matchup").textContent.includes("seed 424242");'), 20000);

    const picked = await page.eval(`
      const tile = document.querySelector("#factories button.tile:not([disabled])");
      if (!tile) return null;
      tile.click();
      return {source: tile.dataset.source, color: tile.dataset.color};
    `);
    if (!picked) throw new Error("no playable tile on the table");
    console.log("picked up:", JSON.stringify(picked));

    const dropped = await page.eval(`
      const row = document.querySelector("#board-human .line.open") || document.querySelector("#board-human .floor.open");
      if (!row) return null;
      const where = row.dataset.row;
      row.click();
      return where;
    `);
    if (dropped === null) throw new Error("no legal destination lit up after picking a colour");
    console.log("dropped into row:", dropped);

    // the AI now spends its budget; the search note is written when it lands
    const note = await until(
      "the AI to reply",
      () => page.eval('const n = document.querySelector("#last-move .search-note"); return n ? n.textContent : null;'),
      Math.max(60000, BUDGET * 8000)
    );
    console.log("AI:", note);
    const match = note.match(/searched ([\d,]+) positions in ([\d.]+)s/);
    let rate = null;
    if (match) {
      const sims = Number(match[1].replace(/,/g, ""));
      const seconds = Number(match[2]);
      rate = sims / seconds;
    }

    const after = await page.eval(`return {
      scores: [...document.querySelectorAll(".score-value")].map((n) => n.textContent),
      turn: document.getElementById("turn").textContent,
      log: document.querySelectorAll("#log li").length,
    };`);
    console.log("after the exchange:", JSON.stringify(after));
    if (after.log < 3) errors.push("the table talk log did not fill in");

    await cleanup();

    if (errors.length) {
      console.error(`FAIL — ${errors.length} console/page error(s):`);
      errors.slice(0, 10).forEach((e) => console.error("  " + e));
      return 1;
    }
    console.log("\nno console errors, no uncaught exceptions, a full human/AI exchange played out");
    if (rate) {
      console.log(`in-browser search rate: ${Math.round(rate).toLocaleString()} positions/s ` +
        `(${match[1]} positions in ${match[2]}s at a ${BUDGET}s budget, headless Chrome)`);
    }
    return 0;
  } catch (err) {
    await cleanup();
    console.error("FAIL —", err.message);
    return 1;
  }
}

process.exit(await main());
