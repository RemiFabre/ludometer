/* Azul against the net — the table, in your browser.
 *
 * Adapted from web/play/app.js, which drew the same board but asked a Flask
 * server for the rules and the AI. Everything is local now: `GameSession` (a port
 * of the server's session object) owns the game, and a Web Worker owns the
 * search. What did not change is the interaction — you pick up a colour, the
 * legal rows light up, you drop them, the AI's tiles fly across the table — and
 * legality still comes from one list of action ids rather than from the page
 * re-deriving the rules.
 *
 * A turn plays out in two beats so the table moves at a human pace: your own move
 * lands immediately, then the AI visibly spends its thinking budget (with a live
 * position count, since it is your CPU doing the work), and only then do its tiles
 * travel. Input stays locked from the moment a move is sent until the last tile
 * has landed.
 */
"use strict";

import { CENTER, FLOOR, COLOR_NAMES, CUM_PENALTY, FLOOR_PENALTIES, encodeAction } from "./engine.js";
import { GameSession } from "./game.js";
import { describeAction, finalReport } from "./report.js";

const COLORS = COLOR_NAMES;
const FLASH_MS = 1400;
/* move animation: pick up, fly, settle — about 1.7 s all told */
const PICKUP_MS = 420;
const FLIGHT_MS = 780;
const STAGGER_MS = 55;
const SETTLE_MS = 220;
const TILING_STEP_MS = 150; // one wall tile lands every ...
const TILING_LAST_MS = 520; // ... plus the last tile's own animation

const el = (id) => document.getElementById(id);
const ui = {
  matchup: el("matchup"), setup: el("setup"), seed: el("seed"), first: el("first"),
  think: el("think"), facing: el("facing"), deal: el("deal"),
  engineBar: el("engine-bar"), engineText: el("engine-text"), aboutMeta: el("about-meta"),
  factories: el("factories"), center: el("center"), turn: el("turn"),
  thinking: el("thinking"), thinkingText: el("thinking-text"), fly: el("fly"),
  scoreHuman: el("score-human"), scoreAi: el("score-ai"), boardAi: el("board-ai"),
  boardHuman: el("board-human"), lastMove: el("last-move"), log: el("log"),
  supply: el("supply"), prompt: el("prompt"), hint: el("hint"), cancel: el("cancel"),
  overlay: el("overlay"), overlayTitle: el("overlay-title"), overlayBody: el("overlay-body"),
  overlayOk: el("overlay-ok"), toasts: el("toasts"),
};

let session = null;    // the GameSession that owns the rules
let S = null;          // its last snapshot, what the page draws from
let sel = null;        // {source, color} — tiles the player has picked up
let suggestion = null; // {source, color, dest} from the policy head
let busy = false;      // a move is being computed, or is playing out
let sheets = [];       // queued round-end / game-end overlays
let sheetWaiters = []; // resolvers waiting for the overlay queue to drain
let meta = null;       // model/model_meta.json
let engineReady = false;
let liveSims = 0;      // positions the current search has visited

const reducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const sleep = (ms) => new Promise((done) => setTimeout(done, reducedMotion ? 0 : ms));

/* ------------------------------------------------------------------ plumbing */
function node(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = text;
  return n;
}

function toast(message, kind) {
  const t = node("div", "toast" + (kind ? " " + kind : ""), message);
  ui.toasts.appendChild(t);
  setTimeout(() => t.remove(), 6000);
}

function setBusy(on) {
  busy = on;
  document.body.classList.toggle("locked", on);
  ui.deal.disabled = on || !engineReady;
  ui.hint.disabled = on || !engineReady || !S || !S.your_turn;
}

/* ---------------------------------------------------------------- the worker */
/* One worker, kept for the life of the page: it holds the 13 MB session, which
 * is far too expensive to rebuild per move. Requests are matched by id. */
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
let nextId = 1;
const pending = new Map();

worker.onmessage = (event) => {
  const msg = event.data;
  const entry = pending.get(msg.id);
  if (!entry) return;
  if (msg.type === "loading" || msg.type === "progress") {
    if (entry.onProgress) entry.onProgress(msg);
    return;
  }
  pending.delete(msg.id);
  if (msg.type === "error") entry.reject(new Error(msg.message));
  else entry.resolve(msg);
};

worker.onerror = (event) => {
  engineFailed(event.message || "the worker crashed");
};

function ask(message, onProgress) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, onProgress });
    worker.postMessage({ ...message, id });
  });
}

function engineFailed(reason) {
  engineReady = false;
  ui.engineBar.classList.remove("ready");
  ui.engineBar.classList.add("failed");
  ui.engineText.textContent = "The net could not be loaded: " + reason;
  ui.deal.disabled = true;
  ui.hint.disabled = true;
}

async function bootEngine() {
  try {
    meta = await (await fetch("model/model_meta.json", { cache: "no-cache" })).json();
  } catch (err) {
    meta = null;
  }
  describeModel();
  const sizeMB = meta && meta.onnx_bytes ? (meta.onnx_bytes / 1e6).toFixed(1) : "13";
  ui.engineText.textContent = `Downloading the net (${sizeMB} MB, cached after the first visit)…`;
  try {
    await ask(
      {
        type: "init",
        ortUrl: new URL("../vendor/onnxruntime-web/ort.wasm.bundle.min.mjs", import.meta.url).href,
        wasmUrl: new URL("../vendor/onnxruntime-web/ort-wasm-simd-threaded.wasm", import.meta.url).href,
        modelUrl: new URL("../model/model.onnx", import.meta.url).href,
      },
      (msg) => {
        if (msg.type !== "loading" || !msg.total) return;
        const pct = Math.round((msg.received / msg.total) * 100);
        ui.engineText.textContent = `Downloading the net — ${pct}% of ${(msg.total / 1e6).toFixed(1)} MB`;
      }
    );
  } catch (err) {
    engineFailed(err.message);
    return;
  }
  engineReady = true;
  ui.engineBar.classList.add("ready");
  ui.engineText.textContent = engineLine();
  ui.deal.disabled = false;
  newGame();
}

function engineLine() {
  if (!meta) return "Net ready — searching on your CPU.";
  const elo = typeof meta.elo === "number" ? `${meta.elo >= 0 ? "+" : ""}${Math.round(meta.elo)} Elo` : "unrated";
  const params = meta.num_params ? `${(meta.num_params / 1e6).toFixed(1)}M parameters` : "";
  return `${meta.run}/${meta.checkpoint} · ${elo} on our internal ladder · ${params} · searching on your CPU`;
}

function describeModel() {
  if (!meta) {
    ui.aboutMeta.textContent = "";
    return;
  }
  const bits = [
    `${meta.run}/${meta.checkpoint}`,
    meta.games ? `${Number(meta.games).toLocaleString()} self-play games` : null,
    meta.num_params ? `${meta.num_params.toLocaleString()} parameters` : null,
    meta.onnx_bytes ? `${(meta.onnx_bytes / 1e6).toFixed(1)} MB ONNX` : null,
    meta.exported_at ? `exported ${meta.exported_at.slice(0, 10)}` : null,
  ].filter(Boolean);
  ui.aboutMeta.textContent = bits.join(" · ");
}

/* -------------------------------------------------------------- game control */
/** Seconds the AI may think per move — 0 means "policy head, no search". */
function currentThinkTime() {
  const seconds = Number(ui.think.value);
  return Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
}

/** The AI's brain, handed to the session: one search request to the worker. */
async function think(state, onThinking) {
  const budgetS = session ? session.thinkTimeS : 0;
  liveSims = 0;
  const reply = await ask(
    { type: budgetS > 0 ? "search" : "policy", setup: state.toSetup(), budgetS },
    (msg) => {
      if (msg.type !== "progress") return;
      liveSims = msg.sims;
      if (onThinking) onThinking(msg);
    }
  );
  return { action: reply.action, search: reply.search };
}

function newGame(event) {
  if (event) event.preventDefault();
  if (!engineReady) {
    toast("The net is still loading.");
    return;
  }
  let seed = Math.floor(Math.random() * (1 << 30));
  const typed = ui.seed.value.trim();
  if (typed) {
    if (!/^-?\d+$/.test(typed)) {
      toast("The seed must be a whole number.");
      return;
    }
    seed = Number(typed) >>> 0;
  }
  sheets = [];
  ui.overlay.classList.add("hidden");
  session = new GameSession({
    seed,
    humanPlaysFirst: ui.first.checked,
    agentName: meta ? meta.checkpoint : "the net",
    opponentInfo: meta ? { checkpoint: meta.checkpoint, elo: meta.elo, run: meta.run } : {},
    thinkTimeS: currentThinkTime(),
    think,
  });
  adopt();
  toast("New game dealt against " + session.agentName + ".", "good");
  // the AI opens when you took the second seat
  if (session.aiTurn) resumeIfPending();
}

/** One turn: your move lands, the AI thinks, then its move plays out. */
async function play(actionId) {
  if (busy || !session) return;
  setBusy(true);
  try {
    const { move, reports } = session.playHuman(actionId);
    void move;
    adopt();
    reports.forEach(queueRoundSheet);
    if (session.state.isTerminal) queueFinalSheet();
    await animateTiling(reports);
    if (!session.aiTurn) {
      showNextSheet();
      return;
    }
    await runAiTurn(showNextSheet);
  } catch (err) {
    stopThinking();
    toast(err.message);
    adopt();
  } finally {
    setBusy(false);
  }
}

/** If the position says the AI still owes a move, let it move. */
async function resumeIfPending() {
  if (busy || !session || !session.aiTurn) return;
  setBusy(true);
  try {
    await runAiTurn();
  } catch (err) {
    stopThinking();
    toast(err.message);
  } finally {
    setBusy(false);
  }
}

/**
 * Let the AI move, show the clock running, then play the move out.
 * `whileThinking` runs once the search is under way, so a round-end sheet can be
 * read during it instead of delaying it.
 */
async function runAiTurn(whileThinking) {
  const firstReport = session.roundReports.length;
  const pendingMoves = session.aiReplies();
  startThinking();
  if (whileThinking) whileThinking();
  let moves;
  try {
    moves = await pendingMoves;
  } finally {
    stopThinking();
  }
  await waitForSheets();
  await animateAiMoves(moves);
  adopt();
  const reports = session.reportsSince(firstReport);
  reports.forEach(queueRoundSheet);
  if (session.state.isTerminal) queueFinalSheet();
  await animateTiling(reports);
  showNextSheet();
}

/** The policy head's own pick, shown as a suggestion (one forward pass). */
async function askHint() {
  if (!session || !S || !S.your_turn || busy) return;
  setBusy(true);
  try {
    const reply = await ask({ type: "policy", setup: session.state.toSetup() });
    const move = describeAction(session.state, reply.action);
    suggestion = { source: move.source, color: move.color, dest: move.dest };
    sel = { source: move.source, color: move.color };
    render();
    toast("Try: " + move.text, "good");
  } catch (err) {
    toast(err.message);
  } finally {
    setBusy(false);
  }
}

function adopt() {
  S = session.snapshot();
  sel = null;
  suggestion = null;
  render();
}

/* -------------------------------------------------------------- legality map */
function legalSet() {
  return new Set((S && S.human_legal_actions) || []);
}

function destsFor(source, color) {
  const legal = legalSet();
  const out = [];
  for (let d = 0; d <= FLOOR; d++) if (legal.has(encodeAction(source, color, d))) out.push(d);
  return out;
}

function poolCount(source, color) {
  const st = S.state;
  return source === CENTER ? st.center[color] : st.factories[source][color];
}

/** Why `dest` is closed for the held colour — the hover text on a dark row. */
function blockedReason(color, dest) {
  const me = S.state.players[S.human_seat];
  const line = me.pattern_lines[dest];
  if (line.count >= line.capacity) return "This row is already full.";
  if (line.count > 0 && line.color !== color) return "This row holds " + COLORS[line.color] + " tiles.";
  if (me.wall[dest][(color + dest) % 5]) return COLORS[color] + " is already on your wall in this row.";
  return "Not playable right now.";
}

/** What playing `sel` into `dest` costs: tiles placed, spill, extra floor penalty. */
function preview(dest) {
  const me = S.state.players[S.human_seat];
  const count = poolCount(sel.source, sel.color);
  const takesMarker = sel.source === CENTER && S.state.marker_in_center;
  let overflow = count;
  let placed = 0;
  if (dest !== FLOOR) {
    const line = me.pattern_lines[dest];
    placed = Math.min(count, line.capacity - line.count);
    overflow = count - placed;
  }
  const occupied = me.floor.reduce((a, b) => a + b, 0) + (me.floor_marker ? 1 : 0);
  const before = CUM_PENALTY[Math.min(7, occupied)];
  const after = CUM_PENALTY[Math.min(7, occupied + overflow + (takesMarker ? 1 : 0))];
  return {
    count, placed, overflow, takesMarker,
    penalty: after - before,
    completes: dest !== FLOOR && placed + me.pattern_lines[dest].count === me.pattern_lines[dest].capacity,
  };
}

/* -------------------------------------------------------------- tile factory */
function tile(color, cls) {
  const t = node("div", "tile" + (cls ? " " + cls : ""));
  t.dataset.color = color;
  return t;
}

function tileButton(source, color) {
  const b = node("button", "tile");
  b.type = "button";
  b.dataset.color = color;
  b.dataset.source = source;
  b.dataset.key = source + ":" + color;
  const dests = S.your_turn && !S.state.is_terminal ? destsFor(source, color) : [];
  if (!dests.length) {
    b.disabled = true;
    if (S.your_turn) b.title = "Nothing you can do with these yet.";
  } else {
    b.title = poolCount(source, color) + " " + COLORS[color] + " — click to pick them up";
    b.addEventListener("click", () => {
      sel = sel && sel.source === source && sel.color === color ? null : { source, color };
      suggestion = null;
      render();
    });
  }
  return b;
}

function markerChip(tiny) {
  const m = node("div", "marker" + (tiny ? " tiny" : ""), "1");
  m.title = "First-player marker: whoever takes it starts the next round and loses a floor slot.";
  return m;
}

/* ------------------------------------------------------- middle of the table */
function renderMiddle() {
  const st = S.state;
  ui.factories.innerHTML = "";
  st.factories.forEach((counts, i) => {
    const dish = node("div", "factory");
    dish.dataset.source = i;
    const total = counts.reduce((a, b) => a + b, 0);
    if (!total) {
      dish.classList.add("empty");
      dish.appendChild(node("span", "empty-note", "empty"));
    } else {
      counts.forEach((n, c) => {
        for (let k = 0; k < n; k++) dish.appendChild(tileButton(i, c));
      });
    }
    const id = node("span", "fac-id", String(i + 1));
    id.title = "Factory " + (i + 1);
    dish.appendChild(id);
    ui.factories.appendChild(dish);
  });

  ui.center.innerHTML = "";
  ui.center.dataset.source = CENTER;
  const centreTotal = st.center.reduce((a, b) => a + b, 0);
  if (st.marker_in_center) ui.center.appendChild(markerChip(false));
  if (!centreTotal) {
    ui.center.appendChild(node("span", "empty-note", st.marker_in_center ? "middle — marker only" : "middle — empty"));
  } else {
    st.center.forEach((n, c) => {
      if (!n) return;
      const group = node("div", "group");
      for (let k = 0; k < n; k++) group.appendChild(tileButton(CENTER, c));
      ui.center.appendChild(group);
    });
  }
}

/* -------------------------------------------------------------------- boards */
/** Pattern line `r` and wall row `r` are one grid row, as on the cardboard. */
function renderBoard(host, seat, interactive) {
  const st = S.state;
  const me = st.players[seat];
  host.innerHTML = "";
  host.dataset.seat = seat;

  const grid = node("div", "board-grid");
  const open = interactive && sel ? destsFor(sel.source, sel.color) : [];

  for (let r = 0; r < 5; r++) {
    const line = me.pattern_lines[r];
    const isOpen = open.indexOf(r) !== -1;
    const row = node(interactive && sel ? "button" : "div", "line");
    if (row.tagName === "BUTTON") row.type = "button";
    row.dataset.row = r;

    const note = node("span", "row-note");
    if (interactive && sel) {
      if (isOpen) {
        const p = preview(r);
        let text = "take " + p.count + " → " + p.placed + " here";
        if (p.overflow) text += ", " + p.overflow + " spills (" + p.penalty + ")";
        else if (p.completes) text += ", completes the row";
        note.textContent = text;
        row.classList.add("open");
        row.addEventListener("click", () => play(encodeAction(sel.source, sel.color, r)));
      } else {
        row.classList.add("blocked");
        row.title = blockedReason(sel.color, r);
        note.textContent = row.title;
        row.setAttribute("aria-disabled", "true");
      }
    } else {
      note.textContent = line.count
        ? line.count + "/" + line.capacity + " " + COLORS[line.color]
        : line.capacity + (line.capacity === 1 ? " slot" : " slots");
    }
    row.appendChild(note);

    // the line fills from the wall end backwards, so empty slots sit on the left
    for (let k = 0; k < line.capacity - line.count; k++) row.appendChild(node("div", "slot"));
    for (let k = 0; k < line.count; k++) row.appendChild(tile(line.color));
    if (suggestion && interactive && suggestion.dest === r) row.classList.add("flash");
    grid.appendChild(row);

    const wallRow = node("div", "wall-row");
    wallRow.dataset.wallRow = r;
    if (r === 0) wallRow.classList.add("top");
    if (r === 4) wallRow.classList.add("bottom");
    for (let col = 0; col < 5; col++) {
      const colour = (col - r + 5) % 5;
      if (me.wall[r][col]) {
        const t = tile(colour, "placed");
        t.dataset.row = r;
        t.dataset.col = col;
        t.title = COLORS[colour] + " — row " + (r + 1) + ", column " + (col + 1);
        wallRow.appendChild(t);
      } else {
        const cell = node("div", "cell bisque");
        cell.dataset.color = colour;
        cell.dataset.row = r;
        cell.dataset.col = col;
        cell.title = COLORS[colour] + " goes here";
        wallRow.appendChild(cell);
      }
    }
    grid.appendChild(wallRow);
  }
  host.appendChild(grid);

  const wrap = node("div", "floor-wrap");
  const floorOpen = interactive && sel && open.indexOf(FLOOR) !== -1;
  const floor = node(floorOpen ? "button" : "div", "floor");
  if (floorOpen) {
    floor.type = "button";
    floor.classList.add("open");
    floor.addEventListener("click", () => play(encodeAction(sel.source, sel.color, FLOOR)));
    const p = preview(FLOOR);
    floor.title = "Dump all " + p.count + " tiles here (" + p.penalty + " this round)";
  }
  floor.dataset.row = FLOOR;
  const occupants = [];
  if (me.floor_marker) occupants.push("marker");
  me.floor.forEach((n, c) => {
    for (let k = 0; k < n; k++) occupants.push(c);
  });
  for (let i = 0; i < 7; i++) {
    const cellwrap = node("div", "cellwrap");
    const here = occupants[i];
    if (here === "marker") cellwrap.appendChild(markerChip(true));
    else if (here !== undefined) cellwrap.appendChild(tile(here));
    else cellwrap.appendChild(node("div", "slot"));
    cellwrap.appendChild(node("span", "pen", FLOOR_PENALTIES[i]));
    floor.appendChild(cellwrap);
  }
  wrap.appendChild(floor);

  const penalty = me.floor_penalty;
  const note = node(
    "span",
    "floor-note" + (penalty ? " warn" : ""),
    penalty ? "Floor line: " + penalty + " at the end of this round" : "Floor line: clean"
  );
  if (floorOpen) {
    const p = preview(FLOOR);
    note.textContent = "Floor line: " + penalty + " now, " + (penalty + p.penalty) + " if you dump here";
    note.classList.add("warn");
  }
  wrap.appendChild(note);
  host.appendChild(wrap);
}

/* -------------------------------------------------------------------- panels */
function renderScores() {
  const st = S.state;
  const seats = [
    [ui.scoreHuman, S.human_seat, "You"],
    [ui.scoreAi, S.ai_seat, "AI · " + S.agent_name],
  ];
  seats.forEach(([host, seat, who]) => {
    const me = st.players[seat];
    host.querySelector(".score-who").textContent = who;
    host.querySelector(".score-value").textContent = me.score;
    const bits = [];
    if (me.floor_penalty) bits.push(me.floor_penalty + " on the floor");
    if (me.completed_rows) bits.push(me.completed_rows + " full row" + (me.completed_rows > 1 ? "s" : ""));
    if (me.floor_marker) bits.push("holds the marker");
    host.querySelector(".score-note").textContent = bits.join(" · ");
    host.classList.toggle("to-move", !st.is_terminal && st.current_player === seat);
  });
}

function renderSupply() {
  const st = S.state;
  ui.supply.innerHTML = "";
  [["Bag", st.bag], ["Lid", st.lid]].forEach(([label, counts]) => {
    const row = node("div", "supply-row");
    row.appendChild(node("span", "supply-label", label));
    counts.forEach((n, c) => {
      const sw = node("span", "swatch");
      const chip = node("span", "chip");
      chip.dataset.color = c;
      chip.title = COLORS[c];
      sw.appendChild(chip);
      sw.appendChild(node("span", null, String(n)));
      row.appendChild(sw);
    });
    row.appendChild(node("span", "supply-total", counts.reduce((a, b) => a + b, 0) + " tiles"));
    ui.supply.appendChild(row);
  });
  const row = node("div", "supply-row");
  row.appendChild(node("span", "supply-label", "Table"));
  row.appendChild(node("span", "swatch", st.tiles_left + " tiles still to take · round " + (st.round + 1)));
  ui.supply.appendChild(row);
}

function renderNarration() {
  const last = S.last_ai_move;
  ui.lastMove.innerHTML = "";
  if (S.state.is_terminal && S.final) {
    ui.lastMove.textContent = S.final.headline;
  } else if (last) {
    ui.lastMove.append("AI " + last.text + ".");
    if (last.search_text) ui.lastMove.appendChild(node("span", "search-note", last.search_text));
  } else {
    ui.lastMove.textContent = "You open. Pick a colour from a factory or the middle.";
  }

  ui.facing.textContent = S.opponent_blurb || "";
  ui.facing.classList.toggle("hidden", !S.opponent_blurb);

  ui.log.innerHTML = "";
  (S.log || []).forEach((entry) => {
    ui.log.appendChild(node("li", entry.kind === "move" ? entry.side : entry.kind, entry.text));
  });
  ui.log.scrollTop = ui.log.scrollHeight;
}

function renderStatus() {
  const st = S.state;
  const info = S.opponent_info || {};
  const rating = typeof info.elo === "number" ? " (" + Math.round(info.elo) + " Elo)" : "";
  ui.matchup.textContent = "you versus " + S.agent_name + rating + " · seed " + S.seed;
  if (st.is_terminal) {
    ui.turn.innerHTML = "";
    ui.turn.appendChild(node("strong", null, "Game over."));
    ui.turn.append(" " + (S.final ? S.final.headline : ""));
    ui.prompt.textContent = "Deal again to play another.";
  } else if (S.your_turn) {
    ui.turn.innerHTML = "";
    ui.turn.appendChild(node("strong", null, "Your turn"));
    ui.turn.append(" — round " + (st.round + 1) + ", " + st.tiles_left + " tiles left on the table.");
    ui.prompt.textContent = sel
      ? "Now pick a row (or the floor line) for your " + COLORS[sel.color] + " tiles."
      : "Pick a colour from a factory or the middle.";
  } else {
    ui.turn.innerHTML = "";
    ui.turn.appendChild(node("strong", null, "The AI's turn"));
    ui.turn.append(S.think_time_s ? " — it thinks for " + S.think_time_s + "s per move." : " — it is choosing a move.");
    ui.prompt.textContent = "";
  }
  ui.cancel.classList.toggle("hidden", !sel);
  ui.hint.disabled = busy || !engineReady || !S.your_turn;
}

function render() {
  if (!S) return;
  renderStatus();
  renderMiddle();
  renderScores();
  renderBoard(ui.boardHuman, S.human_seat, true);
  renderBoard(ui.boardAi, S.ai_seat, false);
  renderSupply();
  renderNarration();

  // held tiles ride above the table; everything else steps back
  const all = [].concat(
    [].slice.call(ui.factories.querySelectorAll("button.tile")),
    [].slice.call(ui.center.querySelectorAll("button.tile"))
  );
  all.forEach((b) => {
    if (!sel) return;
    if (b.dataset.key === sel.source + ":" + sel.color) b.classList.add("taken");
    else b.classList.add("dimmed");
  });
  if (suggestion) flash(sourceElement(suggestion.source));
}

/* ----------------------------------------------------------------- animation */
function flash(element) {
  if (!element) return;
  element.classList.remove("flash");
  void element.offsetWidth; // restart the animation
  element.classList.add("flash");
  setTimeout(() => element.classList.remove("flash"), FLASH_MS);
}

function sourceElement(source) {
  return source === CENTER ? ui.center : ui.factories.querySelector('.factory[data-source="' + source + '"]');
}

/** Where the AI's tiles are coming from: the drawn tiles of that colour. */
function sourceTiles(source, color, count) {
  const host = sourceElement(source);
  if (!host) return [];
  return [].slice.call(host.querySelectorAll('.tile[data-color="' + color + '"]'), 0, count);
}

/**
 * Where they are going: the pattern line's rightmost free slots first (a line
 * fills towards the wall), then the floor line's free slots. Tiles beyond both go
 * straight to the lid, and are aimed at the floor's edge to fade out there.
 */
function travelTargets(host, move) {
  const targets = [];
  if (move.dest !== FLOOR && move.placed) {
    const row = host.querySelector('.line[data-row="' + move.dest + '"]');
    const slots = row ? [].slice.call(row.querySelectorAll(".slot")) : [];
    targets.push.apply(targets, slots.slice(Math.max(0, slots.length - move.placed)));
  }
  const floor = host.querySelector(".floor");
  const floorSlots = floor ? [].slice.call(floor.querySelectorAll(".slot")) : [];
  let taken = 0;
  while (targets.length < move.count && taken < floorSlots.length) targets.push(floorSlots[taken++]);
  const last = targets[targets.length - 1] || floor;
  while (targets.length < move.count) targets.push(last); // overflow to the lid
  return targets;
}

/** Fly clones of `tiles` to `targets`, staggered, and resolve when they land. */
function flyTiles(tiles, targets) {
  if (reducedMotion || !tiles.length) return Promise.resolve();
  const layer = ui.fly;
  const flights = [];
  tiles.forEach((from, i) => {
    const to = targets[i];
    if (!to) return;
    const a = from.getBoundingClientRect();
    const b = to.getBoundingClientRect();
    const ghost = node("div", "fly-tile tile");
    ghost.dataset.color = from.dataset.color;
    ghost.style.left = a.left + "px";
    ghost.style.top = a.top + "px";
    ghost.style.width = a.width + "px";
    ghost.style.height = a.height + "px";
    ghost.style.transitionDelay = i * STAGGER_MS + "ms";
    layer.appendChild(ghost);
    from.style.visibility = "hidden";
    flights.push([ghost, b.left - a.left, b.top - a.top, b.width / (a.width || 1)]);
  });
  if (!flights.length) return Promise.resolve();
  return new Promise((done) => {
    requestAnimationFrame(() => {
      flights.forEach(([ghost, dx, dy, scale]) => {
        ghost.style.transform = "translate(" + dx + "px, " + dy + "px) scale(" + scale.toFixed(3) + ")";
      });
      setTimeout(() => {
        flights.forEach(([ghost]) => ghost.remove());
        done();
      }, FLIGHT_MS + flights.length * STAGGER_MS);
    });
  });
}

/** Highlight the source, then walk the tiles over to the AI's board. */
async function animateAiMove(move) {
  const source = sourceElement(move.source);
  if (source) source.classList.add("picked");
  const row = ui.boardAi.querySelector('.line[data-row="' + move.dest + '"]');
  if (row) row.classList.add("incoming");
  await sleep(PICKUP_MS);
  await flyTiles(sourceTiles(move.source, move.color, move.count), travelTargets(ui.boardAi, move));
  await sleep(SETTLE_MS);
  if (source) source.classList.remove("picked");
  if (row) row.classList.remove("incoming");
}

async function animateAiMoves(moves) {
  for (const move of moves) await animateAiMove(move);
}

/** Round end: wall tiles light up one after another, in placement order. */
async function animateTiling(reports) {
  if (reducedMotion || !reports || !reports.length) return;
  let steps = 0;
  reports.forEach((report) => {
    [[S.human_seat, ui.boardHuman], [S.ai_seat, ui.boardAi]].forEach(([seat, host]) => {
      const player = report.players[seat];
      if (!player) return;
      player.tiles.forEach((t, i) => {
        const cell = host.querySelector('.tile.placed[data-row="' + t.row + '"][data-col="' + t.col + '"]');
        if (!cell) return;
        cell.style.animationDelay = i * TILING_STEP_MS + "ms";
        cell.classList.add("landing");
        steps = Math.max(steps, i + 1);
      });
    });
  });
  if (steps) await sleep((steps - 1) * TILING_STEP_MS + TILING_LAST_MS);
}

/* ------------------------------------------------------------- thinking state */
let thinkTimer = null;

function startThinking() {
  if (!S || S.your_turn || S.state.is_terminal) return;
  const started = Date.now();
  const budget = S.think_time_s;
  const label = () => {
    const spent = (Date.now() - started) / 1000;
    if (!budget) {
      ui.thinkingText.textContent = "The AI is picking a move";
      return;
    }
    const counted = liveSims ? " · " + liveSims.toLocaleString() + " positions" : "";
    ui.thinkingText.textContent = "The AI is thinking — " + spent.toFixed(1) + "s of " + budget + "s" + counted;
  };
  label();
  ui.thinking.classList.remove("hidden");
  ui.scoreAi.classList.add("thinking-now");
  if (thinkTimer) clearInterval(thinkTimer);
  thinkTimer = setInterval(label, 100);
}

function stopThinking() {
  if (thinkTimer) {
    clearInterval(thinkTimer);
    thinkTimer = null;
  }
  ui.thinking.classList.add("hidden");
  ui.scoreAi.classList.remove("thinking-now");
}

/* ------------------------------------------------------------------ overlays */
function miniTile(color) {
  const t = tile(color);
  t.style.width = "26px";
  t.style.height = "26px";
  return t;
}

function tallyFor(report, seat, label) {
  const p = report.players[seat];
  const card = node("div", "tally-player");
  const head = node("div", "tally-head");
  head.appendChild(node("span", "tally-who", label));
  head.appendChild(
    node("span", "tally-delta", (p.delta >= 0 ? "+" + p.delta : String(p.delta)) + " → " + p.score_after + " points")
  );
  card.appendChild(head);

  if (p.tiles.length) {
    const placed = node("div", "placed");
    p.tiles.forEach((t) => {
      const box = node("div", "placed-tile");
      box.appendChild(miniTile(t.color));
      box.appendChild(node("span", "pts", "+" + t.points));
      box.appendChild(node("span", "where", "r" + (t.row + 1) + " c" + (t.col + 1)));
      box.title =
        t.color_name + " → row " + (t.row + 1) + ", column " + (t.col + 1) +
        " (run of " + t.h_run + " across, " + t.v_run + " down)";
      placed.appendChild(box);
    });
    card.appendChild(placed);
    card.appendChild(
      node("div", "tally-line", "Wall tiles: +" + p.tiling_points + " from " + p.tiles.length + " tile" + (p.tiles.length > 1 ? "s" : ""))
    );
  } else {
    card.appendChild(node("div", "tally-line", "No pattern line was full — nothing moved to the wall."));
  }

  const floorBits = [];
  if (p.floor.tiles.length) floorBits.push(p.floor.tiles.length + " tile" + (p.floor.tiles.length > 1 ? "s" : ""));
  if (p.floor.marker) floorBits.push("the first-player marker");
  card.appendChild(
    node("div", "tally-line", floorBits.length ? "Floor line: " + floorBits.join(" and ") + " → " + p.floor.penalty : "Floor line: clean.")
  );
  if (p.carried_rows.length) {
    card.appendChild(
      node("div", "tally-line", "Carried over: " + p.carried_rows.map((r) => r.count + " " + COLORS[r.color] + " in row " + (r.row + 1)).join(", "))
    );
  }
  return card;
}

function queueRoundSheet(report) {
  const body = node("div", "tally");
  body.appendChild(tallyFor(report, S.human_seat, "You"));
  body.appendChild(tallyFor(report, S.ai_seat, "AI · " + S.agent_name));
  if (!report.game_over) {
    const you = report.next_first_player === S.human_seat;
    body.appendChild(node("div", "tally-line", (you ? "You" : "The AI") + " start" + (you ? "" : "s") + " the next round."));
  }
  sheets.push({ title: "Round " + (report.round + 1) + " tiled", body, cta: "Carry on" });
}

function queueFinalSheet() {
  const final = finalReport(session.state, session.humanSeat);
  if (!final) return;
  const body = node("div");
  const table = node("table", "bonus-table");
  const head = node("thead");
  const hr = node("tr");
  ["", "Before bonus", "Rows ×2", "Columns ×7", "Colours ×10", "Bonus", "Final"].forEach((h) => hr.appendChild(node("th", null, h)));
  head.appendChild(hr);
  table.appendChild(head);
  const tbody = node("tbody");
  [[S.human_seat, "You"], [S.ai_seat, "AI · " + S.agent_name]].forEach(([seat, label]) => {
    const b = final.bonuses[seat];
    const tr = node("tr");
    tr.appendChild(node("td", null, label));
    tr.appendChild(node("td", null, String(b.score_before_bonus)));
    tr.appendChild(node("td", null, b.rows + " (+" + b.row_points + ")"));
    tr.appendChild(node("td", null, b.cols + " (+" + b.col_points + ")"));
    tr.appendChild(node("td", null, b.colors + " (+" + b.color_points + ")"));
    tr.appendChild(node("td", null, "+" + b.total));
    const total = node("td", "final-score", String(b.final_score));
    if (final.winner === seat) total.classList.add("win");
    tr.appendChild(total);
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  body.appendChild(table);
  if (final.winner === null) {
    body.appendChild(node("p", "tally-line", "Level on score and on completed rows — the rulebook calls that a draw."));
  }
  if (final.exhausted) {
    body.appendChild(node("p", "tally-line", "The bag and the lid ran dry, so the game stopped early."));
  }
  sheets.push({ title: final.headline, body, cta: "Deal another" });
}

/** Resolves once no overlay is up — tiles should never fly behind a sheet. */
function waitForSheets() {
  if (ui.overlay.classList.contains("hidden") && !sheets.length) return Promise.resolve();
  return new Promise((done) => sheetWaiters.push(done));
}

function showNextSheet() {
  const sheet = sheets.shift();
  if (!sheet) {
    ui.overlay.classList.add("hidden");
    const waiting = sheetWaiters;
    sheetWaiters = [];
    waiting.forEach((done) => done());
    return;
  }
  ui.overlayTitle.textContent = sheet.title;
  ui.overlayBody.innerHTML = "";
  ui.overlayBody.appendChild(sheet.body);
  ui.overlayOk.textContent = sheet.cta || "Carry on";
  ui.overlay.classList.remove("hidden");
  ui.overlayOk.focus();
}

/* --------------------------------------------------------------------- wiring */
ui.setup.addEventListener("submit", newGame);
ui.think.addEventListener("change", () => {
  if (session) session.thinkTimeS = currentThinkTime();
  if (S) adopt();
});
ui.hint.addEventListener("click", askHint);
ui.cancel.addEventListener("click", () => {
  sel = null;
  suggestion = null;
  render();
});
ui.overlayOk.addEventListener("click", () => {
  const wasFinal = ui.overlayOk.textContent === "Deal another";
  showNextSheet();
  if (wasFinal && !sheets.length) newGame();
});
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (!ui.overlay.classList.contains("hidden")) showNextSheet();
  else if (sel) {
    sel = null;
    suggestion = null;
    render();
  }
});

bootEngine();
