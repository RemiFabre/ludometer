/* Azul against the net — the table, in your browser.
 *
 * The reusable half of this page lives in ../ui/: the boards, the status band,
 * the move log, the inline scoring panels and the tile flights are the same
 * framework-free modules the local GUI uses (a copy of web/play/ui/, see its
 * PORTING.md). This file is the part that is specific to *this* page — the part
 * the local GUI hands to a Flask server: it owns the game (`GameSession`, a port
 * of the server's session object), asks a Web Worker running ONNX for the AI's
 * moves, and decides what the status band says.
 *
 * How a turn plays out — and why there is no overlay, sheet or toast anywhere on
 * this page:
 *
 *   1. you click a colour, then a row; with coach mode on, the search rates your
 *      move first and the band shows the rating clock;
 *   2. the tiles fly, in a straight line, from the dish to your board, and the
 *      dish's leftovers fly to the middle;
 *   3. if that ended the round, full pattern lines fly to the wall and the floor
 *      line flies to the lid, and the round's scoring appears *inline* under the
 *      boards — the position behind it stays readable;
 *   4. the AI thinks against the clock in the status band (on your CPU, so the
 *      band also counts the positions it has visited), then its move plays out on
 *      the right-hand board exactly the same way.
 *
 * Input is locked from the moment a move is played until the last tile lands, and
 * for no longer than that.
 *
 * One known gap, shared with the local GUI: when the AI moves twice in a row
 * across a round boundary, only its first move is animated — the intermediate
 * position is never drawn. The second move still appears in the log and on the
 * board.
 */
"use strict";

import { flyTiles, prefersReducedMotion, sleep } from "../ui/animate.js";
import { createBoard, createMiddle } from "../ui/board.js";
import { COLORS, FLOOR, node, preview } from "../ui/dom.js";
import { renderLog } from "../ui/log.js";
import { clearScoring, renderFinalPanel, renderRoundPanel } from "../ui/scoring.js";
import { createStatus } from "../ui/status.js";
import { GameSession } from "./game.js";
import { describeAction } from "./report.js";

const el = (id) => document.getElementById(id);
const ui = {
  matchup: el("matchup"), setup: el("setup"), seed: el("seed"), first: el("first"),
  think: el("think"), deal: el("deal"), engineBar: el("engine-bar"),
  engineText: el("engine-text"), aboutMeta: el("about-meta"), middle: el("middle"),
  prompt: el("prompt"), hint: el("hint"), cancel: el("cancel"), fly: el("fly"),
  scoring: el("scoring"), log: el("log"), coach: el("coach"),
  coachField: el("coach-field"), coachLegend: el("coach-legend"), supply: el("supply"),
};

const status = createStatus(el("status"));
const middle = createMiddle(ui.middle, { onPick: pick });
const boards = {
  human: createBoard(el("board-human"), { seat: 0, interactive: true, onPlay: play }),
  ai: createBoard(el("board-ai"), { seat: 1 }),
};

let session = null;    // the GameSession that owns the rules
let S = null;          // its last snapshot, what the page draws from
let sel = null;        // {source, color} — tiles you are holding
let suggestion = null; // {source, color, dest} from the policy head
let busy = false;      // a move is being computed, or is still playing out
let meta = null;       // model/model_meta.json
let engineReady = false;
let liveSims = 0;      // positions the current search has visited
let coachOn = false;   // rate my moves with the AI's own search
let notice = "";       // a passing message, shown in the band, never over the board
let noticeTimer = null;

/* The coach's own clock, as in ludometer/gui/coach.py: it runs *before* your move
 * is committed, so it is deliberately shorter than the opponent's can be. */
const COACH_THINK_S = 2;
const COACH_MAX_THINK_S = 3;
const COACH_LEGEND =
  "Coach mode rates your move with the AI's own search: 0.00 = the move it would " +
  "have played, −1 ≈ a whole win thrown away. It thinks before your move lands, " +
  "so each turn takes a couple of seconds longer.";

/* ------------------------------------------------------------------ plumbing */
/** A passing message. It goes in the status band's detail line — never a pop-up. */
function say(message) {
  notice = message || "";
  if (noticeTimer) clearTimeout(noticeTimer);
  if (notice) {
    noticeTimer = setTimeout(() => {
      notice = "";
      if (S) renderStatus();
    }, 6000);
  }
  if (S) renderStatus();
}

function setBusy(on) {
  const changed = busy !== on;
  busy = on;
  document.body.classList.toggle("locked", on);
  ui.deal.disabled = on || !engineReady;
  ui.hint.disabled = on || !engineReady || !S || !S.your_turn;
  // the tiles you may pick up depend on this, and unlocking has to give them
  // back — safe here because tiles only ever fly while busy
  if (changed && S) render();
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
  status.set({
    headline: "The net could not be loaded",
    detail: reason + " — reload the page to try again.",
    tone: "end",
  });
}

async function bootEngine() {
  status.set({ headline: "Loading the net", detail: "it runs in this tab, so it downloads once", tone: "idle" });
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
        if (msg.type !== "loading") return;
        // GitHub Pages gzips the .onnx, so content-length is the *compressed*
        // size while `received` counts decompressed bytes: prefer the true size
        // from the metadata, and never show more than 100%.
        const total = (meta && meta.onnx_bytes) || msg.total;
        if (!total) return;
        const pct = Math.min(100, Math.round((msg.received / total) * 100));
        ui.engineText.textContent = `Downloading the net — ${pct}% of ${(total / 1e6).toFixed(1)} MB`;
        status.set({
          headline: `Downloading the net — ${pct}%`,
          detail: "nothing is sent anywhere; the net plays from your own machine",
          tone: "idle",
        });
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

/* ---------------------------------------------------------------- the boards */
function aiLabel() {
  return S && S.agent_name ? "AI · " + S.agent_name : "AI";
}

/** Fix the two boards' seats to this game's seating (you are always on the left). */
function seatBoards() {
  boards.human = createBoard(el("board-human"), {
    seat: S.human_seat, interactive: true, onPlay: play,
  });
  boards.ai = createBoard(el("board-ai"), { seat: S.ai_seat });
}

function render() {
  if (!S) return;
  const st = S.state;
  renderStatus();
  middle.render({
    state: st,
    legalActions: S.human_legal_actions,
    canPick: S.your_turn && !st.is_terminal && !busy,
    selection: sel,
  });
  boards.human.render({
    state: st,
    legalActions: S.human_legal_actions,
    selection: sel,
    title: "You",
    toMove: !st.is_terminal && st.current_player === S.human_seat,
    highlightRow: suggestion ? suggestion.dest : undefined,
  });
  boards.ai.render({
    state: st,
    title: aiLabel(),
    toMove: !st.is_terminal && st.current_player === S.ai_seat,
  });
  renderSupply();
  renderLog(ui.log, S.log || []);
  ui.cancel.classList.toggle("hidden", !sel);
  ui.hint.disabled = busy || !engineReady || !S.your_turn;
  syncCoach();
  if (suggestion) {
    const dish = middle.sourceEl(suggestion.source);
    if (dish) {
      dish.classList.add("picked");
      setTimeout(() => dish.classList.remove("picked"), 1600);
    }
  }
}

function renderStatus() {
  const st = S.state;
  const info = S.opponent_info || {};
  const rating = typeof info.elo === "number" ? " (" + Math.round(info.elo) + " Elo)" : "";
  ui.matchup.textContent = "you versus " + S.agent_name + rating + " · seed " + S.seed;
  status.setScore(st.scores[S.human_seat], st.scores[S.ai_seat], "You", "AI");

  if (st.is_terminal) {
    const mine = st.scores[S.human_seat];
    const theirs = st.scores[S.ai_seat];
    const verb = mine > theirs ? "You won " : theirs > mine ? "The AI won " : "A draw, ";
    status.set({
      headline: verb + mine + "–" + theirs,
      detail: "The final scoring is below; the board stays exactly as it ended.",
      tone: "end",
    });
    ui.prompt.textContent = "Deal again for another game.";
    return;
  }
  if (S.your_turn) {
    status.set({
      headline: sel
        ? "Your turn — pick a row for your " + COLORS[sel.color] + " tiles"
        : "Your turn — pick a colour",
      detail: turnDetail(),
      tone: "you",
    });
    ui.prompt.textContent = sel
      ? "Or press Escape to put them back."
      : "Take every tile of one colour from a factory, or from the middle.";
  } else {
    status.set({ headline: "The AI is choosing a move", detail: turnDetail(), tone: "ai" });
    ui.prompt.textContent = "";
  }
}

/** The line under the headline: where the game is, and what the AI just did. */
function turnDetail() {
  const st = S.state;
  const bits = ["Round " + (st.round + 1), st.tiles_left + " tiles left on the table"];
  const last = S.last_ai_move;
  if (last) {
    bits.push("AI " + last.text);
    if (last.search_text) bits.push(last.search_text);
  }
  if (notice) bits.push(notice);
  return bits.join(" · ");
}

function renderSupply() {
  const st = S.state;
  ui.supply.innerHTML = "";
  [["Bag", st.bag, null], ["Lid", st.lid, "lid-row"]].forEach(([label, counts, id]) => {
    const row = node("div", "supply-row");
    if (id) row.id = id;
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
  row.appendChild(node("span", "swatch", st.tiles_left + " tiles still to take"));
  ui.supply.appendChild(row);
}

/* -------------------------------------------------------------- game control */
/** Seconds the AI may think per move — 0 means "policy head, no search". */
function currentThinkTime() {
  const seconds = Number(ui.think.value);
  return Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
}

/** The coach's budget for an opponent that thinks for `thinkTimeS` seconds. */
function coachBudget() {
  return Math.min(currentThinkTime() || COACH_THINK_S, COACH_MAX_THINK_S);
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
    say("the net is still loading");
    return;
  }
  let seed = Math.floor(Math.random() * (1 << 30));
  const typed = ui.seed.value.trim();
  if (typed) {
    if (!/^-?\d+$/.test(typed)) {
      say("the seed must be a whole number");
      return;
    }
    seed = Number(typed) >>> 0;
  }
  clearScoring(ui.scoring);
  session = new GameSession({
    seed,
    humanPlaysFirst: ui.first.checked,
    agentName: meta ? meta.checkpoint : "the net",
    opponentInfo: meta ? { checkpoint: meta.checkpoint, elo: meta.elo, run: meta.run } : {},
    thinkTimeS: currentThinkTime(),
    think,
  });
  adopt({ reseat: true });
  say("new tiles dealt");
  // the AI opens when you took the second seat
  if (session.aiTurn) resumeIfPending();
}

function adopt(options) {
  const first = !S || (options && options.reseat);
  S = session.snapshot();
  sel = null;
  suggestion = null;
  if (first) seatBoards();
  render();
}

function pick(source, color) {
  if (busy || !S || !S.your_turn) return;
  sel = sel && sel.source === source && sel.color === color ? null : { source, color };
  suggestion = null;
  render();
}

/**
 * One turn: your move (rated first if coach mode is on), then the AI's reply.
 *
 * The tiles are animated from the *current* DOM, before the new state is drawn,
 * so what you see leaving the dish is what actually left it — which is why the
 * move is described before `playHuman` applies it.
 */
async function play(id) {
  if (busy || !session || !S || !S.your_turn) return;
  const move = localMove(id);
  sel = null;
  setBusy(true);
  clearScoring(ui.scoring);
  try {
    const coach = coachOn ? await rateMove(id) : null;
    const { move: applied, reports } = session.playHuman(id);
    if (coach) {
      const entry = session.log[applied.log_n];
      if (entry) entry.coach = coach;
    }
    await animateTake(move, boards.human, middle);
    await settle(reports, "human");
    if (!session.aiTurn) return;
    await runAiTurn();
  } catch (err) {
    status.stopClock();
    say(err.message);
    adopt();
  } finally {
    setBusy(false);
    resumeIfPending(); // a failed search leaves the AI owing a move
  }
}

/** If the position says the AI still owes a move, let it move. */
async function resumeIfPending() {
  if (busy || !session || !session.aiTurn) return;
  setBusy(true);
  try {
    await runAiTurn();
  } catch (err) {
    status.stopClock();
    say(err.message);
    adopt();
  } finally {
    setBusy(false);
  }
}

/** The AI's turn: the clock runs in the status band, then its move plays out. */
async function runAiTurn() {
  const firstReport = session.roundReports.length;
  const budget = session.thinkTimeS;
  liveSims = 0;
  const thinking = session.aiReplies();
  status.set({ headline: "The AI is thinking", detail: turnDetail(), tone: "ai", keepClock: true });
  status.startClock({
    budget,
    label: (spent, cap) => {
      // it is your CPU doing the work, so the band says how much work that is
      const counted = liveSims ? " · " + liveSims.toLocaleString() + " positions" : "";
      return cap
        ? "AI is thinking — " + spent.toFixed(1) + "s of " + cap + "s" + counted
        : "AI is picking a move";
    },
  });
  let moves;
  try {
    moves = await thinking;
  } finally {
    status.stopClock();
  }
  if (moves.length) {
    const first = moves[0];
    status.set({
      headline: "AI " + first.text,
      detail: first.search_text || turnDetail(),
      tone: "ai",
    });
    await animateTake(first, boards.ai, middle);
  }
  await settle(session.reportsSince(firstReport), "ai");
}

/**
 * Show what just happened: the round-end animations first (wall tiling, floor to
 * the lid), then the new position and the inline scoring for the round.
 */
async function settle(reports, mover) {
  if (reports.length) {
    status.set({
      headline: "Round " + (reports[0].round + 1) + " scoring",
      detail: "full lines move to the wall, the floor line goes to the lid",
      tone: "scoring",
    });
    await animateTiling(reports);
  }
  adopt();
  if (reports.length) {
    const last = reports[reports.length - 1];
    flashWall(last);
    if (!last.game_over) renderRoundPanel(ui.scoring, last, sides());
  }
  if (S.state.is_terminal && S.final) {
    renderFinalPanel(ui.scoring, S.final, sides());
  } else if (mover === "ai" && S.last_ai_move) {
    // let the AI's own move stand as the headline for a beat before your turn
    status.set({ headline: "AI " + S.last_ai_move.text, detail: turnDetail(), tone: "ai" });
    await sleep(500);
  }
  render();
}

function sides() {
  return [[S.human_seat, "You"], [S.ai_seat, aiLabel()]];
}

/* ---------------------------------------------------------------- coach mode */
/**
 * Rate the move you are about to play with the AI's own search.
 *
 * Same definition as the local GUI (ludometer/gui/coach.py): the search runs on
 * the position *before* your move is applied, and the verdict is the gap between
 * your move's Q at the root and the best explored child's.
 */
async function rateMove(id) {
  const budget = coachBudget();
  status.set({
    headline: "Rating your move",
    detail: "the AI is searching your position",
    tone: "ai",
    keepClock: true,
  });
  status.startClock({
    budget,
    label: (spent, cap) =>
      cap
        ? "Rating your move — " + spent.toFixed(1) + "s of " + cap + "s"
        : "Rating your move — " + spent.toFixed(1) + "s",
  });
  try {
    const reply = await ask({
      type: "rate",
      setup: session.state.toSetup(),
      actionId: id,
      budgetS: budget,
    });
    return reply.coach;
  } catch (err) {
    return { unrated: true, reason: err.message };
  } finally {
    status.stopClock();
  }
}

function syncCoach() {
  ui.coach.checked = coachOn;
  ui.coach.disabled = !engineReady;
  ui.coachField.classList.toggle("off", !coachOn);
  ui.coachLegend.textContent = COACH_LEGEND;
  ui.coachLegend.hidden = !coachOn;
}

/* --------------------------------------------------------------------- hints */
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
    say("the policy head would " + move.text);
  } catch (err) {
    say(err.message);
  } finally {
    setBusy(false);
  }
}

/* ----------------------------------------------------------------- animation */
/** What your click is about to do, worked out from the position on screen. */
function localMove(id) {
  const source = Math.floor(id / 30);
  const color = Math.floor((id % 30) / 6);
  const dest = id % 6;
  const p = preview(S.state, S.human_seat, source, color, dest);
  return {
    source, color, dest,
    count: p.count, placed: p.placed, overflow: p.overflow, took_marker: p.takesMarker,
  };
}

function lidTarget() {
  return document.getElementById("lid-row") || ui.supply;
}

/** Where a move's tiles are going, in landing order: line, then floor, then lid. */
function travelTargets(board, move) {
  const targets = [];
  if (move.dest !== FLOOR && move.placed) {
    const slots = board.lineSlots(move.dest);
    targets.push.apply(targets, slots.slice(Math.max(0, slots.length - move.placed)));
  }
  const floorSlots = board.floorSlots();
  let taken = move.took_marker ? 1 : 0; // the marker claims the first free slot
  while (targets.length < move.count && taken < floorSlots.length) {
    targets.push(floorSlots[taken++]);
  }
  const lid = lidTarget();
  while (targets.length < move.count) targets.push(lid); // the rest overflow to the lid
  return targets;
}

/** Tiles leave a dish; the dish's other colours are pushed into the middle. */
async function animateTake(move, board, table) {
  if (prefersReducedMotion()) return;
  const dish = table.sourceEl(move.source);
  const row = board.lineRow(move.dest);
  if (dish) dish.classList.add("picked");
  if (row) row.classList.add("incoming");

  const taken = table.sourceTiles(move.source, move.color, move.count);
  const targets = travelTargets(board, move);
  const flights = taken.map((from, i) => ({ from, to: targets[i], color: move.color }));

  // leftovers slide to the middle; the marker, if taken, drops onto the floor
  const centre = table.centerEl();
  table.remainderTiles(move.source, move.color).forEach((from) => {
    flights.push({ from, to: centre, color: Number(from.dataset.color) });
  });
  if (move.took_marker) {
    const chip = centre.querySelector(".marker");
    const slot = board.floorSlots()[0];
    if (chip && slot) flights.push({ from: chip, to: slot, color: "marker" });
  }

  await flyTiles(flights, { layer: ui.fly });
  if (dish) dish.classList.remove("picked");
  if (row) row.classList.remove("incoming");
}

/** Round end: full lines travel to the wall, the floor line travels to the lid. */
async function animateTiling(reports) {
  if (prefersReducedMotion() || !reports.length) return;
  const report = reports[reports.length - 1];
  const wall = [];
  const lid = [];
  const lidEl = lidTarget();
  [[S.human_seat, boards.human], [S.ai_seat, boards.ai]].forEach(([seat, board]) => {
    const player = report.players[seat];
    if (!player) return;
    player.tiles.forEach((t) => {
      const tiles = board.lineTiles(t.row);
      const from = tiles[tiles.length - 1] || board.lineRow(t.row);
      const to = board.wallCell(t.row, t.col);
      if (from && to) wall.push({ from, to, color: t.color, hide: false });
    });
    board.floorTiles().forEach((from) => {
      lid.push({ from, to: lidEl, color: Number(from.dataset.color) });
    });
  });
  await flyTiles(wall, { layer: ui.fly, duration: 520, stagger: 70 });
  if (lid.length) {
    lidEl.classList.add("receiving");
    await flyTiles(lid, { layer: ui.fly, duration: 420, stagger: 40 });
    setTimeout(() => lidEl.classList.remove("receiving"), 400);
  }
}

/** The tiles that just reached the wall glaze over as they land. */
function flashWall(report) {
  if (prefersReducedMotion()) return;
  [[S.human_seat, boards.human], [S.ai_seat, boards.ai]].forEach(([seat, board]) => {
    const player = report.players[seat];
    if (!player) return;
    player.tiles.forEach((t, i) => {
      const cell = board.wallTile(t.row, t.col);
      if (!cell) return;
      cell.style.animationDelay = i * 70 + "ms";
      cell.classList.add("landing");
    });
  });
}

/* -------------------------------------------------------------------- wiring */
ui.setup.addEventListener("submit", newGame);
ui.think.addEventListener("change", () => {
  if (session) session.thinkTimeS = currentThinkTime();
  if (S) render();
});
ui.hint.addEventListener("click", askHint);
ui.cancel.addEventListener("click", () => {
  sel = null;
  suggestion = null;
  render();
});
ui.coach.addEventListener("change", () => {
  coachOn = ui.coach.checked;
  syncCoach();
  say(
    coachOn
      ? "coach mode on — your moves are scored by the AI's own search"
      : "coach mode off"
  );
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && sel) {
    sel = null;
    suggestion = null;
    render();
  }
});

syncCoach();
bootEngine();
