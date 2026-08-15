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
 *   1. you click a colour and the selection is acted out at once: your tiles fly
 *      to a small "hand" tray on the middle panel and the dish's leftovers fly to
 *      the centre — a preview of the move, undone the instant you change your
 *      mind (Escape, the Clear button, or another colour);
 *   2. you click a row and the held tiles continue from the hand to your board;
 *      nothing ever waits for the coach — with coach mode on, the rating runs in
 *      the background *after* the turn and fills in on the move's log entry;
 *   3. if that ended the round, full pattern lines fly to the wall — each tile
 *      popping the points it earns as it lands, the floor line popping its cost —
 *      and the round's scoring appears *inline* under the boards;
 *   4. the AI thinks against the clock in the status band (on your CPU, so the
 *      band also counts the positions it has visited), then its move plays out on
 *      the right-hand board exactly the same way.
 *
 * Input is locked from the moment a move is played until the last tile lands, and
 * for no longer than that.
 *
 * When the AI moves twice in a row across a round boundary, both moves are
 * animated: each move carries the position it was played from (`state_before`),
 * so the refilled table the second move starts on is drawn before it plays.
 *
 * ← and → walk back through the game — pure replay of positions this page has
 * already produced (see ../ui/history.js), with no re-search and no effect on
 * the live game.
 */
"use strict";

import { animated, flyTiles, initSpeed, scaled, sleep } from "../ui/animate.js";
import { createBoard, createMiddle } from "../ui/board.js";
import { COLORS, CUM_PENALTY, FLOOR, node } from "../ui/dom.js";
import { bindHistoryKeys, createHistory } from "../ui/history.js";
import { renderLog } from "../ui/log.js";
import { popScore } from "../ui/popups.js";
import { clearScoring, renderFinalPanel, renderRoundPanel } from "../ui/scoring.js";
import { createSettings } from "../ui/settings.js";
import { createStatus } from "../ui/status.js";
import { analyticsOn, track } from "./analytics.js";
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
  settings: el("settings"), nav: el("nav"),
  hand: el("hand"), handTiles: el("hand-tiles"), pops: el("pops"),
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
let held = null;       // the acted-out selection: parked tile clones + hidden originals
let navToken = 0;      // invalidates a history-step animation when another step lands

initSpeed(); // before anything can animate: the stored speed, or 1×
const status = createStatus(el("status"));
const settings = createSettings(ui.settings, { popups: true });
const middle = createMiddle(ui.middle, { onPick: pick });
const boards = {
  human: createBoard(el("board-human"), { seat: 0, interactive: true, onPlay: play }),
  ai: createBoard(el("board-ai"), { seat: 1 }),
};
const nav = createHistory(ui.nav, {
  log: () => (S && S.log) || [],
  enabled: () => !busy,
  onChange: onNavChange,
});

/* The coach's own clock, as in ludometer/gui/coach.py — but unlike the Python
 * GUI it runs *after* the turn, in the background, so your move lands the
 * moment you click it. It is still kept shorter than the opponent's budget. */
const COACH_THINK_S = 2;
const COACH_MAX_THINK_S = 3;
const COACH_LEGEND =
  "Coach mode rates your move with the AI's own search: 0.00 = the move it would " +
  "have played, −1 ≈ a whole win thrown away. Your move lands at once; the " +
  "verdict fills in on its log entry when the search finishes.";

/* ------------------------------------------------------------------ plumbing */
/** A passing message. It goes in the status band's detail line — never a pop-up. */
function say(message) {
  notice = message || "";
  if (noticeTimer) clearTimeout(noticeTimer);
  if (notice) {
    noticeTimer = setTimeout(() => {
      notice = "";
      if (S) renderStatus(nav.frame());
    }, 6000);
  }
  if (S) renderStatus(nav.frame());
}

function setBusy(on) {
  const changed = busy !== on;
  busy = on;
  document.body.classList.toggle("locked", on);
  ui.deal.disabled = on || !engineReady;
  ui.hint.disabled = on || !engineReady || !S || !S.your_turn;
  nav.draw(); // browsing the history is off while tiles are in the air
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

/**
 * Draw the page.
 *
 * There are two things it can be showing: the live game, or a position out of
 * the history. Browsing is read-only by construction — the board is drawn from
 * the frame with no legal actions and no selection, so there is nothing to
 * click even before the buttons are disabled.
 */
function render() {
  if (!S) return;
  clearHeld(); // a redraw rebuilds the dishes, so any acted-out selection resets
  const frame = nav.frame();
  const live = !frame;
  const st = live ? S.state : frame.state;
  renderStatus(frame);
  middle.render({
    state: st,
    legalActions: live ? S.human_legal_actions : [],
    canPick: live && S.your_turn && !st.is_terminal && !busy,
    selection: live ? sel : null,
  });
  boards.human.render({
    state: st,
    legalActions: live ? S.human_legal_actions : [],
    selection: live ? sel : null,
    title: "You",
    toMove: live && !st.is_terminal && st.current_player === S.human_seat,
    highlightRow: live && suggestion ? suggestion.dest : undefined,
  });
  boards.ai.render({
    state: st,
    title: aiLabel(),
    toMove: live && !st.is_terminal && st.current_player === S.ai_seat,
  });
  renderSupply(st);
  renderLog(ui.log, (live ? S.log : frame.log) || []);
  ui.cancel.classList.toggle("hidden", !live || !sel);
  ui.hint.disabled = busy || !engineReady || !live || !S.your_turn;
  syncCoach();
  if (live && suggestion) {
    const dish = middle.sourceEl(suggestion.source);
    if (dish) {
      dish.classList.add("picked");
      setTimeout(() => dish.classList.remove("picked"), 1600);
    }
  }
}

/** Draw one position and nothing else — a history frame, or a step in a turn. */
function drawPosition(st) {
  if (!st) return;
  middle.render({ state: st, legalActions: [], canPick: false, selection: null });
  boards.human.render({ state: st, title: "You", toMove: false });
  boards.ai.render({ state: st, title: aiLabel(), toMove: false });
  renderSupply(st);
}

function renderStatus(frame) {
  const st = frame ? frame.state : S.state;
  const info = S.opponent_info || {};
  const rating = typeof info.elo === "number" ? " (" + Math.round(info.elo) + " Elo)" : "";
  ui.matchup.textContent = "you versus " + S.agent_name + rating + " · seed " + S.seed;
  status.setScore(st.scores[S.human_seat], st.scores[S.ai_seat], "You", "AI");

  if (frame) {
    status.set({
      headline: "Viewing move " + frame.ply + " of " + frame.of,
      detail: "← and → step through the game · End, or Latest, returns to play",
      tone: "history",
    });
    ui.prompt.textContent = "This is a recorded position. The live game is untouched.";
    return;
  }

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

function renderSupply(st) {
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
  track("game-start");
  // the AI opens when you took the second seat
  if (session.aiTurn) resumeIfPending();
}

function adopt(options) {
  const first = !S || (options && options.reseat);
  S = session.snapshot();
  sel = null;
  suggestion = null;
  if (first) {
    seatBoards();
    nav.reset();
  }
  noteFrames((options && options.moves) || []);
  render();
}

/**
 * File these moves' positions into the move history.
 *
 * Every move carries the position it was played from, and the position a move
 * was played from is the position *after* the move before it — so ply k's frame
 * is ply k+1's `state_before`, and the tail of the list is the live position.
 * Nothing is recomputed and nothing is searched again.
 */
function noteFrames(moves) {
  moves.forEach((m) => {
    if (m && m.state_before && typeof m.ply === "number") nav.note(m.ply - 1, m.state_before);
  });
  nav.note(S.ply, S.state);
}

function pick(source, color) {
  if (busy || !S || !S.your_turn || nav.browsing()) return;
  sel = sel && sel.source === source && sel.color === color ? null : { source, color };
  suggestion = null;
  render();
  if (sel) holdSelection();
}

/* ------------------------------------------------------- the held selection */
/**
 * Act the selection out, immediately: the tiles you clicked fly to the hand
 * tray on the middle panel, and the dish's leftovers fly to the centre — the
 * table as it *will* look if you commit. It is pure picture: parked clones
 * over hidden originals, with the game state untouched, so any redraw (another
 * colour, Escape, browsing the history) puts everything back at once.
 */
function holdSelection() {
  if (!sel || !animated() || !S) return;
  const { source, color } = sel;
  const taken = middle.sourceTiles(source, color);
  if (!taken.length) return;
  ui.handTiles.innerHTML = "";
  const slots = taken.map(() => {
    const s = node("div", "slot");
    ui.handTiles.appendChild(s);
    return s;
  });
  ui.hand.hidden = false;
  const rest = middle.remainderTiles(source, color);
  held = { source, color, taken: [], rest: [], hidden: taken.concat(rest) };
  flyTiles(
    taken.map((from, i) => ({ from, to: slots[i], color })),
    { layer: ui.fly, keep: true, collect: held.taken }
  );
  if (rest.length) {
    // the leftovers spread into a neat row at the centre of the dish they join
    const centre = middle.centerEl().getBoundingClientRect();
    const tw = rest[0].getBoundingClientRect().width || 32;
    const gap = 4;
    const x0 = centre.left + (centre.width - (rest.length * (tw + gap) - gap)) / 2;
    const y = centre.top + (centre.height - tw) / 2;
    flyTiles(
      rest.map((from, i) => ({
        from,
        to: { left: x0 + i * (tw + gap), top: y, width: tw, height: tw },
        color: Number(from.dataset.color),
      })),
      { layer: ui.fly, keep: true, collect: held.rest }
    );
  }
}

/** Put the acted-out selection back: clones gone, originals visible, tray shut. */
function clearHeld() {
  if (!held) {
    ui.hand.hidden = true;
    return;
  }
  held.taken.forEach((g) => g.remove());
  held.rest.forEach((g) => g.remove());
  held.hidden.forEach((elm) => {
    if (elm && elm.style) elm.style.visibility = "";
  });
  ui.hand.hidden = true;
  ui.handTiles.innerHTML = "";
  held = null;
}

/**
 * Hand the held selection to the move that commits it. The caller now owns the
 * clones (they become the move's flight sources), and render() will no longer
 * reset them.
 */
function takeHeld() {
  const h = held;
  held = null;
  if (h) ui.hand.hidden = true;
  return h;
}

/**
 * One turn: your move — applied and animated at once — then the AI's reply.
 *
 * The tiles are animated from the *current* DOM, before the new state is drawn,
 * so what you see leaving the dish is what actually left it. With coach mode on
 * the position is snapshotted first and the rating is queued: it runs on the
 * worker *after* the AI has replied, and fills in on the move's log entry —
 * nothing about your move, the board or the animations ever waits for it.
 */
async function play(id) {
  if (busy || !session || !S || !S.your_turn || nav.browsing()) return;
  sel = null;
  const carried = takeHeld(); // the acted-out selection becomes the move's take-off
  setBusy(true);
  clearScoring(ui.scoring);
  try {
    const setupBefore = coachOn ? session.state.toSetup() : null;
    const { move: applied, reports } = session.playHuman(id);
    if (setupBefore) queueRating(setupBefore, id, session.log[applied.log_n]);
    await settle([applied], boards.human, reports, "human", carried);
    if (session.aiTurn) await runAiTurn();
  } catch (err) {
    status.stopClock();
    say(err.message);
    adopt();
  } finally {
    // whatever happened, no clone may outlive the turn that owned it
    if (carried) carried.taken.concat(carried.rest).forEach((g) => g.remove());
    setBusy(false);
    resumeIfPending(); // a failed search leaves the AI owing a move
    flushRatings(); // now the table is quiet, let the coach think
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
  await settle(moves, boards.ai, session.reportsSince(firstReport), "ai");
}

/**
 * Play a list of moves out on the table, one after another.
 *
 * The AI can move twice in a row — a round boundary is resolved inside the
 * engine's `apply`, and the next round is opened by whoever holds the marker.
 * Its second move is therefore played on a table that was scored and refilled
 * inside the first one, and which this page had never drawn. Each move carries
 * the position it was played from, so we draw that first and then animate: both
 * moves are shown, in order, from the right board.
 */
async function playMoves(moves, board, reports, mover, carried) {
  let taken = 0;
  for (let i = 0; i < moves.length; i++) {
    const move = moves[i];
    if (i > 0 && move.state_before) {
      drawPosition(move.state_before);
      await sleep(420); // a beat, so the refilled table registers as a new one
    }
    if (mover === "ai") {
      status.set({
        headline: "AI " + move.text,
        detail: move.search_text || turnDetail(),
        tone: "ai",
      });
    }
    await animateTake(move, board, middle, i === 0 ? carried : null);
    // an observability hook, so a test can prove that *every* move animates —
    // including the second of the AI's double move across a round boundary
    document.dispatchEvent(
      new CustomEvent("azul:animated", { detail: { ply: move.ply, side: move.side } })
    );
    if (move.ended_round && reports[taken]) {
      const report = reports[taken++];
      status.set({
        headline: "Round " + (report.round + 1) + " scoring",
        detail: "full lines move to the wall, the floor line goes to the lid",
        tone: "scoring",
      });
      await animateTiling(report);
    }
  }
}

/** Play out `moves`, then show the position they led to and the round's scoring. */
async function settle(moves, board, reports, mover, carried) {
  await playMoves(moves, board, reports, mover, carried);
  adopt({ moves });
  // the leftovers' parked clones bow out in the same paint that draws the real
  // tiles they stood in for
  if (carried) carried.rest.forEach((g) => g.remove());
  if (reports.length) {
    const last = reports[reports.length - 1];
    flashWall(last);
    if (!last.game_over) renderRoundPanel(ui.scoring, last, sides());
  }
  if (S.state.is_terminal && S.final) {
    renderFinalPanel(ui.scoring, S.final, sides());
    track(
      S.final.winner_side === "human"
        ? "game-win"
        : S.final.winner_side === "ai"
          ? "game-loss"
          : "game-draw"
    );
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
/* Same definition as the local GUI (ludometer/gui/coach.py): the search runs on
 * the position *before* your move was applied — snapshotted at the moment you
 * clicked — and the verdict is the gap between your move's Q at the root and
 * the best explored child's. The difference is *when*: the rating is queued and
 * runs on the worker after the opponent has replied, so your move, the flights
 * and the AI's answer never wait for it. Until it lands, the move's log entry
 * says "rating…". */
const ratings = []; // {setup, actionId, entry, forSession} still to be rated
let ratingNow = false;

function queueRating(setup, actionId, entry) {
  if (!entry) return;
  entry.coach = { pending: true };
  ratings.push({ setup, actionId, entry, forSession: session });
}

/** Work the rating queue, one search at a time. Reentry-safe; never throws. */
async function flushRatings() {
  if (ratingNow) return;
  ratingNow = true;
  try {
    while (ratings.length) {
      const job = ratings.shift();
      let verdict;
      try {
        const reply = await ask({
          type: "rate",
          setup: job.setup,
          actionId: job.actionId,
          budgetS: coachBudget(),
        });
        verdict = reply.coach;
      } catch (err) {
        verdict = { unrated: true, reason: err.message };
      }
      job.entry.coach = verdict;
      // the entry lives in the session's own log, so any later redraw shows the
      // verdict; redraw now only if the table is quiet and still this game's
      if (job.forSession === session && S && !busy) render();
    }
  } finally {
    ratingNow = false;
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
  if (!session || !S || !S.your_turn || busy || nav.browsing()) return;
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

/**
 * Tiles leave a dish; everything else in that dish is pushed into the middle.
 *
 * Both halves matter: the tiles you took travel to your board, and the dish's
 * *remainder* travels to the centre. Watching the leftovers arrive in the
 * middle is how you learn what the next player can take.
 *
 * When the move commits a selection that was already acted out (`carried`, from
 * takeHeld()), the story continues instead of restarting: the tiles fly on from
 * the hand tray — wherever they are, even mid-air — and the leftovers, already
 * sitting in the middle, stay put. The dish's freshly-redrawn originals are
 * hidden for the duration; the redraw at the end of the turn replaces them.
 */
async function animateTake(move, board, table, carried) {
  const dish = table.sourceEl(move.source);
  const row = board.lineRow(move.dest);
  if (dish) dish.classList.add("picked");
  if (row) row.classList.add("incoming");

  const targets = travelTargets(board, move);
  const handoff =
    carried && carried.source === move.source && carried.color === move.color;
  let flights;
  if (handoff) {
    const rects = carried.taken.map((g) => g.getBoundingClientRect());
    carried.taken.forEach((g) => g.remove());
    // what is visually in the hand and in the middle must not also sit in the dish
    table.sourceTiles(move.source, move.color, move.count).forEach((elm) => {
      elm.style.visibility = "hidden";
    });
    table.remainderTiles(move.source, move.color).forEach((elm) => {
      elm.style.visibility = "hidden";
    });
    flights = rects.map((r, i) => ({ from: r, to: targets[i], color: move.color }));
  } else {
    const taken = table.sourceTiles(move.source, move.color, move.count);
    flights = taken.map((from, i) => ({ from, to: targets[i], color: move.color }));
    // leftovers slide to the middle
    const centreEl = table.centerEl();
    table.remainderTiles(move.source, move.color).forEach((from) => {
      flights.push({ from, to: centreEl, color: Number(from.dataset.color) });
    });
  }
  // the marker, if taken, drops onto the floor
  if (move.took_marker) {
    const chip = table.centerEl().querySelector(".marker");
    const slot = board.floorSlots()[0];
    if (chip && slot) flights.push({ from: chip, to: slot, color: "marker" });
  }

  await flyTiles(flights, { layer: ui.fly });
  popFloorCost(move, board);
  if (dish) dish.classList.remove("picked");
  if (row) row.classList.remove("incoming");
}

/** What this move just added to the floor bill, popped off the floor line. */
function popFloorCost(move, board) {
  const dropped = (move.to_floor || 0) + (move.took_marker ? 1 : 0);
  if (!dropped || !move.state_before) return;
  const me = move.state_before.players[move.player];
  if (!me) return;
  const occupied = me.floor.reduce((a, b) => a + b, 0) + (me.floor_marker ? 1 : 0);
  const cost =
    CUM_PENALTY[Math.min(7, occupied + dropped)] - CUM_PENALTY[Math.min(7, occupied)];
  if (!cost) return;
  popScore(ui.pops, board.floorEl(), String(cost).replace("-", "−"), "loss");
}

/* ------------------------------------------------- animated history steps */
/**
 * A step through the history plays its move — forwards, or in reverse.
 *
 * The plan is read from the *outgoing* frame's DOM before the target frame is
 * drawn; the target is then rendered immediately (browsing must never feel
 * slower than the keypress), and the moved tiles fly over the finished board
 * from where they were to where they now are. A step that crosses a round
 * boundary just redraws — half the board changes there, and a lone flight
 * would tell less of the story than the scoring panel already did.
 */
function onNavChange(change) {
  const token = ++navToken;
  const plan = change ? planNavStep(change) : null;
  render();
  if (plan) flyNavStep(plan, token);
}

function planNavStep(change) {
  if (!animated() || !S) return null;
  const delta = change.to - change.from;
  if (Math.abs(delta) !== 1) return null;
  const ply = Math.max(change.from, change.to); // the move between the two frames
  const entry = (S.log || []).find((e) => e.kind === "move" && e.ply === ply);
  if (!entry || typeof entry.color !== "number") return null;
  const a = nav.stateAt(change.from);
  const b = nav.stateAt(change.to) || (change.to === nav.latest() ? S.state : null);
  if (!a || !b || a.round !== b.round) return null;
  const board = entry.side === "human" ? boards.human : boards.ai;
  const placed = Math.max(0, entry.count - (entry.overflow || 0));
  const floorCount = entry.dest === FLOOR ? entry.count : entry.overflow || 0;
  let from;
  if (delta > 0) {
    // forwards: the tiles leave the dish they were taken from
    from = middle.sourceTiles(entry.source, entry.color, entry.count);
  } else {
    // backwards: they leave the row (and the floor) they had landed on
    const rowTiles = entry.dest !== FLOOR ? board.lineTiles(entry.dest).slice(-placed) : [];
    from = rowTiles.concat(floorCount ? board.floorTiles().slice(-floorCount) : []);
  }
  const rects = from.map((elm) => elm.getBoundingClientRect());
  return { dir: delta, entry, board, placed, floorCount, rects };
}

/** Fly a planned step over the already-rendered target frame. */
async function flyNavStep(plan, token) {
  if (token !== navToken) return;
  const { entry, board } = plan;
  let dests;
  if (plan.dir > 0) {
    const rowTiles =
      entry.dest !== FLOOR ? board.lineTiles(entry.dest).slice(-plan.placed) : [];
    dests = rowTiles.concat(
      plan.floorCount ? board.floorTiles().slice(-plan.floorCount) : []
    );
  } else {
    dests = middle.sourceTiles(entry.source, entry.color, entry.count);
  }
  const flights = [];
  const covered = [];
  dests.forEach((elm, i) => {
    const from = plan.rects[i];
    if (!from || !elm) return;
    elm.style.visibility = "hidden"; // the flight is this tile, mid-journey
    covered.push(elm);
    flights.push({ from, to: elm, color: entry.color, hide: false });
  });
  await flyTiles(flights, { layer: ui.fly, duration: 300, stagger: 35 });
  covered.forEach((elm) => {
    elm.style.visibility = "";
  });
}

/** Round end: full lines travel to the wall, the floor line travels to the lid —
 * each wall tile popping its points as it lands, each floor line its bill. */
async function animateTiling(report) {
  if (!report) return;
  const wall = [];
  const pops = [];
  const lid = [];
  const lidEl = lidTarget();
  const floorBills = [];
  [[S.human_seat, boards.human], [S.ai_seat, boards.ai]].forEach(([seat, board]) => {
    const player = report.players[seat];
    if (!player) return;
    player.tiles.forEach((t) => {
      const tiles = board.lineTiles(t.row);
      const from = tiles[tiles.length - 1] || board.lineRow(t.row);
      const to = board.wallCell(t.row, t.col);
      if (from && to) {
        wall.push({ from, to, color: t.color, hide: false });
        // a rect, not the element: the pop may outlive this render of the cell
        pops.push({ at: to.getBoundingClientRect(), text: "+" + t.points });
      }
    });
    board.floorTiles().forEach((from) => {
      lid.push({ from, to: lidEl, color: Number(from.dataset.color) });
    });
    if (player.floor.penalty) {
      floorBills.push([board.floorEl(), String(player.floor.penalty).replace("-", "−")]);
    }
  });
  // each "+N" appears as its tile touches down — or on a steady beat when the
  // tiles are not animating, so the arithmetic is still told one step at a time
  const land = scaled(520);
  const beat = scaled(70);
  pops.forEach((p, i) => {
    setTimeout(() => popScore(ui.pops, p.at, p.text), land ? land + i * beat : i * 240);
  });
  await flyTiles(wall, { layer: ui.fly, duration: 520, stagger: 70 });
  floorBills.forEach(([floorEl, bill]) => popScore(ui.pops, floorEl, bill, "loss"));
  if (lid.length) {
    lidEl.classList.add("receiving");
    await flyTiles(lid, { layer: ui.fly, duration: 420, stagger: 40 });
    setTimeout(() => lidEl.classList.remove("receiving"), 400);
  }
}

/** The tiles that just reached the wall glaze over as they land. */
function flashWall(report) {
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
  if (event.key !== "Escape") return;
  if (nav.browsing()) {
    nav.toLatest();
    return;
  }
  if (sel) {
    sel = null;
    suggestion = null;
    render();
  }
});
// ← / → / End walk the game, exactly as they do in a chess client
bindHistoryKeys(nav, { enabled: () => !busy });

// The tally is off until analytics.js names an endpoint. When it is on, the
// About panel must stop claiming a silence the page no longer keeps.
if (analyticsOn()) {
  ui.aboutMeta.before(
    node(
      "p",
      null,
      "One anonymous, cookie-free counter ping records that a game was played " +
        "and how it ended — nothing else, and nothing about you."
    )
  );
  track("pageview");
}

syncCoach();
bootEngine();
