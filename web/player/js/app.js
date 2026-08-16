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
 *   2. you click a row and the held tiles continue from the hand to your board.
 *      With confirm mode on (the default) the page simply draws the position the
 *      move would leave — real tiles in their real places, the new ones glowing —
 *      and a banner under the middle asks for the word: "Play this move" commits
 *      it, Cancel goes all the way back to before the pick. The page is never
 *      *between* positions: every view is a fully drawn state, and the flights
 *      are transient garnish over it (a lesson learned from floating clones that
 *      scrolled apart from the table). Nothing ever waits for the coach — with
 *      coach mode on, the rating runs in the background *after* the turn and
 *      fills in on the move's log entry;
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
import { confirmOn } from "../ui/confirm.js";
import {
  CENTER,
  COLORS,
  CUM_PENALTY,
  FLOOR,
  markerChip,
  node,
  poolCount,
  tileEl,
} from "../ui/dom.js";
import { bindHistoryKeys, createHistory } from "../ui/history.js";
import { renderLog } from "../ui/log.js";
import { popScore } from "../ui/popups.js";
import { clearScoring, renderFinalPanel, renderRoundPanel } from "../ui/scoring.js";
import { createSettings } from "../ui/settings.js";
import { createStatus } from "../ui/status.js";
import { analyticsOn, statsUrl, track } from "./analytics.js";
import { GameSession } from "./game.js";
import { BACKENDS } from "./net.js";
import { describeAction } from "./report.js";

const el = (id) => document.getElementById(id);
const ui = {
  matchup: el("matchup"), setup: el("setup"), seed: el("seed"), first: el("first"),
  think: el("think"), deal: el("deal"), engineBar: el("engine-bar"),
  engineText: el("engine-text"), aboutMeta: el("about-meta"), middle: el("middle"),
  prompt: el("prompt"), hint: el("hint"), cancel: el("cancel"), fly: el("fly"),
  scoring: el("scoring"), log: el("log"), coach: el("coach"),
  coachField: el("coach-field"), coachLegend: el("coach-legend"), counts: el("counts"),
  settings: el("settings"), nav: el("nav"),
  hand: el("hand"), handTiles: el("hand-tiles"), pops: el("pops"),
  confirm: el("confirm"), confirmBar: el("confirm-bar"),
  confirmDetail: el("confirm-detail"), confirmCancel: el("confirm-cancel"),
};

let session = null;    // the GameSession that owns the rules
let S = null;          // its last snapshot, what the page draws from
let sel = null;        // {source, color} — tiles you are holding
let suggestion = null; // {source, color, dest} from the policy head
let busy = false;      // a move is being computed, or is still playing out
let meta = null;       // model/model_meta.json
let backend = null;    // {name, batch, margin, rate} — what the worker settled on
let engineReady = false;
let liveSims = 0;      // positions the current search has visited
let coachOn = false;   // rate my moves with the AI's own search
let notice = "";       // a passing message, shown in the band, never over the board
let noticeTimer = null;
let proposal = null;   // {id, move} — a placed move waiting for "Play this move"
let committing = null; // the move whose held view stays up while its tiles fly
let analysis = null;   // the coach reading the position while you think (see below)
let navToken = 0;      // invalidates a history-step animation when another step lands

initSpeed(); // before anything can animate: the stored speed, or 1×
const status = createStatus(el("status"));
const settings = createSettings(ui.settings, { popups: true, confirm: true, boards: true });
const middle = createMiddle(ui.middle, { onPick: pick });
const boards = {
  human: createBoard(el("board-human"), { seat: 0, interactive: true, onPlay: route }),
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
  "have played, −1 ≈ a whole win thrown away. The coach reads the position while " +
  "you think, so the verdict usually lands with your move.";

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
  let ready = null;
  try {
    ready = await ask(
      {
        type: "init",
        // In preference order. The worker takes the first one that actually
        // works in this browser and tells us which that was; a browser with no
        // WebGPU never downloads the WebGPU runtime at all.
        backends: ["webgpu", "wasm"].map((name) => ({
          name,
          ep: BACKENDS[name].ep,
          module: new URL(BACKENDS[name].module, import.meta.url).href,
          wasm: new URL(BACKENDS[name].wasm, import.meta.url).href,
        })),
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
  backend = { name: ready.backend, batch: ready.batch, margin: !!ready.margin, rate: null };
  ui.engineBar.classList.add("ready");
  ui.engineText.textContent = engineLine();
  describeModel();
  ui.deal.disabled = false;
  newGame();
}

/** Where the net is running, in the words a player would use. */
function backendLabel() {
  if (!backend) return "your CPU";
  return backend.name === "webgpu" ? "your GPU (WebGPU)" : "your CPU (WebAssembly)";
}

function engineLine() {
  const where = `searching on ${backendLabel()}`;
  if (!meta) return `Net ready — ${where}.`;
  const elo = typeof meta.elo === "number" ? `${meta.elo >= 0 ? "+" : ""}${Math.round(meta.elo)} Elo` : "unrated";
  const params = meta.num_params ? `${(meta.num_params / 1e6).toFixed(1)}M parameters` : "";
  const rate =
    backend && backend.rate ? ` · ${Math.round(backend.rate).toLocaleString()} positions/s` : "";
  return `${meta.run}/${meta.checkpoint} · ${elo} on our internal ladder · ${params} · ${where}${rate}`;
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
    // What the search is actually running on, and how hard: this is the number
    // the thinking-time selector is really spending, so it belongs on the page
    // rather than in a console log.
    backend
      ? `${backend.name === "webgpu" ? "WebGPU" : "WebAssembly"}, ${backend.batch} positions per pass`
      : null,
    backend && backend.rate ? `${Math.round(backend.rate).toLocaleString()} positions/s here` : null,
    backend && backend.margin ? "margin head: decisive play" : null,
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
    seat: S.human_seat, interactive: true, onPlay: route,
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
/**
 * Draw the page. The page is *always* a fully drawn position — never a limbo of
 * floating clones (those scrolled apart from the table and collided with real
 * tiles; see the hand tray and confirm mode below). Three positions can be on
 * show: the live game; the live game with the selection lifted into the hand
 * tray (`heldView`); or, in confirm mode, the position the placed move would
 * leave (`placedView`), its new tiles glowing until you validate or cancel.
 * Browsing the history is read-only by construction, as before.
 */
function render() {
  if (!S) return;
  const frame = nav.frame();
  const live = !frame;
  if (!live && proposal) proposal = null; // browsing abandons a placed move
  const placed = live && proposal ? proposal.move : null;
  const heldInfo = placed || !live ? null : sel ? pickInfo(sel.source, sel.color) : committing;
  const base = live ? S.state : frame.state;
  const st = placed ? placedView(placed) : base;
  const mid = placed ? st : heldInfo ? heldView(heldInfo) : base;
  renderStatus(frame);
  syncBanner();
  middle.render({
    state: mid,
    legalActions: live && !placed ? S.human_legal_actions : [],
    canPick: live && S.your_turn && !base.is_terminal && !busy && !placed,
    selection: null, // the hand tray *is* the selection; dimming forty tiles says less
  });
  renderHand(heldInfo);
  boards.human.render({
    state: st,
    legalActions: live && !placed ? S.human_legal_actions : [],
    selection: live && !placed ? sel : null,
    title: "You",
    toMove: live && !base.is_terminal && base.current_player === S.human_seat,
    highlightRow: live && suggestion ? suggestion.dest : undefined,
  });
  if (placed) glowPlacement(placed);
  boards.ai.render({
    state: st,
    title: aiLabel(),
    toMove: live && !base.is_terminal && base.current_player === S.ai_seat,
  });
  renderCounts(st);
  renderLog(ui.log, (live ? S.log : frame.log) || []);
  ui.cancel.classList.toggle("hidden", !live || !sel || !!placed);
  ui.hint.disabled = busy || !engineReady || !live || !S.your_turn || !!placed;
  syncCoach();
  if (live && !placed && suggestion) {
    const dish = middle.sourceEl(suggestion.source);
    if (dish) {
      dish.classList.add("picked");
      setTimeout(() => dish.classList.remove("picked"), 1600);
    }
  }
  startAnalysis(); // with coach mode on and your turn, the coach starts reading
}

/** Draw one position and nothing else — a history frame, or a step in a turn. */
function drawPosition(st) {
  if (!st) return;
  middle.render({ state: st, legalActions: [], canPick: false, selection: null });
  boards.human.render({ state: st, title: "You", toMove: false });
  boards.ai.render({ state: st, title: aiLabel(), toMove: false });
  renderCounts(st);
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
    if (proposal) {
      status.set({
        headline: "Your move is placed",
        detail: "This is the position it would leave — validate or cancel below.",
        tone: "you",
      });
      ui.prompt.textContent = "";
      return;
    }
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

/* One quiet line of totals — never per-colour counts, Rémi retired those. The
 * lid keeps a real element (#lid-row) because discarded tiles fly to it. */
function renderCounts(st) {
  const sum = (counts) => counts.reduce((a, b) => a + b, 0);
  ui.counts.innerHTML = "";
  ui.counts.appendChild(node("span", "count", "Bag " + sum(st.bag)));
  const lid = node("span", "count", "Lid " + sum(st.lid));
  lid.id = "lid-row";
  ui.counts.appendChild(lid);
  ui.counts.appendChild(node("span", "count", st.tiles_left + " on the table"));
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
  // The worker measures its own rate; the first real search is the first honest
  // positions/s this machine has produced, so the page stops guessing at it.
  if (backend && reply.search && reply.search.rate && !backend.rate) {
    backend.rate = reply.search.rate;
    ui.engineText.textContent = engineLine();
    describeModel();
  }
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
  proposal = null;
  committing = null; // the real position takes over from any derived view
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
  if (busy || !S || !S.your_turn || nav.browsing() || proposal) return;
  suggestion = null;
  if (sel && sel.source === source && sel.color === color) {
    sel = null; // putting tiles back is instant and unambiguous
    render();
    return;
  }
  // capture the outgoing dish before the held view replaces it
  const takenRects = middle.sourceTiles(source, color).map((t) => t.getBoundingClientRect());
  const chip =
    source === CENTER && S.state.marker_in_center
      ? middle.centerEl().querySelector(".marker")
      : null;
  const markerRect = chip ? chip.getBoundingClientRect() : null;
  const restRects = {};
  if (source !== CENTER) {
    middle.remainderTiles(source, color).forEach((t) => {
      (restRects[t.dataset.color] = restRects[t.dataset.color] || []).push(
        t.getBoundingClientRect()
      );
    });
  }
  sel = { source, color };
  flyInto(() => {
    const plan = [];
    const handTiles = [].slice.call(ui.handTiles.querySelectorAll(".tile"));
    takenRects.forEach((r, i) => plan.push([r, handTiles[i], color]));
    if (markerRect) plan.push([markerRect, ui.handTiles.querySelector(".marker"), "marker"]);
    Object.keys(restRects).forEach((c) => {
      const now = middle.centerEl().querySelectorAll('.tile[data-color="' + c + '"]');
      const olds = restRects[c];
      olds.forEach((r, i) => plan.push([r, now[now.length - olds.length + i], Number(c)]));
    });
    return plan;
  });
}

/* ---------------------------------------------------------- derived positions */
/** What a pick moves, before it has a destination. */
function pickInfo(source, color) {
  return {
    source,
    color,
    count: poolCount(S.state, source, color),
    took_marker: source === CENTER && S.state.marker_in_center,
  };
}

/** The live position with the selection lifted into the hand: the source pool
 * emptied, a factory's leftovers pushed to the middle, a taken marker in hand. */
function heldView(info) {
  const st = structuredClone(S.state);
  if (info.source === CENTER) {
    st.center[info.color] = 0;
    if (info.took_marker) st.marker_in_center = false;
  } else {
    const dish = st.factories[info.source];
    dish[info.color] = 0;
    for (let c = 0; c < COLORS.length; c++) {
      st.center[c] += dish[c];
      dish[c] = 0;
    }
  }
  return st;
}

/** The position `move` would leave — what confirm mode draws while it waits. */
function placedView(move) {
  const st = heldView(move);
  const me = st.players[S.human_seat];
  if (move.dest !== FLOOR && move.placed) {
    const line = me.pattern_lines[move.dest];
    line.color = move.color;
    line.count += move.placed;
  }
  if (move.to_floor) me.floor[move.color] += move.to_floor;
  if (move.to_lid) st.lid[move.color] += move.to_lid;
  if (move.took_marker) me.floor_marker = true;
  const occupied =
    me.floor.reduce((a, b) => a + b, 0) + (me.floor_marker ? 1 : 0);
  me.floor_penalty = CUM_PENALTY[Math.min(7, occupied)];
  st.tiles_left -= move.count;
  return st;
}

/** The hand tray: the selection as real tiles, in the page, scrolling with it. */
function renderHand(info) {
  ui.handTiles.innerHTML = "";
  if (!info || !info.count) {
    ui.hand.hidden = true;
    return;
  }
  for (let k = 0; k < info.count; k++) ui.handTiles.appendChild(tileEl(info.color));
  if (info.took_marker) ui.handTiles.appendChild(markerChip(true));
  ui.hand.hidden = false;
}

/** A soft light around the tiles a placed move just put down. */
function glowPlacement(move) {
  placedTiles(move).forEach((t) => t.classList.add("proposed"));
}

/** The placed move's own tiles on the freshly drawn board, in take order. */
function placedTiles(move) {
  const board = boards.human;
  const out = [];
  if (move.dest !== FLOOR && move.placed) {
    out.push.apply(out, board.lineTiles(move.dest).slice(-move.placed));
  }
  if (move.to_floor) {
    const mine = board.floorTiles().filter((t) => Number(t.dataset.color) === move.color);
    out.push.apply(out, mine.slice(-move.to_floor));
  }
  if (move.took_marker) {
    const floor = board.floorEl();
    const chip = floor && floor.querySelector(".marker");
    if (chip) out.push(chip);
  }
  return out;
}

/**
 * Redraw, then fly clones from rectangles captured off the outgoing page to the
 * elements `find()` names on the new one. The page is never *between* positions:
 * the new position is drawn at once, and the pieces that moved are briefly
 * covered while their clones travel. With animation off, the redraw is all.
 */
async function flyInto(find) {
  render();
  if (!animated()) return;
  const plan = find() || [];
  const flights = [];
  const covered = [];
  plan.forEach(([rect, target, color]) => {
    if (!rect || !target) return;
    if (target.style) {
      target.style.visibility = "hidden";
      covered.push(target);
    }
    flights.push({ from: rect, to: target, color, hide: false });
  });
  await flyTiles(flights, { layer: ui.fly });
  covered.forEach((elm) => {
    elm.style.visibility = "";
  });
}

/* ------------------------------------------------------------- confirm mode */
/**
 * A row was clicked. With confirm mode off this *is* the move; with it on (the
 * default) the page draws the position the move would leave — real tiles in
 * their real places, the new ones glowing — and the banner under the middle
 * asks for the word. Cancel goes all the way back to before the pick.
 */
function route(id) {
  if (busy || !session || !S || !S.your_turn || nav.browsing() || proposal) return;
  if (!confirmOn()) {
    play(id);
    return;
  }
  propose(id);
}

/** The banner a placed move answers to. It swaps in for the action row's usual
 * controls (see board.css), so showing it never moves the page. */
function syncBanner() {
  const on = !!proposal && !busy;
  ui.confirmBar.hidden = !on;
  ui.confirmBar.parentElement.classList.toggle("placing", on);
  if (on) {
    ui.confirmDetail.textContent =
      "You " + proposal.move.text + " — nothing is final until you play it.";
  }
}

/** Place the move: draw its position, glow its tiles, and wait for the word. */
function propose(id) {
  const move = describeAction(session.state, id);
  // take-off points: the hand tray the selection is sitting in
  const rects = [].slice
    .call(ui.handTiles.querySelectorAll(".tile"))
    .map((t) => t.getBoundingClientRect());
  const chip = ui.handTiles.querySelector(".marker");
  const markerRect = chip ? chip.getBoundingClientRect() : null;
  proposal = { id, move };
  sel = null; // a placed move is validated or cancelled whole
  flyInto(() => {
    const landed = placedTiles(move);
    const lidRect =
      move.to_lid > 0 ? lidTarget().getBoundingClientRect() : null;
    const plan = [];
    rects.forEach((r, i) => {
      // in take order: the row, then the floor; overflow melts into the lid
      const target = landed[i] || lidRect;
      plan.push([r, target, move.color]);
    });
    if (markerRect) {
      const floor = boards.human.floorEl();
      plan.push([markerRect, floor && floor.querySelector(".marker"), "marker"]);
    }
    return plan;
  });
}

/** The banner's Cancel: all the way back, as if nothing had been picked. */
function cancelMove() {
  proposal = null;
  sel = null;
  suggestion = null;
  render();
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
  const move = describeAction(session.state, id);
  const committed = !!(proposal && proposal.id === id); // a validated placement
  // the direct path takes off from the hand the selection is sitting in
  const fromHand = !committed && !ui.hand.hidden;
  const handRects = fromHand
    ? [].slice.call(ui.handTiles.querySelectorAll(".tile")).map((t) => t.getBoundingClientRect())
    : null;
  const handChip = fromHand ? ui.handTiles.querySelector(".marker") : null;
  const markerRect = handChip ? handChip.getBoundingClientRect() : null;
  if (!committed) committing = move; // keeps the held view up while the tiles fly
  sel = null;
  suggestion = null;
  setBusy(true);
  clearScoring(ui.scoring);
  try {
    const setupBefore = coachOn ? session.state.toSetup() : null;
    // the coach's head start covers exactly this position: stop it where it
    // stands (the opponent needs the worker) and grade from what it read.
    // Cancelled even if coach mode was switched off meanwhile — a stale
    // analysis must never keep the opponent waiting.
    const reading =
      analysis && analysis.forSession === session && analysis.ply === S.ply
        ? analysis
        : null;
    if (reading) worker.postMessage({ type: "cancel" });
    const headStart = coachOn ? reading : null;
    const { move: applied, reports } = session.playHuman(id);
    const entry = session.log[applied.log_n];
    if (setupBefore && entry) {
      if (headStart) {
        entry.coach = { pending: true };
        finishFromAnalysis(headStart, id, entry, setupBefore);
      } else {
        queueRating(setupBefore, id, entry);
      }
    }
    const takeoff = committed
      ? { skip: true }
      : handRects
        ? { fromRects: handRects, markerRect }
        : null;
    await settle([applied], boards.human, reports, "human", takeoff);
    if (session.aiTurn) await runAiTurn();
  } catch (err) {
    status.stopClock();
    say(err.message);
    adopt();
  } finally {
    committing = null;
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
async function playMoves(moves, board, reports, mover, takeoff) {
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
    await animateTake(move, board, middle, i === 0 ? takeoff : null);
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
async function settle(moves, board, reports, mover, takeoff) {
  await playMoves(moves, board, reports, mover, takeoff);
  adopt({ moves });
  if (reports.length) {
    const last = reports[reports.length - 1];
    flashWall(last);
    if (!last.game_over) renderRoundPanel(ui.scoring, last, sides());
  }
  if (S.state.is_terminal && S.final) {
    renderFinalPanel(ui.scoring, S.final, sides());
    // one coarse path per net and outcome (so the dashboard can show win
    // rates per model), with the score line as the hit's detail
    const model = meta ? meta.run + "-" + meta.checkpoint : "unknown";
    const result =
      S.final.winner_side === "human"
        ? "human-wins"
        : S.final.winner_side === "ai"
          ? "net-wins"
          : "draw";
    track("game-end/" + model + "/" + result, {
      title:
        S.state.scores[S.human_seat] + "–" + S.state.scores[S.ai_seat] +
        " in " + S.final.rounds_played + " rounds",
    });
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

/* The coach's head start. The rating search runs on the position *before* your
 * move and only reads the played move out of the finished tree afterwards — so
 * it can run while you are still choosing. `startAnalysis` kicks it off the
 * moment it is your turn (idempotent per position); when you move, the search
 * is cancelled where it stands, the opponent gets the worker back, and the
 * verdict is read from whatever tree had grown. A tree too thin to be honest
 * (you moved within a fraction of a second) falls back to the old post-move
 * rating, so the verdict is never worse than it used to be. */
const ANALYSIS_MIN_SIMS = 400;

function startAnalysis() {
  if (!coachOn || !engineReady || busy || !session || !S || !S.your_turn) return;
  if (analysis && analysis.forSession === session && analysis.ply === S.ply) return;
  const job = { forSession: session, ply: S.ply, budgetS: coachBudget() };
  job.promise = ask({ type: "analyze", setup: session.state.toSetup(), budgetS: job.budgetS })
    .then((reply) => reply.analysis)
    .catch(() => null);
  analysis = job;
}

/** The verdict `rate` would have given, read out of an analyzed tree. */
function verdictFrom(a, budgetS, actionId) {
  const base = { budgetS, legal: a.legal, sims: a.sims, elapsedS: a.elapsedS };
  if (a.forced) return { ...base, delta: 0, forced: true };
  const explored = a.children || [];
  if (!explored.length) {
    return { ...base, unrated: true, reason: "the search had no time to explore this position" };
  }
  const best = explored.reduce((x, y) => (y.q > x.q ? y : x));
  const mine = explored.find((c) => c.action === actionId);
  if (!mine) {
    return { ...base, unrated: true, reason: "the search never explored this move" };
  }
  return {
    ...base,
    delta: Math.min(0, mine.q - best.q),
    your_q: mine.q,
    best_q: best.q,
    visits: mine.visits,
    best_visits: best.visits,
    best_text: a.best_text,
    explored: explored.length,
  };
}

/** Settle a move's verdict from the head-start analysis — or fall back. */
async function finishFromAnalysis(job, actionId, entry, setup) {
  const a = await job.promise;
  if (!a || (!a.forced && a.sims < ANALYSIS_MIN_SIMS)) {
    // the move came faster than the coach could read: judge it the old way
    queueRating(setup, actionId, entry);
    flushRatings();
    return;
  }
  entry.coach = verdictFrom(a, job.budgetS, actionId);
  if (job.forSession === session && S && !busy) render();
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
  if (!session || !S || !S.your_turn || busy || nav.browsing() || proposal) return;
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
  return document.getElementById("lid-row") || ui.counts;
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
 * `takeoff` says where this story already got to. `{skip: true}` is a validated
 * placement — the board has shown the final position since the tiles were
 * placed, so nothing moves. `{fromRects, markerRect}` is a selection sitting in
 * the hand tray — the tiles continue from there, and the leftovers, already
 * drawn in the middle by the held view, stay put. `null` is the AI (or a bare
 * page): everything flies from the dish, as it always has.
 */
async function animateTake(move, board, table, takeoff) {
  if (takeoff && takeoff.skip) {
    // a validated placement: the board has been showing exactly this position
    // since the tiles were placed, so there is nothing left to move
    popFloorCost(move, board);
    return;
  }
  const dish = table.sourceEl(move.source);
  const row = board.lineRow(move.dest);
  if (dish) dish.classList.add("picked");
  if (row) row.classList.add("incoming");

  const targets = travelTargets(board, move);
  let flights;
  if (takeoff && takeoff.fromRects) {
    // the selection was sitting in the hand tray; it continues from there, and
    // the tray empties the moment the tiles push off
    ui.hand.hidden = true;
    ui.handTiles.innerHTML = "";
    flights = takeoff.fromRects.map((r, i) => ({ from: r, to: targets[i], color: move.color }));
    if (move.took_marker && takeoff.markerRect) {
      const slot = board.floorSlots()[0];
      if (slot) flights.push({ from: takeoff.markerRect, to: slot, color: "marker" });
    }
  } else {
    // the AI's move (or a hintless page state): straight from the dish
    const taken = table.sourceTiles(move.source, move.color, move.count);
    flights = taken.map((from, i) => ({ from, to: targets[i], color: move.color }));
    const centreEl = table.centerEl();
    table.remainderTiles(move.source, move.color).forEach((from) => {
      flights.push({ from, to: centreEl, color: Number(from.dataset.color) });
    });
    if (move.took_marker) {
      const chip = centreEl.querySelector(".marker");
      const slot = board.floorSlots()[0];
      if (chip && slot) flights.push({ from: chip, to: slot, color: "marker" });
    }
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
 * each wall tile popping its points as it lands, each floor line its bill.
 * One player is scored completely — wall, floor bill, lid — before the other
 * starts, the way a table is counted in real life, so you can follow along. */
async function animateTiling(report) {
  if (!report) return;
  const lidEl = lidTarget();
  // with the tiles not animating, the pops still tell the arithmetic one step
  // at a time — this offset keeps the second player's count after the first's
  let beatOffset = 0;
  for (const [seat, board] of [[S.human_seat, boards.human], [S.ai_seat, boards.ai]]) {
    const player = report.players[seat];
    if (!player) continue;
    const wall = [];
    const pops = [];
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
    const lid = board.floorTiles().map((from) => ({
      from, to: lidEl, color: Number(from.dataset.color),
    }));
    // each "+N" appears as its tile touches down — or on a steady beat when
    // the tiles are not animating
    const land = scaled(520);
    const beat = scaled(70);
    pops.forEach((p, i) => {
      setTimeout(
        () => popScore(ui.pops, p.at, p.text),
        land ? land + i * beat : beatOffset + i * 240
      );
    });
    beatOffset += pops.length * 240;
    await flyTiles(wall, { layer: ui.fly, duration: 520, stagger: 70 });
    if (player.floor.penalty) {
      popScore(ui.pops, board.floorEl(), String(player.floor.penalty).replace("-", "−"), "loss");
    }
    if (lid.length) {
      lidEl.classList.add("receiving");
      await flyTiles(lid, { layer: ui.fly, duration: 420, stagger: 40 });
      setTimeout(() => lidEl.classList.remove("receiving"), 400);
    }
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
ui.confirm.addEventListener("click", () => {
  if (!proposal || busy) return;
  play(proposal.id); // play() reads the proposal itself; it stays up until the
  // real position takes over, so the board never flashes back
});
ui.confirmCancel.addEventListener("click", cancelMove);
ui.cancel.addEventListener("click", () => {
  sel = null;
  suggestion = null;
  render();
});
ui.coach.addEventListener("change", () => {
  coachOn = ui.coach.checked;
  syncCoach();
  startAnalysis(); // switched on mid-turn: start reading this position now
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
  if (proposal && !busy) {
    cancelMove(); // all the way back, same as the banner's Cancel
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

// The corner buttons: "?" walks you to the What-this-is panel; the bars open
// the public tally. If the tally is ever switched off (COUNT_URL emptied), the
// page stops pointing at it — it never claims a transparency it does not keep.
el("corner-about").addEventListener("click", () => {
  const about = el("about");
  about.scrollIntoView({ behavior: "smooth", block: "start" });
  about.classList.add("lit");
  setTimeout(() => about.classList.remove("lit"), 1600);
});
if (analyticsOn()) {
  el("corner-stats").href = statsUrl();
  track("pageview");
} else {
  el("corner-stats").remove();
  const note = el("tally-note");
  if (note) note.remove();
}

syncCoach();
bootEngine();
