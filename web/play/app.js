/* Azul play-vs-AI — the page's own glue.
 *
 * The reusable half of this app lives in ui/: the boards, the status band, the
 * log, the scoring panels and the tile flights are framework-free modules that
 * take state and give back elements (see ui/PORTING.md). This file is the part
 * that is specific to the local server: it talks to /api/*, owns the turn
 * sequence, and decides what the status band says.
 *
 * How a turn plays out — and why there is no overlay anywhere on this page:
 *
 *   1. you click a colour, then a row; the move is sent (with coach mode on,
 *      the server rates it first and the band shows the rating clock);
 *   2. the tiles fly, in a straight line, from the dish to your board, and the
 *      dish's leftovers fly to the middle;
 *   3. if that ended the round, full pattern lines fly to the wall and the
 *      floor line flies to the lid, and the round's scoring appears *inline*
 *      under the boards;
 *   4. the AI thinks against the clock in the status band, then *every* move it
 *      makes plays out on the right-hand board the same way — including the
 *      second of a double move across a round boundary, which starts from the
 *      refilled table the server reports as that move's `state_before`.
 *
 * Input is locked from the moment a move is sent until the last tile lands, and
 * for no longer than that.
 *
 * ← and → walk back through the game. That is pure replay of positions this
 * page has already been handed (see ui/history.js): no request, no re-search,
 * and the live game is not touched while you look.
 */

import { flyTiles, initSpeed, sleep } from "./ui/animate.js";
import { createBoard, createMiddle } from "./ui/board.js";
import { COLORS, FLOOR, node } from "./ui/dom.js";
import { bindHistoryKeys, createHistory } from "./ui/history.js";
import { renderLog } from "./ui/log.js";
import { clearScoring, renderFinalPanel, renderRoundPanel } from "./ui/scoring.js";
import { createSettings } from "./ui/settings.js";
import { createStatus } from "./ui/status.js";

const el = (id) => document.getElementById(id);
const ui = {
  matchup: el("matchup"), setup: el("setup"), opponent: el("opponent"),
  specField: el("spec-field"), spec: el("spec"), seed: el("seed"), first: el("first"),
  opponentNote: el("opponent-note"), thinkField: el("think-field"), think: el("think"),
  deal: el("deal"), middle: el("middle"), prompt: el("prompt"),
  hint: el("hint"), cancel: el("cancel"), fly: el("fly"), scoring: el("scoring"),
  log: el("log"), coach: el("coach"), coachField: el("coach-field"),
  coachLegend: el("coach-legend"), supply: el("supply"), toasts: el("toasts"),
  settings: el("settings"), nav: el("nav"),
};

let S = null;           // last server snapshot
let sel = null;         // {source, color} — tiles you are holding
let suggestion = null;  // {source, color, dest} from /api/hint
let busy = false;       // a request is in flight, or tiles are still moving
let bestInfo = null;    // /api/agents "best": which checkpoint is strongest
let thinkSeconds = 0;   // the AI's per-move budget
let coachOn = false;    // rate my moves with the AI's own search

initSpeed(); // before anything can animate: the stored speed, or 1×
const status = createStatus(el("status"));
const settings = createSettings(ui.settings);
const middle = createMiddle(ui.middle, { onPick: pick });
const boards = {
  human: createBoard(el("board-human"), { seat: 0, interactive: true, onPlay: play }),
  ai: createBoard(el("board-ai"), { seat: 1, interactive: false }),
};
const nav = createHistory(ui.nav, {
  log: () => (S && S.log) || [],
  enabled: () => !busy,
  onChange: () => render(),
});

const COACH_LEGEND =
  "Coach mode rates your move with the AI's own search: 0.00 = the move it would " +
  "have played, −1 ≈ a whole win thrown away. It thinks before your move lands, " +
  "so each turn takes a couple of seconds longer.";

/* ------------------------------------------------------------------ plumbing */
function toast(message, kind) {
  const t = node("div", "toast" + (kind ? " " + kind : ""), message);
  ui.toasts.appendChild(t);
  setTimeout(() => t.remove(), 6000);
}

async function api(path, options) {
  const res = await fetch(path, options);
  let data = null;
  try { data = await res.json(); } catch (err) { data = null; }
  if (!res.ok) {
    const error = new Error((data && data.error) || res.status + " " + res.statusText);
    error.status = res.status;
    throw error;
  }
  return data;
}

const postJSON = (path, body) => api(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

function setBusy(on) {
  const changed = busy !== on;
  busy = on;
  document.body.classList.toggle("locked", on);
  ui.deal.disabled = on;
  ui.hint.disabled = on || !S || !S.your_turn;
  nav.draw(); // browsing the history is off while tiles are in the air
  // the tiles you may pick up depend on this, and unlocking has to give them
  // back — the redraw is safe here because tiles only ever fly while busy
  if (changed && S) render();
}

/* ---------------------------------------------------------------- the boards */
function seatsOf(snapshot) {
  return snapshot ? { you: snapshot.human_seat, ai: snapshot.ai_seat } : { you: 0, ai: 1 };
}

function aiLabel() {
  return S && S.agent_name ? "AI · " + S.agent_name : "AI";
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
  ui.hint.disabled = busy || !live || !S.your_turn;
  syncCoachAvailability();
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

/** Fix the two boards' seats to this game's seating (you are always on the left). */
function seatBoards() {
  const seats = seatsOf(S);
  boards.human = createBoard(el("board-human"), {
    seat: seats.you, interactive: true, onPlay: play,
  });
  boards.ai = createBoard(el("board-ai"), { seat: seats.ai, interactive: false });
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
        ? "Your turn: pick a row for your " + COLORS[sel.color] + " tiles"
        : "Your turn: pick a colour",
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
function currentSpec() {
  const choice = ui.opponent.value;
  // "best" resolves server-side at deal time, so an overnight run's newest
  // strongest checkpoint is picked up without touching the page.
  if (choice === "best") return "best?sims=" + ((bestInfo && bestInfo.default_sims) || 400);
  if (choice !== "custom") return choice;
  return ui.spec.value.trim();
}

/** Seconds the AI may think per move — 0 for the baselines, which don't search. */
function currentThinkTime() {
  if (ui.opponent.value !== "best") return 0;
  const seconds = Number(ui.think.value);
  return Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
}

async function newGame(event) {
  if (event) event.preventDefault();
  const spec = currentSpec();
  if (!spec) { toast("Type an agent spec, or pick one from the list."); return; }
  const body = {
    opponent_spec: spec,
    human_plays_first: ui.first.checked,
    think_time_s: currentThinkTime(),
  };
  const seed = ui.seed.value.trim();
  if (seed) {
    if (!/^-?\d+$/.test(seed)) { toast("The seed must be a whole number."); return; }
    body.seed = Number(seed);
  }
  setBusy(true);
  try {
    clearScoring(ui.scoring);
    adopt(await postJSON("/api/new", body), { reseat: true });
    toast("New game dealt against " + S.agent_name + ".", "good");
  } catch (err) {
    toast(err.message);
  } finally {
    setBusy(false);
  }
}

function adopt(snapshot, options) {
  const first = !S || (options && options.reseat);
  S = snapshot;
  sel = null;
  suggestion = null;
  if (first) {
    seatBoards();
    nav.reset();
  }
  noteFrames(snapshot);
  render();
}

/**
 * File this payload's positions into the move history.
 *
 * Every move carries the position it was played from, and the position a move
 * was played from is the position *after* the move before it — so ply k's frame
 * is ply k+1's `state_before`, and the tail of the list is the snapshot itself.
 * Nothing is recomputed and nothing is fetched.
 */
function noteFrames(payload) {
  const moves = [];
  if (payload.human_move) moves.push(payload.human_move);
  (payload.ai_moves || []).forEach((m) => moves.push(m));
  (payload.last_ai_moves || []).forEach((m) => moves.push(m));
  moves.forEach((m) => {
    if (m && m.state_before && typeof m.ply === "number") nav.note(m.ply - 1, m.state_before);
  });
  nav.note(payload.ply, payload.state);
}

async function refresh() {
  try {
    adopt(await api("/api/state"), { reseat: true });
  } catch (err) {
    if (err.status === 409) { await newGame(); return; }
    toast(err.message);
  }
}

function pick(source, color) {
  if (busy || !S || !S.your_turn || nav.browsing()) return;
  sel = sel && sel.source === source && sel.color === color ? null : { source, color };
  suggestion = null;
  render();
}

/**
 * One turn: your move (rated first if coach mode is on), then the AI's reply.
 * The tiles are animated from the *current* DOM, before the new state is
 * adopted, so what you see leaving the dish is what actually left it.
 */
async function play(id) {
  if (busy || !S || !S.your_turn || nav.browsing()) return;
  sel = null;
  setBusy(true);
  clearScoring(ui.scoring);
  try {
    const request = postJSON("/api/act", {
      action_id: id, defer_ai: true, coach: coachOn && !!S.coach_available,
    });
    const mine = await awaitMove(request);
    await settle(mine, mine.human_move ? [mine.human_move] : [], boards.human, "human");
    if (!mine.ai_pending) return;
    await runAiTurn();
  } catch (err) {
    status.stopClock();
    toast(err.message);
    await refresh();
  } finally {
    setBusy(false);
    resumeIfPending(); // a failed /api/ai leaves the AI owing a move
  }
}

/** Wait for /api/act; if the coach is thinking, say so with the clock running. */
async function awaitMove(request) {
  let settled = false;
  const done = request.then(
    (value) => { settled = true; return value; },
    (err) => { settled = true; throw err; }
  );
  await Promise.race([done.catch(() => null), sleep(140)]);
  if (!settled) {
    status.set({ headline: "Rating your move", detail: "the AI is searching your position", tone: "ai", keepClock: true });
    status.startClock({
      budget: (S && S.coach_time_s) || 0,
      label: (spent, budget) =>
        budget
          ? "Rating your move: " + spent.toFixed(1) + "s of " + budget + "s"
          : "Rating your move: " + spent.toFixed(1) + "s",
    });
  }
  const value = await done;
  status.stopClock();
  return value;
}

/** If the snapshot says the AI still owes a move, let it move. */
async function resumeIfPending() {
  if (busy || !S || !S.ai_pending) return;
  setBusy(true);
  try {
    await runAiTurn();
  } catch (err) {
    status.stopClock();
    toast(err.message);
  } finally {
    setBusy(false);
  }
}

/** The AI's turn: the clock runs in the status band, then its move plays out. */
async function runAiTurn() {
  const pending = postJSON("/api/ai", {});
  const budget = (S && S.think_time_s) || thinkSeconds;
  status.set({ headline: "The AI is thinking", detail: turnDetail(), tone: "ai", keepClock: true });
  status.startClock({
    budget,
    label: (spent, cap) =>
      cap
        ? "AI is thinking: " + spent.toFixed(1) + "s of " + cap + "s"
        : "AI is thinking: " + spent.toFixed(1) + "s",
  });
  let theirs;
  try {
    theirs = await pending;
  } finally {
    status.stopClock();
  }
  await settle(theirs, theirs.ai_moves || [], boards.ai, "ai");
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
async function playMoves(moves, board, reports, mover) {
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
    await animateTake(move, board, middle);
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

/**
 * Play out `moves`, then adopt the snapshot they led to and show the scoring.
 */
async function settle(payload, moves, board, mover) {
  const reports = payload.round_reports || [];
  await playMoves(moves, board, reports, mover);
  adopt(payload);
  if (reports.length) {
    const last = reports[reports.length - 1];
    flashWall(last);
    if (!last.game_over) renderRoundPanel(ui.scoring, last, sides());
  }
  if (payload.state.is_terminal && payload.final) {
    renderFinalPanel(ui.scoring, payload.final, sides());
  } else if (mover === "ai" && moves.length) {
    // let the AI's own move stand as the headline for a beat before your turn
    const last = moves[moves.length - 1];
    status.set({ headline: "AI " + last.text, detail: turnDetail(), tone: "ai" });
    await sleep(500);
  }
  render();
}

function sides() {
  return [[S.human_seat, "You"], [S.ai_seat, aiLabel()]];
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
 */
async function animateTake(move, board, table) {
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
async function animateTiling(report) {
  if (!report) return;
  const lidEl = lidTarget();
  // one player is scored completely — wall, then lid — before the other
  // starts, the way a table is counted in real life
  for (const [seat, board] of [[S.human_seat, boards.human], [S.ai_seat, boards.ai]]) {
    const player = report.players[seat];
    if (!player) continue;
    const wall = [];
    player.tiles.forEach((t) => {
      const tiles = board.lineTiles(t.row);
      const from = tiles[tiles.length - 1] || board.lineRow(t.row);
      const to = board.wallCell(t.row, t.col);
      if (from && to) wall.push({ from, to, color: t.color, hide: false });
    });
    const lid = board.floorTiles().map((from) => ({
      from, to: lidEl, color: Number(from.dataset.color),
    }));
    await flyTiles(wall, { layer: ui.fly, duration: 520, stagger: 70 });
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

/* --------------------------------------------------------------------- hints */
async function askHint() {
  if (!S || !S.your_turn || busy || nav.browsing()) return;
  setBusy(true);
  try {
    const data = await api("/api/hint");
    suggestion = { source: data.move.source, color: data.move.color, dest: data.move.dest };
    sel = { source: data.move.source, color: data.move.color };
    render();
    toast(data.text, "good");
  } catch (err) {
    toast(err.message);
  } finally {
    setBusy(false);
  }
}

/* ---------------------------------------------------------------- coach mode */
function syncCoachAvailability() {
  const available = !!(S && S.coach_available);
  ui.coach.disabled = !available;
  if (!available && coachOn) coachOn = false;
  ui.coach.checked = coachOn;
  ui.coachField.classList.toggle("off", !coachOn);
  ui.coachField.title = available
    ? "Score your moves with the AI's own search"
    : "Coach mode needs a searching opponent, deal against a trained checkpoint";
  ui.coachLegend.textContent = COACH_LEGEND;
  ui.coachLegend.hidden = !coachOn;
}

/* -------------------------------------------------------------------- wiring */
function syncOpponentFields(focusSpec) {
  const choice = ui.opponent.value;
  ui.specField.classList.toggle("hidden", choice !== "custom");
  ui.thinkField.classList.toggle("hidden", choice !== "best");
  if (choice === "best" && bestInfo && bestInfo.available) {
    ui.opponentNote.textContent = bestInfo.detail;
  } else if (choice === "best" && bestInfo) {
    ui.opponentNote.textContent = "no rated checkpoint on disk yet";
  } else {
    ui.opponentNote.textContent = "";
  }
  if (focusSpec && choice === "custom") ui.spec.focus();
}

/** Fill the dropdown from the server, so the two lists cannot drift apart. */
async function loadAgentList() {
  const custom = ui.opponent.querySelector('option[value="custom"]');
  try {
    const data = await api("/api/agents");
    if (!data.baselines || !custom) return;
    bestInfo = data.best || null;
    ui.opponent.innerHTML = "";
    if (bestInfo && bestInfo.available) {
      const option = document.createElement("option");
      option.value = "best";
      option.textContent = bestInfo.label || "Strongest trained (auto)";
      ui.opponent.appendChild(option);
      const choices = bestInfo.think_choices || [];
      if (choices.length) {
        ui.think.innerHTML = "";
        choices.forEach((n) => {
          const opt = document.createElement("option");
          opt.value = String(n);
          opt.textContent = n > 0 ? n + " seconds" : "Instant reply";
          ui.think.appendChild(opt);
        });
      }
      thinkSeconds = bestInfo.default_think_s || 0;
      ui.think.value = String(thinkSeconds);
    }
    data.baselines.forEach((spec) => {
      const option = document.createElement("option");
      option.value = spec;
      option.textContent = spec.charAt(0).toUpperCase() + spec.slice(1);
      ui.opponent.appendChild(option);
    });
    ui.opponent.appendChild(custom);
    ui.opponent.value = data.default || data.fallback_default || data.baselines[0];
    if (data.custom_example) ui.spec.placeholder = data.custom_example;
  } catch (err) {
    // the markup already lists the baselines; nothing to do
  }
  syncOpponentFields(false);
}

ui.setup.addEventListener("submit", newGame);
ui.opponent.addEventListener("change", () => syncOpponentFields(true));
ui.think.addEventListener("change", () => { thinkSeconds = Number(ui.think.value) || 0; });
ui.hint.addEventListener("click", askHint);
ui.cancel.addEventListener("click", () => { sel = null; suggestion = null; render(); });
ui.coach.addEventListener("change", () => {
  coachOn = ui.coach.checked;
  syncCoachAvailability();
  if (coachOn) toast("Coach mode on: your moves are scored by the AI's own search.", "good");
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    if (nav.browsing()) { nav.toLatest(); return; }
    if (sel) { sel = null; suggestion = null; render(); }
  }
});
// ← / → / End walk the game, exactly as they do in a chess client
bindHistoryKeys(nav, { enabled: () => !busy });

status.set({ headline: "Dealing…", detail: "picking up the tiles", tone: "idle" });
syncCoachAvailability();
loadAgentList();
refresh().then(resumeIfPending);
