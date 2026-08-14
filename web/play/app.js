/* Azul play-vs-AI — vanilla JS, no external resources.
 *
 * The server owns the rules; this file owns the table. It keeps one snapshot of
 * the game (`S`, straight from /api/state) plus one piece of local UI state
 * (`sel`, the colour the player has picked up), and redraws from those. Legality
 * always comes from the server's action-id list: a destination lights up because
 * `source*30 + colour*6 + dest` is in it, never because we re-derived the rules.
 */
"use strict";

const COLORS = ["blue", "yellow", "red", "black", "teal"];
const FLOOR_PENALTIES = [-1, -1, -2, -2, -2, -3, -3];
const CUM_PENALTY = [0, -1, -2, -4, -6, -8, -11, -14];
const CENTER = 5;
const FLOOR = 5;
const FLASH_MS = 1400;

const el = (id) => document.getElementById(id);
const ui = {
  matchup: el("matchup"), setup: el("setup"), opponent: el("opponent"),
  specField: el("spec-field"), spec: el("spec"), seed: el("seed"), first: el("first"),
  deal: el("deal"), factories: el("factories"), center: el("center"), turn: el("turn"),
  scoreHuman: el("score-human"), scoreAi: el("score-ai"), boardAi: el("board-ai"),
  boardHuman: el("board-human"), lastMove: el("last-move"), log: el("log"),
  supply: el("supply"), prompt: el("prompt"), hint: el("hint"), cancel: el("cancel"),
  overlay: el("overlay"), overlayTitle: el("overlay-title"), overlayBody: el("overlay-body"),
  overlayOk: el("overlay-ok"), toasts: el("toasts"),
};

let S = null;          // last server snapshot
let sel = null;        // {source, color} — tiles the player is holding
let suggestion = null; // {source, color, dest} from /api/hint
let busy = false;      // a request is in flight
let sheets = [];       // queued round-end / game-end overlays

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

async function api(path, options) {
  const res = await fetch(path, options);
  let data = null;
  try { data = await res.json(); } catch (err) { data = null; }
  if (!res.ok) {
    const msg = (data && data.error) || res.status + " " + res.statusText;
    const error = new Error(msg);
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
  busy = on;
  ui.deal.disabled = on;
  ui.hint.disabled = on || !S || !S.your_turn;
}

/* -------------------------------------------------------------- game control */
function currentSpec() {
  const choice = ui.opponent.value;
  if (choice !== "custom") return choice;
  return ui.spec.value.trim();
}

async function newGame(event) {
  if (event) event.preventDefault();
  const spec = currentSpec();
  if (!spec) { toast("Type an agent spec, or pick one from the list."); return; }
  const body = { opponent_spec: spec, human_plays_first: ui.first.checked };
  const seed = ui.seed.value.trim();
  if (seed) {
    if (!/^-?\d+$/.test(seed)) { toast("The seed must be a whole number."); return; }
    body.seed = Number(seed);
  }
  setBusy(true);
  try {
    adopt(await postJSON("/api/new", body));
    toast("New game dealt against " + S.agent_name + ".", "good");
  } catch (err) {
    toast(err.message);
  } finally {
    setBusy(false);
  }
}

async function play(actionId) {
  if (busy) return;
  setBusy(true);
  try {
    const data = await postJSON("/api/act", { action_id: actionId });
    adopt(data);
    (data.round_reports || []).forEach(queueRoundSheet);
    if (data.state.is_terminal && data.final) queueFinalSheet(data.final);
    showNextSheet();
    flashAiMoves(data.ai_moves || []);
  } catch (err) {
    toast(err.message);
    refresh();
  } finally {
    setBusy(false);
  }
}

async function askHint() {
  if (!S || !S.your_turn || busy) return;
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

async function refresh() {
  try {
    adopt(await api("/api/state"));
  } catch (err) {
    if (err.status === 409) { await newGame(); return; }
    toast(err.message);
  }
}

function adopt(snapshot) {
  S = snapshot;
  sel = null;
  suggestion = null;
  render();
}

/* -------------------------------------------------------------- legality map */
function legalSet() {
  return new Set((S && S.human_legal_actions) || []);
}

function actionId(source, color, dest) { return source * 30 + color * 6 + dest; }

function destsFor(source, color) {
  const legal = legalSet();
  const out = [];
  for (let d = 0; d <= FLOOR; d++) if (legal.has(actionId(source, color, d))) out.push(d);
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
  return { count, placed, overflow, takesMarker, penalty: after - before,
           completes: dest !== FLOOR && placed + me.pattern_lines[dest].count === me.pattern_lines[dest].capacity };
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
      sel = (sel && sel.source === source && sel.color === color) ? null : { source, color };
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

/* ------------------------------------------------------------- middle of the table */
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
      counts.forEach((n, c) => { for (let k = 0; k < n; k++) dish.appendChild(tileButton(i, c)); });
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
function renderBoard(host, seat, interactive) {
  const st = S.state;
  const me = st.players[seat];
  host.innerHTML = "";
  host.dataset.seat = seat;

  const main = node("div", "board-main");
  const lines = node("div", "lines");
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
        row.addEventListener("click", () => play(actionId(sel.source, sel.color, r)));
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

    for (let k = 0; k < line.capacity - line.count; k++) row.appendChild(node("div", "slot"));
    for (let k = 0; k < line.count; k++) row.appendChild(tile(line.color));
    if (suggestion && interactive && suggestion.dest === r) row.classList.add("flash");
    lines.appendChild(row);
  }
  main.appendChild(lines);

  const wall = node("div", "wall");
  for (let r = 0; r < 5; r++) {
    for (let col = 0; col < 5; col++) {
      const colour = (col - r + 5) % 5;
      if (me.wall[r][col]) {
        const t = tile(colour, "placed");
        t.dataset.row = r;
        t.dataset.col = col;
        t.title = COLORS[colour] + " — row " + (r + 1) + ", column " + (col + 1);
        wall.appendChild(t);
      } else {
        const cell = node("div", "cell bisque");
        cell.dataset.color = colour;
        cell.title = COLORS[colour] + " goes here";
        wall.appendChild(cell);
      }
    }
  }
  main.appendChild(wall);
  host.appendChild(main);

  const wrap = node("div", "floor-wrap");
  const floorOpen = interactive && sel && open.indexOf(FLOOR) !== -1;
  const floor = node(floorOpen ? "button" : "div", "floor");
  if (floorOpen) {
    floor.type = "button";
    floor.classList.add("open");
    floor.addEventListener("click", () => play(actionId(sel.source, sel.color, FLOOR)));
    const p = preview(FLOOR);
    floor.title = "Dump all " + p.count + " tiles here (" + p.penalty + " this round)";
  }
  floor.dataset.row = FLOOR;
  const occupants = [];
  if (me.floor_marker) occupants.push("marker");
  me.floor.forEach((n, c) => { for (let k = 0; k < n; k++) occupants.push(c); });
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
  const note = node("span", "floor-note" + (penalty ? " warn" : ""),
    penalty ? "Floor line: " + penalty + " at the end of this round"
            : "Floor line: clean");
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
  const seats = [[ui.scoreHuman, S.human_seat, "You"], [ui.scoreAi, S.ai_seat, "AI · " + S.agent_name]];
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
  if (S.state.is_terminal && S.final) ui.lastMove.textContent = S.final.headline;
  else if (last) ui.lastMove.textContent = "AI " + last.text + ".";
  else ui.lastMove.textContent = "You open. Pick a colour from a factory or the middle.";

  ui.log.innerHTML = "";
  (S.log || []).forEach((entry) => {
    const li = node("li", entry.kind === "move" ? entry.side : entry.kind, entry.text);
    ui.log.appendChild(li);
  });
  ui.log.scrollTop = ui.log.scrollHeight;
}

function renderStatus() {
  const st = S.state;
  ui.matchup.textContent = "you versus " + S.agent_name + " · seed " + S.seed;
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
    ui.turn.textContent = "The AI is taking its turn…";
    ui.prompt.textContent = "";
  }
  ui.cancel.classList.toggle("hidden", !sel);
  ui.hint.disabled = busy || !S.your_turn;
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
  const buttons = ui.factories.querySelectorAll("button.tile");
  const all = [].concat([].slice.call(buttons), [].slice.call(ui.center.querySelectorAll("button.tile")));
  all.forEach((b) => {
    if (!sel) return;
    if (b.dataset.key === sel.source + ":" + sel.color) b.classList.add("taken");
    else b.classList.add("dimmed");
  });
  if (suggestion) {
    const source = suggestion.source === CENTER ? ui.center : ui.factories.querySelector('.factory[data-source="' + suggestion.source + '"]');
    if (source) flash(source);
  }
}

/* ------------------------------------------------------------------ flashing */
function flash(element) {
  if (!element) return;
  element.classList.remove("flash");
  void element.offsetWidth; // restart the animation
  element.classList.add("flash");
  setTimeout(() => element.classList.remove("flash"), FLASH_MS);
}

function flashAiMoves(moves) {
  moves.forEach((move) => {
    const source = move.source === CENTER
      ? ui.center
      : ui.factories.querySelector('.factory[data-source="' + move.source + '"]');
    flash(source);
    const row = ui.boardAi.querySelector('[data-row="' + move.dest + '"]');
    flash(row);
  });
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
  head.appendChild(node("span", "tally-delta",
    (p.delta >= 0 ? "+" + p.delta : String(p.delta)) + " → " + p.score_after + " points"));
  card.appendChild(head);

  if (p.tiles.length) {
    const placed = node("div", "placed");
    p.tiles.forEach((t) => {
      const box = node("div", "placed-tile");
      box.appendChild(miniTile(t.color));
      box.appendChild(node("span", "pts", "+" + t.points));
      box.appendChild(node("span", "where", "r" + (t.row + 1) + " c" + (t.col + 1)));
      box.title = t.color_name + " → row " + (t.row + 1) + ", column " + (t.col + 1) +
        " (run of " + t.h_run + " across, " + t.v_run + " down)";
      placed.appendChild(box);
    });
    card.appendChild(placed);
    card.appendChild(node("div", "tally-line",
      "Wall tiles: +" + p.tiling_points + " from " + p.tiles.length + " tile" + (p.tiles.length > 1 ? "s" : "")));
  } else {
    card.appendChild(node("div", "tally-line", "No pattern line was full — nothing moved to the wall."));
  }

  const floorBits = [];
  if (p.floor.tiles.length) floorBits.push(p.floor.tiles.length + " tile" + (p.floor.tiles.length > 1 ? "s" : ""));
  if (p.floor.marker) floorBits.push("the first-player marker");
  card.appendChild(node("div", "tally-line", floorBits.length
    ? "Floor line: " + floorBits.join(" and ") + " → " + p.floor.penalty
    : "Floor line: clean."));
  if (p.carried_rows.length) {
    card.appendChild(node("div", "tally-line", "Carried over: " + p.carried_rows
      .map((r) => r.count + " " + COLORS[r.color] + " in row " + (r.row + 1)).join(", ")));
  }
  return card;
}

function queueRoundSheet(report) {
  const body = node("div", "tally");
  body.appendChild(tallyFor(report, S.human_seat, "You"));
  body.appendChild(tallyFor(report, S.ai_seat, "AI · " + S.agent_name));
  if (!report.game_over) {
    body.appendChild(node("div", "tally-line",
      (report.next_first_player === S.human_seat ? "You" : "The AI") + " start" +
      (report.next_first_player === S.human_seat ? "" : "s") + " the next round."));
  }
  sheets.push({ title: "Round " + (report.round + 1) + " tiled", body: body, cta: "Carry on" });
}

function queueFinalSheet(final) {
  const body = node("div");
  const table = node("table", "bonus-table");
  const head = node("thead");
  const hr = node("tr");
  ["", "Before bonus", "Rows ×2", "Columns ×7", "Colours ×10", "Bonus", "Final"]
    .forEach((h) => hr.appendChild(node("th", null, h)));
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
  sheets.push({ title: final.headline, body: body, cta: "Deal another" });
}

function showNextSheet() {
  const sheet = sheets.shift();
  if (!sheet) { ui.overlay.classList.add("hidden"); return; }
  ui.overlayTitle.textContent = sheet.title;
  ui.overlayBody.innerHTML = "";
  ui.overlayBody.appendChild(sheet.body);
  ui.overlayOk.textContent = sheet.cta || "Carry on";
  ui.overlay.classList.remove("hidden");
  ui.overlayOk.focus();
}

/* --------------------------------------------------------------------- wiring */
/** Fill the dropdown from the server, so the two lists cannot drift apart. */
async function loadAgentList() {
  const custom = ui.opponent.querySelector('option[value="custom"]');
  try {
    const data = await api("/api/agents");
    if (!data.baselines || !custom) return;
    ui.opponent.innerHTML = "";
    data.baselines.forEach((spec) => {
      const option = document.createElement("option");
      option.value = spec;
      option.textContent = spec.charAt(0).toUpperCase() + spec.slice(1);
      ui.opponent.appendChild(option);
    });
    ui.opponent.appendChild(custom);
    ui.opponent.value = data.default || data.baselines[0];
    if (data.custom_example) ui.spec.placeholder = data.custom_example;
  } catch (err) {
    // the markup already lists the baselines; nothing to do
  }
}

ui.setup.addEventListener("submit", newGame);
ui.opponent.addEventListener("change", () => {
  const custom = ui.opponent.value === "custom";
  ui.specField.classList.toggle("hidden", !custom);
  if (custom) ui.spec.focus();
});
ui.hint.addEventListener("click", askHint);
ui.cancel.addEventListener("click", () => { sel = null; suggestion = null; render(); });
ui.overlayOk.addEventListener("click", () => {
  const wasFinal = ui.overlayOk.textContent === "Deal another";
  showNextSheet();
  if (wasFinal && !sheets.length) newGame();
});
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (!ui.overlay.classList.contains("hidden")) showNextSheet();
  else if (sel) { sel = null; suggestion = null; render(); }
});

loadAgentList();
refresh();
