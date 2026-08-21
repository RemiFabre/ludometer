/* The Azul furniture: one player board, and the middle of the table.
 *
 * `createBoard(host, {...})` owns one container element and redraws it from a
 * view object; `createMiddle(host, {...})` does the same for the factories and
 * the centre dish. Both boards are built by the same function with the same
 * markup, so the human's and the AI's are identical down to the pixel — only
 * `interactive` differs.
 *
 * The modules never fetch anything and never name an endpoint: a caller hands
 * them state and callbacks, and they hand back the elements an animation needs
 * to aim at.
 *
 *   const you = createBoard(el, { seat: 0, interactive: true, onPlay: play });
 *   you.render({ state, legalActions, selection, toMove: true, title: "You" });
 *   you.lineSlots(2);  // where tiles going to row 3 should land
 */

import {
  CENTER,
  COLORS,
  FLOOR,
  FLOOR_PENALTIES,
  destsFor,
  markerChip,
  node,
  poolCount,
  preview,
  tileEl,
  actionId,
} from "./dom.js";

const EMPTY_VIEW = { state: null, legalActions: [], selection: null };

/* --------------------------------------------------------------- player board */
export function createBoard(host, options = {}) {
  const opts = Object.assign(
    { seat: 0, interactive: false, onPlay: null, onBlocked: null },
    options
  );
  host.innerHTML = ""; // re-seating a board rebuilds it in place
  host.classList.add("board");
  host.dataset.seat = opts.seat;

  const head = node("div", "board-head");
  const who = node("span", "board-who", opts.title || "");
  const score = node("span", "board-score", "0");
  const notes = node("span", "board-notes", "");
  head.append(who, score, notes);

  const grid = node("div", "board-grid");
  const floorWrap = node("div", "floor-wrap");
  host.append(head, grid, floorWrap);

  let view = EMPTY_VIEW;

  function draw() {
    const state = view.state;
    grid.innerHTML = "";
    floorWrap.innerHTML = "";
    if (!state) return;

    const seat = opts.seat;
    const me = state.players[seat];
    const legal = new Set(view.legalActions || []);
    const sel = opts.interactive ? view.selection : null;
    let open = sel ? destsFor(legal, sel.source, sel.color) : [];
    // a suggestion points at ONE row: only that row invites the click, and the
    // others stay neutral rather than "blocked" (they are legal, just not it)
    const suggesting = sel && view.openOnly != null;
    if (suggesting) open = open.filter((r) => r === view.openOnly);

    who.textContent = view.title || who.textContent;
    if (score.dataset.counting !== undefined && score.dataset.counting === String(me.score)) {
      // a scoreCount() is writing this number as arithmetic; let it finish
    } else {
      delete score.dataset.counting;
      score.textContent = me.score;
    }
    const bits = [];
    if (me.floor_penalty) bits.push(me.floor_penalty + " on the floor");
    if (me.completed_rows) {
      bits.push(me.completed_rows + " full row" + (me.completed_rows > 1 ? "s" : ""));
    }
    if (me.floor_marker) bits.push("holds the marker");
    notes.textContent = bits.join(" · ");
    host.classList.toggle("to-move", !!view.toMove);
    host.classList.toggle("mine", !!opts.interactive);

    for (let r = 0; r < 5; r++) {
      const line = me.pattern_lines[r];
      const isOpen = open.indexOf(r) !== -1;
      const row = node(sel ? "button" : "div", "line");
      if (row.tagName === "BUTTON") row.type = "button";
      row.dataset.row = r;

      const note = node("span", "row-note");
      if (sel) {
        if (isOpen) {
          const p = preview(state, seat, sel.source, sel.color, r);
          let text = p.placed + " of " + p.count + " here";
          if (p.overflow) text += ", " + p.overflow + " spills (" + p.penalty + ")";
          else if (p.completes) text += ", fills the row";
          note.textContent = text;
          row.classList.add("open");
          row.addEventListener("click", () => {
            if (opts.onPlay) opts.onPlay(actionId(sel.source, sel.color, r), r);
          });
        } else if (suggesting) {
          note.textContent = line.count
            ? line.count + "/" + line.capacity + " " + COLORS[line.color]
            : line.capacity + (line.capacity === 1 ? " slot" : " slots");
        } else {
          row.classList.add("blocked");
          row.title = blockedReason(state, seat, sel.color, r);
          note.textContent = row.title;
          row.setAttribute("aria-disabled", "true");
        }
      } else {
        note.textContent = line.count
          ? line.count + "/" + line.capacity + " " + COLORS[line.color]
          : line.capacity + (line.capacity === 1 ? " slot" : " slots");
      }
      row.appendChild(note);

      // a pattern line fills towards the wall, so the empty slots sit on the left
      for (let k = 0; k < line.capacity - line.count; k++) {
        row.appendChild(node("div", "slot"));
      }
      for (let k = 0; k < line.count; k++) row.appendChild(tileEl(line.color));
      if (view.highlightRow === r) row.classList.add("incoming");
      grid.appendChild(row);

      const wallRow = node("div", "wall-row");
      wallRow.dataset.wallRow = r;
      if (r === 0) wallRow.classList.add("top");
      if (r === 4) wallRow.classList.add("bottom");
      for (let col = 0; col < 5; col++) {
        const colour = (col - r + 5) % 5;
        if (me.wall[r][col]) {
          const t = tileEl(colour, "placed");
          t.dataset.row = r;
          t.dataset.col = col;
          t.title = COLORS[colour] + ", row " + (r + 1) + ", column " + (col + 1);
          wallRow.appendChild(t);
        } else {
          // an empty square is a neutral well with the glaze only outlined in it
          const cell = node("div", "cell empty");
          cell.dataset.color = colour;
          cell.dataset.row = r;
          cell.dataset.col = col;
          cell.title = COLORS[colour] + " goes here";
          wallRow.appendChild(cell);
        }
      }
      grid.appendChild(wallRow);
    }

    const floorOpen = sel && open.indexOf(FLOOR) !== -1;
    const floor = node(floorOpen ? "button" : "div", "floor");
    floor.dataset.row = FLOOR;
    if (floorOpen) {
      floor.type = "button";
      floor.classList.add("open");
      const p = preview(state, seat, sel.source, sel.color, FLOOR);
      floor.title = "Drop all " + p.count + " tiles here (" + p.penalty + " this round)";
      floor.addEventListener("click", () => {
        if (opts.onPlay) opts.onPlay(actionId(sel.source, sel.color, FLOOR), FLOOR);
      });
    }
    const occupants = [];
    if (me.floor_marker) occupants.push("marker");
    me.floor.forEach((n, c) => {
      for (let k = 0; k < n; k++) occupants.push(c);
    });
    for (let i = 0; i < 7; i++) {
      const cellwrap = node("div", "cellwrap");
      const here = occupants[i];
      if (here === "marker") cellwrap.appendChild(markerChip(true));
      else if (here !== undefined) cellwrap.appendChild(tileEl(here));
      else cellwrap.appendChild(node("div", "slot"));
      cellwrap.appendChild(node("span", "pen", FLOOR_PENALTIES[i]));
      floor.appendChild(cellwrap);
    }
    floorWrap.appendChild(floor);

    const penalty = me.floor_penalty;
    const floorNote = node(
      "span",
      "floor-note" + (penalty ? " warn" : ""),
      penalty ? "Floor line: " + penalty + " at the end of this round" : "Floor line: clean"
    );
    if (floorOpen) {
      const p = preview(state, seat, sel.source, sel.color, FLOOR);
      floorNote.textContent =
        "Floor line: " + penalty + " now, " + (penalty + p.penalty) + " if you drop here";
      floorNote.classList.add("warn");
    }
    floorWrap.appendChild(floorNote);
  }

  return {
    el: host,
    seat: opts.seat,
    render(next) {
      view = Object.assign({}, EMPTY_VIEW, next);
      draw();
    },
    setTitle(text) {
      who.textContent = text;
    },
    lineRow: (r) => grid.querySelector('.line[data-row="' + r + '"]'),
    lineSlots: (r) => {
      const row = grid.querySelector('.line[data-row="' + r + '"]');
      return row ? [].slice.call(row.querySelectorAll(".slot")) : [];
    },
    lineTiles: (r) => {
      const row = grid.querySelector('.line[data-row="' + r + '"]');
      return row ? [].slice.call(row.querySelectorAll(".tile")) : [];
    },
    floorEl: () => floorWrap.querySelector(".floor"),
    floorSlots: () => [].slice.call(floorWrap.querySelectorAll(".floor .slot")),
    floorTiles: () => [].slice.call(floorWrap.querySelectorAll(".floor .tile")),
    /**
     * The round's arithmetic, written at the score itself: "10 + 5 = 15",
     * with the 15 exactly where the score sits (the node is right-aligned),
     * then the equation fades and only the sum remains. `opts2` is
     * {hold, fade} in real ms — pass 0/0 for an instant update. The node
     * guards itself against re-renders of the same final score while the
     * equation plays; any other render (new game, browsing) replaces it.
     */
    scoreCount(before, delta, after, opts2 = {}) {
      if (!delta) return;
      const hold = Number(opts2.hold) || 0;
      const fade = Number(opts2.fade) || 0;
      if (!hold && !fade) {
        score.textContent = String(after);
        return;
      }
      const sign = delta > 0 ? " + " : " \u2212 ";
      const eqn = node("span", "score-eqn", before + sign + Math.abs(delta) + " = ");
      score.dataset.counting = String(after);
      score.textContent = "";
      score.append(eqn, document.createTextNode(String(after)));
      setTimeout(() => {
        if (score.dataset.counting !== String(after)) return;
        eqn.style.transition = "opacity " + fade + "ms ease";
        eqn.style.opacity = "0";
        setTimeout(() => {
          if (score.dataset.counting !== String(after)) return;
          delete score.dataset.counting;
          score.textContent = String(after);
        }, fade + 50);
      }, hold);
    },
    wallCell: (r, c) =>
      grid.querySelector('.wall-row[data-wall-row="' + r + '"] [data-col="' + c + '"]'),
    wallTile: (r, c) =>
      grid.querySelector('.tile.placed[data-row="' + r + '"][data-col="' + c + '"]'),
  };
}

/** Why a row is closed for the colour in hand — the hover text on a dark row. */
export function blockedReason(state, seat, color, dest) {
  const me = state.players[seat];
  const line = me.pattern_lines[dest];
  if (line.count >= line.capacity) return "This row is already full.";
  if (line.count > 0 && line.color !== color) {
    return "This row holds " + COLORS[line.color] + " tiles.";
  }
  if (me.wall[dest][(color + dest) % 5]) {
    return COLORS[color] + " is already on your wall in this row.";
  }
  return "Not playable right now.";
}

/* ------------------------------------------------------- middle of the table */
export function createMiddle(host, options = {}) {
  const opts = Object.assign({ onPick: null }, options);
  host.innerHTML = "";
  host.classList.add("middle-table");
  const dishes = node("div", "factories");
  const centre = node("div", "center-dish");
  centre.dataset.source = CENTER;
  host.append(dishes, centre);

  function tileButton(state, legal, canPick, source, color) {
    const b = node("button", "tile");
    b.type = "button";
    b.dataset.color = color;
    b.dataset.source = source;
    b.dataset.key = source + ":" + color;
    const dests = canPick ? destsFor(legal, source, color) : [];
    if (!dests.length) {
      b.disabled = true;
      if (canPick) b.title = "Nothing you can do with these yet.";
    } else {
      b.title =
        poolCount(state, source, color) + " " + COLORS[color] + ", click to pick them up";
      b.addEventListener("click", () => {
        if (opts.onPick) opts.onPick(source, color);
      });
    }
    return b;
  }

  return {
    el: host,
    render(view) {
      const state = view.state;
      dishes.innerHTML = "";
      centre.innerHTML = "";
      if (!state) return;
      const legal = new Set(view.legalActions || []);
      const canPick = !!view.canPick;
      const sel = view.selection;

      state.factories.forEach((counts, i) => {
        const dish = node("div", "factory");
        dish.dataset.source = i;
        const total = counts.reduce((a, b) => a + b, 0);
        if (!total) {
          dish.classList.add("empty");
          dish.appendChild(node("span", "empty-note", "empty"));
        } else {
          counts.forEach((n, c) => {
            for (let k = 0; k < n; k++) {
              dish.appendChild(tileButton(state, legal, canPick, i, c));
            }
          });
        }
        const id = node("span", "fac-id", String(i + 1));
        id.title = "Factory " + (i + 1);
        dish.appendChild(id);
        dishes.appendChild(dish);
      });

      const centreTotal = state.center.reduce((a, b) => a + b, 0);
      if (state.marker_in_center) centre.appendChild(markerChip(false));
      if (!centreTotal) {
        centre.appendChild(
          node(
            "span",
            "empty-note",
            state.marker_in_center ? "middle (marker only)" : "middle (empty)"
          )
        );
      } else {
        state.center.forEach((n, c) => {
          if (!n) return;
          const group = node("div", "group");
          for (let k = 0; k < n; k++) {
            group.appendChild(tileButton(state, legal, canPick, CENTER, c));
          }
          centre.appendChild(group);
        });
      }

      if (sel) {
        host.querySelectorAll("button.tile").forEach((b) => {
          if (b.dataset.key === sel.source + ":" + sel.color) b.classList.add("taken");
          else b.classList.add("dimmed");
        });
      }
    },
    sourceEl: (source) =>
      source === CENTER ? centre : dishes.querySelector('.factory[data-source="' + source + '"]'),
    centerEl: () => centre,
    /** The drawn tiles of one colour in one dish — what a move takes away. */
    sourceTiles(source, color, count) {
      const dish = this.sourceEl(source);
      if (!dish) return [];
      const all = dish.querySelectorAll('.tile[data-color="' + color + '"]');
      return [].slice.call(all, 0, count === undefined ? all.length : count);
    },
    /** Everything else in that dish — the tiles that get pushed to the middle. */
    remainderTiles(source, color) {
      if (source === CENTER) return [];
      const dish = this.sourceEl(source);
      if (!dish) return [];
      return [].slice
        .call(dish.querySelectorAll(".tile"))
        .filter((t) => Number(t.dataset.color) !== color);
    },
  };
}
