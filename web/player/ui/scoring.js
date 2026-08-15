/* Scoring, shown on the table instead of over it.
 *
 * Two panels, both inline: the round tally (what each wall tile earned, what the
 * floor cost) and the final reckoning (the end-game bonuses, broken down). They
 * sit in a two-column grid under the boards, left column = left board, so each
 * player's numbers stay under their own wall and the position behind them is
 * still fully readable.
 *
 * Both take the server's own report objects — no number here is re-derived.
 */

import { COLORS, node, tileEl } from "./dom.js";

function miniTile(color) {
  const t = tileEl(color, "mini");
  return t;
}

function playerCard(label, kids) {
  const card = node("div", "score-card");
  card.appendChild(node("h3", "score-card-who", label));
  kids.forEach((k) => card.appendChild(k));
  return card;
}

function line(text) {
  return node("p", "score-line", text);
}

/** One player's round: the tiles that reached the wall, and the floor's bill. */
function roundCard(report, seat, label) {
  const p = report.players[seat];
  const kids = [];
  const delta = node(
    "p",
    "score-delta" + (p.delta < 0 ? " down" : p.delta > 0 ? " up" : ""),
    (p.delta >= 0 ? "+" + p.delta : String(p.delta)) + " → " + p.score_after
  );
  kids.push(delta);

  if (p.tiles.length) {
    const placed = node("div", "score-tiles");
    p.tiles.forEach((t) => {
      const box = node("div", "score-tile");
      box.appendChild(miniTile(t.color));
      box.appendChild(node("span", "pts", "+" + t.points));
      box.appendChild(node("span", "where", "r" + (t.row + 1) + "·c" + (t.col + 1)));
      box.title =
        t.color_name +
        " → row " +
        (t.row + 1) +
        ", column " +
        (t.col + 1) +
        " (run of " +
        t.h_run +
        " across, " +
        t.v_run +
        " down)";
      placed.appendChild(box);
    });
    kids.push(placed);
    kids.push(
      line(
        "Wall: +" +
          p.tiling_points +
          " from " +
          p.tiles.length +
          " tile" +
          (p.tiles.length > 1 ? "s" : "")
      )
    );
  } else {
    kids.push(line("No pattern line was full — nothing reached the wall."));
  }

  const floorBits = [];
  if (p.floor.tiles.length) {
    floorBits.push(p.floor.tiles.length + " tile" + (p.floor.tiles.length > 1 ? "s" : ""));
  }
  if (p.floor.marker) floorBits.push("the marker");
  kids.push(
    line(
      floorBits.length
        ? "Floor: " + floorBits.join(" and ") + " → " + p.floor.penalty
        : "Floor: clean."
    )
  );
  if (p.carried_rows.length) {
    kids.push(
      line(
        "Carried over: " +
          p.carried_rows
            .map((r) => r.count + " " + COLORS[r.color] + " in row " + (r.row + 1))
            .join(", ")
      )
    );
  }
  return playerCard(label, kids);
}

/**
 * Draw the round tally into `host`.
 * `sides` is `[[seat, label], [seat, label]]`, left column first.
 */
export function renderRoundPanel(host, report, sides) {
  host.innerHTML = "";
  host.classList.add("scoring");
  host.hidden = false;
  const head = node("div", "scoring-head");
  head.appendChild(node("h2", "scoring-title", "Round " + (report.round + 1) + " scored"));
  if (!report.game_over) {
    const yours = sides[0][0] === report.next_first_player;
    const who = yours ? sides[0][1] + " start" : sides[1][1] + " starts";
    head.appendChild(node("span", "scoring-note", who + " the next round"));
  }
  host.appendChild(head);
  const grid = node("div", "scoring-grid");
  sides.forEach(([seat, label]) => grid.appendChild(roundCard(report, seat, label)));
  host.appendChild(grid);
}

/** One player's end-game bonuses, itemised. */
function bonusCard(final, seat, label) {
  const b = final.bonuses[seat];
  const kids = [];
  const total = node("p", "score-delta big", String(b.final_score));
  if (final.winner === seat) total.classList.add("won");
  kids.push(total);

  const rows = [
    ["On the board", b.score_before_bonus, ""],
    ["Full rows ×2", "+" + b.row_points, b.rows + " row" + (b.rows === 1 ? "" : "s")],
    ["Full columns ×7", "+" + b.col_points, b.cols + " column" + (b.cols === 1 ? "" : "s")],
    [
      "All five of a colour ×10",
      "+" + b.color_points,
      b.colors + " colour" + (b.colors === 1 ? "" : "s"),
    ],
    ["Bonus", "+" + b.total, ""],
  ];
  const table = node("dl", "bonus-list");
  rows.forEach(([name, value, note], i) => {
    const dt = node("dt", null, name);
    const dd = node("dd", null, String(value));
    if (i === rows.length - 1) {
      dt.classList.add("sum");
      dd.classList.add("sum");
    }
    if (note) dd.title = note;
    table.append(dt, dd);
  });
  kids.push(table);
  return playerCard(label, kids);
}

/** Draw the final reckoning into `host` — the board above stays readable. */
export function renderFinalPanel(host, final, sides) {
  host.innerHTML = "";
  host.classList.add("scoring", "final");
  host.hidden = false;
  const head = node("div", "scoring-head");
  head.appendChild(node("h2", "scoring-title", final.headline));
  head.appendChild(
    node(
      "span",
      "scoring-note",
      "Final score " +
        final.scores[sides[0][0]] +
        "–" +
        final.scores[sides[1][0]] +
        " after " +
        final.rounds_played +
        " rounds"
    )
  );
  host.appendChild(head);
  const grid = node("div", "scoring-grid");
  sides.forEach(([seat, label]) => grid.appendChild(bonusCard(final, seat, label)));
  host.appendChild(grid);
  if (final.winner === null) {
    host.appendChild(line("Level on score and on completed rows — the rulebook calls that a draw."));
  }
  if (final.exhausted) {
    host.appendChild(line("The bag and the lid ran dry, so the game stopped early."));
  }
}

/** Hide whichever panel is showing (a new deal, or the next round starting). */
export function clearScoring(host) {
  host.innerHTML = "";
  host.hidden = true;
  host.classList.remove("final");
}
