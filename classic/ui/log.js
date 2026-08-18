/* The move log: every move of the game, in one flat list.
 *
 * Newest at the **top**. You look at a log to check what just happened far more
 * often than to read the game from the beginning, and putting the newest line
 * where your eye already is saves a scroll on every single turn.
 *
 * Deliberately uniform — same type, same weight, same rule for every line,
 * including the newest one. A log entry is a record, not a headline; the status
 * band is where "what is happening now" lives, and highlighting the top line
 * would make this list compete with it.
 *
 * Moves are drawn **pictographically**: the tiles that moved are little tiles,
 * in the same glazes they have on the board, so "took 3 ochre" is three ochre
 * squares rather than a colour word you have to translate back into a colour.
 * Only the places stay as words, because "factory 3" has no picture.
 *
 * An entry is `{n, kind, text, ply?, side?, coach?}`, and a move entry also
 * carries `{count, color, source, dest, took_marker?, overflow?}` — with those
 * missing, the entry falls back to its sentence, so any port that has not
 * caught up still reads correctly.
 */

import { CENTER, COLORS, FLOOR, node } from "./dom.js";

const TAGS = { move: "", round: "round", end: "final", think: "search", start: "setup" };

/** Short enough to sit next to the tiles: "factory 3", "middle", "row 5". */
const shortSource = (source) => (source === CENTER ? "middle" : "factory " + (source + 1));
const shortDest = (dest) => (dest === FLOOR ? "floor" : "row " + (dest + 1));

/** One tile, at log size. */
export function glyph(color) {
  const g = node("i", "glyph");
  g.dataset.color = color;
  g.title = COLORS[color];
  return g;
}

/** `count` tiles of `color`, as a row of little tiles. */
export function glyphRun(color, count, cls) {
  const wrap = node("span", "glyphs" + (cls ? " " + cls : ""));
  for (let i = 0; i < count; i++) wrap.appendChild(glyph(color));
  return wrap;
}

/** How the delta reads: at the AI's choice, a slip, or a real mistake. */
export function grade(delta) {
  if (delta === null || delta === undefined) return "unrated";
  if (delta >= -0.005) return "best";
  if (delta >= -0.05) return "slip";
  return "blunder";
}

export function formatDelta(delta) {
  if (delta === null || delta === undefined) return "unrated";
  const rounded = Math.abs(delta) < 0.005 ? 0 : delta;
  return (rounded === 0 ? "0.00" : rounded.toFixed(2)).replace("-", "−");
}

/** The coach's verdict on one move, as a small inline chip. */
export function coachChip(coach) {
  const chip = node("span", "coach-chip");
  if (!coach) return chip;
  if (coach.pending) {
    chip.dataset.grade = "pending";
    chip.appendChild(node("span", "coach-delta", "rating…"));
    return chip;
  }
  if (coach.unrated) {
    chip.dataset.grade = "unrated";
    chip.appendChild(node("span", "coach-delta", "unrated"));
    chip.appendChild(
      node("span", "coach-why", coach.reason || "the search never explored this move")
    );
    return chip;
  }
  chip.dataset.grade = grade(coach.delta);
  chip.appendChild(node("span", "coach-delta", formatDelta(coach.delta)));
  if (coach.best_text && coach.delta <= -0.02) {
    chip.appendChild(node("span", "coach-why", "AI preferred: " + coach.best_text));
  } else if (coach.forced) {
    chip.appendChild(node("span", "coach-why", "only move"));
  }
  return chip;
}

/** Whether an entry carries enough to be drawn as pictures. */
function isPictorial(entry) {
  return (
    entry.kind === "move" &&
    typeof entry.color === "number" &&
    typeof entry.count === "number" &&
    typeof entry.source === "number" &&
    typeof entry.dest === "number"
  );
}

/** The body of one move entry: tiles, then where from and where to. */
function moveBody(entry) {
  const body = node("span", "log-text");
  body.appendChild(glyphRun(entry.color, entry.count));
  if (entry.took_marker) {
    const wrap = node("span", "glyphs");
    const chip = node("i", "glyph marker-glyph", "1");
    chip.title = "and the first-player marker";
    wrap.appendChild(chip);
    body.appendChild(wrap);
  }
  body.appendChild(node("span", "log-where", shortSource(entry.source)));
  body.appendChild(node("span", "log-arrow", "→"));
  body.appendChild(node("span", "log-where", shortDest(entry.dest)));
  if (entry.overflow) {
    body.appendChild(
      node("span", "log-aside", " · " + entry.overflow + " to the floor")
    );
  }
  body.title = entry.text || "";
  return body;
}

/**
 * Draw the whole log into `host` (an `<ol>`), **newest first**.
 *
 * `entries` stays in play order — the reversing happens here, so callers never
 * have to think about it and the ply numbers keep meaning what they say.
 */
export function renderLog(host, entries) {
  host.innerHTML = "";
  const list = (entries || []).slice().reverse();
  list.forEach((entry) => {
    const li = node("li", "log-entry");
    li.dataset.kind = entry.kind;
    if (entry.side) li.dataset.side = entry.side;
    if (entry.ply !== undefined) li.dataset.ply = entry.ply;
    const tag = entry.kind === "move" ? entry.side || "" : TAGS[entry.kind] || entry.kind;
    li.appendChild(node("span", "log-tag", tag === "human" ? "you" : tag));
    const body = isPictorial(entry) ? moveBody(entry) : node("span", "log-text", entry.text);
    if (entry.coach) body.appendChild(coachChip(entry.coach));
    li.appendChild(body);
    host.appendChild(li);
  });
  host.scrollTop = 0; // the newest line is at the top, so that is where we look
}
