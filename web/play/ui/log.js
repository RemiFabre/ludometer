/* The move log: every move of the game, in one flat list.
 *
 * Deliberately uniform — same type, same weight, same rule for every line,
 * newest at the bottom. A log entry is a record, not a headline, so nothing is
 * highlighted; the status band is where "what is happening now" lives.
 *
 * An entry is `{n, kind, text, side?, coach?}`. `coach` (see `coachChip`) is the
 * verdict the search returned on one of your own moves.
 */

import { node } from "./dom.js";

const TAGS = { move: "", round: "round", end: "final", think: "search", start: "setup" };

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

/** Draw the whole log into `host` (a `<ol>`), oldest first, scrolled to the end. */
export function renderLog(host, entries) {
  host.innerHTML = "";
  (entries || []).forEach((entry) => {
    const li = node("li", "log-entry");
    li.dataset.kind = entry.kind;
    if (entry.side) li.dataset.side = entry.side;
    const tag = entry.kind === "move" ? entry.side || "" : TAGS[entry.kind] || entry.kind;
    li.appendChild(node("span", "log-tag", tag === "human" ? "you" : tag));
    const body = node("span", "log-text", entry.text);
    if (entry.coach) body.appendChild(coachChip(entry.coach));
    li.appendChild(body);
    host.appendChild(li);
  });
  host.scrollTop = host.scrollHeight;
}
