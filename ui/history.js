/* Move navigation — the game as a list of positions you can walk back through.
 *
 * Chess clients have taught everyone the same two keys, so this is those two
 * keys: ← steps back a move, → steps forward, and stepping past the last move
 * puts you back in the live game. It is pure replay of positions the page has
 * *already* been given — there is no request, no re-search and no engine call
 * behind these buttons, and the live game is not touched while you browse.
 *
 * Frames are stored by ply, so `frames[k]` is the position after k moves:
 * `frames[0]` is the deal. Every move the back end reports carries the position
 * it was played from, which is exactly the position after the move before it,
 * so the array fills itself in as the game goes on.
 *
 *   const nav = createHistory(el, { log: () => S.log, onChange: render });
 *   nav.reset(state, 0);          // a new deal
 *   nav.note(ply, state);         // a position we have seen
 *   const frame = nav.frame();    // null while live; {state, log, ply} while browsing
 */

import { node } from "./dom.js";

export function createHistory(host, options = {}) {
  const opts = Object.assign({ log: () => [], onChange: null, enabled: null }, options);
  host.classList.add("nav");
  host.innerHTML = "";

  const prev = node("button", "nav-btn", "◀");
  prev.type = "button";
  prev.title = "Previous move (←)";
  prev.setAttribute("aria-label", "Previous move");
  const count = node("span", "nav-count", "");
  const next = node("button", "nav-btn", "▶");
  next.type = "button";
  next.title = "Next move (→)";
  next.setAttribute("aria-label", "Next move");
  // stepping forward one move at a time is no way back from move 3 of 40
  const live = node("button", "nav-btn nav-live", "Latest");
  live.type = "button";
  live.title = "Back to the live game (End)";
  live.hidden = true;
  host.append(prev, count, next, live);

  const frames = [];   // frames[ply] = state JSON
  let latest = 0;      // the live game's ply
  let viewing = null;  // the ply being looked at, or null when live

  const has = (ply) => ply >= 0 && ply <= latest && !!frames[ply];

  /** The nearest recorded ply at or beyond `from` in direction `step`. */
  function seek(from, step) {
    for (let p = from; p >= 0 && p <= latest; p += step) {
      if (has(p)) return p;
    }
    return null;
  }

  /** The log as it stood after `ply` moves — entries carry the ply they belong to. */
  function logAt(ply) {
    return (opts.log() || []).filter((e) => e.ply === undefined || e.ply <= ply);
  }

  /* `change` is {from, to}: the ply that was showing and the ply that now is
   * (the live ply when not browsing). Pages that animate the step read it;
   * pages that just redraw ignore the argument, as they always have. */
  function announce(change) {
    draw();
    if (opts.onChange) opts.onChange(change);
  }

  const usable = () => !opts.enabled || opts.enabled();

  function draw() {
    const browsing = viewing !== null;
    const on = usable();
    prev.disabled = !on || seek(browsing ? viewing - 1 : latest - 1, -1) === null;
    next.disabled = !on || !browsing;
    live.hidden = !browsing;
    live.disabled = !on;
    count.textContent = browsing ? viewing + " / " + latest : latest ? "move " + latest : "";
    host.classList.toggle("browsing", browsing);
    document.body.classList.toggle("viewing", browsing);
  }

  /** Step by `delta` moves. Stepping past the last move resumes the live game. */
  function step(delta) {
    if (!usable()) return false;
    const from = viewing === null ? latest : viewing;
    const target = from + delta;
    if (target >= latest) {
      if (viewing === null) return false;
      viewing = null;
      announce({ from, to: latest });
      return true;
    }
    const found = seek(target, delta < 0 ? -1 : 1);
    if (found === null || found === viewing) return false;
    viewing = found;
    announce({ from, to: found });
    return true;
  }

  /** Jump back to the live game. */
  function toLatest() {
    if (viewing === null || !usable()) return false;
    const from = viewing;
    viewing = null;
    announce({ from, to: latest });
    return true;
  }

  prev.addEventListener("click", () => step(-1));
  next.addEventListener("click", () => step(1));
  live.addEventListener("click", toLatest);
  draw();

  return {
    el: host,

    /** Start a new game: `state` is the position after `ply` moves. */
    reset(state, ply = 0) {
      frames.length = 0;
      latest = ply;
      viewing = null;
      if (state) frames[ply] = state;
      draw();
    },

    /** Record the position after `ply` moves. Fills holes; extends the game's end. */
    note(ply, state) {
      if (!state || typeof ply !== "number" || ply < 0) return;
      frames[ply] = state;
      if (ply > latest) latest = ply;
      draw();
    },

    /** The frame to draw, or null when the live game is what should be drawn. */
    frame() {
      if (viewing === null) return null;
      return { ply: viewing, state: frames[viewing], log: logAt(viewing), of: latest };
    },

    browsing: () => viewing !== null,
    latest: () => latest,
    /** The recorded position after `ply` moves, or null. */
    stateAt: (ply) => (has(ply) ? frames[ply] : null),
    step,
    toLatest,
    draw,
    prevButton: prev,
    nextButton: next,
    liveButton: live,
  };
}

/**
 * ← / → anywhere on the page, except while typing in a field.
 *
 * Returns the listener so a host page could remove it; every page we ship keeps
 * it for the life of the document.
 */
export function bindHistoryKeys(nav, options = {}) {
  const listener = (event) => {
    const back = event.key === "ArrowLeft";
    const forward = event.key === "ArrowRight";
    const home = event.key === "End";
    if (!back && !forward && !home) return;
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const target = event.target;
    if (target && /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName)) return;
    if (options.enabled && !options.enabled()) return;
    const moved = home ? nav.toLatest() : nav.step(back ? -1 : 1);
    if (moved) event.preventDefault();
  };
  document.addEventListener("keydown", listener);
  return listener;
}
