/* Score pop-ups: every point announced where it is earned.
 *
 * A pop is a small number that floats off the square that produced it — "+3"
 * off a wall tile as it lands, "−2" off a floor line as it bills you. They
 * exist to teach: watching the numbers appear one by one is how a new player
 * learns what a position is worth, so they are ON by default and switchable in
 * the settings panel (persisted, like the animation speed).
 *
 * Pops live in their own fixed layer, deliberately not the flight layer — a
 * number is not a tile, and the tests that count tile flights must not count
 * these.
 *
 *   popScore(layer, cellEl, "+3");            // a gain, in the gain colour
 *   popScore(layer, floorEl, "−2", "loss");   // a cost, in the loss colour
 */

const STORE_KEY = "ludometer.pops";

let current = true;
const listeners = [];

function readStored() {
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    return raw === null ? true : raw !== "off";
  } catch (err) {
    return true; // private mode: teach anyway
  }
}

/** Whether score pop-ups are on. */
export const popupsOn = () => current;

/** Choose. Persists, and tells every listener. */
export function setPopups(on) {
  current = !!on;
  try {
    window.localStorage.setItem(STORE_KEY, current ? "on" : "off");
  } catch (err) {
    /* forgetting the choice is no reason to refuse it */
  }
  listeners.forEach((fn) => fn(current));
  return current;
}

export function onPopupsChange(fn) {
  listeners.push(fn);
  return () => listeners.splice(listeners.indexOf(fn), 1);
}

/** Load the stored choice. Call once, early. */
export function initPopups() {
  current = readStored();
  return current;
}

/**
 * Float `text` off `target` (an element or a client rectangle) into `layer`.
 *
 * `kind` is "gain" (default) or "loss" and only picks the colour. The pop
 * removes itself; nothing waits on it. A no-op while pop-ups are off, or when
 * the target has no on-screen rectangle.
 */
export function popScore(layer, target, text, kind) {
  if (!current || !layer || !target) return;
  const r = typeof target.getBoundingClientRect === "function"
    ? target.getBoundingClientRect()
    : target;
  if (!r || (!r.width && !r.height)) return;
  const pop = document.createElement("span");
  pop.className = "score-pop";
  pop.dataset.kind = kind || "gain";
  pop.textContent = text;
  pop.style.left = r.left + r.width / 2 + "px";
  pop.style.top = r.top + "px";
  layer.appendChild(pop);
  setTimeout(() => pop.remove(), 1400);
}

/**
 * Drop every pop still floating. Pops are fixed-position, so a re-render that
 * moves the table leaves them hanging over whatever now sits at their old
 * coordinates — a stray "−2" over a pattern row reads as a mystery artifact.
 * Call this when the layout is about to shift (a round re-render); mid-render
 * nobody sees a pop vanish.
 */
export function clearPops(layer) {
  if (layer) layer.textContent = "";
}
