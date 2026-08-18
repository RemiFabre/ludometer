/* Board layout: how much room the two boards take.
 *
 * "side" keeps the whole game on one screen: the factories on the left, both
 * boards — smaller, but both visible at once — beside them. "stack" is the
 * classic table: big boards below the factories. The default follows the
 * device — a phone starts on "side" (Rémi's phone spec), a desktop starts on
 * "stack" (Rémi tried "side" there and reached for the + at once).
 *
 * The choice is written onto <body data-boards="…">; the page's stylesheet
 * does the rest. Persisted like the animation speed, switched from the
 * settings panel (the − / + row).
 */

const STORE_KEY = "ludometer.boards";
const MODES = ["side", "stack"];

const defaultMode = () =>
  window.matchMedia && window.matchMedia("(max-width: 720px)").matches ? "side" : "stack";

let current = "stack";
const listeners = [];

function readStored() {
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    return MODES.includes(raw) ? raw : defaultMode();
  } catch (err) {
    return defaultMode(); // private mode: still the right default for the device
  }
}

/** The current mode: "side" or "stack". */
export const boardsMode = () => current;

/** Choose. Persists, writes the body attribute, tells every listener. */
export function setBoardsMode(mode) {
  current = MODES.includes(mode) ? mode : "side";
  try {
    window.localStorage.setItem(STORE_KEY, current);
  } catch (err) {
    /* forgetting the choice is no reason to refuse it */
  }
  document.body.dataset.boards = current;
  listeners.forEach((fn) => fn(current));
  return current;
}

export function onBoardsChange(fn) {
  listeners.push(fn);
  return () => listeners.splice(listeners.indexOf(fn), 1);
}

/** Load the stored choice and stamp it on the body. Call once, early. */
export function initBoards() {
  current = readStored();
  document.body.dataset.boards = current;
  return current;
}
