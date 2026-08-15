/* Confirm mode: a move is placed first, played only on your word.
 *
 * Clicking a row *places* the tiles — you see the board exactly as it would
 * stand — and nothing is committed until you press "Play this move"; "Take it
 * back" returns the tiles to your hand. The pace every board-game site has
 * taught people, and the difference between exploring a position and being
 * bound by a misclick — so it is ON by default, with a switch in the settings
 * panel for players who would rather commit on the first click.
 *
 * This module is only the remembered choice; the placement itself lives in the
 * page that owns the game (see the player's app.js).
 */

const STORE_KEY = "ludometer.confirm";

let current = true;
const listeners = [];

function readStored() {
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    return raw === null ? true : raw !== "off";
  } catch (err) {
    return true; // private mode: the careful default stands
  }
}

/** Whether moves wait for confirmation. */
export const confirmOn = () => current;

/** Choose. Persists, and tells every listener. */
export function setConfirm(on) {
  current = !!on;
  try {
    window.localStorage.setItem(STORE_KEY, current ? "on" : "off");
  } catch (err) {
    /* forgetting the choice is no reason to refuse it */
  }
  listeners.forEach((fn) => fn(current));
  return current;
}

export function onConfirmChange(fn) {
  listeners.push(fn);
  return () => listeners.splice(listeners.indexOf(fn), 1);
}

/** Load the stored choice. Call once, early. */
export function initConfirm() {
  current = readStored();
  return current;
}
