/* Straight-line tile flights.
 *
 * Every movement on this table is the same gesture: a tile leaves one place and
 * travels in a straight line to another. Five things move — tiles from a dish
 * to a board, a dish's leftovers to the middle, the first-player marker, a full
 * pattern line to the wall, and the floor line to the lid — and all of them are
 * the same call.
 *
 * A flight is a clone in a fixed-position layer: the real tiles are hidden for
 * the duration, and the page re-renders from the new state when the group
 * lands. Nothing here knows what a move *means*; callers pass elements (or
 * rectangles) and a colour.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * WHY THERE IS NO `prefers-reduced-motion` CHECK IN THIS FILE
 *
 * There used to be one, and it was a bug with a long tail. `flyTiles` returned
 * immediately when the OS asked for reduced motion, `sleep` collapsed to zero,
 * and board.css additionally forced `transition-duration: .001ms !important` on
 * everything. macOS ships "Reduce motion" switched on for a lot of people, and
 * Safari and Chrome both forward it, so those players saw tiles teleport with
 * no way whatsoever to ask for the animation back — while every headless test
 * passed, because the test harness set `no-preference` first.
 *
 * Motion is now governed by one thing only: the speed setting in the page
 * (`speed()` / `setSpeed()` below, persisted in localStorage). The OS flag is
 * still readable through `prefersReducedMotion()` so the settings panel can
 * *mention* it, but it never silences anything on its own.
 * ───────────────────────────────────────────────────────────────────────────── */

export const FLIGHT_MS = 460; // one group of tiles, door to door, at 1×
export const STAGGER_MS = 45; // ... plus this per tile after the first
export const LAND_MS = 90; // a beat to let the last tile settle

const STORE_KEY = "ludometer.anim.speed";
/** What the settings panel offers. 0 means "no animation at all". */
export const SPEEDS = [0, 0.5, 1, 2];
export const DEFAULT_SPEED = 1;

let current = DEFAULT_SPEED;
const listeners = [];

function readStored() {
  try {
    const raw = window.localStorage.getItem(STORE_KEY);
    if (raw === null) return DEFAULT_SPEED;
    const value = Number(raw);
    return SPEEDS.indexOf(value) === -1 ? DEFAULT_SPEED : value;
  } catch (err) {
    return DEFAULT_SPEED; // private mode, a locked-down profile: animate anyway
  }
}

/** The current animation speed: 0 (off), 0.5 (slower), 1 (default) or 2 (fast). */
export const speed = () => current;

/** Whether tiles move at all. */
export const animated = () => current > 0;

/**
 * Choose a speed. Persists it, reflects it on `<body data-anim>` (so CSS can
 * see it) and tells every listener.
 */
export function setSpeed(value) {
  const next = SPEEDS.indexOf(Number(value)) === -1 ? DEFAULT_SPEED : Number(value);
  current = next;
  try {
    window.localStorage.setItem(STORE_KEY, String(next));
  } catch (err) {
    /* not being able to remember it is not a reason to refuse it */
  }
  if (document.body) document.body.dataset.anim = next === 0 ? "off" : String(next);
  listeners.forEach((fn) => fn(next));
  return next;
}

export function onSpeedChange(fn) {
  listeners.push(fn);
  return () => listeners.splice(listeners.indexOf(fn), 1);
}

/** Load the stored speed. Call once, early, before anything animates. */
export function initSpeed() {
  return setSpeed(readStored());
}

/** How long `ms` at 1× lasts at the current speed. Zero when animation is off. */
export function scaled(ms) {
  if (!current) return 0;
  return ms / current;
}

/**
 * Whether the *operating system* asks for reduced motion.
 *
 * Informational only — see the note at the top of this file. The settings panel
 * uses it to explain itself; nothing else may branch on it.
 */
export const prefersReducedMotion = () =>
  !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

/** A pause in a turn sequence, in 1× milliseconds. */
export const sleep = (ms) =>
  new Promise((done) => setTimeout(done, scaled(ms)));

/** A rectangle for an element, or the rectangle itself if one is passed. */
export function rectOf(target) {
  if (!target) return null;
  if (typeof target.getBoundingClientRect === "function") {
    const r = target.getBoundingClientRect();
    if (!r.width && !r.height) return null;
    return r;
  }
  return target;
}

/**
 * Fly `flights` and resolve once they have landed.
 *
 * Each flight is `{from, to, color, hide}`: `from`/`to` are elements or client
 * rectangles, `color` is 0..4 (or "marker"), `hide` defaults to true and hides
 * the source element while its clone is in the air.
 *
 * `options`: `{layer, duration, stagger, scale}` — `duration` and `stagger` are
 * 1× milliseconds and are scaled by the speed setting here. Returns a promise
 * that always resolves (never rejects) so a caller can `await` it in a turn
 * sequence.
 */
export function flyTiles(flights, options = {}) {
  const layer = options.layer;
  const baseDuration = options.duration === undefined ? FLIGHT_MS : options.duration;
  const baseStagger = options.stagger === undefined ? STAGGER_MS : options.stagger;
  if (!layer || !flights || !flights.length || !animated()) return Promise.resolve();
  const duration = scaled(baseDuration);
  const stagger = scaled(baseStagger);

  const airborne = [];
  flights.forEach((flight, i) => {
    const a = rectOf(flight.from);
    const b = rectOf(flight.to);
    if (!a || !b) return;
    const ghost = document.createElement("div");
    ghost.className = "fly-tile " + (flight.color === "marker" ? "marker" : "tile");
    if (flight.color !== "marker") ghost.dataset.color = flight.color;
    else ghost.textContent = "1";
    ghost.style.left = a.left + "px";
    ghost.style.top = a.top + "px";
    ghost.style.width = a.width + "px";
    ghost.style.height = a.height + "px";
    ghost.style.transitionDuration = duration + "ms";
    ghost.style.transitionDelay = i * stagger + "ms";
    layer.appendChild(ghost);
    if (flight.hide !== false && flight.from && flight.from.style) {
      flight.from.style.visibility = "hidden";
    }
    const scale = options.scale === false ? 1 : b.width / (a.width || 1);
    const dx = b.left + (b.width - a.width * scale) / 2 - a.left;
    const dy = b.top + (b.height - a.height * scale) / 2 - a.top;
    airborne.push([ghost, dx, dy, scale]);
  });

  if (!airborne.length) return Promise.resolve();
  const total = duration + (airborne.length - 1) * stagger + scaled(LAND_MS);
  return new Promise((done) => {
    requestAnimationFrame(() => {
      airborne.forEach(([ghost, dx, dy, scale]) => {
        ghost.style.transform =
          "translate(" + dx + "px, " + dy + "px) scale(" + scale.toFixed(3) + ")";
      });
      setTimeout(() => {
        airborne.forEach(([ghost]) => ghost.remove());
        done();
      }, total);
    });
  });
}

/** Total wall time `flyTiles` will take for `count` tiles — for pacing a turn. */
export function flightDuration(count, duration = FLIGHT_MS, stagger = STAGGER_MS) {
  if (!count || !animated()) return 0;
  return scaled(duration + (count - 1) * stagger + LAND_MS);
}
