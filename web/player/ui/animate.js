/* Straight-line tile flights.
 *
 * Every movement on this table is the same gesture: a tile leaves one place and
 * travels in a straight line to another. Four things move — tiles from a dish to
 * a board, a dish's leftovers to the middle, a full pattern line to the wall, and
 * the floor line to the lid — and all four are the same call.
 *
 * A flight is a clone in a fixed-position layer: the real tiles are hidden for
 * the duration, and the page re-renders from the new state when the group lands.
 * Nothing here knows what a move *means*; callers pass elements (or rectangles)
 * and a colour.
 */

export const FLIGHT_MS = 460; // one group of tiles, door to door
export const STAGGER_MS = 45; // ... plus this per tile after the first
export const LAND_MS = 90; // a beat to let the last tile settle

export const prefersReducedMotion = () =>
  !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);

export const sleep = (ms) =>
  new Promise((done) => setTimeout(done, prefersReducedMotion() ? 0 : ms));

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
 * `options`: `{layer, duration, stagger, scale}`. Returns a promise that always
 * resolves (never rejects) so a caller can `await` it in a turn sequence.
 */
export function flyTiles(flights, options = {}) {
  const layer = options.layer;
  const duration = options.duration === undefined ? FLIGHT_MS : options.duration;
  const stagger = options.stagger === undefined ? STAGGER_MS : options.stagger;
  if (!layer || !flights || !flights.length || prefersReducedMotion()) {
    return Promise.resolve();
  }

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
  const total = duration + (airborne.length - 1) * stagger + LAND_MS;
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
  if (!count || prefersReducedMotion()) return 0;
  return duration + (count - 1) * stagger + LAND_MS;
}
