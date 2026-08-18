/* Shared primitives for the Azul board modules.
 *
 * Framework-free, no network, no globals beyond the DOM. Everything here is
 * pure presentation of the engine's own state JSON (`AzulState.to_json()`,
 * which the browser port produces too), so these modules drop into any page
 * that can hand them a state object.
 */

export const COLORS = ["blue", "yellow", "red", "black", "teal"];
export const CENTER = 5;
export const FLOOR = 5;
export const FLOOR_PENALTIES = [-1, -1, -2, -2, -2, -3, -3];
export const CUM_PENALTY = [0, -1, -2, -4, -6, -8, -11, -14];

/** `document.createElement` with a class and text in one call. */
export function node(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = text;
  return n;
}

/** A glazed tile of `color` (0..4). */
export function tileEl(color, cls) {
  const t = node("div", "tile" + (cls ? " " + cls : ""));
  t.dataset.color = color;
  return t;
}

/** The first-player marker: an unglazed chip, not a tile. */
export function markerChip(tiny) {
  const m = node("div", "marker" + (tiny ? " tiny" : ""), "1");
  m.title =
    "First-player marker: whoever takes it starts the next round and loses a floor slot.";
  return m;
}

/** The engine's action encoding — the one rule this layer is allowed to know. */
export function actionId(source, color, dest) {
  return source * 30 + color * 6 + dest;
}

export function decodeAction(id) {
  return { source: Math.floor(id / 30), color: Math.floor((id % 30) / 6), dest: id % 6 };
}

export function sourceLabel(source) {
  return source === CENTER ? "the middle" : "factory " + (source + 1);
}

export function destLabel(dest) {
  return dest === FLOOR ? "the floor line" : "row " + (dest + 1);
}

/** How many tiles of `color` are in `source` right now. */
export function poolCount(state, source, color) {
  return source === CENTER ? state.center[color] : state.factories[source][color];
}

/** Destinations that are legal for a held colour, from a legal-action set. */
export function destsFor(legal, source, color) {
  const out = [];
  for (let d = 0; d <= FLOOR; d++) {
    if (legal.has(actionId(source, color, d))) out.push(d);
  }
  return out;
}

/** What playing `source`/`color` into `dest` would cost this player. */
export function preview(state, seat, source, color, dest) {
  const me = state.players[seat];
  const count = poolCount(state, source, color);
  const takesMarker = source === CENTER && state.marker_in_center;
  let overflow = count;
  let placed = 0;
  if (dest !== FLOOR) {
    const line = me.pattern_lines[dest];
    placed = Math.min(count, line.capacity - line.count);
    overflow = count - placed;
  }
  const occupied = me.floor.reduce((a, b) => a + b, 0) + (me.floor_marker ? 1 : 0);
  const before = CUM_PENALTY[Math.min(7, occupied)];
  const after = CUM_PENALTY[Math.min(7, occupied + overflow + (takesMarker ? 1 : 0))];
  const line = dest !== FLOOR ? me.pattern_lines[dest] : null;
  return {
    count,
    placed,
    overflow,
    takesMarker,
    penalty: after - before,
    completes: line ? placed + line.count === line.capacity : false,
  };
}
