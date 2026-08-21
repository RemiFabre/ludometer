/* Plain-language descriptions of what the engine just did — a port of
 * ludometer/gui/moves.py.
 *
 * `apply()` mutates in place and resolves the round boundary itself, so nothing
 * in the engine reports *which* tiles moved to the wall or what each one earned.
 * This rebuilds that story from a snapshot taken **before** the move, using
 * exactly the engine's rules — the same reconstruction the Python GUI server did,
 * moved into the page now that there is no server.
 */

import {
  CENTER,
  COL_BONUS,
  COLOR_BONUS,
  COLOR_NAMES,
  CUM_PENALTY,
  FLOOR,
  FLOOR_PENALTIES,
  FLOOR_SLOTS,
  NUM_COLORS,
  NUM_ROWS,
  ROW_BONUS,
  WALL_IDX,
  decodeAction,
} from "./engine.js";

export const sourceLabel = (source) => (source === CENTER ? "the center" : `factory ${source + 1}`);
export const destLabel = (dest) => (dest === FLOOR ? "the floor line" : `row ${dest + 1}`);

/** Describe `actionId` as it would play out in `state` (no mutation). */
export function describeAction(state, actionId) {
  const [source, color, dest] = decodeAction(actionId);
  const player = state.currentPlayer;
  const pool = source === CENTER ? state.center : state.factories[source];
  const count = pool[color];
  const tookMarker = source === CENTER && state.markerInCenter;

  let placed = 0;
  let overflow = count;
  if (dest !== FLOOR) {
    const room = dest + 1 - state.plCount[player][dest];
    placed = Math.min(count, room);
    overflow = count - placed;
  }
  const occupied = state.floorOccupied(player) + (tookMarker ? 1 : 0);
  const toFloor = Math.max(0, Math.min(overflow, FLOOR_SLOTS - occupied));

  let text = `took ${count} ${COLOR_NAMES[color]} from ${sourceLabel(source)} → ${destLabel(dest)}`;
  if (dest !== FLOOR && overflow) text += ` (${overflow} to the floor)`;
  if (tookMarker) text += " + first-player marker";

  return {
    action_id: actionId,
    player,
    source,
    color,
    dest,
    count,
    placed,
    overflow,
    to_floor: toFloor,
    to_lid: overflow - toFloor,
    took_marker: tookMarker,
    color_name: COLOR_NAMES[color],
    source_label: sourceLabel(source),
    dest_label: destLabel(dest),
    text,
  };
}

function runLengths(wall, row, col) {
  const base = row * 5;
  let h = 1;
  for (let i = col - 1; i >= 0 && wall[base + i]; i--) h += 1;
  for (let i = col + 1; i < 5 && wall[base + i]; i++) h += 1;
  let v = 1;
  for (let i = row - 1; i >= 0 && wall[i * 5 + col]; i--) v += 1;
  for (let i = row + 1; i < 5 && wall[i * 5 + col]; i++) v += 1;
  return [h, v];
}

/** Replay the engine's wall-tiling for one player on a snapshot. */
function tilePlayer(before, player) {
  const wall = before.walls[player].slice();
  const colors = before.plColor[player];
  const counts = before.plCount[player];
  const tiles = [];
  let gain = 0;
  for (let row = 0; row < NUM_ROWS; row++) {
    if (counts[row] !== row + 1) continue;
    const color = colors[row];
    const idx = WALL_IDX[color * 5 + row];
    wall[idx] = 1;
    const col = idx - row * 5;
    const [h, v] = runLengths(wall, row, col);
    const points = h > 1 || v > 1 ? (h > 1 ? h : 0) + (v > 1 ? v : 0) : 1;
    gain += points;
    tiles.push({ row, col, color, color_name: COLOR_NAMES[color], points, h_run: h, v_run: v, discarded: row });
  }

  const floor = before.floor[player];
  const floorTiles = [];
  for (let c = 0; c < NUM_COLORS; c++) for (let k = 0; k < floor[c]; k++) floorTiles.push(c);
  const occupied = before.floorOccupied(player);
  const penalty = CUM_PENALTY[Math.min(FLOOR_SLOTS, occupied)];
  const scoreBefore = before.scores[player];
  const scoreAfter = Math.max(0, scoreBefore + gain + penalty);
  const carried = [];
  for (let r = 0; r < NUM_ROWS; r++) {
    if (counts[r] && counts[r] !== r + 1) carried.push({ row: r, color: colors[r], count: counts[r] });
  }
  return {
    seat: player,
    tiles,
    tiling_points: gain,
    carried_rows: carried,
    floor: {
      tiles: floorTiles,
      marker: before.floorMarker[player],
      occupied,
      penalty,
      slot_penalties: FLOOR_PENALTIES.slice(0, Math.min(FLOOR_SLOTS, occupied)),
    },
    score_before: scoreBefore,
    score_after: scoreAfter,
    delta: scoreAfter - scoreBefore,
  };
}

/**
 * `before` with `move`'s tiles placed — the state the engine tiles from.
 *
 * `apply` places the tiles *and* resolves the round in one step, so the position
 * wall-tiling actually starts from is never observable. Only the mover's own
 * pattern lines and floor change, and `describeAction` already worked out how the
 * tiles split, so replaying that much is enough.
 */
export function afterPlacement(before, move) {
  const state = before.clone();
  const player = move.player;
  if (move.took_marker) state.floorMarker[player] = true;
  const dest = move.dest;
  if (dest !== FLOOR) {
    state.plColor[player][dest] = move.color;
    state.plCount[player][dest] += move.placed;
  }
  if (move.to_floor) state.floor[player][move.color] += move.to_floor;
  return state;
}

/** Wall-tiling result of the round ended by `move` played in `before`. */
export function roundReport(before, move) {
  const state = afterPlacement(before, move);
  return {
    round: state.roundIndex,
    players: [tilePlayer(state, 0), tilePlayer(state, 1)],
  };
}

function bonusBreakdown(state, player) {
  const rows = state.completedRows(player);
  const cols = state.completedCols(player);
  const colors = state.completedColors(player);
  const rowPts = ROW_BONUS * rows;
  const colPts = COL_BONUS * cols;
  const colorPts = COLOR_BONUS * colors;
  const total = rowPts + colPts + colorPts;
  return {
    seat: player,
    rows,
    row_points: rowPts,
    cols,
    col_points: colPts,
    colors,
    color_points: colorPts,
    total,
    score_before_bonus: state.scores[player] - total,
    final_score: state.scores[player],
  };
}

/** End-of-game summary: bonuses per player, winner, why the game stopped. */
export function finalReport(state, humanSeat, opponent = "The AI") {
  if (!state.isTerminal) return null;
  const outcome = state.outcome() || 0;
  const winner = outcome === 0 ? null : outcome > 0 ? 0 : 1;
  let headline;
  if (winner === null) headline = "A draw: same score, same completed rows.";
  else if (winner === humanSeat) headline = "You win!";
  else headline = opponent + " wins.";
  return {
    winner,
    winner_side: winner === null ? null : winner === humanSeat ? "human" : "ai",
    headline,
    scores: state.scores.slice(),
    bonuses: [bonusBreakdown(state, 0), bonusBreakdown(state, 1)],
    exhausted: state.exhausted,
    rounds_played: state.roundIndex + 1,
  };
}
