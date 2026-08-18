/* Does a submitted game actually replay? That question is the whole defence.
 *
 * A faience-game/1 record is deterministic: the seed reproduces every shuffle
 * (mulberry32, the page's own RNG), so replaying the recorded actions in the
 * real engine must reproduce the recorded deals, the recorded scores and the
 * recorded round count exactly. Spam, corruption and mischief all fail that
 * check, which is a far better filter than any secret token in client-side
 * JavaScript could be (docs/HUGGINGFACE.md §5).
 *
 * `verifyRecord` returns `{ ok: false, reason }` or `{ ok: true, record }`
 * where `record` is a CANONICAL rebuild: only known fields, every number
 * coerced, every string bounded. What gets stored is that rebuild, never the
 * raw submission, so nothing can ride into the dataset inside an unexpected
 * key or an over-long string.
 */

import { ACTION_SPACE, AzulState, Rng } from "./engine.js";

export const FORMAT = "faience-game/1";
export const MAX_MOVES = 200; // matches MAX_GAME_MOVES in the player's mcts.js
export const MAX_ROUNDS = 20; // a real 2-player game ends in 5 to 8

const isInt = Number.isInteger;

function str(value, max) {
  return typeof value === "string" && value.length <= max ? value : null;
}

function num(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function intArray(value, length, max) {
  if (!Array.isArray(value) || value.length !== length) return null;
  if (!value.every((n) => isInt(n) && n >= 0 && n <= max)) return null;
  return value.slice();
}

function sameInts(a, b) {
  return a.length === b.length && a.every((n, i) => n === b[i]);
}

/** The deal the engine just produced, in the record's own shape. */
function dealNow(state) {
  return {
    round: state.roundIndex,
    factories: state.factories.map((f) => f.slice()),
    bag: state.bagCounts(),
    lid: state.lid.slice(),
  };
}

function dealMatches(kept, real) {
  if (!kept || kept.round !== real.round) return false;
  if (kept.factories.length !== real.factories.length) return false;
  for (let i = 0; i < real.factories.length; i++) {
    if (!sameInts(kept.factories[i], real.factories[i])) return false;
  }
  return sameInts(kept.bag, real.bag) && sameInts(kept.lid, real.lid);
}

export function verifyRecord(raw) {
  const fail = (reason) => ({ ok: false, reason });
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return fail("not an object");
  if (raw.format !== FORMAT) return fail("unknown format");

  // ---- shape, before any replay work is spent on it -------------------------
  const seed = raw.seed;
  if (!isInt(seed) || seed < 0 || seed > 0xffffffff) return fail("bad seed");
  const humanSeat = raw.human_seat;
  if (humanSeat !== 0 && humanSeat !== 1) return fail("bad human_seat");

  if (!Array.isArray(raw.moves) || raw.moves.length < 1 || raw.moves.length > MAX_MOVES) {
    return fail("bad moves");
  }
  const moves = [];
  for (let i = 0; i < raw.moves.length; i++) {
    const m = raw.moves[i];
    if (!m || typeof m !== "object") return fail("bad move");
    if (!isInt(m.action) || m.action < 0 || m.action >= ACTION_SPACE) return fail("bad action id");
    if (m.player !== 0 && m.player !== 1) return fail("bad move player");
    if (m.ply !== i + 1) return fail("bad ply sequence");
    const move = { ply: m.ply, player: m.player, action: m.action };
    if (isInt(m.sims) && m.sims > 0 && m.sims <= 1e9) move.sims = m.sims;
    const value = num(m.value);
    if (value !== null && value >= -1 && value <= 1) move.value = value;
    moves.push(move);
  }

  if (!Array.isArray(raw.deals) || raw.deals.length < 1 || raw.deals.length > MAX_ROUNDS) {
    return fail("bad deals");
  }
  const deals = [];
  for (const d of raw.deals) {
    if (!d || typeof d !== "object") return fail("bad deal");
    if (!isInt(d.round) || d.round < 0 || d.round >= MAX_ROUNDS) return fail("bad deal round");
    if (!Array.isArray(d.factories)) return fail("bad deal factories");
    const factories = d.factories.map((f) => intArray(f, 5, 20));
    if (factories.some((f) => f === null)) return fail("bad deal factories");
    const bag = intArray(d.bag, 5, 20);
    const lid = intArray(d.lid, 5, 20);
    if (!bag || !lid) return fail("bad deal counts");
    deals.push({ round: d.round, factories, bag, lid });
  }

  const final = raw.final;
  if (!final || typeof final !== "object") return fail("bad final");
  const finished = !!final.finished;
  if (!Array.isArray(final.scores) || final.scores.length !== 2) return fail("bad scores");
  if (!final.scores.every((s) => isInt(s) && s >= -50 && s <= 400)) return fail("bad scores");

  // ---- the replay -----------------------------------------------------------
  const state = AzulState.newGame(seed, new Rng(seed));
  let dealsSeen = 0;
  const checkDeal = () => {
    const real = dealNow(state);
    const kept = deals[dealsSeen];
    dealsSeen += 1;
    return kept && dealMatches(kept, real);
  };
  if (!checkDeal()) return fail("first deal does not match");
  for (const move of moves) {
    if (state.isTerminal) return fail("moves continue past the end");
    if (move.player !== state.currentPlayer) return fail("wrong player to move");
    if (!state.isLegal(move.action)) return fail("illegal move");
    const round = state.roundIndex;
    state.apply(move.action);
    if (state.roundIndex !== round && !state.isTerminal) {
      if (!checkDeal()) return fail("a redeal does not match");
    }
  }
  if (dealsSeen !== deals.length) return fail("extra deals recorded");
  if (finished !== state.isTerminal) {
    return fail(finished ? "claims finished, replay is not" : "claims unfinished, replay is over");
  }
  if (!sameInts(final.scores, state.scores)) return fail("scores do not replay");
  if (final.rounds !== state.roundIndex + 1) return fail("round count does not replay");

  // ---- the canonical rebuild ------------------------------------------------
  const net = raw.net && typeof raw.net === "object" ? raw.net : {};
  const elo = num(net.elo);
  const params = isInt(net.params) && net.params > 0 ? net.params : null;
  const thinkTime = num(raw.think_time_s);
  const createdAt = str(raw.created_at, 40);
  const record = {
    format: FORMAT,
    created_at: createdAt && !Number.isNaN(Date.parse(createdAt)) ? createdAt : null,
    seed,
    human_seat: humanSeat,
    human_first: humanSeat === 0,
    net: {
      run: str(net.run, 64),
      checkpoint: str(net.checkpoint, 64),
      elo: elo !== null && Math.abs(elo) < 1e5 ? elo : null,
      params,
      backend: str(net.backend, 32),
    },
    think_time_s: thinkTime !== null && thinkTime >= 0 && thinkTime <= 600 ? thinkTime : null,
    moves,
    deals,
    final: {
      finished,
      scores: final.scores.slice(),
      outcome: finished ? outcomeOf(final.scores, humanSeat) : null,
      rounds: state.roundIndex + 1,
      exhausted: !!state.exhausted,
    },
  };
  return { ok: true, record };
}

function outcomeOf(scores, humanSeat) {
  const mine = scores[humanSeat];
  const theirs = scores[1 - humanSeat];
  if (mine > theirs) return "human";
  if (theirs > mine) return "ai";
  return "draw";
}

/** What makes two submissions "the same game": the content, not the clock. */
export function recordKey(record) {
  return JSON.stringify([
    record.seed,
    record.human_seat,
    record.moves.map((m) => m.action),
    record.final.scores,
    record.final.finished,
  ]);
}
