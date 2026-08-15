/* Prove the JavaScript engine against the Python one, move by move.
 *
 * scripts/dump_fixtures.py plays seeded random games with ludometer/azul/engine.py
 * and records everything it saw. This replays each of those games here — same
 * deals (the recorded shuffles are fed in through ScriptedRng) and same moves —
 * and demands an exact match on:
 *
 *   1. legal_actions(), as an ordered list (order is part of the port, since the
 *      lookup tables it comes from are);
 *   2. to_json(), the whole snapshot, key for key;
 *   3. encode(), all 182 floats rounded to 6 decimals — the vector the net eats;
 *   4. the terminal state: scores, outcome, exhausted flag, round count.
 *
 * No framework: `node web/player/test/engine.test.mjs`, exit code 0 or 1.
 */
import { gunzipSync } from "node:zlib";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { AzulState, ScriptedRng, ENCODED_SIZE } from "../js/engine.js";

/* The random games cover the ordinary run of play; `cases` are hand-built
 * positions for the branches sampling never reaches (an all-monochrome round
 * end, a bag that runs dry, a score clamped at zero, the end-game bonuses). */

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = process.env.LUDOMETER_FIXTURES || join(HERE, "fixtures", "games.json.gz");

let failures = 0;
const problems = [];

function fail(where, detail) {
  failures += 1;
  if (problems.length < 12) problems.push(`${where}: ${detail}`);
}

/** Deterministic key-sorted JSON, so object key order never fakes a mismatch. */
function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    const keys = Object.keys(value).sort();
    return `{${keys.map((k) => `${JSON.stringify(k)}:${canonical(value[k])}`).join(",")}}`;
  }
  if (typeof value === "number" && Object.is(value, -0)) return "0";
  return JSON.stringify(value === undefined ? null : value);
}

const round6 = (x) => {
  const r = Math.round(x * 1e6) / 1e6;
  return Object.is(r, -0) ? 0 : r;
};

function checkEncoding(where, state, expected) {
  const enc = state.encode();
  if (enc.length !== ENCODED_SIZE || expected.length !== ENCODED_SIZE) {
    fail(where, `encoding length ${enc.length} vs ${expected.length}`);
    return;
  }
  for (let i = 0; i < ENCODED_SIZE; i++) {
    const got = round6(enc[i]);
    if (got !== expected[i]) {
      fail(where, `encode()[${i}] = ${got}, python had ${expected[i]}`);
      return;
    }
  }
}

function checkState(where, state, expected) {
  const got = canonical(state.toJSON());
  const want = canonical(expected);
  if (got === want) return;
  // narrow it down to the first differing top-level key, so the report is useful
  const mine = state.toJSON();
  for (const key of Object.keys(expected)) {
    if (canonical(mine[key]) !== canonical(expected[key])) {
      fail(where, `to_json().${key}: js ${canonical(mine[key])} vs py ${canonical(expected[key])}`);
      return;
    }
  }
  fail(where, "to_json() differs but every key matched (extra key on the JS side?)");
}

function replay(game, label) {
  const rng = new ScriptedRng(game.shuffles);
  const state = game.setup ? AzulState.fromSetup(game.setup, rng) : AzulState.newGame(0, rng);
  let moveNo = 0;

  for (const move of game.moves) {
    const where = `${label} move ${moveNo}`;
    checkState(where, state, move.state);
    const legal = state.legalActions();
    if (canonical(legal) !== canonical(move.legal)) {
      fail(where, `legal_actions(): js ${legal.length} ids, py ${move.legal.length} — ${canonical(legal)} vs ${canonical(move.legal)}`);
      return moveNo;
    }
    if (!state.isLegal(move.action)) fail(where, `isLegal(${move.action}) is false but python played it`);
    checkEncoding(where, state, move.enc);
    if (canonical(state.scores) !== canonical(move.scores)) {
      fail(where, `scores ${canonical(state.scores)} vs ${canonical(move.scores)}`);
    }
    state.apply(move.action);
    moveNo += 1;
  }

  const where = `${label} final`;
  checkState(where, state, game.final_state);
  checkEncoding(where, state, game.final_enc);
  if (canonical(state.scores) !== canonical(game.scores)) {
    fail(where, `final scores ${canonical(state.scores)} vs ${canonical(game.scores)}`);
  }
  if (state.outcome() !== game.outcome) fail(where, `outcome ${state.outcome()} vs ${game.outcome}`);
  if (state.isTerminal !== game.is_terminal) fail(where, `is_terminal ${state.isTerminal} vs ${game.is_terminal}`);
  if (state.exhausted !== game.exhausted) fail(where, `exhausted ${state.exhausted} vs ${game.exhausted}`);
  if (state.roundIndex + 1 !== game.rounds) fail(where, `rounds ${state.roundIndex + 1} vs ${game.rounds}`);
  return moveNo;
}

function main() {
  let payload;
  try {
    payload = JSON.parse(gunzipSync(readFileSync(FIXTURES)).toString("utf8"));
  } catch (err) {
    console.error(`cannot read fixtures at ${FIXTURES}: ${err.message}`);
    console.error("regenerate them with: nice -n 15 uv run python scripts/dump_fixtures.py");
    return 1;
  }

  const started = Date.now();
  let moves = 0;
  payload.games.forEach((game, i) => {
    moves += replay(game, `game ${i} (seed ${game.seed})`);
  });
  let caseMoves = 0;
  const cases = payload.cases || [];
  cases.forEach((c) => {
    caseMoves += replay(c, `case ${c.name}`);
  });
  const seconds = (Date.now() - started) / 1000;

  const expected = payload.totals || {};
  if (expected.moves && moves !== expected.moves) {
    fail("totals", `replayed ${moves} moves, fixtures hold ${expected.moves}`);
  }
  if (expected.case_moves && caseMoves !== expected.case_moves) {
    fail("totals", `replayed ${caseMoves} case moves, fixtures hold ${expected.case_moves}`);
  }

  if (failures) {
    console.error(`FAIL — ${failures} mismatch(es) across ${payload.games.length} games + ${cases.length} cases`);
    problems.forEach((p) => console.error("  " + p));
    return 1;
  }
  const total = moves + caseMoves;
  console.log(
    `engine.js matches ludometer/azul/engine.py: ${payload.games.length} random games (${moves} moves) + ` +
      `${cases.length} handcrafted cases (${caseMoves} moves), ${total * 4} assertions ` +
      `(state / legal / encoding / scores) in ${seconds.toFixed(1)}s`
  );
  cases.forEach((c) => console.log(`  case ${c.name}: ok (${c.moves.length} moves, scores ${c.scores.join("–")})`));
  return 0;
}

process.exit(main());
