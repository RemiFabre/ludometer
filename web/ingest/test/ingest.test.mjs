/* The collector, proven end to end without touching Hugging Face.
 *
 * Real games are generated with the real engine (random legal moves), written
 * down exactly the way web/player/js/record.js writes them, and POSTed to a
 * live server whose dataset store is a fake that captures commits. Then the
 * gate is rattled: duplicates, tampered scores, tampered deals, garbage, and
 * oversized bodies must all bounce while the counters say so.
 *
 *   node web/ingest/test/ingest.test.mjs
 */
import { AzulState, Rng } from "../engine.js";
import { verifyRecord } from "../verify.js";
import { createIngestServer } from "../server.js";

/** A whole game by random legal moves, recorded the way the page records it. */
function playGame(seed, { truncate = null } = {}) {
  const state = AzulState.newGame(seed, new Rng(seed));
  const picker = new Rng(seed ^ 0x9e3779b9);
  const moves = [];
  const deals = [dealOf(state)];
  while (!state.isTerminal && moves.length < 200) {
    if (truncate !== null && moves.length >= truncate) break;
    const legal = state.legalActions();
    const action = legal[picker.randrange(legal.length)];
    moves.push({ ply: moves.length + 1, player: state.currentPlayer, action });
    const round = state.roundIndex;
    state.apply(action);
    if (state.roundIndex !== round && !state.isTerminal) deals.push(dealOf(state));
  }
  return {
    format: "faience-game/1",
    created_at: new Date().toISOString(),
    seed,
    human_seat: 0,
    human_first: true,
    net: { run: "run4", checkpoint: "ckpt-test", elo: 2361, params: 1000, backend: "test" },
    think_time_s: 0,
    moves,
    deals,
    final: {
      finished: !!state.isTerminal,
      scores: state.scores.slice(),
      outcome: null,
      rounds: state.roundIndex + 1,
      exhausted: !!state.exhausted,
    },
  };
}

function dealOf(state) {
  return {
    round: state.roundIndex,
    factories: state.factories.map((f) => f.slice()),
    bag: state.bagCounts(),
    lid: state.lid.slice(),
  };
}

const errors = [];
const check = (label, got, want) => {
  const same = JSON.stringify(got) === JSON.stringify(want);
  if (!same) errors.push(`${label}: got ${JSON.stringify(got)}, wanted ${JSON.stringify(want)}`);
  console.log(`  ${same ? "ok" : "FAIL"}  ${label}`);
};

// ---- the verifier alone -------------------------------------------------------
{
  const good = playGame(1234);
  check("a real finished game verifies", verifyRecord(good).ok, true);
  check("canonical outcome is derived", verifyRecord(good).record.final.outcome !== null, true);

  const abandoned = playGame(1234, { truncate: 10 });
  abandoned.final.finished = false;
  check("an abandoned game verifies, flagged unfinished", verifyRecord(abandoned).ok, true);
  check(
    "unfinished stays unfinished in the canonical record",
    verifyRecord(abandoned).record.final.finished,
    false
  );

  const inflated = playGame(1234);
  inflated.final.scores[0] += 1;
  check("a tampered score is rejected", verifyRecord(inflated).reason, "scores do not replay");

  const cooked = playGame(1234);
  cooked.deals[0].factories[0][0] = (cooked.deals[0].factories[0][0] + 1) % 5;
  check("a tampered deal is rejected", verifyRecord(cooked).reason, "first deal does not match");

  const liar = playGame(1234, { truncate: 10 });
  liar.final.finished = false;
  liar.final.scores = [99, 0];
  check("a lied mid-game score is rejected", verifyRecord(liar).reason, "scores do not replay");

  const wrongSeat = playGame(1234);
  wrongSeat.moves[3].player = 1 - wrongSeat.moves[3].player;
  check("a wrong mover is rejected", verifyRecord(wrongSeat).reason, "wrong player to move");

  const smuggler = playGame(1234);
  smuggler.contraband = "x".repeat(1000);
  smuggler.net.run = "run4";
  const rebuilt = verifyRecord(smuggler);
  check("unknown fields do not survive the rebuild", "contraband" in rebuilt.record, false);
}

// ---- the server ---------------------------------------------------------------
const saves = [];
let failNextSave = false;
const store = {
  repo: "fake/faience-games",
  async save(path, lines) {
    if (failNextSave) {
      failNextSave = false;
      throw new Error("simulated outage");
    }
    saves.push({ path, lines });
  },
};

const server = await createIngestServer({ port: 0, store, batchMax: 3, adminToken: "secret" });
const url = `http://127.0.0.1:${server.port}`;
const post = (body) =>
  fetch(`${url}/game`, { method: "POST", body }).then((r) => r.status);

{
  check("a good game answers 204", await post(JSON.stringify(playGame(42))), 204);
  check("a duplicate answers 204 too", await post(JSON.stringify(playGame(42))), 204);
  check("garbage answers 204 (the page never reads replies)", await post("]{not json"), 204);
  check("a tampered game answers 204", await post(JSON.stringify({ ...playGame(43), final: { finished: true, scores: [999, 0], rounds: 1 } })), 204);

  let stats = await fetch(`${url}/stats`).then((r) => r.json());
  check("accepted", stats.accepted, 1);
  check("duplicates", stats.duplicates, 1);
  check("rejected", stats.rejected, 2);
  check("pending", stats.pending, 1);

  // batchMax = 3: two more games trip an automatic flush
  await post(JSON.stringify(playGame(44)));
  await post(JSON.stringify(playGame(45)));
  await server.flush();
  stats = await fetch(`${url}/stats`).then((r) => r.json());
  check("a full batch was committed", stats.committed, 3);
  check("the committed shard holds 3 lines", saves[0].lines.length, 3);
  check("the shard path is dated", /^games\/\d{4}-\d{2}-\d{2}\//.test(saves[0].path), true);
  const entry = JSON.parse(saves[0].lines[0]);
  check("stored entries carry received_at", typeof entry.received_at, "string");
  check("stored entries are canonical records", entry.format, "faience-game/1");

  // a failed commit keeps the batch for the next try
  failNextSave = true;
  await post(JSON.stringify(playGame(46)));
  await server.flush();
  stats = await fetch(`${url}/stats`).then((r) => r.json());
  check("a failed commit keeps the game pending", stats.pending, 1);
  check("the failure is counted", stats.commit_failures, 1);
  await server.flush();
  stats = await fetch(`${url}/stats`).then((r) => r.json());
  check("the retry drains it", stats.pending, 0);

  // an oversized body is cut off, not buffered
  const big = await post("x".repeat(80 * 1024)).catch(() => "destroyed");
  check("an oversized body is destroyed", big, "destroyed");

  // /flush is gated
  check("flush without the token is refused", (await fetch(`${url}/flush`, { method: "POST" })).status, 403);
  check(
    "flush with the token works",
    (await fetch(`${url}/flush`, { method: "POST", headers: { Authorization: "Bearer secret" } })).status,
    200
  );
}

server.close();
if (errors.length) {
  console.error(`\n${errors.length} failure(s)`);
  process.exit(1);
}
console.log("\nall ingest checks passed");
