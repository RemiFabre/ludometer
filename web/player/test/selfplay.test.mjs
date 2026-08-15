/* Play whole games JS-vs-JS with the real net, under node.
 *
 * The fixture test proves the rules; this proves the *stack*: engine + MCTS +
 * the exported ONNX graph through onnxruntime-web, driving complete games to a
 * terminal state. It asserts, every ply, that the action the search returned was
 * in `legalActions()` — a search that quietly plays an illegal move would still
 * "work" from the outside, so this is the check worth having — and that each game
 * actually terminates rather than looping forever.
 *
 * It also reports positions/second, which is the number the page's thinking-time
 * selector is really spending.
 *
 *   node web/player/test/selfplay.test.mjs [--games 2] [--budget 0.5]
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { AzulState, Rng, ACTION_SPACE } from "../js/engine.js";
import { MCTSAgent, MAX_GAME_MOVES } from "../js/mcts.js";
import { OnnxEvaluator } from "../js/net.js";
import * as ort from "../vendor/onnxruntime-web/ort.wasm.bundle.min.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..");

function arg(name, fallback) {
  const i = process.argv.indexOf("--" + name);
  return i >= 0 && process.argv[i + 1] !== undefined ? Number(process.argv[i + 1]) : fallback;
}

const GAMES = arg("games", 2);
const BUDGET = arg("budget", 0.5);

async function main() {
  ort.env.wasm.numThreads = 1;
  // node's fetch cannot read file:// URLs, so hand the runtime the bytes
  // directly; the page instead points wasmPaths at the served .wasm.
  ort.env.wasm.wasmBinary = readFileSync(join(ROOT, "vendor", "onnxruntime-web", "ort-wasm-simd-threaded.wasm")).buffer;
  ort.env.logLevel = "error";

  const meta = JSON.parse(readFileSync(join(ROOT, "model", "model_meta.json"), "utf8"));
  const bytes = readFileSync(join(ROOT, "model", "model.onnx"));
  const evaluator = await OnnxEvaluator.create(ort, bytes);
  console.log(`model ${meta.run}/${meta.checkpoint} (elo ${meta.elo}, ${meta.num_params.toLocaleString()} params)`);

  let failures = 0;
  let totalSims = 0;
  let totalSearchS = 0;
  let totalMoves = 0;

  for (let g = 0; g < GAMES; g++) {
    const state = AzulState.newGame(1000 + g, new Rng(1000 + g));
    const agents = [
      new MCTSAgent(evaluator, {}, new Rng(11 + g)),
      new MCTSAgent(evaluator, {}, new Rng(99 + g)),
    ];
    let moves = 0;
    while (!state.isTerminal && moves < MAX_GAME_MOVES) {
      const agent = agents[state.currentPlayer];
      const legal = state.legalActions();
      const action = await agent.act(state, { timeLimitS: BUDGET });
      if (!Number.isInteger(action) || action < 0 || action >= ACTION_SPACE) {
        console.error(`game ${g} ply ${moves}: action ${action} out of range`);
        failures += 1;
        break;
      }
      if (!legal.includes(action)) {
        console.error(`game ${g} ply ${moves}: search returned illegal action ${action}`);
        failures += 1;
        break;
      }
      if (!state.isLegal(action)) {
        console.error(`game ${g} ply ${moves}: isLegal(${action}) disagrees with legalActions()`);
        failures += 1;
        break;
      }
      if (agent.lastSearch.sims) {
        totalSims += agent.lastSearch.sims;
        totalSearchS += agent.lastSearch.elapsedS;
      }
      state.apply(action);
      moves += 1;
    }
    totalMoves += moves;
    const census = [0, 0, 0, 0, 0];
    for (const c of state.bag) census[c] += 1;
    for (let c = 0; c < 5; c++) {
      census[c] += state.lid[c] + state.center[c] + state.floor[0][c] + state.floor[1][c];
      for (const f of state.factories) census[c] += f[c];
    }
    for (let p = 0; p < 2; p++) {
      for (let r = 0; r < 5; r++) if (state.plCount[p][r]) census[state.plColor[p][r]] += state.plCount[p][r];
      for (let r = 0; r < 5; r++) {
        for (let col = 0; col < 5; col++) if (state.walls[p][r * 5 + col]) census[(col - r + 5) % 5] += 1;
      }
    }
    const lost = census.some((n) => n !== 20);
    if (lost) {
      console.error(`game ${g}: tiles went missing — census ${census}`);
      failures += 1;
    }
    if (!state.isTerminal) {
      console.error(`game ${g}: did not terminate in ${MAX_GAME_MOVES} moves`);
      failures += 1;
    }
    console.log(
      `game ${g}: ${moves} plies, scores ${state.scores[0]}–${state.scores[1]}, ` +
        `outcome ${state.outcome()}, rounds ${state.roundIndex + 1}, census ok=${!lost}`
    );
  }

  const rate = totalSearchS > 0 ? totalSims / totalSearchS : 0;
  console.log(
    `\n${GAMES} games, ${totalMoves} plies, ${totalSims.toLocaleString()} simulations in ` +
      `${totalSearchS.toFixed(1)}s of search — ${rate.toFixed(0)} positions/s ` +
      `(node, wasm, ${BUDGET}s budget per move)`
  );
  if (failures) {
    console.error(`FAIL — ${failures} problem(s)`);
    return 1;
  }
  console.log("selfplay ok: every move legal, every game terminated, no tiles lost");
  return 0;
}

process.exit(await main());
