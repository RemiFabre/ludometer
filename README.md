# Ludometer

**Does a good game teach linearly?**

Ludometer is a reinforcement-learning framework for measuring the *quality* of a board
game through the shape of an AI agent's learning curve.

## The hypothesis

- If an agent learns **too slowly at the start**, the rules are probably hard to grasp.
- If it learns **too fast / plateaus sharply**, the game probably lacks depth.
- A *good* game should produce a roughly **linear Elo progression** in a learning agent.

We calibrate the method on **Azul** (Michael Kiesling, Spiel des Jahres 2018) — a game
widely considered excellent — then reuse the framework to evaluate new board game designs.

## Components

- `ludometer/azul/` — full 2-player Azul rules engine (fast, deterministic, fully tested)
- `ludometer/agents/` — baseline agents (random, greedy, heuristic) and the neural agent
- `ludometer/train/` — AlphaZero-style self-play training (PyTorch, Apple MPS)
- `ludometer/eval/` — arena + Elo rating of checkpoints against a fixed anchor pool
- `ludometer/export/` — checkpoint → ONNX, for the browser player
- `web/` — training dashboard, a local GUI to play against the AI, and `web/player/`:
  the whole thing (rules, search, net) reimplemented in JavaScript and published to
  GitHub Pages

## Play in your browser (no install)

### **→ [remifabre.github.io/ludometer](https://remifabre.github.io/ludometer/) ←**

The strongest trained net, playable by anyone, with **nothing running on a server**.
Open the page and the tab downloads the exported net once (~16 MB over the wire, cached
afterwards), then plays Azul against you entirely on your own machine — the same PUCT tree
search the trainer uses, running in a Web Worker on top of onnxruntime-web (WASM). No
account, no upload, works offline once loaded, and fine on a phone.

The header names which checkpoint you are facing and its internal Elo; **AI thinks for**
(instant / 3 / 5 / 10 s) is the same strength dial as the local GUI, and the table talk
reports the true count every move — *“searched 16,384 positions in 5.0s”*. On an M-series
Mac that is **~3,300 positions/second**, i.e. **~16,000 per move at the default 5 s**; a
slower laptop or a phone will do less, and the page will tell you exactly how much.

How it is put together, and why you can trust it:

| piece | where | how it is checked |
|---|---|---|
| ONNX export of the best checkpoint | `ludometer/export/onnx_export.py` | 100 real positions through torch **and** onnxruntime, max abs diff < 1e-4 (`tests/test_export_onnx.py`) |
| Azul rules, ported to JS | `web/player/js/engine.js` | 30 seeded Python games + 5 handcrafted edge positions replayed move by move — states, legal-action lists, all 182 encoding floats and outcomes must match exactly (`web/player/test/engine.test.mjs`) |
| PUCT search, ported to JS | `web/player/js/mcts.js` | full JS-vs-JS games under node: every move legal, every game terminates, no tiles lost (`web/player/test/selfplay.test.mjs`) |
| the page itself | `web/player/` | driven in headless Chrome: no console errors, a real human/AI exchange, no sideways scroll at 390 px (`web/player/test/browser.test.mjs`) |

Run those four locally with:

```bash
uv run --group export python -m ludometer.export.onnx_export   # refresh model/
uv run --group export pytest tests/test_export_onnx.py
node web/player/test/engine.test.mjs
node web/player/test/parity.test.mjs
node web/player/test/selfplay.test.mjs --games 1 --budget 0.3
node web/player/test/browser.test.mjs                          # add --live for the deployed site
```

To publish a newer checkpoint (re-export, re-check, rebuild `gh-pages`, wait for the site):

```bash
./scripts/deploy_player.sh
```

The site is a copy of `web/player/` minus its `test/` directory. `deploy_player.sh` writes
the `gh-pages` branch through a temporary git index, so it never checks anything out and
never touches your working tree — safe to run while a training run is writing into `runs/`.

## Quick start

```bash
uv sync
uv run pytest              # engine tests
uv run ludometer-gui       # play Azul against the current best agent
```

## Play against the AI (locally, with the trainer's own agents)

```bash
uv sync
uv run ludometer-gui       # opens http://127.0.0.1:8737/ and deals a game
```

The **Opponent** dropdown defaults to **“Strongest trained (auto)”** — the agent spec
`best`. That spec scans every `runs/*/elo.jsonl`, takes the highest-Elo checkpoint whose
`.pt` file is still on disk, and plays it. The dropdown shows which one that is
(`ckpt-023040 · +1920 Elo · run run1`), and the table-talk panel repeats it once the game
starts: *“You’re facing ckpt-023040, rated +1920 on our internal ladder.”* The other
entries are the scripted baselines (heuristic / greedy / random) and
`Trained checkpoint…`, which takes a spec by hand
(`mcts:runs/run1/checkpoints/ckpt-020992.pt?sims=400`).

Resolution happens **when you press “Deal tiles”**, not when the page loads: while a run is
training the model keeps improving, so every new game automatically faces the newest
strongest checkpoint. Nothing to edit, no path to copy.

**AI thinks for** is the per-move time budget — the strength/pace dial. The search keeps
running simulations until the clock runs out instead of stopping at a fixed count:

| Budget | Positions searched per move | Feel |
|--------|-----------------------------|------|
| Instant reply | ~400 | replies as fast as it can, weakest setting |
| 3 seconds | ~4,000–5,000 | quick game, still well above its rating |
| **5 seconds** (default) | **~7,500–9,000** | you get time to read the position |
| 10 seconds | ~15,000–18,000 | strongest; it will punish sloppy floor lines |

(The 5 s row is measured on this Mac against run1's best checkpoint — ~7,700 positions with
a training run eating most of the cores — and the others scale from it. Whatever the
machine, the page reports the true count in the table talk: *“searched 7,712 positions in
5.0s”*.)

Elo ratings are measured at the trainer's own eval sims (100 per move), so at a 5 s budget
the checkpoint searches **50–100× more positions than its listed rating was measured at**
and plays meaningfully above it. The listed Elo is a floor, not a ceiling.

A turn now plays out in three beats: your move lands immediately, the AI visibly thinks for
its budget (with the clock running under the table), then its move is animated — the source
factory lights up and the tiles travel to its board before the position updates. Your input
is locked while that happens, and completed pattern lines slide into the wall at round end.

The board is laid out like the cardboard one: pattern line row *r* sits level with wall row
*r*, right-aligned, so you can see at a glance which wall square a line is feeding. Tile
colours match the base game (cobalt, ochre, terracotta, charcoal, ice), including the
ghosted diamonds on the empty wall squares.

Same thing without the browser (the GUI, the arena and the trainer share one registry):

```bash
uv run python -c "from ludometer.agents.registry import find_best_checkpoint; print(find_best_checkpoint())"
```

A time budget is available on any neural spec as `&think=<seconds>`, e.g.
`mcts:runs/run1/checkpoints/ckpt-024064.pt?sims=400&think=5`. It is off by default, so
training and the Elo ladder keep searching a fixed number of simulations per move.

## Status

Work in progress — built autonomously by Claude. Progress notes in `NOTES_FOR_REMI.md`.
