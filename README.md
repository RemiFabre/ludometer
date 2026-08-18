# Ludometer

**Does a good game teach linearly?**

Ludometer is a reinforcement-learning framework for measuring the *quality* of a board
game through the shape of an AI agent's learning curve.

## The hypothesis

- If an agent learns **too slowly at the start**, the rules are probably hard to grasp.
- If it learns **too fast / plateaus sharply**, the game probably lacks depth.
- A *good* game should produce a **nice Elo progression** in a learning agent.

What does nice mean in this context? A line? A log?

We calibrate the method on **Azul** (Michael Kiesling, Spiel des Jahres 2018) — a game
widely considered excellent — then reuse the framework to evaluate new board game designs.
**Uno** is the first contrast case (`configs/uno1.json`). Elo is only defined against a
game's own anchor pool, so the two are never compared as numbers: what carries across
games is the *shape* of the curve.

## ▶ Play it — and read how it works

- **Play against the AI in your browser, no install:**
  **[remifabre-faience.static.hf.space](https://remifabre-faience.static.hf.space/)** (details below;
  the old GitHub Pages address now points here)
- **How it all works**, explained for any engineer, every term defined:
  **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**

> **Note from the author** — This work has been done mostly autonomously by a coding
> agent. I have been guiding it and testing a lot, but I am far from having reviewed
> everything, so consider this an alpha version: early work.

## Components

- `ludometer/azul/` — full 2-player Azul rules engine (fast, deterministic, fully tested)
- `ludometer/uno/` — 2-player Uno (match to 500), the second game on the same rig; hidden
  hands are handled by determinizing at the search root. `ludometer/games.py` is the
  registry a config picks from with `"game": "azul" | "uno"`
- `ludometer/agents/` — baseline agents (random, greedy, heuristic) and the neural agent
- `ludometer/train/` — AlphaZero-style self-play training (PyTorch, Apple MPS)
- `ludometer/eval/` — arena + Elo rating of checkpoints against a fixed anchor pool
- `ludometer/export/` — checkpoint → ONNX, for the browser player
- `web/` — training dashboard, a local GUI to play against the AI, and `web/player/`:
  the whole thing (rules, search, net) reimplemented in JavaScript and published to a
  [Hugging Face Space](https://huggingface.co/spaces/RemiFabre/faience); `web/ingest/`
  is the small companion Space that collects shared games into the public dataset
  [faience-games](https://huggingface.co/datasets/RemiFabre/faience-games), and the old
  GitHub Pages address serves a moved notice (an unlinked emergency fallback stays
  deployed at `classic/`)

## Play in your browser (no install)

### **→ [remifabre-faience.static.hf.space](https://remifabre-faience.static.hf.space/) ←**

The public game is called **Faïence** — a free, open-source implementation of the rules
of Azul, with its own code and artwork. It lives on
[Hugging Face](https://huggingface.co/spaces/RemiFabre/faience); the original address,
[remifabre.github.io/ludometer](https://remifabre.github.io/ludometer/), shows a moved
notice with a button, so every link already shared keeps working. (The previous build
stays deployed at `classic/` as an unlinked emergency fallback; it is not advertised
because games played there are never recorded.) How many people play it, and how the games go,
is counted by an anonymous, cookie-free tally that is
**[public for anyone to read](https://faience.goatcounter.com)** — that link is also in
the game itself, so players can see exactly what is recorded (visits, games dealt, and
final results per net; nothing about anyone).

Faïence is a research project, and the games are the research material: when a game ends
(or is abandoned), the page sends an anonymous record — the moves, the tiles dealt, which
net played, and the score — to a tiny collector Space (`web/ingest/`) that **replays every
submission in the real engine** and keeps only games that reproduce their own deals and
final score. Verified games land in the public dataset
**[faience-games](https://huggingface.co/datasets/RemiFabre/faience-games)**, where they
become training data. Sharing is on by default and the switch is in the game's Settings;
the page says all of this in its About panel, in the same words. No IPs or user agents
are logged or stored by the collector, and everything it keeps is public.

The strongest trained net, playable by anyone, with **nothing running on a server**.
Open the page and the tab downloads the exported net once (~16 MB over the wire, cached
afterwards), then plays Azul against you entirely on your own machine — the same PUCT tree
search the trainer uses, running in a Web Worker on top of onnxruntime-web (WASM). No
account, no upload of anything about you — the only outbound request is the public,
anonymous tally ping described above. Works offline once loaded (the tally simply skips),
and fine on a phone.

The header names which checkpoint you are facing and its internal Elo; **AI thinks for**
(instant / 3 / 5 / 10 s) is the same strength dial as the local GUI, and the table talk
reports the true count every move — *“searched 16,384 positions in 5.0s”*. Measured in
headless Chrome on an M-series Mac: **~3,300 positions/second on an idle machine and
~1,000/s with a training run eating the cores**, i.e. **5,000–16,000 positions per move at
the default 5 s**. A phone will do less; whatever your machine manages, the page tells you
the real number rather than a claim.

It is the same table as the local GUI, from the same modules: every tile movement animated
in a straight line, a gear that sets the animation speed (`Off` / `0.5×` / `1×` / `2×`,
remembered between visits), ← and → to walk back through the game, a pictographic
newest-first move log, and a board where a filled square and an empty one cannot be
confused. See the local-GUI section below for what each of those does.

How it is put together, and why you can trust it:

| piece | where | how it is checked |
|---|---|---|
| ONNX export of the best checkpoint | `ludometer/export/onnx_export.py` | 100 real positions through torch **and** onnxruntime, max abs diff < 1e-4 (`tests/test_export_onnx.py`) |
| Azul rules, ported to JS | `web/player/js/engine.js` | 30 seeded Python games + 5 handcrafted edge positions replayed move by move — states, legal-action lists, all 182 encoding floats and outcomes must match exactly (`web/player/test/engine.test.mjs`) |
| PUCT search, ported to JS | `web/player/js/mcts.js` | full JS-vs-JS games under node: every move legal, every game terminates, no tiles lost (`web/player/test/selfplay.test.mjs`) |
| the page itself | `web/player/` | driven in headless Chrome: no console errors, a real human/AI exchange, tiles that fly *with the OS asking for reduced motion*, no sideways scroll at 390 px (`web/player/test/browser.test.mjs`), plus the full table check in `web/play/test/gui.test.mjs --only player` |

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

**The table, at a glance.** Nothing on this page ever covers the board — there are no
pop-ups at all. A wide **status band** across the top always says what is happening
(*“Your turn — pick a colour”*, *“AI is thinking — 2.1s of 5s”*, *“AI took 3 red from
factory 2 → row 4”*, *“You won 74–68”*) and keeps the running score; when the AI is
searching, the band itself is the clock. Both boards sit **side by side, identical** — you
on the left, the AI on the right, same size, same rows — laid out like the cardboard one,
pattern line row *r* level with wall row *r*. Round scoring and the end-game bonus breakdown
appear **inline under the boards**, so the final position stays there to be inspected.

**Filled and empty are not alike.** The palette runs on one rule: *hue means occupied*. A
tile that is on the board is saturated, rimmed and raised, in the base game's colours
(cobalt, ochre, terracotta, charcoal, ice); every empty square — wall, pattern line, floor —
is the same neutral, recessed well, with the wall's pattern kept as a thin outlined diamond
in the tile's own ink. The board reads as holes with a few jewels in it, from across the
room. Every colour lives in **one file**, `web/play/ui/theme.css`, and a skin is that file's
list of properties redeclared under `[data-skin]` — see `web/play/ui/THEMING.md`.

**Everything that moves is animated**, in a straight line, about half a second per group:
the tiles you take travelling to your pattern line or floor, the rest of the factory falling
into the middle, the first-player marker, full pattern lines onto the wall at round end, and
the floor line into the lid. The AI's moves play out the same way — *both* of them when a
round boundary puts it on move twice, since each move reports the position it was played
from and the refilled table is drawn before the second one. Input is locked only while the
tiles are actually travelling.

The **gear** under the status band sets the pace: `Off`, `0.5×`, `1×` (default) or `2×`,
remembered in `localStorage`. That switch is the *only* thing that governs motion. An OS
“reduce motion” setting deliberately does not silence the animation any more — macOS ships
it on for a lot of people, and it used to turn every flight into a teleport with no way to
ask for the animation back.

**Walk the game with ← and →.** Previous/next step through every position of the game, `End`
(or the **Latest** button) returns to play, and the status band says *“Viewing move 12 of
31”* while you look. It is pure client-side replay of positions the page already holds —
no request, no re-search — and the live game is untouched: while browsing there is nothing
to click, and jumping back to the latest position resumes play exactly where it was. It
works on finished games too.

The **move log** is pictographic and newest-first: the tiles that moved are drawn as tiles
in the same glazes as the board — *▪▪▪ factory 3 → row 5* — with words kept only for the
places. Every entry is drawn identically, including the newest one; the status band is where
“what is happening now” lives.

**Coach mode** (the toggle above the move log) scores *your* moves with the AI's own
evaluation — not a metric of our own. Before your move is applied, the same PUCT search the
opponent plays with runs on your position, and the log shows

    delta = Q(your move) − max Q over the children the search explored

on the network's own [−1, 1] value scale: `0.00` means you played the move the AI would have
played, `−0.06` means the search values yours six hundredths of a win worse, and at `−0.02`
or worse the entry also names the move it preferred. A move the search never visited is
reported as **unrated** rather than given a made-up number. It costs a couple of seconds per
turn (a dedicated ~2 s budget, capped at 3 s even against a 10 s opponent) and the band shows
a *“rating your move”* clock while it thinks. It needs a searching opponent, so the toggle is
disabled against the scripted baselines.

The board, the animations, the settings panel, the move navigator, the status band, the log
and the scoring panels live in `web/play/ui/` as framework-free modules that take state JSON
and know nothing about the server — `web/play/ui/PORTING.md` explains how the GitHub Pages
player adopts them, and `web/player/ui/` is a byte-for-byte copy that `tests/test_gui.py`
refuses to let drift. Both tables are driven in headless Chrome by

```bash
node web/play/test/gui.test.mjs        # --only play | --only player, --shots DIR
```

which checks — in **both** `prefers-reduced-motion` states, because that is the gap the
animations fell through — that tiles actually fly and for how long, that the speed presets
change it, that ← and → replay the game, that the log is drawn as tiles newest-first, and
that a filled square and an empty one are not close in colour.

Same thing without the browser (the GUI, the arena and the trainer share one registry):

```bash
uv run python -c "from ludometer.agents.registry import find_best_checkpoint; print(find_best_checkpoint())"
```

A time budget is available on any neural spec as `&think=<seconds>`, e.g.
`mcts:runs/run1/checkpoints/ckpt-024064.pt?sims=400&think=5`. It is off by default, so
training and the Elo ladder keep searching a fixed number of simulations per move.

## Status

Work in progress — built autonomously by Claude. Progress notes in `NOTES_FOR_REMI.md`.

## License

MIT — see [LICENSE](LICENSE). The trained model weights shipped with the browser player
are covered by the same terms. Faïence implements the rules of *Azul* (Michael Kiesling)
as a fan research project; it is not affiliated with or endorsed by the game's publishers.
