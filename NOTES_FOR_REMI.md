# Notes for Remi

Running log of decisions, findings and things you should know. Newest entries on top.

---

## 2026-08-15 — run2 retired at +2020, run3 (structured net) is training

- **run2 final**: best checkpoint **ckpt-023040 at +2020 ± 39** after ~24.5k games — a hair
  above run1's +2014 but with clearly better per-game learning. I stopped it at midday: its
  gains had flattened to ~+20 Elo/1k games, and the compute is better spent on run3.
- **run3 is the redesign**: instead of a flat MLP, a *structured* net that sees the board as
  22 entities (factories, pattern rows tied to their wall rows, floors, supply) mixed by
  self-attention, with a factorized source×color×destination policy head. Plus: 512 sims/move
  self-play (2× run2) made affordable by MCTS tree reuse, and it **warm-starts by pretraining
  on run2's entire 500k-position replay buffer** — it begins where run2's knowledge left off
  rather than from scratch. Both run1's and run2's best are pinned as Elo anchors, same ruler.
- Mid-day incident, resolved: run2 crashed once when the run3 build agent edited MCTS code
  in-place (training workers import live source). Fixed by moving all build work to isolated
  git worktrees; run2 lost ~20 minutes, nothing else.
- The dashboard now shows all three runs; the browser player still serves run2's best and
  I'll redeploy it the moment run3 produces a stronger checkpoint.

## 2026-08-15 — Anyone can now play our best net, in their browser: **https://remifabre.github.io/ludometer/**

Send that link to anyone. It opens a full Azul game against **run2/ckpt-023040 (+2020 Elo)**
and **nothing runs on a server** — the tab downloads the net once and does all the thinking
locally. No account, no install, works offline once loaded, playable on a phone.

**What is actually being served.** A *snapshot* of the best rated checkpoint, exported to
ONNX. It does not follow the training run: when run2 finishes (or any checkpoint out-rates
this one), re-publish with

```bash
./scripts/deploy_player.sh
```

which re-exports the best checkpoint, re-runs the correctness gates, rebuilds the `gh-pages`
branch and waits for the site to answer 200. It writes that branch through a temporary git
index — **it never checks anything out, so it is safe to run while run2 is writing into
`runs/`** (I ran it that way today).

**Speed, measured not guessed.** Headless Chrome against the live URL, twice, hours apart:
**16,384 positions in 5.0 s (~3,300/s)** while the machine was quiet, and **5,179 in 5.0 s
(~1,000/s)** with run2 back on all the cores. So call it **5,000–16,000 positions per move
at the default 5 s on this Mac**, tracking how busy the machine is; a phone will be lower
still. The page reports the true count every move in the table talk, so a visitor sees their
own figure rather than my marketing. For scale, the ladder rates checkpoints at 100
sims/move, so even the pessimistic end is ~50× more search than the rating was measured at
— the same "the Elo is a floor" story as the local GUI.

**Payload.** 26.9 MB on disk, **~15.9 MB over the wire** (GitHub Pages gzips both big
files): 13.3 MB ONNX → 12.4 MB, 13.5 MB onnxruntime wasm → 3.5 MB, plus ~150 KB of my own
JS/CSS/HTML. All vendored, no CDN, no third-party request at runtime. The page streams the
download and shows a percentage, because on a phone that is several seconds of nothing.

**The part I was most worried about, and how it is nailed down.** The browser needs the Azul
rules and the exact 182-float observation the net was trained on, in JavaScript. A hand port
that is 99% right would produce an AI that looks fine and plays subtly nonsense, so the port
is *proved*, not reviewed:

- `scripts/dump_fixtures.py` plays 30 seeded games with the **Python** engine and records
  every move — the full `to_json()` state, the legal-action list *in engine order*, all 182
  encoding floats, the scores, the outcome — plus the bag ordering of every shuffle (JS
  cannot reproduce CPython's Mersenne Twister, so it replays Python's deals instead).
- `web/player/test/engine.test.mjs` replays all of it in JS and demands an exact match:
  **2,155 moves, 9,384 assertions, all green**.
- Random play never reaches some branches, so five positions are built by hand (all-
  monochrome round end, bag running dry, score clamped at 0, the end-game bonuses, marker on
  a floor line) — 191 more moves. I mutation-tested the gate: five deliberate bugs
  (a shifted encoding offset, a wrong column bonus, an off-by-one floor overflow, reordered
  legal actions, the wrong marker fallback) — the first four were caught by the random games,
  and the fifth *only* by the handcrafted case, which is why those exist.
- The net itself: exported ONNX matches torch to **3.2e-05** on 100 real positions, checked
  twice — once against onnxruntime-python in the exporter, once against the actual vendored
  onnxruntime-**web** build in `parity.test.mjs`, since the browser runs the latter.
- Whole-stack: JS-vs-JS games under node (every move legal, every game terminates, tile
  census intact) and the real page driven in headless Chrome, including against the deployed
  URL (`node web/player/test/browser.test.mjs --live`).

**Caveats, honestly.**

- **It is a snapshot.** The live site does *not* track `runs/`; it is whatever
  `deploy_player.sh` last published. The header names the checkpoint and its Elo.
- **First load is heavy.** ~16 MB. Fine on wifi, slow on a train. After that it is cached.
- **The search runs on one thread.** SharedArrayBuffer needs COOP/COEP headers, which
  GitHub Pages does not send, so no wasm threading. A batch-of-one MLP gains little from
  threads anyway, but it does mean a phone will be several times slower than this Mac.
- **The chance handling is the full Python one** (re-sampled determinizations with a
  reshuffled bag at refills, capped at 4 outcomes per edge) — I did *not* take the
  documented shortcut of cutting the tree at round boundaries.
- **Dirichlet noise is not ported.** It is a self-play training device; against a human it
  would only make the AI worse.
- **`onnx` + `onnxruntime` are new dependencies**, in a non-default `export` group
  (`uv run --group export ...`). Nothing in training, self-play or the arena imports them,
  and `uv lock` added them without moving a single existing pin — torch and numpy are
  untouched.
- I did not touch `ludometer/train/`, `ludometer/eval/`, `configs/` or anything under
  `runs/`, and I ran only `tests/test_export_onnx.py`, never the full suite.

## 2026-08-15 — Morning report: run2 caught run1 overnight

- run2 (the 3× bigger net) trained all night at ~31 games/min and is at **+2005 ± 32 after
  22.5k games** — statistically level with run1's best (+2014), winning ~45% of direct
  head-to-heads. It learned far more per game than run1 (+1750 at 10k games vs run1's +1410)
  but plays fewer games per hour, so the wall-clock race was closer than the per-game one.
- The run continues toward its 60k budget through the day; the GUI's "Strongest trained
  (auto)" will flip from run1's checkpoint to run2's the moment one out-rates it.
- Practical note for your next game: the Elo ladder rates checkpoints at 100 sims/move, but
  with the 5 s thinking budget the AI searches ~50-100× more than that — it plays well above
  its listed rating. Expect it to be noticeably stronger than yesterday's opponent.
- Science note: run2's curve is also concave in raw games (fast to ~+1400 by 4k, grind after)
  — same shape as run1, which is evidence the shape belongs to *the game + method*, not to a
  particular net size.

## 2026-08-14 — GUI: board aligned like the real one, real tile colours, the AI now thinks

All four things you flagged after your first game, all in `web/play/` plus a small additive
change to the search:

- **Board alignment.** Pattern line *r* and wall row *r* are now one grid row (`.board-grid`,
  fixed row height, right-aligned lines), for both boards — exactly the cardboard layout, so
  you can see which wall square a line feeds. Verified in a headless browser: every pattern
  line's top edge is within 1 px of its wall row's.
- **Real base-game colours.** Cobalt `#17509e`, ochre `#d99a12`, terracotta `#b23a26`,
  charcoal `#23272d`, ice cyan `#31b8d1`. The ghosted empty wall squares now carry a diamond
  in the true glaze on a pale ground, so the wall's colour pattern reads at a glance.
- **The AI thinks on a clock.** New selector "AI thinks for": instant / 3 / 5 / 10 seconds,
  **default 5 s**. `MCTS.search(state, time_limit_s=...)` keeps simulating until the budget
  is spent (wall clock checked every 8 sims) with `sims` demoted to a ceiling that the GUI
  raises to 20,000. Additive and default-off: the trainer never passes it, so **training
  behaviour is unchanged** — I only ran `tests/test_gui.py` while run2 is training.
- **Why this matters for strength**: the Elo ladder rates checkpoints at **100 sims/move**.
  At a 5 s budget run1's best searches **~7,700 positions per move** (measured with run2
  hogging most of the cores), i.e. **50–100× more search than its rating was measured at**.
  It plays meaningfully above its listed +2014 — treat the number as a floor. The page tells
  you the truth every move: *"searched 7,712 positions in 5.0s"* in the table talk.
- **The move is now animated.** A turn is three beats: your move lands at once, the AI
  thinks (a kiln-dot indicator with the clock running), then the source dish lights up and
  its tiles fly to its board over ~1.7 s before the position updates. Your input is locked
  throughout; completed lines glaze into the wall one tile at a time at round end.
- Mechanically that needed the turn split in two requests: `POST /api/act` with
  `defer_ai: true` returns as soon as your move is on the board, then `POST /api/ai` spends
  the budget. The old single-request `/api/act` still works exactly as before.

## 2026-08-14 — run1 finished (+2014 Elo), run2 launched for the night

- **run1 final**: 25,000 games in 3 h 08 min. Final Elo **+2001**, best checkpoint
  **ckpt-024064 at +2014 ± 56** — that's ~640 Elo above our strongest scripted baseline,
  i.e. it should beat the heuristic ~9 games in 10. Curve: overall slope 61 Elo/1k games,
  **R² = 0.909** against a straight line. Shape: fast start (~+300/1k to 4k games), long
  steady ~+50/1k grind after — mildly concave, still rising at the cap. Azul, by your
  hypothesis' lens, reads as "easy to pick up, keeps rewarding study", which honestly
  matches the real game.
- **run2 is now training overnight**: ~3× bigger net (5×768), deeper search (256 sims/move
  vs 160), 60k-game budget. Crucially, run1's best checkpoint is **pinned in the anchor pool
  at +2014**, so run2's Elo axis is directly comparable — if run2 ends above +2014 you're
  looking at a genuinely stronger agent, same ruler.
- Play tip: "Strongest trained (auto)" in the GUI will silently switch to run2 checkpoints
  the moment one out-rates run1's best.

## 2026-08-14 — You can now play the strongest model in two commands

- **How to play it** (this is the whole thing):
  ```bash
  uv run ludometer-gui      # http://127.0.0.1:8737/ — then press "Deal tiles"
  ```
  The Opponent dropdown already sits on **“Strongest trained (auto)”**, and next to it you
  see which checkpoint that is (e.g. `ckpt-023040 · +1920 Elo · run run1`). The table-talk
  panel says it again once you start: *“You're facing ckpt-023040, rated +1920 on our
  internal ladder.”* Nothing to copy-paste any more.
- **It resolves at deal time, not page load.** The new `best` agent spec scans
  `runs/*/elo.jsonl` and picks the highest-Elo checkpoint whose `.pt` still exists. run1 is
  training as I write this, so every new game you deal faces the newest strongest
  checkpoint — leave the tab open overnight, hit "Deal tiles" in the morning and you are
  playing a stronger opponent than tonight, with no config change.
- **Sims selector** next to the dropdown: 100 (blink-fast, weaker) / **400** (default,
  ~1 s per move) / 1200 (strongest, a few seconds per move). Elo is measured at the
  trainer's eval sims, so 100 plays below its rating and 1200 a bit above.
- Fair warning: at +1920 Elo it beats the heuristic baseline ~83% of the time, so expect to
  lose. The "Suggest a move" button is still the heuristic, not this net.
- Baselines (heuristic / greedy / random) and hand-typed
  `mcts:runs/run1/checkpoints/<name>.pt?sims=N` specs are still in the dropdown if you want
  to feel the difference between rungs of the ladder.

## 2026-08-14 — run1 halfway report

- 12,288 / 25,000 games. Current Elo **+1467 ± 55** — it has passed every scripted baseline
  (greedy +1220 at ~5k games, heuristic +1378 at ~9k) and now beats the heuristic 75%.
- **First linearity readout: slope ≈ 101 Elo per 1k games, R² = 0.924 over 25 evals.**
  Early curve was steeper (~+300/1k up to 4k games) and it eased to ~+50/1k after 8k —
  so on raw game count the curve is concave rather than strictly linear. Worth discussing:
  Elo-vs-log(games) or Elo-vs-wall-clock may be the fairer x-axis for your hypothesis; the
  dashboard shows the raw-games fit.
- No crashes, heartbeat steady, ~135 self-play games/min sustained.

## 2026-08-14 — run1 is training (evening)

- **The full stack is built**: engine (57 tests), baselines + Bradley-Terry Elo
  (heuristic ≈ +1378 vs random), AlphaZero-style trainer (MCTS + 1M-param net, 8 self-play
  workers ≈ 135 games/min), and a **playable GUI**: `uv run ludometer-gui` → azulejo-styled
  board at 127.0.0.1:8737, opponents: random/greedy/heuristic or any checkpoint
  (`mcts:runs/run1/checkpoints/<name>.pt?sims=400`).
- **run1 launched**: 25,000 self-play games (~4-5 h), Elo eval every 512 games against the
  fixed anchor ladder. Watch it live on the dashboard — the sample run is gone as soon as
  real points arrive; the Elo plot's linear fit + R² is your hypothesis readout.
- Interesting Azul-specific findings from the build: turn order does NOT strictly alternate
  (marker holder can move twice across a round boundary); two deterministic arg-max players
  can loop Azul *forever* (nobody completes a line → tiles cycle bag→floor→lid); and MCTS had
  to reshuffle cloned bags so the search can't peek at the exact next deal. All handled + tested.
- Sanity: after just 512 games the net was already ≈ +253 Elo vs random.

## 2026-08-14 — Engine + dashboard done (afternoon)

- **Azul engine finished**: 57 tests green, ~6,000 random games/sec single-core (3× my target),
  full official rules incl. edge cases (tile-conservation checked at every step). Encoded
  observation is 182 floats from the current player's perspective.
- **This dashboard is live**: `web/dashboard.html` auto-refreshes every 30 s; I keep a
  regenerator running while I work. The Elo plot you're seeing under "sample-run" is synthetic
  demo data so you can see the layout — real runs will replace it.
- In flight right now: baseline agents (random / greedy / heuristic) + arena + Bradley-Terry
  Elo fitting. Next: the play-vs-AI GUI and the AlphaZero-style trainer.

## 2026-08-14 — Project kickoff

- **Game confirmed as Azul** — the voice transcript said "Asur"; you confirmed Azul when I asked before you left.
- **Repo name: `ludometer`** — "measuring games". I named it after the real thesis (linear Elo
  progression as a proxy for game quality) rather than just Azul, since you want to reuse this
  on your own game designs afterwards.
- **Plan**: full 2-player Azul engine (official rules, tested) → baseline agents + Elo arena →
  AlphaZero-style self-play training on your Mac (MPS, 12 cores) → repeated training runs with
  improvements between them → browser dashboard updated as training progresses.
- **Elo methodology** (this matters for your linearity hypothesis): checkpoints are rated
  against a *fixed anchor pool* (random = 0 Elo anchor, plus greedy/heuristic baselines and
  frozen past checkpoints), so the curve is comparable across the whole run and across runs.
- I'll commit and push regularly, and update this file whenever there's something worth telling you.
