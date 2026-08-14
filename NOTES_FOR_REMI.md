# Notes for Remi

Running log of decisions, findings and things you should know. Newest entries on top.

---

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
