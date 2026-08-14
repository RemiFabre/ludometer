# Notes for Remi

Running log of decisions, findings and things you should know. Newest entries on top.

---

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
