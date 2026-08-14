# Notes for Remi

Running log of decisions, findings and things you should know. Newest entries on top.

---

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
