# The road to Porcelain

*A handoff for the agent whose job is to produce the browser player's next
top opponent. Written 2026-09-05 by the agent that built the player's bot
ladder and deployment pipeline. The goal is fixed; the road is yours.*

## The goal, exactly

Train an Azul net that beats the current strongest, **run4/ckpt-037888
("Cobalt", 2361)**, by **at least +150 Elo, wall-clock honest** — measured
over **≥300 games at matched think time**, not at fixed simulations. That
number (~2511+) is Rémi's bar, set 2026-08-22, and it is deliberately high:
players must feel the difference. When you clear it, the bot ships under
the reserved name **Porcelain** (then Lapis Lazuli, then Ultramarine).

**The shipping half is not this document.** Follow `docs/BOT_DEPLOYMENT.md`
to the letter — it is self-contained: gate, export trio, bots.json,
staging Space, tests, production. This document is about getting a
checkpoint worth gating.

## Two traps that already burned an agent each

1. **Fixed-sims Elo lies.** runs/ft1-ft2 checkpoints rated 2394–2445 on
   the sims=100 ladder and LOSE to Cobalt in real time (they are ~4×
   slower per position). The browser plays wall-clock budgets. Inference
   speed is part of strength here; evaluate candidates the way a visitor's
   CPU will.
2. **`find_best_checkpoint()` is game-blind.** It once handed the Azul
   exporter an 84-input Uno net. Explicit `--ckpt` always.

## Where the last attempt stopped, and why it was still a milestone

runs/mid1 (distill-then-polish, see `docs/HUMAN_GAMES.md` §18 for the
recipe) gated at **52.5% over 100 games vs Cobalt at matched think — the
first candidate ever above 50% at wall-clock parity** — but that is only
~+17 Elo. The recipe works; it needs scale. The prior owner's own
diagnosis: better teacher as the corpus grows, longer polish, possibly the
4.2M-parameter "midB" body. runs/mid2 and ft3–ft5 continued along this
road after; check their elo.jsonl and finetune.jsonl for where things
stand before starting anything (mid2 was rating ~2300–2400 at fixed sims
in early September — remember trap 1 before celebrating).

## What you have that no previous attempt had

- **Human games, growing daily, two sources:**
  - The browser harvest: the public dataset
    [RemiFabre/faience-games](https://huggingface.co/datasets/RemiFabre/faience-games)
    (~2,300 games as of 2026-09-05, roughly 100+/day, every game
    replay-verified by the ingest before it is stored; `finished: false`
    marks abandoned games — position data, no outcome).
  - The BGA elite crawl: `data/human/replay*.npz` (managed by the
    human-games pipeline, `ludometer/human/`; ~10k+ positions from
    top-ranked play, growing with the daily crawl). `teacher_labeled*.npz`
    are search-relabelled variants from the §18 experiments.
- A converter path from browser records to training rows exists in
  `ludometer/human/` (the browser record replays deterministically from
  its seed in `ludometer/azul/engine.py`'s JS twin; deals are stored per
  round, so conversion needs no RNG port).
- The trainer's `--pretrain` flag consumes replay.npz-style files.

## Idea space (none of these are instructions)

- **Scale the validated recipe**: distill from a stronger/slower teacher
  (searched positions, not raw policy), then polish with self-play. The
  corpus doubling since §18 is the cheapest lever.
- **Human data as a style/opening prior**: the browser games are weak-to-
  mid human play (value-head caution: abandoned games carry no outcome);
  the BGA elite set is strong play. Mixing them naively hurt in §18's
  first pass — the mix ratio and masking convention matter.
- **Buy Elo with speed, not just accuracy**: the bar is wall-clock. A body
  that evaluates 2× faster at equal accuracy is worth ~a search doubling
  (~+100-ish Elo in this regime). Distillation into a smaller/faster body,
  quantization for onnxruntime-web, or attention-free blocks are all live
  options. Measure positions/s in the actual browser stack
  (`web/player/test/selfplay.test.mjs` reports it) before believing any
  win.
- **Tree reuse between moves** is implemented nowhere (each browser search
  starts cold — confirmed 2026-08-21). It is an engine change, not a
  training change, and it would strengthen EVERY bot including Cobalt,
  which moves the bar too. Worth doing for the game's sake; coordinate
  with Rémi on whether it counts toward Porcelain or resets the baseline.
- run6 explored aux heads (wall-placement prediction); its exporter
  support already exists (`aux_heads` in the meta).

## Ground rules

- Wall-clock gate before any excitement; 100 games to rule out, ≥300 to
  ship.
- Coordinate through files, not chat: log findings in NOTES_FOR_REMI.md,
  keep this file current as the road changes, and record negative results
  (the stand-downs are half the value of the ledger above).
- Other agents work in this repo daily (training runs, the BGA crawl, the
  browser player). Commit path-scoped; never touch `data/human/` state,
  `.bga_cookies.txt`, or running processes.
- The compute budget on this machine is shared with live training runs:
  `nice -n 15`, one thread for exports, and check what is running before
  claiming the GPU/CPU.
