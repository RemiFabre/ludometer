# Roadmap: the next levels for the Azul nets

*One page, kept current. Written 2026-09-06 after Porcelain shipped and the
first night toward Lapis Lazuli. Details live in the files it points to; this
is the map, not the territory.*

## Where we are

| bot | honest Elo | how it was made | status |
|---|---|---|---|
| Cobalt | 2361 | run4: self-play, 1.8M params | regular rung |
| **Porcelain** | **2564** | W160 body (3.9M) pretrained on the 7M teacher's 1024-sim searched self-play + every position of the expert games searched by the same teacher (`/Users/remi/ludometer/docs/PORCELAIN.md`, `/Users/remi/ludometer/docs/superpowers/specs/2026-09-05-cloud-selfplay-design.md`) | **default opponent** |
| Lapis Lazuli | 2673 (+109) | Porcelain polished by on-policy self-play, 100k games | experimental (Settings switch) |
| Ultramarine | reserved | | needs +150 over the regular strongest |

The rule for a regular rung: **+150 wall-clock honest over the current
strongest, over ≥300 games at matched think time** (`/Users/remi/ludometer/docs/BOT_DEPLOYMENT.md`).
The gate runs on this Mac at `think=1.0`, because that is what a visitor's
browser experiences; a faster body earns Elo, a slower one pays it (W160 pays
about 55 Elo against Cobalt's body and wins anyway on quality).

## What the first night taught (the reading)

- **Searched targets from a stronger teacher are the lever.** Distilling the
  teacher's raw outputs gave +17 and +52 (mid1, mid2); distilling its
  *searched* opinions at scale gave +203 (Porcelain). Value target: half game
  outcome, half the search's root value (`value_search_weight`).
- **A net cannot be its own stronger teacher.** Distilling Porcelain's own
  2048-sim play into a fresh student gives parity; self-play polish gives
  ~+100 and plateaus; a 6-epoch re-pretraining on the same corpus gives
  parity. Every route that uses Porcelain as the teacher converges on +100.
- **Compute is now cheap.** The Rust engine (`/Users/remi/ludometer/docs/RUST_ENGINE.md`) makes an
  L4 job produce ~200k positions/s; a 2048-sim teacher game costs ~$0.005.
  The corpus that made Porcelain would cost ~$3 today. Money is no longer
  the constraint; a stronger teacher is.

## The next steps for strength, in order

1. **A genuinely stronger teacher** (the Lapis Lazuli road, ~$20-30, two
   days). Polish the 19.5M seed (`/Users/remi/ludometer/runs/big_t/checkpoints/ckpt-000000.pt`,
   rates 2500) on the fleet at 2048 sims for ~200k games with the Rust
   engine (a big net is nearly free on an L4 now), then distill it into the
   W160 body with `/Users/remi/ludometer/scripts/porcelain_pretrain.sh`, then gate. Expected: the
   student gains what the teacher gained beyond Porcelain's search.
2. **Strategy from the expert games** (the task force,
   `/Users/remi/ludometer/docs/STRATEGY_TASKFORCE.md`, $50). The hypothesis is Rémi's: elite
   humans out-plan the net in rounds 1-2 and lose to its calculation later.
   If the opening study confirms a gap, a human opening prior folded into
   the next student is a lever that no amount of self-play buys.
3. **A faster student, not a bigger one.** The browser fixes the student's
   speed. Quantisation (int8 in onnxruntime-web), a leaner body at Cobalt's
   latency, or tree reuse between moves in the browser (`/Users/remi/ludometer/web/player/js/mcts.js`
   starts every search cold) each buy search depth at equal think time.
   Measure positions/s in the browser stack before believing any of it
   (`/Users/remi/ludometer/web/player/test/selfplay.test.mjs`).
4. **Ultramarine** (+300 over Porcelain) is two turns of the crank, and the
   second one is where 3 becomes necessary: order of magnitude $80-150 and
   one to two weeks.

## What not to repeat

- More self-play of the shipped net past ~70k games (plateau).
- Distilling a net from its own longer search.
- Cloud CPU flavors for self-play (a fifth of the Mac per job; the L4 is
  the same price per position and 8x the throughput).
- Any gate at fixed sims: it overstates slow nets (the ft1/ft2 trap).

## Housekeeping that keeps this possible

- `/Users/remi/ludometer/NOTES_FOR_REMI.md`, newest on top, is the ledger of results; negative
  results go there too.
- Every cloud job through `ludometer.cloud.fleet` (spend ledger, cap).
- Other agents share this repo; commit path-scoped; never touch a running
  run's directory or `/Users/remi/ludometer/data/human/` state.
