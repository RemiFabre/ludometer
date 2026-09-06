# Human-Data Exploitation Experiments — Implementation Plan

> **CLOSED 2026-08-21.** All tasks executed overnight; full results and the
> deploy decision (deliberately NO deploy — no same-speed candidate cleared the
> wall-clock gate) are recorded in `docs/HUMAN_GAMES.md` §18, which supersedes
> this plan. Extra work not in the original plan: dual-target policy rows
> (`--dual-target-min-elo 800`, per Rémi mid-run), `ludometer/train/distill.py`,
> and the ft2/ft3/ft3b/ds1 candidate ladder.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how much the elite-human data (799 BGA games / 21,286 elite positions; 382 Faïence games, 85 human wins+draws) can improve the Azul agent, and deploy a stronger agent to the site if one emerges.

**Architecture:** Four experiments in increasing cost: (E0) policy-agreement diagnostic of existing nets vs elite human moves; (E1) rebuild the pretrain buffer with balanced value labels; (E2) fine-tune the current best net on human data with self-play rehearsal, gauntlet it; (E3) full pretrain-then-selfplay A/B using run6 as the control arm. Deploy only on a clear gauntlet win.

**Tech Stack:** existing ludometer packages (`train`, `eval`, `human`, `azul`, `agents`), PyTorch on MPS, numpy npz buffers. No new dependencies.

**Spec:** the previous conversation turn (methods 1–5 + user approval). Key repo docs: `docs/HUMAN_GAMES.md`, `docs/HUMAN_DOWNLOAD_STRATEGY.md`.

## Global Constraints

- **Never break the online site** (https://remifabre-faience.static.hf.space/). Deploy only via `scripts/deploy_player.sh` (it runs export + JS/onnx/webgpu parity tests before any push); use `--dry-run` first; deploy only a net that clearly beats both run5/ckpt-006912 (internal 2382.9) and run4/ckpt-037888 (the deployed net, 2360.6) in a gauntlet at play-time settings.
- **Do not disturb** the BGA crawl runner (`continuous_runner.sh`, quota-sleeping) or the leftover idle uno workers. Do not touch `ludometer/azul/engine.py`.
- **Do not commit** — the working tree carries another session's uncommitted human-pipeline work (per Rémi, deliberately uncommitted). New code stays uncommitted too; document results in `docs/HUMAN_GAMES.md` §18 instead.
- Training device `mps`; nice long jobs; keep the machine usable.
- Human value labels are win-skewed (744/55/0) — never pretrain the value head on elite-rows-only data without the opponent value-only rows.
- Elo convention: internal ladder anchored at random=0; BGA displayed = raw − 1300.

---

### Task 1: Agreement diagnostic (E0)

**Files:**
- Create: `ludometer/human/agreement.py` (module with `__main__`)

**Interfaces:**
- Consumes: `data/human/replay.npz` (+ `replay.stats.json` game_records for per-game spans), `ludometer.train.net.load_net`, checkpoints `runs/run4/checkpoints/ckpt-037888.pt`, `runs/run5/checkpoints/ckpt-006912.pt`, `runs/run6/checkpoints/ckpt-009984.pt`.
- Produces: printed + JSON report: per-net top-1 / top-3 agreement, mean NLL on elite policy rows, split by within-game ply quartile (early/mid/late strategic-vs-tactical probe). Saved to `data/human/agreement.json`.

Steps:
- [ ] Write `agreement.py`: load npz rows where `policy_mask==1`; batch states through each net (`torch.no_grad`, mask illegal moves is NOT needed — compare argmax of raw policy logits over the 180-action space vs one-hot target; but DO renormalize over legal actions if a `legals` mask is derivable — it is not stored, so report raw-argmax agreement and note the caveat); compute top-1/top-3/NLL; quartile split from cumulative `game_records[].rows`.
- [ ] Run on the three checkpoints; sanity: a random-init net should sit near chance (~1/60 legal); trained nets far above.
- [ ] Save report, record numbers for the final write-up.

Acceptance: report exists with plausible numbers (trained nets ≫ random baseline), runs in <10 min.

### Task 2: Balanced pretrain buffer (E1)

**Files:**
- Modify: none (CLI flags exist)
- Produce: `data/human/replay_full.npz` (elite policy rows + opponent value-only rows)

Steps:
- [ ] `uv run python -m ludometer.human.cli --out data/human dataset --npz data/human/replay_full.npz --min-target-elo 650 --keep-opponent-rows`
- [ ] Verify with `ReplayBuffer.load`: ~43k rows, ~21k policy targets, value labels now two-sided (mean |value| sanity check).

Acceptance: npz loads, positions ≈ 43k, policy rows ≈ 21k, value labels include both signs in near-balanced proportion.

### Task 3: Faïence converter

**Files:**
- Create: `ludometer/human/faience.py`
- Data: snapshot already at scratchpad `faience-games/`; copy to `data/faience/games/` (gitignored under data/).

**Interfaces:**
- Consumes: faience-game/1 jsonl records: `seed`, `human_seat`, `moves[{ply,player,action}]`, `deals[{round,factories,bag,lid}]`, `final{finished,scores}`.
- Produces: `load_game(record) -> (positions, actions, human_mask, outcome, margin)` by replaying in `AzulState`; `build_npz(records, out, who="human", only_results={"win","draw"})` writing the same replay.npz schema via `ludometer.train.replay.ReplayBuffer` (mirror `human/dataset.py` conventions: value/margin in mover frame, elite rows = human rows of win/draw games, policy one-hot, opponent rows value-only).
- Replay strategy: `AzulState.new_game(seed)` then, per round, verify the engine's factories equal the recorded deal; if they differ, script the deal with `ludometer.human.convert.apply_deal`. Final engine scores must equal `final.scores` or the game is rejected (mirrors verify.js).

Steps:
- [ ] Write converter with strict validation; reject unfinished games.
- [ ] Run over all 382 games: expect ≥95% to replay with exact score match (site already verifies-by-replay, so ~100%).
- [ ] Write `data/faience/human_windraw.npz` (human rows of the 85 win/draw games, ≈2,000 policy rows + opponent value rows) and print stats.
- [ ] Bonus diagnostic: run Task 1's agreement on these rows (humans who beat the bot) — measures where winners diverged from the net.

Acceptance: ≥95% replay exactly; npz loads; stats printed.

### Task 4: Fine-tune with rehearsal (E2) — the fast experiment

**Files:**
- Create: `ludometer/train/finetune.py`
- Produce: `runs/ft1/` with checkpoints `ft-<steps>.pt`

**Interfaces:**
- Consumes: `runs/run5/checkpoints/ckpt-006912.pt` (base net), `runs/run5/checkpoints/replay.npz` (rehearsal self-play buffer), `data/human/replay_full.npz`, `data/faience/human_windraw.npz`.
- Produces: fine-tuned checkpoints in normal checkpoint format (loadable by `mcts:` agent specs).

Design (keep simple, measure before adding knobs):
- Batches mixed at ratio 1 human : 3 self-play (human policy rows oversampled ~×50 relative to natural frequency; self-play anchors against forgetting).
- Loss: the trainer's own `_losses` composition; human value rows carry both signs thanks to Task 2. LR 1e-4 (vs training peak), grad clip as config.
- Save at 250 / 500 / 1000 / 2000 steps; evaluate which (if any) helps before going further.
- Optional second variant if v1 regresses: add KL(π_ft ‖ π_base) penalty on self-play batches. Only build if needed.

Steps:
- [ ] Write `finetune.py` (standalone: load net via `load_net`, two `ReplayBuffer`s, mixed minibatches, save checkpoints; ~150 lines).
- [ ] Smoke: 20 steps on cpu, losses finite, checkpoint round-trips through `load_net`.
- [ ] Full run on mps (minutes, not hours).
- [ ] Gauntlet (Task 5 harness): each ft checkpoint vs base at sims=100, 200+ games.

Acceptance: at least one ft checkpoint ≥50% vs base with error bar, OR a clear negative result recorded.

### Task 5: Gauntlet evaluation harness

**Files:** none new — `ludometer.eval.gauntlet` CLI.

Steps:
- [ ] Baseline sanity match: `run5best=mcts:runs/run5/checkpoints/ckpt-006912.pt?sims=100` vs `run4site=mcts:runs/run4/checkpoints/ckpt-037888.pt?sims=100`, `--games 100`, confirm ordering matches rerate (~53-55% for run5best).
- [ ] Candidates (ft checkpoints, later hp1 best) into the same gauntlet, `--games 200+`, anchor run4site=2360.6.
- [ ] Finalist only: repeat at `?think=1.0` (closer to site conditions) before any deploy decision.

Acceptance: reproducible cross tables with Elo fits; decisions cite them.

### Task 6: Pretrain-then-selfplay A/B (E3) — the overnight experiment

**Files:**
- Create: `configs/hp1.json` = copy of `runs/run6/config.json` with `run=hp1`, `pretrain=data/human/replay_full.npz`, `pretrain_epochs=3`, fresh seed.
- Control arm: **run6 itself** (identical config, no pretrain, 9,984 games already on disk) — costs zero compute.

Steps:
- [ ] Write config; validate with `--max-games 128` smoke on a throwaway run dir (delete after).
- [ ] Launch `uv run python -m ludometer.train.run --config configs/hp1.json --max-games 9984` in background (~5h on mps); monitor via `runs/hp1/status.json` + `elo.jsonl`.
- [ ] Compare `runs/hp1/elo.jsonl` vs `runs/run6/elo.jsonl` at equal game counts (same anchor pool); also run Task 1 agreement on hp1's best ckpt (did human prior persist?).
- [ ] Gauntlet hp1 best ckpt vs run5best + run4site.

Acceptance: curve comparison plotted/tabulated; verdict on "does human pretraining shift the curve at 21k positions" recorded honestly, including a null result.

### Task 7: Decision, deploy, write-up

Steps:
- [ ] Pick the strongest candidate across E2/E3. Deploy bar: ≥55% vs run5best at sims=100 over ≥200 games AND ≥50% at think=1.0, i.e. clearly ahead of everything on disk. (run5best itself is NOT deployed today — if no human-data candidate clears the bar but run5best clearly beats run4site in the Task 5 sanity match, offer-deploy run5best as a separate cheap win, respecting the same parity-test gate.)
- [ ] Deploy: `HF_TOKEN=<from hf auth> ./scripts/deploy_player.sh --dry-run` then real. Verify site answers and `model_meta.json` updated. The exporter picks the best-rated checkpoint automatically — pass `--ckpt` explicitly to control it.
- [ ] Append §18 to `docs/HUMAN_GAMES.md`: numbers from E0–E3, decision, and what the next data milestone should trigger.
- [ ] Final report to Rémi: agreement numbers, gauntlet tables, deploy status, recommendation on when to rerun (e.g. at 2,000 BGA games).

## Self-Review Notes

- Spec coverage: methods 1 (Task 6), 2 (Task 4), diagnostics (Tasks 1, 3), weighting (already structural; per-Elo weighting deferred — single-player corpus makes it a no-op today, noted in write-up). Method 5 (seeded self-play) deliberately deferred: needs selfplay-loop surgery; recorded as next step in §18.
- Value-skew risk handled by Task 2 before Tasks 4/6 use the buffer.
- Ordering: 1→2→3 are cheap and independent of long jobs; launch Task 6's run early (it's the wall-clock bottleneck) once Task 2 is done, and do Tasks 3–5 while it runs.
