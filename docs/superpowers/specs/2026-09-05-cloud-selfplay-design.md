# The road to Porcelain, with cloud compute

*Written 2026-09-05 for Rémi, by the agent running the Porcelain push. The first
half answers your questions and states the plan; the second half is the design
of what gets built. Budget cap: $100 of Hugging Face Jobs credits. Everything
is logged in `runs/cloud/ledger.jsonl`, so the spend is auditable at any time.*

## 0. Short answers to your questions

**Is this project a good fit for the microduck-style compute?** Yes, but not
in the way microduck uses it. Microduck's training is GPU-bound (4096 MuJoCo
Warp envs on one card). Ludometer's self-play is bound by the **pure-Python
tree search**, about 90 microseconds per simulation per core. The net forward
pass is already batched across 64-128 concurrent games, so a GPU only shaves
the smaller half of the cost. What a self-play run needs is *cores*, and the
Jobs price list is extremely lopsided in our favour:

| flavor | vCPU | $/hour | $/vCPU-hour |
|---|---|---|---|
| cpu-upgrade | 8 | 0.03 | 0.004 |
| cpu-xl | 16 | 1.00 | 0.063 |
| l4x1 (GPU) | 8 | 0.80 | 0.100 |
| a10g-large (GPU) | 12 | 1.50 | 0.125 |

A `cpu-upgrade` job is roughly 25× cheaper per core than any GPU flavor. On
one core of this Mac, a batched self-play driver does 2.7k positions/s with
the 7M teacher at 1024 sims, 7k with the mid net, 9.3k with the site net.

**Measured on the Jobs (evening, `ludometer.cloud.bench`), the picture moved.**
The cloud CPUs are slow at exactly the teacher's cost, a 7M-parameter forward
pass: 459 positions/s per thread at batch 64 on `cpu-upgrade` (AMD EPYC)
against 3,418 on a Mac core, and the Python search itself is 3.9× slower. A
`cpu-upgrade` job yields ~3,400 positions/s for $0.03/hour. A `t4-small`
driver runs at 3,600 positions/s (search-bound, the T4 makes the net free), so
`l4x1` (8 vCPU) should give ~29k/s for $0.80/hour. And **this Mac's GPU does
17,500 positions/s on its own** (24k in fp16), five cpu-upgrade jobs' worth.

| generator | positions/s | $/hour | $ per 1k positions/s-hour |
|---|---|---|---|
| this Mac (MPS, 6 drivers × 128 games) | 17,500 (fp16: ~24k) | 0 | 0 |
| cpu-upgrade (8 drivers × 32 games) | ~3,400 | 0.03 | 0.009 |
| t4-small (3 drivers) | ~10,000 | 0.40 | 0.040 |
| l4x1 (8 drivers × 64 games, fp16) | ~29,000 | 0.80 | 0.028 |

So: CPU jobs are the cheapest per position by 3-4×, GPU jobs are 8× more
throughput per job, the Mac is free. The fleet runs as a mix: the Mac, 36
cpu-upgrade generators, 2 l4x1 generators, 8 cpu-upgrade labelling jobs. At
~25k evaluations per teacher game that is roughly 15-20k games/hour, so the
100k-game corpus is an overnight job for under $15. The concurrency quota
allowed 20 jobs without complaint; the 36-job wave is the next probe.

**Naming.** Jobs run in the `pollen-robotics` namespace (that is where the
credits are) under the neutral name `rl-experiment`. The private repos that
hold the code bundle, the weights and the generated games live under your
personal account (`RemiFabre/rl-experiment-*`), so nothing Azul-shaped shows
up in the org's repo list. The job command prints `[rl-experiment]`, nothing
more specific. This is discreet, not deceptive: anyone opening the bundle
sees what it is.

**Will it reach Porcelain?** My honest estimate: 50-60% for Porcelain (+150
wall-clock Elo over Cobalt) within this budget, 10-15% for "superhuman" in any
meaningful sense (nobody has a human-strength anchor for Azul; the BGA elite
corpus gives an agreement metric, not an Elo). The reasoning is in §2.

## 0b. Outcome (22:05 the same day)

**Porcelain shipped.** The wide body pretrained on the teacher corpus (59,792
teacher self-play games at 1024 sims + all 3,795 BGA elite games searched by the
teacher) beat Cobalt **229-0-71 over 300 games at matched think time**, honest
Elo 2564 (+203; the bar was +150), with no self-play polish at all. Fleet
spend at ship time about $25. Phase B (on-policy polish, fleet plays the
student) runs overnight toward Lapis Lazuli, +150 over Porcelain.

## 1. Where things stand (checked today, not inherited)

- Cobalt = run4/ckpt-037888, 1.81M params, 2361 on the fixed-sims ladder,
  0.206 ms/position on one idle core.
- mid2 (2.94M, the last attempt) reached **2405 ± 29 at fixed sims** at game
  9216, then was stopped. Nobody wall-clock-gated it. The mid body runs at
  0.65× Cobalt's speed, which costs roughly 60 Elo at matched think time, so
  my prior was that mid2-9216 would be about even with Cobalt in real time.
  **Measured: 57-1-42 over 100 games at matched think time**
  (`runs/gates/mid2-009216_wallclock.json`), about +52 honest Elo, the best
  candidate ever measured and a third of the bar. The road is real.
- The strongest thing we own at equal search is ft2/ft-004000 (7.04M, ~2445
  pooled). It loses at wall clock because it is 3.4× slower. It is the
  natural **teacher**.
- Body latency sweep (batch-1, one core, interleaved with a reference net):

  | body | params | ms/position | speed vs Cobalt |
  |---|---|---|---|
  | Cobalt body (embed 96, 1 layer, body 1024) | 1.81M | 0.206 | 1.00× |
  | midA (embed 128, 2 layers, body 1280) | 2.94M | 0.319 | 0.65× |
  | W160 (embed 160, 1 layer, body 1536) | 3.92M | 0.330 | 0.62× |
  | W192 (embed 192, 1 layer, body 1536) | 4.14M | 0.374 | 0.55× |
  | ft2 teacher body | 7.04M | ~0.70 | 0.30× |

  W160 holds 33% more parameters than midA at the same speed. Both students
  will be trained; the gate decides.

## 2. The plan, and the intuition behind it

The recipe that produced the only above-parity candidate so far (mid1, 52.5%)
is *distill from a stronger teacher, then polish with self-play*. Its author's
diagnosis was "it needs scale". I agree, with one sharpening: the distillation
targets so far were the teacher's **raw network outputs** (soft policy, value).
The far better target is the teacher's **searched** policy, the visit
distribution after 1024 simulations of the 7M net. That is a policy the small
body could never produce on its own, and it is exactly what AlphaZero-style
training improves on: search-improved targets. Producing them is pure
self-play compute with a slow net, which is what the fleet is for.

Phases, each gated by a measurement:

1. **Phase A, teacher corpus (fleet, fixed weights).** ft2-4000 plays itself
   at 1024 sims on N cpu-upgrade jobs. Target: 100k+ games, ~5M positions,
   for a few dollars. Students (Cobalt body and W160) are pretrained on the
   corpus as it grows: policy cross-entropy on the searched visits, value on
   the outcome, margin on the score. This alone is a candidate: a Cobalt-speed
   net trained on 2600-strength targets.
2. **Phase B, on-policy polish (fleet + local learner).** The trainer runs on
   this Mac with a new self-play engine, `selfplay: "hub"`: it publishes its
   weights to the hub every few minutes and consumes the game shards the fleet
   uploads. The fleet plays the *student* at 1024 sims. Everything else in the
   trainer (checkpoints, fixed-sims Elo curve, dashboard) is untouched, so the
   dashboard shows the run like any other.
3. **Gate, ship.** 100-game wall-clock screen of the best fixed-sims
   checkpoints against Cobalt; a ≥300-game gate for the winner; if ≥ +150,
   `docs/BOT_DEPLOYMENT.md` to the letter, staging first. The name is
   Porcelain.
4. **Beyond Porcelain (if budget remains).** Iterate: the polished student
   becomes the next teacher's initialisation, the 7M body is polished on the
   fleet too, and the next distillation cycle starts from stronger targets.
   Ideas from the literature worth trying at that point, in order of my
   confidence: Gumbel-style policy targets (completed Q-values, much better
   targets per simulation), higher self-play sims (2048) now that they are
   cheap, and a wider-not-deeper body at Cobalt's latency.

What could break it, and what I will do: (a) the Jobs quota allows only a
few concurrent jobs: use `cpu-performance` (32 vCPU, $1.90/hour) for the
rest, still cheap; (b) hub vCPUs are much slower than measured: measured on
the smoke job before any fleet launch; (c) the small body cannot absorb the
targets (ds1/ds2 failed with raw targets): W160 is the hedge, and the
wall-clock gate has the last word; (d) the local learner cannot keep pace:
the fleet is paused, not the learner, and the `l4x1` fallback exists.

Stop rules: total ledger spend ≥ $90 (nothing new is launched past $80 of
committed timeouts), or two consecutive distill-polish cycles that fail to
move the wall-clock screen. Negative results go in `docs/PORCELAIN.md`.

## 3. Design

### 3.1 Components

```
ludometer/cloud/
  hub.py         thin HfApi wrapper: upload/download with retries, list, pointer files
  bundle.py      source tarball of the tracked tree (code + configs, no runs/ data/)
  shards.py      GameRecord <-> shard .npz (per-game boundaries kept, so add_game works)
  generator.py   JOB SIDE: fetch weights, play blocks, upload shards, poll for new weights
  hub_selfplay.py TRAINER SIDE: the "hub" self-play engine (start/set_weights/play/close)
  fleet.py       LOCAL CLI: launch / ps / cancel / ledger, spend cap enforced
```

Repos (all private, personal account):
- `RemiFabre/rl-experiment-src` (dataset): `bundle-<sha>.tar.gz`
- `RemiFabre/rl-experiment-weights` (model): `<run>/current.pt` + `<run>/current.json`
  (`{"version": n, "sha256": ..., "games": ...}`) so a generator can tell a new
  version without downloading it
- `RemiFabre/rl-experiment-shards` (dataset): `<run>/v<version>/<job>-<k>.npz`

### 3.2 Data flow

Generator (in the job): `current.json` → if version changed, download
`current.pt` and broadcast weights to its `BatchedSelfPlayPool(workers=vCPUs,
device=cpu)` → play a block of `--block` games with seeds
`hash(job_id, block_index)` → write shard → upload → repeat until the job's
timeout kills it. The block in flight when the timeout hits is lost (about a
minute of work), nothing else.

Hub engine (in the trainer): `start(weights)` publishes version 1;
`set_weights` publishes a new version at most every `hub_publish_s` seconds;
`play(n)` lists the shards repo, downloads unseen shards, converts them back
to `GameRecord`s, and returns once it has `n` games (respecting
`should_stop`). Each record carries the weights version it was played with;
the trainer's log line reports the lag distribution. Shards older than
`hub_max_lag` versions are skipped, not fed.

### 3.3 Trainer changes (minimal)

`TrainConfig` gains `selfplay: "hub"` plus `hub_run`, `hub_publish_s`,
`hub_max_lag`, `hub_shard_dir`. `make_selfplay` dispatches to `HubSelfPlay`.
Nothing else in `trainer.py` changes: the engine honours the pool interface.

### 3.4 Job bootstrap

Image `python:3.12-slim`; `pip install numpy huggingface_hub` plus the CPU
torch wheel (no 2 GB CUDA download); download and extract the bundle;
`python -m ludometer.cloud.generator ...`. Secrets: `HF_TOKEN` only.
Timeout per job: 8h (billing stops at timeout, relaunch is one command).

### 3.5 Error handling

- Every hub call retries with backoff (the hub rate-limits bursts).
- A generator that cannot reach the hub keeps playing and retries the upload;
  after 30 minutes of failures it exits so the job stops billing.
- The fleet CLI refuses to launch when `ledger committed + requested > $90`
  where committed = Σ flavor price × timeout of every job not yet finished.
- The trainer treats an empty poll as "no games yet" and heartbeats, so
  `status.json` stays fresh and the dashboard keeps rendering.

### 3.6 Testing

- Unit: shard round-trip (records → npz → records, bit-exact); pointer-file
  version logic; ledger arithmetic; generator seed uniqueness across jobs.
- Integration without the network: `HubSelfPlay` and the generator share a
  `LocalHub` fake (a directory) so the whole loop runs in a test with a tiny
  net, and the trainer's existing smoke test runs with `selfplay: "hub"`.
- Live: one smoke job on `cpu-upgrade` for 10 minutes ($0.005), which also
  measures hub positions/s and decides the fleet size.

## 3.7 Human positions, searched (added after Rémi's note on the BGA crawl)

The BGA elite crawl (3,795 validated games on 2026-09-05, ~200/day) enters
the corpus a second way: `ludometer/cloud/label.py` replays every game in the
engine (deals are scripted, the replay is exact) and the fleet searches every
decision point of *both* players with the teacher at 1024 sims. The rows look
exactly like self-play rows (visit policy, outcome value, search value,
margin, final walls), so the same pretrain path consumes them. The student
learns what the teacher would play there; the human's role is to have reached
a position self-play rarely visits. `label export` writes the compact
positions file, `fleet launch --entry label --asset ...` runs it.

## 4. Ledger

`runs/cloud/ledger.jsonl`, one line per job: id, namespace, flavor, $/hour,
timeout, launched-at, purpose, and when known, ended-at and cost. The
`fleet ledger` command prints the running total. Target: finish Porcelain
under $40, keep the rest for the "beyond" phase.
