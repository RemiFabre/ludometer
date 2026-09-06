# Task force: do the experts out-plan the net in the opening?

*Handoff written 2026-09-06 for a separate agent. Budget: **$50 of cloud
compute**, every job through `ludometer.cloud.fleet` (ledger in
`runs/cloud/ledger.jsonl`; the CLI refuses past the cap, but state flavor
× timeout before every launch anyway). Read `docs/ROADMAP.md` first for
where this sits; `NOTES_FOR_REMI.md` (newest on top) for what was measured;
`docs/PORCELAIN.md` for how the nets were made.*

## 1. The hypothesis, in Rémi's words

The net has converged on a way of playing. It is an excellent calculator:
as a round nears its end the position is constrained enough that search
sees everything, and humans get punished there, often with heavy floor
penalties. But in **rounds 1 and 2**, where the long-term plan is chosen,
the elite humans are probably still the better strategists. The few
thousand expert games we have are top-tier play. Two questions follow:

1. **Measure it.** Where, and how much, do the net and the experts disagree
   in the opening, and who is right?
2. **Use it.** If a strong net plays the early moves like the experts, does
   it get stronger? Or does it play the following moves badly because it is
   not used to those positions?

Start with **round 1**, the first three moves, and widen only if that pays.

## 2. What you have

- **Expert games**: `data/cloud/bga_positions.json.gz`, 3,795 validated
  games in compact form (deals per round, action ids, first seat, outcome,
  scores). `ludometer.cloud.label.replay_positions(game)` replays one into a
  list of `AzulState` positions with the mover per position; deals are
  scripted so the replay is exact. Per-game Elo of the players and the
  target-seat annotations live in the human pipeline's dataset
  (`ludometer/human/dataset.py`, `data/human/replay.stats.json`);
  `docs/HUMAN_GAMES.md` is its manual. **Never write under `data/human/`**:
  the daily crawl owns it. In public-facing text say "expert games" and do
  not name the source.
- **The nets**: Porcelain `runs/porc_w-p0905-2038/checkpoints/ckpt-000000.pt`
  (3.9M, honest 2564), Lapis Lazuli `.../ckpt-096768.pt` (+109 over it),
  the 7M teacher `runs/ft2/checkpoints/ft-004000.pt`, the big seed
  `runs/big_t/checkpoints/ckpt-000000.pt` (19.5M, rates 2500). Load with
  `ludometer.train.net.load_net`; agent specs `mcts:<ckpt>?sims=N` or
  `?think=<s>`, and `&engine=rust` for the fast tree.
- **Search labels at scale**: `ludometer.cloud.label` replays games and
  searches every position with a published net (`fleet launch --entry
  label --rust ...`; see `docs/PORCELAIN.md` §2026-09-05 and
  `ludometer/cloud/label.py`). Add a `--rounds 0-1` filter (a small change:
  keep positions whose `state.round_index` is in the range) so a run labels
  only the opening. With the Rust engine an L4 job does ~200k positions/s:
  11,000 opening positions at 4,096 sims is ~45M evaluations, a few minutes,
  under a dollar.
- **Agreement metric**: `ludometer/human/agreement.py` (top-1/top-3 rate of
  a net's move against the expert's, by game quartile). August numbers for
  the older nets: ~40% top-1 overall, 33% in the first quartile, 48% in the
  last. Rerun it for Porcelain first; it is the baseline of this study.
- **Fine-tuning with rehearsal**: `ludometer/train/finetune.py` (human rows
  mixed with the net's own replay buffer; `policy_mask` per row decides
  which rows carry a policy target). Porcelain's polish buffer
  (`runs/porc_w-p0905-2038/checkpoints/replay.npz`, 1.2M positions) is the
  natural rehearsal set.
- **Gates**: `ludometer.eval.gauntlet --games N --workers 8 ... ?think=1.0`
  on this Mac, JSON into `runs/gates/`. 100 games to screen, ≥300 to claim.
  The comparison that matters for this study is *against Porcelain*, at
  matched think time; the fixed-sims ladder is only a sanity check.
- **The engine and the rules**: `ludometer/azul/engine.py` (module docstring
  has the state layout; `wall_col(color, row) = (color + row) % 5`).

## 3. Experiments, in order

Each has a measurable answer and a stop rule. Write the results, including
the negative ones, in `NOTES_FOR_REMI.md` as you go.

**E1. The opening atlas (measure, no training).** For every expert decision
in round 1 (moves 1-3; then all of rounds 1-2), label the position with
Porcelain at 4,096 sims (no root noise, `label.py` with the round filter).
Record: the expert's move, the net's visit distribution, the net's value
for the position, and the net's value after the expert's move versus after
its own best move (the "value loss" it assigns to the human choice). Then:

- agreement by move index (1, 2, 3, ...) and by round, for Porcelain and
  for the 7M teacher, against the August baseline;
- the distribution of value loss: how often the net thinks the expert
  blundered in the opening, and by how much;
- the top 20 disagreements (largest value loss, most frequent pattern).
  Describe each in words a player understands (what the expert took, where
  it went, what the net wanted). Rémi will read these.
- **Who is right?** Two cheap checks. (a) Search deeper: relabel the
  disagreements at 16,384 sims with the 7M teacher and see whether the
  net's opinion moves toward the human's; if it does, the net was wrong at
  its usual depth. (b) Play it out: from each disagreement position, play
  Porcelain-vs-Porcelain 20 games after the expert's move and 20 after the
  net's move (same seeds, `think=1.0`), and compare the score from the
  mover's side. This measures the net's own valuation, not the truth, but a
  gap between (a) and (b) is itself informative.

Cost: cents. Output: a short markdown atlas under `docs/opening/` plus the
numbers in the notes. Stop rule: if agreement in round 1 is already above
60% and the value losses are small, the hypothesis is weak; say so and
move to E4.

**E2. Human openings, net calculation (the cleanest test of "use it").**
Build an opening policy from the experts alone: a small imitation net (the
`hp1` recipe in `configs/hp1.json` was pure behaviour cloning on the whole
game; here train only on round-1 decisions of the higher-rated seat, ~11k
rows, minutes on the Mac). Then an agent that plays the imitation policy
(sampled or argmax) for moves 1-k of round 1 and hands over to Porcelain's
search for the rest. Gate that agent against plain Porcelain at `think=1.0`,
100 games per k for k in {1, 2, 3, whole round 1}. No retraining of the
strong net, so any gain or loss is attributable to the opening choice
alone. Implement it as an agent in `ludometer/agents/` (a wrapper holding
two agents and a move counter), registered so the gauntlet can spec it.

Stop rule: if every k loses by more than the noise, the human opening
does not survive contact with the net's own follow-up, which is exactly
Rémi's worry; report it and try E3 before giving up.

**E3. Teach the strong net the human opening (fine-tune with rehearsal).**
Fine-tune Porcelain with `finetune.py`: human rows = round-1 (then rounds
1-2) expert decisions with a policy target, all other rows policy-masked
(value/margin still from the outcome); rehearsal = Porcelain's polish
buffer; low learning rate (5e-5), a few thousand steps, checkpoints at
500/1000/2000/4000. Screen each at 100 games vs Porcelain, gate the best
at 300. Variant worth one run: fine-tune only on the E1 disagreement
positions. Cost: Mac time only.

**E4. Symmetry (measure before using).** Rémi's intuition is that rows 1
and 5 (and 2 and 4) are "the same". They are not, exactly: the wall is a
Latin square with `col = (color + row) % 5`, and a 180° rotation of the
wall *is* a consistent colour relabelling (0→0, 1→4, 2→3, 3→2, 4→1), but
the pattern lines hold 1, 2, 3, 4, 5 tiles in rows 0-4, so reversing the
rows swaps a one-tile line with a five-tile line; and a cyclic colour shift
moves every wall column by one, which breaks horizontal adjacency at the
wrap (column 4 is not next to column 0). The only exact symmetry is seat
swap, which the encoding already uses (player-to-move frame). So do not
augment with rotated games as if they were exact; measure the *approximate*
symmetry first: for a few thousand positions, apply the colour relabelling
that a 180° rotation induces together with the row reversal, ask the net
for both, map the actions back, and report the policy KL and the top-1
change rate. If the net is already nearly invariant, there is nothing to
gain; if it is wildly asymmetric on positions where the asymmetry should
not matter (nothing on the walls yet, i.e. the first moves of round 1,
where the pattern-line capacities are the only difference), that is a
generalisation gap worth an augmentation experiment restricted to the
opening, with the capacity difference respected (map row r to row 4-r only
for the *draft* decision of which colour to take, never for the placement).

**E5. If E2 or E3 wins: fold it into the next student.** The pretraining
cycle (`scripts/porcelain_pretrain.sh`) consumes one replay file; add the
expert opening rows (policy-masked to rounds 1-2) to the corpus with a
sampling weight, or run the E3 fine-tune as the last step of the cycle,
and gate the student against Porcelain. This is the point where the $50
buys something: a corpus from the stronger teacher (see the roadmap, step
1) plus the opening prior is the Lapis Lazuli candidate. Coordinate through
`NOTES_FOR_REMI.md` with whoever runs the teacher polish.

## 4. Traps

- **Wall-clock only.** A net that plays better openings but searches slower
  is not stronger; every claim is a gauntlet at `think=1.0` on this Mac
  against Porcelain, and 100 games is a screen, not a result.
- **Elo of the expert seat.** Not every seat in an expert game is an
  expert; the dataset builder knows which seat is the target and its Elo at
  game time (`--min-target-elo`). Use the target seat's decisions for
  policy targets; the opponent's rows are positions, not lessons.
- **Round boundaries.** Turn order does not alternate across a refill (the
  marker holder starts the next round); `state.round_index` and
  `state.current_player` are the truth, not move parity.
- **The engines.** Rust and Python trees are exact twins (gauntlet 45-2-53);
  use `engine=rust` freely for speed, keep the Python one for anything you
  want to step through.
- **Other agents.** The BGA crawl runs daily from `data/human/`; a teacher
  polish may be running from `runs/big_t`. `ps` before claiming cores;
  `nice -n 10` anything long; never touch a running run's directory.

## 5. What a good report looks like

Numbers first: the agreement table by move and round, the value-loss
distribution, the E2 gauntlets per k, the E3 screens. Then the twenty
disagreements as a player would read them. Then one paragraph: is the
hypothesis confirmed, where exactly, and what it is worth in Elo. Rémi
reads `NOTES_FOR_REMI.md`; the atlas goes under `docs/opening/`; anything
that would embarrass the source of the expert games stays out of public
pages.
