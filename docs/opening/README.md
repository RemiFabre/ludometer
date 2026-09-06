# The opening atlas: do the experts out-plan the net in rounds 1-2?

*Written 2026-09-06 by the opening task force (`docs/STRATEGY_TASKFORCE.md`).
Numbers first, then the disagreements, then the verdict. Everything here is
reproducible from `ludometer/opening/` and the files under `data/cloud/opening/`;
the two long reports with the boards are `porcelain.md` and `teacher7m.md`.*

## 0. The short version

- **The experts and the net disagree a lot in the opening, and the disagreement
  is concentrated in the first three moves of round 1**: Porcelain's search
  (4,096 sims) picks the expert's move 30% of the time on move 1, 36% on move 2,
  39% on move 3, then 49%, 67%, 80%, 87% as the round runs out of tiles. The
  net thinks the expert's first move is a clear mistake (a win-probability loss
  of more than 5 points) in a third of the games.
- **Every check of "who is right" sides with the net.** A stronger and slower
  teacher (7M parameters) agrees with the experts *less*, not more; searching it
  four times deeper (16,384 sims) moves it toward the human move in under 1% of
  Porcelain's clear disagreements; a full search from the two child positions
  keeps the gap on average (though it prefers the expert's move in one
  disagreement out of five); and playing the two moves out with Porcelain on
  both seats scores the net's move higher (+0.14 score share on its 20 largest
  disagreements). One thing the checks did overturn: **the net's confidence**.
  The root-edge Q of a move the search barely visits overstates the expert's
  loss about fivefold (a 0.78 gap that plays out to 0.14), and in 5 of those
  20 positions the expert's move did better. Direction right, magnitude wrong.
- **Using the human opening does not buy Elo at matched think time.** The
  cleanest test, an imitation policy for the first *k* moves handing over to
  Porcelain, scores 44%, 50%, 49% (400 games) and 40% for k = 1, 2, 3 and the
  whole round; fine-tuning Porcelain toward the expert's round-1 choices with
  rehearsal lands at 51% over 300 games; toward rounds 1-2 at 52% over 700;
  toward the round-1 *disagreements only* at 54.1% over 700 (+28 ± 13 Elo,
  a +45 after 400 games that regressed). At most a few tens of Elo, far from
  a rung.
- **The 180° board rotation is not a usable symmetry**: it swaps the 1-tile and
  5-tile lines, and the search itself changes its draft in three quarters of the
  rotated positions. The net's asymmetry is the game's.
- **Verdict: the hypothesis is not confirmed at the level where it would earn
  a rung, but it is not empty either.** The opening is where the net and the
  experts differ most; the human opening as a *policy* does not beat the net's,
  and the net's own checks side with it on average, yet nudging Porcelain
  toward the experts on exactly the positions where they disagree is worth
  about +28 ± 13 Elo at matched think time over 700 games, for four minutes
  of GPU. That is the only piece worth carrying into the next student (E5),
  as the last fine-tune of the cycle, and it is a hint, not a lever. What the
  study did find is that agreement with the experts rises with net strength
  (Cobalt 41%, Porcelain 48%, Lapis Lazuli 50% raw top-1 over the whole game),
  so the expert games remain a good *diagnostic*, and a good source of searched
  positions, without being a better *policy* than the net's own.

## 1. What was measured

**Data.** 3,795 expert games (2 players, replayed exactly from their deals and
actions), 86,613 positions in rounds 1-2 for both seats, of which 45,876 are
decisions of the expert seat (the dataset's target player, plus both seats of
the 235 games where both players clear the pipeline's higher Elo floor).
Expert raw Elo (the crawl's own scale) ranges from 1,950 to 2,486; 20,065 of
the 22,083 round-1 expert decisions are by players rated 2,350 or more.

**Labels.** Every position is searched with no root noise at 4,096 simulations
by Porcelain (`runs/porc_w-p0905-2038/checkpoints/ckpt-000000.pt`, 3.9M params,
the default opponent, honest 2564) on this Mac's GPU (2.4 h with three
workers) and by the 7M teacher (`runs/ft2/checkpoints/ft-004000.pt`) on one L4
job (2.6 h, $2.1). Kept per position: the expert's move, the net's raw prior,
the visit distribution, each root edge's win-Q and margin-Q, the root value.
The **value loss** of the expert's move is `Q(net's pick) - Q(expert's move)`
in the mover's frame, win-probability units on [-1, 1] (0.10 is five
percentage points of win chance); the net's pick is the move it would actually
play (`decisive_action`: best win-Q among the well-visited children, biggest
margin among near-ties).

**The raw-prior baseline** (`ludometer.human.agreement`, no search, the whole
game, 114,885 expert decisions, quartiles of the game):

| net | top-1 | top-3 | Q1 top-1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|---|
| Cobalt (run4) | 40.8% | 68.8% | 35.8% | 39.4% | 40.1% | 48.7% |
| **Porcelain** | 48.5% | 76.5% | 45.2% | 47.3% | 47.5% | 54.3% |
| Lapis Lazuli (+109) | 50.3% | 78.2% | 49.6% | 48.9% | 49.0% | 53.9% |
| 7M teacher | 48.4% | 76.9% | 46.7% | 46.9% | 46.7% | 53.6% |
| 19.5M seed | 49.4% | 77.1% | 46.3% | 48.2% | 48.4% | 55.1% |

August's numbers for run4/run5/run6 on the then 21k rows were 38-40% overall
and 32-34% in the first quartile. Agreement has risen with strength, and the
first-quartile gap to the last quartile has shrunk from 15 points to 4-9.

## 2. Agreement by move (E1)

Porcelain at 4,096 sims, the expert seat's decisions. *top-1* is the net's own
pick; *loss > 0.10* is the share of decisions the net calls a clear mistake.

| round 1, move | n | top-1 | top-3 | raw prior | mean loss | loss > 0.10 |
|---|---|---|---|---|---|---|
| 1 | 3,794 | 29.5% | 55.8% | 23.2% | 0.085 | 32.0% |
| 2 | 3,794 | 36.1% | 65.4% | 33.0% | 0.070 | 25.8% |
| 3 | 3,794 | 38.7% | 71.3% | 37.8% | 0.068 | 24.1% |
| 4 | 3,794 | 49.1% | 79.3% | 44.5% | 0.063 | 22.5% |
| 5 | 3,740 | 66.8% | 91.9% | 64.3% | 0.039 | 13.7% |
| 6 | 2,539 | 80.0% | 98.0% | 80.7% | 0.019 | 6.3% |
| 7 | 583 | 87.3% | 99.1% | 89.5% | 0.014 | 4.6% |
| **moves 1-3** | 11,382 | **34.8%** | 64.2% | 31.4% | 0.074 | 27.1% |
| round 1, all | 22,083 | 49.4% | 76.4% | 46.5% | 0.058 | 20.8% |
| round 1, the other seat | 22,125 | 32.3% | 58.5% | 32.6% | 0.117 | 39.0% |
| round 2, moves 1-3 | 11,382 | 38.2% | 69.7% | 39.0% | 0.056 | 19.1% |
| round 2, all | 21,097 | 52.3% | 80.3% | 49.8% | 0.048 | 16.2% |

Three readings. The curve within a round is the "calculator" shape the
hypothesis predicted: once the round is constrained, the net and the expert
see the same thing. The expert seat agrees with the net far more than the
other seat (49% vs 32% in round 1), and the loss the net assigns to the other
seat's moves is twice as large, so **agreement with the net tracks playing
strength**. And the round-1 Elo split says the same: experts under 2,200 raw
agree 43% and lose 0.082 per move by the net's account; 2,200-2,350 agree
54% and lose 0.051; the 2,350+ majority sits at 49% / 0.057.

**The distribution of value loss** (round 1, expert seat): median 0 (the net
agrees or calls it a tie in half the decisions), p75 0.076, p90 0.205, p95
0.288, p99 0.452; mean 0.058, which the margin head reads as +0.7 points per
decision. Among the 48% of decisions where the net's pick differs: 20% are
within 0.02 of the net's own move (a coin flip to the net), 43% are a clear
mistake by the net's account (> 0.10), 21% a blunder (> 0.20).

**What the disagreements look like** (round 1, loss > 0.05, 6,312 decisions):
73% take a different colour; in 20% the expert took from the center where the
net took from a factory and in 11% the reverse; 17% take the same colour to a
different row (the expert to the lower, larger row 11%, to the higher, smaller
row 7%); floor-versus-build in under 5%. The most frequent single pattern is
"different colour, same row". In words: the experts and the net mostly
disagree about *which colour to start*, not about how much to take or where
to put it.

**The 7M teacher** (`teacher7m.md`) agrees less: 21%, 25%, 33% on moves 1-3
and 42% over round 1, with mean losses twice Porcelain's. Porcelain and the
teacher pick the same move in 57% of the expert decisions; where neither
matches the expert they still agree with each other 48% of the time. A
stronger net at fixed sims has not moved toward the experts; the student that
distilled the teacher's searches (and was then rated stronger at wall clock)
did.

## 3. Who is right (E1a, E1b)

**(a) Search deeper.** The 2,000 round-1 expert decisions where Porcelain's
loss is largest (all above 0.21) plus 500 where it agrees, relabelled by the
teacher at 4,096 and at 16,384 sims (one L4 job, 25 min, $0.30):

| on Porcelain's 2,000 clearest disagreements | sides with the expert | mean loss | loss > 0.10 |
|---|---|---|---|
| Porcelain, 4,096 sims | 0% (by construction) | 0.322 | 100% |
| teacher, 4,096 sims | 4.2% | 0.361 | 84.8% |
| teacher, 16,384 sims | 4.8% | 0.352 | 82.7% |
| *on the 500 agreements:* teacher 4,096 / 16,384 | 72.0% / 74.6% | 0.032 / 0.029 | 9.7% / 9.5% |

Going from 4k to 16k sims lowered the teacher's loss for the expert's move in
38% of the rows and raised it in 42%: no drift toward the human. The deeper
teacher picks Porcelain's move in 56% of these positions and its own
4k-sims move in 75%. Even in the bucket where Porcelain's loss is 0.2-0.3
(the most contestable), the deep teacher sides with the expert 6% of the time.

**(b) Search the children.** Because the expert's move often gets a handful of
visits at the root (its Q is then one or two backups of noise), every round-1
disagreement of the expert seat (11,184 positions, 22,306 child positions,
62 round-ending moves excluded) was re-searched from the position *after* the
expert's move and after the net's, at 4,096 sims each (11,092 pairs, 1.3 h on
the Mac GPU). The gap survives but shrinks, and it is not one-sided:

| root-edge loss bucket | n | child-search loss, mean | median | child loss > 0.10 | expert's child better (< -0.02) |
|---|---|---|---|---|---|
| 0 - 0.02 | 1,368 | 0.002 | 0.003 | 11% | 33% |
| 0.02 - 0.05 | 1,744 | 0.030 | 0.027 | 20% | 26% |
| 0.05 - 0.10 | 1,944 | 0.068 | 0.063 | 34% | 17% |
| 0.10 - 0.20 | 2,178 | 0.115 | 0.113 | 54% | 13% |
| 0.20 - 0.30 | 1,221 | 0.179 | 0.180 | 72% | 9% |
| > 0.30 | 919 | 0.280 | 0.274 | 82% | 6% |
| **all disagreements** | 11,092 | 0.109 | 0.077 | 44% | **19%** |

Read across: the root-edge Q and the child search agree on the ranking
(correlation 0.57) and on the size (mean 0.120 at the root, 0.109 from the
children), but in one disagreement out of five the full search of the two
children prefers the expert's move, and in a third of the "coin flip"
disagreements. By move: the child search calls the expert's move worse by
0.154 on move 1 (expert better in 13%), 0.10 on moves 2-4 (expert better in
20%), 0.078 on move 5 (23%). The first move of the game is where the net is
most sure and least often overturned.

**(c) Play it out.** From the 20 largest round-1 disagreements, 20 games after
the expert's move and 20 after the net's, Porcelain against itself at
`think=1.0`, the same seeds on both branches (800 games, 1.2 h on 6 workers). These 20 are
the extreme cases: the root Q called the expert's move a certain loss (gap
0.69-0.98, the expert's move had 1-10 visits of 4,096).

| | net's move minus expert's move |
|---|---|
| root-edge Q gap (the atlas' number) | 0.78 mean, i.e. ~39 points of win chance |
| child-search gap (4,096 sims each side) | 0.57 (19 of the 20; one round-ending move) |
| **played out, score share** | **+0.14** (net better in 12 positions, expert better in 5, tied in 3) |
| played out, final margin | +7.0 points |

Played out, the net's move still wins more, but the gap is a fifth of what
the root Q claimed, and in five positions the expert's move did better; in
one of them (a 2-tile pick at the end of round 1 that the net wanted floored)
the expert's move won 95% of the play-outs and the net's 15%. The root Q of a
move the search never explores is the value head's first guess, and it
overstates how bad the experts' unusual choices are. **The gap between (a)
and (b) is real: the net's confidence in its opening judgement is
miscalibrated by a factor of about five on its largest disagreements, while
its direction is right on average.**

## 4. Using it (E2, E3)

Every gate is a wall-clock gauntlet on this Mac against Porcelain at
`think=1.0` (`runs/gates/*_vs_porcelain*.json`); 100 games is a screen (±5
points), 300 a result (±3).

**E2, human openings then the net's calculation.** An opening policy trained on
the experts' 24,619 round-1 decisions alone (`ludometer.opening.imitation`;
10% of games held out): from scratch, a 2.1M net reaches 53.0% held-out top-1;
starting from Porcelain's weights, 62.3% (Porcelain's own prior: 50.8%, Lapis
Lazuli's: 56.8% on the same rows). The hybrid agent
(`hybrid:<opening>|<main>?k=3&think=1.0`) plays the imitation policy's argmax
for its first *k* own decisions of round 1 and Porcelain's search after.

| opening policy | k | games | W-D-L | score |
|---|---|---|---|---|
| Porcelain-initialised imitation | 1 | 100 | 43-2-55 | 44% |
| | 2 | 100 | 49-2-49 | 50% |
| | 3 | 100 + 300 | 56-1-43, then 137-4-159 | 56.5%, then 46.3% (pooled 48.9%) |
| | whole round 1 | 100 | 39-2-59 | 40% |
| from-scratch imitation | 3 | 100 | 49-1-50 | 49.5% |
| Porcelain-initialised, sampled (temp 1) | 3 | 100 | 40-2-58 | 41% |

The expert opening for one to three moves is indistinguishable from
Porcelain's own at matched think; handing the whole first round to the
imitation policy loses clearly, which is where the round's endgame needs the
calculation. Nothing in the strong net was retrained, so this isolates the
opening choice: **the human opening does not beat the net's, by the net's own
follow-up**.

**E3, teach the strong net the human opening.** `ludometer.train.finetune` on
Porcelain, human rows with a policy target, Porcelain's 1.2M-position polish
buffer as rehearsal (1:3), lr 5e-5, checkpoints at 500/1000/2000/4000 steps
(4 minutes each on the Mac GPU):

| human rows | checkpoint | raw Q1 top-1 (whole game) | games | W-D-L | score |
|---|---|---|---|---|---|
| round-1 expert decisions (24,619) | 1000 | 54.8% | 100 | 51-4-45 | 53% |
| | 4000 | 57.1% | 100 + 300 | 53-0-47, then 151-4-145 | 53%, then 51% (pooled 51.5%) |
| rounds 1-2 (48,149) | 4000 | | 100 + 300 + 300 | 53-3-44, 161-5-134, 141-10-149 | 54.5%, 54.5%, 48.7% (pooled **52.0% over 700, +14 ± 13 Elo**) |
| round-1 disagreements only (11,889) | 2000 | | 100 | 53-1-46 | 54% |
| | 4000 | | 100 + 300 + 300 | 55-1-44, 167-7-126, 149-7-144 | 56%, 56.8%, 50.8% (pooled **54.1% over 700, +28 ± 13 Elo**) |
| rounds 1-2 disagreements only (22,610) | 4000 | | 100 | 48-2-50 | 49% |

The fine-tune does what it is asked (Porcelain's first-quartile raw agreement
goes from 45% to 57% with the later quartiles unchanged). The round-1 rows
give parity within the noise of 400 games. The rounds 1-2 rows looked like
+30 Elo after 400 games and came back to 52% after 700 (+14 ± 13): noise.
The round-1 *disagreement* rows, i.e. only the positions where Porcelain's
search did not pick the expert's move, looked like +45 after 400 games and
settled at 54.1% over 700 (+28 ± 13, 2.1 standard errors); the rounds 1-2
disagreement variant screened at 49%. The honest reading of E3: the human
opening as a fine-tuning target is worth at most a few tens of Elo, the best
variant is a 2-sigma +28, and every variant regressed toward parity as its
gate grew. Cheap enough to carry as the last step of the next student's cycle
(E5), not a result to build on.

## 5. Symmetry (E4)

`ludometer.opening.symmetry`: the 180° rotation is the colour relabelling
0→0, 1→4, 2→3, 3→2, 4→1 with the rows reversed; it is exact on the wall and
wrong on the pattern lines (row *r* holds *r*+1 tiles), so it is only applied
where every line still fits and every full line stays full (6,268 of the
86,613 positions, among them all 3,795 first moves of a game). Raw policy on
the original versus the rotated position mapped back:

| net | slice | n | policy KL | top-1 changed | draft (colour+source) KL | draft top-1 changed |
|---|---|---|---|---|---|---|
| Porcelain | first move of the game | 3,795 | 7.27 | 99.5% | 2.97 | 76.2% |
| Porcelain | round 1, lines started | 2,312 | 5.10 | 98.9% | 2.45 | 80.8% |
| Lapis Lazuli | first move of the game | 3,795 | 5.06 | 81.2% | 3.78 | 79.4% |
| **Porcelain's search, 1,024 sims** | first move of the game | 1,000 | 17.5 | 93.4% | 9.25 | 75.7% |
| Lapis Lazuli's search, 1,024 sims | first move of the game | 1,000 | 13.0 | 81.7% | 11.0 | 79.4% |

The search changes its draft after the rotation as often as the raw net does
(76% vs 76%), so the asymmetry is a property of the game (the 1-tile line is
worth taking one tile for; the 5-tile line is not), not a generalisation gap.
There is nothing here to augment with; the only exact symmetry stays the seat
swap the encoding already uses.

## 6. Cost and files

Cloud: $2.59 of the $50 (one L4 job for the teacher's rounds 1-2, one for the
16k-sims relabel, one cancelled early attempt). Everything else ran on the Mac.

- `ludometer/opening/atlas.py` — `label` (the atlas), `children` (the child
  searches); `--workers`, `--net name=hub:<run>` and `--upload` for the fleet
  (`fleet launch --entry atlas`).
- `ludometer/opening/report.py` — `atlas` (this kind of report), `compare`.
- `ludometer/opening/playout.py`, `imitation.py`, `symmetry.py`;
  `ludometer/agents/hybrid.py` and the `hybrid:` spec in the registry;
  `ludometer.cloud.label --rounds`.
- `data/cloud/opening/` — the label files (`porcelain_r01_s4096.npz`,
  `teacher7m_r01_s4096.npz`, `teacher7m_r0dis_s16384.npz`,
  `porcelain_r0_children_s4096.npz`), the report JSONs (with the source's
  table ids, which stay out of `docs/`), the play-out and symmetry results,
  the fine-tuning row files.
- `runs/imit_r1*`, `runs/ft_open*`, `runs/gates/*_vs_porcelain*.json`.
