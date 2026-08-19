# Notes for Remi

Running log of decisions, findings and things you should know. Newest entries on top.

---

## 2026-08-19 — unoplus1 finished: the house rules roughly tripled Uno's learnable depth

**The rule-knob experiment has its answer.** Same trainer, same net shape, same search;
four house rules. On its own ladder (random = 0, rated over matches to 500):

- **Uno+ reaches +1853 and is still climbing at the end** (last eval is the best one,
  0.81 vs its greedy, 0.94 vs its heuristic). Plain Uno reached +662 and spent its whole
  back half flat at its greedy's level.
- Learned range: **~+1150 Elo of climb for Uno+ vs ~+510 for Uno**; slope-half at 82% of
  budget vs 46%. The log fit beats the line for both (0.879 / 0.878 vs 0.820 / 0.776).
- The value head prices "one card from going out" at **+0.75 in Uno+ vs +0.98 in Uno**
  (boundary diagnostic on the final checkpoint; fresh hands at 0.0 in both). That drop is
  the rules working: stacking and the 7-swap make a near-win genuinely revocable, which
  is where the extra counterplay - and depth - comes from.

Honesty box, before this becomes a slide: (1) the two Uno ladders share only their
random=0 anchor, so cross-reading +1853 vs +662 inherits the usual cross-pool caveat -
the games-0 MCTS already rates +701 on Uno+'s ladder vs +144 on Uno's, i.e. the scales
are not identical; the *learned range* comparison is the safer number. (2) The decisions
budget overshot: early near-random nets bank cards and played ~445-decision hands, so the
run consumed **7.5M decisions, 3.4x uno1's 2.2M** - visible and honest on the decisions
axis, but "same budget" it was not (at uno1's 2.2M mark Uno+ was already ~+1550). Even
the trained net keeps hands at ~95 searched decisions: voluntarily drawing is part of
good Uno+ play, not just early-net noise. (3) max_game_moves=2500 truncation applied to
some early self-play hands; the trainer log shows it fading to zero as the net learned.

## 2026-08-19 — Perfect play is ratable after all, and tic-tac-toe's ceiling is *reached*

Rémi's observation, confirmed: "perfect play has unbounded Elo" only holds while the
opponent loses every game. Once the agent draws, the fit is finite. So `ttt:perfect` /
`c4:perfect` now exist (exact WDL-optimal moves from the solver, random among the optimal
set), the gauntlet learned `--game`, and a 200-games-per-pairing tic-tac-toe gauntlet
(random anchored at 0) says:

perfect +527 ± 25, **ttt1-net@256sims +519 ± 25** (head-to-head with perfect: 0.50 — all
draws), net@64sims +506, heuristic +488, greedy +381. The trained net is statistically
indistinguishable from perfect play. Even against random, perfect only scores 0.95: from a
drawn game it cannot force wins against accidental good defense, which is exactly why its
rating stays bounded. The measured line is on the tictactoe panel of web/compare.html.

Connect Four gets the same treatment once c4_1 exists (and once the solver proves it can
afford the opening — to be probed when the suite build frees the CPU).

## 2026-08-19 — ttt1: the calibration flatline, in both metrics, in 2.1 minutes

The first solved game through the full pipeline, and the ludometer passes its own
calibration test. **Elo:** +351 at zero games (that is the 96-sim search alone), one jump
to ~+550 after the first 256 games, then dead flat — every later checkpoint draws its
predecessor 0.5. **% optimal** (the headline for solved games, on the 2,000-position
suite): 94.0% at zero games → saturated at ~97.8% by game 768 → never moves again.
Blunder rate settles at ~2.5%. The residual ~2% is the *evaluator's* 64-sim budget, not
missing knowledge — the same final checkpoint scores 99.2% optimal at 256 sims. On the
combined chart, tic-tac-toe's entire 30k-decision budget is a sliver that ends before
Azul's first bend. If the study's statistics did not scream "trivial" here, they would be
broken; they scream it.

## 2026-08-19 — uno1 finished: Uno holds about +650 Elo of learnable skill, and stops

`runs/uno1` completed its 120,000 hands cleanly (~4.8 h wall time, 59 eval points).
The final read, all rated over matches to 500 on Uno's own ladder (random = 0,
greedy +572, heuristic +396 pre-rated):

- **Curve:** +144 at 0 hands → climbs with Azul-like slope to ~+550 by 40k hands →
  spends the entire back half of the run oscillating in the **+570–660 band**. Best
  checkpoint **+662** (ckpt-116736); last point +651 ± 58. The smoothed slope halves at
  **46% of budget** and never recovers.
- **Against the ladder:** the final checkpoints beat `uno:random` ~0.95, `uno:heuristic`
  ~0.5–0.6, and sit at 0.25–0.44 against `uno:greedy` — i.e. the trained net ends *at*
  the level of the best hand-written baseline, ~80 Elo above it at best. Two plausible
  readings, not distinguished by this run: the net stopped finding skill, or hand-win
  optimisation genuinely tops out near greedy's match play (greedy's dump-expensive-cards
  rule is match-scoring-aware in a way "win the hand" training is not).
- **The headline picture** (web/compare.html, now the combined absolute chart per Rémi's
  call): over comparable ~2M-decision budgets, **Azul reaches +2001 above its own random
  while Uno reaches +662** — a 3.3× gap in distinguishable skill, and Uno's rating scale
  is *stretched* by match amplification, so the true gap is larger, not smaller.
- **Shape:** the log fit beats the line for every run (uno1: R² 0.878 log vs 0.776
  linear; Azul run2: 0.962 vs 0.686). Rémi's expectation that learning is log-shaped —
  fast early, slow late — is what the data says; the discriminating quantity across
  games is where the slope dies and how much range the game covers before it does.

Queue: ttt1 next (minutes), then unoplus1 (~3 h), then c4_1 (overnight).

## 2026-08-19 — Uno verified, Uno+ built, the calibrated pair built, and the comparison page exists

Everything below happened while `runs/uno1` trained (it was untouched; engine edits only
reach spawned eval workers, and every change on those paths is behaviour-identical or
never-hit for the live run).

**Uno verification (the double-check you asked for).** Three layers:

1. All 22 uno tests pass, 446 tests total across the repo.
2. The §1 value-boundary diagnostic on the live checkpoint (52k games): value one card
   from going out **+0.979** (p10 +0.959), fresh hand **+0.094**, mid-hand +0.086. The
   value function is healthy — compare the buggy attempts' +0.75.
3. A high-effort multi-agent code review of `ludometer/uno`. Three findings confirmed by
   execution and fixed: `apply()` accepted illegal actions where AzulState raises (a
   mismatched colored card silently rewrote `current_color`); a match truncated at the
   MAX_HANDS backstop went to the winner of the *last hand* instead of the score leader
   (fixed with an explicit `_horizon` flag so search horizons keep being decided by the
   current hand — the §1 trap is now pinned by a test); and `hand_index` counted hands
   differently on the two terminal paths. Guards added: `tree_reuse` and `margin_head`
   are rejected for Uno configs with an explanation (silent no-op / saturated scale).
   `configs/uno_smoke.json` shipped a stale key and could not load — every config in
   `configs/` is now load-tested. Backlog (real but deferred): the discard list is
   redundant state copied on every clone; `_has_playable` duplicates the playability
   predicate.

**Elo records now carry the cross-game x-axis.** Both self-play engines count
*decisions* (moves with >1 legal action) per game; the trainer accumulates and writes
`positions` and `decisions` into every elo.jsonl record. Measured constants for the runs
that predate the field: Azul 52.98 decisions/game, uno1 43.9 moves/hand x 0.428 searched
(with the trained net) = 18.8 decisions/hand.

**Uno+ is implemented** (`ludometer/uno/plus.py`, `configs/unoplus1.json`): draw always
legal and atomic (cap 15), +2/+4 like-on-like stacking with the victim choosing, the
7-swap with per-observer `known[]` counts that `search_root` deals to the opponent before
determinizing the rest (the §2 trap), 9-card deal. Acceptance, measured over 300 random
games as §2 asks: **forced turns 63.9% → 18.6%** (27.1% under greedy) — the target was
<35%; mean legal actions per decision turn 3.2 → 3.8 (4.3 under greedy) — **the >5.0
target is missed** and recorded as such (you said "significantly different is enough";
the forced-turn collapse is the effect that matters). One surprise worth knowing:
*random* players bank cards under draw-always-legal — a random Uno+ hand averages
~1,700 moves (p90 4,400) where greedy plays out in 41. Truncation backstops are sized
for the early near-random net (max_game_moves 2500); an MCTS player at uniform priors
already ends hands in ~84 moves with zero truncations. Budget: 50,000 hands ≈ uno1's
~2.0M decisions.

**Tic-tac-toe + Connect Four are implemented** with the solver harness of §3: bitboard
engines, shared two-ply greedy/heuristic baselines, a memoised TTT solver, a WDL
alpha-beta C4 solver (shared TT, mirror normalisation, forced-block/double-threat
prunings), stratified solved suites in `data/solved/`, and `ludometer.eval.optimal`
which walks a run's checkpoints into `optimal.jsonl`. One deviation: **the C4 suite
samples from ply ≥ 12, not §3's ply ≥ 8** — the pure-Python solver is the constraint
(ply 8 was hours; deepest-first solving with a warm TT makes ply 12 minutes). Baseline
sanity on the TTT suite: random 60.6% optimal, greedy 90.5%, heuristic 96.0% — the
ladder orders correctly. Configs: `ttt1.json` (4,096 games, expect a flatline at ~100%),
`c4_1.json` (30,000 games ≈ 0.9M decisions, expect it short of 100%).

**The comparison page exists**: `web/make_compare.py` → `web/compare.html`, built to
§4's rules (normalised shape headline with slope-half markers, Elo as one-scale-per-game
small multiples, decisions on the x axis with games in every hover, the truncation-R²
table). New runs appear automatically; the %-optimal panel appears once a solved run has
an `optimal.jsonl`.

Queue from here: uno1 finishes → journal + final charts → ttt1 (minutes) → unoplus1
(~3 h) → c4_1 (overnight), each added to the page as it lands.

---

## 2026-08-18 — Faïence lives on Hugging Face now, and every shared game becomes data

The docs/HUGGINGFACE.md plan is implemented and live (§8 there has the full report).
What you need for editing your posts:

- **The play link is https://remifabre-faience.static.hf.space/** — that is the one to
  put everywhere. The old remifabre.github.io/ludometer/ address now shows a small
  "the game has moved" page with a button (same social card as before, so old links
  still unfurl). The previous build stays deployed at `classic/` as an emergency
  fallback but is deliberately not linked from the stub (your call, 2026-08-18):
  games played there are never recorded, so everyone is pointed at the new address.
- **Every finished or abandoned game is shared by default**, with a visible
  "Share played games" switch in the game's Settings and honest text in the About
  panel. Records go to the ingest Space (https://remifabre-faience-ingest.hf.space,
  a separate Space, not openwarlock-signal: no cost to keep them apart, and a
  redeploy of one can never take down the other), which **replays each game in the
  real engine** and rejects anything that does not reproduce its own deals and score.
  Verified games land in the public dataset
  https://huggingface.co/datasets/RemiFabre/faience-games (CC0, JSONL, format
  unchanged: `faience-game/1`).
- The ingest carries your HF_TOKEN as a Space secret. `GET /stats` on it shows live
  counters; no IPs or user agents are ever logged.
- Deploys: `scripts/deploy_player.sh` now pushes the Space and the gh-pages stub in
  one go; `scripts/deploy_ingest.sh` does the collector. The live browser test
  (`browser.test.mjs --live`) targets the Space and switches sharing off so test
  games never pollute the dataset.

---

## 2026-08-18 — What we build next is written down: `docs/NEXT_GAMES.md`

Agreed after the Uno run got going. The brief is written for an implementer with no
context — decisions, traps, acceptance criteria — so you can hand it to another agent.

- **Uno+ first** — same engine, one rule knob, two curves. Draw always legal, +2/+4
  stacking, the 7-swap, and a 9-card opening hand. This is the only experiment in the plan
  that varies exactly one thing, and it is the question the ludometer is ultimately for.
  Acceptance is measured *before* training: plain Uno gives 3.2 legal moves per turn and
  60% forced turns; Uno+ has to clear 5.0 and drop under 35% or the rules did not work and
  there is nothing worth training.
- **Then tic-tac-toe and Connect Four, together.** Same genre, zero luck, zero hidden
  information, one obviously shallow and one not. If the ludometer cannot separate those
  two, nothing else it reports is trustworthy.
- **Solved games are headlined by % optimal play, not Elo.** You were right that a perfect
  player in the anchor pool changes what Elo means — it is worse than that, it diverges,
  since Elo against an opponent you never beat is unbounded. So the solver stays out of the
  pool, gets rated separately as a reference line where the fit is finite, and the headline
  becomes the fraction of positions where the agent preserves the game-theoretic result.
  That has a real ceiling at 100%, it is absolute, and it tests the linearity thesis far
  more sharply than Elo can. Same idea as your 2800 line on the Azul chart, with an exact
  number instead of an assumption.
- **Chess is an explicit non-goal**, and the reason is worth keeping: it would fail in the
  direction that looks like success. Every curve is linear at the start; a from-scratch
  chess run on this Mac would spend its whole budget in that first sliver and come back
  perfectly linear, and we would "conclude" chess is a perfectly designed game from a
  measurement that never reached the interesting part. Onitama, Hex or Breakthrough give
  the same character at a size where the curve can bend.
- The document also fixes the rules of the chart: **the x axis is decisions, not games**
  (an Azul game is ~52 decisions, a Uno hand ~17, a Uno match ~440 — a 26x spread), Elo
  ceilings are never compared across games, and run1 is the Azul line to compare against
  with run2 drawn as a search-budget band around it.

## 2026-08-18 — Second game on the same rig: **Uno is training** (`runs/uno1`)

The point of the framework was always to compare *shapes* of learning curves across games.
Azul is calibrated; Uno is the contrast case — everybody knows it, the rules take a minute,
and the expectation is that its curve flattens early where Azul's kept climbing.

**What was added, and what was not.** Nothing in the Azul path changed behaviour: all 443
existing tests pass untouched. The framework was already duck-typed on a state object, so a
second game needed three small seams — a `ludometer/games.py` registry (`"game": "azul"`
by default, absent from every run1-run6 config), the four chance/fingerprint helpers moved
out of `mcts.py` onto the state classes (verbatim), and the encoding/action widths read off
the state instead of imported as constants. The new code is `ludometer/uno/` (engine +
three baselines) and `tests/test_uno.py`.

**Hidden information.** Uno is not a perfect-information game, which the search assumes.
It is handled by determinizing at the root (PIMC): the mover keeps everything they can
legitimately see — their hand, the discard pile, the opponent's card *count*, both scores —
and the unseen cards are redealt at random into the opponent's hand and the deck. Every
draw stays a chance node, so the tree cannot memorise the deck order it was handed. The
encoding deliberately shows only the observation, so the net cannot read the
determinization off its own input.

**Two things worth knowing before reading the curve:**

1. **Elo scales are not shared across games.** A Uno 1500 and an Azul 1500 are different
   numbers — different anchor pools, different variance. The dashboard now draws one
   overview chart per game for exactly that reason, and tags every run with its game.
   The honest cross-game comparison is the *shape*, and Elo normalised by each game's own
   random→ceiling span.
2. **A match to 500 is ~20 near-independent hands**, and getting the value function right
   across that took two tries — both failures are worth knowing about, because both would
   have produced a flat Uno curve that had nothing to do with Uno.

   *First attempt:* label a position mostly by the hand it was played in
   (`segment_value_weight`). Wrong, and measurably so — that is not a value function, it
   resets at every hand boundary. The trained net priced a position one card from going
   out at **+0.75** and the position immediately after actually going out at **+0.02**, so
   the search saw winning a hand as falling off a cliff and steered away from it.

   *Second attempt:* train on single hands (one hand = one episode, label = who went out),
   rate over full matches. Correct for training, still wrong for rating: the search inside
   a match still bootstrapped across hand boundaries. The same checkpoint won **41.5% of
   single hands against `uno:greedy` and 0.0% of matches** — which is arithmetically
   impossible for an undistorted agent.

   *What is in the tree now:* the search's horizon ends with the current hand
   (`UnoState.search_root`). Hands are all but independent — only the score carries over,
   and the encoding does not even show it — so "win this hand" is what the net predicts
   and the only thing the tree can bootstrap consistently. Same checkpoint, same
   opponents, after the fix: **0.150 vs greedy, 0.375 vs heuristic, 0.775 vs random**, up
   from 0.000 / 0.175 / 0.600. `tests/test_uno.py` pins all of it.

**Measured before launch:** a random Uno match is ~1,100 moves and ~20 hands, a single
hand is ~42 moves, and an Azul game is ~52 — but only ~40% of Uno moves have more than one
legal action, and the mean branching factor is **3.2** against Azul's 20-60. So 96
sims/move here is a *deeper* search than run1's 160 was for Azul. **Which means one "game"
is not one unit of practice across the two studies:** an Azul game is ~52 decisions, a Uno
hand ~17, a Uno match ~440. Plot the cross-game comparison against *decisions* and keep
games as the secondary axis — and treat a conclusion that only survives on one axis as no
conclusion. Throughput: **1,540 hands/min**, so the 120,000-hand budget is ~3 hours
including evaluations.

**Baseline ladder** (120 games per pairing, matches to 500): `uno:random` 0,
`uno:heuristic` +396, `uno:greedy` +572. Note the ordering — the naive "dump your most
expensive card, save the wilds" greedy beats the hand-written positional heuristic, and no
amount of parameter tuning on the heuristic closed the gap. That is itself a small data
point about how much structure Uno actually has for a human-legible rule to exploit.

## 2026-08-17 — Experiment verdict: strategic heads didn't pay (yet); back to the proven recipe

- **run6 (aux strategic heads + deep policy targets) ran 10k games and underperformed**:
  average ~+2240, winning only ~38% vs run5's best — while run5 had reached an honest
  **+2383 in just 6.9k games and was still climbing when I paused it**. Following the
  evidence: run6 stopped, **run5 resumed for a long run** (53k games of budget left), now
  with run6's genuinely useful bugfixes kept (a stall-breaker hole in the decisive-move path,
  plus an eval backstop that ends pathological marathon games — the cause of yesterday's
  35-minute evals).
- The aux-head idea isn't dead — it may need a lighter weight and more full searches — but
  it doesn't get more compute until the proven capacity recipe stops paying.
- Ladder recap (honest, re-rated ruler): run1 +2014 → run2 +2020 → run3 +2255 →
  run4 +2361 → run5 +2383 (at only 6.9k games) → run5 continues today.
- Also fixed overnight: the Mac went to sleep and interrupted work once — a caffeinate now
  keeps it awake for the duration.

## 2026-08-17 — run6 is ready: **you said the tactics are good and the long game is weak, so run6 changes what the net is taught, not how big it is**

Your verdict on run5 was specific, and it is not the verdict a capacity problem gives.
"Tactically good, strategically weak" is what a net looks like when **nothing in its
training ever asks it about the far future** — and that is literally true of run1-run5.
Look at everything the loss has ever contained: who won, by how much, and which move the
search liked *here*. Three labels, none of which is about the shape the game will end in.
run5 also confirmed capacity is not the binding constraint: 3.9x the parameters, same
plateau.

So run6 spends nothing on the net (same 7.0M architecture, tensor for tensor) and
everything on **supervision and horizon**. Three changes, and they attack the same thing
from three sides.

### 1. The net now predicts the *final walls* — 30 extra outputs, and this is the main bet

From the same trunk, for **both players**: will they close wall row 1..5, column 1..5,
colour 1..5, by the end of the game? 30 sigmoids, trained with BCE at weight 0.1 against
the true final board.

The reason to expect strategy from this, rather than just "more parameters", is the
**time horizon of the label**. Whether your row 3 ends up closed is settled four or five
rounds after the position being labelled. No amount of tactical reading answers it — the
trunk has to carry something like a *plan* to predict it at all. That is exactly the
faculty you say is missing, and it is the same trick that bought AlphaGo (territory) and
KataGo (ownership, score) far more than their parameter count suggested.

It is cheap in every dimension that matters. The targets are read off the finished board
once per game and stored as **30 bits — 4 packed bytes — per position** (2 MB across a
500k buffer, against 726 MB of states). The head is two Linears, 0.4M of 7.4M parameters.
The search never evaluates it, so visit counts and therefore policy targets are untouched.
And the weight is deliberately small: its job is to shape the trunk, not to compete with
the policy.

### 2. 1024-simulation policy targets, for *less* search than run5 spent

Playout-cap randomization, from KataGo. Each self-play **move** independently draws:

- with probability 0.25 — a **full** 1024-simulation search with root noise, whose visit
  distribution is recorded as the policy target;
- otherwise — a **cheap** 256-simulation search with no root noise. The position still
  enters the buffer (its value, margin and final-wall labels come from the *end of the
  game*, not from the search) but with **no policy target at all**.

Expected simulations per move: `0.25 x 1024 + 0.75 x 256 = 448`, **below run5's flat 512**.
So the policy targets get twice run5's depth and the game volume is not paid for. 512
simulations resolve an exchange; a two-round plan needs more, and the policy target is
precisely what the net imitates.

The masking is done properly, which matters more than it sounds: a cheap position's policy
row is zeroed *and masked*, and the policy loss is a masked mean. A zero row already
contributes no gradient — but it would still divide the batch mean, so without the mask
the policy loss (and the effective learning rate on that head) would silently shrink by
whatever fraction of moves were cheap. `tests/test_pcr.py` pins this by asserting that
adding 8 masked rows to a batch changes neither the loss nor a single gradient.

### 3. Twice as many determinizations at each round boundary

`chance_children` 4 -> 8. Every round boundary is a chance node re-sampled from a handful
of guesses at the refill, so a plan that pays off two rounds later is averaged over four
guesses at each of two refills — noise on exactly the comparisons a long-horizon plan has
to win. Doubling the sample halves that variance.

**And a finding on the way: the "mean" chance backup you might have expected to have to
write already exists, by construction.** There is no chance-node object and no averaging
step anywhere in the search — but the `(N, W)` counters live in the *parent*, so every
simulation through a refill edge adds to the same pair whichever determinization it landed
in, which makes `Q = W/N` exactly the visit-weighted mean over the sampled subtrees. So no
code changed. `chance_backup: "mean"` is now a config key that *names* the behaviour (and
rejects anything else), and `tests/test_train_mcts.py` demonstrates the identity
numerically — running the simulations one at a time, recording which determinization each
took, and checking the pieces account for every visit and all of the value — rather than
leaving it as a claim in a docstring.

### Measured, on this M3 Pro, **with run5 still training on the same machine throughout**

One driver, one full 128-game wave each, run5's config and run6's back to back so both
carry the same competing load:

| | games/min, 1 driver | projected at 6 | positions/s | evals/move | policy targets | MPS/driver |
| --- | --- | --- | --- | --- | --- | --- |
| run5 (flat 512) | 17.3 | **104** | 4,046 | 257.1 | 100% | 136 MB |
| run6 (aux + pcr + cc8) | 13.4 | **80** | 3,085 | 254.2 | 27.6% | 136 MB |

**run6 runs at 77% of run5's throughput, and 80 games/min finishes 60k games in about
12.5 hours.** Two things in that table are worth reading twice:

- **evals/move is essentially unchanged — 254 against 257.** Playout-cap randomization
  plus tree reuse land almost exactly where run5's flat 512 did, which is the whole point
  of the design: the 1024-simulation policy targets are *free* in evaluation count. The
  23% that is lost is the 5.7% bigger net (GPU share of the wall clock goes 54% -> 66%)
  and the deeper trees costing more Python per descent — not more network calls.
- **the policy-target rate is 27.6%**, against the 25% the config asks for. The excess is
  forced moves (one legal action, no search, always a real target), and it is a nice
  independent check that the schedule is doing what it says.

**MPS memory is identical at 136 MB per driver**, so 6 drivers sit under a gigabyte.

**What `chance_children = 8` costs, measured properly.** The full-wave run could not see
it — at 32 concurrent games the driver spends 85% of its wall clock inside the forward
pass, and the run-to-run spread from the competing load was ±30%, so the first attempt
came out *faster* at 8 than at 4, which is obviously noise. Measuring the search alone
with a stub evaluator (12 near-round-end positions, 1024 simulations, three repeats):

| | ms per 1024-sim search | nodes per search |
| --- | --- | --- |
| `chance_children = 4` | 42.9 (42.2-43.8) | 1,025 |
| `chance_children = 8` | 44.8 (43.9-45.7) | 1,025 |

**+4.4% of the Python descent, and exactly zero extra memory.** The node count is
identical because the tree size is bounded by the *simulation count*, not by how many
refills each chance edge samples — 8 determinizations means the same nodes distributed
over more subtrees, not more nodes. The descent is about a third of a batched driver's
wall clock, so end to end that is ~1.5%.

### Two bugs found and fixed while measuring, both worth knowing about

**An evaluation game that never ended would have killed the run.** The arena's move
ceiling was 2000 and it **raised** on reaching it — inside a `pool.imap` worker, which
propagates out of the Elo evaluation and fails the whole training run. A stalled game also
costs 100 network calls a move for 2000 moves before it gets there. The ceiling is now 400
(`mcts.MAX_GAME_MOVES`, what self-play has always used, and seven times a real game's ~54
moves) and a truncated game is **scored as a draw**, exactly as in self-play, with the
count surfaced in `elo.jsonl` and the eval log line so it can never be silent. Related: the
stall breaker now lives inside `select_play_action` instead of being each caller's job to
remember — and it now overrides the **margin tie-break** too, which was the one
deterministic pick randomness could not reach (`decisive_action` ignores visit counts among
equally-winning children, so raising the temperature never touched it).

To be clear about what I did *not* find: I measured real games first, and they are fine.
run5 vs run5, run5 vs run3, run3 vs run3, and run5 vs random/greedy/heuristic at 100 sims
are all **5 rounds and ~54 moves**, every game. The backstop is a backstop.

**"A batched game is bit-identical to the sequential one" is not quite true, and never
was.** Chasing an equivalence-test failure I could not explain, I found it reproduces
identically on pristine `main`, at seeds the existing test does not use. The cause is not
the search: **CPU matmul is not batch-invariant.** The same position evaluated alone and in
a batch of 40 differs by ~4e-8 — nothing to a *value*, but PUCT is a *ranking*, so once in
a while it flips a comparison and two trajectories part. The honest statement, now written
into `selfplay_batched.py` and pinned by a test, is: exact at `games=1`, exact per game
given identical evaluations (the bookkeeping itself adds nothing — that is the property
that actually matters and it is directly tested), statistical at `games=128` on CPU as much
as on MPS. It costs training nothing: every trajectory is a legitimate game.

### To launch it

`configs/run6.json` is ready except for the same one hole run5 had, marked in
`_note_anchors`: **run5's best checkpoint has to be re-rated before it is used as an
anchor**, because the maximum of ~80 noisy ratings overstates the truth. The exact gauntlet
command is in the config note. Everything else — pretraining from run5's buffer with the
aux targets masked out, margin native, anchors pinned to the same scale — is set.

## 2026-08-16 — run5 is ready to launch: **self-play now batches onto the GPU, so the net could get 4x bigger**

run3 and run4 both flattened out around +2290 and stayed there for 30k+ games. The browser
profile already said the ceiling is the net, not the search (92% of a think is the net; an
infinitely fast engine would buy +2%). The reason a bigger net was unaffordable was not the
net — it was that **self-play asked for one position per forward pass**, on a CPU thread, in
each of 8 processes. That is the cheapest thing a GPU-shaped machine can possibly do.

### What changed

`ludometer/train/selfplay_batched.py`: `selfplay_games` games are searched **concurrently**
inside one process, every tree's next leaf goes into **one tensor**, one forward pass answers
all of them, and `workers` such drivers run side by side. It is selected by one config key
(`"selfplay": "batched"`); every old config still says `"workers"` and is bit-for-bit
unaffected.

The important property, and the one the tests pin down: **each tree still searches strictly
sequentially.** Batching across games changes the schedule, not the search — a batched game is
*bit-identical*, state for state and visit for visit, to the game the run1-run4 engine would
have played from the same seed. Within-tree virtual-loss batching (the browser's trick, ramp
included) is implemented too, but run5 leaves it off: with 128 concurrent games there is
already a full tensor to fill without paying its price in search quality.

### Measured, with run4 still training on the same machine (so all of it is pessimistic)

| | positions/s | games/min |
| --- | --- | --- |
| old path, 8 CPU workers, run4's 1.81M net | ~10,400 (5,800 measured under that load) | 44-47 |
| batched, 6 drivers x 192 games, **same net** | **29,400** | ~130 |
| batched, 6 drivers x 128 games, **run5's 7.04M net** | **14,300** | ~63 |

So the bigger net runs self-play at the positions/s the old path got with a net a quarter of
the size — and *faster* in games/min than run4 is managing right now.

Two findings worth keeping:

- **Python's garbage collector was costing 40%.** 128 concurrent 512-simulation trees keep
  ~100k node objects alive, and CPython's generation-2 collection walks every one of them
  every ~70k allocations. Pushing generations 1 and 2 out for the duration of a self-play
  batch took one driver from 2,803 to 4,255 positions/s. (Fully disabling the collector gives
  4,466; not worth betting the process on nothing anywhere making a cycle.)
- **Cloning the engine's RNG was costing 20% of the search, in every path.**
  `AzulState.clone` built a `random.Random()` — which seeds itself from the OS *twice* — only
  to throw that away with `setstate`. Allocating without seeding is bit-identical and takes
  clone from 30.1 to 11.0 us. Sequential self-play got 25% faster too, for free.

### run5's net: 7.04M parameters, and *where* they go is the point

run3/run4 put **4%** of their weights in the relational trunk (one attention layer, width 96)
and 58% in a pooled MLP that only ever sees a single vector. run5's trunk is 4 layers of width
256, 8 heads, 3x feed-forward = **2.6M, a 35x increase** in the only part of the net that can
compare two entities — which is what Azul decisions actually are. Width beats depth here by
measurement, not taste: with 22 tokens each attention layer is kernel-launch-bound on MPS, so
4 wide layers beat 6 narrow ones by 15-25% throughput at equal parameters.

The ONNX exporter needed no changes and round-trips it to 1e-7: **28.3 MB, 24.7 MB gzipped**
(against ~6.5 MB today). That is a real decision for the browser later — not made here, and
nothing is deployed.

### To launch it

`configs/run5.json` is ready except for one hole, marked in `_note_anchors`: run4's best
checkpoint has to be **re-rated** before it is used as an anchor, because the maximum of ~80
noisy ratings overstates the truth (same winner's-curse correction run3 needed). The exact
gauntlet command is in the config note.

## 2026-08-16 — Morning report: run4 at +2181 after 10k games, endgames now decisive

- **run4 overnight**: pretrained start +2092 → currently **+2181 at 10,240 games** (peak +2219),
  after a brief dip while run3's pretraining data aged out of the replay buffer. It beats
  run2's best 90% and trades roughly evenly with run3's re-rated best — on a *stricter ruler*
  than before (honest re-rated anchors + calibrated greedy/heuristic pins), so numbers read
  slightly lower than run3's inflated late-night ones.
- **Your endgame fix is in and training**: the margin head + lexicographic play ("win first,
  win big second"). This mostly changes *style*, not headline Elo — the sloppy-looking
  endgame moves should be gone. Judge it at the board.
- **To play run4 explicitly today**: the "Strongest (auto)" dropdown still points at run3's
  ckpt-020992 (its recorded +2336 spike still tops the log even though its true strength is
  +2255). Until run4's recorded rating passes that, pick run4 by hand:
  `mcts:runs/run4/checkpoints/<latest ckpt>.pt?sims=400&think=5` — it will play decisively;
  the public site also auto-detects the margin head the moment I export a run4 model.
- run4 continues toward 60k games. Next decision point: whether the +2255-level plateau breaks
  with more games, or whether run5 should be the "bigger net on GPU" step the WebGPU
  measurements opened up.

## 2026-08-16 — "Would a faster language help?" — **no, and here is the profile**. The browser player now uses your GPU

You asked whether re-implementing the search in a much faster language, to explore more
nodes, would get us to superhuman. I profiled a real 5-second think instead of guessing.
**A faster engine language buys +2%. The answer was never the language — it was the batch
size.** The player now searches **6× more positions per second** on any browser with WebGPU,
and 1.8× more on the ones without.

### Where a 5-second think actually goes

One search from a real midgame position, node + the vendored WASM runtime, every call timed:

| | share of the wall clock |
| --- | --- |
| `session.run` — the neural net | **91.6 %** |
| everything the "engine" does (clone, apply, legal moves, encode) | **2.3 %** |
| the search itself (PUCT select, backup, tensor alloc, softmax, awaits) | 6.1 % |

7,154 simulations in 5.00 s = **1,431 positions/s**, of which 0.64 ms per position is the net.
The same search with the net replaced by a stub that returns instantly runs at **34,610
positions/s** — 24× faster. So an infinitely fast Rust/C engine, with the same net, would take
us from 1,431 to about 1,464 positions/s. **+2.3%.** Not a rounding error away from nothing,
but close enough that it would be the worst-value week of work on this project.

That is the honest answer to the question: the JavaScript is not the problem. **We are
inference-bound, and we were inference-bound in the dumbest possible way — one position per
forward pass.**

### What was actually being wasted

A 1.7M-parameter net evaluated one position at a time uses a fraction of the machine. Raw
`session.run` throughput in Chrome on the M3 Pro:

| batch | WASM (CPU) | WebGPU |
| --- | --- | --- |
| 1 | 2,079 pos/s | **249 pos/s** |
| 8 | 3,634 | 1,935 |
| 16 | 3,969 | 4,555 |
| 32 | 4,177 | 8,202 |
| 64 | 4,167 | **16,537** |
| 256 | 4,108 | **49,797** |

Read the WebGPU column carefully, because it is the trap. **A GPU dispatch costs ~3.9 ms
whatever you put in it.** At batch 1 WebGPU is *eight times slower* than the CPU. Anyone who
"adds WebGPU" to this player without touching the search would ship a 6× regression and a
2.9 MB download to pay for it. That is the measurement that decided the whole design.

### What shipped

**Batched search with virtual loss.** The search now gathers up to N leaves per forward pass:
each descent lays a virtual loss on the edges it walks so the next descent is pushed
elsewhere, all N are evaluated in one dispatch, then the real values are backed up and the
assumed losses removed. The bookkeeping is exact — a finished batch leaves the tree in
precisely the state the same leaf evaluations would have left it in sequentially, which the
tests assert directly (visits sum to the simulation count; no Q escapes [-1, 1]).

**A second onnxruntime build, downloaded only if it will be used.** The worker feature-detects
`navigator.gpu` + JSPI and loads the WebGPU runtime; anything else — Safari today, Firefox
today, an old Chrome, a machine with no adapter — takes the WASM path that shipped before, and
never requests the WebGPU binary at all. A WebGPU session that creates and then fails on its
first dispatch also falls back, because the warm-up evaluation is inside the try.

Full search, Chrome, 4-second budget, same position:

| | positions/s | vs what was deployed |
| --- | --- | --- |
| WASM, batch 1 (what was live) | 1,639 | — |
| WASM, batch 16 (now live) | 2,943 | **1.8×** |
| WebGPU, batch 64 (now live) | 10,117 | **6.2×** |

The page says which one it got: the engine bar now reads "searching on your GPU (WebGPU)" or
"…your CPU (WebAssembly)", and the about line carries the batch size and the machine's own
measured positions/s once the first search has run.

### The part I nearly got wrong

More positions is only worth having if it wins games, so I checked instead of assuming — and
the first version of this was **much weaker**. At equal simulation counts, a flat batch of 64
lost **3–17** to the old one-at-a-time search. With nothing in the tree yet, virtual loss
shoves all 64 descents down 64 different early branches and an 800-simulation search never
recovers from spending its first eighth that way.

The damage is a function of *batch ÷ tree*, so the batch now starts small and grows with the
tree (never more than a sixteenth of it, with a floor of 8 on the GPU where a batch of one is
just a wasted dispatch). That took the same equal-node match from 15% to 25% — still a real
per-simulation cost, which is the honest way to describe batching: **you are buying quantity
at a small price in quality.**

The trade only matters if the quantity wins. Head to head in Chrome, 1.5 s a move, the new
player (WebGPU, batch 64) against exactly what was deployed yesterday (WASM, batch 1):
**7 wins, 3 losses, 2 draws — 66.7%, about +120 Elo**, at 3.85× the
positions/s (10,743 vs 2,788). Twelve games is a wide error bar and I will not pretend
otherwise, but the sign is not in doubt and the node count is measured, not estimated.

### Payload

Nothing changed for a visitor without WebGPU. A visitor with it downloads the WebGPU runtime
instead of the CPU one, which is **+0.24 MB gzipped** (3.66 MB vs 3.42 MB) on top of the 6.5 MB
model. I picked the JSPI build for exactly this reason: the other two WebGPU builds
onnxruntime ships are +2.5 MB and +2.9 MB gzipped for the same feature, and both measured
slower (16.5k vs 11.2k vs 9.5k positions/s at batch 64). The repo carries both runtimes —
35 MB in `web/player/` — but no one downloads both.

Safari and Firefox have WebGPU but not JSPI yet, so they stay on the CPU path today and will
flip to the GPU on their own, with no change here, when they ship it.

### run4's margin head is already wired up

You said run4 gains a third output that predicts the score gap. The deployed player now
**feature-detects it**: it reads the session's output names, and if `margin` is there the
search averages it up the tree alongside the value and picks the root move
lexicographically — among the moves whose win-Q is within 0.03 of the best, play the one with
the biggest expected gap. A win is never traded for points, and a move the search barely
looked at cannot define "the best" (a candidate needs a tenth of the top child's visits — a
one-visit edge with a lucky Q of +1.0 would otherwise drag the window). Two-output models keep
the current behaviour bit for bit.

So when you export run4, the live page starts playing decisively **without a JavaScript
redeploy**. It is tested against two hand-built ONNX graphs (`test/fixtures/toy_*.onnx`, three
outputs and two) that route their input straight to their outputs, so the expected answer is
exact rather than approximate.

### What this means for superhuman

The ceiling moved, but it is worth being clear about which ceiling. At 5 seconds a move the
player went from ~8,000 positions to ~50,000 on a WebGPU machine. In MCTS that is worth
roughly two and a half doublings of search — real, and the sort of thing that shows up as the
opponent no longer missing tactics, but it is not a different kind of player. **The remaining
gap to superhuman is in the net, not in the search budget**, and the profile says so: we spend
92% of the clock asking a 1.7M-parameter net what it thinks, and its answer is the thing that
is not yet superhuman.

Two things follow that are worth knowing before spending a week anywhere:

- **A bigger net is now affordable on the GPU and not on the CPU.** At batch 64 the GPU is
  doing 16.5k positions/s on a net that only needs 4.2k to keep the old search fed. There is
  roughly 4× of net capacity available for free on WebGPU machines before the search slows
  back to where it was. If run5 wants to be 4× bigger, the browser can already carry it —
  though the CPU fallback could not, so that would become a two-model decision.
- **The engine language question comes back, but only later.** At 1,431 positions/s the engine
  was 2.3% of the clock. At 10,117 it is closer to 25%, and if the net were ever made much
  faster still it would be the wall. So "rewrite it in Rust" is not wrong forever — it is
  wrong *now*, by a factor of forty, and it becomes worth measuring again only after the net
  stops being the bottleneck. WASM-compiled Rust for the engine, not a rewrite of the page.

Gates: engine fixtures, WASM parity against torch, **WebGPU parity against the same torch
reference** (value agrees to 8×10⁻⁷, and it never ranks a different move first), the new
margin/batching test, selfplay, and the headless-Chrome page test all green before the deploy.

## 2026-08-16 — run3 retired (~+2255 true), run4 training with a "win big" head

- **run3 final**: 29,000 games, ~12.9 h. Its headline +2336 was a measurement spike: I re-rated
  the top checkpoint with 240 fresh games and its honest strength is **+2255 ± 40** ("winner's
  curse" — pick the max of 55 noisy ratings and you overpick luck; the methodology page
  explains this). Still ~+235 above run2, and the checkpoint your wife lost 4/5 against.
- **run4 is training now**, aimed at your two asks:
  1. **Decisive endgames**: the net gains a third output that predicts the final score *gap*.
     The search still maximizes winning first, but among near-equal winning moves it now plays
     the one that wins by more — no more "lazy but technically correct" endgame moves.
  2. It warm-starts from run3's 500k positions (a nice trick even recovered exact margin
     labels from run3's stored values), so no knowledge is lost.
- Ladder hygiene, from the methodology audit: greedy/heuristic are now *pinned* at their
  calibrated ratings so eval games against them actually inform the fit, and run4's anchor for
  run3 uses the honest re-rated +2255, not the spike. run4's curve is a slightly better ruler.
- Your win vs ckpt-020992 and the 4/5 session are logged in `runs/human_benchmarks.jsonl` —
  keep the results coming, they're our only human calibration.

## 2026-08-16 — **Methodology page**: how the AI learns, explained end to end

There is now a "How it works" button in the dashboard header. It opens
`web/methodology.html`, a second generated page that explains the whole system to
someone who knows how to code and nothing about RL — every term defined at first use,
plus a glossary. Source of truth is `docs/METHODOLOGY.md`; `web/make_dashboard.py`
renders it with the dashboard's look, a table of contents, and four hand-drawn inline
SVG diagrams (the training loop, one MCTS simulation, run3's 22-token network drawn
from `net2.py`, and the `runs/` data layout). It regenerates with the dashboard, so
the watcher already keeps it current.

**Every number in it came out of the logs, not out of my head.** The run comparison
table is built from `runs/*/config.json` + `status.json` + `elo.jsonl` at build time,
so it can never drift; I verified all 18 of its cells against the raw files.

Three things I found while writing it that you should know, because they are not
written down anywhere else and two of them affect how you read the curves:

- **`greedy` and `heuristic` contribute nothing to a checkpoint's Elo.** Only pairings
  involving the candidate are in the fit, and only `random` and the pinned prior-run
  checkpoints are anchored — which leaves greedy and heuristic as free parameters on a
  star graph, so the fit just reproduces their observed win rates and passes no
  information back. run1's first rating, +125.3, is reproduced to 0.1 Elo by the
  `random` edge alone. They are readable milestones, not measuring instruments.
- **The scale is now a ratchet, and run3 is standing on it.** Once you beat `random`
  100% of the time that edge carries no Fisher information, so a rating is effectively
  "the previously published checkpoint's Elo plus the score share against it". And
  `eval_frozen` pins the *all-time strongest* checkpoint — a max over ~50 noisy
  estimates, i.e. textbook winner's curse, biased high. run3's `ckpt-020992` (+2336)
  has been the pinned anchor for 7k games, later checkpoints score ~0.46 against it,
  and the curve has not exceeded it since. "Learning slowed" and "the ruler ran out"
  look identical in this data.
- **The concave shape is the real result so far.** run1 104.5 → 45.7 Elo/1k games
  (first half → second half), run2 118.4 → 12.4, run3 8.8 → 5.6. Same shape across a
  1.0M MLP at 160 sims, a 3.3M MLP at 256 sims, and a 1.7M attention net at 512 sims
  with a warm start. That the shape survived every change to the learner is evidence it
  belongs to the game plus the method rather than to any particular net.

The doc says plainly what run4 would have to do to separate those last two: a gauntlet
of run3's checkpoints against each other and against run1/run2 at play-time settings,
anchored on run1's +2014.

I touched only `docs/METHODOLOGY.md`, `web/make_dashboard.py`, `web/methodology.html`
and this file. Nothing under `runs/` or in `ludometer/` was modified — the trainer kept
running throughout.

---

## 2026-08-15 — GUI v3, and **why you never saw the animations**

You asked for tile animations repeatedly, and kept telling me the tiles still moved
instantly. Every agent before me implemented the flights, verified them in headless Chrome,
and reported them done. You were both right. Here is the actual cause.

**Your Mac has "Reduce motion" switched on.**

```
$ defaults read com.apple.Accessibility ReduceMotionEnabled
1
```

Safari and Chrome forward that to pages as `prefers-reduced-motion: reduce`, and the GUI
honoured it in **two** independent places:

1. `web/play/ui/animate.js` — `flyTiles()` returned immediately when the flag was set, and
   `sleep()` collapsed to zero, so no tile ever left its square;
2. `web/play/ui/board.css` — a blanket `@media (prefers-reduced-motion: reduce)` rule forced
   `transition-duration: .001ms !important` on *every element on the page*, which would have
   killed the flights even if the JavaScript had run.

So the code was doing exactly what it said, there was no setting anywhere that could turn
the animation back on, and the tests passed because
`web/player/test/browser.test.mjs` set `prefers-reduced-motion: no-preference` before it
looked — a line with a comment explaining that the page "honours that by not animating at
all". The blind spot was documented and then tested around.

Both gates are gone. **Motion is now governed by one thing: a setting in the page.**

### What shipped, in the local GUI *and* on the public site

- **A gear under the status band** opens an inline panel (no pop-up — this page still has
  none) with `Off / 0.5× / 1× / 2×`, default **1×**, remembered in `localStorage`. When your
  OS asks for reduced motion the panel says so out loud and tells you this switch decides —
  it does not quietly obey.
- **Every tile movement animates**: the tiles you take, *the rest of the factory falling into
  the middle* (the one you called out), the first-player marker, full pattern lines onto the
  wall at round end, the floor line into the lid — and the AI's moves the same way.
- **Both** of the AI's moves when a round boundary puts it on move twice. That gap was real:
  its second move starts from a table the engine scored and refilled *inside* the first move,
  a position the page had never been given. Every move now reports `state_before`, so the
  refilled table is drawn and then the second move plays out.
- **Filled vs empty, rebuilt.** This is your wife's complaint and she was right: every empty
  wall square used to be a pale tint of the colour that belongs there, so each board was
  forty pastel squares with five real tiles hidden in them. New rule — **hue means
  occupied**: a tile is saturated, rimmed and raised; every empty square is the same neutral,
  recessed well; the wall keeps its pattern as a thin *outlined* diamond in the tile's ink.
  Glazes nudged closer to the physical tiles. Before/after screenshots are worth a look.
- **One theme file.** Every colour in the game — five glazes, boards, slots, dishes, panels,
  buttons — is a custom property in `web/play/ui/theme.css`. `board.css` and both page shells
  contain **zero** hex literals now, and a test fails if one appears. `THEMING.md` explains
  how to write a skin and ships one (`dusk`) as a worked example, so restyling is one file.
- **Move navigation like chess.** ← / → step through every position, `End` (or the **Latest**
  button) returns to play, and the band reads *"Viewing move 12 of 31"*. Pure client-side
  replay of positions the page already holds — no request, no re-search, live game untouched.
  Works on finished games.
- **Pictographic move log, newest first.** The tiles that moved are drawn as tiles in the
  board's glazes (*▪▪▪ factory 3 → row 5*); only the places stay as words. Every entry drawn
  identically, including the newest — the status band is where "now" lives.

### How it is proved this time

`web/play/test/gui.test.mjs` drives **both** tables in headless Chrome — it starts the Flask
GUI itself — and runs every motion check **twice, once in each `prefers-reduced-motion`
state**. It counts flight clones and measures their transition durations: 460 ms at 1×,
230 ms at 2×, 920 ms at 0.5×, none at Off. It also walks the history with the arrow keys,
checks the log is glyphs and newest-first, asserts filled and empty squares differ in
*computed colour* (not class name), and flips `[data-skin]` to prove the palette really is
centralised. On the last run: **31/31 and 29/29 moves animated, 3 and 2 double AI moves
respectively**. `tests/test_gui.py` is at 51 tests (257 across the suite), and the two `ui/`
copies must be byte-identical or the suite fails.

### One thing to know

If you *liked* not having animations, the gear now has an `Off` preset and it sticks. The
difference is that it is your choice rather than a side effect of an accessibility setting
you turned on years ago for something else.

The public site is redeployed and now serves **`run3/ckpt-007680`, +2198 Elo** (was
ckpt-001024, +2185) — `deploy_player.sh` picks the strongest checkpoint on disk each time.

---

## 2026-08-15 — Public player now matches the new GUI, and serves a **run3** checkpoint

<https://remifabre.github.io/ludometer/> is redeployed with the redesigned table **and** a
stronger opponent.

- **Now serving `run3/ckpt-001024`, +2185 Elo** (was `run2/ckpt-023040`, +2020). The structured
  net is also much smaller: **6.8 MB of ONNX, 1.68 M parameters** (was 13.3 MB / 3.32 M), so
  the first-visit download roughly halved. Live `model_meta.json` confirms it; browser parity
  re-checked at export (policy 1.03e-5, value 1.40e-6, tol 1e-4).
- **The `web/play/ui/` kit was lifted in as-is** — the seven files are byte-identical copies in
  `web/player/ui/` (gh-pages has to be self-contained), and the page's own glue
  (`web/player/js/app.js`) is now only the in-browser back end: it owns the `GameSession`, asks
  the Web Worker for moves, and tells the status band what to say. +60 KB of payload on a 19 MB
  site.
- So the hosted page now has exactly what the local GUI has: **no pop-ups at all** (the
  overlay/sheet *and* the toasts are gone — messages go into the status band's detail line),
  the big status band with its filling clock (*"AI is thinking — 2.1s of 5s · 12,300
  positions"* — it counts your own CPU's work), **twin identical boards** side by side, the
  flat move log below them, straight-line tile flights for **both** players plus round-end
  tiling, and **inline** round/final scoring under the boards.
- **Coach mode made it in.** `mcts.js` grew one accessor (`rootChildren()`, the port of
  `RootStatsMCTS.root_children`) and the worker a `rate` message, so the delta is the *same*
  definition as the local GUI: `Q(your move) − max Q over explored children` at the root of a
  fresh search, capped at 3 s (2 s default), `unrated` when the search never visited your move.
  It is off by default and needs no server, so it works on the hosted page too.
- **Kept**: the model-download progress strip, the think-time selector, the
  "runs entirely in your browser" badge, the hint button and the "how this works" panel.

Gates, all green: `engine.test.mjs` (9,384 assertions), `parity.test.mjs` against the new
checkpoint, `selfplay.test.mjs`, and `browser.test.mjs` — which I extended to assert the
redesign's promises: **zero** overlay/dialog/sheet/toast nodes in the DOM, the two boards side
by side and identical to the pixel, tiles actually flying (it counts clones in the flight
layer), and a **whole game** played out to the inline final scoring panel with the board still
on screen. It passes both locally and with `--live`.

Two things worth knowing: headless Chrome asks for reduced motion by default, so the test now
emulates `no-preference` before checking that anything animates; and the same known gap as the
local GUI remains — when the AI moves twice across a round boundary, only its first move is
animated.

## 2026-08-15 — Local GUI redesigned: no pop-ups, twin boards, inline scoring, **coach mode**

`uv run ludometer-gui` looks quite different, from your notes after playing more games.

- **Every pop-up is gone.** No round-end modal, no game-end modal, no overlay element left in
  the DOM at all (a headless-Chrome check asserts that). What replaced them is a wide
  **status band** at the top that always answers "what is happening": *"Your turn — pick a
  colour"*, *"AI is thinking — 2.1s of 5s"* (the band's own bar fills as the budget burns),
  *"AI took 3 red from factory 2 → row 4"*, *"Round 3 scoring"*, *"You won 74–68"*, with the
  running score on the right.
- **Both boards side by side, identical** — you left, AI right, same size, same rows, same
  code path (one `createBoard()`, `interactive` is the only difference). The move log moved
  *below* the boards and every entry is now drawn identically; the last entry is no longer
  styled like a title.
- **Scoring is inline.** The round tally and the end-game bonus breakdown (rows ×2, columns
  ×7, colours ×10, per player) appear in a panel under the boards, so the final position
  stays on screen and inspectable instead of being hidden behind a sheet.
- **Everything that moves, moves.** Straight-line flights, ~0.5 s per group: factory →
  pattern line/floor (**your own moves too**, which was the biggest omission), the factory's
  leftovers → the middle, full pattern lines → the wall at round end, floor line → the lid.
  A plain turn locks input for ~1.9 s (your flight + the AI's + a beat), a round-boundary
  turn for ~3.3 s; measured in headless Chrome.

**Coach mode** is the new toggle, above the move log. It scores your moves with *the AI's own
evaluation* — nothing invented:

> Before your move is applied, the same PUCT search the opponent plays with (same checkpoint,
> same net, same c_puct) runs on your position, and the log entry shows
> `delta = Q(your move) − max Q over explored children` on the net's own [−1, 1] scale.
> `0.00` = you played its move; `−0.06` = it values yours six hundredths of a win worse. From
> `−0.02` down, the entry also names the move it preferred. A move the search never visited
> is **"unrated"**, never a fake number.

Turn it on mid-game whenever you like; it needs a searching opponent (greyed out against the
baselines). Decisions worth knowing:

- **Where the time goes.** The rating runs inside `POST /api/act`, before the move is
  committed — it has to see the position you chose from. Budget: your opponent's think time,
  but **capped at 3 s** (default 2 s when the opponent replies instantly), so a 10 s opponent
  does not double every turn. The page shows a *"rating your move — 1.2s of 2s"* clock, then
  your tiles fly as usual.
- **It cannot disturb the opponent.** `ludometer/gui/coach.py` subclasses MCTS
  (`RootStatsMCTS`) purely to read the root's child statistics back out, and runs its own
  tree over the opponent's *evaluator* — same weights, separate search, Dirichlet noise off,
  tree reuse off. **`ludometer/train/mcts.py` was not touched**, which matters while run3 is
  importing it live.
- **Never costs you a move**: any failure (no torch, a bad checkpoint, a search that ran out
  of time) comes back as an `unrated` verdict with the reason, and the move still lands.

Also: `web/play/ui/` is now a clean, framework-free kit (board renderer, status band, log,
scoring panels, flights, shared CSS) that takes state JSON and has no idea a server exists —
`web/play/ui/PORTING.md` is the note for whoever ports it into the GitHub Pages player next.
Tests: `tests/test_gui.py` is green (46 tests, coach mode included at 0.1 s budgets), plus a
scripted API game with the coach on every move.

One known gap, deliberately left: when the AI plays twice in a row across a round boundary,
only its first move is animated — the page never observes the intermediate position after the
refill. The move still appears in the log and on the board.

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
JS/CSS/HTML. All vendored, no CDN, no third-party *code* at runtime. (Since 2026-08-16
there is exactly one third-party *request*: the anonymous, cookie-free GoatCounter tally
ping — Rémi's call, publicly readable at faience.goatcounter.com, and disclosed in the
game's About panel.) The page streams the
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
