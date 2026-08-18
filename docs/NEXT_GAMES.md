# Next games — the implementation goal

Status: **agreed 2026-08-18**, not started. This document is the brief. It is written to
be executed by someone (or some agent) who has not seen the conversation that produced it,
so it states the decisions, the traps, and the acceptance criteria rather than the
reasoning that got there.

Read `docs/METHODOLOGY.md` first for what the project measures and `docs/DESIGN.md` for the
run-directory schemas. `NOTES_FOR_REMI.md` is the running journal; append to it, newest on
top, when a phase lands.

---

## State of play at hand-off (2026-08-18, 20:15 UTC)

**`runs/uno1` is training right now** and should finish on its own. Nothing needs doing
unless it stops early.

- Started 19:13 UTC, ~44k of 120,000 hands done, ~1,800 hands/min, ETA ~1.5 h.
- Launched with `caffeinate -i` so the Mac stays awake; the console log is mirrored to
  `runs/uno1/train.log` by a `tail -f`. The authoritative data is `runs/uno1/elo.jsonl`,
  `train.jsonl` and `status.json`, as for every run.
- The dashboard watcher (`python web/make_dashboard.py --watch 30 --quiet`) is running and
  rewrites `web/dashboard.html` every 30 s.
- Curve so far: +87 at 0 hands → +225 at 4k → +367 at 16k → a flat stretch around
  +300-370 to 27k → **+465 at 41k**, now beating `uno:heuristic` 0.59 and `uno:greedy`
  0.34. The flat stretch was a pause, not the ceiling — a reminder that four eval points
  at +/-45 Elo each is not a verdict. Read the finished curve before concluding anything.
- If it died: `uv run python -m ludometer.train.run --resume runs/uno1`.
- Two earlier, *discarded* uno1 attempts are in the session scratchpad under
  `uno1-buggy` and `uno1-boundary-bug`. They are the two value-function failures in §1.
  Do not treat their curves as data.

Nothing else is in flight. The Azul runs are all stopped; `run5` is the strongest at
~+2383 and was still climbing when it was paused.

## 0. Where the code already is

Two games exist. Everything above the rules engine is duck-typed on a state object, so a
third game is additive.

| piece | file | note |
|---|---|---|
| game registry | `ludometer/games.py` | `"game": "azul" \| "uno" \| "uno_hand"` in a config; absent means Azul |
| Azul engine | `ludometer/azul/engine.py` | 2 players, perfect information |
| Uno engine | `ludometer/uno/engine.py` | 2 players, hidden hands, match to 500 |
| baselines | `ludometer/agents/`, `ludometer/uno/agents.py` | registered as `random` / `uno:greedy` / … |
| search | `ludometer/train/mcts.py` | calls `state.is_stochastic/determinize/chance_key/fingerprint/search_root` |
| trainer | `ludometer/train/trainer.py` | `game` trains, `eval_game` rates (may differ) |

**A new game must provide**, as methods on its state class: `new_game(seed, **options)`,
`clone`, `legal_actions`, `is_legal`, `apply`, `outcome`, `encode`, `is_terminal`,
`current_player`, `scores`, `round_index`, `wall_summary(p)` (15 zeros if it has no
auxiliary target), plus the five search hooks above and the class attributes
`ACTION_SPACE` / `ENCODED_SIZE`. Then one entry in `GAMES`, baselines in the agent
registry, and a config. Nothing else changes.

---

## 1. Traps already paid for — do not rediscover these

**The value function must be bootstrappable across every boundary the search can cross.**
Uno cost two failed attempts here, both of which produced a flat curve that looked like a
statement about Uno and was a statement about our code.

1. Labelling a position by the *hand* it was played in, blended with the match result, is
   not a value function: it resets at each hand boundary. The trained net priced a
   position one card from going out at **+0.75** and the position immediately after going
   out at **+0.02**, so the search treated winning a hand as a cliff and avoided it.
2. Training on single hands and *rating* over matches fixed training and not rating: the
   search inside a match still crossed hand boundaries. The same checkpoint won **41.5% of
   hands and 0.0% of matches** against `uno:greedy` — arithmetically impossible for an
   undistorted agent.

The fix in the tree now: `UnoState.search_root` truncates the search horizon to the end of
the current hand, so the tree's terminal value is the label the net was trained on.
`tests/test_uno.py::test_the_search_horizon_ends_with_the_current_hand` pins it.

**Rule of thumb for any new game:** if a game has internal scoring segments, either make a
segment the whole episode or make the value a true expected final outcome. Never blend.

**Diagnostic to run on any new game before trusting a curve** (the script that caught
both bugs lives in the journal entry; reproduce it in ~30 lines): take a checkpoint, find
positions immediately before and after every boundary in the game, and check that the
value head does not jump across it in the winner's frame.

**PIMC determinization must respect everything the mover knows.**
`search_root` redeals the *unseen* cards. Any rule that reveals a specific hidden card to
the opponent (the 7-swap below is exactly this) makes the naive "redeal the whole opponent
hand" wrong, and it will silently weaken play rather than crash.

**Forced moves dilute the policy target.** ~60% of Uno turns have one legal action; they
enter the buffer with a one-hot policy and full mask weight. It is not a bug, but it means
positions-per-game is a poor proxy for decisions-per-game. Count both.

---

## 2. Phase 1 — Uno+ (the rule-knob experiment)

**Why this first.** It is the only experiment that varies *one thing*: same engine, same
network, same search, same budget, one rule set. Everything else in the study compares
different games and inherits every confound at once. It is also the question the ludometer
is ultimately for — *did this rule change make the game better?*

**The problem being fixed, measured:** plain Uno offers **3.2 legal moves on average** and
**~60% of turns have exactly one legal action**. There is almost nothing to decide.

### Rules (all four are in; ship them together, ablate later)

Register as `unoplus` (match to 500, for rating) and `unoplus_hand` (one hand, for
training), mirroring the existing `uno` / `uno_hand` pair.

**R1 — Draw is always legal.** Today `DRAW` appears only when nothing is playable. Make it
legal on every turn. Drawing is atomic: take one card, your turn ends (you do *not* get to
play it). This keeps the action space at 61 and keeps every hand progressing, while adding
the real choice — *play now, or bank a card and keep the option*.
*Guards:* `DRAW` is illegal once your hand reaches `DRAW_CAP = 15` cards, and illegal when
deck and discard are both exhausted.
*Alternative considered:* draw-then-optionally-play, which needs a `PASS` action (62
actions) and a two-step turn. Cheap to add later if R1 proves too blunt.

**R2 — Stacking.** A `+2` may be answered with a `+2` and a `+4` with a `+4` (like on
like only — a `+4` does **not** answer a `+2`). The penalty accumulates. State gains
`pending_draw: int` and `pending_kind: int` (0, 2 or 4). While `pending_draw > 0` the only
legal actions are the matching stack cards and `DRAW`, which now means *take the whole
accumulated stack and lose your turn*. Once taken, `pending_draw` resets to 0.

**R3 — 7-swap.** Playing a `7` swaps your hand with your opponent's, then the turn passes
normally (a `7` is otherwise an ordinary number card). If the `7` was your last card you go
out and **no swap happens** — the hand ends first.

*This is the rule that carries the design's intent* (making the opponent's hidden hand
worth modelling) *and the one with the implementation trap.* After a swap you know their
hand exactly, and that knowledge decays as they play the cards. Track a per-observer
`known[card]` count; `search_root` must deal those cards to the opponent first and
determinize only the remainder, decrementing as each known card is played or swapped away.
Without this the search throws away the very information the rule exists to create, and
the experiment measures nothing. Encode `known[]` (54 floats) as an input.

**R4 — Opening hand of 9** instead of 7.

### Encoding

Start from `UnoState.encode()` and append: `pending_draw / 4`, `pending_kind` one-hot (3),
and the `known[]` vector (54). Keep everything else identical so the two nets differ only
where the games differ. Match score and hand index stay *out* (see `uno/engine.py`).

### Acceptance criteria

Measure **before** training, over 300 random games, and record the numbers in the journal:

| quantity | plain Uno | Uno+ target |
|---|---|---|
| mean legal actions per turn | 3.2 | **> 5.0** |
| turns with exactly one legal action | ~60% | **< 35%** |
| moves per hand | ~42 | report it |

If Uno+ misses those targets the rules did not do their job and there is no point training
it — tune the rules, not the trainer.

Then train with a config identical to `configs/uno1.json` except the game names and the
budget in *decisions* (not games — see §4). Compare the two curves.

**The result is interesting either way.** A higher, later-flattening Uno+ curve says the
rules added skill the way the house-rule folklore claims. A curve that is the same shape
says Uno's ceiling is set by its randomness, not by its decision space — which is a finding
about card games, not a failed experiment.

---

## 3. Phase 2 — tic-tac-toe and Connect Four (the calibrated pair)

Build them **together**; they share one solver harness and one metric, and their value is
in the contrast. Same genre, zero luck, zero hidden information, and one is obviously
shallow while the other is not. **If the ludometer cannot separate these two, nothing else
it reports is trustworthy.** That is the point of the pair.

### Engines

Trivial by the standards of this repo. Tic-tac-toe: 9 actions, terminal in ≤ 9 plies.
Connect Four: 7 actions, 7×6 board, ≤ 42 plies. Both perfect information and fully
deterministic, so `is_stochastic` is always `False`, `determinize` never fires, and
`search_root` is `clone()`. Baselines: `random`, plus a one-ply `greedy` (take an
immediate win, block an immediate loss) and a `heuristic` (greedy plus centre preference
for C4 / fork detection for TTT).

### The metric — % optimal, and it is the headline

**Decision: for solved games the headline chart is % optimal play, with internal Elo kept
as the secondary chart.**

A perfect player cannot go in the anchor pool. It never loses, and Elo against an opponent
you never beat is unbounded — the fit does not shift, it diverges. Worse, adding it would
silently re-scale what every other rating in that game means.

Instead:

1. **Build a fixed position suite, once.** Sample ~2,000 reachable positions from real
   self-play games (stratified by ply so the opening is not over-represented), solve each
   with alpha-beta over bitboards plus a transposition table, and cache to
   `data/solved/<game>_suite.json`. Solving is a one-off cost; evaluation is then a dict
   lookup. Sample from ply ≥ 8 for Connect Four to keep the one-off solve cheap.
2. **% optimal** = fraction of suite positions where the agent's chosen move preserves the
   game-theoretic value (a won position stays won, a drawn one stays drawn).
3. **Blunder rate** = fraction where it turns a win into a non-win. Report both; the second
   is the more sensitive early signal.
4. The solver may still be **rated separately** on the existing anchor pool and drawn as a
   horizontal "perfect play" reference line on the Elo chart — but only where the fit is
   finite, and never as a training anchor.

This is the same move as the "superhuman ≈ 2800" estimate on the Azul chart, except the
number is exact instead of assumed. It also gives the linearity thesis a much sharper test
than Elo can: **is % optimal linear in decisions played?** — a question with a real ceiling
at 100%.

### Acceptance criteria

- Tic-tac-toe reaches ~100% optimal within a few hundred games and then flatlines. If the
  linearity statistic does not scream *trivial* here, the statistic needs work — fix it
  before reading any other curve.
- Connect Four climbs slowly and is still short of 100% at the end of a laptop budget.
- Both games' Elo charts and % optimal charts tell the same story about *where the slope
  dies*. If they disagree, that disagreement is itself the most interesting result in the
  study and belongs in the write-up.

---

## 4. Cross-game comparison — the rules of the chart

These apply to every game in the study and exist because the naive version is wrong.

**The x-axis is decisions, not games.** One "game" is not one unit of practice:

| unit | moves | searched decisions |
|---|---|---|
| Azul game | ~52 | ~52 |
| Uno hand | ~42 | ~17 |
| Uno match to 500 | ~1,100 | ~440 |

A 26× spread. Plot the headline against decisions and keep games as a secondary panel.
**A conclusion that survives on only one axis is not a conclusion.**

*Task:* `elo.jsonl` records `games` and `t` but not positions. Add a cumulative
`positions` and `decisions` field to the eval record in `trainer.py` so this stops needing
reconstruction. Existing runs can be converted with `games × moves_per_game × searched
fraction` (Azul 52 × 1.00; Uno hand 49 × 0.40).

**The question is what the curve looks like, not whether it is straight.** The original
framing ("does a good game teach linearly?") is a hypothesis, not the goal. What is wanted
is a *characterisation* of each game's curve — how fast it rises, where the slope falls
off, how much of its own range it covers and when — from which rules and observations can
be drawn afterwards. Report the shape and let the conclusions follow; do not fit a line and
score games on R².

There is a concrete reason to distrust R² here, measured on this project's own data. Fit a
line to the first *N%* of Azul run1 and the fit gets **better the less you see**: R² =
0.996 over the first 5% of the budget, 0.975 at 10%, 0.960 at 25%, 0.929 at 50%, 0.909
over the whole run. run2 goes 0.883 → 0.686. Every curve is straight at the start, so a
linearity score mostly measures how far into the curve the run got. The informative
quantity is **where the slope dies**, which is why a run that cannot reach that region
produces no signal at all — see the chess note below.

**Never compare Elo ceilings across games.** Elo measures distinguishability, not skill.
Uno matches *amplify* a small per-hand edge (41% of hands → 15% of matches), so Uno's
match-Elo scale is stretched, not compressed. The comparable quantity is the **shape**:
normalise each curve to its own range (0 = random, 1 = that run's best checkpoint) and
plot fraction-of-the-way against fraction-of-budget. Report the R² of the linear fit *and*
the point where the slope halves.

**Which Azul run to compare against.** run1 is the primary: only run1 and run2 start from
scratch (run3 onwards warm-start from the previous run's replay buffer). Plot run2
alongside it as a *band*, not a competitor — the two differ only in search budget (160 vs
256 sims) and that alone doubles the learning rate on the same game (run2 hits +1000 Elo
at 2,560 games where run1 needs 5,120). Any cross-game claim must beat that band.

---

## 5. Backlog, and the three knobs

Azul-vs-Uno moves three dials at once — depth, luck, hidden information — so a difference
in shape cannot be attributed. Later additions should move one dial each.

| game | depth | luck | hidden info | why it earns a slot |
|---|---|---|---|---|
| **Lost Cities** (Knizia) | high | high | yes | The scientific control for Uno: same dials, universally rated excellent. If Uno flattens and this does not, the flatness is Uno's. ~250 lines, ~70 actions. **Highest value in this table.** |
| **Can't Stop** (Sackson) | medium | very high | no | "High luck + good design" against Uno's "high luck + thin design". Tiny action space. |
| **Patchwork** (Rosenberg) | medium | very low | no | Depth among *good* games; 2-player by design. A week, not a day — polyomino placement inflates the action space. |
| **Onitama** / **Hex 7×7** / **Breakthrough** | high | none | no | Chess's character at a size where the curve can actually bend. |
| **Backgammon** | high | very high | no | The historical RL benchmark, with GNU Backgammon as an external absolute yardstick. Heavy: move-sequence action space. |

### Chess — not by self-play on this hardware

Not for lack of interest. Two reasons, both quantified.

**It cannot reach the informative region.** Extrapolating from measured throughput (uno1
sustains ~42,000 evaluations/second on MPS with a 0.94M-parameter MLP): a chess-capable
network is roughly 25x the FLOPs per evaluation, so ~1,700 evals/s. A chess game is ~80
plies at ~600 simulations per move, i.e. ~48,000 evaluations, i.e. ~28 s/game — **about 2
games per minute**. 100,000 games is **35 days of continuous laptop time**, and 100k games
of chess self-play produces a player that still hangs pieces. AlphaZero used 44 million
games on 5,000 TPUs. The shortfall is four orders of magnitude, not a factor of two.

**A truncated chess run would produce a confident wrong answer**, for the reason in §4:
the linear fit to Azul run1 is *better* over the first 5% (R² = 0.996) than over the whole
run (0.909). Chess here would be a permanent 5% truncation, and would look like the
best-designed game in the study.

**Three ways to have chess anyway, in order of cost:**

1. **Overlay the published curve.** AlphaZero and Leela both published Elo-against-games.
   Since the cross-game quantity is the normalised *shape* (§4) and not the Elo, chess's
   published shape can go on the chart as an external reference line, clearly labelled as
   not our measurement (different net, different search, different anchors). Free, and it
   captures most of chess's real value here, which is that readers know what chess is and
   do not know what Azul is.
2. **Minichess** — Gardner 5x5 (all piece types) or Los Alamos 6x6 (no bishops). Chess's
   character at a completable size. Note Gardner 5x5 is reported weakly solved (a draw),
   but *we* cannot build that oracle (~10^18 positions, nothing like Connect Four), so it
   gives the character without the %-optimal yardstick — and it is a real engine build
   (move generation, promotion, repetition and 50-move draws). Onitama and Breakthrough
   occupy the same slot in the knob table for far less work, so minichess only earns a
   place if the point is specifically *chess-like*.
3. **Rented GPUs.** The only route to real chess measured by us, and a budget question
   rather than a technical one: thousands of GPU-hours to reach the bend.

**On "is Azul really that much simpler than chess?"** — careful, that is not what the
numbers say. Azul's branching (20-60) is comparable to chess's ~35 and its games are of
similar length (~52 moves against ~80 plies). Two other things drive the gap: the network
chess needs is far larger, and — the interesting one — **how much distinguishable skill
sits above the level we reached**. 25k games gave a strong Azul player; 25k games of chess
gives a beginner. That difference is a statement about the games rather than an
inconvenience, and it is close to what this project exists to measure. Two honest caveats:
it is confounded with network size and compute, and we never found Azul's ceiling either
(run5 plateaued around +2383 and was still climbing when it was paused), so "Azul is
shallower" is partly "we stopped earlier".

---

## 6. Decisions already taken (do not re-open without a reason)

- **Order:** Uno+ first, then tic-tac-toe + Connect Four together. Lost Cities after.
- **Uno+ ships all four rules at once** (draw-always, stacking, 7-swap, 9-card hand).
  Ablate individual rules only if the combined variant shows an effect worth attributing.
- **Solved games are headlined by % optimal**, Elo secondary, solver never in the anchor
  pool.
- **Two players everywhere**, to stay comparable with Azul.
- **The +4 challenge rule is deliberately out of scope for now.** It is real bluffing and
  therefore genuinely interesting, but PIMC search is famously bad at bluffing — a null
  result would confound "the rule adds nothing" with "our search cannot use it". Revisit
  only with a search that models beliefs.
