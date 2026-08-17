# How the Ludometer AI learns

This is the explainer for the whole system: what it does, why each piece is shaped
the way it is, what the numbers on the dashboard actually measure, and where every
byte of it lives on disk. It assumes you are a strong engineer and assumes nothing
about reinforcement learning. Jargon is used freely, but every term is defined the
first time it appears, and again in the glossary at the end.

The companion documents are `docs/DESIGN.md` (the contract: rules, schemas, APIs)
and `NOTES_FOR_REMI.md` (the running journal). Where this page and the code
disagree, the code is right — but tell someone, because that is a bug.

## 1. What the system is

Ludometer teaches a neural network to play two-player **Azul** by having it play
against itself, on one Mac (12 cores, 36 GB, Apple MPS), and it records how strong
that network gets as a function of how much compute went in. Nothing about the game
is taught by a human: there is no opening book, no library of expert games, no
hand-written evaluation inside the learner. The only human input is the rule book,
encoded in `ludometer/azul/engine.py`.

Three ingredients make that work, and they are worth naming before we go deeper:

- A **policy/value network**. One neural network with two outputs. The *policy head*
  outputs a probability for each of the 180 possible moves — "which move looks
  promising here". The *value head* outputs a single number in `[-1, +1]` — "how
  likely is the player about to move to win this position". Together they are a
  fast, shallow, learned intuition.
- A **search** — Monte Carlo tree search (MCTS). Given a position, it uses the
  network to explore a few hundred lines of play and comes back with a better move
  distribution than the raw policy head produced. Search is the slow, deep,
  compute-bought part.
- A **training loop** that closes the circle: the search's output is used as the
  training target for the policy head, and the eventual game result is used as the
  training target for the value head. The network gets better; the search built on
  it gets better; its output is a better target still.

That circularity — search improves the network, the improved network improves the
search — is the whole trick, and it is why this family of methods is named after
AlphaZero. What we run is deliberately a *lite* version: no distributed cluster, no
1000-GPU league, one machine and a few hours per run.

## 2. The premise we are testing

The project is not really about Azul. It is about a hypothesis of Remi's:

**A good game teaches linearly.** Plot the skill of a learner against the effort it
has spent learning; a well-designed game should give a long, straight, still-rising
line — every hour of study buys about the same amount of strength. A shallow game
saturates fast (the line bends flat early). A game with a wall in it produces a
plateau then a jump.

To test that we need (a) a learner whose skill can be measured on a stable scale,
and (b) a curve. Section 7 explains the scale (Elo against a *fixed* anchor pool),
section 8 shows the three curves we have, and section 10 is the honest accounting
of what those curves do and do not establish. The short version: on raw game count,
all three runs are **concave**, not linear — steep early, then a long grind — and
that shape reproduced across three quite different network architectures, which is
the most interesting result so far.

## 3. The loop, in one picture

<!-- ludometer:diagram:alphazero-loop -->

```
  [1 self-play] --positions--> [2 replay buffer] --batches--> [3 gradient steps]
        ^                                                            |
        |                                                            v
  [4 checkpoint + Elo rating] <----------------------------- updated network
```

One turn of that loop is called an **iteration** and is implemented by
`Trainer._iteration` in `ludometer/train/trainer.py`. Concretely, per iteration:

1. **Self-play.** 8 worker processes play `games_per_iter = 64` complete games
   against themselves, using the current network inside MCTS. A game is about 53
   positions long. Every position played is recorded as a training example.
2. **Replay buffer.** Those examples are appended to a fixed-size ring of the most
   recent 500,000 positions (300,000 in run1). Old positions fall off the end.
3. **Gradient steps.** Batches of 256 positions are sampled uniformly at random
   from the whole buffer and used to update the network with Adam on the Mac's GPU
   (MPS). The number of steps is `positions_added * 1.5 / 256`, i.e. each new
   position is expected to be trained on about 1.5 times before it ages out.
4. **Checkpoint and rating.** Every 512 self-play games the current weights are
   written to `runs/<run>/checkpoints/ckpt-<games>.pt` and that frozen copy plays 40
   evaluation games against each member of a fixed pool of opponents. A
   Bradley-Terry fit turns those results into one Elo number, appended to
   `runs/<run>/elo.jsonl`. That file *is* the curve on the dashboard.

Then the new weights are pushed to the 8 workers and the loop repeats. Over a
12-hour run this happens about 430 times.

Two details that matter and are easy to miss. First, self-play runs on **CPU**, one
thread per worker, one position at a time — batch-1 latency, not throughput, is what
limits it, which is why network *size* is chosen by measured milliseconds per
position rather than by parameter count (section 8.3). Second, the workers hold
their own copy of the weights and are refreshed once per iteration, so self-play is
always a few dozen games behind the trainer. That is fine and is standard.

## 4. Inside one move: the search

<!-- ludometer:diagram:mcts-cycle -->

```
  select  ->  expand  ->  evaluate  ->  back up      x 512, then play the
  (PUCT)      (priors)    (value)       (N, W)       most-visited move
```

### 4.1 Why search at all

The policy head alone would play a legal, plausible game. It would also be blind to
anything it has not already internalised. Search fixes specific mistakes at play
time: it looks at the consequences. And because the search's answer is better than
the network's answer, the difference between them is exactly the training signal.
AlphaZero-style learning is, in one sentence, **"train the network to predict what
the search would have said, then search with the improved network."**

MCTS builds a tree whose root is the current position. Each *edge* is a legal move
and stores two numbers: `N` (how many simulations went through it) and `W` (the sum
of values that came back through it). `Q = W / N` is the edge's running average
value. A **simulation** is one pass through the four steps below, and run3 does 512
of them per move.

### 4.2 The four steps

**Select.** Starting at the root, repeatedly pick the child that maximises the PUCT
score. PUCT ("Predictor + Upper Confidence bounds applied to Trees") is the rule
that balances "the move that has looked good so far" against "the move the network
likes but we have barely tried":

```
score(a) = Q(a) + c_puct * P(a) * sqrt(N_parent + 1) / (1 + N(a))
```

`P(a)` is the network's prior probability for move `a`, `c_puct = 1.4` sets how much
exploration is bought, and the `1 / (1 + N(a))` term makes an edge less attractive
the more it has already been tried. An edge that has never been visited takes
`Q = 0`, i.e. "assume a draw" — see `_select` in `ludometer/train/mcts.py`.

**Expand.** When selection reaches a position that has no children yet (a *leaf*),
the network is called on it once. The policy head's logits are softmaxed **over the
legal moves only**, and the result becomes the priors `P(a)` for that node's new
edges. Illegal moves never get an edge at all, so masking is structural rather than
a `-inf` trick.

**Evaluate.** The same network call returns the value head's `v`, a number in
`[-1, +1]` meaning "how good is this for the player about to move here". Note what
is *absent*: there is no random playout to the end of the game. The value head
replaces the rollout entirely. That is the single biggest difference between
AlphaZero-style MCTS and the classic 2006 version.

**Back up.** Walk back to the root adding `1` to `N` and `v` to `W` on every edge of
the path. Azul does not strictly alternate turns — the player holding the
first-player marker can move twice across a round boundary — so the implementation
keeps every value in player 0's frame and flips the sign per node according to whose
turn that node is, rather than assuming alternation.

After 512 simulations the root's visit counts are normalised into a distribution
over the 180 actions. That **visit distribution** is both (a) how the move is chosen
and (b) the training target for the policy head. It is stored verbatim.

### 4.3 Keeping self-play honest

Two mechanisms stop self-play from collapsing into the same game over and over.

**Dirichlet root noise.** At the root of every self-play search, the priors are
mixed with a random sample from a Dirichlet distribution:
`P <- 0.75 * P + 0.25 * noise`, with concentration `alpha = 10 / n_legal`. In plain
terms: a random subset of the root's moves gets its prior boosted, so the search is
forced to look seriously at a move it would otherwise have dismissed. This is
applied only during self-play, only at the root, and never during evaluation.

**Temperature.** For the first `temp_moves = 12` plies of a game, the move actually
played is *sampled* from the visit distribution raised to the power `1/T` with
`T = 1.0` — i.e. sampled proportionally to visits. After that, the most-visited move
is played deterministically. Early randomness gives opening variety; later
determinism keeps the game quality high. Independently of which move is played, the
*stored target* is always the raw visit distribution.

There is also a stall guard: two deterministic players can loop an Azul game
forever, because if neither ever completes a pattern line the tiles just cycle
bag → floor → lid. From round 16 (`stall_rounds`) sampling resumes, and 400 moves
(`max_game_moves`) is a hard cap after which the game is scored as a draw.

### 4.4 Chance: the bag

Azul is deterministic *within* a round; randomness enters only at the round
boundary, when the factories are refilled from the bag. MCTS handles this by
**determinization**: an edge whose move empties the board is a *chance node*. Rather
than one child, it holds a small table of sampled outcomes — the clone's bag is
reshuffled with a fresh seed before the move is applied, and up to
`chance_children` distinct refills are kept and thereafter reused uniformly at
random. This is what stops the search from cheating: without the reshuffle, a cloned
state would contain the *exact* upcoming deal and the search would plan against
cards it cannot see. (Bag and lid *counts* are public information and are in the
network's input; the order is not.)

The edge's value is the **visit-weighted mean over the determinizations sampled
below it**, which is worth being precise about because it is not a rule anyone
wrote: the `(N, W)` counters live in the parent, so every simulation through the
edge adds to the same pair whichever refill it landed in, and `W/N` is that mean by
construction. `chance_backup: "mean"` in a config names the behaviour; the identity
is demonstrated numerically in `tests/test_train_mcts.py`.

How wide the sample is matters for *cross-round* planning specifically. run1-run5
used `chance_children = 4`: a plan that pays off two rounds later is averaged over
four guesses at each of two refills, which is a lot of noise on exactly the
comparisons a long-horizon plan needs to win. run6 doubles it to 8, at a measured
cost of a few percent of self-play throughput and no meaningful memory.

### 4.5 Tree reuse

By default each move starts a fresh tree and throws away everything the previous
move learned, which is wasteful: the subtree under the move you actually played is
still valid. With `tree_reuse: true` (run3), that subtree is kept as the next
search's root and topped up to `sims` total visits, so 512 sims cost only about 320
new network evaluations per move. The subtree is dropped whenever the state cannot
be guaranteed to match — in particular across a refill boundary — and fresh
Dirichlet noise is mixed into the reused root's priors. A cheap position fingerprint
is checked before reuse; a mismatch silently falls back to a fresh tree. Tree reuse
is used in self-play only, never in evaluation, so ratings stay comparable.

## 5. What the network sees, and what it says

### 5.1 The 182 numbers

`AzulState.encode()` turns a position into a fixed 182-float vector, always from the
point of view of the player about to move ("me" and "them", never "player 0" and
"player 1"). That convention is what lets one value head mean one thing.

| Offset | Size | Contents |
| --- | --- | --- |
| 0 | 25 | my wall, 5x5, 0/1 |
| 25 | 25 | their wall |
| 50 | 30 | my 5 pattern lines: colour one-hot (5) + fill fraction (1) each |
| 80 | 30 | their pattern lines |
| 110 | 7 | my floor: 5 colour counts /7, occupancy /7, first-player-marker flag |
| 117 | 7 | their floor |
| 124 | 2 | scores /100 |
| 126 | 25 | 5 factories x 5 colour counts, /4 |
| 151 | 5 | per-factory non-empty flags |
| 156 | 5 | centre colour counts /10 |
| 161 | 1 | centre total /20 |
| 162 | 1 | first-player marker still in the centre |
| 163 | 5 | bag colour counts /20 |
| 168 | 5 | lid (discard) colour counts /20 |
| 173 | 1 | tiles left on the board this round /20 |
| 174 | 1 | do I start the next round |
| 175 | 1 | round index, capped at 10, /10 |
| 176 | 3 | my completed rows / columns / colours, each /5 |
| 179 | 3 | theirs |

The output side is a fixed 180-action space, `action_id = source * 30 + colour * 6 +
destination`, with 6 sources (5 factories + the centre), 5 colours and 6
destinations (5 pattern lines + the floor). Most are illegal in any given position;
legality comes from the engine, not the network.

### 5.2 run1 and run2: a residual MLP

The first two runs used `ludometer/train/net.py`: feed the 182 floats into a stem
linear layer, then a stack of residual blocks (`x + ReLU(LayerNorm(Linear(x)))`),
then two heads. run1 was 3 blocks of 512 units (1,010,997 parameters); run2 was 5
blocks of 768 (3,315,061 parameters).

This works, and it is the right thing to try first, but it wastes capacity on
structure the game gives you for free. The MLP has to *learn* that features 126..130
and 131..135 are two interchangeable factory displays, and that "put blue in row 3"
is the same kind of operation as "put blue in row 4". Every symmetry it has to
rediscover costs parameters and data.

### 5.3 run3: 22 entities and one attention layer

<!-- ludometer:diagram:structured-net -->

```
  182 floats -> 22 entity tokens -> shared embedders -> self-attention
             -> readout -> factorised policy head (180) + value head (1)
```

`ludometer/train/net2.py` keeps the *same* 182-float encoding — so old replay
buffers and the browser player keep working — and slices it into **22 entity
tokens**:

| Token type | Count | Raw dims | What it is |
| --- | --- | --- | --- |
| pool | 6 | 6 | one per tile source: 5 factories + the centre |
| pattern row | 10 | 11 | 2 players x 5 rows, each carrying its own wall row |
| wall set | 2 | 4 | completed rows / columns / colours + score |
| floor | 2 | 7 | one per player |
| supply | 1 | 10 | bag and lid counts |
| globals | 1 | 7 | scores, round, marker, tiles left |

Three design decisions follow from that slicing:

- **Weight sharing.** All six pool tokens go through the *same* small embedder, and
  all ten pattern-row tokens through another. The factory embedder therefore sees 6
  training examples per position instead of 1, and the row embedder 10. Identity is
  not lost: a learned per-slot bias is added after the shared embedding, so the net
  still knows which factory it is looking at.
- **Self-attention, not pooling.** One pre-LayerNorm transformer layer (4 heads)
  mixes the 22 tokens. The choice was deliberate: Azul decisions are *relational* —
  "is the black tile on factory 3 useful for **my** row 2, and does taking it hand
  the centre to my opponent" is a question about a pair of tokens. Pooling can weight
  tokens; only attention can compare them. With 22 tokens the attention matrix is
  22x22, which is free next to the projections.
- **Factorised policy head.** Instead of 180 independent output units, each of the 6
  source tokens emits a per-colour key `A[s, c]` in R^32 and each of the 6
  destination tokens emits a query `B[d]` in R^32; the logit for "take colour c from
  source s and put it in destination d" is the dot product `A[s,c] . B[d]` plus a
  per-(source, colour) bias, plus a 180-wide correction read off the pooled trunk. So
  the 180 logits share statistical strength, and each is scored by the tokens that
  actually own the facts involved.

A pattern-row token carries its matching *wall* row as well as its own contents,
because that is exactly what decides whether a colour is still legal there and what
it would score. Answering "can I still use this row" inside a single token is worth
more than it sounds.

The result is 1,679,002 parameters — **half of run2's MLP** — at about 0.20 ms per
position on one idle CPU thread.

## 6. How the weights actually move

### 6.1 The replay buffer

Three preallocated numpy arrays and a write cursor: `states (capacity, 182)`,
`policies (capacity, 180)`, `values (capacity,)`, all float32. New games overwrite
the oldest positions (a FIFO ring), and training samples **uniformly with
replacement** from whatever is currently in it. Capacity is 500,000 positions in
run2/run3, roughly 9,400 games of history.

Why a buffer at all, rather than training on each game as it arrives? Because
consecutive positions from one game are extremely correlated, and a gradient step on
a correlated batch is a bad estimate of the true gradient. Sampling from a large
buffer decorrelates the batch and lets each position be reused a few times.

### 6.2 The loss

Two terms, added:

```
loss_p = masked_ce(logits, target_policy)            # cross-entropy on the visit distribution
loss_v = mse(value, target_value)
loss_m = masked_mse(margin, target_margin)           # run4 onwards, see below
loss_a = masked_bce(wall_logits, final_wall_bits)    # run6 onwards, see below
loss   = loss_p + value_weight * loss_v + margin_weight * loss_m + aux_weight * loss_a
```

The policy target is the MCTS visit distribution stored during self-play. The value
target is *not* simply the win/loss result. It is

```
v = (1 - w) * outcome + w * tanh(score_difference / 20),   w = value_score_weight = 0.15
```

with `outcome` in `{+1, 0, -1}`. Blending in 15% of the (squashed) score margin gives
the value head a denser gradient — winning by 40 is different from winning by 2 —
without letting margin-chasing override actually winning. It has a visible side
effect, noted after a human game: with a win already decided, the AI plays "lazy"
endgame moves, because at that point the outcome term is saturated and only a small
margin term is still moving.

**run4 stops blending and splits the question in two** (`"margin_head": true`, see
docs/DESIGN.md). A third head predicts `m = tanh(score_difference / 20)` on its own,
trained with a masked MSE at weight `margin_weight = 0.25`, and the value head goes
back to a pure win/draw/loss target (`value_score_weight = 0`). The blend above was
always a cheap approximation of that head, and one output cannot answer two
questions well. The search is not affected — PUCT still descends on the win value
alone, so the visit counts that become policy targets are unchanged — but MCTS now
backs up the margin alongside the value, and the move actually *played* at
temperature 0 is chosen lexicographically: of the root children within
`decisive_eps = 0.03` of the best win-Q (and with at least a tenth of the top
child's visits), play the one with the largest margin-Q. Winning stays first; the
margin only ever breaks a tie. That is the direct answer to the lazy endgame.

**run6 adds a term about the far future** (`"aux_heads": true`). Look at the three
losses above and notice what is missing: `outcome` and `margin` are both a single
number about the *end* of the game, and the policy target is about *this* move.
Nothing in the loss ever asks the network what shape the game will end in — which is
exactly the complaint a strong human made about run5 ("the tactics are good, the
long-term play is weak"). So a fourth head predicts, for both players, which wall
rows, columns and colours their **final** wall will hold: 30 sigmoids, binary
cross-entropy, weight `aux_weight = 0.1`.

The reason to expect this to buy strategy rather than just parameters is the *time
horizon of the label*. Whether row 3 ends up closed is settled four or five rounds
after the position being labelled, and no amount of local tactical reading answers
it — the trunk has to carry something like a plan to predict it at all. Auxiliary
targets that share a trunk are a well-travelled trick for precisely this reason
(AlphaGo's territory head, KataGo's ownership and score heads); the extra head is
2 Linears and 0.4M of 7.4M parameters, and the weight is deliberately small because
its job is to shape the trunk, not to compete with the policy.

The targets cost nothing to produce: they are read off the finished board
(`AzulState.wall_summary`) once per game and stored as 30 **bits** per position, 4
bytes packed. Positions from an older buffer have no such label, so they load with
`aux_mask = 0` and the term skips them entirely — which is what makes it possible to
warm-start run6 from run5's 500k positions without inventing a single label.

The policy term is masked too, from run6 on, for a different reason: under
playout-cap randomization only about a quarter of self-play moves get a deep search,
and only those record a visit distribution. The other positions are still trained on
for value, margin and final walls — those labels come from the end of the game, not
from the search — but their policy row is zeroed and masked out. Masking rather than
zeroing matters: a zero row already contributes no gradient, but it would still
divide the batch mean, so the policy loss (and the effective learning rate on that
head) would silently shrink with the cheap fraction.

Gradients are clipped to norm 1.0. Adam, weight decay 1e-4.

One asymmetry worth knowing: at inference the policy is softmaxed over legal moves
only, but the training cross-entropy uses a plain softmax over all 180 logits.
Illegal actions contribute nothing positive to the loss but do sit in the
normaliser, so the network learns to push their logits down. It works, and it means
the exported browser model needs no mask at load time.

### 6.3 Learning rate

Cosine decay from `lr` to `lr_min` over `lr_total_steps`, re-set on every step. run3
uses 6e-4 down to 5e-5 over 20,000 steps, which was derived rather than guessed:
`60,000 games x ~55 positions x 1.5 replays / 256 per batch = ~19.3k steps`, so the
cosine lands on its floor exactly when the game budget runs out.

### 6.4 Behaviour-cloning pretraining

run3 does something the first two runs did not: before a single self-play game, the
fresh structured network is fitted to **run2's entire 500,000-position replay
buffer** for 3 epochs — same loss, plain supervised learning on a fixed dataset.
This is *behaviour cloning*: copying another agent's decisions rather than
discovering them. Those positions are kept in run3's buffer, so iteration 1 already
trains on 500k good positions instead of on noise.

Two knock-on adjustments. The peak learning rate was dropped from 1e-3 to 6e-4,
because a full-size peak would undo the warm start within a few hundred steps. And
the pretraining steps deliberately do *not* advance the step counter, so the
self-play cosine schedule still starts at its peak.

What it bought is unambiguous: run3's **first** rated checkpoint, at zero self-play
games, scored **+2107 Elo** — above both finished runs' best checkpoints. What it
cost is discussed in section 10.

## 7. How strength is measured

A number like "+2,336 Elo" is meaningless without saying what it is measured
against. Here is exactly what it is.

### 7.1 A match

`ludometer/eval/arena.py` plays a match of `n` games between two agents. Games are
played in pairs: games `2k` and `2k+1` share the same deal seed and swap seats, so
both agents see the same bag order from both sides. That is paired-comparison
variance reduction, and it means a 40-game match is 20 deals played twice. Every
result is a pure function of `(specs, seed, seat)`, so a match is bit-for-bit
reproducible and parallelising it over 8 processes changes nothing.

A draw counts as half a win to each side, everywhere.

### 7.2 Bradley-Terry, and what "Elo" means here

Elo is a rating scale on which a 400-point gap means a 10:1 expected score:

```
P(i beats j) = 1 / (1 + 10^(-(r_i - r_j) / 400))
```

Given a table of results between several players, we fit all their ratings at once by
maximum likelihood under that model. That is the **Bradley-Terry** model — the same
model Elo formalises, fitted properly to all games at once instead of updated
incrementally after each one. `ludometer/eval/elo.py` solves it with a damped Newton
iteration.

Two implementation details with real consequences. A small prior (half a phantom
drawn game per pairing that was actually played) keeps a 100% sweep from implying an
infinite rating — a 40-0 result implies at most about +883 Elo, not infinity. And the
error bars are the asymptotic 1-sigma from the Fisher information matrix, which
means they are the uncertainty *given that the anchors are exactly right*.

### 7.3 The fixed anchor pool

This is the part that makes the curves comparable, so it is worth being precise.

Each rated checkpoint plays 40 games against each opponent in a pool, and the fit is
run with some of those opponents' ratings **held fixed** (not estimated — pinned as
constants, and the Newton step is solved only over the free parameters). The pool is:

- `random` — uniformly random legal move. Pinned at exactly **0 Elo**. This is the
  origin of the whole scale.
- `greedy` — a 1-ply search maximising immediate banked score.
- `heuristic` — a 1-ply search with a hand-tuned positional evaluation (adjacency,
  floor damage, set bonuses, denying the opponent). Measured at about +1,378 vs
  random.
- Pinned checkpoints from earlier runs. run2's pool contains run1's best at +2,014;
  run3's contains both run1's best (+2,014) and run2's best (+2,020.3).
- Up to `eval_frozen = 2` of this run's own earlier checkpoints — specifically the
  most recently rated one and the all-time strongest one — each pinned at the Elo it
  was previously assigned.

Because those anchors never move, a rating produced at game 500 and a rating
produced at game 25,000 are on the same axis, and so are ratings from different runs.
That is the entire point: without it, an Elo curve is a curve of a self-referential
quantity and its slope means nothing.

Evaluation always runs at `eval_sims = 100` simulations per move, temperature 0, no
Dirichlet noise, no tree reuse — a fixed, cheap, deterministic setting, so what the
curve measures is the *network*, not the search budget.

### 7.4 What the number does not mean

Three things fall out of the code that are not obvious from the dashboard:

- **`greedy` and `heuristic` contribute nothing to the rating.** Only pairings
  involving the candidate are in the fit, and only `random` and the pinned
  checkpoints are anchored. That makes `greedy` and `heuristic` free parameters
  hanging off a star graph; the fit simply reproduces their observed win rates and
  passes no information back. They are useful as *milestones you can read*, not as
  measuring instruments. (Check it yourself: run1's first rating, +125.3, is
  reproduced to 0.1 Elo by the `random` edge alone.)
- **The scale is a ratchet.** Once the candidate beats `random` 100% of the time,
  that edge carries almost no information (the Fisher weight `n p (1-p)` goes to
  zero). From then on, a rating is effectively "the previous checkpoint's published
  Elo, plus whatever the score share against it implies". The scale rests on a chain
  of previously published numbers.
- **The strongest checkpoint is the winner's curse.** `eval_frozen` pins the
  all-time strongest checkpoint as an anchor — but "strongest" is a maximum over ~50
  noisy estimates with about ±30-55 Elo error bars each, and a max of noisy draws is
  biased upward. So the reference is a little too high, and everything measured
  against it is pulled a little too low. run3 shows the fingerprint clearly:
  `ckpt-020992` scored 2336.2 ± 32.0, stayed pinned as the anchor for 6,000 further
  games, and later checkpoints have scored around 0.46 against it ever since without
  the curve exceeding it again.

### 7.5 Gauntlets, for a different question

`ludometer/eval/gauntlet.py` exists because the curve's question ("how is this run
progressing, on a stable axis") is not the same as "which of these finished agents
plays better at the settings a human would use". A gauntlet plays a full round robin
between arbitrary agents, each written as `[label=]spec` where the spec can carry its
own budget — `?sims=400` or `?think=5.0` (seconds of wall clock per move) — and
prints a cross table plus a Bradley-Terry fit with optional `--anchor NAME=ELO` to
put the result back on the run curve's scale. It runs at `nice 19` by default,
because a gauntlet is never more urgent than the training run it is measuring.

## 8. Three runs, three ideas

<!-- ludometer:runs-table -->

| run | network | sims | games | wall clock | best Elo | Elo / 1k games | R² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| run1 | 3 x 512 residual MLP, 1.01M params | 160 | 25,000 | 3 h 07 m | +2,013.9 ± 56.2 | 61.4 | 0.909 |
| run2 | 5 x 768 residual MLP, 3.32M params | 256 | 24,448 | 12 h 40 m | +2,020.3 ± 38.6 | 50.1 | 0.686 |
| run3 | 22-token attention net, 1.68M params | 512 | 27,712 | 12 h 50 m | +2,336.2 ± 32.0 | 6.5 | 0.780 |

(Snapshot taken 2026-08-16 while run3 was still training; the table on the HTML page
is regenerated from the logs every time the page is built.)

### 8.1 run1 — does any of this work at all

3 blocks of 512, 160 simulations per move, 8 workers, anchored only on random /
greedy / heuristic. 25,000 games in **3 hours 07 minutes** — about 8,000 games per
hour, or 135 games per minute.

It worked immediately: +253 Elo after 512 games, past `greedy` at ~5k games, past
`heuristic` at ~9k, and finishing at **+2,013.9 ± 56.2** (checkpoint `ckpt-024064`),
roughly 640 Elo above the strongest scripted baseline. The straight-line fit over all
49 ratings gives 61.4 Elo per 1,000 games at R² = 0.909.

But the fit hides the shape. Over the first half of the run the slope is 104.5 Elo
per 1k games; over the second half, 45.7. Steep then grinding — concave, not linear,
and still rising at the cap. Read through the project's own lens, Azul comes out as
"easy to pick up, keeps rewarding study".

### 8.2 run2 — more capacity, more search

The obvious next lever: roughly 3x the network (5 blocks of 768, 3.32M parameters)
and 256 simulations per move instead of 160, with run1's best checkpoint pinned in
the pool at +2,014 so the axis stays comparable.

Per game, it worked. At 5,000 games run2 was at +1,530 where run1 had been at +950;
at 10,000 games, +1,757 vs +1,376. It learned substantially more from each game.

Per hour, it lost. A bigger network at more simulations costs about 4x more compute
per move, and throughput fell to about 1,950 games per hour (31 games per minute).
Three hours in — the point where run1 had *finished* at +2,014 — run2 was at +1,502.
It needed nearly 12 hours to reach **+2,020.3 ± 38.6** (`ckpt-023040`), statistically
level with run1's best, winning about 45% of direct head-to-heads.

Its curve is concave too, and more sharply: 118.4 Elo per 1k games over the first
half, 12.4 over the second, R² 0.686 against a straight line. It was retired at
midday with gains flattened to roughly +20 Elo per 1k games.

**The lesson is the axis.** "Bigger network, deeper search" is unambiguously better
per game and roughly break-even per hour. On a single Mac, wall clock is the budget
that is actually scarce, so run3 had to buy strength somewhere other than raw size.

### 8.3 run3 — structure, reuse, and a warm start

Three changes, each justified by a measurement rather than by taste.

**A structured network instead of a bigger one** (section 5.3). Chosen against a
time budget, not a parameter budget: 0.20 ms per position single-threaded on a quiet
machine (budget was 0.45), and 0.66 ms with 8 workers competing for cores — versus
0.78 ms for run2's 5x768 MLP in the same 8-way test. Half the parameters, faster per
position, and the inductive biases the MLP had to learn are built in. The benchmark
also priced the alternatives: widening the embedding to 128 costs +20%, a second
attention layer another +15%.

**512 simulations per move with tree reuse.** Twice run2's search depth, but reuse
turns 512 simulations into roughly 320 fresh network evaluations, so the throughput
lands back at run2's ~2,200 games per hour while searching twice as deep.

**Pretraining on run2's replay buffer** (section 6.4). The starting point moves from
"random noise" to "everything run2 learned in 12 hours".

The result is the strongest agent so far — **+2,336.2 ± 32.0** at `ckpt-020992` — and
also the most awkward curve in the project. It *starts* at +2,107 and has gained only
about +176 Elo over 27,600 games, roughly 6 Elo per 1,000 games, with ±30 error bars.
Whether that is "the warm start already extracted most of what this method gets out
of Azul" or "the anchor pool has saturated and the measurement has gone blind" is
exactly the question section 10 is about, and it is the main reason a run4 is worth
running.

### 8.4 The same shape, three times

The most interesting thing in the three curves is what they have in common. All
three are concave in raw game count — a fast start, then a long grind — across a
1.0M-parameter MLP at 160 sims, a 3.3M-parameter MLP at 256 sims, and a 1.7M-parameter
attention network at 512 sims with a warm start. The shape survived every change we
made to the learner, which is evidence that it belongs to **the game plus the
method**, not to any particular network.

That is a real, if preliminary, result for the project's premise. It also says the
"linear" in "does a good game teach linearly" needs an x-axis: raw games gives
concave curves, and log-games or wall clock may be the fairer axis for the
hypothesis. The dashboard currently fits the raw-games version, which is the
conservative choice.

### 8.5 What a run4 should test

The honest answer to "is run3's flat curve real or an artefact" needs a measurement
the current pool cannot make: put run3's checkpoints in a gauntlet against each
other and against run1/run2's best at play-time settings, anchored on run1's +2,014,
and see whether the internal ladder's ordering survives. A run4 that adds a
*human-calibrated* anchor, or simply a longer budget at run3's settings without a
warm start, would separate "learning has stopped" from "measuring has stopped".

## 9. Where the data lives

<!-- ludometer:diagram:data-layout -->

```
runs/
  human_benchmarks.jsonl
  run3/
    config.json  status.json  train.jsonl  elo.jsonl
    checkpoints/ ckpt-020992.pt  latest.pt  replay.npz   (git-ignored)
```

Everything observable about a run is a plain text file under `runs/<run>/`. Nothing
lives in a database, nothing needs the trainer to be running to be read, and the
append-only files are safe to read while they are being written — the dashboard
tolerates a torn last line by design.

### 9.1 The schemas

Quoted from `docs/DESIGN.md`, which is the contract:

```
- config.json — run hyperparameters, free-form dict, plus "run", "started" (ISO time).
- status.json — heartbeat, rewritten atomically by the trainer:
  {"run", "state": "running"|"done"|"failed", "started", "updated", "ended": <iso|null>,
   "error": <str|null>, "games", "steps", "note"}
- train.jsonl — appended every logging interval:
  {"t": <sec since run start>, "games": <total self-play games>, "steps": <optimizer steps>,
   "loss": <total>, "loss_p": <policy>, "loss_v": <value>, "loss_m": <margin, run4+>,
   "loss_a": <final walls, run6+>, "buffer": <replay size>, "lr": <lr>}
- elo.jsonl — appended after each checkpoint evaluation:
  {"t": <sec>, "games": <self-play games at ckpt>, "ckpt": "<name>", "elo": <float>,
   "elo_err": <float>, "vs": {"<opponent>": <winrate 0..1>, ...}, "n_games": <eval games>,
   "pool": [<anchor/opponent names with their fixed Elos where anchored>]}
```

All timestamps are UTC ISO-8601 with an explicit offset. Where fields are duplicated,
`status.json` wins over `config.json` and over the last `train.jsonl` line. Draws
count as half-wins in every win rate. Pretraining epochs are logged to `train.jsonl`
with `"phase": "pretrain"`; self-play lines have no `phase` field.

A real `elo.jsonl` line, from run3:

```json
{"t": 45255.5, "games": 27648, "ckpt": "ckpt-027648", "elo": 2282.9, "elo_err": 30.8,
 "vs": {"random": 1.0, "greedy": 1.0, "heuristic": 0.95,
        "mcts:runs/run1/checkpoints/ckpt-024064.pt?sims=100": 0.75,
        "mcts:runs/run2/checkpoints/ckpt-023040.pt?sims=100": 0.95,
        "ckpt-027136": 0.463, "ckpt-020992": 0.463},
 "n_games": 280,
 "pool": ["random=0.0", "greedy", "heuristic",
          "mcts:runs/run1/checkpoints/ckpt-024064.pt?sims=100=2014.0",
          "mcts:runs/run2/checkpoints/ckpt-023040.pt?sims=100=2020.3",
          "ckpt-027136=2252.7", "ckpt-020992=2336.2"]}
```

Everything needed to audit that rating is in the line: who was played, at what score
share, over how many games, and which of them were pinned and where.

### 9.2 The replay buffer file

`checkpoints/replay.npz` is an uncompressed numpy archive, written
oldest-position-first so the file is a chronological record. Arrays have only ever
been **added**, never removed or reordered, so a reader that knows the first three
keys can read any file this project has ever written:

| array | shape | dtype | since |
| --- | --- | --- | --- |
| `states` | (N, 182) | float32 | run1 |
| `policies` | (N, 180) | float32 | run1 |
| `values` | (N,) | float32 | run1 |
| `meta` | (5,) | int64: capacity, size, total_added, games_added, seed | run1 |
| `margins` | (N,) | float32 — `tanh(final score diff / 20)` | run4 |
| `margin_mask` | (N,) | float32 — 1 where `margins` is real | run4 |
| `aux` | (N, 4) | uint8 — 30 final-wall bits, `np.packbits` | run6 |
| `aux_mask` | (N,) | float32 — 1 where `aux` is real | run6 |
| `policy_mask` | (N,) | float32 — 1 where `policies` is real | run6 |

A missing mask means "the data is not there" for `margins` and `aux`, and "the data
*is* there" for `policy_mask` — every position written before run6 came from a full
search. Packing the aux bits is what keeps the addition invisible: 4 bytes a
position instead of 120, i.e. 2 MB rather than 60 MB across a full buffer.

At the full 500,000 positions that is 726 MB. It is rewritten on every checkpoint
(atomically, via a `.tmp` and a rename), which is what makes both resuming a run and
pretraining the next one possible.

### 9.3 Checkpoints

A checkpoint is a plain `torch.save` dict. Rated checkpoints (`ckpt-<games>.pt`)
carry:

| key | contents |
| --- | --- |
| `format` | schema version, currently 1 |
| `net_config` | the architecture, including `arch` |
| `state_dict` | the weights, on CPU |
| `games`, `steps` | where in the run this was cut |
| `config` | the full run configuration |

`latest.pt` carries all of that plus the optimizer state, the iteration counter and
the list of ratings so far, which is what a resume needs. Because `net_config`
records `arch`, any loader — the trainer, the evaluator, the local GUI, the ONNX
exporter — can open any run's checkpoints without being told which network produced
them.

### 9.4 Human benchmarks

`runs/human_benchmarks.jsonl` is the one file a human writes by hand: one line per
session against a real person, with the opponent checkpoint, the number of games, how
many the human won, the seed, and a free-text note. There are three lines so far. It
is small, it is anecdotal, and it is the only tie the internal Elo scale has to
anything a person would recognise — which is why the notes matter as much as the
counts ("lazy endgame moves when the win was already decided" came from there).

### 9.5 Poke at it yourself

Every one of these runs from the repository root and touches nothing:

```bash
# the whole Elo curve of a run, as text
uv run python -c "import json; [print(r['games'], r['elo']) for r in map(json.loads, open('runs/run3/elo.jsonl'))]"

# the last rating of every run
for f in runs/*/elo.jsonl; do echo "$f -> $(tail -1 "$f")"; done

# how many positions are in the saved replay buffer, and their shapes
uv run python -c "import numpy as np; z=np.load('runs/run3/checkpoints/replay.npz'); print({k: z[k].shape for k in z.files})"

# what architecture a checkpoint holds, and where in the run it was cut
uv run python -c "import torch; c=torch.load('runs/run3/checkpoints/ckpt-020992.pt', map_location='cpu'); print(c['net_config'], c['games'], c['steps'])"

# rebuild the dashboard and this page from the current logs
python3 web/make_dashboard.py
```

And to make two finished agents actually play each other at play-time settings:

```bash
uv run python -m ludometer.eval.gauntlet --games 40 --workers 8 \
    heuristic \
    run1=mcts:runs/run1/checkpoints/ckpt-024064.pt?sims=100 \
    run3=best?sims=400 \
    --anchor run1=2014
```

## 10. What we have measured, and what it does not prove

The results, stated plainly:

- Self-play from zero reaches roughly +2,000 Elo above random in about 3 hours on
  one Mac, and about +2,340 with a warm start and a better architecture.
- The Elo-vs-games curve is **concave** on all three runs — fast early, slow later,
  still rising. The same shape appeared across three different architectures, three
  search budgets and two training regimes.
- Per game, capacity and search depth help a lot. Per hour, on this hardware, they
  roughly cancel. Structure (run3) is the change that improved both at once.

And the caveats, which are load-bearing:

- **Internal Elo is not human Elo.** The scale is anchored on a uniformly random
  player at 0. A rating of +2,336 says "10:1 against something we rated +1,936", not
  anything about Board Game Arena. The only bridge is three logged human games, in
  which a top-0.1% BGA player lost 4 of 5 to a +2,185 checkpoint, and Remi beat the
  +2,336 one once. Treat the mapping as unknown.
- **Ratings are measured at 100 simulations per move; the agent plays at thousands.**
  The evaluation budget is deliberately small and fixed so the curve tracks the
  network. At a 5-second budget the same checkpoint searches on the order of 5,000 to
  16,000 positions per move — 50-100x more. The published Elo is a floor on play-time
  strength, not an estimate of it.
- **The anchor pool has saturated.** run3 scores 1.00 / 1.00 / 0.95 against random /
  greedy / heuristic. The entire scripted ladder is exhausted, so the rating now
  floats on a chain of previously published checkpoint ratings (section 7.4). That
  chain includes an all-time-best anchor chosen as a max over noisy estimates, which
  biases it upward and drags subsequent ratings down. run3's flat late curve is
  consistent with both "learning has slowed" and "the ruler has run out", and the
  current data cannot separate them.
- **The linearity question needs its x-axis stated.** Elo vs raw games, Elo vs
  log(games) and Elo vs wall clock are three different hypotheses. We fit the first;
  it is the least flattering and the least ambiguous.
- **One game, one machine, three runs.** Nothing here is a claim about games in
  general yet. It is a working instrument and three data points.

## 11. Glossary

| Term | Meaning |
| --- | --- |
| **Action space** | The fixed list of 180 encoded moves, `source * 30 + colour * 6 + destination`. Most are illegal in any given position. |
| **AlphaZero-lite** | This project's scale of the AlphaZero recipe: self-play, MCTS guided by a policy/value network, train on the search's own output, no human games. One Mac, hours per run. |
| **Anchor** | An opponent whose rating is held fixed during the Bradley-Terry fit instead of being estimated. `random` is pinned at 0; strong checkpoints from earlier runs are pinned at their published Elo. Fixed anchors are what make curves comparable across time and across runs. |
| **Behaviour cloning** | Supervised learning that copies another agent's decisions from stored data, rather than discovering them by playing. Used once, to warm-start run3 from run2's replay buffer. |
| **Bradley-Terry** | The statistical model behind Elo: `P(i beats j) = 1 / (1 + 10^(-(r_i - r_j)/400))`. Fitted by maximum likelihood to all games at once, rather than updated incrementally. |
| **Chance node** | A search-tree edge whose move triggers a random event (in Azul, refilling the factories from the bag). Handled by determinization. |
| **Checkpoint** | A frozen copy of the network at a given game count, saved as `ckpt-<games>.pt`, rated once and thereafter immutable. |
| **Determinization** | Sampling a concrete outcome for a random event and searching that as if it were deterministic. Here: reshuffle the cloned bag, apply the move, keep up to 4 such outcomes per chance edge. |
| **Dirichlet root noise** | Random perturbation added to the root priors during self-play only, `P <- 0.75 P + 0.25 noise` with `alpha = 10 / n_legal`, so the search is forced to consider moves the network dismisses. |
| **Elo** | A rating scale where a 400-point gap means a 10:1 expected score. Here it is always relative to a random player pinned at 0. |
| **FPU (first-play urgency)** | The `Q` value assumed for a search edge that has never been visited. Here 0 — "assume a draw". |
| **Iteration** | One turn of the training loop: 64 self-play games, the gradient steps they earn, and possibly a checkpoint and rating. |
| **MCTS** | Monte Carlo tree search: build a tree of the moves worth considering by repeated select / expand / evaluate / back-up passes. Here the network's value head replaces the classic random rollout. |
| **Policy head** | The network output that gives a probability to each of the 180 actions — a fast guess at which moves deserve search. |
| **PUCT** | The formula that picks which child to descend into: `Q(a) + c_puct * P(a) * sqrt(N_parent + 1) / (1 + N(a))`. Balances observed value against the prior and the visit count. |
| **Replay buffer** | The fixed-size FIFO ring of the most recent positions (500,000 in run2/run3), sampled uniformly to build training batches so that consecutive, correlated positions do not dominate a gradient step. |
| **Temperature** | The exponent applied to the visit distribution before sampling the move to play. `T = 1` for the first 12 plies (variety); `T = 0` (argmax) afterwards. It never changes the stored training target. |
| **Tree reuse** | Keeping the subtree under the move you just played as the next search's root instead of starting fresh, so 512 simulations cost ~320 fresh network evaluations. Dropped across chance boundaries. Self-play only. |
| **Value head** | The network output that estimates, in `[-1, +1]`, how good the position is for the player about to move. |
| **Visit distribution** | The root's simulation counts, normalised over the 180 actions. It is both the move-selection distribution and the training target for the policy head. |
| **Winner's curse** | The upward bias you get when you select the maximum of many noisy estimates. The "best checkpoint" of a run is such a maximum, and pinning it as an anchor propagates the bias to everything measured against it. |
