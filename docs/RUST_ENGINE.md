# Brief: a Rust engine for Azul self-play

*Written 2026-09-06 for the agent that will do this work, by the agent that
built the cloud fleet and shipped Porcelain. Self-contained: everything you
need to know about the current stack is here or pointed to by path. The goal
is fixed; the road is yours.*

## 1. The goal, and the number that justifies it

Make self-play generation **at least 20× faster per CPU core** by moving the
game rules and the tree walk of the search from Python to Rust, while keeping
every rule, every target and every interface bit-for-bit compatible with what
exists. The net stays in PyTorch/ONNX; Rust owns everything that is not a
matrix multiply.

Why this is the lever, measured on 2026-09-05 (`ludometer.cloud.bench`):

| what | positions/s | where the time goes |
|---|---|---|
| one Python driver, tiny net (search-bound), Mac core | 13,780 | ~90 µs per simulation, all Python |
| same, cloud `cpu-upgrade` core (AMD EPYC) | 3,546 | same code, slower core |
| one Python driver, 7M-param teacher, T4 GPU job | 3,639 | half search, half waiting on the GPU round trip |
| Mac GPU, 6 drivers × 128 games, fp16 | ~24,000 | GPU mostly idle, 6 cores of Python |
| `l4x1` job, 8 drivers × 64 games, fp16 | ~24,000 | GPU mostly idle, 8 cores of Python |

A simulation is one descent of the tree plus one net evaluation of a leaf.
The net evaluation is already batched across 64-128 concurrent games per
process and costs almost nothing on a GPU. The descent is pure Python:
select a child by PUCT, clone a game state (lists of lists), apply a move,
compute the legal moves, encode the leaf. That is what ~90 µs buys, and it
is what a GPU machine's 8 cores spend all their time on while the GPU waits.

The target: **≤ 3 µs per simulation for the search bookkeeping, ≤ 2 µs for
clone + apply + legal moves, ≤ 1 µs to encode**. That makes one core worth
~200k positions/s search-bound, i.e. one `l4x1` job becomes GPU-bound at
several hundred thousand positions/s instead of 24k, and this Mac's GPU
would saturate with two cores instead of six. The whole Porcelain corpus
(100k games at 1,024 sims, 2.5 billion evaluations) took the fleet ~4 hours
and ~$25; at 20× it is a lunch break on one L4.

Non-goals: do not port the net, the trainer, the replay buffer, the hub
plumbing, the browser player (which has its own JS engine twin in
`web/player/js/engine.js` and `mcts.js`, not in scope), or the GUI. Do not
change any rule, target, file format or config key.

## 2. What exists (read these files, in this order)

- `ludometer/azul/engine.py` (870 lines): the rules. `AzulState` with plain
  Python lists; the module docstring gives the state layout; the `encode()`
  layout is the block of `OFF_*` offsets around line 160 (182 floats, from
  the player-to-move's perspective). 50 tests in `tests/test_engine.py`.
- `ludometer/train/mcts.py` (1,064 lines): PUCT search. The module docstring
  is the specification of everything subtle: non-alternating turn order,
  chance edges by re-sampled determinization, tree reuse, decisive play with
  the margin head, time budgets, and the batched leaf protocol. 14 tests in
  `tests/test_train_mcts.py`, 11 in `tests/test_tree_reuse.py`, 30 in
  `tests/test_margin.py`.
- `ludometer/train/selfplay_batched.py` (611 lines): the batched driver:
  `games` concurrent trees, one forward pass per round, playout-cap
  randomization, the GC trick, the worker pool. 20 tests in
  `tests/test_selfplay_batched.py` assert **bit-identical games** between the
  batched and the sequential engine from the same seed: that is the standard
  of equivalence this project uses.
- `ludometer/train/selfplay.py`: the sequential game loop
  (`play_selfplay_game`), `GameRecord`, `SelfPlayConfig`, the target helpers
  (`value_target`, `margin_targets`, `aux_targets`), the worker pool base.
- `ludometer/train/mcts_agent.py`, `ludometer/eval/gauntlet.py`,
  `ludometer/eval/arena.py`: the tournament path (one search per move, a
  `?think=<s>` wall-clock budget, no tree reuse). The gate that decides what
  ships runs here at `think=1.0`.
- `ludometer/cloud/generator.py` and `label.py`: the fleet-side consumers of
  the batched engine (`make_selfplay(kind="batched")`) and of the leaf
  protocol (`BatchLabeler`).
- `ludometer/train/benchmark.py`, `ludometer/cloud/bench.py`: the
  measurements above; your before/after numbers come from the same tools.
- `docs/DESIGN.md`: the original design; `docs/PORCELAIN.md` and
  `docs/superpowers/specs/2026-09-05-cloud-selfplay-design.md`: why speed
  matters and what the fleet looks like.

Two golden datasets you should use as tests:

- `data/cloud/bga_positions.json.gz`: 3,795 real human games (deals scripted
  per round, action ids, first seat, final scores, outcome). Every one of them
  replays exactly in the Python engine (`ludometer.cloud.label.replay_positions`)
  and reaches the reported final scores. The Rust engine must do the same,
  position for position, encoding for encoding.
- Any `runs/<run>/checkpoints/replay.npz` or cloud shard
  (`data/cloud/rlx_teacher/shards/*.npz`, `ludometer.cloud.shards.read_shard`):
  encoded states produced by the Python engine during search, useful for
  encode() parity on positions self-play actually reaches.

## 3. The rules, and the places where they bite

Official 2-player Azul (5 factories of 4 tiles, 100 tiles, 20 per colour).
Actions are `source * 30 + color * 6 + dest` with source 0-4 factories, 5
center; colour 0-4 = blue, yellow, red, black, teal; dest 0-4 pattern rows,
5 floor. Wall column of colour `c` in row `r` is `(c + r) % 5`. Floor
penalties `(-1, -1, -2, -2, -2, -3, -3)`, scores clamp at 0. Row bonus 2,
column bonus 7, colour bonus 10. The game ends after the round in which a
wall row completes; a truncated game (`max_moves`, 400) scores as a draw.

Things that already cost an agent a day each; read `engine.py` for the exact
handling rather than re-deriving them:

- **Turn order does not alternate.** The marker holder starts the next round,
  so a player can move twice in a row across a round boundary. Nothing in the
  search may assume alternation: values are propagated in player 0's frame
  and flipped per node by `node.player` (see `_simulate`, `_backup`).
- **The refill is the only chance event.** A move is stochastic iff it takes
  the last tiles of the round (`pool[color] == tiles_left`). The bag *contents*
  are public, the *order* is not: the engine pre-shuffles the bag, so a plain
  clone would let the search peek at the next deal. `determinize(action, seed)`
  reseeds the clone's RNG, reshuffles the bag, applies the move. Post-refill
  positions are keyed by factory + center contents (`chance_key`); distinct
  outcomes per edge are capped at `chance_children`, past which a traversal
  picks a stored one uniformly. Edge statistics live in the parent, so a
  chance edge's Q is the visit-weighted mean over its determinizations, and
  `tests/test_train_mcts.py` checks that identity numerically.
- **Bag refill from the lid** when the bag runs short mid-refill (`_refill`),
  and a short deal at the very end of the bag is legal; **the BGA replays
  script the deal** (`ludometer/human/convert.py::apply_deal`) by taking the
  observed tiles out of bag + lid + the engine's own draw, so a Rust engine
  needs the same hook: "replace the refill the engine just made with this
  one".
- **Two deterministic arg-max players can loop forever** (nobody completes a
  line; tiles cycle bag → floor → lid). Self-play re-introduces sampling past
  `stall_rounds`; the arena has a `max_moves` backstop. Keep both.
- **Tile conservation** is checked at every step in tests (`tile_census` ==
  20 per colour); make it a debug assertion in Rust too.
- `clone()` must not seed anything (Python's `random.Random()` seeding from
  the OS twice was 20% of the search until `_NEW_RANDOM` skipped it). In Rust
  a state should be a small `Copy`-able struct: 2 × 25 wall bits, 2 × 5
  pattern-line colours and counts, 2 × 5 floor counts, 5 × 5 factory counts,
  5 center counts, 5 bag counts + an order, 5 lid counts, scores, flags. No
  heap allocation on clone.

## 4. The search, exactly

PUCT with edge stats `(N, W, P)` in the parent, `score = Q + c_puct * P *
sqrt(N_parent + 1) / (1 + N)`, `Q = fpu` for an unvisited edge, one leaf
expanded per simulation, value backed up along the path with per-node frame
flips. Read `MCTSConfig` (line 228) for every knob; all are config-driven and
must keep their meaning:

- `sims`, `c_puct`, `fpu`;
- root Dirichlet noise: `alpha = max(dirichlet_alpha_scale / n_legal, 1e-3)`,
  mixed with `dirichlet_eps`, **only on the root's priors**, only when the
  caller asks (`add_noise`), and never on a one-move root (so the RNG streams
  of noisy and forced roots stay in step);
- `chance_children`, `chance_backup = "mean"` (the only rule);
- `tree_reuse`: `advance(action)` keeps the chosen child as the next root
  *only across a deterministic edge*; a fingerprint mismatch falls back to a
  fresh root; the budget is a total (`sims - root.N` new simulations); noise
  is re-mixed at the reused root on raw priors (exact, see the docstring);
- margin head: every backup carries a pair `(v, m)`; PUCT uses `v` only; the
  visit distribution is the policy target unchanged; `select_play_action`
  breaks ties among adequately visited, equally winning children by margin
  (`decisive_eps`, `decisive_min_visit_frac`);
- within-tree batching (`search_batch`, `search_batch_ramp`,
  `search_min_batch`, `virtual_loss`): off by default (`search_batch = 1`);
  when on, several descents are made before one forward pass, each laying a
  virtual loss. Keep it, the browser and some configs use it;
- **the leaf protocol** (`start_search` / `leaf_requests` / `apply_leaves` /
  `search_done` / `finish_search`): the search is driven from outside and
  never calls the evaluator itself. This is the interface the batched driver,
  the labeller and the trainer's tests build on, and it is what makes
  cross-game batching possible. The Rust engine must expose the same shape
  (see §5), because the net stays on the Python side;
- time budget (`search(state, time_limit_s=...)`): keep simulating until the
  clock runs out; this is the gate's mode and the GUI's;
- `SearchResult`: policy (visit distribution over 180 actions, 0 on illegal),
  root value, visits per action, sims, elapsed, and with a margin head the
  per-action win-Q and margin-Q plus the root margin. The **root value** is
  `sum(root.wins) / total` in the root player's frame, and since 2026-09-05
  it is stored per position as `search_values` (see `GameRecord`).

Playout-cap randomization (`SelfPlayConfig.pcr_*`, `pcr_sims`, `pcr_rng`):
per move, draw whether the search is the full or the cheap one from a
per-game RNG seeded from the game seed, taken at the same point of the move
loop in every engine; a cheap search stores a zeroed policy with
`policy_mask = 0` and no root noise. Match the draw order.

The game loop (`play_selfplay_game` and `BatchedSelfPlay._pump`) records per
position: `encode()`, the policy target, the mover, `policy_mask`,
`search_values`/`search_mask`; and per game the outcome, scores, moves,
rounds, seed, decisions (moves with more than one legal action), evals,
duration, truncated. Temperature `temperature` for the first `temp_moves`
moves and past `stall_rounds`, else the decisive pick. Value targets are the
outcome (± the score blend `value_score_weight`, historically 0.15, 0 in
every run since run4) in the mover's frame; margins `tanh(diff / 20)`; aux
targets the 30 final-wall bits (`wall_summary`) in the mover's frame.

## 5. The interface to build

One Rust crate, `ludometer-engine` (workspace under `rust/`), exposed to
Python with PyO3 + maturin as the optional package `ludometer_rs`. Three
layers, each independently testable:

1. **`azul` module**: `State` (`Copy`), `new_game(seed)`, `clone`,
   `legal_actions() -> SmallVec<u8>`, `is_legal`, `apply(action)`,
   `is_stochastic(action)`, `determinize(action, seed)`, `chance_key`,
   `fingerprint`, `encode(&mut [f32; 182])`, `outcome`, `scores`,
   `wall_summary`, `tile_census`, `apply_deal(factories)` (the BGA hook),
   plus `is_terminal`, `current_player`, `round_index`, `tiles_left`,
   `first_player`. The RNG is a small explicit PRNG owned by the state
   (splitmix/PCG); it is only consumed at refill and by `determinize`.
2. **`mcts` module**: `Tree` with an arena of nodes (`Vec<Node>`, edges as
   contiguous slices, children indices, chance tables as small vectors of
   `(key, node_idx)`), `MctsConfig` mirroring `MCTSConfig`, and the leaf
   protocol: `start_search(state, add_noise, sims)`, `leaf_requests(max) ->
   &[LeafRef]` giving each pending leaf's encoded row and legal list,
   `apply_leaves(priors_by_leaf, values, margins)`, `search_done`,
   `finish_search -> SearchResult`, `advance(action)`, `reset`. Plus the
   convenience `search(state, evaluator_callback, sims | time_limit)`.
3. **`arena` module**: many games, one batch: `Arena::new(config, games,
   seeds)`, `step() -> (obs: [B, 182] f32, legal: ragged, slot ids)`,
   `apply(policies: [B, 180] f32 or per-leaf priors, values: [B], margins:
   [B])`, `drain() -> Vec<GameRecord>`, `set_stop()`. This replaces
   `BatchedSelfPlay._play` + `_pump` + `_play_searched_move` + `_finish`; the
   Python side keeps only the evaluator call and the shard/record plumbing.
   The same arena, with `add_noise = false` and one position per slot, is the
   labeller (`ludometer.cloud.label.BatchLabeler`).

Python integration, in this order, each step a separate PR:

- `ludometer/azul/engine_rs.py`: `AzulState` API-compatible wrapper over the
  Rust state (same method names and return types, numpy `encode()`), selected
  by `LUDOMETER_ENGINE=rust` or a config key `engine: "rust"`, default
  unchanged. `ludometer/games.py` registers it as a game spec so the trainer,
  arena and gauntlet can use it without edits.
- `ludometer/train/mcts_rs.py`: `MCTS` API-compatible wrapper over `Tree`
  (the leaf protocol first, `search()` second), same selection rule.
- `ludometer/train/selfplay_rust.py`: the `make_selfplay(kind="rust")`
  engine built on `Arena`, with the pool interface (`start`, `set_weights`,
  `play(n, seed, progress, should_stop, on_record)`, `close`) and the
  per-tick `("progress", worker_id, positions)` messages the fleet relies on.
  `ludometer.cloud.generator` gets `--engine rust`.
- Packaging: `maturin develop` for local work; wheels for `manylinux
  x86_64` and `macosx arm64` built in CI and published as a private artifact
  the fleet bootstrap (`ludometer/cloud/fleet.py::BOOTSTRAP`) can `pip
  install`. Until then the bootstrap can `cargo build` in the job (Rust
  toolchain install adds ~2 minutes; acceptable for an 8-hour job).

## 6. Acceptance: what "bit-identical" means here

The project's standard, set by `tests/test_selfplay_batched.py`, is games
that are identical **given identical evaluations**. Floating-point
evaluations differ across batch shapes and backends (documented there: 1e-8
at batch 8), so equivalence is layered:

1. **Rules, exact.** For 10,000 random-play games from seeds 0..9999, the
   Rust and Python engines must agree on every legal-move list, every
   `apply` result, every `encode()` row (exact float equality, they are
   integers scaled), every score and outcome. Same for all 3,795 BGA games
   (`replay_positions`), including the scripted deals. Same `chance_key`,
   `fingerprint` and `is_stochastic` on every position visited.
2. **Search, exact, no randomness.** With `UniformEvaluator` or the
   deterministic `HeuristicEvaluator` from the tests, `add_noise = false`,
   `chance_children = 1` with a fixed determinization seed, the visit counts,
   root value and policy from `search(state, sims)` must equal the Python
   ones for a few hundred positions. Then with tree reuse across a full game.
   Then with `search_batch > 1` and virtual loss (the bookkeeping is exact
   too: `test_pumped_search_is_identical_to_the_blocking_one`).
3. **Search with randomness, statistical.** Root noise and determinization
   draws come from the Rust PRNG, which is not Python's Mersenne Twister and
   need not be. Acceptance: over 200 self-play games with the same net,
   the distributions of game length, outcome, decisions per game, evals per
   game, mean root value and mean policy entropy match the Python engine's
   within sampling error, and a checkpoint's fixed-sims rating on the ladder
   (`eval_games` 40 vs the anchors, as `trainer.py::_evaluate` does) agrees
   within one standard error. A cheap extra: with `LUDOMETER_RNG=python`
   the Rust engine may accept a caller-provided stream of uniforms for the
   noise and the determinization seeds, which makes 3 exact as well; nice to
   have, not required.
4. **The full stack.** `configs/smoke5.json` run end to end with
   `engine: "rust"` (tests/test_train_run.py style), the hub loop test in
   `tests/test_cloud.py` with the rust engine, and one real gauntlet:
   `runs/porc_w-p0905-2038/checkpoints/ckpt-000000.pt` (Porcelain) against
   itself, Rust vs Python, 100 games at `?sims=400`, must be 50 ± 10%.
5. **Speed.** `ludometer.train.benchmark --games 4` and `ludometer.cloud.bench`
   before/after on this Mac and on one `l4x1` job (`fleet launch --entry
   bench`, $0.10). Report positions/s search-bound (tiny net) and with the
   7M teacher; the target is the table in §1.

## 7. Working rules

- The Python engine stays the reference and stays in the repo; the Rust one
  is opt-in until every acceptance point above is green, then becomes the
  default for self-play only. The gate (`gauntlet ... ?think=1.0` on this
  Mac) keeps using whichever engine is the default for the *browser's*
  behaviour to be represented, and the browser's engine is JS. Document
  which engine a gate used in its JSON.
- Other agents work in this repo daily (a Phase B polish run, teacher
  generation jobs, the BGA crawl). Commit path-scoped; never touch
  `data/human/` state, `.bga_cookies.txt`, `runs/*/` of a running run, or a
  running process. Check `ps` and `runs/*/status.json` before claiming the
  Mac's cores; `nice -n 15` anything long.
- Coordinate through files: log progress and negative results in
  `NOTES_FOR_REMI.md` (newest on top) and keep this document current.
- Budget for cloud checks: state flavor × timeout before launching, every
  job through `ludometer.cloud.fleet` so the ledger sees it.

## 8. Expected shape of the work

1. Rules crate + exact tests against the Python engine and the BGA replays.
   This is the bulk of the risk; the rest is bookkeeping. (~1 day.)
2. Tree + leaf protocol + exact search tests. (~1 day.)
3. Arena + Python wrappers + the batched-equivalence tests. (~1 day.)
4. Fleet integration, wheels, before/after numbers, a corpus generated with
   it, one student trained on that corpus and gated to prove nothing moved.
   (~half a day.)

If step 1 shows the engines disagree on a rule, the Python engine is right
until proven otherwise by a BGA replay: it has survived 3,795 of them.

## 9. Status and results (2026-09-06, the night after the brief)

Shipped, opt-in, on `main` (`rust/ludometer-engine`, package `ludometer_rs`;
plan and per-task checklist in
`docs/superpowers/plans/2026-09-06-rust-engine.md`; running log in
`NOTES_FOR_REMI.md`).

| layer | file | Python twin | selected by |
|---|---|---|---|
| rules | `rust/ludometer-engine/src/azul.rs` | `ludometer/azul/engine.py` | `ludometer.azul.engine_rs.AzulState`, `LUDOMETER_ENGINE=rust` |
| tree + leaf protocol | `src/mcts.rs` | `ludometer/train/mcts.py` | `ludometer.train.mcts_rs.MCTS`, `?engine=rust` in agent specs |
| arena | `src/arena.rs` | `ludometer/train/selfplay_batched.py` | `ludometer.train.selfplay_rust`, `"selfplay": "rust"`, `generator --engine rust`, `fleet launch --rust` |
| RNG | `src/rng.rs` | CPython `random.Random` | `rng="python"` (tests) / `rng="fast"` (default, splitmix64) |

Design decisions that differ from §5, and why:

- **The state carries its bag order** (`[u8; 100]`, popped from the end like
  `list.pop()`) and a 3-word RNG descriptor instead of a count-only bag: with
  `rng="python"` the Rust engine replays CPython's Mersenne Twister exactly
  (seeding, `shuffle`, `randrange`, `random()`), so "same seed, same game"
  holds against the Python engine and the parity tests need no scripted deals.
  The MT state is rebuilt from `(seed, outputs consumed)` at the rare real-game
  refill; the state stays `Copy` (~250 bytes).
- **Dirichlet noise is the one stream not reproduced** (numpy's PCG64 + gamma
  sampler): exact tests run with `add_noise=False` or `dirichlet_eps=0`; the
  noisy comparison is statistical.
- **`advance()` re-roots without copying; the kept subtree is compacted into a
  scratch arena at the next `start_search`**, so memory is bounded by two trees
  whatever the game length.
- **Softmax over legal logits happens in Rust** (`apply_logits`, float32 in
  numpy's operation order) so the Python driver only runs the net; the exact
  tests use `apply_leaves` with numpy-computed priors.

Acceptance (§6), as of this writing:

1. **Rules, exact**: 10,000 random-play games from seeds 0..9999 and all 3,795
   BGA replays, every observable identical (`tests/test_rust_engine.py`,
   `LUDOMETER_SLOW=1` runs the full sets in ~100 s).
2. **Search, exact**: visits/policy/value/Q/margins identical to `MCTS` on
   isolated positions, whole games with tree reuse, the pumped leaf protocol,
   `search_batch=4` with virtual loss, margin head (`tests/test_rust_mcts.py`).
   The arena reproduces `BatchedSelfPlay`'s `GameRecord`s array for array
   (`tests/test_rust_arena.py`).
3. **Statistical**: 200 games per engine, tiny net, noise on: moves 63.5±0.7 vs
   63.7±0.8, outcome -0.05±0.07 vs -0.06±0.07, decisions 61.5 vs 61.7, evals/game
   4406±57 vs 4484±65, mean root value, mean policy entropy 2.523±0.011 vs
   2.517±0.010: every |z| < 1. Fixed-sims ladder rating: not run yet.
4. **Full stack**: `configs/smoke5_rust.json` end to end + resume, the hub loop
   with `--engine rust` (`tests/test_rust_stack.py`). Porcelain Rust vs Python at
   `?sims=400`, 100 games: see `runs/gates/rust_vs_python_porcelain_sims400.json`
   (running at the time of writing; the note in `NOTES_FOR_REMI.md` has the result).
5. **Speed** (Mac M3 Pro, fully loaded by two training jobs and a gauntlet, so
   absolute numbers are pessimistic): tree walk **0.41 µs/simulation** with a
   constant evaluator (target ≤ 6; Python ~90), clone+apply+legal 0.23 µs,
   encode 0.17 µs (`cargo run --release --example bench_tree`). End to end,
   tiny net on CPU, one driver, 32 games x 256 sims: 10.9k evals/s Rust vs 3.2k
   Python batched (search 1.3 s vs 31 s, torch 17 s vs 32 s). The forward pass
   is now the whole cost, so **the lever is batch size**: `selfplay_games`
   256-1024 per driver (~0.5 MB per tree at 256 sims), 1-2 drivers per GPU.
   **l4x1** (job `6a9ca323e686246ca69a4824`, Porcelain teacher weights, 2048
   sims, 256 games, one driver): Python batched on cuda fp16 6,203
   positions/s; **Rust on cuda fp16 107,207 positions/s (17x)**; on the job's
   CPU 2,660 vs 4,349 (the 3.9M forward pass binds there). The L4 forward pass
   is 1.09 ms at batch 256 (235k positions/s) and flat from batch 16, so the
   Rust driver's remaining cost is ~1 ms of per-round overhead: run 2-3 drivers
   x 512 games per job to sit at the GPU ceiling. The crate builds in 35 s
   inside the job (`fleet launch --rust`).

Not done: prebuilt wheels (the job builds from source behind `--rust`, ~2-3
minutes); a corpus generated with the Rust engine and a student gated on it;
the GUI (Python engine, by design). `engine_rs.AzulState` attributes are
copies, so code that edits `state.factories[...]` by hand must use the Python
engine or `to_dict`/`from_dict`.
