# Rust Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Rust crate (`rust/ludometer-engine`, Python package `ludometer_rs`) that owns Azul's rules, the PUCT tree walk and the many-games arena, bit-compatible with the Python engine, so self-play generation gets ≥ 20× faster per core.

**Architecture:** Three Rust layers (`azul` rules, `mcts` tree + leaf protocol, `arena` many-games driver), each mirrored 1:1 on the Python module it replaces (`ludometer/azul/engine.py`, `ludometer/train/mcts.py`, `ludometer/train/selfplay_batched.py`). PyO3/maturin exposes them as `ludometer_rs`; thin Python wrappers (`engine_rs.py`, `mcts_rs.py`, `selfplay_rust.py`) plug them into the existing registry, trainer, gauntlet and fleet. The net stays in PyTorch: the Rust side hands out `[B, 182]` float32 rows and takes back priors/values/margins.

**Tech Stack:** Rust 1.93 (stable), PyO3 0.24 + numpy crate, maturin (installed into `.venv` with `uv pip install`, never via `pyproject.toml` so the other agent's `uv run` never triggers a cargo build), pytest.

**Spec:** `docs/RUST_ENGINE.md` (read it first; §3 lists the rule traps, §4 the search, §6 the acceptance layers).

**Status (2026-09-06 morning):** Tasks 1-6 done and pushed; see `docs/RUST_ENGINE.md` §9 for results. Deviations from this plan: the state keeps its bag order and a CPython-exact MT19937 (`rng="python"`) so parity needs no scripted deals; the arena's `step()` became `gather()`/`observations()`/`apply_logits()`/`drain()`; wheels in CI were not built (the job builds the crate from source in 35 s behind `fleet launch --rust`). Open: the Porcelain Rust-vs-Python gauntlet result (`runs/gates/rust_vs_python_porcelain_sims400.json`).

## Global Constraints

- No rule, target, file format or config key changes. The Python engine stays the reference.
- Every float the Python side computes in float64 then stores as float32 is computed the same way in Rust (`(a as f64 / b as f64) as f32`), never with f32 arithmetic.
- Edge statistics are f64, priors are f64 converted from the f32 the net returns (`float(p)` in Python), and every expression keeps Python's evaluation order (`q + (scale * prior) / (1 + n)`, `wins += vl + v0 * flip`).
- The Rust `State` is `Copy`, ~200 bytes, no heap: bag order is a `[u8; 100]` + length (popped from the end like Python's `list.pop()`), the RNG is `{kind, seed: u64, consumed: u32}`.
- Two RNG kinds. `Fast` (splitmix64) for production. `Python` (MT19937 + CPython's `seed(int)`, `getrandbits`, `_randbelow`, `shuffle`, `random()`), which reproduces `random.Random` exactly and exists so the rules and the search can be tested **bit-exact** against Python (acceptance layers 1 and 2). The MT state is never stored: it is rebuilt from `(seed, consumed)` on demand (a refill happens once per ~20 real moves, a determinization reseeds anyway).
- Dirichlet noise is never exact across engines (numpy's PCG64 + gamma sampler is not reproduced); exact tests use `add_noise=False` or `dirichlet_eps=0.0`.
- Commit path-scoped (`rust/`, `ludometer/azul/engine_rs.py`, `ludometer/train/mcts_rs.py`, `ludometer/train/selfplay_rust.py`, `tests/test_rust_*.py`, `docs/RUST_ENGINE.md`, `NOTES_FOR_REMI.md`, this plan). Never `git add -A`. Another agent has uncommitted work in `ludometer/human/`, `web/`, `runs/`.
- `nice -n 15` every cargo build and every long test; the Mac runs a training job and a gauntlet.

---

### Task 1: Crate skeleton, Python-compatible MT19937, splitmix64

**Files:**
- Create: `rust/Cargo.toml` (workspace), `rust/ludometer-engine/Cargo.toml`, `rust/ludometer-engine/src/lib.rs`, `rust/ludometer-engine/src/rng.rs`
- Create: `rust/ludometer-engine/pyproject.toml` (maturin), `rust/README.md`
- Test: `rust/ludometer-engine/src/rng.rs` unit tests with vectors generated from CPython

**Interfaces:**
- Produces: `pub struct Mt19937 { mt: [u32; 624], idx: usize }` with `seed_int(u64)`, `genrand_u32()`, `getrandbits(k: u32) -> u64`, `randbelow(n: u64) -> u64`, `shuffle<T>(&mut [T])`, `random() -> f64`, `skip(n)`; `pub struct SplitMix64(u64)` with `next_u64`, `below(n)`, `shuffle`, `f64()`.
- Produces: `pub enum RngKind { Fast, Python }` and `pub struct Rng { kind, seed: u64, consumed: u32 }` with `seed(kind, seed)`, `shuffle(&mut self, &mut [u8])`, `below(n)`, `random()`; for `Python` it re-seeds an MT and skips `consumed` outputs, then records the new `consumed`.

Steps:
- [x] Vectors: `python -c "import random; r=random.Random(12345); print([r.getrandbits(32) for _ in range(5)]); r=random.Random(12345); x=list(range(10)); r.shuffle(x); print(x); print(r.random()); print(r.randrange(7))"` and the same for seed 0 and seed 2**40+3 (multi-word key).
- [x] Write failing Rust unit tests with those vectors; `cargo test -p ludometer-engine rng` fails.
- [x] Implement `init_by_array`, `genrand_u32`, `getrandbits` (k ≤ 32: `u >> (32-k)`; k > 32: little-endian 32-bit words), `randbelow` (k = bit_length(n); rejection), `shuffle` (`for i in (1..n).rev() { j = randbelow(i+1); swap }`), `random()` (`a = u>>5, b = u>>6, (a*67108864 + b) / 2^53`).
- [x] `cargo test` passes. Commit: `rust: crate skeleton, CPython-exact MT19937 and splitmix64`.

### Task 2: Azul rules in Rust, exact against Python

**Files:**
- Create: `rust/ludometer-engine/src/azul.rs`
- Create: `rust/ludometer-engine/src/py.rs` (PyO3 `State` class), `tests/test_rust_engine.py`
- Modify: `rust/ludometer-engine/src/lib.rs`

**Interfaces:**
- `#[derive(Clone, Copy)] pub struct State { factories: [[u8;5];5], center: [u8;5], bag: [u8;100], bag_len: u8, lid: [u8;5], walls: [u32;2] (bit r*5+col), pl_color: [[i8;5];2], pl_count: [[u8;5];2], floor: [[u8;5];2], floor_marker: [bool;2], marker_in_center: bool, scores: [i32;2], current_player: u8, first_player: u8, round_index: u16, tiles_left: u8, is_terminal: bool, exhausted: bool, open_mask: [[u8;5];2], rng: Rng }`
- `State::new_game(seed: u64, kind: RngKind)`, `legal_actions(&self, out: &mut Vec<u8>)` (same order as Python: factories 0..4 then center, colours 0..4, rows ascending then floor), `is_legal`, `apply(action) -> Result<(), &'static str>`, `is_stochastic`, `determinize(action, seed)`, `chance_key() -> [u8; 30]`, `fingerprint() -> u64` (hash of the same tuple), `encode(&self, out: &mut [f32; 182])`, `outcome() -> Option<f32>`, `wall_summary(p) -> [u8; 15]`, `tile_census() -> [u8;5]`, `apply_deal(factories: [[u8;5];5]) -> Result` (Python `convert.apply_deal`: pool = bag + lid + engine deal; new_bag = remaining − lid, or the merge; the bag list becomes **sorted by colour**, unshuffled), `completed_rows/cols/colors`, `floor_occupied`, `recount`.
- PyO3 `State`: all of the above as methods, plus `to_lists()` (dict mirroring the Python attributes for the wrapper) and `from_python(dict)` (build from a Python `AzulState`, bag order included).

Steps:
- [x] Failing pytest: `tests/test_rust_engine.py::test_random_play_matches_python` — for seeds 0..199 (fast) and a `--slow` marker for 0..9999: play random moves (`random.Random(seed)` picks an index into the **Python** legal list); at every step assert legal lists, `encode()` rows (`np.array_equal`), `chance_key`, `fingerprint` equality class, `is_stochastic` on every legal action, scores, `is_terminal`, `outcome`, `tile_census == [20]*5`. Python-RNG mode, same seed ⇒ same deals.
- [x] Failing pytest: `test_bga_replays_match_python` — all 3,795 games of `data/cloud/bga_positions.json.gz` through `replay_positions` on both engines (Rust via `apply_deal`), encode rows identical, final scores identical.
- [x] Failing pytest: `test_determinize_matches_python` on `near_round_end_state()`-style positions for 50 seeds.
- [x] Implement `azul.rs` (mirror `engine.py` function by function, keep `_refill`'s pop-from-the-end and lid merge order), `py.rs`, build with `nice -n 15 .venv/bin/maturin develop --release -m rust/ludometer-engine/Cargo.toml`.
- [x] Both tests green; also `cargo test` unit tests (census, wall scoring on a hand-built wall, floor overflow to lid).
- [x] Commit: `rust: Azul rules exact against the Python engine (random play + 3,795 BGA replays)`.

### Task 3: Tree + leaf protocol, exact against Python MCTS

**Files:**
- Create: `rust/ludometer-engine/src/mcts.rs`, `tests/test_rust_mcts.py`
- Modify: `rust/ludometer-engine/src/py.rs` (`MctsConfig`, `Tree` classes)

**Interfaces:**
- `pub struct MctsConfig { sims, c_puct, dirichlet_alpha_scale, dirichlet_eps, chance_children, fpu, tree_reuse, decisive_eps, decisive_min_visit_frac, search_batch, search_batch_ramp, search_min_batch, virtual_loss }` — same defaults as `MCTSConfig`.
- `pub struct Node { state: State, player: u8, legal_start: u32, n_legal: u16 (edges are slices into tree-level Vecs: legal: Vec<u8>, priors: Vec<f64>, visits: Vec<u32>, wins: Vec<f64>, margins: Vec<f64>, child: Vec<u32> (NONE=u32::MAX, or a node, or CHANCE|table_idx)), expanded, n_visits: u32, terminal_v0: f64, terminal_m0: f64, pending: u32 (request index during a gather) }`; chance tables `Vec<Vec<([u8;30], u32)>>`.
- `pub struct Tree` with `new(config, has_margin, seed, rng_kind)`, `seed(n)`, `reset_tree()`, `advance(action) -> bool`, `start_search(&State, add_noise, sims: Option<u32>)`, `search_done()`, `leaf_requests(max) -> &[u32]` (request node ids; the PyO3 layer returns `(obs: ndarray [n,182], legal: list[list[int]])`), `apply_leaves(priors: &[Vec<f32>] or flat, values: &[f32], margins: &[f32])`, `finish_search() -> SearchResult { policy: [f32;180], value: f64, visits: Vec<(u8,u32)>, sims, elapsed, has_margin, q, margins, margin }`, `search(&State, eval: FnMut(&State, &[u8]) -> (Vec<f32>, f32, f32), sims, time_limit)`, `evals`, `nodes_created`, `reused_visits`, `rng_random()`, `rng_below(n)` (the tree's own RNG for the driver's move sampling).
- Determinization seed sequence exactly Python's: `counter += 1; seed = (self.seed * 1_000_003 + counter * 2_654_435_761) & 0x7FFF_FFFF`; the chance-table pick past the cap uses `rng.below(len)` (Python mode: MT `randrange`).
- Dirichlet in Fast mode: Marsaglia–Tsang gamma with the boost for alpha < 1, normalised.
- The Node arena is cleared by `reset_tree()`; `advance()` keeps the whole arena and just re-roots (garbage is reclaimed at the next `reset_tree`; a game's tree stays bounded by sims × moves — measure, and compact by copying the kept subtree if memory says so).

Steps:
- [x] Failing pytests (Python mode, `add_noise=False`): (a) `search(sims=160)` on 200 positions from random-play games (seeds 0..19) with `UniformEvaluator` and `ScoreEvaluator` (the tree-reuse test's), `chance_children` 1 and 4: visits dict, policy, value, `sims`, and with a margin evaluator `q`/`margins`/`margin` equal to Python's; (b) tree reuse across whole self-play games driven by both `MCTS.search`+`advance` and `Tree.search`+`advance`, the Rust state re-synced from the Python state each move; (c) the leaf protocol pumped from Python is identical to `search`, with `search_batch = 4, ramp 4, min 1, virtual_loss 1.0` too (mirrors `test_pumped_search_is_identical_to_the_blocking_one`); (d) time budget: `search(time_limit_s=0.05)` returns `sims ≤ config.sims` and > 0; (e) noise changes the policy in Fast mode, sums to 1, and a one-move root consumes none.
- [x] Implement `mcts.rs` mirroring `_simulate`, `_collect`, `_backup`, `_select`, `_child`, `_reuse_for`, `_open_root`, `_result`; `py.rs` classes; build.
- [x] Green. `cargo test` unit tests for `_select` ties (first max wins) and the chance table cap.
- [x] Commit: `rust: PUCT tree with the leaf protocol, exact against ludometer.train.mcts`.

### Task 4: Arena (many games, one batch) + `GameRecord` parity

**Files:**
- Create: `rust/ludometer-engine/src/arena.rs`, `tests/test_rust_arena.py`
- Modify: `rust/ludometer-engine/src/py.rs` (`Arena` class)

**Interfaces:**
- `pub struct SelfPlayConfig { mcts: MctsConfig, temp_moves, temperature, stall_rounds, max_moves, value_score_weight, pcr_full_sims, pcr_cheap_sims, pcr_full_prob }` (from the Python `SelfPlayConfig` fields).
- `pub struct Arena` with `new(config, has_margin, games, rng_kind)`, `play(n_games, seed_start)` bookkeeping, `step(leaf_cap) -> (obs, legal ragged, counts per slot)`, `apply(priors ragged, values, margins)`, `drain() -> Vec<GameRecord>` (finished since the last drain), `finished()`, `positions()`, `set_stop()`; `GameRecord { states: Vec<[f32;182]>, policies: Vec<[f32;180]>, values, margins, aux: Vec<[u8;30]>, policy_mask, outcome, scores, moves, rounds, seed, decisions, evals, duration, truncated, search_values, search_mask }` → PyO3 returns numpy arrays in the Python `GameRecord` layout.
- Per-slot seeds exactly Python's: `Tree` seed `(seed*2+1) & 0x7FFFFFFF`, `pcr_rng` seed `((seed*2+1) ^ 0x9E3779B9)` (Python mode: `random.Random(that).random()`), the `pcr_sims` draw at the same point of the loop, `select_play_action` (temperature sampling with `tree.rng_random()`, `decisive_action` tie order `(margin, visits, -action)`, stall breaker).

Steps:
- [x] Failing pytest: `test_arena_matches_batched_selfplay_given_identical_evaluations` — the tiny torch net from `tests/test_selfplay_batched.py`, `games=1`, `dirichlet_eps=0.0`, Python RNG mode, seeds 0..3: every array of the two `GameRecord`s equal (`states`, `policies`, `values`, `margins`, `aux`, `policy_mask`, `search_values`, `search_mask`), plus `outcome, scores, moves, rounds, decisions, evals, truncated`. Also with `pcr` on (`full_sims 24, cheap_sims 8, full_prob 0.5`) and with `search_batch 2`.
- [x] Failing pytest: isolation — `games=8` records for seed 3 equal `games=1` records with a `UniformEvaluator`-style constant net (no batch-shape float drift).
- [x] Implement `arena.rs` (mirror `_pump`, `_play_searched_move`, `_finish`), `py.rs`; build; green.
- [x] Commit: `rust: many-games arena reproduces BatchedSelfPlay's GameRecords given identical evaluations`.

### Task 5: Python wrappers and registry

**Files:**
- Create: `ludometer/azul/engine_rs.py`, `ludometer/train/mcts_rs.py`, `ludometer/train/selfplay_rust.py`, `tests/test_rust_integration.py`
- Modify: `ludometer/games.py` (register `"azul_rs"` and `engine: "rust"` selection helper), `ludometer/train/selfplay.py::make_selfplay` (`kind="rust"`), `ludometer/train/trainer.py` (accept `"selfplay": "rust"`), `ludometer/cloud/generator.py` (`--engine rust`), `ludometer/eval/gauntlet.py`/`ludometer/agents/registry.py` (`?engine=rust` spec key → `MCTSAgent` built on `mcts_rs.MCTS`).

**Interfaces:**
- `engine_rs.AzulState`: same attribute names as `AzulState` (properties reading the Rust struct: `current_player`, `scores`, `is_terminal`, `round_index`, `tiles_left`, `first_player`, `factories`, `center`, `lid`, `walls`, `pl_color`, `pl_count`, `floor`, `floor_marker`, `marker_in_center`), `new_game(seed)`, `clone`, `legal_actions`, `is_legal`, `apply`, `is_stochastic`, `determinize`, `chance_key`, `fingerprint`, `search_root`, `encode` (numpy), `outcome`, `wall_summary`, `tile_census`, `completed_*`, `render_text`, `to_json`, `ACTION_SPACE`, `ENCODED_SIZE`; `LUDOMETER_ENGINE=rust` selects it in `get_game("azul")`.
- `mcts_rs.MCTS(evaluator, config, seed, add_noise)`: `search`, `start_search`, `leaf_requests` (returns objects with `.state`, `.legal`), `apply_leaves`, `search_done`, `finish_search`, `advance`, `reset_tree`, `seed`, `rng` (an object with `random()`/`randrange`), `evals`, `has_margin`, `config`; returns `ludometer.train.mcts.SearchResult`.
- `selfplay_rust.RustSelfPlay(net_config, config, games, device, max_batch, half)` and `RustSelfPlayPool(...)` with `start`, `set_weights`, `play(n, seed, progress, should_stop, on_record)`, `close`, `worker_positions`, `positions`, the `("progress", worker_id, positions)` tick.

Steps:
- [x] Failing pytests: `get_game("azul")` under `LUDOMETER_ENGINE=rust` returns the wrapper and `test_engine.py`'s core assertions pass on it (parametrize a handful: new game deal size, legal count, apply/census, terminal scoring); `mcts_rs.MCTS` passes `test_train_mcts.py::test_search_returns_a_distribution_over_legal_actions`-style checks; `make_selfplay(kind="rust", workers=1)` plays 4 games with the tiny net and yields valid `GameRecord`s; `RustSelfPlayPool(workers=2)` streams records and progress ticks.
- [x] Implement; green; `uv run pytest tests/test_rust_*.py -q`.
- [x] Commit: `rust: Python wrappers (engine_rs, mcts_rs, selfplay_rust), engine selection, make_selfplay(kind="rust")`.

### Task 6: Full stack, statistics, gauntlet, speed

**Files:**
- Create: `configs/smoke5_rust.json` (smoke5 + `"selfplay": "rust"`), `tests/test_rust_stack.py`
- Modify: `docs/RUST_ENGINE.md` (status + numbers), `NOTES_FOR_REMI.md`, `ludometer/train/benchmark.py` (`--engine rust`), `ludometer/cloud/bench.py` (`--engine`), `ludometer/cloud/fleet.py::BOOTSTRAP` (optional cargo build behind `RLX_RUST=1`).

Steps:
- [x] `smoke5_rust.json` end to end in a subprocess (as `test_train_run.py` does); the hub loop test with the rust engine.
- [x] Statistics: 200 games each engine, the tiny net, same config: game length, outcome, decisions, evals, mean root value, mean policy entropy within 2 SE.
- [x] Gauntlet: Porcelain (`runs/porc_w-p0905-2038/checkpoints/ckpt-000000.pt`) Rust vs Python, 100 games `?sims=400`, `nice -n 15`, must be 50 ± 10%. Write the JSON to `runs/gates/rust_vs_python_porcelain_sims400.json`.
- [x] Speed: `ludometer.train.benchmark --games 4` and `--engine rust`; the raw `Arena` loop with a constant evaluator (search-bound µs/sim), and with the tiny net on MPS. Table into `docs/RUST_ENGINE.md` §9 "Results".
- [x] Commit each; push.

## Self-review

- Spec coverage: §5 layers → Tasks 2/3/4; Python integration order → Task 5; packaging (maturin develop) → Task 1, wheels/CI → left as a follow-up in the doc (Task 6 records it); §6 acceptance 1 → Task 2, 2 → Task 3, 3/4/5 → Task 6.
- Type consistency: `RngKind`, `State`, `Tree`, `Arena`, `MctsConfig`, `SelfPlayConfig`, `GameRecord` named identically in Tasks 1–5.
