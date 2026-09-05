//! Many self-play games, one batch: the twin of
//! `ludometer/train/selfplay_batched.py` (`BatchedSelfPlay._play`, `_pump`,
//! `_play_searched_move`, `_finish`) with the evaluator left to the caller.
//!
//! The caller's loop is
//!
//! ```text
//!     arena.begin(n_games, seed_start);
//!     while !arena.finished() {
//!         let n = arena.gather(leaf_cap);        // seat, pump, collect leaves
//!         if n == 0 { continue; }
//!         let obs = arena.observations();        // [n, 182] f32
//!         ... one forward pass ...
//!         arena.apply_logits(&logits, &values, margins);   // softmax + backup
//!         for record in arena.drain() { ... }
//!     }
//! ```
//!
//! Every per-game stream is seeded exactly as the Python engine seeds it (the
//! tree from `(seed * 2 + 1) & 0x7FFFFFFF`, the playout-cap draw from
//! `(seed * 2 + 1) ^ 0x9E3779B9`), the draw is taken at the same point of the
//! move loop, and the move is picked by the same `select_play_action`, so with
//! `RngKind::Python` and `dirichlet_eps = 0` a game is bit-identical to the
//! Python engine's given identical evaluations (`tests/test_rust_arena.py`).

use std::time::Instant;

use crate::azul::{State, ACTION_SPACE, ENCODED_SIZE};
use crate::mcts::{margin_target, select_play_action, MctsConfig, Tree};
use crate::rng::{Mt19937, RngKind, SplitMix64};

pub const PCR_RNG_SALT: u64 = 0x9E37_79B9;

/// `ludometer.train.selfplay.SelfPlayConfig`, minus the game name (Azul only).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelfPlayConfig {
    pub mcts: MctsConfig,
    pub temp_moves: u32,
    pub temperature: f64,
    pub stall_rounds: u32,
    pub max_moves: u32,
    pub value_score_weight: f64,
    pub pcr_full_sims: u32,
    pub pcr_cheap_sims: u32,
    pub pcr_full_prob: f64,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            mcts: MctsConfig::default(),
            temp_moves: 12,
            temperature: 1.0,
            stall_rounds: crate::mcts::STALL_ROUNDS,
            max_moves: crate::mcts::MAX_GAME_MOVES,
            value_score_weight: 0.15,
            pcr_full_sims: 0,
            pcr_cheap_sims: 0,
            pcr_full_prob: 0.0,
        }
    }
}

impl SelfPlayConfig {
    /// Is playout-cap randomization on?
    pub fn pcr(&self) -> bool {
        self.pcr_full_prob > 0.0 && self.pcr_cheap_sims > 0
    }
}

/// `value_target(outcome, score_diff, config)`: player-0 value target.
pub fn value_target(outcome: f64, score_diff: i32, config: &SelfPlayConfig) -> f64 {
    let w = config.value_score_weight;
    if w <= 0.0 {
        return outcome;
    }
    let margin = (score_diff as f64 / crate::mcts::MARGIN_SCALE).tanh();
    (1.0 - w) * outcome + w * margin
}

/// The playout-cap draw's own generator (`pcr_rng(seed)`).
pub enum ScheduleRng {
    Fast(SplitMix64),
    Python(Mt19937),
}

impl ScheduleRng {
    pub fn new(kind: RngKind, seed: u64) -> Self {
        let s = (seed * 2 + 1) ^ PCR_RNG_SALT;
        match kind {
            RngKind::Fast => ScheduleRng::Fast(SplitMix64::new(s)),
            RngKind::Python => ScheduleRng::Python(Mt19937::seed_int(s)),
        }
    }
    fn random(&mut self) -> f64 {
        match self {
            ScheduleRng::Fast(g) => g.random(),
            ScheduleRng::Python(g) => g.random(),
        }
    }
}

/// `pcr_sims(config, rng)`: `(budget override, is this a full search)`.
pub fn pcr_sims(config: &SelfPlayConfig, rng: &mut ScheduleRng) -> (Option<u32>, bool) {
    if !config.pcr() {
        return (None, true);
    }
    let full = rng.random() < config.pcr_full_prob;
    if full {
        let sims = if config.pcr_full_sims != 0 { config.pcr_full_sims } else { config.mcts.sims };
        return (Some(sims), true);
    }
    (Some(config.pcr_cheap_sims), false)
}

/// One finished game, flat arrays in the `GameRecord` layout.
#[derive(Clone, Debug, Default)]
pub struct GameRecord {
    pub states: Vec<f32>,   // T x 182
    pub policies: Vec<f32>, // T x 180
    pub values: Vec<f32>,
    pub margins: Vec<f32>,
    pub aux: Vec<u8>, // T x 30
    pub policy_mask: Vec<f32>,
    pub outcome: f32,
    pub scores: [i32; 2],
    pub moves: u32,
    pub rounds: u32,
    pub seed: u64,
    pub decisions: u32,
    pub evals: u64,
    pub duration: f64,
    pub truncated: bool,
    pub search_values: Vec<f32>,
    pub search_mask: Vec<f32>,
}

impl GameRecord {
    pub fn len(&self) -> usize {
        self.values.len()
    }
}

struct Slot {
    seed: u64,
    state: State,
    tree: Tree,
    started: Instant,
    states: Vec<f32>,
    policies: Vec<f32>,
    players: Vec<u8>,
    policy_mask: Vec<f32>,
    search_values: Vec<f32>,
    search_mask: Vec<f32>,
    moves: u32,
    decisions: u32,
    searching: bool,
    pending: usize,
    full: bool,
    schedule: ScheduleRng,
}

/// `games` concurrent self-play games whose leaves the caller evaluates.
pub struct Arena {
    pub config: SelfPlayConfig,
    pub has_margin: bool,
    pub games: usize,
    pub rng_kind: RngKind,
    slots: Vec<Slot>,
    n_games: u64,
    seed_start: u64,
    started: u64,
    done: u64,
    stop: bool,
    finished: Vec<GameRecord>,
    /// Leaf requests of the last gather, `(slot index, count)` in batch order.
    pending: Vec<(usize, usize)>,
    pub positions: u64,
    pub batches: u64,
}

impl Arena {
    pub fn new(config: SelfPlayConfig, has_margin: bool, games: usize, rng_kind: RngKind) -> Arena {
        Arena {
            config,
            has_margin,
            games: games.max(1),
            rng_kind,
            slots: Vec::new(),
            n_games: 0,
            seed_start: 0,
            started: 0,
            done: 0,
            stop: false,
            finished: Vec::new(),
            pending: Vec::new(),
            positions: 0,
            batches: 0,
        }
    }

    /// Plan `n_games` games with seeds `seed_start..`; in-flight games are dropped.
    pub fn begin(&mut self, n_games: u64, seed_start: u64) {
        self.slots.clear();
        self.finished.clear();
        self.pending.clear();
        self.n_games = n_games;
        self.seed_start = seed_start;
        self.started = 0;
        self.done = 0;
        self.stop = false;
    }

    /// Abandon the games still in flight (a half-played game has no target).
    pub fn set_stop(&mut self) {
        self.stop = true;
    }

    pub fn finished(&self) -> bool {
        self.stop || self.done >= self.n_games
    }

    pub fn done(&self) -> u64 {
        self.done
    }

    pub fn active(&self) -> usize {
        self.slots.len()
    }

    fn new_slot(&self, seed: u64) -> Slot {
        Slot {
            seed,
            state: State::new_game(seed, self.rng_kind),
            tree: Tree::new(self.config.mcts, self.has_margin, (seed * 2 + 1) & 0x7FFF_FFFF, true, self.rng_kind),
            started: Instant::now(),
            states: Vec::new(),
            policies: Vec::new(),
            players: Vec::new(),
            policy_mask: Vec::new(),
            search_values: Vec::new(),
            search_mask: Vec::new(),
            moves: 0,
            decisions: 0,
            searching: false,
            pending: 0,
            full: true,
            schedule: ScheduleRng::new(self.rng_kind, seed),
        }
    }

    /// One round of the driver loop: seat new games, push every game as far as
    /// it goes without the net, then collect the leaves that need it. Returns
    /// the number of pending leaves (0: nothing to evaluate this round; call
    /// `finished()` and loop).
    pub fn gather(&mut self, leaf_cap: u32) -> usize {
        if self.finished() {
            return 0;
        }
        while self.slots.len() < self.games && self.started < self.n_games {
            let slot = self.new_slot(self.seed_start + self.started);
            self.slots.push(slot);
            self.started += 1;
        }
        let mut i = 0;
        while i < self.slots.len() {
            match Self::pump(&self.config, &mut self.slots[i]) {
                None => i += 1,
                Some(record) => {
                    self.finished.push(record);
                    self.done += 1;
                    self.slots.remove(i);
                }
            }
        }
        self.pending.clear();
        if self.finished() {
            return 0;
        }
        let mut total = 0;
        for (k, slot) in self.slots.iter_mut().enumerate() {
            let n = slot.tree.leaf_requests(leaf_cap).map(|q| q.len()).unwrap_or(0);
            slot.pending = n;
            if n > 0 {
                self.pending.push((k, n));
                total += n;
            }
        }
        total
    }

    /// Encode the pending leaves into `out` (row-major, `ENCODED_SIZE` per row).
    pub fn observations(&self, out: &mut [f32]) {
        let mut at = 0;
        for &(k, _n) in &self.pending {
            let slot = &self.slots[k];
            for req in slot.tree.queue() {
                let row: &mut [f32; ENCODED_SIZE] = (&mut out[at..at + ENCODED_SIZE]).try_into().unwrap();
                slot.tree.node_state(req.node).encode(row);
                at += ENCODED_SIZE;
            }
        }
    }

    pub fn pending_leaves(&self) -> usize {
        self.pending.iter().map(|&(_, n)| n).sum()
    }

    /// Legal lists of the pending leaves, batch order.
    pub fn pending_legal(&self) -> Vec<Vec<u8>> {
        let mut out = Vec::with_capacity(self.pending_leaves());
        for &(k, _) in &self.pending {
            let slot = &self.slots[k];
            for req in slot.tree.queue() {
                out.push(slot.tree.node_legal(req.node).to_vec());
            }
        }
        out
    }

    /// Feed the evaluations back: one `(priors, value, margin)` per pending leaf.
    pub fn apply(&mut self, results: &[(&[f32], f64, f64)]) -> Result<(), String> {
        let total = self.pending_leaves();
        if results.len() != total {
            return Err(format!("expected {total} evaluations, got {}", results.len()));
        }
        let mut at = 0;
        for &(k, n) in &self.pending {
            self.slots[k].tree.apply_leaves(&results[at..at + n])?;
            at += n;
        }
        self.pending.clear();
        self.batches += 1;
        self.positions += total as u64;
        Ok(())
    }

    /// Raw net outputs (`logits` `[n, 180]`, `values`, optional `margins`): the
    /// softmax over each leaf's legal actions is done here in float32.
    pub fn apply_logits(&mut self, logits: &[f32], values: &[f32], margins: Option<&[f32]>) -> Result<(), String> {
        let total = self.pending_leaves();
        if logits.len() != total * ACTION_SPACE || values.len() != total {
            return Err(format!("expected logits [{total}, 180] and values [{total}]"));
        }
        if let Some(m) = margins {
            if m.len() != total {
                return Err(format!("expected margins [{total}]"));
            }
        }
        let mut priors: Vec<Vec<f32>> = Vec::with_capacity(total);
        for legal in self.pending_legal() {
            let k = priors.len();
            priors.push(softmax_over(&logits[k * ACTION_SPACE..(k + 1) * ACTION_SPACE], &legal));
        }
        let results: Vec<(&[f32], f64, f64)> = (0..total)
            .map(|i| (priors[i].as_slice(), values[i] as f64, margins.map(|m| m[i] as f64).unwrap_or(0.0)))
            .collect();
        self.apply(&results)
    }

    /// The games finished since the last call.
    pub fn drain(&mut self) -> Vec<GameRecord> {
        std::mem::take(&mut self.finished)
    }

    // ------------------------------------------------------------------ guts
    /// `_pump`: push a game as far as it can go without the net.
    fn pump(config: &SelfPlayConfig, slot: &mut Slot) -> Option<GameRecord> {
        loop {
            if slot.searching {
                if !slot.tree.search_done() {
                    return None;
                }
                Self::play_searched_move(config, slot);
                slot.searching = false;
                continue;
            }
            if slot.state.is_terminal || slot.moves >= config.max_moves {
                return Some(Self::finish(config, slot));
            }
            let legal = slot.state.legal_actions();
            let mut row = [0.0f32; ENCODED_SIZE];
            slot.state.encode(&mut row);
            slot.states.extend_from_slice(&row);
            slot.players.push(slot.state.current_player);
            if legal.len() == 1 {
                let mut policy = [0.0f32; ACTION_SPACE];
                policy[legal[0] as usize] = 1.0;
                slot.policies.extend_from_slice(&policy);
                slot.policy_mask.push(1.0);
                slot.search_values.push(0.0);
                slot.search_mask.push(0.0);
                slot.state.apply(legal[0]).expect("legal");
                slot.tree.advance(legal[0]);
                slot.moves += 1;
                continue;
            }
            slot.decisions += 1;
            let (sims, full) = pcr_sims(config, &mut slot.schedule);
            slot.full = full;
            slot.tree.start_search(&slot.state, Some(full), sims).expect("start_search on a live state");
            slot.searching = true;
            if slot.tree.search_done() {
                continue;
            }
            return None;
        }
    }

    /// `_play_searched_move`: close the search, record the target, play the move.
    fn play_searched_move(config: &SelfPlayConfig, slot: &mut Slot) {
        let result = slot.tree.finish_search().expect("an open search");
        if slot.full {
            slot.policies.extend_from_slice(&result.policy);
        } else {
            slot.policies.extend_from_slice(&[0.0f32; ACTION_SPACE]);
        }
        slot.policy_mask.push(if slot.full { 1.0 } else { 0.0 });
        slot.search_values.push(if slot.full { result.value as f32 } else { 0.0 });
        slot.search_mask.push(if slot.full { 1.0 } else { 0.0 });
        let stalling = slot.state.round_index as u32 >= config.stall_rounds;
        let explore = slot.moves < config.temp_moves || stalling;
        let action = select_play_action(
            &result,
            if explore { config.temperature } else { 0.0 },
            &mut slot.tree.rng,
            config.mcts.decisive_eps,
            config.mcts.decisive_min_visit_frac,
            stalling,
        ) as u8;
        slot.state.apply(action).expect("the search only returns legal moves");
        slot.tree.advance(action);
        slot.moves += 1;
    }

    /// `_finish`: the record of a finished (or truncated) game.
    fn finish(config: &SelfPlayConfig, slot: &mut Slot) -> GameRecord {
        let state = &slot.state;
        let truncated = !state.is_terminal;
        let outcome = state.outcome().unwrap_or(0.0) as f64;
        let score_diff = state.scores[0] - state.scores[1];
        let v0 = value_target(outcome, score_diff, config);
        let m0 = margin_target(score_diff as f64);
        let walls = [state.wall_summary(0), state.wall_summary(1)];
        let t = slot.players.len();
        let mut values = Vec::with_capacity(t);
        let mut margins = Vec::with_capacity(t);
        let mut aux = Vec::with_capacity(t * 30);
        for &p in &slot.players {
            let (v, m) = if p == 0 { (v0, m0) } else { (-v0, -m0) };
            values.push(v as f32);
            margins.push(m as f32);
            let (me, them) = (p as usize, 1 - p as usize);
            aux.extend_from_slice(&walls[me]);
            aux.extend_from_slice(&walls[them]);
        }
        GameRecord {
            states: std::mem::take(&mut slot.states),
            policies: std::mem::take(&mut slot.policies),
            values,
            margins,
            aux,
            policy_mask: std::mem::take(&mut slot.policy_mask),
            outcome: outcome as f32,
            scores: state.scores,
            moves: slot.moves,
            rounds: state.round_index as u32 + 1,
            seed: slot.seed,
            decisions: slot.decisions,
            evals: slot.tree.evals,
            duration: slot.started.elapsed().as_secs_f64(),
            truncated,
            search_values: std::mem::take(&mut slot.search_values),
            search_mask: std::mem::take(&mut slot.search_mask),
        }
    }
}

/// Softmax over the legal logits only, in float32 (numpy's order of
/// operations: subtract the max, exp, divide by the sum).
pub fn softmax_over(row: &[f32], legal: &[u8]) -> Vec<f32> {
    let mut sel: Vec<f32> = legal.iter().map(|&a| row[a as usize]).collect();
    if sel.is_empty() {
        return sel;
    }
    let mx = sel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in sel.iter_mut() {
        *x = (*x - mx).exp();
        sum += *x;
    }
    for x in sel.iter_mut() {
        *x /= sum;
    }
    sel
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(config: SelfPlayConfig, games: usize, n: u64, seed: u64) -> Vec<GameRecord> {
        let mut arena = Arena::new(config, true, games, RngKind::Fast);
        arena.begin(n, seed);
        let mut obs = Vec::new();
        while !arena.finished() {
            let k = arena.gather(0);
            if k == 0 {
                continue;
            }
            obs.resize(k * ENCODED_SIZE, 0.0);
            arena.observations(&mut obs);
            // A "net": logits from a hash of the row, value from the score field.
            let mut logits = vec![0.0f32; k * ACTION_SPACE];
            let mut values = vec![0.0f32; k];
            let margins = vec![0.0f32; k];
            for i in 0..k {
                let row = &obs[i * ENCODED_SIZE..(i + 1) * ENCODED_SIZE];
                for a in 0..ACTION_SPACE {
                    logits[i * ACTION_SPACE + a] = ((a * 31) % 7) as f32 * 0.1 + row[0] * 0.01;
                }
                values[i] = (row[124] - row[125]) * 2.0;
            }
            arena.apply_logits(&logits, &values, Some(&margins)).unwrap();
        }
        let mut out = arena.drain();
        out.sort_by_key(|r| r.seed);
        out
    }

    #[test]
    fn plays_valid_games_and_records_every_position() {
        let config = SelfPlayConfig {
            mcts: MctsConfig { sims: 24, tree_reuse: true, chance_children: 2, ..Default::default() },
            temp_moves: 4,
            max_moves: 120,
            value_score_weight: 0.0,
            ..Default::default()
        };
        let records = run(config, 3, 5, 100);
        assert_eq!(records.len(), 5);
        for r in &records {
            let t = r.len();
            assert_eq!(r.states.len(), t * ENCODED_SIZE);
            assert_eq!(r.policies.len(), t * ACTION_SPACE);
            assert_eq!(r.aux.len(), t * 30);
            assert_eq!(r.policy_mask.len(), t);
            assert_eq!(r.search_values.len(), t);
            assert_eq!(r.moves as usize, t);
            assert!(r.decisions <= r.moves);
            assert!(r.evals > 0);
            if !r.truncated {
                assert!(r.outcome == 1.0 || r.outcome == -1.0 || r.outcome == 0.0);
            }
            for row in r.policies.chunks(ACTION_SPACE) {
                let s: f32 = row.iter().sum();
                assert!((s - 1.0).abs() < 1e-4, "policy sums to {s}");
            }
        }
    }

    #[test]
    fn a_game_does_not_depend_on_how_many_share_its_batches() {
        let config = SelfPlayConfig {
            mcts: MctsConfig { sims: 16, tree_reuse: true, chance_children: 2, ..Default::default() },
            temp_moves: 4,
            max_moves: 100,
            value_score_weight: 0.0,
            ..Default::default()
        };
        let alone = run(config, 1, 1, 7);
        let together = run(config, 4, 4, 5);
        let same = together.iter().find(|r| r.seed == 7).unwrap();
        assert_eq!(alone[0].states, same.states);
        assert_eq!(alone[0].policies, same.policies);
        assert_eq!(alone[0].values, same.values);
        assert_eq!(alone[0].evals, same.evals);
    }

    #[test]
    fn pcr_masks_cheap_moves() {
        let config = SelfPlayConfig {
            mcts: MctsConfig { sims: 16, chance_children: 2, ..Default::default() },
            temp_moves: 4,
            max_moves: 60,
            value_score_weight: 0.0,
            pcr_full_sims: 16,
            pcr_cheap_sims: 4,
            pcr_full_prob: 0.5,
            ..Default::default()
        };
        let records = run(config, 2, 2, 3);
        let masked: usize = records.iter().map(|r| r.policy_mask.iter().filter(|&&m| m == 0.0).count()).sum();
        let total: usize = records.iter().map(|r| r.len()).sum();
        assert!(masked > 0 && masked < total);
    }
}
