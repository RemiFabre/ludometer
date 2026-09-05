//! PUCT search over Azul states: the twin of `ludometer/train/mcts.py`.
//!
//! Read that module's docstring for the specification (non-alternating turn
//! order, chance edges by re-sampled determinization, tree reuse, the margin
//! head, the leaf protocol, within-tree batching with virtual loss). This file
//! keeps its structure: `simulate` is `_simulate`, `collect` is `_collect`,
//! `backup` is `_backup`, `select` is `_select`, `child` is `_child`. Edge
//! statistics are `f64` (Python floats), priors are `f64` converted from the
//! `f32` the net returns (`float(p)`), and every expression keeps Python's
//! evaluation order so that, given identical evaluations, the two searches
//! produce identical visit counts, values and policies.
//!
//! Storage: nodes live in one `Vec<Node>` per tree, each node's edges are a
//! contiguous slice of the tree-level edge vectors, and a chance edge points
//! at a small table of `(chance_key, node)` pairs in insertion order (Python's
//! dict order). `advance` re-roots without copying; the kept subtree is
//! compacted into a fresh arena when the next search starts, so memory stays
//! bounded by two trees whatever the game length.

use std::time::Instant;

use crate::azul::{State, ACTION_SPACE};
use crate::rng::{Mt19937, RngKind, SplitMix64};

pub const MARGIN_SCALE: f64 = 20.0;
pub const STALL_ROUNDS: u32 = 16;
pub const MAX_GAME_MOVES: u32 = 400;
pub const TIME_CHECK_EVERY: u32 = 8;

/// `tanh(score_diff / 20)`: the margin head's target and the search's backup.
#[inline]
pub fn margin_target(score_diff: f64) -> f64 {
    (score_diff / MARGIN_SCALE).tanh()
}

/// Search hyperparameters; same names and defaults as `MCTSConfig`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MctsConfig {
    pub sims: u32,
    pub c_puct: f64,
    pub dirichlet_alpha_scale: f64,
    pub dirichlet_eps: f64,
    pub chance_children: usize,
    pub fpu: f64,
    pub tree_reuse: bool,
    pub decisive_eps: f64,
    pub decisive_min_visit_frac: f64,
    pub search_batch: u32,
    pub search_batch_ramp: u32,
    pub search_min_batch: u32,
    pub virtual_loss: f64,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            sims: 160,
            c_puct: 1.4,
            dirichlet_alpha_scale: 10.0,
            dirichlet_eps: 0.25,
            chance_children: 4,
            fpu: 0.0,
            tree_reuse: false,
            decisive_eps: 0.03,
            decisive_min_visit_frac: 0.1,
            search_batch: 1,
            search_batch_ramp: 16,
            search_min_batch: 1,
            virtual_loss: 1.0,
        }
    }
}

impl MctsConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.sims < 1 {
            return Err("sims must be >= 1".into());
        }
        if self.chance_children < 1 {
            return Err("chance_children must be >= 1".into());
        }
        if self.search_batch < 1 || self.search_min_batch < 1 {
            return Err("search_batch and search_min_batch must be >= 1".into());
        }
        if self.search_batch_ramp < 1 {
            return Err("search_batch_ramp must be >= 1".into());
        }
        Ok(())
    }
}

const NONE: u32 = u32::MAX;
const CHANCE: u32 = 1 << 31;

/// A search node: one concrete state plus (through `edge_start`) its edges.
#[derive(Clone, Copy)]
pub struct Node {
    pub state: State,
    pub player: u8,
    pub expanded: bool,
    pub n_visits: u32,
    pub terminal_v0: f64,
    pub terminal_m0: f64,
    pub edge_start: u32,
    pub n_legal: u16,
    /// During a gather: index of this node's `LeafRequest`, or `NONE`.
    pending: u32,
}

/// One position waiting for the net, plus the descents that stopped there.
#[derive(Default)]
pub struct LeafRequest {
    pub node: u32,
    pub is_root: bool,
    /// Each path is a slice of `Tree::path_pool`: `(start, len)`.
    pub paths: Vec<(u32, u32)>,
}

/// What one search produced (see `mcts.SearchResult`).
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub policy: [f32; ACTION_SPACE],
    pub value: f64,
    /// `(action, visits)` in legal order.
    pub visits: Vec<(u8, u32)>,
    pub sims: u32,
    pub elapsed_s: f64,
    pub has_margin: bool,
    /// `(action, win-Q)` for visited actions, legal order (margin nets only).
    pub q: Vec<(u8, f64)>,
    pub margins: Vec<(u8, f64)>,
    pub margin: f64,
}

impl SearchResult {
    pub fn argmax_policy(&self) -> usize {
        argmax_f32(&self.policy)
    }
}

/// `np.argmax`: the first maximal index.
pub fn argmax_f32(xs: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in xs.iter().enumerate() {
        if x > xs[best] {
            best = i;
        }
    }
    best
}

/// The search's own generator (`MCTS._rng`): Python's when parity is wanted.
pub enum TreeRng {
    Fast(SplitMix64),
    Python(Mt19937),
}

impl TreeRng {
    fn new(kind: RngKind, seed: u64) -> Self {
        match kind {
            RngKind::Fast => TreeRng::Fast(SplitMix64::new(seed)),
            RngKind::Python => TreeRng::Python(Mt19937::seed_int(seed)),
        }
    }
    /// `rng.random()`.
    pub fn random(&mut self) -> f64 {
        match self {
            TreeRng::Fast(g) => g.random(),
            TreeRng::Python(g) => g.random(),
        }
    }
    /// `rng.randrange(n)`.
    pub fn randrange(&mut self, n: u64) -> u64 {
        match self {
            TreeRng::Fast(g) => g.below(n),
            TreeRng::Python(g) => g.randbelow(n),
        }
    }
}

struct BatchSearch {
    noise: bool,
    cap: u32,
    started: Instant,
    done: u32,
    need_root: bool,
    forced: bool,
    root_value: f64,
    root_margin: f64,
}

/// A PUCT search tree; one per playing agent (it owns its RNGs).
pub struct Tree {
    pub config: MctsConfig,
    pub has_margin: bool,
    pub add_noise: bool,
    pub rng_kind: RngKind,
    seed: u64,
    counter: u64,
    pub rng: TreeRng,
    noise_rng: SplitMix64,
    pub nodes_created: u64,
    pub evals: u64,
    pub reused_visits: u32,

    nodes: Vec<Node>,
    legal: Vec<u8>,
    priors: Vec<f64>,
    visits: Vec<u32>,
    wins: Vec<f64>,
    margins: Vec<f64>,
    child: Vec<u32>,
    chance: Vec<Vec<([u8; 30], u32)>>,

    root: u32,
    reuse_root: u32,
    reuse_fp: Option<crate::azul::Fingerprint>,
    search: Option<BatchSearch>,
    queue: Vec<LeafRequest>,
    path_pool: Vec<(u32, u32)>,
    path_scratch: Vec<(u32, u32)>,
    legal_scratch: Vec<u8>,
    // A second arena the kept subtree is compacted into (swapped, reused).
    scratch: Option<Arena>,
}

/// The node/edge storage, separable so a compaction can swap arenas.
#[derive(Default)]
struct Arena {
    nodes: Vec<Node>,
    legal: Vec<u8>,
    priors: Vec<f64>,
    visits: Vec<u32>,
    wins: Vec<f64>,
    margins: Vec<f64>,
    child: Vec<u32>,
    chance: Vec<Vec<([u8; 30], u32)>>,
}

impl Tree {
    pub fn new(config: MctsConfig, has_margin: bool, seed: u64, add_noise: bool, rng_kind: RngKind) -> Tree {
        let seed = seed & 0x7FFF_FFFF;
        Tree {
            config,
            has_margin,
            add_noise,
            rng_kind,
            seed,
            counter: 0,
            rng: TreeRng::new(rng_kind, seed),
            noise_rng: SplitMix64::new(seed ^ 0xD1CE_D1CE),
            nodes_created: 0,
            evals: 0,
            reused_visits: 0,
            nodes: Vec::new(),
            legal: Vec::new(),
            priors: Vec::new(),
            visits: Vec::new(),
            wins: Vec::new(),
            margins: Vec::new(),
            child: Vec::new(),
            chance: Vec::new(),
            root: NONE,
            reuse_root: NONE,
            reuse_fp: None,
            search: None,
            queue: Vec::new(),
            path_pool: Vec::new(),
            path_scratch: Vec::new(),
            legal_scratch: Vec::with_capacity(64),
            scratch: None,
        }
    }

    /// `MCTS.seed(n)`: reseed every stream and forget the tree.
    pub fn seed(&mut self, n: u64) {
        self.seed = n & 0x7FFF_FFFF;
        self.rng = TreeRng::new(self.rng_kind, self.seed);
        self.noise_rng = SplitMix64::new(self.seed ^ 0xD1CE_D1CE);
        self.counter = 0;
        self.reset_tree();
    }

    /// Forget the kept subtree (a new game, or a caller that lost track).
    pub fn reset_tree(&mut self) {
        self.clear_arena();
        self.root = NONE;
        self.reuse_root = NONE;
        self.reuse_fp = None;
        self.reused_visits = 0;
        self.search = None;
        self.queue.clear();
        self.path_pool.clear();
    }

    fn clear_arena(&mut self) {
        self.nodes.clear();
        self.legal.clear();
        self.priors.clear();
        self.visits.clear();
        self.wins.clear();
        self.margins.clear();
        self.child.clear();
        self.chance.clear();
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    // ------------------------------------------------------------- tree reuse
    /// Follow `action` from the current root; keep its subtree if we can.
    pub fn advance(&mut self, action: u8) -> bool {
        let node = self.root;
        self.root = NONE;
        self.reuse_root = NONE;
        self.reuse_fp = None;
        if node == NONE || !self.config.tree_reuse || !self.nodes[node as usize].expanded {
            return false;
        }
        let n = self.nodes[node as usize];
        let start = n.edge_start as usize;
        let end = start + n.n_legal as usize;
        let Some(index) = self.legal[start..end].iter().position(|&a| a == action) else {
            return false;
        };
        let entry = self.child[start + index];
        if entry == NONE || entry & CHANCE != 0 {
            return false;
        }
        self.root = entry;
        self.reuse_root = entry;
        self.reuse_fp = Some(self.nodes[entry as usize].state.fingerprint());
        true
    }

    /// `_reuse_for`: the kept subtree if it really is `state`'s node, else NONE.
    /// On success the subtree is compacted into a fresh arena (root = node 0).
    fn reuse_for(&mut self, state: &State) -> u32 {
        let root = self.reuse_root;
        self.reuse_root = NONE;
        let fp = self.reuse_fp.take();
        if root == NONE || !self.config.tree_reuse {
            self.clear_arena();
            return NONE;
        }
        let n = self.nodes[root as usize];
        if n.state.is_terminal || !n.expanded || n.n_legal == 0 {
            self.clear_arena();
            return NONE;
        }
        if fp != Some(state.fingerprint()) {
            self.clear_arena();
            return NONE;
        }
        self.compact(root);
        0
    }

    /// Copy the subtree under `root` into the scratch arena and swap.
    fn compact(&mut self, root: u32) {
        let mut dst = self.scratch.take().unwrap_or_default();
        dst.nodes.clear();
        dst.legal.clear();
        dst.priors.clear();
        dst.visits.clear();
        dst.wins.clear();
        dst.margins.clear();
        dst.child.clear();
        dst.chance.clear();
        // Breadth-first copy; `stack` holds (old index, new index).
        let mut stack: Vec<(u32, u32)> = Vec::with_capacity(64);
        let push = |dst: &mut Arena, src: &Tree, old: u32| -> u32 {
            let mut node = src.nodes[old as usize];
            let start = node.edge_start as usize;
            let end = start + node.n_legal as usize;
            node.edge_start = dst.legal.len() as u32;
            node.pending = NONE;
            dst.legal.extend_from_slice(&src.legal[start..end]);
            dst.priors.extend_from_slice(&src.priors[start..end]);
            dst.visits.extend_from_slice(&src.visits[start..end]);
            dst.wins.extend_from_slice(&src.wins[start..end]);
            dst.margins.extend_from_slice(&src.margins[start..end]);
            dst.child.extend_from_slice(&src.child[start..end]);
            dst.nodes.push(node);
            (dst.nodes.len() - 1) as u32
        };
        let new_root = push(&mut dst, self, root);
        stack.push((root, new_root));
        while let Some((old, new)) = stack.pop() {
            let node = self.nodes[old as usize];
            let start = node.edge_start as usize;
            let new_start = dst.nodes[new as usize].edge_start as usize;
            for i in 0..node.n_legal as usize {
                let entry = self.child[start + i];
                if entry == NONE {
                    continue;
                }
                if entry & CHANCE != 0 {
                    let table = &self.chance[(entry & !CHANCE) as usize];
                    let mut new_table = Vec::with_capacity(table.len());
                    for &(key, child_old) in table {
                        let child_new = push(&mut dst, self, child_old);
                        stack.push((child_old, child_new));
                        new_table.push((key, child_new));
                    }
                    dst.chance.push(new_table);
                    dst.child[new_start + i] = CHANCE | (dst.chance.len() as u32 - 1);
                } else {
                    let child_new = push(&mut dst, self, entry);
                    stack.push((entry, child_new));
                    dst.child[new_start + i] = child_new;
                }
            }
        }
        // Swap. The old arena is dropped rather than kept as the next scratch:
        // keeping it doubled a tree's footprint (measured 8 MB per tree at 2048
        // sims with 512 concurrent games), and re-growing the vectors costs a
        // few microseconds per move against milliseconds of search.
        let old = Arena {
            nodes: std::mem::take(&mut self.nodes),
            legal: std::mem::take(&mut self.legal),
            priors: std::mem::take(&mut self.priors),
            visits: std::mem::take(&mut self.visits),
            wins: std::mem::take(&mut self.wins),
            margins: std::mem::take(&mut self.margins),
            child: std::mem::take(&mut self.child),
            chance: std::mem::take(&mut self.chance),
        };
        self.nodes = dst.nodes;
        self.legal = dst.legal;
        self.priors = dst.priors;
        self.visits = dst.visits;
        self.wins = dst.wins;
        self.margins = dst.margins;
        self.child = dst.child;
        self.chance = dst.chance;
        drop(old);
        self.scratch = None;
    }

    // ------------------------------------------------------------------ nodes
    fn new_node(&mut self, state: State) -> u32 {
        self.nodes_created += 1;
        self.legal_scratch.clear();
        state.legal_actions_into(&mut self.legal_scratch);
        let n = self.legal_scratch.len();
        let edge_start = self.legal.len() as u32;
        self.legal.extend_from_slice(&self.legal_scratch);
        self.priors.resize(self.priors.len() + n, 0.0);
        self.visits.resize(self.visits.len() + n, 0);
        self.wins.resize(self.wins.len() + n, 0.0);
        self.margins.resize(self.margins.len() + n, 0.0);
        self.child.resize(self.child.len() + n, NONE);
        let (terminal_v0, terminal_m0) = if state.is_terminal {
            (
                state.outcome().unwrap_or(0.0) as f64,
                margin_target((state.scores[0] - state.scores[1]) as f64),
            )
        } else {
            (0.0, 0.0)
        };
        self.nodes.push(Node {
            state,
            player: state.current_player,
            expanded: false,
            n_visits: 0,
            terminal_v0,
            terminal_m0,
            edge_start,
            n_legal: n as u16,
            pending: NONE,
        });
        (self.nodes.len() - 1) as u32
    }

    #[inline]
    fn edges(&self, node: u32) -> (usize, usize) {
        let n = &self.nodes[node as usize];
        let s = n.edge_start as usize;
        (s, s + n.n_legal as usize)
    }

    /// `Node.init_edges(priors)`: priors are the net's f32 values, kept as f64.
    fn init_edges(&mut self, node: u32, priors: &[f32]) {
        let (s, e) = self.edges(node);
        debug_assert_eq!(e - s, priors.len(), "priors must align with the legal list");
        for (i, &p) in priors.iter().enumerate() {
            self.priors[s + i] = p as f64;
        }
        self.nodes[node as usize].expanded = true;
    }

    pub fn node_state(&self, node: u32) -> &State {
        &self.nodes[node as usize].state
    }

    pub fn node_legal(&self, node: u32) -> &[u8] {
        let (s, e) = self.edges(node);
        &self.legal[s..e]
    }

    pub fn root_state(&self) -> Option<&State> {
        if self.root == NONE {
            None
        } else {
            Some(&self.nodes[self.root as usize].state)
        }
    }

    // ------------------------------------------------------------------ guts
    fn budget(&self, sims: Option<u32>) -> u32 {
        match sims {
            None => self.config.sims,
            Some(n) => n.max(1),
        }
    }

    /// `_apply_noise`: Dirichlet noise mixed into the root's priors.
    fn apply_noise(&mut self, root: u32) {
        let (s, e) = self.edges(root);
        let n = e - s;
        if n < 2 {
            return;
        }
        let alpha = (self.config.dirichlet_alpha_scale / n as f64).max(1e-3);
        let eps = self.config.dirichlet_eps;
        let mut noise = vec![0.0f64; n];
        let mut total = 0.0;
        for x in noise.iter_mut() {
            *x = gamma(&mut self.noise_rng, alpha);
            total += *x;
        }
        if !(total > 0.0) || !total.is_finite() {
            for x in noise.iter_mut() {
                *x = 1.0 / n as f64;
            }
        } else {
            for x in noise.iter_mut() {
                *x /= total;
            }
        }
        for i in 0..n {
            self.priors[s + i] = (1.0 - eps) * self.priors[s + i] + eps * noise[i];
        }
    }

    /// `_select`: PUCT over the node's edges; the first maximum wins ties.
    #[inline]
    fn select(&self, node: u32) -> usize {
        let n = &self.nodes[node as usize];
        let s = n.edge_start as usize;
        let e = s + n.n_legal as usize;
        let scale = self.config.c_puct * ((n.n_visits + 1) as f64).sqrt();
        let fpu = self.config.fpu;
        let mut best = -1e30f64;
        let mut best_i = 0usize;
        for i in s..e {
            let nv = self.visits[i];
            let q = if nv != 0 { self.wins[i] / nv as f64 } else { fpu };
            let score = q + scale * self.priors[i] / (1 + nv) as f64;
            if score > best {
                best = score;
                best_i = i - s;
            }
        }
        best_i
    }

    fn determinize(&mut self, state: &State, action: u8) -> State {
        self.counter += 1;
        let seed = (self.seed.wrapping_mul(1_000_003).wrapping_add(self.counter.wrapping_mul(2_654_435_761))) & 0x7FFF_FFFF;
        state.determinize(action, seed)
    }

    /// `_child`: the node behind edge `index`, creating or sampling it.
    fn child(&mut self, node: u32, index: usize) -> u32 {
        let (s, _) = self.edges(node);
        let entry = self.child[s + index];
        if entry != NONE && entry & CHANCE == 0 {
            return entry;
        }
        let action = self.legal[s + index];
        let parent_state = self.nodes[node as usize].state;
        if entry == NONE && !parent_state.is_stochastic(action) {
            let mut st = parent_state;
            st.apply(action).expect("legal action");
            let c = self.new_node(st);
            self.child[s + index] = c;
            return c;
        }
        let table_idx = if entry == NONE {
            self.chance.push(Vec::with_capacity(self.config.chance_children));
            let t = (self.chance.len() - 1) as u32;
            self.child[s + index] = CHANCE | t;
            t
        } else {
            entry & !CHANCE
        } as usize;
        let len = self.chance[table_idx].len();
        if len >= self.config.chance_children {
            let k = self.rng.randrange(len as u64) as usize;
            return self.chance[table_idx][k].1;
        }
        let st = self.determinize(&parent_state, action);
        let key = st.chance_key();
        if let Some(&(_, c)) = self.chance[table_idx].iter().find(|(k, _)| *k == key) {
            return c;
        }
        let c = self.new_node(st);
        self.chance[table_idx].push((key, c));
        c
    }

    // ---------------------------------------------------------- blocking search
    /// `MCTS.search`: run simulations from `state`, calling `evaluate(state,
    /// legal) -> (priors, value, margin)` one leaf at a time.
    pub fn search<F>(
        &mut self,
        state: &State,
        mut evaluate: F,
        add_noise: Option<bool>,
        time_limit_s: Option<f64>,
        sims: Option<u32>,
    ) -> Result<SearchResult, String>
    where
        F: FnMut(&State, &[u8]) -> (Vec<f32>, f64, f64),
    {
        let noise = add_noise.unwrap_or(self.add_noise);
        let started = Instant::now();
        let mut root = self.reuse_for(state);
        self.reused_visits = if root != NONE { self.nodes[root as usize].n_visits } else { 0 };
        let mut value = 0.0;
        let mut margin = 0.0;
        if root == NONE {
            root = self.new_node(*state);
            if self.nodes[root as usize].state.is_terminal {
                return Err("cannot search a terminal state".into());
            }
            let (v, m) = self.expand(root, &mut evaluate);
            value = v;
            margin = m;
        }
        self.root = root;
        if self.nodes[root as usize].n_legal == 1 {
            return Ok(self.forced_result(root, value, margin));
        }
        if noise {
            self.apply_noise(root);
        }
        let cap = self.budget(sims).saturating_sub(self.nodes[root as usize].n_visits);
        match time_limit_s {
            Some(budget) if budget > 0.0 => {
                let mut done = 0;
                while done < cap {
                    let chunk = TIME_CHECK_EVERY.min(cap - done);
                    for _ in 0..chunk {
                        self.simulate(root, &mut evaluate);
                    }
                    done += chunk;
                    if started.elapsed().as_secs_f64() >= budget {
                        break;
                    }
                }
            }
            _ => {
                for _ in 0..cap {
                    self.simulate(root, &mut evaluate);
                }
            }
        }
        Ok(self.result(root, value, margin, started.elapsed().as_secs_f64()))
    }

    fn expand<F>(&mut self, node: u32, evaluate: &mut F) -> (f64, f64)
    where
        F: FnMut(&State, &[u8]) -> (Vec<f32>, f64, f64),
    {
        let (s, e) = self.edges(node);
        let (priors, value, margin) = {
            let st = self.nodes[node as usize].state;
            let legal = &self.legal[s..e];
            evaluate(&st, legal)
        };
        self.evals += 1;
        self.init_edges(node, &priors);
        (value, margin)
    }

    fn simulate<F>(&mut self, root: u32, evaluate: &mut F)
    where
        F: FnMut(&State, &[u8]) -> (Vec<f32>, f64, f64),
    {
        let mut node = root;
        self.path_scratch.clear();
        let (v0, m0);
        loop {
            let n = self.nodes[node as usize];
            if n.state.is_terminal {
                v0 = n.terminal_v0;
                m0 = n.terminal_m0;
                break;
            }
            if !n.expanded {
                let (value, margin) = self.expand(node, evaluate);
                let flip = if n.player == 0 { 1.0 } else { -1.0 };
                v0 = value * flip;
                m0 = margin * flip;
                break;
            }
            let index = self.select(node);
            self.path_scratch.push((node, index as u32));
            node = self.child(node, index);
        }
        let has_margin = self.has_margin;
        for k in 0..self.path_scratch.len() {
            let (parent, index) = self.path_scratch[k];
            let p = &mut self.nodes[parent as usize];
            let i = p.edge_start as usize + index as usize;
            self.visits[i] += 1;
            p.n_visits += 1;
            let flip = if p.player == 0 { 1.0 } else { -1.0 };
            self.wins[i] += v0 * flip;
            if has_margin {
                self.margins[i] += m0 * flip;
            }
        }
    }

    fn forced_result(&self, root: u32, value: f64, margin: f64) -> SearchResult {
        let (s, _) = self.edges(root);
        let action = self.legal[s];
        let mut policy = [0.0f32; ACTION_SPACE];
        policy[action as usize] = 1.0;
        let n = self.nodes[root as usize].n_visits;
        let (mut v, mut m) = (value, margin);
        if n != 0 {
            v = self.sum_wins(root) / n as f64;
            m = self.sum_margins(root) / n as f64;
        }
        SearchResult {
            policy,
            value: v,
            visits: vec![(action, 1)],
            sims: 0,
            elapsed_s: 0.0,
            has_margin: self.has_margin,
            q: Vec::new(),
            margins: Vec::new(),
            margin: m,
        }
    }

    /// Python's `sum(root.wins)`: left-to-right.
    fn sum_wins(&self, root: u32) -> f64 {
        let (s, e) = self.edges(root);
        let mut t = 0.0;
        for i in s..e {
            t += self.wins[i];
        }
        t
    }

    fn sum_margins(&self, root: u32) -> f64 {
        let (s, e) = self.edges(root);
        let mut t = 0.0;
        for i in s..e {
            t += self.margins[i];
        }
        t
    }

    /// `_result`: the root's edge statistics as a `SearchResult`.
    fn result(&self, root: u32, value: f64, margin: f64, elapsed: f64) -> SearchResult {
        let total = self.nodes[root as usize].n_visits;
        let (s, e) = self.edges(root);
        let mut policy = [0.0f32; ACTION_SPACE];
        let mut visits = Vec::with_capacity(e - s);
        let mut q = Vec::new();
        let mut margins = Vec::new();
        let want_margin = self.has_margin;
        for i in s..e {
            let action = self.legal[i];
            let n = self.visits[i];
            visits.push((action, n));
            if n != 0 && total != 0 {
                policy[action as usize] = (n as f64 / total as f64) as f32;
            }
            if want_margin && n != 0 {
                q.push((action, self.wins[i] / n as f64));
                margins.push((action, self.margins[i] / n as f64));
            }
        }
        let root_value = if total != 0 { self.sum_wins(root) / total as f64 } else { value };
        let root_margin = if total != 0 { self.sum_margins(root) / total as f64 } else { margin };
        SearchResult {
            policy,
            value: root_value,
            visits,
            sims: total,
            elapsed_s: elapsed,
            has_margin: want_margin,
            q,
            margins,
            margin: root_margin,
        }
    }

    // ------------------------------------------------------- batched interface
    /// `MCTS.start_search`: open a search the caller will drive.
    pub fn start_search(&mut self, state: &State, add_noise: Option<bool>, sims: Option<u32>) -> Result<(), String> {
        if self.search.is_some() {
            return Err("a batched search is already in progress".into());
        }
        let noise = add_noise.unwrap_or(self.add_noise);
        let started = Instant::now();
        let mut root = self.reuse_for(state);
        self.reused_visits = if root != NONE { self.nodes[root as usize].n_visits } else { 0 };
        if root == NONE {
            root = self.new_node(*state);
            if self.nodes[root as usize].state.is_terminal {
                self.clear_arena();
                return Err("cannot search a terminal state".into());
            }
        }
        self.root = root;
        let need_root = !self.nodes[root as usize].expanded;
        self.search = Some(BatchSearch {
            noise,
            cap: self.budget(sims).saturating_sub(self.nodes[root as usize].n_visits),
            started,
            done: 0,
            need_root,
            forced: false,
            root_value: 0.0,
            root_margin: 0.0,
        });
        self.queue.clear();
        self.path_pool.clear();
        if !need_root {
            self.open_root();
        }
        Ok(())
    }

    /// `_open_root`: settle the forced case, then mix in the noise.
    fn open_root(&mut self) {
        let root = self.root;
        let forced = self.nodes[root as usize].n_legal == 1;
        let noise = self.search.as_ref().map(|s| s.noise).unwrap_or(false);
        if let Some(st) = self.search.as_mut() {
            if forced {
                st.forced = true;
                return;
            }
        }
        if noise {
            self.apply_noise(root);
        }
    }

    /// Has the open search spent its budget? (`false` with none open.)
    pub fn search_done(&self) -> bool {
        match &self.search {
            None => false,
            Some(st) if st.need_root => false,
            Some(st) => st.forced || st.done >= st.cap,
        }
    }

    pub fn search_open(&self) -> bool {
        self.search.is_some()
    }

    /// `leaf_requests(max_leaves)`: descend until at least one position needs
    /// the net; the pending requests are then `self.queue`.
    pub fn leaf_requests(&mut self, max_leaves: u32) -> Result<&[LeafRequest], String> {
        if self.search.is_none() {
            return Err("no batched search in progress".into());
        }
        if !self.queue.is_empty() {
            return Ok(&self.queue);
        }
        if self.search.as_ref().unwrap().need_root {
            self.queue.push(LeafRequest { node: self.root, is_root: true, paths: Vec::new() });
            return Ok(&self.queue);
        }
        let cfg = self.config;
        let floor = cfg.search_min_batch.min(cfg.search_batch);
        while self.queue.is_empty() && !self.search_done() {
            let root_visits = self.nodes[self.root as usize].n_visits;
            let st = self.search.as_ref().unwrap();
            let mut want = floor.max(cfg.search_batch.min(root_visits / cfg.search_batch_ramp));
            want = want.min(st.cap - st.done);
            if max_leaves > 0 {
                want = want.min(max_leaves);
            }
            want = want.max(1);
            for _ in 0..want {
                self.collect();
            }
            self.search.as_mut().unwrap().done += want;
        }
        Ok(&self.queue)
    }

    /// Number of pending requests (after `leaf_requests`).
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    /// `_collect`: one descent laying virtual loss; ends at a terminal or a leaf.
    fn collect(&mut self) {
        let vl = self.config.virtual_loss;
        let mut node = self.root;
        self.path_scratch.clear();
        loop {
            let n = self.nodes[node as usize];
            if n.state.is_terminal {
                let path = self.push_path();
                self.backup(path, n.terminal_v0, n.terminal_m0);
                return;
            }
            if !n.expanded {
                let path = self.push_path();
                let req = if n.pending == NONE {
                    self.queue.push(LeafRequest { node, is_root: false, paths: Vec::new() });
                    let r = (self.queue.len() - 1) as u32;
                    self.nodes[node as usize].pending = r;
                    r
                } else {
                    n.pending
                };
                self.queue[req as usize].paths.push(path);
                return;
            }
            let index = self.select(node);
            {
                let p = &mut self.nodes[node as usize];
                let i = p.edge_start as usize + index;
                self.visits[i] += 1;
                p.n_visits += 1;
                self.wins[i] -= vl;
            }
            self.path_scratch.push((node, index as u32));
            node = self.child(node, index);
        }
    }

    fn push_path(&mut self) -> (u32, u32) {
        let start = self.path_pool.len() as u32;
        self.path_pool.extend_from_slice(&self.path_scratch);
        (start, self.path_scratch.len() as u32)
    }

    /// `_backup`: undo the virtual loss along the path and credit the result.
    fn backup(&mut self, path: (u32, u32), v0: f64, m0: f64) {
        let vl = self.config.virtual_loss;
        let has_margin = self.has_margin;
        let (start, len) = (path.0 as usize, path.1 as usize);
        for k in start..start + len {
            let (parent, index) = self.path_pool[k];
            let p = &self.nodes[parent as usize];
            let i = p.edge_start as usize + index as usize;
            let flip = if p.player == 0 { 1.0 } else { -1.0 };
            self.wins[i] += vl + v0 * flip;
            if has_margin {
                self.margins[i] += m0 * flip;
            }
        }
    }

    /// `apply_leaves`: one `(priors, value, margin)` per pending request, in order.
    pub fn apply_leaves(&mut self, results: &[(&[f32], f64, f64)]) -> Result<(), String> {
        if self.search.is_none() {
            return Err("no batched search in progress".into());
        }
        if results.len() != self.queue.len() {
            return Err(format!("expected {} evaluations, got {}", self.queue.len(), results.len()));
        }
        let queue = std::mem::take(&mut self.queue);
        for (request, &(priors, value, margin)) in queue.iter().zip(results) {
            let node = request.node;
            let (s, e) = self.edges(node);
            if priors.len() != e - s {
                self.queue = queue;
                return Err(format!("leaf priors have {} entries, legal list has {}", priors.len(), e - s));
            }
            self.init_edges(node, priors);
            self.evals += 1;
            self.nodes[node as usize].pending = NONE;
            if request.is_root {
                let st = self.search.as_mut().unwrap();
                st.root_value = value;
                st.root_margin = margin;
                continue;
            }
            let flip = if self.nodes[node as usize].player == 0 { 1.0 } else { -1.0 };
            let v0 = value * flip;
            let m0 = margin * flip;
            for &path in &request.paths {
                self.backup(path, v0, m0);
            }
        }
        self.queue = queue;
        self.queue.clear();
        self.path_pool.clear();
        let need_root = self.search.as_ref().unwrap().need_root;
        if need_root {
            self.search.as_mut().unwrap().need_root = false;
            self.open_root();
        }
        Ok(())
    }

    /// `finish_search`: close the open search and report it.
    pub fn finish_search(&mut self) -> Result<SearchResult, String> {
        let Some(st) = self.search.take() else {
            return Err("no batched search in progress".into());
        };
        let root = self.root;
        if st.forced {
            return Ok(self.forced_result(root, st.root_value, st.root_margin));
        }
        Ok(self.result(root, st.root_value, st.root_margin, st.started.elapsed().as_secs_f64()))
    }

    pub fn queue(&self) -> &[LeafRequest] {
        &self.queue
    }
}

// ------------------------------------------------------------- distributions
/// Marsaglia–Tsang gamma variate with shape `alpha` (any positive alpha).
fn gamma(rng: &mut SplitMix64, alpha: f64) -> f64 {
    if alpha < 1.0 {
        // gamma(a) = gamma(a + 1) * U^(1/a)
        let u = rng.random().max(1e-300);
        return gamma(rng, alpha + 1.0) * u.powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = rng.random();
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Standard normal by Marsaglia's polar method.
fn normal(rng: &mut SplitMix64) -> f64 {
    loop {
        let u = 2.0 * rng.random() - 1.0;
        let v = 2.0 * rng.random() - 1.0;
        let s = u * u + v * v;
        if s > 0.0 && s < 1.0 {
            return u * (-2.0 * s.ln() / s).sqrt();
        }
    }
}

// ------------------------------------------------------------- move selection
/// numpy's `pairwise_sum` for a contiguous float64 array (what `.sum()` does),
/// reproduced so a sampled move lands on the same index as the Python driver's.
pub fn numpy_sum(a: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &x in a {
            res += x;
        }
        return res;
    }
    if n <= BLOCK {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i + 8 <= n {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let n2 = (n / 2) - (n / 2) % 8;
    numpy_sum(&a[..n2]) + numpy_sum(&a[n2..])
}

/// `select_action(policy, temperature, rng)`.
pub fn select_action(policy: &[f32; ACTION_SPACE], temperature: f64, rng: &mut TreeRng) -> usize {
    if temperature <= 0.0 {
        return argmax_f32(policy);
    }
    let mut probs: Vec<f64> = policy.iter().map(|&p| p as f64).collect();
    if temperature != 1.0 {
        let inv = 1.0 / temperature;
        for p in probs.iter_mut() {
            *p = p.powf(inv);
        }
    }
    let total = numpy_sum(&probs);
    if !total.is_finite() || total <= 0.0 {
        return argmax_f32(policy);
    }
    for p in probs.iter_mut() {
        *p /= total;
    }
    let draw = rng.random() * 1.0;
    let mut acc = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if draw <= acc {
            return i;
        }
    }
    // float rounding: np.argmax(probs)
    let mut best = 0;
    for (i, &p) in probs.iter().enumerate() {
        if p > probs[best] {
            best = i;
        }
    }
    best
}

/// `decisive_action(result, eps, min_visit_frac)`.
pub fn decisive_action(result: &SearchResult, eps: f64, min_visit_frac: f64) -> usize {
    if !result.has_margin || result.q.is_empty() {
        return result.argmax_policy();
    }
    let best_visits = result.visits.iter().map(|&(_, n)| n).max().unwrap_or(0);
    if best_visits == 0 {
        return result.argmax_policy();
    }
    let floor = min_visit_frac * best_visits as f64;
    let q_of = |a: u8| result.q.iter().find(|(x, _)| *x == a).map(|&(_, q)| q);
    let m_of = |a: u8| result.margins.iter().find(|(x, _)| *x == a).map(|&(_, m)| m).unwrap_or(0.0);
    let mut candidates: Vec<(u8, u32, f64)> = Vec::new();
    for &(a, n) in &result.visits {
        if (n as f64) >= floor {
            if let Some(q) = q_of(a) {
                candidates.push((a, n, q));
            }
        }
    }
    if candidates.is_empty() {
        return result.argmax_policy();
    }
    let best_q = candidates.iter().map(|c| c.2).fold(f64::NEG_INFINITY, f64::max);
    // max(keep, key=(margin, visits, -a)): strictly greater key wins, first kept otherwise.
    let mut best: Option<(f64, u32, i64, u8)> = None;
    for &(a, n, q) in &candidates {
        if q >= best_q - eps {
            let key = (m_of(a), n, -(a as i64), a);
            match best {
                None => best = Some(key),
                Some(b) => {
                    if (key.0, key.1, key.2) > (b.0, b.1, b.2) {
                        best = Some(key);
                    }
                }
            }
        }
    }
    best.map(|b| b.3 as usize).unwrap_or_else(|| result.argmax_policy())
}

/// `select_play_action(result, temperature, rng, eps, min_visit_frac, stalling)`.
pub fn select_play_action(
    result: &SearchResult,
    temperature: f64,
    rng: &mut TreeRng,
    eps: f64,
    min_visit_frac: f64,
    stalling: bool,
) -> usize {
    if stalling {
        return select_action(&result.policy, temperature.max(1.0), rng);
    }
    if temperature > 0.0 || !result.has_margin {
        return select_action(&result.policy, temperature, rng);
    }
    decisive_action(result, eps, min_visit_frac)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(_s: &State, legal: &[u8]) -> (Vec<f32>, f64, f64) {
        let n = legal.len();
        (vec![1.0 / n as f32; n], 0.0, 0.0)
    }

    fn score_eval(s: &State, legal: &[u8]) -> (Vec<f32>, f64, f64) {
        let n = legal.len();
        let me = s.current_player as usize;
        let margin = (s.scores[me] - s.scores[1 - me]) as f64
            + 2.0 * (s.completed_rows(me) as f64 - s.completed_rows(1 - me) as f64);
        (vec![1.0 / n as f32; n], (margin / 15.0).tanh(), 0.0)
    }

    #[test]
    fn search_returns_a_distribution_over_legal_actions() {
        let state = State::new_game(1, RngKind::Fast);
        let mut tree = Tree::new(MctsConfig { sims: 48, ..Default::default() }, false, 2, false, RngKind::Fast);
        let r = tree.search(&state, uniform, None, None, None).unwrap();
        let legal = state.legal_actions();
        let sum: f32 = r.policy.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        for (a, &p) in r.policy.iter().enumerate() {
            if !legal.contains(&(a as u8)) {
                assert_eq!(p, 0.0);
            }
        }
        assert_eq!(r.sims, 48);
        assert_eq!(r.visits.iter().map(|&(_, n)| n).sum::<u32>(), 48);
        assert_eq!(tree.evals, 49);
    }

    #[test]
    fn pumped_search_is_identical_to_the_blocking_one() {
        for reuse in [false, true] {
            for batch in [1u32, 4] {
                let config = MctsConfig {
                    sims: 160,
                    tree_reuse: reuse,
                    search_batch: batch,
                    search_batch_ramp: 4,
                    ..Default::default()
                };
                let mut plain = State::new_game(5, RngKind::Fast);
                let mut pumped = State::new_game(5, RngKind::Fast);
                let mut a = Tree::new(config, false, 17, false, RngKind::Fast);
                let mut b = Tree::new(config, false, 17, false, RngKind::Fast);
                for _ in 0..6 {
                    let want = a.search(&plain, score_eval, None, None, None).unwrap();
                    b.start_search(&pumped, None, None).unwrap();
                    while !b.search_done() {
                        let n = b.leaf_requests(0).unwrap().len();
                        let mut evals = Vec::with_capacity(n);
                        for k in 0..n {
                            let node = b.queue()[k].node;
                            let st = *b.node_state(node);
                            let legal = b.node_legal(node).to_vec();
                            evals.push(score_eval(&st, &legal));
                        }
                        let refs: Vec<(&[f32], f64, f64)> = evals.iter().map(|(p, v, m)| (p.as_slice(), *v, *m)).collect();
                        b.apply_leaves(&refs).unwrap();
                    }
                    let got = b.finish_search().unwrap();
                    if batch == 1 {
                        assert_eq!(got.visits, want.visits, "reuse {reuse}");
                        assert_eq!(got.policy, want.policy);
                        assert!((got.value - want.value).abs() < 1e-12);
                    }
                    assert_eq!(got.sims, want.sims);
                    assert_eq!(got.visits.iter().map(|&(_, n)| n).sum::<u32>(), 160);
                    let action = want.argmax_policy() as u8;
                    plain.apply(action).unwrap();
                    pumped.apply(action).unwrap();
                    a.advance(action);
                    b.advance(action);
                }
                if batch == 1 {
                    assert_eq!(a.evals, b.evals);
                    assert_eq!(a.nodes_created, b.nodes_created);
                }
            }
        }
    }

    #[test]
    fn tree_reuse_keeps_visits_and_compacts() {
        let config = MctsConfig { sims: 200, tree_reuse: true, ..Default::default() };
        let mut state = State::new_game(3, RngKind::Fast);
        let mut tree = Tree::new(config, false, 1, false, RngKind::Fast);
        let mut reused_any = false;
        for _ in 0..12 {
            let r = tree.search(&state, score_eval, None, None, None).unwrap();
            assert_eq!(r.sims, 200);
            if tree.reused_visits > 0 {
                reused_any = true;
            }
            assert!(tree.node_count() <= 2 * 201 + 10, "arena grew to {}", tree.node_count());
            let action = r.argmax_policy() as u8;
            state.apply(action).unwrap();
            tree.advance(action);
        }
        assert!(reused_any);
    }

    #[test]
    fn time_budget_stops_early() {
        let state = State::new_game(1, RngKind::Fast);
        let mut tree = Tree::new(MctsConfig { sims: 1_000_000, ..Default::default() }, false, 2, false, RngKind::Fast);
        let r = tree.search(&state, uniform, None, Some(0.02), None).unwrap();
        assert!(r.sims > 8 && r.sims < 1_000_000);
    }

    #[test]
    fn numpy_sum_matches_reference_values() {
        // Values checked against numpy in tests/test_rust_mcts.py; here: structure.
        let a: Vec<f64> = (0..180).map(|i| 1.0 / (i as f64 + 1.0)).collect();
        let s = numpy_sum(&a);
        let seq: f64 = a.iter().sum();
        assert!((s - seq).abs() < 1e-12);
    }

    #[test]
    fn noise_changes_the_root_priors_and_sums_to_one() {
        let state = State::new_game(4, RngKind::Fast);
        let mut a = Tree::new(MctsConfig { sims: 30, ..Default::default() }, false, 7, true, RngKind::Fast);
        let mut b = Tree::new(MctsConfig { sims: 30, ..Default::default() }, false, 7, false, RngKind::Fast);
        let ra = a.search(&state, uniform, None, None, None).unwrap();
        let rb = b.search(&state, uniform, None, None, None).unwrap();
        assert_ne!(ra.policy, rb.policy);
        let (s, e) = a.edges(a.root);
        let total: f64 = a.priors[s..e].iter().sum();
        assert!((total - 1.0).abs() < 1e-6); // f32 uniform priors, like the Python test
    }
}
