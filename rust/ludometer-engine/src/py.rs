//! PyO3 bindings: the `ludometer_rs` module.
//!
//! `State` mirrors `ludometer.azul.engine.AzulState` (same method names, numpy
//! `encode()`), `Tree` mirrors `ludometer.train.mcts.MCTS` through the leaf
//! protocol, `Arena` runs many games for `ludometer.train.selfplay_rust`.

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

use crate::azul::{self, State as RsState, ACTION_SPACE, ENCODED_SIZE, NUM_COLORS, NUM_FACTORIES};
use crate::arena::{Arena as RsArena, GameRecord, SelfPlayConfig};
use crate::mcts::{self, MctsConfig, SearchResult, Tree as RsTree};
use crate::rng::RngKind;

fn rng_kind(name: &str) -> PyResult<RngKind> {
    match name {
        "fast" => Ok(RngKind::Fast),
        "python" => Ok(RngKind::Python),
        other => Err(PyValueError::new_err(format!("unknown rng {other:?} (fast | python)"))),
    }
}

fn rng_name(kind: RngKind) -> &'static str {
    match kind {
        RngKind::Fast => "fast",
        RngKind::Python => "python",
    }
}

fn to_u8_action(action: i64) -> PyResult<u8> {
    if !(0..ACTION_SPACE as i64).contains(&action) {
        return Err(PyValueError::new_err(format!("action {action} out of range")));
    }
    Ok(action as u8)
}

/// One Azul position (see `ludometer.azul.engine.AzulState`).
#[pyclass(name = "State", module = "ludometer_rs", skip_from_py_object)]
#[derive(Clone)]
pub struct PyState {
    pub inner: RsState,
}

#[pymethods]
impl PyState {
    #[classattr]
    const ACTION_SPACE: usize = ACTION_SPACE;
    #[classattr]
    const ENCODED_SIZE: usize = ENCODED_SIZE;

    /// `State.new_game(seed, rng="fast")`; `rng="python"` deals exactly like the Python engine.
    #[staticmethod]
    #[pyo3(signature = (seed, rng = "fast"))]
    fn new_game(seed: i64, rng: &str) -> PyResult<Self> {
        if seed < 0 {
            return Err(PyValueError::new_err("seed must be >= 0"));
        }
        Ok(PyState { inner: RsState::new_game(seed as u64, rng_kind(rng)?) })
    }

    fn clone(&self) -> Self {
        Clone::clone(self)
    }

    fn __copy__(&self) -> Self {
        Clone::clone(self)
    }

    #[getter]
    fn rng_kind(&self) -> &'static str {
        rng_name(self.inner.rng.kind)
    }

    fn legal_actions(&self) -> Vec<i64> {
        self.inner.legal_actions().iter().map(|&a| a as i64).collect()
    }

    fn is_legal(&self, action: i64) -> bool {
        self.inner.is_legal(action)
    }

    fn apply(&mut self, action: i64) -> PyResult<()> {
        let a = to_u8_action(action)?;
        self.inner.apply(a).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn is_stochastic(&self, action: i64) -> PyResult<bool> {
        Ok(self.inner.is_stochastic(to_u8_action(action)?))
    }

    fn determinize(&self, action: i64, seed: i64) -> PyResult<Self> {
        let a = to_u8_action(action)?;
        if !self.inner.is_legal(action) {
            return Err(PyValueError::new_err(format!("illegal action {action}")));
        }
        Ok(PyState { inner: self.inner.determinize(a, seed as u64) })
    }

    fn chance_key<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.chance_key())
    }

    /// The same tuple `AzulState.fingerprint()` returns.
    fn fingerprint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let s = &self.inner;
        let pl_color = |p: usize| -> Vec<i64> { s.pl_color[p].iter().map(|&c| c as i64).collect() };
        let pl_count = |p: usize| -> Vec<i64> { s.pl_count[p].iter().map(|&c| c as i64).collect() };
        let items: Vec<Py<PyAny>> = vec![
            (s.current_player as i64).into_pyobject(py)?.into_any().unbind(),
            (s.round_index as i64).into_pyobject(py)?.into_any().unbind(),
            (s.tiles_left as i64).into_pyobject(py)?.into_any().unbind(),
            s.marker_in_center.into_pyobject(py)?.to_owned().into_any().unbind(),
            PyTuple::new(py, [s.scores[0] as i64, s.scores[1] as i64])?.into_any().unbind(),
            PyTuple::new(py, pl_count(0))?.into_any().unbind(),
            PyTuple::new(py, pl_count(1))?.into_any().unbind(),
            PyTuple::new(py, pl_color(0))?.into_any().unbind(),
            PyTuple::new(py, pl_color(1))?.into_any().unbind(),
            (s.walls[0].count_ones() as i64).into_pyobject(py)?.into_any().unbind(),
            (s.walls[1].count_ones() as i64).into_pyobject(py)?.into_any().unbind(),
            PyBytes::new(py, &s.chance_key()).into_any().unbind(),
        ];
        Ok(PyTuple::new(py, items)?.into_any())
    }

    fn search_root(&self) -> Self {
        Clone::clone(self)
    }

    /// Float32 observation of length 182 (a fresh numpy array).
    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let v = self.inner.encoded();
        PyArray1::from_slice(py, &v)
    }

    fn outcome(&self) -> Option<f32> {
        self.inner.outcome()
    }

    fn wall_summary(&self, player: usize) -> Vec<i64> {
        self.inner.wall_summary(player).iter().map(|&x| x as i64).collect()
    }

    fn tile_census(&self) -> Vec<i64> {
        self.inner.tile_census().iter().map(|&x| x as i64).collect()
    }

    fn completed_rows(&self, player: usize) -> u32 {
        self.inner.completed_rows(player)
    }
    fn completed_cols(&self, player: usize) -> u32 {
        self.inner.completed_cols(player)
    }
    fn completed_colors(&self, player: usize) -> u32 {
        self.inner.completed_colors(player)
    }
    fn floor_occupied(&self, player: usize) -> usize {
        self.inner.floor_occupied(player)
    }
    fn floor_penalty(&self, player: usize) -> i32 {
        self.inner.floor_penalty(player)
    }
    fn recount(&mut self) {
        self.inner.recount()
    }

    /// The BGA hook: replace the refill just made with `factories` (5 x 5 counts).
    fn apply_deal(&mut self, factories: Vec<Vec<i64>>) -> PyResult<()> {
        if factories.len() != NUM_FACTORIES {
            return Err(PyValueError::new_err(format!(
                "deal has {} factories, engine has {NUM_FACTORIES}",
                factories.len()
            )));
        }
        let mut target = [[0u8; NUM_COLORS]; NUM_FACTORIES];
        for (f, row) in factories.iter().enumerate() {
            if row.len() != NUM_COLORS {
                return Err(PyValueError::new_err(format!("deal factory has {} colors", row.len())));
            }
            for (c, &n) in row.iter().enumerate() {
                if !(0..=100).contains(&n) {
                    return Err(PyValueError::new_err("deal has a bad tile count"));
                }
                target[f][c] = n as u8;
            }
        }
        self.inner.apply_deal(&target).map_err(PyValueError::new_err)
    }

    // ----------------------------------------------------------- attributes
    #[getter]
    fn current_player(&self) -> u8 {
        self.inner.current_player
    }
    #[setter]
    fn set_current_player(&mut self, p: u8) {
        self.inner.current_player = p;
    }
    #[getter]
    fn first_player(&self) -> u8 {
        self.inner.first_player
    }
    #[setter]
    fn set_first_player(&mut self, p: u8) {
        self.inner.first_player = p;
    }
    #[getter]
    fn round_index(&self) -> u16 {
        self.inner.round_index
    }
    #[getter]
    fn tiles_left(&self) -> u8 {
        self.inner.tiles_left
    }
    #[getter]
    fn is_terminal(&self) -> bool {
        self.inner.is_terminal
    }
    #[getter]
    fn exhausted(&self) -> bool {
        self.inner.exhausted
    }
    #[getter]
    fn marker_in_center(&self) -> bool {
        self.inner.marker_in_center
    }
    #[getter]
    fn num_players(&self) -> usize {
        2
    }
    #[getter]
    fn scores(&self) -> Vec<i64> {
        self.inner.scores.iter().map(|&x| x as i64).collect()
    }
    #[getter]
    fn factories(&self) -> Vec<Vec<i64>> {
        self.inner.factories.iter().map(|f| f.iter().map(|&x| x as i64).collect()).collect()
    }
    #[getter]
    fn center(&self) -> Vec<i64> {
        self.inner.center.iter().map(|&x| x as i64).collect()
    }
    #[getter]
    fn lid(&self) -> Vec<i64> {
        self.inner.lid.iter().map(|&x| x as i64).collect()
    }
    /// The bag in draw order (last element is drawn first), like `AzulState.bag`.
    #[getter]
    fn bag(&self) -> Vec<i64> {
        self.inner.bag_slice().iter().map(|&x| x as i64).collect()
    }
    fn bag_counts(&self) -> Vec<i64> {
        self.inner.bag_counts().iter().map(|&x| x as i64).collect()
    }
    #[getter]
    fn walls(&self) -> Vec<Vec<i64>> {
        (0..2).map(|p| self.inner.wall_cells(p).iter().map(|&x| x as i64).collect()).collect()
    }
    #[getter]
    fn pl_color(&self) -> Vec<Vec<i64>> {
        self.inner.pl_color.iter().map(|x| x.iter().map(|&c| c as i64).collect()).collect()
    }
    #[getter]
    fn pl_count(&self) -> Vec<Vec<i64>> {
        self.inner.pl_count.iter().map(|x| x.iter().map(|&c| c as i64).collect()).collect()
    }
    #[getter]
    fn floor(&self) -> Vec<Vec<i64>> {
        self.inner.floor.iter().map(|x| x.iter().map(|&c| c as i64).collect()).collect()
    }
    #[getter]
    fn floor_marker(&self) -> Vec<bool> {
        self.inner.floor_marker.to_vec()
    }

    /// Everything as plain Python data (the `AzulState` attribute names).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("factories", self.factories())?;
        d.set_item("center", self.center())?;
        d.set_item("marker_in_center", self.inner.marker_in_center)?;
        d.set_item("bag", self.bag())?;
        d.set_item("lid", self.lid())?;
        d.set_item("walls", self.walls())?;
        d.set_item("pl_color", self.pl_color())?;
        d.set_item("pl_count", self.pl_count())?;
        d.set_item("floor", self.floor())?;
        d.set_item("floor_marker", self.floor_marker())?;
        d.set_item("scores", self.scores())?;
        d.set_item("current_player", self.inner.current_player)?;
        d.set_item("first_player", self.inner.first_player)?;
        d.set_item("round_index", self.inner.round_index)?;
        d.set_item("is_terminal", self.inner.is_terminal)?;
        d.set_item("exhausted", self.inner.exhausted)?;
        d.set_item("tiles_left", self.inner.tiles_left)?;
        Ok(d)
    }

    /// Build a state from `to_dict()`-shaped data (or a Python `AzulState`'s
    /// attributes). The RNG stream is *not* imported: refills made by the real
    /// game after this point follow the Rust generator seeded with `rng_seed`.
    #[staticmethod]
    #[pyo3(signature = (data, rng = "fast", rng_seed = 0))]
    fn from_dict(data: &Bound<'_, PyDict>, rng: &str, rng_seed: i64) -> PyResult<Self> {
        let mut s = RsState::blank(rng_kind(rng)?);
        s.rng = crate::rng::Rng::new(rng_kind(rng)?, rng_seed as u64);
        let get = |k: &str| -> PyResult<Bound<'_, PyAny>> {
            data.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("missing key {k:?}")))
        };
        let factories: Vec<Vec<u8>> = get("factories")?.extract()?;
        for f in 0..NUM_FACTORIES {
            for c in 0..NUM_COLORS {
                s.factories[f][c] = factories[f][c];
            }
        }
        let center: Vec<u8> = get("center")?.extract()?;
        s.center.copy_from_slice(&center);
        s.marker_in_center = get("marker_in_center")?.extract()?;
        let bag: Vec<u8> = get("bag")?.extract()?;
        if bag.len() > 100 {
            return Err(PyValueError::new_err("bag longer than 100"));
        }
        s.set_bag(&bag);
        let lid: Vec<u8> = get("lid")?.extract()?;
        s.lid.copy_from_slice(&lid);
        let walls: Vec<Vec<u8>> = get("walls")?.extract()?;
        for p in 0..2 {
            let mut cells = [0u8; 25];
            cells.copy_from_slice(&walls[p]);
            s.set_wall_cells(p, &cells);
        }
        let pl_color: Vec<Vec<i8>> = get("pl_color")?.extract()?;
        let pl_count: Vec<Vec<u8>> = get("pl_count")?.extract()?;
        let floor: Vec<Vec<u8>> = get("floor")?.extract()?;
        let floor_marker: Vec<bool> = get("floor_marker")?.extract()?;
        for p in 0..2 {
            s.pl_color[p].copy_from_slice(&pl_color[p]);
            s.pl_count[p].copy_from_slice(&pl_count[p]);
            s.floor[p].copy_from_slice(&floor[p]);
            s.floor_marker[p] = floor_marker[p];
        }
        let scores: Vec<i32> = get("scores")?.extract()?;
        s.scores.copy_from_slice(&scores);
        s.current_player = get("current_player")?.extract()?;
        s.first_player = get("first_player")?.extract()?;
        s.round_index = get("round_index")?.extract()?;
        s.is_terminal = get("is_terminal")?.extract()?;
        s.exhausted = get("exhausted")?.extract()?;
        s.recount();
        Ok(PyState { inner: s })
    }

    fn __repr__(&self) -> String {
        format!(
            "<ludometer_rs.State round={} player={} scores={:?} terminal={}>",
            self.inner.round_index, self.inner.current_player, self.inner.scores, self.inner.is_terminal
        )
    }
}


// ------------------------------------------------------------------- Tree
/// `MCTSConfig.from_dict` for the Rust config.
pub fn config_from_dict(data: Option<&Bound<'_, PyDict>>) -> PyResult<MctsConfig> {
    let mut cfg = MctsConfig::default();
    if let Some(d) = data {
        macro_rules! get {
            ($key:literal, $field:ident, $ty:ty) => {
                if let Some(v) = d.get_item($key)? {
                    cfg.$field = v.extract::<$ty>()?;
                }
            };
        }
        get!("sims", sims, u32);
        get!("c_puct", c_puct, f64);
        get!("dirichlet_alpha_scale", dirichlet_alpha_scale, f64);
        get!("dirichlet_eps", dirichlet_eps, f64);
        get!("chance_children", chance_children, usize);
        get!("fpu", fpu, f64);
        get!("tree_reuse", tree_reuse, bool);
        get!("decisive_eps", decisive_eps, f64);
        get!("decisive_min_visit_frac", decisive_min_visit_frac, f64);
        get!("search_batch", search_batch, u32);
        get!("search_batch_ramp", search_batch_ramp, u32);
        get!("search_min_batch", search_min_batch, u32);
        get!("virtual_loss", virtual_loss, f64);
        if let Some(v) = d.get_item("chance_backup")? {
            let name: String = v.extract()?;
            if name != "mean" {
                return Err(PyValueError::new_err(format!(
                    "unknown chance_backup {name:?} (only 'mean' exists)"
                )));
            }
        }
    }
    cfg.validate().map_err(PyValueError::new_err)?;
    Ok(cfg)
}

pub fn config_to_dict<'py>(py: Python<'py>, cfg: &MctsConfig) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("sims", cfg.sims)?;
    d.set_item("c_puct", cfg.c_puct)?;
    d.set_item("dirichlet_alpha_scale", cfg.dirichlet_alpha_scale)?;
    d.set_item("dirichlet_eps", cfg.dirichlet_eps)?;
    d.set_item("chance_children", cfg.chance_children)?;
    d.set_item("chance_backup", "mean")?;
    d.set_item("fpu", cfg.fpu)?;
    d.set_item("tree_reuse", cfg.tree_reuse)?;
    d.set_item("decisive_eps", cfg.decisive_eps)?;
    d.set_item("decisive_min_visit_frac", cfg.decisive_min_visit_frac)?;
    d.set_item("search_batch", cfg.search_batch)?;
    d.set_item("search_batch_ramp", cfg.search_batch_ramp)?;
    d.set_item("search_min_batch", cfg.search_min_batch)?;
    d.set_item("virtual_loss", cfg.virtual_loss)?;
    Ok(d)
}

/// A `SearchResult` as the dict `ludometer.train.mcts_rs` turns into the dataclass.
pub fn result_to_dict<'py>(py: Python<'py>, r: &SearchResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("policy", PyArray1::from_slice(py, &r.policy))?;
    d.set_item("value", r.value)?;
    let visits = PyDict::new(py);
    for &(a, n) in &r.visits {
        visits.set_item(a as i64, n)?;
    }
    d.set_item("visits", visits)?;
    d.set_item("sims", r.sims)?;
    d.set_item("elapsed_s", r.elapsed_s)?;
    d.set_item("has_margin", r.has_margin)?;
    let q = PyDict::new(py);
    for &(a, v) in &r.q {
        q.set_item(a as i64, v)?;
    }
    d.set_item("q", q)?;
    let m = PyDict::new(py);
    for &(a, v) in &r.margins {
        m.set_item(a as i64, v)?;
    }
    d.set_item("margins", m)?;
    d.set_item("margin", r.margin)?;
    Ok(d)
}

/// Unpack an evaluator's `(priors, value)` or `(priors, value, margin)`.
fn unpack_eval(out: &Bound<'_, PyAny>) -> PyResult<(Vec<f32>, f64, f64)> {
    let tuple = out.cast::<PyTuple>().map_err(|_| PyValueError::new_err("evaluator must return a tuple"))?;
    let n = tuple.len();
    if n != 2 && n != 3 {
        return Err(PyValueError::new_err("evaluator must return (priors, value[, margin])"));
    }
    let priors_obj = tuple.get_item(0)?;
    let priors: Vec<f32> = match priors_obj.extract::<PyReadonlyArray1<f32>>() {
        Ok(arr) => {
            let sl: &[f32] = arr.as_slice()?;
            sl.to_vec()
        }
        Err(_) => priors_obj.extract::<Vec<f64>>()?.into_iter().map(|x| x as f32).collect(),
    };
    let value: f64 = tuple.get_item(1)?.extract()?;
    let margin: f64 = if n == 3 { tuple.get_item(2)?.extract()? } else { 0.0 };
    Ok((priors, value, margin))
}

/// A PUCT search tree (see `ludometer.train.mcts.MCTS`).
#[pyclass(name = "Tree", module = "ludometer_rs", unsendable)]
pub struct PyTree {
    pub inner: RsTree,
}

#[pymethods]
impl PyTree {
    #[new]
    #[pyo3(signature = (config = None, has_margin = false, seed = 0, add_noise = false, rng = "fast"))]
    fn new(config: Option<&Bound<'_, PyDict>>, has_margin: bool, seed: i64, add_noise: bool, rng: &str) -> PyResult<Self> {
        let cfg = config_from_dict(config)?;
        Ok(PyTree { inner: RsTree::new(cfg, has_margin, seed as u64, add_noise, rng_kind(rng)?) })
    }

    fn seed(&mut self, n: i64) {
        self.inner.seed(n as u64);
    }

    fn reset_tree(&mut self) {
        self.inner.reset_tree();
    }

    fn advance(&mut self, action: i64) -> PyResult<bool> {
        Ok(self.inner.advance(to_u8_action(action)?))
    }

    #[getter]
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        config_to_dict(py, &self.inner.config)
    }

    #[setter]
    fn set_config(&mut self, config: &Bound<'_, PyDict>) -> PyResult<()> {
        self.inner.config = config_from_dict(Some(config))?;
        Ok(())
    }

    #[getter]
    fn has_margin(&self) -> bool {
        self.inner.has_margin
    }
    #[getter]
    fn add_noise(&self) -> bool {
        self.inner.add_noise
    }
    #[setter]
    fn set_add_noise(&mut self, v: bool) {
        self.inner.add_noise = v;
    }
    #[getter]
    fn evals(&self) -> u64 {
        self.inner.evals
    }
    #[setter]
    fn set_evals(&mut self, v: u64) {
        self.inner.evals = v;
    }
    #[getter]
    fn nodes_created(&self) -> u64 {
        self.inner.nodes_created
    }
    #[getter]
    fn reused_visits(&self) -> u32 {
        self.inner.reused_visits
    }
    #[getter]
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }
    #[getter]
    fn rng_kind(&self) -> &'static str {
        rng_name(self.inner.rng_kind)
    }

    /// The search's own RNG: `random.random()` / `random.randrange(n)`.
    fn rng_random(&mut self) -> f64 {
        self.inner.rng.random()
    }
    fn rng_randrange(&mut self, n: i64) -> PyResult<i64> {
        if n < 1 {
            return Err(PyValueError::new_err("empty range for randrange()"));
        }
        Ok(self.inner.rng.randrange(n as u64) as i64)
    }

    /// `select_action(policy, temperature, self.rng)` from `ludometer.train.mcts`.
    fn select_action(&mut self, policy: PyReadonlyArray1<f32>, temperature: f64) -> PyResult<i64> {
        let sl = policy.as_slice()?;
        if sl.len() != ACTION_SPACE {
            return Err(PyValueError::new_err("policy must have 180 entries"));
        }
        let mut arr = [0.0f32; ACTION_SPACE];
        arr.copy_from_slice(sl);
        Ok(mcts::select_action(&arr, temperature, &mut self.inner.rng) as i64)
    }

    /// The blocking search: `evaluator(state, legal) -> (priors, value[, margin])`.
    #[pyo3(signature = (state, evaluator, add_noise = None, time_limit_s = None, sims = None))]
    fn search<'py>(
        &mut self,
        py: Python<'py>,
        state: &PyState,
        evaluator: &Bound<'py, PyAny>,
        add_noise: Option<bool>,
        time_limit_s: Option<f64>,
        sims: Option<u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut err: Option<PyErr> = None;
        let result = {
            let eval = |st: &RsState, legal: &[u8]| -> (Vec<f32>, f64, f64) {
                if err.is_some() {
                    return (vec![0.0; legal.len()], 0.0, 0.0);
                }
                let legal_py: Vec<i64> = legal.iter().map(|&a| a as i64).collect();
                let call = || -> PyResult<(Vec<f32>, f64, f64)> {
                    let ps = Py::new(py, PyState { inner: *st })?;
                    let out = evaluator.call1((ps, legal_py))?;
                    let (priors, v, m) = unpack_eval(&out)?;
                    if priors.len() != legal.len() {
                        return Err(PyValueError::new_err(format!(
                            "evaluator returned {} priors for {} legal actions",
                            priors.len(),
                            legal.len()
                        )));
                    }
                    Ok((priors, v, m))
                };
                match call() {
                    Ok(x) => x,
                    Err(e) => {
                        err = Some(e);
                        (vec![0.0; legal.len()], 0.0, 0.0)
                    }
                }
            };
            self.inner.search(&state.inner, eval, add_noise, time_limit_s, sims)
        };
        if let Some(e) = err {
            return Err(e);
        }
        let r = result.map_err(PyValueError::new_err)?;
        result_to_dict(py, &r)
    }

    // --------------------------------------------------------- leaf protocol
    #[pyo3(signature = (state, add_noise = None, sims = None))]
    fn start_search(&mut self, state: &PyState, add_noise: Option<bool>, sims: Option<u32>) -> PyResult<()> {
        self.inner.start_search(&state.inner, add_noise, sims).map_err(PyRuntimeError::new_err)
    }

    fn search_done(&self) -> bool {
        self.inner.search_done()
    }

    /// Gather leaves: `(obs [n, 182] float32, legal: list[list[int]])`.
    #[pyo3(signature = (max_leaves = 0))]
    fn leaf_requests<'py>(&mut self, py: Python<'py>, max_leaves: u32) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyList>)> {
        let n = self.inner.leaf_requests(max_leaves).map_err(PyRuntimeError::new_err)?.len();
        let obs = PyArray2::<f32>::zeros(py, [n, ENCODED_SIZE], false);
        let legal = PyList::empty(py);
        {
            let view = unsafe { obs.as_slice_mut()? };
            for (k, req) in self.inner.queue().iter().enumerate() {
                let st = self.inner.node_state(req.node);
                let row: &mut [f32; ENCODED_SIZE] = (&mut view[k * ENCODED_SIZE..(k + 1) * ENCODED_SIZE]).try_into().unwrap();
                st.encode(row);
                let l: Vec<i64> = self.inner.node_legal(req.node).iter().map(|&a| a as i64).collect();
                legal.append(l)?;
            }
        }
        Ok((obs, legal))
    }

    /// The pending leaves as `State` objects (for evaluators that need them).
    fn leaf_states(&self) -> Vec<PyState> {
        self.inner.queue().iter().map(|r| PyState { inner: *self.inner.node_state(r.node) }).collect()
    }

    /// Evaluations for the pending leaves: one `priors` array per leaf (aligned
    /// with its legal list), plus values and optional margins.
    #[pyo3(signature = (priors, values, margins = None))]
    fn apply_leaves(&mut self, priors: Vec<PyReadonlyArray1<f32>>, values: Vec<f64>, margins: Option<Vec<f64>>) -> PyResult<()> {
        let n = priors.len();
        if values.len() != n || margins.as_ref().map(|m| m.len() != n).unwrap_or(false) {
            return Err(PyValueError::new_err("priors, values and margins must have one entry per leaf"));
        }
        let mut slices: Vec<&[f32]> = Vec::with_capacity(n);
        for p in &priors {
            slices.push(p.as_slice()?);
        }
        let results: Vec<(&[f32], f64, f64)> = (0..n)
            .map(|i| (slices[i], values[i], margins.as_ref().map(|m| m[i]).unwrap_or(0.0)))
            .collect();
        self.inner.apply_leaves(&results).map_err(PyValueError::new_err)
    }

    /// Raw net outputs for the pending leaves: `logits [n, 180]`, `values [n]`,
    /// `margins [n]` (or None); the softmax over each leaf's legal actions is
    /// done here, in float32 like the Python evaluator.
    #[pyo3(signature = (logits, values, margins = None))]
    fn apply_logits(&mut self, logits: PyReadonlyArray2<f32>, values: PyReadonlyArray1<f32>, margins: Option<PyReadonlyArray1<f32>>) -> PyResult<()> {
        let n = self.inner.queue().len();
        if logits.shape() != [n, ACTION_SPACE] || values.len() != n {
            return Err(PyValueError::new_err(format!("expected logits [{n}, 180] and values [{n}]")));
        }
        let lg = logits.as_slice()?;
        let vs = values.as_slice()?;
        let ms = match &margins {
            Some(m) => Some(m.as_slice()?),
            None => None,
        };
        let mut priors: Vec<Vec<f32>> = Vec::with_capacity(n);
        for (k, req) in self.inner.queue().iter().enumerate() {
            let legal = self.inner.node_legal(req.node);
            let row = &lg[k * ACTION_SPACE..(k + 1) * ACTION_SPACE];
            priors.push(softmax_over(row, legal));
        }
        let results: Vec<(&[f32], f64, f64)> = (0..n)
            .map(|i| (priors[i].as_slice(), vs[i] as f64, ms.map(|m| m[i] as f64).unwrap_or(0.0)))
            .collect();
        self.inner.apply_leaves(&results).map_err(PyValueError::new_err)
    }

    fn finish_search<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let r = self.inner.finish_search().map_err(PyRuntimeError::new_err)?;
        result_to_dict(py, &r)
    }

    fn root_state(&self) -> Option<PyState> {
        self.inner.root_state().map(|s| PyState { inner: *s })
    }
}


// ------------------------------------------------------------------ Arena
/// `SelfPlayConfig` from a flat dict: the MCTS keys plus `temp_moves`,
/// `temperature`, `stall_rounds`, `max_moves`, `value_score_weight`,
/// `pcr_full_sims`, `pcr_cheap_sims`, `pcr_full_prob` (all optional).
pub fn selfplay_config_from_dict(data: Option<&Bound<'_, PyDict>>) -> PyResult<SelfPlayConfig> {
    let mut cfg = SelfPlayConfig { mcts: config_from_dict(data)?, ..SelfPlayConfig::default() };
    if let Some(d) = data {
        macro_rules! get {
            ($key:literal, $field:ident, $ty:ty) => {
                if let Some(v) = d.get_item($key)? {
                    cfg.$field = v.extract::<$ty>()?;
                }
            };
        }
        get!("temp_moves", temp_moves, u32);
        get!("temperature", temperature, f64);
        get!("stall_rounds", stall_rounds, u32);
        get!("max_moves", max_moves, u32);
        get!("value_score_weight", value_score_weight, f64);
        get!("pcr_full_sims", pcr_full_sims, u32);
        get!("pcr_cheap_sims", pcr_cheap_sims, u32);
        get!("pcr_full_prob", pcr_full_prob, f64);
    }
    Ok(cfg)
}

fn record_to_dict<'py>(py: Python<'py>, r: GameRecord) -> PyResult<Bound<'py, PyDict>> {
    let t = r.len();
    let d = PyDict::new(py);
    d.set_item("states", PyArray1::from_vec(py, r.states).reshape([t, ENCODED_SIZE])?)?;
    d.set_item("policies", PyArray1::from_vec(py, r.policies).reshape([t, ACTION_SPACE])?)?;
    d.set_item("values", PyArray1::from_vec(py, r.values))?;
    d.set_item("margins", PyArray1::from_vec(py, r.margins))?;
    d.set_item("aux", PyArray1::from_vec(py, r.aux).reshape([t, 30])?)?;
    d.set_item("policy_mask", PyArray1::from_vec(py, r.policy_mask))?;
    d.set_item("outcome", r.outcome as f64)?;
    d.set_item("scores", (r.scores[0] as i64, r.scores[1] as i64))?;
    d.set_item("moves", r.moves)?;
    d.set_item("rounds", r.rounds)?;
    d.set_item("seed", r.seed)?;
    d.set_item("decisions", r.decisions)?;
    d.set_item("evals", r.evals)?;
    d.set_item("duration", r.duration)?;
    d.set_item("truncated", r.truncated)?;
    d.set_item("search_values", PyArray1::from_vec(py, r.search_values))?;
    d.set_item("search_mask", PyArray1::from_vec(py, r.search_mask))?;
    Ok(d)
}

/// Many concurrent self-play games; the caller evaluates the leaves
/// (see `ludometer.train.selfplay_rust`).
#[pyclass(name = "Arena", module = "ludometer_rs", unsendable)]
pub struct PyArena {
    inner: RsArena,
    leaf_cap: u32,
}

#[pymethods]
impl PyArena {
    #[new]
    #[pyo3(signature = (config = None, has_margin = false, games = 64, rng = "fast", leaf_cap = 0))]
    fn new(config: Option<&Bound<'_, PyDict>>, has_margin: bool, games: usize, rng: &str, leaf_cap: u32) -> PyResult<Self> {
        let cfg = selfplay_config_from_dict(config)?;
        let leaf_cap = if leaf_cap == 0 { cfg.mcts.search_batch.max(1) } else { leaf_cap };
        Ok(PyArena { inner: RsArena::new(cfg, has_margin, games, rng_kind(rng)?), leaf_cap })
    }

    fn begin(&mut self, n_games: u64, seed_start: u64) {
        self.inner.begin(n_games, seed_start);
    }

    fn set_stop(&mut self) {
        self.inner.set_stop();
    }

    fn finished(&self) -> bool {
        self.inner.finished()
    }

    #[getter]
    fn done(&self) -> u64 {
        self.inner.done()
    }
    #[getter]
    fn active(&self) -> usize {
        self.inner.active()
    }
    #[getter]
    fn positions(&self) -> u64 {
        self.inner.positions
    }
    #[getter]
    fn batches(&self) -> u64 {
        self.inner.batches
    }
    #[getter]
    fn games(&self) -> usize {
        self.inner.games
    }
    #[getter]
    fn has_margin(&self) -> bool {
        self.inner.has_margin
    }

    /// Seat, pump and collect: the pending leaves as `[n, 182]` float32 (n may be 0).
    fn gather<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let n = self.inner.gather(self.leaf_cap);
        let obs = PyArray2::<f32>::zeros(py, [n, ENCODED_SIZE], false);
        if n > 0 {
            let view = unsafe { obs.as_slice_mut()? };
            self.inner.observations(view);
        }
        Ok(obs)
    }

    /// Legal lists of the pending leaves (for an evaluator that softmaxes in Python).
    fn pending_legal(&self) -> Vec<Vec<i64>> {
        self.inner.pending_legal().iter().map(|l| l.iter().map(|&a| a as i64).collect()).collect()
    }

    #[pyo3(signature = (logits, values, margins = None))]
    fn apply_logits(&mut self, logits: PyReadonlyArray2<f32>, values: PyReadonlyArray1<f32>, margins: Option<PyReadonlyArray1<f32>>) -> PyResult<()> {
        let lg = logits.as_slice()?;
        let vs = values.as_slice()?;
        let ms = match &margins {
            Some(m) => Some(m.as_slice()?),
            None => None,
        };
        self.inner.apply_logits(lg, vs, ms).map_err(PyValueError::new_err)
    }

    /// One priors array per pending leaf (aligned with `pending_legal()`).
    #[pyo3(signature = (priors, values, margins = None))]
    fn apply_leaves(&mut self, priors: Vec<PyReadonlyArray1<f32>>, values: Vec<f64>, margins: Option<Vec<f64>>) -> PyResult<()> {
        let n = priors.len();
        if values.len() != n || margins.as_ref().map(|m| m.len() != n).unwrap_or(false) {
            return Err(PyValueError::new_err("priors, values and margins must have one entry per leaf"));
        }
        let mut slices: Vec<&[f32]> = Vec::with_capacity(n);
        for p in &priors {
            slices.push(p.as_slice()?);
        }
        let results: Vec<(&[f32], f64, f64)> = (0..n)
            .map(|i| (slices[i], values[i], margins.as_ref().map(|m| m[i]).unwrap_or(0.0)))
            .collect();
        self.inner.apply(&results).map_err(PyValueError::new_err)
    }

    /// Records of the games finished since the last call (dicts of numpy arrays).
    fn drain<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner.drain().into_iter().map(|r| record_to_dict(py, r)).collect()
    }
}

/// Softmax over the legal logits only, in float32 (numpy's arithmetic order:
/// subtract the max, exp, divide by the sum).
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

/// numpy's pairwise `sum()` of a float64 array (exposed for the parity test).
#[pyfunction]
fn numpy_sum(values: Vec<f64>) -> f64 {
    mcts::numpy_sum(&values)
}

#[pyfunction]
fn encode_action(source: usize, color: usize, dest: usize) -> u8 {
    azul::encode_action(source, color, dest)
}

#[pyfunction]
fn decode_action(action: i64) -> PyResult<(usize, usize, usize)> {
    Ok(azul::decode_action(to_u8_action(action)?))
}

#[pymodule]
fn ludometer_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyState>()?;
    m.add_class::<PyTree>()?;
    m.add_class::<PyArena>()?;
    m.add_function(wrap_pyfunction!(numpy_sum, m)?)?;
    m.add_function(wrap_pyfunction!(encode_action, m)?)?;
    m.add_function(wrap_pyfunction!(decode_action, m)?)?;
    m.add("ACTION_SPACE", ACTION_SPACE)?;
    m.add("ENCODED_SIZE", ENCODED_SIZE)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
