//! PyO3 bindings: the `ludometer_rs` module.
//!
//! `State` mirrors `ludometer.azul.engine.AzulState` (same method names, numpy
//! `encode()`), `Tree` mirrors `ludometer.train.mcts.MCTS` through the leaf
//! protocol, `Arena` runs many games for `ludometer.train.selfplay_rust`.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};

use crate::azul::{self, State as RsState, ACTION_SPACE, ENCODED_SIZE, NUM_COLORS, NUM_FACTORIES};
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
    m.add_function(wrap_pyfunction!(encode_action, m)?)?;
    m.add_function(wrap_pyfunction!(decode_action, m)?)?;
    m.add("ACTION_SPACE", ACTION_SPACE)?;
    m.add("ENCODED_SIZE", ENCODED_SIZE)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
