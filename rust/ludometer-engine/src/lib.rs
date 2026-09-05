//! ludometer-engine: Azul rules, PUCT tree and self-play arena in Rust.
//!
//! Layered exactly like the Python package it mirrors:
//! [`rng`] (CPython-exact and fast generators), [`azul`] (the rules),
//! [`mcts`] (the tree and the leaf protocol), [`arena`] (many games, one batch).
//! The `python` feature adds the PyO3 module `ludometer_rs`.

pub mod rng;

#[cfg(feature = "python")]
mod py;
