# ludometer-engine (Rust)

The Rust twin of `ludometer/azul/engine.py`, `ludometer/train/mcts.py` and
`ludometer/train/selfplay_batched.py`: rules, tree walk and the many-games
arena. The neural net stays in PyTorch; Python hands in priors and values,
Rust hands out encoded leaves. See `docs/RUST_ENGINE.md` for the brief and
`docs/superpowers/plans/2026-09-06-rust-engine.md` for the plan.

Build for local work (installs `ludometer_rs` into `.venv`):

    uv pip install --python .venv/bin/python maturin
    nice -n 15 .venv/bin/maturin develop --release -m rust/ludometer-engine/Cargo.toml

Pure Rust tests (no Python needed):

    cd rust && cargo test --release
