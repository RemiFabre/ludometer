//! Search-bound speed of the tree: microseconds per simulation with a constant
//! evaluator (no net), the number docs/RUST_ENGINE.md §1 targets (<= 6 us).
//!
//!     cargo run --release --example bench_tree
use std::time::Instant;

use ludometer_rs::azul::State;
use ludometer_rs::mcts::{MctsConfig, Tree};
use ludometer_rs::rng::RngKind;

fn main() {
    let sims = 1024u32;
    let games = 8;
    let mut total_sims = 0u64;
    let mut total_moves = 0u64;
    let mut clone_apply = 0u64;
    let t0 = Instant::now();
    for seed in 0..games {
        let mut state = State::new_game(seed, RngKind::Fast);
        let mut tree = Tree::new(
            MctsConfig { sims, tree_reuse: true, chance_children: 4, ..Default::default() },
            true,
            seed,
            true,
            RngKind::Fast,
        );
        let mut moves = 0;
        while !state.is_terminal && moves < 400 {
            let legal = state.legal_actions();
            let action = if legal.len() == 1 {
                legal[0]
            } else {
                let r = tree
                    .search(&state, |_s, l| (vec![1.0 / l.len() as f32; l.len()], 0.01, 0.0), None, None, None)
                    .unwrap();
                total_sims += (r.sims - tree.reused_visits) as u64;
                r.argmax_policy() as u8
            };
            state.apply(action).unwrap();
            tree.advance(action);
            moves += 1;
        }
        total_moves += moves;
        clone_apply += tree.nodes_created;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "{} games, {} moves, {} new simulations, {} nodes in {:.2}s: {:.2} us/sim, {:.0} sims/s",
        games,
        total_moves,
        total_sims,
        clone_apply,
        dt,
        dt * 1e6 / total_sims as f64,
        total_sims as f64 / dt
    );
    // Rules alone: clone + apply + legal over random play.
    let t1 = Instant::now();
    let mut n = 0u64;
    let mut g = ludometer_rs::rng::SplitMix64::new(1);
    let mut legal = Vec::new();
    for seed in 0..2000u64 {
        let mut s = State::new_game(seed, RngKind::Fast);
        while !s.is_terminal {
            s.legal_actions_into(&mut legal);
            let a = legal[g.below(legal.len() as u64) as usize];
            let mut c = s;
            c.apply(a).unwrap();
            s = c;
            n += 1;
        }
    }
    let dt = t1.elapsed().as_secs_f64();
    println!("rules: {} clone+apply+legal in {:.2}s: {:.3} us each", n, dt, dt * 1e6 / n as f64);
    let t2 = Instant::now();
    let s = State::new_game(3, RngKind::Fast);
    let mut v = [0.0f32; 182];
    let mut acc = 0.0f32;
    for _ in 0..1_000_000 {
        s.encode(&mut v);
        acc += v[7];
    }
    let dt = t2.elapsed().as_secs_f64();
    println!("encode: {:.3} us each ({})", dt, acc);
}
