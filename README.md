# Ludometer

**Does a good game teach linearly?**

Ludometer is a reinforcement-learning framework for measuring the *quality* of a board
game through the shape of an AI agent's learning curve.

## The hypothesis

- If an agent learns **too slowly at the start**, the rules are probably hard to grasp.
- If it learns **too fast / plateaus sharply**, the game probably lacks depth.
- A *good* game should produce a roughly **linear Elo progression** in a learning agent.

We calibrate the method on **Azul** (Michael Kiesling, Spiel des Jahres 2018) — a game
widely considered excellent — then reuse the framework to evaluate new board game designs.

## Components

- `ludometer/azul/` — full 2-player Azul rules engine (fast, deterministic, fully tested)
- `ludometer/agents/` — baseline agents (random, greedy, heuristic) and the neural agent
- `ludometer/train/` — AlphaZero-style self-play training (PyTorch, Apple MPS)
- `ludometer/eval/` — arena + Elo rating of checkpoints against a fixed anchor pool
- `web/` — browser dashboard (training progress) and a GUI to play against the AI

## Quick start

```bash
uv sync
uv run pytest              # engine tests
uv run ludometer-gui       # play Azul against the current best agent
```

## Play against the AI

```bash
uv sync
uv run ludometer-gui       # opens http://127.0.0.1:8737/ and deals a game
```

The **Opponent** dropdown defaults to **“Strongest trained (auto)”** — the agent spec
`best`. That spec scans every `runs/*/elo.jsonl`, takes the highest-Elo checkpoint whose
`.pt` file is still on disk, and plays it. The dropdown shows which one that is
(`ckpt-023040 · +1920 Elo · run run1`), and the table-talk panel repeats it once the game
starts: *“You’re facing ckpt-023040, rated +1920 on our internal ladder.”* The other
entries are the scripted baselines (heuristic / greedy / random) and
`Trained checkpoint…`, which takes a spec by hand
(`mcts:runs/run1/checkpoints/ckpt-020992.pt?sims=400`).

Resolution happens **when you press “Deal tiles”**, not when the page loads: while a run is
training the model keeps improving, so every new game automatically faces the newest
strongest checkpoint. Nothing to edit, no path to copy.

**Sims** is how many positions the MCTS search visits per move — the strength/speed dial:

| Sims | Feel |
|------|------|
| 100  | replies in a blink, noticeably weaker |
| 400  | default: strong, roughly a second per move |
| 1200 | strongest, a few seconds per move |

Elo ratings are measured at the trainer's own eval sims, so a checkpoint played at 100 sims
is weaker than its rating suggests and at 1200 sims somewhat stronger.

Same thing without the browser (the GUI, the arena and the trainer share one registry):

```bash
uv run python -c "from ludometer.agents.registry import find_best_checkpoint; print(find_best_checkpoint())"
```

## Status

Work in progress — built autonomously by Claude. Progress notes in `NOTES_FOR_REMI.md`.
