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

## Status

Work in progress — built autonomously by Claude. Progress notes in `NOTES_FOR_REMI.md`.
