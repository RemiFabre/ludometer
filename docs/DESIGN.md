# Ludometer design

Source of truth for all contributors (human or agent). If code and this doc disagree, fix one of them.

## Goal

Train an RL agent to play 2-player **Azul** as strongly as possible on a single Mac
(12 cores, 36 GB, Apple MPS), while logging an **Elo-vs-training-compute curve** whose
shape (linearity) we use as a measure of game quality.

## Azul rules (official, 2 players) — engine spec

- 100 tiles: 20 each of 5 colors (indices 0..4: blue, yellow, red, black, teal).
- **5 factory displays** (2-player count) + a shared **center**. First-player marker starts in center.
- **Round setup**: fill each factory with 4 tiles drawn uniformly at random from the bag.
  If the bag empties mid-refill, refill the bag from the lid (discard) and continue.
  If both bag and lid are empty, leave remaining factories partially filled/empty and play on.
- **Turn** (players alternate, starting with first-player-marker holder):
  1. Either take ALL tiles of one color from one factory (remaining tiles of that factory go
     to the center), or take ALL tiles of one color from the center. The first player to take
     from the center each round also takes the first-player marker and places it on their
     floor line (it occupies a floor slot).
  2. Place ALL taken tiles into exactly one pattern line (rows of capacity 1..5) **or** the
     floor line. A pattern line may only contain one color; a color is forbidden in a pattern
     line if that color is already on the wall in the same row. Tiles beyond the line's
     capacity overflow to the floor line. Floor line has 7 slots with penalties
     [-1,-1,-2,-2,-2,-3,-3]; tiles beyond 7 go to the lid.
- **Round end** (factories and center all empty) — wall tiling:
  - For each **complete** pattern line, top to bottom: move one tile to the wall in that row;
    the rest of the line's tiles go to the lid. Wall column for color `c` in row `r` is
    `(c + r) % 5` (standard fixed wall). Incomplete lines carry over.
  - **Scoring per placed tile**: let `h` = length of the contiguous horizontal run through the
    tile, `v` = vertical run. Score `(h if h>1 else 0) + (v if v>1 else 0)`, or `1` if both
    runs are length 1.
  - Apply floor penalties (floor tiles go to the lid; the first-player marker counts a penalty
    slot but returns to its holder). Score is floored at 0.
  - Marker holder starts next round; marker returns to center at refill.
- **Game end**: after the round where any player completes a horizontal row of 5.
  Bonuses: +2 per complete row, +7 per complete column, +10 per color with all 5 on the wall.
  Winner: highest score; tie-break by more complete rows; then a draw.

## Action space (fixed, 180 actions)

`action_id = source * 30 + color * 6 + dest`

- `source`: 0..4 = factories, 5 = center
- `color`: 0..4
- `dest`: 0..4 = pattern lines, 5 = floor

## Engine API (`ludometer/azul/engine.py`)

```python
class AzulState:
    ACTION_SPACE: int = 180
    @classmethod
    def new_game(cls, seed: int, num_players: int = 2) -> "AzulState": ...
    current_player: int          # 0 or 1
    def legal_actions(self) -> list[int]: ...
    def apply(self, action_id: int) -> None      # mutates; handles round/game transitions
    def clone(self) -> "AzulState": ...
    is_terminal: bool
    scores: list[int]
    def outcome(self) -> float | None            # +1 P0 wins, -1 P1 wins, 0 draw
    def encode(self) -> np.ndarray               # float32, current-player perspective, fixed size
    def render_text(self) -> str                 # human-readable board
    def to_json(self) -> dict                    # full state for the GUI
```

Design constraints: pure Python + numpy only in the engine (no torch), deterministic given
seed (use an internal `random.Random`), fast — target ≥2,000 full random games/sec single-core.
Chance only occurs at round refill; within a round the game is deterministic (MCTS relies on this).

## Agents (`ludometer/agents/`)

Common interface: `act(state) -> action_id`. Baselines:
- `RandomAgent`
- `GreedyAgent` — 1-ply immediate-score maximizer
- `HeuristicAgent` — hand-tuned (values adjacency, avoids floor overflow, denies opponent)

## Training (`ludometer/train/`)

AlphaZero-lite: policy+value MLP/ResNet on encoded state, MCTS with action masking (PUCT),
self-play with Dirichlet noise at root and temperature schedule, replay buffer, train on MPS.
At round-boundary chance nodes in MCTS, sample refills from the known bag distribution
(bag+lid contents are public information).

## Evaluation (`ludometer/eval/`)

Fixed anchor pool: Random (anchored 0 Elo), Greedy, Heuristic, and frozen checkpoints.
Each new checkpoint plays N games (alternating first player) vs the pool; ratings fit by
maximum likelihood (Bradley-Terry) with anchors held fixed. Results appended as JSONL to
`runs/<run>/elo.jsonl`. This makes curves comparable across the run and across runs.

## Logging & dashboard

Everything observable lives in `runs/<run_name>/`: `config.json`, `train.jsonl` (losses,
buffer size, games played), `elo.jsonl`, `checkpoints/`. The dashboard (`web/`) renders these.
