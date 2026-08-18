---
license: cc0-1.0
pretty_name: "Faïence human-vs-net Azul games"
tags:
  - game-records
  - azul
  - reinforcement-learning
  - human-play
---

<!-- Source of truth for this card: web/ingest/DATASET.md in
     https://github.com/RemiFabre/ludometer -->

# Faïence: human-vs-net Azul games

Every game played on [Faïence](https://remifabre-faience.static.hf.space/), a
free browser implementation of the rules of *Azul* (Michael Kiesling) against
a neural net trained by self-play, unless the player switched sharing off.
This dataset is the training pile the playing page tells its players about,
and it is public precisely so that a player can read everything the project
collects. Records are anonymous by construction: moves, deals, which net
played, and the score. No names, no accounts, no IPs, no user agents.

## Layout

`games/YYYY-MM-DD/<timestamp>-<n>.jsonl`, one file per ingest batch, one JSON
object per line. Nothing is ever rewritten; new batches only add files.

## Record format (`faience-game/1`)

Each line is a canonical record rebuilt by the collector
([RemiFabre/faience-ingest](https://huggingface.co/spaces/RemiFabre/faience-ingest)),
which replayed the game in the real engine and kept it only if the recorded
deals, final scores and round count reproduce exactly. Fields:

- `received_at` (server clock, ISO) and `created_at` (client clock, may be null)
- `seed`: the game's RNG seed (mulberry32, the page's own RNG)
- `human_seat`, `human_first`: which of the two seats the human held
- `net`: `{run, checkpoint, elo, params, backend}` of the opponent
- `think_time_s`: the AI's search budget per move (0 = policy head only)
- `moves`: `[{ply, player, action, sims?, value?}]`, `action` encoded as
  `source*30 + color*6 + dest` (identical in the JS and Python engines);
  `sims` is the positions the net searched for its move on the visitor's
  machine, `value` its root value on a [-1, 1] scale
- `deals`: per round, the five factories plus bag and lid counts, so a record
  replays independently of any RNG port
- `final`: `{finished, scores, outcome, rounds, exhausted}`; `finished:
  false` marks an abandoned game (position data with no outcome; train the
  value head on these with care, or not at all)

## Caveats

- The collector deduplicates retried submissions by content, but a restart
  can rarely let a duplicate through: deduplicate by
  `(seed, human_seat, moves, final.scores, final.finished)` when it matters.
- An abandoned game that was later resumed and finished in the same tab can
  appear twice: once `finished: false`, once `finished: true` with the same
  seed and a longer move list. Prefer the finished one.
- Play strength varies wildly: these are self-selected browser visitors, from
  first-time players to strong club players.

## Provenance and license

Collected by the [Faïence ingest Space](https://huggingface.co/spaces/RemiFabre/faience-ingest)
from the [Faïence playing page](https://remifabre-faience.static.hf.space/);
code and methodology in [RemiFabre/ludometer](https://github.com/RemiFabre/ludometer).
The records are dedicated to the public domain (CC0). Azul is a game by
Michael Kiesling; this fan research project is not affiliated with or
endorsed by its publishers.
