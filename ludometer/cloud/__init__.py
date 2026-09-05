"""Self-play on a fleet of Hugging Face Jobs, training at home.

The self-play loop is bound by the pure-Python tree search (~90 us per
simulation per core), not by the net, so what a run needs is *cores*. A
``cpu-upgrade`` Job (8 vCPU) costs $0.03/hour — some 25x less per core than any
GPU flavor — and a fleet of them produces games at 10-20x this Mac's rate for
under a dollar an hour. This package is the plumbing:

* :mod:`shards`      — a block of finished games as one ``.npz``, per-game
  boundaries kept, so the trainer's replay buffer ingests it game by game;
* :mod:`hub`         — one tiny file-store interface with two backends: a Hub
  repo (retries, versioned weights) and a local directory (tests);
* :mod:`generator`   — the job side: fetch weights, play, upload, poll;
* :mod:`hub_selfplay`— the trainer side: a self-play engine with the pool
  interface that publishes weights and consumes shards, so ``trainer.py`` does
  not know the games were played elsewhere;
* :mod:`fleet`       — launch / list / cancel jobs and keep the spend ledger.

Design: docs/superpowers/specs/2026-09-05-cloud-selfplay-design.md.
"""
