"""The trainer-side engine: games come from the hub, weights go to it.

Same four methods as every other self-play engine (``start``, ``set_weights``,
``play``, ``close``), so :class:`~ludometer.train.trainer.Trainer` is unchanged.
What differs is where the games come from:

* ``start(weights)`` publishes the run's config and weights version 1;
* ``set_weights(weights)`` publishes a new version, at most every
  ``publish_s`` seconds (the trainer calls it every iteration; a fleet does
  not need a new net every 90 seconds, and every publish is a commit);
* ``play(n, ...)`` polls the shards repo for files it has not consumed, pulls
  them, unpacks the games, and returns once it holds ``n`` — respecting
  ``should_stop`` and calling ``progress`` on every poll so the trainer's
  heartbeat keeps ``status.json`` alive while the fleet is quiet.

Shards played with weights older than ``max_lag`` versions behind the current
one are skipped (still marked consumed). The set of consumed shard names is
persisted in ``state_dir`` so a resumed run does not feed old games twice.

The seeds the trainer passes to ``play`` are ignored: generators own their
seeds (see :mod:`generator`), and the record carries the one it used.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Self

import numpy as np

from ludometer.cloud.hub import Hub, hub_from_spec, publish_weights
from ludometer.cloud.shards import read_shard
from ludometer.train.selfplay import GameRecord, SelfPlayConfig

__all__ = ["HubSelfPlay"]


class HubSelfPlay:
    def __init__(
        self,
        net_config: Any,
        config: SelfPlayConfig,
        run: str,
        shards: str | Hub,
        weights: str | Hub,
        state_dir: str | os.PathLike[str],
        train_config: dict[str, Any] | None = None,
        publish_s: float = 180.0,
        poll_s: float = 15.0,
        max_lag: int = 3,
        token: str | None = None,
        log: Any = print,
    ) -> None:
        self.net_config = net_config
        self.config = config
        self.run = run
        self.shards = (
            shards if not isinstance(shards, str) else hub_from_spec(shards, token)
        )
        self.weights = (
            weights if not isinstance(weights, str) else hub_from_spec(weights, token)
        )
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.train_config = dict(train_config or {})
        self.publish_s = float(publish_s)
        self.poll_s = float(poll_s)
        self.max_lag = int(max_lag)
        self._log = log or (lambda _m: None)
        self.workers = 0
        self.device = "hub"
        self.version = 0
        self._last_publish = 0.0
        self._pending: dict[str, np.ndarray] | None = None
        self._queue: deque[GameRecord] = deque()
        self._consumed: set[str] = set()
        self._state_path = self.state_dir / "hub_state.json"
        self._load_state()
        # diagnostics for the trainer's log line
        self.lag_hist: dict[int, int] = {}
        self.skipped = 0

    # ------------------------------------------------------------- state
    def _load_state(self) -> None:
        if self._state_path.exists():
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            self.version = int(data.get("version", 0))
            self._consumed = set(data.get("consumed", []))

    def _save_state(self) -> None:
        tmp = self._state_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"version": self.version, "consumed": sorted(self._consumed)}),
            encoding="utf-8",
        )
        os.replace(tmp, self._state_path)

    # ------------------------------------------------------------- weights
    def _net_config_dict(self) -> dict[str, Any]:
        cfg = self.net_config
        return cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)

    def _publish(self, weights: dict[str, np.ndarray], force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now - self._last_publish < self.publish_s:
            self._pending = weights
            return False
        self.version += 1
        publish_weights(
            self.weights,
            self.run,
            weights,
            self._net_config_dict(),
            self.version,
            extra={"run": self.run},
        )
        self._last_publish = now
        self._pending = None
        self._save_state()
        self._log(
            f"[hub] published weights v{self.version} to {self.weights.describe()}"
        )
        return True

    def start(self, weights: dict[str, np.ndarray] | None = None) -> None:
        # The generators read the run's search settings from here, once.
        if self.train_config:
            self.weights.put_bytes(
                json.dumps(self.train_config, indent=1).encode("utf-8"),
                f"{self.run}/config.json",
            )
        if weights is not None:
            self._publish(weights, force=True)

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        self._publish(weights)

    # ------------------------------------------------------------- games
    def _pull_new_shards(self) -> int:
        """Download every unconsumed shard of this run; returns games queued."""
        names = [n for n in self.shards.list(f"{self.run}/") if n.endswith(".npz")]
        fresh = [n for n in names if n not in self._consumed]
        got = 0
        for name in fresh:
            local = self.state_dir / "shards" / Path(name).name
            try:
                self.shards.get(name, local)
                records, meta = read_shard(local)
            except Exception as exc:  # noqa: BLE001 - a bad shard must not kill the run
                self._log(f"[hub] skipping unreadable shard {name}: {exc}")
                self._consumed.add(name)
                continue
            version = int(meta.get("weights_version", 0))
            lag = self.version - version
            self.lag_hist[lag] = self.lag_hist.get(lag, 0) + len(records)
            self._consumed.add(name)
            if lag > self.max_lag:
                self.skipped += len(records)
                local.unlink(missing_ok=True)
                continue
            self._queue.extend(records)
            got += len(records)
            local.unlink(missing_ok=True)
        if fresh:
            self._save_state()
        return got

    def play(
        self,
        n_games: int,
        seed_start: int,
        progress: Any = None,
        should_stop: Any = None,
    ) -> list[GameRecord]:
        del seed_start  # generators own their seeds
        if n_games <= 0:
            return []
        # A publish that was rate-limited earlier goes out now if it is due.
        if self._pending is not None:
            self._publish(self._pending)
        while len(self._queue) < n_games:
            got = self._pull_new_shards()
            if progress is not None:
                progress(min(len(self._queue), n_games), n_games)
            if should_stop is not None and should_stop():
                break
            if got == 0:
                time.sleep(self.poll_s)
        out = [self._queue.popleft() for _ in range(min(n_games, len(self._queue)))]
        return out

    def close(self) -> None:
        self._save_state()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
