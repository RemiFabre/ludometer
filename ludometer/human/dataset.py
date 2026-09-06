"""Turn validated human games into a ``replay.npz`` that ``--pretrain`` can read.

The output is byte-for-byte the format
:class:`ludometer.train.replay.ReplayBuffer` writes, because it *is* that class
doing the writing — importing it costs nothing (that module is numpy-only) and
guarantees we never drift from the format the trainer expects. Per row:

===============  ====================================================================
``states``       the position before the human move, mover's frame (182 floats)
``policies``     **one-hot on the move the human played** (180 floats)
``values``       the game outcome in the mover's frame, +1 / 0 / -1
``margins``      ``tanh(final score diff / 20)``, mover's frame, mask 1
``aux``          the 30 final-wall bits for both walls, mover's frame, mask 1
``policy_mask``  1 — every row that carries a policy target
===============  ====================================================================

Elite-player-only learning
--------------------------
This is the part that decides whether the dataset is worth training on. A BGA
table between a top-200 player and a 400-Elo opponent contains two players'
decisions, and imitating the weaker one is worse than having no data: the policy
head would learn a mixture of "what a strong player does" and "what a beginner
does" with no way to tell them apart.

So **one game contributes one player's turns**: the *target*, chosen by
:func:`~ludometer.human.fetch.choose_target_player` (the ranked player whose game
list the table came from; the higher-Elo one when both seats are ranked). Rows
where the opponent is to move are still *replayed* — that is what validates the
game in :func:`~ludometer.human.convert.convert_game` — but by default they
produce no training row at all:

* ``opponent_rows="drop"`` (default) — only the target's positions are written.
  Every written row then has a policy target, a value and a margin **in the
  target's own frame**, because the mover *is* the target;
* ``opponent_rows="value-only"`` — every position is written, but opponent rows
  carry a zeroed policy and ``policy_mask = 0``, exactly the convention
  ``replay.npz`` already uses for run6's cheaply-searched positions
  (:mod:`ludometer.train.replay`). Their value/margin/aux are real labels in the
  mover's frame, so the value head still learns from them while the policy head
  never sees a weak player's move. Use it if the value head turns out to be data
  starved; it is not the default, because "learn only from the elite player" is
  the stated goal.

:class:`GameMeta` carries **both** seats' Elos, so ``min_target_elo_raw`` can put
a floor under the *target* specifically (a floor on "somebody at the table" is
what the fetcher already does and is much weaker).

Two things to know about the policy target itself. It is a *hard* one-hot, not a
visit distribution, so its gradient is a plain cross-entropy towards "what a
strong human did"; that is the standard imitation signal and it is the reason a
human-pretrained net starts with sane move preferences instead of noise. And it is
inevitably noisier than an MCTS target — humans blunder, and Azul's floor-line
sacrifices look like blunders until several rounds later — which is why the Elo
floor on the target matters more here than dataset size does.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ludometer.azul.engine import ACTION_SPACE, ENCODED_SIZE
from ludometer.human.convert import HumanGame
from ludometer.human.fetch import choose_target_player, player_elos
from ludometer.train.replay import ReplayBuffer

__all__ = [
    "OPPONENT_DROP",
    "OPPONENT_VALUE_ONLY",
    "DatasetStats",
    "GameMeta",
    "add_game",
    "build_dataset",
    "one_hot_policies",
    "target_mask",
]

#: opponent-turn handling: leave those positions out of the file entirely.
OPPONENT_DROP = "drop"
#: opponent-turn handling: keep them with a zeroed policy and ``policy_mask = 0``.
OPPONENT_VALUE_ONLY = "value-only"


@dataclass(frozen=True)
class GameMeta:
    """Who was who at one table — the elite-only filter's input.

    Written by the crawl into every raw payload (``payload["meta"]``) and into the
    state file, so a dataset can be rebuilt with a different Elo floor without a
    single extra request.
    """

    table_id: int
    target_player_id: int | None = None
    source_player_id: int | None = None
    rank: int | None = None
    #: raw (~1500-centred) Elo per BGA player id, both seats when known
    elos: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_raw(
        cls,
        table_id: int,
        payload: dict[str, Any],
        ranked_ids: Iterable[int] = (),
        ranked_elos: dict[int, float] | None = None,
    ) -> GameMeta:
        """Read the crawl's ``meta`` block, falling back to ``tableinfos``.

        Raw payloads fetched before the crawl existed have no ``meta``, so the
        target is re-derived from the Elos in ``infos`` — the same rule, one step
        weaker because it cannot know whose history the table came from.

        ``ranked_elos`` maps a BGA player id to their **raw** ladder Elo (from the
        ranking snapshot in ``state.json``). It is the *authoritative* source of the
        target's Elo: a real ``tableinfos`` payload carries **no** per-seat Elo
        (docs/HUMAN_GAMES.md §12.3), so ``meta.elos`` and ``player_elos(infos)`` are
        both empty on real data, and without this backfill the Elo floor would read
        every target as ``None`` and drop the whole dataset. The crawl also records
        ``source_elo_raw`` for the ladder player whose history the table came from,
        which is honoured too.
        """
        meta = payload.get("meta") or {}
        infos = payload.get("infos") or {}
        ranked_elos = {int(k): float(v) for k, v in (ranked_elos or {}).items()}
        elos = {int(k): float(v) for k, v in (meta.get("elos") or {}).items()}
        if not elos:
            elos = player_elos(infos)
        seats = sorted(elos) or [
            int(p)
            for p in ((infos.get("data") or infos).get("players") or {})
            if str(p).isdigit()
        ]
        source = meta.get("source_player_id")
        target = meta.get("target_player_id")
        if target is None:
            target = choose_target_player(
                seats, elos, source_player_id=source, ranked_ids=ranked_ids
            )
        # Backfill Elos the log/tableinfos never carried, from the ladder snapshot.
        # `setdefault` so anything the payload *did* record still wins.
        source_elo = meta.get("source_elo_raw")
        if source is not None and source_elo is not None:
            elos.setdefault(int(source), float(source_elo))
        for pid in (*seats, *( () if target is None else (int(target),) )):
            if pid in ranked_elos:
                elos.setdefault(pid, ranked_elos[pid])
        return cls(
            table_id=int(table_id),
            target_player_id=None if target is None else int(target),
            source_player_id=None if source is None else int(source),
            rank=None if meta.get("rank") is None else int(meta["rank"]),
            elos=elos,
        )

    def target_elo(self) -> float | None:
        if self.target_player_id is None:
            return None
        return self.elos.get(int(self.target_player_id))

    def opponent_elo(self) -> float | None:
        others = [
            elo
            for pid, elo in self.elos.items()
            if self.target_player_id is None or pid != int(self.target_player_id)
        ]
        return min(others) if others else None


@dataclass
class DatasetStats:
    """What went into the file — printed by the CLI, read by the harvest page."""

    games: int = 0
    #: rows actually written to the buffer
    positions: int = 0
    #: rows carrying a policy target from the **target** player
    elite_positions: int = 0
    #: decision points replayed in the accepted games (both players)
    all_positions: int = 0
    games_rejected: int = 0
    skipped_no_target: int = 0
    skipped_below_elo: int = 0
    elo_floor_raw: float = 0.0
    #: raw-Elo floor above which the *opponent's* moves also become policy
    #: targets (0 = off, the classic elite-seat-only behaviour)
    dual_floor_raw: float = 0.0
    #: games where both seats cleared ``dual_floor_raw`` and both became targets
    dual_target_games: int = 0
    opponent_rows: str = OPPONENT_DROP
    tables: list[int] = field(default_factory=list)
    outcomes: dict[str, int] = field(
        default_factory=lambda: {"p0": 0, "p1": 0, "draw": 0}
    )
    #: the same three from the **target's** point of view, which is the one that
    #: matters when the value target is the target player's outcome
    target_outcomes: dict[str, int] = field(
        default_factory=lambda: {"win": 0, "loss": 0, "draw": 0}
    )
    reasons: dict[str, int] = field(default_factory=dict)
    #: one record per game, in the order its rows were appended to the npz (the
    #: buffer holds every row, so cumulative ``rows`` offsets reconstruct exact
    #: per-row spans). ``target_elo_raw`` is the target's raw ladder Elo — the
    #: hook for Elo-weighted training without changing the npz format.
    game_records: list[dict[str, Any]] = field(default_factory=list)

    def note(
        self,
        game: HumanGame,
        rows: int = 0,
        elite_rows: int = 0,
        target_seat: int | None = None,
        target_elo_raw: float | None = None,
        target_elo_after: float | None = None,
        dual: bool = False,
        opponent_elo_raw: float | None = None,
    ) -> None:
        self.games += 1
        if dual:
            self.dual_target_games += 1
        self.positions += int(rows)
        self.elite_positions += int(elite_rows)
        self.all_positions += len(game)
        self.tables.append(game.table_id)
        self.game_records.append(
            {
                "table_id": game.table_id,
                "rows": int(rows),
                # today's ladder snapshot (always known for a crawled target)...
                "target_elo_raw": target_elo_raw,
                # ...and the target's raw Elo right after THIS game (from the
                # getGames history row), the time-accurate number — None for
                # games listed before rows were captured (2026-08-19).
                "target_elo_after": target_elo_after,
                # the opponent's raw Elo when known, and whether their moves are
                # policy targets too (both seats above the dual floor)
                "opponent_elo_raw": opponent_elo_raw,
                "dual": bool(dual),
            }
        )
        key = "draw" if game.outcome == 0 else ("p0" if game.outcome > 0 else "p1")
        self.outcomes[key] += 1
        if target_seat is not None:
            sign = 1.0 if target_seat == 0 else -1.0
            value = game.outcome * sign
            self.target_outcomes[
                "draw" if value == 0 else ("win" if value > 0 else "loss")
            ] += 1

    def note_rejected(self, reason: str) -> None:
        self.games_rejected += 1
        key = reason.strip()[:80] or "unknown"
        self.reasons[key] = self.reasons.get(key, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "games": self.games,
            "positions": self.positions,
            "elite_positions": self.elite_positions,
            "all_positions": self.all_positions,
            "games_rejected": self.games_rejected,
            "skipped_no_target": self.skipped_no_target,
            "skipped_below_elo": self.skipped_below_elo,
            "elo_floor_raw": self.elo_floor_raw,
            "dual_floor_raw": self.dual_floor_raw,
            "dual_target_games": self.dual_target_games,
            "opponent_rows": self.opponent_rows,
            "outcomes": dict(self.outcomes),
            "target_outcomes": dict(self.target_outcomes),
            "reasons": dict(self.reasons),
            "game_records": list(self.game_records),
        }

    def write(self, path: str | Path) -> Path:
        """Write the sidecar the progress page reads (``<npz>.stats.json``).

        The page is stdlib-only and must not open a 200 MB ``.npz`` to answer "how
        many positions do we have?", so the answer is written next to it.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        tmp.replace(path)
        return path


def stats_path_for(npz_path: str | Path) -> Path:
    """``data/human/replay.npz`` -> ``data/human/replay.stats.json``."""
    path = Path(npz_path)
    return path.with_name(path.stem + ".stats.json")


def one_hot_policies(actions: np.ndarray) -> np.ndarray:
    """``(T,)`` action ids -> ``(T, 180)`` one-hot float32 policy targets."""
    actions = np.asarray(actions, dtype=np.int64).reshape(-1)
    if actions.size and (actions.min() < 0 or actions.max() >= ACTION_SPACE):
        raise ValueError(f"action id outside 0..{ACTION_SPACE - 1}")
    policies = np.zeros((len(actions), ACTION_SPACE), dtype=np.float32)
    policies[np.arange(len(actions)), actions] = 1.0
    return policies


def target_mask(game: HumanGame, target_seat: int | None) -> np.ndarray:
    """``(T,)`` bool: the rows where the **target** player is to move.

    With ``target_seat is None`` (no target could be identified) every row is
    selected, which is the legacy "learn from both players" behaviour — callers that
    care pass ``require_target=True`` to :func:`build_dataset` and drop the game
    instead.
    """
    if target_seat is None:
        return np.ones(len(game), dtype=bool)
    return np.asarray(game.movers, dtype=np.int64) == int(target_seat)


def add_game(
    buffer: ReplayBuffer,
    game: HumanGame,
    target_seat: int | None = None,
    opponent_rows: str = OPPONENT_DROP,
) -> tuple[int, int]:
    """Append one converted game's rows to ``buffer``.

    Returns ``(rows_written, rows_with_a_policy_target)``. With the default
    ``opponent_rows="drop"`` the two are equal and both count only the target
    player's turns — every stored row's value and margin are then in the target's
    frame, because ``movers[i] == target_seat`` for every row kept.
    """
    if game.states.shape[1:] != (ENCODED_SIZE,):
        raise ValueError(
            f"states have shape {game.states.shape}, expected (T, {ENCODED_SIZE})"
        )
    if opponent_rows not in (OPPONENT_DROP, OPPONENT_VALUE_ONLY):
        raise ValueError(
            f"opponent_rows must be {OPPONENT_DROP!r} or {OPPONENT_VALUE_ONLY!r}, "
            f"got {opponent_rows!r}"
        )
    mask = target_mask(game, target_seat)
    elite = int(mask.sum())
    buffer.games_added += 1
    if opponent_rows == OPPONENT_DROP:
        if not elite:
            return 0, 0
        policies = one_hot_policies(np.asarray(game.actions)[mask])
        written = buffer.add(
            game.states[mask],
            policies,
            game.values()[mask],
            game.margins()[mask],
            margin_mask=1.0,
            aux=game.aux[mask],
            aux_mask=1.0,
            policy_mask=1.0,
        )
        return written, elite
    # value-only: keep every position, but zero the policy where the opponent moved
    policies = one_hot_policies(game.actions)
    policies[~mask] = 0.0
    written = buffer.add(
        game.states,
        policies,
        game.values(),
        game.margins(),
        margin_mask=1.0,
        aux=game.aux,
        aux_mask=1.0,
        policy_mask=mask.astype(np.float32),
    )
    return written, elite


def _split(item: Any) -> tuple[HumanGame, GameMeta | None]:
    if isinstance(item, HumanGame):
        return item, None
    game, meta = item
    return game, meta


def build_dataset(
    games: Iterable[HumanGame | tuple[HumanGame, GameMeta | None]],
    path: str | Path,
    capacity: int | None = None,
    seed: int = 0,
    min_target_elo_raw: float = 0.0,
    dual_target_min_elo_raw: float = 0.0,
    opponent_rows: str = OPPONENT_DROP,
    require_target: bool = False,
    stats: DatasetStats | None = None,
    write_stats: bool = True,
    elo_at_game: dict[int, dict[int, float]] | None = None,
) -> DatasetStats:
    """Write every game in ``games`` to a ``replay.npz`` at ``path``.

    ``games`` yields either bare :class:`~ludometer.human.convert.HumanGame` objects
    or ``(game, meta)`` pairs; the meta is what identifies the elite player, so the
    real pipeline always passes it (see :meth:`GameMeta.from_raw`).

    ``min_target_elo_raw`` is a floor on the **target** player's raw Elo (add 1300
    to a displayed number). ``require_target=True`` drops games whose elite seat
    cannot be identified rather than falling back to learning from both players.

    ``dual_target_min_elo_raw`` (per Rémi, 2026-08-20): when the *opponent* is
    also super strong — their raw Elo at game time (``elo_after``) or, failing
    that, their ladder-snapshot Elo clears this floor — their moves are just as
    much worth learning from, so **both** seats' rows get policy targets instead
    of the opponent's being kept value-only/dropped. 0 = off. An opponent whose
    Elo is simply unknown (most non-crawled players) never qualifies.

    ``capacity`` defaults to the number of rows the games actually produce, so the
    file holds all of them and nothing is silently dropped by the ring; pass an
    explicit capacity to cap the dataset (the newest rows win, as in training).
    """
    stats = stats or DatasetStats()
    stats.elo_floor_raw = float(min_target_elo_raw)
    stats.dual_floor_raw = float(dual_target_min_elo_raw)
    stats.opponent_rows = opponent_rows
    kept: list[
        tuple[HumanGame, int | None, float | None, float | None, bool, float | None]
    ] = []
    for item in games:
        game, meta = _split(item)
        target_seat = None
        target_elo = None
        elo_after = None
        dual = False
        opp_elo = None
        if meta is not None:
            target_seat = game.seat_of(meta.target_player_id)
            target_elo = meta.target_elo()
            if meta.target_player_id is not None:
                elo_after = (elo_at_game or {}).get(game.table_id, {}).get(
                    int(meta.target_player_id)
                )
            if target_seat is None:
                if require_target:
                    stats.skipped_no_target += 1
                    continue
            elif min_target_elo_raw:
                # the per-game `elo_after` (when the history row was captured) is
                # the honest number for "was this player elite WHEN they played
                # this game"; the ladder snapshot is the fallback
                floor_elo = elo_after if elo_after is not None else target_elo
                if floor_elo is None or floor_elo < float(min_target_elo_raw):
                    stats.skipped_below_elo += 1
                    continue
            if target_seat is not None and dual_target_min_elo_raw:
                # same honesty rule for the opponent: per-game Elo first, ladder
                # snapshot second, and "unknown" never qualifies
                opp_ids = [
                    int(pid)
                    for pid in game.player_ids
                    if meta.target_player_id is None
                    or int(pid) != int(meta.target_player_id)
                ]
                for pid in opp_ids:
                    at_game = (elo_at_game or {}).get(game.table_id, {}).get(pid)
                    opp_elo = at_game if at_game is not None else meta.elos.get(pid)
                if opp_elo is not None and opp_elo >= float(dual_target_min_elo_raw):
                    dual = True
        elif require_target:
            stats.skipped_no_target += 1
            continue
        kept.append((game, target_seat, target_elo, elo_after, dual, opp_elo))

    total = 0
    for game, target_seat, _, _, dual, _ in kept:
        rows = int(target_mask(game, None if dual else target_seat).sum())
        total += len(game) if opponent_rows == OPPONENT_VALUE_ONLY else rows
    buffer = ReplayBuffer(capacity=max(1, capacity or total), seed=seed)
    for game, target_seat, target_elo, elo_after, dual, opp_elo in kept:
        written, elite = add_game(
            buffer,
            game,
            # a dual game learns from BOTH seats: target_seat=None makes every
            # row a policy target (value/margin stay in the mover's frame)
            target_seat=None if dual else target_seat,
            opponent_rows=opponent_rows,
        )
        stats.note(
            game,
            rows=written,
            elite_rows=elite,
            target_seat=target_seat,
            target_elo_raw=target_elo,
            target_elo_after=elo_after,
            dual=dual,
            opponent_elo_raw=opp_elo,
        )
    buffer.save(Path(path))
    if write_stats:
        stats.write(stats_path_for(path))
    return stats
