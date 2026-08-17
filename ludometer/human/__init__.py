"""Learning from human games: Board Game Arena (BGA) Azul replays.

Read ``docs/HUMAN_GAMES.md`` first — it is the handoff document for this package
(endpoints, cookies, ToS reality, mapping table, what is still missing).

The pipeline is four stages, each usable on its own:

1. :mod:`ludometer.human.client` — a rate-limited, cookie-jar HTTP client for BGA.
2. :mod:`ludometer.human.fetch` — resumable fetcher (ranking, player histories,
   replay logs) with a JSON state file, so a run can be stopped and restarted.
3. :mod:`ludometer.human.parse` — BGA replay log JSON -> :class:`ReplayGame`
   (an ordered list of picks plus the per-round factory deals).
4. :mod:`ludometer.human.convert` — :class:`ReplayGame` -> engine actions,
   **replayed in our own engine** with strict validation, then
   :mod:`ludometer.human.dataset` writes a ``replay.npz`` for ``--pretrain``.

Nothing here imports torch, and nothing here touches the engine's source: the
converter scripts the chance events by writing ``factories``/``bag``/``lid`` on a
normal :class:`~ludometer.azul.engine.AzulState` and calling ``recount()``.
"""

from __future__ import annotations

__all__ = [
    "AZUL_GAME_ID",
    "BgaClient",
    "ClientConfig",
    "ConversionError",
    "Fetcher",
    "ReplayGame",
    "convert_game",
    "parse_log",
]

from ludometer.human.client import AZUL_GAME_ID, BgaClient, ClientConfig
from ludometer.human.convert import ConversionError, convert_game
from ludometer.human.fetch import Fetcher
from ludometer.human.parse import ReplayGame, parse_log
