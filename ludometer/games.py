"""Game registry: a name -> its rules engine, its sizes and its baselines.

Everything above the engine (MCTS, self-play, replay, arena, trainer) is
duck-typed on a state object, so a second game only has to say how big its
encoding and action space are and which hand-written agents anchor its Elo
scale. ``configs/*.json`` picks one with ``"game": "azul" | "uno"``; the key is
absent from every run1-run6 config and defaults to Azul, so nothing that
existed before this module changes.

Baseline agent specs are game-qualified strings (``"uno:greedy"``), which keeps
:func:`ludometer.agents.make_agent` a one-argument function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ludometer.azul import engine as _azul
from ludometer.uno import engine as _uno
from ludometer.uno import plus as _unoplus

__all__ = ["DEFAULT_GAME", "GAMES", "GameSpec", "get_game"]

DEFAULT_GAME = "azul"


@dataclass(frozen=True)
class GameSpec:
    name: str
    encoded_size: int
    action_space: int
    baselines: tuple[str, ...]
    #: hard ceiling on moves in one game (self-play and arena backstop)
    max_moves: int

    state_cls: Any
    #: extra keyword arguments handed to ``state_cls.new_game``
    options: tuple[tuple[str, Any], ...] = ()

    def new_game(self, seed: int) -> Any:
        return self.state_cls.new_game(seed=seed, **dict(self.options))


GAMES: dict[str, GameSpec] = {
    "azul": GameSpec(
        name="azul",
        encoded_size=_azul.ENCODED_SIZE,
        action_space=_azul.ACTION_SPACE,
        baselines=("random", "greedy", "heuristic"),
        max_moves=400,
        state_cls=_azul.AzulState,
    ),
    "uno": GameSpec(
        name="uno",
        encoded_size=_uno.ENCODED_SIZE,
        action_space=_uno.ACTION_SPACE,
        baselines=("uno:random", "uno:greedy", "uno:heuristic"),
        max_moves=5000,  # 300 random matches peaked at 2,082 moves
        state_cls=_uno.UnoState,
    ),
    # One hand: the training episode. Same engine, same net, but the game ends
    # when somebody goes out, so the label the search backs up is the episode's
    # own result (see UnoState._end_hand).
    "uno_hand": GameSpec(
        name="uno_hand",
        encoded_size=_uno.ENCODED_SIZE,
        action_space=_uno.ACTION_SPACE,
        baselines=("uno:random", "uno:greedy", "uno:heuristic"),
        max_moves=400,
        state_cls=_uno.UnoState,
        options=(("hand_limit", 1),),
    ),
    # Uno+ (uno/plus.py): the rule-knob experiment — same deck and actions,
    # but draw-always-legal, +2/+4 stacking, the 7-swap and a 9-card deal.
    "unoplus": GameSpec(
        name="unoplus",
        encoded_size=_unoplus.PLUS_ENCODED_SIZE,
        action_space=_unoplus.UnoPlusState.ACTION_SPACE,
        baselines=("unoplus:random", "unoplus:greedy", "unoplus:heuristic"),
        max_moves=8000,  # voluntary draws stretch a match; measured backstop
        state_cls=_unoplus.UnoPlusState,
    ),
    "unoplus_hand": GameSpec(
        name="unoplus_hand",
        encoded_size=_unoplus.PLUS_ENCODED_SIZE,
        action_space=_unoplus.UnoPlusState.ACTION_SPACE,
        baselines=("unoplus:random", "unoplus:greedy", "unoplus:heuristic"),
        max_moves=2500,  # early near-random nets bank cards; measured p50 ~1,059
        state_cls=_unoplus.UnoPlusState,
        options=(("hand_limit", 1),),
    ),
}


def get_game(name: str | None) -> GameSpec:
    """The spec for ``name`` (``None``/empty means Azul)."""
    key = (name or DEFAULT_GAME).lower()
    try:
        return GAMES[key]
    except KeyError:
        raise ValueError(f"unknown game {name!r}; known: {sorted(GAMES)}") from None
