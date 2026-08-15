"""Coach mode: score a human move with the AI's own evaluation.

The rating is not a metric of our own invention. Before your move is applied,
the *same* PUCT search the opponent plays with — same checkpoint, same network,
same ``c_puct``/FPU config — is run on your position, and the root's edge
statistics are read out:

    delta = Q(root child = the move you played) - max over explored children Q

``Q`` is the search's own value estimate in the root player's frame (that is
yours), on the network's [-1, 1] scale where +1 is a won game. So ``0.00`` means
you played the move the AI would have played, and ``-0.06`` means the search
values your move six hundredths of a win worse than its own choice.

Only *explored* children can be compared: an edge the search never visited has
no ``Q`` at all, so a move that got no visits is reported as **unrated** rather
than given a made-up number.

Two implementation notes:

* :class:`RootStatsMCTS` is a **subclass**, not an edit. ``ludometer/train/mcts.py``
  is imported live by the trainer's workers, so it must not grow a GUI-only
  accessor; everything coach mode needs is exposed here instead.
* the coach runs its own :class:`MCTS` instance over the opponent's *evaluator*
  (the loaded net), so it cannot disturb the opponent's tree, RNG or noise
  state. Dirichlet noise is off: a rating must be reproducible, not exploratory.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ludometer.azul.engine import AzulState
from ludometer.gui.moves import describe_action
from ludometer.train.mcts import MCTS

__all__ = [
    "COACH_MAX_THINK_S",
    "COACH_SIMS_CAP",
    "COACH_THINK_S",
    "CoachUnavailable",
    "MoveCoach",
    "RootStatsMCTS",
    "coach_time_for",
]

# The coach's own clock, in seconds. It runs *before* your move is committed, so
# it is deliberately shorter than the opponent's budget can be: a 10 s opponent
# would otherwise double every turn you take.
COACH_THINK_S = 2.0
COACH_MAX_THINK_S = 3.0
# With a time budget the sim count is only a ceiling; keep it out of the way.
COACH_SIMS_CAP = 20_000

# Below this the move is "what the AI would have played" — a hundredth of a win
# is inside the noise of two searches of the same position.
BEST_EPSILON = 0.005
# At or beyond this, the log also names the move the search preferred.
SHOW_BEST_AT = -0.02


class CoachUnavailable(RuntimeError):
    """Raised when the opponent has no search to borrow (baselines, no torch)."""


class RootStatsMCTS(MCTS):
    """:class:`~ludometer.train.mcts.MCTS` that reports its root's edge stats.

    The base class already keeps ``(N, W, P)`` per root child for its own PUCT
    descent; this only reads them back out after :meth:`MCTS.search`.
    """

    def root_children(self) -> list[dict[str, Any]]:
        """One entry per legal action at the last search's root.

        ``q`` is ``None`` for an edge the search never visited — the caller must
        not invent a value for it (see the module docstring).
        """
        root = getattr(self, "_root", None)
        if root is None or not root.expanded:
            return []
        out: list[dict[str, Any]] = []
        for i, action in enumerate(root.legal):
            visits = root.visits[i]
            out.append(
                {
                    "action": int(action),
                    "visits": int(visits),
                    "q": (root.wins[i] / visits) if visits else None,
                    "prior": float(root.priors[i]) if root.priors else 0.0,
                }
            )
        return out

    @property
    def root_player(self) -> int | None:
        root = getattr(self, "_root", None)
        return None if root is None else int(root.player)


def coach_time_for(think_time_s: float | None) -> float:
    """The coach's budget for an opponent that thinks for ``think_time_s``."""
    budget = float(think_time_s or 0.0) or COACH_THINK_S
    return min(budget, COACH_MAX_THINK_S)


class MoveCoach:
    """Rates moves with the search behind ``agent`` (see the module docstring)."""

    def __init__(self, agent: Any, time_budget_s: float | None = None, seed: int = 0):
        evaluator = getattr(agent, "evaluator", None)
        opponent = getattr(agent, "mcts", None)
        if evaluator is None or opponent is None:
            raise CoachUnavailable(
                "coach mode needs a searching opponent: deal against a trained "
                "checkpoint ('Strongest trained' or an mcts: spec)"
            )
        self.agent_name = getattr(agent, "name", "the AI")
        self.time_budget_s = float(time_budget_s or COACH_THINK_S)
        config = replace(
            opponent.config,
            tree_reuse=False,  # every rating is a fresh look at your position
            sims=max(opponent.config.sims, COACH_SIMS_CAP),
        )
        self.search = RootStatsMCTS(evaluator, config, seed=seed, add_noise=False)

    # ------------------------------------------------------------------ rating
    def analyse(self, state: AzulState) -> dict[str, Any]:
        """One rating search over ``state`` (never mutated); no move is picked.

        Returns the root table: every explored child with its visit count and
        ``Q``, plus the best of them. :meth:`verdict` turns that into a rating
        for a particular move, so one search can score several candidates —
        which is also how the tests compare the AI's choice with a worse one.
        """
        legal = state.legal_actions()
        analysis: dict[str, Any] = {
            "budget_s": self.time_budget_s,
            "legal": len(legal),
            "sims": 0,
            "elapsed_s": 0.0,
            "children": [],
            "best": None,
            "forced": len(legal) == 1,
        }
        if len(legal) <= 1:
            return analysis
        result = self.search.search(
            state, add_noise=False, time_limit_s=self.time_budget_s
        )
        explored = [
            c for c in self.search.root_children() if c["visits"] and c["q"] is not None
        ]
        analysis["sims"] = int(result.sims)
        analysis["elapsed_s"] = float(result.elapsed_s)
        analysis["children"] = explored
        analysis["explored"] = len(explored)
        if explored:
            best = max(explored, key=lambda c: c["q"])
            analysis["best"] = best
            analysis["best_text"] = describe_action(state, best["action"])["text"]
        return analysis

    @staticmethod
    def verdict(analysis: dict[str, Any], action: int) -> dict[str, Any]:
        """Score one move against an :meth:`analyse` table. Runs no search."""
        base = {
            key: analysis[key]
            for key in ("sims", "elapsed_s", "budget_s", "legal")
            if key in analysis
        }
        base["explored"] = len(analysis.get("children") or [])
        if analysis.get("forced"):
            base.update(rated=True, forced=True, delta=0.0, visits=0)
            return base
        best = analysis.get("best")
        if best is None:  # pragma: no cover - only with a sub-millisecond budget
            base.update(
                rated=False,
                unrated=True,
                reason="the search had no time to explore this position",
            )
            return base
        base.update(
            best_action=best["action"],
            best_visits=best["visits"],
            best_text=analysis.get("best_text"),
        )
        mine = next(
            (c for c in analysis["children"] if c["action"] == int(action)), None
        )
        if mine is None:
            base.update(
                rated=False,
                unrated=True,
                reason="the search never explored this move",
            )
            return base
        # Q is already in the root player's frame, so this can only be <= 0;
        # clamp the float noise away rather than showing "+0.00".
        delta = min(0.0, float(mine["q"]) - float(best["q"]))
        base.update(
            rated=True,
            delta=delta,
            your_q=float(mine["q"]),
            best_q=float(best["q"]),
            visits=mine["visits"],
            matched_best=delta > -BEST_EPSILON,
            show_best=delta <= SHOW_BEST_AT,
        )
        return base

    def rate(self, state: AzulState, action: int) -> dict[str, Any]:
        """Score ``action`` in ``state`` (never mutated) for the player to move."""
        if action not in state.legal_actions():
            return {"rated": False, "unrated": True, "reason": "that move is not legal"}
        return self.verdict(self.analyse(state), action)

    @staticmethod
    def summarise(rating: dict[str, Any]) -> str:
        """One line for the log, e.g. ``-0.06 (AI preferred: took 2 blue ...)``."""
        if not rating:
            return ""
        if rating.get("unrated"):
            return "unrated — " + str(rating.get("reason", "not explored"))
        if rating.get("forced"):
            return "0.00 (only move)"
        text = f"{rating['delta']:+.2f}".replace("+0.00", "0.00")
        if rating.get("show_best") and rating.get("best_text"):
            text += f" (AI preferred: {rating['best_text']})"
        return text
