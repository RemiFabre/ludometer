"""One human-vs-AI game, held in server memory.

The session owns an :class:`~ludometer.azul.engine.AzulState`, the opponent agent
built from an agent spec (see :func:`ludometer.agents.registry.load_agent`), a
game log, and the round-end reports the browser draws inline under the boards.

Turn flow: the human plays one action, then :meth:`play_human` lets the AI reply.
Players alternate *within* a round, but a round boundary is resolved inside
``apply``, and the next round is started by whoever holds the first-player marker
— which can be the AI twice in a row. Hence the AI reply is a loop, not a single
move.

``play_human(action_id, defer_ai=True)`` stops after the human's move and leaves
``ai_pending`` set; the caller then asks for :meth:`ai_reply` separately. The
page uses that split so the human's own move appears immediately, then the AI
visibly thinks for its time budget instead of the board jumping two moves at
once.

``think_time_s`` is that budget: a neural opponent searches by the clock rather
than by a sim count, and each AI move carries a ``search`` block saying how many
positions it actually visited.

Every move the session reports carries ``state_before``, the position it was
played from. That is what lets the page animate *both* of the AI's moves when it
moves twice across a round boundary: the second one starts from a refilled table
that ``apply`` produced inside the first, and which the page would otherwise
never see. It also gives the page a free move history — position ``k`` is the
``state_before`` of ply ``k + 1`` — which is what the ← / → move navigator
replays, with no request and no re-search.

``play_human(..., coach=True)`` turns on **coach mode**: before the move is
applied, the opponent's own search rates it (see :mod:`ludometer.gui.coach`) and
the verdict is attached to the move's log entry as ``coach``. The rating happens
on the ``/api/act`` request, i.e. while the page is showing "rating your move…",
because it must see the position *before* the move is committed.
"""

from __future__ import annotations

import random
from typing import Any

from ludometer.agents.registry import load_agent
from ludometer.azul.engine import AzulState
from ludometer.gui.coach import CoachUnavailable, MoveCoach, coach_time_for
from ludometer.gui.moves import describe_action, final_report, round_report

__all__ = ["GameSession", "IllegalMove"]

MAX_AI_REPLIES = 12  # safety net; 2 is already unusual
HINT_SPEC = "heuristic"


class IllegalMove(ValueError):
    """Raised for an action the engine would reject (mapped to HTTP 400)."""


class GameSession:
    """A single game: state + opponent + narration."""

    def __init__(
        self,
        opponent_spec: str = "heuristic",
        human_plays_first: bool = True,
        seed: int | None = None,
        think_time_s: float | None = None,
    ) -> None:
        if seed is None:
            seed = random.randrange(1 << 30)
        self.seed = int(seed)
        self.opponent_spec = opponent_spec
        # built first: a bad spec must not leave a half-initialised session
        self.agent = load_agent(opponent_spec, seed=self.seed ^ 0x5EED)
        self.agent_name = getattr(self.agent, "name", opponent_spec)
        # a search budget only means something to a searching agent
        self.think_time_s = float(think_time_s or 0.0)
        budget = getattr(self.agent, "set_time_budget", None)
        if callable(budget):
            budget(self.think_time_s or None)
        elif self.think_time_s:
            self.think_time_s = 0.0
        # for "best"/"mcts:" specs: which checkpoint the spec resolved to
        self.opponent_info: dict[str, Any] = dict(
            getattr(self.agent, "spec_info", None) or {}
        )
        self.human_plays_first = bool(human_plays_first)
        self.human_seat = 0 if self.human_plays_first else 1
        self.ai_seat = 1 - self.human_seat
        self.state = AzulState.new_game(seed=self.seed)
        self.ply = 0
        self.log: list[dict[str, Any]] = []
        self.round_reports: list[dict[str, Any]] = []
        self.last_ai_moves: list[dict[str, Any]] = []
        self._hint_agent = None
        # coach mode: built on first use, over the opponent's own net
        self.coach_time_s = coach_time_for(self.think_time_s)
        self._coach: MoveCoach | None = None
        self.last_coach: dict[str, Any] | None = None
        self._log(
            "start",
            f"New game — you are player {self.human_seat + 1}, "
            f"the AI ({self.agent_name}) is player {self.ai_seat + 1}. "
            f"{'You' if self.human_plays_first else 'The AI'} start"
            f"{'' if self.human_plays_first else 's'}.",
        )
        if self.opponent_blurb:
            self._log("start", self.opponent_blurb)
        # the AI opens when the human took the second seat
        self.last_ai_moves = self._ai_replies()

    # ------------------------------------------------------------------ helpers
    def _log(self, kind: str, text: str, **extra: Any) -> dict[str, Any]:
        entry = {"n": len(self.log), "kind": kind, "text": text, **extra}
        self.log.append(entry)
        return entry

    @property
    def opponent_blurb(self) -> str:
        """One line naming the checkpoint behind a ``best``/``mcts:`` opponent."""
        info = self.opponent_info
        ckpt = info.get("checkpoint")
        if not ckpt:
            return ""
        elo = info.get("elo")
        rated = f", rated {elo:+.0f} on our internal ladder" if elo is not None else ""
        sims = info.get("sims")
        if self.think_time_s:
            thinking = f" It thinks for {self.think_time_s:g}s per move."
        elif sims:
            thinking = f" It searches {sims} positions per move."
        else:
            thinking = ""
        return f"You're facing {ckpt}{rated}.{thinking}"

    def side_of(self, player: int) -> str:
        return "human" if player == self.human_seat else "ai"

    def label_of(self, player: int) -> str:
        return "You" if player == self.human_seat else "AI"

    @property
    def human_turn(self) -> bool:
        return (
            not self.state.is_terminal and self.state.current_player == self.human_seat
        )

    @property
    def ai_turn(self) -> bool:
        return not self.state.is_terminal and self.state.current_player == self.ai_seat

    def legal_for_human(self) -> list[int]:
        return self.state.legal_actions() if self.human_turn else []

    # -------------------------------------------------------------------- coach
    @property
    def coach_available(self) -> bool:
        """Whether the opponent has a search worth borrowing for a rating."""
        return getattr(self.agent, "mcts", None) is not None and (
            getattr(self.agent, "evaluator", None) is not None
        )

    def coach(self) -> MoveCoach:
        """The rater, built once, over the opponent's own evaluator."""
        if self._coach is None:
            self._coach = MoveCoach(
                self.agent, time_budget_s=self.coach_time_s, seed=self.seed ^ 0xC0AC
            )
        return self._coach

    def rate_move(self, action_id: int) -> dict[str, Any]:
        """Coach mode's verdict on ``action_id`` in the current position.

        Never raises: a coach that cannot run returns an ``unrated`` verdict, so
        turning the toggle on can never cost you a move.
        """
        try:
            rating = self.coach().rate(self.state, int(action_id))
        except CoachUnavailable as exc:
            return {"rated": False, "unrated": True, "reason": str(exc)}
        except Exception as exc:  # noqa: BLE001 - a rating is never worth a 500
            return {
                "rated": False,
                "unrated": True,
                "reason": f"the search failed: {type(exc).__name__}: {exc}",
            }
        self.last_coach = rating
        return rating

    # -------------------------------------------------------------------- moves
    def _apply(
        self, action_id: int, coach: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Apply one legal action and describe it; append any round-end report."""
        state = self.state
        player = state.current_player
        if not state.is_legal(action_id):
            raise IllegalMove(f"action {action_id} is not legal right now")
        move = describe_action(state, action_id)
        move["side"] = self.side_of(player)
        move["label"] = self.label_of(player)
        before = state.clone()
        round_before = state.round_index
        # The position this move was played from. The page animates from it: when
        # the AI moves twice across a round boundary its second move starts from a
        # refilled table nobody ever saw, and without this the page could only
        # animate the first of the two.
        move["state_before"] = before.to_json()
        state.apply(action_id)
        self.ply += 1
        move["ply"] = self.ply
        extra: dict[str, Any] = {}
        if coach is not None:
            # the page draws this next to the entry, so the verdict travels with
            # the move it is about rather than in a separate stream
            extra["coach"] = coach
            move["coach"] = coach
        entry = self._log(
            "move",
            f"{move['label']} {move['text']}",
            side=move["side"],
            player=player,
            action_id=action_id,
            ply=self.ply,
            # what the log draws as little tiles instead of colour words
            color=move["color"],
            count=move["count"],
            source=move["source"],
            dest=move["dest"],
            overflow=move["overflow"],
            took_marker=move["took_marker"],
            **extra,
        )
        move["log_n"] = entry["n"]

        move["ended_round"] = state.round_index != round_before or state.is_terminal
        if move["ended_round"]:
            report = round_report(before, move)
            report["game_over"] = state.is_terminal
            report["next_first_player"] = state.first_player
            report["scores_after"] = state.scores[:]
            report["labels"] = [self.label_of(p) for p in range(state.num_players)]
            report["sides"] = [self.side_of(p) for p in range(state.num_players)]
            self.round_reports.append(report)
            you, ai = (
                report["players"][self.human_seat],
                report["players"][self.ai_seat],
            )
            self._log(
                "round",
                f"End of round {report['round'] + 1}: "
                f"you {you['delta']:+d} → {you['score_after']}, "
                f"AI {ai['delta']:+d} → {ai['score_after']}.",
                round=report["round"],
                report_n=len(self.round_reports) - 1,
                # every entry carries the ply it belongs to, so the move navigator
                # can show the log as it stood at any point in the game
                ply=self.ply,
            )
            if state.is_terminal:
                final = final_report(state, self.human_seat) or {}
                self._log(
                    "end",
                    f"{final.get('headline', 'Game over.')} "
                    f"Final score {state.scores[self.human_seat]}"
                    f"–{state.scores[self.ai_seat]}.",
                    ply=self.ply,
                )
        return move

    def _ai_move(self) -> dict[str, Any]:
        """One AI move, with what the search cost attached."""
        action_id = int(self.agent.act(self.state))
        move = self._apply(action_id)
        stats = dict(getattr(self.agent, "last_search", None) or {})
        if stats.get("sims"):
            move["search"] = stats
            move["search_text"] = (
                f"searched {stats['sims']:,} positions "
                f"in {stats.get('elapsed_s', 0.0):.1f}s"
            )
            self._log("think", f"AI {move['search_text']}.", ply=move["ply"])
        return move

    def _ai_replies(self) -> list[dict[str, Any]]:
        """Let the AI move until it is the human's turn again (or the game ends)."""
        moves: list[dict[str, Any]] = []
        for _ in range(MAX_AI_REPLIES):
            if not self.ai_turn:
                break
            moves.append(self._ai_move())
        return moves

    def ai_reply(self) -> dict[str, Any]:
        """Compute the AI's pending reply (the page asks for this separately)."""
        if self.state.is_terminal:
            raise IllegalMove("the game is over — start a new one")
        if not self.ai_turn:
            raise IllegalMove("it is not the AI's turn")
        first_report = len(self.round_reports)
        self.last_ai_moves = self._ai_replies()
        payload = self.snapshot()
        payload["ai_moves"] = self.last_ai_moves
        payload["round_reports"] = self.round_reports[first_report:]
        return payload

    def play_human(
        self, action_id: int, defer_ai: bool = False, coach: bool = False
    ) -> dict[str, Any]:
        """Apply the human's action, then the AI's reply (possibly a couple).

        With ``defer_ai`` the AI is left to move: the payload's ``ai_pending``
        tells the caller to ask for :meth:`ai_reply` next.

        With ``coach`` the move is rated *before* it is applied — the search has
        to see the position you actually chose from — which is what makes this
        request the slow one when coach mode is on.
        """
        if self.state.is_terminal:
            raise IllegalMove("the game is over — start a new one")
        if self.state.current_player != self.human_seat:
            raise IllegalMove("it is not your turn")
        if (
            not isinstance(action_id, int)
            or not 0 <= action_id < AzulState.ACTION_SPACE
        ):
            raise IllegalMove(
                f"action id must be an integer in 0..179, got {action_id!r}"
            )
        first_report = len(self.round_reports)
        if not self.state.is_legal(action_id):
            raise IllegalMove(f"action {action_id} is not legal right now")
        rating = self.rate_move(action_id) if coach else None
        human_move = self._apply(action_id, coach=rating)
        self.last_ai_moves = [] if defer_ai else self._ai_replies()
        payload = self.snapshot()
        payload["human_move"] = human_move
        payload["coach"] = rating
        payload["ai_moves"] = self.last_ai_moves
        # reports created by this request (human move and/or AI replies)
        payload["round_reports"] = self.round_reports[first_report:]
        return payload

    def hint(self) -> dict[str, Any]:
        """The heuristic agent's suggestion for the human's current turn."""
        if self.state.is_terminal:
            raise IllegalMove("the game is over — nothing to suggest")
        if self.state.current_player != self.human_seat:
            raise IllegalMove("it is not your turn")
        if self._hint_agent is None:
            self._hint_agent = load_agent(HINT_SPEC, seed=self.seed)
        action_id = int(self._hint_agent.act(self.state))
        move = describe_action(self.state, action_id)
        move["text"] = f"Try: {move['text']}"
        return {"action_id": action_id, "move": move, "text": move["text"]}

    # ----------------------------------------------------------------- snapshot
    def snapshot(self) -> dict[str, Any]:
        """Everything the page needs to draw itself."""
        state = self.state
        return {
            "state": state.to_json(),
            "seed": self.seed,
            "opponent_spec": self.opponent_spec,
            "agent_name": self.agent_name,
            "opponent_info": self.opponent_info or None,
            "opponent_blurb": self.opponent_blurb or None,
            "human_seat": self.human_seat,
            "ai_seat": self.ai_seat,
            "human_plays_first": self.human_plays_first,
            "your_turn": self.human_turn,
            "ai_pending": self.ai_turn,
            "think_time_s": self.think_time_s,
            "coach_available": self.coach_available,
            "coach_time_s": self.coach_time_s,
            "last_coach": self.last_coach,
            "legal_actions": state.legal_actions(),
            "human_legal_actions": self.legal_for_human(),
            "ply": self.ply,
            "last_ai_move": self.last_ai_moves[-1] if self.last_ai_moves else None,
            "last_ai_moves": self.last_ai_moves,
            "log": self.log,
            "last_round_report": self.round_reports[-1] if self.round_reports else None,
            "final": final_report(state, self.human_seat),
            "render_text": state.render_text(),
        }
