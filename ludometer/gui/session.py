"""One human-vs-AI game, held in server memory.

The session owns an :class:`~ludometer.azul.engine.AzulState`, the opponent agent
built from an agent spec (see :func:`ludometer.agents.registry.load_agent`), a
game log, and the round-end reports the browser needs for its overlays.

Turn flow: the human plays one action, then :meth:`play_human` lets the AI reply.
Players alternate *within* a round, but a round boundary is resolved inside
``apply``, and the next round is started by whoever holds the first-player marker
— which can be the AI twice in a row. Hence the AI reply is a loop, not a single
move.
"""

from __future__ import annotations

import random
from typing import Any

from ludometer.agents.registry import load_agent
from ludometer.azul.engine import AzulState
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
    ) -> None:
        if seed is None:
            seed = random.randrange(1 << 30)
        self.seed = int(seed)
        self.opponent_spec = opponent_spec
        # built first: a bad spec must not leave a half-initialised session
        self.agent = load_agent(opponent_spec, seed=self.seed ^ 0x5EED)
        self.agent_name = getattr(self.agent, "name", opponent_spec)
        self.human_plays_first = bool(human_plays_first)
        self.human_seat = 0 if self.human_plays_first else 1
        self.ai_seat = 1 - self.human_seat
        self.state = AzulState.new_game(seed=self.seed)
        self.ply = 0
        self.log: list[dict[str, Any]] = []
        self.round_reports: list[dict[str, Any]] = []
        self.last_ai_moves: list[dict[str, Any]] = []
        self._hint_agent = None
        self._log(
            "start",
            f"New game — you are player {self.human_seat + 1}, "
            f"the AI ({self.agent_name}) is player {self.ai_seat + 1}. "
            f"{'You' if self.human_plays_first else 'The AI'} start"
            f"{'' if self.human_plays_first else 's'}.",
        )
        # the AI opens when the human took the second seat
        self.last_ai_moves = self._ai_replies()

    # ------------------------------------------------------------------ helpers
    def _log(self, kind: str, text: str, **extra: Any) -> dict[str, Any]:
        entry = {"n": len(self.log), "kind": kind, "text": text, **extra}
        self.log.append(entry)
        return entry

    def side_of(self, player: int) -> str:
        return "human" if player == self.human_seat else "ai"

    def label_of(self, player: int) -> str:
        return "You" if player == self.human_seat else "AI"

    @property
    def human_turn(self) -> bool:
        return (
            not self.state.is_terminal and self.state.current_player == self.human_seat
        )

    def legal_for_human(self) -> list[int]:
        return self.state.legal_actions() if self.human_turn else []

    # -------------------------------------------------------------------- moves
    def _apply(self, action_id: int) -> dict[str, Any]:
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
        state.apply(action_id)
        self.ply += 1
        move["ply"] = self.ply
        entry = self._log(
            "move",
            f"{move['label']} {move['text']}",
            side=move["side"],
            player=player,
            action_id=action_id,
            ply=self.ply,
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
            )
            if state.is_terminal:
                final = final_report(state, self.human_seat) or {}
                self._log(
                    "end",
                    f"{final.get('headline', 'Game over.')} "
                    f"Final score {state.scores[self.human_seat]}"
                    f"–{state.scores[self.ai_seat]}.",
                )
        return move

    def _ai_replies(self) -> list[dict[str, Any]]:
        """Let the AI move until it is the human's turn again (or the game ends)."""
        moves: list[dict[str, Any]] = []
        state = self.state
        for _ in range(MAX_AI_REPLIES):
            if state.is_terminal or state.current_player != self.ai_seat:
                break
            action_id = int(self.agent.act(state))
            moves.append(self._apply(action_id))
        return moves

    def play_human(self, action_id: int) -> dict[str, Any]:
        """Apply the human's action, then the AI's reply (possibly a couple)."""
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
        human_move = self._apply(action_id)
        self.last_ai_moves = self._ai_replies()
        payload = self.snapshot()
        payload["human_move"] = human_move
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
            "human_seat": self.human_seat,
            "ai_seat": self.ai_seat,
            "human_plays_first": self.human_plays_first,
            "your_turn": self.human_turn,
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
