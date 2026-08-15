/* One human-vs-AI game, held in the page — a port of ludometer/gui/session.py.
 *
 * The Python GUI kept this object on a Flask server and shipped snapshots over
 * /api/state. Here it lives in the tab: same state, same log, same round-end
 * reports, no network. The only real difference is that the AI's move is awaited
 * from a Web Worker instead of an HTTP request, which is why `aiReply` takes
 * callbacks rather than returning everything at once.
 *
 * Turn flow: the human plays one action, then the AI replies. Players alternate
 * *within* a round, but a round boundary is resolved inside `apply`, and the next
 * round is started by whoever holds the first-player marker — which can be the AI
 * twice in a row. Hence the AI reply is a loop, not a single move.
 */

import { AzulState, Rng, ACTION_SPACE } from "./engine.js";
import { describeAction, finalReport, roundReport } from "./report.js";

const MAX_AI_REPLIES = 12; // safety net; 2 is already unusual

export class IllegalMove extends Error {}

export class GameSession {
  /**
   * @param {object} opts
   * @param {number} opts.seed
   * @param {boolean} opts.humanPlaysFirst
   * @param {string} opts.agentName
   * @param {object} opts.opponentInfo  {checkpoint, elo, run, ...} for the blurb
   * @param {number} opts.thinkTimeS
   * @param {(state) => Promise<number>} opts.think  asks the AI for one action
   */
  constructor(opts) {
    this.seed = opts.seed >>> 0;
    this.agentName = opts.agentName || "the net";
    this.opponentInfo = opts.opponentInfo || {};
    this.thinkTimeS = Number(opts.thinkTimeS) || 0;
    this.think = opts.think;
    this.humanPlaysFirst = opts.humanPlaysFirst !== false;
    this.humanSeat = this.humanPlaysFirst ? 0 : 1;
    this.aiSeat = 1 - this.humanSeat;
    this.state = AzulState.newGame(this.seed, new Rng(this.seed));
    this.ply = 0;
    this.log = [];
    this.roundReports = [];
    this.lastAiMoves = [];
    this._logEntry(
      "start",
      `New game — you are player ${this.humanSeat + 1}, the AI (${this.agentName}) is player ${this.aiSeat + 1}. ` +
        `${this.humanPlaysFirst ? "You" : "The AI"} start${this.humanPlaysFirst ? "" : "s"}.`
    );
    if (this.opponentBlurb) this._logEntry("start", this.opponentBlurb);
  }

  /* -------------------------------------------------------------- helpers */
  _logEntry(kind, text, extra = {}) {
    const entry = { n: this.log.length, kind, text, ...extra };
    this.log.push(entry);
    return entry;
  }

  /** One line naming the checkpoint behind the opponent. */
  get opponentBlurb() {
    const info = this.opponentInfo;
    if (!info || !info.checkpoint) return "";
    const elo = typeof info.elo === "number" ? `, rated ${info.elo >= 0 ? "+" : ""}${Math.round(info.elo)} on our internal ladder` : "";
    const thinking = this.thinkTimeS ? ` It thinks for ${this.thinkTimeS}s per move, in this tab.` : " It replies from the policy head, with no search.";
    return `You're facing ${info.checkpoint}${elo}.${thinking}`;
  }

  sideOf(player) {
    return player === this.humanSeat ? "human" : "ai";
  }

  labelOf(player) {
    return player === this.humanSeat ? "You" : "AI";
  }

  get humanTurn() {
    return !this.state.isTerminal && this.state.currentPlayer === this.humanSeat;
  }

  get aiTurn() {
    return !this.state.isTerminal && this.state.currentPlayer === this.aiSeat;
  }

  legalForHuman() {
    return this.humanTurn ? this.state.legalActions() : [];
  }

  /* ---------------------------------------------------------------- moves */
  _apply(actionId) {
    const state = this.state;
    const player = state.currentPlayer;
    if (!state.isLegal(actionId)) throw new IllegalMove(`action ${actionId} is not legal right now`);
    const move = describeAction(state, actionId);
    move.side = this.sideOf(player);
    move.label = this.labelOf(player);
    const before = state.clone();
    const roundBefore = state.roundIndex;
    state.apply(actionId);
    this.ply += 1;
    move.ply = this.ply;
    const entry = this._logEntry("move", `${move.label} ${move.text}`, {
      side: move.side,
      player,
      action_id: actionId,
      ply: this.ply,
    });
    move.log_n = entry.n;

    move.ended_round = state.roundIndex !== roundBefore || state.isTerminal;
    if (move.ended_round) {
      const report = roundReport(before, move);
      report.game_over = state.isTerminal;
      report.next_first_player = state.firstPlayer;
      report.scores_after = state.scores.slice();
      report.labels = [this.labelOf(0), this.labelOf(1)];
      report.sides = [this.sideOf(0), this.sideOf(1)];
      this.roundReports.push(report);
      const you = report.players[this.humanSeat];
      const ai = report.players[this.aiSeat];
      const sign = (n) => (n >= 0 ? `+${n}` : String(n));
      this._logEntry(
        "round",
        `End of round ${report.round + 1}: you ${sign(you.delta)} → ${you.score_after}, AI ${sign(ai.delta)} → ${ai.score_after}.`,
        { round: report.round, report_n: this.roundReports.length - 1 }
      );
      if (state.isTerminal) {
        const final = finalReport(state, this.humanSeat) || {};
        this._logEntry(
          "end",
          `${final.headline || "Game over."} Final score ${state.scores[this.humanSeat]}–${state.scores[this.aiSeat]}.`
        );
      }
    }
    return move;
  }

  /** One AI move, with what the search cost attached. */
  async _aiMove(onThinking) {
    const { action, search } = await this.think(this.state, onThinking);
    const move = this._apply(action);
    if (search && search.sims) {
      move.search = search;
      move.search_text = `searched ${search.sims.toLocaleString()} positions in ${(search.elapsedS || 0).toFixed(1)}s`;
      this._logEntry("think", `AI ${move.search_text}.`, { ply: move.ply });
    }
    return move;
  }

  /** Let the AI move until it is the human's turn again (or the game ends). */
  async aiReplies(onThinking) {
    const moves = [];
    for (let i = 0; i < MAX_AI_REPLIES; i++) {
      if (!this.aiTurn) break;
      moves.push(await this._aiMove(onThinking));
    }
    this.lastAiMoves = moves;
    return moves;
  }

  /** Apply the human's action. The caller then asks for `aiReplies`. */
  playHuman(actionId) {
    if (this.state.isTerminal) throw new IllegalMove("the game is over — start a new one");
    if (this.state.currentPlayer !== this.humanSeat) throw new IllegalMove("it is not your turn");
    if (!Number.isInteger(actionId) || actionId < 0 || actionId >= ACTION_SPACE) {
      throw new IllegalMove(`action id must be an integer in 0..179, got ${actionId}`);
    }
    const first = this.roundReports.length;
    const move = this._apply(actionId);
    return { move, reports: this.roundReports.slice(first) };
  }

  /** Reports created since index `first` — the overlays still to be shown. */
  reportsSince(first) {
    return this.roundReports.slice(first);
  }

  /* ------------------------------------------------------------- snapshot */
  snapshot() {
    return {
      state: this.state.toJSON(),
      seed: this.seed,
      agent_name: this.agentName,
      opponent_info: this.opponentInfo,
      opponent_blurb: this.opponentBlurb || null,
      human_seat: this.humanSeat,
      ai_seat: this.aiSeat,
      your_turn: this.humanTurn,
      ai_pending: this.aiTurn,
      think_time_s: this.thinkTimeS,
      human_legal_actions: this.legalForHuman(),
      ply: this.ply,
      last_ai_move: this.lastAiMoves.length ? this.lastAiMoves[this.lastAiMoves.length - 1] : null,
      log: this.log,
      final: finalReport(this.state, this.humanSeat),
    };
  }
}
