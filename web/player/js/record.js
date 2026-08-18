/* A finished game, written down: the smallest thing that replays exactly.
 *
 * The point of this file is data collection, so the rule it follows is *record
 * only what cannot be recomputed*. Search targets, position encodings, coach
 * verdicts and value estimates can all be regenerated offline by replaying the
 * game at any depth, so none of them are stored. Three things cannot:
 *
 *   1. **the chance events.** The bag is shuffled by this page's own RNG
 *      (mulberry32, see engine.js), and Python's Mersenne Twister cannot
 *      reproduce it from the same seed. So the deal of every round is written
 *      down explicitly: the five factories, plus the bag and lid counts, which
 *      is what a converter needs to check that no tile appeared or vanished.
 *      The seed is kept as well, because *this* engine does reproduce a game
 *      from it, which is how the tests verify a record end to end.
 *   2. **the moves**, as action ids. The encoding is `source*30 + color*6 +
 *      dest` in this engine and in the Python one, character for character, and
 *      the engine parity test holds them to identical ordered legal actions and
 *      identical 182-float encodings. So an integer here means exactly the same
 *      move there.
 *   3. **the circumstances**: which seat the human held, which net answered,
 *      how long it was allowed to think, and how many positions it actually
 *      visited per move (that last one depends on the visitor's machine, so it
 *      is not recoverable either).
 *
 * Nothing here sends anything anywhere: this module only builds the value.
 * Sending is js/upload.js's job, governed by the sharing switch in Settings
 * (on by default) and described honestly in the About panel; the Save and
 * Copy buttons hand the very same record to the player, so what is shared
 * and what a player can read are one and the same thing.
 */

export const FORMAT = "faience-game/1";

/**
 * Build the record of `session` as it stands.
 *
 * `extra` carries what the session does not know about itself:
 * `{net, backend}` — the exported model's metadata and the runtime that ran it.
 * Safe to call at any point in a game; `final.finished` says whether the
 * position is terminal.
 */
export function buildRecord(session, extra = {}) {
  if (!session) return null;
  const state = session.state;
  const net = extra.net || {};
  const info = session.opponentInfo || {};

  const moves = session.log
    .filter((e) => e.kind === "move")
    .map((e) => {
      const move = { ply: e.ply, player: e.player, action: e.action_id };
      const think = session.log.find((t) => t.kind === "think" && t.ply === e.ply);
      if (think) {
        if (Number.isFinite(think.sims) && think.sims > 0) move.sims = think.sims;
        // the net's own read of the position it moved from, on its [-1, 1]
        // scale: the cheapest signal for "the AI did not see this coming"
        if (Number.isFinite(think.value)) move.value = round4(think.value);
      }
      return move;
    });

  return {
    format: FORMAT,
    created_at: new Date().toISOString(),
    seed: session.seed,
    human_seat: session.humanSeat,
    human_first: !!session.humanPlaysFirst,
    net: {
      run: net.run || info.run || null,
      checkpoint: net.checkpoint || info.checkpoint || null,
      elo: typeof net.elo === "number" ? net.elo : typeof info.elo === "number" ? info.elo : null,
      params: net.num_params || null,
      backend: extra.backend || null,
    },
    think_time_s: session.thinkTimeS,
    moves,
    deals: (session.deals || []).map((d) => ({
      round: d.round,
      factories: d.factories.map((f) => f.slice()),
      bag: d.bag.slice(),
      lid: d.lid.slice(),
    })),
    final: {
      finished: !!state.isTerminal,
      scores: state.scores.slice(),
      outcome: outcomeOf(session),
      rounds: state.roundIndex + 1,
      exhausted: !!state.exhausted,
    },
  };
}

function outcomeOf(session) {
  const state = session.state;
  if (!state.isTerminal) return null;
  const mine = state.scores[session.humanSeat];
  const theirs = state.scores[session.aiSeat];
  if (mine > theirs) return "human";
  if (theirs > mine) return "ai";
  return "draw";
}

function round4(v) {
  return Math.round(v * 1e4) / 1e4;
}

/** `faience-run4-ckpt-037888-seed31337-ai.json` — sortable, self-describing. */
export function recordFilename(record) {
  const bits = ["faience"];
  if (record.net.run) bits.push(record.net.run);
  if (record.net.checkpoint) bits.push(record.net.checkpoint);
  bits.push("seed" + record.seed);
  if (record.final.outcome) bits.push(record.final.outcome);
  return bits.join("-").replace(/[^A-Za-z0-9._-]/g, "") + ".json";
}

/** Indented for a file, so a human can read what they are about to send. */
export const asFile = (record) => JSON.stringify(record, null, 1);

/** One line, for pasting into an issue or a chat. */
export const asLine = (record) => JSON.stringify(record);

/**
 * Hand `record` to the player as a download. Returns true if the browser took
 * it; the caller says so in the page (this module never touches the status).
 */
export function downloadRecord(record) {
  try {
    const blob = new Blob([asFile(record)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = recordFilename(record);
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    a.remove();
    // let the click start before the URL goes away
    setTimeout(() => URL.revokeObjectURL(url), 4000);
    return true;
  } catch (err) {
    return false;
  }
}

/** Copy `record` to the clipboard as one line. Resolves true on success. */
export async function copyRecord(record) {
  const text = asLine(record);
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    // an older browser, or a page the clipboard API refuses: fall back to the
    // oldest trick there is, a hidden field and the document's own copy command
    try {
      const box = document.createElement("textarea");
      box.value = text;
      box.setAttribute("readonly", "");
      box.style.position = "fixed";
      box.style.top = "-1000px";
      document.body.appendChild(box);
      box.select();
      const ok = document.execCommand("copy");
      box.remove();
      return ok;
    } catch (err2) {
      return false;
    }
  }
}
