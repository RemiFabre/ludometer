/* Sharing games: the page's side of the research bargain.
 *
 * Faïence is a research project, and the games people play are the research
 * material. So when a game ends (and when a game is abandoned mid-way), this
 * module sends the record that js/record.js builds to a small collector
 * (web/ingest/, a separate Space) which replays it in the real engine and
 * commits it to the public dataset. The About panel says all of this in the
 * page, and Settings holds the switch: sharing is ON by default, visibly, and
 * turning it off is respected everywhere in this file.
 *
 * Two promises this module keeps:
 *
 *   1. **A game never depends on the upload.** Everything here is fire and
 *      forget: sendBeacon on the way out of the page, a fetch whose failure
 *      is swallowed everywhere else, and no caller ever awaits either. If
 *      the collector is asleep, blocked or gone, the game plays on and the
 *      record waits in localStorage for the visitor's next visit.
 *   2. **The record carries no identity.** It is the moves, the deals, the
 *      net and the score, exactly what the Save button offers as a file. The
 *      collector stores nothing else (it never logs IPs or user agents), and
 *      everything it stores is public, so a player can read the whole take.
 *
 * The dev guard mirrors analytics.js: localhost plays thousands of games in
 * the test suites and none of them are players, so none of them may send.
 */

export const INGEST_BASE = "https://remifabre-faience-ingest.hf.space";
export const DATASET_URL = "https://huggingface.co/datasets/RemiFabre/faience-games";

const SHARE_KEY = "faience.share";
const OUTBOX_KEY = "faience.outbox";
const OUTBOX_MAX = 8; // a courtesy buffer, not an archive: oldest games drop first

/** Whether sharing is on. Anything but a stored "off" means yes: the default. */
export function sharingOn() {
  try {
    return localStorage.getItem(SHARE_KEY) !== "off";
  } catch (err) {
    return true;
  }
}

export function setSharing(on) {
  try {
    localStorage.setItem(SHARE_KEY, on ? "on" : "off");
  } catch (err) {
    /* private mode; the choice just will not survive a reload */
  }
}

/** True while the page is somewhere no data should ever be sent from:
 * local development, the test suites, and any staging copy of the site. */
function devHost() {
  const host = location.hostname;
  return host === "localhost" || host === "127.0.0.1" || host === "" || host.includes("staging");
}

function readOutbox() {
  try {
    const kept = JSON.parse(localStorage.getItem(OUTBOX_KEY) || "[]");
    return Array.isArray(kept) ? kept : [];
  } catch (err) {
    return [];
  }
}

function writeOutbox(entries) {
  try {
    localStorage.setItem(OUTBOX_KEY, JSON.stringify(entries.slice(-OUTBOX_MAX)));
  } catch (err) {
    /* full or forbidden storage loses the retry, never the game */
  }
}

/**
 * Queue `record` and try to deliver the whole outbox. The normal path for a
 * finished game: the collector's reply confirms delivery, so what fails
 * (asleep, offline, blocked) stays queued for the next call on any visit.
 */
export function shareRecord(record) {
  if (!sharingOn() || devHost() || !record || !record.moves || !record.moves.length) return;
  writeOutbox([...readOutbox(), JSON.stringify(record)]);
  flushOutbox();
}

let flushing = false;
export async function flushOutbox() {
  if (flushing || !sharingOn() || devHost()) return;
  const entries = readOutbox();
  if (!entries.length) return;
  flushing = true;
  try {
    const kept = [];
    for (const body of entries) {
      try {
        // no Content-Type header: the default text/plain keeps this a simple
        // CORS request, same as the beacon, and the collector reads the body
        // not the label. The collector answers 204 once it has the record
        // (duplicates are deduplicated there, by content).
        const reply = await fetch(INGEST_BASE + "/game", { method: "POST", body });
        if (!reply.ok) kept.push(body);
      } catch (err) {
        kept.push(body);
      }
    }
    writeOutbox(kept);
  } finally {
    flushing = false;
  }
}

/**
 * The way out of a closing page: sendBeacon is built for exactly this moment
 * and survives the navigation. Delivery cannot be confirmed, so the record is
 * also queued; if the beacon did land, the collector deduplicates the retry.
 */
export function shareOnLeave(record) {
  if (!sharingOn() || devHost() || !record || !record.moves || !record.moves.length) return;
  const body = JSON.stringify(record);
  writeOutbox([...readOutbox(), body]);
  try {
    navigator.sendBeacon(INGEST_BASE + "/game", body);
  } catch (err) {
    /* the queue already has it */
  }
}

/**
 * A page load costs the collector one /health ping, so a Space that fell
 * asleep is awake long before the first game here could finish. Fire-and-
 * forget by construction; also the moment to deliver anything left over.
 */
export function warmCollector() {
  if (!sharingOn() || devHost()) return;
  try {
    fetch(INGEST_BASE + "/health", { mode: "no-cors" }).catch(() => {});
  } catch (err) {
    /* nothing to do: the retry path covers it */
  }
  flushOutbox();
}
