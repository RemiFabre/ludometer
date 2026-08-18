/* The Faïence game collector: a mailbox with a rules lawyer at the slot.
 *
 * The playing page (a static Space, see web/player/) sends every finished
 * game here as a small JSON record, via navigator.sendBeacon, unless the
 * player switched sharing off. This process:
 *
 *   1. replays each submission in the real engine (verify.js) and keeps only
 *      games that reproduce their own recorded deals and final score;
 *   2. buffers the survivors and commits them in batches to the PUBLIC dataset
 *      RemiFabre/faience-games, one JSONL file per batch, so the git history
 *      stays sane and a Space restart can never corrupt a shard;
 *   3. answers GET /stats with its counters, so anyone can see it working.
 *
 * What it deliberately does NOT do: it never logs, stores or forwards an IP
 * address or a user agent. Node's http server writes no access log, nothing
 * here reads req.socket.remoteAddress or req.headers beyond the method and
 * path, and the stored record is a canonical rebuild containing only the
 * game itself. The playing page promises players exactly that.
 *
 * A game must never depend on this process: the page fires and forgets, and
 * retries from localStorage on the visitor's next game if this Space was
 * asleep. So restarts here cost at most one batching window of data, and the
 * batching window (or SIGTERM, which the Space sends before sleeping) bounds
 * that loss.
 *
 * Run: node server.js          (PORT env, default 7860; HF_TOKEN enables
 *                               persistence, without it games are counted
 *                               and dropped, which is what local dev wants)
 */

import http from "node:http";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";

import { recordKey, verifyRecord } from "./verify.js";

const MAX_BODY = 64 * 1024; // a real game is ~2.7 KB; 64 KB is generous

/** Commits batches of records to a Hugging Face dataset repo. */
export function createHfGameStore({
  token = process.env.HF_TOKEN,
  repo = process.env.DATASET_REPO || "RemiFabre/faience-games",
  fetchFn = globalThis.fetch,
} = {}) {
  if (!token) return null;
  return {
    repo,
    async save(path, lines) {
      const body = [
        JSON.stringify({ key: "header", value: { summary: `games: ${lines.length} record(s)` } }),
        JSON.stringify({
          key: "file",
          value: {
            path,
            encoding: "base64",
            content: Buffer.from(lines.join("\n") + "\n").toString("base64"),
          },
        }),
      ].join("\n");
      const r = await fetchFn(`https://huggingface.co/api/datasets/${repo}/commit/main`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/x-ndjson" },
        body,
      });
      if (!r.ok) throw new Error(`HF commit failed: ${r.status}`);
    },
  };
}

export function createIngestServer({
  port = 7860,
  store = null, // library default OFF, so tests never touch HF
  batchMax = 25, // a flush this size goes out at once…
  flushMs = 5 * 60_000, // …otherwise the timer bounds the data-loss window
  adminToken = null, // POST /flush bearer, for ops; unset = endpoint off
} = {}) {
  const stats = {
    ok: true,
    since: new Date().toISOString(),
    received: 0,
    accepted: 0,
    finished: 0,
    unfinished: 0,
    duplicates: 0,
    rejected: 0,
    reasons: {},
    committed: 0,
    commit_failures: 0,
    pending: 0,
    dataset: store ? `https://huggingface.co/datasets/${store.repo}` : null,
    note: "no IP addresses or user agents are logged or stored, ever",
  };

  let buffer = []; // verified records waiting for the next dataset commit
  const seen = new Set(); // content hashes, so a retried beacon is not stored twice
  const SEEN_MAX = 5000;
  let seq = 0;

  function takeRecord(record) {
    const hash = createHash("sha256").update(recordKey(record)).digest("hex");
    if (seen.has(hash)) {
      stats.duplicates += 1;
      return;
    }
    seen.add(hash);
    if (seen.size > SEEN_MAX) seen.delete(seen.keys().next().value);
    stats.accepted += 1;
    stats[record.final.finished ? "finished" : "unfinished"] += 1;
    buffer.push({ received_at: new Date().toISOString(), ...record });
    stats.pending = buffer.length;
    if (buffer.length >= batchMax) void flush();
  }

  // The batch leaves the buffer BEFORE the await, so records landing mid-commit
  // start the next batch; a failed commit puts its batch back at the front.
  let flushing = Promise.resolve();
  function flush() {
    flushing = flushing.then(async () => {
      if (!store || !buffer.length) return;
      const batch = buffer;
      buffer = [];
      stats.pending = 0;
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      const path = `games/${stamp.slice(0, 10)}/${stamp}-${(seq += 1)}.jsonl`;
      try {
        await store.save(path, batch.map((entry) => JSON.stringify(entry)));
        stats.committed += batch.length;
      } catch {
        stats.commit_failures += 1;
        buffer = batch.concat(buffer);
      }
      stats.pending = buffer.length;
    });
    return flushing;
  }

  const CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };

  const server = http.createServer((req, res) => {
    const path = new URL(req.url, "http://x").pathname;
    if (req.method === "OPTIONS") {
      res.writeHead(204, CORS);
      res.end();
    } else if (path === "/game" && req.method === "POST") {
      // any content type (sendBeacon ships text/plain), any garbage: the reply
      // is 204 either way, because the page never reads it and an attacker
      // probing the validator learns nothing from a status code
      let body = "";
      req.on("data", (chunk) => {
        body += chunk;
        if (body.length > MAX_BODY) req.destroy();
      });
      req.on("error", () => {});
      req.on("end", () => {
        stats.received += 1;
        let verdict = null;
        try {
          verdict = verifyRecord(JSON.parse(body));
        } catch {
          verdict = { ok: false, reason: "not JSON" };
        }
        if (verdict.ok) {
          takeRecord(verdict.record);
        } else {
          stats.rejected += 1;
          stats.reasons[verdict.reason] = (stats.reasons[verdict.reason] || 0) + 1;
        }
        res.writeHead(204, CORS);
        res.end();
      });
    } else if ((path === "/stats" || path === "/") && req.method === "GET") {
      res.writeHead(200, { "Content-Type": "application/json", ...CORS });
      res.end(JSON.stringify(stats, null, 1));
    } else if (path === "/health") {
      res.writeHead(200, { "Content-Type": "application/json", ...CORS });
      res.end(JSON.stringify({ ok: true }));
    } else if (path === "/flush" && req.method === "POST") {
      const auth = req.headers.authorization || "";
      if (!adminToken || auth !== `Bearer ${adminToken}`) {
        res.writeHead(403, CORS);
        res.end();
        return;
      }
      flush().then(() => {
        res.writeHead(200, { "Content-Type": "application/json", ...CORS });
        res.end(JSON.stringify({ ok: true, committed: stats.committed, pending: stats.pending }));
      });
    } else {
      res.writeHead(404, CORS);
      res.end();
    }
  });

  const flusher = setInterval(flush, flushMs);
  flusher.unref?.();
  let onTerm = null;
  if (store) {
    // the Space sends SIGTERM before sleeping: the buffer's last chance
    onTerm = () => {
      void flush().finally(() => process.exit(0));
    };
    process.once("SIGTERM", onTerm);
  }

  return new Promise((resolve) => {
    server.listen(port, () =>
      resolve({
        port: server.address().port,
        stats,
        flush, // tests await this; prod uses batchMax + the timer + SIGTERM
        close: () => {
          clearInterval(flusher);
          if (onTerm) process.removeListener("SIGTERM", onTerm);
          server.close();
        },
      })
    );
  });
}

// CLI entry: node server.js
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const store = createHfGameStore();
  const s = await createIngestServer({
    port: Number(process.env.PORT || 7860),
    store,
    adminToken: process.env.HF_TOKEN || null,
  });
  console.log(
    `faience ingest on :${s.port} — verified games ${
      store ? `commit to ${store.repo}` : "are counted and dropped (no HF_TOKEN)"
    }`
  );
}
