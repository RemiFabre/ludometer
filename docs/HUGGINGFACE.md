# Moving Faïence to Hugging Face, and logging every game

*Written 2026-08-18 by the agent that built the browser player, for the agent
that will do this work. Rémi set the goals; the analysis and the
recommendations are mine and are meant to be argued with. Nothing here is a
recipe. Where I had to guess, I say so, and §7 lists what I could not settle
from outside an HF account.*

---

## 1. The goals

In Rémi's priority order.

1. **Faïence lives on Hugging Face.** The HF version becomes the canonical one,
   and every link we control points at it: README, the About panel, the social
   cards, the dashboard, the deploy script, the posts.
2. **The GitHub Pages URL keeps working** and sends people to the HF version.
   It is already printed in a Reddit thread, a LinkedIn post and every shared
   card, so it can never become a 404.
3. **Every finished game is logged** so it can be used for training. Every game,
   not a filtered subset. (Rémi first considered keeping only draws and human
   wins; he has since settled on everything, which is also what I would argue
   for: filtering at collection is a one way door, and at roughly a kilobyte a
   game there is no storage argument for it.)
4. **The page says so, honestly.** The framing Rémi wants is "this is a research
   project, when you play you help the AI train". Today the About panel promises
   that nothing but an anonymous tally ever leaves the browser, and that
   sentence stops being true the moment this ships. Getting the text right is
   part of the work, not a follow-up.

## 2. What has to stay true

The long version is `docs/HANDOFF-mobile.md`; these are the ones this project
can break.

- **A game must never depend on the upload.** The engine runs in the tab and
  that is the whole point. If the endpoint is asleep, rate limited, blocked by
  an ad blocker or simply gone, the game must play exactly as it does now.
  Fire and forget, no awaiting, no error surfaced to the player.
- **No overlays, no pop-ups, no layout shift.** Consent text and any switch go
  inline, in the page's own idiom.
- **The guardrail is `node web/play/test/gui.test.mjs`** plus
  `web/player/test/browser.test.mjs --live` after a deploy. Both must pass, and
  the live one should be pointed at whatever the canonical URL becomes.
- **No em dashes in anything a player reads.** Colons, commas, full stops,
  brackets.
- **This repo has other agents in it.** At the time of writing one is porting a
  second game into `ludometer/`. This work is almost entirely under `web/`, and
  it should stay that way. Note also that `scripts/deploy_player.sh` stages the
  **working tree**, not the index, so check `git status` before deploying.

## 3. The shape I would expect

Four pieces, and they are deliberately separable.

- **The game**: a Space serving the same static files as today. See D1 for
  static versus Docker, which is a more interesting question than it looks.
- **The ingest**: a small, separate Space. Rémi's
  [openwarlock-signal](https://huggingface.co/spaces/RemiFabre/openwarlock-signal)
  is the working precedent and a good one: a Docker Space running Node, a
  141 byte Dockerfile and a single 16 KB source file. Something that size is
  enough here too. Keeping it separate from the game means a restart or a
  redeploy of the ingest can never take the game down.
- **The data**: a Hugging Face Dataset repo, which is durable, versioned and
  loadable directly by the training pipeline. A Space's own filesystem is
  ephemeral, so writing games there loses them on the next restart. This is the
  single easiest thing to get wrong.
- **GitHub Pages**: a redirect, and, if D2 goes my way, a fallback that still
  plays.

## 4. Decisions worth making before any code

Each has my recommendation, and each is genuinely open.

**D1. Static Space or Docker Space for the game?**
This is the one with a real prize attached. `web/player/js/worker.js` currently
pins `numThreads = 1`, with a comment saying why: `SharedArrayBuffer` needs
cross-origin isolation and GitHub Pages cannot send the `Cross-Origin-Opener-Policy`
and `Cross-Origin-Embedder-Policy` headers that turn it on. The search is
therefore single threaded today. On a Space we control the headers, and this
page is unusually well suited to isolation because it deliberately loads nothing
from any other origin. If we can set those two headers, the tree search can use
real threads, and the AI gets faster for everybody on a multi-core machine.
That is a user-visible improvement that has nothing to do with hosting politics,
and I think it is the strongest argument for the move.
*Recommendation: whichever Space type lets us set those headers, tested early,
because the answer decides the rest of the layout. If a static Space cannot,
Docker with a tiny web server can.* One consequence to plan for: under
`require-corp`, cross-origin subresource loads are blocked, and the GoatCounter
tally is currently fired as `new Image().src` to another origin. Moving it to
`navigator.sendBeacon` sidesteps that, and is the right transport for the game
upload anyway.

**D2. Does GitHub Pages redirect only, or redirect and remain playable?**
Rémi asked for a redirect. My push-back is small but real: GitHub Pages has been
completely reliable and free, Spaces restart, sleep and occasionally break, and
the moment we redirect we inherit the Space's uptime for every link already in
the wild.
*Recommendation: redirect, but keep a working copy of the game on Pages at a
stable path so there is always a URL to hand out when the Space is unhappy. Two
further details: keep the Open Graph tags on the redirect stub, because link
unfurlers and crawlers do not reliably follow JavaScript redirects and the cards
already circulating are generated from that HTML; and decide whether the
redirect is instant or a visible "this has moved" line, since an instant
redirect on a shared link can read as a hijack to a suspicious reader.*

**D3. Is the games dataset public or private?**
*Recommendation: public.* The project's whole posture is that anyone can read
everything it records, the tally is public for exactly that reason, and a public
dataset of human-versus-net Azul games is a genuinely useful research artifact
that costs nothing to share. It also keeps the About panel's honesty claim
simple: everything we collect, you can go and read. If it is private, the text
has to be more careful.

**D4. Is there a switch to turn logging off?**
*Recommendation: yes, visible, in the settings panel, on by default.* Almost
nobody will use it, so the data cost is near zero, and it is the difference
between "this is a research project and here is what it collects" and "we
started uploading your games". Given how much of this project's character comes
from that kind of honesty, I would not trade it for a percent of data.

**D5. Abandoned games too, or only finished ones?**
About half of dealt games are never finished (171 dealt, 83 finished on the
first Reddit day). `navigator.sendBeacon` on page hide is built precisely for
this and would roughly double the harvest.
*Recommendation: log them, flagged as unfinished.* They carry no outcome, so
they are position data rather than labelled data, and the training side should
be able to tell the difference at a glance.

**D6. What does the ingest Space log about the sender?**
A game record contains nothing personal, but an HTTP endpoint sees IP addresses,
and default access logging in most stacks writes them down. Rémi is in the EU
and the project makes explicit privacy claims.
*Recommendation: do not persist IPs or user agents, say so in the text, and
check the default logging of whatever server ends up being used rather than
assuming.*

**D7. What is the record format once it moves?**
Measured on a real 51 move game, today's record is 2,712 bytes: 1,757 for the
moves, 627 for the deals, 224 header, 77 final. The moves block is written as
one object per move (`{"ply":1,"player":0,"action":12}`, 32 bytes to carry about
7.5 bits) and both `ply` and `player` are derivable when replaying, so a flat
array of action ids costs 185 bytes instead of 1,757. The information floor for
a whole game is somewhere around 50 to 80 bytes.
*Recommendation: flatten the moves, which is free and removes 58% of the file,
and keep the per-round deals even though the seed could regenerate them once
mulberry32 is ported to Python, because deals make a record replayable
independently of our own code version. That lands around 1.1 KB, roughly 400
bytes gzipped, at which point size stops mattering and legibility wins.*

## 5. Inputs the next agent may or may not want

**A record verifies itself, and that makes a public endpoint safe.** Because a
record replays deterministically, the ingest can replay each submission in the
real engine and reject anything that does not reproduce its own claimed final
score. Spam, corruption and mischief all fail that check cheaply, which is a
much better defence than a secret token in client-side JavaScript could ever be.
It does mean the ingest needs the engine, either the Python one from this repo
or the JavaScript one, and either the recorded deals or a port of mulberry32
plus the JS shuffle loop, which an earlier investigation sized at about twenty
lines.

**Commit in batches.** One dataset commit per game would produce an unusable git
history. Buffer and commit periodically, and remember the buffer is lost if the
Space restarts, so the batching window is a data-loss window; pick it with that
in mind.

**Retry cheaply.** If the endpoint is asleep or the request fails, stashing the
record in `localStorage` and trying again on the visitor's next game costs very
little and turns cold starts from lost data into delayed data.

**The Python half already exists.** `ludometer/human/` converts a validated game
into the `replay.npz` that the trainer's `--pretrain` flag reads, with a policy
mask convention for rows that should not teach the policy head, and
`data/human/replay.npz` already holds 8,563 positions from 319 elite BGA games.
`ludometer/human/fixture.py` is the natural template for an adapter from browser
records. I deliberately did not write that adapter, because another agent is
working in `ludometer/`.

**What is already there on the browser side.** `web/player/js/record.js` builds
the record and documents why each field exists; the session captures each
round's deal and each AI move's simulation count and root value;
`window.faience.record()` returns the current game. Today the page only *offers*
the record as a file, which Rémi is right to think will not produce data at any
useful rate. That UI should probably be replaced by the automatic path plus the
consent text rather than kept alongside it, though a save button is still nice
for the curious.

**One thing to be careful about in the text.** The About panel, the README, the
social card descriptions and the engine strip all currently say some version of
"nothing leaves your browser". They need to change together, and the live test
asserts on some of that copy.

## 6. What finished looks like

Written as outcomes so the implementer can get there any way they like.

- A person opens the canonical HF URL on a phone and on a desktop, plays a full
  game, and nothing about the experience is worse than it is today. If D1 went
  well, the search is measurably faster.
- That game appears in the dataset, and replaying it offline reproduces its
  final score exactly.
- Games that are abandoned, and games played while the endpoint is unreachable,
  never produce a broken page or a visible error.
- The old GitHub Pages URL still leads a player to a playable game.
- The page's description of what it collects matches what it actually collects,
  in every place that description appears.
- `gui.test.mjs` passes on both pages, and the live test passes against the new
  canonical URL.

## 7. Questions I could not answer from here

These need an HF account or an experiment, and they shape the design, so they
are worth resolving first rather than discovering later.

1. Can a **static** Space set `Cross-Origin-Opener-Policy` and
   `Cross-Origin-Embedder-Policy`? If yes, the game can stay purely static and
   still get threads. If no, D1 turns into a Docker Space with a small server.
2. What does a Space do about the payload size? The site is roughly 34 MB on
   disk, dominated by a 7.3 MB ONNX model and two onnxruntime WASM builds, of
   which any one visitor downloads about 10 MB gzipped. Check LFS behaviour and
   what cache headers the Space serves, since the model is the whole first-load
   cost.
3. Under Rémi's Pro account, what is the actual sleep and restart behaviour of
   the ingest Space, and can it be kept warm?
4. What are the practical rate limits on commits to a Dataset repo, which sets
   the batching window in §5?
5. Does anything in the HF Space environment interfere with Web Workers or with
   serving `.wasm` with the right MIME type? The player runs its search in a
   worker and is fussy about both.

## 8. What was actually built (2026-08-18, the implementing agent)

Shipped, live, and tested end to end. The decisions, where they differ from or
settle the sections above:

- **The layout.** Three repos on the Hub, exactly the shape §3 proposed:
  - the game: [Space RemiFabre/faience](https://huggingface.co/spaces/RemiFabre/faience),
    static SDK, canonical play URL **https://remifabre-faience.static.hf.space/**;
  - the ingest: [Space RemiFabre/faience-ingest](https://huggingface.co/spaces/RemiFabre/faience-ingest)
    (https://remifabre-faience-ingest.hf.space), Docker, Node, no dependencies,
    source in `web/ingest/`, deployed by `scripts/deploy_ingest.sh`;
  - the data: [dataset RemiFabre/faience-games](https://huggingface.co/datasets/RemiFabre/faience-games),
    public (D3), CC0, JSONL shards under `games/YYYY-MM-DD/`, one file per
    ingest batch, nothing ever rewritten. Card source: `web/ingest/DATASET.md`.
- **D1 went static, without the threads.** Rémi wants the page to stay a plain
  static site people could even download and run, so no Docker, no COOP/COEP
  experiment, `numThreads` stays 1. The prize §D1 described is still on the
  table for later; nothing shipped forecloses it.
- **D2: a visible moved-notice, not an instant redirect** (Rémi's call), from
  `web/pages/index.html`: same Open Graph tags as ever (the circulating cards
  keep unfurling), one button to the new address, a line pointing at
  **classic/**, where the previous build keeps playing as a fallback, and one
  `/moved` tally event so the old link's remaining traffic is readable on the
  GoatCounter dashboard.
- **D4: the switch is in Settings** ("Share played games", On by default),
  honoured by `web/player/js/upload.js` on every path a record can leave the
  page. D5: abandoned games go too, flagged `finished: false`, sent by
  `sendBeacon` on `pagehide` and when a live game is dealt over; failed sends
  wait in localStorage and retry on the next visit (at most 8 queued). A page
  load pings the collector's `/health`, so a sleeping Space is warm long
  before the first game could finish.
- **D6 as recommended:** the collector never reads, logs or stores IPs or user
  agents; `GET /stats` says so and shows the live counters.
- **D7: the format stays `faience-game/1`, unflattened.** The harvest and
  training side is being built against it by another agent right now, it is
  ~400 bytes gzipped, and changing the wire format mid-flight was the one
  collision Rémi warned about. Flatten later, in a `/2`, if it ever matters.
- **The §5 replay defence is the whole gate:** `web/ingest/verify.js` replays
  every submission in a byte-for-byte copy of the page's engine (the deploy
  script stamps `web/player/js/engine.js` into the Space), demands the
  recorded deals, scores and round count reproduce exactly, then stores a
  canonical rebuild (bounded strings, known fields only), deduplicated by
  content. Tested by `web/ingest/test/ingest.test.mjs` (28 checks) and live:
  a real game was accepted and committed, a tampered score bounced.
- **The tests:** `gui.test.mjs --only player` passes with the new copy and
  switch; `browser.test.mjs --live` now targets the Space and switches
  sharing OFF before loading, so harness games can never enter the dataset
  (the localhost guard covers every local run, same as the tally).

Ops notes: the ingest Space carries Rémi's `HF_TOKEN` as a secret for the
dataset commits; batches flush at 25 games, every 5 minutes, on SIGTERM, and
on `POST /flush` (Bearer HF_TOKEN). Free CPU Spaces sleep after ~48 h idle;
the health-ping plus the retry queue turn that from lost data into delayed
data, so nothing is pinned.

## 9. Where things are

- The game: `web/player/`, deployed by `scripts/deploy_player.sh` (reads the
  working tree, pushes the Space and `gh-pages`, waits for both live URLs).
- The collector: `web/ingest/`, deployed by `scripts/deploy_ingest.sh`; the
  moved-notice stub: `web/pages/index.html`; the sharing client:
  `web/player/js/upload.js`.
- The record: `web/player/js/record.js`, plus deal capture in
  `web/player/js/game.js` and the row in `index.html` / `css/style.css`.
- The tally: `web/player/js/analytics.js`, GoatCounter, dashboard public.
- Tests: `web/play/test/gui.test.mjs` (both pages, headless Chrome, numeric
  assertions) and `web/player/test/browser.test.mjs` (a full game, `--live`
  against production).
- The human-games pipeline and its own long handoff: `ludometer/human/` and
  `docs/HUMAN_GAMES.md`.
