# Handoff: mobile layout for the Faïence player

*Written 2026-08-16 by the agent that built GUI v3–v6, for the agent that will
do the phone layout. Rémi will give the detailed spec in the session itself —
this file is the context he should not have to repeat.*

## The mission (as Rémi framed it)

The game at <https://remifabre.github.io/ludometer/> (source: `web/player/`)
works on a phone but is "a scaled computer browser" — it needs a **rethought
layout, not a shrunk one**. His starting points, to be refined with him:

- Copy what Board Game Arena does for this kind of game on phones.
- On one vertical screen: **your whole board and the factories together**;
  ideally the opponent's board too — if that does not fit, a *fast* way to
  glance at it (he said "a fast scroll").
- Expect to add **configurability in the settings panel** for layout choices.
- He is *not* sure of the design yet. Ask, propose, mock up — do not guess big.

## Non-negotiable invariants (learned the hard way)

1. **The page always renders a complete position.** No floating clones parked
   over the board, no half-states. Derived views (`heldView`, `placedView` in
   `web/player/js/app.js`) draw the hand tray and confirm mode; tile flights
   are transient clones that self-remove (see `ui/animate.js`). v4.1 tried
   parked fixed-position clones: they scrolled apart from the table and
   collided with real tiles. Rémi explicitly killed that approach.
2. **No overlays over the table.** No pop-ups, sheets, dialogs. The test suite
   rejects `.overlay, .sheet, [role=dialog], [aria-modal]`. Inline panels and
   in-row swaps (see the confirm banner living inside `.middle-actions`) are
   the house pattern. Also: **zero layout shift** for state changes — Rémi
   measured and complained when placing a move moved the page.
3. **Every colour comes from `ui/theme.css` tokens.** Not one hex literal in
   board.css; skins ("dusk") must keep working. Read `ui/THEMING.md`.
4. **Motion is governed only by the in-page speed setting** (`ui/animate.js`),
   never by `prefers-reduced-motion` — deliberate, documented in that file.
5. **`web/play/ui/` and `web/player/ui/` are identical copies.** After editing
   one: `cp` to the other, `diff -rq` to prove it. The local Flask GUI
   (`web/play/`) shares the kit and must not regress.
6. **The guardrail is `node web/play/test/gui.test.mjs`** (headless Chrome,
   real page, numeric assertions: contrast, flight durations at every speed,
   every move animated, confirm-mode flow, coach flow, history navigation, no
   overlays). Run `--only player` while iterating; run both before shipping.
   Extend it with mobile checks (narrow-viewport run) rather than around it.

## Architecture map (player page)

- `web/player/index.html` — shell; brand is **Faïence** (Azul only as factual
  attribution in the About panel — trademark decision, do not resurface it).
- `js/app.js` — the page: render() draws one of {live, heldView, placedView,
  history frame}; `route()` → confirm mode (`propose`/`cancelMove`) or `play()`;
  coach pre-analysis (`startAnalysis`, worker `analyze`, cancel-on-move,
  fallback to post-move rating under 400 sims); nav-step animations; score
  pops; analytics events.
- `js/analytics.js` — GoatCounter pings (`https://faience.goatcounter.com`),
  cookie-free; guards out localhost so tests never pollute the public tally.
  Events: pageview, game-start, `game-end/<model>/<result>` + score title.
- `ui/` — the shared kit: board.js (twin boards + middle), animate.js
  (flights: style-flush before transform, per-tile "landed" seating),
  settings.js (speed / score pop-ups / confirm-each-move rows), popups.js,
  confirm.js, history.js, status.js, log.js, scoring.js, board.css, theme.css.
- `css/style.css` — page-only chrome (topbar, layout grid, corner buttons).
  The existing `@media (max-width: 900px/420px)` rules are the "shrunk
  browser" the mobile work replaces.

## Workflow that works here

- Iterate: edit → `node web/play/test/gui.test.mjs --only player` → screenshot
  via a throwaway CDP script (pattern: serve `web/player/`, headless Chrome
  `--window-size`, deal with seed 31337 + think=0, click around, capture).
  Look at the screenshots — Rémi judges visually and so should you.
- Ship: commit to main (style: story-first messages), then
  `./scripts/deploy_player.sh --no-export` (`--no-export` keeps the current
  net; without it, the best-rated checkpoint is re-exported — takes minutes,
  fine to run during training). The script tests, stages, pushes `gh-pages`,
  and waits for the live URL. CDN freshness lags ~10–60 s; verify with
  `curl "…/index.html?v=$RANDOM"`.

## Rémi's working preferences

- Voice messages; "Azul" may arrive as "Asul", GoatCounter as "GoatComputer".
- Once he has stated a direction, implement, test, deploy, and report — he
  plays the deployed site, not localhost. Confirm-first only for scope changes.
- Simplicity beats cleverness; when an interaction fights you, he prefers
  removing the mechanism over patching it. Cancels reset all the way.
- He flags riskiness explicitly ("only do this if it won't break the game") —
  respect that with fallbacks and cancellation paths, as the coach work did.
- Honesty features matter to him: the public tally link, the "what this is"
  panel, the not-affiliated disclaimer. Keep them intact in any relayout.
