# `web/play/ui/` — the portable half of the table

These files draw an Azul game. They know the engine's **state JSON** and the
engine's **action encoding** (`source*30 + colour*6 + destination`), and nothing
else: no framework, no build step, no `fetch`, no endpoint name, no globals. The
local GUI (`web/play/app.js`) and the GitHub Pages player (`web/player/`) can
therefore draw the same table from two completely different back ends — one a
Flask server, one a Web Worker running ONNX in the tab.

`web/player/ui/` is a **byte-for-byte copy** of this directory. Change these
files, then copy them across; `tests/test_gui.py` fails if the two ever drift.

| file | exports | what it owns |
|---|---|---|
| `theme.css` | — | **every colour in the game**, as custom properties, plus a skin (see THEMING.md) |
| `dom.js` | `node`, `tileEl`, `markerChip`, `actionId`, `preview`, `destsFor`, colour tables | the primitives, plus the two rules the view layer is allowed to know |
| `board.js` | `createBoard`, `createMiddle` | one player board (both players use the same call), and the factories + centre |
| `status.js` | `createStatus` | the status band, including the AI's clock |
| `log.js` | `renderLog`, `glyphRun`, `coachChip`, `formatDelta` | the move log — newest first, tiles drawn as tiles |
| `scoring.js` | `renderRoundPanel`, `renderFinalPanel`, `clearScoring` | inline round and end-game scoring |
| `animate.js` | `flyTiles`, `flightDuration`, `sleep`, `speed`, `setSpeed`, `initSpeed` | straight-line tile flights, and the one switch that governs motion |
| `settings.js` | `createSettings` | the gear and its inline panel |
| `history.js` | `createHistory`, `bindHistoryKeys` | ← / → move navigation over stored positions |
| `board.css` | — | tiles, boards, middle, status band, settings, navigator, log, scoring, flights |

## The state the modules expect

Exactly what `AzulState.to_json()` returns (Python) — which is exactly what
`web/player/js/engine.js`'s `toJSON()` returns, field for field. That is the
whole compatibility contract:

```js
{ round, current_player, factories: [[5], ×5], center: [5], marker_in_center,
  bag: [5], lid: [5], tiles_left, scores: [2], is_terminal,
  players: [{ score, wall: [5][5], pattern_lines: [{capacity, color, count}],
              floor: [5], floor_marker, floor_penalty, completed_rows, ... }] }
```

The scoring panels additionally take the reports built by
`ludometer/gui/moves.py` (`round_report` / `final_report`). A port that has no
such reports can skip `scoring.js` and keep the rest.

## Wiring it up (the whole integration)

```js
import { createBoard, createMiddle } from "./ui/board.js";
import { createStatus } from "./ui/status.js";
import { createSettings } from "./ui/settings.js";
import { createHistory, bindHistoryKeys } from "./ui/history.js";
import { renderLog } from "./ui/log.js";
import { flyTiles, initSpeed } from "./ui/animate.js";

initSpeed();                                   // do this before anything animates
createSettings(document.getElementById("settings"));
const status = createStatus(document.getElementById("status"));
const middle = createMiddle(document.getElementById("middle"), { onPick: pick });
const you = createBoard(document.getElementById("board-human"),
                        { seat: 0, interactive: true, onPlay: play });
const them = createBoard(document.getElementById("board-ai"), { seat: 1 });
const nav = createHistory(document.getElementById("nav"),
                          { log: () => S.log, onChange: render });
bindHistoryKeys(nav, { enabled: () => !busy });

you.render({ state, legalActions, selection, title: "You", toMove: true });
them.render({ state, title: "AI" });
status.set({ headline: "Your turn — pick a colour", detail: "Round 1", tone: "you" });
```

`onPlay(actionId)` is the only thing a host page must implement; `legalActions`
is an array of action ids from wherever the rules live. Boards are identical by
construction — `interactive` is the only difference — so a page cannot
accidentally make one player's board bigger than the other's.

Animations aim at elements the board hands back: `lineSlots(row)`,
`floorSlots()`, `wallCell(row, col)`, `lineTiles(row)`, plus `sourceTiles()` and
`remainderTiles()` on the middle. Build the flights **before** re-rendering, so
the tiles fly from where they actually were:

```js
await flyTiles(taken.map((from, i) => ({ from, to: targets[i], color })),
               { layer: flyLayer });
adopt(newState);   // now redraw
```

## Notes for whoever ports this

* **Motion is governed by the settings panel, and by nothing else.** There used
  to be a `prefers-reduced-motion` check in `animate.js` and a blanket
  `transition-duration: .001ms !important` in `board.css`. macOS ships "Reduce
  motion" on for a lot of people, so those players watched tiles teleport with
  no way to ask for the animation back, while every headless test passed
  (the harness set `no-preference` first). Do not reintroduce either gate: add
  a setting instead. `prefersReducedMotion()` is still exported, purely so the
  settings panel can explain itself.
* **Every move carries the position it was played from** (`state_before`). That
  is what lets a page animate *both* of the AI's moves when it moves twice
  across a round boundary, and it is what `history.js` replays. A back end that
  does not report it loses those two features and nothing else.
* **Overlays: there are none, by design.** Round and game-end scoring is drawn
  inline under the boards, the status band carries every "please wait" state,
  and the settings panel opens in the flow rather than over the page. Keep it
  that way — the board must stay readable at all times.
* **The log is flat and newest-first.** Same type, same weight, same rule for
  every entry, including the top one. Highlighting the newest makes it read as a
  heading, which is what this redesign removed.
* **Colours live in `theme.css`.** No hex literal belongs in `board.css`, in a
  page shell, or in a JS module. See THEMING.md.
* **Coach verdicts are just log data.** `renderLog` draws `entry.coach` through
  `coachChip`; a port that has its own search can produce the same shape
  (`{delta, best_text, unrated, reason, forced}`) and get the same chip for free.
