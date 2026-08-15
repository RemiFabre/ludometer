# `web/play/ui/` — the portable half of the table

These files draw an Azul game. They know the engine's **state JSON** and the
engine's **action encoding** (`source*30 + colour*6 + destination`), and nothing
else: no framework, no build step, no `fetch`, no endpoint name, no globals. The
local GUI (`web/play/app.js`) and the GitHub Pages player (`web/player/`) can
therefore draw the same table from two completely different back ends — one a
Flask server, one a Web Worker running ONNX in the tab.

`tests/test_gui.py::test_the_shared_ui_modules_are_page_agnostic` fails if a
module ever grows a server reference.

| file | exports | what it owns |
|---|---|---|
| `dom.js` | `node`, `tileEl`, `markerChip`, `actionId`, `preview`, `destsFor`, colour tables | the primitives, plus the two rules the view layer is allowed to know |
| `board.js` | `createBoard`, `createMiddle` | one player board (both players use the same call), and the factories + centre |
| `status.js` | `createStatus` | the status band, including the AI's clock |
| `log.js` | `renderLog`, `coachChip`, `formatDelta` | the move log, every entry drawn identically |
| `scoring.js` | `renderRoundPanel`, `renderFinalPanel`, `clearScoring` | inline round and end-game scoring |
| `animate.js` | `flyTiles`, `flightDuration`, `sleep` | straight-line tile flights |
| `board.css` | — | tokens, tiles, boards, middle, status band, log, scoring, flight layer |

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
import { renderLog } from "./ui/log.js";
import { flyTiles } from "./ui/animate.js";

const status = createStatus(document.getElementById("status"));
const middle = createMiddle(document.getElementById("middle"), { onPick: pick });
const you = createBoard(document.getElementById("board-human"),
                        { seat: 0, interactive: true, onPlay: play });
const them = createBoard(document.getElementById("board-ai"), { seat: 1 });

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

* **Overlays: there are none, by design.** Round and game-end scoring is drawn
  inline under the boards and the status band carries every "please wait" state.
  Keep it that way — the board must stay readable at all times.
* **Reduced motion is respected in `animate.js`**, which returns immediately, so
  a turn sequence still runs; do not gate animations anywhere else.
* **The log is deliberately flat.** Same type, same weight, same rule for every
  entry, newest at the bottom. Highlighting the last entry makes it read as a
  heading, which is what this redesign removed.
* **Coach verdicts are just log data.** `renderLog` draws `entry.coach` through
  `coachChip`; a port that has its own search can produce the same shape
  (`{delta, best_text, unrated, reason, forced}`) and get the same chip for free.
* **One known gap in the local app** (not in these modules): when the AI plays
  twice in a row across a round boundary, only its first move is animated — the
  intermediate position after the refill is never observed by the page. The
  second move still appears in the log and on the board.
