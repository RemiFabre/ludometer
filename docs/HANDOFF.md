# Handoff: the public game, as of 2026-08-18

*Written by the agent that did the layout, wording and game-record work over
16 to 18 August, for whoever picks it up next. This is the state of the world
and the things you cannot read off the code. Your actual task is almost
certainly `docs/HUGGINGFACE.md`, which Rémi treats as the objective definition;
read this first for context, then that.*

---

## 1. Where things stand

**The game is live, popular for the first time, and healthy.**
<https://remifabre.github.io/ludometer/> serves run4/ckpt-037888 (internal Elo
2361, 1.81M parameters, 7.3 MB ONNX). It went on Reddit's r/boardgames on
17 August. Last counts I took, late on 18 August: 432 visits, 171 games dealt,
83 finished, of which the net won 72, humans won 9, and 2 were draws. The
GoatCounter dashboard is public at <https://faience.goatcounter.com> and events
land as rows in its Pages list. Headless test games do not pollute it, because
GoatCounter filters them as bots.

**Training continues under other sessions.** Do not assume the deployed net is
the newest one; `scripts/deploy_player.sh` without `--no-export` re-exports the
best rated checkpoint, which takes minutes and is safe to run while training.

## 2. What changed in this session

All of it is deployed and verified live.

- **Layout, rethought rather than shrunk.** There are now two board layouts,
  chosen in the settings panel with a minus/plus row and defaulted per device:
  a desktop starts on the classic table (big boards below the factories) and a
  phone starts on the compact one (factories left, both boards beside them, the
  whole game on one screen). Lives in `web/player/ui/layout.js` and
  `body[data-boards]`. Rémi tried the compact layout on his desktop and switched
  straight back, hence the split default.
- **The settings panel absorbed the AI clock.** "AI thinks for" is no longer in
  the top bar; all the knobs are in one place. The bottom strip keeps only the
  tile totals, which also hosts `#lid-row`, the element discarded tiles fly to.
- **Per-colour supply counts are gone for good.** Rémi's call, do not bring them
  back.
- **Round scoring counts one player, then the other**, in both the hosted player
  and the local GUI. It used to interleave, which made the arithmetic
  impossible to follow.
- **The About panel was reordered and de-jargoned**, and every em dash in
  player-facing text on both pages became a colon, comma, full stop or bracket.
- **Game records.** `web/player/js/record.js` turns any game into a
  `faience-game/1` record that replays exactly, and the page offers it at game
  end as a file, a line of text, or a prefilled GitHub issue. Nothing is
  uploaded. `window.faience.record()` returns the current game.

Both test suites were extended to cover all of this, and they pass.

## 3. Your likely task

`docs/HUGGINGFACE.md`. Rémi wants Faïence hosted on Hugging Face as the
canonical version, the GitHub Pages link redirecting there, and **every** game
logged automatically for training, with honest text about it. That document
holds the goals, seven open decisions with my recommendations, the things I
would push back on, and five questions I could not answer without an HF account.

Two things from it worth knowing before you even open it. First, the manual
export I built is not the plan: Rémi is right that asking people to share by
hand will not produce data, so it should be replaced by the automatic path, and
the record format is the part worth keeping. Second, the most interesting
finding is that `web/player/js/worker.js` pins `numThreads = 1` because GitHub
Pages cannot send the cross-origin isolation headers that `SharedArrayBuffer`
needs; on a Space we control those headers, so the move can make the search
multi-threaded, and that is a better reason for it than hosting preference.

## 4. Hazards in this repo right now

**You are not alone in it.** At least two other sessions are working here.

- One is **porting a second game (Uno) into the framework**, and it is modifying
  the core: `ludometer/azul/engine.py`, `ludometer/train/*`, `ludometer/eval/arena.py`,
  plus new `ludometer/games.py`, `ludometer/uno/`, `configs/uno*.json`,
  `docs/NEXT_GAMES.md`. Expect `ludometer/` to move under you. Web work is the
  safe zone; I stayed entirely inside `web/` for that reason.
- One runs **training and the BGA human-games harvest**: `ludometer/human/`,
  `docs/HUMAN_GAMES.md`, `data/`, `web/harvest.html`, `web/make_harvest.py`,
  and the dashboard files.
- **`README.md` and `NOTES_FOR_REMI.md` are currently dirty** from another
  session. The Hugging Face work needs to edit README links, so coordinate or
  you will collide. Other sessions can be reached with `ListAgents` and
  `SendMessage`; that worked well this session.

**`scripts/deploy_player.sh` stages the working tree, not the git index.** It
rsyncs `web/player/` minus `test/`. Before deploying, run `git status` and
confirm nothing under `web/player/` belongs to someone else. This has shipped
another session's in-progress work to production once already, on 16 August.

**The CDN lags 10 to 60 seconds** behind a deploy. Verify with a cache-busting
fetch before running the live test, or it will test the old build and fail
confusingly:

```
until curl -s "https://remifabre.github.io/ludometer/js/record.js?v=$RANDOM" | grep -q faience-game; do sleep 5; done
```

## 5. How to verify anything

- `node web/play/test/gui.test.mjs --only player` while iterating, both pages
  before shipping. Headless Chrome against the real page, numeric assertions:
  contrast, flight durations at every speed, every move animated, confirm-mode
  flow, coach flow, history, no overlays, the two layouts and their per-device
  defaults, and that a game record replays to the same scores, round count and
  deals in a fresh engine.
- `node web/player/test/browser.test.mjs --budget 2` plays a full game locally;
  add `--live` to play one on production after a deploy. Note it fires real
  GoatCounter beacons, which are filtered as bot traffic.
- Screenshots: serve `web/player/`, drive headless Chrome over CDP, deal seed
  31337 with think 0, click around, capture. Look at the images; Rémi judges
  visually and so should you.

## 6. How Rémi works

- **Voice messages**, so expect transcription noise. "Azul" can arrive as
  "Asul", GoatCounter as "GoatComputer", Hugging Face as "Hogan Face". If a
  substitution is obvious, make it; if it is not, ask.
- **Brainstorming is not an instruction.** I implemented and deployed an idea he
  was still thinking out loud about, and he corrected me. Wait for a direction.
  Once he gives one, though, implement, test, deploy and report: he plays the
  deployed site, not localhost.
- **No em dashes anywhere a person reads.** He finds they smell of AI writing.
  Code comments are exempt.
- **Pages get "too charged" easily.** When he asks for something to be added,
  check what can come out.
- **Honesty features are load-bearing**: the public tally, the "what this is"
  panel, the not-affiliated disclaimer. Any change to what the page collects has
  to change what the page says, in the same commit.
- He flags risk explicitly ("only do this if it won't break the game"), and
  prefers removing a mechanism over patching one that fights him.

## 7. Loose threads nobody owns

- **`post.md`** holds the live Reddit post and a LinkedIn draft that Rémi had
  not published when I stopped. The body and first comment are split because
  LinkedIn downranks posts with external links in the body.
- **`media/ludometer-elo-ladder.png`** was regenerated to his notes (24 hours on
  one laptop, no brand names, no plus signs on Elo values). The generator is not
  in the repo; `media/` is gitignored. If it needs changing again it will have
  to be rebuilt.
- **`data/human/replay.npz`** holds 8,563 positions from 319 validated elite BGA
  games and **is not referenced by any run config**. That is a larger and
  cleaner dataset than browser traffic will produce for a while, and it is
  sitting unused.
- **An adapter from browser records to `replay.npz`** is the missing link
  between the harvest and training. `ludometer/human/fixture.py` is the
  template. I did not write it because of the Uno port in flight.
- **`docs/HANDOFF-mobile.md`** is the previous handoff. Its mission is complete,
  but its invariants section is still the best short statement of the rules this
  page lives by.
