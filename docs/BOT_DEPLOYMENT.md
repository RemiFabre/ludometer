# Shipping a new opponent to the browser player

*The complete procedure, self-contained on purpose: any agent (or Rémi) must
be able to execute it with no other context and no live coordination with
whoever wrote it. If you change the procedure, change this file in the same
commit.*

## The bar for shipping

1. **A new strongest bot must be significantly stronger than the current
   strongest — at least +150 Elo, wall-clock honest** (Rémi's bar,
   2026-08-22). Marginal improvements do not ship; players should feel the
   difference, not read about it.
2. **"Wall-clock honest" is the only Elo that counts.** Ladder ratings at
   fixed sims (e.g. `sims=100`) overstate slow nets: runs/ft1-ft2
   checkpoints rated 2394–2445 on the ladder but LOSE to run4/ckpt-037888
   in real time. The number published for a bot must come from a
   **≥300-game gate at matched think time** against the current strongest
   (the browser plays wall-clock budgets, so this is what a player faces).
   A net that is only stronger at equal search does not qualify as
   stronger. (Decided 2026-08-22; a slower-but-stylish net was proposed as
   an extra opponent and Rémi declined.)
3. Weaker rungs may be added to fill Elo gaps in the ladder without any
   gate beyond their honest rating; they must slot into the naming order.

## Names

The ladder is ceramics, from raw ground to precious glaze, weakest first:

    Sand · Clay · Brick · Ochre · Charcoal · Ice · Cobalt

Reserved for future STRONGER-than-current bots, in this order:

    Porcelain  →  Lapis Lazuli  →  Ultramarine

(Also recorded in `web/player/model/bots.json` under
`reserved_for_future_nets`, with suggested swatch colours.) Rules:
**never rename an existing bot**, never reuse a name, take the next
reserved name for the next top bot. If the reserved list runs out, extend
it in the same spirit and record the extension here and in bots.json.

## The procedure

1. Gate the checkpoint (≥300 games, matched think time, vs the current
   strongest). Keep the resulting Elo; it is what ships.
2. Export — **always an explicit `--ckpt`**; the default
   `find_best_checkpoint()` path scans every run and this repo trains
   several games now (it once picked an 84-input Uno net for the Azul
   player; the parity gate caught it):

       uv run --group export python -m ludometer.export.onnx_export \
         --ckpt runs/<run>/checkpoints/<ckpt>.pt \
         --elo <wall-clock-honest number> \
         --out web/player/model/<id>

   The exporter regenerates the torch reference fixture. **model.onnx,
   model_meta.json and web/player/test/fixtures/torch_reference.json.gz
   are a trio: commit them together**, always (a half-committed trio is
   exactly the drift the deploy gate once caught).
3. Add the entry to `web/player/model/bots.json` (id, name, dir, swatch).
   The page defaults to the strongest bot **by the `elo` field in each
   model_meta.json** automatically — publishing an inflated Elo would
   wrongly seize the default slot, which is why only gated numbers ship.
4. Stage first, never straight to production:

       ./scripts/deploy_staging.sh          # private-ish test Space
       node web/play/test/gui.test.mjs --only player   # must be green

   Have Rémi (or the requesting human) confirm on staging.
5. Production:

       ./scripts/deploy_player.sh --no-export
       node web/player/test/browser.test.mjs --live    # must be green

   The deploy script pushes the Hugging Face Space and the GitHub Pages
   stub together and waits for both URLs.

## Where things are

- Roster + reserved names: `web/player/model/bots.json`
- Advice always comes from the strongest net (`js/worker.js` "coach"
  message; the page fetches it lazily) — a new top bot automatically
  becomes the adviser too.
- Records carry `net.run/checkpoint/elo`, so the dataset stays attributable
  across bot changes; the ingest does not need any update for a new bot.
- History of these decisions: NOTES_FOR_REMI.md (2026-08-21 entry) and
  docs/HUGGINGFACE.md §8.
