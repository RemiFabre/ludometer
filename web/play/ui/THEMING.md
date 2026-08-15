# Theming the table

Every colour this game draws with lives in **`theme.css`**, as a CSS custom
property on `:root`. Nothing else — not `board.css`, not either page's
`style.css`, not a JS module — is allowed a hex literal. If you want to restyle
the table, that is the only file you open.

```
web/play/ui/theme.css      ← edit here
web/player/ui/theme.css    ← byte-for-byte copy (tests/test_gui.py enforces it)
```

## The one rule the palette is built on

**Hue means occupied.**

A player's board is mostly empty: 25 wall squares, 15 pattern-line slots, 7
floor slots. Those used to be painted in a pale tint of the colour that belongs
there, which turned each board into forty pastel squares with five real tiles
hidden among them. Now:

* a **tile** is saturated, carries a fired rim (`--cN-rim`) and casts a shadow —
  it stands *on* the board;
* an **empty slot** is `--slot`: one neutral, everywhere, sunk into the board
  with an inset shadow — it is a hole *in* the board;
* the wall keeps its pattern as a thick **outlined** diamond in `--cN-motif`
  (the full glaze by default, so the colour reads from across the table), with
  a `--cN-wash` inside it, at `--slot-motif` opacity. Outlined, never filled —
  enough to plan with, never enough to look like a tile. A skin whose slot is
  too close to a glaze re-points that colour's motif (dusk does this for
  charcoal: `--c3-motif: var(--c3-ink)`).

Keep that contrast, whatever else you change. `web/play/test/gui.test.mjs`
checks it numerically: every empty square must be near-neutral (low chroma)
while the four chromatic glazes must be strongly saturated, and every empty
square must be visibly recessed while every tile is visibly raised.

## The token groups

| group | what it is | examples |
|---|---|---|
| the table | cloth, plaster, mortar, ink | `--linen`, `--plaster`, `--grout`, `--ink-soft` |
| the five glazes | one set of five per tile colour | `--c0` … `--c4`, plus `-hi` `-lo` `-rim` `-ink` `-wash` |
| empty slots | the neutral well | `--slot`, `--slot-lo`, `--slot-rim`, `--slot-shade`, `--slot-motif`, `--cN-motif` |
| fixtures | factory dishes, the centre, the marker, the lid | `--dish`, `--marker-face`, `--lid-receiving` |
| meaning | what the status band and the coach are *about* | `--tone-you`, `--tone-ai`, `--grade-blunder`, `--gain` |
| chrome | panels, buttons, fields, focus | `--panel-top`, `--btn-primary-top`, `--focus-ring` |
| geometry | tile size, corner radius, the three type families | `--tile`, `--radius`, `--font-display` |

Colours in the *meaning* group are aliases: `--tone-ai: var(--c0)`. That is
deliberate — the interface never invents an accent that is not already a tile on
the table, so a new skin gets a coherent UI for free.

## The five glazes

Each colour needs five values. `--cN` is the glaze; `--cN-hi` and `--cN-lo` are
the lit and shaded edges of the bevel; `--cN-rim` is the darker fired edge drawn
around a placed tile; `--cN-ink` is the same hue as a *line* — outlines, small
text, the wall's diamonds — and must stay legible against `--slot` and against
`--plaster`. `--cN-wash` is a transparent version for highlight backgrounds.

The shipped values are matched to the tiles in the physical base game — cobalt,
mustard ochre, brick terracotta, charcoal, and a pale cyan-turquoise ("ice")
that must read clearly lighter than the cobalt at a glance. That last constraint
is the one people get wrong: if ice and cobalt are close in lightness the board
becomes unreadable, whatever their hues.

## Writing a skin

A skin is the same list of properties, redeclared under a `data-skin` attribute
on the root element. `theme.css` ships one, `dusk`, as a worked example — same
factory, lit at the end of the day.

```css
:root[data-skin="seaside"] {
  --linen: #dfe7e6;
  --plaster: #f7fbfa;
  --grout: #a9bcbb;
  --ink: #1d2b2b;

  --c0: #1d4f8f; --c0-hi: #5d8ec9; --c0-lo: #0d2f5c; --c0-rim: #0a2749; --c0-ink: #164079;
  /* … c1 through c4 … */

  --slot: #c2cbca;
  --slot-lo: #adb8b7;
  --slot-rim: #8d9a99;
}
```

Turn it on:

```js
document.documentElement.dataset.skin = "seaside";   // or delete it for the default
```

Nothing else needs touching. There is no skin picker in the interface today —
the mechanism exists so a future change is one file and one line, not a sweep
through the stylesheets.

### Checklist for a new skin

1. Redeclare **every** token you are changing, including the `-hi`/`-lo`/`-rim`/
   `-ink`/`-wash` members of each glaze. Half a glaze looks broken, not eclectic.
2. Keep `--slot` near-neutral and darker than `--plaster`. That is the whole
   readability argument above.
3. Check `--cN-ink` against both `--slot` and `--plaster` — it is used for text.
4. If your ground is dark, redeclare `--shadow-panel`, `--shadow-tile` and
   `--shadow-tile-raised`: warm brown shadows disappear on a slate table.
5. Run `node web/play/test/gui.test.mjs` — it renders the board and asserts the
   filled-versus-empty contrast holds.

## What is deliberately *not* themed

The tile's bevel — the pinwheel conic gradient and the glaze highlight in
`board.css` — is geometry, not colour, and it is drawn from `--base`, `--hi` and
`--lo`. A skin changes the clay; it does not need to redraw the kiln.

Do not reproduce the published Azul artwork or the layout of any commercial Azul
client here. The palette is matched to the physical tiles, which is a fact about
the game; the drawing is ours.
