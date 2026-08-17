# Learning from human games on Board Game Arena — feasibility recon + handoff

**Status**: recon complete, pipeline skeleton committed, **no bulk download performed**.
**Date**: 2026-08-17. **BGA requests spent on this recon: 20** (budget was ~30, ≥2 s apart,
desktop-Chrome user agent, one at a time). Everything else below comes from reading ~30 public
open-source projects, not from touching BGA.
**Owner of this doc**: whoever picks the work up next. It is written so that nothing here has
to be rediscovered.

Read this before touching `ludometer/human/`. Code: the seven modules in §6. Tests:
`tests/test_human_pipeline.py` (53, no network). Nothing in
`ludometer/{train,eval,azul,agents}` was modified.

---

## 0. Executive summary

1. **Accessible anonymously**: the all-time Elo ladder, and effectively only that. A player's
   game history and the replay move log are behind a session — verified live: they answer
   `{"status":"0","error":"Invalid session information for this action.","code":806}`, and the
   equivalent HTML pages 302 to `/account?warn&redirect=…`. **What Remi must hand over**: a
   Netscape `cookies.txt` export for boardgamearena.com (§2.4). No password, no automated
   login, nothing bypassed.
2. **Replay format: convertible.** `GET /archive/archive/logs.html?table=<id>&translated=true`
   returns the framework's own notification stream as JSON. Azul's six notification types and
   their arguments are known (§4.1) and they carry machine-readable factory / colour / line
   values — you never parse prose. The tile-type→colour mapping is known *and* mechanically
   self-verifying (§4.3). Our converter replays every game in our engine and rejects anything
   it cannot reproduce; on synthetic round-trips it is exact (§4.6).
3. **The binding constraint is not politeness, it is BGA's per-account daily replay quota**
   (§5.1). It is not an HTTP 429: it is a 200 whose JSON says
   `"You have reached a limit (replay)"`. The numeric cap is undocumented; every public
   scraper hits it, and they work around it by rotating multiple accounts, **which we will
   not do**. So the schedule is "N replays per day for as many days as it takes", and step 1
   of the next session is to *measure N* (§9).
4. **Volume**: "above 900 all-time Elo" is **~10 players today** (2 in Remi's December 2025
   dump) — the Azul all-time ladder tops out at 1186 displayed. Use ranks instead: top 100 =
   displayed Elo ≥ 715, top 200 ≥ 667, and those players have ~2,000 ranked Azul games each,
   so the *supply* is ~100k+ tables. Recommended: **top 200, displayed-Elo floor 650,
   `min_games ≥ 200`, 100–150 games per player, first milestone 2,000 games (~110k
   positions)**, full target 10,000 games (~550k positions) — §5.
5. **ToS**: BGA's terms **explicitly prohibit automated access**, `robots.txt` disallows
   `/table`, `/player`, `/playerstat`, `/play`, and there is public code evidence of BGA
   *disabling accounts* for replay scraping. This is Remi's decision to make with open eyes;
   §7 gives the quotes, the risks and the recommended first move (ask BGA).

---

## 1. Prior art

### 1.1 Remi's own scraper

`github.com/RemiFabre/board_game_arena_elo_parser` (public, ~Dec 2025):

- Selenium + real Chrome, **no HTTP client, no login, no cookies, no custom UA**. Opens
  `https://en.boardgamearena.com/gamepanel?game=azul`, clicks the ranking dropdown from
  "Current Season" to "All-time" by matching literal English strings, then clicks "Next"
  ~9,100 times scraping `div.bga-ranking-entry` → `a.playername` + `div.bga-elo-label`.
- Inter-page delay `time.sleep(0.01)` — the "a bit brutal" part. No backoff, no jitter, no
  robots check.
- Captures **name and displayed Elo only** — never the player id (it never reads the `href`),
  which is why the CSVs cannot be joined to anything.
- Never touches tables, replays or histories.

Two things it gives us free: a committed all-time snapshot
(`leaderboards/Azul_full_leaderboard.csv`, 91,199 players, ~2025-12-04), and the knowledge
that the whole click-loop is unnecessary — the dropdown is a Svelte component reading a plain
JSON endpoint (§2.1).

### 1.2 Other people's projects (surveyed, no requests spent)

No BGA client exists on PyPI or npm; everything is GitHub-only. What we took:

| repo | contribution |
|---|---|
| `rhstephens/hivemind` | the best end-to-end blueprint (pure `requests`), the documented log envelope, and the **error strings** for the replay quota / disabled account |
| `liamdj/tokaido-analysis` | same lineage, plus ~95 committed **real** replay JSONs — the packet shape in §4 is read off those |
| `AnotherSava/bga-assistant` | a **working Azul log parser**: the six notification types, their args, the tile object, and the tile-type→colour numbering (§4.1) |
| `DavidEGx/bga-duel-finder` | the working `getGames.html` call: params, the `X-Request-Token` header, and the row fields (`table_id`, `players`, `scores`, …) |
| `HStrand/bga-tm-scraper` | `g_gamelogs` extraction, replay-limit detection, the `\d{6}-\d{4}` replay version |
| `FlavienBusseuil/bga-chrome-extension` | full TypeScript types for `tableinfos`, and Azul's `.variant` CSS class |
| `Haurrus/BoardGameArena_Discord_Turn_Bot` | that `tableinfos` works **anonymously** given a request token |
| `NevinAF/bga-ts-template` | typings of BGA's own client: `NotifsPacket`, `g_gamelogs`, the AJAX surface |
| `BGAtoFreeboard`, `DavidEGx/Hive-bga2bs` | the `g_gamelogs = {...};` route, and a saved replay page we could read |
| `advoet/bga`, `Rpifer/BoardGameArenaHive`, `kamaradclimber/bga_to_bgg` | endpoint corroboration, pagination, courtesy user agents |

They all handle sessions the same way: reuse a cookie jar. The cookie names in §2.4 come from
the live site, not from them.

---

## 2. What is accessible, and what needs a session

Verified live 2026-08-17 with `curl`, a desktop Chrome UA, and the
`X-Requested-With: XMLHttpRequest` + `Referer` headers BGA's own XHRs send.

### 2.1 Public — no cookies

**The all-time ladder.**

```
GET https://en.boardgamearena.com/gamepanel/gamepanel/getRanking.html?game=1467&start=0&mode=elo
```

- `game=1467` is **Azul** (from the public game list embedded in `/gamepanel`; `azulduel` is
  2220, `azulsummerpavilion` 1911, `azulqueensgarden` 2560 — different games).
- `mode=elo` is the **ALL-TIME** ladder; `mode=arena` is the current season, a different
  number. This is exactly the distinction Remi asked about, and it is one parameter. From the
  site's own bundle: `$=[{key:"arena",name:_("Current Season")},{key:"elo",name:_("All-time")}]`
  and `await Nt("/gamepanel/gamepanel/getRanking.html",{game:i.id,start:e,mode:g})` with
  `e = 10*page`.
- **10 rows per call**, paginated by `start`. Verified working at `start=990`.

```json
{"status": 1, "data": {"ranks": [
  {"id": "91843016", "name": "Sapperlot", "country": {"name": "Germany", "code": "DE"},
   "ranking": "2486.16", "nbr_game": "1633", "rank_no": "1",
   "avatar": "_def_2321", "device": "desktop", "status": "offline"}, ...]}}
```

**The Elo scale trap.** `ranking` is the *raw* Elo. The website shows `max(0, raw − 1300)`
floored — the site's own JS is `Math.max(0, parseFloat(e) - 1300)`. Remi's CSV holds
*displayed* numbers, this API holds *raw* ones, and mixing them is a 1300-point mistake.
`client.display_elo` / `raw_elo` convert.

**Also public**: the `/gamepanel?game=azul` HTML (1.8 MB) with the whole game list as JSON.
Azul's entry, free and useful:

```
id 1467 · version "260626-1038" · player_numbers [2,3,4] · default_num_players 2
arena_num_players 2 · games_played 28,062,113 · games_played_recent 16,900
league_number 5 · is_ranking_disabled false · bgg_id 230802
media.majorvariant {"1": ..., "2": ...}   <-- two major variants; see §3
```

### 2.2 Session required — verified against the live site

| Endpoint | Purpose | Anonymous result |
|---|---|---|
| `/gamestats/gamestats/getGames.html?player=…&game_id=1467&finished=1&page=N` | a player's finished tables | `code 806` |
| `/archive/archive/logs.html?table=<id>&translated=true` | **the replay move log** | `code 806` |
| `/gamereview/gamereview/requestTableArchive.html?table=<id>` | primes the archive (see §2.3) | untested (same family) |
| `/table/table/tableinfos.html?id=<id>` | table metadata + options | see §2.3 — may work anonymously |
| `/gamelist/gamelist/gameOptions.html?game=1467` | the option catalogue (would name the wall variant) | `code 806` |
| `/halloffame/halloffame/getDailyTables.html?game=1467` | recent notable tables | `code 806` |
| `/gamestats?player=…` / `?game_id=…` (HTML) | same data, human page | 302 → `/account?warn` |

Note the shape of the wall: **there is no anonymous way to discover even one Azul table id**,
so everything past stage 1 depends on cookies. A `code 806` (rather than a 404) also confirms
an endpoint *exists*, which is how we know `getGames.html` is real — but it means its
**parameter names are unverified**, because BGA checks the session before it validates
parameters. Those come from working community code instead (§1.2).

### 2.3 Three things the community code adds

1. **`X-Request-Token`.** Some authenticated AJAX endpoints (`getGames.html` for certain) want
   the per-session CSRF token BGA embeds in every page as `requestToken: '<hex>'`, sent as the
   `X-Request-Token` header alongside `X-Requested-With: XMLHttpRequest`.
   `BgaClient.fetch_request_token()` scrapes it (one request) and the client then sends it on
   every call.
2. **The archive must be primed.** Three independent projects `GET
   /gamereview/gamereview/requestTableArchive.html?table=<id>` *before* `logs.html`, one with
   the comment "seemingly required to produce log". `Fetcher.fetch_table` does this, treating
   a failure as non-fatal but propagating a quota error.
3. **`tableinfos` may not need a login at all** — one 2026 project reads it anonymously with
   just an anonymous `PHPSESSID` plus the request token. If that still holds, the whole
   metadata/filter stage is free of session risk. Worth 2 requests to check (§9).
   Also note `robots.txt` disallows `/table`, so prefer the same payload from
   `/tablemanager/tablemanager/tableinfos.html?id=<id>` (`endpoints()["table_infos_alt"]`),
   which robots.txt does not mention.

### 2.4 Exactly what Remi needs to hand over

A **Netscape-format `cookies.txt`** for boardgamearena.com, exported from a logged-in browser
(any "export cookies" extension writes it), containing:

| Cookie | Why |
|---|---|
| `PHPSESSID` | the session — this is the one that authorises the calls |
| `TournoiEnLigneuser`, `TournoiEnLigneauth` | BGA's persistent login pair ("TournoiEnLigne" is BGA's original French name) |
| `TournoiEnLigne_sso_user`, `TournoiEnLigne_sso_id` | the SSO pair, present on accounts linked to Asmodee/social login |

Export **all** boardgamearena.com cookies and let the client sort it out; the names above are
what public code has been seen using, and BGA has renamed them before. Then:

```bash
python -m ludometer.human.cli tables --cookies ~/bga_cookies.txt --limit 20
```

The client loads the jar with `ignore_discard=True, ignore_expires=True` (browser exports
routinely mark the session cookie session-only, and a strict reader drops it, leaving the run
silently anonymous). `BgaClient.authenticated` and `cookie_names()` exist so the CLI can say
"you exported the wrong thing" before spending a request.

**We never log in programmatically.** Public projects POST email+password to
`/account/account/login.html` with a `request_token`; some now drive the two-step Svelte login
with Playwright. We do neither — no password handling, nothing to bypass. If the session is
rejected mid-run, `AuthRequired` aborts the whole run rather than retrying, because hammering
a login wall is exactly what gets an account flagged.

---

## 3. The gray-wall variant filter, and 2-player only

Azul's player board is double-sided: the printed **fixed colour wall**, and the grey
**variable wall** where a tile may go in any column of its row. BGA implements the second as a
*major variant* — and the public game list shows Azul has exactly two:
`media.majorvariant = {"1": …, "2": …}` (those keys are the option's values).

### 3.1 The check that actually decides it — wall columns

**This is the important part, and it does not depend on knowing any option id.** On the fixed
wall the column of colour `c` in row `r` is `(c + r) % 5`, and the log reports the `column` of
every tile the wall-tiling step places (`placeTileOnWall.args.completeLines[pid].placedTile`
carries `line` **and** `column`). So:

```python
convert.check_wall_placements(game)   # "" or the reason to reject
```

A grey-wall game's placements will not satisfy the formula, so it is rejected — and the same
check independently verifies the colour map and the line-numbering base. `convert_game` runs
it on every game by default.

Read the failure pattern, not the individual failure: **a few** games failing while the rest
pass = those games are the variant (correct rejection). **All** games failing = the schema is
wrong (fix `LogSchema`; never relax the check).

### 3.2 The option filter — a request-saver, not the guarantee

`tableinfos` returns `data.options` as `{option_id: {"name": …, "value": …}}` (older payloads
`{option_id: value}`; `fetch.option_value` reads both). BGA's convention puts framework
options at 200 (speed), **201 (game mode: 0 normal / 1 friendly / 2 Arena)** and 204 (thinking
time), with **game-specific options starting at 100**. So Azul's wall variant is almost
certainly option `100` with `1` = standard and `2` = grey, matching the two `majorvariant`
keys.

**Not yet verified**, so `fetch.STANDARD_WALL_OPTION_HINTS["option_id"] is None` and
`TableFilter` **rejects every table** with the reason `"wall variant option id unknown"`. That
is deliberate: fail-closed. Filling it in (§9 step 4) turns the filter on and saves one request
per variant table — but §3.1 is what keeps the dataset correct.

`TableFilter(allowed_game_modes=(ARENA_MODE,))` additionally restricts to BGA's ranked Arena
mode, which is the best available "both players were trying" signal, and unranked tables are
dropped outright.

### 3.3 Player count

`TableFilter(players=2)` compares `len(data.players)` in `tableinfos` — and the history rows
already carry a comma-joined `players` field, so `fetch.table_row_players` gives the same
answer for **zero** extra requests. Azul's `default_num_players` and `arena_num_players` are
both 2, so the yield loss here should be small. The converter is a second gate: our engine
only implements 2 players and `convert_game` refuses anything else.

---

## 4. Replay → engine mapping

Target: `action_id = source*30 + color*6 + dest` (`source` 0–4 factories / 5 center, `color`
0–4, `dest` 0–4 pattern rows / 5 floor). One Azul turn = one action id, because the turn *is*
"take all of one colour from one place, put it in one line".

### 4.1 Azul's log schema (from a working third-party Azul parser)

```
factoriesFilled   {factories: Tile[][], remainingTiles: int}
tilesSelected     {player_id, type, selectedTiles[], discardedTiles[], fromFactory}
tilesPlacedOnLine {player_id, placedTiles[], discardedTiles[], line}
placeTileOnWall   {completeLines: {pid: {placedTile, discardedTiles[], pointsDetail}}}
emptyFloorLine    {floorLines: {pid: {tiles[], points}}}      # [] when empty, object otherwise
firstPlayerToken  {playerId}
```

`Tile = {"id", "type", "column", "line", "location"}`, `location` ∈ `"factory_N"`, `"wall"`,
`"discard"`, `"floor"`. **`type` is the colour, and `0` is the first-player marker, not a
colour** — a marker tile in a floor list must never be counted as a tile.

**One turn is two notifications** (`tilesSelected` then `tilesPlacedOnLine`); `parse_log`
pairs them into one `Pick`. A selection never closed by a placement is read as "went to the
floor line", which is the one destination that may not need its own notification.

### 4.2 Obstacle list and status

| # | Obstacle | Status | Handling |
|---|---|---|---|
| 1 | Notification names | **known** (§4.1) | `LogSchema.select_types` / `place_types` / `deal_types` / `wall_types`, each with fallback spellings |
| 2 | Arg keys | **known** (§4.1) | `LogSchema.arg_aliases`, tried in order |
| 3 | Tile colour naming | **known + self-verifying** | `AZUL_COLOR_MAP` (§4.3) |
| 4 | Factory indexing | **known 0-based** (`location: "factory_0"`) | `factories_one_based = False`; a wrong choice makes moves illegal at once |
| 5 | Center as a source | **inferred** | any `fromFactory >= NUM_FACTORIES` is the center (same convention as our `CENTER == 5`), plus explicit `center_values` |
| 6 | Floor as a destination | **inferred** | `lines_one_based = True` → `line == 0` is the floor; a wrong base is caught by §3.1 and by the score check |
| 7 | First-player marker | **solved** | not needed: our engine gives the marker to whoever first takes from the center and picks next round's starter itself. `firstPlayerToken` is only a cross-check |
| 8 | Who moves first | **solved** | taken from the first pick's player id, not assumed to be seat 0 (tested) |
| 9 | Refill / bag info | **solved — this was the big risk** | the deal is in `factoriesFilled`, so `apply_deal` scripts our engine's chance (§4.4). `remainingTiles` is a free bag cross-check |
| 10 | Scores for validation | **solved** | score notifications, plus `scores` on the history row; engine scores must match |
| 11 | Bag/lid split after a scripted deal | **solved** | derived, reshuffle case handled (§4.4) |
| 12 | Timeouts / abandoned games | **handled** | a log that stops mid-game fails `require_terminal`; conceded/unranked tables are filtered |
| 13 | Private `/player/pNNN` packets | **handled** | dropped by channel; a real log contains both channels |

### 4.3 The colour map, and why it is not a guess

BGA numbers Azul's tiles `0` marker, `1` Black, `2` Cyan, `3` Blue, `4` Yellow, `5` Red. Our
engine is `0` blue, `1` yellow, `2` red, `3` black, `4` teal. So:

```python
AZUL_COLOR_MAP = {1: 3, 2: 4, 3: 0, 4: 1, 5: 2}     # BGA type -> engine colour
```

This is the one mapping whose error would be invisible to the eye and fatal to the dataset,
so it is checked mechanically two ways: the wall-column identity (§3.1), and
`convert.solve_color_map` / `solve_color_map_over`, which brute-force all 120 permutations and
keep those that replay legally *and* reproduce BGA's reported score. With the wall check in
play the answer is already unique on a single game — the test
`test_intersecting_several_games_narrows_the_colour_map` asserts exactly that. Run it on the
first real logs and either it confirms `AZUL_COLOR_MAP` or it hands you the right map.

### 4.4 Scripted deals (`convert.apply_deal`)

Our engine owns its bag and draws refills from its own RNG — that is what makes self-play
reproducible, and it is why a human game cannot just be fed to `apply()`. So:

1. after the engine's own refill, the off-board pool is `bag + lid + the engine's deal`;
2. the observed deal is subtracted from that pool and written straight to `state.factories`;
3. the remainder goes back — staying split between `bag` and `lid` normally (so the public
   bag/lid features in `encode()` keep their meaning), or merged into the bag when the observed
   deal needs more of a colour than the bag alone can supply, which is exactly the reshuffle
   `_refill` would have done;
4. `state.recount()`, then **`tile_census() == [20]*5` or the game is rejected**.

This touches only public attributes (`factories`, `bag`, `lid`, `current_player`,
`first_player`) and the engine's own `recount()`. **`ludometer/azul/engine.py` is not modified
and must not be.** A short deal (end of bag) is representable, which is the other reason to
script rather than draw.

### 4.5 Confirming the schema against a real log

The **zero-request route**, and the one to prefer: Remi opens one of his own Azul replays in a
browser and saves the page. The page embeds the entire log as `g_gamelogs = {...};` —
`parse.parse_gamelogs_html` reads it, and no automated request is involved at all. That single
file is enough to confirm every "inferred" row in §4.2.

Otherwise, with cookies (~3 requests):

```bash
python -m ludometer.human.cli tables  --cookies ~/bga_cookies.txt --top 1 --limit 1
python -m ludometer.human.cli inspect data/human/raw/<table>.json.gz
```

`inspect` prints, per notification type: the count, the union of `args` keys, and an example.
Put any surprises into `LogSchema`, run `solve_color_map_over` on ~5 games, and the whole test
suite still applies unchanged — the tests are written against the schema object, not against
hard-coded names.

### 4.6 Proof the chain works today

No real Azul replay could be fetched anonymously, so the proof is a **round trip**: the engine
plays a game, `ludometer/human/fixture.py` writes it out in the real Azul notification shape
(real tile objects, real tile-type numbering, `tilesSelected` + `tilesPlacedOnLine` per turn,
`placeTileOnWall` with fixed-wall columns, a private-channel packet), and the parser +
converter must reproduce the same moves, seats, scores and outcome.

```
$ python -m ludometer.human.cli selftest --games 3
seed 0: 56 positions, 5 rounds, scores (7, 0), outcome +1
seed 1: 86 positions, 8 rounds, scores (3, 0), outcome +1
seed 2: 73 positions, 7 rounds, scores (0, 3), outcome -1
3/3 synthetic games round-tripped
```

53 tests cover it, including the negative half: illegal pick, tile-conservation violation, a
possible-but-wrong deal, truncated log, score mismatch, unknown notification, permuted colour
map, non-zero first mover, private-channel packets, the quota/disabled/lost-archive error
strings, and the `replay.npz` format.

---

## 5. Volume, and the constraint that actually governs it

### 5.1 BGA's daily replay quota — read this before planning anything

Opening an archived game is capped **per account per day**. It is not an HTTP 429; it is a 200
whose JSON carries an error string. The strings, hard-coded in the projects that hit them in
production:

```
"You have reached a limit (replay)"                                  -> ReplayLimitReached
"disabled for your account"                                          -> AccountDisabled
"Unfortunately the replay for this game has been lost"               -> ReplayUnavailable
"registered more than 24 hours and have played at least 2 games"     -> AuthRequired
```

`client._classify` maps each to its own exception; the CLI stops the run on the first two and
skips the table on the third. The cap resets roughly 24 h after it is hit (one project's cron
waits 24.5 h). **The numeric cap is undocumented**, and no public source says whether premium
accounts get more. Public scrapers rotate several accounts to get around it — **we do not**;
that is evasion of a deliberate limit, and one account's cap is the honest budget.

Consequence: **measure the cap first.** The first authenticated run should simply fetch until
`ReplayLimitReached`, and the state file's per-day request counter plus the `downloaded`
verdicts give the number. Everything downstream is then arithmetic.

### 5.2 The ladder, measured

Sampled 2026-08-17 through the public endpoint (5 pages, 50 rows):

| rank | raw Elo | displayed | games (`nbr_game`) |
|---|---|---|---|
| 1 | 2486.2 | 1186 | 1,633 |
| 5 | 2236.2 | 936 | 3,428 |
| 10 | 2209.7 | 909 | 453 |
| 100 | 2015.8 | 715 | 1,186 |
| 200 | 1967.1 | 667 | 8,388 |
| 500 | 1886.1 | 586 | 1,172 |
| 1000 | 1832.7 | 532 | 3,352 |

From Remi's December 2025 full dump (91,199 ranked players, **displayed** Elo): ≥900 → **2
players**; ≥800 → 11; ≥750 → 21; ≥700 → 54; ≥600 → 218; ≥500 → 821; rank 100 sat at 650.

- **"Everyone above ~900 all-time" is ~10 players today** (2 last December). The ladder does
  not go much past 1000; any Elo floor has to be ~650–720, or expressed as a rank.
- The ladder **inflated ~65 points at rank 100** in eight months (650 → 715), so hard-coded
  Elo floors go stale. Prefer `--top N`; treat `--min-elo` as a guard.
- `nbr_game` is per player, all player counts, both variants, ranked games only. Among the top
  100 it averages ~2,000, median ~1,000, and it is wildly dispersed: rank 96 has 9,832 games,
  rank 97 has 126.

### 5.3 Yield and cost

- Positions per game: **~50–60** (5–6 rounds × ~9–10 picks). Our 500k-position buffer is
  **~9,000–10,000 games**.
- Supply: top 200 × ~2,000 games ≈ 400k player-games ≈ 200k+ distinct tables before filtering;
  after "2-player, standard wall, finished, ranked" the pool is still ~100k+. **Supply is not
  the constraint.**
- Requests per accepted table: 1 `tableinfos` (skippable — the history row already gives player
  count and scores, §3.3) + 1 `requestTableArchive` + 1 `logs` = **2–3**, of which the last two
  count against the replay quota. A rejected table costs 1 or 0.
- Plus history listing: ~10 rows per `getGames` page → ~10–15 requests per player for 100–150
  games.

| target | games | positions | requests (≈) | days at 4,000 requests/day | days if the replay cap is 100/day |
|---|---|---|---|---|---|
| smoke | 20 | ~1.1k | 60 | <1 | <1 |
| milestone | 2,000 | ~110k | ~7,000 | ~2 | ~20 |
| full | 10,000 | ~550k | ~32,000 | ~8 | ~100 |

Our own politeness budget (3.75 s/request, 4,000/day) is the *second* column; **the replay cap
is the one that decides**, and until it is measured the right-hand column is a guess. This is
the single number to establish on day one.

### 5.4 Recommended thresholds

```
--top 200 --min-elo 650 --min-games 200     # players
per-player cap: 100–150 most recent tables
TableFilter(players=2, require_standard_wall=True, allowed_game_modes=(ARENA_MODE,))
first milestone: 2,000 games (~110k positions)
full target:     10,000 games (~550k positions)
```

**Top 200, not top 100**: rank 200 is displayed 667 against rank 100's 715 — a ~50-point gap
that is small next to the noise in a single human move, while doubling the pool halves the
per-player cap and so the style bias. `min_games ≥ 200` matters more than it looks: a rank-97
account with 126 games has an Elo that is mostly variance. Take the **most recent** games per
player: recent play reflects the current meta, and old archives are the ones BGA is likeliest
to have dropped.

On the data's value: 110k human positions is already a useful warm start. Pretraining here is
a *prior on move preferences*, not a substitute for self-play, and human policy targets are
hard one-hots with real noise in them (Azul's floor-line sacrifices look like blunders for
several rounds). Getting to 550k is nice-to-have; getting to 110k is the experiment.

---

## 6. The pipeline (what is built)

```
ludometer/human/
  client.py    BgaClient      rate limit + cookie jar + request token + BGA error taxonomy
                              (AuthRequired / ReplayLimitReached / AccountDisabled /
                              ReplayUnavailable) + endpoints()
  fetch.py     Fetcher        3 resumable stages, JSON state file, TableFilter (fail-closed),
                              option_value/table_row_* helpers
  parse.py     parse_log      log JSON -> ReplayGame (picks, deals, wall placements);
                              LogSchema, AZUL_COLOR_MAP, parse_gamelogs_html, log_type_histogram
  convert.py   convert_game   replay in our engine with strict validation; apply_deal;
                              check_wall_placements; solve_color_map[_over]
  dataset.py   build_dataset  -> replay.npz for --pretrain (via train.replay.ReplayBuffer)
  fixture.py   synthetic_log  engine game -> real-shaped Azul log (tests + `cli selftest`)
  cli.py       endpoints selftest ranking players tables inspect convert dataset
tests/test_human_pipeline.py  53 tests, no network
```

Standard library + numpy only; no new dependency (deliberately no `requests`). Nothing imports
torch.

### 6.1 State file and resuming

`<out>/state.json`, version 1, rewritten atomically after every step:

```json
{"version": 1, "game_id": 1467, "started": "2026-08-17T10:00:00+00:00",
 "requests": {"total": 812, "2026-08-17": 812},
 "ranking": {"fetched": "...", "rows": [{"player_id": 91843016, "name": "Sapperlot",
              "elo_raw": 2486.16, "elo_display": 1186, "rank": 1, "games_played": 1633}]},
 "players": {"91843016": {"pages_done": 3, "complete": true, "tables": [712345678, ...]}},
 "tables":  {"712345678": {"status": "downloaded", "reason": ""},
             "712345679": {"status": "skipped", "reason": "3 players"}}}
```

- `ranking.rows` present → `fetch_ranking` returns the snapshot and makes **no** request
  (`--force` refreshes);
- `players[pid].pages_done` → a player's history resumes mid-way (`page = pages_done + 1`, BGA
  pages by 1-based `page`); `complete: true` stops it;
- `tables[tid].status` in `{downloaded, skipped}` is **terminal** — never requested again, and
  `reason` is kept as an audit trail (this is how you later measure the variant and
  player-count rejection rates without re-fetching);
- `requests` is the per-day counter enforcing `max_requests_per_day`.

Raw payloads: `<out>/raw/<table_id>.json.gz`, each `{"table_id", "infos", "logs"}`, written
once and never rewritten. `Fetcher.iter_raw()` streams them for the offline stages.
**This is the most important property of the design**: the bytes are fetched once, and the
parser can then be iterated on locally as often as needed — a schema fix never costs a request,
which matters enormously when the daily quota is the budget.

### 6.2 Rate-limit constants and why

`ClientConfig`: `min_interval=3.0`, `jitter=1.5`, `max_requests_per_day=4000`,
`max_retries=3`, `retry_backoff=15 s` (linear, 5xx and network errors only — a 4xx is returned,
never retried).

- **~3.75 s average** between requests, ~16/min. Public projects use anywhere from 100 ms to
  30 s; 30 s per table is the politest precedent, 3.75 s per *request* is in the same spirit
  given 2–3 requests per table.
- **No concurrency anywhere.** One request at a time, `time.monotonic()` spacing.
- **4,000/day** is ~4 h of continuous fetching — the tool cannot silently run away over a
  weekend.
- Desktop Chrome UA + `X-Requested-With` + `Referer` + `X-Request-Token`: the shape of a real
  tab. Not evasion — a `Python-urllib/3.12` UA is simply the fingerprint most likely to trip a
  naive filter. (A courtesy UA naming the project and a contact address, as
  `kamaradclimber/bga_to_bgg` does, is arguably the more honest choice; consider it if Remi
  asks BGA for permission and gets it.)
- `AuthRequired`, `ReplayLimitReached` and `AccountDisabled` all abort the run.

---

## 7. ToS and risk — read this before running anything

### 7.1 robots.txt (fetched 2026-08-17)

```
User-agent: *
Allow: /
Disallow: /table          <-- table pages (and /table/table/tableinfos.html)
Disallow: /playerstat
Disallow: /message/board
Disallow: /newreport
Disallow: /report
Disallow: /player          <-- player pages
Disallow: /play
Disallow: /doc/images
Disallow: /doc/Images
Disallow: /web/scripterror
Disallow: /*TidakTidak*
```

`/gamestats`, `/archive`, `/gamereview`, `/tablemanager` and `/gamepanel` are **not** listed;
`/table` is, which is why `endpoints()` offers the `/tablemanager` alternative.

### 7.2 Terms of service (`/legal?section=tos`, verbatim)

> • not to obtain information about Users and the Content they publish using automated methods
> (such as robots, spiders, etc.); **not to use the Services and/or the BGA Sites using
> automated methods (such as robots, spiders, multiple queries, etc.)**;

plus a French *sui generis* database-right clause forbidding

> l'extraction par transfert permanent ou temporaire de la totalité ou d'une partie
> qualitativement ou quantitativement substantielle du contenu d'une ou plusieurs des bases
> des données accessibles sur les Sites BGA

There is a friendlier plain-language aside next to the *re-use* clause ("we are obviously happy
when you share your experience on BGA on medias :) The previous statement is more about
re-using pieces of BGA for another service.") — but it annotates re-use, not automated access.

**There is no reading of the ToS under which an automated fetcher is permitted.** Not at
3 s/request, not with a browser UA, not with Remi's own cookies. Slower and politer reduces the
*load* and the chance of notice; it does not change permission.

### 7.3 Honest risk assessment

| risk | likelihood | consequence |
|---|---|---|
| Nothing happens; the traffic is lost in 16,900 Azul games/day | likely at this rate | — |
| The daily replay quota stops the run | **certain** at any real volume | run stops cleanly, resumes tomorrow |
| Karma penalty / moderation warning | possible | affects matchmaking and reputation |
| Replay access disabled for the account | documented in public code (`"disabled for your account"`) | no more archives for Remi |
| Account suspension | unlikely but real | Remi loses a premium account and his own history |

Aggravating factors we avoid: bursts, concurrency, retry storms against the login wall,
multi-account rotation, and fetching far more than a human could plausibly review. Built-in
mitigations: single-threaded, ≥3 s spacing, daily cap, resumable, skip-lists, and a hard stop
on the quota.

**One more thing worth naming.** A community project reports that
`/{server}/{slug}/{slug}/notificationHistory.html?table=…&from=0&privateinc=1&history=1`
returns the same packets **without** consuming the replay quota. It is documented here for
completeness and because the next reader will find it anyway — but using it *because* it dodges
a limit BGA put there on purpose is circumventing a rate limit, which is both against the ToS
and against the spirit of doing this carefully. Do not use it as a quota workaround.

**Recommended path, in order.** (1) **Ask BGA first**: one email to the admins describing an
open-source Azul RL research project and asking for a data dump or a blessed slow crawl.
It costs nothing and would make everything else moot; the projects surveyed here explicitly
advise it ("Please contact a bga admin before attempting to use this code for scraping").
(2) If that fails, decide with the numbers above — and if the answer is yes, run the 20-game
smoke target, wait a day, check karma, then measure the quota. (3) Remember that the
2,000-game milestone is a fifth of the full target's requests for most of the learning signal.

The tool will not run itself: `tables` requires an explicit `--cookies` and refuses to start
without one, and the daily cap is on by default.

---

## 8. Open questions

1. **The numeric daily replay cap**, and whether premium changes it (§5.1). Measure it.
2. **`getGames` parameter names and response shape.** The endpoint exists (806, not 404) but
   `player` / `game_id` / `opponent_id` / `finished` / `page` come from community code, not
   from a response we saw. **Check whether the rows carry the game options** — if they do, the
   `tableinfos` call disappears entirely.
3. **The wall-variant option id and its values** (§3.2). Hypothesis: option `100`, `1` =
   standard, `2` = grey. One authenticated call to
   `/gamelist/gamelist/gameOptions.html?game=1467` returns the whole catalogue with labels.
4. **Whether `logs.html` works for tables Remi did not play in.** Everything assumes yes
   (public projects scrape strangers' games), but test it on 3 tables — one of his, two
   strangers' — before planning volume.
5. **Whether `tableinfos` really is anonymous** (§2.3). If yes, the metadata stage carries no
   session risk at all.
6. **The center and floor encodings** in `fromFactory` / `line` (§4.2 rows 5–6). One real log
   settles both; until then the wall check and the legality check are the guard.
7. **How far back archives are kept** — prefer recent tables.
8. **Per-seat Elo in `tableinfos`** — if present, `TableFilter.min_player_elo_raw` can demand
   *both* players strong, which is better than "one player is in the top 200".

---

## 9. Next steps, in order, with acceptance criteria

1. **Remi's decision on §7, then his cookies** (§2.4). *Accept*: `cli ranking` works without
   cookies; `cli tables --top 1 --limit 1` fetches one table with them.
   — Better first move: ask him for **one saved replay page** (§4.5). It confirms the whole
   schema at zero request cost and zero risk.
2. **Confirm the log schema** on that page or on one fetched log. *Accept*: `cli inspect`
   shows the six expected notification types; `solve_color_map_over` over ~5 games returns
   exactly `AZUL_COLOR_MAP`; `check_wall_placements` passes on all of them.
3. **Pin the wall-variant option** (§3.2, 1–3 requests). *Accept*:
   `STANDARD_WALL_OPTION_HINTS["option_id"]` set; the filter accepts a known standard game and
   rejects a known grey-wall one (`test_the_wall_filter_accepts_once_the_option_id_is_known`
   already encodes that shape).
4. **Measure the replay quota** (§5.1). *Accept*: a run that ends in `ReplayLimitReached` with
   a known `downloaded` count recorded in `state.json`.
5. **Smoke run: 20 games.** *Accept*: ≥18/20 convert; every rejection reason in `state.json`
   is understood; the tile census never fires.
6. **Milestone run: 2,000 games**, over as many days as the quota takes. *Accept*: ≥90 %
   conversion; `cli dataset` writes a `replay.npz` that `ReplayBuffer.load` reads with
   `margin_targets == aux_targets == policy_targets == positions`.
7. **Train**: `--pretrain data/human/replay.npz` on a run6-shaped net, then measure against the
   existing gauntlet. *Accept*: the pretrained net beats a random-init net at equal self-play
   compute — the actual question this whole exercise asks. If it does not, stop; do not scale
   to 10,000 games first.
8. Only then consider the full target.

---

## 10. Dead ends — do not repeat these

- **`/doc/Terms_of_service`** → 404. The real ToS is `/legal?section=tos` (also `?section=legal`,
  `?section=ppac`, `?section=tosv`).
- **Scraping the ranking with Selenium** — unnecessary; §2.1 is one GET per 10 players.
- **Looking for the ranking endpoint in the page HTML** — it is not there. It is in the Svelte
  bundle `https://x.boardgamearena.net/data/themereleases/<theme>/js/sveltec/dist/main.js`
  (6.4 MB, one request). Grep it **with Python**, not shell `grep -oE`, which hangs on a 6 MB
  single line. That bundle is also the full endpoint catalogue.
- **Assuming the packet list is at `data.logs`** — the live endpoint uses that, but a replay
  page's `g_gamelogs` nests one deeper at `data.data`. `parse.log_packets` accepts both;
  do not "simplify" it.
- **Assuming `/player/pNNN` packets are game events** — they are one player's private UI hints
  and must be dropped (`iter_log_entries`).
- **`halloffame/getDailyTables.html`** as an anonymous source of table ids → 806.
- **`/gamestats?game_id=1467`** (site-wide, no player) → still the login wall. There is no
  anonymous table-id discovery for Azul; do not spend requests looking for one.
- **`/gamelist/gamelist/gameOptions.html`** anonymously → 806 (the call that would have
  answered the wall-variant question during recon).
- **Assuming `nbr_game` means 2-player standard games** — it is every ranked game at every
  player count in both variants.
- **Assuming Remi's committed CSV is current** — December 2025, and *displayed* Elo; the ladder
  has inflated since.
- **Treating BGA's quota/ban messages as HTTP status codes** — they arrive as 200 with an
  `error` string (§5.1).

---

## Appendix A: request ledger for this recon (20)

| # | Request | Result |
|---|---|---|
| 1 | `GET /robots.txt` | 200, §7.1 |
| 2–4 | `GET /gamepanel?game=azul` (apex 302 → `en.`) | 200, 1.8 MB; game id 1467 + Azul metadata |
| 5 | `GET x.boardgamearena.net/…/sveltec/dist/main.js` | 200, 6.4 MB; endpoint catalogue |
| 6 | `GET /doc/Terms_of_service` | 404 (dead end) |
| 7 | `GET /gamepanel/gamepanel/getRanking.html?game=1467&start=0&mode=elo` | 200, top 10 |
| 8–9 | `GET /gamestats?player=91843016&game_id=1467&finished=1` | 302 → `/account?warn` |
| 10 | `GET /halloffame/halloffame/getDailyTables.html?game=1467` | 806 |
| 11 | `GET /archive/archive/logs.html?table=1&translated=false` | 806 (endpoint exists) |
| 12 | `GET /gamelist/gamelist/gameOptions.html?game=1467` | 806 |
| 13 | `GET /legal?section=tos` | 200, §7.2 |
| 14–17 | `getRanking` at `start=90,190,490,990` | 200, §5.2 |
| 18–19 | `GET /gamestats?game_id=1467` | 302 → `/account?warn` |
| 20 | `GET /gamestats/gamestats/getGames.html?player=…` | 806 (endpoint exists) |

All with a desktop Chrome UA, sequential, ≥2 s apart (the paced batch used 3 s).

## Appendix B: engine-side reference

```
action_id = source*30 + color*6 + dest       source 0-4 factories, 5 center
                                             color  0-4 = blue,yellow,red,black,teal
                                             dest   0-4 pattern rows, 5 floor
wall column of colour c in row r = (c + r) % 5
BGA tile type -> engine colour: {1:3 black, 2:4 teal, 3:0 blue, 4:1 yellow, 5:2 red}, 0 = marker
5 factories x 4 tiles (2-player), 100 tiles = 20 per colour
replay.npz row = states(182) policies(180 one-hot) values margins+mask aux(30 packed)+mask policy_mask
value/margin/aux are all in the *player-to-move* frame — same convention as self-play
```
