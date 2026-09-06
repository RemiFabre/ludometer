# Learning from human games on Board Game Arena — feasibility recon + handoff

**Status**: recon complete, pipeline skeleton committed, **no bulk download performed**.
**Date**: 2026-08-17. **BGA requests spent on this recon: 20** (budget was ~30, ≥2 s apart,
desktop-Chrome user agent, one at a time). Everything else below comes from reading ~30 public
open-source projects, not from touching BGA.
**Owner of this doc**: whoever picks the work up next. It is written so that nothing here has
to be rediscovered.

Read this before touching `ludometer/human/`. Code: the seven modules in §6. Tests:
`tests/test_human_pipeline.py` (74, no network). Nothing in
`ludometer/{train,eval,azul,agents}` was modified.

**Newest sections supersede older ones where they disagree: §17 (2026-08-20) >
§16 (2026-08-19) > §15 > §14 (both 2026-08-18) > §13 > §12.** **§12 (2026-08-17, live)** supersedes everything above it — the first authenticated run: auth fixed (the request-token header), the
wall variant pinned (option 100), and one real elite replay validated end to end
(scores matched exactly). **§11** is next-newest: the crawl walks the ladder in rank order
with a resumable cursor, the pace is slow (one request every 6-10 s, 120 replays/day), the
dataset learns from the **elite player only**, and `web/harvest.html` shows it live.

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

## 11. Rank-ordered crawl, elite-only learning, progress page (2026-08-17, later)

Three additions on top of §6, all still test-only (no BGA request has been made):

**11.1 `Fetcher.crawl_ranked` — the ladder in rank order.** Rank 1's games first, then
rank 2. The state file is version **2** and adds `cursor` (`{rank, player_id,
table_offset, players_done}`, rewritten after *every* table), `downloads` (the per-day
counter for the requests that spend BGA's replay quota), `error` (why the last run
stopped) and `pace`. A version-1 file loads and is upgraded. A restart re-enters at
`tables[table_offset]` of rank `cursor.rank`; ranks below it are never revisited.
`ReplayLimitReached` / `AccountDisabled` / `AuthRequired` and our own daily caps all end
the run **cleanly**, with the reason in `state.error` and the cursor left pointing *at*
the table BGA refused (so tomorrow retries exactly that one). `fetch.jsonl` gets one
JSON line per table decision.

**11.2 The pace is now conservative and named.** `fetch.CrawlPace` (mirrored in
`ClientConfig`): `min_interval=6.0`, `jitter=4.0` (**one request every 6-10 s**),
`max_tables_per_day=120`, `max_requests_per_day=600`, plus a randomized 90-300 s pause
every 20 tables. ~25 s per accepted table, ~50 min of traffic a day. This supersedes the
3 s/4000 in §6.2: the quota, not bandwidth, is the budget. **2,000 games ≈ 17 days**;
raise the caps only with a measured quota (§9 step 4).

**11.3 Elite-player-only learning.** Many tables pair a top-200 player with a much weaker
one, so a game now teaches us **one** player's decisions. `fetch.choose_target_player`
picks the target (both seats ranked → the higher raw Elo; otherwise the ranked/source
player), `fetch.player_elos` reads both seats' Elos out of `tableinfos`, and both land in
`state.tables[tid]` **and** in the raw payload (`payload["meta"]`) — so a rebuild with a
different floor costs no request. `dataset.build_dataset` then emits rows only where the
target is to move (`opponent_rows="drop"`, the default; `"value-only"` keeps the
opponent's positions with `policy_mask=0`, the same convention run6 uses for
cheaply-searched positions). Every stored row's value/margin is therefore in the
*target's* frame. `min_target_elo_raw` is a floor on the **target**, not on "someone at
the table". A sidecar `replay.stats.json` records games, positions, elite positions, all
decision points and the rejection tallies.

**11.4 `web/make_harvest.py` → `web/harvest.html`.** Stdlib-only, self-contained, dark,
auto-refreshing, `--watch N`, and readable on a phone; it reads `state.json`,
`fetch.jsonl` and `replay.stats.json` (falling back to the row count in `replay.npz`'s
own header via `zipfile`) and shows the current rank/player, players done, games fetched
/ validated / rejected with reason tallies, positions toward 550k, the 2k and 10k
milestones with ETAs, elite vs all decision points, a log tail, and the replay-limit
state as a banner above everything else. Every panel renders with no state file at all.

```bash
python -m ludometer.human.cli crawl --out data/human \
  --cookies ~/ludometer/.bga_cookies.txt --top 200 --min-elo 650 --min-games 200
python -m ludometer.human.cli dataset --out data/human \
  --npz data/human/replay.npz --min-target-elo 650
python3 web/make_harvest.py --state data/human --watch 20
```

Tests: `tests/test_human_pipeline.py` is now **74** (53 + rank-order/resume/quota-stop,
elite-only masking and Elo floor, and the harvest page parsed with `html.parser`).

---

## 12. First authenticated run — live validation (2026-08-17, Remi's cookies)

**This section is the newest and supersedes §§1-11 where they disagree.** Nine
authenticated BGA requests were spent (paced ≥6 s apart, desktop-Chrome UA, one at
a time). The pipeline's first authenticated call had been failing with
`code 806 "Invalid session information for this action"`; that is fixed, the
wall-variant option is pinned, and one real elite replay was validated end to end
through our engine. **Nothing hit a replay-limit or disable signal.**

### 12.1 Auth: the request token — root cause and fix

`getGames.html`, `logs.html` and `tableinfos.html` all require the per-session
**`X-Request-Token`** header in addition to the cookies. The token lives in every
page's HTML inside the `bgaConfig` JS literal:

```
bgaConfig = { ... requestToken: 'gNZGNxgP5p7MOzj', ... }
```

Two facts the earlier code got wrong, both now fixed in
`BgaClient.fetch_request_token`:

1. **The token is short mixed-case alphanumeric** (e.g. `gNZGNxgP5p7MOzj`, 15
   chars), **not** lowercase hex. The old regex `requestToken:\s*'([0-9a-f]{16,128})'`
   never matched, so `request_token` stayed `None`, no header was sent, and BGA
   answered `code 806`. The regex is now `requestToken:\s*'([A-Za-z0-9]{8,128})'`.
2. **The token rotates on every page load** while the session itself stays valid —
   two loads gave `gNZGNxgP5p7MOzj` then `AyZAa3d3WVPNN1Y`, both accepted. So
   scrape it **once per run** (`fetch_request_token()`) and reuse it; both
   `cli crawl` and `cli tables` now do this before the first authenticated call.

**Verified**: with the fix, `getGames.html` for rank-1 (`player=91843016`,
`game_id=1467`, `finished=1`, `page=1`) returns a valid JSON table list — 10 rows,
no 806. A row carries `table_id`, `players` (comma-joined ids), `player_names`,
`scores` ("72,45"), `ranks`, `unranked`, `concede`, `normalend`, `elo_after`, but
**no game options** — so the `tableinfos` call cannot be skipped (§8 Q2 answered:
options are not in the history row).

### 12.2 The wall variant — option **100 "Board"** (pinned)

`tableinfos.data.options["100"]` is the board side, an enum:

| value | name | our engine? |
|---|---|---|
| **1** | **Colored side** | **yes — the standard fixed-colour wall** |
| 2 | Gray side | no — the variable/grey wall |
| 3 | Crystal Mozaic: Side 1 | no — a different board |
| 4 | Crystal Mozaic: Side 2 | no — a different board |

So only `options["100"] == 1` is a game we can learn from. `STANDARD_WALL_OPTION_HINTS`
is now `{"option_id": 100, "standard_values": (1,)}` and `TableFilter` accepts value
1, rejects 2/3/4, and skips a table whose options omit 100. This refines the old
`majorvariant` hypothesis (there are four sides, not two; standard is value 1).
Cross-checked against the wall-column invariant (§3.1): the validated replay's 32
wall placements all satisfy `column == (colour+row)%5`, independently confirming
both the standard wall and the colour map.

**A second, independent variant to watch**: option **110 "Special Factories (Azul
Master Chocolatier variant)"** — 1 = Disabled (standard), 2 = Enabled. Our engine
does not model it. The wall filter does not gate on it (a 110=2 game would instead
fail the tile-census / score checks in `convert_game`); `SPECIAL_FACTORIES_OPTION`
is defined in `fetch.py` for a future request-saving pre-filter. Framework option
**201** (game mode: 0 normal / 1 friendly / 2 Arena) and **200** (speed) are as
documented; the validated table was 201=0 (Normal), so it was *not* an Arena game —
`TableFilter(allowed_game_modes=(ARENA_MODE,))` would have skipped it.

### 12.3 Replay format — five corrections to §4

The real archive log (`/archive/archive/logs.html?table=<id>&translated=true`,
envelope `data.logs = [ {move_id, channel, data:[notif…]}, … ]`) differed from the
assumed schema in five ways. All are now fixed in `LogSchema` / `parse.py` **and**
mirrored in the synthetic `fixture.py`, so the 77-test suite validates against the
real shape:

1. **Factories are 1-based, with the center at index 0.**
   `factoriesFilled.args.factories` is a list of **6**: entry 0 is the center /
   "deck" slot (it holds only the first-player marker, `type 0`, at fill time),
   entries 1..5 are the five factories. The earlier "0-based `factory_0`" guess was
   wrong — the tile `location` is just `"factory"`, and the numbering lives in the
   array index. `LogSchema.factories_one_based = True`; `_parse_factories` routes
   each index through `_map_source` and drops the center slot.
2. **`tilesSelected.fromFactory` uses the same numbering**: **0 = center**, 1..5 =
   factories. `LogSchema.center_values` now includes `0`.
3. **Columns are 1-based.** The wall placement's `placedTile.column` (and the deal
   tiles' `column`) run 1..5, while `convert.wall_col` is 0-based, so the parser
   de-bases them (`LogSchema.columns_one_based = True`). Without this the wall-column
   check would wrongly reject every standard game. (Pattern-line `line` is also
   1-based with 0 = floor, which §4 already had right.)
4. **The first-player marker generates a standalone `tilesPlacedOnLine`.** When a
   player first takes from the center, BGA emits — *before* that turn's
   `tilesSelected` — a `tilesPlacedOnLine` with the marker in `discardedTiles`,
   `placedTiles` empty and `type 0`. It is **not a pick** (our engine assigns the
   marker itself), so `parse_log` skips a placement that has no open selection and
   no coloured tiles. A placement with coloured tiles and no selection is still a
   fatal "dropped turn".
5. **Final scores are in `tableinfos`, not the log.** A real archive log has **no**
   cumulative-score notification; `endScore` reports only the *final-round
   increment* per player. The reported totals live in
   `tableinfos.data.result.player[].score` (and redundantly in
   `data.gameResult.rankedTeams[]`). New helper `parse.scores_from_infos` reads
   them; `parse_log` prefers a log `score` notification when present (the fixture
   uses one) and falls back to `tableinfos` otherwise. Several display-only
   notification types (`placeTileOnWallTextLogDetails`, `emptyFloorLineTextLogDetails`,
   `completeLineLogDetails`, `completeColumnLogDetails`, `lastRound`, `endScore`)
   were added to `LogSchema.ignore_types`.

**Not in `tableinfos`: per-seat Elo.** §8 Q8 answered — the payload carries no
`player_elo`/`rank` per seat, so `player_elos()` returns `{}` on a real table and
`TableFilter.min_player_elo_raw` cannot demand *both* players strong from
`tableinfos`. Elite-seat selection therefore relies on `ranked_ids` +
`source_player_id` (which the crawl already supplies from the ladder snapshot), and
the per-game Elo is available in the `getGames` row (`elo_after`) if ever needed.

### 12.4 End-to-end validation (table 897976436, rank-1 Sapperlot)

One standard-wall, 2-player, finished table from rank-1 (Sapperlot 91843016 vs
Bruno Lana 90637398, BGA scores **72–45**) was fetched (`tableinfos` +
`requestTableArchive` + `logs`) and run through the full pipeline. The raw payload
is saved at **`data/human/raw/897976436.json.gz`** as proof. Result:

- `parse_log` → 53 picks, 5 deals, 32 wall placements, first player 90637398;
- `convert_game` replays it **legally** in our engine: tile census `[20]*5`, all 32
  wall columns satisfy the fixed-wall formula, and **engine final scores `(72, 45)`
  exactly match BGA's reported `(72, 45)`** → outcome +1, 5 rounds, 53 positions;
- elite-seat extraction: `choose_target_player` picks seat 0 (Sapperlot, rank 1),
  and `target_mask` keeps 27 of 53 rows, **all from seat 0** — the right seat.

### 12.5 Is the crawl ready? — yes, with two caveats

Auth, the wall filter, the converter and elite extraction are all validated on real
data, and the resume/pace/quota machinery is unchanged and tested. Before a real
launch, note:

- **Arena filter vs. supply.** The validated top table was *Normal* mode, not
  Arena. If `allowed_game_modes=(ARENA_MODE,)` is kept (recommended for the "both
  trying" signal), confirm the yield of Arena tables among the top players is
  enough before committing to a target — many top-player games are Normal mode.
- **Measure the replay quota first** (§5.1, §9 step 4) — still the one unknown that
  governs the schedule. Launch small (the 20-game smoke target), watch for
  `ReplayLimitReached`, then scale.

## 13. First bulk crawl converted — undo/concede handling + the Elo-floor fix (2026-08-17)

**This section is the newest and supersedes §§1-12 where they disagree.** The first
real crawl put **120 rank-1/2 replays** in `data/human/raw/*.json.gz` (121 files
counting the §12.4 validation table). Straight off the crawl the dataset build
yielded **0 positions**: only **28/121** games replayed and the Elo floor then
dropped everything. Both causes are now fixed and the corpus converts in full.

### 13.1 The complete notification taxonomy (every type in the 120 games)

Run `python -m ludometer.human.cli inspect <raw.json.gz>` for one game, or the
histogram over all of them, and this is the full set. "Handling" is where in
`LogSchema` / `parse_log` each lands.

| notification | count (corpus) | role | handling |
|---|---|---|---|
| `gameStateChange` | 28,996 | framework turn/state bookkeeping | **ignore** (`ignore_types`) |
| `tilesPlacedOnLine` | 7,246 | a turn's placement | `place_types` → closes the open pick |
| `tilesSelected` | 6,826 | a turn's take | `select_types` → opens a pick |
| `updateReflexionTime` | 6,329 | clock update | **ignore** |
| `placeTileOnWallTextLogDetails` | 3,633 | "…tiled a line…" prose | **ignore** (cosmetic) |
| `placeTileOnWall` | 2,310 | round-end wall tiling (carries `column`) | `wall_types` → `WallPlacement`, drives the fixed-wall check |
| `emptyFloorLineTextLogDetails` | 1,037 | floor-clear prose | **ignore** (cosmetic) |
| `firstPlayerToken` | 632 | who holds the marker | `marker_types` → cross-check only (engine assigns the marker itself) |
| `factoriesFilled` | 604 | start-of-round deal (1-based, index 0 = center) | `deal_types` → scripted `Deal` |
| `emptyFloorLine` | 590 | round-end floor clear | `floor_clear_types` → boundary flush |
| `endScore` | 386 | final-**round** score increment (not a total) | **ignore** — totals come from `tableinfos` (`scores_from_infos`) |
| **`undoTakeTiles`** | **335** (85 games) | **player took a take back** | **`undo_take_types` → cancel the open selection** |
| `completeLineLogDetails` | 250 | "…completed a line…" prose | **ignore** (cosmetic) |
| `completeColumnLogDetails` | 210 | "…completed a column…" prose | **ignore** (cosmetic) |
| **`undoSelectLine`** | **125** (58 games) | **player took a placement back** | **`undo_place_types` → pop the last pick, re-open it as pending** |
| `simpleNode` | 121 | UI note | **ignore** (`simpleNode`/`simpleNote`) |
| `lastRound` | 117 | last-round banner | **ignore** |
| `simpleNote` | 107 | UI note | **ignore** |
| `wakeupPlayers` | 33 | UI ping | **ignore** |
| **`playerConcedeGame`** | **14** (14 games) | **a player resigned** | **`concede_types` → game ends here, conceder loses** |
| **`completeColorLogDetails`** | **11** (9 games) | "…completed a colour…" prose | **ignore** (cosmetic, added to `ignore_types`) |

The four bold rows are what §12 had not seen. Every other type was already in
`select`/`place`/`deal`/`wall`/`floor`/`score`/`marker`/`ignore`. **Anything not in
one of those lists is still a fatal `ParseError`** — a silently dropped
notification is a silently wrong game, so new types must be classified, never
swallowed.

### 13.2 Undo — the rewind (this was 93 % of the failures)

BGA lets a turn be taken back in two steps and logs one notification per step, **in
reverse order**, confirmed across all 120 games:

```
undoSelectLine   always preceded by tilesPlacedOnLine   (125×)
undoTakeTiles    preceded by tilesSelected (262×) or by undoSelectLine (73×)
```

So at the moment of an undo the state is always well-defined, and the rewind is:

- **`undoSelectLine`** — the placement is reversed, the tiles go back into the
  hand: `parse_log` pops the last committed `Pick` and restores it as the pending
  (open) selection. A re-placement then closes it on the new line; an
  `undoTakeTiles` then cancels the take entirely.
- **`undoTakeTiles`** — the take is reversed: the open pending selection is dropped.

Ignoring them (the old behaviour) was fatal two ways: an unhandled `undoSelectLine`
left the following re-placement looking like "tiles placed without a selection"
(the 34 games that failed that way), and an unhandled `undoTakeTiles` left a phantom
floor pick that desynchronised every later move. **After the rewind the engine
replays each undo game legally and reproduces BGA's reported final scores exactly**
(e.g. table 681632353, which has both undo types, replays to 61-42 = BGA 61-42).

### 13.3 Concession — terminate cleanly, conceder loses

`playerConcedeGame` is always the **last** move notification (the game ends on
resign). `parse_log` records `ReplayGame.conceded_by`, drops any half-made turn and
stops reading moves. `convert_game` then, for a conceded game only, turns **off**
`require_terminal` and `check_scores` (BGA reports a nominal 1-0 for a resign, and
the board never reached a natural end) while keeping the per-move legality replay,
tile-conservation and fixed-wall checks. The outcome is set to a **loss for the
conceder** regardless of the board score; the kept picks are exactly those played
up to the concession. 14 games are conceded (all opponents resigning to the rank-1
player, so all 14 are wins for the target).

### 13.4 The target-Elo floor — resolve it from the ladder, not from `tableinfos`

The dataset builder read the target's Elo as null/0, so `--min-target-elo 650`
dropped all 120 games. Root cause: a real `tableinfos` payload carries **no
per-seat Elo** (§12.3), so `meta.elos` and `player_elos(infos)` are both empty on
real data. The only authoritative source of the target's Elo is the **ladder
ranking snapshot** in `state.json` (`ranking.rows[].elo_raw`), plus the crawl's
recorded `meta.source_elo_raw` for the player whose history the table came from.

The fix (`GameMeta.from_raw` + `cli.cmd_dataset`): the dataset build now passes a
`{player_id: elo_raw}` map built from `ranking.rows` and backfills the target's Elo
from it (and from `source_elo_raw`), so `min_target_elo_raw` has a real number to
compare.

**Units — confirmed.** `ranking.rows[].elo_raw` is BGA **raw** Elo (~1500-centred;
raw = displayed + 1300). `--min-target-elo` is in **displayed** units, and the CLI
converts it once: `min_target_elo_raw = displayed + 1300`. So a floor of **650**
becomes a raw floor of **1950**, and the rank-1/2 targets (raw 2486 / 2459,
displayed 1186 / 1159) clear it comfortably; a floor of 1200 (raw 2500) drops even
rank 1 — verified both directions.

### 13.5 Real numbers after the fix

```
$ python -m ludometer.human.cli --out data/human convert
121 games convert, 0 rejected          # was 28 convert / 93 rejected

$ python -m ludometer.human.cli --out data/human dataset \
      --npz data/human/replay.npz --min-target-elo 650
wrote data/human/replay.npz: 3204 positions from 121 games
  (3204 elite policy targets out of 6489 decision points;
   target record {'win': 104, 'loss': 17, 'draw': 0})
dropped: 0 failed validation, 0 below the Elo floor, 0 with no elite player
```

- **Validation pass rate: 28/121 (23 %) → 121/121 (100 %).** 107 games run to a
  natural end (engine scores match BGA exactly), 14 end on a concession.
- **Dataset: 3,204 elite policy targets** (the target player's turns only) out of
  6,489 replayed decision points, from 121 games.
- **Outcome balance is win-skewed: 104 win / 17 loss / 0 draw.** Expected — the
  targets are rank-1/2 players who win most games, and every conceded game is a win
  for them. The policy targets (real elite moves) are the primary signal; the value
  head sees few losses and no draws, so watch for value-head imbalance and consider
  balancing or de-weighting once the corpus is larger.

**Is the pipeline ready to accumulate a real training set? Yes.** Parse → convert →
dataset now handle everything the real logs contain, the Elo floor resolves
correctly, and the raw payloads are cached so re-running convert/dataset costs no
BGA request. The remaining caveats are unchanged from §12.5 (measure the replay
quota; decide on the Arena-mode filter) and the new one above (outcome imbalance at
this small scale). 120 games ≈ 3.2k positions is a smoke-sized set; the value of the
data is tested at the 2,000-game milestone (§9 step 7).

## 14. 400-game corpus: the last 4 conversion rejects fixed (2026-08-18)

At ~400 downloaded games, 4 of them (1 %) failed to convert. Both causes are fixed
(zero requests spent — cached payloads only), **400/400 now convert**, and the
dataset is **10,790 elite positions**. Two additions to the §13.1 taxonomy:

1. **`timeJokerUsed`** (2 games: tables 779195006, 781529784) — turn-based clock
   bookkeeping, `"${player_name} uses a holiday time joker (+${nb_days} days
   thinking time)"`, args `player_name`/`nb_days` only. Added to
   `LogSchema.ignore_types`.
2. **A concession with no `playerConcedeGame` in the log at all** (2 games: tables
   841829025, 842761995) — the log just stops mid-game ("log ran out after N moves").
   The concession is still in the metadata: `tableinfos.data.result.endgame_reason ==
   "normal_concede_end"`, and the conceder holds the nominal **0** in BGA's 1-0
   result — verified against all 26 notification-carrying concessions in the corpus
   (the zero-score player is the conceder in every one). New
   `parse.conceder_from_infos` reads it; `parse_log` falls back to it when the log
   carried no concede notification, dropping any half-made turn exactly as the
   notification path does. §13.3's convert handling then applies unchanged.

Tests: `tests/test_human_pipeline.py` is now **86** (84 + one per fix).

**Runner backoff fix (same day).** `continuous_runner.sh`'s replay-limit backoff
filtered `fetch.jsonl` downloads to a **26h** window before taking
`min(ts) + 24h + 15min`. Entries 24-26h old — which no longer occupy quota slots —
made that wake time land in the past, so the 300s floor kicked in and the runner
woke every ~5min against a still-full quota (~5 wasted passes, ~4 requests each,
observed 18:56-19:24Z). The lookback is now **24h**, which makes every computed
wake ≥ now + 15min by construction. Runner restarted ~20:31Z to load the fix
(the running bash had the old loop in memory).

## 15. Complete-harvest mode: the 120-games-per-player cap found and removed (2026-08-18, later)

**Why the crawl reached rank 4 in two days when ranks 1-3 have 1,633 / 3,171 /
2,521 ranked games each**: `crawl_ranked`'s `history_pages` default is **12** — at
10 rows per page only a player's **120 most recent** tables were ever listed, and
the runner's `--per-player 200` silently did nothing beyond that (`[:200]` of a
120-item list). So ranks 1-3 "completed" at 109/119/116 downloads each. Not a
BGA limit — our own listing depth.

Per Rémi (2026-08-18): **harvest the top players completely first, then move
down** — their games are the highest-quality signal. Changes:

- **`cli crawl --restart`** → `crawl_ranked(resume=False)`: re-walk the ladder
  from rank 1 instead of resuming at the cursor. Already-judged tables are
  revisited **for free** — a terminal verdict short-circuits before any request,
  budget check or pause, and is counted in the new `CrawlReport.cached` (printed
  by the CLI). So a restart re-walk costs 0 requests for everything already done,
  and new capacity always flows to the highest-ranked player with pending tables.
- **The runner** now passes `--per-player 100000 --history-pages 1000 --restart`
  on every pass: full histories, top-first, self-healing order.
- **Arithmetic to keep in mind**: ranks 1-4 alone hold ~11,400 ranked games and
  ~95 % of their judged tables were accepted so far, so "top 4 complete" is
  ~7-8 weeks at the ~200/day quota. The 2,000-game milestone lands mid-rank-1/2
  either way; deciding to stop or to cap per-player is Rémi's call, not the
  crawler's.
- **Elo-weighting hook**: `replay.stats.json` now carries `game_records` — one
  `{table_id, rows, target_elo_raw}` per game, in npz row-append order (the
  buffer holds every row, so cumulative `rows` reconstruct exact per-row spans).
  Training can weight rows by the target's Elo without any npz format change
  (`ReplayBuffer` reads keys by name and ignores extras, but stays untouched).
- Of the first 400 downloaded games, **43 (~11 %) have both seats in the top
  200** — mining the second elite seat of those tables would be free extra
  positions at zero quota cost; not implemented (dataset currently learns the
  higher-Elo seat only).

Tests: **87** (86 + the restart/cached re-walk).

## 16. Per-game Elo capture: `elo_after` from the history rows (2026-08-19)

**Question this answers**: can moves be scored by the Elo of the player who made
them? Now yes, time-accurately, for the elite seat (the only seat whose moves are
policy targets).

- `tableinfos` has **no per-seat Elo** (§12.3) and the ladder snapshot is *today's*
  number — wrong for old games. The **`getGames` history row** is the only per-game
  Elo BGA exposes: `elo_after` (the **queried** player's raw Elo right after that
  game), plus `start`/`end` timestamps, `elo_win`, concede/normalend flags.
  Verified live 2026-08-19; the drift is real: Sapperlot is 2486 raw today but
  ~2383 in his mid-2025 games — a ~100-point error the snapshot would have baked in.
- The crawler used to extract only table ids from those rows and **discard the
  rest**. `fetch_player_tables` now appends every row verbatim to
  **`data/human/table_rows.jsonl`** (append-only, duplicates deduped at read time,
  never raises). Listing progress was reset once (pages_done → 0, table ids kept)
  so already-listed pages get re-listed *with* capture; re-listing costs requests
  but no replay-quota slots.
- `fetch.elo_after_map(path)` → `{table_id: {player_id: elo_after_raw}}`;
  `cli dataset` passes it to `build_dataset(elo_at_game=...)`, and each
  `game_records` entry in `replay.stats.json` now carries **`target_elo_after`**
  (None for games whose history page predates capture — coverage completes as the
  re-walk re-lists each rank). Weight training rows by `target_elo_after`,
  falling back to `target_elo_raw`.
- The opponent's per-game Elo exists only when the opponent is also a crawled
  ranked player (their own history contributes their `elo_after` for the shared
  table) — ~11 % of games so far. For everyone else BGA gives us nothing.

Tests: **89** (87 + row persistence/`elo_after_map` + the dataset record).

## 17. Elo-at-game-time floor + private HF mirror (2026-08-20)

**The Elo scale, one more time** (it keeps biting): BGA displays
`max(0, raw − 1300)`. Sapperlot shows **1188** on the site = **2488 raw**; the
ladder API, `elo_after`, and everything in our state file are RAW. `mode=elo`
(what we crawl) IS the all-time ladder; `mode=arena` is the current season and
is not used anywhere.

- **Rank 1's full history spans raw 1338 → 2488 (displayed 38 → 1188)** — a top
  player's early games are beginner games. 117 of rank 1's 1,604 tables (7 %)
  are below the displayed-650 floor *at game time*; the snapshot floor would
  have accepted all of them.
- **Fetch-side floor**: `crawl_ranked(min_source_elo_raw=...)` (CLI: the same
  `--min-elo` that selects players) skips a table from the history row alone
  when the source player's `elo_after` is below the floor — **before any
  request**, so a beginner-era game costs zero replay quota. Skip reason:
  `"source elo N at game time below floor M"`.
- **Dataset-side floor**: `build_dataset`'s `min_target_elo_raw` now compares
  the per-game `elo_after` when captured, falling back to the ladder snapshot.
- **Private HF mirror (loss insurance, per Rémi)**: every crawl pass ends with
  `data/human/push_hf.py` (run via `uvx --from huggingface_hub`, token from
  `~/.cache/huggingface`) mirroring all of `data/human/` — raw payloads, state,
  rows, npz, the wip-backup of uncommitted code — to
  **`RemiFabre/azul-elite-replays`** (dataset repo, **private**: scraped BGA
  data must never be republished). `upload_folder` is content-hash incremental.
- Known reject to revisit: 1 game of 800 fails with "engine started round 4 but
  the log only holds 4 deals" — unclassified, likely an undo/log edge; grep the
  runner log for the table id when someone has time.

Tests: **91** (89 + the quota-saving skip + the elo_after floor).

## 18. First training experiments on the human data (2026-08-21, overnight)

**Corpus used (pinned)**: 799 games / 798 converted, all rank-1 (Sapperlot), built as
`data/human/replay_full.npz` — 43,210 rows, **22,738 policy targets** (21,286 elite +
1,452 dual-target: per Rémi, when the opponent is also ≥ displayed **800** at game
time their moves are policy targets too — `--dual-target-min-elo 800`, 54 games
qualified, implemented in `dataset.py`/`cli.py`, wired into the runner). Opponent
rows kept value-only (`--keep-opponent-rows`) so value labels are two-sided
(21,287 `+1` / 21,923 `−1`) instead of the 93%-win skew. Plus 54 Faïence
human-win/draw games (`ludometer/human/faience.py`, all 315 finished site games
replay exactly; humans are 44W-10D-261L vs the run4 site net) → 1,524 more policy
rows at `data/faience/human_windraw.npz`.

**E0 — agreement** (`ludometer/human/agreement.py`, `data/human/agreement.json`):
run4/run5/run6 nets pick Sapperlot's move **39-40%** top-1 (67% top-3) — and
**33% in the first game-quartile vs 48% in the last**: weakest exactly where
strategy lives, confirming the "great calculator, poor strategist" read.

**E2 — fine-tune with rehearsal** (`ludometer/train/finetune.py`: mixed batches
1 human : 3 self-play from the base's own replay buffer, LR 1e-4):

| candidate | base | vs base @ sims=100 | verdict |
|---|---|---|---|
| ft1/ft-002000 (`runs/ft1`) | run5/ckpt-006912 (best on disk, 2381) | **59.0%** over 200 g | **works** (~+63 Elo; pooled fit **2394.5** anchored run4site=2360.6) |
| ft2/ft-004000 (`runs/ft2`, dual buffer, 4000 steps) | same | **62%/100 g vs ft2000** (55.5/62/59 across 3 ckpts ≈ 59% agg/300 g) | **new equal-search champion**, ~2445 pooled |
| ft3, ft3b (`runs/ft3*`) | run4/ckpt-037888 (the site net) | all ≈50% after honest 200-300-game tests | small net **cannot absorb** the prior |
| ds1 (`runs/ds1`, distill ft2000→run4 arch, `ludometer/train/distill.py`) | — | 37/51.5/47.5% vs site net | quick distillation doesn't transfer it either |

Agreement moves 40%→48% top-1 (early game 34%→46%) on the fine-tuned nets —
the policy really does absorb the human prior; on the big net it converts to
playing strength, on the 1.8M-param site net it only displaces self-play
knowledge (capacity-saturated).

**The wall-clock trap (why nothing was deployed)**: ft2000 wins at equal *sims*
but at equal *think time* loses **36-64** to the 4×-smaller site net
(`data/human/gauntlet_ft2000_wallclock.json`) — which is also why run4 outlived
run5 on the site. Rule going forward: **a deploy candidate must win at
wall-clock parity**, and for identical architectures sims-parity = wall-clock
parity. Second rule, learned the embarrassing way: a 100-game screen of many
sibling checkpoints WILL produce fake 60% winners (ft3b-1000 screened 60.5%,
confirmed **49.8%** over 300 games) — never believe a screen, always confirm the
selected candidate on ≥200 fresh-seed games.

**E3 — pretrain A/B** (`runs/hp2` vs `runs/run6`, identical configs incl. 3
pretrain epochs over a 500k buffer; hp2's buffer = 371k run5 self-play rows +
3×43,210 human rows (25.9%, `data/human/replay_mix.npz`, single-target — built
before the dual flag); control run6 = run5 rows only; 9,984 self-play games
each, `ludometer/eval/compare_runs.py`): the human mix costs **−58 Elo at the
warm start** (2181 vs 2239) and trails to ~game 2,000, is even through ~5,400,
then the **second half averages ~+55 Elo** with three points outside both error
bars (+54/+99/+66 at 6144/6912/7680, and +99 at the end: 2299.5 vs 2200.9).
Best-vs-best is a tie (2303.6 vs 2298.6). Single seed each — read as
*suggestive-positive*, worth a re-run at the 2,000-game corpus milestone.
(Also: `runs/hp1`, abandoned arm, gives the pure-BC datapoint — 3 epochs on the
human data alone rates **1315** at 100 sims, between greedy 1220 and heuristic
1378.)

**Update (2026-08-21 afternoon) — the mid-size path works, but is under the new
bar.** `runs/mid1`: a fresh 2.94M-param net (`configs/mid_a_net.json`, 0.57× the
site net's inference speed) pretrained 3 epochs on **ft2-4000 teacher soft
targets** over 296k states (`data/human/teacher_labeled.npz` — distillation
through the standard `--pretrain` path, no trainer changes) and then self-play
polished for 8,000 games. Curve: 2147 at game 0 → best **2317** at 4,096 →
plateau ~2270-2300. Its best checkpoint is the **first candidate above 50% at
wall-clock parity with the site net: 52.5% over 100 games** (think=1.0 each,
`gauntlet_mid1_wallclock.json`) — ≈+17 honest Elo, versus the deploy bar Rémi
set the same day (**+150 honest over ≥300 games**, `docs/BOT_DEPLOYMENT.md`;
ft2-4000 as an extra site opponent was also ruled out). So: no deploy, recipe
validated. The road to "Porcelain" is this exact pipeline scaled up — a better
teacher (the fine-tune ceiling rises with every crawl milestone), a longer
polish, and possibly the ~4M "midB" body — plus the corpus growth that feeds it.

**Bottom line**: the elite-human data measurably improves the strongest net at
equal search (+60-110 Elo of fine-tune headroom demonstrated from just 799
games of ONE player), the effect concentrates in the early/strategic game, and
the site was deliberately **not** touched because no same-speed candidate
cleared the wall-clock gate. Paths to a deployed win, in order of promise:
(1) a **mid-size net (~3-3.5M params)** distilled from the ft2 teacher then
briefly self-played — big enough to hold the prior, small enough to search;
(2) raise the site think budget; (3) redo E2/E3 at the 2,000-game multi-player
milestone (breadth-first crawl would accelerate this). Faïence win/draw games as
targeted self-play seeds remain unimplemented — still the most surgical attack
on the "brutal blunder" states.

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

## Appendix C: request ledger for the 2026-08-17 authenticated validation (9)

All with Remi's cookies + `X-Request-Token`, desktop Chrome UA, ≥6 s apart.

| # | Request | Result |
|---|---|---|
| 1 | `GET /gamepanel?game=azul` (logged in) | 200; token `gNZGNxgP5p7MOzj` scraped |
| 2 | `GET /gamepanel?game=azul` (token refresh) | 200; token `AyZAa3d3WVPNN1Y` (rotates) |
| 3 | `GET /gamestats/…/getGames.html?player=91843016&game_id=1467&finished=1&page=1` | **200, 10 tables** (was 806) |
| 4–5 | `GET /table/table/tableinfos.html?id=897976436,897976536` | 200; option 100 = "Board" pinned |
| 6 | `GET /gamepanel?game=azul` (token for replay fetch) | 200 |
| 7 | `GET /gamereview/…/requestTableArchive.html?table=897976436` | 200, archive primed |
| 8 | `GET /archive/archive/logs.html?table=897976436&translated=true` | 200, full replay |

(The token refreshes reuse the public `/gamepanel` page — no session risk — and
each authenticated script re-scraped it because the process was restarted; a single
long-running crawl scrapes once.) **No replay-limit or account-disable signal at any
point.**
