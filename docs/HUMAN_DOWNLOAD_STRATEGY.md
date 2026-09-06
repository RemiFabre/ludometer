# Handoff: downloading high-Elo human Azul games (BGA)

**Goal.** Build a pretraining set of *elite* human Azul play from Board Game Arena (BGA),
to test whether human-game pretraining breaks the self-play Elo plateau (~+2,380 internal).
We crawl the **all-time** Azul ladder in rank order (rank 1 first) and, for each ranked
player's games, keep **only the elite player's moves** as policy targets (outcome from their
perspective). Full pipeline + format details: **`docs/HUMAN_GAMES.md`** (this is the short version).

## State as of 2026-08-18 (evening)
- **~399 games downloaded → 400 convert → 10,790 elite positions.** Cursor at rank 4.
  (400 > 399: the §12.4 validation table is in `raw/` too. All 4 former conversion
  rejects are fixed — see `docs/HUMAN_GAMES.md` §14: `timeJokerUsed` is now ignored,
  and a concession missing from the log is read from `tableinfos.endgame_reason`.)
- Runner **`data/human/continuous_runner.sh`** is detached and RUNNING (restarted
  2026-08-18 ~20:31Z to pick up a backoff fix: the quota lookback is now 24h, not 26h —
  the wider window caused premature-wake retry loops against a still-full quota).
- **Complete-harvest mode since 2026-08-18 evening** (`docs/HUMAN_GAMES.md` §15): the old
  `history_pages=12` default silently capped every player at their 120 most recent games —
  that is why the cursor reached rank 4 in two days. The runner now passes
  `--per-player 100000 --history-pages 1000 --restart`: full histories, each rank
  harvested **completely** before moving down (per Rémi), cached tables revisited free.
  Ranks 1-4 alone ≈ 11,400 ranked games ≈ 7-8 weeks at ~200/day — revisit the target-size
  decision with that number in mind.
- First complete-harvest pass ran 20:47-21:08Z: rank 1's FULL history is now listed
  (161 pages, complete, **1,604 tables**, ~1,480 of them pending), cursor re-anchored at
  rank 1 offset 21, sleeping until ~13:04Z 2026-08-19. Watch item: 3 old rank-1 tables
  error with BGA's "Cannot find gamenotifs log file (code 100)" — non-terminal, so they
  are retried at the head of every pass; if code 100 proves permanent (likely a lost
  archive), classify it as ReplayUnavailable → terminal skip in `client._classify`.
- Cookies at **`~/ludometer/.bga_cookies.txt`** (Netscape; PHPSESSID + TournoiEnLigne*). The
  client also needs BGA's per-session **X-Request-Token** (scraped from a page each run — already wired).
- **Nothing about the human pipeline is committed** (per Rémi). All under `data/human/` + `ludometer/human/`.

## The download limit (measured, not guessed)
- BGA caps replays at **~200 per account per ROLLING ~24h window** — a cumulative quota, **not**
  a rate limit (we hit it at gentle ~87s spacing), and **not** a UTC-midnight reset (still blocked
  at 08:23 the next morning). A slot frees ~24h after each download.
- The runner handles this: on the "You have reached a limit (replay)" error it sleeps until
  **24h + 15min after the earliest download in the window**, then resumes. So it self-paces at
  ~200 games/day with no babysitting. Do **not** try to beat the quota (multiple accounts = ToS
  trouble; the account is Rémi's).
- **ToS caveat:** automated access is against BGA's terms; we stay under the quota and stop the
  instant BGA signals. Rémi accepted this risk knowingly.

## Operating it
```bash
# progress numbers
python3 - <<'PY'
import json; d=json.load(open('data/human/state.json'))
print('downloads', (d.get('downloads') or {}).get('total'), '| cursor rank', (d.get('cursor') or {}).get('rank'), '| error', (d.get('error') or {}).get('kind'))
PY

# is the runner alive?  restart it if not (detached, survives your session):
pgrep -f continuous_runner.sh || (cd ~/ludometer && nohup data/human/continuous_runner.sh >/dev/null 2>&1 &)

# live progress page
python3 web/make_harvest.py --state data/human && open web/harvest.html

# (re)build the training set from whatever is downloaded so far
uv run python -m ludometer.human.cli --out data/human convert
uv run python -m ludometer.human.cli --out data/human dataset --npz data/human/replay.npz --min-target-elo 650
# -> data/human/replay.npz, loadable by the trainer's --pretrain

# stop the crawl
pkill -f continuous_runner.sh
```
`--min-target-elo` is in DISPLAYED Elo (BGA raw = displayed + 1300); target Elo is resolved from
the ladder in `state.json`, NOT from table metadata (per-seat Elo is absent there).

## Open decisions (ask Rémi)
1. **Target size** — runner aims at 10,000 games (~50 days). Rémi leaned toward capping at
   **2,000 (~10 days)** as enough to test the hypothesis. Set `TARGET_GAMES` in the runner.
2. **Early experiment** — at ~1,000 games, run a pretrain-from-human-games experiment and rate
   it against the current best on the honest Azul ladder before collecting the rest.
3. **Watch the outcome imbalance** — elite players win ~90% of their games, so value targets are
   win-skewed; fine for a policy prior, a caveat for the value head at scale.
