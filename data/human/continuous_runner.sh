#!/usr/bin/env bash
# Continuous BGA harvest loop. Detached (nohup) so it survives session end.
# No artificial daily cap: it downloads continuously at a polite per-request
# rate (one replay every ~6-10s) until the target game count is reached OR BGA
# actually pushes back (replay-limit / disabled), in which case it backs off
# 90 min and retries — this discovers BGA's real limit empirically instead of
# guessing. Rebuilds the dataset + harvest page after every crawl pass.
# Stop with:  pkill -f continuous_runner.sh
set -u
cd /Users/remi/ludometer || exit 1
LOG=data/human/continuous_runner.log
COOKIES=/Users/remi/ludometer/.bga_cookies.txt
TARGET_GAMES=100000           # per Rémi 2026-10-12: keep going until he says stop or 100k games
BIG=1000000                   # effectively-unlimited per-day caps (rate is the throttle)

games_downloaded() {
  python3 - <<'PY'
import json
try:
    d = json.load(open('data/human/state.json'))
    print((d.get('downloads') or {}).get('total') or 0)
except Exception:
    print(0)
PY
}
error_kind() {
  python3 - <<'PY'
import json
try:
    d = json.load(open('data/human/state.json'))
    print((d.get('error') or {}).get('kind') or '')
except Exception:
    print('')
PY
}

while true; do
  have=$(games_downloaded)
  if [ "$have" -ge "$TARGET_GAMES" ]; then
    echo "=== target $TARGET_GAMES reached ($have) $(date -u +%FT%TZ); stopping ===" >> "$LOG"
    break
  fi
  echo "=== crawl pass $(date -u +%FT%TZ) (have $have) ===" >> "$LOG"
  # --restart + huge per-player/history-pages = harvest each rank COMPLETELY
  # before moving down (per Rémi 2026-08-18): every pass re-walks from rank 1,
  # already-judged tables are free (cached, no request), so new capacity always
  # goes to the highest-ranked player with pending tables. The old defaults
  # (history-pages 12) silently capped each player at their 120 most recent
  # games, which is why ranks 1-3 "finished" in a day.
  nice -n 10 uv run python -m ludometer.human.cli \
    --out data/human --cookies "$COOKIES" \
    --min-interval 6 --jitter 4 --max-per-day "$BIG" --max-tables-per-day "$BIG" \
    --top 200 --min-elo 650 --min-games 200 \
    crawl --per-player 100000 --history-pages 1000 --restart >> "$LOG" 2>&1
  # rebuild dataset + page from whatever we have so far
  nice -n 15 uv run python -m ludometer.human.cli --out data/human convert >> "$LOG" 2>&1
  # --dual-target-min-elo 800 (per Rémi 2026-08-20): when BOTH seats are strong
  # (opponent displayed >=800), keep the opponent's moves as policy targets too
  # — an elite-vs-elite game teaches from both players. ~11% of games qualify.
  nice -n 15 uv run python -m ludometer.human.cli --out data/human \
    dataset --npz data/human/replay.npz --min-target-elo 650 \
    --dual-target-min-elo 800 >> "$LOG" 2>&1
  python3 web/make_harvest.py --state data/human >> "$LOG" 2>&1
  # loss insurance: mirror data/human to the PRIVATE HF dataset
  # RemiFabre/azul-elite-replays (incremental; non-fatal if HF is unreachable)
  nice -n 15 uvx --from huggingface_hub python data/human/push_hf.py >> "$LOG" 2>&1
  kind=$(error_kind)
  case "$kind" in
    replay-limit)
      # BGA caps replays at ~200 per account (measured 2026-08-17: stopped at 199
      # despite gentle spacing = a cumulative quota, not a rate limit). Overnight
      # probing proved it is NOT a UTC-midnight calendar reset (still blocked at
      # 08:23 UTC the next day), so it's a ROLLING ~24h window: a slot frees 24h
      # after each download. Correct backoff = sleep until 24h + 15min after the
      # EARLIEST recent download, when the first slots come back. This also
      # confirms the theory: if downloads resume right around then, it's rolling-24h.
      secs=$(python3 -c "
import json, time
ts=[json.loads(l).get('ts') for l in open('data/human/fetch.jsonl') if '\"downloaded\"' in l]
# Only downloads in the trailing 24h occupy quota slots. A wider lookback (the old
# 26h) let 24-26h-old entries produce wake times in the past -> the 300s floor ->
# a retry loop against a still-full quota (~5 wasted passes on 2026-08-18).
ts=[x for x in ts if x and x > time.time()-24*3600]
if ts:
    wake=min(ts)+24*3600+900
    print(max(300, int(wake-time.time())))
else:
    print(3600)")
      echo "=== replay quota hit $(date -u +%FT%TZ); rolling-24h: sleeping ${secs}s until first slots free ===" >> "$LOG"
      sleep "$secs" ;;
    account-disabled|auth)
      echo "=== BGA auth/disable ($kind) $(date -u +%FT%TZ); STOPPING — needs attention ===" >> "$LOG"
      break ;;
    *)
      # normal completion of available tables, or transient — short pause, continue
      sleep 60 ;;
  esac
done
