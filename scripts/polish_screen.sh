#!/usr/bin/env bash
# Overnight companion to a Phase B run: every INTERVAL seconds, take the
# best-rated checkpoint (fixed-sims ladder, elo.jsonl) that has not been
# screened yet and play it 100 games against the shipped Porcelain at matched
# think time. Results: runs/gates/<run>-<ckpt>_vs_porcelain.json and one line
# per screen in runs/gates/polish_screens.log. Niced, so the learner and the
# fleet come first.
#
#   scripts/polish_screen.sh runs/porc_w-p0905-2038 [interval_s] [games]
set -u
cd "$(dirname "$0")/.."
RUN=${1:?run dir}; INTERVAL=${2:-2400}; GAMES=${3:-100}
NAME=$(basename "$RUN")
PORC="mcts:runs/porc_w-p0905-2038/checkpoints/ckpt-000000.pt?think=1.0"
mkdir -p runs/gates
while true; do
  BEST=$(python3 - "$RUN" <<'PY'
import json, sys, os
run = sys.argv[1]
rows = [json.loads(l) for l in open(f"{run}/elo.jsonl")]
done = set()
for f in os.listdir("runs/gates"):
    if f.startswith(os.path.basename(run)) and f.endswith("_vs_porcelain.json"):
        done.add(f[len(os.path.basename(run)) + 1 : -len("_vs_porcelain.json")])
cands = [r for r in rows if r["ckpt"] != "ckpt-000000" and r["ckpt"] not in done
         and os.path.exists(f"{run}/checkpoints/{r['ckpt']}.pt")]
if cands:
    best = max(cands, key=lambda r: r["elo"]); print(best["ckpt"], best["elo"])
PY
)
  if [ -n "$BEST" ]; then
    CKPT=${BEST%% *}; ELO=${BEST##* }
    OUT="runs/gates/${NAME}-${CKPT}_vs_porcelain.json"
    LINE=$(nice -n 12 uv run python -m ludometer.eval.gauntlet --games "$GAMES" --workers 6 --seed 20260907 \
      --json "$OUT" "cand=mcts:$RUN/checkpoints/$CKPT.pt?think=1.0" "porcelain=$PORC" 2>&1 | grep "cand vs porcelain" | head -1)
    echo "$(date +%m-%d\ %H:%M) $CKPT fixed-sims $ELO | $LINE" >> runs/gates/polish_screens.log
  fi
  sleep "$INTERVAL"
done
