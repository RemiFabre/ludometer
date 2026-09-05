#!/usr/bin/env bash
# One Porcelain pretraining cycle: pull the fleet's shards, fold them into the
# corpus, pretrain a student config on it (no self-play yet), rate it on the
# fixed-sims ladder, then screen it at matched think time against Cobalt.
#
#   scripts/porcelain_pretrain.sh porc_a [gate_games]
#
# Runs are written to runs/<config>-p<N> (N = corpus build number) so every
# corpus size gets its own curve. HF_TOKEN must be set.
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=${1:?config name, e.g. porc_a}
GAMES=${2:-100}
STAMP=$(date +%m%d-%H%M)
RUN="${CFG}-p${STAMP}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "== pull"
uv run python -m ludometer.cloud.corpus pull --run rlx_teacher rlx_bga 2>&1 | grep corpus || true
echo "== build"
uv run python -m ludometer.cloud.corpus build --run rlx_teacher rlx_bga --out data/cloud/porcelain_corpus.npz 2>&1 | grep corpus
uv run python -m ludometer.cloud.corpus stats --run rlx_teacher rlx_bga 2>&1 | grep -E '"games"|"positions"|"jobs"' | tr -d ' ,' | paste -sd' ' -

echo "== pretrain + rate: runs/$RUN"
nice -n 5 uv run python -m ludometer.train.run --config "configs/$CFG.json" --run "$RUN" --max-games 0 \
  > "runs/$RUN.log" 2>&1
grep -E "pretrain epoch|eval ckpt" "runs/$RUN.log" | tail -4

echo "== wall-clock screen vs Cobalt ($GAMES games)"
mkdir -p runs/gates
nice -n 10 uv run python -m ludometer.eval.gauntlet --games "$GAMES" --workers 8 --seed 20260906 \
  --json "runs/gates/${RUN}_wallclock.json" \
  "cand=mcts:runs/$RUN/checkpoints/ckpt-000000.pt?think=1.0" \
  "cobalt=mcts:runs/run4/checkpoints/ckpt-037888.pt?think=1.0" 2>&1 | grep -E "cand vs|games$" | head -3
