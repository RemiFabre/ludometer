#!/usr/bin/env bash
cd /Users/remi/ludometer
P=runs/porc_w-p0905-2038/checkpoints/ckpt-000000.pt
PORC="porcelain=mcts:$P?think=1.0&engine=rust"
while kill -0 58907 2>/dev/null; do sleep 30; done
while [ ! -f runs/ft_open1d/checkpoints/ft-004000.pt ]; do sleep 30; done
run() { local name=$1 games=$2 seed=$3 spec=$4
  [ -f "runs/gates/$name.json" ] && { echo "skip $name"; return; }
  echo "== $name $(date +%H:%M)"
  .venv/bin/python -m ludometer.eval.gauntlet --games "$games" --workers 8 --seed "$seed" --nice 10 \
    --json "runs/gates/$name.json" "$spec" "$PORC" 2>&1 | grep -E " vs " | head -2
}
run ft_open1d-004000_vs_porcelain 100 20260906 "ft1d_4000=mcts:runs/ft_open1d/checkpoints/ft-004000.pt?think=1.0&engine=rust"
run ft_open1d-002000_vs_porcelain 100 20260906 "ft1d_2000=mcts:runs/ft_open1d/checkpoints/ft-002000.pt?think=1.0&engine=rust"
run ft_open12-004000_vs_porcelain_300 300 20260907 "ft12_4000=mcts:runs/ft_open12/checkpoints/ft-004000.pt?think=1.0&engine=rust"
echo "== queue3 done $(date +%H:%M)"
