#!/usr/bin/env bash
cd /Users/remi/ludometer
P=runs/porc_w-p0905-2038/checkpoints/ckpt-000000.pt
IM=runs/imit_r1_porc/checkpoints/best.pt
IS=runs/imit_r1/checkpoints/best.pt
PORC="porcelain=mcts:$P?think=1.0&engine=rust"
while kill -0 57384 2>/dev/null; do sleep 20; done
run() { # name games seed spec
  local name=$1 games=$2 seed=$3 spec=$4
  [ -f "runs/gates/$name.json" ] && { echo "skip $name"; return; }
  echo "== $name $(date +%H:%M)"
  .venv/bin/python -m ludometer.eval.gauntlet --games "$games" --workers 8 --seed "$seed" --nice 10 \
    --json "runs/gates/$name.json" "$spec" "$PORC" 2>&1 | grep -E " vs " | head -2
}
run hybrid_imitporc_k3_vs_porcelain_300 300 20260907 "hyb_k3=hybrid:$IM|$P?k=3&think=1.0&engine=rust"
run hybrid_imitscratch_k3_vs_porcelain 100 20260906 "hybs_k3=hybrid:$IS|$P?k=3&think=1.0&engine=rust"
run ft_open12-004000_vs_porcelain 100 20260906 "ft12_4000=mcts:runs/ft_open12/checkpoints/ft-004000.pt?think=1.0&engine=rust"
run ft_open1-004000_vs_porcelain_300 300 20260907 "ft4000=mcts:runs/ft_open1/checkpoints/ft-004000.pt?think=1.0&engine=rust"
run hybrid_imitporc_k3_temp1_vs_porcelain 100 20260906 "hyb_k3t1=hybrid:$IM|$P?k=3&temp=1.0&think=1.0&engine=rust"
echo "== queue2 done $(date +%H:%M)"
