#!/usr/bin/env bash
# Sequential wall-clock gates for the opening task force (one gauntlet at a time).
cd /Users/remi/ludometer
P=runs/porc_w-p0905-2038/checkpoints/ckpt-000000.pt
IM=runs/imit_r1_porc/checkpoints/best.pt
IS=runs/imit_r1/checkpoints/best.pt
PORC="porcelain=mcts:$P?think=1.0&engine=rust"
while kill -0 39986 2>/dev/null; do sleep 20; done
run() { # name spec
  local name=$1 spec=$2
  [ -f "runs/gates/$name.json" ] && { echo "skip $name"; return; }
  echo "== $name $(date +%H:%M)"
  .venv/bin/python -m ludometer.eval.gauntlet --games 100 --workers 8 --seed 20260906 --nice 10 \
    --json "runs/gates/$name.json" "$spec" "$PORC" 2>&1 | grep -E "vs|Elo|games" | head -6
}
run hybrid_imitporc_kall_vs_porcelain "hyb_all=hybrid:$IM|$P?k=all&think=1.0&engine=rust"
run hybrid_imitporc_k1_vs_porcelain "hyb_k1=hybrid:$IM|$P?k=1&think=1.0&engine=rust"
run ft_open1-004000_vs_porcelain "ft4000=mcts:runs/ft_open1/checkpoints/ft-004000.pt?think=1.0&engine=rust"
run hybrid_imitporc_k2_vs_porcelain "hyb_k2=hybrid:$IM|$P?k=2&think=1.0&engine=rust"
run ft_open1-001000_vs_porcelain "ft1000=mcts:runs/ft_open1/checkpoints/ft-001000.pt?think=1.0&engine=rust"
run hybrid_imitscratch_k3_vs_porcelain "hybs_k3=hybrid:$IS|$P?k=3&think=1.0&engine=rust"
run ft_open12-004000_vs_porcelain "ft12_4000=mcts:runs/ft_open12/checkpoints/ft-004000.pt?think=1.0&engine=rust"
run ft_open1-002000_vs_porcelain "ft2000=mcts:runs/ft_open1/checkpoints/ft-002000.pt?think=1.0&engine=rust"
run hybrid_imitporc_k3_temp1_vs_porcelain "hyb_k3t1=hybrid:$IM|$P?k=3&temp=1.0&think=1.0&engine=rust"
run ft_open1-000500_vs_porcelain "ft500=mcts:runs/ft_open1/checkpoints/ft-000500.pt?think=1.0&engine=rust"
echo "== queue done $(date +%H:%M)"
