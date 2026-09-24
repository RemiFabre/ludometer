#!/usr/bin/env bash
cd /Users/remi/ludometer
P=runs/porc_w-p0905-2038/checkpoints/ckpt-000000.pt
PORC="porcelain=mcts:$P?think=1.0&engine=rust"
while kill -0 85424 2>/dev/null; do sleep 30; done
name=ft_open1d-004000_vs_porcelain_300
[ -f "runs/gates/$name.json" ] || { echo "== $name $(date +%H:%M)"; .venv/bin/python -m ludometer.eval.gauntlet --games 300 --workers 8 --seed 20260907 --nice 10 --json "runs/gates/$name.json" "ft1d_4000=mcts:runs/ft_open1d/checkpoints/ft-004000.pt?think=1.0&engine=rust" "$PORC" 2>&1 | grep -E " vs " | head -2; }
echo "== queue4 done $(date +%H:%M)"
