#!/bin/zsh
# Redraws a voxel store as v2 (hinge cube outcomes, intended actions; old tower rule so the Haiku labels still join).
# usage: OPS_NODE=<id> scripts/draw_demos_v2.sh full 0   |   ... build 1
cd "$(dirname "$0")/.." || exit 1
kind=$1; seed=$2
if [ "$kind" = full ]; then
  extra=(--scenes 1200 --coverage 0.3 --objects 1.0)
else
  extra=(--scenes 1600 --steps 144 --build 1.0 --only build)
fi
OMP_NUM_THREADS=2 nice -n 5 .venv/bin/python -m beebot.vla.voxel.demos --out data/vla/voxel_${kind}_v2/s$seed --seed $seed --cube --rule switch_cube --outcomes mujoco_hinge_v1 --robots 6 115 --counts log --fit --compact "${extra[@]}" > logs/demos-v2-$kind-s$seed.out 2>&1
