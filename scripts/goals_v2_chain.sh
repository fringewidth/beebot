#!/bin/bash
# Convert every goal source into data/goal_library_v2 (voxeliser search, cap 800), one after another, 4 processes.
cd "$(dirname "$0")/.."
for s in stabletext2brick iglu_single iglu_multiturn craftassist_houses hack337 craftassist_parts; do
  nice -n 10 .venv/bin/python scripts/goals_convert.py $s --procs 4 --out data/goal_library_v2 > logs/goals-v2-$s.out 2>&1
done
echo finished > logs/goals-v2-done.out
