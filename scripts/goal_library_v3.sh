#!/bin/bash
# Re-convert the dataset sources into data/goal_library_v3 with kept size targets spanning 8-256 cells
# (one variant per target where an accepted scale lies within a factor 2 of it). CPU only, 3 processes.
# text2shape (raw zip gone) and objaverse_cap3d (about 1.9 GB of meshes to re-stream) are skipped.
cd "$(dirname "$0")/.." || exit 1
OUT=data/goal_library_v3
KEEP="8 16 32 64 128 256"
for source in stabletext2brick craftassist_parts craftassist_houses iglu_single iglu_multiturn hack337; do
  nice -n 10 .venv/bin/python -I scripts/goals_convert.py "$source" --procs 3 --keep-targets $KEEP --tolerance 2 --out "$OUT" > "logs/goal-library-v3-$source.out" 2>&1
done
# oga_vox and itch_vox are dropped by the 2026-11-13 label audit (G.DROPPED_SOURCES, runs/label_audit/report.md), so they are
# no longer converted by default. Set CONVERT_DROPPED=1 to convert them anyway.
if [ "${CONVERT_DROPPED:-0}" = 1 ]; then
  for source in oga_vox itch_vox; do
    nice -n 10 .venv/bin/python -I scripts/goals_sources_more.py "$source" --procs 3 --keep-targets $KEEP --tolerance 2 --out "$OUT" > "logs/goal-library-v3-$source.out" 2>&1
  done
fi
