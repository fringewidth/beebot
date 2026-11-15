#!/bin/zsh
# Pushes module-physical progress to ops every 5 minutes until runs/module_phys/.stop exists (max 8 h):
# the worst move success of the reaction-wheel cube and the smallest edge length that passes the machines.
cd "$(dirname "$0")/.."
P=/opt/homebrew/bin/python3
for i in $(seq 1 96); do
  [ -f runs/module_phys/.stop ] && break
  $P - <<'PY' | $P ~/ops/ops.py push >/dev/null 2>&1
import json, os
def rows(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out
w = [r["worst_move_success"] for r in rows("runs/module_phys/rw_sim/results.jsonl") if isinstance(r.get("worst_move_success"), (int, float))]
if w:
    print(f"rw_worst_move_success {max(w)}")
m = [r["min_edge_mm"] for r in rows("runs/module_phys/joint/results.jsonl") if isinstance(r.get("min_edge_mm"), (int, float))]
if m:
    print(f'module_min_edge_mm{{scope="all"}} {min(m)}')
PY
  sleep 300
done
