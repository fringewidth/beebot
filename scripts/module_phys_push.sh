#!/bin/zsh
# Pushes the parts-catalogue count to ops every 5 minutes until runs/module_phys/.stop exists (max 6 h).
cd "$(dirname "$0")/.."
for i in $(seq 1 72); do
  [ -f runs/module_phys/.stop ] && break
  n=$(/opt/homebrew/bin/python3 -c "import json;print(len(json.load(open('runs/module_phys/parts.json'))))" 2>/dev/null || echo 0)
  /opt/homebrew/bin/python3 ~/ops/ops.py push "module_parts_catalogued $n" >/dev/null 2>&1
  sleep 300
done
