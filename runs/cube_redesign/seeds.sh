set -e
cd /Users/hrishik/beebot/.claude/worktrees/cube-redesign
for s in 1000 2000; do
  PYTHONPATH=src /Users/hrishik/beebot/.venv/bin/python -m beebot.hardware.cube_outcomes measure --set layouts --trials 20 --moves 8 --drive '{"torque": 0.3}' --setting hinge-e03 --seed $s --workers 6 --out runs/cube_redesign/layouts-hinge-e03-s$s.jsonl > logs/layouts-hinge-e03-s$s.log 2>&1
done
echo done > logs/hinge-seeds.done
