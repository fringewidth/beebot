set -e
cd /Users/hrishik/beebot/.claude/worktrees/cube-redesign
run() { PYTHONPATH=src /Users/hrishik/beebot/.venv/bin/python -m beebot.hardware.cube_outcomes measure --set layouts --trials 20 --moves 8 --drive "$2" --setting $1 --seed 0 --workers 6 --out runs/cube_redesign/layouts-$1.jsonl > logs/layouts-$1.log 2>&1; }
run hinge-t015 '{"torque": 0.15}'
run hinge-t010 '{"torque": 0.1}'
run hinge-nofloorlatch '{"torque": 0.3, "floor_latch": false}'
run hinge-catch03 '{"torque": 0.3, "catch": 0.03}'
run hinge-squareedges '{"torque": 0.3, "edges": false}'
echo done > logs/stress.done
