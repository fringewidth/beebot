#!/bin/zsh
# Re-record the live-steering instruction (index 16) in MuJoCo, then on the raw grid (one Mac GPU job at a time).
cd "$(dirname "$0")/.."
for physics in mujoco grid; do
  caffeinate -i ~/.local/bin/uv run python scripts/field_demo_record.py --policy lm --physics $physics \
    --init_run runs/kaggle/beebot-sft-final/runs/lm-expert-sft-final --checkpoint policy_latest.pt \
    --only 16 --rollouts 8 --keep 8 --out runs/field-demo/recordings-steer-dwell-$physics.json \
    --log logs/field-demo-steer-dwell-$physics.out > logs/field-demo-steer-dwell-$physics.stdout 2> logs/field-demo-steer-dwell-$physics.stderr
done
