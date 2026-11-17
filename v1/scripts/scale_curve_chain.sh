#!/bin/zsh
# Model-size scale curve for the 2D swarm policy: fresh LoRA (rank 64) on Qwen3.5-0.8B, 2B and 4B, one
# after another on the Mac GPU, with identical data, data order, batch and step count, logging the held-out
# demonstration loss every 100 steps to logs/scale-curve.out. Optional control arm: 0.8B with random weights.
#
#   caffeinate -i zsh v1/scripts/scale_curve_chain.sh [arm ...]   # ops:scale-curve
#
# Arms default to 0.8b 2b 4b; add 0.8b-random for the control. Each arm waits until no other beebot GPU
# job (expert_sft, think_train) is running, then runs to its step cap. Plot: v1/scripts/scale_curve_plot.py.
cd "$(dirname "$0")/../.."
UV=${UV:-$HOME/.local/bin/uv}
STEPS=${STEPS:-800}
arms=(${@:-0.8b 2b 4b})
for arm in $arms; do
  case $arm in
    0.8b) model=Qwen/Qwen3.5-0.8B; extra=() ;;
    2b) model=Qwen/Qwen3.5-2B; extra=() ;;
    4b) model=Qwen/Qwen3.5-4B; extra=() ;;
    0.8b-random) model=Qwen/Qwen3.5-0.8B; extra=(--random_init) ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  while pgrep -f "beebot.vla.expert_sft|beebot.vla.voxel.think_train|beebot.vla.lm_train" >/dev/null; do sleep 30; done
  echo "$(date '+%F %T') start $arm" >> logs/scale-curve.out
  $UV run python -m beebot.vla.expert_sft train --name scale-curve-$arm --arm $arm --fresh_model $model $extra \
    --weights_dtype bf16 --checkpointing --store data/vla/experts_g32 data/vla/experts_herd_g32 \
    --caption_share 0 --distill_weight 0 --loss_only --batch 16 --max_steps $STEPS --eval_every_steps 100 \
    --val_batches 16 --minutes 100000 --device mps --seed 0 >> logs/scale-curve.out 2>&1  # ops:scale-curve
  rc=$?
  echo "$(date '+%F %T') end $arm exit $rc" >> logs/scale-curve.out
done
$UV run python v1/scripts/scale_curve_plot.py --log logs/scale-curve.out --out runs/scale-curve > /dev/null
