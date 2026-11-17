#!/bin/zsh
# Start the move-37 search once the GPU job before it exits (one Mac GPU job at a time).
#     nohup v1/scripts/move37_chain.sh WAIT_PID HOURS SEED > logs/move37-chain.out 2>&1 &
cd "$(dirname "$0")/../.."
wait_pid=$1
hours=${2:-3}
seed=${3:-0}
while kill -0 "$wait_pid" 2>/dev/null; do
  sleep 20
done
echo "$(date '+%F %T') pid $wait_pid gone, starting the search"
exec caffeinate -i ~/.local/bin/uv run python v1/scripts/move37_search.py \
  --init_run runs/kaggle/beebot-sft-final/runs/lm-expert-sft-final --checkpoint policy_latest.pt \
  --policy lm --device mps --hours "$hours" --batch 16 --seed "$seed" --out runs/move37 \
  >> logs/move37-search.out 2>> logs/move37-search.stderr
