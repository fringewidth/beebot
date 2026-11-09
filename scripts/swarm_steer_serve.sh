#!/bin/zsh
# Keep the steering site up: restart the server if it exits (the tunnel, and so the URL, stays).
#     nohup caffeinate -i scripts/swarm_steer_serve.sh RUN_DIR [CHECKPOINT] > logs/swarm-steer.out 2>&1 &
cd "$(dirname "$0")/.."
while true; do
  uv run python scripts/swarm_steer.py --init_run "$1" --checkpoint "${2:-policy_latest.pt}"
  echo "$(date '+%F %T') server exited $?, restarting"
  sleep 3
done
