#!/usr/bin/env bash
# Per-size learning rate for the Qwen3 size curve (ops scale-curve-qwen3, pre-registered 2026-11-18
# 00:40): pull the sweep kernels beebot-q3-c..f, then on the Mac GPU, one job at a time:
#   1. in-distribution validation loss (each run's own held-out split) at step 1500 for every run;
#      the LR with the lowest one is each size's pick (never the unseen-shape test);
#   2. unseen-shape loss at step 1500 for every run, the picks first.
# Waits for scale_q3_chain.sh to finish the x1 4B at step 1500, then stops that chain (its remaining
# jobs are the lower-priority step 0/500/1000 scores). Logs: logs/scale-q3-val.out (validation),
# logs/scale-q3.out (unseen shapes), per-episode npz in logs/scale-q3/.
#
#     nohup caffeinate -i bash v1/scripts/scale_q3_sweep_chain.sh > logs/scale-q3-sweep.out 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
RVITM="$HOME/.kaggle-friends/rvitm/access_token"
STORES="data/vla/experts_heldout_g32_s1 data/vla/experts_heldout_g32_s2 data/vla/experts_newshapes_g32_s3 data/vla/experts_newshapes_g32_s4"
ARMS="q3-0.6b-x0.25 q3-0.6b-x0.5 q3-0.6b-x1 q3-0.6b-x2 q3-1.7b-x0.25 q3-1.7b-x0.5 q3-1.7b-x1 q3-1.7b-x2 q3-4b-x0.25 q3-4b-x0.5 q3-4b-x1 q3-4b-x2"
run_of() {
  case $1 in
    q3-0.6b-x0.25) echo runs/kaggle/beebot-q3-f/runs/q3-0.6b-x0.25 ;; q3-0.6b-x0.5) echo runs/kaggle/beebot-q3-e/runs/q3-0.6b-x0.5 ;;
    q3-0.6b-x1) echo runs/kaggle/beebot-q3-b/runs/q3-0.6b-x1 ;; q3-0.6b-x2) echo runs/kaggle/beebot-q3-e/runs/q3-0.6b-x2 ;;
    q3-1.7b-x0.25) echo runs/kaggle/beebot-q3-d/runs/q3-1.7b-x0.25 ;; q3-1.7b-x0.5) echo runs/kaggle/beebot-q3-c/runs/q3-1.7b-x0.5 ;;
    q3-1.7b-x1) echo runs/kaggle/beebot-q3-a/runs/q3-1.7b-x1 ;; q3-1.7b-x2) echo runs/kaggle/beebot-q3-d/runs/q3-1.7b-x2 ;;
    q3-4b-x0.25) echo runs/kaggle/beebot-q3-c/runs/q3-4b-x0.25 ;; q3-4b-x0.5) echo runs/kaggle/beebot-q3-b/runs/q3-4b-x0.5 ;;
    q3-4b-x1) echo runs/kaggle/beebot-q3-a/runs/q3-4b-x1 ;; q3-4b-x2) echo runs/kaggle/beebot-q3-f/runs/q3-4b-x2 ;;
  esac
}
label_of() {  # the unseen-shape log's label (the first chain's labels for the runs it scored)
  case $1 in q3-4b-x0.5) echo q3-4b-half ;; *-x1) echo "${1%-x1}" ;; *) echo "$1" ;; esac
}
pull_when_done() {  # slug, then "rvitm" for the teammate account
  local slug=$1
  [ "$2" = rvitm ] && export KAGGLE_API_TOKEN="$(cat "$RVITM")"  # runs in a subshell (&): no leak
  until uv run python v1/scripts/vla_kaggle.py status "$slug" 2>&1 | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 60; done
  echo "$(date '+%H:%M:%S') pulling $slug"
  uv run python v1/scripts/vla_kaggle.py pull "$slug" 2>&1 | tail -1
}
pull_when_done beebot-q3-c &
pull_when_done beebot-q3-d &
pull_when_done beebot-q3-e rvitm &
pull_when_done beebot-q3-f rvitm &
until grep -q "end q3-4b step 1500" logs/scale-q3-chain.out; do sleep 30; done
pkill -f "scale_q3_chain.sh"; sleep 2; pkill -f "scale_curve_score.py.*--log logs/scale-q3.out"
echo "$(date '+%H:%M:%S') stopped scale_q3_chain.sh after the x1 4B"

ready() { [ -f "$(run_of "$1")/policy_step1500.pt" ] && [ -f "$(run_of "$1")/config.json" ]; }
DONE="|"
while [ "$(echo "$DONE" | tr -cd '|' | wc -c)" -le 24 ]; do
  ran=0
  for arm in $ARMS; do  # stage 1: validation loss for every run
    case $DONE in *"|val $arm|"*) continue ;; esac
    ready "$arm" || continue
    echo "$(date '+%H:%M:%S') start val $arm"
    .venv/bin/python v1/scripts/scale_curve_score.py "$(run_of "$arm")" --arm "$arm" --steps 1500 --log logs/scale-q3-val.out
    echo "$(date '+%H:%M:%S') end val $arm exit $?"
    DONE="${DONE}val $arm|"; ran=1; break
  done
  [ $ran -eq 1 ] && continue
  # stage 2 order: each size's current pick (lowest validation loss so far) first, then the rest
  PICKS=$(.venv/bin/python -c "
import json
best = {}
for line in open('logs/scale-q3-val.out'):
    r = json.loads(line)
    if r.get('step') != 1500 or 'demo_val_loss' not in r: continue
    size = r['arm'].rsplit('-x', 1)[0]
    if size not in best or r['demo_val_loss'] < best[size][1]: best[size] = (r['arm'], r['demo_val_loss'])
print(' '.join(a for a, _ in best.values()))
" 2>/dev/null)
  for arm in $PICKS $ARMS; do  # stage 2: unseen shapes (runs scored by the first chain are skipped by the scorer)
    case $DONE in *"|test $arm|"*) continue ;; esac
    ready "$arm" || continue
    echo "$(date '+%H:%M:%S') start test $arm"
    .venv/bin/python v1/scripts/scale_curve_score.py "$(run_of "$arm")" --arm "$(label_of "$arm")" --steps 1500 --store $STORES \
      --items shapes,new_shapes --windows 2 --log logs/scale-q3.out --per_episode logs/scale-q3
    echo "$(date '+%H:%M:%S') end test $arm exit $?"
    DONE="${DONE}test $arm|"; ran=1; break
  done
  [ $ran -eq 0 ] && sleep 60
done
wait
echo Q3_SWEEP_DONE
