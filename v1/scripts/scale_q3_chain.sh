#!/usr/bin/env bash
# The Qwen3 dense size curve (ops scale-curve-qwen3): pull the two Kaggle kernels when they finish,
# then score move loss on shapes never seen in training (the held-out store's heart/triangle item and
# the unseen-shape store), one GPU job at a time in priority order: the pre-registered steps 1500
# (with the null instruction) and 1000 for 0.6B, 1.7B and 4B at LR x1, then the 4B at LR x0.5, then
# the floors (step 0) and step 500. Results: logs/scale-q3.out, per-episode npz in logs/scale-q3/.
#
#     nohup caffeinate -i bash v1/scripts/scale_q3_chain.sh > logs/scale-q3-chain.out 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
export PATH="$HOME/.local/bin:$PATH"
TEAMMATE_TOKEN="${KAGGLE_TEAMMATE_TOKEN:-}"  # path to the teammate account's access_token file
run_of() {
  case $1 in
    q3-0.6b) echo runs/kaggle/beebot-q3-b/runs/q3-0.6b-x1 ;;
    q3-1.7b) echo runs/kaggle/beebot-q3-a/runs/q3-1.7b-x1 ;;
    q3-4b) echo runs/kaggle/beebot-q3-a/runs/q3-4b-x1 ;;
    q3-4b-half) echo runs/kaggle/beebot-q3-b/runs/q3-4b-x0.5 ;;
  esac
}
pull_when_done() {  # slug, then "teammate" for the teammate account
  local slug=$1
  [ "$2" = teammate ] && export KAGGLE_API_TOKEN="$(cat "$TEAMMATE_TOKEN")"  # runs in a subshell (&): no leak
  until uv run python v1/scripts/vla_kaggle.py status "$slug" 2>&1 | grep -qE "COMPLETE|ERROR|CANCEL"; do sleep 60; done
  echo "$(date '+%H:%M:%S') pulling $slug"
  uv run python v1/scripts/vla_kaggle.py pull "$slug" 2>&1 | tail -2
}
pull_when_done beebot-q3-b teammate &
pull_when_done beebot-q3-a &
JOBS=("q3-0.6b 1500 --null" "q3-1.7b 1500 --null" "q3-4b 1500 --null" "q3-0.6b 1000" "q3-1.7b 1000" "q3-4b 1000"
      "q3-4b-half 1500 --null" "q3-4b-half 1000" "q3-0.6b 0" "q3-1.7b 0" "q3-4b 0" "q3-0.6b 500" "q3-1.7b 500" "q3-4b 500")
STORES="data/vla/experts_heldout_g32_s1 data/vla/experts_heldout_g32_s2 data/vla/experts_newshapes_g32_s3 data/vla/experts_newshapes_g32_s4"
DONE="|"
while [ "$(echo "$DONE" | tr -cd '|' | wc -c)" -le ${#JOBS[@]} ]; do
  ran=0
  for job in "${JOBS[@]}"; do
    case $DONE in *"|$job|"*) continue ;; esac
    set -- $job
    arm=$1; step=$2; shift 2
    run=$(run_of "$arm")
    [ -f "$run/policy_step$step.pt" ] && [ -f "$run/config.json" ] || continue
    echo "$(date '+%H:%M:%S') start $arm step $step $*"
    .venv/bin/python v1/scripts/scale_curve_score.py "$run" --arm "$arm" --steps "$step" --store $STORES \
      --items shapes,new_shapes --windows 2 "$@" --log logs/scale-q3.out --per_episode logs/scale-q3
    rc=$?
    echo "$(date '+%H:%M:%S') end $arm step $step exit $rc"
    DONE="$DONE$job|"
    ran=1
    break  # back to the top: a higher-priority checkpoint may have arrived meanwhile
  done
  [ $ran -eq 0 ] && sleep 60
done
wait
echo Q3_CHAIN_DONE
