#!/bin/zsh
# Re-caption every text2shape and objaverse_cap3d shape with Haiku (claude -p), then check each new caption
# against its grey render (Haiku vision). Both steps are resumable: re-run this script after a stop.
# Progress goes to ops every 3 minutes (recaption_items_done, recaption_vision_done).
#
#     caffeinate -i zsh scripts/recaption_all.sh > logs/recaption-all.out 2>&1 &

set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
OUT=data/goal_library_v2/recaptions.jsonl
VISION=data/goal_library_v2/recaption_vision
OPS=(/usr/bin/python3 "$HOME/ops/ops.py" -p beebot)

push_loop() {
  while true; do
    n=$( [ -f $OUT ] && wc -l < $OUT || echo 0 )
    v=$( [ -f $VISION/vision.jsonl ] && wc -l < $VISION/vision.jsonl || echo 0 )
    "${OPS[@]}" push "recaption_items_done ${n// /}" "recaption_vision_done ${v// /}" --as "beebot-1d recaption" > /dev/null 2>&1
    sleep 180
  done
}
push_loop &
PUSHER=$!
trap 'kill $PUSHER 2>/dev/null' EXIT

nice .venv/bin/python scripts/recaption_run.py --out $OUT --workers 8 || exit 1
nice .venv/bin/python scripts/recaption_vision.py --recaptions $OUT --out $VISION --workers 4 || exit 1
n=$(wc -l < $OUT); v=$(wc -l < $VISION/vision.jsonl)
"${OPS[@]}" push "recaption_items_done ${n// /}" "recaption_vision_done ${v// /}" --as "beebot-1d recaption" > /dev/null 2>&1
echo "all done"
