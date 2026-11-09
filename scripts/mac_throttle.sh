#!/bin/zsh
# Keep the Mac usable during work hours while experiments run around the clock: every 5 minutes,
# move each beebot python job (training, scorecards, data generation) into macOS background
# priority (PRIO_DARWIN_BG: efficiency cores, throttled I/O, lower GPU priority) during work hours,
# and back out of it outside them. Jobs started at any time are picked up on the next pass.
#
#     nohup scripts/mac_throttle.sh > logs/mac-throttle.out 2>&1 &
#
# WORK_DAYS: day numbers (1 = Monday ... 7 = Sunday); WORK_START / WORK_END: HHMM, local time.

WORK_DAYS=${WORK_DAYS:-12345}
WORK_START=${WORK_START:-0930}
WORK_END=${WORK_END:-1930}
mode=""
while true; do
  day=$(date +%u)
  now=$(date +%H%M)
  if [[ $WORK_DAYS == *$day* && $now -ge $WORK_START && $now -lt $WORK_END ]]; then
    want=background
  else
    want=full
  fi
  for pid in $(pgrep -f "python.* -m beebot\."); do
    if [[ $want == background ]]; then
      taskpolicy -b -p $pid 2>/dev/null
    else
      taskpolicy -B -p $pid 2>/dev/null
    fi
  done
  if [[ $want != $mode ]]; then
    echo "$(date '+%Y-%m-%d %H:%M') $want"
    mode=$want
  fi
  sleep 300
done
