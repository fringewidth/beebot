#!/bin/zsh
# Draws the v2 stores (full s0..s7, build s0, s1), three at a time (a new one starts as soon as one ends), and appends each store's
# join line (scripts/demos_v2_join.py) to logs/demos-v2.out when it finishes.
#   OPS_NODE=demos-v2 nohup scripts/draw_demos_v2_all.sh > logs/demos-v2-all.out 2>&1 &
# A store whose directory already has a `done` file is skipped, so the script can be started again.
cd "$(dirname "$0")/.." || exit 1

wait_for_memory() {
  while true; do
    free=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d '%')
    [ -n "$free" ] && [ "$free" -ge 10 ] && return
    echo "$(date +%T) free memory ${free}%, waiting"
    sleep 120
  done
}

one() {
  kind=$1; seed=$2
  out=data/vla/voxel_${kind}_v2/s$seed
  if [ ! -f $out/done ]; then
    scripts/draw_demos_v2.sh $kind $seed
  fi
  if [ "$kind" = full ]; then
    dirs=(data/vla/haiku/run-s$seed data/vla/haiku/run-s$seed-later)
  else
    dirs=(data/vla/haiku/build-s$seed)
  fi
  nice -n 10 .venv/bin/python scripts/demos_v2_join.py $out $dirs --name $kind-s$seed --append logs/demos-v2.out > logs/demos-v2-join-$kind-s$seed.out 2>&1
  echo "$(date +%T) finished $kind s$seed"
}

jobs_list=("full 0" "full 1" "full 2" "full 3" "full 4" "full 5" "full 6" "full 7" "build 0" "build 1")
# `jobs -r` reports nothing in a non-interactive zsh (all ten started at once on 2026-11-12), so track the pids.
MAX=${MAX:-3}
pids=()
for job in $jobs_list; do
  while true; do
    alive=()
    for p in $pids; do kill -0 $p 2>/dev/null && alive+=($p); done
    pids=($alive)
    [ ${#pids} -lt $MAX ] && break
    sleep 20
  done
  wait_for_memory
  one ${=job} &
  pids+=($!)
  sleep 60
done
wait
echo "$(date +%T) all stores finished"
