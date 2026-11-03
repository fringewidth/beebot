#!/bin/zsh
# Run Rivals pairs one after another on a single GPU slot, each to completion (supervised).
#     ./queue_rivals.sh <after-run> <game:seed> [<game:seed> ...]
# Waits for runs/<after-run>/summary.json first. Extra train-rivals flags go in RIVALS_FLAGS.
after=$1
shift
while [[ ! -f runs/$after/summary.json ]]; do sleep 60; done
for job in "$@"; do
    game=${job%%:*}
    seed=${job##*:}
    TRAINER=train-rivals ./supervise.sh v6-rivals-$game-s$seed --game $game --seed $seed --minutes 360 ${=RIVALS_FLAGS}
done
