#!/bin/zsh
# Score the night's checkpoints on one frozen set of test windows (each run's own config, but
# loaded now, so every run sees the same play data and the same windows). Appends to
# <run>/inference_sweep.jsonl and logs/vla-final-table.out.
cd /Users/hrishik/beebot
export OMP_NUM_THREADS=2
CUTOFF=${CUTOFF:-$(date +%s)}
echo "play cutoff $CUTOFF" >> logs/vla-final-table.out
sweep() { .venv/bin/python scripts/vla_inference_sweep.py "$1" --play-cutoff $CUTOFF "${@:2}" >> logs/vla-final-table.out 2>&1; }
sweep runs/vla-d-g32-play-goal --settings 4,1,10 4,2,10 1,1,10
sweep runs/vla-d-g32-play-masked-ki --settings 4,1,10 4,2,10 1,1,10
sweep runs/vla-d-g32-play-masked-ki --decode readout --settings 4,1,1 1,1,1
for run in vla-d-g32-play-readout vla-d-g32-play-readout-world; do
  sweep runs/$run --settings 4,1,1 4,2,1 1,1,1 1,2,1
done
echo finished >> logs/vla-final-table.out
