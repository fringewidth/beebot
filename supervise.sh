#!/bin/zsh
# Run a train-feast run to completion, resuming from its last checkpoint whenever the
# process dies (the Metal driver has crashed long MPS runs on this machine).
#     ./supervise.sh <name> [train-feast flags...]
# A run is complete once runs/<name>/summary.json exists.
name=$1
shift
mkdir -p logs
while [[ ! -f runs/$name/summary.json ]]; do
    if [[ -f runs/$name/checkpoint.json ]]; then
        echo "supervise: resuming $name $(date)" >> logs/$name.out
        caffeinate -is .venv/bin/train-feast --name $name --resume true "$@" >> logs/$name.out 2>&1
    else
        caffeinate -is .venv/bin/train-feast --name $name "$@" >> logs/$name.out 2>&1
    fi
    [[ -f runs/$name/summary.json ]] || sleep 30
done
