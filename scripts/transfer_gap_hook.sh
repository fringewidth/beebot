#!/bin/zsh
# Chain hook (mac_chain.sh runs runs/<name>/extra_eval.sh <checkpoint> after each segment's basics eval):
# the sim-to-sim transfer gap of the checkpoint (scripts/transfer_gap.py). Cannot hang the chain: the
# script ends itself after DEADLINE minutes, a shell watchdog kills it 5 minutes later, and this hook
# always exits 0 so training goes on.
cd "${0:A:h}/.." || exit 0
checkpoint=${1:-}
[[ -z $checkpoint ]] && { echo "transfer_gap hook: no checkpoint"; exit 0; }
DEADLINE=${DEADLINE:-40}
PY=${PY:-.venv/bin/python}
echo "transfer_gap: $checkpoint ($(date))"
nice -n 10 $PY scripts/transfer_gap.py --checkpoint "$checkpoint" --device mps --deadline $DEADLINE \
  --out logs/transfer-gap.jsonl &
pid=$!
( sleep $(( (DEADLINE + 5) * 60 )); kill -9 $pid 2>/dev/null ) &
guard=$!
wait $pid
echo "transfer_gap: exited $? ($(date))"
kill $guard 2>/dev/null
exit 0
