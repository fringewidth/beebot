#!/bin/zsh
# The Mac's training chain for a think-then-act run (voxel/think_train.py): two-hour --resume
# segments of runs/<name>, each followed by the basics eval (voxel/basics.py) of the segment's
# newest checkpoint, until the schedule's last step. One GPU job at a time (--device mps). Every
# segment starts a new process from this checkout, so it runs main's current code.
#
#     nohup caffeinate -i scripts/mac_chain.sh think-08b-cam > logs/mac-chain-think-08b-cam.out 2>&1 &
#
# The first segment (no runs/<name>/resume.pt yet) starts from INIT (weights only: a policy_step*.pt
# or a resume.pt; empty for the LM as downloaded). Logs: logs/<name>.out (training, appended),
# logs/basics-<name>.out and logs/basics-<name>.jsonl (per-episode rows, appended).
#
# NAME: the run; INIT, FRAME_DIRS, STEPS, SEGMENT_MINUTES: environment overrides. After each basics eval the
# chain runs runs/<name>/extra_eval.sh <checkpoint> if that file exists and is executable (logs/extra-eval-<name>.out).

set -u
NAME=${1:?usage: scripts/mac_chain.sh <run name>}
INIT=${INIT:-}
FRAME_DIRS=${FRAME_DIRS:-data/vla/haiku/frame-camera}
STEPS=${STEPS:-3600}
SEGMENT_MINUTES=${SEGMENT_MINUTES:-120}
PY=${PY:-.venv/bin/python}

cd "${0:A:h}/.."
RUN=runs/$NAME
STORES=data/vla/voxel_build/s0,data/vla/voxel_build/s1$(for i in {0..7}; do printf ',data/vla/voxel_full/s%d' $i; done)
HAIKU=data/vla/haiku/build-s0,data/vla/haiku/build-s1$(for i in {0..7}; do printf ',data/vla/haiku/run-s%d,data/vla/haiku/run-s%d-later' $i $i; done)
DATA_KWARGS="{\"mismatch\": \"drop\", \"mix\": \"runs/task_mix.json\", \"quarantine\": \"runs/quarantine.json\", \"frame_dirs\": [\"$FRAME_DIRS\"]}"

say() { echo "$(date +%H:%M) $*"; }

last_step() {
  [[ -f $RUN/history.jsonl ]] || { echo 0; return; }
  $PY -c "import json,sys; print(max([json.loads(l).get('step', 0) for l in open(sys.argv[1]) if l.strip()] or [0]))" $RUN/history.jsonl
}

while true; do
  if [[ -f $RUN/resume.pt ]]; then
    start=(--resume $RUN/resume.pt)
  else
    start=(--init "$INIT")
  fi
  say "segment from $(last_step) (${start[*]})"
  $PY -m beebot.vla.voxel.think_train --name $NAME --model Qwen/Qwen3.5-0.8B --data think \
    --stores $STORES --haiku_dirs $HAIKU --data_kwargs "$DATA_KWARGS" --views scene \
    --val_seed 0 --val_examples 128 --channels 10 --lora_rank 0 --train_embeddings false \
    --train_vision false --train_projector true --weights_dtype fp32 "${start[@]}" \
    --batch 2 --accumulate 8 --bucket true --ce_chunk 512 --checkpointing true --precision auto \
    --lr_lm 2e-5 --lr_new 3e-4 --weight_decay 0.01 --warmup_steps 50 --min_lr_ratio 0.1 \
    --steps $STEPS --minutes $SEGMENT_MINUTES --distill_weight 1.0 --general hf --lm_batch 2 \
    --lm_length 256 --log_every 5 --val_every_minutes 60 --save_every_minutes 120 \
    --save_resume true --prefetch 4 --device mps --seed 0 >> logs/$NAME.out 2>&1
  code=$?
  checkpoint=$(ls -t $RUN/policy_step*.pt 2>/dev/null | head -1)
  if [[ -n $checkpoint ]]; then
    say "basics on $checkpoint"
    $PY -m beebot.vla.voxel.basics --checkpoint $checkpoint --arms policy --device mps \
      --out logs/basics-$NAME.jsonl >> logs/basics-$NAME.out 2>&1
  fi
  if [[ -n $checkpoint && -x $RUN/extra_eval.sh ]]; then
    say "extra eval on $checkpoint ($RUN/extra_eval.sh)"
    $RUN/extra_eval.sh $checkpoint >> logs/extra-eval-$NAME.out 2>&1
  fi
  step=$(last_step)
  say "step $step"
  if [[ $code -ne 0 ]]; then
    say "think_train exited with $code: chain stopped"
    exit $code
  fi
  if (( step >= STEPS )); then
    say "schedule done"
    exit 0
  fi
done
