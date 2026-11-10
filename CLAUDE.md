# Beebot

## Compute

Kaggle GPU quota is the scarcest resource. Every Kaggle minute should be a gradient step.

1. Keep the Mac training around the clock. It does about 0.6 of a T4 (expert SFT at batch 16: ~35 steps/min vs ~58 on one T4), so running nonstop it matches our whole weekly Kaggle quota. Wrap long runs in `caffeinate -i` so the machine does not sleep mid-run. When the Mac would otherwise be idle, queue the next experiment.
1. No evaluations on Kaggle. Kaggle runs train and save step-named checkpoints only; the Mac scores those checkpoints (scorecard, grounding, language). In-kernel evals cost 115-140 minutes of a 12 h session in November 2026 (`beebot-sft-long`, `beebot-sft-30k`, `beebot-vox-gen1`).
1. No full-length A/Bs on Kaggle. Settle a comparison on the Mac over 1-2 hours by validation loss; the scorecard mean moves about 0.03 between neighbouring evals and cannot rank close arms. If a Kaggle A/B is unavoidable, kill the losing arm once validation losses separate and give its GPU to the winner (DDP). In `beebot-sft-long` the weight-decay arms separated by 80-120 minutes and both still ran 500.

## Plan, claims and numbers: `ops`

Several Claude sessions run experiments in this repo at once. The plan (goals, tasks, dependencies), who runs what on the Mac GPU and Kaggle, the queues and messages between sessions live in the `ops` repo (`~/ops`, project `beebot`), shown at https://hrishik.me/ops. Read `~/ops/README.md` once per session; it is short. `BOARD.md` is retired.

1. Identify yourself by your session name (the one ListAgents reports as "This session is ...", e.g. `beebot-b3`) plus a 1-3 word topic: `--as "beebot-b3 compute"` on every `ops` write. The name is also your address for SendMessage.
1. Run `ops board` before starting anything on the Mac GPU or Kaggle, and `ops claim <id> --pid <pid> --needs mac-gpu --cmd .. --ends ..` before you launch (Kaggle: `--needs kaggle-<account> --kaggle <owner/slug>`). A queued node (`--status queued`) is a reservation.
1. Every task has a title, a one-sentence summary and a why of at most two sentences, and reduces to one number (`--metric`, `--direction`). Push run numbers with `ops tail logs/<run>.out --label run=<run>` (it follows the JSON lines the trainers print) or `ops push`.
1. Trust processes over claims. `ops board` marks dead PIDs; also check `pgrep -fl beebot`. Finish a dead claim with `ops done <id> --failed --result "died, found by <you>"` and message its owner.
1. When you finish, `ops done <id> --result "<one line, with the log or run directory>"`, and hand the slot to the first queued node for that resource: message its session (ListAgents, SendMessage) if you can reach it.
1. A failure stays a node (`--failed`); add the workaround with `--mitigates <id>` and point dependents at it. Never delete nodes; drop them.
1. Queues are first come, first served. Only the user reorders or preempts. If two sessions need a decision (priority, killing a run), ask the user through ntfy.sh/hrishik rather than settling it between sessions.
1. Mac (M4 Pro, 24 GB): one GPU job at a time (training or evaluation on `--device mps`). Two jobs share the GPU and both slow down, and 24 GB does not hold two LM runs safely. Never train or evaluate an LM on `--device cpu`. Light work needs no claim: scripts, demo servers, Kaggle polling, plotting.
1. Kaggle: a session on an account uses both T4s. After launching or finishing, update hours left: `ops push 'kaggle_gpu_hours_left{account="<account>"} <hours>'`.
1. Delete messages once acted on (`ops msg --rm <id>`).
