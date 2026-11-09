# Beebot

## Compute

Kaggle GPU quota is the scarcest resource. Every Kaggle minute should be a gradient step.

1. Keep the Mac training around the clock. It does about 0.6 of a T4 (expert SFT at batch 16: ~35 steps/min vs ~58 on one T4), so running nonstop it matches our whole weekly Kaggle quota. Wrap long runs in `caffeinate -i` so the machine does not sleep mid-run. When the Mac would otherwise be idle, queue the next experiment.
1. No evaluations on Kaggle. Kaggle runs train and save step-named checkpoints only; the Mac scores those checkpoints (scorecard, grounding, language). In-kernel evals cost 115-140 minutes of a 12 h session in November 2026 (`beebot-sft-long`, `beebot-sft-30k`, `beebot-vox-gen1`).
1. No full-length A/Bs on Kaggle. Settle a comparison on the Mac over 1-2 hours by validation loss; the scorecard mean moves about 0.03 between neighbouring evals and cannot rank close arms. If a Kaggle A/B is unavoidable, kill the losing arm once validation losses separate and give its GPU to the winner (DDP). In `beebot-sft-long` the weight-decay arms separated by 80-120 minutes and both still ran 500.

## Sharing the Mac and Kaggle between sessions

Several Claude sessions run experiments in this repo at once. `BOARD.md` at the repo root holds the live state: who runs what on the Mac GPU and Kaggle, the queues, messages between sessions and a short log. It is gitignored (live state; the repo is public). If it is missing, recreate it with sections Mac (Running, Queue), Kaggle (Accounts, Queue), Messages, Log.

1. Identify yourself by the first 8 characters of your session id (the uuid in your scratchpad path) plus a 1-3 word topic, e.g. `c5a781cd compute`.
1. Read `BOARD.md` before starting anything on the Mac GPU or Kaggle, and edit it before you launch. Edit with the Edit tool, never Write or shell redirection: Edit refuses if someone changed the file since your read, so re-read and retry.
1. A claim lists the PID (Mac) or kernel slug (Kaggle), the command or run name, start time and expected end. Without a PID or slug it is a reservation and expires after 15 minutes.
1. Trust processes over the board. Check `pgrep -fl beebot` and `ps -p <pid>` before believing a claim or concluding the Mac is free. A claim whose PID is dead is stale: move it to Log with "died, found by <you>" and message its owner.
1. When you finish, move your claim to Log with the result's location (log file, run directory) in one line, and hand the slot to the first entry in the queue: message that session (ListAgents, SendMessage) if you can reach it.
1. Queues are first come, first served. Only the user reorders or preempts. If two sessions need a decision (priority, killing a run), ask the user through ntfy.sh/hrishik rather than settling it between sessions.
1. Mac (M4 Pro, 24 GB): one GPU job at a time (training or evaluation on `--device mps`). Two jobs share the GPU and both slow down, and 24 GB does not hold two LM runs safely. Never train or evaluate an LM on `--device cpu`. Light work needs no claim: scripts, demo servers, Kaggle polling, plotting.
1. Kaggle: keep each account's running kernel and hours left in `BOARD.md`; a session on an account uses both T4s.
1. Keep Log to the last ~20 lines and delete messages once acted on.
