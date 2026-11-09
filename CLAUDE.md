# Beebot

## Compute

Kaggle GPU quota is the scarcest resource. Every Kaggle minute should be a gradient step.

1. Keep the Mac training around the clock. It does about 0.6 of a T4 (expert SFT at batch 16: ~35 steps/min vs ~58 on one T4), so running nonstop it matches our whole weekly Kaggle quota. Wrap long runs in `caffeinate -i` so the machine does not sleep mid-run. When the Mac would otherwise be idle, queue the next experiment.
1. No evaluations on Kaggle. Kaggle runs train and save step-named checkpoints only; the Mac scores those checkpoints (scorecard, grounding, language). In-kernel evals cost 115-140 minutes of a 12 h session in November 2026 (`beebot-sft-long`, `beebot-sft-30k`, `beebot-vox-gen1`).
1. No full-length A/Bs on Kaggle. Settle a comparison on the Mac over 1-2 hours by validation loss; the scorecard mean moves about 0.03 between neighbouring evals and cannot rank close arms. If a Kaggle A/B is unavoidable, kill the losing arm once validation losses separate and give its GPU to the winner (DDP). In `beebot-sft-long` the weight-decay arms separated by 80-120 minutes and both still ran 500.

## Sharing the Mac and Kaggle between sessions

Several Claude sessions run experiments in this repo at once. Before using the Mac GPU or any Kaggle account, read `BOARD.md` at the repo root and follow its protocol: claim the slot or join the queue first, release it when done. `BOARD.md` is gitignored; if it is missing, recreate it from the protocol in git history (commit that added this section).
