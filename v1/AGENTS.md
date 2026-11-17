# v1: the 2D language-steered swarm

One pretrained language model steers about 100 robots on a 32 x 32 grid from plain English. This is
the system in the demo video (MuJoCo next to the raw grid). Work on it stopped on 2026-11-10, when the
project went all in on 3D (`v2/`). The checkpoint, the demo and the live steering page stay as they
are. The root `AGENTS.md` has the project summary; read it first.

## How it works

1. **World** (`src/beebot/vla/world.py`): a bounded grid with robots, terrain, coloured cheeses (4 x 4
   squares) and floor boxes. One robot per square; each step every robot picks one of 9 moves (8
   directions or stay). Robots push a cheese when enough of them push together (3 robots' worth on a
   32 x 32 grid) and eat it when they surround it. `src/beebot/vla/mujoco_world.py` runs the same policy
   with MuJoCo as the world: robots are flat cylinders on slide joints, and the grid is only what the
   policy sees (positions snapped to squares).
1. **Policy** (`src/beebot/vla/lm_policy.py`): Qwen3-0.6B with LoRA. The instruction goes in as text
   tokens and the grid as 4 x 4 patch tokens with learned row and column embeddings; text stays causal,
   scene tokens attend both ways. A linear readout turns each scene token into its squares' moves (9
   choices each) for the next 8 steps; one move per square is sampled and the policy re-plans every few
   steps.
1. **Tasks** (`src/beebot/vla/tasks.py`): generated scenes, instruction families with programmatic
   checkers, held-out wordings and compositions. `task_wordings.json` holds the wordings.
1. **Data** (`src/beebot/vla/experts.py`, `controller.py`): scripted experts turn a task into attractor
   points, and one shared controller turns attractors into a move field. Their episodes are the
   demonstrations.
1. **Training** (`src/beebot/vla/expert_sft.py`): behaviour cloning on the expert demonstrations, mixed
   with captioned play windows and distillation of the frozen LM on general text, so the language side
   is not lost. Trained on Kaggle T4s through `scripts/vla_kaggle.py` (about 40 T4 hours in total).
1. **Live steering** (`src/beebot/vla/steer.py`, `scripts/swarm_steer.py`): many swarms advanced
   together, one per visitor, each taking new text at any time.

The earlier designs are in `docs/`: `VLA.md` (first notes), `VLA_PLAN.md`, `VLA_HILLCLIMB.md` (the
custom-model hill-climb), `VLA_ARCHITECTURE.md` (why a plain readout beat a flow-matching action expert),
`VLA_RELATED_WORK.md`, and `docs/vla_explainer/`.

## Checkpoint and results

1. The final checkpoint (`config.json` and `policy_latest.pt`, LoRA and readout weights only) is public at
   https://www.kaggle.com/datasets/fringewidth/beebot-2d-policy. Put both files in one folder and pass it as
   `--init_run`. It is undertrained: training stopped when compute ran out, with the validation loss still
   falling (0.40 at the start, 0.31 at the last checkpoint, step 29177, the lowest of the run).
1. Fresh scenes, 8 rollouts each, MuJoCo / raw grid: eat red 8/7, either yellow or purple 7/7, clear the
   board 7/6, fill the blue box 7/8, move left 7/7. Steering (three commands one after another, each held
   at least 15 steps once it takes effect): last command completed 6/8 / 7/8, every command met 2/8 in
   both. Recorded by `scripts/field_demo_record.py`; the episodes are in `runs/field-demo/` (local).
1. What did not work, so do not repeat it without new evidence:
   1. Low-level RL on robot actions (`src/beebot/vla/lm_rl.py`, Kaggle `beebot-rl-gen1`): training success
      flat at 0.16-0.20 over about 200 updates. One scalar reward for ~6,000 robot actions per rollout.
   1. Language-level plans (`src/beebot/vla/plans.py`): even oracle plans reach only 0.10-0.33 on held-out
      logic tasks, because the executor eats forbidden cheeses on the way (~40% of episodes).
1. Known weak spots: cheeses in corners and pockets are hard to push or surround; whether to keep them
   out of sampling is an open decision.

## Running

All commands run from the repo root.

1. Demo video (two panels per task: MuJoCo, and the same scene on the raw grid):
   `uv run python v1/scripts/field_demo_record.py --physics mujoco` (and `--physics grid`), then
   `uv run python v1/scripts/field_demo_video.py --recordings runs/field-demo/recordings-mujoco-final.json --grid runs/field-demo/recordings-grid-final.json --out runs/field-demo`.
   `v1/scripts/steer_dwell_chain.sh` re-records the steering task in both worlds.
1. Live steering in a browser: `uv run python v1/scripts/swarm_steer.py --init_run <folder with the checkpoint>`.
   The static page with recorded runs is `v1/site/2d-demo/` (published by `.github/workflows/pages.yml`).
1. Grounding (does it follow the words or the scene): `uv run python v1/scripts/vla_grounding_eval.py RUN`.
1. Expert data: `uv run python -m beebot.vla.experts --out data/vla/experts_g32 --episodes 20000`.
1. Kaggle: `uv run python v1/scripts/vla_kaggle.py code` uploads the source (all four parts merged into one
   `src/`, since `beebot` is split over them), then `run`, `status`, `logs`, `pull`, `quota`. No
   evaluations on Kaggle: in-kernel evals cost about two hours of a 12 h session, so checkpoints are scored
   locally.
1. Every Kaggle launch is followed by a watcher, started in the background (`run_in_background`) right
   after the push: `.venv/bin/python v1/scripts/kaggle_watch.py OWNER SLUG` (a teammate's account:
   `OWNER=CONFIG_DIR`, the folder holding their token). It follows the log into `logs/SLUG.out` and exits,
   with one line also sent to the ntfy topic in `--ntfy` (default `$BEEBOT_NTFY`, none if unset), on the first of: a Traceback,
   argparse error, CUDA OOM or non-zero run exit (exit 10); no training step within 30 minutes (11); a NaN
   loss or one back above its first value after warmup (12); with `--compare A B --separate 0.05 --after 300`,
   one arm worse than the other at each of the last 4 steps both logged (13); the session ending (0, with a
   summary). It alerts only; `--kill-on a,c` also sends kill for those checks (`d` kills only the losing arm).
   Every threshold is a flag (`--help`). A check that fired is not fired again by a restarted watcher.
   `--replay runs/kaggle/SLUG/SLUG.log` replays a downloaded log offline in simulated time: on
   `beebot-scale-curve-a` the 2B arm is flagged at 26 minutes (of 152), on `beebot-scale-curve-b` the argparse
   error at 15.
1. Kill a run: `vla_kaggle.py` pushes every kernel (`run`, `lm-run`, `voxel-run`, `tpu-run`) with a kill switch
   (`scripts/kernel_control.py`, inlined) on a private ntfy topic named after the slug (`--no-control` for
   none). `.venv/bin/python v1/scripts/run_control.py kill SLUG` terminates every trainer and ends the session
   within about two minutes (outputs already written stay); `kill SLUG ARM` ends one trainer only (the losing
   arm of a comparison; e.g. `2b` for `scale-curve-2b`); `stop SLUG` lets trainers that use `beebot.control`
   save and exit and terminates the others; `run_control.py list` shows the known runs. The topic URL is a
   secret: it lives in `~/.config/beebot/runs/` and in the pushed copy under `data/` only, never in git or a
   log. `scripts/kaggle_control_smoke.py` is a CPU-only kernel (no quota) for testing all of this.
1. Tests: `uv run pytest -q v1/tests`.

## Shared code that lives here

`v2/` and `hardware/` import from v1, so a change here can break them; run their tests too.

1. `src/beebot/config.py`: `parse_config` (dataclass config as command-line flags) and `awake_seconds`.
1. `src/beebot/control.py` and `scripts/run_control.py`: stop or checkpoint a long run through a private
   ntfy topic; `scripts/kernel_control.py`: the Kaggle kernels' kill switch and log relay;
   `scripts/kaggle_watch.py`: the Mac-side watcher.
1. `src/beebot/vla/lm_policy.py`, `tasks.py`, `world.py`: the v2 voxel LM builds on them.
1. `scripts/vla_kaggle.py` and its kernel templates (`kaggle_train/`, `kaggle_lm/`, `kaggle_tpu/`,
   `kaggle_embed/`), `scripts/kaggle_follow.py`, `scripts/kaggle_rest.py`.
1. `scripts/ops_session_brief.sh` and `scripts/ops_launch_guard.py`: session-start and pre-launch hooks for
   the maintainers' local Claude Code setup, against their task tracker.
1. `scripts/reorg_2026_11_17.py` and `.json`: the record of the 2026-11-17 split (old path to new path).
