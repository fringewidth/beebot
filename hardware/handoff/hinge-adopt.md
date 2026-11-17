# hinge-adopt handoff (updated 2026-11-12)

1. Merge: main fast-forwarded to cube-redesign 03b37ef (no conflicts; main 6494b03 was its base). Untracked seed/stress rows and scripts copied from the worktree into hardware/runs/cube_redesign.
1. Overhang rule: world.overhangs, applied in world.rule_table for every rule (scenes with gravity). A move is forbidden when, with the mover at its target, fewer robots are supported than before. In the hinge walks 110 of 2249 moves were such overhangs (MuJoCo harm on them 32 = 0.014 per move).
1. Table: v2/src/beebot/vla/voxel/outcomes/mujoco_hinge_v1.json via `cube_outcomes fit --legal` (new flag, drops 118 moves the rule forbids), 2511 moves / 64 classes. DEMO_MOVES = mujoco_hinge_v1; OUTCOME_MODELS keeps the old tables.
1. Gap: hardware/runs/cube_redesign/holdout_gap.py -> hardware/runs/cube_redesign/holdout-gap-hinge-v1.txt.
1. Experts: v2/scripts/expert_harm.py gained --overhangs (old rule) and "fell"; four runs writing hardware/runs/cube_redesign/experts/{v3,hinge}-{old,rule}.json.
1. Closed loop: closed_loop.make_swarm + --drive JSON (HingeSwarm); move_together defers to HingeSwarm.moves for several movers. Smoke: hardware/runs/closed_loop/goto-hinge-smoke.{npz,mp4}, log logs/hinge-adopt-closed-loop-smoke.out.
1. Left: README/HARDWARE notes, tests, commit, push main + tpu-port.

## Results so far
1. Held-out gap (stores + other two seeds): every move harm MuJoCo/grid 0.021/0.024, 0.030/0.022, 0.021/0.036, TV 0.028/0.036/0.057; moves the grid would make: harm 0.003/0.002, 0.003/0.001, 0.004/0.001, TV 0.054/0.003/0.020.
1. Experts (60 scenes): v3 before 0.021/0.63/0.55/62/1.9 (reproduced); v3+rule 0.021/0.62/0.53/60/1.95; hinge before 0.0018/0.95/0.99/116/0.078 (fell 0.0072); hinge+rule 0.0017/0.96/0.98/114/0.026 (fell 0.0003).
1. Closed-loop smoke: goto index 0, 24 steps, passed at step 13, 155 moves, 0 harmful; hardware/runs/closed_loop/goto-hinge-smoke.mp4.
1. Docs: hardware/docs/HARDWARE.md "Adopting the hinge cube (2026-11-12)", voxel README rule + tables + --outcomes row.
1. Done: full suite 638 passed; commit 0cdcec0 pushed to main and tpu-port (fast-forward). Branch cube-control untouched.
