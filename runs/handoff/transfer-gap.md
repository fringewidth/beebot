# Transfer gap (sim-to-sim), 2026-11-13

What it measures: the same policy checkpoint on the same tasks, scored once in the voxel grid and once with its moves
carried out by cubes in MuJoCo. The number to follow per checkpoint is `transfer_gap`.

## Method

1. `scripts/transfer_gap.py --checkpoint <policy_stepN.pt>` draws 16 episodes (4 per basics category: side, corner,
   compass, steps) with fixed seeds (`--seed 4243`, same draws every run). Each is a basics episode: same phrases,
   camera side drawn per episode, instruction said as the user says it, board task and feasibility filter from
   `beebot.vla.voxel.basics`, scored by the suite's checker and pass mark.
1. Grid arm: `rollout.play` with the policy exactly as basics plays it (argmax, rule switch_cube, the training failure
   rates on the grid's pivots).
1. MuJoCo arm: the same episodes, a fresh actor, `closed_loop.Executor` in place of the grid's step. The hinge cube
   (`cube_hinge.HingeSwarm`, drive torque 0.3, the demo's), mode `simultaneous` (movers whose cells do not clash move
   in one wave), no synthetic failures: the physics decides. The policy sees only the grid read back from MuJoCo.
   Episodes are spread over 8 worker processes (MuJoCo does not scale in threads); the policy stays in the main process.
1. Hook: `runs/think-08b-cam/extra_eval.sh` (a copy of the logic is `scripts/transfer_gap_hook.sh`; the runs/ file
   just execs it) runs after each segment's basics eval, logs to `logs/extra-eval-think-08b-cam.out`, per-episode rows
   to `logs/transfer-gap.jsonl`. Safety: the script ends itself after `--deadline` (40 min, prints an `error` row, exit
   0), a shell guard kills it 5 minutes later, the hook always exits 0, any exception becomes an `error` row.

## Why not basics' own episodes

Only about 1 in 9 of basics' scenes is hostable (MuJoCo hosts a flat floor and cubes: no terrain, no cheese). So the
transfer episodes are fresh draws on flat, gravity, no-cheese scenes with 10-12 robots (basics: 6-115 robots, hills and
gaps). The grid score here is therefore not basics' score; read the gap and compare checkpoints with each other.

## Reading the numbers

1. `transfer_gap` = grid score - MuJoCo score on the same episodes (and `transfer_pass_gap` for pass rate). Positive:
   the policy loses to physics. Near 0: the grid is a faithful stand-in for what the policy does. Negative: physics was
   kinder than the grid's failure model (the grid draws 10 percent pivot failures, 23 percent for up-over; the hinge
   cube fails far less on a flat floor, 0.003 harmful per planned move).
1. 16 episodes is small: one episode moving from pass to fail is 0.0625 of pass rate. Track the trend over checkpoints
   and the per-category rows; do not rank two checkpoints that differ by less than about 0.05 in score.
1. `transfer_harmful_per_move`: share of attempted moves MuJoCo classed off / strayed / knock (cube left its cell
   wrongly, or knocked a neighbour).
1. `transfer_disagree_share` and `disagreements`: of the moves the grid let happen (the grid's step carries out every
   allowed move), the share MuJoCo did not complete as "reached", by the pivot class MuJoCo planned (floor_roll,
   floor_peel, convex_up, convex_down, ...; `none` = MuJoCo's plan refused or the cube was off its cell). `detail` gives
   reached / stayed / knock / off / strayed / blocked / refused.
1. `wall_seconds`: grid, mujoco, total.

## Known differences between the worlds

1. The grid applies its gravity and rule; MuJoCo's physics decides what stands (a cube off the lattice cannot move
   again; no snap-back here).
1. Grid failures are random draws per pivot (training condition); MuJoCo failures are physical and depend on the scene.
1. Move order and claims follow the grid's priority scores in both, but MuJoCo moves some in the same wave
   (simultaneous), which the grid does not model.
1. MuJoCo has no terrain or cheese, hence the flat scenes and demo-sized swarm.
1. The step cap is basics' 72 for both arms.

## Smoke results (no LM)

Scripted expert through the same code, 16 episodes: see the end of this file for the numbers and runtime.

Scripted expert, `--actor expert`, 16 episodes, 72 steps, 8 workers (this Mac, niced, GPU idle): grid score 1.000 and MuJoCo
score 1.000 (gap 0.000, all 32 plays pass), 2,857 MuJoCo moves, 2 harmful (0.0007 per move), 4.7 percent of moves
not completed as the grid has them (floor_peel 58 of 621 stayed, convex_up 33 of 248, concave_up 13 of 24, convex_down 17
of 228, 12 blocked by an earlier mover in the same step; floor_roll 0 of 1,605). Wall: grid 33 s, MuJoCo 897 s, total 934 s.
Sequential mode took about 5 times as long per move on one episode (60 s against 12 s for 63 moves), hence `simultaneous`.

## Runtime estimate for a policy checkpoint

MuJoCo about 15 minutes (a policy moves a different number of cubes than the expert; the expert's 2,857 moves is the
scale), the policy's two plays of 72 think-then-act steps over a batch of 16 on mps a few minutes more, plus model
load. Expect 20-30 minutes; the hard limit is 40 (the script ends itself), 45 (shell guard). Not measured with the LM:
the real policy path (`--actor policy`) was not run, since the GPU belongs to the training chain; it uses the same
`rollout.PolicyActor` as `closed_loop episode --checkpoint`. The first hook run will show the real time; if it is over
35 minutes lower `--per-category` or `--steps` in `scripts/transfer_gap_hook.sh`.
