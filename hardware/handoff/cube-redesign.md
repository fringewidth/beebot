# Handoff: cube redesign (alternative actuation to the flywheel flip)

Updated 2026-11-12 15:05. Done; nothing running. Worktree `.claude/worktrees/cube-redesign`, branch `cube-redesign` (from main 6494b03), commits 22004f3, 919814e, e29be6c, ca8776c, 03b37ef. Not merged.

## Result
1. Survey and ranking: hardware/docs/HARDWARE.md "Redesign options". No published lattice module meets 0.80 / 0.03 at 5 cm in 1 g; the reliable ones are quasi-static and closed loop (TERMES, Roombots). Ranked: (1) latched-edge servo pivot ("hinge cube"), (2) relative robots on passive blocks, (3) closed-loop reaction wheel on the present cube.
1. Model: `hardware/src/beebot/hardware/cube_hinge.py` (HingeSwarm, Drive); `cube_outcomes measure --drive '{json}'`. Tests `hardware/tests/test_cube_hinge.py`.
1. hinge-e03 (0.3 N m): ten-cube walks 0.82 reached / 0.021 harmful (759 pivots, seed 0; m14-sel same seed 0.48 / 0.053); stores 0.93 / 0.010. Two thirds of the walk harm is a cube held up only by the mover dropping (planner issue). Rests on dropping box contacts along shared edges (chamfered edges): with them kept, 0.73 / 0.027.

## Open
1. Walk seeds 1000 and 2000; 0.15 N m runs (motor margin).
1. One wall crawl scattered six cubes (likely solver, unexamined): hardware/runs/cube_redesign/layouts-hinge-e03.jsonl, layout wall seed 2.
1. Bench test of a latch + motor at one edge; mechanism unknown.
1. Grid rule: refuse moves that leave a cube held up only by the mover.
