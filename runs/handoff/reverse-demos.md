# reverse-demos: demonstrations generated backwards from goal states

Written 2026-11-13. Prototype only: nothing reads it in training yet.

Code: `src/beebot/vla/voxel/reverse.py` (goals, the walk backwards, replay, store layout),
`scripts/reverse_demos.py` (check with throughput; GIFs for the browse page), tests in
`tests/test_vla_voxel_reverse.py`.

## Method

1. A goal generator scripts an ordered end state plus an instruction in a stranger's words. Four families:
   free-standing tower of 3-6 (single column, nothing touching it), crossing over a trench, letter or flat
   shape on the floor (L, T, C, U, H, X, J, plus, hollow square, line, 2x2), and an object pushed onto a floor
   patch (2x2x2 box, flat tile, or bar, light, never food). Every goal must stay the same when all robots STAY
   (`fixed_point`). Rule `switch_cube` with gravity on (the demo preset's rule), deterministic: no outcome
   model, no jitter.
1. Walking backwards: from S' we propose earlier states S with the joint codes a. A proposal is kept only if
   `world.step(S, a, rule)` gives exactly S', `world.refuse` lets every recorded move through, and
   `world.fall(S) == S` (S is at rest). So the recorded steps never contain a fall, an eaten object or a refused
   move. Proposals: single-robot moves come from `pivot_moves` on the scene with that robot removed (which
   sources could pivot into its cell). Joint proposals combine moves of several robots. Pushes are proposed as
   the object one cell back with the robots behind it pressing. The pressing robots do not move, as in
   `move_robots`.
1. Choosing a step: among the verified proposals we keep the one that raises the disorder potential most. The
   potential adds, per robot, its distance from the goal's cells, 0.3 x its horizontal distance from their
   centre, and minus its height above the terrain (a robot beside a terrain wall climbs it for free). It
   subtracts 0.5 for each pair of robots touching face to face. A step that lowers the potential is never
   taken. Up to 4 level steps in a row are allowed. At a dead end the walk backtracks depth first, at most 30
   times per episode, and never revisits a state.
1. Stop: when the scene looks like a natural start. That means every robot rests on the floor or terrain, at
   most one face-to-face pair per 4 robots, all robots but one at least 2 cells from the goal's cells, a mean
   distance of at least 4, and for pushes the object at least 2 cells from where it ends. Otherwise the walk
   stops at a dead end ("stalled") or at 96 steps. A stalled episode still replays correctly, but its start is
   not natural.
1. Push family: a phase machine runs first. The pushers step back one cell, then the object steps back one cell
   while they press, and this repeats 3-6 times. After that the scatter walk runs.
1. Output: `Episode` holds frames start..goal and per-voxel codes on each frame (STAY where no robot is: the
   store's format). `to_store` pads episodes with the goal plus STAY (a fixed point, so the padding replays too)
   and gives the dense fields `lm.world_at` reads. Labels are {e, w0, w1, family, text}.

## Numbers (200 episodes per family, seed 0, one thread, batches of 50; runs/reverse_demos/check.json)

| family | replay pass | natural start | steps mean / max | proposed | verified | accepted | backtracks | episodes/s | steps/s | natural steps/s |
|---|---|---|---|---|---|---|---|---|---|---|
| tower | 1.00 | 0.97 | 10.5 / 43 | 14740 | 8507 | 2365 | 269 | 4.6 | 47.8 | 43.1 |
| bridge | 1.00 | 0.77 | 11.1 / 53 | 21366 | 20763 | 3727 | 1499 | 2.5 | 28.2 | 10.7 |
| letter | 1.00 | 1.00 | 4.8 / 15 | 5652 | 5624 | 971 | 2 | 10.1 | 48.8 | 48.8 |
| push | 1.00 | 1.00 | 11.9 / 18 | 5187 | 5187 | 2387 | 9 | 12.3 | 145.8 | 145.8 |

1. Replay: all 800 episodes play forward from their start under `world.step` with the recorded codes. Every
   step is legal and matches the next frame, and every episode ends exactly in its goal, which is a fixed point.
1. Accepted per proposed is 0.16-0.17 for the scatter families. Several proposals per step are made by design
   and one is kept. Verified per proposed is 0.58 for towers (joint moves next to a column often lean on a
   co-mover or break support) and 0.97-1.00 for the others. Push is 0.46: two proposals per push-phase step.
1. Throughput per CPU core, counting goals and the walk, without the replay: 28-146 steps/s, i.e. that many
   (state, action) examples per second. Training eats about 0.7 examples/s on the Mac and 10-20/s on a Kaggle
   T4 pair, so one core is enough for the Mac and 1-2 cores are enough for Kaggle, even counting only natural
   bridge starts (10.7/s). The cost is in `world.step`/`refuse` (`overhangs` loops in Python) on the proposal
   batch, about 2 ms per proposal.

## What is hard / does not work

1. Bridges over air cannot be closed under the pivot rule. The last cube into a gap between two cubes (or a cube
   and the bank) must swing through one of them (`pivot_moves` swing cells), and a deck over air has no side
   support for the perpendicular pivot. So a width-1 deck at the plateau level never verifies backwards from
   the goal: the end robot cannot leave. The bridge family is therefore a solid crossing standing on the
   trench floor: trench 1-3 deep, 2-4 wide, at most 9 robots. A real deck would need scaffolding (helpers that
   arrive and leave). In reverse, that means steps that move toward the structure, which a greedy potential
   never takes. Needs lookahead or search.
1. Bridges stall in 23% of episodes and towers in 3%: robots crowd into a trench end or cling to a wall
   (local minima of the potential). Backtracking fixes most; the rest replay fine but do not start naturally.
   In a loader, drop them and draw a new goal.
1. Letters are short (4.8 steps). On a flat floor robots roll apart in a few joint steps. The natural-start
   thresholds (mean distance 4) set the length. Raising them gives longer, more scattered starts.
1. Orientation of letters is in grid rows (y) and columns (x), not checked against the camera, so an "L" may
   read mirrored from some views.
1. Reversed data has no failures or recoveries: every step is progress and every move happens. The policy never
   sees a move that did not happen, a fall, or a detour.
1. Heavy objects (friction 6, chain press) and pushes up and over are not proposed. Only light objects are
   pushed straight along one axis. A turn in the push path (pushers walking round) needs goal-directed reverse
   moves, as with scaffolding.

## Next

1. Goal generator at scale: compose goals (several structures, objects plus structures, terrain from
   `draw_terrain`), random shapes from voxel art or text-to-shape, and wordings with Haiku paraphrases (as the
   3D wording todo). The verifier makes any goal safe to try: a goal that cannot be taken apart just stalls.
1. Search instead of greed: beam or best-first search on (stage, potential) with lookahead of 2-3 steps, so
   scaffolding, helpers and push turns appear. Cost grows with beam width, and there is headroom.
1. Online loader: a worker process per core runs `make_goal` -> `take_apart` -> keep natural -> yield (frame,
   codes, text) examples, with `lm.world_at`-compatible worlds for `think_data` rendering. `ThinkData` still
   wants "instructions"/"subgoals" per episode: give it a reverse source with one instruction at t=0 and the
   goal as the only subgoal.
1. Failures and recoveries: mix in forward noise. Replay the reversed plan forward under the outcome model
   (`mujoco_hinge_v1`), and when a move misses, replan from the new state by walking back from the same goal
   and splicing. Or keep reversed data as one source beside the expert stores, which do have failures.

Rendered: 10 episodes per family on the session browse page
(`/private/tmp/claude-501/-Users-hrishik-beebot/64934076-03d9-47b9-8c56-3a54bf73e29c/scratchpad/browse/index.html`,
section "Reverse demonstrations"; GIFs `reverse-<family>-<i>.gif`, cards `cards-reverse.json`), made by
`scripts/reverse_demos.py browse --per 10 --seed 7`.
