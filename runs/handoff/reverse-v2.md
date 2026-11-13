# reverse-v2: long take-apart, scaffolding, guided walks

Written 2026-11-13. Upgrade of `runs/handoff/reverse-demos.md` (commit 8eff668). Still not read by training.

Code: `src/beebot/vla/voxel/reverse.py`, `scripts/reverse_demos.py` (check, browse), tests in
`tests/test_vla_voxel_reverse.py`.

## What changed

1. Verification is unchanged: a reverse step is kept only if `world.step(S, a, switch_cube)` gives exactly S',
   `refuse` passes every move and S is at rest. Every episode below replays forward exactly.
1. Local pivot test. `sources()` runs `pivot_moves` on a 7^3 crop round each robot instead of the 16^3 grid
   with that robot removed. It gives the same answer as the full grid (a test checks this on object scenes).
1. Extraction instead of greedy take-apart. Structures are taken apart top down. The highest robots that must
   move and have a legal move are tried first. Each walks out by breadth-first search over single-robot reverse
   moves (`_reverse_path`; the rest of the scene stays still; every cell it stops in touches something held by
   a face). It stops at a drop cell: on the ground, at least 3 from the goal's cells and touching no robot. If
   no such cell is reachable, any ground cell off the goal will do. The walk counts only if every step
   verifies. Played forward, a robot comes from the floor, climbs in and places itself, so each step brings it
   closer to its cell. Up to 4 walks are planned one after another. `_pipeline` then overlaps them: each starts
   at the earliest of two offsets after the previous one's start at which the affected steps verify, or else
   after all earlier walks have finished. The sequential plan is valid as it stands, so the fallback needs no
   check.
1. Natural start, measured. `start_stats` reduces a scene to six numbers:
   1. share of robots not resting on the floor or terrain
   1. face-to-face pairs per robot
   1. highest robot above the terrain
   1. other robots within two columns, per robot
   1. share of the goal's cells holding a robot
   1. mean column distance of the robots from the goal's centre

   `natural_bands` draws 32 scenes with the same terrain, robot count and objects, placing robots the way
   `sample_scenes` / `sample_object_scenes` do. Each statistic must lie at or below the 97.5 % quantile of
   those scenes, and spread at or above the 2.5 % quantile. Calibration: natural scenes pass their own bands
   in natural_reference_pass of cases (table below), so "natural" means "inside the bulk of natural scenes"
   and is not stricter than chance. After extraction, the joint greedy walk spreads robots until all six
   statistics are in band. It climbs a potential:
   1. off the goal's cells: +3
   1. geodesic cube-move distance to the ground: -1 per move (over cells with support below or beside, never
      only above)
   1. counted only on the ground: +0.5 x distance from the goal (cap 3) and +0.15 x distance from its centre
   1. -0.5 per face pair and -0.1 per pair within two columns

   It allows up to 4 level steps, backtracks dead ends, and joint proposals keep their robots 3 cells apart.
1. Scaffolding (`_scaffold`). Run when no robot that must move has a way out. It looks for a pivot into a
   stuck robot's cell whose swing cells are free and which lacks 1-3 support cells. If the helpers would hang
   in the air, it adds one more beside them. Helpers arrive in support order: other robots (not on the goal's
   cells) walk there by the same breadth-first search, verified. Then the stuck robot leaves over them.
   Forward: the helpers walk in, the last robot closes the gap, and the helpers leave. Only spans over open air
   need this: a robot between two solids along an axis has no pivot without support beside it. Arches open
   from a pillar top, tables and chairs from their edges, an L from its tip.
1. Guided walks (`Goal.start = A`). They use the same extraction, with A's empty cells as destinations and
   robots off A as the movers. Robots already on A's cells never move. The walk ends when the state is exactly
   A (`end == "reached"`, `Episode.distance` = A's cells without a robot). Families:
   1. `edit`: lower or raise the arm, tower one taller or shorter, arm one longer or shorter.
   1. `retarget`: from a letter, table, L or chair (A), k = 3-6 of its robots build a tower beside it. Spares
      go first, then shape robots top down, each removal keeping the rest at rest. Everything else stays where
      it is.
1. Unused robots (`Goal.unused`). `parked` puts them in a row along a named grid edge, and the instruction
   says so. `free` scatters them, or leaves them where they stood in A for guided goals.
1. New natural families:
   1. `span`: a one-wide deck over 1-3 cells of air, flush with both banks, with 3-5 robots nearby.
   1. `arch`: flush lintel between the pillar tops, or resting on top of them.
   1. `table`: 2-4 x 2-4 top on corner legs 1-3 high.
   1. `chair`: table plus a back 1-3 high.
   1. `cantilever`: L or T, column 2-5, arm 2-4.
   1. `big`: pyramid, stairs, room with a door, solid block or big table, with spares to 60-115 robots.
1. `Episode` adds `kinds` per forward step (extract / scaffold / progress / level / push), `stats`,
   `distance` and `touched` (robots that ever move, tracked by identity through moves). `stats` from
   `take_apart` adds `extract_calls`, `extract_ok` and `scaffolds` (attempts).

## Numbers

Seed 0, one thread per process, four processes at nice 10, on a Mac with load average 85-100 from other
sessions' jobs (12 cores). Rates are therefore per CPU second of the process (`time.process_time`), not wall
time. Wall-clock rates in the JSON are about 4x lower and say more about the load than about the code. Files:
`runs/reverse_demos/check-v2-{big,table,span,tower}.json` (not committed).

1. Start as asked: natural mode, all six statistics in band; guided, exactly A.
1. Natural scenes in band: fresh natural scenes tested against the same bands, 8 per goal. A real natural
   scene passes 83-91 % of the time, so the test does not demand more than natural scenes give.
1. Greedy accepted / proposed covers the joint spreading walk on the floor. Extractions found / tried counts
   extraction calls that found a walk. A failure moves on to the next drop tier, scaffolding or the next robot.

| family | episodes | robots mean / max | replay pass | start as asked | steps mean / max | robots moving per step | greedy accepted / proposed | extractions found / tried | robots touched | distance to A (mean, exact) | natural scenes in band | steps per CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tower | 40 | 6.4 / 10 | 1.00 | 1.00 | 12.9 / 28 | 1.96 | 0.17 | 1.00 | 5.5 | - | 0.91 | 27.2 |
| bridge | 40 | 7.2 / 12 | 1.00 | 1.00 | 12.8 / 29 | 2.15 | 0.17 | 1.00 | 6.8 | - | 0.88 | 32.9 |
| letter | 40 | 6.9 / 10 | 1.00 | 1.00 | 11.5 / 19 | 2.07 | 0.17 | 1.00 | 6.4 | - | 0.86 | 43.5 |
| push | 40 | 3.6 / 6 | 1.00 | 1.00 | 10.6 / 17 | 2.02 | 0.63 | - | 2.7 | - | - | 123.9 |
| span | 20 | 8.5 / 11 | 1.00 | 1.00 | 28.6 / 105 | 1.77 | 0.16 | 0.52 | 8.1 | - | 0.88 | 19.2 |
| arch | 20 | 13.2 / 16 | 1.00 | 1.00 | 32.2 / 48 | 2.19 | 0.17 | 1.00 | 12.1 | - | 0.88 | 31.3 |
| table | 20 | 19.4 / 29 | 1.00 | 1.00 | 59.7 / 111 | 2.12 | 0.17 | 1.00 | 18.4 | - | 0.85 | 17.7 |
| chair | 20 | 18.9 / 29 | 1.00 | 1.00 | 51.5 / 92 | 2.43 | 0.17 | 1.00 | 18.1 | - | 0.83 | 22.4 |
| cantilever | 20 | 8.8 / 14 | 1.00 | 1.00 | 34.0 / 68 | 1.72 | 0.17 | 1.00 | 8.5 | - | 0.91 | 37.7 |
| big | 6 | 79.5 / 102 | 1.00 | 1.00 | 235.2 / 299 | 2.24 | 0.17 | 1.00 | 49.2 | - | 0.85 | 4.7 |
| edit | 20 | 8.2 / 12 | 1.00 | 1.00 | 8.4 / 21 | 1.12 | - | 1.00 | 1.7 | 0.00, 1.00 | - | 46.7 |
| retarget | 20 | 14.2 / 27 | 1.00 | 0.95 | 37.6 / 92 | 1.41 | - | 0.81 | 4.7 | 0.10, 0.95 | - | 34.1 |

Step kinds (share of forward steps):
1. extract 0.86-0.99 for the structure and letter families; the rest is the joint spreading walk.
1. span: extract 0.38, scaffold 0.33, progress 0.27.
1. push: 0.80 push phase.
1. edit: 1.00 extract; retarget: 0.95 extract, 0.05 scaffold.

Level steps are at most 0.02 anywhere.

1. Every episode in every family replays exactly: 306 episodes.
1. Every forward step is a robot approaching its cell along a shortest walk (extract), a push, or a step of
   the spreading walk that lowers the disorder potential (progress), except span scaffold steps. In those,
   helpers walk in and later leave.
1. Spans: all 20 closed. Each needed one to four scaffold attempts. The deck robots have no legal move at the
   goal (checked in a test), so every span demo has helpers.
1. Big goals (69-102 robots, pyramid / block / room / big table): all 6 natural, 235 steps on average, 2.2
   robots moving per step. Before stacking drops were added (a robot taken out may stand on a lone grounded
   robot, as natural scenes stack), 3 of 6 stalled: the floor was too full for tables, and a 102-robot block
   had too many face pairs.
1. Throughput, steps (examples) per CPU second:
   1. small families 27-124
   1. tables and chairs 18-22
   1. spans 19
   1. big goals 4.7

   The Mac trainer (0.7 examples/s) needs a fraction of one core. Kaggle at 10-20/s needs one core for a mix
   of small families and about four cores for big goals alone. Cost on big goals: the breadth-first walks
   (`pivot_moves` on crops, about half) and verification, where `world.overhangs` runs `supported()` per
   critical robot (the other half).


## What still fails / is weak

1. retarget 1 of 20 stalled one cell short of A (distance 1). Refilling A's cells from the floor in the order
   found leaves a cell that no walk reaches. Scaffolding was tried (8 attempts in the family) and did not
   close it. Episodes that do not reach A keep `end == "stalled"` and `distance > 0`: drop them in a loader.
1. Extraction is greedy in its order (top layer first, a few robots tried). Spans find a walk on only 0.52 of
   extraction calls, retargets on 0.81; the rest of the progress comes from scaffolding or the next try.
1. Parallelism: 1.1-2.4 robots move per step. Walks overlap at most 4 at a time, at offsets of 1 or half the
   previous walk. Natural starts are reached, but the forward demo is a procession of robots arriving, not a
   swarm moving at once.
1. Scaffolding covers missing supports of 1-3 cells (plus one cell to connect them) with helpers that walk
   there one at a time. It has only been exercised on spans (and a few guided cases). Arches turned out never
   to need it: a flush lintel opens from a pillar top.
1. Natural mode has only one notion of natural: `sample_scenes` density on the goal's terrain. It never makes
   starts with half-built structures or other shapes standing, which real instructions will have. Guided mode
   covers those when A is given.
1. Goal text for the new families is template wording, not Haiku paraphrases. Letter orientation is
   unchanged from v1 (not checked against the camera).
1. Push family: unchanged phase machine (light objects, straight pushes only).
1. Tests: tests/test_vla_voxel_reverse.py, 27 passed on the final code in 107 s (12 minutes at load 95), most of
   it in the big-goal fixture.

## Next

1. Run it on the goal library (scripts/goals_reverse.py already calls `take_apart` with the same Goal API).
   The family is "library", natural mode, so it gets extraction and scaffolding as they are.
1. Throughput on big goals is dominated by `world.overhangs` inside verification. It loops in Python over
   every movable robot whose removal unsupports another, and runs `supported()` once per such robot. A
   vectorised overhang check in world.py would roughly double big-goal speed. It is not done here because
   world.py is shared.
1. Parallelism: extraction overlaps at most 4 walks. Planning walks jointly (time-expanded search) would give
   swarm-like starts with many robots moving at once.
1. Guided walks toward A only reach A when every empty A cell can be filled from the floor in some order.
   Edits that need scaffolding toward A use the same `_scaffold`, which is untested on them.
