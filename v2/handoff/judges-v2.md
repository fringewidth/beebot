# judges-v2: strict pass checks for the voxel suite

Written 2026-11-15 (task data-audit). The user browsed v2 episodes: "the data is really subpar. more
unambiguous tasks. MORE diversity. MORE trick questions. also we are assuming the user knows about
cheese and stripes." Rule: a pass must be obvious on the grey video, as the words say.

## What a pass needs now

Every checker returns 1 when the job is done and otherwise its old partial credit capped at
`suite.UNDONE` = 0.45 (`suite.strict`), under every pass mark, so evaluations keep a graded score
and only a done job passes. README "Strict judges" has the same table.

| Job | Before (passed at 0.75, formation 0.5) | Now (done) |
|---|---|---|
| tower (construction, parallel, selector, conditional, count) | tallest vertical run of robots anywhere >= 0.75 h (`maker_problem` short_tower then wanted h): a pile, a staircase or a column with robots hanging on it passed | `suite.towers`: one to four columns in a 2 x 2 footprint, each exactly h robots on the ground with nothing on top, no robot above the ground layer in the cells around it |
| formation mask | F1 >= 0.5: a 32-voxel "NI" with 12 of its voxels passed | every mask voxel taken and no other robot touching the shape |
| fill | 45% of the robots inside the trench: 2 of 3 robots in a trench passed | every trench voxel below the plateau top taken (no swarm of <= 115 robots can, so fill drops out) |
| bridge | longest robot run touching solid at both ends over 2 air cells | the same, in the plateaus' top layer, robots below not counted as air |
| parallel | mean of the two job scores >= 0.75, region = any 26-group with its centre in the box, tower = tallest run of the whole swarm: one tower inside the region passed both | at most one visible group per job (`suite.visible_groups`: groups at least two empty cells apart); each region, tower and surround job done by a group of its own; a region group all at its region (`suite.at_region`: box grown by a cell, centre in the box); the tower outside every region asked; eat jobs eaten |
| surround (parallel, things, conditionals) | shell share >= 0.25 (0.5 of the old full mark) | shell share >= 0.5 and robots on all four sides (`suite.surrounded`, `sides_held`) |
| goto, conditional goto, goto_zone | 75% of robots in the box grown by a cell | all of them (`at_region` for regions) |
| gather | largest 26-group >= 75% | one touching group |
| groups m | m largest groups hold 80%, m-th >= half an even share | exactly m visible groups, none under half an even share |
| spread | 75% with no robot face to face (checkerboard targets touched at edges) | no robot touches another, not even at an edge or corner (`suite.apart`); the expert's targets keep a cell between robots |
| line, approach, follow, perch, climb, line_zone, line_blocks | 75% | all of them |
| stay | 75% of robots on their start voxels | all of them, or every floor column holding the robots it held (a robot drawn floating that drops when gravity comes on has not moved) |
| move | centroid shift >= 2.25 cells | centroid shift within half a cell of the steps asked, under a cell sideways |
| selectors | job value x share of left-out robots still >= 0.75 | job done and every left-out robot still in place |
| eat and/all/ordered | share eaten >= 0.75 (3 cheeses: all) | every asked cheese, in order |
| herd region, push_thing | 0.8 x share of the way passed at 94% of the way | the object in place |
| build named object | IoU >= 0.75 (arch missing a leg passed) | IoU 1.0 (`build.OBJECT_PASS`, `suite.SUCCESS["build"]`) |

Expert changes that go with them: a tower segment takes only as many robots as its column has voxels
(`Expert.targets_of`, the objects' tower `cap` len(points) instead of +6), so the rest stay out of the
way instead of piling on; spread targets keep a cell between robots (`JobSetup.spread_points`).

The draw is unchanged: `fits`, `score_bound`, `coverage_bound`, the frozen `drawn_*` functions and
`objects.already_done` (now on the raw `job_value`) are as before. Checked: 444 attempt rows of the
demo draw and 404 of a full-store draw line up row for row (family, kind, robots) with the old code,
and the draw's CPU time per chunk is the same (new 31.5/3.0/6.0/2.3/1.6/1.4/3.1 s, old
30.7/3.3/6.3/2.1/2.7/2.8/2.8 s for the first seven chunks). Haiku labels joined by draw index still
line up. Consequence: some drawn tasks can never pass now (a letter larger than the swarm, every
fill); they cost play time and are dropped.

The draws that looked hung on 2026-11-14 (after_demo 3.7 h, after2_demo 8.5 h, pytest 6.5 h) were
the Mac asleep on battery with the lid closed (pmset: Maintenance Sleep, 15 min asleep per 20 s
awake), not a loop: the same 200-scene draw took 408 s awake on 2026-11-15, and its per-chunk CPU
time matches the old code. Still, as asked:

`suite.draw_guard` (DRAW_LIMIT = 20000 scene draws per task asked): every sampler loop (sample_tasks,
sample_coverage, sample_objects, sample_build, scenes) raises with a message instead of spinning
when nothing it draws fits. Normal draws come nowhere near it, so they are unchanged.

## Plain words

`demos --plain` (on in the demo preset; config["plain"]): the stored instruction text goes through
`suite.plain_words`: cheese -> "food block" ("food" for mass uses: "eat all the food", "leave the
food alone"), cheeses -> "food blocks", stripe -> "line on the floor", zone -> "patch of floor".
Applied when the store is written, never to Task.text, so draws and digests are unchanged. Every
template with these nouns (suite, coverage, the 2D logic pools, wordings.json; about 1250) was run
through it and read; the odd ones left are the source's own ("eat {a} and {b} food block"). Not
changed: the canonical "plain" sentence in records (wordings.parse_canonical inverts it) and the
Haiku records and scene summaries, which still say cheese.

## Measured

Demo store, seed 0, 200 scenes, `--demo --thin_towers --compact`, drawn with the old code and with
the new judges plus the new spread targets (no tower crew cap yet): same draw, rows lined up, 0
mismatches; the episodes are the same except spread ones.

| family | tried | kept, old | kept, new | old passes rejected |
|---|---|---|---|---|
| build | 200 | 200 | 187 | 13 (arch, cross, standing ring, number 1, gate, bridge, letter T) |
| construction (tower) | 39 | 29 | 1 | 28 |
| formation | 56 | 55 | 2 | 53 |
| logic (eat:absent) | 8 | 8 | 8 | 0 |
| motion | 37 | 37 | 35 | 2 (gather; spread now passes with the new targets) |
| parallel | 22 | 9 | 12 | 0 (3 region+region the old judge had dropped as done at the start now pass) |
| reasoning (conditional @else) | 10 | 9 | 5 | 4 (eat_tower@else: piles) |
| subset | 44 | 43 | 43 | 1 (tower@back; groups@left newly passes) |
| switch | 28 | 26 | 27 | 0 (spread>spread newly passes) |
| total | 444 | 416 | 320 | 101 |

Final draw (the committed code: judges, tower crew cap, spread targets, plain words), same seed and
options, 408 s with the Mac awake (logs/judges-after3-demo.out); rows lined up with the old draw, 0
mismatches:

| family | tried | kept, old | kept, final | old passes now dropped | newly kept |
|---|---|---|---|---|---|
| build | 200 | 200 | 187 | 13 | 0 |
| construction (tower) | 39 | 29 | 29 | 5 | 5 |
| formation | 56 | 55 | 2 | 53 | 0 |
| logic | 8 | 8 | 8 | 0 | 0 |
| motion | 37 | 37 | 35 | 2 | 0 |
| parallel | 22 | 9 | 12 | 0 | 3 |
| reasoning | 10 | 9 | 9 | 1 | 1 |
| subset | 44 | 43 | 43 | 1 | 1 |
| switch | 28 | 26 | 27 | 0 | 1 |
| total | 444 | 416 | 352 | 75 | 11 |

With the crew cap the towers come back (29 kept, clean single columns with the other cubes on the
floor away from them: "pile up into a tower of height 3", "make a column 8 robots high", "stack up 6
high" looked at). Formation stays at 2: letters and walls need every voxel and 8 to 12 cubes cover
few masks whole. No stored instruction says cheese; 14 say "food block" (the absent-colour tricks
and the conditionals).

Full-store options (v2: `--cube --rule switch_cube --outcomes mujoco_hinge_v1 --robots 6 115
--counts log --fit --coverage 0.3 --objects 1.0`), seed 3, 96 scenes, old episodes re-judged:
255 kept -> 147; 108 old passes rejected: fill 15 of 15, tower 20 of 21, formation 25 of 25,
parallel 19 of 23 (every pair kind), count:tower 4 of 4, conditionals 6 of 7, goto_zone 4 of 4,
surround_thing 3 of 6, line_zone 2 of 9, motion goto 2 of 3, follow 2 of 5, switch 4, subset 2.
Unchanged: herd 29, runner 13, logic 30, blocks 15, things eat/push.

## What the episodes looked like

Looked at start and end frames (scripts in the session scratchpad: render_scene of the first and
last frame with the instruction).

1. Rejected, rightly: "make a column 6 robots high" ended as an F (column with arms); "pile up into
   a tower of height 3/4" ended as 3 x 2 piles; "column 8 high" had two cubes hanging on its side;
   "draw a big NI" was 12 scattered cubes; "fill the trench" (full store) had 2 robots in a trench;
   "go to the back left, build a tower 3 high" (full store) had no tower at all; region+region and
   region+surround in the full store were robots scattered over both plateaus.
1. Kept, rightly: a clean 6-high column with the other robots two cells away; "form a wall" (4 x 3
   wall, complete); "half go left, half build a tower 4 high" (separate group and column); "break
   up into three groups" (three visibly separate piles); spread with no two cubes touching.
1. Borderline, rejected: "everyone to the center" with about 100 robots in two layers spilling past
   the centre box grown by a cell; "everyone get together" where one cube sits one cell off the
   group (one touching group is the rule). Strict on purpose.

## Still loose or open

1. Eating is not visible: a cheese vanishes in one frame (world rule). The judge is right but the
   video does not show "eat". Needs a world change (e.g. the cheese shrinks over a few steps, or
   the robots hold the surround for a few frames before it goes); not done here.
1. Colour is invisible in the grey sim: every task that names a cheese, block or zone by colour
   (logic, runner, herd, things, zones, most reasoning) cannot read as its text there. Dropping them
   is a distribution decision for the user; nothing dropped here. The demo store (empty floor) has
   none of these except the trick ones that ask for a missing thing.
1. Gravity is switched on after the scene is drawn (draws with a rule): robots drawn floating fall
   in the first steps. "stay" allows that (column counts), but every selector's "the rest stay" is
   still judged voxel by voxel, so those fail in such scenes (motion/subset expert pass rate in the
   coverage tests fell to 0.44-0.5). Fix: draw with gravity on, or judge from the settled frame.
1. Bounds are not strict: `fits` still uses F1 >= 0.5 for masks, 0.75 for towers and nothing for
   fill, so draws include tasks that can never pass. Making the bounds strict changes the draw
   (Haiku joins by draw index): do it when stores are next redrawn from scratch.
1. Selector towers (`tower@...`) still send the whole group to the column (JobSetup), and the free-
   cube planner leaves its climbing steps next to the tower, so both rarely pass now.
1. Region margin: `at_region` allows a cell beyond the box (the experts target box +- 1), so a group
   on a boundary can read as two regions; its centre must be in the box.

## Trick questions and diversity

Already there: `logic` "absent" (eat a colour not in the scene: do nothing) and "none" (don't eat
anything), "stay", switches ending in "stay" (`demos.NOTHING_TO_DO`), and conditionals on the else
branch ("eat the blue cheese if there is one, otherwise build a tower"). In the demo store they are
33 of 416 kept episodes nothing-to-do and 9 conditional-else; full store 31 of 255 and 6.

Proposed (none built: each needs a maker, not only a judge):

1. Already done: "go to the left" when everyone is on the left; right answer stay. Today such draws
   are dropped (done_at_start). Keep them as nothing-to-do with the stay judge.
1. Names nothing in the scene: "the group on the left: spread out" with one group; "the tower: come
   down" with no tower. Right answer stay; the selector picks nobody (chosen() returns None today and
   the maker drops it).
1. Too big: "build a tower 12 high" with 8 cubes, "form two groups" with 2 cubes. Needs a convention
   (stay, or the best it can) before data; propose stay and a spoken refusal at Haiku time.
1. Contradictory: "go to the left and go to the right", "stay still and spread out": stay.
1. Negations of jobs: "don't build a tower", "don't move to the back": stay.
1. Counting traps: "as many groups as there are towers" when there are none.

Diversity within the judges' reach: the demo store is 45% build (200 of 444 tries); the rest is
towers, walls, letters, groups, lines, moves. More shapes the swarm can make complete (small
letters, short walls, L and T floor shapes) would add variety without touching the draw rule.

## Files

`v2/src/beebot/vla/voxel/suite.py` (strict judges, plain_words, draw_guard, expert tower cap and
spread targets), `objects.py` (job_value returns (value, done), already_done on the raw value,
tower caps), `build.py` (OBJECT_PASS 1.0, draw_guard), `demos.py` (--plain, demo preset),
`v2/src/beebot/vla/voxel/README.md` ("Strict judges"), tests: `test_vla_voxel_suite.py` (new strict-judge tests; the
construction expert threshold), `test_vla_voxel_coverage.py` (stay and selector scores are capped;
expert pass thresholds 0.4 for motion and subset), `test_vla_voxel_objects.py` (thresholds zones
0.25, reasoning 0.4), `test_vla_voxel_free_cube.py` (the planner test checks the column height, not
the strict tower), `test_vla_voxel_coverage.py` also checks that a robot dropping in its column
still counts as staying (the scorecard test's do-nothing policy needs a stay pass there).
