# domain-randomization: physics and placement randomization for live reverse episodes

Written 2026-11-13. Opt-in. Nothing turns it on by default; training that does not pass `physics` gets today's
examples byte for byte (`tests/think_data_digest.py` fixture still passes).

Code:
1. `src/beebot/vla/voxel/randomize.py`: `Randomization` (strengths), `draw_physics`, `DrawnPhysics` (one episode's
   physics as a step function), `randomize_row` (execute a live episode under a draw, re-plan after failures),
   `replay_drawn` (validity check), `placement_of`.
1. `src/beebot/vla/voxel/views.py`: `Placement`, `placement(...)` context manager, `placement_faces`,
   `scene_views(..., jitter=)`; `Faces` gained optional `yaw`/`pivot` (None: today's axis-aligned path, unchanged).
1. `src/beebot/vla/voxel/reverse_live.py`: `generate(..., physics=)`, `Live(..., physics=)`; `LiveEpisode.world(t)`
   carries `stuck`; examples render with the episode's placement. `reverse.py` is untouched.
1. `scripts/randomize_measure.py` (drop rate and cost), `scripts/randomize_browse.py` (GIFs for the browse page),
   `tests/test_randomize.py` (20 tests).

## Switch on

    think_train --data_kwargs '{..., "live": {"share": 0.2, "workers": 4, "physics": {"strength": 0.3}}}'

`physics` is a dict of `Randomization` fields; `{}` takes the defaults. Per item: `failure`, `weak`, `broken`, `drop`,
`late`, `harm`, `placement` (each None: `strength`; 0: off). For the transfer measurement:

    draw = randomize.draw_physics(randomize.Randomization(strength=1.0), rng, world, plan_codes, goal_cells)
    physics = randomize.DrawnPhysics(draw, world)
    world = physics.start(world)                       # broken cubes become world.stuck
    world, realized = physics.step(world, codes, t)    # world.step under the draw; t = 0, 1, ... in order

## What is randomized

Strength s of a physical item is the share of episodes that have it (default `strength` 0.3: most draws are nominal;
a nominal episode has no failures at all, as the reverse episodes always had). Magnitudes are fixed:

| item | per episode | range |
|---|---|---|
| failure | multiplier on the MuJoCo table's failure rates (`OutcomeModel` `mujoco_hinge_v1`, per pivot class) | 2^U(-1, 1), 0.5x-2x |
| harm | of the episodes with failures, the share where the table's harmful outcomes (stuck, knocked cubes: stored MuJoCo aftermaths) happen too; else only harmless stays | probability `harm`, default 0.5 |
| weak | 1-4 cubes fail U(2, 4) times as often (follows the cube as it moves); switches the table on at x1 if failure is not drawn | |
| broken | 1 cube (2 with probability 0.3) stops responding: `world.stuck` from the start, commands are no-ops; it looks like any robot | probability 0.5 s |
| drop | a commanded move is not executed this step | 0.04 per moving cube |
| late | a commanded move is executed one step later; the expert waits one step with everyone staying | 0.03 per moving cube |
| placement | rendering only, always drawn, size = its strength (default 0.5): robots off their cells in x, y and turned about the vertical, a fixed draw per (episode, cell) with 30 % slow drift in the step | 0.1 s cells, 5 s degrees |

Why these ranges: HARDWARE.md's MuJoCo variation is friction 0.3-0.7, magnet strength +-25 % per cube, placement error
up to 0.03 L. Those are inputs of the table's measurement, not failure rates; the real cube's rates are unknown, so
the failure and weak ranges are a prior wider than that (a factor 2 either way, a few cubes 2-4x worse). Placement
at strength 1 is above 0.03 L on purpose (images that look like real cubes); at 0.3 it is 0.03 cell and 1.5 degrees.
Only yaw is drawn for tilt: the renderer's faces are axis aligned and yaw is the one rotation it takes without a
change of the splatter; roll and pitch are not drawn.

## How episodes stay valid

Reversal (`reverse.take_apart`) plans and verifies under nominal rules. `randomize_row` leaves that alone and
executes the recorded plan forward from its start under one physics draw. Three things can differ afterwards:

1. A command that did not happen (drop, a stayed outcome of the table, a broken cube): the cubes whose move did not
   happen make it again next step (a "redo", the plan then goes on unchanged) if the state is exactly the plan's frame
   with those cubes back on their cells; the harmless stays and drops are all handled this way, at no planning cost.
1. A late command: a wait step (every robot stays), then the comparison as above.
1. Anything else (a harmful outcome: a cube knocked off or stuck; a robot that a stuck cube blocks; a redo that failed
   3 times in a row): the expert plans again from the actual state. First a guided walk (`take_apart` with the actual
   state as the start A, as `reverse_live` does) to a later state of the original plan (3 or 6 steps ahead, if every
   stuck cube is still on its cell and not moved by the rest of the plan), then the plan goes on from there; failing
   that a guided walk to the goal. Cubes marked stuck cannot move in the walk. Verified step by step as usual.

The episode is dropped (`randomize_row` returns None, `generate` draws again) if it does not reach the goal in the
step cap, in 12 plans, if a re-plan runs out of its budget (200 walk steps, 10 s), or if no plan exists (a stuck cube on
a cell the goal does not hold, a failed walk). Push episodes (objects move) get placement only.

Valid means (`replay_drawn`, run on every kept episode in the tests and in the measurement):
1. Every recorded frame t + 1 equals `DrawnPhysics.step(frame t, codes of step t, t)`. The step is a pure function of
   the draw, the state and t: drops, late commands and outcome noise come from a generator seeded by (draw seed, t), so
   the replay reproduces the realized states, stuck cells and objects exactly.
1. No recorded command is one the pivot rule (`switch_cube`) forbids on the state it is given in (a broken cube's
   command is allowed: the cube does not know).
1. The last frame is the goal (`row["goal_mass"]`).
The recorded codes are the commands, the frames the realized states. `meta` gets `physics` (the draw), `placement`,
`replans`, `failed_steps`, `repeats`; `row["stuck"]` holds the stuck cells per frame (not in the observation: see
`stuck-visible`).

The policy sees the current frame only (`LiveEpisode.example` builds one observation and its images; earlier steps
reach it only as its own thoughts in the text). A failed move therefore shows as the state the next example starts
from and a redo as the same move commanded again; it never sees the step before.

## Numbers (`runs/randomize/measure.json`, `scripts/randomize_measure.py`)

12 nominal episodes (all 15 families mixed by seed, 6-40 robots), 2 physics draws each, one item alone at the
strength, 24 draws a row, CPU only, machine heavily loaded. "affected": kept episodes whose draw had a physical effect
(stuck frames recorded).

| item @ strength | dropped | causes | affected | redos | re-plans (rejoin / to goal / failed) | re-plan CPU s median / p95 | randomize CPU s per episode | steps vs nominal |
|---|---|---|---|---|---|---|---|---|
| failure @ 0.3 | 4 % | stuck cell not in goal 1 | 4 | 15.0 | 1 (0 / 0 / 1) | 0.00 / 0.00 | 0.20 | x1.23 |
| failure @ 1.0 | 17 % | stuck 3, cpu budget 1 | 20 | 14.4 | 10 (1 / 5 / 4) | 0.52 / 2.96 | 1.27 | x1.17 |
| weak @ 0.3 | 8 % | stuck 2 | 4 | 10.8 | 6 (2 / 2 / 2) | 0.12 / 1.44 | 0.46 | x1.15 |
| weak @ 1.0 | 25 % | stuck 4, walk failed 2 | 18 | 11.6 | 20 (8 / 6 / 6) | 0.11 / 6.1 | 1.81 | x1.24 |
| broken @ 0.3 | 0 % | | 1 | 4.0 | 1 (0 / 1 / 0) | 1.18 / 1.18 | 0.08 | x1.75 |
| broken @ 1.0 | 0 % | | 4 | 2.0 | 2 (0 / 2 / 0) | 1.35 / 1.42 | 0.19 | x0.82 |
| drop @ 0.3 | 0 % | | 5 | 5.4 | 0 | | 0.08 | x1.13 |
| drop @ 1.0 | 0 % | | 24 | 8.0 | 0 | | 0.47 | x1.06 |
| late @ 0.3 | 0 % | | 8 | 0 | 2 (0 / 2 / 0) | 0.01 / 0.01 | 0.15 | x0.97 |
| late @ 1.0 | 0 % | | 24 | 0 | 4 (0 / 4 / 0) | 0.01 / 0.02 | 0.43 | x1.00 |
| failure harmless only @ 1.0 | 0 % | | 24 | 11.5 | 4 (2 / 2 / 0) | 0.24 / 0.69 | 0.77 | x1.12 |
| failure harmful only @ 1.0 | 25 % | stuck 5, walk failed 1 | 18 | 10.4 | 12 (2 / 4 / 6) | 0.04 / 2.18 | 0.87 | x1.12 |
| all items @ 0.3 (the default) | 4 % | stuck 1 | 17 | 10.6 | 13 (1 / 11 / 1) | 0.38 / 1.53 | 0.98 | x1.35 |
| all items @ 1.0 | 29 % | too many re-plans 5, stuck 1, walk failed 1 | 17 | 17.1 | 97 (62 / 33 / 2) | 0.04 / 1.27 | 2.53 | x1.92 |

Reading it:
1. The main drop cause is "a stuck cube on a cell the goal does not hold": a harmful outcome strands a cube where the
   goal has none, and no plan can fix that. It is inherent to harmful outcomes, scales with how often they happen,
   and is cheap (found before any walk). Harmless stays and drops (the common failure) are never dropped.
1. Re-plan CPU: median 0.0-1.4 s, 95th percentile 1.3-6 s over episodes of up to 40 robots; the 10 s budget cut off 1
   failure and 2 weak draws. Above 40 robots was not measured (the manager's note on v3's walks: the budget is what
   bounds it). Added CPU per live episode: 0.1-0.5 s at strength 0.3 for one item, about 1 s for all items at the
   default, 2.5 s at strength 1; generating a nominal episode costs about 1.9 s.
1. Broken cubes are drawn only from robots that start on a cell the goal holds (a cube that never leaves its cell
   has to be part of the goal's end state), which is rare for most families (some `move`, `spread` starts); with
   so few, treat the table's broken rows as small samples (1 and 4 episodes). The broken cube is stuck from step 0
   and the expert plans around it once the move fails.
1. `weak` alone was measured with harmful outcomes on (the default of a draw without `failure`); with `failure` the
   `harm` share applies.

Replay passed on every kept episode in the measurement (`replay_pass` 1.0), and `tests/test_randomize.py` checks it.

## Render check

GIFs on the browse page ("Domain randomization", `dr-<kind>-<k>.gif`, 3 each for failure, weak, broken, drop, late,
placement; grey render, caption under the picture; the step line says how many commanded moves did not happen; drop
and late rates are raised to 12 % in the pictures only). Placement shows the perfect grid left and the jittered robots
right in the top camera.

## Not done

1. Roll and pitch for placement.
1. A goal that adapts to a broken cube (a broken cube off the goal drops the draw; only goal-cell robots are drawn).
1. The policy cannot see a stuck cube (ops node `stuck-visible`); `row["stuck"]` and `world.stuck` per frame are
   there for it.
1. Failures above 40 robots were not measured.
