# Voxel world

A 3D version of the swarm world (`vla/world.py`), with a scripted data generator, window facts
and an isometric renderer. Everything is batched torch on the CPU; nothing here touches the 2D
modules.

| file | what |
| --- | --- |
| `world.py` | rules, scene sampling, observations |
| `controller.py` | the attractor controller (target voxels to actions) |
| `scripted.py` | random task sequences and the data store |
| `suite.py` | instruction families, checkers, the expert that demonstrates them |
| `objects.py` | objects with properties, blocks, zones and reasoning over them: four more families, their checkers and expert (section "Objects") |
| `demos.py` | expert demonstrations as a store, also for pivoting cubes |
| `free_cube.py` | the planner for cubes that cannot leave a bond, and the comparison of rules and experts |
| `shapes.py` | named objects as parts programs (text form), the support and climbing rules, functions, the overlap score, a mesh loader |
| `build.py` | the build family: draws, records, the build expert, the score, the per-object measurement |
| `haiku.py` | open-vocabulary instructions and thought labels written by Haiku, verified by a second call, joined to demo stores |
| `rollout.py` | closed-loop episodes of the think-then-act policy (and of the expert) on the training conditions |
| `scorecard.py` | the 3D scorecard: every family and held-out split, controls, thought checks, the oracle-thought arm, gates (section "Think-then-act scorecard") |
| `facts.py` | window facts as hashable atoms |
| `render.py` | isometric PNG/GIF |
| `views.py` | camera views (oblique and top-down) with visual domain randomization for the vision policy (section "Views and randomization") |

Tests: `OMP_NUM_THREADS=4 uv run pytest -q tests/test_vla_voxel.py`.

## Rules

Axes are (z, y, x); z is height and z = 0 is the layer on the floor. Grids are D x H x W (16^3
by default, tested up to 32^3). Each voxel holds at most one robot.

1. **Actions.** There are 27 actions: the 26 neighbours (`DIRS`) plus STAY (code 26). Actions are
   stored as int8 codes on the grid. Voxels with no robot always hold STAY, so the action field
   carries no information about empty space.
1. **Moves.** Moves resolve in rounds, with random priority as in 2D:
   1. A robot moves only into a free voxel: one with no robot, wall or cheese.
   1. When several robots claim the same voxel, the highest score wins.
   1. A voxel vacated in one round can be claimed in the next. So a line of robots advances
      together, while two robots trying to swap both stay put.
1. **Cheese push.** Cheeses are 2 x 2 x 2 blocks.
   1. Robots that try to move into a cheese press on it; the press is the sum of their move
      directions.
   1. When the press along its strongest axis reaches `FRICTION` (2), the cheese slides one
      voxel that way.
   1. The slide only happens if the leading face (2 x 2) is clear of walls, other cheeses and
      the grid edge.
   1. Robots standing in the leading face are shoved one voxel further. If any of them cannot
      move, nothing moves.
1. **Eat.** A cheese's shell is the 56 voxels touching it by a face, edge or corner. The open
   shell is the part of the shell inside the grid and not walls. A cheese is eaten when robots
   fill at least `EAT_SHARE` (0.75) of its open shell and at least `EAT_MIN_SHARE` (0.4) of the
   full shell. A cheese on the floor has 40 open shell voxels, so it needs 30 robots.
1. **Gravity** (a per-scene switch, on in half the sampled scenes).
   1. A robot is supported when it is connected through face-adjacent robots to a robot that
      either sits on the floor or is face-adjacent to a wall or a cheese.
   1. Every step, unsupported robots fall one voxel. Stacks fall together.
   1. A cheese with nothing under its footprint falls one voxel per step.
   1. The rule is deliberately permissive, with no load or torque, so cantilevers and climbing
      on each other work. That is what makes towers and bridges buildable.
1. **Step order.** Each step runs in this order: move, push, fall, eat.
1. **Observation.** `observe(world)` returns (B, 10, D, H, W) with channels robot, wall, object
   mask, object RGB, food and zone RGB (section "Objects"); the first six are the observation as
   it was before objects (robot, wall, cheese mask, cheese RGB).
1. **Scenes.** `sample_scenes` builds terrain as a heightmap of walls, in one of three layouts:
   flat, low hills, or a gap (two plateaus with a trench between them).
   1. Robots rest on the terrain at a density of 0.2 to 0.45 per floor cell.
   1. Each scene has 0 to 3 cheeses, placed on level footprints.

## Scripted generator

Each scene runs its own sequence of segments. A segment lasts 8 to 24 steps at 16^3 and scales
with the grid size. Each segment draws one intent:

| intent | when | what the robots do |
| --- | --- | --- |
| goto | always | head for one random voxel (on the ground under gravity) |
| gather | always | head for the centroid |
| split | always | head for the nearer of two random voxels |
| push | a cheese is alive | crowd behind a cheese and press it along one axis (horizontal under gravity) |
| eat | a cheese is alive | fill the cheese's shell |
| tower | gravity on | fill a 2 x 2 column 3 to 8 voxels high on level ground |
| bridge | gravity on, gap layout | fill a 2-wide span across the trench, level with the plateau tops |
| stay | always | nobody moves |

Every intent becomes a set of target voxels. Each robot heads for the nearest target that no
robot holds yet, or for the nearest target once every target is held. It takes the action whose
destination is closest to that target, choosing only among destinations that are not walls or
cheeses. A robot already on a target stays. Pushers whose step lands on the pushed cheese take
that step. Ten percent of moves are jittered to a random action.

```python
from beebot.vla.voxel.scripted import generate
from beebot.vla.voxel.facts import store_facts
from beebot.vla.voxel.render import render_episode

store = generate(size=16, n_episodes=256, seed=0)  # batch_size=64; steps default 64 at 16^3, 96 at 24^3
facts = store_facts(store, e=0, t0=0, t1=24)
render_episode(store, 0, "episode.gif", every=2)
```

The store has these fields:

1. `mass`: (E, T + 1, D, H, W) uint8.
1. `actions`: (E, T, D, H, W) int8, codes 0..26.
1. `cheese_pos`: (E, T + 1, K, 3) int8, the lowest corner of each cheese.
1. `cheese_alive`: (E, T + 1, K).
1. `walls`: (E, D, H, W).
1. `cheese_color`: (E, K), with -1 for an empty slot.
1. `gravity`: (E,).
1. `phases`: the intent name for each step.

Generation is deterministic under a seed.

### Speed

These timings are CPU-only, with `OMP_NUM_THREADS=4`, a batch of 64 and 128 episodes. They were
measured while the machine was heavily loaded by other jobs (load average about 38), so they are
pessimistic and noisy.

| grid | robots per scene | steps per episode | seconds per 1000 episode-steps |
| --- | --- | --- | --- |
| 16^3 | about 84 | 64 | 0.8 to 2.9 |
| 24^3 | about 185 | 96 | 2.6 to 3.5 |

Most of the time goes to the gravity support flood, followed by the controller and the move
rounds. Moves and the controller work on lists of robots rather than the dense grid.

## Facts

`window_facts(static, start, end)` returns lines, types, data and atoms, as in 2D. The atoms are
hashable tuples, so the 2D facts score applies unchanged; `facts_score` is the Jaccard overlap.

| atom | when |
| --- | --- |
| `("move", direction)` | the centroid moved at least 2 voxels (at 16^3; scales with size) |
| `("end region", region)` | where the centroid ended, stated with move; one of 27 regions such as `"low front left"` or `"center"` |
| `("groups", n)` | the number of 26-connected groups (each at least 10% of the swarm) changed; 4 means 4 or more |
| `("pushed", colour, direction)` | a cheese moved; under gravity, falling does not count |
| `("ate", colour)` | a cheese was eaten |
| `("around", colour)` | robots came to hold at least half of a cheese's open shell |
| `("tower", bucket)` | the tallest vertical run of robots rose into a higher bucket: low (3+), medium (5+) or tall (8+) at 16^3; gravity scenes only |
| `("bridge",)` | a straight horizontal run of robots now has a solid voxel at each end and at least 2 robots over empty voxels; gravity scenes only |

Directions name each axis whose component is at least 0.4 of the largest one, vertical first.
The axis words are up and down, front and back (y; the front is +y, nearest the default viewer,
as in the regions), and left and right (x). Examples: `"up"`, `"front left"`, `"down back right"`.

## Expert demonstrations for pivoting cubes

`demos.py` plays the suite's training tasks (`suite.py`) with the expert and keeps the episodes
the checker passes. `--cube` turns on four options that make the data fit a policy for pivoting
cubes; each can also be set alone, and without them the generator behaves as before.

```
OMP_NUM_THREADS=4 uv run python -m beebot.vla.voxel.demos --out data/vla/voxel_demos_cube --scenes 6000 --cube
```

| option | `--cube` value | what |
| --- | --- | --- |
| `--moves cube` | on | only the 14 moves with no horizontal diagonal (`world.CUBE_MOVES`): one axis, or up/down with one horizontal axis. The controller picks among them and jitter draws among them. |
| `--steps` | 72 | episode length (48 otherwise). |
| `--robots LOW HIGH` | 6 115 | each scene's robot count, uniform over the range (otherwise a density of 0.2 to 0.45, 51 to 115 robots). Tasks are fitted to the swarm, see below. |
| `--failure` | 0.15 | the probability that a robot's move does not happen this step (it stays). `0.15,up_over=0.3` sets single move classes: `level`, `up_over`, `down_over`, `vertical` (`world.failure_rates`). |

1. **Labels.** The recorded action is the one the expert chose, whether or not the move
   happened. The expert plans again from the current state every step, so the recording shows
   the recovery. Which moves failed is not stored; the rates are, in `small.pt` under `config`.
1. **Tasks that fit the swarm.** With `--robots`, a task is drawn only if `suite.fits` says the
   swarm is large enough: `suite.score_bound`, the best checker score that many robots could
   reach, must reach the family's pass mark (0.5 for formations, 0.75 otherwise). The bound comes
   from the rules and the plan's target sets, not from tuning:
   1. Formation: F1 of n robots against the mask, so between a third of and three times the mask's voxels.
   1. Tower of height h: the plan fills a 2 x 2 column, so 3 h robots (heights 3 to 8 need 9 to 24).
   1. Bridge: three quarters of its two lanes (5 or 6 robots). Fill: any count.
   1. Eating a cheese: three quarters of its open shell and at least 23 robots, 30 on open floor;
      never for a cheese in a corner. A runner counts as 23, since it moves.
   1. Herd: two robots (the press that slides a cheese).
   1. Parallel: the best split of the swarm between the two jobs. A region job takes a tenth of
      the swarm, a surround job half the open shell (20 robots) for full marks.
   1. Eating nothing ("none", "absent"): any count.

   So swarms under 23 robots get herding, towers, bridges, trench filling, small formations,
   "eat nothing" and parallel pairs of region, tower and surround; they never get eating or
   runners. The bound is necessary, not sufficient: the kept/tried rates are still below 1.
1. **Sub-goals.** `small.pt` holds `subgoals`: for each kept episode a list of entries
   `{"t": step, "index": plan segment, "goal": {...}}`, one whenever the expert's sub-goal
   changes. The action at step `t` was chosen under the last entry with `entry["t"] <= t`
   (`suite.subgoal_at`). Goals:

   | goal | fields |
   | --- | --- |
   | `letters` | `word`, `standing` |
   | `shape` | `shape` |
   | `tower` | `height`, `at` (lowest corner of the column, z y x) |
   | `bridge`, `fill` | none |
   | `region` | `region` (only inside a union) |
   | `eat`, `surround` | `colour` |
   | `push` | `colour`, `direction` (dz, dy, dx), `to` (a floor region or `trench`); a change of direction within the segment is a new entry with the same index |
   | `union` | `parts`: the two jobs of a parallel task, each one of the above |
   | `stay` | the task has no plan (eat nothing); index -1 |
   | `done` | the last eat or push segment is finished and the robots idle; index = the plan's length |

   A plan moves to its next segment when the segment is done (the cheese is gone, the cheese is
   in its region) or after 24 steps.
1. **Other files.** `small.pt` also holds `robots` (swarm size per kept episode). `attempts.jsonl`
   has one row per episode played, kept or not: family, kind, robots, score, unasked, and the
   store row `e` (-1 when dropped). `labels.jsonl` and the store fields are unchanged, so
   `lm.batch_of` reads the store as before.

A 300-scene sample (seed 0, `--cube`) took 37 s on 4 threads, about 12 s per 100 scenes, and
kept 875 of 1098 episodes (0.5 GB; an episode is 0.6 MB). The store files are allocated for six
episodes per scene and are sparse, so their apparent size is about twice what they use.

Three more options, each off unless given (`--cube` does not turn them on):

| option | what |
| --- | --- |
| `--compact` | write `mass` and `actions` by robot instead of as dense grids (next section). |
| `--counts log` | draw the robot count log-uniformly over `--robots` (`suite.draw_count`), so every doubling of the swarm gets the same share of scenes. Of 6 to 115 robots, 31% of scenes have under 15, 34% have 15 to 40 and 35% more; the uniform draw gives 8%, 24% and 68%. |
| `--fit` | apply the swarm check without `--robots` too, on the density range's 51 to 115 robots. It leaves out what no expert can pass there: eating a cheese in a corner, and formations with over three robots per mask voxel (two-letter words, the ring). With `--robots` or `--cube` the check is always on. It changes which scenes are drawn, so it stays opt-in. |

The same sample with `--counts log --compact` (300 scenes, seed 0) kept 719 of 915 episodes in 216 s (small swarms are redrawn more often until a task fits; the uniform sample took 37 s):

| robots | kept / tried | construction | formation | herd | logic | parallel | runner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| under 15 | 110 / 139 | 34 / 39 | 1 / 3 | 41 / 57 | 22 / 22 | 12 / 18 | none |
| 15 to 40 | 233 / 305 | 68 / 71 | 52 / 55 | 32 / 55 | 29 / 37 | 34 / 44 | 18 / 43 |
| over 40 | 376 / 471 | 82 / 86 | 78 / 85 | 20 / 63 | 65 / 74 | 60 / 79 | 71 / 84 |

The uniform sample kept 31 / 42, 173 / 219 and 671 / 837 in the same buckets.

Small swarms never get runners (a cheese takes 23 robots to eat) and logic gives them only
"eat nothing".

Tests: `tests/test_vla_voxel_cube.py`, `tests/test_vla_voxel_compact.py`.

## Free cubes: the moves a real pivoting cube can make

`--cube` leaves out horizontal diagonals but still lets any robot step into any free neighbouring
voxel. The MuJoCo studies of the mechanism (HARDWARE.md, "Round 3" and "Ten cubes in one scene")
say a real cube cannot: it moves by pivoting on a support, and permanent magnets hold it whenever
a second cube touches one of its faces (0 of 121 attempts, at any flywheel speed). The rule is
opt-in; without it everything is as before.

```
OMP_NUM_THREADS=4 uv run python -m beebot.vla.voxel.demos --out data/vla/voxel_demos_free_cube --scenes 6000 --free-cube --counts log --compact
```

`--free-cube` is `--cube` with `--rule free_cube --expert planner`: the world refuses every move
the rule forbids, gravity is on in every scene, and the expert is the planner of `free_cube.py`.
`--rule switch_cube` is the same rule for cubes that can release their bonds (switchable or
electropermanent magnets). `--apart` sets every swarm down again before the episode so that no
two robots touch, as a swarm laid out by hand would start; it is off unless given. `small.pt`
records `rule`, `expert` and `apart` under `config`.

### The rule (`world.pivot_moves`, `world.refuse`)

For a cube at c, on the scene as the step begins:

| move | pivot | needs | must be empty |
| --- | --- | --- | --- |
| one axis h, level (4) | quarter turn: floor roll, traverse; wall crawl | a surface under c that continues under c + h; failing that, a surface beside c (at right angles to h) that continues beside c + h | c + h, and the two cells on the far side of c and c + h from the support |
| straight up or down (2) | quarter turn: concave up, concave down | a surface beside c that continues beside c + h | the same |
| up and sideways e (4) | half turn: convex up (from the floor or from hanging) | a support at c + e; the cube lands on top of it | c + up, c - e, c + up - e, c + up + e, c + 2 up, c + 2 up + e |
| down and sideways e (4) | half turn: convex down | a support under c; the cube lands beside it | c + e, c + up, c + e + up, c + e + down, c + 2 e, c + 2 e + down |

1. **Free.** The move happens only if no other cube touches a face of c, except the support
   itself (`free_cube`). So a cube with a second face neighbour stays, a cube on the floor beside
   one cube can only climb it, and a cube standing on a cube cannot climb a third. The floor,
   terrain and cheeses are not bonds. `switch_cube` drops this condition and keeps the rest.
1. **Supports.** The floor, terrain and cubes on any side but above; a cheese only from below.
1. **One at a time.** A cube does not pivot on a cube that is itself pivoting in the same step
   (its support, or the cube it comes to rest against).
1. **Presses.** A move into a cheese is a press, not a pivot, and is not restricted.
1. **Consequence.** Every pivot ends face to face with its support or the surface it continues
   on. A cube that has touched a group can only crawl over it; a group never splits; a group in
   which every cube has two face neighbours inside it (a 2 x 2 patch) never moves again. Only a
   cube whose every contact is floor or terrain travels freely.

What was measured and what is extrapolated:

| | basis |
| --- | --- |
| one extra face bond holds the cube | measured: 0 of 121 (77 with exactly one bond); floor peel, convex up and down, traverse, wall crawl. Concave moves with a bond were not tried and are assumed held too |
| free cubes move | measured: floor roll 34 of 34, concave up 12 of 14, concave down 5 of 5, convex up from hanging 21 of 33, convex up 0.77 to 0.91 with retries. Here an allowed move simply happens; `--failure` is the only noise, and a harmful failure (off the lattice, a knocked neighbour: 3 to 9% of convex ups, more in the ten-cube scene) is not modelled |
| the pivot table (support, swing) | `cube_swarm.plan`, which every measured move went through |
| terrain as a support and never a bond | extrapolated from the steel floor (grip 0.3); no terrain in MuJoCo |
| a cheese: rolled over, never climbed, never a bond; presses unrestricted | assumed; no cheese in MuJoCo. A cube held by a bond can still press |
| an edge contact does not hold | optimistic: hinged movers reached 16 of 23, but a floor roll away from an edge contact stayed 18 of 22 times at the calibrated speed |
| sliding into a slot beside a cube works | optimistic: 3 of 22 got in, 6 ended off the lattice. `pivot_moves(rubs=True)` refuses it; no named rule uses it, because it rules out every filled 2 x 2 footprint |
| one at a time | assumed; only single movers were simulated |
| what holds a structure up | unchanged: the voxel world's support rule |

`hardware/physics.py` (`allowed_moves`) is a different, looser rule for a geared crawler: it
charges torque for the bonds a move leaves and lets a strong design pry them. Every `free_cube`
move is allowed there; half of what it allows a permanent-magnet design `free_cube` refuses
(HARDWARE.md, "The rule in the voxel world").

### The planner (`free_cube.py`)

`FreeCubeExpert` runs the suite's plans and sub-goals with a planner in place of the attractor
controller. Lone cubes are its material: each step a breadth-first search from the open target
voxels through the rule's own moves gives every voxel its number of pivots to a target; lone
cubes descend it nearest first and never land beside another cube in transit; cubes in groups
crawl by the same distances afterwards. The route up a wall of cubes (roll to it, concave up to
hang, convex up from hanging) comes out of the search. No jitter, and no recorded action breaks
the rule, failures or not.

### What the expert can still do

```
OMP_NUM_THREADS=4 uv run python -m beebot.vla.voxel.free_cube --scenes 192 --steps 72 144 --out runs/free_cube/compare.jsonl
```

192 scenes drawn as `demos --cube --counts log` draws them (seed 0), 575 episodes, every arm from
the same scenes with the same seeds and `--failure 0.15`. Cells are mean checker score (0 when a
forbidden cheese was eaten) / share of episodes that pass as demonstrations. Rows in
`runs/free_cube/{attractor,planner,extra}.jsonl`.

| arm | expert | rule | start |
| --- | --- | --- | --- |
| cube | attractor | none (any free neighbouring voxel) | as drawn |
| switch | attractor | `switch_cube`: pivots, bonds released | as drawn, gravity on |
| free | attractor, choosing among allowed moves | `free_cube` | as drawn, gravity on |
| free + planner | `FreeCubeExpert` | `free_cube` | as drawn, gravity on |
| free + planner, apart | `FreeCubeExpert` | `free_cube` | the same swarm set down again with no two robots touching (`demos --apart`) |

72 steps:

| | n | cube | switch | free | free + planner | free + planner, apart |
| --- | --- | --- | --- | --- | --- | --- |
| all | 575 | 0.77 / 0.79 | 0.70 / 0.71 | 0.37 / 0.24 | 0.49 / 0.40 | 0.59 / 0.55 |
| formation | 91 | 0.67 / 0.91 | 0.61 / 0.80 | 0.27 / 0.08 | 0.40 / 0.30 | 0.48 / 0.58 |
| construction | 123 | 0.95 / 0.93 | 0.92 / 0.92 | 0.58 / 0.33 | 0.80 / 0.72 | 0.85 / 0.80 |
| logic | 83 | 0.87 / 0.84 | 0.75 / 0.73 | 0.58 / 0.58 | 0.58 / 0.58 | 0.77 / 0.77 |
| parallel | 86 | 0.83 / 0.77 | 0.80 / 0.77 | 0.61 / 0.36 | 0.68 / 0.44 | 0.69 / 0.51 |
| herd | 112 | 0.62 / 0.58 | 0.56 / 0.47 | 0.17 / 0.12 | 0.33 / 0.26 | 0.40 / 0.29 |
| runner | 80 | 0.69 / 0.69 | 0.53 / 0.53 | 0.00 / 0.00 | 0.01 / 0.01 | 0.33 / 0.33 |
| under 15 robots | 96 | 0.87 / 0.84 | 0.79 / 0.73 | 0.55 / 0.46 | 0.63 / 0.53 | 0.71 / 0.65 |
| 15 to 40 | 171 | 0.70 / 0.74 | 0.65 / 0.67 | 0.38 / 0.23 | 0.53 / 0.49 | 0.58 / 0.56 |
| over 40 | 308 | 0.79 / 0.80 | 0.71 / 0.72 | 0.31 / 0.18 | 0.42 / 0.31 | 0.57 / 0.52 |
| tower | 79 | 0.95 / 0.92 | 0.93 / 0.92 | 0.58 / 0.20 | 0.85 / 0.73 | 0.86 / 0.77 |
| bridge | 22 | 0.91 / 0.91 | 0.82 / 0.82 | 0.27 / 0.27 | 0.50 / 0.50 | 0.68 / 0.68 |
| fill | 22 | 1.00 / 1.00 | 1.00 / 1.00 | 0.87 / 0.82 | 0.92 / 0.86 | 0.99 / 1.00 |
| eat one cheese | 89 | 0.72 / 0.72 | 0.53 / 0.53 | 0.00 / 0.00 | 0.01 / 0.01 | 0.38 / 0.38 |
| herd to a region | 85 | 0.62 / 0.56 | 0.55 / 0.44 | 0.22 / 0.14 | 0.31 / 0.21 | 0.37 / 0.24 |
| herd into the trench | 27 | 0.63 / 0.63 | 0.59 / 0.59 | 0.04 / 0.04 | 0.41 / 0.41 | 0.48 / 0.48 |

144 steps, all tasks: cube 0.78 / 0.80, switch 0.76 / 0.77, free 0.38 / 0.24, free + planner
0.49 / 0.40, free + planner, apart 0.60 / 0.55.

1. **The rule halves what the attractor expert can do** (0.77 to 0.37; passes 0.79 to 0.24), most
   of all with many robots. At the start 83% of the robots of a swarm under 15 touch no other
   robot, 67% from 15 to 40 and 31% over 40; of a swarm over 40, 6% sit in a patch that can never
   move and the rest of the grouped ones can only crawl. The attractor sends everyone at once, so
   lone cubes meet on the way and are stuck together. Gravity alone costs nothing (cube with
   gravity on everywhere: 0.78 / 0.78).
1. **The planner wins back about 30% of the loss** (0.49 / 0.40) and time does not help it:
   by step 72 every cube that can still move has arrived. Lone cubes alone, with groups left
   where they are, give 0.45 / 0.35. What limits it is how many cubes can still move, not what
   they can reach.
1. **Where the swarm starts matters as much as the planner.** Set down with gaps, the same
   swarms reach 0.59 / 0.55, and a cheese is eaten in 38% of the tries instead of 1%. The
   attractor does not profit from the gaps (0.40 / 0.28): it closes them.
1. **Released bonds make almost all of it go away.** With the unchanged attractor expert,
   `switch_cube` keeps 0.70 / 0.71 of the cube world's 0.77 / 0.79 at 72 steps and 0.76 / 0.77 of
   0.78 / 0.80 at 144: the pivot geometry costs time, not tasks. The planner under `switch_cube`
   scores the same (0.71 / 0.70), so with switchable magnets the existing expert is enough.

What cubes that cannot leave a bond can do in principle:

| | |
| --- | --- |
| towers, walls, columns | yes, by the measured route: a lone cube rolls to the foot, hangs on the side, climbs; one cube at a time per side. Tower 0.85 against 0.95 |
| trench fill, bridge | yes: cubes step off the plateau edge and hang, and a bridge grows as cubes walk over it and step off its end. Bridge 0.50 against 0.91, 0.68 from a spaced start |
| gathering in a region, surrounding a cheese | yes, as far as lone cubes and crawling pairs get there |
| flat formations (letters on the floor) | yes when each voxel can be approached from outside, inside first; the score is capped by the robots that cannot leave their group and lie off the shape |
| eating a cheese | only by accretion: 30 cubes on open floor, lower ring first, each arriving free. Needs that many cubes still lone, so mostly a question of the start |
| anything in sequence with the same cubes (eat a, then b; a second shape) | no: the cubes around the first cheese are bonded in a ring |
| herding | by the rule, one voxel per fresh cube at best: two pushers side by side bond and cannot follow the cheese; one more cube has to slide into the slot for every voxel, and a pusher on top can walk along the row they leave behind |
| catching a runner | no closing in: every approach that fails leaves a frozen group. 0.01; 0.33 from a spaced start, when the first approach succeeds |
| floating shapes, standing letters | not as asked (gravity); only as far as a pile happens to cover them |
| anything that needs a cube out of a filled layer or a group split | no |

As demonstrations: the 300-scene sample of the section above (seed 0, `--counts log --compact`,
915 episodes played, of which `--cube` kept 719) keeps 379 with `--free-cube` (618 s on 2
threads, 5.3 MB, `data/vla/voxel_demos_free_cube_sample`) and 508 with `--free-cube --apart`
(7.4 MB, `data/vla/voxel_demos_free_cube_apart_sample`). Kept / tried:

| robots | `--free-cube` | construction | formation | herd | logic | parallel | runner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| under 15 | 73 / 139 | 28 / 39 | 1 / 3 | 13 / 57 | 22 / 22 | 9 / 18 | none |
| 15 to 40 | 147 / 305 | 57 / 71 | 26 / 55 | 16 / 55 | 19 / 37 | 29 / 44 | 0 / 43 |
| over 40 | 159 / 471 | 60 / 86 | 21 / 85 | 11 / 63 | 32 / 74 | 32 / 79 | 3 / 84 |

| robots | `--free-cube --apart` | construction | formation | herd | logic | parallel | runner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| under 15 | 84 / 139 | 31 / 39 | 2 / 3 | 16 / 57 | 22 / 22 | 13 / 18 | none |
| 15 to 40 | 177 / 305 | 55 / 71 | 41 / 55 | 16 / 55 | 25 / 37 | 34 / 44 | 6 / 43 |
| over 40 | 247 / 471 | 68 / 86 | 40 / 85 | 15 / 63 | 49 / 74 | 36 / 79 | 39 / 84 |

Without `--apart` all 73 kept logic episodes are "eat nothing" (0 of 60 eating tasks pass, 3 of
127 runners); with it 23 of 60 and 45 of 127. A policy trained on the first store alone sees no
eating. Tasks are still drawn for the
cube world (`suite.fits` knows nothing of the rule), so tasks no free cube can do are played
and dropped.

Tests: `tests/test_vla_voxel_free_cube.py`.

## Coverage

What a person can ask a swarm of pivoting cubes to do, as a grammar: **who** (a selector) x
**what** (a job) x **when** (modifiers). The suite's six families (`FAMILIES`) cover part of it;
three more families (`suite.COVERAGE`: `motion`, `subset`, `switch`) add the rest that an expert
can demonstrate. They are drawn by `suite.sample_coverage`, never by `sample_tasks`, so the
draws of the six families are unchanged (`tests/test_vla_voxel_coverage.py` pins their digests
for the default, `--cube` and `--counts log` draws).

### Who

| selector | spec (`spec["who"]`) | canonical phrase | status |
| --- | --- | --- | --- |
| everyone | `{"select": "all"}` | (none) | every family |
| the smaller / larger group (two groups), the smallest / largest (three) | `{"select": "group", "by": "size", "which": ...}` | "the smaller group" | added (in scenes of 2 or 3 separate groups, `place_groups`) |
| the group on the left / right / back / front | `{"select": "group", "by": "side", "which": side}` | "the group on the left" | added |
| the group nearest the red cheese | `{"select": "group", "by": "cheese", "which": colour}` | "the group nearest the red cheese" | added |
| everyone except one of those | the same with `"except": true` | "everyone except the smaller group" | added |
| the robots on top (standing on other robots) | `{"select": "top"}` | "the robots standing on other robots" | added |
| N of you (3 to 8), half of you | `{"select": "count", "count": N}`, `{"select": "half"}` | "5 of you", "half of you" | added |
| two groups, each its own job | `parallel` (jobs region, tower, surround, eat) | "split into two groups: one should ..., the other should ..." | exists |
| the one / ones in a region ("the robots in the back left") | | | covered by the side selector in multi-group scenes; a bare region cut through a group is not drawn (a person would not ask it of one blob) |
| everyone except the tower, the robots that built X | | | out: needs memory of who did what, or a structure standing at the start; a tower is a group like any other once it stands, so "the larger group" etc. apply |
| robot number 7, the red robot | | | out: robots have no identity or colour |

A group is a 26-connected set of robots (`suite.components`). Group selectors are drawn only in
scenes whose groups the words tell apart: sizes at least 1.4 times apart, a side or a cheese with
one group clearly nearest (3 voxels clear of the next). The robots a selector leaves out must stay
where they are.

### What

| job | spec kind | canonical sentence | status |
| --- | --- | --- | --- |
| stay, stop, freeze, stay where you are | `stay` | "stay where you are" | added (motion; also the usual second instruction of a switch) |
| go to a region / corner | `goto` | "go to the back left of the floor" | added for everyone and for selected robots (`parallel` had it only as one of two jobs) |
| go next to a cheese (without eating it) | `approach` | "go next to the red cheese without eating it" | added |
| gather into one group | `gather` | "gather into one group" | added |
| spread out | `spread` | "spread out so that no two of you touch" | added |
| split into two / three groups | `groups` | "split into two separate groups" | added |
| line up | `line` | "line up in one row from left to right" (or back to front) | added, swarms of up to 14 (`LINE_MAX`) |
| climb onto a plateau | `climb` | "climb onto the plateau on the left" | added (trench scenes; everyone, or N of you) |
| climb on top of a cheese | | "3 of you: climb on top of the red cheese" | out: under the pivot rule a cheese is no hinge, so a cube reaches its top only over a ladder of other cubes, and the count would have to exempt the ladder (0 of 9 under `switch_cube`). The checker and expert code (`perch`) stay; the draw leaves it out |
| move a few steps in a direction | `move` | "move three steps toward the left" | added |
| follow / chase a cheese without eating it | `follow` | "follow the red cheese without eating it" | added (the cheese flees, as a runner) |
| catch / eat cheeses by colour logic | `eat` | "eat the red cheese", "eat every cheese except ..." | exists (logic, runner); eating by a selected group added |
| surround a cheese | `surround` | "surround the red cheese" | exists (parallel) |
| push a cheese to a region or into the trench | `herd_region`, `herd_trench` | "push the red cheese to the left of the floor without eating it" | exists |
| letters, shapes | `mask` | "write A flat on the floor", "form a hollow cube" | exists |
| tower, bridge, fill, stairs | `tower`, `bridge`, `fill`, `stairs` | "build a tower 5 robots tall" | exists; a tower by a selected group added |
| avoid a cheese, keep away from it | | "stay away from the red cheese" | out for now: not added (a spread-like expert would do); "do not eat any cheese" exists |
| swap the two groups, form a circle around the other group | | | out: needs identity across the episode or a checker of relative layouts |
| undo, go back to where you started | | | out: the policy sees one frame; the start is gone by then. Needs memory (a history in the context, or the start frame as input) |
| come to me, follow me, look at me | | | out: no user in the world |
| faster, slower, carefully | | | out: one step per tick, no speed to set |
| dance, wave, celebrate | | | out: no checkable outcome |
| how many of you are there, which group is larger (questions) | | | not an action: the answer is in `scene_summary` and belongs to the text side, with "stay" as the action |
| carry a cheese, lift a cheese, open / close, push a lever | | | out: cheeses are only pushed; no levers or doors (README, "Left out") |
| build a ramp, a wall to block a runner, a shelter | | | out for now: no checker; a wall is a shape (formation) |

### When (modifiers)

| modifier | example | status |
| --- | --- | --- |
| except (on colours) | "eat every cheese except the red one" | exists (logic) |
| except (on robots) | "everyone except the smaller group: ..." | added (selector `except`) |
| then, in order (eating) | "eat the red cheese first, then the blue cheese" | exists (logic ordered) |
| a new instruction mid-episode | "go to the left" ... (step k) "stop" / "actually, gather into one group" | added (`switch`): task A from step 0, task B from step k (8 to 24); B is stay half the time, else go to another region, gather or spread |
| then, for other jobs in one instruction ("go left, then build a tower") | | out for now: the checker sees the start and the end only; a sequence needs the step at which the first part was done. The switch family covers the same behaviour with the second part given as a new instruction |
| until (an event) | "spread out until the red cheese is gone" | out: until-conditions on the world's own events do not occur (cheeses do not vanish by themselves); "until I say stop" is the switch |
| at the same time (two jobs) | "half of you go left, the other half build a tower" | exists (parallel) |
| only N / at most N robots | "use only 3 robots" | added as the count selector |

### What the expert does on them

The coverage jobs run through the suite's `Expert` as one `{"kind": "job"}` plan segment.
`JobSetup` makes a job concrete on the scene as the job starts: which robots take part (a count
selector takes the N nearest where the job goes, and targets no voxel a robot started on, since
the checker counts the robots off their start voxels), which are frozen (they stay, jitter or
not), and the target voxels (piles grown outward and up from a floor point, a ring at distance
1 to 3 from a cheese, one level row, the shape shifted three cells, piles around the centres of
two or three far-apart regions, a 2 x 2 column). Spreading and following are recomputed every
step. Robots on a target of a line or a spread hold still (`HOLD_STILL`); elsewhere the jitter
stays, since a ring or pile needs it to let the last robots in. When the targets stand on raised
ground at least 3 high, robots that are neither up there nor at the foot of its cliff go to the
foot first (`JobSetup.detour`): the greedy controller never steps down a cliff by itself.

```
PYTHONPATH=src OMP_NUM_THREADS=4 python scripts/vla_voxel_coverage.py --scenes 144 --out runs/coverage/expert.jsonl
```

144 fresh scenes per family and arm (seed 777, not a seed the tests or the pinned draws use).
Cells are mean checker score / share passed (score at least 0.75 and nothing forbidden eaten), as
`demos` keeps a demonstration. default: all 27 moves, 48 steps, 51 to 115 robots. cube: `demos
--cube` (14 moves, 72 steps, 6 to 115 robots log-uniform, failures 0.15). switch: cube with
`--rule switch_cube`, gravity on everywhere: the training set-up. Table and rows in
`runs/coverage/expert.{txt,jsonl}`.

| | default | cube | switch |
| --- | --- | --- | --- |
| motion | 0.91 / 0.83 | 0.95 / 0.93 | 0.95 / 0.92 |
| subset | 0.78 / 0.69 | 0.92 / 0.89 | 0.92 / 0.88 |
| switch | 0.89 / 0.79 | 0.95 / 0.92 | 0.95 / 0.92 |

The suite's own families under the same arms passed 0.79 (cube) and 0.71 (switch) in the free-cube
comparison above. Per kind under `switch_cube` (n): stay 1.00 (20), goto 1.00 (18), approach 1.00
(19), follow 0.94 (16), gather 0.85 (13), groups 0.95 (19), line 0.86 (21), climb 0.77 (13), spread
0.75 (4); selected robots: goto 0.86 to 1.00, approach 0.71 to 0.90, line 0.80 to 1.00, climb
(N of you) 0.80, move 0.80 to 1.00, groups (one group splits) 0.70, spread 0.67 to 1.00, tower
1.00 (4); switch episodes scored on the second instruction: then stay 1.00 (80), then gather 1.00
(19), then goto 0.91 (33), then spread 0.33 (12). Weak spots: spreading a large swarm that a first
instruction has just packed together (close to the floor's capacity of spaced cells), large
swarms climbing a plateau in 72 steps (scores still rising at the end), and eating by a selected
group in 48 steps (default arm 0.11 of 9; the cube arms drew none, since a group of 30 robots is
rare in log-uniform swarms). A failed attempt costs nothing but its row: demos keeps passes only.

### Drawing them into demonstrations and labels

`demos --coverage F` (and `haiku.py export --coverage F`) adds round(F x scenes) coverage tasks
to every chunk of `demos.draws`, after the suite's own tasks and from a generator of their own
(seeded by `(seed, 7)`), so the suite's part of the draw is the same with or without them. Haiku
labels join to draws by (seed, index), so the coverage draw is pinned by a digest test
(`test_the_coverage_draw_is_pinned`). What is drawn depends only on the makers and the frozen
draw code in `suite.py` (`drawn_score`, `drawn_pile`, `drawn_spread_cells`, `coverage_bound`,
`COVERAGE_WORDINGS`), never on the expert or the checker: an expert that fails a task only marks
its row of `attempts.jsonl` with `e = -1`.

The store records which instruction is in force at every step:

1. `labels.jsonl` has one row per instruction: a switch episode has two rows for the same `e`,
   `w0 = 0, w1 = k` with the first instruction and `w0 = k, w1 = steps` with the second, so
   `lm.batch_of` reads it as before and labels each step with the instruction in force.
1. `small.pt` holds `instructions`: per kept episode `[{"t": first step, "text", "record":
   canonical_record}]`, one entry per instruction. `play_store.instructions_of(store, e, labels)`
   reads it, and for an older store without it rebuilds `[{"t", "text"}]` from the label rows;
   `play_store.instruction_at(entries, t)` gives the one in force at step t.
1. The compact grid files are unchanged; `config` records `coverage`.
1. `subgoals` go on across the switch: a new entry at step k with plan index 0 of the new
   instruction.

### Records for labelling

Two functions give every task and scene in plain JSON (names stable; the Haiku labelling pipeline
reads them):

```python
record = suite.canonical_record(task)    # any Task, old or new
summary = suite.scene_summary(world, i)  # scene i of a VoxelWorld batch
```

`canonical_record(task)` returns `{"family", "kind", "plain", "spec", "subgoals"}`:

1. `kind`: `suite.task_kind`, e.g. `"eat:ordered"`, `"goto@smaller"`, `"switch:goto>stay"`.
1. `plain`: one canonical English sentence (`wordings.canonical`; `wordings.parse_canonical`
   inverts it). A subset task reads `"the smaller group: go to the back left of the floor;
   everyone else: stay where you are"`. A switch task joins its instructions with
   `"; then, on a new instruction: "`.
1. `spec`: `suite.record_spec`, what an instruction has to pin down in this scene (the verifier
   of `haiku.py` reads an instruction back into it): the six families as `haiku.py` had them
   (an eat task's forbid set left out, an eat task naming an absent colour `{"kind":
   "refuse"}`); a coverage task its job's kind and fields (`goto`: region, `approach`, `follow`,
   `perch`: colour, `groups`: count, `line`: axis, `climb`: side, `move`: toward and steps,
   `tower`: height, `eat`: eat, any_of, order) plus the selector as flat fields: `who` (group,
   top, count, half), `by` and `which` for a group, `count`, and `"except": true`. No `who`:
   everyone. A switch task's spec is `{"kind": "switch", "parts": [{"t": first step, "record":
   record}, ...]}`. `suite.SPEC_NOTES` explains every kind for the verifier's menu.
1. `subgoals`: one plain sentence per plan segment, in plan order: an eat segment eats the next
   colour, any other segment is the task's plain sentence; a switch task's parts in turn.

`scene_summary(world, i)` returns `{"gravity", "terrain", "robots", "cheeses"}` (x left to right,
y back to front, z up; areas are the nine `FLOOR_REGIONS`):

1. `terrain`: `{"layout": "flat floor"}`, `{"layout": "trench between two plateaus", "trench":
   {"runs", "from_x" or "from_y", "width", "depth"}, "plateau_height"}` or `{"layout": "low
   hills", "highest"}`.
1. `robots`: `count`, `mostly_around` (the area of the centroid), `clumps` (groups of at least a
   tenth of the swarm, `facts.groups`), `groups` (every 26-connected group of 2 or more robots,
   largest first: `size`, `area`, `x`, `y` of its centre, `height`), `alone` (robots touching no
   other), `on_top` (robots standing on robots), `towers` (vertical runs of 3 or more:
   `height`, `area`).
1. `cheeses`: `colour`, `area`, `on` (floor, plateau or hill), `x`, `y`, `z` (lowest corner).

These are the shapes of `haiku.py`'s former stand-ins (`local_canonical_record`,
`local_scene_summary`), which `haiku.py` now replaces with the suite's own; the summary only adds
fields to the `robots` entry. A test checks both against the stand-ins on the six families.

## Objects

Objects with properties, passive blocks, floor zones, and reasoning over them (`world.py`,
`objects.py`). A world gets them through optional fields of `VoxelWorld`; a world without them
(every draw of the six families and of the coverage families) is the world above, unchanged, bit
for bit (`test_default_objects_step_and_observe_as_before`).

### The world

| field | shape | what |
| --- | --- | --- |
| `form` | (B, K, 4, 4, 4) bool | the object's voxels from `cheese_pos` (its lowest corner) |
| `colour2` | (B, K) long | second colour of a pattern, -1 for none |
| `pattern` | (B, K) long | `PATTERNS`: plain, striped, two-tone |
| `friction` | (B, K) float | 2 (light, `FRICTION`) or 6 (heavy, `HEAVY_FRICTION`) |
| `food` | (B, K) bool | a cheese; False for a passive block |
| `zones` | (B, H, W) long | the colour painted on each floor cell, -1 for none |

1. **Shapes** (`object_form`): a cube of side 1 to 4, a slab (side 2 to 4, one high), a rod
   (length 2 to 4, one thick, along x or y), an L lying flat (arms 2 to 4). Size is the longest
   side; "big" in a superlative is the voxel count.
1. **Patterns** run along the longest side (ties: x, then y, then z): striped alternates the two
   colours voxel layer by layer, two-tone gives each half its colour. Striped needs a side of 3.
1. **Push.** As for cheeses: the press along the strongest axis must reach the object's friction
   and the voxels it moves into must be clear; robots there are shoved on (all or none). A heavy
   object also counts robots pushing in line behind a pressing robot with the same move, up to
   `CHAIN_DEPTH` (3) behind it (`chain_press`), so six robots move it whatever the size of its
   face; under the pivot rule such a move into a pressing robot is a press too. A light object
   counts only the robots moving into it, as before.
1. **Eat.** Food only: robots fill `EAT_SHARE` of the object's open shell and `EAT_MIN_SHARE` of
   its full shell (every voxel touching it). On the open floor that is 13 robots for a side-1
   cube, 30 for side 2 (the old cheese), 55 for side 3, 87 for side 4. Blocks are never eaten.
1. **Fall.** An object falls when nothing (wall, object, robot, floor) is under any of its lowest
   voxels. Runners (`suite.flee`) slide any form the same way.
1. **Weight is not observed.** Nothing in the observation tells a light object from a heavy one.

`sample_object_scenes` draws terrain as `sample_scenes` does (its terrain code is now
`draw_terrain`, called by both), 0 to 3 zones (patches 3 to 4 cells a side, or stripes one cell
wide and 6 to 10 long, on level ground, each its own colour), up to `MAX_OBJECTS` (6) objects on
level ground off the zones and a voxel apart (shape cube 0.4, slab, rod, L 0.2 each; pattern plain
0.6, striped 0.2, two-tone 0.2; heavy 0.35; food with a given probability), then the robots.

### Observation channels

| channel | content |
| --- | --- |
| 0 | robot |
| 1 | wall |
| 2 | object (any object voxel, cheese or block) |
| 3, 4, 5 | object RGB, per voxel (patterns show) |
| 6 | food (object voxels that can be eaten) |
| 7, 8, 9 | zone RGB, on the voxel just above the ground of each painted cell (where a robot standing on the zone is) |

`world.CHANNELS` names them; `N_CHANNELS` is 10 and `OLD_CHANNELS` 6. Channels 0 to 5 are the old
observation, so an old world gives the old six, food equal to the cheese mask and no zones.
`observe(world, channels)` cuts the list; `lm.VoxelLMPolicy(..., channels=OLD_CHANNELS)` builds a
policy for a checkpoint trained before objects (it reads the first six of any observation).

### Families

`objects.OBJECT_FAMILIES`, drawn by `objects.sample_objects`, never by `sample_tasks` or
`sample_coverage`:

| family | kinds (`task_kind`) | example |
| --- | --- | --- |
| things | `eat_thing:attrs`, `push_thing:attrs>region` / `>trench`, `surround_thing:attrs`, `catch_thing:attrs`, `push_thing:weight>...` | "gobble up the striped cheese", "push the heavy cheese to the back left" |
| blocks | `push_thing:attrs>...`, `push_thing:weight>...`, `line_blocks` | "push the long white block into the trench", "line the blocks up in one row from left to right" |
| zones | `goto_zone`, `line_zone`, `push_thing:attrs>zone` | "go to the blue zone", "line up along the red stripe", "push the large block onto the green zone" |
| reasoning | `{eat,push,surround}_thing:{super,compare,near,next_to}`, `conditional:{eat_tower,surround_goto}@{then,else}`, `count:tower` | "eat the biggest cheese", "push the smaller of the two blue blocks to the left", "surround the cheese furthest from the green zone", "push the cheese next to the yellow block into the trench", "eat the red cheese if there is one, otherwise build a tower 5 robots tall", "build a tower as tall as the number of blocks" |

1. **References** (`spec["ref"]`, `objects.resolve`): `attrs` (properties among size word, pattern,
   colour, shape word, weight word; a colour matches either colour of a patterned object), `super`
   (biggest, smallest by voxels, 1.5 times the next), `compare` (the smaller or larger of exactly
   two matching, 1.5 times apart), `near` (nearest the trench or a zone, furthest from it, 2 cells
   clear of the next), `next_to` (within 3 cells of an anchor named by `attrs`, 2 cells clearer
   than the next). A task is drawn only when its reference picks out exactly one object, and the
   spec keeps that slot as `target`. Descriptions in `things` and `blocks` use the fewest
   properties that pick the object out, at random among those.
1. **Weight** is only named when the task is about it ("the heavy block"): exactly one of the
   objects of that noun is heavy and each can be tested. The expert finds it by trying (below).
1. **Conditionals** name a colour that one cheese has half the time and no cheese has otherwise;
   the spec keeps which branch holds. **Counting** asks a tower as tall as the number of cheeses,
   blocks or objects (3 to 6).
1. **Feasible draws only.** A push goes to a floor region on a flat floor (terrain would block
   the way, as for the suite's herd family) or into the trench of a gap scene with gravity; the
   object must be pushable that way by the expert's team (`pushable`: two robots on its face if
   light, six on its face and the rows behind if heavy) and the goal not already reached. Eat and
   surround jobs are drawn on flat or hill scenes (`LOW_GROUND`: the greedy controller does not
   climb a plateau's cliff). `bound` is the swarm check (eat_need for any form, a full team of
   eight for a heavy push, four robots a tower level).

### The expert (`objects.ObjectExpert`)

`suite.Expert` with three more segment kinds; everything else is the suite's (eat and surround
work on any form through the world's shells).

1. `shove`: a pushing team stands behind the object, `team_cells`: the face and up to three rows
   behind it, low and near first (twice the height plus the depth), eight cells at most. Each
   step every robot in a team cell steps toward the object, overriding jitter; robots in the
   object's way step aside, ahead or up, whichever is free (`adjust`, a hook `suite.Expert` calls
   before the pivot rule). The heading is the floor axis with the larger offset to the goal whose
   way is clear, else a sidestep away from what blocks it. The team is the same whatever the
   object weighs: the expert does not use weight either.
1. `find` ("the heavy block"): two probers press each candidate in turn, nearest first, along a
   way whose ground face holds two and whose way ahead is clear (`probe_way`), with two spares. A
   candidate that moves is light; one that has not moved after five steps of both pressing is
   heavy (two robots move any light object). Then the shove of the one found. The sub-goal record
   carries the candidates and, once known, `found`.
1. `fixed`: target voxels with a cap on the robots taking part (zones, stripes, towers).

Expert success on fresh seeds under the training set-up (`demos --cube --rule switch_cube`: 14
cube moves, 72 steps, 6 to 115 robots log-uniform, failures 0.15, gravity on), seed 909, 144
scenes per family; cells are mean score / share passed (score at least 0.75, nothing unasked
eaten). Rows in `runs/objects/expert.{txt,jsonl}`:

```
PYTHONPATH=src OMP_NUM_THREADS=4 nice python scripts/vla_voxel_objects.py --scenes 144 --arms switch --out runs/objects/expert.jsonl
```

| family | score / passed (n) |
| --- | --- |
| things | 0.80 / 0.76 (144) |
| blocks | 0.80 / 0.74 (144) |
| zones | 0.86 / 0.83 (144) |
| reasoning | 0.85 / 0.83 (144) |

Per kind (n): things: eat 0.76 (38), catch 0.73 (22), surround 0.95 (38), push to a region 0.63
(35), into the trench 0.71 (7), the heavy one to a region 0.00 (3), into the trench 1.00 (1).
Blocks: push to a region 0.73 (92), into the trench 0.90 (10), the heavy one to a region 0.18
(11), into the trench 1.00 (3), line up 0.93 (28). Zones: go to 0.94 (66), line up on a stripe
0.77 (48), push onto 0.67 (30). Reasoning: conditionals 0.93 to 1.00 (4 to 28 per form and
branch), count tower 0.89 (9), superlatives: eat 0.67 (3), push 0.67 (9) and 1.00 into the
trench (4), surround 1.00 (9); comparisons: eat 0.33 (9), surround 1.00 (5); nearness: push 0.50
(6), surround 1.00 (5); next to: eat 0.86 (7), push 0.45 (11), 0.33 into the trench (3). By swarm:
under 15 robots 0.76, 15 to 40 0.80, over 40 0.79.

Weak spots: pushing the heavy one to a floor region (in the episodes traced the probe found the
right one; the push then moved it slowly, six of eight pressers must succeed in one step, and
stalled on robots it could not shove, as in an L's notch); pushes that need a route around
another object (the heading only sidesteps); eating in a comparison (3 of 9, not traced). A
failed attempt costs nothing but its row: demos keeps passes only.

### Drawing them into demonstrations and labels

`demos --objects F` (and `haiku.py export --objects F`) adds round(F x scenes) object tasks to
every chunk of `demos.draws`, after the suite's own and the coverage tasks, from a generator of
their own seeded by `(seed, 11)`: the draws before them are unchanged
(`test_objects_come_after_the_other_draws_and_leave_them_alone`). With `--objects` every world of
the draw carries the object fields and `MAX_OBJECTS` slots (old scenes padded with empty slots and
default fields, which play exactly as before), the expert is `ObjectExpert` (attractor only; the
free-cube planner does not know the new segments), and `small.pt` holds per episode `form`,
`colour2`, `pattern`, `friction`, `food` and `zones`. What is drawn depends on the makers, the
sampler and the draw-time checks of `objects.py` (`describe`, `resolve`, `pushable`, `reachable`,
`team_cells`, `probe_way`, `bound`), never on the expert's run or the checker: a digest test pins
it (`test_the_object_draw_is_pinned`). Change those knowingly.

**Old stores** read as before: `lm.world_at` passes the object fields only when the store has
them, `observe` gives an old world's old six channels plus food and empty zone channels, and
`batch_of` returns 10-channel observations from any store.

### Records and wordings

1. `canonical_record(task)`: `plain` from the wording bank's object intents (`thing.eat`,
   `thing.push_region`, `thing.push_trench`, `thing.push_zone`, `thing.surround`, `thing.catch`,
   `blocks.line_x`, `blocks.line_y`, `zone.goto`, `zone.line`, `cond.eat_tower`,
   `cond.surround_goto`, `count.tower`; held out `cond.eat_goto`, `count.groups`), whose `thing`
   slot is a canonical phrase ("the large striped red cheese", "the smaller of the two blue
   blocks", "the cheese nearest the trench", "the block next to the yellow cheese";
   `wordings.thing_pattern` reads it back). A phrase that is only a colour and "cheese" reads back
   as the colour intent of the same sentence ("eat the red cheese"). `spec`: the job's kind and
   `target`, the object's `id` in the scene summary, with where it goes (`region`, `trench`,
   `zone`), `weight: heavy` when the instruction names the weight; a conditional `{"holds",
   "do"}`; counting the number. `subgoals`: each segment with the object named by every visible
   property ("push the huge striped purple flat block to the front right of the floor"), "find
   the heavy block by pushing each block with two robots, then push it ..." for a find.
1. `scene_summary(world, i, task)` adds `objects` (`id`, `kind` cheese or block, `colour`,
   `second_colour`, `pattern`, `shape`, `size` word, `sides`, `voxels`, `area`, `on`, `x`, `y`,
   `z`, and `weight` only when the task tells it) and `zones` (`colour`, `kind` zone or stripe,
   `area`, `x0`, `x1`, `y0`, `y1`); in a world with objects `cheeses` lists food only.
1. `SPEC_NOTES` has notes for every new spec kind (the verifier's menu).
1. **Words** (`wordings.json` `properties`): per property a canonical word, training expressions
   and held-out ones, which `slot_expressions` lists, so the token and leak tests cover them:
   sizes (tiny, small, large, huge; held out minuscule, petite, sizeable, colossal), weights
   (heavy, light; hefty, airy), patterns (striped, two-tone; banded, bicolour), shapes (cube-shaped,
   flat, long, L-shaped; plate-like, elongated, angled), nouns and plurals (block: brick; crate),
   superlatives and comparatives (biggest, smallest, smaller, larger; bulkiest, puniest, punier,
   greater), nearness and relations (nearest, furthest from, next to; handiest to, most distant
   from, alongside), zone and stripe (patch, strip; pad, streak). The "words" split rewrites a
   thing phrase with the held-out words; training rewrites never use them. Shared tokens added:
   to, from, two, l (rulings in the file).

### Held out (`objects.HELD_OUT`, test splits of `sample_objects`, no plan)

| split | what |
| --- | --- |
| `ref_job` | the compositions (compare, push) and (next_to, surround): each part is trained, the pair never |
| `condition` | `cond.eat_goto`: "eat the c cheese if there is one, otherwise go to the r of the floor" (both branches are trained in other conditionals) |
| `count` | `count.groups`: "split into as many groups as there are blocks" |
| `wording` | the held-out templates of every object intent |
| words | the held-out property words above, through the wordings' "words" split |

### Assumed, not measured

1. Weight as friction 6 with presses through robots in line: no hardware number behind it.
1. Objects are rigid, never rotate, never tip, and an L or a rod does not pivot when pushed off
   its middle; a rod pushed end on needs its one face voxel plus the rows behind it.
1. A robot standing on an object is supported, and robots can climb objects (shells include
   the top), as for cheeses.
1. Zones are paint: they do not change physics. Pressure plates and gates are not implemented.

## Build: objects by name

"build a chair", "make an arch", "build something you can sit on": the swarm builds a recognisable
object of a library of named shapes, part by part, so a language model post-trained on the
demonstrations can bring in what it knows of objects. `shapes.py` holds the library and the rules,
`build.py` the family, its expert and its score. The family is drawn by `build.sample_build` and by
`demos --build F` (F build tasks per scene, after the coverage and object tasks, from a generator seeded by
`(seed, 13)`), never by `sample_tasks` or `sample_coverage`, so every other draw is unchanged
(`tests/test_vla_voxel_build.py` pins the build draw's digest). Training set-up: `--cube --rule
switch_cube`, flat floor, gravity on, 0 or 1 cheese, 6 to 115 robots.

### Objects are parts programs

An object is a short program of parameterised parts, each a primitive with a position (its lowest
corner, z y x), a size (h d w), an axis where it has one and a label: the group (sub-goal) it
belongs to. A stool:

    legs: col 0 0 0 3 1 1, col 0 0 2 3 1 1, col 0 2 0 3 1 1, col 0 2 2 3 1 1; seat: slab 3 0 0 1 3 3

That is also the text form (`shapes.program_text`, `shapes.parse_program`, `shapes.program_mask`):
groups `label: part, part` separated by `; `, a part `prim z y x h d w [axis] [char]`. It parses
back to the same parts and the same voxels (tested for every library object), so a thought can
describe a new object in parts and the description can be turned into a target.

| primitive | voxels |
| --- | --- |
| `box`, `col`, `slab`, `wall`, `beam` | a solid block (the names say how it is used) |
| `arch` | a block with an opening under a one-cell lintel, legs one cell thick at both ends of the axis |
| `ring` | a rectangular outline, h layers; with an axis, an upright frame |
| `tube`, `disc` | an elliptic outline, a filled ellipse |
| `ramp` | steps rising along a signed axis, solid under each step |
| `roof` | a hollow A-frame along the axis, bands two cells wide so each step rests on the one below |
| `pyramid` | stepped and solid |
| `dome` | a hollow half ellipsoid, stepped so every layer rests on the one below |
| `glyph` | a letter or digit of the 5 x 7 font standing upright; every diagonal step of the font gets the cell under its upper cell, so strokes are bonded face to face |

The plain sub-goals are the groups in build order, one sentence each (`shapes.subgoal_sentences`):
"four legs 3 tall", "a 3 by 3 seat on top", "a back 4 long and 2 tall on top". They are the
record's `subgoals` and so the thought's switch lines. Programs turn about the vertical
(`shapes.rotate`, equal to `np.rot90` of the voxels), and a draw turns each object a random
number of quarter turns.

### Support: what stands, what can be built cube by cube

The world's gravity is unchanged and permissive (anything face-connected to the ground stands).
The library uses a stricter rule for bonded cubes (`shapes.levers`, `shapes.stands`):

1. A cube on the floor (or terrain) has lever 0. A cube resting on a held cube has that cube's
   lever. A cube bonded face to face beside a held cube, or hanging under one, has one more.
1. A cube is held if its lever is at most `CANTILEVER` (3). So a bonded overhang reaches three
   cells out from a column, a span between two columns can be six cells wide, and a column
   standing on a cantilever's tip keeps the tip's lever. No load, torque or bond strength.
1. **Buildable** (`build_order`): there is an order in which every partial structure stands.
   Adding a cube never raises a lever, so a cube that can be placed stays placeable and the greedy
   order (lowest first) finds one whenever one exists. An object is built group by group in
   program order and passes only if every group can be completed on the earlier ones.
1. **Climbable** (`climb_order`, `reach`): every cube can also walk to its voxel when it is placed,
   a lone cube pivoting by the rule (`world.pivot_moves`, bonds released) from the floor beside the
   site over the cubes placed so far. Two things this rules out, both real for pivoting cubes:
   1. A **slot**, an empty voxel with placed cubes on two opposite sides in its layer: every pivot
      into it swings through one of them. Layers are therefore filled as a raster sweep (the first
      of the eight raster orders that works), and a part that fills a gap goes before the part that
      closes it: wall with a door is left wall, lintel, right wall; a shelf is left side, shelves,
      right side.
   1. **Overhangs out of reach**: a cube cannot get round an overhang from below (no support above),
      so a crown wider than a plus on a one-cell trunk cannot be finished: trees, pine trees,
      mushrooms and the lighthouse's gallery fail. Building them needs scaffolding that is taken
      away afterwards (possible with switchable bonds; not done).

`fit_variant` draws only variants that fit the grid (floor plan at most 12 x 12, at most 11 tall),
need at most the swarm's robots, are buildable and climbable, aiming at a size between 0.4 and 1.0
times the swarm.

### Library

81 names (45 objects, 26 letters, 10 numbers) in 11 categories; 16 held out. Shares over 32 draws per name (`shapes.survey`, `runs/build/survey.json`): stands under the support rule, buildable group by group in program order, climbable as well; robots (min): the fewest robots any draw needs (`build.min_sizes`, `none`: never climbable, so never drawn). 63 names can be drawn with 6 to 115 robots.

| category | object | held out | stands | buildable | climbable | robots (min) | voxels (min, median, max) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| furniture | stool |  | 1.00 | 1.00 | 1.00 | 8 | 8, 17, 28 |
| furniture | chair |  | 1.00 | 1.00 | 1.00 | 14 | 18, 28, 44 |
| furniture | throne | yes | 1.00 | 1.00 | 1.00 | 29 | 32, 46, 60 |
| furniture | table |  | 0.91 | 0.91 | 0.84 | 21 | 20, 46, 75 |
| furniture | bench |  | 1.00 | 1.00 | 0.94 | 12 | 12, 24, 64 |
| furniture | sofa |  | 1.00 | 1.00 | 1.00 | 32 | 32, 52, 76 |
| furniture | bed |  | 1.00 | 1.00 | 0.94 | 24 | 22, 40, 56 |
| furniture | shelf |  | 1.00 | 0.59 | 0.59 | 10 | 12, 28, 52 |
| structure | arch |  | 1.00 | 1.00 | 0.47 | 5 | 6, 12, 20 |
| structure | bridge |  | 0.69 | 0.69 | 0.09 | 9 | 9, 36, 49 |
| structure | gate |  | 1.00 | 1.00 | 0.25 | 10 | 11, 12, 15 |
| structure | wall with a door |  | 1.00 | 1.00 | 1.00 | 13 | 11, 23, 41 |
| structure | fence |  | 1.00 | 1.00 | 1.00 | 11 | 11, 16, 29 |
| structure | castle tower |  | 1.00 | 1.00 | 1.00 | 28 | 28, 56, 120 |
| structure | staircase | yes | 1.00 | 1.00 | 1.00 | 6 | 6, 19, 63 |
| structure | pyramid | yes | 1.00 | 1.00 | 1.00 | 10 | 10, 35, 56 |
| structure | ladder |  | 1.00 | 1.00 | 1.00 | 10 | 10, 18, 24 |
| structure | obelisk |  | 1.00 | 1.00 | 1.00 | 14 | 14, 44, 107 |
| structure | podium |  | 1.00 | 1.00 | 1.00 | 6 | 6, 18, 54 |
| structure | slide |  | 1.00 | 1.00 | 1.00 | 8 | 8, 19, 42 |
| structure | lamp post |  | 0.75 | 0.75 | 0.38 | 6 | 6, 7, 8 |
| building | house |  | 1.00 | 1.00 | 1.00 | 48 | 58, 98, 156 |
| building | hut |  | 1.00 | 1.00 | 1.00 | 25 | 25, 60, 84 |
| building | doghouse | yes | 1.00 | 1.00 | 1.00 | 28 | 28, 38, 48 |
| building | igloo |  | 1.00 | 1.00 | 1.00 | 16 | 16, 40, 83 |
| building | tent |  | 1.00 | 1.00 | 1.00 | 24 | 24, 48, 84 |
| building | lighthouse |  | 1.00 | 1.00 | 0.00 | none | - |
| vehicle | car | yes | 1.00 | 1.00 | 0.91 | 16 | 16, 22, 34 |
| vehicle | truck | yes | 0.88 | 0.88 | 0.62 | 38 | 38, 53, 81 |
| vehicle | train | yes | 1.00 | 1.00 | 1.00 | 11 | 11, 22, 42 |
| vehicle | plane | yes | 0.44 | 0.44 | 0.38 | 12 | 12, 16, 20 |
| vehicle | boat | yes | 1.00 | 1.00 | 1.00 | 32 | 34, 50, 65 |
| vehicle | rocket | yes | 1.00 | 1.00 | 1.00 | 13 | 13, 54, 90 |
| container | cup |  | 1.00 | 1.00 | 1.00 | 17 | 17, 41, 73 |
| container | pot |  | 1.00 | 1.00 | 1.00 | 20 | 20, 33, 80 |
| container | crate |  | 1.00 | 1.00 | 1.00 | 17 | 17, 46, 96 |
| plant | tree |  | 1.00 | 1.00 | 0.00 | none | - |
| plant | pine tree |  | 1.00 | 1.00 | 0.00 | none | - |
| plant | cactus | yes | 1.00 | 1.00 | 1.00 | 6 | 4, 10, 13 |
| plant | mushroom |  | 1.00 | 1.00 | 0.00 | none | - |
| symbol | cross |  | 1.00 | 1.00 | 1.00 | 6 | 6, 9, 13 |
| symbol | standing ring |  | 1.00 | 1.00 | 0.56 | 8 | 8, 15, 32 |
| figure | snowman |  | 1.00 | 1.00 | 1.00 | 61 | 61, 62, 62 |
| animal | dog |  | 1.00 | 1.00 | 1.00 | 17 | 17, 20, 25 |
| animal | giraffe |  | 1.00 | 1.00 | 1.00 | 23 | 23, 25, 27 |
| letter | 26 glyphs; buildable: A B D H I J K L N O Q T U V W X Y; not: C E F G M P R S Z | K Q Z | 0.69 | 0.69 | 0.65 | | |
| number | 10 glyphs; buildable: 0 1 4 6 8; not: 2 3 5 7 9 | 4 7 | 0.50 | 0.50 | 0.50 | | |

Over all 81 names, mean share of draws: stands 0.82, buildable 0.82, climbable 0.72; names with at least one climbable draw: 63.

Not drawable: tree, pine tree, mushroom, lighthouse (overhangs out of reach), letters C E F G M P R S Z and numbers 2 3 5 7 9 (an overhang of four, or a hook that hangs four out). The variants that do not stand are draws whose spans or overhangs come out too long (a bridge over 8, a table with legs 5 apart, a lamp post's lamp hanging under a 3-long arm, a plane's wings).

### Functions

Checks of what a structure is good for, on any occupancy grid (`shapes.FUNCTIONS`); they score the
use tasks ("build something you can sit on": any training object that serves the use, chosen by
the expert) and can serve as an RL reward.

| use | check | objects that serve it |
| --- | --- | --- |
| sit | a supported flat surface, at least 2 x 2, clear above, at height 2 to 4 (top layer z 1 to 3) | stool, chair, bench, sofa, bed (throne held out) |
| table | a supported clear 2 x 2 top at z 2 or higher with open space under at least half of it | table |
| shelter | empty voxels with structure above and on at least two sides within 3 cells (2 such voxels) | house, hut, igloo, tent (doghouse held out) |
| span | a straight run at z 1 or higher with two cells carried by unbroken columns and, between them, a cell with nothing under it | arch, bridge |
| contain | a cavity: empty voxels walled in all round in their layer, on a bottom of their own (not the bare floor), open above | cup, pot, crate (boat held out) |

Every library object with a use passes its check in at least 80% of draws (tested); a scattered
swarm passes none of them in more than a quarter of scenes, and a draw is refused if the start
already passes.

### Score

A named object scores the best IoU of the final robots with its target voxels over every placement
on the grid and the four quarter turns (`shapes.object_iou`): hits / (target + robots inside the
target's box grown by one cell across the floor and one up - hits). The instruction says neither
where nor which way round, and robots away from the object (the ones not needed) cost nothing.
Pass: IoU at least 0.75 (`build.OBJECT_PASS`). A use scores its check (pass at 1.0). The checker is
also offered under `suite.score` (spec kind `build`).

### The expert (`build.make_expert`)

Under a pivot rule, `BuildPlanner`: `free_cube`'s planner (breadth-first pivots to the targets
through the rule's own moves) with build segments; without a rule, `BuildExpert` (the attractor).

1. One plan segment per group of parts, in program order; its points are every voxel of this group
   and the earlier ones, in the placing order of `climb_order`. A segment ends when all its points
   are filled, or after 12 + 4 x its voxels steps.
1. Robots on a filled voxel hold still. The open voxels are the first 4 (`WINDOW`) empty ones of the
   placing order that the support rule lets a cube take now, skipping one while an earlier
   neighbour in its layer is empty.
1. A crew of 3 robots per open voxel (`CREW`), those with the fewest pivots to go that can move at
   all, follow the planner; a crew robot the planner leaves standing takes a random allowed move
   one time in four (`JITTER`: jams of robots in each other's swing). Everyone else stays; those
   within two cells of the site's box park on free ground three to six cells out.
1. A filled voxel that leaves an earlier voxel of the order in a slot gives its robot up (it is
   counted empty and its robot leaves), and is filled again after the gap: robots that start on
   the site, or arrive early, otherwise wall in holes for good.
1. The sub-goal record (`subgoals` in `small.pt`) has `{"kind": "build", "object", "part", "index",
   "of"}` per group.

What did not work: the attractor under `switch_cube` (robots stall under overhangs and in jams:
mean IoU 0.58 on a first sample); the lowest-layer-first frontier without the placing order
(slots: mean 0.70 on another); a crew of one robot per open voxel (one chair: 29 of 36 voxels at the end against 35 with three).

Measured on fresh seeds (`python -m beebot.vla.voxel.build --per-object 4 --seed 777`): 4 scenes per drawable name and per use, `--cube --rule switch_cube` (14 moves, gravity on, failures 0.15), 6 to 115 robots log-uniform (redrawn until the object fits), each episode `episode_steps` long. Held-out objects get a plan for this measurement only. Score: IoU (or the use's check); steps to pass: the first step (in steps of 4) the score passed. Rows in `runs/build/expert.jsonl`, tables in `runs/build/tables.md`.

| object | held out | n | voxels (median) | robots (median) | IoU or check | passed | steps given (median) | steps to pass (median) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dog |  | 4 | 21 | 56 | 1.00 | 1.00 | 144 | 44.0 |
| giraffe |  | 4 | 27 | 34 | 0.99 | 1.00 | 156 | 70.0 |
| house |  | 4 | 53 | 68 | 0.94 | 1.00 | 236 | 90.0 |
| hut |  | 4 | 56 | 71 | 0.95 | 1.00 | 248 | 114.0 |
| igloo |  | 4 | 29 | 46 | 1.00 | 1.00 | 134 | 30.0 |
| tent |  | 4 | 36 | 58 | 1.00 | 1.00 | 156 | 30.0 |
| crate |  | 4 | 40 | 64 | 0.96 | 1.00 | 182 | 46.0 |
| cup |  | 4 | 25 | 36 | 0.99 | 1.00 | 124 | 46.0 |
| pot |  | 4 | 24 | 34 | 1.00 | 1.00 | 120 | 36.0 |
| snowman |  | 4 | 61 | 88 | 0.91 | 1.00 | 280 | 156.0 |
| bed |  | 4 | 36 | 47 | 0.97 | 1.00 | 178 | 60.0 |
| bench |  | 4 | 13 | 16 | 0.98 | 1.00 | 76 | 46.0 |
| chair |  | 4 | 22 | 41 | 0.98 | 1.00 | 124 | 46.0 |
| shelf |  | 4 | 44 | 62 | 0.90 | 1.00 | 212 | 124.0 |
| sofa |  | 4 | 62 | 92 | 0.88 | 1.00 | 286 | 202.0 |
| stool |  | 4 | 16 | 23 | 1.00 | 1.00 | 88 | 32.0 |
| table |  | 4 | 36 | 51 | 1.00 | 1.00 | 168 | 50.0 |
| letter A |  | 4 | 30 | 62 | 0.73 | 0.25 | 132 | 48.0 |
| letter B |  | 4 | 48 | 61 | 0.99 | 1.00 | 204 | 60.0 |
| letter D |  | 4 | 30 | 74 | 1.00 | 1.00 | 132 | 46.0 |
| letter H |  | 4 | 34 | 67 | 0.97 | 1.00 | 148 | 46.0 |
| letter I |  | 4 | 30 | 44 | 0.95 | 1.00 | 132 | 62.0 |
| letter J |  | 4 | 20 | 32 | 1.00 | 1.00 | 90 | 40.0 |
| letter L |  | 4 | 22 | 45 | 0.99 | 1.00 | 100 | 24.0 |
| letter N |  | 4 | 28 | 28 | 0.96 | 1.00 | 126 | 58.0 |
| letter O |  | 4 | 20 | 32 | 0.98 | 1.00 | 92 | 36.0 |
| letter T |  | 4 | 22 | 69 | 0.97 | 1.00 | 100 | 60.0 |
| letter U |  | 4 | 34 | 74 | 0.98 | 1.00 | 148 | 32.0 |
| letter V |  | 4 | 34 | 62 | 0.96 | 1.00 | 148 | 42.0 |
| letter W |  | 4 | 28 | 37 | 0.91 | 1.00 | 126 | 42.0 |
| letter X |  | 4 | 20 | 26 | 0.69 | 0.00 | 92 | - |
| letter Y |  | 4 | 28 | 56 | 0.97 | 1.00 | 124 | 54.0 |
| number 0 |  | 4 | 38 | 72 | 0.84 | 1.00 | 162 | 64.0 |
| number 1 |  | 4 | 15 | 26 | 1.00 | 1.00 | 72 | 30.0 |
| number 6 |  | 4 | 40 | 64 | 0.93 | 1.00 | 172 | 52.0 |
| number 8 |  | 4 | 25 | 34 | 0.91 | 0.75 | 112 | 56.0 |
| arch |  | 4 | 10 | 11 | 1.00 | 1.00 | 50 | 30.0 |
| bridge |  | 4 | 22 | 28 | 1.00 | 1.00 | 120 | 52.0 |
| castle tower |  | 4 | 39 | 48 | 0.93 | 1.00 | 180 | 44.0 |
| fence |  | 4 | 14 | 26 | 1.00 | 1.00 | 82 | 38.0 |
| gate |  | 4 | 10 | 13 | 1.00 | 1.00 | 64 | 40.0 |
| ladder |  | 4 | 18 | 33 | 0.80 | 0.50 | 110 | 78.0 |
| lamp post |  | 4 | 9 | 41 | 1.00 | 1.00 | 72 | 34.0 |
| obelisk |  | 4 | 18 | 42 | 1.00 | 1.00 | 108 | 46.0 |
| podium |  | 4 | 9 | 12 | 0.98 | 1.00 | 72 | 20.0 |
| slide |  | 4 | 30 | 40 | 0.91 | 1.00 | 146 | 50.0 |
| wall with a door |  | 4 | 20 | 30 | 0.92 | 1.00 | 118 | 52.0 |
| cross |  | 4 | 13 | 45 | 0.93 | 1.00 | 76 | 44.0 |
| standing ring |  | 4 | 22 | 32 | 0.98 | 1.00 | 100 | 38.0 |
| doghouse | yes | 4 | 28 | 36 | 0.90 | 1.00 | 136 | 98.0 |
| throne | yes | 4 | 49 | 82 | 0.98 | 1.00 | 244 | 98.0 |
| letter K | yes | 4 | 20 | 26 | 0.72 | 0.25 | 92 | 56.0 |
| letter Q | yes | 4 | 46 | 72 | 0.94 | 1.00 | 196 | 50.0 |
| number 4 | yes | 4 | 32 | 50 | 0.80 | 0.50 | 140 | 72.0 |
| cactus | yes | 4 | 14 | 50 | 0.92 | 1.00 | 80 | 50.0 |
| pyramid | yes | 4 | 15 | 20 | 1.00 | 1.00 | 72 | 24.0 |
| staircase | yes | 4 | 22 | 42 | 0.98 | 1.00 | 102 | 28.0 |
| boat | yes | 4 | 45 | 70 | 0.91 | 1.00 | 228 | 52.0 |
| car | yes | 4 | 16 | 23 | 1.00 | 1.00 | 100 | 36.0 |
| plane | yes | 4 | 18 | 53 | 0.99 | 1.00 | 122 | 50.0 |
| rocket | yes | 4 | 15 | 36 | 1.00 | 1.00 | 96 | 34.0 |
| train | yes | 4 | 41 | 65 | 0.79 | 0.75 | 200 | 52.0 |
| truck | yes | 4 | 44 | 70 | 0.95 | 1.00 | 222 | 96.0 |
| use:contain |  | 4 | 40 | 60 | 1.00 | 1.00 | 182 | 66.0 |
| use:shelter |  | 4 | 42 | 82 | 1.00 | 1.00 | 180 | 16.0 |
| use:sit |  | 4 | 12 | 13 | 1.00 | 1.00 | 72 | 40.0 |
| use:span |  | 4 | 16 | 18 | 1.00 | 1.00 | 80 | 48.0 |
| use:table |  | 4 | 33 | 41 | 1.00 | 1.00 | 156 | 46.0 |

- named, training: 196 episodes, mean 0.95, passed 0.95
- named, held out: 56 episodes, mean 0.92, passed 0.89
- uses: 20 episodes, mean 1.00, passed 1.00
- all: 272 episodes, mean 0.95, passed 0.94

Weak spots, not diagnosed: the letter X (0 of 4, IoU 0.69) and A (1 of 4, 0.73); K and the number 4 (2 of 4); the ladder (2 of 4). Large objects take long: the sofa (62 voxels) passes at a median of step 202, the snowman at 156. A use passes as soon as its check holds, which for shelter is well before the object is finished (median step 16).

### Episode lengths

`build.episode_steps(task)`: the sum of the groups' time limits (12 + 4 per voxel each), so a plan
always runs to its end. Over the 272 episodes:

| voxels | n | passed | steps given (median) | steps to pass: median | 90th percentile | max |
| --- | --- | --- | --- | --- | --- | --- |
| 0 to 14 | 53 | 0.96 | 64 | 32.0 | 44.0 | 56 |
| 15 to 30 | 113 | 0.90 | 108 | 44.0 | 68.0 | 104 |
| 31 to 60 | 89 | 0.97 | 192 | 56.0 | 105 | 212 |
| 61 to 115 | 17 | 1.00 | 304 | 120.0 | 218 | 260 |


So `--steps 144` covers more than 90% of the passes of objects up to 60 voxels and `--steps 240` about 90% of the largest.

The store format has one episode length per store (`--steps`), so build episodes in a mixed store
run for the store's length and are kept only if they pass by then.

### Held out

1. Whole category: vehicles (car, truck, train, plane, boat, rocket). Their primitives (col, slab,
   box, beam, wall, ring, pyramid) are all trained elsewhere, so they test composing known parts
   under a new name.
1. Single names: throne, doghouse, cactus, and the suite's own held-out items: pyramid, staircase
   (the suite holds out stairs), letters K, Q, Z and numbers 4 and 7. Z and 7 cannot be built (an
   overhang of 4), so the drawable held-out set is 14 of 63 drawable names
   (22%).
1. `sample_build(split="objects")` draws only them, without a plan; `split="wording"` uses the
   held-out templates (`wordings.json` intents `build` and `build.use` have 30 or 32 training and 6
   held-out templates each). Haiku writes the open phrasings; there is no synonym bank for objects.

### Records

`canonical_record(task)`: `plain` "build a chair" (`build {o}`, the object with its article) or
"build something you can sit on" (`build something {f}`); `spec` `{"kind": "build", "object"}` or
`{"kind": "build", "use"}`, plus `"program"` (the parts text as built) and `"voxels"`, which
`haiku.IGNORED_KEYS` leaves out of the verifier's comparison and menu (an instruction does not pin
them down); `subgoals` the group sentences. `scene_summary` is unchanged. `SPEC_NOTES["build"]`
explains the kind to the verifier.

### Meshes: a loader and a proposed dataset (nothing downloaded)

`shapes.read_off` (ModelNet's format), `shapes.read_obj`, `shapes.voxelise` (dense surface
sampling, flood fill of the inside), `shapes.hollow`, `shapes.boxes_of` (any voxel set as a
program of boxes grouped by layer) and `shapes.mesh_shape`, which voxelises a mesh at the largest
resolution (12 down to 3) that fits the robots and keeps it only if it is buildable and climbable.

Proposal, to decide (sizes and licences recalled from memory, to verify before fetching):

1. **ModelNet40, aligned** (Princeton; Sedaghat et al.'s orientation-aligned version): 12,311 CAD
   meshes in 40 categories, OFF files, about 2 GB zipped (`modelnet.cs.princeton.edu`,
   `ModelNet40.zip`). Categories overlap the library (chair, table, stool, bench, sofa, bed,
   bookshelf, desk, stairs, tent, cup, bowl, car, airplane, plant, person, lamp, door ...). Terms:
   for academic research; the meshes come from 3D Warehouse with no stated licence, so not for
   redistribution. `read_off` reads it as is; no new dependency. The pilot I would run: 20 models
   per category, `mesh_shape(..., robots=115)`, report the share kept per category.
1. **Objaverse-LVIS** (Allen AI): the about 46,000 objects of Objaverse 1.0 annotated with 1,156
   LVIS categories; per-object licences, mostly CC-BY 4.0 (filter to CC-BY and CC0). Fetched with
   the `objaverse` package (`load_lvis_annotations`, `load_objects(uids)`, GLB files from the
   `allenai/objaverse` Hugging Face dataset), tens of GB for the subset. Reading GLB needs a mesh
   library (`trimesh`), which is not installed. This is the open-vocabulary step after ModelNet.

### Commands

    PYTHONPATH=src OMP_NUM_THREADS=4 python -m beebot.vla.voxel.build --per-object 4 --seed 777 --out runs/build/expert.jsonl
    PYTHONPATH=src OMP_NUM_THREADS=4 python -m beebot.vla.voxel.demos --out data/vla/voxel_demos_build --scenes 6000 --cube --rule switch_cube --counts log --compact --build 0.5 --steps 144

Build tasks alone (`--only build`, in `demos` and `haiku export`): no suite, coverage or object
tasks are drawn; the build generator keeps its own `(seed, 13)` seeding and each chunk its run
seed, so the tasks are exactly the build part of the full draw with the same seed, chunk and
`--build` share (tested). Indices then count build tasks only: a Haiku export joins to a store
only when both were made with `--only build` (`export --store` reads it from the store's config).

    PYTHONPATH=src OMP_NUM_THREADS=4 python -m beebot.vla.voxel.demos --out data/vla/voxel_demos_build_only --scenes 6000 --cube --rule switch_cube --counts log --compact --build 1.0 --only build --steps 144
    PYTHONPATH=src python -m beebot.vla.voxel.haiku export --name build-s0 --seed 0 --items 50 --cube --rule switch_cube --counts log --build 1.0 --only build

Tests: `tests/test_vla_voxel_build.py`.

## Compact store

A dense store spends 0.6 MB on an episode of 72 steps (145 grids of 4096 bytes) to hold 6 to
115 robots. The compact format (`play_store.py`) stores each frame's robots instead:

| file | content |
| --- | --- |
| `compact.json` | `{"dims": [D, H, W], "voxel_dtype": "uint16", "stay": 26}`; its presence marks the format |
| `offsets.npy` | int64 (episodes, steps + 2): frame (e, t) is entries `offsets[e, t]` to `offsets[e, t + 1]` of the two files below |
| `voxels.bin` | raw uint16: the flat voxel index `(z H + y) W + x` of every robot of every frame, ascending within a frame (uint32 for grids over 65536 voxels) |
| `codes.bin` | raw int8, parallel to `voxels.bin`: that robot's action code; an episode's last frame has no action and holds STAY |

`small.pt`, `labels.jsonl` and `attempts.jsonl` are the same in both formats.

1. **Exact.** The format needs a mass of 0 or 1 and STAY wherever no robot is; the writer
   refuses anything else, so the dense grids come back bit for bit.
1. **Reading.** `play_store.load` detects `compact.json` and returns `mass` and `actions` as
   `CompactGrids`: `store["mass"][e, t]` is the same (D, H, W) tensor as from a dense store,
   slices of episodes and steps work, and `shape` and `dtype` are those of the dense array. The
   three files are memory-mapped and only the frames asked for are read. `lm.world_at`,
   `lm.batch_of`, `facts` and `render` read either format unchanged. Code that calls tensor
   methods on the whole field (`store["mass"].flatten(2)`) needs `store["mass"][:]` first.
1. **Writing.** `demos --compact`, or for an existing store (the source is only read, and only
   its rows in use are written):

   ```
   uv run python -m beebot.vla.voxel.play_store --convert data/vla/voxel_demos_cube_sample --out data/vla/voxel_demos_cube_sample_compact
   ```

Measured on the 875-episode cube sample (M4 Pro, 3 threads, both stores in the page cache):

| | dense | compact |
| --- | --- | --- |
| grids, bytes per episode | 593,920 in use (1,221,779 apparent: 1800 rows allocated) | 15,083 (voxels 9,661, codes 4,830, offsets 592) |
| with `small.pt` (5,122 per episode) | 599,042 | 20,205 |
| 30,000 episodes | 17.8 GB (36.7 GB apparent) | 0.45 GB, 0.61 GB with `small.pt` |
| `lm.batch_of`, batches of 16 on the CPU | 4,700 rows/s on a quiet machine; 900 to 1,100 with other jobs running | 3,400 rows/s; 600 to 1,200 with other jobs running |
| the two grids alone | 19,000 to 58,000 rows/s | 10,000 to 17,000 rows/s |

The trainer draws 30 to 60 batches of 16 a second (480 to 960 rows); most of a `batch_of` call is `observe`, not the read, and the same rows gave the same checksum from both stores. A
compact store of 30,000 episodes fits in memory many times over; the dense one does not, and
every random row would then be a disk read.

## Haiku instructions and thoughts (`haiku.py`)

Training text written by Claude Haiku for the expert's demonstrations: per task three
instructions on a ladder of indirection (`close`, `indirect`, `casual`), each with a first
thought ("you asked: ... i see ... <resolved targets>. first: <first sub-goal>."), one line per
later sub-goal and a final "done."; about 10% of tasks also get an instruction that cannot or
should not be followed (`refuse` or `clarify`, with the reply). A second call sees only an
instruction and the scene and writes the task back as JSON; a pair is kept only if that equals the
task's spec. The module docstring lists the commands and files; in short:

    PYTHONPATH=src python -m beebot.vla.voxel.haiku pilot                 # 50 items end to end, review.txt
    PYTHONPATH=src python -m beebot.vla.voxel.haiku export --name g1 --items 5000 [--store DIR]
    PYTHONPATH=src python -m beebot.vla.voxel.haiku generate --name g1    # resumable
    PYTHONPATH=src python -m beebot.vla.voxel.haiku verify --name g1      # keep rates in report.json
    PYTHONPATH=src python -m beebot.vla.voxel.haiku testset               # frozen, held-out words only

Every command takes `--dry-run`: a fake provider, the exact prompts and the parsed fake outputs
printed, nothing written under `data/vla/haiku`.

1. **Join.** Items are keyed by (seed, index): index n is the n-th task `demos.draws` yields for
   the seed, which is line n of a store's `attempts.jsonl` written with the same seed and draw
   options (`--chunk`, `--robots`, `--counts`, `--fit`, `--rule`, `--apart`); that line's `e` is
   the store row. `export --store DIR` reads the options from the store, exports only its kept
   episodes and checks every family and kind; `Labels(name).by_episode(DIR)` returns
   `{e: choices}`.
1. **Checks before verification.** No held-out word, category, frame or language in training
   text (wordings.json splits); a thought names only colours and terrain the scene has (or the
   task's sentence names), names each sub-goal's colours in order, and stays under 45 words
   besides the quoted instruction; one switch line per later sub-goal.
1. **Test set.** Its own seed, a separately worded prompt that offers only held-out expressions
   for the task's words and keeps the plain and training words out, verified the same way, in
   `data/vla/haiku/test` with a `HELD_OUT` marker. `Labels` refuses it and asserts no training
   pair shares an episode or an instruction with it.
1. **Records.** The pipeline reads only `{family, kind, plain, spec, subgoals}` records and scene
   summaries. `local_canonical_record` and `local_scene_summary` stand in until the suite has its
   own `canonical_record` and `scene_summary`; one line switches. A suite task whose colour is
   not in the scene (`eat:absent`) has the spec `{"kind": "refuse"}`.

## Think-then-act training examples (`think_data.py`)

`ThinkData(stores, haiku_dirs, tokenizer)` turns demo stores and verified Haiku labels into one
example per sampled step: the scene at step t (the voxel observation, and images of it from a
pluggable `views(obs, rng)`, rendered at sample time), the text before the reply (memory, then the
user turn with `<IMAGES>` and the instruction in force), the reply (a thought in Qwen3.5's
`<think>` tags, empty on most steps, then `<tool_call>`, or a text answer for a refusal) and the
expert's action codes. The module docstring has the interface and the exact
text layout; `python -m beebot.vla.voxel.think_data --stores ... --haiku ... --show` prints the
counts, one rendered example of each kind and the sampling speed.

1. **Thought steps.** The first step ("you asked: ... i see ... first: ..."), each instruction
   change of a switch task (the new part's first thought), each sub-goal switch (a short line) and
   "done." where the expert records an eat or push task as finished. Switches are read from
   `small.pt["subgoals"]` (`Expert.subgoals`: an entry whenever the segment index or its goal
   changes; a switch is a rise of the segment index within one instruction), so no store needs
   rewriting and no draw changes.
1. **Memory.** An earlier assistant turn with the latest first thought before step t and the
   latest line after it: fixed size, never the whole history. On the step an instruction changes
   it still holds the old instruction's thought.
1. **Labels.** Haiku choices where verified pairs exist (one rung drawn per sample; a switch
   episode only when every part is verified), otherwise the store's template instruction with a
   templated thought from `canonical_record` and `scene_summary` at the part's first step. The join
   is by (seed, index) through `attempts.jsonl`; a label directory drawn with other options than
   the store is an error, and so is a label that does not belong to its episode: its family, kind
   and spec must be the store's record of the instruction it labels, and its exported item's
   family, kind and robot count the attempt's (`mismatch="drop"` drops such labels and counts
   them instead).
1. **What images show.** The policy sees images, so `visible` cuts every thought to what a viewer
   can read off them: robot counts, coordinates and measured heights go unless the instruction or
   the task states the number; colours, kinds, sizes and coarse places stay. On the 2026-11-10
   labels it changed 8,402 of 31,184 Haiku thoughts and 261 of 20,761 template thoughts.
1. **The user's frame.** The camera can stand anywhere round the arena (the views report its
   azimuth), and the user stands behind it, to its left, to its right or opposite it, said in the
   user turn ("(i'm standing to the left of the camera)"). Every direction word in the instruction,
   thoughts, memory and answers is turned into that user's frame (`in_user_frame`, quarter turns);
   wording that cannot be turned reliably (near/far side, north, toward me, ...: about 1.6% of the
   Haiku choices) is used only when the user's frame is the world's. Observations and action codes
   stay in the world's frame.
1. **Known problem.** In `voxel_full` s0, s1, s2 and s7 the stores' object-task draws part from
   the Haiku export at one index per seed (s0 715, s1 816, s2 783, s7 2745); `demos.draws` alone
   reproduces the export, `demos.write` does not. 894 of 30,905 episodes have labels that fail the
   check; load these stores with `mismatch="drop"` until the writer is fixed.
1. **Sampling.** `thought_share` (default 1/3) of act examples at a thought step; `refusal_share`
   (default 0.05) refusal examples; `haiku_share` None draws episodes uniformly. 5% of episodes
   (a hash of (seed, index)) are held out for `validation(n, seed)`.

## Views and randomization

`views.py` renders a voxel observation as camera images for the vision-language policy. The
policy (Qwen3.5-4B) sees images and acts per voxel. Later it must work from a phone camera
looking at real cubes on a steel sheet, so it never sees one lighting, colour scheme, camera or
render style: every image is drawn with visual domain randomization, as in robot sim-to-real
work.

```python
from beebot.vla.voxel.views import scene_views, sample_look, ViewSpec
images, pose = scene_views(obs, rng, ViewSpec())          # obs (C, D, H, W), 6 to 10 channels
look = sample_look(rng, spec)                              # one episode: fixed camera, lights, materials
images, pose = scene_views(obs_t, rng, spec, look=look)    # every frame of that episode
```

`images` holds one PIL image per camera of `spec.cameras` (default: the oblique perspective view,
then the top-down view), each `spec.size` square (default 256). `ViewSpec.clean()` turns every
randomization off, for evaluation and debugging.

**Rendering.** The renderer is numpy only. Every exposed voxel face that faces the camera is
splatted over its projected pixel box. The exact ray-plane hit of each pixel decides coverage,
depth and the face's (u, v), and a z-buffer (`np.minimum.at`) keeps the nearest face. The floor
is one plane hit per pixel. Textures, lights, a shadow map (the same splatter, seen from the
main light), outlines and image effects follow. Both cameras are real projections (`Camera`:
position, look-at, field of view, roll, principal point).

**Frame and pose.** The world frame has x to the right, y to the front and z up, as seen by a
person at the front (`suite.FLOOR_REGIONS`). `pose["azimuth"]` is the oblique camera's place in
degrees, in the world frame: 0 at the front (+y), 90 at the right (+x), 180 at the back, 270 at
the left. With camera randomization on, the azimuth is drawn uniformly within 30 degrees of a
side. It is therefore never within 15 degrees of a diagonal (45, 135, 225, 315), where a person
cannot tell front from right. This matters because the data side turns direction words by
quarter turns from this azimuth. `pose["side"]` names the nearest side. The top-down view turns
with the same azimuth, plus up to 6 degrees of its own jitter: its image "up" points away from
the oblique camera. `pose["views"]` lists each camera's projection, azimuth, elevation, field
of view, position and style. `test_world_frame_of_both_cameras` pins all of this.

**Registration.** Faint grid lines are always drawn on every arena cell boundary and on the
arena border. Four corner markers are always drawn just outside the arena's corners. Only how
they look is randomized. The lines are lighter or darker than the sheet, at opacity 0.15-0.5
and width 0.04-0.1 cells. A marker is an L bracket, square, dot or cross of 0.4-1 cells. So
pixels register to voxels from any viewpoint.

**Clean mode.**

1. Cameras: the oblique camera sits at azimuth 0 and elevation 45 degrees, with a 50 degree
   field of view. It is framed on the arena and 4 voxels above it. The top view is
   orthographic.
1. Lighting: the lit style, with one light from the front left at 60 degrees, and shadows.
1. Sheet: dark slate (`CLEAN_FLOOR`), at least 30 CIE Lab units from every object colour. The
   zero-shot probe missed white cheeses on the old light floor.
1. Robots and terrain: robots are mid-grey cubes with dark edge lines. Terrain is sandy.
1. Objects: exact colours, no noise.
1. Top view: faces are shaded by height (darker low, full brightness at 6 voxels and above) and
   every height step is outlined, so stacks and stepped pyramids read from above.
1. Food: food voxels show darker round holes in every mode, so a cheese reads apart from a
   passive block. Food is channel 6, or every object of a 6-channel observation.

**Randomizations.** Each group has a strength in [0, 1]: its own `ViewSpec` field, or None to
use the global `spec.strength` (`DEFAULT_STRENGTH`, 0.75). Strength 0 gives the clean value. The
ranges below are at strength s and shrink linearly to the clean value; a probability p means
p * s. Any item can be turned off by name: `ViewSpec(off={"jpeg"})`.

| group | item | range at strength s (clean value) |
| --- | --- | --- |
| camera | azimuth | uniform within 30 deg of a side, not scaled by s (0) |
| | elevation | 45 +- 20 s deg (45) |
| | fov | 50 +- 15 s deg vertical, distance refit to frame the arena (50) |
| | distance | x U(1 - 0.05 s, 1 + 0.4 s) of the fitted distance (1) |
| | look_at | target +-1.5 s cells in x, y, +-0.5 s in z (0) |
| | roll | +-6 s deg (0) |
| | aspect | fx/fy +-4 s %, principal point +-3 s % (0) |
| | distortion | p 0.6: radial k1 in U(-0.04, 0.22) s, mostly barrel, refit so the corners stay (0) |
| | crop | zoom in x U(1, 1 + 0.1 s), shifted anywhere inside the render (1) |
| | resolution | rendered at U(1 - 0.5 s, 1) x size, then upsampled to size, as a low-resolution sensor (1) |
| | top_jitter | top view: +-6 s deg rotation, up to 8 s deg tilt (0) |
| | top_perspective | p 0.5: the top view becomes a perspective camera high above, fov 20-45 deg (orthographic) |
| lighting | lights | p 0.5 each for a second and a third light of 0.2-0.6 (one light) |
| | light_direction | azimuth uniform, elevation 60 +- 30 s deg (front left, 60) |
| | intensity | 0.75 x U(1 - 0.35 s, 1 + 0.35 s) (0.75) |
| | ambient | 0.4 +- 0.22 s, tinted toward the light or a cool sky (0.4, white) |
| | temperature | 5500 +- 2700 s K per light (5500 K) |
| | shadows | p 0.35 off; 1 or 4 PCF taps (on, 4) |
| | specular | Blinn-Phong. Robots ks U(0, 0.55 s), shininess 8-64; steel floors ks 0.1-0.5; objects ks up to 0.15 s (robots 0.15) |
| style | style | p 0.3 flat (no lighting, a fixed shade per face direction); p 0.2 toon (three light bands, dark outlines); otherwise lit. Outlines p 0.35 (lit, outlined) |
| materials | floor | steel, brushed metal, wood, paper (p 0.3 graph lines), fractal noise or rotated tiles, at low saturation. Its mean is at least 20 Lab units from every object colour and 15 from the robots (dark slate) |
| | table | the surface past the sheet: the same types, colours at least 25 Lab units from every named colour, fading into the background. Sheet margin 0.6-3 cells (light grey, 1.2) |
| | terrain | sand, wood, stone, concrete, cardboard or foam. Sides 0.75-1 of the top, faint cell edges, at least 15 Lab units from every object colour (sand) |
| | robots | base grey 0.12-0.85 with saturation up to 0.08, at least 15 Lab units from every object colour. Edge lines 0.03-0.12 of a face, darker or lighter; p 0.5 face markings (dot, ring, cross, square, stripe); per-cube brightness +-6 s % (mid grey, dark edges) |
| | object_noise | object surface texture amplitude up to 0.12 s (0) |
| | grid_lines, markers | appearance only, see Registration (light lines at 0.35, white L brackets) |
| colour | colour_jitter | per named colour of objects and zones: hue, saturation and value within `COLOUR_JITTER[name]` x s (exact) |
| distractors | clutter | Poisson(4 s) things on the table outside the sheet, never cube-like: papers (1-4 cells, 0.02-0.08 high), books (0.2-0.6 high), cable- or pen-like bars (2-7 long, 0.15-0.45 thick). They cast shadows (none) |
| | stickers | Poisson(5 s) flat ellipses, rectangles and stripes on the table (none) |
| | background | procedural backdrop where rays miss the floor: a gradient plus fractal blobs (grey gradient) |
| | hand_shadow | per image, p 0.3: a soft hand shadow (palm and fingers) from an edge, darkening by 0.25-0.5 (none) |
| | cable_shadow | per image, p 0.3: a soft curved cable shadow, darkening by 0.25-0.5 (none) |
| corruptions | gaussian_noise | p 0.8: sigma U(0, 0.035 s) |
| | shot_noise | p 0.6: variance k x intensity, k U(0, 0.02 s) |
| | motion_blur | p 0.25: a line of 2 to 2 + 6 s pixels at any angle |
| | defocus | p 0.3: Gaussian radius 0.3 to 0.3 + 1.3 s pixels |
| | jpeg | p 0.6: quality U(95 - 65 s, 95) |
| | vignetting | p 0.5: corners darkened by up to 0.45 s |
| | exposure | p 0.8: x 2^U(-0.6 s, 0.6 s) |
| | white_balance | p 0.6: per-channel gains 1 +- 0.1 s |
| | gamma | p 0.6: 1 +- 0.25 s |
| | chromatic | p 0.4: red and blue at radial scales 1 +- U(0, 0.006 s) |
| | cutout | p 0.4: 1-3 patches of 4-12% of the side, noise or a flat colour as unnameable as the distractors, placed only where they cover no object or zone pixel, so a target is never erased from any view |

Nothing outside the arena may look like an object an instruction can name. So every clutter,
sticker, table, backdrop and flat cutout colour has saturation of at most 0.3-0.4 and lies at least
`DISTRACTOR_GAP` (25) Lab units from every named colour (`test_distractors_are_unnameable`).

Camera, lighting, style, materials, colour and clutter are drawn once per look, which is one
episode. Hand and cable shadows and the corruptions are drawn for every image.

**Colour names survive jitter.** `COLOUR_JITTER` bounds the hue shift, saturation and value of
each named colour. White gets a small tint and a value range instead.
`test_colour_jitter_keeps_names` checks, for every colour, a 5 x 5 x 5 grid over its bounds and
400 draws: the nearest named colour (CIE Lab, `nearest_colour_name`) stays its own. Red and
green have the tightest bounds, because a dark, desaturated red lies nearest to pink.

**State noise** (`perturb_state(obs, rng, StateNoise())`) is separate and off by default:
nothing calls it. It corrupts only the robot channel, as imperfect state estimation on real
hardware would:

1. Each robot is dropped with probability 0.02 (`drop`).
1. Each robot is moved to a free face neighbour with probability 0.02 (`shift`).
1. For each robot, a phantom robot is added next to something solid with probability 0.01 (`phantom`).

**Readability and speed** (`scripts/vla_view_probe.py`):

1. `--sheets` writes contact sheets of the zero-shot probe's items (colours, letters, shapes),
   clean and at strengths 0.25, 0.5, 0.75 and 1: `runs/views/sheet_<setting>_<probe>.jpg`.
1. `--qwen` asks Qwen3.5-4B the probe questions zero-shot under each setting:
   `runs/views/qwen.txt`.
1. `--speed` times rendering: `runs/views/speed.txt`.

Zero-shot Qwen3.5-4B (bf16, MPS, greedy) on both views. There are 30 items per probe and
setting, so one item is 0.033 and a standard error is about 0.08. Run on 2026-11-10
(`runs/views/qwen.txt`, items in `runs/views/qwen.jsonl`):

| setting | strength | colours | letters | shapes | mean |
| --- | --- | --- | --- | --- | --- |
| clean | 0 | 0.90 | 0.63 | 0.63 | 0.72 |
| s25 | 0.25 | 0.83 | 0.33 | 0.67 | 0.61 |
| medium | 0.5 | 0.77 | 0.33 | 0.73 | 0.61 |
| s75 | 0.75 | 0.77 | 0.50 | 0.77 | 0.68 |
| strong | 1 | 0.70 | 0.33 | 0.73 | 0.59 |

1. **Shapes do not depend on strength.** They stay at 0.63-0.77 across all settings. The old
   isometric renderer scored 0.68.
1. **Colours fall slowly.** They read 0.90 clean (the old renderer: 0.85; white on the light
   floor is no longer missed), then 0.77 at 0.5 and 0.75, and 0.70 at 1. Of the 31 colour
   errors, 22 only add a colour that is not in the scene (mostly yellow, green or pink): the
   model names terrain, table or noise colours as cheeses. The other 9 miss a colour. In 4 an
   orange cheese is read as yellow (at 0.5 and 1), 4 miss a white cheese (from 0.25 up), and 1
   misses purple.
1. **Letters depend on the camera's side, not on strength.** The azimuth is drawn from all four
   sides at any strength above 0, so a letter on the floor is usually seen turned or upside
   down. Over the four randomized settings, front views score 16/26 (0.62, like clean). Back
   views score 15/42 and left and right views 14/52.
1. **Default.** The default is 0.75, the strongest strength at which colours and shapes stay at
   the 0.5 level. At strength 1, colours drop a further 0.07. Zero-shot readability is a floor:
   training on these views teaches the model what counts as an object.

## Think-then-act scorecard (`rollout.py`, `scorecard.py`)

The policy of `think.py` played closed loop, scored by the suite's own checkers. The module
docstrings hold the details; in short:

    PYTHONPATH=src python -m beebot.vla.voxel.scorecard --checkpoint runs/X/policy_step800.pt --device mps --out runs/X/card_step800
    PYTHONPATH=src python -m beebot.vla.voxel.scorecard --arms expert --out runs/card_expert       # no model, CPU
    PYTHONPATH=src python -m beebot.vla.voxel.scorecard compare runs/A/card/episodes.jsonl runs/B/card/episodes.jsonl

1. **A step.** Render the views (`views.scene_views`, the episode's look),
   build the context (instruction in force said in the user's frame, where the user stands, memory
   = the latest first thought and the latest switch line after it), `think_then_act` for every
   live episode of the batch, argmax (or `--sample`) per robot, `world.step` under switch_cube
   with failures 0.10 (0.23 up and over). A switch task's second instruction comes at its step; a
   reply in words ends the episode with no moves. 72 steps, 144 for build.
1. **Arms.** policy; shuffled (another episode's instruction, same scene and seed: the gap says
   whether the policy acts on the instruction); plain (the canonical sentence); oracle (the
   canonical first thought fed on each instruction's first step); expert (the suite's expert, the
   ceiling; it stays still on held-out compositions, which have no plan); still (no robot moves:
   the floor; a switch to "stay" or "eat no cheese" passes so, so read every arm and gate b against it).
1. **Splits.** test (48 per family, training templates), wording (held-out templates), held (the
   held-out compositions of suite, objects and build), testset (the frozen Haiku test set, scenes
   redrawn from its config and checked against each item's record).
1. **Thought checks.** Programmatic, against `canonical_record`: the thought parses ("you asked:
   ... first: ..."), names the first sub-goal's key facts (colours, areas, numbers, shapes, letters,
   object names; the kind of move when there are none) and no wrong colour, and resolves the task
   (every sub-goal's key facts in the reasoning). On 25,378 verified Haiku training labels (run-s0
   to run-s3 and run-s0-later): parse 1.00, first 0.97, resolves 0.97; parallel is the lowest
   (0.84: labels often end "first: split into two groups" without the jobs).
1. **Gates.** a) on the test split, thought parse and first-sub-goal match of the policy arm at
   0.9 or better, with the shuffled gap and the pass share of episodes with a right against a
   wrong first thought; b) per held-out split, pass(oracle thought) / pass(plain instruction) at
   0.9 or better.
1. **Seeds.** Scenes from `default_rng([700001, crc32(split), crc32(family)])`, disjoint from the
   training stores (seeds 0..7 and their (k, 7), (k, 11), (k, 13) streams), the Haiku test set
   (900001) and the menu (4242). An episode's own seed drives its world noise, failures, action
   sampling and views in every arm, so arms and checkpoints are paired episode by episode.
1. **Live numbers for ops.** While it runs, `rollout_success` lines (arm model, oracle_thought,
   expert, plain, shuffled, still; per group and per arm) go to `--rollout-log`
   (`logs/think_act_loop.jsonl`), and `scorecard3d_mean` per arm and split, then
   `scorecard3d_noise` (the standard deviation of the mean across `--repeats` cards on fresh scenes
   and seeds, or the macro standard error of a single card) to `--log` (`logs/scorecard3d.jsonl`).
1. **Training's text and views.** Contexts come from `think_data.context` (a test checks that the
   rollout's context at every step of a training episode equals ThinkData's), oracle thoughts from
   `think_data.template_thought` cut by `visible`, views from `views.scene_views` with one
   `sample_look` per episode (`--views stand_in` for think_data's isometric image). The user's
   place is drawn per episode as think_data draws it and every instruction is said in that frame
   (`in_user_frame`); one whose direction words cannot be turned keeps the world's frame.

## Left out, and extension points

1. **Simple machines** (a lever: a rigid bar on a pivot) are not implemented. The natural place
   is a new object type next to cheeses in `VoxelWorld`:
   1. A bar of length L lies along one horizontal axis, with a pivot voxel under its middle.
   1. The press from robots on each arm gives a torque, using the same press accounting as
      `move_robots`.
   1. When the net torque reaches a threshold, the bar tilts one notch. Robots standing on the
      raised arm are lifted, with the same all-or-none shove rule as `push_cheeses`.
   1. A fact `("tilted", side)` would follow the pattern in `facts.py`.
1. Zones and passive blocks exist now (section "Objects"); pressure plates and gates do not.
1. Gravity has no load limits and no notion of stability beyond face connectivity.
1. The controller is greedy with no path planning. Robots can stall behind terrain, and bridge
   segments succeed only part of the time.
