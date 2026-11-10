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
| `demos.py` | expert demonstrations as a store, also for pivoting cubes |
| `free_cube.py` | the planner for cubes that cannot leave a bond, and the comparison of rules and experts |
| `facts.py` | window facts as hashable atoms |
| `render.py` | isometric PNG/GIF |

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
1. **Observation.** `observe(world)` returns (B, 6, D, H, W) with channels robot, wall, cheese
   mask and cheese RGB.
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
The axis words are up and down, forward and back (y), and left and right (x). Examples: `"up"`,
`"forward left"`, `"down back right"`.

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
| the smaller / larger group (two groups), the smallest / largest (three) | `{"select": "group", "by": "size", "which": ...}` | "the smaller group" | added |
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
| climb onto a plateau | `climb` | "climb onto the plateau on the left" | added (trench scenes) |
| climb on top of a cheese | `perch` | "3 of you: climb on top of the red cheese; ..." | added with a count selector (four voxels on top) |
| move a few steps in a direction | `move` | "move three steps toward the left" | added |
| follow / chase a cheese without eating it | `follow` | "follow the red cheese without eating it" | added (the cheese flees, as a runner) |
| catch / eat cheeses by colour logic | `eat` | "eat the red cheese", "eat every cheese except ..." | exists (logic, runner); eating by a selected group added |
| surround a cheese | `surround` | "surround the red cheese" | exists (parallel) |
| push a cheese to a region or into the trench | `herd_region`, `herd_trench` | "push the red cheese to the left of the floor without eating it" | exists |
| letters, shapes | `mask` | "write A flat on the floor", "form a hollow cube" | exists |
| tower, bridge, fill, stairs | `tower`, `bridge`, `fill`, `stairs` | "build a tower 5 robots tall" | exists; a tower by a selected group added |
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
1. `spec`: the checker's spec, JSON-safe (sets as sorted lists, arrays left out), with the
   canonical slots under `"slots"`. A switch task's spec is `{"kind": "switch", "parts": [{"t":
   first step, "record": record}, ...]}`.
1. `subgoals`: ordered plain sub-goal sentences: the cheeses of an eat task in order, the two
   jobs of a parallel task, the selected robots' job then "everyone else: stay where you are",
   the instructions of a switch task in turn.

`scene_summary(world, i)` returns `{"robots", "gravity", "terrain", "cheeses", "groups",
"scattered", "towers", "on_top"}`: terrain `{"kind": "flat"}`, `{"kind": "trench", "runs",
"region", "width", "depth", "plateaus"}` or `{"kind": "hills", "highest", "regions"}`; each cheese
`{"colour", "region", "at": [z, y, x], "on"}`; groups (26-connected, at least 2 robots and a tenth
of the swarm, largest first) `{"size", "region", "centre": [y, x], "height"}`; towers (vertical
runs of 3 or more) `{"height", "region"}`; `on_top` counts robots standing on robots. Regions are
the nine floor regions of `FLOOR_REGIONS`.

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

## Left out, and extension points

1. **Simple machines** (a lever: a rigid bar on a pivot) are not implemented. The natural place
   is a new object type next to cheeses in `VoxelWorld`:
   1. A bar of length L lies along one horizontal axis, with a pivot voxel under its middle.
   1. The press from robots on each arm gives a torque, using the same press accounting as
      `move_robots`.
   1. When the net torque reaches a threshold, the bar tilts one notch. Robots standing on the
      raised arm are lifted, with the same all-or-none shove rule as `push_cheeses`.
   1. A fact `("tilted", side)` would follow the pattern in `facts.py`.
1. There are no boxes or zones (the 2D `box_rect`), no captions or text instructions, and no
   policy, training or evaluation hookup. The store mirrors the 2D one, so `make_windows` or
   phase-based windows can be built the same way.
1. Gravity has no load limits and no notion of stability beyond face connectivity.
1. The controller is greedy with no path planning. Robots can stall behind terrain, and bridge
   segments succeed only part of the time.
