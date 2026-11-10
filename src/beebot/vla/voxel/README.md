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
