# Hardware: what should a module be?

The question: what physical module should the swarm be made of? Two candidates:

- **Drones.** Free 3D motion.
- **Interlocking lattice modules.** They can bear load, exert force and build structures.

Rather than argue it, the answer is searched for. Every candidate design is scored by one number
in simulation: how much of the generalist task mix a swarm of those modules gets done.

Code: `src/beebot/hardware/`. Tests: `uv run pytest -q tests/test_hardware.py`.

## The metric: work score

**Work score** is the mean over a fixed sample of the voxel generalist suite's training tasks
(`vla/voxel/suite.py`) of each episode's score. An episode that eats a cheese it was told to leave
scores 0. The score is in [0, 1], and higher is better.

The suite has six families:

| family | what the swarm does |
| --- | --- |
| formation | forms shapes and letters |
| construction | builds towers and bridges, fills a trench |
| logic | eats by colour |
| parallel | splits into two groups doing two jobs |
| herd | pushes cheeses |
| runner | catches fleeing cheeses |

The run conditions:

1. **Driver.** The suite's expert drives the swarm, with the same plans for every design. Its
   controller only chooses among moves the design can actually make.
1. **Physics.** The world runs under the design's physics, with gravity always on.
1. **Plan change.** Terrain does not stick to modules, so a bridge flush with the trench rims
   cannot hang from them. The expert therefore also lays a deck one layer up, from two voxels
   before the trench to two after. Every design gets this plan, and so does the permissive
   voxel-world baseline.

So the score answers this: given a competent operator, how much of the task mix can these modules
physically carry out? It is a hardware number, not a policy number. A learned policy can only do
worse than the expert, and the ranking between designs is what matters.

Every design is scored on the same tasks (common random numbers), so differences between designs
are paired comparisons.

## Design space

A module is a cube of side L. Its volume is split into shares:

| share | what | fixed or searched |
| --- | --- | --- |
| shell and electronics | 0.15 at 1200 kg/m^3 | fixed |
| actuator battery | 0.10 at 1800 kg/m^3 | fixed |
| connectors | six face magnets, permanent (NdFeB) or electropermanent | searched |
| actuator | geared motor that pivots the module about an edge | searched |
| flight system | motors and battery for rotors inside the cube's footprint, plus 0.15 of ducted rotor volume | searched |

Rotors stay inside the footprint so that modules still pack face to face. A drone with rotors
sticking out cannot interlock.

Each design is reduced to five numbers in the module's own units. These are all the simulator
sees.

| symbol | meaning | how it is derived |
| --- | --- | --- |
| s | face connector pull-off force, in module weights | magnet holding pressure times magnet face area; area from the connector share, with magnets 0.25 as thick as they are wide |
| a | actuator torque, in m g L | torque density times actuator volume |
| r | thrust, in module weights | momentum theory: T = (2 rho A)^(1/3) (eta P)^(2/3), rotor disc A = 0.6 L^2, P = specific power times flight mass |
| c | weight of the 2 x 2 x 2 object the swarm pushes and carries (density 1000 kg/m^3), in module weights | object mass over module mass |
| switchable | whether connectors can be turned off (electropermanent) | connector type |

Square-cube scaling falls out of these: s and a scale as 1/L, and r as L^(-1/3).

### Constants and where they come from

These are first-order. Each should be replaced by a measurement from a prototype.

| constant | value | source or reasoning |
| --- | --- | --- |
| magnet holding pressure | 90 kPa | K&J Magnetics pull data for N42: a thin plate (thickness/width 0.125) holds about 87 kPa, a cube about 320 kPa. Taken at thickness/width 0.25 (about 150 kPa), then derated by 0.6 for gap and misalignment. |
| electropermanent pressure | 0.5 x permanent | Only the switchable (AlNiCo) half of an electropermanent magnet can be turned off. The coil raises density to 8000 kg/m^3. |
| actuator torque density | 8e4 N m / m^3, density 3000 | Pololu micro metal gearmotor, HP 298:1: gearbox limit about 0.25 N m from about 3.1 cm^3 and 9.5 g. Gearbox-limited torque scales with volume. |
| flight specific power | 1000 W/kg of motors and battery; efficiency 0.3 (motor 0.6 x figure of merit 0.5) | Crazyflie 2.1: about 16 g of motors and battery give about 60 g of maximum thrust. This model gives 73 g for its geometry. |
| friction | mu = 0.5 everywhere | |

## Physics per design (`physics.py`, `structure.py`)

1. **Moves.**
   1. A module that cannot fly (r < 1) moves only along surfaces. Its destination must touch a
      module, wall, cheese or the floor by a face, and it cannot move along all three axes at
      once.
   1. A move pivots the module about an edge. That needs torque of 0.5 m g L on the level and 1.0
      going up.
   1. With permanent magnets, the module must also pry open every bond it leaves, at 0.5 s m g L
      per bond.
   1. A module that flies moves into any free voxel. It leaves bonds by pulling them off with
      spare thrust or by prying them.
   1. Electropermanent connectors switch off, so leaving bonds costs nothing.
1. **Pushing.** A module pressing on a cheese pushes with min(2a, mu (1 + b)) weights, where b is
   the number of modules lined up behind it (up to 4), bracing it with their traction. A flying
   module may push with tilted thrust instead, sqrt(r^2 - 1). The cheese slides when the press
   reaches mu c.
1. **Statics.** Each step solves the structure as rigid cubes joined at face contacts:
   1. Each face corner has a compression-only spring.
   1. Module-to-module faces also have a centre magnet spring that works in tension only, up to s.
   1. Tangential springs at the corners make shear fail past mu times the clamping force.
   1. An active-set iteration releases corners in tension. The worst failing connector is then
      removed and the structure solved again.
   1. Modules left as a mechanism fall one voxel per step.
   1. A flying module also holds its position with up to r weights of thrust.
   1. A cheese resting on modules loads them with its weight.
   1. Checks against hand calculations, in `tests/test_hardware.py`:
      1. A row tips off a ledge exactly when the overhang's moment exceeds the counterweight's.
      1. The root connector of an n-module overhang fails when s < n^2.

## Known simplifications

1. **Time.** Every design moves one voxel per step. Actuator speed, energy and flight endurance
   are not scored. Flight endurance would hurt drones holding a structure for long.
1. **Moves.** The pivot model only asks for a surface at the destination. It does not check the
   swept path for collisions.
1. **Terrain sliding.** Terrain contacts are not checked for sliding.
1. **Lattice.** Only cubes on a cubic lattice are covered. Other space-filling shapes (rhombic
   dodecahedron, truncated octahedron) are not yet compared.
1. **Driver.** The expert is a greedy target-filler. A design that needs cleverer plans, such as
   scaffolding for floating shapes, is scored by what the greedy expert manages.

## Results

### Round 1: grid (2026-11-08)

**Grid.** 258 designs over these axes:

| axis | values |
| --- | --- |
| side | 1, 3, 10 cm |
| connector | permanent, electropermanent |
| connector share | 0 to 0.25 |
| actuator share | 0.01 to 0.4 |
| flight share | 0, 0.2, 0.35, 0.5 |

Two kinds of design were skipped:

1. Designs whose flight system cannot lift them.
1. Most actuator variants of drones, which barely use their actuator.

Each design was scored on the same 96 tasks (seed 0): `runs/hardware/grid2.jsonl`,
`uv run python -m beebot.hardware.analyze runs/hardware/grid2.jsonl`.

**Validation.** The best design of every (size, locomotion, connector) group, the top three
overall and the references were re-scored on 288 fresh tasks (seed 1), never used for selection:
`runs/hardware/validate1.jsonl`.

| design | grid (96) | validation (288) | formation | construction | logic | parallel | herd | runner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 cm drone, no connectors (L1cm-magnet-c0-a0.01-f0.5) | 0.763 | **0.785** | 0.640 | 1.000 | 0.932 | 0.756 | 0.428 | 0.961 |
| 1 cm drone with switchable connectors (L1cm-epm-c0.06-a0.01-f0.5) | 0.763 | **0.785** | 0.638 | 1.000 | 0.932 | 0.756 | 0.428 | 0.961 |
| ideal drone (unbreakable connectors, unlimited thrust) | 0.770 | 0.784 | 0.639 | 1.000 | 0.932 | 0.756 | 0.421 | 0.961 |
| voxel world's own rules | 0.784 | 0.771 | 0.631 | 0.997 | 0.919 | 0.703 | 0.430 | 0.961 |
| 3 cm drone, no connectors | 0.751 | 0.763 | 0.640 | 1.000 | 0.932 | 0.756 | 0.281 | 0.961 |
| 3 cm crawler, permanent magnets (L3cm-magnet-c0.12-a0.25-f0) | 0.753 | **0.749** | 0.646 | 0.994 | 0.878 | 0.717 | 0.394 | 0.863 |
| ideal crawler (unbreakable connectors, unlimited torque) | 0.742 | 0.742 | 0.632 | 0.977 | 0.892 | 0.683 | 0.376 | 0.902 |
| 1 cm crawler, switchable (L1cm-epm-c0.25-a0.03-f0) | 0.743 | 0.742 | 0.632 | 0.977 | 0.892 | 0.683 | 0.376 | 0.902 |
| 10 cm crawler, permanent magnets (L10cm-magnet-c0.12-a0.4-f0) | **0.778** | 0.733 | 0.627 | 0.956 | 0.865 | 0.675 | 0.378 | 0.902 |
| 10 cm crawler, switchable | 0.761 | 0.721 | 0.622 | 0.950 | 0.878 | 0.638 | 0.389 | 0.863 |
| crawler with no connectors (any size) | 0.715 | 0.662 | 0.627 | 0.963 | 0.757 | 0.620 | 0.172 | 0.804 |

What the round says:

1. **The grid's winner was luck.** The 10 cm permanent-magnet crawler topped the grid at 0.778
   but drops to 0.733 on fresh tasks. With 258 designs on 96 tasks, the top is selected on
   noise. Grid ranks are only good for picking candidates, never for conclusions.
1. **Drones: about 0.04 above crawlers, and only at small size.**
   1. 1 cm drones match the ideal drone (0.785 against 0.784).
   1. At 3 cm drones lose most of their pushing (herd 0.43 to 0.28): thrust per weight falls
      as L^(-1/3), so they have little spare thrust to push with.
   1. At 10 cm no flight system that fits in the cube lifts the module.
   1. The edge over crawlers is mobility, seen in logic, parallel and runner: drones go
      straight to the target. It is not force.
1. **Connectors are worth about 0.08 to a crawler.** Without connectors a crawler cannot climb
   on others or hang off them; pushing falls to 0.17 and eating (logic) to 0.76. With them, the
   best crawler reaches the ideal crawler.
1. **The connector type depends on size.**
   1. Permanent magnets lead at 3 cm, where a heavy actuator can pry bonds open.
   1. Switchable (electropermanent) connectors take over as size grows: at 10 cm, the best
      permanent design on the grid needed 40% of the module as actuator.
1. **At 1 cm a module can have both flight and connectors.** The 1 cm drone with switchable
   connectors (strength 28 module weights, thrust 1.27) scores the same as the connector-less
   one. Structural ability costs nothing on this metric there. The square-cube law makes both
   thrust per weight and magnet hold per weight abundant at small size.

### Sensitivity to the constants

Each run halves one constant and re-scores on the 288 validation tasks:
`runs/hardware/sensitivity1.jsonl`.

| design | as estimated | magnet pressure x 0.5 | flight power x 0.5 |
| --- | --- | --- | --- |
| 1 cm drone, switchable connectors | 0.785 | | 0.721 (cannot fly) |
| 1 cm drone, no connectors | 0.785 | | 0.746 (thrust 1.07, herd 0.17) |
| 3 cm drone, no connectors | 0.763 | | 0.662 (cannot fly) |
| 3 cm crawler, permanent magnets | 0.749 | 0.750 | |
| 1 cm crawler, switchable | 0.742 | 0.743 | |
| 10 cm crawler, permanent magnets | 0.733 | 0.723 | |

What it shows:

1. **Crawlers hold up.** Their scores barely move with connector strength.
1. **The drone lead does not.** It rests on the flight power estimate, taken from Crazyflie-class
   parts. At half of it:
   1. The 1 cm drone with connectors and the 3 cm drone can no longer fly.
   1. The connector-less 1 cm drone barely flies and falls just below the best crawler.

   Before trusting the drone result, the flight constant needs measuring on a 1 cm prototype.

### What the metric cannot see yet

The task mix decides the answer, and the suite is mostly about mobility.

1. **Herd** is the only family that needs force. All good designs sit at about 0.4 there, which
   the expert's herding limits, not the hardware.
1. **Construction** tasks are towers, flat bridges and filling a trench, and mostly need
   compression. Even crawlers without connectors get 0.96.
1. **Nothing in the suite needs** tension, reach, a lever or carrying a load across a structure.
1. **Arching.** A two-layer bridge with no connectors stands as a flat arch, like dry masonry,
   because terrain friction under resting modules is not limited.

So "drones win" here means only that drones win this suite. They lose the moment the suite asks
for force or structure. To make the metric weigh what the swarm is for, add structure-and-force
tasks to the suite:

1. carry the cheese across the bridge;
1. reach out over a gap;
1. lift with a lever;
1. hold a load.

That changes the product's task distribution, so it is a decision for us, not for the search.

## Rigid-body check: a pivoting cube in MuJoCo

The voxel physics above treats a move as allowed or not. To decide what to build, the moves
themselves have to be simulated: does a real cube land in the next cell? `hardware/cube.py`
simulates an M-Blocks-style module in MuJoCo, and `hardware/cube_search.py` scores and
hill-climbs designs. Tests: `uv run pytest -q tests/test_cube.py`.

### Model

1. **Module.** A rigid cube with a flywheel inside. The flywheel starts at speed, then a band
   brake (Coulomb friction of fixed torque) stops it within milliseconds. Its angular momentum
   jumps into the cube, which pivots about an edge.
1. **Magnets.** Two edge magnets per edge and one alignment magnet per face, all inset from the
   surface. Every pair of magnets on different cubes attracts with a force that peaks where twins
   sit in face contact and falls as r^-3. Edge magnets of cubes that share only an edge still
   attract, and that attraction is the hinge.
1. **Contacts.** MuJoCo's soft contacts with Coulomb friction.
1. **Variation per trial.** Friction 0.3 to 0.7, magnet strength ±25% per cube, brake torque
   ±40%, flywheel speed ±2%, mass ±5%, placement errors up to 0.03 L.
1. **Control.** The only knob is the flywheel speed, tuned for each move as a real module would
   be calibrated: a speed grid with a refinement pass, then fresh trials at the chosen speed.
   From round 3 on a move may also be retried after a harmless failure (see round 3).

### Moves

| primitive | scene | turn |
| --- | --- | --- |
| floor_roll | lone cube on the floor rolls one cell | 90° |
| floor_peel | the same, bonded to a neighbour behind and one beside | 90° |
| traverse | on top of a row, to the next cell | 90° |
| convex_up / convex_down | up onto a neighbour's top / down off it | 180° |
| concave_up / concave_down | climb a wall to hang on its side / back down | 90° |
| wall_crawl | hanging on a wall, sideways to the next cell (flywheel vertical) | 90° |

A move succeeds when the cube rests aligned in the target cell and no neighbour moved. A
harmless failure leaves it aligned where it started, so the policy can simply retry. A harmful
failure leaves it off the lattice or knocks a neighbour. Moves with neighbours run twice: once
with the neighbours anchored (part of a large lattice) and once free (lone modules).

### Calibration against the 3D M-Blocks

Reference: Romanishin et al., ICRA 2015.

- **Module:** 50 mm cube, 150 g.
- **Flywheel:** 8.4e-6 kg m^2, up to 20000 rpm.
- **Brake:** peaks at 2.6 N m over 10 to 30 ms.
- **Magnets:** about 23 N face pull and an 18 N hinge, from the M-Blocks patents.
- **Measured success:** traverse 100%, convex 95 to 100%, concave 55 to 90%, overall above 88%.

With the edge magnets at 4.5 module weights per twin pair, the simulated rates with anchored
neighbours are:

| move | sim | real M-Blocks |
| --- | --- | --- |
| traverse | 0.90 to 0.97 | 1.00 |
| convex up / down | 0.70 to 0.78 | 0.95 to 1.00 |
| concave up / down | 0.78 to 1.00 | 0.55 to 0.90 |
| wall crawl | 0.82 to 0.85 | no counterpart measured |

Two fixes were needed to get there:

1. MuJoCo's joint `frictionloss` is a soft constraint. As a brake it acted like a damper peaking
   below 1 N m, so no pivot could ever break a bond. The brake is now an explicit Coulomb torque,
   capped at what would stop the flywheel within two steps, which keeps it stable at any size.
1. Off-lattice trials needed early exits so that a design scores in about a minute.

### What the voxel expert asks of a cube

These are the moves the suite's expert issued to the best crawler of round 1
(L3cm-magnet-c0.12-a0.25-f0): 189k moves over 192 tasks.

| move | share |
| --- | --- |
| convex up | 0.33 |
| horizontal diagonal, elevated | 0.20 |
| horizontal diagonal, on the floor | 0.15 |
| convex down | 0.11 |
| wall crawl | 0.09 |
| floor roll | 0.05 |
| floor peel | 0.03 |
| concave down / up | 0.02 / 0.02 |
| traverse | 0.005 |

Round 1 lets a crawler move along two horizontal axes at once (`physics.allowed_moves` only bans
three-axis moves). A pivoting cube cannot: a diagonal takes two pivots. More than a third of
the expert's moves are such diagonals, so round 1 flattered crawlers. Rounds 2 and 2b counted a
diagonal as the matching straight move squared.

Since 2026-11-10 the policy issues single pivots only, so the score leaves the two diagonal rows
out and renormalises the rest (`MOVE_MIX` in `cube_search.py`). The mix of an expert that plans
without diagonals has not been measured; this is the old mix with two rows removed.

| move | share without diagonals |
| --- | --- |
| convex up | 0.509 |
| convex down | 0.169 |
| wall crawl | 0.133 |
| floor roll | 0.074 |
| floor peel | 0.051 |
| concave down | 0.031 |
| concave up | 0.025 |
| traverse | 0.008 |

Half of all moves are convex up, so that one move decides the score.

### Score

**Move score** is the sum, over moves, of each move's share in the mix times its validated
success rate. Anchored and free neighbours are averaged. The score is in [0, 1]. The scores in
rounds 2 and 2b use the mix with diagonals; round 3 uses the mix without.

The design space is scaled from the calibrated M-Blocks. Brake torque goes as L^3 and magnet
pull as L^2. The flywheel radius stays at 0.38 L, and density is 1200 kg/m^3. The knobs are:

- flywheel mass share
- motor top speed
- brake strength
- magnet strength
- a steel floor
- the floor's grip on each magnet (0 is no steel; thinner steel or a spacer lowers it)

### Round 2: 3 cm cube (2026-11-08)

**Climb.** Greedy hill-climb from the M-Blocks design scaled to 3 cm, on 3 cm cubes. Each round
evaluates every one-step neighbour and moves to the best one when it wins by at least 0.01.
Selection uses validation seeds, and the final design is re-scored on fresh test seeds:
`runs/cube/climb-3cm.jsonl`, `runs/cube/steel-sweep-3cm.jsonl`, `runs/cube/climb-3cm-steel.jsonl`.

| design | move score | harm (off or knock) |
| --- | --- | --- |
| M-Blocks scaled to 3 cm | 0.230 | 0.30 |
| climbed: flywheel share 0.25, brake x 2.56, magnets x 0.39, no steel (test seeds) | 0.573 | 0.18 |
| the same on steel with grip 0.3 (test seeds) | 0.663 | 0.02 |
| climbed again on steel: magnets x 0.63 (test seeds) | **0.716** | **0.01** |

Steel grip, on the climbed design (validation seeds):

| grip | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
| --- | --- | --- | --- | --- | --- |
| move score | 0.556 | 0.644 | 0.664 | 0.663 | 0.586 |
| harm | 0.23 | 0.10 | 0.11 | 0.04 | 0.16 |

What it says:

1. **A 3 cm cube needs more flywheel than M-Blocks scaled down.** The flywheel's momentum scales
   as L^5 at a given speed, while a pivot needs momentum scaling as L^4.5. The climb raised the
   flywheel's share of the mass from 0.155 to 0.25 and the brake to 2.56 times the scaled
   M-Blocks brake (about 1.4 N m).
1. **Magnets should be weaker at 3 cm.** Magnet pull scales with face area, so per module weight
   it grows as 1/L, and at 3 cm the reference magnets out-hold the pivot. The climb settled at
   0.39 of the reference volume.
1. **On steel, the magnets want to be stronger again.** The second climb, started on steel at grip
   0.3, raised the magnets to 0.63 of the reference volume (0.716 on test seeds, harm 0.013).
   Without steel, strong magnets dragged lone neighbours along. Once the floor holds them down, a
   stronger hinge carries the cube through the 180-degree climb over a neighbour: convex up rose
   from 0.42 to 0.66, and that move is a third of the mix. Three more rounds found nothing better.
   In physical units the design is:
   1. a 34 g cube;
   1. an 8 g flywheel ring at 11 mm radius, at about 14,000 rpm;
   1. a brake that stops it in about 1 ms (1.4 N m);
   1. about 3 N per hinge edge and 12 N per face.
1. **The arena floor should be steel with a light grip.** It holds lone neighbours down, so a
   mover pivoting over them no longer drags them. At grip 0.3 harmful failures fall from 0.18 to
   0.02. Grip 0.5 holds the mover itself too hard: convex up falls to 0.2. In practice that means
   a thin steel sheet or a spacer such as paint or a mat; the right one needs a pull test on the
   bench.
1. **Weak spots that remain.**
   1. Concave up (0.40).
   1. Convex up (0.66).
   1. Rolling away from neighbours bonded behind and beside (0 everywhere). In-plane magnet shear
      holds the side bond, which matters for packed crowds on the floor.

   The policy has to plan around these, or the hardware needs a way to release a bond, such as an
   electropermanent face.
1. **One artefact.** A lower motor top speed was accepted in round 7, but its only effect is a
   finer speed grid for tuning, about +0.02. It is not physics.

Caveats: the scaling laws for the brake (L^3) and magnets (L^2) are assumptions, and convex up is
only simulated from the floor, not on top of a structure. Every number here should be replaced by
bench measurements of a prototype's magnet pull, brake torque and flywheel speed.

### Round 2b: 5 cm cube on steel, for the first build (2026-11-08)

The first swarm is 10 cubes at 5 cm on a ₹10,000 budget, built from off-the-shelf modules. This
round climbs from the calibrated M-Blocks on a steel floor at grip 0.3:
`runs/cube/climb-5cm-steel.jsonl`.

| design | move score | harm |
| --- | --- | --- |
| M-Blocks on steel (validation seeds) | 0.511 | 0.04 |
| climbed: brake x 2.56, the rest as M-Blocks (test seeds) | **0.721** | **0.07** |

At 5 cm only the brake needed to change. Raising the top speed, strengthening the magnets or
stiffening the brake further did not help. A brake of x 2.56 is 6.7 N m, which stops the flywheel
in about 2.5 ms; a spring-loaded pin dropping into the flywheel's rim gives a stop that hard.
Most of the remaining harm is convex up over a lone neighbour, which gets dragged along (0.38).

Targets for the build:

| quantity | target |
| --- | --- |
| cube | 150 g |
| flywheel | inertia 8.4e-6 kg m^2: a 23 g thin ring at 19 mm radius (a brass or steel pipe offcut about 38 mm across), or about 45 g of 38 mm washers, since a solid disc stores half as much per gram as a ring |
| brake | stops the flywheel within about 2.5 ms |
| magnets | 6.6 N per twin pair, 13 N per hinge edge |
| floor | steel; spacer chosen so each magnet's grip is 0.3 of its pull on a twin |

Tuned flywheel speeds per move (rpm):

| move | rpm |
| --- | --- |
| floor roll | 4,800 |
| concave down / convex down | 8,400 to 9,800 |
| traverse / wall crawl | 10,600 to 11,500 |
| convex up | 17,000 to 19,000 |
| concave up | 20,000 |

### Round 3: retries for the 5 cm cube, single pivots only (2026-11-10)

The build is fixed at the round 2b design (5 cm, 150 g, brake x 2.56, steel floor at grip 0.3), so
this round changes only what a module does with it. Scores use the mix without diagonals. The
policy will be trained with random move failures, so a harmless failure (the cube ends aligned
where it started) costs a retry, and a harmful one (off the lattice, or a neighbour knocked) is
what breaks a demo: harm first, then success.

**What changed** (`cube.py`: `trial(..., retries=...)`; `cube_search.py`: `Control`):

1. **Retry after a harmless failure.** A module that finds itself at rest with the same face down
   (accelerometer) spins up and tries again, up to 6 attempts, at the same speed or 6% faster each
   time. The retry runs in the same simulated scene: same friction, magnets and masses, the cubes
   wherever the failed attempt left them, a new draw of brake torque and speed error. Knocks are
   judged against where the neighbours first stood, so drift over attempts counts.
1. **Harm-aware calibration.** The start speed and step are picked by success rate minus 3 x the
   harmful rate, on up to 96 shared tuning trials for the best candidates (8 for all). Round 2b
   picked by success alone on 5 trials per speed, which chose 17,100 rpm for convex up over a lone
   cube, where a third of the moves knock it over.
1. **Moves that never pay are not attempted.** If no speed has a positive value in calibration the
   module does not try (floor peel; concave up against a lone two-cube stack, which topples).

Nothing in the simulator changed: friction range, magnet model, tolerances and per-trial variation
are as before.

**Result.** Round 2b design throughout. "Before" is round 2b's control on its 40 test trials per
row (`runs/cube/nodiag-baseline.jsonl`, reproducing 0.721 on the old mix). "After" is 120 test
trials per row on seeds not used for anything else (`runs/cube/ladder-5cm-steel-fresh.jsonl`).
Cells are success / harmless failure / harmful failure, per move including its retries.

| move | share | neighbours | before | after | attempts |
| --- | --- | --- | --- | --- | --- |
| convex up | 0.509 | anchored | 0.63 / 0.35 / 0.03 | 0.77 / 0.20 / 0.03 | 2.7 |
| | | free (lone cube) | 0.60 / 0.03 / 0.38 | 0.91 / 0.00 / 0.09 | 1.2 |
| convex down | 0.169 | anchored | 0.95 / 0.05 / 0.00 | 1.00 / 0.00 / 0.00 | 1.2 |
| | | free | 0.93 / 0.08 / 0.00 | 0.99 / 0.00 / 0.01 | 1.1 |
| wall crawl | 0.133 | anchored | 1.00 / 0.00 / 0.00 | 1.00 / 0.00 / 0.00 | 1.0 |
| | | free | 0.93 / 0.03 / 0.05 | 0.96 / 0.00 / 0.04 | 1.3 |
| floor roll | 0.074 | none | 1.00 / 0.00 / 0.00 | 1.00 / 0.00 / 0.00 | 1.0 |
| floor peel | 0.051 | anchored, free | 0.00 / 1.00 / 0.00 | 0.00 / 1.00 / 0.00 | not tried |
| concave down | 0.031 | anchored | 1.00 / 0.00 / 0.00 | 1.00 / 0.00 / 0.00 | 1.0 |
| | | free | 0.98 / 0.00 / 0.03 | 0.98 / 0.00 / 0.03 | 1.0 |
| concave up | 0.025 | anchored | 0.90 / 0.10 / 0.00 | 1.00 / 0.00 / 0.00 | 1.4 |
| | | free (lone stack) | 0.00 / 0.93 / 0.08 | 0.00 / 1.00 / 0.00 | not tried |
| traverse | 0.008 | anchored, free | 1.00 / 0.00 / 0.00 | 1.00 / 0.00 / 0.00 | 1.0 |

| control | move score | harm | harmful failures in 300 moves |
| --- | --- | --- | --- |
| round 2b: one attempt, speed picked by success | 0.722 | 0.107 | 32 |
| one attempt, harm-aware calibration | 0.747 | 0.049 | 15 |
| up to 6 attempts, harm-aware calibration | **0.850** | **0.036** | **11** |

A first pass of the retry control on test seeds 0 to 119, before the last change to the
calibration (a third round of tuning trials), gave 0.862 and 0.031
(`runs/cube/ladder-5cm-steel.jsonl`); the two agree within noise. One row's rate at 120 trials is
good to about +-0.05, the move score to about +-0.02. A move takes 1.5 attempts on average, so a
300-move episode is about 460 flywheel spin-ups.

**What limits it** (from traces, `runs/cube/sweep-convex-up-5cm.jsonl`):

1. **Convex up is a throw, not a hinge.** At launch the mover is wedged between the floor and the
   neighbour's face, and the brake's impulse (by a rigid-pivot estimate, about 140 N at the pivot
   edge for 2.5 ms) is ten times what the edge magnets hold (12 N). The mover leaves the edge by 0.1 to 0.3 L, flies half a
   turn and lands on top. Whether it lands depends on friction (0.3 to 0.7) and the brake draw:
   over anchored neighbours the threshold speed runs from 15,000 rpm to above the motor's 20,000,
   too slow falls back (harmless), too fast bounces off backward. Between 15,600 and 15,800 rpm, 6 of
   40 identical scenes changed outcome, so no calibration finds a clean window; retries help because each
   attempt draws again.
1. **A lone cube cannot take the landing.** Over a free neighbour the remaining harm is the
   neighbour: the launch shoves it (more with a hard brake draw), or the landing mover tips the
   pair over its far edge. This is 0.09 of such moves and two thirds of all remaining harm.
   A second cube behind the neighbour removes it: climbing onto the end of a free row of two, four
   attempts at 15,900 or 16,600 rpm gave 40 of 40 with no harm at either speed, and one attempt at
   17,400 rpm gave 39 of 40 (`runs/cube/convex-up-backed-5cm.jsonl`; fixed speeds, 40 trials, not
   part of the score). A cube beside the neighbour instead of behind it does not help (0.63 at best, with
   harm 0.08).
1. **The design knobs do not buy a wider window.** One-knob and two-knob variations around the
   design (brake x 1 to 3, magnets x 0.7 to 1.4, grip 0.2 to 0.6, flywheel share up to 0.25) all
   gave narrower or dirtier convex-up windows than the round 2b design on 16 trials per speed. A
   heavier flywheel only moves the same window to lower speed (share 0.22: 11,000 to 14,000 rpm
   instead of 15,000 to 20,000), which is worth having if the motor cannot hold 20,000 rpm with
   the flywheel on it, since convex up and concave up now run at the motor's limit.
1. **Floor peel never moves.** With bonds on two faces and the floor, even 20,000 rpm leaves the
   cube exactly in place over anchored neighbours, and knocks free ones (8 of 10). It is always
   harmless when not forced, and 5% of the mix: the policy has to learn it fails.

**Is 0.95 per move reachable?** In this model: yes already for traverse, floor roll, convex and
concave down, wall crawl and concave up on a lattice (0.96 to 1.00 with retries); convex up
reaches it only onto a backed neighbour; floor peel and concave up against a lone stack do not
work at all. With convex up at 0.84 the mix cannot pass about 0.85. The cheapest gain left is in
the policy, not the cube: do not climb a lone cube, climb the end of a row.

**Where the model is probably wrong.** The magnet model was calibrated to M-Blocks success rates,
not to their forces. With the calibrated magnets a face holds 53 N and a hinge edge 12 N
(`Magnets.wrench` on two cubes, face to face and pivoted 45 degrees about a shared edge); the
M-Blocks patents give about 23 N and 18 N. A face 2.3 times too strong and a hinge a third too
weak both push the same way: the simulated convex up loses its hinge and is thrown, where the real
one pivots and is measured at 0.95 to 1.00, and the simulated floor peel is glued in place. The
model was left as it is. A bench test of one hinge (pull-off force at an edge and on a face) would
say which way to correct it, and it is the first thing to measure on the prototype.

For the build, beyond round 2b's targets: an accelerometer per cube (to detect a harmless
failure), and calibration per move of start speed and step. Start speeds from this round (rpm):
floor roll 4,800; concave down 8,400 to 9,100; convex down 9,100 (step 6%); wall crawl 9,800 to
10,600; traverse 10,600; convex up 15,800 over a lone cube and 17,800 on a lattice; concave up
17,100 (step 6%).

### Ten cubes in one scene (2026-11-10)

`hardware/cube_swarm.py` puts N of the round 2b cubes in one persistent scene on steel. Any cube
can be the mover; `move(cube, direction)` picks the primitive from the cube's surroundings and
runs it with cube.py's mechanism, and `lattice()` reads the grid back from the poses, so a caller
can re-plan after a failed pivot. Tests: `tests/test_cube_swarm.py`.

    python -m beebot.hardware.cube_swarm demo --seed 14 --video runs/cube_swarm/demo.mp4
    python -m beebot.hardware.cube_swarm mixed --moves 240 --seed 100

What differs from cube.py:

1. **Flywheel.** One body of isotropic inertia on a ball joint, spun about whichever world axis the
   pivot needs. About one axis it is cube.py's ring, so the calibrated speeds carry over. A build
   needs three flywheels (or one that can be turned); the mass of the idle two is not modelled.
1. **Magnet pairs.** Only cube pairs with centres closer than sqrt(3) L + 6 d0 (1.91 L) are
   searched. No magnet pair within cube.py's 6 d0 cut-off can be further apart, so the forces are
   the same (tested against cube.py's search).
1. **Variation.** Friction is drawn once per scene, magnet strength and mass once per cube, brake
   and speed once per move. Cubes start with cube.py's placement error and poses are never reset,
   so errors carry from move to move.
1. **Speeds.** cube_search's speeds for round 2b with anchored neighbours, except three measured
   in this scene on 10 to 12 jittered trials per speed: concave up 17,800 rpm (20,000 reached 9
   of 12; 17,083 and 18,500 reached 12 of 12), convex down 12,500 (9,838 reached 7 of 10 off the
   end of a row; 11,500 to 13,500 reached 10 of 10), and convex up from hanging on a wall 14,500
   (8 of 10; cube.py only climbs from the floor).

**A cube moves only when nothing but its support holds it.** Outcomes of 207 single moves drawn
at random by kind from five ten-cube layouts (`runs/cube_swarm/mixed.jsonl`):

| mover | moves | reached | stayed | harmful |
| --- | --- | --- | --- | --- |
| free: touches only its support and the cube it pivots on | 63 | 55 | 2 | 6 |
| bonded: another cube face to face (side, behind, or below when climbing) | 121 | 0 | 121 | 0 |
| hinged: another cube touching by an edge | 23 | 16 | 6 | 1 |

By primitive, free movers: floor roll 34 of 34, concave up 12 of 14, concave down 5 of 5,
traverse 1 of 1, convex up 3 of 9 (5 off the lattice, 1 knock). Hinged: convex up from hanging 5
of 8, convex down 3 of 3, concave down 3 of 3, concave up 3 of 4, wall crawl 2 of 3, floor roll 0
of 1, convex up from the floor 0 of 1. Every bonded move failed harmlessly at every primitive: floor peel 42, convex up 46, convex
down 19, traverse 12, wall crawl 2. A packed 2 x 5 patch is therefore frozen: 48 of 48 moves
stayed, and separate speed sweeps up to 20,000 rpm did not free a patch cube.

Two more ways a cube is held, found while scripting:

1. **Sliding into a slot.** A roll into a cell with a cube beside it along the flywheel's axis
   (the mover's face slides down that cube's face): in an earlier script, of 22 runs that tried
   it 3 got in, 13 stayed through four tries and 6 ended leaning, off the lattice. Meeting a cube
   head on works (6 of 6, and every such roll in the final script).
1. **Edge contact.** A cube touching another only by a vertical edge is held by the edge magnets.
   In an earlier script a floor roll away from such a cube at the calibrated 4,839 rpm stayed
   through four tries in 18 of 22 runs, and the 4 that passed stayed at the next such roll. One
   trial at 7,000 rpm broke free.

   Both counts are from runs of earlier scripts whose logs were not kept; `plan` reports the
   three hindrances for every move as `bonds`, `hinges` and `rubs`.

**Scripted demonstration.** A row of five and five loose cubes; 20 moves build a three-high
tower with a step on either side (`DEMO_SCRIPT`). The loose cubes are routed so that they touch
nothing until they meet the structure head on. After each move the grid is read back, a harmless
failure is retried up to three times and a harmful one stops the script. Over seeds 0 to 23
(`runs/cube_swarm/demo-seeds.jsonl`): 11 runs finished all 20 moves, 13 stopped at a harmful
move (2 at the first convex up, 6 at the convex up from hanging, 4 at the second convex up, 1 at
the last). Per attempt: 293 reached, 5 stayed, 13 harmful of 311; floor roll 198 of 199, concave
up 22 of 22, traverse 12 of 12, convex up 45 of 53, convex up from hanging 16 of 25. Video of
seed 14 (one retry): `runs/cube_swarm/demo.mp4`.

**Cost.** A move takes 0.40 s of simulated time (0.05 s before the brake, 0.25 s of brake, 0.1 s
of stillness; the longest seen was 0.63 s). Ten cubes: 0.25 s of wall clock per move in one
process on the M4 Pro, 0.42 s with three processes side by side. Per time step, measured on a loaded machine: 40 us for one cube, 270
us for ten packed, 790 us for twenty packed, 1,370 us for forty packed, 205 us for twenty spread
out. Contacts and magnet pairs grow with the number of touching faces, not with N^2.

What this asks of the policy and the hardware:

1. The voxel expert moves cubes out of packed groups all the time. With permanent magnets those
   moves do nothing. Either the planner only moves free cubes (ends of rows, loose cubes, cubes
   hanging on a wall) and approaches head on, or the hardware can release a bond (an
   electropermanent face), which is the same conclusion as round 2.
1. Convex up is the weak move: about one in five ends off the lattice or knocks a cube (13 of 62
   from the floor and 6 of 33 from a wall, both runs together). A closed loop needs a recovery for a cube lying off the lattice,
   which this file does not have: `move` refuses such a cube.
1. Not modelled: chamfered edges, which may ease sliding into a slot; a harmful outcome that
   leaves the mover aligned in a third cell ("strayed") is counted as harmful, as in cube.py,
   though a planner could carry on from it.

### The rule in the voxel world (2026-11-10)

The voxel world now has this as an opt-in move rule, `free_cube` (`vla/voxel/world.py`:
`pivot_moves`, `refuse`; `step(..., rule="free_cube")`), with the same pivot table as
`cube_swarm.plan`: support side, swing clearance, and no cube on any face but the support's.
`switch_cube` is the same rule for a cube that can release its bonds. What the expert can still
do under each is in `src/beebot/vla/voxel/README.md` ("Free cubes").

It is not the rule of `physics.allowed_moves` (round 1), and the two should not be mixed up:

| | `physics.allowed_moves`, permanent magnets | `free_cube` |
| --- | --- | --- |
| a bond the move leaves | costs torque, 0.5 s m g L each; the best round-1 crawler (L3cm-magnet-c0.12-a0.25-f0: s = 28, torque 34) pries two at once | holds the cube: 0 of 121 in MuJoCo, one bond is enough (77 of the 121 had exactly one) |
| a bond the move keeps (a neighbour it stays beside) | free | holds the cube all the same (in-plane shear) |
| what it pivots on | the destination must touch something by a face | a support beside or below the cube, continuing under the destination (quarter turn) or turned around (half turn); never a cheese's side, never a ceiling |
| room to swing | not checked | the cells the turning cube sweeps must be empty |
| horizontal diagonals | allowed | never (two pivots) |
| a support that moves in the same step | not checked | the move does not happen |
| structure | statics with connector strength | the voxel world's support rule (any face-connected group holds) |
| pushing a cheese | force model | untouched: a press is not a pivot |

On 48 sampled cube scenes (1,831 robots, `demos --cube --counts log` draws) a robot has 8.0 of
the 14 cube moves open (destination empty and inside). `physics.allowed_moves` lets the
permanent-magnet crawler make 4.3 of them, `switch_cube` 3.2 and `free_cube` 2.1. Every
`free_cube` move is also allowed by `physics.allowed_moves` (0 of 3,791 refused); `free_cube`
refuses 51% of the moves physics allows and `switch_cube` 25%. 23% of robots have no move at
all under `free_cube`, 9% under `switch_cube`.

The reason is the actuator, not a bug in either: round 1 models a geared hinge that can hold a
torque for as long as it takes to pry a face open; the cube has a flywheel impulse of a few
milliseconds, which cannot shear a face bond at any speed it reaches.

### Switchable magnets in the swarm (2026-11-11)

The training stores use the voxel world's `switch_cube` rule: a cube releases its bonds to move.
With permanent magnets `cube_swarm` refuses all of those moves (0 of 121, above), and 45% of the
stores' pivots have a face bond besides the support (71% an edge contact). The cubes are to have
switchable magnets (electromagnets or electropermanent), so `cube_swarm` now has them:
`Swarm(release=True)`. While a cube moves, its magnet pairs with every cube except the ones it
leans on (`Pivot.leans`: its support, and for a quarter turn the cube it comes to rest against)
are off; when the scene is still they are back on and the scene settles for 0.1 s more. The steel
floor still pulls on every magnet. `Swarm.moves` also runs several moves at once (planned on the
grid before any of them moves, flywheels braked together); `move` is its one-order case, step for
step the old move (checked on 33 moves in four scenes: identical results and poses). `Swarm.spin` is a raw
flywheel spin about any axis. Measurements: `hardware/cube_outcomes.py`, rows in
`runs/cube_outcomes/`, the full tables in `runs/cube_outcomes/report.txt`; how the voxel world
uses them in `src/beebot/vla/voxel/README.md`, "Move outcomes measured in MuJoCo".

**Release frees edge contacts and slots, but not a cube in a crowd.** One mover in a small scene,
60 trials released, 200 permanent (lattice outcome once the scene rests):

| move | permanent: reached / stayed / harmful | released |
| --- | --- | --- |
| floor roll away from an edge contact | 0.15 / 0.85 / 0 | 0.93 / 0.07 / 0 |
| floor roll into a slot | 0.01 / 0.71 / 0.28 | 0.48 / 0.48 / 0.03 |
| floor peel (away from a face bond) | 0 / 1.00 / 0 | 0.28 / 0.72 / 0 |
| convex up from hanging | 0.65 / 0.29 / 0.06 | 0.85 / 0 / 0.15 |
| convex up onto a lone cube | 0.17 / 0.04 / 0.79 | 0.20 / 0.02 / 0.78 |
| convex down off a lone stack of two | 0.71 / 0 / 0.29 | 0.78 / 0 / 0.22 |
| concave up a lone stack of two | 0 / 0 / 1.00 (it topples) | 0 / 0 / 1.00 |

Every other template (floor roll, traverse, convex up onto a row, convex down off a row, concave
up a backed wall, concave down, wall crawl) is within a few percent either way. In the stores'
own scenes (380 moves, cropped to five columns around the mover) a released floor roll with a
bond or an edge contact stays two times in three, and 0.90 of those with a bond, an edge contact
and a slot stay: the cubes beside the mover are pulled onto it by their own edge magnets to the
cubes around it, and grip it by friction. Releasing only the mover's pairs does not free it. At
4,839 rpm a cube in the middle of a row rolled out sideways 3 times in 10; at 6,000 rpm 10 of 10;
faster threw it off the lattice (7,500: 3 of 10, 9,000: 8 of 10). A cube in the front row of a
2 x 5 patch got out at most 2 times in 10 at any speed up to 9,000 rpm (10 trials each).
`cube_swarm.RPM` was calibrated with permanent magnets and is unchanged; release mode has not been
calibrated.

**Harm per pivot.** In the ten-cube layouts, moves drawn at random by class: 529 pivots, reached
0.52, stayed 0.36, off the lattice 0.09, strayed 0.01, knock 0.02: **12.3% harmful** (the scripted
demonstration's 4.2% chose its moves). Over the stores' mix of moves: 9.3% harmful, 47% reached,
44% stayed. A small-swarm episode of about 115 pivots meets about 11 harmful ends.

**Where harm comes from.** Store-weighted: convex down 36% of the harm, convex up 24%, concave up
15%, floor roll 12%, convex up from hanging 9%. A lone support (a cube or stack of two that
touches nothing else) is 2.7% of the pivots and 21% of the harm; 12 of the 17 knocks seen were
a lone support shoved or dropped with the mover thrown past it. Nearly every other harmful end is
the mover left tilted, in its start cell (24 of 82) or one cell along its motion (23).

**Cube-level levers.** The five harm-source moves above (convex up onto a lone cube, onto a row,
from hanging, convex down off a lone stack, concave up a lone stack), released, 30 trials each
with the same seeds in every variant (`runs/cube_outcomes/levers-release.jsonl`). Harmful share:

| variant | convex up, lone | convex up, row | convex up, hanging | convex down, lone | concave up, lone | all five |
| --- | --- | --- | --- | --- | --- | --- |
| as built | 0.80 | 0.07 | 0.27 | 0.37 | 1.00 | 0.50 |
| pivot speed x 0.85 | 0.03 | 0.03 | 0.03 | 0.00 | 1.00 | 0.22 |
| pivot speed x 1.15 | 0.90 | 0.27 | 0.67 | 0.47 | 1.00 | 0.66 |
| magnets x 1.4 (edge and face) | 0.14 | 0.07 | 0.00 | 0.03 | 0.90 | 0.23 |
| magnets x 0.7 | 1.00 | 0.70 | 0.87 | 0.50 | 1.00 | 0.81 |
| brake torque x 0.6 | 0.47 | 0.10 | 0.20 | 0.07 | 0.90 | 0.35 |
| brake torque x 1.5 | 0.80 | 0.10 | 0.33 | 0.43 | 1.00 | 0.53 |

1. **Slower convex moves are the cheapest cut.** At 0.85 of the calibrated speed convex up onto a
   lone cube lands 0.67 of the time with 0.03 harm (0.20 and 0.80 now), convex down off a lone
   stack lands every time (0.63 now); the cost is more harmless stays (from hanging 0.50, onto a
   row 0.30). Round 3's harm-aware calibration found the same for permanent magnets (15,800 rpm
   over a lone cube, `RPM` uses 18,852 for every convex up).
1. **Stronger magnets** (x 1.4) cut harm as much, with fewer stays; weaker ones (x 0.7) make most
   convex moves harmful. A softer brake (x 0.6) helps a little.
1. **Nothing saves a lone stack of two under concave up**: it topples 0.90 to 1.00 of the time at
   every setting. The planner has to climb backed walls only.
1. Before a demonstration, `RPM` should be recalibrated for release mode (speed per primitive and
   per held or free, success minus 3 x harm as in round 3); the voxel world's outcome table
   (`outcomes/mujoco_release.json`) then has to be measured and fit again.

**A cube off the lattice.** Of 30 tilted movers (replayed from their seeds: all 30 ended off
again), none settled after another second. A raw spin (`Swarm.spin`, released, about each of the
6 world axes at 5,000, 10,000, 15,000 and 20,000 rpm, each from a fresh replay) left the cube
aligned in a cell without disturbing another in 28 of 720 tries, and at least one of the 24 worked
for 17 of the 30 cubes. So a cube can be righted, by searching, and `plan` does not try: it
refuses a cube that is off. A recovery move would need its own primitive and controller.

**Several movers at once.** 80 steps of the stores with 2 to 7 pivots within two cells of each
other, run together and the first mover again alone from the same scene: it ended the same way in
73 (5 worse with company, 2 better); a knock came in 7% of mover rows together and 5% alone. No
difference shows at this count.
