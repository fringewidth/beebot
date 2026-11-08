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
the expert's moves are such diagonals, so round 1 flattered crawlers. The cube score counts a
diagonal as the matching straight move squared.

### Score

**Move score** is the sum, over moves, of each move's share in the mix times its validated
success rate. Anchored and free neighbours are averaged. The score is in [0, 1].

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
