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
