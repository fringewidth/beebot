# When do modules add up?

A literature survey (2026-11-18) on configurations where a modular robot's force, torque or load capacity
grows with the number of modules. It follows the lattice cube review (`hardware/handoff/lattice_cube_board.md`),
where cubes on rotary hinges needed 1.2-1.4 N·m at joints and carried 3-10 N·m at bonds, against about
0.3 N·m from a micro servo or the proposed dock.

Marks: (S) read in the source, (abs) abstract or search snippet only, (inf) our inference.

## Short answer

Modules add when two conditions hold:

1. **The load runs along the members, not across a joint.** Struts in tension or compression, or a thick
   block whose bonds share a cross-section, add capacity with every member. A cantilevered chain of rotary
   joints is the worst case: the root joint carries the moment of everything beyond it.
1. **Something fixes the force split between actuators.** A differential, equal-current (torque) commands, a
   shared fluid line or a shared tendon makes actuators share a load. Independent position-controlled
   actuators on one load fight instead.

Our cube design violates both. The literature names this exact limit for chain and pivoting modules.

## Evidence

### Chains and pivoting cubes do not add

1. "Adding more modules makes the robot bigger, but not stronger": the torque needed to lift a chain grows
   quadratically with module count, which has kept real systems under 40 modules (S: Tosun, Yim et al.,
   [arXiv 1710.01840](https://arxiv.org/pdf/1710.01840)).
1. SMORES (100 x 100 x 90 mm, 0.52 kg, joints 1.2-2.3 N·m, two motors combined through a differential on
   tilt): characteristic strength, the number of modules held as a horizontal cantilever, is 3 by connector
   strength alone (S: [Davey, Kwok, Yim, IROS 2012](https://www.modlabupenn.org/wp-content/uploads/SMORES.pdf));
   the spec table lists 2 by joint strength. Connector 11.4 N·m torsion (calculated). This is the only
   explicit module-count limit found, and it matches our simulation.
1. Roombots: 4 N·m worst case for the stretched horizontal pose, 5-7 N·m recommended (abs).
1. 3D M-Blocks (50 mm, 150 g): 2.6 N·m inertial actuator, one per module; no multi-module load result
   ([Romanishin et al.](https://people.csail.mit.edu/sclaici/pdf/romanishin20153d.pdf), S).

### Trusses add

1. Variable Topology Truss (Spinos, Yim, UPenn): members are linear actuators loaded only axially, nodes are
   passive spherical joints that carry no moment. A 1.5 m Spiral Zipper member supports 530 N at about 10:1
   strength-to-weight (S: [IROS 2017](https://www.modlabupenn.org/wp-content/uploads/2017/10/spinos2017vttdesign.pdf));
   a 2026 truss manipulates objects with up to 200 N per actuator (S: [arXiv 2605.13086](https://arxiv.org/pdf/2605.13086)).
   Built at metre scale only.
1. Tetrobot (Hamlin, Sanderson): the same principle, built up to 18 nodes and 48 links (strength numbers not found).
1. (inf) In a truss a moment M becomes chord forces M/d spread over many members, so no single joint has to
   resist M.

### Lattices add while beams, not joints, govern

1. Discretely assembled lattices (Jenett, Cheung, Gershenfeld, [Sci. Adv. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7673809), S):
   75 mm voxels; modulus scales linearly with density, and reaches 9, 56, 73, 89% of the continuum value at
   1-4 voxels across, because nodes (2x) and fasteners (4x) were made stronger than the beams.
1. Where joints are weak, scaling breaks: one acetal voxel holds 210.6 N in tension, two clipped voxels
   82.6 N (S: [Smith et al., ICRA 2024](https://cba.mit.edu/docs/papers/24.05.ICRA.pdf)); magnet joints
   (50 N) are called "not sufficient" for stiff structures (S: [Abdel-Rahman et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC10955888)).
1. Collective actuation (Campbell and Pillai, IJRR 2008; Lengiewicz et al.): force proportional to
   actuator volume, but only with strong fixed links forming frames and active modules moving between them;
   force per module falls as the actuator gets taller (deflection and friction). Simulation only
   ([Lengiewicz](https://www.ippt.pan.pl/repository/open/o7360.pdf), S). This is the closest thing to "throw
   more cubes at it", and it is untested in hardware.

### Actuators add with a fixed force split

1. Differential "cooperative actuation" (Tello leg, UIUC): two ±10 N·m motors gave −20 N·m at the knee;
   joint torque rises in proportion to motors recruited, provided joints do not peak at the same time
   (S: [paper](https://par.nsf.gov/servlets/purl/10397676)).
1. Redundant motors on one joint fight unless driven with equal current (abs: [NASA](https://ntrs.nasa.gov/api/citations/19930010209/downloads/19930010209.pdf),
   [MODUR, arXiv 2510.13356](https://arxiv.org/pdf/2510.13356)).
1. Muscle rule: units in parallel add force, in series add stroke. A 4-actuator pneumatic fascicle pack gave
   over 112 N, about 23% more than one actuator of the same volume (S: [Soft Robotics 2017](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5346960/)).

### Swarms add force only when aligned and large enough

1. Longhorn crazy ants: load speed rises linearly with carriers up to 15; 1-2 carriers fight and are slower
   than one (S: [Gelblum et al. 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4525283)).
1. Ant teams and 2-4 robot replicas move at the speed of the slowest member (S: [Wilson et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6227946)).
1. s-bots: cooperative transport "does not scale well" with robot count; the gripper link breaks above
   about 10 N (abs).

## What this means for the cube (inf)

1. A long horizontal arm of identical rotary-joint cubes will not get stronger with more cubes; at our
   joint torques expect 2-3 cubes, as SMORES shows.
1. Bonds can add if structures are thick and the dock is strong in tension: a block's bending capacity grows
   with its cross-section. The dock then has to be the strongest part (the Jenett rule), not the weakest.
1. Actuator torque can add at one joint only through a mechanical split (a differential or shared shaft with
   torque-controlled motors), never through several position servos.
1. Forms that keep load in the ground or in braces (crawling, trusses, props) avoid the problem.

## Gaps

1. No small (< 75 mm) module with measured additive strength.
1. No self-reconfiguring arm where every module's motor adds to the base moment; the nearest are
   tendon-driven continuum arms with base motors, which do not reconfigure.
1. Not read: Spinos and Jenett theses, ARMADAS strength data, the Campbell and Pillai full text, force
   numbers for Odin, Telecube, Crystalline and Tetrobot.
