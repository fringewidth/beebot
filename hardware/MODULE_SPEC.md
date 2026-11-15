# Module spec: one reusable component for most robot tasks

Status (2026-11-15): nothing in this document is built, designed in CAD or simulated. It is a
requirements spec: the properties the module must have, with target numbers derived from tasks.
Every number is marked **derived** (computed here, formula shown), **source** (read from a cited
page) or **guess** (a choice to be checked). The mechanism comes after this spec is agreed.

Hard constraint (user, 2026-11-15): one indivisible component. Every robot is the same component
arranged differently: positions, rotations, order. No second part type, no add-on wheels,
grippers or batteries.

## 1. What tasks, and what share they are

Most used robot tasks by units sold:

| market | application | share | source |
| --- | --- | --- | --- |
| industrial robots, installations 2022 | handling (pick and place, palletising, machine tending, packing) | about 48 % | 266k of 553k units (source, secondary; share derived) |
| | welding | about 16 % | 87k (source, secondary) |
| | assembly | about 11 % | 61k (source, secondary) |
| | the rest (dispensing, painting, processing, cleanroom, other) | about 25 % | remainder (derived) |
| professional service robots, sales 2025 | transportation and logistics | 47 % | IFR World Robotics 2026 (source) |
| | hospitality (serving, delivery indoors) | 14 % | IFR presentation (source, secondary) |
| | professional cleaning (mostly floors) | 12 % | IFR (source) |

What a bare module can do and what it cannot:

1. **Covered by arrangement alone:** handling, transport and delivery, assembly motions
   (insert, press, turn a screw that fits a face), holding and fixturing, inspection (carry a
   sensor), structures (bridge, ramp, shelf, wall), opening doors and drawers, pushing.
1. **Not covered without a tool:** welding, painting, dispensing, machining, vacuuming. These need
   a process tool (torch, nozzle, spindle, brush).
1. By the shares above, arrangement alone covers about 60 % (handling + assembly ~ 59 % of
   industrial; transport + hospitality ~ 61 % of professional service) (derived). **The 80 % target
   is only reachable if the connector is also an open tool interface** (section 5, property 5.12),
   so a third-party tool can be held by a module the way a module holds another module. The tool is
   not part of the module; the interface is.

### Reference tasks (the coverage set)

Each is a configuration the module must be able to form. They drive the numbers below.

| # | task | stands for | configuration (estimate, to be checked in MuJoCo) |
| --- | --- | --- | --- |
| T1 | pick an object of 50-150 g from a table, place it 30 cm away | handling | 4-DOF arm + 2-finger gripper, about 8-12 modules |
| T2 | carry 300 g across a room, turn, stop at a mark | transport | 1-4 modules rolling (property 2.4) |
| T3 | open a drawer, push a door | domestic manipulation | T2 base + 2-3 DOF arm |
| T4 | press a peg into a hole, turn a knob | assembly | T1 arm with a wrist roll and force sensing |
| T5 | walk over a 1-module step | legged locomotion on rough floor | quadruped, about 12-16 modules |
| T6 | crawl through a gap 1.5 modules high | inspection | snake, 4-8 modules |
| T7 | move objects along a line | conveying, sorting | a row of modules as rollers |
| T8 | span a 3-module gap, hold a 200 g load | structure | bridge, about 8-12 modules |
| T9 | hold a part still while another arm works on it | fixturing | two short arms or a clamp |
| T10 | assemble T1 from loose modules on the floor | self-reconfiguration | modules drive and dock |

## 2. Degrees of freedom

Every robot in section 1 is rigid links plus revolute joints (bounded for arms and legs, unbounded
for wheels and tools) plus a grip. Straight-line joints are rare and are built from revolute ones.
Two joint types appear:

- **twist:** the axis runs along the links on both sides (wrist roll, base yaw, wheel axle);
- **bend:** the axis is perpendicular to both links (shoulder, elbow, knee, finger).

A twist joint plus one offset module makes a bend: links on the module's side face and on a
module carried by its disc both lie perpendicular to the axis, and the angle between them changes
(a UR-style offset elbow). So twist alone covers every joint type, at a cost of one extra module
per bend.

### Options

| layout | motors | lone module moves? | bend costs | wrist (2 axes) costs | mechanism risk |
| --- | --- | --- | --- | --- | --- |
| A. one twist face | 1 | no | 2 modules | 2-3 modules | low (designed once in `cube_full.py`) |
| **B. two twist faces, opposite, each its own motor** | 2 | **yes, rolls on the two discs** | 2 modules | 2-3 modules | low-medium: one axis, two drives |
| C. two twist faces, adjacent | 2 | no | 2 modules | 1 module | medium: two bearing axes in one cube |
| D. twist + bend in one module (M-TRAN, SMORES tilt) | 2 | no | 1 module | 1 module | high: the cube changes shape while bent; halves collide |
| E. 3-4 DOF (SMORES-EP: two wheels, pan, tilt) | 3-4 | yes | 1 module | 1 module | high, and SMORES is 80 mm, 500 g |

**Recommendation: B**, to be confirmed by the coverage check (section 11).

1. A lone module is a two-wheeled robot: its two discs are its wheels, an edge of its body is its
   skid (property 3.5). That gives transport (T2, 47 % of service robots) from one module, and lets
   loose modules gather to dock (T10) without anyone carrying them.
1. Both discs share one axis, so there is one bearing line, both motors lie along it, and both
   slip contacts (property 6.3) have the same form.
1. Between its two discs the module is a twist joint (relative angle of the discs); between a side
   face and a disc it is a single twist joint; with an offset module it is a bend.
1. Cost: an elbow takes two modules. Arm torque grows with the square of the module count
   (property 4.1), so arms of B are shorter than arms of D for the same motor. This is the main
   trade and the reason the coverage check must count modules per task.

## 3. Shape and size

1. **3.1 Space-filling:** the module tiles 3D space so structures (T8) are solid and the voxel
   policy's grid still applies. Cube (recommended), or rhombic dodecahedron (12 neighbours,
   awkward faces). Spheres and cylinders do not tile.
1. **3.2 Symmetric faces:** every face mates with every face of another module (disc to disc,
   disc to side, side to side) in each of 4 relative quarter turns. A face pattern must map onto
   itself under 90-degree rotation and under mirroring.
1. **3.3 Self-aligning:** chamfers or cones guide a mating face in from the capture range
   (property 5.4); no fine positioning needed to dock.
1. **3.4 Size:** as small as the joint torque allows (property 4.1). Design point **50 mm**, stretch
   **40 mm** (guess). Below about 40 mm the bought parts (gearmotor 12 x 10 x 24 mm, controller
   board 18 x 23 mm, battery) no longer fit two motors (guess, to check in CAD).
1. **3.5 Rolling profile:** each disc's rim stands at least 1 mm proud of the body radially, so a
   lone module stands on its two rims with its body clear of the floor. The body can tilt about
   2 degrees before an edge touches the floor and becomes the skid (derived: edge at 35.4 mm
   from the axis, rim at 26 mm, asin(26 / 35.4) - 45 = 2.3 degrees). **Open conflict:** a rim proud of the body hits neighbours on the
   adjacent side faces. Candidate fix: the side faces' connector plates stand 1 mm proud as well,
   so the lattice pitch is set by the plates and the rims sit in the gap.
1. **3.6 Mass:** at most 120 g at 50 mm, 70 g at 40 mm (guess, scaled by volume). Every gram
   costs arm reach (property 4.1).
1. **3.7 Grip surface:** the face around the connector has a high-friction rim (rubber, mu >= 0.8,
   guess) so two modules can pinch an object that is not a module.
1. **3.8 Nothing sweeps outside the cell:** a turning disc stays inside the module's outline at
   every angle, so a joint can move while its neighbours are packed around it.

## 4. Joints (each of the two twist drives)

1. **4.1 Torque:** a horizontal arm of n modules of mass m and side L, with a payload m_p at its
   tip, needs at its base about tau = m g L n^2 / 2 + m_p g n L (derived).

   | L, m | n = 2 | n = 3 | n = 4 |
   | --- | --- | --- | --- |
   | 50 mm, 120 g, payload 100 g | 0.22 N m | 0.41 N m | 0.67 N m |
   | 40 mm, 70 g, payload 100 g | 0.13 N m | 0.24 N m | 0.38 N m |

   Target: **continuous 0.4 N m, holding 0.8 N m** at 50 mm (a 3-module reach with 100 g). An N20
   gearbox is rated about 0.2 N m (source: Pololu 298:1 HP, read 2026-11-14 in
   `runs/handoff/rotating-cube-poc.md`), so the drive needs an extra stage or a worm. This is the
   first feasibility check.
1. **4.2 Speed:** at least 60 degrees per second at rated torque for joints (guess); at least
   0.15 m/s rolling, i.e. about 55 rpm on a 52 mm disc (derived). One gear ratio may not give
   both; a two-speed drive is out of budget, so pick the ratio from the coverage check.
1. **4.3 Unbounded rotation:** both discs turn without end stops (wheels, screwing, conveyor).
1. **4.4 Holds pose with power off:** a structure (T8) or a held load (T9) costs no energy, and
   a dead battery drops nothing. A worm stage or a brake gives this.
1. **4.5 Absolute angle sensing:** the module knows its disc angle at power-up (magnetic encoder)
   with at most 0.5 degree error (guess), so there is no homing move inside a packed structure.
1. **4.6 Torque sensing:** motor current read at the controller, so the module can grip with a set
   force (T1), feel contact (T4) and detect a stall.
1. **4.7 Quarter-turn index:** the disc can be locked at 0, 90, 180 and 270 degrees so a bond made
   there lines up with the lattice. The lock is not needed while the joint is in use.
1. **4.8 Stall is harmless:** a joint that cannot reach its target stops at a current limit and
   can drive back; nothing breaks or drops (`cube-reliability-bar` rule).
1. **4.9 Moment capacity:** the disc bearing carries the full bending moment of the arm hanging
   off it (0.8 N m, property 4.1) without tilting more than 1 degree (guess).

## 5. Connectors (all six faces)

1. **5.1 On every face**, discs included, so any face can bond in any arrangement.
1. **5.2 Genderless:** any face mates with any face.
1. **5.3 Holds with zero power, switches with a short pulse or move.** Bonds last hours; current
   only flows while switching.
1. **5.4 Capture range:** pulls in from at least +/- 3 mm lateral and +/- 5 degrees (derived: 1
   degree of joint backlash at a 150 mm reach is 2.6 mm; FDM parts are about +/- 0.2 mm each).
1. **5.5 Strength** (derived from 5 modules hanging and the arm moment, safety factor 2-2.5):
   tension >= 15 N, shear >= 10 N, bending moment >= 0.8 N m, torsion >= 0.8 N m. The old locked
   face reached 1.33 N m bending (derived, rotating-cube-poc); the old disc only 0.37 N m.
1. **5.6 Release from either side alone:** a module can let go of a neighbour even if the
   neighbour is dead, so a failed module can always be removed.
1. **5.7 Bond state is sensed:** each face reports open, closed or partly closed (Hall sensor or
   switch). No bond is assumed.
1. **5.8 Neighbour identity:** each face reads which module and which of its faces it touches,
   and at what quarter turn (data line through the face, property 6.4).
1. **5.9 Power and data through the face:** concentric contacts (centre pin, ring), which mate at
   any rotation and, on a disc, also work as a slip ring (guess: two contacts power, one data).
1. **5.10 Cycle life:** at least 10,000 bond-release cycles (guess; Soldercubes failed at 220).
1. **5.11 No dedicated motor per face.** Six face motors do not fit a 50 mm cube
   (`physics-revision` notes, 2026-11-14). Options to evaluate: electropermanent magnets bought,
   not hand-wound; one actuator switching all faces with a selector; shape-memory latches.
1. **5.12 Open tool interface:** the face standard is documented so a tool can be built to it
   (property in section 1). This is how the module reaches 80 % without a second part type.

## 6. Power and electronics

1. **6.1 Own battery** in every module (indivisible: no battery modules). At least 30 minutes of
   typical use: 0.6 W average x 0.5 h = 0.3 Wh, about 80 mAh at 3.7 V; take 150-200 mAh (derived).
1. **6.2 Peak current:** two motors at stall (about 1.6 A each for an N20 HP at 6 V, source as in
   4.1) plus a connector switch pulse: at least 4 A peak. The cell must be rated for it, or a
   capacitor bank covers the pulse (derived).
1. **6.3 Power shared through faces:** a bonded assembly shares one bus, so a module with a flat
   battery keeps working and a whole robot charges from one point.
1. **6.4 Charging** through any face (a charging pad is a face), no cable to plug in.
1. **6.5 Battery protection** and a low-voltage state that first finishes the current safe stop.
1. **6.6 Controller:** joint loops at 500 Hz or more on board (guess); firmware updated over the
   air; same firmware in every module.

## 7. Communication and knowing the arrangement

1. **7.1 Wireless** to the host (ESP-NOW or BLE) for commands and telemetry.
1. **7.2 Face-to-face data** for topology: the assembly reports its own graph (which module, which
   face, which quarter turn), so a configuration is never typed in by hand.
1. **7.3 Synchronised joints:** all joints in an assembly of up to 30 modules start a commanded
   motion within 10 ms of each other (guess).
1. **7.4 Identity:** a unique ID per module, readable by the camera too (a mark on each face) for
   the external tracking the policy uses now.

## 8. Sensing

1. Joint angle and current per drive (4.5, 4.6).
1. Bond state and neighbour per face (5.7, 5.8).
1. 6-axis IMU: orientation of the module and of the assembly, falls, slope.
1. Optional, decided by cost: a distance sensor on some faces for docking and obstacles; a camera
   is not in the module (external camera, as now).

## 9. Reliability

Raw success compounds: a task of K operations each succeeding with p succeeds with p^K (derived).

| task | operations K (estimate) | p needed for 95 % task success |
| --- | --- | --- |
| T1 pick and place | ~30 joint motions | 0.998 |
| T10 assemble 12 modules | ~12 docks + ~40 motions | 0.999 |

1. **9.1 Every operation is verified by a sensor** (angle reached, bond closed, neighbour seen).
1. **9.2 Every failure is detectable and reversible**, so it can be retried. With 2 retries a raw
   p of 0.95 gives 1 - 0.05^3 = 0.99988 per operation if failures are independent and harmless
   (derived). This is why 9.1 and 9.2 matter more than raw p.
1. **9.3 Targets, measured on the bench:** dock (verified) >= 0.997 over 1,000 attempts; joint
   motion to target >= 0.999 over 10,000; zero harmful outcomes (drop, break, knocked neighbour)
   in those runs. Also meets the earlier per-move bar (>= 0.80 reached, <= 0.03 harmful) by a
   wide margin.
1. **9.4 Fail safe:** loss of power, radio or firmware leaves every bond closed and every joint
   held (4.4, 5.3).
1. **9.5 Wear:** no friction-only drive (EM-Cube's moves swung from 0.24 to 0.96 with the
   surface, HARDWARE.md); positive engagement only.

## 10. Build, cost and service

1. **10.1 Parts bought** from Indian retail or a PCB house, each with a datasheet link and size
   checked against it. No part made by hand in counts above two per module.
1. **10.2 Printed parts** on a hobby FDM printer, clearances >= 0.3 mm, no supports on running
   surfaces.
1. **10.3 Assembly:** a written order, screw access for every screw, at most 30 minutes per
   module (guess). Battery and motors replaceable without desoldering.
1. **10.4 One motor type, one fastener size**, so ten modules need short part lists.
1. **10.5 Cost per module:** open (decision below). The earlier budget was ₹10,000 for ten
   (memory from 2026-11-08, may be out of date); the old 50 mm cube was about 126 USD a module
   (guess).
1. **10.6 Safety:** current limits on every drive, no pinch point that closes on a finger with more
   than the rated torque, LiPo protection.

## 11. Acceptance: how the spec is checked, in order

1. **Coverage check (CPU, no hardware):** build T1-T10 from layout B (and A, C for comparison) in
   MuJoCo with the module's mass, size and torque. Record module count, peak joint torque and
   connector moment per task. Pass: every task formed with joint torque <= 0.4 N m at 50 mm.
1. **Drive bench:** one disc on a bought gearmotor plus extra stage: torque, speed, holding with
   power off, backlash.
1. **Connector bench:** two faces: capture range, strength (5.5), 1,000 docks, cycle life.
1. **Two modules:** a two-module arm lifting a third module, 10,000 motions.
1. Only then the whole module in CAD, with every part from 10.1.

## 12. Decisions owed by the user

1. DOF layout: B (recommended), or another from section 2.
1. Size: 50 mm design point with 40 mm stretch, or a hard limit.
1. Cost cap per module.
1. Self-reconfiguration level: hand-assembled only; assemblies attach and detach their own
   modules; or lone modules also drive to dock (T10, recommended, comes with B).
1. Whether the open tool interface (5.12) counts toward the 80 %.

## Sources

1. [IFR World Robotics 2026, service robots](https://ifr.org/ifr-press-releases/news/world-robotics-2026)
1. [Transportation and logistics account for 47 % of professional robot sales](https://www.supplychain247.com/article/transportation-logistics-professional-robot-sales/supply_chain_management)
1. [Global robot demand doubles over 10 years (2022 installations by application)](https://www.assemblymag.com/articles/99618-global-robot-demand-doubles-over-last-10-years)
1. [IFR executive summary, World Robotics 2025, industrial robots](https://ifr.org/img/worldrobotics/Executive_Summary_WR_2025_Industrial_Robots.pdf)
