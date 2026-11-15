# Smallest self-powered cube: actuation concepts and scaling (2026-11-15)

Status: nothing is built, bought or simulated dynamically. This is an analytic scaling study:
`scripts/module_phys/scaling.py` (about 2 minutes on the Mac CPU, no GPU) writes
`runs/module_phys/scaling.json`, `runs/module_phys/scaling_margins.png` and
`runs/module_phys/scaling_stdout.txt`. The envelope check of the top pick is
`hardware/module_phys/rw_cube_envelope.py` (writes `hardware/module_phys/rw_cube_7mm_envelope.step`).
Nothing was committed, nothing was written to `ops`, nothing was bought.

Every number carries a tag: **V** (read on a vendor page or paper this session, sometimes via a search
snippet, said where), **E** (estimated from stated physics), **A** (assumed: a choice or a number from
memory, to be measured). Parts come from `runs/module_phys/parts.json`, where 71 of 79 entries are
unverified; those uncertainties pass straight into the margins below.

## 1. Answer in one paragraph

No design reaches 5 mm. The smallest cube that does every swarm behaviour alone and with neighbours
(self-right from any face on a bare floor, roll, turn, dock, split, climb, build a 2x2x2, survive a
20 cm fall, run alone for 2 minutes) is a **reaction-wheel cube of about 7 mm** (6.75 mm at the
model's limit): two flywheels printed into the face flex as motors and caught by small pawls, eight
free N52 corner spheres as genderless bonds and hinges, three TDK CeraCharge cells, a CH32V003. It
needs flywheels at about 78,000 rpm for climbs, which needs jewel bearings; with plain pin bearings
(6,000 rad/s) the same design closes at **11 mm**. Neither does the arm, the walker or the crank:
every joint that has to hold torque against gravity (spec 4.1) needs a gear reduction. A face motor
across the bond reaches only 0.18-0.25 of the arm's torque and 0.10-0.13 of the walker's, at every
size from 3 to 50 mm. The only design that passes **every** behaviour, machines included, is the
**geared hobby baseline at 40 mm** (N20 gearmotors plus a printed 3:1 stage, hook latches still
undesigned). So the user has a real choice to make: very small cubes that reconfigure but cannot be
joints, or about 40 mm cubes that can. Section 8 lists what would close the gap.

## 2. What was modelled

### Behaviours and how each is scored (all scaled to L through the cube mass m and L)

| behaviour | requirement (margin = available / required, pass at 1.5) |
| --- | --- |
| lone self-right, lone translate | tip about a bottom edge from any face: energy (sqrt2 - 1)/2 m g L + capillary adhesion; inertial designs get an internal angular impulse, with the edge slipping at mu 0.3 (efficiency 0.87, derivation below) |
| lone turn | impulsive yaw of up to 45 deg against mu 0.8 friction (torque mu m g 0.383 L) |
| split | peel one face bond (energy of the far sphere pair) plus a floor roll |
| climb | peel plus 180 deg over a neighbour's edge (COM rises 1.207 L); the edge hinge has to hold the centripetal and outward gravity load at every angle of the swing |
| bonds | moment for a 3-cube cantilever x 2 dynamic factor (spec 4.1/5.5), tension for 5 hanging cubes x 2 |
| dock | lateral capture by the magnets against mu 0.8 friction at least 0.10 L (assumed placement error) |
| lone run | 2 minutes, one roll every 4 s, 1 mW idle (A); storage chosen by energy, continuous and pulse power, each with 1.5x |
| lone speed | at least 0.1 body lengths per second (A, soft) |
| fall | 20 cm onto a hard floor on a corner: Hertz contact (corner radius 0.05 L, E* 2.8 GPa), peak deceleration, stress on the heaviest part's mount |
| machine twist | continuous joint torque of layout B's arm lift (0.42 N m at 50 mm, 120 g), rescaled by m L |
| packing | (inner volume x best-case fill) / sum of part envelopes >= 1.5; best-case fill 0.80 for parts on a folded flex (phone-class), 0.65 for hand-assembled hobby parts, so 0.53 and 0.43 effective |
| peak current | the cell's pulse power against one motor starting at stall |

### Physics checked against published data

1. **Magnets.** Exact dipole force for spheres (a uniform sphere's field is exactly a dipole) and the
   exact Coulombian formula for coaxial cylinders (Bessel integral), with the image method for steel.
   Against four published pull values: K&J D22-N52 (3.74 N published, 5.57 N model), supermagnete
   S-02-0.7-N (0.73 / 1.14), S-05-01-N (3.14 / 4.59), supermagnete K-03-C 3 mm sphere (1.27 / 1.58).
   The model overpredicts by a steady factor: published/model 0.64-0.69 for discs, 0.80 for the sphere.
   All magnet forces in the model are multiplied by these factors (0.66 cylinders, 0.80 spheres). The
   simulation-stack agent found the same bias independently (image method overpredicts vendor ratings
   by 13-36 %, `runs/module_phys/simstack.md`). Values are V, secondary (via search snippets).
1. **Electromagnets do not get better when small.** At a fixed current density J, a coil's
   ampere-turns go as J L^2, its gap field as J L, its pull as J^2 L^4, the same as weight times L. So
   ElectroVoxel-style pivoting gets *worse* as the cube shrinks. Mid-swing margin in 1 g (E): 0.19 at
   5 mm, 0.38 at 10 mm, 0.76 at 20 mm, 2.3 at 60 mm but at 2.4 kW per coil pair. ElectroVoxel itself
   (60 mm, 103 g, 800-turn ferrite coils at 1.06 A and 11.7 W each) pivoted only on an air table and in
   microgravity (V, arXiv 2207.05732). "Magnetic forces go as L^2" is true of **permanent** magnets
   only, and that is what the top designs use.
1. **Permanent magnets dominate at mm size.** Two 1 mm N52 spheres touching pull about 0.18 N (E,
   calibrated), 50 times the weight of a 0.35 g, 5 mm cube.
1. **Surface forces.** Worst case capillary adhesion (humid, wetting) on four corner feet of radius
   0.1 mm is 4 x 4 pi gamma r = 0.36 mN (E), 11 % of a 5 mm cube's weight and 6 % of a 7 mm cube's;
   it scales as r while weight scales as L^3, so below about 3 mm it controls everything. It is added
   to every tipping load. Friction coefficients are size-independent at this scale (A); adhesion adds
   to the normal load.
1. **Ballistic rolls are forgiving.** A square block landing flat keeps r = 1 - 1.5 sin^2(45 deg) =
   0.25 of its angular velocity (Housner 1963), so 6 % of its energy, and rolls on only if that exceeds
   the tip barrier: a floor roll succeeds for any start energy from 1x to 16x the barrier (E). The
   M-Blocks reliability problems were brakes, edge slip and magnets, not the energy window.
1. **Impulse with a slipping edge.** For an internal impulse H about a bottom edge with floor
   friction mu < 1, solving the rigid-body impulse equations (vertical edge velocity zero, friction
   impulse mu times normal) gives omega = H / (I_cm + (1 + mu) m L^2 / 4). After subtracting the
   sliding energy lost at the edge, the energy left for tipping is 0.87 (mu 0.3), 0.95 (0.5) and 0.99
   (0.8) of the no-slip value (E).
1. **Falls.** Impact speed from 20 cm is 2.0 m/s at any size (terminal velocity of a 5 mm, 0.35 g
   cube is about 12 m/s). Impact energy 0.68 mJ at 5 mm, 1.7 mJ at 7 mm, 127 mJ at 40 mm. Peak
   deceleration grows as 1/L (8,200 g at 5 mm, 6,150 g at 7 mm, 1,540 g at 40 mm), but the stress on a
   part's mount, part mass x deceleration / mount area, is nearly size-independent: 17-22 MPa on the
   flywheel pin in a resin bushing (allowable 40 MPa, A), margin 2.0 at 7 mm. A tower of 10 cubes of
   7 mm is 7 cm, so 20 cm covers a fall from anything the cubes build.

## 3. Architectures

Each was asked: how does a **lone cube on a bare, non-magnetic floor, resting on any face**, get up,
move, turn, come back and dock, and how does it pair, split, climb and build a 2x2x2.

| # | architecture | lone cube from any face | min L, all behaviours | min L, all but load-holding joints | binding constraint | cost per cube at 10 / 1k / 1M (USD, A) | main risks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RW-P-20k | **reaction-wheel cube, flex-PCB flywheel motors, jewel bearings (to 20,000 rad/s), corner spheres, CeraCharge** | rolls by pawl-caught flywheel impulse in 4 directions; yaw by the vertical flywheel; any face | none (no joint torque) | **6.75 mm** (pick: 7 mm) | below 6.75: packing 1.40; at 6.75: climb hinge 1.50 | 24.8 / 8.0 / 1.8 | ballistic moves; 78k rpm bearings; pawl wear; hand placing 8 spheres of 1 mm; CeraCharge supply |
| RW-P | same with pin-in-bushing bearings (6,000 rad/s) | same | none | **11 mm** | below 11: spin-up time (1.28), lone energy | 14.8 / 4.3 / 0.9 | as above, but a 301010 LiPo fits and ordinary bearings |
| G-N20 | **geared layout C made real**: N20 for wheel, twist disc and a pivot selector that also swings a kick leg and drives hook latches; 150 mAh whoop cell | kick leg tips it (geared, quasi-static); wheel drives; skid turn | **40 mm** | 40 mm | packing 1.40 at 38 mm | 17.1 / 8.8 / 3.1 | hook latches and the selector are undesigned (assumed strong); 3 motors; wheel lean (retract it) |
| G-0615 | same with 6 mm planetary gearmotors | kick leg | none | none | 24-30 mm: packing and climb; above 36 mm the leg is too weak; arm twist 0.09-0.57 | 13-15 (est.) | gearbox ratings on listings are unreliable |
| G-0408 | same with 4 mm planetary | kick leg | none | none | arm twist 0.03-0.42, climb fails above 24 mm | 12 (est.) | fragile gearbox |
| G-BMN04 | same with Orbray BMN04-0829P337 (4 x 18.4 mm, 5.44 mN m stall, V) | kick leg | none | none | arm twist 0.04; too long below 20 mm | about 200 at 10 (A, inquiry only) | no price, no India route |
| RW-P-T* | RW-P plus a geared twist face (0408, BMN04, 0615, 0820, N20 + 3:1) | as RW-P | none | none | the gearmotor does not fit where its torque would suffice: at 12 mm BMN04 would give 1.1x the arm torque but is 18.4 mm long; at 30 mm N20 fits but the flywheel spin-up fails | - | - |
| RW-B | flywheels on bought motors (0408 coreless; Orbray BMN02 2.4 x 5 mm) | as RW-P | none | none | BMN02 tops out at 6,000 rpm (V): 0.2-0.6 of the roll impulse; 0408 coreless fails climbs (0.3-1.0) and spin-up | - | - |
| RW-P-W | RW-P plus a coil-driven retractable wheel | wheel when wheel face down, flywheel otherwise | none | none | a direct-drive mm wheel gives 0.3-0.5 of the traction-limited torque | - | extra parts for a small speed gain |
| EV | ElectroVoxel edge electromagnets | **cannot**: nothing to push on | none | none | lone gates fail by construction; 1 g pivot 0.19 at 5 mm and worse when smaller | - | power (W per coil) |
| VIB | vibration or coil stick-slip only (Kilobot style) | translates and turns; **cannot tip** | none | none | self-right fails by construction | - | direction control |
| PZ | piezo stick-slip | translates; cannot tip; needs 30-150 V | none | none | self-right | - | - |
| ES | electrostatic zipping or adhesion | 1-6 kPa at 200 V; cannot tip | none | none | self-right | - | humidity |
| SMA | SMA kick flaps (12) and SMA latches | tips and rolls but **cannot change yaw**; 24 crimped terminations per cube | none | none | lone turn | - | assembly at mm size |
| EPM (prose only) | electropermanent face bonds instead of permanent ones | needs flywheels anyway | not modelled | - | switching a mm-scale AlNiCo rod needs about 150 kA/m: 100s of ampere-turns, hand-wound coils on every face (spec 10.1 forbids), and a thin AlNiCo disc cannot hold its magnetisation | - | - |

Plot of all margins against L for four of these: `runs/module_phys/scaling_margins.png`.

### Lone-cube mobility, compared

| mode | lone speed | turning | self-right | success-rate risk |
| --- | --- | --- | --- | --- |
| inertial (flywheel + pawl) | 4.5 mm/s at 7 mm (0.65 L/s), set by the CeraCharge's 4.2 mW: 4.9 mJ a roll; 11 mm/s at 11 mm on a LiPo | impulsive yaw, only from the 4 faces on which one flywheel axis is vertical; from the other 2, roll first | from any face, in 4 directions | roll window 1x-16x energy (forgiving); climbs are not (the hinge and the landing magnets must catch the cube); pawl wear |
| retractable wheel (Orbray 1.5 mm geared, or coil-driven) | faster when it works (direct drive torque 0.3-0.5 of traction need) | needs the flywheel anyway | none of its own | adds a retract actuator; the lean problem of layout C returns while the wheel is out |
| stick-slip or vibration | Kilobot does 1 cm/s at 33 mm (0.3 L/s) | yes (two actuators) | none | surface-dependent |

Pairs: in the reaction-wheel designs a pair is not faster than a lone cube (a bonded pair rolls about
its long edge one cell at a time). "Pairs travel fast" is met only by the geared design (G-N20).

## 4. Why each design stops where it does

1. **RW-P-20k below 6.75 mm: packing, energy and climb impulse.** At 5 mm (0.35 g): packing 1.41,
   lone energy 0.22 (one CPH3225A supercap gives 7 rolls; a CeraCharge, 4.5 mm long, does not fit a
   4.2 mm cavity), climb 1.22 (it needs 157,000 rpm). At 5.75 mm it passes if a lone run of 30 s is
   enough (sensitivity table in `scaling.json`).
1. **The flywheel bearing sets the size.** Climbs need omega ~ L^-0.5 (E): 78,000 rpm at 7 mm,
   48,000 rpm at 10 mm. With 6,000 rad/s pin bearings the minimum is 11 mm, with 10,000 rad/s 7.5 mm,
   with 20,000 rad/s 6.75 mm.
1. **Joint torque is the wall for machines.** The arm and walker torques go as m L ~ L^4. A face motor
   across the bond (coils in one cube's face, the neighbour's face magnets as rotor) gives a shear
   stress of 1.2-1.7 kPa at steady state (air-core copper, 2 layers 1 oz, rotor field 0.2-0.3 T) at
   every size:

   | L | arm lift need | face motor | margin | walker need | margin | crane | conveyor |
   | --- | --- | --- | --- | --- | --- | --- | --- |
   | 5 mm | 121 uN m | 25 uN m | 0.21 | 231 uN m | 0.11 | 0.44 | 1.76 |
   | 6.75 mm | 382 uN m | 70 uN m | 0.18 | 727 uN m | 0.10 | 0.38 | 1.54 |
   | 10 mm | 1.0 mN m | 0.25 mN m | 0.25 | 1.9 mN m | 0.13 | 0.52 | 2.08 |
   | 50 mm | 0.21 N m | 0.036 N m | 0.17 | 0.41 N m | 0.09 | 0.35 | 1.42 |

   It turns a conveyor roller and lifts about one cube cantilevered (n_max = 1.0-1.2 from
   tau = m g L n^2 / 2). Worse, it conflicts with the bond: the same face magnets give a magnet-to-
   magnet detent of the order of the bond pressure (20+ kPa), ten times what the coils can push, so
   the joint cannot step out of its own detent unless the stator face carries no magnets, and then that
   face is no longer a genderless bond. The face motor is not adopted.
1. **G-N20 below 40 mm: packing.** Three N20s, a 6 x 12 x 30 mm cell, an ESP32-C3 board and the
   selector need about 38 mm inside. With the extra 3:1 printed stage the N20 gives 2.3x the arm torque
   at 40 mm.

## 5. Pareto front (L, cost, worst-behaviour reliability, lone speed, joint torque)

| design | L | cost at 10 / 1M | worst behaviour (risk) | lone speed | joint torque / arm need |
| --- | --- | --- | --- | --- | --- |
| RW-P-20k | 7 mm | 24.8 / 1.8 | ballistic climb (risk high) | 0.65 L/s | 0 (no joint) |
| RW-P | 11 mm | 14.8 / 0.9 | ballistic climb (risk high) | 1.0 L/s | 0 |
| G-N20 | 40 mm | 17.1 / 3.1 | undesigned hook latch (risk medium), quasi-static moves | wheel, about 3 L/s | 2.3 |

All three are non-dominated: each is best on at least one objective. Every other row above is
dominated or fails a gate. Reliability here is a qualitative ranking; the numbers have to come from
MuJoCo outcome runs and the bench (sections 9 and 10).

## 6. Top picks and layouts

### Pick 1: RW-P, the 7 mm reaction-wheel cube (smallest that does every swarm behaviour)

Envelope check (`hardware/module_phys/rw_cube_envelope.py`): every part below fits with no overlap in
the 6.2 mm cavity, using 85 of 238 mm^3. Not counted there: pins, bushings, flex folds, solder.

Face by face (local frame; the flex PCB is the skin of all six faces, folded around a printed resin
frame of 0.3 mm bars along the 12 edges with a spherical pocket at each corner):

| face | flex layer content | behind it, inside |
| --- | --- | --- |
| +x | 3-phase stator of flywheel X (3 sector coils) | flywheel X: diametric N52 core radius 1.05 mm, brass or tungsten rim to 2.1 mm, 0.75 mm thick, on a 0.5 mm steel pin in jewel bearings, 0.1 mm over the coils; pawl X at its rim |
| +y | stator of flywheel Y | flywheel Y, pawl Y (the two rotors clear each other: each stays below 2.25 mm from the shared edge) |
| -z | charge pads (the cube charges on a pad face down) | three TDK CeraCharge 1812 stacked (4.4 x 3.2 x 3.4 mm) |
| -x | CH32V003F4U6 (QFN-20 3 x 3) | - |
| -y | LIS2DW12 accelerometer (2 x 2) | - |
| +z | boost converter 1.4 V to 3.3 V, 6 half-bridges (12 SOT-723 FETs or 3 DRV8837), 16 Schottky diodes 0201, pawl drivers | - |
| every face | an IR LED and photodiode (0402) at the centre for neighbour and host data | - |
| every corner | - | a free N52 sphere of 1.0 mm (model asks 0.92 mm), touching the neighbour's sphere through the corner opening |

1. **Mass** 0.85 g (E): rotors 2 x 156 mg, spheres 32 mg, cells 120 mg, flex and frame about 0.3 g.
1. **Flywheel motor coils** (E): rotor radius 2.1 mm, stator annulus 0.63-2.1 mm, 3 sector coils at
   120 deg, 2 layers, 75 um trace and 75 um space (JLC flex 3/3 mil), about 8 turns per layer per
   coil, 16 per coil; mean turn 5.8 mm, so about 0.6 ohm per coil. At the 10 mW spin budget: 0.13 A,
   4.0 uN m, spin-up to the climb speed in 0.8 s, roll impulse in about 0.3 s. Motor constant
   4.0e-5 N m / sqrt(W). Back-EMF at 78,000 rpm about 0.25 V, so the motor runs from the 1.4 V cell
   through the H-bridges with no boost.
1. **Pawls:** a 0.5 mm N52 cube on a flexure, flipped by a 10-turn flex coil, drops into a notch
   in the rotor rim; the rotor stops in about 50 us and its angular momentum passes to the cube.
1. **Moves** (E): roll 4.9 mJ, yaw 0.9 mJ, split 26 mJ, climb 43 mJ, each from the flywheel stopped
   by the pawl (energy is lost in the catch; the cube needs H, not energy). Flywheel speeds: roll about
   26,000 rpm, climb 78,000 rpm.
1. **Bonds** (E, calibrated): 4 sphere pairs a face, tension 0.27 N (33 cube weights), peel moment
   0.80 mN m (1.53x the 3-cube cantilever with a dynamic factor of 2), peel energy 51 uJ, lateral
   capture 2.2 mm (0.32 L). The hinge holds the whole climb with 1.5x. The free spheres make every
   face genderless at every quarter turn; a fixed pattern would need at least 8 magnets a face (8 on a
   ring at 22.5 deg + k 45 deg, alternating, is the smallest fixed genderless pattern, E).
1. **Power through bonds:** the corner spheres are nickel-plated conductors. Each corner goes through
   two Schottky diodes to the rails, so a bonded neighbour can feed power at any orientation and
   polarity (corner parity alternates, so fixed polarities would short). A chain fed from a power bank
   recharges its cells; data goes over IR face to face.
1. **Lone run:** 1.5 J in three cells, 168 rolls, at most 4.2 mW continuous (V, TDK: 1 mA continuous,
   3 mA for 1 s at 1.4 V), so a roll every 1.5 s: 4.5 mm/s, about 6 minutes alone.
1. **Fall:** 20 cm gives 1.7 mJ, 6,150 g on a corner, 20 MPa on the flywheel pin (allowable 40 MPa, A).
1. **Cost** (A): at 10 cubes 24.8 USD (flex PCBA with setup about 7, three CeraCharges 12, rest 6);
   at 1k 8.0; at 1M 1.8, of which the cells are 1.2. Without cells, chain-powered only, about 0.6 USD
   at 1M. The 10 USD target at 10 units is not met because of the cells and the PCBA setup.
1. **Why not 5 mm** (E): it needs a 7-roll supercap instead of the cells, 157,000 rpm for a climb,
   and packing fails (1.41). A 5 mm cube that can only roll and dock (no climbs, chain power only) is
   closer: roll margin 3.95 at 5 mm.

### Pick 2: RW-P at 11 mm with plain bearings (same concept, lower risk, buildable first)

Same faces; flywheels on 0.5 mm pins in resin bushings (6,000 rad/s), a 301010 LiPo (10 mAh, 3 x 10 x
10 mm) instead of the CeraCharges, 1.6 mm spheres, climbs at 37,000 rpm. 3.1 g, 11 mm/s alone, about
5,000 rolls a charge, 14.8 USD at 10. Its tightest margins: climb 1.54, packing 1.59, hinge 1.62. The way to prove the concept before paying for jewel bearings.

### Pick 3: G-N20 at 40 mm (the only design with the arm and walker)

Layout C made real: N20 1:298 wheel motor (retractable on the selector), N20 twist disc with a 3:1
printed stage, N20 selector that drives the edge hooks and swings a kick leg on the floor; 8 corner
spheres of 6.7 mm for capture; 150 mAh 1S whoop cell; ESP32-C3 SuperMini. 65 g. Torque 2.3x the arm
need. The hook latch and selector are the undesigned parts; the model assumes them strong.

## 7. Move primitives for the voxel policy (pick 1)

Each primitive is one grid action with an outcome model to measure in MuJoCo and on the bench:

| action | what the cube does | grid effect | measured outcome classes |
| --- | --- | --- | --- |
| ROLL(dir) | floor or top-of-structure roll about the edge in one of 4 directions | one cell, a quarter turn of orientation | reached / stayed (too little impulse, retry) / double roll / slid off a structure |
| CLIMB(edge) | peel the bond and swing 180 deg over a neighbour's top edge | up one cell onto the neighbour | reached / fell back (retry) / overshot / dislodged the neighbour (harmful) |
| DESCEND(edge) | swing 180 deg down from a structure edge | down one cell | reached / bounced / fell |
| SPLIT(face) | peel one bond by rolling away | bond removed, one cell | released / held (retry) / pulled the neighbour along (harmful) |
| YAW(sign) | impulsive yaw on a face with a vertical flywheel axis | none (heading trim before a dock) | within 3 deg / not |
| BOND | passive: the spheres close any face that arrives within 2.2 mm | bond added | closed / gap left |

There is no TURN_JOINT action in pick 1 (no twist joint). Pick 3 has DRIVE, PIVOT, BOND/UNBOND (hook)
and TURN_JOINT.

## 8. What would close the gap to 5 mm and to machines

1. **5 mm swarm cube:** a storage cell of about 0.5 J in under 3 x 3 x 1.5 mm with at least 10 mW
   pulses (no orderable part found), or chain power only; jewel bearings rated over 150,000 rpm; the
   electronics on two faces instead of three.
1. **Machines at small size:** a gearmotor at N20 torque density (about 160 N m per litre at stall,
   A) in a 3-4 mm package. With that, the arm closes at about 20 mm (E). Orbray's 1.5 mm geared motor
   has no published torque; for a twist face in a 7 mm cube it would need at least 0.6 mN m continuous
   in at most 2 mm x 5 mm. Ask Orbray for exactly that number.
1. **Orbray, as studied:** BMN02 (2.4 x 5 mm, 10 uN m stall, 6,000 rpm max, V) is too slow for a
   flywheel (0.2-0.6 of the roll impulse); BMN04-0829P337 (5.44 mN m, V) gives 1.1x the arm torque in a
   12 mm cube but is 18.4 mm long, and at the 20 mm where it fits it gives 0.18. Price and India route
   unknown (inquiry only). The size they buy is nil for the swarm pick and none for machines.

## 9. Bench tests that a model cannot settle (first, in this order)

1. **Sphere bond and hinge:** pull-off, peel moment and lateral capture of 1.0 and 1.5 mm N52 spheres
   in printed corner pockets, and whether they stay free to turn (friction in a resin pocket, dust,
   nickel chipping after 1,000 snaps). Scale: 0.001 g kitchen scale and a screw stage.
1. **Pawl catch:** a 4 mm brass flywheel spun by a 0408 coreless motor, caught by a pawl, mounted in
   a 20 mm test cube: angular momentum transfer, tooth wear over 10,000 catches, floor roll success on
   wood, tile, paper and glass (slip, bounce, adhesion). This is the dynamically similar test of the
   whole concept at 3x size (omega scales as L^-0.5).
1. **Flex stator:** torque and heating of a 2-layer flex 3-coil stator under a 4 x 1 mm diametric disc
   at 0.1 mm gap; motor constant against the 4e-5 N m / sqrt(W) estimate.
1. **Bearings:** a 0.5 mm pin in resin and in a watch jewel at 20,000-80,000 rpm: friction torque,
   run-down time, wear.
1. **CeraCharge:** actual pulse current and voltage sag at 3 mA for 0.3 s, repeated every 1.5 s.
1. **Print accuracy:** corner pockets, 0.3 mm frame bars and bearing seats on the team's resin printer
   (MSLA XY about 0.03-0.05 mm, A) and from JLC3DP.
1. **Fall:** a filled 7 mm dummy (tungsten putty to 0.85 g) dropped 100 times from 20 cm.

## 10. Bench kit to order tonight (pick 1 and its 20 mm scale model)

| item | qty | source (parts.json unless noted) | why |
| --- | --- | --- | --- |
| N52 spheres 1.0, 1.5, 2.0 mm (fallback: 1 x 1 mm cylinders) | 100 each | AliExpress (availability of 1 mm spheres unverified) | bonds, hinge tests |
| N52 spheres 3 mm | 50 | AliExpress / supermagnete K-03-C (V) | 20 mm scale model |
| diametric N52 discs 4 x 1 or 4 x 2 mm | 20 | AliExpress (parts.json 4 x 2) | rotor cores |
| brass tube 5 mm OD, 0.5 mm wall; brass rod 4 mm | 1 each | hobby shop | flywheel rims |
| steel pins 0.5 mm, watch jewel bearing assortment | 1 pack each | AliExpress (not in parts.json, A) | flywheel bearings |
| 0408 coreless motors | 10 | AliExpress, 1 USD | 20 mm scale-model flywheels |
| 0615 coreless motors | 5 | AliExpress | margin for the scale model |
| TDK CeraCharge 1812 (100 uAh) | 10 | Mouser or DigiKey India (V capacity, price A) | storage |
| Seiko CPH3225A 11 mF | 10 | DigiKey / LCSC | pulse buffer |
| LiPo 301010 10 mAh | 5 | AliExpress | pick 2 storage |
| CH32V003F4U6 and WCH-LinkE programmer | 20 + 1 | LCSC C5299908 (V) | MCU |
| LIS2DW12, DRV8837, SOT-723 FETs, 0201 Schottky, 0402 IR LED and photodiode | 20 each | LCSC | electronics |
| JLC flex PCB, 2 layers, 0.11 mm, 1 oz: panel of stator test coils (3-coil, 16 turns), pawl coils, and an unfolded 7 mm cube net | 5 panels | JLCPCB (setup fees A) | stators and the first folded cube |
| JLC rigid 0.6 mm FR4 copy of the same coils | 5 | JLCPCB | faster coil tests |
| magnet wire 0.05 and 0.1 mm | 1 spool each | Robu.in / AliExpress | hand-wound coil comparison |
| resin for the team's printer (or JLC3DP 8K resin quotes) | - | - | frames and pockets |
| Orbray inquiry email: 1.5 mm and 2.4 mm geared and brushless specs (torque, speed, length, price, lead time to India), and the No Electricity Locking System | - | Orbray sales partners (no India rep, V) | the only path to a twist joint at small size |

## 11. For the optimiser (src/beebot/hardware/mmsim)

Objectives, for every pick: minimise L; minimise cost per cube at 10 and at 1k units; maximise the
worst behaviour's success probability (from Monte Carlo over tolerances); maximise lone speed in
L/s; maximise joint torque over the arm need (or payload). Report the Pareto front, not a weighted sum.

### Pick 1 and 2 (reaction-wheel cube)

1. **Parameters and ranges:** L 5-14 mm; sphere diameter 0.6-2.0 mm and cover 0-0.15 mm; rotor
   radius 0.25-0.48 L, thickness 0.3-1.5 mm, rim material {brass, tungsten}, rim inner radius
   0.3-0.7 of rotor radius; bearing omega cap {6k, 10k, 20k} rad/s; stator layers {2, 4}, copper
   {0.5, 1} oz, trace width 0.075-0.15 mm, turns 6-30 per coil, coils per phase {1, 2}; pawl tooth
   count 1-4; storage {n CeraCharge 0-4, n CPH3225A 0-4, 301010 0-1}; spin power budget 2-20 mW;
   frame bar 0.2-0.5 mm.
1. **Constraints:** every margin of section 2 >= 1.5; parts inside the cavity with no overlap
   (envelope check); rotor hoop stress below 0.3 of the rim's yield; coil temperature rise below 60 K
   over a 1 s move; storage pulse current within its rating; a 20 cm fall below the pin's bearing
   allowable.
1. **Behaviours to simulate (MuJoCo with magpylib forces, mjcoupling), with pass criteria:**
   floor roll on mu 0.3-0.8 from each face (>= 0.99 reached, 0 double rolls over 1,000 seeds);
   yaw trim to 3 deg; dock into a row and into the corner of a 2x2x2 (>= 0.997 bonded); split from a
   bonded pair and from inside a row (>= 0.99, neighbour moved < 0.1 L); climb onto a neighbour and
   descend (>= 0.99 reached, 0 harmful); assemble a 2x2x2 from 8 loose cubes (whole build >= 0.95);
   20 cm fall onto each face, edge and corner (no part beyond its stress); 2-minute lone run with the
   storage model (CeraCharge 1 mA continuous, 3 mA for 1 s).

### Pick 3 (G-N20)

1. **Parameters and ranges:** L 36-56 mm; N20 ratio {100, 150, 298, 1000}; extra stage 1-5:1; wheel
   radius 0.1-0.25 L and protrusion 0-2 mm (retractable); kick-leg length 0.5-1.0 L; hook engagement
   0.5-2 mm; corner sphere 3-8 mm; cell {601422, 150 mAh whoop, 502030}.
1. **Constraints:** margins >= 1.5 including peak current; hook latch stress below 0.3 of printed
   material strength; no wheel lean during docks (retracted).
1. **Behaviours and pass criteria:** every layout B demo at its criterion (lone roll, cart, conveyor,
   crane, bridge and shelf, screwdriver, crank and slider, door, 14-module walker 50 cm in four
   cycles, crawler, 11-module arm placing 40 g within 3 mm), plus layout C's pairing, pair travel,
   split, reorient, climb, staircase and the 2x2x2 that layout C could not do; each >= 0.99 per
   operation over 1,000 seeds with 0 harmful outcomes.

### Minimum machine set to demonstrate (from module-demos.md)

Arm (11 modules, pick and place), walker (14 modules), crane (slew, luff), cart (300 g, turn, stop),
conveyor, crank and slider, screwdriver, door, crawler, bridge and shelf. Pick 1 can do the bridge
and shelf (bonds), the conveyor only as cubes rolled hand to hand, and none of the jointed ones.

## Sources

1. [ElectroVoxel, Nisser et al., ICRA 2022, arXiv 2207.05732](https://arxiv.org/abs/2207.05732) (read via ar5iv; no alphaxiv overview exists for it)
1. [TDK CeraCharge presentation (100 uAh, 1 mA continuous, 3 mA pulse)](https://www.tdk-electronics.tdk.com/download/2427688/6fa95084f0dd78caac95dcb8f051082a/06-dl---ceracharge-presentation.pdf)
1. [K&J D22-N52 (case 1 pull 0.84 lb), via albmaterials listing](https://www.albmaterials.com/D22-N52-neodymium-cylinder-magnets.html)
1. [supermagnete S-02-0.7-N](https://www.supermagnete.fi/eng/disc-magnets-neodymium/disc-magnet-2mm-0.7mm_S-02-0.7-N), [S-05-01-N](https://www.supermagnete.fi/eng/disc-magnets-neodymium/disc-magnet-5mm-1mm_S-05-01-N), [K-03-C sphere](https://www.supermagnete.de/eng/sphere-magnets-neodymium/sphere-magnet-3mm_K-03-C)
1. [Robot Pebbles, Gilpin et al. (12 mm cubes, EPM bonds, 85x weight)](https://dspace.mit.edu/handle/1721.1/70987)
1. Housner, G. W. (1963), The behavior of inverted pendulum structures during earthquakes, BSSA 53(2) (restitution of a rocking block; from memory)
