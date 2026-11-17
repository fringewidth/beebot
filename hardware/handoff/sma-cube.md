# SMA cube: one integrated cube, every behaviour simulated

Written 2026-11-15/16. **In progress.** This memo is rewritten as Monte Carlo batches finish. Each
table says how many trials stand behind its numbers.

**Status: simulation, analysis and CAD only. Nothing is built.** Nothing was committed, written to
`ops` or bought. CPU only, at most 6 processes. Tags: V = vendor page or datasheet read in this
repo; E = estimated (arithmetic or simulation); A = assumed. Some catalogue facts are from memory;
they are flagged "A, from memory".

**Code**
- `hardware/src/beebot/hardware/sma_cube.py`: design physics, MuJoCo world, grid moves.
- `hardware/scripts/module_phys/sma_cube_sim.py`: behaviours, Monte Carlo, NSGA-II, outcomes, assembly.
- `hardware/scripts/module_phys/sma_cube_machines.py`: arm, walker, crane, crank and boom with latching bonds.
- `hardware/scripts/module_phys/sma_cube_media.py`: MP4 and contact sheet.
- `hardware/scripts/module_phys/sma_cube_cad.py`: CadQuery model, written by a subagent.
- `hardware/tests/test_sma_cube.py`: 9 tests, about 10 s.

Outputs are in `hardware/runs/module_phys/sma_cube/` and `hardware/sma_cube/`.

## 1. Chosen design

The chosen point is **L = 17.84 mm** (`chosen.json`). It is the smallest edge in 16-25.4 mm where packing, joint torque at 1.5x on the walker and the arm, leg torque and a lone-return battery all close together.

| item | value |
| --- | --- |
| Mass | 2.99 g (E) |
| Joint | geared, `geared_joint.nominal_design`: flex-PCB rotor, Didel worm and spur stages, printed split-ring output. Rotor torque constant scaled with rotor volume (E). 17.9 mN m continuous, 25.1 mN m peak, 116 deg/s; self-locking worm |
| Legs | 4 edge legs, 0.867 L long. Each runs on a capstan ratchet with 5 strands of 38 um Flexinol. Drive 3.0 mN m (5.8 m g L), lock 5.4 mN m, 4.6 deg a step |
| Bond | latching edge bolts on the 4 x-parallel edges, each a hooked bar on a rack ratchet driven by a 76 um wire, plus a shared 25 um unlock wire. Travel 2.9 mm: the first 0.6 mm frees the hooks, the rest retracts that edge's two corner spheres. Latched plain bond: 40 N tension, 0.29 N m peel (E). Magnet-only plain bond: 0.45 N |
| Discs | disc bonds latched by a bayonet: a 45 deg twist of the joint |
| Capture | corner spheres 0.06 L (1.07 mm, N52) for capture and hinge. Ring spheres on the discs, steel inserts on the plain faces |
| Battery and power | LiPo 301010 (10 mAh, A) plus 6 x 220 uF polymer tantalum. Lone: 49 floor tips, about 25 min, 0.58 mm/s (E). Chains share power through the plated corner spheres |
| Electronics | CH32V003, LIS2DW12, 19 FETs and IR pairs on a folded flex. 13 SMA wires, 26 crimps |
| Cost | 53.1 USD at 10, 21.5 at 1k, 2.84 at 1M (E/A; 69 % of the 10-unit price is verified) |

## 2. How the point was found, and what moved it

1. **First search, 8-25.4 mm, analytic packing rule.** The rule was part volume x 1.5 <= 0.8 of the inner volume. It picked 12.18 mm with the geared joint.
1. **CAD check.** The CadQuery layout of that design (`hardware/sma_cube/README.md`) does not assemble:
   - 164 overlapping part pairs (206 mm3) at 12.18 mm, and 159 pairs (173 mm3) at 14 mm.
   - The joint stack needs 6.4 mm along x on each side; 4.5 mm is free beside the cell.
   - No edge-bolt hook tab fits.
   - No IC fits on the outer flex net.
1. **Layout check.** I replaced the packing rule as the binding constraint with a layout check fitted to the CAD sweep at 12.18, 14, 16 and 18 mm (`Design.layout()`). It checks three things:
   - each joint stack's depth along x beside the cell;
   - hook travel plus one tab inside the edge region;
   - the cell across the cavity.

   The joint first fits beside the cell at about 17.3 mm.
1. **Measured leg needs.** The first band search let through a 1.2 m g L leg, which then stalled in KICK and CLIMB in MuJoCo. Peak leg lock torques measured in nominal runs are CLIMB 1.97 m g L and KICK 1.19 m g L (BOOK 0.72). These replaced the static estimates (0.71 and 0.5) in the margins.
1. **Joint routes over 16-25.4 mm.**
   - **SMA ratchet joint, scaled up:** never reaches a margin of 1 against the walker. The best is 0.99 at 16 mm, falling to about 0.5 at 25 mm. Hold grows roughly with L, but m g L grows with L^4. It is also slow: 0.08-0.12 rad/s, so a quarter turn takes 13-20 s, at 2-8 J a radian.
   - **Geared joint:** closes from about 17.6 mm once the rotor's torque constant scales with its volume.
   - **Bought N20 or N10 1000:1 gearmotor with a selector:** does not fit. The body with its gearbox is about 24-27 mm long (A, from memory; Pololu micro metal HP 1000:1 and generic N20s are 10 x 12 mm in section). The inner side at 25.4 mm is 24.6 mm. The face diagonal (34.7 mm) would hold it, but it crosses two edge-bolt corners and the cell.
1. **Leg lock face width now grows with L** (0.9 mm at 14 mm, E). With the fixed 0.9 mm, a 20 cm fall loaded the closed leg lock to 1.51x its 30 MPa design stress, past the resin's assumed 45 MPa strength. With the scaled width it is 1.18x.

## 3. Behaviours

### 3.1 First full batch: old leg layout, 17.84 mm, 20 trials a move

This batch ran before the leg-pair change of section 6: XN on the -y/-z edge, DP in +z, no tip pad. It used the 0.3 mm edge round, the 0.5 mm insert recess and dt 1e-4 s. Trials are in `hardware/runs/module_phys/sma_cube/trials_oldlegs_1784/`. Each move is one leg pivot about an edge, so the per-move physics is the same in the new layout. The orientation draws and the leg tip differ, which is why the new layout is being re-run (section 6).

| move | single try | 95 % interval | harmful | failure classes |
| --- | --- | --- | --- | --- |
| tip_book | 19/20 | 0.76-0.99 | 0 | 1 off_lattice |
| tip_kick | 19/20 | 0.76-0.99 | 0 | 1 off_lattice |
| yaw | 20/20 | 0.84-1.00 | 0 | |
| self_right | 19/20 | 0.76-0.99 | 0 | 1 not_home |
| fall (20 cm) | 20/20 | 0.84-1.00 | 0 | |
| dock, assumed error (+-0.15 L, +-6 deg) | 16/20 | 0.58-0.92 | 1 | 2 off_lattice, 1 unlatched, 1 knocked_support |
| dock, assumed error, IR back-off | 17/20 | 0.64-0.95 | 1 | |
| dock, measured landing error | 20/20 | 0.84-1.00 | 0 | |
| dock, measured error, IR back-off | 20/20 | 0.84-1.00 | 0 | |
| climb_row | 17/20 | 0.64-0.95 | 3 | 3 knocked_support |
| climb_top | 19/20 | 0.76-0.99 | 1 | 1 knocked_support |
| roll_top | 20/20 | 0.84-1.00 | 0 | |
| book_descend (free 135 deg swing) | 14/20 | 0.48-0.85 | 5 | 5 knocked_support, 1 off_lattice |
| split_pair | 20/20 | 0.84-1.00 | 0 | |
| split_row | 20/20 | 0.84-1.00 | 0 | |
| climb_lone (never offered) | 0/20 | 0-0.16 | 20 | support knocked over, as expected |

The dock error model matters more than the back-off rule. The assumed model (+-0.15 L uniform) is much wider than the measured landing error: 37 BOOK/KICK landings, median 0.45 mm, 90th percentile 1.3 mm, max 2.0 mm, heading 90th percentile 0.34 deg. With the measured model, docking passes 20/20 with or without back-off.

## 4. Simulation fixes found while running

1. **Time step.** A 2e-4 s step made the structure vibrate after a landing. The floor normal swung by ±30 % and the cubes walked across a 0.21-friction floor at 1.3 mm/s, although the magnetic forces summed to zero. The same seed at 1e-4 s lands cleanly: roll_top 4.2 mm error became 0.32 mm, and climb_row became reached. The Monte Carlo now runs at 1e-4 s.
1. **Event skipping** keeps it affordable. While every velocity is below 2e-5 m/s, 2e-3 rad/s and 5e-3 rad/s (hinges), and no stroke or bolt is moving, time jumps to the next stroke start. Conditions are checked before each jump. Checked on four moves: same class, landing within 0.03 mm, 2-10x faster.
1. **The geared joint was simulated as the SMA stepper** until 15 Nov 22:30. It is now a 1 deg servo at its loaded speed, torque-limited and self-locking.
1. **Controller changes, all things a built cube can do:**
   - a pivot stops when the IMU shows the planned turn, instead of grinding the leg into the support (this removed 117-172 failed steps);
   - the leg backs off 3 teeth before relatching;
   - floor tips fire whole strands.
1. **Bolt travel was too short.** Retracting the corner spheres by 1.2 mm left 2.2 cube weights of pull, so a SPLIT lifted the neighbour. With more travel the pull is 0.9 weights at 14 mm, and the margins put it in the search.

## 5. Machines with latching bonds (17.84 mm, geared joint)

`hardware/scripts/module_phys/sma_cube_machines.py` runs the machines of `hardware/scripts/small_joint_machines.py` without editing it: it swaps in the cube's mass and replaces the rig with `LatchRig`, which applies the bond rules below.

**Bond rules**
- A plain bond between cubes with parallel x axes is latched: 40 N, 0.29 N m peel (E).
- A plain bond between crossed x axes is magnetic only: 0.45 N.
- Disc bonds are bayonet-latched.

**Layout changes**
- Passive cubes are turned so their plain bonds latch (walker 5, 7, 10, 13; arm 9).
- Where the crossed cube carries a joint, its parent is turned instead. This puts the bond on the parent's disc, held by that joint's self-locking worm (crane cube 2).

**Arm:** it adds a 50 N/m flexure finger, as the SMA arm had. A stiff worm closing on the block broke the wrist tooth (38 mN m in joint_gear14).

**Walker:** it runs a 20 s gait period. At the gear-speed period (about 7.8 s) the teeter at each leg lift loads the hip joints to 0.92 of the tooth limit, which fails at 1.5x. At 20 s it is 0.36, and at 30 s 0.31.

**Boom (new machine)**
- Layout: five latched cubes reaching four cells out from a pitch disc on a tower, slewed on the bolted base disc.
- Static moment at the boom root is 2.0 mN m at 12.18 mm. A magnet-only disc bond there peels at 0.83 mN m, so layout B could not hold it.
- In the first version the boom slewed on the tower disc (0p). That broke 0p at 41-45 mN m in both runs.

Nominal results and Monte Carlo (10 trials a machine) are in `hardware/runs/module_phys/sma_cube/machines/`.

Monte Carlo at 17.84 mm, 10 trials a machine at 1.5x margin (`machines/mc_17.84mm.jsonl`):

| machine | passed | peak joint / tooth limit (nominal) | peak bond use (nominal) |
| --- | --- | --- | --- |
| arm | 10/10 | 0.28 | 0.10 |
| walker (20 s period) | 10/10 | 0.35 | 0.10 |
| crane | 10/10 | 0.13 | 0.02 |
| boom | 9/10 | 0.24 | 0.07 |
| crank | 8/10 | 0.30 | |

## 6. Descent: diagnosis, the leg pair and braked moves (16 Nov)

**Why book_descend knocks its support.** Traced at fine time steps on seed 1001 (`logs/sma_cube/descend_trace.py`):
1. The lift leg lies in the bottom face and pushes on the row's top. It can only stop the cube rolling back, so it raises the cube quasi-statically to the balance point (45 deg) and no further.
1. From there the cube swings 135 deg about the magnetic hinge in under 0.2 s, reaching about 45 rad/s. The hinge's inward pull (about 75 mN plus gravity) is several times the row's grip on the floor (mu x 3 m g = 13-25 mN), so the whole 3-cube structure skids about 1.9 mm toward the end of the row.
1. The bolts read hooks-free and far-spheres-retracted throughout. The 4-tooth leg lock and the re-seat act only after the landing, so neither changes it.
1. Successful trials show the same drag at half the size (harm_ratio about 0.55).

**The leg pair (adopted).** With the old layout no edge carried a leg in each of its two faces, so a cube could not brake itself past the balance point. XN now lies in +z on the same +y/+z edge as XP, sharing XP's pin with interleaved knuckles. DP moved to -y, hinged on +x/-y.
1. **Lone mobility improves** (bfs_lone): all 24 orientations still reach home within 6 moves; 12 reach home without leaving the start cell (was 4); 48 neighbour-cell targets are reachable (was 32); the 4-cell box holds 928 states (was 432).
1. **Structure moves** keep the same number of orientation options.
1. **Cost of the change.** No mass or BOM change: still 4 legs and 4 capstans, one hinge pin fewer. The CAD check at 17.84 mm is unchanged in substance (115 static overlaps, 65.8 mm3, against 116 and 66.0; 27 swept). Both hook tabs of bolt pp sit under the pair and are dropped, against one on pp and one on nn before; 6 tabs either way.
1. **Leg tip pad.** Each leg plate sits 0.1 mm deeper in its recess, with a flush 0.1 x 0.8 mm pad at the tip. A leg lying on a neighbour's face then bears at its tip, at the full lever. Without the pad, the brake leg bore near its knuckle and its push opened the hinge by up to 3.7 mm.

**Braked moves** (`descend_braked`, `roll_top_braked`, `climb_row_braked`; `descend_endbrake` is the planner's fallback, braked by the row's end cube over IR):
1. The partner leg is opened so that it meets the support just past the balance point. The pivot leg takes the cube there, and the partner then steps shut a tooth at a time, lowering the cube.
1. **Descent: the swing drag is gone.** From 49 to 160 deg the row moved at most 0.33 mm, and the hinge stayed within 0.8 mm of its line (seeds 1001 and 1002).
1. **Descent: touchdown still shifts the row about 1.9 mm.** The magnetic hinge sags 0.1-0.4 mm, from rolling on the rounded edges and the brake's push against the magnets' compliance. So the cube's floor-side edge lands at 164-174 deg rather than 180. The cube finishes the turn about that floor edge, which moves its hinge edge about L sin(6 deg) = 1.9 mm out, and the hinge magnets pull the row after it.
   - Sharp edges do the same (seed 1001 harm 2.2; seed 1002 reached at 0.97).
   - Retracting the hinge spheres before touchdown leaves the row alone (harm 0.01-0.33), but the cube then slides off the hinge and lands 2.6-5 mm out, unlatched.
   - Pressing flat with the lift leg after the brake folds does not prevent it.
   - In every one of these failures the row moves as one rigid piece with all bonds intact. Results now carry `support_shift_mm` and `rel_harm_ratio` (disturbance relative to the structure) next to the absolute `harm_ratio`. The class still uses the absolute rule.

**Braked against free flop, 20 seeds a move at 17.84 mm on the new layout with the tip pad** (`hardware/runs/module_phys/sma_cube/trials/`):

| move | free flop | braked | braked failures | median harm in successes (free / braked) |
| --- | --- | --- | --- | --- |
| descend | 14/20 (0.48-0.85) | 6/20 (0.15-0.52) | 8 knocked_support, all rigid row shifts of 2.0-2.5 mm with bonds intact; 6 off_lattice, cube 4.7-6.4 mm out and unlatched, on floors with mu about 0.56-0.76 | 0.53 / 0.60 |
| climb_row | 19/20 (0.76-0.99) | 6/20 (0.15-0.52) | 14 off_lattice, cube 3.4-7.3 mm out and unlatched, no harm | 0.12 / 0.04 |
| roll_top | 20/20 (0.84-1.00) | 19/20 (0.76-0.99) | 1 knocked_support | 0.11 / 0.02 |

The free-flop descend fails as in the old layout: 5 knocked_support and 1 unlatched.

A quasi-static landing ends with the landing edge touching a few degrees early, because the magnetic hinge sags. The cube then pivots about that edge. It either drags the row with it (slippery floor) or stops several mm out with the latch open (grippy floor). A free flop arrives with momentum and the magnets snap it into place.

**Decision (coordinator, 16 Nov):**
1. Keep the free flops for descend, climb and roll_top, and keep the leg pair, which improves lone mobility at no hardware cost.
1. Braking is dropped from the re-run at the final L.
1. Touchdown work is parked until the CAD closes.

**Possible fix for later (not done): a real pin hinge on the hinge edge.**
- **Root cause.** The touchdown error comes from hinge sag, that is, the compliance of the magnet-only hinge. Descent is the weakest move the planner uses (14/20 in the old layout).
- **Proposed hinge.** The bolt on the hinge edge would hook around the neighbour's coincident edge and act as a pin during the swing, so the hinge would be rigid and a braked landing could be precise. The idea is to keep the hinge-edge bolt engaged and shape its hook as a round pin coaxial with the shared edge line, turning in a round eye in the neighbour's edge.
- **Geometric problem.** Each cube's bolt bar sits inside its own corner, about cover + bar/2 off the edge line, and the 0.3 mm round leaves no material on the line itself. So the pin would have to reach across into the neighbour's corner.
- **What it would take.** It needs a CAD study of the corner and a MuJoCo model with a hinge equality (two connect points on the edge line) in place of the weld while the swing runs.

## 7. Packing actually closed: L = 19.5 mm (CAD, separate from the motion results)

**The earlier layout check was wrong.** It said the joint fits beside the cell from about 17.3 mm, but it checked only the joint stack's depth along x. At 17.84 mm the full CadQuery check found 115 static overlaps (65.8 mm3) and 27 swept pairs. 114 of the static overlaps were real collisions:
1. rotor vs the fixed split ring and the disc, about 19 mm3;
1. selector gears vs battery, 12.3 mm3;
1. capstans with no free layer, 8.5 mm3;
1. capstan pawls, about 9 mm3;
1. rack pawls vs groove bosses, 6.1 mm3;
1. worm vs planets, 4.5 mm3.

The only artefact was the doubled pin of the leg pair.

**Re-layout (option B, `cad_sweep/B_<L>/`):**
1. battery cell moved off-axis along +z, inboard worm, rotor centred under each +-y face;
1. capstan radius cut from 0.14 L to 0.10 L, leg width from 0.36 L to 0.32 L;
1. ring wall 0.3 mm, rotor 0.15 mm thinner;
1. rack fin 0.46 mm;
1. bolt pp notched under the leg pair, bayonet-boss footprints cut from the bolts, +x carrier rails cut back.

| L (mm) | real static | real swept | note |
| --- | --- | --- | --- |
| 17.84-18.4 | - | - | capstan pawls do not fit their boxes |
| 18.6 | 0 | 0 | only with a 0.30 mm rack fin, whose core of 0.14 mm cannot be printed |
| 18.6, 0.46 mm fin | - | - | DP pawls do not fit |
| 19.0 | 2 (0.013 mm3 each) | 0 | rotor rim against the np and pn drive pawls |
| **19.5** | **0** | **0** | **closing L**, all parts printable |
| 20.0 | 0 | 0 | |

**Checker exclusions, each justified in `sma_cube_cad.py`:**
1. a fold pin against its own wire;
1. a wire against the pawl it pulls;
1. gear meshes and shafts in bores, reported separately as intended contacts;
1. flex components, reported separately;
1. disc hubs, disc pins and hook tabs, which stand proud of the faces by design.

**At 19.5 mm:**
1. joint x-depth margin 0.46 mm; worm to cell 0.05 mm;
1. 1 hook per side, 6 tabs, none on edge pp;
1. legs 16.9 x 6.2 mm, capstan r 1.95 mm;
1. LiPo 301010.

**Model at 19.5 mm** (`chosen.json`; the 17.84 mm point is kept as `chosen_17.84mm_cad_open.json`):
1. margins: climb 2.0, floor 1.98, split 2.56, joint 1.57, latch 7.8;
1. leg step 5.9 deg;
1. 3.42 g;
1. 53.3 / 21.7 / 2.88 USD at 10 / 1k / 1M.

**Open, and decisive for a built cube** (CAD work started 16 Nov):
1. **SMA wire folds.** The routing folds the wires over 0.15 mm pins, about 13 % bend strain for 38 um wire. The alloy recovers far less. The routing has to respect Dynalloy's minimum bend radius, and the actuator stroke, force and cycle time follow from the new routing (Design fields leg_wire_len_m, leg_wire_eta, rel_wire_len_m, rel_lever).
1. **Electronics have no home.** MCU, FETs, IMU, IR, charger and crimps are unplaced. A straight inner flex tail does not fit, and flex components overlap bolts and carriers on the -y/-z faces.
1. **Carrier guidance.** The +x carrier rail starts about 0.5 mm into the carrier's travel, so the carrier is guided only by its sphere in the skin pocket over that first 0.5 mm. This is a bench item.
