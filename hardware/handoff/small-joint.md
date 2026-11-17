# A load-holding joint for the 7-11 mm cube

Written 2026-11-15. **Status: simulation and analysis only. Nothing is built.** The SMA ratchet joint
is designed only as an envelope: every part is a box, cylinder or sphere at its size in CadQuery,
checked for overlaps (`hardware/module_phys/sma_joint_cube_8.25mm_envelope.step`). There is no
detailed CAD, no print and no test of the ratchet, the pawls, the wires or the bonds. Nothing was
committed and nothing was written to `ops`. CPU only, at most 6 processes.

Every number is marked **V** (verified from a datasheet or vendor page), **E** (estimated: arithmetic
or simulation on stated inputs) or **A** (assumed). Simulation results are E: they are only as good
as the idealisations listed in section 5.

## 1. Answer

**Minimum L: 8.25 mm** for the joint, with an **SMA ratchet stepper** in each disc (four 18-strand 25 um
Flexinol wires per joint, a printed ratchet ring, a square-tooth lock pawl that holds with no power).

1. At 8.25 mm the joint never fails: in 400 Monte Carlo trials its worst load is 0.66 of the tooth
   limit. The crane (100 %, 99 % at 1.5x) and the crank (100 %, 100 %) pass with margin. The arm
   passes 84 % (46 % at 1.5x) and the walker 72 % (15 % at 1.5x); every one of their failures is a
   magnet bond opening, not the joint (section 4). No L in 6-12 mm passes all four at 1.5x on bonds
   (`min_edge_mm_strict` null in results.jsonl): the bonds, not the joint, bind.
1. The walker (layout B's 14-cube quadruped) walks in every nominal run and holds its bonds at full
   strength, but **not with a 1.5x bond margin**: when a leg lifts, the body teeters on the diagonal
   between two feet for about 0.1 s, and the hip and body bonds see 5 ms loads up to 0.92 of their
   strength (0.80 at a 1e-4 s time step). That is the gait, not the joint: joint loads stay at
   0.45 of the tooth limit. Section 4 gives the Monte Carlo rate at full strength and at 1.5x.
1. **The size is set by packing, not by torque.** The SMA joint has torque to spare at any L in the
   range (the drive is 22x m g L, the lock holds 40x m g L). Below 8.0 mm the parts do not fit with the
   1.5x packing rule (1.49 at 8.0, 1.13 at 7.0, E). Above 8.25 mm the reaction-wheel spin-up time falls
   under 1.5x (1.41 at 8.5 mm, E); if the swarm moves become quasi-static (the reaction-wheel track's
   result, section 7), that upper limit goes away and 8.25-9 mm is open.
1. Grow the cube for: self-powered machines. A 8.25 mm cube's CeraCharge cells (1.5 J, 4.2 mW
   continuous, V) cannot heat the wires (0.92 W pulses, 37 mJ a step, E). Machines run on chain power
   through the corner spheres, or a tether. Self-powered machine work needs the 11 mm cube with a
   10 mAh LiPo (concepts.md pick 2): about 2,800 steps a charge (E: 133 J, 70 % conversion, 34 mJ a step at 11 mm).

### Battery power against size (E, battery figures partly A)

| L (mm) | step energy | pulse | one joint stepping continuously | battery that fits | self-powered? |
| --- | --- | --- | --- | --- | --- |
| 7-8.25 | 37 mJ | 0.92 W, 0.47 A | 0.23 W | TDK CeraCharge x3: 1.5 J, 4.2 mW (V) | no: about 40 steps a charge, one per 9 s; chain or tether power |
| 11 | 51 mJ | 1.26 W | 0.32 W | 301010 LiPo, 10 mAh, about 130 J; 1-2C (A) | yes, slowly: a supercap buffers each pulse, the cell's 40-75 mW allows 1-2 steps/s (5-8 deg/s), about 2,500 steps a charge |
| 13-14 | 60-65 mJ | 1.6 W | 0.41 W | 15-30 mAh LiPo (A) | yes, 2-3x the energy, still rate-limited |
| 20-25 | | | | high-rate LiPo, about 40 mAh at 10C+ (A) | full speed on its own cell |

The smallest cube that runs on its own battery is about 11 mm (reduced speed); full speed needs
about 20 mm or more. At 8.25 mm the joint machines run on power shared through the corner spheres.

### Charging (designed on paper only)

1. Through the bonds (concepts.md pick 1): nickel-plated corner spheres conduct; each corner reaches
   the rails through two Schottky diodes, so any bonded neighbour feeds power at any orientation and
   polarity. One cube of a cluster on a powered base or a power-bank tether charges the rest.
1. A charge pad on one face for a lone cube parked face down.
1. In the 11 mm cube, a chip-scale single-cell linear charger (BQ25100 class, from memory, A) at
   about 10 mA (1C), about an hour to full, with cell protection.
1. Risk: SMA pulses are 0.47 A a wire. Running joints from chain power puts amps through the first
   point contacts of a long chain. Let each cube's own cell or supercap supply the pulses and the chain
   carry only the average (tens of mA); bench-test plated sphere contact resistance at 0.5 A and the
   sag along a 10-cube chain first.

### What this work shows (judgement)

1. The joint is solved even at 8 mm: in 400 Monte Carlo trials no joint broke (worst 0.66 of the
   tooth limit), and it holds with no power.
1. The magnet bonds are the bottleneck: every arm and walker failure is a bond opening. Load paths
   through a few bonds in bending (cantilevers, a teetering body) fail; table-supported or
   compressive ones pass (the crank went from 2.9x to 0.19 by resting on the table).
1. So the highest-leverage next step is a power-off latching bond: it fixes machine reliability and
   gives the releasable bond the reaction-wheel track needs.
1. Scaling: loads L^4 against actuators L^2-L^3, packing and assembly against small L: a sweet spot
   at 8-16 mm, 8.25 mm the floor, about 14 mm geared the practical optimum.
1. SMA is the joint when size dominates; its strengths are tiny size and power-off holding (form,
   then lock), not work (0.5 % efficient).
1. Simulation details moved results 2-5x (contact stiffness, rigid loops, stepper toggling): bench
   tests (bond strength, ratchet print, contact resistance, wire termination) before more simulation.

## 2. Routes compared (at L = 8.25 mm unless stated)

| route | torque (drive / hold) | speed | energy per arm move | hold with power off | packing | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| SMA ratchet stepper | 2.30 / 4.21 mN m peak, 2.11 continuous (E) | 26.7 deg/s, 4.21 deg steps (E) | 3.1 J measured (E) | yes, square-tooth lock pawl | 1.58 at 8.25 mm (E) | **chosen** |
| micro gear train: flywheel motor, Didel worm, printed split-ring stage | 4.19 mN m peak at 8.25 mm, teeth limit (E) | 133 deg/s no load (E) | 0.07 J (E) | yes, self-locking worm | worm does not fit below 11 mm; printed module 0.135 mm at 8.25 mm, not printable | min L 14 mm (E) |
| piezo stick-slip ring | 1.09 mN m (E/A) | 126 deg/s (E/A) | small | friction only, equal to the drive torque: no margin against a foot landing | 45 mm3 (A) | rejected: no hold margin, 150 V driver |
| bought gearmotor (Precision Microdrives 206-10C, 700:1) | 10 mN m nominal (V) | 222 deg/s (V) | small | no (spur gearhead back-drives, A) | fits only from 22 mm (E) | rejected |
| bought gearmotor (Pololu 1000:1 HP) | 245 mN m gearbox limit (V) | 186 deg/s (V) | small | no | fits only from 31 mm (E) | rejected |

The gear train counts only if printed or bought: Didel module 0.2 POM gears are bought (V: G209,
G220, G240, worm GW205); the split-ring stage would be printed. At 14 mm it gives ratio 1,547,
forward efficiency 0.098, a self-locking worm (lead angle 4.8 deg against a friction angle of 8.5
deg at mu 0.15, E) and a reflected rotor inertia of 0.01 kg m2, so the joint is sluggish under a
controller. Below 11 mm the worm does not fit and the printed module drops under what a 30-50 um
MSLA printer resolves. It is in `hardware/src/beebot/hardware/geared_joint.py` (`nominal_design`,
`worm_last_design`) and in `hardware/runs/module_phys/joint/results.jsonl` against L.

## 3. The SMA ratchet joint (one per disc, two per cube)

Specification at L = 8.25 mm (`sma_design(8.25e-3).table()` in `hardware/src/beebot/hardware/geared_joint.py`):

1. Ratchet ring on the disc hub: radius 2.72 mm (0.33 L), 86 square teeth at 0.20 mm pitch (A: 7-10
   pixels on a 20-35 um MSLA printer), tough resin. Step 4.21 deg.
1. Lock pawl: three teeth engaged, flanks tapered 5 deg (below the friction angle, so still
   self-locking), sprung into the gap so the spring takes up the 30-40 um print clearance; residual
   play 10 um at the ring, 3.7 mrad at the disc (E). Holds 4.21 mN m peak, 2.11 continuous, with no
   power (resin tooth bending, allowable 30 / 15 MPa A).
1. Drive: two phases, two directions, so four wires a joint, eight crimps. Each wire: Flexinol 90 C,
   25 um, looped over pins into 18 strands, 5.96 mm long; 1.57 N pull at 172 MPa (Dynalloy's
   recommended stress, V), 3 % strain (Dynalloy: 2-6 % repeatable, tens of millions of cycles at
   this stress, V). A pawl lever of 1.4 turns the wire's 0.18 mm stroke into 1.25 teeth. Drive torque
   2.30 mN m after a pawl efficiency of 0.75 (E; 0.55-0.90 in Monte Carlo), 1.5x the largest measured
   joint load (arm lift 1.53 mN m). The drive exceeds the 2.11 continuous tooth rating but not the
   4.21 peak, so a stalled stroke cannot break teeth; a stalled joint rests 1 s and retries (a grip
   does not retry).
1. Reversing: pick the other wire pair. Free disc (crank pin, piston pin): a bistable flexure latch
   holds the lock pawl up; firing both wires of a phase flips it (A).
1. Timing: 0.12 s stroke (heating pulse plus transformation lag, E), 0.195 s cooling inside the cube
   (Dynalloy still-air value x 1.3, E), two phases overlap: 0.158 s a step, 26.7 deg/s.
1. Electrical: 4.25 ohm a wire group, 1.98 V, 0.47 A, 0.92 W during a pulse. 36.9 mJ a step, 8.8 mJ a
   degree. Efficiency 0.46 % (E). The 1-3 % figure in the brief is for thicker wire at lower duty;
   25 um wire at 3 % strain and 172 MPa does about 0.5 % (latent heat and heating to 90 C dominate).
   The 1.5 N quoted for 50 um wire is about 760 MPa; Dynalloy rates it 0.35 N at 172 MPa (V).
1. Mass 0.23 g for both joints, 82 mm3 (E). Whole cube 1.30 g.
1. Cost per cube (all A; base from concepts.md pick 1, itself A):

   | | 10 cubes | 1,000 | 1,000,000 |
   | --- | --- | --- | --- |
   | base cube (flex PCBA, MCU, IR, 3 CeraCharges, corner spheres, reaction wheels) | 24.8 | 8.0 | 1.8 (cells 1.2) |
   | Flexinol 25 um, about 1 m | 3 | 1.5 | 0.2-0.3 |
   | 24 face-ring spheres | 1.2 | 0.5 | 0.1-0.2 |
   | crimps, pins, springs | 1 | 0.3 | 0.05 |
   | ratchet, pawls, hubs (printed; micro-moulded at 1M) | 1 | 0.5 | 0.05-0.1 |
   | wire drivers, supercap, contacts | 2 | 1 | 0.2-0.3 |
   | assembly | own time, 1-2 h a cube | 5-10 | 1-2 (automated, tooling amortised) |
   | per cube | about 33 plus own time | about 16-20 | about 3.5-5 |

   A million cubes: 3.5-5 M USD, plus 1-3 M USD up front (micro-moulding tools, an automated SMA
   winding and welding line, test fixtures, cell certification), about 5-8 USD a cube all in. At
   volume, swap the 18-strand loops for welded or stamped SMA elements; chain power with a small
   reserve could save about 0.8-1.0 a cube in cells.

1. **The 1 USD cube (target, all A; superseded in part: a battery is required, about +0.4 a cell).**
   Reachable at 1,000,000 units for parts plus assembly only, with 1-2 M USD up front paid separately
   (1 USD all in needs about 10 M units); impossible at 10 or 1k. It drops the cells (bus or
   powered-mat power only, a pulse capacitor; a disconnected cube is inert), the reaction wheels (every move quasi-static), most IR (signal over the contacts), and
   per-corner switching (centre pad plus ring contacts), and uses thick wire:

   | wire | drive | speed | wire a cube |
   | --- | --- | --- | --- |
   | 25 um x 18 (simulated) | 2.3 mN m | 27 deg/s | 0.86 m |
   | 50 um x 4 | 2.1 mN m | 16.5 deg/s | 0.19 m |
   | 76 um x 2 | 2.3 mN m | 8 deg/s | 0.10 m |

   Bill at 1M: MCU 0.10, flex PCB and SMT 0.15, wire drivers 0.06, face contacts and rectifiers
   0.10, capacitor 0.05, IR 0.03, spheres 0.10, SMA 0.03-0.05, moulded frame and ratchets 0.06,
   automated assembly and test 0.20-0.30: about 0.90-1.10 USD. Dropping cells and wheels frees about
   80 mm3, so packing alone would allow about 7.5 mm.

How to assemble (A, untested): print the ring hub and pawl carrier as one resin part per disc;
thread each wire over its pins with tweezers under a stereo microscope and crimp both ends in brass
tube crimps; set the lock pawl spring last. A hobbyist can buy every part (Flexinol from Dynalloy or
Robotshop, N52 spheres, crimps) and print the rest on an MSLA printer; the 0.20 mm tooth pitch must be
proven by a test print first.

## 4. Simulation (MuJoCo, L = 8.25 mm)

Rebuilt from layout B (`hardware/scripts/small_joint_machines.py`, the stepper model in
`hardware/src/beebot/hardware/geared_joint.py`, `GearedRig._stepper`): real stepping (4.21 deg teeth, a step
fires only when the wire is ready, a stroke against a load the wire cannot beat stalls and the lock
drops back, a failed step), the lock's clearance and stiffness, the wire's pull saturating at the
drive torque, bearing friction, and power-off holding.

Nominal (`hardware/runs/module_phys/joint/machines_8.25mm.jsonl`; joint peak against the tooth limit, bond load
against strength averaged over 5 ms, bond opening against the failing opening at strength / 1.5):

| machine | task (measured) | joint peak / tooth limit | bond load / strength (5 ms) | worst bond | opening at strength / 1.5 | energy | pass | pass at 1.5x |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arm | lifted 5.76 mm, carried 29.4 mm, 0.44 mm from the mark | 0.37 | 0.64 | 0+x-1-z | 0.10 | 3.1 J | yes | yes |
| walker | walked 62.2 mm in 4 cycles (65 s a cycle) | 0.45 | 0.92 | 0-z-1+z | 2.00 | 50.5 J | yes | no |
| crane | raised 8.41 mm, slewed -88.5 deg | 0.14 | 0.22 | 0+x-1-y | 0.00 | 3.2 J | yes | yes |
| crank | 3.29 turns, stroke 17.29 mm | 0.22 | 0.19 | 4+y-5-y | 0.00 | 10.4 J | yes | yes |

(Opening 1.0 or more means the bond opens at strength / 1.5; values are capped at 2.)

Monte Carlo (`hardware/runs/module_phys/joint/mc_8.25mm.jsonl`, all parameters varied together in every trial:
pawl efficiency 0.55-0.90, wire pull 0.80-1.10, cooling 0.8-2.0x, lock clearance 0.5-2x, bond strength
0.75-1.10x, table friction 0.5-1.0 (the crank's sheet 0.15-0.35), cube friction 0.4-0.8; 95 % Wilson
intervals):

| machine | n | pass (95 % CI) | pass at 1.5x (95 % CI) | joint / tooth limit, median / p95 / max | bond load / strength, median / p95 | failures |
| --- | --- | --- | --- | --- | --- | --- |
| arm | 100 | 0.84 (0.76-0.90) | 0.46 (0.37-0.56) | 0.46 / 0.58 / 0.66 | 0.71 / 1.77 | task 16, bond opened 12, opened at 1.5x 54, joint broken 0 |
| walker | 100 | 0.72 (0.62-0.80) | 0.15 (0.09-0.23) | 0.43 / 0.46 / 0.47 | 0.97 / 1.21 | task 0, bond opened 28, opened at 1.5x 85, joint broken 0 |
| crane | 100 | 1.00 (0.96-1.00) | 0.99 (0.95-1.00) | 0.14 / 0.25 / 0.62 | 0.23 / 0.29 | task 0, bond opened 0, opened at 1.5x 1, joint broken 0 |
| crank | 100 | 1.00 (0.96-1.00) | 1.00 (0.96-1.00) | 0.22 / 0.23 / 0.23 | 0.20 / 0.25 | task 0, bond opened 0, opened at 1.5x 0, joint broken 0 |

No joint broke or exceeded its tooth limit / 1.5 in any of the 400 trials. Every arm and walker
failure is a bond: the arm's base disc bond peels while lowering the block (the block then tips, which
also fails the task: most of the 16 task failures), the walker's hip and body bonds open in the teeter
at leg lift. The first Monte Carlo with 12-strand wires (drive 1.53 mN m) also had stalled joints (14 %
of arms missed); 18 strands fixed that; its partial rows are in `mc_8.25mm_12strand_partial.jsonl`.

What it takes to pass, and what changed against layout B:

1. **Arm.** The jaw squeezes at a set wire force (0.53 mN m, layout B's grip in units of m g L; a
   regulated, partial heating pulse, E) instead of stalling at full drive; at full drive the fixed
   finger's bond reached 1.03 of its strength. The fixed finger bonds to the wrist by a plain face.
   Moves wait for the joints (the controller counts steps).
1. **Crane** passes unchanged; the slew waits for slow (hot-ambient) joints.
1. **Crank.** Layout B's crank hangs its rod and piston off one disc bond: 2.9x that bond's strength
   at 8.25 mm. Redesigned to rest on the table: a frame bolted overhead turns the hub; the crank arm's
   lower disc is the crank pin; a four-cube rod lies on the table; the piston rides on the rod end's
   upper disc between rails on the line through the crank axis. Two things it needs: a smooth sheet
   and rails (friction 0.25, A), because a rod pushed at its top on a table with friction above 0.5
   tips instead of sliding; and axial play at the crank pin (0.05 mm and the tilt it allows, A), so the
   frame, crank and rod are not a rigid loop through the table. Peak bond load is then 0.19.
1. **Walker** as layout B, wave gait. Its weak point is the gait (section 1). Tried and dropped:
   duty 0.85 (worse), a 19-cube tripod hexapod (its five-cube body sags, 4.7x).

Video: `hardware/handoff/media/small-joint.mp4` (arm, walker, crank at 8.25 mm, captioned).

## 5. Idealisations (all E or A)

1. Cubes are rigid boxes welded at their bonds; a bond's load is read from the weld. A bond fails in
   two views: the 5 ms average of load over strength (bending / peel strength + tension / pull
   strength, or twist / twist strength) above 1, and a bond-opening model (Newmark sliding block: the
   excess load turns the two sides apart about the load's axis, each side about its own centre of
   mass, conservative; rigid-plastic; failed at a far gap of 0.1 d0 in peel, d0 / 4 in twist). Pass at
   1.5x means neither the opening model at strength / 1.5 nor the tooth load at limit / 1.5 is
   exceeded.
1. Bond strengths come from magpylib forces on free N52 spheres (derated 0.8, A): corner spheres
   1.03 mm, four face-ring spheres 1.16 mm on every face. Plain bond: 1.19 N, 4.5 mN m peel, 2.7 mN m
   twist; disc bond: 0.67 N, 2.5 mN m peel, twist through two steel pins (E).
1. The stepper: the ring moves in whole teeth along a minimum-jerk stroke; coupling spring
   1.48 N m/rad (lock path at 2e5 N/m, E), clearance 3.7 mrad. No wire fatigue, no wire creep, no
   temperature build-up over many steps (cooling time is varied 0.8-2x instead).
1. Table contact: time constant 1.5 ms (Hertz on a 0.2 mm printed corner, E). Stiffer contact makes a
   foot rolling over a cube edge chatter from one time step to the next and inflates bond loads up to
   1.3x. The walker runs at 2e-4 s, the rest at 1e-4 s.
1. Gravity ramps in over 0.2 s; loads count after 0.25 s. Cubes are placed, not assembled.

## 6. Packing (E)

The envelope at 8.25 mm (`hardware/scripts/module_phys/small_joint.py envelope`) puts both ratchet rings
under the discs, pawls and lock on one side, wires along the edges, 24 face-ring spheres on the face
diagonals and the reaction-wheel parts of concepts.md pick 1. Parts 209 mm3 against 413 mm3 inside,
packing 1.58 (rule: inner^3 x 0.8 / parts >= 1.5). No overlaps in the envelope.

| L (mm) | packing | mass (g) | weakest swarm margin |
| --- | --- | --- | --- |
| 7.0 | 1.13 | 1.00 | climb hinge 1.66 |
| 7.5 | 1.31 | 1.11 | climb 1.70 |
| 8.0 | 1.49 | 1.24 | spin time 1.62 |
| 8.25 | 1.58 | 1.30 | spin time 1.51 |
| 8.5 | 1.66 | 1.37 | spin time 1.41 |

## 7. The same actuator beyond the twist joint (for the reaction-wheel track's findings)

The reaction-wheel track found that ballistic climbs, descents and splits fail, so they must be
quasi-static, that permanent bonds (1.5/2.5/3.8 mN m to peel at 7/8/9 mm) must be releasable, and
that a lone cube needs a reaction against the floor. What this joint can do (E, not simulated):

1. **Edge pivot: no new output needed for climbs and descents.** A cube that rides on a neighbour's
   horizontal disc swings about that disc's axis, over the neighbour and onto its top (layout B's
   climb). Need: m g P = 0.11 mN m at 8.25 mm, against a 2.30 mN m drive, 21x; it also covers the
   94 uN m edge lift at 8 mm (16x). It needs a neighbour with a disc on the right face and a bond that
   lets go at the start and end. A true edge hinge would need a crown-gear take-off from the ring at
   module 0.2 or less, the printability problem of the gear route; not recommended.
1. **Bond release: a separate SMA plunger, not the ratchet.** The ratchet sits at the two disc
   faces and cannot reach the corners. The release the reaction-wheel track proposes (push the far
   spheres 0.2-0.3 mm deeper, pull falls to about 0.4x) is the stroke one wire already makes: 0.18 mm
   at 3 % of 6 mm, x1.4 lever = 0.25 mm, with 1.05 N against 0.13 N a corner pair. One plunger a corner
   (8), each retracting its sphere along the diagonal: about 3.5 mm3 a corner with wire and crimps,
   28 mm3 in all. Corners are shared by three faces, so releasing one face also weakens two corners
   of each of its four neighbours for that moment (their ring spheres still hold); the planner must
   allow for it, or the corner spheres move to face-only pockets (more spheres, more volume).
1. **Floor yaw: free.** A cube resting on its disc face (disc 0.33 mm proud) turns its body on the
   disc: disc grip on the table about 16 uN m against a bearing drag of about 14 uN m, too close to
   rely on (E); a rubber ring on the disc rim (A) would settle it.
1. **Floor translation: a kick leg through a selector.** A plunger cannot tip the cube (the push
   point must travel about 6.5 mm). The ratchet ring can: a shift fork (a bistable SMA latch, two more
   wires) moves a sliding dog from the disc hub to a cam that swings a foot out of a face and tips the
   cube over an edge. Need: m g L / 2 = 53 uN m, 29x under the drive. Speed: about 4 s a roll at
   27 deg/s, about 2 mm/s (E). Lone, off a chain, the CeraCharge gives about 60 steps (1.5 J / 25 mJ),
   one every 6 s: lone translation is a few cube lengths a charge at 8 mm, about 2,800 steps on the
   11 mm LiPo.
1. **What it adds:** release plungers 28 mm3, a selector on one joint about 8 mm3, kick cam and foot
   about 15 mm3, about 51 mm3 (E). Parts go from 209 to 260 mm3; packing at 8.25 mm falls to 1.27.
   Keeping the reaction wheels, the cube grows to about **8.9 mm** (packing 1.5 needs inner^3 x 1.24).
   If the reaction wheels go (no ballistic moves left for them), their rotors and stators free about
   35 mm3 (E) and the cube stays at about **8.4 mm**. Spin time stops binding either way.

### Battery: required (user, 2026-11-15)

A battery stays in (an earlier "no reserve" option is withdrawn). Either a get-home reserve at 8.25 mm
(1-3 CeraCharges, 0.5-1.5 J, about 5-15 kick-leg rolls; about +0.4 USD a cell at 1M, so about 1.4 USD
a stripped cube with one cell) or real autonomy at about 11 mm (10 mAh LiPo, about 2,500 steps at a
third of full speed off the bus; about 1.5-2 USD at 1M). Both keep a pulse capacitor: neither cell
supplies 0.5 A pulses.

### Joints in parallel and industrial fit (E, judgement)

1. One rotary axis: at most two joints share a load (a link bonded between two fixed cubes' discs on
   the same line), about 2x torque; joints on parallel, offset axes fight. Through linkages (each
   cube's crank into a shared slider) force and power add linearly. Holding adds with cross-section:
   tension about k^2 x 1.19 N (a 10 x 10 bundle of 8.25 mm cubes, about 120 N), bending about k^3.
1. Power: 1.1 mW out for 0.23 W in per joint (0.5 %). A thousand-cube machine: about 1 W of motion for
   230 W of heat. Not a power actuator.
1. Fits: reconfigurable fixtures and work-holding (form, then hold with zero power), conformable
   grippers, reconfigurable moulds and shape displays, slow precise positioning in tight spaces.
   Not: continuous or fast motion, real loads, hot environments (keep ambient under about 50-60 C
   for 90 C wire).
1. For industrial work: the same lattice at 20-50 mm with self-locking gearmotors (they fit from about
   22 mm; the 40 mm geared cube manages about 0.8 N m), about 100-1000x the power a joint.

### Chain power with a get-home reserve (E, not simulated)

The cluster is the power bus (corner spheres and diodes); each cube keeps a reserve only to get back
when it loses the bus, and the bus tops it up. That needs 1-2 J, not the hundreds a machine uses.

1. Lone roll: 22 uJ of work to tip an 8.25 mm cube; at SMA efficiency the floor is about 5 mJ. Through
   the main joint wires a 90 deg kick cam takes about 21 steps x 37 mJ = 0.8 J (two rolls on 1.5 J): too
   costly. A dedicated 2-strand kick actuator (about 4 mJ a step, 0.26 mN m against 53 uN m) takes
   about 85 mJ a roll: about 15 rolls, 12 cm, on the three CeraCharges already in pick 1; about 25 min
   to recharge on the bus at 1 mA. On the 11 mm LiPo, about 1,500 rolls.
1. Pulses from a per-cube supercap (CPH3225A class, about 7 mm3, A); the bus carries only the average,
   tens of mA a cube, which also keeps amps out of the sphere point contacts.
1. The bus should be a mesh (two or more powered bonds a cube where possible) so one bond letting go
   does not drop everything downstream. A ten-cube machine with three joints stepping draws about
   0.7 W from the bus.
1. Head home on a reserve threshold (distance back plus margin); find the cluster by its IR beacons.
1. Adds the supercap and kick actuator to the section-7 budget: about 8.4 mm without reaction wheels,
   8.9 mm with them. To simulate next: kick-leg roll energy and reliability, return and dock.

### How the power chain would work (designed on paper, A/E)

1. Contacts: nickel-plated N52 corner spheres meet in pairs at shared lattice corners, about 0.13 N
   a pair; two contacts a rail a bond (10-50 mOhm each, A).
1. Polarity: a cube's corners alternate parity, so fixed polarities would short half the
   placements. Every corner rectifies into the rails (Schottky to cold-start, then MOSFET ideal
   diodes); the cluster drives even world lattice points + and odd ones -; a powered cube infers its
   parity from its input corners and sets its other corners with MOSFET half-bridges. A cube unsure of
   its parity only receives.
1. Voltage: a 5-9 V bus with a per-cube regulator cuts current and drop (5-8 mm3, E); Schottky-only
   would lose about 0.3 V a hop.
1. Current: per-cube supercaps take the 0.47 A pulses; the bus carries averages and the machine's
   controller caps simultaneous steps (about 0.7 W for a 10-cube machine with three joints moving).
   Cubes near the source want two or more powered bonds.
1. Sources: a USB-tethered root cube, a battery tender cube, a powered dock, or a striped powered
   floor mat (every resting cube powered; stripe geometry not worked out).
1. Data: IR face to face for discovery and parity; optional signalling on the bus.
1. Bench tests first: sphere contact resistance and arcing at 0.1-0.5 A, drop along 10 cubes, and
   whether the per-corner switches fit.

### Pareto front up to one inch (analysis, E; only 8.25 mm SMA is simulated)

| L | route | worst machine joint margin | speed | arm move | battery | assembly | cost at 1M (A) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8.25 | SMA 25 um x 18 | walker 1.1, arm 1.6 | 27 deg/s | 3.1 J | CeraCharge reserve | very hard | 1.4-2.2 |
| 11 | SMA | walker 1.0, arm 1.4 (needs more strands; packing 1.31 with the LiPo) | 20 deg/s | 5.7 J | 10 mAh LiPo | hard | 1.5-2 |
| 14 | gears: flex-PCB flywheel motor, Didel worm, printed split ring | arm 2.2, walker 2.3, crane 4.5 | 133 deg/s no load | 0.4 J | LiPo, about 20x the moves of SMA | normal | 1.5-2.5 |
| 18 | gears | 1.3 | 133 deg/s | 0.8 J | larger LiPo | easy | 2-3 |
| 22-25.4 | gears or SMA | 0.4-0.7 | | 1.4-2.1 J | large | easy | 2.5-4 |

Machine loads grow as L^4, fixed-proportion joints as L^2-L^3, so past about 16 mm margins fall under
1.5 without a much larger motor (concepts.md's 40 mm N20 cube); bought micro gearmotors fit from 22 mm
but give 0.2-0.5 of the arm's need there. The knee is the 14 mm geared cube: self-locking worm,
5x the speed and 1/20 the energy of SMA, a real battery, bought gears. Caveats: analysis only; the
geared joint was sluggish in an earlier 12 mm MuJoCo run (reflected inertia about 0.01 kg m2); the
printed split-ring stage needs a test print; flywheel swarm moves fail above about 11 mm, so swarm
moves must be quasi-static on the joints. Next: the same machines and Monte Carlo at 14 mm with
ROUTE=gear.

## 8. Riskiest unknowns

1. **The 0.20 mm ratchet teeth in resin** (A): pitch, edge sharpness and wear after 10^5 steps. A test
   print and a bench step test come before anything else. The hold (4.2 mN m) is 2.5x the largest
   measured joint load, so a coarser pitch (0.3 mm: 6.3 deg steps) is affordable.
1. **Wire assembly at 25 um** (A): 16 crimps a cube, 12-strand loops. Yield and strand load sharing
   are unknown; one slack strand cuts the pull by 1/12.
1. **Bond strengths** (E): magpylib on ideal spheres, derated 0.8. The walker's margin depends on them
   directly; Monte Carlo covers 0.75-1.10x.
1. **Walker gait margin**: the teeter at leg lift is a property of the 4-leg wave gait; a gait that
   shifts the body over the support triangle before each lift, or a wider rear stance, is untested.
1. **Power**: chain power through the corner spheres at 0.23 A a wire pulse, a step at a time per
   cube; with four wires a joint and two joints, peak 0.5 A a cube if two fire at once (E).
1. **Heat**: 0.92 W pulses inside a 1.3 g cube; continuous stepping at 6 steps/s is about 0.23 W
   average. Cooling 0.8-2x is in the Monte Carlo, but steady temperature rise over a long walk is not
   modelled.

### Realism (judgement)

1. Sound: the torque budget (2x+ headroom in every run), power-off holding by a square-tooth pawl,
   SMA as the actuator (25 um Flexinol is sold, V), slow charging through contacts.
1. Unproven, bench-test first: 0.2 mm resin ratchet teeth (0.3 mm is affordable), sphere-bond
   strengths, the lone kick roll.
1. Not realistic as described at 8 mm: hand assembly of 18-strand 25 um loops and 16 crimps a cube;
   per-corner polarity switching beside the joints (needs a pro flex PCBA with 0201 parts); pulse
   current through 1 mm sphere point contacts. Working self-reconfiguring cubes are around 30-50 mm
   (M-Blocks, from memory); 8 mm integrated is a goal, not a plan.
1. Simpler power: a centre pad and an annular sprung ring on every face (same under any 90 deg turn),
   fixed polarity, no switching; disc faces need a slip ring. Not checked in detail.
1. Path: the same SMA ratchet at 20-25 mm with thicker, fewer wires; then 11 mm (LiPo fits, power
   works slowly); 8.25 mm last, with pro assembly and welded or bonded wire ends.

## 9. Files

1. `hardware/src/beebot/hardware/geared_joint.py`: gear train, SMA stepper, bonds, MuJoCo builder and rig
   (stepper and gear backends, bond loads and the bond-opening model).
1. `hardware/scripts/small_joint_machines.py`: arm, walker, crane, crank; `run`, `mc`, `summary`, `video`.
1. `hardware/scripts/module_phys/small_joint.py`: route analysis against L, NSGA-II, envelope, mass fit.
1. `hardware/runs/module_phys/joint/`: `results.jsonl` (with `min_edge_mm`), `analysis.json`,
   `machines_8.25mm.jsonl`, `mc_8.25mm.jsonl`, logs.
1. `hardware/module_phys/sma_joint_cube_8.25mm_envelope.step`.
1. `hardware/tests/test_geared_joint.py` (9 tests, 8 s).
