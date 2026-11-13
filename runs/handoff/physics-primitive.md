# physics-primitive: the smallest unit that carries machine function

Written 2026-11-13. A design memo; no source, hardware file or `ops` node changed. It continues
`runs/handoff/physics-options.md` (the options and their costs) and does not repeat its tables.
Inputs: that memo, `runs/handoff/machine-parts.md`, `runs/handoff/joint-strength.md`, HARDWARE.md
("Redesign options", "The hinge cube in MuJoCo", "Adopting the hinge cube", "Joint strength"),
`scripts/physics_rig.py`, and `data/machine_parts`. One new script, `scripts/physics_primitive_probe.py`
(numpy, CPU, about 6 minutes), produced the numbers marked "measured here". Every number is marked
measured (with its file) or guess; arithmetic of mine on measured inputs is marked "derived".

## Summary

1. The primitive is not a part and not a cell. It is the **bond between two cells**, and it needs
   three states: open, locked, and turning about one axis. A machine is then a set of rigid bodies
   of cells (locked bonds inside each body) joined by a few turning bonds. Everything else in the
   datasets is either a composition of those or outside what cubes can do.
1. The datasets say this themselves. Grouped by the joint they serve, 73.3% of MCB parts exist to
   lock two bodies together and 21.8% to let one turn against another (23.0% with gears, chains
   and pulleys); slides are 0.04%, springs 0.27%, fluid parts 2.8% and plain bodies 0.6% (measured
   here). In Fusion, rigid plus revolute is 90.0% of joints (`machine-parts.md`), and 97 of the 142
   jointed assemblies use nothing else (measured here).
1. The voxel shapes carry none of the function. At about 200 cells a spur gear is a toothless
   disc, a spring is two walls, a hinge is an L-shaped plate and a ball bearing is a ring with a
   bore of 2-3 cells (looked at here, one sample each). Teeth, coils, knuckles and rolling elements
   are below the grid. So function cannot come from building the shape better; it has to come
   from what the bond can do.
1. Two of the three states exist: open and locked (the face magnets). The hinge cube's latched
   edge is a turning bond too, but its axis lies on the boundary of both cells, so it stops at 180
   degrees. The one missing thing is a turn whose axis passes through the bond: **a face that
   turns about its own normal without limit and re-locks at quarter turns**. That is the options
   memo's rotating face (driven) and passive swivel cube (free) seen as one primitive; this memo
   argues it is the only addition needed, that its free state matters more than its driven one,
   and that sliding and spring cells are not needed.
1. The lock's load rating is a parameter of the same bond, not a second primitive. It is too low
   today for bending parts (cantilever rig passes 29-35%, `physics-options.md`), and depth and
   gussets substitute for part of it (`joint-strength.md`).
1. Not covered, by any cell-level primitive of plausible cost: sealed fluid handling (2.8% of
   MCB), elastic energy storage (0.27%), and, on today's evidence, toothed gears (1.2%): 0 of 38
   voxel gear placements transmitted rotation in a geometry check (measured here).
1. First experiment: an ideal turning bond in MuJoCo (a hinge joint between two cube bodies with
   today's face-joint limits) and three rigs: an axle under load, a roller carriage as a slider,
   and a four-bar with one driven joint. 1-2 CPU hours (guess), 2-3 days of code, no hardware.

## Families: what each must do

Shares: MCB from `data/machine_parts/mcb_parts.jsonl` grouped by me (`physics_primitive_probe.py
counts`, the `ROLE` table; all 68 classes assigned); Fusion from `machine-parts.md` unless marked.
"Bond state" is the answer this memo argues for; the test column is how the function reads in
physics without reference to what the part looks like.

| family | share | what it must do | test in physics | bond state that does it |
| --- | --- | --- | --- | --- |
| fasteners, pins, keys, snap rings, clamps, collars, spacers | MCB 73.3% (17,751 parts, 35 classes); Fusion rigid joints 69.2% | hold two bodies as one against tension, shear, bending and torsion; keys also pass torque from shaft to hub | pull, shear and twist the two bodies; no relative motion up to the rated load | locked, with a load rating |
| hinges, articulations, levers, knobs, castors, wheels, switches | MCB 9.1% (2,204) | let one body turn about one axis relative to another and block the other five motions | a small torque about the axis moves it; forces along the other five do not | turning |
| bearings, bushes, thrust washers | MCB 8.6% (2,090) | the same, under a radial or axial load, without limit | the same with a load across the axis, over full turns | turning (the bearing is the bond; no ring is built) |
| shafts, axles | no class of their own; rods are 47% of MCB shapes, 28% of Fusion | carry torque along an axis between two turning supports | twist one end, the far end follows; the supports stay put | locked bonds in a row (torsion), turning at the supports |
| motors, fans, impellers, turbines | MCB 4.1% (991) | turn continuously under their own power, or be turned by a flow | sustained rotation at a torque | turning, driven |
| gears, chain drives, pulleys | MCB 1.2% (281) | pass rotation between two axes at a ratio | turn one, the other turns by the ratio, no jam over full turns | two turning bonds plus tooth contact between bodies; not shown to work (below) |
| sliders, guides | MCB 0.04% (9); Fusion slider 6.1% of joints, cylindrical 2.2% | one translation free, the rest blocked | a small force along the axis moves it; the rest hold | composed: rollers (turning bonds) in a channel of cells, or a linkage |
| ball, planar, pin-slot joints | Fusion 0.6%, 1.0%, 0.1% of joints | 3 rotations; 2 translations and 1 rotation; 1 and 1 | as above, per free motion | composed: turning bonds in series; a body resting on a plane; a peg cell in a slot of cells |
| plates, frames, brackets, housings, hooks | MCB 0.6% (141); Fusion: 78.6% of bodies are in no joint | carry load across a span or around a corner; enclose | the span and cantilever rigs of `physics-options.md` | locked, with a load rating; depth and gussets |
| springs | MCB 0.27% (65); Fusion 36 names | force that grows with deflection and returns the energy | deflect, release, it comes back | none; a weighted lever on a free turning bond returns under gravity (see below) |
| fluid parts (fittings, plugs, nozzles, valves) | MCB 2.8% (683) | contain, route and gate a fluid | no leak; flow opens and closes | routing is shape; a gate is a quarter turn (turning); sealing is out of reach |

Three readings of the table:

1. **MCB is a catalogue of joint hardware.** 95.1% of its parts (lock 73.3% plus turn 21.8%) are
   pieces of one of two joints. `physics-options.md` concluded that fasteners are joints, not
   goals. I checked whether that conclusion should be overturned and it should be extended
   instead: bearings, bushes, hinges and articulations are joints too, for the same reason. A
   ring of cubes shaped like a bearing bears nothing, exactly as a bolt-shaped stack fastens
   nothing.
1. **Where the fastener conclusion is too generous** is its second half, that "the cube's face
   connector already performs" the fastener's function. A fastener's function includes its
   rating: preload well above the service load, and shear and torque taken by shape (dowels, keys,
   splines: 987 key parts) instead of friction. The face joint holds 53 N and 1.34 N m and takes
   shear and torsion by friction only (`joint-strength.md`), and the cantilever rig fails for
   65-71% of sampled parts (`physics-options.md`). So fasteners are a joint whose spec is not met
   yet, which is a statement about the lock's rating.
1. **Function is a mobility signature.** Every row's test is the same test: for a pair of bodies,
   which of the six relative motions are free under a small force, and what load the blocked ones
   hold. A hinge is whatever answers pushes like a hinge. This needs no family label and no
   look-alike shape, and Fusion's joint records (type and axis per body pair) give the target
   signature directly.

## What the turning bond is, and why the hinge cube's edge is not enough

1. **A full turn needs the axis to pass through the bond.** Body B can turn without limit against
   body A only if neither sweeps into the other, so the interface between them has to be a
   surface of revolution about the axis: a face turning about its own normal (ATRON, in
   HARDWARE.md's survey, is the one surveyed module built this way). The hinge cube's axis is a
   shared edge, on the boundary of both cells; B goes from one face of A to the next and stops:
   180 degrees. This is geometry, not motor size.
1. **Edge hinges in series do not add up to a shaft.** Two hinges with parallel axes give 360
   degrees of orientation, three give 540, but the winding is bounded by the sum of the ranges and
   the axis wanders. A swarm can re-grip (B walks round A by four convex pivots a turn), which is
   unbounded but is an orbit in steps, not a body spinning on a fixed axis, and each step is a
   move with its own failure rate: convex up and down reach 0.75 and harm 0.04-0.06 per move in the
   ten-cube walks (HARDWARE.md). Ten turns are 40 such moves; at 0.05 harm a move that is a 13%
   chance of no harmful move (derived: 0.95^40). Usable for a slow winch, not for a wheel, a fan
   or a gear train.
1. **So one primitive, in two placements.** An edge-axis turn (have it: it is how cubes move, and
   as a group pivot it gives doors, lids and levers) and a face-normal turn (missing: it gives
   everything that keeps turning). Both are "a bond that either holds or turns".
1. **Free matters more than driven.** Real machines have many passive joints and few motors (MCB:
   4,294 bearing and hinge parts against 732 motors, measured here). A mechanism with one degree
   of freedom needs exactly one driven joint; every other joint must give way, or the joints
   fight each other in any closed loop. The hinge cube's edge motor is taken as not backdrivable
   (a guess in `physics-options.md`), so a four-bar made of edge hinges would need four motors in
   exact step, which is a hand-coded controller per mechanism. A free turning bond lets physics
   solve the loop. Fusion assemblies with a free joint have a median of 3 of them (quartiles 1 and
   7, 95th percentile 14, measured here), so loops of several passive joints are the normal case.

## Primitive candidates

Costs: hardware and mass figures are the guesses of `physics-options.md` unless marked; the sim
column is MuJoCo plus the grid world; "policy" is what the learned controller must newly represent.

| candidate | families it covers | what it cannot give | cube hardware | sim | policy |
| --- | --- | --- | --- | --- | --- |
| rigid bonded cell (have) | plates, frames, housings; fasteners within 53 N and 1.34 N m a face (measured, `joint-strength.md`) | any relative motion; bending parts past 4-9 cells of cantilever (measured, same) | none | none | none |
| contact only: a shaft of cells loose in a hole of cells (no new hardware) | in principle bearings and sliders | a 1-cell shaft needs a 5-cell hole and has up to 1 cell of play, as wide as itself; a 7-cell shaft needs 37 + 40 cells a layer for the same 1 cell of play (measured here, `journal`); the faces toward the gap must be switched off; in a 16-cell world one joint uses most of a part | none | contact-rich, slow, jam-prone (HARDWARE.md: box-on-box edge contact stalls a third of turns) | none new, but one joint costs about 20-80 cells a layer |
| load-rated lock, about 270 N a face | fasteners to spec; cantilever rig 29-35% to 60-76% (measured, `physics-options.md`) | motion | +20-40 g if the edge latches can hold under load (guess); six face latches otherwise | one strength number per face | none (a goal filter) |
| edge hinge with group pivot (have the hinge) | swing joints: doors, lids, levers, clamps; about half of Fusion's revolute bodies (guess, `physics-options.md`) | more than 180 degrees; passive joints; bodies past about 2 cubes from a horizontal axis (derived there) | none | bond state per face, a body-turn move, torque and sweep checks | a new move type that must meet the bar (0.80 reached, 0.03 harmful) |
| **turning bond: a face that turns about its normal, free or driven, locking at quarter turns** | bearings, bushes, hinges, wheels, castors, knobs, shafts, motors, rotors, valve gates; composed: sliders, cylindrical, ball, planar, linkages, chains. MCB 21.8% directly; Fusion revolute plus cylindrical 424 of 568 free joints (`physics-options.md`) | sealing, elastic storage; gears unproven; fine motion below a cell | free only (passive swivel cube): $5-10, 100-150 g, no electronics (guess). Driven (rotating face): +40-80 g, +$15-30, one more motor and lock, likely the 8-10 cm cube (guess). Re-locking needs the detent within the 3 mm latch capture, about 7 degrees at the face edge | MuJoCo: one hinge joint per turning bond, native and cheap. Grid: a state per face (open, locked, turning), an orientation per cell, a swept-cylinder check | one more value in the per-face bond state; a spin command for a driven face; goals carry joint labels |
| sliding cell | sliders: Fusion 6.1% of joints, MCB 0.04% (measured) | rotation of any kind (translations compose only to translations) | rack or stepping-magnet faces; conflicts with flat faces and pivot edges; EM-Cube failed against gravity (HARDWARE.md survey) | a slide move while bonded | a new move type |
| compliant or spring cell | springs: MCB 0.27% (measured) | continuous rotation (bounded strain gives a bounded angle); it also softens every structure it is in and eats into the 3 mm capture the latches need | a flexure or spring layer per cell (guess: cheap alone, costly in lost stiffness) | soft constraints; stiff-and-soft mixes are slow to integrate | none |

## Recommendation

**One primitive plus the swarm: the bond with three states, open, locked and turning.** The cube
has two and a half of them. The addition is the face-normal turn, and its essential form is the
free one with a lock at quarter turns; drive is an option on the same unit.

Why it is the minimal one:

1. **Nothing cheaper produces it.** Rigid cells and contact alone give a joint with play as large
   as the shaft at any size that fits a 16-cell world (measured here), and the grid has already
   erased the teeth and knuckles that would have carried function by shape. Edge hinges stop at
   180 degrees. A lock does not move. A slide cannot be turned into a rotation. A compliant cell
   bends through a bounded angle.
1. **It produces the others, the way real machines do.**
    1. Slider: a carriage on rollers in a channel of cells is how linear guides are built; two
       hinge triplets at right angles (a Sarrus linkage) or a scissor give an exact straight line
       from turning joints alone (textbook kinematics, not measured here).
    1. Cylindrical, ball, planar: turning bonds in series, or a body resting on a plane. A
       three-turn wrist of cubes has its axes a cell apart, so it gives any orientation with
       some coupled translation: the function, not the form.
    1. Shaft: a row of locked cells between two turning bonds. Torsion is carried by friction at
       the faces, roughly 0.7 N m a face (guess: 0.5 x 53 N x 28 mm lever from the magnet layout
       in `joint-strength.md`), above the 0.3 N m motor.
    1. Wheel, rotor: a body balanced about the axis puts no gravity torque on the turn, so the
       0.3 N m motor's limit of 2 cubes out (group pivot, `physics-options.md`) does not apply;
       an unbalanced crank arm is limited to 1-2 cubes out on a horizontal axis (derived:
       n(n+1)/2 x 0.074 N m against 0.2 N m at the 1.5 margin).
    1. Spring, partly: a weighted lever on a free turning bond is pushed back by gravity in
       proportion to small deflections and returns the energy. It works only in one orientation
       and is weak; springs are 0.27% of MCB and I do not think they justify more.
    1. Valve gate: a quarter turn, which is the detent.
1. **The lock's rating is a parameter, not a primitive.** Strength composes from geometry up to a
   point (1 to 3 cells deep takes a cantilever from 4 to 9, `joint-strength.md`), and past that it
   is the bench question `physics-options.md` already poses for the edge latch. It should be
   pursued in parallel; it changes how many parts pass, not what kind of thing a machine is.
1. **What stays out.** Sealing (cube faces with chamfered edges leak; granular flow would work),
   elastic storage at useful stiffness, belts, and anything whose function lives in features
   finer than a cell. Gears are in this list until shown otherwise (next section). Together 4.3%
   of MCB (fluid 2.8%, gears, chains and pulleys 1.2%, springs 0.27%, measured here).

How this differs from `physics-options.md`: it listed the rotating face and the passive swivel
cube as options 6 and 4 to choose between after counting swing against continuous joints. I
argue they are one primitive, that the free state is the one linkages and bearings need and the
driven state is the one rotors need, and that the sliding joint and springs can be dropped now on
kinematic grounds, not only by share. The sim can adopt the abstract bond state (open, locked,
turning free, turning driven) before the hardware choice between "a swivel cube type beside the
hinge cube" and "a rotating face on every cube" is made; both look the same to the policy.

### Gears: the open risk (measured here)

`physics_primitive_probe.py gears`: two identical voxel gears on two fixed axes at cell centres,
2D geometry only, cells shrunk 0.02 a side, 2 degree grid; gear 1 is turned four times and gear
2 moves only when pushed (to the nearest free angle within 6 degrees, else a jam). Nine shapes
(crosses of arm 1-4, voxel discs of radius 1.5-4.2 with 4, 8 or 12 one-cell teeth; 5-69 cells),
every centre offset from just touching to just clear along a lattice axis and one cell off it:
38 placements.

1. 0 of 38 transmitted rotation. 17 never jammed but the teeth skip: gear 2 turned at most 0.046
   of gear 1's angle (a pair of equal gears should give 1). 15 jammed (7 within 36-84 degrees, 8 at
   the start). 6 had no free starting pose.
1. The reason is the lattice: centre distances come in whole cells and teeth are one cell wide, so
   a pair is either too close (the free band is narrower than the tooth pitch and pinches shut) or
   too far (the free band is wider than the pitch and the teeth pass).
1. Limits of this check: no dynamics or friction, a crude push rule (it prefers one direction
   when both are free), identical gears only, nine shapes. Pin wheels, unequal pairs and
   Geneva-style indexers were not tried. It shows that the obvious voxel gears do not work, not
   that none can.
1. Consequence: do not count gears as covered. Rotation can still pass between parallel axes at
   1:1 with coupling rods (turning bonds only, as on locomotive wheels); a ratio needs teeth or a
   wrap. If ratios matter to the user, that is a separate question (a gear-face cube type or a
   finer cell), and by count it is 1.2% of MCB and 57 Fusion names.

## The bitter-lesson check

1. **The interface is uniform and tiny.** Per face: open, locked, turning; per driven face: a
   turn command. No action names a part or a family. A bearing, a hinge, a wheel and a valve are
   the same action in different places.
1. **Operation needs no per-part behaviour.** Once built, a mechanism with free joints is run by
   driving one joint; physics does the rest. The alternative (every joint a position servo)
   needs a coordinated trajectory per mechanism, which is the hand-coded behaviour to avoid.
1. **The goal is data, not a rule.** A goal becomes occupancy plus a label on the faces between
   two bodies, taken from the dataset's joint record. Building it is the same problem the policy
   already has (reach the occupancy), plus setting bond states.
1. **The test is generic.** One mobility-signature test (push each body pair along and about
   each axis; compare free and blocked motions with the joint record; then load the blocked
   ones) replaces a rig per family for anything that moves. The span and cantilever rigs stay for
   static parts. The eight family rigs of `physics-options.md` are hand-made evaluations; this
   reduces them to two, both derived from the data.
1. **Where hand knowledge could creep back in, and should not:** the compositions above (roller
   carriage, Sarrus, coupling rods) are my existence arguments for coverage, not templates to
   generate data from. If mechanism goals are wanted, they should come from real joint records
   (two parts and their joint), not from a generator of textbook mechanisms.
1. **The data is thin for this.** 100 assemblies in archive 04 have a free joint (measured here)
   and whole assemblies voxelise badly at 16 cells (2 of 6 readable, `machine-parts.md`). The
   natural source is the Fusion Assembly-Joint set (pairs of parts with their joint, 2.8 GB,
   skipped so far per `machine-parts.md`): each sample is two bodies and one bond label, which is
   exactly one instance of the primitive.

## Falsification experiments

The claim to break: "rigid cells with open, locked and face-normal turning bonds reproduce the
mobility signature of the lock and turn families and of sliders, at sizes that fit a 16-cell
world and with today's face-joint strength." CPU costs are guesses from HARDWARE.md's run times
(the hinge closed loop ran near real time: 153 s simulated in 159 s).

| id | experiment | falsified if | cost | hardware |
| --- | --- | --- | --- | --- |
| F1 (first) | MuJoCo, ideal turning bond: a hinge joint between two cube bodies; the faces either side keep `joint_sim`'s magnets and limits. Three rigs, 3 sizes, 20 jittered seeds each (180 runs of about 10 s): (a) axle: a cart of 4 one-cube wheels and a chassis of 4-16 cubes, one wheel pair driven at 0.3 N m, rolls 10 cells on the floor; (b) slider: a carriage on 4 free rollers in a channel of cells, pushed 6 cells and back, side load 1 W; (c) four-bar: three free turning bonds and one driven, 20 cycles | any of: the wheel's face bond opens under the chassis weight or the motor stalls at 4 cubes of chassis; the carriage jams or leaves the channel in more than 3% of runs (the harm bar); the four-bar jams with free joints. One failure removes that family from the claim; (a) failing removes the primitive's main use | 1-2 CPU hours (guess); 2-3 days of code | no |
| F0 (null, same code) | the same three rigs with today's cube only: edge hinges, group pivots, re-gripping | the turning bond is unnecessary if today's cube passes (a) or (c) at the bar. I expect it cannot (the 180 degree argument and 0.95^40) | included in F1 | no |
| F2 | gears in MuJoCo with dynamics: the best non-jamming placements above, plus pin wheels and unequal pairs; 4 turns under a load torque | gears are covered if any pair holds its ratio within 15 degrees of backlash over 4 turns in 19 of 20 seeds; otherwise they stay out | about 1 CPU hour (guess; M5 in `physics-options.md`) | no |
| F3 | policy: goals with one bond label (two bodies from a Fusion joint pair, one turning bond), reverse demos, a short Mac run by validation loss | the policy cannot place and set a turning bond where the label says, at shape-goal success rates | Mac GPU, 1-2 hours a comparison; needs the Assembly-Joint download first | no |
| F4 | bench: one face on a bearing with a 90 degree detent between two cubes: re-lock rate against the 3 mm capture, free-turn friction torque, holding moment across the turning face | re-lock under 0.97 a turn (guess at a bar: the harm bar per move), or friction torque above about 0.05 N m (guess: the gravity torque of one cube about an edge), or the mechanism does not fit beside 12 edge latches | parts and days on the bench | yes |

F1 is the smallest experiment that can falsify the recommendation and it needs no hardware. F4
is the only one that needs hardware and it decides packaging (swivel cube type or rotating face),
not whether the primitive is right.

## Caveats

1. The MCB grouping by role is mine and coarse: pins are counted as locks though clevis pins are
   hinge axes, thrust washers as turning, sockets as locks, switches as turning. Moving pins
   wholesale to "turn" would shift 4,829 parts (19.9%) between the two states and change no
   conclusion, since both are states of the same bond.
1. Counts by part are not importance. Gears and springs are 1.5% of MCB together and central to
   many machines; the memo leaves them uncovered on purpose and says so.
1. Fusion numbers are archive 04 of 11 (751 assemblies, not a random draw). Joint axes are stored
   in joint-local frames ("ZAxisJointDirection" for 394 of 405 revolute records, measured here),
   so whether real axes line up with lattice axes was not measured; a face-normal turning bond
   only offers lattice axes.
1. Swing against continuous is still unmeasured (M6 of `physics-options.md`: 26 of 379 revolute
   joints have both limits set). It sets how much the face-normal turn adds over the group pivot;
   it does not change that free passive joints are needed for linkages either way.
1. The voxel shapes looked at here are five single samples at 150-300 cells (spur gear, ball
   bearing, spring, hinge, bush), read by eye from mid-height slices; `machine-parts.md` reports
   that a toothed gear and a lockwasher kept their teeth at 250 cells in its renders, so teeth
   survive on some parts and scales.
1. The gear and journal checks are 2D geometry with no dynamics; the torsion and crank figures
   are arithmetic on the modelled 5 cm, 150 g cube, not simulations. Nothing here was run in
   MuJoCo.
1. Kinematic completeness (revolute joints generate straight lines and general plane curves) is
   a statement about ideal joints. With one-cell links, lattice axes and 53 N faces, a composed
   slider may be too bulky or too weak to be worth building; F1 (b) is there to find out.
1. All hardware costs are guesses carried from `physics-options.md`; no mechanism for a turning
   face has been drawn. HARDWARE.md expects the first build at 8-10 cm, and capacities in cube
   units hold only if motor and magnets scale with it.
