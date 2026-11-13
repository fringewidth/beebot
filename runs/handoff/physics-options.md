# physics-options: what "same function" means for machine parts, and what the cube needs for it

Written 2026-11-13 for a brainstorm. A design memo, not an implementation: no code or hardware file
changed. Inputs: `runs/handoff/machine-parts.md` (the data), `runs/handoff/joint-strength.md` and
HARDWARE.md "Joint strength" (what the joints hold), HARDWARE.md "Redesign options" and "Adopting the
hinge cube" (the cube and the bar: >= 0.80 reached, <= 0.03 harmful per move), `vla/voxel/world.py`
(`rule_table`, `overhangs`, `supported`), `hardware/cube_hinge.py`, `hardware/joint_strength.py`.
Numbers marked "guess" are mine and unmeasured; everything else is from the repo or measured here.

## Summary

1. By count, the datasets are mostly parts whose function is static: fasteners and pins are 72% of
   MCB, and 91.5% of Fusion bodies take part in no free joint. For those, "same function" is a joint
   or a structure carrying a load, and today's sim and cube can be tested on it now.
1. A part alone has no function. A beam spans, a hinge lets one body turn relative to another, a
   bearing holds a shaft. Every functional test is a rig: the part, what it attaches to, a load or
   a drive, and a pass condition. The rig is the definition of "same function"; it needs one rig per
   family (eight below), not per part.
1. Measured here on 600 sampled part variants (median 108 cells, 11 cells long): resting on their
   largest face 98% stand, which tests nothing. Put in a rig with one module weight of load, 76-77%
   work as a beam on two pillars with today's joints, but only 29-35% work as a cantilever from a
   wall (3 of 244 of 12 cells or longer). Face latches of about 270 N (x5) lift the cantilever share to
   60-76%; the floor latch changes nothing in these rigs.
1. The hinge cube is already a servo hinge: a latched edge plus a 0.3 N m motor, 90 or 180 degrees
   about any of its 12 edges. Letting it carry the cubes bonded to it (a "group pivot") turns that
   into a hinge between two sub-assemblies with no hardware change. The motor limits the moving
   body to about 2 cubes out from the axis per driving cube on a horizontal axis (more on a vertical
   axis, where gravity does not load the turn).
1. What no option short of new hardware gives is continuous rotation: wheels, shafts in bearings,
   gears, pulleys, rotors. That is 75% of the non-rigid joints in Fusion (revolute and
   cylindrical, 424 of 568) and 12-19% of MCB. The rotating face (one cube face that spins on a bearing, locked at 90 degree
   detents) is the one addition that covers it; a passive swivel cube is the cheap alternative.
1. Recommendation: first, function rigs plus the group pivot (software and MuJoCo only: about 9
   hours on 4 threads for the rigs over the library, 2-3 CPU hours for the carry and latch studies);
   in parallel, make the hinge cube's edge latch load-rated on the bench; then decide rotating face
   against passive swivel cube on two MuJoCo studies (3-4 CPU hours) and a swing-or-continuous count.

## What the data asks for

MCB dataset A (24,215 parts in 68 classes; no assemblies, no joints, no real size), grouped by
function. Fusion 360 archive 04 (13,207 bodies in 751 assemblies; 1,843 joints in 142 assemblies;
3,008 bodies voxelised). Fusion has few labels (37.9% of bodies have any human name), so its
families come from joints and measured shape.

| family | MCB classes (parts, share) | Fusion evidence |
| --- | --- | --- |
| fasteners | rivets, bolts, screws, nuts, studs, eye screws, keys and splines, clamps, snap rings, split pins, lock and convex washers: 12,666 (52.3%) | names with screw, bolt, nut or washer: a few hundred bodies |
| pins | grooved, cylindrical, taper, roll, locating: 4,829 (19.9%) | names with pin: 71 |
| hinge and revolute bodies | articulations and eyelets 1,632, lever 206, knob 131, castor 99, hinge 54, wheel 48: 2,170 (9.0%) | revolute: 847 bodies (6.4%), 384 joints (20.8% of joints) |
| bearings, bushes, rings | bushes 764, flanged block bearing 404, thrust washers 466, radial ball bearings 239, flanged plain 110, accessories 107, collars 52, spacers 22: 2,164 (8.9%) | names with bearing/bush: 39 |
| motors | helical geared motors 732 (3.0%) | names with motor or stepper: a few dozen |
| fluid parts | fittings, plugs, nozzles, valves: 683 (2.8%) | - |
| gears and transmission | spur 88, angular 60, toothed 9, chain drives 100, pulleys 24: 281 (1.2%) | names with gear: 57 |
| rotors | fan 213, impeller 29, turbine 17: 259 (1.1%) | - |
| plates and structure | plates 73, square 14, hook 54: 141 (0.6%) | measured shape of 3,008 bodies: plate 28%, rod 28%, block 24%, cube 12%, strip 9% |
| springs | springs 65, spring washers 11: 76 (0.3%) | names with spring: 36 |
| sliders | plain guidings 9 (0.04%) | slider: 184 bodies (1.4%), 113 joints (6.1%) |

Joint facts that matter for the options:

1. Fusion parts: 78.6% in no joint, 12.9% only rigid, 8.5% in a free joint (revolute 847, slider 184,
   cylindrical 73, planar 33, ball 22, pin-slot 9 bodies).
1. Revolute joints with both limits enabled: 26 of 379 (13 within 90 degrees, 12 within 180, 1 more).
   The other 93% have no enabled range. Designers often leave limits off, so this does not prove
   continuous rotation, but the labelled revolute bodies read as a mix: wheels, tyres, rims,
   crankshafts, a stepper, planet gears, bushings and plain bearings (continuous) beside doors,
   pliers, clamps, levers, thigh and shin links (a swing). A guess: about half need continuous
   rotation. Measuring it is cheap (M6 below).
1. Free-joint bodies by shape (meshed ones): plate 84, block 76, rod 71, cube 33, strip 19 for
   revolute; plates and rods dominate, i.e. doors, lids, links and shafts.

## Same function, family by family

The pass condition in each rig is checked after the swarm builds the shape on the grid; static rigs
run in `joint_strength` (validated against MuJoCo to a median 4.7%), anything that moves runs in
MuJoCo (`joint_sim`, `cube_hinge`), because the grid has no state between lattice poses. "Today"
means the voxel world as it is (`switch_cube` rule, face-connected support, the overhang rule) and the
hinge cube as modelled (5 cm, 150 g, 53 N and 1.34 N m per face joint, 0.3 N m motor, latch floor for
moves, steel floor pull 0.3 of a face).

| family | function | functional test (rig and pass) | today: sim | today: cube |
| --- | --- | --- | --- | --- |
| beam, plate | carry a load across a span | ends on two free one-cube pillars, 1 module weight (W) at midspan on top; load factor >= 1.5 (rigid blocks do not deflect, so "deflection below d" becomes "no face opens 1 mm", MuJoCo's failure rule) | check exists, not wired into goals or the world (`world.supported` passes any face-connected shape) | spans to 16-18 cubes (80-90 cm) 1 deep on pillars; fine in compression |
| bracket | carry a load around a corner | one end welded to a wall, 1 W at the far top; factor >= 1.5 | same | cantilever 4 (1 deep) to 9 (3 deep); a 3-deep gusset makes an 8-long arm hold (MuJoCo) |
| shaft, pin | an axis something turns about (or, for dowels, locating two parts: a rigid joint) | a body B on the shaft turns >= 90 degrees (or a full turn) about the shaft's axis, the axis moves < 0.1 L, B stays attached | no relative rotation except one cube's own pivot | only a cube edge can be an axis (the hinge latch), and only for the cube itself; a square shaft cannot turn in a square hole |
| hinge (revolute) | one rotational DOF between two bodies | B turns 90 degrees about the axis relative to A and back, holds at both ends under 1 W, no face of A or B opens; for a free hinge, B also swings when pushed with a small torque | single-cube pivots only; no body moves as a unit; no bond state (every face contact is a bond) | passes for a one-cube B (that is the hinge cube's move); a servo hinge, not a free one (geared, not backdrivable: guess) |
| slider | one translational DOF | B moves >= 1 cell along the axis, stays attached, nothing else moves | one cube rolling along a row is a stepped slider (latched at every instant; the carriage turns 90 degrees per step); a multi-cube carriage only by moving its cubes rear to front, which keeps the shape every few steps but is not a rigid body | same as the sim; no true slide |
| gear | transmit rotation at a ratio | turning gear 1 by a set angle turns gear 2 by -r1/r2 of it (backlash < 15 degrees, guess at a bar), no jam over a full turn | fails: nothing turns continuously | fails |
| bearing, bush, ring | hold a shaft: rotation free, radial motion blocked | shaft turns inside under a radial 1 W, play < 0.1 L | fails | fails: a 1-cell square shaft needs a hole of sqrt 2 cells to turn, i.e. 0.2 L of play a side |
| spring | store energy | deflection proportional to load and returns | out of scope: cubes are rigid and the magnet force falls past its peak, so a joint has no restoring range. A hinge cube under low-gain position control is a virtual torsional spring (stiffness up to 0.3 N m per its angle range), with no stored energy | out of scope |
| fastener | hold two parts together | the joint between A and B holds tension, shear and moment set by the assembly | the face joint is the fastener: 53 N clamp, 1.34 N m about an edge, about 27 N shear by friction | a bolt-shaped stack of cubes fastens nothing; the cube's own connector does the job |

Two consequences:

1. Fasteners and dowel pins (72% of MCB) are not build goals with a function. Their function is
   the joint, which the cube's face connector already performs. In an assembly they become a
   strength requirement on the faces between two parts (a rigid joint with a holding force); as
   standalone goals they are shapes only. This also removes the bolt-versus-frame scale problem
   (`machine-parts.md`: a bolt is 1-4 cells where the frame is 16).
1. A part's resting pose tests nothing. Voxelised parts rest on their largest face, so they are
   nearly all in compression: 235 of 240 sampled variants (40-120 cells) stand at a load factor of
   1.5 as built. The function shows only in a rig (next section).

## Measured here: function rigs on sampled parts

Script: `joint_strength.check_many` on 300 variants per source, one per part, 40-300 cells (median
108-109), lying flat, chosen at random (seed 1); run time about 10 minutes per source and joint
variant on 4 threads. Two rigs per variant, both with self-weight plus 1 W, pass at load factor
>= 1.5 (gravity mode):

1. Span: the part lifted one cell, its two end slices along the longer horizontal axis on free
   one-cube pillars, 1 W on top at midspan (the validated "beam on free pillars" family; the check
   is optimistic there by 2-23%).
1. Cantilever: one end slice held fixed (a wall), joined to the rest by ordinary face joints, the
   rest free, 1 W on top at the far end.

| source, joints | span passes | cantilever passes | both | cantilever by length: 1-5 / 6-8 / 9-11 / 12-16 cells |
| --- | --- | --- | --- | --- |
| MCB, as built | 230 (77%) | 106 (35%) | 91 (30%) | 30/33, 59/92, 14/67, 3/108 |
| MCB, latch floor | 230 | 106 | 91 | unchanged |
| MCB, face latches x5 + latch floor | 250 (83%) | 229 (76%) | 199 (66%) | 33/33, 86/92, 53/67, 57/108 |
| MCB, bolted lattice x20 + keys (not run on Fusion) | 272 (91%) | 287 (96%) | 265 (88%) | 33/33, 91/92, 63/67, 100/108 |
| Fusion, as built | 228 (76%) | 86 (29%) | 75 (25%) | 31/31, 48/68, 7/65, 0/136 |
| Fusion, latch floor | 229 | 86 | 75 | unchanged |
| Fusion, face latches x5 + latch floor | 251 (84%) | 181 (60%) | 160 (53%) | 31/31, 65/68, 50/65, 35/136 |

1. **Beams mostly work today; cantilevers do not past about 8 cells.** That matches the joint-strength
   study (a 1-deep cantilever holds 4-6, a 3-deep one 9); the parts are deeper than 1 on average,
   which is why lengths of 6-8 still pass two times in three.
1. **Thin parts span but do not cantilever.** Variants under two cells thick: 63 of 77 (MCB) and 82
   of 96 (Fusion) pass the span, 2 and 2 pass the cantilever. Plates: 5 of 82 and 4 of 104
   cantilever. A plate bracket or a shelf is the case that needs latches.
1. **The floor latch does nothing here** because these rigs fail at midspan or at the wall root,
   not at the floor; its value is for free-standing columns under side load (6 -> 18 cubes,
   joint-strength study), which these rigs do not include.
1. 18-22% of the as-built checks did not converge (more with x20: 52%); their factor is a certified
   lower bound, so every pass count above is a lower bound.

## Options, cheapest to most radical

"Unlocks" is the function each option adds on top of the ones above it. Shares are of the counts
in "What the data asks for". Hardware masses and costs are guesses unless a source is given.

| # | option | sim change | cube hardware change | unlocks (share of data) | risk |
| --- | --- | --- | --- | --- | --- |
| 1 | rigid only, function by structure | function rigs per family; `joint_strength` as a goal gate; fasteners and pins become joint specs; no new moves | none | beams, plates, brackets within today's strength (span rig 76-77% of sampled variants, cantilever rig 29-35%); fasteners and dowels via the face joint (MCB 72%); a one-cube hinge and a one-cube stepped slider | low; the rigs are the only new definition |
| 2 | group pivot: a hinge cube turns the body bonded to it | bond state per face (6 bits a cell); a move that turns a bonded body about the mover's latched edge by 90 or 180 degrees, with swept-volume, motor-torque and `joint_strength` checks; the overhang rule applied to the body | none new: firmware keeps the body's internal bonds on and releases its bonds to the rest; several cubes on one axis drive together | swing hinges (doors, lids, flaps, levers, clamps, legs), up to 2 cubes out from a horizontal axis per driving cube; carrying passive cubes into place (needed by 4 and 5). Share: about half of Fusion's revolute bodies (guess) and MCB hinge and lever (1.1%) | latch strength under a cantilevered body (unmodelled); several motors in step; every new move type has to meet the bar |
| 3a | static floor latch | floor and terrain pull 1.0 of a face in the check | low: the latch features the floor moves already need, used to hold | free-standing parts: column with 1 W at the top 6 -> 18 cubes; no change in the span and cantilever rigs | low |
| 3b | load-rated face latches, about 270 N a face (x5) | per-face strength in the check | high as six face latches; moderate if the hinge cube's 12 edge latches are made to hold under load (two opposite edges of a face give about latch force x L of moment): guess +20-40 g | bending parts: cantilever 4 -> 10, bracket 7 -> 13, beam on pillars 16 -> 26 (39 with 3a); cantilever rig 29-35% -> 60-76% of sampled variants, both rigs 25-30% -> 53-66%; also strengthens every hinge of option 2 | latch wear and capture; mass lowers capacity in module weights |
| 4 | passive swivel cube | a second cell type with no moves, an orientation state (4 detents); moved only by group pivots | a passive cube of two halves on a bearing about an axis through two opposite face centres, steel faces, a spring-ball detent at 90 degrees: guess $5-10, 100-150 g, no electronics | free revolutes about face normals: passive hinges, pendulums, linkage joints with a hinge cube as the driver, axles; bearing function moves into the cube. Continuous rotation only if something drives it | needs option 2 to place it; a free joint makes carried bodies floppy (the detent must hold during carries); no drive of its own |
| 5 | two or more cube types in one swarm | cell type in the observation, rules per type, the planner chooses types | passive types (swivel, latch block, rail) beside the active hinge cube; cheaper per cell | as 4, plus any passive mechanism worth a type | the "every cube is a robot" story weakens toward the TERMES fallback; the policy must reason about types |
| 6 | rotating face on the active cube | orientation per cell; a spin move (turn the body bonded to a face about its normal by 90 degrees; continuous in MuJoCo rigs); swept-cylinder check | one face (or one per axis) is an EPM plate on a slewing bearing with a gearmotor and a 90 degree lock; no power across it (a mechanical joint only); guess +40-80 g, +$15-30, one more motor and lock; likely needs the 8-10 cm cube HARDWARE.md already expects | continuous revolutes: shafts, wheels, castors, knobs, pulleys, rotors, cranks, bearings (as the joint), gears if voxel teeth mesh. Fusion revolute and cylindrical: 75% of free joints, 7.0% of bodies; MCB bearing, gear, rotor, wheel, castor, knob: 12.3% (19% with articulations) | mechanism volume next to 12 edge latches, a motor and 6 EPMs; re-latching after a spin needs the detent within the latch capture (3 mm, about 7 degrees at the face edge); cube orientation becomes part of every plan |
| 7 | sliding joint | a one-cell slide while bonded | rack or stepping-magnet faces (EM-Cube: 6 steps a cell, convex transitions 0.24-0.96 with the surface, failed against gravity); conflicts with flat EPM faces and pivot edges | true sliders: Fusion 6.1% of joints, 1.4% of bodies; MCB 0.04%. Options 1 and 2 already give stepped sliders | high, for the smallest share; not recommended |
| 8 | v3 cube | new MuJoCo model, outcome tables, closed loop, all refit | edge pivot + load-rated edge latches + rotating face + latch floor in one module, or a two-cell module (Roombots, M-TRAN); guess 8-10 cm, 400-600 g, months of work | everything above except springs and true sliders | schedule (cube prototype planned for about New Year); a heavier cube carries less in module weights unless latches scale; the hinge cube's bar evidence does not carry over |

### Notes on the options

1. **Group pivot capacity (option 2).** The motor limit is 0.3 N m; one cube's weight moment about
   an edge is m g L = 0.15 x 9.81 x 0.05 = 0.074 N m, so the motor gives 4.1 m g L, or 2.7 at the
   1.5 margin. A row of n cubes turning about the edge at one end has a weight moment of
   n sqrt((n/2)^2 + 1/4) m g L at its worst angle: 0.71 for the mover alone, 2.24 for n = 2, 4.74
   for n = 3. So on a horizontal axis each driving cube lifts its own row of 2 (the mover and one
   more) with margin; n = 3 needs 1.0 N m, which HARDWARE.md tried on single moves and found harm
   rose (0.071) from dragging gripped neighbours. A flap of any width works if every cube on the
   axis drives its row. On a vertical axis (a door) gravity puts no torque on the turn; the limit
   is the latch holding the door as a cantilever while the mover's faces are released, which is
   not modelled: the joint-strength numbers (a 1-deep cantilever holds 4 at margin 1.5) are an upper
   bound only once the faces close again.
1. **The stall rule makes carries safe by construction.** A carry that does not finish is driven
   back (hinge cube "Stall"); a body cube that fails to release from the rest of the structure
   stalls the turn instead of tearing it. The harm to watch is the one HARDWARE.md already names:
   a cube held up only by a moving body drops. The overhang rule extends to bodies directly.
1. **The rotating face is where function and form part most.** A wheel becomes a cross or a disc
   of cubes on one spinning face; a bearing becomes the spinning face itself, with no ring; a gear
   pair becomes two toothed bodies of cubes on two spinning faces, and whether 1-cell square teeth
   mesh without jamming is a MuJoCo question (M5). This is what "the function must be the same,
   even if it doesn't look as is" allows.
1. **Option 4 against option 6.** A passive swivel is cheap and leaves the active cube alone, but
   it has to be carried into place (option 2, with its payload limits), and something still has to
   drive it. One rotating face per active cube costs a motor per cube but drives itself and needs no
   second type. If continuous rotation is a minority goal (M6), option 4 is enough.

## Recommendation

1. **Now, software and MuJoCo only: options 1 and 2.** Define the eight rigs; gate goals with
   `joint_strength` in their function rig, not their resting pose; turn fasteners and dowels into
   joint specs; add the group pivot to the grid (bond state, body turn, checks) and to `HingeSwarm`
   (carry mode). This covers the static families and swing hinges, which by count is most of both
   datasets, without touching the cube.
1. **On the bench, alongside the hinge cube's first test: measure the edge latch's holding force
   under a moment.** If the edge latch can be rated to a few hundred newtons, option 3b comes almost
   free with the hinge cube, and with it doubled cantilevers and stronger hinges. Use the floor
   latches statically (3a) as soon as they exist.
1. **Decide continuous rotation after M4-M6:** rotating face (6) if continuous rotation is a large
   share of the goals the user cares about and a face spin re-latches reliably; passive swivel
   cubes (4) if it is a minority. Do not pursue a sliding joint (7) or springs.
1. Keep v3 (8) as the destination if 6 is chosen: a hinge cube with a rotating face and load-rated
   edge latches is v3 in all but name; better to bench the parts one at a time than design it whole.

## Measurements that would settle it

CPU hours are guesses from the run times in HARDWARE.md (a hinge move is 1.2 s simulated and the
closed loop ran near real time) and from the rig run here.

| id | experiment | settles | rough cost |
| --- | --- | --- | --- |
| M1 | function rigs (span, cantilever) over the whole library, with the option 3 joint variants | which share of goals passes per option; which goals to drop or reinforce (gussets, depth) | one variant per part (about 8,300 parts), both rigs, two joint variants: about 9 h on 4 threads (the sample ran at about 1 s a check) |
| M2 | group pivot in MuJoCo: `HingeSwarm` carry mode, rows of 1-4 cubes, widths 1-3, horizontal and vertical axes, motor 0.3 / 0.6 / 1.0 N m, 20 jittered seeds (about 700 carries) | payload per driving cube; reached and harmful per carry against the bar | 1-2 CPU h, plus 1-2 days of code |
| M3 | latch hold: the hinge latch as a force-limited constraint, bisect the force a door 1-6 wide (vertical axis) and a flap 1-3 deep need during the turn | the latch spec for the bench | 0.5 CPU h |
| M4 | rotating face in MuJoCo: spin a column, a cross and a disc of cubes about horizontal and vertical axes; torque needed; re-latch rate at the 90 degree detent over 20 seeds | whether option 6's motor and lock are feasible at 5 and 8-10 cm | 2-3 CPU h, plus 1-2 days of code |
| M5 | voxel gear pair: two cross or star bodies on spinning faces at centre distances 3-5 cells | transmission, backlash, jam rate of voxel teeth | about 1 CPU h |
| M6 | label the 847 Fusion revolute bodies and the MCB revolute classes as swing or continuous (names, assembly context) | how much of the target set needs option 4 or 6 at all | no GPU; a few dollars of Haiku calls or an hour by hand |

## Caveats

1. Counts inherit `machine-parts.md`'s sampling: Fusion is one archive in eleven, MCB voxels are
   the first 120 parts per class. Family groupings of MCB classes are mine (e.g. keys counted as
   fasteners, thrust washers as bearings).
1. Rig results are from `joint_strength`, which is optimistic by 2-23% for spans on free pillars
   and does not model dynamic loads; a pass at 1.5 is the bar it was validated for, not a guarantee.
1. All hardware numbers are for the modelled 5 cm, 150 g hinge cube. HARDWARE.md expects the first
   build at 8-10 cm; capacities in cube units hold only if the motor and magnets scale with it.
1. The rig loads (1 W at midspan or at the tip) are my choice; the data records no loads.
