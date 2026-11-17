# Layout C cubes: drive in pairs, pivot singly, build upward (MuJoCo)

Written 2026-11-15. **Status: nothing is built or designed in CAD.** This is a simplified MuJoCo
model of a new cube ("layout C") and a scripted planner. The edge pivot's drive and latch
mechanism is **not designed**: in the sim it is a torque applied to the cube about the edge, plus
two soft point constraints for the latch. The connectors are **not designed**: a bond is a weld
constraint switched on and off, allowed to form only inside the capture range. Nothing was
committed and nothing was written to `ops`. CPU only, `nice`, one process at a time.

Idealised (the full list is the docstring of `hardware/src/beebot/hardware/module_c.py`):

1. Pivot: torque about the edge, PD on a minimum-jerk path at 150 deg/s, clipped at 0.1 N m
   (cube_hinge's model; kd lowered from 0.05 to 0.02, see below). Latch: engages only if the two
   edges are within 3 mm. On a flat floor there is no latch: the edge holds by friction, and the
   torque reacts into the ground.
1. While a cube pivots it does not collide with the cube it is latched to, nor with cubes touching
   it only by an edge or corner (cube_hinge's chamfered-edge idealisation). Its wheel collides again
   over the last 15 degrees; its box collides again on arrival, or after the landing bonds form.
1. Bonds: weld constraints with unlimited strength, formed only if the faces are within +/- 3 mm
   lateral, +/- 5 degrees (spec 5.4) and a gap of at most 3 mm (guess; the spec gives none), checked
   on the true sim state at the moment of the attempt. No pull-in force model beyond the weld.
1. Drives: ideal torque sources behind PD loops at 1 kHz: wheel 0.03 N m (guess), disc 0.8 N m,
   pivot 0.1 N m. Wheels and discs hold when stopped. No speed-torque curve, no backlash.
1. Sensing: camera poses with 1 mm and 1 degree Gaussian noise per frame, IMU lean with 1 degree
   noise (guesses). Placement error for arranged starts: +/- 0.75 mm, +/- 1.5 degrees (guess).
1. Mass 120 g: 100 g box, 6 g disc, 4 g carriage, 10 g wheel (split guessed). Friction: floor 0.5,
   box 0.4, disc 0.8, wheel 1.0 (guesses). Box edges are sharp.
1. The planner is hand-written for these goals, moves one cube or pair at a time, and has no path
   planner beyond a fixed detour around the structure it docks to.

Every number is marked **measured** (from the sim, `hardware/runs/module_pairs/results.jsonl`),
**derived** (arithmetic on stated inputs) or **guess**.

Regenerate everything (about 25 minutes on the Mac CPU): `.venv/bin/python hardware/scripts/module_c_demos.py all`.
Builder and planner: `hardware/src/beebot/hardware/module_c.py`. Tests: `hardware/tests/test_module_c.py` (7 pass).
Media: reel `hardware/handoff/media/module-pairs.mp4` (2 min, nine captioned clips), contact sheet
`hardware/handoff/media/module-pairs-sheet.png`, one clip per demo in `hardware/runs/module_pairs/`.

## The cube as modelled

1. 50 mm, 120 g, tiles at a 50 mm pitch. Nothing stands proud except the wheel.
1. Disc: 42 mm, flush in the +x face, hinge about the face normal, 0.8 N m clip (unused in these
   demos except as a bond face).
1. Wheel: 20 mm, 6 mm wide (guess), in the **-x face, opposite the disc**. Why: the two largest
   intrusions (disc bearing, wheel well) then sit at opposite ends of one axis and leave the
   middle for motors, battery and board (the packing check below); and a cube driving on its wheel
   has its disc on top, free to carry or bond to whatever is placed on it. The wheel is on a slide
   joint with a spring: 1.5 mm out and 2 N at the stop, 3.5 N when pushed flush (guess: 1 N/mm).
   A lone cube (1.18 N) does not push it in, so it rests on the wheel and one bottom edge, leaning
   3.4 degrees (derived: atan(1.5 / 25); measured 3.4-3.5).
1. Edge pivot as above. Gravity torque about an edge is at most m g L / sqrt(2) = 0.042 N m
   (derived), so the 0.1 N m clip is a margin of 2.4 (derived).

## Results

10 seeds per demo; placement, headings and sensor noise differ per seed. "ok" means success rate
>= 0.8. All numbers measured unless marked.

| # | demo | success | key numbers (mean, range) |
| --- | --- | --- | --- |
| 1 | lone cube: 30 cm straight, heading turned 90 degrees, 20 cm on | 10/10 | sideways drift 0.9 mm (0.5-1.9) over 30 cm; 30 cm at 0.099 m/s average including start and fine stop (cruise 0.15); heading turn 89.8 degrees (89.5-90.2) by 3 floor pivots in 2.85 s (0.95 s a pivot); pivot peak 0.047 N m |
| 2 | pairing: two loose cubes at random headings dock side by side | 10/10 | 20.7 s (13.6-34.2); 3.4 pivots (1-9); 1 failed dock in 11 attempts; pivot peak 0.092 N m (0.076-0.1), at the clip for up to 0.03 s |
| 3 | pair travel: 1 m with a quarter turn, stop at a mark | 10/10 | 0.098 m/s (0.096-0.101) over the course; stop error 2.4 mm (1.7-3.6) |
| 3b | same metre by single-cube floor pivots | 10/10 | 0.053 m/s (19.0 s, 20 pivots); stop error 7.9 mm open loop; pivot peak 0.048 N m |
| 4 | split: pair stops, unbonds, cubes go separate ways | 10/10 | 25 cm apart (21-29); 3 pivots for the cube that turns away |
| 5 | reorient: a cube on a non-wheel face pivots wheel down, then pairs | 10/10 | 1.2 pivots to wheel down (1-2: one from a side face, two from wheel up); 0 failed docks in 10 |
| 6 | climb: end cube of a bonded row of three climbs onto its neighbour and back down | 10/10 | lands 49.91 mm above (target 50); pivot peak 0.073 N m (0.069-0.080), never at the clip; 0 failed bonds in 20 |
| 7 | assembly: 6 loose cubes build a two-layer staircase | 10/10 on the tuning seeds; **9/10 on held-out seeds** | 108 s (100-121); 20 pivots (11-26); 5.6 cube-metres on wheels (4.9-6.5); 0 failed docks in 100 bond attempts; 0 harmful events; every cube within 0.19 mm and 0.21 degrees of its cell |
| F | two pairs docking front to back (kept as a failure) | **0/10** | 30 of 30 attempts out of capture: 3.1-3.6 mm out of line when both lean the same way, 7 degrees when they lean opposite ways |

Torques: the pivot never needed more than the 0.1 N m clip to finish a move, but it touched the
clip in the floor moves that drag two cubes (the vertical-edge quarter turn in pairing, up to 0.03 s
per pairing, 0.06 s per assembly) and when landing a cube wheel face down on a neighbour, where the
spring needs up to 3.5 N x 25 mm = 0.09 N m (derived). Climbs (0.069-0.080) and floor rolls
(0.047) stay well under it. Wheel motors hit their 0.03 N m clip on every hard start or stop
(traction slips at about 0.012 N m, derived).

Bond loads (weld constraint forces, sampled every 10 ms): 1.3-5.6 N in demos 2-6; in the assembly
up to 57.9 N and 0.94 N m (seed maxima, mean 20 N). These spikes were not traced to an event; the
likely sources are hard stops of the six-cube row (the lean pulses) and weld pull-in. They exceed
spec 5.5's 15 N tension, so they need checking before they are trusted either way.

## Speed: pairs on wheels vs single-cube pivots

| | pair on wheels | one cube by pivots |
| --- | --- | --- |
| 1 m course with a turn and a stop (measured) | 10.2 s, 0.098 m/s, stop error 2.4 mm | 19.0 s, 0.053 m/s, stop error 7.9 mm |
| cruise (derived) | 0.15 m/s (15 rad/s on a 10 mm wheel) | 0.053 m/s (one cell a 0.95 s pivot) |
| ratio | 1.9x on this course (measured), 2.8x at cruise (derived) | |

But a pair has to be made first: pairing two loose cubes took 20.7 s (measured). Wheels save
about 9-12 s per metre over pivots (derived from the rows above), so pairing pays for itself only
on trips of about 2 m or more (derived). For the 0.4-0.5 m trips of the assembly, single-cube
pivots would have been about as fast as pairing then driving. The pivot cube here is faster than
the hinge cube's 10 cells in 27.0 s in hardware/docs/HARDWARE.md because there are no magnets to switch and the
settle after a move is shorter (0.95 s against 2.7 s per pivot).

## Assembly

Plan (scripted for this goal, `hardware/scripts/module_c_demos.py`, function `assembly`): six cubes start
in three areas 0.40-0.45 m from the site, two per area, each lying on a random face at a random
heading. Each cube not on its wheel pivots onto it; each two pair up; pair 1 drives to the site
with its axles along the future row; pairs 2 and 3 slide in at the end of the row, axle to axle;
then the end cube climbs onto its neighbour (half turn), rolls two cells along the top (two
quarter turns), and the next cube climbs after it. Result: a floor row of four and two cubes on
top over the last two, all six planned bonds closed.

1. Tuning seeds 0-9: 10/10 (measured). These are the seeds the planner was debugged on (it built
   6/10 before the last planner fixes), so this number is optimistic.
1. Held-out seeds 10-19: 9/10 (measured, `logs/module_c_heldout.out`, not in results.jsonl). The
   failure: in the first pairing the sliding cube stopped 8.6 mm off along its line; the retry's
   yaw trim then read a nonsense 88 degree error and could not latch (edges 8 mm apart), twice,
   and the pairing gave up. A planner defect in a retry path, not physics.
1. Time 108 s per build (measured), one actor at a time: pairing about 60 s, docking the row
   about 40 s, the four climbing moves about 6 s. Pairings in parallel would cut it to about 70 s
   (derived).
1. 20 pivots (11-26) against 5.6 cube-metres on wheels (measured): about 3.6 pivots per cube-metre.
   Most pivots are spent pairing (heading turns, the vertical quarter turn, yaw trims), not climbing.
1. 0 failed docks in 100 bond attempts and 0 harmful events in the 10 tuning seeds (measured).
   Harm was checked after every step: a cube tipped more than 10 degrees off its faces, or a
   bystander moved more than 10 mm.
1. A 2x2x2 block was not built. Its floor layer needs two pairs to dock front to back, which fails
   the capture range every time (demo F). The staircase is built only from bonds whose faces are
   normal to the wheel axles.

## Packing check (arithmetic boxes, `module_c.packing`, `hardware/runs/module_pairs/packing.json`)

Inside a 50 mm cube with 2 mm walls (46 mm cube of space, 97.3 cm3), all parts fit with no overlap
(derived). Sizes marked bought are catalogue sizes from memory, not checked against a datasheet.

| part | box (mm, x along disc-to-wheel axis) | source |
| --- | --- | --- |
| disc 42 x 3, in the +x wall | 3 x 42 x 42 | guess |
| bearing 6803, 26 OD x 5 | 5 x 26 x 26 | bought |
| disc gearmotor N20, 12 x 10 x 24, beside the disc axis with a spur gear to the hub | 24 x 12 x 10 | bought |
| wheel 20 x 6, in the -x wall | 20 x 6 x 20 | guess |
| wheel carriage and spring behind it | 3 x 12 x 20 | guess |
| wheel gearmotor N20, axis along the axle, rides on the carriage (+1.5 mm travel) | 11.5 x 24 x 12 | bought |
| pivot gearmotor N20 | 12 x 24 x 10 | bought |
| 1S LiPo 20 x 25 x 5 | 25 x 20 x 5 | bought |
| ESP32-C3 SuperMini 22.5 x 18, 4 mm with parts | 22.5 x 4 x 18 | bought (thickness guess) |

1. Fits: yes, as boxes. Parts take 21.2 cm3; 76.1 cm3 is free (derived).
1. Free within 4 mm of each face, for connectors: +x (disc) 4.7 cm3, -x (wheel) 6.8, the four side
   faces 6.5-7.9 cm3 each (derived).
1. The first layout failed: the disc motor on the disc axis collides with the wheel well (bearing 5
   + motor 24 + carriage 3 + wheel 20 = 52 mm on one line, against 46 mm inside). The disc motor
   has to sit beside the axis and drive the hub through a gear (derived).
1. Not counted: wires, gear trains, the wheel's slide guide, six connectors (spec 5.11), and the
   pivot's latch and drive train to 12 edges, which is the undesigned part and probably the
   largest.

## Failures (kept, with reasons)

1. **Pairs cannot dock front to back (demo F, 0/10).** A cube on its wheel leans 3.4 degrees on an
   edge. Two faces normal to the lean direction are 2 x 1.5 = 3.0 mm out of line when both lean the
   same way (derived; measured 3.1-3.6 mm with stop errors), or 7 degrees apart when they lean
   opposite ways (measured 6.8-7.1). Spec 5.4's capture is +/- 3 mm and +/- 5 degrees. Pressing
   harder makes them lean apart. Faces normal to the axle are not affected (the lean turns them in
   their own plane), which is why pairing and the staircase use only those.
1. **A lone cube cannot steer**, so two loose cubes cannot simply drive together. Pairing takes a
   sequence: one cube drives to where the other's line crosses its own, the other drives into its
   side face and pushes (this squares its yaw), the first lines up, the second latches a vertical
   edge and turns a quarter into the diagonal cell, then slides in alongside. Average 21 s and 3.4
   pivots (measured).
1. **The quarter turn on the floor drags both cubes.** Both wheels and edges scrub, so the partner
   turns too (about 43 degrees in a probe, measured) and the pivot sits at its clip briefly. After
   release the cubes settle on wheel and edge and the relative heading moved by up to 7-14
   degrees in probes (measured), which broke docking until a yaw trim was added while the corner
   edges still coincide.
1. **Sliding in beside a cube is a 3 mm window.** A cube or pair cannot move sideways, so it must
   approach on a line 0-3 mm outside its cell: closer and its corner catches the neighbour's corner
   and shoves it (one run pushed a row's end cube 30 mm and stretched its bond, measured); farther
   and the gap is out of capture (3.0-3.15 mm, measured). The quarter turn is commanded 1.5 degrees
   long so the line points slightly outward.
1. **Leans must match before a bond.** Bonding fails at 7 degrees if two cubes lean opposite ways
   about parallel axles. The planner sets each lean with an open-loop pulse (5 cm/s for 60 ms
   back, then forward, hard stops); 2 cm/s pulses set a lean but cannot flip one (measured).
1. **Sim-model problems found and fixed on the way** (each is a modelling choice, listed in the
   idealisations): the hinge axis must follow the support's leaning edge, not the world
   horizontal (a 6-8 degree landing error otherwise); restoring box contacts at landing shoved a
   landed cube off (from 89 to 63 degrees against the full clip) and spun a whole row 36 degrees;
   a landing wheel face started 1.5 mm inside its support; cube_hinge's kd of 0.05 was unstable
   for two light cubes turning in yaw (bang-bang at the clip).
1. Held-out assembly seed 16 (above). Pairing alone was also 10/10 on its own held-out seeds in
   the demo and 20/20 in probes on other seeds.

## What this says about the design and the spec

1. **Building upward from a flat floor works with an edge pivot.** Climbs and rolls on top were
   10/10 and 40/40 within the assembly, under 0.08 N m for climbs, landing within 0.2 mm (measured,
   idealised pivot). Layout B could not do this at all.
1. **The wheel's 1.5 mm lean is the main design problem.** It breaks front-to-back docking (so no
   2D footprint on the floor, no 2x2x2), makes leans a thing the planner must manage before every
   bond, and leaves a pair resting on a skid edge. Options: a wheel that retracts on command
   (an actuator, so a lone cube sits flat when it is not driving), less protrusion (1.0 mm gives a
   2.0 mm step, derived), or a capture range of at least +/- 4 mm and +/- 8 degrees (derived).
1. **Wheels are faster but not by much on short trips.** 1.9x over a 1 m course, 2.8x at cruise,
   but a pairing costs about 21 s, so pivots win below about 2 m per pairing (derived).
1. **Pivot torque 0.1 N m is enough with a 2.4x margin over gravity**, except that floor moves
   that drag a wheel and a landing that compresses a wheel spring reach it. Spec 4.1's 0.4 N m
   joint target is for the disc, not the pivot; the pivot needs about 0.1 N m (measured peaks
   0.047-0.1).
1. **Contradictions with the spec:** spec 5.4's capture range is exactly the 3.0 mm lean step, so
   front-to-back docking sits on the limit and fails with any error; spec 3.3's chamfers are
   assumed in three places (edge contacts during pivots, landing, sliding past a corner) and the
   sim failed without them, so they are a requirement, not a nicety; the spec's recommendation of
   layout B (section 2) assumed a lone module can steer, which layout C gives up; spec 5.5's 15 N
   tension may be exceeded by dynamic events (58 N weld force peaks, untraced); the spec has no
   capture gap number, and the slide-in window depends on it.
1. **Reliability (spec 9) is not shown.** 10 seeds per demo cannot show 0.997 per dock; what the
   runs show is that the failures that remain are planner and geometry failures (lean, sideways
   approach window), not the pivot. The assembly passed 19 of 20 seeds overall, 9 of 10 held out.

## Files

1. `hardware/src/beebot/hardware/module_c.py`: `World` (builder, drives, pivots, bonds, sensing, recording),
   `Pivot`, `Planner` (lone driving, lean pulses, pair driving and docking, pairing, climbs,
   rolls on top), `packing`.
1. `hardware/scripts/module_c_demos.py`: the demos, reel and sheet; `--seeds` reruns chosen seeds without
   recording.
1. `hardware/tests/test_module_c.py`: build and mass, orientation, lone cube lean and straight drive, capture
   range, latch catch, roll and climb up and down, packing.
1. `hardware/runs/module_pairs/results.jsonl` (one line per demo with per-seed rows), `packing.json`, clips
   and stills; logs in `logs/module_c_*.out`.
