# Rotating cube: CAD of the whole cube and a MuJoCo proof of concept

Written 2026-11-14. Replaces `runs/handoff/rotating-cube-cad.md` (its content is merged here).
Nothing was committed and nothing was written to `ops`. No GPU, no rented machine: everything ran
on the Mac's CPU with 5 worker processes.

Every number is marked **measured** (with the file), **derived** (arithmetic on measured or stated
inputs), **bought** (catalogue size of an off-the-shelf part, from memory unless a source is named)
or **guess**.

## Verdict

1. **It works in simulation.** A cube turns a bonded neighbour and a load cube twice round a
   horizontal axis, stopping and re-locking at every quarter turn: 40 of 40 runs, 320 of 321
   quarter turns re-locked (the other was driven back and repeated), nothing dropped (measured,
   `runs/cube_turning/crank-turning.jsonl`, `crank-turning-seeds1000.jsonl`). A cart on four
   square wheels rolled 10 cells in 38 of 40 runs with no harm (measured, `cart-turning*.jsonl`).
1. **It is not needed for either task.** With the arm clear of the tower, plain hinge cubes
   circle the load round the arm by edge pivots 20 of 20 times, in 15.6 s against 13.8 s
   (measured, `crank2-hinge-free.jsonl`), and in that layout the turning face is the one that
   fails: it drops the arm in 11 of 20 runs (measured, `crank2-turning.jsonl`). The same ten
   cubes cross 10 cells on their own by floor rolls 20 of 20 times, in 27.0 s against the cart's
   12.3 s (measured, `march-hinge.jsonl`). The turning face won one comparison (arm one cube
   long, where the hinge cube's load rubs the tower and stalls, 4 of 20) and bought a factor of
   about 2 in speed on the cart.
1. **What only it can do** is turn a bonded body about a fixed axis without limit: a wheel, a
   shaft, a crank that keeps going. No sequence of edge pivots does that. Whether that is worth
   anything depends on whether goals with parts that keep turning are wanted
   (`runs/handoff/physics-primitive.md`); shape building and moving do not need it.
1. **It does not solve a machining problem; it adds some.** The face needs a bearing seat, 0.2 mm
   running clearances, a detent, a disc that stands 0.3 mm proud while turning and flush
   otherwise (not designed), and four more magnets on every face of every cube. It adds 27 g
   (13 %) and a second gearmotor (derived from the CAD's mass table).
1. **Its bond is weak while it turns.** Only the disc's four magnets hold the neighbour then:
   about 0.37 N m against 1.33 N m for a locked face (derived). Two cubes on the face are safe,
   three are not (measured, `carry-*.jsonl`).
1. **The cube's real problem is elsewhere.** The CAD could not find a way to drive and latch
   the edge hinge, which every result of the hinge cube assumes. An N20 gearmotor is rated
   0.2 N m (Pololu 298:1 HP 6 V: gearbox limit 2 kg cm, stall 4.0 kg cm extrapolated, 1.6 A; read
   from pololu.com/product/994 on 2026-11-14); both motors are assumed to give 0.3 N m. At
   0.2 N m the crank succeeds 17 of 20 and the cart 0 of 20 (measured). The switchable magnets
   are 96 g of a 207 g cube (guess).

My recommendation: do not put a turning face on every cube now. It does not earn 27 g, a motor
and a weaker lattice for the goals in hand. Spend the bench time on the edge hinge and the
magnets. If turning parts become a goal, build the face as a separate cube type and give the
disc a key or stronger magnets first.

## The cube (CAD)

Scripts, all CadQuery 2.8, in `hardware/rotating_cube/`:

1. `cube_full.py`: the complete cube. Exports `cube_full_<part>.step/.stl` for 18 kinds of part and
   `cube_full_assembly.step`, runs the interference checks, prints mass, inertia and cost. About
   4 minutes. Its output is saved in `runs/cube_turning/cad.txt`.
1. `render_exploded.py`: `runs/handoff/media/rotating-cube-cad.png` (outside, and a section) and
   `rotating-cube-exploded.png` (every part pulled out and labelled). About 3 minutes.
1. `cube_turning_face.py`: the first draft of body and disc by the earlier session, kept; its
   sizes for the recess, bearing and detent balls are reused. Its body and disc are superseded.
1. `edge_hinge.py`: recreated from the exported files and the constants left in the bytecode
   (the source was lost). Same five parts, same bounding boxes as the old STL files (measured).
   It illustrates one pivot between two cubes; it is not a mechanism.

The cube is 50 mm, x, y = -25..25, z = 0..50; the turning face is +Z.

| part | what it is | size, mm | status |
| --- | --- | --- | --- |
| body | printed shell, 2 mm walls, 1.5 mm chamfers on all 12 edges, a solid block above z = 33.6 that carries the bearing | 50 cube | guess |
| turning disc | plate 35.6 dia x 9.5 with a 17 mm hub through the bearing, bored for the motor's D shaft; 0.2 mm clear of the recess all round; stands 0.3 mm above the face | recess 36 dia | guess |
| bearing | 6803: 17 x 26 x 5 | | bought |
| retainer | washer screwed under the hub; holds the disc in the bearing | 19.6 dia x 1 | guess |
| turning-face gearmotor | N20, 298:1, in the body on the disc's axis; its shaft drives the hub | 12 x 10 x 24, shaft 3 x 10 | bought |
| detent | 4 steel balls 3 mm in radial pockets in the disc's rim, a 1 N spring behind each, 4 dimples in the recess wall every 90 degrees | balls stand 0.5 mm out | guess |
| edge-hinge gearmotor | N20, along y, low in the body | 12 x 10 x 24 | bought |
| edge axles and latch hooks | on each of the 12 edges a 3 mm steel axle 12 long, 3.5 mm inside both faces, with two hooks | | guess; stand-ins |
| edge magnets | 24 electropermanent units, two an edge at 12.5 mm from its middle, flush with both faces | 10 x 6 x 6 | guess |
| face magnets | 4 units a face on a 12 mm circle at 45 degrees; 20 in the body, 4 in the disc | 10 x 7 x 6 | guess |
| ring board and spring pins | two copper tracks under the disc, two spring pins in the body | 27/35 dia x 0.6; pins 1.5 x 5.6 | guess |
| battery | 1S LiPo 502025, about 200 mAh | 5 x 20 x 25 | bought |
| controller board | board with controller, radio, 18 pulse channels | 21 x 24 x 4.6 | guess |
| wire duct | from the board up to the spring pins | 4 x 3 | guess |

### Choices and why

1. **Power across the bearing: motor in the body, plus a two-track face slip ring.** With the
   motor in the body its wires never cross the bearing and its reaction torque goes into the
   body. The disc still carries magnets that must switch. They are electropermanent, so they
   need a current pulse only when they switch, and they switch only at rest. Two spring pins in
   the body riding on two copper rings under the disc carry that pulse; the contact does not
   have to be clean while turning. Rejected: a battery in the disc (needs a second controller
   and radio there, and a way to charge); a capsule slip ring (about 12 mm dia x 17 long, bought
   size from memory, on the axis where the motor is, and rated about 2 A against pulses of
   several amperes, guess).
1. **Switchable magnets: electropermanent, 48 identical units.** Each is an AlNiCo 5 rod and
   an N42 rod, 4 dia x 4, between two steel pole plates in a coil. Sized from Knaian's Robot
   Pebbles unit (3.2 mm rods, 4.4 N, from memory) by rod area: 7 N a unit, 2 g (guess). A face
   needs 53 N to match the hinge cube: 8 units' worth. Force per gram is roughly fixed (OpenGrab
   EPM v3: 65 g for about 250 N, from memory, gives 14 g for 53 N; this layout gives 16 g), so
   six faces cost about 90 g whatever the layout.
1. **Where the magnets go.** A face pattern must map onto itself under quarter turns (or the
   bond does not re-lock) and under mirroring (or two faces do not mate). Three units of three
   faces do not fit in one corner. So the outer magnets are in the edges, each serving two
   faces, as `cube_swarm`'s edge magnets do and as M-Blocks do. The inner four are needed on
   every face of every cube, because a neighbour's face has to mate with a disc; the disc
   (17.8 mm radius) reaches none of the edge magnets.
1. **The disc stands 0.3 mm proud.** Otherwise the disc's magnets press the neighbour onto the
   ring of the body round the disc and that friction fights the motor: 26.5 N x 0.5 x 25 mm =
   0.33 N m, more than the motor (derived).
1. **Edge hinge: not solved.** HARDWARE.md says the latch and torque source are not designed.
   I tried to lay one out and failed: an axle 3.5 mm from two faces cannot carry a gear larger
   than 3.5 mm radius; a train or belt from one motor to four parallel edges runs through the
   corner zones the edge magnets occupy; the motor itself (10 x 12 mm) cannot sit on the edge.
   The CAD carries the motor, 12 axles and 24 hooks as stand-ins that reserve volume and mass,
   and 10 g for a train that is not drawn. No hook is shown taking hold of a neighbour.

### Interference (measured, `runs/cube_turning/cad.txt`)

475 pair checks of CadQuery intersection volumes over the 18 parts: all pairs with the disc at
0 degrees, disc parts against fixed parts at 22.5, 45, 67.5 and 90 degrees, hooks against
everything at 0, 45 and 90 degrees of swing. No pair shares more than 0.01 mm3. Fits are nominal
(no press or slip allowance). Between detents the balls are drawn pushed in 0.5 mm, as the
springs allow. Found and fixed on the way: hooks on the turning face's edges swung into the
recess (they now rest pointing down); the board met a hook (shortened 2 mm).

### Mass and cost (`runs/cube_turning/cad.txt`)

| part | g | basis |
| --- | --- | --- |
| body | 27.0 | CAD volume x PETG 1.27 x 0.7 fill (guess) |
| disc | 8.6 | CAD volume x 1.27 x 0.8 (guess) |
| 6803 bearing | 7.0 | bought (a solid steel ring of its size is 11.9) |
| two N20 gearmotors | 19.0 | bought, 9.5 each |
| 48 electropermanent units | 96.0 | guess, 2.0 each |
| 12 axles, 24 hooks | 13.9 | steel volume; hooks guess |
| battery | 5.0 | bought |
| board | 7.0 | guess |
| balls, springs, retainer, ring board, pins, duct | 3.1 | guess |
| not drawn: wiring 4, screws and mounts 6, edge drive train 10 | 20.0 | guess |
| **total** | **206.6** | |

1. The turning face's share (disc, bearing, retainer, its motor, balls, springs, ring board,
   pins): 27.2 g, 13 % (derived). The same table without it gives about 179 g for a hinge cube,
   not the 150 g HARDWARE.md models (derived from guesses).
1. Inertia about a centre axis: 9.5 to 9.9 e-5 kg m2, 0.189 x mass x side^2 (a uniform cube:
   0.167); centre of mass 1.3 mm above the middle. Disc assembly: 18.5 g, 2.8 e-6 kg m2 about its
   axis (computed from the CAD, parts taken as uniform; the masses are the guesses above).
1. Cost of one cube in small numbers: about 126 USD, all guesses: 48 magnet units 60, two
   gearmotors with encoders 16, 18 pulse channels 14, controller 8, edge train 10, the rest 18.
   The turning face's share is about 15 (motor 8, bearing 2, ring board and pins 3, small parts 2).

## The MuJoCo model (`src/beebot/hardware/cube_turning.py`)

`TurningSwarm` is a `HingeSwarm`: same time step, contacts, magnet law, steel floor, placement
error, jitter and latched edge pivot. `cube_hinge.py` and `joint_sim.py` are unchanged. Added:

1. **Mass and inertia** from the CAD (207 g). Magnet pulls are kept in newtons (53 N a face), so
   in module weights they are weaker than the hinge cube's.
1. **Turning face**: a disc on a hinge joint about one face's normal; a motor torque on the joint,
   clipped at 0.3 N m (x 0.8 to 1.2 per cube), PD along a minimum-jerk reference at 100 deg/s;
   reflected rotor inertia 0.005 kg m2 (guess); drag 0.02 N m (guess).
1. **Detent**: -0.08 sin(pi phi / 3.6 deg) N m within 3.6 degrees of a quarter turn, zero
   outside (derived from the ball and dimple sizes and 1 N springs; x 0.75 to 1.25 per cube).
1. **Magnets**: `cube_swarm`'s 30 sites keep half of each face's pull; the other half sits on
   four inner sites a face at 12 mm, which on the turning face ride on the disc. Same law,
   range and steel floor. With no inner share the forces equal `SwarmMagnets`' (tested).
1. **Contact**: the disc stands 0.3 mm proud on eight pads. Torque passes to the neighbour by the
   pads' friction and the inner magnets' sideways pull only. No key.
1. **A quarter turn**: the carried cubes switch off toward everything else except the driver's
   disc; the disc turns; within 3 degrees and nearly still the magnets come back on and the edge
   magnets pull the pair square; the motor lets go. A stall is driven back.

A quarter turn counts as re-locked when the disc rests within 2 degrees of the quarter, the
carried cubes sit aligned in the cells the turn should put them in, and nothing else moved.

Three things I had to change on the way, each a finding:

1. **Motor.** At the hinge servo's 150 deg/s the 0.2 N m motor saturated lifting the load,
   overshot by 18 degrees and let go there (measured, first runs). Now 100 deg/s, and arrival
   needs the disc nearly still.
1. **Share of pull on the disc.** At one third the disc bond slips in torsion before the motor
   stalls: a cart's rear wheels fell 40 degrees behind their discs (measured, first runs). At
   one half it holds. In the final runs one third gives crank 19 of 20 with one drop, cart 17 of
   20 with three carts pulled apart (measured, `*-share33.jsonl`).
1. **Proud disc on the floor.** A cube that rolls onto its turning face stands 0.3 mm high; the
   magnets tilt it and its neighbour 0.26 degrees and lift the neighbour's near edge 0.2 mm
   (measured). Latched like that the next pivot jammed. The model now draws the pads in and
   settles 50 ms before any edge pivot, which a fixed proud disc cannot do. The real cube needs
   a disc that is flush when it is not turning.

## The task and the numbers

All runs: 20 jittered seeds (friction 0.3 to 0.7, magnets +-25 %, mass +-5 %, placement
+-0.03 L, motor +-20 %, detent +-25 %), `scripts/cube_turning_runs.sh`, rows in
`runs/cube_turning/*.jsonl`, summary in `runs/cube_turning/run_all.out`. All measured.

### Crank

A tower of three; the top cube is the driver, its turning face looks sideways. An arm cube sits
on that face. A load cube starts on top of the driver. The load pivots onto the arm (edge
hinge), is carried twice round the axis with a stop at every quarter turn (turning face, 8
quarter turns, the load 1 cell out: 0.10 N m of gravity), and pivots back (edge hinge). Success:
every step reached and every cube back in its cell. Two retries after a harmless failure.

The hinge cube's version: the load circles the arm by half turns about the arm's edges, four a
revolution. `cube_swarm`'s rules have no move that passes under an overhang, so "as is" it is
refused. "Rule lifted" (`FreeHinge`) allows those two half turns with the same latch and servo;
they have never been measured in the store or walk runs.

| design | arm | success | harm | re-locked per quarter turn tried | time, s |
| --- | --- | --- | --- | --- | --- |
| **turning face, 207 g, 0.3 N m** | 1 | **20 / 20** | **0** | **160 / 160** | 11.7 |
| the same, seeds 1000 to 1019 | 1 | 20 / 20 | 0 | 160 / 161 | 11.8 |
| hinge cube as is, 150 g | 1 | 0 / 20 | 0 | refused at the first pass under | |
| hinge cube, rule lifted, 150 g | 1 | 4 / 20 | 0 | 56 / 160 half turns | 15.1 |
| hinge cube, rule lifted, 207 g | 1 | 3 / 20 | 0 | 45 / 160 half turns | 15.7 |
| turning face | 2 | 9 / 20 | 11 | 74 / 85 | 13.8 |
| hinge cube, rule lifted, 150 g | 2 | 20 / 20 | 0 | 160 / 160 half turns | 15.6 |

1. Re-lock error over the 160 turns of the first row: mean 0.25 degrees, largest 0.91.
   Edge pivots in those runs: 40 of 40 reached.
1. Arm 1, hinge cube: every failure is the pass under the arm, stalled and driven back with the
   load rubbing flush down the tower's face (50 of 70 such half turns). Harmless. The turning
   face does not rub there because the proud disc holds the arm 0.3 mm off the tower.
1. Arm 2, turning face: all 11 failures are the arm falling off the disc during a quarter turn.
   The disc itself arrived each time (error 0.07 degrees). Load on the disc bond: 0.36 N m
   against about 0.37 (derived: 26.5 N on pads 13.9 mm from the axis).

Sensitivity, turning face, arm 1:

| change | success | harm | re-locked | failures |
| --- | --- | --- | --- | --- |
| motor 0.2 N m (the N20's rating) | 17 / 20 | 0 | 145 / 154 | stalls lifting the load from the side: gravity 0.10 + detent 0.08 + drag 0.02 N m |
| one third of the pull on the disc | 19 / 20 | 1 | 154 / 155 | one arm dropped |
| 150 g cubes | 20 / 20 | 0 | 160 / 160 | |
| no detent balls | 20 / 20 | 0 | 160 / 160 | re-lock error up to 1.36 degrees |

The last row says the edge magnets do the re-locking in this model; the detent adds nothing here
and costs motor torque. With gear backlash, which is not modelled, that may differ.

### What the turning bond carries (`carry-*.jsonl`, 10 seeds, one full turn, no retries)

| cubes along the axis | cubes sideways | success | what happened |
| --- | --- | --- | --- |
| 1 | 0 | 10 / 10 | |
| 1 | 1 | 10 / 10 | |
| 2 | 0 | 9 / 10 | one harmless stall |
| 2 | 1 | 4 / 10 | 6 dropped |
| 1 | 2 | 0 / 10 | 6 did not stand when set down, 4 dropped |
| 3 | 0 | 0 / 10 | 9 did not stand when set down, 1 dropped |

"Did not stand when set down" is before any turn, with every magnet on: the tower with that
overhang at 207 g, not the turning face. I did not separate the two further.

### Cart (second rig)

A chassis of six cubes (2 x 3); the four at its corners have their turning faces outward and a
wheel cube on each; two cubes ride in the middle. The four discs turn a quarter at a time, the
wheels are square, each quarter turn rolls the cart a cell. Open loop, 10 cells. The wheels hang
on the disc bonds the whole way (nothing re-locks). Success: all 10 steps done, the cart in one
piece and level, within half a cell of 10 cells and of its line.

| design | success | harm | quarter-turn steps done | time, s |
| --- | --- | --- | --- | --- |
| **cart, turning faces, 207 g** | **19 / 20** | **0** | 190 / 191 | 12.3 |
| the same, seeds 1000 to 1019 | 19 / 20 | 0 | 200 / 200 | 12.3 |
| hinge cubes walking, 150 g: each column rolls a cell in turn | 20 / 20 | 0 | 2000 / 2000 moves | 27.0 |
| cart, one third of the pull on the disc | 17 / 20 | 3 | 174 / 177 | 12.3 |
| cart, motors 0.2 N m | 0 / 20 | 0 | 106 / 118 | |

1. The two failures of the first rows: one wheel's disc stalled on the first step and turned
   back (harmless); one cart finished 0.73 cells short and 0.61 to the side, turned 7.7 degrees.
1. Drift of the carts that passed: along the track 0.03 to 0.05 cells on average, at most 0.28;
   sideways at most 0.3 cells; heading at most 3.6 degrees. A cart does not end on the lattice.
1. At 0.2 N m the cart stalls (12) or arrives too far off (8).
1. The hinge cubes need no turning face to cross the floor and end exactly on the lattice. The
   cart is 2.2 times faster and carries the two riders without their doing anything.

## Videos (`runs/handoff/media/`)

1. `rotating-cube-full.mp4`, 73 s (`scripts/rotating_cube_video.py all`): the CAD turning, the
   blow-up with labels, reassembly and a section with the disc turning (animated); a cube
   crossing the floor and climbing over a neighbour by edge pivots; a cube pivoting onto another's
   turning face, 8 quarter turns with stops, a full turn without stopping, and release; the
   crank, seed 0, then the failure with the arm two cubes long, seed 1. Everything after the
   CAD scene is MuJoCo physics with the cubes drawn from the CAD's STL meshes.
1. `rotating-cube-crank-best.mp4` (seed 0), `rotating-cube-crank-fail.mp4` (arm 2, seed 1: the
   arm falls off the disc), `rotating-cube-cart.mp4` (seed 0).
1. `rotating-cube-cad.png`, `rotating-cube-exploded.png`.
1. `rotating-cube-motion.mp4` is deleted; the full video replaces it. The `edge-hinge-*` files of
   the earlier session are untouched.

## Tests

`tests/test_cube_turning.py`, 9 tests, 4 s: the magnets equal `SwarmMagnets` with no inner share;
a closed face pulls the same at any share; a quarter turn re-locks; a weak motor turns back; a
cube with a turning face still pivots about an edge; the detent's shape; the baseline's planner.
With the existing hardware tests (`test_cube_hinge`, `test_cube_swarm`, `test_cube`,
`test_hardware`, `test_joint_strength`): 57 passed in 9 s (measured, `runs/cube_turning/tests.txt`).

## What is still a guess or not modelled

1. **The edge hinge mechanism**: latch, drive train, how a hook takes the neighbour's axle, the
   latch floor. Not designed. This is the largest gap in the cube, with or without a turning face.
1. **Motor torque**: 0.3 N m is 1.5 times the N20 gearbox's rating, for both motors.
1. **Magnets**: unit size, 7 N a unit, 2 g, that an off unit pulls nothing (its iron poles would
   still be pulled by a live neighbour), pulse energy and the 18 channels' size on the board.
1. **The disc's two heights**: proud to turn, flush otherwise. Not designed.
1. **The bearing**: one 5 mm wide 6803 takes the arm's bending moment (0.1 to 0.36 N m). Not
   checked; it probably needs a second bearing, and the motor column leaves little room.
1. **Torque by friction only**: holds at half the pull on the disc in these runs, slips at one
   third. A key (pins and dimples) would be surer but pins on a face block sliding moves.
1. **Not in the model**: gear backlash, encoder error, compliance of printed parts, the spring
   pins, wear, heat, battery sag under two stalled motors (1.6 A each, bought, against a 200 mAh
   cell).
1. **Fit**: 0.2 mm clearances and nominal fits on printed parts. Nothing printed, nothing weighed.
1. **The baselines' limits**: the "rule lifted" hinge moves are unmeasured elsewhere; the hinge
   cube runs at its modelled 150 g; the hinge baseline for the cart moves every cube, so it could
   not carry a passive load.
1. **One layout each**: the crank and the cart are single layouts. 40 runs each show the
   mechanism works there, not a rate for general use.
