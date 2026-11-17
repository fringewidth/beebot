# Layout B module: eleven robots and machines in MuJoCo

Written 2026-11-15. Status: **nothing is built.** This is a simplified simulation of the layout B
module of `hardware/docs/MODULE_SPEC.md` (a 50 mm, 120 g cube with two turning discs on opposite faces,
each with its own drive clipped at 0.8 N m). Every robot below is that one module arranged
differently; objects being moved and fixtures (a post, rails, a door, a knob) are scene parts.
Nothing was committed and nothing was written to `ops`. CPU only, one process at a time, `nice`.

Idealised (the sim leaves out):

1. Bonds are rigid and never fail: a bonded module is a fixed child body of its neighbour. No
   connector, no capture, no backlash, no compliance, no bond strength. Bond loads are measured
   and compared with spec 5.5 afterwards, not enforced.
1. The side faces' connector plates are left out of the collision model (see finding 1), and
   neighbours that touch at build time do not collide with each other.
1. The drive is an ideal torque source behind a 1 kHz PD loop, clipped at 0.8 N m, with a guessed
   gear inertia (1e-3 kg m2) and drag (0.005 N m). No speed-torque curve, no heating, no backlash.
   It can be switched to zero torque (free), which a worm stage (spec 4.4) could not do.
1. Angle sensing is exact; the controllers also use the true position and yaw of bodies (the
   external camera tracking the spec assumes).
1. Mass is a uniform 110 g box plus two 5 g discs. Friction: discs (rim and face) 1.0, body 0.6,
   floor 0.8, no rolling resistance.
1. Bases are bolted to the world or a fixture where noted (crane, arm, crank, screwdriver).

Every number is marked **measured** (from the sim, `runs/module_demos/results.jsonl` unless named),
**derived** (arithmetic on stated inputs) or **guess**.

Regenerate everything (about 6 minutes on the Mac CPU): `.venv/bin/python hardware/scripts/module_demos.py all`.
Builder: `hardware/src/beebot/hardware/module_b.py`. Tests: `hardware/tests/test_module_b.py` (6 pass).

Media: reel `hardware/handoff/media/module-demos.mp4` (2.5 min, eleven captioned clips), contact
sheet `hardware/handoff/media/module-demos-sheet.png`, one clip per demo in `runs/module_demos/`.

## Results (all measured)

All eleven final arrangements pass their stated criterion. "Above 0.4" is the time the busiest
joint spent above the 0.4 N m continuous target; "bond" is the largest bending moment at any
bonded face, averaged over 50 ms so a single contact impulse does not count.

| # | demo | modules | criterion and result | peak joint torque | above 0.4 N m | bond moment |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | lone module rolls and turns on the spot | 1 | 29.7 cm, turned 89.6 degrees, 22.2 cm | < 0.05 N m | none | 0 |
| 2 | cart carries 300 g, turns, stops at a mark | 4 | turned 93 degrees, stopped 1.7 cm from the mark, box stayed on | 0.13 | none | 0.12 |
| 3 | conveyor of rollers | 7 | parcel 17.6 cm out, 11.4 cm back | < 0.05 | none | 0.15 |
| 4 | crane: slew, luff, level platform | 5 | load raised 5.2 cm, slewed 90 degrees, stayed on | 0.20 | none | 0.42 |
| 5 | bridge over 150 mm and a shelf, 200 g each | 14 | both held | < 0.05 | none | 0.73 |
| 6 | screwdriver turns a 0.3 N m knob | 3 | 3.3 turns | 0.71 (head) | 0.04 s of 9.5 s | 5.1 (not trustworthy, see 6 below) |
| 7 | crank and slider (piston) | 7 | 3.4 crank turns, stroke 104.8 mm | 0.80 (crank) | 1.23 s of 9.8 s | 2.8 |
| 8 | lone module pushes a sprung door open | 1 | door to 76 degrees, drove through | 0.09 | none | 0 |
| 9 | quadruped, four rotating legs (RHex style) | 14 | 51.9 cm in four 4 s cycles | 0.80 (back legs) | 1.36 s of 18 s per back leg | 2.1 |
| 10 | inchworm-like crawler, two bend joints | 7 | 19.3 cm in 8 cycles (32 s), 1.2 cm drift | 0.46 | 0.18 s of 32 s | 0.51 |
| 11 | swing-arm pick and place, two-finger jaw | 11 | 40 g block lifted 2.9 cm, carried 17.9 cm, set 2.9 mm from the mark | 0.42 (lift) | 4.05 s of 22 s | 1.13 |

Notes per demo:

1. Lone: the body tips onto an edge (the skid) when it accelerates; its lowest corner reached
   -0.05 mm, i.e. touched. Rolling at 0.15 m/s needs under 0.05 N m.
1. Cart: two wheel modules in a row, two deck modules on top with upright axes. Skid steer turns
   it on the spot with 0.13 N m.
1. Conveyor: five modules with axes across the line stand on two leg modules; the top rims carry a
   70 mm wide parcel.
1. Crane: the bottom module is bolted down by its lower disc and slews the rest; the luff joint
   needs 0.20 N m with 100 g on a 52 mm jib.
1. Structures: the 0.73 N m is the shelf column carrying a three-module shelf with 200 g at its
   tip (0.68 N m derived for the static load). Joint torques are near zero because the loads go
   through bonds and bearings.
1. Screwdriver: the bit is a whole module hung on the head's lower disc; its square body drives
   the two lugs of a slotted knob. Steady drive torque is 0.31-0.36 N m (measured in a probe run):
   the knob's 0.3 N m plus drag. The bond figure comes from a rigid square bit driving one lug
   corner at a time with up to 44 N of mostly radial contact force; a compliant real bit would not
   see that. The knob's dry friction is a smoothed torque applied each step, because MuJoCo's
   joint `frictionloss` went unstable against the bit.
1. Crank: the closed loop closes by contact (the piston module slides between two rails); the
   crank pin and piston pin are discs at zero torque. The crank drive hits 0.8 N m at the dead
   centres when the piston wedges between the rails; that also gives the 2.8 N m bond moment and
   23.8 N bond tension.
1. Door: a 150 g door on a 0.04 N m/rad closer; traction of one module's two rims is enough.
1. Walker: front legs on the front module's discs, back legs on the back module's discs one
   module further out, a four-module body between. Wave gait with a smoothed Buehler clock:
   stance 40 degrees in 3 s, swing over the top in 1 s (peak 5.6 rad/s, 53 rpm, derived). The back
   legs exceed 0.4 N m at touchdown, when the heavier back sags and the swinging foot meets the
   floor early.
1. Crawler: two feet (a module on held rims under a post) joined by a three-module link with a
   bend joint at each end. Run the joint loop backwards and it crawls backwards (measured in a
   probe: -25 cm in 8 cycles). The feet rock on the round edge of their held rims, so the gait
   depends on the foot shape.
1. Arm: base yaw (bolted by its lower disc), a two-module tower, a lift joint that swings the
   seven-module hand like a pendulum, and a jaw: one finger swung by the wrist's disc, the other
   fixed. The finger grips at a 0.3 N m current limit (spec 4.6). The lift joint carries 0.37 N m
   with the empty hand at 24 degrees and 0.42 N m with the block.

## Above the 0.4 N m continuous target

1. Arm lift joint: 0.42 N m for the whole 4 s it carries the block, 4 % over (measured).
1. Crawler joints: 0.46 N m peaks, 15 % over, for 0.18 s per joint (measured).
1. Screwdriver head: 0.71 N m spikes at engagement, 0.04 s (measured); the steady load is under 0.4.
1. Crank drive: reaches 0.8 N m (the holding limit) at the dead centres; above 0.4 for 13 % of
   the run (measured).
1. Walker legs: reach 0.8 N m at touchdown; back legs above 0.4 for 8 % of the run and at the
   limit for 3 % (measured).
1. Everything else stays under 0.21 N m: cart, conveyor, crane, door, structures, lone module.

## Failures

No final demo fails its criterion. These failed on the way and changed the design; each is a
finding about the module, not a bug in the sim:

1. **A load carried on wheel modules falls off the back.** The first cart had the 300 g box on
   the wheel modules' tops; the top rims (1 mm proud) turn backwards under it and conveyed it off
   (measured). Fix: a deck of two more modules with upright axes.
1. **A turning module jams on a diagonal neighbour's rim at a 50 mm pitch.** With a 50.4 mm pitch
   the crane's luff joint stalled at 0.8 N m (measured): the rim, 1 mm proud, reaches 0.6 mm into
   the diagonal cell, and the turning module's rim and edge hit it. Fix in the sim: a 52.4 mm
   pitch.
1. **A lone module with the spec's side plates does not roll.** With plates proud enough to keep
   rims off neighbours (1.2 mm), the module stands on its bottom plate and the discs spin in the
   air: 0.0 m in 2 s (measured). See finding 1.
1. **Rotating legs hit the other legs' hubs.** With a three-module body the front legs swept 82 mm
   (derived) and clipped the back hubs by 2.5 mm, stalling at 0.8 N m (measured). Fix: four-module
   body (14 modules in all).
1. **The pick-and-place hand knocked the block over** while slewing until the block was lowered to
   20 mm and turned to face the jaw (measured): a pendulum lift of 24 degrees raises the fixed
   finger's lowest module only about 9 mm (derived).

Not attempted or not met:

1. **No snake.** A bend costs an offset module, and an offset module cannot turn at floor level
   (finding 3), so a chain lying on the floor has no usable bend joint. The crawler (demo 10) puts
   its two joints on posts instead.
1. **T1 as written is not met.** The arm places 17.9 cm away (spec: 30 cm) on a fixed circle and
   lifts 2.9 cm. A 4-DOF arm reaching 30 cm with the five-module jaw would need about 3 N m at the
   shoulder (derived from spec 4.1 with n = 6 and a 650 g payload), far over 0.8.
1. Not simulated: walking over a step (T5), crawling through a gap (T6), docking (T10), a drawer
   (T3), a peg in a hole (T4).

## What the sim says about the spec

1. **Spec 3.5's open conflict is real and has no clean fix as written.** Rims 1 mm proud reach
   into neighbouring cells unless the pitch is at least 52 mm; at a 50.4 mm pitch a turning module
   jammed on a diagonal neighbour's rim (measured). Plates 1.2 mm proud keep rims clear but then a
   lone module stands on its plate and cannot roll (measured, 0.0 m). With plates exactly 1 mm
   proud (the spec's candidate), plates and rims touch the floor together and two neighbours' rims
   touch each other (derived). The demos only work because the plates are left out of the
   collision model. One of: retractable or recessed plates, rims that are not proud (and rolling
   on something else), or giving up lone rolling, has to be chosen.
1. **A lone module does roll on a 1 mm rim, on a flat floor.** 0.15 m/s at under 0.05 N m, turns
   on the spot, the body edge works as the skid (measured). It touches the floor with its edge
   while accelerating; any bump over about 1 mm will catch the body (derived from the clearance).
1. **Elbows cost an offset module, and offset modules need room.** An offset module turns about
   its own centre, so its corners sweep 35 mm and it needs every neighbouring cell in its plane
   empty (derived), and it cannot turn at floor level, where a module's axis is only 26 mm up
   (derived; the walker's hub touched the floor when the body sagged, measured). This is why the
   walker needs 14 modules, the arm's fixed finger needs three modules to reach the moving
   finger's plane, and there is no snake.
1. **A gripper is expensive.** Beyond the wrist module whose disc drives it, the jaw is five
   modules (600 g, derived): two for the swinging finger, three for the fixed one. It closes in an arc, so wide objects are met at a steep angle
   (about 37 degrees for a 30 mm block, derived) and squeeze out; the demo used a 14 mm block.
1. **0.4 N m continuous is enough for wheels, conveyors, cranes with short jibs, doors and
   structures**, and is exceeded by every jointed load-carrier in the set: arm lift (4 % over),
   crawler (15 % over, briefly), walker and crank (to 0.8 N m at touchdown and dead centre).
   Spec 4.1's arm table (100 g payload) is optimistic for real arms because the jaw alone is
   five modules, 600 g.
1. **Bond and bearing moments exceed spec 5.5 and 4.9's 0.8 N m in ordinary use.** The swing arm
   holds 0.89 N m at rest and 1.12 N m with the hand lifted at its base bonds and bearings
   (measured), 40 % over. The shelf reached 0.73 N m. Dynamic peaks: 2.1 N m at the walker's
   touchdowns, 2.8 N m and 23.8 N of tension (spec: 15 N) when the piston wedges (measured,
   50 ms averages).
1. **Passive joints conflict with a non-backdrivable drive (spec 4.4).** The crank and slider
   needs two pins that turn freely; with a worm stage they would have to be driven in step with
   the crank, or the drive needs a clutch.
1. **Reaction torques need a fixed base.** Every demo that turns something against a load (crane,
   arm, crank, screwdriver) was bolted to the world or a post. A free-standing frame on the floor
   would have to resist the screwdriver's 0.3 N m by friction (derived: about 0.1 N m available
   from a 360 g frame with friction 0.7 at a 50 mm lever). The spec's open tool interface (5.12) or a bench mount carries this.
1. **Speed is not the limit in any demo.** The fastest joint is the walker's swing at 5.6 rad/s
   (53 rpm, derived), inside spec 4.2's 55 rpm.

## Files

1. `hardware/src/beebot/hardware/module_b.py`: `Module`, `lattice` (cells to modules), `build` (MJCF),
   `Rig` (PD loop, current limits, scripted paths, torque and bond-load recording).
1. `hardware/scripts/module_demos.py`: the eleven demos, the reel and the sheet.
1. `hardware/tests/test_module_b.py`: builder validity, bond orientation, torque limits, a lone module
   rolls and turns, plates stop it rolling.
1. `runs/module_demos/results.jsonl`: one line per demo (demo, ok, criterion, modules,
   peak_torque_nm per joint, over_0_4_s, at_limit_s, bond and bearing loads, measured, notes).
