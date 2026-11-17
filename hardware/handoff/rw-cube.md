# Reaction-wheel cube (RW-P) in physics simulation: which moves work, and how reliably (2026-11-15)

Status: simulated only. Nothing built, bought or committed, and nothing was written to `ops`. Model:
`hardware/src/beebot/hardware/rw_cube.py`. Runner: `hardware/scripts/module_phys/rw_sim.py` (calibrate, mc, opt, outcomes,
render). Chains: `hardware/scripts/module_phys/rw_sim_chain.sh` (the Monte Carlo below) and
`hardware/scripts/module_phys/rw_sim_after.sh` (NSGA-II and films). Tests: `hardware/tests/test_rw_cube.py` (10 tests, about 8 s).
Outputs are in `hardware/runs/module_phys/rw_sim/`:
1. `results.jsonl`: one line per move, design and policy, with `worst_move_success`.
1. `trials/<design>__<policy>/<move>.jsonl`: every trial.
1. `calibrate.jsonl`: the k x catch-duration grid.
1. `outcomes.json`: the outcome model.
1. `opt_evals.jsonl` and `pareto.json`: the NSGA-II run.

Films are `hardware/handoff/media/rw-cube.mp4` and `hardware/handoff/media/rw-cube-sheet.png`: bare floor, single try, and
for each of 11 moves the best, a typical and the most common failure seed, slowed 25x and replayed from the
recorded Monte Carlo seeds. Per-clip files are in `hardware/runs/module_phys/rw_sim/clips/`. The per-step force kernels
use numba (0.68.1, in the venv). Like magpylib, ngsolve, pymoo and cma, it is not yet in `pyproject.toml`. The mmsim change is a
momentum-conserving dipole coupling, `DipoleCoupling` in `hardware/src/beebot/hardware/mmsim/mjcoupling.py`. It is
exact for spheres and matches magpylib to 0.03 % (force).

## Answer in one paragraph

On a bare floor the 7 mm cube reliably does nothing that involves a neighbour. A lone floor roll succeeds
0.67 single-try (Wilson 95 % [0.60, 0.73], n = 200) and 0.78 on floors with friction of at least 0.3. A dock
(a roll into a neighbour's face) succeeds 0.42, passive bond capture 0.90, and a yaw trim to 3 degrees 0.87
with corrections. Climbs, descents and splits sit at 0-0.15 single-try (0-0.22 with retries), and 56-100 % of
those tries knock a neighbour out of its cell. No move reaches 0.99, so no move needed the 2,000-trial
confirmation. The 0.997 dock and 0.999 joint targets are out of reach for this design by more than an order of
magnitude in failure rate.

The coordinator's guess is confirmed, and the data splits it into two separate mechanisms:
1. The reaction kicks the support. Removing it with a steel-sheet floor that holds every cube down cuts
   harm from 0.72 to 0.04 (climb onto a row).
1. Even then the impulse cannot hit the energy window. Steel-floor climbs reach 0.27 single-try (0.31 with
   escalating retries) and splits stay at 0-0.04.

A quasi-static pivot is the way out only together with a bond that can be released. With the permanent
sphere bonds, a slow pivot must out-torque a 1.0 mN m peel moment at 7 mm, about 34 times what tips the
support cube over. The pivot would roll the neighbour instead of peeling the bond.

## What was simulated

The design point is concepts.md pick 1:
1. L = 7 mm, 0.855 g (concepts.md: 0.85 g).
1. Two flywheels: radius 2.1 mm, 0.75 mm thick, tungsten rim, I = 3.87e-10 kg m^2.
1. 1.0 mm N52 corner spheres with a 0.1 mm cover and 20 um pocket clearance.
1. Three CeraCharge cells.

Each trial:
1. Builds the scene in MuJoCo and lets it settle (magnets close a 0.12 mm placement gap).
1. Gives the rotor its commanded speed, with the sampled speed error.
1. Catches it with the pawl after the sampled delay.
1. Runs until everything is at rest, then classifies the outcome.

The commanded impulse is H = k sqrt(2 I E / eta). E is the controller's nominal energy barrier along the arc:
gravity plus free-aligned sphere-pair energy, with nominal magnets, not the trial's sampled ones. k is
calibrated per move in simulation (`calibrate.jsonl`).

The Monte Carlo variables, all uniform:
1. Magnet derate 0.70-0.90 per lot (calibrated 0.80), and per-magnet moment ±10 %.
1. Cylinder derate 0.60-0.85.
1. Sphere pocket position ±0.05 mm per axis, pocket clearance ±0.025 mm, cube size ±0.05 mm.
1. Pocket friction 0.1-0.4.
1. Flywheel speed error ±3 %.
1. Catch delay 0.05-1 ms, catch duration 0.5-2x nominal, pawl damping ±0.3.
1. Floor friction 0.2-0.6 and cube-cube friction 0.2-0.6.
1. Capillary adhesion 0-0.09 mN per corner foot, over a rupture range of 0.02-0.2 mm.
1. Contact damping ratio 0.2-1.0 (restitution).
1. Placement ±0.02 mm and ±0.5 degrees.

Time step: 20 us, 1-33 us through the catch, 10 us for falls. The convergence check in mmsim requires
5e-5 or less at 5 mm.

Moves (concepts.md section 7):
1. ROLL on the floor (`roll_floor`, from all 16 orientations with a horizontal flywheel, so every resting face).
1. ROLL on top of a 3-cube row (`roll_top`).
1. CLIMB onto a lone cube or a 2-cube row (`climb_lone`, `climb_row`).
1. DESCEND off them (`descend_*`).
1. SPLIT from a pair or a row (`split_pair`, `split_row`).
1. DOCK: a roll that ends against a neighbour.
1. YAW to 15 degrees with a 3-degree tolerance.
1. BOND: passive capture from a 0-0.25 L gap, ±0.15 L lateral offset and ±8 degrees.
1. FALL_PAIR: a bonded pair dropped 20 cm in random orientation.

Harm follows the project's convention (`cube.DISTURBED`): a neighbour that moved more than 0.1 L or turned more
than 10 degrees. The trials record the displacements, so you can re-threshold. Typical knocks are 1-3 mm and
lone neighbours are often tipped 90 degrees, so this is not a threshold effect.

## Move table, 7 mm (n = 200 per row, Wilson 95 % intervals)

The two policies:
1. "Single": one try at the calibrated k.
1. "Retry": first try at 0.893 k; after stayed, fell back or held, it re-tries at 1.06x the previous impulse,
   at most 5 re-tries. Retry runs in the same trial, with the same magnets and friction, so the correlation
   between tries is real.

Harm is the rate at the end of the policy. Bare floor uses a 1 ms pawl; steel floor uses a 50 us pawl.

| move | bare single | bare retry | bare harm (single / retry) | steel single | steel retry | steel harm (single / retry) |
| --- | --- | --- | --- | --- | --- | --- |
| roll_floor | 0.670 [0.602, 0.731] | 0.660 [0.592, 0.722] | 0 / 0 | 0.630 [0.561, 0.694] | 0.620 [0.551, 0.684] | 0 / 0 |
| dock | 0.415 [0.349, 0.484] | 0.310 [0.250, 0.377] | 0.075 / 0.05 | 0.630 [0.561, 0.694] | 0.645 [0.577, 0.708] | 0.015 / 0.025 |
| roll_top (3-row) | 0.145 [0.103, 0.200] | 0.080 [0.050, 0.126] | 0.645 / 0.835 | 0.625 [0.556, 0.689] | 0.765 [0.702, 0.818] | 0.015 / 0.015 |
| climb_row | 0.130 [0.090, 0.184] | 0.150 [0.107, 0.206] | 0.72 / 0.695 | 0.265 [0.209, 0.330] | 0.305 [0.245, 0.372] | 0.04 / 0.17 |
| climb_lone | 0.000 [0, 0.019] | 0.000 [0, 0.019] | 0.995 / 0.995 | 0.155 [0.111, 0.212] | 0.135 [0.094, 0.189] | 0.60 / 0.735 |
| descend_row | 0.115 [0.078, 0.167] | 0.220 [0.168, 0.282] | 0.84 / 0.75 | 0.450 [0.383, 0.519] | 0.665 [0.597, 0.727] | 0.015 / 0.025 |
| descend_lone | 0.005 [0.001, 0.028] | 0.005 [0.001, 0.028] | 0.995 / 0.995 | 0.660 [0.592, 0.722] | 0.670 [0.602, 0.731] | 0.055 / 0.145 |
| split_pair | 0.000 [0, 0.019] | 0.000 [0, 0.019] | 0.925 / 0.995 | 0.000 [0, 0.019] | 0.005 [0.001, 0.028] | 0.11 / 0.31 |
| split_row | 0.000 [0, 0.019] | 0.000 [0, 0.019] | 0.63 / 0.96 | 0.005 [0.001, 0.028] | 0.040 [0.020, 0.077] | 0.015 / 0.135 |
| yaw to 3 deg | 0.365 [0.301, 0.434] | 0.870 [0.816, 0.910] (corrections) | 0 | 0.000 [0, 0.019] | 0.065 [0.038, 0.108] | 0 |
| bond (passive) | 0.900 [0.851, 0.934] | - | 0 | 0.110 [0.074, 0.161] | - | 0 |
| fall_pair, bond held | 0.560 [0.491, 0.627] | - | - | 0.590 [0.521, 0.656] | - | - |

The single-policy first-try class distributions are in `outcomes.json` and in `mc_log.txt`. Some examples:
1. Bare roll_floor: reached 134, off_lattice 46, stayed 13, strayed 7.
1. Bare climb_row: dislodged 144, reached 26, fell_back 19, overshot 6, off_lattice 4, no rotor along the
   axis 1.
1. Steel climb_row with retries: the first try is mostly fell_back (118 of 200). The end state is mostly
   off_lattice (72) and reached (61). Escalating from too little impulse jumps straight to too much.

Self-right from every resting face, from the bare roll_floor per-face rates (single): +x 0.77, -x 0.58,
+y 0.61, -y 0.61, +z 0.73, -z 0.67. The spread comes from the centre-of-mass offset: rotors at +x/+y, cells
at -z. Getting the charge face (-z) down from +z takes two rolls, so about 0.45 single-try per attempt. A
failed roll usually leaves the cube off the lattice rather than on a wrong face, so a self-right with retries
still converges if the policy can see the cube's attitude. This is not simulated as a sequence.

## The best roll_floor and dock rates, and what limits them

1. **roll_floor:** 0.67 single-try overall; 0.78 [0.71, 0.84] on floors with friction 0.3-0.6
   (0.76 / 0.80 / 0.79 by friction band); 0.22 below 0.3. The best grid cell in calibration was 0.80
   (k = 1.1, 1-3 ms catch, n = 20). Retries do not lift it (0.66), because the main failure is off_lattice,
   not stayed. Three things limit it:
   1. **Edge slip on low-friction floors.** The catch drives the pivot edge backward at about
      (1 - mu) omega L / 2 and it slides v^2 / (2 mu g). At mu 0.25 that is 2-3.5 mm (0.3-0.5 L), so the cube
      lands a third of a cell back. scaling.py's impulse efficiency counted only the energy lost (13 % at
      mu 0.3), not the displacement.
   1. **The cube becomes a projectile above about 1.6x the tip energy.** At the start of the swing gravity
      supplies g cos 45 degrees toward the edge. Above omega^2 r = 0.71 g the cube leaves the floor and lands
      bouncing (off_lattice, double_roll). Housner's [1, 16] energy window in concepts.md assumes the block
      stays on its edge. A cube does not, so the real window is about [1.0, 1.6] in energy and [1.0, 1.26] in
      impulse. With a 6.5 % spread of the impulse needed (efficiency with friction 0.2-0.6) and a 3 % speed
      error, that window is too narrow to hold at 0.99.
   1. **Landing restitution.** The contact damping ratio is a Monte Carlo variable, and bouncy landings slide
      off-lattice. A 0.5 mm median and 1.2 mm 90th-percentile placement error remains even on reached rolls.
1. **dock:** best 0.645 [0.577, 0.708] (steel floor, retries) and 0.42 on the bare floor; 0.80 in one
   calibration cell (n = 20). It has the roll's two limits plus pull-in: the neighbour's spheres accelerate
   the arriving cube, which overshoots or knocks the neighbour (harm 0.05-0.08 bare). On the bare floor the
   dock rate falls with friction (0.63 at 0.2-0.3 down to 0.20 at 0.5-0.6). Higher friction stops the
   arriving cube short of capture (stayed 62 of 200).
1. **Passive bond capture** (no roll) closes 0.90 from up to 0.25 L gap, ±0.15 L lateral and ±8 degrees. The
   misses are the largest gaps and offsets.
1. **What would raise them** (all bench questions):
   1. Grippy corner feet (friction above 0.6) remove the edge slip.
   1. A closed-loop impulse (measure the response, as the yaw trim does) or a quasi-static pivot removes the
      hop window.
   1. Damped corner feet (low restitution) reduce bounce-off.

## Are ballistic climbs and splits fundamentally unreliable? Yes, for two independent reasons

1. **The reaction kicks the support (momentum, not a modelling detail).**
   1. The mover leaves the catch with linear momentum m omega L / 2 (about 1.3e-4 N s at 7 mm).
   1. Through the hinge spheres the support takes the equal and opposite kick. A 2-3 cube support of 0.85 g
      each slides v^2 / (2 mu g), about 0.3-1 mm, against a 0.7 mm (0.1 L) limit.
   1. The face bond is 40 cube weights (338 mN at 7 mm), so any peel drags the neighbour along before the
      bond lets go. Splits pull the partner 1-6 mm.
   1. In descents the swinging mover pulls a lone support cube over (rotated 90 degrees in most trials).
   1. Evidence: on the bare floor, harm is 0.56-1.0 for every move that peels a bond (roll on top, climbs,
      descents, splits); docking and passive capture, which only close bonds, sit at 0-0.08. A steel floor,
      which holds supports by their own spheres, cuts it to 0.015-0.17 for rows, 0.11-0.31 for split_pair, and
      0.06-0.74 for lone supports.
   1. A slower catch (a 20 ms friction brake) removes the kick but cannot deliver the energy: rolls, climbs
      and splits all end "stayed" or "held".
1. **The impulse cannot be dosed to the window.**
   1. A climb or split must carry the bond-peel energy: barrier 150-220 uJ, against 12 uJ for a floor roll.
   1. The peel energy varies about ±20 % (derate 0.70-0.90 and per-magnet ±10 %), so ±30-45 uJ.
   1. Once the bond lets go, all excess energy becomes motion. The landing tolerates roughly 0.6x the gravity
      barrier of excess, about 7 uJ for a quarter turn.
   1. The spread is 4-6x wider than the window. Open-loop single tries therefore split into "held / fell
      back" and "off_lattice / overshot".
   1. A 6 % impulse step (12 % in energy, 18-26 uJ) jumps over the window, so escalating retries rarely land
      in it. Evidence: on the steel floor, steel climb_row goes from fell_back 118 to off_lattice 72 under
      retries. Splits go straight from held to off_lattice in every catch duration tried (50 us, 1 ms, 3 ms)
      and at every k from 1.0 to 2.0 (calibration, both floors).
1. Lone-cube climbs and descents are the worst cases: the support has the mover's own mass.

Nothing in the model is known to make these moves harder than a built cube would. The idealisations that
matter most here (rigid sharp-edged boxes, MuJoCo soft contact, quasi-static sphere pockets) are listed below.
Rounded edges, rubber feet and a damped pawl could change the rates by some tens of percent. They cannot
change the momentum budget or the energy-window ratio.

## What a quasi-static edge pivot would need (slow geared or SMA joint)

These are static numbers from the model's derated magnets: nominal derate 0.80, 1.0 mm spheres at 7 mm scaled
with L.

| L | mass | face bond | far-pair peel moment | climb gravity torque (max) | support cube tips at |
| --- | --- | --- | --- | --- | --- |
| 7 mm | 0.85 g | 338 mN (40 weights) | 0.98 mN m | 42 uN m | 29 uN m |
| 8 mm | 1.13 g | 480 mN | 1.60 mN m | 63 uN m | 44 uN m |
| 9 mm | 1.48 g | 650 mN | 2.44 mN m | 92 uN m | 65 uN m |

1. **With the permanent sphere bonds it does not work.**
   1. A pivot that peels the far pair needs 1.5 x (peel + gravity), which is 1.5, 2.5 and 3.8 mN m at 7, 8
      and 9 mm.
   1. That torque reacts into the support cube through the hinge pair. The peel moment alone is 34-38 times
      the torque that tips a lone support (m g L / 2), and about 11 times what a 3-cube row resists
      (3 m g L / 2) on a bare floor.
   1. The pivot would roll or drag the neighbour instead of opening the bond.
   1. A ratchet or gear joint that holds load does not change this. The bond, not the joint, is the
      problem.
1. **With a releasable far pair it does.** If the far sphere pair can be pushed apart or turned to repel
   before the swing, the pivot only has to lift the cube:
   1. Joint torque: 42-92 uN m peak gravity at 7-9 mm, and 62, 94 and 138 uN m with 1.5x margin.
   1. Hinge: the two hinge pairs at mid-swing still pull 42-81 mN, 5 times the mover's weight, so a magnetic
      hinge holds a slow swing. The hinge must also take the actuator's reaction couple, at most the torque
      above over a lever of about 0.5 L, i.e. 18-31 mN. It holds with margin about 2.5.
   1. The support only has to resist the mover's static weight moment. A lone cube still tips: 42 uN m
      against 29 uN m at 7 mm. A row or a steel floor holds.
   1. Speed: a slow swing (seconds) stores no kinetic energy, so the energy window and the reaction kick both
      disappear.
1. **Release options** (all undesigned):
   1. A small slide or cam that pushes the far sphere 0.2-0.3 mm deeper into its pocket. Force falls as
      d^-4: +0.3 mm gives 0.4x.
   1. Electropermanent corners, which concepts.md rules out at this size.
   1. Bonds made weaker by design (smaller spheres), traded against the 3-cube cantilever.
1. The joint agent's SMA ratchet or micro gear train at about 8 mm needs to deliver about 0.1 mN m
   continuous (with margin) to lift, plus a release mechanism. Or it needs 2.5 mN m to peel, and that only
   works if the support is anchored by a structure or a steel floor.

## Motor and energy (the concept's flex-stator motor, computed with magpylib)

1. The diametric NdFeB core (radius 1.05 mm) over three sector coils gives k_t = 3.9e-6 N m/A (3.1e-6 with
   the 0.66 cylinder derate):
   1. That is 10 turns per coil from the 75/75 um geometry, R = 0.36 ohm per phase, 1.23 ohm with switches.
   1. Motor constant 2.8e-6 N m/sqrt(W), against the 4e-5 assumed in concepts.md, which used an
      axial-magnet field formula for a diametric magnet.
   1. An axially magnetised 2-pole full-radius rotor gives 1.1e-5 N m/A (3.6x), at the cost of the tungsten
      rim's inertia.
1. Spin-up from the three CeraCharges (12.6 mW for 1 s, then 4.2 mW):
   1. Roll speed (2,400 rad/s): 6.6 s and 30 mJ.
   1. roll_top (9,150 rad/s): 69 s and 230 mJ.
   1. Split (9,400 rad/s): 80 s and 265 mJ.
   1. Climb and descend (10,100-10,900 rad/s): more than 120 s and more than 0.39 J. concepts.md had 0.8 s
      and 43 mJ.
   1. With 1.5 J stored, that is 3-4 climbs or about 50 rolls a charge.
   1. Climbs need 11-13k rad/s, so jewel bearings (the plain-pin 6,000 rad/s cap excludes them at 7 mm).
      Floor rolls need under 2,900 rad/s, which plain bearings allow.
1. The spin-up reaction torque (at most 0.27 uN m) is far below the tip (29 uN m) and yaw-slip (more than
   4.7 uN m) thresholds, so the cube stays still while spinning up. This was checked analytically, not
   simulated in every trial.

## Fall (20 cm, bonded pair, random orientation)

1. Bond survival is 0.56 [0.49, 0.63]. Any landing that loads the bond in tension opens it, because the
   impulse the second cube needs (1.7e-3 N s) is far beyond the bond's 0.34 N over a 0.1 ms contact.
   MuJoCo's soft contact is longer than real impact, so this is if anything optimistic.
1. Hertz corner impact gives 6,130 g peak (scaling.py's model), with these consequences:
   1. Flywheel pin bearing stress 19.8 MPa against 40 MPa allowable (margin 2.0, as scaling.py).
   1. A corner sphere's inertial load is 0.24 N on its 0.1 mm resin cover, about 110 MPa Hertz contact
      stress. That is above typical resin compressive strength (80-100 MPa, assumed), so the cover may crack.
      This needs the bench drop test.

## NSGA-II and Pareto front

The search was capped at the coordinator's request, because no design is near the targets. It ran on the
steel floor with the escalating policy:
1. Variables: L 6-12 mm, sphere diameter 0.09-0.20 L, pocket clearance 5-50 um, flywheel radius 0.6-1.0 of
   its fit, thickness 0.3-1.5 mm, catch duration 50 us-5 ms, impulse scale 0.9-1.15.
1. Objectives: worst-move success over roll_floor, roll_top, climb_row, descend_row, split_row and dock, with
   8 trials per move; L; mean spin-up energy per move.
1. Size: population 12, 2 generations.

The front is in `hardware/runs/module_phys/rw_sim/pareto.json` (summary below, filled in from the run). At 8 trials per
move it cannot rank designs closer than about 0.3 in success. Its only robust message is that split_row stays
the worst move (0-0.12) for every design sampled.

The run: 24 evaluations in 9 minutes on 5 workers. Front (worst-move success from 8 trials per move with
retries; L; mean spin-up energy per move):

| L | worst move | energy per move | catch | spheres / L | note |
| --- | --- | --- | --- | --- | --- |
| 6.75 mm | 0.00 | 6.7 J (spin-up not reachable, penalised) | 4.9 ms brake | 0.16 | every move fails: the brake cannot punch through |
| 7.86 mm | 0.00 | 1.8 J | 0.70 ms | 0.14 | split_row and climb_row 0 of 8 |
| 9.50 mm | 0.125 | 5.1 J | 0.87 ms | 0.13 | climb and split 1 of 8 |
| 10.37 mm | 0.125 | 1.8 J | 0.35 ms | 0.11 | roll_top 8/8, descend 7/8, roll 6/8, dock 5/8; climb and split 1/8 |
| 11.91 mm | 0.00 | 1.4 J | 0.65 ms | 0.19 | big spheres: splits 0/8 |

The best point found (10.4 mm, 0.35 ms pawl, smaller spheres at 0.11 L) has the same failing pair, climb and
split at 1/8. Smaller spheres (weaker bonds) and short pawl catches score best, and brakes (catch over 1.5 ms)
score worst. No design came near 0.8 for climbs or splits, so the search was not extended. Energy per move is
dominated by the climb and split spin-ups (over 1 J each with the computed motor constant). That makes the
energy objective a motor problem, not a flywheel problem.

## Failure modes and what they need

| failure | where | fix type | what |
| --- | --- | --- | --- |
| support kicked or dragged (harm) | every move with a neighbour, bare floor | design or environment | quasi-static pivot plus releasable bond; or anchored supports (steel floor, larger structures) |
| impulse outside the energy window | climbs, splits, descents, rolls on top | design | quasi-static pivot; ballistic cannot be dosed to ±7 uJ against ±30-45 uJ of bond scatter |
| edge slip at friction below 0.3 | floor roll, dock | design plus bench | high-friction corner feet; measure friction on target floors |
| projectile hop above 1.6x tip energy | floor roll, dock | control | dose to 1.05-1.2x; closed loop from the IMU |
| yaw open loop to 3 degrees | yaw | control | corrections from the measured response (0.37 to 0.87) |
| yaw and passive capture on steel | steel floor variant | environment | the floor holds the cube: no yaw (0.07), capture 0.11 |
| bond opens in a fall | fall | accept or design | 44 % of pairs separate; bench drop test |
| sphere cover stress in a fall | fall | bench | 110 MPa Hertz on a 0.1 mm resin cover |
| slow, lossy spin-up | all, energy | design | axial 2-pole rotor (3.6x k_t), larger storage pulse power |
| catch damping and backlash | all ballistic | bench | modelled V-notch 5 degrees, damped pad (zeta 0.4-1.0); a 15 degree, zeta 0.3 notch rattles and makes the cube hop |

## Idealisations kept (every simplification is listed here and in `rw_cube.IDEALISATIONS`)

1. Corner spheres are uniformly magnetised rigid dipoles. This is exact for spheres; demagnetisation and
   induced magnetisation are ignored.
1. Rotor cores are point dipoles (far field; within 15 % of magpylib at the sphere positions, tested).
1. Sphere translation in the pocket is quasi-static: the sphere sits on the wall toward its magnetic force,
   with a low-pass over its travel time. No rattle; the 3 mg inertia is not simulated.
1. Pocket friction acts on sphere rotation only: mu_p x |magnetic force| x r. Impact loads do not add to it.
1. Bonded pairs start head to tail along their axis, as a formed bond leaves them. Other spheres start
   aligned to their local field.
1. Cubes are rigid boxes with sharp edges (MuJoCo box contacts). Real frames have rounded edges and 0.1 mm
   corner feet.
1. MuJoCo soft contact (0.1 ms time constant, stiffer during the catch): impacts and bounces are not Hertz
   physics. Restitution is a Monte Carlo variable.
1. Capillary adhesion acts per corner foot toward the floor only, not between cubes.
1. The pawl is a one-sided linear spring-damper (duration and damping as parameters), then a rigid lock. No
   wear, bounce-out or post-catch rattle. Catch durations above 1.5 ms are a constant-torque friction brake.
1. Spin-up is not simulated inside each trial. The rotor starts at the commanded speed with its error, and
   the reaction during spin-up is checked analytically.
1. Motor: ideal 6-step drive, torque constant averaged over angle, no cogging, power-limited by the
   CeraCharge datasheet numbers, converter efficiency 0.8 (A), no coil heating.
1. Bearings are ideal hinges with Couette drag; the speed caps (6k plain, 20k jewel) are checks, not dynamics.
1. The controller knows its own mass properties and uses nominal magnets and friction 0.4. Its impulse
   multiplier k is calibrated in simulation on n = 20 grids. The calibration ran before the sphere-start fix,
   and is used only to choose k.
1. Print tolerance enters as pocket position, clearance and cube size. Face flatness and frame warp are not
   modelled.
1. No air drag. Sensing for classification is ideal.
1. Steel floor variant: an ideal flat permeable half-space by image dipoles, scaled by 0.66 for saturation
   (simstack FEM).

## Not done, and why

1. The 11 mm pick-2 design was not run through the Monte Carlo (time). Its model builds (3.12 g, LiPo,
   plain bearings), and the physical mechanisms above are scale-independent: momentum ratio, bond weights
   (45 at 9 mm) and the energy-window ratio. Expect the same pattern.
1. 2,000-trial confirmation runs: no move exceeded 0.99, so none qualified.
1. 2x2x2 assembly and multi-move sequences were not simulated. With per-move rates of 0-0.7 they cannot
   reach the 0.95 whole-build target.
1. Re-running the calibration after the start-state fix (see idealisations).

## Bench tests this points to (first)

1. Pawl catch: momentum transfer, rebound, backlash, and whether the cube hops at 1.2-1.6x tip energy.
1. Edge slip and landing offset on wood, tile, paper and glass, with resin feet and rubber feet.
1. Peel moment and its scatter for 1.0 mm sphere pairs in printed pockets. This sets the energy window for
   climbs and splits, and the pivot torque.
1. Drop test of a filled dummy pair: bond survival and cover cracking.
