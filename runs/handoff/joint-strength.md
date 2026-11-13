# joint-strength: can a cube structure hold itself up?

Written 2026-11-13. Nothing in the grid world calls it yet; it is a filter and a design tool.

Code: `src/beebot/hardware/joint_strength.py` (the check), `src/beebot/hardware/joint_sim.py` (MuJoCo
ground truth), `scripts/joint_validate.py` (validation table), `scripts/joint_requirements.py` (parts
and hardware options), tests `tests/test_joint_strength.py`. Rows: `runs/joint_strength/`.

## Why

The grid world's rule (`world.supported`) says a cube stands if it is face-connected to the floor,
terrain or an object. A long cantilever, a wide span or a tall tower under a side load opens at its
weakest joint long before that. Goal libraries (shape datasets, later CAD parts) need a check that
says which shapes the hinge cube can actually hold, and the hardware plan needs to know what the
joints would need for more.

## The model

Rigid-block limit analysis (the lower-bound theorem): the structure stands if some set of joint
forces balances every cube within every joint's limits.

1. Cubes are rigid. The floor, terrain cells and anchored cubes are fixed.
1. Magnets (the hinge cube's, `cube_swarm.DESIGN`: two per edge, a quarter of the edge from its
   middle, and one at each face centre). Twin magnets across a closed face sit exactly at the peak
   of `cube.Magnets`' force law, so a closed face is clamped by all 8 edge pairs (4.5 module weights
   each) and the centre pair (0.3) at full pull: 36.3 weights, 53 N. These are constant forces, not
   capacities: any opening lowers the pull (the law falls past its peak), so a joint fails as soon
   as it must open. About one of its edges a face holds 4 x 4.5 + 0.5 x 0.3 = 18.15 m g L =
   1.34 N m. Cubes sharing only an edge are pulled together by two pairs at 0.905 of a twin's pull.
   The steel floor pulls each bottom magnet at 0.3 of a twin's (a cube holds 5.45 m g L to the
   floor); terrain is taken as the same steel.
1. Contacts press anywhere on a face: compression at the four corners, each with a friction
   pyramid (|f1| + |f2| <= mu c, mu 0.5; it also resists torsion). An edge-only contact is a pin
   (presses in the wedge between the two face normals, friction along the edge, no moment about
   it).
1. LP: unknowns are the corner forces along the pyramid edges (all >= 0) and a load factor;
   6 equilibrium rows per cube; maximise the factor (capped at `Joints.cap`, 10). "gravity" mode scales gravity
   and any extra load; "live" mode keeps 1 g and scales the extra load. The dual is the collapse
   mechanism: the cubes that move and the joints that open come out of the same solve.
1. Solver: no scipy in the venv (and no downloads), so a batched Mehrotra interior-point method in
   torch (float64, CPU). The normal matrix is assembled from 12 x 12 blocks (each force touches two
   cubes); structures split into connected components and are batched by size.

Not modelled: magnets' resistance to sliding (friction only: conservative), dynamic loads (moves,
impacts), wear, placement gaps, cube crushing, and geometric effects: a flat beam wedged between
immovable supports can carry any load by arching in rigid-block theory (see Spans).

## Validation against MuJoCo

`joint_sim.ramp`: cube_swarm's cubes and magnets as plain boxes (no flywheel), anchored cubes welded
to the world, every face closed at the start. The load factor rises at 0.25 per second (0.5 for the
long spans) from 0.05; the structure fails at the first factor at which a face joint opens 0.02 L
(1 mm) at a corner, a corner slides 0.05 L, or a cube moves 0.25 L. A first version used "any cube
moved 0.05 L", which caught elastic sag of long spans (soft contacts), not failure; the rows below
use the joint-opening rule. The load is then held 0.3 s and the cubes that moved 0.1 L or turned 5
degrees are the collapse set. Nominal cubes plus jittered seeds (cube_swarm's Jitter: magnet
strength +/- 25 % and mass +/- 5 % per cube, friction 0.3-0.7 per scene); the check is fed the same
draws per seed.

Load factor at failure (`>x`: held to the ramp's end; towers: side load at the top in module
weights at 1 g; everything else: multiple of g). Rows where everything held are listed after the
table. Collapse set overlap: Jaccard of the cubes MuJoCo moved and the check's mechanism.

| family | param | check (nominal) | MuJoCo nominal | MuJoCo seeds 1-3 | check seeds 1-3 | collapse set overlap |
| --- | --- | --- | --- | --- | --- | --- |
| cantilever | 4 | 2.27 | 2.21 | 2.07 / 2.01 / 2.01 | 2.12 / 1.97 / 2.05 | 1.00 |
| cantilever | 5 | 1.45 | 1.44 | 1.44 / 1.33 / 1.67 | 1.40 / 1.33 / 1.64 | 1.00 |
| cantilever | 6 | 1.01 | 1.03 | 0.81 / 1.21 / 0.89 | 0.79 / 1.17 / 0.86 | 1.00 |
| cantilever | 7 | 0.74 | 0.77 | 0.76 / 0.79 / 0.81 | 0.72 / 0.74 / 0.77 | 1.00 |
| cantilever | 8 | 0.57 | 0.61 | 0.66 / 0.55 / 0.54 | 0.60 / 0.50 / 0.50 | 1.00 |
| L | 3 | 1.81 | 1.87 | 1.76 / 1.48 / 1.58 | 1.71 / 1.42 / 1.53 | 1.00 |
| L | 4 | 0.84 | 0.89 | 0.91 / 0.82 / 1.01 | 0.86 / 0.76 / 0.96 | 1.00 |
| L | 5 | 0.49 | 0.55 | 0.43 / 0.60 / 0.46 | 0.38 / 0.55 / 0.40 | 1.00 |
| L | 6 | 0.33 | 0.39 | 0.43 / 0.40 / 0.37 | 0.38 / 0.34 / 0.32 | 1.00 |
| tower_side | 2 | 4.30 | 4.32 | >5 / 3.59 / 4.47 | >5 / 3.56 / 4.45 | 1.00 |
| tower_side | 3 | 2.78 | 2.79 | 2.60 / 2.91 / 2.37 | 2.58 / 2.88 / 2.34 | 1.00 |
| tower_side | 4 | 2.13 | 2.16 | 2.10 / 2.34 / 2.11 | 2.07 / 2.30 / 2.07 | 1.00 |
| tower_side | 5 | 1.77 | 1.82 | 2.01 / 1.63 / 1.80 | 1.97 / 1.58 / 1.75 | 1.00 |
| tower_side | 6 | 1.54 | 1.59 | 1.55 / 1.37 / 1.42 | 1.50 / 1.31 / 1.36 | 1.00 |
| tower_side | 7 | 1.38 | 1.44 | 1.46 / 1.34 / 1.53 | 1.40 / 1.28 / 1.47 | 1.00 |
| tower_side | 8 | 1.26 | 1.32 | 1.16 / 1.35 / 1.19 | 1.09 / 1.31 / 1.12 | 1.00 |
| tower_side | 9 | 1.17 | 1.24 | 1.32 / 1.22 / 1.18 | 1.25 / 1.19 / 1.13 | 1.00 |
| tower_side | 10 | 1.10 | 1.13 | 1.19 / 1.08 / 1.18 | 1.11 / 1.00 / 1.10 | 1.00 |
| tower_side | 11 | 1.04 | 1.12 | 1.07 / 1.10 / 1.10 | 1.00 / 1.02 / 1.02 | 1.00 |
| tower_side | 12 | 1.00 | 1.07 | 1.11 / 1.09 / 1.09 | 1.06 / 1.03 / 1.01 | 1.00 |
| tower_side | 13 | 0.96 | 0.99 | 0.96 / 0.98 / 1.07 | 0.91 / 0.93 / 1.00 | 1.00 |
| tower_side | 14 | 0.92 | 0.98 | 0.99 / 1.02 / 1.06 | 0.91 / 0.94 / 1.01 | 1.00 |
| tower_side | 15 | 0.89 | 0.97 | 0.90 / 1.05 / 0.93 | 0.82 / 0.98 / 0.85 | 1.00 |
| bracket | 0 | 0.57 | 0.61 | 0.59 / 0.61 / 0.65 | 0.54 / 0.56 / 0.60 | 1.00 |
| bracket | 1 | 0.74 | 0.79 | 0.71 / 0.91 / 0.81 | 0.66 / 0.87 / 0.78 | 1.00 |
| bracket | 2 | 1.01 | 0.93 | 0.91 / 0.96 / 1.05 | 0.87 / 0.92 / 1.01 | 1.00 |
| bracket | 3 | 1.45 | 1.48 | 1.60 / 1.77 / 1.30 | 1.60 / 1.73 / 1.32 | 1.00 |
| span_fixed_long | 15 | >6 | 3.53 | 3.24 | >6 | - |
| span_fixed_long | 20 | >6 | 1.39 | 1.19 | >6 | - |
| span_fixed_long | 25 | >6 | 0.57 | 0.64 | >6 | - |
| span_fixed_long | 30 | >6 | 0.31 | 0.32 | >6 | - |
| span_fixed_long | 40 | >6 | 0.14 | 0.11 | >6 | - |
| span_pillars_long | 12 | 2.98 | 2.43 | 2.45 | 2.86 | 0.94 |
| span_pillars_long | 14 | 2.09 | 1.87 | 1.83 | 1.98 | 0.97 |
| span_pillars_long | 16 | 1.55 | 1.48 | 1.50 | 1.55 | 0.90 |
| span_pillars_long | 18 | 1.19 | 1.18 | 1.16 | 1.22 | 0.95 |
| span_pillars_long | 20 | 0.93 | 0.86 | 0.85 | 0.84 | 0.96 |
| span_pillars_long | 24 | 0.60 | 0.58 | 0.57 | 0.58 | 1.00 |

Held to 3 g in both: cantilevers 1-3, L 1-2, corbelled arches of 2-4 steps, spans of 2-10 between
fixed abutments and on free pillars, tables (slab on four corner legs) of side 3, 5, 7, 9.

Rows: `runs/joint_strength/validate-a.jsonl` (cantilever, L, arch, bracket, tower; nominal + seeds
1-3), `validate-b.jsonl` (spans, tables; nominal + seed 1), table `validate-table.txt`.

**Agreement.** Stands at the design load (factor 1), all families but the spans between fixed
abutments, 196 runs: fails predicted and failed 43, holds predicted and held 147, failure predicted
but it held 5 (all within 0.08 of factor 1: towers of 12-15 and the L of 4, one seed each), held predicted but it
failed 1 (bracket with a 2-deep gusset, nominal: 1.01 against 0.93). False-fail rate 5 / 48 of the
predicted failures, false-hold rate 1 / 148 of the predicted holds. Error in the failure factor (119
runs where both fail before the ramp ends): median 4.7 %, mean -3.6 % (the check is slightly
conservative), worst 23 % (span on pillars 12, check 2.98 against 2.43; the pillar spans are the
one family where it is optimistic, by 2-23 %, falling with span). Error in the failure length,
nominal: cantilever 0 (both: 6 holds, 7 fails), L 0 (4), span on pillars 0 (20), tower with 1 W at
the top one storey short (check 12, MuJoCo 13), bracket one gusset step optimistic (check: a 2-deep
gusset holds an 8-long arm at 1.01; MuJoCo 0.93, it needs 3). The check fed each seed's jitter tracks MuJoCo seed by seed (cantilever 6: MuJoCo 0.81 / 1.21
/ 0.89, check 0.79 / 1.17 / 0.86). The collapse sets agree (Jaccard 0.90-1.00) and so does where
it fails: the root face of cantilevers, the floor under the column of an L or a tower, midspan and
the pillar bases of spans.

**Where it is wrong: a flat beam wedged between immovable supports.** Rigid blocks between rigid
abutments can carry any load by arching (the thrust runs from the top edge at the supports to the
bottom edge at midspan), so the check says spans between fixed walls never fail. MuJoCo's (soft)
contacts let the arch shorten and sag, and the beam fails at 3.5 g (span 15), 1.4 (20), 0.6 (25),
0.31 (30), 0.14 (40), approaching moment-only plastic theory (2 x 2 x 18.15 x 8 / w^2 = 290 / w^2)
at long spans. This matters for crossings between terrain banks (a reverse-demo family). Until the
check has a compressive-stress limit (bounded corner forces) or a stiffness-based arch check, treat
any shape that spans between terrain or fixed cubes by applying 290 / w^2 >= 1 (w up to 17) or by
checking it on free pillars instead.

**Speed** (4 threads, random face-connected blobs): 12 ms a shape of 10 cubes, 75 ms of 30, about
3 s of 100, about 20 s of 300 (batched 16 at a time; most of the time on large structures goes to
the normal matrix; a structure that stands with a large margin is the slowest). Of 100-cube blobs
about a fifth do not converge; then the check returns a certified lower bound (the best
primal-feasible factor, or the factor without edge contacts, which can only be lower) with
`converged=False`. Good enough as a filter on thousands of shapes up to about 30 cubes; for the
reversal's per-step verification, run it only on states the face rule already accepts and only
when a cube overhangs (a structure of full columns on the floor is pure compression and stands).

## Requirements study: what cube structures can carry and what the cube would need

Scope (per the user): basic structural parts, not assemblies or mechanisms. 5 cm cubes of 150 g:
a cube structure weighs 1.2 g/cm^3, like solid plastic. One face joint today: 53 N clamp, 1.34 N m
about an edge (18.15 cube weights x L), shear about 27 N by friction; an edge-only contact is a
pin. For comparison, a 5 x 5 cm section of solid timber holds about 800 N m in bending: in bending
a cube lattice is about 600 times weaker than the material it imitates, and in compression it is
as good as the cube shell. So the question is which parts are bending-dominated.

### What the hinge cube carries today (`scripts/joint_requirements.py`, load factor >= 1.5)

The margin is 1.5 because the validation shows +/- 10 % between cubes (magnet strength) and the
moves add dynamic loads that are not modelled. Sizes in cubes (5 cm each). Rows:
`runs/joint_strength/requirements.json`.

| part | as built | EPM faces x1.7 | latch floor | shear keys (mu 1.5) | face latches x5 | face latches x5 + latch floor | rigid lattice (x20, latch floor, keys) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cantilever from a wall, 1 deep (shelf bracket, awning) | 4 | 6 | 4 | 4 | 10 | 10 | 21 |
| cantilever, 2 deep | 7 | 9 | 7 | 6+ | 16 | 16 | 31+ |
| cantilever, 3 deep | 9 | 12 | 9 | 9 | 20 | 20 | 38+ |
| cantilever, 1 deep, 5 W (0.75 kg) at the tip | 2 | 3 | 2 | 2 | 7 | 7 | 17 |
| bracket: arm with a 3-deep stepped gusset | 7 | 9 | 7 | 7 | 13 | 13 | 24 |
| beam on two free-standing pillars, 1 deep | 16 | 20 | 18 | 20 | 26 | 39 | 40 (end of range) |
| beam on free pillars, 2 deep | 19 | 25 | 21 | 24 | 37 | 40 | 40 |
| beam between fixed walls, 1 deep | (40: the arching artefact; MuJoCo about 18) | | | | | | |
| column, 1 W side load at the top (height) | 6 | 10 | 18 | 6 | 6 | 40 | 40 |
| slab on 4 corner legs (side) | 13 | 15 | 13 | 15 | 15 | 15 | 15 (end of range) |
| portal frame 6 high, 1 W side load (width) | 18 | 20 | 19 | 18 | 20 | 20 | 20 (end of range) |
| braced portal frame (stepped diagonal) | 20 | 20 | 20 | 20 | 20 | 20 | 20 (end of range) |
| corbelled arch (steps) | 10 | 10 | 10 | 10 | 10 | 10 | 10 (end of range) |

`+`: the next size did not converge and its factor is a lower bound, so the part may reach further.
Options: EPM faces x1.7 is SMORES-EP's 88 N a face (the floor's pull scales with it, as it is
defined relative to a face); latch floor: the floor holds a cube as strongly as a face (the hinge
cube already needs latch features at the floor's cell edges for its floor moves); face latches x5:
about 270 N a face, the steel floor as built; shear keys: interlocking faces, friction raised to
1.5; rigid lattice: 20 x, about 1,070 N a face (a bolted joint).

### Holding force per 5 cm face that typical parts need (load factor 1.5)

Found by bisection on the faces' pull. Today: 53 N a face (x1).

| part | faces only (the steel floor as built) | faces and floor latch alike |
| --- | --- | --- |
| shelf: 30 cm cantilever (6), 2 kg at the tip | 400 N (x7.5) | 400 N |
| awning: 1 m cantilever (20), 1 deep | 890 N (x17) | 890 N |
| awning: 1 m cantilever, 2 deep | 384 N (x7.2) | 384 N |
| balcony: 2 m cantilever (40), 3 deep | 980 N (x18) | 980 N |
| beam: 1 m (20) on free pillars | 116 N (x2.2) | 65 N (x1.2) |
| beam: 2 m (40) on free pillars, 2 deep | impossible (the pillars tip over on steel) | 207 N (x3.9) |
| column: 1 m (20), 1 W side load at the top | impossible on steel | 57 N (x1.07) |
| column: 1 m, 5 W (0.75 kg) side load | impossible on steel | 403 N (x7.5) |
| table: 50 cm slab (10) on 4 legs | 17 N (x0.32) | 17 N |
| table: 75 cm slab (15) on 4 legs | 56 N (x1.05) | 54 N |

### Reading

1. **Compression parts are fine today; bending parts are not.** Slabs on legs to 65 cm, arches,
   braced and unbraced frames to about 1 m, beams on free pillars to 80 cm and walls and columns
   under their own weight need nothing new (the table even has 3 x margin at 50 cm). Cantilevers
   stop at 20 cm (1 deep) to 45 cm (3 deep), a column with a light side load at 30 cm. At the 1.5
   margin only the braced frame and the arch reach 1 m (20 cubes) as built; beams on pillars,
   the unbraced frame and the beam between walls fall just short (16-19), and the cantilevers and
   side-loaded columns far short.
1. **The floor is the weakest joint of every free-standing part.** The steel floor holds a cube at
   0.3 of a face, so Ls, towers and pillars tip over at the floor long before any cube joint opens
   (validated: the L and the towers fail at the floor in MuJoCo). No amount of face strength helps
   a free-standing column on steel ("impossible" above).
1. **Cantilevers scale badly with joint strength.** The root moment grows as n^2 (n^2 d / 2 for a
   d-deep arm), the capacity as d^2 times the pull, so the reach grows as the square root of the
   pull: x5 latches double a cantilever (4 -> 10), x20 makes it 21. Depth (a deeper section or a
   gusset) is cheaper than strength: 1 -> 3 deep takes it from 4 to 9.
1. **Bracing is free.** A stepped diagonal made of ordinary cubes takes a portal frame from 18 to
   the end of the range; a 3-deep gusset takes an 8-long arm from 0.57 to 1.45 (MuJoCo 0.61 ->
   1.48). A shape-to-cube converter should add gussets and depth before asking for stronger joints.

### Hardware augmentations, ranked by what they unlock per cost

1. **Latch floor used statically** (lock each floor cube to latch features in the floor cells).
   Unlocks every free-standing part: a column with a 1 W side load from 6 to 18 cubes, and with x5
   faces from 6 to the end of the range; beams on pillars 16 -> 18 (39 with latches). Cost: low,
   the hinge cube's floor moves already assume these latch features; the open question is holding
   force under load (a face's 53 N is enough for the gains above). Does nothing for parts hung from
   walls.
1. **Mechanical face latches, about 270 N a face** (x5; a bayonet or hook per face, or the hinge
   cube's edge latch made to hold under load on two opposite edges of a face, which gives about
   latch force x L of moment). Doubles cantilevers (4 -> 10, 3-deep 9 -> 20), brackets 7 -> 13,
   beams on pillars 16 -> 26. Cost: high (six latches or twelve edge latches a cube, their mass and
   actuation), but it is the one change that moves bending parts. A shelf with a 2 kg book needs
   400 N; a 1 m awning 384-890 N.
1. **Stronger switchable magnets (EPM x1.7, 88 N a face).** 4 -> 6 for a cantilever, 16 -> 20 for
   a beam, column 6 -> 10 (the floor's pull grows with it). Cost: moderate (magnet volume, mass,
   switching energy); keeps the design as it is. A modest, low-risk step.
1. **A rigidly locking lattice (bolted joints, about 1 kN a face).** Cantilevers to 1 m (21) and
   2 m with a 3-deep section; the cube becomes a structural material. Cost: very high (a screw or
   cam drive per face, slow to lock, heavy); and self-weight still limits a 1-deep arm at 1 m.
1. **Shear keys (interlocking faces).** Help only parts limited by sliding: beams on pillars
   16 -> 20 (the pillar bases sliding outward), nothing for cantilevers or columns. They also
   collide with the hinge cube's pivots unless retractable. Cheapest as passive geometry; lowest
   value.

Not ranked (dropped by scope): rotating joints and mechanisms.
