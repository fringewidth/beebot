# Simulation stack for millimetre-scale modular cubes

Code: `src/beebot/hardware/mmsim/`. Tests: `tests/test_mmsim.py` (21 tests, about 30 s). Raw numbers:
`runs/module_phys/simstack_validation.json` and `runs/module_phys/simstack_runs.json`. Reproduce with
`.venv/bin/python -m beebot.hardware.mmsim.validation` and `.venv/bin/python -m beebot.hardware.mmsim.report_runs`.
Nothing here is a design; this is the toolchain and how far each tool can be trusted.

## Headline

Validation rows passed, counting every published-data row (pull force 15 %, electrical 10 %, thermal 15 %): **65/86**.
That total is dominated by easy rows, so read it by category:

| category | passed / total | comment |
|---|---|---|
| pull to steel, first principles (image method, ideal iron, no tuning) | 1/8 | overpredicts vendor ratings by 13 to 36 % for 2-9.5 mm magnets, 2.4x and 3.5x for the two thin K&J discs |
| pull to steel, NGSolve FEM with saturating generic steel | 0/2 | both rows -15.3 %, just outside the 15 % bar; no tuning |
| pull to steel, image method times one derating factor (0.839) set on 3 rows | 3/5 held-out rows (+ 3 calibration rows excluded) | the 3 passing held-out rows are within 1.6 points of the bar; both thin K&J discs fail badly |
| inductance, magpylib flux linkage | 16/16 | max error 0.67 % |
| inductance, Mohan current-sheet formula | 16/16 | max error 4.2 % |
| inductance, Aebischer 2019 formula | 16/16 | max error 0.71 % |
| inductance, modified Wheeler | 6/16 | max error 27 %; do not use for these spirals |
| thermal, Onderdonk fusing current (adiabatic limit) | 4/4 | weak check, same physics |

Counting rule used for the 65/86: all rows in the JSON (18 pull rows = 8 image + 8 derated + 2 FEM; 64 inductance rows
= 16 coils x 4 methods; 4 thermal rows). Counting only the three methods one would actually use (image method undetated,
flux, Mohan) the picture is: pull 1/8, inductance 32/32.

Not in the count (verification rows, comparison against independent analytic calculation, not published values): 3 magnet-magnet
contact pulls versus a surface-pole integral (all within 2.6 %), FEM versus image method for linear steel (-3.6 %), RLC discharge
peak current and capacitor voltage versus a matrix exponential (both 0.00 %), and capacitor energy balance (0.00 %).

## What is installed and working

1. `magpylib` 6.3.0: working (force and torque via `getFT`, current polylines, cylinders and cuboids).
1. `ngsolve` 6.2.2608 (with `netgen-mesher`): installed from a macOS arm64 wheel in about 1 minute, working. Axisymmetric
   magnetostatics, linear and nonlinear (Picard on a Brauer B-H curve).
1. `pymoo` 0.6.2 (NSGA-II, GA) and `cma` 4.5.0: working.
1. MuJoCo 4.0.0 (already in the repo environment), scipy 1.18.1, numpy 2.5.4.
1. `uv` was not on PATH; used `/Users/hrishik/.hermes/tools/uv-0.12.3-darwin-arm64/uv pip install --python .venv/bin/python`.
   The packages are not added to `pyproject.toml` (no commit rule); add `magpylib ngsolve pymoo cma` there before merging.

## What failed to install or was not attempted

1. ngspice: no `brew` or conda on this machine, and `uv pip install ngspice` resolves to nothing. PySpice needs the libngspice
   shared library, so it was not tried. The power circuit is a plain scipy ODE (`circuit.py`); it cannot do arbitrary netlists.
1. K&J and supermagnete product pages return HTTP 403 to curl and WebFetch. All published pull-force values were read from web
   search result snippets that quote the product pages, not from the pages themselves. Treat the values as second-hand; the
   K&J D201-N52 value (0.25 lb) is the least certain (read from a similar-products listing, case number unconfirmed).
1. No published magnet-to-magnet pull force with a gap was found (K&J shows a distance chart as an image, not in text). The
   magnet-magnet code path is the same `getFT` call as the image-method path (the image of a magnet in an ideal plate is an
   identical magnet), so the steel rows test it, and it is cross-checked against an independent pole-model integral.

## Components

### Magnetics, analytic (`magnetics.py`)

Magnets (cylinder, cuboid, polarisation along local z) and current polylines. `force_magnet_magnet`, `force_coil_magnet`,
`force_magnet_coil`, `force_coil_coil` return force and torque (about a chosen pivot) as functions of pose. Magnets are meshed
into cells (default 12x3x4 for cylinders; the contact pull of a 3.175 x 0.794 mm disc is 2.628 N at this mesh against 2.670 N at
48x6x16, so the default is 1.6 % low at contact). Steel plate by the image method (`force_magnet_steel`): the image is the
mirrored magnet with negated polarisation (scaled by (mu_r-1)/(mu_r+1)); it assumes a linear, unsaturated, flat, infinite iron
half-space. Remanence by grade (N52 1.45 T, K&J quotes 1.48 T) was fixed before any validation and not adjusted.

Pull-force rows (image method, gap 0, no tuning):

| id | quantity | published | source | simulated | error % | tol % | pass | held-out |
|---|---|---|---|---|---|---|---|---|
| kj_d101_n52_image | pull to steel, image method, ideal iron, kj_d101_n52 | 0.3114 N | https://www.kjmagnetics.com/proddetail.asp?prod=d101-n52 | 1.082 | +247.5 | 15 | NO | yes |
| kj_d201_n52_image | pull to steel, image method, ideal iron, kj_d201_n52 | 1.112 N | https://www.kjmagnetics.com/d201-n52-neodymium-disc-magnet | 2.656 | +138.8 | 15 | NO | yes |
| kj_b444_n52_image | pull to steel, image method, ideal iron, kj_b444_n52 | 23.4 N | https://www.kjmagnetics.com/b444-n52-neodymium-block-magnet | 27.88 | +19.1 | 15 | NO | yes |
| kj_b666_n52_image | pull to steel, image method, ideal iron, kj_b666_n52 | 55.42 N | https://www.kjmagnetics.com/b666-n52-neodymium-block-magnet | 62.72 | +13.2 | 15 | yes | yes |
| sm_s0502_n52_image | pull to steel, image method, ideal iron, sm_s0502_n52 | 6.669 N | https://www.supermagnete.de/eng/disc-magnets-neodymium/disc-magnet-5mm-2mm_S-05-02-N52N | 8.985 | +34.7 | 15 | NO | yes |
| sm_s0502_n40_image | pull to steel, image method, ideal iron, sm_s0502_n40 | 5.099 N | https://www.supermagnete.de/eng/disc-magnets-neodymium/disc-magnet-5mm-2mm_S-05-02-N40N | 6.893 | +35.2 | 15 | NO | yes |
| sm_s0302_n48_image | pull to steel, image method, ideal iron, sm_s0302_n48 | 2.942 N | https://www.supermagnete.nl/eng/disc-magnets-neodymium/disc-magnet-3mm-2mm_S-03-02-N | 3.981 | +35.3 | 15 | NO | yes |
| sm_s0202_n48_image | pull to steel, image method, ideal iron, sm_s0202_n48 | 1.471 N | https://www.supermagnete.de/eng/disc-magnets-neodymium/disc-magnet-2mm-2mm_S-02-02-N | 2.008 | +36.5 | 15 | NO | yes |

The vendor numbers sit 26 to 27 % below the ideal-iron model for every supermagnete 2-5 mm magnet (a consistent ratio), 16 %
below for the K&J 6.35 mm cube, 12 % for the 9.5 mm cube, and far below for the two thin K&J discs (0.8 mm thick: the model gives
2.4x and 3.5x the published value). Candidate causes, none separated by this work: iron saturation (the FEM below shows
peak flux density in the plate of 1.0 to 1.2 T in the saturating run and 1.3 to 1.6 T in the linear run, and saturation alone removes about 35 % of the ideal-iron force),
plating and surface gap, vendor derating or guaranteed-minimum ratings, and tilted peel-off in vendor tests. Because the cause is
unresolved, the model is an upper bound on real adhesion, not an estimate.

Derating row set (one factor, the median of vendor/model on three calibration rows `kj_b444`, `kj_b666`, `sm_s0502_n52`; the other five rows
are held out):

| id | quantity | published | source | simulated | error % | tol % | pass | held-out |
|---|---|---|---|---|---|---|---|---|
| kj_d101_n52_derated | pull to steel, image method x derate 0.839, kj_d101_n52 | 0.3114 N | https://www.kjmagnetics.com/proddetail.asp?prod=d101-n52 | 0.9081 | +191.6 | 15 | NO | yes |
| kj_d201_n52_derated | pull to steel, image method x derate 0.839, kj_d201_n52 | 1.112 N | https://www.kjmagnetics.com/d201-n52-neodymium-disc-magnet | 2.229 | +100.5 | 15 | NO | yes |
| kj_b444_n52_derated | pull to steel, image method x derate 0.839, kj_b444_n52 | 23.4 N | https://www.kjmagnetics.com/b444-n52-neodymium-block-magnet | 23.4 | +0.0 | 15 | yes | no |
| kj_b666_n52_derated | pull to steel, image method x derate 0.839, kj_b666_n52 | 55.42 N | https://www.kjmagnetics.com/b666-n52-neodymium-block-magnet | 52.64 | -5.0 | 15 | yes | no |
| sm_s0502_n52_derated | pull to steel, image method x derate 0.839, sm_s0502_n52 | 6.669 N | https://www.supermagnete.de/eng/disc-magnets-neodymium/disc-magnet-5mm-2mm_S-05-02-N52N | 7.541 | +13.1 | 15 | yes | no |
| sm_s0502_n40_derated | pull to steel, image method x derate 0.839, sm_s0502_n40 | 5.099 N | https://www.supermagnete.de/eng/disc-magnets-neodymium/disc-magnet-5mm-2mm_S-05-02-N40N | 5.785 | +13.4 | 15 | yes | yes |
| sm_s0302_n48_derated | pull to steel, image method x derate 0.839, sm_s0302_n48 | 2.942 N | https://www.supermagnete.nl/eng/disc-magnets-neodymium/disc-magnet-3mm-2mm_S-03-02-N | 3.342 | +13.6 | 15 | yes | yes |
| sm_s0202_n48_derated | pull to steel, image method x derate 0.839, sm_s0202_n48 | 1.471 N | https://www.supermagnete.de/eng/disc-magnets-neodymium/disc-magnet-2mm-2mm_S-02-02-N | 1.685 | +14.6 | 15 | yes | yes |

### Magnetics, FEM (`fem.py`)

NGSolve installed and works. Axisymmetric vector-potential formulation, permanent magnet as a polarisation source, force from the
Maxwell stress integrated over a plane in the air gap (needs gap > 0; 20 um used for plating). Steel is linear (mu_r) or nonlinear
(generic Brauer curve, mu_r about 1900 at 1 T and 30 at 2 T, not a specific steel grade, not fitted). Checks:
linear steel mu_r 1000, 5x2 mm N52, gap 0.1 mm: FEM 6.67 N vs image 6.92 N (-3.6 %, mesh plus magnet permeability 1.0).
The saturating curve gives 0.66 of the linear result at gaps of 25 and 100 um for the same magnet.

| id | quantity | published | source | simulated | error % | tol % | pass | held-out |
|---|---|---|---|---|---|---|---|---|
| sm_s0502_n52_fem | pull to steel, NGSolve axisymmetric FEM nonlinear generic steel, sm_s0502_n52 | 6.669 N | https://www.supermagnete.de/eng/disc-magnets-neodymium/disc-magnet-5mm-2mm_S-05-02-N52N | 5.648 | -15.3 | 15 | NO | yes |
| sm_s0302_n48_fem | pull to steel, NGSolve axisymmetric FEM nonlinear generic steel, sm_s0302_n48 | 2.942 N | https://www.supermagnete.nl/eng/disc-magnets-neodymium/disc-magnet-3mm-2mm_S-03-02-N | 2.491 | -15.3 | 15 | NO | yes |

Both FEM rows are the same physics with a different magnet, so they are one piece of evidence, not two. Mesh: about 1200 triangles,
order 2, 0.1 s per solve; mesh refinement from 700 to 1240 elements changed the linear force by 6 %, so the FEM numbers carry a few percent of mesh error.

### Coil electrical and thermal (`coil.py`)

Planar spiral polylines (square following Aebischer fig. 1, or round), 1-4 layers in series, resistance with copper tempco 0.00393/K,
inductance by Mohan, modified Wheeler, Aebischer 2019, and a magpylib flux-linkage integral. The flux method treats the spiral as nested
rings of the same total length, integrates the outer ring's field through the inner ring's area (graded Gauss-Legendre quadrature),
and uses Maxwell's geometric mean distance 0.2235 (w + t) for self terms. Validated against the 16 coils of Aebischer 2019 (tables 3
and 4, FastHenry values, copper 35 um, N 3-5, A 10-100 mm, measurements agree within 2.1 %). All 16 coils are 10-100 mm; the
sub-5 mm, 12-35 um copper, 75-100 um trace regime has no published coil data in this set. The formulas are scale-invariant in
the dimensionless ratios, which is the only argument for using them at 4 mm.

Inductance, flux method (every row):

| id | quantity | published | source | simulated | error % | tol % | pass | held-out |
|---|---|---|---|---|---|---|---|---|
| L_aeb01_flux | inductance, square spiral N=3 A=20 mm pitch 12 mil trace 6 mil, flux | 542 nH | https://www.aemjournal.org/index.php/AEM/article/view/1074 (Aebischer, Adv. Electromagn. 8(4), 2019, tables 3 and 4) | 539.8 | -0.4 | 10 | yes | yes |
| L_aeb02_flux | inductance, square spiral N=3 A=20 mm pitch 20 mil trace 6 mil, flux | 480 nH | https://www.aemjournal.org/index.php/AEM/article/view/1074 (Aebischer, Adv. Electromagn. 8(4), 2019, tables 3 and 4) | 478.1 | -0.4 | 10 | yes | yes |
| L_aeb03_flux | inductance, square spiral N=3 A=20 mm pitch 30 mil trace 6 mil, flux | 426 nH | https://www.aemjournal.org/index.php/AEM/article/view/1074 (Aebischer, Adv. Electromagn. 8(4), 2019, tables 3 and 4) | 424.7 | -0.3 | 10 | yes | yes |
| L_aeb04_flux | inductance, square spiral N=3 A=20 mm pitch 40 mil trace 6 mil, flux | 385 nH | https://www.aemjournal.org/index.php/AEM/article/view/1074 (Aebischer, Adv. Electromagn. 8(4), 2019, tables 3 and 4) | 383.8 | -0.3 | 10 | yes | yes |
| L_aeb05_flux | inductance, square spiral N=3 A=20 mm pitch 30 mil trace 12 mil, flux | 401 nH | https://www.aemjournal.org/index.php/AEM/article/view/1074 (Aebischer, Adv. Electromagn. 8(4), 2019, tables 3 and 4) | 398.3 | -0.7 | 10 | yes | yes |
| L_aeb06_flux | inductance, square spiral N=3 A=50 mm pitch 12 mil trace 6 mil, flux | 1716 nH | https://www.aemjournal.org/index.php/AEM/article/view/1074 (Aebischer, Adv. Electromagn. 8(4), 2019, tables 3 and 4) | 1711 | -0.3 | 10 | yes | yes |
| L_aeb07_flux | inductance, square spiral N=3 A=50 mm pitch 20 mil trace 6 mil, flux | 1575 nH | https://www.aemjournal.org/index.php/AEM/article/view/1074 (Aebischer, Adv. Electromagn. 8(4), 2019, tables 3 and 4) | 1572 | -0.2 | 10 | yes | yes |
| L_aeb08_flux | inductance, square spiral N=3 A=50 mm pitch 40 mil trace 6 mil, flux | 1372 nH | https://www.aemjournal.org/index.php/AEM/article/view/1074 (Aebischer, Adv. Electromagn. 8(4), 2019, tables 3 and 4) | 1370 | -0.2 | 10 | yes | yes |
| L_aeb09_flux | inductance, square spiral N=3 A=50 mm pitch 20 mil trace 12 mil, flux | 1510 nH | https://www.aemjournal.org/index.php/AEM/article/view/1074 (Aebischer, Adv. Electromagn. 8(4), 2019, tables 3 and 4) | 1502 | -0.5 | 10 | yes | yes |
| L_aeb10_flux | inductance, square spiral N=3 A=50 mm pitch 30 mil trace 12 mil, flux | 1391 nH | https://www.aemjournal.org/index.php/AEM/article/view/1074 (Aebischer, Adv. Electromagn. 8(4), 2019, tables 3 and 4) | 1387 | -0.3 | 10 | yes | yes |
| L_aeb11_flux | inductance, square spiral N=3 A=50 mm pitch 40 mil trace 12 mil, flux | 1305 nH | https://www.aemjournal.org/index.php/AEM/article/view/1074 (Aebischer, Adv. Electromagn. 8(4), 2019, tables 3 and 4) | 1301 | -0.3 | 10 | yes | yes |
| L_aeb12_flux | inductance, square spiral N=4 A=50 mm pitch 40 mil trace 12 mil, flux | 2037 nH | https://www.aemjournal.org/index.php/AEM/article/view/1074 (Aebischer, Adv. Electromagn. 8(4), 2019, tables 3 and 4) | 2033 | -0.2 | 10 | yes | yes |
| L_aeb13_flux | inductance, square spiral N=5 A=50 mm pitch 40 mil trace 12 mil, flux | 2849 nH | https://www.aemjournal.org/index.php/AEM/article/view/1074 (Aebischer, Adv. Electromagn. 8(4), 2019, tables 3 and 4) | 2844 | -0.2 | 10 | yes | yes |
| L_aeb14_flux | inductance, square spiral N=3 A=100 mm pitch 40 mil trace 12 mil, flux | 3173 nH | https://www.aemjournal.org/index.php/AEM/article/view/1074 (Aebischer, Adv. Electromagn. 8(4), 2019, tables 3 and 4) | 3167 | -0.2 | 10 | yes | yes |
| L_aeb15_flux | inductance, square spiral N=5 A=10 mm pitch 12 mil trace 6 mil, flux | 446 nH | https://www.aemjournal.org/index.php/AEM/article/view/1074 (Aebischer, Adv. Electromagn. 8(4), 2019, tables 3 and 4) | 443.5 | -0.6 | 10 | yes | yes |
| L_aeb16_flux | inductance, square spiral N=3 A=50 mm pitch 30 mil trace 18 mil, flux | 1352 nH | https://www.aemjournal.org/index.php/AEM/article/view/1074 (Aebischer, Adv. Electromagn. 8(4), 2019, tables 3 and 4) | 1344 | -0.6 | 10 | yes | yes |

Rows for the other three methods are in the JSON (`rows`, ids `L_aeb01_mohan` etc.).

Thermal: two lumped nodes (copper, polyimide) with temperature-dependent resistance, convection h = 10 W/m2/K over both faces.
Only the adiabatic copper limit is checked against published data (Onderdonk equation). Convection and the polyimide node are not
validated against published data; polyimide constants (rho 1420, cp 1090, k 0.12) are from memory of the Kapton datasheet.

| id | quantity | published | source | simulated | error % | tol % | pass | held-out |
|---|---|---|---|---|---|---|---|---|
| onderdonk_0.0001s | copper trace 75 um x 18 um fusing current for 0.0001 s, adiabatic limit of the thermal model | 39.21 A | https://pcdandf.com/pcdesign/index.php/editorial/menu-features/17314-onderdonk-s-and-preece-s-equations-how-do-they-compare | 39.59 | +1.0 | 15 | yes | yes |
| onderdonk_0.001s | copper trace 75 um x 18 um fusing current for 0.001 s, adiabatic limit of the thermal model | 12.4 A | https://pcdandf.com/pcdesign/index.php/editorial/menu-features/17314-onderdonk-s-and-preece-s-equations-how-do-they-compare | 12.52 | +1.0 | 15 | yes | yes |
| onderdonk_0.01s | copper trace 75 um x 18 um fusing current for 0.01 s, adiabatic limit of the thermal model | 3.921 A | https://pcdandf.com/pcdesign/index.php/editorial/menu-features/17314-onderdonk-s-and-preece-s-equations-how-do-they-compare | 3.959 | +1.0 | 15 | yes | yes |
| onderdonk_0.1s | copper trace 75 um x 18 um fusing current for 0.1 s, adiabatic limit of the thermal model | 1.24 A | https://pcdandf.com/pcdesign/index.php/editorial/menu-features/17314-onderdonk-s-and-preece-s-equations-how-do-they-compare | 1.252 | +1.0 | 15 | yes | yes |

Example coils (supply 3 V through 0.3 ohm of driver and wiring; safe current for copper peak 100 C from 25 C ambient, pulse durations
1 ms / 10 ms / 100 ms / 1 s; polyimide node on, h 10 W/m2/K). Inductance here is the flux method, all layers in series:

| coil | total turns | R ohm | L nH | I at 3 V A | NI at 3 V | safe I A | safe NI |
|---|---|---|---|---|---|---|---|
| 4mm 2L 18um 75/75 | 10 | 1.67 | 617 | 1.52 | 15.2 | 5.26 / 2.32 / 0.95 / 0.32 | 53 / 23 / 10 / 3 |
| 4mm 4L 12um 75/75 | 20 | 5.01 | 2389 | 0.56 | 11.3 | 3.62 / 1.72 / 0.70 / 0.23 | 72 / 34 / 14 / 5 |
| 4mm 2L 35um 100/100 | 8 | 0.51 | 389 | 3.69 | 29.5 | 13.21 / 5.08 / 1.96 / 0.65 | 106 / 41 / 16 / 5 |
| 8mm 2L 18um 75/75 | 20 | 6.57 | 4791 | 0.44 | 8.7 | 5.26 / 2.32 / 0.95 / 0.32 | 105 / 46 / 19 / 6 |

The pulsed limits are set by heating, not the supply: at 1 ms the safe NI is 50 to 100 ampere-turns for these coils, while a 3 V
supply through 0.3 ohm drives only 9 to 30 ampere-turns DC.

### Power circuit (`circuit.py`)

Storage element (MLCC bank with optional DC-bias derating, supercap with ESR, or a cell with open-circuit voltage linear in
state of charge and series resistance), H-bridge with Rds_on per switch, coil R and L, optional back EMF, freewheel after the drive
phase. Outputs droop, peak current, energy per pulse (terminal, coil loss, switch loss). Example, coil = 4 mm 2-layer 18 um spiral
(R = 1.67 ohm, L = 0.62 uH), 200 us drive, Rds_on 0.1 ohm per switch:

| storage | peak I A | droop at peak V | droop of open-circuit after pulse V | terminal energy uJ | coil loss uJ | switch loss uJ |
|---|---|---|---|---|---|---|
| MLCC 4x10uF X5R 3.3V derate0.5 | 1.725 | 0.0765 | 3.09 | 201.832 | 180.245 | 21.587 |
| supercap 10mF 5ohm ESR 3.3V | 0.4803 | 2.4 | 0.00959 | 86.056 | 76.852 | 9.204 |
| CeraCharge 100uAh 200ohm | 0.007926 | 1.59 | 2.64e-06 | 0.024 | 0.021 | 0.003 |

The CeraCharge preset uses 100 uAh, 1.0 to 1.6 V from the TDK datasheet copy at novapart.co (B73180A0101M062); the 200 ohm
internal resistance comes from a third-party listing of a sibling part number and is unconfirmed. At 200 ohm the cell delivers
8 mA and about 0.02 uJ per 200 us pulse into this coil, so a cell of this type cannot drive coil pulses directly; it must charge a
capacitor. The circuit model is verified against analytic RLC solutions and energy balance, not against published circuit
measurements.

### MuJoCo coupling (`mjcoupling.py`, `demo.py`)

`MagneticCoupling` takes `Attached(body, Magnet|Coil)` sources, rebuilds world poses from `data.xpos/xmat` each step, evaluates the
wrench on body b from the sources of body a once per unordered pair, and derives the reaction on a:
F_a = -F_b and tau_a = -tau_b - (p_b - p_a) x F_b (about the two centres of mass). Linear momentum and angular momentum about any
point are then conserved by construction. Forces go through `xfrc_applied` at the centre of mass.

`CylPairTable` is a 5-D lookup (rho, z, nx, ny, hemisphere) for identical cylinder pairs, valid to rho = 4.5 d, |z| up to h + 3 d, and
30 degrees of axis tilt from +-z; out-of-range pairs fall back to direct evaluation per pair (`hybrid_wrench`). Penetration is clamped to contact.

`SurfaceForces` adds adhesion (constant pull per contact point) and Coulomb friction with stiction (regularised Stribeck law: mu falls from
mu_s to mu_k over v_stick, force ~ v / sqrt(v^2 + v_eps^2), so stiction creeps at about v_eps = 10 um/s). The geoms' own MuJoCo friction
must be zero when it is on.

Momentum checks (free space, two cubes with 4 magnets each plus a 2 A coil, 3 ms): linear momentum drift 3.4e-21
kg m/s against a scale of 2.7e-04 (machine precision). Angular momentum about the system centre drifts by
1.0e-08, 4.3e-09, 1.8e-09 kg m2/s for dt = 40, 20, 10 us: first order in dt, an
integrator effect (the wrench balance itself is exact to 1e-10 relative in the test).

Speed, a pair of 5 mm cubes with 12 magnets of 1 mm x 1 mm each (144 magnet pairs, M = 12x3x4 cells per magnet):

| method | time per pair of cubes | notes |
|---|---|---|
| direct getFT, default mesh | 24.6 ms | 144 pairs, one call |
| direct getFT, 24x4x8 mesh | 205 ms | about 1 % closer to converged at contact |
| lookup table (hybrid) | 1.8 ms | 99.5 % of pairs served by the table, rest direct |

Table: 34400 points (16 x 43 x 5 x 5 x 2 hemispheres), build 31 s once per magnet type. Interpolation error against direct
evaluation: pair of cubes, force 2.1 % median and 2.8 % at the 90th percentile, torque 1.8 % and 3.4 %;
single magnet pair over random poses (rho up to 2.5 mm, gap 0.06 to 2 mm, tilt up to 25 degrees): 2.7 % median, 7.4 % at 90th percentile,
27 % worst case near contact. The lookup is about 14x faster than direct for this pair; in the 4-magnet seating demo the
whole step is 4-6 ms against 7 ms because MuJoCo contacts and the surface model are a fixed cost.

Seating demo: two free 5 mm cubes of 0.3 g, four 1 mm x 1 mm N52 magnets near the corners of the facing faces (axes along the normal,
checkerboard polarity so matched faces attract and a 90 degree twist repels), starting 1.5 mm apart with 0.3 mm lateral offset and 2 degrees of
misalignment, on a floor with gravity. Approach speed is about 1 m/s; the cubes tilt like a closing book (face-to-face attraction is unstable against
tilt about an axis in the face) and seat flush at about 3 ms. With adhesion 0.2 mN per contact point and mu_s 0.5, mu_k 0.3, they stay seated.
Gap (mm, negative = penetration), residual lateral offset in y (mm, started at 0.3), twist (degrees), wall milliseconds per step, 30 ms runs:

| coupling | dt s | surface forces | gap mm | y mm | twist deg | ms per step |
|---|---|---|---|---|---|---|
| direct | 0.0001 | plain | +39.1105 | -18.547 | 1.1019 | 6.5 |
| direct | 0.0001 | surface | +10.7212 | +0.232 | 31.3739 | 7.0 |
| direct | 5e-05 | plain | +0.0241 | +0.358 | 0.1007 | 6.9 |
| direct | 5e-05 | surface | -0.0031 | +0.007 | 0.0000 | 7.6 |
| direct | 2e-05 | plain | +0.1467 | +0.491 | 0.0114 | 7.0 |
| direct | 2e-05 | surface | -0.0044 | +0.198 | 0.0000 | 7.6 |
| direct | 1e-05 | plain | +0.0615 | -0.084 | 0.0008 | 7.0 |
| direct | 1e-05 | surface | -0.0038 | +0.192 | 0.0000 | 7.7 |
| lookup | 0.0001 | plain | +19.1523 | +5.469 | 19.1645 | 13.3 |
| lookup | 0.0001 | surface | +0.0883 | +0.024 | 3.1062 | 4.7 |
| lookup | 5e-05 | plain | +0.0142 | -0.490 | 0.0728 | 3.9 |
| lookup | 5e-05 | surface | -0.3809 | -1.883 | 0.1620 | 5.6 |
| lookup | 2e-05 | plain | -0.0019 | +0.129 | 0.0041 | 4.1 |
| lookup | 2e-05 | surface | -0.0069 | +0.250 | 0.0061 | 5.5 |
| lookup | 1e-05 | plain | +0.0063 | +0.082 | 0.0000 | 4.3 |
| lookup | 1e-05 | surface | -0.0037 | +0.189 | 0.0000 | 5.8 |

Reading: dt of 5e-5 s and below is stable for direct evaluation; dt = 1e-4 s fails (the cubes rebound and leave, MuJoCo contact time
constant is set to 2.2 dt). With surface forces the final gap is -0.003 to -0.004 mm at dt 5e-5 to 1e-5 (direct) and the cubes stay put; without them
the cubes chatter (gap 0.02 to 0.15 mm, lateral offset varies run to run). Stiction freezes the lateral offset at 0.0 to 0.2 mm of the starting 0.3 mm; the exact
value depends on dt and on the stiction creep scale, so the final lateral alignment is not a predicted number. The lookup at dt 5e-5 with surface forces ended
-0.38 mm penetrated and 1.9 mm off in y: the table's 27 % worst-case error near contact is enough to disturb a tight dynamic event; use direct evaluation or dt of
2e-5 and below for seating studies.

### Optimisation and uncertainty (`opt.py`)

`run_search(fn, lower, upper, n_obj, algo in {nsga2, ga, cma}, workers)` evaluates the objective in a spawn process pool, capped at 6 workers
(`_cap(64)` returns 6, tested). `monte_carlo(fn, params, n, tolerances, workers)` perturbs parameters by name prefix
(`magnet_pos` +-0.05 mm absolute, `magnetisation` +-5 % relative, `friction` +-30 % relative, `print_dim` +-0.05 mm absolute; uniform within the
bounds by default, normal option takes the half-width as 2 sigma) and returns the success rate with a Wilson 95 % interval.

Toy demonstrations (6 workers): NSGA-II on ZDT1, 480 evaluations in 0.6 s with a 23-point front; GA and CMA-ES on a shifted sphere reach
1.5e-04 and 4.7e-06. NSGA-II on a real objective (magnet pull vs volume over gap and diameter, magpylib in the loop): 128 evaluations,
15-point front, 0.8 s. Monte Carlo on a toy latch (two 1 mm magnets at 0.3 mm gap must give at least 0.165 N and hold a
0.075 N load by stiction): 164/300 = 0.547, Wilson 95 % [0.490, 0.602], in 0.8 s.

## Model limits the designers must respect

1. Pull forces to steel are an upper bound. First-principles (ideal iron, no saturation) overpredicts vendor adhesion ratings by 13 to 37 % for
   2-9.5 mm magnets, and by 2.4x to 3.5x for 0.8 mm thick, 1.6-3.2 mm diameter discs. The derating factor 0.84 was set on three
   rows only and does not rescue thin discs. Any design using sub-mm-thick magnets or steel pole pieces needs a measured pull-force number
   (one afternoon with a force gauge) before the optimiser trusts these tools. No published magnet-to-magnet gap data was found, so the
   magnet-to-magnet path is verified only against an independent analytic integral.
1. Saturating steel matters. With a generic mild-steel curve the FEM pull is 0.66 of the linear result and the plate carries 1.0 to 1.2 T (1.3 to 1.6 T if linear) under
   a 5 x 2 mm magnet. The image method has no saturation, so for steel or soft-magnetic flex layers use `fem.py`, and note that the B-H curve used
   is generic: the FEM result is only as good as the real material curve.
1. Inductance is validated only for 10-100 mm coils (16 published coils, flux and Aebischer within 1 %, Mohan within 4.2 %, Wheeler unusable). Nothing in
   this set is below 10 mm, with thin copper and 75-100 um traces. Resistance and thermal behaviour have no published coil measurements here:
   resistance is rho L / (w t) with no corner, via or skin-effect terms; the thermal model is validated only in the adiabatic limit and its
   convection coefficient (10 W/m2/K) and polyimide properties are assumptions. At the 75 um trace scale expect fabricated copper to be
   thinner and rougher than nominal, raising R.
1. Multi-layer inductance (flux method) assumes concentric rings and identical layers; layers are in series with the same sense. Mohan's formula
   is single layer only. Mutual coupling between layers is included in the flux method only.
1. Coil forces use filament currents (no conductor width): accurate beyond about one trace pitch from the coil, optimistic closer. Coil-on-magnet
   forces were checked against a dipole formula only (3 %), not against published data.
1. The circuit is an ODE with ideal switches, no dead time, no ringing, no parasitic capacitance, no thermal feedback into the coil resistance.
   The solid-state cell is a linear-OCV model with a single series resistance; real cells have a rate and temperature dependence not modelled.
1. The MuJoCo coupling applies forces at the start of each step (explicit coupling). Stable at dt of 5e-5 s and below for 5 mm cubes with 1 mm magnets
   and 0.3 g mass, and not at 1e-4 s. Smaller or lighter modules (shorter time constants) need smaller steps: scale dt with sqrt(m d / F).
   Contact is MuJoCo soft contact, not impact physics; rebound and chatter at seating are not predictive.
1. Stiction and adhesion are parametric (mu_s, mu_k, adhesion per contact, Stribeck width), with no published micro-scale values behind them. They
   change the seated lateral offset from 0 to 0.2 mm in the demo. Treat contact friction as a Monte Carlo variable, not a constant.
1. The lookup table is valid only for identical cylinder magnets within its rho, z and tilt box (about 2 % median, up to 27 % worst-case error, mostly
   near contact). Cuboid magnets and coils always use direct evaluation (about 0.2 ms per magnet pair at the default mesh).
1. Sintered magnet remanence at mm scale: N52 Br (1.45 T used) is a grade nominal; magnet to magnet scatter of a few percent and demagnetisation at corners
   are not modelled. Temperature is not modelled for magnets.
1. All comparisons use second-hand vendor snippets (pages return 403); re-verify the K&J D101, D201 values on the live pages before using the thin-disc conclusion.
