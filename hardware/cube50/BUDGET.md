# Cube50 budget and build time

**Changed 2026-11-17 11:24 (+0530):** hinge deploy drive priced (lead-screw gearmotor D1, nut D2,
third DRV8833 E9, tie rods S6, 4 more switches), offset coupling C1 removed, hinge train 18T-18T-18T
in m0.5, tiles about 60 mm and 30 g. The disc-train rows (and B1, S3, P2, F1) still carry the 50 mm
figures until the CAD agent's final bought-parts list arrives.

Prices from BOM.md (looked up 2026-11-17, 1 USD = 88 INR); totals computed by `bom_build.py`.
Every total includes, per order: JLCPCB assembly fees (4,907 INR), shipping from three suppliers
(5,544 INR, estimate) and 30 % duty and GST on imported parts (estimate). Spares: 20 % rounded, at
least one, for any fleet above one cube. Floor tiles: two per cube, about 60 mm, 30 g of PLA each
(39 INR).

## Per cube (parts only)

Updated 2026-11-17 11:24 for the hinge deploy drive (BOM rows D1, D2, E3, E9, G1, S1, S6; C1 and G2
removed). Nothing in the hinge is a placeholder any more; the disc-train rows wait for the 60 mm CAD.

| | INR | USD |
| --- | --- | --- |
| with Pololu HP gearmotors (the motors the torque budgets use) | 19,496 | 222 |
| with generic N20s (specs unverified, bench test 2 and 3 first) | 9,986 | 113 |

The four Pololu motors are 10,542 INR, 54 % of a cube. Next largest: gears 2,720 (hardened-steel m0.5
hinge and disc trains), the deploy drive 1,857 (two DFRobot FIT0577 lead-screw gearmotors 1,756, the
third DRV8833, nuts, tie rods), main board parts 1,021, disc contacts 549, SMA wire 433, lip rings 240.

## Fleets

| build | cubes built | Pololu motors, INR | USD | generic motors, INR | USD |
| --- | --- | --- | --- | --- | --- |
| 1 prototype | 1 | 34,749 | 395 | 19,436 | 221 |
| a pair (pivot tests) | 2 + 1 spare | 83,267 | 946 | 42,607 | 484 |
| quadruped | 14 + 3 spares | 423,280 | 4,810 | 205,197 | 2,332 |
| quadruped and arm sets | 25 + 5 spares | 738,997 | 8,398 | 356,163 | 4,047 |

Suggested path: build the pair with Pololu motors (known figures; tests 1-4 of ASSEMBLY.md), fit the
pair's spare cube with generic N20s instead and run tests 2 and 3 on it with ten generics of each
ratio, then choose the fleet's motor. About 83,267 - 10,542 + 20 x 250 = 77,725 INR (derived: the
spare's Pololu motors swapped for 20 trial generics, duty on them ignored). The deploy gearmotors are
bought in India (Fab.to.Lab), so they add no import duty (domestic shipping not counted).

## One-off costs

| item | INR | status |
| --- | --- | --- |
| PETG 1 kg (about 11 cubes) | 799 | snippet, robu.in |
| PLA 1 kg (about 33 tiles at 30 g) | 1,297 | snippet, Zbotic (eSun) |
| temperature-controlled soldering station | 2,750 | snippet |
| push-pull force gauge, 50 N digital | 11,400 | snippet, imported (a luggage scale, about 300, estimate, is enough for a first pass) |
| calipers, multimeter, hand tools, pin vise, reamer, heat-set tip, kitchen scale | about 4,800 | estimate |
| gateway: ESP32-S3 dev board and cable | about 900 | estimate |
| **subtotal without a printer** | **about 21,950** (about 10,850 with a luggage scale instead of the gauge) | |
| hobby FDM printer, if none is at hand: Bambu Lab A1 mini / Creality Ender-3 V3 SE | 20,099 / 17,999 | snippet, LayerX |
| step 0, face-magnet bench calibration (ASSEMBLY.md), before the full parts order: 50 magnets (NdFeB N35, 4 x 1 mm, Justkraft NM4X1, 3 INR each) 150, about 40 g of PETG for 2 coupons and 4 test plates 32, shipping about 100; about 1.5 h of printing and 2-3 h at the bench (kitchen scale and shims from the tool list) | about 300 (not in the subtotal) | price: Justkraft listing 2026-11-17; time and shipping: guess |

No LiPo balance charger is needed: each cube charges its 1S cell over USB-C.

## Time

| step | one cube | 30 cubes (25 + spares) | basis |
| --- | --- | --- | --- |
| print (body about 3 h, two feet 2.4 h, discs and rings 1.1 h; 50 mm figures, a 60 mm body prints longer) | 6.5 h | 195 h, about 8 days on one printer running day and night; tiles 50 x 1 h more | guess (slicer not run) |
| assemble and wire | 3 h for the first, about 1.5 h after five | about 48 h | guess |
| flash, calibrate, bench test 4 (100 pivots) | 2 h | sampled: full tests on 3 cubes, a 20-pivot check on the rest, about 25 h | guess |
| lead time for parts | Pololu about 1-2 weeks to India, JLCPCB 1-2 weeks, AliExpress 2-4 weeks | the same | guess |

First pair on the bench: about 4 weeks from ordering (parts lead time dominates), about 20 hours of
hands-on work. Fleet of 25: about 2 months with one printer and one builder part-time.
