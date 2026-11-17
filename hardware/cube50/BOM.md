# Cube50 bill of materials

**Changed 2026-11-17 11:24 (+0530): hinge deploy drive.** The offset coupling (C1) is gone; D1 is now
the lead-screw gearmotor that slides each hinge sled; new rows D2, E9 and S6; the hinge train is
18T-18T-18T in m0.5 (G1, replacing the m0.4 G1 and G2); E3 has 12 switches; tiles are about 60 mm. The
disc-train rows (G3-G5) and B1, S3, P2, F1 wait for the CAD agent's final bought-parts list for the
60 mm cube (marked "pending CAD").

Prices looked up on **2026-11-17**; 1 USD = 88 INR. `bom.csv` holds the same rows (generated with
the totals by `bom_build.py`). Status: **opened** = price read on the supplier's page; **snippet** =
price seen only in a search result (the page did not open or blocked fetching); **estimate** = no
price found, our estimate. Indian prices often exclude 18 % GST where the listing says so. Nothing
was ordered.

## Purchased parts, per cube

| id | part | exact spec | qty | unit INR | unit USD | status | supplier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | disc gearmotor | Pololu micro metal gearmotor HP 6V 150:1 (#997): 210 rpm free, stall 2.4 kg cm and 1.6 A at 6 V, max recommended load 2 kg cm, 10 x 12 mm, 3 mm D shaft | 2 | 2,108 | 23.95 | opened | [Pololu](https://www.pololu.com/product/997) |
| M2 | hinge gearmotor | Pololu micro metal gearmotor HP 6V 1000:1 (#1595): 31 rpm, stall 12 kg cm (theoretical) and 1.6 A, max recommended 2.5 kg cm | 2 | 3,164 | 35.95 | opened | [Pololu](https://www.pololu.com/product/1595) |
| B1 | disc bearing (pending CAD) | 6700ZZ, 10 x 15 x 4 mm | 2 | 80 | 0.91 | estimate | Indian bearing shops / Amazon.in (only 6704, 20 x 27 x 4, was priced: 495, [snippet](https://www.thebikeaffair.com/products/tripeak-6704-high-precision-steel-bearing-abec3-20x27x4mm)) |
| G1 | steel spur gear, hinge train (since 2026-11-17 11:24) | hardened steel, module 0.5, 18 T, OD 10 mm, bore 3 mm; the hinge train is 18T-18T-18T, 3 per hinge | 6 | 180 | 2.05 | estimate | AliExpress ["steel spur gears 0.5M"](https://www.aliexpress.com/w/wholesale-steel-spur-gears-0.5m.html) (listings mix steel, aluminium and plastic: confirm steel and hardening with the seller). Fallback, opened: [GPM MG318T/0.5M](https://www.rcjaz.co.uk/mg318t05m-18t-40cr-motor-pinion-for-05m-module-3175mm-bore-by-gpm-p-90085762.html), hardened 40Cr, bore 3.175 mm, two set screws, GBP 6.00 |
| G3 | steel spur gear (pending CAD) | hardened steel, module 0.5, 12 T, OD 7 mm: 2 motor pinions (bore 3 mm D, 3 mm face) and 2 compound small gears (bore 3 mm, 4 mm face) | 4 | 180 | 2.05 | estimate | AliExpress (0.5M 45# steel gear) |
| G4 | steel spur gear (pending CAD) | hardened steel, module 0.5, 30 T, OD 16 mm, bore 3 mm, 3 mm face (compound big gear, pinned to its 12T on a 3 mm shaft) | 2 | 220 | 2.50 | estimate | AliExpress |
| G5 | steel spur gear (pending CAD) | hardened steel, module 0.5, 32 T, OD 17 mm, bore 4 mm, 4 mm face (centre gear, cross-pinned to the shaft) | 2 | 240 | 2.73 | estimate | AliExpress |
| P1 | lip rings | 0.5 mm stainless 304, laser cut, 27 mm OD, 11 mm ID: 4 passive (`cad/bought/lip_ring_passive.dxf`) and 2 disc (`lip_ring_disc.dxf`) | 6 | 40 | 0.45 | estimate | JLCPCB sheet metal or a local laser cutter |
| S1 | hinge axle (since 2026-11-17 11:24) | 3 mm carbon-steel dowel, 3 x 50 mm, cut to the CAD length; one per hinge | 2 | 18 | 0.20 | snippet | [Moglix CE 3x50 mm dowel, pack of 2, 35 incl. tax](https://www.moglix.com/ce-3x50mm-carbon-steel-solid-dowel-pin-pack-of-2/mp/msn8580e1xw492) (not hardened; a second Moglix page shows a different price) |
| S6 | sled tie rod (since 2026-11-17 11:24) | 3 mm steel rod, one per sled, cut from the same 3 x 50 mm dowel | 2 | 18 | 0.20 | snippet | as S1 |
| P2 | hinge carriers (pending CAD) | 1.2 mm stainless 304, laser-cut, 2 per hinge; may become part of the sled in the 60 mm CAD | 4 | 30 | 0.34 | estimate | with P1 |
| D1 | deploy gearmotor (since 2026-11-17 11:24) | **DFRobot FIT0577**, N20 micro metal gearmotor with lead screw: 6 V; 98 rpm free (+/-10 %), 24 mA free; 60 rpm and 185 mA at its 550 g cm rated load; stall 420 mA (+/-10 %), locking torque 1000 g cm (0.098 N m); 1:150; output shaft an M3 x 20 mm threaded rod; N20 body, gearbox section 12 x 10 mm; about 10 g; nut not included | 2 | 878 | 9.98 | opened | [Fab.to.Lab](https://www.fabtolab.com/accessories/dfrobot-micro-metal-dc-geared-motor-lead-screw-6v-98rpm) (India, in stock; GST not stated). Maker: [DFRobot](https://www.dfrobot.com/product-1733.html), USD 9.00 (opened) |
| D2 | lead-screw nut (since 2026-11-17 11:24) | M3 brass hex nut, DIN 934 (5.5 mm AF, 2.4 mm), captive in each sled | 2 | 3 | 0.03 | estimate | local. The screw's lead is assumed 0.5 mm (M3 coarse); DFRobot's "1.1 mm/s" implies 0.7 mm: check on arrival before buying nuts |
| S2 | centre shaft | 4 mm silver steel, 12.4 mm long, 1.5 mm cross pin for the dog | 2 | 20 | 0.23 | estimate | local |
| S3 | pins (pending CAD) | 2 mm steel rod: 2 hinge idler pins 7 mm, 2 lock pins 6 mm; 3 mm rod: 2 compound shafts 8 mm, 2 stow bolts 9.4 mm | 1 set | 80 | 0.91 | estimate | local (hardened 2 x 10 dowels not found in India) |
| S4 | bond pins | M2 x 8 pan head (ISO 1580, 4 mm head) and M2 nut, stainless | 8 | 3 | 0.03 | estimate | buysupplies.in or local |
| S5 | screws, inserts | about 14 M2 pan-head screws (6/8/12 mm) + 12 M2 x 4 brass heat-set inserts | 1 set | 242 | 2.75 | snippet | screws [buysupplies.in](https://buysupplies.in/products/m2-304-stainless-steel-csk-phillips-screws) (1,000 for 736), inserts [cart2india](https://www.cart2india.com/nut-inserts/hitefu-m2-heat-set-insert-100-pcs-threaded-inserts-for-3d-printing-brass-metric-knurled-nuts-for-plastic-parts/00000000004393336756) (100 for 1,900) |
| K1 | springs | 2 bolt return (compression, about 3 mm OD), 2 over-centre ring springs, 2 lock-pin springs | 6 | 21 | 0.24 | snippet | [Ubuy India assortment](https://www.ubuy.co.in/product/QDT3KLUOW-115pcs-21-sizes-torsional-springs-and-mini-compression-spring-assortment-kit-45-90-120-degree-stainless-steel-mechanical-small-torsion-spring-for), 115 for 2,457 (the ring springs may need forming from wire) |
| A1 | SMA wire | Flexinol 0.15 mm (0.006 in), 90 C transition: 4 x 85 mm + 2 x 50 mm + crimp ends = 0.6 m | 0.6 m | 722 / m | 8.20 / m | snippet | [Images SI](https://www.imagesco.com/nitinol/wire.html) (US; import). Pull and current from memory: about 3 N at about 0.4 A, check the datasheet |
| A2 | SMA crimps | brass crimp ferrules (or M1.6 clamp screws) | 12 | 3 | 0.03 | estimate | local |
| E1 | encoder magnets | 6 x 2.5 mm diametric NdFeB | 4 | 60 | 0.68 | estimate | AliExpress (no Indian listing found) |
| E2 | disc contacts | sprung contacts, 5 per foot (Mill-Max 0906 or an SMD spring finger) | 10 | 55 | 0.62 | snippet | [DigiKey India](https://www.digikey.in/en/products/detail/mill-max-manufacturing-corp/0906-0-15-20-76-14-11-0/1147048) (1 mm tip not confirmed) |
| E3 | switches | 3 x 4 x 2 mm SMD tact: 4 ring end, 2 foot seated, 2 bolt up, 4 sled end (out and in per hinge, since 2026-11-17 11:24) | 12 | 6 | 0.07 | snippet | [Evelta](https://evelta.com/tactile-switch-3x4x2mm-smd-2-pin-with-column-button) |
| E4 | cell | 1S LiPo pouch 302535 (3 x 25 x 35 mm), about 220 mAh, **high-rate: >= 2.5 A continuous (11C)**, JST-PH 2.0 | 1 | 250 | 2.84 | estimate; **open part: no stock high-rate cell of this size found yet** | Indian retail or AliExpress (micro-drone / vape high-discharge pouches). A 2S pack (CNHL 350 mAh 2S, 538, snippet) does not fit the core |
| E5 | main board parts | ESP32-S3-MINI-1-N8 (C2913206, about 4.0 USD), 2 x DRV8833PWPR (C50506, 0.55; the third is E9), MCP23017-E/SS (C506653, 1.4), 6 x AO3400A (C20917, 0.08), AO3401A (C15127, 0.05), LSM6DS3TR-C (C967633, 0.9), USB-C 16P (C165948, 0.16); TPS61088 boost, 2.2 uH 6 A inductor, TP4056, DW01A + FS8205A, AP2112K-3.3, passives | 1 set | 1,021 | 11.60 | snippet (LCSC parts listed with numbers); estimate (the rest) | [LCSC via JLCPCB](https://jlcpcb.com/help/article/pcb-assembly-price) |
| E6 | encoder chips | AS5600-ASOM (C79815), on 4 small boards | 4 | 56 | 0.64 | snippet | LCSC (16 in stock when seen) |
| E7 | PCBs | main 33 x 26 mm 4-layer + 2 foot boards + 2 disc ring PCBs + 4 encoder boards, one panel; per-cube share at 10 sets | 1 set | 264 | 3.00 | estimate | JLCPCB (2-layer 5 pcs is 2 USD, snippet) |
| E8 | flex and wire | 2 x 6-pin 0.5 mm FFC 50 mm ([Hubtronics](https://hubtronics.in/6p-0.5mm-same-side-ffc-cable-50mm), 13), 4 FFC sockets (15, estimate), 30 AWG silicone wire and JST-SH (about 60) | 1 set | 146 | 1.66 | snippet | Hubtronics, [Zbotic](https://zbotic.in/product/30awg-extrasoft-silicone-wire-black-length5m/), [Probots](https://probots.co.in/jst-sh1-0mm-assorted-connector-kit-2-5-pin-male-female-housings-200pcs-with-crimp-pins.html) |
| E9 | deploy driver (since 2026-11-17 11:24) | third DRV8833PWPR (LCSC C50506, TSSOP-16-EP, about 0.64 USD at 1 piece) and 2 x 0.43 R 0805 sense resistors (0.47 A chop, the motor's stall current) | 1 set | 60 | 0.68 | snippet (DRV8833 in a search result; resistors estimate) | [LCSC](https://www.lcsc.com/product-detail/C50506.html) via JLCPCB (extended part, already counted in the PCBA fee) |
| F1 | filament (pending CAD) | PETG, about 90 g a cube (62 g of parts + supports, brim, failures) | 0.09 kg | 799 / kg | 9.08 / kg | snippet | [robu.in PETG](https://st.robu.in/?p=857923) |
| T1 | floor tiles | PLA, about 60 mm, 30 g each (since 2026-11-17 11:24), 2 per cube in a fleet's floor | 2 | 39 | 0.44 | derived from PLA at 1,297 / kg ([snippet](https://zbotic.in/?p=670008)) | printed |

**Per cube (parts only): 19,496 INR = 222 USD with Pololu motors; 9,986 INR = 113 USD with generic
N20s** (derived, `bom_build.py`, 2026-11-17 11:24; the disc-train rows are still the 50 mm ones). The
four Pololu motors are 10,542 INR, 54 % of the cube; the deploy drive (D1, D2, E9, S6) is 1,857 INR.

### Deploy gearmotor: why FIT0577, and the alternatives

Looked up 2026-11-17. The brief asked for a listing with a stated stall torque, about 50-200 rpm at
6 V, India first.

| listing | rpm at 6 V | torque, current | screw | price | status |
| --- | --- | --- | --- | --- | --- |
| **DFRobot FIT0577** at [Fab.to.Lab](https://www.fabtolab.com/accessories/dfrobot-micro-metal-dc-geared-motor-lead-screw-6v-98rpm) | 98 free, 60 at 550 g cm | locking 1000 g cm, stall 420 mA, 1:150 | M3 x 20, no nut | 878.12 INR (9.98 USD) | opened; the only India listing found with stall torque and current |
| Probots GA12-N20-M3*55, [100 rpm](https://probots.co.in/100-rpm-6v-linear-actuator-n20-micro-metal-gear-dc-motor-with-m3-screw-lead-shaft.html) or [60 rpm](https://probots.co.in/60-rpm-6v-linear-actuator-n20-micro-metal-gear-dc-motor-with-m3-screw-lead-shaft.html) | 100 / 60 | "0.2 kg cm" (rated or stall not said), no current | M3 x 55, no nut (cut to length) | 499 INR incl. GST (5.67 USD) | opened, in stock; cheaper, specs thin |
| DFRobot FIT0577 at [DigiKey India](https://www.digikey.in/en/products/detail/dfrobot/FIT0577/9490122) | as above | as above | as above | 988.67 INR (241+ tier only), not stocked, 9 weeks | opened |
| AliExpress ["micro lead screw motor"](https://www.aliexpress.us/w/wholesale-micro-lead-screw-motor.html) (N20, M3 x 55 screw, 15-1000 rpm options) | per option | not verified | M3 x 55 | about 3-4 USD (estimate) | search page only; no single listing verified |

robu.in, electronicscomp and zbotic list plain N20s whose "screw size M3" is the mounting thread, not a
lead screw (snippets). **Speed:** a 7.9 mm stroke is about 16 turns of an M3 screw, so a 98 rpm motor
takes about 10 s per stroke and a released pivot gains about 20 s. A 300-500 rpm version of the same
motor would take 2-3 s and still give tens of newtons of thrust (derived from a guessed 0.2-0.3 kg cm
stall at that ratio), if pivot time matters more than margin.

Generic alternative to M1/M2 (specs unverified, **bench test before buying in quantity**): N20 6 V
about 200 rpm (1.2 kg cm stall claimed: too weak for the disc) or about 100 rpm, and N20 6 V 30 rpm
(10 kg cm stall claimed), about 225-265 INR each: [electronicscomp](https://www.electronicscomp.com/n20-6v-30-rpm-micro-metal-gear-motor)
(opened, out of stock), [Zbotic](https://zbotic.in/product/n20-6v-200-rpm-micro-metal-gear-box-dc-motor/) (snippet).
The torque budgets in DESIGN.md hold only for Pololu's figures.

## Per order (not per cube)

| item | INR | basis |
| --- | --- | --- |
| JLCPCB economic PCBA: setup 8.18 USD + stencil 1.53 + about 15 extended part types x 3.07 | 4,907 | opened ([page](https://jlcpcb.com/help/article/pcb-assembly-price), updated 2026-10-22), part count estimate |
| shipping: JLCPCB/LCSC 25 USD, Pololu 30 USD, AliExpress 8 USD | 5,544 | estimate |
| import duty and GST on imported parts | 30 % of their value | estimate |

## Gateway (one per fleet)

| part | INR | status |
| --- | --- | --- |
| ESP32-S3 dev board (DevKitC-1 or a clone) and a USB-C cable | about 900 | estimate |

## Not verified

1. The deploy screw's lead (0.5 mm assumed) and FIT0577's body length (dimensions only in a drawing); the 18T m0.5 hinge gears with a 3 mm bore in hardened steel (no single listing verified).
1. Every gear price, and that hardened-steel m0.5 gears with 4 mm faces are sold (brass m0.8 gears were not found, which is why every gear is m0.5; the disc trains must be steel, DESIGN.md).
1. 6700ZZ price, 6 x 2.5 diametric magnets in India, a high-rate 1S 302535 cell, Flexinol's pull and current from a page, the FFC socket.
1. LCSC prices are search-result values, not live quotes; the ESP32-S3-MINI showed 3.26 and 4.70 USD in two places.
1. Pololu motor mass (not on the product pages; 9.5 g for the 10:1 HP from a snippet).
