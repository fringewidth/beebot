"""Cube50 bill of materials: writes bom.csv and prints the per-cube cost and fleet totals used in
BOM.md and BUDGET.md. Prices looked up on 2026-11-17 (status per row); 1 USD = 88 INR.

Changed 2026-11-17 11:24 (+0530): hinge deploy drive. C1 (offset coupling) is gone; D1 is the
lead-screw gearmotor; new rows D2 (lead-screw nut), E9 (third DRV8833), S6 (sled tie rods); G1 is the
18T-18T-18T hinge train (G2 dropped); E3 has 12 switches; tiles are 60 mm. Rows in DISC_TRAIN and
PENDING_CAD wait for the CAD agent's final bought-parts list: replace them there and re-run.

Run from the repo root:  .venv/bin/python hardware/cube50/bom_build.py
"""

import csv
from pathlib import Path

HERE = Path(__file__).parent
USD = 88.0

# (id, part, exact spec, qty per cube, unit INR, status, supplier, url, note)
# status: opened = price read on the supplier page; snippet = price seen only in a search result;
# estimate = no price found, our estimate.
PARTS_HEAD = [
    ("M1", "Disc gearmotor", "Pololu micro metal gearmotor HP 6V 150:1 (#997): 210 rpm, stall 2.4 kg cm, 1.6 A at 6 V; max recommended load 2 kg cm", 2, 23.95 * USD, "opened", "Pololu", "https://www.pololu.com/product/997", "import: add shipping and duty"),
    ("M2", "Hinge gearmotor", "Pololu micro metal gearmotor HP 6V 1000:1 (#1595): 31 rpm, stall 12 kg cm (theoretical), max recommended load 2.5 kg cm", 2, 35.95 * USD, "opened", "Pololu", "https://www.pololu.com/product/1595", "import: add shipping and duty"),
    ("B1", "Disc bearing", "6700ZZ deep-groove, 10 x 15 x 4 mm", 2, 80, "estimate", "Indian bearing retail / Amazon.in", "", "agent priced 6704 (20 x 27 x 4) at 495; the design moved to 6700"),
    ("G1", "Steel spur gear m0.5 18T (hinge train)", "hardened steel, module 0.5, 18 teeth, OD 10 mm, bore 3 mm: the hinge train is 18T-18T-18T, 3 per hinge (since 2026-11-17 11:24)", 6, 180, "estimate", "AliExpress (0.5M steel spur gear, 3 mm bore)", "https://www.aliexpress.com/w/wholesale-steel-spur-gears-0.5m.html", "fallback, opened: GPM MG318T/0.5M, hardened 40Cr, 18T m0.5, bore 3.175 mm, two set screws, GBP 6.00 at rcjaz.co.uk (needs a 1/8 in shaft or a sleeve)"),
]

# Disc-train rows: gear counts and sizes change with the 60 mm CAD. Replace this list from the CAD
# agent's final bought-parts list (same tuple layout), then re-run.
DISC_TRAIN = [
    ("G3", "Steel spur gear m0.5 12T", "hardened steel, module 0.5, 12 teeth, OD 7 mm: 2 motor pinions (bore 3 mm D, 3 mm face) + 2 compound small gears (bore 3 mm, 4 mm face)", 4, 180, "estimate", "AliExpress (0.5M 45# steel gear)", "", "the 4 mm-face pair carries the joint torque: hardened, not brass"),
    ("G4", "Steel spur gear m0.5 30T", "hardened steel, module 0.5, 30 teeth, OD 16 mm, bore 3 mm, 3 mm face (compound big gear, pinned to its 12T on a 3 mm shaft)", 2, 220, "estimate", "AliExpress", "", ""),
    ("G5", "Steel spur gear m0.5 32T", "hardened steel, module 0.5, 32 teeth, OD 17 mm, bore 4 mm, 4 mm face (centre gear, cross-pinned to the centre shaft)", 2, 240, "estimate", "AliExpress", "", ""),
]

PARTS_REST = [
    ("P1", "Lip rings", "0.5 mm stainless 304 laser-cut rings, 27 mm OD, 11 mm ID, with the four slots (6 per cube: 4 passive faces, 2 discs with pin arcs); DXF from the CAD", 6, 40, "estimate", "JLCPCB sheet metal or a local laser cutter", "", "dropped into the print at a pause; the pin heads bear on steel, not PETG"),
    ("S1", "Hinge axle", "3 mm carbon-steel dowel, 3 x 50 mm, cut to the CAD length (one per hinge)", 2, 35 / 2, "snippet", "Moglix (CE 3x50 mm carbon steel solid dowel pin, pack of 2, 35 incl. tax)", "https://www.moglix.com/ce-3x50mm-carbon-steel-solid-dowel-pin-pack-of-2/mp/msn8580e1xw492", "not hardened; Moglix shows two prices for this pin"),
    ("S6", "Sled tie rod", "3 mm steel rod, one per sled, cut from the same 3 x 50 mm dowel (length from the CAD)", 2, 35 / 2, "snippet", "Moglix (as S1)", "https://www.moglix.com/ce-3x50mm-carbon-steel-solid-dowel-pin-pack-of-2/mp/msn8580e1xw492", ""),
    ("P2", "Hinge carriers", "1.2 mm stainless 304 laser-cut plates, 2 per hinge (since 2026-11-17); may become part of the sled in the 60 mm CAD", 4, 30, "estimate", "JLCPCB sheet metal or a local laser cutter (with P1)", "", ""),
    ("D1", "Deploy gearmotor", "DFRobot FIT0577 micro metal gearmotor with lead screw: 6 V, 98 rpm free (+/-10 %), 24 mA free, 185 mA at 550 g cm, stall 420 mA, locking torque 1000 g cm, 1:150, M3 x 20 mm threaded output shaft, N20 body 12 x 10 mm, about 10 g; no nut", 2, 878.12, "opened", "Fab.to.Lab (India)", "https://www.fabtolab.com/accessories/dfrobot-micro-metal-dc-geared-motor-lead-screw-6v-98rpm", "GST not stated; DFRobot USD 9.00 (opened, dfrobot.com/product-1733.html); cheaper India option: Probots GA12-N20 100 rpm M3 x 55 at 499 incl. GST (opened; 0.2 kg cm, stall not stated)"),
    ("D2", "Lead-screw nut", "M3 brass hex nut (DIN 934, 5.5 mm AF, 2.4 mm), captive in each sled; check the screw's lead on arrival (0.5 mm assumed)", 2, 3, "estimate", "local", "", ""),
    ("S2", "Centre shaft", "4 mm silver steel rod, 12.4 mm long, dog cross-pinned 1.5 mm", 2, 20, "estimate", "local", "", ""),
    ("S3", "Pins", "2 mm steel pins: 2 hinge idler pins 7 mm, 2 disc lock pins 6 mm; 3 mm rod: 2 compound shafts 8 mm, 2 stow bolts 9.4 mm", 1, 80, "estimate", "local / cut from rod", "", "hardened 2 x 10 dowels not found in India"),
    ("S4", "Pin screws", "M2 x 8 pan head (ISO 1580, 4 mm head) + M2 nut, stainless: the four bond pins of each disc", 8, 3, "estimate", "buysupplies.in or local", "", ""),
    ("S5", "Screws and inserts", "M2 x 6/8/12 pan head (about 14) + M2 x 4 brass heat-set inserts (about 12)", 1, 14 * 1 + 12 * 19, "snippet", "buysupplies.in; cart2india (Hitefu inserts)", "https://www.cart2india.com/nut-inserts/hitefu-m2-heat-set-insert-100-pcs-threaded-inserts-for-3d-printing-brass-metric-knurled-nuts-for-plastic-parts/00000000004393336756", "screws from a 1000-pack at 736"),
    ("K1", "Springs", "about 6 per cube from an assortment: 2 bolt return, 2 over-centre ring springs, 2 lock-pin springs", 6, 2457 / 115, "snippet", "Ubuy India (Waltool 115 pcs)", "https://www.ubuy.co.in/product/QDT3KLUOW-115pcs-21-sizes-torsional-springs-and-mini-compression-spring-assortment-kit-45-90-120-degree-stainless-steel-mechanical-small-torsion-spring-for", "the two ring springs may need custom forming"),
    ("A1", "SMA wire", "Flexinol 0.15 mm (0.006 in), 90 C: 4 x 85 mm ring wires + 2 x 50 mm bolt wires + crimp allowance = 0.6 m", 0.6, 8.20 * USD / 1.0, "snippet", "Images SI (Flexinol)", "https://www.imagesco.com/nitinol/wire.html", "per metre; import"),
    ("A2", "SMA crimps", "brass crimp ferrules or M1.6 clamp screws, 12 per cube", 12, 3, "estimate", "local", "", ""),
    ("E1", "Encoder magnets", "6 x 2.5 mm diametric NdFeB (AS5600)", 4, 60, "estimate", "AliExpress", "", "no Indian listing found"),
    ("E2", "Disc contacts", "sprung contacts, 5 per foot (Mill-Max 0906 through-hole or an SMD spring finger)", 10, 54.88, "snippet", "DigiKey India", "https://www.digikey.in/en/products/detail/mill-max-manufacturing-corp/0906-0-15-20-76-14-11-0/1147048", "1 mm tip not confirmed"),
    ("E3", "Tact switches", "3 x 4 x 2 mm SMD: 4 ring end switches, 2 foot-seated, 2 bolt, 4 sled end (out and in per hinge)", 12, 6, "snippet", "Evelta", "https://evelta.com/tactile-switch-3x4x2mm-smd-2-pin-with-column-button", ""),
    ("E4", "Cell", "1S LiPo pouch 302535 (3 x 25 x 35 mm), about 220 mAh, >= 5C, JST-PH 2.0 lead", 1, 250, "estimate", "Indian retail (robu.in / electronicscomp)", "", "the design packs this size; a 2S pack does not fit"),
    ("E5", "Main board parts", "LCSC: ESP32-S3-MINI-1-N8 4.0, 2x DRV8833PWPR 1.1 (the third is E9), MCP23017-E/SS 1.4, 6x AO3400A 0.48, AO3401A 0.05, LSM6DS3TR-C 0.9, TPS61088 boost 1.3, 2.2 uH 6 A inductor 0.35, TP4056 0.1, DW01A + FS8205A 0.15, AP2112K-3.3 0.1, USB-C 0.16, passives and connectors 1.5 (USD)", 1, 11.6 * USD, "snippet", "LCSC via JLCPCB", "https://jlcpcb.com/help/article/pcb-assembly-price", "TPS61088, inductor, TP4056, DW01A, AP2112K, passives: estimate"),
    ("E6", "Encoder chips", "AS5600-ASOM on four small boards (two per foot side)", 4, 0.64 * USD, "snippet", "LCSC C79815", "", "16 in stock when seen"),
    ("E7", "PCBs", "main 40 x 33 mm 4-layer + 2 foot ring PCBs + 2 foot boards + 4 encoder boards, one panel, per cube share at 10 sets", 1, 3.0 * USD, "estimate", "JLCPCB", "https://jlcpcb.com/help/article/pcb-assembly-price", "2-layer 5 pcs is $2 (snippet); 4-layer and panel estimated"),
    ("E8", "Flex and wire", "2x 6-pin 0.5 mm FFC 50 mm (13 each) + 4 FFC connectors (15 each) + 30 AWG silicone wire and JST-SH (about 60)", 1, 2 * 13 + 4 * 15 + 60, "snippet", "Hubtronics, Zbotic, Probots", "https://hubtronics.in/6p-0.5mm-same-side-ffc-cable-50mm", "FFC connector price estimated"),
    ("E9", "Deploy driver", "third DRV8833PWPR (LCSC C50506, TSSOP-16-EP, about 0.64 USD) + 2 x 0.43 R 0805 sense resistors (0.47 A chop = motor stall)", 1, 0.68 * USD, "snippet", "LCSC via JLCPCB", "https://www.lcsc.com/product-detail/C50506.html", "resistors estimated"),
    ("F1", "Printed parts", "PETG: body 31 g, 2 feet 27 g, 2 discs 5 g, 2 cam rings 1 g, plus 40 % supports, brim and failures = about 90 g", 0.09, 799, "snippet", "robu.in PETG 1 kg", "https://st.robu.in/?p=857923", "per kg"),
]

# Per-order costs that do not scale per cube (INR)
PCBA_ORDER = (8.18 + 1.53 + 15 * 3.07) * USD  # JLC economic setup + stencil + about 15 extended part types (opened)
SHIPPING_ORDER = {"JLCPCB/LCSC": 25 * USD, "Pololu": 30 * USD, "AliExpress": 8 * USD}  # estimate per order
DUTY = 0.30  # import duty and GST on imported parts, estimate
IMPORTED = {"M1", "M2", "G1", "G3", "G4", "G5", "P1", "A1", "E1", "E5", "E6", "E7", "E9"}
PARTS = PARTS_HEAD + DISC_TRAIN + PARTS_REST
PENDING_CAD = {r[0] for r in DISC_TRAIN} | {"B1", "S3", "P2", "F1"}  # specs or counts may change with the 60 mm CAD

GENERIC_MOTORS = {"M1": 250.0, "M2": 266.0}  # N20 6 V generics (snippet, electronicscomp/zbotic); specs unverified

TILE = ("T1", "Floor tile (passive face, chamfered edges)", "PLA, about 60 x 60 x 10 mm, about 30 g at 30 % infill", 30 / 1000 * 1297)


def per_cube(generic=False):
    total, imported = 0.0, 0.0
    for pid, _, _, qty, unit, *_ in PARTS:
        u = GENERIC_MOTORS.get(pid, unit) if generic else unit
        cost = qty * u
        total += cost
        if pid in IMPORTED and not (generic and pid in GENERIC_MOTORS):
            imported += cost
    return total, imported


def fleet(n, generic=False, spares=0.2, tiles_per_cube=2):
    base, imported = per_cube(generic)
    built = n + max(1, round(n * spares)) if n > 1 else n
    parts = base * built
    duty = imported * built * DUTY
    orders = PCBA_ORDER + sum(SHIPPING_ORDER.values()) - (0 if not generic else SHIPPING_ORDER["Pololu"])
    tiles = TILE[3] * tiles_per_cube * n
    return built, parts + duty + orders + tiles


def main():
    rows = []
    for pid, part, spec, qty, unit, status, supplier, url, note in PARTS:
        rows.append({"id": pid, "part": part, "spec": spec, "qty_per_cube": qty, "unit_inr": round(unit, 1), "unit_usd": round(unit / USD, 2),
                     "line_inr": round(qty * unit, 1), "price_status": status, "looked_up": "2026-11-17", "supplier": supplier, "url": url,
                     "note": (note + "; " if note else "") + "pending the CAD bought-parts list" if pid in PENDING_CAD else note})
    with (HERE / "bom.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for generic in (False, True):
        total, imported = per_cube(generic)
        label = "generic N20 motors" if generic else "Pololu motors"
        print(f"per cube, parts only ({label}): {total:,.0f} INR = {total / USD:,.0f} USD; imported share {imported:,.0f} INR")
        for n in (1, 2, 14, 25):
            built, cost = fleet(n, generic)
            print(f"  {n:>2} cubes (+spares -> {built} built): {cost:,.0f} INR = {cost / USD:,.0f} USD")
    groups = {"motors (M1, M2, D1)": {"M1", "M2", "D1"}, "gears": {r[0] for r in PARTS if r[0].startswith("G")}, "deploy drive (D1, D2, E9, S6)": {"D1", "D2", "E9", "S6"}}
    for name, ids in groups.items():
        print(f"  {name}: {sum(r[3] * r[4] for r in PARTS if r[0] in ids):,.0f} INR")
    print(f"  tile {TILE[3]:,.0f} INR each")
    print(f"per-order fixed: PCBA {PCBA_ORDER:,.0f} INR, shipping {sum(SHIPPING_ORDER.values()):,.0f} INR; duty {DUTY:.0%} on imported parts")


if __name__ == "__main__":
    main()
