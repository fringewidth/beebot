"""Envelope model (not a CAD design) of the 7 mm reaction-wheel cube RW-P from
scripts/module_phys/scaling.py: boxes and cylinders at the sizes the scaling model picked, placed
to check they fit together. Run: .venv/bin/python hardware/module_phys/rw_cube_envelope.py
Writes hardware/module_phys/rw_cube_7mm_envelope.step and prints overlaps (should be none)."""

from pathlib import Path

import cadquery as cq

L = 7.0  # mm
WALL = 0.40  # flex 0.11 + resin frame 0.3 (scaling.wall)
IN = L - 2 * WALL  # 6.2
H = IN / 2
SPHERE_D = 1.0  # N52 sphere, model asks 0.92
COVER = 0.10
ROTOR_R, ROTOR_T = 2.1, 0.75
GAP = 0.10

parts = {}
# shell as a thin box frame (outer minus inner)
parts["shell"] = cq.Workplane().box(L, L, L).cut(cq.Workplane().box(IN, IN, IN))
# 8 corner spheres in pockets: centre inset COVER + r from each outer face
c = L / 2 - COVER - SPHERE_D / 2
for i, (sx, sy, sz) in enumerate([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]):
    parts[f"sphere{i}"] = cq.Workplane().sphere(SPHERE_D / 2).translate((sx * c, sy * c, sz * c))
# rotor X: disc near +x face (axis x), rotor Y near +y face (axis y); stators are the face flex
parts["rotorX"] = cq.Workplane("YZ").circle(ROTOR_R).extrude(ROTOR_T).translate((H - GAP - ROTOR_T, 0, 0))
parts["rotorY"] = cq.Workplane("XZ").circle(ROTOR_R).extrude(-ROTOR_T).translate((0, H - GAP - ROTOR_T, 0))
# storage: three CeraCharge 4.5 x 3.2 x 1.1 stacked flat on the -z face, clear of the rotors (x, y < 2.25) and corner spheres
for k in range(3):
    parts[f"cera{k}"] = cq.Workplane().box(4.4, 3.2, 1.1).translate((0.0, -0.7, -H + 0.55 + 1.12 * k))
# electronics as boxes standing on the face flex: MCU (QFN-20 3x3x0.8) on -x, IMU (2x2x0.7) on -y,
# boost + FETs + diodes (3x2x0.6 block) on +z; their pads are on the flex, so they sit in the wall slab
parts["mcu"] = cq.Workplane().box(0.8, 3.0, 3.0).translate((-H + 0.4, 0.0, 0.0))
parts["imu"] = cq.Workplane().box(2.0, 0.7, 2.0).translate((0.0, -H + 0.35, 1.5))
parts["power"] = cq.Workplane().box(3.0, 2.0, 0.6).translate((-0.5, 0.0, H - 0.3))
# pawls: 0.6 mm cubes beside each rotor rim
parts["pawlX"] = cq.Workplane().box(0.6, 0.6, 0.6).translate((H - GAP - ROTOR_T / 2, 0, ROTOR_R + 0.5))
parts["pawlY"] = cq.Workplane().box(0.6, 0.6, 0.6).translate((ROTOR_R + 0.5, H - GAP - ROTOR_T / 2, 0))

names = list(parts)
overlaps = []
for i, a in enumerate(names):
    for b in names[i + 1:]:
        if a == "shell" or b == "shell":
            continue
        v = parts[a].intersect(parts[b]).val().Volume() if parts[a].intersect(parts[b]).vals() else 0.0
        if v > 1e-4:
            overlaps.append((a, b, round(v, 4)))
inside = []
box = cq.Workplane().box(IN, IN, IN)
for n in names:
    if n == "shell" or n.startswith("sphere"):
        continue
    out = parts[n].cut(box)
    v = out.val().Volume() if out.vals() else 0.0
    if v > 1e-4:
        inside.append((n, round(v, 4)))
vol = {n: round(parts[n].val().Volume(), 3) for n in names if n != "shell"}
print("overlaps:", overlaps)
print("outside the inner cavity:", inside)
print("part volumes mm3:", vol, "sum", round(sum(vol.values()), 2), "of inner", round(IN**3, 1))
assy = cq.Assembly()
for n, w in parts.items():
    assy.add(w, name=n)
out = Path(__file__).with_name("rw_cube_7mm_envelope.step")
assy.save(str(out))
print("wrote", out)
