"""What cube structures can carry, and what the joints would need for more (hardware/joint_strength).

For basic structural parts built from 5 cm cubes (cantilevers, simply supported and fixed-end
beams, columns under a side load, slabs on legs, gusseted brackets, braced and unbraced frames,
corbelled arches) this finds, with the validated check:

1. the largest part each hardware option holds at 1 g with a margin (load factor >= --margin);
2. the holding force per 5 cm face (N) a part of a given size needs, by bisection on the joints'
   pull (magnets or latches, scaled together with the floor's).

Hardware options are joint_strength.Joints variants: stronger switchable magnets, a latch floor,
mechanical face latches, shear keys (interlocking: friction raised), and combinations.

    .venv/bin/python scripts/joint_requirements.py --out runs/joint_strength/requirements.json
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from joint_validate import arch, bracket, span_fixed, span_pillars  # noqa: E402

from beebot.hardware.cube import G  # noqa: E402
from beebot.hardware.cube_swarm import DESIGN  # noqa: E402
from beebot.hardware.joint_strength import Joints, check_many  # noqa: E402

CUBE = DESIGN.cube()
WEIGHT_N = CUBE.mass * G  # one module weight in newtons


def cantilever(n, d=1, payload=0.0):
    """A wall-anchored cantilever n long and d deep (1 wide); payload module weights at the tip."""
    cells = [(z, 0, x) for z in range(1, d + 1) for x in range(n)]
    load = np.zeros((len(cells), 3))
    if payload:
        load[cells.index((d, 0, n - 1)), 2] = -payload
    return {"cells": cells, "anchored": [(z, 0, -1) for z in range(d + 1)], "load": load}


def beam_on_pillars(w, d=1, h=2):
    """A beam d deep across w between two free-standing pillars h high (pillars as deep as the
    beam's ends need)."""
    top = h - 1
    cells = [(z, 0, x) for z in range(top + d) for x in (-1, w)]
    cells += [(z, 0, x) for z in range(top, top + d) for x in range(w)]
    return {"cells": sorted(set(cells))}


def tower(h, side=1.0):
    load = np.zeros((h, 3))
    load[-1, 0] = side
    return {"cells": [(z, 0, 0) for z in range(h)], "load": load, "mode": "live"}


def slab(w, legs=2):
    cells = [(z, y, x) for z in range(legs) for y in (0, w - 1) for x in (0, w - 1)]
    cells += [(legs, y, x) for y in range(w) for x in range(w)]
    return {"cells": cells}


def frame(h, w, brace=False):
    """A portal frame: two columns h high, a beam on top w between them, a side load of 1 module
    weight at the top corner. brace: a stepped diagonal (face-connected stair, two cubes a step)
    from one column's foot to the other's top."""
    cells = {(z, 0, x) for z in range(h) for x in (0, w + 1)} | {(h - 1, 0, x) for x in range(w + 2)}
    if brace:
        for k in range(1, min(h - 1, w + 1)):
            cells |= {(k, 0, k), (k - 1, 0, k)}
    cells = sorted(cells)
    load = np.zeros((len(cells), 3))
    load[cells.index((h - 1, 0, 0)), 0] = 1.0
    return {"cells": cells, "load": load, "mode": "live"}


PARTS = {
    "cantilever d=1 (shelf bracket, awning)": (lambda n: cantilever(n, 1), range(1, 41)),
    "cantilever d=2": (lambda n: cantilever(n, 2), range(1, 41)),
    "cantilever d=3": (lambda n: cantilever(n, 3), range(1, 41)),
    "cantilever d=1, tip load 5 W (0.75 kg)": (lambda n: cantilever(n, 1, 5.0), range(1, 41)),
    "bracket: arm with a depth-3 stepped gusset (arm length)": (lambda n: bracket(3, n=n, h=4), range(3, 41)),
    "beam on free pillars, d=1": (lambda w: span_pillars(w), range(1, 41)),
    "beam on free pillars, d=2": (lambda w: beam_on_pillars(w, 2), range(1, 41)),
    "beam between fixed walls, d=1": (lambda w: span_fixed(w), range(1, 41)),
    "column, side load 1 W at the top (height)": (lambda h: tower(h), range(2, 41)),
    "slab on 4 corner legs (side)": (lambda w: slab(w), range(3, 16, 2)),
    "portal frame h=6, side load 1 W (width)": (lambda w: frame(6, w), range(1, 21)),
    "braced portal frame h=6, side load 1 W (width)": (lambda w: frame(6, w, brace=True), range(1, 21)),
    "corbelled arch (steps)": (lambda k: arch(k), range(2, 11)),
}

OPTIONS = {
    "as built (hinge cube, steel floor)": Joints(),
    "EPM faces x1.7 (SMORES-EP 88 N a face)": Joints(edge_pull=4.5 * 1.7, face_pull=0.3 * 1.7),
    "latch floor (floor holds like a face)": Joints(floor_pull=1.0, terrain_pull=1.0),
    "shear keys (friction 1.5)": Joints(mu=1.5),
    "face latches x5 (about 270 N a face)": Joints(edge_pull=4.5 * 5, face_pull=0.3 * 5, floor_pull=0.3 / 5 * 1.0),
    "face latches x5 + latch floor": Joints(edge_pull=4.5 * 5, face_pull=0.3 * 5, floor_pull=1.0, terrain_pull=1.0),
    "rigid lattice: latches x20 + latch floor + keys": Joints(edge_pull=4.5 * 20, face_pull=0.3 * 20, floor_pull=1.0, terrain_pull=1.0, mu=1.5),
}


def face_newtons(joints):
    return (8 * joints.edge_pull + joints.face_pull) * WEIGHT_N


def largest(make, sizes, joints, margin):
    """The largest size that holds at factor >= margin with every smaller size holding too
    (sizes in increasing order, solved in groups of four until the first that fails), and the
    factors seen."""
    best, factors = None, []
    for start in range(0, len(sizes), 4):
        group = sizes[start : start + 4]
        results = check_many([make(s) for s in group], joints)
        for s, r in zip(group, results):
            factors.append(r.factor)
            if r.factor < margin:
                return (best if r.converged else f"{best}+"), factors  # +: the next size's factor is only a lower bound
            best = s
    return best, factors


def needed_pull(spec, margin, base=Joints(), floor_with_faces=True, lo=0.01, hi=200.0):
    """The smallest multiplier k on the cube faces' pull (the floor's with it when floor_with_faces;
    otherwise the floor keeps base's absolute pull) at which spec holds at factor >= margin; None above hi."""

    def ok(k):
        j = replace(base, edge_pull=base.edge_pull * k, face_pull=base.face_pull * k, floor_pull=base.floor_pull if floor_with_faces else base.floor_pull / k, terrain_pull=base.terrain_pull if floor_with_faces else base.terrain_pull / k, cap=max(base.cap, 2 * margin))
        return check_many([spec], j)[0].factor >= margin

    if not ok(hi):
        return None
    if ok(lo):
        return lo
    for _ in range(30):
        mid = (lo * hi) ** 0.5
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
        if hi / lo < 1.01:
            break
    return hi


REQUIRED = [
    ("shelf: cantilever 30 cm (6), tip load 13 W (2 kg)", cantilever(6, 1, 13.0)),
    ("awning: cantilever 1 m (20), d=1", cantilever(20, 1)),
    ("awning: cantilever 1 m (20), d=2", cantilever(20, 2)),
    ("balcony: cantilever 2 m (40), d=3", cantilever(40, 3)),
    ("beam: 1 m (20) on free pillars, d=1", span_pillars(20)),
    ("beam: 2 m (40) on free pillars, d=2", beam_on_pillars(40, 2)),
    ("column: 1 m (20), side load 1 W at top", tower(20)),
    ("column: 1 m (20), side load 5 W at top", tower(20, 5.0)),
    ("table: 50 cm (10) slab on 4 legs", slab(10)),
    ("table: 75 cm (15) slab on 4 legs", slab(15)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=float, default=1.5, help="load factor a part must reach (1.5: margin for moves, wear and jitter)")
    parser.add_argument("--out", default="runs/joint_strength/requirements.json")
    args = parser.parse_args()
    report = {"margin": args.margin, "weight_N": WEIGHT_N, "face_N": {k: round(face_newtons(j), 1) for k, j in OPTIONS.items()}, "largest": {}, "required": {}}
    for part, (make, sizes) in PARTS.items():
        row = {}
        for name, joints in OPTIONS.items():
            best, factors = largest(make, list(sizes), joints, args.margin)
            row[name] = best
        report["largest"][part] = row
        print(part, json.dumps(row), flush=True)
    for name, spec in REQUIRED:
        k_steel = needed_pull(spec, args.margin, floor_with_faces=False)  # the steel floor stays as built
        k_latch = needed_pull(spec, args.margin, Joints(floor_pull=1.0, terrain_pull=1.0))  # the floor latches as strongly as a face
        report["required"][name] = {
            "faces only (steel floor as built): x as built": None if k_steel is None else round(k_steel, 2),
            "faces only (steel floor as built): N a face": None if k_steel is None else round(k_steel * face_newtons(Joints()), 0),
            "faces and floor latch alike: x as built": None if k_latch is None else round(k_latch, 2),
            "faces and floor latch alike: N a face": None if k_latch is None else round(k_latch * face_newtons(Joints()), 0),
        }
        print(name, json.dumps(report["required"][name]), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
