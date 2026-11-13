# Function rigs on sampled part variants: span (ends on free one-cube pillars, 1 W at midspan top) and
# cantilever (one end slice welded to a wall, 1 W at the free end's top). Gravity mode: the factor scales
# self-weight and the load together.
import json, random, sys
import numpy as np, torch
torch.set_num_threads(4)
from beebot.hardware.joint_strength import Joints, check_many
OPTS = {
    "as_built": Joints(),
    "latch_floor": Joints(floor_pull=1.0, terrain_pull=1.0),
    "x5_latch_floor": Joints(edge_pull=4.5 * 5, face_pull=0.3 * 5, floor_pull=1.0, terrain_pull=1.0),
    "x20_keys": Joints(edge_pull=4.5 * 20, face_pull=0.3 * 20, floor_pull=1.0, terrain_pull=1.0, mu=1.5),
}
def rigs(c):
    c = c - c.min(0)
    ext = c.max(0)
    ax = 1 if ext[1] >= ext[2] else 2
    lo, hi = c[:, ax].min(), c[:, ax].max()
    lifted = c + np.array([1, 0, 0])
    ends = lifted[((lifted[:, ax] == lo) | (lifted[:, ax] == hi)) & (lifted[:, 0] == 1)]
    pillars = ends - np.array([1, 0, 0])
    cells = np.concatenate([pillars, lifted])
    mid = (lo + hi) // 2
    col = lifted[lifted[:, ax] == mid]
    top = col[col[:, 0].argmax()]
    load = np.zeros((len(cells), 3)); load[len(pillars) + int(np.where((lifted == top).all(1))[0][0]), 2] = -1.0
    span = {"cells": cells, "load": load}
    root = c[c[:, ax] == lo]
    free = c[c[:, ax] != lo] + np.array([1, 0, 0])
    anch = root + np.array([1, 0, 0])
    tip = free[free[:, ax] == free[:, ax].max()]
    tip = tip[tip[:, 0].argmax()]
    load2 = np.zeros((len(free), 3)); load2[int(np.where((free == tip).all(1))[0][0]), 2] = -1.0
    cant = {"cells": free, "anchored": anch, "load": load2}
    return span, cant, int(hi - lo + 1)
rng = random.Random(1)
out = open(sys.argv[1], "w")
for src in ("mcb", "fusion360"):
    z = np.load(f"/Users/hrishik/beebot/data/machine_parts/{src}.npz")
    cells, starts = z["cells"], z["starts"]
    by_part = {}
    for i, l in enumerate(open(f"/Users/hrishik/beebot/data/machine_parts/{src}.jsonl")):
        d = json.loads(l)
        if 40 <= d["cells"] <= 300 and d["extra"].get("orientation") == "flat" and d.get("rot", 0) == 0:
            by_part.setdefault(d["source_id"], []).append((i, d))
    parts = sorted(by_part); rng.shuffle(parts)
    pick = [rng.choice(by_part[p]) for p in parts[:300]]
    built = [rigs(cells[starts[i]:starts[i + 1]].astype(int)) for i, _ in pick]
    for name, j in OPTS.items():
        rs = check_many([b[0] for b in built], joints=j)
        rc = check_many([b[1] for b in built], joints=j)
        for (i, d), b, a, c in zip(pick, built, rs, rc):
            out.write(json.dumps({"src": src, "label": d.get("label"), "shape": d["extra"].get("shape_class"), "thick": d["extra"].get("thickness_cells"), "cells": d["cells"], "length": b[2], "opt": name, "span": float(a.factor), "cant": float(c.factor), "conv": bool(a.converged and c.converged)}) + "\n")
        out.flush(); print(src, name, flush=True)
