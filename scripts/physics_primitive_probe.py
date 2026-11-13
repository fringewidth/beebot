# Probe for runs/handoff/physics-primitive.md. CPU only, numpy only, about a minute.
#   counts: MCB classes grouped by the joint they serve; Fusion joint kinds per assembly and axis labels.
#   gears:  2D geometry of two voxel gears on two fixed parallel axes (cell centres): the free set of
#           (angle 1, angle 2), then gear 1 is turned and gear 2 moves only when pushed. No dynamics,
#           no friction; cells shrunk by CLEAR per side (chamfer).
#   journal: a voxel shaft in a voxel hole with no joint at all: smallest hole that lets it turn, and its play.
import collections, json, sys
import numpy as np

DATA = "/Users/hrishik/beebot/data/machine_parts"
CLEAR = 0.02
ROLE = {
    "lock": ["Conventional rivets", "Screws and bolts with hexagonal head", "Eye screws", "Hexagonal nuts",
             "Keys and keyways, splines", "Studs", "Screws and bolts with cylindrical head",
             "Screws and bolts with countersunk head", "Tapping screws", "Setscrew", "Castle nuts", "Cap nuts",
             "Threaded rods", "Washer bolt", "Clamps", "Snap rings", "Convex washer", "Lockwashers", "Flange nut",
             "Locknuts", "T-nut", "Slotted nuts", "Rivet nut", "Square nuts", "Wingnuts", "Split pins",
             "Grooved pins", "Cylindrical pins", "Taper pins", "Roll pins", "Locating pins", "Collars", "Spacers",
             "Spring washers", "Socket"],
    "turn": ["Articulations, eyelets and other articulated joints", "Bushes", "Thrust washers",
             "Flanged block bearing", "Radial contact ball bearings", "Flanged plain bearings",
             "Bearing accessories", "Lever", "Knob", "Castor", "Hinge", "Wheel", "Switch"],
    "turn, driven or driving": ["Helical geared motors", "Fan", "Impeller", "Turbine"],
    "turn + tooth or wrap contact": ["Right spur gears", "Right angular gearings", "Toothed", "Chain drives",
                                     "Pulleys"],
    "slide": ["Plain guidings"],
    "spring": ["Springs"],
    "fluid": ["Elbow fitting", "Standard fitting", "T-shape fitting", "Plugs", "Nozzle", "Valve"],
    "body": ["Plates, circulate plates", "Hook", "Square"],
}


def counts():
    by_class = collections.Counter(json.loads(l)["label"] for l in open(f"{DATA}/mcb_parts.jsonl"))
    total = sum(by_class.values())
    seen = set()
    for role, classes in ROLE.items():
        n = sum(by_class[c] for c in classes)
        seen.update(classes)
        print(json.dumps({"mcb_role": role, "parts": n, "share": round(n / total, 4), "classes": len(classes)}))
    print(json.dumps({"mcb_unassigned": sorted(set(by_class) - seen), "total": total}))

    FREE = ("revolute", "slider", "cylindrical", "planar", "ball", "pin_slot")
    tables = [json.loads(l) for l in open(f"{DATA}/fusion360_assembly_table.jsonl")]
    jointed = [t for t in tables if t["joints"]]
    cover = collections.Counter()
    free_per = []
    for t in jointed:
        kinds = {k.lower().replace("jointtype", "").replace("joint", "") for k in t["joint_kinds"]}
        free = kinds - {"rigid"}
        free_per.append(sum(v for k, v in t["joint_kinds"].items() if "rigid" not in k.lower()))
        cover["rigid only"] += not free
        cover["rigid + revolute"] += free <= {"revolute"}
        cover["rigid + revolute + cylindrical"] += free <= {"revolute", "cylindrical"}
        cover["rigid + revolute + slider"] += free <= {"revolute", "slider"}
        cover["rigid + slider"] += free <= {"slider"}
    print(json.dumps({"fusion_assemblies": len(tables), "jointed": len(jointed), "kinds_seen": sorted(
        {k for t in jointed for k in t["joint_kinds"]}), "covered_by": cover}))
    with_free = sorted(n for n in free_per if n)
    q = lambda p: with_free[int(p * (len(with_free) - 1))]
    print(json.dumps({"assemblies_with_free_joint": len(with_free), "free_joints_per_assembly_q25_50_75_95_max": [
        q(0.25), q(0.5), q(0.75), q(0.95), with_free[-1]]}))

    axes = collections.Counter()
    done = set()
    for l in open(f"{DATA}/fusion360_parts.jsonl"):
        d = json.loads(l)
        for j in d["joints"]:
            if j["kind"] not in FREE:
                continue
            key = (d["assembly"], j.get("name"), j["kind"], frozenset((d.get("occurrence_name"), j.get("partner"))))
            if key in done:
                continue
            done.add(key)
            axis = j.get("rotation_axis") or j.get("slide_direction") or "none"
            axes[(j["kind"], axis if isinstance(axis, str) else "vector")] += 1
    print(json.dumps({"free_joint_axis_labels": {f"{k[0]}:{k[1]}": v for k, v in sorted(axes.items())}}))


def samples(cells, step=0.125):
    g = np.arange(-0.5 + CLEAR, 0.5 - CLEAR + 1e-9, step)
    g = np.concatenate([g, [0.5 - CLEAR]])
    off = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    return (np.array(sorted(cells), float)[:, None] + off[None]).reshape(-1, 2)


def grid(cells):
    c = np.array(sorted(cells))
    r = int(np.abs(c).max()) + 2
    occ = np.zeros((2 * r + 1, 2 * r + 1), bool)
    occ[c[:, 0] + r, c[:, 1] + r] = True
    return occ, r


def inside(points, occ, r):
    idx = np.rint(points).astype(int)
    near = np.abs(points - idx).max(-1) <= 0.5 - CLEAR
    ok = (np.abs(idx) <= r).all(-1)
    idx = np.clip(idx, -r, r)
    return ok & near & occ[idx[..., 0] + r, idx[..., 1] + r]


def rot(points, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.stack([points[..., 0] * c - points[..., 1] * s, points[..., 0] * s + points[..., 1] * c], -1)


def free_set(a, b, centre, steps):
    # free[i, j]: gear a at angle i and gear b (axis at `centre`) at angle j do not overlap
    angles = np.arange(steps) * 2 * np.pi / steps
    pa, pb = samples(a), samples(b)
    oa, ra = grid(a)
    ob, rb = grid(b)
    centre = np.array(centre, float)
    free = np.ones((steps, steps), bool)
    for i, t1 in enumerate(angles):
        wa = rot(pa, t1) - centre
        for j, t2 in enumerate(angles):
            if inside(rot(wa, -t2), ob, rb).any():
                free[i, j] = False
            elif inside(rot(rot(pb, t2) + centre, -t1), oa, ra).any():
                free[i, j] = False
    return free


def drive(free, turns=4, reach=3):
    # gear 1 advances one step at a time; gear 2 stays unless hit, then moves to the nearest free angle
    # within `reach` steps; none there is a jam. Tried from every free start angle of gear 2.
    steps = free.shape[0]
    best = None
    for start in np.flatnonzero(free[0]):
        j, moved, jam = int(start), 0, None
        for k in range(1, turns * steps + 1):
            col = free[k % steps]
            if col[j % steps]:
                continue
            for d in range(1, reach + 1):
                hit = [s for s in (d, -d) if col[(j + s) % steps]]
                if hit:
                    j += hit[0]
                    moved += hit[0]
                    break
            else:
                jam = k
                break
        row = {"jam_at_deg": None if jam is None else round(jam * 360 / steps, 1), "ratio": round(moved / (turns * steps), 3)}
        if best is None or (row["jam_at_deg"] is None) > (best["jam_at_deg"] is None) or (
                row["jam_at_deg"] is not None and best["jam_at_deg"] is not None and row["jam_at_deg"] > best["jam_at_deg"]):
            best = row
    return best


def plus(r):
    return {(i, 0) for i in range(-r, r + 1)} | {(0, i) for i in range(-r, r + 1)}


def disc(rho):
    n = int(rho) + 1
    return {(x, y) for x in range(-n, n + 1) for y in range(-n, n + 1) if x * x + y * y <= rho * rho}


def toothed(rho, teeth):
    # voxel disc with single-cell teeth one cell out, at `teeth` evenly spaced angles
    base = disc(rho)
    out = set(base)
    for k in range(teeth):
        a = 2 * np.pi * k / teeth
        for rr in np.arange(rho, rho + 2.01, 0.25):
            cell = (int(np.rint(rr * np.cos(a))), int(np.rint(rr * np.sin(a))))
            if cell not in base:
                out.add(cell)
                break
    return out


def gears(steps=180):
    shapes = {"plus1": plus(1), "plus2": plus(2), "plus3": plus(3), "plus4": plus(4),
              "disc1.5+4": toothed(1.5, 4), "disc2.3+8": toothed(2.3, 8), "disc3.2+8": toothed(3.2, 8),
              "disc3.2+12": toothed(3.2, 12), "disc4.2+12": toothed(4.2, 12)}
    for name, cells in shapes.items():
        reach = int(np.ceil(max(np.hypot(x, y) for x, y in cells)))
        for centre in [(d, 0) for d in range(reach + 1, 2 * reach + 2)] + [(d, 1) for d in range(reach, 2 * reach + 1)]:
            free = free_set(cells, cells, centre, steps)
            if free.all() or not free.any():
                continue
            width = free.sum(1) * 360 / steps
            row = {"gear": name, "cells": len(cells), "centre": centre, "free_share": round(float(free.mean()), 3),
                   "free_deg_min": float(width.min()), **(drive(free) or {})}
            print(json.dumps(row), flush=True)


def journal():
    for rho in (0.5, 1.5, 2.5, 3.5):
        shaft = disc(rho)
        pts = samples(shaft)
        sweep = float(np.hypot(*(np.abs(np.array(sorted(shaft))) + 0.5 - CLEAR).T).max())
        for hole in np.arange(rho, rho + 3, 0.25):
            n = int(hole) + 4
            wall = {(x, y) for x in range(-n, n + 1) for y in range(-n, n + 1)} - disc(hole)
            if not wall or min(np.hypot(abs(x) - 0.5, abs(y) - 0.5) for x, y in wall if x and y) < 0:
                continue
            occ, r = grid(wall)
            angles = np.arange(0, 90, 3) * np.pi / 180
            if any(inside(rot(pts, a), occ, r).any() for a in angles):
                continue
            play = []
            for a in angles:
                p = rot(pts, a)
                for direction in ((1, 0), (0.7071, 0.7071)):
                    s = 0.0
                    while s < 3 and not inside(p + (s + 0.05) * np.array(direction), occ, r).any():
                        s += 0.05
                    play.append(s)
            ring = {c for c in wall if any((c[0] + dx, c[1] + dy) in disc(hole) for dx in (-1, 0, 1) for dy in (-1, 0, 1))}
            print(json.dumps({"shaft_rho": rho, "shaft_cells": len(shaft), "swept_radius": round(sweep, 2),
                              "hole_cells": len(disc(hole)), "ring_cells": len(ring), "play_min": round(min(play), 2),
                              "play_max": round(max(play), 2)}))
            break


if __name__ == "__main__":
    for part in sys.argv[1:] or ["counts", "journal", "gears"]:
        {"counts": counts, "gears": gears, "journal": journal}[part]()
