"""Cube50 in the repo's MuJoCo models: does this design's mass and torque still do the moves?

Run from the repo root (CPU, a few minutes):
    nice -n 10 .venv/bin/python hardware/cube50/sim/cube50_sim.py [pivot] [walker] [arm]
Writes hardware/cube50/sim/results.jsonl (one line per check) and prints a summary.

What is reused and what is changed (nothing in hardware/src is edited; constants are patched in
this process only):
1. Pivots: module_c's World and Planner (a torque about a latched edge, PD on a minimum-jerk path,
   welds for bonds). Patched to Cube50's mass (183 g from the CAD), its hinge torque limit
   (0.11 N m at the foot: the retractable hinge's smaller last gear, 0.245 x 12/24 x 0.95^2) and a
   90 deg/s reference; also at 0.11 / 1.3. module_c's cube has a
   sprung wheel on -x; here every cube rests on +y so the wheel never touches. module_c lets any
   edge be the pivot; Cube50 pivots only about its two hinge edges, so these runs check torque,
   mass and landing, not which edge is available.
2. Joints: module_b's Rig with the walker and arm of hardware/scripts/module_demos.py. Patched to
   200 g modules and Cube50's disc drive: a 1.13 N m cap (the firmware's 1.30 A limit, under Pololu's
   2 kg cm gearbox rating, through the 6.67:1 steel train) and a speed-torque line, 1.415 N m at
   stall falling to zero at 3.30 rad/s. Each load case also runs with the cap divided by 1.3 (margin). module_b's geometry is kept (discs with proud rims, 52.4 mm pitch); Cube50's
   discs are recessed, so the feet differ. Video and the demos' results file are switched off.
3. Hinge knuckles (checks named pivot_knuckle_<variant>, walker_knuckle_<variant>, arm_knuckle_<variant>):
   the knuckle and drive-coupling cylinders of the retractable hinge (new) and of the earlier fixed
   hinge (old) are added as colliding geoms, so a knuckle that hits a bonded partner or the floor
   shows up as contacts. Pivots (module_c): every cube's box becomes a convex mesh of the 7.5 mm
   chamfered cube (same mass; module_c's 42 mm disc collides as a 35 mm disc so that it stays inside
   the chamfered outline), the floor is a field of 50 mm V-groove tiles (top edges chamfered 7.5 mm,
   tops at z = 0, the plane moved to z = -10 mm), a floor roll is latched to the world at its edge
   (connect equalities, standing in for the bonded foot and the hinge), and each cube carries
   deployed-knuckle geoms on all 12 edges, off except the mover's at the pivot edge while a pivot
   runs (they collide with the support and the floor although module_c turns mover-support contacts
   off). Joints (module_b): the stowed (new) or old knuckles sit in a fixed child body of each
   module's body, so the build-time neighbour exclusions do not cover them; parent filtering is
   switched off so that a partner bonded on a disc collides with them (no other contact pair
   changes: every parent-child pair in module_b is excluded explicitly). Knuckle contacts (steps,
   deepest penetration, largest normal force, partners) and the knuckle overlaps at the build pose
   that the contact filters hide (same rigid body or excluded) are reported.

    nice -n 10 .venv/bin/python hardware/cube50/sim/cube50_sim.py               # everything
    nice -n 10 .venv/bin/python hardware/cube50/sim/cube50_sim.py knuckles      # the knuckle checks only
    args: pivot walker arm pivot_knuckle pivot_old walker_knuckle arm_knuckle knuckles walker_stowed arm_stowed quadruped
    magnets: the pivots at 60 mm with the face magnets (sim/magnets.py, params.json "magnets"); size,
    chamfer, mass, CoM, inertia, hinge torque and rate from params.json; no / bonded / free side
    neighbours, at the hinge torque and at it / 1.5; one case:
    magnets:<none|bonded|free>:<torque>:<seeds>[:off][:skin=<mm>]. Writes pivot_magnets_* rows.
    (the new-design variants are new2_*, the diagonal retract; new_* is the first version)
"""

import contextlib
import inspect
import json
import math
import re
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "hardware" / "src"))
sys.path.insert(0, str(ROOT / "hardware" / "scripts"))
OUT = Path(__file__).parent / "results.jsonl"

MASS = 0.200  # kg, CAD (hardware/cube50/cad/report.txt: 200.4 g)
HINGE_TORQUE = 0.11  # N m at the foot: retractable hinge, 0.245 N m x 12/24 x 0.95^2 = 0.1106
HINGE_RATE = 90.0  # deg/s
DISC_HOLD = 1.13  # N m: firmware current limit 1.30 A through 6.67:1 (DESIGN.md)
DISC_STALL = 1.415  # N m at 6 V
DISC_FREE = 3.30  # rad/s at 6 V
MARGIN = 1.3  # the load cases must also pass with the cap divided by this


def emit(row):
    print(json.dumps(row), flush=True)
    with OUT.open("a") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------- hinge knuckles (geometry)
# Cylinders per hinge edge in the cube50 frame (discs on +-z, hinges on +x, edges at x = +25 mm,
# z = +-25 mm along y), mm: (radius, offset of the axis from the edge line inward along the hinge
# face normal (x), the same along the disc normal (z), y from, y to), y measured toward the train
# side (+y for the z = +25 hinge, -y for z = -25).
KNUCKLE = {
    # retractable hinge, second version: the stowed axle retracts diagonally, fully inside the cell
    "new2_deployed": [(2.8, 0.0, 0.0, -15.6, 15.6), (4.59, 1.4, 1.55, 15.8, 17.4), (2.8, 2.8, 3.1, 17.6, 19.4)],  # axle on the edge line; coupling sweep; last gear stays stowed
    "new2_stowed": [(2.8, 2.8, 3.1, -15.6, 15.6), (2.8, 2.8, 3.1, 15.8, 17.4), (2.8, 2.8, 3.1, 17.6, 19.4)],  # nothing beyond x = 25 or z = 25
    # retractable hinge, first version (stowed parts stand 2.8 mm beyond the +x face)
    "new_deployed": [(2.8, 0.0, 0.0, -15.6, 15.6), (4.05, 0.0, 1.55, 15.8, 17.6)],
    "new_stowed": [(2.8, 0.0, 3.1, -15.6, 15.6), (2.8, 0.0, 3.1, 15.8, 19.4)],
    "old": [(4.8, 0.0, 0.0, -24.4, 24.4), (5.0, 0.0, 0.0, 18.1, 21.3)],  # knuckle and gear tip on the edge line, always out
}
CHAMFER = 0.0075  # m, every cube edge and the tiles' top edges
TILE = (0.050, 0.010)  # m: tile side and thickness
KNUCKLE_SCALE = 1.0  # KNUCKLE is the 50 mm design; the 60 mm runs scale it by 1.2 (guess until the 60 mm CAD)
INERT = 4  # conaffinity of a switched-off knuckle: matches no contype, but keeps the geom in the collision
# structures (MuJoCo drops a geom compiled with contype = conaffinity = 0, and setting them later does nothing)


def edge_cylinders(variant, k, s_disc, s_hinge, half=0.025):
    """(from, to, radius) in m of the knuckle cylinders on the edge that runs along axis k at
    coordinate s_disc * half on axis (k + 1) % 3 (the disc normal's role) and s_hinge * half on axis
    (k + 2) % 3 (the hinge face's role). In the cube50 frame (k = y, disc z, hinge x) this is the
    design; module_c uses the same rule for all 12 edges of its cubes."""
    a, b = (k + 1) % 3, (k + 2) % 3
    train = s_disc * s_hinge  # cube50: +y at z = +25, -y at z = -25
    out = []
    for radius, hinge_in, disc_in, y0, y1 in KNUCKLE[variant]:
        p = np.zeros(3)
        u = KNUCKLE_SCALE / 1000
        p[a] = s_disc * (half - disc_in * u)
        p[b] = s_hinge * (half - hinge_in * u)
        p1, p2 = p.copy(), p.copy()
        p1[k], p2[k] = train * y0 * u, train * y1 * u
        out.append((p1, p2, radius * u))
    return out


def _vertices(points):
    return " ".join(f"{v:.6g}" for p in points for v in p)


def chamfered_cube(half=0.025, chamfer=None):
    """Vertices of a cube with every edge chamfered (the convex hull of the 24 points)."""
    chamfer = CHAMFER if chamfer is None else chamfer
    inner = half - chamfer
    pts = []
    for k in range(3):
        for s in (-1, 1):
            for t1 in (-1, 1):
                for t2 in (-1, 1):
                    p = [0.0, 0.0, 0.0]
                    p[k], p[(k + 1) % 3], p[(k + 2) % 3] = s * half, t1 * inner, t2 * inner
                    pts.append(p)
    return pts


def v_tile(side=None, thick=None, chamfer=None):
    """Vertices of a floor tile, top at z = 0, its four top edges chamfered."""
    side, thick, chamfer = side or TILE[0], thick or TILE[1], CHAMFER if chamfer is None else chamfer
    h, inner = side / 2, side / 2 - chamfer
    return [(x * s, y * s, z) for z, s in ((0.0, inner), (-chamfer, h), (-thick, h)) for x in (-1, 1) for y in (-1, 1)]


class KnuckleLog:
    """Contacts of knuckle geoms, per label: steps with any contact, deepest penetration, largest
    normal force, and the steps each partner was touched."""

    def __init__(self):
        self.rows = {}
        self.label = "all"
        self.f6 = np.zeros(6)

    def scan(self, m, d, ids):
        n = d.ncon
        if not n:
            return
        g = np.asarray(d.contact.geom)
        hit = np.isin(g, ids)
        rows = np.nonzero(hit.any(axis=1))[0]
        if not len(rows):
            return
        rec = self.rows.setdefault(self.label, {"steps": 0, "pen_mm": 0.0, "force_n": 0.0, "partners": {}})
        rec["steps"] += 1
        seen = set()
        for r in rows:
            mujoco.mj_contactForce(m, d, int(r), self.f6)
            rec["pen_mm"] = max(rec["pen_mm"], -float(d.contact.dist[r]) * 1000)
            rec["force_n"] = max(rec["force_n"], float(self.f6[0]))
            mine, other = (int(g[r, 0]), int(g[r, 1])) if hit[r, 0] else (int(g[r, 1]), int(g[r, 0]))
            b = m.geom_bodyid[other]
            seen.add(re.sub(r"k.*", "k", m.geom(mine).name) + ">" + (m.body(b).name if b else re.sub(r"_.*", "", m.geom(other).name or "world")))
        for name in seen:
            rec["partners"][name] = rec["partners"].get(name, 0) + 1

    def summary(self):
        out = {}
        for label, rec in self.rows.items():
            out[label] = {"steps_with_contact": rec["steps"], "max_penetration_mm": round(rec["pen_mm"], 3), "max_force_n": round(rec["force_n"], 2), "partners": rec["partners"]}
        return out


# ---------------------------------------------------------------- pivots (module_c)
@contextlib.contextmanager
def _knuckle_c(variant):
    """module_c with chamfered mesh cubes on V-groove tiles, floor rolls latched to the world and
    `variant` knuckle geoms on every edge (off unless that edge is a pivot in progress). Restores
    module_c on exit."""
    from beebot.hardware import module_c as mc

    saved = {k: getattr(mc, k) for k in ("build", "_cube_xml")}
    saved_cls = [(c, k, c.__dict__[k]) for c, k in ((mc.World, "__init__"), (mc.World, "pivot"), (mc.Pivot, "__init__"), (mc.Pivot, "release"))]
    log = KnuckleLog()
    cube_xml, build = mc._cube_xml, mc.build

    def knuckles(i):
        out = ""
        for k in range(3):
            for sd in (-1, 1):
                for sh in (-1, 1):
                    for j, (p1, p2, r) in enumerate(edge_cylinders(variant, k, sd, sh, mc.HALF)):
                        name = f"c{i}k{k}{'p' if sd > 0 else 'n'}{'p' if sh > 0 else 'n'}{j}"
                        out += f'<geom name="{name}" type="cylinder" fromto="{_vertices([p1, p2])}" size="{r:.6g}" contype="0" conaffinity="{INERT}" mass="0" rgba="0.9 0.2 0.2 0.6" friction="{mc.BODY_FRICTION} 0.005 0.0001"/>\n'
        return out

    def patched_cube_xml(i, p):
        x = cube_xml(i, p)
        x = re.sub(r'(<geom name="c\d+box" )type="box" size="[^"]*"', r'\1type="mesh" mesh="cube_ch"', x)
        x = re.sub(r'(<geom name="c\d+disc" type="cylinder" size=")[^ ]+', rf"\g<1>{mc.HALF - CHAMFER:.6g}", x)
        return x.replace(f'<body name="c{i}disc"', knuckles(i) + f'<body name="c{i}disc"', 1)

    def patched_build(placements, scene="", timestep=mc.DT, welds=None, latches=4):
        x = build(placements, scene, timestep, welds, latches + 2)
        meshes = f'<mesh name="cube_ch" vertex="{_vertices(chamfered_cube(mc.HALF))}"/>\n<mesh name="tile" vertex="{_vertices(v_tile())}"/>\n'
        tiles = "".join(f'<geom name="tile_{a}_{b}" type="mesh" mesh="tile" pos="{a * TILE[0]:.6g} {b * TILE[0]:.6g} 0" rgba="0.70 0.70 0.68 1" friction="{mc.FLOOR_FRICTION} 0.005 0.0001"/>\n' for a in range(-2, 4) for b in range(-2, 5))
        x = x.replace("<asset>\n", "<asset>\n" + meshes, 1)
        x = x.replace('<geom name="floor" type="plane"', f'<geom name="floor" type="plane" pos="0 0 {-TILE[1]:.6g}"', 1)
        return re.sub(r'(<geom name="floor"[^\n]*\n)', lambda hit: hit[1] + tiles, x, count=1)

    def world_init(self, placements, *a, **k):
        saved_cls[0][2](self, placements, *a, **k)
        m = self.model
        self.knuckle_geoms = {}
        for g in range(m.ngeom):
            name = m.geom(g).name
            hit = re.fullmatch(r"c(\d+)k(\d)([pn])([pn])\d", name or "")
            if hit:
                key = (int(hit[1]), int(hit[2]), 1 if hit[3] == "p" else -1, 1 if hit[4] == "p" else -1)
                self.knuckle_geoms.setdefault(key, []).append(g)
        self.knuckle_ids = np.array(sorted(g for v in self.knuckle_geoms.values() for g in v))
        self.watch = lambda w: log.scan(w.model, w.data, w.knuckle_ids)

    def pivot_init(self, world, mover, point, axis, goal, support, tolerance=mc.TOLERANCE):
        saved_cls[2][2](self, world, mover, point, axis, goal, support, tolerance)
        if not self.latched:
            return
        m, d = world.model, world.data
        bm = world.body[mover]
        x_m, R_m = d.xpos[bm].copy(), d.xmat[bm].reshape(3, 3).copy()
        if support is None:  # floor roll: the edge latched to the world (the bonded foot and the hinge)
            self.floor_slots = world.free_latches(2)
            for slot, end in zip(self.floor_slots, (-0.35, 0.35)):
                anchor = self.edge + end * mc.SIDE * self.axis
                m.eq_obj1id[slot], m.eq_obj2id[slot] = bm, 0
                m.eq_data[slot][:3] = R_m.T @ (anchor - x_m)
                m.eq_data[slot][3:6] = anchor
                d.eq_active[slot] = 1
        local = R_m.T @ (self.edge - x_m)
        k = int(np.argmin(np.abs(local)))
        key = (mover, k, int(np.sign(local[(k + 1) % 3])), int(np.sign(local[(k + 2) % 3])))
        for g in world.knuckle_geoms[key]:  # on before World.pivot copies contype, so its restores keep it on
            m.geom_contype[g] = m.geom_conaffinity[g] = 1

    def pivot_release(self):
        saved_cls[3][2](self)
        for slot in getattr(self, "floor_slots", []):
            self.world.data.eq_active[slot] = 0
            self.world.latch_busy -= {slot}
        self.floor_slots = []

    def world_pivot(self, mover, *a, **k):
        try:
            return saved_cls[1][2](self, mover, *a, **k)
        finally:  # retracted after the pivot's settle
            for (i, *_), ids in self.knuckle_geoms.items():
                if i == mover:
                    self.model.geom_contype[ids] = 0
                    self.model.geom_conaffinity[ids] = INERT

    mc._cube_xml, mc.build = patched_cube_xml, patched_build
    mc.World.__init__, mc.World.pivot, mc.Pivot.__init__, mc.Pivot.release = world_init, world_pivot, pivot_init, pivot_release
    try:
        yield mc, log
    finally:
        for k, v in saved.items():
            setattr(mc, k, v)
        for c, k, v in saved_cls:
            setattr(c, k, v)


def pivots(seeds=10, torque=HINGE_TORQUE, knuckle=None):
    """knuckle None: module_c as it is (check "pivot"). knuckle "new2_deployed" or "old": the
    chamfered cubes on V-groove tiles with that knuckle at the pivot edge (check
    "pivot_knuckle_<variant>")."""
    from beebot.hardware import module_c as mc

    mc.WHEEL_MASS, mc.CARRIAGE_MASS, mc.DISC_MASS = 0.001, 0.001, 0.006
    mc.BODY_MASS = MASS - mc.WHEEL_MASS - mc.CARRIAGE_MASS - mc.DISC_MASS
    mc.MASS = MASS
    mc.PIVOT_TORQUE, mc.PIVOT_RATE = torque, HINGE_RATE
    out = {"roll": [], "climb_up": [], "climb_down": [], "peaks": [], "landing_mm": [], "outcomes": []}
    t = time.time()
    ctx = _knuckle_c(knuckle) if knuckle else contextlib.nullcontext((mc, None))
    with ctx as (mc, log):
        def label(name):
            if log is not None:
                log.label = name

        for seed in range(seeds):
            label("settle")
            lone = mc.World([mc.Placement((0.0, 0.0), down="+y")], seed=seed)
            lone.run(0.2)
            start = lone.pos(0)
            label("floor_roll")
            r = lone.floor_roll(0, [1, 0, 0])
            out["roll"].append(r == "arrived" and abs(lone.pos(0)[0] - start[0] - mc.SIDE) < 0.003)
            out["peaks"] += [p[1] for p in lone.pivot_peaks]
            label("settle")
            pls = [mc.Placement((0.0, k * (mc.SIDE + 0.0005)), down="+y") for k in range(3)]
            w = mc.World(pls, seed=seed, bonds=[(0, 1, None), (1, 2, None)])
            planner = mc.Planner(w)
            w.run(0.3)
            label("climb_up")
            up = planner.climb(2, 1)
            out["climb_up"].append(up == "arrived" and abs(w.pos(2)[2] - w.pos(1)[2] - mc.SIDE) < 0.003)
            out["landing_mm"].append(abs(w.pos(2)[2] - w.pos(1)[2] - mc.SIDE) * 1000)
            label("climb_down")
            down = planner.climb_down(2, 1, w.pos(1) - w.pos(0))
            out["climb_down"].append(down == "arrived" and abs(w.pos(2)[2] - w.pos(1)[2]) < 0.003)
            out["peaks"] += [p[1] for p in w.pivot_peaks]
            out["outcomes"].append([r, up, down])
    row = {
        "check": "pivot" if knuckle is None else f"pivot_knuckle_{knuckle}",
        "mass_kg": MASS,
        "torque_nm": torque,
        "rate_deg_s": HINGE_RATE,
        "seeds": seeds,
        "floor_roll": f"{sum(out['roll'])}/{seeds}",
        "climb_up": f"{sum(out['climb_up'])}/{seeds}",
        "climb_down": f"{sum(out['climb_down'])}/{seeds}",
        "peak_torque_nm": round(max(out["peaks"]), 4),
        "landing_error_mm_max": round(max(out["landing_mm"]), 2),
        "gravity_torque_nm_derived": round(MASS * 9.81 * mc.SIDE / math.sqrt(2), 4),
    }
    if knuckle is not None:
        row.update({"floor": "V-groove tiles, plane at -10 mm", "cube": "chamfered 7.5 mm mesh", "floor_roll_latch": "world",
                    "knuckle_contacts": log.summary(), "outcomes": out["outcomes"], "wall_s": round(time.time() - t, 1)})
    emit(row)
    return row


# ---------------------------------------------------------------- face magnets in the pivots (60 mm)
P60 = json.loads((Path(__file__).parent / "params.json").read_text())  # the 60 mm design: read at import, so a rerun picks up new numbers
SIZE60 = {"SIDE": P60["size_m"], "CHAMFER": P60["edge_chamfer_m"], "TILE": (P60["size_m"], 0.012), "KNUCKLE_SCALE": 1.0}
MASS60 = P60["mass_kg"]
HINGE60 = P60["pivot_torque_nm"]
RATE60 = P60["pivot_rate_deg_s"]
INERTIA60 = P60["inertia_about_centre_kg_m2"]  # cube50 frame (x hinge face, z discs)
COM60 = P60["centre_of_mass_m"]
_h = P60["hinge"]
KNUCKLE["sled_deployed"] = [(_h["parts_radius_m"] * 1000, 0.0, 0.0, _h["parts_span_y_m"][0] * 1000, _h["parts_span_y_m"][1] * 1000)]  # axle, knuckle and train all ride the sled onto the edge line
Q50 = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])  # cube50 frame -> module_c frame (disc +z -> +x, hinge +x -> +y)


@contextlib.contextmanager
def _size(size):
    """module_c and this file's geometry at another cube size (restored on exit)."""
    from beebot.hardware import module_c as mc

    g = globals()
    saved = {k: g[k] for k in ("CHAMFER", "TILE", "KNUCKLE_SCALE")}
    saved_mc = {k: getattr(mc, k) for k in ("SIDE", "HALF")}
    g.update({k: v for k, v in size.items() if k in saved})
    mc.SIDE, mc.HALF = size["SIDE"], size["SIDE"] / 2
    try:
        yield mc
    finally:
        g.update(saved)
        for k, v in saved_mc.items():
            setattr(mc, k, v)


class MagnetField:
    """Exact face-magnet (and AS5600-magnet) wrenches between every pair of cubes, added to
    xfrc_applied before every mj_step. A pair whose relative pose has moved less than 0.5 um and
    1e-5 rad since its last evaluation reuses that evaluation (moved with the pair); pairs bonded by
    a weld are skipped (their magnetic load is internal to the welded pair). Also logs the magnetic
    torque on a pivoting mover about its pivot axis, the stray field at every AS5600 and the moves
    of the cubes that are not pivoting. While a pivot runs, the mover keeps colliding with every
    cube (module_c switches its contacts with the support and edge neighbours off, which lets the
    magnets pull the faces through each other)."""

    def __init__(self, mc, skin=None):
        import magnets as mgn

        self.mgn, self.mc = mgn, mc
        lay = mgn.layout()
        if skin is not None:
            lay["skin"] = skin / 1000
        self.skin_mm = lay["skin"] * 1000
        self.local = mgn.cube_set(lay).moved(Q50, np.zeros(3))
        self.body = mgn.Body(self.local)
        self.chips = {k: (Q50 @ c, Q50 @ n, m) for k, (c, n, m) in mgn.chips(lay).items()}
        self.cache = {}
        self.world = None
        self.label = "all"
        self.rows = {}
        self.evals = self.reuses = 0
        self.tick = 0

    def attach(self, world):
        self.world, self.cache, self.tick = world, {}, 0
        d = world.data
        self.start = {i: (d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()) for i, b in enumerate(world.body)}

    def row(self):
        return self.rows.setdefault(self.label, {"resist_nm": 0.0, "assist_nm": 0.0, "mover_force_n": 0.0, "stray_h1_mt": {}, "stray_mean_mt": {}, "moved_mm": {}, "tilt_deg": {}})

    def step(self, m, d):
        w = self.world
        if w is None:
            return
        n = w.n
        R = [d.xmat[b].reshape(3, 3).copy() for b in w.body]
        X = [d.xpos[b].copy() for b in w.body]
        C = [d.xipos[b].copy() for b in w.body]
        F = np.zeros((n, 3))
        T = np.zeros((n, 3))
        welded = {k for k in w.bonds}
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(X[i] - X[j]) > 1.8 * self.mc.SIDE or frozenset((i, j)) in welded:
                    continue
                Rij = R[i].T @ R[j]
                xij = R[i].T @ (X[j] - X[i])
                hit = self.cache.get((i, j))
                if hit is not None and np.linalg.norm(hit[1] - xij) < 5e-7 and np.linalg.norm(hit[0] - Rij) < 1e-5:
                    Fj, Tj = R[i] @ hit[2], R[i] @ hit[3]  # in i's frame, torque about j's centre of mass
                    self.reuses += 1
                else:
                    src = self.local.moved(R[i], X[i])
                    Fj, Tj = self.body.wrench_from(src, R[j], X[j], C[j])
                    self.cache[(i, j)] = (Rij, xij, R[i].T @ Fj, R[i].T @ Tj)
                    self.evals += 1
                F[j] += Fj
                T[j] += Tj
                F[i] -= Fj
                T[i] -= Tj + np.cross(C[j] - C[i], Fj)
        d.xfrc_applied[:, :3] = 0.0
        for i, b in enumerate(w.body):
            d.xfrc_applied[b, :3] += F[i]
            d.xfrc_applied[b, 3:] += T[i]
        rec = self.row()
        for p in w.pivots:
            if p.phase in ("refused", "released"):
                continue
            for j in range(n):  # module_c turns the mover's contacts with the support (and edge neighbours) off
                if j != p.mover:  # during a pivot; with magnets pulling the faces together they must stay on
                    m.geom_contype[w.geoms[j]] = 1
                    m.geom_conaffinity[w.geoms[j]] = 3
            _, R_s = p.frames()
            axis = R_s @ p.axis_s
            k = w.body.index(w.body[p.mover])
            tq = float((T[k] + np.cross(C[k] - p.edge, F[k])) @ axis) * math.copysign(1.0, p.goal)
            rec["resist_nm"] = max(rec["resist_nm"], -tq)
            rec["assist_nm"] = max(rec["assist_nm"], tq)
            rec["mover_force_n"] = max(rec["mover_force_n"], float(np.linalg.norm(F[k])))
            busy = {p.mover}
            for i in range(n):  # neighbours: how far each has moved from its start of this world
                if i in busy:
                    continue
                x0, R0 = self.start[i]
                rec["moved_mm"][i] = round(max(rec["moved_mm"].get(i, 0.0), float(np.linalg.norm(X[i] - x0)) * 1000), 3)
                ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R0.T @ R[i]) - 1) / 2))))
                rec["tilt_deg"][i] = round(max(rec["tilt_deg"].get(i, 0.0), ang), 3)
        self.tick += 1
        if self.tick % 20 == 0:
            self.stray(R, X, rec)

    def stray(self, R, X, rec):
        """First harmonic of Bz round each AS5600's 1 mm Hall circle from the other cubes' magnets."""
        w = self.world
        for i in range(w.n):
            near = [j for j in range(w.n) if j != i and np.linalg.norm(X[j] - X[i]) < 1.8 * self.mc.SIDE]
            if not near:
                continue
            src = self.local.moved(R[near[0]], X[near[0]])
            for j in near[1:]:
                src = src + self.local.moved(R[j], X[j])
            for name, (c, nrm, _) in self.chips.items():
                h1, _, mean = self.mgn.as5600_error(lambda p: self.mgn.field(src, p), R[i] @ c + X[i], R[i] @ nrm)
                key = name
                rec["stray_h1_mt"][key] = round(max(rec["stray_h1_mt"].get(key, 0.0), h1 * 1000), 4)
                rec["stray_mean_mt"][key] = round(max(rec["stray_mean_mt"].get(key, 0.0), abs(mean) * 1000), 4)


@contextlib.contextmanager
def _magnets_c(mc, field):
    """Route module_c's mj_step through `field` (magnet wrenches every step) and attach it to each
    new World. Restores module_c on exit."""
    real_mujoco = mc.mujoco
    init = mc.World.__init__

    class _Proxy:
        def __getattr__(self, name):
            return getattr(real_mujoco, name)

        @staticmethod
        def mj_step(m, d):
            field.step(m, d)
            real_mujoco.mj_step(m, d)

    def world_init(self, *a, **k):
        init(self, *a, **k)
        field.attach(self)

    mc.mujoco, mc.World.__init__ = _Proxy(), world_init
    try:
        yield
    finally:
        mc.mujoco, mc.World.__init__ = real_mujoco, init


@contextlib.contextmanager
def _inertial60(mc, mass):
    """Give every cube body the 60 mm design's centre of mass and inertia (params.json), mapped
    into module_c's frame; the disc, carriage and wheel children keep their small masses."""
    cube_xml = mc._cube_xml
    body = mass - mc.WHEEL_MASS - mc.CARRIAGE_MASS - mc.DISC_MASS
    com = Q50 @ np.array(COM60)
    diag = np.abs(Q50) @ np.array(INERTIA60) - body * (com @ com - com**2)  # about the centre of mass
    tag = f'<inertial pos="{_vertices([com])}" mass="{body:.6g}" diaginertia="{_vertices([diag])}"/>'

    def patched(i, p):
        return re.sub(r'(<freejoint name="f\d+"/>)', lambda hit: hit[1] + "\n" + tag, cube_xml(i, p), count=1)

    mc._cube_xml = patched
    try:
        yield
    finally:
        mc._cube_xml = cube_xml


def pivots_magnets(seeds=10, torque=HINGE60, sides=None, mass=MASS60, magnets=True, tag="", skin=None):
    """The pivot checks at 60 mm (chamfer 9, tiles 60 mm, deployed knuckles new2 scaled 1.2) with
    the face magnets on (check "pivot_magnets"). sides: None, "free" (a lone cube on each side of the
    climber, touching its side faces, unbonded on the tiles) or "bonded" (the same, welded to the
    world as if bonded to their tiles). The floor tiles carry no magnets and no steel."""
    import magnets as mgn  # noqa: F401  (fails early if magpylib is missing)

    t = time.time()
    with _size(SIZE60) as mc, _knuckle_c("sled_deployed") as (mc, log):
        mc.WHEEL_MASS, mc.CARRIAGE_MASS, mc.DISC_MASS = 0.001, 0.001, 0.006
        mc.BODY_MASS = mass - mc.WHEEL_MASS - mc.CARRIAGE_MASS - mc.DISC_MASS
        mc.MASS = mass
        mc.PIVOT_TORQUE, mc.PIVOT_RATE = torque, RATE60
        field = MagnetField(mc, skin)
        ctx = _magnets_c(mc, field) if magnets else contextlib.nullcontext()
        out = {"roll": [], "climb_up": [], "climb_down": [], "peaks": [], "landing_mm": [], "outcomes": []}
        with ctx, _inertial60(mc, mass):
            for seed in range(seeds):
                field.label = "settle"
                lone = mc.World([mc.Placement((0.0, 0.0), down="+y")], seed=seed)
                lone.run(0.2)
                start = lone.pos(0)
                field.label = log.label = "floor_roll"
                r = lone.floor_roll(0, [1, 0, 0])
                out["roll"].append(r == "arrived" and abs(lone.pos(0)[0] - start[0] - mc.SIDE) < 0.003)
                out["peaks"] += [p[1] for p in lone.pivot_peaks]
                field.label = log.label = "settle"
                pitch = mc.SIDE + 0.0005
                pls = [mc.Placement((0.0, k * pitch), down="+y") for k in range(3)]
                if sides:
                    pls += [mc.Placement((s * pitch, 2 * pitch), down="+y") for s in (1, -1)]
                w = mc.World(pls, seed=seed, bonds=[(0, 1, None), (1, 2, None)])
                if sides == "bonded":
                    for j in (3, 4):
                        _weld_to_world(w, j)
                planner = mc.Planner(w)
                w.run(0.3)
                field.label = log.label = "climb_up"
                up = planner.climb(2, 1)
                out["climb_up"].append(up == "arrived" and abs(w.pos(2)[2] - w.pos(1)[2] - mc.SIDE) < 0.003)
                out["landing_mm"].append(abs(w.pos(2)[2] - w.pos(1)[2] - mc.SIDE) * 1000)
                field.label = log.label = "climb_down"
                down = planner.climb_down(2, 1, w.pos(1) - w.pos(0))
                out["climb_down"].append(down == "arrived" and abs(w.pos(2)[2] - w.pos(1)[2]) < 0.003)
                out["peaks"] += [p[1] for p in w.pivot_peaks]
                out["outcomes"].append([r, up, down])
                print(f"seed {seed}: {r} {up} {down} evals {field.evals} reuses {field.reuses} {time.time() - t:.0f} s", flush=True)
        g = mass * 9.81 * mc.SIDE / math.sqrt(2)
    row = {
        "check": "pivot_magnets" + tag, "size_m": SIZE60["SIDE"], "mass_kg": mass, "torque_nm": torque, "rate_deg_s": RATE60, "seeds": seeds,
        "inertia_cube50_frame_kg_m2": INERTIA60, "com_cube50_frame_m": COM60, "knuckle": "sled_deployed", "skin_mm": round(field.skin_mm, 3),
        "magnets": bool(magnets), "side_neighbours": sides or "none",
        "floor_roll": f"{sum(out['roll'])}/{seeds}", "climb_up": f"{sum(out['climb_up'])}/{seeds}", "climb_down": f"{sum(out['climb_down'])}/{seeds}",
        "peak_hinge_torque_nm": round(max(out["peaks"]), 4), "landing_error_mm_max": round(max(out["landing_mm"]), 2),
        "gravity_torque_nm_derived": round(g, 4), "outcomes": out["outcomes"], "knuckle_contacts": log.summary(),
        "magnet_log": field.rows if magnets else None, "wall_s": round(time.time() - t, 1),
    }
    emit(row)
    return row


def _weld_to_world(w, j):
    """Weld cube j to the world where it stands (a cube bonded to its tile)."""
    m, d = w.model, w.data
    slot = next(k for k in w.weld_ids if k not in {s for s, *_ in w.bonds.values()})
    b = w.body[j]
    m.eq_obj1id[slot], m.eq_obj2id[slot] = b, 0
    m.eq_data[slot][:3] = 0.0
    m.eq_data[slot][3:6] = -d.xmat[b].reshape(3, 3).T @ d.xpos[b]  # the world origin in the cube's frame
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, d.xmat[b])
    qi = np.zeros(4)
    mujoco.mju_negQuat(qi, q)
    m.eq_data[slot][6:10] = qi
    m.eq_data[slot][10] = 1.0
    d.eq_active[slot] = 1
    w.weld_ids = [k for k in w.weld_ids if k != slot]  # reserved for good


# ---------------------------------------------------------------- joints (module_b + module_demos)
class _Null:
    def __init__(self, *a, **k):
        self.still = None
        self.still_at = None
        self.caption = ""

    def __call__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None


_B_BUILD = None
_RIG = []


def _knuckle_build(build, variant):
    """module_b's build plus knuckle bodies (fixed children of each module's body), knuckle-own-disc
    exclusions and parent filtering off. variant None: only parent filtering off (the baseline)."""
    from beebot.hardware import module_b as mb

    def patched(modules, scene="", extra="", timestep=mb.DT, options="", plates=False):
        x = build(modules, scene, extra, timestep, options=options, plates=plates)
        x = re.sub(r"<option ([^>]*)/>", r'<option \1><flag filterparent="disable"/></option>', x, count=1)
        if variant is None:
            return x
        geoms = ""
        for sd in (1, -1):  # cube50 hinge edges: x = +25 (k = y), z = +-25; module_b frame = (cube50 z, x, y)
            for j, (p1, p2, r) in enumerate(edge_cylinders(variant, 1, sd, 1, mb.HALF)):
                q1, q2 = p1[[2, 0, 1]], p2[[2, 0, 1]]
                geoms += f'<geom name="m{{i}}k{"p" if sd > 0 else "n"}{j}" type="cylinder" fromto="{_vertices([q1, q2])}" size="{r:.6g}" mass="0" rgba="0.9 0.2 0.2 1" friction="{mb.BODY_FRICTION} 0.005 0.0001"/>\n'

        def add(match):
            i = match[2]
            return match[1] + f'<body name="m{i}k">\n' + geoms.replace("{i}", i) + "</body>\n"

        x = re.sub(r'(<body name="m(\d+)b"[^>]*>\n(?:<freejoint/>\n|<joint [^>]*/>\n)?)', add, x)
        # A knuckle does not collide with its own discs, nor with the discs of modules bolted to the
        # same rigid body (bonded body face to body face): module_b's 52 mm discs stand 1 mm proud of
        # the side faces radially and only spin in place there, while Cube50's discs are recessed in
        # their faces. Everything bonded on those discs, and every disc that moves past, still collides.
        group = list(range(len(modules)))

        def root(k):
            while group[k] != k:
                k = group[k]
            return k

        for k, mod in enumerate(modules):
            if mod.parent is not None and mb.part_of(mod.face) == "b" and mb.part_of(mod.mate) == "b":
                group[root(k)] = root(mod.parent)
        excl = "".join(f'<exclude body1="m{i}k" body2="m{j}{p}"/>\n' for i in range(len(modules)) for j in range(len(modules)) if root(i) == root(j) for p in "pn")
        return x.replace("<contact>\n", "<contact>\n" + excl, 1)

    return patched


def _static_overlaps(rig, ids):
    """Knuckle geoms that overlap another module's geom (or the floor) at the build pose, split into
    pairs that collide and pairs the contact filters hide (same rigid body, or an excluded pair)."""
    m, d = rig.model, rig.data
    excl = set(re.findall(r'<exclude body1="([^"]+)" body2="([^"]+)"/>', rig.xml))
    ft = np.zeros(6)
    seen, hidden = {}, {}
    for g in ids:
        bg = m.geom_bodyid[g]
        own = re.fullmatch(r"m(\d+)k", m.body(bg).name)[1]
        for h in range(m.ngeom):
            if h in ids or not (m.geom_contype[h] or m.geom_conaffinity[h]):
                continue
            bh = m.geom_bodyid[h]
            nh = m.body(bh).name or "world"
            if re.fullmatch(rf"m{own}[bpnk]", nh):
                continue
            depth = -mujoco.mj_geomDistance(m, d, int(g), int(h), 0.01, ft) * 1000
            if depth <= 0.01:
                continue
            pair = (m.body(bg).name, nh)
            filtered = m.body_weldid[bg] == m.body_weldid[bh] or pair in excl or pair[::-1] in excl
            box = hidden if filtered else seen
            key = f"{pair[0]}-{nh}"
            box[key] = round(max(box.get(key, 0.0), depth), 2)
    return {"colliding": seen, "hidden_by_filters": hidden}


def _patch_b(speed_torque=True, knuckles=False):
    """knuckles False: module_b as it is. None: parent filtering off, no knuckles (baseline of the
    knuckle checks). "new2_stowed" or "old": knuckle geoms added and their contacts recorded."""
    global _B_BUILD
    from beebot.hardware import module_b as mb

    if _B_BUILD is None:
        _B_BUILD = mb.build
    mb.build = _B_BUILD if knuckles is False else _knuckle_build(_B_BUILD, knuckles)
    mb.BODY_MASS = MASS - 2 * mb.DISC_MASS
    mb.HOLD = DISC_HOLD
    _RIG.clear()

    def clipped_step(self):
        d, m = self.data, self.model
        if self.steps == 0:
            _RIG.append(self)
            self.knuckle_ids = np.array([g for g in range(m.ngeom) if re.fullmatch(r"m\d+k", m.body(m.geom_bodyid[g]).name or "")])
            self.knuckle_log = KnuckleLog()
            self.static = _static_overlaps(self, self.knuckle_ids) if len(self.knuckle_ids) else {}
            self.clip_s = np.zeros(m.nu)
        q_ref, qd_ref = self._reference(d.time)
        for a, path in self.paths.items():
            q_ref[a], qd_ref[a] = path(d.time)
        q, qd = d.qpos[self.qadr], d.qvel[self.vadr]
        tau = self.kp * (q_ref - q) + self.kv * (qd_ref - qd)
        cap = np.minimum(self.cap, DISC_HOLD)
        if speed_torque:
            line = np.clip(DISC_STALL * (1 - np.abs(qd) / DISC_FREE), 0.0, None)
            cap = np.where(tau * qd > 0, np.minimum(cap, line), cap)
        self.clip_s += ((np.abs(tau) >= cap - 1e-9) & ~self.free) * m.opt.timestep
        tau = np.where(self.free, 0.0, np.clip(tau, -cap, cap))
        d.ctrl[:] = tau
        mb.mujoco.mj_step(m, d)
        if len(self.knuckle_ids):
            self.knuckle_log.scan(m, d, self.knuckle_ids)
        force = np.abs(d.actuator_force)
        self.peak = np.maximum(self.peak, force)
        self.over += (force > mb.CONTINUOUS + 1e-9) * m.opt.timestep
        self.limit += (force > DISC_HOLD - 1e-3) * m.opt.timestep
        self.steps += 1
        if self.steps % self.every == 0:
            self.bond_series.append(self.loads())

    mb.Rig.step = clipped_step
    return mb


def _knuckle_fields(knuckles):
    if knuckles is False:
        return {}
    rig = _RIG[0]
    return {"knuckle": knuckles or "none", "knuckle_contacts": rig.knuckle_log.summary().get("all", {"steps_with_contact": 0}),
            "knuckle_overlap_at_build_mm": rig.static, "at_limit_total_s": round(float(rig.limit.sum()), 3),
            "clipped_s": {n: round(float(c), 3) for n, c in zip(rig.names, rig.clip_s) if c > 0.001}}


def _demos():
    import module_demos as md

    md.OUT = Path(__file__).parent / "demo_scratch"
    md.Reel, md.Camera = _Null, _Null
    md.finish = lambda *a, **k: None
    rows = {}

    def record(name, ok, criterion, modules, summary, notes, measured):
        rows[name] = {"ok": bool(ok), "criterion": criterion, "measured": measured, "summary": summary}
        return rows[name]

    md.record = record
    return md, rows


def walker(period=4.0, speed_torque=True, hold=None, knuckles=False):
    global DISC_HOLD
    if hold is not None:
        DISC_HOLD = hold
    mb = _patch_b(speed_torque, knuckles)
    md, rows = _demos()
    src = inspect.getsource(md.walker).replace("period, duty, sweep = 4.0,", f"period, duty, sweep = {period},")
    ns = dict(vars(md))
    exec(src, ns)
    t = time.time()
    check = "walker" if knuckles is False else f"walker_knuckle_{knuckles or 'none'}"
    try:
        ns["walker"](None)
    except RuntimeError as e:
        emit({"check": check, "period_s": period, "hold_nm": DISC_HOLD, "ok": False, "error": str(e), **_knuckle_fields(knuckles), "wall_s": round(time.time() - t, 1)})
        return
    r = rows["walker"]
    s = r["summary"]
    emit({"check": check, "period_s": period, "speed_torque_line": speed_torque, "mass_kg": MASS, "hold_nm": DISC_HOLD,
          "ok": r["ok"], "measured": r["measured"], "peak_torque_nm": s.get("peak_torque_nm"), "at_limit_s": s.get("at_limit_s"),
          "bond_moment_nm": s.get("bond_moment_nm"), "bond_tension_n": s.get("bond_tension_n"), **_knuckle_fields(knuckles), "wall_s": round(time.time() - t, 1)})


def arm(speed_torque=True, hold=None, knuckles=False):
    global DISC_HOLD
    if hold is not None:
        DISC_HOLD = hold
    _patch_b(speed_torque, knuckles)
    md, rows = _demos()
    t = time.time()
    check = "arm" if knuckles is False else f"arm_knuckle_{knuckles or 'none'}"
    try:
        md.arm(None)
    except RuntimeError as e:
        emit({"check": check, "hold_nm": DISC_HOLD, "ok": False, "error": str(e), **_knuckle_fields(knuckles), "wall_s": round(time.time() - t, 1)})
        return
    r = rows["arm"]
    s = r["summary"]
    emit({"check": check, "speed_torque_line": speed_torque, "mass_kg": MASS, "hold_nm": DISC_HOLD, "ok": r["ok"], "measured": r["measured"],
          "peak_torque_nm": s.get("peak_torque_nm"), "at_limit_s": s.get("at_limit_s"), "bond_moment_nm": s.get("bond_moment_nm"),
          "bond_tension_n": s.get("bond_tension_n"), **_knuckle_fields(knuckles), "wall_s": round(time.time() - t, 1)})


if __name__ == "__main__":
    what = set(sys.argv[1:] or ["pivot", "walker", "arm", "knuckles", "quadruped"])
    if "knuckles" in what:
        what |= {"pivot_knuckle", "pivot_old", "walker_knuckle", "arm_knuckle"}
    full = DISC_HOLD
    if "pivot" in what:
        pivots(10, HINGE_TORQUE)
        pivots(10, HINGE_TORQUE / MARGIN)
    if "walker" in what:
        for period in (8.0, 10.0, 12.0):
            walker(period, hold=full)
        walker(10.0, hold=full / MARGIN)
    if "arm" in what:
        arm(hold=full)
        arm(hold=full / MARGIN)
    if "pivot_knuckle" in what:
        pivots(10, HINGE_TORQUE, knuckle="new2_deployed")
        pivots(10, HINGE_TORQUE / MARGIN, knuckle="new2_deployed")
    if "pivot_old" in what:
        pivots(10, HINGE_TORQUE, knuckle="old")
    if "walker_knuckle" in what:
        for variant in (None, "new2_stowed", "old"):
            walker(10.0, hold=full, knuckles=variant)
    if "arm_knuckle" in what:
        for variant in (None, "new2_stowed", "old"):
            arm(hold=full, knuckles=variant)
    if "walker_stowed" in what:
        walker(10.0, hold=full, knuckles="new2_stowed")
    if "arm_stowed" in what:
        arm(hold=full, knuckles="new2_stowed")
    if "magnets" in what:  # face magnets at 60 mm: pivots with and without side neighbours, then at the hinge torque / 1.5
        seeds = 10
        for sides in (None, "bonded", "free"):
            pivots_magnets(seeds, HINGE60, sides)
            pivots_magnets(seeds, HINGE60 / 1.5, sides)
    for arg in what:  # single runs: magnets:<sides>:<torque>:<seeds>[:off]
        if arg.startswith("magnets:"):
            _, sides, tq, n, *opt = arg.split(":")
            skin = next((float(o[5:]) for o in opt if o.startswith("skin=")), None)
            pivots_magnets(int(n), float(tq), None if sides == "none" else sides, magnets="off" not in opt, tag=f"_{sides}", skin=skin)
    if "quadruped" in what:  # 60 mm modules, walking and a 60 mm step-up: demand, then the peak / 1.5 (hardware/cube50/sim/quadruped.py)
        import quadruped as quad

        quad.run_all()
