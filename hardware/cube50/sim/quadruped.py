"""Load case: a quadruped of 60 mm modules walks on flat ground and steps up a 60 mm ledge. Measures
the disc torque, joint speed and bond loads that the gait needs (demand), then reruns with a lower
cap to confirm the gait still passes.

Run from the repo root (CPU):
    nice -n 10 .venv/bin/python hardware/cube50/sim/cube50_sim.py quadruped
    nice -n 10 .venv/bin/python hardware/cube50/sim/quadruped.py [cap_nm ...]

Model (module_b, patched in this process only):
1. 60 mm modules at a 60 mm pitch (faces touching), bodies chamfered 7.5 mm (convex mesh, collides)
   with the mass and inertia of a uniform 60 mm cube (a non-colliding box carries them), 30 mm discs
   flush with their faces. Module mass MODULE_MASS (0.25 kg, a guess until the CAD mass is in).
1. 24 modules: a body of 8 (two rows of four along x, every body module's disc axis along y), and four
   legs of 4: hip link on a body disc (hip, pitch), knee module, and a two-module shank on the knee
   module's disc (knee, pitch). Thigh 60 mm (hip axis to knee axis), shank 90 mm (knee axis to sole).
   Feet are the chamfered bottom faces of the lowest modules.
1. Contacts inside the robot are off except between different legs (faces touch at a 60 mm pitch, so
   a leg rubs the body or itself on a shared plane without them; a real robot needs a gap there).
   The floor and the ledge collide with everything.
1. Gait: a static crawl (right back, right front, left back, left front). Before each step the body
   shifts along x (and in height and pitch) so that the centre of mass, computed from the model,
   sits inside the triangle of the three standing feet; then the leg swings (lift, carry, set down).
   Body pose and foot positions are read from the sim at the start of each phase (a perfect camera)
   and joint references come from planar inverse kinematics, PD at 1 kHz. Knees all point forward. Pitch-only legs cannot
   shift the body sideways, so stability comes from the fore-aft shift alone.
1. Drives: ideal torque sources, clipped at the cap, no speed-torque line (demand run).
"""

import contextlib
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import cube50_sim as cs  # noqa: E402  (sets sys.path for beebot.hardware)
from beebot.hardware import module_b as mb  # noqa: E402

SIDE = 0.060  # m
HALF = SIDE / 2
CHAMFER = 0.0075  # m
MODULE_MASS = 0.25  # kg (guess; parameter)
DISC_RADIUS = 0.015  # m: 30 mm disc, flush
DISC_MASS = 0.005  # kg each, inside MODULE_MASS
LEDGE = 0.060  # m: one cube high
LEDGE_X = 0.50  # m: the ledge's front face (world x)
D = 3 * SIDE  # m between front and back hip axes
L1, L2 = SIDE, SIDE + HALF  # m: thigh (hip axis to knee axis), shank (knee axis to sole)
H_NOM, H_MIN, H_MAX = 0.12, 0.08, 0.14  # m: hip height above its feet (nominal, limits)
REACH = 0.147  # m: largest hip-to-sole distance (the leg is 0.150 straight)
INWARD = 0.025  # m: a foot may sit at most this far behind its hip (knees forward; the shank stays near vertical with the foot ahead of the hip)
STRIDE = 0.03  # m per step
CLEAR = 0.025  # m: swing clearance above the higher of start and target
EDGE_UP = 0.035  # m: least distance of a landing sole centre beyond the ledge's front face
NOMINAL = 0.035  # m: stance foot ahead of its hip (knee forward, shank near vertical)
MAX_STEP = 0.10  # m: longest step
SPAN = 0.30  # m: largest distance from the rearmost back foot to a front foot (the hips are 0.18 apart; longer, the back legs cannot follow)
MARGIN = 0.006  # m: least distance of the centre of mass inside the support triangle
SCRAPE = 0.004  # m: largest planned intrusion of the foot into the ledge's edge on a swing (the sim then resolves the scrape by contact; its loads count)
SHIFT_T, SWING_T = 1.0, 1.5  # s
KP = {"hip": 150.0, "knee": 150.0, "other": 10.0}
KV = {"hip": 3.0, "knee": 1.5, "other": 0.25}
ORDER = ("BR", "FR", "BL", "FL")
KNEE = {"FL": 1, "FR": 1, "BL": -1, "BR": -1}  # +1: knee forward. Knees out: the shanks stay near vertical in stance
EVERY = 5  # steps between bond-load samples (5 ms; 10 samples = 50 ms)
# The foot's chamfered cross-section in the shank frame (forward, down from the knee axis), m: the
# planner holds the corner that touches (the lowest), not the sole centre, since a tilted shank stands
# the foot on an edge.
VERTS = [(sx * (HALF - CHAMFER), L2) for sx in (-1, 1)] + [(sx * HALF, L2 - CHAMFER) for sx in (-1, 1)] + \
        [(sx * HALF, L2 - SIDE + CHAMFER) for sx in (-1, 1)] + [(sx * (HALF - CHAMFER), L2 - SIDE) for sx in (-1, 1)]


def minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def cells():
    """Lattice cells (module_b.lattice) and the leg map {leg: (hip module, side, H, K, S, S2)}."""
    c = [{"at": (0, 0, 2), "axis": "+y", "z0": HALF + 0.0005}]
    for x in (1, 2, 3):
        c.append({"at": (x, 0, 2), "axis": "+y", "parent": x - 1})
    for x in range(4):
        c.append({"at": (x, 0, 3), "axis": "+y", "parent": x})
    legs = {}
    for name, hx, hip, s in (("FL", 3, 3, 1), ("FR", 3, 3, -1), ("BL", 0, 0, 1), ("BR", 0, 0, -1)):
        k0 = len(c)
        c.append({"at": (hx, s, 2), "axis": "+x", "parent": hip})  # hip link, on the hip disc
        c.append({"at": (hx, s, 1), "axis": "+y", "parent": k0})  # knee module
        c.append({"at": (hx, 2 * s, 1), "axis": "+x", "parent": k0 + 1})  # shank, on the knee disc
        c.append({"at": (hx, 2 * s, 0), "axis": "+x", "parent": k0 + 2})  # foot
        legs[name] = (hip, s, k0, k0 + 1, k0 + 2, k0 + 3)
    return c, legs


@contextlib.contextmanager
def module60(mass, cap, excludes):
    names = ("SIDE", "HALF", "PITCH", "PLATE", "DISC_RADIUS", "DISC_MASS", "BODY_MASS", "RECESS", "HOLD", "build", "_geoms")
    saved = {k: getattr(mb, k) for k in names}
    mb.SIDE, mb.HALF, mb.PITCH, mb.PLATE = SIDE, HALF, SIDE, 0.0
    mb.DISC_RADIUS, mb.DISC_MASS, mb.BODY_MASS, mb.RECESS, mb.HOLD = DISC_RADIUS, DISC_MASS, mass - 2 * DISC_MASS, 0.0, cap
    geoms, build = saved["_geoms"], saved["build"]

    def patched_geoms(i, part, rgba, plates=False):
        x = geoms(i, part, rgba, plates)
        if part == "b":  # the box carries mass and inertia; the chamfered mesh collides
            x = x.replace(f'<geom name="m{i}b" type="box"', f'<geom name="m{i}b" type="box" contype="0" conaffinity="0"', 1)
            x += f'<geom name="m{i}bc" type="mesh" mesh="cube60" mass="0" rgba="{rgba}" friction="{mb.BODY_FRICTION} 0.005 0.0001"/>\n'
        return x

    def patched_build(modules, scene="", extra="", timestep=mb.DT, options="", plates=False):
        x = build(modules, scene, extra, timestep, options=options, plates=plates)
        x = x.replace("<asset>\n", f'<asset>\n<mesh name="cube60" vertex="{cs._vertices(cs.chamfered_cube(HALF, CHAMFER))}"/>\n', 1)
        extra_ex = "".join(f'<exclude body1="m{a}{pa}" body2="m{b}{pb}"/>\n' for a, b in excludes for pa in "bpn" for pb in "bpn")
        return x.replace("<contact>\n", "<contact>\n" + extra_ex, 1)

    mb._geoms, mb.build = patched_geoms, patched_build
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(mb, k, v)


class Quad:
    def __init__(self, cap=20.0, mass=MODULE_MASS, swing=SWING_T, shift=SHIFT_T):
        self.cap, self.mass, self.swing_t, self.shift_t = cap, mass, swing, shift
        c, self.legmap = cells()
        groups = [set(range(8))] + [set(v[2:]) for v in self.legmap.values()]
        excludes = set()
        for g in groups:  # inside the body, inside each leg
            excludes |= {(a, b) for a in g for b in g if a < b}
        for v in self.legmap.values():  # each leg against the body
            excludes |= {(min(a, b), max(a, b)) for a in v[2:] for b in range(8)}
        scene = f'<geom name="ledge" type="box" pos="{LEDGE_X + 0.5:.4f} 0 {LEDGE / 2:.4f}" size="0.5 0.4 {LEDGE / 2:.4f}" rgba="0.6 0.6 0.58 1" friction="0.8 0.005 0.0001"/>\n'
        with module60(mass, cap, sorted(excludes)):
            self.modules = mb.lattice(c)
            self.rig = mb.Rig(self.modules, scene=scene)
        m, d = self.rig.model, self.rig.data
        self.m, self.d = m, d
        self.rig.cap[:] = cap
        self.joint = {}
        for leg, (hip, s, H, K, S, S2) in self.legmap.items():
            side = "p" if s > 0 else "n"
            self.joint[leg] = (f"{hip}{side}", f"{K}{side}")
        kinds = {}
        for leg, (jh, jk) in self.joint.items():
            kinds[jh], kinds[jk] = "hip", "knee"
        for a, n in enumerate(self.rig.names):
            kind = kinds.get(n, "other")
            self.rig.kp[a], self.rig.kv[a] = KP[kind], KV[kind]
        self.kinds = kinds
        self.root = m.body("m0b").id
        self.hip_body = {leg: m.body(f"m{v[0]}b").id for leg, v in self.legmap.items()}
        self.foot_body = {leg: m.body(f"m{v[5]}b").id for leg, v in self.legmap.items()}
        self.vert_local = {leg: [d.xmat[b].reshape(3, 3).T @ np.array([f, 0.0, -(dd - SIDE)]) for f, dd in VERTS] for leg, b in self.foot_body.items()}
        self.vertex = {leg: 0 for leg in self.foot_body}
        self.centre_local = {leg: d.xmat[b].reshape(3, 3).T @ np.array([0.0, 0.0, -HALF]) for leg, b in self.foot_body.items()}
        self.front = {"FL": 1, "FR": 1, "BL": -1, "BR": -1}
        # bonds: the hip link's anchor (on the hip disc) and the shank's anchor (on the knee disc)
        self.bonds = {}
        for leg, (hip, s, H, K, S, S2) in self.legmap.items():
            for kind, mod in (("hip", H), ("knee", S)):
                b = m.body(f"m{mod}b").id
                normal = unit_toward(d, b, m.body(f"m{hip if kind == 'hip' else K}b").id)
                self.bonds[f"{leg}_{kind}"] = (b, normal)
        self.sign = {}
        for leg, (jh, jk) in self.joint.items():
            for j in (jh, jk):
                self.sign[j] = self._probe(leg, j)
        self.log_tau, self.log_qd, self.log_bond = [], [], []
        self.events = []
        self.inward = INWARD

    # ---------------------------------------------------------------- geometry
    def _probe(self, leg, j):
        m = self.m
        d2 = mujoco.MjData(m)
        d2.qpos[:] = self.d.qpos
        mujoco.mj_kinematics(m, d2)
        before = self.foot(leg, d2)[0]
        d2.qpos[self.rig.qadr[self.rig.index[j]]] += 0.05
        mujoco.mj_kinematics(m, d2)
        return 1.0 if self.foot(leg, d2)[0] > before else -1.0

    def foot(self, leg, d=None, idx=None):
        """World position of the foot corner `idx` (default: the one the leg stands on)."""
        d = self.d if d is None else d
        b = self.foot_body[leg]
        return d.xpos[b] + d.xmat[b].reshape(3, 3) @ self.vert_local[leg][self.vertex[leg] if idx is None else idx]

    def centre(self, leg):
        """World (x, z) of the sole centre: where a foot is, for planning (its corners are for contact)."""
        b = self.foot_body[leg]
        return (self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ self.centre_local[leg])[[0, 2]]

    def home(self, leg, centres):
        """The leg's stance spot (x) relative to the feet's centroid, which stands for the neutral
        body position (the body itself sits shifted for the last step's balance)."""
        middle = float(np.mean([c[0] for c in centres.values()]))
        return middle + self.front[leg] * D / 2 + KNEE[leg] * NOMINAL

    def centres(self):
        return {leg: self.centre(leg) for leg in self.joint}

    def corner_target(self, leg, pose, xc, zs):
        """(corner, point): the foot corner that ends lowest, and where it must go, for the sole
        centre to be at x = xc with the foot resting on the surface at height zs."""
        best = None
        for idx in range(4):
            pt = np.array([xc, zs])
            for _ in range(4):
                phi1, phi2 = self.angles(leg, pose, pt, idx)
                c = self.corners(leg, pose, phi1, phi2)
                centre = self.hip(leg, pose) + L1 * np.array([math.sin(phi1), -math.cos(phi1)]) + L2 * np.array([math.sin(phi2), -math.cos(phi2)])
                pt = np.array([xc + c[idx][0] - centre[0], zs])
            below = zs - float(c[:, 1].min())
            if best is None or below < best[0]:
                best = (below, idx, pt)
        return best[1], best[2]

    def lowest(self, leg):
        return int(np.argmin([self.foot(leg, idx=k)[2] for k in range(len(VERTS))]))

    def body_pose(self, d=None):
        d = self.d if d is None else d
        back, front = d.xpos[self.hip_body["BL"]], d.xpos[self.hip_body["FL"]]
        c = (back + front) / 2
        return np.array([c[0], c[2], math.atan2(front[2] - back[2], front[0] - back[0])])

    def hip(self, leg, pose):
        cx, cz, th = pose
        f = self.front[leg]
        return np.array([cx + f * D / 2 * math.cos(th), cz + f * D / 2 * math.sin(th)])

    def angles(self, leg, pose, point, idx=None):
        """Absolute thigh and shank angles (rad from straight down, positive forward) that put foot
        corner `idx` at `point`."""
        f, dd = VERTS[self.vertex[leg] if idx is None else idx]
        l2, delta = math.hypot(f, dd), math.atan2(f, dd)
        v = np.asarray(point) - self.hip(leg, pose)
        dist = float(np.clip(np.linalg.norm(v), abs(l2 - L1) + 1e-3, L1 + l2 - 1e-4))
        beta = math.acos(np.clip((dist**2 - L1**2 - l2**2) / (2 * L1 * l2), -1, 1))
        beta = -beta if KNEE[leg] > 0 else beta  # +1: the knee points forward
        alpha = math.atan2(v[0], -v[1])
        phi1 = alpha - math.atan2(l2 * math.sin(beta), L1 + l2 * math.cos(beta))
        return phi1, phi1 + beta - delta

    def ik(self, leg, pose, point, idx=None):
        phi1, phi2 = self.angles(leg, pose, point, idx)
        jh, jk = self.joint[leg]
        return {jh: self.sign[jh] * (phi1 - pose[2]), jk: self.sign[jk] * (phi2 - phi1)}

    def corners(self, leg, pose, phi1, phi2):
        """World (x, z) of every foot corner for the given angles (planar kinematics)."""
        knee = self.hip(leg, pose) + L1 * np.array([math.sin(phi1), -math.cos(phi1)])
        u, w = np.array([math.sin(phi2), -math.cos(phi2)]), np.array([math.cos(phi2), math.sin(phi2)])
        return np.array([knee + dd * u + f * w for f, dd in VERTS])

    def landing(self, leg, pose, target):
        """The corner that is lowest when it is put at `target` (fixed point of a few iterations)."""
        idx = self.vertex[leg]
        for _ in range(4):
            c = self.corners(leg, pose, *self.angles(leg, pose, target, idx))
            new = int(np.argmin(c[:, 1] - 1e-6 * (np.arange(len(VERTS)) == idx)))
            if new == idx:
                break
            idx = new
        return idx

    def com_for(self, pose, feet):
        """Centre of mass (x, y) with the body at `pose` and the feet at `feet` (model kinematics)."""
        m = self.m
        d2 = mujoco.MjData(m)
        d2.qpos[:] = self.d.qpos
        now = self.body_pose()
        dth = pose[2] - now[2]
        rot = np.array([[math.cos(-dth), 0, math.sin(-dth)], [0, 1, 0], [-math.sin(-dth), 0, math.cos(-dth)]])  # about +y by -dth: nose up
        root = self.d.xpos[self.root].copy()
        centre = np.array([now[0], root[1], now[1]])
        new_centre = np.array([pose[0], root[1], pose[1]])
        d2.qpos[0:3] = new_centre + rot @ (root - centre)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, quat_of(rot), self.d.qpos[3:7])
        d2.qpos[3:7] = q
        for leg, f in feet.items():
            for j, val in self.ik(leg, pose, f).items():
                d2.qpos[self.rig.qadr[self.rig.index[j]]] = val
        mujoco.mj_kinematics(m, d2)
        mujoco.mj_comPos(m, d2)
        return d2.subtree_com[self.root][:2].copy()

    def body_height(self, current, target):
        """Hip heights (front, back) for current and target foot heights, inside the leg's reach."""
        out = {}
        for end, legs in (("F", ("FL", "FR")), ("B", ("BL", "BR"))):
            zc = [current[k][1] for k in legs]
            zt = [target[k][1] for k in legs]
            z = np.mean(zt) + H_NOM
            z = max(z, max(zc + zt) + H_MIN)
            z = min(z, min(zc + zt) + H_MAX)
            out[end] = z
        return out

    def pose_for(self, cx, heights):
        zf, zb = heights["F"], heights["B"]
        th = math.asin(np.clip((zf - zb) / D, -0.9, 0.9))
        return np.array([cx, (zf + zb) / 2, th])

    # ---------------------------------------------------------------- motion
    def phase(self, T, pose0, pose1, path):
        """Run T s: the body goes pose0 -> pose1 (min jerk), each foot follows path(leg, s)."""
        ts = np.linspace(0.0, T, int(T * 200) + 1)
        refs = {j: np.zeros(len(ts)) for pair in self.joint.values() for j in pair}
        for k, t in enumerate(ts):
            u = minjerk(t / T)
            pose = pose0 + (pose1 - pose0) * u
            for leg in self.joint:
                for j, val in self.ik(leg, pose, path(leg, t / T)).items():
                    refs[j][k] = val
        t0 = self.d.time
        for j, arr in refs.items():
            rate = np.gradient(arr, ts)
            self.rig.follow(j, lambda t, arr=arr, rate=rate: (float(np.interp(t - t0, ts, arr)), float(np.interp(t - t0, ts, rate))))
        self.run(T + 0.15)

    def run(self, seconds):
        rig, m, d = self.rig, self.m, self.d
        end = d.time + seconds
        while d.time < end - 1e-9:
            q_ref, qd_ref = rig._reference(d.time)
            for a, path in rig.paths.items():
                q_ref[a], qd_ref[a] = path(d.time)
            q, qd = d.qpos[rig.qadr], d.qvel[rig.vadr]
            tau = np.clip(rig.kp * (q_ref - q) + rig.kv * (qd_ref - qd), -rig.cap, rig.cap)
            d.ctrl[:] = tau
            before = d.time
            mujoco.mj_step(m, d)
            if d.time < before:
                raise RuntimeError(f"MuJoCo reset at t = {before:.3f} s")
            rig.steps += 1
            self.log_tau.append(np.abs(d.actuator_force).copy())
            self.log_qd.append(np.abs(d.qvel[rig.vadr]).copy())
            if rig.steps % EVERY == 0:
                self.log_bond.append(self.bond_loads())

    def bond_loads(self):
        """Per bond: (bending moment N m, tension N) at the bond face centre (as module_b's loads)."""
        m, d = self.m, self.d
        mujoco.mj_rnePostConstraint(m, d)
        out = []
        for b, n_world0 in self.bonds.values():
            rot = d.xmat[b].reshape(3, 3)
            normal = rot @ n_world0
            point = d.xpos[b] + normal * HALF
            torque, force = d.cfrc_int[b, :3], d.cfrc_int[b, 3:]
            moment = torque + np.cross(d.subtree_com[m.body_rootid[b]] - point, force)
            out.append((float(np.linalg.norm(moment - (moment @ normal) * normal)), float(force @ normal)))
        return out

    def feet(self):
        """The corner each foot stands on (the lowest), world (x, z); re-chosen at every call."""
        for leg in self.joint:
            self.vertex[leg] = self.lowest(leg)
        return {leg: self.foot(leg)[[0, 2]] for leg in self.joint}

    def foot_y(self, leg):
        return self.foot(leg)[1]

    def violation(self, pose, feet, idx=None):
        """How far (m) any foot is outside its leg's workspace with the body at `pose`."""
        out = 0.0
        for k, f in feet.items():
            v = np.asarray(f) - self.hip(k, pose)
            f_, dd = VERTS[self.vertex[k] if idx is None else idx]
            out += max(0.0, float(np.linalg.norm(v)) - (L1 + math.hypot(f_, dd) - 0.003)) + max(0.0, -KNEE[k] * v[0] - self.inward) + max(0.0, H_MIN - 0.02 + v[1])
        return out

    def shift_x(self, lift, feet, target, heights, idx=None):
        """Body centre x nearest the feet's centroid that puts the centre of mass at least MARGIN
        (plus 4 mm) inside the triangle of the three feet other than `lift`, with every foot (and
        the swing target) inside its workspace; else the best compromise. Returns (x, margin m,
        workspace violation m)."""
        stance = [leg for leg in self.joint if leg != lift]
        tri = np.array([[feet[k][0], self.foot_y(k)] for k in stance])
        middle = float(np.mean([f[0] for f in feet.values()]))
        options, fallback = [], (-1e9, middle, 0.0, 1e9)
        for cx in np.arange(middle - 0.08, middle + 0.0801, 0.0025):
            pose = self.pose_for(cx, heights)
            bad = self.violation(pose, feet) + self.violation(pose, {lift: target}, idx)
            margin = triangle_margin(self.com_for(pose, feet), tri)
            if bad < 1e-4 and margin >= MARGIN:
                rank = abs(cx - middle) if margin >= MARGIN + 0.004 else 1.0 - margin  # comfortable first, nearest the middle; then by margin
                options.append((rank, cx, margin, bad))
            if margin - 10 * bad > fallback[0]:
                fallback = (margin - 10 * bad, cx, margin, bad)
        self.options = [(cx, margin) for _, cx, margin, _ in sorted(options)]
        best = min(options) if options else fallback
        return best[1], best[2], best[3]

    def swing(self, leg, pose, feet, target):
        """The swing path for `leg` (sole centre to `target`, x and surface height) with the body still at `pose`: lift, carry, set
        down, with the smallest backoff from the wall and extra height that keep every foot corner
        out of the ledge and the floor. Returns (intrusion m, path, backoff m, extra height m)."""
        saved = self.vertex[leg]
        idx, p1 = self.corner_target(leg, pose, target[0], target[1])
        self.vertex[leg] = idx
        p0 = self.corners(leg, pose, *self.angles(leg, pose, feet[leg], saved))[idx] if saved != idx else np.asarray(feet[leg])
        floor0 = LEDGE if p0[0] > LEDGE_X - HALF else 0.0  # what the foot starts over (the ledge if any part of it might be over the top)
        top = max(p0[1], p1[1], floor0) + CLEAR

        def make(back, top):
            def path(k, s):
                if k != leg:
                    return feet[k]
                if s < 0.3:
                    u = minjerk(s / 0.3)
                    return np.array([p0[0] - back * u, p0[1] + (top - p0[1]) * u])
                if s < 0.7:
                    return np.array([p0[0] - back + (p1[0] - p0[0] + back) * minjerk((s - 0.3) / 0.4), top])
                return np.array([p1[0], top + (p1[1] - 0.002 - top) * minjerk((s - 0.7) / 0.3)])
            return path

        def hits(path):
            """Largest intrusion (m) of any foot corner into the ledge or the floor along the path."""
            worst = 0.0
            for s_ in np.linspace(0.0, 1.0, 41):
                c = self.corners(leg, pose, *self.angles(leg, pose, path(leg, s_)))
                for x, z in c:
                    if LEDGE_X - 0.004 < x < LEDGE_X + 0.004:
                        worst = max(worst, LEDGE + 0.002 - z)
                    elif x >= LEDGE_X + 0.004:
                        worst = max(worst, LEDGE - 0.006 - z)
                    else:
                        worst = max(worst, -0.006 - z)
                worst = max(worst, inside(c, np.array([LEDGE_X, LEDGE])))  # the ledge's edge through the foot's sole or side
            return worst

        best = None
        for back in (0.0, 0.01, 0.02, 0.03):
            for extra in (0.0, 0.01, 0.02, 0.03):
                cand = make(back, top + extra)
                w = hits(cand)
                if best is None or w < best[0]:
                    best = (w, cand, back, extra)
                if w <= 0.0:
                    break
            if best[0] <= 0.0:
                break
        if best[0] > SCRAPE:
            self.vertex[leg] = saved
        return best

    def plan(self, leg, feet, xs=None):
        """Swing target for `leg`: the longest step up to STRIDE (or the first of `xs`) whose shift
        is feasible (inside the workspace, centre of mass at least MARGIN inside); near the ledge,
        on top or short of it."""
        centres = self.centres()
        x0 = centres[leg][0]
        if xs is None:
            desired = self.home(leg, centres) + STRIDE  # a stride ahead of its stance spot
            desired = min(desired, x0 + MAX_STEP)
            floor = [x for x in np.arange(desired, x0 + 0.0049, -0.005) if not LEDGE_X - 0.001 < x < LEDGE_X + EDGE_UP]
            ups = [LEDGE_X + EDGE_UP + k * 0.01 for k in range(5)] if centres[leg][1] < LEDGE / 2 and desired > LEDGE_X - 0.06 else []
            cands = [x for x in ups if x <= x0 + MAX_STEP + 0.02] + floor
            if self.front[leg] > 0:
                cands = [x for x in cands if x <= min(centres["BL"][0], centres["BR"][0]) + SPAN]
        else:
            cands = list(xs)
        seen = []
        for x in cands:
            if any(abs(x - y) < 1e-6 for y in seen):
                continue
            seen.append(x)
            target = np.array([x, LEDGE if x > LEDGE_X else 0.0])
            idx, corner = self.corner_target(leg, self.body_pose(), x, target[1])
            tgt = dict(feet)
            tgt[leg] = corner
            heights = self.body_height(feet, tgt)
            cx, margin, bad = self.shift_x(leg, feet, corner, heights, idx)
            hit = None
            for cx_, margin_ in self.options[:12]:  # feasible body positions, best first, until the swing clears
                hit = self.swing(leg, self.pose_for(cx_, heights), feet, target)[0]
                if hit <= SCRAPE:
                    return target, heights, cx_, margin_
            self.why = {"x": round(float(x), 3), "margin_mm": round(margin * 1000, 1), "reach_mm": round(bad * 1000, 1), "hit_mm": None if hit is None else round(hit * 1000, 1)}
        return None

    def step(self, leg, plan):
        target, heights, cx, margin = plan
        feet = self.feet()
        pose0, pose1 = self.body_pose(), self.pose_for(cx, heights)
        self.phase(self.shift_t, pose0, pose1, lambda k, s: feet[k])
        feet = self.feet()
        pose = self.body_pose()
        best = self.swing(leg, pose, feet, target)
        p1 = np.asarray(target)
        if best[0] > SCRAPE + 0.002:  # after the real shift the swing would hit the ledge: stay down
            self.events.append({"t": round(self.d.time, 2), "leg": leg, "aim": [round(float(v), 3) for v in p1], "aborted_hit_mm": round(best[0] * 1000, 1)})
            self.aborted += 1
            return
        path = best[1]
        self.phase(self.swing_t, pose, pose, path)
        self.events.append({"t": round(self.d.time, 2), "leg": leg, "aim": [round(float(v), 3) for v in p1], "to": [round(float(v), 3) for v in self.centre(leg)],
                            "margin_mm": round(margin * 1000, 1), "backoff_mm": round(best[2] * 1000), "raise_mm": round(best[3] * 1000), "path_hit_mm": round(best[0] * 1000, 1)})

    def on_top(self):
        feet = self.feet()
        return int(sum(feet[k][1] > LEDGE - 0.01 and c[0] > LEDGE_X + 0.02 for k, c in self.centres().items()))

    def all_up(self):
        return self.on_top() == 4

    def upright(self):
        r = self.d.xmat[self.root].reshape(3, 3)
        return math.degrees(math.acos(min(1.0, abs(float(r[1, 0])))))  # tilt of the hip axis (local x) from world y: roll

    def walk(self, max_cycles=16):
        self.run(0.5)
        feet = self.feet()
        pose = self.body_pose()
        crouch = self.pose_for(pose[0], {"F": H_NOM, "B": H_NOM})
        self.phase(1.0, pose, crouch, lambda k, s: feet[k])
        self.run(0.3)
        self.inward = 0.06  # start-up: each foot steps out from under its hip to the knees-out stance
        for leg in ORDER:
            feet = self.feet()
            f = KNEE[leg]
            plan = self.plan(leg, feet, xs=[self.centre(leg)[0] + f * (0.03 - k * 0.005) for k in range(5)])
            if plan is None:
                raise RuntimeError(f"start-up: no feasible step for {leg}")
            self.step(leg, plan)
        self.inward = INWARD
        self.aborted = 0
        self.log_tau, self.log_qd, self.log_bond = [], [], []  # demand is measured from here (walking and the step-up)
        self.startup_events = len(self.events)
        x_start = self.body_pose()[0]
        cycles = 0
        cycles = 0
        for n in range(max_cycles * 4):  # free crawl: the leg furthest behind its stance spot that can step, steps
            feet = self.feet()
            pose = self.body_pose()
            centres = self.centres()
            lag = {leg: self.home(leg, centres) - centres[leg][0] for leg in self.joint}
            plan = None
            for leg in sorted(self.joint, key=lambda k: -lag[k]):
                plan = self.plan(leg, feet)
                if plan is not None:
                    break
                self.events.append({"t": round(self.d.time, 2), "leg": leg, "skipped": getattr(self, "why", True)})
            if plan is None:
                break
            self.step(leg, plan)
            if self.aborted >= 3:
                break
            cycles = (n + 1) / 4
            if self.upright() > 20:
                return {"fell": True, "cycles": cycles}
            if self.all_up():
                break
        feet = self.feet()
        pose = self.body_pose()
        heights = self.body_height(feet, feet)
        self.phase(1.0, pose, self.pose_for(float(np.mean([c[0] for c in self.centres().values()])), heights), lambda k, s: feet[k])
        self.run(1.0)
        on_top = self.on_top()
        centres = self.centres()
        stage = ("all four on the ledge" if on_top == 4 else "front feet on the ledge, back feet at the wall" if on_top >= 2 and min(centres["BL"][0], centres["BR"][0]) > LEDGE_X - 0.06
                 else f"{on_top} feet on the ledge")
        return {"fell": self.upright() > 20, "stage": stage, "steps": cycles * 4, "feet_on_ledge": int(on_top), "advanced_cm": round((self.body_pose()[0] - x_start) * 100, 1),
                "roll_deg": round(self.upright(), 1), "pitch_deg": round(math.degrees(self.body_pose()[2]), 1), "sim_s": round(self.d.time, 1)}

    # ---------------------------------------------------------------- measurement
    def measure(self):
        tau, qd = np.array(self.log_tau), np.array(self.log_qd)
        out = {"torque_peak_nm": {}, "torque_peak_50ms_nm": {}, "torque_p999_nm": {}, "torque_p95_nm": {}, "speed_p95_rad_s": {}, "speed_peak_50ms_rad_s": {}, "speed_peak_rad_s": {}}
        box = np.ones(50) / 50  # 50 ms at 1 kHz
        for leg, (jh, jk) in self.joint.items():
            for kind, j in (("hip", jh), ("knee", jk)):
                a = self.rig.index[j]
                key = f"{leg}_{kind}"
                out["torque_peak_nm"][key] = round(float(tau[:, a].max()), 3)
                out["torque_peak_50ms_nm"][key] = round(float(np.convolve(tau[:, a], box, "valid").max()), 3)
                out["torque_p999_nm"][key] = round(float(np.percentile(tau[:, a], 99.9)), 3)
                out["speed_peak_50ms_rad_s"][key] = round(float(np.convolve(qd[:, a], box, "valid").max()), 3)
                out["torque_p95_nm"][key] = round(float(np.percentile(tau[:, a], 95)), 3)
                out["speed_p95_rad_s"][key] = round(float(np.percentile(qd[:, a], 95)), 3)
                out["speed_peak_rad_s"][key] = round(float(qd[:, a].max()), 3)
        at_cap = tau >= self.cap - 1e-3
        out["time_at_cap_s"] = {f"{leg}_{kind}": round(float(at_cap[:, self.rig.index[j]].sum()) * self.m.opt.timestep, 3)
                                for leg, (jh, jk) in self.joint.items() for kind, j in (("hip", jh), ("knee", jk)) if at_cap[:, self.rig.index[j]].any()}
        others = [a for a, n in enumerate(self.rig.names) if n not in self.kinds]
        out["other_joints_torque_peak_nm"] = round(float(tau[:, others].max()), 3)
        bonds = np.array(self.log_bond)  # samples x bonds x 2
        width = int(round(0.05 / (EVERY * self.m.opt.timestep)))
        kernel = np.ones(width) / width
        out["bond"] = {}
        for k, name in enumerate(self.bonds):
            mom, ten = bonds[:, k, 0], bonds[:, k, 1]
            out["bond"][name] = {"moment_peak_nm": round(float(mom.max()), 3), "moment_50ms_nm": round(float(np.convolve(mom, kernel, "valid").max()), 3),
                                 "tension_peak_n": round(float(ten.max()), 2), "tension_50ms_n": round(float(np.convolve(ten, kernel, "valid").max()), 2)}
        for kind in ("hip", "knee"):
            keys = [k for k in out["torque_peak_nm"] if k.endswith(kind)]
            out[f"{kind}_max"] = {"torque_peak_nm": max(out["torque_peak_nm"][k] for k in keys), "torque_peak_50ms_nm": max(out["torque_peak_50ms_nm"][k] for k in keys),
                                  "torque_p999_nm": max(out["torque_p999_nm"][k] for k in keys), "torque_p95_nm": max(out["torque_p95_nm"][k] for k in keys),
                                  "speed_peak_50ms_rad_s": max(out["speed_peak_50ms_rad_s"][k] for k in keys),
                                  "speed_p95_rad_s": max(out["speed_p95_rad_s"][k] for k in keys),
                                  "bond_moment_50ms_nm": max(out["bond"][f"{k}"]["moment_50ms_nm"] for k in keys),
                                  "bond_moment_peak_nm": max(out["bond"][f"{k}"]["moment_peak_nm"] for k in keys),
                                  "bond_tension_50ms_n": max(out["bond"][f"{k}"]["tension_50ms_n"] for k in keys)}
        return out


def unit_toward(d, b, other):
    """The bond normal in body b's frame: from b's centre toward the neighbour it is bonded to."""
    v = d.xpos[other] - d.xpos[b]
    local = d.xmat[b].reshape(3, 3).T @ v
    n = np.zeros(3)
    k = int(np.argmax(np.abs(local)))
    n[k] = np.sign(local[k])
    return n


def quat_of(rot):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(rot, float).reshape(-1))
    return q


def inside(corners, point):
    """Depth (m) of `point` inside the convex foot outline given by its corners (0 if outside)."""
    hull = corners[[0, 2, 4, 6, 7, 5, 3, 1]]  # around the cross-section: bottom, side, top, other side
    centre = hull.mean(axis=0)
    depth = 1e9
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        e = b - a
        n = np.array([-e[1], e[0]]) / max(np.linalg.norm(e), 1e-12)
        if (centre - a) @ n < 0:
            n = -n
        depth = min(depth, float((point - a) @ n))
    return max(depth, 0.0)


def triangle_margin(p, tri):
    """Signed distance of point p (x, y) inside the triangle (positive inside)."""
    c = tri.mean(axis=0)
    out = 1e9
    for i in range(3):
        a, b = tri[i], tri[(i + 1) % 3]
        e = b - a
        n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
        if (c - a) @ n < 0:
            n = -n
        out = min(out, float((p - a) @ n))
    return out


def quadruped(cap=20.0, mass=MODULE_MASS, label="demand", swing=SWING_T, shift=SHIFT_T):
    t = time.time()
    q = Quad(cap, mass, swing, shift)
    try:
        result = q.walk()
        error = None
    except RuntimeError as e:
        result, error = {"fell": True}, str(e)
    ok = not result.get("fell") and result.get("feet_on_ledge") == 4 and error is None  # the full task: all four feet on top
    row = {"check": f"quadruped_{label}", "cap_nm": cap, "speed_torque_line": False, "module_mass_kg": mass, "modules": len(q.modules),
           "module_mm": SIDE * 1000, "pitch_mm": SIDE * 1000, "disc_mm": DISC_RADIUS * 2000, "chamfer_mm": CHAMFER * 1000, "ledge_mm": LEDGE * 1000,
           "gait": {"kind": "static crawl, the leg furthest behind its stance spot that can step goes next", "shift_s": shift, "swing_s": swing, "per_step_s": shift + swing,
                    "cycle_s": 4 * (shift + swing), "stride_mm": STRIDE * 1000, "clearance_mm": CLEAR * 1000, "hip_height_mm": H_NOM * 1000, "stability_margin_mm": MARGIN * 1000,
                    "ledge_scrape_allowed_mm": SCRAPE * 1000, "kp_kv": {"hip": [KP["hip"], KV["hip"]], "knee": [KP["knee"], KV["knee"]]}},
           "measured_from": "after start-up (stand, crouch, feet out to the stance): walking, front step-up, back legs to the wall",
           "ok": ok, "result": result, "error": error, **q.measure(), "steps": q.events, "wall_s": round(time.time() - t, 1)}
    cs.emit(row)
    return row


def run_all(caps=()):
    """Demand at a 20 N m cap, then a rerun at each cap in `caps` (default: the 50 ms-averaged peak / 1.5); a rerun
    passes when it reaches the demand run's stage without falling."""
    demand = quadruped(20.0, label="demand")
    peak = max(demand["hip_max"]["torque_peak_50ms_nm"], demand["knee_max"]["torque_peak_50ms_nm"])  # 50 ms average: the raw peak is a few-ms contact spike
    for cap in caps or (round(peak / 1.5, 3),):
        row = quadruped(cap, label=f"cap_{cap:g}")
        same = row["result"].get("stage") == demand["result"].get("stage") and not row["result"].get("fell")
        cs.emit({"check": f"quadruped_cap_{cap:g}_vs_demand", "cap_nm": cap, "same_stage_as_demand": bool(same), "demand_stage": demand["result"].get("stage"),
                 "stage": row["result"].get("stage"), "advanced_cm": [demand["result"].get("advanced_cm"), row["result"].get("advanced_cm")]})


if __name__ == "__main__":
    run_all(tuple(float(a) for a in sys.argv[1:]))
