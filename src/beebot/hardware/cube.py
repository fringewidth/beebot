"""Rigid-body simulation of a pivoting cube module (M-Blocks style) in MuJoCo.

The module is a cube of side L that moves by braking an internal flywheel: the flywheel's angular
momentum jumps into the cube, which pivots about an edge held by magnets. Each move the voxel
policy issues is one primitive here (floor roll, lattice traverse, convex and concave transitions),
run as a short scene with a mover and passive neighbours, and judged by where the mover lands.

Model:

1. Bodies. Each cube is a rigid box. The mover also carries a flywheel (a ring on a hinge about the
   module's y axis); the flywheel's mass is part of the module's mass.
1. Drive. The flywheel starts at speed (spin-up is slow enough not to move the cube). At t = BRAKE_AT
   a band brake engages: Coulomb friction of brake_torque on the flywheel hinge (applied as an
   explicit torque; MuJoCo's frictionloss is a soft constraint and acts like a damper at these
   time scales), held for BRAKE_HOLD seconds, so the flywheel's momentum passes to the cube over a
   few milliseconds.
1. Magnets. Every cube has 24 edge magnets (two per edge) and 6 face-centre magnets, all inset from
   the surface. Every pair of magnets on different cubes attracts along the line between them with
   F(r) = A (r / d0) / (1 + r^2 / d0^2)^2, which peaks at r = d0 / sqrt(3) and falls as r^-3. The
   inset puts twin magnets of two cubes in face contact exactly at that peak, so a pull-off force
   of `pull` holds them and lateral offset is resisted too. Edge magnets of cubes that share only
   an edge also attract, which is the hinge of a pivot. A steel floor, when present, attracts each
   magnet like a twin at its mirror image.
1. Contacts. MuJoCo's soft contacts with Coulomb friction (elliptic cones).

Units: SI unless a field says otherwise.

    from beebot.hardware.cube import Cube, trial
    trial(Cube(), "floor_roll", omega=400.0, rng=np.random.default_rng(0))
"""

import math
from dataclasses import dataclass

import mujoco
import numpy as np

G = 9.81
DT = 2e-4
BRAKE_AT = 0.05  # s, contacts settle first
BRAKE_HOLD = 0.25  # s, the band brake's longest actuation (3D M-Blocks)
HORIZON = 2.0  # s, longest episode
SETTLE = 0.1  # s of stillness, with the mover lattice-aligned, that ends an episode early
SHAPE_PEAK = (1 / math.sqrt(3)) / (1 + 1 / 3) ** 2  # max of x / (1 + x^2)^2
ALIGNED = math.cos(math.radians(10))
LANDED = 0.15  # L: the mover is in a cell when its centre is this close
DISTURBED = 0.1  # L: a neighbour moved this far counts as knocked


@dataclass(frozen=True)
class Cube:
    side: float = 0.05  # L, m
    mass: float = 0.15  # kg, whole module
    flywheel_share: float = 0.155  # flywheel mass / module mass
    flywheel_radius: float = 0.38  # ring radius / L
    max_rpm: float = 20000.0
    brake_torque: float = 2.6  # N m while the band slips (3D M-Blocks' measured peak)
    edge_pull: float = 4.5  # pull-off force of one pair of twin edge magnets, in module weights
    face_pull: float = 0.3  # pull-off force of the pair of face-centre (alignment) magnets, in module weights
    magnet_range: float = 0.03  # d0 / L
    floor_pull: float = 0.0  # 1.0: a steel floor holds each magnet like its twin would; 0: no steel
    friction: float = 0.5

    def flywheel_inertia(self):
        return self.flywheel_share * self.mass * (self.flywheel_radius * self.side) ** 2

    def max_omega(self):
        return self.max_rpm * 2 * math.pi / 60


# ---------------------------------------------------------------------------------------------
# Primitives: cells are (x, y, z) in L, z up. axis is the flywheel's axis and sign the direction of
# its spin (+1 about +y moves the top of the cube toward +x). The defaults (Cube()) are the 3D
# M-Blocks as calibrated against their measured success rates (HARDWARE.md).

PRIMITIVES = {
    "floor_roll": {"substrate": [], "mover": (0, 0, 0), "target": (1, 0, 0), "sign": 1},
    "floor_peel": {"substrate": [(-1, 0, 0), (0, 1, 0)], "mover": (0, 0, 0), "target": (1, 0, 0), "sign": 1},
    "traverse": {"substrate": [(0, 0, 0), (1, 0, 0)], "mover": (0, 0, 1), "target": (1, 0, 1), "sign": 1},
    "convex_up": {"substrate": [(1, 0, 0)], "mover": (0, 0, 0), "target": (1, 0, 1), "sign": 1},
    "convex_down": {"substrate": [(0, 0, 0)], "mover": (0, 0, 1), "target": (1, 0, 0), "sign": 1},
    "concave_up": {"substrate": [(1, 0, 0), (1, 0, 1)], "mover": (0, 0, 0), "target": (0, 0, 1), "sign": 1},
    "concave_down": {"substrate": [(1, 0, 0), (1, 0, 1)], "mover": (0, 0, 1), "target": (0, 0, 0), "sign": -1},
    # hanging on a wall (y = 1) with nothing below, pivot about a vertical edge to the next cell
    "wall_crawl": {"substrate": [(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)], "mover": (0, 0, 1), "target": (1, 0, 1), "sign": 1, "axis": "z"},
}


def magnet_sites(inset):
    """(30, 3) magnet positions in a unit cube's frame and (30,) bool, True for edge magnets.
    Edge magnets sit a quarter of the edge from its middle, inset by `inset` along both face
    normals; face magnets sit at face centres, inset along the normal."""
    points, edge = [], []
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        for su in (-0.5, 0.5):
            for sv in (-0.5, 0.5):
                for t in (-0.25, 0.25):
                    p = np.zeros(3)
                    p[axis] = t
                    p[u] = su - math.copysign(inset, su)
                    p[v] = sv - math.copysign(inset, sv)
                    points.append(p)
                    edge.append(True)
    for axis in range(3):
        for s in (-0.5, 0.5):
            p = np.zeros(3)
            p[axis] = s - math.copysign(inset, s)
            points.append(p)
            edge.append(False)
    return np.array(points), np.array(edge)


def scene_xml(cube, cells, mover_index, friction, anchored=False, axis="y"):
    """MJCF for cubes centred at `cells` (N, 3) in metres; the cube at mover_index has the flywheel.
    anchored: the other cubes are fixed in place, as if part of a large lattice. axis: the
    flywheel's axis, "y" or "z"."""
    h = cube.side / 2
    inertia = cube.flywheel_inertia()
    fly_mass = cube.flywheel_share * cube.mass
    bodies = []
    for i, c in enumerate(cells):
        shell = cube.mass - (fly_mass if i == mover_index else 0.0)
        flywheel = ""
        if i == mover_index:
            flywheel = (
                f'<body name="fly"><joint name="fly" type="hinge" axis="{"0 1 0" if axis == "y" else "0 0 1"}"/>'
                f'<inertial pos="0 0 0" mass="{fly_mass}" diaginertia="{inertia / 2} {inertia if axis == "y" else inertia / 2} {inertia / 2 if axis == "y" else inertia}"/></body>'
            )
        joint = "" if anchored and i != mover_index else "<freejoint/>"
        bodies.append(f'<body name="c{i}" pos="{c[0]} {c[1]} {c[2]}">{joint}<geom type="box" size="{h} {h} {h}" mass="{shell}"/>{flywheel}</body>')
    return f"""<mujoco>
  <option timestep="{DT}" integrator="implicitfast" cone="elliptic" impratio="10" gravity="0 0 {-G}"/>
  <default><geom friction="{friction} 0.005 0.0001" condim="4" solref="0.002 1" solimp="0.95 0.99 0.0005"/></default>
  <worldbody><geom type="plane" size="2 2 0.1"/>{''.join(bodies)}</worldbody>
</mujoco>"""


class Magnets:
    """Pairwise magnet forces between cubes and toward a steel floor. Only site pairs closer than
    CUTOFF (in d0) interact; the list of such pairs is rebuilt every REFRESH steps."""

    CUTOFF = 6.0  # d0; the force there is 0.4% of its peak
    REFRESH = 10

    def __init__(self, cube, n, strength_scale):
        d0 = cube.magnet_range * cube.side
        inset = d0 / math.sqrt(3) / 2 / cube.side  # twins in face contact sit at the force peak
        sites, edge = magnet_sites(inset)
        weight = cube.mass * G
        pull = np.where(edge, cube.edge_pull, cube.face_pull) * weight  # (30,)
        self.local = sites * cube.side
        self.d0 = d0
        self.n = n
        self.strength = (pull[None, :] * strength_scale[:, None]).reshape(-1)  # (n * 30,)
        self.owner = np.repeat(np.arange(n), len(sites))
        self.floor = cube.floor_pull * self.strength / SHAPE_PEAK
        self.steps = 0
        self.first = self.second = np.zeros(0, dtype=int)
        self.coef = np.zeros(0)

    def refresh(self, flat):
        r = flat[None, :, :] - flat[:, None, :]
        near = ((r * r).sum(-1) < (self.CUTOFF * self.d0) ** 2) & (self.owner[:, None] < self.owner[None, :])
        self.first, self.second = np.nonzero(near)
        self.coef = np.sqrt(self.strength[self.first] * self.strength[self.second]) / SHAPE_PEAK / self.d0

    def wrench(self, pos, rot):
        """pos (n, 3) cube centres, rot (n, 3, 3). Returns (n, 6) force and torque about the centres."""
        arm = np.einsum("nij,sj->nsi", rot, self.local)  # (n, 30, 3)
        flat = (pos[:, None, :] + arm).reshape(-1, 3)
        if self.steps % self.REFRESH == 0:
            self.refresh(flat)
        self.steps += 1
        force = np.zeros_like(flat)
        if len(self.first):
            r = flat[self.second] - flat[self.first]
            f = (self.coef / (1 + (r * r).sum(-1) / self.d0**2) ** 2)[:, None] * r
            np.add.at(force, self.first, f)
            np.add.at(force, self.second, -f)
        if self.floor.any():
            rz = -2 * np.maximum(flat[:, 2], 0.0)
            force[:, 2] += self.floor * (rz / self.d0) / (1 + rz**2 / self.d0**2) ** 2
        force = force.reshape(self.n, -1, 3)
        torque = np.stack([arm[..., 1] * force[..., 2] - arm[..., 2] * force[..., 1], arm[..., 2] * force[..., 0] - arm[..., 0] * force[..., 2], arm[..., 0] * force[..., 1] - arm[..., 1] * force[..., 0]], -1)
        return np.concatenate([force.sum(1), torque.sum(1)], axis=1)


def aligned(rot):
    return bool((np.abs(rot).max(axis=1) > ALIGNED).all())


@dataclass(frozen=True)
class Jitter:
    """Trial-to-trial variation: manufacturing spread and placement error."""

    friction: tuple = (0.3, 0.7)
    magnet: float = 0.25  # per-cube strength x U(1 - m, 1 + m)
    brake: float = 0.4  # brake torque x U(1 - b, 1 + b)
    speed: float = 0.02  # flywheel speed x N(1, s)
    mass: float = 0.05
    offset: float = 0.03  # L, the mover's placement error along y and its gap to neighbours


NOMINAL = Jitter(friction=(0.5, 0.5), magnet=0.0, brake=0.0, speed=0.0, mass=0.0, offset=0.0)


def judge(final, rot, place, start, target, slack):
    """Outcome of a move from where the cubes rest. final (n, 3) cells and rot (n, 3, 3), mover
    last; place (n, 3) where the cubes were put (centres, in L); slack: the mover's placement
    error allowed around its start cell."""
    mover = len(final) - 1
    knocked = any(np.abs(final[i] - (place[i] - np.array([0, 0, 0.5]))).max() > DISTURBED or not aligned(rot[i]) for i in range(mover))
    upright = aligned(rot[mover])
    outcome = "off"
    if knocked:
        outcome = "knock"
    elif upright and np.abs(final[mover] - target).max() < LANDED:
        outcome = "success"
    elif upright and np.abs(final[mover] - start).max() < LANDED + slack:
        outcome = "stay"
    return outcome


def trial(cube, primitive, omega, rng, jitter=Jitter(), anchored=False, trace=None, watch=None, retries=()):
    """Run one primitive at flywheel speed omega (rad/s, the policy's knob; capped at the motor's
    max). Returns a dict with outcome in {"success", "stay", "off", "knock"}:
    success: the mover sits aligned in the target cell and no neighbour moved;
    stay: the mover is aligned in its start cell and no neighbour moved (a harmless failure);
    knock: a neighbour was displaced or turned; off: anything else. watch(model, data), if given,
    is called after every step (for rendering).

    retries: flywheel speeds for further attempts. An attempt that ends in "stay" (a module can
    tell from its accelerometer: at rest, same face down) is followed by the next speed, in the
    same scene: the cubes keep their friction, magnets, mass and wherever the failed attempt left
    them, the flywheel is spun up again, and the brake torque and speed error are drawn anew.
    Knocks are judged against where the neighbours were first put, so drift over attempts counts."""
    spec = PRIMITIVES[primitive]
    cells = np.array(spec["substrate"] + [spec["mover"]], dtype=float)
    n = len(cells)
    mover = n - 1
    start = cells[mover].copy()
    target = np.array(spec["target"], dtype=float)
    centres = cells + np.array([0, 0, 0.5])
    lo, hi = jitter.friction
    friction = rng.uniform(lo, hi) if hi > lo else lo
    place = centres.copy()
    if jitter.offset > 0:
        lateral = 1 if spec.get("axis", "y") == "y" else 2  # along the pivot axis, no neighbour touches
        place[mover, lateral] += rng.uniform(-jitter.offset, jitter.offset)
        gaps = set()
        for i in range(mover):  # a small gap from face neighbours, once per direction (not from the floor)
            away = centres[mover] - centres[i]
            if np.abs(away).sum() == 1 and away[2] == 0:
                gaps.add(tuple(away))
        for g in gaps:
            place[mover] += np.array(g) * rng.uniform(0, jitter.offset)
    mass_scale = 1 + rng.uniform(-jitter.mass, jitter.mass) if jitter.mass > 0 else 1.0
    body = Cube(**{**cube.__dict__, "mass": cube.mass * mass_scale})
    model = mujoco.MjModel.from_xml_string(scene_xml(body, place * cube.side, mover, friction, anchored, spec.get("axis", "y")))
    data = mujoco.MjData(model)
    strength = 1 + rng.uniform(-jitter.magnet, jitter.magnet, size=n) if jitter.magnet > 0 else np.ones(n)
    magnets = Magnets(body, n, strength)
    fly = model.joint("fly")
    flywheel_inertia = body.flywheel_inertia()
    body_ids = np.array([model.body(f"c{i}").id for i in range(n)])
    outcome = "stay"
    attempts = 0
    for attempt_omega in (omega, *retries):
        if outcome != "stay":
            break
        attempts += 1
        brake = cube.brake_torque * (1 + rng.uniform(-jitter.brake, jitter.brake) if jitter.brake > 0 else 1.0)
        speed = spec["sign"] * min(abs(attempt_omega), cube.max_omega()) * (1 + rng.normal(0, jitter.speed) if jitter.speed > 0 else 1.0)
        data.qvel[fly.dofadr[0]] = speed
        began = data.time
        still = 0.0
        while data.time < began + HORIZON:
            since = data.time - began
            braking = BRAKE_AT <= since < BRAKE_AT + BRAKE_HOLD
            stop = 0.5 * flywheel_inertia * data.qvel[fly.dofadr[0]] / DT  # torque that would halt the flywheel in about two steps
            data.qfrc_applied[fly.dofadr[0]] = -np.clip(stop, -brake, brake) if braking else 0.0
            pos = data.xpos[body_ids]
            rot = data.xmat[body_ids].reshape(n, 3, 3)
            data.xfrc_applied[body_ids] = magnets.wrench(pos, rot)
            mujoco.mj_step(model, data)
            since = data.time - began
            if watch is not None:
                watch(model, data)
            if trace is not None and int(round(data.time / DT)) % 50 == 0:
                trace.append((data.time, data.xpos[body_ids].copy() / cube.side, data.xmat[body_ids].reshape(n, 3, 3).copy()))
            if since > BRAKE_AT + 0.05:
                v = np.array([data.cvel[b] for b in body_ids])  # (n, 6): angular then linear
                calm = np.abs(v[:, 3:]).max() < 0.2 * cube.side and np.abs(v[:, :3]).max() < 0.5
                still = still + DT if calm else 0.0
                resting = still >= SETTLE and aligned(data.xmat[body_ids[mover]].reshape(3, 3))
                stuck = still >= 3 * SETTLE  # resting off the lattice, e.g. leaning
                gone = np.abs(data.xpos[body_ids[mover]] / cube.side - (start + target) / 2 - np.array([0, 0, 0.5])).max() > 3
                shoved = mover > 0 and np.abs(data.xpos[body_ids[:mover]] / cube.side - place[:mover]).max() > 3 * DISTURBED  # a knock either way
                if (resting or stuck) and since > BRAKE_AT + BRAKE_HOLD or gone or shoved:
                    break
        final = data.xpos[body_ids] / cube.side - np.array([0, 0, 0.5])
        outcome = judge(final, data.xmat[body_ids].reshape(n, 3, 3), place, start, target, jitter.offset)
    return {"outcome": outcome, "attempts": attempts, "time": round(float(data.time), 3), "final": [round(float(v), 3) for v in final[mover]], "friction": round(friction, 3), "brake": round(brake, 4), "omega": round(abs(speed), 1)}
