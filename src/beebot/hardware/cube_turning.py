"""The hinge cube with one turning face: a face that turns about its normal and re-locks at quarter turns.

Proof of concept for runs/handoff/physics-primitive.md (the bond with open, locked and turning
states) and for the CAD in hardware/rotating_cube/. Everything not listed is cube_hinge's and
cube_swarm's (time step, contacts, magnet law, steel floor, placement error, jitter, the latched
edge pivot and its servo): TurningSwarm is a HingeSwarm, so the cubes still move by edge pivots.

Model:

1. Mass. The module's mass and inertia are the CAD's (hardware/rotating_cube/cube_full.py prints
   them; MASS, INERTIA, DISC_MASS, DISC_INERTIA below), not the hinge cube's 150 g. The magnets'
   pulls are kept in newtons (53 N a face), so in module weights they are weaker than the hinge
   cube's.
1. Turning face. Each cube has a disc on a hinge joint about the normal of one face (axis through
   the cube's centre; the CAD's 6803 bearing). The gearmotor sits in the body and turns the disc:
   a torque on the joint, clipped at Face.torque (jittered per cube), PD on the joint angle along a
   minimum-jerk reference at Face.rate. The gear train's reflected rotor inertia is the joint's
   armature; bearing, detent-ball and gearbox drag are a Coulomb torque Face.drag.
1. Detent. Four spring-loaded balls in the disc's rim drop into dimples every 90 degrees: a
   torque -detent sin(pi phi / capture) within `capture` degrees of a quarter turn, zero outside
   (the balls ride on the wall), jittered per cube.
1. Magnets. cube_swarm's 30 sites a cube (24 edge, 6 face centre) keep 1 - Face.share of their
   pull; the rest of each face's pull (share x 53 N) sits on four inner magnets on a circle of
   Face.radius, at 45 degrees, on every face, so that a neighbour's face mates with a disc at
   every quarter turn. On the turning face those four ride on the disc. The force law, range,
   steel floor and switching are cube.Magnets'; a closed face holds 53 N as before.
1. Contact. The disc stands Face.proud above its face on eight pads, so a bonded neighbour rides
   on the disc and clears the ring of the body around it (otherwise the inner magnets would press
   it onto the ring and the ring's friction would fight the motor). Torque passes from the disc to
   the neighbour by the pads' friction and the inner magnets' sideways pull only: no key.
1. A turn (TurningSwarm.turns). The cubes the disc carries (the neighbour on the face and whatever
   is bonded to it) switch their magnets off toward every other cube and the steel, except that
   the driver's disc magnets keep holding the neighbour. The disc turns a quarter; within
   Face.tolerance of the goal the magnets come back on (the edge magnets of the body's ring and
   the neighbour's face pull the pair square: the re-lock), then the motor lets go and the scene
   settles. A stall is driven back, as cube_hinge does.

Judged per quarter turn: reached (the disc is within RELOCK degrees of the quarter, the carried
cubes sit aligned in the cells the turn should put them in, nothing else moved), stayed (driven
back, everything where it was) or harmful (a carried cube dropped or sits elsewhere, another cube
was knocked, or the disc is stuck between quarters).

    from beebot.hardware.cube_turning import TurningSwarm
    swarm = TurningSwarm([(0, 0, 0), (1, 0, 0), (1, 0, 1)])  # every turning face looks along +x
    swarm.turn(1, 1, carried=[2])

Rigs (results in runs/handoff/rotating-cube-poc.md; every run of it is in scripts/cube_turning_runs.sh):
crank (a load carried round the driver's axis, with the hinge cube's way of doing it as the
baseline), cart (four square wheels), march (the cart's cubes walking instead), carry (how much
the disc bond holds while it turns).

    python -m beebot.hardware.cube_turning crank --seeds 20 --workers 5 --out runs/cube_turning/crank-turning.jsonl
    python -m beebot.hardware.cube_turning crank --seed 0 --video runs/handoff/media/rotating-cube-crank-best.mp4 --meshes hardware/rotating_cube
"""

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace

import mujoco
import numpy as np

from beebot.hardware import cube_swarm
from beebot.hardware.cube import DISTURBED, DT, G, SETTLE, SHAPE_PEAK, Jitter
from beebot.hardware.cube_hinge import Drive, HingeSwarm, servo_row, smooth
from beebot.hardware.cube_swarm import COLOURS, DESIGN, Film, SwarmMagnets, disturbed

# From the CAD (hardware/rotating_cube/cube_full.py, "mass and inertia" in its output).
MASS = 0.207  # kg, whole module
INERTIA = 0.189  # the module's moment of inertia about a centre axis / (MASS side^2); a uniform cube is 0.167
DISC_MASS = 0.0185  # kg: disc, its four magnets, ring board, retainer, detent balls and springs
DISC_INERTIA = 2.8e-6  # kg m2 about the disc's axis
RELOCK = 2.0  # degrees: a quarter turn has re-locked when the disc rests this close to it


def module(mass=MASS):
    """cube_swarm's 5 cm cube on steel at another mass, with the same magnets in newtons."""
    base = DESIGN.cube()
    return replace(base, mass=mass, edge_pull=base.edge_pull * base.mass / mass, face_pull=base.face_pull * base.mass / mass)


@dataclass(frozen=True)
class Face:
    torque: float = 0.3  # N m: the turning-face gearmotor's limit at the disc, the same as cube_hinge's edge motor (an N20's gearbox is rated about 0.2)
    spread: float = 0.2  # the limit is x U(1 - s, 1 + s) per cube when the swarm is jittered
    rate: float = 100.0  # deg/s: mean rate of the reference (cube_hinge: 150; at 150 the motor saturates lifting the load and overshoots)
    kp: float = 10.0  # N m / rad
    kd: float = 0.5  # N m s / rad
    detent: float = 0.08  # N m: peak detent torque (four 3 mm balls, 1 N springs, 18 mm out)
    capture: float = 3.6  # degrees either side of a quarter turn in which the balls sit in their dimples
    drag: float = 0.02  # N m: Coulomb drag of bearing, balls on the wall and the unpowered gearbox
    armature: float = 0.005  # kg m2: rotor inertia reflected through the gearbox
    share: float = 0.5  # of a face's pull that sits on the four inner magnets (the disc's, on the turning face); at 1/3 the disc bond slips in torsion before the motor stalls
    radius: float = 0.24  # L: the inner magnets' circle (12 mm)
    proud: float = 0.006  # L: the disc stands this far above its face (0.3 mm)
    pads: float = 0.30  # L: the pads' circle (15 mm)
    tolerance: float = 3.0  # degrees: arrived
    stall: float = 0.5  # s past the reference's end before the controller turns back
    near: float = 6.0  # degrees: a stall this close counts as arrived (the edge magnets' capture is about 7)
    hold: float = 0.1  # s: magnets on, then motor off after this long
    ring: bool = True  # False: no re-lock; the carried cubes stay switched off toward the others (a wheel that keeps turning)


AXES = np.eye(3)
MESH_FIXED = ["body", "epm_edge", "epm_inner", "axles", "hooks"]  # CAD parts drawn on the cube's body
MESH_DISC = ["disc", "epm_disc", "balls"]  # and on its disc
MESH_COLOURS = {"epm_edge": "0.55 0.25 0.65", "epm_inner": "0.75 0.30 0.55", "axles": "0.30 0.30 0.33", "hooks": "0.20 0.60 0.80", "disc": "0.93 0.93 0.90", "epm_disc": "0.90 0.35 0.45", "balls": "0.85 0.85 0.90"}


def inner_sites(side, inset, radius):
    """(24, 3) the four inner magnets of each face, faces ordered -x, +x, -y, +y, -z, +z."""
    points = []
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        for sign in (-1, 1):
            for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
                p = np.zeros(3)
                p[axis] = sign * (side / 2 - inset)
                p[u], p[v] = a * radius / math.sqrt(2), b * radius / math.sqrt(2)
                points.append(p)
    return np.array(points)


class TurningMagnets(SwarmMagnets):
    """cube_swarm's magnets plus four inner magnets a face; on a cube's turning face the inner
    four ride on the disc. turning: per cube (axis 0-2, sign). With share 0 the forces on the
    cubes are SwarmMagnets' exactly."""

    def __init__(self, cube, n, strength_scale, face, turning):
        super().__init__(cube, n, strength_scale)
        outer = self.strength.reshape(n, -1) * (1 - face.share)
        inset = self.d0 / math.sqrt(3) / 2
        closed = (8 * cube.edge_pull + cube.face_pull) * cube.mass * G  # N: a closed face
        inner = np.full((n, 24), face.share * closed / 4) * np.asarray(strength_scale)[:, None]
        self.local = np.concatenate([self.local, inner_sites(cube.side, inset, face.radius * cube.side)])
        self.per = len(self.local)
        self.strength = np.concatenate([outer, inner], axis=1).reshape(-1)
        self.owner = np.repeat(np.arange(n), self.per)
        self.floor = cube.floor_pull * self.strength / SHAPE_PEAK
        self.locals = np.tile(self.local, (n, 1, 1))
        self.disc = np.zeros((n, self.per), dtype=bool)  # sites that ride on the cube's disc
        for i, (axis, sign) in enumerate(turning):
            first = 30 + 4 * (2 * axis + (sign > 0))
            self.disc[i, first : first + 4] = True
            self.locals[i, first : first + 4, axis] += sign * face.proud * cube.side
        self.grip = np.zeros(n, dtype=bool)  # discs whose magnets stay on through a cut (they are turning a neighbour)

    def refresh(self, flat):
        sites = flat.reshape(self.n, -1, 3)
        centres = sites.mean(1)
        gap = centres[None, :, :] - centres[:, None, :]
        a, b = np.nonzero(np.triu((gap * gap).sum(-1) < self.reach**2, 1))
        r = sites[b][:, None, :, :] - sites[a][:, :, None, :]
        pair, i, j = np.nonzero((r * r).sum(-1) < (self.CUTOFF * self.d0) ** 2)
        a, b = a[pair], b[pair]
        on = np.ones(len(a), dtype=bool)
        if self.cut is not None:
            on = ~self.cut[a, b] | (self.disc[a, i] & self.grip[a]) | (self.disc[b, j] & self.grip[b])
        self.first, self.second = (a * self.per + i)[on], (b * self.per + j)[on]
        if self.quiet is not None:
            on = ~(self.quiet[self.first] | self.quiet[self.second])
            self.first, self.second = self.first[on], self.second[on]
        self.coef = np.sqrt(self.strength[self.first] * self.strength[self.second]) / self.d0 / SHAPE_PEAK

    def wrench(self, pos, rot, disc_rot=None):
        """pos (n, 3), rot (n, 3, 3) of the cubes and disc_rot (n, 3, 3) of their discs. Returns
        the wrenches on the cube bodies and on the discs, (n, 6) each, about the cubes' centres."""
        arm = np.einsum("nij,nsj->nsi", rot, self.locals)
        if disc_rot is not None:
            arm = np.where(self.disc[..., None], np.einsum("nij,nsj->nsi", disc_rot, self.locals), arm)
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
        both = np.concatenate([force, np.cross(arm, force)], axis=-1)
        on_disc = self.disc[..., None]
        return (both * ~on_disc).sum(1), (both * on_disc).sum(1)


class Spin:
    """One disc's controller for one turn of `quarters` quarter turns (signed)."""

    def __init__(self, swarm, cube, quarters, carried):
        self.cube = cube
        self.quarters = quarters
        self.carried = list(carried)
        self.start = float(swarm.data.qpos[swarm.disc_qpos[cube]])
        self.home = self.start
        self.end = (swarm.quarter[cube] + quarters) * math.pi / 2
        self.duration = abs(math.degrees(self.end - self.start)) / swarm.face.rate
        self.phase = "go"
        self.clock = 0.0
        self.arrived = None
        self.outcome = "go"
        self.peak = 0.0

    def torque(self, swarm, dt):
        face = swarm.face
        self.clock += dt
        angle = float(swarm.data.qpos[swarm.disc_qpos[self.cube]])
        rate = float(swarm.data.qvel[swarm.disc_dofs[self.cube]])
        duration = max(abs(self.end - self.start) / math.radians(face.rate), 1e-3)
        near = abs(angle - self.end) < math.radians(face.tolerance) and abs(rate) < 0.3  # there and nearly still
        if self.phase in ("go", "back") and near and self.clock >= duration:
            self.arrived = self.clock
            self.outcome = "arrived" if self.phase == "go" else "returned"
            self.phase = "hold"
        elif self.phase == "go" and self.clock > duration + face.stall and abs(angle - self.end) < math.radians(face.near):
            self.arrived = self.clock
            self.outcome = "near"
            self.phase = "hold"
        elif self.phase == "go" and self.clock > duration + face.stall:
            self.phase, self.start, self.end, self.clock = "back", angle, self.home, 0.0
            self.outcome = "stalled"
            duration = max(abs(self.end - self.start) / math.radians(face.rate), 1e-3)
        elif self.phase == "back" and self.clock > duration + face.stall:
            self.phase = "hold"
            self.arrived = self.clock
            self.outcome = "stuck"
        u, du = smooth(self.clock / duration)
        target, target_rate = self.start + (self.end - self.start) * u, (self.end - self.start) * du / duration
        if self.phase in ("hold", "magnets"):
            target, target_rate = (angle if self.outcome == "stuck" else self.end), 0.0
        wanted = face.kp * (target - angle) + face.kd * (target_rate - rate)
        self.peak = max(self.peak, abs(wanted))
        limit = swarm.motor[self.cube]
        return float(np.clip(wanted, -limit, limit))


def quarter_turn(vector, axis, quarters):
    """An integer vector turned `quarters` quarter turns (right-handed) about an integer unit axis."""
    v, a = np.array(vector), np.array(axis)
    for _ in range(quarters % 4):
        v = a * (a @ v) + np.cross(a, v)
    return v


class TurningSwarm(HingeSwarm):
    """HingeSwarm whose cubes are the CAD's module (mass) and carry a turning face (module
    docstring). turning: per cube (axis 0, 1 or 2 for x, y, z in the cube's frame, sign); default
    +x for all. face: Face. The jitter is cube_swarm's, plus the motor limit and detent torque per
    cube."""

    def __init__(self, cells, turning=None, face=Face(), drive=Drive(), cube=None, meshes=None, **options):
        cells = [tuple(c) for c in cells]
        self.face = face
        self.meshes = meshes  # a directory with the CAD's cube_full_<part>.stl: the cubes are drawn as those parts (looks only)
        self.turning = [(0, 1)] * len(cells) if turning is None else [tuple(t) for t in turning]
        self.ready = False
        super().__init__(cells, drive=drive, cube=cube or module(), **options)

    def scene_xml(self, centres):
        cube, face = self.cube, self.face
        h = cube.side / 2
        bodies = []
        for i, (c, mass) in enumerate(zip(centres, self.mass)):
            fly_mass = cube.flywheel_share * mass  # no flywheel in this cube: an inner mass the base class keeps braked
            fly_inertia = fly_mass * (cube.flywheel_radius * cube.side) ** 2
            shell = mass - fly_mass - DISC_MASS
            inertia = INERTIA * mass * cube.side**2 - fly_inertia - DISC_INERTIA
            axis, sign = self.turning[i]
            normal = sign * AXES[axis]
            u, v = [AXES[a] for a in range(3) if a != axis]
            pad_r = 0.04 * cube.side
            pads = ""
            for k in range(8):
                angle = math.pi / 4 * k + math.pi / 8
                p = normal * (h + face.proud * cube.side - pad_r) + face.pads * cube.side * (math.cos(angle) * u + math.sin(angle) * v)
                pads += f'<geom name="p{i}_{k}" type="sphere" size="{pad_r}" pos="{p[0]} {p[1]} {p[2]}" rgba="0.1 0.1 0.1 1"/>'
            shown, shown_disc = "", ""
            if self.meshes:  # the CAD's frame: z = 0..50 mm, the turning face +z
                frame = np.column_stack([u, v, normal]) if np.linalg.det(np.column_stack([u, v, normal])) > 0 else np.column_stack([v, u, normal])
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, frame.reshape(-1))
                at = -frame @ np.array([0, 0, h])
                pose = f'pos="{at[0]} {at[1]} {at[2]}" quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}" contype="0" conaffinity="0"'
                tint = COLOURS[i % len(COLOURS)]
                shown = "".join(f'<geom type="mesh" mesh="{name}" {pose} rgba="{tint if name == "body" else MESH_COLOURS[name]} 1"/>' for name in MESH_FIXED)
                shown_disc = "".join(f'<geom type="mesh" mesh="{name}" {pose} rgba="{MESH_COLOURS[name]} 1"/>' for name in MESH_DISC)
            alpha = 0 if self.meshes else 1
            centre = normal * (h - 0.0048)
            look = normal * (h - 0.0009)
            mark = look + 0.2 * cube.side * u
            size = [0.0015 if a == axis else 0.2 * cube.side if AXES[a] @ u else 0.003 for a in range(3)]
            disc = (
                f'<body name="d{i}"><joint name="t{i}" type="hinge" axis="{normal[0]} {normal[1]} {normal[2]}" armature="{face.armature}"/>'
                f'<inertial pos="{centre[0]} {centre[1]} {centre[2]}" mass="{DISC_MASS}" diaginertia="{DISC_INERTIA} {DISC_INERTIA} {DISC_INERTIA}"/>'
                f'{pads}{shown_disc}<geom type="cylinder" size="0.0178 0.001" pos="{look[0]} {look[1]} {look[2]}" zaxis="{normal[0]} {normal[1]} {normal[2]}" contype="0" conaffinity="0" rgba="0.95 0.95 0.95 {alpha}"/>'
                f'<geom type="box" size="{size[0]} {size[1]} {size[2]}" pos="{mark[0]} {mark[1]} {mark[2]}" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 {alpha}"/></body>'
            )
            bodies.append(
                f'<body name="c{i}" pos="{c[0]} {c[1]} {c[2]}"><freejoint name="c{i}"/>'
                f'<inertial pos="0 0 0" mass="{shell}" diaginertia="{inertia} {inertia} {inertia}"/>'
                f'<geom type="box" size="{h} {h} {h}" rgba="{COLOURS[i % len(COLOURS)]} {alpha}"/>{shown}'
                f'<body name="fly{i}"><joint name="fly{i}" type="ball"/><inertial pos="0 0 0" mass="{fly_mass}" diaginertia="{fly_inertia} {fly_inertia} {fly_inertia}"/></body>'
                f"{disc}</body>"
            )
        meshes = "".join(f'<mesh name="{name}" file="{os.path.join(self.meshes, "cube_full_" + name + ".stl")}" scale="0.001 0.001 0.001"/>' for name in MESH_FIXED + MESH_DISC) if self.meshes else ""
        solref = f"{self.drive.stiffness} 1"
        latches = "".join(f'<connect name="latch{i}{end}" body1="c{i}" anchor="0 0 0" active="false" solref="{solref}"/>' for i in range(self.n) for end in "ab")
        return f"""<mujoco>
  <option timestep="{DT}" integrator="implicitfast" cone="elliptic" impratio="10" gravity="0 0 {-G}"/>
  <visual><global offwidth="1920" offheight="1080"/><headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5"/></visual>
  <asset><texture name="grid" type="2d" builtin="checker" rgb1="0.24 0.26 0.29" rgb2="0.20 0.22 0.25" width="64" height="64"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="{1 / cube.side} {1 / cube.side}"/>{meshes}</asset>
  <default><geom friction="{self.friction} 0.005 0.0001" condim="4" solref="0.002 1" solimp="0.95 0.99 0.0005"/></default>
  <worldbody><geom type="plane" size="2 2 0.1" pos="{h} {h} 0" material="grid"/>
    <light pos="0 -1 2" dir="0 0.4 -1" diffuse="0.6 0.6 0.6"/>
    {''.join(bodies)}</worldbody>
  <equality>{latches}</equality>
</mujoco>"""

    def setup(self):
        """Once the base class has built the scene: the magnets with inner sites, the discs."""
        n, model = self.n, self.model
        scale = self.magnets.strength.reshape(n, -1)[:, 0] / (self.cube.edge_pull * self.cube.mass * G)
        self.magnets = TurningMagnets(self.cube, n, scale, self.face, self.turning)
        self.discs = np.array([model.body(f"d{i}").id for i in range(n)])
        self.disc_dofs = np.array([model.joint(f"t{i}").dofadr[0] for i in range(n)])
        self.disc_qpos = np.array([model.joint(f"t{i}").qposadr[0] for i in range(n)])
        self.pad_geoms = np.array([[model.geom(f"p{i}_{k}").id for k in range(8)] for i in range(n)])
        self.normals = np.array([sign * AXES[axis] for axis, sign in self.turning])
        self.quarter = np.zeros(n, dtype=int)  # the quarter turn each disc is locked at
        self.command = np.zeros(n)  # N m: the motors' torques on the discs
        jittered = self.jitter.magnet > 0
        self.motor = self.face.torque * (1 + self.rng.uniform(-self.face.spread, self.face.spread, size=n) if jittered else np.ones(n))
        self.detent = self.face.detent * (1 + self.rng.uniform(-0.25, 0.25, size=n) if jittered else np.ones(n))
        self.ready = True

    def passive(self):
        """(n,) N m on the disc joints: detent and drag."""
        angle = self.data.qpos[self.disc_qpos]
        rate = self.data.qvel[self.disc_dofs]
        phi = angle - math.pi / 2 * np.round(angle / (math.pi / 2))
        width = math.radians(self.face.capture)
        tau = np.where(np.abs(phi) < width, -self.detent * np.sin(math.pi * phi / width), 0.0)
        stop = 0.5 * (self.face.armature + DISC_INERTIA) * rate / DT  # as the flywheel brake: the torque that would halt it in about two steps
        return tau - np.clip(stop, -self.face.drag, self.face.drag)

    def step(self, free=-1, brake=None):
        if not self.ready:
            self.setup()
        data = self.data
        spin = data.qvel[self.fly_dofs]
        stop = 0.5 * self.fly_inertia[:, None] * spin / DT
        size = np.linalg.norm(stop, axis=1, keepdims=True)
        data.qfrc_applied[self.fly_dofs] = -stop * np.minimum(1.0, self.cube.brake_torque / np.maximum(size, 1e-12))
        pos = data.xpos[self.bodies]
        on_cubes, on_discs = self.magnets.wrench(pos, data.xmat[self.bodies].reshape(self.n, 3, 3), data.xmat[self.discs].reshape(self.n, 3, 3))
        on_discs[:, 3:] -= np.cross(data.xipos[self.discs] - pos, on_discs[:, :3])  # about the disc's own centre of mass
        data.xfrc_applied[self.bodies] = on_cubes
        data.xfrc_applied[self.discs] = on_discs
        data.qfrc_applied[self.disc_dofs] = self.passive() + self.command
        mujoco.mj_step(self.model, data)
        self.steps += 1

    def actuate(self, spins, limit, cut=None, watch=None, quiet=None):
        """cube_hinge's pivot. The pads of the movers and of the cubes they lean on are drawn in
        first and the scene settles for 50 ms: cube_hinge drops the contact between a mover and
        what it leans on, and the pads are separate geoms that would keep it. It also matters
        physically: a cube standing on its proud disc sits 0.3 mm high and its magnets tilt it and
        its neighbour a quarter of a degree; latched like that, the next pivot jams (measured).
        The real cube needs the disc flush when it is not turning; that is not designed."""
        at = {tuple(int(v) for v in c): i for i, c in enumerate(self.reading.cells)}
        involved = set()
        for cube, _, _ in spins:
            involved.add(cube)
            involved.update(at[c] for c in self.pivots[cube].leans if c in at)
        geoms = self.pad_geoms[sorted(involved)].reshape(-1)
        kept = self.model.geom_pos[geoms].copy()
        self.model.geom_pos[geoms] *= 0.9
        self.run(0.05, watch)
        super().actuate(spins, limit, cut, watch, quiet)
        self.model.geom_pos[geoms] = kept
        self.run(SETTLE, watch)

    def turn(self, cube, quarters=1, carried=(), watch=None, floor_off=()):
        return self.turns([(cube, quarters, carried)], watch, floor_off)[0]

    def turns(self, orders, watch=None, floor_off=()):
        """Turn discs at once: orders [(cube, quarters, carried cubes)]. floor_off: cubes whose
        magnets are off toward the steel meanwhile, besides the carried ones (a cart's chassis).
        Returns one dict per order."""
        if not self.ready:
            self.setup()
        began = self.data.time
        before = self.lattice()
        centres_before, rot_before = self.poses()
        n, per, face = self.n, self.magnets.per, self.face
        floor = self.magnets.floor.copy()
        cut = np.zeros((n, n), dtype=bool)
        spins = [Spin(self, cube, quarters, carried) for cube, quarters, carried in orders]
        everything = sorted({j for s in spins for j in s.carried})
        for j in everything:
            cut[j] = cut[:, j] = True
        for s in spins:  # the cubes one disc carries stay bonded to each other
            cut[np.ix_(s.carried, s.carried)] = False
        for j in everything + list(floor_off):
            self.magnets.floor[j * per : (j + 1) * per] = 0.0
        for s in spins:
            self.magnets.grip[s.cube] = True
        self.magnets.switch(cut)
        horizon = max(2 * (s.duration + face.stall) for s in spins) + 1.0
        elapsed = 0.0
        over = set()
        while elapsed < horizon and len(over) < len(spins):
            for k, s in enumerate(spins):
                if k in over:
                    continue
                self.command[s.cube] = s.torque(self, DT)
                if s.phase == "hold" and s.clock - s.arrived > face.hold:
                    if face.ring:
                        for j in s.carried:
                            cut[j] = cut[:, j] = False
                            if j not in floor_off:
                                self.magnets.floor[j * per : (j + 1) * per] = floor[j * per : (j + 1) * per]
                        self.magnets.switch(cut if cut.any() else None)
                    s.phase, s.arrived = "magnets", s.clock
                elif s.phase == "magnets" and s.clock - s.arrived > face.hold:
                    over.add(k)
                    self.command[s.cube] = 0.0
            self.step()
            elapsed += DT
            if watch is not None:
                watch(self)
        for s in spins:
            if s.outcome in ("arrived", "near"):
                self.quarter[s.cube] += s.quarters
            elif s.phase not in ("hold", "magnets"):
                s.outcome = "timeout"
        self.command[:] = 0.0
        if face.ring:
            self.magnets.floor[:] = floor
            self.magnets.grip[:] = False
            self.magnets.switch(None)
        self.run(SETTLE, watch)
        self.rest(watch=watch)
        after = self.lattice()
        centres, rot = self.poses()
        others = ~np.isin(np.arange(n), everything)
        moved = np.abs(centres - centres_before).max(axis=1)
        knocked = tuple(int(j) for j in np.nonzero(others & ((moved > DISTURBED) | disturbed(rot_before, rot)))[0]) if face.ring else ()
        rows = []
        for s in spins:
            angle = math.degrees(float(self.data.qpos[self.disc_qpos[s.cube]]))
            error = angle - 90.0 * self.quarter[s.cube]
            axis = np.rint(rot_before[s.cube] @ self.normals[s.cube]).astype(int)  # (x, y, z)
            done = s.outcome in ("arrived", "near")
            placed = True
            for j in s.carried:
                offset = before.cells[j][::-1] - before.cells[s.cube][::-1]
                expected = before.cells[s.cube][::-1] + (quarter_turn(offset, axis, s.quarters) if done else offset)
                placed = placed and not after.off[j] and tuple(after.cells[j][::-1]) == tuple(expected)
            relocked = done and abs(error) < RELOCK and placed
            detail = "reached" if relocked else "stayed" if s.outcome == "returned" and placed else "jam" if placed else "dropped"
            if knocked:
                detail = "knock"
            rows.append(
                {
                    "cube": s.cube,
                    "quarters": s.quarters,
                    "servo": s.outcome,
                    "detail": detail,
                    "outcome": detail if detail in ("reached", "stayed") else "harmful",
                    "error": round(error, 2),
                    "peak": round(s.peak, 3),
                    "knocked": knocked,
                    "sim_seconds": round(float(self.data.time - began), 3),
                }
            )
        return rows


class FreeHinge(HingeSwarm):
    """The plain hinge cube, allowed the two half turns cube_swarm's rule table has no name for
    (down and under an overhang, and up from under it), so that a cube can circle another. The
    latch and servo are cube_hinge's; only the planner's refusal is lifted. The baseline's best
    case: these moves have never been measured in the store or walk runs."""

    def __init__(self, cells, **options):
        super().__init__(cells, **options)
        self.speeds["under"] = 0.0  # no flywheel speed: the base class looks one up by name

    def plan(self, cube, direction, reading=None):
        named = cube_swarm.primitive_name

        def any_name(h, s, quarters, on_floor, hanging):
            return named(h, s, quarters, on_floor, hanging) or ("under" if quarters == 2 else "")

        cube_swarm.primitive_name = any_name
        try:
            pivot = super().plan(cube, direction, reading)
        finally:
            cube_swarm.primitive_name = named
        return pivot


# ---------------------------------------------------------------------------------------------
# The task: a crank. A tower of three (cubes 0, 1 and 2, the top one the driver), an arm cube (3)
# on the driver's turning face, and a load cube (4) that starts on top of the driver. The load
# pivots onto the arm (edge hinge), is carried round the driver's axis `revolutions` times with a
# stop and a re-lock at every quarter turn (turning face), and pivots back (edge hinge).

CRANK = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 0, 1), (3, 0, 0)]
DRIVER = 2
ORBIT = [(-1, -1, 0), (-1, 1, 0), (1, 1, 0), (1, -1, 0)]  # the hinge cube's way round the arm: four half turns


def crank(seed, design="turning", revolutions=2, mass=None, face=Face(), drive=Drive(), jitter=Jitter(), retries=2, watch=None, built=None, meshes=None, reach=1):
    """One run of the task. design: "turning" (TurningSwarm), "hinge" (HingeSwarm as it is: its
    planner refuses to pass under the arm) or "hinge-free" (FreeHinge). mass: kg (default: the
    CAD's for turning, 150 g for the hinge cube). built(swarm) is called once the scene exists.
    reach: the arm's length in cubes; the load walks out to its end and circles there (at 1 it
    passes flush along the tower's face on the way round, at 2 it is clear of it).
    Returns a dict: success, harm, the steps."""
    cells = CRANK[:3] + [(2, 0, x) for x in range(1, reach + 1)] + [CRANK[4]]
    load = len(cells) - 1
    if design == "turning":
        swarm = TurningSwarm(cells, seed=seed, jitter=jitter, face=face, drive=drive, cube=module(mass or MASS), meshes=meshes)
    else:
        swarm = (FreeHinge if design == "hinge-free" else HingeSwarm)(cells, seed=seed, jitter=jitter, drive=drive, cube=module(mass) if mass else None)
    if built is not None:
        built(swarm)
    start = swarm.lattice()
    began = swarm.data.time
    script = [("pivot", (0, 0, 1))] * reach
    for _ in range(revolutions):
        script += [("quarter", 1)] * 4 if design == "turning" else [("pivot", d) for d in ORBIT]
    script += [("pivot", (0, 0, -1))] * reach
    steps = []
    alive = True
    for index, (kind, what) in enumerate(script):
        attempt = 0
        outcome = "stayed"
        while alive and outcome == "stayed" and attempt <= retries:
            if kind == "pivot":
                move = swarm.move(load, what, watch=watch)
                row = {"kind": "pivot", "primitive": move.primitive, "detail": move.detail, "outcome": move.outcome, "sim_seconds": move.sim_seconds, **(servo_row(swarm.hinges[0]) if swarm.hinges else {})}
            else:
                row = {"kind": "quarter", **swarm.turn(DRIVER, what, carried=list(range(3, len(cells))), watch=watch)}
            steps.append({"step": index, "attempt": attempt, **row})
            outcome = row["outcome"]
            attempt += 1
        alive = alive and outcome == "reached"
    end = swarm.lattice()
    home = bool((end.cells == start.cells).all() and not end.off.any() and not start.off.any())
    quarters = [s for s in steps if s["kind"] == "quarter"]
    asked = {s["step"] for s in steps if s["kind"] == "quarter"} if design == "turning" else {s["step"] for s in steps if reach <= s["step"] < len(script) - reach}
    return {
        "design": design,
        "seed": seed,
        "mass": swarm.cube.mass,
        "success": bool(alive and home),
        "harm": any(s["outcome"] == "harmful" for s in steps) or not home and alive,
        "failed_at": None if alive else steps[-1]["step"],
        "why": None if alive else steps[-1]["detail"],
        "quarters_asked": 4 * revolutions,
        "quarters_tried": len(asked),
        "quarters_reached": sum(s["outcome"] == "reached" for s in steps if s["step"] in asked),
        "relock_attempts": len(quarters),
        "relocks": sum(s["outcome"] == "reached" for s in quarters),
        "retries": sum(s["attempt"] > 0 for s in steps),
        "sim_seconds": round(float(swarm.data.time - began), 2),
        "friction": round(swarm.friction, 3),
        "steps": steps,
    }


# ---------------------------------------------------------------------------------------------
# Second rig: a cart. A chassis of six cubes (2 x 3); the four at its corners have their turning
# faces outward along y and a wheel cube on each (two cells apart: square wheels side by side would
# strike each other). All four discs turn a quarter at a time; the wheels are square, so each
# quarter turn rolls the cart one cell along x. Nothing re-locks to the body's ring while rolling
# (Face.ring False): the wheels hang on the disc bonds for the whole trip.

CART = [(0, 1, 0), (0, 1, 2), (0, 2, 0), (0, 2, 2), (0, 0, 0), (0, 0, 2), (0, 3, 0), (0, 3, 2), (0, 1, 1), (0, 2, 1)]
CART_TURNING = [(1, -1), (1, -1), (1, 1), (1, 1)] + [(0, 1)] * 6  # cubes 0-3 drive, 4-7 are wheels, 8 and 9 ride in the middle


def cart(seed, cells=10, mass=None, face=Face(), jitter=Jitter(), watch=None, built=None, meshes=None):
    """Roll the cart `cells` cells along +x, open loop. Returns a dict: success (every quarter turn
    done, the cart in one piece and level, within half a cell of `cells` along x and of its line
    across), harm (it came apart or a cube ended up off the floor), travel, sideways and yaw."""
    face = replace(face, ring=False)
    swarm = TurningSwarm(CART, turning=CART_TURNING, seed=seed, jitter=jitter, face=face, cube=module(mass or MASS), meshes=meshes)
    if built is not None:
        built(swarm)
    start = swarm.lattice()
    began = swarm.data.time
    steps = []
    alive = True
    while alive and len(steps) < cells:
        # +x travel: the wheels turn right-handed about +y; the discs at -y look along -y
        rows = swarm.turns([(0, -1, [4]), (1, -1, [5]), (2, 1, [6]), (3, 1, [7])], watch=watch, floor_off=range(len(CART)))
        centres, rot = swarm.poses()
        layout = (centres - centres.mean(0)) @ rot[8]
        together = bool(np.abs(layout - (start.cells[:, ::-1] - start.cells[:, ::-1].mean(0))).max() < 0.15)
        reached = all(r["servo"] in ("arrived", "near") for r in rows) and together
        steps.append({"step": len(steps), "servo": [r["servo"] for r in rows], "peak": max(r["peak"] for r in rows), "together": together, "x": round(float(centres[:, 0].mean() - start.cells[:, 2].mean()), 3)})
        alive = reached
    swarm.magnets.grip[:] = False
    swarm.magnets.floor[:] = 0.0  # the cart stays a cart: its magnets toward the steel stay off
    swarm.magnets.switch(None)
    swarm.run(0.3, watch)
    centres, rot = swarm.poses()
    travel = centres.mean(0) - start.cells[:, ::-1].mean(0)
    yaw = math.degrees(math.atan2(rot[8][1, 0], rot[8][0, 0]))
    layout = (centres - centres.mean(0)) @ rot[8]  # in the frame of a cube that rides in the middle
    together = bool(np.abs(layout - (start.cells[:, ::-1] - start.cells[:, ::-1].mean(0))).max() < 0.15)
    level = bool(np.abs(centres[:, 2]).max() < 0.1)
    arrived = together and level and abs(travel[0] - cells) < 0.5 and abs(travel[1]) < 0.5
    why = None if alive and arrived else "stalled" if not alive and steps[-1]["together"] else "apart" if not together or not level else "drift"
    return {
        "design": "turning-cart",
        "seed": seed,
        "mass": swarm.cube.mass,
        "success": bool(alive and arrived),
        "harm": not together or not level,
        "why": why,
        "failed_at": None if alive else len(steps) - 1,
        "quarters_asked": cells,
        "quarters_reached": sum(all(v in ("arrived", "near") for v in s["servo"]) and s["together"] for s in steps),
        "travel": round(float(travel[0]), 2),
        "sideways": round(float(travel[1]), 2),
        "yaw": round(yaw, 1),
        "sim_seconds": round(float(swarm.data.time - began), 2),
        "friction": round(swarm.friction, 3),
        "steps": steps,
    }


def march(seed, cells=10, mass=None, drive=Drive(), jitter=Jitter(), retries=2, watch=None, built=None):
    """The cart's baseline: the same ten cubes as plain hinge cubes cross `cells` cells along +x
    on their own, by floor rolls. Each cell of travel is three steps: the front column rolls one
    cell (its cubes together), then the middle one, then the back one. Returns a dict like cart()'s."""
    swarm = HingeSwarm(CART, seed=seed, jitter=jitter, drive=drive, cube=module(mass) if mass else None)
    if built is not None:
        built(swarm)
    start = swarm.lattice()
    began = swarm.data.time
    columns = [[i for i, c in enumerate(CART) if c[2] == x] for x in (2, 1, 0)]
    steps = []
    alive = True
    done = 0
    while alive and done < cells:
        for column in columns:
            waiting = list(column)
            attempt = 0
            while alive and waiting and attempt <= retries:
                moves = swarm.moves([(i, (0, 0, 1)) for i in waiting], watch=watch)
                steps.append({"cell": done, "attempt": attempt, "details": [m.detail for m in moves]})
                alive = not any(m.outcome == "harmful" for m in moves)
                waiting = [i for i, m in zip(waiting, moves) if m.outcome != "reached"]
                attempt += 1
            alive = alive and not waiting
        done += alive
    end = swarm.lattice()
    shift = end.cells - start.cells
    arrived = bool((shift == [0, 0, cells]).all() and not end.off.any())
    return {
        "design": "hinge-march",
        "seed": seed,
        "mass": swarm.cube.mass,
        "success": bool(alive and arrived),
        "harm": any(d not in ("reached", "stayed", "refused") for s in steps for d in s["details"]),
        "why": None if alive and arrived else steps[-1]["details"],
        "failed_at": None if alive else done,
        "quarters_asked": cells,
        "quarters_reached": done,
        "moves": sum(len(s["details"]) for s in steps),
        "travel": float(np.median(shift[:, 2])),
        "sim_seconds": round(float(swarm.data.time - began), 2),
        "friction": round(swarm.friction, 3),
        "steps": steps,
    }


def carry(seed, along=1, out=1, mass=None, face=Face(), jitter=Jitter(), watch=None, built=None, meshes=None):
    """How much the disc bond carries while it turns: a tower of three, the driver on top, `along`
    cubes in a row out along the driver's axis and `out` more cubes sideways from the last of
    them. One full turn in four quarters, no retries. Returns a dict like crank()'s."""
    cells = CRANK[:3] + [(2, 0, x) for x in range(1, along + 1)] + [(2, y, along) for y in range(1, out + 1)]
    swarm = TurningSwarm(cells, seed=seed, jitter=jitter, face=face, cube=module(mass or MASS), meshes=meshes)
    if built is not None:
        built(swarm)
    start = swarm.lattice()
    began = swarm.data.time
    steps = []
    alive = not start.off.any()
    while alive and len(steps) < 4:
        steps.append({"step": len(steps), **swarm.turn(DRIVER, 1, carried=list(range(3, len(cells))), watch=watch)})
        alive = steps[-1]["outcome"] == "reached"
    end = swarm.lattice()
    home = bool((end.cells == start.cells).all() and not end.off.any() and not start.off.any())
    return {
        "design": f"carry-{along}-{out}",
        "seed": seed,
        "mass": swarm.cube.mass,
        "success": bool(alive and home),
        "harm": start.off.any() or any(s["outcome"] == "harmful" for s in steps) or not home and alive,
        "failed_at": None if alive else len(steps) - 1,
        "why": None if alive else steps[-1]["detail"] if steps else "fell before the first turn",
        "quarters_asked": 4,
        "quarters_reached": sum(s["outcome"] == "reached" for s in steps),
        "relock_attempts": len(steps),
        "relocks": sum(s["outcome"] == "reached" for s in steps),
        "sim_seconds": round(float(swarm.data.time - began), 2),
        "friction": round(swarm.friction, 3),
        "steps": steps,
    }


def job(args):
    task, seed, options = args
    return {"cart": cart, "crank": crank, "carry": carry, "march": march}[task](seed, **options)


def tally(rows):
    """One line of rates for a list of crank(), cart(), carry() or march() rows. Re-locks are
    counted per quarter turn tried, retries included; a cart's wheels never re-lock."""
    n = len(rows)
    asked = sum(r["quarters_asked"] for r in rows)
    tried = sum(r.get("relock_attempts", 0) for r in rows)
    good = [r["sim_seconds"] for r in rows if r["success"]]
    return {
        "runs": n,
        "success": round(sum(r["success"] for r in rows) / n, 3),
        "harm": round(sum(bool(r["harm"]) for r in rows) / n, 3),
        "quarters_reached_of_asked": round(sum(r["quarters_reached"] for r in rows) / asked, 3),
        "relock_per_attempt": round(sum(r["relocks"] for r in rows) / tried, 3) if tried else None,
        "relock_attempts": tried,
        "seconds_per_success": round(float(np.mean(good)), 1) if good else None,
        "why": sorted({f"{r.get('why')}@{r.get('failed_at')}" for r in rows if not r["success"]}),
    }


def film(path, task, seed, options, slow=1.0, size=(1280, 720)):
    """Render one run to an MP4 (cube_swarm.Film)."""
    holder = {}

    def built(swarm):
        movie = Film(swarm, path, slow=slow, size=size)
        centres, _ = swarm.poses()
        movie.camera.lookat = [*(centres[:, :2].mean(0) * swarm.cube.side), 1.6 * swarm.cube.side]
        movie.camera.distance = 9 * swarm.cube.side
        movie.camera.azimuth, movie.camera.elevation = (35, -15) if task == "crank" else (60, -25)
        movie.still(swarm, 0.5)
        holder["film"] = movie

    def watch(swarm):
        movie = holder["film"]
        if task == "cart":
            centres, _ = swarm.poses()
            movie.camera.lookat[0] = float(centres[:, 0].mean() * swarm.cube.side)
        movie.frame(swarm)

    row = job((task, seed, {**options, "watch": watch, "built": built}))
    holder["film"].close()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["crank", "cart", "carry", "march"])
    parser.add_argument("--reach", type=int, default=1, help="crank: the arm's length in cubes")
    parser.add_argument("--along", type=int, default=1, help="carry: cubes out along the axis")
    parser.add_argument("--side", type=int, default=1, help="carry: cubes sideways from the last of them")
    parser.add_argument("--design", default="turning", choices=["turning", "hinge", "hinge-free"])
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    parser.add_argument("--mass", type=float, help="kg; default the CAD's (turning) or 0.15 (hinge)")
    parser.add_argument("--face", default="{}", help="JSON of Face fields")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", help="append one JSON line per run here")
    parser.add_argument("--video", help="render the first seed here instead")
    parser.add_argument("--slow", type=float, default=1.0)
    parser.add_argument("--meshes", help="video: draw the cubes as the CAD parts in this directory (hardware/rotating_cube)")
    args = parser.parse_args()
    options = {"mass": args.mass, "face": Face(**json.loads(args.face))}
    if args.task == "march":
        del options["face"]
    if args.task == "crank":
        options.update(design=args.design, reach=args.reach)
    if args.task == "carry":
        options.update(along=args.along, out=args.side)
    if args.video:
        rows = [film(args.video, args.task, args.seed, {**options, **({"meshes": args.meshes} if args.meshes and args.design == "turning" else {})}, args.slow)]
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(job, [(args.task, args.seed + k, options) for k in range(args.seeds)]))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a") as f:
            f.writelines(json.dumps({**r, "face": asdict(options["face"]) if "face" in options else None}, default=float) + "\n" for r in rows)
    for r in rows:
        print(f"seed {r['seed']:3d}  success {r['success']!s:5s} harm {r['harm']!s:5s}  quarters {r['quarters_reached']}/{r['quarters_asked']}  {r['sim_seconds']:6.1f} s  {r.get('why') or ''} {r.get('failed_at') if r.get('failed_at') is not None else ''}")
    print(json.dumps(tally(rows)))


if __name__ == "__main__":
    main()
