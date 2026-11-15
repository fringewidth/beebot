"""Layout B of hardware/MODULE_SPEC.md as a simplified MuJoCo model: one 50 mm cube with two turning
discs on opposite faces (local +x and -x), each a hinge about x with its own torque-limited drive.

A robot is a list of Module rows: a root, then children bonded to a face of an earlier module
(`face`, in the parent's frame) by one of their own faces (`mate`) at a quarter turn (`turn`).
`build` returns the MJCF; `Rig` runs it with a joint controller and records torques and bond loads.

    from beebot.hardware.module_b import Module, Rig
    rig = Rig([Module()])  # a lone module standing on its two rims, axis along world x
    rig.command(speeds={"0p": 5.8, "0n": 5.8})  # both discs at 5.8 rad/s
    rig.run(2.0)  # rolls along -y at about 0.15 m/s

Idealised (what this model leaves out):

1. Bonds are rigid: a bonded module is a fixed child body of its neighbour's body or disc. No
   connector, no capture, no compliance, no backlash, no bond strength; the loads a bond would
   carry are measured (Rig.bonds) and compared with property 5.5 afterwards, never enforced.
1. Neighbours that touch at build time do not collide with each other (their contacts are
   excluded), so the open conflict of property 3.5 (a rim 1 mm proud hits the neighbour on the
   adjacent face) is ignored. Modules that are not neighbours at build time collide normally.
1. The drive is an ideal torque source behind a PD loop at 1 kHz, clipped at +/-0.8 N m (holding);
   0.4 N m (continuous) is only recorded. Gearbox: a reflected inertia (ARMATURE) and a Coulomb
   drag (DRAG). No backlash, no compliance, no speed-torque curve, no heating, and the drive can be
   switched to zero torque (free), which a worm stage (property 4.4) could not do.
1. Absolute angle sensing is exact (the controller reads qpos).
1. Mass is a uniform 110 g box plus two 5 g discs (120 g, property 3.6); no battery or motor
   placement. Friction: discs (rim and face) 1.0, body 0.6, one coefficient each, no rolling
   resistance.
1. A disc is 52 mm across and 2.5 mm thick, its outer face flush with the body face. The lattice
   pitch is 52.4 mm: the four side faces carry connector plates 1.2 mm proud (spec 3.5's candidate
   fix), so a rim (1 mm proud radially) clears every neighbour, on a face or on a diagonal. The
   plates are drawn and collide only with `plates=True`; by default they are left out, which is what
   lets a lone module roll (with them it stands on its bottom plate; see the lone demo).
"""

import math
from dataclasses import dataclass

import mujoco
import numpy as np

SIDE = 0.050  # m: cube side
HALF = SIDE / 2
PITCH = 0.0524  # m: lattice pitch: the side faces' connector plates stand 1.2 mm proud (spec 3.5's candidate fix), so rims clear every neighbour
PLATE = (PITCH - SIDE) / 2  # m: plate stand-off
BODY_MASS = 0.110  # kg (guess: 120 g module, property 3.6, less two discs)
DISC_MASS = 0.005  # kg each (guess)
DISC_RADIUS = 0.026  # m: 52 mm disc, rim 1 mm proud of the side faces (property 3.5)
DISC_THICK = 0.0025  # m
PROUD = 0.0  # m: disc face out of the body face, axially (none: the disc is inside the outline)
RECESS = 0.0  # m: the disc's colliding face sits this far inside the body face
HOLD = 0.8  # N m: holding torque, the clip (property 4.1)
CONTINUOUS = 0.4  # N m: continuous target, recorded only
ARMATURE = 1e-3  # kg m2: gear train inertia at the disc (guess: N20-class rotor through about 400:1)
DRAG = 0.005  # N m: Coulomb drag of bearing and gearbox (guess)
KP, KV = 10.0, 0.25  # PD gains of the joint loop, N m / rad and N m s / rad
DT = 0.001  # s
DISC_FRICTION, BODY_FRICTION = 1.0, 0.6

AXES = "xyz"
FACES = ("+x", "-x", "+y", "-y", "+z", "-z")
COLOURS = ["0.85 0.55 0.25 1", "0.35 0.55 0.80 1", "0.45 0.70 0.45 1", "0.80 0.45 0.55 1", "0.65 0.55 0.80 1", "0.80 0.75 0.40 1"]


def unit(face):
    """'+y' -> (0, 1, 0)."""
    v = np.zeros(3)
    v[AXES.index(face[1])] = 1.0 if face[0] == "+" else -1.0
    return v


def face_of(vector):
    """The face whose outward normal is `vector` (a lattice unit vector)."""
    k = int(np.argmax(np.abs(vector)))
    if abs(abs(vector[k]) - 1) > 1e-6 or np.abs(vector).sum() > 1 + 1e-6:
        raise ValueError(f"not a face normal: {vector}")
    return ("+" if vector[k] > 0 else "-") + AXES[k]


def part_of(face):
    """Which rigid part owns a face: 'p' (+x disc), 'n' (-x disc) or 'b' (body)."""
    return {"+x": "p", "-x": "n"}.get(face, "b")


def rotation_about(axis, angle):
    axis = np.asarray(axis, float)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def _tangent(normal):
    k = int(np.argmax(np.abs(normal)))
    t = np.zeros(3)
    t[(k + 1) % 3] = 1.0
    return t


def bond_rotation(face, mate, turn):
    """Child frame in the parent's frame: the child's `mate` face looks back at the parent's `face`,
    then the child is turned `turn` quarter turns about the bond normal."""
    n_p, n_c = unit(face), unit(mate)
    t_p, t_c = _tangent(n_p), _tangent(n_c)
    child = np.column_stack([n_c, t_c, np.cross(n_c, t_c)])
    parent = np.column_stack([-n_p, t_p, np.cross(-n_p, t_p)])
    return np.round(rotation_about(n_p, turn * math.pi / 2) @ parent @ child.T)


def axes_rotation(axes):
    """('+z', '+x') -> the rotation whose local x points along world +z and local y along world +x."""
    x, y = unit(axes[0]), unit(axes[1])
    if abs(x @ y) > 1e-9:
        raise ValueError(f"axes not perpendicular: {axes}")
    return np.column_stack([x, y, np.cross(x, y)])


@dataclass
class Module:
    """One module. A root has parent None; `pos` (m) and `axes` (where its local x and y point in
    the world) place it, and `mount` names the part bolted to the world (None: free; 'b', 'p' or
    'n'). A child has a parent index, the parent's `face` it bonds to, its own `mate` face and a
    quarter `turn` (0-3) about the bond normal."""

    parent: int | None = None
    face: str = "+z"
    mate: str = "-z"
    turn: int = 0
    pos: tuple = (0.0, 0.0, DISC_RADIUS)
    axes: tuple = ("+x", "+y")
    mount: str | None = None
    rgba: str | None = None


def lattice(cells):
    """Modules from lattice cells, which is how the demos are written. Each cell is a dict:
    at (i, j, k) in pitches, axis (world direction of the module's local x: its disc axis),
    optional up (world direction of local y), parent (index; None for a root), and for a root
    optional z0 (m, height of the lattice origin's centre, default DISC_RADIUS) and mount.
    The bond face, mate face and turn are worked out from the geometry."""
    modules, world = [], []
    for cell in cells:
        axis = unit(cell["axis"])
        up = unit(cell["up"]) if "up" in cell else next(unit(f) for f in ("+y", "+z", "+x") if abs(unit(f) @ axis) < 0.5)
        rot = np.column_stack([axis, up, np.cross(axis, up)])
        at = np.asarray(cell["at"], float)
        if cell.get("parent") is None:
            z0 = cell.get("z0", DISC_RADIUS)
            names = (face_of(rot[:, 0]), face_of(rot[:, 1]))
            modules.append(Module(pos=tuple(at * PITCH + np.array([0, 0, z0])), axes=names, mount=cell.get("mount"), rgba=cell.get("rgba")))
        else:
            p = cell["parent"]
            p_at, p_rot = world[p]
            step = at - p_at
            face = face_of(p_rot.T @ step)
            relative = np.round(p_rot.T @ rot)
            mate = face_of(rot.T @ -step)
            turn = next((k for k in range(4) if np.allclose(bond_rotation(face, mate, k), relative)), None)
            if turn is None:
                raise ValueError(f"cell {len(modules)} cannot bond to {p}")
            modules.append(Module(parent=p, face=face, mate=mate, turn=turn, rgba=cell.get("rgba")))
        world.append((at, rot))
    return modules


def poses(modules):
    """World rotation and centre of every module with all joints at zero."""
    rots, centres = [], []
    for m in modules:
        if m.parent is None:
            rots.append(axes_rotation(m.axes))
            centres.append(np.asarray(m.pos, float))
        else:
            r_p = rots[m.parent]
            rots.append(r_p @ bond_rotation(m.face, m.mate, m.turn))
            centres.append(centres[m.parent] + r_p @ unit(m.face) * PITCH)
    return rots, centres


def _quat(rot):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(rot, float).reshape(-1))
    return " ".join(f"{v:.6g}" for v in q)


def _geoms(i, part, rgba, plates=False):
    if part == "b":
        out = f'<geom name="m{i}b" type="box" size="{HALF} {HALF} {HALF}" mass="{BODY_MASS}" rgba="{rgba}" friction="{BODY_FRICTION} 0.005 0.0001"/>\n'
        if plates:
            for axis in (1, 2):
                for sign in (1, -1):
                    pos, size = [0.0, 0.0, 0.0], [0.015, 0.015, 0.015]
                    pos[axis], size[axis] = sign * (HALF + PLATE / 2), PLATE / 2
                    out += f'<geom name="m{i}plate{axis}{sign}" type="box" size="{size[0]:.6g} {size[1]:.6g} {size[2]:.6g}" pos="{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}" mass="0" rgba="0.3 0.3 0.3 1" friction="{BODY_FRICTION} 0.005 0.0001"/>\n'
        return out
    sign = 1 if part == "p" else -1
    x = sign * (HALF - RECESS - DISC_THICK / 2)
    seen = sign * (HALF + 0.0003 - DISC_THICK / 2)  # drawn a hair proud so it shows; collides recessed
    mark = sign * (HALF + 0.0004)
    plain = 'contype="0" conaffinity="0" mass="0"'
    return (
        f'<geom name="m{i}{part}" type="cylinder" size="{DISC_RADIUS} {DISC_THICK / 2}" pos="{x:.6g} 0 0" euler="0 90 0" mass="{DISC_MASS}" rgba="0 0 0 0" friction="{DISC_FRICTION} 0.005 0.0001"/>\n'
        f'<geom type="cylinder" size="{DISC_RADIUS} {DISC_THICK / 2}" pos="{seen:.6g} 0 0" euler="0 90 0" rgba="0.25 0.25 0.28 1" {plain}/>\n'
        f'<geom type="box" size="0.0003 0.003 0.022" pos="{mark:.6g} 0 0.004" rgba="0.95 0.95 0.95 1" {plain}/>\n'
        f'<geom type="box" size="0.0003 0.007 0.003" pos="{mark:.6g} 0 0.023" rgba="0.95 0.95 0.95 1" {plain}/>\n'
    )


def _joint(i, part, inverted):
    axis = "-1 0 0" if inverted else "1 0 0"
    return f'<joint name="j{i}{part}" type="hinge" axis="{axis}" armature="{ARMATURE}" frictionloss="{DRAG}" solreffriction="0.002 1"/>\n'


def build(modules, scene="", extra="", timestep=DT, options="", plates=False):
    """MJCF for the modules on a floor. `scene` goes into the worldbody (objects, fixtures),
    `extra` after it (equality, sensors). Joints are named j<i>p and j<i>n (the +x and -x disc of
    module i; the angle of the disc on the body), motors m<i>p and m<i>n."""
    children = {}
    for k, m in enumerate(modules):
        if m.parent is not None:
            if m.parent >= k:
                raise ValueError("a parent must come before its children")
            children.setdefault((m.parent, part_of(m.face)), []).append(k)
            if any(modules[c].face == m.face for c in children[(m.parent, part_of(m.face))] if c != k):
                raise ValueError(f"face {m.face} of module {m.parent} bonded twice")

    def body(i, pos, quat, root_joint=""):
        m = modules[i]
        rgba = m.rgba or COLOURS[i % len(COLOURS)]
        anchor = (m.mount or "b") if m.parent is None else part_of(m.mate)
        order = {"b": ["b", "p", "n"], "p": ["p", "b", "n"], "n": ["n", "b", "p"]}[anchor]

        def part_xml(part, joint, head):
            inner = _geoms(i, part, rgba, plates)
            for c in children.get((i, part), []):
                cm = modules[c]
                inner += body(c, unit(cm.face) * PITCH, bond_rotation(cm.face, cm.mate, cm.turn))
            if part == anchor and anchor != "b":
                inner += part_xml("b", _joint(i, anchor, inverted=True), "")
            elif part == "b":
                for other in [x for x in "pn" if x != anchor]:
                    inner += part_xml(other, _joint(i, other, inverted=False), "")
            return f'<body name="m{i}{part}" {head}>\n{joint}{inner}</body>\n'

        return part_xml(anchor, root_joint, f'pos="{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}" quat="{_quat(quat)}"')

    roots = ""
    for i, m in enumerate(modules):
        if m.parent is None:
            joint = "<freejoint/>\n" if m.mount is None else ""
            roots += body(i, m.pos, axes_rotation(m.axes), joint)

    rots, centres = poses(modules)
    excluded = {(f"m{a}b", f"m{a}{p}") for a in range(len(modules)) for p in "pn"}  # parent-child filtering skips parts welded to the world
    for a in range(len(modules)):
        for b in range(a + 1, len(modules)):
            if np.linalg.norm(centres[a] - centres[b]) < PITCH * 1.01:
                for pa in "bpn":
                    for pb in "bpn":
                        excluded.add((f"m{a}{pa}", f"m{b}{pb}"))
    excludes = "".join(f'<exclude body1="{x}" body2="{y}"/>\n' for x, y in sorted(excluded))
    motors = "".join(f'<motor name="m{i}{p}" joint="j{i}{p}" ctrllimited="true" ctrlrange="-{HOLD} {HOLD}" forcelimited="true" forcerange="-{HOLD} {HOLD}"/>\n' for i in range(len(modules)) for p in "pn")
    return f"""<mujoco model="module_b">
<compiler angle="degree" autolimits="true"/>
<option timestep="{timestep}" integrator="implicitfast" {options or 'cone="elliptic" impratio="10"'}/>
<size memory="64M"/>
<visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>
<default><geom solref="0.004 1" solimp="0.95 0.99 0.001"/></default>
<asset>
<texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.82 0.80" rgb2="0.74 0.74 0.72" width="512" height="512"/>
<material name="grid" texture="grid" texrepeat="60 60" reflectance="0.05"/>
<texture name="sky" type="skybox" builtin="gradient" rgb1="0.92 0.93 0.95" rgb2="0.70 0.74 0.80" width="256" height="256"/>
</asset>
<worldbody>
<light pos="0.3 -0.6 1.2" dir="-0.2 0.4 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
<geom name="floor" type="plane" size="3 3 0.05" material="grid" friction="0.8 0.005 0.0001"/>
{scene}
{roots}</worldbody>
<contact>
{excludes}</contact>
<actuator>
{motors}</actuator>
{extra}
</mujoco>
"""


class Rig:
    """A built robot with its joint controller and recorder. Each motor is in one of three modes:
    position (minimum-jerk reference, PD), speed (reference angle advancing at a set rate, PD) or
    free (zero torque). Torques are recorded per motor; bond and bearing loads every `every` steps."""

    def __init__(self, modules, scene="", extra="", kp=KP, kv=KV, timestep=DT, every=5, plates=False, options=""):
        self.modules = modules
        self.xml = build(modules, scene, extra, timestep, options=options, plates=plates)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.names = [m.actuator(a).name[1:] for a in range(m.nu)]  # "3p"
        self.index = {n: a for a, n in enumerate(self.names)}
        self.qadr = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self.vadr = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self.kp = np.full(m.nu, kp)
        self.kv = np.full(m.nu, kv)
        self.free = np.zeros(m.nu, bool)
        self.cap = np.full(m.nu, HOLD)  # N m: a current limit per motor (gripping with a set torque, property 4.6)
        self.q0 = np.zeros(m.nu)
        self.q1 = np.zeros(m.nu)
        self.start = np.zeros(m.nu)
        self.span = np.ones(m.nu)
        self.rate = np.zeros(m.nu)
        self.rate0 = np.zeros(m.nu)
        self.ramp = np.zeros(m.nu)
        self.peak = np.zeros(m.nu)
        self.over = np.zeros(m.nu)
        self.limit = np.zeros(m.nu)
        self.every = every
        self.paths = {}  # motor index -> f(t) -> (angle, rate): a scripted reference
        self.steps = 0
        mujoco.mj_forward(m, self.data)
        self._bonds()
        self.bond_series = []  # per sample: (max bond moment, max bond tension, max bearing moment)

    def _bonds(self):
        """Bodies whose link to their parent is a bond (anchor of a child module) or a disc bearing."""
        m = self.model
        self.bond_bodies, self.bond_normals, self.bearing_bodies = [], [], []
        for i, mod in enumerate(self.modules):
            if mod.parent is not None:
                anchor = part_of(mod.mate)
                b = m.body(f"m{i}{anchor}").id
                if m.body_weldid[b] != 0:  # a bond inside a part bolted to the world is not measured
                    self.bond_bodies.append(b)
                    self.bond_normals.append(unit(mod.mate))
            for part in "pn":
                b = m.body(f"m{i}{part}").id
                jnt = m.body_jntadr[b]
                if jnt >= 0:
                    self.bearing_bodies.append((b, 1.0 if part == "p" else -1.0))
                else:
                    other = m.body(f"m{i}b").id  # the disc is the anchor: the body hangs on it
                    self.bearing_bodies.append((other, -1.0 if part == "p" else 1.0))

    def angle(self, name):
        return float(self.data.qpos[self.qadr[self.index[name]]])

    def angles(self):
        return {n: float(self.data.qpos[self.qadr[a]]) for a, n in enumerate(self.names)}

    def _reference(self, t):
        s = np.clip((t - self.start) / self.span, 0.0, 1.0)
        shape = 10 * s**3 - 15 * s**4 + 6 * s**5
        slope = (30 * s**2 - 60 * s**3 + 30 * s**4) / self.span
        moving = (t >= self.start) & (t <= self.start + self.span)
        tau = np.maximum(t - self.start, 0.0)
        ramp = np.minimum(tau, self.ramp)
        q = self.q0 + (self.q1 - self.q0) * shape + self.rate0 * tau + (self.rate - self.rate0) * (ramp**2 / (2 * np.maximum(self.ramp, 1e-9)) + (tau - ramp))
        qd = (self.q1 - self.q0) * slope * moving + self.rate0 + (self.rate - self.rate0) * np.where(self.ramp > 0, ramp / np.maximum(self.ramp, 1e-9), 1.0)
        return q, qd

    def command(self, targets=None, seconds=0.5, speeds=None, free=(), ramp=0.3, caps=None):
        """targets: {motor: angle in rad}, reached along a minimum-jerk path in `seconds`.
        speeds: {motor: rad/s}, the reference advancing from the present reference, its rate
        changing linearly over `ramp` seconds. free: motors set to zero torque. caps: {motor: N m}
        current limits (up to HOLD). Motors not named keep what they were doing."""
        for name, cap in (caps or {}).items():
            self.cap[self.index[name]] = min(cap, HOLD)
        t = self.data.time
        q_now, qd_now = self._reference(t)
        for a in range(self.model.nu):
            q = self.data.qpos[self.qadr[a]]
            if self.free[a] or a in self.paths or abs(q_now[a] - q) > 0.05:  # free, scripted, or held off its reference (a grip at its current limit)
                q_now[a], qd_now[a] = q, 0.0
        for name, angle in (targets or {}).items():
            a = self.index[name]
            self.paths.pop(a, None)
            self.q0[a], self.q1[a], self.start[a], self.span[a], self.free[a] = q_now[a], angle, t, max(seconds, 1e-6), False
            self.rate[a] = self.rate0[a] = self.ramp[a] = 0.0
        for name, rate in (speeds or {}).items():
            a = self.index[name]
            self.paths.pop(a, None)
            self.q0[a] = self.q1[a] = q_now[a]
            self.rate0[a] = 0.0 if self.free[a] else qd_now[a]
            self.start[a], self.span[a], self.rate[a], self.ramp[a], self.free[a] = t, 1.0, rate, ramp, False
        for name in free:
            self.free[self.index[name]] = True

    def follow(self, name, path):
        """Track a scripted reference f(t) -> (angle rad, rate rad/s); path None returns the motor
        to position mode at its present angle."""
        a = self.index[name]
        if path is None:
            self.paths.pop(a, None)
            self.hold_here([name])
        else:
            self.paths[a] = path
            self.free[a] = False

    def hold_here(self, names=None):
        """Position mode at the present angle for the named motors (all if None)."""
        for name in names or self.names:
            a = self.index[name]
            q = self.data.qpos[self.qadr[a]]
            self.q0[a], self.q1[a], self.start[a], self.span[a], self.free[a] = q, q, self.data.time, 1.0, False
            self.rate[a] = self.rate0[a] = self.ramp[a] = 0.0

    def step(self):
        d, m = self.data, self.model
        q_ref, qd_ref = self._reference(d.time)
        for a, path in self.paths.items():
            q_ref[a], qd_ref[a] = path(d.time)
        q, qd = d.qpos[self.qadr], d.qvel[self.vadr]
        tau = self.kp * (q_ref - q) + self.kv * (qd_ref - qd)
        tau = np.where(self.free, 0.0, np.clip(tau, -self.cap, self.cap))
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        force = np.abs(d.actuator_force)
        self.peak = np.maximum(self.peak, force)
        self.over += (force > CONTINUOUS + 1e-9) * m.opt.timestep
        self.limit += (force > HOLD - 1e-3) * m.opt.timestep
        self.steps += 1
        if self.steps % self.every == 0:
            self.bond_series.append(self.loads())

    def run(self, seconds, watch=None):
        end = self.data.time + seconds
        while self.data.time < end - 1e-9:
            before = self.data.time
            self.step()
            if self.data.time < before:
                raise RuntimeError(f"MuJoCo reset the simulation at t = {before:.3f} s (unstable)")
            if watch:
                watch(self)

    def loads(self):
        """(largest bond bending moment N m, largest bond tension N, largest bearing bending moment
        N m) at this instant, at the bond face centre / the disc's face centre."""
        m, d = self.model, self.data
        mujoco.mj_rnePostConstraint(m, d)
        bend, pull, bear = 0.0, 0.0, 0.0
        for b, n_local in zip(self.bond_bodies, self.bond_normals):
            rot = d.xmat[b].reshape(3, 3)
            normal = rot @ n_local
            point = d.xpos[b] + normal * HALF
            torque, force = d.cfrc_int[b, :3], d.cfrc_int[b, 3:]
            centre = d.subtree_com[m.body_rootid[b]]
            moment = torque + np.cross(centre - point, force)
            bend = max(bend, float(np.linalg.norm(moment - (moment @ normal) * normal)))
            pull = max(pull, float(force @ normal))
        for b, sign in self.bearing_bodies:
            rot = d.xmat[b].reshape(3, 3)
            axis = rot[:, 0] * sign
            point = d.xpos[b] + axis * HALF
            torque, force = d.cfrc_int[b, :3], d.cfrc_int[b, 3:]
            moment = torque + np.cross(d.subtree_com[m.body_rootid[b]] - point, force)
            bear = max(bear, float(np.linalg.norm(moment - (moment @ axis) * axis)))
        return bend, pull, bear

    def summary(self, smooth=0.05):
        """Peak torques per motor that ever exceeded 0.05 N m, time above 0.4 N m, and bond loads
        (peaks of a `smooth`-second moving average, so a single contact impulse does not count)."""
        series = np.array(self.bond_series) if self.bond_series else np.zeros((1, 3))
        width = max(1, int(round(smooth / (self.every * self.model.opt.timestep))))
        kernel = np.ones(width) / width
        smoothed = [np.convolve(series[:, k], kernel, mode="valid") if len(series) >= width else series[:, k] for k in range(3)]
        used = {n: round(float(p), 3) for n, p in zip(self.names, self.peak) if p > 0.05}
        return {
            "peak_torque_nm": used,
            "over_0_4_s": {n: round(float(o), 3) for n, o in zip(self.names, self.over) if o > 0},
            "at_limit_s": {n: round(float(o), 3) for n, o in zip(self.names, self.limit) if o > 0},
            "bond_moment_nm": round(float(smoothed[0].max()), 3),
            "bond_tension_n": round(float(smoothed[1].max()), 2),
            "bearing_moment_nm": round(float(smoothed[2].max()), 3),
        }

    def position(self, body):
        return self.data.xpos[self.model.body(body).id].copy()

    def heading(self, body):
        """Yaw (rad) of a body's local x axis in the world."""
        x = self.data.xmat[self.model.body(body).id].reshape(3, 3)[:, 0]
        return math.atan2(x[1], x[0])
