"""Layout C as a simplified MuJoCo model: 50 mm cubes that drive in pairs on wheels, pivot singly
about an edge, and bond and unbond with switchable connectors during a run.

One cube (local frame, centre at the origin):

1. +x face: a flush turning disc (hinge about x, 42 mm across, inside the outline), torque-limited
   at 0.8 N m like layout B's discs (time above 0.4 N m is recorded).
1. -x face: a small drive wheel (20 mm across, axle along local y, rolling along local z) on a
   spring: a slide joint along the face normal holds it 1.5 mm out of the face when free with about
   2 N, and lets a bonded neighbour push it flush. A lone cube with its wheel face down rests on the
   wheel and one bottom edge and can only drive straight; two cubes bonded side by side (wheel axles
   in a line) drive as a differential pair.
1. Edge pivot: the cube turns about one of its edges, latched to a neighbour's edge (two soft point
   constraints, as in cube_hinge.Hinge) or resting on the floor, under a position-controlled torque
   clipped at 0.1 N m.
1. Bonds: a switchable connector on every face, modelled as a weld equality made and broken during
   the run. A bond forms only if the two faces are inside the capture range (+/- 3 mm lateral,
   +/- 5 degrees, a gap of at most 3 mm), measured from the sim state at the moment of the attempt;
   otherwise the attempt is a failed dock and nothing is welded.

    from beebot.hardware.module_c import World, Placement
    world = World([Placement((0, 0), yaw=0.0)], seed=0)
    world.drive_straight(0, 0.3)  # 30 cm along its rolling direction

Idealised (what this model leaves out):

1. The edge pivot's drive and latch are NOT designed. In the sim the pivot is a torque applied to
   the mover about the edge (and the opposite torque to the support cube, if any), PD on the hinge
   angle along a minimum-jerk reference at 150 deg/s, clipped at 0.1 N m; the latch is two soft
   point constraints that engage only if the two edges are within 3 mm (cube_hinge's catch, 0.06 L).
   A floor pivot has no latch (the floor is flat): the edge stays put by friction alone, and the
   torque reacts into the ground, which a real mechanism would have to do by pushing on the floor.
1. While a cube pivots, it does not collide with the cube it is latched to, nor with cubes that touch
   it only by an edge or a corner (cube_hinge's chamfered-edge idealisation); faces that slide on
   each other keep their friction, and the floor stays. The mover's wheel collides again over the
   last 15 degrees (so a wheel on the landing face is pushed in, not started 1.5 mm deep); its box
   collides again on arrival, or, when the move ends in bonds, after they have formed (box-on-box
   contact along the hinge line otherwise shoves a landed cube off, which chamfers would prevent).
1. A horizontal hinge runs along the support's real edge (a cube on its wheel leans 3.4 degrees);
   a vertical hinge (two cubes turning on the floor) stays plumb. The PD gains are cube_hinge's
   kp with kd lowered from 0.05 to 0.02 N m s/rad: two light cubes free in yaw made 0.05 unstable at
   1 kHz. The gear inertia of a real pivot drive would change this.
1. Connectors are NOT designed. A bond is a weld equality (soft, about 20 ms time constant) between
   the two cubes at the ideal lattice pose, switched on and off with eq_active. It has unlimited
   strength; the loads it carries are recorded and compared with spec 5.5 afterwards. A bond on the
   disc face welds to the disc, so the disc can turn what it carries. Capture is a pass/fail check
   on the true pose; there is no pull-in force model beyond the weld's own pull once captured.
1. Drives are ideal torque sources behind PD loops at 1 kHz, clipped (wheel 0.03 N m, disc 0.8 N m,
   pivot 0.1 N m). Wheels and discs hold their angle when stopped (geared). No speed-torque curve,
   no backlash, no heating, no battery.
1. Sensing: the planner reads every cube's pose from an external camera with Gaussian noise
   (1 mm, 1 degree per reading, guess); hinge and wheel angles are exact.
1. Mass: 100 g box, 6 g disc, 4 g wheel carriage, 10 g wheel (120 g; split is a guess), each
   uniform. Friction: floor 0.5, box 0.4, disc 0.8, wheel 1.0 (guesses; MuJoCo uses the larger of
   the two geoms' values). Box edges are sharp in contact with the floor.
1. Nothing inside the cube is modelled beyond the masses (no motors, battery or board volumes; see
   `packing` for the arithmetic check that they fit). Box edges are sharp (no chamfers).
1. The planner (class Planner) is scripted for the demos and knows its own groups' geometry (which
   cubes are bonded, which way their wheels turn); positions and headings come only from the noisy
   camera, leans from the IMU (1 degree noise). It moves one cube or group at a time and does no
   path planning beyond a fixed detour around the structure it docks to.
"""

import itertools
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from beebot.hardware.cube_hinge import smooth

SIDE = 0.050  # m
HALF = SIDE / 2
BODY_MASS = 0.100  # kg (guess: 120 g total, spec 3.6)
DISC_MASS = 0.006
CARRIAGE_MASS = 0.004
WHEEL_MASS = 0.010
DISC_RADIUS = 0.021  # m: 42 mm disc, inside the 50 mm face (spec 3.8)
DISC_THICK = 0.003
WHEEL_RADIUS = 0.010  # m: 20 mm wheel (guess)
WHEEL_WIDTH = 0.006  # m (guess)
PROUD = 0.0015  # m: the wheel stands this far out of its face when free
SPRING_FORCE = 2.0  # N at full extension (guess); the wheel carries a lone cube's 1.18 N without moving in
SPRING_K = 1000.0  # N/m (guess): 3.5 N when pushed flush
WHEEL_TORQUE = 0.03  # N m (guess: N20 class at about 150:1); traction slips at about 0.012 N m
WHEEL_KP, WHEEL_KV = 0.05, 0.004
WHEEL_MAX = 15.0  # rad/s: 0.15 m/s at the rim (spec 4.2)
DISC_TORQUE = 0.8  # N m holding (spec 4.1)
DISC_CONTINUOUS = 0.4
DISC_KP, DISC_KV = 10.0, 0.25
ARMATURE = 1e-3  # kg m2: disc gear train (as module_b)
WHEEL_ARMATURE = 1e-5
DRAG = 0.005  # N m: disc bearing and gearbox
PIVOT_TORQUE = 0.1  # N m (task: try 0.1; gravity needs at most m g L / sqrt(2) = 0.042)
PIVOT_RATE = 150.0  # deg/s (cube_hinge)
PIVOT_KP, PIVOT_KD = 3.0, 0.02  # cube_hinge's kp; kd lowered from 0.05: two light cubes free in yaw (I about 2.5e-5 kg m2) made cube_hinge's kd unstable at 1 kHz (kd dt / I about 2)
CATCH = 0.003  # m: latch engages when the edges are this close (cube_hinge: 0.06 L)
TOLERANCE = 3.0  # degrees: arrived
STALL = 0.5  # s past the reference's end before driving back
NEAR = 10.0  # degrees: a stall this close to the goal counts as arrived
LANDING = 15.0  # degrees before the goal at which the mover's wheel touches again (a proud wheel on the landing face would otherwise start 1.5 mm deep)
HOLD_TIME = 0.15  # s held at the goal before bonding and unlatching
CAPTURE_LATERAL = 0.003  # m (spec 5.4)
CAPTURE_ANGLE = 5.0  # degrees (spec 5.4)
CAPTURE_GAP = 0.003  # m: largest face gap a connector pulls in from (guess; the spec gives none)
SENSE_POS = 0.001  # m, 1 sigma per reading (guess)
SENSE_ANGLE = 1.0  # degrees, 1 sigma (guess)
PLACE_POS = 0.0015  # m: placement error of cubes set down in an arrangement, uniform +/- (guess)
PLACE_ANGLE = 1.5  # degrees, uniform +/- (guess)
FLOOR_FRICTION, BODY_FRICTION, DISC_FRICTION, WHEEL_FRICTION = 0.5, 0.4, 0.8, 1.0
DT = 0.001
MASS = BODY_MASS + DISC_MASS + CARRIAGE_MASS + WHEEL_MASS
G = 9.81
COLOURS = [(0.85, 0.55, 0.25), (0.35, 0.55, 0.80), (0.45, 0.70, 0.45), (0.80, 0.45, 0.55), (0.65, 0.55, 0.80), (0.80, 0.75, 0.40), (0.40, 0.70, 0.75), (0.75, 0.60, 0.45)]
AXES = "xyz"
FACES = ("+x", "-x", "+y", "-y", "+z", "-z")
DISC_FACE, WHEEL_FACE = "+x", "-x"


def unit(face):
    v = np.zeros(3)
    v[AXES.index(face[1])] = 1.0 if face[0] == "+" else -1.0
    return v


def face_of(vector):
    """The face whose outward normal is closest to `vector` (any direction)."""
    k = int(np.argmax(np.abs(vector)))
    return ("+" if vector[k] > 0 else "-") + AXES[k]


def rotation_about(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def rotation_angle(rot):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rot) - 1) / 2))))


LATTICE = [p for p in (np.diag(s) @ np.eye(3)[list(perm)] for perm in itertools.permutations(range(3)) for s in itertools.product((1, -1), repeat=3)) if np.linalg.det(p) > 0.5]


def nearest_lattice(rot):
    """The lattice rotation (one of 24) closest to `rot`, and the angle between them in degrees."""
    best = min(LATTICE, key=lambda p: rotation_angle(rot @ p.T))
    return best, rotation_angle(rot @ best.T)


def quat(rot):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(rot, float).reshape(-1))
    return q


def orientation(down="-x", heading=0.0):
    """Rotation of a cube with its face `down` on the floor, turned `heading` rad about world z. The
    first of local +z, +y, +x that lies along the floor points along world +x at heading 0 (for the
    wheel face down that is the rolling direction)."""
    n = unit(down)
    b = next(unit(f) for f in ("+z", "+y", "+x") if abs(unit(f) @ n) < 0.5)
    local = np.column_stack([n, b, np.cross(n, b)])
    world = np.column_stack([[0, 0, -1.0], [1.0, 0, 0], np.cross([0, 0, -1.0], [1.0, 0, 0])])
    return rotation_about([0, 0, 1], heading) @ world @ local.T


@dataclass
class Placement:
    """Where a cube starts: centre (x, y) on the floor in m, which face is down and its heading (rad)."""

    xy: tuple
    down: str = "-x"
    yaw: float = 0.0
    z: float = None


def _cube_xml(i, p):
    rot = orientation(p.down, p.yaw)
    z = p.z if p.z is not None else HALF + (PROUD + 0.0005 if p.down == WHEEL_FACE else 0.0005)
    r, g, b = COLOURS[i % len(COLOURS)]
    rgba = f"{r} {g} {b} 1"
    q = " ".join(f"{v:.8g}" for v in quat(rot))
    plain = 'contype="0" conaffinity="0" mass="0"'
    disc_x = HALF - DISC_THICK / 2
    wheel_x = -(HALF - WHEEL_RADIUS)
    return f"""<body name="c{i}" pos="{p.xy[0]:.6g} {p.xy[1]:.6g} {z:.6g}" quat="{q}">
<freejoint name="f{i}"/>
<geom name="c{i}box" type="box" size="{HALF} {HALF} {HALF}" mass="{BODY_MASS}" rgba="{rgba}" friction="{BODY_FRICTION} 0.005 0.0001"/>
<geom type="box" size="0.0004 0.010 0.010" pos="{-HALF:.6g} 0 {HALF - 0.012:.6g}" rgba="0.25 0.25 0.28 1" {plain}/>
<body name="c{i}disc" pos="{disc_x:.6g} 0 0">
<joint name="c{i}d" type="hinge" axis="1 0 0" armature="{ARMATURE}" frictionloss="{DRAG}" solreffriction="0.002 1"/>
<geom name="c{i}disc" type="cylinder" size="{DISC_RADIUS} {DISC_THICK / 2}" euler="0 90 0" mass="{DISC_MASS}" rgba="0.25 0.25 0.28 1" friction="{DISC_FRICTION} 0.005 0.0001"/>
<geom type="box" size="0.0004 0.0025 0.017" pos="{DISC_THICK / 2:.6g} 0 0.003" rgba="0.95 0.95 0.95 1" {plain}/>
<geom type="box" size="0.0004 0.006 0.0025" pos="{DISC_THICK / 2:.6g} 0 0.018" rgba="0.95 0.95 0.95 1" {plain}/>
</body>
<body name="c{i}car" pos="{wheel_x:.6g} 0 0">
<joint name="c{i}s" type="slide" axis="-1 0 0" range="0 {PROUD}" solreflimit="0.001 1" stiffness="{SPRING_K}" springref="{PROUD + SPRING_FORCE / SPRING_K:.6g}" damping="2"/>
<geom type="sphere" size="0.002" {plain.replace('mass="0"', f'mass="{CARRIAGE_MASS}"')} rgba="0 0 0 0"/>
<body name="c{i}wh">
<joint name="c{i}w" type="hinge" axis="0 1 0" armature="{WHEEL_ARMATURE}"/>
<geom name="c{i}wh" type="cylinder" size="{WHEEL_RADIUS} {WHEEL_WIDTH / 2}" euler="90 0 0" mass="{WHEEL_MASS}" rgba="0.12 0.12 0.12 1" friction="{WHEEL_FRICTION} 0.005 0.0001"/>
<geom type="box" size="0.0035 0.0032 0.0035" pos="0 0 0.006" rgba="0.9 0.9 0.9 1" {plain}/>
</body>
</body>
</body>
"""


def build(placements, scene="", timestep=DT, welds=None, latches=4):
    """MJCF: the cubes on a flat floor, a pool of weld equalities (bonds) and connect equalities
    (pivot latches), all inactive, and a disc and a wheel motor per cube."""
    n = len(placements)
    welds = welds if welds is not None else max(4, 3 * n)
    cubes = "".join(_cube_xml(i, p) for i, p in enumerate(placements))
    excludes = "".join(f'<exclude body1="c{i}" body2="c{i}wh"/>\n<exclude body1="c{i}disc" body2="c{i}car"/>\n<exclude body1="c{i}disc" body2="c{i}wh"/>\n' for i in range(n))
    b2 = "c1" if n > 1 else "world"
    pool = "".join(f'<weld name="bond{k}" body1="c0" body2="{b2}" active="false" solref="0.02 1" solimp="0.95 0.99 0.001"/>\n' for k in range(welds))
    pool += "".join(f'<connect name="latch{k}" body1="c0" body2="{b2}" anchor="0 0 0" active="false" solref="0.005 1"/>\n' for k in range(latches))
    motors = "".join(f'<motor name="d{i}" joint="c{i}d" ctrllimited="true" ctrlrange="-{DISC_TORQUE} {DISC_TORQUE}"/>\n<motor name="w{i}" joint="c{i}w" ctrllimited="true" ctrlrange="-{WHEEL_TORQUE} {WHEEL_TORQUE}"/>\n' for i in range(n))
    return f"""<mujoco model="module_c">
<compiler angle="degree" autolimits="true"/>
<option timestep="{timestep}" integrator="implicitfast" cone="elliptic" impratio="10"/>
<size memory="128M"/>
<visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>
<default><geom solref="0.004 1" solimp="0.95 0.99 0.001"/></default>
<asset>
<texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.82 0.80" rgb2="0.74 0.74 0.72" width="512" height="512"/>
<material name="grid" texture="grid" texrepeat="40 40" reflectance="0.05"/>
<texture name="sky" type="skybox" builtin="gradient" rgb1="0.92 0.93 0.95" rgb2="0.70 0.74 0.80" width="256" height="256"/>
</asset>
<worldbody>
<light pos="0.3 -0.6 1.5" dir="-0.2 0.4 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
<geom name="floor" type="plane" size="2 2 0.05" material="grid" friction="{FLOOR_FRICTION} 0.005 0.0001"/>
{scene}
{cubes}</worldbody>
<contact>
{excludes}</contact>
<equality>
{pool}</equality>
<actuator>
{motors}</actuator>
</mujoco>
"""


class Failure(Exception):
    """A step of a plan that could not be done (a dock that kept failing, a pivot that stalled)."""


class Pivot:
    """One edge pivot in progress: the mover turns `goal` rad about `axis` (world, right hand) through
    the edge at `point`, latched to `support` (a cube) or resting on the floor (support None)."""

    def __init__(self, world, mover, point, axis, goal, support, tolerance=TOLERANCE):
        self.world, self.mover, self.support = world, mover, support
        self.tolerance = tolerance
        axis = np.asarray(axis, float) / np.linalg.norm(axis)
        if abs(axis[2]) < 0.9:  # a horizontal hinge runs along a real edge of a (leaning) cube: snap the asked axis to it
            frame = world.rot(mover if support is None else support)
            axis = frame @ unit(face_of(frame.T @ axis))
        self.axis = axis  # a vertical hinge (two cubes turning on the floor) stays plumb
        self.goal = goal
        self.duration = abs(math.degrees(goal)) / PIVOT_RATE
        d, m = world.data, world.model
        bm = world.body[mover]
        x_m, R_m = d.xpos[bm].copy(), d.xmat[bm].reshape(3, 3).copy()
        edge = world.edge_of(mover, point, self.axis)  # the mover's own edge nearest the requested line
        self.edge = edge
        if support is None:
            x_s, R_s = np.zeros(3), np.eye(3)
            self.miss = 0.0
            self.latched = True  # nothing to latch; the floor holds the edge by friction
        else:
            bs = world.body[support]
            x_s, R_s = d.xpos[bs].copy(), d.xmat[bs].reshape(3, 3).copy()
            mate = world.edge_of(support, edge, self.axis)
            gap = edge - mate
            gap = gap - self.axis * (gap @ self.axis)
            self.miss = float(np.linalg.norm(gap))
            self.latched = self.miss <= CATCH
            if self.latched:
                self.slots = world.free_latches(2)
                for k, end in zip(self.slots, (-0.35, 0.35)):
                    m.eq_obj1id[k] = bm
                    m.eq_obj2id[k] = bs
                    m.eq_data[k][:3] = R_m.T @ (edge + end * SIDE * self.axis - x_m)
                    m.eq_data[k][3:6] = R_s.T @ (mate + end * SIDE * self.axis - x_s)
                    d.eq_active[k] = 1
        self.R0 = R_s.T @ R_m
        self.axis_s = R_s.T @ self.axis
        self.angle = 0.0
        self.phase = "go" if self.latched else "refused"
        self.clock = 0.0
        self.start, self.end = 0.0, goal
        self.outcome = "go" if self.latched else "unlatched"
        self.peak = 0.0  # N m: largest torque applied (after the clip), landing included
        self.moving_peak = 0.0  # N m: largest torque applied before arriving
        self.saturated = 0.0  # s at the clip before arriving
        self.arrived = None

    def frames(self):
        d = self.world.data
        R_m = d.xmat[self.world.body[self.mover]].reshape(3, 3)
        R_s = np.eye(3) if self.support is None else d.xmat[self.world.body[self.support]].reshape(3, 3)
        return R_m, R_s

    def measure(self):
        R_m, R_s = self.frames()
        turn = R_s.T @ R_m @ self.R0.T
        q = quat(turn)
        angle = 2 * math.atan2(float(q[1:] @ self.axis_s), float(q[0]))
        angle += 2 * math.pi * round((self.angle - angle) / (2 * math.pi))
        self.angle = angle
        d = self.world.data
        w_m = d.cvel[self.world.body[self.mover]][:3]
        w_s = np.zeros(3) if self.support is None else d.cvel[self.world.body[self.support]][:3]
        return angle, float((w_m - w_s) @ (R_s @ self.axis_s))

    def torque(self, dt):
        self.clock += dt
        angle, rate = self.measure()
        span = self.end - self.start
        duration = max(abs(span) / math.radians(PIVOT_RATE), 1e-3)
        u, du = smooth(self.clock / duration)
        target, target_rate = self.start + span * u, span * du / duration
        near = abs(angle - self.end) < math.radians(self.tolerance)
        if self.phase in ("go", "back") and near and self.clock >= duration:
            self.arrived, self.outcome, self.phase = self.clock, ("arrived" if self.phase == "go" else "returned"), "hold"
        elif self.phase == "go" and self.clock > duration + STALL and abs(angle - self.end) < math.radians(NEAR):
            self.arrived, self.outcome, self.phase = self.clock, "near", "hold"
        elif self.phase == "go" and self.clock > duration + STALL:
            self.phase, self.start, self.end, self.clock, self.outcome = "back", angle, 0.0, 0.0, "stalled"
        elif self.phase == "back" and self.clock > duration + STALL:
            self.arrived, self.outcome, self.phase = self.clock, "stuck", "hold"
        if self.phase == "back":
            span = self.end - self.start
            duration = max(abs(span) / math.radians(PIVOT_RATE), 1e-3)
            u, du = smooth(self.clock / duration)
            target, target_rate = self.start + span * u, span * du / duration
        if self.phase == "hold":
            target, target_rate = self.end, 0.0
        wanted = PIVOT_KP * (target - angle) + PIVOT_KD * (target_rate - rate)
        tau = float(np.clip(wanted, -PIVOT_TORQUE, PIVOT_TORQUE))
        self.peak = max(self.peak, abs(tau))
        if self.phase == "go":
            self.moving_peak = max(self.moving_peak, abs(tau))
            self.saturated += dt * (abs(wanted) >= PIVOT_TORQUE)
        _, R_s = self.frames()
        return tau, R_s @ self.axis_s

    def release(self):
        if self.support is not None and self.latched:
            for k in self.slots:
                self.world.data.eq_active[k] = 0
            self.world.latch_busy -= set(self.slots)


class World:
    """Cubes on a floor with their drives, pivots, bonds, noisy sensing and a recorder. All motion
    comes from the wheel and disc motors and the pivot torque; nothing sets qpos after the start."""

    def __init__(self, placements, seed=0, scene="", bonds=(), watch=None, timestep=DT):
        self.n = len(placements)
        self.placements = placements
        self.xml = build(placements, scene, timestep)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        m, d = self.model, self.data
        self.rng = np.random.default_rng(seed)
        self.body = [m.body(f"c{i}").id for i in range(self.n)]
        self.disc_body = [m.body(f"c{i}disc").id for i in range(self.n)]
        self.wheel_body = [m.body(f"c{i}wh").id for i in range(self.n)]
        self.slide_q = [m.jnt_qposadr[m.joint(f"c{i}s").id] for i in range(self.n)]
        self.wheel_q = np.array([m.jnt_qposadr[m.joint(f"c{i}w").id] for i in range(self.n)])
        self.wheel_v = np.array([m.jnt_dofadr[m.joint(f"c{i}w").id] for i in range(self.n)])
        self.disc_q = np.array([m.jnt_qposadr[m.joint(f"c{i}d").id] for i in range(self.n)])
        self.disc_v = np.array([m.jnt_dofadr[m.joint(f"c{i}d").id] for i in range(self.n)])
        self.wheel_act = np.array([m.actuator(f"w{i}").id for i in range(self.n)])
        self.disc_act = np.array([m.actuator(f"d{i}").id for i in range(self.n)])
        self.geoms = [[g for g in range(m.ngeom) if m.geom_bodyid[g] in (self.body[i], self.disc_body[i], self.wheel_body[i], m.body(f"c{i}car").id) and m.geom_contype[g]] for i in range(self.n)]
        self.weld_ids = [m.equality(f"bond{k}").id for k in range(sum(1 for k in range(m.neq) if m.equality(k).name.startswith("bond")))]
        self.latch_ids = [m.equality(f"latch{k}").id for k in range(sum(1 for k in range(m.neq) if m.equality(k).name.startswith("latch")))]
        self.latch_busy = set()
        self.bonds = {}  # frozenset({i, j}) -> (slot, face_i, i, face_j, j)
        for i in range(self.n):
            d.qpos[self.slide_q[i]] = PROUD  # the wheel starts out at its stop (start state only)
        mujoco.mj_forward(m, d)
        # wheel and disc controllers
        self.wheel_ref = d.qpos[self.wheel_q].copy()
        self.wheel_rate = np.zeros(self.n)
        self.wheel_cap = np.full(self.n, WHEEL_TORQUE)
        self.disc_ref = np.zeros(self.n)
        self.disc_rate = np.zeros(self.n)
        self.disc_target = None  # (q0, q1, t0, span) per cube for scripted disc moves
        self.disc_moves = {}
        self.pivots = []
        self.watch = watch
        # records
        self.events = []
        self.dock_attempts = 0
        self.failed_docks = 0
        self.pivot_count = 0
        self.pivot_peaks = []
        self.pivot_outcomes = []
        self.driven = 0.0  # cube-metres driven on wheels
        self.peak_wheel = 0.0
        self.peak_disc = 0.0
        self.disc_over = 0.0
        self.peak_bond_force = 0.0
        self.peak_bond_torque = 0.0
        self.harm = []
        self.last_pos = np.array([d.xpos[b].copy() for b in self.body])
        for i, j, fi in bonds:
            if not self.bond(i, j, fi, count=False):
                raise ValueError(f"start bond {i}-{j} out of capture range")

    # ------------------------------------------------------------------ state

    def pos(self, i):
        return self.data.xpos[self.body[i]].copy()

    def rot(self, i):
        return self.data.xmat[self.body[i]].reshape(3, 3).copy()

    def sense(self, i):
        """Pose of cube i as the camera reads it: (centre, rotation) with noise."""
        p = self.pos(i) + self.rng.normal(0, SENSE_POS, 3)
        axis = self.rng.normal(size=3)
        noise = rotation_about(axis, math.radians(self.rng.normal(0, SENSE_ANGLE)))
        return p, noise @ self.rot(i)

    def world_face(self, i, face, rot=None):
        return (self.rot(i) if rot is None else rot) @ unit(face)

    def face_toward(self, i, direction, rot=None):
        """Name of cube i's face whose outward normal is closest to the world `direction`."""
        r = self.rot(i) if rot is None else rot
        return face_of(r.T @ np.asarray(direction, float))

    def down_face(self, i):
        return self.face_toward(i, [0, 0, -1])

    def wheel_down(self, i):
        return self.down_face(i) == WHEEL_FACE

    def rolling(self, i, rot=None):
        """World direction cube i moves for a positive wheel speed (horizontal unit vector)."""
        r = self.rot(i) if rot is None else rot
        axle = r @ np.array([0.0, 1.0, 0.0])
        v = np.cross(axle, [0.0, 0.0, 1.0]) * WHEEL_RADIUS  # centre velocity per rad/s
        v[2] = 0
        return v / max(np.linalg.norm(v), 1e-9)

    def edge_of(self, i, point, axis):
        """Midpoint of the edge of cube i that runs along `axis` and lies nearest `point`."""
        r, c = self.rot(i), self.pos(i)
        a_local = r.T @ axis
        k = int(np.argmax(np.abs(a_local)))
        best, dist = None, 1e9
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                v = np.zeros(3)
                others = [x for x in range(3) if x != k]
                v[others[0]], v[others[1]] = s1 * HALF, s2 * HALF
                w = c + r @ v
                dd = np.linalg.norm((w - point) - axis * ((w - point) @ axis))
                if dd < dist:
                    best, dist = w, dd
        return best

    def free_latches(self, count):
        free = [k for k in self.latch_ids if k not in self.latch_busy][:count]
        if len(free) < count:
            raise RuntimeError("latch pool exhausted")
        self.latch_busy |= set(free)
        return free

    # ------------------------------------------------------------------ stepping

    def step(self):
        d, m = self.data, self.model
        t = d.time
        # wheels: reference angle advancing at the commanded rate, PD, clipped
        self.wheel_ref += self.wheel_rate * m.opt.timestep
        q, qd = d.qpos[self.wheel_q], d.qvel[self.wheel_v]
        err = self.wheel_ref - q
        slip = np.abs(err) > 0.3  # slipping or blocked: do not wind up
        self.wheel_ref = np.where(slip, q + np.sign(err) * 0.3, self.wheel_ref)
        tau = np.clip(WHEEL_KP * (self.wheel_ref - q) + WHEEL_KV * (self.wheel_rate - qd), -self.wheel_cap, self.wheel_cap)
        d.ctrl[self.wheel_act] = tau
        self.peak_wheel = max(self.peak_wheel, float(np.abs(tau).max()))
        # discs
        for i, (q0, q1, t0, span) in list(self.disc_moves.items()):
            u, du = smooth((t - t0) / span)
            self.disc_ref[i], self.disc_rate[i] = q0 + (q1 - q0) * u, (q1 - q0) * du / span
        dq, dqd = d.qpos[self.disc_q], d.qvel[self.disc_v]
        dtau = np.clip(DISC_KP * (self.disc_ref - dq) + DISC_KV * (self.disc_rate - dqd), -DISC_TORQUE, DISC_TORQUE)
        d.ctrl[self.disc_act] = dtau
        self.peak_disc = max(self.peak_disc, float(np.abs(dtau).max()))
        self.disc_over += float((np.abs(dtau) > DISC_CONTINUOUS).any()) * m.opt.timestep
        # pivots
        d.xfrc_applied[:, 3:] = 0.0
        for p in self.pivots:
            if p.phase in ("refused", "released"):
                continue
            tau_p, axis = p.torque(m.opt.timestep)
            d.xfrc_applied[self.body[p.mover], 3:] += tau_p * axis
            if p.support is not None:
                d.xfrc_applied[self.body[p.support], 3:] -= tau_p * axis
        before = d.time
        mujoco.mj_step(m, d)
        if d.time < before:
            raise RuntimeError(f"MuJoCo reset at t = {before:.3f} s (unstable)")
        # odometry for metres driven
        now = np.array([d.xpos[b] for b in self.body])
        moving = np.abs(self.wheel_rate) > 1e-6
        if moving.any():
            self.driven += float(np.linalg.norm((now - self.last_pos)[moving][:, :2], axis=1).sum())
        self.last_pos = now
        if int(round(d.time / m.opt.timestep)) % 10 == 0 and self.bonds:
            self._bond_loads()
        if self.watch is not None:
            self.watch(self)

    def run(self, seconds):
        end = self.data.time + seconds
        while self.data.time < end - 1e-9:
            self.step()

    def _bond_loads(self):
        d, m = self.data, self.model
        for slot, *_ in self.bonds.values():
            rows = np.nonzero((d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == slot))[0]
            if len(rows) >= 6:
                f = d.efc_force[rows]
                self.peak_bond_force = max(self.peak_bond_force, float(np.linalg.norm(f[:3])))
                self.peak_bond_torque = max(self.peak_bond_torque, float(np.linalg.norm(f[3:6])))

    # ------------------------------------------------------------------ wheels

    def set_wheels(self, speeds, cap=None):
        """speeds: {cube: rad/s}. Stopped wheels hold their angle."""
        for i, w in speeds.items():
            self.wheel_rate[i] = float(np.clip(w, -WHEEL_MAX, WHEEL_MAX))
            self.wheel_cap[i] = WHEEL_TORQUE if cap is None else cap

    def stop(self, cubes, settle=0.3):
        """Ramp the wheels of `cubes` to zero over 0.15 s and wait."""
        start = {i: self.wheel_rate[i] for i in cubes}
        for k in range(15):
            self.set_wheels({i: w * (1 - (k + 1) / 15) for i, w in start.items()})
            self.run(0.01)
        self.run(settle)

    # ------------------------------------------------------------------ bonds

    def capture(self, i, j, fi=None):
        """Capture-range check between cube i's face fi (default: the face toward j) and cube j, on
        the true state. Returns (ok, lateral m, angle deg, gap m, fi, fj, relative rotation)."""
        Ri, Rj, ci, cj = self.rot(i), self.rot(j), self.pos(i), self.pos(j)
        fi = fi or face_of(Ri.T @ (cj - ci))
        n = unit(fi)
        rel = Ri.T @ Rj
        lattice, angle = nearest_lattice(rel)
        offset = Ri.T @ (cj - ci) - n * SIDE
        gap = float(offset @ n)
        lateral = float(np.linalg.norm(offset - n * gap))
        fj = face_of(Rj.T @ (ci - cj))
        ok = lateral <= CAPTURE_LATERAL and angle <= CAPTURE_ANGLE and -0.0015 <= gap <= CAPTURE_GAP
        return ok, lateral, angle, gap, fi, fj, lattice

    def bond(self, i, j, fi=None, count=True):
        """Try to bond cube i (face fi, default the face toward j) to cube j. Forms a weld at the ideal
        lattice pose only inside the capture range; otherwise records a failed dock."""
        ok, lateral, angle, gap, fi, fj, lattice = self.capture(i, j, fi)
        if count:
            self.dock_attempts += 1
        self.events.append({"t": round(self.data.time, 3), "bond": [i, j], "ok": bool(ok), "lateral_mm": round(lateral * 1000, 2), "angle_deg": round(angle, 2), "gap_mm": round(gap * 1000, 2)})
        if not ok:
            if count:
                self.failed_docks += 1
            return False
        key = frozenset((i, j))
        if key in self.bonds:
            return True
        m, d = self.model, self.data
        busy = {s for s, *_ in self.bonds.values()}
        slot = next(k for k in self.weld_ids if k not in busy)
        # body1: the disc if the face is i's disc face, else i's box; body2: j's disc or box
        b1 = self.disc_body[i] if fi == DISC_FACE else self.body[i]
        b2 = self.disc_body[j] if fj == DISC_FACE else self.body[j]
        R_i, c_i = self.rot(i), self.pos(i)
        Rj_ideal = R_i @ lattice
        cj_ideal = c_i + R_i @ unit(fi) * SIDE
        R1, x1 = d.xmat[b1].reshape(3, 3), d.xpos[b1]
        R2, x2 = d.xmat[b2].reshape(3, 3), d.xpos[b2]
        # body2's pose in body2's own cube frame (constant)
        R2_in_j = self.rot(j).T @ R2
        x2_in_j = self.rot(j).T @ (x2 - self.pos(j))
        R2_ideal = Rj_ideal @ R2_in_j
        x2_ideal = cj_ideal + Rj_ideal @ x2_in_j
        m.eq_obj1id[slot], m.eq_obj2id[slot] = b1, b2
        m.eq_data[slot][:3] = 0.0
        m.eq_data[slot][3:6] = R1.T @ (x2_ideal - x1)
        m.eq_data[slot][6:10] = quat(R1.T @ R2_ideal)
        m.eq_data[slot][10] = 1.0
        d.eq_active[slot] = 1
        self.bonds[key] = (slot, fi, i, fj, j)
        return True

    def unbond(self, i, j):
        key = frozenset((i, j))
        if key in self.bonds:
            self.data.eq_active[self.bonds[key][0]] = 0
            del self.bonds[key]

    def bonded(self, i):
        return [next(iter(k - {i})) for k in self.bonds if i in k]

    def group_of(self, i):
        seen, todo = {i}, [i]
        while todo:
            for j in self.bonded(todo.pop()):
                if j not in seen:
                    seen.add(j)
                    todo.append(j)
        return sorted(seen)

    # ------------------------------------------------------------------ pivots

    def pivot(self, mover, point, axis, angle, support=None, bond_to=(), unbond_from=None, leans=None, tolerance=TOLERANCE):
        """Turn `mover` by `angle` rad about the world `axis` through its edge nearest `point`, latched
        to `support` (None: on the floor). Order: latch, release the mover's bonds (all, or
        `unbond_from`), contacts with the support and edge-only neighbours off, turn (the mover's
        wheel touches again over the last LANDING degrees), hold, bond to the cubes in `bond_to`
        (capture check; "near": every cube one pitch away), contacts back, unlatch, settle. With
        nothing to bond, contacts come back on arrival. Returns the outcome."""
        axis = np.asarray(axis, float)
        pv = Pivot(self, mover, point, axis, angle, support, tolerance)
        self.pivot_count += 1
        if not pv.latched:
            self.pivot_outcomes.append("unlatched")
            self.events.append({"t": round(self.data.time, 3), "pivot": mover, "outcome": "unlatched", "miss_mm": round(pv.miss * 1000, 2)})
            return "unlatched"
        for j in list(self.bonded(mover) if unbond_from is None else unbond_from):
            self.unbond(mover, j)
        m = self.model
        contype, conaff = m.geom_contype.copy(), m.geom_conaffinity.copy()
        quiet = set(leans or []) | ({support} if support is not None else set())
        c = self.pos(mover)
        r = self.rot(mover)
        for j in range(self.n):  # cubes touching the mover only by an edge or a corner
            if j == mover:
                continue
            local = r.T @ (self.pos(j) - c) / SIDE
            if np.abs(local).max() < 1.3 and (np.abs(local) > 0.6).sum() >= 2:
                quiet.add(j)
        for g in range(m.ngeom):
            if m.geom_contype[g]:
                m.geom_conaffinity[g] = 3
        for g in self.geoms[mover]:
            m.geom_contype[g] = m.geom_conaffinity[g] = 2
        for j in quiet:
            for g in self.geoms[j]:
                m.geom_contype[g] = m.geom_conaffinity[g] = 1
        self.pivots.append(pv)
        horizon = 2 * (pv.duration + STALL) + 1.0
        t0 = self.data.time
        restored = wheel_back = False
        wheel = m.geom(f"c{mover}wh").id
        while self.data.time - t0 < horizon:
            self.step()
            if pv.phase == "go" and not wheel_back and abs(pv.angle - pv.end) < math.radians(LANDING):
                m.geom_contype[wheel] = m.geom_conaffinity[wheel] = 1  # the wheel touches again first: a wheel on the landing face is pushed in gradually
                wheel_back = True
            if pv.phase == "hold" and not restored and (not bond_to or pv.outcome not in ("arrived", "near")):  # arrived (or back) with nothing to bond: contacts return while still latched and held
                m.geom_contype[:], m.geom_conaffinity[:] = contype, conaff
                restored = True
            if pv.phase == "hold" and pv.clock - pv.arrived > HOLD_TIME:
                break
        if bond_to == "near":  # every cube whose centre the camera puts one pitch away
            p_m = self.sense(mover)[0]
            bond_to = [j for j in range(self.n) if j != mover and abs(np.linalg.norm(self.sense(j)[0] - p_m) - SIDE) < 0.008]
        for j in bond_to:
            if pv.outcome in ("arrived", "near"):
                self.bond(mover, j)
        if not restored:  # a move that ends in bonds: they form first (the weld holds the mover flush), then contacts return
            m.geom_contype[:], m.geom_conaffinity[:] = contype, conaff
            self.run(0.05)
        pv.release()
        pv.phase = "released"
        self.pivots.remove(pv)
        m.geom_contype[:], m.geom_conaffinity[:] = contype, conaff
        self.run(0.2)
        self.pivot_peaks.append((pv.moving_peak, pv.peak, pv.saturated))
        self.pivot_outcomes.append(pv.outcome)
        self.events.append({"t": round(self.data.time, 3), "pivot": mover, "outcome": pv.outcome, "moving_peak_nm": round(pv.moving_peak, 4), "peak_nm": round(pv.peak, 4), "saturated_s": round(pv.saturated, 3), "miss_mm": round(pv.miss * 1000, 2)})
        return pv.outcome

    def floor_roll(self, i, direction):
        """Roll lone cube i one quarter turn on the floor toward the world horizontal `direction`."""
        direction = np.asarray(direction, float)
        direction[2] = 0
        direction /= np.linalg.norm(direction)
        c = self.pos(i)
        point = c + direction * HALF - np.array([0, 0, c[2]])
        axis = np.cross([0, 0, 1.0], direction)
        return self.pivot(i, point, axis, math.pi / 2)

    # ------------------------------------------------------------------ discs

    def turn_disc(self, i, angle, seconds):
        q0 = float(self.data.qpos[self.disc_q[i]])
        self.disc_moves[i] = (self.disc_ref[i], self.disc_ref[i] + angle, self.data.time, seconds)
        del q0

    # ------------------------------------------------------------------ harm

    def check_harm(self, cubes, label):
        """A cube that is not lattice-aligned with the floor (tipped beyond 10 degrees off any face)
        or below the floor counts as fallen."""
        for i in cubes:
            _, angle = nearest_lattice(self.rot(i))
            tilt = self._tilt(i)
            if tilt > 10.0:
                self.harm.append({"t": round(self.data.time, 2), "cube": i, "what": f"tipped {tilt:.0f} deg", "during": label})

    def _tilt(self, i):
        """Angle between the cube's face nearest to down and the world down direction (degrees)."""
        r = self.rot(i)
        down = r.T @ np.array([0, 0, -1.0])
        return math.degrees(math.acos(min(1.0, float(np.abs(down).max()))))


# ---------------------------------------------------------------------- planner primitives
#
# Everything below reads poses only through World.sense (camera noise) and acts only through the
# wheel speeds, the pivot and bond calls. CONTROL is the planner's tick.

CONTROL = 0.02  # s
DECEL = 0.4  # m/s2: the planner's braking profile into a stop, v = sqrt(2 DECEL s)
OVERTURN = 1.5  # degrees past the quarter turn when a cube turns into the cell diagonal to its partner: a
# cube cannot move sideways, so it must slide in alongside with its line pointing slightly outward (a
# turn that stops short, as friction makes it, would aim it into the partner's corner)


def horizontal(v):
    v = np.array(v, float)
    v[2] = 0.0
    return v / max(np.linalg.norm(v), 1e-12)


def signed_angle(a, b):
    """Angle (rad) from horizontal a to horizontal b about +z."""
    return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])


class Planner:
    """Scripted motions of cubes and pairs on a World, closed loop on the noisy camera poses."""

    def __init__(self, world):
        self.w = world

    # ----------------------------------------------------------- sensing helpers

    def look(self, i, frames=1):
        """Camera pose of cube i averaged over `frames` readings taken while time runs on."""
        ps, rs = [], []
        for k in range(frames):
            p, r = self.w.sense(i)
            ps.append(p)
            rs.append(r)
            if frames > 1:
                self.w.run(CONTROL / 2)
        r = np.mean(rs, axis=0)
        u, _, vt = np.linalg.svd(r)
        return np.mean(ps, axis=0), u @ vt

    def pitch(self, i):
        """Lean (deg) of cube i's top toward its rolling direction, from the IMU (1 degree noise)."""
        r = self.w.rot(i)
        up = r @ unit(face_of(r.T @ np.array([0, 0, 1.0])))
        lean = math.degrees(math.asin(float(np.clip(up @ self.w.rolling(i), -1, 1))))
        return lean + self.w.rng.normal(0, SENSE_ANGLE)

    # ----------------------------------------------------------- lone cube

    def drive_to(self, i, target, speed=0.15, tol=0.0008, timeout=15.0, fine=True):
        """Lone cube i drives along its own line until its centre is level with `target` (the
        target's projection on the line): closed loop on single camera frames to within 2 mm, then
        (fine) slow nudges at 8 mm/s on camera poses averaged over 5 frames, which do not change
        which edge the cube leans on. Returns the camera-read remaining error (m)."""
        w = self.w
        t0 = w.data.time
        while w.data.time - t0 < timeout:
            p, r = self.look(i)
            d = w.rolling(i, r)
            s = float((np.asarray(target) - p)[:2] @ d[:2])
            if abs(s) < max(tol, 0.002 if fine else tol):
                break
            v = math.copysign(min(speed, max(0.01, math.sqrt(2 * DECEL * abs(s)))), s)
            w.set_wheels({i: v / WHEEL_RADIUS})
            w.run(CONTROL)
        w.stop([i], settle=0.15)
        for _ in range(4 if fine else 0):
            p, r = self.look(i, 5)
            s = float((np.asarray(target) - p)[:2] @ w.rolling(i, r)[:2])
            if abs(s) < tol:
                break
            w.set_wheels({i: math.copysign(0.008, s) / WHEEL_RADIUS})
            w.run(abs(s) / 0.008)
            w.set_wheels({i: 0.0})
            w.run(0.15)
        p, r = self.look(i, 3)
        return float((np.asarray(target) - p)[:2] @ w.rolling(i, r)[:2])

    def drive_straight(self, i, distance, speed=0.15):
        p, r = self.look(i, 3)
        return self.drive_to(i, p + self.w.rolling(i, r) * distance, speed)

    def push(self, i, toward, seconds=0.8, speed=0.05, cap=WHEEL_TORQUE):
        """Lone cube i drives toward the point `toward` along its line until it stops moving (contact)
        and pushes on for a moment. Returns the distance it moved."""
        w = self.w
        p0, r = self.look(i)
        d = w.rolling(i, r)
        sign = 1.0 if (np.asarray(toward) - p0)[:2] @ d[:2] > 0 else -1.0
        w.set_wheels({i: sign * speed / WHEEL_RADIUS}, cap=cap)
        last, still, t0 = w.pos(i), 0.0, w.data.time
        while w.data.time - t0 < 6.0:
            w.run(0.05)
            now = w.pos(i)
            still = still + 0.05 if np.linalg.norm((now - last)[:2]) < 0.0004 else 0.0
            last = now
            if still >= seconds:
                break
        w.stop([i], settle=0.15)
        w.set_wheels({i: 0.0})
        return float(np.linalg.norm((w.pos(i) - p0)[:2]))

    def tip(self, cubes, direction):
        """Leave lone cube or group `cubes` leaning toward the horizontal `direction`. A cube on its
        wheel rests on one bottom edge and leans the way it last stopped: an open-loop pulse back and
        forth at 5 cm/s for 60 ms each, with hard stops, sets the lean (and flips an existing one)."""
        w = self.w
        i = cubes[0]
        u = horizontal(direction)
        if self.pitch(i) * float(w.rolling(i) @ u) > 0.5 * math.degrees(math.atan(PROUD / HALF)):  # clearly leaning that way already (a balanced cube reads 0 +/- noise)
            return False
        along = float(np.sign(self.group_forward(cubes) @ u)) or 1.0
        for sign in (-along, along):
            self.group_speed(cubes, sign * 0.05, 0.0)
            w.run(0.06)
            w.set_wheels({k: 0.0 for k in cubes})
            w.run(0.15)
        return True

    def roll(self, i, direction):
        """Quarter-turn floor pivot of lone cube i toward the world `direction` (snapped to its
        nearest face direction)."""
        r = self.w.rot(i)
        face = face_of(r.T @ horizontal(direction))
        return self.w.floor_roll(i, horizontal(r @ unit(face)))

    def turn_heading(self, i, side, away=None):
        """Change lone cube i's heading by 90 degrees with three floor pivots: roll toward `side`,
        then one cell along its line (away from the point `away` if given), then back. It ends one
        cell along its old line."""
        w = self.w
        d = w.rolling(i)
        if away is not None and (w.pos(i) - np.asarray(away)) @ d < 0:
            d = -d
        u = horizontal(side)
        u = horizontal(u - d * (u @ d))
        outcomes = [self.roll(i, u), self.roll(i, d), self.roll(i, -u)]
        return outcomes

    def wheel_down(self, i, free_direction):
        """Floor pivots until cube i's wheel face is down (at most two). Returns the pivots used."""
        w = self.w
        used = 0
        for _ in range(3):
            if w.wheel_down(i):
                return used
            r = w.rot(i)
            n = r @ unit(WHEEL_FACE)
            direction = horizontal(n) if abs(n[2]) < 0.7 else horizontal(free_direction)
            out = w.floor_roll(i, direction)
            used += 1
            if out not in ("arrived", "near"):
                raise Failure(f"reorient pivot {out}")
        if not w.wheel_down(i):
            raise Failure("wheel face still not down")
        return used

    # ----------------------------------------------------------- groups (differential pairs and up)

    def group_forward(self, cubes, rots=None):
        r = self.w.rot(cubes[0]) if rots is None else rots[0]
        return self.w.rolling(cubes[0], r)

    def group_speed(self, cubes, v, yaw_rate, poses=None):
        """Wheel speeds for the group to move at v (m/s, along cube 0's rolling direction) while
        turning at yaw_rate (rad/s) about its centre."""
        w = self.w
        poses = poses or [(w.pos(i), w.rot(i)) for i in cubes]
        c = np.mean([p for p, _ in poses], axis=0)
        f = self.group_forward(cubes, [r for _, r in poses])
        rates = {}
        for i, (p, r) in zip(cubes, poses):
            vel = v * f + yaw_rate * np.cross([0, 0, 1.0], p - c)
            rates[i] = float(vel[:2] @ w.rolling(i, r)[:2]) / WHEEL_RADIUS
        top = max(abs(x) for x in rates.values())
        if top > WHEEL_MAX:
            rates = {i: x * WHEEL_MAX / top for i, x in rates.items()}
        w.set_wheels(rates)

    def group_pose(self, cubes, frames=1):
        poses = [self.look(i, frames) if frames > 1 else self.w.sense(i) for i in cubes]
        c = np.mean([p for p, _ in poses], axis=0)
        return c, self.group_forward(cubes, [r for _, r in poses]), poses

    def turn_to(self, cubes, u, tol=0.8, timeout=8.0):
        """Turn the group on the spot until its rolling direction lies along +/- u."""
        w = self.w
        t0 = w.data.time
        while w.data.time - t0 < timeout:
            c, f, poses = self.group_pose(cubes)
            target = horizontal(u) if f @ horizontal(u) >= 0 else -horizontal(u)
            err = signed_angle(f, target)
            if abs(math.degrees(err)) < tol:
                break
            rate = float(np.clip(3.0 * err, -2.5, 2.5))
            if abs(rate) < 0.25:
                rate = math.copysign(0.25, rate)
            self.group_speed(cubes, 0.0, rate, poses)
            w.run(CONTROL)
        w.stop(cubes, settle=0.15)

    def follow(self, cubes, start, target, speed=0.15, creep=0.0, creep_speed=0.025, stop=0.0, timeout=30.0, push=0.0, frames=1, blocked=None, track=None):
        """Drive the group's centre along the line from `start` to `target` (steering onto it) until
        it is `stop` m short of the target. Within `creep` m of the target, slow to creep_speed.
        push > 0: on reaching the target keep creeping (pressing on a face) for `push` s.
        frames: camera frames averaged per tick. blocked: stop if the group has moved less than a
        third of what it was asked to over that many seconds (it ran into something); the call
        then returns False. track: steer this cube of the group onto the line instead of the
        group's centre."""
        w = self.w
        u = horizontal(np.asarray(target) - np.asarray(start))
        n = np.cross([0, 0, 1.0], u)
        t0 = w.data.time
        still = 0.0
        history = []
        free = True
        while w.data.time - t0 < timeout:
            c, f, poses = self.group_pose(cubes, frames=frames)
            if track is not None:
                c = poses[cubes.index(track)][0]
            s = float((np.asarray(target) - c)[:2] @ u[:2])
            if s <= stop and push <= 0:
                break
            if blocked:
                history.append((w.data.time, s))
                old = [h for h in history if h[0] <= w.data.time - blocked]
                if old and (old[-1][1] - s) < creep_speed * blocked / 3:
                    free = False
                    break
            e = float((c - np.asarray(target))[:2] @ n[:2])
            sg = 1.0 if f @ u >= 0 else -1.0
            heading = signed_angle(u, sg * f)
            want = -math.atan(e / 0.03)
            v = min(speed, max(creep_speed, math.sqrt(2 * DECEL * max(s - stop, 0.0))))
            if creep > 0 and s < creep:
                v = creep_speed
            yaw_rate = float(np.clip(4.0 * (want - heading), -2.0, 2.0))
            self.group_speed(cubes, sg * v, yaw_rate, poses)
            w.run(CONTROL)
            if push > 0 and s <= stop + 0.001:  # at the face: press gently for `push` s, then stop
                still += CONTROL
                if still >= push:
                    break
        w.stop(cubes, settle=0.15)
        return free

    def goto(self, cubes, target, final=None, speed=0.15, stop=0.0):
        """Turn toward `target`, drive there along the straight line, then turn to `final` (a
        direction) if given."""
        c, f, _ = self.group_pose(cubes, frames=3)
        if np.linalg.norm((np.asarray(target) - c)[:2]) > 0.01:
            self.turn_to(cubes, horizontal(np.asarray(target) - c))
            c, f, _ = self.group_pose(cubes, frames=3)
            self.follow(cubes, c, target, speed=speed, stop=stop)
        if final is not None:
            self.turn_to(cubes, final)

    # ----------------------------------------------------------- docking

    def dock(self, cubes, partners, approach, tries=3, back=0.05):
        """Head-on dock of a group: its cubes end against the cubes `partners` (one each, matched by
        their order across the approach line), approaching along the horizontal `approach`
        direction. Targets are read from the camera; the bond forms only if every face pair is
        inside the capture range. A failed attempt backs off `back` m and tries again. Groups on one
        line of wheels lean 3.4 degrees on an edge: the arriving group ends leaning back off the face
        it touched, so a parked pair partner is first made to lean toward it (both parallel).
        Returns True when bonded."""
        w = self.w
        u = horizontal(approach)
        n = np.cross([0, 0, 1.0], u)
        held = w.group_of(partners[0])
        for attempt in range(tries):
            if len(held) == 2:
                self.tip(held, -u)  # the parked pair leans toward the arriving group, which leans back off the face on contact
            targets = [self.look(j, 3)[0] - u * SIDE for j in partners]
            target = np.mean(targets, axis=0)
            c, f, _ = self.group_pose(cubes, frames=3)
            pre = target - u * 0.05
            if np.linalg.norm((pre - c)[:2]) > 0.006 or abs(abs(f @ u) - 1) > 1e-3:
                self.goto(cubes, pre, final=u, speed=0.12)
            mine = sorted(cubes, key=lambda i: float(self.look(i)[0] @ n))
            theirs = sorted(partners, key=lambda j: float(self.look(j)[0] @ n))
            self.tip(cubes, u)
            c, f, _ = self.group_pose(cubes, frames=3)
            self.follow(cubes, c, target, speed=0.05, creep=0.012, creep_speed=0.015, stop=0.0005)
            self.tip(cubes, -u)  # touching a face leaves a group leaning back off it; make sure
            w.dock_attempts += 1
            checks = [w.capture(i, j) for i, j in zip(mine, theirs)]
            w.events.append({"t": round(w.data.time, 3), "dock": list(mine), "to": list(theirs), "ok": all(ch[0] for ch in checks), "lateral_mm": [round(ch[1] * 1000, 2) for ch in checks], "angle_deg": [round(ch[2], 2) for ch in checks], "gap_mm": [round(ch[3] * 1000, 2) for ch in checks]})
            if all(ch[0] for ch in checks):
                for i, j in zip(mine, theirs):
                    w.bond(i, j, count=False)
                w.run(0.2)
                return True
            w.failed_docks += 1
            c, f, _ = self.group_pose(cubes, frames=3)
            self.follow(cubes, c, c - u * back, speed=0.05)
        return False

    def slide_dock(self, cubes, targets, bond_to, tries=3, gap=0.0018):
        """Dock a group by sliding it in along its rolling direction beside a structure, so the
        bonding faces are normal to the wheel axles (the lean of a cube on its wheel turns such a
        face in its own plane). cubes[k] goes to the camera-read position targets[k] (matched by
        their order across the line); the group cube that arrives next to the structure cube
        `bond_to` bonds to it. The
        group follows a line `gap` m outside the target (a cube cannot move sideways, so it must not
        touch on the way in), stops level with it, both sides lean the way it arrived, then the bonds
        are tried (capture check). Returns True when bonded."""
        w = self.w
        tc = np.mean(targets, axis=0)
        held = [bond_to]
        structure = w.group_of(bond_to)
        for attempt in range(tries):
            c, f, _ = self.group_pose(cubes, frames=3)
            u = horizontal(f)
            a = horizontal(np.mean([t for t in targets], axis=0) - np.mean([self.look(j)[0] for j in held], axis=0))
            u = horizontal(u - a * (u @ a))
            u_app = u if (tc - c) @ u > 0 else -u
            line_end = tc + a * gap
            pre = line_end - u_app * 0.10
            # route around the structure: straight out to `clear` m from its line on the approach
            # side, along to abreast of the target, then in (no path planner beyond this)
            side = -u_app
            clear = 0.2
            if float((c - tc) @ side) < clear - 0.01:
                self.goto(cubes, c + side * (clear - float((c - tc) @ side)), speed=0.12)
            self.goto(cubes, tc + side * clear, speed=0.12)
            self.goto(cubes, pre, final=u_app, speed=0.12)
            self.tip(structure, u_app)
            p_s = self.look(bond_to, 3)[0]
            near = min(cubes, key=lambda i: float(np.linalg.norm(self.look(i, 3)[0] - p_s)))
            near_end = p_s + a * (SIDE + gap)  # the near cube's own line: `gap` m outside its cell
            if not self.follow(cubes, near_end - u_app * 0.10, near_end, speed=0.04, creep=0.02, creep_speed=0.01, stop=0.0005, frames=2, blocked=0.4, track=near):
                w.events.append({"t": round(w.data.time, 3), "slide_dock": list(cubes), "note": "ran into the structure on the way in"})
            self.tip(cubes, u_app)
            self.tip(structure, u_app)
            p_s = self.look(bond_to, 3)[0]
            near = min(cubes, key=lambda i: float(np.linalg.norm(self.look(i)[0] - p_s)))
            for _ in range(3):  # fine: slow nudges on averaged camera frames (they keep the lean)
                err = float((self.look(bond_to, 5)[0] - self.look(near, 5)[0]) @ u_app)
                if abs(err) < 0.0007:
                    break
                self.group_speed(cubes, math.copysign(0.008, err * float(self.group_forward(cubes) @ u_app)), 0.0)
                w.run(abs(err) / 0.008)
                w.set_wheels({i: 0.0 for i in cubes})
                w.run(0.15)
            w.dock_attempts += 1
            bonds = [(near, bond_to)]
            checks = [w.capture(i, j) for i, j in bonds]
            w.events.append({"t": round(w.data.time, 3), "slide_dock": list(cubes), "bonds": [list(b) for b in bonds], "ok": all(ch[0] for ch in checks), "lateral_mm": [round(ch[1] * 1000, 2) for ch in checks], "angle_deg": [round(ch[2], 2) for ch in checks], "gap_mm": [round(ch[3] * 1000, 2) for ch in checks]})
            if all(ch[0] for ch in checks):
                for i, j in bonds:
                    w.bond(i, j, count=False)
                w.run(0.2)
                return True
            w.failed_docks += 1
            c, f, _ = self.group_pose(cubes, frames=3)
            self.follow(cubes, c, c - u_app * 0.12, speed=0.06)
        return False

    def climb(self, mover, support, down=False):
        """Half turn of `mover` about the top edge it shares with `support`: up onto the support
        from beside it, or (down) from on top of it to the floor cell beside it, on the side the
        planner names by passing the support's neighbour cell through `support` geometry."""
        w = self.w
        pm, ps = self.look(mover, 3)[0], self.look(support, 3)[0]
        d = horizontal(ps - pm)
        point = (pm + ps) / 2 + np.array([0, 0, HALF])
        axis = np.cross([0, 0, 1.0], d)
        return w.pivot(mover, point, axis, math.pi, support=support, bond_to="near")

    def climb_down(self, mover, support, toward):
        """Half turn of `mover`, on top of `support`, down to the floor cell beside the support in
        the horizontal direction `toward`."""
        w = self.w
        ps = self.look(support, 3)[0]
        d = horizontal(toward)
        point = ps + d * HALF + np.array([0, 0, HALF])
        axis = np.cross([0, 0, 1.0], d)
        return w.pivot(mover, point, axis, math.pi, support=support, bond_to="near")

    def traverse(self, mover, onto):
        """Quarter turn of `mover`, lying on top of a structure, across the top edge onto the top of
        the neighbouring cube `onto` (latched to it)."""
        w = self.w
        pm, po = self.look(mover, 3)[0], self.look(onto, 3)[0]
        d = horizontal(po + np.array([0, 0, SIDE]) - pm)
        point = pm + d * HALF - np.array([0, 0, HALF])
        axis = np.cross([0, 0, 1.0], d)
        return w.pivot(mover, point, axis, math.pi / 2, support=onto, bond_to="near")

    # ----------------------------------------------------------- pairing two loose cubes

    def pair_up(self, a, b, free_side=None, tries=3):
        """Two loose cubes, wheel faces down, become a bonded differential pair (axles in a line).

        1. If their rolling directions are within 35 degrees of parallel, b turns 90 degrees (three
           floor pivots).
        1. a drives along its line to where b's line crosses it; b drives along its line into a's
           side face and pushes until its front face lies flat on it (this squares b's yaw).
        1. a drives along its line until b's centre is level with its own (b backed off 3 mm).
        1. b latches its vertical edge to a's and turns a quarter about it (a turns some too: the
           reaction), into the cell diagonal to a, axle now parallel to a's.
        1. a leans the way b will arrive; b drives along its line into the cell beside a; bond.
        Returns True when bonded."""
        w = self.w
        d_a, d_b = w.rolling(a), w.rolling(b)
        if abs(d_a @ d_b) > math.cos(math.radians(55)):
            side = horizontal(w.pos(b) - w.pos(a))
            outs = self.turn_heading(b, side, away=w.pos(a))
            if any(o not in ("arrived", "near") for o in outs):
                raise Failure(f"heading turn pivots {outs}")
        for attempt in range(tries):
            pa, ra = self.look(a, 3)
            pb, rb = self.look(b, 3)
            d_a, d_b = w.rolling(a, ra), w.rolling(b, rb)
            # where b's line crosses a's line
            m = np.array([[d_a[0], -d_b[0]], [d_a[1], -d_b[1]]])
            sa, sb = np.linalg.solve(m, (pb - pa)[:2])
            cross = pa + d_a * sa
            if abs(sb) < 0.11:  # b too close to the crossing: back off along its line first
                self.drive_to(b, pb + d_b * (math.copysign(0.13, -sb) + sb) * -1 if False else pb - d_b * math.copysign(0.13 - abs(sb), sb))
            self.drive_to(a, cross)
            pa, ra = self.look(a, 3)
            self.push(b, pa)
            # which face of a did b meet?
            pa, ra = self.look(a, 3)
            pb, rb = self.look(b, 3)
            x_a = w.rolling(a, ra)
            y_a = np.cross([0, 0, 1.0], x_a)
            rel = pb - pa
            if abs(rel @ y_a) < abs(rel @ x_a):  # met a's front or back: a turns 90 degrees and b tries again
                w.events.append({"t": round(w.data.time, 3), "pair": [a, b], "note": "met front face; a turns"})
                self.drive_to(b, pb + horizontal(rel) * 0.08)
                outs = self.turn_heading(a, -horizontal(rel) if free_side is None else free_side, away=w.pos(b))
                if any(o not in ("arrived", "near") for o in outs):
                    raise Failure(f"heading turn pivots {outs}")
                continue
            sy = 1.0 if rel @ y_a > 0 else -1.0
            # line b up with a: b backs off 3 mm, a drives so that b's centre is level with it
            self.drive_to(b, pb + horizontal(rel) * 0.003, speed=0.03, tol=0.0005)
            for _ in range(3):
                pa, ra = self.look(a, 3)
                pb, rb = self.look(b, 3)
                x_a = w.rolling(a, ra)
                off = float((pb - pa)[:2] @ x_a[:2])
                if abs(off) < 0.0012:
                    break
                self.drive_to(a, pa + x_a * off, speed=0.03, tol=0.0005)
            pa, ra = self.look(a, 3)
            self.push(b, pa, seconds=0.3, speed=0.02)
            # quarter turn about the shared vertical edge, away from a
            pa, ra = self.look(a, 3)
            x_a = w.rolling(a, ra)
            y_a = np.cross([0, 0, 1.0], x_a)
            if free_side is not None:
                sx = 1.0 if horizontal(free_side) @ x_a >= 0 else -1.0
            else:
                sx = 1.0
            edge = pa + sx * HALF * x_a + sy * HALF * y_a
            out = w.pivot(b, edge, [0, 0, -sx * sy], math.pi / 2 + math.radians(OVERTURN), support=a)
            if out not in ("arrived", "near"):
                w.events.append({"t": round(w.data.time, 3), "pair": [a, b], "note": f"vertical pivot {out}"})
                if out == "unlatched":
                    continue
                raise Failure(f"vertical pivot {out}")
            # trim b's yaw while the two corner edges still coincide (the turn stops short or long
            # by a few degrees: friction, and the cubes settling on wheel and edge after release)
            for _ in range(2):
                pa, ra = self.look(a, 5)
                pb, rb = self.look(b, 5)
                x_a, d_b = w.rolling(a, ra), w.rolling(b, rb)
                y_a = np.cross([0, 0, 1.0], x_a)
                along = d_b if d_b @ x_a >= 0 else -d_b
                err = math.degrees(signed_angle(x_a, along))
                bias = -sx * sy * OVERTURN
                if abs(err - bias) <= 2.5:
                    break
                edge = pa + sx * HALF * x_a + sy * HALF * y_a
                out = w.pivot(b, edge, [0, 0, 1.0], math.radians(bias - err), support=a, tolerance=0.5)
                w.events.append({"t": round(w.data.time, 3), "pair": [a, b], "note": f"yaw trim {bias - err:.1f} deg: {out}"})
            # b slides along its line into the cell beside a; a leans the way b arrives
            pa, ra = self.look(a, 3)
            x_a = w.rolling(a, ra)
            y_a = np.cross([0, 0, 1.0], x_a)
            for k in range(tries):
                pb, rb = self.look(b, 3)
                beside = pa + sy * SIDE * y_a
                arrive = horizontal(beside - pb)
                self.tip([a], arrive)
                pa, ra = self.look(a, 3)
                x_a = w.rolling(a, ra)
                y_a = np.cross([0, 0, 1.0], x_a)
                beside = pa + sy * SIDE * y_a
                self.drive_to(b, beside, speed=0.06, fine=False)
                self.tip([b], arrive)
                self.tip([a], arrive)
                pa, ra = self.look(a, 5)
                beside = pa + sy * SIDE * np.cross([0, 0, 1.0], w.rolling(a, ra))
                self.drive_to(b, beside, speed=0.03, tol=0.0006)
                pa, ra = self.look(a, 5)
                pb, rb = self.look(b, 5)
                rel = pb - pa
                if abs(rel @ w.rolling(a, ra)) > 0.015 or abs(abs(rel @ np.cross([0, 0, 1.0], w.rolling(a, ra))) - SIDE) > 0.012:
                    w.events.append({"t": round(w.data.time, 3), "pair": [a, b], "note": "b is not beside a: give up"})
                    return False
                w.dock_attempts += 1
                ok = w.bond(b, a, count=False)
                if ok:
                    w.run(0.2)
                    return True
                w.failed_docks += 1
                # back into the diagonal cell; if the camera reads a yaw error, latch the shared
                # vertical edge again and trim it with a small pivot
                pa, ra = self.look(a, 5)
                x_a = w.rolling(a, ra)
                y_a = np.cross([0, 0, 1.0], x_a)
                diagonal = pa + sx * SIDE * x_a + sy * SIDE * y_a
                self.drive_to(b, diagonal, speed=0.05)
                pa, ra = self.look(a, 5)
                pb, rb = self.look(b, 5)
                x_a, d_b = w.rolling(a, ra), w.rolling(b, rb)
                y_a = np.cross([0, 0, 1.0], x_a)
                along = d_b if d_b @ x_a >= 0 else -d_b
                err = math.degrees(signed_angle(x_a, along))  # b's line relative to a's, about +z
                bias = -sx * sy * OVERTURN  # the outward lean of b's line that the overturn gives
                if abs(err - bias) > 1.5:
                    edge = pa + sx * HALF * x_a + sy * HALF * y_a
                    out = w.pivot(b, edge, [0, 0, 1.0], math.radians(bias - err), support=a)
                    w.events.append({"t": round(w.data.time, 3), "pair": [a, b], "note": f"yaw trim {bias - err:.1f} deg: {out}"})
                pa, ra = self.look(a, 3)
                x_a = w.rolling(a, ra)
                y_a = np.cross([0, 0, 1.0], x_a)
            return False
        return False


# ---------------------------------------------------------------------- packing check (arithmetic)
#
# Boxes in the cube's frame, mm, centre at the origin: +x is the disc face, -x the wheel face, the
# wheel's axle along y and its rolling direction along z (as in the MuJoCo model). Walls are 2 mm,
# so the inside is -23..23 on every axis. Every size is marked bought (a catalogue size written from
# memory, not checked against a datasheet) or guess.

WALL_MM = 2.0
PARTS = [
    # name, (x0, x1), (y0, y1), (z0, z1), source
    ("disc (42 x 3, in the +x wall)", (22.0, 25.0), (-21.0, 21.0), (-21.0, 21.0), "guess"),
    ("disc bearing 6803 (26 OD x 5)", (17.0, 22.0), (-13.0, 13.0), (-13.0, 13.0), "bought"),
    ("disc gearmotor N20 (12 x 10 x 24), off the disc axis, spur gear to the hub", (-7.0, 17.0), (9.0, 21.0), (-5.0, 5.0), "bought"),
    ("wheel (20 x 6), flush position", (-25.0, -5.0), (-3.0, 3.0), (-10.0, 10.0), "guess"),
    ("wheel carriage and spring (3 mm behind the wheel)", (-5.0, -2.0), (-6.0, 6.0), (-10.0, 10.0), "guess"),
    ("wheel gearmotor N20, axis along y, rides on the carriage (+1.5 mm travel)", (-23.0, -11.5), (-12.0, 12.0), (11.0, 23.0), "bought"),
    ("pivot gearmotor N20, axis along y (its drive train to the edges is not designed)", (-1.0, 11.0), (-12.0, 12.0), (-23.0, -13.0), "bought"),
    ("1S LiPo (20 x 25 x 5)", (-10.0, 15.0), (-21.0, -1.0), (13.0, 18.0), "bought"),
    ("ESP32-C3 SuperMini (22.5 x 18 x 4 with parts)", (-10.0, 12.5), (-23.0, -19.0), (-9.0, 9.0), "bought (thickness guess)"),
]


def packing(parts=PARTS, wall=WALL_MM, side=SIDE * 1000):
    """Check the boxes fit inside the walls and do not overlap. Returns a dict: fits, outside (parts
    poking through a wall, the disc and wheel excepted: they sit in their faces), overlaps (pairs),
    part and free volumes (cm3), and the free volume within 4 mm of each face (for connectors)."""
    inner = side / 2 - wall
    outside, overlaps = [], []
    for name, *box, _ in parts:
        if "disc (" in name or "wheel (" in name:
            continue
        if any(lo < -inner - 1e-9 or hi > inner + 1e-9 for lo, hi in box):
            outside.append(name)
    for (na, *a, _), (nb, *b, _) in itertools.combinations(parts, 2):
        if all(min(a[k][1], b[k][1]) - max(a[k][0], b[k][0]) > 1e-9 for k in range(3)):
            overlaps.append((na, nb))

    def volume(box, clip):
        v = 1.0
        for lo, hi in box:
            lo, hi = max(lo, clip[0]), min(hi, clip[1])
            v *= max(hi - lo, 0.0)
        return v

    interior = (2 * inner) ** 3
    used = sum(volume(box, (-inner, inner)) for _, *box, _ in parts)
    faces = {}
    for k, axis in enumerate("xyz"):
        for sign in (1, -1):
            layer = [(-inner, inner)] * 3
            layer[k] = (inner - 4, inner) if sign > 0 else (-inner, -inner + 4)
            taken = sum(volume([(max(b[j][0], layer[j][0]), min(b[j][1], layer[j][1])) for j in range(3)], (-inner, inner)) for _, *b, _ in parts)
            faces[("+" if sign > 0 else "-") + axis] = round(((2 * inner) ** 2 * 4 - taken) / 1000, 2)
    return {
        "fits": not outside and not overlaps,
        "outside": outside,
        "overlaps": overlaps,
        "interior_cm3": round(interior / 1000, 2),
        "parts_cm3": round(used / 1000, 2),
        "free_cm3": round((interior - used) / 1000, 2),
        "free_4mm_face_layer_cm3": faces,
    }
