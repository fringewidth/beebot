"""Controlled pivots: cubes that turn about a latched edge under a geared, torque-limited motor.

The redesign candidate of HARDWARE.md ("Redesign options"). cube_swarm's cubes pivot ballistically:
a braked flywheel throws the mover about an edge and magnets have to catch it. Here the same cubes
(mass, size, magnets, contacts, placement error and jitter of cube_swarm) instead latch the edge
they turn about to their support and turn about it under position control, like a servo hinge.
The mechanism that would do this (a latch and a motor at whichever edge is the pivot) is not
modelled; what is modelled is what any such mechanism has to live with:

1. Latch. While a cube moves, its pivot edge is held to the support's matching edge (a cube's, or
   the floor's when `floor_latch`: a floor with latch features at the cell edges) by two point
   constraints a little inside the edge's ends (MuJoCo `connect`, soft), i.e. a hinge. The latch
   engages only when the mover's edge lies within `catch` L of the support's edge (placement error
   and drift); otherwise the move is refused before anything moves and counts as stayed.
1. Motor. A torque about the hinge, applied to the mover and, opposite, to the support (a lone
   support feels the reaction, as it would), clipped to `torque` N m. A PD law tracks a
   minimum-jerk reference from 0 to 90 or 180 degrees at a mean `rate` deg/s.
1. Edges. Between the mover and the cubes it leans on (its support, and for a quarter turn the
   cube it comes to rest against) there is no contact while it turns: the four boxes meet at the
   hinge edge, and MuJoCo's box contacts there (squeezed by the magnets) jam a turn that chamfered
   or rounded edges would let pass. The latch fixes where the mover is relative to its support;
   contact comes back when the mover arrives (or has come back).
1. Magnets. The mover's magnets (cube_swarm's model) are all off for the move, toward cubes and
   toward the steel floor (switchable magnets: electropermanent, or electromagnets held off); the
   support and every other cube keep theirs. When the mover arrives (within `tolerance` degrees)
   its magnets come back on, then the latch opens.
1. Stall. If the mover has not arrived `stall` s after the reference ends (held by friction from
   squeezed neighbours, or by too little torque), the controller drives it back to 0 degrees, then
   releases: a harmless failure by construction, as long as the way back is free. A stall within
   `near` degrees of the goal (typically the mover's far edge against a cube that sits a little
   into the target cell) counts as arrived: the magnets come on and pull it home.

What is judged is cube_swarm's (Swarm.moves): reached, stayed, off, strayed, knock. The flywheel
of cube_swarm is still in the model (its mass) and stays braked.

    from beebot.hardware.cube_hinge import HingeSwarm, Drive
    swarm = HingeSwarm(patch(1, 5), seed=0, drive=Drive(torque=0.3))
    swarm.move(4, (1, 0, -1))
"""

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from beebot.hardware.cube import DT, SETTLE
from beebot.hardware.cube_swarm import Swarm


@dataclass(frozen=True)
class Drive:
    torque: float = 0.3  # N m: the motor's limit at the hinge (a micro gearmotor at stall: 0.2-0.5)
    rate: float = 150.0  # deg/s: mean angular rate of the reference (90 degrees in 0.6 s)
    kp: float = 3.0  # N m / rad
    kd: float = 0.05  # N m s / rad
    catch: float = 0.06  # L: the latch engages when the mover's edge is this close to the support's
    tolerance: float = 3.0  # degrees: arrived
    stall: float = 0.5  # s past the reference's end before the controller turns back
    near: float = 10.0  # degrees: a stall this close to the goal counts as arrived (the magnets pull it home)
    hold: float = 0.1  # s: magnets on, then latch off after this long, then SETTLE
    floor_latch: bool = True  # the floor has latch features at the cell edges (a special floor)
    stiffness: float = 0.005  # s: the latch constraint's time constant (MuJoCo solref)
    face_friction: float = None  # friction of the cubes' faces (slippery skins); the floor keeps the scene's. MuJoCo takes the larger of two


def smooth(u):
    """Minimum-jerk profile on [0, 1] and its derivative."""
    u = min(max(u, 0.0), 1.0)
    return 10 * u**3 - 15 * u**4 + 6 * u**5, 30 * u**2 - 60 * u**3 + 30 * u**4


class Hinge:
    """One mover's latch and controller for one pivot."""

    def __init__(self, swarm, cube, pivot, reading, slot):
        L = swarm.cube.side
        drive = swarm.drive
        here = np.array(reading.cells[cube][::-1])
        leans = [np.array(c[::-1]) for c in pivot.leans]
        s = leans[0] - here
        half = len(leans) == 1
        h = np.array(pivot.target[::-1]) - here - (s if half else 0)
        self.cube = cube
        self.axis = np.array(pivot.axis, dtype=float)
        self.goal = math.pi if half else math.pi / 2
        self.duration = math.degrees(self.goal) / drive.rate
        at = {tuple(int(v) for v in c[::-1]): i for i, c in enumerate(reading.cells)}
        support_cell = tuple(int(v) for v in here + s)
        self.support = at.get(support_cell)  # cube index, or None for the floor
        data = swarm.data
        body = swarm.bodies[cube]
        x_m, R_m = data.xpos[body].copy(), data.xmat[body].reshape(3, 3).copy()
        edge = x_m + (h + s) / 2 * L  # the mover's own pivot edge (it is aligned: within 10 degrees)
        if self.support is None:
            self.support_body = 0
            x_s, R_s = np.zeros(3), np.eye(3)
            mate = (here + (h + s) / 2 + [0, 0, 0.5]) * L  # the floor's latch at the cell edge
        else:
            self.support_body = swarm.bodies[self.support]
            x_s, R_s = data.xpos[self.support_body].copy(), data.xmat[self.support_body].reshape(3, 3).copy()
            mate = x_s + (h - s) / 2 * L  # world offsets: the support may lie turned
        gap = edge - mate
        gap_across = gap - self.axis * (gap @ self.axis)
        self.miss = float(np.abs(gap_across).max() / L)
        self.latched = self.miss <= drive.catch and (self.support is not None or drive.floor_latch)
        self.R0 = R_s.T @ R_m  # mover relative to support at the start
        self.axis_s = R_s.T @ self.axis  # the hinge axis in the support's frame
        self.slot = slot
        self.leans = [at[c[::-1]] for c in pivot.leans if c[::-1] in at]  # cubes it does not touch meanwhile (chamfered edges)
        if self.latched:
            for k, end in zip(slot, (-0.35, 0.35)):
                point = mate + end * L * self.axis
                swarm.model.eq_obj1id[k] = body
                swarm.model.eq_obj2id[k] = self.support_body
                swarm.model.eq_data[k][:3] = R_m.T @ (edge + end * L * self.axis - x_m)
                swarm.model.eq_data[k][3:6] = R_s.T @ (point - x_s)
                swarm.data.eq_active[k] = 1
        self.angle = 0.0
        self.phase = "go" if self.latched else "done"
        self.clock = 0.0
        self.start = 0.0  # angle the present reference starts from
        self.end = self.goal
        self.arrived = None  # time the mover arrived (or came back)
        self.outcome = "go"
        self.peak = 0.0  # N m: the largest hinge torque asked for (before the clip)

    def frames(self, swarm):
        data = swarm.data
        R_m = data.xmat[swarm.bodies[self.cube]].reshape(3, 3)
        R_s = np.eye(3) if self.support is None else data.xmat[self.support_body].reshape(3, 3)
        return R_m, R_s

    def measure(self, swarm):
        """Hinge angle (rad, unwrapped) and rate (rad/s) of the mover relative to its support."""
        R_m, R_s = self.frames(swarm)
        turn = R_s.T @ R_m @ self.R0.T  # in the support's frame
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, turn.reshape(-1))
        angle = 2 * math.atan2(float(q[1:] @ self.axis_s), float(q[0]))
        angle += 2 * math.pi * round((self.angle - angle) / (2 * math.pi))
        self.angle = angle
        w_m = swarm.data.cvel[swarm.bodies[self.cube]][:3]
        w_s = np.zeros(3) if self.support is None else swarm.data.cvel[self.support_body][:3]
        return angle, float((w_m - w_s) @ (R_s @ self.axis_s))

    def torque(self, swarm, dt):
        """Advance the controller by dt; returns the hinge torque (N m) and the world axis."""
        drive = swarm.drive
        self.clock += dt
        angle, rate = self.measure(swarm)
        span = self.end - self.start
        duration = max(abs(span) / math.radians(drive.rate), 1e-3)
        u, du = smooth(self.clock / duration)
        target, target_rate = self.start + span * u, span * du / duration
        near = abs(angle - self.end) < math.radians(drive.tolerance)
        if self.phase in ("go", "back") and near and self.clock >= duration:
            self.arrived = self.clock
            self.outcome = "arrived" if self.phase == "go" else "returned"
            self.phase = "hold"
        elif self.phase == "go" and self.clock > duration + drive.stall and abs(angle - self.end) < math.radians(drive.near):
            self.arrived = self.clock  # stopped short against a cube at the target: magnets on, they close the gap
            self.outcome = "near"
            self.phase = "hold"
        elif self.phase == "go" and self.clock > duration + drive.stall:
            self.phase, self.start, self.end, self.clock = "back", angle, 0.0, 0.0
            self.outcome = "stalled"
        elif self.phase == "back" and self.clock > duration + drive.stall:
            self.phase = "hold"  # could not get back either: give up where it is
            self.arrived = self.clock
            self.outcome = "stuck"
        if self.phase == "back":
            span = self.end - self.start
            duration = max(abs(span) / math.radians(drive.rate), 1e-3)
            u, du = smooth(self.clock / duration)
            target, target_rate = self.start + span * u, span * du / duration
        if self.phase == "hold":
            target, target_rate = self.end, 0.0
        wanted = drive.kp * (target - angle) + drive.kd * (target_rate - rate)
        self.peak = max(self.peak, abs(wanted))
        tau = float(np.clip(wanted, -drive.torque, drive.torque))
        _, R_s = self.frames(swarm)
        return tau, R_s @ self.axis_s


def servo_row(hinge):
    """What the servo did on one move, for an outcome row."""
    return {"servo": hinge.outcome if hinge.latched else "unlatched", "miss": round(hinge.miss, 4), "peak": round(hinge.peak, 3)}


class HingeSwarm(Swarm):
    """cube_swarm.Swarm whose cubes pivot about a latched edge under a torque-limited servo instead
    of a flywheel kick (module docstring). Magnets are switchable (release) always."""

    def __init__(self, cells, drive=Drive(), **options):
        self.drive = drive
        options.pop("release", None)
        options.pop("speeds", None)
        super().__init__(cells, release=True, **options)
        self.angular = self.cube_dofs[:, 3:]  # the free joints' angular dofs (body frame)
        self.hinges = []

    def scene_xml(self, centres):
        xml = super().scene_xml(centres)
        if self.drive.face_friction is not None:  # cube on cube slides at face_friction, cube on floor grips at the scene's
            xml = xml.replace('type="box"', f'type="box" friction="{self.drive.face_friction} 0.005 0.0001"').replace('type="plane"', f'type="plane" friction="{self.friction} 0.005 0.0001"')
        solref = f"{self.drive.stiffness} 1"
        latches = "".join(f'<connect name="latch{i}{end}" body1="c{i}" anchor="0 0 0" active="false" solref="{solref}"/>' for i in range(self.n) for end in "ab")
        return xml.replace("</mujoco>", f"<equality>{latches}</equality>\n</mujoco>")

    def moves(self, orders, rpm=None, watch=None):
        self.hinges = []
        self.reading = self.lattice()
        self.pivots = {cube: self.plan(cube, direction, self.reading) for cube, direction in orders}
        return super().moves(orders, rpm, watch)

    def actuate(self, spins, limit, cut=None, watch=None, quiet=None):
        """Carry out the pivots of the movers in spins [(cube, axis, rpm)] (the rpm is unused):
        latch, magnets off, servo to the goal (or back), magnets on, unlatch, settle."""
        hinges = [Hinge(self, cube, self.pivots[cube], self.reading, (2 * cube, 2 * cube + 1)) for cube, _, _ in spins]
        self.hinges = hinges
        active = [hinge for hinge in hinges if hinge.latched]
        if not active:
            return
        floor = self.magnets.floor.copy()
        sites = len(self.magnets.local)
        off = np.zeros((self.n, self.n), dtype=bool)
        for hinge in active:
            off[hinge.cube] = off[:, hinge.cube] = True
            off[hinge.cube, hinge.cube] = False
            self.magnets.floor[hinge.cube * sites : (hinge.cube + 1) * sites] = 0.0
        self.magnets.switch(off)
        contype, conaffinity = self.model.geom_contype.copy(), self.model.geom_conaffinity.copy()
        geom = self.model.body_geomadr[self.bodies]
        self.model.geom_conaffinity[:] = 3  # the floor and every cube: 1 | 2
        for hinge in active:  # mover (2, 2) and the cubes it leans on (1, 1) do not touch
            self.model.geom_contype[geom[hinge.cube]] = self.model.geom_conaffinity[geom[hinge.cube]] = 2
            for j in hinge.leans:
                self.model.geom_contype[geom[j]] = self.model.geom_conaffinity[geom[j]] = 1
        horizon = max(2 * (h.duration + self.drive.stall) for h in active) + 1.0
        elapsed = 0.0
        released = set()
        while elapsed < horizon and len(released) < len(active):
            torques = np.zeros((self.n, 3))
            for hinge in active:
                if hinge.cube in released:
                    continue
                tau, axis = hinge.torque(self, DT)
                torques[hinge.cube] += tau * axis
                if hinge.support is not None:
                    torques[hinge.support] -= tau * axis
                if hinge.phase == "hold" and self.model.geom_contype[geom[hinge.cube]] == 2:  # arrived: touches again
                    self.model.geom_contype[geom[hinge.cube]] = 1
                if hinge.phase == "hold" and hinge.clock - hinge.arrived > self.drive.hold:
                    on = np.zeros((self.n, self.n), dtype=bool)
                    for other in active:  # this mover's magnets back on; the others' still off
                        if other is not hinge and other.cube not in released:
                            on[other.cube] = on[:, other.cube] = True
                            on[other.cube, other.cube] = False
                    self.magnets.floor[hinge.cube * sites : (hinge.cube + 1) * sites] = floor[hinge.cube * sites : (hinge.cube + 1) * sites]
                    self.magnets.switch(on if on.any() else None)
                    hinge.phase = "magnets"
                    hinge.arrived = hinge.clock
                elif hinge.phase == "magnets" and hinge.clock - hinge.arrived > self.drive.hold:
                    for k in hinge.slot:
                        self.data.eq_active[k] = 0
                    released.add(hinge.cube)
            rot = self.data.xmat[self.bodies].reshape(self.n, 3, 3)
            self.data.qfrc_applied[self.angular] = np.einsum("nji,nj->ni", rot, torques)  # world torque into each body's frame
            self.step()
            elapsed += DT
            if watch is not None:
                watch(self)
        for hinge in active:  # anything still latched at the horizon
            for k in hinge.slot:
                self.data.eq_active[k] = 0
            if hinge.cube not in released:
                hinge.outcome = "timeout"
        self.data.qfrc_applied[self.angular] = 0.0
        self.model.geom_contype[:], self.model.geom_conaffinity[:] = contype, conaffinity
        self.magnets.floor[:] = floor
        self.magnets.switch(None)
        self.run(SETTLE, watch)
