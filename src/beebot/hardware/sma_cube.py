"""SMA cube (SC): one integrated cube of side 8.25-11 mm for the Beebot swarm, every move quasi-static.

Memo: runs/handoff/sma-cube.md. Behaviours, Monte Carlo, optimisation and media:
scripts/module_phys/sma_cube_sim.py. Tests: tests/test_sma_cube.py.

The cube (body frame: twist discs on +x and -x, plain faces +-y and +-z):

1. Two SMA ratchet twist joints, one per disc (runs/handoff/small-joint.md, geared_joint.SmaStepper).
1. Eight free N52 corner spheres for plain-face bonds, set back from the disc faces so they do not
   act across a disc; each disc carries four ring spheres and every plain face four steel inserts at
   the ring positions, so a disc bonds to any face and turns it.
1. Four release carriers, one per x-parallel edge: an SMA wire pulls the two corner spheres of that
   edge inward along the edge bisector, so a face bond can be opened edge by edge (one edge holds as a
   magnetic hinge while the other lets go). Disc bonds open by turning the disc 45 degrees (ring spheres
   off the steel inserts).
1. Four edge legs ("flaps"), each a resin plate lying in a face and hinged at the lattice edge line,
   driven through a tendon by a small SMA ratchet capstan with a power-off lock:
   x-legs XP (edge +y/+z, lies in +y) and XN (the same edge +y/+z, lies in +z) pivot the cube about
   x-parallel edges (rolls, climbs, descents and splits in structures). The two x-legs share one hinge
   pin with interleaved knuckles, so on that edge one leg lifts and the other brakes: past the balance
   point of a 180 deg descent (or a climb's last 45 deg) the second leg rests on the support and
   lowers the cube a tooth at a time instead of letting it swing on the magnetic hinge. d-legs DP
   (edge +x/-y, lies in -y) and DN (edge -x/-z, lies in -z) tip it on and off its disc faces (heading
   changes on the floor). Until 2026-11-16 XN sat on the -y/-z edge (lies in -y) and DP on +x/+z
   (lies in +z): no edge carried two legs and descents swung 135 deg free. Every
   leg tip carries a steel shim that anchors on a neighbour's corner spheres, so a pivot is held
   through the whole swing (no free fall) wherever a neighbour is there to anchor on.
1. Storage (CeraCharge cells or a 301010 LiPo) plus a polymer-tantalum pulse bank; chain power through
   the nickel-plated corner spheres.

Three layers, as geared_joint.py: (1) design physics without MuJoCo (`Design`, bonds, actuators,
mass, packing, power, cost); (2) a MuJoCo world of free cubes with discs, legs, release carriers and
explicit dipole magnetics (`World`); (3) a discrete move model on the voxel grid (`moves`, `bfs`).

Every number is tagged V (vendor page read in this repo's handoffs), E (estimated from stated physics)
or A (assumed). Every simplification is in IDEALISATIONS.
"""

from __future__ import annotations

import itertools
import os
import math
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache

import mujoco
import numpy as np

from beebot.hardware.geared_joint import FLEXINOL, NITINOL_C, NITINOL_DENSITY, NITINOL_LATENT, BondModel, SmaStepper

G = 9.81
MU0 = 4e-7 * math.pi
BR_N52 = 1.45  # T (grade nominal, as mmsim)
RHO = {"ndfeb": 7500.0, "resin": 1150.0, "steel": 7850.0, "flex": 1800.0, "nitinol": NITINOL_DENSITY, "lipo": 1170.0, "cera": 3100.0, "tant": 4000.0, "silicon": 2330.0}

IDEALISATIONS = [
    "Corner and ring spheres are free N52 dipoles whose moments align with the local field of the other cubes' magnets at every magnet update (quasi-static pocket rotation); pocket friction and sphere rattle are not simulated; magnets inside one cube do not act on each other.",
    "Magnet forces are exact dipole-dipole forces (exact for uniformly magnetised spheres), derated by the Monte Carlo lot factor (nominal 0.80, scaling.py/simstack calibration) with a per-magnet moment spread.",
    "Steel inserts and leg-tip shims are induced dipoles aligned with the local field, with a moment that grows linearly with the field up to saturation; their strength relative to a sphere pair is an assumed factor (A, simstack.md: thin steel and thin magnets are overpredicted 2.4-3.5x by the image method, so the factor is set low and varied).",
    "Cubes are rigid boxes with sharp edges (MuJoCo box contacts); legs are thin rigid plates; discs are rigid cylinders standing proud by the tread height.",
    "MuJoCo soft contacts (time constant 1 ms, damping ratio varied for restitution); not Hertz impact physics. The fall uses a separate stiffer contact for its load estimate.",
    "Leg, joint and release actuators are ratchet steppers: whole teeth, a minimum-jerk stroke when the wire is ready, a stroke against a load above the wire's drive stalls and falls back, a lock that holds both ways up to its tooth strength (exceeding it counts as broken). Wire fatigue, creep and temperature build-up are not modelled; cooling time is a Monte Carlo variable.",
    "Leg hinges sit exactly on the lattice edge line (a tendon knuckle inside a 0.5 mm edge chamfer, A); the chamfer is not in the collision geometry.",
    "Release carriers move kinematically: they advance at the wire's stroke rate while the wire force (through its lever) exceeds the magnetic pull on their spheres along the stroke, else they stall; their return is the spring plus the magnets.",
    "The controller knows the planned move, counts steps and reads the IMU tilt without noise or lag (it stops a pivot when the body has turned the planned angle, a KICK at the balance point, and corrects a yaw); bond state is read without noise for classification.",
    "Latches are MuJoCo welds that close when two plain faces with parallel x sit inside the hooks' lead-in (0.35 mm, 4 deg, A) and open when both bolts on that face pass the hook travel; hook play, hook stiffness and load sharing between hooks are not simulated. Machines judge latched bonds against Design.bonds() (hook bending at 30 MPa in resin, E).",
    "The Monte Carlo runs at dt 2e-4 s (checked against 1e-4 s on a climb: same outcome, landing within 0.2 mm); the fall runs at 2e-5 s.",
    "With-retry rates assume a retry from a recoverable failure (stayed, unlatched, bolt stall) succeeds at the single-try rate, independently.",
    "Energy per SMA stroke is the heating minimum (sensible plus latent heat) times 1.25; real pulses with conduction losses cost 1.5-3x (A).",
    "The design search ranks reliability by capacity margins against needs measured at the 14 mm point (scaled by m g L); only the chosen point has a Monte Carlo.",
    "Print tolerance enters as sphere pocket position (per axis), cube size, leg length and hinge position; face flatness and warp are not modelled.",
    "No air drag; electrical limits enter the energy and time budget, not the dynamics.",
]


# ---------------------------------------------------------------------------------------------
# Magnet physics
# ---------------------------------------------------------------------------------------------


def sphere_moment(d: float) -> float:
    """Magnetic moment (A m2) of an N52 sphere of diameter d."""
    return BR_N52 / MU0 * math.pi / 6 * d**3


def pair_force(d: float, gap: float, derate: float = 0.80) -> float:
    """Attraction (N) of two free N52 spheres of diameter d, surfaces `gap` apart (aligned head to
    tail, the state free spheres take). Exact dipole value, derated."""
    m = sphere_moment(d)
    r = d + gap
    return derate * 3 * MU0 * m * m / (2 * math.pi * r**4)


def insert_force(d_sphere: float, gap: float, k_insert: float, derate: float = 0.80) -> float:
    """Attraction of a sphere to a steel insert (A: k_insert times the pull of a sphere pair at the same
    centre distance; the image method gives at most 1 for a thick plate, thin inserts saturate)."""
    return k_insert * pair_force(d_sphere, gap, derate)


# ---------------------------------------------------------------------------------------------
# Actuators
# ---------------------------------------------------------------------------------------------


def wire_mass(d_wire: float, length: float, strands: int) -> float:
    return strands * length * math.pi * d_wire**2 / 4 * NITINOL_DENSITY


def heat_energy(d_wire: float, length: float, strands: int) -> float:
    """Heat (J) to take wires through their transformation from 25 C, plus 25 % lost in a short
    pulse (as SmaStepper.step_energy, E)."""
    return 1.25 * wire_mass(d_wire, length, strands) * (NITINOL_C * 75.0 + NITINOL_LATENT)


def wire_pull(d_wire: float, strands: int) -> float:
    return strands * FLEXINOL[d_wire][1]


def wire_resistance(d_wire: float, length: float) -> float:
    return FLEXINOL[d_wire][0] * length


@dataclass(frozen=True)
class Ratchet:
    """A small SMA ratchet capstan (legs) or wire-and-lever slide (release): numbers the simulation
    needs. Rotary: step in rad, torques in N m. Linear: step in m, forces in N."""

    step: float
    stroke: float  # s, heating pulse plus transformation lag
    period: float  # s between step starts in one direction
    drive: float
    hold: float
    stiffness: float
    clearance: float
    step_energy: float  # J per step


# ---------------------------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------------------------

STORAGE = {
    # name: (energy J, continuous W, volume mm3, mass g, price at 10 / 1k / 1M USD, verified, note)
    "cera1": (0.50, 1.4e-3, 16.0, 0.040, (6.0, 1.5, 0.40), False, "TDK CeraCharge 1812 100 uAh at 1.4 V (V capacity, from simstack); 1 mA continuous (V); price A (bench_kit: 6 USD at 10, page not read)"),
    "cera2": (1.00, 2.8e-3, 32.0, 0.080, (12.0, 3.0, 0.80), False, "two CeraCharge 1812"),
    "cera3": (1.50, 4.2e-3, 48.0, 0.120, (18.0, 4.5, 1.20), False, "three CeraCharge 1812 (concepts.md pick 1)"),
    "lipo301010": (133.0, 0.074, 300.0, 0.35, (2.0, 0.8, 0.40), False, "LiPo 301010 10 mAh 3.7 V (parts.json, A: 1-2C, 37-74 mW); none found in India (bench_kit); 301020 45 mAh is the nearest stocked"),
    "lipo401012": (200.0, 0.11, 480.0, 0.50, (2.0, 0.8, 0.40), False, "LiPo 401012 15 mAh (parts.json A, 1-2C)"),
    "lipo401020": (333.0, 0.19, 800.0, 0.80, (2.2, 0.9, 0.45), False, "LiPo 501015/401020 20-30 mAh, 25 mAh used (parts.json A)"),
    "lipo402020": (532.0, 0.30, 1600.0, 1.60, (2.5, 1.0, 0.50), False, "LiPo 402020 class, 40 mAh (A: catalogue class from memory, not checked)"),
    "lipo502020": (1066.0, 0.60, 2000.0, 2.20, (3.0, 1.2, 0.55), False, "LiPo 502020 class, 80 mAh (A: catalogue class from memory, not checked)"),
}
STORAGE_FIT = {"cera1": (4.4, 3.2), "cera2": (4.4, 3.2), "cera3": (4.4, 3.4), "lipo301010": (10.0, 10.0), "lipo401012": (10.0, 12.0), "lipo401020": (10.0, 20.0), "lipo402020": (20.0, 20.0), "lipo502020": (20.0, 20.0)}  # mm, the two largest dimensions
STORAGE_T = {"cera1": 1.1, "cera2": 2.2, "cera3": 3.3, "lipo301010": 3.0, "lipo401012": 4.0, "lipo401020": 4.0, "lipo402020": 4.0, "lipo502020": 5.0}  # mm, thickness (laid across x)


@dataclass(frozen=True)
class Design:
    """Design point (SI). Defaults are the starting point of the search, not the chosen point."""

    L: float = 10.8e-3
    # bonds
    corner_frac: float = 0.08  # corner sphere diameter / L (capture and hinge only; strength is the latch)
    cover: float = 0.08e-3  # resin cover over each sphere
    setback_frac: float = 0.10  # corner spheres set back from the disc faces (along x) / L
    ring_frac: float = 0.10  # disc ring sphere diameter / L
    ring_r_frac: float = 0.27  # ring radius on the disc (diagonals) / L
    k_insert: float = 0.35  # steel insert pull / sphere pair pull (A)
    disc_r_frac: float = 0.42
    disc_t_frac: float = 0.06
    tread: float = 0.10e-3  # rubber tread proud of the disc face (A)
    # legs
    leg_l_frac: float = 0.86  # leg length (from the hinge, across its face) / L
    leg_w_frac: float = 0.36  # leg width (along the hinge) / L, between the steel inserts
    leg_t: float = 0.35e-3
    tip_k: float = 0.25  # leg tip shim pull on a corner sphere / sphere pair pull (A)
    leg_wire: float = 0.038e-3
    leg_strands: int = 4
    leg_cap_r_frac: float = 0.14  # capstan ratchet ring radius / L (inside the cube)
    leg_ratio: float = 1.0  # capstan radius / knuckle radius: >1 gears the leg up (faster, weaker)
    leg_pitch: float = 0.20e-3
    # release carriers
    insert_recess: float = 0.15e-3  # steel insert depth below the face cover (a corner sphere seated on an insert must pull less than the bolt can retract)
    edge_round: float = 0.0  # 45 deg chamfer leg on all 12 edges (0 = sharp box); corner spheres move in to keep their cover
    leg_wire_len_m: float = 0.0  # active length of each leg wire (m); 0 = 0.80 x inner (straight run, the first model)
    leg_wire_eta: float = 1.0  # extra efficiency of the leg wire route (pulley or bellcrank friction), from the CAD routing
    rel_wire_len_m: float = 0.0  # active length of each bolt drive wire (m); 0 = 0.80 x inner
    rel_lever: float = 1.0  # bolt drive lever (wire stroke x lever = pawl stroke); force / lever
    rel_stroke: float = 2.40e-3  # edge bolt travel along x: 0.6 mm frees the hooks, the rest retracts the edge's corner spheres
    hook_free: float = 0.60e-3
    rel_wire: float = 0.076e-3
    rel_strands: int = 1
    rel_eta: float = 0.75
    leg_floor_frac: float = 0.5  # floor moves fire this share of each leg's strands (two wire groups, A)
    # joints
    joint_route: str = "sma"  # 'sma' (SmaStepper) or 'gear' (geared_joint.worm_last_design, L >= 11 mm)
    joint_strands: int = 18
    joint_phases: int = 1
    # latch (the bond's strength): edge bolts on the 4 x-parallel edges, a bayonet on each disc
    hook_w: float = 0.11  # hook tab width / L
    hook_t: float = 0.045  # hook tab thickness / L
    hook_l: float = 0.035  # hook overhang (lever) / L
    hooks_per_side: int = 2  # tabs of one bolt into one face's neighbour
    resin_sigma: float = 30e6  # Pa, peak bending allowable of printed tough resin (A, as geared_joint.RESIN)
    # power
    storage: str = "lipo301010"
    pulse_caps: int = 4  # 220 uF polymer tantalum 1206 (A)
    # magnet lot
    derate: float = 0.80

    # ------------------------------------------------------------------ geometry
    @property
    def h(self):
        return self.L / 2

    @property
    def inner(self):
        return self.L - 0.8e-3  # flex 0.11 mm on a 0.3 mm resin skin, both sides (E, as small-joint)

    @property
    def d_corner(self):
        return self.corner_frac * self.L

    @property
    def d_ring(self):
        return self.ring_frac * self.L

    @property
    def d0_corner(self):
        return self.d_corner + 2 * self.cover

    @property
    def d0_ring(self):
        return self.d_ring + 2 * self.cover

    @property
    def corner_a(self):
        """Corner sphere centre: |y|, |z| = a_yz (touching the plain-face covers), |x| = a_x (set back)."""
        a_yz = self.h - self.cover - self.d_corner / 2
        a_x = a_yz - self.setback_frac * self.L
        return a_x, a_yz

    @property
    def ring_r(self):
        return self.ring_r_frac * self.L

    @property
    def disc_r(self):
        return self.disc_r_frac * self.L

    # ------------------------------------------------------------------ bonds
    @property
    def f_corner(self):
        return pair_force(self.d_corner, 2 * self.cover + 2 * self.corner_shift, self.derate)

    @property
    def f_ring(self):
        return pair_force(self.d_ring, 2 * self.cover, self.derate)

    @property
    def f_insert(self):
        return insert_force(self.d_ring, 2 * self.cover + self.insert_recess - 0.15e-3, self.k_insert, self.derate)

    def hook_force(self) -> float:
        """Peak load of one hook tab: a resin cantilever, sigma b t^2 / (6 l) (E with A strength)."""
        b, t, l = self.hook_w * self.L, self.hook_t * self.L, self.hook_l * self.L
        return self.resin_sigma * b * t * t / (6 * l)

    def bonds(self) -> dict:
        """Strengths (E). Magnetic part (capture and hinge): plain faces 4 corner pairs; disc faces 4 ring
        spheres on steel inserts (on a plain face) or on ring spheres (disc on disc).
        Latched part: a plain-plain bond between cubes with parallel x axes is held by the two edge bolts of
        that face, hooks_per_side tabs each (tension: all tabs; peel about an x-parallel edge: the far bolt's
        tabs at lever L; about a y- or z-edge: one tab of each bolt at 0.8 L); a bond on a disc is held by
        the disc bayonet (4 tabs under a lip at 0.9 disc radius; peel: 2 tabs at 1.8 disc radius) and twist
        by two sprung steel pins. Machines use the latched values; swarm moves open the latch first and
        only the magnets act."""
        a_x, a_yz = self.corner_a
        fc, fr, fi = self.f_corner, self.f_ring, self.f_insert
        lever_c = 2 * min(a_x, a_yz) + (self.h - min(a_x, a_yz))
        mag_plain = {"tension_N": 4 * fc, "peel_Nm": 2 * fc * lever_c, "twist_Nm": 0.286 * 4 * fc * math.hypot(a_x, a_yz)}
        ring_lever = self.disc_r + self.ring_r / math.sqrt(2)
        pin = 2 * 40e6 * (0.03 * self.L) ** 2 * self.ring_r  # two pins 0.03 L across, 0.03 L deep, 40 MPa bearing (A)
        mag_disc_plain = {"tension_N": 4 * fi, "peel_Nm": 2 * fi * ring_lever, "twist_Nm": pin}
        mag_disc_disc = {"tension_N": 4 * fr, "peel_Nm": 2 * fr * ring_lever, "twist_Nm": pin}
        fh = self.hook_force()
        n = self.hooks_per_side
        lat_plain = {"tension_N": 2 * n * fh + 4 * fc, "peel_Nm": min(n * fh * self.L, 2 * fh * 0.8 * self.L) + mag_plain["peel_Nm"], "twist_Nm": 2 * n * fh * 0.3 * self.L}
        r_b = 0.9 * self.disc_r
        lat_disc = {"tension_N": 4 * fh + 4 * fi, "peel_Nm": 2 * fh * 2 * r_b + mag_disc_plain["peel_Nm"], "twist_Nm": pin}
        return {"plain": lat_plain, "disc": lat_disc, "mag_plain": mag_plain, "mag_disc_plain": mag_disc_plain, "mag_disc_disc": mag_disc_disc,
                "corner_pair_N": fc, "ring_pair_N": fr, "insert_N": fi, "hook_N": fh}

    def released_fraction(self, stroke: float | None = None) -> float:
        """Pull of a released corner pair over the closed pull (both faces next to the edge): the
        carrier moves each sphere stroke / sqrt(2) away from each face (E, d^-4)."""
        s = self.rel_stroke if stroke is None else stroke
        d0 = self.d0_corner
        return (d0 / (d0 + s / math.sqrt(2))) ** 4

    @property
    def tip_force(self):
        """Leg tip shim on a neighbour's corner sphere (E/A)."""
        return self.tip_k * pair_force(self.d_corner, 2 * self.cover + 0.05e-3, self.derate)

    # ------------------------------------------------------------------ joints
    def joint(self) -> SmaStepper:
        return SmaStepper(L=self.L, strands=self.joint_strands, phases=self.joint_phases)

    def gear_joint(self):
        from beebot.hardware.geared_joint import nominal_design
        # the joint the joint_gear14 machines run; the flex-PCB rotor's torque constant scales with
        # the rotor volume (E: same current density and fill), 4e-4 N m/A at 14 mm
        Lj = max(self.L, 11e-3)
        return nominal_design(Lj, k_t=4.0e-4 * (Lj / 14e-3) ** 3)

    def joint_summary(self) -> dict:
        """drive (cont) and peak hold in N m, speed in rad/s, energy per radian, wires, route."""
        if self.joint_route == "gear":
            g = self.gear_joint()
            t = g.table()
            return {"route": "gear", "drive_Nm": t["usable_cont_mNm"] * 1e-3, "hold_Nm": t["usable_peak_mNm"] * 1e-3, "speed_rad_s": t["no_load_speed_rad_s"] * 0.7,
                    "J_per_rad": 0.4 / (math.pi / 2 * 6), "self_locking": t["self_locking"], "fits": self.L >= 11e-3, "note": "geared_joint.nominal_design: flex-PCB flywheel motor, Didel worm and spur stages, printed split-ring output; J_per_rad from the coordinator's 0.4 J per arm move (A)"}
        js = self.joint()
        return {"route": "sma", "drive_Nm": js.drive, "hold_Nm": js.hold(True), "speed_rad_s": js.speed, "J_per_rad": js.step_energy / js.step, "self_locking": True, "fits": True, "note": "geared_joint.SmaStepper"}

    # ------------------------------------------------------------------ legs
    @property
    def leg_len(self):
        return self.leg_l_frac * self.L

    @property
    def leg_wire_len(self):
        return self.leg_wire_len_m if self.leg_wire_len_m > 0 else 0.80 * self.inner

    def leg_actuator(self, force_scale=1.0, cool_scale=1.0, eta=0.75) -> Ratchet:
        """Capstan ratchet: ring radius r_c, pawl stroke 1.25 teeth from the wire's 3 % strain through
        a lever; the capstan winds a tendon on the leg knuckle; torque at the leg = capstan torque /
        leg_ratio, step at the leg = tooth step x leg_ratio (E)."""
        r_c = self.leg_cap_r_frac * self.L
        stroke_len = 0.03 * self.leg_wire_len
        lever = 1.25 * self.leg_pitch / stroke_len
        pull = wire_pull(self.leg_wire, self.leg_strands) * force_scale
        cap_torque = eta * self.leg_wire_eta * pull / lever * r_c
        drive = cap_torque / self.leg_ratio
        # lock: 4 teeth, face width 0.9 mm at 14 mm and up with L above (E: the capstan layer grows with
        # the cube), tooth height 0.4 p, root 0.5 p, resin 30 MPa peak (as SmaStepper.hold)
        b, p = 0.9e-3 * max(1.0, self.L / 14e-3), self.leg_pitch
        hold = 4 * 30e6 * b * (0.5 * p) ** 2 / (6 * 0.4 * p) * r_c / self.leg_ratio  # 4 teeth engaged: with 3, 2 of 16 falls from 20 cm loaded a closed leg's lock past the resin's strength
        step = self.leg_pitch / r_c * self.leg_ratio
        cool = FLEXINOL[self.leg_wire][4] * 1.3 * cool_scale
        stroke = 0.12 if self.leg_wire <= 0.038e-3 else 0.16
        return Ratchet(step=step, stroke=stroke, period=stroke + cool, drive=drive, hold=hold,
                       stiffness=2e5 * r_c**2 / self.leg_ratio**2, clearance=10e-6 / r_c * self.leg_ratio,
                       step_energy=heat_energy(self.leg_wire, self.leg_wire_len, self.leg_strands))

    # ------------------------------------------------------------------ release carriers
    def release_actuator(self, force_scale=1.0, cool_scale=1.0) -> Ratchet:
        """Rack ratchet: the drive wire (rel_strands x rel_wire) pushes a pawl one rack tooth (0.2 mm) a
        stroke with no lever, so its whole pull (x rel_eta) acts on the carrier; a lock pawl holds it with
        no power. One shared unlock wire (25 um) lifts all four locks and the magnets with a light spring
        pull the carriers back out (E/A)."""
        length = self.rel_wire_len_m if self.rel_wire_len_m > 0 else 0.80 * self.inner
        # the rack pawl needs one 0.2 mm tooth plus margin a stroke: 3 % of the wire times the lever
        if 0.03 * length * self.rel_lever < 0.25e-3:
            drive = -1.0  # wire too short to step the rack: the bolt cannot move
        else:
            drive = self.rel_eta * wire_pull(self.rel_wire, self.rel_strands) * force_scale / self.rel_lever - 0.02  # spring preload (A); load: bolt friction and any load on the hooks
        cool = FLEXINOL[self.rel_wire][4] * 1.3 * cool_scale
        stroke = 0.12 if self.rel_wire <= 0.038e-3 else (0.16 if self.rel_wire <= 0.05e-3 else 0.22)
        return Ratchet(step=0.20e-3, stroke=stroke, period=stroke + cool, drive=drive, hold=np.inf,
                       stiffness=0.0, clearance=0.0, step_energy=heat_energy(self.rel_wire, length, self.rel_strands))

    @property
    def rel_steps(self) -> int:
        return int(math.ceil(self.rel_stroke / 0.20e-3 - 1e-9))

    def release_energy(self) -> float:
        """J for one release (all rack steps) plus its later unlock."""
        return self.rel_steps * self.release_actuator().step_energy + heat_energy(0.025e-3, 0.8 * self.inner, 1)

    # ------------------------------------------------------------------ parts, mass, packing
    def parts(self) -> dict:
        """Part volumes (mm3, inside the skin unless noted) and masses (g), E/A."""
        L = self.L * 1e3
        mm = lambda v: v * 1e9  # m3 -> mm3
        out = {}

        def add(name, vol_mm3, rho, inside=True, mass_g=None):
            out[name] = {"vol_mm3": vol_mm3, "mass_g": vol_mm3 * 1e-9 * rho * 1e3 if mass_g is None else mass_g, "inside": inside}

        add("skin: flex PCB 0.11 mm on 0.3 mm resin, six faces", 6 * L * L * 0.11, RHO["flex"], inside=False)
        add("skin resin", 6 * L * L * 0.30 * 0.55, RHO["resin"], inside=False)  # skin is a frame with windows for legs, discs, spheres (A: 55 % solid)
        add("frame: hinge knuckle bars along 8 leg edges, 8 corner pocket blocks", 8 * L * 0.16 + 8 * (self.d_corner * 1e3 + 0.6) ** 3 * 0.4, RHO["resin"])
        add("corner spheres x8", 8 * math.pi / 6 * (self.d_corner * 1e3) ** 3, RHO["ndfeb"])
        add("disc ring spheres x8", 8 * math.pi / 6 * (self.d_ring * 1e3) ** 3, RHO["ndfeb"])
        add("steel inserts x16 (0.8 x 0.3 mm)", 16 * math.pi * 0.4**2 * 0.3, RHO["steel"])
        st = self.joint()
        add("discs x2 (resin, tread)", 2 * math.pi * (self.disc_r * 1e3) ** 2 * self.disc_t_frac * L, RHO["resin"], inside=False)
        if self.joint_route == "gear":
            joint_vol = 70.0 * (L / 14.0) ** 2 + 25.0  # Didel worm GW205, G209/G220/G240 stages, printed worm wheel, flywheel rotor and stator flex (A; small-joint.md: the train fits from 11-14 mm)
            add("twist joints x2: flex motor rotor, Didel module 0.2 train, printed worm stage, selector coil", 2 * joint_vol, 1400.0)
        else:
            joint_vol = 41.0 * (L / 8.25) ** 3 * (0.6 + 0.4 * self.joint_strands / 18)  # small-joint.md: 82 mm3 for both at 8.25 mm (E)
            add("twist joints x2: ratchet ring, hub, pawls, lock, wires, crimps", 2 * joint_vol, 1500.0)
        leg_plate = self.leg_len * 1e3 * self.leg_w_frac * L * self.leg_t * 1e3
        r_c = self.leg_cap_r_frac * L
        cap = math.pi * r_c**2 * 0.9 + 3.0 + 0.4 * self.leg_strands * (self.leg_wire / 0.025e-3) ** 2
        add("legs x4: plates", 4 * leg_plate, RHO["resin"])
        add("legs x4: capstan ratchets, tendons, pawls, wires, crimps", 4 * cap, 1400.0)
        add("edge bolts x4: hooked bars, racks, pawls, wires, crimps; shared unlock wire", 4 * ((L - 2 * self.setback_frac * L) * (0.06 * L) ** 2 * 1.6 + 2.5 + 0.2 * self.rel_strands * (self.rel_wire / 0.025e-3) ** 2) + 1.5, 1300.0)
        add("disc bayonet tabs, lips and pins (both discs, every face's groove)", 2 * 4 * (self.hook_w * L) * (self.hook_t * L) * (0.1 * L) + 6 * 0.5, RHO["resin"])
        e, p, vol, mass, *_ = STORAGE[self.storage]
        add(f"storage: {self.storage}", vol, 1.0, mass_g=mass)
        add(f"pulse caps x{self.pulse_caps} (220 uF 1206 polymer tantalum)", self.pulse_caps * 3.2 * 1.6 * 1.1, RHO["tant"])
        nsw = self.wires()["wires"]
        add(f"electronics: CH32V003 QFN-20, {nsw} SOT-523 FETs, LIS2DW12, charger, 6 IR pairs, 16 corner diodes, passives", 7.2 + nsw * 0.9 + 2.8 + 3.0 + 6 * 0.6 + 16 * 0.06 + 6.0, RHO["silicon"])
        return out

    @property
    def mass(self) -> float:
        return sum(p["mass_g"] for p in self.parts().values()) * 1e-3

    @property
    def corner_shift(self) -> float:
        """Inward shift (each of the two in-face coordinates) of a corner sphere so it keeps its cover under
        a 45 deg chamfer of leg edge_round: c / 2 - (1 - 1 / sqrt 2)(cover + d / 2), at least 0."""
        return max(0.0, self.edge_round / 2 - (1 - 1 / math.sqrt(2)) * (self.cover + self.d_corner / 2))

    @property
    def floor_frac(self) -> float:
        """Share of a leg's strands fired on the floor, in whole strands (one strand: all of it)."""
        return math.ceil(self.leg_floor_frac * self.leg_strands - 1e-9) / self.leg_strands

    def packing(self) -> float:
        """inner^3 x 0.8 / parts inside (rule of small-joint.md: >= 1.5 for hand assembly; the storage's
        own volume has no slack, so it is counted at 1.0)."""
        parts = self.parts()
        vol = sum(p["vol_mm3"] for k, p in parts.items() if p["inside"])
        inner = (self.inner * 1e3) ** 3
        return 0.8 * inner / vol

    def layout(self) -> dict:
        """Geometric fit of the parts the CadQuery layout (scripts/module_phys/sma_cube_cad.py) found
        binding, fitted to its sweep at 12.18, 14, 16 and 18 mm (E): (1) each joint stack's depth along
        x beside the cell, which lies across x in the middle; (2) one hook tab per bolt face: hook travel
        plus retraction plus a 1.2 mm tab inside the edge region; (3) the cell across the cavity.
        The analytic packing ratio (packing()) passed points the CAD could not assemble, so this check
        is the binding one."""
        Lmm = self.L * 1e3
        t_b = STORAGE_T[self.storage]
        free_x = (Lmm - 0.1 - t_b) / 2
        if self.joint_route == "gear":
            need_x = 5.15 + 0.10 * Lmm  # CAD: 6.37 at 12.18, 6.75 at 16, 6.95 at 18
        else:
            need_x = 0.25 * Lmm + 1.0  # SMA ratchet ring and wire anchors (A: no CAD of the scaled joint)
        region = 0.675 * Lmm - 1.4  # CAD: 9.39 at 16, 10.74 at 18
        slot = (self.hook_free + self.rel_stroke) * 1e3 + 1.2
        out = {"joint_x_free_mm": round(free_x, 2), "joint_x_need_mm": round(need_x, 2), "hook_region_mm": round(region, 2), "hook_slot_need_mm": round(slot, 2), "storage_fits": self.storage_fits()}
        out["fits"] = bool(free_x >= need_x and region >= slot and out["storage_fits"])
        return out

    def storage_fits(self) -> bool:
        a, b = STORAGE_FIT[self.storage]
        return self.inner * 1e3 >= max(a, b) + 0.2 - 1e-9  # the cell's longest side across the cavity, 0.1 mm each side (E)

    # ------------------------------------------------------------------ power
    def power(self) -> dict:
        """Energy and rate budget for lone floor moves (E). A tip opens a leg about 100 deg (book) or 215
        deg (kick) and closes it again; floor moves fire leg_floor_frac of the leg's strands. Pulses come
        from the tantalum bank, recharged from the storage; the storage's continuous rating limits the
        step rate. Structure moves run on bus power through the bonds and are not counted here."""
        e_store, p_cont, *_ = STORAGE[self.storage]
        leg = self.leg_actuator()
        step_deg = math.degrees(leg.step)
        book_steps = 2 * math.ceil(100 / step_deg)
        kick_steps = 2 * math.ceil(215 / step_deg)
        e_step = leg.step_energy * self.floor_frac
        e_tip = 0.5 * (book_steps + kick_steps) * e_step
        bank = self.pulse_caps * 0.5 * 220e-6 * (4.2**2 - 2.5**2)  # J usable from 4.2 V down to 2.5 V
        p_idle = 0.6e-3  # MCU at a few MHz, IMU, IR beacons at low duty (A)
        p_avail = max(p_cont - p_idle, 1e-6)
        t_step_power = e_step / p_avail
        t_step = max(leg.period, t_step_power)
        t_tip = 0.5 * (book_steps + kick_steps) * t_step
        tips = 0.85 * e_store / (e_tip + p_idle * t_tip)  # 85 % of the cell used (A)
        jsum = self.joint_summary()
        return {
            "storage": self.storage,
            "storage_J": e_store,
            "continuous_W": p_cont,
            "pulse_bank_mJ": round(bank * 1e3, 2),
            "leg_floor_step_mJ": round(e_step * 1e3, 2),
            "leg_pulse_fits_bank": bool(bank >= e_step),
            "joint_J_per_quarter_turn": round(jsum["J_per_rad"] * math.pi / 2, 3),
            "tip_mJ": round(e_tip * 1e3, 1),
            "tip_s": round(t_tip, 1),
            "lone_tips_per_charge": int(tips),
            "lone_speed_mm_s": round(self.L * 1e3 / t_tip, 3),
            "lone_run_min": round(tips * t_tip / 60, 1),
            "unlatch_mJ": round(self.release_energy() * 1e3, 1),
            "storage_limited": bool(t_step_power > leg.period),
        }

    # ------------------------------------------------------------------ cost
    def bom(self) -> list:
        """(item, qty, unit USD at 10, 1k, 1M, verified, source). Prices from runs/module_phys/bench_kit.md
        (verified = read on the vendor page) and parts.json (mostly assumed)."""
        e, p, vol, mass, price, ver, note = STORAGE[self.storage]
        wire_m = lambda d, strands, n: n * strands * 0.80 * self.inner + n * 0.004  # m of wire with crimp tails
        flex_price = {25e-6: (14.0, 7.0, 1.5), 38e-6: (2.2, 1.5, 0.5), 50e-6: (10.0, 6.0, 1.0), 76e-6: (8.2, 6.0, 1.0), 100e-6: (8.2, 6.0, 1.0)}

        def fp(d):
            return flex_price[min(flex_price, key=lambda k: abs(k - d))]

        w = self.wires()
        nfet = w["wires"] + (6 if self.joint_route == "gear" else 0)
        lw = wire_m(self.leg_wire, self.leg_strands, 8)
        rw = wire_m(self.rel_wire, self.rel_strands, 4) + wire_m(0.025e-3, 1, 1)
        rows = [
            ("CH32V003F4U6 MCU", 1, 0.308, 0.22, 0.08, True, "LCSC C5299908, 5-tier 0.3079, 150-tier 0.2220 (bench_kit V); 1M A"),
            ("DMG1012T-7 N-FET (wire and coil switches)", nfet, 0.0905, 0.0295, 0.008, True, "LCSC C20512 0.0905 at 20, clone C2830301 0.0295 (bench_kit V); 1M A"),
            ("LIS2DW12 accelerometer", 1, 0.912, 0.79, 0.35, True, "LCSC C189624 (bench_kit V at 1 and 10); 1M A"),
            ("IR LED 0402-class", 6, 0.281, 0.21, 0.02, True, "LCSC C5382142 (bench_kit V at 10 and 100); 1M A"),
            ("IR photodiode (0402 class; bench part is 2x3 mm)", 6, 0.101, 0.08, 0.02, False, "bench_kit: no 0402 photodiode at LCSC; LED as detector is the fallback (A)"),
            ("Schottky diodes 0201 (corner power)", 16, 0.02, 0.01, 0.002, False, "A"),
            ("charger + LDO (TP4054 class) and passives", 1, 0.30, 0.12, 0.04, False, "parts.json A"),
            (f"battery: {self.storage}", 1, price[0], price[1], price[2], ver, note),
            ("polymer tantalum 220 uF 1206", self.pulse_caps, 0.25, 0.10, 0.04, False, "A"),
            ("JLCPCB 2-layer flex, unfolded net, ENIG, stiffeners", 1, 2.5, 0.40, 0.08, False, "parts.json A (about 1 USD at 10 from memory; ENIG and stiffeners added)"),
            ("JLCPCB PCBA setup and placement share", 1, 4.0, 0.60, 0.10, False, "parts.json A (setup about 8 USD per order, extended-part fees)"),
            ("N52 Ni spheres (8 corner + 8 ring, capture and hinge)", 16, 0.15, 0.06, 0.01, False, "bench_kit A: 15 USD per 100, no vendor page found for 1.0-1.5 mm spheres"),
            (f"Flexinol {self.leg_wire * 1e6:.0f} um (legs), m", lw, *fp(self.leg_wire), self.leg_wire == 0.038e-3, "38 um MowerBot Pro 11 USD per 5 m (bench_kit V); others quote/A"),
            (f"Flexinol {self.rel_wire * 1e6:.0f} um (edge bolts) and 25 um unlock, m", rw, *fp(self.rel_wire), self.rel_wire == 0.076e-3, "76 um Images SI 8.2 USD/m (bench_kit V)"),
            ("brass tube crimps", w["crimps"], 0.03, 0.01, 0.002, False, "bench_kit A: cut from 0.5/0.3 mm tube"),
            ("steel inserts, pins, springs, Dyneema tendon", 1, 0.40, 0.10, 0.02, False, "A"),
            ("resin parts (frame, discs, legs, ratchets, bolts), printed; moulded at 1M", 1, 0.0001 * (self.L * 1e3) ** 3 * 0.25, 0.00007 * (self.L * 1e3) ** 3 * 0.25, 0.03 * (self.L / 14e-3) ** 2, False, "Siraya Fast ABS-like 7,800 INR/kg (bench_kit tracker, A); moulded at 1M (A)"),
            ("silicone tread rings", 2, 0.05, 0.02, 0.005, False, "A"),
        ]
        if self.joint_route == "gear":
            rows.append(("Didel module 0.2 gears and worm (G209, G220, G240, GW205), two joints", 2, 14.7, 6.0, 0.15, True, "didel.com Microgears.pdf 2008 list (small-joint.md V, CHF ~ USD): 2.20 + 4.00 + 6.50 + ... per set at 10; 1k and 1M A (moulded POM)"))
            rows.append(("rotor magnets (diametric 4 x 1 mm) and flywheel rims, two joints", 2, 0.4, 0.15, 0.03, False, "parts.json A"))
        else:
            jw = wire_m(0.025e-3, self.joint_strands, 2 * 2 * self.joint_phases)
            rows.append(("Flexinol 25 um (joints), m", jw, *fp(0.025e-3), False, "Dynalloy price guide 14-7 USD/m (bench_kit, quote); 1M A"))
        return rows

    def cost(self) -> dict:
        rows = self.bom()
        out = {}
        for k, tier in enumerate(("10", "1k", "1M")):
            out[tier] = round(sum(row[1] * row[2 + k] for row in rows), 2)
        out["assembly"] = {"10": "own time, about 3-5 h a cube (A)", "1k": "contract assembly 8-15 USD a cube (A)", "1M": "automated, 1-2 USD a cube after tooling (A)"}
        out["verified_share_10"] = round(sum(r[1] * r[2] for r in rows if r[5]) / max(out["10"], 1e-9), 2)
        return out

    def wires(self) -> dict:
        """Wire loops (each crimped at both ends): SMA joints 2 directions x phases each (none for the
        geared route), legs 2 groups a direction (floor and full drive), edge bolts 4 plus one shared unlock."""
        n_joint = 2 * 2 * self.joint_phases if self.joint_route == "sma" else 0
        n_leg = 4 * 2 * (2 if self.floor_frac < 1 else 1)
        n_rel = 4 + 1
        return {"wires": n_joint + n_leg + n_rel, "crimps": 2 * (n_joint + n_leg + n_rel), "actuators": 2 + 4 + 4, "joint": n_joint, "leg": n_leg, "release": n_rel}

    # ------------------------------------------------------------------ summary
    def table(self) -> dict:
        b = self.bonds()
        leg = self.leg_actuator()
        rel = self.release_actuator()
        js = self.joint_summary()
        m = self.mass
        mgl = m * G * self.L
        return {
            "L_mm": round(self.L * 1e3, 3),
            "joint_route": self.joint_route,
            "mass_g": round(m * 1e3, 3),
            "packing": round(self.packing(), 3),
            "storage_fits": self.storage_fits(),
            "layout": self.layout(),
            "corner_sphere_mm": round(self.d_corner * 1e3, 3),
            "ring_sphere_mm": round(self.d_ring * 1e3, 3),
            "hook_N": round(b["hook_N"], 3),
            "latched_plain": {k: round(v, 4) for k, v in b["plain"].items()},
            "latched_disc": {k: round(v, 4) for k, v in b["disc"].items()},
            "magnet_plain": {k: round(v, 4) for k, v in b["mag_plain"].items()},
            "magnet_plain_in_weights": round(b["mag_plain"]["tension_N"] / (m * G), 1),
            "magnet_far_pair_peel_over_leg": round(2 * b["corner_pair_N"] * self.L / leg.drive, 3),
            "tip_anchor_in_weights": round(self.tip_force / (m * G), 2),
            "leg": {"step_deg": round(math.degrees(leg.step), 2), "drive_mNm": round(leg.drive * 1e3, 4), "drive_over_mgL": round(leg.drive / mgl, 2), "hold_mNm": round(leg.hold * 1e3, 3), "period_s": round(leg.period, 3), "step_mJ": round(leg.step_energy * 1e3, 2)},
            "bolt": {"drive_N": round(rel.drive, 4), "stroke_mm": round(self.rel_stroke * 1e3, 3), "steps": self.rel_steps, "mJ": round(self.release_energy() * 1e3, 1)},
            "joint": {"drive_mNm": round(js["drive_Nm"] * 1e3, 3), "hold_mNm": round(js["hold_Nm"] * 1e3, 3), "speed_deg_s": round(math.degrees(js["speed_rad_s"]), 1), "J_per_quarter": round(js["J_per_rad"] * math.pi / 2, 3), "drive_over_mgL": round(js["drive_Nm"] / mgl, 1), "fits": js["fits"]},
            "power": self.power(),
            "cost_usd": self.cost(),
            "wires": self.wires(),
        }


def joint_mass_fit(L):
    """Reference: small-joint.md's mass fit, for comparison only."""
    return 978.9957909924643 * L**3 + 10.918360804960212 * L**2


# ---------------------------------------------------------------------------------------------
# Body-frame layout of legs and release carriers
# ---------------------------------------------------------------------------------------------

AXES = {"+x": np.array([1.0, 0, 0]), "-x": np.array([-1.0, 0, 0]), "+y": np.array([0, 1.0, 0]), "-y": np.array([0, -1.0, 0]), "+z": np.array([0, 0, 1.0]), "-z": np.array([0, 0, -1.0])}

# leg name -> (face it lies in, the other face of its hinge edge)
LEGS = {"XP": ("+y", "+z"), "XN": ("+z", "+y"), "DP": ("-y", "+x"), "DN": ("-z", "-x")}
# tip pad: 0.1 mm proud of the leg plate over the last 0.8 mm (the plate sits 0.1 mm deeper in its recess,
# so a folded leg stays flush). Added 2026-11-16 with the braked descent: a leg lying flat on a
# neighbour's face otherwise bears near its knuckle, and the small lever multiplies the push that
# opens the magnetic hinge.
TIP_PAD = 0.1e-3
TIP_PAD_LEN = 0.8e-3
# the x-legs are a pair on one edge: each one's partner lies in the other face of the same hinge edge
LEG_PARTNER = {"XP": "XN", "XN": "XP"}
# release carrier -> (sign y, sign z) of its x-parallel edge
CARRIERS = {"pp": (1, 1), "pn": (1, -1), "np": (-1, 1), "nn": (-1, -1)}


def leg_frame(name):
    """(lie normal, other normal, d0 = direction from the hinge into the face, hinge axis a): positive
    rotation about a opens the leg out of its face."""
    lie, other = (AXES[f] for f in LEGS[name])
    d0 = -other
    a = np.cross(lie, other)
    return lie, other, d0, a


def carrier_of_corner(sy, sz):
    return next(k for k, v in CARRIERS.items() if v == (sy, sz))


def rot_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


# ---------------------------------------------------------------------------------------------
# Dipole magnetics (numba)
# ---------------------------------------------------------------------------------------------

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        def wrap(f):
            return f
        return wrap if not a or not callable(a[0]) else a[0]


@njit(cache=True)
def _magnet_kernel(P, U, mag, cube, steel, rcut2, b_sat, it):
    """Fields, alignment (free spheres and steel turn to the local field; steel moments grow with the
    field up to b_sat), forces. P positions, U unit moment directions (updated in place), mag moment
    magnitudes (sphere) or saturation moments (steel). Pairs inside one cube and steel-steel pairs are
    skipped. Returns forces (n, 3) and the effective moments' magnitudes."""
    n = P.shape[0]
    k = 1e-7  # mu0 / 4 pi
    meff = mag.copy()
    F = np.zeros((n, 3))
    for _ in range(it):
        for i in range(n):  # Gauss-Seidel: each magnet turns to the field of the others' current moments
            bx = 0.0
            by = 0.0
            bz = 0.0
            for j in range(n):
                if j == i or cube[i] == cube[j] or (steel[i] and steel[j]):
                    continue
                rx = P[i, 0] - P[j, 0]
                ry = P[i, 1] - P[j, 1]
                rz = P[i, 2] - P[j, 2]
                d2 = rx * rx + ry * ry + rz * rz
                if d2 > rcut2 or d2 < 1e-14:
                    continue
                d = math.sqrt(d2)
                ux, uy, uz = rx / d, ry / d, rz / d
                c = k / (d2 * d)
                mjx, mjy, mjz = meff[j] * U[j, 0], meff[j] * U[j, 1], meff[j] * U[j, 2]
                mdot = mjx * ux + mjy * uy + mjz * uz
                bx += c * (3 * mdot * ux - mjx)
                by += c * (3 * mdot * uy - mjy)
                bz += c * (3 * mdot * uz - mjz)
            b = math.sqrt(bx * bx + by * by + bz * bz)
            if b > 1e-6:
                U[i, 0], U[i, 1], U[i, 2] = bx / b, by / b, bz / b
            if steel[i]:
                meff[i] = mag[i] * min(1.0, b / b_sat)
    for i in range(n):
        for j in range(i + 1, n):
            if cube[i] == cube[j] or (steel[i] and steel[j]):
                continue
            rx = P[j, 0] - P[i, 0]
            ry = P[j, 1] - P[i, 1]
            rz = P[j, 2] - P[i, 2]
            d2 = rx * rx + ry * ry + rz * rz
            if d2 > rcut2 or d2 < 1e-14:
                continue
            d = math.sqrt(d2)
            ux, uy, uz = rx / d, ry / d, rz / d
            mi = meff[i]
            mj = meff[j]
            ai = U[i, 0] * ux + U[i, 1] * uy + U[i, 2] * uz
            aj = U[j, 0] * ux + U[j, 1] * uy + U[j, 2] * uz
            ab = U[i, 0] * U[j, 0] + U[i, 1] * U[j, 1] + U[i, 2] * U[j, 2]
            c = 3 * k * mi * mj / (d2 * d2)
            fx = c * (ai * U[j, 0] + aj * U[i, 0] + ab * ux - 5 * ai * aj * ux)
            fy = c * (ai * U[j, 1] + aj * U[i, 1] + ab * uy - 5 * ai * aj * uy)
            fz = c * (ai * U[j, 2] + aj * U[i, 2] + ab * uz - 5 * ai * aj * uz)
            F[j, 0] += fx
            F[j, 1] += fy
            F[j, 2] += fz
            F[i, 0] -= fx
            F[i, 1] -= fy
            F[i, 2] -= fz
    return F, meff


def magnet_forces(P, U, mag, cube, steel, rcut, b_sat, it=2):
    return _magnet_kernel(np.ascontiguousarray(P), U, np.ascontiguousarray(mag), np.ascontiguousarray(cube), np.ascontiguousarray(steel), rcut * rcut, b_sat, it)


# ---------------------------------------------------------------------------------------------
# Ratchet steppers on MuJoCo hinges (vectorised)
# ---------------------------------------------------------------------------------------------


class StepperBank:
    """Ratchet steppers on hinge joints. Per actuator: the lock angle phi moves in whole teeth; a stroke
    starts when the target is more than 0.55 teeth away and the wire for that direction is ready; during
    the stroke phi follows a minimum-jerk path but cannot get further ahead of the joint than the wire's
    drive reaches through the lock stiffness (a stalled stroke creeps with the load); at 1.5 strokes, if
    phi did not reach the next tooth, the stroke fails and phi falls back. Torque on the joint is the
    lock spring k (phi - q) (dead zone of the clearance) plus damping; a torque above the lock's hold
    breaks the tooth (recorded; the joint then slips at the hold torque)."""

    def __init__(self, ratchets, q0, damping):
        self.n = n = len(ratchets)
        f = lambda a: np.array([getattr(r, a) for r in ratchets], float)
        self.step, self.stroke, self.period = f("step"), f("stroke"), f("period")
        self.drive, self.hold, self.k, self.clear = f("drive"), f("hold"), f("stiffness"), f("clearance")
        self.e_step = f("step_energy")
        self.c = np.asarray(damping, float)
        self.drive_frac = np.ones(n)  # share of the strands fired (floor moves)
        self.phi = np.array(q0, float)
        self.target = self.phi.copy()
        self.active = np.zeros(n, bool)
        self.back = np.zeros(n, bool)
        self.dir = np.zeros(n, int)
        self.t0 = np.zeros(n)
        self.frm = np.zeros(n)
        self.ready = np.zeros((n, 2))
        self.fails = np.zeros(n, int)
        self.fail_t = np.full(n, -1e9)
        self.free = np.zeros(n, bool)
        self.broken = np.zeros(n, bool)
        self.energy = 0.0
        self.steps = np.zeros(n, int)
        self.failed = np.zeros(n, int)
        self.peak = np.zeros(n)

    def busy(self, a=None):
        moving = self.active | (np.abs(self.target - self.phi) > 0.55 * self.step)
        return bool(moving.any()) if a is None else bool(np.any(moving[a]))

    def torque(self, t, q, qd, dt):
        st = self
        half = st.clear / 2
        vmax = 2.0 * st.step / st.stroke
        # failed strokes fall back
        back = st.active & st.back
        if back.any():
            mv = np.clip(st.frm - st.phi, -vmax * dt, vmax * dt)
            st.phi = np.where(back, st.phi + mv, st.phi)
            home = back & (np.abs(st.phi - st.frm) < 1e-9)
            st.active[home] = False
            st.back[home] = False
        stroke = st.active & ~st.back
        if stroke.any():
            u = np.clip((t - st.t0) / st.stroke, 0.0, 1.0)
            end = st.frm + st.dir * st.step
            path = st.frm + st.dir * st.step * (10 * u**3 - 15 * u**4 + 6 * u**5)
            drive = st.drive * st.drive_frac
            reach = q + st.dir * (half + drive / np.maximum(st.k, 1e-12))
            mv = np.clip(path - st.phi, -vmax * dt, vmax * dt)
            mv = np.where(st.dir > 0, np.minimum(mv, np.maximum(reach - st.phi, 0.0)), np.maximum(mv, np.minimum(reach - st.phi, 0.0)))
            st.phi = np.where(stroke, st.phi + mv, st.phi)
            done = stroke & (u >= 1.0) & (np.abs(st.phi - end) < 1e-6 * st.step)
            st.phi = np.where(done, end, st.phi)
            st.active[done] = False
            st.fails[done] = 0
            late = stroke & ~done & (t - st.t0 >= 1.5 * st.stroke)
            if late.any():
                st.back[late] = True
                st.fails[late] += 1
                st.fail_t[late] = t
                st.failed[late] += 1
        idle = ~st.active & ~st.free
        e = st.target - st.phi
        want = np.where(e > 0, 1, -1)
        col = np.where(want > 0, 0, 1)
        ready = t >= st.ready[np.arange(st.n), col]
        resting = (st.fails >= 2) & (t - st.fail_t < 1.0)
        go = idle & (np.abs(e) > 0.55 * st.step) & ready & ~resting
        if go.any():
            st.active[go] = True
            st.dir[go] = want[go]
            st.t0[go] = t
            st.frm[go] = st.phi[go]
            idx = np.where(go)[0]
            st.ready[idx, col[go]] = t + st.period[go]
            st.energy += float(np.sum(st.e_step[go] * st.drive_frac[go]))
            st.steps[go] += 1
        x = st.phi - q
        xb = np.where(x > half, x - half, np.where(x < -half, x + half, 0.0))
        tau = st.k * xb - st.c * qd
        over = np.abs(tau) > st.hold
        if over.any():
            st.broken |= over & ~st.free
            tau = np.clip(tau, -st.hold, st.hold)
        tau = np.where(st.free, -st.c * qd, tau)
        st.peak = np.maximum(st.peak, np.abs(tau))
        return tau

    def go_to(self, a, angle):
        """Target an angle, rounded to the nearest tooth from the current lock position."""
        n = np.round((angle - self.phi[a]) / self.step[a])
        self.target[a] = self.phi[a] + n * self.step[a]
        self.fails[a] = 0


# ---------------------------------------------------------------------------------------------
# MuJoCo world
# ---------------------------------------------------------------------------------------------

@dataclass
class Params:
    """Monte Carlo variables of one trial (nominal values by default)."""

    derate: float = 0.80  # magnet lot: published / model pull (0.70-0.90)
    moment_spread: float = 0.0  # per-magnet relative moment spread (uniform +-)
    k_insert: float | None = None  # steel insert factor (None: design value)
    tip_k: float | None = None
    floor_mu: float = 0.5
    cube_mu: float = 0.4
    tread_mu: float = 0.8
    damp: float = 1.0  # contact damping ratio (restitution)
    pocket: float = 0.0  # sphere pocket position error, per axis (uniform +-, m)
    size: float = 0.0  # cube size error (m)
    leg_len: float = 0.0  # leg length error (m)
    hinge: float = 0.0  # leg hinge position error, per axis (m)
    wire: float = 1.0  # wire pull scale (0.8-1.2)
    cool: float = 1.0  # cooling time scale (0.8-2)
    eta: float = 0.75  # pawl efficiency (0.55-0.90)
    seed: int = 0


class World:
    """Free cubes on a floor. cubes: list of (centre (3,), rotation (3, 3)) in world coordinates.
    Per cube: box body; disc bodies p/n on +-x hinges (joints jp, jn); leg bodies on hinges at the
    lattice edge lines; magnets (corner spheres on release carriers, ring spheres on the discs, steel
    inserts on the plain faces, leg tip shims) applied through xfrc_applied."""

    def __init__(self, design: Design, cubes, params: Params | None = None, dt=None, mag_every=None, scene="", fixed=(), legs_open=None, floor=True, gravity=True):
        self.d = design
        self.p = params or Params()
        dt = float(os.environ.get("SMA_DT", "1e-4")) if dt is None else dt  # 2e-4 checked against 1e-4 in the memo
        mag_every = int(os.environ.get("SMA_MAG_EVERY", "2")) if mag_every is None else mag_every
        self.dt = dt
        self.mag_every = mag_every
        self.rng = np.random.default_rng(self.p.seed)
        self.n = len(cubes)
        self.fixed = set(fixed)
        self.xml = self._mjcf(cubes, scene, floor)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        if not gravity:
            self.model.opt.gravity[:] = 0
        m = self.model
        self.body = [m.body(f"c{i}").id for i in range(self.n)]
        # actuators: per cube jp, jn, then legs
        self.act_names = [m.actuator(a).name for a in range(m.nu)]
        self.aidx = {nm: a for a, nm in enumerate(self.act_names)}
        self.qadr = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self.vadr = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        if legs_open:
            for (i, leg), ang in legs_open.items():
                self.data.qpos[self.qadr[self.aidx[f"c{i}{leg}"]]] = ang
        mujoco.mj_forward(m, self.data)
        ratchets, damp = [], []
        js = design.joint()
        from beebot.hardware.geared_joint import stepper_params
        jp = stepper_params(js, None, force_scale=self.p.wire, cool_scale=self.p.cool, eta_pawl=self.p.eta)
        jr = Ratchet(step=jp.step, stroke=jp.stroke_time, period=jp.step_period, drive=jp.drive, hold=jp.tooth_peak, stiffness=0.3, clearance=jp.backlash, step_energy=jp.step_energy)
        if design.joint_route == "gear":
            # geared joint (geared_joint.nominal_design) as a servo moving in 1 deg increments at its
            # loaded speed, torque-limited at its usable drive, self-locking (holds to its usable peak);
            # output backlash 0.5 deg (A); energy from the joint summary's J per rad
            g = design.joint_summary()
            st = math.radians(1.0)
            jr = Ratchet(step=st, stroke=st / g["speed_rad_s"], period=st / g["speed_rad_s"] * 1.05, drive=g["drive_Nm"] * self.p.wire,
                         hold=g["hold_Nm"], stiffness=0.3, clearance=math.radians(0.5), step_energy=g["J_per_rad"] * st)
        lr = design.leg_actuator(force_scale=self.p.wire, cool_scale=self.p.cool, eta=self.p.eta)
        lr = replace(lr, stiffness=0.05)
        self.leg_ratchet, self.joint_ratchet = lr, jr
        for nm in self.act_names:
            kind = nm[-2:]
            r = jr if kind in ("jp", "jn") else lr
            ratchets.append(r)
            arm = r.stiffness * (dt / 0.3) ** 2
            m.dof_armature[self.vadr[self.aidx[nm]]] = max(m.dof_armature[self.vadr[self.aidx[nm]]], arm)
            damp.append(2 * 0.7 * math.sqrt(r.stiffness * (arm + 1e-9)))
        self.steppers = StepperBank(ratchets, self.data.qpos[self.qadr], damp)
        self._magnets()
        self.rel = np.zeros((self.n, 4))  # carrier positions along their strokes (m)
        self.rel_target = np.zeros((self.n, 4))
        self.rel_ready = np.zeros((self.n, 4))
        self.rel_stall = np.zeros((self.n, 4), int)
        rr = design.release_actuator(force_scale=self.p.wire, cool_scale=self.p.cool)
        self.rel_ratchet = rr
        self.rel_energy = 0.0
        self.steps_done = 0
        self.frames = []
        self.watch = None
        self.watch_every_step = False  # True turns event skipping off (a recording that must see every step)
        self.skipped = 0.0
        self.log = {"broken": [], "events": []}

    # ------------------------------------------------------------------ MJCF
    def _mjcf(self, cubes, scene, floor):
        d, p = self.d, self.p
        h = d.h + p.size / 2
        js = d.joint()
        m_total = d.mass
        parts = d.parts()
        m_leg_plate = d.leg_len * d.leg_w_frac * d.L * d.leg_t * RHO["resin"]
        m_disc = d.disc_r**2 * math.pi * d.disc_t_frac * d.L * RHO["resin"] + 4 * math.pi / 6 * d.d_ring**3 * RHO["ndfeb"] + 0.02e-3
        m_body = m_total - 2 * m_disc - 4 * m_leg_plate
        tc = 1e-3
        solref = f"{tc:.6g} {p.damp:.4g}"
        out = []
        out.append(f'<mujoco><option timestep="{self.dt:.6g}" integrator="implicitfast" cone="elliptic" impratio="10"/><visual><global offwidth="1280" offheight="720"/></visual>')
        out.append('<default><geom solref="%s" solimp="0.95 0.99 0.0002" condim="4"/></default>' % solref)
        if d.edge_round > 0:  # all 12 edges chamfered at 45 deg (convex hull of the cut corners)
            c_ = d.edge_round
            vs = []
            for sx in (1, -1):
                for sy in (1, -1):
                    for sz in (1, -1):
                        vs += [(sx * (h - c_), sy * h, sz * h), (sx * h, sy * (h - c_), sz * h), (sx * h, sy * h, sz * (h - c_))]
            out.append('<asset><mesh name="chamfer" vertex="' + " ".join(f"{v:.7g}" for xyz in vs for v in xyz) + '"/></asset>')
        out.append('<worldbody><light pos="0 0 0.3" dir="0 0 -1" directional="true"/>')
        if floor:
            out.append(f'<geom name="floor" type="plane" size="0.5 0.5 0.01" friction="{p.floor_mu:.4g} 0.002 0.0001" rgba="0.82 0.82 0.80 1"/>')
        out.append(scene)
        colours = ["0.85 0.55 0.25 1", "0.35 0.55 0.80 1", "0.45 0.70 0.45 1", "0.80 0.45 0.55 1", "0.65 0.55 0.80 1", "0.80 0.75 0.40 1", "0.55 0.75 0.75 1", "0.75 0.60 0.45 1", "0.60 0.60 0.60 1", "0.45 0.45 0.75 1", "0.75 0.45 0.35 1", "0.40 0.65 0.55 1"]
        excl = []
        lr = d.leg_actuator()
        for i, (c, R) in enumerate(cubes):
            q = rot_quat(R)
            col = colours[i % len(colours)]
            joint = "" if i in self.fixed else "<freejoint/>"
            out.append(f'<body name="c{i}" pos="{c[0]:.7g} {c[1]:.7g} {c[2]:.7g}" quat="{q[0]:.7g} {q[1]:.7g} {q[2]:.7g} {q[3]:.7g}">{joint}')
            if d.edge_round > 0:
                out.append(f'<geom name="c{i}box" type="mesh" mesh="chamfer" mass="{m_body:.6g}" rgba="{col}" friction="{p.cube_mu:.4g} 0.002 0.0001"/>')
            else:
                out.append(f'<geom name="c{i}box" type="box" size="{h:.6g} {h:.6g} {h:.6g}" mass="{m_body:.6g}" rgba="{col}" friction="{p.cube_mu:.4g} 0.002 0.0001"/>')
            # face marks (+z: a dot pattern so orientation is visible)
            out.append(f'<geom type="box" size="{0.3 * h:.5g} {0.3 * h:.5g} 0.00002" pos="0 0 {h + 2e-5:.6g}" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0" mass="0"/>')
            for part, sgn in (("p", 1), ("n", -1)):
                tread = d.tread
                cx = sgn * (h - d.disc_t_frac * d.L / 2 + tread / 2)
                out.append(f'<body name="c{i}d{part}" pos="0 0 0"><joint name="c{i}j{part}" type="hinge" axis="{sgn} 0 0"/>')
                out.append(f'<geom name="c{i}d{part}g" type="cylinder" size="{d.disc_r:.6g} {(d.disc_t_frac * d.L + tread) / 2:.6g}" pos="{cx:.6g} 0 0" euler="0 90 0" mass="{m_disc:.6g}" rgba="0.25 0.25 0.28 1" friction="{p.tread_mu:.4g} 0.002 0.0001"/>')
                out.append(f'<geom type="box" size="0.00002 {0.06 * d.L:.5g} {0.35 * d.disc_r:.5g}" pos="{sgn * (h + tread + 2e-5):.6g} 0 {0.5 * d.disc_r:.5g}" rgba="0.95 0.95 0.95 1" contype="0" conaffinity="0" mass="0"/>')
                out.append("</body>")
            for leg in LEGS:
                lie, other, d0, a = leg_frame(leg)
                hinge = h * lie + h * other + p.hinge * self.rng.uniform(-1, 1, 3)
                l = d.leg_len + p.leg_len
                w = d.leg_w_frac * d.L
                # the plate sits TIP_PAD deeper in its recess and a TIP_PAD-thick pad at its tip is flush
                # with the face: lying on a neighbour, a leg bears on its tip only, at the full lever
                centre = d0 * (l / 2) - lie * (d.leg_t / 2 + TIP_PAD)
                pad_c = d0 * (l - TIP_PAD_LEN / 2) - lie * (TIP_PAD / 2)
                R_leg = np.column_stack([d0, np.cross(lie, d0), lie])  # local x along the leg, z out of the face
                ql = rot_quat(R_leg)
                out.append(f'<body name="c{i}l{leg}" pos="{hinge[0]:.7g} {hinge[1]:.7g} {hinge[2]:.7g}"><joint name="c{i}{leg}h" type="hinge" axis="{a[0]:.0f} {a[1]:.0f} {a[2]:.0f}" range="-0.05 4.2" limited="true"/>')
                out.append(f'<geom name="c{i}l{leg}g" type="box" size="{l / 2:.6g} {w / 2:.6g} {d.leg_t / 2:.6g}" pos="{centre[0]:.7g} {centre[1]:.7g} {centre[2]:.7g}" quat="{ql[0]:.6g} {ql[1]:.6g} {ql[2]:.6g} {ql[3]:.6g}" mass="{m_leg_plate:.6g}" rgba="0.92 0.92 0.92 1" friction="{p.cube_mu:.4g} 0.002 0.0001"/>')
                out.append(f'<geom name="c{i}l{leg}p" type="box" size="{TIP_PAD_LEN / 2:.6g} {w / 2:.6g} {TIP_PAD / 2:.6g}" pos="{pad_c[0]:.7g} {pad_c[1]:.7g} {pad_c[2]:.7g}" quat="{ql[0]:.6g} {ql[1]:.6g} {ql[2]:.6g} {ql[3]:.6g}" mass="0" rgba="0.6 0.6 0.62 1" friction="{p.cube_mu:.4g} 0.002 0.0001"/>')
                out.append("</body>")
            out.append("</body>")
            names = [f"c{i}dp", f"c{i}dn"] + [f"c{i}l{leg}" for leg in LEGS]
            for a_, b_ in itertools.combinations(names, 2):
                excl.append(f'<exclude body1="{a_}" body2="{b_}"/>')
        out.append("</worldbody>")
        out.append("<contact>" + "".join(excl) + "</contact>")
        if len(cubes) > 1:
            out.append("<equality>" + "".join(f'<weld name="w{a}_{b}" body1="c{a}" body2="c{b}" active="false" solref="0.002 1"/>' for a, b in itertools.combinations(range(len(cubes)), 2)) + "</equality>")
        out.append("<actuator>")
        for i in range(len(cubes)):
            for part in ("p", "n"):
                out.append(f'<motor name="c{i}j{part}" joint="c{i}j{part}" gear="1"/>')
            for leg in LEGS:
                out.append(f'<motor name="c{i}{leg}" joint="c{i}{leg}h" gear="1"/>')
        out.append("</actuator></mujoco>")
        return "\n".join(out)

    # ------------------------------------------------------------------ magnets
    def _magnets(self):
        d, p, m = self.d, self.p, self.model
        h = d.h + p.size / 2
        rows = []  # (body, local pos, kind, moment, cube, carrier index or -1, retraction dir)
        mc = sphere_moment(d.d_corner) * math.sqrt(p.derate)
        mr = sphere_moment(d.d_ring) * math.sqrt(p.derate)
        k_ins = d.k_insert if p.k_insert is None else p.k_insert
        k_tip = d.tip_k if p.tip_k is None else p.tip_k
        a_x = h - d.cover - d.d_corner / 2 - d.setback_frac * d.L
        a_yz = h - d.cover - d.d_corner / 2 - d.corner_shift  # a chamfer moves the corner spheres in to keep their cover
        cidx = {k: n for n, k in enumerate(CARRIERS)}
        for i in range(self.n):
            b = self.body[i]
            for sx in (1, -1):
                for sy in (1, -1):
                    for sz in (1, -1):
                        pos = np.array([sx * a_x, sy * a_yz, sz * a_yz]) + p.pocket * self.rng.uniform(-1, 1, 3)
                        u = -np.array([0, sy, sz]) / math.sqrt(2)  # inward along the x-edge bisector
                        rows.append((b, pos, 0, mc * (1 + p.moment_spread * self.rng.uniform(-1, 1)), i, cidx[carrier_of_corner(sy, sz)], u, np.array([sx, sy, sz]) / math.sqrt(3)))
            for part, sx in (("p", 1), ("n", -1)):
                bd = m.body(f"c{i}d{part}").id
                xr = sx * (h + d.tread - d.cover - d.d_ring / 2)
                for k in range(4):
                    ang = math.pi / 4 + k * math.pi / 2
                    pos = np.array([xr, d.ring_r * math.cos(ang), d.ring_r * math.sin(ang)]) + p.pocket * self.rng.uniform(-1, 1, 3)
                    rows.append((bd, pos, 1, mr * (1 + p.moment_spread * self.rng.uniform(-1, 1)), i, -1, np.zeros(3), np.array([sx, 0, 0.0])))
            # steel inserts on the plain faces, behind the face cover
            for f in ("+y", "-y", "+z", "-z"):
                n_ = AXES[f]
                t1 = np.array([1.0, 0, 0])
                t2 = np.cross(n_, t1)
                depth = h - d.cover - d.insert_recess
                for k in range(4):
                    ang = math.pi / 4 + k * math.pi / 2
                    pos = n_ * depth + d.ring_r * (math.cos(ang) * t1 + math.sin(ang) * t2)
                    rows.append((b, pos, 2, k_ins * mr * 2.0, i, -1, np.zeros(3), n_.copy()))
            # leg tip shims (on the leg bodies, local frame of the leg body = cube frame rotated about the hinge)
            for leg in LEGS:
                lie, other, d0, a = leg_frame(leg)
                bl = m.body(f"c{i}l{leg}").id
                pos = d0 * (d.leg_len - 0.6e-3)  # outer face of the plate, near its tip, centred
                rows.append((bl, pos, 3, k_tip * mc * 2.0, i, -1, np.zeros(3), lie.copy()))
        self.mag_body = np.array([r[0] for r in rows], int)
        self.mag_pos0 = np.array([r[1] for r in rows], float)
        self.mag_kind = np.array([r[2] for r in rows], int)
        self.mag_m = np.array([r[3] for r in rows], float)
        self.mag_cube = np.array([r[4] for r in rows], np.int64)
        self.mag_carrier = np.array([r[5] for r in rows], int)
        self.mag_retract = np.array([r[6] for r in rows], float)
        self.mag_U0 = np.array([r[7] for r in rows], float)
        self.mag_steel = self.mag_kind >= 2
        # world directions start along each magnet's outward axis
        self.U = np.zeros_like(self.mag_U0)
        self.rcut = 3.2 * max(d.d0_corner, d.d0_ring) + 1.0e-3
        # field at which a steel piece saturates: a sphere's field at its own pair distance
        self.b_sat = 2 * 1e-7 * sphere_moment(d.d_ring) / (d.d0_ring) ** 3
        self._world_U()
        self.mag_F = np.zeros((len(rows), 3))

    def _world_U(self):
        R = self.data.xmat[self.mag_body].reshape(-1, 3, 3)
        self.U = np.einsum("nij,nj->ni", R, self.mag_U0)

    def mag_world(self):
        local = self.mag_pos0.copy()
        sel = self.mag_carrier >= 0
        local[sel] += np.maximum(self.rel[self.mag_cube[sel], self.mag_carrier[sel]] - self.d.hook_free, 0.0)[:, None] * self.mag_retract[sel]
        R = self.data.xmat[self.mag_body].reshape(-1, 3, 3)
        return self.data.xpos[self.mag_body] + np.einsum("nij,nj->ni", R, local)

    def _apply_magnets(self):
        P = self.mag_world()
        F, meff = magnet_forces(P, self.U, self.mag_m, self.mag_cube, self.mag_steel, self.rcut, self.b_sat)
        self.mag_F = F
        xf = self.data.xfrc_applied
        xf[:] = 0
        com = self.data.xipos[self.mag_body]
        T = np.cross(P - com, F)
        np.add.at(xf, (self.mag_body, slice(0, 3)), F)
        np.add.at(xf, (self.mag_body, slice(3, 6)), T)
        return P, F

    # ------------------------------------------------------------------ release carriers
    def _carriers(self, t):
        """Kinematic carriers: advance one rack tooth per stroke while the wire beats the magnets'
        pull along the stroke; return (unlock) at the cooling rate."""
        rr = self.rel_ratchet
        F = self.mag_F
        sel = self.mag_carrier >= 0
        R = self.data.xmat[self.mag_body].reshape(-1, 3, 3)
        u_w = np.einsum("nij,nj->ni", R, self.mag_retract)
        load = np.zeros((self.n, 4))
        f_along = -(F * u_w).sum(1) * (self.rel[self.mag_cube, np.maximum(self.mag_carrier, 0)] >= self.d.hook_free - 1e-9)  # magnetic force opposing the ramp (only past the hook travel)
        np.add.at(load, (self.mag_cube[sel], self.mag_carrier[sel]), f_along[sel])
        rate = rr.step / rr.stroke
        dt = self.dt * self.mag_every
        want_in = self.rel_target > self.rel + 1e-9
        want_out = self.rel_target < self.rel - 1e-9
        can = load < rr.drive
        adv = want_in & can
        self.rel = np.where(adv, np.minimum(self.rel + rate * dt, self.rel_target), self.rel)
        stall = want_in & ~can
        self.rel_stall += stall
        self.rel = np.where(want_out, np.maximum(self.rel - rate * dt, self.rel_target), self.rel)
        self._rel_load = load

    def release(self, i, carriers, retract=True, level=None):
        """Slide edge bolts of cube i: level 'hooks' frees the hooks only (0.6 mm), 'full' also retracts
        the edge's corner spheres; retract=False lets them back (shared unlock wire). Energy per rack tooth."""
        for c in carriers:
            k = list(CARRIERS).index(c)
            tgt = (self.d.hook_free if level == "hooks" else self.d.rel_stroke) if retract else 0.0
            if retract and self.rel_target[i, k] < tgt:
                self.rel_energy += math.ceil((tgt - self.rel_target[i, k]) / 0.2e-3 - 1e-9) * self.rel_ratchet.step_energy
            if not retract and self.rel_target[i, k] > 0:
                self.rel_energy += heat_energy(0.025e-3, 0.8 * self.d.inner, 1)
            self.rel_target[i, k] = tgt

    # ------------------------------------------------------------------ latches
    FACE_BOLTS = {"+y": ("pp", "pn"), "-y": ("np", "nn"), "+z": ("pp", "np"), "-z": ("pn", "nn")}

    def face_between(self, i, j):
        """Face of cube i (body label) that faces cube j, and the offset error (m) and angle error (rad)
        from an exact lattice neighbour."""
        p_i, R_i = self.pose(i)
        p_j, R_j = self.pose(j)
        rel = R_i.T @ (p_j - p_i)
        ax = int(np.argmax(np.abs(rel)))
        sgn = 1 if rel[ax] > 0 else -1
        face = ("+" if sgn > 0 else "-") + "xyz"[ax]
        pitch = self.d.L + (self.d.tread if ax == 0 else 0.0)
        ideal = np.zeros(3)
        ideal[ax] = sgn * pitch
        _, ang = snap_rotation(R_i.T @ R_j)
        return face, float(np.linalg.norm(rel - ideal)), float(ang)

    def can_latch(self, i, j, pos_tol=0.35e-3, ang_tol=math.radians(4)):
        """Edge bolts latch a plain face onto a plain face of a cube with a parallel x axis, inside the
        hooks' lead-in (A: 0.35 mm, 4 deg)."""
        fi, e, a = self.face_between(i, j)
        fj, _, _ = self.face_between(j, i)
        _, R_i = self.pose(i)
        _, R_j = self.pose(j)
        parallel = abs(float(R_i[:, 0] @ R_j[:, 0])) > 0.99
        return fi[1] != "x" and fj[1] != "x" and parallel and e < pos_tol and a < ang_tol

    def _weld(self, i, j):
        a, b = min(i, j), max(i, j)
        return self.model.equality(f"w{a}_{b}").id

    def latch(self, i, j):
        """Close the latch between i and j if the faces are inside the lead-in: the weld takes the
        current relative pose (the hooks pull the faces shut). Returns success."""
        if not self.can_latch(i, j):
            return False
        a, b = min(i, j), max(i, j)
        k = self._weld(a, b)
        p_a, R_a = self.pose(a)
        p_b, R_b = self.pose(b)
        relpos = R_a.T @ (p_b - p_a)
        q = rot_quat(R_a.T @ R_b)
        self.model.eq_data[k, :] = 0
        self.model.eq_data[k, 3:6] = relpos
        self.model.eq_data[k, 6:10] = q
        self.model.eq_data[k, 10] = 1.0
        self.data.eq_active[k] = 1
        return True

    def unlatch(self, i, j):
        self.data.eq_active[self._weld(i, j)] = 0

    def latched(self, i, j):
        return bool(self.data.eq_active[self._weld(i, j)])

    def open_face(self, i, j, far_edge=None, timeout=20.0):
        """Unlatch cube i from neighbour j through i's bolts on that face: both bolts free their hooks;
        the bolt on far_edge (a carrier name) also retracts its corner spheres. The weld opens when both
        bolts pass the hook travel. Returns False if a bolt stalls."""
        face, _, _ = self.face_between(i, j)
        bolts = self.FACE_BOLTS.get(face, ())
        for c in bolts:
            self.release(i, [c], True, level="full" if c == far_edge else "hooks")
        ks = [list(CARRIERS).index(c) for c in bolts]
        ok = self.run_until(lambda: all(self.rel[i, k] >= self.d.hook_free - 1e-9 for k in ks), timeout)
        if ok:
            self.unlatch(i, j)
        self.run_until(lambda: all(abs(self.rel[i, k] - self.rel_target[i, k]) < 1e-9 for k in ks), timeout)
        return ok

    # ------------------------------------------------------------------ stepping
    def step(self):
        d = self.data
        if self.steps_done % self.mag_every == 0:
            self._apply_magnets()
            self._carriers(d.time)
        q = d.qpos[self.qadr]
        qd = d.qvel[self.vadr]
        d.ctrl[:] = self.steppers.torque(d.time, q, qd, self.dt)
        mujoco.mj_step(self.model, d)
        self.steps_done += 1
        if self.watch is not None:
            self.watch(self)

    SKIP_V = 2e-5  # m/s or rad/s: below this everywhere, with no stroke and no bolt moving, the world is static

    def _skip(self, t_end):
        """Event skipping: while nothing moves (every velocity below SKIP_V, no stroke in progress, no bolt
        sliding), the state is static and only the wires' cooling clocks run, so time jumps to the next
        stroke start (or t_end) without integrating. Exact for a static state; creep slower than SKIP_V
        is lost (at most 20 um and 0.1 deg per skipped second; hinges up to 0.3 deg/s of lock jitter)."""
        if self.watch_every_step:
            return
        st = self.steppers
        if st.active.any() or self._rel_moving():
            return
        if getattr(self, "_vtol", None) is None:
            m = self.model
            tol = np.full(m.nv, 5e-3)  # hinge dofs: lock-spring jitter of a few mrad/s does not drift
            for j in range(m.njnt):
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    a = m.jnt_dofadr[j]
                    tol[a:a + 3] = self.SKIP_V  # translation
                    tol[a + 3:a + 6] = 2e-3  # rotation, rad/s
            self._vtol = tol
        if np.any(np.abs(self.data.qvel) > self._vtol):
            return
        t = self.data.time
        e = st.target - st.phi
        want = (np.abs(e) > 0.55 * st.step) & ~st.free
        col = np.where(e > 0, 0, 1)
        rdy = st.ready[np.arange(st.n), col]
        rest = np.where((st.fails >= 2), st.fail_t + 1.0, -np.inf)
        nxt = np.maximum(rdy, rest)
        t_next = float(np.min(nxt[want])) if want.any() else np.inf
        t_to = min(t_next, t_end) - self.dt
        if t_to - t > 10 * self.dt:
            self.data.time = t_to
            self.skipped += t_to - t

    def run(self, seconds):
        t_end = self.data.time + seconds
        while self.data.time < t_end - 0.5 * self.dt:
            if self.steps_done % 20 == 0:
                self._skip(t_end)
            if self.data.time >= t_end - 0.5 * self.dt:
                break
            self.step()

    def run_until(self, cond, timeout):
        t_end = self.data.time + timeout
        while self.data.time < t_end:
            if self.steps_done % 50 == 0:
                if cond():  # checked before any skip, so a skip never jumps past the condition
                    return True
                self._skip(t_end)
            self.step()
        return bool(cond())

    def wait_actuators(self, timeout=60.0, settle=0.15):
        ok = self.run_until(lambda: not self.steppers.busy() and not self._rel_moving(), timeout)
        self.run(settle)
        return ok

    def _rel_moving(self):
        return bool(np.any(np.abs(self.rel_target - self.rel) > 1e-7) and np.any((self.rel_target > self.rel + 1e-7) & (self._rel_load < self.rel_ratchet.drive) | (self.rel_target < self.rel - 1e-7)))

    # ------------------------------------------------------------------ commands
    def leg(self, i, leg, angle, frac=1.0):
        a = self.aidx[f"c{i}{leg}"]
        self.steppers.drive_frac[a] = frac
        self.steppers.go_to(a, angle)

    def joint(self, i, part, angle):
        a = self.aidx[f"c{i}j{part}"]
        self.steppers.go_to(a, angle)

    def angle(self, name):
        return float(self.data.qpos[self.qadr[self.aidx[name]]])

    # ------------------------------------------------------------------ state
    def pose(self, i):
        b = self.body[i]
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def bonded_pairs(self, i, j, tol=1.15):
        """Corner-sphere pairs between cubes i and j closer than tol x d0 (and ring/insert pairs)."""
        P = self.mag_world()
        a = np.where((self.mag_cube == i) & (self.mag_kind <= 1))[0]
        b = np.where((self.mag_cube == j) & (self.mag_kind <= 1))[0]
        if len(a) == 0 or len(b) == 0:
            return 0
        dist = np.linalg.norm(P[a][:, None] - P[b][None], axis=2)
        d0 = np.where(self.mag_kind[a][:, None] == 0, self.d.d0_corner, self.d.d0_ring)
        return int(np.sum(dist < tol * d0 + 0.02e-3))


def snap_rotation(R):
    """Nearest of the 24 lattice rotations and the angle (rad) to it."""
    best, err = None, 9.0
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for r, (c, s) in enumerate(zip(perm, signs)):
                M[r, c] = s
            if np.linalg.det(M) < 0:
                continue
            D = M.T @ R
            ang = math.acos(max(-1.0, min(1.0, (np.trace(D) - 1) / 2)))
            if ang < err:
                best, err = M, ang
    return best, err


LATTICE_ROTATIONS = []
for _perm in itertools.permutations(range(3)):
    for _signs in itertools.product((1, -1), repeat=3):
        _M = np.zeros((3, 3))
        for _r, (_c, _s) in enumerate(zip(_perm, _signs)):
            _M[_r, _c] = _s
        if np.linalg.det(_M) > 0:
            LATTICE_ROTATIONS.append(_M)


# ---------------------------------------------------------------------------------------------
# Discrete move model on the voxel grid
# ---------------------------------------------------------------------------------------------

Z = np.array([0, 0, 1])


def rot90(axis, quarter=1):
    """Integer rotation matrix: quarter turns about a lattice axis (a unit int vector)."""
    a = np.asarray(axis, int)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3, dtype=int)
    for _ in range(quarter % 4):
        R = (np.eye(3, dtype=int) + K + K @ K) @ R  # exact for 90 deg: I + sin K + (1 - cos) K^2
    return R


def key(R):
    return tuple(int(v) for v in np.asarray(R).reshape(-1))


@dataclass(frozen=True)
class Move:
    kind: str  # BOOK, KICK, CLIMB, DESCEND, BOOK_DESCEND, YAW
    leg: str | None
    delta: tuple  # cell change
    R_new: tuple  # new rotation key
    catch: str | None = None  # leg that can catch the landing (KICK)
    release: tuple = ()  # carriers that must be released for this move (bonded cases)


def leg_world(R, leg):
    lie, other = (AXES[f] for f in LEGS[leg])
    return (R @ lie).round().astype(int), (R @ other).round().astype(int)


def moves(R, cell, occupied, legs=tuple(LEGS), yaw=True, supports_ok=None, bond_legs=("XP", "XN")):
    """Grid moves of a cube with rotation R (body -> world) at `cell` among `occupied` cells (a set of
    tuples, not including this cube). supports_ok(cell_of_support) -> bool decides whether a support
    is braced (a row or a structure) for climbs and descents; default: braced if it has a neighbour
    other than the mover or a cube under it.

    Moves that open a bond while pivoting (CLIMB, BOOK on a structure, BOOK_DESCEND) need the far edge
    of the bonded face to be x-parallel, so its bolt can retract that edge's spheres while the hinge
    edge holds; only the x-legs (bond_legs) give that. There is no DESCEND by a vertical-face leg: to
    start rolling a cube off a support's edge the leg would have to pull on the support's side (the leg
    tip only pushes and its shim has no steel to hold on there), so descents are BOOK_DESCEND only.
    CLIMB pushes quasi-statically up to the balance point (135 deg); the last 45 deg the cube settles
    onto the top under gravity about the magnetic hinge (like KICK on the floor)."""
    R = np.asarray(R, int)
    c = np.asarray(cell, int)
    occ = set(occupied)

    def free(v):
        v = tuple(int(x) for x in v)
        return v not in occ and v[2] >= 0

    def supported(v):
        v = np.asarray(v, int)
        return v[2] == 0 or tuple(v - Z) in occ

    def braced(s):
        if supports_ok is not None:
            return supports_ok(tuple(s))
        s = np.asarray(s, int)
        nb = [tuple(s + o) for o in (np.array(v) for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, -1)))]
        return any(n in occ for n in nb if n != tuple(c))

    out = []
    for leg in legs:
        wl, wo = leg_world(R, leg)
        if (wl == -Z).all() and (c[2] == 0 or leg in bond_legs):  # leg in the bottom face: book-open over its hinge edge toward wo
            dvec = wo
            dest = c + dvec
            axis = np.cross(Z, dvec)
            if free(dest) and supported(dest):
                out.append(Move("BOOK", leg, tuple(dvec), key(rot90(axis) @ R)))
            elif free(dest) and free(dest - Z) and supported(dest - Z) and (c[2] > 0):
                out.append(Move("BOOK_DESCEND", leg, tuple(dvec - Z), key(rot90(axis, 2) @ R)))
        if (wo == -Z).all() and wl[2] == 0 and (c[2] == 0 or leg in bond_legs):  # hinge on the bottom edge, leg in a vertical face
            push = c + wl
            dvec = -wl
            dest = c + dvec
            axis = np.cross(Z, dvec)
            R1 = rot90(axis) @ R
            if free(push) and supported(push) and free(dest) and supported(dest):
                # catch: a leg hinged at the landing face's far edge (landing face = old dvec face, far edge = with old top)
                land = R.T @ dvec
                top = R.T @ Z
                catch = next((lg for lg in legs if (AXES[LEGS[lg][0]] == land).all() and (AXES[LEGS[lg][1]] == top).all()), None)
                out.append(Move("KICK", leg, tuple(dvec), key(R1), catch=catch))
        if (wo == Z).all() and wl[2] == 0 and leg in bond_legs:  # hinge on the top edge, leg in a vertical face: climb onto the cube it faces
            s = c + wl
            if tuple(s) in occ and free(s + Z) and free(c + Z) and braced(s):
                out.append(Move("CLIMB", leg, tuple(wl + Z), key(rot90(np.cross(Z, wl), -2) @ R)))
    if yaw and (abs((R @ np.array([1, 0, 0]))[2]) == 1):
        for q in (1, -1):
            out.append(Move("YAW", None, (0, 0, 0), key(rot90(Z, q) @ R)))
    return out


def bfs_lone(legs=tuple(LEGS), yaw=True, radius=4):
    """Lone cube on a bare floor: from each of the 24 orientations, the set of (cell, orientation)
    reachable within the box |x|, |y| <= radius; returns the minimum number of moves to reach every
    orientation at the start cell and every neighbouring cell in the home orientation (R = I)."""
    from collections import deque

    starts = [key(R) for R in LATTICE_ROTATIONS]
    home = key(np.eye(3, dtype=int))
    res = {}
    for s in starts:
        dist = {((0, 0, 0), s): 0}
        dq = deque([((0, 0, 0), s)])
        while dq:
            cell, Rk = dq.popleft()
            R = np.array(Rk).reshape(3, 3)
            for mv in moves(R, cell, set(), legs, yaw):
                nc = tuple(int(a + b) for a, b in zip(cell, mv.delta))
                if abs(nc[0]) > radius or abs(nc[1]) > radius:
                    continue
                st = (nc, mv.R_new)
                if st not in dist:
                    dist[st] = dist[(cell, Rk)] + 1
                    dq.append(st)
        to_home_here = dist.get(((0, 0, 0), home))
        to_home_any = min((v for (c, r), v in dist.items() if r == home), default=None)
        nbr = [dist.get((cc, home)) for cc in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))]
        res[s] = {"home_here": to_home_here, "home_any": to_home_any, "home_at_neighbours": nbr, "reached": len(dist)}
    return res
