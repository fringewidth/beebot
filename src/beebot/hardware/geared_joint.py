"""A geared twist joint for the small reaction-wheel cube, and the machines built from it.

The 7-14 mm reaction-wheel cube (runs/module_phys/concepts.md, picks 1 and 2) has two flywheels
spun by motors printed in the face flex. Here one flywheel motor at a time drives a twist disc
through a gear train instead: a swing-gear selector meshes the motor pinion with the train, the
train ends in a self-locking worm whose wheel is the disc's hub, and the disc carries its own bond.
Memo: runs/handoff/small-joint.md. Analysis and search: scripts/module_phys/small_joint.py.
Machines, Monte Carlo and video: scripts/small_joint_machines.py.

Three layers:

1. Gear, motor and bond physics (no MuJoCo): `Mesh`, `Train`, `FlexMotor`, `JointDesign`,
   `sphere_pair_force`, `BondModel`. Every constant carries a tag: V (vendor page or paper read
   this session), E (estimated from stated physics), A (assumed, to be measured).
1. `JointParams`: the numbers the simulation needs, taken from a `JointDesign`.
1. A MuJoCo builder for cubes of side L with two geared discs on their +x and -x faces (the
   semantics of layout B in module_b.py, so its machine lattices carry over), and `GearedRig`,
   which integrates each joint's motor and gear train in Python every step.

What the rig models per joint: a DC motor (k_t, R, supply voltage, current limit) on the rotor
inertia reflected through the ratio, a train with forward efficiency and the worm's backward
efficiency (self-locking when the worm's lead angle is below its friction angle, so a load cannot
back-drive it and the joint holds with no power), backlash as a dead zone and compliance as a
spring-damper between the train output and the disc, the tooth strength of the output stage
(a joint whose transmitted torque passes it is counted as broken), and the selector (a free disc
turns with a small drag). Bonds are rigid in the simulation; their loads (bending, tension, twist)
are compared every few steps with capacities from the magnet model, smoothed over a few
milliseconds, and the first bond to pass its capacity counts as a failure of the run.

Idealised (left out): rigid bonds (no compliance, no opening; failure is judged from loads, the
structure does not come apart), exact joint angle sensing (the controller reads the disc angle),
no motor heating feedback, no cogging, the rotor speed is capped at the bearing limit instead of a
bearing friction law, MuJoCo soft contacts at the feet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

from beebot.hardware.module_b import axes_rotation, bond_rotation, face_of, part_of, rotation_about, unit

G = 9.81
MU0 = 4e-7 * math.pi
RHO_CU = 1.72e-8  # ohm m (V, handbook)
BR_N52 = 1.45  # T, as mmsim.magnetics.BR_BY_GRADE (V, grade table)
DERATE_SPHERE = 0.80  # published / model pull for a 3 mm N42 sphere (scaling.py calibration, V secondary)
DERATE_MMSIM = 0.839  # mmsim image-method derating (simstack.md, set on 3 rows)
MARGIN = 1.5

# ---------------------------------------------------------------------------------------------
# Gear physics
# ---------------------------------------------------------------------------------------------

# Lewis form factor Y (load at the tip, 20 deg full depth), by tooth count (E, standard tables;
# 9 and 10 extrapolated, undercut pinions are weaker still).
LEWIS = {9: 0.18, 10: 0.20, 11: 0.226, 12: 0.245, 13: 0.261, 14: 0.277, 15: 0.290, 16: 0.296, 17: 0.303, 18: 0.309, 19: 0.314, 20: 0.322, 22: 0.331, 24: 0.337, 26: 0.346, 28: 0.353, 30: 0.359, 34: 0.371, 40: 0.389, 50: 0.408, 60: 0.421}


def lewis_y(z: int) -> float:
    keys = sorted(LEWIS)
    if z <= keys[0]:
        return LEWIS[keys[0]]
    if z >= keys[-1]:
        return LEWIS[keys[-1]]
    return float(np.interp(z, keys, [LEWIS[k] for k in keys]))


@dataclass(frozen=True)
class Material:
    name: str
    sigma_cont: float  # Pa, allowable root bending stress for repeated loading
    sigma_peak: float  # Pa, allowable for rare peaks (impacts, stalls)
    density: float  # kg/m3
    modulus: float  # Pa
    tag: str


POM = Material("POM, moulded (Didel)", 25e6, 50e6, 1410.0, 2.8e9, "A: POM flexural strength about 90 MPa; 25 MPa is the usual plastic-gear allowable at room temperature, 50 MPa for rare peaks")
RESIN = Material("tough resin, hobby MSLA", 15e6, 30e6, 1150.0, 2.0e9, "A: tough resins quote 40-55 MPa tensile; a third and two thirds of that for printed teeth with layer lines")


@dataclass(frozen=True)
class Mesh:
    """One gear mesh, driving member first. kind 'spur' (z1 = pinion teeth) or 'worm' (z1 = worm
    starts, z2 = wheel teeth, worm_r = worm pitch radius). module and face in metres. The driven
    member's tooth bending strength (Lewis, weaker of the two members) sets the torque capacity."""

    name: str
    kind: str
    module: float
    z1: int
    z2: int
    face: float
    material: Material
    eta_spur: float = 0.88  # E: Didel claims about 0.9 a stage (V, vendor claim); 0.88 with pin bearings in plastic
    worm_r: float = 0.0
    mu: float = 0.25  # worm-wheel sliding friction, dry resin or POM (A)
    backlash: float = 40e-6  # m, circumferential play at the driven pitch circle (E: 20-40 um centre distance allowance x 2 tan 20 deg, plus tooth error)
    source: str = ""

    @property
    def ratio(self) -> float:
        return self.z2 / self.z1

    @property
    def r2(self) -> float:
        return self.module * self.z2 / 2

    @property
    def lead_angle(self) -> float:
        if self.kind != "worm":
            return float("nan")
        lead = self.z1 * math.pi * self.module
        return math.atan(lead / (2 * math.pi * self.worm_r))

    def efficiency(self, mu: float | None = None) -> float:
        """Forward efficiency (driving member drives). Worm: tan(lam) / tan(lam + phi), with the
        normal pressure angle folded into phi (E, standard worm formula, small angles)."""
        if self.kind == "spur":
            return self.eta_spur
        phi = math.atan((self.mu if mu is None else mu) / math.cos(math.radians(20)))
        lam = self.lead_angle
        return math.tan(lam) / math.tan(lam + phi)

    def back_efficiency(self, mu: float | None = None) -> float:
        """Efficiency when the driven member drives back. Worm: tan(lam - phi) / tan(lam), negative
        when self-locking (the load cannot turn the worm; the motor must push even to lower it)."""
        if self.kind == "spur":
            return self.eta_spur
        phi = math.atan((self.mu if mu is None else mu) / math.cos(math.radians(20)))
        lam = self.lead_angle
        return math.tan(lam - phi) / math.tan(lam)

    def self_locking(self, mu: float | None = None) -> bool:
        return self.kind == "worm" and self.back_efficiency(mu) <= 0.0

    def capacity(self, peak: bool = False) -> float:
        """Torque (N m) on the driven member at which the weaker tooth reaches its allowable stress.
        Worm: the wheel's teeth (the worm's thread is a continuous helix and stronger)."""
        sigma = self.material.sigma_peak if peak else self.material.sigma_cont
        z_weak = self.z2 if self.kind == "worm" else self.z1
        force = sigma * self.face * self.module * lewis_y(z_weak)
        return force * self.r2

    def stiffness(self) -> float:
        """Torsional mesh stiffness at the driven member (N m/rad): c' b r2^2 with c' scaled from
        steel's 14 N/(mm um) by the modulus (E, rough; plastic gears also creep)."""
        c_steel = 14e9  # N/m per m of face width (14 N/(mm um))
        c = c_steel * self.material.modulus / 206e9
        return c * self.face * self.r2**2


@dataclass(frozen=True)
class Wolfrom:
    """Split-ring (Wolfrom) planetary, printed: sun S drives n planets of P teeth that mesh a fixed
    ring R1 = S + 2P and an output ring R2 = R1 + n (same planet, profile-shifted), so
    ratio = (1 + R1 / S) R2 / (R2 - R1). The output ring is the disc's hub. The torque is shared by
    the planets at the output ring's radius, which is why this stage carries several times what a
    single pinion of the same module can (E). Efficiency 1 / (1 + (ratio - 1) eps) with eps the
    loss per unit of ratio (E: eps 0.025 for greased printed gears; printed Wolfrom boxes are
    reported at 30-60 %); backward efficiency 1 - (ratio - 1) eps, negative when self-locking."""

    name: str
    module: float
    sun: int
    planet: int
    planets: int
    face: float
    material: Material
    eps: float = 0.025
    share: float = 0.7  # load share factor over the planets (E: printed tolerances)
    backlash: float = 60e-6
    source: str = "printed (A)"
    kind: str = "wolfrom"

    @property
    def ring1(self) -> int:
        return self.sun + 2 * self.planet

    @property
    def ring2(self) -> int:
        return self.ring1 + self.planets

    @property
    def ratio(self) -> float:
        return (1 + self.ring1 / self.sun) * self.ring2 / (self.ring2 - self.ring1)

    @property
    def r2(self) -> float:
        return self.module * self.ring2 / 2

    def efficiency(self, mu=None, eps=None) -> float:
        e = self.eps if eps is None else eps
        return 1.0 / (1.0 + (self.ratio - 1.0) * e)

    def back_efficiency(self, mu=None, eps=None) -> float:
        e = self.eps if eps is None else eps
        return 1.0 - (self.ratio - 1.0) * e

    def self_locking(self, mu=None) -> bool:
        return self.back_efficiency() <= 0

    def capacity(self, peak: bool = False) -> float:
        """Output ring torque at which the planet teeth (the weakest) reach their allowable stress."""
        sigma = self.material.sigma_peak if peak else self.material.sigma_cont
        force = sigma * self.face * self.module * lewis_y(self.planet) * self.planets * self.share
        return force * self.r2

    def stiffness(self) -> float:
        c = 14e9 * self.material.modulus / 206e9
        return c * self.face * self.planets * self.share * self.r2**2

    def outer_radius(self) -> float:
        return self.r2 + 1.25 * self.module + 0.5e-3  # ring teeth plus a 0.5 mm printed wall


def wolfrom_for(L: float, t_rotor: float, module: float | None = None, face: float = 1.4e-3) -> Wolfrom:
    """The largest split-ring stage under the rotor slabs: outer radius at most h - t_rotor - 0.2 mm
    (h the inner half side), sun 10, planets 8, three planets, module at least 0.25 mm (A: the
    finest module trusted on a 20-35 um hobby MSLA printer without a test print, see the memo)."""
    h = L / 2 - 0.4e-3
    room = h - t_rotor - 0.2e-3
    sun, planet, n = 10, 8, 3
    ring2 = sun + 2 * planet + n
    if module is None:
        module = (room - 0.5e-3) / (ring2 / 2 + 1.25)
    return Wolfrom(f"printed split ring, sun {sun}, planets {planet} x{n}, rings {sun + 2 * planet}/{ring2}, module {module * 1e3:.2f}", module, sun, planet, n, face, RESIN)


@dataclass
class Train:
    meshes: list  # motor first, output last

    @property
    def ratio(self) -> float:
        return float(np.prod([m.ratio for m in self.meshes]))

    def efficiency(self, mu: float | None = None, eta_spur: float | None = None) -> float:
        out = 1.0
        for m in self.meshes:
            out *= (m.eta_spur if eta_spur is None else eta_spur) if m.kind == "spur" else m.efficiency(mu)
        return out

    def back_efficiency(self, mu: float | None = None, eta_spur: float | None = None) -> float:
        """Load driving the motor. A self-locking worm anywhere makes it negative."""
        out = 1.0
        for m in self.meshes:
            out *= (m.eta_spur if eta_spur is None else eta_spur) if m.kind == "spur" else m.back_efficiency(mu)
        if any(m.kind != "spur" and m.back_efficiency(mu) < 0 for m in self.meshes):
            out = -abs(out)
        return out

    def self_locking(self, mu: float | None = None) -> bool:
        return any(m.self_locking(mu) for m in self.meshes)

    def capacity(self, peak: bool = False) -> tuple[float, str]:
        """Output torque at which the first mesh reaches its tooth limit while the motor drives
        (upstream torques are the output torque divided by the downstream ratio and efficiency)."""
        best, who = float("inf"), ""
        for k, m in enumerate(self.meshes):
            down = 1.0
            for d in self.meshes[k + 1:]:
                down *= d.ratio * d.efficiency()
            t_out = m.capacity(peak) * down
            if t_out < best:
                best, who = t_out, m.name
        return best, who

    def backlash_out(self) -> float:
        """Output backlash (rad): each mesh's circumferential play over its pitch radius, divided by
        the ratio after it."""
        total = 0.0
        for k, m in enumerate(self.meshes):
            down = float(np.prod([d.ratio for d in self.meshes[k + 1:]])) if k + 1 < len(self.meshes) else 1.0
            total += m.backlash / m.r2 / down
        return total

    def stiffness_out(self) -> float:
        """Series stiffness at the output (N m/rad); upstream meshes count times the ratio squared.
        Shaft and bearing compliance is taken as a 0.5 factor (E)."""
        inv = 0.0
        for k, m in enumerate(self.meshes):
            down = float(np.prod([d.ratio for d in self.meshes[k + 1:]])) if k + 1 < len(self.meshes) else 1.0
            inv += 1.0 / (m.stiffness() * down**2)
        return 0.5 / inv


def disc_field(br, radius, t, z):
    """On-axis field of an axial cylinder magnet at distance z from its face (exact)."""
    return br / 2 * ((z + t) / math.hypot(z + t, radius) - z / math.hypot(z, radius))


@dataclass
class FlexMotor:
    """Axial-flux motor printed in the face flex (scaling.py PcbMotor: 2 layers of 35 um copper at
    55 % fill under a diametric N52 rotor core of radius r_o/2, 0.1 mm gap; torque = 0.6 B J t A r).
    The rotor is the flywheel: NdFeB core of radius r_o/2 and a tungsten heavy-alloy rim to r_o.
    The winding is chosen by k_t (more turns of finer trace raise k_t and R together at fixed k_m)."""

    r_o: float  # m, rotor radius
    t_r: float  # m, rotor thickness
    k_t: float = 4.0e-4  # N m/A (A: chosen so the no-load speed at 3.7 V is about 9,000 rad/s)
    volts: float = 3.7  # V, LiPo or the chain bus (A)
    i_max: float = 0.20  # A, current limit set by the supply and the coil's 1 s heating (A)
    omega_max: float = 6000.0  # rad/s, pin-in-bushing bearing limit (scaling.py OMEGA_MAX_PIN, A)
    layers: int = 2
    rim_density: float = 17500.0  # kg/m3, tungsten heavy alloy (E)

    @property
    def k_m(self) -> float:
        r_i = 0.3 * self.r_o
        area = math.pi * (self.r_o**2 - r_i**2)
        r_mean = 2 / 3 * (self.r_o**3 - r_i**3) / (self.r_o**2 - r_i**2)
        t_eff = self.layers * 35e-6 * 0.55
        b = 0.7 * disc_field(BR_N52, self.r_o / 2, self.t_r, 0.1e-3 + self.layers * 35e-6 / 2)
        return 0.6 * b * r_mean * math.sqrt(t_eff * area / RHO_CU)

    @property
    def resistance(self) -> float:
        return (self.k_t / self.k_m) ** 2

    @property
    def rotor_mass(self) -> float:
        core = 7500.0 * math.pi * (self.r_o / 2) ** 2 * self.t_r
        rim = self.rim_density * math.pi * (self.r_o**2 - (self.r_o / 2) ** 2) * self.t_r
        return core + rim

    @property
    def inertia(self) -> float:
        core = 0.5 * 7500.0 * math.pi * self.t_r * (self.r_o / 2) ** 4
        rim = 0.5 * self.rim_density * math.pi * self.t_r * (self.r_o**4 - (self.r_o / 2) ** 4)
        return core + rim

    @property
    def stall_torque(self) -> float:
        """Motor torque at the current limit (or at stall from the supply, if lower)."""
        return self.k_t * min(self.i_max, self.volts / self.resistance)

    @property
    def no_load_speed(self) -> float:
        return min(self.volts / self.k_t, self.omega_max)


@dataclass
class JointDesign:
    name: str
    motor: FlexMotor
    train: Train
    lock: str = "self-locking worm (output stage)"
    selector: str = "swing gear on the first mesh, moved by the pawl coil"

    def stall_out(self, peak: bool = True) -> float:
        """Output torque the motor can make at its current limit, through the train."""
        return self.motor.stall_torque * self.train.ratio * self.train.efficiency()

    def capacity_out(self, peak: bool = False) -> float:
        """Usable output torque: the lower of the motor-limited torque and the tooth limit."""
        return min(self.stall_out(), self.train.capacity(peak)[0])

    def speed_out(self) -> float:
        return self.motor.no_load_speed / self.train.ratio

    def reflected_inertia(self) -> float:
        return self.motor.inertia * self.train.ratio**2

    def table(self) -> dict:
        tr = self.train
        cap_c, who_c = tr.capacity(False)
        cap_p, who_p = tr.capacity(True)
        return {
            "name": self.name,
            "ratio": round(tr.ratio, 1),
            "eta_forward": round(tr.efficiency(), 3),
            "eta_back": round(tr.back_efficiency(), 3),
            "self_locking": tr.self_locking(),
            "motor_k_m_NmpsqrtW": float(f"{self.motor.k_m:.3g}"),
            "motor_R_ohm": round(self.motor.resistance, 2),
            "motor_stall_uNm": round(self.motor.stall_torque * 1e6, 1),
            "motor_limited_out_mNm": round(self.stall_out() * 1e3, 2),
            "tooth_cont_out_mNm": round(cap_c * 1e3, 2),
            "tooth_peak_out_mNm": round(cap_p * 1e3, 2),
            "weakest_mesh": who_c,
            "usable_cont_mNm": round(min(self.stall_out(), cap_c) * 1e3, 2),
            "usable_peak_mNm": round(min(self.stall_out(), cap_p) * 1e3, 2),
            "no_load_speed_rad_s": round(self.speed_out(), 2),
            "backlash_out_mrad": round(tr.backlash_out() * 1e3, 1),
            "stiffness_out_Nm_rad": round(tr.stiffness_out(), 3),
            "reflected_inertia_kgm2": float(f"{self.reflected_inertia():.3g}"),
            "lock": self.lock,
            "selector": self.selector,
        }


# Bought parts: Didel module 0.2 plastic gears (V: www.didel.com/Microgears.pdf, read 2026-11-15:
# G209 9 teeth 2.2 mm OD 0.61/0.71 bore 2.20 CHF; G220 20/9 4.4 mm OD 0.60 bore 4.00 CHF;
# G240 40/12 8.4 mm OD 6.50 CHF; worm GW205 2.8 mm OD 5.00 CHF; centre distance 9-20 2.90 mm).
# Face widths are not printed in the catalogue (A: 0.8 mm).
DIDEL_FACE = 0.8e-3
DIDEL = {
    "G209": {"teeth": (9,), "od_mm": 2.2, "bore_mm": 0.61, "mass_g": 0.01, "chf": 2.20},
    "G220": {"teeth": (20, 9), "od_mm": 4.4, "bore_mm": 0.60, "mass_g": 0.09, "chf": 4.00},
    "G240": {"teeth": (40, 12), "od_mm": 8.4, "bore_mm": 0.78, "mass_g": 0.13, "chf": 6.50},
    "GW205": {"teeth": (1,), "od_mm": 2.8, "bore_mm": 0.78, "mass_g": 0.021, "chf": 5.00},
}


def worm_last_train(L: float, n_spur: int = 5, worm_module: float = 0.3e-3, worm_r: float = 1.4e-3, wheel_face: float = 1.2e-3, mu: float = 0.25) -> Train:
    """The chosen train for a cube of side L (m). Motor first:

    1. Didel G209 (9) on the rotor pin into the first Didel G220 (20/9), which sits on the
       selector's swing arm.
    1. n_spur - 1 further Didel G220 stages, 9 into 20 (2.22 each).
    1. A printed module 0.3 pair, 10 into 14, onto the worm shaft (the torque there is too high
       for module 0.2 teeth, see `Train.capacity`).
    1. A printed single-start worm (module 0.3, pitch radius 1.4 mm) into a printed worm wheel on
       the disc hub, as large as the cavity allows: radius h - 3.0 mm, with h the inner half side.

    The worm is last so that the self-locking stage, and every impact the joint takes, sits on the
    largest, strongest teeth, and the module 0.2 stages only ever see the motor-side torque."""
    h = L / 2 - 0.4e-3
    r_w = h - 3.0e-3
    z_w = max(12, int(2 * r_w / worm_module))
    meshes = [Mesh("G209 9 into G220 20 (Didel, swing gear)", "spur", 0.2e-3, 9, 20, DIDEL_FACE, POM, source="Didel G209 + G220 (V)")]
    for k in range(n_spur - 1):
        meshes.append(Mesh(f"G220 9 into G220 20 (Didel) #{k + 2}", "spur", 0.2e-3, 9, 20, DIDEL_FACE, POM, source="Didel G220 (V)"))
    meshes.append(Mesh("printed 10 into 14, module 0.3, worm shaft", "spur", 0.3e-3, 10, 14, 1.2e-3, RESIN, backlash=50e-6, source="printed (A)"))
    meshes.append(Mesh(f"printed worm into wheel ({z_w} teeth, module 0.3) on the disc", "worm", worm_module, 1, z_w, wheel_face, RESIN, worm_r=worm_r, mu=mu, backlash=60e-6, source="printed (A)"))
    return Train(meshes)


def worm_last_design(L: float = 14e-3, n_spur: int = 5, k_t: float = 4.0e-4, i_max: float = 0.20, volts: float = 3.7) -> JointDesign:
    """The first design tried (kept for comparison): Didel spur stages, then a printed worm into a
    worm wheel on the disc. Rotor radius 0.25 L, thickness 0.1 L."""
    motor = FlexMotor(r_o=0.25 * L, t_r=0.10 * L, k_t=k_t, i_max=i_max, volts=volts)
    return JointDesign(f"worm-last train, {n_spur} Didel stages, L {L * 1e3:.0f} mm", motor, worm_last_train(L, n_spur))


def rotor_size(L: float) -> tuple[float, float]:
    """Rotor radius and thickness (m) of the geared cube, from the envelope layout: the rotor sits
    under its face, inside the ring of bond spheres (scripts/module_phys/small_joint.py layout)."""
    t_r = max(0.9e-3, 0.09 * L)
    r_o = 0.30 * L
    return r_o, t_r


def wolfrom_train(L: float, module: float | None = None, face: float = 1.4e-3, mu: float = 0.25, eps: float = 0.025) -> Train:
    """The chosen train, motor first:

    1. Didel G209 (9) on the rotor pin, a second G209 as an idler on the selector's swing arm
       (ratio 1, it only makes or breaks the mesh), into a Didel G220 (20) on the worm arbor.
    1. Didel worm GW205 (module 0.2, single start, pitch diameter 2.4 mm) into a Didel G220 (20)
       on the sun shaft: the right angle from the rotor axis to the disc axis, and the power-off
       lock (lead angle 4.8 deg, self-locking for any friction coefficient above 0.083).
    1. Printed split-ring planetary into the disc, as large as fits under the rotor slabs."""
    _, t_r = rotor_size(L)
    meshes = [
        Mesh("G209 9 into G209 9 idler (Didel, swing gear)", "spur", 0.2e-3, 9, 9, DIDEL_FACE, POM, source="Didel G209 x2 (V)"),
        Mesh("G209 idler 9 into G220 20 (Didel, worm arbor)", "spur", 0.2e-3, 9, 20, DIDEL_FACE, POM, source="Didel G209 + G220 (V)"),
        Mesh("worm GW205 into G220 20 (Didel, sun shaft)", "worm", 0.2e-3, 1, 20, DIDEL_FACE, POM, worm_r=1.2e-3, mu=mu, backlash=40e-6, source="Didel GW205 + G220 (V)"),
    ]
    stage = wolfrom_for(L, t_r, module, face)
    meshes.append(replace(stage, eps=eps))
    return Train(meshes)


def nominal_design(L: float = 12e-3, k_t: float = 4.0e-4, i_max: float = 0.20, volts: float = 3.7, module: float | None = None, face: float = 1.4e-3) -> JointDesign:
    """Joint of the chosen cube: flex motor on the flywheel rotor, Didel spur and worm stages,
    printed split-ring output."""
    r_o, t_r = rotor_size(L)
    motor = FlexMotor(r_o=r_o, t_r=t_r, k_t=k_t, i_max=i_max, volts=volts)
    return JointDesign(f"Didel worm + printed split ring, L {L * 1e3:.1f} mm", motor, wolfrom_train(L, module, face))


# ---------------------------------------------------------------------------------------------
# Bonds: free N52 spheres (always attract, so genderless), magpylib, derated
# ---------------------------------------------------------------------------------------------


def sphere_pair_force(d_sphere: float, gap: float = 0.1e-3, derate: float = DERATE_SPHERE, use_magpylib: bool = True) -> float:
    """Attraction (N) of two free N52 spheres of diameter d_sphere whose surfaces are `gap` apart
    (two covers). Free spheres turn to align with the line between them, so the coaxial value is
    the one that acts. magpylib (the mmsim engine) with a fine mesh; the exact dipole value agrees
    to 0.1 % (a uniform sphere's field is exactly a dipole). Derated by the published/model ratio."""
    r = d_sphere / 2
    d = 2 * r + gap
    if use_magpylib:
        import magpylib as mp

        a = mp.magnet.Sphere(diameter=d_sphere, polarization=(0, 0, BR_N52), position=(0, 0, 0))
        b = mp.magnet.Sphere(diameter=d_sphere, polarization=(0, 0, BR_N52), position=(0, 0, d))
        f, _ = mp.getFT(a, b, meshing=100)
        force = float(-np.asarray(f).reshape(-1)[2])
    else:
        m = BR_N52 / MU0 * 4 / 3 * math.pi * r**3
        force = 3 * MU0 * m * m / (2 * math.pi * d**4)
    return derate * force


def free_pair_potential(r: float, d: np.ndarray) -> np.ndarray:
    """Energy (J) of two free N52 spheres of radius r, centres d apart, aligned head to tail."""
    m = BR_N52 / MU0 * 4 / 3 * math.pi * r**3
    return -MU0 * m * m / (2 * math.pi * np.asarray(d) ** 3)


@dataclass
class BondModel:
    """Face bonds of the geared cube. Every face has 4 corner-sphere pairs only on the four plain
    faces' edges and a ring of 4 spheres around its centre; the discs carry the ring only.

    1. Corner spheres sit `setback` further from the +x and -x faces than from the others, so they
       barely act across a disc face (no detent against the joint, E).
    1. Plain bond (two plain faces): 4 corner pairs and 4 ring pairs.
    1. Disc bond (a disc against any face): the 4 ring pairs; twist goes through two sprung steel
       pins in sockets (pins on the disc, sockets on every face), so a disc bond only peels or
       pulls off, it does not slip.
    Peel is about the bond's outer edge (cube edge for a plain bond, disc rim for a disc bond);
    the moment at the face centre M and the tension T combine as M / M_cap + T / T_cap <= 1."""

    L: float
    d_corner: float
    d_ring: float
    ring_radius: float  # m, from the face centre to a ring sphere (on the diagonals)
    disc_radius: float
    cover: float = 0.05e-3
    setback: float = 3.0e-3
    derate: float = DERATE_SPHERE
    ring_count: int = 4
    pin_d: float = 0.4e-3
    pin_depth: float = 0.4e-3
    pin_bearing: float = 40e6  # Pa, steel pin in a resin socket (A)

    def __post_init__(self):
        self.f_corner = sphere_pair_force(self.d_corner, 2 * self.cover, self.derate)
        self.f_ring = sphere_pair_force(self.d_ring, 2 * self.cover, self.derate)

    @property
    def corner_inset(self) -> float:
        return self.d_corner / 2 + self.cover

    def plain(self) -> dict:
        lever_c = self.L - 2 * self.corner_inset - self.setback  # worst direction: peel about an edge parallel to the faces' y or z, far spheres set back
        tension = 4 * self.f_corner + self.ring_count * self.f_ring
        peel = 2 * self.f_corner * lever_c + self.ring_count * self.f_ring * self.L / 2  # far corner pair at lever_c, the hinge pair at 0; ring centred
        # twist: each free-sphere pair resists sideways shift with at most 0.286 of its pull (dipole
        # pull ~ d^-4, sideways component peaks at a shift of d0 / 2), plus friction 0.3 x tension at
        # the mean radius (E); corner spheres sit at the set-back radius on the faces next to a disc
        r_c = math.hypot(self.L / 2 - self.corner_inset, self.L / 2 - self.corner_inset - self.setback)
        twist = 0.286 * (4 * self.f_corner * r_c + self.ring_count * self.f_ring * self.ring_radius) + 0.3 * tension * 0.5 * (r_c + self.ring_radius)
        return {"tension_N": tension, "peel_Nm": peel, "twist_Nm": twist}

    def disc(self) -> dict:
        tension = self.ring_count * self.f_ring
        peel = self.ring_count * self.f_ring * self.disc_radius  # about the disc rim
        twist = 2 * self.pin_bearing * self.pin_d * self.pin_depth * self.ring_radius
        return {"tension_N": tension, "peel_Nm": peel, "twist_Nm": twist}

    def disc_aligned(self) -> dict:
        """A disc bond when the disc sits at a quarter turn (a locked disc, or a joint that stopped
        there): the corner spheres of both cubes line up across the disc's stand-off and add their
        pull, reduced by (d0 / (d0 + stand-off))^4; peel about the disc rim, far corner pair at the
        rim plus the corner radius (E)."""
        d0 = 2 * self.corner_inset
        k = (d0 / (d0 + self.setback)) ** 4
        r_c = math.sqrt(2) * (self.L / 2 - self.corner_inset)
        base = self.disc()
        return {
            "tension_N": base["tension_N"] + 4 * k * self.f_corner,
            "peel_Nm": base["peel_Nm"] + 2 * k * self.f_corner * (self.disc_radius + r_c),
            "twist_Nm": base["twist_Nm"],
        }

    def detent(self, n: int = 181) -> float:
        """Peak torque (N m) that the corner spheres of a plain neighbour and of this cube put on a
        disc turning that neighbour (through the set-back): numerical derivative of the pair
        energy, nearest pairs at the corner radius, axial distance d0 + setback."""
        r = self.d_corner / 2
        rc = math.sqrt(2) * (self.L / 2 - self.corner_inset)
        z = 2 * self.corner_inset + self.setback
        th = np.linspace(0, math.pi / 2, n)
        energy = np.zeros(n)
        for k in range(4):
            for j in range(4):
                a = math.pi / 4 + k * math.pi / 2
                b = math.pi / 4 + j * math.pi / 2 + th
                lateral = 2 * rc * np.abs(np.sin((b - a) / 2))
                energy += free_pair_potential(r, np.hypot(lateral, z))
        torque = -np.gradient(energy, th) * self.derate
        return float(np.max(np.abs(torque)))

    def summary(self) -> dict:
        p, d = self.plain(), self.disc()
        return {
            "corner_sphere_mm": round(self.d_corner * 1e3, 2),
            "ring_sphere_mm": round(self.d_ring * 1e3, 2),
            "corner_pair_N": round(self.f_corner, 4),
            "ring_pair_N": round(self.f_ring, 4),
            "plain_tension_N": round(p["tension_N"], 3),
            "plain_peel_mNm": round(p["peel_Nm"] * 1e3, 3),
            "disc_tension_N": round(d["tension_N"], 3),
            "disc_peel_mNm": round(d["peel_Nm"] * 1e3, 3),
            "disc_twist_mNm": round(d["twist_Nm"] * 1e3, 2),
            "detent_mNm": round(self.detent() * 1e3, 4),
        }


def nominal_bonds(L: float = 8e-3) -> BondModel:
    """Bonds of the chosen cube. Corner spheres as the reaction-wheel cube (0.125 L, concepts.md
    pick 1); a ring of 0.12 L spheres at 0.37 L from every face centre, outside the flywheel rotor
    under that face; the disc is 0.45 L in radius and stands 0.04 L proud when it turns a
    neighbour, so the corner spheres act across it only weakly (`setback` is that stand-off).
    Four ring spheres a face (on the diagonals), so a disc bond is four sphere pairs."""
    return BondModel(L=L, d_corner=0.125 * L, d_ring=0.14 * L, ring_radius=0.37 * L, disc_radius=0.45 * L, setback=0.04 * L)


# ---------------------------------------------------------------------------------------------
# Shape memory alloy ratchet stepper
# ---------------------------------------------------------------------------------------------

# Dynalloy Flexinol table (V: www.dynalloy.com/tech_data_wire.php, read 2026-11-15): diameter m ->
# (resistance ohm/m, pull at 172 MPa N, current for a 1 s contraction A, cooling time s in still
# air for 70 C wire, for 90 C wire). Dynalloy: 2-6 % repeatable strain at <= 172 MPa gives "tens of
# millions" of cycles; beyond it, hundreds to thousands.
FLEXINOL = {
    0.025e-3: (1425.0, 8.9e-3 * G, 0.045, 0.18, 0.15),
    0.038e-3: (890.0, 20e-3 * G, 0.055, 0.24, 0.20),
    0.050e-3: (500.0, 36e-3 * G, 0.085, 0.40, 0.30),
    0.076e-3: (232.0, 80e-3 * G, 0.150, 0.80, 0.70),
    0.100e-3: (126.0, 143e-3 * G, 0.200, 1.10, 0.90),
}
NITINOL_DENSITY = 6450.0  # kg/m3 (V, Dynalloy)
NITINOL_C = 837.0  # J/(kg K), 0.2 cal/(g C) (V, Dynalloy)
NITINOL_LATENT = 24.2e3  # J/kg, 5.78 cal/g (V, Dynalloy)


@dataclass
class SmaStepper:
    """A ratchet stepper on the disc hub, driven by nitinol wire; one per disc.

    1. A ring of `pitch`-spaced square teeth (printed resin) on the disc hub at radius ring_frac L.
    1. A lock pawl, sprung into a tooth gap: square flanks, so it holds both ways with no power
       (no torque can lift it); `lock_teeth` teeth engage.
    1. Drive pawls on flexures, one per direction per phase. A drive wire pulls its pawl into a
       gap and then along by one tooth; a cam on the pawl lever lifts the lock pawl for that part of
       the stroke, and the lock drops into the next gap at the end of it. When the wire cools, a
       flexure returns the pawl clear of the teeth. Reversing is picking the other wire; with
       `phases` = 2 a second pawl per direction, half a tooth out of phase, fires while the first
       wire cools, doubling the step rate.
    1. Each wire is one length of `d_wire` Flexinol looped over pins into `strands` parallel
       strands, wire_frac of the inner side long, crimped at both ends; the pawl lever matches
       the wire's stroke (strain x length) to 1.25 teeth.
    1. Free disc (crank and slider's passive pins): a bistable flexure latch holds the lock pawl
       up; it is flipped by firing both drive wires of a phase together (A).

    Holding torque is the lock teeth's bending strength; driving torque is the wire force through
    the lever and the pawl's sliding efficiency."""

    L: float
    d_wire: float = 0.025e-3
    strands: int = 18  # 18 strands: drive 2.3 mN m at 8.25 mm, 1.5x the largest measured joint load (arm lift 1.53, walker 1.42; 12 strands stalled 14 % of Monte Carlo arms)
    phases: int = 2
    strain: float = 0.03  # working strain (Dynalloy: 2-6 % repeatable)
    wire_frac: float = 0.80
    pitch: float = 0.20e-3  # m, ratchet tooth pitch (A: about 7-10 pixels on a 20-35 um MSLA printer)
    ring_frac: float = 0.33
    face_frac: float = 0.10
    lock_teeth: int = 3
    eta_pawl: float = 0.75  # E: pawl tip sliding on resin with friction 0.2-0.3, lever pins
    pulse: float = 0.04  # s heating pulse
    stroke: float = 0.12  # s for the pawl to travel a tooth under load (E: heating pulse plus the transformation lag)
    cool_scale: float = 1.3  # E: cooling inside the cube against Dynalloy's still-air value
    hot_wire: bool = True  # 90 C Flexinol (faster cooling)
    material: Material = RESIN
    stiffness_lin: float = 2e5  # N/m, lock pawl and tooth path at the ring radius (E)
    clearance: float = 10e-6  # m, residual play at the ring radius: the lock pawl's teeth taper 5 deg (below the friction angle, so still self-locking) and its spring takes up the 30-40 um printed clearance (E)

    @property
    def inner(self):
        return self.L - 0.8e-3

    @property
    def r_ring(self):
        return self.ring_frac * self.L

    @property
    def step(self):
        return self.pitch / self.r_ring

    @property
    def wire_length(self):
        return self.wire_frac * self.inner

    @property
    def wire_force(self):
        return self.strands * FLEXINOL[self.d_wire][1]

    @property
    def lever(self):
        """Pawl travel / wire travel."""
        return 1.25 * self.pitch / (self.strain * self.wire_length)

    @property
    def drive(self):
        return self.eta_pawl * self.wire_force / self.lever * self.r_ring

    def hold(self, peak=False):
        """Lock teeth in bending: tooth height 0.4 p, root thickness 0.5 p, face width b."""
        sigma = self.material.sigma_peak if peak else self.material.sigma_cont
        b, p = self.face_frac * self.L, self.pitch
        force = self.lock_teeth * sigma * b * (0.5 * p) ** 2 / (6 * 0.4 * p)
        return force * self.r_ring

    @property
    def cool_time(self):
        return FLEXINOL[self.d_wire][4 if self.hot_wire else 3] * self.cool_scale

    @property
    def cycle(self):
        return self.stroke + self.cool_time

    @property
    def step_period(self):
        return self.cycle / self.phases

    @property
    def speed(self):
        return self.step / self.step_period

    @property
    def wire_mass(self):
        return self.strands * self.wire_length * math.pi * self.d_wire**2 / 4 * NITINOL_DENSITY

    @property
    def step_energy(self):
        """Heat to take the wire from 25 C through its transformation (75 K and the latent heat),
        plus 25 % lost to the air and the crimps during a 40 ms pulse (E)."""
        return 1.25 * self.wire_mass * (NITINOL_C * 75.0 + NITINOL_LATENT)

    @property
    def efficiency(self):
        return self.drive * self.step / self.step_energy

    def electrical(self):
        r_strand = FLEXINOL[self.d_wire][0] * self.wire_length
        power = self.step_energy / self.pulse
        best = None
        for groups in range(1, self.strands + 1):  # strands split into series groups
            if self.strands % groups:
                continue
            per = self.strands // groups  # parallel strands per group
            r = groups * r_strand / per
            v = math.sqrt(power * r)
            if v <= 3.7 and (best is None or v > best[1]):
                best = (r, v, v / r, groups)
        if best is None:
            r = r_strand / self.strands
            best = (r, math.sqrt(power * r), math.sqrt(power / r), 1)
        return {"resistance_ohm": round(best[0], 2), "volts": round(best[1], 2), "amps": round(best[2], 3), "series_groups": best[3], "pulse_power_W": round(power, 3)}

    def table(self) -> dict:
        return {
            "wire": f"Flexinol {'90' if self.hot_wire else '70'} C, {self.d_wire * 1e3:.3f} mm x {self.strands} strands, {self.wire_length * 1e3:.2f} mm long",
            "wire_pull_N": round(self.wire_force, 3),
            "wire_stress_MPa": 172,
            "strain": self.strain,
            "lever_pawl_over_wire": round(self.lever, 2),
            "ring_radius_mm": round(self.r_ring * 1e3, 2),
            "teeth": int(round(2 * math.pi * self.r_ring / self.pitch)),
            "step_deg": round(math.degrees(self.step), 2),
            "drive_mNm": round(self.drive * 1e3, 3),
            "hold_cont_mNm": round(self.hold(False) * 1e3, 3),
            "hold_peak_mNm": round(self.hold(True) * 1e3, 3),
            "cooling_s": round(self.cool_time, 3),
            "step_period_s": round(self.step_period, 3),
            "speed_deg_s": round(math.degrees(self.speed), 1),
            "step_energy_mJ": round(self.step_energy * 1e3, 2),
            "energy_per_deg_mJ": round(self.step_energy / math.degrees(self.step) * 1e3, 2),
            "efficiency": round(self.efficiency, 4),
            "electrical": self.electrical(),
            "wires_per_joint": 2 * self.phases,
            "crimps_per_joint": 4 * self.phases,
            "backlash_mrad": round(self.clearance / self.r_ring * 1e3, 1),
            "fatigue": "<= 172 MPa and 3 % strain: Dynalloy rates tens of millions of cycles (V); a walker cycle is about 75 steps a leg",
        }


def sma_design(L: float, **kw) -> SmaStepper:
    return SmaStepper(L=L, **kw)


# ---------------------------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------------------------


@dataclass
class JointParams:
    """Per-joint numbers for GearedRig (SI, output side unless named)."""

    ratio: float
    eta_f: float
    eta_b: float  # negative: self-locking
    rotor_inertia: float  # kg m2 at the motor
    k_t: float
    resistance: float
    volts: float
    i_max: float
    omega_max: float  # rad/s at the motor
    stiffness: float  # N m/rad
    backlash: float  # rad, full width
    tooth_peak: float  # N m: output torque that breaks the weakest tooth
    tooth_cont: float
    drag_free: float = 0.0  # N m: drag of a free (selector neutral) disc
    detent: float = 0.0  # N m peak, period 90 deg
    kind: str = "gear"  # 'gear' or 'stepper'
    step: float = 0.0  # rad, stepper: one ratchet tooth
    stroke_time: float = 0.0  # s, stepper: heating pulse during which the ring moves one tooth
    step_period: float = 0.0  # s, stepper: shortest time between step starts in one direction
    drive: float = 0.0  # N m, stepper: torque the wire can push the ring with
    step_energy: float = 0.0  # J per step attempt

    @staticmethod
    def from_design(design: JointDesign, bonds: BondModel | None = None, mu: float | None = None, eta_spur: float | None = None, backlash_scale: float = 1.0) -> "JointParams":
        tr = design.train
        cap_c = tr.capacity(False)[0]
        cap_p = tr.capacity(True)[0]
        return JointParams(
            ratio=tr.ratio,
            eta_f=tr.efficiency(mu, eta_spur),
            eta_b=tr.back_efficiency(mu, eta_spur),
            rotor_inertia=design.motor.inertia,
            k_t=design.motor.k_t,
            resistance=design.motor.resistance,
            volts=design.motor.volts,
            i_max=min(design.motor.i_max, cap_p / (tr.ratio * tr.efficiency(mu, eta_spur) * design.motor.k_t)),
            omega_max=design.motor.omega_max,
            stiffness=tr.stiffness_out(),
            backlash=tr.backlash_out() * backlash_scale,
            tooth_peak=cap_p,
            tooth_cont=cap_c,
            drag_free=0.02 * cap_c,
            detent=bonds.detent() if bonds is not None else 0.0,
        )


def stepper_params(st: SmaStepper, bonds: BondModel | None = None, force_scale: float = 1.0, cool_scale: float = 1.0, eta_pawl: float | None = None, backlash_scale: float = 1.0) -> JointParams:
    """JointParams for an SMA ratchet stepper, with Monte Carlo scales on the wire force, the
    cooling time, the pawl efficiency and the tooth clearance."""
    st = replace(st, cool_scale=st.cool_scale * cool_scale, eta_pawl=st.eta_pawl if eta_pawl is None else eta_pawl)
    return JointParams(
        ratio=1.0, eta_f=1.0, eta_b=-1.0, rotor_inertia=0.0, k_t=1.0, resistance=1.0, volts=0.0, i_max=0.0, omega_max=0.0,
        stiffness=st.stiffness_lin * st.r_ring**2,
        backlash=st.clearance / st.r_ring * backlash_scale,
        tooth_peak=st.hold(True),
        tooth_cont=st.hold(False),
        drag_free=0.02 * st.hold(False),
        detent=bonds.detent() if bonds is not None else 0.0,
        kind="stepper",
        step=st.step,
        stroke_time=st.stroke,
        step_period=st.step_period,
        drive=st.drive * force_scale,
        step_energy=st.step_energy,
    )


# ---------------------------------------------------------------------------------------------
# MuJoCo: cubes of side L with geared discs on +x and -x
# ---------------------------------------------------------------------------------------------


@dataclass
class Cube:
    """One cube, as module_b.Module: a root has parent None, `pos` and `axes` (world directions of
    its local x and y) and `mount` (None: free; 'b', 'p' or 'n' bolted); a child has a parent index,
    the parent's `face`, its own `mate` face and a quarter `turn`."""

    parent: int | None = None
    face: str = "+z"
    mate: str = "-z"
    turn: int = 0
    pos: tuple = (0.0, 0.0, 0.0)
    axes: tuple = ("+x", "+y")
    mount: str | None = None
    rgba: str | None = None
    play: str = ""  # discs ('p', 'n') given axial play AXIAL_PLAY on their bearing (a free pin in a closed linkage)


def lattice(cells: list, pitch: float) -> list:
    """Cubes from lattice cells (module_b.lattice with the pitch as a parameter). Each cell: `at`
    (i, j, k) in pitches, `axis` (world direction of the disc axis, local x), optional `up`,
    `parent`; for a root `z0` (m, centre height of the lattice origin) and `mount`; optional `play`
    (discs with axial play)."""
    cubes, world = [], []
    for cell in cells:
        axis = unit(cell["axis"])
        up = unit(cell["up"]) if "up" in cell else next(unit(f) for f in ("+y", "+z", "+x") if abs(unit(f) @ axis) < 0.5)
        rot = np.column_stack([axis, up, np.cross(axis, up)])
        at = np.asarray(cell["at"], float)
        if cell.get("parent") is None:
            z0 = cell.get("z0", pitch / 2)
            cubes.append(Cube(pos=tuple(at * pitch + np.array([0, 0, z0])), axes=(face_of(rot[:, 0]), face_of(rot[:, 1])), mount=cell.get("mount"), rgba=cell.get("rgba"), play=cell.get("play", "")))
        else:
            p = cell["parent"]
            p_at, p_rot = world[p]
            step = at - p_at
            face = face_of(p_rot.T @ step)
            relative = np.round(p_rot.T @ rot)
            mate = face_of(rot.T @ -step)
            turn = next((k for k in range(4) if np.allclose(bond_rotation(face, mate, k), relative)), None)
            if turn is None:
                raise ValueError(f"cell {len(cubes)} cannot bond to {p}")
            cubes.append(Cube(parent=p, face=face, mate=mate, turn=turn, rgba=cell.get("rgba"), play=cell.get("play", "")))
        world.append((at, rot))
    return cubes


def poses(cubes: list, pitch: float):
    rots, centres = [], []
    for c in cubes:
        if c.parent is None:
            rots.append(axes_rotation(c.axes))
            centres.append(np.asarray(c.pos, float))
        else:
            r_p = rots[c.parent]
            rots.append(r_p @ bond_rotation(c.face, c.mate, c.turn))
            centres.append(centres[c.parent] + r_p @ unit(c.face) * pitch)
    return rots, centres


RETRY_S = 1.0  # s a stalled stepper waits before trying the same direction again
AXIAL_PLAY = 0.05e-3  # m, axial play of a disc on its bearing where a linkage asks for it (A: printed hub, 0.02-0.1 mm typical)

COLOURS = ["0.85 0.55 0.25 1", "0.35 0.55 0.80 1", "0.45 0.70 0.45 1", "0.80 0.45 0.55 1", "0.65 0.55 0.80 1", "0.80 0.75 0.40 1"]


@dataclass
class CubeSpec:
    """Geometry and mass of the simulated cube."""

    L: float
    mass: float  # kg, whole cube
    disc_mass_frac: float = 0.03  # each disc's share (E: printed disc, worm wheel, pins, ring spheres)
    disc_radius_frac: float = 0.40
    disc_thick_frac: float = 0.06
    body_friction: float = 0.6
    floor_friction: float = 0.8
    # s, contact time constant against the table: Hertz contact of a printed cube corner (radius about
    # 0.2 mm, resin E* about 1.5 GPa) under 10-50 mN is about 5e4 N/m, a period of about 1.5 ms for a
    # few grams (E). Stiffer settings make a foot rolling over a cube edge chatter from step to step
    # (contacts switching on and off each time step), with 5 ms bond loads up to 1.3x the smooth value.
    floor_timeconst: float = 1.5e-3

    gap_frac: float = 0.048  # lattice pitch 1.048 L, as layout B's 52.4 mm for 50 mm cubes (module_b.PITCH)

    @property
    def half(self):
        return self.L / 2

    @property
    def pitch(self):
        return self.L * (1 + self.gap_frac)

    @property
    def disc_radius(self):
        return self.disc_radius_frac * self.L

    @property
    def disc_thick(self):
        return self.disc_thick_frac * self.L

    @property
    def disc_mass(self):
        return self.disc_mass_frac * self.mass

    @property
    def body_mass(self):
        return self.mass - 2 * self.disc_mass


def build(cubes: list, spec: CubeSpec, scene: str = "", extra: str = "", timestep: float = 1e-4, armature: float = 1e-7) -> str:
    """MJCF of the cubes on a floor. Joints j<i>p / j<i>n (disc on body, +x / -x), each with a
    motor actuator m<i>p / m<i>n of unlimited range: GearedRig writes the gear train's output torque
    into it. Bonded neighbours do not collide with each other."""
    L, h = spec.L, spec.half
    children = {}
    for k, c in enumerate(cubes):
        if c.parent is not None:
            if c.parent >= k:
                raise ValueError("a parent must come before its children")
            key = (c.parent, part_of(c.face))
            if any(cubes[o].face == c.face for o in children.get(key, [])):
                raise ValueError(f"face {c.face} of cube {c.parent} bonded twice")
            children.setdefault(key, []).append(k)
    rd, td = spec.disc_radius, spec.disc_thick
    bearing = 0.3 * 3 * spec.mass * G * rd  # disc rim bearing: friction 0.3 under about three cubes' weight (E)
    fr = spec.body_friction
    solref = f"{max(4 * timestep, 4e-4):.6g} 1"

    def geoms(i, part, rgba):
        if part == "b":
            return f'<geom name="c{i}b" type="box" size="{h:.6g} {h:.6g} {h:.6g}" mass="{spec.body_mass:.6g}" rgba="{rgba}" friction="{fr} 0.005 0.0001" solref="{solref}"/>\n'
        sign = 1 if part == "p" else -1
        x = sign * (h - td / 2)
        seen = sign * (h + 0.02 * L - td / 2)
        mark = sign * (h + 0.025 * L)
        plain = 'contype="0" conaffinity="0" mass="0"'
        return (
            f'<geom name="c{i}{part}" type="cylinder" size="{rd:.6g} {td / 2:.6g}" pos="{x:.6g} 0 0" euler="0 90 0" mass="{spec.disc_mass:.6g}" rgba="0 0 0 0" contype="0" conaffinity="0"/>\n'
            f'<geom type="cylinder" size="{rd:.6g} {td / 2:.6g}" pos="{seen:.6g} 0 0" euler="0 90 0" rgba="0.25 0.25 0.28 1" {plain}/>\n'
            f'<geom type="box" size="{0.01 * L:.6g} {0.06 * L:.6g} {0.42 * rd:.6g}" pos="{mark:.6g} 0 {0.5 * rd:.6g}" rgba="0.95 0.95 0.95 1" {plain}/>\n'
        )

    def joint(i, part, inverted):
        axis = "-1 0 0" if inverted else "1 0 0"
        out = f'<joint name="j{i}{part}" type="hinge" axis="{axis}" armature="{armature:.6g}" frictionloss="{bearing:.6g}"/>\n'
        if part in cubes[i].play:  # play of the disc on its bearing (A): axial play and the tilt it allows across the disc, so a closed linkage resting on the table is not forced against its own bonds
            tilt = math.degrees(AXIAL_PLAY / rd)
            out += f'<joint name="s{i}{part}" type="slide" axis="{axis}" limited="true" range="{-AXIAL_PLAY:.6g} {AXIAL_PLAY:.6g}" armature="{armature:.6g}" damping="{0.05 * spec.mass:.6g}"/>\n'
            for k, ax in (("y", "0 1 0"), ("z", "0 0 1")):
                out += f'<joint name="t{i}{part}{k}" type="hinge" axis="{ax}" limited="true" range="{-tilt:.6g} {tilt:.6g}" armature="{armature:.6g}" damping="{0.05 * spec.mass * L * L:.6g}"/>\n'
        return out

    def body(i, pos, rot, root_joint=""):
        c = cubes[i]
        rgba = c.rgba or COLOURS[i % len(COLOURS)]
        anchor = (c.mount or "b") if c.parent is None else part_of(c.mate)

        def part_xml(part, jnt, head):
            inner = geoms(i, part, rgba)
            for k in children.get((i, part), []):
                ck = cubes[k]
                inner += body(k, unit(ck.face) * spec.pitch, bond_rotation(ck.face, ck.mate, ck.turn))
            if part == anchor and anchor != "b":
                inner += part_xml("b", joint(i, anchor, inverted=True), "")
            elif part == "b":
                for other in [x for x in "pn" if x != anchor]:
                    inner += part_xml(other, joint(i, other, inverted=False), "")
            return f'<body name="c{i}{part}" {head}>\n{jnt}{inner}</body>\n'

        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.asarray(rot, float).reshape(-1))
        head = f'pos="{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}" quat="{" ".join(f"{v:.6g}" for v in q)}"'
        return part_xml(anchor, root_joint, head)

    roots = ""
    for i, c in enumerate(cubes):
        if c.parent is None:
            roots += body(i, c.pos, axes_rotation(c.axes), "<freejoint/>\n" if c.mount is None else "")
    rots, centres = poses(cubes, spec.pitch)
    excluded = {(f"c{a}b", f"c{a}{p}") for a in range(len(cubes)) for p in "pn"}
    for a in range(len(cubes)):
        for b in range(a + 1, len(cubes)):
            if np.linalg.norm(centres[a] - centres[b]) < spec.pitch * 1.01:
                for pa in "bpn":
                    for pb in "bpn":
                        excluded.add((f"c{a}{pa}", f"c{b}{pb}"))
    excludes = "".join(f'<exclude body1="{x}" body2="{y}"/>\n' for x, y in sorted(excluded))
    motors = "".join(f'<motor name="m{i}{p}" joint="j{i}{p}"/>\n' for i in range(len(cubes)) for p in "pn")
    span = 60 * L
    return f"""<mujoco model="geared_cubes">
<compiler angle="degree" autolimits="true"/>
<option timestep="{timestep}" integrator="implicitfast" cone="elliptic" impratio="10"/>
<size memory="64M"/>
<visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/><map znear="0.001"/></visual>
<default><geom solref="{solref}" solimp="0.95 0.99 0.0002"/></default>
<asset>
<texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.82 0.80" rgb2="0.74 0.74 0.72" width="512" height="512"/>
<material name="grid" texture="grid" texrepeat="{span / (2 * L):.6g} {span / (2 * L):.6g}" reflectance="0.05"/>
<texture name="sky" type="skybox" builtin="gradient" rgb1="0.92 0.93 0.95" rgb2="0.70 0.74 0.80" width="256" height="256"/>
</asset>
<worldbody>
<light pos="{6 * L} {-12 * L} {24 * L}" dir="-0.2 0.4 -1" diffuse="0.6 0.6 0.6" castshadow="true"/>
<geom name="floor" type="plane" size="{span:.6g} {span:.6g} {0.05 * span:.6g}" material="grid" friction="{spec.floor_friction} 0.005 0.0001" priority="1" solref="{spec.floor_timeconst:.6g} 1"/>
{scene}
{roots}</worldbody>
<contact>
{excludes}</contact>
<actuator>
{motors}</actuator>
{extra}
</mujoco>
"""


class GearedRig:
    """Cubes with geared discs, the joints' motors and trains integrated here, bond loads judged.

    Per joint, the controller reads the disc angle q and the train speed, asks for an output torque
    tau* = kp (q_ref - q) + kd (qd_ref - w), turns it into a motor current through the ratio and the
    forward efficiency, clips the voltage and current, and the train dynamics decide what moves:

        J phi'' = N k_t i - Lambda,   Lambda = tau_c / eta_f (driving) or tau_c eta_b (back-driven)
        tau_c   = k (deadzone(phi - q, b/2)) + c (phi' - q')   on the disc

    with J = I_rotor N^2 and phi the train output angle. With eta_b <= 0 a load cannot move the
    train (self-locking); a stationary train starts only when the motor overcomes the friction."""

    def __init__(self, cubes, spec: CubeSpec, joint: JointParams, bonds: BondModel, scene="", extra="", timestep=1e-4, every=5, smooth=0.005, bandwidth=12.0, bond_scale=1.0, settle=0.25):
        self.cubes, self.spec, self.joint, self.bonds = cubes, spec, joint, bonds
        jp = joint
        # numerical armature: keeps the coupling spring stable on a disc with nothing on it
        armature = max(1e-9, jp.stiffness * (timestep / 0.3) ** 2)
        self.xml = build(cubes, spec, scene, extra, timestep, armature)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.names = [m.actuator(a).name[1:] for a in range(m.nu)]
        self.index = {n: a for a, n in enumerate(self.names)}
        self.qadr = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        self.vadr = np.array([m.jnt_dofadr[m.actuator_trnid[a, 0]] for a in range(m.nu)])
        n = m.nu
        self.J = max(jp.rotor_inertia * jp.ratio**2, 1e-12)
        w_c = bandwidth
        self.kp = np.full(n, self.J * w_c**2)
        self.kd = np.full(n, 2 * self.J * w_c)
        mujoco.mj_forward(m, self.data)
        self.phi = self.data.qpos[self.qadr].copy()
        self.w = np.zeros(n)
        self.free = np.zeros(n, bool)
        self.cap_i = np.full(n, jp.i_max)  # per-joint current limit (gripping)
        self.grip = np.zeros(n, bool)  # a joint closing on an object: once stalled it stays stalled (no retries)
        self.drive_lim = np.full(n, np.inf)  # stepper: per-joint wire force limit as a torque (a grip at set squeeze: a regulated, partial heating pulse, E)
        load = spec.mass * spec.L**2 * 10
        self.k = np.full(n, jp.stiffness)  # per joint: set_compliance() softens a gripper finger
        self.c_t = np.full(n, 2 * 0.3 * math.sqrt(jp.stiffness * load))
        self.q0 = self.phi.copy()
        self.q1 = self.phi.copy()
        self.start = np.zeros(n)
        self.span = np.ones(n)
        self.rate = np.zeros(n)
        self.rate0 = np.zeros(n)
        self.ramp = np.zeros(n)
        self.paths = {}
        self.every = every
        self.steps = 0
        self.peak = np.zeros(n)  # largest |transmitted torque| per joint
        self.peak_smooth = np.zeros(n)
        self._tau_hist = []
        self.energy = 0.0  # J drawn from the supply by all motors
        self.copper = 0.0
        self.motor_time = 0.0
        self.broken_joint = None  # (time, joint) of the first tooth overload
        self._bond_setup()
        self.bond_scale = bond_scale
        self.series = []  # per sample: utilisation per bond
        self.loads_series = []
        self.smooth_n = max(1, int(round(smooth / (every * timestep))))
        self.first_bond_failure = None
        # assembly under load: gravity ramps in over `settle` s; loads are judged after it
        self.settle = settle
        self.gravity = self.model.opt.gravity.copy()
        self.model.opt.gravity[:] = 0.0
        # a self-locking joint inside its tolerance draws no current (hold), with hysteresis
        self.hold_tol = 0.6 * jp.backlash + 2e-3
        self.holding = np.zeros(n, bool)
        self.step_state = {"back": np.zeros(n, bool), "active": np.zeros(n, bool), "dir": np.zeros(n, int), "t0": np.zeros(n), "from": np.zeros(n), "fails": np.zeros(n, int), "fail_t": np.full(n, -1e9), "fail_dir": np.zeros(n, int), "ready": np.zeros((n, 2))}
        self.step_count = 0
        self.failed_steps = 0

    def set_compliance(self, name, stiffness):
        """A compliant element between the joint and its load (a finger flexure), N m/rad."""
        a = self.index[name]
        load = self.spec.mass * self.spec.L**2 * 10
        self.k[a] = stiffness
        self.c_t[a] = 2 * 0.3 * math.sqrt(stiffness * load)

    # ------------------------------------------------------------------ references
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

    def command(self, targets=None, seconds=0.5, speeds=None, free=(), ramp=0.3, currents=None):
        """targets {joint: rad} on minimum-jerk paths; speeds {joint: rad/s}; free: selector to
        neutral; currents {joint: A}: a current limit (a grip at set torque)."""
        for name, cur in (currents or {}).items():
            self.cap_i[self.index[name]] = cur
            if self.joint.kind == "stepper":  # k_t = ratio = 1: the 'current' is the torque limit; 0 restores the full wire
                self.drive_lim[self.index[name]] = cur if cur > 0 else np.inf
                self.grip[self.index[name]] = cur > 0
        t = self.data.time
        q_now, qd_now = self._reference(t)
        q_meas = self.data.qpos[self.qadr]
        for a in range(self.model.nu):
            if self.free[a] or a in self.paths or abs(q_now[a] - q_meas[a]) > 0.05:
                q_now[a], qd_now[a] = q_meas[a], 0.0
        for name, angle in (targets or {}).items():
            a = self.index[name]
            self.paths.pop(a, None)
            if self.free[a]:
                self.phi[a], self.w[a] = q_meas[a], 0.0
            self.q0[a], self.q1[a], self.start[a], self.span[a], self.free[a] = q_now[a], angle, t, max(seconds, 1e-6), False
            self.rate[a] = self.rate0[a] = self.ramp[a] = 0.0
            self.step_state["fails"][a] = 0
        for name, rate in (speeds or {}).items():
            a = self.index[name]
            self.paths.pop(a, None)
            self.q0[a] = self.q1[a] = q_now[a]
            self.rate0[a] = 0.0 if self.free[a] else qd_now[a]
            self.start[a], self.span[a], self.rate[a], self.ramp[a], self.free[a] = t, 1.0, rate, ramp, False
        for name in free:
            self.free[self.index[name]] = True

    def follow(self, name, path):
        a = self.index[name]
        if path is None:
            self.paths.pop(a, None)
            q = self.data.qpos[self.qadr[a]]
            self.q0[a], self.q1[a], self.start[a], self.span[a] = q, q, self.data.time, 1.0
            self.rate[a] = self.rate0[a] = self.ramp[a] = 0.0
        else:
            self.paths[a] = path
            self.free[a] = False

    # ------------------------------------------------------------------ joints
    def _stepper(self):
        """SMA ratchet stepper. The lock angle phi moves only in whole teeth: a step starts when the
        reference is more than 0.55 teeth away and the wire for that direction is ready; during the
        heating pulse phi moves one tooth unless the coupling torque against the step exceeds the
        wire's drive torque (the pawl then advances only as far as the coupling spring reaches
        that torque); at the end of the pulse the lock drops into the next
        gap if phi got past 60 % of a tooth, else back into the old one (a failed step). Two failed
        steps in a row in one direction stop further tries for RETRY_S (or until a new command)."""
        jp, d = self.joint, self.data
        dt = self.model.opt.timestep
        t = d.time
        q, qd = d.qpos[self.qadr], d.qvel[self.vadr]
        q_ref, _ = self._reference(t)
        for a, path in self.paths.items():
            q_ref[a], _ = path(t)
        x = self.phi - q
        half = jp.backlash / 2
        xb = np.where(x > half, x - half, np.where(x < -half, x + half, 0.0))
        engaged = xb != 0.0
        tau_c = self.k * xb + self.c_t * (self.w - qd) * engaged
        tau_c = np.where(self.free, 0.0, tau_c)
        st = self.step_state
        free = self.free
        self.phi = np.where(free, q, self.phi)
        self.w = np.where(free, 0.0, self.w)
        st["active"] &= ~free
        st["back"] &= ~free
        vmax = 2.0 * jp.step / jp.stroke_time
        sgn = st["dir"]
        # failed strokes: the wire cools and the load returns the ring to its old gap
        back = st["active"] & st["back"]
        if back.any():
            move = np.clip(st["from"] - self.phi, -vmax * dt, vmax * dt)
            self.phi = np.where(back, self.phi + move, self.phi)
            self.w = np.where(back, move / dt, self.w)
            home = back & (np.abs(self.phi - st["from"]) < 1e-9)
            self.w[home] = 0.0
            st["active"][home] = False
            st["back"][home] = False
        # strokes under way: minimum-jerk, stopped while the load pushes back harder than the wire
        stroke = st["active"] & ~st["back"]
        if stroke.any():
            u = np.clip((t - st["t0"]) / jp.stroke_time, 0.0, 1.0)
            end = st["from"] + sgn * jp.step
            target = st["from"] + sgn * jp.step * (10 * u**3 - 15 * u**4 + 6 * u**5)
            # the wire's pull saturates at the drive torque (its stress plateau): the pawl advances only
            # as far as the coupling spring reaches that torque, so a stalled stroke creeps with the
            # load instead of switching on and off from one time step to the next
            drive = np.minimum(jp.drive, self.drive_lim)
            reach = q + sgn * (half + drive / self.k)
            move = np.clip(target - self.phi, -vmax * dt, vmax * dt)
            move = np.where(sgn > 0, np.minimum(move, np.maximum(reach - self.phi, 0.0)), np.maximum(move, np.minimum(reach - self.phi, 0.0)))
            self.phi = np.where(stroke, self.phi + move, self.phi)
            self.w = np.where(stroke, move / dt, self.w)
            done = stroke & (u >= 1.0) & (np.abs(self.phi - end) < 1e-6 * jp.step)
            self.phi = np.where(done, end, self.phi)
            self.w[done] = 0.0
            st["active"][done] = False
            st["fails"][done] = 0
            late = stroke & ~done & (t - st["t0"] >= 1.5 * jp.stroke_time)
            if late.any():  # the lock cannot drop into the next gap: a failed step
                st["back"][late] = True
                st["fails"][late] += 1
                st["fail_t"][late] = t
                self.failed_steps += int(late.sum())
        # new steps
        idle = ~st["active"] & ~free
        e = q_ref - self.phi
        want = np.where(e > 0, 1, -1)
        col = np.where(want > 0, 0, 1)
        ready = t >= st["ready"][np.arange(len(q)), col]
        stuck = (st["fails"] >= 2) & (st["fail_dir"] == want) & ((t - st["fail_t"] < RETRY_S) | self.grip)  # a stalled joint (a grip) rests, then the controller tries again
        go = idle & (np.abs(e) > 0.55 * jp.step) & ready & ~stuck
        if go.any():
            reset = go & (st["fails"] > 0) & (st["fail_dir"] != want)
            st["fails"][reset] = 0
            st["active"][go] = True
            st["dir"][go] = want[go]
            st["t0"][go] = t
            st["from"][go] = self.phi[go]
            st["fail_dir"][go] = want[go]
            idx = np.where(go)[0]
            st["ready"][idx, col[go]] = t + jp.step_period
            n_go = int(go.sum())
            self.energy += jp.step_energy * n_go
            self.step_count += n_go
        drag = jp.drag_free * np.tanh(qd / 0.05)
        detent = -jp.detent * np.sin(4 * q)
        self.data.ctrl[:] = np.where(self.free, -drag, tau_c) + detent
        mag = np.abs(np.where(self.free, 0.0, tau_c))
        self.peak = np.maximum(self.peak, mag)
        self._tau_hist.append(mag)
        if len(self._tau_hist) > self.smooth_n * self.every:
            self._tau_hist.pop(0)
        if self.steps % self.every == 0 and self._tau_hist and t > self.settle:
            avg = np.mean(self._tau_hist, axis=0)
            self.peak_smooth = np.maximum(self.peak_smooth, avg)
            over = np.where(avg > jp.tooth_peak)[0]
            if len(over) and self.broken_joint is None:
                self.broken_joint = (round(float(t), 3), self.names[over[0]])

    def _joints(self):
        jp, d = self.joint, self.data
        if jp.kind == "stepper":
            return self._stepper()
        dt = self.model.opt.timestep
        q, qd = d.qpos[self.qadr], d.qvel[self.vadr]
        q_ref, qd_ref = self._reference(d.time)
        for a, path in self.paths.items():
            q_ref[a], qd_ref[a] = path(d.time)
        n_ratio = jp.ratio
        # controller -> motor current -> motor torque (output-referred)
        err = q_ref - q
        still = np.abs(qd_ref) < 1e-3
        if jp.eta_b <= 0:
            self.holding = still & np.where(self.holding, np.abs(err) < 2 * self.hold_tol, np.abs(err) < self.hold_tol)
        tau_star = self.kp * err + self.kd * (qd_ref - self.w)
        eta_ff = np.where(tau_star * self.w >= 0, jp.eta_f, 1.0)
        i_cmd = tau_star / (n_ratio * jp.k_t * eta_ff)
        omega_m = self.w * n_ratio
        volts = np.clip(jp.resistance * i_cmd + jp.k_t * omega_m, -jp.volts, jp.volts)
        current = np.clip((volts - jp.k_t * omega_m) / jp.resistance, -self.cap_i, self.cap_i)
        current = np.where(self.free | self.holding, 0.0, current)
        tau_m = n_ratio * jp.k_t * current
        # coupling: backlash dead zone, stiffness, damping when engaged
        x = self.phi - q
        half = jp.backlash / 2
        xb = np.where(x > half, x - half, np.where(x < -half, x + half, 0.0))
        engaged = xb != 0.0
        tau_c = self.k * xb + self.c_t * (self.w - qd) * engaged
        tau_c = np.where(self.free, 0.0, tau_c)
        # train dynamics with direction-dependent efficiency
        moving = np.abs(self.w) > 1e-6
        s = np.where(moving, np.sign(self.w), np.sign(tau_m - tau_c))
        driving = s * tau_c > 0
        lam = np.where(driving, tau_c / jp.eta_f, tau_c * jp.eta_b)
        net = tau_m - lam
        w_new = self.w + dt * net / self.J
        flipped = moving & (np.sign(w_new) != np.sign(self.w))
        start_ok = (~moving) & (s * net > 0)
        w_new = np.where(moving, np.where(flipped, 0.0, w_new), np.where(start_ok, w_new, 0.0))
        w_max = jp.omega_max / n_ratio
        w_new = np.clip(w_new, -w_max, w_max)
        w_new = np.where(self.free, qd, w_new)
        self.w = w_new
        self.phi = np.where(self.free, q, self.phi + dt * self.w)
        # disc torque: coupling, or drag when free; corner detent through the set-back
        drag = jp.drag_free * np.tanh(qd / 0.05)
        detent = -jp.detent * np.sin(4 * q)
        ctrl = np.where(self.free, -drag, tau_c) + detent
        self.data.ctrl[:] = ctrl
        # bookkeeping
        power = volts * current
        self.energy += float(np.sum(np.maximum(power, 0.0))) * dt
        self.copper += float(np.sum(current**2) * jp.resistance) * dt
        self.motor_time += float(np.sum(np.abs(current) > 1e-4)) * dt
        mag = np.abs(np.where(self.free, 0.0, tau_c))
        self.peak = np.maximum(self.peak, mag)
        self._tau_hist.append(mag)
        if len(self._tau_hist) > self.smooth_n * self.every:
            self._tau_hist.pop(0)
        if self.steps % self.every == 0 and self._tau_hist and d.time > self.settle:
            avg = np.mean(self._tau_hist, axis=0)
            self.peak_smooth = np.maximum(self.peak_smooth, avg)
            over = np.where(avg > jp.tooth_peak)[0]
            if len(over) and self.broken_joint is None:
                self.broken_joint = (round(float(d.time), 3), self.names[over[0]])

    def step(self):
        t = self.data.time
        if t <= self.settle:
            self.model.opt.gravity[:] = self.gravity * min(1.0, t / (0.8 * self.settle))
        self._joints()
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        if self.steps % self.every == 0:
            self._bond_sample()

    def run(self, seconds, watch=None):
        end = self.data.time + seconds
        while self.data.time < end - 1e-12:
            before = self.data.time
            self.step()
            if self.data.time < before:
                raise RuntimeError(f"MuJoCo reset the simulation at t = {before:.4f} s (unstable)")
            if watch:
                watch(self)

    # ------------------------------------------------------------------ bonds
    def _bond_setup(self):
        m = self.model
        self.bond_bodies, self.bond_normals, self.bond_kind, self.bond_names, self.bond_joint = [], [], [], [], []
        for i, c in enumerate(self.cubes):
            if c.parent is None:
                continue
            anchor = part_of(c.mate)
            b = m.body(f"c{i}{anchor}").id
            if m.body_weldid[b] == 0:
                continue
            disc = anchor != "b" or part_of(c.face) != "b"
            if disc:  # the joint whose disc this bond sits on
                self.bond_joint.append(self.index[f"{i}{anchor}"] if anchor != "b" else self.index[f"{c.parent}{part_of(c.face)}"])
            else:
                self.bond_joint.append(-1)
            self.bond_bodies.append(b)
            self.bond_normals.append(unit(c.mate))
            self.bond_kind.append("disc" if disc else "plain")
            self.bond_names.append(f"{c.parent}{c.face}-{i}{c.mate}")
        caps_p, caps_d = self.bonds.plain(), self.bonds.disc()
        self.cap_t = np.array([(caps_d if k == "disc" else caps_p)["tension_N"] for k in self.bond_kind])
        self.cap_m = np.array([(caps_d if k == "disc" else caps_p)["peel_Nm"] for k in self.bond_kind])
        self.cap_tw = np.array([(caps_d if k == "disc" else caps_p)["twist_Nm"] for k in self.bond_kind])
        caps_a = self.bonds.disc_aligned()
        self.cap_t_al = np.array([(caps_a if k == "disc" else caps_p)["tension_N"] for k in self.bond_kind])
        self.cap_m_al = np.array([(caps_a if k == "disc" else caps_p)["peel_Nm"] for k in self.bond_kind])
        self.bond_joint = np.array(self.bond_joint, int)
        self._slip_setup()

    def _group_inertia(self, bodies) -> tuple:
        """Mass and inertia tensor (world axes, about their joint COM) of a set of bodies."""
        m, d = self.model, self.data
        mass = float(sum(m.body_mass[j] for j in bodies))
        inertia = np.zeros((3, 3))
        if mass <= 0:
            return 0.0, inertia
        com = sum(m.body_mass[j] * d.xipos[j] for j in bodies) / mass
        for j in bodies:
            rot = d.ximat[j].reshape(3, 3)
            inertia += rot @ np.diag(m.body_inertia[j]) @ rot.T
            r = d.xipos[j] - com
            inertia += m.body_mass[j] * (r @ r * np.eye(3) - np.outer(r, r))
        return mass, inertia

    def _slip_inertia(self):
        """Per bond, the inertia tensors of its two sides (about each side's COM; a mounted or
        missing side is infinite). Refreshed every 20 samples as the machine moves."""
        self.slip_tensors = []
        for sub, rest, mounted in self.slip_groups:
            _, i_s = self._group_inertia(sub)
            i_r = None if mounted else self._group_inertia(rest)[1]
            self.slip_tensors.append((i_s, i_r))

    def _slip_effective(self, k, axis):
        """Effective inertia of bond k for a relative rotation about `axis` (unit, world): the two
        sides' moments about axes through their own COMs (the smallest of all parallel axes, so
        conservative) in series."""
        i_s, i_r = self.slip_tensors[k]
        a = float(axis @ i_s @ axis)
        if i_r is None:
            return a
        b = float(axis @ i_r @ axis)
        return a * b / (a + b) if a + b > 0 else a

    def _slip_setup(self):
        """Bond opening model (a Newmark sliding block, Newmark 1965): a bond is rigid while its load
        is under its strength; the excess load turns the two sides apart about the load's own axis,
        each side about its own COM (the smallest moment of all parallel axes, so conservative, E),
        the two sides in series; once the load falls back under the strength, the magnets pull the
        faces shut again. Rigid-plastic: the strength is held at its closed value while the bond
        opens (pull ~ (d0 / (d0 + gap))^4, so this is optimistic during the opening), and to make up
        for it the bond counts as failed early: peel or pull-off at a far gap of 0.1 d0 (pull down
        to 0.68, still above the working loads), twist at a shift of d0 / 4 (half the shift where a
        pair's sideways pull peaks). Run twice: at full strength (does the bond open: a failure)
        and at strength / 1.5 (does it hold with a 1.5x margin)."""
        m = self.model
        b_model = self.bonds
        parent = m.body_parentid
        n = len(self.bond_bodies)
        self.slip_groups = []
        for k, b in enumerate(self.bond_bodies):
            root = m.body_rootid[b]
            tree = [j for j in range(1, m.nbody) if m.body_rootid[j] == root]

            def under(j, b=b):
                while j > 0:
                    if j == b:
                        return True
                    j = parent[j]
                return False

            sub = [j for j in tree if under(j)]
            rest = [j for j in tree if not under(j)]
            mounted = m.body_jntnum[root] == 0 or m.jnt_type[m.body_jntadr[root]] != mujoco.mjtJoint.mjJNT_FREE
            self.slip_groups.append((sub, rest, bool(mounted or not rest)))
        self._slip_inertia()
        self._slip_count = 0
        d0_c = b_model.d_corner + 2 * b_model.cover
        d0_r = b_model.d_ring + 2 * b_model.cover
        lever = np.array([self.spec.L if kd == "plain" else 2 * b_model.disc_radius for kd in self.bond_kind])
        r_tw = math.hypot(self.spec.L / 2 - b_model.corner_inset, self.spec.L / 2 - b_model.corner_inset)
        self.slip_lever, self.slip_d0 = lever, min(d0_c, d0_r)
        self.slip_crit_peel = 0.1 * min(d0_c, d0_r) / lever  # rad: far gap 0.1 d0, where the pull has fallen to 0.68, still above the working loads (p99 <= 0.55 of strength)
        self.slip_crit_twist = 0.5 * (0.5 * min(d0_c, d0_r)) / r_tw  # rad; disc bonds twist through pins, not slip
        self.slip_margins = (1.0, 1.5)
        self.slip_state = np.zeros((2, 2, n, 2))  # margin, mode (peel+pull, twist), bond, (opening, rate)
        self.slip_peak = np.zeros((2, 2, n))  # peak opening / failure opening
        self.slip_failure = [None, None]

    def _slip_step(self, loads, cap_m, cap_t, h, axes):
        self._slip_count += 1
        if self._slip_count % 20 == 0:
            self._slip_inertia()
        inertia = np.array([[self._slip_effective(k, axes[k, 0]) for k in range(len(axes))], [self._slip_effective(k, axes[k, 1]) for k in range(len(axes))]])
        twist_cap = np.where(np.array(self.bond_kind) == "disc", np.inf, self.cap_tw * self.bond_scale)
        for g, margin in enumerate(self.slip_margins):
            u = loads[:, 0] / (cap_m / margin) + np.maximum(loads[:, 1], 0) / (cap_t / margin)
            excess = np.stack([(u - 1.0) * cap_m / margin, loads[:, 2] - twist_cap / margin])
            excess = np.where(np.isfinite(excess), excess, -1.0)
            crit = np.stack([self.slip_crit_peel, np.full_like(self.slip_crit_peel, self.slip_crit_twist)])
            st = self.slip_state[g]
            act = (excess > 0) | (st[..., 0] > 0)
            acc = excess / np.maximum(inertia, 1e-15)
            st[..., 1] = np.where(act, st[..., 1] + acc * h, 0.0)
            st[..., 0] = np.where(act, st[..., 0] + st[..., 1] * h, 0.0)
            closed = st[..., 0] <= 0
            st[..., 0][closed] = 0.0
            st[..., 1][closed] = 0.0
            over = st[..., 0] > 2 * crit  # failed long ago: stop the integrator from running away
            st[..., 0] = np.where(over, 2 * crit, st[..., 0])
            st[..., 1] = np.where(over, 0.0, st[..., 1])
            ratio = st[..., 0] / crit
            self.slip_peak[g] = np.maximum(self.slip_peak[g], ratio)
            if self.slip_failure[g] is None and np.any(ratio >= 1.0):
                mode, k = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
                self.slip_failure[g] = (round(float(self.data.time), 3), self.bond_names[k], ("peel" if mode == 0 else "twist"))

    def bond_loads(self, axes=None):
        """(bending N m, tension N, twist N m) per bond at the face centre; with `axes` (an array
        (bonds, 2, 3)), also fills each bond's bending-moment direction and normal (world)."""
        m, d = self.model, self.data
        mujoco.mj_rnePostConstraint(m, d)
        out = np.zeros((len(self.bond_bodies), 3))
        h = self.spec.half
        for k, (b, n_local) in enumerate(zip(self.bond_bodies, self.bond_normals)):
            rot = d.xmat[b].reshape(3, 3)
            normal = rot @ n_local
            point = d.xpos[b] + normal * h
            torque, force = d.cfrc_int[b, :3], d.cfrc_int[b, 3:]
            moment = torque + np.cross(d.subtree_com[m.body_rootid[b]] - point, force)
            twist = float(moment @ normal)
            bend = moment - twist * normal
            out[k] = (np.linalg.norm(bend), float(force @ normal), abs(twist))
            if axes is not None:
                nb = np.linalg.norm(bend)
                axes[k, 0] = bend / nb if nb > 0 else np.cross(normal, [1.0, 0.0, 0.0]) / max(np.linalg.norm(np.cross(normal, [1.0, 0.0, 0.0])), 1e-9)
                axes[k, 1] = normal
        return out

    def _bond_sample(self):
        if not self.bond_bodies:
            return
        axes = np.zeros((len(self.bond_bodies), 2, 3))
        loads = self.bond_loads(axes)
        self.loads_series.append(loads)
        if len(self.loads_series) >= self.smooth_n and self.data.time > self.settle:
            avg = np.mean(self.loads_series[-self.smooth_n:], axis=0)
            q = self.data.qpos[self.qadr[np.maximum(self.bond_joint, 0)]]
            aligned = (self.bond_joint >= 0) & (np.abs(np.remainder(q + math.pi / 4, math.pi / 2) - math.pi / 4) < math.radians(5))
            cap_m = np.where(aligned, self.cap_m_al, self.cap_m) * self.bond_scale
            cap_t = np.where(aligned, self.cap_t_al, self.cap_t) * self.bond_scale
            util = avg[:, 0] / cap_m + np.maximum(avg[:, 1], 0) / cap_t
            self._slip_step(loads, cap_m, cap_t, self.every * self.model.opt.timestep, axes)
            util_tw = avg[:, 2] / (self.cap_tw * self.bond_scale)
            u = np.maximum(util, util_tw)
            self.series.append(u)
            if self.first_bond_failure is None and np.any(u > 1.0):
                k = int(np.argmax(u))
                self.first_bond_failure = (round(float(self.data.time), 3), self.bond_names[k], round(float(u[k]), 2))
        if len(self.loads_series) > 4 * self.smooth_n:
            self.loads_series = self.loads_series[-self.smooth_n:]

    # ------------------------------------------------------------------ results
    def summary(self) -> dict:
        jp = self.joint
        used = {n: round(float(p) * 1e3, 3) for n, p in zip(self.names, self.peak_smooth) if p > 0.02 * jp.tooth_cont}
        util = np.array(self.series) if self.series else np.zeros((1, max(1, len(self.bond_bodies))))
        worst = util.max(axis=0) if len(util) else np.zeros(1)
        k = int(np.argmax(worst)) if worst.size else 0
        mgl = self.spec.mass * G * self.spec.L
        peak = float(self.peak_smooth.max())
        return {
            "peak_joint_mNm": used,
            "peak_joint_over_mgL": round(peak / mgl, 2),
            "peak_joint_over_capacity": round(peak / jp.tooth_peak, 3),
            "joint_capacity_peak_mNm": round(jp.tooth_peak * 1e3, 2),
            "joint_broken": self.broken_joint,
            "peak_bond_utilisation": round(float(worst.max()) if worst.size else 0.0, 3),
            "worst_bond": self.bond_names[k] if self.bond_names else None,
            "first_bond_failure": self.first_bond_failure,
            "peak_util_disc_bonds": round(float(max([w for w, kd in zip(worst, self.bond_kind) if kd == "disc"], default=0.0)), 3),
            "peak_util_plain_bonds": round(float(max([w for w, kd in zip(worst, self.bond_kind) if kd == "plain"], default=0.0)), 3),
            "bond_opening": round(float(self.slip_peak[0].max()), 3) if self.bond_bodies else 0.0,
            "bond_opening_at_1.5x": round(float(self.slip_peak[1].max()), 3) if self.bond_bodies else 0.0,
            "bond_opened": self.slip_failure[0] if self.bond_bodies else None,
            "bond_opened_at_1.5x": self.slip_failure[1] if self.bond_bodies else None,
            "energy_J": round(self.energy, 4),
            "steps": self.step_count,
            "failed_steps": self.failed_steps,
            "copper_J": round(self.copper, 4),
        }

    def position(self, body):
        return self.data.xpos[self.model.body(body).id].copy()

    def angle(self, name):
        return float(self.data.qpos[self.qadr[self.index[name]]])
