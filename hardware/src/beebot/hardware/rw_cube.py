"""Reaction-wheel cube RW-P (hardware/runs/module_phys/concepts.md, sections 6-7) in MuJoCo.

One cube: a free body (shell, frame, flex, cells, electronics) with
1. two flywheels (rotor X near the +x face, rotor Y near the +y face) on hinges, spun by a flex-PCB
   motor (torque constant and resistance from the stator geometry with magpylib, back-EMF, a power
   limit from the storage) and stopped by a pawl: a one-sided spring-damper on the rotor angle with
   the catch duration and damping as parameters, solved at a fine time step, then a joint lock;
2. eight free N52 corner spheres in spherical pockets: ball joints (free to turn, pocket friction as a
   load-dependent friction loss), the sphere's place in its pocket quasi-static (against the pocket
   wall in the direction of its net magnetic force, clearance as a parameter);
3. magnetic forces between all spheres and rotor magnets of all cubes as point dipoles through
   mmsim.mjcoupling.DipoleCoupling (exact for spheres, far-field for the rotor cores), with the
   calibrated derates of scaling.py (0.80 spheres, 0.66 cylinders) and their spread;
4. floor friction (MuJoCo elliptic cones) and capillary adhesion at the corner feet, both Monte Carlo
   variables, plus contact damping (restitution) as one.

Moves (concepts.md section 7) are scenes plus an impulse: ROLL on the floor and on top of a row,
CLIMB and DESCEND (onto and off a lone cube or a row), SPLIT from a pair, YAW trims, BOND capture,
DOCK (a roll that ends next to a neighbour) and a 20 cm FALL. Each trial is classified into the
outcome classes of section 7. Every simplification is listed in IDEALISATIONS.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from beebot.hardware.mmsim.mjcoupling import DipoleCoupling

MU0 = 4e-7 * math.pi
G = 9.81
BR_N52 = 1.45  # T, as mmsim and scaling.py
RHO = {"ndfeb": 7500.0, "brass": 8500.0, "tungsten": 17500.0, "shell": 0.5 * (1200.0 + 3500.0)}
RHO_CU = 1.72e-8
K_SPHERE = 0.80  # published / model pull, 3 mm sphere (scaling.py calibration)
K_CYL = 0.66  # published / model pull, discs (scaling.py calibration); simstack found 0.73-0.87 for 2-5 mm magnets
WALL = 0.40e-3  # flex 0.11 mm on a 0.3 mm resin frame (scaling.wall for L <= 12 mm)
STATOR_GAP = 0.10e-3  # rotor face to stator copper
CELL_STACK = (4.4e-3, 3.2e-3, 3.4e-3)  # three CeraCharge 1812 stacked on the -z face
P_IDLE = 1.0e-3

IDEALISATIONS = [
    "Corner spheres are uniformly magnetised rigid dipoles (exact for N52 spheres; demagnetisation and induced magnetisation ignored, mu_r about 1.05).",
    "Rotor cores are point dipoles at the rotor centre (far-field: error grows near the corner spheres; checked against magpylib in tests).",
    "Sphere translation in the pocket is quasi-static: the sphere sits on the pocket wall in the direction of its net magnetic force (low-pass over its travel time); its 3 mg translational inertia and rattle are not simulated.",
    "Pocket friction acts on sphere rotation only, as Coulomb friction mu_p x |magnetic force on the sphere| x radius; inertial loads on the pocket during impacts do not add to it.",
    "Cubes are rigid boxes with sharp edges (MuJoCo box contacts, at most 4-8 contact points); real frames have rounded edges and corner feet of 0.1 mm radius.",
    "Contact is MuJoCo soft contact (time constant 0.1 ms): impacts, bounces and chatter are not Hertz impact physics; restitution is a Monte Carlo variable through the contact damping ratio.",
    "Adhesion is a capillary pull per corner foot within a rupture range, applied only between cubes and the floor, not between cubes.",
    "The pawl stop is a one-sided linear spring-damper on the rotor angle (duration and damping are parameters), then a rigid joint lock; tooth wear, pawl bounce-out and backlash rattle after the catch are not modelled.",
    "Spin-up is not simulated inside each trial: the rotor starts at the commanded speed (with its speed error) and the cube at rest; the motor reaction during spin-up is checked against tip and yaw-slip thresholds separately (spinup()).",
    "Motor: ideal 6-step commutation, torque constant averaged over rotor angle, no cogging (air-core stator), drive power limited by the storage (TDK CeraCharge: 3 mA pulses for 1 s then 1 mA per cell), converter efficiency 0.8, no coil heating.",
    "Bearings are ideal hinges with Couette drag in the 0.1 mm stator gap; bearing friction, wobble and the bearing speed limit (6,000 rad/s plain, 20,000 rad/s jewel) are not dynamic: designs beyond the limit are marked infeasible.",
    "The controller knows its own mass properties and commands H = k x sqrt(2 I E_barrier / eta) from a nominal model (nominal friction 0.4, nominal magnets); k per move is calibrated in simulation.",
    "Print tolerance enters as sphere pocket position (+-0.05 mm per axis), pocket clearance (+-0.025 mm on the radius) and cube size (+-0.05 mm); face flatness and frame warp are not modelled.",
    "No air drag on cubes; no electrical or thermal limits during a move; IMU and IR sensing are ideal for classification (the policy sees the classified outcome).",
]


# ---------------------------------------------------------------------------------------------
# Design and derived quantities
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Design:
    """Design point. SI units. Defaults: concepts.md pick 1 (7 mm, jewel bearings, 1.0 mm spheres)."""

    L: float = 7.0e-3
    sphere_d: float = 1.0e-3
    cover: float = 0.10e-3
    clearance: float = 0.02e-3  # radial clearance of a sphere in its pocket
    rotor_r: float = 2.10e-3
    rotor_t: float = 0.75e-3
    core_frac: float = 0.5  # NdFeB core radius / rotor radius (diametric core, heavy rim outside)
    rim: str = "tungsten"
    omega_cap: float = 20000.0  # bearing limit, rad/s (jewel 20k, plain 6k)
    catch_T: float = 50e-6  # pawl stop duration (half period of the tooth spring), s
    catch_zeta: float = 0.7  # damping ratio of the tooth contact (damped pad on the pawl)
    teeth: int = 2  # notches on the rotor rim
    backlash: float = math.radians(5.0)  # V-notch play
    # command multipliers on the nominal-model impulse (calibrated per move in simulation)
    k_roll: float = 1.6
    k_top: float = 1.4
    k_climb: float = 1.15
    k_descend: float = 1.3
    k_split: float = 1.4
    yaw_target_deg: float = 15.0
    floor_mu: tuple = (0.2, 0.6)  # Monte Carlo range of cube-floor friction (corner foot material x floor)
    steel_floor: bool = False  # variant: a steel sheet floor, image dipoles of every magnet (saturation derate steel_k)
    steel_k: float = 0.66

    def key(self) -> str:
        return (f"L{self.L * 1e3:.2f}_s{self.sphere_d * 1e3:.2f}_c{self.clearance * 1e6:.0f}_r{self.rotor_r * 1e3:.2f}"
                f"_t{self.rotor_t * 1e3:.2f}_T{self.catch_T * 1e6:.0f}_kc{self.k_climb:.2f}")


@dataclass
class Derived:
    L: float
    inner: float
    m_body: float
    com_body: np.ndarray
    inertia_body: np.ndarray  # 3x3 about com_body
    rotor_mass: float
    rotor_i_axial: float
    rotor_i_trans: float
    rotor_pos: dict  # 'x' / 'y' -> local centre
    core_moment: float
    sphere_r: float
    sphere_mass: float
    sphere_moment: float  # ideal (Br / mu0) V, before derate
    pocket_inset: float  # sphere-pocket centre inset from each face
    m_total: float
    storage: dict
    motor: dict
    feasible: bool
    notes: list = field(default_factory=list)


def _cyl_inertia(m, r, t):
    return 0.5 * m * r * r, m * (3 * r * r + t * t) / 12.0


def rotor_props(d: Design):
    rc = d.core_frac * d.rotor_r
    m_core = RHO["ndfeb"] * math.pi * rc**2 * d.rotor_t
    m_rim = RHO[d.rim] * math.pi * (d.rotor_r**2 - rc**2) * d.rotor_t
    ia = 0.5 * m_core * rc**2 + 0.5 * m_rim * (d.rotor_r**2 + rc**2)
    it = m_core * (3 * rc**2 + d.rotor_t**2) / 12 + m_rim * (3 * (d.rotor_r**2 + rc**2) + d.rotor_t**2) / 12
    moment = BR_N52 / MU0 * math.pi * rc**2 * d.rotor_t
    return m_core + m_rim, ia, it, moment


def storage_for(L):
    if L - 2 * WALL >= 10.0e-3 - 1e-9:
        return {"name": "LiPo 301010 10 mAh", "mass": 0.35e-3, "p_pulse": 0.074, "t_pulse": 1e9, "p_cont": 0.074, "energy": 106.6, "dims": (10e-3, 10e-3, 3e-3)}
    return {"name": "3 x TDK CeraCharge 1812 (100 uAh)", "mass": 0.12e-3, "p_pulse": 3 * 3e-3 * 1.4, "t_pulse": 1.0, "p_cont": 3 * 1e-3 * 1.4, "energy": 1.5, "dims": CELL_STACK}


def derive(d: Design) -> Derived:
    L = d.L
    inner = L - 2 * WALL
    notes = []
    feasible = True
    r_s = d.sphere_d / 2
    pocket_inset = r_s + d.cover + d.clearance
    m_rot, ia, it, core_moment = rotor_props(d)
    # rotors must clear each other and the frame (scaling.arch_rw geometry)
    if d.rotor_r > inner / 2 - d.rotor_t - 0.15e-3:
        feasible = False
        notes.append("rotor does not fit")
    if d.sphere_d + 2 * d.cover + 2 * d.clearance > 0.32 * L:
        feasible = False
        notes.append("corner pocket too large for the frame")
    st = storage_for(L)
    sd = st["dims"]
    if max(sd) > inner + 1e-9:
        feasible = False
        notes.append("storage does not fit")
    # body parts (local frame at the box centre): shell, cells (-z face), MCU (-x), IMU (-y), power (+z)
    shell = RHO["shell"] * (L**3 - inner**3)
    elec_mcu, elec_imu, elec_pwr = 0.02e-3, 0.01e-3, 0.01e-3 + 0.02e-3 * (L / 5e-3)
    parts = [  # (mass, position, inertia about own centre 3x3)
        (shell, np.zeros(3), np.eye(3) * RHO["shell"] * (L**5 - inner**5) / 6.0),
        (st["mass"], np.array([0.0, -0.1 * inner, -inner / 2 + sd[2] / 2]), np.diag([sd[1]**2 + sd[2]**2, sd[0]**2 + sd[2]**2, sd[0]**2 + sd[1]**2]) * st["mass"] / 12),
        (elec_mcu, np.array([-inner / 2 + 0.4e-3, 0, 0]), np.zeros((3, 3))),
        (elec_imu, np.array([0, -inner / 2 + 0.35e-3, 0.24 * inner]), np.zeros((3, 3))),
        (elec_pwr, np.array([-0.08 * inner, 0, inner / 2 - 0.3e-3]), np.zeros((3, 3))),
    ]
    m_sph = RHO["ndfeb"] * 4 / 3 * math.pi * r_s**3
    x_r = inner / 2 - STATOR_GAP - d.rotor_t / 2
    rotor_pos = {"x": np.array([x_r, 0, 0]), "y": np.array([0, x_r, 0])}
    raw = sum(p[0] for p in parts) + 2 * m_rot + 8 * m_sph
    margin = 0.10 * raw  # frame bars, pins, bushings, flex folds, solder (as scaling.arch_rw)
    parts.append((margin, np.zeros(3), np.eye(3) * margin * L * L / 6))
    m_body = sum(p[0] for p in parts)
    com = sum(p[0] * p[1] for p in parts) / m_body
    I = np.zeros((3, 3))
    for m, p, Ic in parts:
        r = p - com
        I += Ic + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    m_total = m_body + 2 * m_rot + 8 * m_sph
    motor = motor_model(d)
    return Derived(L=L, inner=inner, m_body=m_body, com_body=com, inertia_body=I, rotor_mass=m_rot, rotor_i_axial=ia,
                   rotor_i_trans=it, rotor_pos=rotor_pos, core_moment=core_moment, sphere_r=r_s, sphere_mass=m_sph,
                   sphere_moment=BR_N52 / MU0 * 4 / 3 * math.pi * r_s**3, pocket_inset=pocket_inset, m_total=m_total,
                   storage=st, motor=motor, feasible=feasible, notes=notes)


# ---------------------------------------------------------------------------------------------
# Motor: flex stator geometry -> R and k_t (magpylib), power-limited drive, spin-up
# ---------------------------------------------------------------------------------------------

def sector_loops(r_in, r_out, half_angle, pitch, n_arc=24):
    """Nested closed loops of one sector coil (one layer), outermost first: (n, 2) arrays."""
    loops, k = [], 0
    while True:
        ro = r_out - pitch / 2 - k * pitch
        ri = r_in + pitch / 2 + k * pitch
        if ro - ri < 0.5 * pitch:
            break
        ao = half_angle - (k + 0.5) * pitch / ro
        ai = half_angle - (k + 0.5) * pitch / ri
        if ai <= 0.02 or ao <= 0.02:
            break
        a1 = np.linspace(-ao, ao, n_arc)
        a2 = np.linspace(ai, -ai, n_arc)
        pts = np.vstack([np.c_[ro * np.cos(a1), ro * np.sin(a1)], np.c_[ri * np.cos(a2), ri * np.sin(a2)]])
        loops.append(np.vstack([pts, pts[:1]]))
        k += 1
    return loops


@lru_cache(maxsize=256)
def stator_constants(rotor_r: float, core_r: float, t: float, trace=75e-6, space=75e-6, cu=35e-6, layers=2,
                     layer_pitch=60e-6, n_angles=6):
    """Three sector coils on the face flex under a diametric core (2-pole), 6-step drive, two phases in series.

    Returns phase resistance (ohm), torque constant (N m / A, two phases on, averaged over a 60 degree
    commutation sector, magnet at full remanence) and turns per coil."""
    import magpylib as mp

    pitch = trace + space
    r_o = rotor_r
    r_i = 0.3 * r_o
    half = math.pi / 3 - 0.5 * space / ((r_i + r_o) / 2)
    loops = sector_loops(r_i, r_o, half, pitch)
    length = sum(np.linalg.norm(np.diff(l, axis=0), axis=1).sum() for l in loops) * layers
    R_phase = RHO_CU * length / (trace * cu)

    def coil(phase_angle, current):
        c, s = math.cos(phase_angle), math.sin(phase_angle)
        objs = []
        for lay in range(layers):
            z = -(STATOR_GAP + cu / 2 + lay * layer_pitch)
            for l in loops:
                v = np.c_[c * l[:, 0] - s * l[:, 1], s * l[:, 0] + c * l[:, 1], np.full(len(l), z)]
                objs.append(mp.current.Polyline(current=current, vertices=v))
        return objs

    coils = {(a, b): coil(2 * math.pi * a / 3, 1.0) + coil(2 * math.pi * b / 3, -1.0) for a in range(3) for b in range(3) if a != b}
    taus = []
    for th in np.linspace(0, math.pi / 3, n_angles + 1)[:-1]:
        mag = mp.magnet.Cylinder(polarization=(BR_N52 * math.cos(th + math.pi / 6), BR_N52 * math.sin(th + math.pi / 6), 0),
                                 dimension=(2 * core_r, t), position=(0, 0, t / 2))
        best = 0.0
        for src in coils.values():
            _, T = mp.getFT(src, mag, pivot=(0, 0, t / 2), meshing=(12, 3, 4))
            best = max(best, float(np.asarray(T).reshape(-1, 3).sum(0)[2]))
        taus.append(best)
    return float(R_phase), float(np.mean(taus)), len(loops) * layers


def motor_model(d: Design, r_switch=0.5, v_bus=3.3, i_max=0.5, eta_conv=0.8):
    R_phase, kt_ideal, turns = stator_constants(round(d.rotor_r, 7), round(d.core_frac * d.rotor_r, 7), round(d.rotor_t, 7))
    kt = kt_ideal * math.sqrt(K_CYL)  # magnet derate as for pull forces (Br_eff / Br = sqrt(0.66))
    # Couette drag of the rotor face in the 0.1 mm stator gap (air), both faces counted
    c_drag = 2 * 1.8e-5 * math.pi * d.rotor_r**4 / (2 * STATOR_GAP)
    return {"R_phase": R_phase, "R_m": 2 * R_phase + r_switch, "k_t": kt, "k_t_ideal": kt_ideal, "turns_per_coil": turns,
            "v_bus": v_bus, "i_max": i_max, "eta_conv": eta_conv, "c_drag": c_drag}


def motor_current(mot, omega, p_avail):
    """Coil current (A) at rotor speed omega with p_avail W at the converter input."""
    Rm, ke = mot["R_m"], mot["k_t"]
    p = p_avail * mot["eta_conv"]
    bemf = ke * abs(omega)
    i_p = (-bemf + math.sqrt(bemf * bemf + 4 * Rm * p)) / (2 * Rm)
    i_v = max((mot["v_bus"] - bemf) / Rm, 0.0)
    return min(mot["i_max"], i_p, i_v)


def spinup(d: Design, der: Derived, omega_target: float, torque_cap: float | None = None, dt=2e-3, t_max=120.0):
    """Spin a rotor from rest to omega_target with the storage's power. Returns time (s), energy drawn from
    the storage (J, motor only), peak motor torque (the reaction the cube feels) and reachability."""
    mot, st = der.motor, der.storage
    I = der.rotor_i_axial
    w, t, e, peak = 0.0, 0.0, 0.0, 0.0
    if omega_target > d.omega_cap:
        return {"t": float("inf"), "energy": float("inf"), "peak_torque": 0.0, "ok": False, "why": "above bearing limit"}
    while w < omega_target:
        p = st["p_pulse"] if t < st["t_pulse"] else st["p_cont"]
        p = max(p - P_IDLE, 1e-5)
        i = motor_current(mot, w, p)
        tau = mot["k_t"] * i
        if torque_cap is not None and tau > torque_cap:
            i = torque_cap / mot["k_t"]
            tau = torque_cap
        peak = max(peak, tau)
        acc = (tau - mot["c_drag"] * w) / I
        if acc <= 1e-3:
            return {"t": float("inf"), "energy": float("inf"), "peak_torque": peak, "ok": False, "why": "torque cannot reach the speed"}
        w += acc * dt
        e += (i * i * mot["R_m"] + mot["k_t"] * w * i) / mot["eta_conv"] * dt
        t += dt
        if t > t_max:
            return {"t": t, "energy": e, "peak_torque": peak, "ok": False, "why": "longer than t_max"}
    return {"t": t, "energy": e, "peak_torque": peak, "ok": True, "why": ""}


# ---------------------------------------------------------------------------------------------
# Monte Carlo variation
# ---------------------------------------------------------------------------------------------

@dataclass
class Variation:
    mu_floor: float = 0.4
    mu_cc: float = 0.4
    adhesion: float = 0.045e-3  # N per corner foot (worst case 4 pi gamma r at r = 0.1 mm is 0.09 mN)
    adh_range: float = 0.1e-3
    dampratio: float = 0.6
    derate: float = K_SPHERE
    derate_cyl: float = K_CYL
    sphere_strength: np.ndarray | None = None  # (ncube, 8)
    pocket_offset: np.ndarray | None = None  # (ncube, 8, 3)
    clearance: np.ndarray | None = None  # (ncube, 8)
    size: np.ndarray | None = None  # (ncube,)
    mu_pocket: float = 0.25
    speed_err: float = 0.0
    catch_delay: float = 0.3e-3
    catch_T: float | None = None
    catch_zeta: float | None = None
    rotor_angle: np.ndarray | None = None  # (ncube, 2) angle of locked rotors
    place_jitter: np.ndarray | None = None  # (ncube, 3) metres, (ncube,) yaw added at placement

    def asdict_small(self):
        return {k: (float(v) if isinstance(v, (float, int, np.floating)) else None) for k, v in asdict(self).items()
                if not isinstance(v, np.ndarray) and v is not None}


def nominal_variation(d: Design, ncube: int) -> Variation:
    return Variation(sphere_strength=np.ones((ncube, 8)), pocket_offset=np.zeros((ncube, 8, 3)),
                     clearance=np.full((ncube, 8), d.clearance), size=np.full(ncube, d.L),
                     rotor_angle=np.zeros((ncube, 2)), catch_T=d.catch_T, catch_zeta=d.catch_zeta,
                     place_jitter=np.zeros((ncube, 4)))


MC_RANGES = {
    "mu_floor": (0.2, 0.6),
    "mu_cc": (0.2, 0.6),
    "adhesion": (0.0, 0.09e-3),
    "adh_range": (0.02e-3, 0.2e-3),
    "dampratio": (0.2, 1.0),
    "derate": (0.70, 0.90),
    "derate_cyl": (0.60, 0.85),
    "strength": (0.90, 1.10),  # per magnet, on the moment (force +-10 % per magnet)
    "print": 0.05e-3,  # +- per axis
    "clearance": 0.025e-3,  # +- on the radius
    "mu_pocket": (0.1, 0.4),
    "speed_err": 0.03,  # +- relative
    "catch_delay": (0.05e-3, 1.0e-3),
    "catch_T": (0.5, 2.0),  # x nominal
    "catch_zeta": 0.3,  # +- around the design value
    "place": 0.02e-3,  # +- placement jitter of cubes in a scene (m), plus +-0.5 degree yaw
}


def sample_variation(d: Design, ncube: int, rng: np.random.Generator) -> Variation:
    u = lambda key: rng.uniform(*MC_RANGES[key])
    pr = MC_RANGES["print"]
    return Variation(
        mu_floor=rng.uniform(*d.floor_mu), mu_cc=u("mu_cc"), adhesion=u("adhesion"), adh_range=u("adh_range"), dampratio=u("dampratio"),
        derate=u("derate"), derate_cyl=u("derate_cyl"),
        sphere_strength=rng.uniform(*MC_RANGES["strength"], size=(ncube, 8)),
        pocket_offset=rng.uniform(-pr, pr, size=(ncube, 8, 3)),
        clearance=np.clip(d.clearance + rng.uniform(-MC_RANGES["clearance"], MC_RANGES["clearance"], size=(ncube, 8)), 0.002e-3, None),
        size=d.L + rng.uniform(-pr, pr, size=ncube),
        mu_pocket=u("mu_pocket"), speed_err=rng.uniform(-MC_RANGES["speed_err"], MC_RANGES["speed_err"]),
        catch_delay=u("catch_delay"), catch_T=d.catch_T * u("catch_T"),
        catch_zeta=float(np.clip(d.catch_zeta + rng.uniform(-MC_RANGES["catch_zeta"], MC_RANGES["catch_zeta"]), 0.05, 1.5)),
        rotor_angle=rng.uniform(0, 2 * math.pi, size=(ncube, 2)),
        place_jitter=np.c_[rng.uniform(-MC_RANGES["place"], MC_RANGES["place"], size=(ncube, 3)), rng.uniform(-0.5, 0.5, size=ncube)],
    )


# ---------------------------------------------------------------------------------------------
# Scenes and the MuJoCo model
# ---------------------------------------------------------------------------------------------

CORNERS = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
PLACE_GAP = 0.12e-3  # initial gap between neighbouring cells (> 2 x print tolerance + placement jitter): no start overlap


def cube_rotations():
    """The 24 proper rotations of a cube, as 3x3 integer matrices."""
    out = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for signs in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1), (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)):
            M = np.zeros((3, 3))
            for i in range(3):
                M[i, perm[i]] = signs[i]
            if np.linalg.det(M) > 0:
                out.append(M)
    return out


ROTS = cube_rotations()


@dataclass
class CubeSpec:
    cell: tuple  # grid cell (i, j, k), k = layer
    rot: np.ndarray  # 3x3, local -> world


def quat_wxyz(Rm):
    q = R.from_matrix(Rm).as_quat()  # x y z w
    return np.array([q[3], q[0], q[1], q[2]])


def _f(v):
    return " ".join(f"{x:.9g}" for x in np.ravel(v))


def build_xml(d: Design, der: Derived, cubes: list, var: Variation, dt: float, floor=True, cam=None) -> str:
    """MJCF for the scene. Cube k has body c{k}, rotors c{k}_rx / c{k}_ry (hinges, locked by equality
    unless released) and spheres c{k}_s{i} (ball joints)."""
    L = d.L
    pitch = L + PLACE_GAP  # start apart (covers the size tolerance); magnets close the gaps while the scene settles
    tc = max(1e-4, 4 * dt)
    body_xml, eq_xml = [], []
    rs = der.sphere_r
    for k, c in enumerate(cubes):
        size = var.size[k]
        jit = var.place_jitter[k]
        pos = np.array([c.cell[0] * pitch + jit[0], c.cell[1] * pitch + jit[1], c.cell[2] * pitch + size / 2 + 2e-6])
        rot = R.from_euler("z", jit[3], degrees=True).as_matrix() @ c.rot
        Ib = der.inertia_body
        rgba = "0.85 0.62 0.25 1" if k == 0 else "0.55 0.6 0.68 1"
        b = [f'<body name="c{k}" pos="{_f(pos)}" quat="{_f(quat_wxyz(rot))}">',
             f'<freejoint name="c{k}_free"/>',
             f'<inertial pos="{_f(der.com_body)}" mass="{der.m_body:.9g}" fullinertia="{Ib[0,0]:.9g} {Ib[1,1]:.9g} {Ib[2,2]:.9g} {Ib[0,1]:.9g} {Ib[0,2]:.9g} {Ib[1,2]:.9g}"/>',
             f'<geom name="c{k}_box" type="box" size="{size / 2:.9g} {size / 2:.9g} {size / 2:.9g}" friction="{var.mu_cc:.4g} 0 0" rgba="{rgba}" mass="0"/>']
        # face markers so orientation is visible in renders (+x red, +y green, +z blue)
        for ax, col in ((0, "0.85 0.2 0.2 1"), (1, "0.2 0.7 0.3 1"), (2, "0.2 0.35 0.85 1")):
            p = np.zeros(3)
            p[ax] = size / 2
            hs = [0.18 * size] * 3
            hs[ax] = 0.004e-3
            b.append(f'<geom type="box" pos="{_f(p)}" size="{_f(hs)}" rgba="{col}" contype="0" conaffinity="0" mass="0"/>')
        for name, axis in (("x", (1, 0, 0)), ("y", (0, 1, 0))):
            rp = der.rotor_pos[name]
            ia, it = der.rotor_i_axial, der.rotor_i_trans
            diag = (ia, it, it) if name == "x" else (it, ia, it)
            zaxis = "0 1 0 90" if name == "x" else "1 0 0 90"
            b.append(f'<body name="c{k}_r{name}" pos="{_f(rp)}">')
            b.append(f'<joint name="c{k}_r{name}" type="hinge" axis="{_f(axis)}" damping="0" armature="0"/>')
            b.append(f'<inertial pos="0 0 0" mass="{der.rotor_mass:.9g}" diaginertia="{_f(diag)}"/>')
            b.append(f'<geom type="cylinder" size="{d.rotor_r:.9g} {d.rotor_t / 2:.9g}" axisangle="{zaxis}" rgba="0.3 0.3 0.32 0.6" contype="0" conaffinity="0" mass="0" group="2"/>')
            b.append("</body>")
            eq_xml.append(f'<joint name="c{k}_lock{name}" joint1="c{k}_r{name}" polycoef="0 0 0 0 0" solref="{2 * dt:.3g} 1"/>')
        Is = 0.4 * der.sphere_mass * rs * rs
        for i, sc in enumerate(CORNERS):
            pc = sc * (size / 2 - der.pocket_inset + d.clearance - var.clearance[k, i]) + var.pocket_offset[k, i]
            b.append(f'<body name="c{k}_s{i}" pos="{_f(pc)}">')
            b.append(f'<joint name="c{k}_s{i}" type="ball" damping="{2e-10 * (rs / 0.5e-3) ** 3:.3g}"/>')
            b.append(f'<inertial pos="0 0 0" mass="{der.sphere_mass:.9g}" diaginertia="{Is:.9g} {Is:.9g} {Is:.9g}"/>')
            b.append(f'<geom type="sphere" size="{rs:.9g}" rgba="0.75 0.75 0.78 1" contype="0" conaffinity="0" mass="0" group="1"/>')
            b.append(f'<geom type="cylinder" size="{0.25 * rs:.9g} {0.55 * rs:.9g}" rgba="0.8 0.1 0.1 1" contype="0" conaffinity="0" mass="0" group="1"/>')
            b.append("</body>")
        b.append("</body>")
        body_xml.append("\n".join(b))
    fs = (round(0.1 / L) + 0.5) * L  # checker squares of one cell, edges half a cell from the cell centres
    floor_rgba = "0.62 0.66 0.72 1" if d.steel_floor else "1 1 1 1"
    floor_xml = (f'<geom name="floor" type="plane" size="{fs:.6g} {fs:.6g} 0.001" priority="1" friction="{var.mu_floor:.4g} 0 0" material="grid" rgba="{floor_rgba}"/>'
                 if floor else "")
    asset_xml = (f'<asset><texture name="grid" type="2d" builtin="checker" rgb1="0.93 0.93 0.91" rgb2="0.84 0.84 0.82" width="64" height="64"/>'
                 f'<material name="grid" texture="grid" texrepeat="{fs / L:.6g} {fs / L:.6g}"/></asset>')
    return f"""<mujoco model="rw_cube">
<option timestep="{dt:.6g}" gravity="0 0 -9.81" cone="elliptic" integrator="implicitfast" impratio="10"/>
<size memory="64M"/>
<visual><global offwidth="1280" offheight="720"/><map znear="0.002" zfar="80"/><quality shadowsize="4096"/>
<headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5"/></visual>
<statistic extent="0.03" center="0 0 0.005"/>
{asset_xml}
<default>
<geom solref="{tc:.3g} {var.dampratio:.3g}" solimp="0.9 0.95 2e-5 0.5 2" condim="3"/>
<joint solreffriction="{max(2 * dt, 4e-5):.3g} 1"/>
</default>
<worldbody>
<light pos="0.02 -0.03 0.08" dir="-0.2 0.4 -1" diffuse="0.6 0.6 0.6"/>
{floor_xml}
{chr(10).join(body_xml)}
</worldbody>
<equality>
{chr(10).join(eq_xml)}
</equality>
</mujoco>"""



# ---------------------------------------------------------------------------------------------
# Per-step force kernels (numba; same physics as mmsim.mjcoupling.dipole_pairs, checked in tests)
# ---------------------------------------------------------------------------------------------

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]


@njit(cache=True)
def _dipole_kernel(xpos, xmat, xipos, bodies, laxis, moment, ii, jj, xfrc, Fout, Bout):
    n = bodies.shape[0]
    P = np.empty((n, 3))
    M = np.empty((n, 3))
    for a in range(n):
        b = bodies[a]
        for r in range(3):
            P[a, r] = xpos[b, r]
            M[a, r] = (xmat[b, 3 * r] * laxis[a, 0] + xmat[b, 3 * r + 1] * laxis[a, 1] + xmat[b, 3 * r + 2] * laxis[a, 2]) * moment[a]
        for r in range(3):
            Fout[a, r] = 0.0
            Bout[a, r] = 0.0
    km = 1e-7
    for p in range(ii.shape[0]):
        i = ii[p]
        j = jj[p]
        rx = P[j, 0] - P[i, 0]
        ry = P[j, 1] - P[i, 1]
        rz = P[j, 2] - P[i, 2]
        d2 = rx * rx + ry * ry + rz * rz
        d = np.sqrt(d2)
        ux, uy, uz = rx / d, ry / d, rz / d
        miu = M[i, 0] * ux + M[i, 1] * uy + M[i, 2] * uz
        mju = M[j, 0] * ux + M[j, 1] * uy + M[j, 2] * uz
        mimj = M[i, 0] * M[j, 0] + M[i, 1] * M[j, 1] + M[i, 2] * M[j, 2]
        inv3 = km / (d2 * d)
        c4 = 3 * km / (d2 * d2)
        u = (ux, uy, uz)
        for r in range(3):
            Bout[j, r] += (3 * u[r] * miu - M[i, r]) * inv3
            Bout[i, r] += (3 * u[r] * mju - M[j, r]) * inv3
            f = c4 * (miu * M[j, r] + mju * M[i, r] + mimj * u[r] - 5 * miu * mju * u[r])
            Fout[j, r] += f
            Fout[i, r] -= f
    for a in range(n):
        b = bodies[a]
        Tx = M[a, 1] * Bout[a, 2] - M[a, 2] * Bout[a, 1]
        Ty = M[a, 2] * Bout[a, 0] - M[a, 0] * Bout[a, 2]
        Tz = M[a, 0] * Bout[a, 1] - M[a, 1] * Bout[a, 0]
        dx = P[a, 0] - xipos[b, 0]
        dy = P[a, 1] - xipos[b, 1]
        dz = P[a, 2] - xipos[b, 2]
        Fx, Fy, Fz = Fout[a, 0], Fout[a, 1], Fout[a, 2]
        xfrc[b, 0] += Fx
        xfrc[b, 1] += Fy
        xfrc[b, 2] += Fz
        xfrc[b, 3] += Tx + dy * Fz - dz * Fy
        xfrc[b, 4] += Ty + dz * Fx - dx * Fz
        xfrc[b, 5] += Tz + dx * Fy - dy * Fx


@njit(cache=True)
def _image_kernel(xpos, xmat, xipos, bodies, laxis, moment, kimg, xfrc):
    """Force and torque on every dipole from the images of all dipoles in a permeable floor at z = 0
    (image of m at (x, y, z) is kimg * (-mx, -my, mz) at (x, y, -z)); the reaction goes to the floor."""
    n = bodies.shape[0]
    P = np.empty((n, 3))
    M = np.empty((n, 3))
    for a in range(n):
        b = bodies[a]
        for r in range(3):
            P[a, r] = xpos[b, r]
            M[a, r] = (xmat[b, 3 * r] * laxis[a, 0] + xmat[b, 3 * r + 1] * laxis[a, 1] + xmat[b, 3 * r + 2] * laxis[a, 2]) * moment[a]
    km = 1e-7
    for j in range(n):
        F0 = 0.0
        F1 = 0.0
        F2 = 0.0
        B0 = 0.0
        B1 = 0.0
        B2 = 0.0
        for i in range(n):
            mi0 = -kimg * M[i, 0]
            mi1 = -kimg * M[i, 1]
            mi2 = kimg * M[i, 2]
            rx = P[j, 0] - P[i, 0]
            ry = P[j, 1] - P[i, 1]
            rz = P[j, 2] + P[i, 2]
            d2 = rx * rx + ry * ry + rz * rz
            d = np.sqrt(d2)
            ux, uy, uz = rx / d, ry / d, rz / d
            miu = mi0 * ux + mi1 * uy + mi2 * uz
            mju = M[j, 0] * ux + M[j, 1] * uy + M[j, 2] * uz
            mimj = mi0 * M[j, 0] + mi1 * M[j, 1] + mi2 * M[j, 2]
            inv3 = km / (d2 * d)
            c4 = 3 * km / (d2 * d2)
            B0 += (3 * ux * miu - mi0) * inv3
            B1 += (3 * uy * miu - mi1) * inv3
            B2 += (3 * uz * miu - mi2) * inv3
            F0 += c4 * (miu * M[j, 0] + mju * mi0 + mimj * ux - 5 * miu * mju * ux)
            F1 += c4 * (miu * M[j, 1] + mju * mi1 + mimj * uy - 5 * miu * mju * uy)
            F2 += c4 * (miu * M[j, 2] + mju * mi2 + mimj * uz - 5 * miu * mju * uz)
        b = bodies[j]
        Tx = M[j, 1] * B2 - M[j, 2] * B1
        Ty = M[j, 2] * B0 - M[j, 0] * B2
        Tz = M[j, 0] * B1 - M[j, 1] * B0
        dx = P[j, 0] - xipos[b, 0]
        dy = P[j, 1] - xipos[b, 1]
        dz = P[j, 2] - xipos[b, 2]
        xfrc[b, 0] += F0
        xfrc[b, 1] += F1
        xfrc[b, 2] += F2
        xfrc[b, 3] += Tx + dy * F2 - dz * F1
        xfrc[b, 4] += Ty + dz * F0 - dx * F2
        xfrc[b, 5] += Tz + dx * F1 - dy * F0


@njit(cache=True)
def _pocket_adhesion_kernel(xpos, xmat, xipos, cube_b, Fs, pocket, clear, off, dt, m_s, mu_p, r_s, half, adhesion, adh_range,
                            corners, body_pos_out, fric_out, xfrc):
    nc = cube_b.shape[0]
    for k in range(nc):
        b = cube_b[k]
        for i in range(8):
            f = k * 8 + i
            # local force: R^T F
            lx = xmat[b, 0] * Fs[f, 0] + xmat[b, 3] * Fs[f, 1] + xmat[b, 6] * Fs[f, 2]
            ly = xmat[b, 1] * Fs[f, 0] + xmat[b, 4] * Fs[f, 1] + xmat[b, 7] * Fs[f, 2]
            lz = xmat[b, 2] * Fs[f, 0] + xmat[b, 5] * Fs[f, 1] + xmat[b, 8] * Fs[f, 2]
            fn = np.sqrt(lx * lx + ly * ly + lz * lz)
            c = clear[k, i]
            tau = np.sqrt(4 * c * m_s / max(fn, 1e-6))
            tau = min(max(tau, dt), 5e-3)
            g = min(1.0, dt / tau)
            inv = 1.0 / max(fn, 1e-12)
            t3 = (c * lx * inv, c * ly * inv, c * lz * inv)
            for r in range(3):
                off[k, i, r] += (t3[r] - off[k, i, r]) * g
                body_pos_out[f, r] = pocket[k, i, r] + off[k, i, r]
            fric_out[f] = mu_p * r_s * fn
        if adhesion > 0:
            for q in range(8):
                cx = xpos[b, 0] + (xmat[b, 0] * corners[q, 0] + xmat[b, 1] * corners[q, 1] + xmat[b, 2] * corners[q, 2]) * half[k]
                cy = xpos[b, 1] + (xmat[b, 3] * corners[q, 0] + xmat[b, 4] * corners[q, 1] + xmat[b, 5] * corners[q, 2]) * half[k]
                cz = xpos[b, 2] + (xmat[b, 6] * corners[q, 0] + xmat[b, 7] * corners[q, 1] + xmat[b, 8] * corners[q, 2]) * half[k]
                fz = adhesion * min(max(1 - cz / adh_range, 0.0), 1.0)
                if fz > 0:
                    dx = cx - xipos[b, 0]
                    dy = cy - xipos[b, 1]
                    xfrc[b, 2] -= fz
                    # torque r x (0, 0, -fz)
                    xfrc[b, 3] += -dy * fz
                    xfrc[b, 4] += dx * fz


class Sim:
    """A scene of RW cubes with the magnetic, pocket, adhesion, motor and pawl models stepped together."""

    def __init__(self, d: Design, cubes: list, var: Variation, dt: float = 2e-5, floor=True):
        self.d, self.var, self.dt0 = d, var, dt
        self.der = der = derive(d)
        self.cubes = cubes
        self.n = len(cubes)
        self.m = mujoco.MjModel.from_xml_string(build_xml(d, der, cubes, var, dt, floor))
        self.data = mujoco.MjData(self.m)
        m = self.m
        bid = lambda nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        jid = lambda nm: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
        self.cube_b = np.array([bid(f"c{k}") for k in range(self.n)])
        self.sph_b = np.array([[bid(f"c{k}_s{i}") for i in range(8)] for k in range(self.n)])
        self.rot_b = {(k, a): bid(f"c{k}_r{a}") for k in range(self.n) for a in "xy"}
        self.rot_j = {(k, a): jid(f"c{k}_r{a}") for k in range(self.n) for a in "xy"}
        self.rot_dof = {key: m.jnt_dofadr[j] for key, j in self.rot_j.items()}
        self.rot_qpos = {key: m.jnt_qposadr[j] for key, j in self.rot_j.items()}
        self.eq = {(k, a): mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"c{k}_lock{a}") for k in range(self.n) for a in "xy"}
        self.sph_dof = np.array([[m.jnt_dofadr[jid(f"c{k}_s{i}")] for i in range(8)] for k in range(self.n)])
        self.sph_qpos = np.array([[m.jnt_qposadr[jid(f"c{k}_s{i}")] for i in range(8)] for k in range(self.n)])
        self.floor_g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # pocket centres (local, cube frame) and clearances
        self.pocket = np.array([[m.body_pos[self.sph_b[k, i]].copy() for i in range(8)] for k in range(self.n)])
        self.clear = var.clearance.copy()
        self.sph_off = np.zeros((self.n, 8, 3))
        # dipoles: 8 spheres per cube, then 2 rotor cores per cube
        s_scale = math.sqrt(var.derate)
        c_scale = math.sqrt(var.derate_cyl)
        bodies, lpos, laxis, mom = [], [], [], []
        for k in range(self.n):
            for i in range(8):
                bodies.append(self.sph_b[k, i]); lpos.append(np.zeros(3)); laxis.append([0, 0, 1.0])
                mom.append(der.sphere_moment * s_scale * var.sphere_strength[k, i])
        self.n_sph = len(bodies)
        for k in range(self.n):
            for a in "xy":
                bodies.append(self.rot_b[(k, a)]); lpos.append(np.zeros(3))
                laxis.append([0, 1.0, 0] if a == "x" else [0, 0, 1.0])  # diametric: perpendicular to the hinge axis
                mom.append(der.core_moment * c_scale)
        self.dip = DipoleCoupling(m, self.data, bodies, lpos, laxis, mom)
        self.sph_flat_b = np.array(bodies[: self.n_sph])
        self.sph_flat_dof = self.sph_dof.reshape(-1)
        self.motor_on = {}  # (k, axis) -> target omega (motor drives toward it) ; unused in trials (rotor starts at speed)
        self.pawl = None  # active catch: dict
        self.t_catch_done = None
        self.xfrc = np.zeros((m.nbody, 6))
        self.size = var.size
        self.half = var.size / 2
        # lock all rotors at their sampled angles
        for k in range(self.n):
            for j, a in enumerate("xy"):
                self.lock(k, a, var.rotor_angle[k, j])
        mujoco.mj_forward(m, self.data)
        self.start_penetration = float(max([0.0] + [-self.data.contact[i].dist for i in range(self.data.ncon)]))
        self._align_spheres()
        self.energy_log = []
        nd = len(self.dip.bodies)
        self._F = np.zeros((nd, 3))
        self._B = np.zeros((nd, 3))
        self._bpos = np.zeros((self.n * 8, 3))
        self._fric = np.zeros(self.n * 8)
        self._sph_flat = self.sph_b.reshape(-1)
        self._fric_dofs = (self.sph_dof.reshape(-1)[:, None] + np.arange(3)[None, :]).reshape(-1)
        self.dip.local_axis = np.ascontiguousarray(self.dip.local_axis)

    # ------------------------------------------------------------------ helpers
    def lock(self, k, a, angle=None):
        qa = self.rot_qpos[(k, a)]
        if angle is None:
            angle = self.data.qpos[qa]
        else:
            self.data.qpos[qa] = angle
        e = self.eq[(k, a)]
        self.m.eq_data[e, 0] = angle
        self.data.eq_active[e] = 1

    def unlock(self, k, a):
        self.data.eq_active[self.eq[(k, a)]] = 0

    def _align_spheres(self, iters=6):
        """Start orientation of the spheres: pairs that face each other across a bond (closer than d0 + 0.3 mm)
        point head to tail along their axis, as a formed bond leaves them; every other sphere turns to its
        local field (minimum energy start)."""
        d = self.data
        mujoco.mj_kinematics(self.m, d)
        P = d.xpos[self.sph_flat_b]
        owner = np.repeat(np.arange(self.n), 8)
        d0 = 2 * (self.der.sphere_r + self.d.cover) + 2 * self.d.clearance + 2 * PLACE_GAP + 0.3e-3
        fixed = set()
        for f in range(len(P)):
            if f in fixed:
                continue
            dist = np.linalg.norm(P - P[f], axis=1)
            dist[owner == owner[f]] = np.inf
            for g in list(fixed):
                dist[g] = np.inf
            g = int(np.argmin(dist))
            if dist[g] < d0:
                u = (P[g] - P[f]) / dist[g]
                for h in (f, g):
                    k, i = divmod(h, 8)
                    Rc = d.xmat[self.cube_b[k]].reshape(3, 3)
                    q = R.align_vectors([Rc.T @ u], [[0, 0, 1.0]])[0].as_quat()
                    d.qpos[self.sph_qpos[k, i]: self.sph_qpos[k, i] + 4] = [q[3], q[0], q[1], q[2]]
                fixed.update((f, g))
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, d)
            self.dip.compute()
            B = self.dip.B[: self.n_sph]
            for f, b in enumerate(self.sph_flat_b):
                if f in fixed:
                    continue
                k, i = divmod(f, 8)
                Rc = d.xmat[self.cube_b[k]].reshape(3, 3)
                bl = Rc.T @ B[f]
                if np.linalg.norm(bl) < 1e-9:
                    continue
                q = R.align_vectors([bl / np.linalg.norm(bl)], [[0, 0, 1.0]])[0].as_quat()
                d.qpos[self.sph_qpos[k, i]: self.sph_qpos[k, i] + 4] = [q[3], q[0], q[1], q[2]]
        mujoco.mj_forward(self.m, d)

    def pose(self, k):
        return self.data.xpos[self.cube_b[k]].copy(), self.data.xmat[self.cube_b[k]].reshape(3, 3).copy()

    # ------------------------------------------------------------------ one step
    def step(self):
        m, d, var = self.m, self.data, self.var
        xf = self.xfrc
        xf[:] = 0.0
        dp = self.dip
        _dipole_kernel(d.xpos, d.xmat, d.xipos, dp.bodies, dp.local_axis, dp.moment, dp.ii, dp.jj, xf, self._F, self._B)
        if self.d.steel_floor:
            _image_kernel(d.xpos, d.xmat, d.xipos, dp.bodies, dp.local_axis, dp.moment, self.d.steel_k, xf)
        # sphere sits against its pocket wall in the direction of its magnetic force (low-pass); pocket friction; adhesion
        _pocket_adhesion_kernel(d.xpos, d.xmat, d.xipos, self.cube_b, self._F, self.pocket, self.clear, self.sph_off, m.opt.timestep,
                                self.der.sphere_mass, var.mu_pocket, self.der.sphere_r, self.half, var.adhesion, var.adh_range,
                                CORNERS, self._bpos, self._fric, xf)
        m.body_pos[self._sph_flat] = self._bpos
        m.dof_frictionloss[self._fric_dofs] = np.repeat(self._fric, 3)
        d.xfrc_applied[:] = xf
        d.qfrc_applied[:] = 0.0
        if self.pawl is not None:
            self._pawl_torque()
        mujoco.mj_step(m, d)

    # ------------------------------------------------------------------ pawl
    def catch(self, k, a, delay):
        """Engage the pawl of rotor (k, a) after `delay`: it enters the next notch the rotor brings round."""
        I = self.der.rotor_i_axial
        Tc = self.var.catch_T
        kk = I * (math.pi / Tc) ** 2
        c = 2 * self.var.catch_zeta * math.sqrt(kk * I)
        self.pawl = {"k": k, "a": a, "t_engage": self.data.time + delay, "kk": kk, "c": c, "wall": None, "t_hit": None, "Tc": Tc}

    def _pawl_torque(self):
        p = self.pawl
        d = self.data
        if d.time < p["t_engage"]:
            return
        key = (p["k"], p["a"])
        if p["Tc"] > BRAKE_ABOVE:
            # friction brake: constant torque against the relative speed, sized to stop the rotor in Tc
            w = d.qvel[self.rot_dof[key]]
            if "tau_b" not in p:
                p["tau_b"] = self.der.rotor_i_axial * abs(w) / p["Tc"]
                p["w0"] = abs(w)
                p["t_hit"] = d.time
            chatter = 2 * p["w0"] * self.m.opt.timestep / p["Tc"]
            if abs(w) < max(chatter, 1.0) or d.time > p["t_hit"] + 3 * p["Tc"]:
                self.lock(p["k"], p["a"])
                self.pawl = None
                self.t_catch_done = d.time
                return
            d.qfrc_applied[self.rot_dof[key]] = -math.copysign(p["tau_b"], w)
            return
        th = d.qpos[self.rot_qpos[key]]
        w = d.qvel[self.rot_dof[key]]
        n = self.d.teeth
        if p["wall"] is None:
            # the next notch wall the rotor reaches in its direction of travel
            s = 1.0 if w >= 0 else -1.0
            step = 2 * math.pi / n
            p["sign"] = s
            p["wall"] = (math.floor(th / step) + 1) * step if s > 0 else (math.ceil(th / step) - 1) * step
            # fine steps through the stop, and contacts stiffer than the stop (real resin on tile is far stiffer still)
            self.m.opt.timestep = min(self.dt0, p["Tc"] / 30)
            self._solref0 = self.m.geom_solref[:, 0].copy()
            self.m.geom_solref[:, 0] = np.minimum(self._solref0, max(4 * self.m.opt.timestep, p["Tc"] / 5))
        s = p["sign"]
        delta = s * (th - p["wall"])  # penetration into the front wall
        back = s * ((p["wall"] - s * self.d.backlash) - th)  # penetration into the back wall
        if not p.get("in_notch") and back <= 0:
            p["in_notch"] = True  # the notch has come round under the pawl: it drops in, both walls now act
        if not p.get("in_notch"):
            back = -1.0
        tq = 0.0
        if delta > 0:
            f = p["kk"] * delta + p["c"] * s * w
            tq = -s * max(f, 0.0)
            if p["t_hit"] is None:
                p["t_hit"] = d.time
        elif back > 0:
            f = p["kk"] * back - p["c"] * s * w
            tq = s * max(f, 0.0)
        d.qfrc_applied[self.rot_dof[key]] = tq
        if p["t_hit"] is not None and d.time > p["t_hit"] + 12 * p["Tc"]:
            self.lock(p["k"], p["a"])
            self.m.opt.timestep = self.dt0
            self.m.geom_solref[:, 0] = self._solref0
            self.pawl = None
            self.t_catch_done = d.time

    # ------------------------------------------------------------------ run helpers
    def run(self, seconds, callback=None, every=None):
        t_end = self.data.time + seconds
        nxt = self.data.time
        while self.data.time < t_end:
            self.step()
            if callback is not None and self.data.time >= nxt:
                callback(self)
                nxt += every
            if not np.isfinite(self.data.qacc).all():
                raise FloatingPointError("simulation diverged")

    def speeds(self):
        """Largest linear (m/s) and angular (rad/s) speed among cube bodies."""
        v = self.data.cvel[self.cube_b]
        return float(np.abs(v[:, 3:]).max()), float(np.abs(v[:, :3]).max())

    def run_until_rest(self, t_min, t_max, v_tol=2e-3, w_tol=0.5, hold=0.02, callback=None, every=None):
        t0 = self.data.time
        quiet_since = None
        nxt = self.data.time
        while self.data.time - t0 < t_max:
            self.step()
            if callback is not None and self.data.time >= nxt:
                callback(self)
                nxt += every
            if not np.isfinite(self.data.qacc).all():
                raise FloatingPointError("simulation diverged")
            if self.data.time - t0 < t_min:
                continue
            v, w = self.speeds()
            if v < v_tol and w < w_tol:
                quiet_since = quiet_since or self.data.time
                if self.data.time - quiet_since > hold:
                    return True
            else:
                quiet_since = None
        return False

    def spin(self, k, a, omega):
        """Release rotor (k, a) and give it speed omega relative to the cube (spin-up done quasi-statically)."""
        self.unlock(k, a)
        self.data.qvel[self.rot_dof[(k, a)]] = omega
        mujoco.mj_forward(self.m, self.data)

    def composite(self, k):
        """Mass, centre of mass (world) and inertia (world, about that centre) of cube k with its rotors and spheres."""
        d = self.data
        mujoco.mj_forward(self.m, d)
        root = self.cube_b[k]
        bodies = [b for b in range(self.m.nbody) if b == root or self.m.body_parentid[b] == root]
        ms = self.m.body_mass[bodies]
        com = (ms[:, None] * d.xipos[bodies]).sum(0) / ms.sum()
        I = np.zeros((3, 3))
        for b, mb in zip(bodies, ms):
            Rb = d.ximat[b].reshape(3, 3)
            Ib = Rb @ np.diag(self.m.body_inertia[b]) @ Rb.T
            r = d.xipos[b] - com
            I += Ib + mb * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        return float(ms.sum()), com, I

    def magnetic_energy_between(self, movers, others, P=None):
        """Potential energy (J) of free-aligned sphere pairs between two sphere sets (dipole indices), with the
        nominal (derated) moments the controller assumes, not the trial's sampled ones."""
        P = self.dip.P if P is None else P
        mom = self.der.sphere_moment * math.sqrt(K_SPHERE)
        dd = np.linalg.norm(P[np.asarray(movers)][:, None, :] - P[np.asarray(others)][None, :, :], axis=2)
        return float(-(MU0 * mom * mom / (2 * math.pi * dd**3)).sum())


# ---------------------------------------------------------------------------------------------
# Scenes for the move primitives (all moves go toward +x unless stated; roll axis +y)
# ---------------------------------------------------------------------------------------------

FACE_NAMES = {(0, 1): "+x", (0, -1): "-x", (1, 1): "+y", (1, -1): "-y", (2, 1): "+z", (2, -1): "-z"}


def face_down(rot: np.ndarray) -> str:
    """Name of the cube's local face that points down (local frame: rotor X at +x, rotor Y at +y, cells at -z)."""
    j = int(np.argmax(np.abs(rot[2, :])))
    return FACE_NAMES[(j, -int(np.sign(rot[2, j])))]


def rotor_along(rot: np.ndarray, world_axis: np.ndarray):
    """Which rotor ('x' or 'y') of a cube with orientation rot has its axis along world_axis, and the sign."""
    for j, a in enumerate("xy"):
        v = rot[:, j]
        dot = float(np.dot(v, world_axis))
        if abs(dot) > 0.99:
            return a, (1.0 if dot > 0 else -1.0)
    return None, 0.0


def orientations_for_axis(world_axis, vertical_rotor=False):
    """Cube orientations with a rotor along world_axis (or, for yaw, along world z)."""
    ax = np.array([0, 0, 1.0]) if vertical_rotor else np.asarray(world_axis, float)
    return [Rm for Rm in ROTS if rotor_along(Rm, ax)[0] is not None]


SCENES = {
    # name: (cells of the cubes, mover index 0 first, target cell, roll axis sign (+1: toward +x), structure)
    "roll_floor": ([(0, 0, 0)], (1, 0, 0), +1),
    "roll_top": ([(0, 0, 1), (0, 0, 0), (1, 0, 0), (2, 0, 0)], (1, 0, 1), +1),
    "climb_lone": ([(0, 0, 0), (1, 0, 0)], (1, 0, 1), +1),
    "climb_row": ([(0, 0, 0), (1, 0, 0), (2, 0, 0)], (1, 0, 1), +1),
    "descend_lone": ([(1, 0, 1), (1, 0, 0)], (0, 0, 0), -1),
    "descend_row": ([(1, 0, 1), (1, 0, 0), (2, 0, 0)], (0, 0, 0), -1),
    "split_pair": ([(0, 0, 0), (1, 0, 0)], (-1, 0, 0), -1),
    "split_row": ([(0, 0, 0), (1, 0, 0), (2, 0, 0)], (-1, 0, 0), -1),
    "dock": ([(0, 0, 0), (2, 0, 0)], (1, 0, 0), +1),
}

MOVE_KIND = {"roll_floor": "roll", "roll_top": "top", "climb_lone": "climb", "climb_row": "climb", "descend_lone": "descend",
             "descend_row": "descend", "split_pair": "split", "split_row": "split", "dock": "roll"}

RETRYABLE = {"stayed", "fell_back", "held"}
BRAKE_ABOVE = 1.5e-3  # catch durations above this are a friction brake (constant torque), below a pawl in a notch
DISTURBED = 0.1  # L: a neighbour moved this far is knocked (as cube.DISTURBED)
TURNED_DEG = 10.0  # a neighbour turned this much is knocked (cube.ALIGNED)


@dataclass
class Trial:
    move: str
    seed: int
    cls: str
    cls_first: str
    attempts: int
    metrics: dict


def pivot_line(scene, cells_world_mover, sign, L):
    """Pivot edge (point on the line, world) of the mover for each scene, in nominal grid metres."""
    i, j, k = cells_world_mover
    if scene.startswith("roll") or scene.startswith("split") or scene == "dock":
        return np.array([(i + 0.5 * sign) * L, 0, k * L])  # bottom edge toward the move
    if scene.startswith("climb"):
        return np.array([(i + 0.5) * L, 0, (k + 1) * L])  # top edge shared with the neighbour
    if scene.startswith("descend"):
        return np.array([(i - 0.5) * L, 0, k * L])  # top edge of the cube below, on the -x side
    raise ValueError(scene)


def impulse_efficiency(mu):
    """Fraction of the no-slip tipping energy left when the floor edge slips during the catch (scaling.py)."""
    i_cm = 1 / 6
    if mu >= 1:
        return 1.0
    w = 1.0 / (i_cm + (1 + mu) / 4)
    vx, vy = -mu * w / 2, -w / 2
    t = 0.5 * (vx**2 + vy**2) + 0.5 * i_cm * w**2
    slide = 0.5 * (w / 2 * (1 - mu)) ** 2
    return (t - slide) / (1 / (2 * (2 / 3)))


def barrier(sim: Sim, scene: str, sign: float, pivot: np.ndarray):
    """Nominal-model energy barrier of the move: gravity plus free-aligned sphere-pair energy between the
    mover's spheres and every other cube's, along the rotation arc about the pivot edge (y axis)."""
    m_tot, com, I = sim.composite(0)
    axis = np.array([0, sign, 0.0])
    sim.dip.compute()
    P0 = sim.dip.P.copy()
    movers = list(range(8))
    others = list(range(8, sim.n_sph))
    total_angle = math.pi if (scene.startswith("climb") or scene.startswith("descend")) else math.pi / 2
    E0 = None
    Emax = -1e9
    for phi in np.linspace(0, total_angle, 91):
        Rr = R.from_rotvec(axis * phi).as_matrix()
        P = P0.copy()
        P[:8] = (Rr @ (P0[:8] - pivot).T).T + pivot
        c = Rr @ (com - pivot) + pivot
        E = m_tot * G * c[2] + (sim.magnetic_energy_between(movers, others, P) if others else 0.0)
        if sim.d.steel_floor:  # each mover sphere with its own image (free-aligned, vertical): -k mu0 m^2 / (4 pi (2h)^3)
            mom = sim.der.sphere_moment * math.sqrt(K_SPHERE)
            h = np.maximum(P[:8, 2], 1e-5)
            E -= float((sim.d.steel_k * MU0 * mom * mom / (4 * math.pi * (2 * h) ** 3)).sum())
        E0 = E if E0 is None else E0
        Emax = max(Emax, E - E0)
    r = com - pivot
    r_perp = r - np.dot(r, axis) * axis
    I_pivot = float(axis @ I @ axis + m_tot * np.dot(r_perp, r_perp))
    return Emax, I_pivot, m_tot


def classify(sim: Sim, scene: str, start: list, target_cell: tuple, mover_rot0: np.ndarray, sign: float):
    """Outcome class of a move from the final scene, and metrics."""
    L = sim.d.L
    pitch = L  # settled structures sit at pitch L (faces touching)
    p, Rm = sim.pose(0)
    cell_f = np.array([p[0] / pitch, p[1] / pitch, (p[2] - L / 2) / pitch])
    cell = np.round(cell_f).astype(int)
    off = np.linalg.norm((cell_f - cell)[:2]) * pitch
    up = np.abs(Rm[2, :]).max()  # a local axis is vertical
    flat = up > math.cos(math.radians(10))
    zok = abs((cell_f[2] - cell[2]) * pitch) < 0.25 * L
    # neighbours disturbed
    moved = []
    for k in range(1, sim.n):
        p0, R0 = start[k]
        pk, Rk = sim.pose(k)
        dist = float(np.linalg.norm(pk - p0))
        ang = float(np.degrees(np.linalg.norm(R.from_matrix(Rk @ R0.T).as_rotvec())))
        moved.append((dist, ang))
    disturbed = any(dist > DISTURBED * L or ang > TURNED_DEG for dist, ang in moved)
    # expected orientation after a quarter (or half) turn
    turns = {"climb": 2, "descend": 2}.get(MOVE_KIND.get(scene, ""), 1)
    axis = np.array([0, sign, 0.0])
    R_exp = R.from_rotvec(axis * turns * math.pi / 2).as_matrix() @ mover_rot0
    ang_exp = float(np.degrees(np.linalg.norm(R.from_matrix(Rm @ R_exp.T).as_rotvec())))
    ang_start = float(np.degrees(np.linalg.norm(R.from_matrix(Rm @ mover_rot0.T).as_rotvec())))
    start_cell = np.round(np.array([start[0][0][0] / pitch, start[0][0][1] / pitch, (start[0][0][2] - L / 2) / pitch])).astype(int)
    on_lattice = flat and off < 0.25 * L and zok
    bonded = bonded_faces(sim, 0)
    metrics = {"cell": cell.tolist(), "offset_mm": off * 1e3, "flat": bool(flat), "ang_target_deg": ang_exp, "ang_start_deg": ang_start,
               "neighbour_moved_mm": [round(m_[0] * 1e3, 4) for m_ in moved], "neighbour_rot_deg": [round(m_[1], 3) for m_ in moved],
               "bonds": bonded}
    kind = MOVE_KIND.get(scene, scene)
    if disturbed:
        return "dislodged", metrics
    tgt = np.array(target_cell)
    if on_lattice and (cell == tgt).all() and ang_exp < 15:
        if kind in ("top", "climb", "descend") and not bonded:
            return "reached_unbonded", metrics
        return "reached", metrics
    if on_lattice and (cell == start_cell).all() and ang_start < 15:
        if kind == "climb":
            return "fell_back", metrics
        if kind == "split":
            return "held", metrics
        return "stayed", metrics
    if on_lattice and (cell == start_cell + 2 * (tgt - start_cell)).all() and kind in ("roll", "top", "split"):
        return "double_roll", metrics
    if kind in ("top", "climb") and cell[2] == 0 and on_lattice:
        return "slid_off" if kind == "top" else "fell_off", metrics
    if kind == "climb" and on_lattice and cell[2] == 1:
        return "overshot", metrics
    if kind == "descend" and on_lattice and cell[2] == 1:
        return "stayed", metrics
    if on_lattice:
        return "strayed", metrics
    return "off_lattice", metrics


def bonded_faces(sim: Sim, k: int, gap_tol=0.06e-3, lat_tol=0.15):
    """Neighbours whose face is flush against cube k (gap below gap_tol, faces overlapping, aligned)."""
    out = []
    pk, Rk = sim.pose(k)
    L = sim.d.L
    for j in range(sim.n):
        if j == k:
            continue
        pj, Rj = sim.pose(j)
        dlt = Rk.T @ (pj - pk)
        ax = int(np.argmax(np.abs(dlt)))
        gap = abs(dlt[ax]) - (sim.half[k] + sim.half[j])
        lat = np.delete(dlt, ax)
        rel = Rk.T @ Rj
        aligned = np.abs(rel).max(axis=0).min() > math.cos(math.radians(8))
        if gap < gap_tol and np.abs(lat).max() < lat_tol * L and aligned:
            out.append(j)
    return out


def setup_scene(d: Design, scene: str, var: Variation, rng: np.random.Generator, dt=2e-5, rot_index=None):
    cells, target, sign = SCENES[scene]
    axis = np.array([0, sign, 0.0])
    choices = orientations_for_axis(axis)
    Rm = choices[rot_index % len(choices)] if rot_index is not None else choices[rng.integers(len(choices))]
    cubes = [CubeSpec(cells[0], Rm)] + [CubeSpec(c, ROTS[rng.integers(24)]) for c in cells[1:]]
    sim = Sim(d, cubes, var, dt=dt)
    return sim, Rm, target, sign


def run_move(d: Design, scene: str, seed: int, nominal=False, retries=2, escalate=1.12, dt=2e-5, k_override=None,
             rot_index=None, callback=None, every=None, var=None):
    """One Monte Carlo trial of a move with up to `retries` re-tries after a retryable outcome."""
    rng = np.random.default_rng(seed)
    ncube = len(SCENES[scene][0])
    if var is None:
        var = nominal_variation(d, ncube) if nominal else sample_variation(d, ncube, rng)
    sim, Rm, target, sign = setup_scene(d, scene, var, rng, dt=dt, rot_index=rot_index)
    settled = sim.run_until_rest(0.02, 0.15, callback=callback, every=every)
    start = [sim.pose(k) for k in range(sim.n)]
    kind = MOVE_KIND[scene]
    kmul = k_override if k_override is not None else {"roll": d.k_roll, "top": d.k_top, "climb": d.k_climb, "descend": d.k_descend, "split": d.k_split}[kind]
    cls_first = None
    metrics = {"face_down": face_down(Rm), "var": var.asdict_small(), "settled": bool(settled),
               "start_penetration_um": sim.start_penetration * 1e6}
    attempts = 0
    for attempt in range(retries + 1):
        attempts += 1
        p0, R0 = sim.pose(0)
        pivot = pivot_line(scene, tuple(np.round(np.array([p0[0], p0[1], p0[2] - d.L / 2]) / d.L).astype(int)), sign, d.L)
        Eb, I_piv, m_tot = barrier(sim, scene, sign, pivot)
        eta = impulse_efficiency(0.4) if kind in ("roll", "split") else 1.0
        H = kmul * (escalate ** attempt) * math.sqrt(2 * I_piv * max(Eb, 0.0) / eta)
        a, s = rotor_along(R0, np.array([0, sign, 0.0]))
        if a is None:
            cls, metrics = "no_rotor_along_axis", {}
            break
        omega_cmd = s * H / sim.der.rotor_i_axial
        omega = omega_cmd * (1 + var.speed_err)
        metrics.update({"H": H, "omega_cmd": abs(omega_cmd), "E_barrier_uJ": Eb * 1e6, "I_pivot": I_piv, "rotor": a})
        if abs(omega_cmd) > d.omega_cap:
            cls = "over_speed_limit"
            break
        sim.spin(0, a, omega)
        sim.catch(0, a, var.catch_delay)
        t_max = {"roll": 0.6, "top": 0.6, "split": 0.6, "climb": 0.8, "descend": 0.8}[kind]
        rest = sim.run_until_rest(0.06, t_max, callback=callback, every=every)
        cls, m2 = classify(sim, scene, start, target, Rm, sign)
        metrics.update(m2)
        metrics["at_rest"] = rest
        if cls_first is None:
            cls_first = cls
        if cls not in RETRYABLE:
            break
    return Trial(scene, seed, cls, cls_first or cls, attempts, metrics)


SUCCESS = {"reached"}
HARMFUL = {"dislodged"}


# ---------------------------------------------------------------------------------------------
# YAW, BOND, FALL
# ---------------------------------------------------------------------------------------------

def run_yaw(d: Design, seed: int, nominal=False, retries=2, dt=2e-5, callback=None, every=None, mu_nominal=0.4):
    """Heading trim on a face with a vertical rotor: target yaw_target_deg, open loop on the nominal friction,
    then up to `retries` corrections that use the measured response to update the friction estimate."""
    rng = np.random.default_rng(seed)
    var = nominal_variation(d, 1) if nominal else sample_variation(d, 1, rng)
    choices = orientations_for_axis(None, vertical_rotor=True)
    Rm = choices[rng.integers(len(choices))]
    sim = Sim(d, [CubeSpec((0, 0, 0), Rm)], var, dt=dt)
    sim.run_until_rest(0.01, 0.05, callback=callback, every=every)
    a, s = rotor_along(Rm, np.array([0, 0, 1.0]))
    target = math.radians(d.yaw_target_deg)
    m_tot, com, I = sim.composite(0)
    Iz = I[2, 2]
    # friction torque model of the controller: 4 corner feet at L / sqrt2 from the centre
    tau_est = mu_nominal * m_tot * G * d.L / math.sqrt(2)
    yaw0 = _yaw(sim)
    remaining = target
    cls_first = None
    errs = []
    attempts = 0
    for attempt in range(retries + 1):
        attempts += 1
        H = math.copysign(math.sqrt(2 * Iz * tau_est * abs(remaining)), remaining)
        omega = s * H / sim.der.rotor_i_axial * (1 + var.speed_err)
        before = _yaw(sim)
        sim.spin(0, a, omega)
        sim.catch(0, a, var.catch_delay)
        sim.run_until_rest(0.02, 0.3, callback=callback, every=every)
        turned = _wrap(_yaw(sim) - before)
        if abs(turned) > math.radians(0.5) and abs(H) > 0:
            # theta = H^2 / (2 Iz tau): update the friction estimate from the response, within 4x of nominal
            tau_nom = mu_nominal * m_tot * G * d.L / math.sqrt(2)
            tau_est = float(np.clip(H * H / (2 * Iz * abs(turned)), 0.25 * tau_nom, 4 * tau_nom))
        err = _wrap(_yaw(sim) - yaw0 - target)
        errs.append(math.degrees(err))
        remaining = -err
        ok = abs(math.degrees(err)) <= 3.0
        flat = np.abs(sim.pose(0)[1][2, :]).max() > math.cos(math.radians(5))
        p = sim.pose(0)[0]
        cls = "within_3deg" if ok and flat else ("tipped" if not flat else "not_within_3deg")
        if cls_first is None:
            cls_first = cls
        if cls != "not_within_3deg":
            break
    return Trial("yaw", seed, cls, cls_first, attempts, {"errors_deg": errs, "translation_mm": float(np.linalg.norm(p[:2]) * 1e3),
                                                          "face_down": face_down(Rm), "var": var.asdict_small()})


def _yaw(sim):
    Rm = sim.pose(0)[1]
    # heading of the horizontal local axes: use the local axis mapped closest to world x
    j = int(np.argmax(np.abs(Rm[0, :])))
    v = Rm[:, j] * np.sign(Rm[0, j])
    return math.atan2(v[1], v[0])


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


BOND_RANGES = {"gap": (0.0, 0.25), "lateral": (-0.15, 0.15), "yaw_deg": (-8.0, 8.0)}  # fractions of L


def run_bond(d: Design, seed: int, nominal=False, dt=2e-5, callback=None, every=None, gap=None, lateral=None, yaw=None):
    """Passive capture: a cube released at rest near a neighbour's face (gap, lateral offset, yaw error
    drawn from BOND_RANGES, the spread a floor roll leaves). Closed: flush, aligned face bond."""
    rng = np.random.default_rng(seed)
    var = nominal_variation(d, 2) if nominal else sample_variation(d, 2, rng)
    g = rng.uniform(*BOND_RANGES["gap"]) if gap is None else gap
    lat = rng.uniform(*BOND_RANGES["lateral"]) if lateral is None else lateral
    yw = rng.uniform(*BOND_RANGES["yaw_deg"]) if yaw is None else yaw
    R0 = ROTS[rng.integers(24)]
    R1 = ROTS[rng.integers(24)]
    cubes = [CubeSpec((0, 0, 0), R.from_euler("z", yw, degrees=True).as_matrix() @ R0), CubeSpec((1, 0, 0), R1)]
    var.place_jitter[0, :3] += np.array([-(g * d.L), lat * d.L, 0.0])  # face gap is g L plus PLACE_GAP
    sim = Sim(d, cubes, var, dt=dt)
    start = [sim.pose(k) for k in range(2)]
    sim.run_until_rest(0.03, 0.4, callback=callback, every=every)
    bonded = bonded_faces(sim, 0)
    p1, R1f = sim.pose(1)
    moved = float(np.linalg.norm(p1 - start[1][0]))
    cls = "closed" if 1 in bonded else "gap_left"
    if cls == "closed":
        Rm = sim.pose(0)[1]
        flat = np.abs(Rm[2, :]).max() > math.cos(math.radians(5))
        if not flat:
            cls = "closed_tilted"
    return Trial("bond", seed, cls, cls, 1, {"gap_mm": g * d.L * 1e3 + PLACE_GAP * 1e3, "gap_frac": g, "lateral_frac": lat, "yaw_deg": yw, "neighbour_moved_mm": moved * 1e3,
                                             "var": var.asdict_small()})


def hertz_peak(m, v, rc, e_star=2.8e9):
    """Peak force and deceleration of a sphere-on-flat Hertz impact (corner or edge radius rc)."""
    delta = (15 * m * v * v / (16 * e_star * math.sqrt(rc))) ** 0.4
    f = 4 / 3 * e_star * math.sqrt(rc) * delta**1.5
    return f, f / m


def run_fall(d: Design, seed: int, pair=True, dt=1e-5, height=0.20, callback=None, every=None, nominal=False):
    """20 cm fall onto a hard floor: a bonded pair (or a lone cube) released just above the floor with the
    free-fall speed, random orientation and a small random spin. Reports whether the bond survived, whether
    any sphere left its pocket (it cannot here: pockets are closed), and the Hertz peak stresses."""
    rng = np.random.default_rng(seed)
    n = 2 if pair else 1
    var = nominal_variation(d, n) if nominal else sample_variation(d, n, rng)
    cubes = [CubeSpec((0, 0, 0), np.eye(3))] + ([CubeSpec((1, 0, 0), ROTS[rng.integers(24)])] if pair else [])
    sim = Sim(d, cubes, var, dt=dt, floor=True)
    # settle bonded in the air first (no floor contact): lift, no gravity
    sim.m.opt.gravity[:] = 0
    lift = 0.03
    for k in range(n):
        sim.data.qpos[sim.m.jnt_qposadr[sim.m.body_jntadr[sim.cube_b[k]]] + 2] += lift
    mujoco.mj_forward(sim.m, sim.data)
    sim.run(0.01)
    bonded0 = 1 in bonded_faces(sim, 0) if pair else True
    # random orientation of the whole group about its centre, then drop from just above the floor
    Rr = R.random(random_state=rng.integers(1 << 31))
    qadr = [sim.m.jnt_qposadr[sim.m.body_jntadr[sim.cube_b[k]]] for k in range(n)]
    cen = np.mean([sim.data.qpos[a: a + 3] for a in qadr], axis=0)
    for a in qadr:
        p = sim.data.qpos[a: a + 3] - cen
        q = sim.data.qpos[a + 3: a + 7]
        Rq = R.from_quat([q[1], q[2], q[3], q[0]])
        Rn = Rr * Rq
        sim.data.qpos[a: a + 3] = Rr.apply(p)
        qq = Rn.as_quat()
        sim.data.qpos[a + 3: a + 7] = [qq[3], qq[0], qq[1], qq[2]]
    mujoco.mj_forward(sim.m, sim.data)
    # lowest corner 0.2 mm above the floor
    lowest = min((sim.data.xpos[sim.cube_b[k]] + (sim.data.xmat[sim.cube_b[k]].reshape(3, 3) @ (CORNERS * sim.half[k]).T).T)[:, 2].min() for k in range(n))
    for a in qadr:
        sim.data.qpos[a + 2] -= lowest - 0.2e-3
    v = math.sqrt(2 * G * height)
    w = rng.normal(0, 5.0, size=3)  # rad/s tumble picked up in the fall
    for k, a in enumerate(qadr):
        dofs = sim.m.jnt_dofadr[sim.m.body_jntadr[sim.cube_b[k]]]
        sim.data.qvel[dofs: dofs + 3] = [0, 0, -v]
        sim.data.qvel[dofs + 3: dofs + 6] = w
    sim.m.opt.gravity[:] = [0, 0, -G]
    mujoco.mj_forward(sim.m, sim.data)
    peak_contact = 0.0

    def watch(s):
        nonlocal peak_contact
        f6 = np.zeros(6)
        for i in range(s.data.ncon):
            mujoco.mj_contactForce(s.m, s.data, i, f6)
            peak_contact = max(peak_contact, f6[0])
        if callback is not None:
            callback(s)

    sim.run_until_rest(0.05, 0.6, callback=watch, every=dt)
    bonded1 = (1 in bonded_faces(sim, 0)) if pair else True
    der = sim.der
    # what hit first: corner (worst), Hertz on the corner radius 0.05 L, as scaling.fall
    f_h, a_h = hertz_peak(der.m_total, v, max(0.05 * d.L, 0.1e-3))
    pin_area = max(0.3 * d.rotor_r * d.rotor_t, 0.02e-6)
    pin_stress = der.rotor_mass * a_h / pin_area
    # sphere on its 0.1 mm cover: the sphere's inertial load over the contact patch of the cover (A: 0.2 r)
    sph_load = der.sphere_mass * a_h
    if pair:
        cls = "bond_held" if (bonded0 and bonded1) else "bond_broke"
    else:
        cls = "intact"
    return Trial("fall_pair" if pair else "fall_lone", seed, cls, cls, 1, {
        "bonded_before": bonded0, "bonded_after": bonded1, "v_impact": v, "hertz_peak_g": a_h / G,
        "pin_stress_MPa": pin_stress / 1e6, "pin_allowable_MPa": 40.0, "sphere_inertial_load_N": sph_load,
        "sphere_magnetic_pull_N": K_SPHERE * 3 * MU0 * der.sphere_moment**2 / (2 * math.pi * (2 * (der.sphere_r + d.cover)) ** 4),
        "mujoco_peak_contact_N": peak_contact})
