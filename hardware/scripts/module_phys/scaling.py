"""Analytic scaling model for the smallest self-powered modular cube (Beebot, 2026-11-15).

For each actuation architecture, every quantity is a function of the cube edge L: mass (parts it
carries plus shell), the torques, forces and energies each behaviour needs, what the actuators,
bonds and storage give, and the packing. An architecture passes at L when every margin
(available / required) is at least MARGIN (1.5). The minimum L is the smallest grid value at which
it passes and keeps passing for the next few grid steps.

Run: .venv/bin/python hardware/scripts/module_phys/scaling.py   (CPU, a few seconds)
Writes hardware/runs/module_phys/scaling.json, hardware/runs/module_phys/scaling_margins.png, prints the tables.

Status tags on inputs (kept next to each constant): V = read on a vendor page or published
source this session, E = estimated from physics or a stated reference, A = assumed (a choice or a
number from memory, to be measured). Part entries come from hardware/runs/module_phys/parts.json where it
has them; parts.json marks almost all of them unverified.

What this model is not: it is quasi-static and lumped. The magnet forces use exact dipole (spheres)
and exact Coulombian coaxial (cylinders) formulas, calibrated against three published pull values;
everything off-axis, multi-body or dynamic belongs to hardware/src/beebot/hardware/mmsim (magpylib, MuJoCo).
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import integrate, special

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "hardware" / "runs" / "module_phys"
PARTS = OUT / "parts.json"

G = 9.81
MU0 = 4e-7 * math.pi
RHO_CU = 1.72e-8  # ohm m (V, handbook)
CV_CU = 3.45e6  # J/(m3 K), volumetric heat capacity of copper (V, handbook)
GAMMA_WATER = 0.072  # N/m (V, handbook)
MARGIN = 1.5
BR = {"N52": 1.45, "N45": 1.35, "N42": 1.30}  # T, mid-range of grade spec (V, grade tables)
RHO = {"ndfeb": 7500.0, "brass": 8500.0, "tungsten": 17500.0, "resin": 1200.0, "pa12": 1010.0, "pla": 1240.0, "flex": 3500.0}  # kg/m3; tungsten = heavy alloy 90W (E); flex = 2-layer 1 oz polyimide, copper share (E)

# Requirements (scaled to L through the cube mass m and L)
N_CANTILEVER = 3  # cubes held out horizontally from one bond (spec 4.1 reach)
N_HANG = 5  # cubes hanging from one bond (spec 5.5)
DYNAMIC = 2.0  # load factor on static bond loads (spec 5.5 uses 2-2.5) (A)
ARM_TORQUE_50 = 0.42  # N m, arm lift joint at 50 mm / 120 g, continuous (measured in module_b demo 11)
ARM_REF = 0.120 * 0.050  # kg m, m L of that module
MU_FLOOR = 0.5  # cube on a bare floor (A; MuJoCo demos used 0.5)
MU_LOW, MU_HIGH = 0.3, 0.8  # friction spread for robustness checks (A)
FALL_HEIGHT = 0.20  # m (user requirement)
LONE_MINUTES = 2.0  # minutes a lone cube must run (user: "a short while (minutes)")
LONE_MOVE_PERIOD = 4.0  # s, one move every 4 s during a lone run (A)
P_IDLE = 1.0e-3  # W, MCU duty-cycled + IR listen + IMU (A; CH32V003 at 8 MHz with sleep)
POS_ERROR = 0.10  # fraction of L: placement error a dock must capture (A)
MIN_SPEED = 0.10  # body lengths per second, lone translation (A, soft gate)
OMEGA_MAX_PIN = 6000.0  # rad/s (57k rpm): pin-in-bushing flywheel bearing limit (A)
SPIN_TIME_MAX = 2.0  # s to spin a flywheel up (A)
MOTOR_EFF = 0.3  # electrical to flywheel kinetic energy at mm scale (A)
# Packing: the margin is (inner volume x best-case fill) / sum of part envelopes, and like every
# margin it must reach 1.5, so the effective fill demanded is FILL / 1.5. Best-case fill of
# rectilinear parts on a folded flex (phone-class packaging reaches about 0.8) gives 0.53 effective;
# hand-assembled hobby parts (round motors in square pockets pi/4, 0.3 mm FDM clearances, wiring,
# screw bosses) best case 0.65 gives 0.43 effective. Layout C's check found 21 cm3 of 97 cm3 used
# with gear trains, wires and connectors not counted (module-pairs.md), consistent with these (E).
FILL_FLEX = 0.80
FILL_HOBBY = 0.65
PAWL_EFF = 1.0
ROTOR_SHARE = 0.40  # the two flywheels may be up to 40 % of the cube mass (M-Blocks-like; A, an optimiser variable)  # angular momentum is conserved through a pawl stop (E, exact for a rigid catch)


# ----------------------------------------------------------------------------------------------
# Magnets: exact dipole for spheres, exact Coulombian coaxial formula for cylinders.
# ----------------------------------------------------------------------------------------------

def sphere_moment(r, br):
    return br / MU0 * 4.0 / 3.0 * math.pi * r**3


def sphere_pair_force(r, d, br=BR["N52"]):
    """Attraction (N) between two equal uniformly magnetised spheres, centres d apart, moments free
    to align head to tail along the line joining them (the minimum-energy state of free spheres).
    The external field of a uniform sphere is exactly a dipole, so this is exact for ideal magnets."""
    m = sphere_moment(r, br)
    return 3.0 * MU0 * m * m / (2.0 * math.pi * d**4)


def sphere_steel_force(r, gap, br):
    """Sphere on a thick, unsaturated steel plate: image dipole at 2 (r + gap)."""
    return sphere_pair_force(r, 2.0 * (r + gap), br)


def _f_coax(radius, z, br1, br2):
    """Force (N) between two coaxial uniformly charged discs (Coulombian surface charge Br/mu0),
    radius R, separation z: pi Br1 Br2 R^2 / mu0 * int_0^inf J1(x)^2 exp(-x z/R) / x dx."""
    pref = math.pi * br1 * br2 * radius**2 / MU0
    if z <= 0:
        return pref * 0.5
    a = z / radius
    upper = min(4000.0, 60.0 / a)
    val, _ = integrate.quad(lambda x: special.j1(x) ** 2 * math.exp(-a * x) / x, 0.0, upper, limit=4000)
    tail = math.exp(-a * upper) / (math.pi * upper)  # J1^2 ~ 1/(pi x) on average
    return pref * (val + tail)


def cylinder_pair_force(radius, t1, t2, gap, br1=BR["N52"], br2=BR["N52"]):
    """Attraction (N) between two coaxial axially magnetised cylinders, end faces gap apart."""
    f = lambda z: _f_coax(radius, z, br1, br2)
    return f(gap) - f(gap + t1) - f(gap + t2) + f(gap + t1 + t2)


def cylinder_steel_force(radius, t, gap, br):
    """Cylinder on thick steel, image method: a twin at 2 gap."""
    return cylinder_pair_force(radius, t, t, 2.0 * gap, br, br)


# Published pull values (case 1 / adhesive force: magnet on a thick steel plate, direct contact).
CALIBRATION = [
    # (name, kind, dims (m), grade, published N, source)
    ("K&J D22-N52 cylinder 1/8 x 1/8 in", "cyl", (1.5875e-3, 3.175e-3), "N52", 0.84 * 4.448, "K&J case 1 0.84 lb via albmaterials/K&J search snippet 2026-11-15 (V, secondary)"),
    ("supermagnete S-02-0.7-N disc 2 x 0.7 mm", "cyl", (1.0e-3, 0.7e-3), "N45", 0.726, "supermagnete product page via search 2026-11-15 (V, secondary)"),
    ("supermagnete S-05-01-N disc 5 x 1 mm", "cyl", (2.5e-3, 1.0e-3), "N45", 3.14, "supermagnete product page via search 2026-11-15 (V, secondary)"),
    ("supermagnete K-03-C sphere 3 mm", "sphere", (1.5e-3,), "N42", 1.27, "supermagnete K-03-C page via search 2026-11-15 (V, secondary)"),
]


def calibrate():
    rows = []
    for name, kind, dims, grade, published, source in CALIBRATION:
        if kind == "cyl":
            model = cylinder_steel_force(dims[0], dims[1], 0.0, BR[grade])
        else:
            model = sphere_steel_force(dims[0], 0.0, BR[grade])
        rows.append({"magnet": name, "model_N": round(model, 3), "published_N": round(published, 3), "ratio": round(published / model, 3), "source": source})
    sph = [r["ratio"] for r in rows if "sphere" in r["magnet"]]
    cyl = [r["ratio"] for r in rows if "sphere" not in r["magnet"]]
    return rows, float(np.mean(sph)), float(np.mean(cyl))


CAL_ROWS, K_SPHERE, K_CYL = calibrate()


# ----------------------------------------------------------------------------------------------
# Parts and storage
# ----------------------------------------------------------------------------------------------

def load_parts():
    if not PARTS.exists():
        return {}
    return {p["name"]: p for p in json.loads(PARTS.read_text())}


PARTS_DB = load_parts()


def part_price(name, fallback):
    p = PARTS_DB.get(name)
    if p and p.get("price_usd_unit_at_10"):
        return p["price_usd_unit_at_10"]
    return fallback


@dataclass
class Storage:
    name: str
    dims: tuple  # mm
    energy_j: float  # usable
    p_cont_w: float  # continuous
    p_peak_w: float  # pulse (seconds)
    mass_g: float
    cost: tuple  # USD at 10, 1k, 1M units
    status: str


STORAGE = [
    # CeraCharge: 100 uAh at 1.4 V nominal = 0.50 J; 1 mA continuous (10 C), 3 mA for 1 s (V: TDK presentation via search, secondary). 1.4 V: needs a boost to run the MCU.
    Storage("TDK CeraCharge 1812 (100 uAh)", (4.5, 3.2, 1.1), 0.50, 1.4e-3, 4.2e-3, 0.04, (4.0, 1.5, 0.4), "V capacity/current (secondary); price A"),
    # CPH3225A: 11 mF, 3.3 V, ESR ~160 ohm (parts.json, unverified). Usable 3.3 -> 1.8 V: 42 mJ. Matched-load peak V^2/4R = 17 mW; use half.
    Storage("Seiko CPH3225A supercap 11 mF", (3.2, 2.5, 0.9), 0.5 * 0.011 * (3.3**2 - 1.8**2), 8e-3, 8e-3, 0.02, (2.0, 1.2, 0.4), "A (parts.json unverified)"),
    Storage("LiPo 301010 10 mAh", (3.0, 10.0, 10.0), 10e-3 * 3600 * 3.7 * 0.8, 0.074, 0.074, 0.35, (2.0, 0.8, 0.3), "A (parts.json: 1-2 C)"),
    Storage("LiPo 401012 15 mAh", (4.0, 10.0, 12.0), 15e-3 * 3600 * 3.7 * 0.8, 0.11, 0.11, 0.5, (2.0, 0.8, 0.3), "A (parts.json: 1-2 C)"),
    Storage("LiPo 5-10C 50 mAh 601422", (6.0, 14.0, 22.0), 50e-3 * 3600 * 3.7 * 0.8, 1.0, 1.8, 2.0, (3.5, 1.5, 0.5), "A (parts.json: 250-1000 mA claimed)"),
    Storage("LiPo 502030 100 mAh", (5.0, 20.0, 30.0), 100e-3 * 3600 * 3.7 * 0.8, 0.37, 0.7, 2.3, (2.5, 1.0, 0.4), "A (parts.json: 1-2 C, some 5 C)"),
    Storage("LiPo 1S 150 mAh 20C whoop cell", (6.0, 12.0, 30.0), 150e-3 * 3600 * 3.7 * 0.8, 5.0, 10.0, 4.0, (4.0, 2.0, 0.6), "A (not in parts.json; tiny-whoop class)"),
]


def fits(dims, inner):
    """A box fits a cube of side inner (mm) in some orientation with 0.1 mm clearance."""
    return max(dims) <= inner - 0.1


def box_vol(dims):
    return dims[0] * dims[1] * dims[2]


def choose_storage(inner, budget_mm3, e_need, p_need, p_peak):
    """Smallest-volume combination of up to 3 reservoirs of one kind plus up to 4 supercaps that
    gives energy e_need (J), continuous power p_need and pulse power p_peak (W)."""
    best = None
    cph = STORAGE[1]
    for s in STORAGE:
        if s is cph or not fits(s.dims, inner):
            continue
        for n in (0, 1, 2, 3):
            for k in (0, 1, 2, 3, 4):
                if n == 0 and k == 0:
                    continue
                e = n * s.energy_j + k * cph.energy_j
                pc = n * s.p_cont_w + (k * cph.p_cont_w if n == 0 else 0.0)
                pp = n * s.p_peak_w + k * cph.p_peak_w
                vol = n * box_vol(s.dims) + k * box_vol(cph.dims)
                if vol > budget_mm3:
                    continue
                worst = min(e / (MARGIN * e_need), pc / (MARGIN * p_need), pp / (MARGIN * p_peak))
                score = (worst >= 1.0, -vol if worst >= 1.0 else worst)  # smallest feasible, else the most capable
                if best is None or score > best[0]:
                    best = (score, {"reservoir": s.name if n else None, "n": n, "supercaps": k, "energy_j": e, "p_cont_w": pc, "p_peak_w": pp, "vol_mm3": vol, "mass_g": n * s.mass_g + k * cph.mass_g, "cost": tuple(n * a + k * b for a, b in zip(s.cost, cph.cost))})
    if best is None:  # nothing fits: supercaps only, as many as fit
        k = 1 if fits(cph.dims, inner) else 0
        return {"reservoir": None, "n": 0, "supercaps": k, "energy_j": k * cph.energy_j, "p_cont_w": k * cph.p_cont_w, "p_peak_w": k * cph.p_peak_w, "vol_mm3": k * box_vol(cph.dims), "mass_g": k * cph.mass_g, "cost": tuple(k * b for b in cph.cost)}
    return best[1]


# ----------------------------------------------------------------------------------------------
# Shared sub-models
# ----------------------------------------------------------------------------------------------

def wall(L_mm):
    """Wall thickness (mm) and process by size (A, with stated reasons)."""
    if L_mm <= 12:
        return 0.40, "flex PCB skin 0.11 mm on a resin frame 0.3 mm (hobby MSLA resin, XY about 0.03-0.05 mm)"
    if L_mm <= 30:
        return 0.80, "resin shell 0.8 mm (JLC3DP minimum wall about 0.8 mm)"
    return 1.20, "FDM shell 1.2 mm (3 perimeters, 0.4 mm nozzle)"


def shell_mass(L, t, rho):
    return rho * (L**3 - (L - 2 * t) ** 3)


def adhesion_force(L):
    """Worst-case capillary adhesion (humid air, wetting) of a cube resting on four corner feet of
    radius r_f: 4 x 4 pi gamma r_f (E, Laplace bridge at a sphere-flat contact). Dry and rough is
    10x lower. Weight scales as L^3 and this as r_f, so it matters most below a few mm."""
    r_f = max(0.1e-3, 0.02 * L)
    return 4 * 4 * math.pi * GAMMA_WATER * r_f


def requirements(L, m):
    w = m * G
    fa = adhesion_force(L)
    return {
        "weight_N": w,
        "adhesion_N": fa,
        "tip_torque": w * L / 2 + fa * L,  # floor roll, quasi-static peak (+adhesion at the lifting edge)
        "climb_torque": w * L / math.sqrt(2),  # pivot over a neighbour, gravity peak
        "e_roll": w * L * (math.sqrt(2) - 1) / 2 + fa * 0.2e-3,  # J, raise COM over the edge (+peel a 0.2 mm capillary bridge)
        "e_climb": w * L * (1 + 1 / math.sqrt(2) - 0.5),  # J, COM from L/2 to L + L/sqrt2 (180 deg over a neighbour)
        "yaw_friction": MU_FLOOR * (w + fa) * 0.3826 * L,  # N m, uniform pressure on a square
        "cantilever_moment": DYNAMIC * w * L * N_CANTILEVER**2 / 2,
        "hang_tension": DYNAMIC * N_HANG * w,
        "arm_torque": ARM_TORQUE_50 * (m * L) / ARM_REF,  # continuous, scaled from module_b demo 11
    }


@dataclass
class SphereBond:
    """Free-rotating N52 sphere in each of the 8 corners, inset under a cover. A face bond is 4
    sphere pairs; an edge hinge is the 2 pairs on that edge. Genderless at every quarter turn and
    self-aligning, because free spheres always turn to attract (E)."""

    r: float  # sphere radius, m
    cover: float  # material over each sphere, m
    L: float

    @property
    def d0(self):  # centre distance of a bonded pair (the two covers between them)
        return 2 * (self.r + self.cover)

    def pair(self, d):
        return K_SPHERE * sphere_pair_force(self.r, d)

    @property
    def tension(self):
        return 4 * self.pair(self.d0)

    @property
    def inset(self):
        return self.r + self.cover

    @property
    def moment(self):
        """Peel moment about an edge: the far pair at lever L - 2 inset; hinge pair contributes 0."""
        return 2 * self.pair(self.d0) * (self.L - 2 * self.inset)

    @property
    def peel_energy(self):
        """Work to pull the far pair from d0 to infinity: F ~ d^-4 so int = F(d0) d0 / 3."""
        return 2 * self.pair(self.d0) * self.d0 / 3

    def hinge_force(self, phi):
        """Pull of the 2 hinge pairs after the mover has turned phi (rad) about the shared edge.
        Sphere centres sit sqrt2 inset from the edge line; they start 90 deg apart about it and end
        270 deg apart, so the centre distance is 2 sqrt2 inset sin((pi/2 + phi) / 2)."""
        d = 2 * math.sqrt(2) * self.inset * math.sin((math.pi / 2 + phi) / 2)
        return 2 * self.pair(max(d, self.d0))

    def hinge_margin(self, m, L, ke0):
        """Worst ratio over the swing of hinge pull to the load that would open it: centripetal
        m w^2 r_c plus the outward share of gravity, with w from the energy left after the peel
        and the rise (the impulse is given at phi = 0, the far bond already peeled)."""
        i_edge = 2 / 3 * m * L**2
        r_c = L / math.sqrt(2)
        worst = float("inf")
        for phi in np.linspace(0.0, math.pi, 61):
            th = -math.pi / 4 + phi
            rise = r_c * (math.sin(th) - math.sin(-math.pi / 4))
            ke = ke0 - self.peel_energy - m * G * rise
            if ke < 0:
                return 0.0
            load = m * (2 * ke / i_edge) * r_c - m * G * math.sin(th)
            if load > 0:
                worst = min(worst, self.hinge_force(phi) / load)
        return worst

    def capture(self, m):
        """Largest lateral offset (fraction of L) at which the 4 pairs pull a resting cube across
        the floor against friction (MU_HIGH, margin 1.5)."""
        need = MARGIN * MU_HIGH * m * G
        xs = np.linspace(1e-6, 0.5 * self.L, 400)
        lat = np.array([4 * self.pair(math.hypot(self.d0, x)) * x / math.hypot(self.d0, x) for x in xs])
        ok = xs[lat >= need]
        return float(ok.max() / self.L) if len(ok) else 0.0

    def volume_mm3(self):
        return 8 * (2 * self.r * 1e3 + 0.1) ** 3

    def mass(self):
        return 8 * RHO["ndfeb"] * 4 / 3 * math.pi * self.r**3


def size_spheres(L, m, cover, hinge=True):
    """Smallest sphere radius whose bond meets 1.6x the cantilever moment and hang tension and
    whose edge hinge holds a ballistic climb with 1.6x margin (if hinge). Bisection: all three
    margins grow with the radius over the range searched."""
    req = requirements(L, m)

    def good(r):
        b = SphereBond(r, cover, L)
        if b.moment < 1.6 * req["cantilever_moment"] or b.tension < 1.6 * req["hang_tension"]:
            return False
        return (not hinge) or b.hinge_margin(m, L, MARGIN * (b.peel_energy + req["e_climb"])) >= 1.6

    lo, hi = 0.1e-3, 0.18 * L
    if not good(hi):
        return SphereBond(hi, cover, L)
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if good(mid) else (mid, hi)
    return SphereBond(hi, cover, L)


def impulse_efficiency(mu):
    """A cube on the floor receives an internal angular impulse H about the axis of one bottom edge
    (flywheel caught by a pawl). With friction mu < 1 the edge slips backwards during the impulse.
    Returns the kinetic energy left for tipping, as a fraction of the no-slip H^2 / (2 I_edge)
    (E: rigid-body impulse equations, derivation in concepts.md)."""
    i_cm, i_edge = 1 / 6, 2 / 3  # units of m L^2
    if mu >= 1:
        return 1.0
    w = 1.0 / (i_cm + (1 + mu) / 4)  # omega for H = 1, m = L = 1
    vx, vy = -mu * w / 2, -w / 2
    t = 0.5 * (vx**2 + vy**2) + 0.5 * i_cm * w**2
    slide = 0.5 * (w / 2 * (1 - mu)) ** 2
    return (t - slide) / (1 / (2 * i_edge))


def housner_window():
    """Ballistic roll window: a square block landing flat keeps r = 1 - 1.5 sin^2(45 deg) = 0.25 of
    its angular velocity (Housner 1963), so 6.25 % of its energy. It rolls a second time only if
    that exceeds the e_roll barrier: start energy above 16 e_roll. Window = [1, 16] x e_roll (E)."""
    r = 1 - 1.5 * math.sin(math.radians(45)) ** 2
    return 1 / r**2


def fall(L, m, parts_mass_max, bond_area, allowable=5e6):
    """Fall from FALL_HEIGHT onto a hard floor, landing on a corner (worst). Hertz sphere-flat with
    corner radius 0.05 L, resin on tile (E* 2.8 GPa). Returns impact energy, peak deceleration and
    the stress on the bond of the heaviest internal part (allowable 5 MPa, cyanoacrylate on resin)."""
    v = math.sqrt(2 * G * FALL_HEIGHT)
    e_star = 2.8e9
    rc = max(0.05 * L, 0.1e-3)
    delta = (15 * m * v**2 / (16 * e_star * math.sqrt(rc))) ** 0.4
    f = 4 / 3 * e_star * math.sqrt(rc) * delta**1.5
    a = f / m
    stress = parts_mass_max * a / bond_area
    return {"impact_mJ": 0.5 * m * v**2 * 1e3, "v_mps": v, "peak_g": a / G, "part_stress_MPa": stress / 1e6, "allowable_MPa": allowable / 1e6, "margin": allowable / stress}


# ----------------------------------------------------------------------------------------------
# Actuators
# ----------------------------------------------------------------------------------------------

def disc_field(br, radius, t, z):
    """On-axis field of an axial cylinder magnet at distance z from its face (exact)."""
    return br / 2 * ((z + t) / math.hypot(z + t, radius) - z / math.hypot(z, radius))


@dataclass
class PcbMotor:
    """Axial-flux motor printed in the face flex: 2-layer, 1 oz (35 um) copper, 55 % fill, rotor a
    diametric (2-pole) NdFeB disc of radius r_o and thickness t_r over the coils at 0.1 mm.
    Torque = B J t_eff A r_mean x 0.6 (end turns and pole shape, A). All dimensions in m."""

    r_o: float
    t_r: float
    layers: int = 2
    t_cu: float = 35e-6
    fill: float = 0.55
    gap: float = 0.1e-3

    @property
    def area(self):
        r_i = 0.3 * self.r_o
        return math.pi * (self.r_o**2 - r_i**2)

    @property
    def r_mean(self):
        r_i = 0.3 * self.r_o
        return 2 / 3 * (self.r_o**3 - r_i**3) / (self.r_o**2 - r_i**2)

    @property
    def t_eff(self):
        return self.layers * self.t_cu * self.fill

    @property
    def b_gap(self):
        z = self.gap + self.layers * self.t_cu / 2
        return 0.7 * disc_field(BR["N52"], self.r_o / 2, self.t_r, z)  # 2-pole: pole radius ~ r_o/2, 0.7 fringing (A)

    def k_m(self):
        """Motor constant tau / sqrt(P) (N m / sqrt W): P = J^2 rho t_eff A."""
        return 0.6 * self.b_gap * self.r_mean * math.sqrt(self.t_eff * self.area / RHO_CU)

    def torque_at(self, power, pulse_s=1.0):
        j_power = math.sqrt(power / (RHO_CU * self.t_eff * self.area))
        j_heat = math.sqrt(60.0 * CV_CU / (RHO_CU * pulse_s))  # 60 K adiabatic over the pulse
        j = min(j_power, j_heat)
        return 0.6 * self.b_gap * j * self.t_eff * self.area * self.r_mean


@dataclass
class BoughtMotor:
    name: str
    dims: tuple  # mm (dia, length) or box
    mass_g: float
    stall_nm: float
    omega0: float  # rad/s at the output, no load
    i_stall: float  # A
    volts: float
    cost: tuple
    status: str

    def usable(self):
        return 0.4 * self.stall_nm  # continuous rule of thumb for gearheads (A)

    def usable_staged(self):
        """With one extra printed 3:1 spur or planetary stage at 70 % (spec 4.1: N20 needs an extra
        stage); output limited to 3x the gearhead's own continuous rating (A)."""
        return 0.4 * self.stall_nm * 3 * 0.7


GEARMOTORS = [
    BoughtMotor("Orbray 1.5 mm geared (inquiry; no specs)", (1.5, 8.0), 0.05, float("nan"), float("nan"), float("nan"), 3.0, (float("nan"),) * 3, "no published specs (parts.json)"),
    BoughtMotor("4 mm planetary 0408 (AliExpress)", (4.0, 18.0), 0.8, 5e-3, 30.0, 0.15, 3.0, (3.5, 2.0, 0.6), "A: listings 2-20 mN m, middle-low value"),
    BoughtMotor("Orbray BMN04-0829P337", (4.0, 18.4), 1.0, 5.44e-3, 98 * 2 * math.pi / 60, 0.1, 3.0, (60.0, 25.0, 5.0), "V torque/speed; mass, current, price A (inquiry)"),
    BoughtMotor("6 mm planetary 0615 (AliExpress)", (6.0, 20.0), 1.5, 15e-3, 30.0, 0.4, 3.0, (3.0, 1.5, 0.5), "A: listings 5-50 mN m"),
    BoughtMotor("8 mm planetary 0820 (AliExpress)", (8.0, 24.0), 2.5, 30e-3, 30.0, 0.5, 3.7, (3.5, 1.8, 0.6), "A: listings 10-100 mN m"),
    BoughtMotor("N20 1:298 (Robu.in)", (10.0, 12.0, 26.0), 9.0, 0.2 / 0.4, 10.0, 0.6, 3.7, (2.2, 1.2, 0.5), "A: spec 4.1 gives 0.2 N m usable; 0.6 A stall at 3.7 V"),
]

FLYWHEEL_MOTORS = [
    # Coreless and brushless motors driving a flywheel directly. stall from k_t I_stall with k_t = V / omega0 (E).
    BoughtMotor("Orbray BMN02-0593 2.4 mm brushless", (2.4, 5.0), 0.15, 0.01e-3, 6000 * 2 * math.pi / 60, 0.2, 4.0, (40.0, 15.0, 3.0), "V stall/speed (100-6000 rpm); mass, current, price A"),
    BoughtMotor("0408 coreless 4x8 mm", (4.0, 8.0), 0.5, 3.7 / 3140 * 0.1, 3140.0, 0.1, 3.7, (1.0, 0.4, 0.1), "A: 30k rpm, 100 mA stall"),
    BoughtMotor("0615 coreless 6x15 mm", (6.0, 15.0), 1.3, 3.7 / 3660 * 0.5, 3660.0, 0.5, 3.7, (1.0, 0.4, 0.1), "A: 35k rpm, 500 mA stall (parts.json 400-600)"),
]


# ----------------------------------------------------------------------------------------------
# Architectures
# ----------------------------------------------------------------------------------------------

@dataclass
class Result:
    L_mm: float
    mass_g: float
    margins: dict
    info: dict = field(default_factory=dict)


def flywheel_block(L, m, i_f, omega_max, bond, motor_torque, e_store, p_cont, mu=MU_FLOOR):
    """Shared inertial-move model (two orthogonal flywheels, pawl-caught). Returns margins and info."""
    req = requirements(L, m)
    i_edge = 2 / 3 * m * L**2
    i_z = m * L**2 / 6
    eta = impulse_efficiency(MU_LOW)  # slip worst case
    h_roll = math.sqrt(2 * i_edge * MARGIN * req["e_roll"] / eta)
    # quasi-static check: the cube must not tip back while the flywheel is spun up: spin torque below tip torque
    e_split = MARGIN * (bond.peel_energy + req["e_roll"])
    h_split = math.sqrt(2 * i_edge * e_split / eta)
    e_climb = MARGIN * (bond.peel_energy + req["e_climb"])
    h_climb = math.sqrt(2 * i_edge * e_climb)  # hinge pairs hold: no slip
    tau_f = MU_HIGH * (m * G + req["adhesion_N"]) * 0.3826 * L
    h_yaw = math.sqrt(2 * i_z * tau_f * math.radians(45))  # one impulse turns up to 45 deg at high friction
    h_cap = i_f * omega_max
    omega_climb = h_climb / i_f
    e_move = {k: 0.5 * h**2 / i_f / MOTOR_EFF for k, h in (("roll", h_roll), ("split", h_split), ("climb", h_climb), ("yaw", h_yaw))}
    # Spin-up must not tip the cube by reaction: spin torque below 0.5 x tip torque; spin time bound.
    spin_torque = min(motor_torque, 0.5 * req["tip_torque"])
    t_spin_climb = i_f * omega_climb / max(spin_torque, 1e-12)
    t_spin_roll = h_roll / max(spin_torque, 1e-12)
    # Lone run
    e_need = LONE_MINUTES * 60 * (P_IDLE + e_move["roll"] / LONE_MOVE_PERIOD)
    t_mech = t_spin_roll + 0.15 + 2 * math.sqrt(L / G) + 2 * h_roll / max(spin_torque, 1e-12)  # spin, settle, fall, despin slowly
    t_energy = e_move["roll"] / max(p_cont - P_IDLE, 1e-9)
    t_move = max(t_mech, t_energy)
    speed = 1.0 / t_move  # body lengths per second
    margins = {
        "lone_self_right": h_cap / h_roll,
        "lone_translate": h_cap / h_roll,
        "lone_turn": h_cap / h_yaw,
        "split": h_cap / h_split,
        "climb": h_cap / h_climb,
        "climb_hinge_holds": bond.hinge_margin(m, L, e_climb),
        "spin_time": SPIN_TIME_MAX / max(t_spin_climb, 1e-9),
        "lone_energy": e_store / e_need,
        "lone_speed": speed / MIN_SPEED,
    }
    info = {
        "h_roll": h_roll, "h_climb": h_climb, "h_cap": h_cap, "omega_climb_rpm": omega_climb * 60 / (2 * math.pi),
        "e_move_mJ": {k: v * 1e3 for k, v in e_move.items()}, "lone_moves": e_store / max(e_move["roll"] + P_IDLE * LONE_MOVE_PERIOD, 1e-12),
        "lone_speed_mm_s": speed * L * 1e3, "t_move_s": t_move, "slip_eta": eta, "housner_window": housner_window(),
    }
    return margins, info


def bond_margins(L, m, bond):
    req = requirements(L, m)
    cap = bond.capture(m)
    return {
        "bond_cantilever": bond.moment / req["cantilever_moment"],
        "bond_hang": bond.tension / req["hang_tension"],
        "dock_capture": cap / POS_ERROR,
    }, {"sphere_d_mm": 2 * bond.r * 1e3, "bond_tension_N": bond.tension, "bond_moment_mNm": bond.moment * 1e3, "peel_energy_uJ": bond.peel_energy * 1e6, "capture_frac_L": cap}


def packing_margin(inner_mm, vols, fill):
    need = sum(vols) / fill
    return inner_mm**3 / need


def arch_rw(L_mm, motor="pcb", twist=None, wheel=False, flywheel_mat="tungsten", epm=False, omega_cap=OMEGA_MAX_PIN):
    """Reaction-wheel cube: two orthogonal flywheels caught by coil-flipped pawls (inertial moves),
    free corner spheres as bonds, flex PCBA skin with CH32V003, storage chosen by need."""
    L = L_mm * 1e-3
    t_w, process = wall(L_mm)
    inner = L_mm - 2 * t_w
    cover = 0.1e-3 if L_mm <= 12 else 0.3e-3
    # flywheel geometry: two orthogonal discs near two adjacent faces must not collide
    t_f = max(0.4e-3, 0.12 * L)
    r_f = (inner * 1e-3) / 2 - t_f - 0.15e-3
    if motor != "pcb":
        bm = next(x for x in FLYWHEEL_MOTORS if x.name == motor)
        if not fits((bm.dims[0], bm.dims[0], bm.dims[1]), inner) or bm.dims[1] + t_f * 1e3 + 0.3 > inner:
            return None
        r_f = min(r_f, (inner * 1e-3) / 2 - 0.2e-3)
    if r_f <= 0.3e-3:
        return None
    t_max = t_f

    def rotor(t):
        if motor == "pcb" and flywheel_mat != "ndfeb":  # NdFeB core (pole) of radius r_f/2, heavy rim outside it
            mass = RHO["ndfeb"] * math.pi * (r_f / 2) ** 2 * t + RHO[flywheel_mat] * math.pi * (r_f**2 - (r_f / 2) ** 2) * t
            inertia = 0.5 * RHO["ndfeb"] * math.pi * t * (r_f / 2) ** 4 + 0.5 * RHO[flywheel_mat] * math.pi * t * (r_f**4 - (r_f / 2) ** 4)
        else:
            mass = RHO["ndfeb" if motor == "pcb" else flywheel_mat] * math.pi * r_f**2 * t
            inertia = 0.5 * mass * r_f**2
        return mass, inertia

    omega_eff = omega_cap if motor == "pcb" else min(omega_cap, bm.omega0 * 0.8)
    rotor_mass, i_f = rotor(t_f)
    # first pass mass guess, then iterate (sphere size and flywheel size depend on mass)
    m = 0.5 * 2000 * L**3
    for _ in range(8):
        bond = size_spheres(L, m, cover)
        if epm:
            bond = SphereBond(bond.r, cover, L)
        req0 = requirements(L, m)
        h_need = 1.7 * math.sqrt(2 * (2 / 3 * m * L**2) * MARGIN * (bond.peel_energy + req0["e_climb"]))
        t_need = t_max * (h_need / omega_eff) / rotor(t_max)[1]  # thinnest rotor that stores the climb impulse at omega_eff
        t_mass = t_max * (ROTOR_SHARE * m / 2) / rotor(t_max)[0]  # thickest rotor pair within ROTOR_SHARE of the cube mass
        t_f = min(t_max, max(0.2e-3, t_need, t_mass))  # thicker is better for energy per move (E = H^2 / 2 I)
        rotor_mass, i_f = rotor(t_f)
        parts_mass = 2 * rotor_mass + bond.mass()
        elec_mass = 0.04e-3 + 0.02e-3 * (L_mm / 5)
        storage_guess = 0.06e-3 if L_mm < 11 else 0.4e-3
        motor_mass = 0.0 if motor == "pcb" else 2 * bm.mass_g * 1e-3
        twist_mass = 0.0
        if twist is not None:
            twist_mass = twist.mass_g * 1e-3 + RHO["brass"] * math.pi * (0.4 * L) ** 2 * 0.3e-3
        shell = shell_mass(L, t_w * 1e-3, RHO["resin"] if L_mm > 12 else 0.5 * (RHO["resin"] + RHO["flex"]))
        m_new = 1.10 * (parts_mass + elec_mass + storage_guess + motor_mass + twist_mass + shell)
        if abs(m_new - m) < 1e-9:
            break
        m = m_new
    req = requirements(L, m)
    # storage: energy for the lone run, power for spin-up
    e_roll_guess = 0.5 * (math.sqrt(2 * (2 / 3 * m * L**2) * MARGIN * req["e_roll"] / impulse_efficiency(MU_LOW))) ** 2 / i_f / MOTOR_EFF
    e_need = LONE_MINUTES * 60 * (P_IDLE + e_roll_guess / LONE_MOVE_PERIOD)
    p_need = P_IDLE + e_roll_guess / LONE_MOVE_PERIOD
    fixed_vol = bond.volume_mm3() + 2 * (2 * r_f * 1e3 + 0.2) ** 2 * (t_f * 1e3 + 0.3) + 14.0 + 2 * 0.5  # spheres, rotors, electronics (MCU 3x3, IMU 2x2, 12 FETs, diodes, passives: 14 mm3 incl. height), 2 pawls
    if motor != "pcb":
        fixed_vol += 2 * bm.dims[0] ** 2 * bm.dims[1]
    if twist is not None:
        fixed_vol += twist.dims[0] ** 2 * twist.dims[-1] + (0.8 * L_mm) ** 2 * 0.6
    if wheel:
        fixed_vol += (0.3 * L_mm) ** 2 * (0.2 * L_mm) + 1.0  # wheel well, coil-driven wheel, bistable retract
    fill = FILL_FLEX if L_mm <= 12 else FILL_HOBBY
    budget = inner**3 * fill - fixed_vol
    st = choose_storage(inner, max(budget, 0.0), e_need, p_need, 8e-3)
    p_spin = max(st["p_peak_w"], 1e-4) * 0.8
    if motor == "pcb":
        pm = PcbMotor(r_f, t_f)
        motor_torque = pm.torque_at(p_spin, pulse_s=1.0)
        omega_max = omega_cap
        motor_info = {"k_m": pm.k_m(), "b_gap_T": pm.b_gap, "spin_power_mW": p_spin * 1e3, "spin_torque_uNm": motor_torque * 1e6}
    else:
        motor_torque = min(bm.stall_nm * 0.5, bm.stall_nm * p_spin / (bm.volts * bm.i_stall))
        omega_max = min(omega_cap, bm.omega0 * 0.8)
        motor_info = {"motor": bm.name, "stall_uNm": bm.stall_nm * 1e6, "spin_torque_uNm": motor_torque * 1e6}
    m += 1.1 * (st["mass_g"] * 1e-3 - storage_guess)
    req = requirements(L, m)
    margins, info = flywheel_block(L, m, i_f, omega_max, bond, motor_torque, st["energy_j"], st["p_cont_w"] if st["p_cont_w"] > 0 else st["p_peak_w"] * 0.5)
    bm_, binfo = bond_margins(L, m, bond)
    margins.update(bm_)
    margins["packing_volume"] = packing_margin(inner, [fixed_vol, st["vol_mm3"]], fill)
    pin_area = max(0.3 * r_f * t_f, 0.02e-6)  # rotor on its steel pin: projected bearing area, pin dia 0.3 r_f
    fall_rw = fall(L, m, rotor_mass, pin_area, allowable=40e6)  # resin bushing bearing stress (A: resin compressive 60-100 MPa)
    margins["fall"] = fall_rw["margin"]
    if twist is None:
        margins["machine_twist"] = 0.0
    else:
        margins["machine_twist"] = twist.usable_staged() / req["arm_torque"] if not math.isnan(twist.stall_nm) else float("nan")
        if not fits((twist.dims[0], twist.dims[0], twist.dims[-1]), inner):
            margins["packing_volume"] = 0.0
    info.update(binfo)
    info.update(motor_info)
    info.update({"process": process, "inner_mm": inner, "rotor_r_mm": r_f * 1e3, "rotor_t_mm": t_f * 1e3, "rotor_mass_mg": rotor_mass * 1e6, "i_f": i_f, "storage": st, "fall": fall_rw, "req": req})
    if wheel:
        # coil-driven wheel, direct drive: traction torque mu m g r_w, PCB motor of radius 0.15 L
        r_w = 0.12 * L
        wm = PcbMotor(0.15 * L, 0.1 * L)
        tq = wm.torque_at(st["p_peak_w"] * 0.8, pulse_s=1.0)
        margins["wheel_torque"] = tq / (MU_FLOOR * m * G * r_w)
        info["wheel_speed_mm_s"] = min(0.2, 300 * r_w) * 1e3  # 300 rad/s cap (A)
    return Result(L_mm, m * 1e3, margins, info)


def arch_geared(L_mm, motor_name, cell_name, board="custom", n_motors=3):
    """Hobby baseline: layout C made real. Gearmotors for wheel, twist disc and the pivot/latch
    selector (which also swings a kick leg for floor self-righting and retracts the wheel). Bonds are
    geared hook latches (undesigned; assumed to meet strength) plus corner spheres for capture."""
    L = L_mm * 1e-3
    t_w, process = wall(L_mm)
    inner = L_mm - 2 * t_w
    gm = next(x for x in GEARMOTORS if x.name == motor_name)
    if math.isnan(gm.stall_nm):
        return None
    cell = next(s for s in STORAGE if s.name == cell_name)
    mdims = gm.dims if len(gm.dims) == 3 else (gm.dims[0], gm.dims[0], gm.dims[1])
    if not fits(mdims, inner) or not fits(cell.dims, inner):
        return None
    brd = (22.5, 18.0, 4.0, 2.5, 2.0) if board == "esp32" else (max(8.0, 0.5 * L_mm), max(8.0, 0.5 * L_mm), 2.0, 0.3, 1.5)
    if board == "esp32" and not fits(brd[:3], inner):
        return None
    m = 0.5 * 1500 * L**3
    for _ in range(6):
        bond = size_spheres(L, m, 0.3e-3, hinge=False)
        shell = shell_mass(L, t_w * 1e-3, RHO["resin"] if L_mm <= 30 else RHO["pla"])
        mech = 0.10 * shell + RHO["pla"] * (0.6 * L) ** 2 * 1.5e-3 + RHO["brass"] * 0.0  # selector, hooks, kick leg, disc (A)
        m_new = 1.10 * (n_motors * gm.mass_g * 1e-3 + cell.mass_g * 1e-3 + brd[3] * 1e-3 + bond.mass() + shell + mech)
        if abs(m_new - m) < 1e-9:
            break
        m = m_new
    req = requirements(L, m)
    vols = [n_motors * box_vol(mdims), box_vol(cell.dims), box_vol(brd[:3]), bond.volume_mm3(), 0.12 * inner**3]  # selector + latches + wheel well: 12 % of inner volume (A)
    usable = gm.usable()
    i_drive = gm.i_stall * 1.2  # one motor starting at stall plus the rest of the cube (selector: one motor at a time)
    t_drive = 120.0
    e_need = LONE_MINUTES * 60 * (P_IDLE * 20 + gm.volts * gm.i_stall * 0.3 * 0.5)
    margins = {
        "lone_self_right": usable / req["tip_torque"],  # kick leg, lever ~1:1
        "lone_translate": usable / (MU_FLOOR * m * G * 0.2 * L),  # wheel radius 0.2 L traction torque
        "lone_turn": usable / (MU_FLOOR * m * G * 0.2 * L),  # skid turn on wheel + edge, or kick-leg pivot
        "split": usable / req["climb_torque"],
        "climb": usable / (req["climb_torque"] * 1.3),  # + wheel spring / latch friction (module-pairs: up to 0.09 vs 0.042)
        "climb_hinge_holds": 2.0,  # hooks (A)
        "spin_time": 2.0,
        "lone_energy": cell.energy_j / e_need,
        "lone_speed": 3.0,
        "bond_cantilever": 2.0,  # hook latch assumed strong (undesigned)
        "bond_hang": 2.0,
        "dock_capture": bond.capture(m) / POS_ERROR,
        "packing_volume": packing_margin(inner, vols, FILL_HOBBY),
        "fall": fall(L, m, gm.mass_g * 1e-3, (mdims[0] * mdims[-1]) * 1e-6 * 0.3, allowable=5e6)["margin"],
        "machine_twist": gm.usable_staged() / req["arm_torque"],
        "peak_current": cell.p_peak_w / (gm.volts * i_drive),
    }
    info = {"process": process, "inner_mm": inner, "motor": gm.name, "cell": cell.name, "req": req, "sphere_d_mm": 2 * bond.r * 1e3, "fall": fall(L, m, gm.mass_g * 1e-3, (mdims[0] * mdims[-1]) * 1e-6 * 0.3)}
    return Result(L_mm, m * 1e3, margins, info)


def arch_electrovoxel(L_mm):
    """Coil electromagnets in the 12 edges, attract to hinge, repel/attract to drive (ElectroVoxel,
    Nisser et al. 2022, 60 mm, 103 g, 800-turn ferrite-core coils at 1.06 A, 11.7 W per magnet;
    only demonstrated on an air table and in microgravity). A lone cube on a bare floor has nothing to
    push on: lone gates fail by construction. Here: the mid-swing torque in 1 g vs L."""
    L = L_mm * 1e-3
    m = 0.5 * 2000 * L**3 * 1.2
    req = requirements(L, m)
    j = math.sqrt(60.0 * CV_CU / (RHO_CU * 1.0))  # 1 s adiabatic, 60 K
    ni = j * 0.6 * (0.1 * L) * (0.9 * L)  # winding window 0.1 L x 0.9 L, fill 0.6
    g_mid = 0.5 * L
    b = MU0 * ni / (2 * g_mid)
    f = b**2 * (0.12 * L) ** 2 / (2 * MU0)
    p = j**2 * RHO_CU * 0.6 * (0.1 * L) * (0.9 * L) * 2 * math.pi * 0.08 * L * 2  # two coils in the circuit
    margins = {k: 0.0 for k in ("lone_self_right", "lone_translate", "lone_turn")}
    margins["climb"] = f * L / req["climb_torque"]
    margins["support_tips"] = (m * G * L / 2 + m * G * L) / (MARGIN * req["climb_torque"])  # support cube resisting the reaction
    return Result(L_mm, m * 1e3, margins, {"mid_swing_force_mN": f * 1e3, "coil_power_W": p, "note": "fails every lone gate by construction"})


def arch_fails(L_mm, name):
    """Architectures that fail a lone-mobility gate by construction (no mechanism for it)."""
    L = L_mm * 1e-3
    m = 0.5 * 2000 * L**3
    reasons = {
        "vibration": ("lone_self_right", "stick-slip / vibration translates and turns a cube on the face it rests on; it cannot tip it onto another face"),
        "piezo": ("lone_self_right", "piezo stick-slip strokes are micrometres; no tipping; needs a 30-150 V boost"),
        "electrostatic": ("lone_self_right", "electrostatic zipping gives about 1-6 kPa at 200 V; no tipping; humidity-sensitive"),
        "sma_flaps": ("lone_turn", "rolling about edges cannot change yaw; 12 SMA flaps with 24 crimped terminations per cube"),
    }
    gate, why = reasons[name]
    margins = {gate: 0.0}
    return Result(L_mm, m * 1e3, margins, {"reason": why})


# ----------------------------------------------------------------------------------------------
# Cost
# ----------------------------------------------------------------------------------------------

def cost_rw(res, motor="pcb", twist=None, wheel=False):
    """USD per cube at 10, 1k, 1M units (A unless the part entry says otherwise)."""
    st = res.info["storage"]["cost"]
    items = {
        "flex PCBA (2-layer flex, 0201, JLC; setup amortised)": (7.0, 1.2, 0.20),
        "resin frame / MJF core": (0.5, 0.25, 0.02),
        "8 N52 spheres": (0.4, 0.12, 0.03),
        "2 rotors (diametric disc + brass or tungsten rim) and pins": (2.0, 0.6, 0.06),
        "2 pawls (steel pin + magnet)": (0.4, 0.1, 0.01),
        "CH32V003 + IMU + FETs + diodes + IR": (2.5, 1.2, 0.30),
        "storage": st,
    }
    if motor != "pcb":
        bm = next(x for x in FLYWHEEL_MOTORS if x.name == motor)
        items["2 flywheel motors"] = tuple(2 * c for c in bm.cost)
    if twist is not None:
        items["twist gearmotor + disc + bearing"] = tuple(c + d for c, d in zip(twist.cost, (0.5, 0.2, 0.03)))
    if wheel:
        items["coil-driven retractable wheel"] = (1.0, 0.3, 0.03)
    tot = tuple(sum(v[i] for v in items.values()) for i in range(3))
    return tot, items


def cost_geared(res, motor_name, cell_name, n_motors=3):
    gm = next(x for x in GEARMOTORS if x.name == motor_name)
    cell = next(s for s in STORAGE if s.name == cell_name)
    items = {
        f"{n_motors} x {gm.name}": tuple(n_motors * c for c in gm.cost),
        cell.name: cell.cost,
        "board (CH32V003 + nRF or ESP32-C3)": (4.0, 2.0, 0.8),
        "printed shell, selector, hooks, wheel": (1.5, 0.8, 0.1),
        "8 N52 spheres, bearing, fasteners": (1.0, 0.4, 0.1),
    }
    tot = tuple(sum(v[i] for v in items.values()) for i in range(3))
    return tot, items


# ----------------------------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------------------------

L_GRID = np.round(np.concatenate([np.arange(3.0, 12.0, 0.25), np.arange(12.0, 30.0, 0.5), np.arange(30.0, 62.0, 2.0)]), 2)
MACHINE = {"machine_twist"}


def passes(res, skip=()):
    if res is None:
        return False
    vals = [v for k, v in res.margins.items() if k not in skip]
    return all((not math.isnan(v)) and v >= MARGIN for v in vals)


def binding(res, skip=()):
    if res is None:
        return "does not fit"
    items = [(k, v) for k, v in res.margins.items() if k not in skip]
    nan = [k for k, v in items if math.isnan(v)]
    if nan:
        return f"{nan[0]} unknown (no spec)"
    k, v = min(items, key=lambda kv: kv[1])
    return f"{k} {v:.2f}"


def min_L(fn, skip=()):
    """Smallest L that passes and keeps passing for the next 3 grid points (or to the grid end)."""
    results = [fn(L) for L in L_GRID]
    ok = [passes(r, skip) for r in results]
    for i in range(len(L_GRID)):
        if all(ok[i:i + 4]):
            return float(L_GRID[i]), results[i], results
    return None, None, results


ARCHS = {
    "RW-P": dict(title="Reaction-wheel cube, flex-PCB motors (frontier)", fn=lambda L: arch_rw(L, "pcb")),
    "RW-P-20k": dict(title="RW-P with jewel bearings, flywheel to 20k rad/s", fn=lambda L: arch_rw(L, "pcb", omega_cap=20000.0)),
    "RW-B-0408-W": dict(title="RW with 0408 coreless motors, tungsten rims", fn=lambda L: arch_rw(L, "0408 coreless 4x8 mm", flywheel_mat="tungsten")),
    "RW-P-W": dict(title="RW-P + coil-driven retractable wheel", fn=lambda L: arch_rw(L, "pcb", wheel=True)),
    "RW-B-0408": dict(title="Reaction-wheel cube, bought 0408 coreless motors", fn=lambda L: arch_rw(L, "0408 coreless 4x8 mm", flywheel_mat="brass")),
    "RW-B-BMN02": dict(title="Reaction-wheel cube, Orbray BMN02 2.4 mm brushless", fn=lambda L: arch_rw(L, "Orbray BMN02-0593 2.4 mm brushless", flywheel_mat="tungsten")),
    "RW-P-T0408": dict(title="RW-P + geared twist face (0408 planetary)", fn=lambda L: arch_rw(L, "pcb", twist=GEARMOTORS[1])),
    "RW-P-TBMN04": dict(title="RW-P + geared twist face (Orbray BMN04 337:1)", fn=lambda L: arch_rw(L, "pcb", twist=GEARMOTORS[2])),
    "RW-P-T0615": dict(title="RW-P + geared twist face (0615 planetary)", fn=lambda L: arch_rw(L, "pcb", twist=GEARMOTORS[3])),
    "RW-P-T0820": dict(title="RW-P + geared twist face (0820 planetary)", fn=lambda L: arch_rw(L, "pcb", twist=GEARMOTORS[4])),
    "RW-P-TN20": dict(title="RW-P + geared twist face (N20 1:298 + 3:1 stage)", fn=lambda L: arch_rw(L, "pcb", twist=GEARMOTORS[5])),
    "RW-B-0615-TN20": dict(title="RW with 0615 coreless flywheels (tungsten) + N20 twist face", fn=lambda L: arch_rw(L, "0615 coreless 6x15 mm", twist=GEARMOTORS[5], flywheel_mat="tungsten")),
    "RW-B-0615-T0820": dict(title="RW with 0615 coreless flywheels (tungsten) + 0820 twist face", fn=lambda L: arch_rw(L, "0615 coreless 6x15 mm", twist=GEARMOTORS[4], flywheel_mat="tungsten")),
    "RW-P-TOrb1.5": dict(title="RW-P + geared twist face (Orbray 1.5 mm, no specs)", fn=lambda L: arch_rw(L, "pcb", twist=GEARMOTORS[0])),
    "G-0615": dict(title="Geared layout C, 6 mm planetary x3 + kick leg (hobby)", fn=lambda L: arch_geared(L, "6 mm planetary 0615 (AliExpress)", "LiPo 5-10C 50 mAh 601422")),
    "G-0408": dict(title="Geared layout C, 4 mm planetary x3 + kick leg", fn=lambda L: arch_geared(L, "4 mm planetary 0408 (AliExpress)", "LiPo 5-10C 50 mAh 601422")),
    "G-BMN04": dict(title="Geared layout C, Orbray BMN04 x3", fn=lambda L: arch_geared(L, "Orbray BMN04-0829P337", "LiPo 401012 15 mAh")),
    "G-N20": dict(title="Geared layout C, N20 x3 (hobby baseline)", fn=lambda L: arch_geared(L, "N20 1:298 (Robu.in)", "LiPo 1S 150 mAh 20C whoop cell", board="esp32")),
    "EV": dict(title="ElectroVoxel edge electromagnets", fn=arch_electrovoxel),
    "VIB": dict(title="Vibration / coil stick-slip only", fn=lambda L: arch_fails(L, "vibration")),
    "PZ": dict(title="Piezo stick-slip", fn=lambda L: arch_fails(L, "piezo")),
    "ES": dict(title="Electrostatic zipping / adhesion", fn=lambda L: arch_fails(L, "electrostatic")),
    "SMA": dict(title="SMA kick flaps + SMA latches", fn=lambda L: arch_fails(L, "sma_flaps")),
}


def summarise():
    rows = []
    for key, a in ARCHS.items():
        lm, res, _ = min_L(a["fn"])
        lm_s, res_s, _ = min_L(a["fn"], skip=MACHINE)
        row = {"arch": key, "title": a["title"], "min_L_all_mm": lm, "min_L_no_machine_mm": lm_s}
        probe = res or res_s
        if probe is None:
            # report the binding constraint at a reference size
            for Lr in (5.0, 10.0, 20.0, 50.0):
                r = a["fn"](Lr)
                if r is not None:
                    row["binding_at_ref"] = f"L={Lr:g}: {binding(r)}"
                    row["reason"] = r.info.get("reason") or r.info.get("note")
                    break
        else:
            row["mass_g"] = round(probe.mass_g, 3)
        if res is not None:
            row["binding_all"] = binding(res)
        if res_s is not None:
            row["binding_no_machine"] = binding(res_s, skip=MACHINE)
            # what binds just below the minimum
            below = [x for x in L_GRID if x < lm_s]
            if below:
                rb = a["fn"](below[-1])
                row["fails_below_by"] = binding(rb, skip=MACHINE)
        if lm is None and lm_s is not None:
            row["machine_note"] = binding(res_s)
        if key.startswith("RW") and probe is not None:
            twist = {"RW-B-0615-TN20": GEARMOTORS[5], "RW-B-0615-T0820": GEARMOTORS[4], "RW-P-T0408": GEARMOTORS[1], "RW-P-TBMN04": GEARMOTORS[2], "RW-P-T0615": GEARMOTORS[3], "RW-P-T0820": GEARMOTORS[4], "RW-P-TN20": GEARMOTORS[5], "RW-P-TOrb1.5": GEARMOTORS[0]}.get(key)
            motor = {"RW-B-0615-TN20": "0615 coreless 6x15 mm", "RW-B-0615-T0820": "0615 coreless 6x15 mm", "RW-B-0408-W": "0408 coreless 4x8 mm", "RW-B-0408": "0408 coreless 4x8 mm", "RW-B-BMN02": "Orbray BMN02-0593 2.4 mm brushless"}.get(key, "pcb")
            row["cost_usd_10_1k_1M"] = [round(c, 2) for c in cost_rw(probe, motor, twist, wheel=key == "RW-P-W")[0]]
        if key.startswith("G-") and probe is not None:
            row["cost_usd_10_1k_1M"] = [round(c, 2) for c in cost_geared(probe, probe.info["motor"], probe.info["cell"])[0]]
        rows.append(row)
    return rows


def detail(key, L_mm):
    r = ARCHS[key]["fn"](L_mm)
    if r is None:
        return None
    out = {"L_mm": L_mm, "mass_mg": round(r.mass_g * 1e3, 1), "margins": {k: (round(v, 3) if not math.isnan(v) else None) for k, v in r.margins.items()}}
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, float):
            return float(f"{x:.4g}")
        return x
    out["info"] = clean(r.info)
    return out


# Machine joints, layout B demo peaks (module-demos.md) in units of m g L (module 120 g, 50 mm)
MACHINES = {
    "arm lift (11 modules, demo 11)": 0.42,
    "walker legs at touchdown (14 modules, demo 9)": 0.80,
    "crank at dead centre (demo 7)": 0.80,
    "crawler bend (demo 10)": 0.46,
    "crane luff (demo 4)": 0.20,
    "cart skid steer (demo 2)": 0.13,
    "door push (demo 8)": 0.09,
    "conveyor roller (demo 3)": 0.05,
}


def face_motor(L_mm, m):
    """Planar motor across a bond: flex coils in cube A's face (stator), magnets in cube B's face
    (rotor), turning about the shared face normal. Annulus 0.2 L to 0.45 L, 2 layers 1 oz at 55 %
    fill, rotor field B over the gap of two covers (0.2 mm). Shear stress sigma = B K; K limited
    (a) steady by free-convection loss from both cubes' faces (h 50 W/m2K, 40 K rise, E) and
    (b) for a 1 s move by adiabatic heating of the copper (60 K, E)."""
    L = L_mm * 1e-3
    r_o, r_i = 0.45 * L, 0.2 * L
    area = math.pi * (r_o**2 - r_i**2)
    r_mean = 2 / 3 * (r_o**3 - r_i**3) / (r_o**2 - r_i**2)
    t_eff = 2 * 35e-6 * 0.55
    b = 0.7 * disc_field(BR["N52"], 0.12 * L, 0.15 * L, 0.2e-3)
    h, dt = 50.0, 40.0
    k_steady = math.sqrt(h * dt * 6 * L**2 * t_eff / (RHO_CU * area))  # all six faces of the cube shed the heat
    k_pulse = math.sqrt(60.0 * CV_CU / (RHO_CU * 1.0)) * t_eff
    out = {"sigma_steady_kPa": 0.6 * b * k_steady / 1e3, "sigma_1s_kPa": 0.6 * b * k_pulse / 1e3}
    tau_s = 0.6 * b * k_steady * area * r_mean
    tau_p = 0.6 * b * k_pulse * area * r_mean
    p_pulse = (k_pulse / t_eff) ** 2 * RHO_CU * t_eff * area
    out.update({"torque_steady_uNm": tau_s * 1e6, "torque_1s_uNm": tau_p * 1e6, "power_1s_W": p_pulse})
    mgl = m * G * L
    rows = {}
    for name, peak in MACHINES.items():
        need = peak / (0.120 * G * 0.050) * mgl
        rows[name] = {"need_uNm": need * 1e6, "margin_steady": tau_s / need, "margin_1s": tau_p / need}
    out["machines"] = rows
    out["n_max_cantilever_1s"] = math.sqrt(2 * tau_p / (MARGIN * mgl))  # tau ~ m g L n^2 / 2
    # bond detent: magnet-to-magnet pressure across the same face, for comparison
    out["detent_pressure_kPa_vs_coil"] = None
    return out


def face_motor_table():
    rows = []
    for L in (3.0, 5.0, 6.75, 10.0, 20.0, 50.0):
        r = arch_rw(L, "pcb", omega_cap=20000.0)
        m = (r.mass_g if r else 2.5 * L**3 * 1e-3) * 1e-3
        fm = face_motor(L, m)
        fm["L_mm"] = L
        fm["mass_g"] = m * 1e3
        rows.append(fm)
    return rows


def sensitivity():
    """Minimum L (swarm behaviours) of RW-P under the assumptions that move it most."""
    global LONE_MINUTES, MOTOR_EFF, P_IDLE
    out = []
    base = (LONE_MINUTES, MOTOR_EFF, P_IDLE)
    for cap in (6000.0, 10000.0, 20000.0):
        for minutes in (0.5, 1.0, 2.0):
            for eff in (0.15, 0.3):
                LONE_MINUTES, MOTOR_EFF = minutes, eff
                lm, res, _ = min_L(lambda L: arch_rw(L, "pcb", omega_cap=cap), skip=MACHINE)
                out.append({"omega_cap_rad_s": cap, "lone_minutes": minutes, "motor_eff": eff, "min_L_mm": lm, "binding_below": binding(arch_rw(L_GRID[max(0, list(L_GRID).index(lm) - 1)], "pcb", omega_cap=cap), skip=MACHINE) if lm else None})
    LONE_MINUTES, MOTOR_EFF, P_IDLE = base
    return out


def ev_table():
    return [{"L_mm": L, "climb_margin_1g": round(arch_electrovoxel(L).margins["climb"], 3), "coil_power_W": round(arch_electrovoxel(L).info["coil_power_W"], 3)} for L in (1, 2, 3, 5, 10, 20, 60)]


def plot(rows):
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["RW-P-20k", "RW-P", "G-N20", "RW-P-TBMN04"]
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.tab20.colors)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, key in zip(axes.flat, keys):
        Ls, data = [], {}
        for L in L_GRID:
            r = ARCHS[key]["fn"](L)
            if r is None:
                continue
            Ls.append(L)
            for k, v in r.margins.items():
                data.setdefault(k, []).append(v if v > 0 else np.nan)
        for k, v in data.items():
            if len(v) == len(Ls):
                ax.plot(Ls, v, label=k, lw=1.2)
        ax.axhline(MARGIN, color="k", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(0.1, 1e3)
        ax.set_title(f"{key}: {ARCHS[key]['title']}", fontsize=9)
        ax.set_xlabel("cube edge L (mm)")
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].set_ylabel("margin (available / required)")
    axes[1, 0].set_ylabel("margin (available / required)")
    handles, labels = [], []
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "scaling_margins.png", dpi=120)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = summarise()
    details = {}
    for key, Ls in {"RW-P-20k": (5.0, 6.0, 7.0, 7.75, 8.0, 10.0), "RW-P-TN20": (30.0, 34.0, 40.0), "RW-P": (5.0, 6.0, 7.0, 8.0, 10.0), "RW-B-0408": (10.0, 12.0, 14.0), "RW-B-BMN02": (6.0, 8.0, 10.0), "RW-P-TBMN04": (20.0, 22.0, 24.0), "RW-P-T0408": (20.0, 24.0), "RW-P-T0615": (24.0, 30.0), "G-0615": (24.0, 30.0, 40.0), "G-N20": (40.0, 50.0), "RW-P-W": (6.0, 8.0)}.items():
        details[key] = [d for d in (detail(key, L) for L in Ls) if d]
    result = {
        "calibration": {"rows": CAL_ROWS, "k_sphere": K_SPHERE, "k_cylinder": K_CYL},
        "impulse_efficiency": {f"mu={mu}": round(impulse_efficiency(mu), 3) for mu in (0.3, 0.5, 0.8)},
        "housner_window_x_e_roll": housner_window(),
        "electrovoxel_1g": ev_table(),
        "sensitivity_rw_p": sensitivity(),
        "face_motor_across_bond": face_motor_table(),
        "summary": rows,
        "details": details,
        "margin_rule": MARGIN,
    }
    (OUT / "scaling.json").write_text(json.dumps(result, indent=1, default=str))
    plot(rows)
    print("calibration (published / model):")
    for r in CAL_ROWS:
        print(f"  {r['magnet']:45s} model {r['model_N']:.3f} N published {r['published_N']:.3f} N ratio {r['ratio']:.2f}")
    print(f"  k_sphere {K_SPHERE:.3f}  k_cylinder {K_CYL:.3f}")
    print("impulse efficiency", result["impulse_efficiency"], "housner window", round(housner_window(), 1))
    print("electrovoxel 1 g:", ev_table())
    print("face motor across a bond:")
    for r in result["face_motor_across_bond"]:
        print(f"  L {r['L_mm']} mm m {r['mass_g']:.3f} g sigma steady {r['sigma_steady_kPa']:.2f} kPa 1s {r['sigma_1s_kPa']:.2f} kPa torque 1s {r['torque_1s_uNm']:.2f} uNm P {r['power_1s_W']:.3f} W n_max {r['n_max_cantilever_1s']:.2f}")
        for k, v in r["machines"].items():
            print(f"      {k:48s} need {v['need_uNm']:.3g} uNm  margin steady {v['margin_steady']:.3f}  1 s {v['margin_1s']:.3f}")
    print("sensitivity RW-P:")
    for r in result["sensitivity_rw_p"]:
        print("  ", r)
    print()
    print(f"{'arch':14s} {'min L all':>10s} {'min L swarm':>12s}  binding / note")
    for r in rows:
        print(f"{r['arch']:14s} {str(r['min_L_all_mm']):>10s} {str(r['min_L_no_machine_mm']):>12s}  {r.get('binding_all') or r.get('binding_no_machine') or r.get('binding_at_ref')}  | below: {r.get('fails_below_by')} | cost {r.get('cost_usd_10_1k_1M')} | mass {r.get('mass_g')}")


if __name__ == "__main__":
    main()
