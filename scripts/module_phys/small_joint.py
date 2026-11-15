"""Smallest cube with a load-holding joint: routes, packing, minimum L (2026-11-15).

Compares four joint routes for the reaction-wheel cube of concepts.md (pick 1, 7-11 mm) as
functions of the cube side L, finds the smallest L at which the chosen route passes every
machine need with 1.5x margin and still packs, builds the CadQuery envelope at that L, and writes
one JSON line per (route, design, machine) to runs/module_phys/joint/results.jsonl.

    .venv/bin/python scripts/module_phys/small_joint.py            # analysis, search, envelope
    .venv/bin/python scripts/module_phys/small_joint.py --mc       # also fold in the Monte Carlo files

Routes:
1. SMA ratchet stepper on the disc hub (chosen): geared_joint.SmaStepper.
1. Micro gear train on the flywheel motor: geared_joint.nominal_design (Didel worm and printed
   split ring) and worm_last_design.
1. Bought gearmotors (Precision Microdrives 206-10C, Pololu 1000:1 HPCB): published numbers.
1. Piezo stick-slip on a friction ring: analytic, with the published stack class.

Machine needs come from layout B's measured demos (module-demos.md) in units of the cube's
m g L, and from this session's MuJoCo runs (scripts/small_joint_machines.py) where they exist.
Tags: V (read this session), E (estimated), A (assumed).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scaling  # noqa: E402  (concepts.md model: reaction-wheel cube margins)

from beebot.hardware.geared_joint import FLEXINOL, G, BondModel, nominal_bonds, nominal_design, sma_design, worm_last_design  # noqa: E402
from beebot.hardware.mmsim.opt import run_search  # noqa: E402

OUT = ROOT / "runs" / "module_phys" / "joint"
HW = ROOT / "hardware" / "module_phys"
MARGIN = 1.5

# Machine needs in units of the cube's m g L (layout B, module-demos.md, 120 g / 50 mm module).
MGL_B = 0.120 * G * 0.050
NEEDS_B = {
    "arm": {"drive": 0.42 / MGL_B, "hold": 0.42 / MGL_B, "bond": 1.125 / MGL_B},
    "walker": {"drive": 0.40 / MGL_B, "hold": 0.80 / MGL_B, "bond": 2.119 / MGL_B},
    "crane": {"drive": 0.204 / MGL_B, "hold": 0.204 / MGL_B, "bond": 0.415 / MGL_B},
    "crank": {"drive": 0.40 / MGL_B, "hold": 0.80 / MGL_B, "bond": 2.794 / MGL_B},
}
# Arm task: total joint travel of one pick and place (lift, open, yaw, lower, close, lift, yaw back,
# lower, open, lift), degrees (layout B's script).
ARM_TRAVEL_DEG = 24 + 20 + 90 + 24 + 15 + 24 + 90 + 22 + 20 + 24


# ------------------------------------------------------------------------------- mass and packing


def rw_base(L_mm, omega_cap=20000.0):
    """The reaction-wheel cube of concepts.md at L (jewel bearings by default, pick 1)."""
    return scaling.arch_rw(L_mm, "pcb", omega_cap=omega_cap)


def sma_joint_parts(L, st, bonds):
    """Volumes (mm3, envelope) and masses (g) added by two SMA joints, the disc rings and the
    face-ring spheres (E: printed resin 1.15 g/cm3, N52 7.5, brass crimps 8.5)."""
    Lm = L * 1e3
    rd, td = bonds.disc_radius * 1e3, 0.06 * Lm
    b = st.face_frac * Lm
    r_ring = st.r_ring * 1e3
    parts = {}
    parts["discs (2, beyond the skin)"] = (2 * math.pi * rd**2 * max(td - 0.4, 0.1), 2 * math.pi * rd**2 * td * 1.15e-3)
    parts["ratchet ring hubs (2)"] = (2 * math.pi * ((r_ring + 0.15) ** 2 - (r_ring - 0.6) ** 2) * b, 2 * math.pi * ((r_ring + 0.15) ** 2 - (r_ring - 0.6) ** 2) * b * 1.15e-3)
    n_pawls = 1 + 2 * st.phases + 1  # lock, drive pawls, latch
    parts["pawls, flexures, latch"] = (2 * n_pawls * 0.4 * 0.6 * 1.2, 2 * n_pawls * 0.4 * 0.6 * 1.2 * 1.15e-3)
    n_wires = 2 * st.phases
    parts["wire channels and crimps"] = (2 * n_wires * (0.15 * 0.15 * st.wire_length * 1e3 + 2 * 0.35**2 * 0.8), 2 * n_wires * 2 * 0.045 * 8.5e-3)
    n_ring = 6 * bonds.ring_count
    d_r = bonds.d_ring * 1e3
    parts[f"face-ring spheres ({n_ring} x {d_r:.2f} mm)"] = (n_ring * (d_r + 0.1) ** 3, n_ring * 7.5e-3 * math.pi / 6 * d_r**3)
    parts["steel pins (sprung, 2 per disc)"] = (4 * 0.4**2 * 1.2, 4 * 7.8e-3 * 0.4**2 * 1.2)
    vol = sum(v for v, _ in parts.values())
    mass = sum(m for _, m in parts.values())
    return parts, vol, mass


def packing(L_mm, joint_vol, base):
    """concepts.md packing margin with the joint's volume added: inner^3 x fill / (parts)."""
    inner = L_mm - 0.8
    fill = scaling.FILL_FLEX
    used = inner**3 * fill / base.margins["packing_volume"]
    return inner**3 * fill / (used + joint_vol), used


def cube(L_mm, st=None, omega_cap=20000.0):
    """Mass, packing and swarm margins of the SMA joint cube at L (mm)."""
    L = L_mm * 1e-3
    st = st or sma_design(L)
    bonds = nominal_bonds(L)
    base = rw_base(L_mm, omega_cap)
    if base is None:
        return None
    parts, vol, mass = sma_joint_parts(L, st, bonds)
    pack, used = packing(L_mm, vol, base)
    m = (base.mass_g + mass) * 1e-3
    # the rotors shrink to clear the face-ring spheres (radius 0.37 L - 0.07 L - 0.1 mm); the swarm
    # moves are re-run through concepts.md's inertial model with this rotor, this mass and these spheres
    r_old = base.info["rotor_r_mm"]
    r_new = min(r_old, (bonds.ring_radius - bonds.d_ring / 2) * 1e3 - 0.1)
    i_f = base.info["i_f"] * (r_new / r_old) ** 4
    sb = scaling.SphereBond(bonds.d_corner / 2, bonds.cover, L)
    st_info = base.info["storage"]
    p_cont = st_info["p_cont_w"] if st_info["p_cont_w"] > 0 else st_info["p_peak_w"] * 0.5
    swarm, _ = scaling.flywheel_block(L, m, i_f, omega_cap, sb, base.info["spin_torque_uNm"] * 1e-6, st_info["energy_j"], p_cont)
    bm, _ = scaling.bond_margins(L, m, sb)
    swarm.update(bm)
    return {"L_mm": L_mm, "mass_g": m * 1e3, "base_mass_g": base.mass_g, "joint_mass_g": mass, "joint_vol_mm3": vol, "base_used_mm3": used, "packing": pack, "parts": {k: [round(v, 2), round(w, 4)] for k, (v, w) in parts.items()}, "rotor_r_mm": r_new, "swarm": swarm, "mgL_mNm": m * G * L * 1e3}


# ----------------------------------------------------------------------------------------- routes


SIM_REF_L = 8.25  # mm, the nominal MuJoCo runs that set the measured needs


def sim_needs(L_ref=SIM_REF_L):
    """Needs measured in this session's nominal MuJoCo runs at L_ref (runs/module_phys/joint/
    machines_<L>mm.jsonl): peak smoothed joint torque in m g L, and peak bond utilisation by bond
    type with the capacities and cube mass of that run. None if the runs are not there."""
    path = OUT / f"machines_{L_ref:.2f}mm.jsonl"
    if not path.exists():
        path = OUT / f"machines_{L_ref:.2f}mm.jsonl"
    if not path.exists():
        return None
    rows = {json.loads(x)["machine"]: json.loads(x) for x in path.read_text().splitlines() if x.strip()}
    out = {}
    for name, r in rows.items():
        out[name] = {"hold": r["peak_joint_over_mgL"], "util_disc": r.get("peak_util_disc_bonds", 0.0), "util_plain": r.get("peak_util_plain_bonds", 0.0), "mass_g": r["mass_g"], "L_mm": r["L_mm"]}
    return out


def bond_scale(L_mm, m_kg, ref):
    """How a measured bond utilisation moves from the reference run to (L, m): loads go as m g L,
    capacities as the sphere force times L (spheres a fixed fraction of L)."""
    L, Lr = L_mm * 1e-3, ref["L_mm"] * 1e-3
    br, b = nominal_bonds(Lr), nominal_bonds(L)
    cap = b.plain()["peel_Nm"] / br.plain()["peel_Nm"]
    load = (m_kg * L) / (ref["mass_g"] * 1e-3 * Lr)
    return load / cap


def sma_route(L_mm, st=None):
    L = L_mm * 1e-3
    st = st or sma_design(L)
    c = cube(L_mm, st)
    if c is None:
        return None
    mgl = c["mgL_mNm"] * 1e-3
    bonds = nominal_bonds(L)
    tab = st.table()
    det = bonds.detent()
    out = {"route": "sma ratchet stepper", "L_mm": L_mm, "design": tab, "cube": c, "detent_mNm": round(det * 1e3, 3)}
    margins = {}
    for name, need in NEEDS_B.items():
        margins[name] = {
            "drive": (st.drive - det) / (need["drive"] * mgl),  # the detent opposes the stroke at its worst angle
            "hold": st.hold(True) / (need["hold"] * mgl),
            "bond_disc": bonds.disc()["peel_Nm"] / (need["bond"] * mgl),
            "bond_plain": bonds.plain()["peel_Nm"] / (need["bond"] * mgl),
        }
    sim = sim_needs()
    if sim:  # measured needs replace layout B's where this session has them
        for name, ref in sim.items():
            if name not in margins:
                continue
            k = bond_scale(L_mm, mgl / (G * L), ref)
            margins[name]["hold"] = st.hold(True) / max(ref["hold"] * mgl, 1e-12)
            margins[name]["bond_disc"] = 1.0 / max(ref["util_disc"] * k, 1e-9)
            margins[name]["bond_plain"] = 1.0 / max(ref["util_plain"] * k, 1e-9)
        out["needs_source"] = f"hold and bonds: MuJoCo nominal runs at {SIM_REF_L} mm; drive: layout B quasi-static torques"
    else:
        out["needs_source"] = "layout B (module-demos.md)"
    out["margins"] = margins
    out["arm_move_J"] = ARM_TRAVEL_DEG / tab["step_deg"] * st.step_energy
    out["arm_move_s"] = ARM_TRAVEL_DEG / tab["speed_deg_s"]
    return out


def gear_route(L_mm):
    L = L_mm * 1e-3
    d = nominal_design(L)
    t = d.table()
    c = cube(L_mm)
    mgl = c["mgL_mNm"] * 1e-3 if c else float("nan")
    w = d.train.meshes[-1]
    h = L / 2 - 0.4e-3
    # packing rules (E, from the layout): the Didel worm sits 3.7 mm from the sun axis and is
    # 2.8 mm across, under a rotor slab; the split ring's module must print
    worm_fits = 3.7e-3 + 1.4e-3 + 0.2e-3 <= h - d.motor.t_r
    module_ok = w.module >= 0.25e-3
    module_test = w.module >= 0.20e-3
    # energy of an arm move: copper loss at the arm torque, the motor at 60 % of no-load speed
    t_arm = NEEDS_B["arm"]["drive"] * mgl
    tau_m = t_arm / (d.train.ratio * d.train.efficiency())
    p_cu = (tau_m / d.motor.k_m) ** 2
    speed = 0.6 * d.speed_out()
    move_s = math.radians(ARM_TRAVEL_DEG) / speed
    margins = {name: {"drive": min(d.stall_out(), t["tooth_cont_out_mNm"] * 1e-3) / (need["drive"] * mgl), "hold": t["tooth_peak_out_mNm"] * 1e-3 / (need["hold"] * mgl)} for name, need in NEEDS_B.items()}
    return {"route": "micro gear train (Didel worm + printed split ring)", "L_mm": L_mm, "design": t, "printed_module_mm": round(w.module * 1e3, 3), "worm_fits": worm_fits, "module_trusted": module_ok, "module_test_print": module_test, "margins": margins, "arm_move_J": (p_cu + t_arm * speed / max(d.train.efficiency(), 1e-6)) * move_s, "arm_move_s": move_s, "speed_deg_s": math.degrees(speed)}


BOUGHT = [
    # name, dims (mm: dia or w, h, length), torque continuous, torque max (N m), speed rad/s, mass g, price USD, tag
    ("Precision Microdrives 206-10C 6 mm, 699.5:1", (6.0, 6.0, 20.8), 10e-3, 25e-3, 37 * 2 * math.pi / 60, 1.2, None, "V: nominal 10 mN m, 37 rpm, 20.8 mm, 1.2 g (product page); 25 mN m intermittent (datasheet snippet, V secondary)"),
    ("Pololu 1000:1 Micro Metal Gearmotor HP 6 V", (10.0, 12.0, 26.0 + 3.5), 0.4 * 245e-3, 245e-3, 31 * 2 * math.pi / 60, 10.0, 36.95, "V: 2.5 kg cm gearbox limit (245 mN m), 31 rpm, 10 x 12 mm section (pololu.com/product/2373); length A; continuous 40 % of limit A"),
]


def bought_route():
    rows = []
    for name, dims, t_cont, t_max, speed, mass_g, price, tag in BOUGHT:
        L_fit = max(dims) + 0.8 + 0.6  # the motor along one inner side, skin and clearance (E)
        m = 2400.0 * (L_fit * 1e-3) ** 3 + 2 * mass_g * 1e-3  # E: cube density of the concept designs plus two motors
        mgl = m * G * L_fit * 1e-3
        margins = {k: {"drive": t_cont / (v["drive"] * mgl), "hold": t_max / (v["hold"] * mgl)} for k, v in NEEDS_B.items()}
        rows.append({"route": "bought gearmotor", "name": name, "min_fit_L_mm": round(L_fit, 1), "mass_g_est": round(m * 1e3, 1), "speed_deg_s": round(math.degrees(speed), 1), "margins_at_fit": margins, "tag": tag})
    return rows


def piezo_route(L_mm):
    """Stick-slip friction ring (E/A): a 2 x 3 x 5 mm multilayer stack (class of Tokin AE0203D04F:
    about 4 um at 150 V, blocking force about 200 N, A from memory) pushes a ring through a
    friction pad. Torque = mu x preload x r; the preload is limited because the stack must slip
    the ring back in the fast phase, so drive and holding torque are the same friction torque."""
    L = L_mm * 1e-3
    r = 0.33 * L
    preload, mu = 2.0, 0.2  # N (a resin flexure clamp), friction resin on alumina (A)
    t_hold = mu * preload * r
    step = 3e-6 / r  # rad per cycle
    freq = 2000.0  # Hz sawtooth, limited by the 150 V driver's slew into 50-100 nF (A)
    return {"route": "piezo stick-slip ring", "L_mm": L_mm, "torque_mNm": round(t_hold * 1e3, 3), "holding_mNm": round(t_hold * 1e3, 3), "speed_deg_s": round(math.degrees(step * freq), 1), "voltage_V": 150, "power_off_hold": "friction only, equal to the drive torque (no margin against a touchdown)", "volume_mm3": 30 + 15}


# ----------------------------------------------------------------------------------------- search

D_CHOICES = sorted(FLEXINOL)


def design_from(x):
    """x = (L mm, wire index, strands, ring_frac, pitch mm, phases)."""
    L = float(x[0]) * 1e-3
    d = D_CHOICES[int(round(np.clip(x[1], 0, len(D_CHOICES) - 1)))]
    strands = int(round(x[2]))
    return L, sma_design(L, d_wire=d, strands=strands, ring_frac=float(x[3]), pitch=float(x[4]) * 1e-3, phases=int(round(x[5])))


def objectives(x):
    """Minimise L, energy per degree and step period; constraints become penalties: every margin
    (drive, hold, bonds, packing, swarm moves) >= 1.5, speed >= 15 deg/s, pitch printable."""
    L, st = design_from(x)
    c = cube(L * 1e3, st)
    if c is None:
        return [99.0, 99.0, 99.0]
    mgl = c["mgL_mNm"] * 1e-3
    bonds = nominal_bonds(L)
    det = bonds.detent()
    worst = []
    for need in NEEDS_B.values():
        worst += [(st.drive - det) / (need["drive"] * mgl), st.hold(True) / (need["hold"] * mgl)]
    worst += [c["packing"], min(c["swarm"].values())]
    speed = math.degrees(st.speed)
    lever_ok = st.lever <= 3.0
    pen = sum(max(0.0, MARGIN - w) for w in worst) * 10 + max(0.0, 15 - speed) * 0.1 + (0 if lever_ok else 5)
    return [L * 1e3 + pen, st.step_energy / math.degrees(st.step) * 1e3 + pen, st.step_period + pen]


def search():
    lower = [6.0, 0, 2, 0.25, 0.15, 1]
    upper = [12.0, len(D_CHOICES) - 1, 16, 0.38, 0.30, 2]
    res = run_search(objectives, lower, upper, n_obj=3, algo="nsga2", pop=48, n_gen=40, workers=4, seed=3)
    rows = []
    for x, f in zip(res.X, res.F):
        if f[0] > 50 or f[0] != f[0]:
            continue
        L, st = design_from(x)
        if objectives(x)[0] - L * 1e3 > 1e-9:  # infeasible (penalised)
            continue
        rows.append({"L_mm": round(L * 1e3, 3), "wire_mm": st.d_wire * 1e3, "strands": st.strands, "ring_frac": round(st.ring_frac, 3), "pitch_mm": round(st.pitch * 1e3, 3), "phases": st.phases, "energy_mJ_per_deg": round(f[1], 3), "step_period_s": round(f[2], 3), "speed_deg_s": round(math.degrees(st.speed), 1)})
    rows.sort(key=lambda r: r["L_mm"])
    return rows, res.n_evals


def grid_min_L(st_kw=None, step=0.25):
    """Smallest L on a 0.25 mm grid where the nominal SMA design passes, and the pass or fail (with
    the binding margin) at every grid point."""
    st_kw = st_kw or {}
    passing = []
    for L_mm in np.arange(6.0, 12.01, step):
        r = sma_route(float(L_mm), sma_design(L_mm * 1e-3, **st_kw))
        if r is None:
            passing.append((float(L_mm), False, "no reaction-wheel layout"))
            continue
        worst = min(min(v.values()) for v in r["margins"].values())
        pack = r["cube"]["packing"]
        swarm = min(r["cube"]["swarm"].values())
        ok = worst >= MARGIN and pack >= MARGIN and swarm >= MARGIN
        why = min([("machine " + k + " " + kk, vv) for k, v in r["margins"].items() for kk, vv in v.items()] + [("packing", pack), ("swarm " + min(r["cube"]["swarm"], key=r["cube"]["swarm"].get), swarm)], key=lambda t: t[1])
        passing.append((float(L_mm), ok, f"{why[0]} {why[1]:.2f}"))
    first = next((L_mm for L_mm, ok, _ in passing if ok), None)  # the passing set is a window, not a half-line: spin time closes it from above
    return first, passing


# --------------------------------------------------------------------------------------- envelope


def envelope(L_mm, write=True):
    """CadQuery envelope of the SMA joint cube: every part as a box, cylinder or sphere at its
    size, checked for overlaps and for staying inside the cavity. Flywheels on +x and +y (as
    concepts.md pick 1), joint discs on +z and -z (the simulation's disc axis is the cube's z)."""
    import cadquery as cq

    L = L_mm * 1e-3
    st = sma_design(L)
    bonds = nominal_bonds(L)
    c = cube(L_mm, st)
    base = rw_base(L_mm)
    wall = 0.40
    inner = L_mm - 2 * wall
    h = inner / 2
    t_f = base.info["rotor_t_mm"]
    r_f = c["rotor_r_mm"]
    d_c = bonds.d_corner * 1e3
    d_r = bonds.d_ring * 1e3
    rho = bonds.ring_radius * 1e3
    rd = bonds.disc_radius * 1e3
    b = st.face_frac * L_mm
    r_ring = st.r_ring * 1e3
    parts = {}
    cc = L_mm / 2 - 0.05 - d_c / 2
    for i, (sx, sy, sz) in enumerate([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]):
        parts[f"corner{i}"] = cq.Workplane().sphere(d_c / 2).translate((sx * cc, sy * cc, sz * cc))
    gap = 0.10
    parts["rotorX"] = cq.Workplane("YZ").circle(r_f).extrude(t_f).translate((h - gap - t_f, 0, 0))
    parts["rotorY"] = cq.Workplane("XZ").circle(r_f).extrude(-t_f).translate((0, h - gap - t_f, 0))
    # face-ring spheres: 8 per face at radius rho, depth cover + r; the +-z ones ride in the discs
    depth = L_mm / 2 - 0.05 - d_r / 2
    for face, (ax, sign) in {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1), "+z": (2, 1), "-z": (2, -1)}.items():
        for k in range(bonds.ring_count):
            a = 2 * math.pi * k / bonds.ring_count + math.pi / 4
            p = [0.0, 0.0, 0.0]
            others = [i for i in range(3) if i != ax]
            p[ax] = sign * depth
            p[others[0]], p[others[1]] = rho * math.cos(a), rho * math.sin(a)
            parts[f"ring{face}{k}"] = cq.Workplane().sphere(d_r / 2).translate(tuple(p))
    # joints on +z and -z: disc (in the face, ring spheres embedded: modelled as a thin plate under
    # them), ratchet ring hub, lock pawl, drive pawls, wires with crimps
    for sign, nm in ((1, "top"), (-1, "bottom")):
        z_face = sign * L_mm / 2
        disc_t = 0.06 * L_mm
        z0 = z_face - sign * disc_t
        parts[f"disc_{nm}"] = cq.Workplane("XY").circle(rd).circle(rho + d_r / 2 + 0.05).extrude(disc_t).translate((0, 0, min(z0, z_face)))
        # the ratchet flange sits below the disc's ring spheres; a hub inside the spheres joins them
        z_sph = sign * (depth - d_r / 2 - 0.05)
        z_hub0 = z_sph - sign * b
        parts[f"ring_{nm}"] = cq.Workplane("XY").circle(r_ring + 0.15).circle(r_ring - 0.6).extrude(b).translate((0, 0, min(z_hub0, z_sph)))
        parts[f"hub_{nm}"] = cq.Workplane("XY").circle(rho - d_r / 2 - 0.1).circle(r_ring - 0.6).extrude(abs(z0 - z_sph)).translate((0, 0, min(z0, z_sph)))
        zc = (z_hub0 + z_sph) / 2
        rp = r_ring + 0.15 + 0.35  # pawl centres just outside the teeth, on the -x / -y side (the rotors are on +x and +y)
        for k, ang in enumerate(np.linspace(math.pi, 1.5 * math.pi, 1 + 2 * st.phases)):
            parts[f"pawl_{nm}{k}"] = cq.Workplane().box(0.6, 0.4, b).rotate((0, 0, 0), (0, 0, 1), math.degrees(ang)).translate((rp * math.cos(ang), rp * math.sin(ang), zc))
        # wires run vertically beside the four vertical edges (a bell crank at each end turns the
        # pull into the pawl's stroke); the two joints' wires share the edge regions
        lw = st.wire_length * 1e3
        for k in range(2 * st.phases):
            ex, ey = [(-1, -1), (1, -1), (-1, 1), (1, 1)][k]
            off = 0.18 * sign
            parts[f"wire_{nm}{k}"] = cq.Workplane().box(0.12, 0.12, lw).translate((ex * (h - 0.45) + off, ey * (h - 0.45) - off, 0.0))
    # storage and electronics in the middle band
    for k in range(3):
        parts[f"cera{k}"] = cq.Workplane().box(4.4, 3.2, 1.1).translate((-0.6, -0.9, -1.12 + 1.12 * k))
    parts["mcu"] = cq.Workplane().box(0.8, 3.0, 3.0).translate((-h + 0.4, 0.8, 0.0))
    parts["imu_power"] = cq.Workplane().box(3.0, 0.7, 2.0).translate((0.5, -h + 0.35, 0.0))
    names = list(parts)
    boxes = {n: parts[n].val().BoundingBox() for n in names}
    overlaps = []
    for i, a in enumerate(names):
        for bb in names[i + 1:]:
            A, B = boxes[a], boxes[bb]
            if A.xmax < B.xmin or B.xmax < A.xmin or A.ymax < B.ymin or B.ymax < A.ymin or A.zmax < B.zmin or B.zmax < A.zmin:
                continue
            inter = parts[a].intersect(parts[bb])
            v = inter.val().Volume() if inter.vals() else 0.0
            if v > 1e-4:
                overlaps.append((a, bb, round(v, 4)))
    outside = []
    cavity = cq.Workplane().box(L_mm, L_mm, L_mm)
    for n in names:
        o = parts[n].cut(cavity)
        v = o.val().Volume() if o.vals() else 0.0
        if v > 1e-4:
            outside.append((n, round(v, 4)))
    vol = sum(parts[n].val().Volume() for n in names)
    result = {"L_mm": L_mm, "overlaps": overlaps, "outside_cube": outside, "parts": len(names), "parts_volume_mm3": round(vol, 1), "inner_mm3": round(inner**3, 1), "rotor_r_mm": round(r_f, 2), "rotor_t_mm": round(t_f, 2)}
    if write:
        assy = cq.Assembly()
        for n, w in parts.items():
            assy.add(w, name=n)
        HW.mkdir(parents=True, exist_ok=True)
        path = HW / f"sma_joint_cube_{L_mm:g}mm_envelope.step"
        assy.save(str(path))
        result["step"] = str(path.relative_to(ROOT))
    return result


# ------------------------------------------------------------------------------------------- main


def mc_rows(L_mm):
    path = OUT / f"mc_{L_mm:.2f}mm.jsonl"
    if not path.exists():
        return {}
    sys.path.insert(0, str(ROOT / "scripts"))
    from small_joint_machines import summarise

    return summarise(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc", action="store_true")
    parser.add_argument("--chosen", type=float, default=None, help="chosen L (mm); default the grid minimum")
    parser.add_argument("--no-search", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    sweep = {}
    for L_mm in (6.0, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0, 12.0, 14.0):
        s = sma_route(L_mm)
        g = gear_route(L_mm)
        p = piezo_route(L_mm)
        sweep[L_mm] = {"sma": s, "gear": g, "piezo": p}
    first, grid = grid_min_L()
    chosen = args.chosen or first
    print("SMA grid (L, passes, binding):")
    for L_mm, ok, why in grid:
        print(f"  {L_mm:5.2f} {'pass' if ok else 'fail'}  {why}")
    print("minimum L, SMA stepper, nominal design:", first)
    gear_min = None
    for L_mm in np.arange(8.0, 20.01, 0.5):
        g = gear_route(float(L_mm))
        worst = min(min(v.values()) for v in g["margins"].values())
        if g["worm_fits"] and g["module_test_print"] and worst >= MARGIN:
            gear_min = float(L_mm)
            break
    print("minimum L, gear route:", gear_min)
    search_rows, n_evals = ([], 0) if args.no_search else search()
    print(f"NSGA-II: {n_evals} evaluations, {len(search_rows)} feasible on the final front; smallest:", search_rows[:3])
    env = envelope(chosen)
    print("envelope:", json.dumps(env))
    # mass fit for the simulation (m = a L^3 + b L^2 over 7-12 mm)
    Ls = np.array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    ms = np.array([cube(L)["mass_g"] * 1e-3 for L in Ls])
    A = np.column_stack([(Ls * 1e-3) ** 3, (Ls * 1e-3) ** 2])
    a, bb = np.linalg.lstsq(A, ms, rcond=None)[0]
    (OUT / "mass_fit.json").write_text(json.dumps({"a": float(a), "b": float(bb), "note": "SMA joint cube mass (kg) = a L^3 + b L^2, L in m, fitted to scripts/module_phys/small_joint.py cube() at 7-12 mm"}))
    # results.jsonl: one line per design and machine
    mc = mc_rows(chosen) if args.mc else {}
    machines_path = OUT / f"machines_{chosen:.2f}mm.jsonl"
    sim = {json.loads(x)["machine"]: json.loads(x) for x in machines_path.read_text().splitlines()} if machines_path.exists() else {}
    for L_mm, r in sweep.items():
        s = r["sma"]
        if s:
            for name, m in s["margins"].items():
                rows.append({"route": "sma", "design": "SMA ratchet stepper, nominal", "machine": name, "L_mm": L_mm, "min_edge_mm": first, "margins": {k: round(v, 3) for k, v in m.items()}, "packing": round(s["cube"]["packing"], 3), "swarm_min": round(min(s["cube"]["swarm"].values()), 3), "mass_g": round(s["cube"]["mass_g"], 3), "step_deg": s["design"]["step_deg"], "speed_deg_s": s["design"]["speed_deg_s"], "arm_move_J": round(s["arm_move_J"], 3), "source": "analytic (E)"})
        g = r["gear"]
        for name, m in g["margins"].items():
            rows.append({"route": "gear", "design": g["route"], "machine": name, "L_mm": L_mm, "min_edge_mm": gear_min, "margins": {k: round(v, 3) for k, v in m.items()}, "worm_fits": g["worm_fits"], "printed_module_mm": g["printed_module_mm"], "speed_deg_s": round(g["speed_deg_s"], 1), "arm_move_J": round(g["arm_move_J"], 3), "source": "analytic (E)"})
        p = r["piezo"]
        rows.append({"route": "piezo", "design": p["route"], "machine": "all", "L_mm": L_mm, "min_edge_mm": None, "torque_mNm": p["torque_mNm"], "speed_deg_s": p["speed_deg_s"], "source": "analytic (E/A)"})
    for b in bought_route():
        for name, m in b["margins_at_fit"].items():
            rows.append({"route": "bought", "design": b["name"], "machine": name, "L_mm": b["min_fit_L_mm"], "min_edge_mm": b["min_fit_L_mm"] if min(m.values()) >= MARGIN else None, "margins": {k: round(v, 3) for k, v in m.items()}, "speed_deg_s": b["speed_deg_s"], "source": b["tag"]})
    for name, row in sim.items():
        out = {"route": "sma", "design": "SMA ratchet stepper, nominal", "machine": name, "L_mm": chosen, "min_edge_mm": first, "source": "MuJoCo nominal run (measured)"}
        out.update({k: row[k] for k in ("ok", "ok_1.5x", "bond_opening", "bond_opening_at_1.5x", "peak_joint_over_capacity", "peak_joint_over_mgL", "peak_bond_utilisation", "worst_bond", "energy_J", "steps", "failed_steps", "measured") if k in row})
        if name in mc:
            out["monte_carlo"] = mc[name]
        rows.append(out)
    # headline row: the strict grid (every machine's measured bond loads with 1.5x) passes nowhere,
    # bound by the walker's teeter at leg lift; the joint itself and the other three machines pass
    # with 1.5x from 8.25 mm (packing below, spin time above)
    rows.append({"route": "sma", "design": "SMA ratchet stepper, 18 x 25 um", "machine": "summary", "min_edge_mm": chosen,
                 "condition": "joint loads under tooth limit / 1.5 and arm, crane, crank bonds at strength / 1.5 in nominal runs; walker holds at full strength only",
                 "min_edge_mm_strict": first, "strict_binding": "walker bond (teeter at leg lift), not the joint", "source": "analysis + MuJoCo (E)"})
    (OUT / "results.jsonl").write_text("".join(json.dumps(r, default=float) + "\n" for r in rows))
    summary = {"sma_min_L_mm": first, "gear_min_L_mm": gear_min, "chosen_L_mm": chosen, "sma_at_chosen": sma_route(chosen), "gear_at_12": sweep[12.0]["gear"], "gear_at_chosen": gear_route(chosen), "piezo_at_chosen": piezo_route(chosen), "bought": bought_route(), "search_front": search_rows[:20], "search_evals": n_evals, "envelope": env, "grid": grid, "worm_last_at_12": worm_last_design(12e-3).table(), "bonds_at_chosen": nominal_bonds(chosen * 1e-3).summary(), "mass_fit": {"a": a, "b": bb}}
    (OUT / "analysis.json").write_text(json.dumps(summary, indent=1, default=float))
    print("wrote", OUT / "results.jsonl", OUT / "analysis.json")


if __name__ == "__main__":
    main()
