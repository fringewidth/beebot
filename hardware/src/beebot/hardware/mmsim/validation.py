"""Validation rows against published data. Run: .venv/bin/python -m beebot.hardware.mmsim.validation

Each row: quantity, published value and source URL, simulated value, error. Pull-force tolerance 15 %,
electrical 10 %. Rows with group 'verification' compare against an independent analytic calculation, not a
published value, and are not in the headline count.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import magpylib as mp
import numpy as np

from . import circuit, coil
from .fem import pull_force_axisym
from .magnetics import Magnet, pull_force_magnet_magnet, pull_force_to_steel, wrench

LB = 4.4482216152605
G = 9.80665
TOL = {"pull": 0.15, "electrical": 0.10, "thermal": 0.15, "verification": 0.05}

KJ = "https://www.kjmagnetics.com/"
SM = "https://www.supermagnete.de/eng/disc-magnets-neodymium/"
AEB = "https://www.aemjournal.org/index.php/AEM/article/view/1074 (Aebischer, Adv. Electromagn. 8(4), 2019, tables 3 and 4)"


@dataclass
class Row:
    id: str
    group: str
    quantity: str
    published: float
    unit: str
    source: str
    simulated: float
    error_pct: float
    tol_pct: float
    passed: bool
    held_out: bool
    note: str = ""


def mk(id, group, quantity, published, unit, source, simulated, held_out=True, note=""):
    err = (simulated - published) / published * 100.0
    tol = TOL[group] * 100
    return Row(id, group, quantity, float(published), unit, source, float(simulated), float(err), tol,
               bool(abs(err) <= tol), held_out, note)


# magnets: (id, kind, dim (m), grade, published N, source, note)
STEEL_CASES = [
    ("kj_d101_n52", "cyl", (1.5875e-3, 0.7938e-3), "N52", 0.07 * LB, KJ + "proddetail.asp?prod=d101-n52",
     "K&J D101-N52 1/16 x 1/32 in, case 1 (thick steel plate), 0.07 lb"),
    ("kj_d201_n52", "cyl", (3.175e-3, 0.7938e-3), "N52", 0.25 * LB, KJ + "d201-n52-neodymium-disc-magnet",
     "0.25 lb read from a similar-products listing through a web-search snippet; case number not confirmed"),
    ("kj_b444_n52", "cuboid", (6.35e-3,) * 3, "N52", 5.26 * LB, KJ + "b444-n52-neodymium-block-magnet",
     "1/4 in cube, case 1 5.26 lb (outside the 1-5 mm size range)"),
    ("kj_b666_n52", "cuboid", (9.525e-3,) * 3, "N52", 12.46 * LB, KJ + "b666-n52-neodymium-block-magnet",
     "3/8 in cube, case 1 12.46 lb (outside the 1-5 mm size range)"),
    ("sm_s0502_n52", "cyl", (5e-3, 2e-3), "N52", 0.680 * G, SM + "disc-magnet-5mm-2mm_S-05-02-N52N",
     "supermagnete 5x2 N52, adhesive force approx 680 g"),
    ("sm_s0502_n40", "cyl", (5e-3, 2e-3), "N40", 0.520 * G, SM + "disc-magnet-5mm-2mm_S-05-02-N40N",
     "supermagnete 5x2 N40, approx 520 g"),
    ("sm_s0302_n48", "cyl", (3e-3, 2e-3), "N48", 0.300 * G,
     "https://www.supermagnete.nl/eng/disc-magnets-neodymium/disc-magnet-3mm-2mm_S-03-02-N",
     "supermagnete 3x2 N48, approx 300 g"),
    ("sm_s0202_n48", "cyl", (2e-3, 2e-3), "N48", 0.150 * G, SM + "disc-magnet-2mm-2mm_S-02-02-N",
     "supermagnete 2x2 N48, approx 150 g"),
]
BR = {"N52": 1.45, "N48": 1.40, "N40": 1.27}
BR_KJ_N52 = 1.48  # K&J quotes Brmax 14800 G for N52 magnets on its pages
CAL_IDS = ("kj_b444_n52", "kj_b666_n52", "sm_s0502_n52")  # used to set the single derating factor


def _mesh(kind):
    return (24, 4, 8) if kind == "cyl" else (8, 8, 8)


def steel_rows():
    rows = []
    ratios = {}
    sims = {}
    for cid, kind, dim, grade, pub, src, note in STEEL_CASES:
        br = BR_KJ_N52 if cid.startswith("kj_") and grade == "N52" else BR[grade]
        sim = pull_force_to_steel(kind, dim, br, gap=0.0, mag_mesh=_mesh(kind))
        sims[cid] = (sim, pub, src, note, dim, br)
        ratios[cid] = pub / sim
    for cid, (sim, pub, src, note, dim, br) in sims.items():
        rows.append(mk(cid + "_image", "pull", f"pull to steel, image method, ideal iron, {cid}", pub, "N", src,
                       sim, held_out=True, note=note + f"; Br {br} T, gap 0, no tuning"))
    # calibrated derating on three rows, scored on the others
    k = float(np.median([ratios[c] for c in CAL_IDS]))
    for cid, (sim, pub, src, note, dim, br) in sims.items():
        held = cid not in CAL_IDS
        rows.append(mk(cid + "_derated", "pull", f"pull to steel, image method x derate {k:.3f}, {cid}", pub, "N",
                       src, sim * k, held_out=held,
                       note=("held-out" if held else "calibration row, not independent") + f"; derate set on {CAL_IDS}"))
    return rows, k, ratios


def fem_rows():
    rows = []
    for cid in ("sm_s0502_n52", "sm_s0302_n48"):
        row = [c for c in STEEL_CASES if c[0] == cid][0]
        _, kind, dim, grade, pub, src, note = row
        res = pull_force_axisym(dim[0], dim[1], BR[grade], gap=20e-6, mu_r=None, maxh_fine=dim[1] / 24)
        rows.append(mk(cid + "_fem", "pull", f"pull to steel, NGSolve axisymmetric FEM nonlinear generic steel, {cid}",
                       pub, "N", src, res["force"], held_out=True,
                       note=note + f"; gap 20 um plating, 5 mm plate, Brauer B-H (not a specific grade), "
                                   f"peak B in steel {res['b_steel_max']:.2f} T, {res['mesh_elements']} elements"))
    return rows


def pole_model_force(d, h, br, n=300):
    """Independent check: surface-pole (charge) model for two coaxial touching identical cylinders."""
    a = mp.magnet.Cylinder(dimension=(d, h), polarization=(0, 0, br), position=(0, 0, 0))
    xs = (np.arange(n) + 0.5) / n * d - d / 2
    X, Y = np.meshgrid(xs, xs)
    m = (X**2 + Y**2) <= (d / 2) ** 2
    dA = (d / n) ** 2

    def bz(z):
        P = np.column_stack([X[m], Y[m], np.full(m.sum(), z)])
        return a.getB(P)[:, 2].sum() * dA

    return -(br / (4e-7 * np.pi)) * (bz(1.5 * h) - bz(0.5 * h))


def magnet_verification_rows():
    rows = []
    for d, h, br in [(1e-3, 1e-3, 1.45), (3e-3, 2e-3, 1.45), (5e-3, 2e-3, 1.40)]:
        ref = pole_model_force(d, h, br)
        sim = pull_force_magnet_magnet("cyl", (d, h), br, 0.0, mag_mesh=(24, 4, 8))
        rows.append(mk(f"mm_pole_{d*1e3:.0f}x{h*1e3:.0f}", "verification",
                       f"magnet-magnet contact pull, getFT mesh vs surface-pole integral, {d*1e3:.0f}x{h*1e3:.0f} mm",
                       ref, "N", "internal: independent pole-model integral (not published)", sim,
                       note="analytic cross-check, no published magnet-to-magnet pull value was found"))
    # image method equals magnet-to-magnet by construction; FEM linear vs image for steel (verification)
    res = pull_force_axisym(5e-3, 2e-3, 1.45, gap=100e-6, mu_r=1000.0, maxh_fine=2e-3 / 24, mu_magnet=1.0)
    img = pull_force_to_steel("cyl", (5e-3, 2e-3), 1.45, 100e-6, mu_r=1000.0, mag_mesh=(24, 4, 8))
    rows.append(mk("fem_vs_image_linear", "verification", "linear steel mu_r 1000: FEM vs image method, 5x2 N52, gap 0.1 mm",
                   img, "N", "internal", res["force"], note="FEM with magnet mu_r 1.0; difference is mesh and finite mu_r"))
    return rows


# ---------------------------------------------------------------- electrical

AEB_ROWS = [  # N, A mm, w mil (pitch), s mil (trace), published precise L nH (FastHenry, Aebischer table 4)
    (3, 20, 12, 6, 542), (3, 20, 20, 6, 480), (3, 20, 30, 6, 426), (3, 20, 40, 6, 385), (3, 20, 30, 12, 401),
    (3, 50, 12, 6, 1716), (3, 50, 20, 6, 1575), (3, 50, 40, 6, 1372), (3, 50, 20, 12, 1510),
    (3, 50, 30, 12, 1391), (3, 50, 40, 12, 1305), (4, 50, 40, 12, 2037), (5, 50, 40, 12, 2849),
    (3, 100, 40, 12, 3173), (5, 10, 12, 6, 446), (3, 50, 30, 18, 1352),
]


def inductance_rows(methods=("flux", "mohan", "wheeler", "aebischer")):
    mil = 25.4e-6
    rows = []
    fn = {"flux": coil.inductance_flux, "mohan": coil.inductance_mohan, "wheeler": coil.inductance_wheeler,
          "aebischer": coil.inductance_aebischer}
    for k, (N, A, w, s, pub) in enumerate(AEB_ROWS, 1):
        sp = coil.Spiral(N, A * 1e-3, s * mil, (w - s) * mil, cu_t=35e-6)
        for mth in methods:
            sim = fn[mth](sp) * 1e9
            rows.append(mk(f"L_aeb{k:02d}_{mth}", "electrical",
                           f"inductance, square spiral N={N} A={A} mm pitch {w} mil trace {s} mil, {mth}",
                           pub, "nH", AEB, sim, held_out=True,
                           note="published = FastHenry DC value; measured values agree within 2.1 %"))
    return rows


def thermal_rows():
    rows = []
    sp = coil.Spiral(5, 4e-3, 75e-6, 75e-6, cu_t=18e-6, layers=2)
    area = sp.trace * sp.cu_t
    from scipy.optimize import brentq
    for t in (1e-4, 1e-3, 1e-2, 1e-1):
        f = lambda I: coil.peak_temperature(sp, I, t, include_pi=False, adiabatic=True, t_amb=20.0) - 1083.0
        sim = brentq(f, 0.01, 100)
        pub = coil.onderdonk_current(area, t)
        rows.append(mk(f"onderdonk_{t:g}s", "thermal",
                       f"copper trace 75 um x 18 um fusing current for {t:g} s, adiabatic limit of the thermal model",
                       pub, "A", "https://pcdandf.com/pcdesign/index.php/editorial/menu-features/17314-onderdonk-s-and-preece-s-equations-how-do-they-compare",
                       sim, held_out=True,
                       note="Onderdonk equation; same physics as the model's adiabatic limit, a weak check. "
                            "Convection and polyimide terms are not validated against published data"))
    return rows


def circuit_rows():
    rows = []
    st = circuit.mlcc_bank(4, 10e-6, 5e-3, 3.3, 6.3)
    drv = circuit.Driver(0.1)
    r_c, l_c = 1.67, 0.6e-6
    res = circuit.simulate_pulse(st, r_c, l_c, 20e-6, drv)
    R = r_c + 2 * 0.1 + st.esr
    # analytic RLC series discharge: solve with fine independent integration (matrix exponential)
    from scipy.linalg import expm
    A_ = np.array([[0, -1 / st.c_f], [1 / l_c, -R / l_c]])  # [vc, i]: dvc/dt = -i/C ; di/dt = (vc - R i)/L
    A_ = np.array([[0, -1 / st.c_f], [1 / l_c, -R / l_c]])
    x20 = expm(A_ * 20e-6) @ np.array([3.3, 0.0])
    ts = np.linspace(0, 20e-6, 400)
    ipeak = max((expm(A_ * t) @ np.array([3.3, 0.0]))[1] for t in ts)
    rows.append(mk("rlc_peak_current", "verification", "series RLC discharge peak current (matrix exponential)",
                   ipeak, "A", "internal analytic", float(np.abs(res["i"][res["t"] <= 20e-6]).max()),
                   note="mlcc bank 4x10 uF, ESR 5 mOhm/4, coil 1.67 ohm 0.6 uH, 2x0.1 ohm Rds"))
    rows.append(mk("rlc_cap_voltage", "verification", "capacitor open-circuit voltage after 20 us",
                   x20[0], "V", "internal analytic", float(res["v_open"][np.searchsorted(res["t"], 20e-6)]),
                   note="sampled at the nearest output time"))
    bal = st.c_f * (3.3**2 - res["v_end"] ** 2) / 2
    loss = res["e_coil_loss"] + res["e_switch_loss"] + st.esr * np.trapezoid(res["i"] ** 2, res["t"])
    rows.append(mk("energy_balance", "verification", "energy balance: capacitor energy drop vs dissipation",
                   bal, "J", "internal", float(loss)))
    return rows


# ---------------------------------------------------------------- benchmarks

def random_cube_pair(rng, n=12):
    """Two 5 mm cubes, n 1 mm x 1 mm cylinder magnets each on the facing faces (random axes along x), 1 mm gap."""
    from .mjcoupling import rot_z_to
    from scipy.spatial.transform import Rotation as R

    def mags(sign):
        out = []
        for _ in range(n):
            y, z = rng.uniform(-2.0e-3, 2.0e-3, 2)
            pol = rng.choice([-1.0, 1.0])
            out.append(Magnet("cyl", (1e-3, 1e-3), 1.45, (sign * 2.0e-3, y, z), rot_z_to([pol, 0, 0])))
        return out

    a = [m for m in mags(+1)]
    b = [Magnet(m.kind, m.dim, m.br, (m.pos[0] - 6e-3 - 4e-3 + 4e-3 * 0, m.pos[1], m.pos[2]), m.rot) for m in mags(-1)]
    # B at +6 mm along x: positions are local; shift
    b = [Magnet("cyl", (1e-3, 1e-3), 1.45, (6e-3 + (-2.0e-3), p[1], p[2]), r) for p, r in
         [((0, m.pos[1], m.pos[2]), m.rot) for m in b]]
    return a, b


def speed_and_interp(n_pairs=30, table=None, seed=0):
    from .mjcoupling import CylPairTable

    rng = np.random.default_rng(seed)
    if table is None:
        table = CylPairTable(1e-3, 1e-3, 1.45, n_rho=16, n_gap=18, n_tilt=5, mag_mesh=(12, 3, 4))
    out = {"table_build_s": table.build_seconds, "table_points": int(np.prod(table.vals[1].shape[:4]) * 2)}
    cases = [random_cube_pair(rng) for _ in range(n_pairs)]
    # direct
    t0 = time.perf_counter()
    direct = [wrench(a, b, pivot=(6e-3, 0, 0), mag_mesh=(12, 3, 4)) for a, b in cases]
    out["direct_ms"] = (time.perf_counter() - t0) / n_pairs * 1e3
    t0 = time.perf_counter()
    for a, b in cases[:5]:
        wrench(a, b, pivot=(6e-3, 0, 0), mag_mesh=(24, 4, 8))
    out["direct_fine_mesh_ms"] = (time.perf_counter() - t0) / 5 * 1e3
    # lookup
    from scipy.spatial.transform import Rotation as R

    from .mjcoupling import hybrid_wrench

    def lookup(a, b):
        com = np.array([6e-3, 0, 0])
        info = {}
        F, T = hybrid_wrench(table, a, b, com, mag_mesh=(12, 3, 4), info=info)
        return F, T, np.array([info.get("table_fraction", 0.0)])

    t0 = time.perf_counter()
    look = [lookup(a, b) for a, b in cases]
    out["lookup_ms"] = (time.perf_counter() - t0) / n_pairs * 1e3
    errs_f, errs_t, oks = [], [], []
    for (Fd, Td), (Fl, Tl, ok) in zip(direct, look):
        oks.append(ok.mean())
        errs_f.append(np.linalg.norm(Fl - Fd) / max(np.linalg.norm(Fd), 1e-9))
        errs_t.append(np.linalg.norm(Tl - Td) / max(np.linalg.norm(Td), 1e-9))
    out["lookup_in_range_fraction"] = float(np.mean(oks))
    out["force_err_median_pct"] = float(np.median(errs_f) * 100)
    out["force_err_p90_pct"] = float(np.percentile(errs_f, 90) * 100)
    out["torque_err_median_pct"] = float(np.median(errs_t) * 100)
    out["torque_err_p90_pct"] = float(np.percentile(errs_t, 90) * 100)
    # single-pair interpolation error on in-range random single-magnet poses
    from .mjcoupling import rot_z_to
    e1 = []
    for _ in range(200):
        rho = rng.uniform(0, 2.5e-3)
        gap = 10 ** rng.uniform(-4.2, -2.7)
        psi = rng.uniform(0, 2 * np.pi)
        tilt = np.deg2rad(rng.uniform(0, 25))
        az = rng.uniform(0, 2 * np.pi)
        sgn = rng.choice([-1, 1])
        nvec = np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), sgn * np.cos(tilt)])
        tp = np.array([rho * np.cos(psi), rho * np.sin(psi), 1e-3 + gap])
        s = Magnet("cyl", (1e-3, 1e-3), 1.45)
        t_ = Magnet("cyl", (1e-3, 1e-3), 1.45, tuple(tp), rot_z_to(nvec))
        F, T = wrench([s], [t_], pivot=t_.pos, mag_mesh=(12, 3, 4))
        Fl, Tl, ok = table.query(np.zeros(3), R.identity(), tp, rot_z_to(nvec))
        e1.append(np.linalg.norm(Fl[0] - F) / np.linalg.norm(F))
    out["single_pair_force_err_median_pct"] = float(np.median(e1) * 100)
    out["single_pair_force_err_p90_pct"] = float(np.percentile(e1, 90) * 100)
    out["single_pair_force_err_max_pct"] = float(np.max(e1) * 100)
    return out


def mesh_convergence():
    d, h, br = 3.175e-3, 0.7938e-3, 1.48
    res = {}
    for mesh in [(8, 2, 2), (12, 3, 4), (24, 4, 8), (48, 6, 16)]:
        res[str(mesh)] = pull_force_to_steel("cyl", (d, h), br, 0.0, mag_mesh=mesh)
    return res


def summarize(rows):
    out = {}
    for g in ("pull", "electrical", "thermal"):
        r = [x for x in rows if x.group == g]
        out[g] = {"passed": sum(x.passed for x in r), "total": len(r)}
    return out


def run_all(path=None):
    t0 = time.time()
    steel, k, ratios = steel_rows()
    rows = steel + fem_rows() + inductance_rows() + thermal_rows()
    ver = magnet_verification_rows() + circuit_rows()
    summ = summarize(rows)
    bench = speed_and_interp()
    out = {
        "rows": [asdict(r) for r in rows],
        "verification": [asdict(r) for r in ver],
        "summary": summ,
        "derate_factor": k,
        "vendor_over_model_ratio": ratios,
        "bench": bench,
        "mesh_convergence_N": mesh_convergence(),
        "seconds": time.time() - t0,
    }
    if path:
        Path(path).write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "hardware/runs/module_phys/simstack_validation.json"
    r = run_all(p)
    print(json.dumps({k: r[k] for k in ("summary", "derate_factor", "bench", "seconds")}, indent=1, default=float))
