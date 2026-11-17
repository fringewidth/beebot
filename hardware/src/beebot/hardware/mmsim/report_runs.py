"""Produces the numbers quoted in hardware/runs/module_phys/simstack.md. Run: .venv/bin/python -m beebot.hardware.mmsim.report_runs"""
import json
import sys
import time

import numpy as np

from . import circuit, coil, demo, opt, toys
from .mjcoupling import CylPairTable, SurfaceForces


def coil_examples():
    out = {}
    for name, sp in {
        "4mm 2L 18um 75/75": coil.Spiral(5, 4e-3, 75e-6, 75e-6, 18e-6, layers=2),
        "4mm 4L 12um 75/75": coil.Spiral(5, 4e-3, 75e-6, 75e-6, 12e-6, layers=4),
        "4mm 2L 35um 100/100": coil.Spiral(4, 4e-3, 100e-6, 100e-6, 35e-6, layers=2),
        "8mm 2L 18um 75/75": coil.Spiral(10, 8e-3, 75e-6, 75e-6, 18e-6, layers=2),
    }.items():
        at = coil.ampere_turns(sp, 3.0, r_series=0.3)
        tab = coil.safe_current_table(sp, [1e-3, 1e-2, 1e-1, 1.0], 100.0)
        out[name] = {
            "turns_total": sp.turns_total, "length_m": coil.track_length(sp), "R_ohm": coil.resistance(sp),
            "L_flux_nH": coil.inductance_flux(sp) * 1e9, "L_mohan_single_layer_nH": coil.inductance_mohan(sp) * 1e9,
            "at_3V_0p3ohm": at, "safe_current_A_dT75": tab,
            "safe_NI_dT75": [(t, i * sp.turns_total) for t, i in tab],
        }
    return out


def circuit_examples():
    out = {}
    sp = coil.Spiral(5, 4e-3, 75e-6, 75e-6, 18e-6, layers=2)
    r, l = coil.resistance(sp), coil.inductance_flux(sp)
    drv = circuit.Driver(0.1)
    cases = {
        "MLCC 4x10uF X5R 3.3V derate0.5": circuit.mlcc_bank(4, 10e-6, 5e-3, 3.3, 6.3, derate=0.5),
        "supercap 10mF 5ohm ESR 3.3V": circuit.supercap(10e-3, 5.0, 3.3),
        "CeraCharge 100uAh 200ohm": circuit.ceracharge(),
    }
    for name, st in cases.items():
        res = circuit.simulate_pulse(st, r, l, 200e-6, drv, t_total=600e-6)
        out[name] = {k: v for k, v in res.items() if np.ndim(v) == 0}
    out["coil"] = {"R": r, "L": l}
    return out


def seating():
    out = {}
    surf = SurfaceForces(adhesion=2e-4, mu_s=0.5, mu_k=0.3)
    table = CylPairTable(demo.MAG_D, demo.MAG_H, demo.MAG_BR, n_rho=16, n_gap=18, n_tilt=3, mag_mesh=None)
    out["table_build_s"] = table.build_seconds
    for mode in ("direct", "lookup"):
        for dt in (1e-4, 5e-5, 2e-5, 1e-5):
            for sf in (None, surf):
                r = demo.run_seating(dt=dt, t_end=0.03, mode=mode, table=table if mode == "lookup" else None,
                                     surface=sf, record_every=int(1e-3 / dt))
                key = f"{mode}_dt{dt:g}_{'surface' if sf else 'plain'}"
                out[key] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in r.items() if k != "log"}
                out[key]["gap_mm"] = float(r["gap_mm"])
    mom = {}
    for dt in (4e-5, 2e-5, 1e-5):
        mom[str(dt)] = demo.run_momentum_test(n_steps=int(0.003 / dt), dt=dt)
    out["momentum"] = mom
    return out


def optimisation():
    out = {}
    for algo, nobj, fn in (("nsga2", 2, toys.zdt1), ("ga", 1, toys.sphere), ("cma", 1, toys.sphere)):
        t = time.time()
        r = opt.run_search(fn, [0] * 4, [1] * 4, n_obj=nobj, algo=algo, pop=24, n_gen=20, workers=6)
        out[algo] = {"n_evals": r.n_evals, "best_F": r.F.min(0).tolist(), "seconds": time.time() - t,
                     "pareto_size": int(len(r.F))}
    p = {"magnet_pos_x": 0.0, "magnet_pos_y": 0.0, "magnetisation": 1.0, "print_dim_gap": 0.3e-3, "threshold": 0.165,
         "friction_mu": 0.5, "load": 0.075}
    t = time.time()
    out["mc_latch"] = opt.monte_carlo(toys.toy_latch, p, n=300, workers=6)
    out["mc_latch"]["seconds"] = time.time() - t
    # real-objective NSGA-II: (-pull force, volume) over gap and diameter
    t = time.time()
    r = opt.run_search(toys.toy_latch_objectives, [0.05, 0.5], [1.0, 3.0], n_obj=2, algo="nsga2", pop=16, n_gen=8, workers=6)
    out["nsga2_magnet"] = {"n_evals": r.n_evals, "pareto_size": int(len(r.F)), "seconds": time.time() - t}
    return out


if __name__ == "__main__":
    res = {"coil": coil_examples(), "circuit": circuit_examples(), "optimisation": optimisation(), "seating": seating()}
    with open(sys.argv[1] if len(sys.argv) > 1 else "hardware/runs/module_phys/simstack_runs.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("done")
