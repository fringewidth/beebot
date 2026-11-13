"""Validate hardware/joint_strength.py against MuJoCo (hardware/joint_sim.py).

For each test structure (cantilevers from a wall, a free-standing L, spans between fixed abutments
and on free pillars, towers with a side load, corbelled arches, tables, gusseted brackets) MuJoCo
ramps the load until the structure gives, nominal cubes and jittered seeds; the check is fed the
same cubes (magnet strength, mass, friction per seed). Writes one row per run and prints the
agreement table (handoff runs/handoff/joint-strength.md).

    nice -n 10 .venv/bin/python scripts/joint_validate.py --workers 4 --out runs/joint_strength/validate.jsonl
    .venv/bin/python scripts/joint_validate.py --table runs/joint_strength/validate.jsonl
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def cantilever(n):
    return {"cells": [(1, 0, x) for x in range(n)], "anchored": [(0, 0, -1), (1, 0, -1)]}


def l_free(n, h=3):
    return {"cells": [(z, 0, 0) for z in range(h)] + [(h - 1, 0, x) for x in range(1, n + 1)]}


def span_fixed(w):
    return {"cells": [(1, 0, x) for x in range(w)], "anchored": [(0, 0, -1), (1, 0, -1), (0, 0, w), (1, 0, w)]}


def span_pillars(w):
    return {"cells": [(0, 0, -1), (1, 0, -1), (0, 0, w), (1, 0, w)] + [(1, 0, x) for x in range(w)]}


def tower_side(h):
    load = np.zeros((h, 3))
    load[h - 1, 0] = 1.0
    return {"cells": [(z, 0, 0) for z in range(h)], "load": load, "mode": "live"}


def arch(k):
    d = 2 * k + 2
    cells = []
    for z in range(k):
        cells += [(z, 0, z), (z, 0, z + 1), (z, 0, d - z), (z, 0, d - z - 1)]
    cells += [(k, 0, x) for x in range(k, d - k + 1)]
    return {"cells": sorted(set(cells))}


def table(w, legs=2):
    cells = [(z, 0 if a == 0 else w - 1, 0 if b == 0 else w - 1) for z in range(legs) for a in (0, 1) for b in (0, 1)]
    cells += [(legs, y, x) for y in range(w) for x in range(w)]
    return {"cells": cells}


def bracket(k, n=8, h=4):
    """An arm of n at height h from a wall, with a stepped gusset of depth k under it."""
    cells = [(h, 0, x) for x in range(n)]
    for d in range(1, k + 1):
        cells += [(h - d, 0, x) for x in range(k - d + 1)]
    return {"cells": cells, "anchored": [(z, 0, -1) for z in range(h + 1)]}


FAMILIES = {
    "cantilever": (cantilever, range(1, 9), {"stop": 3.0}),
    "L": (l_free, range(1, 7), {"stop": 3.0}),
    "span_fixed": (span_fixed, range(2, 11), {"stop": 3.0}),
    "span_pillars": (span_pillars, range(2, 11), {"stop": 3.0}),
    "tower_side": (tower_side, range(2, 16), {"stop": 5.0}),
    "arch": (arch, range(2, 5), {"stop": 3.0}),
    "table": (table, (3, 5, 7, 9), {"stop": 3.0}),
    "bracket": (bracket, range(0, 4), {"stop": 3.0}),
    "span_fixed_long": (span_fixed, (15, 20, 25, 30, 40), {"stop": 6.0, "rate": 0.5}),
    "span_pillars_long": (span_pillars, (12, 14, 16, 18, 20, 24), {"stop": 6.0, "rate": 0.5}),
}


def run_one(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    from beebot.hardware.joint_sim import ramp

    family, param, seed = job
    make, _, options = FAMILIES[family]
    options = dict(options)
    spec = make(param)
    began = time.time()
    out = ramp(spec["cells"], spec.get("anchored", ()), spec.get("load"), spec.get("mode", "gravity"), seed=seed, **options)
    out.update(family=family, param=param, seed=seed, stop=options["stop"], wall=round(time.time() - began, 1))
    return out


def jaccard(a, b):
    a, b = {tuple(c) for c in a}, {tuple(c) for c in b}
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def add_check(rows):
    from beebot.hardware.joint_strength import check_many

    specs = []
    for r in rows:
        spec = dict(FAMILIES[r["family"]][0](r["param"]))
        n = len(spec["cells"])
        if r["seed"] is not None:
            spec.update(strength=r["strength"], weights=r["masses"][:n], mu=r["friction"])
        specs.append(spec)
    results = check_many(specs)
    for r, res in zip(rows, results):
        r["check"] = round(res.factor, 4)
        r["check_moving"] = res.collapse.tolist() if res.collapse is not None else []
        r["check_opening"] = [list(map(str, o)) for o in res.weak_joints]
        r["converged"] = res.converged


def table_rows(rows):
    """Agreement: stands at the design load (factor 1: 1 g, or 1 module weight of side load on a
    tower) per run, and the error in the failure factor where both fail below the ramp's stop."""
    lines = ["| family | param | check (nominal) | MuJoCo nominal | MuJoCo seeds 1-3 | check seeds 1-3 | collapse set overlap |", "| --- | --- | --- | --- | --- | --- | --- |"]
    by = {}
    for r in rows:
        by.setdefault((r["family"], r["param"]), []).append(r)
    fmt = lambda v, stop: f">{stop:g}" if v is None or v >= stop else f"{v:.2f}"
    for family in FAMILIES:
        for (fam, param), group in sorted(by.items(), key=lambda kv: kv[0][1]):
            if fam != family:
                continue
            nominal = [r for r in group if r["seed"] is None]
            seeds = sorted([r for r in group if r["seed"] is not None], key=lambda r: r["seed"])
            nom = nominal[0] if nominal else None
            overlap = [jaccard(r["moved"], r["check_moving"]) for r in group if r["factor"] is not None and r["check"] < r["stop"]]
            cells = [
                family,
                str(param),
                fmt(nom["check"], nom["stop"]) if nom else "-",
                fmt(nom["factor"], nom["stop"]) if nom else "-",
                " / ".join(fmt(r["factor"], r["stop"]) for r in seeds),
                " / ".join(fmt(r["check"], r["stop"]) for r in seeds),
                f"{np.mean(overlap):.2f}" if overlap else "-",
            ]
            lines.append("| " + " | ".join(cells) + " |")
    return lines


def summary(rows):
    tp = fp = tn = fn = 0
    errors = []
    for r in rows:
        sim_holds = r["factor"] is None or r["factor"] >= 1.0
        check_holds = r["check"] >= 1.0
        tp += (not check_holds) and (not sim_holds)
        fp += (not check_holds) and sim_holds
        tn += check_holds and sim_holds
        fn += check_holds and (not sim_holds)
        if r["factor"] is not None and r["check"] < r["stop"]:
            errors.append((r["check"] - r["factor"]) / r["factor"])
    errors = np.array(errors)
    return {
        "runs": len(rows),
        "fails_predicted_and_failed": tp,
        "fails_predicted_but_held": fp,
        "holds_predicted_and_held": tn,
        "holds_predicted_but_failed": fn,
        "factor_rel_error_mean": round(float(errors.mean()), 4) if len(errors) else None,
        "factor_rel_error_abs_median": round(float(np.median(np.abs(errors))), 4) if len(errors) else None,
        "factor_rel_error_abs_max": round(float(np.abs(errors).max()), 4) if len(errors) else None,
        "compared": len(errors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--out", default="runs/joint_strength/validate.jsonl")
    parser.add_argument("--table", default=None, help="only print the table of existing files (comma-separated); the check is run again")
    args = parser.parse_args()
    if args.table:
        rows = [json.loads(line) for path in args.table.split(",") for line in open(path)]
        add_check(rows)
    else:
        jobs = []
        for family in args.families.split(","):
            _, params, _ = FAMILIES[family]
            for param in params:
                jobs += [(family, param, None)] + [(family, param, s) for s in range(1, args.seeds + 1)]
        jobs.sort(key=lambda j: -len(FAMILIES[j[0]][0](j[1])["cells"]))  # big ones first
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        rows = []
        with ProcessPoolExecutor(args.workers) as pool, open(args.out, "w") as f:
            for r in pool.map(run_one, jobs):
                rows.append(r)
                print(json.dumps({k: r[k] for k in ("family", "param", "seed", "factor", "why", "first_joint", "wall")}), flush=True)
        add_check(rows)
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print("\n".join(table_rows(rows)))
    print(json.dumps(summary(rows), indent=1))
    for family in FAMILIES:
        sub = [r for r in rows if r["family"] == family]
        if sub:
            print(family, json.dumps(summary(sub)))


if __name__ == "__main__":
    main()
