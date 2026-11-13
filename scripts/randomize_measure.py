"""Drop rate and cost of domain randomization (src/beebot/vla/voxel/randomize.py) per item and strength:
nominal live episodes (reverse_live.generate) are executed under each item alone, and each strength, over
several physics draws; a draw whose episode does not reach its goal (step cap, 12 plans, no plan) is dropped.
Also replays every kept episode (randomize.replay_drawn).

    OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python scripts/randomize_measure.py --episodes 40 --procs 4 --out runs/randomize/measure.json
"""

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel import randomize as DR
from beebot.vla.voxel import reverse_live as L

ITEMS = ("failure", "weak", "broken", "drop", "late")
STRENGTHS = (0.3, 1.0)


def _init():
    torch.set_num_threads(1)
    os.nice(10)


def _nominal(seed):
    families = list(L.FAMILIES)
    row, _ = L.generate(seed, [families[seed % len(families)]], robots=(6, 40))
    return row


def _run(job):
    row, config, seed = job
    rand = DR.Randomization(**{**dict(strength=0.0, placement=0.0), **config})
    stats = {}
    t0 = time.process_time()
    out = DR.randomize_row(row, rand, seed, stats=stats)
    cpu = time.process_time() - t0
    result = {"dropped": out is None, "why": stats.get("dropped"), "cpu": cpu, "family": row["meta"]["family"], "broken": False, "physical": False, "stats": stats}
    if out is not None:
        meta = out["meta"]
        result.update(replans=meta["replans"], repeats=meta["repeats"], failed=meta["failed_steps"], steps=meta["steps"], nominal_steps=row["meta"]["steps"], broken=bool(meta["physics"]["broken"]), physical="stuck" in out)
        if "stuck" in out:
            result["replay"] = bool(DR.replay_drawn(out)["passed"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--draws", type=int, default=3, help="physics draws per episode and config")
    parser.add_argument("--procs", type=int, default=4)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default="", help="comma separated configs (failure@0.3, all@1.0, failure_harmless@1.0 ...); default all")
    args = parser.parse_args()
    with multiprocessing.get_context("spawn").Pool(args.procs, initializer=_init) as pool:
        rows = [r for r in pool.map(_nominal, range(args.episodes)) if r is not None]
        configs = [(item, s) for item in ITEMS for s in STRENGTHS] + [("all", s) for s in STRENGTHS] + [("failure_harmless", 1.0), ("failure_harmful", 1.0)]
        report = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
        for item, s in configs:
            if args.only and f"{item}@{s}" not in args.only.split(","):
                continue
            config = {"strength": s} if item == "all" else {"failure": s, "harm": 0.0} if item == "failure_harmless" else {"failure": s, "harm": 1.0} if item == "failure_harmful" else {item: s}
            jobs = [(row, config, 1000 * i + d) for i, row in enumerate(rows) for d in range(args.draws)]
            results = pool.map(_run, jobs, chunksize=1)
            kept = [r for r in results if not r["dropped"]]
            affected = [r for r in kept if r["physical"]]
            cpus = [c for r in results for c in r["stats"].get("replan_cpu", [])]
            why = {}
            for r in results:
                if r["dropped"]:
                    why[r["why"]] = why.get(r["why"], 0) + 1
            report[f"{item}@{s}"] = {
                "draws": len(results),
                "drop_rate": round(1 - len(kept) / len(results), 3),
                "drop_reasons": why,
                "episodes_with_physics": len(affected),
                "drop_rate_of_those": round(len(why) and sum(why.values()) / max(sum(why.values()) + len(affected), 1), 3),
                "broken_kept": sum(r["broken"] for r in kept),
                "repeats_mean": round(float(np.mean([r["repeats"] for r in affected])), 2) if affected else 0.0,
                "replans_total": sum(r["stats"].get("replanned", 0) for r in results),
                "rejoined": sum(r["stats"].get("rejoined", 0) for r in results),
                "to_goal": sum(r["stats"].get("to goal", 0) for r in results),
                "replan_failures": sum(r["stats"].get("replanned", 0) - r["stats"].get("rejoined", 0) - r["stats"].get("to goal", 0) for r in results),
                "replan_cpu_median": round(float(np.median(cpus)), 3) if cpus else None,
                "replan_cpu_p95": round(float(np.percentile(cpus, 95)), 3) if cpus else None,
                "replans_mean": round(float(np.mean([r["replans"] for r in affected])), 2) if affected else 0.0,
                "steps_ratio": round(float(np.mean([r["steps"] / max(r["nominal_steps"], 1) for r in affected])), 2) if affected else 1.0,
                "cpu_mean": round(float(np.mean([r["cpu"] for r in results])), 2),
                "replay_pass": float(np.mean([r["replay"] for r in affected])) if affected else None,
            }
            print(item, s, json.dumps(report[f"{item}@{s}"]), flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, indent=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
