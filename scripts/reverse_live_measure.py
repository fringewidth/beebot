"""Throughput of live reverse episodes (src/beebot/vla/voxel/reverse_live.py): per family, `per` episodes
at the stores' robot-count draw (log-uniform 6-115), each on one thread in a pool of --procs processes
(nice). Per episode: robots, steps, CPU seconds of the walk and its checks (draws that failed
included), draws tried, why draws failed, and whether it replays under world.step (--replay; not
counted in the CPU time). Prints and writes per family and robot bin: episodes per CPU second (one
worker), and whether 4-6 workers keep up with a trainer that uses `--reuse` examples per episode.

    OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python scripts/reverse_live_measure.py --per 24 --procs 4 --replay --out runs/reverse_live/measure.json
"""

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

BINS = ((6, 20), (21, 50), (51, 115))


def job(args):
    import torch

    from beebot.vla.voxel import reverse_live as L

    torch.set_num_threads(1)
    family, seed, replay = args
    t0 = time.time()
    row, info = L.generate(seed, [family], replay=replay)
    out = {"family": family, "seed": seed, "ok": row is not None, "cpu": info["cpu"], "tries": info["tries"], "fails": info["fails"], "wall": time.time() - t0}
    if row is not None:
        out.update(robots=row["meta"]["robots"], steps=row["meta"]["steps"], moving=row["meta"]["moving"], bytes=int(row["mass"].nbytes + row["codes"].nbytes))
    return out


def init():
    os.nice(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per", type=int, default=24)
    parser.add_argument("--procs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--families", default="")
    parser.add_argument("--out", default="runs/reverse_live/measure.json")
    args = parser.parse_args()
    from beebot.vla.voxel import reverse_live as L

    families = args.families.split(",") if args.families else list(L.FAMILIES)
    jobs = [(f, args.seed * 100000 + 1000 * k + j, args.replay) for k, f in enumerate(families) for j in range(args.per)]
    rows = []
    with multiprocessing.get_context("spawn").Pool(args.procs, initializer=init) as pool:
        for r in pool.imap_unordered(job, jobs):
            rows.append(r)
            print(json.dumps(r), flush=True)
    report = {}
    for f in families:
        mine = [r for r in rows if r["family"] == f]
        done = [r for r in mine if r["ok"]]
        cpu = sum(r["cpu"] for r in mine)
        entry = {"episodes": len(done), "draws_failed_whole": len(mine) - len(done), "cpu_per_episode": cpu / max(len(done), 1), "episodes_per_cpu_s": len(done) / max(cpu, 1e-9), "tries_mean": float(np.mean([r["tries"] for r in mine])), "robots_mean": float(np.mean([r["robots"] for r in done])) if done else 0, "steps_mean": float(np.mean([r["steps"] for r in done])) if done else 0, "replay_passed": sum(r["ok"] for r in done), "fails": {}}
        for r in mine:
            for k, v in r["fails"].items():
                entry["fails"][k] = entry["fails"].get(k, 0) + v
        for lo, hi in BINS:
            b = [r for r in done if lo <= r["robots"] <= hi]
            if b:
                entry[f"{lo}-{hi}"] = {"n": len(b), "cpu_per_episode": float(np.mean([r["cpu"] for r in b])), "steps": float(np.mean([r["steps"] for r in b]))}
        report[f] = entry
    total_cpu = sum(r["cpu"] for r in rows)
    total_eps = sum(r["ok"] for r in rows)
    report["all"] = {"episodes": total_eps, "cpu_per_episode": total_cpu / max(total_eps, 1), "episodes_per_cpu_s": total_eps / max(total_cpu, 1e-9), "bytes_per_episode": float(np.mean([r["bytes"] for r in rows if r["ok"]]))}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rows": rows, "report": report}, indent=1))
    for f, e in report.items():
        print(f, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()}))


if __name__ == "__main__":
    main()
