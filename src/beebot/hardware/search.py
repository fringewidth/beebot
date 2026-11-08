"""Search over module designs by work score (evaluate.py), resumable, in parallel processes.

A grid over size, connector type and the volume shares of connector, actuator and flight
system. Designs that do not fit in the cube are skipped, and so are designs that carry a flight
system too weak to lift them (they are the same crawler, only heavier). Every design is scored on
the same task sample, so differences between designs are paired.

    uv run python -m beebot.hardware.search --out runs/hardware/grid1.jsonl --tasks 96 --workers 4
"""

import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

from beebot.hardware.design import Capability, Design, describe

SIDES = (0.01, 0.03, 0.10)
CONNECTORS = ("magnet", "epm")
CONNECTOR_SHARES = (0.0, 0.01, 0.03, 0.06, 0.12, 0.25)
ACTUATOR_SHARES = (0.01, 0.03, 0.06, 0.12, 0.25, 0.4)
FLIGHT_SHARES = (0.0, 0.2, 0.35, 0.5)
REFERENCES = {
    "ideal-crawler": Capability(strength=1e6, torque=1e6, thrust=0.0, cheese=4.0, switchable=True),
    "ideal-flyer": Capability(strength=1e6, torque=1e6, thrust=1e6, cheese=4.0, switchable=True),
}


def grid():
    out = []
    for side, connector, c, a, f in itertools.product(SIDES, CONNECTORS, CONNECTOR_SHARES, ACTUATOR_SHARES, FLIGHT_SHARES):
        if c == 0 and connector != "magnet":
            continue  # no connector: the type does not matter
        design = Design(side, connector, c, a, f)
        if not design.feasible():
            continue
        if f > 0 and not design.capability().flies:
            continue
        out.append(design)
    return out


def score_one(job):
    name, cap, row, tasks, seed = job
    from beebot.hardware.evaluate import evaluate

    started = time.time()
    work, families = evaluate(cap, tasks, seed)
    row = dict(row)
    row.update({"name": name, "work": round(work, 4), "families": families, "tasks": tasks, "seed": seed, "seconds": round(time.time() - started)})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="runs/hardware/grid1.jsonl")
    parser.add_argument("--tasks", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", default="", help="comma-separated design labels to score instead of the grid")
    parser.add_argument("--voxel", action="store_true", help="also score the voxel world's own rules")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["name"] for line in out.read_text().splitlines() if line.strip()}
    jobs = []
    if args.voxel and "voxel-world" not in done:
        jobs.append(("voxel-world", None, {"design": "voxel-world"}, args.tasks, args.seed))
    for name, cap in REFERENCES.items():
        if name not in done:
            jobs.append((name, cap, {"design": name, **{k: v for k, v in cap.__dict__.items()}}, args.tasks, args.seed))
    designs = grid()
    if args.only:
        wanted = set(args.only.split(","))
        designs = [d for d in designs if d.label() in wanted]
    for design in designs:
        if design.label() not in done:
            jobs.append((design.label(), design.capability(), {**describe(design), "side": design.side, "connector": design.connector, "connector_share": design.connector_share, "actuator_share": design.actuator_share, "flight_share": design.flight_share}, args.tasks, args.seed))
    print(f"{len(jobs)} to score ({len(done)} done)", flush=True)
    with Pool(args.workers) as pool, out.open("a") as fh:
        for row in pool.imap_unordered(score_one, jobs):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(row["name"], row["work"], row["seconds"], flush=True)


if __name__ == "__main__":
    main()
