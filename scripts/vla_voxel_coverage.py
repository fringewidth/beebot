"""Expert success on the voxel suite's coverage families (suite.COVERAGE: motion, subset, switch),
per task kind, on fresh seeds, in three worlds:

  default  all 27 moves, 48 steps, the world's density range (51 to 115 robots)
  cube     demos --cube: the 14 cube moves, 72 steps, 6 to 115 robots (log-uniform), failures 0.15
  switch   demos --cube --rule switch_cube: cube, gravity on, pivots with released bonds

    PYTHONPATH=src OMP_NUM_THREADS=4 nice python scripts/vla_voxel_coverage.py --scenes 120 --out runs/coverage/expert.jsonl

A row per episode: arm, family, kind (suite.task_kind), robots, score, unasked, passed (score at
the pass mark and nothing forbidden eaten, as demos keeps a demonstration).
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel import suite as S
from beebot.vla.voxel.world import CUBE_MOVES, failure_rates

ARMS = {
    "default": {"moves": None, "steps": S.EPISODE_STEPS, "robots": None, "failure": False, "rule": None},
    "cube": {"moves": CUBE_MOVES, "steps": S.CUBE_STEPS, "robots": S.ROBOTS, "failure": True, "rule": None},
    "switch": {"moves": CUBE_MOVES, "steps": S.CUBE_STEPS, "robots": S.ROBOTS, "failure": True, "rule": "switch_cube"},
}


def bucket(n):
    return "under 15" if n < 15 else "15 to 40" if n <= 40 else "over 40"


def play(arm, family, n_scenes, seed, chunk=24):
    """Rows of n_scenes episodes of one coverage family in one arm."""
    cfg = ARMS[arm]
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(0, n_scenes, chunk):
        world, metas, tasks = S.sample_coverage(rng, min(chunk, n_scenes - c), family=family, robots=cfg["robots"], counts="log")
        if cfg["rule"] is not None:
            world = replace(world, gravity=torch.ones_like(world.gravity))
        run_seed = seed * 1_000_003 + c
        expert = S.Expert(tasks, moves=cfg["moves"], seed=run_seed, rule=cfg["rule"])
        results, _ = S.run(world, metas, tasks, expert, steps=cfg["steps"], seed=run_seed, failure=failure_rates() if cfg["failure"] else None, rule=cfg["rule"])
        for i, (task, (score, unasked)) in enumerate(zip(tasks, results)):
            passed = not unasked and score >= S.SUCCESS.get(task.family, S.PASS_MARK)
            rows.append({"arm": arm, "family": task.family, "kind": S.task_kind(task), "robots": int(world.mass[i].sum()), "score": round(float(score), 4), "unasked": bool(unasked), "passed": bool(passed)})
    return rows


def short_kind(row):
    """The job of a row, and for subset tasks the selector kind: "goto@group", "switch:...>stay"."""
    kind = row["kind"]
    if kind.startswith("switch:"):
        return "switch, then " + kind.split(">")[-1].split("@")[0]
    if "@" in kind:
        job, who = kind.split("@")
        who = "count" if who in ("count", "half") else "top" if who == "top" else "except" if who.startswith("not_") else "group"
        return f"{job}@{who}"
    return kind


def table(rows, key):
    """Markdown table: per key value and arm, mean score / pass share (n)."""
    arms = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    names = sorted({key(r) for r in rows})
    lines = ["| | " + " | ".join(arms) + " |", "| --- |" + " --- |" * len(arms)]
    for name in names:
        cells = []
        for arm in arms:
            sel = [r for r in rows if r["arm"] == arm and key(r) == name]
            cells.append(f"{np.mean([r['score'] for r in sel]):.2f} / {np.mean([r['passed'] for r in sel]):.2f} ({len(sel)})" if sel else "")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=120, help="per family and arm")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--families", nargs="+", default=list(S.COVERAGE))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    rows = []
    t0 = time.time()
    for arm in args.arms:
        for k, family in enumerate(args.families):
            rows += play(arm, family, args.scenes, args.seed + 101 * k)
            print(f"{arm} {family}: {time.time() - t0:.0f} s", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    print(table(rows, lambda r: r["family"]))
    print()
    print(table(rows, short_kind))
    print()
    print(table(rows, lambda r: bucket(r["robots"])))


if __name__ == "__main__":
    main()
