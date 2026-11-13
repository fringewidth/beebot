"""Run the reversal (beebot.vla.voxel.reverse.take_apart) on a sample of library goals and record, per goal,
whether it can be taken apart today. The library is never filtered on this.

    nice -n 10 .venv/bin/python scripts/goals_reverse.py --per-source 120 --batch 12

Results: data/goal_library/reverse_<source>.jsonl (id, cells, fixed_point, end, natural, steps, replay) and a
summary line per source in logs/goal-library.out. CPU only, torch threads capped at 3.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel import goal_library as G
from beebot.vla.voxel import reverse

LOG = Path("logs/goal-library.out")
SOURCES = ("stabletext2brick", "hack337", "craftassist_houses", "craftassist_parts", "iglu_multiturn", "iglu_single")


def sample_ids(lines, n, rng):
    """n indices spread over the cell-count range (stratified in the library's 4-115 histogram bins)."""
    bins = np.array([4, 8, 16, 32, 64, 116])
    cells = np.array([line["cells"] for line in lines])
    which = np.digitize(cells, bins) - 1
    picked = []
    per = max(n // (len(bins) - 1), 1)
    for b in range(len(bins) - 1):
        pool = np.nonzero(which == b)[0]
        if len(pool):
            picked += rng.choice(pool, size=min(per, len(pool)), replace=False).tolist()
    return picked


def run_source(name, per, batch, rng, max_steps):
    lines, cells = G.load(name)
    ids = sample_ids(lines, per, rng)
    rows = []
    start = time.time()
    for lo in range(0, len(ids), batch):
        chunk = ids[lo : lo + batch]
        goals = [G.to_goal(cells[i], lines[i]["text"]) for i in chunk]
        fixed = reverse.fixed_point(reverse.W.concat([g.world for g in goals]))
        try:
            episodes, _ = reverse.take_apart(goals, rng, max_steps=max_steps)
            replay = reverse.replay(episodes)["passed"]
        except Exception as error:
            for i, ok in zip(chunk, fixed):
                rows.append({"id": lines[i]["id"], "cells": lines[i]["cells"], "fixed_point": bool(ok), "error": f"{type(error).__name__}: {error}"[:100]})
            continue
        for j, i in enumerate(chunk):
            ep = episodes[j]
            rows.append(
                {
                    "id": lines[i]["id"],
                    "cells": lines[i]["cells"],
                    "fixed_point": bool(fixed[j]),
                    "end": ep.end,
                    "natural": bool(ep.natural),
                    "steps": int(ep.steps),
                    "replay": bool(replay[j]),
                }
            )
    return rows, time.time() - start


def summarise(name, rows, seconds):
    done = [r for r in rows if "end" in r]
    ok = [r for r in done if r["replay"] and r["natural"]]
    by_size = {}
    for lo, hi in ((4, 15), (16, 31), (32, 63), (64, 115)):
        group = [r for r in done if lo <= r["cells"] <= hi]
        by_size[f"{lo}-{hi}"] = {"n": len(group), "ok": sum(1 for r in group if r["replay"] and r["natural"])}
    return {
        "source": name,
        "reversal_sample": len(rows),
        "reversal_errors": len(rows) - len(done),
        "fixed_point": sum(1 for r in rows if r["fixed_point"]),
        "replay_pass": sum(1 for r in done if r["replay"]),
        "natural": sum(1 for r in done if r["natural"]),
        "reversal_ok": len(ok),
        "reversal_ok_share": round(len(ok) / max(len(rows), 1), 3),
        "ends": {e: sum(1 for r in done if r["end"] == e) for e in {r["end"] for r in done}},
        "by_cells": by_size,
        "seconds": round(seconds),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-source", type=int, default=120)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--sources", nargs="*", default=SOURCES)
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(3)
    rng = np.random.default_rng(0)
    for name in args.sources:
        rows, seconds = run_source(name, args.per_source, args.batch, rng, args.max_steps)
        with open(G.LIBRARY / f"reverse_{name}.jsonl", "w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        summary = summarise(name, rows, seconds)
        print(json.dumps(summary))
        if not args.no_log:
            with open(LOG, "a") as handle:
                handle.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
