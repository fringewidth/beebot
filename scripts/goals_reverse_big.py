"""Does the take-apart scale? Reverse 3 goals of one cell-count bin from a library, one at a time, with a step
cap and a wall-clock timeout per goal.

    nice -n 10 .venv/bin/python scripts/goals_reverse_big.py --lo 251 --hi 500 [--n 3] [--max-steps 2000] [--timeout 900]

Prints one JSON line per goal: id, cells, ok (replay passed and start natural), steps, cpu seconds, wall seconds
(or timeout). CPU only, one torch thread.
"""

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel import goal_library as G
from beebot.vla.voxel import reverse


class Timeout(Exception):
    pass


def alarm(signum, frame):
    raise Timeout()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/goal_library_v2")
    parser.add_argument("--source", default="stabletext2brick")
    parser.add_argument("--lo", type=int, required=True)
    parser.add_argument("--hi", type=int, required=True)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    torch.set_num_threads(1)
    lines, cells = G.load(args.source, root=Path(args.root))
    rng = np.random.default_rng(args.lo)
    pool = [i for i, line in enumerate(lines) if args.lo <= line["cells"] <= args.hi]
    signal.signal(signal.SIGALRM, alarm)
    for i in rng.choice(pool, size=args.n, replace=False).tolist():
        goal = G.to_goal(cells[i], lines[i]["text"])
        cpu, wall = time.process_time(), time.time()
        row = {"id": lines[i]["id"], "cells": lines[i]["cells"], "bin": f"{args.lo}-{args.hi}"}
        signal.alarm(args.timeout)
        try:
            episodes, _ = reverse.take_apart([goal], rng, max_steps=args.max_steps)
            signal.alarm(0)
            replay = reverse.replay(episodes)["passed"]
            ep = episodes[0]
            row.update(ok=bool(replay[0]) and bool(ep.natural), replay=bool(replay[0]), natural=bool(ep.natural), end=ep.end, steps=int(ep.steps))
        except Timeout:
            row.update(ok=False, timeout=True)
        except Exception as error:
            signal.alarm(0)
            row.update(ok=False, error=f"{type(error).__name__}: {error}"[:100])
        row.update(cpu_s=round(time.process_time() - cpu, 1), wall_s=round(time.time() - wall, 1))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
