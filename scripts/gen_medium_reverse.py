"""Reversal check for gen_medium goals: take_apart + replay on a few goals per generator.

    nice -n 10 .venv/bin/python scripts/gen_medium_reverse.py --per 3 --out logs/gen-medium-reverse.jsonl
"""

import argparse
import json

import numpy as np
import torch

from beebot.vla.voxel import goal_gen_medium as M
from beebot.vla.voxel import goal_library as G
from beebot.vla.voxel import reverse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per", type=int, default=3)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--name", default="gen_medium")
    parser.add_argument("--out", default="logs/gen-medium-reverse.jsonl")
    args = parser.parse_args()
    torch.set_num_threads(3)
    rng = np.random.default_rng(0)
    lines, cells = G.load(args.name)
    picked = []
    for generator in M.GENERATORS:
        pool = [i for i, line in enumerate(lines) if line["generator"] == generator]
        picked += rng.choice(pool, size=min(args.per, len(pool)), replace=False).tolist()
    rows = []
    for lo in range(0, len(picked), args.batch):
        chunk = picked[lo : lo + args.batch]
        goals = [G.to_goal(cells[i], lines[i]["text"]) for i in chunk]
        episodes, _ = reverse.take_apart(goals, rng, max_steps=args.max_steps)
        replay = reverse.replay(episodes)["passed"]
        for j, i in enumerate(chunk):
            ep = episodes[j]
            rows.append(
                {
                    "id": lines[i]["id"],
                    "generator": lines[i]["generator"],
                    "cells": lines[i]["cells"],
                    "end": ep.end,
                    "natural": bool(ep.natural),
                    "steps": int(ep.steps),
                    "replay": bool(replay[j]),
                    "ok": bool(replay[j] and ep.natural),
                }
            )
        print(json.dumps({"done": len(rows), "of": len(picked)}), flush=True)
    with open(args.out, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    by = {}
    for row in rows:
        by.setdefault(row["generator"], []).append(row["ok"])
    print(json.dumps({g: f"{sum(v)}/{len(v)}" for g, v in by.items()}))
    print(json.dumps({"ok_share": round(sum(r["ok"] for r in rows) / len(rows), 3)}))


if __name__ == "__main__":
    main()
