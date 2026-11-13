"""Render 6 goals per cell-count bin from data/goal_library_v3 (all sources together, grey render) for the browse page.

    nice -n 10 .venv/bin/python scripts/goals_render_v3.py --out <browse dir>

Writes <out>/v3-<bin>-<k>.png and <out>/cards-goals-v3.json (name, bin, source, cells, text); the browse page
prints source, cells and the text under each picture.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from beebot.vla.voxel import goal_library as G

ROOT = Path("data/goal_library_v3")
BINS = [(4, 15), (16, 31), (32, 63), (64, 127), (128, 191), (192, 256)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-bin", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    pools = {b: [] for b in BINS}
    sources = {}
    for path in sorted(ROOT.glob("*.npz")):
        name = path.stem
        if name.endswith("_start") or not (ROOT / f"{name}.jsonl").exists():
            continue
        lines, cells = G.load(name, ROOT)
        sources[name] = (lines, cells)
        for i, line in enumerate(lines):
            for b in BINS:
                if b[0] <= line["cells"] <= b[1] and line.get("text"):
                    pools[b].append((name, i))
    cards = []
    for b in BINS:
        # round-robin over sources so every source shows up in a bin where it has goals
        by_source = {}
        for name, i in pools[b]:
            by_source.setdefault(name, []).append(i)
        names = sorted(by_source)
        order = rng.permutation(len(names))
        for k in range(args.per_bin):
            if not names:
                break
            name = names[int(order[k % len(names)])]
            i = by_source[name][int(rng.integers(len(by_source[name])))]
            lines, cells = sources[name]
            label = f"{b[0]}-{b[1]}"
            file = f"v3-{label}-{k}.png"
            G.render_views(cells[i]).save(out / file)
            cards.append({"name": file, "bin": label, "source": name, "cells": lines[i]["cells"], "text": lines[i]["text"], "id": lines[i]["id"]})
    json.dump(cards, open(out / "cards-goals-v3.json", "w"), indent=1)
    print(len(cards), "goals rendered to", out)


if __name__ == "__main__":
    main()
