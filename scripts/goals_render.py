"""Render a sample of library goals as still images (two views side by side) for the session browse page.

    nice -n 10 .venv/bin/python scripts/goals_render.py --out <browse dir> --per-source 10

Writes <out>/goal-<source>-<k>.png and <out>/cards-goals.json (name, fam=source, kind, robots=cells, text,
label, scale, rot, source_id, plus the browse card fields e, store, said, rungs left empty). Then rerun
browse_page.py. The sample takes goals spread across each source's cell counts, preferring ones with text.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from beebot.vla.voxel import goal_library as G

SOURCES = ("stabletext2brick", "hack337", "craftassist_houses", "craftassist_parts", "iglu_multiturn", "iglu_single")


def pick(lines, n, rng):
    """n distinct indices spread over the cell-count order, preferring goals that carry text."""
    order = sorted(range(len(lines)), key=lambda i: lines[i]["cells"])
    texty = [i for i in order if lines[i]["text"]]
    pool = texty if len(texty) >= n else order
    chosen = []
    for q in range(n):
        lo, hi = int(len(pool) * q / n), int(len(pool) * (q + 1) / n)
        chosen.append(pool[int(rng.integers(lo, max(hi, lo + 1)))])
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-source", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cards = []
    for name in SOURCES:
        lines, cells = G.load(name)
        for k, i in enumerate(pick(lines, args.per_source, rng)):
            line = lines[i]
            file = f"goal-{name}-{k}.png"
            G.render_views(cells[i]).save(out / file)
            cards.append(
                {
                    "name": file,
                    "fam": name,
                    "kind": line["kind"],
                    "robots": line["cells"],
                    "text": line["text"],
                    "label": line["label"],
                    "scale": line["scale"],
                    "rot": line["rot"],
                    "source_id": line["source_id"],
                    "id": line["id"],
                }
            )
    json.dump(cards, open(out / "cards-goals.json", "w"), indent=1)
    print(len(cards), "goals rendered to", out)


if __name__ == "__main__":
    main()
