"""Browse sample of the v2 goal library: per source 8 random items with text, each at its largest accepted
size, one view (the left half of render_views), caption (cell count, source text) under its own boxed picture.

    nice -n 10 .venv/bin/python scripts/goals_browse_v2.py <out dir> [--root data/goal_library_v2]

Writes <out dir>/goalv2-<source>-<k>.png and <out dir>/cards-goals-v2.json.
"""

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from beebot.vla.voxel import goal_library as G

SOURCES = ("stabletext2brick", "hack337", "craftassist_houses", "craftassist_parts", "iglu_multiturn", "iglu_single")


def picture(cells, caption, scale=11):
    full = G.render_views(cells, scale=scale)
    left = full.crop((0, 0, full.width // 2, full.height))
    out = Image.new("RGB", (left.width, left.height + 70), (255, 255, 255))
    out.paste(left, (0, 0))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, out.width - 1, out.height - 1], outline=(120, 120, 120))
    y = left.height + 3
    for line in caption:
        for part in textwrap.wrap(line, 52)[:3]:
            draw.text((4, y), part, fill=(20, 20, 20))
            y += 13
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--root", default="data/goal_library_v2")
    parser.add_argument("--per-source", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    cards = []
    for name in SOURCES:
        lines, cells = G.load(name, root=Path(args.root))
        largest = {}
        for i, line in enumerate(lines):
            if not (line["text"] or "").strip():
                continue
            key = (line["source_id"], line["item"])
            if key not in largest or line["cells"] > lines[largest[key]]["cells"]:
                largest[key] = i
        ids = sorted(largest.values())
        for k, i in enumerate(rng.choice(ids, size=min(args.per_source, len(ids)), replace=False).tolist()):
            line = lines[i]
            png = f"goalv2-{name}-{k}.png"
            picture(cells[i], [f"{line['cells']} cells ({line['dims']})", line["text"][:150]]).save(out / png)
            cards.append({"name": png, "generator": name, "cells": line["cells"], "text": line["text"][:300], "params": f"{line['source_id']} scale {line['scale']} dims {line['dims']}"})
    (out / "cards-goals-v2.json").write_text(json.dumps(cards, indent=1))
    print(len(cards))


if __name__ == "__main__":
    main()
