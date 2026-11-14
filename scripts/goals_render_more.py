"""Render 12 goals per extra source with the text under each picture (tile boxed), colour where the source has it.

    nice -n 10 .venv/bin/python scripts/goals_render_more.py --out <browse dir> [--per-source 12] [--root data/goal_library_v2]

Writes <out>/more-<source>-<k>.png and <out>/cards-more.json (name, source, text, cells, label, id). The 12 are
spread over the cell counts of the source (4-115 up to 501-800), preferring goals with text.
"""

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from beebot.vla.voxel import goal_library as G
from beebot.vla.voxel import goal_sources as GS

SOURCES = ("oga_vox", "itch_vox", "objaverse_cap3d", "text2shape")


def pick(lines, n, rng):
    order = sorted(range(len(lines)), key=lambda i: lines[i]["cells"])
    pool = [i for i in order if lines[i]["text"]] or order
    chosen = []
    for q in range(n):
        lo, hi = int(len(pool) * q / n), int(len(pool) * (q + 1) / n)
        chosen.append(pool[int(rng.integers(lo, max(hi, lo + 1)))])
    return chosen


def tile(picture, caption, width):
    font = ImageFont.load_default(size=15)
    lines = textwrap.wrap(caption, 62)[:4] or ["(no text)"]
    height = picture.height + 14 + 20 * len(lines)
    out = Image.new("RGB", (max(width, picture.width) + 16, height + 16), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    out.paste(picture, (8, 8))
    for k, text in enumerate(lines):
        draw.text((10, picture.height + 14 + 20 * k), text, fill=(20, 20, 20), font=font)
    draw.rectangle([0, 0, out.width - 1, out.height - 1], outline=(60, 60, 60), width=2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-source", type=int, default=12)
    parser.add_argument("--root", default="data/goal_library_v2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cards = []
    for name in SOURCES:
        if not (Path(args.root) / f"{name}.jsonl").exists():
            continue
        lines, cells = G.load(name, root=args.root, include_dropped=True)
        colours = GS.load_rgb(name, args.root, include_dropped=True)
        for k, i in enumerate(pick(lines, args.per_source, rng)):
            picture = GS.render_coloured(cells[i], colours[i]) if colours is not None else G.render_views(cells[i])
            line = lines[i]
            caption = f"{line['text']}  [{line['cells']} cells]"
            file = f"more-{name}-{k}.png"
            tile(picture, caption, 560).save(out / file)
            cards.append({"name": file, "source": name, "text": line["text"], "cells": line["cells"], "label": line["label"], "id": line["id"], "coloured": colours is not None})
    json.dump(cards, open(out / "cards-more.json", "w"), indent=1)
    print(len(cards), "tiles to", out)


if __name__ == "__main__":
    main()
