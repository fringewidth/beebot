"""Browse cards for the machine parts: 60 single parts and ~10 whole assemblies (a colour per part).

    .venv/bin/python scripts/machine_parts_render.py <browse dir> [--parts 60] [--assemblies 10]

Reads data/machine_parts/{mcb,fusion360}.{npz,jsonl} and fusion360_assemblies.{npz,jsonl}; writes parts-NN.png,
asm-NN.png and cards-parts.json into the browse dir (see the "Machine parts" section of browse_page.py).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from beebot.vla.voxel import goal_library as G
from beebot.vla.voxel.render import BACKGROUND, SHADES, _shade

PALETTE = [
    (214, 80, 70), (70, 130, 200), (90, 170, 100), (230, 170, 50), (150, 100, 190), (60, 180, 180),
    (200, 110, 150), (140, 140, 60), (240, 130, 60), (100, 100, 100), (120, 190, 60), (60, 90, 150),
]
OUT_ROOT = Path("data/machine_parts")


def draw_ids(cells, dims, scale=9, turned=False):
    """Isometric view of (N, 4) cells (z, y, x, part) in a centred 16^3 box, one colour per part."""
    colour = {}
    h, w = dims[1], dims[2]
    oy, ox0 = (G.SIZE - h) // 2, (G.SIZE - w) // 2
    for z, y, x, part in cells.astype(int):
        y, x = y + oy, x + ox0
        if turned:
            y, x = G.SIZE - 1 - y, G.SIZE - 1 - x
        colour[(z, y, x)] = PALETTE[part % len(PALETTE)]
    d = max(z for z, _, _ in colour) + 1
    s = scale
    width = (G.SIZE + G.SIZE) * s + 2 * s
    height = (G.SIZE + G.SIZE) * s // 2 + d * s + 2 * s
    ox, oyy = G.SIZE * s + s, d * s + s
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    corners = [(0, 0), (0, G.SIZE), (G.SIZE, G.SIZE), (G.SIZE, 0)]
    draw.polygon([(ox + (x - y) * s, oyy + (x + y) * s / 2) for y, x in corners], outline=(200, 200, 195), fill=(236, 236, 230))
    for (z, y, x), rgb in sorted(colour.items(), key=lambda item: (sum(item[0]), item[0][0])):
        cx, cy = ox + (x - y) * s, oyy + (x + y) * s / 2 - z * s
        top = [(cx, cy - s), (cx + s, cy - s / 2), (cx, cy), (cx - s, cy - s / 2)]
        left = [(cx - s, cy - s / 2), (cx, cy), (cx, cy + s), (cx - s, cy + s / 2)]
        right = [(cx, cy), (cx + s, cy - s / 2), (cx + s, cy + s / 2), (cx, cy + s)]
        draw.polygon(top, fill=_shade(rgb, SHADES[0]))
        draw.polygon(left, fill=_shade(rgb, SHADES[1]))
        draw.polygon(right, fill=_shade(rgb, SHADES[2]))
    return image


def two_views(cells, dims, scale=9):
    left = draw_ids(cells, dims, scale)
    right = draw_ids(cells, dims, scale, turned=True)
    image = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), BACKGROUND)
    image.paste(left, (0, 0))
    image.paste(right, (left.width, 0))
    return image


def pick_parts(lines, count, rng):
    """One variant per part, spread over labels (round robin over a shuffled label list), preferring flat
    orientation and a cell count near 250."""
    by_label = {}
    for i, line in enumerate(lines):
        key = line["label"] or line["extra"].get("assembly", "")
        by_label.setdefault(key, []).append(i)
    labels = list(by_label)
    rng.shuffle(labels)
    chosen, seen_parts = [], set()
    while len(chosen) < count and labels:
        for label in list(labels):
            pool = [i for i in by_label[label] if lines[i]["source_id"] not in seen_parts]
            if not pool:
                labels.remove(label)
                continue
            pool.sort(key=lambda i: (lines[i]["extra"].get("orientation") != "flat", abs(lines[i]["cells"] - 250), rng.random()))
            pick = pool[0]
            seen_parts.add(lines[pick]["source_id"])
            chosen.append(pick)
            if len(chosen) >= count:
                break
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--parts", type=int, default=60)
    parser.add_argument("--assemblies", type=int, default=10)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    cards = []
    shares = {"mcb": args.parts // 2, "fusion360": args.parts - args.parts // 2}
    n = 0
    for source, count in shares.items():
        if not (OUT_ROOT / f"{source}.jsonl").exists():
            continue
        lines, cells = G.load(source, root=OUT_ROOT, include_dropped=True)
        for i in pick_parts(lines, count, rng):
            line = lines[i]
            extra = line["extra"]
            n += 1
            name = f"parts-{n:02d}.png"
            G.render_views(cells[i]).save(out / name)
            joints = [f"{k}" for k in extra.get("joint_kinds", [])]
            cards.append(
                {
                    "name": name,
                    "source": source,
                    "assembly": extra.get("assembly_name") or extra.get("assembly") or "",
                    "part": extra.get("part_name") or line["source_id"],
                    "label": line["label"],
                    "joints": joints,
                    "dims_mm": extra.get("dims_mm"),
                    "cells": line["cells"],
                    "scale": line["scale"],
                    "thickness_cells": extra.get("thickness_cells"),
                    "orientation": extra.get("orientation"),
                }
            )
    asm_path = OUT_ROOT / "fusion360_assemblies.jsonl"
    if asm_path.exists():
        lines = [json.loads(t) for t in open(asm_path)]
        data = np.load(OUT_ROOT / "fusion360_assemblies.npz")
        starts, flat = data["starts"], data["cells"]
        order = sorted(range(len(lines)), key=lambda i: (-(3 <= len(lines[i]["parts"]) <= 14), -len(lines[i]["parts"]) if len(lines[i]["parts"]) <= 14 else 99, rng.random()))
        order = [i for i in order if lines[i]['cells'] >= 80]
        taken = 0
        for i in order:
            if taken >= args.assemblies:
                break
            cells = flat[starts[i] : starts[i + 1]]
            line = lines[i]
            taken += 1
            name = f"asm-{taken:02d}.png"
            two_views(cells, line["dims"]).save(out / name)
            cards.append(
                {
                    "name": name,
                    "source": "fusion360 (whole assembly)",
                    "assembly": line["assembly_name"] or line["assembly"],
                    "part": f"{len(line['parts'])} parts in view",
                    "label": ", ".join(line["categories"]) if line["categories"] else "",
                    "joints": sorted({k for p in line["parts"] for k in p["joints"]}),
                    "dims_mm": None,
                    "cells": line["cells"],
                    "scale": line["scale"],
                    "legend": [f"colour {p['index'] % len(PALETTE)}: {p['name']}{' (auto name)' if p['generic'] else ''}" for p in line["parts"]],
                }
            )
    with open(out / "cards-parts.json", "w") as handle:
        json.dump(cards, handle)
    print(len(cards), "cards")


if __name__ == "__main__":
    main()
