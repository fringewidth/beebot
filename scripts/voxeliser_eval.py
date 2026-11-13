"""Score the goal-library voxeliser: old rule (threshold 0.3, no phase search, no repair) against the search
(thresholds x grid phases x thin-part repair), on a fixed sample, and render a comparison.

    nice -n 10 .venv/bin/python scripts/voxeliser_eval.py <out dir> [--n-brick 200] [--n-house 50]

Per (item, scale) accepted by both rules: mean IoU of the top, front and side silhouettes against the source's
(the source's present-voxel bins at the same grid), soft 3D IoU against the source fill, the share of the
source's face-connected components (largest 8) and of its floor contacts (components of the z = 0 layer) that
keep at least one coarse cell. Writes voxeliser-eval.json and voxeliser-compare.png into <out dir>.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
import goals_convert as C  # noqa: E402

from beebot.vla.voxel import goal_library as G  # noqa: E402
from beebot.vla.voxel.render import render_scene  # noqa: E402


def survival(base, comps, floors, cand):
    """(component survival, floor-contact survival) of one candidate."""
    if cand["grid"] is None:
        return 1.0, 1.0
    pads, edges = cand["grid"]
    mask = cand["mask"]

    def kept(part):
        total, _ = G._sum(part, pads, edges)
        return bool(((total > 0) & mask).any())

    comp = np.mean([kept(c) for c in comps]) if comps else 1.0
    floor = np.mean([kept(f) for f in floors]) if floors else 1.0
    return float(comp), float(floor)


def parts_of(base):
    labels, n = ndimage.label(base)
    sizes = ndimage.sum(base, labels, range(1, n + 1))
    order = np.argsort(-np.asarray(sizes))[:8]
    comps = [labels == (int(i) + 1) for i in order]
    flabels, fn = ndimage.label(base[0])
    floors = []
    for k in range(1, fn + 1):
        f = np.zeros_like(base)
        f[0] = flabels == k
        floors.append(f)
    return comps, floors


def measure(item, max_cells=800):
    base = G.crop_bbox(item.vox)
    if base is None:
        return []
    comps, floors = parts_of(base)
    rows = []
    for scale in G.SCALES:
        pair = {}
        for name, search in (("old", False), ("new", True)):
            found = G.best_variant(base, scale, max_cells, search=search)
            if isinstance(found, str):
                break
            comp, floor = survival(base, comps, floors, found)
            pair[name] = {"iou": found["iou"], "soft": found["soft"], "comp": comp, "floor": floor, "cells": int(found["vox"].sum())}
        if len(pair) == 2 and scale < 1:
            rows.append({"scale": scale, **{f"{k}_{n}": v for n, d in pair.items() for k, v in d.items()}})
    return rows


def sample_items(n_brick, n_house):
    rng = np.random.default_rng(0)
    bricks = list(C.tasks_stabletext2brick())
    houses = list(C.tasks_craftassist_houses())
    items = []
    for i in rng.choice(len(bricks), n_brick, replace=False):
        items += C.make_stabletext2brick(bricks[int(i)])
    for i in rng.choice(len(houses), n_house, replace=False):
        items += C.make_craftassist_houses(houses[int(i)])[:1]
    return items


def tile(vox, caption, size=(250, 215)):
    vox = np.asarray(vox, dtype=bool)
    d, h, w = vox.shape
    s = max(1, min(10, 230 // (h + w + 2)))
    image = render_scene(vox.astype(np.float32), np.zeros_like(vox), scale=s)
    image.thumbnail(size)
    out = Image.new("RGB", (size[0], size[1] + 62), (255, 255, 255))
    out.paste(image, ((size[0] - image.width) // 2, 4))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, out.width - 1, out.height - 1], outline=(120, 120, 120))
    y = size[1] + 2
    for line in caption:
        draw.text((4, y), line[:44], fill=(20, 20, 20))
        y += 13
    return out


def compare(items, path, count=12, target=250):
    wanted = ("table", "chair", "sofa", "bench", "lamp", "bed", "cabinet", "house")
    picks = []
    for word in wanted:
        for item in items:
            if word in item.text.lower() and item not in picks:
                picks.append(item)
                break
    for item in items:
        if len(picks) >= count:
            break
        if item not in picks:
            picks.append(item)
    tiles = []
    for item in picks[:count]:
        base = G.crop_bbox(item.vox)
        best = None
        for scale in G.SCALES:
            if scale >= 1:
                continue
            new = G.best_variant(base, scale, 800, search=True)
            if isinstance(new, str):
                continue
            gap = abs(int(new["vox"].sum()) - target)
            if best is None or gap < best[0]:
                best = (gap, scale, new)
        text = (item.text or item.label or "")[:60]
        row = [tile(base, [f"source {'x'.join(map(str, base.shape))}", f"{int(base.sum())} voxels", text])]
        if best:
            old = G.best_variant(base, best[1], 800, search=False)
            if not isinstance(old, str):
                row.append(tile(old["vox"], ["old voxeliser", f"{int(old['vox'].sum())} cells, IoU {old['iou']:.2f}", f"scale {best[1]:.3f}"]))
            else:
                row.append(tile(np.zeros((1, 1, 1), bool), ["old voxeliser", f"rejected: {old}", f"scale {best[1]:.3f}"]))
            new = best[2]
            row.append(tile(new["vox"], ["new voxeliser", f"{int(new['vox'].sum())} cells, IoU {new['iou']:.2f}", f"scale {best[1]:.3f}"]))
        tiles.append(row)
    cols = 2
    rows = [tiles[i : i + cols] for i in range(0, len(tiles), cols)]
    tw, th = tiles[0][0].size
    sheet = Image.new("RGB", (cols * 3 * tw + 8 * (cols * 3 + 1), len(rows) * (th + 8) + 8), (240, 240, 236))
    for r, group in enumerate(rows):
        x = 8
        for row in group:
            for t in row:
                sheet.paste(t, (x, 8 + r * (th + 8)))
                x += tw + 8
    sheet.save(path)
    return [{"text": i.text, "source": i.source, "id": i.source_id} for i in picks[:count]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--n-brick", type=int, default=200)
    parser.add_argument("--n-house", type=int, default=50)
    parser.add_argument("--no-compare", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = sample_items(args.n_brick, args.n_house)
    report = {}
    for source in ("stabletext2brick", "craftassist"):
        rows = []
        n = 0
        for item in items:
            if item.source == source:
                n += 1
                rows += measure(item)
        keys = ("iou", "soft", "comp", "floor", "cells")
        report[source] = {"items": n, "pairs": len(rows)}
        for name in ("old", "new"):
            report[source][name] = {k: round(float(np.mean([r[f"{k}_{name}"] for r in rows])), 4) for k in keys}
        print(source, json.dumps(report[source]), flush=True)
    if not args.no_compare:
        report["compared"] = compare(items, out / "voxeliser-compare.png")
    (out / "voxeliser-eval.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
