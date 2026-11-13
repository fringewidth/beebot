"""Coloured goals for the extra sources (voxel art packs, Text2Shape, Objaverse meshes).

`goal_library.variants` works on a boolean occupancy; here a source also carries a colour per voxel
(`rgb`, (z, y, x, 3) uint8) and `coloured_variants` returns, per accepted variant, the colour of every cell:
the mean colour of the source voxels under the coarse cell (cells reconnected by `repair`, which hold no source
voxels, take the mean of their coloured neighbours). The accepted shapes are the same ones `variants(...,
search=True)` accepts: same scales, threshold and phase search, repair, grounding, keep targets, rotations.

Colours are stored beside the cells: `<name>.npz` holds `cells`, `starts` and `rgb` (N, 3) uint8, aligned with
`cells`; `goal_library.load` ignores `rgb`, `load_rgb` returns it per goal.
"""

import json
import re
from pathlib import Path

import numpy as np

from beebot.vla.voxel import goal_library as G


def _bin_mean(rgb, occupied, pads, edges, shape):
    """Mean colour of the occupied source voxels per coarse cell: (*shape, 3) float, and the count per cell."""
    zs, ys, xs = np.nonzero(occupied)
    bins = []
    for axis, coord in enumerate((zs, ys, xs)):
        shifted = coord + pads[axis]
        bins.append(np.clip(np.searchsorted(edges[axis], shifted, side="right") - 1, 0, len(edges[axis]) - 2))
    total = np.zeros(tuple(shape) + (3,), dtype=np.float64)
    count = np.zeros(tuple(shape), dtype=np.float64)
    np.add.at(count, tuple(bins), 1.0)
    values = rgb[zs, ys, xs].astype(np.float64)
    for channel in range(3):
        np.add.at(total[..., channel], tuple(bins), values[:, channel])
    return total, count


def colour_cells(rgb, occupied, found, scale):
    """Colours (z, y, x, 3) uint8 for `found["mask"]`: mean source colour per cell, holes filled from neighbours."""
    mask = found["mask"]
    if found["grid"] is None:
        shape = np.array(occupied.shape)
        out = np.maximum(1, np.rint(shape * scale).astype(int))
        index = [np.minimum(((np.arange(n) + 0.5) / scale).astype(int), s - 1) for n, s in zip(out, shape)]
        return rgb[np.ix_(*index)]
    pads, edges = found["grid"]
    total, count = _bin_mean(rgb, occupied, pads, edges, mask.shape)
    colour = np.zeros(mask.shape + (3,), dtype=np.float64)
    have = count > 0
    colour[have] = total[have] / count[have][:, None]
    missing = mask & ~have
    guard = 0
    while missing.any() and guard < 20:
        guard += 1
        acc = np.zeros_like(colour)
        num = np.zeros(mask.shape)
        for axis in range(3):
            for step in (1, -1):
                shifted = np.roll(colour, step, axis=axis)
                known = np.roll(have, step, axis=axis)
                edge = [slice(None)] * 3
                edge[axis] = slice(0, 1) if step == 1 else slice(-1, None)
                known[tuple(edge)] = False
                acc += np.where(known[..., None], shifted, 0.0)
                num += known
        fill = missing & (num > 0)
        if not fill.any():
            break
        colour[fill] = acc[fill] / num[fill][:, None]
        have = have | fill
        missing = mask & ~have
    return np.rint(colour).astype(np.uint8)


def coloured_variants(rgb, occupied, rng, keep=None, rotations=2, max_cells=None, tolerance=None):
    """Like goal_library.variants(search=True) but each result also has per-cell colours.

    rgb: (z, y, x, 3) uint8, occupied: (z, y, x) bool, same shape. Returns ([(scale, rot, vox, colours)], tally)
    where colours is (z, y, x, 3) of the cropped vox (zeros where empty)."""
    keep = G.KEEP_TARGETS if keep is None else keep
    max_cells = G.MAX_CELLS if max_cells is None else max_cells
    if not occupied.any():
        return [], {"oversize": 0, "small": 0, "big": 0, "ungrounded": 0}
    zs, ys, xs = np.nonzero(occupied)
    box = (slice(zs.min(), zs.max() + 1), slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    base = np.ascontiguousarray(occupied[box])
    colours = np.ascontiguousarray(rgb[box])
    tally = {"oversize": 0, "small": 0, "big": 0, "ungrounded": 0}
    seen = {}
    for scale in G.SCALES:
        found = G.best_variant(base, scale, max_cells, search=True)
        if isinstance(found, str):
            tally[found] += 1
            continue
        scaled = found["vox"]
        seen.setdefault((scaled.shape, scaled.tobytes()), (scale, scaled, found))
    pool = sorted(seen.values(), key=lambda it: int(it[1].sum()))
    chosen = {}
    for target in keep:
        if not pool:
            break
        best = min(pool, key=lambda it: abs(int(it[1].sum()) - target))
        if tolerance is not None and not target / tolerance <= int(best[1].sum()) <= target * tolerance:
            continue
        chosen.setdefault(round(best[0], 4), best)
    out = []
    for scale, scaled, found in chosen.values():
        full = colour_cells(colours, base, found, scale)
        mask = found["mask"]
        zz, yy, xx = np.nonzero(mask)
        crop = (slice(zz.min(), zz.max() + 1), slice(yy.min(), yy.max() + 1), slice(xx.min(), xx.max() + 1))
        paint = np.ascontiguousarray(np.where(scaled[..., None], full[crop], 0)).astype(np.uint8)
        out.append((scale, 0, np.ascontiguousarray(scaled), paint))
        if rotations > 1:
            k = int(rng.integers(1, 4))
            turned = np.ascontiguousarray(G.rotate(scaled, k))
            if not (turned.shape == scaled.shape and (turned == scaled).all()):
                out.append((scale, k, turned, np.ascontiguousarray(np.rot90(paint, k, axes=(1, 2)))))
    return out, tally


def indices_to_rgb(indices, palette):
    """(z, y, x) palette indices (0 empty, i -> palette row i - 1) to (z, y, x, 3) uint8."""
    out = np.zeros(indices.shape + (3,), dtype=np.uint8)
    filled = indices > 0
    out[filled] = palette[indices[filled].astype(int) - 1]
    return out


# ---------------------------------------------------------------------------------------------
# Storage with colours


def save_coloured(name, lines, cell_lists, rgb_lists, root):
    """<name>.jsonl and <name>.npz (cells, starts, rgb); rgb[i] aligns with cell_lists[i] row by row."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    starts = np.zeros(len(cell_lists) + 1, dtype=np.int64)
    for i, cells in enumerate(cell_lists):
        starts[i + 1] = starts[i] + len(cells)
    flat = np.concatenate(cell_lists).astype(np.uint8) if cell_lists else np.zeros((0, 3), dtype=np.uint8)
    arrays = {"cells": flat, "starts": starts}
    if rgb_lists is not None:  # sources without colour (meshes) store no rgb array
        arrays["rgb"] = np.concatenate(rgb_lists).astype(np.uint8) if rgb_lists else np.zeros((0, 3), dtype=np.uint8)
    np.savez_compressed(root / f"{name}.npz", **arrays)
    with open(root / f"{name}.jsonl", "w") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")


def load_rgb(name, root, max_cells=G.MAX_CELLS):
    """List of (n, 3) uint8 colours per goal, or None when the source has no colours. Goals with more than
    `max_cells` cells are dropped (None keeps all), matching goal_library.load."""
    data = np.load(Path(root) / f"{name}.npz")
    if "rgb" not in data.files:
        return None
    starts, rgb = data["starts"], data["rgb"]
    return [rgb[starts[i] : starts[i + 1]] for i in range(len(starts) - 1) if max_cells is None or starts[i + 1] - starts[i] <= max_cells]


def cell_colours(vox, paint):
    """Colours of the cells in `G.vox_to_cells(vox)` order (np.argwhere order)."""
    return paint[np.nonzero(vox)]


def render_coloured(cells, rgb, scale=9):
    """Isometric view (usual corner and turned 180 degrees) with per-cell colours, as one PIL image."""
    from PIL import Image, ImageDraw

    from beebot.vla.voxel.render import BACKGROUND, _shade, SHADES

    cells = np.asarray(cells).astype(int)
    d, h, w = cells.max(axis=0) + 1

    def one(flip):
        y = cells[:, 1] + (G.SIZE - h) // 2
        x = cells[:, 2] + (G.SIZE - w) // 2
        if flip:
            y, x = G.SIZE - 1 - y, G.SIZE - 1 - x
        z = cells[:, 0]
        s = scale
        size = G.SIZE
        width = (size + size) * s + 2 * s
        height = (size + size) * s // 2 + max(d, 1) * s + 2 * s
        ox, oy = size * s + s, d * s + s
        image = Image.new("RGB", (width, height), BACKGROUND)
        draw = ImageDraw.Draw(image)
        corners = [(0, 0), (0, size), (size, size), (size, 0)]
        draw.polygon([(ox + (cx - cy) * s, oy + (cx + cy) * s / 2) for cy, cx in corners], outline=(200, 200, 195), fill=(236, 236, 230))
        order = np.lexsort((z, x + y + z))
        for i in order:
            cx, cy = ox + (x[i] - y[i]) * s, oy + (x[i] + y[i]) * s / 2 - z[i] * s
            rgb_i = tuple(int(v) for v in rgb[i])
            top = [(cx, cy - s), (cx + s, cy - s / 2), (cx, cy), (cx - s, cy - s / 2)]
            left = [(cx - s, cy - s / 2), (cx, cy), (cx, cy + s), (cx - s, cy + s / 2)]
            right = [(cx, cy), (cx + s, cy - s / 2), (cx + s, cy + s / 2), (cx, cy + s)]
            draw.polygon(top, fill=_shade(rgb_i, SHADES[0]))
            draw.polygon(left, fill=_shade(rgb_i, SHADES[1]))
            draw.polygon(right, fill=_shade(rgb_i, SHADES[2]))
        return image

    left, right = one(False), one(True)
    image = Image.new("RGB", (left.width + right.width, left.height), BACKGROUND)
    image.paste(left, (0, 0))
    image.paste(right, (left.width, 0))
    return image


# ---------------------------------------------------------------------------------------------
# Text from names


def words(name):
    """'Watch_Tower-02' -> 'watch tower'. Digits and an extension are dropped."""
    stem = re.sub(r"\.(vox|obj|glb|zip)$", "", name.strip(), flags=re.I)
    stem = re.sub(r"([a-z])([A-Z])", r"\1 \2", stem)
    stem = re.sub(r"[_\-\.\s]+", " ", stem)
    stem = re.sub(r"\b\d+\b", " ", stem)
    return re.sub(r"\s+", " ", stem).strip().lower()
