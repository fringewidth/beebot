"""Goal library: end-state shapes for the swarm, converted from public 3D block and brick datasets.

A source (StableText2Brick, IGLU, CraftAssist, Hack337/Minecraft-Schematics) is parsed to a boolean voxel
occupancy (z, y, x), z up, plus any text (captions, instructions, titles) and part labels. Each structure
yields items (a whole structure, labelled parts, random crops). Each item is re-voxelised at several scales
(and rotated about the vertical axis) to fit the 16 x 16 x 16 world, and kept as an accepted goal when

1. it has between MIN_CELLS and MAX_CELLS cells and fits in SIZE cells along every axis,
1. every cell is face-connected to the floor through other cells (what world.supported asks of robots
   standing on the floor with no terrain: magnets hold overhangs, but a cell with no face-connected path
   down is not a goal).

Geometry is bbox-cropped (the structure's lowest layer is z = 0) and stored as a cell list. `to_goal` places
an entry in a world (centred in y, x) as a `reverse.Goal`, ready for `reverse.take_apart`. Reversal results
are recorded per goal by scripts/goals_reverse.py; the library is never filtered on them.

Stored layout, per source `data/goal_library/<source>`: `<source>.npz` with `cells` (N, 3) uint8 (z, y, x) and
`starts` (G + 1,) int64 offsets, and `<source>.jsonl`, one line per goal with fields id, source, source_id,
item, kind, scale, rot, dims, cells (count), text, label, extra.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SIZE = 16
MIN_CELLS = 4
MAX_CELLS = 115
SCALES = tuple(2.0 ** (-k / 2) for k in range(-3, 13))  # 2.83 ... 0.0156 in steps of sqrt(2)
KEEP_TARGETS = (100, 45, 15)  # per item, the accepted scale whose cell count is closest to each of these
THRESHOLD = 0.3  # a coarse voxel is filled if this share of the source voxels under it are
CROPS_PER_STRUCTURE = 2
LIBRARY = Path("data/goal_library")


# ---------------------------------------------------------------------------------------------
# Voxel geometry


def crop_bbox(vox):
    """The occupancy cut to its bounding box (a view); None when empty."""
    if not vox.any():
        return None
    zs, ys, xs = np.nonzero(vox)
    return vox[zs.min() : zs.max() + 1, ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def rotate(vox, k):
    """Turn the structure k quarter turns about the vertical axis."""
    return np.rot90(vox, k, axes=(1, 2))


def rescale(vox, factor, threshold=THRESHOLD):
    """Re-voxelise by `factor` per axis. Up: nearest source voxel. Down: a coarse voxel is filled when at
    least `threshold` of the source voxels it covers are (box average over uneven bins)."""
    shape = np.array(vox.shape)
    out = np.maximum(1, np.rint(shape * factor).astype(int))
    if factor >= 1:
        index = [np.minimum(((np.arange(n) + 0.5) / factor).astype(int), s - 1) for n, s in zip(out, shape)]
        return vox[np.ix_(*index)]
    edges = [np.floor(np.arange(n + 1) * s / n).astype(int) for n, s in zip(out, shape)]
    total = vox.astype(np.float32)
    for axis in range(3):
        total = np.add.reduceat(total, edges[axis][:-1], axis=axis)
    volume = np.ones(tuple(out), dtype=np.float32)
    for axis in range(3):
        size = np.diff(edges[axis]).astype(np.float32)
        volume = volume * size.reshape([-1 if a == axis else 1 for a in range(3)])
    return (total / volume) >= threshold


def _grow(mask, allowed):
    """One step of face growth of `mask` inside `allowed`."""
    out = mask.copy()
    out[1:] |= mask[:-1]
    out[:-1] |= mask[1:]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[:, :, 1:] |= mask[:, :, :-1]
    out[:, :, :-1] |= mask[:, :, 1:]
    return out & allowed


def flood(seed, allowed):
    """Cells of `allowed` face-connected to `seed` (through allowed cells)."""
    held = seed & allowed
    while True:
        grown = _grow(held, allowed)
        if grown.sum() == held.sum():
            return held
        held = grown


def grounded(vox):
    """True when every cell is face-connected to the floor layer (z = 0) through cells of the structure."""
    seed = np.zeros_like(vox)
    seed[0] = vox[0]
    return bool(seed.any()) and bool((flood(seed, vox) == vox).all())


def connected(vox):
    """True when the structure is one face-connected piece."""
    first = np.zeros_like(vox)
    first[tuple(np.argwhere(vox)[0])] = True
    return bool((flood(first, vox) == vox).all())


def histogram(counts, edges=(4, 8, 16, 24, 32, 48, 64, 80, 100, 116)):
    """{"lo-hi": n} cell-count histogram over [edges[i], edges[i + 1])."""
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        out[f"{lo}-{hi - 1}"] = int(sum(1 for c in counts if lo <= c < hi))
    return out


# ---------------------------------------------------------------------------------------------
# Items and variants


@dataclass
class Item:
    """One thing to turn into goals: a whole structure, a labelled part or a crop."""

    source: str
    source_id: str
    kind: str  # whole, part, crop
    vox: np.ndarray  # (z, y, x) bool, any size
    text: str = ""  # caption, instruction or title shown to a person
    label: str = ""  # part label (roof, door ...) or category
    extra: dict = field(default_factory=dict)

    @property
    def key(self):
        return f"{self.source}/{self.source_id}/{self.kind}/{self.label}"


def variants(vox, rng, keep=KEEP_TARGETS, rotations=2):
    """Accepted re-voxelisations of one occupancy: (scale, rot, vox) tuples, plus a reject tally.

    Every scale of SCALES is tried; those that fit the world, hold MIN_CELLS to MAX_CELLS cells and stand on
    the floor are distinct shapes; per target in `keep` the one with the closest cell count is kept, in the
    original orientation and in one further random quarter turn (when that differs from the first)."""
    base = crop_bbox(vox)
    tally = {"oversize": 0, "small": 0, "big": 0, "ungrounded": 0}
    if base is None:
        return [], tally
    seen = {}
    for scale in SCALES:
        scaled = crop_bbox(rescale(base, scale))
        if scaled is None:
            tally["small"] += 1
            continue
        if max(scaled.shape) > SIZE:
            tally["oversize"] += 1
            continue
        cells = int(scaled.sum())
        if cells < MIN_CELLS:
            tally["small"] += 1
            continue
        if cells > MAX_CELLS:
            tally["big"] += 1
            continue
        if not grounded(scaled):
            tally["ungrounded"] += 1
            continue
        seen.setdefault((scaled.shape, scaled.tobytes()), (scale, scaled))
    pool = sorted(seen.values(), key=lambda it: int(it[1].sum()))
    chosen = {}
    for target in keep:
        if not pool:
            break
        best = min(pool, key=lambda it: abs(int(it[1].sum()) - target))
        chosen.setdefault(round(best[0], 4), best)
    out = []
    for scale, scaled in chosen.values():
        out.append((scale, 0, np.ascontiguousarray(scaled)))
        if rotations > 1:
            k = int(rng.integers(1, 4))
            turned = np.ascontiguousarray(rotate(scaled, k))
            if not (turned.shape == scaled.shape and (turned == scaled).all()):
                out.append((scale, k, turned))
    return out, tally


def crops(vox, rng, n=CROPS_PER_STRUCTURE, lo=4):
    """Up to n random sub-boxes of a structure, resting on its bottom layer (z = 0), each side lo to SIZE."""
    base = crop_bbox(vox)
    out = []
    if base is None:
        return out
    d, h, w = base.shape
    if h <= lo or w <= lo:
        return out
    for _ in range(n * 4):
        if len(out) == n:
            break
        ch = int(rng.integers(lo, min(h, SIZE) + 1))
        cw = int(rng.integers(lo, min(w, SIZE) + 1))
        if ch == h and cw == w:
            continue
        cd = int(rng.integers(min(lo, d), min(d, SIZE) + 1))
        z0 = 0 if rng.random() < 0.7 else int(rng.integers(0, max(d - cd, 0) + 1))
        y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
        piece = base[z0 : z0 + cd, y0 : y0 + ch, x0 : x0 + cw]
        if piece.sum() >= MIN_CELLS:
            out.append(piece)
    return out


def seed_of(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


def process(item):
    """All accepted goals of one item: (list of dicts with 'vox', 'scale', 'rot'), reject tally."""
    rng = np.random.default_rng(seed_of(item.key))
    found, tally = variants(item.vox, rng)
    return [{"scale": s, "rot": k, "vox": v} for s, k, v in found], tally


# ---------------------------------------------------------------------------------------------
# Sources: each yields Items


def parse_bricks(text, size=20):
    """StableText2Brick `bricks` field ("hxw (x,y,z)" lines, z up, a brick covers h cells in x and w in y)
    to a (z, y, x) bool grid."""
    vox = np.zeros((size, size, size), dtype=bool)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        dims, pos = line.split(" ")
        h, w = (int(v) for v in dims.split("x"))
        x, y, z = (int(v) for v in pos.strip("()").split(","))
        vox[z, y : y + w, x : x + h] = True
    return vox


def parse_blueprint(blueprint, empty=("air", "cave_air", "void_air", "structure_void", "water", "lava")):
    """Hack337 blueprint JSON (size [w, h, d], palette, layers[y][row][col]) to (z=y, y=row, x=col) bool."""
    solid = {char for char, name in blueprint["palette"].items() if name.split(":")[-1] not in empty}
    layers = blueprint["layers"]
    d = len(layers)
    h = max(len(layer) for layer in layers)
    w = max((len(row) for layer in layers for row in layer), default=0)
    vox = np.zeros((d, h, w), dtype=bool)
    for z, layer in enumerate(layers):
        for y, row in enumerate(layer):
            for x, char in enumerate(row):
                if char in solid:
                    vox[z, y, x] = True
    return vox


def parse_blocks(blocks):
    """IGLU world state [[x, y, z, colour], ...] to (y up -> z, z -> y, x) bool, bbox-cropped."""
    if not blocks:
        return None
    arr = np.asarray(blocks)[:, :3]
    arr = arr - arr.min(axis=0)
    vox = np.zeros((arr[:, 1].max() + 1, arr[:, 2].max() + 1, arr[:, 0].max() + 1), dtype=bool)
    vox[arr[:, 1], arr[:, 2], arr[:, 0]] = True
    return vox


def craftassist_occupancy(schematic):
    """CraftAssist schematic (y, z, x, [id, meta]) or (y, z, x) ids to a (z=y, y=z, x) bool: non-air."""
    ids = schematic[..., 0] if schematic.ndim == 4 else schematic
    return ids != 0


# ---------------------------------------------------------------------------------------------
# Storage


def entry_line(item, goal, gid):
    vox = goal["vox"]
    return {
        "id": gid,
        "source": item.source,
        "source_id": item.source_id,
        "item": item.key,
        "kind": item.kind,
        "scale": round(float(goal["scale"]), 4),
        "rot": int(goal["rot"]),
        "dims": [int(v) for v in vox.shape],
        "cells": int(vox.sum()),
        "text": item.text,
        "label": item.label,
        "extra": item.extra,
    }


def save(name, lines, cell_lists, root=LIBRARY):
    """Write <name>.jsonl and <name>.npz (cells, starts)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    starts = np.zeros(len(cell_lists) + 1, dtype=np.int64)
    for i, cells in enumerate(cell_lists):
        starts[i + 1] = starts[i] + len(cells)
    flat = np.concatenate(cell_lists).astype(np.uint8) if cell_lists else np.zeros((0, 3), dtype=np.uint8)
    np.savez_compressed(root / f"{name}.npz", cells=flat, starts=starts)
    with open(root / f"{name}.jsonl", "w") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")


def load(name, root=LIBRARY):
    """(list of meta dicts, list of (n, 3) uint8 cell arrays (z, y, x))."""
    root = Path(root)
    with open(root / f"{name}.jsonl") as handle:
        lines = [json.loads(text) for text in handle]
    data = np.load(root / f"{name}.npz")
    starts, cells = data["starts"], data["cells"]
    return lines, [cells[starts[i] : starts[i + 1]] for i in range(len(lines))]


def cells_to_vox(cells):
    cells = np.asarray(cells).astype(int)
    vox = np.zeros(tuple(cells.max(axis=0) + 1), dtype=bool)
    vox[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    return vox


def vox_to_cells(vox):
    return np.argwhere(vox).astype(np.uint8)


# ---------------------------------------------------------------------------------------------
# Into the world


def to_goal(cells, text="", offset=None):
    """A reverse.Goal of family "library": the cells filled with robots in an empty 16^3 world, centred in
    y and x unless `offset` (y, x) is given. The goal's own cells are not guaranteed a fixed point (overhangs)."""
    import torch

    from beebot.vla.voxel import reverse

    cells = np.asarray(cells).astype(int)
    d, h, w = cells.max(axis=0) + 1
    if offset is None:
        offset = ((SIZE - h) // 2, (SIZE - w) // 2)
    world = reverse.blank()
    target = torch.zeros(SIZE, SIZE, SIZE, dtype=torch.bool)
    z, y, x = cells[:, 0], cells[:, 1] + offset[0], cells[:, 2] + offset[1]
    world.mass[0, z, y, x] = 1.0
    target[z, y, x] = True
    return reverse.Goal(family="library", text=text, world=world, target=target)


# ---------------------------------------------------------------------------------------------
# Pictures


def render_views(cells, scale=9):
    """Two isometric views of a goal's cells (the usual corner, and the world turned 180 degrees about
    the vertical axis), side by side, as one PIL image. Cells are drawn as robots in the centred 16^3 world."""
    from PIL import Image

    from beebot.vla.voxel.render import render_scene

    cells = np.asarray(cells).astype(int)
    d, h, w = cells.max(axis=0) + 1
    mass = np.zeros((SIZE, SIZE, SIZE), dtype=np.float32)
    mass[cells[:, 0], cells[:, 1] + (SIZE - h) // 2, cells[:, 2] + (SIZE - w) // 2] = 1.0
    walls = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    left = render_scene(mass, walls, scale=scale)
    right = render_scene(mass[:, ::-1, ::-1].copy(), walls, scale=scale)
    image = Image.new("RGB", (left.width + right.width, left.height), (246, 246, 243))
    image.paste(left, (0, 0))
    image.paste(right, (left.width, 0))
    return image
