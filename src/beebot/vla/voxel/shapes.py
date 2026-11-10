"""Named 3D objects for the voxel world, as small programs of parameterised parts, and the
rules that decide which of them pivoting cubes with switchable bonds can stand up and build.

Parts. Every object is a list of parts. A part is a primitive (PRIMS) with a position `at`
(z, y, x: its lowest corner in the object's frame, z up), a size (h, d, w: extent along z, y, x),
an axis where the primitive has a direction, a glyph character for `glyph`, and a label: the
sub-goal it belongs to ("legs", "seat", "back"). A stool is

    legs: col 0 0 0 3 1 1, col 0 0 2 3 1 1, col 0 2 0 3 1 1, col 0 2 2 3 1 1; seat: slab 3 0 0 1 3 3

which is also the text form (program_text, parse_program): groups of parts with one label,
separated by "; ", parts within a group by ", ", each part "prim z y x h d w [axis] [char]". The
text parses back to the same parts and so to the same voxels; a language model can write it.

Primitives (render):
  box, col, slab, wall, beam  a solid block h x d x w (the names say how it is used)
  arch     a block with an opening: legs one cell thick at both ends of the axis (x or y), a lintel
           on top, the opening below it
  ring     a rectangular outline, d x w, h layers (axis x or y: an upright frame in that plane,
           thickness along the other floor axis)
  tube     an elliptic outline d x w, h layers;  disc: the filled ellipse
  ramp     steps rising along a signed axis (+x, -x, +y, -y), solid below each step
  roof     an A-frame along an axis: two sloping bands two cells wide that meet at the ridge; hollow
  pyramid  stepped, solid, each layer one cell smaller on every side
  dome     a half ellipsoid, hollow, stepped so that every layer rests on the one below
  glyph    a letter or digit of the 5 x 7 font standing upright, columns along a signed axis,
           square corners where the font has a diagonal step (so every stroke is bonded face to face)

Support (levers, buildable, build_order). A cube is held when it rests on the floor or terrain, on
a held cube, or is bonded face to face to a held cube beside it or above it. Bonds hold
overhangs, but only so far: the lever of a cube is the number of sideways (or hanging) bonds
between it and the nearest column that carries it to the ground, and a structure stands only if
every cube's lever is at most CANTILEVER (3). Weight, torque and the strength of a bond are not
modelled; the lever stands in for all three. A structure is buildable cube by cube when there is
an order in which every partial structure stands. Adding a cube never raises another cube's
lever, so a cube that can be placed stays placeable and the greedy order (lowest first) finds an
order whenever one exists. An object is built part group by part group (program order); it
passes when every group can be completed on top of the earlier ones.

Functions (FUNCTIONS): checks of what a structure is good for, on any occupancy grid: sit (a
supported flat surface of at least 2 x 2 at height 2 to 4), table (a flat top at height 3 or
more with open space under it), shelter (a roofed space walled on two or more sides), span (a
straight run between two supports with open space under it), contain (a cavity open at the top).
They score instructions that name a use ("something you can sit on") and can serve as rewards.

The library (LIBRARY) holds parametric generators: each draw varies size and proportions.
fit_variant draws one that a swarm of n robots can build. Meshes (mesh_shape, read_off,
read_obj) are voxelised, decimated to fit and decomposed into box parts (boxes_of), then kept only
if they pass the support rule; README "Build" proposes a dataset. Nothing here downloads anything.
"""

import re
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

from beebot.vla.shapes import FONT

CANTILEVER = 3  # the longest overhang (bonds sideways or hanging from a column) a cube may sit at
GRID = 16
FOOTPRINT = 12  # an object's floor plan at most this many cells on a side (the build site keeps a margin)
HEIGHT = 11  # and at most this tall
PRIMS = ("box", "col", "slab", "wall", "beam", "arch", "ring", "tube", "disc", "ramp", "roof", "pyramid", "dome", "glyph")
AXES = ("x", "y", "+x", "-x", "+y", "-y")
DIGITS = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}
GLYPHS = {**FONT, **DIGITS}
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


# ---------------------------------------------------------------------------------------------
# Parts and their voxels


@dataclass(frozen=True)
class Part:
    prim: str
    at: tuple  # (z, y, x) lowest corner in the object's frame
    size: tuple  # (h, d, w)
    label: str = ""  # the sub-goal (group) this part belongs to
    axis: str = ""
    char: str = ""


@dataclass
class Shape:
    """A named object: its parts program in its own frame (lowest corner at the origin)."""

    name: str
    category: str
    parts: tuple
    functions: tuple = ()
    params: dict = field(default_factory=dict)

    def cells(self):
        return program_cells(self.parts)

    def text(self):
        return program_text(self.parts)


def square_glyph(rows):
    """(7, 5) bool of a glyph with a cell added under the upper cell of every diagonal step of the
    font (row 0 is the top), so every stroke is bonded face to face."""
    g = np.array([[c == "1" for c in row] for row in rows])
    out = g.copy()
    for r in range(g.shape[0] - 1):
        for c in range(g.shape[1]):
            for s in (-1, 1):
                if 0 <= c + s < g.shape[1] and g[r, c] and g[r + 1, c + s] and not g[r + 1, c] and not g[r, c + s]:
                    out[r + 1, c] = True
    return out


def _ellipse(d, w, solid_fill=True):
    yy, xx = np.mgrid[0:d, 0:w]
    inside = ((yy - (d - 1) / 2) / (d / 2)) ** 2 + ((xx - (w - 1) / 2) / (w / 2)) ** 2 <= 1.0
    if solid_fill:
        return inside
    pad_grid = np.pad(inside, 1)
    core = pad_grid[1:-1, 1:-1] & pad_grid[:-2, 1:-1] & pad_grid[2:, 1:-1] & pad_grid[1:-1, :-2] & pad_grid[1:-1, 2:]
    return inside & ~core


def part_grid(part):
    """(h, d, w) bool: the voxels of a part in its own box."""
    h, d, w = part.size
    prim, axis = part.prim, part.axis
    g = np.zeros((h, d, w), bool)
    if min(h, d, w) <= 0:
        return g
    if prim in ("box", "col", "slab", "wall", "beam"):
        g[:] = True
    elif prim == "arch":
        g[:] = True
        if axis.endswith("y"):
            g[: h - 1, 1 : d - 1, :] = False
        else:
            g[: h - 1, :, 1 : w - 1] = False
    elif prim == "ring":
        if axis == "":
            g[:] = True
            g[:, 1 : d - 1, 1 : w - 1] = False
        elif axis.endswith("x"):  # upright in the x-z plane
            g[:] = True
            g[1 : h - 1, :, 1 : w - 1] = False
        else:
            g[:] = True
            g[1 : h - 1, 1 : d - 1, :] = False
    elif prim in ("tube", "disc"):
        g[:] = _ellipse(d, w, prim == "disc")[None]
    elif prim == "ramp":
        along = w if axis.endswith("x") else d
        for k in range(along):
            top = max(1, int(np.ceil((k + 1) * h / along)))
            j = k if axis.startswith("+") else along - 1 - k
            if axis.endswith("x"):
                g[:top, :, j] = True
            else:
                g[:top, j, :] = True
    elif prim == "roof":
        across = d if axis.endswith("x") else w  # the ridge runs along the axis
        for layer in range(h):
            lo, hi = layer, across - 1 - layer
            if lo > hi:
                break
            band = np.zeros(across, bool)
            band[lo : min(lo + 2, hi + 1)] = True
            band[max(hi - 1, lo) : hi + 1] = True
            if layer == h - 1:
                band[lo : hi + 1] = True  # the ridge closes
            if axis.endswith("x"):
                g[layer, band, :] = True
            else:
                g[layer, :, band] = True
    elif prim == "pyramid":
        for layer in range(h):
            if layer > (d - 1) / 2 or layer > (w - 1) / 2:
                if layer == 0:
                    g[0] = True
                break
            g[layer, layer : d - layer, layer : w - layer] = True
    elif prim == "dome":  # stepped: each layer the outline of its ellipse widened to sit on the layer below
        zz, yy, xx = np.mgrid[0:h, 0:d, 0:w]
        solid = ((zz + 0.5) / h) ** 2 + ((yy - (d - 1) / 2) / (d / 2)) ** 2 + ((xx - (w - 1) / 2) / (w / 2)) ** 2 <= 1.0
        for z in range(h):
            above = solid[z + 1] if z + 1 < h else np.zeros((d, w), bool)
            pad_grid = np.pad(above, 1)
            inner = pad_grid[1:-1, 1:-1] & pad_grid[:-2, 1:-1] & pad_grid[2:, 1:-1] & pad_grid[1:-1, :-2] & pad_grid[1:-1, 2:]
            g[z] = solid[z] & ~inner if z + 1 < h else solid[z]
    elif prim == "glyph":
        glyph = square_glyph(GLYPHS[part.char])[::-1]  # row 0 at the bottom
        rows, cols = glyph.shape
        if axis.startswith("-"):
            glyph = glyph[:, ::-1]
        if axis.endswith("x"):
            g[:rows, :, :cols] = glyph[:h, None, :w]
        else:
            g[:rows, :cols, :] = glyph[:h, :d, None]
    else:
        raise ValueError(f"unknown primitive {prim!r}")
    return g


def glyph_size(char, thickness=1, axis="+x"):
    """(h, d, w) of an upright glyph."""
    return (7, thickness, 5) if axis.endswith("x") else (7, 5, thickness)


def part_cells(part):
    """(n, 3) int: the voxels of a part in the object's frame."""
    return np.argwhere(part_grid(part)) + np.array(part.at)


def program_cells(parts):
    """(n, 3) int: the distinct voxels of a parts program, sorted."""
    if not parts:
        return np.zeros((0, 3), int)
    cells = np.concatenate([part_cells(p) for p in parts])
    return np.unique(cells, axis=0)


def groups(parts):
    """[(label, [parts])] in program order; a label that comes back after another starts a new group."""
    out = []
    for p in parts:
        if out and out[-1][0] == p.label:
            out[-1][1].append(p)
        else:
            out.append((p.label, [p]))
    return out


def group_cells(parts):
    """[(label, (n, 3) cells new in this group)] in program order: a voxel belongs to the first
    group that has it."""
    known = set()
    out = []
    for label, members in groups(parts):
        cells = [tuple(c) for c in program_cells(members).tolist() if tuple(c) not in known]
        known.update(cells)
        out.append((label, np.array(cells, int).reshape(-1, 3)))
    return out


def normalise(parts):
    """The parts moved so that their voxels' lowest corner is the origin."""
    cells = program_cells(parts)
    low = cells.min(axis=0) if len(cells) else np.zeros(3, int)
    return tuple(replace(p, at=tuple(int(a - b) for a, b in zip(p.at, low))) for p in parts)


def to_grid(cells, dims=None):
    """(D, H, W) bool of cells (n, 3); dims default to their extent."""
    cells = np.asarray(cells, int).reshape(-1, 3)
    dims = tuple(cells.max(axis=0) + 1) if dims is None else dims
    out = np.zeros(dims, bool)
    out[tuple(cells.T)] = True
    return out


ROTATE_AXIS = {"": "", "x": "y", "y": "x", "+x": "+y", "+y": "-x", "-x": "-y", "-y": "+x"}


def rotate(parts, k=1):
    """The program turned k quarter turns about the vertical (seen from above, y down and x to the
    right: x goes to y). Voxels map as (z, y, x) -> (z, x, H - 1 - y) per turn, H the program's
    extent along y; to_grid of the result equals np.rot90 of the grid over axes (1, 2), k times
    clockwise (np.rot90(grid, -k, axes=(1, 2)))."""
    parts = normalise(parts)
    for _ in range(k % 4):
        extent = program_cells(parts).max(axis=0) + 1
        turn_parts = []
        for p in parts:
            z, y, x = p.at
            h, d, w = p.size
            turn_parts.append(replace(p, at=(z, x, int(extent[1]) - y - d), size=(h, w, d), axis=ROTATE_AXIS[p.axis]))
        parts = normalise(turn_parts)
    return parts


# ---------------------------------------------------------------------------------------------
# The text form


def part_text(p):
    words = [p.prim, *map(str, p.at), *map(str, p.size)]
    if p.axis:
        words.append(p.axis)
    if p.char:
        words.append(p.char)
    return " ".join(words)


def program_text(parts):
    """The compact text of a parts program: "label: part, part; label: part"."""
    return "; ".join(f"{label}: " + ", ".join(part_text(p) for p in members) for label, members in groups(parts))


PART_PATTERN = re.compile(r"^(?P<prim>[a-z]+) (?P<nums>-?\d+(?: -?\d+){5})(?: (?P<axis>[+-]?[xy]))?(?: (?P<char>[A-Z0-9]))?$")


def parse_program(text):
    """The parts of a program's text form (program_text). Raises ValueError on anything else."""
    parts = []
    for group in text.split(";"):
        group = group.strip()
        if not group:
            continue
        if ":" not in group:
            raise ValueError(f"group without a label: {group!r}")
        label, body = group.split(":", 1)
        for item in body.split(","):
            match = PART_PATTERN.match(item.strip())
            if match is None or match["prim"] not in PRIMS:
                raise ValueError(f"not a part: {item.strip()!r}")
            nums = [int(v) for v in match["nums"].split()]
            if match["prim"] == "glyph" and (match["char"] is None or match["char"] not in GLYPHS):
                raise ValueError(f"a glyph needs a character: {item.strip()!r}")
            parts.append(Part(match["prim"], tuple(nums[:3]), tuple(nums[3:]), label.strip(), match["axis"] or "", match["char"] or ""))
    if not parts:
        raise ValueError("empty program")
    return tuple(parts)


def program_mask(text, dims=None):
    """(D, H, W) bool target of a program's text, in its own frame (or on a grid of dims)."""
    return to_grid(program_cells(normalise(parse_program(text))), dims)


# ---------------------------------------------------------------------------------------------
# Sub-goal sentences


def count_word(n):
    return NUMBER_WORDS.get(n, str(n))


def group_sentence(label, members):
    """A plain sentence for one group of parts: "four legs 3 tall", "a 3 by 3 seat on top"."""
    first = members[0]
    h, d, w = first.size
    n = len(members)
    alike = all(m.prim == first.prim and m.size == first.size for m in members)
    high_part = min(m.at[0] for m in members) > 0
    on_top = " on top" if high_part else ""
    lead = count_word(n) if n > 1 else "a"
    if alike and first.prim == "col":
        return f"{count_word(n) if n > 1 else 'a'} {label} {h} tall" + ("" if (d, w) == (1, 1) else f", {d} by {w}")
    if alike and first.prim in ("slab", "box") and n == 1:
        return f"a {d} by {w} {label}{on_top}" + ("" if h == 1 else f", {h} tall")
    if alike and first.prim in ("wall", "beam"):
        length = max(d, w)
        return f"{lead} {label} {length} long" + ("" if h == 1 else f" and {h} tall") + on_top
    if alike and first.prim == "glyph":
        return f"the {label} {first.char} standing {h} tall"
    if alike and n == 1:
        return f"a {label}: a {first.prim} {h} tall, {d} by {w}{on_top}"
    return f"the {label}{on_top}: " + ", ".join(f"a {m.prim} {m.size[0]} tall, {m.size[1]} by {m.size[2]}" for m in members)


def subgoal_sentences(parts):
    """One plain sentence per group of parts, in build order."""
    return [group_sentence(label, members) for label, members in groups(parts)]


# ---------------------------------------------------------------------------------------------
# Support: levers and build order

UP, DOWN = (1, 0, 0), (-1, 0, 0)
SIDEWAYS = ((0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _grounded(cell, walls):
    z, y, x = cell
    return z == 0 or (walls is not None and z - 1 < walls.shape[0] and bool(walls[z - 1, y, x]))


def levers(cells, walls=None, limit=CANTILEVER, hang=True):
    """{cell: lever} of the held cubes of a structure (cells: iterable of (z, y, x)): 0 on the
    floor or terrain (walls, (D, H, W) bool, the voxel below), the lever of the cube below for a
    cube resting on it, one more than a held neighbour's for a cube bonded beside it or (hang)
    hanging under it; only levers up to limit hold. Cubes missing from the result fall."""
    cellset = {tuple(int(v) for v in c) for c in cells}
    lever = {}
    queue = deque()
    for c in cellset:
        if _grounded(c, walls):
            lever[c] = 0
            queue.append(c)
    steps = [(UP, 0)] + [(s, 1) for s in SIDEWAYS] + ([(DOWN, 1)] if hang else [])
    while queue:
        c = queue.popleft()
        here = lever[c]
        for (dz, dy, dx), cost in steps:
            n = (c[0] + dz, c[1] + dy, c[2] + dx)
            if n not in cellset:
                continue
            value = here + cost
            if value <= limit and value < lever.get(n, limit + 1):
                lever[n] = value
                if cost == 0:
                    queue.appendleft(n)
                else:
                    queue.append(n)
    return lever


def stands(cells, walls=None, limit=CANTILEVER):
    """Whether every cube of the structure is held."""
    cells = [tuple(int(v) for v in c) for c in cells]
    return len(levers(cells, walls, limit)) == len(set(cells))


def placeable(cell, lever, walls=None, limit=CANTILEVER, hang=True):
    """The lever a cube would have if added to a structure whose held cubes have `lever`, or None."""
    if _grounded(cell, walls):
        return 0
    z, y, x = cell
    best = None
    below = lever.get((z - 1, y, x))
    if below is not None:
        best = below
    for dz, dy, dx in SIDEWAYS + ((UP,) if hang else ()):
        n = lever.get((z + dz, y + dy, x + dx))
        if n is not None and n + 1 <= limit and (best is None or n + 1 < best):
            best = n + 1
    return best


def build_order(groups_of_cells, walls=None, limit=CANTILEVER, base=()):
    """[[cells in placing order] per group] such that every partial structure stands (on top of
    `base`, cubes already standing), lowest first, or None if a group cannot be completed on the
    groups before it. groups_of_cells: [(n, 3) cells] in build order."""
    structure_cells = [tuple(int(v) for v in c) for c in base]
    lever = levers(structure_cells, walls, limit)
    out = []
    for group in groups_of_cells:
        remaining = {tuple(int(v) for v in c) for c in group} - set(structure_cells)
        order = []
        while remaining:
            options = []
            for c in remaining:
                value = placeable(c, lever, walls, limit)
                if value is not None:
                    options.append((c[0], value, c))
            if not options:
                return None
            _, _, c = min(options)
            remaining.discard(c)
            order.append(c)
            structure_cells.append(c)
            lever = levers(structure_cells, walls, limit)
        out.append(order)
    return out


def buildable(parts, limit=CANTILEVER):
    """Whether a program stands and can be built group by group in program order."""
    return build_order([cells for _, cells in group_cells(parts)], limit=limit) is not None


CLIMB_MARGIN = 2  # free cells around the structure that cubes arrive from and walk over


def reach(structure, rule="switch_cube"):
    """(D, H, W) bool: the voxels a lone cube can get to from the floor at the edge of the box,
    moving by the pivot rule (world.pivot_moves; switch_cube: bonds released) over `structure`
    (D, H, W) bool, nothing else in the way."""
    import torch

    from beebot.vla.voxel.world import DIRS, _at, _padded, pivot_moves

    cubes = torch.from_numpy(structure).unsqueeze(0)
    nothing = torch.zeros_like(cubes)
    paths = pivot_moves(cubes, nothing, nothing, release=rule == "switch_cube")[0]
    reachable = torch.zeros_like(cubes[0])
    reachable[0, 0, :] = reachable[0, -1, :] = reachable[0, :, 0] = reachable[0, :, -1] = True
    reachable &= ~cubes[0]
    codes = [c for c in range(len(DIRS)) if bool(paths[c].any())]
    while True:
        ahead = reachable.clone()
        for code in codes:
            moving = _padded((reachable & paths[code]).unsqueeze(0), False)
            back = tuple(-v for v in DIRS[code])
            ahead |= _at(moving, back)[0]
        ahead &= ~cubes[0]
        if bool((ahead == reachable).all()):
            return reachable.numpy()
        reachable = ahead


SWEEPS = tuple((major, sign_major, sign_minor) for major in (1, 2) for sign_major in (1, -1) for sign_minor in (1, -1))  # raster orders over a layer


def sweep_key(cell, sweep):
    major, sign_major, sign_minor = sweep
    minor = 3 - major
    return (cell[0], sign_major * cell[major], sign_minor * cell[minor])


def makes_slot(cell, remaining, structure, shift):
    """Whether placing cell leaves a voxel still to fill with placed cubes on two opposite sides
    in its layer (a slot no pivot can enter)."""
    z, y, x = cell
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        gap = (z, y + dy, x + dx)
        if gap not in remaining or gap == cell:
            continue
        far = np.array((z, y + 2 * dy, x + 2 * dx)) - shift
        if (far >= 0).all() and (far < np.array(structure.shape)).all() and structure[tuple(far)]:
            return True
    return False


def climb_order(groups_of_cells, rule="switch_cube", limit=CANTILEVER):
    """build_order with every cube also reachable when it is placed: a lone cube can walk to its
    voxel from the floor beside the structure by the pivot rule, over the cubes placed so far
    (reach). Each layer is swept as a raster (the first of SWEEPS that works), so a cube is never
    asked into a slot between two placed ones, which no pivot can enter. [[cells in order] per
    group] or None."""
    every = np.concatenate([np.asarray(g, int).reshape(-1, 3) for g in groups_of_cells])
    low = every.min(axis=0)
    shift = np.array([0, low[1] - CLIMB_MARGIN, low[2] - CLIMB_MARGIN])
    dims = tuple(int(v) for v in every.max(axis=0) - shift + np.array([3, CLIMB_MARGIN + 1, CLIMB_MARGIN + 1]))
    for sweep in SWEEPS:
        structure = np.zeros(dims, bool)
        structure_cells = []
        lever = {}
        out = []
        for group in groups_of_cells:
            remaining = {tuple(int(v) for v in c) for c in group} - set(structure_cells)
            order = []
            while remaining:
                options = sorted((sweep_key(c, sweep), c) for c in remaining if placeable(c, lever, None, limit) is not None)
                reachable = reach(structure, rule) if options else None
                reachable_options = [c for _, c in options if reachable[tuple(np.array(c) - shift)]]
                pick = next((c for c in reachable_options if not makes_slot(c, remaining, structure, shift)), reachable_options[0] if reachable_options else None)
                if pick is None:
                    break
                remaining.discard(pick)
                order.append(pick)
                structure_cells.append(pick)
                structure[tuple(np.array(pick) - shift)] = True
                lever = levers(structure_cells, None, limit)
            if remaining:
                break
            out.append(order)
        if len(out) == len(groups_of_cells):
            return out
    return None


def climbable(parts, rule="switch_cube"):
    """Whether a program can be built group by group with every cube walking to its place."""
    return climb_order([cells for _, cells in group_cells(parts)], rule) is not None


# ---------------------------------------------------------------------------------------------
# Functions: what a structure is good for, on an occupancy grid (D, H, W) bool


def held_mask(occupancy, walls=None):
    """(D, H, W) bool: occupied voxels face-connected to the floor or terrain (the world's rule)."""
    hold = np.zeros_like(occupancy)
    hold[0] = occupancy[0]
    if walls is not None:
        hold[1:] |= occupancy[1:] & walls[:-1]
    while True:
        step_mask = hold.copy()
        step_mask[1:] |= hold[:-1]
        step_mask[:-1] |= hold[1:]
        step_mask[:, 1:] |= hold[:, :-1]
        step_mask[:, :-1] |= hold[:, 1:]
        step_mask[:, :, 1:] |= hold[:, :, :-1]
        step_mask[:, :, :-1] |= hold[:, :, 1:]
        step_mask &= occupancy
        if (step_mask == hold).all():
            return hold
        hold = step_mask


def _windows(layer):
    """(H - 1, W - 1) int: how many of each 2 x 2 window's cells are set."""
    a = layer.astype(int)
    return a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:]


def _top(occupancy, hold, z):
    above = occupancy[z + 1] if z + 1 < occupancy.shape[0] else np.zeros_like(occupancy[z])
    return hold[z] & ~above


def sit(occupancy, walls=None):
    """A supported flat surface of at least 2 x 2 at height 2 to 4 (its cubes' tops two to four
    cubes above the floor: z 1 to 3) with nothing on it: the share of the best 2 x 2 window."""
    hold = held_mask(occupancy, walls)
    best = 0
    for z in range(1, min(4, occupancy.shape[0])):
        best = max(best, int(_windows(_top(occupancy, hold, z)).max(initial=0)))
    return best / 4


def table(occupancy, walls=None):
    """A supported flat top at height 3 or more (z >= 2): a clear 2 x 2 window of it with open
    space under at least half of its cells. 1.0 when found, else the best share of such a window."""
    hold = held_mask(occupancy, walls)
    best = 0.0
    for z in range(2, occupancy.shape[0]):
        top = _top(occupancy, hold, z)
        if not top.any():
            continue
        under = ~occupancy[z - 1] & (~walls[z - 1] if walls is not None else True)
        both = _windows(top) == 4
        if (both & (_windows(top & under) >= 2)).any():
            return 1.0
        best = max(best, _windows(top).max(initial=0) / 4 * 0.5)
    return best


def _roofed(occupancy):
    """(D, H, W) bool: voxels with an occupied voxel somewhere above them in their column."""
    above = np.zeros_like(occupancy)
    for z in range(occupancy.shape[0] - 2, -1, -1):
        above[z] = above[z + 1] | occupancy[z + 1]
    return above


def _walled(occupancy, axis, sign, reach=8):
    """(D, H, W) bool: an occupied voxel within reach along a floor axis (1 or 2) and sign."""
    out = np.zeros_like(occupancy)
    moving = occupancy.copy()
    for _ in range(reach):
        moving = np.roll(moving, -sign, axis=axis)
        edge = [slice(None)] * 3
        edge[axis] = slice(-1, None) if sign > 0 else slice(0, 1)
        moving[tuple(edge)] = False
        out |= moving
    return out


SHELTER_REACH = 3  # a shelter's walls stand within this many cells of the space they keep


def shelter(occupancy, walls=None, need=2):
    """Roofed space: empty voxels with structure above them and, within SHELTER_REACH cells at
    their height, on at least two of their four sides; min(1, count / need)."""
    free = ~occupancy & (~walls if walls is not None else True)
    sides = sum(_walled(occupancy, axis, sign, SHELTER_REACH).astype(int) for axis in (1, 2) for sign in (-1, 1))
    count = int((free & _roofed(occupancy) & (sides >= 2)).sum())
    return min(1.0, count / need)


def span(occupancy, walls=None):
    """A straight horizontal run (z >= 1) with two cells carried by unbroken columns down to the
    floor and, between them, a cell with open space all the way down: 1.0 or 0.0."""
    solid = occupancy | (walls if walls is not None else False)
    column = np.cumprod(solid, axis=0).astype(bool)  # solid from the floor up to here
    below = np.zeros(solid.shape, int)
    below[1:] = np.cumsum(solid, axis=0)[:-1]  # solid voxels under each voxel
    prop_grid = np.zeros_like(occupancy)
    prop_grid[1:] = column[:-1]
    for z in range(1, occupancy.shape[0]):
        for layer, hold, gap in ((occupancy[z], prop_grid[z], below[z] == 0), (occupancy[z].T, prop_grid[z].T, (below[z] == 0).T)):
            for row, hold_row, gap_row in zip(layer, hold, gap):
                start = None
                for j in range(len(row) + 1):
                    if j < len(row) and row[j]:
                        start = j if start is None else start
                        continue
                    if start is not None:
                        props = np.nonzero(hold_row[start:j])[0]
                        if len(props) >= 2 and gap_row[start:j][props[0] : props[-1]].any():
                            return 1.0
                        start = None
    return 0.0


def enclosed(layer):
    """(H, W) bool: empty cells of a layer that no path of empty cells (four-connected) joins to
    the layer's edge."""
    outside = np.zeros_like(layer)
    outside[0], outside[-1], outside[:, 0], outside[:, -1] = True, True, True, True
    outside &= ~layer
    while True:
        step_mask = outside.copy()
        step_mask[1:] |= outside[:-1]
        step_mask[:-1] |= outside[1:]
        step_mask[:, 1:] |= outside[:, :-1]
        step_mask[:, :-1] |= outside[:, 1:]
        step_mask &= ~layer
        if (step_mask == outside).all():
            return ~layer & ~outside
        outside = step_mask


def contain(occupancy, walls=None, need=1):
    """A cavity open at the top: empty voxels walled in all round within their layer, resting on
    a solid voxel (a bottom of its own: the bare floor is not one), with nothing above them; min(1, count / need)."""
    solid = occupancy | (walls if walls is not None else False)
    bottom = np.zeros_like(occupancy)
    bottom[1:] = solid[:-1]
    inside = np.stack([enclosed(layer) for layer in solid])
    count = int((inside & bottom & ~_roofed(occupancy)).sum())
    return min(1.0, count / need)


FUNCTIONS = {"sit": sit, "table": table, "shelter": shelter, "span": span, "contain": contain}
FUNCTION_PASS = 1.0
USES = {"sit": "you can sit on", "table": "you can put things on", "shelter": "you can shelter in", "span": "that spans a gap between two supports", "contain": "that can hold things"}  # canonical words: "build something you can sit on"


# ---------------------------------------------------------------------------------------------
# Overlap with a target, robust to where and which way round it was built


def yaw_cells(cells):
    """The four quarter turns of cells (n, 3) about the vertical, each moved to the origin."""
    out = []
    grid = to_grid(cells)
    for k in range(4):
        out.append(np.argwhere(np.rot90(grid, -k, axes=(1, 2))))
    return out


def placement_iou(occupancy, cells, margin=1):
    """(best IoU, (dz, dy, dx)) of the target cells (n, 3, at the origin) moved anywhere on the
    grid: hits / (target + robots inside the target's box grown by margin across the floor and
    one cell up - hits). Robots far from the target do not count, so idle robots elsewhere cost
    nothing."""
    occ = occupancy.astype(np.int32)
    dims = np.array(occ.shape)
    extent = cells.max(axis=0) + 1
    shifts = dims - extent + 1
    if (shifts <= 0).any():
        return 0.0, (0, 0, 0)
    nz, ny, nx = shifts
    hits = np.zeros((nz, ny, nx), np.int32)
    for z, y, x in cells.tolist():
        hits += occ[z : z + nz, y : y + ny, x : x + nx]
    table = np.zeros(tuple(dims + 1), np.int64)
    table[1:, 1:, 1:] = occ.cumsum(0).cumsum(1).cumsum(2)
    sz, sy, sx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    z0, z1 = sz, np.minimum(sz + extent[0] + 1, dims[0])
    y0, y1 = np.maximum(sy - margin, 0), np.minimum(sy + extent[1] + margin, dims[1])
    x0, x1 = np.maximum(sx - margin, 0), np.minimum(sx + extent[2] + margin, dims[2])
    inside = table[z1, y1, x1] - table[z0, y1, x1] - table[z1, y0, x1] - table[z1, y1, x0] + table[z0, y0, x1] + table[z0, y1, x0] + table[z1, y0, x0] - table[z0, y0, x0]
    iou = hits / (len(cells) + inside - hits)
    best = np.unravel_index(int(np.argmax(iou)), iou.shape)
    return float(iou[best]), tuple(int(v) for v in best)


def object_iou(occupancy, cells, turns=True):
    """The best placement_iou of the target over its four quarter turns (or as given)."""
    cells = np.asarray(cells, int)
    cells = cells - cells.min(axis=0)
    options = yaw_cells(cells) if turns else [cells]
    return max(placement_iou(occupancy, c)[0] for c in options)


# ---------------------------------------------------------------------------------------------
# Generators. Each takes a numpy Generator and returns parts in build order.


def _ri(rng, low, high):
    return int(rng.integers(low, high + 1))


def P(prim, at, size, label, axis="", char=""):
    return Part(prim, tuple(int(v) for v in at), tuple(int(v) for v in size), label, axis, char)


def spaced(length, gap):
    """Positions 0 .. length - 1 with the ends and no two neighbours more than gap apart."""
    if length <= 1:
        return [0]
    count = int(np.ceil((length - 1) / gap)) + 1
    return sorted({int(round(k * (length - 1) / (count - 1))) for k in range(count)})


def legs_under(d, w, h, rng, label="legs", y0=0, x0=0):
    """Column legs h tall under a d x w top: the corners, plus more along the edges at a random
    spacing (2 to 4), so some draws overhang too far and fail the support rule."""
    gap = _ri(rng, 2, 5)
    ys, xs = spaced(d, gap), spaced(w, gap)
    cells = sorted({(y, x) for y in ys for x in xs if y in (ys[0], ys[-1]) or x in (xs[0], xs[-1])})
    return [P("col", (0, y0 + y, x0 + x), (h, 1, 1), label) for y, x in cells]


def gen_stool(rng):
    s, h = _ri(rng, 2, 4), _ri(rng, 1, 3)
    return [P("col", (0, y, x), (h, 1, 1), "legs") for y in (0, s - 1) for x in (0, s - 1)] + [P("slab", (h, 0, 0), (1, s, s), "seat")]


def gen_chair(rng):
    d, w, h, b = _ri(rng, 3, 4), _ri(rng, 2, 4), _ri(rng, 1, 3), _ri(rng, 2, 4)  # 3 deep: a 2 x 2 seat in front of the back
    return [P("col", (0, y, x), (h, 1, 1), "legs") for y in (0, d - 1) for x in (0, w - 1)] + [P("slab", (h, 0, 0), (1, d, w), "seat"), P("wall", (h + 1, 0, 0), (b, 1, w), "back")]


def gen_throne(rng):
    d, w, h, b = _ri(rng, 3, 4), _ri(rng, 3, 5), _ri(rng, 1, 2), _ri(rng, 4, 6)
    return [P("col", (0, y, x), (h, 1, 1), "legs") for y in (0, d - 1) for x in (0, w - 1)] + [P("slab", (h, 0, 0), (1, d, w), "seat"), P("wall", (h + 1, 0, 0), (b, 1, w), "back")] + [P("col", (h + 1, 1, x), (1, d - 1, 1), "arms") for x in (0, w - 1)]


def gen_table(rng):
    d, w, h = _ri(rng, 3, 6), _ri(rng, 3, 7), _ri(rng, 2, 4)
    return legs_under(d, w, h, rng) + [P("slab", (h, 0, 0), (1, d, w), "top")]


def gen_bench(rng):
    back = rng.random() < 0.5
    d, w, h = 3 if back else 2, _ri(rng, 4, 8), _ri(rng, 1, 2)  # a seat 2 deep in front of the back
    xs = spaced(w, _ri(rng, 3, 5))
    parts = [P("wall", (0, 0, x), (h, d, 1), "legs") for x in xs] + [P("slab", (h, 0, 0), (1, d, w), "seat")]
    if back:
        parts.append(P("wall", (h + 1, 0, 0), (_ri(rng, 1, 2), 1, w), "back"))
    return parts


def gen_sofa(rng):
    d, w = _ri(rng, 3, 4), _ri(rng, 4, 7)
    return [P("box", (0, 0, 0), (2, d, w), "base"), P("wall", (2, 0, 0), (_ri(rng, 1, 2), 1, w), "back"), P("col", (2, 1, 0), (1, d - 1, 1), "arms"), P("col", (2, 1, w - 1), (1, d - 1, 1), "arms")]


def gen_bed(rng):
    d, w = _ri(rng, 3, 4), _ri(rng, 5, 8)
    return legs_under(d, w, 1, rng) + [P("slab", (1, 0, 0), (1, d, w), "mattress"), P("wall", (2, 0, 0), (_ri(rng, 1, 3), d, 1), "headboard")]


def gen_shelf(rng):
    d, w, h = _ri(rng, 1, 2), _ri(rng, 3, 7), _ri(rng, 4, 8)
    gap = _ri(rng, 2, 3)
    return [P("wall", (0, 0, 0), (h, d, 1), "left side")] + [P("slab", (z, 0, 1), (1, d, w - 2), "shelves") for z in range(0, h, gap)] + [P("wall", (0, 0, w - 1), (h, d, 1), "right side")]


def gen_arch(rng):
    span_, h, t = _ri(rng, 3, 8), _ri(rng, 2, 6), _ri(rng, 1, 2)
    return [P("arch", (0, 0, 0), (h, t, span_), "arch", "x")]


def gen_bridge(rng):
    h, length, d = _ri(rng, 1, 4), _ri(rng, 5, 10), _ri(rng, 1, 3)
    parts = [P("wall", (0, 0, x), (h, d, 1), "piers") for x in (0, length - 1)] + [P("slab", (h, 0, 0), (1, d, length), "deck")]
    if d >= 2 and rng.random() < 0.5:
        parts += [P("beam", (h + 1, y, 0), (1, 1, length), "railings") for y in (0, d - 1)]
    return parts


def gen_gate(rng):
    h, gap = _ri(rng, 3, 5), _ri(rng, 2, 5)
    parts = [P("col", (0, 0, x), (h, 1, 1), "posts") for x in (0, gap + 1)] + [P("beam", (h, 0, 0), (1, 1, gap + 2), "top bar")]
    if h >= 4 and rng.random() < 0.5:
        parts.append(P("beam", (h // 2, 0, 1), (1, 1, gap), "middle bar"))
    return parts


def gen_door_wall(rng):
    length, h = _ri(rng, 5, 9), _ri(rng, 3, 5)
    door = _ri(rng, 1, 2)
    tall = _ri(rng, 2, h - 1)
    remaining = _ri(rng, 1, length - door - 1)
    return [P("wall", (0, 0, 0), (h, 1, remaining), "left wall"), P("wall", (tall, 0, remaining), (h - tall, 1, door), "lintel"), P("wall", (0, 0, remaining + door), (h, 1, length - remaining - door), "right wall")]  # the lintel before the wall it meets: no cube can enter a slot


def gen_fence(rng):
    h, length = _ri(rng, 2, 3), _ri(rng, 5, 11)
    xs = spaced(length, _ri(rng, 2, 4))
    return [P("col", (0, 0, x), (h, 1, 1), "posts") for x in xs] + [P("beam", (h, 0, 0), (1, 1, length), "rail")]


def gen_castle_tower(rng):
    s, h = _ri(rng, 3, 5), _ri(rng, 3, 7)
    ring = P("ring", (0, 0, 0), (h, s, s), "walls")
    battlements = [P("col", (h, y, x), (1, 1, 1), "battlements") for y in range(s) for x in range(s) if (y in (0, s - 1) or x in (0, s - 1)) and (y + x) % 2 == 0]
    return [ring] + battlements


def gen_staircase(rng):
    steps, d = _ri(rng, 3, 6), _ri(rng, 1, 3)
    return [P("ramp", (0, 0, 0), (steps, d, steps), "steps", "+x")]


def gen_pyramid(rng):
    b = _ri(rng, 3, 6)
    return [P("pyramid", (0, 0, 0), ((b + 1) // 2, b, b), "pyramid")]


def gen_ladder(rng):
    h, w = _ri(rng, 4, 8), _ri(rng, 3, 4)
    return [P("col", (0, 0, 0), (h, 1, 1), "left rail")] + [P("beam", (z, 0, 1), (1, 1, w - 2), "rungs") for z in range(1, h, 2)] + [P("col", (0, 0, w - 1), (h, 1, 1), "right rail")]


def gen_obelisk(rng):
    s, h = (1, 3)[_ri(rng, 0, 1)], _ri(rng, 4, 8)
    return [P("slab", (0, 0, 0), (1, s + 2, s + 2), "base"), P("col", (1, 1, 1), (h, s, s), "shaft"), P("pyramid", (h + 1, 1, 1), ((s + 1) // 2, s, s), "tip")]


def gen_podium(rng):
    d, w = _ri(rng, 1, 3), _ri(rng, 1, 3)
    return [P("box", (0, 0, w), (3, d, w), "first place block"), P("box", (0, 0, 0), (2, d, w), "second place block"), P("box", (0, 0, 2 * w), (1, d, w), "third place block")]


def gen_slide(rng):
    h, d, length = _ri(rng, 3, 5), _ri(rng, 1, 2), _ri(rng, 3, 6)
    return [P("col", (0, 0, 0), (h, d, 1), "tower"), P("ramp", (0, 0, 1), (h - 1, d, length), "chute", "-x")]


def gen_lamp_post(rng):
    h, arm = _ri(rng, 4, 7), _ri(rng, 1, 3)
    return [P("col", (0, 0, 0), (h, 1, 1), "post"), P("beam", (h - 1, 0, 1), (1, 1, arm), "arm"), P("col", (h - 2, 0, arm), (1, 1, 1), "lamp")]


def gen_house(rng):
    d, w, h = _ri(rng, 4, 7), _ri(rng, 4, 7), _ri(rng, 2, 3)
    return [P("ring", (0, 0, 0), (h, d, w), "walls"), P("roof", (h, 0, 0), ((d + 1) // 2, d, w), "roof", "x")]


def gen_doghouse(rng):
    d, w, h = _ri(rng, 3, 4), _ri(rng, 3, 4), 2
    return [P("ring", (0, 0, 0), (h, d, w), "walls"), P("roof", (h, 0, 0), ((d + 1) // 2, d, w), "roof", "x")]


def gen_hut(rng):
    d, w, h = _ri(rng, 3, 6), _ri(rng, 3, 6), _ri(rng, 2, 3)
    return [P("ring", (0, 0, 0), (h, d, w), "walls"), P("slab", (h, 0, 0), (1, d, w), "roof")]


def gen_igloo(rng):
    s, h = _ri(rng, 4, 7), _ri(rng, 2, 4)
    parts = [P("dome", (0, 0, 0), (h, s, s), "dome")]
    if rng.random() < 0.6:
        mid = (s - 3) // 2
        parts.append(P("arch", (0, s, mid), (2, 2, 3), "entrance", "x"))
    return parts


def gen_tent(rng):
    d, w = _ri(rng, 5, 7), _ri(rng, 3, 7)  # 5 wide or more: room under the ridge
    return [P("roof", (0, 0, 0), ((d + 1) // 2, d, w), "tent", "x")]


def gen_lighthouse(rng):
    h = _ri(rng, 4, 8)
    return [P("ring", (0, 1, 1), (h, 3, 3), "tower"), P("ring", (h - 1, 0, 0), (1, 5, 5), "gallery"), P("slab", (h, 1, 1), (1, 3, 3), "lamp room")]


def gen_car(rng):
    w, d = _ri(rng, 4, 6), _ri(rng, 2, 3)
    cabin = _ri(rng, 2, w - 2)
    return [P("col", (0, y, x), (1, 1, 1), "wheels") for y in (0, d - 1) for x in (0, w - 1)] + [P("slab", (1, 0, 0), (1, d, w), "body"), P("box", (2, 0, (w - cabin) // 2), (1, d, cabin), "cabin")]


def gen_truck(rng):
    w, d = _ri(rng, 6, 9), _ri(rng, 2, 3)
    xs = spaced(w, _ri(rng, 2, 3))
    return [P("col", (0, y, x), (1, 1, 1), "wheels") for y in (0, d - 1) for x in xs] + [P("slab", (1, 0, 0), (1, d, w), "chassis"), P("box", (2, 0, w - 2), (2, d, 2), "cab"), P("ring", (2, 0, 0), (_ri(rng, 1, 2), d + (d < 3), w - 2), "cargo box")]


def gen_train(rng):
    cars, d = _ri(rng, 1, 2), _ri(rng, 1, 2)
    parts = []
    for k in range(cars):
        x0 = k * 5
        parts += [P("col", (0, 0, x0 + x), (1, d, 1), "wheels") for x in (0, 3)]
    parts += [P("box", (1, 0, k * 5), (2, d, 4), "carriages") for k in range(cars)]
    parts += [P("col", (3, 0, 1), (_ri(rng, 1, 2), 1, 1), "chimney")]
    return parts


def gen_plane(rng):
    length, half = _ri(rng, 5, 8), _ri(rng, 1, 3)
    wing = _ri(rng, 1, length - 3)
    return [P("col", (0, half, x), (1, 1, 1), "landing gear") for x in (wing, length - 1)] + [P("beam", (1, half, 0), (1, 1, length), "fuselage"), P("slab", (1, 0, wing), (1, 2 * half + 1, 2), "wings"), P("col", (2, half, length - 1), (_ri(rng, 1, 2), 1, 1), "tail")]


def gen_boat(rng):
    d, w = _ri(rng, 3, 4), _ri(rng, 5, 8)
    mast = _ri(rng, 3, 6)
    c, m = d // 2, w // 2
    return [P("slab", (0, 0, 0), (1, d, w), "hull"), P("ring", (1, 0, 0), (1, d, w), "sides"), P("col", (1, c, m), (mast, 1, 1), "mast"), P("wall", (2, c, m + 1), (mast - 1, 1, _ri(rng, 1, 3)), "sail")]


def gen_rocket(rng):
    h, s = _ri(rng, 4, 8), (1, 3)[_ri(rng, 0, 1)]
    o = 1
    return [P("col", (0, o, o), (h, s, s), "body")] + [P("col", (0, y, x), (2, 1, 1), "fins") for y, x in ((o + s // 2, 0), (o + s // 2, o + s), (0, o + s // 2), (o + s, o + s // 2))] + [P("pyramid", (h, o, o), ((s + 1) // 2, s, s), "nose")]


def gen_cup(rng):
    s, h = _ri(rng, 3, 5), _ri(rng, 1, 3)
    return [P("slab", (0, 0, 0), (1, s, s), "base"), P("ring", (1, 0, 0), (h, s, s), "wall")]


def gen_pot(rng):
    s, h = _ri(rng, 4, 6), _ri(rng, 1, 3)
    return [P("disc", (0, 0, 0), (1, s, s), "base"), P("tube", (1, 0, 0), (h, s, s), "wall")]


def gen_crate(rng):
    d, w, h = _ri(rng, 3, 6), _ri(rng, 3, 6), _ri(rng, 1, 3)
    return [P("slab", (0, 0, 0), (1, d, w), "bottom"), P("ring", (1, 0, 0), (h, d, w), "sides")]


def gen_tree(rng):
    h, s = _ri(rng, 2, 5), (3, 5)[_ri(rng, 0, 1)]
    c = s // 2
    return [P("col", (0, c, c), (h, 1, 1), "trunk"), P("disc", (h, 0, 0), (_ri(rng, 1, 3), s, s), "crown")]


def gen_pine_tree(rng):
    h = _ri(rng, 1, 3)
    return [P("col", (0, 2, 2), (h, 1, 1), "trunk"), P("disc", (h, 0, 0), (1, 5, 5), "lower branches"), P("disc", (h + 1, 1, 1), (1, 3, 3), "upper branches"), P("col", (h + 2, 2, 2), (_ri(rng, 1, 2), 1, 1), "top")]


def gen_cactus(rng):
    h = _ri(rng, 4, 7)
    parts = [P("col", (0, 0, 2), (h, 1, 1), "stem")]
    for side in (-1, 1):
        if rng.random() < 0.8:
            z = _ri(rng, 1, h - 2)
            reach = _ri(rng, 1, 2)
            x = 2 + side * reach if side > 0 else 2 - reach
            parts.append(P("beam", (z, 0, min(x, 2 + side)), (1, 1, reach), "arms"))
            parts.append(P("col", (z + 1, 0, 2 + side * reach), (_ri(rng, 1, 2), 1, 1), "arms"))
    return parts


def gen_mushroom(rng):
    h, s = _ri(rng, 2, 3), (3, 5)[_ri(rng, 0, 1)]
    c = s // 2
    return [P("col", (0, c, c), (h, 1, 1), "stem"), P("disc", (h, 0, 0), (1, s, s), "cap")]


def gen_cross(rng):
    h, arm = _ri(rng, 4, 7), _ri(rng, 1, 3)
    return [P("col", (0, 0, arm), (h, 1, 1), "post"), P("beam", (h - _ri(rng, 2, 3), 0, 0), (1, 1, 2 * arm + 1), "crossbar")]


def gen_standing_ring(rng):
    h, w, t = _ri(rng, 3, 7), _ri(rng, 3, 7), _ri(rng, 1, 2)
    return [P("ring", (0, 0, 0), (h, t, w), "ring", "x")]


def gen_snowman(rng):
    head = _ri(rng, 1, 2)
    return [P("disc", (0, 0, 0), (2, 5, 5), "bottom ball"), P("disc", (2, 1, 1), (2, 3, 3), "middle ball"), P("col", (4, 2, 2), (head, 1, 1), "head")]


def gen_dog(rng):
    h, w = _ri(rng, 1, 2), _ri(rng, 3, 5)
    return [P("col", (0, y, x), (h, 1, 1), "legs") for y in (0, 1) for x in (0, w - 1)] + [P("box", (h, 0, 0), (1, 2, w), "body"), P("col", (h + 1, 0, w - 1), (1, 2, 1), "neck"), P("box", (h + 2, 0, w - 1), (1, 2, 2), "head"), P("col", (h + 1, 0, 0), (1, 1, 1), "tail")]


def gen_giraffe(rng):
    h, w, neck = 3, _ri(rng, 3, 4), _ri(rng, 3, 5)
    return [P("col", (0, y, x), (h, 1, 1), "legs") for y in (0, 1) for x in (0, w - 1)] + [P("box", (h, 0, 0), (1, 2, w), "body"), P("col", (h + 1, 0, w - 1), (neck, 1, 1), "neck"), P("beam", (h + neck, 0, w), (1, 1, 2), "head")]


def gen_glyph(chars):
    def make(rng):
        char = str(rng.choice(list(chars)))
        t = _ri(rng, 1, 2)
        return [P("glyph", (0, 0, 0), glyph_size(char, t), "letter" if char.isalpha() else "number", "+x", char)]

    return make


@dataclass(frozen=True)
class Entry:
    name: str
    category: str
    make: object
    functions: tuple = ()
    held_out: bool = False
    article: str = "a"


LETTERS = tuple(FONT)
NUMBERS = tuple(DIGITS)
HELD_GLYPHS = ("K", "Q", "Z", "4", "7")  # the suite's held-out letters, and two digits

LIBRARY = {}


def _add(name, category, make, functions=(), held_out=False, article="a"):
    LIBRARY[name] = Entry(name, category, make, tuple(functions), held_out, article)


_add("stool", "furniture", gen_stool, ("sit",))
_add("chair", "furniture", gen_chair, ("sit",))
_add("throne", "furniture", gen_throne, ("sit",), held_out=True)
_add("table", "furniture", gen_table, ("table",))
_add("bench", "furniture", gen_bench, ("sit",))
_add("sofa", "furniture", gen_sofa, ("sit",))
_add("bed", "furniture", gen_bed, ("sit",))
_add("shelf", "furniture", gen_shelf)
_add("arch", "structure", gen_arch, ("span",), article="an")
_add("bridge", "structure", gen_bridge, ("span",))
_add("gate", "structure", gen_gate)
_add("wall with a door", "structure", gen_door_wall)
_add("fence", "structure", gen_fence)
_add("castle tower", "structure", gen_castle_tower)
_add("staircase", "structure", gen_staircase, held_out=True)
_add("pyramid", "structure", gen_pyramid, held_out=True)
_add("ladder", "structure", gen_ladder)
_add("obelisk", "structure", gen_obelisk, article="an")
_add("podium", "structure", gen_podium)
_add("slide", "structure", gen_slide)
_add("lamp post", "structure", gen_lamp_post)
_add("house", "building", gen_house, ("shelter",))
_add("hut", "building", gen_hut, ("shelter",))
_add("doghouse", "building", gen_doghouse, ("shelter",), held_out=True)
_add("igloo", "building", gen_igloo, ("shelter",), article="an")
_add("tent", "building", gen_tent, ("shelter",))
_add("lighthouse", "building", gen_lighthouse)
_add("car", "vehicle", gen_car, held_out=True)
_add("truck", "vehicle", gen_truck, held_out=True)
_add("train", "vehicle", gen_train, held_out=True)
_add("plane", "vehicle", gen_plane, held_out=True)
_add("boat", "vehicle", gen_boat, ("contain",), held_out=True)
_add("rocket", "vehicle", gen_rocket, held_out=True)
_add("cup", "container", gen_cup, ("contain",))
_add("pot", "container", gen_pot, ("contain",))
_add("crate", "container", gen_crate, ("contain",))
_add("tree", "plant", gen_tree)
_add("pine tree", "plant", gen_pine_tree)
_add("cactus", "plant", gen_cactus, held_out=True)
_add("mushroom", "plant", gen_mushroom)
_add("cross", "symbol", gen_cross)
_add("standing ring", "symbol", gen_standing_ring)
_add("snowman", "figure", gen_snowman)
_add("dog", "animal", gen_dog)
_add("giraffe", "animal", gen_giraffe)
for _char in LETTERS:
    _add(f"letter {_char}", "letter", gen_glyph(_char), held_out=_char in HELD_GLYPHS, article="the")
for _char in NUMBERS:
    _add(f"number {_char}", "number", gen_glyph(_char), held_out=_char in HELD_GLYPHS, article="the")

HELD_CATEGORIES = ("vehicle",)  # whole categories held out; every one of their primitives is trained elsewhere
TRAIN_NAMES = tuple(name for name, entry in LIBRARY.items() if not entry.held_out)
HELD_NAMES = tuple(name for name, entry in LIBRARY.items() if entry.held_out)


def phrase(name):
    """The object as the canonical sentence says it: "a chair", "an arch", "the letter A"."""
    return f"{LIBRARY[name].article} {name}"


def variant(name, rng):
    """One draw of a named object (its parts normalised to the origin)."""
    entry = LIBRARY[name]
    return Shape(name, entry.category, normalise(entry.make(rng)), entry.functions)


def fits_grid(cells):
    extent = cells.max(axis=0) + 1
    return extent[0] <= HEIGHT and extent[1] <= FOOTPRINT and extent[2] <= FOOTPRINT


def fit_variant(name, rng, robots, tries=24, low=0.4, climb=True, checks=4):
    """A draw of the object that stands, can be built in program order, fits the grid and needs
    at most `robots` cubes, as close as the tries get to a size drawn uniformly between low and
    1.0 times the swarm; with climb, also climbable (cubes can walk to every voxel as it is
    placed; the `checks` closest are tried in turn). None when none fits."""
    want = rng.uniform(low, 1.0) * robots
    options = []
    for k in range(tries):
        shape = variant(name, rng)
        cells = shape.cells()
        if len(cells) > robots or not fits_grid(cells) or not buildable(shape.parts):
            continue
        options.append((abs(len(cells) - want), k, shape))
    for _, _, shape in sorted(options, key=lambda item: item[:2])[: checks if climb else 1]:
        if not climb or climbable(shape.parts):
            return shape
    return None


def min_size(name, draws=24, seed=0, checks=6):
    """The fewest cubes any of `draws` draws of the object needs that is buildable and climbable
    (the `checks` smallest buildable ones are tried for climbing); inf if none is."""
    rng = np.random.default_rng(seed)
    sizes = []
    for _ in range(draws):
        shape = variant(name, rng)
        if fits_grid(shape.cells()) and buildable(shape.parts):
            sizes.append((len(shape.cells()), shape))
    for size, shape in sorted(sizes, key=lambda item: item[0])[:checks]:
        if climbable(shape.parts):
            return size
    return float("inf")


def survey(draws=64, seed=0):
    """{name: {"category", "held_out", "draws", "stand" (share whose full structure stands under
    the support rule), "buildable" (share that also fit the grid and build group by group in
    program order), "climbable" (share that also let every cube walk to its voxel), "sizes" [min,
    median, max] of the climbable draws}} over `draws` draws per object."""
    rng = np.random.default_rng(seed)
    out = {}
    for name, entry in LIBRARY.items():
        stand_count, ok, climbing, sizes = 0, 0, 0, []
        for _ in range(draws):
            shape = variant(name, rng)
            cells = shape.cells()
            stand_count += stands(cells)
            if fits_grid(cells) and buildable(shape.parts):
                ok += 1
                if climbable(shape.parts):
                    climbing += 1
                    sizes.append(len(cells))
        out[name] = {"category": entry.category, "held_out": entry.held_out, "draws": draws, "stand": stand_count / draws, "buildable": ok / draws, "climbable": climbing / draws, "sizes": [int(min(sizes)), int(np.median(sizes)), int(max(sizes))] if sizes else []}
    return out


# ---------------------------------------------------------------------------------------------
# Meshes: an interface for an external dataset (nothing is downloaded here)


def read_off(path):
    """(vertices (V, 3) float, faces (F, 3) int) of an OFF file (ModelNet's format); polygons are
    fanned into triangles."""
    tokens = []
    for line in open(path):
        line = line.split("#")[0].strip()
        if line:
            tokens.append(line)
    head = tokens[0]
    if head.startswith("OFF") and len(head) > 3:  # some ModelNet files run "OFF" into the counts
        tokens[0] = head[3:]
    else:
        tokens = tokens[1:]
    nv, nf = (int(v) for v in tokens[0].split()[:2])
    vertices = np.array([[float(v) for v in tokens[1 + k].split()[:3]] for k in range(nv)])
    faces = []
    for k in range(nf):
        values = [int(v) for v in tokens[1 + nv + k].split()]
        ring = values[1 : 1 + values[0]]
        faces += [(ring[0], ring[j], ring[j + 1]) for j in range(1, len(ring) - 1)]
    return vertices, np.array(faces, int)


def read_obj(path):
    """(vertices, faces) of a Wavefront OBJ file (v and f lines only)."""
    vertices, faces = [], []
    for line in open(path):
        words = line.split()
        if not words:
            continue
        if words[0] == "v":
            vertices.append([float(v) for v in words[1:4]])
        elif words[0] == "f":
            ring = [int(w.split("/")[0]) - 1 for w in words[1:]]
            faces += [(ring[0], ring[j], ring[j + 1]) for j in range(1, len(ring) - 1)]
    return np.array(vertices), np.array(faces, int)


def voxelise(vertices, faces, resolution, up=1, fill=True, samples=12):
    """(D, H, W) bool occupancy of a triangle mesh scaled so its longest side spans `resolution`
    cells, z up (`up`: which mesh axis is up; ModelNet and ShapeNet use 1, y up). The surface is
    sampled densely on every triangle; fill closes the interior (flood fill from outside)."""
    v = np.asarray(vertices, float)
    order = {0: (0, 1, 2), 1: (1, 2, 0), 2: (2, 0, 1)}[up]
    v = v[:, order]  # (up, depth, width)
    v = v - v.min(axis=0)
    scale = (resolution - 1e-6) / max(v.max(), 1e-9)
    v = v * scale
    dims = tuple(int(np.floor(m)) + 1 for m in v.max(axis=0))
    grid = np.zeros(dims, bool)
    tri = v[np.asarray(faces, int)]  # (F, 3, 3)
    a, b = np.meshgrid(np.linspace(0, 1, samples), np.linspace(0, 1, samples))
    keep = a + b <= 1
    a, b = a[keep], b[keep]
    points = tri[:, None, 0] + a[None, :, None] * (tri[:, None, 1] - tri[:, None, 0]) + b[None, :, None] * (tri[:, None, 2] - tri[:, None, 0])
    cells = np.clip(np.floor(points.reshape(-1, 3)).astype(int), 0, np.array(dims) - 1)
    grid[tuple(cells.T)] = True
    if fill:
        outside = np.zeros_like(grid)
        pad_grid = np.pad(grid, 1)
        reach = np.zeros_like(pad_grid)
        reach[0], reach[-1], reach[:, 0], reach[:, -1], reach[:, :, 0], reach[:, :, -1] = True, True, True, True, True, True
        reach &= ~pad_grid
        while True:
            step_mask = reach.copy()
            step_mask[1:] |= reach[:-1]
            step_mask[:-1] |= reach[1:]
            step_mask[:, 1:] |= reach[:, :-1]
            step_mask[:, :-1] |= reach[:, 1:]
            step_mask[:, :, 1:] |= reach[:, :, :-1]
            step_mask[:, :, :-1] |= reach[:, :, 1:]
            step_mask &= ~pad_grid
            if (step_mask == reach).all():
                break
            reach = step_mask
        outside = reach[1:-1, 1:-1, 1:-1]
        grid = ~outside
    return grid


def hollow(grid):
    """The voxels of a solid on its surface (a face toward empty space; the floor does not count)."""
    pad_grid = np.pad(grid, 1)
    pad_grid[0] = pad_grid[1]
    core = pad_grid[1:-1, 1:-1, 1:-1] & pad_grid[:-2, 1:-1, 1:-1] & pad_grid[2:, 1:-1, 1:-1] & pad_grid[1:-1, :-2, 1:-1] & pad_grid[1:-1, 2:, 1:-1] & pad_grid[1:-1, 1:-1, :-2] & pad_grid[1:-1, 1:-1, 2:]
    return grid & ~core


def boxes_of(grid, label_by_layer=True):
    """A parts program covering a voxel grid exactly with boxes: greedy, lowest layer first, each
    box grown along x, then y, then up while it stays inside the grid's voxels. Groups are layers
    ("layer 1", ...) of the box's lowest voxel."""
    remaining = grid.copy()
    parts = []
    while remaining.any():
        z, y, x = (int(v) for v in np.argwhere(remaining)[0])
        w = 1
        while x + w < grid.shape[2] and remaining[z, y, x + w]:
            w += 1
        d = 1
        while y + d < grid.shape[1] and remaining[z, y + d, x : x + w].all():
            d += 1
        h = 1
        while z + h < grid.shape[0] and remaining[z + h, y : y + d, x : x + w].all():
            h += 1
        remaining[z : z + h, y : y + d, x : x + w] = False
        prim = "col" if (d, w) == (1, 1) and h > 1 else "slab" if h == 1 and min(d, w) > 1 else "beam" if h == 1 else "box"
        parts.append(P(prim, (z, y, x), (h, d, w), f"layer {z + 1}" if label_by_layer else "part"))
    return tuple(parts)


def mesh_shape(vertices, faces, name, category, robots=115, up=1, fill=True, shell=False, climb=True):
    """A library Shape from a mesh, or None: voxelised at the largest resolution (the grid's
    FOOTPRINT down to 3) whose voxels (only the surface when shell) number at most `robots` and
    that stands under the support rule, builds layer by layer and (climb) lets every cube walk to
    its voxel; a resolution that fails is passed over for the next smaller one. Its program is
    boxes_of: boxes grouped by layer."""
    for resolution in range(FOOTPRINT, 2, -1):
        grid = voxelise(vertices, faces, resolution, up=up, fill=fill)
        if shell:
            grid = hollow(grid)
        cells = np.argwhere(grid)
        if not len(cells) or len(cells) > robots or not fits_grid(cells):
            continue
        parts = normalise(boxes_of(grid))
        if buildable(parts) and (not climb or climbable(parts)):
            return Shape(name, category, parts, ())
    return None
