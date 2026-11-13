"""Complex goal generators: multi-part scenes, structure edits, structure-to-structure and trick instructions.

The complex tier of the goal generators (goal_gen_simple: plain shapes; goal_gen_medium: everyday objects). Each
generator is a program with random parameters, `name(rng) -> dict`, and yields thousands of distinct goals:

    cells        (N, 3) (z, y, x) world cells of the end state B, every robot (parked ones included)
    start_cells  the start A in the same frame (same robot count), or None: take apart to a natural start
    turns        user texts, oldest first; with a start, turns[:-1] built A and turns[-1] asks for B
    reply        what the model should say (tricks; empty otherwise)
    kind         scene, edit, retarget or trick
    params       parts, parking, changed cells, overhang, span, trick behaviour

Cells are in the world frame, not centred: a scene says "to the left of", "in the back left corner", "nearer me", so
where things stand matters. Words follow camera_frame.py at the camera's own side (quarter turn 0, reverse.SIDES):
x runs to the viewer's right, y toward the viewer (front = high y, back = low y), z up. A part is built in a local
frame facing the viewer (+y) and turned by k quarter turns about z; FACING[k] is where its front then points.

Categories:
1. scene: two or three parts with a relation the viewer checks at a glance (left of, behind, between, on top of,
   facing each other, taller than, in a corner of the floor).
1. edit: a built A and a change ("now lower the arms", "move the door to the back wall", "make the left tower as
   tall as the right one", "undo that"). B differs from A in exactly what the text says; params["changed"] lists
   the cells that fill and empty. Robots the edit does not need stay where they are; spares stand in a parked row
   along one edge of the floor (filled from the middle of the edge outward, so A's row and B's row share a prefix
   and the robots on it never move).
1. retarget: A and B are different structures with the same robot count ("turn the arch into a tower five high"),
   padded with parked robots. Parts come from this module or from goal_gen_simple / goal_gen_medium.
1. trick: impossible requests (more robots asked for than exist: the closest possible, or nothing when the text
   says so), already satisfied requests (no change), negation ("anything except a tower"), false premises ("take
   the roof off" a house without one), conditionals. Each records the right end state and the reply to say.

Every state passes: 4-256 robots inside the 16^3 world, every robot face-connected to the floor
(goal_library.grounded) and at rest (reverse.fixed_point). Parts may be scaled (each cell of the shape an s x s x s
block of robots, s = 2 or 3) because big builds read better; a scaled part is worded without counts. Overhang and
unsupported span are recorded (not filtered beyond MAX_OVERHANG / MAX_SPAN times the scale) for the joint-strength
study:

    overhang  longest horizontal lever: sideways steps, through the structure, from the nearest ground pillar (a
              column filled down to the floor); stepping up or down costs nothing (a hanging arm's lever is its
              shoulder's)
    span      longest straight row of cells with nothing directly under them

Texts never name a robot count of ten or more and never use the simulator's words (cheese, stripes, zones, voxels,
cells).

    .venv/bin/python -m beebot.vla.voxel.goal_gen_complex --n 10000 --seed 0 --render-dir <dir>
    .venv/bin/python -m beebot.vla.voxel.goal_gen_complex --measure 1000      # per generator acceptance etc.
    .venv/bin/python -m beebot.vla.voxel.goal_gen_complex --reverse 3         # reverse.take_apart on samples
"""

import argparse
import hashlib
import json
import re
import time
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from beebot.vla.voxel import goal_library as G

SIZE = 16
SOURCE = "gen_complex"
MAX_OVERHANG = 4  # per unit of scale
MAX_SPAN = 4
MAX_ROBOTS = G.MAX_CELLS  # end states and starts up to the goal library cap (256 robots, user, 2026-11-13)
ROW = 14  # parked robots per row along an edge (positions 1-14)
MAX_ROWS = 3
FACING = ("front", "left", "back", "right")  # where a part's front points after k quarter turns
NUMBERS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
FORBIDDEN = re.compile(r"\b(cheese|cheeses|stripe|stripes|striped|zone|zones|voxel|voxels|cell|cells)\b")
LARGE = re.compile(
    r"\b(ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
    r"fifty|sixty|seventy|eighty|ninety|hundred|dozen|dozens)\b"
)
DIGITS = re.compile(r"\b\d+\b")


class Reject(Exception):
    """A draw that cannot be built (does not fit, too many robots, a precondition fails)."""


def need(ok, why="reject"):
    if not ok:
        raise Reject(why)


def N(n):
    need(0 <= n < 10, "count of ten or more")
    return NUMBERS[n]


def pick(rng, options):
    return options[int(rng.integers(len(options)))]


def chance(rng, p=0.5):
    return bool(rng.random() < p)


def between(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


# ---------------------------------------------------------------------------------------------
# Parts: local cells (z, y, x) round an anchor at (0, 0), front toward +y


def _rot(cells, k):
    out = set(cells)
    for _ in range(k % 4):
        out = {(z, x, -y) for z, y, x in out}  # front (+y) turns to the left (-x)
    return frozenset(out)


def b_tower(h, t=1):
    return {(z, i, j) for z in range(h) for i in range(t) for j in range(t)}


def b_wall(length, height, gate=0):
    cells = {(z, 0, j) for z in range(height) for j in range(length)}
    if gate:
        start = (length - gate) // 2
        cells -= {(z, 0, j) for z in range(height - 1) for j in range(start, start + gate)}
    return cells


def house_door(w, d, side):
    return {"front": (d - 1, w // 2), "back": (0, w // 2), "left": (d // 2, 0), "right": (d // 2, w - 1)}[side]


def b_house(w, d, h, roof=True, door="front", upper=0, upper_roof=False):
    cells = {(z, i, j) for z in range(h) for i in range(d) for j in range(w) if i in (0, d - 1) or j in (0, w - 1)}
    if door:
        i, j = house_door(w, d, door)
        cells -= {(z, i, j) for z in range(min(2, h))}
    if roof:
        cells |= {(h, i, j) for i in range(d) for j in range(w)}
    if upper:  # a second floor: walls on the roof's rim, and a roof of its own when asked
        cells |= {(h + 1 + z, i, j) for z in range(upper) for i in range(d) for j in range(w) if i in (0, d - 1) or j in (0, w - 1)}
        if upper_roof:
            cells |= {(h + 1 + upper, i, j) for i in range(d) for j in range(w)}
    return cells


def b_table(a, b, legs):
    cells = {(legs, i, j) for i in range(a) for j in range(b)}
    for i, j in ((0, 0), (0, b - 1), (a - 1, 0), (a - 1, b - 1)):
        cells |= {(z, i, j) for z in range(legs)}
    return cells


def b_chair(a, legs, back):
    return b_table(a, a, legs) | {(legs + k, 0, j) for k in range(1, back + 1) for j in range(a)}


def b_bench(length, back):
    cells = {(1, i, j) for i in range(2) for j in range(length)}
    cells |= {(0, i, j) for i in range(2) for j in (0, length - 1)}
    return cells | {(1 + k, 0, j) for k in range(1, back + 1) for j in range(length)}


def b_arch(gap, p):
    cells = {(z, 0, j) for z in range(p) for j in (0, gap + 1)}
    return cells | {(p, 0, j) for j in range(gap + 2)}


def b_pyramid(base):
    return {(z, i, j) for z in range((base + 1) // 2) for i in range(z, base - z) for j in range(z, base - z)}


def b_stairs(width, steps):
    return {(z, -i, j) for i in range(steps) for z in range(i + 1) for j in range(width)}  # rises away from the front


ARM_CELLS = {
    "out": lambda S, s: {(S, 0, s), (S, 0, 2 * s), (S, 0, 3 * s)},
    "up": lambda S, s: {(S, 0, s), (S, 0, 2 * s), (S + 1, 0, 2 * s), (S + 2, 0, 2 * s)},
    "down": lambda S, s: {(S, 0, s), (S, 0, 2 * s), (S - 1, 0, 2 * s), (S - 2, 0, 2 * s)},
}


def b_figure(left, right, legs=2):
    """A standing figure, flat in the x-z plane facing +y: two legs, hips, a torso, a head and two arms ("out",
    "up" or "down"); `left` is the arm on the -x side."""
    cells = {(z, 0, s) for z in range(legs) for s in (-1, 1)} | {(legs, 0, s) for s in (-1, 0, 1)}
    cells |= {(z, 0, 0) for z in range(legs + 1, legs + 4)}
    S = legs + 3
    cells.add((S + 1, 0, 0))
    return cells | ARM_CELLS[left](S, -1) | ARM_CELLS[right](S, 1)


def b_platform(a, b):
    return {(0, i, j) for i in range(a) for j in range(b)}


def b_post(h, arm):
    return {(z, 0, 0) for z in range(h)} | {(h - 1, i, 0) for i in range(1, arm + 1)}


def b_custom(cells):
    return set(map(tuple, cells))


BUILD = {
    "tower": b_tower, "wall": b_wall, "house": b_house, "table": b_table, "chair": b_chair, "bench": b_bench,
    "arch": b_arch, "pyramid": b_pyramid, "stairs": b_stairs, "figure": b_figure, "platform": b_platform,
    "post": b_post, "custom": b_custom,
}
ORIENTED = {"house", "chair", "bench", "stairs", "figure", "post"}  # have a front a viewer can tell


def _upscale(cells, s):
    if s == 1:
        return cells
    return frozenset((z * s + a, y * s + b, x * s + c) for z, y, x in cells for a in range(s) for b in range(s) for c in range(s))


class Part:
    """A part: kind, its builder's parameters, k quarter turns, and a scale s (every cell of the shape becomes an
    s x s x s block of robots: big builds read better). `cells` are local, round the anchor. Texts never count the
    cells of a part with s > 1 (describe words it without numbers)."""

    def __init__(self, kind, params, k=0, scale=1):
        self.kind, self.params, self.k, self.scale = kind, dict(params), k % 4, scale
        self.cells = _upscale(_rot(BUILD[kind](**self.params), self.k), scale)

    def turned(self, k):
        return Part(self.kind, self.params, self.k + k, self.scale)

    def changed(self, **kw):
        return Part(self.kind, {**self.params, **kw}, self.k, self.scale)

    def scaled(self, scale):
        return Part(self.kind, self.params, self.k, scale)

    @property
    def facing(self):
        return FACING[self.k]

    def meta(self):
        params = {key: v for key, v in self.params.items() if key != "cells"}
        return {"kind": self.kind, "params": params, "k": self.k, "scale": self.scale}


class Placed:
    """A part standing in the world with its anchor at (z, y, x)."""

    def __init__(self, part, y, x, z=0):
        self.part, self.y, self.x, self.z = part, int(y), int(x), int(z)
        self.cells = frozenset((z + self.z, y + self.y, x + self.x) for z, y, x in part.cells)

    def moved(self, dy=0, dx=0, dz=0):
        return Placed(self.part, self.y + dy, self.x + dx, self.z + dz)

    def swapped(self, part):
        return Placed(part, self.y, self.x, self.z)

    @property
    def foot(self):
        ys = [c[1] for c in self.cells]
        xs = [c[2] for c in self.cells]
        return min(ys), max(ys), min(xs), max(xs)

    @property
    def top(self):
        return max(c[0] for c in self.cells)

    @property
    def centre(self):
        y0, y1, x0, x1 = self.foot
        return (y0 + y1) / 2, (x0 + x1) / 2

    def meta(self):
        return {**self.part.meta(), "at": [self.z, self.y, self.x]}


def local_foot(part):
    ys = [c[1] for c in part.cells]
    xs = [c[2] for c in part.cells]
    return min(ys), max(ys), min(xs), max(xs)


def centred_at(part, cy, cx, z=0):
    """The part placed with its footprint centre at (cy, cx) (rounded down)."""
    y0, y1, x0, x1 = local_foot(part)
    return Placed(part, int(np.floor(cy - (y0 + y1) / 2)), int(np.floor(cx - (x0 + x1) / 2)), z)


def union(placed):
    out = set()
    for p in placed:
        out |= p.cells
    return out


def apart(placed, sep=2):
    """Different parts do not touch (Chebyshev distance at least `sep` between their cells, in plan and height)."""
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            a, b = np.array(sorted(placed[i].cells)), np.array(sorted(placed[j].cells))
            if np.abs(a[:, None, :] - b[None, :, :]).max(axis=2).min() < sep:
                return False
    return True


REL_AXIS = {"left": (2, -1), "right": (2, 1), "behind": (1, -1), "front": (1, 1)}
OPPOSITE = {"left": "right", "right": "left", "behind": "front", "front": "behind"}


def next_to(ref, part, rel, gap, rng=None, z=0):
    """`part` placed beside the placed `ref` on side `rel` with `gap` empty columns between their footprints,
    centred on ref along the other axis (shifted by up to one when rng is given)."""
    ry0, ry1, rx0, rx1 = ref.foot
    py0, py1, px0, px1 = local_foot(part)
    jitter = int(rng.integers(-1, 2)) if rng is not None else 0
    cy = (ry0 + ry1) / 2 - (py0 + py1) / 2
    cx = (rx0 + rx1) / 2 - (px0 + px1) / 2
    if rel == "left":
        return Placed(part, int(np.floor(cy)) + jitter, rx0 - gap - 1 - px1, z)
    if rel == "right":
        return Placed(part, int(np.floor(cy)) + jitter, rx1 + gap + 1 - px0, z)
    if rel == "behind":
        return Placed(part, ry0 - gap - 1 - py1, int(np.floor(cx)) + jitter, z)
    return Placed(part, ry1 + gap + 1 - py0, int(np.floor(cx)) + jitter, z)


def overlap_ok(a, b, rel):
    """The two footprints overlap along the axis across `rel` (so the relation reads unambiguously)."""
    ay0, ay1, ax0, ax1 = a.foot
    by0, by1, bx0, bx1 = b.foot
    if rel in ("left", "right"):
        return min(ay1, by1) >= max(ay0, by0)
    return min(ax1, bx1) >= max(ax0, bx0)


def fit(groups, rng, reserve=None, margin=1):
    """Shift every Placed of every group by one random (dy, dx) so all cells lie inside the floor (keeping
    `margin` free at each edge, and rows + 1 at the edge `reserve` = (side, rows) kept for parked robots)."""
    cells = set()
    for g in groups:
        cells |= union(g)
    need(cells, "empty")
    zs, ys, xs = (np.array([c[i] for c in cells]) for i in range(3))
    need(zs.min() >= 0 and zs.max() < SIZE, "too tall")
    lo_y, hi_y, lo_x, hi_x = margin, SIZE - 1 - margin, margin, SIZE - 1 - margin
    if reserve is not None:
        side, rows = reserve
        if side == "back":
            lo_y = max(lo_y, rows + 1)
        elif side == "front":
            hi_y = min(hi_y, SIZE - 2 - rows)
        elif side == "left":
            lo_x = max(lo_x, rows + 1)
        else:
            hi_x = min(hi_x, SIZE - 2 - rows)
    dy0, dy1 = lo_y - ys.min(), hi_y - ys.max()
    dx0, dx1 = lo_x - xs.min(), hi_x - xs.max()
    need(dy0 <= dy1 and dx0 <= dx1, "does not fit")
    dy, dx = between(rng, dy0, dy1), between(rng, dx0, dx1)
    return [[p.moved(dy, dx) for p in g] for g in groups]


# ---------------------------------------------------------------------------------------------
# Parked robots: rows along one edge of the floor, filled from the middle outward


SIDES = ("back", "front", "left", "right")
_ORDER = [7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2, 13, 1, 14]


def park_slots(side, rows):
    out = []
    for r in range(rows):
        index = r if side in ("back", "left") else SIZE - 1 - r
        out += [(0, index, j) if side in ("back", "front") else (0, j, index) for j in _ORDER]
    return out


def rows_for(n):
    return 0 if n <= 0 else -(-n // ROW)


def park_fits(cells, side, rows):
    """No structure cell within rows + 1 of the edge (a gap of one between the rows and the structure)."""
    for _, y, x in cells:
        v = {"back": y, "front": SIZE - 1 - y, "left": x, "right": SIZE - 1 - x}[side]
        if v < rows + 1:
            return False
    return True


def choose_side(rng, cells, rows):
    for k in rng.permutation(4):
        if park_fits(cells, SIDES[int(k)], rows):
            return SIDES[int(k)]
    return None


EDGE_WORDS = {
    "back": ("back edge", "far edge", "back edge of the floor"),
    "front": ("front edge", "edge nearest me", "front edge of the floor"),
    "left": ("left edge", "left-hand edge", "left edge of the floor"),
    "right": ("right edge", "right-hand edge", "right edge of the floor"),
}


def park_clause(rng, side):
    edge = pick(rng, EDGE_WORDS[side])
    return pick(rng, [
        f", and line the rest of you up along the {edge}",
        f"; everyone else waits in a row along the {edge}",
        f", with the spare ones parked in a row by the {edge}",
        f". The rest of you, stand in a row along the {edge}",
    ])


def leftover_clause(rng, side):
    edge = pick(rng, EDGE_WORDS[side])
    return pick(rng, [
        f", and put the leftover ones in a row along the {edge}",
        f"; any of you not needed, line up along the {edge}",
        f". Spare ones go to the {edge}",
    ])


def with_parking(rng, a_struct, b_struct, extra=0, side=None):
    """Pad A and B to the same robot count with parked rows on one edge: (A cells, B cells, info). A's row and
    B's row are prefixes of the same slot order, so the robots both share never move."""
    total = max(len(a_struct), len(b_struct)) + extra
    na, nb = total - len(a_struct), total - len(b_struct)
    n = max(na, nb)
    info = {"side": None, "a": na, "b": nb}
    if n == 0:
        return set(a_struct), set(b_struct), info
    rows = rows_for(n)
    need(rows <= MAX_ROWS, "too many parked")
    side = side or choose_side(rng, a_struct | b_struct, rows)
    need(side is not None and park_fits(a_struct | b_struct, side, rows), "no edge to park")
    slots = park_slots(side, rows)
    info["side"] = side
    return set(a_struct) | set(slots[:na]), set(b_struct) | set(slots[:nb]), info


# ---------------------------------------------------------------------------------------------
# Words


DIR_WORDS = {"front": "toward me", "back": "away from me", "left": "to the left", "right": "to the right"}
FACE_WORDS = {
    "front": ("facing me", "facing toward me", "turned toward me"),
    "back": ("facing away from me", "with its back to me", "turned away from me"),
    "left": ("facing left", "facing to the left", "turned to face left"),
    "right": ("facing right", "facing to the right", "turned to face right"),
}
SHORT = {
    "tower": "tower", "wall": "wall", "house": "house", "table": "table", "chair": "chair", "bench": "bench",
    "arch": "arch", "pyramid": "pyramid", "stairs": "staircase", "figure": "figure", "platform": "platform",
    "post": "post",
}
PLURAL = {
    "tower": "towers", "wall": "walls", "house": "houses", "table": "tables", "chair": "chairs", "bench": "benches",
    "arch": "arches", "pyramid": "pyramids", "stairs": "staircases", "figure": "figures", "platform": "platforms",
    "post": "posts",
}
POSE_WORDS = {"out": "straight out", "up": "raised", "down": "down"}


def arms_phrase(left, right, k, rng):
    if left == right:
        return pick(rng, {
            "out": ["both arms straight out to the sides", "its arms stretched out sideways", "arms out wide"],
            "up": ["both arms up", "its arms raised", "both arms up in the air"],
            "down": ["its arms down at its sides", "both arms hanging down", "arms down"],
        }[left])
    sl, sr = arm_side(-1, k), arm_side(1, k)
    return f"the arm on the {sl} {POSE_WORDS[left]} and the arm on the {sr} {POSE_WORDS[right]}"


def arm_side(s, k):
    """The viewer's word for the side of a figure turned k quarter turns that its local -x (s=-1) / +x arm is on."""
    v = list(_rot({(0, 0, s)}, k))[0]
    if v[2] != 0:
        return "left" if v[2] < 0 else "right"
    return "far side" if v[1] < 0 else "near side"


def describe_big(part, rng):
    """A noun phrase without counts for a scaled-up part."""
    p, kind = part.params, part.kind
    big = pick(rng, ["big", "large", "huge"] if part.scale >= 3 else ["big", "large"])
    if kind == "tower":
        return pick(rng, [f"a {big} thick tower", f"a {big}, wide tower"] + (["a tall, thick tower"] if p["h"] >= 5 else []))
    if kind == "wall":
        return f"a {big} wall with a gate in the middle" if p.get("gate") else pick(rng, [f"a {big} thick wall", f"a {big} solid wall"])
    if kind == "house":
        roof, door = p.get("roof", True), p.get("door")
        side = "" if door in (None, "front") else f" with the door on the {DOOR_SIDE[door]}"
        if roof and door:
            base = pick(rng, [f"a {big} house with a door and a flat roof", f"a {big} house with a flat roof", f"a {big} house with a doorway"])
        elif roof:
            base = f"a {big} house with a roof but no door"
        elif door:
            base = pick(rng, [f"a {big} house with no roof", f"a {big} roofless house with a doorway"])
        else:
            base = f"a {big} ring of walls with no roof and no door"
        return base + side
    if kind == "table":
        return pick(rng, [f"a {big} table", f"a {big} table on four legs"])
    if kind == "chair":
        return pick(rng, [f"a {big} chair with a back", f"a {big} chair"])
    if kind == "bench":
        return pick(rng, [f"a {big} bench with a backrest", f"a {big} bench with a back"])
    if kind == "arch":
        return pick(rng, [f"a {big} arch", f"a {big} archway"])
    if kind == "pyramid":
        return f"a {big} stepped pyramid"
    if kind == "stairs":
        rise = DIR_WORDS[FACING[(part.k + 2) % 4]]
        return pick(rng, [f"a {big} staircase going up {rise}", f"{big} wide stairs rising {rise}"])
    if kind == "figure":
        arms = arms_phrase(p["left"], p["right"], part.k, rng)
        return pick(rng, [f"a {big} figure with {arms}", f"a {big} stick figure with {arms}"])
    if kind == "platform":
        return f"a {big} flat platform"
    if kind == "post":
        return f"a {big} upside-down L with its arm pointing {DIR_WORDS[part.facing]}"
    raise Reject("no words")


def describe(part, rng):
    """An indefinite noun phrase a viewer can check against the part."""
    if part.scale > 1:
        return describe_big(part, rng)
    p, kind = part.params, part.kind
    if kind == "tower":
        h, t = p["h"], p.get("t", 1)
        if t == 2:
            return pick(rng, [f"a thick tower {N(h)} high", f"a two by two tower {N(h)} high", f"a fat tower {N(h)} tall"])
        opts = [f"a tower {N(h)} high", f"a {N(h)}-high tower", f"a column {N(h)} tall", f"a stack of {N(h)}"]
        if h >= 6:
            opts.append("a tall tower")
        return pick(rng, opts)
    if kind == "wall":
        L, H, gate = p["length"], p["height"], p.get("gate", 0)
        if gate:
            return pick(rng, ["a wall with a gate in the middle", f"a wall {N(H)} high with an opening through the middle", "a wall with a doorway in the middle"])
        return pick(rng, [f"a wall {N(L)} long and {N(H)} high", f"a straight wall {N(L)} long, {N(H)} high", f"a {N(H)}-high wall {N(L)} long"])
    if kind == "house":
        roof, door = p.get("roof", True), p.get("door")
        side = "" if door in (None, "front") else f" with the door on the {DOOR_SIDE[door]}"
        if roof and door:
            base = pick(rng, ["a small house", "a little house with a door and a flat roof", "a house with a flat roof", "a small house with a doorway"])
            if door == "front" and "door" not in base and chance(rng, 0.4):
                side = pick(rng, [" with its door facing me", " with the doorway toward me"])
        elif roof:
            base = pick(rng, ["a house with no door", "a closed house with a roof but no door"])
        elif door:
            base = pick(rng, ["a house with no roof", "the walls of a small house, open on top, with a doorway", "a roofless house"])
        else:
            base = pick(rng, ["four walls with no roof and no door", "a closed ring of walls with no roof"])
        return base + side
    if kind == "table":
        a, b, legs = p["a"], p["b"], p["legs"]
        return pick(rng, ["a table", f"a table with legs {N(legs)} high", f"a {N(a)} by {N(b)} table", "a table on four legs"])
    if kind == "chair":
        return pick(rng, ["a chair", "a chair with a backrest", "a chair with a back"] + (["a chair with a tall back"] if p["back"] >= 3 else []))
    if kind == "bench":
        return pick(rng, ["a bench with a backrest", "a bench with a back", f"a bench {N(p['length'])} long with a back"])
    if kind == "arch":
        return pick(rng, ["an arch", f"an arch with an opening {N(p['gap'])} wide", "two posts with a beam across the top", f"an arch {N(p['p'] + 1)} high"])
    if kind == "pyramid":
        levels = (p["base"] + 1) // 2
        return pick(rng, ["a stepped pyramid", f"a pyramid {N(levels)} levels high", "a small stepped pyramid" if levels == 2 else "a stepped pyramid with a point on top"])
    if kind == "stairs":
        rise = DIR_WORDS[FACING[(part.k + 2) % 4]]
        return pick(rng, [f"a staircase {N(p['steps'])} steps high going up {rise}", f"stairs with {N(p['steps'])} steps rising {rise}", f"a flight of stairs going up {rise}"])
    if kind == "figure":
        arms = arms_phrase(p["left"], p["right"], part.k, rng)
        return pick(rng, [f"a figure with {arms}", f"a little person with {arms}", f"a stick figure with {arms}"])
    if kind == "platform":
        return pick(rng, [f"a flat platform {N(p['a'])} by {N(p['b'])}", f"a low flat slab {N(p['a'])} by {N(p['b'])}", "a flat platform one high"])
    if kind == "post":
        return pick(rng, [f"a post {N(p['h'])} high with an arm sticking out {DIR_WORDS[part.facing]} at the top", f"an upside-down L with its arm pointing {DIR_WORDS[part.facing]}"])
    raise Reject("no words")


DOOR_SIDE = {"front": "side facing me", "back": "back wall", "left": "left side", "right": "right side"}


def the(part):
    return "the " + SHORT[part.kind]


def polish(text, rng):
    text = text.strip()
    if chance(rng, 0.5):
        text = text[0].upper() + text[1:]
    if chance(rng, 0.3) and text[-1] not in ".!?":
        text += "."
    return text


def text_ok(text):
    low = text.lower()
    if FORBIDDEN.search(low) or LARGE.search(low):
        return False
    return all(int(d) < 10 for d in DIGITS.findall(low))


# ---------------------------------------------------------------------------------------------
# Random parts


SCENE_KINDS = ("tower", "wall", "house", "table", "chair", "bench", "arch", "pyramid", "stairs", "figure", "platform", "post")
PAIR_WEIGHTS = {"tower": 3, "wall": 1.5, "house": 2, "table": 2, "chair": 1.5, "bench": 1.5, "arch": 1.5, "pyramid": 1, "stairs": 1, "figure": 1.5, "platform": 0.7, "post": 0.8}


def pick_scale(rng, p2=0.3, p3=0.1):
    """1 (most draws), 2 or 3: parts built from s x s x s blocks of robots."""
    r = rng.random()
    return 3 if r < p3 else 2 if r < p2 + p3 else 1


def rand_kind(rng, exclude=(), weights=PAIR_WEIGHTS):
    kinds = [k for k in weights if k not in exclude]
    w = np.array([weights[k] for k in kinds], dtype=float)
    return kinds[int(rng.choice(len(kinds), p=w / w.sum()))]


def rand_part(rng, kind, small=False):
    if kind == "tower":
        return Part("tower", {"h": between(rng, 3, 6 if small else 8), "t": 2 if chance(rng, 0.12) else 1})
    if kind == "wall":
        if chance(rng, 0.35) and not small:
            L = int(pick(rng, [5, 6, 7, 8, 9]))
            return Part("wall", {"length": L, "height": between(rng, 2, 4), "gate": 1 if L % 2 else 2}, int(pick(rng, [0, 1])))
        return Part("wall", {"length": between(rng, 3, 5 if small else 8), "height": between(rng, 1, 3)}, int(pick(rng, [0, 1])))
    if kind == "house":
        hi = 4 if small else 5
        return Part("house", {"w": between(rng, 3, hi), "d": between(rng, 3, hi), "h": 2 if small else between(rng, 2, 3), "roof": chance(rng, 0.8), "door": "front"})
    if kind == "table":  # a long side of 3 or more, else the corner legs fill the footprint and it reads as a block
        a, b = between(rng, 3, 3 if small else 5), between(rng, 3, 4 if small else 5)  # both sides 3 or more: a 2-wide side shows as a solid block
        return Part("table", {"a": a, "b": b, "legs": between(rng, 1, 3)})
    if kind == "chair":
        return Part("chair", {"a": 3, "legs": between(rng, 1, 2), "back": between(rng, 1, 3)})
    if kind == "bench":
        return Part("bench", {"length": between(rng, 4, 5), "back": between(rng, 1, 2)})
    if kind == "arch":
        return Part("arch", {"gap": between(rng, 1, 3), "p": between(rng, 2, 4)}, int(pick(rng, [0, 1])))
    if kind == "pyramid":
        return Part("pyramid", {"base": 3 if small else int(pick(rng, [3, 5]))})
    if kind == "stairs":
        return Part("stairs", {"width": between(rng, 1, 2 if small else 3), "steps": between(rng, 3, 5)}, between(rng, 0, 3))
    if kind == "figure":
        return Part("figure", {"left": pick(rng, list(ARM_CELLS)), "right": pick(rng, list(ARM_CELLS)), "legs": between(rng, 2, 3)})
    if kind == "platform":
        return Part("platform", {"a": between(rng, 2, 5), "b": between(rng, 2, 5)})
    if kind == "post":
        return Part("post", {"h": between(rng, 2, 5), "arm": between(rng, 2, 3)}, between(rng, 0, 3))
    raise Reject(kind)


# ---------------------------------------------------------------------------------------------
# Goal records


def overhang_span(cells):
    """(overhang, span) of a set of cells (module docstring)."""
    occ = set(map(tuple, cells))
    pillar = set()
    for c in sorted(occ):
        z, y, x = c
        if z == 0 or (z - 1, y, x) in pillar:
            pillar.add(c)
    dist = {c: 0 for c in pillar}
    queue = deque(pillar)
    while queue:
        u = queue.popleft()
        z, y, x = u
        for v in ((z + 1, y, x), (z - 1, y, x), (z, y + 1, x), (z, y - 1, x), (z, y, x + 1), (z, y, x - 1)):
            if v not in occ or v in pillar:
                continue
            cost = 0 if v[0] != z else 1
            if dist[u] + cost < dist.get(v, 1 << 30):
                dist[v] = dist[u] + cost
                if cost == 0:
                    queue.appendleft(v)
                else:
                    queue.append(v)
    over = max(dist.values()) if dist else 0
    hanging = {c for c in occ if c[0] > 0 and (c[0] - 1, c[1], c[2]) not in occ}
    span = 0
    for z, y, x in hanging:
        for dy, dx in ((0, 1), (1, 0)):
            if (z, y - dy, x - dx) in hanging:
                continue
            n = 0
            while (z, y + n * dy, x + n * dx) in hanging:
                n += 1
            span = max(span, n)
    return int(over), int(span)


def record(kind, cells, turns, start=None, reply="", parts=None, parking=None, **extra):
    cells = sorted(set(map(tuple, cells)))
    params = {"parts": [p.meta() for p in (parts or [])], "parked": parking or {"side": None, "a": 0, "b": 0}}
    params["scale"] = max([p.part.scale for p in (parts or [])] + [1])
    if start is not None:
        start = sorted(set(map(tuple, start)))
        a, b = set(start), set(cells)
        params["changed"] = {"fill": sorted(b - a), "empty": sorted(a - b)}
    params.update(extra)
    return {"cells": cells, "start_cells": start, "turns": list(turns), "reply": reply, "kind": kind, "params": params}


def struct_with_spares(rng, struct, extra, text, side=None):
    """A scene's cells with `extra` parked robots (and the parking clause added to the text)."""
    if extra <= 0:
        return set(struct), text, None
    _, cells, info = with_parking(rng, set(struct), set(struct), extra=extra, side=side)
    return cells, text + park_clause(rng, info["side"]), info


# ---------------------------------------------------------------------------------------------
# Scenes (category 1)


def rel_words(rel, ref, rng, oriented):
    """`rel` (left, right, front, behind) of the noun phrase `ref`, in the viewer's frame."""
    if rel in ("left", "right"):
        if oriented:
            return pick(rng, [f"to the {rel} of {ref} as I look at it", f"to the {rel} of {ref}, seen from where I am", f"to the {rel} of {ref} from my side"])
        return pick(rng, [f"to the {rel} of {ref}", f"on the {rel} side of {ref}", f"just {rel} of {ref}"])
    if rel == "front":
        return pick(rng, [f"in front of {ref}", f"between me and {ref}", f"on the near side of {ref}", f"closer to me than {ref}"])
    return pick(rng, [f"behind {ref}", f"on the far side of {ref}", f"farther from me than {ref}, behind it"])


def pair_text(rng, xp, yp, rel):
    X, Y = describe(xp.part, rng), describe(yp.part, rng)
    if " with " in X:
        X += ","  # "a post with an arm out at the top, to the left of ..."
    oriented = yp.part.kind in ORIENTED
    forms = [
        f"build {X} {rel_words(rel, Y, rng, oriented)}",
        f"put {X} {rel_words(rel, Y, rng, oriented)}",
        f"make {Y}, and {X} {rel_words(rel, 'it', rng, oriented)}",
        f"I want {Y}, and {X} {rel_words(rel, 'it', rng, oriented)}",
        f"{X} {rel_words(rel, Y, rng, oriented)}, please",
    ]
    if rel in ("left", "right"):
        other = OPPOSITE[rel]
        forms += [f"{X} on the {rel} and {Y} on the {other}", f"on the {other}, {Y}; on the {rel}, {X}", f"build two things side by side: {X} on the {rel}, {Y} on the {other}"]
    else:
        near, far = (X, Y) if rel == "front" else (Y, X)
        forms += [f"{far} at the back and {near} in front of it, nearer to me", f"{near} close to me and {far} behind it"]
    return pick(rng, forms)


def pair_scene(rng, kinds=None, rel=None, small=None):
    """Two different parts, X `rel` Y, gap 1-3. Returns (xp, yp, rel, gap)."""
    small = chance(rng, 0.6) if small is None else small
    ky = kinds[1] if kinds else rand_kind(rng)
    kx = kinds[0] if kinds else rand_kind(rng, exclude=(ky,))
    Y, X = rand_part(rng, ky, small), rand_part(rng, kx, small)
    if chance(rng, 0.25):  # one big part (the other at its own size): "a big house with a tower to its left"
        Y = Y.scaled(2)
    if Y.kind in ORIENTED and Y.kind not in ("stairs", "post"):
        Y = Y.turned(-Y.k)
    if X.kind in ORIENTED and X.kind not in ("stairs", "post"):
        X = X.turned(-X.k)
    rel = rel or pick(rng, ["left", "right", "front", "behind"])
    gap = between(rng, 1, 3)
    yp = Placed(Y, 0, 0)
    xp = next_to(yp, X, rel, gap, rng)
    need(overlap_ok(xp, yp, rel) and apart([xp, yp]), "pair")
    return xp, yp, rel, gap


def gen_scene_pair(rng):
    xp, yp, rel, gap = pair_scene(rng)
    extra = between(rng, 1, 6) if chance(rng, 0.25) else 0
    side = pick(rng, SIDES) if extra else None
    [[xp, yp]] = fit([[xp, yp]], rng, reserve=(side, 1) if extra else None)
    cells, text, park = struct_with_spares(rng, xp.cells | yp.cells, extra, pair_text(rng, xp, yp, rel), side)
    return record("scene", cells, [polish(text, rng)], parts=[xp, yp], parking=park, relation=rel, gap=gap)


def gen_scene_row3(rng):
    kinds = []
    for _ in range(3):
        kinds.append(rand_kind(rng, exclude=kinds))
    parts = [rand_part(rng, k, small=True) for k in kinds]
    parts = [p.turned(-p.k) if p.kind in ORIENTED and p.kind not in ("stairs", "post") else p for p in parts]
    axis = "x" if chance(rng, 0.7) else "y"
    lo, hi = ("left", "right") if axis == "x" else ("behind", "front")
    gap = between(rng, 1, 3)
    mid = Placed(parts[1], 0, 0)
    first = next_to(mid, parts[0], lo, gap)
    last = next_to(mid, parts[2], hi, gap)
    placed = [first, mid, last]
    need(apart(placed), "row")
    [placed] = fit([placed], rng)
    A, B, C = (describe(p, rng) for p in parts)
    if axis == "x":
        text = pick(rng, [
            f"from left to right: {A}, {B} and {C}",
            f"a row of three things: {A} on the left, {B} in the middle and {C} on the right",
            f"line up {A}, {B} and {C}, in that order from left to right",
            f"put {B} in the middle, {A} to its left and {C} to its right",
        ])
    else:
        text = pick(rng, [
            f"from the back to the front: {A}, {B} and {C}",
            f"{A} farthest from me, then {B}, then {C} closest to me",
            f"a row going toward me: {A} at the back, {B} in the middle, {C} at the front",
        ])
    return record("scene", union(placed), [polish(text, rng)], parts=placed, axis=axis, gap=gap)


def gen_scene_facing(rng):
    mode = pick(rng, ["facing", "facing", "back_to_back", "side_me", "side_away", "table_chairs"])
    if mode == "table_chairs":
        table = Part("table", {"a": between(rng, 3, 4), "b": between(rng, 3, 4), "legs": between(rng, 1, 2)})
        chair = Part("chair", {"a": 3, "legs": 1, "back": between(rng, 1, 2)})
        tp = Placed(table, 0, 0)
        sides = pick(rng, [("left", "right"), ("front", "behind"), ("left", "right", "front", "behind")])
        placed = [tp]
        for s in sides:
            k = {"left": 3, "right": 1, "front": 2, "behind": 0}[s]  # the chair's front points at the table
            placed.append(next_to(tp, chair.turned(k), s, 1))
        need(apart(placed), "chairs")
        [placed] = fit([placed], rng)
        if len(sides) == 4:
            text = pick(rng, ["a table with a chair on each of its four sides, every chair facing the table", "put four chairs round a table, one per side, all facing in"])
        elif sides[0] == "left":
            text = pick(rng, ["a table with a chair on the left and a chair on the right, both facing it", "a table with two chairs pulled up to it, one on each side from where I stand left and right"])
        else:
            text = pick(rng, ["a table with one chair in front of it and one behind it, both facing the table", "a table with a chair on the side nearest me and one on the far side, both turned toward the table"])
        return record("scene", union(placed), [polish(text, rng)], parts=placed, mode=mode)
    kind = pick(rng, ["bench", "chair"])
    part = rand_part(rng, kind)
    part = part.turned(-part.k).scaled(pick_scale(rng, 0.35, 0.0))
    word = PLURAL[kind] if part.scale == 1 else "big " + PLURAL[kind]
    if mode in ("facing", "back_to_back"):
        axis = pick(rng, ["x", "y"])
        if axis == "x":
            ka, kb, rel = (3, 1, "right") if mode == "facing" else (1, 3, "right")
        else:
            ka, kb, rel = (0, 2, "front") if mode == "facing" else (2, 0, "front")
        gap = between(rng, 2, 3) if mode == "facing" else 1
        first = Placed(part.turned(ka), 0, 0)
        second = next_to(first, part.turned(kb), rel, gap)
        placed = [first, second]
        if mode == "facing":
            text = pick(rng, [f"two {word} facing each other", f"set out two {word} that face each other with a gap between them", f"a pair of {word}, face to face"])
            text += pick(rng, ["", ", one on the left and one on the right" if axis == "x" else ", one near me and one farther back"])
        else:
            text = pick(rng, [f"two {word} back to back", f"two {word} facing away from each other, backs close together"])
    else:
        k = 0 if mode == "side_me" else 2
        gap = between(rng, 1, 2)
        first = Placed(part.turned(k), 0, 0)
        second = next_to(first, part.turned(k), "right", gap)
        placed = [first, second]
        face = pick(rng, FACE_WORDS[FACING[k]])
        text = pick(rng, [f"two {word} side by side, both {face}", f"put two {word} next to each other, left and right, both {face}"])
    need(apart(placed), "facing")
    [placed] = fit([placed], rng)
    return record("scene", union(placed), [polish(text, rng)], parts=placed, mode=mode)


def wall_with_towers(L, H, gate, ends, e):
    """A wall along local x with towers H + e high at the chosen ends ("left" = x = -1 side, "right")."""
    cells = b_wall(L, H, gate)
    if "left" in ends:
        cells |= {(z, 0, -1) for z in range(H + e)}
    if "right" in ends:
        cells |= {(z, 0, L) for z in range(H + e)}
    return cells


def gen_scene_wall_towers(rng):
    L, H = between(rng, 4, 8), between(rng, 1, 3)
    gate = 0
    if chance(rng, 0.5) and L >= 5 and H >= 2:
        gate = 1 if L % 2 else 2
    ends = pick(rng, [("left", "right"), ("left", "right"), ("left",), ("right",)])
    e = between(rng, 1, 3)
    along_y = chance(rng, 0.3)
    k = 1 if along_y else 0  # k = 1 turns local -x (left end) to the back (far end)
    part = Part("custom", {"cells": tuple(sorted(wall_with_towers(L, H, gate, ends, e)))}, k)
    [[placed]] = fit([[Placed(part, 0, 0)]], rng)
    end_word = {"left": "far end" if along_y else "left end", "right": "end nearest me" if along_y else "right end"}
    gate_words = pick(rng, [" with a gate in the middle", " with an opening through the middle", " with a doorway in the middle"]) if gate else ""
    run = pick(rng, [" running away from me", " running front to back"]) if along_y else pick(rng, ["", " running left to right"])
    th = N(H + e)
    if len(ends) == 2:
        text = pick(rng, [
            f"a wall{gate_words} and a tower at each end",
            f"a wall {N(H)} high{run}{gate_words}, with a tower {th} high at both ends",
            f"build a wall{run}{gate_words} with a taller tower at each end",
        ])
    else:
        text = pick(rng, [
            f"a wall{run}{gate_words} with a tower at its {end_word[ends[0]]}",
            f"a wall {N(L)} long and {N(H)} high{gate_words}, with a tower {th} high at the {end_word[ends[0]]}",
        ])
    part_meta = Part("wall", {"length": L, "height": H, "gate": gate}, k)
    return record("scene", placed.cells, [polish(text, rng)], parts=[Placed(part_meta, placed.y, placed.x)], ends=list(ends), tower_extra=e)


def gen_scene_between(rng):
    mk = pick(rng, ["table", "arch", "house", "figure", "pyramid", "platform", "bench", "chair", "wall"])
    sk = pick(rng, ["tower", "tower", "tower", "figure", "pyramid", "post"] if mk != "figure" else ["tower", "pyramid"])
    M = rand_part(rng, mk, small=True)
    M = M.turned(-M.k) if mk in ORIENTED else M
    S = rand_part(rng, sk, small=True)
    if sk == "pyramid":
        S = Part("pyramid", {"base": 3})
    if sk == "figure":
        S = S.changed(left=S.params["right"])  # the same both sides so "two figures" reads as a pair
    axis = "x" if chance(rng, 0.7) else "y"
    gap = between(rng, 1, 2)
    mp = Placed(M, 0, 0)
    lo, hi = ("left", "right") if axis == "x" else ("behind", "front")
    S2 = S
    if sk == "post":  # arms point away from the middle
        k_lo, k_hi = {"x": (1, 3), "y": (2, 0)}[axis]
        S, S2 = S.turned(k_lo - S.k), S.turned(k_hi - S.k)
    placed = [next_to(mp, S, lo, gap), mp, next_to(mp, S2, hi, gap)]
    need(apart(placed), "between")
    [placed] = fit([placed], rng)
    m = describe(M, rng)
    if sk == "tower":
        s_desc = pick(rng, [f"two towers {N(S.params['h'])} high", "two towers"]) if S.params.get("t", 1) == 1 else f"two thick towers {N(S.params['h'])} high"
        one = f"a tower {N(S.params['h'])} high"
    elif sk == "post":
        s_desc, one = f"two posts {N(S.params['h'])} high with their arms pointing away from it", f"a post with its arm pointing outward"
    elif sk == "figure":
        s_desc, one = f"two figures with {arms_phrase(S.params['left'], S.params['right'], 0, rng)}", "a figure"
    else:
        s_desc, one = "two small pyramids", "a small pyramid"
    if axis == "x":
        text = pick(rng, [f"{m} between {s_desc}", f"{s_desc} with {m} in the middle", f"{m} with {one} on each side, left and right"])
    else:
        text = pick(rng, [f"{m} with {one} in front of it and another behind it", f"{s_desc}, one near me and one far away, with {m} between them"])
    return record("scene", union(placed), [polish(text, rng)], parts=placed, axis=axis, gap=gap)


def surface(placed):
    """{(y, x): height of the top cell} of a placed part."""
    out = {}
    for z, y, x in placed.cells:
        out[(y, x)] = max(out.get((y, x), -1), z)
    return out


def put_on(base, part, rng, where):
    """`part` standing on the flat top of `base` at `where` (middle, left, right, back, front). The top's
    columns under the part's bottom layer must all be at one height."""
    top = surface(base)
    by0, by1, bx0, bx1 = base.foot
    py0, py1, px0, px1 = local_foot(part)
    h, w = py1 - py0 + 1, px1 - px0 + 1
    cy, cx = (by0 + by1) / 2, (bx0 + bx1) / 2
    if where == "left":
        cx = bx0 + (w - 1) / 2
    elif where == "right":
        cx = bx1 - (w - 1) / 2
    elif where == "back":
        cy = by0 + (h - 1) / 2
    elif where == "front":
        cy = by1 - (h - 1) / 2
    probe = centred_at(part, cy, cx)
    bottom = [(y, x) for z, y, x in probe.cells if z == 0]
    heights = {top.get(c, -1) for c in bottom}
    need(len(heights) == 1 and min(heights) >= 0, "uneven top")
    placed = probe.moved(dz=heights.pop() + 1)
    need(not (placed.cells & base.cells), "overlap")
    return placed


def gen_scene_on_top(rng):
    base_kind = pick(rng, ["table", "platform", "platform", "house", "arch", "wall"])
    if base_kind == "table":
        base = Part("table", {"a": between(rng, 3, 4), "b": between(rng, 3, 4), "legs": between(rng, 1, 3)})
    elif base_kind == "platform":
        base = Part("platform", {"a": between(rng, 3, 5), "b": between(rng, 3, 6)})
    elif base_kind == "house":
        base = Part("house", {"w": between(rng, 3, 4), "d": between(rng, 3, 4), "h": 2, "roof": True, "door": "front"})
    elif base_kind == "arch":
        base = Part("arch", {"gap": between(rng, 1, 3), "p": between(rng, 2, 3)}, int(pick(rng, [0, 1])))
    else:
        base = Part("wall", {"length": between(rng, 3, 7), "height": between(rng, 1, 3)}, int(pick(rng, [0, 1])))
    top_kind = pick(rng, ["tower", "tower", "pyramid"] + (["figure", "chair", "post"] if base_kind in ("platform", "house") else []))
    if top_kind == "tower":
        top = Part("tower", {"h": between(rng, 1, 4), "t": 1})
    elif top_kind == "pyramid":
        top = Part("pyramid", {"base": 3})
    elif top_kind == "figure":
        top = rand_part(rng, "figure")
    elif top_kind == "chair":
        top = Part("chair", {"a": 3, "legs": 1, "back": between(rng, 1, 2)})
    else:
        top = Part("post", {"h": between(rng, 2, 3), "arm": 2}, between(rng, 0, 3))
    bp = Placed(base, 0, 0)
    bw = bp.foot[3] - bp.foot[2] + 1
    tw = local_foot(top)[3] - local_foot(top)[2] + 1
    wheres = ["middle"] + (["left", "right"] if bw - tw >= 2 else [])
    where = pick(rng, wheres)
    tp = put_on(bp, top, rng, where)
    [[bp, tp]] = fit([[bp, tp]], rng)
    B, T = describe(base, rng), describe(top, rng)
    at = {"middle": pick(rng, ["", " in the middle", " right in the middle"]), "left": " at the left end", "right": " at the right end"}[where]
    if top_kind == "tower" and top.params["h"] == 1:
        T = "a single block"
    text = pick(rng, [f"{T} standing on top of {B}{at}", f"put {T} on {B}{at}", f"{B} with {T} on top{at}", f"build {B}, then stand {T} on it{at}"])
    return record("scene", bp.cells | tp.cells, [polish(text, rng)], parts=[bp, tp], where=where)


def towers_row(rng, heights, axis, gap):
    parts = [Part("tower", {"h": h, "t": 1}) for h in heights]
    placed = [Placed(parts[0], 0, 0)]
    rel = "right" if axis == "x" else "front"
    for p in parts[1:]:
        placed.append(next_to(placed[-1], p, rel, gap))
    return placed


def gen_scene_compare(rng):
    mode = pick(rng, ["taller", "same", "double", "increasing", "middle", "walls"])
    axis = "x" if chance(rng, 0.7) else "y"
    gap = between(rng, 1, 3)
    names = ("left", "right") if axis == "x" else ("one at the back", "one nearer me")
    if mode == "walls":
        H1, H2 = between(rng, 1, 2), between(rng, 3, 4)
        L = between(rng, 3, 7)
        if chance(rng):
            H1, H2 = H2, H1
        back, front = Part("wall", {"length": L, "height": H1}), Part("wall", {"length": L, "height": H2})
        bp = Placed(back, 0, 0)
        placed = [bp, next_to(bp, front, "front", gap)]
        [placed] = fit([placed], rng)
        hi = "far" if H1 > H2 else "near"
        text = pick(rng, [f"two parallel walls running left to right, the {hi} one higher", f"two walls one behind the other, the one {'farther from me' if hi == 'far' else 'nearer me'} taller than the other"])
        return record("scene", union(placed), [polish(text, rng)], parts=placed, mode=mode)
    if mode == "taller":
        a, b = between(rng, 1, 6), 0
        b = between(rng, a + 2, 9) if a + 2 <= 9 else 9
        need(b - a >= 2)
        heights = [a, b] if chance(rng) else [b, a]
        tall = names[0] if heights[0] > heights[1] else names[1]
        short = names[1] if tall == names[0] else names[0]
        if axis == "x":
            text = pick(rng, [f"two towers side by side, the {tall} one taller", f"two towers, the one on the {short} shorter than the one on the {tall}", f"two towers next to each other with the taller one on the {tall}"])
        else:
            text = pick(rng, [f"two towers, one behind the other, the {tall} taller", f"two towers in a line going away from me, the {short} shorter"])
    elif mode == "same":
        h = between(rng, 2, 7)
        heights = [h, h]
        text = pick(rng, [f"two towers of the same height, {N(h)} high, side by side" if axis == "x" else f"two towers {N(h)} high, one behind the other", f"two equal towers {N(h)} tall with a gap between them"])
    elif mode == "double":
        a = between(rng, 2, 4)
        heights = [a, 2 * a] if chance(rng) else [2 * a, a]
        big = names[0] if heights[0] > heights[1] else names[1]
        small = names[1] if big == names[0] else names[0]
        if axis == "x":
            text = pick(rng, [f"two towers, the {big} one twice as tall as the {small} one", f"a tower {N(a)} high with another twice its height to its {big}"])
        else:
            text = pick(rng, [f"two towers in a line away from me, the {big} twice as tall as the {small}"])
    elif mode == "increasing":
        a = between(rng, 1, 4)
        steps = [between(rng, 1, 2), between(rng, 1, 2)]
        heights = [a, a + steps[0], a + steps[0] + steps[1]]
        down = chance(rng)
        if down:
            heights = heights[::-1]
        if axis == "x":
            text = pick(rng, [f"three towers getting {'shorter' if down else 'taller'} from left to right", f"a row of three towers, each one {'shorter' if down else 'taller'} than the one on its left", f"three towers in a row, the {'tallest on the left' if down else 'tallest on the right'} and each step down toward the {'right' if down else 'left'}"])
        else:
            text = pick(rng, [f"three towers in a line coming toward me, each one {'shorter' if down else 'taller'} than the one behind it"])
    else:
        a, c = between(rng, 1, 4), between(rng, 1, 4)
        mid = max(a, c) + between(rng, 2, 4)
        heights = [a, mid, c]
        text = pick(rng, ["three towers in a row with the tallest in the middle", f"three towers, the middle one {N(mid)} high and taller than the two outside ones", "a row of three towers where the middle one stands out tallest"])
        if axis == "y":
            text += pick(rng, [", in a line going away from me", ", one behind the other"])
    placed = towers_row(rng, heights, axis, gap)
    [placed] = fit([placed], rng)
    return record("scene", union(placed), [polish(text, rng)], parts=placed, mode=mode, heights=heights)


def gen_scene_around(rng):
    ck = pick(rng, ["table", "house", "pyramid", "platform"])  # a flat figure has no corners to speak of
    C = rand_part(rng, ck, small=True)
    cp = Placed(C, 0, 0)
    y0, y1, x0, x1 = cp.foot
    g = between(rng, 1, 2)
    h = between(rng, 2, 6)
    corners = {"back left": (y0 - g - 1, x0 - g - 1), "back right": (y0 - g - 1, x1 + g + 1), "front left": (y1 + g + 1, x0 - g - 1), "front right": (y1 + g + 1, x1 + g + 1)}
    which = pick(rng, [list(corners), list(corners), ["back left", "front right"], ["back right", "front left"]])
    towers = [Placed(Part("tower", {"h": h, "t": 1}), *corners[c]) for c in which]
    placed = [cp] + towers
    need(apart(placed), "around")
    [placed] = fit([placed], rng)
    c = describe(C, rng)
    if len(which) == 4:
        text = pick(rng, [f"four towers around {c}, one at each corner", f"{c} with a tower {N(h)} high near each of its corners", f"put {c} in the middle and a tower at each of its four corners, not touching it"])
    else:
        text = pick(rng, [f"{c} with a tower {N(h)} high off its {which[0]} corner and another off its {which[1]} corner", f"{c}, and two towers at opposite corners: {which[0]} and {which[1]}"])
    return record("scene", union(placed), [polish(text, rng)], parts=placed, corners=which, tower_h=h)


def gen_scene_corners(rng):
    ka = rand_kind(rng)
    kb = rand_kind(rng, exclude=(ka,))
    A, B = rand_part(rng, ka, small=True), rand_part(rng, kb, small=True)
    A = A.turned(-A.k) if ka in ORIENTED and ka not in ("stairs", "post") else A
    B = B.turned(-B.k) if kb in ORIENTED and kb not in ("stairs", "post") else B
    corners = ["back left", "back right", "front left", "front right"]
    ca = pick(rng, corners)
    mode = pick(rng, ["two corners", "corner and middle"])

    def at_corner(part, name):
        y0, y1, x0, x1 = local_foot(part)
        y = 1 - y0 if name.startswith("back") else SIZE - 2 - y1
        x = 1 - x0 if name.endswith("left") else SIZE - 2 - x1
        return Placed(part, y, x)

    pa = at_corner(A, ca)
    if mode == "two corners":
        cb = pick(rng, [c for c in corners if c != ca])
        pb = at_corner(B, cb)
    else:
        pb = centred_at(B, (SIZE - 1) / 2, (SIZE - 1) / 2)
    need(apart([pa, pb], sep=3), "corners")
    a, b = describe(A, rng), describe(B, rng)
    if mode == "two corners":
        text = pick(rng, [f"{a} in the {ca} corner and {b} in the {cb} corner", f"put {b} in the {cb} corner of the floor and {a} in the {ca} corner", f"{a} over in the {ca} corner, {b} in the {cb} one"])
    else:
        text = pick(rng, [f"{b} in the middle of the floor and {a} in the {ca} corner", f"put {a} in the {ca} corner, and {b} right in the centre"])
    text += pick(rng, ["", " (front is the side nearest me)"]) if "front" in text else ""
    return record("scene", pa.cells | pb.cells, [polish(text, rng)], parts=[pa, pb], mode=mode)


def yard_cells(side_y, side_x, H, gate_side, tower_extra):
    """A square ring of walls H high (side_y x side_x outside), a gate 1 wide and 2 high (or H) in the middle of
    one wall, and a tower H + tower_extra high at each corner (tower_extra 0: none)."""
    cells = {(z, i, j) for z in range(H) for i in range(side_y) for j in range(side_x) if i in (0, side_y - 1) or j in (0, side_x - 1)}
    i, j = house_door(side_x, side_y, gate_side)
    cells -= {(z, i, j) for z in range(min(2, H))}
    if tower_extra:
        for i, j in ((0, 0), (0, side_x - 1), (side_y - 1, 0), (side_y - 1, side_x - 1)):
            cells |= {(z, i, j) for z in range(H, H + tower_extra)}
    return cells


def gen_scene_big(rng):
    """Scenes of 60-115 robots: a walled yard with a gate and corner towers, a big house with towers at its
    sides, a street of small houses."""
    mode = pick(rng, ["yard", "yard", "house_towers", "street"])
    if mode == "yard":
        sy, sx, H = between(rng, 5, 9), between(rng, 5, 9), between(rng, 2, 3)
        gate = pick(rng, ["front", "front", "back", "left", "right"])
        e = between(rng, 0, 2)
        part = Part("custom", {"cells": tuple(sorted(yard_cells(sy, sx, H, gate, e)))})
        [[placed]] = fit([[Placed(part, 0, 0)]], rng)
        where = {"front": pick(rng, ["facing me", "in the wall nearest me", "on my side"]), "back": "in the far wall", "left": "in the left wall", "right": "in the right wall"}[gate]
        towers = pick(rng, [" and a tower at each corner", f" with corner towers {N(H + e)} high", " and a taller post at every corner"]) if e else ""
        text = pick(rng, [
            f"build a walled yard {N(sx)} wide and {N(sy)} deep, walls {N(H)} high, with a gate {where}{towers}",
            f"a square enclosure of walls {N(H)} high with an opening {where}{towers}",
            f"make a courtyard: a ring of walls {N(H)} high, {N(sx)} by {N(sy)}, gate {where}{towers}",
        ])
        params = {"yard": [sy, sx, H], "gate": gate, "tower_extra": e}
        return record("scene", placed.cells, [polish(text, rng)], parts=[Placed(Part("custom", {"cells": ()}), placed.y, placed.x)], mode=mode, **params)
    if mode == "house_towers":
        house = Part("house", {"w": between(rng, 4, 5), "d": between(rng, 4, 5), "h": 3, "roof": True, "door": "front"})
        hp = Placed(house, 0, 0)
        h = between(rng, 4, 7)
        t = 2 if chance(rng, 0.4) else 1
        tower = Part("tower", {"h": h, "t": t})
        gap = between(rng, 1, 2)
        placed = [next_to(hp, tower, "left", gap), hp, next_to(hp, tower, "right", gap)]
        need(apart(placed))
        [placed] = fit([placed], rng)
        one = "thick tower" if t == 2 else "tower"
        text = pick(rng, [
            f"a big house with a flat roof and its door facing me, with a {one} {N(h)} high on each side",
            f"build a house with a roof, and two {one}s {N(h)} high, one on its left and one on its right",
        ])
        return record("scene", union(placed), [polish(text, rng)], parts=placed, mode=mode)
    n = between(rng, 2, 3)
    houses = [Part("house", {"w": between(rng, 3, 4), "d": between(rng, 3, 4), "h": 2, "roof": chance(rng, 0.8), "door": "front"}) for _ in range(n)]
    gap = between(rng, 1, 2)
    placed = [Placed(houses[0], 0, 0)]
    for hpart in houses[1:]:
        placed.append(next_to(placed[-1], hpart, "right", gap))
    placed = [p.moved(dy=-local_foot(p.part)[1]) for p in placed]  # front walls in one line
    need(apart(placed))
    [placed] = fit([placed], rng)
    roofless = [i for i, hpart in enumerate(houses) if not hpart.params["roof"]]
    which = ["left", "middle", "right"] if n == 3 else ["left", "right"]
    if roofless:
        note = ", all with flat roofs except the " + " and ".join(which[i] for i in roofless) + (" one" if len(roofless) == 1 else " ones")
        if len(roofless) == n:
            note = ", none of them with a roof"
    else:
        note = pick(rng, [", all with flat roofs", ""])
    text = pick(rng, [
        f"a street of {N(n)} small houses side by side, doors facing me{note}",
        f"build {N(n)} little houses in a row from left to right with their doors toward me{note}",
    ])
    return record("scene", union(placed), [polish(text, rng)], parts=placed, mode=mode)


# ---------------------------------------------------------------------------------------------
# Edits (category 2)


def edit_record(rng, a_placed, b_placed, first, second, extra_turns=(), park_side_text=True, extra=None, **info):
    """Fit A and B together, pad with parked robots, write the turns: first builds A (with the parking
    clause when A has parked robots), second asks for the change (with a leftover clause when only B parks)."""
    extra = between(rng, 0, 3) if extra is None else extra
    if not park_side_text:  # no words for where leftovers go: A has a parked row, and leftovers join it
        extra = max(extra, 1)
    a_struct, b_struct = union(a_placed), union(b_placed)
    n = max(len(a_struct), len(b_struct)) + extra - min(len(a_struct), len(b_struct))
    rows = rows_for(n)
    need(rows <= MAX_ROWS, "too many parked")
    side = pick(rng, SIDES) if rows else None
    a_placed, b_placed = fit([a_placed, b_placed], rng, reserve=(side, rows) if rows else None)
    a_cells, b_cells, park = with_parking(rng, union(a_placed), union(b_placed), extra=extra, side=side)
    if park["a"] > 0:
        first = first + park_clause(rng, park["side"])
    elif park["b"] > 0 and park_side_text:
        second = second + leftover_clause(rng, park["side"])
    turns = [polish(first, rng)] + [polish(t, rng) for t in extra_turns] + [polish(second, rng)]
    out = record("edit", b_cells, turns, start=a_cells, parts=b_placed, parking=park, start_parts=[p.meta() for p in a_placed], **info)
    out["params"]["scale"] = max([p.part.scale for p in a_placed + b_placed] + [1])
    return out


def gen_edit_figure(rng):
    s = pick_scale(rng, 0.35, 0.0)  # a figure 3x scaled is taller than the world
    fig = rand_part(rng, "figure").scaled(s)
    fig = fig.turned(-fig.k)  # facing the viewer: side-on figures do not read at a glance
    l, r, legs = fig.params["left"], fig.params["right"], fig.params["legs"]
    mode = pick(rng, ["both", "both", "one", "one"] + (["legs"] if s == 1 else []))
    sl, sr = arm_side(-1, fig.k), arm_side(1, fig.k)
    if mode == "both":
        p = pick(rng, [q for q in ARM_CELLS if not (l == q and r == q)])
        new = fig.changed(left=p, right=p)
        second = pick(rng, {
            "out": ["now stretch both arms straight out", "put the arms out to the sides", "now arms straight out sideways"],
            "up": ["now raise both arms", "arms up!", "lift both arms up in the air"],
            "down": ["now lower the arms", "put both arms down by its sides", "hands down, both of them"],
        }[p])
    elif mode == "one":
        s = pick(rng, [-1, 1])
        cur = l if s == -1 else r
        p = pick(rng, [q for q in ARM_CELLS if q != cur])
        new = fig.changed(**({"left": p} if s == -1 else {"right": p}))
        side = sl if s == -1 else sr
        keep = sr if s == -1 else sl
        verb = {"out": "stick {} straight out", "up": "raise {}", "down": "lower {}"}[p].format(f"the arm on the {side}")
        second = pick(rng, [f"now {verb}", f"{verb} and leave the one on the {keep} as it is", f"just {verb}"])
    else:
        dl = pick(rng, [1, -1]) if legs == 2 else -1
        new = fig.changed(legs=legs + dl)
        second = pick(rng, ["make it taller by giving it longer legs", "now make its legs one longer"] if dl > 0 else ["make its legs one shorter", "shorten the legs by one"])
    need(new.params != fig.params)
    a = [Placed(fig, 0, 0)]
    b = [Placed(new, 0, 0)]
    first = pick(rng, ["build ", "make ", ""]) + describe(fig, rng)
    return edit_record(rng, a, b, first, second, edit=mode)


def gen_edit_house(rng):
    w, d = between(rng, 3, 5), between(rng, 3, 5)
    h = between(rng, 2, 3)
    roof = chance(rng, 0.7)
    door = "front" if chance(rng, 0.85) else None
    s = pick_scale(rng, 0.25, 0.1)
    if s > 1:
        w, d, h = min(w, 9 // s + (1 if s == 2 else 0)), min(d, 9 // s + (1 if s == 2 else 0)), 2
    house = Part("house", {"w": w, "d": d, "h": h, "roof": roof, "door": door}, scale=s)
    edits = []
    if roof:
        edits += ["roof_off"]
    else:
        edits += ["roof_on"]
    if door:
        edits += ["door_move", "door_move", "door_close"]
    else:
        edits += ["door_add"]
    if h == 2 and s == 1:
        edits += ["taller"]
    if roof and h == 2 and w <= 4 and d <= 4 and s == 1:
        edits += ["second_floor"]
    edit = pick(rng, edits)
    if edit == "roof_off":
        new = house.changed(roof=False)
        second = pick(rng, ["now take the roof off", "remove the roof", "lift the roof off, keep the walls"])
    elif edit == "roof_on":
        new = house.changed(roof=True)
        second = pick(rng, ["now put a flat roof on it", "cover the top with a roof", "give it a flat roof"])
    elif edit == "door_move":
        to = pick(rng, ["back", "back", "left", "right"])
        new = house.changed(door=to)
        second = pick(rng, {
            "back": ["move the door to the other side", "put the door on the back wall instead", "move the doorway round to the side facing away from me"],
            "left": ["move the door to the left wall", "put the doorway on the left side instead"],
            "right": ["move the door to the right wall", "put the doorway on the right side instead"],
        }[to])
    elif edit == "door_close":
        new = house.changed(door=None)
        second = pick(rng, ["fill in the doorway", "close up the door", "wall up the door so there's no way in"])
    elif edit == "door_add":
        new = house.changed(door="front")
        second = pick(rng, ["cut a door in the wall facing me", "add a doorway on the side nearest me", "make a door in the front wall"])
    elif edit == "taller":
        new = house.changed(h=3)
        second = pick(rng, ["make the walls one higher", "build the walls up by one", "make it one taller"]) + (", roof and all" if roof and chance(rng) else "")
    else:
        top = chance(rng)
        new = house.changed(upper=2, upper_roof=top)
        second = pick(rng, ["add a second floor on top", "build another storey on top of it"]) + (pick(rng, [" with its own roof", ", roofed too"]) if top else pick(rng, [", open at the top", " without a roof"]))
    first = pick(rng, ["build ", "make ", ""]) + describe(house, rng)
    return edit_record(rng, [Placed(house, 0, 0)], [Placed(new, 0, 0)], first, second, edit=edit)


TOWER_POS = {"x": ("left", "middle", "right"), "y": ("far", "middle", "near")}
TOWER_POS_WORDS = {"left": "left", "middle": "middle", "right": "right", "far": "farthest", "near": "nearest"}


def tower_names(n, axis):
    pos = TOWER_POS[axis]
    return [pos[0], pos[2]] if n == 2 else list(pos)


def the_tower(name, axis):
    if axis == "x":
        return f"the {name} tower"
    return {"far": "the tower farthest from me", "near": "the tower nearest me", "middle": "the middle tower"}[name]


def towers_text(rng, heights, axis):
    hs = [N(h) for h in heights]
    if axis == "x":
        if len(heights) == 2:
            return pick(rng, [f"two towers side by side, the left one {hs[0]} high and the right one {hs[1]} high", f"a tower {hs[0]} high, and to its right a tower {hs[1]} high"])
        return pick(rng, [f"three towers in a row, from left to right {hs[0]}, {hs[1]} and {hs[2]} high", f"a row of three towers: {hs[0]} high on the left, {hs[1]} in the middle, {hs[2]} on the right"])
    if len(heights) == 2:
        return pick(rng, [f"two towers, one behind the other: {hs[0]} high at the back and {hs[1]} high in front", f"a tower {hs[0]} high, and in front of it, nearer me, one {hs[1]} high"])
    return pick(rng, [f"three towers in a line coming toward me: {hs[0]}, {hs[1]} and {hs[2]} high from back to front"])


def gen_edit_towers(rng):
    n = between(rng, 2, 3)
    axis = "x" if chance(rng, 0.7) else "y"
    heights = [between(rng, 2, 7) for _ in range(n)]
    gap = between(rng, 2, 3)
    names = tower_names(n, axis)
    edit = pick(rng, ["taller", "shorter", "match", "all_tallest", "all_shortest", "swap", "remove", "add", "move_out"])
    new = list(heights)
    i = between(rng, 0, n - 1)
    ti = the_tower(names[i], axis)
    placed_a = towers_row(rng, heights, axis, gap)
    removed = None
    if edit == "taller":
        k = between(rng, 1, 3)
        need(heights[i] + k <= 9)
        new[i] += k
        second = pick(rng, [f"make {ti} {N(k)} taller", f"add {N(k)} more on top of {ti}", f"stack {N(k)} more onto {ti}"])
    elif edit == "shorter":
        k = between(rng, 1, 3)
        need(heights[i] - k >= 1)
        new[i] -= k
        second = pick(rng, [f"take {N(k)} off the top of {ti}", f"make {ti} {N(k)} shorter", f"lower {ti} by {N(k)}"])
    elif edit == "match":
        j = pick(rng, [q for q in range(n) if q != i])
        need(heights[i] != heights[j])
        new[i] = heights[j]
        tj = the_tower(names[j], axis)
        second = pick(rng, [f"make {ti} as tall as {tj}", f"make {ti} match {tj}", f"change {ti} so it's the same height as {tj}"])
    elif edit == "all_tallest":
        need(len(set(heights)) > 1)
        new = [max(heights)] * n
        second = pick(rng, ["make them all as tall as the tallest one", "bring every tower up to the height of the tallest", "make all the towers the same height as the biggest"])
    elif edit == "all_shortest":
        need(len(set(heights)) > 1)
        new = [min(heights)] * n
        second = pick(rng, ["make them all as short as the shortest one", "cut every tower down to the height of the shortest"])
    elif edit == "swap":
        i, j = 0, n - 1
        need(heights[i] != heights[j])
        new[i], new[j] = heights[j], heights[i]
        second = pick(rng, [f"swap the heights of {the_tower(names[0], axis)} and {the_tower(names[-1], axis)}", f"make {the_tower(names[0], axis)} the height of {the_tower(names[-1], axis)} and the other way round"])
    elif edit == "remove":
        removed = i
        second = pick(rng, [f"take {ti} away completely", f"remove {ti}", f"knock down {ti}, the others stay"])
    elif edit == "add":
        h = between(rng, 1, 7)
        end = pick(rng, [0, n - 1])
        where = {"x": ("left end of the row", "right end of the row"), "y": ("far end of the row, farthest from me", "near end of the row, nearest me")}[axis][0 if end == 0 else 1]
        second = pick(rng, [f"add another tower {N(h)} high at the {where}, with the same gap", f"put one more tower, {N(h)} high, on the {where}"])
        rel = ("left" if end == 0 else "right") if axis == "x" else ("behind" if end == 0 else "front")
        anchor = placed_a[end]
        added = next_to(anchor, Part("tower", {"h": h, "t": 1}), rel, gap)
        placed_b = placed_a + [added]
        return edit_record(rng, placed_a, placed_b, towers_text(rng, heights, axis), second, edit=edit)
    else:
        end = pick(rng, [0, n - 1])
        k = between(rng, 1, 3)
        word = ("left" if end == 0 else "right") if axis == "x" else ("away from me" if end == 0 else "toward me")
        second = pick(rng, [f"move {the_tower(names[end], axis)} {N(k)} further {'to the ' + word if axis == 'x' else word}", f"slide {the_tower(names[end], axis)} {N(k)} {'to the ' + word if axis == 'x' else word}, away from the others"])
        sign = -1 if end == 0 else 1
        placed_b = list(placed_a)
        placed_b[end] = placed_a[end].moved(dy=sign * k if axis == "y" else 0, dx=sign * k if axis == "x" else 0)
        return edit_record(rng, placed_a, placed_b, towers_text(rng, heights, axis), second, edit=edit)
    if removed is not None:
        placed_b = [p for q, p in enumerate(placed_a) if q != removed]
    else:
        need(new != heights)
        placed_b = [p.swapped(Part("tower", {"h": h, "t": 1})) for p, h in zip(placed_a, new)]
    return edit_record(rng, placed_a, placed_b, towers_text(rng, heights, axis), second, edit=edit, heights=[heights, new])


def gen_edit_wall(rng):
    L, H = between(rng, 3, 7), between(rng, 1, 3)
    along_y = chance(rng, 0.3)
    k = 1 if along_y else 0
    ends = {"lo": "far end" if along_y else "left end", "hi": "end nearest me" if along_y else "right end"}
    gate = 0
    if chance(rng, 0.3) and L >= 5 and H >= 2:
        gate = 1 if L % 2 else 2
    wall = Part("wall", {"length": L, "height": H, "gate": gate}, k)
    edits = ["higher"] + ([] if gate else ["longer", "longer_both"] + (["shorter"] if L >= 4 else []) + (["lower"] if H >= 2 else []))
    if not gate and L >= 5 and H >= 2:
        edits.append("gate_cut")
    if gate:
        edits.append("gate_fill")
    edits.append("tower_end")
    edit = pick(rng, edits)
    a = Placed(wall, 0, 0)
    run = pick(rng, [" running away from me", " running front to back"]) if along_y else pick(rng, ["", " running left to right"])
    first = pick(rng, [f"build a wall {N(L)} long and {N(H)} high{run}", f"a straight wall{run}, {N(L)} long, {N(H)} high"])
    if gate:
        first += pick(rng, [" with a gate in the middle", " with an opening through the middle"])
    lo_shift = (0, -1) if not along_y else (-1, 0)  # where the lo end moves when the wall grows there (dy, dx)
    if edit in ("longer", "shorter"):
        kk = between(rng, 1, 2) if edit == "longer" else 1
        need(L + kk <= 9 if edit == "longer" else L - kk >= 3)
        which = pick(rng, ["lo", "hi"])
        dl = kk if edit == "longer" else -kk
        new = wall.changed(length=L + dl)
        b = Placed(new, a.y, a.x)
        if which == "lo":
            b = b.moved(dy=lo_shift[0] * dl, dx=lo_shift[1] * dl)
        verb = "longer" if edit == "longer" else "shorter"
        second = pick(rng, [f"make the wall {N(kk)} {verb} at the {ends[which]}", f"{'extend' if edit == 'longer' else 'shorten'} the wall by {N(kk)} at the {ends[which]}"])
    elif edit == "longer_both":
        need(L + 2 <= 9)
        new = wall.changed(length=L + 2)
        b = Placed(new, a.y, a.x).moved(dy=lo_shift[0], dx=lo_shift[1])
        second = pick(rng, ["make the wall one longer at each end", "add one to both ends of the wall"])
    elif edit == "higher":
        need(H + 1 <= 4)
        new = wall.changed(height=H + 1)
        b = Placed(new, a.y, a.x)
        second = pick(rng, ["make the wall one higher", "add another layer on top of the wall", "raise the wall by one"])
    elif edit == "lower":
        new = wall.changed(height=H - 1)
        b = Placed(new, a.y, a.x)
        second = pick(rng, ["make the wall one lower", "take the top layer off the wall"])
    elif edit == "gate_cut":
        g = 1 if L % 2 else 2
        new = wall.changed(gate=g)
        b = Placed(new, a.y, a.x)
        second = pick(rng, ["cut an opening through the middle of the wall", "put a gate in the middle of the wall", "make a doorway right in the middle"])
    elif edit == "gate_fill":
        new = wall.changed(gate=0)
        b = Placed(new, a.y, a.x)
        second = pick(rng, ["fill in the gap in the wall", "close up the opening", "wall up the gate so it's solid"])
    else:
        which = pick(rng, ["lo", "hi"])
        e = between(rng, 1, 3)
        cells = wall_with_towers(L, H, gate, ("left",) if which == "lo" else ("right",), e)
        new = Part("custom", {"cells": tuple(sorted(cells))}, k)
        b = Placed(new, a.y, a.x)
        second = pick(rng, [f"add a tower {N(H + e)} high at the {ends[which]} of the wall", f"put a tower at the {ends[which]}, {N(e)} higher than the wall"])
    return edit_record(rng, [a], [b], first, second, edit=edit)


def gen_edit_table(rng):
    mode = pick(rng, ["table", "table", "chair"])
    if mode == "table":
        a, b, legs = between(rng, 3, 4), between(rng, 2, 4), between(rng, 1, 3)
        if chance(rng, 0.4):
            b = a
        t = Part("table", {"a": a, "b": b, "legs": legs})
        edit = pick(rng, (["raise"] if legs < 3 else []) + (["lower"] if legs > 1 else []) + ["to_chair"])
        pa = Placed(t, 0, 0)
        if edit == "raise":
            pb = Placed(t.changed(legs=legs + 1), 0, 0)
            second = pick(rng, ["make the legs one longer", "raise the table by one", "put the table up one level on longer legs"])
        elif edit == "lower":
            pb = Placed(t.changed(legs=legs - 1), 0, 0)
            second = pick(rng, ["make the legs one shorter", "lower the table by one"])
        else:
            need(a == b, "a chair needs a square seat")
            side = pick(rng, ["back", "left", "right", "front"])
            back = between(rng, 1, 3)
            k = {"back": 0, "right": 1, "front": 2, "left": 3}[side]  # the chair's back on `side`
            chair = Part("chair", {"a": a, "legs": legs, "back": back}, k)
            pb = centred_at(chair, *pa.centre)
            need(pb.cells >= pa.cells, "chair must keep the table")
            edge = {"back": "the edge farthest from me", "front": "the edge nearest me", "left": "the left edge", "right": "the right edge"}[side]
            second = pick(rng, [f"add a backrest {N(back)} high along {edge} so it becomes a chair", f"turn it into a chair: put a back {N(back)} high on {edge}"])
        first = pick(rng, ["build ", "make ", ""]) + describe(t, rng)
        return edit_record(rng, [pa], [pb], first, second, edit=edit)
    a, legs, back = between(rng, 3, 4), between(rng, 1, 3), between(rng, 1, 3)
    chair = Part("chair", {"a": a, "legs": legs, "back": back}, between(rng, 0, 3))
    pa = Placed(chair, 0, 0)
    edit = pick(rng, ["back_off", "back_taller"] + (["back_shorter"] if back > 1 else []))
    if edit == "back_off":
        t = Part("table", {"a": a, "b": a, "legs": legs})
        pb = centred_at(t, *pa.centre)
        need(pb.cells <= pa.cells)
        second = pick(rng, ["take the back off so it's a table", "remove the backrest", "turn the chair into a table by taking its back away"])
    elif edit == "back_taller":
        need(back + 1 <= 4)
        pb = Placed(chair.changed(back=back + 1), 0, 0)
        second = pick(rng, ["make the backrest one taller", "raise the back of the chair by one"])
    else:
        pb = Placed(chair.changed(back=back - 1), 0, 0)
        second = pick(rng, ["make the backrest one shorter", "take one off the top of the chair's back"])
    face = pick(rng, FACE_WORDS[chair.facing])
    first = pick(rng, [f"build a chair {face}", f"make a chair with a back {N(back)} high, {face}"])
    return edit_record(rng, [pa], [pb], first, second, edit=edit)


def gen_edit_turn(rng):
    kind = pick(rng, ["bench", "chair", "chair", "stairs", "post"])
    part = rand_part(rng, kind).scaled(pick_scale(rng, 0.3, 0.1 if kind != "bench" else 0.0))
    k0 = between(rng, 0, 3)
    part = part.turned(k0 - part.k)
    with_tower = kind in ("chair", "bench") and chance(rng, 0.35)
    if with_tower:
        towards = pick(rng, [s for s in FACING if s != part.facing])
        rel = {"front": "front", "back": "behind", "left": "left", "right": "right"}[towards]
        k1 = FACING.index(towards)
    else:
        k1 = pick(rng, [q for q in range(4) if q != k0])
    new = part.turned(k1 - k0)
    pa = Placed(part, 0, 0)
    pb = Placed(new, 0, 0) if kind == "post" else centred_at(new, *pa.centre)  # a post swings its arm about its column
    a_list, b_list = [pa], [pb]
    if with_tower:
        tower = Part("tower", {"h": between(rng, 3, 6), "t": 1})
        both = Placed(Part("custom", {"cells": tuple(sorted(pa.cells | pb.cells))}), 0, 0)
        tp = next_to(both, tower, rel, between(rng, 1, 2))
        need(apart([both, tp]))
        a_list, b_list = [pa, tp], [pb, tp]
    noun = SHORT[kind]
    if kind == "stairs":
        rise = DIR_WORDS[FACING[(k1 + 2) % 4]]
        first = describe(part, rng)
        second = pick(rng, [f"turn the stairs so they go up {rise}", f"now make the staircase rise {rise} instead"])
    elif kind == "post":
        first = describe(part, rng)
        second = pick(rng, [f"swing the arm round so it points {DIR_WORDS[new.facing]}", f"now make the arm stick out {DIR_WORDS[new.facing]}"])
    else:
        face0 = pick(rng, FACE_WORDS[part.facing])
        if with_tower:
            first = f"{describe(part, rng)} {face0}, with {describe(tp.part, rng)} {rel_words(rel, 'it', rng, True)}"
            second = pick(rng, [f"turn the {noun} so it faces the tower", f"now turn the {noun} to face the tower"])
        else:
            first = f"{describe(part, rng)} {face0}"
            face1 = pick(rng, FACE_WORDS[new.facing])
            if k1 == (k0 + 2) % 4:
                second = pick(rng, [f"turn the {noun} around so it's {face1}", "spin it round to face the other way", f"now turn it around, {face1}"])
            else:
                second = pick(rng, [f"turn the {noun} so it's {face1}", f"rotate it to be {face1}"])
    return edit_record(rng, a_list, b_list, first, second, edit="turn", kind_turned=kind)


def gen_edit_scene_move(rng):
    xp, yp, rel, gap = pair_scene(rng)
    x, y = the(xp.part), the(yp.part)
    edit = pick(rng, ["other_side", "swap", "closer", "farther", "relation"])
    if edit == "other_side":
        nx = next_to(yp, xp.part, OPPOSITE[rel], gap)
        second = pick(rng, [f"now move {x} to the other side of {y}", f"put {x} on the opposite side of {y}"])
        b = [nx, yp]
    elif edit == "swap":
        cy, cx = yp.centre
        ny, nx = xp.centre
        b = [centred_at(xp.part, cy, cx), centred_at(yp.part, ny, nx)]
        second = pick(rng, [f"swap {x} and {y}", f"now swap them round: {x} where {y} is and {y} where {x} is", "swap the two of them over"])
    elif edit == "closer":
        need(gap >= 2)
        b = [next_to(yp, xp.part, rel, 1), yp]
        second = pick(rng, [f"move {x} closer to {y}, leaving just a one-space gap", f"bring {x} right up to {y}, one space apart"])
    elif edit == "farther":
        k = between(rng, 2, 3)
        b = [next_to(yp, xp.part, rel, gap + k), yp]
        second = pick(rng, [f"move {x} {N(k)} farther away from {y}", f"push {x} {N(k)} further from {y}"])
    else:
        new_rel = pick(rng, [r for r in REL_AXIS if r not in (rel, OPPOSITE[rel])])
        b = [next_to(yp, xp.part, new_rel, between(rng, 1, 3), rng), yp]
        oriented = yp.part.kind in ORIENTED
        second = pick(rng, [f"put {x} {rel_words(new_rel, y, rng, oriented)} instead", f"now move {x} so it's {rel_words(new_rel, y, rng, oriented)}"])
    need(apart(b), "touch")
    if edit == "relation":
        need(overlap_ok(b[0], b[1], new_rel), "relation")
    return edit_record(rng, [xp, yp], b, pair_text(rng, xp, yp, rel), second, edit=edit, relation=rel)


def gen_edit_arch(rng):
    g, p = between(rng, 1, 3), between(rng, 2, 4)
    k = int(pick(rng, [0, 1]))
    arch = Part("arch", {"gap": g, "p": p}, k)
    pa = Placed(arch, 0, 0)
    edit = pick(rng, ["wider", "narrower", "taller", "shorter", "beam_off", "beam_on"])
    hi_word = "right" if k == 0 else "near"  # local +x: the post that moves when the gap changes
    if edit == "wider":
        need(g + 1 <= 3)
        new = arch.changed(gap=g + 1)
        pb = Placed(new, 0, 0)
        second = pick(rng, [f"move the {hi_word} post one further out so the opening is one wider", f"widen the opening by one by moving the {hi_word} leg out"])
    elif edit == "narrower":
        need(g >= 2)
        new = arch.changed(gap=g - 1)
        pb = Placed(new, 0, 0)
        second = pick(rng, [f"move the {hi_word} post one closer in so the opening is one narrower", f"make the gap under the arch one narrower by moving the {hi_word} leg in"])
    elif edit == "taller":
        need(p + 1 <= 5)
        pb = Placed(arch.changed(p=p + 1), 0, 0)
        second = pick(rng, ["make the arch one taller", "make both legs one longer so the arch stands higher", "raise the beam by one, legs and all"])
    elif edit == "shorter":
        need(p >= 2)
        pb = Placed(arch.changed(p=p - 1), 0, 0)
        second = pick(rng, ["make the arch one lower", "shorten both legs by one"])
    elif edit == "beam_off":
        posts = Part("custom", {"cells": tuple(sorted(b_arch(g, p) - {(p, 0, j) for j in range(g + 2)}))}, k)
        pb = Placed(posts, 0, 0)
        second = pick(rng, ["take the beam off the top", "remove the top bar, leave the two posts", "lift off the piece across the top"])
    else:
        posts = Part("custom", {"cells": tuple(sorted(b_arch(g, p) - {(p, 0, j) for j in range(g + 2)}))}, k)
        pa, pb = Placed(posts, 0, 0), pa
        first = pick(rng, [f"build two posts {N(p)} high with a gap of {N(g)} between them", f"two towers {N(p)} high, {N(g)} apart"])
        if k == 1:
            first += pick(rng, [", one behind the other", ", in a line going away from me"])
        second = pick(rng, ["now join them with a beam across the top", "put a bar across the top of the two to make an arch", "bridge the two posts at the top"])
        return edit_record(rng, [pa], [pb], first, second, edit=edit)
    first = pick(rng, ["build ", "make ", ""]) + describe(arch, rng) + (pick(rng, [", running left to right", ""]) if k == 0 else pick(rng, [", running away from me", ", its opening facing left and right"]))
    return edit_record(rng, [pa], [pb], first, second, edit=edit)


def gen_edit_stairs(rng):
    width, steps = between(rng, 1, 3), between(rng, 3, 5)
    k = between(rng, 0, 3)
    st = Part("stairs", {"width": width, "steps": steps}, k)
    pa = Placed(st, 0, 0)
    edit = pick(rng, ["more", "fewer", "wider", "narrower"])
    if edit == "more":
        need(steps + 1 <= 6)
        pb = Placed(st.changed(steps=steps + 1), 0, 0)
        second = pick(rng, ["add one more step at the top", "make the staircase one step higher", "extend the stairs by a step at the top"])
    elif edit == "fewer":
        pb = Placed(st.changed(steps=steps - 1), 0, 0)
        second = pick(rng, ["take the top step off", "make the staircase one step lower"])
    elif edit == "wider":
        need(width + 1 <= 4)
        pb = Placed(st.changed(width=width + 1), 0, 0)
        side = arm_side(1, k)  # local +x is where the stairs widen
        second = pick(rng, [f"make the stairs one wider on the {side}", f"widen the staircase by one, on the {side}"])
    else:
        need(width >= 2)
        pb = Placed(st.changed(width=width - 1), 0, 0)
        side = arm_side(1, k)
        second = pick(rng, [f"make the stairs one narrower, trimming the {side}", f"take one off the width of the stairs on the {side}"])
    if "side" in second:
        second = second.replace("on the far side", "on the side away from me").replace("on the near side", "on the side nearest me").replace("trimming the far side", "trimming the side away from me").replace("trimming the near side", "trimming the side nearest me")
    first = pick(rng, ["build ", "make ", ""]) + describe(st, rng)
    return edit_record(rng, [pa], [pb], first, second, edit=edit)


UNDO = ["no, undo that", "actually, put it back how it was", "hmm, I liked it better before. change it back", "undo the last change", "go back to how it was before"]


def gen_edit_undo(rng):
    inner = pick(rng, [gen_edit_figure, gen_edit_house, gen_edit_towers, gen_edit_wall, gen_edit_table, gen_edit_scene_move, gen_edit_arch])(rng)
    a, b = inner["start_cells"], inner["cells"]
    turns = inner["turns"] + [polish(pick(rng, UNDO), rng)]
    params = inner["params"]
    out = record("edit", a, turns, start=b, parking={"side": params["parked"]["side"], "a": params["parked"]["b"], "b": params["parked"]["a"]}, edit="undo", inner=params.get("edit"))
    out["params"]["parts"] = params.get("start_parts", [])
    out["params"]["scale"] = params.get("scale", 1)
    return out


# ---------------------------------------------------------------------------------------------
# Structure to structure (category 3)


def _external():
    out = {}
    for name in ("goal_gen_simple", "goal_gen_medium"):
        try:
            module = __import__(f"beebot.vla.voxel.{name}", fromlist=["GENERATORS"])
            if isinstance(getattr(module, "GENERATORS", None), dict) and module.GENERATORS:
                out[name] = module
        except Exception:  # a sibling tier still in progress: skip it
            continue
    return out


EXTERNAL = _external()


def external_shape(rng):
    """(cells as a local set with z = 0 the floor, text, generator name) from goal_gen_simple / medium."""
    need(EXTERNAL, "no external tier")
    module = EXTERNAL[pick(rng, sorted(EXTERNAL))]
    name = pick(rng, sorted(module.GENERATORS))
    try:
        cells, text, _ = module.GENERATORS[name](rng)
    except Exception:
        raise Reject("external reject")
    cells = np.asarray(cells, dtype=int)
    need(cells.ndim == 2 and len(cells) >= 4 and text_ok(text), "external draw")
    cells = cells - np.array([0, cells[:, 1].min(), cells[:, 2].min()])
    return {tuple(int(v) for v in c) for c in cells}, text.strip(), f"{module.__name__.split('.')[-1]}.{name}"


TURN_INTO = [
    "now turn the {x} into {Y}",
    "take the {x} apart and build {Y} in its place",
    "rebuild it as {Y}",
    "change of plan: make {Y} out of the {x} instead",
    "I'd rather have {Y} there. use the {x}",
    "turn that into {Y}",
]


def gen_retarget_part(rng):
    kx = rand_kind(rng)
    ky = rand_kind(rng, exclude=(kx,))
    s = pick_scale(rng, 0.3, 0.05)
    X, Y = rand_part(rng, kx, small=s > 1).scaled(s), rand_part(rng, ky, small=s > 1).scaled(s)
    X = X.turned(-X.k) if kx in ("house", "chair", "bench", "figure") else X
    Y = Y.turned(-Y.k) if ky in ("house", "chair", "bench", "figure") else Y
    need(max(len(X.cells), len(Y.cells)) <= 3 * min(len(X.cells), len(Y.cells)), "sizes too different")
    px = Placed(X, 0, 0)
    first = pick(rng, ["build ", "make ", ""]) + describe(X, rng)
    if chance(rng):  # in its place (robots on cells both share stay put)
        py = centred_at(Y, *px.centre)
        second = pick(rng, TURN_INTO).format(x=SHORT[kx], Y=describe(Y, rng))
        where = "place"
    else:  # beside where it stands: the footprints do not overlap
        where = pick(rng, ["left", "right", "front", "behind"])
        py = next_to(px, Y, where, between(rng, 1, 2), rng)
        spot = pick(rng, {
            "left": ["just left of where it stands", "to the left of where it is now"],
            "right": ["just right of where it stands", "to the right of where it is now"],
            "front": ["in front of where it stands, nearer me", "a little closer to me than where it is now"],
            "behind": ["behind where it stands, farther from me", "a little farther back than where it is now"],
        }[where])
        y = describe(Y, rng)
        y = y + "," if " with " in y else y
        second = pick(rng, [
            f"take the {SHORT[kx]} apart and build {y} {spot}",
            f"now use the {SHORT[kx]} to make {y} {spot}",
            f"knock down the {SHORT[kx]} and build {y} {spot}",
        ])
    out = edit_record(rng, [px], [py], first, second, edit="retarget", where=where)
    out["kind"] = "retarget"
    return out


def gen_retarget_compose(rng):
    mode = pick(rng, ["wall_to_towers", "towers_to_wall", "flatten", "towers_to_arch", "pyramid_to_stairs", "tower_to_wall"])
    if mode == "wall_to_towers":
        n, h = between(rng, 2, 4), between(rng, 2, 5)
        H = between(rng, 2, 3)
        L = between(rng, 3, 8)
        wall = Part("wall", {"length": L, "height": H})
        pa = Placed(wall, 0, 0)
        tw = towers_row(rng, [h] * n, "x", 1)
        row = Placed(Part("custom", {"cells": tuple(sorted(union(tw)))}), 0, 0)
        pb = centred_at(row.part, *pa.centre)
        first = pick(rng, ["build ", ""]) + describe(wall, rng)
        second = pick(rng, [f"now break the wall up into {N(n)} separate towers {N(h)} high in a row", f"turn the wall into a row of {N(n)} towers, each {N(h)} high, with gaps between"])
    elif mode == "towers_to_wall":
        n, h = between(rng, 2, 4), between(rng, 2, 5)
        tw = towers_row(rng, [h] * n, "x", between(rng, 1, 2))
        pa = Placed(Part("custom", {"cells": tuple(sorted(union(tw)))}), 0, 0)
        L, H = between(rng, 3, 8), between(rng, 1, 3)
        wall = Part("wall", {"length": L, "height": H})
        pb = centred_at(wall, *pa.centre)
        first = pick(rng, [f"build {N(n)} towers {N(h)} high in a row from left to right", f"a row of {N(n)} towers, each {N(h)} tall, side by side with gaps"])
        second = pick(rng, [f"now turn the towers into one solid wall {N(L)} long and {N(H)} high", f"knock them down and build a single wall {N(L)} long, {N(H)} high, where they stood"])
    elif mode == "flatten":
        kx = pick(rng, ["house", "tower", "arch", "table", "pyramid", "figure"])
        X = rand_part(rng, kx, small=True)
        X = X.turned(-X.k) if kx in ORIENTED else X
        pa = Placed(X, 0, 0)
        a, b = between(rng, 2, 5), between(rng, 2, 5)
        pb = centred_at(Part("platform", {"a": a, "b": b}), *pa.centre)
        first = pick(rng, ["build ", ""]) + describe(X, rng)
        second = pick(rng, [f"flatten the {SHORT[kx]} into a flat platform {N(a)} by {N(b)}", f"take it down and lay out a flat slab {N(a)} by {N(b)} where it was"])
    elif mode == "towers_to_arch":
        g, p = between(rng, 1, 3), between(rng, 2, 4)
        posts = b_arch(g, p) - {(p, 0, j) for j in range(g + 2)}
        pa = Placed(Part("custom", {"cells": tuple(sorted(posts))}), 0, 0)
        pb = Placed(Part("arch", {"gap": g, "p": p}), 0, 0)
        first = pick(rng, [f"two towers {N(p)} high with a gap of {N(g)} between them, side by side", f"build two columns {N(p)} tall, {N(g)} apart, left and right"])
        second = pick(rng, ["turn the two towers into an arch by joining their tops", "make an arch out of them: a beam across the top"])
    elif mode == "pyramid_to_stairs":
        pa = Placed(Part("pyramid", {"base": int(pick(rng, [3, 5]))}), 0, 0)
        st = Part("stairs", {"width": between(rng, 1, 3), "steps": between(rng, 3, 5)}, between(rng, 0, 3))
        pb = centred_at(st, *pa.centre)
        first = pick(rng, ["build ", ""]) + describe(pa.part, rng)
        second = pick(rng, TURN_INTO).format(x="pyramid", Y=describe(st, rng))
    else:
        h = between(rng, 3, 8)
        pa = Placed(Part("tower", {"h": h, "t": 1}), 0, 0)
        L, H = between(rng, 3, 7), between(rng, 1, 2)
        wall = Part("wall", {"length": L, "height": H}, int(pick(rng, [0, 1])))
        pb = centred_at(wall, *pa.centre)
        first = pick(rng, ["build ", ""]) + describe(pa.part, rng)
        run = "running left to right" if wall.k == 0 else "running away from me"
        second = pick(rng, [f"now turn the tower into a low wall {N(L)} long and {N(H)} high, {run}", f"take the tower down and make a wall {N(L)} long, {N(H)} high, {run}, in the same spot"])
    out = edit_record(rng, [pa], [pb], first, second, edit=mode)
    out["kind"] = "retarget"
    return out


def gen_retarget_external(rng):
    shape, text, name = external_shape(rng)
    ext = Placed(Part("custom", {"cells": tuple(sorted(shape))}), 0, 0)
    kx = rand_kind(rng)
    X = rand_part(rng, kx)
    X = X.turned(-X.k) if kx in ("house", "chair", "bench", "figure") else X
    need(max(len(shape), len(X.cells)) <= 3 * min(len(shape), len(X.cells)), "sizes too different")
    if chance(rng):  # ours -> theirs
        pa = Placed(X, 0, 0)
        pb = centred_at(ext.part, *pa.centre)
        first = pick(rng, ["build ", ""]) + describe(X, rng)
        second = pick(rng, ["scrap that. {t}", "ok, take that down, and in the same spot: {t}", "change of plan, use the same robots: {t}"]).format(t=text)
    else:
        pa = ext
        pb = centred_at(X, *pa.centre)
        first = text
        second = pick(rng, ["now turn it into {Y}", "take it apart and build {Y} in its place", "rebuild that as {Y}"]).format(Y=describe(X, rng))
    out = edit_record(rng, [pa], [pb], first, second, edit="external", external=name)
    out["kind"] = "retarget"
    return out


# ---------------------------------------------------------------------------------------------
# Tricks and conditionals (category 4)


def gen_trick_too_many(rng):
    k = between(rng, 4, 8)
    ask = pick(rng, ["tower", "tower", "wall", "two_towers", "nothing"])
    if ask == "tower":
        n = between(rng, k + 1, 9)
        part = Part("tower", {"h": k, "t": 1})
        text = pick(rng, [f"build a tower {N(n)} high", f"stack all of you into a tower {N(n)} tall", f"I need a {N(n)}-high tower"])
        reply = pick(rng, [f"There are only {N(k)} of us, so the tallest tower I can make is {N(k)} high.", f"I only have {N(k)} robots, so the tower is {N(k)} high, not {N(n)}."])
    elif ask == "wall":
        n = between(rng, k + 1, 9)
        part = Part("wall", {"length": k, "height": 1}, int(pick(rng, [0, 1])))
        text = pick(rng, [f"build a wall {N(n)} long and one high", f"lay out a low wall {N(n)} long"])
        reply = f"There are only {N(k)} of us, so the wall is {N(k)} long."
    elif ask == "two_towers":
        need(k % 2 == 0)
        n = between(rng, k // 2 + 1, 9)
        h = k // 2
        tw = towers_row(rng, [h, h], "x", between(rng, 1, 3))
        part = Part("custom", {"cells": tuple(sorted(union(tw)))})
        text = pick(rng, [f"build two towers, each {N(n)} high, side by side", f"make two towers {N(n)} tall next to each other"])
        reply = f"With only {N(k)} of us, each tower can be {N(h)} high."
    else:
        n = between(rng, k + 1, 9)
        parts = between(rng, 2, 3)
        cuts = sorted(rng.choice(np.arange(1, k), size=parts - 1, replace=False).tolist())
        heights = [b - a for a, b in zip([0] + cuts, cuts + [k])]
        tw = towers_row(rng, heights, "x", 2)
        [tw] = fit([tw], rng)
        start = union(tw)
        text = pick(rng, [f"build a tower {N(n)} high, and if you can't, don't move at all", f"only if you can make a tower {N(n)} tall, do it; otherwise leave everything as it is"])
        reply = f"There are only {N(k)} of us, not enough for {N(n)}, so I left everything as it was."
        first = towers_text(rng, heights, "x")
        return record("trick", start, [polish(first, rng), polish(text, rng)], start=start, reply=reply, parts=tw, trick="too_many", behaviour="nothing", asked=n, robots=k)
    [[placed]] = fit([[Placed(part, 0, 0)]], rng)
    return record("trick", placed.cells, [polish(text, rng)], reply=reply, parts=[placed], trick="too_many", behaviour="closest", asked=n, robots=k)


def trick_scene(rng):
    """A built scene for tricks: a pair, a single part or a row of towers. (placed list, text)."""
    mode = pick(rng, ["pair", "single", "towers"])
    if mode == "pair":
        xp, yp, rel, _ = pair_scene(rng)
        return [xp, yp], pair_text(rng, xp, yp, rel), {"relation": rel}
    if mode == "single":
        k = rand_kind(rng)
        part = rand_part(rng, k)
        part = part.turned(-part.k) if k in ("house", "chair", "bench", "figure") else part
        return [Placed(part, 0, 0)], pick(rng, ["build ", "make ", ""]) + describe(part, rng), {}
    n = between(rng, 2, 3)
    heights = [between(rng, 1, 7) for _ in range(n)]
    return towers_row(rng, heights, "x", between(rng, 1, 3)), towers_text(rng, heights, "x"), {"heights": heights}


def nothing_record(rng, placed, first, ask, reply, **info):
    [placed] = fit([placed], rng)
    cells = union(placed)
    return record("trick", cells, [polish(first, rng), polish(ask, rng)], start=cells, reply=reply, parts=placed, behaviour="nothing", **info)


def gen_trick_already(rng):
    mode = pick(rng, ["tower_at_least", "house_door", "relation", "arms", "taller_than", "exact_build"])
    if mode == "tower_at_least":
        h = between(rng, 3, 8)
        m = between(rng, 2, h)
        k = rand_kind(rng, exclude=("tower",))
        other = rand_part(rng, k, small=True)
        other = other.turned(-other.k) if k in ORIENTED and k not in ("stairs", "post") else other
        tp = Placed(Part("tower", {"h": h, "t": 1}), 0, 0)
        rel = pick(rng, ["left", "right", "behind", "front"])
        op = next_to(tp, other, rel, between(rng, 1, 3), rng)
        need(apart([tp, op]))
        first = f"build a tower {N(h)} high with {describe(other, rng)} {rel_words(rel, 'it', rng, False)}"
        ask = pick(rng, [f"make sure there's a tower at least {N(m)} high", f"I need a tower of {N(m)} or more somewhere", f"is there a tower at least {N(m)} high? if not, build one"])
        reply = pick(rng, [f"There is already a tower {N(h)} high, so nothing needs to change.", f"The tower is already {N(h)} high."])
        return nothing_record(rng, [tp, op], first, ask, reply, trick="already", mode=mode)
    if mode == "house_door":
        house = rand_part(rng, "house")
        door = pick(rng, ["front", "front", "back", "left", "right"])
        house = house.changed(door=door)
        first = pick(rng, ["build ", ""]) + describe(house, rng)
        ask = pick(rng, ["the house needs a door", "give the house a way in", "put a door on the house"])
        where = {"front": "on the side facing you", "back": "on the back wall", "left": "on the left side", "right": "on the right side"}[door]
        reply = f"The house already has a door, {where}."
        return nothing_record(rng, [Placed(house, 0, 0)], first, ask, reply, trick="already", mode=mode)
    if mode == "relation":
        xp, yp, rel, _ = pair_scene(rng)
        first = pair_text(rng, xp, yp, rel)
        ask = pick(rng, [f"put {the(xp.part)} {rel_words(rel, the(yp.part), rng, yp.part.kind in ORIENTED)}", f"I want {the(xp.part)} {rel_words(rel, the(yp.part), rng, yp.part.kind in ORIENTED)}"])
        reply = f"{the(xp.part).capitalize()} is already there."
        return nothing_record(rng, [xp, yp], first, ask, reply, trick="already", mode=mode)
    if mode == "arms":
        p = pick(rng, list(ARM_CELLS))
        fig = Part("figure", {"left": p, "right": p, "legs": 2}, scale=pick_scale(rng, 0.3, 0.0))
        first = pick(rng, ["build ", ""]) + describe(fig, rng)
        ask = {"out": "stretch both arms out to the sides", "up": "raise both arms", "down": "put both arms down"}[p]
        reply = {"out": "Both arms are already straight out.", "up": "Both arms are already up.", "down": "Both arms are already down."}[p]
        return nothing_record(rng, [Placed(fig, 0, 0)], first, ask, reply, trick="already", mode=mode)
    if mode == "taller_than":
        a, b = between(rng, 3, 8), 0
        b = between(rng, 1, a - 1)
        heights = [a, b] if chance(rng) else [b, a]
        tall = "left" if heights[0] > heights[1] else "right"
        short = "right" if tall == "left" else "left"
        placed = towers_row(rng, heights, "x", between(rng, 1, 3))
        ask = pick(rng, [f"make the {tall} tower taller than the {short} one", f"the {short} tower should be the shorter one"])
        reply = f"The {tall} tower is already the taller one."
        return nothing_record(rng, placed, towers_text(rng, heights, "x"), ask, reply, trick="already", mode=mode)
    k = rand_kind(rng)
    part = rand_part(rng, k)
    part = part.turned(-part.k) if k in ("house", "chair", "bench", "figure") else part
    words = describe(part, rng)
    first = pick(rng, ["build ", ""]) + words
    ask = pick(rng, [f"now build {words}", f"can you make {words}?"])
    reply = pick(rng, ["That's already built, it's right here.", f"{the(part).capitalize()} is already standing, so I left it."])
    return nothing_record(rng, [Placed(part, 0, 0)], first, ask, reply, trick="already", mode=mode)


def gen_trick_negation(rng):
    mode = pick(rng, ["except", "except", "house_no_roof", "house_no_door", "wall_no_gap", "towers_unequal", "tower_at_most", "table_not_square"])
    if mode == "except":
        banned = rand_kind(rng)
        k = rand_kind(rng, exclude=(banned,))
        part = rand_part(rng, k, small=True).scaled(pick_scale(rng, 0.3, 0.1))
        part = part.turned(-part.k) if k in ("house", "chair", "bench", "figure") else part
        text = pick(rng, [f"build anything you like except a {SHORT[banned]}", f"make something, but not a {SHORT[banned]}", f"surprise me, just no {PLURAL[banned]}"])
        reply = f"Here is {describe(part, rng)}."
        placed = [Placed(part, 0, 0)]
    elif mode == "house_no_roof":
        part = rand_part(rng, "house").changed(roof=False)
        text = pick(rng, ["build a small house with no roof", "a house, but leave the roof off", "make the walls of a house, no roof on top"])
        reply = ""
        placed = [Placed(part, 0, 0)]
    elif mode == "house_no_door":
        part = rand_part(rng, "house").changed(door=None, roof=True)
        text = pick(rng, ["build a house without a door", "a house with a roof but no way in", "make a closed house, no doorway"])
        reply = ""
        placed = [Placed(part, 0, 0)]
    elif mode == "wall_no_gap":
        L, H = between(rng, 4, 9), between(rng, 2, 3)
        part = Part("wall", {"length": L, "height": H}, int(pick(rng, [0, 1])))
        text = pick(rng, [f"a wall {N(L)} long and {N(H)} high with no gaps in it", f"build a solid wall {N(L)} long, {N(H)} high, no openings"])
        reply = ""
        placed = [Placed(part, 0, 0)]
    elif mode == "towers_unequal":
        a = between(rng, 1, 6)
        b = between(rng, a + 2, 9) if a + 2 <= 9 else 9
        need(b - a >= 2)
        heights = [a, b] if chance(rng) else [b, a]
        placed = towers_row(rng, heights, "x", between(rng, 1, 3))
        text = pick(rng, ["two towers side by side that are not the same height", "build two towers, and don't make them equal"])
        reply = f"The left one is {N(heights[0])} high and the right one {N(heights[1])}."
    elif mode == "tower_at_most":
        m = between(rng, 2, 4)
        h = between(rng, 2, m)
        placed = [Placed(Part("tower", {"h": h, "t": 1}), 0, 0)]
        text = pick(rng, [f"build a tower, but no taller than {N(m)}", f"a tower that is not more than {N(m)} high"])
        reply = ""
    else:
        a, b = between(rng, 2, 4), between(rng, 3, 5)
        need(a != b)
        placed = [Placed(Part("table", {"a": a, "b": b, "legs": between(rng, 1, 3)}), 0, 0)]
        text = pick(rng, ["build a table whose top is not square", "make a table, long rather than square"])
        reply = f"The top is {N(a)} by {N(b)}."
    [placed] = fit([placed], rng)
    return record("trick", union(placed), [polish(text, rng)], reply=reply, parts=placed, trick="negation", behaviour="build", mode=mode)


def gen_trick_false_premise(rng):
    mode = pick(rng, ["roofless", "absent", "absent", "second", "colour"])
    if mode == "roofless":
        house = rand_part(rng, "house").changed(roof=False)
        first = pick(rng, ["build ", ""]) + describe(house, rng)
        ask = pick(rng, ["take the roof off the house", "now lift the roof off", "make the roof one higher"])
        reply = "The house has no roof, so there is nothing to change."
        return nothing_record(rng, [Placed(house, 0, 0)], first, ask, reply, trick="false_premise", mode=mode)
    placed, first, info = trick_scene(rng)
    kinds = sorted({p.part.kind for p in placed})
    present = " and ".join(f"a {SHORT[k]}" if k not in ("arch",) else "an arch" for k in kinds)
    if mode == "absent":
        absent = rand_kind(rng, exclude=kinds)
        noun = SHORT[absent]
        ask = pick(rng, [f"make the {noun} taller", f"move the {noun} to the left", f"take the {noun} apart", f"turn the {noun} around"])
        reply = f"There's no {noun} here, only {present}."
    elif mode == "second":
        single = [k for k in kinds if sum(p.part.kind == k for p in placed) == 1]
        need(single, "no single kind")
        kind = pick(rng, single)
        noun = SHORT[kind]
        ask = pick(rng, [f"move the second {noun} to the left", f"make the other {noun} match this one", f"take down both {PLURAL[kind]}"])
        reply = f"There is only one {noun} here."
    else:
        noun = SHORT[pick(rng, kinds)]
        colour = pick(rng, ["red", "blue", "green", "yellow"])
        ask = pick(rng, [f"move the {colour} {noun} to the left", f"make the {colour} {noun} taller"])
        reply = f"We are all the same colour, so there is no {colour} {noun}. Which one do you mean?"
    return nothing_record(rng, placed, first, ask, reply, trick="false_premise", mode=mode)


def gen_trick_conditional(rng):
    mode = pick(rng, ["tower", "roof", "unequal"])
    if mode == "tower":
        has = chance(rng)
        k = rand_kind(rng, exclude=("tower",))
        other = rand_part(rng, k, small=True)
        other = other.turned(-other.k) if k in ORIENTED and k not in ("stairs", "post") else other
        op = Placed(other, 0, 0)
        inc, h_new = between(rng, 1, 3), between(rng, 2, 6)
        rel = pick(rng, ["left", "right"])
        if has:
            h = between(rng, 2, 9 - inc)
            tp = next_to(op, Part("tower", {"h": h, "t": 1}), rel, between(rng, 1, 3), rng)
            a, b = [op, tp], [op, tp.swapped(Part("tower", {"h": h + inc, "t": 1}))]
            first = f"build {describe(other, rng)} with a tower {N(h)} high {rel_words(rel, 'it', rng, k in ORIENTED)}"
            reply = f"There was a tower, so I made it {N(inc)} taller."
        else:
            tp = next_to(op, Part("tower", {"h": h_new, "t": 1}), rel, 2)
            a, b = [op], [op, tp]
            first = pick(rng, ["build ", ""]) + describe(other, rng)
            reply = f"There was no tower, so I built one {N(h_new)} high."
        ask = pick(rng, [
            f"if there's a tower, make it {N(inc)} taller; if not, build one {N(h_new)} high, two spaces to the {rel} of the {SHORT[k]}",
            f"check for a tower: if there is one, add {N(inc)} to it, otherwise put a {N(h_new)}-high tower two spaces {rel} of the {SHORT[k]}",
        ])
        need(apart(b))
        out = edit_record(rng, a, b, first, ask, park_side_text=False, extra=between(rng, 0, 2), branch=has)
    elif mode == "roof":
        has = chance(rng)
        house = rand_part(rng, "house").changed(roof=has)
        first = pick(rng, ["build ", ""]) + describe(house, rng)
        ask = pick(rng, ["if the house has a roof, take it off", "take the roof off, if there is one", "if it's got a roof, remove it; otherwise leave it"])
        b = house.changed(roof=False)
        reply = "It had a roof, so I took it off." if has else "It has no roof, so I left it as it is."
        out = edit_record(rng, [Placed(house, 0, 0)], [Placed(b, 0, 0)], first, ask, park_side_text=False, branch=has)
    else:
        n = 2
        a_h = between(rng, 2, 7)
        same = chance(rng, 0.4)
        b_h = a_h if same else pick(rng, [h for h in range(2, 8) if h != a_h])
        heights = [a_h, b_h]
        placed = towers_row(rng, heights, "x", between(rng, 1, 3))
        new = [max(heights)] * 2
        placed_b = [p.swapped(Part("tower", {"h": h, "t": 1})) for p, h in zip(placed, new)]
        ask = pick(rng, ["if the two towers are different heights, make the shorter one match the taller one", "make sure both towers are the same height as the taller one"])
        reply = "They were already the same height, so nothing changed." if same else "The shorter one now matches the taller one."
        out = edit_record(rng, placed, placed_b, towers_text(rng, heights, "x"), ask, park_side_text=False, branch=not same)
        has = not same
    out["kind"] = "trick"
    out["reply"] = reply
    out["params"]["trick"] = "conditional"
    out["params"]["behaviour"] = "nothing" if out["start_cells"] == out["cells"] else "edit"
    return out


# ---------------------------------------------------------------------------------------------
# Registry, checks, sampling


GENERATORS = {
    "scene_pair": (gen_scene_pair, 3.0),
    "scene_row3": (gen_scene_row3, 2.0),
    "scene_facing": (gen_scene_facing, 1.5),
    "scene_wall_towers": (gen_scene_wall_towers, 1.2),
    "scene_between": (gen_scene_between, 1.5),
    "scene_on_top": (gen_scene_on_top, 1.5),
    "scene_compare": (gen_scene_compare, 1.5),
    "scene_around": (gen_scene_around, 1.0),
    "scene_corners": (gen_scene_corners, 1.0),
    "scene_big": (gen_scene_big, 1.2),
    "edit_figure": (gen_edit_figure, 2.0),
    "edit_house": (gen_edit_house, 2.0),
    "edit_towers": (gen_edit_towers, 2.0),
    "edit_wall": (gen_edit_wall, 1.5),
    "edit_table": (gen_edit_table, 1.0),
    "edit_turn": (gen_edit_turn, 1.2),
    "edit_scene_move": (gen_edit_scene_move, 2.0),
    "edit_arch": (gen_edit_arch, 1.0),
    "edit_stairs": (gen_edit_stairs, 0.7),
    "edit_undo": (gen_edit_undo, 1.2),
    "retarget_part": (gen_retarget_part, 4.0),
    "retarget_compose": (gen_retarget_compose, 2.0),
    "retarget_external": (gen_retarget_external, 2.0),
    "trick_too_many": (gen_trick_too_many, 1.0),
    "trick_already": (gen_trick_already, 1.2),
    "trick_negation": (gen_trick_negation, 1.0),
    "trick_false_premise": (gen_trick_false_premise, 1.0),
    "trick_conditional": (gen_trick_conditional, 1.2),
}
if not EXTERNAL:
    del GENERATORS["retarget_external"]


def geometry_problem(cells, scale=1):
    """None when a state is acceptable before the rest test, else the reason."""
    if not 4 <= len(cells) <= MAX_ROBOTS:
        return "count"
    arr = np.asarray(cells)
    if arr.min() < 0 or arr.max() >= SIZE:
        return "outside"
    vox = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    vox[arr[:, 0], arr[:, 1], arr[:, 2]] = True
    if not G.grounded(vox):
        return "ungrounded"
    over, span = overhang_span(cells)
    if over > MAX_OVERHANG * scale or span > MAX_SPAN * scale:
        return "overhang"
    return None


def problem(goal):
    for cells in (goal["cells"], goal["start_cells"]):
        if cells is None:
            continue
        why = geometry_problem(cells, goal["params"].get("scale", 1))
        if why:
            return why
    if goal["start_cells"] is not None and len(goal["start_cells"]) != len(goal["cells"]):
        return "robot count differs"
    if goal["start_cells"] is not None and goal["kind"] != "trick" and goal["start_cells"] == goal["cells"]:
        return "no change"
    for text in goal["turns"] + [goal["reply"]]:
        if text and not text_ok(text):
            return "words"
    return None


def at_rest(cell_lists, batch=64):
    """Per state: reverse.fixed_point (every robot staying changes nothing)."""
    import torch

    from beebot.vla.voxel import reverse
    from beebot.vla.voxel import world as W

    out = []
    for lo in range(0, len(cell_lists), batch):
        worlds = []
        for cells in cell_lists[lo : lo + batch]:
            world = reverse.blank()
            arr = torch.as_tensor(np.asarray(cells), dtype=torch.long)
            world.mass[0, arr[:, 0], arr[:, 1], arr[:, 2]] = 1.0
            worlds.append(world)
        out += reverse.fixed_point(W.concat(worlds)).tolist()
    return out


def draw(name, rng):
    """(goal, None) or (None, reason) for one draw of a generator (geometry and words checked, not rest)."""
    fn = GENERATORS[name][0]
    try:
        goal = fn(rng)
    except Reject as err:
        return None, f"reject: {err}"
    why = problem(goal)
    if why:
        return None, why
    goal["generator"] = name
    states = [goal["cells"]] + ([goal["start_cells"]] if goal["start_cells"] is not None else [])
    ov = [overhang_span(s) for s in states]
    goal["params"]["overhang"] = max(o for o, _ in ov)
    goal["params"]["span"] = max(s for _, s in ov)
    return goal, None


def sample(n, seed=0, names=None, tries=30):
    """n goals drawn by generator weight (all checks passed, at rest included)."""
    rng = np.random.default_rng(seed)
    names = list(names or GENERATORS)
    weights = np.array([GENERATORS[k][1] for k in names], dtype=float)
    weights /= weights.sum()
    out = []
    while len(out) < n:
        pending = []
        for _ in range(min(256, n - len(out))):
            name = names[int(rng.choice(len(names), p=weights))]
            for _ in range(tries):
                goal, _ = draw(name, rng)
                if goal is not None:
                    pending.append(goal)
                    break
        states = [g["cells"] for g in pending] + [g["start_cells"] for g in pending if g["start_cells"] is not None]
        ok = at_rest(states)
        rest = iter(ok[len(pending) :])
        for g, b_ok in zip(pending, ok[: len(pending)]):
            a_ok = next(rest) if g["start_cells"] is not None else True
            if b_ok and a_ok:
                out.append(g)
    return out[:n]


def key_of(goal):
    a = goal["start_cells"]
    blob = json.dumps([goal["cells"], a])
    return hashlib.sha1(blob.encode()).hexdigest()


def measure(draws=1000, seed=1, names=None):
    """Per generator over `draws` draws: acceptance (draws passing every check, at rest included), distinct
    outputs (A, B cells), distinct texts, robots, overhang, span, rejection reasons."""
    out = {}
    for name in names or GENERATORS:
        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        goals, reasons = [], {}
        t0 = time.time()
        for _ in range(draws):
            goal, why = draw(name, rng)
            if goal is None:
                key = why.split(":")[0] if why.startswith("reject") else why
                reasons[key] = reasons.get(key, 0) + 1
            else:
                goals.append(goal)
        states = [g["cells"] for g in goals] + [g["start_cells"] for g in goals if g["start_cells"] is not None]
        ok = at_rest(states) if states else []
        rest = iter(ok[len(goals) :])
        kept = []
        for g, b_ok in zip(goals, ok[: len(goals)]):
            a_ok = next(rest) if g["start_cells"] is not None else True
            if b_ok and a_ok:
                kept.append(g)
            else:
                reasons["not at rest"] = reasons.get("not at rest", 0) + 1
        counts = [len(g["cells"]) for g in kept]
        out[name] = {
            "draws": draws,
            "accepted": len(kept),
            "acceptance": round(len(kept) / draws, 3),
            "distinct": len({key_of(g) for g in kept}),
            "distinct_texts": len({" | ".join(g["turns"]) for g in kept}),
            "robots": [int(min(counts)), round(float(np.mean(counts)), 1), int(max(counts))] if counts else None,
            "changed_mean": round(float(np.mean([len(g["params"]["changed"]["fill"]) for g in kept if "changed" in g["params"]])), 1) if any("changed" in g["params"] for g in kept) else None,
            "overhang_max": max((g["params"]["overhang"] for g in kept), default=None),
            "span_max": max((g["params"]["span"] for g in kept), default=None),
            "rejects": reasons,
            "seconds": round(time.time() - t0, 1),
        }
    return out


# ---------------------------------------------------------------------------------------------
# Into reverse.py


def to_reverse_goal(goal):
    """A reverse.Goal: natural take-apart when there is no start, else the guided walk to A (Goal.start)."""
    import torch

    from beebot.vla.voxel import reverse

    def mask(cells):
        out = torch.zeros(SIZE, SIZE, SIZE, dtype=torch.bool)
        arr = torch.as_tensor(np.asarray(cells), dtype=torch.long)
        out[arr[:, 0], arr[:, 1], arr[:, 2]] = True
        return out

    world = reverse.blank()
    b = mask(goal["cells"])
    world.mass[0] = b.float()
    park = goal["params"]["parked"]
    unused = "parked" if park.get("side") else "free"
    text = goal["turns"][-1]
    if goal["start_cells"] is None:
        parked = set(map(tuple, park_slots(park["side"], MAX_ROWS)[: park["b"]])) if park.get("side") else set()
        struct = [c for c in goal["cells"] if tuple(c) not in parked]
        return reverse.Goal(family=goal["generator"], text=text, world=world, target=mask(struct), unused=unused)
    a = mask(goal["start_cells"])
    target = b & ~a
    if not target.any():
        target = b
    return reverse.Goal(family=goal["generator"], text=text, world=world, target=target, start=a, unused=unused)


def reverse_check(per=3, seed=0, names=None, max_steps=None, min_robots=0, max_robots=MAX_ROBOTS):
    """reverse.take_apart on `per` samples per generator (no-op tricks skipped: A = B needs no walk).
    Returns {generator: {n, ok, replay, steps, robots, ends, seconds}}."""
    import torch

    from beebot.vla.voxel import reverse

    torch.set_num_threads(1)
    out = {}
    for name in names or GENERATORS:
        pool = sample(per * (40 if min_robots else 4), seed=seed, names=[name])
        goals = [g for g in pool if (g["start_cells"] is None or g["start_cells"] != g["cells"]) and min_robots <= len(g["cells"]) <= max_robots][:per]
        if not goals:
            out[name] = {"n": 0, "note": "every sample is a no-op (A = B): no walk needed"}
            continue
        rng = np.random.default_rng(seed)
        t0, c0 = time.time(), time.process_time()
        episodes, _ = reverse.take_apart([to_reverse_goal(g) for g in goals], rng, max_steps=max_steps)
        passed = reverse.replay(episodes)["passed"].tolist()
        ok = [bool(p and e.natural) for p, e in zip(passed, episodes)]
        out[name] = {
            "n": len(goals),
            "ok": sum(ok),
            "replay": sum(bool(p) for p in passed),
            "steps": [int(e.steps) for e in episodes],
            "robots": [len(g["cells"]) for g in goals],
            "ends": [e.end for e in episodes],
            "distance": [int(e.distance) for e in episodes],
            "seconds": round(time.time() - t0, 1),
            "cpu_seconds": round(time.process_time() - c0, 1),
        }
        print(json.dumps({name: out[name]}), flush=True)
    return out


# ---------------------------------------------------------------------------------------------
# Storage and pictures


def save(goals, name=SOURCE, root=G.LIBRARY):
    """<name>.npz / .jsonl through goal_library.save (B), and <name>_start.npz (A; empty for take-apart goals)."""
    lines = []
    for i, g in enumerate(goals):
        arr = np.asarray(g["cells"])
        lines.append({
            "id": f"{name}-{i:06d}",
            "source": SOURCE,
            "generator": g["generator"],
            "kind": g["kind"],
            "frame": "world",
            "dims": [int(v) for v in arr.max(axis=0) - np.array([0, arr[:, 1].min(), arr[:, 2].min()]) + 1],
            "cells": len(g["cells"]),
            "has_start": g["start_cells"] is not None,
            "turns": g["turns"],
            "text": g["turns"][-1],
            "reply": g["reply"],
            "params": g["params"],
        })
    G.save(name, lines, [np.asarray(g["cells"], dtype=np.uint8) for g in goals], root=root)
    starts = [np.asarray(g["start_cells"], dtype=np.uint8) if g["start_cells"] is not None else np.zeros((0, 3), dtype=np.uint8) for g in goals]
    offsets = np.zeros(len(starts) + 1, dtype=np.int64)
    for i, s in enumerate(starts):
        offsets[i + 1] = offsets[i] + len(s)
    flat = np.concatenate(starts) if starts else np.zeros((0, 3), dtype=np.uint8)
    np.savez_compressed(Path(root) / f"{name}_start.npz", cells=flat, starts=offsets)
    return lines


def load(name=SOURCE, root=G.LIBRARY, max_cells=G.MAX_CELLS):
    """(meta lines, B cell arrays, A cell arrays or None). Goals whose B or A holds more than `max_cells`
    cells are dropped (None keeps all)."""
    lines, cells = G.load(name, root, max_cells=None)
    data = np.load(Path(root) / f"{name}_start.npz")
    offsets, flat = data["starts"], data["cells"]
    starts = [flat[offsets[i] : offsets[i + 1]] if line["has_start"] else None for i, line in enumerate(lines)]
    if max_cells is None:
        return lines, cells, starts
    keep = [i for i in range(len(lines)) if len(cells[i]) <= max_cells and (starts[i] is None or len(starts[i]) <= max_cells)]
    return [lines[i] for i in keep], [cells[i] for i in keep], [starts[i] for i in keep]


GREY, FILL, EMPTY, PARK = (70, 80, 95), (230, 120, 40), (90, 160, 230), (150, 150, 150)


def _iso(colour, scale=9):
    from PIL import Image, ImageDraw

    from beebot.vla.voxel.render import BACKGROUND, SHADES, _shade

    s = scale
    width, height = 2 * SIZE * s + 2 * s, SIZE * s + SIZE * s + 2 * s
    ox, oy = SIZE * s + s, SIZE * s + s
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    corners = [(0, 0), (0, SIZE), (SIZE, SIZE), (SIZE, 0)]
    draw.polygon([(ox + (x - y) * s, oy + (x + y) * s / 2) for y, x in corners], outline=(200, 200, 195), fill=(236, 236, 230))
    for (z, y, x), rgb in sorted(colour.items(), key=lambda item: (sum(item[0]), item[0][0])):
        cx, cy = ox + (x - y) * s, oy + (x + y) * s / 2 - z * s
        draw.polygon([(cx, cy - s), (cx + s, cy - s / 2), (cx, cy), (cx - s, cy - s / 2)], fill=_shade(rgb, SHADES[0]))
        draw.polygon([(cx - s, cy - s / 2), (cx, cy), (cx, cy + s), (cx - s, cy + s / 2)], fill=_shade(rgb, SHADES[1]))
        draw.polygon([(cx, cy), (cx + s, cy - s / 2), (cx + s, cy + s / 2), (cx, cy + s)], fill=_shade(rgb, SHADES[2]))
    return image


def _front_top(colour, s=11):
    """Two orthographic views: from the viewer (front: x across, z up, nearer cells drawn darker) and from above
    (x across, back at the top, the viewer's side at the bottom, higher cells drawn darker)."""
    from PIL import Image, ImageDraw

    pad = 14
    image = Image.new("RGB", (2 * SIZE * s + 3 * pad, SIZE * s + 2 * pad + 12), (246, 246, 243))
    draw = ImageDraw.Draw(image)
    front, top = {}, {}
    for (z, y, x), rgb in colour.items():
        if (z, x) not in front or y > front[(z, x)][0]:
            front[(z, x)] = (y, rgb)
        if (y, x) not in top or z > top[(y, x)][0]:
            top[(y, x)] = (z, rgb)
    for ox in (pad, 2 * pad + SIZE * s):
        draw.rectangle([ox, pad, ox + SIZE * s, pad + SIZE * s], outline=(200, 200, 195), fill=(236, 236, 230))
    for (z, x), (y, rgb) in front.items():
        k = 0.55 + 0.45 * (y / (SIZE - 1))
        x0, y0 = pad + x * s, pad + (SIZE - 1 - z) * s
        draw.rectangle([x0, y0, x0 + s - 1, y0 + s - 1], fill=tuple(int(c * (1.6 - k)) if c * (1.6 - k) < 255 else 255 for c in rgb), outline=(246, 246, 243))
    for (y, x), (z, rgb) in top.items():
        k = 1.4 - 0.08 * z
        x0, y0 = 2 * pad + SIZE * s + x * s, pad + y * s
        draw.rectangle([x0, y0, x0 + s - 1, y0 + s - 1], fill=tuple(min(255, max(0, int(c * k))) for c in rgb), outline=(246, 246, 243))
    draw.text((pad, pad + SIZE * s + 2), "seen from me (front)", fill=(60, 60, 60))
    draw.text((2 * pad + SIZE * s, pad + SIZE * s + 2), "from above; me at the bottom", fill=(60, 60, 60))
    return image


def render_goal(goal, path):
    """A and B (iso + front + top views), cells that fill orange in B, cells that empty blue in A, parked grey."""
    import textwrap

    from PIL import Image, ImageDraw

    b, a = set(map(tuple, goal["cells"])), set(map(tuple, goal["start_cells"])) if goal["start_cells"] is not None else None
    park = goal["params"]["parked"]
    parked = set()
    if park.get("side"):
        slots = park_slots(park["side"], MAX_ROWS)
        parked = set(slots[: max(park["a"], park["b"])])
    panels = []
    for name, state, other, mark in (("A (start)", a, b, EMPTY), ("B (end)", b, a, FILL)):
        if state is None:
            continue
        colour = {}
        for c in state:
            colour[c] = PARK if c in parked else GREY
            if other is not None and c not in other:
                colour[c] = mark
        iso, ortho = _iso(colour), _front_top(colour)
        panel = Image.new("RGB", (iso.width + ortho.width, max(iso.height, ortho.height) + 18), (246, 246, 243))
        panel.paste(iso, (0, 18))
        panel.paste(ortho, (iso.width, 18))
        ImageDraw.Draw(panel).text((6, 2), name, fill=(20, 20, 20))
        panels.append(panel)
    lines = [f"[{goal['generator']} / {goal['kind']}] robots {len(goal['cells'])}, overhang {goal['params']['overhang']}, span {goal['params']['span']}"]
    for i, t in enumerate(goal["turns"]):
        lines += textwrap.wrap(f"user {i + 1}: {t}", 150)
    if goal["reply"]:
        lines += textwrap.wrap(f"reply: {goal['reply']}", 150)
    width = max(p.width for p in panels)
    head = 14 * len(lines) + 8
    image = Image.new("RGB", (width, head + sum(p.height for p in panels)), (246, 246, 243))
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        draw.text((6, 4 + 14 * i), line, fill=(20, 20, 20))
    y = head
    for p in panels:
        image.paste(p, (0, y))
        y += p.height
    image.save(path)
    return path


def spread(goals, k, seed=0):
    """k goals covering the generators round-robin."""
    rng = np.random.default_rng(seed)
    by = {}
    for i, g in enumerate(goals):
        by.setdefault(g["generator"], []).append(i)
    pools = [list(rng.permutation(v)) for _, v in sorted(by.items())]
    out = []
    while len(out) < k and any(pools):
        for pool in pools:
            if pool and len(out) < k:
                out.append(int(pool.pop()))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-name", default=SOURCE)
    parser.add_argument("--render-dir", default=None)
    parser.add_argument("--renders", type=int, default=30)
    parser.add_argument("--measure", type=int, default=0, help="draws per generator; prints the table and exits")
    parser.add_argument("--reverse", type=int, default=0, help="samples per generator through reverse.take_apart")
    parser.add_argument("--only", nargs="*", default=None, help="generators (measure / reverse)")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--min-robots", type=int, default=0, help="reverse: only goals with at least this many robots")
    parser.add_argument("--max-robots", type=int, default=MAX_ROBOTS, help="reverse: only goals with at most this many robots")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    if args.measure:
        for name, row in measure(args.measure, args.seed, args.only).items():
            print(json.dumps({name: row}), flush=True)
        return
    if args.reverse:
        reverse_check(args.reverse, args.seed, args.only, args.max_steps, args.min_robots, args.max_robots)
        return
    t0 = time.time()
    goals = sample(args.n, args.seed)
    save(goals, args.out_name)
    kinds = {}
    for g in goals:
        kinds[g["kind"]] = kinds.get(g["kind"], 0) + 1
    summary = {
        "written": len(goals),
        "kinds": kinds,
        "generators": {k: sum(g["generator"] == k for g in goals) for k in GENERATORS},
        "distinct": len({key_of(g) for g in goals}),
        "robots": [int(min(len(g["cells"]) for g in goals)), round(float(np.mean([len(g["cells"]) for g in goals])), 1), int(max(len(g["cells"]) for g in goals))],
        "histogram": G.histogram([len(g["cells"]) for g in goals]),
        "seconds": round(time.time() - t0),
    }
    print(json.dumps(summary), flush=True)
    if args.render_dir:
        out = Path(args.render_dir)
        out.mkdir(parents=True, exist_ok=True)
        cards = []
        for j, i in enumerate(spread(goals, args.renders, args.seed)):
            g = goals[i]
            name = f"gen-complex-{j:02d}"
            render_goal(g, out / f"{name}.png")
            cards.append({"name": name, "generator": g["generator"], "kind": g["kind"], "turns": g["turns"], "reply": g["reply"], "cells": g["cells"], "start_cells": g["start_cells"], "params": g["params"]})
        (out / "cards-gen-complex.json").write_text(json.dumps(cards))


if __name__ == "__main__":
    main()
