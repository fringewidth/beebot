"""Medium-tier goal generators: everyday objects built from blocks, with random sizes, facing and wording.

Each generator is `name(rng) -> (cells, text, params)`; `cells` is an (N, 3) int array of (z, y, x) with z = 0 the
floor, translated to start at 0 on every axis. Conventions are those of reverse.py: front is high y (the side
nearest the viewer of the usual corner render), back is low y, left is low x, right is high x. Shapes are built in a
local frame whose front is +y and then turned a quarter-turn multiple; `Turn.name` converts a local direction to
the word a viewer would use.

    .venv/bin/python -m beebot.vla.voxel.goal_gen_medium --n 20000 --seed 0 --render-dir <dir>

`params` carries the shape's own parameters plus `overhang` (longest horizontal distance of a cell from the
nearest cell with a cell under it, within its connected layer patch) and `span` (longest run of cells in a row
with nothing under them), so real-joint-strength studies can filter later.
"""

import argparse
import json
from collections import deque

import numpy as np

from beebot.vla.voxel import goal_library as G

SIZE = 16
NUMBERS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
FRONT, BACK, LEFT, RIGHT = (1, 0), (-1, 0), (0, -1), (0, 1)
NAMES = {FRONT: "front", BACK: "back", LEFT: "left", RIGHT: "right"}
FORBIDDEN = ("cheese", "stripe", "zone", "voxel", "cell", "robot", "cube")


def num(n):
    return NUMBERS[n] if n < 10 else None


def pick(rng, options):
    return options[int(rng.integers(len(options)))]


def chance(rng, p=0.5):
    return bool(rng.random() < p)


def between(rng, lo, hi):
    """Integer in [lo, hi] inclusive."""
    return int(rng.integers(lo, hi + 1))


class Turn:
    """A random quarter turn about the vertical axis; local front (+y) becomes `facing`."""

    def __init__(self, rng):
        self.k = int(rng.integers(4))

    def rot(self, v):
        y, x = v
        for _ in range(self.k):
            y, x = x, -y
        return (y, x)

    def name(self, v):
        return NAMES[self.rot(v)]

    def axis(self, v):
        """'left to right' or 'front to back' for a local axis direction."""
        y, x = self.rot(v)
        return "front to back" if y else "left to right"

    def cells(self, cells):
        out = []
        for z, y, x in cells:
            ry, rx = self.rot((y, x))
            out.append((z, ry, rx))
        return out


def box(cells, z0, z1, y0, y1, x0, x1):
    """Add [z0, z1) x [y0, y1) x [x0, x1)."""
    for z in range(z0, z1):
        for y in range(y0, y1):
            for x in range(x0, x1):
                cells.add((z, y, x))


def ring(cells, z0, z1, y0, y1, x0, x1):
    """Add the outline of the rectangle [y0, y1) x [x0, x1) for layers [z0, z1)."""
    for z in range(z0, z1):
        for y in range(y0, y1):
            for x in range(x0, x1):
                if y in (y0, y1 - 1) or x in (x0, x1 - 1):
                    cells.add((z, y, x))


def normalise(cells):
    arr = np.array(sorted(cells), dtype=int)
    return arr - arr.min(axis=0)


def finish(cells, turn):
    return normalise(turn.cells(cells))


def blocks(n):
    return f"{NUMBERS[n]} block{'' if n == 1 else 's'}"


def a_or_n(n, noun, plural=None):
    word = num(n)
    plural = plural or noun + "s"
    if word is None:
        return None
    return f"{word} {noun if n == 1 else plural}"


# ---------------------------------------------------------------------------------------------
# Metrics


def overhang_stats(cells):
    """(longest horizontal overhang, largest unsupported run, hanging cells) of a cell array. Overhang is the
    4-neighbour step distance to the nearest cell with a cell under it in the same connected layer patch;
    hanging counts cells of layer patches with nothing under any of them (held from above or the side)."""
    cells = np.asarray(cells).astype(int)
    occupied = {tuple(c) for c in cells}
    over = 0
    span = 0
    hanging = 0
    layers = {}
    for z, y, x in occupied:
        layers.setdefault(z, set()).add((y, x))
    for z, layer in layers.items():
        below = {(y, x) for (zz, y, x) in occupied if zz == z - 1} if z > 0 else None
        if below is None:
            continue
        supported = {p for p in layer if p in below}
        seen = set()
        for start in layer:
            if start in seen:
                continue
            comp = []
            queue = deque([start])
            seen.add(start)
            while queue:
                p = queue.popleft()
                comp.append(p)
                for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (p[0] + d[0], p[1] + d[1])
                    if q in layer and q not in seen:
                        seen.add(q)
                        queue.append(q)
            comp_set = set(comp)
            sources = [p for p in comp if p in supported]
            dist = {p: 0 for p in sources}
            queue = deque(sources)
            while queue:
                p = queue.popleft()
                for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    q = (p[0] + d[0], p[1] + d[1])
                    if q in comp_set and q not in dist:
                        dist[q] = dist[p] + 1
                        queue.append(q)
            if sources:
                over = max(over, max(dist.values()))
            else:
                hanging += len(comp)
        for axis in (0, 1):
            lines = {}
            for p in layer:
                if p not in supported:
                    lines.setdefault(p[axis], []).append(p[1 - axis])
            for values in lines.values():
                values.sort()
                run = 1
                for a, b in zip(values, values[1:]):
                    run = run + 1 if b == a + 1 else 1
                    span = max(span, run)
                span = max(span, 1)
    return over, span, hanging


# ---------------------------------------------------------------------------------------------
# Generators. Local frame: front is +y, left is -x.


def house(rng):
    T = Turn(rng)
    w, d, h = between(rng, 4, 6), between(rng, 4, 6), between(rng, 2, 3)
    cells = set()
    ring(cells, 0, h, 0, d, 0, w)
    side = pick(rng, [FRONT, BACK, LEFT, RIGHT])
    if side in (FRONT, BACK):
        y = d - 1 if side == FRONT else 0
        x = between(rng, 1, w - 2)
        door = [(0, y, x), (1, y, x)]
    else:
        x = w - 1 if side == RIGHT else 0
        y = between(rng, 1, d - 2)
        door = [(0, y, x), (1, y, x)]
    for c in door:
        cells.discard(c)
    window = None
    if h == 3 and chance(rng, 0.6):
        wside = pick(rng, [s for s in (FRONT, BACK, LEFT, RIGHT) if s != side])
        if wside in (FRONT, BACK):
            cells.discard((1, d - 1 if wside == FRONT else 0, between(rng, 1, w - 2)))
        else:
            cells.discard((1, between(rng, 1, d - 2), w - 1 if wside == RIGHT else 0))
        window = T.name(wside)
    roof = pick(rng, ["flat", "gabled"])
    ridge = None
    if roof == "flat":
        box(cells, h, h + 1, 0, d, 0, w)
    else:
        along_x = chance(rng)
        m = d if along_x else w
        ridge = (0, 1) if along_x else (1, 0)
        j = 0
        while m - 2 * j > 0:
            if along_x:
                box(cells, h + j, h + j + 1, j, d - j, 0, w)
            else:
                box(cells, h + j, h + j + 1, 0, d, j, w - j)
            j += 1
    door_name = T.name(side)
    nouns = pick(rng, ["small house", "little house", "hut", "cottage-like house", "toy house"])
    if roof == "flat":
        roof_text = pick(rng, ["a flat roof", "a flat slab on top as a roof", "a flat top"])
    else:
        roof_text = pick(rng, ["a pitched roof that rises to a ridge", "a sloping roof with a ridge along the top"])
        roof_text += f" running {T.axis(ridge)}"
    text = pick(
        rng,
        [
            f"build a {nouns}: hollow walls, a doorway gap in the {door_name} wall and {roof_text}",
            f"make a {nouns} with four walls, an open doorway on the {door_name} side, and {roof_text}",
            f"a {nouns} {num(w)} wide and {num(d)} deep with the door on the {door_name} and {roof_text}",
        ],
    )
    if window:
        text += pick(rng, [f". Add a small window hole in the {window} wall", f", plus a window in the {window} wall"])
    return finish(cells, T), text, {"w": w, "d": d, "wall_height": h, "roof": roof, "door": door_name, "window": window}


def window_wall(rng):
    T = Turn(rng)
    length, height = between(rng, 5, 9), between(rng, 4, 6)
    ww, wh = between(rng, 1, 2), between(rng, 1, 2)
    n = between(rng, 1, min(3, (length - 1) // (ww + 1)))
    z0 = between(rng, 1, height - 1 - wh)
    cells = set()
    box(cells, 0, height, 0, 1, 0, length)
    room = length - 2 - n * ww
    if room < n - 1:
        raise ValueError("windows do not fit")
    gaps = np.zeros(n + 1, dtype=int)
    gaps[0] = 1
    gaps[-1] = 1
    gaps[1:-1] = 1
    extra = length - (n * ww + gaps.sum())
    for _ in range(extra):
        gaps[int(rng.integers(n + 1))] += 1
    x = 0
    for i in range(n):
        x += gaps[i]
        for xx in range(x, x + ww):
            for zz in range(z0, z0 + wh):
                cells.discard((zz, 0, xx))
        x += ww
    facing = T.name(FRONT)
    shape = "square" if ww == wh else ("wide" if ww > wh else "tall")
    text = pick(
        rng,
        [
            f"build a wall {num(length)} blocks long and {num(height)} high, one block thick, with {a_or_n(n, 'window hole')} cut through it",
            f"a flat standing wall, {num(length)} long and {num(height)} tall, with {a_or_n(n, shape + ' opening')} through it at the same height",
            f"make a thin wall facing the {facing}, {num(length)} wide and {num(height)} tall, with {a_or_n(n, 'window')} ({shape}) and a solid top and bottom",
        ],
    )
    return finish(cells, T), text, {"length": length, "height": height, "windows": n, "ww": ww, "wh": wh, "z0": z0}


def gateway(rng):
    T = Turn(rng)
    g, gh = between(rng, 2, 4), between(rng, 3, 5)
    pw, thick = between(rng, 1, 2), between(rng, 1, 2)
    beams = between(rng, 1, 2)
    side = between(rng, 0, 3) if chance(rng, 0.5) else 0
    cells = set()
    total = side + pw + g + pw + side
    box(cells, 0, gh, 0, thick, 0, side + pw)
    box(cells, 0, gh, 0, thick, side + pw + g, total)
    box(cells, gh, gh + beams, 0, thick, 0, total)
    facing = T.name(FRONT)
    walls = f", with a wall {blocks(side)} long continuing on each side" if side else ""
    text = pick(
        rng,
        [
            f"build a gateway: two thick posts with a beam across the top, the opening {num(g)} wide and {num(gh)} high{walls}",
            f"make a doorway frame facing the {facing}, {num(g)} blocks wide inside, {num(gh)} tall, posts {num(pw)} wide and a lintel on top{walls}",
            f"a free-standing gate you can walk through: opening {num(g)} wide, {num(gh)} high, open toward the {facing}{walls}",
        ],
    )
    return finish(cells, T), text, {"opening": g, "height": gh, "post": pw, "thick": thick, "beams": beams, "side": side}


def bench(rng):
    T = Turn(rng)
    length, depth, legs = between(rng, 4, 7), between(rng, 1, 2), between(rng, 1, 2)
    back = chance(rng, 0.6)
    arms = chance(rng, 0.3)
    cells = set()
    box(cells, 0, legs, 0, depth, 0, 1)
    box(cells, 0, legs, 0, depth, length - 1, length)
    box(cells, legs, legs + 1, 0, depth, 0, length)
    if back:
        box(cells, legs + 1, legs + 3, 0, 1, 0, length)
    if arms:
        box(cells, legs + 1, legs + 2, 0, depth, 0, 1)
        box(cells, legs + 1, legs + 2, 0, depth, length - 1, length)
    # the back is at y = 0, so the sitter faces +y
    facing = T.name(FRONT)
    extras = (" with a backrest along one long side" if back else "") + (" and an armrest at each end" if arms else "")
    text = pick(
        rng,
        [
            f"build a bench: a seat {num(length)} blocks long on two solid ends {num(legs)} high{extras}, to be sat on facing the {facing}",
            f"make a long seat on two end supports, {num(length)} long, {num(legs)} off the ground{extras}; you sit facing the {facing}",
            f"a bench {num(length)} wide and {num(depth)} deep{extras}, seat on two end blocks, seat facing the {facing}",
        ],
    )
    return finish(cells, T), text, {"length": length, "depth": depth, "legs": legs, "back": back, "arms": arms}


def sofa(rng):
    T = Turn(rng)
    w, d = between(rng, 4, 6), between(rng, 2, 3)
    arm_h = between(rng, 2, 3)
    back_h = between(rng, 3, 4)
    cells = set()
    box(cells, 0, 1, 0, d, 0, w)
    box(cells, 1, back_h, 0, 1, 0, w)
    box(cells, 1, arm_h, 0, d, 0, 1)
    box(cells, 1, arm_h, 0, d, w - 1, w)
    facing = T.name(FRONT)
    text = pick(
        rng,
        [
            f"build a sofa facing the {facing}: a low seat {num(w)} wide, a tall back behind it and an arm at each end",
            f"make a couch, {num(w)} blocks wide and {num(d)} deep, with a backrest and two armrests, you sit facing the {facing}",
            f"a sofa with a high back on its {T.name(BACK)} side and a raised arm on each end",
        ],
    )
    return finish(cells, T), text, {"w": w, "d": d, "arm": arm_h, "back": back_h}


def bed(rng):
    T = Turn(rng)
    length, width = between(rng, 6, 8), between(rng, 3, 4)
    foot = chance(rng, 0.6)
    pillow = chance(rng, 0.7)
    cells = set()
    box(cells, 0, 1, 0, length, 0, width)
    box(cells, 1, 2, 1, length - 1, 0, width)
    box(cells, 1, 4, 0, 1, 0, width)
    if foot:
        box(cells, 1, 2, length - 1, length, 0, width)
        box(cells, 1, 3, length - 1, length, 0, width)
    if pillow:
        box(cells, 2, 3, 1, 2, 1, width - 1)
    head = T.name(BACK)
    foot_text = " and a low board at the other end" if foot else ""
    pill = " with a pillow lump against the head" if pillow else ""
    text = pick(
        rng,
        [
            f"build a bed {num(length)} long and {num(width)} wide with a tall headboard on the {head} end{foot_text}{pill}",
            f"make a bed: a flat mattress {num(width)} across, a tall board at the {head} end{foot_text}{pill}",
            f"a low bed lying {T.axis(BACK)}, the head (tall end) toward the {head}{foot_text}{pill}",
        ],
    )
    return finish(cells, T), text, {"length": length, "width": width, "foot": foot, "pillow": pillow}


def bookcase(rng):
    T = Turn(rng)
    w, h, d = between(rng, 4, 6), between(rng, 4, 7), between(rng, 2, 3)
    back = chance(rng, 0.3)
    cells = set()
    box(cells, 0, h, 0, d, 0, 1)
    box(cells, 0, h, 0, d, w - 1, w)
    gap = between(rng, 1, 2)
    levels = list(range(0, h, gap + 1))
    if (h - 1) not in levels:
        levels.append(h - 1)
    for z in levels:
        box(cells, z, z + 1, 0, d, 0, w)
    if back:
        box(cells, 0, h, 0, 1, 0, w)
    facing = T.name(FRONT)
    shelves = len(levels)
    back_text = "the opposite side closed off by a flat panel" if back else "nothing closing the opposite side, so you can see through"
    text = pick(
        rng,
        [
            f"build a bookshelf {num(w)} wide and {num(h)} tall, open on the {facing} side, with {a_or_n(shelves, 'shelf', 'shelves')} counting the top and the bottom, and {back_text}",
            f"make a shelf unit: two side walls and boards across at every {'other' if gap == 1 else 'third'} level, {back_text}, the open side toward the {facing}",
            f"a tall bookcase, {num(h)} high, you reach into it from the {facing}, with {back_text}",
        ],
    )
    return finish(cells, T), text, {"w": w, "h": h, "d": d, "shelves": shelves, "back": back}


def turning_stair(rng):
    T = Turn(rng)
    wd = between(rng, 1, 2)
    n1, n2 = between(rng, 2, 4), between(rng, 2, 4)
    turn_left = chance(rng)
    cells = set()
    for i in range(n1):
        box(cells, 0, i + 1, 0, wd, i, i + 1)
    box(cells, 0, n1 + 1, 0, wd, n1, n1 + wd)
    sign = 1 if turn_left else -1
    for j in range(n2):
        y0, y1 = (wd + j, wd + j + 1) if sign > 0 else (-1 - j, -j)
        box(cells, 0, n1 + 2 + j, y0, y1, n1, n1 + wd)
    # flight one climbs toward +x (local right); the second climbs toward +y (local front) or -y (back)
    first = T.name(RIGHT)
    second = T.name(FRONT if sign > 0 else BACK)
    text = pick(
        rng,
        [
            f"build a staircase going up {num(n1)} steps toward the {first}, then a flat landing, then it turns and climbs {num(n2)} more steps toward the {second}",
            f"make stairs that climb toward the {first} and turn a corner onto a square landing, after which they go on up toward the {second} ({num(wd)} wide)",
            f"a solid staircase with a bend: first flight heads {first}, a landing, second flight heads {second}",
        ],
    )
    return finish(cells, T), text, {"width": wd, "steps1": n1, "steps2": n2, "turn": second}


def balcony_tower(rng):
    T = Turn(rng)
    s = between(rng, 3, 4)
    h = between(rng, 6, 8) if s == 3 else between(rng, 5, 6)
    cells = set()
    ring(cells, 0, h, 0, s, 0, s)
    box(cells, h, h + 1, 0, s, 0, s)
    door = chance(rng, 0.6)
    if door:
        cells.discard((0, s - 1, 1))
        cells.discard((1, s - 1, 1))
    bz = between(rng, 3, h - 2)
    o = between(rng, 1, 2)
    side = pick(rng, [FRONT, BACK, LEFT, RIGHT])
    width = s
    if side == FRONT:
        box(cells, bz, bz + 1, s, s + o, 0, width)
    elif side == BACK:
        box(cells, bz, bz + 1, -o, 0, 0, width)
    elif side == LEFT:
        box(cells, bz, bz + 1, 0, s, -o, 0)
    else:
        box(cells, bz, bz + 1, 0, s, s, s + o)
    door_text = ", with a doorway at the bottom on the front" if door else ""
    door_name = T.name(FRONT)
    door_text = door_text.replace("front", door_name)
    text = pick(
        rng,
        [
            f"build a tall hollow tower, {num(s)} blocks across and {num(h)} high with a flat top, and a balcony ledge sticking out {blocks(o)} on the {T.name(side)} side at height {num(bz)}{door_text}",
            f"make a tower with a balcony: a square tube {num(s)} by {num(s)}, {num(h)} tall, closed on top, with a flat platform jutting out to the {T.name(side)} partway up{door_text}",
            f"a slim tower and a small balcony on its {T.name(side)} side, {num(o)} out, {num(bz)} up from the ground{door_text}",
        ],
    )
    return finish(cells, T), text, {"s": s, "h": h, "balcony_z": bz, "out": o, "side": T.name(side), "door": door}


def well(rng):
    T = Turn(rng)
    s = between(rng, 3, 4)
    base = between(rng, 1, 2)
    ph = between(rng, 3, 5)
    roof = chance(rng, 0.4)
    cells = set()
    ring(cells, 0, base, 0, s, 0, s)
    mid = s // 2
    for x in (0, s - 1):
        box(cells, base, base + ph, mid, mid + 1, x, x + 1)
    top = base + ph
    if roof:
        box(cells, top, top + 1, 0, s, 0, s)
    else:
        box(cells, top, top + 1, mid, mid + 1, 0, s)
    axis = T.axis(RIGHT)
    roof_text = " and a flat roof over it" if roof else " joined by a bar across the top"
    text = pick(
        rng,
        [
            f"build a well: a low hollow square wall {num(s)} by {num(s)} and {num(base)} high, with two posts standing {axis} on opposite sides{roof_text}",
            f"make a ring of blocks hollow in the middle, {num(base)} high, with a post on two opposite sides rising {num(ph)} higher{roof_text}",
            f"a wishing-well shape: stone-like rim, a hole in the middle, two uprights at either end of the {axis} line{roof_text}",
        ],
    )
    return finish(cells, T), text, {"s": s, "base": base, "post": ph, "roof": roof}


def bridge(rng):
    T = Turn(rng)
    deck_z = between(rng, 2, 3)
    wd = 3
    pw = between(rng, 1, 2)
    inner = between(rng, 3, 5)
    rails = chance(rng, 0.5)
    ramp = chance(rng, 0.7)
    length = 2 * pw + inner
    cells = set()
    box(cells, deck_z, deck_z + 1, 0, wd, 0, length)
    for x in list(range(0, pw)) + list(range(length - pw, length)):
        box(cells, 0, deck_z, 0, wd, x, x + 1)
    if rails:
        box(cells, deck_z + 1, deck_z + 2, 0, 1, 0, length)
        box(cells, deck_z + 1, deck_z + 2, wd - 1, wd, 0, length)
    if ramp:
        for i in range(deck_z - 1):
            height = deck_z - 1 - i
            box(cells, 0, height, 0, wd, -1 - i, -i)
            box(cells, 0, height, 0, wd, length + i, length + i + 1)
    axis = T.axis(RIGHT)
    text = pick(
        rng,
        [
            f"build a bridge standing on the flat ground: a deck {num(wd)} wide running {axis}, resting on a thick block at each end, {num(deck_z)} up with open air under the middle"
            + (", low rails along both edges" if rails else "")
            + (", and steps at each end" if ramp else ""),
            f"make a raised walkway {axis}, held up by two solid ends with a gap under the middle {num(inner)} long, deck {num(deck_z)} blocks above the floor"
            + (" and a low fence-like rail on each side" if rails else "")
            + (", with a few steps up at both ends" if ramp else ""),
        ],
    )
    return finish(cells, T), text, {"deck_z": deck_z, "pier": pw, "gap": inner, "rails": rails, "ramp": ramp}


def vehicle(rng):
    T = Turn(rng)
    kind = pick(rng, ["car", "truck", "pickup"])
    wheel = 2 if kind != "car" else between(rng, 1, 2)
    length = between(rng, 7, 9)
    width = 3
    cells = set()
    for x in (1, length - 2):
        for y in (0, width - 1):
            box(cells, 0, wheel, y, y + 1, x, x + 1)
    chassis = wheel
    box(cells, chassis, chassis + 1, 0, width, 0, length)
    nose_x = length - 1
    if kind == "car":
        cl = 3
        box(cells, chassis + 1, chassis + 2, 0, width, length - 2 - cl, length - 2)
        desc = "a small car: a flat body on four single-block wheels with a cabin block on top, a little behind the nose"
    elif kind == "truck":
        box(cells, chassis + 1, chassis + 3, 0, width, length - 2, length)
        box(cells, chassis + 1, chassis + 4, 0, width, 0, length - 3)
        desc = "a truck: a flat body on four wheels, a small cab at the nose and a tall box behind it"
    else:
        box(cells, chassis + 1, chassis + 3, 0, width, length - 3, length - 1)
        ring(cells, chassis + 1, chassis + 2, 0, width, 0, length - 3)
        desc = "a pickup: a flat body on four wheels, a tall cab near the nose and an open low tray behind it"
    nose = T.name(RIGHT)
    text = pick(
        rng,
        [f"build {desc}, nose pointing {nose}", f"make {desc}; it drives toward the {nose}", f"{desc}, the front end on the {nose}"],
    )
    return finish(cells, T), text, {"kind": kind, "wheel": wheel, "length": length, "nose": nose}


def boat(rng):
    T = Turn(rng)
    length = between(rng, 6, 9)
    wide = pick(rng, [3, 5])
    cells = set()
    mid = wide // 2
    mask = set()
    for y in range(wide):
        reach = length - 1 if y == mid else (length - 2 if abs(y - mid) == 1 else length - 3)
        for x in range(reach + 1):
            mask.add((y, x))
    for y, x in mask:
        cells.add((0, y, x))
        inner = 1 <= y <= wide - 2 and 1 <= x <= length - 3 and (y + 1, x) in mask and (y - 1, x) in mask and (y, x + 1) in mask
        if not inner:
            cells.add((1, y, x))
    mast = pick(rng, [length // 2, length // 2 - 1])
    mh = between(rng, 3, 5)
    sail = chance(rng, 0.8)
    box(cells, 1, 1 + mh, mid, mid + 1, mast, mast + 1)
    sl = between(rng, 2, 3)
    sh = between(rng, 2, 3)
    if sail:
        box(cells, 1 + mh - sh, 1 + mh, mid, mid + 1, mast - sl, mast)
    bow = T.name(RIGHT)
    sail_text = f" and a flat sail hanging off the mast toward the {T.name(LEFT)} (stern)" if sail else ""
    text = pick(
        rng,
        [
            f"build a boat: a hollow hull {num(wide)} wide, pointed at the {bow} end, with a mast standing up in the middle{sail_text}",
            f"make a little boat whose nose points {bow}: a low open hull and a tall pole in the centre{sail_text}",
        ],
    )
    return finish(cells, T), text, {"length": length, "wide": wide, "mast": mh, "sail": sail}


def tree(rng):
    T = Turn(rng)
    th = between(rng, 3, 6)
    t = pick(rng, [1, 1, 2])
    kind = pick(rng, ["round", "pine", "tiers"])
    cells = set()
    box(cells, 0, th, 0, t, 0, t)
    if kind == "round":
        r = between(rng, 1, 2)
        n_layers = between(rng, 2, 3)
        clip = chance(rng)
        for i in range(n_layers):
            rr = r if i < n_layers - 1 else max(r - 1, 0)
            for y in range(-rr, t + rr):
                for x in range(-rr, t + rr):
                    dy = max(-y, y - (t - 1), 0)
                    dx = max(-x, x - (t - 1), 0)
                    if clip and dy + dx > rr:
                        continue
                    cells.add((th + i, y, x))
        desc = f"a tree: a trunk {num(th)} high with a blocky round leafy top {num(n_layers)} layers thick"
    elif kind == "pine":
        sizes = pick(rng, [(2, 1, 0), (2, 1), (1, 0), (1, 1, 0), (2, 2, 1, 0)])
        for i, rr in enumerate(sizes):
            box(cells, th + i, th + i + 1, -rr, t + rr, -rr, t + rr)
        desc = f"a pine-like tree: a trunk {num(th)} high with wide layers on top, each one smaller than the one below"
    else:
        rr = between(rng, 1, 2)
        gap = between(rng, 1, 2)
        for i in range(2):
            z = th + i * (gap + 1)
            r = rr if i == 0 else rr - 1
            box(cells, z, z + 1, -r, t + r, -r, t + r)
            if i == 0:
                box(cells, z + 1, z + 1 + gap, 0, t, 0, t)
        desc = f"a tree with two separate leafy platforms, one above the other on the same trunk (trunk {num(th)} high)"
    text = pick(rng, [f"build {desc}", f"make {desc}", f"{desc}, standing on the ground"])
    return finish(cells, T), text, {"trunk": th, "kind": kind, "thick": t}


def lamp_post(rng):
    T = Turn(rng)
    ph = between(rng, 5, 8)
    arm = between(rng, 2, 3)
    base = chance(rng, 0.6)
    drop = between(rng, 1, 2)
    cells = set()
    box(cells, 0, ph, 1, 2, 1, 2)
    if base:
        box(cells, 0, 1, 0, 3, 0, 3)
    side = pick(rng, [FRONT, BACK, LEFT, RIGHT])
    dy, dx = side
    for i in range(1, arm + 1):
        cells.add((ph - 1, 1 + dy * i, 1 + dx * i))
    for k in range(1, drop + 1):
        cells.add((ph - 1 - k, 1 + dy * arm, 1 + dx * arm))
    text = pick(
        rng,
        [
            f"build a street lamp: a thin pole {num(ph)} high{' on a flat square foot' if base else ''}, a bar sticking out {blocks(arm)} to the {T.name(side)} at the top, and the lamp hanging {blocks(drop)} down at the end of the bar",
            f"make a tall post{' with a wide base' if base else ''} that bends sideways at the top toward the {T.name(side)} and has a block dangling below the tip",
        ],
    )
    return finish(cells, T), text, {"height": ph, "arm": arm, "base": base, "drop": drop, "side": T.name(side)}


def ladder_wall(rng):
    T = Turn(rng)
    lw, h = between(rng, 5, 7), between(rng, 5, 7)
    a = between(rng, 0, lw - 3)
    lh = between(rng, h - 2, h)
    cells = set()
    box(cells, 0, h, 0, 1, 0, lw)
    for z in range(lh):
        cells.add((z, 1, a))
        cells.add((z, 1, a + 2))
        if z % 2 == 1:
            cells.add((z, 1, a + 1))
    facing = T.name(FRONT)
    where = "the left" if a == 0 else ("the right" if a + 3 == lw else "the middle")
    where = where.replace("the left", f"toward the {T.name(LEFT)} end").replace("the right", f"toward the {T.name(RIGHT)} end").replace("the middle", "in the middle")
    text = pick(
        rng,
        [
            f"build a tall flat wall {num(lw)} wide and {num(h)} high, and a ladder leaning flat against its {facing} face {where}: two upright poles {num(1)} apart with a rung between them every second level",
            f"make a wall with a ladder on the {facing} side, {where}; the ladder reaches {num(lh)} high",
        ],
    )
    return finish(cells, T), text, {"w": lw, "h": h, "ladder_x": a, "ladder_h": lh}


def battlement_wall(rng):
    T = Turn(rng)
    length = between(rng, 7, 12)
    h = between(rng, 2, 4)
    thick = 1 if length >= 10 else between(rng, 1, 2)
    towers = chance(rng, 0.5)
    gate = chance(rng, 0.3)
    cells = set()
    box(cells, 0, h, 0, thick, 0, length)
    for x in range(0, length, 2):
        box(cells, h, h + 1, 0, thick, x, x + 1)
    if towers:
        ts = 3
        for x0 in (0, length - ts):
            ring(cells, 0, h + 2, -(ts - thick) // 2 if False else 0, ts, x0, x0 + ts)
            for y in range(ts):
                for x in range(x0, x0 + ts):
                    if (y in (0, ts - 1) or x in (x0, x0 + ts - 1)) and (y + x) % 2 == 0:
                        cells.add((h + 2, y, x))
    if gate:
        gx = length // 2 - 1
        for x in range(gx, gx + 2):
            for z in range(min(2, h - 1)):
                for y in range(thick):
                    cells.discard((z, y, x))
    facing = T.name(FRONT)
    ln = num(length) or "long"
    tow = " with a tall hollow tower at each end" if towers else ""
    gt = " and a two-wide gateway gap at the bottom in the middle" if gate else ""
    text = pick(
        rng,
        [
            f"build a castle wall running {T.axis(RIGHT)}, {num(h)} high with a top edge that goes up and down like teeth{tow}{gt}",
            f"make a long wall with notches along the top (every other block missing){tow}{gt}, {blocks(thick)} thick",
        ],
    )
    return finish(cells, T), text, {"length": length, "h": h, "thick": thick, "towers": towers, "gate": gate}


def animal(rng):
    T = Turn(rng)
    bl = between(rng, 4, 6)
    lh = between(rng, 1, 2)
    bh = between(rng, 1, 2)
    nh = between(rng, 0, 2)
    tail = chance(rng, 0.6)
    ears = chance(rng, 0.4)
    cells = set()
    for x in (0, bl - 1):
        for y in (0, 2):
            box(cells, 0, lh, y, y + 1, x, x + 1)
    box(cells, lh, lh + bh, 0, 3, 0, bl)
    top = lh + bh
    nx = bl - 1
    box(cells, top, top + nh, 1, 2, nx, nx + 1)
    cells.add((top + nh, 1, nx))
    cells.add((top + nh, 1, nx + 1))
    if ears:
        cells.add((top + nh + 1, 1, nx))
    if tail:
        cells.add((top, 1, 0))
    head = T.name(RIGHT)
    neck = {0: "no neck", 1: "a short neck", 2: "a long neck"}[nh]
    text = pick(
        rng,
        [
            f"build a blocky four-legged animal, a body {num(bl)} long on four legs with {neck} and the head at the {head} end"
            + (", a tail sticking up at the other end" if tail else ""),
            f"make a simple animal standing on four legs, {num(bl)} long, facing {head}, with {neck}" + (" and a small tail" if tail else ""),
        ],
    )
    if ears:
        text += ", and a bump on top of the head for an ear"
    return finish(cells, T), text, {"body": bl, "leg": lh, "body_h": bh, "neck": nh, "tail": tail, "ears": ears}


def person(rng):
    T = Turn(rng)
    lg, tl = between(rng, 2, 4), between(rng, 2, 4)
    al = between(rng, 1, 3)
    pose = pick(rng, ["out", "up", "down"])
    c = 3
    cells = set()
    box(cells, 0, lg, 0, 1, c - 1, c)
    box(cells, 0, lg, 0, 1, c + 1, c + 2)
    box(cells, lg, lg + tl, 0, 1, c - 1, c + 2)
    sh = lg + tl - 1
    for side in (-1, 1):
        for i in range(al):
            if pose == "out":
                cells.add((sh, 0, c + side * (2 + i)))
            elif pose == "up":
                cells.add((sh + i, 0, c + side * 2))
            else:
                cells.add((sh - i, 0, c + side * 2))
    cells.add((lg + tl, 0, c))
    if chance(rng, 0.5):
        cells.add((lg + tl + 1, 0, c))
    facing = T.name(FRONT)
    pose_text = {"out": "arms stretched straight out to the sides", "up": "both arms raised up", "down": "arms hanging down at the sides"}[pose]
    text = pick(
        rng,
        [
            f"build a simple blocky person standing flat facing the {facing}: two legs apart, a wide body, {pose_text} and a head on top",
            f"make a stick-figure person seen face on from the {facing}, {num(lg)} blocks of leg, {pose_text}, a head",
        ],
    )
    return finish(cells, T), text, {"legs": lg, "torso": tl, "arm": al, "pose": pose}


FONT = {
    "A": ["111", "101", "111", "101", "101"],
    "C": ["111", "100", "100", "100", "111"],
    "E": ["111", "100", "110", "100", "111"],
    "F": ["111", "100", "110", "100", "100"],
    "H": ["101", "101", "111", "101", "101"],
    "I": ["111", "010", "010", "010", "111"],
    "L": ["100", "100", "100", "100", "111"],
    "M": ["101", "111", "111", "101", "101"],
    "N": ["110", "101", "101", "101", "101"],
    "O": ["111", "101", "101", "101", "111"],
    "P": ["111", "101", "111", "100", "100"],
    "T": ["111", "010", "010", "010", "010"],
    "U": ["101", "101", "101", "101", "111"],
}
WORDS = ["ON", "UP", "HI", "NO", "TOP", "CAT", "HOT", "MAP", "LOP", "TAP", "POT", "PAT", "MOP", "HAT", "ICE", "ONE", "TEN", "LIP", "PIE", "COT", "CUP", "FIT"]
WORDS = [w for w in WORDS if all(ch in FONT for ch in w)]


def letters(rng):
    T = Turn(rng)
    if chance(rng, 0.6):
        word = pick(rng, WORDS)
        kind = "word"
    else:
        n = between(rng, 2, 3)
        word = "".join(pick(rng, sorted(FONT)) for _ in range(n))
        kind = "letters"
    gap = 1
    thick = between(rng, 1, 2)
    cells = set()
    for i, ch in enumerate(word):
        for r, row in enumerate(FONT[ch]):
            for c, bit in enumerate(row):
                if bit == "1":
                    box(cells, 4 - r, 5 - r, 0, thick, i * (3 + gap) + c, i * (3 + gap) + c + 1)
    facing = T.name(FRONT)
    spelled = " ".join(word)
    if kind == "word":
        text = pick(
            rng,
            [
                f"build the word {word} in tall upright letters, {num(5)} high, standing facing the {facing} so it reads left to right from there",
                f"make big standing block letters that spell {word}, readable from the {facing}",
            ],
        )
    else:
        text = pick(
            rng,
            [
                f"build the letters {spelled} standing upright in a row, {num(5)} blocks tall, readable from the {facing}",
                f"make the {num(len(word))} capital letters {spelled}, side by side, standing up, facing the {facing}",
            ],
        )
    if thick == 2:
        text += ", each letter two blocks thick"
    return finish(cells, T), text, {"word": word, "thick": thick}


def signpost_arrow(rng):
    T = Turn(rng)
    direction = pick(rng, ["up", "left", "right", "right", "left"])
    shaft = between(rng, 3, 6)
    thick = between(rng, 1, 2)
    wing = between(rng, 1, 2)  # how far the head's wings reach out from the shaft
    length = shaft + wing + 1
    # board in (row offset from the shaft, column along the arrow); the tip is at column length - 1
    board = {(0, c) for c in range(length)}
    for w in range(1, wing + 1):
        for c in range(length - 1 - wing, length - w):
            board.add((w, c))
            board.add((-w, c))
    cells = set()
    if direction == "up":
        for off, c in board:
            box(cells, c, c + 1, 0, thick, 2 + off, 3 + off)
        post = 0
        facing = T.name(FRONT)
        text = pick(
            rng,
            [
                f"build a flat arrow standing upright on the ground, pointing straight up, a long shaft with a triangular head, facing the {facing}",
                f"make a big arrow standing on its tail, pointing up, flat like a signboard, facing the {facing}",
            ],
        )
    else:
        post = between(rng, 2, 6)
        px = 1
        for off, c in board:
            cc = c if direction == "right" else length - 1 - c
            box(cells, post + wing + off, post + wing + off + 1, 0, thick, cc, cc + 1)
        box(cells, 0, post + wing, 0, thick, px, px + 1)
        where = T.name(RIGHT if direction == "right" else LEFT)
        text = pick(
            rng,
            [
                f"build a signpost: a post {num(post)} high with a flat upright arrow-shaped board on top pointing {where}, board facing the {T.name(FRONT)}",
                f"make an arrow sign on a pole, the arrow pointing {where} and the pole {num(post)} high",
            ],
        )
    if thick == 2:
        text += ", the arrow two blocks thick"
    return finish(cells, T), text, {"direction": direction, "length": length, "post": post, "thick": thick, "wing": wing}


def heart_star(rng):
    T = Turn(rng)
    kind = pick(rng, ["heart", "star", "heart", "star"])
    n = between(rng, 8, 12) if kind == "heart" else between(rng, 12, 14)
    outline = chance(rng, 0.5)
    solid = np.zeros((n, n), dtype=bool)
    if kind == "heart":
        for r in range(n):
            for c in range(n):
                x = (c + 0.5) / n * 2.6 - 1.3
                y = 1.3 - (r + 0.5) / n * 2.6
                y += 0.1
                solid[r, c] = (x * x + y * y - 1) ** 3 - x * x * y**3 < 0
    else:
        pts = []
        for i in range(10):
            ang = np.pi / 2 + i * np.pi / 5
            rad = 1.0 if i % 2 == 0 else 0.45
            pts.append((rad * np.cos(ang), rad * np.sin(ang)))
        for r in range(n):
            for c in range(n):
                px = (c + 0.5) / n * 2.1 - 1.05
                py = 1.05 - (r + 0.5) / n * 2.1 - 0.1
                inside = False
                j = len(pts) - 1
                for i in range(len(pts)):
                    xi, yi = pts[i]
                    xj, yj = pts[j]
                    if (yi > py) != (yj > py) and px < (xj - xi) * (py - yi) / (yj - yi) + xi:
                        inside = not inside
                    j = i
                solid[r, c] = inside
    if outline:
        eroded = solid.copy()
        for r in range(n):
            for c in range(n):
                if solid[r, c]:
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                        rr, cc = r + dr, c + dc
                        if not (0 <= rr < n and 0 <= cc < n) or not solid[rr, cc]:
                            eroded[r, c] = False
        shape = solid & ~eroded
        if eroded.sum() < 3:
            raise ValueError("no hole")
    else:
        shape = solid
    if shape.sum() < 8:
        raise ValueError("too small")
    cells = set()
    for r in range(n):
        for c in range(n):
            if shape[r, c]:
                cells.add((n - 1 - r, 0, c))
    facing = T.name(FRONT)
    style = "drawn as an outline with a hole in the middle" if outline else "filled in solid"
    text = pick(
        rng,
        [
            f"build a {kind} standing upright, flat like a picture, facing the {facing}, {style}, about {num(n)} blocks tall",
            f"make a {kind} shape stood up on the ground facing the {facing}, {style}",
        ],
    )
    return finish(cells, T), text, {"kind": kind, "n": n, "outline": outline}


def fence(rng):
    T = Turn(rng)
    pen = chance(rng, 0.6)
    ph = between(rng, 2, 3)
    low_rail = ph == 3 and chance(rng)
    cells = set()
    gap_name = None
    if pen:
        w, d = between(rng, 4, 6), between(rng, 4, 6)
        path = [(0, x) for x in range(w)] + [(y, w - 1) for y in range(1, d)] + [(d - 1, x) for x in range(w - 2, -1, -1)] + [(y, 0) for y in range(d - 2, 0, -1)]
        gap = pick(rng, [FRONT, BACK, LEFT, RIGHT])
        if gap == FRONT:
            gp = (d - 1, between(rng, 1, w - 2))
        elif gap == BACK:
            gp = (0, between(rng, 1, w - 2))
        elif gap == LEFT:
            gp = (between(rng, 1, d - 2), 0)
        else:
            gp = (between(rng, 1, d - 2), w - 1)
        gap_name = T.name(gap)
    else:
        length = between(rng, 5, 9)
        path = [(0, x) for x in range(length)]
        gp = None
    for i, (y, x) in enumerate(path):
        if (y, x) == gp:
            continue
        corner = pen and (y in (0, d - 1) and x in (0, w - 1))
        post = corner or i % 2 == 0
        if post:
            box(cells, 0, ph, y, y + 1, x, x + 1)
        else:
            cells.add((ph - 1, y, x))
            if low_rail:
                cells.add((0, y, x))
    if pen:
        for (y, x) in path:
            if (y, x) != gp:
                cells.add((ph - 1, y, x))
    if pen:
        text = pick(
            rng,
            [
                f"build a small square pen made of a fence: posts {num(ph)} high with rails between them, and a one block gap in the {gap_name} side to get in",
                f"make a fenced square yard {num(w)} by {num(d)} with an opening on the {gap_name} side; the fence is {num(ph)} high with a rail along the top",
            ],
        )
    else:
        text = pick(
            rng,
            [
                f"build a straight fence {num(length)} blocks long running {T.axis(RIGHT)}: a post every other block, {num(ph)} high, with a rail along the top",
                f"make a fence: upright posts with gaps between them, joined by a rail on top, {num(length)} long",
            ],
        )
    return finish(cells, T), text, {"pen": pen, "post": ph, "gap": gap_name}


GENERATORS = {
    "house": house,
    "window_wall": window_wall,
    "gateway": gateway,
    "bench": bench,
    "sofa": sofa,
    "bed": bed,
    "bookcase": bookcase,
    "turning_stair": turning_stair,
    "balcony_tower": balcony_tower,
    "well": well,
    "bridge": bridge,
    "vehicle": vehicle,
    "boat": boat,
    "tree": tree,
    "lamp_post": lamp_post,
    "ladder_wall": ladder_wall,
    "battlement_wall": battlement_wall,
    "animal": animal,
    "person": person,
    "letters": letters,
    "signpost_arrow": signpost_arrow,
    "heart_star": heart_star,
    "fence": fence,
}


# ---------------------------------------------------------------------------------------------
# Validation and sampling


def geometry_ok(cells):
    cells = np.asarray(cells)
    if len(cells) < G.MIN_CELLS or len(cells) > G.MAX_CELLS:
        return False
    if (cells.max(axis=0) >= SIZE).any() or cells.min() < 0:
        return False
    if len({tuple(c) for c in cells.tolist()}) != len(cells):
        return False
    return bool(G.grounded(G.cells_to_vox(cells)))


def text_ok(text):
    low = text.lower()
    if any(word in low for word in FORBIDDEN):
        return False
    import re

    return not any(int(n) >= 10 for n in re.findall(r"\d+", low))


def shape_key(cells):
    arr = np.asarray(cells).astype(int)
    arr = arr - arr.min(axis=0)
    order = np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))
    return arr[order].astype(np.uint8).tobytes()


def draw(name, rng):
    """One candidate from generator `name`: (cells, text, params) or None when geometry or text fail."""
    try:
        cells, text, params = GENERATORS[name](rng)
    except (ValueError, KeyError, IndexError):
        return None
    if not text_ok(text) or not geometry_ok(cells):
        return None
    over, span, hanging = overhang_stats(cells)
    params = dict(params)
    params["overhang"] = int(over)
    params["span"] = int(span)
    params["hanging"] = int(hanging)
    return np.asarray(cells).astype(int), text, params


def at_rest(cells_list, batch=32):
    """Bool per cell array: a fixed point of the forward step."""
    from beebot.vla.voxel import reverse

    out = []
    for lo in range(0, len(cells_list), batch):
        goals = [G.to_goal(c, "") for c in cells_list[lo : lo + batch]]
        flags = reverse.fixed_point(reverse.W.concat([g.world for g in goals]))
        out += [bool(f) for f in flags]
    return out


def sample(n, seed, names=None, max_factor=6):
    """n distinct goals, generators drawn uniformly, with retries. Returns a list of dicts."""
    rng = np.random.default_rng(seed)
    names = list(names or GENERATORS)
    seen = set()
    out = []
    attempts = 0
    while len(out) < n and attempts < n * max_factor:
        chunk = []
        for _ in range(min(64, n - len(out))):
            attempts += 1
            name = names[int(rng.integers(len(names)))]
            got = draw(name, rng)
            if got is None:
                continue
            key = (name, shape_key(got[0]))
            if key in seen:
                continue
            chunk.append((name, got, key))
        if not chunk:
            continue
        flags = at_rest([c[1][0] for c in chunk])
        for (name, (cells, text, params), key), ok in zip(chunk, flags):
            if ok and key not in seen and len(out) < n:
                seen.add(key)
                out.append({"generator": name, "cells": cells, "text": text, "params": params})
    return out


def measure(name, draws, seed):
    """Acceptance (geometry, text, at rest), distinct shapes and cell counts over `draws` draws."""
    rng = np.random.default_rng(seed)
    got = []
    for _ in range(draws):
        item = draw(name, rng)
        if item is not None:
            got.append(item)
    flags = at_rest([g[0] for g in got]) if got else []
    good = [g for g, f in zip(got, flags) if f]
    counts = [len(g[0]) for g in good]
    return {
        "generator": name,
        "acceptance": round(len(good) / draws, 3),
        "distinct": len({shape_key(g[0]) for g in good}),
        "cells_min": int(min(counts)) if counts else 0,
        "cells_median": int(np.median(counts)) if counts else 0,
        "cells_max": int(max(counts)) if counts else 0,
        "overhang_max": int(max((g[2]["overhang"] for g in good), default=0)),
        "span_max": int(max((g[2]["span"] for g in good), default=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-name", default="gen_medium")
    parser.add_argument("--render-dir", default=None)
    parser.add_argument("--measure", type=int, default=0, help="draws per generator to measure instead of writing")
    args = parser.parse_args()
    if args.measure:
        for name in GENERATORS:
            print(json.dumps(measure(name, args.measure, args.seed)), flush=True)
        return
    items = sample(args.n, args.seed)
    lines = []
    cell_lists = []
    for i, item in enumerate(items):
        cells = item["cells"]
        lines.append(
            {
                "id": f"gen_medium-{args.seed}-{i}",
                "source": "gen_medium",
                "generator": item["generator"],
                "cells": int(len(cells)),
                "dims": [int(v) for v in cells.max(axis=0) + 1],
                "text": item["text"],
                "params": item["params"],
            }
        )
        cell_lists.append(cells.astype(np.uint8))
    G.save(args.out_name, lines, cell_lists)
    counts = {}
    for item in items:
        counts[item["generator"]] = counts.get(item["generator"], 0) + 1
    print(json.dumps({"written": len(items), "per_generator": counts}))
    if args.render_dir:
        from pathlib import Path

        out = Path(args.render_dir)
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(args.seed + 1)
        names = sorted(counts)
        by = {name: [i for i, it in enumerate(items) if it["generator"] == name] for name in names}
        picks = []
        for k in range(30):
            name = names[k % len(names)]
            picks.append(by[name][int(rng.integers(len(by[name])))])
        cards = []
        for k, i in enumerate(picks):
            item = items[i]
            G.render_views(item["cells"]).save(out / f"gen-medium-{k}.png")
            cards.append(
                {
                    "name": f"gen-medium-{k}",
                    "generator": item["generator"],
                    "text": item["text"],
                    "cells": int(len(item["cells"])),
                    "params": item["params"],
                }
            )
        with open(out / "cards-gen-medium.json", "w") as handle:
            json.dump(cards, handle, indent=1)


if __name__ == "__main__":
    main()
