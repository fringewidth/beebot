"""Simple goal shapes: programs that draw end states with random parameters, each with a text.

A generator is `def name(rng) -> (cells, text, params)`: cells is an integer (N, 3) array of (z, y, x) with
z = 0 the floor, normalised so every axis starts at 0. Generators raise Reject for a draw they cannot build.

Rules a goal must pass (check): 4 to 115 cells, at most 16 along each axis, no repeated cell, every cell
face-connected to the floor (goal_library.grounded), overhang at most MAX_OVERHANG, and at rest under the
switch_cube rule (reverse.fixed_point). The overhang of a cell with nothing directly below it is its
Manhattan distance in (y, x) to the nearest cell of the layer below; the goal's overhang is the largest of these.

Directions in texts follow reverse.SIDES: left is x = 0 side, right the x = 15 side, back the y = 0 side and
front the y = 15 side. Texts state counts only below ten.

Usage (repository root):
    .venv/bin/python -m beebot.vla.voxel.goal_gen_simple --n 20000 --seed 0 --render-dir <dir>
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from beebot.vla.voxel import goal_library as G
from beebot.vla.voxel import reverse

SOURCE = "gen_simple"
TRIES = 4  # draws of the chosen generator before sample picks another one
MAX_OVERHANG = 2
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
RUN = {"x": "left to right", "y": "front to back"}
FORBIDDEN = ("cheese", "stripe", "zone", "voxel", "cell")
LARGE_WORD = re.compile(
    r"\b(ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
    r"fifty|sixty|seventy|eighty|ninety|hundred|dozen|dozens)\b"
)
DIGITS = {  # 3 wide, 5 high, top row first; '#' is a cube
    0: ("###", "#.#", "#.#", "#.#", "###"),
    1: (".#.", "##.", ".#.", ".#.", "###"),
    2: ("###", "..#", "###", "#..", "###"),
    3: ("###", "..#", "###", "..#", "###"),
    4: ("#.#", "#.#", "###", "..#", "..#"),
    5: ("###", "#..", "###", "..#", "###"),
    6: ("###", "#..", "###", "#.#", "###"),
    7: ("###", "..#", "..#", "..#", "..#"),
    8: ("###", "#.#", "###", "#.#", "###"),
    9: ("###", "#.#", "###", "..#", "###"),
}


class Reject(Exception):
    """A draw the generator cannot build."""


# ---------------------------------------------------------------------------------------------
# Helpers


def _pick(rng, options):
    return options[int(rng.integers(len(options)))]


def _size(rng, lo, hi):
    """Integer in [lo, hi], log-uniform so small and large sizes are both common (sizes stay below ten, as
    texts state counts only below ten)."""
    return int(round(float(np.exp(rng.uniform(np.log(lo - 0.5), np.log(hi + 0.5))))))


def _axis(rng):
    return _pick(rng, ("x", "y"))


def _sign(rng):
    return _pick(rng, (1, -1))


def _cell(axis, along, across, z):
    """(z, y, x) of a cube at `along` on `axis` (x or y), `across` on the other horizontal axis."""
    if axis == "x":
        return (z, across, along)
    return (z, along, across)


def _cubes(n):
    return f"{WORDS[n]} cube" if n == 1 else f"{WORDS[n]} cubes"


def _side(axis, sign):
    if axis == "x":
        return "right" if sign > 0 else "left"
    return "front" if sign > 0 else "back"


def _finish(cells):
    arr = np.asarray(sorted(set(cells)), dtype=np.int64).reshape(-1, 3)
    return arr - arr.min(axis=0)


def overhang(cells):
    """Largest horizontal distance from a cell with nothing directly below it to the nearest cell of the
    layer below (0 when every cell stands on a cell)."""
    cells = np.asarray(cells, dtype=np.int64)
    occupied = {tuple(c) for c in cells.tolist()}
    worst = 0
    for z, y, x in cells.tolist():
        if z == 0 or (z - 1, y, x) in occupied:
            continue
        below = cells[cells[:, 0] == z - 1]
        if len(below) == 0:
            below = cells[cells[:, 0] < z]
        worst = max(worst, int((np.abs(below[:, 1] - y) + np.abs(below[:, 2] - x)).min()))
    return worst


def check(cells, text):
    """(None, reason) if the goal passes every rule, else (reason, None)."""
    if cells.ndim != 2 or cells.shape[1] != 3 or len(cells) == 0:
        return "shape"
    n = len(cells)
    if not (G.MIN_CELLS <= n <= G.MAX_CELLS):
        return "count"
    if len(np.unique(cells, axis=0)) != n:
        return "duplicate"
    if (cells.min(axis=0) != 0).any() or (cells.max(axis=0) + 1 > G.SIZE).any():
        return "extent"
    if not G.grounded(G.cells_to_vox(cells)):
        return "grounded"
    if overhang(cells) > MAX_OVERHANG:
        return "overhang"
    if not bool(reverse.fixed_point(G.to_goal(cells, text).world).all()):
        return "rest"
    for word in FORBIDDEN:
        if word in text.lower():
            return "forbidden word"
    if LARGE_WORD.search(text.lower()) or any(int(v) >= 10 for v in re.findall(r"\d+", text)):
        return "large number"
    return None


# ---------------------------------------------------------------------------------------------
# Generators


def wall(rng):
    length = _size(rng, 3, 9)
    height = _size(rng, 2, 9)
    axis = _axis(rng)
    cells = [_cell(axis, i, 0, z) for i in range(length) for z in range(height)]
    run = RUN[axis]
    text = _pick(rng, (
        f"build a wall {WORDS[length]} long and {WORDS[height]} high, running {run}",
        f"make a straight wall {WORDS[height]} high and {WORDS[length]} long that stretches {run}",
        f"can you put up a plain wall, {WORDS[height]} high and {WORDS[length]} long, going {run}",
    ))
    return _finish(cells), text, {"length": length, "height": height, "axis": axis}


def staircase(rng):
    steps = _size(rng, 2, 9)
    run = _size(rng, 1, 3)
    width = _size(rng, 1, 8)
    axis = _axis(rng)
    sign = _sign(rng)
    cells = []
    for p in range(steps * run):
        height = p // run + 1
        along = p if sign > 0 else steps * run - 1 - p
        for j in range(width):
            cells += [_cell(axis, along, j, z) for z in range(height)]
    side = _side(axis, sign)
    text = _pick(rng, (
        f"build a staircase of {WORDS[steps]} steps going up toward the {side}",
        f"make stairs rising to the {side}, {WORDS[steps]} steps high at the top",
        f"can you form a set of steps that climbs up to the {side}, {WORDS[steps]} of them",
    ))
    return _finish(cells), text, {"steps": steps, "tread": run, "width": width, "axis": axis, "up_toward": side}


def pyramid(rng):
    a = _size(rng, 3, 9)
    b = _size(rng, 3, 9)
    asked = _size(rng, 2, 6)
    cells = []
    levels = 0
    for k in range(asked):
        ha, hb = a - 2 * k, b - 2 * k
        if ha < 1 or hb < 1:
            break
        cells += [(k, i + k, j + k) for i in range(ha) for j in range(hb)]
        levels += 1
    if levels < 2:
        raise Reject("one level")
    if a == b:
        text = _pick(rng, (
            f"build a pyramid with a square base {WORDS[a]} across, {WORDS[levels]} levels high",
            f"make a pyramid, a square base and {WORDS[levels]} layers stacked and shrinking",
            f"can you stack a pyramid {WORDS[levels]} levels tall on a square base of {WORDS[a]} cubes a side",
        ))
    else:
        text = _pick(rng, (
            f"build a pyramid on a {WORDS[a]} by {WORDS[b]} base, {WORDS[levels]} levels high",
            f"make a rectangular pyramid, {WORDS[levels]} layers, stepping in on every side",
        ))
    return _finish(cells), text, {"base": [a, b], "levels": levels}


def hollow_box(rng):
    sx = _size(rng, 3, 9)
    sy = _size(rng, 3, 9)
    height = _size(rng, 2, 9)
    cells = [(0, y, x) for y in range(sy) for x in range(sx)]
    for z in range(1, height):
        cells += [(z, y, x) for y in range(sy) for x in range(sx) if y in (0, sy - 1) or x in (0, sx - 1)]
    text = _pick(rng, (
        f"build a box {WORDS[sx]} wide, {WORDS[sy]} deep and {WORDS[height]} high, with a floor and open on top",
        f"make a hollow box with walls all round, {WORDS[height]} high, no lid",
        f"can you build an open-topped box, {WORDS[sx]} by {WORDS[sy]} on the floor and {WORDS[height]} high",
    ))
    return _finish(cells), text, {"width": sx, "depth": sy, "height": height}


def solid_block(rng):
    a, b, c = (_size(rng, 2, 9) for _ in range(3))
    cells = [(z, y, x) for z in range(c) for y in range(b) for x in range(a)]
    if a == b == c:
        text = _pick(rng, (
            f"build a solid cube, {WORDS[a]} on each side",
            f"make a filled block shaped like a cube, {WORDS[a]} cubes along every edge",
        ))
    else:
        text = _pick(rng, (
            f"build a solid block {WORDS[a]} wide, {WORDS[b]} deep and {WORDS[c]} tall",
            f"make a filled brick of cubes, {WORDS[a]} across, {WORDS[b]} front to back, {WORDS[c]} high",
        ))
    return _finish(cells), text, {"width": a, "depth": b, "height": c}


def floor_ring(rng):
    a = _size(rng, 3, 9)
    b = _size(rng, 3, 9)
    cells = [(0, 0, x) for x in range(a)] + [(0, b - 1, x) for x in range(a)]
    cells += [(0, y, 0) for y in range(b)] + [(0, y, a - 1) for y in range(b)]
    if a == b:
        text = _pick(rng, (
            f"build a square outline lying flat on the floor, {WORDS[a]} cubes on each side",
            f"make a hollow square on the ground, {WORDS[a]} across, nothing inside",
        ))
    else:
        text = _pick(rng, (
            f"build a rectangle outline on the floor, {WORDS[a]} wide and {WORDS[b]} deep, hollow inside",
            f"make a flat frame on the ground, {WORDS[a]} long left to right and {WORDS[b]} front to back",
        ))
    return _finish(cells), text, {"width": a, "depth": b}


def floor_slab(rng):
    a = _size(rng, 2, 9)
    b = _size(rng, 2, 9)
    cells = [(0, y, x) for y in range(b) for x in range(a)]
    if a == b:
        text = _pick(rng, (
            f"build a solid square lying flat on the floor, {WORDS[a]} on each side",
            f"make a filled square pad on the ground, {WORDS[a]} cubes across",
        ))
    else:
        text = _pick(rng, (
            f"build a flat rectangle on the floor, {WORDS[a]} wide and {WORDS[b]} deep",
            f"make a solid pad on the ground, {WORDS[a]} left to right and {WORDS[b]} front to back",
        ))
    return _finish(cells), text, {"width": a, "depth": b}


def floor_cross(rng):
    ax = int(rng.integers(1, 4))
    ay = int(rng.integers(1, 4))
    t = int(rng.integers(1, 4))
    cells = [(0, y, x) for x in range(-ax, t + ax) for y in range(t)]
    cells += [(0, y, x) for y in range(-ay, t + ay) for x in range(t)]
    wx, wy = 2 * ax + t, 2 * ay + t
    thick = f"{WORDS[t]} cube" if t == 1 else f"{WORDS[t]} cubes"
    if ax == ay:
        text = _pick(rng, (
            f"make a plus sign lying flat on the floor, {WORDS[wx]} by {WORDS[wy]}",
            f"build a cross on the ground, arms {thick} thick",
        ))
    else:
        text = _pick(rng, (
            f"build a cross lying flat on the floor, {WORDS[wx]} wide left to right and {WORDS[wy]} wide front to back",
            f"make a flat cross on the ground, arms {thick} thick, the long bar running left to right",
        ))
    return _finish(cells), text, {"arm_x": ax, "arm_y": ay, "thickness": t}


def l_standing(rng):
    height = int(rng.integers(3, 9))
    foot = int(rng.integers(3, 8))
    axis = _axis(rng)
    sign = _sign(rng)
    cells = [_cell(axis, 0, 0, z) for z in range(height)]
    cells += [_cell(axis, sign * i, 0, 0) for i in range(1, foot)]
    side = _side(axis, sign)
    text = _pick(rng, (
        f"build a tall letter L standing up, {WORDS[height]} high, with its foot pointing {side}",
        f"make an L standing on the floor: upright {WORDS[height]} high, foot {WORDS[foot]} long going {side}",
        f"can you stand an L up, the foot running {side} for {WORDS[foot]} cubes",
    ))
    return _finish(cells), text, {"height": height, "foot": foot, "axis": axis, "foot_toward": side}


def t_standing(rng):
    height = int(rng.integers(3, 7))
    bar = int(rng.choice([3, 4, 5]))
    axis = _axis(rng)
    stem = (bar - 1) // 2
    cells = [_cell(axis, stem, 0, z) for z in range(height - 1)]
    cells += [_cell(axis, i, 0, height - 1) for i in range(bar)]
    run = RUN[axis]
    text = _pick(rng, (
        f"build a T standing up, its bar {WORDS[bar]} long across the top, running {run}, stem down to the floor",
        f"make a letter T, {WORDS[height]} high, with the crossbar at the top",
        f"can you stand a T on the floor, bar of {WORDS[bar]} cubes across the top",
    ))
    return _finish(cells), text, {"height": height, "bar": bar, "axis": axis}


def u_standing(rng):
    width = int(rng.integers(3, 8))
    height = int(rng.integers(2, 6))
    axis = _axis(rng)
    cells = [_cell(axis, i, 0, 0) for i in range(width)]
    for z in range(1, height + 1):
        cells += [_cell(axis, 0, 0, z), _cell(axis, width - 1, 0, z)]
    text = _pick(rng, (
        f"build a U standing up, open at the top, {WORDS[width]} wide and arms {WORDS[height]} high",
        f"make a cup shape standing on the floor, {WORDS[width]} across, open at the top",
        f"can you form the letter U upright, its bottom {WORDS[width]} cubes wide and {WORDS[height]} high",
    ))
    return _finish(cells), text, {"width": width, "arm_height": height, "axis": axis}


def digit_standing(rng):
    digit = int(rng.integers(0, 10))
    axis = _axis(rng)
    depth = int(rng.integers(1, 3))
    cells = []
    for r, row in enumerate(DIGITS[digit]):
        for c, ch in enumerate(row):
            if ch == "#":
                cells += [_cell(axis, c, k, 4 - r) for k in range(depth)]
    thick = "" if depth == 1 else f", {WORDS[depth]} cubes thick"
    text = _pick(rng, (
        f"build the digit {digit} standing up{thick}, its width running {RUN[axis]}",
        f"make a {digit} out of cubes, standing upright{thick}, the wide part running {RUN[axis]}",
        f"can you form the number {digit} as an upright sign{thick}, its width running {RUN[axis]}",
    ))
    return _finish(cells), text, {"digit": digit, "axis": axis, "depth": depth}


def zigzag_wall(rng):
    height = int(rng.integers(2, 5))
    legs = int(rng.integers(3, 5))
    span = int(rng.integers(2, 5))
    sign = _sign(rng)
    plan = []
    x0 = 0
    for leg in range(legs):
        y = 2 * leg
        xs = [x0 + sign * i for i in range(span)]
        plan += [(y, x) for x in xs]
        if leg < legs - 1:
            plan.append((y + 1, xs[-1]))
        x0 = xs[-1]
        sign = -sign
    cells = [(z, y, x) for y, x in plan for z in range(height)]
    text = _pick(rng, (
        f"build a zigzag wall {WORDS[height]} high that doubles back {WORDS[legs - 1]} times",
        f"make a wall {WORDS[height]} tall that zigzags, turning back on itself {WORDS[legs - 1]} times",
        f"can you put up a zigzag of wall, {WORDS[height]} high, with {WORDS[legs - 1]} sharp turns",
    ))
    return _finish(cells), text, {"height": height, "legs": legs, "leg_length": span}


def tower_row(rng):
    n = int(rng.integers(2, 6))
    step = int(rng.integers(1, 3))
    start = int(rng.integers(1, 4))
    grow = _sign(rng)
    axis = _axis(rng)
    heights = [start + step * k for k in range(n)]
    if max(heights) > 8:
        raise Reject("too tall")
    cells = []
    for k, h in enumerate(heights):
        pos = 2 * k if grow > 0 else 2 * (n - 1 - k)
        cells += [_cell(axis, pos, 0, z) for z in range(h)]
    side = _side(axis, grow)
    text = _pick(rng, (
        f"build {WORDS[n]} towers getting taller toward the {side}",
        f"a row of {WORDS[n]} separate towers, each {_cubes(step)} taller than the one before, rising to the {side}",
        f"line up {WORDS[n]} columns with gaps between, growing taller to the {side}",
    ))
    return _finish(cells), text, {"towers": n, "heights": heights, "step": step, "axis": axis, "grows_toward": side}


def post_row(rng):
    n = int(rng.integers(2, 7))
    height = int(rng.integers(1, 6))
    axis = _axis(rng)
    cells = [_cell(axis, 2 * i, 0, z) for i in range(n) for z in range(height)]
    text = _pick(rng, (
        f"build a row of {WORDS[n]} separate posts, each {WORDS[height]} high, with a gap between each",
        f"make {WORDS[n]} posts standing in a line {RUN[axis]}, {WORDS[height]} high, not joined",
        f"can you stand {WORDS[n]} short posts in a row, {WORDS[height]} cubes tall, spaced apart",
    ))
    return _finish(cells), text, {"posts": n, "height": height, "axis": axis}


def checker_posts(rng):
    a = int(rng.integers(3, 6))
    b = int(rng.integers(3, 6))
    height = int(rng.integers(1, 3))
    parity = int(rng.integers(0, 2))
    cells = [(z, i, j) for i in range(a) for j in range(b) if (i + j) % 2 == parity for z in range(height)]
    text = _pick(rng, (
        f"make a checkerboard of short posts on a {WORDS[a]} by {WORDS[b]} patch, {WORDS[height]} high",
        f"build posts on every other square of a {WORDS[a]} by {WORDS[b]} grid, like a checkerboard, {WORDS[height]} tall",
        f"can you lay out a checkerboard pattern of little posts, {_cubes(height)} high, {WORDS[a]} by {WORDS[b]}",
    ))
    return _finish(cells), text, {"rows": a, "cols": b, "height": height, "parity": parity}


def frame_standing(rng):
    width = int(rng.integers(3, 6))
    height = int(rng.integers(3, 7))
    axis = _axis(rng)
    cells = [_cell(axis, i, 0, 0) for i in range(width)] + [_cell(axis, i, 0, height - 1) for i in range(width)]
    cells += [_cell(axis, 0, 0, z) for z in range(height)] + [_cell(axis, width - 1, 0, z) for z in range(height)]
    shape = "square" if width == height else "rectangle"
    text = _pick(rng, (
        f"build a {shape} frame standing up like a window, {WORDS[width]} across and {WORDS[height]} high, sitting on the floor",
        f"make an upright {shape} outline with its bottom edge on the ground, {WORDS[width]} wide, {WORDS[height]} tall",
        f"can you stand a hollow {shape} frame on the floor, {WORDS[width]} wide {RUN[axis]} and {WORDS[height]} high",
    ))
    return _finish(cells), text, {"width": width, "height": height, "axis": axis}


def ladder(rng):
    width = int(rng.integers(3, 6))
    height = int(rng.integers(4, 8))
    axis = _axis(rng)
    cells = [_cell(axis, 0, 0, z) for z in range(height)] + [_cell(axis, width - 1, 0, z) for z in range(height)]
    rungs = list(range(1, height - 1, 2))
    cells += [_cell(axis, i, 0, z) for z in rungs for i in range(1, width - 1)]
    text = _pick(rng, (
        f"make a ladder standing up, {WORDS[height]} high and {WORDS[width]} wide, with rungs running {RUN[axis]}",
        f"build a small ladder on the floor, {WORDS[len(rungs)]} rungs, {WORDS[width]} cubes between the rails",
        f"can you stand a ladder up, two rails {WORDS[height]} tall joined by rungs",
    ))
    return _finish(cells), text, {"width": width, "height": height, "rungs": len(rungs), "axis": axis}


GENERATORS = {
    "wall": wall,
    "staircase": staircase,
    "pyramid": pyramid,
    "hollow_box": hollow_box,
    "solid_block": solid_block,
    "floor_ring": floor_ring,
    "floor_slab": floor_slab,
    "floor_cross": floor_cross,
    "l_standing": l_standing,
    "t_standing": t_standing,
    "u_standing": u_standing,
    "digit_standing": digit_standing,
    "zigzag_wall": zigzag_wall,
    "tower_row": tower_row,
    "post_row": post_row,
    "checker_posts": checker_posts,
    "frame_standing": frame_standing,
    "ladder": ladder,
}


# ---------------------------------------------------------------------------------------------
# Drawing and sampling


def draw(name, rng):
    """One draw of generator `name`: (goal dict, None) when accepted, else (None, reason)."""
    try:
        cells, text, params = GENERATORS[name](rng)
    except Reject as err:
        return None, f"reject: {err}"
    cells = np.asarray(cells, dtype=np.int64)
    why = check(cells, text)
    if why is not None:
        return None, why
    goal = {
        "generator": name,
        "cells": cells,
        "text": text,
        "params": params,
        "overhang": overhang(cells),
        "dims": [int(v) for v in cells.max(axis=0) + 1],
    }
    return goal, None


def sample(n, seed=0):
    """n accepted goals. Each goal draws a generator uniformly, retries that generator up to TRIES times, then
    draws another generator."""
    rng = np.random.default_rng(seed)
    names = list(GENERATORS)
    out = []
    draws = 0
    while len(out) < n:
        name = _pick(rng, names)
        goal = None
        for _ in range(TRIES):
            draws += 1
            goal, _ = draw(name, rng)
            if goal is not None:
                break
        if goal is not None:
            out.append(goal)
        if draws > 50 * n + 1000:
            raise RuntimeError(f"only {len(out)} of {n} goals accepted in {draws} draws")
    return out


def measure(draws=2000, seed=1):
    """Per generator: acceptance over `draws` raw draws, distinct accepted shapes, cell-count min/median/max, and
    the rejection reasons."""
    rng = np.random.default_rng(seed)
    rows = []
    for name in GENERATORS:
        accepted = []
        reasons = {}
        for _ in range(draws):
            goal, why = draw(name, rng)
            if goal is None:
                reasons[why] = reasons.get(why, 0) + 1
            else:
                accepted.append(goal)
        shapes = {goal["cells"][np.lexsort(goal["cells"].T[::-1])].tobytes() for goal in accepted}
        counts = [len(goal["cells"]) for goal in accepted] or [0]
        rows.append({
            "name": name,
            "accept": len(accepted) / draws,
            "distinct": len(shapes),
            "min": min(counts),
            "median": float(np.median(counts)),
            "max": max(counts),
            "reasons": reasons,
        })
    return rows


def _spread(goals, k, seed):
    """k goals with as even a spread over generators as the pool allows."""
    rng = np.random.default_rng(seed)
    by_name = {}
    for goal in goals:
        by_name.setdefault(goal["generator"], []).append(goal)
    names = list(by_name)
    chosen = []
    round_ = 0
    while len(chosen) < k:
        for name in names:
            if len(chosen) >= k:
                break
            pool = by_name[name]
            if round_ < len(pool):
                chosen.append(pool[int(rng.integers(len(pool)))])
        round_ += 1
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-name", default="gen_simple")
    parser.add_argument("--out-root", default=str(G.LIBRARY))
    parser.add_argument("--render-dir", default=None)
    parser.add_argument("--measure-draws", type=int, default=2000)
    args = parser.parse_args()

    rows = measure(args.measure_draws, seed=args.seed + 1)
    print(f"{'generator':<16} {'accept':>6} {'distinct':>8} {'min':>4} {'median':>6} {'max':>4}  rejects")
    for row in rows:
        reasons = ", ".join(f"{k} {v}" for k, v in sorted(row["reasons"].items(), key=lambda kv: -kv[1]))
        print(f"{row['name']:<16} {row['accept']:>6.2f} {row['distinct']:>8d} {row['min']:>4d} "
              f"{row['median']:>6.1f} {row['max']:>4d}  {reasons}")

    goals = sample(args.n, seed=args.seed)
    lines = []
    cell_lists = []
    for k, goal in enumerate(goals):
        lines.append({
            "id": f"{SOURCE}-{k:06d}",
            "source": SOURCE,
            "generator": goal["generator"],
            "cells": len(goal["cells"]),
            "dims": goal["dims"],
            "text": goal["text"],
            "params": goal["params"],
            "overhang": goal["overhang"],
        })
        cell_lists.append(goal["cells"].astype(np.uint8))
    G.save(args.out_name, lines, cell_lists, root=args.out_root)
    print(f"wrote {len(lines)} goals to {Path(args.out_root) / (args.out_name + '.jsonl')}")

    if args.render_dir:
        render = Path(args.render_dir)
        render.mkdir(parents=True, exist_ok=True)
        picks = _spread(goals, 30, seed=args.seed + 2)
        cards = []
        for k, goal in enumerate(picks, start=1):
            name = f"gen-simple-{k}"
            G.render_views(goal["cells"]).save(render / f"{name}.png")
            cards.append({
                "name": name,
                "generator": goal["generator"],
                "text": goal["text"],
                "cells": goal["cells"].tolist(),
                "params": goal["params"],
            })
        (render / "cards-gen-simple.json").write_text(json.dumps(cards, indent=1))
        print(f"rendered {len(cards)} samples to {render}")


if __name__ == "__main__":
    main()
