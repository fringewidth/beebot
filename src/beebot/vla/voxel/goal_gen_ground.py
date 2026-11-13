"""Grounding goal generators: complete the picture, name a cube by what is visible, match and count, ask.

The policy must learn which cube in the picture is a cube it controls and where a missing cube has to go, from what
it sees. Same output as goal_gen_complex (a dict per goal, one user turn, A in `start_cells` and B in `cells`):

    cells        (N, 3) (z, y, x) world cells of the end state B, every robot (parked ones included)
    start_cells  the start A, same robot count (a question: A = B)
    turns        one user text (the picture is the context)
    reply        what the model should say (questions; empty otherwise)
    kind         complete, refer, match or ask
    params       k, rule, refs, changed (cells that fill and empty), parked, overhang, span, source

Kinds:
1. complete: a structure with k = 1, 2, 3, ... (up to 8) cells missing, the k robots standing elsewhere (parked in a
   row at an edge, or loose on the floor nearby). The text never says where. The missing cells follow from what is
   visible by a rule: symmetry (a mirror or a half turn), pattern (a repeat along the floor), step (a repeat that
   rises) or face (a hole in a solid wall). Ambiguity is checked: over every mirror, half turn and repeat that fits
   the visible structure, no rule may ask for a different set of at most k cells.
1. refer: one cube named by what is visible ("the cube on top of the tallest tower", "the lone cube nearest me",
   "the block sticking out of the side of the wall"), moved somewhere stated. Exactly one cube fits (checked by
   ref_candidates, which the tests repeat).
1. match: "make the left tower as tall as the right one", "add blocks until both walls are the same length", "a second
   one just like this", "copy this to the other side, mirrored". The end state follows from the picture; the blocks
   come from a parked row.
1. ask: a question about the picture, no action (the end state equals the start); the reply carries the answer.

Texts use plain words, never cheese, stripes, zones, voxels or cells, never a count of ten or more.

    .venv/bin/python -m beebot.vla.voxel.goal_gen_ground --n 20000 --seed 0 --render-dir <dir>
    .venv/bin/python -m beebot.vla.voxel.goal_gen_ground --measure 1000
    .venv/bin/python -m beebot.vla.voxel.goal_gen_ground --reverse 3
"""

import argparse
import hashlib
import json
import textwrap
import time
import zlib
from pathlib import Path

import numpy as np

from beebot.vla.voxel import goal_gen_complex as C
from beebot.vla.voxel import goal_library as G

SIZE = C.SIZE
SOURCE = "gen_ground"
Reject = C.Reject
need = C.need
pick = C.pick
chance = C.chance
between = C.between
K_WEIGHTS = (0.20, 0.17, 0.15, 0.13, 0.11, 0.09, 0.08, 0.07)  # k = 1 .. 8: the curriculum starts with one block
FAMILY_WEIGHTS = {"symmetry": 3.0, "pattern": 3.0, "step": 2.0, "face": 2.0}


# ---------------------------------------------------------------------------------------------
# Voxel helpers


def vox(cells):
    v = np.zeros((SIZE, SIZE, SIZE), dtype=bool)
    if cells:
        a = np.asarray(sorted(cells))
        v[a[:, 0], a[:, 1], a[:, 2]] = True
    return v


def to_cells(v):
    return {tuple(int(i) for i in c) for c in np.argwhere(v)}


def shift_by(v, d):
    """out[c] = v[c - d]: the contents moved by d (cells pushed off the floor are dropped)."""
    out = np.zeros_like(v)
    src, dst = [], []
    for di in d:
        if abs(di) >= SIZE:
            return out
        src.append(slice(0, SIZE - di) if di >= 0 else slice(-di, SIZE))
        dst.append(slice(di, SIZE) if di >= 0 else slice(0, SIZE + di))
    out[tuple(dst)] = v[tuple(src)]
    return out


def normalise(cells):
    """Cells moved so the lowest layer is z = 0 and the footprint starts at (0, 0)."""
    arr = np.asarray(sorted(cells))
    arr = arr - np.array([arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].min()])
    return {tuple(int(i) for i in c) for c in arr}


def fit_sets(sets, rng, reserve=None):
    """The cell sets shifted together by one random (dy, dx) so they lie inside the floor."""
    group = [C.Placed(C.Part("custom", {"cells": frozenset(s)}), 0, 0) for s in sets]
    out = C.fit([group[:1], group[1:]] if len(group) > 1 else [group], rng, reserve=reserve)
    return [set(p.cells) for g in out for p in g]


def column(y, x, h, z0=0):
    return {(z, y, x) for z in range(z0, z0 + h)}


# ---------------------------------------------------------------------------------------------
# Rules: what the visible structure implies


def _vecs():
    out = []
    for dz in (0, 1):
        reach = 6 if dz == 0 else 3
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                if dz == 0 and (dy < 0 or (dy == 0 and dx <= 0)):
                    continue
                out.append((dz, dy, dx))
    return out


VECS = _vecs()


def shift_family(d):
    dz, dy, dx = d
    if dz == 0:
        return "face" if abs(dy) + abs(dx) == 1 else "pattern"
    return "face" if (dy, dx) == (0, 0) else "step"


def all_rules(v, k):
    """(rule, family, T) for every mirror, half turn and repeat near the visible structure v: T is the set of
    cells the rule would add (the empty cells whose mirror image / both neighbours at the repeat are present)."""
    idx = np.argwhere(v)
    y0, y1, x0, x1 = idx[:, 1].min(), idx[:, 1].max(), idx[:, 2].min(), idx[:, 2].max()
    w = k + 1
    for c2 in range(int(x0 + x1) - w, int(x0 + x1) + w + 1):
        x2 = c2 - idx[:, 2]
        if x2.min() < 0 or x2.max() >= SIZE:
            continue
        img = np.zeros_like(v)
        img[idx[:, 0], idx[:, 1], x2] = True
        yield ("mirror_x", c2), "symmetry", img & ~v
    for c2 in range(int(y0 + y1) - w, int(y0 + y1) + w + 1):
        y2 = c2 - idx[:, 1]
        if y2.min() < 0 or y2.max() >= SIZE:
            continue
        img = np.zeros_like(v)
        img[idx[:, 0], y2, idx[:, 2]] = True
        yield ("mirror_y", c2), "symmetry", img & ~v
    for cy in range(int(y0 + y1) - w, int(y0 + y1) + w + 1):
        for cx in range(int(x0 + x1) - w, int(x0 + x1) + w + 1):
            y2, x2 = cy - idx[:, 1], cx - idx[:, 2]
            if min(y2.min(), x2.min()) < 0 or max(y2.max(), x2.max()) >= SIZE:
                continue
            img = np.zeros_like(v)
            img[idx[:, 0], y2, x2] = True
            yield ("turn", cy, cx), "symmetry", img & ~v
    for d in VECS:
        both = shift_by(v, d) & shift_by(v, tuple(-i for i in d))
        yield ("shift", d), shift_family(d), both & ~v


def explain(v, holes, k):
    """(rules whose completion is exactly the holes, rules that ask for a different set of at most k cells)."""
    hv = vox(holes)
    match, bad = [], []
    for rule, family, T in all_rules(v, k):
        n = int(T.sum())
        if n == 0:
            continue
        if np.array_equal(T, hv):
            match.append((rule, family))
        elif n <= k and (T & ~hv).any():
            bad.append(rule)
    return match, bad


def rule_name(rule):
    if rule[0] == "shift":
        return "repeat " + ",".join(str(i) for i in rule[1])
    return rule[0]


def partners(rule, c):
    z, y, x = c
    if rule[0] == "mirror_x":
        return [(z, y, rule[1] - x)]
    if rule[0] == "mirror_y":
        return [(z, rule[1] - y, x)]
    if rule[0] == "turn":
        return [(z, rule[1] - y, rule[2] - x)]
    dz, dy, dx = rule[1]
    return [(z + dz, y + dy, x + dx), (z - dz, y - dy, x - dx)]


def intended(v):
    """[(rule, family, removable cells)] for the rules the complete structure v obeys exactly."""
    idx = np.argwhere(v)
    y0, y1, x0, x1 = idx[:, 1].min(), idx[:, 1].max(), idx[:, 2].min(), idx[:, 2].max()
    out = []
    cx2, cy2 = int(x0 + x1), int(y0 + y1)
    for rule in (("mirror_x", cx2), ("mirror_y", cy2), ("turn", cy2, cx2)):
        mapped = np.stack([partners(rule, tuple(int(i) for i in c))[0] for c in idx])
        if mapped.min() < 0 or mapped.max() >= SIZE:
            continue
        img = np.zeros_like(v)
        img[mapped[:, 0], mapped[:, 1], mapped[:, 2]] = True
        if not np.array_equal(img, v):
            continue
        cells = [tuple(int(i) for i in c) for c, m in zip(idx, mapped) if tuple(m) != tuple(c)]
        if cells:
            out.append((rule, "symmetry", cells))
    for d in VECS:
        if d == (1, 0, 0):  # a dent in a tower reads poorly; the rule still counts in the ambiguity check
            continue
        both = shift_by(v, d) & shift_by(v, tuple(-i for i in d))
        if (both & ~v).any():
            continue
        cells = to_cells(both & v)
        if cells:
            out.append((("shift", d), shift_family(d), sorted(cells)))
    return out


def visible(c, rest):
    """The hole shows from the viewer's side (+y): either it is see-through in the front elevation with a filled
    neighbour beside or under it, or nothing of the structure stands in front of or above it (a dent), and it has
    an open side."""
    z, y, x = c
    elevation = rest.any(axis=1)
    near = any(
        0 <= n[0] < SIZE and 0 <= n[1] < SIZE and elevation[n]
        for n in ((z - 1, x), (z, x - 1), (z, x + 1))
    )
    see_through = not elevation[z, x] and near
    dent = not rest[z, y + 1 :, x].any() and not rest[z + 1 :, y, x].any()
    open_side = any(
        not all(0 <= i < SIZE for i in n) or not rest[n]
        for n in ((z + 1, y, x), (z, y + 1, x), (z, y - 1, x), (z, y, x + 1), (z, y, x - 1))
    )
    return (see_through or dent) and open_side


def choose_holes(rng, S, k):
    """k cells of the complete structure S whose absence the rest implies, with the rule: (holes, rule, family,
    all matching rules)."""
    v = vox(S)
    options = [o for o in intended(v) if len(o[2]) >= k]
    need(options, "no rule")
    families = sorted({o[1] for o in options})
    weights = np.array([FAMILY_WEIGHTS[f] for f in families])
    for _ in range(8):
        family = families[int(rng.choice(len(families), p=weights / weights.sum()))]
        rule, _, cells = pick(rng, [o for o in options if o[1] == family])
        taken = []
        for i in rng.permutation(len(cells)):
            c = cells[int(i)]
            if any(p in taken for p in partners(rule, c)):
                continue
            if family != "symmetry" and any(sum(abs(a - b) for a, b in zip(c, t)) == 1 for t in taken):
                continue  # gaps stay separate: a run of missing cells would read as a window drawn on purpose
            taken.append(c)
            if len(taken) == k:
                break
        if len(taken) < k:
            continue
        holes = set(taken)
        rest = S - holes
        rv = vox(rest)
        if not G.grounded(rv) or not all(visible(c, rv) for c in holes):
            continue
        match, bad = explain(rv, holes, k)
        if bad or not any(r == rule for r, _ in match):
            continue
        return holes, rule, family, sorted({rule_name(r) for r, _ in match})
    raise Reject("holes")


# ---------------------------------------------------------------------------------------------
# Complete the picture: sources


def p_row_towers(rng):
    n, gap, h = between(rng, 3, 6), between(rng, 1, 2), between(rng, 3, 6)
    alternate = chance(rng, 0.3)
    cells = set()
    for i in range(n):
        cells |= column(0, i * (gap + 1), h - 2 if alternate and i % 2 else h)
    return cells if chance(rng, 0.8) else {(z, x, y) for z, y, x in cells}


def p_battlement(rng):
    length, base, deep = between(rng, 7, 13), between(rng, 1, 3), between(rng, 1, 2)
    cells = {(z, y, x) for z in range(base) for y in range(deep) for x in range(length)}
    cells |= {(base, y, x) for y in range(deep) for x in range(0, length, 2)}
    return cells if chance(rng, 0.8) else {(z, x, y) for z, y, x in cells}


def p_colonnade(rng):
    n, gap, rows, h = between(rng, 3, 5), between(rng, 1, 2), between(rng, 1, 2), between(rng, 2, 3)
    cells = set()
    for r in range(rows):
        for i in range(n):
            cells |= column(r * 3, i * (gap + 1), h)
        cells |= {(h, r * 3, x) for x in range((n - 1) * (gap + 1) + 1)}
    return cells if chance(rng, 0.8) else {(z, x, y) for z, y, x in cells}


def p_staircase(rng):
    width, steps = between(rng, 1, 3), between(rng, 4, 7)
    cells = {(z, -i, j) for i in range(steps) for z in range(i + 1) for j in range(width)}
    return normalise({(z, x, y) for z, y, x in cells} if chance(rng, 0.85) else cells)  # along x: its profile faces the viewer


def p_bridge(rng):
    pier, gap, spans, deep = between(rng, 2, 3), between(rng, 1, 2), between(rng, 2, 4), between(rng, 1, 2)
    p = gap + 1
    length = spans * p
    cells = {(z, y, x) for z in range(pier) for y in range(deep) for x in range(0, length + 1, p)}
    cells |= {(pier, y, x) for y in range(deep) for x in range(length + 1)}
    return cells if chance(rng, 0.8) else {(z, x, y) for z, y, x in cells}


def p_tower_grid(rng):
    gx, gy, gap, h = between(rng, 2, 4), between(rng, 2, 3), between(rng, 1, 2), between(rng, 2, 5)
    cells = set()
    for i in range(gy):
        for j in range(gx):
            cells |= column(i * (gap + 1), j * (gap + 1), h)
    return cells


PATTERNS = {
    "row_towers": p_row_towers, "battlement": p_battlement, "colonnade": p_colonnade, "staircase": p_staircase,
    "bridge": p_bridge, "tower_grid": p_tower_grid,
}
SCENES = ("scene_between", "scene_wall_towers", "scene_around", "scene_row3", "scene_compare", "scene_on_top", "scene_pair")


def source_pattern(rng):
    names = sorted(PATTERNS)
    weights = np.array([0.5 if n == "staircase" else 1.0 for n in names])
    name = names[int(rng.choice(len(names), p=weights / weights.sum()))]
    return normalise(PATTERNS[name](rng)), name


def source_part(rng):
    kind = C.rand_kind(rng, exclude=("tower",))
    part = C.rand_part(rng, kind)
    return normalise(part.cells), kind


def source_scene(rng):
    name = pick(rng, SCENES)
    goal = C.GENERATORS[name][0](rng)
    park = goal["params"]["parked"]
    parked = set(map(tuple, C.park_slots(park["side"], C.MAX_ROWS)[: park["b"]])) if park.get("side") else set()
    cells = {tuple(c) for c in goal["cells"]} - parked
    need(goal["params"].get("scale", 1) == 1, "scaled")
    return normalise(cells), name


def source_shape(rng):
    cells, _, name = C.external_shape(rng)
    return normalise(cells), name


# ---------------------------------------------------------------------------------------------
# Complete the picture: texts and records


GAP_TEXT = {
    0: ["finish it", "complete the picture", "complete it", "make it whole", "fix it", "finish what is here", "can you finish this?", "complete this", "finish the picture"],
    1: ["one block is missing, put it back", "fill in the missing piece", "there is a gap, fill it in", "a block is missing, can you put it back?", "one piece is missing, finish it", "put the missing block back", "there is one block missing, fix it"],
    2: ["{n} blocks are missing, put them back", "there are {n} gaps, fix them", "fill in the {n} missing pieces", "{n} pieces are missing, finish it", "put the {n} missing blocks back"],
    3: ["some blocks are missing, put them back", "fill in the gaps", "there are gaps, fix them", "a few pieces are missing, finish it"],
}


def complete_text(rng, k):
    if chance(rng, 0.4):
        text = pick(rng, GAP_TEXT[0])
    elif k == 1:
        text = pick(rng, GAP_TEXT[1])
    elif k <= 4 and chance(rng, 0.7):
        text = pick(rng, GAP_TEXT[2]).format(n=C.N(k))
    elif k > 1 and chance(rng, 0.5):
        text = pick(rng, GAP_TEXT[3])
    else:
        text = pick(rng, GAP_TEXT[0])
    return C.polish(text, rng)


def loose_cells(rng, structure, k):
    """k floor cells near the structure, not touching it or each other (a row two apart, or scattered)."""
    plan = np.zeros((SIZE, SIZE), dtype=bool)
    for _, y, x in structure:
        plan[y, x] = True
    near, far = plan.copy(), plan.copy()
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            moved = np.zeros_like(plan)
            ys, xs = slice(max(dy, 0), SIZE + min(dy, 0)), slice(max(dx, 0), SIZE + min(dx, 0))
            moved[ys, xs] = plan[slice(max(-dy, 0), SIZE + min(-dy, 0)), slice(max(-dx, 0), SIZE + min(-dx, 0))]
            if max(abs(dy), abs(dx)) <= 1:
                far |= moved
            if max(abs(dy), abs(dx)) <= 5:
                near |= moved
    free = [(int(y), int(x)) for y, x in np.argwhere(near & ~far)]
    need(len(free) >= k, "no room for loose cubes")
    freeset = set(free)
    if chance(rng, 0.5):
        for _ in range(12):
            y, x = free[int(rng.integers(len(free)))]
            dy, dx = pick(rng, [(0, 2), (2, 0), (0, -2), (-2, 0)])
            row = [(y + i * dy, x + i * dx) for i in range(k)]
            if all(r in freeset for r in row):
                return {(0, y, x) for y, x in row}
    out = []
    for i in rng.permutation(len(free)):
        y, x = free[int(i)]
        if all(max(abs(y - a), abs(x - b)) >= 2 for a, b in out):
            out.append((y, x))
        if len(out) == k:
            return {(0, y, x) for y, x in out}
    raise Reject("loose cubes")


def draw_k(rng):
    return 1 + int(rng.choice(len(K_WEIGHTS), p=np.array(K_WEIGHTS) / sum(K_WEIGHTS)))


def complete_goal(rng, source_fn, label):
    """One k, drawn once; a source that cannot give k implied holes is redrawn a few times (higher k fails more
    often, and the written set keeps the k mix)."""
    k = draw_k(rng)
    for _ in range(6 + 6 * (k - 1)):
        try:
            return complete_k(rng, source_fn, label, k)
        except Reject:
            continue
    raise Reject("holes")


def complete_k(rng, source_fn, label, k):
    base, source = source_fn(rng)
    need(max(10, 3 * k + 4) <= len(base) <= 220, "size")
    mode = pick(rng, ["park", "floor"])
    side = pick(rng, C.SIDES) if mode == "park" else None
    (S,) = fit_sets([base], rng, reserve=(side, 1) if side else None)
    holes, rule, family, rules = choose_holes(rng, S, k)
    rest = S - holes
    if mode == "park":
        a_cells, b_cells, info = C.with_parking(rng, rest, S, side=side)
    else:
        a_cells, b_cells, info = rest | loose_cells(rng, S, k), set(S), {"side": None, "a": 0, "b": 0}
    out = C.record("complete", b_cells, [complete_text(rng, k)], start=a_cells, parking=info, k=k, rule=family, rules=rules, source=source, label=label, where=mode)
    return out


def gen_complete_pattern(rng):
    return complete_goal(rng, source_pattern, "pattern builder")


def gen_complete_part(rng):
    return complete_goal(rng, source_part, "part")


def gen_complete_scene(rng):
    return complete_goal(rng, source_scene, "scene")


def gen_complete_shape(rng):
    return complete_goal(rng, source_shape, "shape")


# ---------------------------------------------------------------------------------------------
# Cubes named by what is visible


def column_towers(cells):
    """[(y, x, h)] for every free-standing column of height 2 or more (nothing within one step in the plan)."""
    cols = {}
    for z, y, x in cells:
        cols.setdefault((y, x), []).append(z)
    plan = set(cols)
    out = []
    for (y, x), zs in sorted(cols.items()):
        alone = all((y + dy, x + dx) not in plan for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
        if alone and sorted(zs) == list(range(len(zs))) and len(zs) >= 2:
            out.append((y, x, len(zs)))
    return out


def lone_cubes(cells):
    cs = set(map(tuple, cells))
    return sorted(c for c in cs if not any((c[0] + a, c[1] + b, c[2] + d) in cs for a in (-1, 0, 1) for b in (-1, 0, 1) for d in (-1, 0, 1) if (a, b, d) != (0, 0, 0)))


def ref_candidates(cells, ref):
    """Every cube of `cells` that the description `ref` fits (a good description fits exactly one)."""
    cs = set(map(tuple, cells))
    which = ref["which"]
    if ref["type"] == "tower":
        towers = column_towers(cs)
        if not towers:
            return []
        xs = sorted({t[1] for t in towers})
        if which == "tallest":
            match = [t for t in towers if t[2] == max(u[2] for u in towers)]
        elif which == "shortest":
            match = [t for t in towers if t[2] == min(u[2] for u in towers)]
        elif which == "left":
            match = [t for t in towers if t[1] == xs[0]]
        elif which == "right":
            match = [t for t in towers if t[1] == xs[-1]]
        else:  # middle
            match = [t for t in towers if len(xs) % 2 == 1 and t[1] == xs[len(xs) // 2]]
        return [(h - 1, y, x) for y, x, h in match]
    if ref["type"] == "lone":
        lone = lone_cubes(cs)
        if not lone:
            return []
        key = {"nearest": lambda c: -c[1], "farthest": lambda c: c[1], "left": lambda c: c[2], "right": lambda c: -c[2]}[which]
        best = min(key(c) for c in lone)
        return [c for c in lone if key(c) == best]
    ys = [c[1] for c in cs]  # bump: cells off the wall's line
    line = max(set(ys), key=ys.count)
    return sorted(c for c in cs if c[1] != line)


def tower_scene(heights, gap, y=0):
    cells, xs = set(), []
    for i, h in enumerate(heights):
        xs.append(i * (gap + 1))
        cells |= column(y, xs[-1], h)
    return cells, xs


def refer_record(rng, a, b, text, refs, **extra):
    a, b = fit_sets([a, b], rng)
    for ref in refs:
        found = ref_candidates(a, ref)
        need(len(found) == 1, "reference not unique")
        ref["target"] = list(found[0])
    return C.record("refer", b, [C.polish(text, rng)], start=a, refs=refs, **extra)


def gen_refer_top_to_floor(rng):
    n, gap = between(rng, 2, 4), between(rng, 1, 3)
    heights = [int(h) for h in rng.permutation(np.arange(3, 8))[:n]]
    S, xs = tower_scene(heights, gap)
    which = pick(rng, ["tallest", "shortest"])
    t = heights.index(max(heights) if which == "tallest" else min(heights))
    top, dest = (heights[t] - 1, 0, xs[t]), (0, 1, xs[t])
    text = pick(rng, [
        f"move the cube on top of the {which} tower down to the floor, in front of it",
        f"take the top block off the {which} tower and put it on the floor in front of the tower",
        f"put the top cube of the {which} tower on the floor just in front of it",
        f"the {which} tower: lower its top block to the floor in front",
    ])
    return refer_record(rng, S, (S - {top}) | {dest}, text, [{"type": "tower", "which": which}], pattern="top_to_floor")


def gen_refer_top_move(rng):
    n, gap = between(rng, 2, 4), between(rng, 1, 3)
    heights = [int(h) for h in rng.permutation(np.arange(3, 8))[:n]]
    S, xs = tower_scene(heights, gap)
    names = ["left", "right", "tallest", "shortest"] + (["middle"] if n == 3 else [])
    src, dst = [int(i) for i in rng.permutation(len(names))[:2]]
    sw, dw = names[src], names[dst]

    def index(which):
        if which == "left":
            return 0
        if which == "right":
            return n - 1
        if which == "middle":
            return 1
        return heights.index(max(heights) if which == "tallest" else min(heights))

    i, j = index(sw), index(dw)
    need(i != j, "same tower")
    top = (heights[i] - 1, 0, xs[i])
    new = (heights[j], 0, xs[j])
    text = pick(rng, [
        f"take the top block off the {sw} tower and put it on the {dw} one",
        f"move the top cube of the {sw} tower onto the {dw} tower",
        f"the {sw} tower gives its top block to the {dw} tower",
        f"put the cube from the top of the {sw} tower on top of the {dw} tower",
    ])
    refs = [{"type": "tower", "which": sw}, {"type": "tower", "which": dw}]
    a, b = fit_sets([S, (S - {top}) | {new}], rng)
    for ref in refs:
        need(len(ref_candidates(a, ref)) == 1, "reference not unique")
    refs[0]["target"] = list(top_of(a - b))
    refs[1]["target"] = list(top_of(b - a, below=True))
    return C.record("refer", b, [C.polish(text, rng)], start=a, refs=refs, pattern="top_move")


def top_of(cells, below=False):
    (c,) = cells
    return (c[0] - 1, c[1], c[2]) if below else c


def lone_scene(rng, n):
    """Loose cubes on the floor with distinct y and x, plus the local frame of the structure (a row at y = 0)."""
    cubes = []
    for _ in range(60):
        y, x = between(rng, 3, 9), between(rng, -1, 9)
        if all(abs(y - a) >= 1 and abs(x - b) >= 1 and max(abs(y - a), abs(x - b)) >= 2 for a, b in cubes):
            cubes.append((y, x))
        if len(cubes) == n:
            return {(0, y, x) for y, x in cubes}
    raise Reject("lone cubes")


def gen_refer_lone_to_wall(rng):
    length, height = between(rng, 4, 8), between(rng, 1, 3)
    wall = {(z, 0, x) for z in range(height) for x in range(length)}
    lone = lone_scene(rng, between(rng, 2, 3))
    which = pick(rng, ["nearest", "farthest", "left", "right"])
    end = pick(rng, ["left", "right"])
    a = wall | lone
    src = ref_candidates(a, {"type": "lone", "which": which})
    need(len(src) == 1)
    dest = (height, 0, 0 if end == "left" else length - 1)
    who = {"nearest": "nearest me", "farthest": "farthest from me", "left": "on the left", "right": "on the right"}[which]
    text = pick(rng, [
        f"the lone cube {who}: put it on top of the wall, at the {end} end",
        f"take the single cube {who} and put it on the wall's {end} end, on top",
        f"move the loose cube {who} onto the top of the wall at the {end} end",
    ])
    return refer_record(rng, a, (a - {src[0]}) | {dest}, text, [{"type": "lone", "which": which}], pattern="lone_to_wall")


def gen_refer_lone_to_tower(rng):
    height = between(rng, 3, 6)
    tower = column(0, between(rng, 2, 6), height)
    lone = lone_scene(rng, between(rng, 2, 3))
    which = pick(rng, ["nearest", "farthest", "left", "right"])
    a = tower | lone
    need(all(max(abs(c[1] - t[1]), abs(c[2] - t[2])) >= 2 for c in lone for t in tower), "cube touches the tower")
    src = ref_candidates(a, {"type": "lone", "which": which})
    need(len(src) == 1)
    (ty, tx) = next((c[1], c[2]) for c in tower)
    who = {"nearest": "nearest me", "farthest": "farthest from me", "left": "on the left", "right": "on the right"}[which]
    text = pick(rng, [
        f"put the lone cube {who} on top of the tower",
        f"take the single cube {who} and stack it on the tower",
        f"the loose cube {who}: put it on top of the tower",
    ])
    return refer_record(rng, a, (a - {src[0]}) | {(height, ty, tx)}, text, [{"type": "lone", "which": which}], pattern="lone_to_tower")


def gen_refer_bump_in(rng):
    length, height = between(rng, 4, 8), between(rng, 1, 3)
    j = between(rng, 0, length - 1)
    out = pick(rng, ["front", "back"])
    wall = {(z, 0, x) for z in range(height) for x in range(length)}
    a = (wall - {(0, 0, j)}) | {(0, 1 if out == "front" else -1, j)}
    text = pick(rng, [
        f"the block sticking out of the {out} of the wall: push it back in line",
        "push the block that sticks out of the wall back into place",
        f"one block has been pushed out of the wall on the {out} side. push it back in line",
        "the cube sticking out of the wall goes back in line with the rest",
    ])
    return refer_record(rng, a, wall, text, [{"type": "bump", "which": "off the wall"}], pattern="bump_in")


# ---------------------------------------------------------------------------------------------
# Matching and counting


def assemble(rng, a, b, extra=0):
    """A and B fitted together, padded with a parked row to the same robot count (A's and B's rows are prefixes of
    the same slot order): (A cells, B cells, info)."""
    n = max(len(a), len(b)) + extra - min(len(a), len(b))
    rows = C.rows_for(n)
    need(rows <= C.MAX_ROWS, "too many parked")
    side = pick(rng, C.SIDES) if rows else None
    a, b = fit_sets([a, b], rng, reserve=(side, rows) if rows else None)
    return C.with_parking(rng, a, b, extra=extra, side=side)


def match_record(rng, a, b, text, extra=0, **info):
    a_cells, b_cells, park = assemble(rng, a, b, extra)
    return C.record("match", b_cells, [C.polish(text, rng)], start=a_cells, parking=park, **info)


POS3 = {2: ("left", "right"), 3: ("left", "middle", "right")}


def gen_match_towers(rng):
    n, gap = between(rng, 2, 3), between(rng, 1, 3)
    names = POS3[n]
    heights = [between(rng, 2, 6) for _ in range(n)]
    t, r = [int(i) for i in rng.permutation(n)[:2]]
    need(heights[t] < heights[r], "target not shorter")
    if n == 2 and chance(rng, 0.3):
        text = pick(rng, ["make the two towers the same height by building up the shorter one", "build up the shorter tower until both are the same height"])
    else:
        a, b = names[t], names[r]
        text = pick(rng, [
            f"make the {a} tower as tall as the {b} one",
            f"build up the {a} tower until it is as tall as the {b} one",
            f"add blocks to the {a} tower so it matches the {b} one in height",
            f"the {a} tower is shorter: make it the same height as the {b} one",
        ])
    A, _ = tower_scene(heights, gap)
    heights_b = list(heights)
    heights_b[t] = heights[r]
    B, _ = tower_scene(heights_b, gap)
    return match_record(rng, A, B, text, extra=between(rng, 0, 2), pattern="towers", added=heights[r] - heights[t])


def gen_match_walls(rng):
    height, gap = between(rng, 1, 3), between(rng, 2, 4)
    long_, short = between(rng, 5, 9), between(rng, 2, 4)
    need(long_ - short >= 2)
    left = chance(rng)
    which = pick(rng, ["front", "back"])  # front = nearer the viewer (high y)
    ys = {"back": 0, "front": gap}
    ya, yb = (ys["front"], ys["back"]) if which == "back" else (ys["back"], ys["front"])  # ya: the long wall's row
    xr = (lambda n: range(n)) if left else (lambda n: range(-n + 1, 1))
    long_wall = {(z, ya, x) for z in range(height) for x in xr(long_)}
    A = long_wall | {(z, yb, x) for z in range(height) for x in xr(short)}
    B = long_wall | {(z, yb, x) for z in range(height) for x in xr(long_)}
    text = pick(rng, [
        "add blocks until both walls are the same length",
        "make the short wall as long as the long one",
        "extend the shorter wall so the two match",
        "lengthen the shorter wall to match the other",
        f"the {'back' if yb == 0 else 'front'} wall is shorter: make it as long as the other one",
    ])
    return match_record(rng, A, B, text, extra=between(rng, 0, 2), pattern="walls", added=(long_ - short) * height)


COPY_KINDS = ("tower", "post", "stairs", "arch", "chair", "bench", "figure", "pyramid", "table")


def gen_match_copy(rng):
    kind = pick(rng, COPY_KINDS)
    part = C.rand_part(rng, kind, small=True)
    S = normalise(part.cells)
    need(4 <= len(S) <= 30, "size")
    arr = np.asarray(sorted(S))
    depth, width = int(arr[:, 1].max()) + 1, int(arr[:, 2].max()) + 1
    side = pick(rng, ["right", "left", "front", "behind"])
    off = {"right": (0, 0, width + 1), "left": (0, 0, -width - 1), "front": (0, depth + 1, 0), "behind": (0, -depth - 1, 0)}[side]
    copy = {(z + off[0], y + off[1], x + off[2]) for z, y, x in S}
    word = {"right": "on its right", "left": "on its left", "front": "in front of it", "behind": "behind it"}[side]
    noun = C.SHORT[kind]
    text = pick(rng, [
        f"make a second {noun} just like this one, {word}, with one space between them",
        f"build another {noun} exactly like this, {word}, leaving a gap of one",
        f"copy this {noun}: put an identical one {word}, one space apart",
    ])
    return match_record(rng, S, S | copy, text, extra=between(rng, 0, 1), pattern="copy", kind_of=kind, side=side)


def asymmetric_shape(rng):
    r = rng.random()
    if r < 0.3:
        L, H = between(rng, 3, 4), between(rng, 3, 5)
        cells = {(0, 0, x) for x in range(L)} | column(0, 0, H)
        return normalise(cells), "L shape"
    if r < 0.55:
        part = C.Part("figure", {"left": "up", "right": pick(rng, ["down", "out"]), "legs": between(rng, 2, 3)})
        return normalise(part.cells), "figure"
    if r < 0.8:
        part = C.Part("stairs", {"width": between(rng, 1, 2), "steps": between(rng, 3, 4)}, pick(rng, [1, 3]))
        return normalise(part.cells), "staircase"
    part = C.Part("post", {"h": between(rng, 3, 5), "arm": between(rng, 2, 3)}, pick(rng, [1, 3]))
    return normalise(part.cells), "post with an arm"


def gen_match_mirror(rng):
    S, noun = asymmetric_shape(rng)
    need(4 <= len(S) <= 40, "size")
    arr = np.asarray(sorted(S))
    width, depth = int(arr[:, 2].max()) + 1, int(arr[:, 1].max()) + 1
    flipped = {(z, y, width - 1 - x) for z, y, x in S}
    need(flipped != S, "symmetric shape")
    rows = C.rows_for(len(S))
    side = pick(rng, ["back", "front"])
    lo = rows + 1 if side == "back" else 1
    hi = SIZE - 2 - (rows if side == "front" else 0) - depth + 1
    need(lo <= hi and 6 - width >= 1, "does not fit")
    y0, x0 = between(rng, lo, hi), between(rng, 1, 7 - width)
    on_left = chance(rng)
    base = {(z, y + y0, x + x0) for z, y, x in S}
    mine = base if on_left else {(z, y, SIZE - 1 - x) for z, y, x in base}
    mirror = {(z, y, SIZE - 1 - x) for z, y, x in mine}
    other = "right" if on_left else "left"
    text = pick(rng, [
        "copy this shape to the other side of the floor, mirrored",
        f"make a mirror image of this on the {other} side",
        "put a mirrored copy of this on the other side",
        f"build the mirror image of this on the {other}",
    ])
    a_cells, b_cells, park = C.with_parking(rng, mine, mine | mirror, side=side)
    return C.record("match", b_cells, [C.polish(text, rng)], start=a_cells, parking=park, pattern="mirror", shape=noun, to=other)


# ---------------------------------------------------------------------------------------------
# Questions about the picture


def ask_record(rng, cells, text, reply, **info):
    (cells,) = fit_sets([cells], rng)
    return C.record("ask", cells, [C.polish(text, rng)], start=cells, reply=reply, **info)


def distractor(rng, x0, kinds=("wall", "platform", "house")):
    """A wall, a platform or a house away from the towers (none is a free-standing column)."""
    kind = pick(rng, list(kinds))
    part = C.rand_part(rng, kind, small=True)
    part = part.turned(-part.k)
    cells = normalise(part.cells)
    return {(z, y + 6, x + x0) for z, y, x in cells}, kind


def gen_ask_height(rng):
    h = between(rng, 2, 9)
    cells = column(0, 0, h)
    if chance(rng, 0.6):
        extra, _ = distractor(rng, 3)
        cells |= extra
    text = pick(rng, ["how tall is the tower?", "how many blocks high is the tower?", "what is the height of the tower?", "how high is the tower?"])
    reply = pick(rng, [f"It is {C.N(h)} blocks tall.", f"The tower is {C.N(h)} blocks high.", f"{C.N(h).capitalize()} blocks."])
    return ask_record(rng, cells, text, reply, pattern="height", answer=h)


def gen_ask_count(rng):
    n, gap = between(rng, 1, 6), between(rng, 1, 3)
    heights = [between(rng, 2, 7) for _ in range(n)]
    cells, xs = tower_scene(heights, gap)
    if chance(rng, 0.6):
        extra, _ = distractor(rng, xs[-1] + 3)
        cells |= extra
    text = pick(rng, ["how many towers are there?", "how many towers can you see?", "count the towers. how many are there?"])
    reply = pick(rng, [f"There {'is' if n == 1 else 'are'} {C.N(n)} tower{'' if n == 1 else 's'}.", f"{C.N(n).capitalize()}."])
    return ask_record(rng, cells, text, reply, pattern="count", answer=n)


def gen_ask_taller(rng):
    gap = between(rng, 1, 4)
    a, b = between(rng, 2, 8), between(rng, 2, 8)
    need(a != b)
    cells, _ = tower_scene([a, b], gap)
    side = "left" if a > b else "right"
    text = pick(rng, ["which tower is taller, left or right?", "which one is taller, the left tower or the right tower?", "is the left tower or the right tower taller?"])
    reply = pick(rng, [f"The {side} one is taller, by {C.N(abs(a - b))}.", f"The {side} tower, by {C.N(abs(a - b))} blocks.", f"The {side} one."])
    return ask_record(rng, cells, text, reply, pattern="taller", answer=side)


def gen_ask_wall(rng):
    length, height = between(rng, 3, 9), between(rng, 1, 4)
    cells = {(z, 0, x) for z in range(height) for x in range(length)}
    what = pick(rng, ["long", "high"])
    if chance(rng, 0.4):
        extra, _ = distractor(rng, length + 2, ("platform", "house"))
        cells |= extra
    n = length if what == "long" else height
    text = pick(rng, [f"how {what} is the wall?", f"how many blocks {what} is the wall?"])
    reply = f"The wall is {C.N(n)} blocks {what}."
    return ask_record(rng, cells, text, reply, pattern="wall", answer=n, asked=what)


GENERATORS = {
    "complete_pattern": (gen_complete_pattern, 4.0),
    "complete_part": (gen_complete_part, 3.5),
    "complete_scene": (gen_complete_scene, 1.5),
    "complete_shape": (gen_complete_shape, 1.5),
    "refer_top_to_floor": (gen_refer_top_to_floor, 0.9),
    "refer_top_move": (gen_refer_top_move, 1.1),
    "refer_lone_to_wall": (gen_refer_lone_to_wall, 0.8),
    "refer_lone_to_tower": (gen_refer_lone_to_tower, 0.8),
    "refer_bump_in": (gen_refer_bump_in, 0.7),
    "match_towers": (gen_match_towers, 1.2),
    "match_walls": (gen_match_walls, 1.0),
    "match_copy": (gen_match_copy, 1.0),
    "match_mirror": (gen_match_mirror, 1.0),
    "ask_height": (gen_ask_height, 0.6),
    "ask_count": (gen_ask_count, 0.65),
    "ask_taller": (gen_ask_taller, 0.55),
    "ask_wall": (gen_ask_wall, 0.55),
}


# ---------------------------------------------------------------------------------------------
# Checks, sampling, measuring


def problem(goal):
    """None when a goal is acceptable before the rest test, else the reason."""
    for cells in (goal["cells"], goal["start_cells"]):
        why = C.geometry_problem(cells, 1)
        if why:
            return why
    if len(goal["start_cells"]) != len(goal["cells"]):
        return "robot count differs"
    same = goal["start_cells"] == goal["cells"]
    if (goal["kind"] == "ask") != same:
        return "no change" if not same else "question changes the picture"
    for text in goal["turns"] + [goal["reply"]]:
        if text and not C.text_ok(text):
            return "words"
    return None


def draw(name, rng):
    """(goal, None) or (None, reason) for one draw of a generator."""
    try:
        goal = GENERATORS[name][0](rng)
    except Reject as err:
        return None, f"reject: {err}"
    why = problem(goal)
    if why:
        return None, why
    goal["generator"] = name
    states = [goal["cells"], goal["start_cells"]]
    ov = [C.overhang_span(s) for s in states]
    goal["params"]["overhang"] = max(o for o, _ in ov)
    goal["params"]["span"] = max(s for _, s in ov)
    return goal, None


def rest_ok(goals):
    states = [g["cells"] for g in goals] + [g["start_cells"] for g in goals if g["start_cells"] != g["cells"]]
    flags = C.at_rest(states) if states else []
    b_flags, a_flags = flags[: len(goals)], iter(flags[len(goals) :])
    return [bool(b and (next(a_flags) if g["start_cells"] != g["cells"] else True)) for g, b in zip(goals, b_flags)]


def sample(n, seed=0, names=None, tries=40):
    """n goals drawn by generator weight (all checks passed, at rest included)."""
    rng = np.random.default_rng(seed)
    names = list(names or GENERATORS)
    weights = np.array([GENERATORS[k][1] for k in names], dtype=float)
    weights /= weights.sum()
    out = []
    while len(out) < n:
        pending = []
        for _ in range(min(128, n - len(out))):
            name = names[int(rng.choice(len(names), p=weights))]
            for _ in range(tries):
                goal, _ = draw(name, rng)
                if goal is not None:
                    pending.append(goal)
                    break
        out += [g for g, ok in zip(pending, rest_ok(pending)) if ok]
    return out[:n]


def key_of(goal):
    return hashlib.sha1(json.dumps([goal["cells"], goal["start_cells"]]).encode()).hexdigest()


def measure(draws=1000, seed=1, names=None):
    out = {}
    for name in names or GENERATORS:
        rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
        goals, reasons = [], {}
        t0 = time.time()
        for _ in range(draws):
            goal, why = draw(name, rng)
            if goal is None:
                key = why.split(":")[1].strip() if why.startswith("reject") else why
                reasons[key] = reasons.get(key, 0) + 1
            else:
                goals.append(goal)
        flags = rest_ok(goals) if goals else []
        kept = [g for g, ok in zip(goals, flags) if ok]
        if len(kept) < len(goals):
            reasons["not at rest"] = len(goals) - len(kept)
        counts = [len(g["cells"]) for g in kept]
        row = {
            "draws": draws,
            "accepted": len(kept),
            "acceptance": round(len(kept) / draws, 3),
            "distinct": len({key_of(g) for g in kept}),
            "distinct_texts": len({" | ".join(g["turns"]) for g in kept}),
            "robots": [int(min(counts)), round(float(np.mean(counts)), 1), int(max(counts))] if counts else None,
            "changed_mean": round(float(np.mean([len(g["params"]["changed"]["fill"]) for g in kept])), 2) if kept else None,
            "rejects": dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:5]),
            "seconds": round(time.time() - t0, 1),
        }
        if name.startswith("complete"):
            row["k"] = {str(k): sum(g["params"]["k"] == k for g in kept) for k in range(1, 9)}
            row["rules"] = {r: sum(g["params"]["rule"] == r for g in kept) for r in sorted(FAMILY_WEIGHTS)}
        out[name] = row
    return out


# ---------------------------------------------------------------------------------------------
# Reversal check


def reverse_check(per=3, seed=0, names=None, max_steps=None):
    import torch

    from beebot.vla.voxel import reverse

    torch.set_num_threads(1)
    out = {}
    for name in names or GENERATORS:
        goals = [g for g in sample(per * 4, seed=seed, names=[name]) if g["start_cells"] != g["cells"]][:per]
        if not goals:
            out[name] = {"n": 0, "note": "question: A = B, no walk"}
            continue
        rng = np.random.default_rng(seed)
        t0 = time.time()
        episodes, _ = reverse.take_apart([C.to_reverse_goal(g) for g in goals], rng, max_steps=max_steps)
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
        }
        print(json.dumps({name: out[name]}), flush=True)
    return out


# ---------------------------------------------------------------------------------------------
# Storage and pictures


def save(goals, name=SOURCE, root=G.LIBRARY):
    """<name>.npz / .jsonl (B) and <name>_start.npz (A), the goal_gen_complex layout."""
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
            "has_start": True,
            "turns": g["turns"],
            "text": g["turns"][-1],
            "reply": g["reply"],
            "params": g["params"],
        })
    G.save(name, lines, [np.asarray(g["cells"], dtype=np.uint8) for g in goals], root=root)
    starts = [np.asarray(g["start_cells"], dtype=np.uint8) for g in goals]
    offsets = np.zeros(len(starts) + 1, dtype=np.int64)
    for i, s in enumerate(starts):
        offsets[i + 1] = offsets[i] + len(s)
    np.savez_compressed(Path(root) / f"{name}_start.npz", cells=np.concatenate(starts), starts=offsets)
    return lines


load = C.load


def tile(cells, other, parked, mark, title):
    """One boxed picture: an isometric view over the front and top views. Cells that differ from `other` in
    colour (`mark`), parked robots light grey."""
    from PIL import Image, ImageDraw

    colour = {}
    for c in cells:
        colour[c] = C.PARK if c in parked else C.GREY
        if c not in other:
            colour[c] = mark
    iso, ortho = C._iso(colour), C._front_top(colour)
    width = max(iso.width, ortho.width) + 12
    image = Image.new("RGB", (width, iso.height + ortho.height + 28), (246, 246, 243))
    image.paste(iso, (6, 22))
    image.paste(ortho, (6, 22 + iso.height))
    draw = ImageDraw.Draw(image)
    draw.text((8, 4), title, fill=(20, 20, 20))
    draw.rectangle([0, 0, image.width - 1, image.height - 1], outline=(60, 60, 60), width=2)
    return image


def render_goal(goal, path):
    """A and B side by side, each boxed, the changed cells coloured (cells that move blue in A, where they go
    orange in B), the caption (turns, reply, facts) under each picture."""
    from PIL import Image, ImageDraw

    a, b = set(map(tuple, goal["start_cells"])), set(map(tuple, goal["cells"]))
    park = goal["params"]["parked"]
    parked = set(C.park_slots(park["side"], C.MAX_ROWS)[: max(park["a"], park["b"])]) if park.get("side") else set()
    tiles = [tile(a, b, parked, C.EMPTY, "A (start)"), tile(b, a, parked, C.FILL, "B (end)")]
    p = goal["params"]
    facts = f"[{goal['generator']}] robots {len(b)}" + (f", k = {p['k']}, rule: {p['rule']} ({', '.join(p['rules'])})" if "k" in p else "")
    left = textwrap.wrap(f"user: {goal['turns'][-1]}", 62) + [facts[i : i + 62] for i in range(0, len(facts), 62)]
    right = textwrap.wrap(f"reply: {goal['reply']}", 62) if goal["reply"] else ["(no reply: the robots act)"]
    lines = max(len(left), len(right))
    width = max(t.width for t in tiles)
    image = Image.new("RGB", (2 * width + 12, max(t.height for t in tiles) + 14 * lines + 12), (246, 246, 243))
    draw = ImageDraw.Draw(image)
    for i, (t, caption) in enumerate(zip(tiles, (left, right))):
        x = i * (width + 12)
        image.paste(t, (x, 0))
        for j, line in enumerate(caption):
            draw.text((x + 6, t.height + 4 + 14 * j), line, fill=(20, 20, 20))
    image.save(path)
    return path


def spread(goals, k, seed=0):
    """k goals: completions across k = 1 .. 8 first, then the other kinds round-robin by generator."""
    rng = np.random.default_rng(seed)
    out, used = [], set()
    for kk in range(1, 9):
        pool = [i for i, g in enumerate(goals) if g["params"].get("k") == kk]
        for i in rng.permutation(pool)[: 1 if kk > 3 else 2]:
            out.append(int(i))
            used.add(int(i))
    by = {}
    for i, g in enumerate(goals):
        if i not in used and g["kind"] != "complete":
            by.setdefault(g["generator"], []).append(i)
    pools = [list(rng.permutation(v)) for _, v in sorted(by.items())]
    while len(out) < k and any(pools):
        for pool in pools:
            if pool and len(out) < k:
                out.append(int(pool.pop()))
    return out[:k]


def _worker(args):
    n, seed = args
    import torch

    torch.set_num_threads(1)
    return sample(n, seed)


def write(n, seed, procs):
    """n distinct goals, drawn in `procs` processes."""
    from multiprocessing import get_context

    goals, seen, round_ = [], set(), 0
    while len(goals) < n:
        want = n - len(goals)
        share = -(-int(want * 1.03) // procs) + 1
        jobs = [(share, seed * 1000 + round_ * 10 + i) for i in range(procs)]
        with get_context("spawn").Pool(procs) as pool:
            batches = pool.map(_worker, jobs)
        for g in (g for b in batches for g in b):
            key = key_of(g)
            if key not in seen and len(goals) < n:
                seen.add(key)
                goals.append(g)
        round_ += 1
    return goals


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-name", default=SOURCE)
    parser.add_argument("--render-dir", default=None)
    parser.add_argument("--renders", type=int, default=30)
    parser.add_argument("--procs", type=int, default=4)
    parser.add_argument("--measure", type=int, default=0, help="draws per generator; prints the table and exits")
    parser.add_argument("--reverse", type=int, default=0, help="samples per generator through reverse.take_apart")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    if args.measure:
        for name, row in measure(args.measure, args.seed, args.only).items():
            print(json.dumps({name: row}), flush=True)
        return
    if args.reverse:
        reverse_check(args.reverse, args.seed, args.only, args.max_steps)
        return
    t0 = time.time()
    goals = write(args.n, args.seed, min(args.procs, 4))
    save(goals, args.out_name)
    kinds = {}
    for g in goals:
        kinds[g["kind"]] = kinds.get(g["kind"], 0) + 1
    summary = {
        "written": len(goals),
        "kinds": kinds,
        "generators": {k: sum(g["generator"] == k for g in goals) for k in GENERATORS},
        "k": {str(k): sum(g["params"].get("k") == k for g in goals) for k in range(1, 9)},
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
            name = f"gen-ground-{j:02d}"
            render_goal(g, out / f"{name}.png")
            cards.append({"name": name, "generator": g["generator"], "kind": g["kind"], "turns": g["turns"], "reply": g["reply"], "cells": g["cells"], "start_cells": g["start_cells"], "params": g["params"]})
        (out / "cards-gen-ground.json").write_text(json.dumps(cards))


if __name__ == "__main__":
    main()
