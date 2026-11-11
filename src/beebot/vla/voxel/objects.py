"""Objects with properties, passive blocks, floor zones and reasoning over them in the voxel world
(README "Objects").

The world (world.sample_object_scenes) gives each scene up to MAX_OBJECTS objects: cubes, slabs,
rods and L shapes of 1 to 4 voxels a side, plain or patterned in two colours, light or heavy
(weight is never observed), food (cheeses) or not (passive blocks), and floor zones (patches and
stripes). Four families use them; they are drawn by sample_objects, never by suite.sample_tasks
or suite.sample_coverage, so those draws are unchanged:

  things     a job on a cheese named by its properties ("the large striped red cheese"): eat it,
             push it to a floor region or into the trench, surround it, catch it (it flees), push
             "the heavy cheese" (the expert finds it by pushing each candidate with two robots)
  blocks     passive blocks: push one to a region or into the trench, push the heavy one, line the
             blocks up in one row
  zones      go to a zone, line up along a stripe, push an object onto a zone
  reasoning  superlatives and comparisons over size ("the biggest cheese", "the smaller of the two
             blue blocks"), distance ("the cheese nearest the trench", "the block furthest from the
             green zone"), relations ("the cheese next to the yellow block"), conditionals ("eat the
             red cheese if there is one, otherwise build a tower 5 robots tall") and counting ("build
             a tower as tall as the number of cheeses")

A task's spec has "objects": True and its kind (JOB_KINDS); a job on an object names it by a
reference (ref: {"by": attrs / super / compare / near / next_to, "noun", "attrs", ...}) that
resolve() turns into a slot, and the spec keeps the slot as "target". Checkers read the object
fields from the scene's meta ("props"), so score() has what it needs from suite.run's static.
"""

import itertools
import math

import numpy as np
import torch

from beebot.vla.voxel import suite as S
from beebot.vla.voxel.controller import direction_code
from beebot.vla.voxel.facts import tallest_run
from beebot.vla.voxel.world import BOX, CHAIN_DEPTH, COLOUR_NAMES, EAT_MIN_SHARE, EAT_SHARE, FRICTION, HEAVY_FRICTION, MAX_OBJECTS, PATTERNS, RIM_OFFSETS, cheese_id_map, concat, sample_object_scenes, shell_mask, with_objects

SIZE = S.SIZE
OBJECT_FAMILIES = ("things", "blocks", "zones", "reasoning")
SIZE_WORDS = {1: "tiny", 2: "small", 3: "large", 4: "huge"}  # by the longest side
SHAPE_WORDS = {"cube": "cube-shaped", "slab": "flat", "rod": "long", "L": "L-shaped"}
PATTERN_WORDS = {"striped": "striped", "two-tone": "two-tone"}
PLURAL = {"cheese": "cheeses", "block": "blocks", "object": "objects"}
ADJECTIVES = ("size", "weight", "pattern", "colour", "shape")  # their order in a phrase (wordings.ADJECTIVE_GROUPS)
SEEN = ("size", "pattern", "colour", "shape")  # what a description may use: weight is not seen
JOB_KINDS = ("eat_thing", "push_thing", "surround_thing", "catch_thing", "line_blocks", "goto_zone", "line_zone", "conditional", "count")
NEXT = 3  # floor-plan (Chebyshev) distance between footprints within which an object is next to another
MARGIN = 2.0  # a nearest or furthest object is this much nearer or further than the next one
BIGGER = 1.5  # a biggest, smallest, smaller or larger object differs from the next by this factor in voxels
PRESS_TEAM = 8  # pushers a shove aims for (face plus robots in line behind it), so a heavy object (HEAVY_FRICTION) moves despite failures
PROBERS = 2  # robots that test whether an object is light
PROBE_HOLD = 5  # steps the probers press without the object moving before it counts as heavy
LOW_GROUND = ("flat", "hills")  # layouts of eat and surround jobs: the greedy controller does not climb a plateau's cliff (suite.JobSetup.detour does that for coverage jobs only)
PATH = 3  # voxels ahead of a pushed object (from its leading voxels on) whose robots step aside
TRAIN_COLOURS = [COLOUR_NAMES.index(c) for c in S.TRAIN_COLOURS]
HELD_OUT = {  # held-out compositions: drawn only by their split, with no plan
    "ref_job": (("compare", "push_thing"), ("next_to", "surround_thing")),
    "condition": ("eat_goto",),
    "count": ("groups",),
}
HELD_FAMILY = {"ref_job": "reasoning", "condition": "reasoning", "count": "reasoning"}


# ---------------------------------------------------------------------------------------------
# Objects as the checkers and descriptions see them


def shape_of(form):
    """The shape name (world.OBJECT_SHAPES) of a (BOX, BOX, BOX) bool form."""
    form = np.asarray(form, bool)
    depth, rows, cols = (int(np.nonzero(form.any(axis=tuple(a for a in range(3) if a != axis)))[0].max()) + 1 for axis in range(3))
    n = int(form.sum())
    if depth == rows == cols and n == depth**3:
        return "cube"
    if depth == 1 and rows == cols and n == rows * cols:
        return "slab"
    if depth == 1 and (rows == 1 or cols == 1):
        return "rod"
    return "L"


def thing_list(pos, alive, colour, props):
    """[dict] of the alive objects of one scene, by slot: slot, noun (cheese or block), colour,
    colour2 (None), pattern, shape, size (longest side), voxels, heavy, pos (z, y, x), extent
    (depth, rows, columns), cells ((n, 2) floor cells y, x), centre (y, x of the footprint's
    middle), cubes ((n, 3) voxels)."""
    out = []
    for k in range(len(alive)):
        if not alive[k] or colour[k] < 0:
            continue
        form = np.asarray(props["form"][k], bool)
        voxels = np.argwhere(form) + np.asarray(pos[k], int)
        cells = np.unique(voxels[:, 1:], axis=0)
        extent = tuple(int(v) for v in voxels.max(axis=0) - voxels.min(axis=0) + 1)
        second = int(props["colour2"][k])
        out.append({
            "slot": k,
            "noun": "cheese" if props["food"][k] else "block",
            "colour": COLOUR_NAMES[int(colour[k])],
            "colour2": COLOUR_NAMES[second] if second >= 0 else None,
            "pattern": PATTERNS[int(props["pattern"][k])],
            "shape": shape_of(form),
            "size": max(extent),
            "voxels": int(form.sum()),
            "heavy": float(props["friction"][k]) > FRICTION,
            "pos": tuple(int(v) for v in pos[k]),
            "extent": extent,
            "cells": cells,
            "centre": cells.mean(axis=0) + 0.5,
            "cubes": voxels,
        })
    return out


def props_of(world, i):
    """The object fields of scene i as numpy (the meta's "props": what the checkers read)."""
    return {"form": world.form[i].numpy(), "colour2": world.colour2[i].numpy(), "pattern": world.pattern[i].numpy(), "friction": world.friction[i].numpy(), "food": world.food[i].numpy(), "zones": world.zones[i].numpy()}


def things_of(world, i):
    return thing_list(world.cheese_pos[i].numpy(), world.cheese_alive[i].numpy(), world.cheese_color[i].numpy(), props_of(world, i))


def things_at(static, snap):
    return thing_list(snap["cheese_pos"], snap["cheese_alive"], np.asarray(static["cheese_color"]), static["meta"]["props"])


def by_slot(things):
    return {t["slot"]: t for t in things}


def zone_list(zones):
    """[{"colour", "kind" (zone or stripe), "box": (y0, y1, x0, x1)}] of a (H, W) zone map."""
    out = []
    for c in sorted(set(np.unique(zones).tolist()) - {-1}):
        ys, xs = np.nonzero(zones == c)
        box = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
        out.append({"colour": COLOUR_NAMES[c], "kind": "stripe" if min(box[1] - box[0], box[3] - box[2]) == 1 else "zone", "box": box})
    return out


def zone_of(zones, colour):
    return next((z for z in zone_list(zones) if z["colour"] == colour), None)


def word(thing, group):
    """The canonical word of one of a thing's properties (None: it has none, a plain pattern)."""
    if group == "size":
        return SIZE_WORDS[thing["size"]]
    if group == "weight":
        return "heavy" if thing["heavy"] else "light"
    if group == "pattern":
        return PATTERN_WORDS.get(thing["pattern"])
    if group == "colour":
        return thing["colour"]
    return SHAPE_WORDS[thing["shape"]]


def matches(thing, noun, attrs):
    """Whether a thing is a `noun` with these properties; a colour matches either of its colours."""
    if noun != "object" and thing["noun"] != noun:
        return False
    for group, value in attrs.items():
        if group == "colour":
            if value not in (thing["colour"], thing["colour2"]):
                return False
        elif word(thing, group) != value:
            return False
    return True


def cell_gap(a, b):
    """Floor-plan Chebyshev distance between two sets of cells (n, 2)."""
    return int(np.abs(a[:, None, :] - b[None, :, :]).max(axis=2).min())


def box_gap(cells, box):
    y0, y1, x0, x1 = box
    dy = np.maximum(np.maximum(y0 - cells[:, 0], cells[:, 0] - (y1 - 1)), 0)
    dx = np.maximum(np.maximum(x0 - cells[:, 1], cells[:, 1] - (x1 - 1)), 0)
    return int(np.maximum(dy, dx).min())


def anchor_distance(thing, anchor, meta):
    """Floor distance from a thing's footprint to the trench or a zone."""
    if anchor.get("trench"):
        axis, start, width, _ = S.trench(meta)
        along = thing["cells"][:, axis - 1]
        return int(np.maximum(np.maximum(start - along, along - (start + width - 1)), 0).min())
    return box_gap(thing["cells"], zone_of(meta["props"]["zones"], anchor["zone"])["box"])


# ---------------------------------------------------------------------------------------------
# References


def ref_words(ref):
    return [ref["attrs"][g] for g in ADJECTIVES if g in ref["attrs"]]


def anchor_phrase(anchor):
    return "the trench" if anchor.get("trench") else f"the {anchor['zone']} {'stripe' if anchor.get('kind') == 'stripe' else 'zone'}"


def phrase(ref):
    """The canonical words of a reference (wordings.thing_pattern reads them)."""
    words = ref_words(ref)
    by = ref["by"]
    if by == "compare":
        return f"the {ref['which']} of the two " + " ".join(words + [PLURAL[ref['noun']]])
    core = " ".join(words + [ref["noun"]])
    if by == "super":
        return f"the {ref['which']} {core}"
    if by == "near":
        return f"the {core} {ref['which']} {anchor_phrase(ref['anchor'])}"
    if by == "next_to":
        return f"the {core} next to {phrase(ref['anchor'])}"
    return f"the {core}"


def full_phrase(thing):
    """Every visible property of a thing: "the large striped red cube-shaped cheese"."""
    return phrase({"by": "attrs", "noun": thing["noun"], "attrs": {g: word(thing, g) for g in SEEN if word(thing, g) is not None}})


def resolve(ref, things, meta):
    """The slot a reference picks out among the things (thing_list), or None when it picks out
    none or more than one."""
    pool = [t for t in things if matches(t, ref["noun"], ref["attrs"])]
    by = ref["by"]
    if by == "attrs":
        return pool[0]["slot"] if len(pool) == 1 else None
    if by == "next_to":
        anchor = resolve(ref["anchor"], things, meta)
        if anchor is None:
            return None
        other = by_slot(things)[anchor]
        pool = [t for t in pool if t["slot"] != anchor]
        if len(pool) < 2:
            return None
        gaps = sorted((cell_gap(t["cells"], other["cells"]), t["slot"]) for t in pool)
        return gaps[0][1] if gaps[0][0] <= NEXT and gaps[1][0] >= gaps[0][0] + MARGIN else None
    if len(pool) < 2 or (by == "compare" and len(pool) != 2):
        return None
    if by in ("super", "compare"):
        big = ref["which"] in ("biggest", "larger")
        ranked = sorted(pool, key=lambda t: t["voxels"], reverse=big)
        first, second = ranked[0]["voxels"], ranked[1]["voxels"]
        return ranked[0]["slot"] if (first >= BIGGER * second if big else second >= BIGGER * first) else None
    if by == "near":
        far = ref["which"] == "furthest from"
        ranked = sorted(((anchor_distance(t, ref["anchor"], meta), t["slot"]) for t in pool), reverse=far)
        return ranked[0][1] if abs(ranked[0][0] - ranked[1][0]) >= MARGIN else None
    raise ValueError(by)


def describe(rng, things, slot, groups=SEEN):
    """Properties (fewest, at random among the fewest) that pick out the thing in this slot among
    its kind: {} when it is the only one, None when none do."""
    thing = by_slot(things)[slot]
    names = [g for g in groups if word(thing, g) is not None]
    for n in range(len(names) + 1):
        fits = []
        for combo in itertools.combinations(names, n):
            attrs = {g: word(thing, g) for g in combo}
            if resolve({"by": "attrs", "noun": thing["noun"], "attrs": attrs}, things, None) == slot:
                fits.append(attrs)
        if fits:
            return fits[int(rng.integers(len(fits)))]
    return None


# ---------------------------------------------------------------------------------------------
# Geometry of placing and pushing


def footprint(thing, pos=None):
    """(n, 2) floor cells of a thing, moved to pos (z, y, x) when given."""
    if pos is None:
        return thing["cells"]
    return thing["cells"] - np.asarray(thing["pos"][1:]) + np.asarray(pos[1:])


def goal_centre(to, meta, thing):
    """The floor point (y, x) a push heads for."""
    if "region" in to:
        y0, y1, x0, x1 = S.region_box(to["region"])
        return np.array([(y0 + y1) / 2, (x0 + x1) / 2])
    if "zone" in to:
        y0, y1, x0, x1 = zone_of(meta["props"]["zones"], to["zone"])["box"]
        return np.array([(y0 + y1) / 2, (x0 + x1) / 2])
    if "trench" in to:
        axis, start, width, _ = S.trench(meta)
        out = thing["centre"].copy()
        out[axis - 1] = start + width / 2
        return out
    axis, coord = to["line"]
    out = thing["centre"].copy()
    out[axis - 1] = coord + 0.5
    return out


def in_place(thing, to, meta):
    """Whether a thing (thing_list) stands where a push asks."""
    cells = thing["cells"]
    if "region" in to:
        y0, y1, x0, x1 = S.region_box(to["region"])
        cy, cx = thing["centre"]
        return bool(y0 <= cy < y1 and x0 <= cx < x1)
    if "zone" in to:
        zones = meta["props"]["zones"]
        colour = COLOUR_NAMES.index(to["zone"])
        over = int((zones[cells[:, 0], cells[:, 1]] == colour).sum())
        return over >= math.ceil(min(len(cells), int((zones == colour).sum())) / 2)
    if "trench" in to:
        axis, start, width, top = S.trench(meta)
        along = cells[:, axis - 1]
        return bool(along.min() >= start and along.max() < start + width and thing["pos"][0] < top)
    axis, coord = to["line"]
    return bool((cells[:, axis - 1] == coord).any())


def push_value(start_thing, end_thing, to, meta):
    """1 when placed, else 0.8 of the share of the way covered toward the goal."""
    if end_thing is None:
        return 0.0
    if in_place(end_thing, to, meta):
        return 1.0
    goal = goal_centre(to, meta, start_thing)
    d0 = np.linalg.norm(start_thing["centre"] - goal)
    d1 = np.linalg.norm(end_thing["centre"] - goal)
    return float(0.8 * np.clip((d0 - d1) / max(d0, 1e-6), 0, 1))


def shell_cells(thing, walls):
    """(open shell voxels (n, 3)): the voxels touching the thing that are in the grid and not walls."""
    form = torch.zeros(1, 1, BOX, BOX, BOX, dtype=torch.bool)
    rel = thing["cubes"] - np.asarray(thing["pos"])
    form[0, 0, rel[:, 0], rel[:, 1], rel[:, 2]] = True
    mask = shell_mask(form)[0, 0].numpy()
    coords = np.asarray(thing["pos"]) + RIM_OFFSETS.numpy()[mask]
    inside = ((coords >= 0) & (coords < np.array(walls.shape))).all(axis=1)
    coords = coords[inside]
    return coords[~walls[coords[:, 0], coords[:, 1], coords[:, 2]]], int(mask.sum())


def eat_need(thing, walls, runner=False):
    """The fewest robots that can eat a thing where it stands (suite.eat_need for any form)."""
    open_, full = shell_cells(thing, walls)
    least = int(np.ceil(EAT_MIN_SHARE * full))
    if runner:
        return least
    return max(int(np.ceil(EAT_SHARE * len(open_))), least) if len(open_) >= least else float("inf")


def press_need(thing):
    return HEAVY_FRICTION if thing["heavy"] else FRICTION


def team_need(thing):
    """Robots a push takes: FRICTION for a light object, a full team (PRESS_TEAM) for a heavy one,
    whose HEAVY_FRICTION pressers must all press in the same step despite failures."""
    return PRESS_TEAM if thing["heavy"] else FRICTION


def faces(thing, solid, axis, sign):
    """(face voxels, leading voxels) of a push along floor axis 1 or 2 by sign: the voxels a robot
    presses from (behind the thing) and the voxels the thing moves into, both (n, 3), in the grid;
    solid (D, H, W) bool marks walls and other objects (faces on them are dropped)."""
    cubes = thing["cubes"]
    own = {tuple(c) for c in cubes.tolist()}
    step = np.zeros(3, int)
    step[axis] = sign
    behind = [tuple(c) for c in (cubes - step).tolist() if tuple(c) not in own]
    ahead = [tuple(c) for c in (cubes + step).tolist() if tuple(c) not in own]
    shape = np.array(solid.shape)
    inside = lambda c: all(0 <= v < n for v, n in zip(c, shape))  # noqa: E731
    behind = [c for c in dict.fromkeys(behind) if inside(c) and not solid[c]]
    return behind, [c for c in dict.fromkeys(ahead)]


def lead_clear(thing, solid, axis, sign):
    """Whether the voxels a thing would move into are in the grid and clear of walls and objects."""
    _, ahead = faces(thing, solid, axis, sign)
    shape = solid.shape
    return all(all(0 <= v < n for v, n in zip(c, shape)) and not solid[c] for c in ahead)


def team_cells(thing, solid, axis, sign, cap=PRESS_TEAM):
    """The voxels a pushing team stands in, at most `cap`: the face voxels behind the thing and up
    to CHAIN_DEPTH more in line behind each, the low ones and the near ones first (rank: twice the
    height above the thing's base, since climbing costs pivots, plus depth), so a team fills from
    the ground and every robot has the one it pushes through in front of it."""
    face, _ = faces(thing, solid, axis, sign)
    base = thing["pos"][0]
    ranked = []
    for z, y, x in face:
        for depth in range(CHAIN_DEPTH + 1):
            cell = (z, y - sign * depth if axis == 1 else y, x - sign * depth if axis == 2 else x)
            if not (0 <= cell[1] < SIZE and 0 <= cell[2] < SIZE) or solid[cell]:
                break
            ranked.append((2 * (z - base) + depth, depth, cell))
    ranked.sort()
    return [cell for _, _, cell in ranked[:cap]]


def press_room(thing, solid, axis, sign):
    """How many robots can press a thing along floor axis 1 or 2 by sign: its face and the rows
    behind it (team_cells without the cap), as a heavy object counts them."""
    return len(team_cells(thing, solid, axis, sign, cap=10**6))


def probe_way(thing, solid, preferred):
    """(axis, sign, PROBERS face voxels on the ground layer) of a push that tests the thing's
    weight: the preferred (axis, sign) or another, with the way ahead clear; None if there is none."""
    ways = [preferred] + [(a, s) for a in (1, 2) for s in (1, -1) if (a, s) != tuple(preferred)]
    for axis, sign in ways:
        face, _ = faces(thing, solid, axis, sign)
        low = sorted(c for c in face if c[0] == thing["pos"][0])
        if len(low) >= PROBERS and lead_clear(thing, solid, axis, sign):
            return axis, sign, low[:PROBERS]
    return None


def solid_of(world, i, k):
    """Walls and the alive objects other than slot k of scene i, (D, H, W) numpy bool."""
    ids = cheese_id_map(world.index(i))[0].numpy()
    return world.walls[i].numpy() | ((ids >= 0) & (ids != k))


# ---------------------------------------------------------------------------------------------
# Checking


def eaten_slots(start, end):
    return {int(k) for k in np.nonzero(np.asarray(start["cheese_alive"]) & ~np.asarray(end["cheese_alive"]))[0]}


def allowed_eats(spec):
    """Slots a task may eat."""
    if spec["kind"] in ("eat_thing", "catch_thing"):
        return {spec["target"]}
    if spec["kind"] == "conditional":
        return allowed_eats(branch(spec))
    return set()


def branch(spec):
    """The job a conditional asks in its scene (its "then" when the condition holds, else its "else")."""
    return spec["then"] if spec["holds"] else spec["else"]


def job_value(spec, static, start, end):
    """How far the job of an object task was done, in [0, 1]."""
    meta = static["meta"]
    kind = spec["kind"]
    meals = eaten_slots(start, end)
    mass = end["mass"] > 0.5
    if kind in ("eat_thing", "catch_thing"):
        return float(spec["target"] in meals)
    if kind == "surround_thing":
        thing = by_slot(things_at(static, end)).get(spec["target"])
        if thing is None:
            return 0.0
        cells, _ = shell_cells(thing, static["walls"])
        share = mass[cells[:, 0], cells[:, 1], cells[:, 2]].sum() / max(len(cells), 1)
        return float(min(1.0, share / 0.5))
    if kind == "push_thing":
        return push_value(by_slot(things_at(static, start))[spec["target"]], by_slot(things_at(static, end)).get(spec["target"]), spec["to"], meta)
    if kind == "line_blocks":
        blocks = [t for t in things_at(static, end) if t["noun"] == "block"]
        axis = 1 if spec["axis"] == "x" else 2  # a row from left to right shares a y row
        aligned = max(sum(bool((t["cells"][:, axis - 1] == c).any()) for t in blocks) for c in range(SIZE))
        return (aligned - 1) / max(len(blocks) - 1, 1)  # one block is always on some row
    if kind == "goto_zone":
        y0, y1, x0, x1 = zone_of(meta["props"]["zones"], spec["zone"])["box"]
        cells = np.argwhere(mass)
        inside = (cells[:, 1] >= y0 - 1) & (cells[:, 1] < y1 + 1) & (cells[:, 2] >= x0 - 1) & (cells[:, 2] < x1 + 1)
        room = 2 * (y1 - y0 + 2) * (x1 - x0 + 2)
        return float(min(1.0, inside.sum() / max(min(len(cells), room), 1)))
    if kind == "line_zone":
        zones = meta["props"]["zones"]
        colour = COLOUR_NAMES.index(spec["zone"])
        ground = static["walls"].sum(axis=0)
        ys, xs = np.nonzero(zones == colour)
        on = sum(bool(mass[min(int(ground[y, x]), SIZE - 1), y, x]) for y, x in zip(ys, xs))
        return float(min(1.0, on / max(min(int(mass.sum()), len(ys)), 1)))
    if kind == "conditional":
        return job_value(branch(spec), static, start, end)
    if kind == "tower":
        return min(1.0, tallest_run(end["mass"]) / spec["height"])
    if kind == "goto":
        y0, y1, x0, x1 = S.region_box(spec["region"])
        cells = np.argwhere(mass)
        inside = (cells[:, 1] >= y0 - 1) & (cells[:, 1] < y1 + 1) & (cells[:, 2] >= x0 - 1) & (cells[:, 2] < x1 + 1)
        return float(min(1.0, inside.sum() / max(len(cells), 1)))
    if kind == "count":
        if spec["job"] == "tower":
            return min(1.0, tallest_run(end["mass"]) / spec["count"])
        m = spec["count"]
        n = max(int(mass.sum()), 1)
        shares = [len(g) / n for g in S.components(mass)] + [0.0] * m
        return float(min(1.0, sum(shares[:m]) / 0.8) * min(1.0, shares[m - 1] / (0.5 / m)))
    raise ValueError(kind)


def score(task, static, start, end, eaten):
    """(score in [0, 1], unasked) of an object task: unasked when it ate any object it does not
    ask to eat (read off the snapshots; `eaten`, the colours, is not needed)."""
    spec = task.spec
    unasked = bool(eaten_slots(start, end) - allowed_eats(spec))
    return job_value(spec, static, start, end), unasked


def bound(task, world, i, n=None):
    """An upper bound on the score a swarm of n robots (default: scene i's) can reach: eating
    takes eat_need robots, surrounding the 0.6 of the open shell the expert fills, pushing a light
    object FRICTION robots and a heavy one a full team (team_need), a tower 4 robots a level (1 for
    a single column, tower_points scaled), m groups 2 m robots. Part of the draw (sample_objects
    keeps tasks that reach the pass mark)."""
    spec = task.spec
    n = int(world.mass[i].sum()) if n is None else n
    kind = spec["kind"]
    if kind == "conditional":
        spec = dict(branch(spec))
        kind = spec["kind"]
    things = by_slot(things_of(world, i))
    walls = world.walls[i].numpy()
    if kind in ("eat_thing", "catch_thing"):
        return float(n >= eat_need(things[spec["target"]], walls, runner=kind == "catch_thing"))
    if kind == "surround_thing":
        cells, _ = shell_cells(things[spec["target"]], walls)
        return float(min(1.0, n / max(0.6 * len(cells), 1)))
    if kind == "push_thing":
        return float(n >= team_need(things[spec["target"]]))
    if kind == "line_blocks":
        return float(n >= max(team_need(things[k]) for k in spec["moves"]))
    points = task.plan[0].get("points") if task.plan else None  # a tower's planned column (suite.level_robots)
    if kind == "tower":
        return min(1.0, n / (S.level_robots(points, spec["height"]) * spec["height"]))
    if kind == "count":
        return min(1.0, n / (S.level_robots(points, spec["count"]) * spec["count"])) if spec["job"] == "tower" else float(n >= 2 * spec["count"])
    if kind == "line_zone":
        return float(n >= 3)
    return 1.0


def task_kind(task):
    """job:reference (and >destination for a push): "eat_thing:attrs", "push_thing:super>trench",
    "push_thing:weight>region", "conditional:eat_tower@then", "count:tower"."""
    spec = task.spec
    kind = spec["kind"]
    if kind == "conditional":
        return f"conditional:{spec['form']}@{'then' if spec['holds'] else 'else'}"
    if kind == "count":
        return f"count:{spec['job']}"
    if "ref" not in spec:
        return kind
    by = "weight" if spec.get("find") else spec["ref"]["by"]
    out = f"{kind}:{by}"
    if kind == "push_thing":
        out += ">" + next(iter(k for k in ("region", "trench", "zone") if k in spec["to"]))
    return out


# ---------------------------------------------------------------------------------------------
# The expert


class ObjectExpert(S.Expert):
    """suite.Expert with three more segment kinds:
      shove  push an object to a goal ({"region"}, {"trench"}, {"zone"} or {"line": [axis, coord]})
             with a team behind it: the robots in the voxels behind its face and up to CHAIN_DEPTH
             rows behind those (PRESS_TEAM in all), every one stepping toward the object. The team
             is the same whatever the object weighs (the expert does not see weight either).
      find   find the heavy (or light) one of several candidates: PROBERS robots press each in turn
             (nearest first); one that moves is light, one that has not moved after PROBE_HOLD steps
             of pressing is heavy. Then the shove of the one found.
      fixed  target voxels with a cap on the robots that take part ("points", "cap").
    Everything else is suite.Expert's."""

    def __init__(self, tasks, moves=None, seed=None, rule=None):
        super().__init__(tasks, moves=moves, seed=seed, rule=rule)
        self.probes = {}  # scene -> probe state of a find segment
        self.teams = [None] * len(tasks)  # per scene: (direction, voxels whose robots step that way) of a push this step

    def thing(self, world, i, k):
        return by_slot(things_of(world, i)).get(k)

    def solid(self, world, i, k):
        return solid_of(world, i, k)

    def heading(self, world, i, thing, to):
        """(axis 1 or 2, sign) of the push toward the goal: the floor axis with the larger offset
        whose leading voxels are clear, else the other."""
        meta = self.tasks[i].spec["meta_ref"]
        if "trench" in to:
            axis, start, width, _ = S.trench(meta)
            centre = thing["centre"][axis - 1]
            return axis, 1 if centre < start + width / 2 else -1
        if "line" in to:
            axis, coord = to["line"]
            cells = thing["cells"][:, axis - 1]
            return axis, 1 if coord > cells.max() else -1
        offset = goal_centre(to, meta, thing) - thing["centre"]
        solid = self.solid(world, i, thing["slot"])
        order = [int(a) for a in np.argsort(-np.abs(offset))]
        for a in order:
            if abs(offset[a]) < 0.5:
                continue
            sign = 1 if offset[a] > 0 else -1
            if lead_clear(thing, solid, a + 1, sign) and team_cells(thing, solid, a + 1, sign):
                return a + 1, sign
        first = order[0]  # the way is blocked (another object, terrain): sidestep along the other floor axis, away from what blocks it
        _, ahead = faces(thing, solid, first + 1, 1 if offset[first] > 0 else -1)
        stops = [c for c in ahead if not all(0 <= v < n for v, n in zip(c, solid.shape)) or solid[c]]
        side = 1 - first
        away = thing["centre"][side] - (np.mean([c[side + 1] for c in stops]) + 0.5) if stops else offset[side]
        for sign in ((1, -1) if away >= 0 else (-1, 1)):
            if lead_clear(thing, solid, side + 1, sign) and team_cells(thing, solid, side + 1, sign):
                return side + 1, sign
        a = order[0]
        return a + 1, 1 if offset[a] > 0 else -1

    def shove_targets(self, world, i, k, to):
        thing = self.thing(world, i, k)
        meta = self.tasks[i].spec["meta_ref"]
        if thing is None or in_place(thing, to, meta):
            return [], None, None
        axis, sign = self.heading(world, i, thing, to)
        pts = team_cells(thing, self.solid(world, i, k), axis, sign)
        direction = [0, 0, 0]
        direction[axis] = sign
        self.teams[i] = (tuple(direction), pts, k)
        return pts, (k, tuple(direction)), len(pts)

    def probe_state(self, world, i, seg):
        state = self.probes.get(i)
        if state is None:
            state = self.probes[i] = {"queue": list(seg["candidates"]), "current": None, "start": None, "presses": 0, "found": None, "tried": []}
        return state

    def probe_targets(self, world, i, seg):
        """The targets of a find segment this step: the probers' face voxels, or once found the
        shove of the one found."""
        state = self.probe_state(world, i, seg)
        while state["found"] is None:
            if state["current"] is None:
                if not state["queue"]:
                    state["found"] = state["tried"][-1] if state["tried"] else seg["candidates"][0]
                    break
                state["current"] = state["queue"].pop(0)
                state["start"] = tuple(int(v) for v in world.cheese_pos[i, state["current"]])
                state["presses"] = 0
                state["tried"].append(state["current"])
            k = state["current"]
            has_moved = tuple(int(v) for v in world.cheese_pos[i, k]) != state["start"]
            light = has_moved
            heavy = not has_moved and state["presses"] >= PROBE_HOLD
            if light or heavy:
                if (heavy and seg["want"] == "heavy") or (light and seg["want"] == "light"):
                    state["found"] = k
                state["current"] = None
                continue
            thing = self.thing(world, i, k)
            way = probe_way(thing, self.solid(world, i, k), self.heading(world, i, thing, seg["to"]))
            if way is None:  # nowhere to test it: take the next
                state["current"] = None
                continue
            axis, sign, face = way
            occupied = world.mass[i].numpy() > 0.5
            if all(occupied[c] for c in face):
                state["presses"] += 1
            direction = [0, 0, 0]
            direction[axis] = sign
            self.teams[i] = (tuple(direction), face, k)
            return face, (k, tuple(direction)), PROBERS + 2  # two spares, in case a prober is stuck
        return self.shove_targets(world, i, state["found"], seg["to"])

    def targets_of(self, i, seg, world, shells):
        kind = seg["kind"]
        if kind == "fixed":
            return seg["points"], None, seg["cap"]
        if kind == "shove":
            return self.shove_targets(world, i, seg["cheese"], seg["to"])
        if kind == "find":
            return self.probe_targets(world, i, seg)
        return super().targets_of(i, seg, world, shells)

    def done(self, i, seg, world):
        kind = seg["kind"]
        if kind == "shove":
            thing = self.thing(world, i, seg["cheese"])
            return thing is None or in_place(thing, seg["to"], self.tasks[i].spec["meta_ref"])
        if kind in ("find", "fixed"):
            return False
        return super().done(i, seg, world)

    def describe(self, i, seg, world):
        kind = seg["kind"]
        if kind == "fixed":
            return dict(seg["label"])
        if kind in ("shove", "find"):
            to = {key: value for key, value in seg["to"].items()}
            goal = {"kind": "push", "to": S.json_safe(to)}
            if kind == "shove":
                goal["object"] = int(seg["cheese"])
            else:
                state = self.probes.get(i, {})
                goal.update(kind="find", want=seg["want"], candidates=[int(k) for k in seg["candidates"]], found=state.get("found"))
            return goal
        return super().describe(i, seg, world)

    def segments(self, world, t):
        self.teams = [None] * len(self.tasks)
        return super().segments(world, t)

    def adjust(self, world, codes):
        """The robots of a pushing team step toward the object (in place of the controller's
        choice and its jitter): the face presses, the rows behind press through it (a heavy object
        counts them, world.chain_press) or close up. Robots standing in the object's way (the
        voxels it moves into and the PATH - 1 beyond) step aside, away from its middle, so the slide
        is not stopped by a robot that cannot be shoved further."""
        occupied = world.mass > 0.5
        for i, team in enumerate(self.teams):
            if team is None or not team[1]:
                continue
            direction, pts, k = team
            code = int(direction_code(torch.tensor(direction)))
            thing = self.thing(world, i, k)
            if thing is not None:
                axis = int(np.nonzero(direction)[0][0])
                side = 3 - axis  # the other floor axis
                solid = self.solid(world, i, k) | (cheese_id_map(world.index(i))[0].numpy() == k)
                _, ahead = faces(thing, self.solid(world, i, k), axis, direction[axis])
                way = {tuple(c[a] + direction[a] * j for a in range(3)) for c in ahead for j in range(PATH)}
                for cell in way:
                    if all(0 <= v < SIZE for v in cell) and bool(occupied[i][cell]):
                        away = 1 if cell[side] + 0.5 >= thing["centre"][side - 1] else -1
                        for step in ([0, 0, 0], [0, 0, 0], list(direction), [1, 0, 0]):  # aside, aside the other way, ahead, up: the first free one
                            if not any(step):
                                step[side] = away
                                away = -away
                            there = tuple(c + s for c, s in zip(cell, step))
                            if all(0 <= v < SIZE for v in there) and not solid[there] and not bool(occupied[i][there]):
                                codes[i][cell] = int(direction_code(torch.tensor(step)))
                                break
            cells = torch.tensor(pts).long()
            mine = occupied[i, cells[:, 0], cells[:, 1], cells[:, 2]]
            if self.frozen[i] is not None:
                mine &= ~torch.from_numpy(self.frozen[i])[cells[:, 0], cells[:, 1], cells[:, 2]]
            cells = cells[mine]
            codes[i, cells[:, 0], cells[:, 1], cells[:, 2]] = code
        return codes


# ---------------------------------------------------------------------------------------------
# Drawing tasks


def say(rng, intent, holdout, **slots):
    """An instruction from the wording bank's templates of an intent (held-out ones for holdout ==
    "wording"), its slots filled with their canonical words."""
    from beebot.vla.voxel import wordings as W

    pool = W.templates(intent, "wording" if holdout == "wording" else "train")
    template = pool[int(rng.integers(len(pool)))]
    words = {name: str(slots[name]) if name in slots else W.LEXICAL[name]["canonical"] for name in W.fields_of(template)}
    return template.format(**words)


def object_task(family, text, spec, plan, holdout, meta, runners=()):
    """A Task of an object family; spec gets "objects": True, "forbid" (unused) and "meta_ref",
    the scene's meta, which the expert reads (records leave it out)."""
    spec = {"objects": True, **spec, "forbid": set(), "meta_ref": meta}
    return S.Task(family, text, spec, plan if holdout in ("", "wording") else [], holdout, tuple(runners))


def pile_points(world, i, centre, n, cells=None, layers=None, avoid_objects=True):
    """suite.stack_targets around a floor point, off objects and their shells."""
    heights = world.walls[i].numpy().sum(0)
    avoid = world.walls[i].numpy() | (cheese_id_map(world.index(i))[0].numpy() >= 0)
    if avoid_objects:
        avoid = avoid | S.shells_of(world, i)
    return S.stack_targets(heights, avoid, centre, n, cells=cells, layers=layers)


def tower_points(world, i, height, rng, scaled=False):
    """A 2 x 2 column `height` high on level ground clear of objects and their shells; with
    scaled a single column when the scene's swarm cannot fill the 2 x 2 one (suite.column_side)."""
    side = S.column_side(int(world.mass[i].sum()), height, scaled)
    heights = world.walls[i].numpy().sum(0)
    avoid = world.walls[i].numpy() | (cheese_id_map(world.index(i))[0].numpy() >= 0) | S.shells_of(world, i)
    for _ in range(200):
        y, x = int(rng.integers(1, SIZE - 2)), int(rng.integers(1, SIZE - 2))
        foot = heights[y : y + side, x : x + side]
        z0 = int(foot[0, 0])
        if (foot == z0).all() and z0 + height <= SIZE and not avoid[z0 : z0 + height, y : y + side, x : x + side].any():
            return [(z, y + a, x + b) for z in range(z0, z0 + height) for a in range(side) for b in range(side)]
    return None


def push_goal(rng):
    """(where a push goes, "region" or "trench", and what its scene needs): a region on a flat
    floor (terrain would block the way, as for the suite's herd family), the trench in a gap scene
    with gravity, which drops the object in."""
    if rng.random() < 0.3:
        return "trench", {"layout": "gap", "gravity": True}
    return "region", {"layout": "flat"}


def destination(rng, world, i, meta, thing, goal="region"):
    """A push goal for a thing: into the trench (goal "trench": a thing on a plateau, narrower
    than the trench; None otherwise) or a floor region it is not in."""
    if goal == "trench":
        gap = S.trench(meta)
        if gap is None:
            return None
        axis, start, width, top = gap
        along = thing["cells"][:, axis - 1]
        sign = 1 if along.mean() < start else -1
        if thing["pos"][0] >= top and np.ptp(along) + 1 < width and not (along.min() >= start and along.max() < start + width) and pushable(world, i, thing, axis, sign):
            return {"trench": True}
        return None
    options = [r for r in S.FLOOR_REGIONS if not in_place(thing, {"region": r}, meta) and reachable(world, i, thing, {"region": r}, meta)]
    return {"region": str(rng.choice(options))} if options else None


def pushable(world, i, thing, axis, sign):
    """Whether the expert's team can push a thing along floor axis 1 or 2 by sign where it stands:
    a light one needs FRICTION robots on its face (it does not count robots in line behind them),
    a heavy one HEAVY_FRICTION on its face and in the rows behind (press_room)."""
    solid = solid_of(world, i, thing["slot"])
    if thing["heavy"]:
        return press_room(thing, solid, axis, sign) >= HEAVY_FRICTION
    return len(faces(thing, solid, axis, sign)[0]) >= FRICTION


def reachable(world, i, thing, to, meta):
    """Whether a push toward a floor goal can start along every floor axis it needs (pushable)."""
    offset = goal_centre(to, meta, thing) - thing["centre"]
    return all(pushable(world, i, thing, a + 1, 1 if offset[a] > 0 else -1) for a in range(2) if abs(offset[a]) >= 1)


def push_slots(ref, to):
    """The wording slots and intent of a push."""
    if "trench" in to:
        return "thing.push_trench", {"o": phrase(ref)}
    if "zone" in to:
        return "thing.push_zone", {"o": phrase(ref), "c": to["zone"]}
    return "thing.push_region", {"o": phrase(ref), "r": to["region"]}


def job_task(rng, family, holdout, world, meta, i, job, ref, target, things, find=None, goal="region"):
    """A Task for a job (eat, push, surround, catch; a dict: push there) on the thing a reference
    picks out; goal: where a push goes (push_goal)."""
    thing = by_slot(things)[target]
    name = full_phrase(thing)
    runners = ()
    if job == "eat":
        if thing["noun"] != "cheese":
            return None
        intent, slots, kind = "thing.eat", {"o": phrase(ref)}, "eat_thing"
        spec = {"kind": kind, "ref": ref, "target": target}
        plan = [{"kind": "eat", "cheese": target, "name": name}]
    elif job == "catch":
        if thing["noun"] != "cheese":
            return None
        intent, slots, kind = "thing.catch", {"o": phrase(ref)}, "catch_thing"
        spec = {"kind": kind, "ref": ref, "target": target}
        plan = [{"kind": "eat", "cheese": target, "name": name}]
        runners = (target,)
    elif job == "surround":
        intent, slots, kind = "thing.surround", {"o": phrase(ref)}, "surround_thing"
        spec = {"kind": kind, "ref": ref, "target": target}
        plan = [{"kind": "surround", "cheese": target, "name": name}]
    else:
        to = job if isinstance(job, dict) else destination(rng, world, i, meta, thing, goal)
        if to is None:
            return None
        if "zone" in to and not reachable(world, i, thing, to, meta):
            return None
        intent, slots = push_slots(ref, to)
        spec = {"kind": "push_thing", "ref": ref, "target": target, "to": to}
        if find is not None:
            spec["find"] = "heavy"
            plan = [{"kind": "find", "candidates": find, "want": "heavy", "to": to}]
        else:
            plan = [{"kind": "shove", "cheese": target, "to": to, "name": name}]
    task = object_task(family, say(rng, intent, holdout, **slots), spec, plan, holdout, meta, runners)
    if already_done(task, world, meta, i):
        return None
    return task


def already_done(task, world, meta, i):
    """Whether the scene as it starts already scores half marks."""
    snap = S.snapshot(world, i)
    return score(task, S.static_of(world, meta, i), snap, snap, [])[0] >= 0.5


def heavy_candidates(world, i, things, noun):
    """The things of a noun if exactly one is heavy and each can be tested (probe_way): else None."""
    pool = [t for t in things if t["noun"] == noun]
    if len(pool) < 2 or sum(t["heavy"] for t in pool) != 1:
        return None
    if any(probe_way(t, solid_of(world, i, t["slot"]), (1, 1)) is None for t in pool):
        return None
    return pool


def make_things(rng, holdout, kind=None, scaled=False):
    job = kind or str(rng.choice(["eat", "push", "surround", "catch", "heavy"], p=[0.3, 0.25, 0.2, 0.1, 0.15]))
    goal, ground = push_goal(rng) if job in ("push", "heavy") else (None, {})

    def make(world, meta, i):
        things = things_of(world, i)
        cheeses = [t for t in things if t["noun"] == "cheese"]
        walls = world.walls[i].numpy()
        if job == "heavy":
            pool = heavy_candidates(world, i, things, "cheese")
            if pool is None:
                return None
            target = next(t["slot"] for t in pool if t["heavy"])
            ref = {"by": "attrs", "noun": "cheese", "attrs": {"weight": "heavy"}}
            centre = world.mass[i].nonzero()[:, 1:].float().mean(dim=0).numpy() if world.mass[i].sum() else np.array([SIZE / 2, SIZE / 2])
            order = sorted(pool, key=lambda t: np.linalg.norm(t["centre"] - centre))
            return job_task(rng, "things", holdout, world, meta, i, "push", ref, target, things, find=[t["slot"] for t in order], goal=goal)
        options = cheeses if job != "eat" else [t for t in cheeses if eat_need(t, walls) < float("inf")]
        if not options:
            return None
        target = options[int(rng.integers(len(options)))]["slot"]
        attrs = describe(rng, things, target)
        if attrs is None:
            return None
        return job_task(rng, "things", holdout, world, meta, i, job, {"by": "attrs", "noun": "cheese", "attrs": attrs}, target, things, goal=goal)

    needs = {"objects": (2, 6), "food": 0.7, "zones": (0, 1), **ground}
    if job in ("eat", "surround"):
        needs["layout"] = LOW_GROUND
    if job == "catch":
        needs["layout"] = "flat"
    if job == "heavy":
        needs.update(objects=(2, 4), food=0.8)
    return needs, make


def make_blocks(rng, holdout, kind=None, scaled=False):
    job = kind or str(rng.choice(["push", "heavy", "line"], p=[0.5, 0.2, 0.3]))
    goal, ground = push_goal(rng) if job in ("push", "heavy") else (None, {})

    def make(world, meta, i):
        things = things_of(world, i)
        blocks = [t for t in things if t["noun"] == "block"]
        if job == "heavy":
            pool = heavy_candidates(world, i, things, "block")
            if pool is None:
                return None
            target = next(t["slot"] for t in pool if t["heavy"])
            ref = {"by": "attrs", "noun": "block", "attrs": {"weight": "heavy"}}
            centre = world.mass[i].nonzero()[:, 1:].float().mean(dim=0).numpy() if world.mass[i].sum() else np.array([SIZE / 2, SIZE / 2])
            order = sorted(pool, key=lambda t: np.linalg.norm(t["centre"] - centre))
            return job_task(rng, "blocks", holdout, world, meta, i, "push", ref, target, things, find=[t["slot"] for t in order], goal=goal)
        if job == "line":
            return line_task(rng, holdout, world, meta, i, blocks)
        if not blocks:
            return None
        target = blocks[int(rng.integers(len(blocks)))]["slot"]
        attrs = describe(rng, things, target)
        if attrs is None:
            return None
        return job_task(rng, "blocks", holdout, world, meta, i, "push", {"by": "attrs", "noun": "block", "attrs": attrs}, target, things, goal=goal)

    needs = {"objects": (2, 5), "food": 0.3, "zones": (0, 1), **ground}
    if job == "line":
        needs.update(objects=(2, 3), food=0.0, layout="flat")
    if job == "heavy":
        needs.update(objects=(2, 4), food=0.2)
    return needs, make


def line_task(rng, holdout, world, meta, i, blocks):
    """Line the blocks up in one row (2 to 4 blocks whose spans along the row leave a cell between
    them, so they fit side by side; the row runs left to right or back to front, whichever fits,
    at random when both do): the plan pushes each block not yet on the row the most blocks already
    touch onto it, nearest first. Every block that moves must be pushable toward the row."""
    if not 2 <= len(blocks) <= 4:
        return None
    names = ["x", "y"]
    rng.shuffle(names)
    for axis_name in names:
        across = 2 if axis_name == "x" else 1  # the floor axis the row runs along
        move = 3 - across  # the axis the blocks are pushed along
        spans = sorted((t["cells"][:, across - 1].min(), t["cells"][:, across - 1].max()) for t in blocks)
        if any(b[0] <= a[1] + 1 for a, b in zip(spans, spans[1:])):
            continue
        counts = [sum(bool((t["cells"][:, move - 1] == c).any()) for t in blocks) for c in range(SIZE)]
        best = max(counts)
        rows = [c for c in range(SIZE) if counts[c] == best]
        coord = int(rows[len(rows) // 2])
        moves = [t for t in blocks if not (t["cells"][:, move - 1] == coord).any()]
        if not moves or any(min(abs(t["cells"][:, move - 1] - coord)) > 7 for t in moves):
            continue
        if not all(pushable(world, i, t, move, 1 if coord > t["cells"][:, move - 1].max() else -1) for t in moves):
            continue
        moves.sort(key=lambda t: min(abs(t["cells"][:, move - 1] - coord)))
        plan = [{"kind": "shove", "cheese": t["slot"], "to": {"line": [move, coord]}, "name": full_phrase(t)} for t in moves]
        spec = {"kind": "line_blocks", "axis": axis_name, "line": [move, coord], "moves": [t["slot"] for t in moves]}
        task = object_task("blocks", say(rng, "blocks.line_" + axis_name, holdout), spec, plan, holdout, meta)
        return None if already_done(task, world, meta, i) else task
    return None


def make_zones(rng, holdout, kind=None, scaled=False):
    job = kind or str(rng.choice(["goto", "line", "push"], p=[0.35, 0.3, 0.35]))

    def make(world, meta, i):
        marks = zone_list(world.zones[i].numpy())
        n = int(world.mass[i].sum())
        if job in ("goto", "push"):
            marks = [z for z in marks if z["kind"] == "zone"]
        else:
            marks = [z for z in marks if z["kind"] == "stripe"]
        if not marks:
            return None
        mark = marks[int(rng.integers(len(marks)))]
        y0, y1, x0, x1 = mark["box"]
        if job == "push":
            things = things_of(world, i)
            fits_in = [t for t in things if len(t["cells"]) <= (y1 - y0) * (x1 - x0) and t["extent"][1] <= y1 - y0 and t["extent"][2] <= x1 - x0]
            if not fits_in:
                return None
            target = fits_in[int(rng.integers(len(fits_in)))]["slot"]
            attrs = describe(rng, things, target)
            if attrs is None:
                return None
            ref = {"by": "attrs", "noun": by_slot(things)[target]["noun"], "attrs": attrs}
            return job_task(rng, "zones", holdout, world, meta, i, {"zone": mark["colour"]}, ref, target, things)
        if job == "goto":
            room = 2 * (y1 - y0 + 2) * (x1 - x0 + 2)
            if n > room:
                return None
            cells = np.zeros((SIZE, SIZE), bool)
            cells[max(y0 - 1, 0) : y1 + 1, max(x0 - 1, 0) : x1 + 1] = True
            pts = pile_points(world, i, ((y0 + y1 - 1) / 2, (x0 + x1 - 1) / 2), n, cells=cells, layers=2)
            spec = {"kind": "goto_zone", "zone": mark["colour"]}
            plan = [{"kind": "fixed", "points": pts, "cap": len(pts), "label": {"kind": "zone", "colour": mark["colour"]}}]
            task = object_task("zones", say(rng, "zone.goto", holdout, c=mark["colour"]), spec, plan, holdout, meta)
        else:
            if n < 3:
                return None
            heights = world.walls[i].numpy().sum(0)
            ys, xs = np.nonzero(world.zones[i].numpy() == COLOUR_NAMES.index(mark["colour"]))
            pts = [(int(heights[y, x]), int(y), int(x)) for y, x in zip(ys, xs)][:n]
            spec = {"kind": "line_zone", "zone": mark["colour"]}
            plan = [{"kind": "fixed", "points": pts, "cap": len(pts), "label": {"kind": "stripe", "colour": mark["colour"]}}]
            task = object_task("zones", say(rng, "zone.line", holdout, c=mark["colour"]), spec, plan, holdout, meta)
        return None if already_done(task, world, meta, i) else task

    needs = {"objects": (0, 4), "food": 0.5, "zones": (1, 3)}
    if job == "push":
        needs.update(objects=(1, 4), layout="flat")
    return needs, make


REF_KINDS = ("super", "compare", "near", "next_to")
REF_JOBS = ("eat_thing", "push_thing", "surround_thing")


def draw_ref(rng, by, things, meta):
    """A reference of kind `by` that picks out exactly one thing in the scene, or None."""
    nouns = [n for n in ("cheese", "block") if sum(t["noun"] == n for t in things) >= 2]
    if not nouns:
        return None
    noun = str(rng.choice(nouns))
    pool = [t for t in things if t["noun"] == noun]
    if by == "super":
        attrs = {}
        colours = [c for c in {t["colour"] for t in pool} if sum(matches(t, noun, {"colour": c}) for t in pool) >= 2]
        if colours and rng.random() < 0.4:
            attrs = {"colour": str(rng.choice(sorted(colours)))}
        ref = {"by": "super", "noun": noun, "attrs": attrs, "which": str(rng.choice(["biggest", "smallest"]))}
    elif by == "compare":
        options = []
        for group in ("colour", "pattern", "shape"):
            for value in {word(t, group) for t in pool} - {None}:
                if sum(matches(t, noun, {group: value}) for t in pool) == 2:
                    options.append({group: value})
        if not options:
            return None
        ref = {"by": "compare", "noun": noun, "attrs": options[int(rng.integers(len(options)))], "which": str(rng.choice(["smaller", "larger"]))}
    elif by == "near":
        anchors = [{"zone": z["colour"], "kind": z["kind"] if z["kind"] == "stripe" else "zone"} for z in zone_list(meta["props"]["zones"])]
        if S.trench(meta) is not None:
            anchors.append({"trench": True})
        if not anchors:
            return None
        ref = {"by": "near", "noun": noun, "attrs": {}, "which": str(rng.choice(["nearest", "furthest from"])), "anchor": anchors[int(rng.integers(len(anchors)))]}
    else:
        others = [t for t in things]
        rng.shuffle(others)
        ref = None
        for anchor in others:
            attrs = describe(rng, things, anchor["slot"])
            if attrs is None:
                continue
            candidate = {"by": "next_to", "noun": noun, "attrs": {}, "anchor": {"by": "attrs", "noun": anchor["noun"], "attrs": attrs}}
            if resolve(candidate, things, meta) is not None:
                ref = candidate
                break
        if ref is None:
            return None
    return ref if resolve(ref, things, meta) is not None else None


def make_reasoning(rng, holdout, kind=None, scaled=False):
    if holdout == "ref_job":
        by, job = HELD_OUT["ref_job"][int(rng.integers(len(HELD_OUT["ref_job"])))]
        mode = "ref"
    elif holdout == "condition":
        mode, form = "conditional", HELD_OUT["condition"][0]
    elif holdout == "count":
        mode, count_job = "count", "groups"
    else:
        mode = kind or str(rng.choice(["ref", "conditional", "count"], p=[0.6, 0.25, 0.15]))
        if mode == "ref":
            while True:
                by, job = str(rng.choice(REF_KINDS)), str(rng.choice(REF_JOBS))
                if (by, job) not in HELD_OUT["ref_job"]:
                    break
        elif mode == "conditional":
            form = str(rng.choice(["eat_tower", "surround_goto"]))
        else:
            count_job = "tower"
    goal, ground = push_goal(rng) if mode == "ref" and job == "push_thing" else (None, {})

    def make(world, meta, i):
        things = things_of(world, i)
        if mode == "ref":
            ref = draw_ref(rng, by, things, meta)
            if ref is None:
                return None
            target = resolve(ref, things, meta)
            name = {"eat_thing": "eat", "push_thing": "push", "surround_thing": "surround"}[job]
            if name == "eat" and eat_need(by_slot(things)[target], world.walls[i].numpy()) == float("inf"):
                return None
            return job_task(rng, "reasoning", holdout, world, meta, i, name, ref, target, things, goal=goal)
        if mode == "conditional":
            return conditional_task(rng, holdout, world, meta, i, things, form, scaled)
        return count_task(rng, holdout, world, meta, i, things, count_job, scaled)

    needs = {"objects": (3, 6), "food": 0.5, "zones": (0, 2), **ground}
    if mode == "ref" and job != "push_thing" and by != "near":
        needs["layout"] = LOW_GROUND
    if mode == "conditional":
        needs.update(objects=(1, 5), food=0.7, layout="flat")
    if mode == "count":
        needs.update(objects=(3, 6))
    return needs, make


def conditional_task(rng, holdout, world, meta, i, things, form, scaled=False):
    """"eat (surround) the c cheese if there is one, otherwise build a tower (go to a region)":
    c names one cheese in the scene half the time and none of them otherwise."""
    cheeses = [t for t in things if t["noun"] == "cheese"]
    holds = bool(rng.random() < 0.5)
    if holds:
        options = [t for t in cheeses if sum(matches(u, "cheese", {"colour": t["colour"]}) for u in cheeses) == 1]
        if not options:
            return None
        thing = options[int(rng.integers(len(options)))]
        colour = thing["colour"]
    else:
        colours_seen = {c for t in cheeses for c in (t["colour"], t["colour2"]) if c}
        missing = [c for c in S.TRAIN_COLOURS if c not in colours_seen]
        colour = str(rng.choice(missing))
        thing = None
    then_job, else_job = {"eat_tower": ("eat", "tower"), "surround_goto": ("surround", "goto"), "eat_goto": ("eat", "goto")}[form]
    ref = {"by": "attrs", "noun": "cheese", "attrs": {"colour": colour}}
    then_spec = {"kind": then_job + "_thing", "ref": ref, "target": None if thing is None else thing["slot"]}
    slots = {"c": colour}
    if else_job == "tower":
        height = int(rng.integers(3, 7))
        else_spec = {"kind": "tower", "height": height}
        slots["n"] = height
    else:
        region = str(rng.choice(list(S.FLOOR_REGIONS)))
        else_spec = {"kind": "goto", "region": region}
        slots["r"] = region
    spec = {"kind": "conditional", "form": form, "colour": colour, "holds": holds, "then": then_spec, "else": else_spec}
    if holds:
        if then_job == "eat" and eat_need(thing, world.walls[i].numpy()) == float("inf"):
            return None
        plan = [{"kind": "eat" if then_job == "eat" else "surround", "cheese": thing["slot"], "name": full_phrase(thing)}]
    elif else_job == "tower":
        pts = tower_points(world, i, else_spec["height"], rng, scaled)
        if pts is None:
            return None
        plan = [{"kind": "fixed", "points": pts, "cap": len(pts) + 6, "label": {"kind": "tower", "height": else_spec["height"], "at": list(pts[0])}}]
    else:
        y0, y1, x0, x1 = S.region_box(else_spec["region"])
        cells = np.zeros((SIZE, SIZE), bool)
        cells[max(y0 - 1, 0) : y1 + 1, max(x0 - 1, 0) : x1 + 1] = True
        n = int(world.mass[i].sum())
        pts = pile_points(world, i, ((y0 + y1 - 1) / 2, (x0 + x1 - 1) / 2), n, cells=cells)
        plan = [{"kind": "fixed", "points": pts, "cap": None, "label": {"kind": "region", "region": else_spec["region"]}}]
    task = object_task("reasoning", say(rng, "cond." + form, holdout, **slots), spec, plan, holdout, meta)
    return None if already_done(task, world, meta, i) else task


def count_task(rng, holdout, world, meta, i, things, job, scaled=False):
    """"build a tower as tall as the number of cheeses (blocks, objects)", or (held out) "split into
    as many groups as there are ...": the count is 3 to 6 for a tower and 2 or 3 for groups."""
    noun = str(rng.choice(["cheese", "block", "object"]))
    count = sum(matches(t, noun, {}) for t in things)
    if (job == "tower" and not 3 <= count <= 6) or (job == "groups" and count not in (2, 3)):
        return None
    spec = {"kind": "count", "job": job, "of": PLURAL[noun], "count": count}
    plan = []
    if job == "tower":
        pts = tower_points(world, i, count, rng, scaled)
        if pts is None:
            return None
        plan = [{"kind": "fixed", "points": pts, "cap": len(pts) + 6, "label": {"kind": "tower", "height": count, "at": list(pts[0])}}]
    task = object_task("reasoning", say(rng, "count." + job, holdout, k=PLURAL[noun]), spec, plan, holdout, meta)
    return None if already_done(task, world, meta, i) else task


MAKERS = {"things": make_things, "blocks": make_blocks, "zones": make_zones, "reasoning": make_reasoning}
SCENE_DRAW = 4  # scenes sample_object_scenes draws at once; the first that fits is used


def object_scenes(rng, robots=None, layout=None, gravity=None, objects=(1, 6), food=0.5, zones=(0, 2), empty=False):
    """(world, meta) of one scene with objects (world.sample_object_scenes) in the layout asked (a
    name or a tuple of them), meta with "props" (props_of). empty: an empty flat floor
    (suite.empty_scene) with the object fields and MAX_OBJECTS empty slots, whatever is asked."""
    if empty:
        world, meta = S.empty_scene(rng, robots)
        world = with_objects(world, MAX_OBJECTS)
        return world, {**meta, "zones": [], "props": props_of(world, 0)}
    while True:
        world, metas = sample_object_scenes(rng, SCENE_DRAW, SIZE, 0.5 if gravity is None else float(gravity), robots, objects, food, zones, TRAIN_COLOURS)
        for i in range(SCENE_DRAW):
            if layout is not None and metas[i]["layout"] not in ((layout,) if isinstance(layout, str) else layout):
                continue
            one = world.index(i)
            meta = dict(metas[i])
            meta["props"] = props_of(one, 0)
            return one, meta


def sample_objects(rng, n, family=None, split="train", robots=None, counts="uniform", check=True, kind=None, keep=False, empty=False, scaled=False):
    """(world, metas, tasks) of n scenes with one object task each (OBJECT_FAMILIES), as
    suite.sample_coverage draws its own: family None draws one per scene; split "train", "test",
    "wording" (held-out templates) or a HELD_OUT key (forces that composition, no plan); robots
    (low, high) draws each scene's robot count (counts: "uniform" or "log"), None the world's
    density range; check keeps only tasks whose bound reaches the pass mark; kind: only that job
    of the family (for measuring). keep: as suite.sample_tasks' keep (the task and its count are
    kept for up to suite.KEEP_TRIES scenes); keep=False draws exactly as before. empty: every
    scene an empty flat floor (object_scenes empty: what is left is a conditional's other
    branch), and with keep a task no scene fits is given up, so fewer than n come back
    (suite.sample_tasks). scaled: a tower the swarm cannot fill as a 2 x 2 column is a single
    one (tower_points)."""
    holdout = split if split in HELD_OUT or split == "wording" else ""
    worlds, metas, out = [], [], []
    kept, tries, given_up = None, 0, 0
    while len(out) + given_up < n:
        if empty and keep and kept is not None and tries >= S.KEEP_TRIES:
            given_up += 1
            kept = None
            continue
        if not keep or kept is None or tries >= S.KEEP_TRIES:
            fam = HELD_FAMILY.get(holdout) or family or str(rng.choice(OBJECT_FAMILIES))
            needs, make = MAKERS[fam](rng, holdout, kind, scaled)
            count = None if robots is None else S.draw_count(rng, robots[0], robots[1], counts)
            kept, tries = (needs, make, count), 0
        needs, make, count = kept
        needs, tries = dict(needs), tries + 1
        world, meta = object_scenes(rng, count, empty=empty, **needs)
        task = make(world, meta, 0)
        if task is None or (check and bound(task, world, 0) < S.SUCCESS.get(task.family, S.PASS_MARK)):
            continue
        kept = None
        worlds.append(world)
        metas.append(meta)
        out.append(task)
    if not out:
        return None, [], []
    return concat(worlds), metas, out


# ---------------------------------------------------------------------------------------------
# Records: canonical slots, specs, sub-goals, scene summaries


def slots_of(task):
    """The canonical wording slots of an object task (wordings.slots_of)."""
    spec = task.spec
    kind = spec["kind"]
    if kind in ("eat_thing", "catch_thing", "surround_thing"):
        return {"intent": {"eat_thing": "thing.eat", "catch_thing": "thing.catch", "surround_thing": "thing.surround"}[kind], "o": phrase(spec["ref"])}
    if kind == "push_thing":
        intent, slots = push_slots(spec["ref"], spec["to"])
        return {"intent": intent, **slots}
    if kind == "line_blocks":
        return {"intent": "blocks.line_" + spec["axis"]}
    if kind in ("goto_zone", "line_zone"):
        return {"intent": "zone.goto" if kind == "goto_zone" else "zone.line", "c": spec["zone"]}
    if kind == "conditional":
        out = {"intent": "cond." + spec["form"], "c": spec["colour"]}
        if spec["else"]["kind"] == "tower":
            out["n"] = spec["else"]["height"]
        else:
            out["r"] = spec["else"]["region"]
        return out
    if kind == "count":
        return {"intent": "count." + spec["job"], "k": spec["of"]}
    raise ValueError(kind)


def record_spec(task):
    """What an instruction pins down in its scene, as plain JSON: the job's kind, "target" (the
    object's id in scene_summary's "objects") and where it goes; a conditional the branch it takes
    ("holds", "do"); counting the number counted."""
    spec = task.spec
    kind = spec["kind"]
    if kind in ("eat_thing", "catch_thing", "surround_thing", "push_thing"):
        out = {"kind": kind, "target": int(spec["target"])}
        if kind == "push_thing":
            to = spec["to"]
            out.update({"trench": True} if "trench" in to else {"zone": to["zone"]} if "zone" in to else {"region": to["region"]})
        if spec.get("find"):
            out["weight"] = spec["find"]
        return out
    if kind == "line_blocks":
        return {"kind": kind, "axis": spec["axis"]}
    if kind in ("goto_zone", "line_zone"):
        return {"kind": kind, "zone": spec["zone"]}
    if kind == "conditional":
        inner = branch(spec)
        do = {"kind": inner["kind"], "target": int(inner["target"])} if "target" in inner else {k: v for k, v in inner.items()}
        return {"kind": "conditional", "holds": bool(spec["holds"]), "do": do}
    if kind == "count":
        return {"kind": "count_" + spec["job"], "of": spec["of"], ("height" if spec["job"] == "tower" else "count"): int(spec["count"])}
    raise ValueError(kind)


def told(task):
    """Slots whose weight the task's instruction tells ("the heavy block")."""
    spec = task.spec
    return {int(spec["target"])} if spec.get("objects") and spec.get("find") else set()


def goal_words(to):
    if "trench" in to:
        return "into the trench"
    if "zone" in to:
        return f"onto the {to['zone']} zone"
    if "region" in to:
        return f"to the {to['region']} of the floor"
    axis, coord = to["line"]
    return f"into the row at {'y' if axis == 1 else 'x'} = {coord}"


def subgoal_sentences(task):
    """One plain sentence per plan segment, in plan order: eat, surround or push the object named
    by every visible property (full_phrase, kept in the segment as "name" when drawn), find the
    heavy one, or the task's plain sentence for target voxels."""
    from beebot.vla.voxel import wordings as W

    plain = W.canonical(W.slots_of(task))
    out = []
    for seg in task.plan:
        kind = seg["kind"]
        if kind in ("eat", "surround"):
            out.append(f"{kind} {seg['name']}")
        elif kind == "shove":
            out.append(f"push {seg['name']} {goal_words(seg['to'])}")
        elif kind == "find":
            noun = task.spec["ref"]["noun"]
            out.append(f"find the heavy {noun} by pushing each {noun} with two robots, then push it {goal_words(seg['to'])}")
        else:
            out.append(plain)
    return out


def summary(world, i, told_slots=()):
    """The objects and zones of scene i for suite.scene_summary: {"objects": [{"id" (the slot),
    "kind" (cheese or block), "colour", "second_colour" (patterned only), "pattern", "shape",
    "size" (word), "sides": {"x", "y", "height"}, "voxels", "area", "on", "x", "y", "z" (lowest
    corner), "weight" (only when the task tells it)}], "zones": [{"colour", "kind" (zone or
    stripe), "area", "x0", "x1", "y0", "y1" (inclusive)}]}."""
    walls = world.walls[i].numpy()
    trench = "trench" in str(S.terrain_summary(walls.sum(0)).get("layout", ""))
    objects = []
    for t in things_of(world, i):
        z, y, x = t["pos"]
        entry = {"id": int(t["slot"]), "kind": t["noun"], "colour": t["colour"]}
        if t["colour2"]:
            entry["second_colour"] = t["colour2"]
        entry.update(pattern=t["pattern"], shape=t["shape"], size=SIZE_WORDS[t["size"]], sides={"x": t["extent"][2], "y": t["extent"][1], "height": t["extent"][0]}, voxels=t["voxels"], area=S.region_of(*t["centre"]), on="floor" if z == 0 else "plateau" if trench else "hill", x=x, y=y, z=z)
        if t["slot"] in told_slots:
            entry["weight"] = "heavy" if t["heavy"] else "light"
        objects.append(entry)
    zones = []
    for mark in zone_list(world.zones[i].numpy()):
        y0, y1, x0, x1 = mark["box"]
        zones.append({"colour": mark["colour"], "kind": mark["kind"], "area": S.region_of((y0 + y1) / 2, (x0 + x1) / 2), "x0": x0, "x1": x1 - 1, "y0": y0, "y1": y1 - 1})
    return {"objects": objects, "zones": zones}


def flee(world, runners):
    """suite.flee for a world with forms: each alive runner slides one voxel sideways away from the
    robots within suite.RUNNER_RADIUS of its middle, if the voxels it moves into are in the grid,
    clear of walls, robots and other objects, and (under gravity) all its lowest voxels stay on
    walls or the floor."""
    pos = world.cheese_pos.clone()
    mass = world.mass.numpy() > 0.5
    walls = world.walls.numpy()
    ids = cheese_id_map(world).numpy()
    b, d, h, w = world.mass.shape
    for i, k in zip(*np.nonzero((runners & world.cheese_alive).numpy())):
        form = world.form[i, k].numpy()
        rel = np.argwhere(form)
        z, y, x = (int(v) for v in pos[i, k])
        middle = rel.mean(axis=0) + np.array([z, y, x]) + 0.5
        lo = np.maximum(np.floor(middle - S.RUNNER_RADIUS).astype(int), 0)
        hi = np.minimum(np.floor(middle + S.RUNNER_RADIUS).astype(int) + 1, [d, h, w])
        near = np.argwhere(mass[i, lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]) + lo
        if not len(near):
            continue
        dy, dx = (near[:, 1:] + 0.5 - middle[1:]).mean(axis=0)
        lowest = np.array([r for r in rel.tolist() if r[0] == 0 or not form[r[0] - 1, r[1], r[2]]])  # voxels with no voxel of the object under them
        for _, my, mx in sorted([(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], key=lambda m: m[1] * dy + m[2] * dx):
            cells = rel + np.array([z, y + my, x + mx])
            if (cells < 0).any() or (cells >= np.array([d, h, w])).any():
                continue
            cz, cy, cx = cells.T
            if walls[i, cz, cy, cx].any() or mass[i, cz, cy, cx].any() or ((ids[i, cz, cy, cx] >= 0) & (ids[i, cz, cy, cx] != k)).any():
                continue
            if bool(world.gravity[i]) and z > 0:
                under = lowest + np.array([z - 1, y + my, x + mx])
                if not walls[i, under[:, 0], under[:, 1], under[:, 2]].all():
                    continue
            pos[i, k] = torch.tensor([z, y + my, x + mx])
            break
    return pos
