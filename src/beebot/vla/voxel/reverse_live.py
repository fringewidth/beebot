"""Basic tasks as live reverse demonstrations (runs/handoff/reverse-loader.md): for one of the basic
task families (move in a direction, go to a corner or side, gather, split into groups, spread out,
line up, stay, climb onto a plateau, split between two places or a place and a tower, a tower, a
bridge over a trench, a coloured patch or strip on the floor, pushing an object onto a patch) draw a
random world (terrain and loose blocks as the stores draw them, robot count drawn log-uniform from 6
to 115), put the robots in an end state that does what the instruction asks at a glance, and walk
backwards from it with reverse.take_apart (every step verified by the world's own forward step).
Played forward, the episode is a demonstration of the instruction. Every draw is a new world, a new
end state and a new walk back, so no two examples share an episode unless the cache reuses one.

Opt-in: ThinkData gives live examples only with `live=` (think_data.ThinkData); without it its
examples are byte for byte what they were (tests/test_reverse_live.py).

    ThinkData(stores, haiku_dirs, ..., live={"share": 0.2, "workers": 4})

The words. Each draw has a record like the stores' ({family, kind, plain, spec, subgoals}, the spec
exactly as the stores write it). A Haiku rewording (and its thought) of an instruction with the
same spec is used when one passes the filter (text_ok: no colour, number, side or thing the spec
does not name, none of the simulator's own nouns cheese, zone or stripe); its thought keeps the
quote, the meaning and the first sub-goal when they pass the same filter, and says what this scene
shows (think_data.sight of the live scene) in place of what Haiku saw. Otherwise the suite's
template wording with think_data.template_thought. Floor zones are said "patch" and "strip".

End states (strict: what a stranger would agree with at a glance) and how the walk runs:

1. goto r        every robot piled in the named ninth of the floor (its box, one cell of margin),
                 low piles (one layer up costs as much as LAYER_COST squared cells); natural start
                 (reverse's natural mode); at most a third of the robots in the box at the start.
1. gather        one pile (one 26-connected group) anywhere; the start has at least two groups and
                 none holds half the robots.
1. groups m      m piles (2 or 3) at the places of suite.GROUP_PAIRS, at least two empty columns
                 apart: exactly m groups; the start's largest group under half.
1. spread        no two robots within one cell of each other (not even at a corner), all on the
                 ground; start: one to three piles (guided walk to that start).
1. line axis     3-14 robots in one straight row on the ground (left to right or back to front),
                 contiguous; the start has no straight run of half of them.
1. move side     the start is the end shifted three columns away from that side (each column's
                 stack moved, resting on the ground there), the swarm in a window of about 2n
                 columns, at most MOVE_MOST (40) robots so the shift shows; guided walk.
1. stay          a natural scene held still for reverse.HOLD_STEPS steps.
1. climb side    every robot piled on that plateau (a trench scene); the start in the trench and
                 on the other plateau (guided walk).
1. split_regions two piles in two named ninths (half the robots each, two columns apart).
1. split_tower   a pile in a named ninth and a tower h (3-6) high, single column or 2 x 2, outside it.
1. tower h       one tower h (3-8) high, single column or 2 x 2; every other robot on the ground,
                 not on another robot and not next to the tower; the start's tallest stack under h.
1. bridge        a trench filled solid across in a lane one or two wide, flush with the plateaus;
                 other robots on the ground, one layer.
1. zone c        every robot on a c patch (3-4 cells a side; at most two layers, so at most twice
                 its cells in robots), with 0-2 other patches or strips of other colours.
1. zone_line c   every robot on a c strip (1 x 6-10), one per cell, in a row; robots at most the
                 strip's length.
1. push          reverse.push_goal: a box, tile or bar on a coloured floor patch, pushers behind it
                 (pushed 3-6 cells), 0-3 others.

The construction (bridge's "fill the trench") is left out: today's counts (at most 115 robots)
cannot fill a trench (3-4 wide, 3-4 deep, 16 long: 144-256 cells), so no end state passes at a
glance.

Workers. An episode takes 0.3-1.7 CPU s at 6-20 robots and 2-9 s at 51-115 (spread up to 20 s; 1.9 s
on average over the families: scripts/reverse_live_measure.py, runs/handoff/reverse-loader.md);
Live(workers=n) runs n processes (nice, one thread each) that keep a cache of the latest `cache`
episodes full; sample draws an episode from the cache (reused while new ones come in) and a step of
it, so the trainer never waits after the first episode. workers=0 generates in the caller (tests,
measurements): a new episode on a draw with probability 1 / reuse.
"""

import hashlib
import json
import os
import pickle
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel import reverse as R
from beebot.vla.voxel import suite as S
from beebot.vla.voxel import think_data as T
from beebot.vla.voxel import world as W
from beebot.vla.voxel.camera_frame import quarter_turns, turn_codes, turn_grid
from beebot.vla.voxel.world import COLOUR_NAMES, N_CHANNELS, SIZE, STAY

FAMILIES = ("goto", "gather", "groups", "spread", "line", "move", "stay", "climb", "split_regions", "split_tower", "tower", "bridge", "zone", "zone_line", "push")
STORE_KIND = {  # live family -> (the stores' family, kind) whose Haiku texts it reuses
    "goto": ("motion", "goto"), "gather": ("motion", "gather"), "groups": ("motion", "groups"), "spread": ("motion", "spread"),
    "line": ("motion", "line"), "move": ("motion", "move"), "stay": ("motion", "stay"), "climb": ("motion", "climb"),
    "split_regions": ("parallel", "parallel:region+region"), "split_tower": ("parallel", "parallel:region+tower"),
    "tower": ("construction", "tower"), "bridge": ("construction", "bridge"), "zone": ("zones", "goto_zone"), "zone_line": ("zones", "line_zone"),
    "push": ("objects", "push"),
}
REGIONS = tuple(S.FLOOR_REGIONS)  # the nine areas: back left ... front right, center
SIDES = ("left", "right", "back", "front")
COLOURS = tuple(c for c in S.TRAIN_COLOURS)
ROBOTS = (6, 115)  # the stores' draw (data/vla/haiku/run-s*/config.json: robots [6, 115], counts log)
LAYER_COST = 3.0  # piles: one layer up costs as much as this many squared cells sideways (level squared)
MARGIN = 1  # a region's box grown by this (suite.job_value's goto)
MOVE_MOST = 40  # move: at most this many robots (a swarm of 60+ fills most of the floor and a shift of three does not show)
TRIES = 8  # draws tried per episode before giving up on the family
HAIKU_DIRS = tuple(f"data/vla/haiku/run-s{s}{later}" for s in range(8) for later in ("", "-later"))
RUNGS = ("close", "indirect", "casual")
TEMPLATE_SHARE = 0.25  # of examples whose spec has Haiku texts: the template wording instead
RUNG = "live"
INTERNAL = re.compile(r"\b(?:cheese|cheeses|zone|zones|stripe|stripes|striped)\b", re.I)
COORDINATES = re.compile(r"\b(?:[xyz]|zero|coordinates?|axis|axes)\b")
FOREIGN_AREAS = {  # area words of the Spanish and French casual rungs, as the English area they name
    "izquierda": "left", "izq": "left", "gauche": "left", "derecha": "right", "der": "right", "droite": "right",
    "adelante": "front", "delante": "front", "frente": "front", "devant": "front", "atras": "back", "atrás": "back",
    "detras": "back", "detrás": "back", "fondo": "back", "derriere": "back", "derrière": "back", "centro": "center",
    "medio": "center", "milieu": "center", "esquina": "corner", "coin": "corner", "lado": "side", "côté": "side", "cote": "side",
}
AREA_WORDS = ("left", "right", "front", "back", "center", "centre", "middle", "corner", "corners", "side", "sides", "edge", "edges")
SCENE_WORDS = re.compile(r"\b(?:i see|there (?:is|are)|here|this scene|already|scattered|clumps?|bunched|mostly|loose|alone|lying|sitting|nearest|runs?|running|deep|wide|plateau is|trench is|trench runs|towers?|stacks?|stacked|piles?|heaps?|broken|currently|now|at the moment|on top|flat|spread over|far|close|apart from)\b", re.I)


# ---------------------------------------------------------------------------------------------
# Worlds and end states


def _colour_ids(names):
    return [COLOUR_NAMES.index(c) for c in names]


def draw_world(rng, layout=None, objects=(0, 2)):
    """(world with every object field and gravity on, heightmap (H, W) long, meta) of one scene
    without robots: terrain as the stores draw it (flat, low hills or a trench between two
    plateaus; `layout` forces one), 0-2 loose blocks (objects that are not food) as
    world.sample_object_scenes places them, no floor zones."""
    for _ in range(64):
        world, meta = W.sample_object_scenes(rng, 1, robots=0, objects=objects, food=0.0, zones=(0, 0), colours=_colour_ids(COLOURS))
        if layout is None or meta[0]["layout"] == layout:
            break
    world = replace(world, gravity=torch.ones(1, dtype=torch.bool), mass=torch.zeros_like(world.mass))
    return world, meta[0]["height"].clone(), meta[0]


def blocked_columns(world):
    """(H, W) bool: columns holding a block (no robot is piled there)."""
    return (W.cheese_id_map(world)[0] >= 0).any(dim=0).numpy()


def pile(rng, height, n, centre, allowed, levels=None, jitter=0.6, layer_cost=LAYER_COST, max_level=None):
    """n cells piled on the ground round centre (y, x), column by column: the next cell is the one
    of least squared distance + layer_cost x level^2 (level 0 on the ground), with a little noise
    per column so piles are not regular. allowed (H, W) bool: columns that may take robots;
    levels (H, W) int: robots already in each column (piles share columns). Returns [(z, y, x)]
    or None when they do not fit."""
    hgt = np.asarray(height)
    h, w = hgt.shape
    level = np.zeros((h, w), int) if levels is None else levels
    ys, xs = np.mgrid[0:h, 0:w]
    base = (ys - centre[0]) ** 2 + (xs - centre[1]) ** 2 + rng.normal(0, jitter, (h, w))
    cells = []
    for _ in range(n):
        cost = base + layer_cost * level.astype(float) ** 2
        bad = ~allowed | (hgt + level >= SIZE)
        if max_level is not None:
            bad |= level >= max_level
        cost[bad] = np.inf
        k = int(np.argmin(cost))
        if not np.isfinite(cost.flat[k]):
            return None
        y, x = divmod(k, w)
        cells.append((int(hgt[y, x] + level[y, x]), y, x))
        level[y, x] += 1
    return cells


def region_columns(name, margin=MARGIN):
    """(H, W) bool: the region's box (suite.region_box) grown by margin."""
    y0, y1, x0, x1 = S.region_box(name)
    out = np.zeros((SIZE, SIZE), bool)
    out[max(y0 - margin, 0) : y1 + margin, max(x0 - margin, 0) : x1 + margin] = True
    return out


def region_centre(rng, name):
    """A pile centre in a region: its box's middle pulled 1.5 cells toward the edges it names
    (a corner pile sits in the corner), jittered by up to a cell."""
    y0, y1, x0, x1 = S.region_box(name)
    cy, cx = (y0 + y1 - 1) / 2, (x0 + x1 - 1) / 2
    words = name.split()
    cy += -1.5 if "back" in words else 1.5 if "front" in words else 0.0
    cx += -1.5 if "left" in words else 1.5 if "right" in words else 0.0
    return cy + rng.uniform(-1, 1), cx + rng.uniform(-1, 1)


def chebyshev_grow(mask, r):
    """(H, W) bool grown by r columns in every direction (8-neighbourhood)."""
    t = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
    return (torch.nn.functional.max_pool2d(t, 2 * r + 1, stride=1, padding=r)[0, 0] > 0).numpy()


def columns_of(cells):
    out = np.zeros((SIZE, SIZE), bool)
    for _, y, x in cells:
        out[y, x] = True
    return out


def ground_layer(rng, height, n, allowed, apart=False):
    """n robots on the ground, one per column, in random allowed columns; apart: no two within one
    column of each other (a corner counts). Returns cells or None."""
    hgt = np.asarray(height)
    free = allowed & (hgt < SIZE)
    cells = []
    for k in rng.permutation(int(free.size)):
        if len(cells) == n:
            break
        y, x = divmod(int(k), SIZE)
        if not free[y, x]:
            continue
        cells.append((int(hgt[y, x]), y, x))
        free[y, x] = False
        if apart:
            free[max(y - 1, 0) : y + 2, max(x - 1, 0) : x + 2] = False
    return cells if len(cells) == n else None


def scatter_stacked(rng, height, n, allowed, cap=R.HEAP):
    """n robots dropped in random allowed columns as natural scenes drop them (stacked when a column
    is drawn again, at most cap high). Returns cells or None when they do not fit."""
    hgt = np.asarray(height)
    level = np.zeros(hgt.shape, int)
    open_ = allowed & (hgt + cap <= SIZE)
    ys, xs = np.nonzero(open_)
    if not len(ys):
        return None
    cells = []
    for k in rng.integers(0, len(ys), 8 * n + 40):
        if len(cells) == n:
            break
        y, x = int(ys[k]), int(xs[k])
        if level[y, x] >= cap:
            continue
        cells.append((int(hgt[y, x] + level[y, x]), y, x))
        level[y, x] += 1
    return cells if len(cells) == n else None


def components(occ):
    """26-connected groups of robots in a (D, H, W) bool grid, largest first ([(n, 3) arrays])."""
    return S.components(np.asarray(occ))


def tallest(occ):
    """The tallest vertical run of robots."""
    occ = np.asarray(occ)
    run = np.zeros(occ.shape[1:], int)
    best = 0
    for z in range(occ.shape[0]):
        run = np.where(occ[z], run + 1, 0)
        best = max(best, int(run.max()))
    return best


def on_ground(occ, height):
    """(D, H, W) bool: robots resting on the floor or the terrain (not on a robot or a block)."""
    occ = np.asarray(occ)
    z = np.arange(occ.shape[0]).reshape(-1, 1, 1)
    return occ & (z == np.asarray(height)[None])


def longest_run(occ, axis):
    """The longest straight run of robots along axis (1: back to front, 2: left to right)."""
    occ = np.asarray(occ)
    best = 0
    run = np.zeros_like(occ, dtype=int)
    for k in range(occ.shape[axis]):
        here = np.take(occ, k, axis=axis)
        prev = np.take(run, k - 1, axis=axis) if k else np.zeros_like(here, dtype=int)
        cur = np.where(here, prev + 1, 0)
        idx = [slice(None)] * 3
        idx[axis] = k
        run[tuple(idx)] = cur
        best = max(best, int(cur.max()) if cur.size else 0)
    return best


def touching_any(occ):
    """Pairs of robots within one cell of each other (26-neighbourhood), counted once."""
    occ = torch.from_numpy(np.asarray(occ)).float()[None, None]
    near = torch.nn.functional.conv3d(occ, torch.ones(1, 1, 3, 3, 3), padding=1)[0, 0] - occ[0, 0]
    return int((near * occ[0, 0]).sum().item()) // 2


# ---------------------------------------------------------------------------------------------
# Draws: per family, a goal (reverse.Goal) with its record and the check of start and end


@dataclass
class Draw:
    family: str
    record: dict  # {"family", "kind", "plain", "spec", "subgoals"} as the stores write them
    templates: list  # template wordings of the instruction (the suite's training ones)
    goal: R.Goal
    check: object  # check(start_occ, end_occ) -> "" when both are what the instruction asks, else why not
    info: dict


def record(family, plain, spec):
    fam, kind = STORE_KIND[family]
    return {"family": fam, "kind": kind, "plain": plain, "spec": spec, "subgoals": [plain]}


def train_wordings(key, **kw):
    pool = S.WORDINGS[key] if key in S.WORDINGS else S.COVERAGE_WORDINGS[key]
    return [w.format(**kw) for w in pool[:-1]]


_RANGE = list(ROBOTS)  # the robot counts drawn (generate(robots=...) narrows it for one call)


def robot_count(rng, low=None):
    """A robot count drawn as the stores draw it (log-uniform over _RANGE), at least low."""
    lo, hi = _RANGE
    lo = max(lo, low or 0)
    return S.draw_count(rng, lo, max(hi, lo), "log")


def place(world, cells):
    for z, y, x in cells:
        world.mass[0, z, y, x] = 1.0
    return world


def goal_of(family, world, cells, start=None, text=""):
    world = place(world, cells)
    target = world.mass[0] > 0.5
    return R.Goal(family, text, world, target, start=start)


def share_in(occ, columns):
    cells = np.argwhere(np.asarray(occ))
    return float(columns[cells[:, 1], cells[:, 2]].mean()) if len(cells) else 0.0


def draw_goto(rng):
    world, height, _ = draw_world(rng)
    r = str(rng.choice(REGIONS))
    n = robot_count(rng)
    cols = region_columns(r) & ~blocked_columns(world)
    cells = pile(rng, height, n, region_centre(rng, r), cols)
    if cells is None:
        return None
    plain = "go to the center of the floor" if r == "center" else f"go to the {r} of the floor"
    box = region_columns(r)

    def check(start, end):
        if share_in(end, box) < 1.0:
            return "end: robots outside the region"
        if share_in(start, box) > 1 / 3:
            return "start: a third already in the region"
        return ""

    return Draw("goto", record("goto", plain, {"kind": "goto", "region": r}), train_wordings("region", r=r), goal_of("goto", world, cells), check, {"region": r})


def draw_gather(rng):
    world, height, _ = draw_world(rng)
    n = robot_count(rng)
    free = ~blocked_columns(world)
    centre = (rng.uniform(3, SIZE - 4), rng.uniform(3, SIZE - 4))
    cells = pile(rng, height, n, centre, free)
    if cells is None:
        return None

    def check(start, end):
        if len(components(end)) != 1:
            return "end: not one group"
        groups = components(start)
        if len(groups) < 2 or len(groups[0]) > n / 2:
            return "start: already mostly together"
        return ""

    return Draw("gather", record("gather", "gather into one group", {"kind": "gather"}), train_wordings("gather"), goal_of("gather", world, cells), check, {})


def draw_groups(rng):
    world, height, _ = draw_world(rng)
    m = 3 if rng.random() < 0.4 else 2
    n = robot_count(rng, 3 * m)
    places = S.GROUP_PAIRS[m][int(rng.integers(len(S.GROUP_PAIRS[m])))]
    sizes = np.full(m, n // m)
    sizes[: n % m] += 1
    free = ~blocked_columns(world)
    levels = np.zeros((SIZE, SIZE), int)
    cells, used = [], np.zeros((SIZE, SIZE), bool)
    for name, k in zip(places, sizes):
        part = pile(rng, height, int(k), region_centre(rng, name), free & ~chebyshev_grow(used, 2), levels)
        if part is None:
            return None
        cells += part
        used |= columns_of(part)
    words = {2: "two", 3: "three"}[m]

    def check(start, end):
        groups = components(end)
        if len(groups) != m or min(len(g) for g in groups) < 2:
            return "end: not exactly that many groups"
        if len(components(start)[0]) > n / 2:
            return "start: one group holds half"
        return ""

    return Draw("groups", record("groups", f"split into {words} separate groups", {"kind": "groups", "count": m}), train_wordings(f"groups.{words}"), goal_of("groups", world, cells), check, {"count": m, "places": list(places)})


def draw_spread(rng):
    world, height, _ = draw_world(rng)
    free = ~blocked_columns(world)
    for _ in range(8):
        n = robot_count(rng)
        cells = ground_layer(rng, height, n, free.copy(), apart=True)
        if cells is not None:
            break
    else:
        return None
    heaps = int(rng.integers(1, 4))
    sizes = np.full(heaps, n // heaps)
    sizes[: n % heaps] += 1
    start_cells, levels = [], np.zeros((SIZE, SIZE), int)
    for k in sizes:
        part = pile(rng, height, int(k), (rng.uniform(2, SIZE - 3), rng.uniform(2, SIZE - 3)), free, levels, max_level=3)
        if part is None:
            return None
        start_cells += part
    a = torch.zeros(SIZE, SIZE, SIZE, dtype=torch.bool)
    for z, y, x in start_cells:
        a[z, y, x] = True

    def check(start, end):
        if touching_any(end):
            return "end: two robots touch"
        if touching_any(start) < n / 2:
            return "start: not clumped"
        return ""

    return Draw("spread", record("spread", "spread out so that no two of you touch", {"kind": "spread"}), train_wordings("spread"), goal_of("spread", world, cells, start=a), check, {"heaps": heaps})


def draw_line(rng):
    world, height, _ = draw_world(rng)
    axis = str(rng.choice(["x", "y"]))
    n = S.draw_count(rng, 3, S.LINE_MAX, "log")
    free = ~blocked_columns(world)
    hgt = np.asarray(height)
    options = []
    for row in range(1, SIZE - 1):
        for start in range(0, SIZE - n + 1):
            ys, xs = (np.full(n, row), np.arange(start, start + n)) if axis == "x" else (np.arange(start, start + n), np.full(n, row))
            if free[ys, xs].all() and (hgt[ys, xs] == hgt[ys[0], xs[0]]).all() and hgt[ys[0], xs[0]] < SIZE:
                options.append((ys, xs))
    if not options:
        return None
    ys, xs = options[int(rng.integers(len(options)))]
    cells = [(int(hgt[y, x]), int(y), int(x)) for y, x in zip(ys, xs)]
    direction = "left to right" if axis == "x" else "back to front"
    run_axis = 2 if axis == "x" else 1

    def check(start, end):
        if longest_run(on_ground(end, hgt), run_axis) != n:
            return "end: not one row"
        if longest_run(start, 1) * 2 >= n or longest_run(start, 2) * 2 >= n:
            return "start: already half a row"
        return ""

    return Draw("line", record("line", f"line up in one row from {direction}", {"kind": "line", "axis": axis}), train_wordings(f"line.{axis}"), goal_of("line", world, cells), check, {"axis": axis})


def _restack(cells, height, shift):
    """cells moved by shift (dy, dx), each column's stack resting on the ground where it lands."""
    hgt = np.asarray(height)
    out = []
    for z, y, x in cells:
        y2, x2 = y + shift[0], x + shift[1]
        out.append((z - int(hgt[y, x]) + int(hgt[y2, x2]), y2, x2))
    return out


def draw_move(rng):
    world, height, _ = draw_world(rng)
    side = str(rng.choice(SIDES))
    axis, sign = S.SIDE_AXES[side]
    e = np.zeros(2, int)
    e[axis - 1] = sign
    back = -S.MOVE_STEPS * e  # where each robot started, relative to where it ends
    n = min(robot_count(rng), MOVE_MOST)
    free = ~blocked_columns(world)
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    y2, x2 = ys + back[0], xs + back[1]
    inside = (y2 >= 0) & (y2 < SIZE) & (x2 >= 0) & (x2 < SIZE)
    ok = free & inside
    ok[inside] &= free[y2[inside], x2[inside]]
    hgt = np.asarray(height)
    # the swarm stands in one window about twice its count in columns, so the shift shows
    span = min(int(np.ceil(np.sqrt(2.0 * n))) + int(rng.integers(0, 3)), SIZE - S.MOVE_STEPS)
    rows = np.nonzero(ok.any(axis=1))[0]
    cols = np.nonzero(ok.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return None
    y0 = int(rng.integers(rows.min(), max(rows.max() - span + 2, rows.min() + 1)))
    x0 = int(rng.integers(cols.min(), max(cols.max() - span + 2, cols.min() + 1)))
    window = np.zeros((SIZE, SIZE), bool)
    window[y0 : y0 + span, x0 : x0 + span] = True
    cells = scatter_stacked(rng, hgt, n, ok & window)
    if cells is None:
        return None
    # each column's stack rests on the ground: natural_scenes stacks from the ground, so the shift keeps it whole
    start_cells = _restack(cells, hgt, back)
    a = torch.zeros(SIZE, SIZE, SIZE, dtype=torch.bool)
    for z, y, x in start_cells:
        if not 0 <= z < SIZE:
            return None
        a[z, y, x] = True
    words = {"left": "left", "right": "right", "back": "back", "front": "front"}[side]

    def check(start, end_occ):
        moved = np.argwhere(np.asarray(end_occ))[:, 1:].mean(axis=0) - np.argwhere(np.asarray(start))[:, 1:].mean(axis=0)
        return "" if abs(float(moved @ e) - S.MOVE_STEPS) < 1e-6 else "end: not shifted"

    return Draw("move", record("move", f"move three steps toward the {words}", {"kind": "move", "toward": side, "steps": S.MOVE_STEPS}), train_wordings("move", r=side), goal_of("move", world, cells, start=a), check, {"toward": side})


def draw_stay(rng):
    world, height, _ = draw_world(rng)
    n = robot_count(rng)
    occ = R.natural_scenes(world.walls[0], n, rng, samples=1, avoid=W.cheese_id_map(world)[0] >= 0)[0]
    cells = [tuple(int(v) for v in c) for c in occ.nonzero().tolist()]

    def check(start, end):
        return "" if (np.asarray(start) == np.asarray(end)).all() else "moved"

    return Draw("stay", record("stay", "stay where you are", {"kind": "stay"}), train_wordings("stay"), goal_of("stay", world, cells, start=occ.clone()), check, {})


def draw_climb(rng):
    world, height, meta = draw_world(rng, layout="gap")
    sides = S.plateaus(meta)
    side = str(rng.choice(sorted(sides)))
    (y0, y1, x0, x1), top = sides[side]
    n = robot_count(rng)
    plateau = np.zeros((SIZE, SIZE), bool)
    plateau[y0:y1, x0:x1] = True
    free = ~blocked_columns(world)
    # the pile sits near the plateau's edge over the trench (the robots come up from there)
    axis, start, width, _ = S.trench(meta)
    edge = start - 1 if side in ("left", "back") else start + width
    along = rng.uniform(3, SIZE - 4)
    inward = -1.5 if side in ("left", "back") else 1.5
    centre = (along, edge + inward) if axis == 2 else (edge + inward, along)
    cells = pile(rng, height, n, centre, plateau & free)
    if cells is None:
        return None
    avoid = torch.from_numpy(np.broadcast_to(plateau, (SIZE, SIZE, SIZE)).copy()) | (W.cheese_id_map(world)[0] >= 0)
    a = R.natural_scenes(world.walls[0], n, rng, samples=1, avoid=avoid)[0]
    if int(a.sum()) != n:
        return None

    def check(start_occ, end):
        cells_end = np.argwhere(np.asarray(end))
        if not (plateau[cells_end[:, 1], cells_end[:, 2]] & (cells_end[:, 0] >= top)).all():
            return "end: not all on the plateau"
        if share_in(start_occ, plateau) > 0:
            return "start: some on the plateau"
        return ""

    return Draw("climb", record("climb", f"climb onto the plateau on the {side}", {"kind": "climb", "side": side}), train_wordings("climb", r=side), goal_of("climb", world, cells, start=a), check, {"side": side})


def _two_regions(rng):
    a, b = (str(v) for v in rng.choice(REGIONS, size=2, replace=False))
    return a, b


def draw_split_regions(rng):
    world, height, _ = draw_world(rng)
    r1, r2 = _two_regions(rng)
    n = robot_count(rng)
    free = ~blocked_columns(world)
    k1 = n // 2 + (n % 2) * int(rng.integers(2))
    first = pile(rng, height, k1, region_centre(rng, r1), region_columns(r1) & free)
    if first is None:
        return None
    second = pile(rng, height, n - k1, region_centre(rng, r2), region_columns(r2) & free & ~chebyshev_grow(columns_of(first), 2))
    if second is None:
        return None
    b1, b2 = region_columns(r1), region_columns(r2)
    words = [S.SUBTASK["region"].format(x=r) for r in (r1, r2)]

    def check(start, end):
        groups = components(end)
        if len(groups) != 2:
            return "end: not two groups"
        if not all(b1[g[:, 1], g[:, 2]].all() or b2[g[:, 1], g[:, 2]].all() for g in groups):
            return "end: a group outside both regions"
        if share_in(start, b1 | b2) > 0.4:
            return "start: already in the regions"
        return ""

    plain = f"split into two groups: one should {words[0]}, the other should {words[1]}"
    return Draw("split_regions", record("split_regions", plain, {"kind": "parallel", "jobs": [["region", r1], ["region", r2]]}), train_wordings("split", a=words[0], b=words[1]), goal_of("split_regions", world, first + second), check, {"regions": [r1, r2]})


def tower_cells(rng, height, h, side, allowed):
    """A tower h high, side x side columns, on level ground in allowed columns; None when none fits."""
    hgt = np.asarray(height)
    spots = []
    for y in range(1, SIZE - side):
        for x in range(1, SIZE - side):
            foot = hgt[y : y + side, x : x + side]
            if allowed[y : y + side, x : x + side].all() and (foot == foot[0, 0]).all() and foot[0, 0] + h <= SIZE:
                spots.append((y, x))
    if not spots:
        return None
    y, x = spots[int(rng.integers(len(spots)))]
    z0 = int(hgt[y, x])
    return [(z, y + a, x + b) for z in range(z0, z0 + h) for a in range(side) for b in range(side)]


def draw_split_tower(rng):
    world, height, _ = draw_world(rng)
    r = str(rng.choice(REGIONS))
    h = int(rng.integers(3, 7))
    n = robot_count(rng, h + 3)
    side = 2 if n - 4 * h >= 3 and rng.random() < 0.5 else 1
    free = ~blocked_columns(world)
    box = region_columns(r)
    tower = tower_cells(rng, height, h, side, free & ~chebyshev_grow(box, 2))
    if tower is None:
        return None
    group = pile(rng, height, n - len(tower), region_centre(rng, r), box & free & ~chebyshev_grow(columns_of(tower), 2))
    if group is None:
        return None
    tower_cols = columns_of(tower)
    words = [S.SUBTASK["region"].format(x=r), S.SUBTASK["tower"].format(x=h)]

    def check(start, end):
        end = np.asarray(end)
        rest = end.copy()
        rest[:, tower_cols] = False
        if tallest(end) != h or not end[:, tower_cols].any(axis=0).all():
            return "end: no tower that high"
        if share_in(rest, box) < 1.0 or tallest(rest) >= h:
            return "end: the others not all in the region"
        if tallest(start) >= h:
            return "start: a stack already that high"
        return ""

    plain = f"split into two groups: one should {words[0]}, the other should {words[1]}"
    return Draw("split_tower", record("split_tower", plain, {"kind": "parallel", "jobs": [["region", r], ["tower", h]]}), train_wordings("split", a=words[0], b=words[1]), goal_of("split_tower", world, group + tower), check, {"region": r, "height": h, "side": side})


def draw_tower(rng):
    world, height, _ = draw_world(rng)
    h = int(rng.integers(3, 9))
    n = robot_count(rng, h)
    side = 2 if n >= 4 * h and rng.random() < 0.5 else 1
    free = ~blocked_columns(world)
    tower = tower_cells(rng, height, h, side, free)
    if tower is None:
        return None
    tower_cols = columns_of(tower)
    spares = ground_layer(rng, height, n - len(tower), free & ~chebyshev_grow(tower_cols, 2))
    if spares is None:
        return None

    def check(start, end):
        end = np.asarray(end)
        rest = end.copy()
        rest[:, tower_cols] = False
        if tallest(end) != h or tallest(rest) > 1:
            return "end: not one tower and the rest on the ground"
        if tallest(start) >= h:
            return "start: a stack already that high"
        return ""

    return Draw("tower", record("tower", f"build a tower {h} robots tall", {"kind": "tower", "height": h}), train_wordings("tower", n=h), goal_of("tower", world, tower + spares), check, {"height": h, "side": side})


def draw_bridge(rng):
    world, height, meta = draw_world(rng, layout="gap")
    axis, start, width, top = S.trench(meta)
    lanes = 2 if rng.random() < 0.5 else 1
    need = width * top * lanes
    n = robot_count(rng, need)
    lane = int(rng.integers(2, SIZE - 2 - lanes))
    cells = []
    for k in range(start, start + width):
        for a in range(lanes):
            y, x = (lane + a, k) if axis == 2 else (k, lane + a)
            cells += [(z, y, x) for z in range(top)]
    free = ~blocked_columns(world)
    deck = columns_of(cells)
    plateaus = np.asarray(height) > 0  # the others wait on the plateaus: the trench holds only the crossing
    spares = ground_layer(rng, height, n - need, free & plateaus & ~chebyshev_grow(deck, 2))
    if spares is None:
        return None
    hgt = np.asarray(height)

    def check(start_occ, end):
        end = np.asarray(end)
        if not all(end[z, y, x] for z, y, x in cells):
            return "end: the crossing is not solid"
        if (np.asarray(start_occ)[:, deck].sum() * 2 >= need):
            return "start: half the crossing there"
        rest = end.copy()
        rest[:, deck] = False
        if (rest & ~on_ground(rest, hgt)).any():
            return "end: others not on the ground"
        return ""

    return Draw("bridge", record("bridge", "build a bridge across the trench", {"kind": "bridge"}), train_wordings("bridge"), goal_of("bridge", world, cells + spares), check, {"lanes": lanes, "width": width, "depth": top})


def _paint(rng, world, height, kind, colour, others=(0, 2)):
    """Paint a floor zone of `kind` (patch 3-4 a side, or a strip 1 x 6-10) of `colour` on level
    ground away from blocks, then 0-2 others of other colours (either kind). Returns its box
    (y0, y1, x0, x1) or None."""
    hgt = np.asarray(height)
    taken = blocked_columns(world).copy()
    box = None
    for _ in range(60):
        if kind == "patch":
            sy, sx = int(rng.integers(3, 5)), int(rng.integers(3, 5))
        elif rng.random() < 0.5:
            sy, sx = 1, int(rng.integers(6, 11))
        else:
            sy, sx = int(rng.integers(6, 11)), 1
        y0, x0 = int(rng.integers(0, SIZE - sy + 1)), int(rng.integers(0, SIZE - sx + 1))
        cells = hgt[y0 : y0 + sy, x0 : x0 + sx]
        if (cells != cells[0, 0]).any() or taken[max(y0 - 1, 0) : y0 + sy + 1, max(x0 - 1, 0) : x0 + sx + 1].any():
            continue
        box = (y0, y0 + sy, x0, x0 + sx)
        break
    if box is None:
        return None
    y0, y1, x0, x1 = box
    world.zones[0, y0:y1, x0:x1] = COLOUR_NAMES.index(colour)
    taken[y0:y1, x0:x1] = True
    rest = [COLOUR_NAMES.index(c) for c in COLOURS if c != colour]
    for mark in W.place_zones(rng, height, int(rng.integers(others[0], others[1] + 1)), rest, torch.from_numpy(taken)):
        a0, a1, b0, b1 = mark["box"]
        world.zones[0, a0:a1, b0:b1] = mark["colour"]
    return box


def draw_zone(rng):
    world, height, _ = draw_world(rng)
    colour = str(rng.choice(COLOURS))
    box = _paint(rng, world, height, "patch", colour)
    if box is None:
        return None
    y0, y1, x0, x1 = box
    patch = np.zeros((SIZE, SIZE), bool)
    patch[y0:y1, x0:x1] = True
    for _ in range(16):
        n = robot_count(rng)
        if n <= 2 * int(patch.sum()):
            break
    else:
        n = 2 * int(patch.sum())
    cells = pile(rng, height, n, ((y0 + y1 - 1) / 2, (x0 + x1 - 1) / 2), patch, max_level=2, layer_cost=50.0)
    if cells is None:
        return None
    patches = ["go to the {c} patch on the floor", "head for the {c} patch", "everyone onto the {c} patch", "go stand on the {c} square on the floor", "get over to the {c} patch"]

    def check(start, end):
        if share_in(end, patch) < 1.0:
            return "end: robots off the patch"
        if share_in(start, patch) > 0.2:
            return "start: a fifth already on it"
        return ""

    return Draw("zone", record("zone", f"go to the {colour} patch on the floor", {"kind": "goto_zone", "zone": colour}), [p.format(c=colour) for p in patches], goal_of("zone", world, cells), check, {"colour": colour, "cells": int(patch.sum())})


def draw_zone_line(rng):
    world, height, _ = draw_world(rng)
    colour = str(rng.choice(COLOURS))
    box = _paint(rng, world, height, "strip", colour)
    if box is None:
        return None
    y0, y1, x0, x1 = box
    length = max(y1 - y0, x1 - x0)
    n = S.draw_count(rng, 3, length, "log")
    first = int(rng.integers(0, length - n + 1))
    hgt = np.asarray(height)
    strip = np.zeros((SIZE, SIZE), bool)
    strip[y0:y1, x0:x1] = True
    if y1 - y0 == 1:
        cells = [(int(hgt[y0, x]), y0, x) for x in range(x0 + first, x0 + first + n)]
    else:
        cells = [(int(hgt[y, x0]), y, x0) for y in range(y0 + first, y0 + first + n)]
    lines = ["line up along the {c} strip on the floor", "line up on the {c} strip", "stand in a row on the {c} strip", "get in line on the {c} strip", "form a line along the {c} strip on the floor"]

    def check(start, end):
        on = on_ground(end, hgt)
        if share_in(end, strip) < 1.0 or int(on.sum()) != n:
            return "end: robots off the strip"
        if share_in(start, strip) > 0.2:
            return "start: a fifth already on it"
        return ""

    return Draw("zone_line", record("zone_line", f"line up along the {colour} strip on the floor", {"kind": "line_zone", "zone": colour}), [p.format(c=colour) for p in lines], goal_of("zone_line", world, cells), check, {"colour": colour, "length": int(length)})


def draw_push(rng):
    goal = R.make_goal("push", rng)
    text = goal.text
    zone = goal.world.zones[0]

    def check(start, end):
        return ""

    spec = {"kind": "push", "shape": goal.info.get("shape"), "pattern": goal.info.get("pattern"), "zone": COLOUR_NAMES[int(zone[zone >= 0][0])] if (zone >= 0).any() else None}
    rec = {"family": "objects", "kind": "push", "plain": text, "spec": spec, "subgoals": [text]}
    return Draw("push", rec, [text], goal, check, dict(goal.info, length=goal.push["length"]))


BUILDERS = {
    "goto": draw_goto, "gather": draw_gather, "groups": draw_groups, "spread": draw_spread, "line": draw_line, "move": draw_move,
    "stay": draw_stay, "climb": draw_climb, "split_regions": draw_split_regions, "split_tower": draw_split_tower,
    "tower": draw_tower, "bridge": draw_bridge, "zone": draw_zone, "zone_line": draw_zone_line, "push": draw_push,
}


# ---------------------------------------------------------------------------------------------
# One episode


def summary_of(world):
    """think_data's scene summary of a one-scene world, JSON-safe (numbers as Python numbers)."""
    return json.loads(json.dumps(S.scene_summary(world, 0), default=lambda v: v.item() if hasattr(v, "item") else str(v)))


def pack(ep, draw, seed, tries, cpu):
    """An episode as compact arrays (bit-packed frames, codes on robot cells only), with what the
    words need: the record, the template wordings, the start's scene summary."""
    mass = (ep.mass > 0).reshape(ep.mass.shape[0], -1).numpy()
    codes, offsets = [], [0]
    for t in range(ep.steps):
        here = mass[t].nonzero()[0]
        codes.append(ep.actions[t].reshape(-1).numpy()[here].astype(np.int8))
        offsets.append(offsets[-1] + len(here))
    start = replace(ep.world, mass=ep.mass[0].float().unsqueeze(0), cheese_pos=ep.cheese_pos[0].unsqueeze(0), cheese_alive=ep.cheese_alive[0].unsqueeze(0))
    meta = {
        "family": draw.family, "record": draw.record, "templates": list(draw.templates), "info": draw.info, "seed": int(seed),
        "robots": int(mass[0].sum()), "steps": int(ep.steps), "end": ep.end, "tries": int(tries), "cpu": float(cpu),
        "summary": summary_of(start), "moving": float(np.mean([(ep.actions[t] != STAY).sum().item() for t in range(ep.steps)])) if ep.steps else 0.0,
    }
    return {"meta": meta, "mass": np.packbits(mass, axis=1), "codes": np.concatenate(codes) if codes else np.zeros(0, np.int8), "offsets": np.asarray(offsets, np.int64), "cheese_pos": ep.cheese_pos.clone(), "cheese_alive": ep.cheese_alive.clone(), "world": ep.world}


class LiveEpisode:
    """One packed episode: frames, codes and observations at any step (world frame). Step `steps`
    is the goal with every robot staying."""

    def __init__(self, row):
        self.row, self.meta = row, row["meta"]
        self.steps = int(self.meta["steps"])

    def occupancy(self, t):
        return np.unpackbits(self.row["mass"][t])[: SIZE**3].reshape(SIZE, SIZE, SIZE).astype(bool)

    def codes(self, t):
        out = np.full(SIZE**3, STAY, np.int64)
        if t < self.steps:
            here = np.unpackbits(self.row["mass"][t])[: SIZE**3].nonzero()[0]
            o = self.row["offsets"]
            out[here] = self.row["codes"][o[t] : o[t + 1]]
        return torch.from_numpy(out.reshape(SIZE, SIZE, SIZE))

    def world(self, t):
        mass = torch.from_numpy(self.occupancy(t)).float().unsqueeze(0)
        return replace(self.row["world"], mass=mass, cheese_pos=self.row["cheese_pos"][t].unsqueeze(0), cheese_alive=self.row["cheese_alive"][t].unsqueeze(0))

    def observation(self, t, channels=N_CHANNELS):
        return W.observe(self.world(t), channels)[0]

    def as_reverse(self):
        """The episode as a reverse.Episode (frames, codes, start and goal worlds) for reverse.replay."""
        frames = torch.stack([torch.from_numpy(self.occupancy(t)).to(torch.uint8) for t in range(self.steps + 1)])
        actions = torch.stack([self.codes(t).to(torch.int8) for t in range(self.steps)]) if self.steps else torch.zeros(0, SIZE, SIZE, SIZE, dtype=torch.int8)
        goal = R.Goal(self.meta["family"], "", self.world(self.steps), frames[-1].bool())
        return R.Episode(family=self.meta["family"], text="", world=self.world(0), mass=frames, cheese_pos=self.row["cheese_pos"], cheese_alive=self.row["cheese_alive"], actions=actions, natural=True, end=self.meta["end"], goal=goal)


def generate(seed, families=FAMILIES, weights=None, tries=TRIES, replay=False, max_steps=None, robots=None):
    """One live episode (pack's dict) of a family drawn from `families` (by `weights`), or None
    when `tries` draws give none that reaches its start and passes its check. Every step is
    verified by take_apart; replay=True also plays it forward with world.step (slower; the tests
    and measurements do); robots (low, high) narrows the robot count's draw. Also returns
    {"family", "tries", "fails": {reason: count}, "cpu"}."""
    if robots is not None:
        saved = list(_RANGE)
        _RANGE[:] = robots
        try:
            return generate(seed, families, weights, tries, replay, max_steps)
        finally:
            _RANGE[:] = saved
    rng = np.random.default_rng(seed)
    families = list(families)
    p = None if weights is None else np.asarray([weights.get(f, 0.0) for f in families], float) / sum(weights.get(f, 0.0) for f in families)
    family = families[int(rng.choice(len(families), p=p))]
    fails = {}
    t0 = time.process_time()
    for k in range(tries):
        draw = BUILDERS[family](rng)
        if draw is None:
            fails["no end state"] = fails.get("no end state", 0) + 1
            continue
        if not bool(R.fixed_point(draw.goal.world).all()):
            fails["end not at rest"] = fails.get("end not at rest", 0) + 1
            continue
        ep = R.take_apart([draw.goal], rng, max_steps=max_steps)[0][0]
        if not ep.natural:
            fails[f"walk {ep.end}"] = fails.get(f"walk {ep.end}", 0) + 1
            continue
        why = draw.check(ep.mass[0].numpy() > 0, ep.mass[-1].numpy() > 0) if draw.family != "push" else ""
        if why:
            fails[why] = fails.get(why, 0) + 1
            continue
        cpu = time.process_time() - t0  # the walk and its checks (a replay below is not counted)
        if replay and not bool(R.replay([ep])["passed"][0]):
            fails["replay"] = fails.get("replay", 0) + 1
            continue
        return pack(ep, draw, seed, k + 1, cpu), {"family": family, "tries": k + 1, "fails": fails, "cpu": cpu}
    return None, {"family": family, "tries": tries, "fails": fails, "cpu": time.process_time() - t0}


# ---------------------------------------------------------------------------------------------
# Words


def numbers_in_text(text):
    """The numbers a text states (digits and number words up to the hundreds)."""
    out = set()
    for m in T.NUMBER_RE.finditer(text.lower()):
        value = T.number_value(m.group(0))
        if value is not None:
            out.add(int(value))
    for word, value in T.UNITS.items():
        if re.search(rf"\b{word}\b", text.lower()):
            out.add(value)
    out |= {int(v) for v in re.findall(r"\b\d+\b", text)}
    return out


def spec_numbers(spec):
    return {int(v) for v in T.numbers_in(spec)}


def allowed_words(rec):
    """Area words and colours an instruction or thought of this record may use: those of its
    plain wording (left and right for a row from left to right, the named region or side), with
    "corner" for a corner region, "side" and "edge" for any side, "middle" for the center."""
    plain = rec["plain"].lower()
    areas = {w for w in AREA_WORDS if re.search(rf"\b{w}\b", plain)}
    if "center" in areas:
        areas |= {"centre", "middle"}
    named = {"left", "right", "front", "back"} & areas
    if named:
        areas |= {"side", "sides", "edge", "edges"}
        if rec["spec"].get("kind") in ("goto", "parallel") and any(len(r.split()) == 2 for r in _regions(rec["spec"])):
            areas |= {"corner", "corners"}
    colours = {c for c in COLOUR_NAMES if re.search(rf"\b{c}\b", plain)}
    return areas, colours


def _regions(spec):
    if spec.get("kind") == "goto":
        return [spec["region"]]
    return [x for job, x in spec.get("jobs", []) if job == "region"]


def allowed_numbers(spec):
    """Numbers a text may state: the spec's, one, and two where "two" is part of the task's own
    words (no two touch, two groups or plateaus, both sides of a trench)."""
    out = spec_numbers(spec) | {1}
    if spec.get("kind") in ("spread", "parallel", "bridge", "climb") or (spec.get("kind") == "groups" and spec.get("count") == 2):
        out.add(2)
    return out


def text_ok(text, rec, scene=False):
    """Whether text says nothing a live scene of this record may lack: no colour or area word the
    record does not name, no number the spec does not hold, none of the simulator's own nouns
    (cheese, zone, stripe); with scene=True (a thought's meaning sentence) also no word that
    describes the particular scene Haiku saw (SCENE_WORDS)."""
    low = text.lower()
    if INTERNAL.search(low) or COORDINATES.search(low):
        return False
    areas, colours = allowed_words(rec)
    if any(re.search(rf"\b{c}\b", low) for c in COLOUR_NAMES if c not in colours):
        return False
    if any(re.search(rf"\b{w}\b", low) for w in AREA_WORDS if w not in areas):
        return False
    if any(english not in areas and re.search(rf"(?<!\w){word}(?!\w)", low) for word, english in FOREIGN_AREAS.items()):
        return False
    if not numbers_in_text(low) <= allowed_numbers(rec["spec"]):
        return False
    if re.search(r"\b(?:all|every|the whole|each)\s+(?:of\s+)?(?:the\s+)?(?:\d+|" + "|".join(T.UNITS) + r")\b", low):
        return False
    if scene and any(m.group(0) not in rec["plain"].lower() for m in SCENE_WORDS.finditer(low)):  # a word of the task's own (a tower's "tower") stays
        return False
    if rec["spec"].get("kind") not in ("bridge", "climb") and re.search(r"\b(?:trench|plateaus?|ditch|gap|cliff|hills?)\b", low):
        return False
    return True


def plainly(text):
    """Floor zones in plain words: zone -> patch, stripe -> strip."""
    text = re.sub(r"\bzones\b", "patches", text)
    text = re.sub(r"\bzone\b", "patch", text)
    return re.sub(r"\bstripe(s?)\b", r"strip\1", text)


def splice(choice, rec, summary):
    """A Haiku thought for a live scene: its quote and, when they pass text_ok, its meaning
    sentences and first sub-goal; think_data.sight of the live scene in place of what Haiku saw;
    the template's meaning or first sub-goal where Haiku's do not pass."""
    thought, instruction = choice["thought"], choice["instruction"]
    q = T.quote_length(thought, instruction)
    head = f"you asked: {T.bare(instruction)}."  # the quote is the instruction itself (Haiku's quote sometimes shortens it)
    body = T.sentences_of(thought[q:])
    middle = [s for s in body if not s.lower().startswith(("i see", "first:"))]
    first = [s for s in body if s.lower().startswith("first:")]
    meaning = [s for s in middle if text_ok(s, rec, scene=True)]
    if len(meaning) != len(middle) or not meaning:
        meaning = [f"that means: {T.bare(rec['plain'])}."]
    first_line = first[0] if first and text_ok(first[0], rec) else f"first: {T.bare(rec['subgoals'][0])}."
    sight = T.sight(summary, rec)
    return " ".join([head, f"i see {sight}.", *meaning, first_line]).lower()


def haiku_index(dirs=HAIKU_DIRS, root=None):
    """{canonical spec: [Haiku choice]} of verified rewordings (close, indirect, casual) of the
    specs the live families draw, the switch episodes' parts included; rows with a subset ("who")
    are left out. Missing directories are skipped."""
    root = Path(root) if root else Path(__file__).resolve().parents[4]
    kinds = {"goto", "gather", "groups", "spread", "line", "move", "stay", "climb", "parallel", "tower", "bridge", "goto_zone", "line_zone"}
    out = {}
    for d in dirs:
        path = Path(d) if Path(d).is_absolute() else root / d
        if not (path / "verified.jsonl").exists():
            continue
        for line in open(path / "verified.jsonl"):
            row = json.loads(line)
            spec = row.get("spec") or {}
            if row.get("rung") not in RUNGS or "who" in spec or spec.get("kind") not in kinds:
                continue
            if spec["kind"] == "parallel" and not all(job[0] in ("region", "tower") for job in spec["jobs"]):
                continue
            out.setdefault(T.canonical_json(spec), []).append({"rung": row["rung"], "instruction": row["instruction"], "thought": row["thought"], "pair_id": row.get("pair_id")})
    return out


class Texts:
    """Words for live episodes: Haiku rewordings of the same spec that pass text_ok, else the
    template. counts: per family, Haiku texts available, kept and dropped (and why)."""

    def __init__(self, dirs=HAIKU_DIRS, root=None, template_share=TEMPLATE_SHARE):
        self.index = haiku_index(dirs, root)
        self.template_share = template_share
        self.kept = {}
        self.stats = {}

    def choices(self, rec, family=None):
        key = T.canonical_json(rec["spec"])
        if key not in self.kept:
            rows = self.index.get(key, [])
            good = [r for r in rows if text_ok(r["instruction"], rec)]
            self.kept[key] = good
            fam = family or rec["kind"]
            row = self.stats.setdefault(fam, {"specs": 0, "haiku": 0, "kept": 0, "internal_nouns": 0})
            row["specs"] += 1
            row["haiku"] += len(rows)
            row["kept"] += len(good)
            row["internal_nouns"] += sum(bool(INTERNAL.search(r["instruction"])) for r in rows)
        return self.kept[key]

    def words(self, meta, rng):
        """(instruction, thought, rung) for an episode's meta, drawn with rng."""
        rec, summary = meta["record"], meta["summary"]
        good = self.choices(rec, meta.get("family"))
        if good and rng.random() >= self.template_share:
            choice = good[int(rng.integers(len(good)))]
            instruction = choice["instruction"]
            thought = splice(choice, rec, summary)
            rung = choice["rung"]
        else:
            templates = meta["templates"]
            instruction = templates[int(rng.integers(len(templates)))]
            thought = T.template_thought(instruction, rec, summary)
            rung = T.TEMPLATE
        thought = plainly(T.visible(thought, instruction, rec))
        return instruction, thought, rung


# ---------------------------------------------------------------------------------------------
# Examples for ThinkData


@dataclass
class LiveExample(T.Example):
    live: str = ""  # the live family
    robots: int = 0
    steps: int = 0


def _worker_init(nice):
    torch.set_num_threads(1)
    if nice:
        try:
            os.nice(nice)
        except OSError:
            pass


def _generate_job(args):
    """generate in a worker; the episode comes back as pickled bytes (tensors by value, not through
    torch's shared-memory file handles, which a long run would pile up)."""
    seed, families, weights, max_steps, robots = args
    row, info = generate(seed, families, weights, max_steps=max_steps, robots=robots)
    return (pickle.dumps(row) if row is not None else None), info


class Live:
    """Live reverse examples (module docstring) for ThinkData's `live=`.

    share     share of all examples that are live (drawn first, one rng draw when set)
    families  live families to draw (default all); weights {family: weight} (default equal)
    workers   processes generating episodes in the background (0: in the caller)
    cache     episodes kept to draw from (the latest); reuse: with workers=0, a draw makes a new
              episode with probability 1 / reuse (else it reuses one of the cache)
    held      episodes of the fixed validation set (generated from held_seed on first use)
    robots    (low, high): the robot counts drawn (default the stores' 6-115, log-uniform)
    """

    def __init__(self, data=None, share=0.1, families=FAMILIES, weights=None, workers=0, cache=64, reuse=8, held=16, held_seed=10**6, thought_share=1 / 3, haiku_dirs=HAIKU_DIRS, root=None, template_share=TEMPLATE_SHARE, channels=None, system=None, nice=10, max_steps=None, robots=None):
        unknown = set(families) - set(FAMILIES)
        if unknown:
            raise ValueError(f"unknown live families {sorted(unknown)}; families are {FAMILIES}")
        if not 0 <= share <= 1:
            raise ValueError(f"live share {share} not in [0, 1]")
        self.data = data
        self.share = float(share)
        self.families = tuple(families)
        self.weights = weights
        self.workers = int(workers)
        self.cache = deque(maxlen=int(cache))
        self.reuse = max(int(reuse), 1)
        self.held_count, self.held_seed = int(held), int(held_seed)
        self.held = None
        self.thought_share = thought_share
        self.texts = Texts(haiku_dirs, root, template_share)
        self.channels = channels or (data.channels if data is not None else N_CHANNELS)
        self.system = system if system is not None else (data.system if data is not None else None)
        self.nice = nice
        self.max_steps = max_steps
        self.robots = tuple(robots) if robots else None
        self.counts = {}
        self.generated, self.drawn = 0, 0
        self.failed = {}
        self.pool = None
        self.lock = threading.Lock()
        self.arrived = threading.Condition(self.lock)
        self.outstanding = 0
        self.closed = False
        self.feeder = None
        self.seeds = None

    # -- episodes

    def _add(self, result):
        row, info = result
        with self.arrived:
            self.outstanding -= 1
            if row is not None:
                self.cache.append(LiveEpisode(pickle.loads(row)))
                self.generated += 1
            else:
                key = info["family"]
                self.failed[key] = self.failed.get(key, 0) + 1
            self.arrived.notify_all()

    def _feed(self):
        while not self.closed:
            with self.lock:
                room = self.outstanding < 2 * self.workers
            if room:
                with self.lock:
                    self.outstanding += 1
                seed = int(self.seeds.integers(2**62))
                self.pool.apply_async(_generate_job, ((seed, self.families, self.weights, self.max_steps, self.robots),), callback=self._add, error_callback=lambda e: self._add((None, {"family": f"error: {e!r}"[:80]})))
            else:
                time.sleep(0.05)

    def start(self, seed=None):
        """Start the background workers (sample starts them on first use)."""
        if self.workers <= 0 or self.pool is not None:
            return
        import multiprocessing

        self.seeds = np.random.default_rng(seed)
        self.pool = multiprocessing.get_context("spawn").Pool(self.workers, initializer=_worker_init, initargs=(self.nice,))
        self.feeder = threading.Thread(target=self._feed, daemon=True)
        self.feeder.start()

    def close(self):
        self.closed = True
        if self.pool is not None:
            self.pool.terminate()
            self.pool = None

    def _make(self, rng):
        for _ in range(4):
            row, info = generate(int(rng.integers(2**62)), self.families, self.weights, max_steps=self.max_steps, robots=self.robots)
            if row is not None:
                self.generated += 1
                return LiveEpisode(row)
            self.failed[info["family"]] = self.failed.get(info["family"], 0) + 1
        raise RuntimeError(f"no live episode in 4 draws ({self.failed})")

    def episode(self, rng):
        """A training episode: from the cache, which the workers keep fresh (with workers=0, a new
        one with probability 1 / reuse)."""
        if self.workers <= 0:
            if not self.cache or rng.random() < 1.0 / self.reuse:
                self.cache.append(self._make(rng))
            return self.cache[int(rng.integers(len(self.cache)))]
        self.start(int(rng.integers(2**62)))
        with self.arrived:
            while not self.cache:
                self.arrived.wait(timeout=1.0)
            return self.cache[int(rng.integers(len(self.cache)))]

    def held_episodes(self):
        """The fixed validation episodes (held_count of them, from held_seed)."""
        if self.held is None:
            rng = np.random.default_rng(self.held_seed)
            jobs = [(int(rng.integers(2**62)), self.families, self.weights, self.max_steps, self.robots) for _ in range(self.held_count * 2)]
            if self.workers > 0:  # the same episodes as in the caller, made by the workers
                self.start(self.held_seed)
                results = self.pool.imap(_generate_job, jobs)
            else:
                results = map(_generate_job, jobs)
            self.held = []
            for row, _ in results:  # in seed order: the same set on every run
                if row is not None:
                    self.held.append(LiveEpisode(pickle.loads(row)))
                if len(self.held) == self.held_count:
                    break
        return self.held

    # -- examples

    def maybe(self, rng, held=False, views=None):
        """A live example with probability share (one rng draw), else None."""
        if not self.share or rng.random() >= self.share:
            return None
        return self.sample(rng, held, views)

    def sample(self, rng, held=False, views=None):
        if held:
            pool = self.held_episodes()
            ep = pool[int(rng.integers(len(pool)))]
        else:
            ep = self.episode(rng)
        return self.example(ep, rng, views or self._views(held))

    def _views(self, held):
        if self.data is None:
            return T.stand_in_views
        return (self.data.validation_views or self.data.views) if held else self.data.views

    def example(self, ep, rng, views):
        """An act example of episode ep: step 0 (it carries the first thought) with thought_share,
        else a step from 1 to steps (steps: the goal, every robot staying)."""
        t = 0 if ep.steps == 0 or rng.random() < self.thought_share else int(rng.integers(1, ep.steps + 1))
        instruction, thought, rung = self.texts.words(ep.meta, rng)
        obs = ep.observation(t, self.channels)
        sayable = T.mappable(instruction) and T.mappable(thought)
        if sayable:
            images, azimuth = T.views_and_pose(views(obs, rng))
        elif T.takes_side(views):
            images, azimuth = T.views_and_pose(views(obs, rng, side="front"))
        else:  # words that cannot be turned and views that cannot be asked for the front: the template, which turns
            instruction = ep.meta["templates"][int(rng.integers(len(ep.meta["templates"])))]
            thought = plainly(T.visible(T.template_thought(instruction, ep.meta["record"], ep.meta["summary"]), instruction, ep.meta["record"]))
            rung = T.TEMPLATE
            images, azimuth = T.views_and_pose(views(obs, rng))
        turns = quarter_turns(azimuth)
        said = T.in_user_frame(instruction, turns)
        thought = T.in_user_frame(thought, turns)
        memory = [thought] if t > 0 else []
        family = ep.meta["family"]
        self.drawn += 1
        self.counts[family] = self.counts.get(family, 0) + 1
        return LiveExample(
            obs=turn_grid(obs, turns),
            images=images,
            context_text=T.context(said, memory, self.system),
            reply_text=T.act_reply(thought if t == 0 else ""),
            act=True,
            codes=turn_codes(ep.codes(t), turns),
            seed=-1,
            index=int(hashlib.sha256(str(ep.meta["seed"]).encode()).hexdigest()[:8], 16),
            e=-1,
            t=t,
            rung=rung,
            family=f"{RUNG}:{family}",
            kind="first" if t == 0 else "plain",
            instruction=said,
            store=RUNG,
            turns=turns,
            camera=azimuth,
            live=family,
            robots=int(ep.meta["robots"]),
            steps=ep.steps,
        )

    def summary(self):
        return {"drawn": dict(self.counts), "generated": self.generated, "examples": self.drawn, "cached": len(self.cache), "failed": dict(self.failed), "texts": dict(self.texts.stats), "examples_per_episode": self.drawn / max(self.generated, 1)}
