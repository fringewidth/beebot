"""Demonstrations generated backwards from goal states (runs/handoff/reverse-demos.md, reverse-v2.md,
reverse-v3.md, reverse-v4.md).

A goal generator scripts an ordered end state (a tower, a crossing, a letter, an object on a patch,
a span over open air, an arch, a table, a chair, an L, big shapes; or any library goal of 4-800
robots) with an instruction in a stranger's words. take_apart then walks backwards from it: from
the current state S' it proposes earlier states S with the joint action codes a that would take S
to S', and keeps a proposal only if the real forward step does exactly that:

    world.step(S, a, rule=RULE) == S'      (every robot and object where S' has it)
    world.refuse(S, a, RULE) == a          (the rule lets every recorded move happen; rule_ok)
    world.fall(S) == S                     (S is at rest: nothing in it is falling)

so every recorded step is forward-causal under the world's own rules (pivots and their supports,
the overhang rule, pushes, gravity, eating). A step whose forward version would make anything
fall, eat anything or refuse a move never verifies. The rule is deterministic here.

Two modes, chosen per goal:

1. natural (Goal.start is None): the robots on the goal's cells walk out, top down, as a swarm
   (_swarm: many at once, WALKER_SPACING apart when they set off, each step of all of them verified
   jointly) to drop cells far from the goal, then on the ground near it, then in heaps up to
   heap_cap high, never under the goal's cells. Once the goal is clear, robots that keep the
   start out of band walk to emptier cells (spread), then a greedy walk climbs a potential: per
   robot, a bonus for being off the goal's cells, minus its geodesic distance (cube moves over
   cells a cube can hold on to) to the floor or terrain, plus distance from the goal; minus
   penalties per pair of robots face to face and per pair within two columns. It stops when every
   statistic of start_stats lies inside the band that sample_scenes-style scenes with the same
   terrain, robot count and goal give (natural_bands: 2.5-97.5 % quantiles of 32 drawn scenes).
   Dense goals (heap_cap above HEAP: 500-800 robots on 16 x 16) plan the start first: a natural
   scene's robots per column (_level_plan); the goal's robots already where it has robots stay,
   the others walk to its cells, the lowest layer first, closing in no cell still to fill
   (_plan_rank, _profile_rank); once the goal is clear the rest walk there too (_level_rank).
1. guided (Goal.start = A, a (D, H, W) bool occupancy at rest with as many robots): the robots off
   A's cells walk, as a swarm, to A's empty cells, lowest layer first and in it the most enclosed
   first, a cell that would close in a neighbour still to fill last (_fill_rank). When nothing can
   move, robots on A's cells round the stuck spot are taken out top down and put back bottom up
   (_dig_region, a local take apart and rebuild). It stops when the state is exactly A
   ("reached"). A = B needs no walk: a hold episode of robots staying still (HOLD_STEPS).
   Above REBUILD_ROBOTS robots, _rebuild first: the robots A and B share that hold themselves up
   stay; B's others walk out to a heap (reverse time), A's others walk out of A forward in time to
   the same heap cells in the same order (_swarm forward), and a short guided walk joins the two;
   forward the changed part comes apart onto the heap and the new one goes up from it.

Moves are proposed from a 7^3 crop round each robot (sources: which cells could have pivoted a
cube into it). Walks are breadth first over those moves with the rest of the scene still
(_reverse_path); the swarm re-verifies every joint step and replans walkers that cannot move.

Scaffolding (take-apart orders that need temporary support, e.g. the last robot of a span over
air): when no robot that must move has any legal move, _scaffold looks for a pivot into one of
them that only lacks one or two support cells, fills those cells with helper robots that walk
there (reverse time; breadth-first search over single-robot moves, every step verified), then
takes the stuck robot out over them. Played forward: helpers walk in, hold the part up while it
is placed, and leave.

An episode (Episode) is the walk played forward: frames from the start to the goal, the codes
recorded on each frame (per voxel, STAY where no robot is), the instruction. replay() checks every
step with world.step; to_store() lays episodes out as the dense fields lm.world_at reads.

    from beebot.vla.voxel import reverse
    rng = np.random.default_rng(0)
    goals = [reverse.make_goal("table", rng) for _ in range(16)]
    episodes, stats = reverse.take_apart(goals, rng)
    report = reverse.replay(episodes)  # report["passed"]: (E,) bool
"""

import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from beebot.vla.voxel import world as W
from beebot.vla.voxel.world import CUBE_MOVES, DIR_TABLE, DIRS, FRICTION, MAX_OBJECTS, SIZE, STAY, VoxelWorld

RULE = "switch_cube"  # the demo's rule (demos.PRESETS["demo"]); gravity is on in every scene
FAMILIES = ("tower", "bridge", "letter", "push")  # the first four (reverse-demos.md)
NEW_FAMILIES = ("span", "arch", "table", "chair", "cantilever", "big")
GUIDED_FAMILIES = ("edit", "retarget")  # walked toward a given start A
ALL_FAMILIES = FAMILIES + NEW_FAMILIES + GUIDED_FAMILIES
SLOTS = MAX_OBJECTS
COLOURS = ("red", "green", "blue", "yellow", "orange", "purple")  # palette indices 0-5 (world.COLOUR_NAMES)
HORIZONTAL = ((0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0))
_FACE_OFFSETS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
MIN_PUSHES = 2
CANDIDATES = 6  # joint proposals per episode and step
TEMPERATURE = 0.5
LEVEL_STEPS = 4  # level steps (potential unchanged within LEVEL_TOL) allowed in a row
LEVEL_TOL = 0.05
MAX_BACKTRACKS = 30  # dead ends an episode may back out of (depth first, never revisiting a state)
SPACING = 3  # robots in one joint proposal: their cells at least this far apart (Chebyshev)
STEPS_BASE, STEPS_PER_ROBOT, MAX_STEPS_CAP = 64, 16, 6000  # step limit of an episode: base + per robot, capped
# natural potential, per robot
W_OUT = 3.0  # off the goal's cells
W_D = 1.0  # per cube move to the floor or terrain (geodesic)
GEO_CAP = 24
W_FAR = 0.5  # per voxel from the goal's cells, up to FAR_CAP
FAR_CAP = 3.0
W_CENTRE = 0.15  # per column from the goal's centre, up to CENTRE_CAP
CENTRE_CAP = 16.0
W_TOUCH = 0.5  # per pair of robots face to face
W_CROWD = 0.1  # per pair of robots within two columns of each other
# guided potential
K_FILL = 4.0  # per robot on a cell of A
W_G = 1.0  # per cube move from the nearest empty cell of A
# natural statistics
REF_SAMPLES = 32
STATS = ("lifted", "touch", "height", "crowd", "fill", "spread")  # all upper bounds but spread (lower)
QUANTILE = 0.975
# scaffolding
MAX_HELPERS = 3
MAX_SCAFFOLDS = 8  # scaffold attempts per episode, plus one per SCAFFOLDS_PER robots
SCAFFOLDS_PER = 16
SCAFFOLD_ROBOTS = 8  # stuck robots tried per attempt
SCAFFOLD_PLANS = 6  # helper layouts tried per attempt
HELPER_TRIES = 4  # robots tried per helper cell
PATH_DEPTH = 40
DROP_DIST = 3.0  # an extracted robot stops at least this far from the goal's cells
RANK_EXTRA = 6  # a walk looks this many moves past the first destination it finds for a better ranked one
DROP_EXTRA = 16  # the same for robots taken out of a natural goal (far drop cells first)
# the swarm (_swarm): robots walking out at once
WALKERS = 64  # walkers at once, at most
WALKER_SPACING = 3  # a robot sets off only this far (Chebyshev) from every walker and its next cell
RECRUIT_TRIES = 8  # breadth-first walks tried per round for new walkers
WAIT_REPLAN = 2  # rounds a walker waits before it plans its walk again
WAIT_DROP = 6  # rounds a walker waits before it gives up
RETRY = 6  # rounds before a robot that found no walk is tried again
IDLE_ROUNDS = 12  # rounds in a row with no move before the swarm stops
UNTIL_EVERY = 4  # rounds between checks of a swarm's stopping test
# guided walks
DIG_RADII = (1, 2, 3, 5, SIZE)  # columns round a stuck cell taken out, growing while digs do not bring A nearer
MAX_DIGS = 12  # digs per episode
MAX_SPREADS = 4  # spreading swarms per natural episode once its goal is clear
HOLD_STEPS = 4  # steps of a hold episode (nothing to do: every robot stays)
HEAP = 3  # natural scenes pile robots at most this high above the terrain (more when they would not fit: heap_cap)
_CUBE_CODES = torch.tensor([c for c in range(STAY) if bool(CUBE_MOVES[c])])  # the 14 moves a pivoting cube makes
_NUMBERS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def code_of(direction):
    return DIRS.index(tuple(int(v) for v in direction))


# ---------------------------------------------------------------------------------------------
# Goals


@dataclass
class Goal:
    family: str
    text: str
    world: VoxelWorld  # one scene, every object field, gravity on
    target: torch.Tensor  # (D, H, W) bool: the goal's cells (the structure, or the object on its patch)
    push: dict = None  # push family: {"slot", "dir" (z, y, x), "length"}
    info: dict = field(default_factory=dict)
    start: torch.Tensor = None  # guided: (D, H, W) bool, the start A to walk toward (as many robots, at rest)
    unused: str = "free"  # robots the goal does not use: "parked" in a row along an edge, or "free" where they are


def blank(height=None):
    """A one-scene world on the flat floor (or terrain from a heightmap (H, W)), no robots, every
    object slot empty, gravity on."""
    d = h = w = SIZE
    walls = torch.zeros(1, d, h, w, dtype=torch.bool)
    if height is not None:
        walls[0] = torch.arange(d).view(d, 1, 1) < height.unsqueeze(0)
    world = VoxelWorld(
        mass=torch.zeros(1, d, h, w),
        walls=walls,
        cheese_pos=torch.zeros(1, SLOTS, 3, dtype=torch.long),
        cheese_alive=torch.zeros(1, SLOTS, dtype=torch.bool),
        cheese_color=torch.full((1, SLOTS), -1, dtype=torch.long),
        gravity=torch.ones(1, dtype=torch.bool),
    )
    world = W.with_objects(world, SLOTS)
    return replace(world, food=torch.zeros(1, SLOTS, dtype=torch.bool))


def _dilate(grid, r=1):
    """(D, H, W) bool grown by r voxels in every direction (26-neighbourhood)."""
    out = torch.nn.functional.max_pool3d(grid.float()[None, None], 2 * r + 1, stride=1, padding=r)
    return out[0, 0] > 0


def scatter_robots(rng, world, n, avoid, near=None):
    """Put n robots on the ground (floor or terrain top) at random, none touching another robot,
    an object or the cells `avoid` (D, H, W) by a face, edge or corner; only in the columns `near`
    (H, W) bool when given. Returns how many it placed."""
    d, h, w = world.mass.shape[1:]
    height = ground(world.walls[0])
    placed = 0
    for _ in range(60 * max(n, 1)):
        if placed == n:
            break
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
        z = int(height[y, x])
        if z >= d or (near is not None and not bool(near[y, x])):
            continue
        solid = (world.mass[0] > 0.5) | (W.cheese_id_map(world)[0] >= 0) | avoid
        z0, z1, y0, y1, x0, x1 = max(z - 1, 0), z + 2, max(y - 1, 0), y + 2, max(x - 1, 0), x + 2
        if solid[z0:z1, y0:y1, x0:x1].any() or world.walls[0, z, y, x]:
            continue
        world.mass[0, z, y, x] = 1.0
        placed += 1
    return placed


def tower_goal(rng):
    n = int(rng.integers(3, 7))
    world = blank()
    y, x = int(rng.integers(3, SIZE - 3)), int(rng.integers(3, SIZE - 3))
    world.mass[0, :n, y, x] = 1.0
    target = world.mass[0] > 0.5
    scatter_robots(rng, world, int(rng.integers(0, 5)), _dilate(target))
    word = _NUMBERS[n]
    text = str(rng.choice([
        f"stack {word} of you into one tall tower",
        f"build a tower {word} high, standing on its own",
        f"make a single column {word} cubes tall",
        f"pile {word} of you straight up into a tower with nothing leaning on it",
        f"I want a {word}-high tower, just one column",
    ]))
    return Goal("tower", text, world, target, info={"height": n})


def bridge_goal(rng):
    """A crossing over a trench: the trench (1 to 3 deep, 2 to 4 wide) filled solid in one row of
    robots, its top flush with the two sides. A deck over air cannot be closed under the rule (the
    last cube into a gap between two cubes swings through one of them; README of
    runs/handoff/reverse-demos.md), so the crossing stands on the trench floor."""
    while True:
        top, width = int(rng.integers(1, 4)), int(rng.integers(2, 5))
        if top * width <= 9:
            break
    left = int(rng.integers(3, SIZE - width - 3))
    height = torch.full((SIZE, SIZE), top, dtype=torch.long)
    height[:, left : left + width] = 0
    transposed = bool(rng.random() < 0.5)
    if transposed:
        height = height.t().contiguous()
    world = blank(height)
    row = int(rng.integers(3, SIZE - 3))
    for k in range(left, left + width):
        y, x = (k, row) if transposed else (row, k)
        world.mass[0, :top, y, x] = 1.0
    target = world.mass[0] > 0.5
    scatter_robots(rng, world, int(rng.integers(1, 4)), _dilate(target))
    text = str(rng.choice([
        "build a crossing over the trench",
        "fill in a path across the ditch so something could roll from one side to the other",
        "make a bridge over the gap, level with both sides",
        "close the gap between the two raised parts with a walkway",
        "plug the trench in one spot so it's flush with the top",
    ]))
    return Goal("bridge", text, world, target, info={"depth": top, "width": width})


LETTERS = {  # rows run along y, columns along x
    "the letter L": ("X..", "X..", "XXX"),
    "the letter T": ("XXX", ".X.", ".X."),
    "the letter C": ("XXX", "X..", "XXX"),
    "the letter U": ("X.X", "X.X", "XXX"),
    "the letter H": ("X.X", "XXX", "X.X"),
    "the letter X": ("X.X", ".X.", "X.X"),
    "the letter J": (".X", ".X", "XX"),
    "a plus sign": (".X.", "XXX", ".X."),
    "a hollow square": ("XXX", "X.X", "XXX"),
    "a straight line": ("XXXX",),
    "a small solid square": ("XX", "XX"),
}


def letter_goal(rng):
    names = sorted(LETTERS)
    name = names[int(rng.integers(len(names)))]
    rows = LETTERS[name]
    if name == "a straight line" and rng.random() < 0.5:  # along y instead of x
        rows = tuple("X" for _ in rows[0])
    world = blank()
    hh, ww = len(rows), len(rows[0])
    y0, x0 = int(rng.integers(2, SIZE - hh - 2)), int(rng.integers(2, SIZE - ww - 2))
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            if ch == "X":
                world.mass[0, 0, y0 + r, x0 + c] = 1.0
    target = world.mass[0] > 0.5
    scatter_robots(rng, world, int(rng.integers(0, 4)), _dilate(target))
    text = str(rng.choice([
        f"spell out {name} on the floor",
        f"lay yourselves out flat as {name}",
        f"make {name} on the ground",
        f"arrange into {name}, all on the floor",
    ]))
    return Goal("letter", text, world, target, info={"shape": name})


def _describe(shape, colour, colour2, pattern, rng):
    c, c2 = COLOURS[colour], (COLOURS[colour2] if colour2 >= 0 else None)
    noun = {"cube": ["box", "block", "cube"], "slab": ["flat tile", "plate", "flat slab"], "rod": ["bar", "stick", "beam"]}[shape]
    noun = noun[int(rng.integers(len(noun)))]
    if pattern == "striped":
        return f"{c} and {c2} striped {noun}"
    if pattern == "two-tone":
        return f"{c} and {c2} {noun}"
    return f"{c} {noun}"


def push_goal(rng):
    """An object on a floor patch with its pushers right behind it: a 2 x 2 x 2 box, a flat tile
    or a bar, light (two robots move it), never food."""
    for _ in range(50):
        world = blank()
        shape = ("cube", "slab", "rod")[int(rng.integers(3))]
        if shape == "cube":
            form = W.object_form("cube", 2)
        elif shape == "slab":
            form = W.object_form("slab", int(rng.integers(2, 4)))
        else:
            form = W.object_form("rod", int(rng.integers(2, 5)), along=("x", "y")[int(rng.integers(2))])
        depth, rows, cols = W.form_extent(form)
        longest = max(rows, cols, depth)
        pattern = "plain"
        draw = rng.random()
        if draw < 0.2 and longest >= 3:
            pattern = "striped"
        elif draw < 0.4 and longest >= 2:
            pattern = "two-tone"
        colours = rng.permutation(len(COLOURS))
        colour, colour2, zone_colour = int(colours[0]), (int(colours[1]) if pattern != "plain" else -1), int(colours[2])
        direction = HORIZONTAL[int(rng.integers(4))]
        axis = 1 if direction[1] else 2
        face = cols if axis == 1 else rows  # base cells across the face the pushers press on
        if face < 2:
            continue
        length = int(rng.integers(3, 7))
        sy, sx = rows + int(rng.integers(0, 3)), cols + int(rng.integers(0, 3))
        zy, zx = int(rng.integers(1, SIZE - sy)), int(rng.integers(1, SIZE - sx))
        oy, ox = zy + int(rng.integers(0, sy - rows + 1)), zx + int(rng.integers(0, sx - cols + 1))
        e = torch.tensor(direction)
        pos = torch.tensor([0, oy, ox])
        start = pos - length * e
        cells = form.nonzero() + pos
        start_cells = form.nonzero() + start
        own = {tuple(c) for c in cells.tolist()}
        behind = [torch.tensor(c) - e for c in cells.tolist() if c[0] == 0 and tuple((torch.tensor(c) - e).tolist()) not in own]
        pushers = behind[: int(rng.integers(2, min(face, 3) + 1))]
        far = [p - (length + 1) * e for p in pushers]
        inside = lambda p: bool(((p >= 0) & (p < SIZE)).all()) and 1 <= int(p[1]) < SIZE - 1 and 1 <= int(p[2]) < SIZE - 1  # noqa: E731
        if not all(inside(c) for c in start_cells) or not all(inside(p) for p in far):
            continue
        world.zones[0, zy : zy + sy, zx : zx + sx] = zone_colour
        world.cheese_pos[0, 0] = pos
        world.cheese_alive[0, 0] = True
        world.cheese_color[0, 0] = colour
        world.colour2[0, 0] = colour2
        world.pattern[0, 0] = W.PATTERNS.index(pattern)
        world.form[0, 0] = form
        world.friction[0, 0] = FRICTION
        for p in pushers:
            world.mass[0, int(p[0]), int(p[1]), int(p[2])] = 1.0
        corridor = torch.zeros(SIZE, SIZE, SIZE, dtype=torch.bool)
        for k in range(length + 2):
            for c in (cells - k * e).tolist() + [(p - k * e).tolist() for p in pushers]:
                if all(0 <= v < SIZE for v in c):
                    corridor[c[0], c[1], c[2]] = True
        target = W.cheese_id_map(world)[0] == 0
        scatter_robots(rng, world, int(rng.integers(0, 4)), _dilate(corridor))
        desc = _describe(shape, colour, colour2, pattern, rng)
        place = COLOURS[zone_colour]
        text = str(rng.choice([
            f"push the {desc} onto the {place} patch on the floor",
            f"slide the {desc} into the {place} square",
            f"get the {desc} onto the {place} mat",
            f"move the {desc} so it sits on the {place} painted area",
        ]))
        return Goal("push", text, world, target, push={"slot": 0, "dir": direction, "length": length}, info={"shape": shape, "pattern": pattern})
    raise RuntimeError("no push goal fits")


SIDES = {"back": (1, 0), "front": (1, SIZE - 1), "left": (2, 0), "right": (2, SIZE - 1)}  # edge name: (axis, index) of the row parked robots stand in


def park_robots(rng, world, n, avoid):
    """Put n robots in one row on the ground along an edge of the grid, side by side (unused robots
    parked out of the way). Returns the edge's name, or None when no edge has room."""
    height = ground(world.walls[0])
    solid = (world.mass[0] > 0.5) | (W.cheese_id_map(world)[0] >= 0) | _dilate(avoid)
    names = list(SIDES)
    for k in rng.permutation(len(names)):
        name = names[int(k)]
        axis, index = SIDES[name]
        cells = []
        for j in range(1, SIZE - 1):
            y, x = (index, j) if axis == 1 else (j, index)
            z = int(height[y, x])
            cells.append(None if z >= SIZE or bool(solid[max(z - 1, 0) : z + 2, y, x].any()) else (z, y, x))
        runs, run = [], []
        for c in cells + [None]:
            if c is None:
                if len(run) >= n:
                    runs.append(run)
                run = []
            else:
                run.append(c)
        if not runs:
            continue
        run = runs[int(rng.integers(len(runs)))]
        first = int(rng.integers(0, len(run) - n + 1))
        for z, y, x in run[first : first + n]:
            world.mass[0, z, y, x] = 1.0
        return name
    return None


def _spares(rng, world, target, count, unused, near=None):
    """count unused robots: parked in a row along an edge, or free (scattered on the ground, in
    the columns `near` when given). Returns the text to add to the instruction."""
    if count <= 0:
        return ""
    if unused == "parked":
        side = park_robots(rng, world, count, target)
        if side is not None:
            return str(rng.choice([f", and park the rest of you in a row along the {side} edge", f"; everyone else lines up along the {side} edge"]))
    scatter_robots(rng, world, count, _dilate(target), near=near)
    return ""


def _near_columns(target, r):
    """(H, W) bool: columns within Chebyshev r of the goal's columns."""
    cols = target.any(dim=0).float()[None, None]
    return torch.nn.functional.max_pool2d(cols, 2 * r + 1, stride=1, padding=r)[0, 0] > 0


def _shape_world(cells, height=None):
    world = blank(height)
    for z, y, x in cells:
        world.mass[0, z, y, x] = 1.0
    return world, world.mass[0] > 0.5


def _unused(rng):
    return "parked" if rng.random() < 0.5 else "free"


def span_goal(rng):
    """A one-wide deck over open air across a trench, flush with both banks (the deck stands on
    nothing: every deck robot is held by its neighbours and the banks). Closing it needs helpers:
    the last robot into a gap between two solids has no pivot without support beside it."""
    t, width = int(rng.integers(2, 5)), int(rng.integers(2, 7))
    left = int(rng.integers(3, SIZE - width - 3))
    height = torch.full((SIZE, SIZE), t, dtype=torch.long)
    height[:, left : left + width] = 0
    transposed = bool(rng.random() < 0.5)
    if transposed:
        height = height.t().contiguous()
    row = int(rng.integers(3, SIZE - 3))
    cells = [((t - 1, k, row) if transposed else (t - 1, row, k)) for k in range(left, left + width)]
    world, target = _shape_world(cells, height)
    scatter_robots(rng, world, int(rng.integers(3, 6)), _dilate(target), near=_near_columns(target, 3))
    text = str(rng.choice([
        "build a bridge one robot wide across the gap, level with the top, with nothing underneath it",
        "span the trench with a single line of you, flush with both edges",
        "make a narrow walkway over the ditch, hanging in the air between the two sides",
        "lay a one-wide deck straight across the gap at the top",
    ]))
    return Goal("span", text, world, target, info={"air": t - 1, "width": width})


def arch_goal(rng):
    """Two pillars and a lintel: flush between the pillar tops (a span: needs helpers) or resting on
    top of them."""
    p, g = int(rng.integers(2, 5)), int(rng.integers(1, 5))
    along_x = bool(rng.random() < 0.5)
    y0, x0 = int(rng.integers(2, SIZE - 2)), int(rng.integers(2, SIZE - g - 4))
    flush = bool(rng.random() < 0.6)

    def at(z, j):
        return (z, y0, x0 + j) if along_x else (z, x0 + j, y0)

    cells = [at(z, j) for j in (0, g + 1) for z in range(p)]
    cells += [at(p - 1, j) for j in range(1, g + 1)] if flush else [at(p, j) for j in range(0, g + 2)]
    world, target = _shape_world(cells)
    scatter_robots(rng, world, int(rng.integers(3, 6)), _dilate(target), near=_near_columns(target, 3))
    text = str(rng.choice([
        "build an arch: two legs with a beam across the top",
        "make a doorway shape, two posts and a bar over the gap",
        f"build an arch with an opening {_NUMBERS.get(g, str(g))} wide",
        "put up two columns and bridge them at the top",
    ]))
    return Goal("arch", text, world, target, info={"pillars": p, "gap": g, "flush": flush})


def _table_cells(y0, x0, a, b, legs, along=None, back=0):
    cells = [(legs, y0 + i, x0 + j) for i in range(a) for j in range(b)]
    for i, j in ((0, 0), (0, b - 1), (a - 1, 0), (a - 1, b - 1)):
        cells += [(z, y0 + i, x0 + j) for z in range(legs)]
    if back:
        edge = [(y0, x0 + j) for j in range(b)] if along == "y0" else [(y0 + i, x0) for i in range(a)]
        cells += [(legs + k, y, x) for k in range(1, back + 1) for y, x in edge]
    return sorted(set(cells))


def table_goal(rng):
    a, b, legs = int(rng.integers(2, 5)), int(rng.integers(2, 5)), int(rng.integers(1, 4))
    y0, x0 = int(rng.integers(2, SIZE - a - 2)), int(rng.integers(2, SIZE - b - 2))
    world, target = _shape_world(_table_cells(y0, x0, a, b, legs))
    unused = _unused(rng)
    extra = _spares(rng, world, target, int(rng.integers(0, 4)), unused)
    text = str(rng.choice([
        f"build a table: a {a} by {b} top on four legs",
        f"make a little table with legs {_NUMBERS.get(legs, 'one') if legs > 1 else 'one'} high",
        "build a table, flat top held up at the four corners",
        "make a table shape with a gap underneath",
    ])) + extra
    return Goal("table", text, world, target, info={"top": [a, b], "legs": legs, "unused": unused}, unused=unused)


def chair_goal(rng):
    a, legs, back = int(rng.integers(2, 4)), int(rng.integers(1, 3)), int(rng.integers(1, 4))
    y0, x0 = int(rng.integers(2, SIZE - a - 2)), int(rng.integers(2, SIZE - a - 2))
    world, target = _shape_world(_table_cells(y0, x0, a, a, legs, along=str(rng.choice(["y0", "x0"])), back=back))
    unused = _unused(rng)
    extra = _spares(rng, world, target, int(rng.integers(0, 4)), unused)
    text = str(rng.choice([
        "build a chair with a backrest",
        "make a chair: a seat on four legs and a back sticking up on one side",
        "build something to sit on, with legs and a back",
    ])) + extra
    return Goal("chair", text, world, target, info={"seat": a, "legs": legs, "back": back, "unused": unused}, unused=unused)


def _cantilever_cells(y, x, h, arm, e, zarm=None, both=False):
    zarm = h - 1 if zarm is None else zarm
    cells = [(z, y, x) for z in range(h)]
    cells += [(zarm, y + j * e[1], x + j * e[2]) for j in range(1, arm + 1)]
    if both:
        cells += [(zarm, y - j * e[1], x - j * e[2]) for j in range(1, arm + 1)]
    return cells


def cantilever_goal(rng):
    """A column with an arm sticking out sideways at the top over air (an L; a T with two arms)."""
    h, arm = int(rng.integers(2, 6)), int(rng.integers(2, 5))
    e = HORIZONTAL[int(rng.integers(4))]
    both = bool(rng.random() < 0.3)
    y, x = int(rng.integers(arm + 2, SIZE - arm - 2)), int(rng.integers(arm + 2, SIZE - arm - 2))
    world, target = _shape_world(_cantilever_cells(y, x, h, arm, e, both=both))
    unused = _unused(rng)
    extra = _spares(rng, world, target, int(rng.integers(0, 4)), unused)
    text = str(rng.choice([
        f"make a post {_NUMBERS.get(h, str(h))} high with an arm sticking out {'both ways' if both else 'sideways'} at the top",
        "build a T shape standing up" if both else "build an upside-down L standing on the floor",
        f"stand a column up and hang a {_NUMBERS.get(arm, str(arm))}-long arm off its top",
    ])) + extra
    return Goal("cantilever", text, world, target, info={"height": h, "arm": arm, "both": both, "unused": unused}, unused=unused)


def _big_cells(rng):
    kind = str(rng.choice(["pyramid", "stairs", "room", "block", "big table"]))
    if kind == "pyramid":
        base = int(rng.choice([5, 7]))
        y0, x0 = int(rng.integers(1, SIZE - base - 1)), int(rng.integers(1, SIZE - base - 1))
        cells = [(z, y0 + i, x0 + j) for z in range((base + 1) // 2) for i in range(z, base - z) for j in range(z, base - z)]
        text = "build a stepped pyramid"
    elif kind == "stairs":
        width, steps = int(rng.integers(3, 6)), int(rng.integers(4, 7))
        while width * steps * (steps + 1) // 2 > 105:
            width -= 1
        y0, x0 = int(rng.integers(1, SIZE - width - 1)), int(rng.integers(1, SIZE - steps - 1))
        cells = [(z, y0 + i, x0 + j) for j in range(steps) for z in range(j + 1) for i in range(width)]
        text = f"build a staircase {_NUMBERS.get(steps, str(steps))} steps high"
    elif kind == "room":
        side, wall = int(rng.integers(5, 8)), int(rng.integers(2, 4))
        y0, x0 = int(rng.integers(1, SIZE - side - 1)), int(rng.integers(1, SIZE - side - 1))
        door = int(rng.integers(1, side - 1))
        cells = [(z, y0 + i, x0 + j) for z in range(wall) for i in range(side) for j in range(side) if (i in (0, side - 1) or j in (0, side - 1)) and not (i == 0 and j == door and z < 2)]
        text = "build the walls of a small room with a doorway"
    elif kind == "block":
        a, b, c = int(rng.integers(3, 6)), int(rng.integers(3, 6)), int(rng.integers(3, 6))
        while a * b * c > 100:
            c -= 1
        y0, x0 = int(rng.integers(1, SIZE - a - 1)), int(rng.integers(1, SIZE - b - 1))
        cells = [(z, y0 + i, x0 + j) for z in range(c) for i in range(a) for j in range(b)]
        text = f"pack together into one solid block {a} by {b}, {c} high"
    else:
        a, b, legs = int(rng.integers(5, 8)), int(rng.integers(5, 8)), int(rng.integers(2, 4))
        y0, x0 = int(rng.integers(1, SIZE - a - 1)), int(rng.integers(1, SIZE - b - 1))
        cells = _table_cells(y0, x0, a, b, legs)
        text = "build one big table with four legs"
    return kind, cells, text


def big_goal(rng):
    """A structure of 35 to 105 robots with spares to bring the scene to 60-115 robots."""
    kind, cells, text = _big_cells(rng)
    world, target = _shape_world(cells)
    unused = _unused(rng)
    total = int(rng.integers(max(60, len(cells)), 116))
    extra = _spares(rng, world, target, total - len(cells), unused)
    return Goal("big", text + extra, world, target, info={"kind": kind, "structure": len(cells), "robots": int((world.mass > 0.5).sum()), "unused": unused}, unused=unused)


def _mask(cells):
    out = torch.zeros(SIZE, SIZE, SIZE, dtype=torch.bool)
    for z, y, x in cells:
        out[z, y, x] = True
    return out


def edit_goal(rng):
    """A small edit of a built shape: the start A is the shape, the goal the shape changed (lower
    or raise an arm, a tower one taller or shorter, an arm one longer or shorter). A spare robot
    stands beside the base where one is needed; 0-3 free robots stay where they are."""
    kind = str(rng.choice(["lower the arm", "raise the arm", "taller", "shorter", "longer arm", "shorter arm"]))
    e = HORIZONTAL[int(rng.integers(4))]
    y, x = int(rng.integers(6, SIZE - 6)), int(rng.integers(6, SIZE - 6))
    h, arm = int(rng.integers(3, 6)), int(rng.integers(2, 4))
    spare = [(0, y - 2 * e[1], x - 2 * e[2])]
    if kind in ("lower the arm", "raise the arm"):
        high = _cantilever_cells(y, x, h, arm, e)
        low = _cantilever_cells(y, x, h - 1, arm, e) + spare
        start, goal = (high, low) if kind == "lower the arm" else (low, high)
        text = str(rng.choice(["lower the arm by one", "bring the arm down a level", "move the sticking-out part one lower"] if kind == "lower the arm" else ["raise the arm by one", "lift the arm up a level", "move the sticking-out part one higher"]))
    elif kind in ("taller", "shorter"):
        tall = [(z, y, x) for z in range(h + 1)]
        short = [(z, y, x) for z in range(h)] + spare
        start, goal = (short, tall) if kind == "taller" else (tall, short)
        text = str(rng.choice(["make the tower one taller", "add one more on top of the tower", "stack one more onto the column"] if kind == "taller" else ["make the tower one shorter", "take the top one off the tower", "lower the column by one"]))
    else:
        long = _cantilever_cells(y, x, h, arm + 1, e)
        short = _cantilever_cells(y, x, h, arm, e) + spare
        start, goal = (short, long) if kind == "longer arm" else (long, short)
        text = str(rng.choice(["make the arm one longer", "extend the arm by one"] if kind == "longer arm" else ["make the arm one shorter", "pull the tip of the arm back in"]))
    world, target = _shape_world(goal)
    a = _mask(start)
    free = blank()
    free.mass[0] = (target | a).float()
    scatter_robots(rng, free, int(rng.integers(0, 4)), _dilate(target | a))
    loose = (free.mass[0] > 0.5) & ~(target | a)
    world.mass[0] = ((world.mass[0] > 0.5) | loose).float()
    return Goal("edit", text, world, target, start=a | loose, info={"edit": kind})


def retarget_goal(rng):
    """A built shape (letter, table, L or chair, with its spares) is the start A; the goal takes k
    of its robots (spares first, then the shape's from the top down, each removal keeping the rest
    at rest) into a tower of k beside it; every other robot stays where it is."""
    for _ in range(20):
        base = make_goal(str(rng.choice(["letter", "table", "cantilever", "chair"])), rng)
        a = base.world.mass[0] > 0.5
        k = int(rng.integers(3, 7))
        if int(a.sum()) < k:
            continue
        free = ~_dilate(a).any(dim=0)
        free[:2], free[-2:], free[:, :2], free[:, -2:] = False, False, False, False
        spots = free.nonzero()
        if len(spots) == 0:
            continue
        cols = a.any(dim=0).nonzero().float()
        dist = torch.cdist(spots.float(), cols).min(dim=1).values
        spots = spots[(dist >= 2) & (dist <= 5)]
        if len(spots) == 0:
            continue
        ty, tx = spots[int(rng.integers(len(spots)))].tolist()
        robots = a.nonzero().tolist()
        order = sorted(robots, key=lambda c: (bool(base.target[c[0], c[1], c[2]]), -c[0], rng.random()))
        left, taken = a.clone(), []
        for c in order:
            if len(taken) == k:
                break
            left[c[0], c[1], c[2]] = False
            held = W.supported(base.world, left.unsqueeze(0))[0]
            if bool((held | ~left).all()):
                taken.append(c)
            else:
                left[c[0], c[1], c[2]] = True
        if len(taken) < k:
            continue
        goal = left.clone()
        goal[:k, ty, tx] = True
        world = replace(base.world, mass=goal.float().unsqueeze(0).clone())
        word = _NUMBERS[k]
        shape = {"letter": base.info.get("shape", "the shape"), "table": "the table", "cantilever": "the post", "chair": "the chair"}[base.family]
        text = str(rng.choice([
            f"now make a tower of {word} out of it",
            f"take {word} of you off {shape} and stack them into a tower beside it",
            f"use {word} robots from {shape} to build a {word}-high tower next to it",
        ]))
        target = goal & ~a
        return Goal("retarget", text, world, target, start=a, info={"from": base.family, "tower": k})
    raise RuntimeError("no retarget goal fits")


MAKERS = {
    "tower": tower_goal, "bridge": bridge_goal, "letter": letter_goal, "push": push_goal,
    "span": span_goal, "arch": arch_goal, "table": table_goal, "chair": chair_goal, "cantilever": cantilever_goal, "big": big_goal,
    "edit": edit_goal, "retarget": retarget_goal,
}


def fixed_point(world):
    """(B,) bool: every robot staying changes nothing (the state is at rest under the rule)."""
    nxt, _ = W.step(world, torch.full(world.mass.shape, STAY, dtype=torch.long), rule=RULE)
    return same_state(nxt, world)


def make_goal(family, rng, tries=20):
    """A goal of `family` that is a fixed point of the forward step (and so is its start A)."""
    for _ in range(tries):
        goal = MAKERS[family](rng)
        worlds = [goal.world] + ([replace(goal.world, mass=goal.start.float().unsqueeze(0))] if goal.start is not None else [])
        if bool(fixed_point(W.concat(worlds)).all()) and (goal.start is None or int(goal.start.sum()) == int((goal.world.mass > 0.5).sum())):
            return goal
    raise RuntimeError(f"no stable {family} goal in {tries} tries")


# ---------------------------------------------------------------------------------------------
# States, local pivot tests, distances


def same_state(a, b):
    """(B,) bool: the same robots, the same live objects at the same places."""
    mass = ((a.mass > 0.5) == (b.mass > 0.5)).flatten(1).all(dim=1)
    alive = (a.cheese_alive == b.cheese_alive).all(dim=1)
    pos = ((a.cheese_pos == b.cheese_pos).all(dim=2) | ~a.cheese_alive).all(dim=1)
    return mass & alive & pos


def touching_pairs(occ):
    """(B,) long: pairs of robots face to face."""
    return (occ[:, 1:] & occ[:, :-1]).flatten(1).sum(1) + (occ[:, :, 1:] & occ[:, :, :-1]).flatten(1).sum(1) + (occ[:, :, :, 1:] & occ[:, :, :, :-1]).flatten(1).sum(1)


def crowd_pairs(occ):
    """(B,) float: pairs of robots whose columns are within two of each other (Chebyshev)."""
    col = occ.sum(dim=1).float()
    box = torch.nn.functional.avg_pool2d(col.unsqueeze(1), 5, stride=1, padding=2, count_include_pad=True)[:, 0] * 25
    return ((col * box).flatten(1).sum(1) - occ.flatten(1).sum(1)) / 2


def distance_field(target):
    """(D, H, W) float: distance from the nearest of the goal's cells."""
    d, h, w = target.shape
    grid = torch.stack(torch.meshgrid(torch.arange(d), torch.arange(h), torch.arange(w), indexing="ij"), dim=-1).float()
    return torch.cdist(grid.view(-1, 3), target.nonzero().float()).min(dim=1).values.view(d, h, w)


_R = 3  # a move's support and swing cells lie within 2 of its source, the source within 1 of its target
_SIDE = 2 * _R + 1
_OFF = torch.stack(torch.meshgrid(*(torch.arange(-_R, _R + 1),) * 3, indexing="ij"), dim=-1).view(-1, 3)
_CODE_DIRS = DIR_TABLE[_CUBE_CODES]  # (14, 3)
_SRC_LOCAL = _R - _CODE_DIRS  # where the source of each move sits in a crop centred on its target
_K14 = torch.zeros(1, 1, 3, 3, 3)
for _d in _CODE_DIRS.tolist():
    _K14[0, 0, 1 + _d[0], 1 + _d[1], 1 + _d[2]] = 1.0


def _pivot_table():
    """Every pivot that can bring a cube into the centre of a 7^3 crop (world.pivot_moves' pivots for
    the 14 cube moves, from src = centre - move), as flat crop indices: (code (P,) index into
    _CUBE_CODES, supports (P, 2) (the crop's size where there is only one), down (P,) bool: leans on
    what is below (cubes, terrain, objects) rather than a surface (cubes, terrain), swing (P, 6) (the
    crop's size where there are fewer))."""
    n = _SIDE ** 3
    codes, supports, down, swings = [], [], [], []

    def flat(c):
        return (c[0] * _SIDE + c[1]) * _SIDE + c[2]

    for k, code in enumerate(_CUBE_CODES.tolist()):
        step = DIRS[code]
        src = tuple(_R - v for v in step)
        if sum(v != 0 for v in step) == 1:
            pivots = [(step, side, False) for side in W._SIDES if not any(a * b for a, b in zip(side, step))]
        elif step[0] > 0:
            pivots = [(W._UP, (0, step[1], step[2]), True)]
        else:
            pivots = [((0, step[1], step[2]), W._DOWN, True)]
        for h, side, half in pivots:
            swing = [h, W._add((0, 0, 0), side, -1), W._add(h, side, -1)] + ([W._add(h, side), W._add(h, h), W._add(W._add(h, h), side)] if half else [])
            leaned = [side] if half else [side, W._add(h, side)]
            codes.append(k)
            supports.append([flat(W._add(src, c)) for c in leaned] + [n] * (2 - len(leaned)))
            down.append(side == W._DOWN)
            swings.append([flat(W._add(src, c)) for c in swing] + [n] * (6 - len(swing)))
    return torch.tensor(codes), torch.tensor(supports), torch.tensor(down), torch.tensor(swings)


_PIVOT_CODE, _PIVOT_SUPPORT, _PIVOT_DOWN, _PIVOT_SWING = _pivot_table()


def _legal_into(cubes, walls, cheese):
    """(N, 14) bool: world.pivot_moves (release, as the rule switch_cube) at each move's source in
    7^3 crops (N, 7, 7, 7) centred on the target, computed at those 14 cells only."""
    m = len(cubes)
    surface = (cubes | walls).reshape(m, -1)
    below = surface | cheese.reshape(m, -1)
    one, zero = torch.ones(m, 1, dtype=torch.bool), torch.zeros(m, 1, dtype=torch.bool)
    lean = torch.where(_PIVOT_DOWN.view(1, -1, 1), torch.cat([below, one], 1)[:, _PIVOT_SUPPORT], torch.cat([surface, one], 1)[:, _PIVOT_SUPPORT]).all(dim=2)
    clear = ~torch.cat([below, zero], 1)[:, _PIVOT_SWING].any(dim=2)
    ok = (lean & clear).float()
    return (ok @ torch.nn.functional.one_hot(_PIVOT_CODE, len(_CUBE_CODES)).float()) > 0


def padded_grids(world, cubes=None):
    """(cubes, walls, cheese) of a batch, each (B, D + 6, H + 6, W + 6) bool: walls solid below the
    floor (as pivot_moves pads them), everything else empty off the grid."""
    occ = (world.mass > 0.5) if cubes is None else cubes
    cheese = W.cheese_id_map(world) >= 0
    b, d, h, w = occ.shape
    out = []
    for grid, below in ((occ, False), (world.walls, True), (cheese, False)):
        p = torch.zeros(b, d + 2 * _R, h + 2 * _R, w + 2 * _R, dtype=torch.bool)
        p[:, :_R] = below
        p[:, _R : _R + d, _R : _R + h, _R : _R + w] = grid
        out.append(p)
    return tuple(out)


def _crop(padded, scene, here):
    b, dp, hp, wp = padded.shape
    c = here.unsqueeze(1) + _OFF.unsqueeze(0) + _R
    flat = (c[..., 0] * hp + c[..., 1]) * wp + c[..., 2]
    return padded.reshape(b, -1)[scene.unsqueeze(1), flat].view(-1, _SIDE, _SIDE, _SIDE)


def sources(padded, scene, here, remove_self=True):
    """Which of the 14 cube moves could have brought a robot into `here` (N, 3) of scenes `scene`
    (N,) of a padded batch (padded_grids) from src = here - move, under the rule with the rest of
    the scene as it is: pivot_moves' test on the 7^3 crop round `here` (the robot there removed when
    remove_self; _legal_into evaluates it at the 14 sources only), src on the grid and free.
    Returns (src (N, 14, 3), legal (N, 14) bool)."""
    cubes, walls, cheese = padded
    if len(here) == 0:
        return torch.zeros(0, len(_CUBE_CODES), 3, dtype=torch.long), torch.zeros(0, len(_CUBE_CODES), dtype=torch.bool)
    cc = _crop(cubes, scene, here)
    if remove_self:
        cc[:, _R, _R, _R] = False
    if RULE == "switch_cube":
        legal = _legal_into(cc, _crop(walls, scene, here), _crop(cheese, scene, here))
    else:
        table = W.pivot_moves(cc, _crop(walls, scene, here), _crop(cheese, scene, here))
        legal = table[:, _CUBE_CODES][:, torch.arange(len(_CUBE_CODES)), _SRC_LOCAL[:, 0], _SRC_LOCAL[:, 1], _SRC_LOCAL[:, 2]]
    src = here.unsqueeze(1) - _CODE_DIRS.unsqueeze(0)
    d, h, w = (n - 2 * _R for n in cubes.shape[1:])
    inside = (src >= 0).all(dim=2) & (src[..., 0] < d) & (src[..., 1] < h) & (src[..., 2] < w)
    hp, wp = cubes.shape[2], cubes.shape[3]
    flat = ((src[..., 0] + _R) * hp + src[..., 1] + _R) * wp + src[..., 2] + _R
    taken = (cubes | walls | cheese).reshape(len(cubes), -1)[scene.unsqueeze(1), flat]
    return src, legal & inside & ~taken


def single_moves(world, padded=None):
    """Every move one robot could have made into its cell: for each robot (scene, here) and each of
    the 14 cube moves, whether a robot at src = here - move could pivot into here with the rest of
    the scene as it is (the joint step is verified later). Returns scene (N,), here (N, 3),
    src (N, 14, 3), legal (N, 14)."""
    occ = world.mass > 0.5
    scene, z, y, x = occ.nonzero(as_tuple=True)
    here = torch.stack([z, y, x], dim=1)
    src, legal = sources(padded_grids(world) if padded is None else padded, scene, here)
    return scene, here, src, legal


def passable(world):
    """(B, D, H, W) bool: cells a cube can be in: robots, and free cells with something solid (a
    robot, terrain, an object, the floor) below them or beside them (the sides a pivot leans on;
    never only above)."""
    occ = world.mass > 0.5
    cheese = W.cheese_id_map(world) >= 0
    solid = occ | world.walls | cheese
    attach = torch.zeros_like(occ)
    attach[:, 0] = True
    attach[:, 1:] |= solid[:, :-1]
    attach[:, :, 1:] |= solid[:, :, :-1]
    attach[:, :, :-1] |= solid[:, :, 1:]
    attach[:, :, :, 1:] |= solid[:, :, :, :-1]
    attach[:, :, :, :-1] |= solid[:, :, :, 1:]
    return ~world.walls & ~cheese & (occ | attach)


def on_ground(world):
    """(B, D, H, W) bool: cells resting on the floor or the terrain."""
    rest = torch.zeros_like(world.walls)
    rest[:, 0] = True
    rest[:, 1:] = world.walls[:, :-1]
    return rest & ~world.walls


def geodesic(seed, ok, cap=GEO_CAP):
    """(B, D, H, W) float: cube moves (the 14) from the cells `seed` through cells `ok`, cap + 1
    where further or unreachable."""
    reached = seed & ok
    dist = torch.full(seed.shape, float(cap + 1))
    dist[reached] = 0.0
    for k in range(1, cap + 1):
        grow = torch.nn.functional.conv3d(reached.float().unsqueeze(1), _K14, padding=1)[:, 0] > 0
        new = grow & ok & ~reached
        if not new.any():
            break
        dist[new] = float(k)
        reached |= new
    return dist


def ground(walls):
    """(..., H, W) long: the terrain height of each column (walls are a heightmap)."""
    return walls.sum(dim=-3)


# ---------------------------------------------------------------------------------------------
# Potentials and natural statistics


def natural_field(target):
    """(2, D, H, W) float: the static parts of what one robot in each voxel adds to the natural
    potential: off the goal's cells; distance from them and from their centre (counted on the
    ground only, step_fields)."""
    d, h, w = target.shape
    grid = torch.stack(torch.meshgrid(torch.arange(d), torch.arange(h), torch.arange(w), indexing="ij"), dim=-1).float()
    centre = target.nonzero().float()[:, 1:].mean(dim=0)
    horizontal = (grid[..., 1:] - centre).norm(dim=-1).clamp(max=CENTRE_CAP)
    return torch.stack([W_OUT * (~target).float(), W_FAR * distance_field(target).clamp(max=FAR_CAP) + W_CENTRE * horizontal])


def step_fields(world, static, starts, guided):
    """(B, D, H, W) float per-voxel potential of the current states: natural rows W_OUT off the
    goal's cells minus W_D x geodesic distance to the ground, plus the spreading terms where that
    distance is 0; guided rows K_FILL on A's cells, minus W_G x geodesic distance to A's empty cells
    elsewhere."""
    ok = passable(world)
    out = static[:, 0].clone()
    nat = ~guided
    if nat.any():
        sub = world.index(nat.nonzero().squeeze(1))
        dg = geodesic(on_ground(sub), ok[nat])
        out[nat] = static[nat, 0] - W_D * dg + static[nat, 1] * (dg == 0).float()
    if guided.any():
        idx = guided.nonzero().squeeze(1)
        a = starts[idx]
        empty = a & ~(world.mass[idx] > 0.5)
        out[idx] = K_FILL * a.float() - W_G * geodesic(empty, ok[idx]) * (~a).float()
    return out


def potential(world, fields, guided):
    """(B,) float: the per-voxel fields summed over robots; natural rows also pay for robots face to
    face and in neighbouring columns."""
    occ = world.mass > 0.5
    pairs = W_TOUCH * touching_pairs(occ).float() + W_CROWD * crowd_pairs(occ)
    return (fields * occ).flatten(1).sum(dim=1) - pairs * (~guided).float()


def start_stats(world, target):
    """(B, 6) float, STATS: share of robots not resting on the floor or terrain, face-to-face pairs
    per robot, the highest robot above the terrain, other robots within two columns per robot,
    share of the goal's cells holding a robot, mean column distance of robots from the goal's
    centre."""
    occ = world.mass > 0.5
    n = occ.flatten(1).sum(1).clamp(min=1).float()
    lifted = (occ & ~on_ground(world)).flatten(1).sum(1) / n
    touch = touching_pairs(occ).float() / n
    d = occ.shape[1]
    above = torch.arange(d).view(1, d, 1, 1) - ground(world.walls).unsqueeze(1)
    height = torch.where(occ, above, torch.full_like(above, -1)).flatten(1).amax(dim=1).float()
    crowd = 2 * crowd_pairs(occ) / n
    fill = (occ & target).flatten(1).sum(1) / target.flatten(1).sum(1).clamp(min=1)
    spread = torch.zeros(len(occ))
    for i in range(len(occ)):
        centre = target[i].nonzero().float()[:, 1:].mean(dim=0)
        robots = occ[i].nonzero().float()[:, 1:]
        spread[i] = (robots - centre).norm(dim=1).mean() if len(robots) else 0.0
    return torch.stack([lifted, touch, height, crowd, fill, spread], dim=1)


def heap_cap(n, walls):
    """How high natural scenes of n robots pile above the terrain: HEAP, or more when n robots would
    fill more than two thirds of HEAP layers over the open columns (500-800 robots on 16 x 16)."""
    cols = int((ground(walls) < walls.shape[0]).sum())
    return max(HEAP, math.ceil(1.5 * n / max(cols, 1)))


def natural_scenes(walls, n, rng, samples=REF_SAMPLES, avoid=None):
    """(samples, D, H, W) bool: n robots dropped on the terrain the way sample_scenes and
    sample_object_scenes place them (random columns, stacked when a column is drawn again, at
    most heap_cap above the terrain: three up to about 400 robots), off the cells `avoid` (objects)."""
    d, h, w = walls.shape
    cap = heap_cap(n, walls)
    height = ground(walls).numpy()
    base = walls.numpy().copy()
    if avoid is not None:
        base |= avoid.numpy()
    out = np.zeros((samples, d, h, w), dtype=bool)
    for s in range(samples):
        taken = base.copy()
        fill = height.copy()
        left = n
        for y, x in zip(rng.integers(0, h, 4 * n + 20), rng.integers(0, w, 4 * n + 20)):
            if left == 0:
                break
            z0 = int(fill[y, x])
            while z0 < d and taken[z0, y, x]:
                z0 += 1
            if z0 >= min(d, int(height[y, x]) + cap):
                continue
            out[s, z0, y, x] = True
            taken[z0, y, x] = True
            fill[y, x] = z0 + 1
            left -= 1
    return torch.from_numpy(out)


def natural_bands(world, target, rng, samples=REF_SAMPLES):
    """(low (6,), high (6,)): the band of each statistic over natural scenes on this terrain with as
    many robots and this goal: QUANTILE above for all but spread, 1 - QUANTILE below for spread."""
    n = int((world.mass[0] > 0.5).sum())
    scenes = natural_scenes(world.walls[0], n, rng, samples, avoid=W.cheese_id_map(world)[0] >= 0)
    ref = replace(world.index([0] * samples), mass=scenes.float())
    stats = start_stats(ref, target.unsqueeze(0).expand(samples, -1, -1, -1)).numpy()
    low = np.full(len(STATS), -np.inf)
    high = np.quantile(stats, QUANTILE, axis=0)
    high[STATS.index("spread")] = np.inf
    low[STATS.index("spread")] = np.quantile(stats[:, STATS.index("spread")], 1 - QUANTILE)
    return torch.tensor(low, dtype=torch.float) - 1e-6, torch.tensor(high, dtype=torch.float) + 1e-6


def in_band(stats, band):
    low, high = band
    return bool(((stats >= low) & (stats <= high)).all())


# ---------------------------------------------------------------------------------------------
# Proposals


@dataclass
class _Candidate:
    episode: int
    moves: list  # [(here (3,), src (3,), code)]
    push: tuple = None  # (slot, dir (3,), pressers [(3,)])


def _object_cells(world, b, slot, pos):
    form = world.form[b, slot]
    return form.nonzero() + torch.as_tensor(pos)


def _push_candidates(world, b, goal):
    """The push phase's two proposals: the object one cell back toward its start with the robots
    behind it pressing, and the robots right behind it stepping one cell back."""
    slot, e = goal.push["slot"], torch.tensor(goal.push["dir"])
    pos = world.cheese_pos[b, slot]
    occ = world.mass[b] > 0.5
    out = []
    back_cells = _object_cells(world, b, slot, pos - e)
    now_cells = _object_cells(world, b, slot, pos)
    now = {tuple(c) for c in now_cells.tolist()}
    others = (W.cheese_id_map(world)[b] >= 0) & (W.cheese_id_map(world)[b] != slot)
    clear = all(0 <= v < SIZE for c in back_cells.tolist() for v in c) and all(
        tuple(c) in now or not (occ[c[0], c[1], c[2]] or world.walls[b, c[0], c[1], c[2]] or others[c[0], c[1], c[2]]) for c in back_cells.tolist()
    )
    robots = occ.nonzero()
    if clear:
        back = {tuple(c) for c in back_cells.tolist()}
        pressers = [r for r in robots if tuple((r + e).tolist()) in back and tuple(r.tolist()) not in back]
        if pressers:
            out.append(_Candidate(b, [], (slot, e, pressers)))
    behind = [r for r in robots if tuple((r + e).tolist()) in now]
    if behind:
        out.append(_Candidate(b, [(tuple(r.tolist()), tuple((r - e).tolist()), code_of(e)) for r in behind]))
    return out


def _spaced(picks, here, src):
    """The moves of `picks` [(row, j)] kept greedily in order when their cells are at least SPACING
    from every kept move's cells."""
    kept, cells = [], np.zeros((0, 3), dtype=np.int64)
    for r, j in picks:
        a, s = here[r], src[r, j]
        if len(cells) and (np.abs(cells - a).max(axis=1) < SPACING).any() | (np.abs(cells - s).max(axis=1) < SPACING).any():
            continue
        kept.append((r, j))
        cells = np.vstack([cells, a, s])
    return kept


def _scatter_candidates(b, rows, here, src, legal, gain, rng, m=CANDIDATES):
    """Up to m proposals for episode b from its robots' single moves (rows into here/src/legal/gain,
    numpy): joint moves of robots with rising moves, kept SPACING apart (each robot's best move; a
    softmax draw; a random half), the two best single moves and a random one that does not lower
    the potential."""
    g = np.where(legal[rows], gain[rows], -np.inf)
    has = (g >= -LEVEL_TOL).any(axis=1)
    if not has.any():
        return []
    rows, g = rows[has], g[has]
    best_j = g.argmax(axis=1)
    best_g = g[np.arange(len(rows)), best_j]
    order = np.argsort(-(best_g + 1e-3 * rng.random(len(rows))))
    rising = [int(r) for r in order if best_g[r] > LEVEL_TOL]
    out = []

    def candidate(picks):
        return _Candidate(b, [(tuple(here[rows[r]].tolist()), tuple(src[rows[r], j].tolist()), int(_CUBE_CODES[j])) for r, j in picks])

    if len(rising) > 1:
        local_here, local_src = here[rows], src[rows]
        drawn = []
        for r in rising:
            ok = np.nonzero(g[r] > LEVEL_TOL)[0]
            weights = np.exp((g[r, ok] - g[r, ok].max()) / TEMPERATURE)
            drawn.append((r, int(ok[rng.choice(len(ok), p=weights / weights.sum())])))
        half = [(r, int(best_j[r])) for r in rising if rng.random() < 0.5]
        for picks in ([(r, int(best_j[r])) for r in rising], drawn, half)[: max(m - 3, 1)]:
            kept = _spaced(picks, local_here, local_src)
            if len(kept) > 1:
                out.append(candidate(kept))
    for r in order[:2]:
        out.append(candidate([(int(r), int(best_j[r]))]))
    r = int(rng.integers(len(rows)))
    ok = np.nonzero(g[r] >= -LEVEL_TOL)[0]
    out.append(candidate([(r, int(ok[rng.integers(len(ok))]))]))
    return out


def _earlier(world, cands):
    """The candidates' earlier states and action codes, batched."""
    episode = torch.tensor([c.episode for c in cands])
    sub = world.index(episode)
    mass = sub.mass.clone()
    pos = sub.cheese_pos.clone()
    codes = torch.full(mass.shape, STAY, dtype=torch.long)
    for i, c in enumerate(cands):
        for h, s, code in c.moves:
            mass[i, h[0], h[1], h[2]] = 0.0
        for h, s, code in c.moves:
            mass[i, s[0], s[1], s[2]] = 1.0
            codes[i, s[0], s[1], s[2]] = code
        if c.push is not None:
            slot, e, pressers = c.push
            pos[i, slot] -= e
            for r in pressers:
                codes[i, r[0], r[1], r[2]] = code_of(e)
    return replace(sub, mass=mass, cheese_pos=pos), codes


def rule_ok(world, codes):
    """(B,) bool: world.refuse(world, codes, RULE) leaves codes unchanged, computed for the robots
    that move only. world.refuse builds the whole rule table, whose overhang test (world.overhangs)
    floods support once per movable robot whose removal unsupports another; here it is run for the
    moving robots alone, with the same test (fewer robots supported with the mover at its target
    than before, for a mover whose removal unsupports another; a mover with no robot on any face
    unsupports nobody). A test checks the two agree."""
    b, d, h, w = world.mass.shape
    occ = world.mass > 0.5
    ok = torch.ones(b, dtype=torch.bool)
    scene, z, y, x = (occ & (codes != STAY)).nonzero(as_tuple=True)
    if scene.numel() == 0:
        return ok
    ids = W.cheese_id_map(world)
    code = codes[scene, z, y, x].long()
    here = torch.stack([z, y, x], dim=1)
    there = here + DIR_TABLE[code]
    flat, inside = W.flat_index(there, (d, h, w))
    if world.friction is not None:
        press = W.chain_press(world, codes, ids)[scene, z, y, x] >= 0
    else:
        press = inside & (ids.reshape(b, -1)[scene, flat] >= 0)
    rows, pos = torch.unique(scene, return_inverse=True)
    sub = world.index(rows)
    cheese = ids[rows] >= 0
    table, side = W.pivot_moves(occ[rows], sub.walls, cheese, release=RULE == "switch_cube", sides=True)
    if sub.stuck is not None:
        table[:, :STAY] &= ~sub.stuck.unsqueeze(1)
    can = table[pos, code, z, y, x]
    faces = here.unsqueeze(1) + torch.tensor(_FACE_OFFSETS).unsqueeze(0)
    face_flat, face_inside = W.flat_index(faces, (d, h, w))
    lonely = ~(occ.reshape(b, -1)[scene.unsqueeze(1), face_flat] & face_inside).any(dim=1)  # no robot on a face: no other robot's support runs through it
    need = can & ~press & world.gravity[scene] & ~lonely
    if need.any():
        m = need.nonzero().squeeze(1)
        held = W.supported(sub, occ[rows])
        before = held.flatten(1).sum(dim=1)
        lifted = occ[scene[m]].clone()
        lifted[torch.arange(len(m)), z[m], y[m], x[m]] = False
        critical = W.supported(world.index(scene[m]), lifted).flatten(1).sum(dim=1) < before[pos[m]] - held[pos[m], z[m], y[m], x[m]].long()
        critical &= inside[m]
        if critical.any():
            k = critical.nonzero().squeeze(1)
            moved = lifted[k]
            t = there[m[k]]
            moved[torch.arange(len(k)), t[:, 0], t[:, 1], t[:, 2]] = True
            after = W.supported(world.index(scene[m[k]]), moved).flatten(1).sum(dim=1)
            can[m[k[after < before[pos[m[k]]]]]] = False
    pivots = can & ~press
    moving = torch.zeros(b, d * h * w, dtype=torch.bool)
    moving[scene[pivots], W.flat_index(here[pivots], (d, h, w))[0]] = True
    lean_flat, lean_inside = W.flat_index(W.leans(here, code, side[pos, code, z, y, x]), (d, h, w))
    unsteady = (moving[scene.unsqueeze(1), lean_flat] & lean_inside).any(dim=1)
    good = (pivots & ~unsteady) | press
    ok[scene[~good]] = False
    return ok


def verify(earlier, codes, later):
    """(C,) bool: earlier is at rest, the rule lets every move in codes happen (rule_ok, the same
    answer as world.refuse), and the forward step takes earlier exactly to later. When the rule
    refuses nothing, world.step with the rule and without it are the same step."""
    ok = rule_ok(earlier, codes)
    if ok.any():
        idx = ok.nonzero().squeeze(1)
        sub = earlier.index(idx)
        nxt, _ = W.step(sub, codes[idx], rule=None)
        ok[idx] = same_state(nxt, later.index(idx)) & same_state(W.fall(sub), sub)
    return ok


def _key(world, b):
    occ = (world.mass[b] > 0.5).flatten().nonzero().squeeze(1)
    return occ.numpy().tobytes() + world.cheese_pos[b].numpy().tobytes()


# ---------------------------------------------------------------------------------------------
# Scaffolding: helpers that hold a part up while it is placed


def _pivot_options():
    """[(code, step, supports, down, swing)]: every pivot pivot_moves knows, with the cells it
    leans on and sweeps through relative to the moving cube (world.pivot_moves)."""
    out = []
    for code, step in enumerate(DIRS):
        if not bool(CUBE_MOVES[code]):
            continue
        if sum(v != 0 for v in step) == 1:
            pivots = [(step, s, False) for s in W._SIDES if not any(a * b for a, b in zip(s, step))]
        elif step[0] > 0:
            pivots = [(W._UP, (0, step[1], step[2]), True)]
        else:
            pivots = [((0, step[1], step[2]), W._DOWN, True)]
        for h, s, half in pivots:
            neg = tuple(-v for v in s)
            swing = [h, neg, W._add(h, s, -1)] + ([W._add(h, s), W._add(h, h), W._add(W._add(h, h), s)] if half else [])
            supports = [s] if half else [s, W._add(h, s)]
            out.append((code, np.array(step), [np.array(c) for c in supports], s == W._DOWN, [np.array(c) for c in swing]))
    return out


_OPTIONS = _pivot_options()


def _inside(c):
    return all(0 <= int(v) < SIZE for v in c)


def _held(one, occ):
    """Every robot of the one-scene occupancy occ (D, H, W) supported."""
    held = W.supported(one, occ.unsqueeze(0))[0]
    return bool((held | ~occ).all())


def _reverse_path(one, p, q, depth=PATH_DEPTH, extra=RANK_EXTRA):
    """Cells [p, ..., q]: a walk in reverse time of the robot at p to q, one cube move per step, the
    rest of the scene still (breadth first over sources(); each cell it stops in face to face with
    something that holds it). q is a cell, a (D, H, W) bool mask (the first cell of it reached), or
    a (D, H, W) float rank (inf: not a destination): the reached cell of lowest rank, nearest first,
    looking at most `extra` moves past the first destination found. None when nothing is reached."""
    rank = None
    if isinstance(q, torch.Tensor):
        rank = torch.where(q, 0.0, float("inf")) if q.dtype == torch.bool else q
        best_possible = float(rank.min())
        if best_possible == float("inf"):
            return None
    occ = one.mass[0] > 0.5
    rest = occ.clone()
    rest[p] = False
    cheese = W.cheese_id_map(one)[0] >= 0
    padded = padded_grids(one, rest.unsqueeze(0))
    held = W.supported(one, rest.unsqueeze(0))[0]
    anchor = W._grow_faces((held | one.walls[0] | cheese).unsqueeze(0), torch.zeros(1, *occ.shape, dtype=torch.bool))[0]
    anchor[0] = True
    anchor &= ~(rest | one.walls[0] | cheese)
    parent = {p: None}
    frontier = [p]
    best, stop = None, depth

    def walk(y):
        path = [y]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        return path[::-1]

    for level in range(depth):
        if level >= stop:
            break
        here = torch.tensor(frontier)
        src, legal = sources(padded, torch.zeros(len(frontier), dtype=torch.long), here, remove_self=False)
        sc = src.clamp(0, SIZE - 1)
        legal &= anchor[sc[..., 0], sc[..., 1], sc[..., 2]]
        if rank is not None:
            found = rank[sc[..., 0], sc[..., 1], sc[..., 2]]
        nxt = []
        for n, k in legal.nonzero().tolist():
            y = tuple(src[n, k].tolist())
            if y in parent:
                continue
            parent[y] = frontier[n]
            if rank is None:
                if y == q:
                    return walk(y)
            else:
                r = float(found[n, k])
                if r < float("inf") and (best is None or r < best[0]):
                    best = (r, y)
                    if r <= best_possible:
                        return walk(y)
                    stop = min(stop, level + 1 + extra)
            nxt.append(y)
        if not nxt:
            break
        frontier = nxt
    return walk(best[1]) if best is not None else None


def _path_steps(one, path):
    """The reverse steps that walk the robot along path (verified), [(earlier mass (1, D, H, W),
    codes (1, D, H, W))], or None if any fails."""
    occ = one.mass[0] > 0.5
    base = occ.clone()
    base[path[0]] = False
    n = len(path) - 1
    later = one.index([0] * n)
    later_mass = base.unsqueeze(0).repeat(n, 1, 1, 1)
    earlier_mass = later_mass.clone()
    codes = torch.full(later_mass.shape, STAY, dtype=torch.long)
    for k in range(n):
        later_mass[(k, *path[k])] = True
        earlier_mass[(k, *path[k + 1])] = True
        codes[(k, *path[k + 1])] = code_of(np.subtract(path[k], path[k + 1]))
    later = replace(later, mass=later_mass.float())
    earlier = replace(later, mass=earlier_mass.float())
    if not bool(verify(earlier, codes, later).all()):
        return None
    return [(earlier_mass[k : k + 1].float(), codes[k : k + 1]) for k in range(n)]


# ---------------------------------------------------------------------------------------------
# The swarm: many robots walking at once


@dataclass
class _Walker:
    cell: tuple  # where the robot is now (reverse time)
    path: list  # cells from `cell` to its destination
    wait: int = 0  # rounds in a row it could not move


def _far(cell, cells, spacing=WALKER_SPACING):
    return all(max(abs(a - b) for a, b in zip(cell, c)) >= spacing for c in cells)


def _joint(state, moves):
    """The earlier state (1, D, H, W) and codes of robots at `here` moved back to `src`, [(here, src)]."""
    mass = state.mass.clone()
    codes = torch.full(mass.shape, STAY, dtype=torch.long)
    for here, src in moves:
        mass[(0, *here)] = 0.0
    for here, src in moves:
        mass[(0, *src)] = 1.0
        codes[(0, *src)] = code_of(np.subtract(here, src))
    return mass, codes


def _ahead(state, moves):
    """The later state (1, D, H, W) mass of robots at `cell` moved on to `nxt`, [(cell, nxt)], and the codes
    on the earlier state (forward time)."""
    mass = state.mass.clone()
    codes = torch.full(mass.shape, STAY, dtype=torch.long)
    for cell, nxt in moves:
        mass[(0, *cell)] = 0.0
        codes[(0, *cell)] = code_of(np.subtract(nxt, cell))
    for cell, nxt in moves:
        mass[(0, *nxt)] = 1.0
    return mass, codes


def _verify_sets(state, sets, forward=False):
    """(len(sets),) bool: each set of walker moves verified as one joint step: reverse time, [(here, src)]
    the robots at `here` came from `src` (state is the later state); forward, [(cell, nxt)] the robots at
    `cell` move on to `nxt` (state is the earlier state)."""
    if forward:
        built = [_ahead(state, moves) for moves in sets]
        earlier = state.index([0] * len(sets))
        later = replace(earlier, mass=torch.cat([m for m, _ in built]))
        return verify(earlier, torch.cat([c for _, c in built]), later)
    built = [_joint(state, moves) for moves in sets]
    later = state.index([0] * len(sets))
    earlier = replace(later, mass=torch.cat([m for m, _ in built]))
    return verify(earlier, torch.cat([c for _, c in built]), later)


def _open_rank(rank_of, state, reserved):
    """rank_of(state, reserved cells as a (D, H, W) mask) with taken and reserved cells removed."""
    mask = torch.zeros(state.mass.shape[1:], dtype=torch.bool)
    for r in reserved:
        mask[r] = True
    rank = rank_of(state, mask).clone()
    rank[(state.mass[0] > 0.5) | mask] = float("inf")
    return rank


def _swarm(one, movers, rank_of, rng, budget, stranded=None, extra=RANK_EXTRA, until=None, once=False, forward=False, first=False, patient=False):
    """Robots that must move (movers (D, H, W) bool: the cells they stand on) walk out together in
    reverse time, each to a destination of lowest rank_of(state, reserved) ((D, H, W) float, inf:
    none; reserved: (D, H, W) bool, the cells walkers are heading for).
    Played forward, the swarm walks in from where the robots stand and builds bottom up.

    Each round: robots that must move and have a legal move are recruited highest first (at most
    RECRUIT_TRIES breadth-first walks a round), WALKER_SPACING from every walker (with `first`: those
    whose first move verifies on its own come first, one that does not only in a round that recruits
    no other; dense take-aparts leave pieces hanging from robots above them, whose walkers would
    otherwise hold the round); each walker's destination is reserved. patient: a round in which
    nobody walks and nobody could be recruited (the robots tried were blocked for RETRY rounds)
    does not end the swarm while robots are left; only IDLE_ROUNDS idle rounds do. Every walker then takes the next move of its walk; the joint step is
    verified, and when it fails, the walkers that verify alone join in order, a group at a time,
    while the joint step still verifies (a group that breaks it is split in halves). A walker that waits WAIT_REPLAN
    rounds plans again from where it stands; after WAIT_DROP it gives up (a robot left on the way
    joins `stranded` and is recruited again like a mover). Stops when nothing is left to move, no
    robot can be recruited and no walker moves for IDLE_ROUNDS rounds, after `budget` steps, or
    when until(state) holds (checked every UNTIL_EVERY rounds). once: a robot that has walked is
    not recruited again (movers is a set of cells: without it, a robot that stops on a cell another
    mover left would walk again).
    forward: the same swarm in forward time (the robots walk out of `one` to their destinations, every
    joint step verified from the earlier state; a single pivot runs both ways through the same cells,
    so the walks of _reverse_path serve both directions). Played forward: a take-apart, top down.
    Returns (steps [(earlier mass, codes)] in the order made (reverse time: from `one` backwards;
    forward: from `one` on), the state reached, stranded cells (set))."""
    state = one
    steps, walkers = [], []
    stranded = set() if stranded is None else set(stranded)
    blocked, reserved, arrived = {}, set(), set()  # arrived: with once, cells a walker ended in (not recruited again)
    idle, t = 0, 0
    while len(steps) < budget:
        t += 1
        if until is not None and t % UNTIL_EVERY == 0 and until(state):
            break
        occ = state.mass[0] > 0.5
        stranded = {c for c in stranded if bool(occ[c])}
        recruited = 0
        if len(walkers) < WALKERS:
            near = [w.cell for w in walkers] + [w.path[1] for w in walkers if len(w.path) > 1]
            walking = {w.cell for w in walkers}
            pool = [tuple(c) for c in (occ & movers).nonzero().tolist()] + sorted(stranded)
            pool = [c for c in dict.fromkeys(pool) if c not in walking and c not in arrived and blocked.get(c, 0) <= t and _far(c, near)]
            pool.sort(key=lambda c: (-c[0], rng.random()))
            pool = pool[: 4 * RECRUIT_TRIES]
            if pool:
                _, legal = sources(padded_grids(state), torch.zeros(len(pool), dtype=torch.long), torch.tensor(pool))
                pool = [c for c, ok in zip(pool, legal.any(dim=1).tolist()) if ok]
            rank = None
            tries = 0
            later = []  # walks whose first move does not verify on its own (the robot holds up one hanging below it): taken only in a round that finds no other
            for c in pool:
                if tries >= RECRUIT_TRIES or len(walkers) >= WALKERS:
                    break
                if not _far(c, near):
                    continue
                tries += 1
                if rank is None:
                    rank = _open_rank(rank_of, state, reserved)
                path = _reverse_path(state, c, rank, extra=extra)
                if path is None or len(path) < 2:
                    blocked[c] = t + RETRY
                    continue
                if first and not bool(_verify_sets(state, [[(c, path[1])]], forward)[0]):
                    later.append((c, path))
                    continue
                walkers.append(_Walker(c, path))
                reserved.add(path[-1])
                rank = None
                near += [c, path[1]]
                stranded.discard(c)
                recruited += 1
            for c, path in later:
                if recruited or path[-1] in reserved or len(walkers) >= WALKERS or not _far(c, near):  # a walk that verifies came first this round: this one waits
                    blocked[c] = t + RETRY
                    continue
                walkers.append(_Walker(c, path))
                reserved.add(path[-1])
                near += [c, path[1]]
                stranded.discard(c)
        props, nexts = [], set()
        for w in walkers:
            nx = w.path[1]
            if bool(occ[nx]) or nx in nexts:
                continue
            nexts.add(nx)
            props.append(w)
        accepted = []
        if props:
            moves = [(w.cell, w.path[1]) for w in props]
            if bool(_verify_sets(state, [moves], forward)[0]):
                accepted = props
            else:
                alone = _verify_sets(state, [[m] for m in moves], forward).tolist()
                good = [w for w, ok in zip(props, alone) if ok]
                accepted = good[:1]

                def add(group):  # greedy bisection: a group joins when the joint step still verifies, else its halves try
                    if not group:
                        return
                    if bool(_verify_sets(state, [[(w.cell, w.path[1]) for w in accepted + group]], forward)[0]):
                        accepted.extend(group)
                    elif len(group) > 1:
                        add(group[: len(group) // 2])
                        add(group[len(group) // 2 :])

                add(good[1:])
        if accepted:
            if forward:
                mass, codes = _ahead(state, [(w.cell, w.path[1]) for w in accepted])
                steps.append((state.mass.clone(), codes))
            else:
                mass, codes = _joint(state, [(w.cell, w.path[1]) for w in accepted])
                steps.append((mass, codes))
            state = replace(state, mass=mass)
            for w in accepted:
                w.cell, w.path, w.wait = w.path[1], w.path[1:], -1
            idle = 0
        else:
            idle += 1
        kept = []
        for w in walkers:
            w.wait += 1
            if len(w.path) == 1:  # arrived
                reserved.discard(w.cell)
                if once:
                    arrived.add(w.cell)
                continue
            if w.wait >= WAIT_DROP:
                reserved.discard(w.path[-1])
                blocked[w.cell] = t + RETRY
                stranded.add(w.cell)
                continue
            if w.wait >= WAIT_REPLAN and w.wait % WAIT_REPLAN == 0:
                reserved.discard(w.path[-1])
                path = _reverse_path(state, w.cell, _open_rank(rank_of, state, reserved), extra=extra)
                if path is not None and len(path) > 1:
                    w.path = path
                reserved.add(w.path[-1])
            kept.append(w)
        walkers = kept
        if not walkers and not recruited and not accepted:  # nothing walking: stop (patient: only when no robot is left to try)
            occ = state.mass[0] > 0.5
            left = [tuple(c) for c in (occ & movers).nonzero().tolist()] + [c for c in stranded if bool(occ[c])]
            if not patient or not [c for c in left if c not in arrived]:
                break
        if idle > IDLE_ROUNDS:
            break
    return steps, state, stranded


def _drop_rank(one, target, dist, cap):
    """(D, H, W) float: where a robot taken out of a structure may stop, by rank (inf: nowhere). 0 on
    the floor or terrain at least DROP_DIST from the goal's cells touching no robot (face, edge or
    corner); 1 on top of a robot standing alone on the ground that far out (natural scenes stack);
    2 anywhere on the ground off the goal and its faces; 3 on anything (robot, terrain, floor) off
    the goal's cells, less than `cap` above the terrain (a heap as natural_scenes piles it), the
    lower the better. Never under the goal's cells (under a table top, in a doorway's lintel's
    shadow), where a robot would close in what is still to come out."""
    rank = torch.full(one.mass.shape[1:], float("inf"))
    for tier in (3, 2, 1, 0):
        rank[_drop_cells(one, target, dist, tier, cap)] = float(tier)
    lift = torch.arange(SIZE).view(SIZE, 1, 1) - ground(one.walls[0]).unsqueeze(0)
    return torch.where(rank == 3.0, 3.0 + 0.1 * lift.float(), rank)


def _drop_cells(one, target, dist=None, tier=0, cap=3):
    """(D, H, W) bool: the cells of _drop_rank's tier."""
    occ = one.mass[0] > 0.5
    dist = distance_field(target) if dist is None else dist
    free = ~(W.cheese_id_map(one)[0] >= 0) & ~occ & ~one.walls[0]
    ground_cells = on_ground(one)[0]
    if tier == 0:
        return free & ground_cells & (dist >= DROP_DIST) & ~_dilate(occ)
    if tier == 1:
        beside = W._grow_faces(occ.unsqueeze(0), torch.zeros(1, *occ.shape, dtype=torch.bool))[0]  # cells with a robot on a face
        lone = occ & ground_cells & ~beside
        on_lone = torch.zeros_like(occ)
        on_lone[1:] = lone[:-1]
        return free & on_lone & (dist >= DROP_DIST)
    under = target.flip(0).cumsum(0).flip(0) > 0  # cells with a goal cell above them (or on one)
    if tier == 2:
        return free & ground_cells & (dist >= 1.5) & ~under
    rests = ground_cells.clone()
    rests[1:] |= occ[:-1] | one.walls[0, :-1]
    lift = torch.arange(SIZE).view(SIZE, 1, 1) - ground(one.walls[0]).unsqueeze(0)
    return free & rests & ~under & (lift < cap)


def _neighbours(occ):
    """(H, W) float: robots within two columns (Chebyshev) of each column, its own included."""
    col = occ.sum(dim=0).float()[None, None]
    return torch.nn.functional.avg_pool2d(col, 5, stride=1, padding=2, count_include_pad=True)[0, 0] * 25


def _spread_movers(state, target, band):
    """(D, H, W) bool: robots that put the start out of band, by statistic: those with more others
    within two columns than natural scenes have per robot (crowd), those not on the floor or terrain
    (lifted), those higher than the band (height), those face to face with another (touch)."""
    occ = state.mass[0] > 0.5
    stats = start_stats(state, target.unsqueeze(0))[0]
    high = band[1]
    out = torch.zeros_like(occ)
    if stats[STATS.index("crowd")] > high[STATS.index("crowd")]:
        out |= occ & ((_neighbours(occ) - 1) > float(high[STATS.index("crowd")])).unsqueeze(0)
    if stats[STATS.index("lifted")] > high[STATS.index("lifted")]:
        out |= occ & ~on_ground(state)[0]
    lift = torch.arange(SIZE).view(SIZE, 1, 1) - ground(state.walls[0]).unsqueeze(0)
    if stats[STATS.index("height")] > high[STATS.index("height")]:
        out |= occ & (lift > high[STATS.index("height")])
    if stats[STATS.index("touch")] > high[STATS.index("touch")]:
        out |= occ & W._grow_faces(occ.unsqueeze(0), torch.zeros(1, *occ.shape, dtype=torch.bool))[0]
    return out


def _spread_rank(target, cap, band):
    """rank_of for the spreading swarm once the goal is clear: resting cells (on the floor, terrain or
    a robot, less than cap above the terrain) where a robot would have fewer others within two
    columns than the crowd band less two: alone on the ground first, then touching another, then
    on top of one; the goal's cells too while robots on them stay under the fill band (natural
    scenes drop robots there as anywhere). The nearest of the best kind wins (_reverse_path)."""
    limit = float(band[1][STATS.index("crowd")])
    on_goal = float(band[1][STATS.index("fill")]) * float(target.sum())

    def rank_of(state, reserved):
        occ = state.mass[0] > 0.5
        taken = occ | reserved
        near = _neighbours(taken).unsqueeze(0).expand_as(occ)
        walls = state.walls[0]
        free = ~taken & ~walls & ~(W.cheese_id_map(state)[0] >= 0)
        rests = on_ground(state)[0].clone()
        rests[1:] |= occ[:-1] | walls[:-1]
        lift = torch.arange(SIZE).view(SIZE, 1, 1) - ground(walls).unsqueeze(0)
        ok = free & rests & (lift < cap) & (near < limit - 2)
        if int((taken & target).sum()) >= on_goal - 1:
            ok &= ~target
        touch = torch.zeros(occ.shape)
        solid = taken.float()
        touch[1:] += solid[:-1]
        touch[:-1] += solid[1:]
        touch[:, 1:] += solid[:, :-1]
        touch[:, :-1] += solid[:, 1:]
        touch[:, :, 1:] += solid[:, :, :-1]
        touch[:, :, :-1] += solid[:, :, 1:]
        rank = (touch > 0).float() + 2.0 * (lift > 0).float()  # alone on the ground, then touching, then lifted
        return torch.where(ok, rank, torch.full_like(rank, float("inf")))

    return rank_of


LEVEL_DRAWS = 64  # natural scenes drawn to pick the column profile the levelling swarm walks toward


def _level_plan(state, target, cap, band, rng, draws=LEVEL_DRAWS, unbiased=False):
    """(H, W) long: robots per column of a natural scene (natural_scenes: as many robots, the same
    terrain) whose statistics lie in the band, the one of `draws` that leaves the fewest robots to
    move from the current state (robots above the scene's count in their column). Walking the
    surplus to the columns short of it makes the start that natural scene, column by column: at
    500-800 robots on 16 x 16 a carpet two to four high with a few empty columns, the goal's cells
    covered as often as any others. unbiased (planning from the goal itself): only scenes that fill
    no more of the goal's cells than the median drawn scene, so the choice does not lean the start
    toward the goal. None when no drawn scene lies in the band."""
    occ = state.mass[0] > 0.5
    n = int(occ.sum())
    scenes = natural_scenes(state.walls[0], n, rng, draws, avoid=W.cheese_id_map(state)[0] >= 0)
    ref = replace(state.index([0] * draws), mass=scenes.float())
    stats = start_stats(ref, target.unsqueeze(0).expand(draws, -1, -1, -1))
    ok = ((stats >= band[0]) & (stats <= band[1])).all(dim=1)
    if not ok.any():
        return None
    if unbiased:
        fill = stats[:, STATS.index("fill")]
        ok &= fill <= fill[ok].median()
    counts = scenes.sum(dim=1)
    surplus = (occ.sum(dim=0).unsqueeze(0) - counts).clamp(min=0).flatten(1).sum(dim=1).float()
    surplus[~ok] = float("inf")
    return counts[int(surplus.argmin())]


def _plan_movers(target, walls, profile, cap):
    """(D, H, W) bool, natural mode with a planned profile: the goal's cells, and anything at
    heap_cap or higher. A goal robot below its column's count in the profile on a column of goal
    robots (or terrain) down to the ground stays where it is: it is already where the natural start
    has a robot. One over a gap moves (the cells under it would be a gap a cube cannot enter)."""
    lift = torch.arange(SIZE).view(SIZE, 1, 1) - ground(walls).unsqueeze(0)
    column = (target | walls).long().cumprod(dim=0).bool()  # goal or terrain all the way down
    stay = target & column & (lift < profile.unsqueeze(0)) & (lift >= 0)
    return (target & ~stay) | (lift >= cap)


def _plan_rank(profile, movers, target, dist, cap):
    """rank_of for a natural take-apart with a planned profile: _profile_rank's cells (the lowest
    layer first: the carpet grows a layer at a time, so robots still walk over a level surface),
    never under a robot still to come out nor touching one (face, edge or corner: the carpet would
    close in the goal's lower layers before they leave), never on the cells robots come out of
    (the levelling swarm fills those once the goal is clear); any other drop cell (_drop_rank) after
    all of them."""

    def rank_of(state, reserved):
        occ = state.mass[0] > 0.5
        left = occ & movers
        under = (left.flip(0).cumsum(0).flip(0) > 0) | _dilate(left) | movers  # the goal's own cells wait for the levelling swarm (a robot put there would have to come out again)
        rank = _profile_rank(state, reserved, profile, cap, under)
        return torch.where(torch.isinf(rank), _drop_rank(state, target, dist, cap) + 100.0, rank)

    return rank_of


def _level_movers(state, profile, cap):
    """(D, H, W) bool: robots at or above their column's count in the profile, or at heap_cap or
    higher above the terrain."""
    occ = state.mass[0] > 0.5
    lift = torch.arange(SIZE).view(SIZE, 1, 1) - ground(state.walls[0]).unsqueeze(0)
    return occ & ((lift >= profile.unsqueeze(0)) | (lift >= cap))


def _side_free(free):
    """(D, H, W) long: horizontal face neighbours of each cell that are in `free` (off the grid: not)."""
    out = torch.zeros(free.shape, dtype=torch.long)
    out[:, 1:] += free[:, :-1].long()
    out[:, :-1] += free[:, 1:].long()
    out[:, :, 1:] += free[:, :, :-1].long()
    out[:, :, :-1] += free[:, :, 1:].long()
    return out


def _profile_rank(state, reserved, profile, cap, under=None):
    """(D, H, W) float: resting cells below their column's count in the profile and below heap_cap
    (not under `under`), ranked by layer, the lowest first, and in a layer: a cell whose filling
    would close in a neighbour still short of the profile half a layer later, then the most
    enclosed first. A cube enters a cell only from a free cell beside it on the same layer (a
    straight pivot leans on the two cells below, a diagonal one sweeps the cell beyond), so a cell
    with all four sides filled can no longer be filled; filling round it first would leave pits
    natural scenes have and the walk cannot make. inf: not a destination."""
    occ = state.mass[0] > 0.5
    walls = state.walls[0]
    rests = on_ground(state)[0].clone()
    rests[1:] |= occ[:-1] | walls[:-1]
    lift = torch.arange(SIZE).view(SIZE, 1, 1) - ground(walls).unsqueeze(0)
    free = ~occ & ~reserved & ~walls & ~(W.cheese_id_map(state)[0] >= 0)
    short = free & (lift < profile.unsqueeze(0)) & (lift < cap)
    room = _side_free(free)
    dying = short & (room == 1)  # one free side left: the cell that fills it closes it in
    kills = _side_free(dying)
    ok = short & rests
    if under is not None:
        ok &= ~under
    rank = lift.float() + 0.5 * (kills > 0).float() + 0.05 * room.float()
    return torch.where(ok, rank, torch.full(occ.shape, float("inf")))


def _level_rank(profile, cap):
    """rank_of for the levelling swarm: resting cells below the profile's count of their column and
    below heap_cap, the lowest layer first, closing in no cell still short (_profile_rank); nearest
    first within a rank (_reverse_path)."""

    def rank_of(state, reserved):
        return _profile_rank(state, reserved, profile, cap)

    return rank_of


def _enclosure(start, walls):
    """(D, H, W) long: faces of each cell that A (start), the terrain or the floor closes."""
    solid = start | walls
    out = torch.zeros(start.shape, dtype=torch.long)
    out[0] += 1
    out[1:] += solid[:-1].long()
    out[:-1] += solid[1:].long()
    out[:, 1:] += solid[:, :-1].long()
    out[:, :-1] += solid[:, 1:].long()
    out[:, :, 1:] += solid[:, :, :-1].long()
    out[:, :, :-1] += solid[:, :, 1:].long()
    return out


CLOSE_PENALTY = 4.0  # fill rank added for a cell whose filling closes in an empty neighbour of A (less than a layer, 8)


def _closing(state, cells, empty, reserved=None):
    """(len(cells),) long: 1 when filling the cell would close in an empty neighbour (a face
    neighbour in `empty`, (D, H, W) bool): a cube can pivot between that neighbour and a free cell
    now and could not with the cell filled (sources on the 7^3 crops, batched over the cells;
    pivots run both ways through the same cells); 2 when that neighbour is `reserved` (a walker is
    on its way there); 0 otherwise."""
    out = torch.zeros(len(cells), dtype=torch.long)
    pairs = []
    for k, c in enumerate(cells):
        for f in _FACE_OFFSETS:
            n = (c[0] + f[0], c[1] + f[1], c[2] + f[2])
            if _inside(n) and bool(empty[n]):
                pairs.append((k, n))
    if not pairs:
        return out
    cubes, walls, cheese = padded_grids(state)
    ks = torch.tensor([k for k, _ in pairs])
    ns = torch.tensor([n for _, n in pairs])
    _, before = sources((cubes, walls, cheese), torch.zeros(len(pairs), dtype=torch.long), ns, remove_self=False)
    filled = cubes.repeat(len(cells), 1, 1, 1)
    at = torch.tensor(cells) + _R
    filled[torch.arange(len(cells)), at[:, 0], at[:, 1], at[:, 2]] = True
    _, after = sources((filled, walls.expand_as(filled), cheese.expand_as(filled)), ks, ns, remove_self=False)
    closes = before.any(dim=1) & ~after.any(dim=1)
    weight = torch.ones(len(pairs), dtype=torch.long)
    if reserved is not None:
        weight += reserved[ns[:, 0], ns[:, 1], ns[:, 2]].long()
    out.scatter_reduce_(0, ks[closes], weight[closes], "amax")
    return out


def _fill_rank(start, enclosure, deferred=None, strict=True):
    """rank_of for the guided walk: A's empty cells, lowest layer first, and in it the cells whose
    filling closes in no empty neighbour (_closing) before those that do, then the most enclosed
    first (the reverse of a top down, outside in take-apart, so no cell of A is closed in before it
    is filled); a cell that would close in a cell a walker is heading for waits until it is filled.
    strict: only the lowest layer of A's empty cells (deferred cells, opened by _dig_region, come
    after every other)."""
    deferred = torch.zeros_like(start) if deferred is None else deferred

    def rank_of(state, reserved=None):
        empty = start & ~(state.mass[0] > 0.5)
        rank = torch.full(start.shape, float("inf"))
        if not empty.any():
            return rank
        z = torch.arange(SIZE).view(SIZE, 1, 1).expand_as(start).float()
        rank[empty] = (z * 8 + (6 - enclosure).float())[empty]
        rank[empty & deferred] += 1000.0
        live = empty & ~deferred
        if strict:
            pool = live if live.any() else empty
            low = int(pool.nonzero()[:, 0].min())
            rank[(z > low) | (deferred & live.any())] = float("inf")
        cells = [tuple(c) for c in (rank < float("inf")).nonzero().tolist()]
        if cells:
            closes = _closing(state, cells, empty, reserved)
            for c, shut in zip(cells, closes.tolist()):
                if shut == 1:
                    rank[c] += CLOSE_PENALTY
                elif shut == 2:  # a walker is on its way to the cell it would close in: not before it is there
                    rank[c] = float("inf")
        return rank

    return rank_of


def _dig_region(one, start, movers, r):
    """(D, H, W) bool: robots on A's cells to take out when the guided walk is stuck (a robot that
    must move has no way out, or the lowest empty cells of A cannot be reached): those in the
    columns within r (Chebyshev) of the stuck cells, from the stuck cells' level up. The swarm takes
    them out top down to parking cells and the fill puts them back bottom up: forward, the part
    round the closed-in cell comes down, the cell is done, and the part goes back up. The stuck
    robots that must move are part of the region."""
    occ = one.mass[0] > 0.5
    focus = torch.zeros_like(occ)
    cells = (occ & movers).nonzero()
    if len(cells):
        _, legal = sources(padded_grids(one), torch.zeros(len(cells), dtype=torch.long), cells)
        stuck = cells[~legal.any(dim=1)]
        focus[stuck[:, 0], stuck[:, 1], stuck[:, 2]] = True
    if not focus.any():
        empty = start & ~occ
        if not empty.any():
            return torch.zeros_like(occ)
        low = int(empty.nonzero()[:, 0].min())
        focus = empty.clone()
        focus[low + 1 :] = False
    level = int(focus.nonzero()[:, 0].min())
    cols = _near_columns(focus, r)
    region = occ & start & cols.unsqueeze(0)
    region[:level] = False
    return region | (focus & occ)


def _scaffold(one, movers, rng):
    """For a one-scene state where robots that must move (movers (D, H, W) bool) are stuck: a pivot
    into one of their cells that lacks only 1 to MAX_HELPERS support cells, those cells filled by
    other robots walking there (reverse time), then the stuck robot leaving over them. Returns the
    reverse steps [(earlier mass, codes)] in order, each verified, or None."""
    occ = one.mass[0] > 0.5
    walls = one.walls[0]
    cheese = W.cheese_id_map(one)[0] >= 0
    stuck = (occ & movers).nonzero().tolist()
    rng.shuffle(stuck)
    plans = []
    for c in stuck[:SCAFFOLD_ROBOTS]:
        c = np.array(c)
        solid = occ | walls | cheese
        solid[tuple(c)] = False
        for code, step, supports, down, swing in _OPTIONS:
            src = c - step
            if not _inside(src) or solid[tuple(src)]:
                continue
            cells = [src + o for o in swing]
            if any(cell[0] < 0 or (_inside(cell) and bool(solid[tuple(cell)])) for cell in cells):
                continue
            missing, bad = [], False
            for o in supports:
                cell = src + o
                if cell[0] < 0:
                    continue
                if not _inside(cell) or (bool(cheese[tuple(cell)]) and not down):
                    bad = True
                    break
                if not bool((occ | walls | cheese)[tuple(cell)]):
                    missing.append(tuple(int(v) for v in cell))
            if bad or not missing or len(missing) > MAX_HELPERS:
                continue
            plans.append((len(missing) + rng.random(), tuple(int(v) for v in c), code, tuple(int(v) for v in src), missing))
    plans.sort(key=lambda t: t[0])
    for _, c, code, src, missing in plans[:SCAFFOLD_PLANS]:
        steps = _try_scaffold(one, movers, c, code, src, missing, rng)
        if steps:
            return steps
    return None


def _try_scaffold(one, movers, c, code, src, helpers, rng):
    occ = one.mass[0] > 0.5
    walls = one.walls[0]
    cheese = W.cheese_id_map(one)[0] >= 0
    earlier = occ.clone()
    earlier[c] = False
    earlier[src] = True
    for q in helpers:
        earlier[q] = True
    helpers = list(helpers)
    for _ in range(MAX_HELPERS - len(helpers)):  # helpers that hang in the air get one more beside them, toward something solid
        if _held(one, earlier):
            break
        held = W.supported(one, earlier.unsqueeze(0))[0]
        loose = earlier & ~held
        ring = W._grow_faces(loose.unsqueeze(0), torch.zeros(1, *occ.shape, dtype=torch.bool))[0] & ~(earlier | walls | cheese)
        ring[c] = False
        best, best_n = None, -1
        for e in ring.nonzero().tolist():
            trial = earlier.clone()
            trial[tuple(e)] = True
            n = int((W.supported(one, trial.unsqueeze(0))[0] & trial).sum())
            if n > best_n:
                best, best_n = tuple(e), n
        if best is None:
            return None
        earlier[best] = True
        helpers.append(best)
    if not _held(one, earlier):
        return None
    order, placed, left = [], occ.clone(), list(helpers)
    while left:  # arrival order: each helper held by what is already there
        for q in left:
            trial = placed.clone()
            trial[q] = True
            if _held(one, trial):
                order.append(q)
                placed = trial
                left.remove(q)
                break
        else:
            return None
    state = one
    steps = []
    used = set()
    for q in order:
        now = state.mass[0] > 0.5
        spare = (now & ~movers).nonzero().tolist()
        spare = [tuple(p) for p in spare if tuple(p) not in used and tuple(p) != src]
        spare.sort(key=lambda p: sum(abs(a - b) for a, b in zip(p, q)) + 0.1 * rng.random())
        for p in spare[:HELPER_TRIES]:
            path = _reverse_path(state, p, q)
            if path is None:
                continue
            walk = _path_steps(state, path)
            if walk is None:
                continue
            steps += walk
            state = replace(state, mass=walk[-1][0].clone())
            used.add(q)
            break
        else:
            return None
    mass = state.mass.clone()
    mass[(0, *c)] = 0.0
    mass[(0, *src)] = 1.0
    codes = torch.full(mass.shape, STAY, dtype=torch.long)
    codes[(0, *src)] = code
    if not bool(verify(replace(state, mass=mass), codes, state)[0]):
        return None
    steps.append((mass, codes))
    return steps


# ---------------------------------------------------------------------------------------------
# Guided walks by taking apart and building again (reverse-v4.md)

REBUILD_ROBOTS = 115  # guided goals with more robots than this walk through _rebuild first
REBUILD_DIGS = 10  # times the kept part may shrink round robots that cannot get out
REBUILD_HEAP = 2  # the meeting heap may pile this much higher than a natural heap (it is never a start)
REBUILD_TRIES = 4  # swarms per side while robots are left on the way (stranded)


def _kept(one, cells):
    """(D, H, W) bool: the robots of `cells` that hold themselves up (connected through each other
    to the floor, the terrain or an object)."""
    return W.supported(one, cells.unsqueeze(0))[0] & cells


def _unkeep(one, keep, stuck, r):
    """The kept part less the robots in the columns within r (Chebyshev) of the stuck cells, from
    their lowest level up, and less what then no longer holds itself up: a local dig round robots
    that cannot get out (forward: that part comes down and goes up again with the rest)."""
    region = keep & _near_columns(stuck, r).unsqueeze(0)
    region[: int(stuck.nonzero()[:, 0].min())] = False
    return _kept(one, keep & ~region)


def _arrivals(one, steps, heap):
    """(D, H, W) float: for each heap cell, the step of the reverse walk `steps` [(earlier mass, ...)]
    from `one` after which a robot stood there to the end (inf off the heap)."""
    since = torch.full(heap.shape, float("inf"))
    held = heap.clone()  # cells still occupied in every later state
    for k in range(len(steps) - 1, -1, -1):
        prev = (steps[k - 1][0][0] > 0.5) if k > 0 else (one.mass[0] > 0.5)
        came = held & ~prev
        since[came] = float(k)
        held &= prev
    return since


def _meet_rank(order, ab, dist, cap):
    """rank_of for A's robots walking out of A (forward): the meeting heap's cells (where B's robots
    went) in the order B's robots arrived there (`order`, _arrivals: an order that walking in is
    known to manage, no cell closed in before it is filled); any other drop cell (_drop_rank) after
    all of them."""
    heap = torch.isfinite(order)
    scale = float(order[heap].max()) + 1.0 if heap.any() else 1.0

    def rank_of(state, reserved):
        occ = state.mass[0] > 0.5
        walls = state.walls[0]
        rests = on_ground(state)[0].clone()
        rests[1:] |= occ[:-1] | walls[:-1]
        rank = _drop_rank(state, ab, dist, cap) + 100.0
        ok = heap & rests & ~occ
        rank[ok] = order[ok] / scale
        return rank

    return rank_of


def _walk_out(one, movers, rank_of, rng, budget, forward):
    """The swarm (_swarm) run again while robots are left on movers' cells or on the way (stranded),
    at most REBUILD_TRIES times. Returns (steps, state, robots still on movers' cells (D, H, W) bool);
    a robot stranded elsewhere stays where it is (the join walk of _rebuild takes care of it)."""
    state, steps, stranded = one, [], set()
    for _ in range(REBUILD_TRIES):
        got, state, stranded = _swarm(state, movers, rank_of, rng, budget - len(steps), stranded, DROP_EXTRA, forward=forward, patient=True)
        steps += got
        left = (state.mass[0] > 0.5) & movers
        if not got or not (left.any() or stranded) or len(steps) >= budget:
            break
    return steps, state, left


def _rebuild(one, start, rng, budget, cap):
    """Guided walk from the goal B (the state `one`) back to the start A by taking apart and building
    again, both sides by the swarm. K, the robots of A and B in common that hold themselves up, stays
    where it is. B's other robots walk out to a heap (reverse time, as the natural take-apart: forward,
    the part of B that changes is built from the heap, bottom up); A's other robots walk out of A forward
    in time (_swarm forward) to the same heap cells (forward: the part of A that changes comes apart,
    top down, onto the heap). Where the two heaps differ, a guided walk (take_apart without rebuild)
    joins them. Robots that cannot get out shrink K round them (_unkeep, a local dig: radius 1, then
    growing slowly) and both sides walk again. Returns (steps [(earlier mass, codes, kind)] in reverse
    time from B, reached: the last earlier state is exactly A)."""
    b_occ = one.mass[0] > 0.5
    ab = start | b_occ
    dist = distance_field(ab)
    drop = lambda state, reserved: _drop_rank(state, ab, dist, cap)  # noqa: E731
    keep = _kept(one, start & b_occ)
    a_one = replace(one, mass=start.float().unsqueeze(0))
    state, steps = one, []
    for dig in range(REBUILD_DIGS + 1):
        r = 1 + dig // 3
        got, state, left = _walk_out(state, b_occ & ~keep, drop, rng, budget - len(steps), forward=False)
        steps += [(m, c, "extract") for m, c in got]
        if len(steps) >= budget:
            return steps, False
        if left.any():  # B's robots that cannot get out: dig round them
            keep = _unkeep(one, keep, left, r)
            continue
        heap = (state.mass[0] > 0.5) & ~keep
        order = _arrivals(one, steps, heap)
        fwd, x_state, a_left = _walk_out(a_one, start & ~keep, _meet_rank(order, ab, dist, cap), rng, budget - len(steps), forward=True)
        if a_left.any():
            keep = _unkeep(one, keep, a_left, r)
            continue
        x = x_state.mass[0] > 0.5
        h1 = state.mass[0] > 0.5
        if not bool((x == h1).all()):  # A's robots that ended off the heap walk on (forward) to its cells still empty
            more, x_state, _ = _walk_out(x_state, x & ~h1, _meet_rank(torch.where(h1 & ~x, order, torch.full_like(order, float("inf"))), ab, dist, 0), rng, budget - len(steps) - len(fwd), forward=True)
            fwd += more
            x = x_state.mass[0] > 0.5
        join = []
        if not bool((x == (state.mass[0] > 0.5)).all()):
            room = budget - len(steps) - len(fwd)
            if room <= 0:
                return steps, False
            sub = Goal("edit", "", replace(state, mass=state.mass.clone()), (state.mass[0] > 0.5) & ~x, start=x)
            eps, _ = take_apart([sub], rng, max_steps=room, rebuild=False, patient=True)
            ep = eps[0]
            if ep.end not in ("reached", "hold"):
                return steps, False
            if ep.end == "reached":
                join = [(ep.mass[t].float().unsqueeze(0), ep.actions[t].long().unsqueeze(0), "join") for t in range(ep.steps - 1, -1, -1)]
        out = steps + join + [(m, c, "dismantle") for m, c in reversed(fwd)]
        return out, len(out) <= budget
    return steps, False


# ---------------------------------------------------------------------------------------------
# The walk backwards


@dataclass
class Episode:
    family: str
    text: str
    world: VoxelWorld  # the start (one scene): static fields as recorded
    mass: torch.Tensor  # (T + 1, D, H, W) uint8, frame 0 the start, frame T the goal
    cheese_pos: torch.Tensor  # (T + 1, K, 3) long
    cheese_alive: torch.Tensor  # (T + 1, K) bool
    actions: torch.Tensor  # (T, D, H, W) int8: the codes recorded on frame t
    natural: bool  # the start is what was asked: a natural scene (natural mode) or exactly A (guided)
    end: str  # why the walk stopped: natural, reached, hold (nothing to do), max_steps, stalled
    goal: Goal = None
    kinds: list = None  # per step forward: "progress" (potential falls), "level", "scaffold", "push"
    stats: list = None  # start_stats of the start (natural mode)
    distance: int = -1  # guided: cells of A without a robot at the start (0: exactly A)
    touched: int = 0  # robots that move at least once

    @property
    def steps(self):
        return self.actions.shape[0]


def _ids_back(later_ids, earlier, codes):
    """Robot identities on the earlier frame: a robot that moves takes the id of the cell its move
    lands in; one that stays or presses on an object keeps its own."""
    ids = torch.full_like(later_ids, -1)
    cells = (earlier.mass[0] > 0.5).nonzero()
    code = codes[0][cells[:, 0], cells[:, 1], cells[:, 2]].long()
    there = cells + DIR_TABLE[code]
    obj = W.cheese_id_map(earlier)[0]
    inside = ((there >= 0) & (there < SIZE)).all(dim=1)
    t = there.clamp(0, SIZE - 1)
    moves = (code != STAY) & inside & (obj[t[:, 0], t[:, 1], t[:, 2]] < 0)
    src = torch.where(moves.unsqueeze(1), t, cells)
    ids[cells[:, 0], cells[:, 1], cells[:, 2]] = later_ids[src[:, 0], src[:, 1], src[:, 2]]
    return ids, set(later_ids[t[moves, 0], t[moves, 1], t[moves, 2]].tolist())


def take_apart(goals, rng, max_steps=None, candidates=CANDIDATES, scaffold=True, rebuild=True, patient=False):
    """Walk every goal backwards (module docstring) in one batch. Returns (episodes, stats): stats
    "proposed", "verified" and "accepted" are per-iteration counts of the joint greedy walk (lists);
    "backtracks" and "scaffolds" (attempts) per episode; "extract_calls" and "extract_ok" the
    extractions tried and found; "rebuilds" per episode, 1 when _rebuild reached A. rebuild: guided
    goals of more than REBUILD_ROBOTS robots walk through _rebuild (an episode it cannot finish ends
    "stalled", with no steps); rebuild=False: the fill and digs of v3 for every guided goal.
    patient: the guided fill's swarms are patient (_swarm; _rebuild's join walk sets it)."""
    world = W.concat([g.world for g in goals])
    b = len(goals)
    guided = torch.tensor([g.start is not None for g in goals])
    starts = torch.stack([g.start if g.start is not None else torch.zeros_like(g.target) for g in goals])
    targets = torch.stack([g.target for g in goals])
    static = torch.stack([natural_field(g.target) for g in goals])
    robots = [int((g.world.mass > 0.5).sum()) for g in goals]
    limit = [max_steps or min(MAX_STEPS_CAP, STEPS_BASE + STEPS_PER_ROBOT * n) for n in robots]
    phase = ["push" if g.push is not None else "extract" for g in goals]
    bands = [natural_bands(g.world, g.target, rng) if g.start is None and g.push is None else None for g in goals]
    fields = static[:, 0].clone()
    dists = [distance_field(g.target) for g in goals]
    pushes, level = [0] * b, [0] * b
    levels = [[] for _ in range(b)]
    floor = [0] * b  # recorded steps a backtrack may not undo (the push phase's)
    backtracks, scaffolds = [0] * b, [0] * b
    done, end = [False] * b, [""] * b
    seen = [{_key(world, i)} for i in range(b)]
    history = [[[(world.mass[i] > 0.5).to(torch.uint8), world.cheese_pos[i].clone(), world.cheese_alive[i].clone()]] for i in range(b)]
    codes_of = [[] for _ in range(b)]
    kinds = [[] for _ in range(b)]
    ids = []
    for i in range(b):
        grid = torch.full((SIZE,) * 3, -1, dtype=torch.long)
        cells = (world.mass[i] > 0.5).nonzero()
        grid[cells[:, 0], cells[:, 1], cells[:, 2]] = torch.arange(len(cells))
        ids.append([grid])
    moved = [set() for _ in range(b)]
    stranded = [set() for _ in range(b)]  # robots a walker left on its way (recruited again)
    caps = [heap_cap(n, g.world.walls[0]) for n, g in zip(robots, goals)]
    enclosure = [_enclosure(starts[i], world.walls[i]) for i in range(b)]
    deferred = [torch.zeros_like(starts[i]) for i in range(b)]  # cells of A a dig opened: filled last
    dists_ab = [distance_field(starts[i] | targets[i] | (goals[i].world.mass[0] > 0.5)) if goals[i].start is not None else None for i in range(b)]
    digs = [0] * b
    spreads = [0] * b
    levelled = [False] * b  # natural: the levelling swarm (_level_plan) has run
    plans = [None] * b  # natural, dense (heap_cap above HEAP): the start's planned robots per column
    for i, g in enumerate(goals):
        if bands[i] is not None and caps[i] > HEAP:
            plans[i] = _level_plan(world.index(i), targets[i], caps[i], bands[i], rng, unbiased=True)
    plan_movers = [None if plans[i] is None else _plan_movers(targets[i], world.walls[i], plans[i], caps[i]) for i in range(b)]
    too_high = [(torch.arange(SIZE).view(SIZE, 1, 1) - ground(world.walls[i]).unsqueeze(0)) >= caps[i] for i in range(b)]  # cells above heap_cap
    deferred_new = [False] * b  # a dig just opened cells: the next fill keeps them for last
    closest = [int((starts[i] & ~(world.mass[i] > 0.5)).sum()) for i in range(b)]  # guided: fewest empty cells of A so far
    rising = [0] * b  # digs since the walk last came nearer A (the radius grows with it)
    rebuilds = [0] * b
    stats = {"proposed": [], "verified": [], "accepted": [], "backtracks": backtracks, "scaffolds": scaffolds, "digs": digs, "rebuilds": rebuilds, "extract_calls": 0, "extract_ok": 0}

    def commit(i, earlier_one, codes, kind):
        levels[i].append(level[i])
        for name in ("mass", "cheese_pos", "cheese_alive"):
            getattr(world, name)[i] = getattr(earlier_one, name)[0]
        seen[i].add(_key(earlier_one, 0))
        history[i].append([(earlier_one.mass[0] > 0.5).to(torch.uint8), earlier_one.cheese_pos[0].clone(), earlier_one.cheese_alive[0].clone()])
        codes_of[i].append(codes[0].to(torch.int8))
        kinds[i].append(kind)
        grid, who = _ids_back(ids[i][-1], earlier_one, codes)
        ids[i].append(grid)
        moved[i].update(who)

    for i, g in enumerate(goals):  # nothing to do (A = B): an episode of robots holding still, built directly
        if g.start is not None and bool(((world.mass[i] > 0.5) == starts[i]).all()):
            hold = torch.full(world.mass.shape[1:], STAY, dtype=torch.int8)
            for _ in range(HOLD_STEPS):
                history[i].append([h.clone() for h in history[i][-1]])
                codes_of[i].append(hold)
                kinds[i].append("hold")
                ids[i].append(ids[i][-1])
            done[i], end[i] = True, "hold"

    for i, g in enumerate(goals):  # big guided goals: take apart and build again (_rebuild)
        if not rebuild or done[i] or g.start is None or robots[i] <= REBUILD_ROBOTS:
            continue
        planned, reached = _rebuild(world.index(i), starts[i], rng, limit[i], caps[i] + REBUILD_HEAP)
        if not reached:  # no fallback: v3's fill and digs reached none of 8 such goals and spent up to 15 CPU minutes
            done[i], end[i] = True, "stalled"
            continue
        for mass, codes_k, kind in planned:
            commit(i, replace(world.index(i), mass=mass), codes_k, kind)
        rebuilds[i] = 1
        done[i] = True
        end[i] = "reached" if bool(((world.mass[i] > 0.5) == starts[i]).all()) else "stalled"

    def finished(i):
        if goals[i].start is not None:
            return bool(((world.mass[i] > 0.5) == starts[i]).all())
        if phase[i] not in ("scatter", "spread") or (goals[i].push is not None and pushes[i] < MIN_PUSHES) or bands[i] is None:
            return False
        return in_band(start_stats(world.index(i), targets[i : i + 1])[0], bands[i])

    for _ in range(MAX_STEPS_CAP + 2 * MAX_BACKTRACKS + 64):
        for i in range(b):
            if not done[i] and len(codes_of[i]) >= limit[i]:
                done[i], end[i] = True, "max_steps"
        active = [i for i in range(b) if not done[i]]
        if not active:
            break
        for i in active:
            if phase[i] != "extract":
                continue
            one = world.index(i)
            occ_i = one.mass[0] > 0.5
            guided_i = goals[i].start is not None
            movers = ~starts[i] if guided_i else targets[i] | too_high[i]  # natural: the goal, and anything above a natural heap (helpers left where they held a part)
            if plan_movers[i] is not None:  # dense: the goal's robots below the planned profile stay
                movers = plan_movers[i]
            stranded[i] = {c for c in stranded[i] if bool(occ_i[c])}
            if not (occ_i & movers).any() and not stranded[i]:
                phase[i] = "spread" if bands[i] is not None else "scatter"
                floor[i] = len(codes_of[i])
                continue
            budget = limit[i] - len(codes_of[i])
            if guided_i:
                ranks = [_fill_rank(starts[i], enclosure[i], deferred[i]), _fill_rank(starts[i], enclosure[i], deferred[i], strict=False)]
            elif plans[i] is not None:
                ranks = [_plan_rank(plans[i], movers, targets[i], dists[i], caps[i])]
            else:
                ranks = [lambda state, reserved, i=i: _drop_rank(state, targets[i], dists[i], caps[i])]
            steps = []
            for rank_of in ranks:  # guided: lowest layer of A first; when nothing moves, any empty cell of A
                steps, _, stranded[i] = _swarm(one, movers, rank_of, rng, budget, stranded[i], RANK_EXTRA if guided_i else DROP_EXTRA, first=plans[i] is not None, patient=plans[i] is not None or (patient and guided_i))
                stats["extract_calls"] += 1
                stats["extract_ok"] += bool(steps)
                if steps:
                    break
            kind = "extract"
            if steps and not deferred_new[i]:
                deferred[i] = torch.zeros_like(deferred[i])
            deferred_new[i] = False
            if not steps and scaffold and scaffolds[i] < MAX_SCAFFOLDS + robots[i] // SCAFFOLDS_PER:
                scaffolds[i] += 1
                stuck = movers.clone()
                for c in stranded[i]:
                    stuck[c] = True
                steps, kind = _scaffold(one, stuck, rng) or [], "scaffold"
            if guided_i:
                gap = int((starts[i] & ~occ_i).sum())
                if gap < closest[i]:  # nearer A than ever: a later dig starts small again
                    closest[i], rising[i] = gap, 0
            if not steps and guided_i and digs[i] < MAX_DIGS:
                digs[i] += 1
                region = _dig_region(one, starts[i], movers, DIG_RADII[min(rising[i], len(DIG_RADII) - 1)])
                rising[i] += 1
                if region.any():
                    ab = starts[i] | (goals[i].world.mass[0] > 0.5)
                    steps, _, stranded[i] = _swarm(one, region, lambda state, reserved, ab=ab, i=i: _drop_rank(state, ab, dists_ab[i], caps[i]), rng, budget, stranded[i], patient=patient)
                    kind = "dig"
                    deferred[i] |= region
                    deferred_new[i] = True
            if not steps:
                if guided_i:
                    done[i], end[i] = True, "stalled"
                else:
                    phase[i] = "spread" if bands[i] is not None else "scatter"
                    floor[i] = len(codes_of[i])
                continue
            for mass, codes_k in steps:
                commit(i, replace(world.index(i), mass=mass), codes_k, kind)
        for i in active:  # natural, the goal clear: robots in crowded spots walk to emptier ones, the swarm again
            if phase[i] != "spread" or done[i]:
                continue
            one = world.index(i)
            steps = []
            done_spreading = lambda state, i=i: in_band(start_stats(state, targets[i : i + 1])[0], bands[i])  # noqa: E731
            if not done_spreading(one):
                if not levelled[i] and caps[i] > HEAP:  # dense (500+ robots on 16 x 16): walk toward the planned profile
                    levelled[i] = True
                    profile = plans[i] if plans[i] is not None else _level_plan(one, targets[i], caps[i], bands[i], rng)
                    if profile is not None:
                        movers = _level_movers(one, profile, caps[i])
                        steps, _, _ = _swarm(one, movers, _level_rank(profile, caps[i]), rng, limit[i] - len(codes_of[i]), extra=DROP_EXTRA, until=done_spreading, once=True, first=True, patient=True)
                    if not steps:
                        continue
                elif spreads[i] < MAX_SPREADS:
                    spreads[i] += 1
                    movers = _spread_movers(one, targets[i], bands[i])
                    if movers.any():
                        steps, _, _ = _swarm(one, movers, _spread_rank(targets[i], caps[i], bands[i]), rng, limit[i] - len(codes_of[i]), until=done_spreading, once=True)
            for mass, codes_k in steps:
                commit(i, replace(world.index(i), mass=mass), codes_k, "spread")
            if not steps:
                phase[i] = "scatter"
                floor[i] = len(codes_of[i])
        cands = []
        for i in active:
            if phase[i] == "push":
                cands += _push_candidates(world, i, goals[i])
        scatter = [i for i in active if phase[i] == "scatter" and not done[i]]
        if scatter:
            sub_idx = torch.tensor(scatter)
            sub = world.index(sub_idx)
            fields[sub_idx] = step_fields(sub, static[sub_idx], starts[sub_idx], guided[sub_idx])
            f = fields[sub_idx]
            scene, here, src, legal = single_moves(sub)
            sc = src.clamp(0, SIZE - 1)
            gain = f[scene.unsqueeze(1), sc[..., 0], sc[..., 1], sc[..., 2]] - f[scene, here[:, 0], here[:, 1], here[:, 2]].unsqueeze(1)
            must = torch.where(guided[sub_idx].view(-1, 1, 1, 1), ~starts[sub_idx], targets[sub_idx])
            mover = must[scene, here[:, 0], here[:, 1], here[:, 2]]
            free_to_go = mover & legal.any(dim=1)  # top down: of the robots that must move, only the highest that can may (forward: built bottom up)
            top = torch.full((len(scatter),), -1, dtype=torch.long).scatter_reduce(0, scene[free_to_go], here[free_to_go, 0], "amax")
            gain[mover & (here[:, 0] < top[scene])] = -1e9
            scene_np, here_np, src_np, legal_np, gain_np = scene.numpy(), here.numpy(), src.numpy(), legal.numpy(), gain.numpy()
            for j, i in enumerate(scatter):
                rows = np.nonzero(scene_np == j)[0]
                cands += _scatter_candidates(i, rows, here_np, src_np, legal_np, gain_np, rng, candidates)
        accepted = 0
        ok = torch.zeros(0, dtype=torch.bool)
        if cands:
            earlier, codes = _earlier(world, cands)
            eps = torch.tensor([c.episode for c in cands])
            later = world.index(eps)
            ok = verify(earlier, codes, later)
            gain = potential(earlier, fields[eps], guided[eps]) - potential(later, fields[eps], guided[eps])
        best = {}
        for k, c in enumerate(cands):
            if not bool(ok[k]):
                continue
            i = c.episode
            if phase[i] == "push":
                score = 1.0 if c.push is not None else 0.0  # a push when it verifies, else the step back
            else:
                score = float(gain[k])
                if score < -LEVEL_TOL or (score <= LEVEL_TOL and level[i] >= LEVEL_STEPS):
                    continue
                score += 1e-3 * rng.random()
            key = _key(earlier, k)
            if key in seen[i]:
                continue
            if i not in best or score > best[i][0]:
                best[i] = (score, k)
        for i in active:
            if done[i] or phase[i] in ("extract", "spread"):
                continue
            if i not in best:
                if phase[i] == "push":
                    phase[i] = "scatter"
                    floor[i] = len(codes_of[i])
                    _rescatter(goals[i], world, i, static, targets, bands, rng)
                    continue
                if len(codes_of[i]) > floor[i] and backtracks[i] < MAX_BACKTRACKS:  # a dead end: undo the last step (its state stays seen) and try another
                    backtracks[i] += 1
                    history[i].pop()
                    codes_of[i].pop()
                    kinds[i].pop()
                    ids[i].pop()
                    level[i] = levels[i].pop()
                    mass, pos, alive = history[i][-1]
                    world.mass[i], world.cheese_pos[i], world.cheese_alive[i] = mass.float(), pos, alive
                    continue
                done[i], end[i] = True, "stalled"
                continue
            score, k = best[i]
            accepted += 1
            c = cands[k]
            kind = "push"
            if phase[i] == "push":
                if c.push is not None:
                    pushes[i] += 1
                    if pushes[i] >= goals[i].push["length"]:
                        phase[i] = "scatter"
                        floor[i] = len(codes_of[i]) + 1
            else:
                kind = "level" if score <= LEVEL_TOL + 1e-3 else "progress"
                level[i] = level[i] + 1 if kind == "level" else 0
            commit(i, earlier.index(k), codes[k : k + 1], kind)
            if phase[i] == "scatter" and c.push is not None:
                _rescatter(goals[i], world, i, static, targets, bands, rng)
        stats["proposed"].append(len(cands))
        stats["verified"].append(int(ok.sum()))
        stats["accepted"].append(accepted)
        for i in active:
            if not done[i] and len(codes_of[i]) > 0 and finished(i):
                done[i], end[i] = True, "reached" if goals[i].start is not None else "natural"
    for i in range(b):
        if not done[i]:
            end[i] = "max_steps"
    episodes = []
    for i, g in enumerate(goals):
        frames_i = history[i][::-1]
        acts = codes_of[i][::-1]
        start = replace(world.index(i), mass=frames_i[0][0].unsqueeze(0).float(), cheese_pos=frames_i[0][1].unsqueeze(0), cheese_alive=frames_i[0][2].unsqueeze(0))
        occ0 = frames_i[0][0].bool()
        episodes.append(Episode(
            family=g.family,
            text=g.text,
            world=start,
            mass=torch.stack([f[0] for f in frames_i]),
            cheese_pos=torch.stack([f[1] for f in frames_i]),
            cheese_alive=torch.stack([f[2] for f in frames_i]),
            actions=torch.stack(acts) if acts else torch.zeros(0, *world.mass.shape[1:], dtype=torch.int8),
            natural=end[i] in ("natural", "reached", "hold"),
            end=end[i],
            goal=g,
            kinds=kinds[i][::-1],
            stats=start_stats(start, g.target.unsqueeze(0))[0].tolist(),
            distance=int((starts[i] & ~occ0).sum()) if g.start is not None else -1,
            touched=len(moved[i]),
        ))
    return episodes, stats


def _rescatter(goal, world, i, static, targets, bands, rng):
    """After the push phase: the walk moves robots away from the object where it now is."""
    cells = W.cheese_id_map(world.index(i))[0] == goal.push["slot"]
    static[i] = natural_field(cells)
    targets[i] = cells
    bands[i] = natural_bands(world.index(i), cells, rng)


# ---------------------------------------------------------------------------------------------
# Checks and layout


def replay(episodes):
    """Play episodes forward from their starts with world.step and the recorded codes (batched;
    shorter ones are padded with the goal and STAY, which a fixed point keeps). Returns {"passed"
    (E,) bool: every step legal (the rule refuses no recorded move) and matching the next frame,
    the last frame the goal, and the goal a fixed point; "first_bad" (E,) long, the first step that
    failed (-1 none)}."""
    e = len(episodes)
    steps = max(ep.steps for ep in episodes)
    world = W.concat([ep.world for ep in episodes])
    passed = torch.ones(e, dtype=torch.bool)
    first_bad = torch.full((e,), -1, dtype=torch.long)
    dims = world.mass.shape[1:]

    def frame(ep, t):
        t = min(t, ep.steps)
        return ep.mass[t], ep.cheese_pos[t], ep.cheese_alive[t]

    for t in range(steps):
        codes = torch.stack([ep.actions[t].long() if t < ep.steps else torch.full(dims, STAY, dtype=torch.long) for ep in episodes])
        legal = (W.refuse(world, codes, RULE) == codes).flatten(1).all(dim=1)
        world, _ = W.step(world, codes, rule=RULE)
        rows = [frame(ep, t + 1) for ep in episodes]
        want = replace(world, mass=torch.stack([r[0] for r in rows]).float(), cheese_pos=torch.stack([r[1] for r in rows]), cheese_alive=torch.stack([r[2] for r in rows]))
        good = legal & same_state(world, want)
        newly = passed & ~good
        first_bad[newly] = t
        passed &= good
    goal = W.concat([ep.goal.world for ep in episodes]) if all(ep.goal is not None for ep in episodes) else None
    if goal is not None:
        passed &= same_state(world, goal)
        passed &= fixed_point(world)
    return {"passed": passed, "first_bad": first_bad}


def to_store(episodes, steps=None):
    """The episodes as the dense fields of a demo store (lm.world_at reads them): every episode
    padded to `steps` (default the longest) with the goal and STAY, which the goal keeps. Returns
    (store dict, labels [{"e", "w0", "w1", "family", "text"}], w1 the episode's own steps)."""
    steps = max(ep.steps for ep in episodes) if steps is None else steps
    dims = tuple(episodes[0].mass.shape[1:])

    def padded(x, n, fill_last=True):
        if x.shape[0] >= n:
            return x[:n]
        extra = x[-1:].expand(n - x.shape[0], *x.shape[1:]) if fill_last else torch.full((n - x.shape[0], *x.shape[1:]), STAY, dtype=x.dtype)
        return torch.cat([x, extra])

    worlds = W.concat([ep.world for ep in episodes])
    store = {
        "mass": np.stack([padded(ep.mass, steps + 1).numpy() for ep in episodes]),
        "actions": np.stack([padded(ep.actions, steps, fill_last=False).numpy() if ep.steps else np.full((steps, *dims), STAY, dtype=np.int8) for ep in episodes]),
        "cheese_pos": torch.stack([padded(ep.cheese_pos, steps + 1) for ep in episodes]),
        "cheese_alive": torch.stack([padded(ep.cheese_alive, steps + 1) for ep in episodes]),
        "walls": worlds.walls,
        "cheese_color": worlds.cheese_color,
        "gravity": worlds.gravity,
    }
    for name in W.EXTRA_FIELDS:
        store[name] = getattr(worlds, name)
    labels = [{"e": i, "w0": 0, "w1": max(ep.steps, 1), "family": ep.family, "text": ep.text} for i, ep in enumerate(episodes)]
    return store, labels
