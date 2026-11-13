"""Demonstrations generated backwards from goal states (runs/handoff/reverse-demos.md, reverse-v2.md).

A goal generator scripts an ordered end state (a tower, a crossing, a letter, an object on a patch,
a span over open air, an arch, a table, a chair, an L, big shapes of 60-115 robots) with an
instruction in a stranger's words. take_apart then walks backwards from it: from the current
state S' it proposes earlier states S with the joint action codes a that would take S to S', and
keeps a proposal only if the real forward step does exactly that:

    world.step(S, a, rule=RULE) == S'      (every robot and object where S' has it)
    world.refuse(S, a, RULE) == a          (the rule lets every recorded move happen)
    world.fall(S) == S                     (S is at rest: nothing in it is falling)

so every recorded step is forward-causal under the world's own rules (pivots and their supports,
the overhang rule, pushes, gravity, eating). A step whose forward version would make anything
fall, eat anything or refuse a move never verifies. The rule is deterministic here.

Two modes, chosen per goal:

1. natural (Goal.start is None): walk until the start is statistically a natural scene. The walk
   climbs a potential: per robot, a bonus for being off the goal's cells, minus its geodesic
   distance (cube moves over cells a cube can hold on to) to the floor or terrain, plus distance
   from the goal; minus penalties per pair of robots face to face and per pair within two columns.
   It stops when every statistic of start_stats lies inside the band that sample_scenes-style
   scenes with the same terrain, robot count and goal give (natural_bands: 2.5-97.5 % quantiles
   of 32 drawn scenes).
1. guided (Goal.start = A, a (D, H, W) bool occupancy at rest with as many robots): walk toward A.
   The potential rewards robots on A's cells and charges the others their geodesic distance to
   A's empty cells; robots already on A's cells are never proposed to move off them. It stops
   when the state is exactly A ("reached").

Moves are proposed from a 7^3 crop round each robot (sources: which cells could have pivoted a
cube into it), joint proposals combine robots three or more cells apart, and the best verified
proposal by potential is kept. A step that lowers the potential is never taken, a few level
steps are allowed, dead ends are backtracked depth first.

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
MIN_PUSHES = 2
CANDIDATES = 6  # joint proposals per episode and step
TEMPERATURE = 0.5
LEVEL_STEPS = 4  # level steps (potential unchanged within LEVEL_TOL) allowed in a row
LEVEL_TOL = 0.05
MAX_BACKTRACKS = 30  # dead ends an episode may back out of (depth first, never revisiting a state)
SPACING = 3  # robots in one joint proposal: their cells at least this far apart (Chebyshev)
STEPS_BASE, STEPS_PER_ROBOT, MAX_STEPS_CAP = 64, 16, 2000  # step limit of an episode: base + per robot, capped
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
MAX_SCAFFOLDS = 8  # scaffold attempts per episode
SCAFFOLD_ROBOTS = 8  # stuck robots tried per attempt
SCAFFOLD_PLANS = 6  # helper layouts tried per attempt
HELPER_TRIES = 4  # robots tried per helper cell
PATH_DEPTH = 40
EXTRACT_TRIES = 6  # robots tried per extraction
PIPELINE = 4  # walks planned in a row and run overlapped
DROP_DIST = 3.0  # an extracted robot stops at least this far from the goal's cells
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
        shape = {"letter": base.info.get("shape", "shape"), "table": "table", "cantilever": "post", "chair": "chair"}[base.family]
        text = str(rng.choice([
            f"now make a tower of {word} out of it",
            f"take {word} of you off the {shape} and stack them into a tower beside it",
            f"use {word} robots from the {shape} to build a {word}-high tower next to it",
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
    the scene as it is: pivot_moves on the 7^3 crop round `here` (the robot there removed when
    remove_self), src on the grid and free. Returns (src (N, 14, 3), legal (N, 14) bool)."""
    cubes, walls, cheese = padded
    if len(here) == 0:
        return torch.zeros(0, len(_CUBE_CODES), 3, dtype=torch.long), torch.zeros(0, len(_CUBE_CODES), dtype=torch.bool)
    cc = _crop(cubes, scene, here)
    if remove_self:
        cc[:, _R, _R, _R] = False
    table = W.pivot_moves(cc, _crop(walls, scene, here), _crop(cheese, scene, here), release=RULE == "switch_cube")
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


def natural_scenes(walls, n, rng, samples=REF_SAMPLES, avoid=None):
    """(samples, D, H, W) bool: n robots dropped on the terrain the way sample_scenes and
    sample_object_scenes place them (random columns, stacked when a column is drawn again, at
    most three above the terrain), off the cells `avoid` (objects)."""
    d, h, w = walls.shape
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
            if z0 >= min(d, int(height[y, x]) + 3):
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


def verify(earlier, codes, later):
    """(C,) bool: earlier is at rest, the rule lets every move in codes happen, and the forward
    step takes earlier exactly to later."""
    nxt, _ = W.step(earlier, codes, rule=RULE)
    ok = same_state(nxt, later) & same_state(W.fall(earlier), earlier)
    if ok.any():
        idx = ok.nonzero().squeeze(1)
        sub = earlier.index(idx)
        kept = (W.refuse(sub, codes[idx], RULE) == codes[idx]).flatten(1).all(dim=1)
        ok[idx] = kept
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


def _reverse_path(one, p, q, depth=PATH_DEPTH):
    """Cells [p, ..., q]: a walk in reverse time of the robot at p to q (a cell, or a (D, H, W) bool
    mask of cells: the first reached), one cube move per step, the rest of the scene still
    (breadth first over sources(); each cell it stops in face to face with something that holds
    it). None when q is not reached."""
    goal = q if isinstance(q, torch.Tensor) else None
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
    for _ in range(depth):
        here = torch.tensor(frontier)
        src, legal = sources(padded, torch.zeros(len(frontier), dtype=torch.long), here, remove_self=False)
        legal &= anchor[src[..., 0].clamp(0, SIZE - 1), src[..., 1].clamp(0, SIZE - 1), src[..., 2].clamp(0, SIZE - 1)]
        nxt = []
        for n, k in legal.nonzero().tolist():
            y = tuple(src[n, k].tolist())
            if y in parent:
                continue
            parent[y] = frontier[n]
            if (bool(goal[y]) if goal is not None else y == q):
                path = [y]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return path[::-1]
            nxt.append(y)
        if not nxt:
            return None
        frontier = nxt
    return None


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


def _extract(one, movers, dest, rng, tries=EXTRACT_TRIES):
    """One robot that must move (movers (D, H, W) bool) walks out in reverse time to the nearest
    cell of dest (D, H, W) bool: the highest such robot that has a move at all and a walk whose
    every step verifies (forward: a structure is built bottom up, each robot climbing in to its
    place). Returns the walk's cells [p, ..., q] or None (none of `tries` robots found a way)."""
    occ = one.mass[0] > 0.5
    cells = (occ & movers).nonzero()
    if len(cells) == 0:
        return None
    _, legal = sources(padded_grids(one), torch.zeros(len(cells), dtype=torch.long), cells)
    has = legal.any(dim=1)
    if not has.any():
        return None
    order = sorted(range(len(cells)), key=lambda k: (not bool(has[k]), -int(cells[k, 0]), rng.random()))
    for k in order[:tries]:
        if not bool(has[k]):
            break
        path = _reverse_path(one, tuple(cells[k].tolist()), dest & ~occ)
        if path is not None and _path_steps(one, path) is not None:
            return path
    return None


def _moved(one, path):
    mass = one.mass.clone()
    mass[(0, *path[0])] = 0.0
    mass[(0, *path[-1])] = 1.0
    return replace(one, mass=mass)


def _schedule_states(one, paths, starts, n):
    """States 0..n and codes 0..n-1 of robots walking their paths from their start times (a robot
    stands at the head of its path before its start and at its end after it)."""
    occ = one.mass[0] > 0.5
    base = occ.clone()
    for path in paths:
        base[path[0]] = False
    states = base.unsqueeze(0).repeat(n + 1, 1, 1, 1)
    codes = torch.full((n, *occ.shape), STAY, dtype=torch.long)
    for path, s in zip(paths, starts):
        last = len(path) - 1
        for t in range(n + 1):
            states[(t, *path[min(max(t - s, 0), last)])] = True
        for t in range(s, min(s + last, n)):
            codes[(t, *path[t - s + 1])] = code_of(np.subtract(path[t - s], path[t - s + 1]))
    return states, codes


def _pipeline(one, paths):
    """Walks planned one after another (each in the state the previous ones leave) run overlapped:
    each starts at the earliest of a few offsets after the previous one's start at which every
    affected step verifies, else when all before it are done (the plan as found, valid as it
    stands). Returns the reverse steps [(earlier mass, codes)]."""
    starts = [0]
    for j in range(1, len(paths)):
        finish = max(s + len(p) - 1 for s, p in zip(starts, paths[:j]))
        prev = starts[-1]
        chosen = finish
        for d in sorted({prev + 1, prev + max(1, (len(paths[j - 1]) - 1) // 2)}):
            if d >= finish:
                break
            n = max(finish, d + len(paths[j]) - 1)
            states, codes = _schedule_states(one, paths[: j + 1], starts + [d], n)
            later = replace(one.index([0] * (n - d)), mass=states[d:n].float())
            earlier = replace(later, mass=states[d + 1 : n + 1].float())
            if bool(verify(earlier, codes[d:n], later).all()):
                chosen = d
                break
        starts.append(chosen)
    n = max(s + len(p) - 1 for s, p in zip(starts, paths))
    states, codes = _schedule_states(one, paths, starts, n)
    return [(states[t + 1 : t + 2].float(), codes[t : t + 1]) for t in range(n)]


def _drop_cells(one, target, dist=None, alone=True):
    """(D, H, W) bool: where a robot taken out of a structure may stop: on the floor or terrain, at
    least `dist` from the goal's cells, touching no robot by a face, edge or corner when alone."""
    occ = one.mass[0] > 0.5
    dist = distance_field(target) if dist is None else dist
    out = on_ground(one)[0] & ~(W.cheese_id_map(one)[0] >= 0) & ~occ
    if alone:
        return out & (dist >= DROP_DIST) & ~_dilate(occ)
    return out & (dist >= 1.5)


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
    end: str  # why the walk stopped: natural, reached, max_steps, stalled
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


def take_apart(goals, rng, max_steps=None, candidates=CANDIDATES, scaffold=True):
    """Walk every goal backwards (module docstring) in one batch. Returns (episodes, stats): stats
    "proposed", "verified" and "accepted" are per-iteration counts of the joint greedy walk (lists);
    "backtracks" and "scaffolds" (attempts) per episode; "extract_calls" and "extract_ok" the
    extractions tried and found."""
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
    stats = {"proposed": [], "verified": [], "accepted": [], "backtracks": backtracks, "scaffolds": scaffolds, "extract_calls": 0, "extract_ok": 0}

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

    def finished(i):
        if goals[i].start is not None:
            return bool(((world.mass[i] > 0.5) == starts[i]).all())
        if phase[i] != "scatter" or (goals[i].push is not None and pushes[i] < MIN_PUSHES) or bands[i] is None:
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
            movers = ~starts[i] if goals[i].start is not None else targets[i]
            if not (occ_i & movers).any():
                phase[i] = "scatter"
                floor[i] = len(codes_of[i])
                continue
            paths, state = [], one
            for _ in range(PIPELINE):  # plan a few walks one after another, then run them overlapped
                occ_s = state.mass[0] > 0.5
                if not (occ_s & movers).any():
                    break
                if goals[i].start is not None:
                    path = _extract(state, movers, starts[i] & ~occ_s, rng)
                else:  # far and alone first, then anywhere on the ground off the goal
                    path = _extract(state, movers, _drop_cells(state, targets[i], dists[i]), rng)
                    if path is None:
                        path = _extract(state, movers, _drop_cells(state, targets[i], dists[i], alone=False), rng)
                if path is None:
                    break
                paths.append(path)
                movers = movers.clone()
                movers[path[-1]] = False  # where it stops is the drop (or A's cell), not the structure
                state = _moved(state, path)
            steps = _pipeline(one, paths) if paths else None
            kind = "extract"
            stats["extract_calls"] += 1
            stats["extract_ok"] += steps is not None
            if steps is None and scaffold and scaffolds[i] < MAX_SCAFFOLDS:
                scaffolds[i] += 1
                steps, kind = _scaffold(one, movers, rng), "scaffold"
            if steps is None:
                if goals[i].start is not None:
                    done[i], end[i] = True, "stalled"
                else:
                    phase[i] = "scatter"
                    floor[i] = len(codes_of[i])
                continue
            for mass, codes_k in steps:
                commit(i, replace(world.index(i), mass=mass), codes_k, kind)
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
            if done[i] or phase[i] == "extract":
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
            natural=end[i] in ("natural", "reached"),
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
