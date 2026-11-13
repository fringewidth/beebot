"""Demonstrations generated backwards from goal states (runs/handoff/reverse-demos.md).

A goal generator scripts an ordered end state (a free-standing tower, a bridge across a trench, a
letter on the floor, an object sitting on a floor patch) with an instruction in a stranger's
words. take_apart then walks backwards from it: from the current state S' it proposes earlier
states S together with the joint action codes a that would take S to S', and keeps a proposal only
if the real forward step does exactly that:

    world.step(S, a, rule=RULE) == S'      (every robot and object where S' has it)
    world.refuse(S, a, RULE) == a          (the rule lets every recorded move happen)
    world.fall(S) == S                     (S is at rest: nothing in it is falling)

so every recorded step is forward-causal under the world's own rules: pivots and their supports,
the overhang rule, pushes (proposed as the object stepping back with its pushers pressing on it,
pushers that do not move, as world.move_robots has it), gravity and eating. A step whose forward
version would make anything fall, eat anything or refuse a move never verifies, so it never
appears. The rule is deterministic here (all allowed moves reach; no outcome model, no jitter).

Which verified proposal is kept: the one that moves furthest away from the goal by a disorder
potential (potential_field: distance from the goal's cells, horizontal distance from their
centre, minus height above the terrain; minus a penalty per pair of robots touching face to
face), so the reversed episode makes progress at every step. Steps that keep the potential level
are allowed a few times when nothing better verifies; a step that lowers it never. Many robots
move in one step when their moves verify jointly. The walk stops when the scene looks like a
natural start (every robot resting on the floor or terrain, no two touching by a face, at most one
robot still in the goal's cells, the object pushed at least MIN_PUSHES cells away) or after
max_steps or when nothing verifies.

The push family is a phase machine on the same verifier: first PUSH phase steps alternate
"pushers step back one cell" and "the object steps back one cell while they press on it" until
the object is `length` cells from its patch, then the scatter walk above.

An episode (Episode) is the walk played forward: frames from the start to the goal, the action
codes recorded on each frame (per voxel, STAY where no robot is, as demos.py stores them), and
the instruction. replay() plays episodes forward with world.step and checks every step; to_store()
lays them out as the dense fields lm.world_at and the play store read.

    from beebot.vla.voxel import reverse
    rng = np.random.default_rng(0)
    goals = [reverse.make_goal("tower", rng) for _ in range(32)]
    episodes, stats = reverse.take_apart(goals, rng)
    report = reverse.replay(episodes)  # report["passed"]: (E,) bool
"""

from dataclasses import dataclass, field, replace

import numpy as np
import torch

from beebot.vla.voxel import world as W
from beebot.vla.voxel.world import CUBE_MOVES, DIR_TABLE, DIRS, FRICTION, MAX_OBJECTS, SIZE, STAY, VoxelWorld

RULE = "switch_cube"  # the demo's rule (demos.PRESETS["demo"]); gravity is on in every scene
FAMILIES = ("tower", "bridge", "letter", "push")
SLOTS = MAX_OBJECTS
COLOURS = ("red", "green", "blue", "yellow", "orange", "purple")  # palette indices 0-5 (world.COLOUR_NAMES)
HORIZONTAL = ((0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0))
DIST_CAP = 6.0  # distance from the goal's cells counts up to this
CENTRE_WEIGHT = 0.3  # horizontal distance from the goal's centre, up to CENTRE_CAP
CENTRE_CAP = 8.0
HEIGHT_WEIGHT = 1.0  # per voxel above the terrain
TOUCH_WEIGHT = 0.5  # per pair of robots face to face
MIN_PUSHES = 2
NEAR_DIST = 2.0  # natural start: all robots but one at least this far from the goal's cells
MEAN_DIST = 4.0  # and this far on average
TOUCH_SHARE = 4  # and at most one pair of robots face to face per this many robots
CANDIDATES = 6  # joint proposals per episode and step in the scatter walk
TEMPERATURE = 0.5
LEVEL_STEPS = 4  # level steps (potential unchanged) allowed in a row before the walk gives up
MAX_BACKTRACKS = 30  # dead ends an episode may back out of (depth first, never revisiting a state)
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


def ground(walls):
    """(..., H, W) long: the terrain height of each column (walls are a heightmap)."""
    return walls.sum(dim=-3)


def _dilate(grid, r=1):
    """(D, H, W) bool grown by r voxels in every direction (26-neighbourhood)."""
    out = torch.nn.functional.max_pool3d(grid.float()[None, None], 2 * r + 1, stride=1, padding=r)
    return out[0, 0] > 0


def scatter_robots(rng, world, n, avoid):
    """Put n robots on the ground (floor or terrain top) at random, none touching another robot,
    an object or the cells `avoid` (D, H, W) by a face, edge or corner. Returns how many it placed."""
    d, h, w = world.mass.shape[1:]
    height = ground(world.walls[0])
    placed = 0
    for _ in range(60 * max(n, 1)):
        if placed == n:
            break
        y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
        z = int(height[y, x])
        if z >= d:
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


MAKERS = {"tower": tower_goal, "bridge": bridge_goal, "letter": letter_goal, "push": push_goal}


def fixed_point(world):
    """(B,) bool: every robot staying changes nothing (the state is at rest under the rule)."""
    nxt, _ = W.step(world, torch.full(world.mass.shape, STAY, dtype=torch.long), rule=RULE)
    return same_state(nxt, world)


def make_goal(family, rng, tries=20):
    """A goal of `family` that is a fixed point of the forward step."""
    for _ in range(tries):
        goal = MAKERS[family](rng)
        if bool(fixed_point(goal.world)[0]):
            return goal
    raise RuntimeError(f"no stable {family} goal in {tries} tries")


# ---------------------------------------------------------------------------------------------
# The walk backwards


def same_state(a, b):
    """(B,) bool: the same robots, the same live objects at the same places."""
    mass = ((a.mass > 0.5) == (b.mass > 0.5)).flatten(1).all(dim=1)
    alive = (a.cheese_alive == b.cheese_alive).all(dim=1)
    pos = ((a.cheese_pos == b.cheese_pos).all(dim=2) | ~a.cheese_alive).all(dim=1)
    return mass & alive & pos


def potential_field(target, walls):
    """(D, H, W) float: what one robot in each voxel adds to the disorder potential."""
    d, h, w = target.shape
    grid = torch.stack(torch.meshgrid(torch.arange(d), torch.arange(h), torch.arange(w), indexing="ij"), dim=-1).float()
    cells = target.nonzero().float()
    dist = torch.cdist(grid.view(-1, 3), cells).min(dim=1).values.view(d, h, w).clamp(max=DIST_CAP)
    centre = cells[:, 1:].mean(dim=0)
    horizontal = (grid[..., 1:] - centre).norm(dim=-1).clamp(max=CENTRE_CAP)
    near = torch.nn.functional.max_pool2d(ground(walls).float()[None, None], 3, stride=1, padding=1)[0, 0]  # a robot beside a terrain wall climbs it for free
    above = (grid[..., 0] - near.unsqueeze(0)).clamp(min=0)
    return dist + CENTRE_WEIGHT * horizontal - HEIGHT_WEIGHT * above


def touching_pairs(occ):
    """(B,) long: pairs of robots face to face."""
    return (occ[:, 1:] & occ[:, :-1]).flatten(1).sum(1) + (occ[:, :, 1:] & occ[:, :, :-1]).flatten(1).sum(1) + (occ[:, :, :, 1:] & occ[:, :, :, :-1]).flatten(1).sum(1)


def potential(world, fields):
    occ = world.mass > 0.5
    return (fields * occ).flatten(1).sum(dim=1) - TOUCH_WEIGHT * touching_pairs(occ)


def distance_field(target):
    """(D, H, W) float: distance from the nearest of the goal's cells."""
    d, h, w = target.shape
    grid = torch.stack(torch.meshgrid(torch.arange(d), torch.arange(h), torch.arange(w), indexing="ij"), dim=-1).float()
    return torch.cdist(grid.view(-1, 3), target.nonzero().float()).min(dim=1).values.view(d, h, w)


def natural(world, dists):
    """(B,) bool: the scene looks like a natural start: every robot resting on the floor or
    terrain, at most one pair face to face per TOUCH_SHARE robots, all robots but one at least NEAR_DIST from the goal's cells
    (dists (B, D, H, W), distance_field) and their mean distance (capped at DIST_CAP) at least
    MEAN_DIST."""
    occ = world.mass > 0.5
    under = torch.ones_like(occ)
    under[:, 1:] = world.walls[:, :-1]
    grounded = (~occ | under).flatten(1).all(dim=1)
    near = (occ & (dists < NEAR_DIST)).flatten(1).sum(dim=1) <= 1
    mean = (dists.clamp(max=DIST_CAP) * occ).flatten(1).sum(dim=1) / occ.flatten(1).sum(dim=1).clamp(min=1)
    robots = occ.flatten(1).sum(dim=1)
    return grounded & (touching_pairs(occ) * TOUCH_SHARE <= robots) & near & (mean >= MEAN_DIST)


def single_moves(world, chunk=256):
    """Every move one robot could have made into its cell: for each robot (scene, here) and each of
    the 14 cube moves c, whether a robot at here - c (the source) could pivot into here under the
    rule with the rest of the scene as it is (pivot_moves on the scene without this robot; the
    joint step is verified later). Returns scene (N,), here (N, 3), src (N, 14, 3), legal (N, 14)."""
    occ = world.mass > 0.5
    b = occ.shape[0]
    dims = occ.shape[1:]
    cheese = W.cheese_id_map(world) >= 0
    solid = occ | world.walls | cheese
    scene, z, y, x = occ.nonzero(as_tuple=True)
    here = torch.stack([z, y, x], dim=1)
    n = len(scene)
    src = here.unsqueeze(1) - DIR_TABLE[_CUBE_CODES].unsqueeze(0)
    flat, inside = W.flat_index(src, dims)
    free = inside & ~solid.reshape(b, -1)[scene.unsqueeze(1), flat]
    legal = torch.zeros(n, len(_CUBE_CODES), dtype=torch.bool)
    for i in range(0, n, chunk):
        rows = torch.arange(i, min(i + chunk, n))
        cubes = occ[scene[rows]].clone()
        cubes[torch.arange(len(rows)), z[rows], y[rows], x[rows]] = False
        table = W.pivot_moves(cubes, world.walls[scene[rows]], cheese[scene[rows]], release=RULE == "switch_cube")
        table = table[:, _CUBE_CODES].flatten(2)
        legal[rows] = table.gather(2, flat[rows].unsqueeze(2)).squeeze(2)
    return scene, here, src, legal & free


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
        out.append(_Candidate(b, [(r, r - e, code_of(e)) for r in behind]))
    return out


def _scatter_candidates(b, rows, here, src, legal, delta, rng, m=CANDIDATES):
    """Up to m joint proposals for episode b: each robot with a move that does not lower the
    potential picks one (softmax on its gain) and joins with a probability that differs per
    proposal; plus the single best move and a random single move."""
    options, rising = [], []
    for r in rows:
        ok = legal[r] & (delta[r] >= -1e-6)
        idx = ok.nonzero().squeeze(1)
        if len(idx):
            options.append((r, idx, delta[r, idx]))
            up = idx[delta[r, idx] > 1e-6]
            if len(up):
                rising.append((r, up, delta[r, up]))
    if not options:
        return []
    out = []
    shares = np.linspace(1.0, 0.3, max(m - 2, 1))
    for share in shares if rising else ():
        moves, used = [], set()
        for r, idx, gain in rising:
            if rng.random() > share:
                continue
            weights = torch.exp(gain / TEMPERATURE).numpy()
            j = int(idx[int(rng.choice(len(idx), p=weights / weights.sum()))])
            s = tuple(src[r, j].tolist())
            if s in used:
                continue
            used.add(s)
            moves.append((here[r], src[r, j], int(_CUBE_CODES[j])))
        if moves:
            out.append(_Candidate(b, moves))
    best = max(((float(gain.max()), r, int(idx[int(gain.argmax())])) for r, idx, gain in options), key=lambda t: t[0])
    out.append(_Candidate(b, [(here[best[1]], src[best[1], best[2]], int(_CUBE_CODES[best[2]]))]))
    r, idx, _ = options[int(rng.integers(len(options)))]
    j = int(idx[int(rng.integers(len(idx)))])
    out.append(_Candidate(b, [(here[r], src[r, j], int(_CUBE_CODES[j]))]))
    return out


def _build(world, cands):
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


@dataclass
class Episode:
    family: str
    text: str
    world: VoxelWorld  # the start (one scene): static fields as recorded
    mass: torch.Tensor  # (T + 1, D, H, W) uint8, frame 0 the start, frame T the goal
    cheese_pos: torch.Tensor  # (T + 1, K, 3) long
    cheese_alive: torch.Tensor  # (T + 1, K) bool
    actions: torch.Tensor  # (T, D, H, W) int8: the codes recorded on frame t
    natural: bool  # the start looks like a natural scene (natural())
    end: str  # why the walk stopped: natural, max_steps, stalled
    goal: Goal = None

    @property
    def steps(self):
        return self.actions.shape[0]


def take_apart(goals, rng, max_steps=96, candidates=CANDIDATES):
    """Walk every goal backwards (module docstring) in one batch. Returns (episodes, stats): stats
    "proposed", "verified" and "accepted" are per-iteration counts of proposals made, passing the
    verifier and kept (lists); "backtracks" per episode."""
    world = W.concat([g.world for g in goals])
    b = len(goals)
    targets = torch.stack([distance_field(g.target) for g in goals])
    fields = torch.stack([potential_field(g.target, g.world.walls[0]) for g in goals])
    phase = ["push" if g.push is not None else "scatter" for g in goals]
    pushes = [0] * b
    level = [0] * b
    levels = [[] for _ in range(b)]  # level before each recorded step, for backtracking
    floor = [0] * b  # recorded steps a backtrack may not undo (the push phase's)
    backtracks = [0] * b
    done = [False] * b
    end = [""] * b
    seen = [{_key(world, i)} for i in range(b)]
    frames = [[(world.mass[i] > 0.5).to(torch.uint8), world.cheese_pos[i].clone(), world.cheese_alive[i].clone()] for i in range(b)]
    history = [[f] for f in frames]
    codes_of = [[] for _ in range(b)]
    stats = {"proposed": [], "verified": [], "accepted": [], "backtracks": backtracks}
    for t in range(max_steps + 2 * MAX_BACKTRACKS):
        for i in range(b):
            if not done[i] and len(codes_of[i]) >= max_steps:
                done[i], end[i] = True, "max_steps"
        active = [i for i in range(b) if not done[i]]
        if not active:
            break
        cands = []
        scatter = [i for i in active if phase[i] == "scatter"]
        for i in active:
            if phase[i] == "push":
                cands += _push_candidates(world, i, goals[i])
        if scatter:
            sub_idx = torch.tensor(scatter)
            sub = world.index(sub_idx)
            scene, here, src, legal = single_moves(sub)
            f = fields[sub_idx]
            gain = f[scene.unsqueeze(1), src[..., 0].clamp(0, SIZE - 1), src[..., 1].clamp(0, SIZE - 1), src[..., 2].clamp(0, SIZE - 1)] - f[scene, here[:, 0], here[:, 1], here[:, 2]].unsqueeze(1)
            for j, i in enumerate(scatter):
                rows = (scene == j).nonzero().squeeze(1).tolist()
                cands += _scatter_candidates(i, rows, here, src, legal, gain, rng, candidates)
        accepted = 0
        ok = torch.zeros(0, dtype=torch.bool)
        if cands:
            earlier, codes = _build(world, cands)
            later = world.index(torch.tensor([c.episode for c in cands]))
            ok = verify(earlier, codes, later)
            gain = potential(earlier, fields[[c.episode for c in cands]]) - potential(later, fields[[c.episode for c in cands]])
        best = {}
        for k, c in enumerate(cands):
            if not bool(ok[k]):
                continue
            i = c.episode
            if phase[i] == "push":
                score = 1.0 if c.push is not None else 0.0  # a push when it verifies, else the step back
            else:
                score = float(gain[k])
                if score < -1e-6 or (score <= 1e-6 and level[i] >= LEVEL_STEPS):
                    continue
                score += 1e-3 * rng.random()
            key = _key(earlier, k)
            if key in seen[i]:
                continue
            if i not in best or score > best[i][0]:
                best[i] = (score, k, key)
        for i in active:
            if i not in best:
                if phase[i] == "push":
                    phase[i] = "scatter"
                    floor[i] = len(codes_of[i])
                    _rescatter(goals[i], world, i, fields, targets)
                    continue
                if len(codes_of[i]) > floor[i] and backtracks[i] < MAX_BACKTRACKS:  # a dead end: undo the last step (its state stays seen) and try another
                    backtracks[i] += 1
                    history[i].pop()
                    codes_of[i].pop()
                    level[i] = levels[i].pop()
                    mass, pos, alive = history[i][-1]
                    world.mass[i], world.cheese_pos[i], world.cheese_alive[i] = mass.float(), pos, alive
                    continue
                done[i], end[i] = True, "stalled"
                continue
            score, k, key = best[i]
            accepted += 1
            c = cands[k]
            levels[i].append(level[i])
            if phase[i] == "push":
                if c.push is not None:
                    pushes[i] += 1
                    if pushes[i] >= goals[i].push["length"]:
                        phase[i] = "scatter"
                        floor[i] = len(codes_of[i]) + 1
            else:
                level[i] = level[i] + 1 if score <= 1e-6 + 1e-3 else 0
            for name in ("mass", "cheese_pos", "cheese_alive"):
                getattr(world, name)[i] = getattr(earlier, name)[k]
            if phase[i] == "scatter" and c.push is not None:
                _rescatter(goals[i], world, i, fields, targets)
            seen[i].add(key)
            history[i].append([(earlier.mass[k] > 0.5).to(torch.uint8), earlier.cheese_pos[k].clone(), earlier.cheese_alive[k].clone()])
            codes_of[i].append(codes[k].to(torch.int8))
        stats["proposed"].append(len(cands))
        stats["verified"].append(int(ok.sum()))
        stats["accepted"].append(accepted)
        live = [i for i in active if not done[i] and phase[i] == "scatter" and (goals[i].push is None or pushes[i] >= MIN_PUSHES)]
        if live:
            nat = natural(world.index(torch.tensor(live)), targets[live])
            for j, i in enumerate(live):
                if bool(nat[j]) and len(codes_of[i]) > 0:
                    done[i], end[i] = True, "natural"
    for i in range(b):
        if not done[i]:
            end[i] = "max_steps"
    episodes = []
    for i, g in enumerate(goals):
        frames_i = history[i][::-1]
        acts = codes_of[i][::-1]
        start = replace(world.index(i), mass=frames_i[0][0].unsqueeze(0).float(), cheese_pos=frames_i[0][1].unsqueeze(0), cheese_alive=frames_i[0][2].unsqueeze(0))
        episodes.append(Episode(
            family=g.family,
            text=g.text,
            world=start,
            mass=torch.stack([f[0] for f in frames_i]),
            cheese_pos=torch.stack([f[1] for f in frames_i]),
            cheese_alive=torch.stack([f[2] for f in frames_i]),
            actions=torch.stack(acts) if acts else torch.zeros(0, *world.mass.shape[1:], dtype=torch.int8),
            natural=end[i] == "natural",
            end=end[i],
            goal=g,
        ))
    return episodes, stats


def _rescatter(goal, world, i, fields, targets):
    """After the push phase: the scatter walk moves robots away from the object where it now is."""
    cells = W.cheese_id_map(world.index(i))[0] == goal.push["slot"]
    fields[i] = potential_field(cells, world.walls[i])
    targets[i] = distance_field(cells)


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
