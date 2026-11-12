"""The voxel world: the 2D robot world (vla/world.py) in three dimensions, batched in torch.

Axes are (z, y, x): z is height, z = 0 the voxel layer resting on the floor. One robot per
voxel. Every step each robot picks one of 27 actions (DIRS: the 26 neighbours, then STAY).
Moves are resolved in rounds with random priority, as in 2D: a robot moves into a free voxel
(no robot, wall or cheese), the winner among claimants is random, a vacated voxel is free for
the next round, so lines advance together and swaps stay put.

Cheeses are 2 x 2 x 2 solid blocks. Robots trying to move into a cheese press on it; when the
press along its strongest axis reaches FRICTION the cheese slides one voxel that way, provided
its leading face is clear of walls, other cheeses and the grid edge; robots in the leading face
are shoved one voxel further if that voxel is free (else nothing moves). A cheese is eaten when
robots fill enough of its shell (the voxels touching it, faces, edges or corners, that are not
walls and are inside the grid): at least EAT_SHARE of the open shell and EAT_MIN_SHARE of the
full shell.

Gravity is a per-scene switch. With it on, a robot must be supported: connected through
face-adjacent robots to a robot on the floor (z = 0) or face-adjacent to a wall or a cheese.
Unsupported robots fall one voxel per step (whole columns fall together); an unsupported cheese
(nothing under its 2 x 2 footprint) falls one voxel per step too. With gravity off robots fly.

Objects (README "Objects"). A world can give its cheese slots properties (VoxelWorld.form and the
fields after it; None everywhere is the world above, unchanged): a shape of 1 to 4 voxels per side
inside a BOX^3 box (cube, slab, rod, L), a two-colour pattern (striped, two-tone), a friction
(light FRICTION, heavy HEAVY_FRICTION) and whether it is food. Objects that are not food are
passive blocks: pushed like cheeses, never eaten. Floor zones are coloured floor cells. Every rule
above holds for any shape: an object slides when the press reaches its friction and the voxels it
moves into are clear, it is eaten (food only) when robots fill EAT_SHARE of its open shell and
EAT_MIN_SHARE of its full shell (so the robots it takes grow with its size), and it falls when
nothing is under any of its lowest voxels. A heavy object also counts the press of robots pushing
in line behind a pressing robot, up to CHAIN_DEPTH deep (chain_press), so about six robots move
it whatever its face; light objects count only the robots moving into them, as before.
"""

from dataclasses import dataclass, fields, replace

import torch

from beebot.vla.world import COLOUR_NAMES, COLOUR_RGB, PALETTE_NAMES, PALETTE_RGB  # noqa: F401  (2D colours: the four play colours first, then the extended ones)

SIZE = 16
CHEESE_SIDE = 2
MAX_CHEESES = 3
DIRS = tuple((dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dz, dy, dx) != (0, 0, 0))
STAY = len(DIRS)  # 26
N_ACTIONS = STAY + 1  # 27
FRICTION = 2.0  # robots' worth of press on a cheese's face (its face is 2 x 2 voxels)
EAT_SHARE = 0.75
EAT_MIN_SHARE = 0.4  # of the full shell (56 voxels for a 2 x 2 x 2 cheese)
OLD_CHANNELS = 6  # robot, wall, cheese mask, cheese RGB: the observation before objects (its first six channels are unchanged)
CHANNELS = ("robot", "wall", "object", "object red", "object green", "object blue", "food", "zone red", "zone green", "zone blue")
N_CHANNELS = len(CHANNELS)  # 10
MAX_OBJECTS = 6  # object slots of a world drawn by sample_object_scenes
BOX = 4  # an object's voxels lie in a BOX^3 box from its lowest corner (cheese_pos)
OBJECT_SHAPES = ("cube", "slab", "rod", "L")
PATTERNS = ("plain", "striped", "two-tone")
HEAVY_FRICTION = 6.0
CHAIN_DEPTH = 3  # robots in line behind a robot pressing a heavy object that add their press


@dataclass
class VoxelWorld:
    mass: torch.Tensor  # (B, D, H, W) float, 1 where a robot is
    walls: torch.Tensor  # (B, D, H, W) bool
    cheese_pos: torch.Tensor  # (B, K, 3) long, lowest corner (z, y, x) of each cheese
    cheese_alive: torch.Tensor  # (B, K) bool
    cheese_color: torch.Tensor  # (B, K) long palette index, -1 for an empty slot
    gravity: torch.Tensor  # (B,) bool
    form: torch.Tensor = None  # (B, K, BOX, BOX, BOX) bool: the voxels of each object from cheese_pos; None: every object a 2 x 2 x 2 cube with none of the fields below
    colour2: torch.Tensor = None  # (B, K) long: the second colour of a pattern, -1 for none
    pattern: torch.Tensor = None  # (B, K) long index into PATTERNS
    friction: torch.Tensor = None  # (B, K) float: the press that slides the object (FRICTION light, HEAVY_FRICTION heavy)
    food: torch.Tensor = None  # (B, K) bool: can be eaten; False for a passive block
    zones: torch.Tensor = None  # (B, H, W) long: the colour painted on each floor cell (on top of the terrain), -1 for none
    stuck: torch.Tensor = None  # (B, D, H, W) bool: robots a move left off the lattice (OutcomeModel); they never act again. None: none

    def index(self, i):
        """A sub-batch: an int, a slice or an index tensor."""
        if isinstance(i, int):
            i = slice(i, i + 1)
        return VoxelWorld(*(None if getattr(self, f.name) is None else getattr(self, f.name)[i] for f in fields(self)))


EXTRA_FIELDS = ("form", "colour2", "pattern", "friction", "food", "zones")


def default_form(b, k, device=None):
    """(b, k, BOX, BOX, BOX) bool: 2 x 2 x 2 cubes in the corner of the box."""
    form = torch.zeros(b, k, BOX, BOX, BOX, dtype=torch.bool, device=device)
    form[:, :, :CHEESE_SIDE, :CHEESE_SIDE, :CHEESE_SIDE] = True
    return form


def with_objects(world, k=None):
    """The world with every object field filled in (defaults: 2 x 2 x 2 light plain food, no
    zones) and k object slots (empty ones appended); unchanged if that is what it has."""
    b, have = world.cheese_alive.shape
    k = have if k is None else k
    if k < have:
        raise ValueError(f"cannot shrink {have} slots to {k}")
    device = world.mass.device
    pad = k - have

    def grow(value, fill):
        if pad == 0:
            return value
        extra = torch.full((b, pad, *value.shape[2:]), fill, dtype=value.dtype, device=device)
        return torch.cat([value, extra], dim=1)

    form = default_form(b, have, device) if world.form is None else world.form
    return replace(
        world,
        cheese_pos=grow(world.cheese_pos, 0),
        cheese_alive=grow(world.cheese_alive, False),
        cheese_color=grow(world.cheese_color, -1),
        form=torch.cat([form, default_form(b, pad, device)], dim=1) if pad else form,
        colour2=grow(torch.full((b, have), -1, dtype=torch.long, device=device) if world.colour2 is None else world.colour2, -1),
        pattern=grow(torch.zeros(b, have, dtype=torch.long, device=device) if world.pattern is None else world.pattern, 0),
        friction=grow(torch.full((b, have), FRICTION, device=device) if world.friction is None else world.friction, FRICTION),
        food=grow(torch.ones(b, have, dtype=torch.bool, device=device) if world.food is None else world.food, True),
        zones=torch.full(world.mass.shape[:1] + world.mass.shape[2:], -1, dtype=torch.long, device=device) if world.zones is None else world.zones,
    )


def concat(worlds):
    """One batch of several VoxelWorlds. Worlds without object fields and with the same number of
    slots concatenate as they are; otherwise every world is filled in (with_objects) to the most
    slots any has."""
    plain = all(all(getattr(w, f) is None for f in EXTRA_FIELDS) for w in worlds) and len({w.cheese_alive.shape[1] for w in worlds}) == 1
    if not plain:
        k = max(w.cheese_alive.shape[1] for w in worlds)
        worlds = [with_objects(w, k) for w in worlds]
    if any(w.stuck is not None for w in worlds):
        worlds = [w if w.stuck is not None else replace(w, stuck=torch.zeros(w.mass.shape, dtype=torch.bool, device=w.mass.device)) for w in worlds]
    return VoxelWorld(*(None if getattr(worlds[0], f.name) is None else torch.cat([getattr(w, f.name) for w in worlds]) for f in fields(VoxelWorld)))


@dataclass
class StepInfo:
    pushed: torch.Tensor  # (B, K) bool
    eaten: torch.Tensor  # (B, K) bool


def cheese_voxels(pos):
    """(B, K, 8, 3): the voxels of each cheese (alive or not)."""
    offsets = torch.tensor([(a, b, c) for a in range(CHEESE_SIDE) for b in range(CHEESE_SIDE) for c in range(CHEESE_SIDE)], device=pos.device)
    return pos.unsqueeze(2) + offsets


SHELL_OFFSETS = torch.tensor(
    [(a, b, c) for a in range(-1, CHEESE_SIDE + 1) for b in range(-1, CHEESE_SIDE + 1) for c in range(-1, CHEESE_SIDE + 1) if not (0 <= a < CHEESE_SIDE and 0 <= b < CHEESE_SIDE and 0 <= c < CHEESE_SIDE)]
)
FULL_SHELL = len(SHELL_OFFSETS)  # 56


def flat_index(coords, dims):
    """coords (..., 3) -> (flat index within one scene clamped onto the grid, inside mask)."""
    d, h, w = dims
    z, y, x = coords.unbind(-1)
    inside = (z >= 0) & (z < d) & (y >= 0) & (y < h) & (x >= 0) & (x < w)
    return (z.clamp(0, d - 1) * h + y.clamp(0, h - 1)) * w + x.clamp(0, w - 1), inside


def gather(grid, coords, fill):
    """grid (B, D, H, W); coords (B, ..., 3) long -> (B, ...), fill off the grid."""
    b = grid.shape[0]
    flat, inside = flat_index(coords, grid.shape[-3:])
    values = grid.reshape(b, -1).gather(1, flat.reshape(b, -1)).view(flat.shape)
    return torch.where(inside, values, torch.as_tensor(fill, dtype=grid.dtype, device=grid.device))


BOX_OFFSETS = torch.tensor([(a, b, c) for a in range(BOX) for b in range(BOX) for c in range(BOX)])  # (BOX^3, 3), the order of form.flatten
RIM_OFFSETS = torch.tensor([(a, b, c) for a in range(-1, BOX + 1) for b in range(-1, BOX + 1) for c in range(-1, BOX + 1)])  # (216, 3): the box with a voxel all round


def object_voxels(world, pos=None):
    """(B, K, BOX^3, 3) voxel coordinates of the box of each object and (B, K, BOX^3) bool, which
    of them the object fills (a world without forms: 2 x 2 x 2 cubes)."""
    pos = world.cheese_pos if pos is None else pos
    b, k = pos.shape[:2]
    form = default_form(b, k, pos.device) if world.form is None else world.form
    return pos.unsqueeze(2) + BOX_OFFSETS.to(pos.device), form.reshape(b, k, -1)


def _rim(form):
    """(B, K, BOX + 2, BOX + 2, BOX + 2) bool: the form placed in the box with a voxel all round."""
    out = torch.zeros(*form.shape[:2], BOX + 2, BOX + 2, BOX + 2, dtype=torch.bool, device=form.device)
    out[:, :, 1:-1, 1:-1, 1:-1] = form
    return out


def _shift(grid, axis, sign):
    """grid (..., n, n, n) bool moved one voxel along spatial axis (0 z, 1 y, 2 x) by sign; what
    leaves the box is dropped."""
    out = torch.zeros_like(grid)
    dim = grid.dim() - 3 + axis
    n = grid.shape[dim]
    if sign > 0:
        out.narrow(dim, 1, n - 1).copy_(grid.narrow(dim, 0, n - 1))
    else:
        out.narrow(dim, 0, n - 1).copy_(grid.narrow(dim, 1, n - 1))
    return out


def shell_mask(form):
    """(B, K, 216) bool over RIM_OFFSETS: the voxels touching each object by a face, edge or
    corner that are not part of it."""
    rim = _rim(form)
    halo = rim.clone()
    for axis in range(3):
        step_ = halo.clone()
        for sign in (-1, 1):
            step_ |= _shift(halo, axis, sign)
        halo = step_
    return (halo & ~rim).flatten(2)


def full_shell(world):
    """(B, K) the full shell size of each object (FULL_SHELL for a 2 x 2 x 2 cube)."""
    if world.form is None:
        return torch.full(world.cheese_alive.shape, float(FULL_SHELL), device=world.mass.device)
    return shell_mask(world.form).sum(dim=2).float()


def cheese_id_map(world, pos=None, alive=None):
    """(B, D, H, W) long: the index of the alive cheese at each voxel, -1 where none."""
    pos = world.cheese_pos if pos is None else pos
    alive = world.cheese_alive if alive is None else alive
    b, k = alive.shape
    dims = world.mass.shape[-3:]
    if world.form is None:
        flat, inside = flat_index(cheese_voxels(pos), dims)
        keep = inside & alive.unsqueeze(2)
    else:
        coords, in_form = object_voxels(world, pos)
        flat, inside = flat_index(coords, dims)
        keep = inside & alive.unsqueeze(2) & in_form
    ids = torch.full((b, dims[0] * dims[1] * dims[2]), -1, dtype=torch.long, device=pos.device)
    rows = torch.arange(b, device=pos.device).view(b, 1, 1).expand_as(flat)
    cheese = torch.arange(k, device=pos.device).view(1, k, 1).expand_as(flat)
    ids[rows[keep], flat[keep]] = cheese[keep]
    return ids.view(b, *dims)


def cheese_masks(world):
    """(B, K, D, H, W) bool, the voxels of each alive cheese (dense; for observations and facts)."""
    ids = cheese_id_map(world)
    k = world.cheese_alive.shape[1]
    return ids.unsqueeze(1) == torch.arange(k, device=ids.device).view(1, k, 1, 1, 1)


def shell(world):
    """(B, K, 56, 3) shell voxel coordinates of each cheese and (B, K, 56) open: inside the grid,
    not a wall, cheese alive. A world with forms: (B, K, 216, 3) over RIM_OFFSETS, open only on
    each object's shell (shell_mask)."""
    if world.form is not None:
        coords = world.cheese_pos.unsqueeze(2) + RIM_OFFSETS.to(world.cheese_pos.device)
        open_ = ~gather(world.walls, coords, True) & world.cheese_alive.unsqueeze(2) & shell_mask(world.form)
        return coords, open_
    coords = world.cheese_pos.unsqueeze(2) + SHELL_OFFSETS.to(world.cheese_pos.device)
    open_ = ~gather(world.walls, coords, True) & world.cheese_alive.unsqueeze(2)
    return coords, open_


DIR_TABLE = torch.tensor(DIRS + ((0, 0, 0),))  # (27, 3), STAY last
CUBE_MOVES = ~((DIR_TABLE[:, 1] != 0) & (DIR_TABLE[:, 2] != 0))  # (27,) bool: the 14 moves a pivoting cube can make (none along both horizontal axes), and STAY
MOVE_CLASSES = ("level", "up_over", "down_over", "vertical")  # how a move fails is set per class (failure_rates)
_SIDEWAYS = (DIR_TABLE[:, 1:] != 0).any(dim=1)
MOVE_CLASS = torch.where(DIR_TABLE[:, 0] == 0, 0, torch.where(~_SIDEWAYS, 3, torch.where(DIR_TABLE[:, 0] > 0, 1, 2)))  # (27,) index into MOVE_CLASSES
FAILURE_RATE = 0.15


def failure_rates(rate=FAILURE_RATE, **by_class):
    """(27,) float: the probability that each action does not happen (the robot stays), `rate`
    for every move unless its class (MOVE_CLASSES) is named, e.g. failure_rates(0.15,
    up_over=0.3). STAY never fails."""
    unknown = set(by_class) - set(MOVE_CLASSES)
    if unknown:
        raise ValueError(f"unknown move classes {sorted(unknown)}")
    per_class = torch.tensor([float(by_class.get(name, rate)) for name in MOVE_CLASSES])
    rates = per_class[MOVE_CLASS]
    rates[STAY] = 0.0
    return rates


def fail_moves(codes, rates, generator=None):
    """Action codes with each move dropped to STAY with its action's probability in rates (27,)."""
    fails = torch.rand(codes.shape, generator=generator) < rates[codes.long()]
    return torch.where(fails, torch.full_like(codes, STAY), codes)


# ---------------------------------------------------------------------------------------------
# Pivot rules: which of the cube moves a real pivoting cube can make from where it is
#
# Measured in MuJoCo (HARDWARE.md, "Ten cubes in one scene"; hardware/cube_swarm.py `plan`): a cube
# moves by pivoting about an edge it shares with a support, and it moves only when no other cube
# touches it face to face. The rule on the grid, for a cube at c:
#
#   one axis h (6 moves, a quarter turn): a support side s at right angles to h with a surface
#     at c + s that continues at c + h + s, and c + h, c - s, c + h - s empty (the swing). s is
#     the side below when that works (floor roll, traverse), else a side at the same height
#     (h up or down: concave up or down; h level: wall crawl). Never the side above.
#   up and sideways e (4 moves, a half turn, convex up): a support at c + e; lands on it at
#     c + up + e; c + up, c - e, c + up - e, c + 2 up, c + 2 up + e empty.
#   down and sideways e (4 moves, a half turn, convex down): a support below; lands beside it at
#     c + e + down; c + e, c + up, c + e + up, c + 2 e, c + 2 e + down empty.
#   free: no cube on any of the six faces except the support's ("free_cube"). The floor, terrain
#     and cheeses are not bonds. "switch_cube" drops this condition (a cube that can release its
#     bonds); the pivot and the swing stay.
#
# The floor, terrain and other cubes can be a support on any side; a cheese only from below
# (nothing to hinge on). A move into a cheese is a press, not a pivot, and is outside the rule.
#
# Within one step every cube pivots at once, so the world (refuse) adds: a cube does not pivot on
# a cube that is itself moving this step (its support at c + s, or the cube at c + h + s it comes
# to rest against). Both conditions are judged on the scene as the step begins.

RULES = ("free_cube", "switch_cube")
_UP, _DOWN = (1, 0, 0), (-1, 0, 0)
_SIDES = (_DOWN, (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0))  # supports, in the order tried (cube_swarm.SUPPORTS); none above
_FACES = (_UP,) + _SIDES
_PAD = 2


def _add(a, b, k=1):
    return tuple(x + k * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _padded(grid, floor, fill=False):
    """grid (B, D, H, W) bool with _PAD voxels around it: `floor` below z = 0, `fill` elsewhere."""
    b, d, h, w = grid.shape
    out = torch.full((b, d + 2 * _PAD, h + 2 * _PAD, w + 2 * _PAD), fill, dtype=torch.bool, device=grid.device)
    out[:, :_PAD] = floor
    out[:, _PAD : _PAD + d, _PAD : _PAD + h, _PAD : _PAD + w] = grid
    return out


def _at(padded, offset):
    """The padded grid read at c + offset for every voxel c: a (B, D, H, W) view."""
    d, h, w = (n - 2 * _PAD for n in padded.shape[1:])
    z, y, x = (_PAD + o for o in offset)
    return padded[:, z : z + d, y : y + h, x : x + w]


def pivot_moves(cubes, walls, cheese, release=False, rubs=False, sides=False):
    """(B, 27, D, H, W) bool: whether a cube standing in each voxel could make each move, given
    the other cubes (cubes), terrain (walls) and cheese voxels, all (B, D, H, W) bool. Defined for
    every voxel, occupied or not, so a planner can ask about cells a cube would pass through.
    STAY is always possible; moves along both horizontal axes never.
    release: the cube can switch its bonds off, so other cubes on its faces do not hold it.
    rubs: also refuse a move whose landing cell (and, for a half turn, the cell passed on the
    way) has a cube, terrain or the floor beside it along the pivot axis (sliding into a slot,
    HARDWARE.md: 3 of 22 got in). Off in both named rules.
    sides: also return (B, 27, D, H, W) int8, the support side of each possible move as an index
    into _SIDES (leans turns it into the voxels leaned on)."""
    surface = _padded(cubes | walls, True)  # what a cube can hinge on, on any side
    below = _padded(cubes | walls | cheese, True)  # what it can roll over; also everything that blocks
    others = _padded(cubes, False)
    inside = _padded(torch.ones_like(cubes), False)
    face_cubes = sum(_at(others, f).to(torch.int8) for f in _FACES)
    out = torch.zeros(cubes.shape[0], N_ACTIONS, *cubes.shape[1:], dtype=torch.bool, device=cubes.device)
    out[:, STAY] = True
    side = torch.zeros(out.shape, dtype=torch.int8, device=cubes.device) if sides else None
    for code, step_ in enumerate(DIRS):
        if not bool(CUBE_MOVES[code]):
            continue
        if sum(v != 0 for v in step_) == 1:
            pivots = [(step_, s, False) for s in _SIDES if not any(a * b for a, b in zip(s, step_))]
        elif step_[0] > 0:
            pivots = [(_UP, (0, step_[1], step_[2]), True)]
        else:
            pivots = [((0, step_[1], step_[2]), _DOWN, True)]
        has_pivot = torch.zeros_like(cubes)
        possible = torch.zeros_like(cubes)
        for h, s, half in pivots:
            support = below if s == _DOWN else surface
            swing = [h, _add((0, 0, 0), s, -1), _add(h, s, -1)] + ([_add(h, s), _add(h, h), _add(_add(h, h), s)] if half else [])
            can = _at(support, s) if half else _at(support, s) & _at(support, _add(h, s))
            for cell in swing:
                can = can & ~_at(below, cell)
            free = can if release else can & (face_cubes == _at(others, s).to(torch.int8))
            if rubs:
                axis = _cross(h, s)
                for cell in [_add(h, s), h] if half else [h]:
                    free = free & ~_at(surface, _add(cell, axis)) & ~_at(surface, _add(cell, axis, -1))
            if sides:
                side[:, code][free & ~has_pivot] = _SIDES.index(s)
            possible |= free & ~has_pivot
            has_pivot |= can
        out[:, code] = possible & _at(inside, step_)
    return (out, side) if sides else out


_SIDE_TABLE = torch.tensor(_SIDES)
_HALF_TURN = (DIR_TABLE != 0).sum(dim=1) == 2  # (27,)


def leans(here, code, side):
    """(M, 2, 3): the two voxels a cube at here (M, 3) leans on when it makes move code (M,) on
    support side `side` (M,) (pivot_moves' index): its support, and for a quarter turn the voxel
    it comes to rest against (for a half turn the support again)."""
    support = _SIDE_TABLE.to(here.device)[side.long()]
    step_ = DIR_TABLE.to(here.device)[code.long()]
    rest = torch.where(_HALF_TURN.to(here.device)[code.long()].unsqueeze(1), support, step_ + support)
    return torch.stack([here + support, here + rest], dim=1)


PRIMITIVES = ("floor_roll", "traverse", "wall_crawl", "concave_up", "concave_down", "convex_up", "convex_up_hang", "convex_down")  # cube_swarm's names; a floor roll is on the floor or terrain, a traverse on a cube
_FACE6 = torch.tensor([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
_EDGE12 = torch.tensor([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if abs(a) + abs(b) + abs(c) == 2])


def _read(grid, scene, coords, outside=False, floor=None):
    """grid (B, D, H, W) bool read at coords (M, ..., 3) of scenes scene (M,): `outside` off the
    grid, `floor` (default `outside`) below z = 0."""
    b, d, h, w = grid.shape
    flat, inside = flat_index(coords, (d, h, w))
    rows = scene.view(-1, *([1] * (flat.dim() - 1))).expand(flat.shape)
    out = torch.where(inside, grid.reshape(b, -1)[rows, flat], torch.tensor(outside, device=grid.device))
    if floor is not None:
        out = torch.where(coords[..., 0] < 0, torch.tensor(floor, device=grid.device), out)
    return out


def pivot_geometry(code, side):
    """(h, s, half) of M pivots: the motion h (M, 3), the support side s (M, 3) and whether it is a
    half turn (M,), from move codes and pivot_moves' support side indices (cube_swarm.plan's h, s)."""
    step_ = DIR_TABLE.to(code.device)[code.long()]
    s = _SIDE_TABLE.to(code.device)[side.long()]
    half = _HALF_TURN.to(code.device)[code.long()]
    return torch.where(half.unsqueeze(1), step_ - s, step_), s, half


def pivot_features(world, scene, here, code, side):
    """What a pivot is and what holds it back, for M pivots of cubes at here (M, 3) in scenes
    scene (M,) making move code (M,) on support side `side` (M,) (pivot_moves' index), on the scene
    as the step begins. A dict of (M,) long tensors, the same counts as cube_swarm.plan:
    primitive  index into PRIMITIVES
    bonds      cubes face to face with the mover, other than its support
    hinges     cubes touching it only by an edge, other than the edge it pivots on
    rubs       solid cells (cube, terrain, floor) beside the cells it passes into, along the pivot axis
    lone       1 when what it leans on (its support, and for a quarter turn the cell it comes to
               rest against) are cubes that touch no other cube or terrain by a face (a single cube
               or a stack of two on the floor, which a launch can shove or tip)"""
    device = here.device
    cubes = world.mass > 0.5
    solid = cubes | world.walls | (cheese_id_map(world) >= 0)
    h, s, half = pivot_geometry(code, side)
    up = s[:, 0] == 0
    support_cube = _read(cubes, scene, here + s)
    below_solid = _read(solid, scene, here + torch.tensor([-1, 0, 0], device=device), outside=False, floor=True)
    primitive = torch.where(
        half,
        torch.where(up, torch.where(below_solid, PRIMITIVES.index("convex_up"), PRIMITIVES.index("convex_up_hang")), PRIMITIVES.index("convex_down")),
        torch.where(
            s[:, 0] < 0,
            torch.where(support_cube, PRIMITIVES.index("traverse"), PRIMITIVES.index("floor_roll")),
            torch.where(h[:, 0] > 0, PRIMITIVES.index("concave_up"), torch.where(h[:, 0] < 0, PRIMITIVES.index("concave_down"), PRIMITIVES.index("wall_crawl"))),
        ),
    )
    faces = _FACE6.to(device)
    edges = _EDGE12.to(device)
    at_faces = _read(cubes, scene, here.unsqueeze(1) + faces)  # (M, 6)
    bonds = (at_faces & (faces.unsqueeze(0) != s.unsqueeze(1)).any(dim=2)).sum(dim=1)
    at_edges = _read(cubes, scene, here.unsqueeze(1) + edges)  # (M, 12)
    hinges = (at_edges & (edges.unsqueeze(0) != (h + s).unsqueeze(1)).any(dim=2)).sum(dim=1)
    axis = torch.linalg.cross(h, s)
    target = here + h + torch.where(half.unsqueeze(1), s, torch.zeros_like(s))
    passed = torch.stack([target, torch.where(half.unsqueeze(1), here + h, target)], dim=1)  # (M, 2, 3); a quarter turn passes into its target only
    hard = cubes | world.walls
    beside = torch.cat([passed + axis.unsqueeze(1), passed - axis.unsqueeze(1)], dim=1)  # (M, 4, 3)
    counted = torch.cat([torch.ones_like(half), half, torch.ones_like(half), half]).view(4, -1).t()
    rubs = (_read(hard, scene, beside, floor=True) & counted).sum(dim=1)
    rest = here + torch.where(half.unsqueeze(1), s, h + s)
    leaned = torch.stack([here + s, rest], dim=1)  # (M, 2, 3)
    leaned_cube = _read(cubes, scene, leaned)
    near = leaned.unsqueeze(2) + faces.view(1, 1, 6, 3)  # (M, 2, 6, 3)
    own = (near == here.view(-1, 1, 1, 3)).all(dim=3) | (near == leaned.flip(1).unsqueeze(2)).all(dim=3)
    touching = (_read(hard, scene, near) & ~own & leaned_cube.unsqueeze(2)).any(dim=(1, 2))
    lone = support_cube & ~touching
    return {"primitive": primitive, "bonds": bonds, "hinges": hinges, "rubs": rubs, "lone": lone.long()}


def rule_table(world, rule, sides=False):
    """pivot_moves of a scene batch under a named rule (RULES); a stuck robot (VoxelWorld.stuck)
    has STAY only."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}")
    out = pivot_moves(world.mass > 0.5, world.walls, cheese_id_map(world) >= 0, release=rule == "switch_cube", sides=sides)
    if world.stuck is not None:
        (out[0] if sides else out)[:, :STAY] &= ~world.stuck.unsqueeze(1)
    return out


def rule_moves(world, rule):
    """(N, 27) bool over the robots in grid order (controller.robot_list): the actions each robot
    can carry out now under the rule. Shaped as the controller's `allowed`."""
    table = rule_table(world, rule)
    return table.permute(0, 2, 3, 4, 1)[world.mass > 0.5]


def refuse(world, codes, rule, ids=None, priority=None):
    """Action codes (B, D, H, W) with every move the rule forbids replaced by STAY: a move no
    pivot makes from there, and a pivot on a cube that is itself pivoting this step. A move into
    a cheese is a press and passes.
    priority: (B, D, H, W) float, for a controller cleaning up its own choice: of a cube and the
    cube it pivots on, only the one with the lower priority is stopped, so what is left passes
    the plain check. Without it both stay, as in the world's step."""
    b, d, h, w = world.mass.shape
    ids = cheese_id_map(world) if ids is None else ids
    scene, z, y, x = (world.mass > 0.5).nonzero(as_tuple=True)
    code = codes[scene, z, y, x].long()
    there = torch.stack([z, y, x], dim=1) + DIR_TABLE.to(codes.device)[code]
    flat, inside = flat_index(there, (d, h, w))
    press = inside & (ids.reshape(b, -1)[scene, flat] >= 0)
    if world.friction is not None:  # robots pushing in line behind a robot that presses a heavy object press too
        press = chain_press(world, codes, ids)[scene, z, y, x] >= 0
    table, side = rule_table(world, rule, sides=True)
    pivots = table[scene, code, z, y, x] & ~press & (code != STAY)
    here = torch.stack([z, y, x], dim=1)
    moving = torch.zeros(b, d * h * w, dtype=torch.bool, device=codes.device)
    moving[scene[pivots], flat_index(here[pivots], (d, h, w))[0]] = True
    lean_flat, lean_inside = flat_index(leans(here, code, side[scene, code, z, y, x]), (d, h, w))
    if priority is None:
        unsteady = (moving[scene.unsqueeze(1), lean_flat] & lean_inside).any(dim=1)
    else:
        rank = torch.where(pivots, priority[scene, z, y, x].float(), -1.0)
        rank_grid = torch.full((b * d * h * w,), -1.0, device=codes.device)
        rank_grid[scene * d * h * w + flat_index(here, (d, h, w))[0]] = rank
        lean_cells = scene.unsqueeze(1) * d * h * w + lean_flat
        support_rank = torch.where(lean_inside, rank_grid[lean_cells], -1.0).max(dim=1).values
        load = torch.full((b * d * h * w,), -1.0, device=codes.device)  # the highest priority of the movers leaning on each voxel
        load.scatter_reduce_(0, lean_cells[pivots].reshape(-1), rank[pivots].repeat_interleave(2), "amax")
        unsteady = (support_rank > rank) | (load[scene * d * h * w + flat_index(here, (d, h, w))[0]] > rank)
    ok = (pivots & ~unsteady) | press | (code == STAY)
    out = codes.clone()
    out[scene[~ok], z[~ok], y[~ok], x[~ok]] = STAY
    return out


def chain_press(world, codes, ids=None):
    """(B, D, H, W) long: the object each robot presses on, -1 for none. A robot presses on the
    object its move lands in; under a world with frictions, also on a heavy object (friction
    above FRICTION) when its move lands on a robot that presses on it with the same move, up to
    CHAIN_DEPTH robots behind the first."""
    b, d, h, w = world.mass.shape
    v = d * h * w
    device = world.mass.device
    ids = cheese_id_map(world) if ids is None else ids
    occupied = (world.mass > 0.5).reshape(-1)
    robots = occupied.nonzero().squeeze(1)
    scene = robots // v
    local = robots % v
    here = torch.stack([local // (h * w), (local // w) % h, local % w], dim=1)
    code = codes.reshape(-1)[robots].long()
    flat, inside = flat_index(here + DIR_TABLE.to(device)[code], (d, h, w))
    dest = scene * v + flat
    moves = inside & (code != STAY)
    target = torch.where(moves, ids.reshape(-1)[dest], -1)
    out = torch.full((b * v,), -1, dtype=torch.long, device=device)
    out[robots] = target
    if world.friction is None:
        return out.view(b, d, h, w)
    heavy = world.friction > FRICTION
    grid_code = torch.full((b * v,), STAY, dtype=torch.long, device=device)
    grid_code[robots] = code
    for _ in range(CHAIN_DEPTH):
        ahead = out[dest]
        joins = (target < 0) & moves & occupied[dest] & (grid_code[dest] == code) & (ahead >= 0)
        joins &= heavy[scene, ahead.clamp(min=0)]
        if not joins.any():
            break
        target = torch.where(joins, ahead, target)
        out[robots] = target
    return out.view(b, d, h, w)


def move_robots(world, codes, ids, score=None):
    """Move robots one voxel by their action codes (B, D, H, W) long in 0..26, in rounds with
    random priority. Works on the list of robots, not the dense grid. Returns the new occupancy
    and the (B, K, 3) press on each cheese (the summed directions of robots moving into it, and
    for a heavy object of the robots in line behind them: chain_press)."""
    b, d, h, w = world.mass.shape
    v = d * h * w
    k = world.cheese_alive.shape[1]
    device = world.mass.device
    occupied = (world.mass > 0.5).reshape(-1).clone()
    robots = occupied.nonzero().squeeze(1)
    scene = robots // v
    local = robots % v
    here = torch.stack([local // (h * w), (local // w) % h, local % w], dim=1)
    code = codes.reshape(-1)[robots].long()
    there = here + DIR_TABLE.to(device)[code]
    flat, inside = flat_index(there, (d, h, w))
    dest = scene * v + flat
    dest_cheese = torch.where(inside, ids.reshape(-1)[dest], -1)
    if world.friction is not None:
        dest_cheese = chain_press(world, codes, ids).reshape(-1)[robots]
    force = torch.zeros(b * k, 3, dtype=world.mass.dtype, device=device)
    pressing = dest_cheese >= 0
    force.index_add_(0, scene[pressing] * k + dest_cheese[pressing], DIR_TABLE.to(device)[code[pressing]].to(force.dtype))
    blocked = world.walls.reshape(-1) | (ids.reshape(-1) >= 0)
    pending = (code != STAY) & inside & ~blocked[dest]
    if score is None:
        score = torch.rand(robots.shape, device=device)
    else:
        score = score.reshape(-1)[robots]
    while True:
        claim = pending & ~occupied[dest]
        if not claim.any():
            break
        best = torch.full((b * v,), -1.0, device=device).scatter_reduce(0, dest[claim], score[claim], "amax")
        win = claim & (score == best[dest])
        occupied[robots[win]] = False
        occupied[dest[win]] = True
        pending &= ~win
    return occupied.view(b, d, h, w).to(world.mass.dtype), force.view(b, k, 3)


def _face(pos, axis, offset):
    """(B, 4, 3): the 2 x 2 voxels at pos[axis] + offset, spanning the cheese on the other axes."""
    b = pos.shape[0]
    others = [a for a in range(3) if a != axis]
    grid = torch.tensor([(i, j) for i in range(CHEESE_SIDE) for j in range(CHEESE_SIDE)], device=pos.device)
    out = pos.view(b, 1, 3).repeat(1, CHEESE_SIDE * CHEESE_SIDE, 1)
    out[..., others[0]] += grid[:, 0]
    out[..., others[1]] += grid[:, 1]
    out[..., axis] = pos[:, axis].view(b, 1) + offset.view(b, 1)
    return out


def push_cheeses(world, mass, force):
    """Slide cheeses whose press reaches FRICTION one voxel along the press's strongest axis.
    Robots in the leading face are shoved one voxel further (all or none). Cheeses are handled
    one slot at a time. Returns (new mass, new positions, pushed (B, K))."""
    if world.form is not None:
        return push_objects(world, mass, force)
    b, k = world.cheese_alive.shape
    pos = world.cheese_pos.clone()
    mass = mass.clone()
    pushed = torch.zeros(b, k, dtype=torch.bool, device=mass.device)
    for c in range(k):
        f = force[:, c]
        axis = f.abs().argmax(dim=1)
        strength = f.gather(1, axis.view(-1, 1)).squeeze(1)
        want = world.cheese_alive[:, c] & (strength.abs() >= FRICTION)
        if not want.any():
            continue
        sign = torch.sign(strength).long()
        others = world.cheese_alive.clone()
        others[:, c] = False
        other_ids = cheese_id_map(world, pos, others)
        for a in range(3):
            sel = want & (axis == a)
            if not sel.any():
                continue
            lead1 = _face(pos[:, c], a, torch.where(sign > 0, CHEESE_SIDE, -1))
            lead2 = lead1.clone()
            lead2[..., a] += sign.view(b, 1)
            blocked1 = gather(world.walls, lead1, True) | (gather(other_ids, lead1, -1) >= 0)
            blocked2 = gather(world.walls, lead2, True) | (gather(other_ids, lead2, -1) >= 0)
            robot1 = gather(mass, lead1, 0.0) > 0.5
            robot2 = gather(mass, lead2, 0.0) > 0.5
            shove_ok = ~robot1 | (~blocked2 & ~robot2)
            can = sel & ~blocked1.any(dim=1) & shove_ok.all(dim=1)
            if not can.any():
                continue
            flat = mass.view(b, -1)
            moving = (can.view(b, 1) & robot1).to(mass.dtype)
            flat.scatter_add_(1, flat_index(lead1, mass.shape[-3:])[0], -moving)
            flat.scatter_add_(1, flat_index(lead2, mass.shape[-3:])[0], moving)
            step_ = torch.zeros(b, 3, dtype=torch.long, device=mass.device)
            step_[:, a] = sign
            pos[:, c] = torch.where(can.view(b, 1), pos[:, c] + step_, pos[:, c])
            pushed[:, c] |= can
    return mass, pos, pushed


def push_objects(world, mass, force):
    """push_cheeses for a world with forms: an object whose press reaches its friction slides one
    voxel along the press's strongest axis if the voxels it moves into (its leading voxels) are
    clear of walls, other objects and the grid edge; robots in them are shoved one voxel further
    (all or none). For a 2 x 2 x 2 cube this is push_cheeses."""
    b, k = world.cheese_alive.shape
    pos = world.cheese_pos.clone()
    mass = mass.clone()
    slides = torch.zeros(b, k, dtype=torch.bool, device=mass.device)
    rim = _rim(world.form)
    offsets = RIM_OFFSETS.to(mass.device)
    friction = world.friction if world.friction is not None else torch.full((b, k), FRICTION, device=mass.device)
    for c in range(k):
        f = force[:, c]
        axis = f.abs().argmax(dim=1)
        strength = f.gather(1, axis.view(-1, 1)).squeeze(1)
        want = world.cheese_alive[:, c] & (strength.abs() >= friction[:, c])
        if not want.any():
            continue
        sign = torch.sign(strength).long()
        others = world.cheese_alive.clone()
        others[:, c] = False
        other_ids = cheese_id_map(world, pos, others)
        for a in range(3):
            for s in (-1, 1):
                sel = want & (axis == a) & (sign == s)
                if not sel.any():
                    continue
                leading = (_shift(rim[:, c], a, s) & ~rim[:, c]).flatten(1)  # (B, 216)
                lead1 = pos[:, c].unsqueeze(1) + offsets
                lead2 = lead1.clone()
                lead2[..., a] += s
                blocked1 = (gather(world.walls, lead1, True) | (gather(other_ids, lead1, -1) >= 0)) & leading
                blocked2 = gather(world.walls, lead2, True) | (gather(other_ids, lead2, -1) >= 0)
                robot1 = (gather(mass, lead1, 0.0) > 0.5) & leading
                robot2 = gather(mass, lead2, 0.0) > 0.5
                shove_ok = ~robot1 | (~blocked2 & ~robot2)
                can = sel & ~blocked1.any(dim=1) & shove_ok.all(dim=1)
                if not can.any():
                    continue
                flat = mass.view(b, -1)
                movers = (can.view(b, 1) & robot1).to(mass.dtype)
                flat.scatter_add_(1, flat_index(lead1, mass.shape[-3:])[0], -movers)
                flat.scatter_add_(1, flat_index(lead2, mass.shape[-3:])[0], movers)
                step_ = torch.zeros(b, 3, dtype=torch.long, device=mass.device)
                step_[:, a] = s
                pos[:, c] = torch.where(can.view(b, 1), pos[:, c] + step_, pos[:, c])
                slides[:, c] |= can
    return mass, pos, slides


def _grow_faces(held, out):
    """out |= held moved one voxel along each of the six faces (in place)."""
    out[:, 1:] |= held[:, :-1]
    out[:, :-1] |= held[:, 1:]
    out[:, :, 1:] |= held[:, :, :-1]
    out[:, :, :-1] |= held[:, :, 1:]
    out[:, :, :, 1:] |= held[:, :, :, :-1]
    out[:, :, :, :-1] |= held[:, :, :, 1:]
    return out


def supported(world, occupied=None):
    """(B, D, H, W) bool: robots connected through face-adjacent robots to the floor (z = 0), a
    wall or a cheese."""
    occupied = (world.mass > 0.5) if occupied is None else occupied
    solid = world.walls | (cheese_id_map(world) >= 0)
    seed = _grow_faces(solid, solid.clone())
    seed[:, 0] = True
    held = occupied & seed
    count = int(held.sum())
    while True:
        held = _grow_faces(held, held.clone()) & occupied
        new_count = int(held.sum())
        if new_count == count:
            return held
        count = new_count


def fall(world):
    """Gravity: unsupported robots and cheeses drop one voxel (scenes with gravity on)."""
    on = world.gravity.nonzero().squeeze(1)
    if on.numel() == 0:
        return world
    sub = world.index(on)
    occupied = sub.mass > 0.5
    loose = occupied & ~supported(sub, occupied)
    solid = sub.walls | (cheese_id_map(sub) >= 0)
    stuck = None if sub.stuck is None else sub.stuck.clone()
    for z in range(1, occupied.shape[1]):
        drop = loose[:, z] & ~occupied[:, z - 1] & ~solid[:, z - 1]
        occupied[:, z - 1] |= drop
        occupied[:, z] &= ~drop
        if stuck is not None:  # a stuck robot falls with its column
            stuck[:, z - 1] |= drop & stuck[:, z]
            stuck[:, z] &= ~drop
    pos = sub.cheese_pos.clone()
    below = None if sub.form is None else (_shift(_rim(sub.form), 0, -1) & ~_rim(sub.form)).flatten(2)  # (B, K, 216): the voxels just under each object's lowest voxels
    for c in range(pos.shape[1]):
        others = sub.cheese_alive.clone()
        others[:, c] = False
        other = cheese_id_map(sub, pos, others) >= 0
        if below is None:
            under = _face(pos[:, c], 0, torch.full_like(pos[:, c, 0], -1))
            clear = ~gather(sub.walls | other | occupied, under, True).any(dim=1)
        else:
            under = pos[:, c].unsqueeze(1) + RIM_OFFSETS.to(pos.device)
            clear = ~(gather(sub.walls | other | occupied, under, True) & below[:, c]).any(dim=1)
        pos[:, c, 0] -= (sub.cheese_alive[:, c] & clear).long()
    mass = world.mass.clone()
    mass[on] = occupied.to(mass.dtype)
    cheese_pos = world.cheese_pos.clone()
    cheese_pos[on] = pos
    if stuck is None:
        return replace(world, mass=mass, cheese_pos=cheese_pos)
    all_stuck = world.stuck.clone()
    all_stuck[on] = stuck
    return replace(world, mass=mass, cheese_pos=cheese_pos, stuck=all_stuck)


def coverage(world):
    """(covered, open) shell voxel counts per cheese, (B, K) each."""
    coords, open_ = shell(world)
    covered = (gather(world.mass, coords, 0.0) > 0.5) & open_
    return covered.sum(dim=2), open_.sum(dim=2)


def eat_cheeses(world):
    covered, open_ = coverage(world)
    if world.form is None:
        return world.cheese_alive & (covered >= EAT_SHARE * open_) & (covered >= EAT_MIN_SHARE * FULL_SHELL)
    food = world.food if world.food is not None else torch.ones_like(world.cheese_alive)
    return world.cheese_alive & food & (covered >= EAT_SHARE * open_) & (covered >= EAT_MIN_SHARE * full_shell(world))


# ---------------------------------------------------------------------------------------------
# Move outcomes measured in MuJoCo (README "Move outcomes"; hardware/cube_outcomes.py)
#
# Without an outcome model a pivot the rule allows simply happens (callers drop moves at random
# with fail_moves). With one, every pivot gets the outcome MuJoCo gives its class (pivot_class:
# primitive, and whether it has bonds, hinges, rubs and a lone support): reached (it moves),
# stayed, or one of the harmful ends MuJoCo left behind in a trial of that class, replayed on the
# grid in the move's frame: the mover and any knocked cube go to the cells they ended in, and a
# cube that ended off the lattice is stuck there (VoxelWorld.stuck: it never acts again, falls
# with its column, and blocks a sliding object). Thinly measured classes borrow from coarser ones:
# each class's (reached, stayed, harmful) is its counts plus PRIOR trials' worth of the class one
# tag coarser (BACKOFF), down to the primitive and to all moves; a harmful draw replays a trial of
# the finest class that has one.

# Tables in outcomes/<name>.json. mujoco_release: cube_swarm with release (switchable magnets, the
# switch_cube rule). mujoco_release_v2: the same with climbs at 0.85 of cube_swarm.RPM (HARDWARE.md,
# "Cutting harm"), judged with the fixes of 2026-11-11. mujoco_release_v3 (the demo's, demos.DEMO_MOVES):
# edge and face magnets x1.4 and the per-primitive kick plans of
# runs/cube_outcomes/settings/plans-m14-sel.json (floor rolls and traverses at 1.2 of RPM, wall crawls
# 1.1, climbs and descents with escalating retries within the move: HARDWARE.md, "Kick plans and
# retries"); a move's outcome is that of its last attempt.
OUTCOME_MODELS = ("mujoco_release", "mujoco_release_v2", "mujoco_release_v3")
TAGS = ("bonded", "hinged", "rubs", "lone")
BACKOFF = (TAGS, ("bonded", "hinged", "lone"), ("bonded", "hinged"), ("bonded",), ())  # finest to coarsest
PRIOR = 8.0  # trials' worth of the coarser class each class's counts are shrunk toward
N_CLASSES = len(PRIMITIVES) << len(TAGS)


def pivot_class(features):
    """(M,) long class index of pivots from pivot_features: primitive, then a bit per tag."""
    bits = [features["bonds"] > 0, features["hinges"] > 0, features["rubs"] > 0, features["lone"] > 0]
    out = features["primitive"].clone()
    for bit in bits:
        out = out * 2 + bit.long()
    return out


def class_name(index, tags=TAGS):
    """The key of class `index` (cube_outcomes.class_key) with only `tags` kept."""
    bits = [(index >> (len(TAGS) - 1 - k)) & 1 for k in range(len(TAGS))]
    kept = [tag for tag, bit in zip(TAGS, bits) if bit and tag in tags]
    return "/".join([PRIMITIVES[index >> len(TAGS)]] + kept)


def _coarse(key, tags):
    name, *rest = key.split("/")
    return "/".join([name] + [tag for tag in rest if tag in tags])


class OutcomeModel:
    """Outcome probabilities per pivot class from a measured table (cube_outcomes fit):
    {"name", "classes": {key: {"reached": n, "stayed": n, "harm": [aftermath, ...]}}}, an aftermath
    being one MuJoCo trial's harmful end: [[from, to, stuck], ...] for the mover (first, from
    (0, 0, 0)) and every cube it knocked, cells in the move's frame (h, s, h x s)."""

    def __init__(self, table):
        self.name = table["name"]
        pooled = {}  # (backoff level, key with that level's tags): counts of every class that falls in it
        for key, entry in table["classes"].items():
            for tags in BACKOFF:
                into = pooled.setdefault((tags, _coarse(key, tags)), {"reached": 0, "stayed": 0, "harm": []})
                into["reached"] += entry["reached"]
                into["stayed"] += entry["stayed"]
                into["harm"] += entry["harm"]
        everything = [sum(e[k] for e in table["classes"].values()) for k in ("reached", "stayed")] + [sum(len(e["harm"]) for e in table["classes"].values())]
        self.reach = torch.zeros(N_CLASSES)
        self.stay = torch.zeros(N_CLASSES)
        self.first = torch.zeros(N_CLASSES, dtype=torch.long)
        self.count = torch.zeros(N_CLASSES, dtype=torch.long)
        self.source = []  # the finest key with trials that each class's counts come from, None when its primitive was never measured
        self.aftermaths = []
        for index in range(N_CLASSES):
            probs = torch.tensor(everything, dtype=torch.float) / max(sum(everything), 1)
            harm, source = [], None
            for tags in reversed(BACKOFF):  # coarsest first: each level is its counts plus PRIOR trials of the level above
                entry = pooled.get((tags, class_name(index, tags)))
                if entry is None:
                    continue
                counts = torch.tensor([entry["reached"], entry["stayed"], len(entry["harm"])], dtype=torch.float)
                probs = (counts + PRIOR * probs) / (counts.sum() + PRIOR)
                harm = entry["harm"] or harm
                source = class_name(index, tags)
            if source is None:
                self.source.append(None)
                self.reach[index] = 1.0  # never measured: the move happens
                continue
            if not harm:  # no harmful end of this primitive was seen: nothing to replay
                probs[2] = 0.0
                probs = probs / probs.sum()
            self.source.append(source)
            self.reach[index], self.stay[index] = probs[0], probs[1]
            self.first[index] = len(self.aftermaths)
            self.count[index] = len(harm)
            self.aftermaths += [torch.tensor([[*a, *b, int(c)] for a, b, c in one], dtype=torch.long) for one in harm]

    @classmethod
    def load(cls, name):
        """The model named `name` (OUTCOME_MODELS), from outcomes/<name>.json beside this module."""
        import json
        from pathlib import Path

        if name not in OUTCOME_MODELS:
            raise ValueError(f"unknown outcome model {name!r}")
        return cls(json.loads((Path(__file__).parent / "outcomes" / f"{name}.json").read_text()))

    def probabilities(self, index):
        """(reached, stayed, harmful) of class `index`."""
        return float(self.reach[index]), float(self.stay[index]), float(1 - self.reach[index] - self.stay[index])

    def odds(self, world, rule, scene=None, here=None):
        """(M, 27, 3) float: the (reached, stayed, harmful) odds of every move of cubes at here
        (M, 3) in scenes scene (M,) (default: every robot, in grid order as controller.robot_list
        and rule_moves), for a planner that weighs what its moves risk. A move the rule does not
        allow, and STAY, is (0, 1, 0)."""
        occupied = world.mass > 0.5
        if scene is None:
            scene, z, y, x = occupied.nonzero(as_tuple=True)
            here = torch.stack([z, y, x], dim=1)
        table, side = rule_table(world, rule, sides=True)
        out = torch.zeros(len(scene), N_ACTIONS, 3)
        out[..., 1] = 1.0
        allowed = table[scene, :, here[:, 0], here[:, 1], here[:, 2]].clone()  # (M, 27)
        allowed[:, STAY] = False
        row, code = allowed.nonzero(as_tuple=True)
        if len(row):
            at = here[row]
            sides = side[scene[row], code, at[:, 0], at[:, 1], at[:, 2]]
            index = pivot_class(pivot_features(world, scene[row], at, code, sides))
            reach, stay = self.reach[index], self.stay[index]
            out[row, code] = torch.stack([reach, stay, (1 - reach - stay).clamp(min=0.0)], dim=1)
        return out

    def draw(self, world, codes, rule, ids, noise):
        """Outcomes of the pivots among codes (after refuse) drawn with noise (B, D, H, W) uniform:
        returns (codes with every pivot that does not reach its target turned to STAY, the harmful
        events (scene, here, h, s, aftermath index) to apply after the robots move)."""
        cubes = world.mass > 0.5
        scene, z, y, x = (cubes & (codes != STAY)).nonzero(as_tuple=True)
        here = torch.stack([z, y, x], dim=1)
        code = codes[scene, z, y, x].long()
        if world.friction is not None:
            press = chain_press(world, codes, ids)[scene, z, y, x] >= 0
        else:
            flat, inside = flat_index(here + DIR_TABLE.to(codes.device)[code], world.mass.shape[-3:])
            press = inside & (ids.reshape(ids.shape[0], -1)[scene, flat] >= 0)
        table, side = rule_table(world, rule, sides=True)
        pivots = table[scene, code, z, y, x] & ~press
        scene, here, code = scene[pivots], here[pivots], code[pivots]
        sides = side[scene, code, here[:, 0], here[:, 1], here[:, 2]]
        index = pivot_class(pivot_features(world, scene, here, code, sides))
        u = noise[scene, here[:, 0], here[:, 1], here[:, 2]].float()
        reach, stay = self.reach[index], self.stay[index]
        out = codes.clone()
        miss = u >= reach
        out[scene[miss], here[miss, 0], here[miss, 1], here[miss, 2]] = STAY
        harm = (u >= reach + stay) & (self.count[index] > 0)
        rest = (1 - reach - stay)[harm].clamp(min=1e-9)
        pick = ((u[harm] - (reach + stay)[harm]) / rest * self.count[index][harm]).long().clamp(min=0)
        pick = self.first[index][harm] + torch.minimum(pick, self.count[index][harm] - 1)
        h, s, _ = pivot_geometry(code[harm], sides[harm])
        return out, (scene[harm], here[harm], h, s, pick)

    def apply(self, mass, stuck, blocked, events):
        """Replay harmful events on mass (B, D, H, W) after the robots moved: each participant of
        the aftermath that is still in its cell goes to the cell it ended in, if that is inside the
        grid and free (else it stays), stuck if it ended off the lattice. blocked: walls and
        objects. Returns (mass, stuck)."""
        scene, here, h, s, pick = events
        if not len(scene):
            return mass, stuck
        mass, stuck = mass.clone(), stuck.clone()
        d, hh, w = mass.shape[-3:]
        frames = torch.stack([h, s, torch.linalg.cross(h, s)], dim=2)  # (E, 3, 3): world = frame @ local
        for e in range(len(scene)):
            b = int(scene[e])
            rows = self.aftermaths[int(pick[e])]
            start = here[e] + rows[:, 0:3] @ frames[e].t()
            end = here[e] + rows[:, 3:6] @ frames[e].t()
            present = [k for k in range(len(rows)) if (start[k] >= 0).all() and start[k, 0] < d and start[k, 1] < hh and start[k, 2] < w and mass[b, start[k, 0], start[k, 1], start[k, 2]] > 0.5]
            was_stuck = {k: bool(stuck[b, start[k, 0], start[k, 1], start[k, 2]]) for k in present}
            for k in present:
                mass[b, start[k, 0], start[k, 1], start[k, 2]] = 0
                stuck[b, start[k, 0], start[k, 1], start[k, 2]] = False
            for k in present:
                cell = end[k]
                free = (cell >= 0).all() and cell[0] < d and cell[1] < hh and cell[2] < w and mass[b, cell[0], cell[1], cell[2]] < 0.5 and not blocked[b, cell[0], cell[1], cell[2]]
                at = cell if free else start[k]
                mass[b, at[0], at[1], at[2]] = 1
                stuck[b, at[0], at[1], at[2]] = was_stuck[k] or bool(rows[k, 6])
        return mass, stuck


def step(world, codes, score=None, rule=None, outcomes=None, noise=None):
    """Advance every scene one step under action codes (B, D, H, W). Returns (world, StepInfo).
    rule: a pivot rule (RULES); moves it forbids do not happen (refuse). None: any move.
    outcomes: an OutcomeModel (needs a rule): each pivot reaches, stays or ends as MuJoCo shows
    for its class, drawn with noise (B, D, H, W) uniform in [0, 1) (default torch.rand). None:
    every allowed pivot happens, as before. A stuck robot (VoxelWorld.stuck) never acts."""
    ids = cheese_id_map(world)
    if rule is not None:
        codes = refuse(world, codes, rule, ids)
    if world.stuck is not None:
        codes = torch.where(world.stuck, torch.full_like(codes, STAY), codes)
    events = None
    if outcomes is not None:
        if rule is None:
            raise ValueError("an outcome model needs a pivot rule")
        noise = torch.rand(world.mass.shape, device=world.mass.device) if noise is None else noise
        codes, events = outcomes.draw(world, codes, rule, ids, noise)
    mass, force = move_robots(world, codes, ids, score)
    stuck = world.stuck
    if events is not None:
        stuck = torch.zeros(mass.shape, dtype=torch.bool, device=mass.device) if stuck is None else stuck
        mass, stuck = outcomes.apply(mass, stuck, world.walls | (ids >= 0), events)
        world = replace(world, stuck=stuck)
    mass, pos, pushed = push_cheeses(world if stuck is None else replace(world, walls=world.walls | stuck), mass, force)
    moved = fall(replace(world, mass=mass, cheese_pos=pos))
    eaten = eat_cheeses(moved)
    return replace(moved, cheese_alive=moved.cheese_alive & ~eaten), StepInfo(pushed=pushed, eaten=eaten)


def paint_of(form, pattern):
    """(B, K, BOX, BOX, BOX) bool: the voxels of each object in its second colour. Patterns run
    along the object's longest axis (ties: x, then y, then z): striped alternates voxel layers
    (first colour first), two-tone gives the far half (the larger half the first colour)."""
    ext = torch.stack([form.any(dim=-1).any(dim=-1).sum(-1), form.any(dim=-1).any(dim=-2).sum(-1), form.any(dim=-2).any(dim=-2).sum(-1)], dim=-1)  # (B, K, 3) extents along z, y, x
    axis = (ext * 4 + torch.tensor([0, 1, 2], device=form.device)).argmax(dim=-1)  # (B, K)
    grid = torch.stack(torch.meshgrid(*(torch.arange(BOX, device=form.device),) * 3, indexing="ij"))  # (3, BOX, BOX, BOX)
    along = grid[axis]  # (B, K, BOX, BOX, BOX)
    length = ext.gather(-1, axis.unsqueeze(-1)).view(*axis.shape, 1, 1, 1)
    striped = along % 2 == 1
    halves = along >= (length + 1) // 2
    p = pattern.view(*pattern.shape, 1, 1, 1)
    return form & torch.where(p == PATTERNS.index("striped"), striped, torch.where(p == PATTERNS.index("two-tone"), halves, torch.zeros_like(form)))


def colour_map(world):
    """(B, D, H, W) long: the palette colour of the object voxel there (its pattern applied), -1
    where none."""
    b, k = world.cheese_alive.shape
    dims = world.mass.shape[-3:]
    coords, in_form = object_voxels(world)
    second = paint_of(world.form, world.pattern).reshape(b, k, -1) & (world.colour2 >= 0).unsqueeze(-1)
    colour = torch.where(second, world.colour2.unsqueeze(-1), world.cheese_color.unsqueeze(-1))
    flat, inside = flat_index(coords, dims)
    keep = inside & world.cheese_alive.unsqueeze(2) & in_form
    out = torch.full((b, dims[0] * dims[1] * dims[2]), -1, dtype=torch.long, device=coords.device)
    rows = torch.arange(b, device=coords.device).view(b, 1, 1).expand_as(flat)
    out[rows[keep], flat[keep]] = colour[keep]
    return out.view(b, *dims)


def observe(world, channels=N_CHANNELS):
    """(B, channels, D, H, W) float, the channels of CHANNELS in order: robot, wall, object mask,
    object RGB (per voxel, so patterns show), food (object voxels that can be eaten), zone RGB
    (on the voxel just above the ground of each painted floor cell, where a robot standing on it
    would be). A world without object fields gives the observation it always gave in the first
    OLD_CHANNELS channels, food on every cheese voxel and no zones. Weight is not observed."""
    ids = cheese_id_map(world)
    present = ids >= 0
    palette = COLOUR_RGB.to(world.mass.device)
    if world.form is None or world.colour2 is None or world.pattern is None:
        colour = palette[world.cheese_color.clamp(min=0)]  # (B, K, 3)
        rgb = colour.gather(1, ids.clamp(min=0).view(ids.shape[0], -1, 1).expand(-1, -1, 3)).view(*ids.shape, 3)
    else:
        rgb = palette[colour_map(world).clamp(min=0)]
    rgb = (rgb * present.unsqueeze(-1)).permute(0, 4, 1, 2, 3)
    parts = [world.mass.unsqueeze(1), world.walls.float().unsqueeze(1), present.float().unsqueeze(1), rgb]
    if channels > OLD_CHANNELS:
        food = present if world.food is None else present & world.food.gather(1, ids.clamp(min=0).flatten(1)).view_as(ids)
        zone = torch.zeros(ids.shape[0], 3, *ids.shape[1:], device=world.mass.device)
        if world.zones is not None and (world.zones >= 0).any():
            scene, y, x = (world.zones >= 0).nonzero(as_tuple=True)
            z = world.walls[scene, :, y, x].sum(dim=1).clamp(max=ids.shape[1] - 1)
            zone[scene, :, z, y, x] = palette[world.zones[scene, y, x]]
        parts += [food.float().unsqueeze(1), zone]
    return torch.cat(parts, dim=1)[:, :channels]


def world_from_arrays(mass, walls=None, cheeses=(), gravity=False):
    """One scene from arrays (D, H, W) and cheeses [(z, y, x, colour)], for tests."""
    mass = torch.as_tensor(mass, dtype=torch.float32).unsqueeze(0)
    walls = torch.zeros_like(mass, dtype=torch.bool) if walls is None else torch.as_tensor(walls, dtype=torch.bool).unsqueeze(0)
    pos = torch.zeros(1, MAX_CHEESES, 3, dtype=torch.long)
    alive = torch.zeros(1, MAX_CHEESES, dtype=torch.bool)
    colour = torch.full((1, MAX_CHEESES), -1, dtype=torch.long)
    for i, (z, y, x, c) in enumerate(cheeses):
        pos[0, i] = torch.tensor([z, y, x])
        alive[0, i] = True
        colour[0, i] = c
    return VoxelWorld(mass, walls, pos, alive, colour, torch.tensor([gravity]))


LAYOUTS = ("flat", "hills", "gap")


def draw_terrain(rng, size=SIZE):
    """(layout, heightmap (size, size) long, gap) of one scene: flat, low hills, or two plateaus
    with a trench between them; gap is (transposed, start, width, plateau height) or None."""
    d = h = w = size
    layout = LAYOUTS[rng.integers(len(LAYOUTS))]
    height = torch.zeros(h, w, dtype=torch.long)
    gap = None
    if layout == "hills":
        for _ in range(rng.integers(1, 4)):
            y0, x0 = rng.integers(0, h - 3), rng.integers(0, w - 3)
            sy, sx = rng.integers(3, max(4, h // 3)), rng.integers(3, max(4, w // 3))
            height[y0 : y0 + sy, x0 : x0 + sx] += int(rng.integers(1, 3))
    elif layout == "gap":
        top = int(rng.integers(3, max(4, d // 3)))
        width = int(rng.integers(3, max(4, w // 3)))
        left = int(rng.integers(2, w - width - 2))
        height[:, :left] = top
        height[:, left + width :] = top
        transposed = bool(rng.random() < 0.5)
        if transposed:
            height = height.t().contiguous()
        gap = (transposed, left, width, top)
    return layout, height, gap


def sample_scenes(rng, b, size=SIZE, gravity_prob=0.5, robots=None):
    """b random scenes on a size^3 grid. Terrain is a heightmap of walls: flat, low hills, or two
    plateaus with a trench between them (a gap to bridge). Robots rest on the terrain in random
    columns (stacked when a column is drawn twice), cheeses sit on level 2 x 2 footprints.
    robots: how many robots each scene gets (None: a density of 0.2 to 0.45 per floor cell).
    Returns (world, meta): meta[i] holds the layout name, the heightmap (size, size) and, for a
    gap, the trench: (transposed, start, width, plateau height)."""
    d = h = w = size
    mass = torch.zeros(b, d, h, w)
    walls = torch.zeros(b, d, h, w, dtype=torch.bool)
    pos = torch.zeros(b, MAX_CHEESES, 3, dtype=torch.long)
    alive = torch.zeros(b, MAX_CHEESES, dtype=torch.bool)
    colour = torch.full((b, MAX_CHEESES), -1, dtype=torch.long)
    gravity = torch.from_numpy(rng.random(b) < gravity_prob)
    meta = []
    for i in range(b):
        layout, height, gap = draw_terrain(rng, size)
        z = torch.arange(d).view(d, 1, 1)
        walls[i] = z < height.unsqueeze(0)
        n_cheese = int(rng.integers(0, MAX_CHEESES + 1))
        taken = walls[i].clone()
        colours = rng.permutation(len(PALETTE_NAMES))
        placed = 0
        for _ in range(40 * n_cheese):
            if placed == n_cheese:
                break
            y, x = int(rng.integers(0, h - 1)), int(rng.integers(0, w - 1))
            foot = height[y : y + 2, x : x + 2]
            base = int(foot[0, 0])
            if (foot != base).any() or base + 2 > d or taken[base : base + 2, max(y - 1, 0) : y + 3, max(x - 1, 0) : x + 3].any():
                continue
            pos[i, placed] = torch.tensor([base, y, x])
            alive[i, placed] = True
            colour[i, placed] = int(colours[placed])
            taken[base : base + 2, y : y + 2, x : x + 2] = True
            placed += 1
        n_robots = int(h * w * rng.uniform(0.2, 0.45)) if robots is None else int(robots)
        fill = height.clone()
        for y, x in zip(rng.integers(0, h, 3 * n_robots), rng.integers(0, w, 3 * n_robots)):
            if n_robots == 0:
                break
            z0 = int(fill[y, x])
            while z0 < d and taken[z0, y, x]:
                z0 += 1
            if z0 >= min(d, int(height[y, x]) + 3):
                continue
            mass[i, z0, y, x] = 1.0
            taken[z0, y, x] = True
            fill[y, x] = z0 + 1
            n_robots -= 1
        meta.append({"layout": layout, "height": height, "gap": gap})
    return VoxelWorld(mass, walls, pos, alive, colour, gravity), meta


# ---------------------------------------------------------------------------------------------
# Objects and floor zones (README "Objects")

ZONE_KINDS = ("patch", "stripe")


def object_form(shape, size, along="x", corner=(0, 0), arm=None):
    """(BOX, BOX, BOX) bool form of a shape (OBJECT_SHAPES) whose longest side is `size` (1 to
    BOX): a cube of side size; a slab size x size, one voxel high; a rod of length size, one voxel
    thick, lying along `along` (x or y); an L lying flat, one voxel high, with an arm of `size`
    along x and one of `arm` along y meeting at `corner` ((0 or 1, 0 or 1): the back or front row,
    the left or right column)."""
    form = torch.zeros(BOX, BOX, BOX, dtype=torch.bool)
    if shape == "cube":
        form[:size, :size, :size] = True
    elif shape == "slab":
        form[0, :size, :size] = True
    elif shape == "rod":
        if along == "x":
            form[0, 0, :size] = True
        else:
            form[0, :size, 0] = True
    elif shape == "L":
        arm = size if arm is None else arm
        row = 0 if corner[0] == 0 else arm - 1
        col = 0 if corner[1] == 0 else size - 1
        form[0, row, :size] = True
        form[0, :arm, col] = True
    else:
        raise ValueError(shape)
    return form


def form_extent(form):
    """(depth z, rows y, columns x) of a form, as ints."""
    zz, yy, xx = form.nonzero(as_tuple=True)
    return int(zz.max()) + 1, int(yy.max()) + 1, int(xx.max()) + 1


def draw_object(rng, colours):
    """One random object: {"shape", "size", "form", "colour", "colour2", "pattern", "heavy"} with
    colours drawn from the palette indices `colours` (food is drawn by the caller)."""
    shape = OBJECT_SHAPES[int(rng.choice(len(OBJECT_SHAPES), p=(0.4, 0.2, 0.2, 0.2)))]
    if shape == "cube":
        size = int(rng.integers(1, BOX + 1))
        form = object_form("cube", size)
    elif shape == "slab":
        size = int(rng.integers(2, BOX + 1))
        form = object_form("slab", size)
    elif shape == "rod":
        size = int(rng.integers(2, BOX + 1))
        form = object_form("rod", size, along=str(rng.choice(["x", "y"])))
    else:
        size, arm = int(rng.integers(2, BOX + 1)), int(rng.integers(2, BOX + 1))
        form = object_form("L", size, corner=(int(rng.integers(2)), int(rng.integers(2))), arm=arm)
        size = max(size, arm)
    longest = max(form_extent(form))
    pattern = PATTERNS[int(rng.choice(3, p=(0.6, 0.2, 0.2)))]
    if (pattern == "striped" and longest < 3) or (pattern == "two-tone" and longest < 2):
        pattern = "plain"
    colour = int(rng.choice(colours))
    colour2 = int(rng.choice([c for c in colours if c != colour])) if pattern != "plain" else -1
    return {"shape": shape, "size": size, "form": form, "colour": colour, "colour2": colour2, "pattern": pattern, "heavy": bool(rng.random() < 0.35)}


def place_zones(rng, height, count, colours, taken):
    """Up to `count` floor zones on level ground, apart from each other and from `taken` (H, W)
    bool, which they are added to: [{"colour", "kind" (ZONE_KINDS), "box": (y0, y1, x0, x1)}],
    each its own colour."""
    h, w = height.shape
    out = []
    free = [int(c) for c in rng.permutation(colours)]
    for _ in range(count):
        if not free:
            break
        for _ in range(30):
            kind = ZONE_KINDS[int(rng.integers(2))]
            if kind == "patch":
                sy, sx = int(rng.integers(3, 5)), int(rng.integers(3, 5))
            elif rng.random() < 0.5:
                sy, sx = 1, int(rng.integers(6, 11))
            else:
                sy, sx = int(rng.integers(6, 11)), 1
            y0, x0 = int(rng.integers(0, h - sy + 1)), int(rng.integers(0, w - sx + 1))
            cells = height[y0 : y0 + sy, x0 : x0 + sx]
            if (cells != cells[0, 0]).any() or taken[max(y0 - 1, 0) : y0 + sy + 1, max(x0 - 1, 0) : x0 + sx + 1].any():
                continue
            taken[y0 : y0 + sy, x0 : x0 + sx] = True
            out.append({"colour": free.pop(0), "kind": kind, "box": (y0, y0 + sy, x0, x0 + sx)})
            break
    return out


def sample_object_scenes(rng, b, size=SIZE, gravity_prob=0.5, robots=None, objects=(1, MAX_OBJECTS), food=0.5, zones=(0, 3), colours=None):
    """b random scenes with objects (README "Objects"): terrain as sample_scenes draws it, 0 to 3
    floor zones (patches 3 to 4 cells a side or stripes one cell wide and 6 to 10 long, level, each
    its own colour), then `objects` (low, high) objects (draw_object), each food with probability
    `food`, on level ground off the zones and a voxel apart, then the robots. colours: the palette
    indices objects and zones are item from (None: all). Returns (world with MAX_OBJECTS slots
    and every object field, meta): meta[i] holds layout, height and gap as sample_scenes gives them
    and "zones" (place_zones)."""
    d = h = w = size
    colours = list(range(len(COLOUR_NAMES))) if colours is None else list(colours)
    k = MAX_OBJECTS
    mass = torch.zeros(b, d, h, w)
    walls = torch.zeros(b, d, h, w, dtype=torch.bool)
    pos = torch.zeros(b, k, 3, dtype=torch.long)
    alive = torch.zeros(b, k, dtype=torch.bool)
    colour = torch.full((b, k), -1, dtype=torch.long)
    form = default_form(b, k)
    colour2 = torch.full((b, k), -1, dtype=torch.long)
    pattern = torch.zeros(b, k, dtype=torch.long)
    friction = torch.full((b, k), FRICTION)
    edible = torch.ones(b, k, dtype=torch.bool)
    zone_map = torch.full((b, h, w), -1, dtype=torch.long)
    gravity = torch.from_numpy(rng.random(b) < gravity_prob)
    meta = []
    for i in range(b):
        layout, height, gap = draw_terrain(rng, size)
        walls[i] = torch.arange(d).view(d, 1, 1) < height.unsqueeze(0)
        floor_taken = torch.zeros(h, w, dtype=torch.bool)
        marks = place_zones(rng, height, int(rng.integers(zones[0], zones[1] + 1)), colours, floor_taken)
        for mark in marks:
            y0, y1, x0, x1 = mark["box"]
            zone_map[i, y0:y1, x0:x1] = mark["colour"]
        occupied_voxels = walls[i].clone()
        near = torch.zeros(d, h, w, dtype=torch.bool)  # object voxels halo by one all round: no object within a voxel of another
        n_placed = 0
        for _ in range(int(rng.integers(objects[0], objects[1] + 1))):
            item = draw_object(rng, colours)
            depth, rows, cols = form_extent(item["form"])
            for _ in range(60):
                y, x = int(rng.integers(0, h - rows + 1)), int(rng.integers(0, w - cols + 1))
                foot = height[y : y + rows, x : x + cols]
                base = int(foot[0, 0])
                if (foot != base).any() or base + depth > d or floor_taken[y : y + rows, x : x + cols].any():
                    continue
                cells = torch.zeros(d + BOX, h + BOX, w + BOX, dtype=torch.bool)
                cells[base : base + BOX, y : y + BOX, x : x + BOX] = item["form"]
                cells = cells[:d, :h, :w]
                if (cells & (occupied_voxels | near)).any():
                    continue
                pos[i, n_placed] = torch.tensor([base, y, x])
                alive[i, n_placed] = True
                colour[i, n_placed] = item["colour"]
                form[i, n_placed] = item["form"]
                colour2[i, n_placed] = item["colour2"]
                pattern[i, n_placed] = PATTERNS.index(item["pattern"])
                friction[i, n_placed] = HEAVY_FRICTION if item["heavy"] else FRICTION
                edible[i, n_placed] = bool(rng.random() < food)
                occupied_voxels |= cells
                halo = cells.unsqueeze(0).clone()
                for axis in (1, 2, 3):
                    halo = halo | torch.roll(halo, 1, axis) | torch.roll(halo, -1, axis)
                near |= halo[0]
                n_placed += 1
                break
        n_robots = int(h * w * rng.uniform(0.2, 0.45)) if robots is None else int(robots)
        fill = height.clone()
        for y, x in zip(rng.integers(0, h, 3 * n_robots), rng.integers(0, w, 3 * n_robots)):
            if n_robots == 0:
                break
            z0 = int(fill[y, x])
            while z0 < d and occupied_voxels[z0, y, x]:
                z0 += 1
            if z0 >= min(d, int(height[y, x]) + 3):
                continue
            mass[i, z0, y, x] = 1.0
            occupied_voxels[z0, y, x] = True
            fill[y, x] = z0 + 1
            n_robots -= 1
        meta.append({"layout": layout, "height": height, "gap": gap, "zones": marks})
    world = VoxelWorld(mass, walls, pos, alive, colour, gravity, form, colour2, pattern, friction, edible, zone_map)
    return world, meta
