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
"""

from dataclasses import dataclass, replace

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
N_CHANNELS = 6  # robot, wall, cheese mask, cheese RGB


@dataclass
class VoxelWorld:
    mass: torch.Tensor  # (B, D, H, W) float, 1 where a robot is
    walls: torch.Tensor  # (B, D, H, W) bool
    cheese_pos: torch.Tensor  # (B, K, 3) long, lowest corner (z, y, x) of each cheese
    cheese_alive: torch.Tensor  # (B, K) bool
    cheese_color: torch.Tensor  # (B, K) long palette index, -1 for an empty slot
    gravity: torch.Tensor  # (B,) bool

    def index(self, i):
        """A sub-batch: an int, a slice or an index tensor."""
        if isinstance(i, int):
            i = slice(i, i + 1)
        return VoxelWorld(self.mass[i], self.walls[i], self.cheese_pos[i], self.cheese_alive[i], self.cheese_color[i], self.gravity[i])


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


def cheese_id_map(world, pos=None, alive=None):
    """(B, D, H, W) long: the index of the alive cheese at each voxel, -1 where none."""
    pos = world.cheese_pos if pos is None else pos
    alive = world.cheese_alive if alive is None else alive
    b, k = alive.shape
    dims = world.mass.shape[-3:]
    flat, inside = flat_index(cheese_voxels(pos), dims)
    keep = inside & alive.unsqueeze(2)
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
    not a wall, cheese alive."""
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


def rule_table(world, rule, sides=False):
    """pivot_moves of a scene batch under a named rule (RULES)."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}")
    return pivot_moves(world.mass > 0.5, world.walls, cheese_id_map(world) >= 0, release=rule == "switch_cube", sides=sides)


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


def move_robots(world, codes, ids, score=None):
    """Move robots one voxel by their action codes (B, D, H, W) long in 0..26, in rounds with
    random priority. Works on the list of robots, not the dense grid. Returns the new occupancy
    and the (B, K, 3) press on each cheese (the summed directions of robots moving into it)."""
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
    for z in range(1, occupied.shape[1]):
        drop = loose[:, z] & ~occupied[:, z - 1] & ~solid[:, z - 1]
        occupied[:, z - 1] |= drop
        occupied[:, z] &= ~drop
    pos = sub.cheese_pos.clone()
    for c in range(pos.shape[1]):
        others = sub.cheese_alive.clone()
        others[:, c] = False
        other = cheese_id_map(sub, pos, others) >= 0
        under = _face(pos[:, c], 0, torch.full_like(pos[:, c, 0], -1))
        clear = ~gather(sub.walls | other | occupied, under, True).any(dim=1)
        pos[:, c, 0] -= (sub.cheese_alive[:, c] & clear).long()
    mass = world.mass.clone()
    mass[on] = occupied.to(mass.dtype)
    cheese_pos = world.cheese_pos.clone()
    cheese_pos[on] = pos
    return replace(world, mass=mass, cheese_pos=cheese_pos)


def coverage(world):
    """(covered, open) shell voxel counts per cheese, (B, K) each."""
    coords, open_ = shell(world)
    covered = (gather(world.mass, coords, 0.0) > 0.5) & open_
    return covered.sum(dim=2), open_.sum(dim=2)


def eat_cheeses(world):
    covered, open_ = coverage(world)
    return world.cheese_alive & (covered >= EAT_SHARE * open_) & (covered >= EAT_MIN_SHARE * FULL_SHELL)


def step(world, codes, score=None, rule=None):
    """Advance every scene one step under action codes (B, D, H, W). Returns (world, StepInfo).
    rule: a pivot rule (RULES); moves it forbids do not happen (refuse). None: any move."""
    ids = cheese_id_map(world)
    if rule is not None:
        codes = refuse(world, codes, rule, ids)
    mass, force = move_robots(world, codes, ids, score)
    mass, pos, pushed = push_cheeses(world, mass, force)
    moved = fall(replace(world, mass=mass, cheese_pos=pos))
    eaten = eat_cheeses(moved)
    return replace(moved, cheese_alive=moved.cheese_alive & ~eaten), StepInfo(pushed=pushed, eaten=eaten)


def observe(world):
    """(B, 6, D, H, W) float: robot, wall, cheese mask, cheese RGB."""
    ids = cheese_id_map(world)
    present = ids >= 0
    colour = COLOUR_RGB.to(world.mass.device)[world.cheese_color.clamp(min=0)]  # (B, K, 3)
    rgb = colour.gather(1, ids.clamp(min=0).view(ids.shape[0], -1, 1).expand(-1, -1, 3)).view(*ids.shape, 3)
    rgb = (rgb * present.unsqueeze(-1)).permute(0, 4, 1, 2, 3)
    return torch.cat([world.mass.unsqueeze(1), world.walls.float().unsqueeze(1), present.float().unsqueeze(1), rgb], dim=1)


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
