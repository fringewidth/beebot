"""Voxel-world physics for a module design (design.Capability), on top of vla/voxel/world.py.

Gravity is always on. What changes from the voxel world's rules:

1. Moves. A module that cannot fly (thrust below 1 weight) moves only along surfaces: its
   destination must touch a module, a wall, a cheese or the floor by a face (its own old voxel does
   not count), and it cannot move along all three axes at once. A move pivots the module about an
   edge, so it needs torque: 0.5 m g L on the level, 1.0 going up, plus, for permanent magnets,
   0.5 s m g L for every bond to a module it leaves (prying a face open about its edge). A module
   that flies can move into any free voxel; to leave bonds it must pull them off with spare
   thrust (r - 1 >= bonds * s) or pry them with torque. Electropermanent connectors switch off,
   so leaving bonds costs nothing. A move that the design cannot make leaves the module in place.
1. Pushing. Each module pressing on a cheese pushes with min(2 a, mu (1 + b)) weights, where b
   counts the modules lined up behind it (they brace it with their traction); a module that flies
   can also push with its tilted thrust, sqrt(r^2 - 1). The cheese slides when the press along
   its strongest axis reaches mu times its weight.
1. Structure. Each step the structure is solved for statics (hardware/structure.py): face
   contacts press anywhere on the face, module-to-module contacts also pull through a connector
   of strength s, contacts slide past friction, and a cheese resting on modules loads them with
   its weight. Modules the structure cannot hold fall one voxel, as in the voxel world.

Units: lengths in L, forces in module weights (m g), torques in m g L.
"""

from dataclasses import replace

import numpy as np
import torch

from beebot.hardware.structure import FACES, loose_modules
from beebot.vla.voxel.controller import robot_list
from beebot.vla.voxel.world import CHEESE_SIDE, DIR_TABLE, FRICTION, STAY, StepInfo, _face, cheese_id_map, eat_cheeses, flat_index, gather, move_robots, push_cheeses

MU = 0.5
BRACE_MAX = 4  # modules behind a pusher that can brace it
GRID_FACES = torch.tensor(FACES)
MOVES = DIR_TABLE[:STAY]  # (26, 3) grid order
KEPT = ((MOVES.unsqueeze(1) - GRID_FACES.unsqueeze(0)).abs().sum(-1) == 1)  # (26, 6): the bond on face f survives move d


# ---------------------------------------------------------------------------------------------
# Moves


def at(grid, scene, coords, fill):
    """Values of grid (B, D, H, W) at coords (N, ..., 3) of the scenes scene (N,); fill off-grid."""
    b, d, h, w = grid.shape
    flat, inside = flat_index(coords, (d, h, w))
    rows = scene.view(-1, *([1] * (coords.dim() - 2)))
    values = grid.reshape(b, -1)[rows, flat]
    return torch.where(inside, values, torch.as_tensor(fill, dtype=grid.dtype))


def allowed_moves(world, cap):
    """(N, 27) bool over the robots in controller.robot_list order: which actions the design can
    carry out from where each module is now. STAY is always allowed."""
    robots, scene, here = robot_list(world)
    n = robots.numel()
    allowed = torch.ones(n, STAY + 1, dtype=torch.bool)
    if n == 0:
        return allowed
    occupied = world.mass > 0.5
    solid = world.walls | (cheese_id_map(world) >= 0)
    near = here.unsqueeze(1) + GRID_FACES  # (N, 6, 3)
    bonded = at(occupied, scene, near, False)  # (N, 6)
    broken = (bonded.unsqueeze(1) & ~KEPT.unsqueeze(0)).sum(-1).float()  # (N, 26)
    dest = here.unsqueeze(1) + MOVES  # (N, 26, 3)
    touching = dest.unsqueeze(2) + GRID_FACES.view(1, 1, 6, 3)  # (N, 26, 6, 3)
    own = (GRID_FACES.view(1, 6, 3) == -MOVES.view(26, 1, 3)).all(-1)  # (26, 6): the face that is the old voxel
    holds = at(occupied, scene, touching, False) | at(solid, scene, touching, False) | (touching[..., 0] < 0)
    substrate = (holds & ~own.unsqueeze(0)).any(-1)  # (N, 26)
    up = (MOVES[:, 0] > 0).float()
    pry = 0.0 if cap.switchable else 0.5 * cap.strength
    pivot = cap.torque >= 0.5 * (1 + up) + pry * broken
    if cap.flies:
        pull = cap.switchable | (broken * cap.strength <= cap.thrust - 1)
        allowed[:, :STAY] = pull | (cap.torque >= 0.5 + pry * broken)
    else:
        corner = MOVES.abs().sum(-1) == 3
        allowed[:, :STAY] = substrate & pivot & ~corner
    return allowed


def push_per_module(world, scene, here, direction, cap):
    """Push force (weights) of each pressing module: traction braced by the modules behind it,
    capped by its actuator; flying designs may push with tilted thrust instead."""
    occupied = world.mass > 0.5
    behind = torch.zeros(len(scene))
    line = torch.ones(len(scene), dtype=torch.bool)
    for k in range(1, BRACE_MAX + 1):
        line &= at(occupied, scene, here - k * direction, False)
        behind += line.float()
    force = torch.clamp(MU * (1 + behind), max=2 * cap.torque)
    if cap.flies:
        force = torch.clamp(force, min=float(np.sqrt(cap.thrust**2 - 1)))
    return force


def press(world, codes, ids, cap):
    """(B, K, 3) press on each cheese, scaled so the voxel world's FRICTION threshold means mu
    times the cheese's weight."""
    b, d, h, w = world.mass.shape
    k = world.cheese_alive.shape[1]
    robots, scene, here = robot_list(world)
    code = codes.reshape(-1)[robots].long()
    direction = DIR_TABLE[code]
    flat, inside = flat_index(here + direction, (d, h, w))
    target = torch.where(inside, ids.reshape(b, -1)[scene, flat], -1)
    pressing = (target >= 0) & (code != STAY)
    force = torch.zeros(b * k, 3)
    if pressing.any():
        each = push_per_module(world, scene[pressing], here[pressing], direction[pressing], cap)
        force.index_add_(0, scene[pressing] * k + target[pressing], direction[pressing].float() * each.unsqueeze(1))
    return (force * FRICTION / (MU * cap.cheese)).view(b, k, 3)


# ---------------------------------------------------------------------------------------------
# Structure and gravity


def cheese_loads(world):
    """(B, D, H, W) float: each alive cheese's weight shared by the modules directly under it (in
    cheese weights; multiplied by the cheese weight in settle)."""
    b, d, h, w = world.mass.shape
    extra = torch.zeros(b, d, h, w)
    for k in range(world.cheese_alive.shape[1]):
        under = _face(world.cheese_pos[:, k], 0, torch.full((b,), -1, dtype=torch.long))  # (B, 4, 3)
        flat, inside = flat_index(under, (d, h, w))
        live = inside & world.cheese_alive[:, k].unsqueeze(1)
        extra.view(b, -1).scatter_add_(1, flat, live.float() / (CHEESE_SIDE * CHEESE_SIDE))
    return extra * (world.mass > 0.5)


def settle(world, cap, memory=None):
    """Structural failures and gravity: modules the structure cannot hold drop one voxel, and
    cheeses with nothing under them drop one voxel (as world.fall). memory: one dict per scene,
    carried between steps (structure.loose_modules)."""
    occupied = world.mass > 0.5
    solid = world.walls | (cheese_id_map(world) >= 0)
    extra = cheese_loads(world) * cap.cheese
    loose = torch.zeros_like(occupied)
    for i in range(occupied.shape[0]):
        loose[i] = torch.from_numpy(loose_modules(occupied[i].numpy(), solid[i].numpy(), extra[i].numpy(), cap.strength, cap.flies, cap.thrust, None if memory is None else memory[i]))
    occupied = occupied.clone()
    for z in range(1, occupied.shape[1]):
        drop = loose[:, z] & ~occupied[:, z - 1] & ~solid[:, z - 1]
        occupied[:, z - 1] |= drop
        occupied[:, z] &= ~drop
    pos = world.cheese_pos.clone()
    for c in range(pos.shape[1]):
        others = world.cheese_alive.clone()
        others[:, c] = False
        other = cheese_id_map(world, pos, others) >= 0
        under = _face(pos[:, c], 0, torch.full_like(pos[:, c, 0], -1))
        clear = ~gather(world.walls | other | occupied, under, True).any(dim=1)
        pos[:, c, 0] -= (world.cheese_alive[:, c] & clear).long()
    return replace(world, mass=occupied.to(world.mass.dtype), cheese_pos=pos)


def step(world, codes, cap, score=None, memory=None):
    """One step under the design's physics: the voxel world's step with feasible moves only,
    design-dependent pushing, and structural settling in place of the support rule."""
    b, d, h, w = world.mass.shape
    robots, scene, here = robot_list(world)
    codes = codes.clone()
    if robots.numel():
        allowed = allowed_moves(world, cap)
        code = codes.reshape(-1)[robots].long()
        ids = cheese_id_map(world)
        flat, inside = flat_index(here + DIR_TABLE[code], (d, h, w))
        into_cheese = inside & (ids.reshape(b, -1)[scene, flat] >= 0)
        ok = allowed[torch.arange(len(code)), code] | into_cheese
        codes.view(-1)[robots[~ok]] = STAY
    ids = cheese_id_map(world)
    force = press(world, codes, ids, cap)
    mass, _ = move_robots(world, codes, ids, score)
    mass, pos, pushed = push_cheeses(world, mass, force)
    moved = settle(replace(world, mass=mass, cheese_pos=pos), cap, memory)
    eaten = eat_cheeses(moved)
    return replace(moved, cheese_alive=moved.cheese_alive & ~eaten), StepInfo(pushed=pushed, eaten=eaten)
