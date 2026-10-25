"""Random layouts: grid size, terrain, cheese shape, starting nest, and the fields derived from them.

The grid is a torus: walking off one edge brings you back on the opposite edge.
Terrain is a set of wall squares that robots can't enter. Distances are walking
distances: the fewest moves to the cheese, where a move goes to any of the 8
neighbors and never enters a wall or the cheese. So all 8 squares touching a
cheese square are at distance 1, and a square behind a wall is as far as the
way around it.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# The swarm is SWARM_DENSITY units of mass per open square of the world, so a
# bigger world gets a proportionally bigger swarm to search it. It always has at
# least MASS_PER_RING_SQUARE per ring square, so every cheese can be surrounded.
SWARM_DENSITY = 0.06
MASS_PER_RING_SQUARE = 1.5

# Terrain is smoothed noise thresholded at the requested wall density.
# TERRAIN_SCALE is the smoothing width in squares: larger means bigger, rounder walls.
TERRAIN_SCALE = 2.0
# Walls can't cut the open space into pieces smaller than this share: the open
# area is the biggest connected region, and anything cut off becomes wall.
MIN_OPEN_SHARE = 0.5
UNREACHABLE = 10_000


@dataclass
class Batch:
    """A batch of layouts on one grid size. Every tensor is (B, H, W)."""

    mass: torch.Tensor  # starting mass, 1 per nest square
    cheese: torch.Tensor  # bool
    walls: torch.Tensor  # bool
    ring: torch.Tensor  # bool, the open squares touching the cheese
    distance: torch.Tensor  # walking distance to the cheese; 0 on cheese, UNREACHABLE on walls

    def to(self, device):
        return Batch(
            mass=self.mass.to(device),
            cheese=self.cheese.to(device),
            walls=self.walls.to(device),
            ring=self.ring.to(device),
            distance=self.distance.to(device),
        )


def _spread(front):
    """Every square within one move of a square in front, on the torus."""
    out = front.copy()
    for dy, dx in NEIGHBORS_8:
        out |= np.roll(front, (dy, dx), axis=(0, 1))
    return out


def walking_distance(sources, passable):
    """Fewest moves from any source square to each square, moving only through passable squares.

    Sources are at distance 0. Squares that can't be reached get UNREACHABLE.
    """
    distance = np.full(sources.shape, UNREACHABLE, dtype=np.int64)
    distance[sources] = 0
    reached = sources.copy()
    front = sources.copy()
    step = 0
    while front.any():
        step += 1
        front = _spread(front) & passable & ~reached
        distance[front] = step
        reached |= front
    return distance


def make_terrain(rng, size, wall_density):
    """Bool wall mask: smoothed noise on the torus, the top wall_density of it made wall."""
    if wall_density <= 0:
        return np.zeros((size, size), dtype=bool)
    noise = rng.normal(size=(size, size))
    k = np.fft.fftfreq(size)
    ky, kx = np.meshgrid(k, k, indexing="ij")
    blur = np.exp(-2 * (np.pi * TERRAIN_SCALE) ** 2 * (kx**2 + ky**2))
    smooth = np.real(np.fft.ifft2(np.fft.fft2(noise) * blur))
    return smooth > np.quantile(smooth, 1 - wall_density)


def grow_shape(rng, size, n_cells, seed, allowed):
    """Random connected shape of n_cells squares, grown from seed through allowed squares.

    Returns a bool mask, or None if the shape got boxed in before reaching n_cells.
    """
    cells = {seed}
    while len(cells) < n_cells:
        candidates = []
        for y, x in cells:
            for dy, dx in NEIGHBORS_4:
                neighbor = ((y + dy) % size, (x + dx) % size)
                if neighbor not in cells and allowed[neighbor]:
                    candidates.append(neighbor)
        if not candidates:
            return None
        cells.add(candidates[rng.integers(len(candidates))])

    mask = np.zeros((size, size), dtype=bool)
    for y, x in cells:
        mask[y, x] = True
    return mask


def swarm_mass(ring, open_squares):
    return max(math.ceil(MASS_PER_RING_SQUARE * ring.sum()), math.ceil(SWARM_DENSITY * open_squares))


def _random_square(rng, mask):
    candidates = np.argwhere(mask)
    return tuple(int(v) for v in candidates[rng.integers(len(candidates))])


def open_region(rng, size, wall_density):
    """Walls, with every open square outside the biggest connected open region turned into wall.

    Returns None if the walls split the world too badly.
    """
    walls = make_terrain(rng, size, wall_density)
    if walls.all():
        return None
    start = np.zeros_like(walls)
    start[_random_square(rng, ~walls)] = True
    region = walking_distance(start, ~walls) < UNREACHABLE
    if region.sum() < MIN_OPEN_SHARE * (~walls).sum():
        return None
    return ~region


def sample_layout(rng, size, cheese_range, distance_range, wall_range=(0.0, 0.0)):
    """One layout as numpy arrays: (cheese mask, wall mask, starting mass, distance field).

    cheese_range: inclusive range for the number of cheese squares.
    distance_range: inclusive range for the walking distance from the cheese to
        the square the nest grows from. The nest may reach closer, but never onto the ring.
    wall_range: range for the share of the world that is wall, sampled uniformly.
    """
    while True:
        walls = open_region(rng, size, rng.uniform(*wall_range))
        if walls is None:
            continue
        n_cheese = rng.integers(cheese_range[0], cheese_range[1] + 1)
        cheese = grow_shape(rng, size, n_cheese, _random_square(rng, ~walls), ~walls)
        if cheese is None:
            continue
        passable = ~walls & ~cheese
        distance = walking_distance(cheese, passable)
        ring = distance == 1
        if not ring.any():
            continue

        reachable = passable & (distance < UNREACHABLE)
        lowest = distance_range[0]
        highest = min(distance_range[1], int(distance[reachable].max()))
        if lowest > highest:
            continue
        target = rng.integers(lowest, highest + 1)
        if not (distance == target).any():
            continue
        nest_seed = _random_square(rng, distance == target)
        nest = grow_shape(rng, size, swarm_mass(ring, reachable.sum()), nest_seed, reachable & (distance >= 2))
        if nest is None:
            continue
        # Every ring square must be reachable from the nest without passing through the cheese.
        from_nest = walking_distance(nest, passable)
        if (from_nest[ring] >= UNREACHABLE).any():
            continue
        return cheese, walls, nest.astype(np.float32), distance


def sample_batch(rng, size, cheese_range, distance_range, batch_size, wall_range=(0.0, 0.0)):
    cheeses, walls, masses, distances = [], [], [], []
    for _ in range(batch_size):
        cheese, wall, mass, distance = sample_layout(rng, size, cheese_range, distance_range, wall_range)
        cheeses.append(cheese)
        walls.append(wall)
        masses.append(mass)
        distances.append(distance)

    distance = torch.tensor(np.stack(distances), dtype=torch.float32)
    return Batch(
        mass=torch.tensor(np.stack(masses)),
        cheese=torch.tensor(np.stack(cheeses)),
        walls=torch.tensor(np.stack(walls)),
        ring=distance == 1,
        distance=distance,
    )
