"""Random layouts: grid size, cheese shape, starting blob, and the fields derived from them.

The grid is a torus: walking off one edge brings you back on the opposite edge.
Distances are Chebyshev (max of |dy|, |dx|), so all 8 squares touching a cheese
square are at distance 1.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

NEIGHBORS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# The robot mass is this many times the number of ring squares, so every cheese
# can be surrounded with some mass to spare.
MASS_PER_RING_SQUARE = 1.5


@dataclass
class Batch:
    """A batch of layouts on one grid size. Every tensor is (B, H, W)."""

    mass: torch.Tensor  # starting mass, 1 per blob square
    smell: torch.Tensor  # exp(-distance / smell_length), 1 on the cheese
    cheese: torch.Tensor  # bool
    ring: torch.Tensor  # bool, the non-cheese squares touching the cheese
    distance: torch.Tensor  # distance to the nearest cheese square

    def to(self, device):
        return Batch(
            mass=self.mass.to(device),
            smell=self.smell.to(device),
            cheese=self.cheese.to(device),
            ring=self.ring.to(device),
            distance=self.distance.to(device),
        )


def torus_distance(cheese: np.ndarray) -> np.ndarray:
    """Chebyshev distance from every square to the nearest cheese square."""
    size = cheese.shape[0]
    cheese_y, cheese_x = np.nonzero(cheese)
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    dy = np.abs(grid_y[None] - cheese_y[:, None, None])
    dx = np.abs(grid_x[None] - cheese_x[:, None, None])
    dy = np.minimum(dy, size - dy)
    dx = np.minimum(dx, size - dx)
    return np.maximum(dy, dx).min(axis=0)


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


def sample_layout(rng, size, cheese_range, distance_range):
    """One layout as numpy arrays: (cheese mask, starting mass, distance field).

    cheese_range: inclusive range for the number of cheese squares.
    distance_range: inclusive range for the distance from the cheese to the
        square the blob grows from. The blob itself may reach closer, but never
        onto the ring.
    """
    while True:
        n_cheese = rng.integers(cheese_range[0], cheese_range[1] + 1)
        seed = (int(rng.integers(size)), int(rng.integers(size)))
        cheese = grow_shape(rng, size, n_cheese, seed, np.ones((size, size), dtype=bool))
        if cheese is None:
            continue
        distance = torus_distance(cheese)
        ring = distance == 1

        lowest = distance_range[0]
        highest = min(distance_range[1], int(distance.max()))
        if lowest > highest:
            continue
        target = rng.integers(lowest, highest + 1)
        candidates = np.argwhere(distance == target)
        blob_seed = tuple(int(v) for v in candidates[rng.integers(len(candidates))])

        n_mass = math.ceil(MASS_PER_RING_SQUARE * ring.sum())
        blob = grow_shape(rng, size, n_mass, blob_seed, distance >= 2)
        if blob is None:
            continue
        return cheese, blob.astype(np.float32), distance


def sample_batch(rng, size, cheese_range, distance_range, batch_size, smell_length):
    cheeses, masses, distances = [], [], []
    for _ in range(batch_size):
        cheese, mass, distance = sample_layout(rng, size, cheese_range, distance_range)
        cheeses.append(cheese)
        masses.append(mass)
        distances.append(distance)

    distance = torch.tensor(np.stack(distances), dtype=torch.float32)
    return Batch(
        mass=torch.tensor(np.stack(masses)),
        smell=torch.exp(-distance / smell_length),
        cheese=torch.tensor(np.stack(cheeses)),
        ring=distance == 1,
        distance=distance,
    )
