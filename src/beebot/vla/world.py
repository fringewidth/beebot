"""The world for language-conditioned control: a bounded grid with a swarm, terrain, cheeses and floor boxes.

Everything is batched: a World holds B scenes as tensors, and step() advances all of them at once.

The swarm is discrete robots, at most one per square (mass is 1 where a robot is, 0 elsewhere).
The action is a velocity field, one 2D vector per square: each robot moves one square in the
nearest of the 8 directions to its square's vector, or stays if the vector is shorter than 0.5.
A robot moving into a wall or off the grid stays. When several robots want one free square, one
wins at random; a robot can move into a square another robot leaves in the same step, so lines
advance together (see move_robots). A robot moving into a cheese stays and pushes on it.

Cheeses are rigid square blocks. Each step the push on a cheese (the directions of the robots
pressing on it, summed) is compared along its dominant axis with FRICTION; if it reaches it, the
cheese moves one square. A robot on a square the cheese moves into is shoved one square further,
which needs a free square there (no wall, cheese or robot); otherwise the cheese stays. A cheese
is eaten (removed) when at least EAT_MIN_SQUARES of its ring squares, and at least EAT_SHARE of
its open ring squares, hold a robot. A cheese pushed into a corner has too few open ring squares
to be eaten.

Boxes are colored floor rectangles; mass and cheeses pass over them freely.

Resolution scales: on a grid of size G, cheeses are G/8 squares on a side, and the number of robots,
terrain smoothing, box sizes and the push and eat thresholds scale with it (see the functions
below), so a 32-grid scene is a finer-grained version of a 16-grid one. The constants are the
values at G = 16.

Coordinates are (row, column); row 0 is the top. A field's channel 0 is the column (x, rightward)
component and channel 1 the row (y, downward) component.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn.functional as F

SIZE = 16
CHEESE_SIDE = 2  # at SIZE; see cheese_side()
MAX_CHEESES = 3
MAX_BOXES = 2

PALETTE_NAMES = ("red", "green", "blue", "yellow")
PALETTE_RGB = torch.tensor(
    [
        [0.90, 0.24, 0.24],
        [0.24, 0.78, 0.35],
        [0.27, 0.43, 0.94],
        [0.94, 0.82, 0.20],
    ]
)

# Colours beyond the four of the play data and its captions. The generalist tasks
# (beebot.vla.tasks) draw scenes from all of them except the held-out ones, which only their tests
# use: a colour word the policy never saw with its colour. Indices 0-3 are PALETTE_NAMES.
EXTRA_COLOURS = {
    "orange": [0.97, 0.56, 0.14],
    "purple": [0.58, 0.31, 0.82],
    "white": [0.96, 0.96, 0.96],
    "cyan": [0.18, 0.82, 0.86],
    "pink": [0.98, 0.56, 0.78],
}
COLOUR_NAMES = PALETTE_NAMES + tuple(EXTRA_COLOURS)
COLOUR_RGB = torch.cat([PALETTE_RGB, torch.tensor(list(EXTRA_COLOURS.values()))])

PHYSICS = "robots"  # recorded with saved play and scripted caches; earlier data used "continuous" mass
CAPACITY = 1.0  # one robot per square
STAY = 8  # robot_directions() code for not moving
FRICTION = 1.5  # at SIZE, in robots' worth of push; scales with the cheese's side (the pressing face)
EAT_MASS = 0.3
EAT_MIN_SQUARES = 6  # at SIZE: half the full ring
EAT_SHARE = 0.75

# The 8 move directions as (dy, dx), ordered by angle from +x toward +y (rows grow downward),
# so direction k sits at angle k * 45 degrees.
DIRS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))

N_CHANNELS = 10  # mass, wall, cheese mask, cheese RGB, box mask, box RGB

# Scene sampling.
MASS_RANGE = (16, 28)
CLUTTERED_SHARE = 0.5  # share of scenes with at least 2 cheeses and 2 boxes
NO_TERRAIN_SHARE = 0.25
WALL_DENSITY_RANGE = (0.05, 0.25)
TERRAIN_SCALE = 1.6
BOX_SIDE_RANGE = (3, 5)
CHEESE_GAP = 3  # empty squares between two cheeses' footprints, so their rings never touch
MIN_OPEN_RING = 8  # two thirds of the full ring


def cheese_side(size):
    return max(2, size // 8)


def friction(size):
    return FRICTION * cheese_side(size) / CHEESE_SIDE


def eat_min_squares(size):
    return 2 * cheese_side(size) + 2


def zoom(size):
    """How much finer than the 16-grid this grid is."""
    return size / SIZE


@dataclass
class World:
    """B scenes. Static: walls, boxes. Dynamic: mass, cheese positions and whether they're alive."""

    mass: torch.Tensor  # (B, H, W) float
    walls: torch.Tensor  # (B, H, W) bool
    cheese_pos: torch.Tensor  # (B, K, 2) long, top-left (row, col) of each 2x2 cheese
    cheese_alive: torch.Tensor  # (B, K) bool; slots that never held a cheese are False
    cheese_color: torch.Tensor  # (B, K) long palette index, -1 for an empty slot
    box_rect: torch.Tensor  # (B, MAX_BOXES, 4) long: row, col, height, width
    box_color: torch.Tensor  # (B, MAX_BOXES) long palette index, -1 for no box

    def to(self, device):
        return World(
            mass=self.mass.to(device),
            walls=self.walls.to(device),
            cheese_pos=self.cheese_pos.to(device),
            cheese_alive=self.cheese_alive.to(device),
            cheese_color=self.cheese_color.to(device),
            box_rect=self.box_rect.to(device),
            box_color=self.box_color.to(device),
        )

    def index(self, i):
        """The sub-batch at index i (an int, slice, or index tensor), keeping the batch dimension."""
        if isinstance(i, int):
            i = slice(i, i + 1)
        return World(
            mass=self.mass[i],
            walls=self.walls[i],
            cheese_pos=self.cheese_pos[i],
            cheese_alive=self.cheese_alive[i],
            cheese_color=self.cheese_color[i],
            box_rect=self.box_rect[i],
            box_color=self.box_color[i],
        )


@dataclass
class StepInfo:
    """What happened during one step, per scene and cheese."""

    pushed: torch.Tensor  # (B, K) bool, the cheese moved this step
    eaten: torch.Tensor  # (B, K) bool, the cheese was eaten this step


def cat_worlds(worlds):
    return World(
        mass=torch.cat([w.mass for w in worlds]),
        walls=torch.cat([w.walls for w in worlds]),
        cheese_pos=torch.cat([w.cheese_pos for w in worlds]),
        cheese_alive=torch.cat([w.cheese_alive for w in worlds]),
        cheese_color=torch.cat([w.cheese_color for w in worlds]),
        box_rect=torch.cat([w.box_rect for w in worlds]),
        box_color=torch.cat([w.box_color for w in worlds]),
    )


# ---------------------------------------------------------------------------------------------
# Grid helpers


def shift(field, dy, dx, fill):
    """out[..., y, x] = field[..., y - dy, x - dx]: content moves by (dy, dx); vacated squares get fill."""
    h, w = field.shape[-2:]
    out = torch.full_like(field, fill)
    ys_dst = slice(max(dy, 0), h + min(dy, 0))
    xs_dst = slice(max(dx, 0), w + min(dx, 0))
    ys_src = slice(max(-dy, 0), h + min(-dy, 0))
    xs_src = slice(max(-dx, 0), w + min(-dx, 0))
    out[..., ys_dst, xs_dst] = field[..., ys_src, xs_src]
    return out


def look(field, dy, dx, fill):
    """out[..., y, x] = field[..., y + dy, x + dx], with fill off the grid."""
    return shift(field, -dy, -dx, fill)


def cheese_masks(world):
    """(B, K, H, W) bool footprint of each alive cheese."""
    b, k = world.cheese_alive.shape
    h, w = world.mass.shape[-2:]
    rows = torch.arange(h, device=world.mass.device).view(1, 1, h, 1)
    cols = torch.arange(w, device=world.mass.device).view(1, 1, 1, w)
    y = world.cheese_pos[..., 0].view(b, k, 1, 1)
    x = world.cheese_pos[..., 1].view(b, k, 1, 1)
    side = cheese_side(h)
    inside = (rows >= y) & (rows < y + side) & (cols >= x) & (cols < x + side)
    return inside & world.cheese_alive.view(b, k, 1, 1)


def cheese_id_map(masks):
    """(B, H, W) long: index of the cheese on each square, -1 where there is none."""
    k = masks.shape[1]
    ids = torch.arange(k, device=masks.device).view(1, k, 1, 1)
    return torch.where(masks, ids, -1).amax(dim=1)


def box_masks(world):
    """(B, MAX_BOXES, H, W) bool."""
    b = world.box_rect.shape[0]
    h, w = world.mass.shape[-2:]
    rows = torch.arange(h, device=world.mass.device).view(1, 1, h, 1)
    cols = torch.arange(w, device=world.mass.device).view(1, 1, 1, w)
    r = world.box_rect
    y, x, bh, bw = (r[..., i].view(b, -1, 1, 1) for i in range(4))
    inside = (rows >= y) & (rows < y + bh) & (cols >= x) & (cols < x + bw)
    return inside & (world.box_color >= 0).view(b, -1, 1, 1)


def ring_masks(world, masks=None):
    """(B, K, H, W) bool: open squares touching each alive cheese (8-neighborhood), not on any cheese."""
    if masks is None:
        masks = cheese_masks(world)
    b, k, h, w = masks.shape
    grown = F.max_pool2d(masks.float().view(b * k, 1, h, w), 3, stride=1, padding=1).view(b, k, h, w) > 0
    any_cheese = masks.any(dim=1, keepdim=True)
    return grown & ~any_cheese & ~world.walls.unsqueeze(1)


# ---------------------------------------------------------------------------------------------
# Physics


def robot_directions(field):
    """(B, 2, H, W) field -> (B, H, W) long: the index into DIRS nearest each square's vector's
    angle, or STAY where the vector is shorter than 0.5."""
    vx, vy = field[:, 0], field[:, 1]
    k = torch.remainder(torch.round(torch.atan2(vy, vx) / (math.pi / 4)), 8).long()
    return torch.where(vx * vx + vy * vy < 0.25, STAY, k)


def move_robots(world, field, masks, score=None):
    """Move robots one square along the field. Returns the new occupancy (float, 0 or 1) and the
    (B, K, 2) push force (row, col) on each cheese: the sum of the directions of the robots trying
    to move into it.

    Moves are resolved in rounds. In each round every robot still waiting to move claims its
    target square; among claimants of a square that is free (no robot, wall or cheese) one wins at
    random and moves, and the square it leaves is free for the next round. So a line of robots
    moving the same way advances together; swaps and cycles can't resolve and those robots stay.
    score (B, H, W) sets the priorities (default: fresh uniform noise), so a test can use the same
    ones on a rotated world.
    """
    occupied = world.mass > 0.5
    ids = cheese_id_map(masks)
    on_cheese = ids >= 0
    direction = robot_directions(field)
    if score is None:
        score = torch.rand(occupied.shape, device=occupied.device)  # random priority, fresh each step

    k = masks.shape[1]
    force = torch.zeros(occupied.shape[0], k, 2, dtype=world.mass.dtype, device=occupied.device)
    pending = torch.zeros_like(occupied)
    for d, (dy, dx) in enumerate(DIRS):
        going = occupied & (direction == d)
        dest_id = look(ids, dy, dx, -1)
        for c in range(k):
            pressing = (going & (dest_id == c)).sum(dim=(-2, -1)).to(force.dtype)
            force[:, c, 0] += pressing * dy
            force[:, c, 1] += pressing * dx
        pending |= going & ~look(world.walls | on_cheese, dy, dx, True)

    blocked = world.walls | on_cheese
    for _ in range(2 * occupied.shape[-1] + 2):
        free = ~occupied & ~blocked
        best = torch.full(occupied.shape, -1.0, device=occupied.device)
        best_dir = torch.full(occupied.shape, -1, dtype=torch.long, device=occupied.device)
        for d, (dy, dx) in enumerate(DIRS):
            claim = shift(torch.where(pending & (direction == d), score, -1.0), dy, dx, -1.0)
            better = claim > best
            best = torch.where(better, claim, best)
            best_dir = torch.where(better, d, best_dir)
        arrive = free & (best >= 0)
        if not arrive.any():
            break
        leave = torch.zeros_like(occupied)
        for d, (dy, dx) in enumerate(DIRS):
            leave |= look(arrive & (best_dir == d), dy, dx, False)
        occupied = (occupied & ~leave) | arrive
        pending &= ~leave
    return occupied.to(world.mass.dtype), force


def _gather(grid, rows, cols, fill):
    """grid (B, H, W); rows, cols (B, N) -> (B, N), fill where off the grid."""
    h, w = grid.shape[-2:]
    inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    flat = (rows.clamp(0, h - 1) * w + cols.clamp(0, w - 1)).long()
    values = grid.reshape(grid.shape[0], -1).gather(1, flat)
    return torch.where(inside, values, torch.as_tensor(fill, dtype=grid.dtype, device=grid.device))


def push_cheeses(world, mass, force):
    """Move cheeses whose push reaches FRICTION. Returns (new mass, new cheese positions, pushed (B, K))."""
    b, k = world.cheese_alive.shape
    device = mass.device
    pos = world.cheese_pos.clone()
    mass = mass.clone().contiguous()
    pushed = torch.zeros(b, k, dtype=torch.bool, device=device)
    batch = torch.arange(b, device=device)
    n_side = cheese_side(mass.shape[-1])
    side = torch.arange(n_side, device=device).view(1, n_side)
    for c in range(k):
        f = force[:, c]
        along_cols = f[:, 1].abs() > f[:, 0].abs()
        strength = torch.where(along_cols, f[:, 1], f[:, 0])
        want = world.cheese_alive[:, c] & (strength.abs() >= friction(mass.shape[-1]))
        sign = torch.sign(strength).long()
        lead1 = torch.where(sign > 0, n_side, -1)
        lead2 = torch.where(sign > 0, n_side + 1, -2)
        y = pos[:, c, 0].view(b, 1)
        x = pos[:, c, 1].view(b, 1)
        cols_mode = along_cols.view(b, 1)
        rows1 = torch.where(cols_mode, y + side, y + lead1.view(b, 1))
        cols1 = torch.where(cols_mode, x + lead1.view(b, 1), x + side)
        rows2 = torch.where(cols_mode, y + side, y + lead2.view(b, 1))
        cols2 = torch.where(cols_mode, x + lead2.view(b, 1), x + side)

        others = world.cheese_alive.clone()
        others[:, c] = False
        other_ids = cheese_id_map(cheese_masks(replace(world, cheese_pos=pos, cheese_alive=others)))
        blocked1 = _gather(world.walls, rows1, cols1, True) | (_gather(other_ids, rows1, cols1, -1) >= 0)
        blocked2 = _gather(world.walls, rows2, cols2, True) | (_gather(other_ids, rows2, cols2, -1) >= 0)
        mass1 = _gather(mass, rows1, cols1, 0.0)
        mass2 = _gather(mass, rows2, cols2, 0.0)
        # A robot in the cheese's way is shoved one square further, into a free square only.
        shove_ok = (mass1 <= 0.5) | (~blocked2 & (mass2 <= 0.5))
        ok = want & ~blocked1.any(dim=1) & shove_ok.all(dim=1)

        amount = torch.where(ok.view(b, 1), mass1, torch.zeros_like(mass1))
        h, w = mass.shape[-2:]
        flat = mass.view(b, -1)
        idx1 = (rows1.clamp(0, h - 1) * w + cols1.clamp(0, w - 1)).long()
        idx2 = (rows2.clamp(0, h - 1) * w + cols2.clamp(0, w - 1)).long()
        flat.scatter_add_(1, idx1, -amount)
        flat.scatter_add_(1, idx2, amount)
        mass = flat.view(b, h, w)

        step = torch.zeros(b, 2, dtype=torch.long, device=device)
        step[batch, along_cols.long()] = sign
        pos[:, c] = torch.where(ok.view(b, 1), pos[:, c] + step, pos[:, c])
        pushed[:, c] = ok
    return mass, pos, pushed


def eat_cheeses(world):
    """(B, K) bool: alive cheeses whose ring is covered enough to be eaten now."""
    masks = cheese_masks(world)
    rings = ring_masks(world, masks)
    open_ring = rings.sum(dim=(-2, -1))
    covered = (rings & (world.mass.unsqueeze(1) >= EAT_MASS)).sum(dim=(-2, -1))
    enough = (covered >= eat_min_squares(world.mass.shape[-1])) & (covered >= EAT_SHARE * open_ring)
    return world.cheese_alive & enough


def step(world, field, score=None):
    """Advance every scene one step under field (B, 2, H, W). Returns (new world, StepInfo).
    score: the robots' priorities for this step (see move_robots); random by default."""
    field = torch.where(world.walls.unsqueeze(1), torch.zeros_like(field), field)
    masks = cheese_masks(world)
    mass, force = move_robots(world, field, masks, score)
    mass, pos, pushed = push_cheeses(world, mass, force)
    moved = replace(world, mass=mass, cheese_pos=pos)
    eaten = eat_cheeses(moved)
    after = replace(moved, cheese_alive=moved.cheese_alive & ~eaten)
    return after, StepInfo(pushed=pushed, eaten=eaten)


# ---------------------------------------------------------------------------------------------
# Observation


def observe(world):
    """(B, 10, H, W) float: mass, wall, cheese mask, cheese RGB, box mask, box RGB."""
    palette = COLOUR_RGB.to(world.mass.device)
    cmasks = cheese_masks(world).float()
    ccolor = palette[world.cheese_color.clamp(min=0)]  # (B, K, 3)
    cheese_rgb = torch.einsum("bkhw,bkc->bchw", cmasks, ccolor)
    bmasks = box_masks(world).float()
    bcolor = palette[world.box_color.clamp(min=0)]
    box_rgb = torch.einsum("bkhw,bkc->bchw", bmasks, bcolor).clamp(max=1.0)
    return torch.cat(
        [
            world.mass.unsqueeze(1),
            world.walls.float().unsqueeze(1),
            cmasks.amax(dim=1, keepdim=True),
            cheese_rgb,
            bmasks.amax(dim=1, keepdim=True),
            box_rgb,
        ],
        dim=1,
    )


# ---------------------------------------------------------------------------------------------
# Scene sampling (numpy, one scene at a time, then stacked)


def make_terrain(rng, size, density):
    """Bool wall mask on a bounded grid: smoothed noise, the top `density` of it made wall."""
    if density <= 0:
        return np.zeros((size, size), dtype=bool)
    pad = round(8 * zoom(size))
    big = size + 2 * pad
    noise = rng.normal(size=(big, big))
    k = np.fft.fftfreq(big)
    ky, kx = np.meshgrid(k, k, indexing="ij")
    scale = TERRAIN_SCALE * zoom(size)
    blur = np.exp(-2 * (np.pi * scale) ** 2 * (kx**2 + ky**2))
    smooth = np.real(np.fft.ifft2(np.fft.fft2(noise) * blur))[pad : pad + size, pad : pad + size]
    return smooth > np.quantile(smooth, 1 - density)


def largest_open_region(walls):
    """Bool mask of the largest 8-connected region of open squares."""
    size = walls.shape[0]
    label = np.full(walls.shape, -1)
    best, best_size = None, 0
    for start in zip(*np.nonzero(~walls)):
        if label[start] >= 0:
            continue
        region = []
        stack = [start]
        label[start] = len(region)
        while stack:
            y, x = stack.pop()
            region.append((y, x))
            for dy, dx in DIRS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < size and 0 <= nx < size and not walls[ny, nx] and label[ny, nx] < 0:
                    label[ny, nx] = 0
                    stack.append((ny, nx))
        if len(region) > best_size:
            best, best_size = region, len(region)
    mask = np.zeros(walls.shape, dtype=bool)
    if best:
        ys, xs = zip(*best)
        mask[list(ys), list(xs)] = True
    return mask


def _place_cheeses(rng, open_mask, n, size):
    """Top-left corners of n 2x2 cheeses on open squares, apart from each other, each with an eatable ring."""
    placed = []
    side = cheese_side(size)
    gap = round(CHEESE_GAP * side / CHEESE_SIDE)
    min_open_ring = (2 * (4 * side + 4)) // 3
    for _ in range(200):
        if len(placed) == n:
            break
        y, x = rng.integers(0, size - side + 1, size=2)
        if not open_mask[y : y + side, x : x + side].all():
            continue
        if any(abs(y - py) < side + gap and abs(x - px) < side + gap for py, px in placed):
            continue
        y0, y1 = max(y - 1, 0), min(y + side + 1, size)
        x0, x1 = max(x - 1, 0), min(x + side + 1, size)
        if open_mask[y0:y1, x0:x1].sum() - side * side < min_open_ring:
            continue
        placed.append((int(y), int(x)))
    return placed


def _place_boxes(rng, open_mask, n, size, cheese_cells):
    boxes = []
    taken = np.zeros_like(open_mask)
    for _ in range(200):
        if len(boxes) == n:
            break
        low, high = (round(v * zoom(size)) for v in BOX_SIDE_RANGE)
        h, w = rng.integers(low, high + 1, size=2)
        y = rng.integers(0, size - h + 1)
        x = rng.integers(0, size - w + 1)
        area = np.zeros_like(open_mask)
        area[y : y + h, x : x + w] = True
        if open_mask[area].mean() < 0.8 or (area & taken).any() or (area & cheese_cells).any():
            continue
        taken |= area
        boxes.append((int(y), int(x), int(h), int(w)))
    return boxes


def _grow_blob(rng, allowed, n_cells, size):
    seed = tuple(rng.choice(np.argwhere(allowed)))
    cells = {seed}
    frontier = [seed]
    while len(cells) < n_cells and frontier:
        y, x = frontier[rng.integers(len(frontier))]
        options = [
            (y + dy, x + dx)
            for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0))
            if 0 <= y + dy < size and 0 <= x + dx < size and allowed[y + dy, x + dx] and (y + dy, x + dx) not in cells
        ]
        if not options:
            frontier.remove((y, x))
            continue
        cell = options[rng.integers(len(options))]
        cells.add(cell)
        frontier.append(cell)
    return cells


def sample_scene(rng, size=SIZE):
    """One scene as numpy arrays, or None if placement failed (the caller retries)."""
    cluttered = rng.random() < CLUTTERED_SHARE
    n_cheeses = int(rng.integers(2, MAX_CHEESES + 1)) if cluttered else int(rng.integers(1, MAX_CHEESES + 1))
    n_boxes = MAX_BOXES if cluttered else int(rng.integers(0, MAX_BOXES + 1))
    density = 0.0 if rng.random() < NO_TERRAIN_SHARE else rng.uniform(*WALL_DENSITY_RANGE)
    walls = make_terrain(rng, size, density)
    open_mask = largest_open_region(walls)
    walls = ~open_mask
    if open_mask.mean() < 0.5:
        return None

    cheeses = _place_cheeses(rng, open_mask, n_cheeses, size)
    if len(cheeses) < n_cheeses:
        return None
    side = cheese_side(size)
    cheese_cells = np.zeros_like(open_mask)
    for y, x in cheeses:
        cheese_cells[y : y + side, x : x + side] = True
    near_cheese = cheese_cells.copy()
    for y, x in cheeses:
        near_cheese[max(y - 1, 0) : y + side + 1, max(x - 1, 0) : x + side + 1] = True
    boxes = _place_boxes(rng, open_mask, n_boxes, size, near_cheese)
    if len(boxes) < n_boxes:
        return None

    area = zoom(size) ** 2
    total = int(rng.integers(round(MASS_RANGE[0] * area), round(MASS_RANGE[1] * area) + 1))
    n_blobs = 1 if rng.random() < 0.7 else 2
    allowed = open_mask & ~near_cheese
    mass = np.zeros((size, size), dtype=np.float32)  # one robot per occupied square
    for share in np.array_split(np.arange(total), n_blobs):
        cells = _grow_blob(rng, allowed & (mass == 0), len(share), size)
        for cell in cells:
            mass[cell] = 1.0
    if mass.sum() <= 0:
        return None

    cheese_colors = rng.choice(len(PALETTE_NAMES), size=n_cheeses, replace=False)
    box_colors = rng.choice(len(PALETTE_NAMES), size=n_boxes, replace=False)
    return {
        "mass": mass,
        "walls": walls,
        "cheeses": cheeses,
        "cheese_colors": [int(c) for c in cheese_colors],
        "boxes": boxes,
        "box_colors": [int(c) for c in box_colors],
    }


def scenes_to_world(scenes):
    b = len(scenes)
    size = scenes[0]["mass"].shape[0]
    cheese_pos = torch.zeros(b, MAX_CHEESES, 2, dtype=torch.long)
    cheese_alive = torch.zeros(b, MAX_CHEESES, dtype=torch.bool)
    cheese_color = torch.full((b, MAX_CHEESES), -1, dtype=torch.long)
    box_rect = torch.zeros(b, MAX_BOXES, 4, dtype=torch.long)
    box_color = torch.full((b, MAX_BOXES), -1, dtype=torch.long)
    for i, s in enumerate(scenes):
        for k, ((y, x), c) in enumerate(zip(s["cheeses"], s["cheese_colors"])):
            cheese_pos[i, k] = torch.tensor([y, x])
            cheese_alive[i, k] = True
            cheese_color[i, k] = c
        for k, (rect, c) in enumerate(zip(s["boxes"], s["box_colors"])):
            box_rect[i, k] = torch.tensor(rect)
            box_color[i, k] = c
    return World(
        mass=torch.from_numpy(np.stack([s["mass"] for s in scenes])).float(),
        walls=torch.from_numpy(np.stack([s["walls"] for s in scenes])),
        cheese_pos=cheese_pos,
        cheese_alive=cheese_alive,
        cheese_color=cheese_color,
        box_rect=box_rect,
        box_color=box_color,
    )


def world_from_arrays(mass, walls=None, cheeses=(), boxes=()):
    """One hand-built scene from plain arrays; cheeses are (row, col, color), boxes (row, col, h, w, color)."""
    size = mass.shape[0]
    cheese_pos = torch.zeros(1, MAX_CHEESES, 2, dtype=torch.long)
    cheese_alive = torch.zeros(1, MAX_CHEESES, dtype=torch.bool)
    cheese_color = torch.full((1, MAX_CHEESES), -1, dtype=torch.long)
    for k, (y, x, c) in enumerate(cheeses):
        cheese_pos[0, k] = torch.tensor([y, x])
        cheese_alive[0, k] = True
        cheese_color[0, k] = c
    box_rect = torch.zeros(1, MAX_BOXES, 4, dtype=torch.long)
    box_color = torch.full((1, MAX_BOXES), -1, dtype=torch.long)
    for k, (y, x, h, w, c) in enumerate(boxes):
        box_rect[0, k] = torch.tensor([y, x, h, w])
        box_color[0, k] = c
    if walls is None:
        walls = np.zeros((size, size), dtype=bool)
    return World(
        mass=torch.tensor(mass, dtype=torch.float32).unsqueeze(0),
        walls=torch.tensor(walls).unsqueeze(0),
        cheese_pos=cheese_pos,
        cheese_alive=cheese_alive,
        cheese_color=cheese_color,
        box_rect=box_rect,
        box_color=box_color,
    )


def sample_world(rng, batch_size, size=SIZE):
    """A World of batch_size freshly sampled scenes."""
    scenes = []
    while len(scenes) < batch_size:
        scene = sample_scene(rng, size)
        if scene is not None:
            scenes.append(scene)
    return scenes_to_world(scenes)
