"""Chase: one cheese on terrain that runs from the swarm. Corner it and fill its ring to eat it.

Same robots, physics and senses as Feast (touch only, pheromones, mass-proportional
mobility). The cheese is a rigid shape that moves one square every CHEESE_PERIOD
steps, toward the side with the least robot mass:
    1. It senses robot mass within SENSE_RADIUS squares of itself, split into four
       cones (north, south, west, east) around its center.
    2. It moves into the cone with the least mass, ties going to the side
       opposite the most mass. If it senses no robots at all, it stays.
    3. It can't move into walls, or onto squares holding TAU or more robot.
       Thinner robot mass in its way is pushed into the squares it leaves, so
       total mass is conserved.
Robots move up to one square per step, and fire 3 steps in 4, so a cheese at
half that speed can be caught, but on a torus only by cutting it off: a swarm that
just follows chases it forever.

The cheese is eaten, as in Feast, once every open square touching it holds more
than TAU. Success is eating it within the time limit. The loss and the fog
constraint are Feast's.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from beebot.ca import FOG_MASS, N_NOTES, blocked_moves, damage, step
from beebot.feast import FeastRollout, first_done_step
from beebot.layout import (
    MASS_PER_RING_SQUARE,
    SWARM_DENSITY,
    UNREACHABLE,
    _random_square,
    _spread,
    grow_shape,
    open_region,
    walking_distance,
)
from beebot.render import OVERRUN, save_gif

TAU = 0.5
FIRE_PROB = 0.75
STEPS_PER_CELL = 6
LAYOUTS_PER_SIZE = 32
WALLS = (0.0, 0.3)
CHEESE_SIZE = (1, 5)
DAMAGE_FRACTION = 0.25
CHEESE_PERIOD = 2
SENSE_RADIUS = 4
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # north, south, west, east

SETS = {
    "val": {"seed": 5000, "interp": (22, 30, 36), "extrap": (48, 64)},
    "test": {"seed": 6000, "interp": (18, 26, 34), "extrap": (56, 72)},
}


@dataclass
class ChaseBatch:
    """A batch of layouts on one grid size. Every tensor is (B, H, W)."""

    mass: torch.Tensor  # starting mass, 1 per nest square
    cheese: torch.Tensor  # bool, where the cheese starts
    walls: torch.Tensor  # bool

    def to(self, device):
        return ChaseBatch(
            mass=self.mass.to(device),
            cheese=self.cheese.to(device),
            walls=self.walls.to(device),
        )


def sample_layout(rng, size, cheese_size=CHEESE_SIZE, wall_range=WALLS):
    """One layout as numpy arrays: (cheese mask, wall mask, starting mass).

    The swarm is big enough to fill the cheese's ring in open floor, and the nest
    starts off the ring, connected to it.
    """
    while True:
        walls = open_region(rng, size, rng.uniform(*wall_range))
        if walls is None:
            continue
        allowed = ~walls
        cheese = grow_shape(rng, size, rng.integers(cheese_size[0], cheese_size[1] + 1), _random_square(rng, allowed), allowed)
        if cheese is None:
            continue
        passable = ~walls & ~cheese
        ring = _spread(cheese) & passable
        if not ring.any():
            continue
        nest_allowed = passable & ~ring
        full_ring = (_spread(cheese) & ~cheese).sum()
        n_mass = max(math.ceil(MASS_PER_RING_SQUARE * full_ring), math.ceil(SWARM_DENSITY * passable.sum()))
        nest = grow_shape(rng, size, n_mass, _random_square(rng, nest_allowed), nest_allowed)
        if nest is None:
            continue
        if (walking_distance(nest, passable)[ring] >= UNREACHABLE).any():
            continue
        return cheese, walls, nest.astype(np.float32)


def sample_batch(rng, size, batch_size, cheese_size=CHEESE_SIZE, wall_range=WALLS):
    layouts = [sample_layout(rng, size, cheese_size, wall_range) for _ in range(batch_size)]
    return ChaseBatch(
        mass=torch.tensor(np.stack([m for _, _, m in layouts])),
        cheese=torch.tensor(np.stack([c for c, _, _ in layouts])),
        walls=torch.tensor(np.stack([w for _, w, _ in layouts])),
    )


def sample_training_batch(seed, size, batch_size):
    """sample_batch with its own generator, so it can run in a worker process."""
    return sample_batch(np.random.default_rng(seed), size, batch_size)


def spread(mask, radius=1):
    """(B, H, W) bool: every square within `radius` moves (8 neighbors) of a square in mask, on the torus."""
    x = F.pad(mask.float()[:, None], (radius,) * 4, mode="circular")
    return F.max_pool2d(x, 2 * radius + 1, stride=1)[:, 0] > 0


def ring_of(cheese, walls):
    return spread(cheese) & ~cheese & ~walls


def cone_masses(mass, cheese):
    """(B, 4) robot mass within SENSE_RADIUS of the cheese in each of the four cones around its center.

    The center is a circular mean, so shapes that wrap around the edge work.
    """
    b, h, w = mass.shape
    ys = torch.arange(h, device=mass.device, dtype=torch.float32)
    xs = torch.arange(w, device=mass.device, dtype=torch.float32)
    weight = cheese.float()

    def center(coords, length, dim):
        angle = 2 * math.pi * coords / length
        s = (weight.sum(dim=dim) * torch.sin(angle)).sum(dim=1)
        c = (weight.sum(dim=dim) * torch.cos(angle)).sum(dim=1)
        return torch.atan2(s, c) * length / (2 * math.pi)

    cy = center(ys, h, 2)
    cx = center(xs, w, 1)
    # Offsets from the center, wrapped into [-L/2, L/2).
    oy = torch.remainder(ys[None, :, None] - cy[:, None, None] + h / 2, h) - h / 2
    ox = torch.remainder(xs[None, None, :] - cx[:, None, None] + w / 2, w) - w / 2
    near = (spread(cheese, SENSE_RADIUS) & ~cheese).float() * mass
    vertical = oy.abs() >= ox.abs()
    cones = [
        vertical & (oy < 0),
        vertical & (oy > 0),
        ~vertical & (ox < 0),
        ~vertical & (ox > 0),
    ]
    return torch.stack([(near * cone).sum(dim=(1, 2)) for cone in cones], dim=1)


def move_cheese(mass, cheese, walls, tau):
    """One flight step for every cheese in the batch. Returns (mass, cheese). Never differentiated through
    for the choice of direction; the displaced mass keeps its gradient."""
    with torch.no_grad():
        sensed = cone_masses(mass, cheese)
        opposite = sensed[:, [1, 0, 3, 2]]
        score = sensed - 1e-3 * opposite
        legal = []
        for dy, dx in DIRECTIONS:
            moved = torch.roll(cheese, (dy, dx), dims=(1, 2))
            entered = moved & ~cheese
            legal.append(~((moved & walls) | (entered & (mass >= tau))).flatten(1).any(dim=1))
        legal = torch.stack(legal, dim=1)
        score = score.masked_fill(~legal, float("inf"))
        choice = score.argmin(dim=1)
        moves = legal.any(dim=1) & (sensed.sum(dim=1) > 0)

    new_cheese = cheese.clone()
    for k, (dy, dx) in enumerate(DIRECTIONS):
        pick = moves & (choice == k)
        if pick.any():
            new_cheese[pick] = torch.roll(cheese[pick], (dy, dx), dims=(1, 2))
    entered = (new_cheese & ~cheese).float()
    left = (cheese & ~new_cheese).float()
    pushed = (mass * entered).sum(dim=(1, 2))
    n_left = left.sum(dim=(1, 2)).clamp_min(1)
    mass = mass * (1 - entered) + left * (pushed / n_left)[:, None, None]
    return mass, new_cheese


def rollout(policy, batch, n_steps, fire_prob, tau, use_checkpoint=False, record=False, damage_step=None, damage_fraction=0.25):
    mass = batch.mass
    b, h, w = mass.shape
    notes = torch.zeros(b, N_NOTES, h, w, device=mass.device)
    walls = batch.walls.float()
    cheese = batch.cheese
    present = torch.ones(b, dtype=torch.bool, device=mass.device)
    total_mass = mass.sum(dim=(1, 2))

    deficit, crowding, fog, fog_share, left = [], [], [], [], []
    masses, pheromones, cheeses, ring_frames = [], [], [], []
    for t in range(n_steps):
        if t == damage_step:
            mass = damage(mass, damage_fraction)
        if t % CHEESE_PERIOD == CHEESE_PERIOD - 1:
            # An eaten cheese is an empty mask, which senses nothing and so never moves.
            mass, moved = move_cheese(mass, cheese & present[:, None, None], batch.walls, tau)
            cheese = torch.where(present[:, None, None], moved, cheese)
        here = cheese & present[:, None, None]
        blocked = blocked_moves(here | batch.walls)
        fire = (torch.rand(b, 1, h, w, device=mass.device) < fire_prob).float()
        args = (policy, mass, notes, here.float(), walls, blocked, fire)
        if use_checkpoint:
            mass, notes = checkpoint(step, *args, use_reentrant=False)
        else:
            mass, notes = step(*args)

        ring = ring_of(cheese, batch.walls)
        occupied = (mass / tau).clamp(0, 1)
        surround = (occupied * ring).sum(dim=(1, 2)) / ring.sum(dim=(1, 2)).clamp_min(1)
        deficit.append((1 - surround) * present)
        crowding.append(F.relu(mass - 1).pow(2).sum(dim=(1, 2)) / total_mass)
        thin = mass < FOG_MASS
        fog.append((torch.sin(torch.pi * mass / FOG_MASS) * thin).sum(dim=(1, 2)) / total_mass)
        fog_share.append((mass * thin).sum(dim=(1, 2)) / total_mass)
        full = ((mass > tau) | ~ring).flatten(1).all(dim=1)
        present = present & ~full
        left.append(present.long())
        if record:
            masses.append(mass.detach().cpu().numpy())
            pheromones.append(notes.detach().abs().amax(dim=1).cpu().numpy())
            cheeses.append((cheese & present[:, None, None]).cpu().numpy())
            ring_frames.append((ring & present[:, None, None]).cpu().numpy())

    deficit = torch.stack(deficit)
    return FeastRollout(
        deficit=deficit,
        crowding=torch.stack(crowding),
        fog=torch.stack(fog),
        fog_share=torch.stack(fog_share),
        unfinished=deficit,  # with one cheese, "some cheese is left" is its own ring deficit
        left=torch.stack(left),
        n_cheese=torch.ones(b, dtype=torch.long, device=mass.device),
        masses=masses,
        pheromones=pheromones,
        cheeses=cheeses,
        rings=ring_frames,
    )


def build_split(seed, sizes):
    rng = np.random.default_rng(seed)
    return [sample_batch(rng, size, LAYOUTS_PER_SIZE) for size in sizes]


def score_batches(policy, batches, device, seed, damage):
    torch.manual_seed(seed)
    successes, time_scores, fogs = [], [], []
    for batch in batches:
        batch = batch.to(device)
        n_steps = STEPS_PER_CELL * batch.mass.shape[-1]
        damage_step = n_steps // 4 if damage else None
        with torch.no_grad():
            result = rollout(policy, batch, n_steps, FIRE_PROB, TAU, damage_step=damage_step, damage_fraction=DAMAGE_FRACTION)
        first = first_done_step(result).float()
        success = first > 0
        successes.append(success.float())
        time_scores.append(torch.where(success, first / n_steps, torch.ones_like(first)))
        fogs.append(result.fog_share.mean(dim=0))
    success = torch.cat(successes).mean().item()
    return {
        "success": success,
        "time": torch.cat(time_scores).mean().item(),
        "eaten_share": success,  # one cheese: eaten share and success coincide
        "fog": torch.cat(fogs).mean().item(),
    }


def evaluate(policy, device, which="val"):
    """Scores per split, plus "score": the mean time over splits (lower is better, 1 = never eats)."""
    spec = SETS[which]
    interp = build_split(spec["seed"], spec["interp"])
    extrap = build_split(spec["seed"] + 1, spec["extrap"])
    results = {
        "interp": score_batches(policy, interp, device, spec["seed"], damage=False),
        "extrap": score_batches(policy, extrap, device, spec["seed"], damage=False),
        "damage": score_batches(policy, interp, device, spec["seed"] + 2, damage=True),
    }
    results["score"] = float(np.mean([results[name]["time"] for name in ("interp", "extrap", "damage")]))
    return results


def record(policy, batch, device, n_steps):
    torch.manual_seed(0)  # same firing pattern for every policy, so videos compare like for like
    with torch.no_grad():
        result = rollout(policy, batch.to(device), n_steps, FIRE_PROB, TAU, record=True)
    start = batch.cheese.numpy()
    cheeses = [start] + result.cheeses
    rings = [ring_of(batch.cheese, batch.walls).numpy()] + result.rings
    return result, cheeses, rings


def _save(path, batch, i, result, cheeses, rings, limit, n_steps):
    save_gif(
        path,
        batch.mass[i].numpy(),
        batch.walls[i].numpy(),
        [c[i] for c in cheeses],
        [r[i] for r in rings],
        [m[i] for m in result.masses],
        [p[i] for p in result.pheromones],
        limit,
        n_steps,
    )


def render(policy, batch, device, path, n_examples=3):
    limit = STEPS_PER_CELL * batch.mass.shape[-1]
    n_steps = OVERRUN * limit
    result, cheeses, rings = record(policy, batch, device, n_steps)
    for i in range(min(n_examples, batch.mass.shape[0])):
        _save(path.with_name(f"{path.stem}_{i}.gif"), batch, i, result, cheeses, rings, limit, n_steps)


def render_examples(policy, device, out_dir):
    """A few val layouts from one interp size and one extrap size."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = SETS["val"]
    render(policy, build_split(spec["seed"], spec["interp"][:1])[0], device, out_dir / "interp")
    render(policy, build_split(spec["seed"] + 1, spec["extrap"][:1])[0], device, out_dir / "extrap")


def render_successes(policy, batch, device, out_dir, prefix, n_examples=3):
    """The policy's slowest catches within the limit: the longest chases it still wins.

    Returns what was picked, for the page's caption.
    """
    size = batch.mass.shape[-1]
    limit = STEPS_PER_CELL * size
    n_steps = OVERRUN * limit
    result, cheeses, rings = record(policy, batch, device, n_steps)
    done = (result.left[:limit] == 0).cpu()
    solved = done.any(dim=0)
    first = done.float().argmax(dim=0) + 1
    order = sorted(np.flatnonzero(solved.numpy()), key=lambda i: -first[i].item())[:n_examples]

    clips = []
    for k, i in enumerate(order):
        name = f"{prefix}_{k}.gif"
        _save(out_dir / name, batch, i, result, cheeses, rings, limit, n_steps)
        clips.append({"file": name, "cheeses": 1, "steps": int(first[i]), "limit": limit})
    return {"split": prefix, "size": size, "layouts": len(solved), "solved": int(solved.sum()), "clips": clips}


def render_demo(policy, device, out_dir):
    """Demo clips for the page: long catches from one interp size and one extrap size."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = SETS["val"]
    return [
        render_successes(policy, build_split(spec["seed"], spec["interp"][:1])[0], device, out_dir, "interp"),
        render_successes(policy, build_split(spec["seed"] + 1, spec["extrap"][:1])[0], device, out_dir, "extrap"),
    ]
