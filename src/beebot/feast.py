"""Feast: several cheeses on terrain. Each cheese vanishes once its ring is full; eat them all.

Same robots, physics and senses as the terrain stage (touch only, pheromones,
mass-proportional mobility), with three differences:
    1. Several cheeses. They never touch each other, though their rings may share squares.
    2. A cheese is eaten the moment every square of its ring holds more than tau,
       and then it disappears, so its squares become open floor.
    3. Nothing requires the whole swarm to gather: with several cheeses the
       swarm has to split. Success is eating every cheese within the time limit.

The loss is the task, the robots' size, and two shaping terms the user asked for
after the first runs dissolved into fog (a film of fractional robots over the
whole world that condenses onto each ring):
    deficit:  the unfilled share of each remaining cheese's ring
    crowding: squares holding more than one robot (robots take up space)
    fog:      squares holding some mass but less than half a robot, per robot
    time:     a cost for every step until the last cheese is eaten
No distance or exploration terms, no curriculum. Splitting the swarm is allowed.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from beebot.ca import FOG_MASS, N_NOTES, blocked_moves, damage, step
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

SETS = {
    "val": {
        "seed": 3000,
        "interp": {"sizes": (22, 30, 36), "n_cheese": (2, 4)},
        "extrap": {"sizes": (48, 64), "n_cheese": (5, 6)},
    },
    "test": {
        "seed": 4000,
        "interp": {"sizes": (18, 26, 34), "n_cheese": (2, 4)},
        "extrap": {"sizes": (56, 72), "n_cheese": (5, 6)},
    },
}


@dataclass
class FeastBatch:
    """A batch of layouts on one grid size. K is the most cheeses any layout in the batch has."""

    mass: torch.Tensor  # (B, H, W) starting mass, 1 per nest square
    cheese: torch.Tensor  # (B, K, H, W) bool, one mask per cheese; unused slots are empty
    walls: torch.Tensor  # (B, H, W) bool
    rings: torch.Tensor  # (B, K, H, W) bool, the open squares touching each cheese

    def to(self, device):
        return FeastBatch(
            mass=self.mass.to(device),
            cheese=self.cheese.to(device),
            walls=self.walls.to(device),
            rings=self.rings.to(device),
        )


def sample_layout(rng, size, n_cheese_range, cheese_size=CHEESE_SIZE, wall_range=WALLS):
    """One layout as numpy arrays: (list of cheese masks, wall mask, starting mass).

    Cheeses are separated by at least one square. The nest starts anywhere off
    the rings, and every ring square is reachable from it without crossing a cheese.
    """
    while True:
        walls = open_region(rng, size, rng.uniform(*wall_range))
        if walls is None:
            continue
        n_cheese = rng.integers(n_cheese_range[0], n_cheese_range[1] + 1)
        cheeses = []
        any_cheese = np.zeros_like(walls)
        for _ in range(n_cheese):
            allowed = ~walls & ~_spread(any_cheese)
            if not allowed.any():
                break
            shape = grow_shape(rng, size, rng.integers(cheese_size[0], cheese_size[1] + 1), _random_square(rng, allowed), allowed)
            if shape is None:
                break
            cheeses.append(shape)
            any_cheese |= shape
        if len(cheeses) < n_cheese:
            continue

        passable = ~walls & ~any_cheese
        rings = [_spread(c) & passable for c in cheeses]
        if not all(r.any() for r in rings):
            continue
        all_rings = np.logical_or.reduce(rings)
        nest_allowed = passable & ~all_rings
        if not nest_allowed.any():
            continue
        n_mass = max(math.ceil(MASS_PER_RING_SQUARE * all_rings.sum()), math.ceil(SWARM_DENSITY * passable.sum()))
        nest = grow_shape(rng, size, n_mass, _random_square(rng, nest_allowed), nest_allowed)
        if nest is None:
            continue
        if (walking_distance(nest, passable)[all_rings] >= UNREACHABLE).any():
            continue
        return cheeses, walls, nest.astype(np.float32)


def sample_batch(rng, size, n_cheese_range, batch_size, cheese_size=CHEESE_SIZE, wall_range=WALLS):
    layouts = [sample_layout(rng, size, n_cheese_range, cheese_size, wall_range) for _ in range(batch_size)]
    k = n_cheese_range[1]
    cheese = np.zeros((batch_size, k, size, size), dtype=bool)
    rings = np.zeros_like(cheese)
    for i, (cheeses, walls, _) in enumerate(layouts):
        for j, c in enumerate(cheeses):
            cheese[i, j] = c
            rings[i, j] = _spread(c) & ~walls & ~np.logical_or.reduce(cheeses)
    return FeastBatch(
        mass=torch.tensor(np.stack([m for _, _, m in layouts])),
        cheese=torch.tensor(cheese),
        walls=torch.tensor(np.stack([w for _, w, _ in layouts])),
        rings=torch.tensor(rings),
    )


def sample_batch_seeded(seed, size, n_cheese_range, batch_size):
    """sample_batch with its own generator, so it can run in a worker process."""
    return sample_batch(np.random.default_rng(seed), size, n_cheese_range, batch_size)


@dataclass
class FeastRollout:
    """Per-step series are (T, B)."""

    deficit: torch.Tensor  # unfilled share of the remaining cheeses' rings, averaged over all cheeses
    crowding: torch.Tensor  # how far squares go over 1 unit of mass
    fog: torch.Tensor  # soft count of squares holding less than FOG_MASS, per unit of swarm mass
    fog_share: torch.Tensor  # share of the mass on squares holding less than FOG_MASS (diagnostic)
    unfinished: torch.Tensor  # soft "some cheese is still left": 0 once everything is eaten
    left: torch.Tensor  # number of cheeses not yet eaten
    n_cheese: torch.Tensor  # (B,)
    masses: list  # (B, H, W) numpy arrays per step, only when recorded
    pheromones: list
    cheeses: list  # (B, H, W) bool numpy arrays per step: the cheese still present after the step
    rings: list  # (B, H, W) bool numpy arrays per step: rings of the cheese still present after the step


def rollout(policy, batch, n_steps, fire_prob, tau, use_checkpoint=False, record=False, damage_step=None, damage_fraction=0.25):
    mass = batch.mass
    b, h, w = mass.shape
    notes = torch.zeros(b, N_NOTES, h, w, device=mass.device)
    walls = batch.walls.float()
    rings = batch.rings.float()
    ring_size = rings.sum(dim=(2, 3)).clamp_min(1)
    present = batch.cheese.flatten(2).any(dim=2)  # (B, K)
    n_cheese = present.sum(dim=1)
    total_mass = mass.sum(dim=(1, 2))

    deficit, crowding, fog, fog_share, unfinished, left = [], [], [], [], [], []
    masses, pheromones, cheeses, ring_frames = [], [], [], []
    for t in range(n_steps):
        if t == damage_step:
            mass = damage(mass, damage_fraction)
        cheese = (batch.cheese & present[:, :, None, None]).any(dim=1)
        blocked = blocked_moves(cheese | batch.walls)
        fire = (torch.rand(b, 1, h, w, device=mass.device) < fire_prob).float()
        args = (policy, mass, notes, cheese.float(), walls, blocked, fire)
        if use_checkpoint:
            mass, notes = checkpoint(step, *args, use_reentrant=False)
        else:
            mass, notes = step(*args)

        occupied = (mass / tau).clamp(0, 1)
        surround = (occupied[:, None] * rings).sum(dim=(2, 3)) / ring_size
        deficit.append(((1 - surround) * present).sum(dim=1) / n_cheese)
        crowding.append(F.relu(mass - 1).pow(2).sum(dim=(1, 2)) / total_mass)
        # A bump that is 0 on empty squares and at FOG_MASS, and peaks in between, so every
        # thin square is pushed either to empty or up to half a robot.
        thin = mass < FOG_MASS
        fog.append((torch.sin(torch.pi * mass / FOG_MASS) * thin).sum(dim=(1, 2)) / total_mass)
        fog_share.append((mass * thin).sum(dim=(1, 2)) / total_mass)
        # 1 - P(every cheese is done), treating each remaining cheese's ring coverage as its chance.
        unfinished.append(1 - torch.prod(1 - (1 - surround) * present, dim=1))
        full = ((mass[:, None] > tau) | ~batch.rings).flatten(2).all(dim=2)
        present = present & ~full
        left.append(present.sum(dim=1))
        if record:
            masses.append(mass.detach().cpu().numpy())
            pheromones.append(notes.detach().abs().amax(dim=1).cpu().numpy())
            cheeses.append((batch.cheese & present[:, :, None, None]).any(dim=1).cpu().numpy())
            ring_frames.append((batch.rings & present[:, :, None, None]).any(dim=1).cpu().numpy())

    return FeastRollout(
        deficit=torch.stack(deficit),
        crowding=torch.stack(crowding),
        fog=torch.stack(fog),
        fog_share=torch.stack(fog_share),
        unfinished=torch.stack(unfinished),
        left=torch.stack(left),
        n_cheese=n_cheese,
        masses=masses,
        pheromones=pheromones,
        cheeses=cheeses,
        rings=ring_frames,
    )


def loss(result, w_crowding, w_fog=0.0, w_time=0.0):
    """Averaged over every step, so eating sooner scores better."""
    per_step = result.deficit + w_crowding * result.crowding + w_fog * result.fog + w_time * result.unfinished
    return per_step.mean()


def first_done_step(result):
    """(B,) step index (1-based) at which the last cheese was eaten, or 0 if some cheese is left."""
    done = result.left == 0
    first = done.float().argmax(dim=0) + 1
    return torch.where(done.any(dim=0), first, torch.zeros_like(first))


def build_split(seed, sizes, n_cheese):
    rng = np.random.default_rng(seed)
    return [sample_batch(rng, size, n_cheese, LAYOUTS_PER_SIZE) for size in sizes]


def score_batches(policy, batches, device, seed, damage):
    torch.manual_seed(seed)
    successes, time_scores, eaten_shares, fogs = [], [], [], []
    for batch in batches:
        batch = batch.to(device)
        n_steps = STEPS_PER_CELL * batch.mass.shape[-1]
        damage_step = n_steps // 4 if damage else None
        with torch.no_grad():
            result = rollout(policy, batch, n_steps, FIRE_PROB, TAU, damage_step=damage_step, damage_fraction=DAMAGE_FRACTION)
        first = first_done_step(result).float()
        success = first > 0
        successes.append(success.float())
        # Fraction of the rollout used to eat everything; a failure counts as the whole rollout.
        time_scores.append(torch.where(success, first / n_steps, torch.ones_like(first)))
        eaten_shares.append(1 - result.left[-1].float() / result.n_cheese.float())
        fogs.append(result.fog_share.mean(dim=0))
    return {
        "success": torch.cat(successes).mean().item(),
        "time": torch.cat(time_scores).mean().item(),
        "eaten_share": torch.cat(eaten_shares).mean().item(),
        "fog": torch.cat(fogs).mean().item(),
    }


def evaluate(policy, device, which="val"):
    """Scores per split, plus "score": the mean time over splits (lower is better, 1 = never eats everything)."""
    spec = SETS[which]
    interp = build_split(spec["seed"], spec["interp"]["sizes"], spec["interp"]["n_cheese"])
    extrap = build_split(spec["seed"] + 1, spec["extrap"]["sizes"], spec["extrap"]["n_cheese"])
    results = {
        "interp": score_batches(policy, interp, device, spec["seed"], damage=False),
        "extrap": score_batches(policy, extrap, device, spec["seed"], damage=False),
        "damage": score_batches(policy, interp, device, spec["seed"] + 2, damage=True),
    }
    results["score"] = float(np.mean([results[name]["time"] for name in ("interp", "extrap", "damage")]))
    return results


def record(policy, batch, device, n_steps):
    """A recorded rollout, with the cheese and ring masks of every frame, the start included."""
    torch.manual_seed(0)  # same firing pattern for every policy, so videos compare like for like
    with torch.no_grad():
        result = rollout(policy, batch.to(device), n_steps, FIRE_PROB, TAU, record=True)
    cheeses = [batch.cheese.any(dim=1).numpy()] + result.cheeses
    rings = [batch.rings.any(dim=1).numpy()] + result.rings
    return result, cheeses, rings


def render(policy, batch, device, path, n_examples=3):
    limit = STEPS_PER_CELL * batch.mass.shape[-1]
    n_steps = OVERRUN * limit
    result, cheeses, rings = record(policy, batch, device, n_steps)
    for i in range(min(n_examples, batch.mass.shape[0])):
        save_gif(
            path.with_name(f"{path.stem}_{i}.gif"),
            batch.mass[i].numpy(),
            batch.walls[i].numpy(),
            [c[i] for c in cheeses],
            [r[i] for r in rings],
            [m[i] for m in result.masses],
            [p[i] for p in result.pheromones],
            limit,
            n_steps,
        )


def render_examples(policy, device, out_dir):
    """A few val layouts from one interp size and one extrap size."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = SETS["val"]
    interp = build_split(spec["seed"], spec["interp"]["sizes"][:1], spec["interp"]["n_cheese"])
    extrap = build_split(spec["seed"] + 1, spec["extrap"]["sizes"][:1], spec["extrap"]["n_cheese"])
    render(policy, interp[0], device, out_dir / "interp")
    render(policy, extrap[0], device, out_dir / "extrap")


def render_successes(policy, batch, device, out_dir, prefix, n_examples=3):
    """The hardest layouts the policy solves within the limit: its successes with the most cheeses.

    Returns what was picked, for the page's caption.
    """
    size = batch.mass.shape[-1]
    limit = STEPS_PER_CELL * size
    n_steps = OVERRUN * limit
    result, cheeses, rings = record(policy, batch, device, n_steps)
    done = (result.left[:limit] == 0).cpu()
    solved = done.any(dim=0)
    first = done.float().argmax(dim=0) + 1
    n_cheese = result.n_cheese.cpu()
    order = sorted(np.flatnonzero(solved.numpy()), key=lambda i: (-n_cheese[i].item(), first[i].item()))[:n_examples]

    clips = []
    for k, i in enumerate(order):
        name = f"{prefix}_{k}.gif"
        save_gif(
            out_dir / name,
            batch.mass[i].numpy(),
            batch.walls[i].numpy(),
            [c[i] for c in cheeses],
            [r[i] for r in rings],
            [m[i] for m in result.masses],
            [p[i] for p in result.pheromones],
            limit,
            n_steps,
        )
        clips.append({"file": name, "cheeses": int(n_cheese[i]), "steps": int(first[i]), "limit": limit})
    return {"split": prefix, "size": size, "layouts": len(solved), "solved": int(solved.sum()), "clips": clips}


def render_demo(policy, device, out_dir):
    """Demo clips for the page: the hardest val layouts solved, from one interp size and one extrap size."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = SETS["val"]
    interp = build_split(spec["seed"], spec["interp"]["sizes"][:1], spec["interp"]["n_cheese"])
    extrap = build_split(spec["seed"] + 1, spec["extrap"]["sizes"][:1], spec["extrap"]["n_cheese"])
    return [
        render_successes(policy, interp[0], device, out_dir, "interp"),
        render_successes(policy, extrap[0], device, out_dir, "extrap"),
    ]
