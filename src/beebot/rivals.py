"""Rivals: two swarms, A and B, in one world. The physics is fixed; the payoff rule (the game) varies.

Each swarm is its own species with its own rule (a RivalNCA), mass and pheromones.
A square senses both species' mass and both species' pheromones through its 3x3
window, and writes only its own species' pheromones. Robots of both species share
square capacity (crowding counts their combined mass), and both are blocked by
walls and cheese. Both species move at the same time.

A cheese is eaten once every square of its ring holds more than TAU of robot,
counting both species together. At that moment each species owns the share of the
ring's mass that is its own, and that is its share of the cheese's value.

Games, each a different payoff on the same physics:
    shared:   both swarms are paid for every cheese eaten, whoever eats it.
              A cooperative control; the same objective as Feast.
    zerosum:  each swarm is paid its share of every cheese minus the other's,
              so one swarm's gain is the other's loss. A competitive control.
    staghunt: two small cheeses either swarm can eat alone (value 1 each), and one
              big cheese (value 4) whose ring needs more mass than either swarm
              has. Each swarm is paid its own share. Hunting the stag together pays
              more, but only if the other swarm comes too.

The loss is each swarm's payoff, made smooth: at every step a remaining cheese pays
its value times its ring's fill times the swarm's share of the ring mass, and an
eaten cheese pays its value times the swarm's share at the moment it was eaten.
Averaged over steps, so eating sooner pays more. Each swarm's update touches only
its own rule.
"""

import math
from dataclasses import dataclass
from functools import partial
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

TAU = 0.5
FIRE_PROB = 0.75
STEPS_PER_CELL = 6
LAYOUTS_PER_SIZE = 32
WALLS = (0.0, 0.3)
DAMAGE_FRACTION = 0.25
GAMES = ("shared", "zerosum", "staghunt")

SMALL_CHEESE = (1, 5)  # squares, shared and zerosum
N_CHEESE = (2, 4)
STAG_SMALL = 1  # squares of each small cheese in the stag hunt
STAG_BIG = (12, 20)  # squares of the stag
STAG_VALUES = (1.0, 1.0, 4.0)
# In the stag hunt each swarm has this share of the mass the stag's ring needs (TAU per
# ring square), so neither can eat it alone and together they have 60% to spare.
STAG_SWARM_SHARE = 0.8

SETS = {
    "val": {"seed": 7000, "interp": (22, 30, 36), "extrap": (48, 64)},
    "test": {"seed": 8000, "interp": (18, 26, 34), "extrap": (56, 72)},
}


@dataclass
class RivalBatch:
    """A batch of layouts on one grid size. K is the most cheeses any layout in the batch has."""

    mass_a: torch.Tensor  # (B, H, W)
    mass_b: torch.Tensor  # (B, H, W)
    cheese: torch.Tensor  # (B, K, H, W) bool; unused slots are empty
    value: torch.Tensor  # (B, K), 0 for unused slots
    walls: torch.Tensor  # (B, H, W) bool
    rings: torch.Tensor  # (B, K, H, W) bool

    def to(self, device):
        return RivalBatch(
            mass_a=self.mass_a.to(device),
            mass_b=self.mass_b.to(device),
            cheese=self.cheese.to(device),
            value=self.value.to(device),
            walls=self.walls.to(device),
            rings=self.rings.to(device),
        )

    def swapped(self):
        """The same layouts with the two nests exchanged, so A and B can be scored from both starts."""
        return RivalBatch(
            mass_a=self.mass_b,
            mass_b=self.mass_a,
            cheese=self.cheese,
            value=self.value,
            walls=self.walls,
            rings=self.rings,
        )


def _cheeses(rng, size, game, walls):
    """Cheese masks and values for one layout, or None if they didn't fit."""
    if game == "staghunt":
        shapes = [STAG_SMALL, STAG_SMALL, int(rng.integers(STAG_BIG[0], STAG_BIG[1] + 1))]
        values = list(STAG_VALUES)
    else:
        n = int(rng.integers(N_CHEESE[0], N_CHEESE[1] + 1))
        shapes = [int(rng.integers(SMALL_CHEESE[0], SMALL_CHEESE[1] + 1)) for _ in range(n)]
        values = [1.0] * n
    cheeses = []
    any_cheese = np.zeros_like(walls)
    for n_cells in shapes:
        allowed = ~walls & ~_spread(any_cheese)
        if not allowed.any():
            return None
        shape = grow_shape(rng, size, n_cells, _random_square(rng, allowed), allowed)
        if shape is None:
            return None
        cheeses.append(shape)
        any_cheese |= shape
    return cheeses, values


def sample_layout(rng, size, game, wall_range=WALLS):
    """One layout as numpy arrays: (cheese masks, values, walls, mass of A, mass of B).

    The two nests don't overlap, both start off every ring, and every ring square is
    reachable from both.
    """
    while True:
        walls = open_region(rng, size, rng.uniform(*wall_range))
        if walls is None:
            continue
        placed = _cheeses(rng, size, game, walls)
        if placed is None:
            continue
        cheeses, values = placed
        any_cheese = np.logical_or.reduce(cheeses)
        passable = ~walls & ~any_cheese
        rings = [_spread(c) & passable for c in cheeses]
        if not all(r.any() for r in rings):
            continue
        all_rings = np.logical_or.reduce(rings)
        if game == "staghunt":
            n_mass = math.ceil(STAG_SWARM_SHARE * TAU * rings[-1].sum())
            if n_mass <= TAU * max(r.sum() for r in rings[:-1]):
                continue  # each swarm must be able to eat a small cheese alone
        else:
            # Each swarm alone can fill the biggest ring, and the two together have Feast's density.
            n_mass = max(math.ceil(MASS_PER_RING_SQUARE * max(r.sum() for r in rings)), math.ceil(SWARM_DENSITY / 2 * passable.sum()))
        nest_allowed = passable & ~all_rings
        nests = []
        for _ in range(2):
            if not nest_allowed.any():
                break
            nest = grow_shape(rng, size, n_mass, _random_square(rng, nest_allowed), nest_allowed)
            if nest is None:
                break
            nests.append(nest)
            nest_allowed = nest_allowed & ~nest
        if len(nests) < 2:
            continue
        if any((walking_distance(nest, passable)[all_rings] >= UNREACHABLE).any() for nest in nests):
            continue
        return cheeses, values, walls, nests[0].astype(np.float32), nests[1].astype(np.float32)


def sample_batch(rng, size, game, batch_size, wall_range=WALLS):
    layouts = [sample_layout(rng, size, game, wall_range) for _ in range(batch_size)]
    k = max(len(c) for c, _, _, _, _ in layouts)
    cheese = np.zeros((batch_size, k, size, size), dtype=bool)
    rings = np.zeros_like(cheese)
    value = np.zeros((batch_size, k), dtype=np.float32)
    for i, (cheeses, values, walls, _, _) in enumerate(layouts):
        any_cheese = np.logical_or.reduce(cheeses)
        for j, (c, v) in enumerate(zip(cheeses, values)):
            cheese[i, j] = c
            rings[i, j] = _spread(c) & ~walls & ~any_cheese
            value[i, j] = v
    return RivalBatch(
        mass_a=torch.tensor(np.stack([a for _, _, _, a, _ in layouts])),
        mass_b=torch.tensor(np.stack([b for _, _, _, _, b in layouts])),
        cheese=torch.tensor(cheese),
        value=torch.tensor(value),
        walls=torch.tensor(np.stack([w for _, _, w, _, _ in layouts])),
        rings=torch.tensor(rings),
    )


def sample_training_batch(seed, size, game, batch_size):
    """sample_batch with its own generator, so it can run in a worker process."""
    return sample_batch(np.random.default_rng(seed), size, game, batch_size)


@dataclass
class RivalRollout:
    """Per-step series are (T, B)."""

    rate_a: torch.Tensor  # A's smooth payoff at each step, as a share of the layout's total value
    rate_b: torch.Tensor
    crowding: torch.Tensor  # how far squares go over 1 unit of combined mass
    fog_a: torch.Tensor  # soft count of A's thin squares, per unit of A's mass (the loss term)
    fog_b: torch.Tensor
    fog_share_a: torch.Tensor  # share of A's mass on squares where A holds less than FOG_MASS
    fog_share_b: torch.Tensor
    eaten_value: torch.Tensor  # share of the layout's value eaten so far
    payoff_a: torch.Tensor  # (B,) A's share of the value eaten, by ring ownership at the moment of eating
    payoff_b: torch.Tensor
    stag_eaten: torch.Tensor  # (B,) bool, the last cheese eaten (the stag, in the stag hunt)
    masses_a: list  # (B, H, W) numpy arrays per step, only when recorded
    masses_b: list
    cheeses: list  # (B, H, W) bool numpy arrays per step: the cheese still present after the step


def _policy(model, other, other_notes):
    """A rival's rule as a one-species policy, with the other species frozen at this step's start."""

    def policy(mass, cheese, walls, notes):
        return model(mass, other, cheese, walls, notes, other_notes)

    return policy


def _fog(mass, total):
    thin = mass < FOG_MASS
    soft = (torch.sin(torch.pi * mass / FOG_MASS) * thin).sum(dim=(1, 2)) / total
    share = (mass * thin).sum(dim=(1, 2)) / total
    return soft, share


def rollout(model_a, model_b, batch, n_steps, fire_prob, tau, use_checkpoint=False, record=False, damage_step=None, damage_fraction=DAMAGE_FRACTION, blackout=False):
    """blackout: each species reads the other's pheromones as zero (a probe of cross-species signalling)."""
    mass_a, mass_b = batch.mass_a, batch.mass_b
    b, h, w = mass_a.shape
    notes_a = torch.zeros(b, N_NOTES, h, w, device=mass_a.device)
    notes_b = torch.zeros_like(notes_a)
    walls = batch.walls.float()
    rings = batch.rings.float()
    ring_size = rings.sum(dim=(2, 3)).clamp_min(1)
    present = batch.cheese.flatten(2).any(dim=2)
    value = batch.value
    total_value = value.sum(dim=1)
    total_a = mass_a.sum(dim=(1, 2))
    total_b = mass_b.sum(dim=(1, 2))
    # Ownership of each cheese at the moment it was eaten (soft, so it keeps its gradient).
    kept_a = torch.zeros_like(value)
    kept_b = torch.zeros_like(value)
    payoff_a = torch.zeros(b, device=mass_a.device)
    payoff_b = torch.zeros(b, device=mass_a.device)

    rate_a, rate_b, crowding, fog_a, fog_b, fog_share_a, fog_share_b, eaten_value = [], [], [], [], [], [], [], []
    masses_a, masses_b, cheeses = [], [], []
    for t in range(n_steps):
        if t == damage_step:
            mass_a = damage(mass_a, damage_fraction)
            mass_b = damage(mass_b, damage_fraction)
        cheese = (batch.cheese & present[:, :, None, None]).any(dim=1)
        blocked = blocked_moves(cheese | batch.walls)
        heard_b = torch.zeros_like(notes_b) if blackout else notes_b
        heard_a = torch.zeros_like(notes_a) if blackout else notes_a
        fire_a = (torch.rand(b, 1, h, w, device=mass_a.device) < fire_prob).float()
        fire_b = (torch.rand(b, 1, h, w, device=mass_a.device) < fire_prob).float()
        args_a = (_policy(model_a, mass_b, heard_b), mass_a, notes_a, cheese.float(), walls, blocked, fire_a)
        args_b = (_policy(model_b, mass_a, heard_a), mass_b, notes_b, cheese.float(), walls, blocked, fire_b)
        if use_checkpoint:
            new_a, new_notes_a = checkpoint(step, *args_a, use_reentrant=False)
            new_b, new_notes_b = checkpoint(step, *args_b, use_reentrant=False)
        else:
            new_a, new_notes_a = step(*args_a)
            new_b, new_notes_b = step(*args_b)
        mass_a, notes_a, mass_b, notes_b = new_a, new_notes_a, new_b, new_notes_b

        total = mass_a + mass_b
        occupied = (total / tau).clamp(0, 1)
        surround = (occupied[:, None] * rings).sum(dim=(2, 3)) / ring_size
        ring_a = (mass_a[:, None] * rings).sum(dim=(2, 3))
        ring_b = (mass_b[:, None] * rings).sum(dim=(2, 3))
        own_a = ring_a / (ring_a + ring_b).clamp_min(1e-6)
        own_b = ring_b / (ring_a + ring_b).clamp_min(1e-6)

        full = ((total[:, None] > tau) | ~batch.rings).flatten(2).all(dim=2) & present
        kept_a = torch.where(full, own_a, kept_a)
        kept_b = torch.where(full, own_b, kept_b)
        payoff_a = payoff_a + (full.float() * value * own_a.detach()).sum(dim=1) / total_value
        payoff_b = payoff_b + (full.float() * value * own_b.detach()).sum(dim=1) / total_value
        present = present & ~full

        # A remaining cheese pays its fill times the ring share; an eaten one pays the share it was eaten at.
        secured_a = torch.where(present, surround * own_a, kept_a)
        secured_b = torch.where(present, surround * own_b, kept_b)
        rate_a.append((value * secured_a).sum(dim=1) / total_value)
        rate_b.append((value * secured_b).sum(dim=1) / total_value)
        crowding.append(F.relu(total - 1).pow(2).sum(dim=(1, 2)) / (total_a + total_b))
        soft, share = _fog(mass_a, total_a)
        fog_a.append(soft)
        fog_share_a.append(share)
        soft, share = _fog(mass_b, total_b)
        fog_b.append(soft)
        fog_share_b.append(share)
        eaten_value.append(((~present).float() * value).sum(dim=1) / total_value)
        if record:
            masses_a.append(mass_a.detach().cpu().numpy())
            masses_b.append(mass_b.detach().cpu().numpy())
            cheeses.append((batch.cheese & present[:, :, None, None]).any(dim=1).cpu().numpy())

    last = (batch.value > 0).float().cumsum(dim=1).argmax(dim=1)
    stag_eaten = ~present.gather(1, last[:, None])[:, 0]
    return RivalRollout(
        rate_a=torch.stack(rate_a),
        rate_b=torch.stack(rate_b),
        crowding=torch.stack(crowding),
        fog_a=torch.stack(fog_a),
        fog_b=torch.stack(fog_b),
        fog_share_a=torch.stack(fog_share_a),
        fog_share_b=torch.stack(fog_share_b),
        eaten_value=torch.stack(eaten_value),
        payoff_a=payoff_a,
        payoff_b=payoff_b,
        stag_eaten=stag_eaten,
        masses_a=masses_a,
        masses_b=masses_b,
        cheeses=cheeses,
    )


def losses(result, game, w_crowding, w_fog_a=0.0, w_fog_b=0.0):
    """(loss of A, loss of B). Each swarm minimizes the negative of its own smooth payoff."""
    if game == "shared":
        payoff_a = payoff_b = (result.rate_a + result.rate_b).mean()
    elif game == "zerosum":
        payoff_a = (result.rate_a - result.rate_b).mean()
        payoff_b = -payoff_a
    elif game == "staghunt":
        payoff_a = result.rate_a.mean()
        payoff_b = result.rate_b.mean()
    else:
        raise ValueError(game)
    crowding = w_crowding * result.crowding.mean()
    loss_a = -payoff_a + crowding + w_fog_a * result.fog_a.mean()
    loss_b = -payoff_b + crowding + w_fog_b * result.fog_b.mean()
    return loss_a, loss_b


def build_split(seed, sizes, game):
    rng = np.random.default_rng(seed)
    return [sample_batch(rng, size, game, LAYOUTS_PER_SIZE) for size in sizes]


def score_batches(model_a, model_b, batches, device, seed, damage, blackout=False):
    """Scores of a pair. Each layout is played twice, once from each nest, so neither swarm gets the better start."""
    torch.manual_seed(seed)
    rows = {"eaten": [], "all_eaten": [], "payoff_a": [], "payoff_b": [], "stag": [], "fog_a": [], "fog_b": []}
    for batch in batches:
        for layout in (batch, batch.swapped()):
            layout = layout.to(device)
            n_steps = STEPS_PER_CELL * layout.mass_a.shape[-1]
            damage_step = n_steps // 4 if damage else None
            with torch.no_grad():
                result = rollout(model_a, model_b, layout, n_steps, FIRE_PROB, TAU, damage_step=damage_step, blackout=blackout)
            final = result.eaten_value[-1]
            rows["eaten"].append(final)
            rows["all_eaten"].append((final > 0.999).float())
            rows["payoff_a"].append(result.payoff_a)
            rows["payoff_b"].append(result.payoff_b)
            rows["stag"].append(result.stag_eaten.float())
            rows["fog_a"].append(result.fog_share_a.mean(dim=0))
            rows["fog_b"].append(result.fog_share_b.mean(dim=0))
    scores = {name: torch.cat(values).mean().item() for name, values in rows.items()}
    scores["welfare"] = scores["payoff_a"] + scores["payoff_b"]
    scores["fog"] = max(scores["fog_a"], scores["fog_b"])
    return scores


def evaluate(model_a, model_b, device, game, which="val", blackout=False):
    """Scores per split for one pair in one game, plus "score": 1 minus the mean eaten share
    (lower is better; the share of value eaten, whoever ate it)."""
    spec = SETS[which]
    seed = spec["seed"] + 10 * GAMES.index(game)
    interp = build_split(seed, spec["interp"], game)
    extrap = build_split(seed + 1, spec["extrap"], game)
    results = {
        "interp": score_batches(model_a, model_b, interp, device, seed, damage=False, blackout=blackout),
        "extrap": score_batches(model_a, model_b, extrap, device, seed, damage=False, blackout=blackout),
        "damage": score_batches(model_a, model_b, interp, device, seed + 2, damage=True, blackout=blackout),
    }
    results["score"] = float(np.mean([1 - results[name]["eaten"] for name in ("interp", "extrap", "damage")]))
    return results
