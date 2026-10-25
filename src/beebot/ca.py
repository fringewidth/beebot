"""The cellular automaton: one step, a full rollout, and the loss.

A policy is anything callable as policy(mass, smell, notes) -> (logits, note_updates):
    mass:  (B, H, W)
    smell: (B, H, W)
    notes: (B, N_NOTES, H, W)
    logits:       (B, 9, H, W), one score per destination in OFFSETS
    note_updates: (B, N_NOTES, H, W), added to the notes
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Destinations a square can send mass to, as (dy, dx). Index 0 is "stay".
# Rows grow downward, so (-1, 0) is north.
OFFSETS = [(0, 0), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
N_NOTES = 8


def look(field, dy, dx):
    """field[..., y + dy, x + dx], aligned so the result is indexed by (y, x)."""
    return torch.roll(field, shifts=(-dy, -dx), dims=(-2, -1))


def blocked_moves(cheese):
    """(B, 9, H, W) bool: True where the destination is cheese. Staying is never blocked."""
    blocked = [torch.zeros_like(cheese)]
    for dy, dx in OFFSETS[1:]:
        blocked.append(look(cheese, dy, dx))
    return torch.stack(blocked, dim=1)


def step(policy, mass, notes, smell, blocked, fire):
    """Advance one step. fire is (B, 1, H, W): 1 where a square acts this step, 0 where it sits out."""
    logits, note_updates = policy(mass, smell, notes)
    fractions = torch.softmax(logits.masked_fill(blocked, float("-inf")), dim=1)

    # A square that sits out keeps all its mass and its notes.
    stay = torch.zeros_like(fractions)
    stay[:, 0] = 1.0
    fractions = fire * fractions + (1 - fire) * stay

    # Every square splits its mass across destinations, so total mass never changes.
    new_mass = torch.zeros_like(mass)
    for k, (dy, dx) in enumerate(OFFSETS):
        new_mass = new_mass + torch.roll(mass * fractions[:, k], shifts=(dy, dx), dims=(1, 2))
    return new_mass, notes + fire * note_updates


@dataclass
class Rollout:
    """Per-step measurements of a rollout. Every tensor is (T, B)."""

    surround: torch.Tensor  # fraction of the ring that counts as occupied, soft
    mean_distance: torch.Tensor  # mass-weighted average distance to the cheese
    crowding: torch.Tensor  # how far squares go over 1 unit of mass
    eaten: torch.Tensor  # bool, every ring square holds more than tau
    initial_mean_distance: torch.Tensor  # (B,)
    masses: list  # (B, H, W) numpy arrays per step, only when recorded


def damage(mass, fraction):
    """Wipe a random `fraction` of the squares that hold mass."""
    holds_mass = mass > 0.01
    wiped = holds_mass & (torch.rand_like(mass) < fraction)
    return mass.masked_fill(wiped, 0.0)


def rollout(
    policy,
    batch,
    n_steps,
    fire_prob,
    tau,
    use_checkpoint=False,
    record=False,
    damage_step=None,
    damage_fraction=0.25,
):
    mass = batch.mass
    b, h, w = mass.shape
    notes = torch.zeros(b, N_NOTES, h, w, device=mass.device)
    blocked = blocked_moves(batch.cheese)
    ring = batch.ring.float()
    ring_size = ring.sum(dim=(1, 2))
    total_mass = mass.sum(dim=(1, 2))
    initial_mean_distance = (mass * batch.distance).sum(dim=(1, 2)) / total_mass

    surround, mean_distance, crowding, eaten, masses = [], [], [], [], []
    for t in range(n_steps):
        if t == damage_step:
            mass = damage(mass, damage_fraction)
            total_mass = mass.sum(dim=(1, 2)).clamp_min(1e-6)
        fire =(torch.rand(b, 1, h, w, device=mass.device) < fire_prob).float()
        if use_checkpoint:
            mass, notes = checkpoint(step, policy, mass, notes, batch.smell, blocked, fire, use_reentrant=False)
        else:
            mass, notes = step(policy, mass, notes, batch.smell, blocked, fire)

        occupied = (mass / tau).clamp(0, 1)
        surround.append((occupied * ring).sum(dim=(1, 2)) / ring_size)
        mean_distance.append((mass * batch.distance).sum(dim=(1, 2)) / total_mass)
        crowding.append(F.relu(mass - 1).pow(2).sum(dim=(1, 2)) / total_mass)
        eaten.append(((mass > tau) | ~batch.ring).flatten(1).all(dim=1))
        if record:
            masses.append(mass.detach().cpu().numpy())

    return Rollout(
        surround=torch.stack(surround),
        mean_distance=torch.stack(mean_distance),
        crowding=torch.stack(crowding),
        eaten=torch.stack(eaten),
        initial_mean_distance=initial_mean_distance,
        masses=masses,
    )


def loss(result, w_surround, w_distance, w_crowding):
    """Weighted sum of the three terms, averaged over every step, so finishing early scores better.

    Distance is divided by its starting value so it begins near 1, like the other terms.
    """
    per_step = (
        w_surround * (1 - result.surround)
        + w_distance * result.mean_distance / result.initial_mean_distance
        + w_crowding * result.crowding
    )
    return per_step.mean()


def first_eaten_step(result):
    """(B,) step index (1-based) at which the cheese was first eaten, or 0 if never."""
    eaten = result.eaten
    ever = eaten.any(dim=0)
    first = eaten.float().argmax(dim=0) + 1
    return torch.where(ever, first, torch.zeros_like(first))
