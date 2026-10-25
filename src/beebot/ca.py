"""The cellular automaton: one step, a full rollout, and the loss.

A policy is anything callable as policy(mass, cheese, walls, notes) -> (logits, note_writes):
    mass:   (B, H, W)
    cheese: (B, H, W) float, 1 on cheese squares. A square can only sense it
            through its 3x3 window, so this is touch, not smell.
    walls:  (B, H, W) float, 1 on wall squares, sensed the same way
    notes:  (B, N_NOTES, H, W), pheromones on the floor
    logits:      (B, 9, H, W), one score per destination in OFFSETS
    note_writes: (B, N_NOTES, H, W), what a robot on that square writes

The notes are pheromones. They live on the floor, fade every step, and only
robots can write them: a square writes in proportion to the robot mass on it,
so empty floor can't compute or relay anything.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Destinations a square can send mass to, as (dy, dx). Index 0 is "stay".
# Rows grow downward, so (-1, 0) is north.
OFFSETS = [(0, 0), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
N_NOTES = 8
PHEROMONE_DECAY = 0.95  # fraction of every pheromone left after one step
# Mass at which a square holds a whole robot. A square with less is only part
# of a robot: it moves and writes in proportion, so a thin fog of mass is
# nearly stranded and nearly mute.
PRESENCE_MASS = 1.0
FOG_MASS = 0.5  # mass on squares holding less than this counts as fog in the diagnostics

# The cheese only counts as eaten once the whole swarm has come home: the ring is
# full and at least GATHER_SHARE of the mass sits within the gather radius, the
# smallest distance from the cheese whose squares could hold all the mass at
# GATHER_PACKING robots per square.
GATHER_SHARE = 0.8
GATHER_PACKING = 0.8


def look(field, dy, dx):
    """field[..., y + dy, x + dx], aligned so the result is indexed by (y, x)."""
    return torch.roll(field, shifts=(-dy, -dx), dims=(-2, -1))


def blocked_moves(obstacles):
    """(B, 9, H, W) bool: True where the destination is cheese or wall. Staying is never blocked."""
    blocked = [torch.zeros_like(obstacles)]
    for dy, dx in OFFSETS[1:]:
        blocked.append(look(obstacles, dy, dx))
    return torch.stack(blocked, dim=1)


def step(policy, mass, notes, cheese, walls, blocked, fire):
    """Advance one step. fire is (B, 1, H, W): 1 where a square acts this step, 0 where it sits out."""
    logits, note_writes = policy(mass, cheese, walls, notes)
    fractions = torch.softmax(logits.masked_fill(blocked, float("-inf")), dim=1)

    # A square acts in proportion to how much robot it holds, and only when it fires.
    # Whatever doesn't act stays put.
    presence = (mass / PRESENCE_MASS).clamp(0, 1)[:, None]
    acting = fire * presence
    stay = torch.zeros_like(fractions)
    stay[:, 0] = 1.0
    fractions = acting * fractions + (1 - acting) * stay

    # Every square splits its mass across destinations, so total mass never changes.
    new_mass = torch.zeros_like(mass)
    for k, (dy, dx) in enumerate(OFFSETS):
        new_mass = new_mass + torch.roll(mass * fractions[:, k], shifts=(dy, dx), dims=(1, 2))
    # Only robots write, in the same proportion. Writes are bounded by tanh.
    new_notes = PHEROMONE_DECAY * notes + acting * torch.tanh(note_writes)
    return new_mass, new_notes


@dataclass
class Rollout:
    """Per-step measurements of a rollout. Every tensor is (T, B)."""

    surround: torch.Tensor  # fraction of the ring that counts as occupied, soft
    mean_distance: torch.Tensor  # mass-weighted average distance to the cheese
    crowding: torch.Tensor  # how far squares go over 1 unit of mass
    gathered: torch.Tensor  # share of the mass within the gather radius
    eaten: torch.Tensor  # bool, every ring square holds more than tau and the swarm has gathered
    fog: torch.Tensor  # share of the mass on squares holding less than FOG_MASS
    exploring: torch.Tensor  # share of open floor not yet visited, while no robot has touched the cheese; 0 after
    initial_mean_distance: torch.Tensor  # (B,)
    masses: list  # (B, H, W) numpy arrays per step, only when recorded
    pheromones: list  # (B, H, W) numpy arrays per step, the strongest note per square, only when recorded


def gather_radius(distance, open_squares, total_mass):
    """(B,) smallest walking distance r such that the open squares within r can hold total_mass at GATHER_PACKING."""
    needed = torch.ceil(total_mass / GATHER_PACKING).long().clamp(min=1)
    ordered = distance.masked_fill(~open_squares, float("inf")).flatten(1).sort(dim=1).values
    return ordered.gather(1, (needed - 1).clamp(max=ordered.shape[1] - 1)[:, None])[:, 0]


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
    damage_fn=None,
):
    if hasattr(policy, "bind"):
        policy.bind(batch)
    mass = batch.mass
    b, h, w = mass.shape
    notes = torch.zeros(b, N_NOTES, h, w, device=mass.device)
    blocked = blocked_moves(batch.cheese | batch.walls)
    cheese = batch.cheese.float()
    walls = batch.walls.float()
    ring = batch.ring.float()
    ring_size = ring.sum(dim=(1, 2))
    total_mass = mass.sum(dim=(1, 2))
    initial_mean_distance = (mass * batch.distance).sum(dim=(1, 2)) / total_mass
    open_squares = ~batch.cheese & ~batch.walls
    radius = gather_radius(batch.distance, open_squares, total_mass)
    home = ((batch.distance <= radius[:, None, None]) & open_squares).float()

    # Exploration: a square counts as visited once it has held robot mass (soft, by
    # the same occupancy as the ring). Until a robot first touches the cheese, the
    # unvisited share is a training signal for searching; after contact it's zero.
    visited = torch.zeros_like(mass)
    contacted = torch.zeros(b, dtype=torch.bool, device=mass.device)
    open_count = open_squares.flatten(1).sum(dim=1)

    surround, mean_distance, crowding, gathered, eaten, fog, exploring = [], [], [], [], [], [], []
    masses, pheromones = [], []
    for t in range(n_steps):
        if t == damage_step:
            mass = damage_fn(mass) if damage_fn else damage(mass, damage_fraction)
            total_mass = mass.sum(dim=(1, 2)).clamp_min(1e-6)
        fire =(torch.rand(b, 1, h, w, device=mass.device) < fire_prob).float()
        if use_checkpoint:
            mass, notes = checkpoint(step, policy, mass, notes, cheese, walls, blocked, fire, use_reentrant=False)
        else:
            mass, notes = step(policy, mass, notes, cheese, walls, blocked, fire)

        occupied = (mass / tau).clamp(0, 1)
        surround.append((occupied * ring).sum(dim=(1, 2)) / ring_size)
        mean_distance.append((mass * batch.distance).sum(dim=(1, 2)) / total_mass)
        crowding.append(F.relu(mass - 1).pow(2).sum(dim=(1, 2)) / total_mass)
        gathered.append((mass * home).sum(dim=(1, 2)) / total_mass)
        ring_full = ((mass > tau) | ~batch.ring).flatten(1).all(dim=1)
        eaten.append(ring_full & (gathered[-1] >= GATHER_SHARE))
        fog.append((mass * (mass < FOG_MASS)).sum(dim=(1, 2)) / total_mass)
        visited = torch.maximum(visited, occupied * open_squares)
        unvisited = 1 - visited.sum(dim=(1, 2)) / open_count
        exploring.append(unvisited * (~contacted).float())
        contacted = contacted | ((mass > tau) & batch.ring).flatten(1).any(dim=1)
        if record:
            masses.append(mass.detach().cpu().numpy())
            pheromones.append(notes.detach().abs().amax(dim=1).cpu().numpy())

    return Rollout(
        surround=torch.stack(surround),
        mean_distance=torch.stack(mean_distance),
        crowding=torch.stack(crowding),
        gathered=torch.stack(gathered),
        eaten=torch.stack(eaten),
        fog=torch.stack(fog),
        exploring=torch.stack(exploring),
        initial_mean_distance=initial_mean_distance,
        masses=masses,
        pheromones=pheromones,
    )


def loss(result, w_surround, w_distance, w_crowding, w_gather, w_explore=0.0):
    """Weighted sum of the terms, averaged over every step, so finishing early scores better.

    w_explore rewards covering the floor until a robot first touches the cheese. A
    blind swarm needs it: on a torus with the cheese placed uniformly at random,
    every square has the same expected distance to the cheese, so the distance term
    carries no signal about how to search.

    Distance is divided by its starting value so it begins near 1, like the other terms.
    """
    per_step = (
        w_surround * (1 - result.surround)
        + w_distance * result.mean_distance / result.initial_mean_distance
        + w_crowding * result.crowding
        + w_gather * (1 - result.gathered)
        + w_explore * result.exploring
    )
    return per_step.mean()


def first_eaten_step(result):
    """(B,) step index (1-based) at which the cheese was first eaten, or 0 if never."""
    eaten = result.eaten
    ever = eaten.any(dim=0)
    first = eaten.float().argmax(dim=0) + 1
    return torch.where(ever, first, torch.zeros_like(first))
