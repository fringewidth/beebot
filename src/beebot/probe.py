"""Robustness probes: conditions the model never trained on, outside the frozen val score.

    uv run probe runs/<name> [runs/<name> ...]

Each probe changes one thing about the world or the robots and reports the same
hard metric as the val set. They use their own seeds, so they never touch val or
test layouts. The baseline is scored alongside for reference.

"pheromone blackout" is the communication check: robots still write pheromones,
but every robot reads all zeros. If the score barely drops, the swarm isn't
using its pheromones to coordinate.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from beebot.ca import first_eaten_step, rollout
from beebot.evaluate import FIRE_PROB, STEPS_PER_CELL, TAU, WALLS
from beebot.layout import UNREACHABLE, Batch, grow_shape, sample_batch, sample_layout, swarm_mass
from beebot.model import NCA, StickyBaseline

SEED = 3000
LAYOUTS = 32
SIZES = (24, 32)
CHEESE = (1, 9)


class PheromoneBlackout:
    """Wraps a policy so every robot reads its pheromones as zeros."""

    def __init__(self, policy):
        self.policy = policy

    def bind(self, batch):
        # Pass each new batch on, or a privileged policy would smell the previous batch's scent.
        if hasattr(self.policy, "bind"):
            self.policy.bind(batch)

    def __call__(self, mass, cheese, walls, notes):
        return self.policy(mass, cheese, walls, torch.zeros_like(notes))


def chunk_damage(mass):
    """Wipe every square within distance 2 of a random square holding mass, per layout."""
    out = mass.clone()
    b, h, w = mass.shape
    for i in range(b):
        occupied = torch.nonzero(mass[i] > 0.01)
        if len(occupied) == 0:
            continue
        cy, cx = occupied[torch.randint(len(occupied), (1,)).item()].tolist()
        rows = [(cy + d) % h for d in range(-2, 3)]
        cols = [(cx + d) % w for d in range(-2, 3)]
        out[i][np.ix_(rows, cols)] = 0.0
    return out


def wipe(fraction):
    def damage(mass):
        holds_mass = mass > 0.01
        return mass.masked_fill(holds_mass & (torch.rand_like(mass) < fraction), 0.0)

    return damage


def split_nest_batch(rng, size, n_nests):
    """Layouts where the swarm starts as n_nests separate nests in different places."""
    cheeses, walls, masses, distances = [], [], [], []
    while len(cheeses) < LAYOUTS:
        cheese, wall, _, distance = sample_layout(rng, size, CHEESE, (2, size // 2), WALLS)
        reachable = distance < UNREACHABLE
        n_mass = swarm_mass(distance == 1, (reachable & ~cheese).sum())
        mass = np.zeros((size, size), dtype=np.float32)
        ok = True
        for k in range(n_nests):
            allowed = reachable & (distance >= 2) & (mass == 0)
            candidates = np.argwhere(allowed & (distance >= 4))
            if len(candidates) == 0:
                ok = False
                break
            seed = tuple(int(v) for v in candidates[rng.integers(len(candidates))])
            piece = n_mass // n_nests + (1 if k < n_mass % n_nests else 0)
            nest = grow_shape(rng, size, piece, seed, allowed)
            if nest is None:
                ok = False
                break
            mass[nest] = 1.0
        if ok:
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


def standard_batches(seed, sizes, cheese, walls=WALLS):
    rng = np.random.default_rng(seed)
    return [sample_batch(rng, s, cheese, (2, s // 2), LAYOUTS, walls) for s in sizes]


def score(policy, batches, device, fire_prob=FIRE_PROB, damage_fn=None):
    torch.manual_seed(SEED)
    successes, times = [], []
    for batch in batches:
        batch = batch.to(device)
        n_steps = STEPS_PER_CELL * batch.mass.shape[-1]
        with torch.no_grad():
            result = rollout(
                policy,
                batch,
                n_steps,
                fire_prob,
                TAU,
                damage_step=n_steps // 4 if damage_fn else None,
                damage_fn=damage_fn,
            )
        first = first_eaten_step(result).float()
        success = first > 0
        successes.append(success.float())
        times.append(torch.where(success, first / n_steps, torch.ones_like(first)))
    return {"success": torch.cat(successes).mean().item(), "time": torch.cat(times).mean().item()}


def probes(policy, device):
    plain = standard_batches(SEED, SIZES, CHEESE)
    rng = np.random.default_rng(SEED + 1)
    two_nests = [split_nest_batch(rng, s, 2) for s in SIZES]
    three_nests = [split_nest_batch(rng, s, 3) for s in SIZES]
    huge = standard_batches(SEED + 2, (128,), CHEESE)
    open_world = standard_batches(SEED + 3, SIZES, CHEESE, (0.0, 0.0))
    dense = standard_batches(SEED + 4, SIZES, CHEESE, (0.4, 0.45))
    return {
        "reference": score(policy, plain, device),
        "pheromone blackout": score(PheromoneBlackout(policy), plain, device),
        "firing 0.5": score(policy, plain, device, fire_prob=0.5),
        "firing 1.0": score(policy, plain, device, fire_prob=1.0),
        "wipe 50%": score(policy, plain, device, damage_fn=wipe(0.5)),
        "chunk cut": score(policy, plain, device, damage_fn=chunk_damage),
        "2 separate nests": score(policy, two_nests, device),
        "3 separate nests": score(policy, three_nests, device),
        "128x128 world": score(policy, huge, device),
        "no walls": score(policy, open_world, device),
        "walls 40-45% (beyond training)": score(policy, dense, device),
    }


def load(run_dir, device):
    config = json.loads((run_dir / "config.json").read_text())
    model = NCA(
        hidden=config["hidden"],
        n_hidden_layers=config["n_hidden_layers"],
        privileged=config.get("privileged", False),
    ).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()
    return model


def main():
    device = torch.device("mps")
    baseline_path = Path("runs/baseline/probes.json")
    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(probes(StickyBaseline(), device), indent=2))
    for arg in sys.argv[1:]:
        run_dir = Path(arg)
        results = probes(load(run_dir, device), device)
        (run_dir / "probes.json").write_text(json.dumps(results, indent=2))
        print(run_dir.name)
        for name, r in results.items():
            print(f"  {name:20s} success {r['success']:.2f}  time {r['time']:.3f}")


if __name__ == "__main__":
    main()
