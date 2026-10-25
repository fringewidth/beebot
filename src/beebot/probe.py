"""Robustness probes: conditions the model never trained on, outside the frozen val score.

    uv run probe runs/<name> [runs/<name> ...]

Each probe changes one thing about the world or the robots and reports the same
hard metric as the val set. They use their own seeds, so they never touch val or
test layouts. The baseline is scored alongside for reference.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from beebot.ca import first_eaten_step, rollout
from beebot.evaluate import FIRE_PROB, SMELL_LENGTH, STEPS_PER_CELL, TAU
from beebot.layout import Batch, MASS_PER_RING_SQUARE, grow_shape, sample_batch, sample_layout, torus_distance
from beebot.model import NCA, GreedyBaseline

SEED = 3000
LAYOUTS = 32
SIZES = (24, 32)
CHEESE = (1, 9)


class NoisySmell:
    """Wraps a policy so every square reads the smell with fresh multiplicative noise each step."""

    def __init__(self, policy, noise):
        self.policy = policy
        self.noise = noise

    def __call__(self, mass, smell, notes):
        noisy = smell * (1 + self.noise * torch.randn_like(smell))
        return self.policy(mass, noisy.clamp(0, 1), notes)


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


def split_blob_batch(rng, size, n_blobs):
    """Layouts where the mass starts as n_blobs separate pieces in different places."""
    cheeses, masses, distances = [], [], []
    while len(cheeses) < LAYOUTS:
        cheese, _, distance = sample_layout(rng, size, CHEESE, (2, size // 2))
        n_mass = math.ceil(MASS_PER_RING_SQUARE * (distance == 1).sum())
        mass = np.zeros((size, size), dtype=np.float32)
        ok = True
        for k in range(n_blobs):
            allowed = (distance >= 2) & (mass == 0)
            candidates = np.argwhere(allowed & (distance >= 4))
            if len(candidates) == 0:
                ok = False
                break
            seed = tuple(int(v) for v in candidates[rng.integers(len(candidates))])
            piece = n_mass // n_blobs + (1 if k < n_mass % n_blobs else 0)
            blob = grow_shape(rng, size, piece, seed, allowed)
            if blob is None:
                ok = False
                break
            mass[blob] = 1.0
        if ok:
            cheeses.append(cheese)
            masses.append(mass)
            distances.append(distance)
    distance = torch.tensor(np.stack(distances), dtype=torch.float32)
    return Batch(
        mass=torch.tensor(np.stack(masses)),
        smell=torch.exp(-distance / SMELL_LENGTH),
        cheese=torch.tensor(np.stack(cheeses)),
        ring=distance == 1,
        distance=distance,
    )


def standard_batches(seed, sizes, cheese):
    rng = np.random.default_rng(seed)
    return [sample_batch(rng, s, cheese, (2, s // 2), LAYOUTS, SMELL_LENGTH) for s in sizes]


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
    two_blobs = [split_blob_batch(rng, s, 2) for s in SIZES]
    three_blobs = [split_blob_batch(rng, s, 3) for s in SIZES]
    huge = standard_batches(SEED + 2, (128,), CHEESE)
    return {
        "reference": score(policy, plain, device),
        "firing 0.5": score(policy, plain, device, fire_prob=0.5),
        "firing 1.0": score(policy, plain, device, fire_prob=1.0),
        "wipe 50%": score(policy, plain, device, damage_fn=wipe(0.5)),
        "chunk cut": score(policy, plain, device, damage_fn=chunk_damage),
        "noisy smell 20%": score(NoisySmell(policy, 0.2), plain, device),
        "noisy smell 50%": score(NoisySmell(policy, 0.5), plain, device),
        "2 separate blobs": score(policy, two_blobs, device),
        "3 separate blobs": score(policy, three_blobs, device),
        "128x128 world": score(policy, huge, device),
    }


def load(run_dir, device):
    config = json.loads((run_dir / "config.json").read_text())
    model = NCA(
        hidden=config["hidden"],
        n_hidden_layers=config["n_hidden_layers"],
        fold_change=config.get("fold_change", False),
    ).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()
    return model


def main():
    device = torch.device("mps")
    baseline_path = Path("runs/baseline/probes.json")
    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(probes(GreedyBaseline(SMELL_LENGTH), device), indent=2))
    for arg in sys.argv[1:]:
        run_dir = Path(arg)
        results = probes(load(run_dir, device), device)
        (run_dir / "probes.json").write_text(json.dumps(results, indent=2))
        print(run_dir.name)
        for name, r in results.items():
            print(f"  {name:18s} success {r['success']:.2f}  time {r['time']:.3f}")


if __name__ == "__main__":
    main()
