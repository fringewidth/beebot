"""Held-out-seed gap of the hinge table (mujoco_hinge_v1): per ten-cube walk seed, fit on the stores
and the other seeds' walks (fit --legal: moves the grid's rule allows), play the held-out seed's
moves on the grid (cube_outcomes.grid_outcomes) and compare harm per move, reached and the outcome
TV with MuJoCo's. "all": every held-out move; "planned": the moves the grid's rule allows in a scene
the grid holds up (cube_swarm's walks drew moves that leave a cube held up only by the mover, and
some walk scenes have a cube MuJoCo holds that the grid's face-support rule lets fall), the moves
the grid's experts can make. Also the stores-only fit, as for v3. Usage: holdout_gap.py [runs]"""
import json
import sys
from collections import Counter

import numpy as np
import torch

from beebot.hardware.cube_outcomes import LATTICE, fit, grid_outcomes, grid_world, lattice_end, legal
from beebot.vla.voxel.world import OutcomeModel, supported

torch.set_num_threads(4)
base = "runs/cube_redesign/"


def stable(row):
    world = grid_world(row["cells"])
    occupied = world.mass > 0.5
    return bool((supported(world, occupied) == occupied).all())


stores = [json.loads(line) for line in open(base + "stores-hinge-e03.jsonl")]
seeds = {s: [json.loads(line) for line in open(base + f"layouts-hinge-e03{suffix}.jsonl")] for s, suffix in ((0, ""), (1000, "-s1000"), (2000, "-s2000"))}
allowed = {id(r): legal(r) for part in [stores, *seeds.values()] for r in part}
runs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
for held, rows in seeds.items():
    planned = [r for r in rows if allowed[id(r)] and stable(r)]
    for label, train in (("stores only", stores), ("stores + other walks", stores + [r for s, part in seeds.items() if s != held for r in part])):
        model = OutcomeModel(json.loads(json.dumps(fit([r for r in train if allowed[id(r)]], "holdout", "release"))))
        for which, part in (("all", rows), ("planned", planned)):
            grid = grid_outcomes(part, model, runs=runs, seed=held).mean(0)
            c = Counter(lattice_end(r)[0] for r in part)
            mujoco = np.array([c[d] / len(part) for d in LATTICE])
            print(f"seed {held:4d} {label:22s} {which:8s} n {len(part)}  harm mujoco {1 - mujoco[0] - mujoco[1]:.3f} grid {1 - grid[0] - grid[1]:.3f}  reached mujoco {mujoco[0]:.3f} grid {grid[0]:.3f}  TV {0.5 * np.abs(mujoco - grid).sum():.3f}", flush=True)
