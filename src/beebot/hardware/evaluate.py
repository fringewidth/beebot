"""The one metric for a module design: its work score.

Work score = the mean, over a fixed sample of the voxel generalist suite's training tasks
(vla/voxel/suite.py: formation, construction, logic, parallel, herd, runner), of the episode's
score, counted as 0 when the swarm ate a cheese it was told not to. The swarm is driven by the
suite's expert (the same plans for every design, restricted to the moves the design can make) and
the world runs under the design's physics (hardware/physics.py) with gravity always on. So the
number answers: with a competent operator, how much of the generalist task mix can a swarm of
these modules carry out? It is in [0, 1]; higher is better.

The one change to the expert's plans: terrain does not stick to modules, so a bridge flush with
the trench rims cannot hang from them. The expert also lays a deck one layer up, from two voxels
before the trench to two after, so the bridge rests on both rims. Every design (and the
permissive voxel-world baseline) gets the same plan.

    uv run python -m beebot.hardware.evaluate --side 0.03 --connector magnet --connector-share 0.05 --actuator-share 0.1
"""

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import torch

from beebot.hardware import physics
from beebot.hardware.design import Design, describe
from beebot.vla.voxel import controller
from beebot.vla.voxel import suite as S
from beebot.vla.voxel.world import COLOUR_NAMES, step as voxel_step

N_TASKS = 192
SEED = 0
DECK_REACH = 2


def anchor_bridge(task, meta):
    """Add the resting deck to a bridge task's plan."""
    if task.spec["kind"] != "bridge" or not task.plan:
        return task
    axis, start, width, top = S.trench(meta)
    if top >= S.SIZE:
        return task
    flush = task.plan[0]["points"]
    other = 3 - axis
    lanes = sorted({p[other] for p in flush})
    deck = []
    for lane in lanes:
        for a in range(max(start - DECK_REACH, 0), min(start + width + DECK_REACH, S.SIZE)):
            p = [top, 0, 0]
            p[axis], p[other] = a, lane
            deck.append(tuple(p))
    return replace(task, plan=[{"kind": "targets", "points": list(flush) + deck}])


def task_sample(n=N_TASKS, seed=SEED):
    """(world, metas, tasks): the fixed task sample, with gravity on everywhere."""
    rng = np.random.default_rng(seed)
    world, metas, tasks = S.sample_tasks(rng, n, "train")
    world = replace(world, gravity=torch.ones_like(world.gravity))
    return world, metas, [anchor_bridge(t, m) for t, m in zip(tasks, metas)]


@contextmanager
def limited_expert(cap):
    """The suite's expert, choosing only among moves the design can make."""
    original = S.act

    def act(world, *args, **kwargs):
        return controller.act(world, *args, allowed=physics.allowed_moves(world, cap), **kwargs)

    S.act = act
    try:
        yield
    finally:
        S.act = original


def rollout(world, metas, tasks, cap, steps=S.EPISODE_STEPS, seed=SEED):
    """Run the expert on every task under the design's physics (cap None: the voxel world's own
    rules). Returns per task (score, unasked)."""
    generator = torch.Generator().manual_seed(seed)
    runners = torch.zeros(world.cheese_alive.shape, dtype=torch.bool)
    for i, task in enumerate(tasks):
        runners[i, list(task.runners)] = True
    start = world
    eaten = [[] for _ in tasks]
    expert = S.Expert(tasks)
    memory = [{} for _ in tasks]
    for t in range(steps):
        before = world.cheese_alive.clone()
        score = torch.rand(world.mass.shape, generator=generator)
        if cap is None:
            codes = expert(world, t)
            world, _ = voxel_step(world, codes.long(), score)
        else:
            with limited_expert(cap):
                codes = expert(world, t)
            world, _ = physics.step(world, codes.long(), cap, score, memory)
        if runners.any() and (t + 1) % S.RUNNER_PERIOD == 0:
            world = replace(world, cheese_pos=S.flee(world, runners))
        for i, k in (before & ~world.cheese_alive).nonzero().tolist():
            eaten[i].append(COLOUR_NAMES[int(world.cheese_color[i, k])])
    results = []
    for i, task in enumerate(tasks):
        static = {"walls": start.walls[i].numpy(), "cheese_color": start.cheese_color[i].tolist(), "gravity": True, "meta": metas[i]}
        results.append(S.score(task, static, S.snapshot(start, i), S.snapshot(world, i), eaten[i]))
    return results


def work_score(results, tasks):
    """(work score, per family) from rollout results."""
    values = np.array([0.0 if unasked else s for s, unasked in results])
    families = {}
    for f in S.FAMILIES:
        idx = [i for i, t in enumerate(tasks) if t.family == f]
        if idx:
            families[f] = round(float(values[idx].mean()), 3)
    return float(values.mean()), families


def evaluate(cap, n=N_TASKS, seed=SEED, batch=64):
    torch.manual_seed(seed)  # the controller's jitter draws from the global generator
    world, metas, tasks = task_sample(n, seed)
    results = []
    for lo in range(0, n, batch):
        idx = list(range(lo, min(lo + batch, n)))
        sub = world.index(torch.tensor(idx))
        results += rollout(sub, [metas[i] for i in idx], [tasks[i] for i in idx], cap, seed=seed + lo)
    return work_score(results, tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--side", type=float, default=0.03)
    parser.add_argument("--connector", default="magnet", choices=("magnet", "epm"))
    parser.add_argument("--connector-share", type=float, default=0.05)
    parser.add_argument("--actuator-share", type=float, default=0.1)
    parser.add_argument("--flight-share", type=float, default=0.0)
    parser.add_argument("--voxel", action="store_true", help="the voxel world's own rules (no hardware limits)")
    parser.add_argument("--tasks", type=int, default=N_TASKS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    started = time.time()
    if args.voxel:
        row = {"design": "voxel-world"}
        cap = None
    else:
        design = Design(args.side, args.connector, args.connector_share, args.actuator_share, args.flight_share)
        row = describe(design)
        cap = design.capability()
    score, families = evaluate(cap, args.tasks, args.seed)
    row.update({"work": round(score, 4), "families": families, "tasks": args.tasks, "seconds": round(time.time() - started)})
    print(json.dumps(row))


if __name__ == "__main__":
    main()
