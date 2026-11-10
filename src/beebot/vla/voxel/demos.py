"""Expert demonstrations of the voxel suite (voxel/suite.py), written as a store with labels for
voxel/lm.py: each scene is played under several training instructions from the same start (one
per family the scene allows), by the expert, and an episode is kept only if the checker says the
expert did what the instruction asks (and ate nothing it forbids). The label is the instruction.

    OMP_NUM_THREADS=4 uv run python -m beebot.vla.voxel.demos --out data/vla/voxel_demos_g16 --scenes 6000

Store fields as voxel/play_store (mass, actions memory-mapped; small.pt with the rest); labels.jsonl
records {"e", "w0", "w1", "family", "form", "text"} covering whole episodes. small.pt also holds,
per kept episode, "robots" (the swarm's size) and "subgoals" (what the expert was after at every
step: Expert.subgoals), and "config" (the options below). attempts.jsonl has one row per episode
played, kept or not: {"family", "kind", "robots", "score", "unasked", "e"} with e = -1 when dropped.

Demonstrations a pivoting cube can carry out (--cube): only the 14 moves without a horizontal
diagonal (--moves cube), 72 steps (--steps), swarms of 6 robots up to the usual maximum with only
tasks that size can do (--robots 6 115), and moves that fail with probability 0.15 (--failure;
per move class as in --failure 0.15,up_over=0.3). The recorded action is always the expert's
choice, whether or not the move happened.

    OMP_NUM_THREADS=4 uv run python -m beebot.vla.voxel.demos --out data/vla/voxel_demos_cube --scenes 6000 --cube
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel import suite as S
from beebot.vla.voxel.world import CUBE_MOVES, FAILURE_RATE, VoxelWorld, failure_rates

SUCCESS = S.SUCCESS  # checker score an episode needs to count as a demonstration
FIELDS = ("mass", "walls", "cheese_pos", "cheese_alive", "cheese_color", "gravity")
MOVES = {"all": None, "cube": CUBE_MOVES}


def contrast(rng, world, metas, tasks, check=False):
    """Each scene's own task plus one training task of every other family its scene allows
    (check: and its swarm is large enough for, suite.fits)."""
    worlds, out_metas, out = [], [], []
    for i, task in enumerate(tasks):
        w = world.index(i)
        worlds.append(w)
        out_metas.append(metas[i])
        out.append(task)
        for family in S.FAMILIES:
            if family == task.family:
                continue
            _, make = S.MAKERS[family](rng, "")
            other = make(w, metas[i], 0)
            if other is not None and check and not S.fits(other, w, 0):
                continue
            if other is not None and other.plan or (other is not None and other.spec.get("form") in ("none", "absent")):
                worlds.append(w)
                out_metas.append(metas[i])
                out.append(other)
    return VoxelWorld(*(torch.cat([getattr(x, f) for x in worlds]) for f in FIELDS)), out_metas, out


def write(out_dir, n_scenes, seed=0, chunk=24, steps=S.EPISODE_STEPS, moves="all", robots=None, failure=None):
    """Write the store. moves: "all" or "cube" (MOVES). robots: (low, high) robot counts, with
    tasks fitted to the swarm (None: the world's density range). failure: (27,) per-action
    failure probabilities (world.failure_rates) or None. Returns the number of demonstrations."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cap = n_scenes * len(S.FAMILIES)
    size = S.SIZE
    big = {
        "mass": np.lib.format.open_memmap(out / "mass.npy", mode="w+", dtype=np.uint8, shape=(cap, steps + 1, size, size, size)),
        "actions": np.lib.format.open_memmap(out / "actions.npy", mode="w+", dtype=np.int8, shape=(cap, steps, size, size, size)),
    }
    small = {k: [] for k in ("cheese_pos", "cheese_alive", "walls", "cheese_color", "gravity")}
    subgoals, swarm = [], []
    labels = open(out / "labels.jsonl", "w")
    attempts = open(out / "attempts.jsonl", "w")
    row, tried, kept = 0, {}, {}
    t0 = time.time()
    for c in range(0, n_scenes, chunk):
        world, metas, tasks = S.sample_tasks(rng, min(chunk, n_scenes - c), "train", robots=robots)
        world, metas, tasks = contrast(rng, world, metas, tasks, check=robots is not None)
        run_seed = seed * 1_000_003 + c
        expert = S.Expert(tasks, moves=MOVES[moves], seed=run_seed)
        results, arrays = S.run(world, metas, tasks, expert, steps=steps, seed=run_seed, record=True, failure=failure)
        for i, ((score, unasked), task) in enumerate(zip(results, tasks)):
            tried[task.family] = tried.get(task.family, 0) + 1
            is_demo = not unasked and score >= SUCCESS.get(task.family, 0.75)
            n_robots = int(world.mass[i].sum())
            attempts.write(json.dumps({"family": task.family, "kind": S.task_kind(task), "robots": n_robots, "score": round(float(score), 4), "unasked": bool(unasked), "e": row if is_demo else -1}) + "\n")
            if not is_demo:
                continue
            kept[task.family] = kept.get(task.family, 0) + 1
            big["mass"][row] = arrays["mass"][i].numpy()
            big["actions"][row] = arrays["actions"][i].numpy()
            small["cheese_pos"].append(arrays["cheese_pos"][i])
            small["cheese_alive"].append(arrays["cheese_alive"][i])
            small["walls"].append(world.walls[i])
            small["cheese_color"].append(world.cheese_color[i])
            small["gravity"].append(world.gravity[i])
            subgoals.append(expert.subgoals[i])
            swarm.append(n_robots)
            labels.write(json.dumps({"e": row, "w0": 0, "w1": steps, "family": task.family, "form": task.spec.get("form", ""), "text": task.text}) + "\n")
            row += 1
        labels.flush()
        attempts.flush()
        print(f"{min(c + chunk, n_scenes)} scenes, {row} demos, {time.time() - t0:.0f} s; kept/tried {json.dumps({f: f'{kept.get(f, 0)}/{tried[f]}' for f in sorted(tried)})}", flush=True)
    for array in big.values():
        array.flush()
    config = {"seed": seed, "steps": steps, "moves": moves, "robots": None if robots is None else list(robots), "failure": None if failure is None else failure.tolist()}
    torch.save({**{k: torch.stack(v) for k, v in small.items()}, "phases": [[] for _ in range(row)], "count": row, "robots": torch.tensor(swarm), "subgoals": subgoals, "config": config}, out / "small.pt")
    (out / "done").write_text(str(row))
    return row


def parse_failure(text):
    """(27,) failure probabilities from "0.15" or "0.15,up_over=0.3,vertical=0.2": the rate of
    every move, then rates of single move classes (world.MOVE_CLASSES)."""
    first, *rest = text.split(",")
    by_class = {name: float(value) for name, value in (part.split("=") for part in rest)}
    return failure_rates(float(first), **by_class)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenes", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cube", action="store_true", help=f"pivoting-cube demonstrations: --moves cube --steps {S.CUBE_STEPS} --robots {S.ROBOTS[0]} {S.ROBOTS[1]} --failure {FAILURE_RATE}, each unless given")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--moves", choices=sorted(MOVES), default=None)
    parser.add_argument("--robots", type=int, nargs=2, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--failure", default=None, help="move failure probability, optionally per class: 0.15,up_over=0.3")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    steps = args.steps or (S.CUBE_STEPS if args.cube else S.EPISODE_STEPS)
    moves = args.moves or ("cube" if args.cube else "all")
    robots = args.robots or (S.ROBOTS if args.cube else None)
    failure = parse_failure(args.failure) if args.failure else failure_rates() if args.cube else None
    write(args.out, args.scenes, args.seed, args.chunk, steps, moves, robots, failure)


if __name__ == "__main__":
    main()
