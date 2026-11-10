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

Further options, all off unless given: --compact writes the grids by robot (play_store's compact
format, about 40 times smaller; load() reads both); --counts log draws the robot count
log-uniformly over --robots, so small swarms are as common as large ones; --fit draws only tasks
the scene's swarm can do (suite.fits) also when --robots is not given.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel import suite as S
from beebot.vla.voxel.play_store import CompactWriter, DenseWriter
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


def write(out_dir, n_scenes, seed=0, chunk=24, steps=S.EPISODE_STEPS, moves="all", robots=None, failure=None, compact=False, counts="uniform", fit=False):
    """Write the store. moves: "all" or "cube" (MOVES). robots: (low, high) robot counts, with
    tasks fitted to the swarm (None: the world's density range). failure: (27,) per-action
    failure probabilities (world.failure_rates) or None. compact: write mass and actions in
    play_store's compact format instead of the dense .npy files. counts: how a scene's robot
    count is drawn from `robots` ("uniform" or "log", suite.draw_count). fit: draw only tasks
    the swarm can do also when robots is None (with robots that is always so). Returns the
    number of demonstrations."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dims = (S.SIZE,) * 3
    writer = CompactWriter(out, dims) if compact else DenseWriter(out, dims, n_scenes * len(S.FAMILIES), steps)
    check = fit or robots is not None
    small = {k: [] for k in ("cheese_pos", "cheese_alive", "walls", "cheese_color", "gravity")}
    subgoals, swarm = [], []
    labels = open(out / "labels.jsonl", "w")
    attempts = open(out / "attempts.jsonl", "w")
    row, tried, kept = 0, {}, {}
    t0 = time.time()
    for c in range(0, n_scenes, chunk):
        world, metas, tasks = S.sample_tasks(rng, min(chunk, n_scenes - c), "train", robots=robots, check=check, counts=counts)
        world, metas, tasks = contrast(rng, world, metas, tasks, check=check)
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
            writer.add(arrays["mass"][i].numpy(), arrays["actions"][i].numpy())
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
    writer.close()
    config = {"seed": seed, "steps": steps, "moves": moves, "robots": None if robots is None else list(robots), "failure": None if failure is None else failure.tolist(), "counts": counts, "fit": check}
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
    parser.add_argument("--compact", action="store_true", help="store mass and actions by robot (play_store's compact format) instead of dense grids")
    parser.add_argument("--counts", choices=S.COUNTS, default="uniform", help="how the robot count is drawn over --robots; log makes small swarms as common as large ones")
    parser.add_argument("--fit", action="store_true", help="draw only tasks the scene's swarm can do, also without --robots (with --robots or --cube this is always on)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    steps = args.steps or (S.CUBE_STEPS if args.cube else S.EPISODE_STEPS)
    moves = args.moves or ("cube" if args.cube else "all")
    robots = args.robots or (S.ROBOTS if args.cube else None)
    failure = parse_failure(args.failure) if args.failure else failure_rates() if args.cube else None
    write(args.out, args.scenes, args.seed, args.chunk, steps, moves, robots, failure, args.compact, args.counts, args.fit)


if __name__ == "__main__":
    main()
