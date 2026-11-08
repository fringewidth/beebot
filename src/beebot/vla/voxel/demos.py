"""Expert demonstrations of the voxel suite (voxel/suite.py), written as a store with labels for
voxel/lm.py: each scene is played under several training instructions from the same start (one
per family the scene allows), by the expert, and an episode is kept only if the checker says the
expert did what the instruction asks (and ate nothing it forbids). The label is the instruction.

    OMP_NUM_THREADS=4 uv run python -m beebot.vla.voxel.demos --out data/vla/voxel_demos_g16 --scenes 6000

Store fields as voxel/play_store (mass, actions memory-mapped; small.pt with the rest); labels.jsonl
records {"e", "w0", "w1", "family", "form", "text"} covering whole episodes.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel import suite as S
from beebot.vla.voxel.world import VoxelWorld

SUCCESS = {"formation": 0.5}  # checker score an episode needs to count as a demonstration (others 0.75)
FIELDS = ("mass", "walls", "cheese_pos", "cheese_alive", "cheese_color", "gravity")


def contrast(rng, world, metas, tasks):
    """Each scene's own task plus one training task of every other family its scene allows."""
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
            if other is not None and other.plan or (other is not None and other.spec.get("form") in ("none", "absent")):
                worlds.append(w)
                out_metas.append(metas[i])
                out.append(other)
    return VoxelWorld(*(torch.cat([getattr(x, f) for x in worlds]) for f in FIELDS)), out_metas, out


def write(out_dir, n_scenes, seed=0, chunk=24, steps=S.EPISODE_STEPS):
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
    labels = open(out / "labels.jsonl", "w")
    row, tried, kept = 0, {}, {}
    t0 = time.time()
    for c in range(0, n_scenes, chunk):
        world, metas, tasks = S.sample_tasks(rng, min(chunk, n_scenes - c), "train")
        world, metas, tasks = contrast(rng, world, metas, tasks)
        results, arrays = S.run(world, metas, tasks, S.Expert(tasks), steps=steps, seed=seed * 1_000_003 + c, record=True)
        for i, ((score, unasked), task) in enumerate(zip(results, tasks)):
            tried[task.family] = tried.get(task.family, 0) + 1
            if unasked or score < SUCCESS.get(task.family, 0.75):
                continue
            kept[task.family] = kept.get(task.family, 0) + 1
            big["mass"][row] = arrays["mass"][i].numpy()
            big["actions"][row] = arrays["actions"][i].numpy()
            small["cheese_pos"].append(arrays["cheese_pos"][i])
            small["cheese_alive"].append(arrays["cheese_alive"][i])
            small["walls"].append(world.walls[i])
            small["cheese_color"].append(world.cheese_color[i])
            small["gravity"].append(world.gravity[i])
            labels.write(json.dumps({"e": row, "w0": 0, "w1": steps, "family": task.family, "form": task.spec.get("form", ""), "text": task.text}) + "\n")
            row += 1
        labels.flush()
        print(f"{c + chunk} scenes, {row} demos, {time.time() - t0:.0f} s; kept/tried {json.dumps({f: f'{kept.get(f, 0)}/{tried[f]}' for f in sorted(tried)})}", flush=True)
    for array in big.values():
        array.flush()
    torch.save({**{k: torch.stack(v) for k, v in small.items()}, "phases": [[] for _ in range(row)], "count": row}, out / "small.pt")
    (out / "done").write_text(str(row))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenes", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    write(args.out, args.scenes, args.seed, args.chunk)


if __name__ == "__main__":
    main()
