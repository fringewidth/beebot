"""The voxel play store on disk: scripted episodes written chunk by chunk, the per-step grids as
memory-mapped .npy files (as play_data.cache_play does in 2D), the rest in small.pt.

    OMP_NUM_THREADS=4 uv run python -m beebot.vla.voxel.play_store --out data/vla/voxel_play_g16 --episodes 12000

Each chunk of episodes comes from generate() with its own seed (seed * 1_000_003 + chunk index), so
any prefix of the store is a deterministic dataset of that size: the data-scaling runs read the
first N episodes. load() returns the same fields as generate(), the big ones memory-mapped.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel.scripted import episode_steps, generate

BIG_KEYS = ("mass", "actions")


def write(out_dir, size=16, n_episodes=12000, seed=0, chunk=512):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    steps = episode_steps(size)
    big = {
        "mass": np.lib.format.open_memmap(out / "mass.npy", mode="w+", dtype=np.uint8, shape=(n_episodes, steps + 1, size, size, size)),
        "actions": np.lib.format.open_memmap(out / "actions.npy", mode="w+", dtype=np.int8, shape=(n_episodes, steps, size, size, size)),
    }
    small = {k: [] for k in ("cheese_pos", "cheese_alive", "walls", "cheese_color", "gravity")}
    phases = []
    t0 = time.time()
    for i, start in enumerate(range(0, n_episodes, chunk)):
        n = min(chunk, n_episodes - start)
        part = generate(size=size, n_episodes=n, seed=seed * 1_000_003 + i)
        for k in BIG_KEYS:
            big[k][start : start + n] = part[k].numpy()
        for k in small:
            small[k].append(part[k])
        phases += part["phases"]
        print(f"{start + n}/{n_episodes} episodes, {time.time() - t0:.0f} s", flush=True)
    for array in big.values():
        array.flush()
    torch.save({**{k: torch.cat(v) for k, v in small.items()}, "phases": phases, "size": size, "seed": seed, "chunk": chunk}, out / "small.pt")
    (out / "done").write_text(str(n_episodes))


def load(out_dir):
    out = Path(out_dir)
    store = torch.load(out / "small.pt")
    for k in BIG_KEYS:
        store[k] = torch.from_numpy(np.load(out / f"{k}.npy", mmap_mode="r"))
    return store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    write(args.out, args.size, args.episodes, args.seed, args.chunk)


if __name__ == "__main__":
    main()
