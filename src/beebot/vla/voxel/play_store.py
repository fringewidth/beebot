"""The voxel play store on disk: scripted episodes written chunk by chunk, the per-step grids as
memory-mapped .npy files (as play_data.cache_play does in 2D), the rest in small.pt.

    OMP_NUM_THREADS=4 uv run python -m beebot.vla.voxel.play_store --out data/vla/voxel_play_g16 --episodes 12000

Each chunk of episodes comes from generate() with its own seed (seed * 1_000_003 + chunk index), so
any prefix of the store is a deterministic dataset of that size: the data-scaling runs read the
first N episodes. load() returns the same fields as generate(), the big ones memory-mapped.

The compact format stores the two grids by robot instead of by voxel (robots are sparse: 6 to 115
in 4096 voxels), about 40 times smaller at 16^3:

    compact.json  {"dims": [D, H, W], "voxel_dtype": "uint16", "stay": 26}; its presence marks the format
    offsets.npy   int64 (episodes, steps + 2): row e holds the bounds of episode e's steps + 1 frames
                  in the two files below, so frame (e, t) is entries offsets[e, t] to offsets[e, t + 1]
    voxels.bin    the flat voxel index (z H + y) W + x of every robot of every frame, ascending
                  within a frame (raw uint16; uint32 for grids over 65536 voxels)
    codes.bin     raw int8, parallel to voxels.bin: that robot's action code (the move it made, a
                  jittered one included); the last frame of an episode has no action and holds STAY
    intended.bin  optional (stores drawn from 2026-11-12 on), the same layout: the action code the
                  expert meant before any jitter; load() returns it as "intended_actions"

It needs a mass of 0 or 1 and STAY wherever no robot is (CompactWriter.add checks both), so the
dense grids come back exactly. load() detects it and returns "mass" and "actions" as CompactGrids,
which index like the dense arrays ([e, t] and slices of both) and read the three files
memory-mapped. demos.write(compact=True) writes it; an existing store is converted with

    uv run python -m beebot.vla.voxel.play_store --convert data/vla/voxel_demos_cube --out data/vla/voxel_demos_cube_compact
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel.scripted import episode_steps, generate
from beebot.vla.voxel.world import STAY

BIG_KEYS = ("mass", "actions")
MARKER = "compact.json"
SIDE_FILES = ("labels.jsonl", "attempts.jsonl")  # copied as they are by convert()


class DenseWriter:
    """Appends episodes to the dense store: mass.npy and actions.npy, allocated for `cap`
    episodes of `steps` steps and memory-mapped."""

    def __init__(self, out_dir, dims, cap, steps):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.mass = np.lib.format.open_memmap(out / "mass.npy", mode="w+", dtype=np.uint8, shape=(cap, steps + 1, *dims))
        self.actions = np.lib.format.open_memmap(out / "actions.npy", mode="w+", dtype=np.int8, shape=(cap, steps, *dims))
        self.row = 0

    def add(self, mass, actions, intended=None):  # intended is kept by the compact format only
        self.mass[self.row] = np.asarray(mass)
        self.actions[self.row] = np.asarray(actions)
        self.row += 1

    def close(self):
        self.mass.flush()
        self.actions.flush()


class CompactWriter:
    """Appends episodes to a compact store (the module docstring has the files) one at a time,
    so the number of episodes need not be known ahead; close() writes the index and the marker."""

    def __init__(self, out_dir, dims):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.dims = tuple(int(v) for v in dims)
        self.voxel_dtype = np.dtype(np.uint16 if int(np.prod(self.dims)) <= 1 << 16 else np.uint32)
        self.voxels = open(self.out / "voxels.bin", "wb")
        self.codes = open(self.out / "codes.bin", "wb")
        self.meant = None
        self.offsets = []
        self.total = 0

    def add(self, mass, actions, intended=None):
        """One episode: mass (steps + 1, D, H, W) of 0 and 1, actions (steps, D, H, W) with STAY
        off the robots (numpy arrays or tensors); intended: the same for the expert's intended
        codes (all episodes of a store give it or none)."""
        mass, actions = np.asarray(mass), np.asarray(actions)
        frames = mass.reshape(len(mass), -1)
        moves = np.concatenate([actions.reshape(len(actions), -1), np.full((1, frames.shape[1]), STAY, actions.dtype)])
        if frames.shape != moves.shape or frames.max(initial=0) > 1 or (moves[frames == 0] != STAY).any():
            raise ValueError("the compact store needs a mass of 0 or 1, one more mass frame than actions, and STAY where no robot is")
        t, voxel = np.nonzero(frames)
        self.voxels.write(voxel.astype(self.voxel_dtype).tobytes())
        self.codes.write(moves[t, voxel].astype(np.int8).tobytes())
        if intended is not None:
            if self.meant is None:
                self.meant = open(self.out / "intended.bin", "wb")
            meant = np.asarray(intended)
            meant = np.concatenate([meant.reshape(len(meant), -1), np.full((1, frames.shape[1]), STAY, meant.dtype)])
            self.meant.write(meant[t, voxel].astype(np.int8).tobytes())
        bounds = self.total + np.concatenate([[0], np.cumsum(np.bincount(t, minlength=len(frames)))])
        self.offsets.append(bounds)
        self.total = int(bounds[-1])

    def close(self):
        self.voxels.close()
        self.codes.close()
        if self.meant is not None:
            self.meant.close()
        steps = len(self.offsets[0]) - 2 if self.offsets else 0
        np.save(self.out / "offsets.npy", np.asarray(self.offsets, dtype=np.int64).reshape(len(self.offsets), steps + 2))
        (self.out / MARKER).write_text(json.dumps({"dims": list(self.dims), "voxel_dtype": self.voxel_dtype.name, "stay": STAY}))


class CompactGrids:
    """One dense field of a compact store, read on demand: "mass" (episodes, steps + 1, D, H, W)
    uint8 or "actions" (episodes, steps, D, H, W) int8. grids[e, t] is the (D, H, W) tensor the
    dense store holds there; e and t may also be slices (or left out), and further indices
    apply to the grid, as with the dense array. Only the frames asked for are read from the
    memory-mapped files."""

    def __init__(self, out_dir, field):
        out = Path(out_dir)
        marker = json.loads((out / MARKER).read_text())
        self.dims = tuple(marker["dims"])
        self.offsets = np.load(out / "offsets.npy", mmap_mode="r")
        self.voxels = self.raw(out / "voxels.bin", marker["voxel_dtype"])
        acts = field in ("actions", "intended_actions")
        self.codes = self.raw(out / ("intended.bin" if field == "intended_actions" else "codes.bin"), np.int8) if acts else None
        self.fill = marker["stay"] if acts else 0
        self.dtype = torch.int8 if acts else torch.uint8
        frames = self.offsets.shape[1] - 1 if len(self.offsets) else 0
        self.shape = (len(self.offsets), frames - 1 if acts and frames else frames, *self.dims)
        self.index = (np.arange(self.shape[0]), np.arange(self.shape[1]))

    @staticmethod
    def raw(path, dtype):
        """A raw file memory-mapped as a flat array (an empty file as an empty array)."""
        return np.memmap(path, dtype=dtype, mode="r") if path.stat().st_size else np.zeros(0, dtype)

    def __len__(self):
        return self.shape[0]

    def frame(self, e, t):
        """The dense (D H W,) numpy grid of frame (e, t)."""
        lo, hi = self.offsets[e, t], self.offsets[e, t + 1]
        grid = np.full(int(np.prod(self.dims)), self.fill, np.int8 if self.codes is not None else np.uint8)
        grid[self.voxels[lo:hi]] = 1 if self.codes is None else self.codes[lo:hi]
        return grid

    def __getitem__(self, index):
        index = index if isinstance(index, tuple) else (index,)
        lead = index[:2] + (slice(None),) * (2 - len(index[:2]))
        episodes, steps = self.index[0][lead[0]], self.index[1][lead[1]]
        dense = np.stack([self.frame(e, t) for e in episodes.reshape(-1) for t in steps.reshape(-1)]) if episodes.size and steps.size else np.zeros((0, int(np.prod(self.dims))), np.int8 if self.codes is not None else np.uint8)
        dense = dense.reshape(*episodes.shape, *steps.shape, *self.dims)
        rest = (slice(None),) * (episodes.ndim + steps.ndim) + index[2:]
        return torch.from_numpy(np.ascontiguousarray(dense[rest]))


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
    """The store's fields; "mass" and "actions" memory-mapped: tensors over the dense .npy files,
    or CompactGrids when the store is compact (it has compact.json)."""
    out = Path(out_dir)
    store = torch.load(out / "small.pt")
    is_compact = (out / MARKER).exists()
    for k in BIG_KEYS:
        store[k] = CompactGrids(out, k) if is_compact else torch.from_numpy(np.load(out / f"{k}.npy", mmap_mode="r"))
    if is_compact and (out / "intended.bin").exists():
        store["intended_actions"] = CompactGrids(out, "intended_actions")
    return store


def instructions_of(store, e, labels=None):
    """The instructions in force over episode e of a demonstration store: [{"t": first step,
    "text", "record"}] from small.pt's "instructions" (demos writes it; a switch episode has one
    entry per instruction). An older store has none: then the label rows of e (labels.jsonl, as
    dicts) give [{"t": w0, "text"}], and without labels the result is None."""
    if "instructions" in store:
        return store["instructions"][e]
    if labels is None:
        return None
    return [{"t": int(row["w0"]), "text": row["text"]} for row in sorted((r for r in labels if r["e"] == e), key=lambda r: r["w0"])]


def instruction_at(entries, t):
    """The entry of instructions_of in force at step t: the last one starting at or before t."""
    return [entry for entry in entries if entry["t"] <= t][-1]


def convert(src_dir, out_dir, chunk=64):
    """Write the store at src_dir (dense or compact) as a compact store at out_dir: the first
    "count" episodes (a demonstration store allocates more rows than it fills; otherwise all),
    small.pt and the label files as they are. The source is only read. Returns the number of
    episodes."""
    src, out = Path(src_dir), Path(out_dir)
    if out.resolve() == src.resolve() or (out / "small.pt").exists():
        raise ValueError(f"{out} already holds a store")
    store = load(src)
    count = int(store.get("count", len(store["gravity"])))
    writer = CompactWriter(out, store["mass"].shape[2:])
    for start in range(0, count, chunk):
        mass, actions = (np.asarray(store[k][start : min(start + chunk, count)]) for k in BIG_KEYS)
        for e in range(len(mass)):
            writer.add(mass[e], actions[e])
    writer.close()
    shutil.copyfile(src / "small.pt", out / "small.pt")
    for name in SIDE_FILES:
        if (src / name).exists():
            shutil.copyfile(src / name, out / name)
    (out / "done").write_text(str(count))
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--convert", default="", metavar="STORE", help="instead of generating play, write this existing store at --out in the compact format")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.convert:
        print(f"{convert(args.convert, args.out)} episodes written to {args.out}", flush=True)
    else:
        write(args.out, args.size, args.episodes, args.seed, args.chunk)


if __name__ == "__main__":
    main()
