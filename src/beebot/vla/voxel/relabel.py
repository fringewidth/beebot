"""Hindsight instructions for voxel play windows (voxel/tasks.py), as vla/relabel.py does in 2D:
per window, one record for each instruction family the window satisfies in full, holding one of
its satisfied instructions at random (eat logic: a form first, uniformly); eat-nothing labels are
thinned to their 2D share (vla/relabel.thin_idle).

    OMP_NUM_THREADS=4 uv run python -m beebot.vla.voxel.relabel --store data/vla/voxel_play_g16 --out data/vla/voxel_play_g16/labels.jsonl

Records: {"e", "w0", "w1", "family", "form", "text"}; texts are plain (no tags).
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from beebot.vla import relabel as R2
from beebot.vla.voxel import tasks as V
from beebot.vla.voxel.facts import store_facts
from beebot.vla.voxel.play_store import load
from beebot.vla.voxel.world import PALETTE_NAMES

WINDOWS = (24, 32, 48)
PER_EPISODE = 3
STORE = None


def window_scene(store, e, w0):
    colours = store["cheese_color"][e].tolist()
    alive = store["cheese_alive"][e, w0].tolist()
    return {"colours": [PALETTE_NAMES[c] for c, a in zip(colours, alive) if c >= 0 and a], "gravity": bool(store["gravity"][e])}


def label_episode(store, e, rng):
    records = []
    steps = store["mass"].shape[1] - 1
    for _ in range(PER_EPISODE):
        length = int(rng.choice(WINDOWS))
        w0 = int(rng.integers(0, steps - length + 1))
        w1 = w0 + length
        facts = store_facts(store, e, w0, w1)
        for family, ok in sorted(V.satisfied_options(window_scene(store, e, w0), facts, rng).items()):
            text, form = R2.pick("logic" if family == "eat" else family, ok, rng)
            records.append({"e": e, "w0": w0, "w1": w1, "family": family, "form": form, "text": text})
    return records


def _init(path):
    global STORE
    STORE = load(path)


def _work(args):
    start, stop, seed = args
    rng = np.random.default_rng(seed)
    return [r for e in range(start, stop) for r in label_episode(STORE, e, rng)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="data/vla/voxel_play_g16")
    parser.add_argument("--out", default="data/vla/voxel_play_g16/labels.jsonl")
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    n = args.episodes or len(load(args.store)["gravity"])
    chunks = [(s, min(s + 250, n), args.seed * 100003 + s) for s in range(0, n, 250)]
    records = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.store,)) as pool:
        for part in pool.map(_work, chunks):
            records += part
            print(f"{len(records)} labels", flush=True)
    records = R2.thin_idle(records, np.random.default_rng(args.seed))
    counts, forms = {}, {}
    with open(args.out, "w") as f:
        for r in records:
            counts[r["family"]] = counts.get(r["family"], 0) + 1
            if r["form"]:
                forms[r["form"]] = forms.get(r["form"], 0) + 1
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} labels to {args.out}: {counts}; eat forms {forms}")


if __name__ == "__main__":
    main()
