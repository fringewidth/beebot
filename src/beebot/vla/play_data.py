"""Load saved play (LLM program runs, later teleop) into the same store format as scripted play.

Each play-run directory holds episodes.npz, program.py and meta.json (see play_cli). The store
gets one row per scene, plus "source": (program id, player, scene index) per row, so captions and
audits can point back to the run that made a window, and "episode_weight": sessions drift toward
re-running their own templates, so programs whose code is nearly the same (token Jaccard at least
NEAR_DUPLICATE) form one group, and a group's episodes share the sampling weight of one program.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from beebot.vla.data import encode_field
from beebot.vla.world import PHYSICS

PLAY_ROOT = Path("/Users/hrishik/beebot-play")
NEAR_DUPLICATE = 0.9


def program_weights(sources):
    """{program text: weight}: 1 / size of its near-duplicate group (connected components of the
    token-Jaccard >= NEAR_DUPLICATE graph)."""
    texts = list(dict.fromkeys(sources))
    tokens = [set(re.findall(r"[A-Za-z_]+|\d+", t)) for t in texts]
    parent = list(range(len(texts)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(texts)):
        for j in range(i):
            if len(tokens[i] & tokens[j]) >= NEAR_DUPLICATE * len(tokens[i] | tokens[j]):
                parent[root(i)] = root(j)
    sizes = {}
    for i in range(len(texts)):
        sizes[root(i)] = sizes.get(root(i), 0) + 1
    return {t: 1.0 / sizes[root(i)] for i, t in enumerate(texts)}


def run_dirs(root=PLAY_ROOT, size=None, physics=PHYSICS, cutoff=0.0):
    """Every play-run directory under root/<player>/data recorded with the given physics (runs from
    before the switch to discrete robots have no "physics" key and count as continuous), optionally
    only one grid size, and (cutoff > 0, a Unix time) only runs recorded before cutoff, so
    evaluations at different times can see the same data while play keeps running."""
    found = []
    for meta_path in sorted(Path(root).glob("*/data/*/meta.json")):
        if cutoff > 0 and meta_path.stat().st_mtime > cutoff:
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("physics", "continuous") == physics and (size is None or meta["size"] == size):
            found.append(meta_path.parent)
    return found


def load_play(dirs):
    """Store of every scene that ran to the end in the given play-run directories (all one grid size)."""
    parts = []
    weights = program_weights([(d / "program.py").read_text() for d in dirs])
    for run_dir in dirs:
        meta = json.loads((run_dir / "meta.json").read_text())
        ok = [i for i, error in enumerate(meta["errors"]) if error is None]
        if not ok:
            continue
        episodes = np.load(run_dir / "episodes.npz")

        def rows(key):
            return torch.from_numpy(episodes[key][ok])

        parts.append(
            {
                "mass": rows("mass"),
                "field": encode_field(rows("field").float()),
                "cheese_pos": rows("cheese_pos"),
                "cheese_alive": rows("cheese_alive"),
                "walls": rows("walls"),
                "cheese_color": rows("cheese_color"),
                "box_rect": rows("box_rect"),
                "box_color": rows("box_color"),
                "phases": [meta["phases"][i] for i in ok],
                "source": [(meta["program_id"], meta["player"], i) for i in ok],
                "episode_weight": torch.full((len(ok),), weights[(run_dir / "program.py").read_text()]),
            }
        )
    sizes = {p["mass"].shape[-1] for p in parts}
    assert len(sizes) <= 1, f"play runs of several grid sizes: {sorted(sizes)}"
    store = {k: torch.cat([p[k] for p in parts]) for k in parts[0] if k not in ("phases", "source")}
    store["phases"] = [row for p in parts for row in p["phases"]]
    store["source"] = [row for p in parts for row in p["source"]]
    return store


BIG_KEYS = ("mass", "field")  # the per-step arrays, kept on disk and memory-mapped


def cache_play(dirs, out_dir, threads=16):
    """Write the store of load_play(dirs) to out_dir without holding it in memory: the per-step
    arrays as .npy files filled one run at a time, the rest in small.pt. Several processes can
    then share one copy through the page cache (load_play_cache) instead of each loading its own."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = program_weights([(d / "program.py").read_text() for d in dirs])
    metas = [json.loads((d / "meta.json").read_text()) for d in dirs]
    oks = [[i for i, error in enumerate(m["errors"]) if error is None] for m in metas]
    total = sum(len(ok) for ok in oks)
    big, small = None, {k: [] for k in ("cheese_pos", "cheese_alive", "walls", "cheese_color", "box_rect", "box_color", "episode_weight")}
    phases, source = [], []
    row = 0
    def read(item):
        run_dir, ok = item
        episodes = np.load(run_dir / "episodes.npz")
        arrays = {k: episodes[k][ok] for k in ("mass", "field", *small) if k != "episode_weight"}
        arrays["field"] = encode_field(torch.from_numpy(arrays["field"]).float()).numpy()
        return arrays

    # Reading is bound by a slow disk (Kaggle's input mount): read runs in parallel, a bounded
    # number at a time, and write them in order.
    work = [(d, m, ok) for d, m, ok in zip(dirs, metas, oks) if ok]
    with ThreadPoolExecutor(threads) as pool:
        for i in range(0, len(work), 4 * threads):
            part = work[i : i + 4 * threads]
            for (run_dir, meta, ok), arrays in zip(part, pool.map(read, [(d, ok) for d, _, ok in part])):
                mass, field = arrays["mass"], arrays["field"]
                if big is None:
                    big = {
                        "mass": np.lib.format.open_memmap(out / "mass.npy", mode="w+", dtype=mass.dtype, shape=(total, *mass.shape[1:])),
                        "field": np.lib.format.open_memmap(out / "field.npy", mode="w+", dtype=field.dtype, shape=(total, *field.shape[1:])),
                    }
                assert mass.shape[1:] == big["mass"].shape[1:], f"play runs of several shapes: {mass.shape} {big['mass'].shape}"
                big["mass"][row : row + len(ok)] = mass
                big["field"][row : row + len(ok)] = field
                for k in small:
                    if k == "episode_weight":
                        small[k].append(torch.full((len(ok),), weights[(run_dir / "program.py").read_text()]))
                    else:
                        small[k].append(torch.from_numpy(arrays[k]))
                phases += [meta["phases"][j] for j in ok]
                source += [(meta["program_id"], meta["player"], j) for j in ok]
                row += len(ok)
    for array in big.values():
        array.flush()
    torch.save({**{k: torch.cat(v) for k, v in small.items()}, "phases": phases, "source": source}, out / "small.pt")
    (out / "done").write_text(str(total))
    return total


def load_play_cache(out_dir):
    """The store written by cache_play, its per-step arrays memory-mapped read-only."""
    out = Path(out_dir)
    store = torch.load(out / "small.pt")
    for k in BIG_KEYS:
        store[k] = torch.from_numpy(np.load(out / f"{k}.npy", mmap_mode="r"))
    return store
