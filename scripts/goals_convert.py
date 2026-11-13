"""Convert a downloaded source into the goal library (data/goal_library/<source>.{npz,jsonl}).

    nice -n 10 .venv/bin/python scripts/goals_convert.py stabletext2brick [--limit N] [--procs 3]
        [--max-cells 800] [--keep-targets 100 250 500 800] [--out data/goal_library]

Sources: stabletext2brick, hack337, craftassist_houses, craftassist_parts, iglu_multiturn, iglu_single.
Raw files are read from data/goal_sources/ (never committed). One JSON progress line per ~2000 items and a
summary line at the end go to logs/goal-library.out (the dashboard follows it).
"""

import argparse
import csv
import glob
import json
import os
import pickle
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from beebot.vla.voxel import goal_library as G

SOURCES = Path("data/goal_sources")
LOG = Path("logs/goal-library.out")
ARGS_LIMIT = [0]


def log(row):
    if ARGS_LIMIT[0]:
        return
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as handle:
        handle.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------------------------
# Per-source task lists and item makers. A task is a small picklable tuple; make(task) -> [Item].


def tasks_stabletext2brick():
    import pyarrow.parquet as pq

    for path in sorted(glob.glob(str(SOURCES / "stabletext2brick/data/*.parquet"))):
        split = "test" if "test-" in path else "train"
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=["structure_id", "object_id", "category_id", "captions", "bricks"])
            for row in table.to_pylist():
                yield (split, row["structure_id"], row["object_id"], row["category_id"], row["captions"], row["bricks"])


def make_stabletext2brick(task):
    split, sid, oid, cat, captions, bricks = task
    vox = G.parse_bricks(bricks)
    text = captions[0] if captions else ""
    return [G.Item("stabletext2brick", sid, "whole", vox, text=text, label=cat, extra={"object": oid, "split": split, "captions": len(captions)})]


def tasks_hack337():
    root = SOURCES / "minecraft_schematics"
    names = {}
    for entry in json.load(open(root / "schematics_mapping.json")):
        names[Path(entry["file"]).stem] = (entry.get("name", ""), entry.get("description", ""))
    short = {}
    for line in open(root / "blueprints_mapping.jsonl"):
        row = json.loads(line)
        short[Path(row["filename"]).stem] = row.get("short_name_en", "")
    for path in sorted(glob.glob(str(root / "blueprints/*.json"))):
        stem = Path(path).stem
        name, description = names.get(stem, ("", ""))
        yield (path, stem, name, description, short.get(stem, ""))


def make_hack337(task):
    path, stem, name, description, short = task
    blueprint = json.load(open(path))
    vox = G.parse_blueprint(blueprint)
    text = (name + ". " + description).strip(". ") if (name or description) else short
    extra = {"short": short}
    items = [G.Item("hack337", stem, "whole", vox, text=text, extra=extra)]
    rng = np.random.default_rng(G.seed_of(stem))
    for i, piece in enumerate(G.crops(vox, rng)):
        items.append(G.Item("hack337", f"{stem}#c{i}", "crop", piece, text=text, extra=extra))
    return items


def tasks_craftassist_houses():
    for path in sorted(glob.glob(str(SOURCES / "craftassist_s3/house_data/houses/*/schematic.npy"))):
        yield (path,)


def make_craftassist_houses(task):
    (path,) = task
    folder = Path(path).parent
    schematic = np.load(path)
    vox = G.craftassist_occupancy(schematic)
    extra = {}
    try:
        placed = json.load(open(folder / "placed.json"))
        extra["placed"] = sum(1 for event in placed if event[-1] == "P")
        extra["broken"] = sum(1 for event in placed if event[-1] == "B")
    except (OSError, ValueError):
        pass
    sid = folder.name
    items = [G.Item("craftassist", sid, "whole", vox, text="house", extra=extra)]
    rng = np.random.default_rng(G.seed_of(sid))
    for i, piece in enumerate(G.crops(vox, rng)):
        items.append(G.Item("craftassist", f"{sid}#c{i}", "crop", piece, text="part of a house", extra=extra))
    return items


def tasks_craftassist_parts():
    for name in ("training_data.pkl", "validation_data.pkl"):
        path = SOURCES / "craftassist_s3/seg/instance_segmentation_data" / name
        data = pickle.load(open(path, "rb"))
        for i, (schematic, annotated, labels, house) in enumerate(data):
            for k in range(1, len(labels)):
                yield (house, k, labels[k], np.packbits(((annotated == k) & (schematic != 0))), annotated.shape)


def make_craftassist_parts(task):
    house, k, label, packed, shape = task
    vox = np.unpackbits(packed)[: int(np.prod(shape))].reshape(shape).astype(bool)
    if label in ("nothing", "none") or vox.sum() < G.MIN_CELLS:
        return []
    return [G.Item("craftassist_parts", f"{house}#{k}", "part", vox, text=label, label=label, extra={"house": house})]


def tasks_iglu_multiturn():
    root = SOURCES / "iglu_mt/x"
    import pandas as pd

    dialogs = pd.read_csv(root / "dialogs.csv")
    for session, group in dialogs.groupby("PartitionKey"):
        steps = glob.glob(str(root / "builder-data" / session / "step-*"))
        if not steps:
            continue
        last = max(steps, key=lambda p: int(p.rsplit("-", 1)[1]))
        texts = [str(v) for v in group.sort_values("StepId")["instruction"].dropna().tolist()]
        yield (session, last, " | ".join(texts))


def make_iglu_multiturn(task):
    session, path, text = task
    blocks = json.load(open(path))["worldEndingState"]["blocks"]
    vox = G.parse_blocks(blocks)
    if vox is None:
        return []
    return [G.Item("iglu", session, "whole", vox, text=text, extra={"colours": len({b[3] for b in blocks})})]


def tasks_iglu_single():
    root = SOURCES / "iglu/datasets/singleturn"
    instruction = {}
    with open(root / "clarifying_questions_train.csv") as handle:
        for row in csv.DictReader(handle):
            instruction.setdefault(row["GameId"].lower(), row["InputInstruction"])
    for folder in sorted(glob.glob(str(root / "target_world_states/builder-data/cq-game-*"))):
        files = glob.glob(folder + "/*")
        if files:
            yield (Path(folder).name, files[0], instruction.get(Path(folder).name, ""))


def make_iglu_single(task):
    game, path, text = task
    blocks = json.load(open(path))["worldEndingState"]["blocks"]
    vox = G.parse_blocks(blocks)
    if vox is None:
        return []
    return [G.Item("iglu_single", game, "whole", vox, text=text)]


SOURCE_TABLE = {
    "stabletext2brick": (tasks_stabletext2brick, make_stabletext2brick),
    "hack337": (tasks_hack337, make_hack337),
    "craftassist_houses": (tasks_craftassist_houses, make_craftassist_houses),
    "craftassist_parts": (tasks_craftassist_parts, make_craftassist_parts),
    "iglu_multiturn": (tasks_iglu_multiturn, make_iglu_multiturn),
    "iglu_single": (tasks_iglu_single, make_iglu_single),
}


def work(args):
    name, task, keep, max_cells = args
    try:
        items = SOURCE_TABLE[name][1](task)
    except Exception as error:  # a malformed raw file is counted, not fatal
        return {"error": f"{type(error).__name__}: {error}"[:120]}
    results = []
    for item in items:
        goals, tally = G.process(item, keep=keep, max_cells=max_cells)
        meta = {"item": item, "goals": goals, "tally": tally}
        item.vox = None  # the raw voxels stay in the worker
        results.append(meta)
    return {"results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=sorted(SOURCE_TABLE))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--procs", type=int, default=3)
    parser.add_argument("--max-cells", type=int, default=G.MAX_CELLS)
    parser.add_argument("--keep-targets", type=int, nargs="+", default=list(G.KEEP_TARGETS))
    parser.add_argument("--out", default=str(G.LIBRARY))
    args = parser.parse_args()
    keep = tuple(args.keep_targets)
    out_root = Path(args.out)
    name = args.source
    ARGS_LIMIT[0] = args.limit
    started = time.time()
    lines, cell_lists = [], []
    stats = Counter()
    tally = Counter()
    errors = Counter()
    with_goal = 0
    seen = set()
    per_kind = Counter()
    accepted_kind = Counter()

    def tasks():
        for i, task in enumerate(SOURCE_TABLE[name][0]()):
            if args.limit and i >= args.limit:
                break
            yield (name, task, keep, args.max_cells)

    with Pool(args.procs) as pool:
        for n, out in enumerate(pool.imap(work, tasks(), chunksize=8), 1):
            stats["structures"] += 1
            if "error" in out:
                errors[out["error"]] += 1
                continue
            for meta in out["results"]:
                item = meta["item"]
                stats["candidates"] += 1
                per_kind[item.kind] += 1
                tally.update(meta["tally"])
                if meta["goals"]:
                    with_goal += 1
                for goal in meta["goals"]:
                    key = (item.kind, item.label, item.text, goal["vox"].shape, goal["vox"].tobytes())
                    if key in seen:
                        stats["duplicates"] += 1
                        continue
                    seen.add(key)
                    line = G.entry_line(item, goal, f"{name}:{len(lines)}")
                    lines.append(line)
                    cell_lists.append(G.vox_to_cells(goal["vox"]))
                    accepted_kind[item.kind] += 1
            if n % 2000 == 0:
                log({"source": name, "structures": stats["structures"], "candidates": stats["candidates"], "accepted": len(lines)})
    G.save(name, lines, cell_lists, root=out_root)
    counts = [line["cells"] for line in lines]
    size = sum(os.path.getsize(out_root / f"{name}.{ext}") for ext in ("npz", "jsonl"))
    summary = {
        "source": name,
        "structures": stats["structures"],
        "candidates": stats["candidates"],
        "candidates_by_kind": dict(per_kind),
        "candidates_with_goal": with_goal,
        "accepted": len(lines),
        "exact_duplicates_dropped": stats["duplicates"],
        "accepted_by_kind": dict(accepted_kind),
        "cells_hist": G.histogram(counts),
        "rejects": dict(tally),
        "errors": dict(errors),
        "library_bytes": size,
        "seconds": round(time.time() - started),
    }
    log(summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
