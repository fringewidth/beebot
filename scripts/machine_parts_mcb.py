"""Mechanical Components Benchmark (MCB, dataset A tarball) -> data/machine_parts/mcb.{npz,jsonl} + mcb_parts.jsonl.

    nice -n 10 .venv/bin/python -I scripts/machine_parts_mcb.py <MCB_A.tar.gz | -> [--per-class 120] [--procs 2] [--limit N]

The tarball is streamed (never extracted). Every .obj is parsed for its bounding box, face count and class (the
directory name); the parts table `mcb_parts.jsonl` has one line per part. Up to --per-class parts per class
(the first ones in tar order) are voxelised and their accepted variants stored.
MCB meshes are normalised (the longest bounding extent is not in mm), so `dims_mm` is null and `dims_norm`
holds the bounding box in the file's unit. Progress and summary lines go to logs/machine-parts.out.
"""

import argparse
import itertools
import json
import sys
import tarfile
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from beebot.vla.voxel import goal_library as G
from beebot.vla.voxel import machine_parts as MP

OUT = Path("data/machine_parts")
LOG = Path("logs/machine-parts.out")
SOURCE = "mcb"
MAX_FACES = 40_000  # parts with more triangles are tabled but not voxelised (memory)


def log(row):
    if ARGS.limit:
        return
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as handle:
        handle.write(json.dumps(row) + "\n")


def work(job):
    name, data, voxelise = job
    parts = Path(name).parts
    split, label, stem = parts[-3], parts[-2], Path(name).stem
    verts, faces = MP.parse_obj(data.decode("utf-8", "replace").splitlines())
    record = {"source": SOURCE, "part_id": f"{split}/{label}/{stem}", "name": stem, "label": label, "split": split}
    if len(verts) == 0 or len(faces) == 0:
        record.update({"error": "empty"})
        return record, [], {}
    dims = verts.max(axis=0) - verts.min(axis=0)
    record.update(
        {
            "dims_mm": None,
            "dims_norm": [round(float(d), 5) for d in dims],
            "shape_class": MP.shape_class(dims),
            "faces": int(len(faces)),
            "joints": [],
            "parent": None,
        }
    )
    if not voxelise or min(dims) < 0 or max(dims) <= 0:
        return record, [], {}
    if len(faces) > MAX_FACES:
        record["skipped"] = "too_many_faces"
        return record, [], {}
    rng = np.random.default_rng(G.seed_of(record["part_id"]))
    found, tally = MP.part_variants(verts, faces, dims, rng)
    if not found:
        record["reject"] = tally
    return record, found, tally


def jobs(tar_path, per_class, limit):
    """Yield (name, bytes, voxelise) streaming the tarball in one pass (a path, or - for stdin so the download
    never touches the disk). The first `per_class` parts of each class in tar order are voxelised; tar order is
    by split then class directory then id."""
    seen = Counter()
    count = 0
    handle = sys.stdin.buffer if tar_path == "-" else open(tar_path, "rb")
    with tarfile.open(fileobj=handle, mode="r|gz") as tar:
        try:
            for member in tar:
                if member.isfile() and member.name.endswith(".obj"):
                    label = Path(member.name).parts[-2]
                    voxelise = seen[label] < per_class
                    seen[label] += 1
                    yield member.name, tar.extractfile(member).read(), voxelise
                    count += 1
                    if limit and count >= limit:
                        return
        except (EOFError, tarfile.ReadError):
            return


def batched(pool, stream, size=64):
    """Results of work() over a job stream, `size` jobs in flight at a time (bounded memory)."""
    while True:
        batch = list(itertools.islice(stream, size))
        if not batch:
            return
        yield from pool.map(work, batch, chunksize=8)


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("tar")
    parser.add_argument("--per-class", type=int, default=120)
    parser.add_argument("--procs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    ARGS = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.time()
    records, lines, cell_lists, tallies = [], [], [], Counter()
    voxelised, with_goal, accepted = 0, 0, 0
    thick, cells_hist = [], []
    rejected_why = Counter()
    suffix = "_probe" if ARGS.limit else ""
    gid = 0
    with Pool(ARGS.procs, maxtasksperchild=25) as pool:
        for record, found, tally in batched(pool, jobs(ARGS.tar, ARGS.per_class, ARGS.limit)):
            if "error" not in record and tally != {} or found:
                voxelised += 1
                tallies.update(tally)
                record["accepted"] = len(found)
                if found:
                    with_goal += 1
                else:
                    rejected_why[max(tally, key=tally.get) if tally else "none"] += 1  # the most common reject per part
            record.setdefault("accepted", None)
            records.append(record)
            item = G.Item(SOURCE, record["part_id"], "part", None, text="", label=record["label"], extra={})
            for goal in found:
                gid += 1
                extra = {
                    "orientation": goal["orientation"],
                    "oriented_by": goal["oriented_by"],
                    "cell_mm": None,
                    "cell_norm": round(goal["cell_mm"], 6),
                    "thickness_cells": round(goal["thickness_cells"], 3),
                    "shape_class": record["shape_class"],
                    "split": record["split"],
                }
                line = G.entry_line(item, goal, f"{SOURCE}-{gid}")
                line["extra"] = extra
                lines.append(line)
                cell_lists.append(G.vox_to_cells(goal["vox"]))
                accepted += 1
                thick.append(goal["thickness_cells"])
                cells_hist.append(int(goal["vox"].sum()))
            if len(records) % 2000 == 0:
                log({"t": round(time.time() - start), "source": SOURCE, "parts_seen": len(records), "voxelised": voxelised, "machine_parts_accepted": accepted})
    G.save(SOURCE + suffix, lines, cell_lists, root=OUT)
    with open(OUT / f"{SOURCE}_parts{suffix}.jsonl", "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    summary = {
        "t": round(time.time() - start),
        "source": SOURCE,
        "parts": len(records),
        "classes": len({r["label"] for r in records}),
        "voxelised": voxelised,
        "parts_with_goal": with_goal,
        "variants_accepted": accepted,
        "reject_tally": dict(tallies),
        "no_goal_main_reason": dict(rejected_why),
        "cells_hist": G.histogram(cells_hist),
        "thickness_cells_quantiles": [round(float(q), 3) for q in np.quantile(thick, [0.05, 0.25, 0.5, 0.75, 0.95])] if thick else [],
        "share_thinner_than_1_cell": round(float(np.mean(np.array(thick) < 1.0)), 4) if thick else None,
        "machine_parts_accepted": accepted,
    }
    log(summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
