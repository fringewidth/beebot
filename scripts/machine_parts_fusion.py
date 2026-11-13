"""Fusion 360 Gallery Assembly dataset (one 7z archive) -> data/machine_parts/fusion360*.{npz,jsonl}.

    nice -n 10 .venv/bin/python scripts/machine_parts_fusion.py a1.0.0_04 [--obj-mb 600] [--procs 2] [--limit N]

The archive is read over HTTP range requests (nothing is stored but what is extracted): every assembly.json
and the body .obj files of the first assemblies whose meshes fit --obj-mb. Then, per assembly, every body
becomes a part row (data/machine_parts/fusion360_parts.jsonl: names, material, physical properties, joints,
parent, bounding box in mm), bodies with meshes are voxelised (fusion360.npz/.jsonl, goal_library layout) and
small assemblies are voxelised whole with a part id per cell (fusion360_assemblies.npz/.jsonl).
Extracted meshes are deleted at the end; the .json files stay. Lines go to logs/machine-parts.out.
Units: the documentation says cm; --unit-mm (default 10) converts a file length to mm.
"""

import argparse
import io
import json
import shutil
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from beebot.vla.voxel import goal_library as G
from beebot.vla.voxel import machine_parts as MP
from beebot.vla.voxel.remote_file import HttpFile

URL = "https://fusion-360-gallery-dataset.s3-us-west-2.amazonaws.com/assembly/a1.0.0/{name}.7z"
RAW = Path("data/machine_sources/fusion360_assembly")
OUT = Path("data/machine_parts")
LOG = Path("logs/machine-parts.out")
SOURCE = "fusion360"
MAX_FACES = 40_000
WHOLE_MAX_BODIES = 120
ARGS = None


def log(row):
    if ARGS is not None and ARGS.limit:
        return
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as handle:
        handle.write(json.dumps(row) + "\n")


def extract(name, obj_mb, limit):
    """Extract assembly.json of every assembly and meshes of the first assemblies within obj_mb. Returns the
    assembly directory names in archive order and those with meshes."""
    import py7zr

    name = Path(name).name.removesuffix(".7z")
    path = RAW / name
    source = Path(ARGS.archive) if Path(ARGS.archive).exists() else None
    handle = open(source, "rb") if source else io.BufferedReader(HttpFile(URL.format(name=name)), 1 << 20)
    archive = py7zr.SevenZipFile(handle)
    infos = archive.list()
    dirs = []
    sizes = Counter()
    for info in infos:
        top = info.filename.split("/")[0]
        if info.filename.endswith("/assembly.json"):
            dirs.append(top)
        if info.filename.endswith(".obj") and not info.filename.endswith("/assembly.obj"):
            sizes[top] += info.uncompressed
    if limit:
        dirs = dirs[:limit]
    budget = obj_mb * 1e6
    with_mesh = []
    for top in dirs:
        if budget - sizes[top] < 0:
            break
        budget -= sizes[top]
        with_mesh.append(top)
    chosen = set(with_mesh)
    keep = set(dirs)
    targets = []
    for info in infos:
        top = info.filename.split("/")[0]
        if top not in keep:
            continue
        if info.filename.endswith("/assembly.json"):
            targets.append(info.filename)
        elif top in chosen and info.filename.endswith(".obj") and not info.filename.endswith("/assembly.obj"):
            targets.append(info.filename)
    log({"source": SOURCE, "archive": name, "assemblies": len(dirs), "with_meshes": len(with_mesh), "extract_files": len(targets), "extract_mb": round(sum(i.uncompressed for i in infos if i.filename in set(targets)) / 1e6)})
    archive.extract(path=path, targets=targets)
    return dirs, with_mesh


def mesh_bbox_mm(verts, unit):
    return [round(float(v) * unit, 3) for v in verts.max(axis=0) - verts.min(axis=0)]


def work(job):
    """One assembly: its part rows, accepted part variants, joints, whole-assembly voxels."""
    directory, name, unit, voxelise = job
    base = RAW / name / directory
    with open(base / "assembly.json") as handle:
        data = json.load(handle)
    parts, joints, parent = MP.parse_assembly(data)
    props = data.get("properties") or {}
    asm = {
        "assembly": directory,
        "assembly_name": props.get("name", ""),
        "categories": props.get("categories", []),
        "industries": props.get("industries", []),
        "bodies": len(parts),
        "occurrences": len(data.get("occurrences", {})),
        "joints": len(joints),
        "joint_kinds": dict(Counter(j["kind"] for j in joints)),
        "contacts": len(data.get("contacts") or []),
        "holes": len(data.get("holes") or []),
    }
    occ_name = {k: v.get("name", "") for k, v in (data.get("occurrences") or {}).items()}
    by_occ = {}
    for j in joints:
        for side, other in (("one", "two"), ("two", "one")):
            by_occ.setdefault(j[side], []).append(
                {
                    "kind": j["kind"],
                    "partner": occ_name.get(j[other], j[other]),
                    "partner_occurrence": j[other],
                    "name": j["name"],
                    **{k: j[k] for k in ("rotation_axis", "slide_direction", "rotation_limits", "slide_limits") if k in j},
                }
            )
    rows, found_all, tallies, placed = [], [], Counter(), []
    for index, part in enumerate(parts):
        row = {
            "source": SOURCE,
            "assembly": directory,
            "assembly_name": asm["assembly_name"],
            "part_id": f"{directory}/{part['body']}",
            "index": index,
            "name": part["body_name"],
            "occurrence_name": part["occurrence_name"],
            "component_name": part["component_name"],
            "part_number": part["part_number"],
            "label": "",
            "material": part["material"],
            "appearance": part["appearance"],
            "volume": part["physical"].get("volume"),
            "mass": part["physical"].get("mass"),
            "grounded": part["grounded"],
            "visible": part["visible"],
            "parent_occurrence": part["parent"],
            "parent_name": occ_name.get(part["parent"], ""),
            "depth": part["depth"],
            "joints": by_occ.get(part["occurrence"], []),
            "dims_mm": None,
            "accepted": None,
        }
        for field in ("occurrence_name", "component_name", "name"):
            text = row[field].split(":")[0].strip()
            if text and not MP.name_is_generic(text):
                row["label"] = text
                break
        mesh = base / (part["obj"] or "")
        if voxelise and part["obj"] and mesh.exists():
            verts, faces = MP.read_obj(mesh)
            if len(verts) and len(faces):
                dims = verts.max(axis=0) - verts.min(axis=0)
                row["dims_mm"] = mesh_bbox_mm(verts, unit)
                row["shape_class"] = MP.shape_class(dims)
                row["faces"] = int(len(faces))
                if len(faces) <= MAX_FACES and max(dims) > 0:
                    rng = np.random.default_rng(G.seed_of(row["part_id"]))
                    found, tally = MP.part_variants(verts, faces, [d * unit for d in dims], rng)
                    tallies.update(tally)
                    row["accepted"] = len(found)
                    if not found:
                        row["reject"] = tally
                    for variant in found:
                        found_all.append((index, variant))
                if part["visible"]:
                    homog = np.c_[verts, np.ones(len(verts))] @ part["world"].T
                    placed.append((homog[:, :3], faces, index))
        rows.append(row)
    whole = None
    if voxelise and 2 <= len(placed) <= WHOLE_MAX_BODIES and len(placed) >= 0.8 * sum(1 for p in parts if p["visible"]):
        ids, pitch = MP.whole_assembly_voxels(placed)
        crop, scale = MP.assembly_variant(ids)
        if crop is not None:
            whole = (crop, scale)
    return asm, rows, found_all, dict(tallies), whole


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--obj-mb", type=int, default=600)
    parser.add_argument("--procs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--unit-mm", type=float, default=10.0)
    parser.add_argument("--skip-extract", action="store_true")
    ARGS = parser.parse_args()
    start = time.time()
    if ARGS.skip_extract:
        dirs = sorted(p.name for p in (RAW / ARGS.archive).iterdir() if (p / "assembly.json").exists())
        with_mesh = {d for d in dirs if any(p.suffix == ".obj" for p in (RAW / ARGS.archive / d).iterdir())}
        with_mesh = [d for d in dirs if d in with_mesh]
    else:
        dirs, with_mesh = extract(ARGS.archive, ARGS.obj_mb, ARGS.limit)
    has_mesh = set(with_mesh)
    log({"t": round(time.time() - start), "source": SOURCE, "stage": "extracted", "assemblies": len(dirs), "with_meshes": len(with_mesh)})
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_probe" if ARGS.limit else ""
    rows_all, asm_all, lines, cell_lists, tallies = [], [], [], [], Counter()
    whole_lines, whole_cells = [], []
    accepted = 0
    thick, cells_hist = [], []
    gid = 0
    jobs = [(d, ARGS.archive, ARGS.unit_mm, d in has_mesh) for d in dirs]
    with Pool(ARGS.procs, maxtasksperchild=10) as pool:
        for n, (asm, rows, found, tally, whole) in enumerate(pool.imap(work, jobs, chunksize=1), 1):
            asm_all.append(asm)
            rows_all += rows
            tallies.update(tally)
            for index, goal in found:
                gid += 1
                row = rows[index]
                item = G.Item(SOURCE, row["part_id"], "part", None, text=row["label"], label=row["label"])
                line = G.entry_line(item, goal, f"{SOURCE}-{gid}")
                line["extra"] = {
                    "assembly": row["assembly"],
                    "assembly_name": row["assembly_name"],
                    "part_name": row["name"],
                    "orientation": goal["orientation"],
                    "oriented_by": goal["oriented_by"],
                    "cell_mm": round(goal["cell_mm"], 4),
                    "thickness_cells": round(goal["thickness_cells"], 3),
                    "dims_mm": row["dims_mm"],
                    "shape_class": row.get("shape_class"),
                    "joint_kinds": sorted({j["kind"] for j in row["joints"]}),
                }
                lines.append(line)
                cell_lists.append(G.vox_to_cells(goal["vox"]))
                accepted += 1
                thick.append(goal["thickness_cells"])
                cells_hist.append(int(goal["vox"].sum()))
            if whole is not None:
                crop, scale = whole
                zs, ys, xs = np.nonzero(crop >= 0)
                cells = np.c_[zs, ys, xs, crop[zs, ys, xs]].astype(np.int16)
                whole_cells.append(cells)
                used = sorted({int(v) for v in crop[crop >= 0]})
                whole_lines.append(
                    {
                        "id": f"{SOURCE}-asm-{len(whole_lines) + 1}",
                        "assembly": asm["assembly"],
                        "assembly_name": asm["assembly_name"],
                        "scale": round(float(scale), 4),
                        "dims": [int(v) for v in crop.shape],
                        "cells": int((crop >= 0).sum()),
                        "parts": [
                            {"index": i, "name": rows[i]["label"] or rows[i]["name"], "generic": not rows[i]["label"], "material": rows[i]["material"], "dims_mm": rows[i]["dims_mm"], "joints": sorted({j["kind"] for j in rows[i]["joints"]})}
                            for i in used
                        ],
                        "categories": asm["categories"],
                    }
                )
            if n % 100 == 0:
                log({"t": round(time.time() - start), "source": SOURCE, "assemblies_done": n, "machine_parts_accepted": accepted})
    first = ARGS.archive == "a1.0.0_04"
    if with_mesh:
        G.save(SOURCE + suffix, lines, cell_lists, root=OUT)
    # whole assemblies: cells (N, 4) int16 z, y, x, part index
    starts = np.zeros(len(whole_cells) + 1, dtype=np.int64)
    for i, c in enumerate(whole_cells):
        starts[i + 1] = starts[i] + len(c)
    flat = np.concatenate(whole_cells) if whole_cells else np.zeros((0, 4), dtype=np.int16)
    if with_mesh:
        np.savez_compressed(OUT / f"{SOURCE}_assemblies{suffix}.npz", cells=flat, starts=starts)
        with open(OUT / f"{SOURCE}_assemblies{suffix}.jsonl", "w") as handle:
            for line in whole_lines:
                handle.write(json.dumps(line) + "\n")
    with open(OUT / f"{SOURCE}_parts{suffix}.jsonl", "w" if first else "a") as handle:
        for row in rows_all:
            handle.write(json.dumps(row) + "\n")
    with open(OUT / f"{SOURCE}_assembly_table{suffix}.jsonl", "w" if first else "a") as handle:
        for asm in asm_all:
            handle.write(json.dumps(asm) + "\n")
    summary = {
        "t": round(time.time() - start),
        "source": SOURCE,
        "archive": ARGS.archive,
        "assemblies": len(asm_all),
        "assemblies_with_meshes": len(with_mesh),
        "parts": len(rows_all),
        "parts_voxelised": sum(1 for r in rows_all if r["accepted"] is not None),
        "parts_with_goal": sum(1 for r in rows_all if r["accepted"]),
        "variants_accepted": accepted,
        "whole_assemblies": len(whole_lines),
        "reject_tally": dict(tallies),
        "cells_hist": G.histogram(cells_hist),
        "machine_parts_accepted": accepted,
    }
    log(summary)
    print(json.dumps(summary, indent=1))
    if not ARGS.skip_extract and not ARGS.limit:
        for d in dirs:
            for mesh in (RAW / ARGS.archive / d).glob("*.obj"):
                mesh.unlink()


if __name__ == "__main__":
    main()
