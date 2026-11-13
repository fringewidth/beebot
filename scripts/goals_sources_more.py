"""Convert the extra goal sources into data/goal_library_v2/<source>.{npz,jsonl} (cells, starts, rgb).

    nice -n 10 .venv/bin/python -I scripts/goals_sources_more.py oga_vox [--limit N] [--procs 3]
    nice -n 10 .venv/bin/python -I scripts/goals_sources_more.py text2shape ...
    nice -n 10 .venv/bin/python -I scripts/goals_sources_more.py objaverse_cap3d ...

Raw files are read from data/goal_sources/ (untrusted: parsed as data only, never executed). One JSON progress
line per ~500 items and a summary line per source go to logs/goal-sources-more.out, with "goal_library_v2_added".
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from beebot.vla.voxel import goal_library as G
from beebot.vla.voxel import goal_sources as GS
from beebot.vla.voxel import vox as V

SOURCES = Path("data/goal_sources")
OUT = Path("data/goal_library_v2")
LOG = Path("logs/goal-sources-more.out")
LIMIT = [0]


def log(row):
    if LIMIT[0]:
        return
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as handle:
        handle.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------------------------
# OpenGameArt voxel packs


OGA = SOURCES / "oga_packs"
SKIP_NAME = re.compile(r"(40x40|\d+x\d+|test|asset|^\d+$|lvl|basta|krimos|sto_i|bluebox|redbox|greenbox|^-)", re.I)
PACK_NOUN = {"voxel-buildings": "building", "voxel-character-starter-pack": "character", "3d-voxel-tower": "tower"}


def oga_text(slug, filename, model_name):
    """The text of a model: its name made readable; numbered files take the pack's noun; letters and numbers get a sentence."""
    stem = Path(filename).stem
    if slug == "alphabet-3d-voxel":
        if len(stem) == 1 and stem.isalpha():
            return f"the letter {stem.lower()}"
        return words_or("", stem)
    if slug == "voxel-numbers":
        return f"the number {stem}" if stem.isdigit() else ""
    name = model_name
    if name.lower().endswith(".vox") or re.search(r"\.vox( \d+)?$", name):
        name = re.sub(r"\.vox( \d+)?$", "", name)
    if " - " in name:  # medieval: "Category - Name - Part"
        parts = [p for p in name.split(" - ") if p.strip()]
        name = " ".join(parts[1:]) if len(parts) > 1 else parts[0]
    text = GS.words(name)
    text = re.sub(r"\b(variant|var|v)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text and slug in PACK_NOUN:
        text = PACK_NOUN[slug]
    return text


def words_or(default, stem):
    return GS.words(stem) or default


def tasks_oga_vox():
    manifest = {m["slug"]: m for m in json.load(open(OGA / "manifest.json"))}
    for path in sorted(OGA.glob("vox/**/*.vox")):
        slug = path.relative_to(OGA / "vox").parts[0]
        if SKIP_NAME.search(path.stem) and "tangled" in slug:
            continue
        if path.stem.startswith("-"):
            continue
        yield (str(path), slug, manifest[slug]["lic"] if slug in manifest else [])


def make_oga_vox(task):
    """[(Item, rgb array (z, y, x, 3))] for every model of a file."""
    path, slug, lic = task
    data = open(path, "rb").read()
    found = V.any_vox(data, fallback=Path(path).name)
    out = []
    for index, (name, indices, palette) in enumerate(found):
        text = oga_text(slug, path, name)
        if not text:
            continue
        rgb = GS.indices_to_rgb(indices, palette)
        item = G.Item(
            "oga_vox",
            f"{slug}/{Path(path).stem}#{index}",
            "whole",
            indices > 0,
            text=text,
            label=slug,
            extra={"pack": slug, "file": Path(path).name, "model": name, "licence": lic, "url": f"https://opengameart.org/content/{slug}", "colours": int(len(np.unique(indices)) - 1)},
        )
        out.append((item, rgb))
    return out


ITCH = SOURCES / "itch_padadu"


def tasks_itch_vox():
    """Free CC0 voxel packs of one itch.io creator (licence stated on each pack page); single .vox files and .vox
    members of a pack zip. Packs whose page states no licence are not downloaded."""
    import zipfile

    manifest = {m["slug"]: m for m in json.load(open(ITCH / "manifest.json"))}
    for path in sorted(ITCH.glob("*/*")):
        slug = path.parent.name
        lic = manifest[slug]["licence_mentions"]
        if "CC0" not in lic:
            continue
        if path.suffix == ".vox":
            yield (str(path), slug, ["CC0"], None)
        elif path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in sorted(archive.namelist()):
                    if member.lower().endswith(".vox"):
                        yield (str(path), slug, ["CC0"], member)


def make_itch_vox(task):
    import zipfile

    path, slug, lic, member = task
    if member is None:
        data = open(path, "rb").read()
        filename = Path(path).name
    else:
        with zipfile.ZipFile(path) as archive:
            data = archive.read(member)
        filename = Path(member).name
    out = []
    for index, (name, indices, palette) in enumerate(V.any_vox(data, fallback=filename)):
        text = oga_text(slug, filename, name)
        if not text:
            continue
        item = G.Item(
            "itch_vox",
            f"{slug}/{Path(filename).stem}#{index}",
            "whole",
            indices > 0,
            text=text,
            label=slug,
            extra={"pack": slug, "file": filename, "licence": lic, "url": f"https://padadu.itch.io/{slug}", "colours": int(len(np.unique(indices)) - 1)},
        )
        out.append((item, GS.indices_to_rgb(indices, palette)))
    return out


# ---------------------------------------------------------------------------------------------
# Objaverse 1.0 with Cap3D captions: meshes streamed into memory (never written to disk), voxelised, dropped


OBJ = SOURCES / "objaverse"
HF = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"
OBJ_RES = 32  # cells along the longest axis before the 16^3 search; the voxeliser then picks scales


def tasks_objaverse_cap3d(max_bytes=1_000_000):
    rows = [json.loads(line) for line in open(OBJ / "select.jsonl")]
    rows = [r for r in rows if r["bytes"] <= max_bytes and r["caption"].strip()]
    order = np.random.default_rng(0).permutation(len(rows))
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    def load(r):
        try:
            return fetch(HF + r["path"]), None
        except Exception as error:
            return None, f"{type(error).__name__}: {error}"[:100]

    window = deque()
    with ThreadPoolExecutor(12) as threads:  # downloads overlap; the pool's processes only voxelise
        for i in order:
            r = rows[int(i)]
            window.append((r, threads.submit(load, r)))
            if len(window) >= 36:
                r, future = window.popleft()
                data, error = future.result()
                yield (r["uid"], r["path"], r["lvis"], r["caption"].strip(), r["license"], r["faces"], data, error)
        while window:
            r, future = window.popleft()
            data, error = future.result()
            yield (r["uid"], r["path"], r["lvis"], r["caption"].strip(), r["license"], r["faces"], data, error)


def fetch(url, tries=3):
    import urllib.request

    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "beebot-goal-library"}), timeout=60) as response:
                return response.read()
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def mesh_to_occupancy(data, res=OBJ_RES):
    """GLB bytes -> (z up, y, x) bool surface voxelisation (filled when the mesh is closed), longest axis `res`."""
    import io

    import trimesh

    scene = trimesh.load(io.BytesIO(data), file_type="glb", force="scene")
    mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene.dump(concatenate=True)
    if len(mesh.faces) == 0:
        raise ValueError("empty mesh")
    extent = float(mesh.extents.max())
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("degenerate mesh")
    grid = mesh.voxelized(extent / (res - 1))
    if mesh.is_watertight:
        grid = grid.fill()
    occupied = np.asarray(grid.matrix, dtype=bool).transpose(1, 2, 0)  # glTF is y up: (x, y, z) -> (y, z, x)
    return occupied


def make_objaverse_cap3d(task):
    uid, path, lvis, caption, licence, faces, data, error = task
    if data is None:
        raise ValueError(error or "download failed")
    occupied = mesh_to_occupancy(data)
    item = G.Item(
        "objaverse_cap3d",
        uid,
        "whole",
        occupied,
        text=caption,
        label=lvis,
        extra={"uid": uid, "licence": licence, "faces": faces, "url": f"https://sketchfab.com/3d-models/{uid}", "caption_licence": "ODC-By 1.0 (Cap3D)"},
    )
    return [(item, np.zeros(occupied.shape + (3,), dtype=np.uint8))]


# ---------------------------------------------------------------------------------------------
# Text2Shape: ShapeNet chairs and tables, solid 32^3 RGBA voxelisations with crowd-written descriptions


T2S = SOURCES / "text2shape"


def parse_nrrd(data):
    """A Text2Shape .nrrd (gzip, uint8, sizes 4 x 32 x 32 x 32 with the channel axis fastest) -> (z, y, x, 4) uint8."""
    import gzip

    end = data.index(b"\n\n")
    header = data[:end].decode("ascii", "replace")
    fields = dict(line.split(": ", 1) for line in header.split("\n") if ": " in line)
    sizes = [int(v) for v in fields["sizes"].split()]
    if fields.get("encoding") != "gzip" or fields.get("type") != "uint8" or len(sizes) != 4:
        raise ValueError("unexpected nrrd header")
    raw = gzip.decompress(data[end + 2 :])
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size != int(np.prod(sizes)):
        raise ValueError("nrrd size mismatch")
    # the first nrrd axis is the fastest: memory order is (x, depth, up, channel)
    return np.ascontiguousarray(arr.reshape(sizes[::-1]).transpose(2, 1, 0, 3))


def tasks_text2shape():
    import csv
    import zipfile

    descriptions = {}
    categories = {}
    with open(T2S / "captions.tablechair.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            descriptions.setdefault(row["modelId"], []).append(row["description"].strip())
            categories[row["modelId"]] = row["category"].lower()
    with zipfile.ZipFile(T2S / "nrrd_256_filter_div_32_solid.zip") as archive:
        names = [n for n in archive.namelist() if n.endswith(".nrrd")]
    for name in sorted(names):
        model = Path(name).stem
        if model in descriptions:
            yield (name, model, categories[model], descriptions[model])


_ZIP = {}


def make_text2shape(task):
    import zipfile

    name, model, category, texts = task
    if "zip" not in _ZIP:
        _ZIP["zip"] = zipfile.ZipFile(T2S / "nrrd_256_filter_div_32_solid.zip")
    grid = parse_nrrd(_ZIP["zip"].read(name))
    occupied = grid[..., 3] > 0
    rgb = np.ascontiguousarray(grid[..., :3])
    item = G.Item("text2shape", model, "whole", occupied, text=texts[0], label=category, extra={"descriptions": texts, "licence": "ShapeNet terms (non-commercial research); Text2Shape descriptions", "url": "https://text2shape.stanford.edu"})
    return [(item, rgb)]


TABLE = {"oga_vox": (tasks_oga_vox, make_oga_vox), "itch_vox": (tasks_itch_vox, make_itch_vox), "text2shape": (tasks_text2shape, make_text2shape), "objaverse_cap3d": (tasks_objaverse_cap3d, make_objaverse_cap3d)}
NO_COLOUR = {"objaverse_cap3d"}


# ---------------------------------------------------------------------------------------------


def work(args):
    name, task, keep, max_cells, tolerance = args
    try:
        pairs = TABLE[name][1](task)
    except Exception as error:  # a malformed raw file is counted, not fatal
        return {"error": f"{type(error).__name__}: {error}"[:120]}
    results = []
    for item, rgb in pairs:
        rng = np.random.default_rng(G.seed_of(item.key))
        found, tally = GS.coloured_variants(rgb, item.vox, rng, keep=keep, max_cells=max_cells, tolerance=tolerance)
        goals = [{"scale": s, "rot": k, "vox": v, "paint": p} for s, k, v, p in found]
        item.vox = None
        results.append({"item": item, "goals": goals, "tally": tally})
    return {"results": results}


def run(name, procs, limit, keep, max_cells, tolerance=None, out_root=OUT):
    LIMIT[0] = limit
    started = time.time()
    lines, cell_lists, rgb_lists = [], [], []
    stats, tally, errors, per_kind, licences = Counter(), Counter(), Counter(), Counter(), Counter()
    with_goal = 0
    seen = set()

    def tasks():
        for i, task in enumerate(TABLE[name][0]()):
            if limit and i >= limit:
                break
            yield (name, task, keep, max_cells, tolerance)

    with Pool(procs) as pool:
        for n, out in enumerate(pool.imap(work, tasks(), chunksize=4), 1):
            stats["tasks"] += 1
            if "error" in out:
                errors[out["error"]] += 1
                continue
            for meta in out["results"]:
                item = meta["item"]
                stats["items"] += 1
                tally.update(meta["tally"])
                if meta["goals"]:
                    with_goal += 1
                for goal in meta["goals"]:
                    key = (item.text, goal["vox"].shape, goal["vox"].tobytes())
                    if key in seen:
                        stats["duplicates"] += 1
                        continue
                    seen.add(key)
                    line = G.entry_line(item, goal, f"{name}:{len(lines)}")
                    lines.append(line)
                    cell_lists.append(G.vox_to_cells(goal["vox"]))
                    rgb_lists.append(GS.cell_colours(goal["vox"], goal["paint"]))
                    lic = item.extra.get("licence")
                    licences[",".join(lic) if isinstance(lic, list) else str(lic)] += 1
            if n % 500 == 0:
                log({"source": name, "tasks": stats["tasks"], "items": stats["items"], "accepted": len(lines), "goal_library_v2_added": len(lines)})
    GS.save_coloured(name, lines, cell_lists, None if name in NO_COLOUR else rgb_lists, out_root if not limit else Path("/private/tmp/goal_sources_test"))
    counts = [line["cells"] for line in lines]
    root = out_root if not limit else Path("/private/tmp/goal_sources_test")
    summary = {
        "source": name,
        "tasks": stats["tasks"],
        "items": stats["items"],
        "items_with_goal": with_goal,
        "accepted": len(lines),
        "goal_library_v2_added": len(lines),
        "duplicates_dropped": stats["duplicates"],
        "cells_hist_bins": {"4-115": sum(4 <= c <= 115 for c in counts), "116-250": sum(116 <= c <= 250 for c in counts), "251-500": sum(251 <= c <= 500 for c in counts), "501-800": sum(501 <= c <= 800 for c in counts)},
        "licences": dict(licences),
        "rejects": dict(tally),
        "errors": dict(errors),
        "library_bytes": sum(os.path.getsize(root / f"{name}.{ext}") for ext in ("npz", "jsonl")),
        "seconds": round(time.time() - started),
    }
    log(summary)
    print(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=sorted(TABLE))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--procs", type=int, default=3)
    parser.add_argument("--max-cells", type=int, default=G.MAX_CELLS)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--tolerance", type=float, default=None, help="skip a keep target when no scale has a count within this factor of it")
    parser.add_argument("--keep-targets", type=int, nargs="+", default=list(G.KEEP_TARGETS))
    args = parser.parse_args()
    run(args.source, args.procs, args.limit, tuple(args.keep_targets), args.max_cells, args.tolerance, Path(args.out))


if __name__ == "__main__":
    main()
