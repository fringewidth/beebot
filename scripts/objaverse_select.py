"""Pick Objaverse 1.0 objects to voxelise: LVIS-named, licence CC0 / CC-BY / CC-BY-SA, low face count, small glb.

    nice -n 10 .venv/bin/python -I scripts/objaverse_select.py data/goal_sources/objaverse [--max-faces 6000]

Reads lvis-annotations.json.gz, object-paths.json.gz and cap3d_captions.json.gz from the directory, downloads each
metadata shard (metadata/<shard>.json.gz from the allenai/objaverse dataset repo; a shard is about 3.6 MB), keeps
the objects named by LVIS and drops the shard. Writes select.jsonl: uid, shard, path, lvis category, caption
(Cap3D, ODC-By 1.0), licence, faces, glb bytes. Objects with animations or other licences are dropped (counted).
"""

import argparse
import gzip
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

HF = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"
OK = {"by": "CC-BY", "by-sa": "CC-BY-SA", "cc0": "CC0"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--max-faces", type=int, default=6000)
    parser.add_argument("--max-bytes", type=int, default=6_000_000)
    args = parser.parse_args()
    root = Path(args.root)
    lvis = json.load(gzip.open(root / "lvis-annotations.json.gz"))
    paths = json.load(gzip.open(root / "object-paths.json.gz"))
    captions = json.load(gzip.open(root / "cap3d_captions.json.gz"))
    category = {}
    for name, uids in lvis.items():
        for uid in uids:
            category[uid] = name
    shards = sorted({paths[uid].split("/")[1] for uid in category})
    drop = Counter()
    kept = Counter()
    out = open(root / "select.jsonl", "w")
    for shard in shards:
        target = root / "meta" / f"{shard}.json.gz"
        target.parent.mkdir(exist_ok=True)
        subprocess.run(["curl", "-sL", "-o", str(target), f"{HF}/metadata/{shard}.json.gz"], check=True)
        try:
            meta = json.load(gzip.open(target))
        except Exception as error:
            print(json.dumps({"shard": shard, "error": str(error)[:80]}), file=sys.stderr)
            target.unlink(missing_ok=True)
            continue
        for uid, entry in meta.items():
            if uid not in category:
                continue
            licence = OK.get(entry.get("license"))
            if licence is None:
                drop["licence " + str(entry.get("license"))] += 1
                continue
            glb = (entry.get("archives") or {}).get("glb") or {}
            faces = entry.get("faceCount") or glb.get("faceCount") or 0
            size = glb.get("size") or 0
            if entry.get("animationCount"):
                drop["animated"] += 1
                continue
            if not faces or faces > args.max_faces or not size or size > args.max_bytes:
                drop["size"] += 1
                continue
            kept[licence] += 1
            out.write(json.dumps({"uid": uid, "path": paths[uid], "lvis": category[uid], "name": entry.get("name", ""), "caption": captions.get(uid, ""), "license": licence, "faces": faces, "bytes": size}) + "\n")
        target.unlink()
        out.flush()
    print(json.dumps({"kept": dict(kept), "dropped": dict(drop)}))


if __name__ == "__main__":
    main()
