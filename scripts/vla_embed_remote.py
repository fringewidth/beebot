"""Embed caption-set texts on a Kaggle GPU instead of locally.

    uv run python scripts/vla_embed_remote.py export --captions data/vla/captions/play_g32_robots.jsonl
    uv run python scripts/vla_embed_remote.py push      # upload texts, start the kernel
    uv run python scripts/vla_embed_remote.py status
    uv run python scripts/vla_embed_remote.py pull      # download, check against local, merge into the cache

export writes the variant texts the embedding cache lacks; pull checks a sample against the local
encoder (cosine above MIN_COSINE) before merging, so a later vla-caption-set build finds every
text cached. Needs Kaggle credentials in ~/.kaggle and a phone-verified account (the kernel downloads the
model, so it needs internet).
"""

import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from beebot.vla.caption_set import EMBED_CACHE, variant_rows

WORK = Path("data/vla/captions/kaggle")
KERNEL_SOURCE = Path(__file__).with_name("kaggle_embed") / "kernel.py"
DATASET_SLUG = "beebot-caption-texts"
KERNEL_SLUG = "beebot-embed"
MIN_COSINE = 0.999


def username():
    view = subprocess.run(["uvx", "--with", "six", "kaggle", "config", "view"], capture_output=True, text=True, check=True).stdout
    return re.search(r"- username: (\S+)", view).group(1)


def kaggle(*args):
    print("kaggle", " ".join(args), flush=True)
    return subprocess.run(["uvx", "--with", "six", "kaggle", *args], check=True)


def export(captions):
    cache = torch.load(EMBED_CACHE, weights_only=False) if EMBED_CACHE.exists() else {}
    texts = sorted({t for t in variant_rows(captions)["texts"] if t not in cache})
    dataset = WORK / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "texts.json").write_text(json.dumps(texts))
    meta = {"title": DATASET_SLUG, "id": f"{username()}/{DATASET_SLUG}", "licenses": [{"name": "CC0-1.0"}]}
    (dataset / "dataset-metadata.json").write_text(json.dumps(meta))
    print(f"{len(texts)} texts to embed ({len(cache)} already cached)")


def push():
    user = username()
    dataset = WORK / "dataset"
    exists = subprocess.run(["uvx", "--with", "six", "kaggle", "datasets", "status", f"{user}/{DATASET_SLUG}"], capture_output=True).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(dataset), "-m", "new texts", "--delete-old-versions")
    else:
        kaggle("datasets", "create", "-p", str(dataset))
    kernel = WORK / "kernel"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.py").write_text(KERNEL_SOURCE.read_text())
    meta = {
        "id": f"{user}/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"],
    }
    (kernel / "kernel-metadata.json").write_text(json.dumps(meta))
    kaggle("kernels", "push", "-p", str(kernel))


def status():
    kaggle("kernels", "status", f"{username()}/{KERNEL_SLUG}")


def pull():
    from beebot.vla.text import TextEncoder

    out = WORK / "output"
    out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{username()}/{KERNEL_SLUG}", "-p", str(out))
    texts = json.loads((out / "texts.json").read_text())
    emb = torch.from_numpy(np.load(out / "embeddings.npy"))
    sample = random.Random(0).sample(range(len(texts)), min(64, len(texts)))
    local = TextEncoder(device="mps").embed([texts[i] for i in sample])
    cosine = (local * emb[sample].float()).sum(-1)
    print(f"cosine to local on {len(sample)} texts: min {cosine.min():.5f}, mean {cosine.mean():.5f}")
    if cosine.min() < MIN_COSINE:
        sys.exit(f"remote embeddings disagree with local (min cosine {cosine.min():.5f} < {MIN_COSINE}); not merging")
    cache = torch.load(EMBED_CACHE, weights_only=False) if EMBED_CACHE.exists() else {}
    cache.update(zip(texts, emb))
    torch.save(cache, EMBED_CACHE)
    print(f"merged {len(texts)} embeddings; cache holds {len(cache)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["export", "push", "status", "pull"])
    parser.add_argument("--captions", default="data/vla/captions/play_g32_robots.jsonl")
    args = parser.parse_args()
    if args.command == "export":
        export(args.captions)
    elif args.command == "push":
        push()
    elif args.command == "status":
        status()
    else:
        pull()


if __name__ == "__main__":
    main()
