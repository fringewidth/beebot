"""Captioned windows ready for training: each caption in a few augmented variants, embedded once
by the frozen text encoder.

    uv run vla-caption-set --captions data/vla/captions/play.jsonl --out data/vla/captions/play_set.pt

A variant is (caption, colour permutation, rotation or reflection). The identity variant always
exists; up to VARIANTS - 1 more are drawn at random and kept only where the rewriter can make the
caption match the transformed scene (a caption that refuses colour swaps gets the identity
permutation, one that refuses rotations gets group 0). Tags are stripped before embedding.

Saved fields:
- keys: (program id, scene, w0, w1) per window; resolve() turns them into rows of whichever
  play store training loads;
- emb (V, D) float16 embeddings; window (V,) the variant's window; caption (V,) a caption index
  shared by all variants of one caption; perm (V, 4); group (V,);
- rotation_free, colour_free (V,) bool: the caption doesn't change under any rotation or
  reflection (or any colour permutation), so training draws those at random for it;
- texts: the embedded text of each variant (for audits and retrieval checks).
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch

from beebot.vla.caption import TAG, strip_tags
from beebot.vla.caption_aug import rewrite
from beebot.vla.world import PALETTE_NAMES

# Captions that give distances or durations ("six squares up", "two steps left") break the brief's
# rule against numbers; they are dropped. Counts are allowed: "3 groups", and "two squares" when
# squares are cheeses (a count of squares is a distance only when a direction follows).
NUMBER = r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty)"
DIRECTION = r"(up|down|left|right|away|over|further|farther|across|toward|towards|higher|lower)\b"
MEASURED = re.compile(
    NUMBER + r"\s+(steps?|cells?|rows?|columns?|spaces?)\b|" + NUMBER + r"\s+squares?\s+(\w+\s+){0,2}?" + DIRECTION, re.I
)
BARE_COLOUR = re.compile(r"\b(" + "|".join(name.lower() for name in PALETTE_NAMES) + r")\b", re.I)


def tag_colours(caption):
    """Wrap palette colour words left outside tags in <c>...</c> (captioners sometimes skip the
    tags; a palette colour word is never anything but a colour here, so tagging it is safe).
    Direction words are left alone: "split up" is not a direction."""
    pieces, last = [], 0
    for m in TAG.finditer(caption):
        pieces.append(BARE_COLOUR.sub(r"<c>\1</c>", caption[last : m.start()]))
        pieces.append(m.group(0))
        last = m.end()
    pieces.append(BARE_COLOUR.sub(r"<c>\1</c>", caption[last:]))
    return "".join(pieces)

VARIANTS = 5


def variants(caption, rng, n=VARIANTS):
    """[(text, perm, group)]: the identity first, then up to n - 1 distinct augmented ones."""
    found = [(strip_tags(caption), (0, 1, 2, 3), 0)]
    seen = {found[0][0]}
    for _ in range(4 * n):
        if len(found) >= n:
            break
        perm = tuple(rng.sample(range(4), 4))
        group = rng.randrange(8)
        k, flip = group % 4, group >= 4
        text = rewrite(caption, k, flip, list(perm))
        if text is None:
            text = rewrite(caption, k, flip, None)
            perm = (0, 1, 2, 3)
        if text is None:
            text = rewrite(caption, 0, False, list(perm))
            group = 0
        if text is None:
            continue
        plain = strip_tags(text)
        if plain not in seen:
            seen.add(plain)
            found.append((plain, perm, group))
    return found


def invariances(caption):
    """(rotation-free, colour-free): the caption reads the same under every rotation and reflection,
    or under every colour permutation, so training can draw those at random for it."""
    rotation_free = all(rewrite(caption, g % 4, g >= 4) == caption for g in range(1, 8))
    colour_free = rewrite(caption, colour_map=[1, 2, 3, 0]) == caption  # a 4-cycle renames every colour
    return rotation_free, colour_free


EMBED_CACHE = Path("data/vla/captions/embed_cache.pt")


def cached_embed(encoder, texts, cache_path=EMBED_CACHE):
    """(N, D) float16 embeddings, reusing a text -> embedding cache on disk across builds."""
    cache = torch.load(cache_path, weights_only=False) if cache_path.exists() else {}
    missing = sorted({t for t in texts if t not in cache})
    if missing:
        print(f"embedding {len(missing)} new texts ({len(cache)} cached)", flush=True)
        for i in range(0, len(missing), 1024):
            part = missing[i : i + 1024]
            cache.update(zip(part, encoder.embed(part).half()))
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cache, cache_path)
    return torch.stack([cache[t] for t in texts])


def variant_rows(captions_path, seed=0):
    """Everything in a caption set except the embeddings: the variant texts and their fields."""
    rng = random.Random(seed)
    keys, texts, window, caption, perm, group, free, tier = [], [], [], [], [], [], [], []
    n_captions = 0
    dropped = 0
    for line in Path(captions_path).read_text().splitlines():
        record = json.loads(line)
        program_id, scene, span = record["id"].rsplit(":", 2)
        w0, w1 = (int(v) for v in span.split("-"))
        keys.append((program_id, int(scene), w0, w1))
        for level, text_tagged in enumerate(record["captions"]):
            if MEASURED.search(strip_tags(text_tagged)):
                dropped += 1
                continue
            text_tagged = tag_colours(text_tagged)
            invariant = invariances(text_tagged)
            for text, p, g in variants(text_tagged, rng):
                free.append(invariant)
                texts.append(text)
                window.append(len(keys) - 1)
                caption.append(n_captions)
                perm.append(p)
                group.append(g)
                tier.append(level)
            n_captions += 1
    print(f"dropped {dropped} captions that give distances or durations", flush=True)
    return {
        "keys": keys,
        "window": torch.tensor(window, dtype=torch.long),
        "caption": torch.tensor(caption, dtype=torch.long),
        "perm": torch.tensor(perm, dtype=torch.long),
        "group": torch.tensor(group, dtype=torch.long),
        "tier": torch.tensor(tier, dtype=torch.long),  # index into caption.TIERS
        "rotation_free": torch.tensor([f[0] for f in free]),
        "colour_free": torch.tensor([f[1] for f in free]),
        "texts": texts,
    }


def build(captions_path, encoder, seed=0):
    rows = variant_rows(captions_path, seed)
    return {**rows, "emb": cached_embed(encoder, rows["texts"])}


def resolve(caption_set, store):
    """(windows (W, 3) long rows of store, or -1 where the store lacks that scene; program (W,) long
    index of each window's program among the store's programs)."""
    row_of = {(program_id, scene): e for e, (program_id, _, scene) in enumerate(store["source"])}
    programs = {p: i for i, p in enumerate(sorted({program_id for program_id, _, _ in store["source"]}))}
    windows, program = [], []
    for program_id, scene, w0, w1 in caption_set["keys"]:
        e = row_of.get((program_id, scene), -1)
        windows.append((e, w0, w1))
        program.append(programs.get(program_id, -1))
    return torch.tensor(windows, dtype=torch.long), torch.tensor(program, dtype=torch.long)


def main():
    from beebot.vla.text import TextEncoder

    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    caption_set = build(args.captions, TextEncoder(device=args.device), args.seed)
    augmented = (caption_set["group"] != 0) | (caption_set["perm"] != torch.arange(4)).any(1)
    print(f"{len(caption_set['keys'])} windows, {int(caption_set['caption'].max()) + 1} captions, {len(caption_set['texts'])} variants ({augmented.float().mean():.0%} augmented)")
    torch.save(caption_set, args.out)


if __name__ == "__main__":
    main()
