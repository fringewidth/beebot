"""Audit LLM captions for weirdness: numbers, sizes or timing, overlong captions, repeated
openings and closings in the casual tier, and untagged words, split by provider and tier.

    uv run python scripts/vla_caption_audit.py data/vla/captions/play_g32_robots.jsonl [--since ISO time]
"""

import argparse
import json
import re
from collections import Counter

from beebot.vla.caption_set import MEASURED

TIERS = ("short", "medium", "detailed", "casual")

WORD_LIMITS = {"short": 8, "medium": 25, "detailed": 45, "casual": 25}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--since", default="")
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()
    records = [json.loads(line) for line in open(args.path) if line.strip()]
    records = [r for r in records if r["time"] >= args.since and len(r["plain"]) == len(TIERS)]
    print(f"{len(records)} four-tier records since {args.since or 'the start'}")
    flagged = {"numeric": [], "long": []}
    for provider in ("haiku", "luna"):
        mine = [r for r in records if r["provider"] == provider]
        if not mine:
            continue
        print(f"\n{provider}: {len(mine)} records")
        for i, tier in enumerate(TIERS):
            texts = [r["plain"][i] for r in mine]
            words = [len(t.split()) for t in texts]
            numeric = [t for t in texts if MEASURED.search(t) or re.search(r"\d+\s*%|percent", t)]
            long = [t for t in texts if len(t.split()) > WORD_LIMITS[tier]]
            untagged = sum(bool(r["tag_problems"][i]) for r in mine)
            flagged["numeric"] += numeric
            flagged["long"] += long
            print(f"  {tier:9s} words {sum(words) / len(words):5.1f}  numeric {len(numeric):3d}  long {len(long):3d}  untagged {untagged / len(mine):4.0%}")
        casual = [r["plain"][3].lower() for r in mine]
        openings = Counter(t.split()[0] for t in casual if t.split())
        closings = Counter(t.split()[-1].strip(".!?") for t in casual if t.split())
        print(f"  casual openings {openings.most_common(4)}  closings {closings.most_common(4)}")
        names = Counter(w for t in casual for w in re.findall(r"\b(cheese|cheeses|block|blocks|square|squares|piece|pieces|one|ones|thing|things|target|targets)\b", t))
        print(f"  casual object words {names.most_common(6)}")
    for kind, texts in flagged.items():
        if texts:
            print(f"\nexamples, {kind}:")
            for t in texts[: args.show]:
                print("  ", t)


if __name__ == "__main__":
    main()
