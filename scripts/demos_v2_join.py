"""Counts how many Haiku-labelled episodes of a store survive ThinkData's join (mismatch="drop").

    .venv/bin/python scripts/demos_v2_join.py <store> <haiku dir> [<haiku dir> ...] --name full-s0 [--append logs/demos-v2.out]

Prints (and appends) one JSON line: {"store", "episodes", "labels_joined", "labels_dropped", "kept"}.
episodes: kept episodes of the store; labels_joined: episodes that carry Haiku labels (choices,
parts or refusals) in the join before the check, labels_dropped: of those, the ones the check drops
(.mismatches), kept: labelled episodes that survive. Also "dropped_by_family" and "labelled_by_family"
(drops on tower-family episodes show in "construction" and "parallel"/"subset"; the rest are other families).
"""

import argparse
import json
from collections import Counter

from beebot.vla.voxel import think_data as T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("store")
    parser.add_argument("haiku", nargs="+")
    parser.add_argument("--name", required=True)
    parser.add_argument("--append", default=None)
    args = parser.parse_args()
    data = T.ThinkData([args.store], args.haiku, mismatch="drop")
    attempts = [json.loads(line) for line in open(f"{args.store}/attempts.jsonl")]
    dropped = Counter(attempts[n]["family"] + "/" + attempts[n]["kind"] for _, n, _ in data.mismatches)
    kept = [ep for ep in data.episodes if ep.choices or ep.parts or ep.refusals]
    kept_families = Counter(ep.family + "/" + ep.kind for ep in kept)
    total = sum(1 for a in attempts if a["e"] >= 0)
    row = {
        "store": args.name,
        "episodes": total,
        "labels_joined": len(kept) + len(data.mismatches),
        "labels_dropped": len(data.mismatches),
        "kept": len(kept),
        "demos_v2_kept": total,
        "dropped_by_kind": dict(dropped),
        "kept_by_kind": dict(kept_families),
    }
    line = json.dumps(row)
    print(line)
    if args.append:
        with open(args.append, "a") as out:
            out.write(line + "\n")


if __name__ == "__main__":
    main()
