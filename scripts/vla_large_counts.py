"""Robot counts of 10 or more in training text (thoughts, switch lines, replies) as ThinkData
writes it (after think_data.visible). The model cannot count that many robots from images.

    python scripts/vla_large_counts.py [--haiku 'data/vla/haiku/run-s*,data/vla/haiku/build-s*,data/vla/haiku/frame']
"""

import argparse
import glob
import json
import sys
from pathlib import Path

from beebot.vla.voxel.think_data import counted, large_counts, visible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--haiku", default="data/vla/haiku/run-s*,data/vla/haiku/build-s*,data/vla/haiku/frame")
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args()
    rows = hits = 0
    shown = []
    for pattern in args.haiku.split(","):
        for directory in sorted(glob.glob(pattern)):
            for line in (Path(directory) / "verified.jsonl").read_text().splitlines():
                pair = json.loads(line)
                instruction = pair.get("instruction") or pair.get("text", "")
                texts = [visible(pair.get("thought", ""), instruction, pair.get("spec"))] + [visible(s, instruction, pair.get("spec")) for s in pair.get("switches") or []] + [pair.get("reply", "")]
                if pair.get("rung") == "refusal" and counted(pair):
                    continue  # ThinkData drops these
                rows += 1
                found = [m for text in texts for m in large_counts(text, instruction)]
                if found:
                    hits += 1
                    if len(shown) < args.show:
                        shown.append(found)
    for found in shown:
        print("  ", found)
    print(json.dumps({"run": "large-counts", "rows": rows, "with_large_counts": hits}))


if __name__ == "__main__":
    sys.exit(main())
