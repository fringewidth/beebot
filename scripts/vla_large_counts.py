"""Robot counts of 10 or more in training text (thoughts, switch lines, replies) as ThinkData
writes it (think_data.visible_choice: visible on thoughts and lines, the reply without large
counts), found by think_data.large_counts: every form the stripping drops ("54 robots", "a swarm
of seventeen", "the other 12", "all 25", "15 of you", "the 13", "half is about 15", bare counts).
The model cannot count that many robots from images. Refusals that reason from such a count
(think_data.counted) are left out, as ThinkData drops them.

    python scripts/vla_large_counts.py [--haiku 'data/vla/haiku/run-s*,data/vla/haiku/build-s*,data/vla/haiku/frame']
"""

import argparse
import glob
import json
import sys
from pathlib import Path

from beebot.vla.voxel.think_data import counted, large_counts, visible_choice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--haiku", default="data/vla/haiku/run-s*,data/vla/haiku/build-s*,data/vla/haiku/frame")
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args()
    rows = hits = count_refusals = 0
    examples = []
    for pattern in args.haiku.split(","):
        for directory in sorted(glob.glob(pattern)):
            for line in (Path(directory) / "verified.jsonl").read_text().splitlines():
                pair = json.loads(line)
                instruction = pair.get("instruction") or pair.get("text", "")
                if pair.get("rung") == "refusal" and counted(pair):
                    count_refusals += 1
                    continue  # ThinkData drops these
                choice = visible_choice({**pair, "instruction": instruction, "thought": pair.get("thought", "")}, pair.get("spec"))
                texts = [choice["thought"]] + list(choice.get("switches") or []) + [choice.get("reply") or ""]
                rows += 1
                counts = [m for text in texts for m in large_counts(text, instruction)]
                if counts:
                    hits += 1
                    if len(examples) < args.show:
                        examples.append((pair.get("pair_id"), counts))
    for example in examples:
        print("  ", example)
    print(json.dumps({"run": "large-counts", "rows": rows, "with_large_counts": hits, "refusals_dropped": count_refusals}))


if __name__ == "__main__":
    sys.exit(main())
