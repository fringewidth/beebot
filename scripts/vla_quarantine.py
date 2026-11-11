"""Episodes and Haiku labels that ThinkData should skip, from the data bug sweep of 2026-11-11.

    python scripts/vla_quarantine.py [--rescored DIR] [--out runs/quarantine.json]

Episodes (think_data.store_key of the store, attempts index n), by reason:
  gravity_shape   formation shapes that need gravity off (cube, sphere, ring, plus, standing
                  letters) drawn before demos.draws turned gravity on under a rule: the expert can
                  only pile robots (B1)
  done_at_start   the task already passes at step 0, or a switch's later instruction already holds
                  when it arrives (B2, B9); from the sweep's re-scoring (rows with "noop"), which
                  covers each store up to its known draw divergence
  short_tower     a tower job that stops short of the asked height (B3), judged by the job's own
                  share (suite.tower_shares) from the stored frames, in any task with a tower
                  job: a tower (also with a selector), a parallel task's, counting's, a
                  conditional's tower branch, a switch's last instruction
  held_out_word   a build object named with a held-out word ("crate", B10)
Labels (Haiku pair_id), by reason:
  hidden_weight   thoughts that name the heavy object, which no image shows (B7)
"""

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path

from beebot.vla.voxel import suite as S
from beebot.vla.voxel.play_store import load
from beebot.vla.voxel.think_data import store_key

FLOATING = {"cube", "sphere", "ring", "plus"}
PASS = {"formation": 0.5}  # the suite's pass marks (others 0.75)
HELD_WORDS = re.compile(r"\bcrates?\b")
NOTHING_TO_DO = re.compile(r"(absent|none|^stay|>stay)$")  # kinds whose right answer is to do nothing


def snapshot(store, e, t):
    """suite.snapshot of a stored episode's frame t (negative: from the end)."""
    return {"mass": store["mass"][e, t].numpy().astype("float32"), "cheese_pos": store["cheese_pos"][e, t].numpy().astype("int64"), "cheese_alive": store["cheese_alive"][e, t].numpy().copy()}


def short_tower(store, e):
    """Whether a kept episode's last instruction has a tower job that ended below the asked
    height (suite.tower_shares, from the instruction's first frame to the episode's last)."""
    last = store["instructions"][e][-1]
    static = {"walls": store["walls"][e].numpy(), "cheese_color": store["cheese_color"][e].tolist()}
    shares = S.tower_shares(last["record"]["spec"], static, snapshot(store, e, int(last["t"])), snapshot(store, e, -1))
    return any(share < 1.0 for share in shares)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stores", default="data/vla/voxel_full/s*,data/vla/voxel_build/s*")
    parser.add_argument("--haiku", default="data/vla/haiku/run-s*,data/vla/haiku/build-s*")
    parser.add_argument("--rescored", default="", help="directory of the sweep's rescore.py output, s<seed>.jsonl per voxel_full store")
    parser.add_argument("--out", default="runs/quarantine.json")
    args = parser.parse_args()
    episodes, counts = [], collections.Counter()
    for pattern in args.stores.split(","):
        for path in sorted(glob.glob(pattern)):
            store = load(path)
            rule = store["config"].get("rule")
            attempts = [json.loads(line) for line in (Path(path) / "attempts.jsonl").read_text().splitlines() if line.strip()]
            rescored = {}
            seed_file = Path(args.rescored) / f"s{store['config']['seed']}.jsonl" if args.rescored and "voxel_full" in path else None
            if seed_file and seed_file.exists():
                for line in seed_file.read_text().splitlines():
                    row = json.loads(line)
                    if not row.get("diverged"):
                        rescored[row["index"]] = row
            for n, row in enumerate(attempts):
                e = row["e"]
                if e < 0:
                    continue
                spec = store["instructions"][e][0]["record"].get("spec", {})
                reasons = []
                if rule is not None and row["family"] == "formation" and (spec.get("shape") in FLOATING or spec.get("standing")):
                    reasons.append("gravity_shape")
                checked = rescored.get(n)
                if checked is not None and checked.get("noop", 0) >= PASS.get(row["family"], 0.75) and not NOTHING_TO_DO.search(row["kind"]):
                    reasons.append("done_at_start")
                if "tower" in row["kind"] and short_tower(store, e):
                    reasons.append("short_tower")
                if row["family"] == "build" and HELD_WORDS.search(" ".join(entry["text"] for entry in store["instructions"][e])):
                    reasons.append("held_out_word")
                for reason in reasons:
                    counts[reason] += 1
                if reasons:
                    episodes.append([store_key(store["config"]), n, reasons])
    labels = []
    for pattern in args.haiku.split(","):
        for directory in sorted(glob.glob(pattern)):
            for line in (Path(directory) / "verified.jsonl").read_text().splitlines():
                pair = json.loads(line)
                if re.search(r"\b(heavy|heavier|heaviest|weigh\w*)\b", pair.get("thought", "")):
                    labels.append([pair["pair_id"], ["hidden_weight"]])
                    counts["hidden_weight"] += 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"episodes": episodes, "labels": labels, "counts": counts}, indent=0))
    print(json.dumps({"run": "quarantine", "episodes": len(episodes), "labels": len(labels), **counts}))


if __name__ == "__main__":
    sys.exit(main())
