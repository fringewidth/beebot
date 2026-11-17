"""Summaries of a design search (search.py output).

    uv run python -m beebot.hardware.analyze runs/hardware/grid2.jsonl
"""

import argparse
import json
from collections import defaultdict


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def kind(row):
    if "side" not in row:
        return "reference"
    return "flyer" if row["flight_share"] > 0 else "crawler"


def table(rows, keys):
    out = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    rows = load(args.path)
    keys = ["name", "work", "strength", "torque", "thrust", "cheese", "mass_g"]
    refs = [r for r in rows if kind(r) == "reference"]
    print("References\n")
    print(table(sorted(refs, key=lambda r: -r["work"]), ["name", "work"]))
    designs = [r for r in rows if kind(r) != "reference"]
    print(f"\nTop {args.top} of {len(designs)} designs\n")
    print(table(sorted(designs, key=lambda r: -r["work"])[: args.top], keys))
    print("\nBest per size, locomotion and connector\n")
    best = defaultdict(lambda: None)
    for r in designs:
        key = (r["side"], kind(r), r["connector"] if r["connector_share"] > 0 else "none")
        if best[key] is None or r["work"] > best[key]["work"]:
            best[key] = r
    summary = [dict(best[k], group=f"{k[0] * 100:g} cm {k[1]} {k[2]}") for k in sorted(best)]
    print(table(summary, ["group", "work", "name", "strength", "torque", "thrust", "cheese"]))
    print("\nFamilies of the best in each size\n")
    for side in sorted({r["side"] for r in designs}):
        top = max((r for r in designs if r["side"] == side), key=lambda r: r["work"])
        print(f"{side * 100:g} cm  {top['name']}  {top['families']}")


if __name__ == "__main__":
    main()
