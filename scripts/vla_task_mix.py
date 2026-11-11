"""Task mix of demonstration stores against the mix their draw options give under demos.draws keep
(each drawn task kept until a scene fits it, contrast tasks with a plan only).

    python scripts/vla_task_mix.py data/vla/voxel_full/s0 [more stores] [--scenes 400] [--out runs/task_mix.json]

For every (family, kind) it prints the share among the stores' episodes (kept demonstrations), the
share among the keep draws of the same options, their ratio, and whether a policy that stands
still passes it (the scorecard's still arm, when --still gives its episodes.jsonl). The JSON output
holds weights[group][family|kind] = keep share / store share within each group of stores with the
same draw options (think_data.mix_group), the per-episode weight that turns a group's mix into its
keep mix (ThinkData's mix argument reads it to sample episodes).
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from beebot.vla.voxel import demos as D
from beebot.vla.voxel import suite as S
from beebot.vla.voxel.think_data import mix_group


def store_mix(store):
    """Counter of family|kind over a store's kept episodes, and its config."""
    config = torch.load(Path(store) / "small.pt", map_location="cpu", weights_only=False)["config"]
    counts = Counter()
    for line in (Path(store) / "attempts.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["e"] >= 0:
            counts[f"{row['family']}|{row['kind']}"] += 1
    return counts, config


def keep_mix(config, scenes):
    """Counter of family|kind over the tasks demos.draws draws with keep for these options."""
    robots = None if config.get("robots") is None else tuple(config["robots"])
    counts = Counter()
    check = bool(config.get("fit")) or robots is not None
    for _, _, _, tasks, _ in D.draws(scenes, int(config["seed"]) + 7919, config.get("chunk", 24), robots, config.get("counts", "uniform"), check, config.get("rule"), bool(config.get("apart")), float(config.get("coverage", 0.0)), float(config.get("objects", 0.0)), float(config.get("build", 0.0)), config.get("only"), keep=True):
        counts.update(f"{t.family}|{S.task_kind(t)}" for t in tasks)
    return counts


def still_passes(path):
    """family|kind -> share of the still arm's episodes that passed (scorecard episodes.jsonl)."""
    seen, passed = Counter(), Counter()
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get("arm") == "still":
            key = f"{row['family']}|{row['kind']}"
            seen[key] += 1
            passed[key] += bool(row.get("passed"))
    return {k: passed[k] / seen[k] for k in seen}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stores", nargs="+")
    parser.add_argument("--scenes", type=int, default=400, help="scenes of keep draws per distinct set of draw options")
    parser.add_argument("--still", default="runs/scorecard3d/expert_still/episodes.jsonl")
    parser.add_argument("--out", default="runs/task_mix.json")
    args = parser.parse_args()
    stored, target = Counter(), Counter()
    groups = {}  # group -> (stored Counter, keep Counter)
    for store in args.stores:
        counts, config = store_mix(store)
        stored += counts
        group = mix_group(config)
        if group not in groups:  # one keep sample per group (its stores differ in seed only)
            groups[group] = (Counter(), keep_mix(config, args.scenes))
        groups[group][0].update(counts)
        print(json.dumps({"run": "task-mix", "store": store, "episodes": sum(counts.values())}), flush=True)
    for mine, theirs in groups.values():  # the overall keep mix weighs each group by its stores' episodes
        n_mine, n_theirs = sum(mine.values()), sum(theirs.values())
        for key, v in theirs.items():
            target[key] += v / n_theirs * n_mine
    still = still_passes(args.still) if Path(args.still).exists() else {}
    n_stored, n_target = sum(stored.values()), sum(target.values())
    rows = []
    for key in sorted(set(stored) | set(target)):
        a, b = stored[key] / n_stored, target[key] / n_target if n_target else 0.0
        rows.append({"key": key, "store": a, "keep": b, "ratio": a / b if b else float("inf"), "still_passes": still.get(key)})
    rows.sort(key=lambda r: -r["store"])
    family = Counter()
    for r in rows:
        family[r["key"].split("|")[0]] += r["store"]
    noop_store = sum(r["store"] for r in rows if (r["still_passes"] or 0) >= 0.5)
    noop_keep = sum(r["keep"] for r in rows if (r["still_passes"] or 0) >= 0.5)
    print(f"{'family|kind':52s} {'store':>7s} {'keep':>7s} {'ratio':>6s} still")
    for r in rows:
        print(f"{r['key']:52s} {r['store']:7.3%} {r['keep']:7.3%} {r['ratio']:6.2f} {'' if r['still_passes'] is None else round(r['still_passes'], 2)}")
    print(f"episodes a still swarm passes (kinds with still pass >= 0.5): store {noop_store:.1%}, keep {noop_keep:.1%}")
    weights = {}
    for group, (mine, theirs) in groups.items():
        n_mine, n_theirs = sum(mine.values()), sum(theirs.values())
        weights[group] = {k: (theirs[k] / n_theirs) / (mine[k] / n_mine) for k in mine}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"stores": args.stores, "scenes": args.scenes, "rows": rows, "weights": weights, "noop_share": {"store": noop_store, "keep": noop_keep}}, indent=1))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
