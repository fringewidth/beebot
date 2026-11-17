"""Probes for one Rivals game, on val, on the CPU (training keeps the GPU).

    uv run rivals-probes <game> [peak]

Uses each finished run's best pair under the fog cap (best_a.pt / best_b.pt) for seeds 0
and 1, or with "peak" each run's best val check with fog allowed (val_<iteration>), where
the swarms still have the skill the fog limit later took apart:
    co-trained:  A and B from the same run, as trained.
    strangers:   A from one seed with B from the other. Swarms that learned general
                 play keep their score; swarms that learned a private convention with
                 their partner lose it.
    blackout:    a co-trained pair where each species reads the other's pheromones as
                 zero (it still reads its own). The drop is what cross-species signals
                 were worth.
Writes runs/v6-rivals-<game>-probes.json (or -probes-peak.json). Probes that need a missing run are skipped.
"""

import json
import sys
from pathlib import Path

import torch

from beebot import rivals
from beebot.train_rivals import load_pair

SEEDS = (0, 1)


def main():
    game = sys.argv[1]
    peak = len(sys.argv) > 2 and sys.argv[2] == "peak"
    device = torch.device("cpu")
    pairs = {}
    for seed in SEEDS:
        run_dir = Path("runs") / f"v6-rivals-{game}-s{seed}"
        if (run_dir / "summary.json").exists():
            config = json.loads((run_dir / "config.json").read_text())
            prefix = "best"
            if peak:
                history = [json.loads(line) for line in (run_dir / "val_history.jsonl").read_text().splitlines()]
                best = min((h for h in history if (run_dir / f"val_{h['iteration']}_a.pt").exists()), key=lambda h: h["val"]["score"])
                prefix = f"val_{best['iteration']}"
                print(json.dumps({"seed": seed, "peak": prefix, "minutes": best["minutes"]}), flush=True)
            pairs[seed] = load_pair(run_dir, prefix, config, device)

    results = {}
    for seed, (a, b) in pairs.items():
        results[f"co-trained s{seed}"] = rivals.evaluate(a, b, device, game, "val")
        results[f"blackout s{seed}"] = rivals.evaluate(a, b, device, game, "val", blackout=True)
        print(json.dumps({"done": f"s{seed}"}), flush=True)
    if len(pairs) == len(SEEDS):
        results["strangers A0+B1"] = rivals.evaluate(pairs[0][0], pairs[1][1], device, game, "val")
        results["strangers A1+B0"] = rivals.evaluate(pairs[1][0], pairs[0][1], device, game, "val")

    out = Path("runs") / f"v6-rivals-{game}-probes{'-peak' if peak else ''}.json"
    out.write_text(json.dumps(results, indent=2))
    for name, scores in results.items():
        cells = " ".join(f"{split} {scores[split]['eaten']:.2f}/fog {scores[split]['fog']:.2f}" for split in ("interp", "extrap", "damage"))
        print(f"{name:18s} score {scores['score']:.3f}  {cells}")


if __name__ == "__main__":
    main()
