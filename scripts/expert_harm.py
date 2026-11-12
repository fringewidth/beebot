"""How often the demo experts' pivots reach, stay and harm under an outcome table, risk-weighing or
blind (demos --outcomes, --blind): plays the demo preset's draws (demos.PRESETS["demo"]) and counts
every pivot world.OutcomeModel.draw decides.

    PYTHONPATH=src python scripts/expert_harm.py --outcomes mujoco_hinge_v1 --scenes 60 [--blind] [--overhangs]

Prints one JSON line: pivots, reached / stayed / harmful per pivot, fell per pivot (robots the grid's
gravity dropped a voxel, counted at every step they drop: a cube left held up by nothing, which the
table's harm does not count), demonstrations kept of tried, the share of episodes passing their
checker, robots stuck off the lattice at the end of a kept demonstration. --overhangs lets the rule
allow moves that leave a cube held up only by the mover (world.overhangs), as before 2026-11-12.
"""

import argparse
import json
import tempfile
from pathlib import Path

import torch

from beebot.vla.voxel import demos as D
from beebot.vla.voxel import world as W


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", choices=W.OUTCOME_MODELS, required=True)
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--scenes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overhangs", action="store_true", help="the rule as before world.overhangs: moves leaving a cube held up only by the mover allowed")
    args = parser.parse_args()
    if args.overhangs:
        W.overhangs = lambda world, table: torch.zeros_like(table)
    torch.set_num_threads(2)
    counts = {"pivots": 0, "reached": 0, "harmful": 0, "fell": 0}
    drawing = [False]  # pivot_class also serves the experts' odds; count only the calls from draw
    pivot_class, draw = W.pivot_class, W.OutcomeModel.draw

    def counted_class(features):
        out = pivot_class(features)
        counts["pivots"] += len(out) if drawing[0] else 0
        return out

    def counted_draw(self, world, codes, rule, ids, noise):
        drawing[0] = True
        out, events = draw(self, world, codes, rule, ids, noise)
        drawing[0] = False
        counts["harmful"] += len(events[0])
        counts["reached"] -= int((out != codes).sum())
        return out, events

    fall = W.fall

    def counted_fall(world):
        on = world.gravity.nonzero().squeeze(1)
        if on.numel():
            sub = world.index(on)
            occupied = sub.mass > 0.5
            counts["fell"] += int((occupied & ~W.supported(sub, occupied)).sum())
        return fall(world)

    W.pivot_class, W.OutcomeModel.draw, W.fall = counted_class, counted_draw, counted_fall
    p = D.PRESETS["demo"]
    with tempfile.TemporaryDirectory() as out:
        D.write(out, args.scenes, args.seed, steps=p["steps"], moves=p["moves"], robots=p["robots"], fit=p["fit"], rule=p["rule"], expert=p["expert"], coverage=p["coverage"], objects=p["objects"], build=p["build"], keep=p["keep"], empty=p["empty"], scaled=p["scaled"], preset="demo", outcomes=args.outcomes, risk=not args.blind, compact=True)
        rows = [json.loads(line) for line in open(Path(out) / "attempts.jsonl")]
        small = torch.load(Path(out) / "small.pt", weights_only=False)
    n = max(counts["pivots"], 1)
    reached = n + counts["reached"]  # pivots less the ones drawn to stay or harm
    stuck = small.get("stuck", [])
    print(json.dumps({
        "outcomes": args.outcomes,
        "risk": not args.blind,
        "overhangs": args.overhangs,
        "scenes": args.scenes,
        "pivots": counts["pivots"],
        "reached": round(reached / n, 4),
        "stayed": round(1 - reached / n - counts["harmful"] / n, 4),
        "harmful": round(counts["harmful"] / n, 4),
        "fell": round(counts["fell"] / n, 4),
        "tried": len(rows),
        "kept": sum(r["e"] >= 0 for r in rows),
        "passing": round(sum(r["score"] >= D.SUCCESS.get(r["family"], 0.75) and not r["unasked"] for r in rows) / max(len(rows), 1), 4),
        "stuck_per_demo": round(sum(sum(t == p["steps"] for t, *_ in s) for s in stuck) / max(len(stuck), 1), 3),  # at the episode's end, kept demonstrations
    }))


if __name__ == "__main__":
    main()
