"""Teacher-forced action probe of a think checkpoint on held-out ThinkData examples: for every robot
voxel, the policy's action distribution against the expert's code, by episode step.

    python scripts/vla_action_probe.py --checkpoint runs/bugcatch-08b/policy_step1200.pt [--examples 192]

Answers whether the policy knows which robots move (then argmax per robot, which picks STAY whenever
no single move beats it, hides that knowledge) or does not (then the action head itself is weak).
Prints one JSON line per step bucket and one for all steps.
"""

import argparse
import dataclasses
import glob
import json
import sys

import numpy as np
import torch

from beebot.vla.voxel import think
from beebot.vla.voxel.lm import patchify3
from beebot.vla.voxel.think_data import ThinkData
from beebot.vla.voxel.views import ViewSpec, scene_views, side_azimuth
from beebot.vla.voxel.world import STAY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--examples", type=int, default=192)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--stores", default="data/vla/voxel_full/s0,data/vla/voxel_full/s1")
    args = parser.parse_args()
    policy, tokenizer = think.load_policy(args.checkpoint, args.device, torch.float32)
    policy.eval()
    spec = ViewSpec()

    def views(obs, rng, side=None):  # side: a camera at that board side (labels said only in the world's frame ask for the front)
        return scene_views(obs, rng, spec if side is None else dataclasses.replace(spec, azimuth=side_azimuth(side, rng)))

    data = ThinkData(args.stores.split(","), sorted(glob.glob("data/vla/haiku/run-s[01]")), tokenizer, 0, mismatch="drop", views=views)
    rng = np.random.default_rng(7)
    examples = []
    while len(examples) < args.examples:  # held-out episodes, acting steps only
        ex = data.draw(data.held_pool, rng)
        if ex.act:
            examples.append(ex)
    rows = []  # (t, target moves, argmax moves, P(target), P(stay), argmax correct)
    with torch.no_grad():
        for k in range(0, len(examples), args.batch):
            batch = examples[k : k + args.batch]
            logits = policy.slot_logits(batch).float().cpu()  # (B * n, PATCH^3, actions)
            probs = logits.softmax(-1)
            n = logits.shape[0] // len(batch)
            for b, ex in enumerate(batch):
                codes = patchify3(ex.codes.view(1, 1, *ex.codes.shape).float()).round().long().reshape(-1)
                robots = (patchify3(ex.obs[:1].unsqueeze(0).float()) > 0.5).reshape(-1)
                p = probs[b * n : (b + 1) * n].reshape(-1, probs.shape[-1])[robots]
                c = codes[robots]
                top = p.argmax(-1)
                for i in range(len(c)):
                    rows.append((ex.t, int(c[i] != STAY), int(top[i] != STAY), float(p[i, c[i]]), float(p[i, STAY]), int(top[i] == c[i])))
            print(json.dumps({"run": "action-probe", "examples": min(k + args.batch, len(examples))}), flush=True)
    rows = np.array(rows)
    for name, lo, hi in (("all", 0, 10**9), ("t0", 0, 1), ("t1-11", 1, 12), ("t12-35", 12, 36), ("t36+", 36, 10**9)):
        m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        if not m.any():
            continue
        r = rows[m]
        movers = r[:, 1] == 1
        print(json.dumps({
            "run": "action-probe", "steps": name, "robots": int(m.sum()),
            "target_moving": round(r[:, 1].mean(), 3),
            "argmax_moving": round(r[:, 2].mean(), 3),
            "sampled_moving": round(1 - r[:, 4].mean(), 3),
            "movers_p_target": round(r[movers, 3].mean(), 3) if movers.any() else None,
            "movers_p_stay": round(r[movers, 4].mean(), 3) if movers.any() else None,
            "stayers_p_stay": round(r[~movers, 4].mean(), 3) if (~movers).any() else None,
            "movers_argmax_correct": round(r[movers, 5].mean(), 3) if movers.any() else None,
            "accuracy": round(r[:, 5].mean(), 3),
        }), flush=True)


if __name__ == "__main__":
    sys.exit(main())
