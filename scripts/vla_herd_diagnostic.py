"""Why herding fails: the policy on herd tasks, by kind (nudge vs into a box), tracking the herded
cheese per step: how far it moved along the asked direction (nudge) or how close it came to the
box (into), whether it was eaten and when, and whether it had already met the goal before being
eaten (overshoot: the policy cannot see where the cheese started, so a nudge has no visible end).

    uv run python scripts/vla_herd_diagnostic.py --init_run runs/lm-expert-sft --checkpoint policy_latest.pt
"""

import argparse
import json

import numpy as np
import torch

from beebot.vla import tasks as T
from beebot.vla.lm_policy import PATCH
from beebot.vla.lm_rl import RLConfig, lm_sampler, load_policy, tokens
from beebot.vla.world import COLOUR_NAMES, World, cheese_side, observe, step


@torch.no_grad()
def trace(policy, tokenizer, pairs, steps, guidance, device):
    scenes, batch = [s for s, _ in pairs], [t for _, t in pairs]
    world, runners = T.stack(scenes)
    world = world.to(device)
    ids, mask = tokens(tokenizer, [t.text for t in batch], device)
    sample = lm_sampler(policy, guidance, T.SIZE // PATCH)
    side = cheese_side(T.SIZE)
    slots = []
    for s, t in pairs:
        colour = t.spec["nudge"][0] if "nudge" in t.spec else t.spec["into"][0][0]
        slots.append([COLOUR_NAMES[c] for c in s["cheese_colours"]].index(colour))
    pos = [[world.cheese_pos[i, k].tolist()] for i, k in enumerate(slots)]
    alive = [[bool(world.cheese_alive[i, k])] for i, k in enumerate(slots)]
    for _ in range(steps):
        _, field = sample(observe(world), (ids, mask))
        world, _ = step(world, field)
        for i, k in enumerate(slots):
            pos[i].append(world.cheese_pos[i, k].tolist())
            alive[i].append(bool(world.cheese_alive[i, k]))
    rows = []
    for (s, t), p, a in zip(pairs, pos, alive):
        p = np.array(p, dtype=float)
        eaten_at = next((j for j, x in enumerate(a) if not x), None)
        upto = eaten_at if eaten_at is not None else len(a)
        if "nudge" in t.spec:
            dy, dx = T.DIRECTIONS[t.spec["nudge"][1]]
            along = (p[:upto] - p[0]) @ np.array([dy, dx])
            met_before = bool((along >= 4).any())
            rows.append({"kind": "nudge", "max_along": float(along.max()), "end_along": float(along[-1]), "eaten_at": eaten_at, "met_then_eaten": met_before and eaten_at is not None, "success": met_before and eaten_at is None and along[-1] >= 4})
        else:
            y, x, h, w = t.spec["into"][1]
            inside = [T.cheese_in_box(tuple(int(v) for v in q), (y, x, h, w), side) for q in p[:upto]]
            centre = np.array([y + h / 2, x + w / 2])
            dist = np.linalg.norm(p[:upto] + side / 2 - centre, axis=1)
            rows.append({"kind": "into", "start_dist": float(dist[0]), "min_dist": float(dist.min()), "ever_inside": bool(any(inside)), "eaten_at": eaten_at, "met_then_eaten": bool(any(inside)) and eaten_at is not None, "success": bool(inside[-1]) and eaten_at is None})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_run", default="runs/lm-expert-sft")
    parser.add_argument("--checkpoint", default="policy_latest.pt")
    parser.add_argument("--tasks", type=int, default=48)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()
    device = torch.device(args.device)
    policy, tokenizer = load_policy(RLConfig(init_run=args.init_run, checkpoint=args.checkpoint), device)
    policy.eval()
    rng = np.random.default_rng(args.seed)
    pairs = [T.sample_task(rng, "train", family="herd") for _ in range(args.tasks)]
    rows = []
    for i in range(0, len(pairs), 24):
        rows += trace(policy, tokenizer, pairs[i : i + 24], args.steps, args.guidance, device)
    for kind in ("nudge", "into"):
        r = [x for x in rows if x["kind"] == kind]
        if not r:
            continue
        summary = {"n": len(r), "success": np.mean([x["success"] for x in r]), "eaten": np.mean([x["eaten_at"] is not None for x in r]), "met_then_eaten": np.mean([x["met_then_eaten"] for x in r]), "median_eaten_step": float(np.median([x["eaten_at"] for x in r if x["eaten_at"] is not None])) if any(x["eaten_at"] is not None for x in r) else None}
        if kind == "nudge":
            summary.update({"mean_max_along": np.mean([x["max_along"] for x in r]), "share_reached_4": np.mean([x["max_along"] >= 4 for x in r])})
        else:
            summary.update({"mean_start_dist": np.mean([x["start_dist"] for x in r]), "mean_min_dist": np.mean([x["min_dist"] for x in r]), "ever_inside": np.mean([x["ever_inside"] for x in r])})
        print(kind, json.dumps({k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in summary.items()}), flush=True)


if __name__ == "__main__":
    main()
