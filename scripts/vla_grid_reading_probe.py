"""How well do the captioners read our ASCII grids? Ask them questions with known answers about
real g32 play frames, rendered at full resolution (the current captioner view) and pooled to 16
(the earlier view), and score the replies.

    uv run python scripts/vla_grid_reading_probe.py [--frames 24] [--per-call 12]

Questions per frame: the region (3x3 thirds) of each cheese, the region holding most robots, the
number of robot groups (8-connected, at least 3 robots), and whether any robot touches each cheese.
"""

import argparse
import glob
import json
import random

import numpy as np

from beebot.vla.caption import call_provider, text_grid
from beebot.vla.world import PALETTE_NAMES, cheese_side

REGIONS = [["top left", "top", "top right"], ["left", "center", "right"], ["bottom left", "bottom", "bottom right"]]
RULES = """Each grid below is one frame of a 32x32 world seen from above (row 0 at the top). Legend:
{legend}
Regions are the 3x3 thirds of the grid: top left, top, top right, left, center, right, bottom left,
bottom, bottom right. A group is a set of robots connected through neighbouring squares (diagonals
count); ignore groups of fewer than 3 robots. A robot touches a cheese if it is on a square next
to the cheese (diagonals count).

For every frame, answer in JSON and nothing else:
{{"frames": [{{"id": "<frame id>", "cheese_region": {{"<colour>": "<region>", ...}},
  "robots_region": "<region holding the most robots>", "groups": <number>,
  "touching": {{"<colour>": true or false, ...}}}}, ...]}}
"""
LEGEND = {
    "full": "one character per square: o robot, # wall, . empty floor, R/G/B/Y a cheese (2x2 squares), r/g/b/y box floor",
    "pooled": "each character is a 2x2 block of squares: digit 1-9 how full of robots it is, # wall, . empty, R/G/B/Y cheese, r/g/b/y box floor",
}


def group_sizes(robots):
    """Sizes of 8-connected robot groups."""
    seen = np.zeros_like(robots)
    sizes = []
    for r0, c0 in zip(*np.nonzero(robots)):
        if seen[r0, c0]:
            continue
        stack, n = [(r0, c0)], 0
        seen[r0, c0] = True
        while stack:
            r, c = stack.pop()
            n += 1
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < robots.shape[0] and 0 <= cc < robots.shape[1] and robots[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
        sizes.append(n)
    return np.array(sizes)


def region(r, c, size):
    return REGIONS[min(2, int(3 * r / size))][min(2, int(3 * c / size))]


def pooled_grid(store, t):
    """The earlier captioner view: 2x2 pooling, robot density as a digit."""
    size, cell = store["mass"].shape[-1], 2
    n = size // cell
    mass = store["mass"][0, t].reshape(n, cell, n, cell).mean(axis=(1, 3))
    walls = store["walls"][0].reshape(n, cell, n, cell).mean(axis=(1, 3)) > 0.5
    rows = [["." for _ in range(n)] for _ in range(n)]
    for (r0, c0, h, w), colour in zip(store["box_rect"][0].tolist(), store["box_color"][0].tolist()):
        if h > 0 and w > 0:
            for r in range(r0 // cell, (r0 + h - 1) // cell + 1):
                for c in range(c0 // cell, (c0 + w - 1) // cell + 1):
                    rows[r][c] = PALETTE_NAMES[colour][0]
    for r, c in zip(*np.nonzero(walls)):
        rows[r][c] = "#"
    for r, c in zip(*np.nonzero(mass > 0.05)):
        rows[r][c] = str(min(9, max(1, round(9 * mass[r, c]))))
    side = cheese_side(size)
    for k, alive in enumerate(store["cheese_alive"][0, t].tolist()):
        if alive:
            r0, c0 = store["cheese_pos"][0, t, k].tolist()
            for r in range(r0 // cell, (r0 + side - 1) // cell + 1):
                for c in range(c0 // cell, (c0 + side - 1) // cell + 1):
                    rows[r][c] = PALETTE_NAMES[store["cheese_color"][0, k]][0].upper()
    return "\n".join("".join(row) for row in rows)


def truth(store, t):
    size = store["mass"].shape[-1]
    robots = store["mass"][0, t] > 0.5
    sizes = group_sizes(robots)
    counts = np.zeros((3, 3))
    for r, c in zip(*np.nonzero(robots)):
        counts[min(2, int(3 * r / size)), min(2, int(3 * c / size))] += 1
    side = cheese_side(size)
    cheese_region, touching = {}, {}
    for k, alive in enumerate(store["cheese_alive"][0, t].tolist()):
        if not alive:
            continue
        name = PALETTE_NAMES[store["cheese_color"][0, k]]
        r0, c0 = store["cheese_pos"][0, t, k].tolist()
        cheese_region[name] = region(r0 + (side - 1) / 2, c0 + (side - 1) / 2, size)
        ring = np.zeros_like(robots)
        ring[max(r0 - 1, 0) : r0 + side + 1, max(c0 - 1, 0) : c0 + side + 1] = True
        touching[name] = bool((robots & ring).any())
    return {
        "cheese_region": cheese_region,
        "robots_region": REGIONS[counts.argmax() // 3][counts.argmax() % 3],
        "groups": int((sizes >= 3).sum()),
        "touching": touching,
    }


def sample_frames(n, seed):
    rng = random.Random(seed)
    dirs = sorted(glob.glob("/Users/hrishik/beebot-play-g32/*/data/*"))
    frames = []
    while len(frames) < n:
        d = rng.choice(dirs)
        z = np.load(d + "/episodes.npz")
        meta = json.load(open(d + "/meta.json"))
        ok = [i for i, err in enumerate(meta["errors"]) if err is None]
        if not ok:
            continue
        i = rng.choice(ok)
        store = {k: z[k][i : i + 1] for k in ("mass", "walls", "cheese_pos", "cheese_alive", "cheese_color", "box_rect", "box_color")}
        store["mass"] = store["mass"].astype(np.float32)
        t = rng.randrange(store["mass"].shape[1])
        if store["cheese_alive"][0, t].any():
            frames.append((f"f{len(frames)}", store, t))
    return frames


def colour_keys(mapping):
    """{colour name: value} from replies keyed 'red', 'Red', 'R' or 'red cheese'."""
    out = {}
    for key, value in (mapping or {}).items():
        word = str(key).strip().lower().split()[0] if str(key).strip() else ""
        name = next((n for n in PALETTE_NAMES if n == word or n[0] == word), word)
        out[name] = value.strip().lower() if isinstance(value, str) else value
    return out


def score(answer, true):
    hits = {}
    answer = {**answer, "cheese_region": colour_keys(answer.get("cheese_region")), "touching": colour_keys(answer.get("touching"))}
    if isinstance(answer.get("robots_region"), str):
        answer["robots_region"] = answer["robots_region"].strip().lower()
    cheeses = true["cheese_region"]
    hits["cheese_region"] = np.mean([answer.get("cheese_region", {}).get(c) == r for c, r in cheeses.items()])
    hits["robots_region"] = float(answer.get("robots_region") == true["robots_region"])
    hits["groups"] = float(answer.get("groups") == true["groups"])
    hits["touching"] = np.mean([answer.get("touching", {}).get(c) == v for c, v in true["touching"].items()])
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--per-call", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--views", default="full,pooled")
    parser.add_argument("--providers", default="haiku,luna")
    parser.add_argument("--effort", default="low")
    args = parser.parse_args()
    import torch

    frames = sample_frames(args.frames, args.seed)
    truths = {fid: truth(store, t) for fid, store, t in frames}
    for view in args.views.split(","):
        for provider in args.providers.split(","):
            results = []
            for start in range(0, len(frames), args.per_call):
                part = frames[start : start + args.per_call]
                grids = []
                for fid, store, t in part:
                    torch_store = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in store.items()}
                    grid = text_grid(torch_store, 0, t) if view == "full" else pooled_grid(store, t)
                    grids.append(f"### Frame {fid}\n```\n{grid}\n```")
                prompt = RULES.format(legend=LEGEND[view]) + "\n\n" + "\n\n".join(grids)
                reply = call_provider(provider, prompt, args.effort)
                with open(f"/private/tmp/claude-501/-Users-hrishik-beebot/64934076-03d9-47b9-8c56-3a54bf73e29c/scratchpad/probe_{view}_{provider}.txt", "a") as f:
                    f.write(reply + "\n----\n")
                try:
                    parsed = json.loads(reply[reply.find("{") : reply.rfind("}") + 1])["frames"]
                except (ValueError, KeyError):
                    parsed = []
                answers = {a.get("id"): a for a in parsed if isinstance(a, dict)}
                results += [score(answers.get(fid, {}), truths[fid]) for fid, _, _ in part]
            means = {k: round(float(np.mean([r[k] for r in results])), 2) for k in results[0]}
            print(f"{view:6s} {provider:5s} effort {args.effort}: {means}", flush=True)
    chance_groups = np.mean([truths[f]["groups"] == 1 for f in truths])
    print(f"baseline: always one group {chance_groups:.2f}; region guess 1/9 = 0.11")


if __name__ == "__main__":
    main()
