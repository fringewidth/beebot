"""Reverse demonstrations (src/beebot/vla/voxel/reverse.py): the replay check with throughput, and
rendered episodes for the browse page. CPU only, light.

    OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python scripts/reverse_demos.py check --per 200 --out runs/reverse_demos/check.json
    OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python scripts/reverse_demos.py browse --per 10 --out <browse dir>

check: per family, `per` goals walked backwards in batches of --batch on one thread, then every
episode replayed forward with world.step (reverse.replay). Prints and writes the replay pass share,
how walks ended, steps, proposals proposed / verified / accepted, and episodes and steps per second
of the walk (one core), with and without the replay.

browse: `per` natural-start episodes per family as GIFs (two isometric views: the usual corner and
the opposite one) named reverse-<family>-<i>.gif, and cards-reverse.json beside them.
"""

import argparse
import json
import textwrap
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from beebot.vla.voxel import reverse as R
from beebot.vla.voxel.render import BACKGROUND, ROBOT, SHADES, WALL, _shade
from beebot.vla.voxel.world import COLOUR_RGB, colour_map


def frame(world, flip=False, s=10):
    """Isometric frame of a one-scene VoxelWorld with every object at its true voxels and floor zones
    as pale tiles; flip turns the grid 180 degrees (copied from the session renderer render_true.py)."""
    mass = (world.mass[0] > 0.5).numpy()
    walls = world.walls[0].numpy().astype(bool)
    objects = colour_map(world)[0].numpy()
    zones = world.zones[0].numpy() if world.zones is not None else np.full(mass.shape[1:], -1)
    if flip:
        mass, walls, objects, zones = mass[:, ::-1, ::-1], walls[:, ::-1, ::-1], objects[:, ::-1, ::-1], zones[::-1, ::-1]
    d, h, w = mass.shape

    def rgb(c):
        return tuple(int(v * 255) for v in COLOUR_RGB[int(c)].tolist())

    colour, tiles = {}, {}
    hidden = np.ones_like(walls)
    hidden[:-1] &= walls[1:]
    hidden[:, :-1] &= walls[:, 1:]
    hidden[:, :, :-1] &= walls[:, :, 1:]
    hidden[-1] = False
    hidden[:, -1] = False
    hidden[:, :, -1] = False
    for z, y, x in np.argwhere(walls & ~hidden):
        colour[(z, y, x)] = WALL
    for z, y, x in np.argwhere(objects >= 0):
        colour[(z, y, x)] = rgb(objects[z, y, x])
    for z, y, x in np.argwhere(mass):
        colour[(z, y, x)] = ROBOT
    for y, x in np.argwhere(zones >= 0):
        col = walls[:, y, x]
        z = int(np.nonzero(col)[0].max()) + 1 if col.any() else 0
        tiles[(z, y, x)] = tuple(int(0.55 * c + 0.45 * 255) for c in rgb(zones[y, x]))
    width = (w + h) * s + 2 * s
    height = (w + h) * s // 2 + d * s + 2 * s
    ox, oy = h * s + s, d * s + s
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    corners = [(0, 0), (0, w), (h, w), (h, 0)]
    draw.polygon([(ox + (x - y) * s, oy + (x + y) * s / 2) for y, x in corners], outline=(200, 200, 195), fill=(236, 236, 230))
    items = [((z, y, x), c, False) for (z, y, x), c in colour.items()] + [((z, y, x), c, True) for (z, y, x), c in tiles.items()]
    items.sort(key=lambda it: (sum(it[0]) - (0.5 if it[2] else 0), it[0][0]))  # a tile lies on the voxel below it: drawn just after it
    for (z, y, x), c, tile in items:
        cx, cy = ox + (x - y) * s, oy + (x + y) * s / 2 - z * s
        top = [(cx, cy - s), (cx + s, cy - s / 2), (cx, cy), (cx - s, cy - s / 2)]
        if tile:
            draw.polygon([(px, py + s) for px, py in top], fill=c)
            continue
        left = [(cx - s, cy - s / 2), (cx, cy), (cx, cy + s), (cx - s, cy + s / 2)]
        right = [(cx, cy), (cx + s, cy - s / 2), (cx + s, cy + s / 2), (cx, cy + s)]
        draw.polygon(top, fill=_shade(c, SHADES[0]))
        draw.polygon(left, fill=_shade(c, SHADES[1]))
        draw.polygon(right, fill=_shade(c, SHADES[2]))
    return image


def world_at(ep, t):
    from dataclasses import replace

    return replace(ep.world, mass=ep.mass[t].unsqueeze(0).float(), cheese_pos=ep.cheese_pos[t].unsqueeze(0), cheese_alive=ep.cheese_alive[t].unsqueeze(0))


def check(per, batch, seed, out):
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    report = {}
    for family in R.FAMILIES:
        passed, natural, steps, ends = [], [], [], {}
        proposed = verified = accepted = backtracks = 0
        walk_s = replay_s = goal_s = 0.0
        for start in range(0, per, batch):
            n = min(batch, per - start)
            t0 = time.time()
            goals = [R.make_goal(family, rng) for _ in range(n)]
            t1 = time.time()
            episodes, stats = R.take_apart(goals, rng)
            t2 = time.time()
            result = R.replay(episodes)
            t3 = time.time()
            goal_s, walk_s, replay_s = goal_s + t1 - t0, walk_s + t2 - t1, replay_s + t3 - t2
            passed += result["passed"].tolist()
            natural += [ep.natural for ep in episodes]
            steps += [ep.steps for ep in episodes]
            for ep in episodes:
                ends[ep.end] = ends.get(ep.end, 0) + 1
            proposed += sum(stats["proposed"])
            verified += sum(stats["verified"])
            accepted += sum(stats["accepted"])
            backtracks += sum(stats["backtracks"])
        generate = goal_s + walk_s
        nat_steps = sum(s for s, n in zip(steps, natural) if n)
        row = {
            "episodes": per,
            "replay_pass": float(np.mean(passed)),
            "natural_start": float(np.mean(natural)),
            "ends": ends,
            "steps_mean": float(np.mean(steps)),
            "steps_max": int(max(steps)),
            "proposed": proposed,
            "verified": verified,
            "accepted": accepted,
            "accepted_per_proposed": accepted / max(proposed, 1),
            "verified_per_proposed": verified / max(proposed, 1),
            "backtracks": backtracks,
            "episodes_per_s": per / generate,
            "steps_per_s": sum(steps) / generate,
            "natural_steps_per_s": nat_steps / generate,
            "steps_per_s_with_replay": sum(steps) / (generate + replay_s),
            "seconds": {"goals": goal_s, "walk": walk_s, "replay": replay_s},
        }
        report[family] = row
        print(family, json.dumps(row), flush=True)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"per": per, "batch": batch, "seed": seed, "threads": 1, "families": report}, indent=1))
    return report


def browse(per, seed, out):
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 13)
    cards = []
    for family in R.FAMILIES:
        goals = [R.make_goal(family, rng) for _ in range(3 * per)]
        episodes, _ = R.take_apart(goals, rng)
        result = R.replay(episodes)
        chosen = [ep for ep, ok in zip(episodes, result["passed"].tolist()) if ep.natural and ok][:per]
        for i, ep in enumerate(chosen):
            frames, durations = [], []
            for t in range(ep.steps + 1):
                world = world_at(ep, t)
                a, b = frame(world, False), frame(world, True)
                label = "start" if t == 0 else ("goal" if t == ep.steps else "")
                text = f"step {t} / {ep.steps} {label}\nyou: {ep.text}"
                canvas = Image.new("RGB", (a.width * 2, a.height + 50), "white")
                canvas.paste(a, (0, 50))
                canvas.paste(b, (a.width, 50))
                ImageDraw.Draw(canvas).multiline_text((8, 6), "\n".join(textwrap.fill(p, 100) for p in text.split("\n")), fill="black", font=font)
                frames.append(canvas)
                durations.append(1500 if t in (0, ep.steps) else 350)
            name = f"reverse-{family}-{i}.gif"
            frames[0].save(out / name, save_all=True, append_images=frames[1:], duration=durations, loop=0)
            robots = int(ep.mass[0].sum())
            cards.append({"name": name, "fam": family, "text": ep.text, "steps": ep.steps, "robots": robots, "info": {k: v for k, v in ep.goal.info.items()}, "replay": True})
            print(name, ep.steps, ep.text, flush=True)
    (out / "cards-reverse.json").write_text(json.dumps(cards))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("check", "browse"))
    parser.add_argument("--per", type=int, default=200)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.mode == "check":
        check(args.per, args.batch, args.seed, args.out)
    else:
        browse(args.per, args.seed, args.out)


if __name__ == "__main__":
    main()
