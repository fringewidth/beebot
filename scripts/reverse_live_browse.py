"""GIFs of live reverse episodes (src/beebot/vla/voxel/reverse_live.py) for the browse page: per family,
`per` episodes (robot counts from a small, a middle and a large bin of the stores' 6-115 draw), each
replayed forward with world.step, drawn as two grey isometric views (scripts/reverse_demos.frame: the
front right corner and the back left one), every k-th frame for long episodes, with the caption under
the picture: the instruction and the first thought (as the live sampler draws them; world frame, the
first view's camera stands at the front right corner), robots and steps. Writes live-<family>-<k>.gif
and cards-live.json.

    OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python scripts/reverse_live_browse.py --out <browse dir> --per 3
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reverse_demos import frame  # noqa: E402

from beebot.vla.voxel import reverse as R  # noqa: E402
from beebot.vla.voxel import reverse_live as L  # noqa: E402

BINS = ((6, 20), (21, 50), (51, 115))


def gif(ep, words, path, font, max_frames=80):
    steps = ep.steps
    every = max(1, -(-steps // max_frames))
    shown = sorted(set(range(0, steps + 1, every)) | {steps})
    instruction, thought, rung = words
    meta = ep.meta
    caption = [f"you: {instruction}", f"thought ({rung}): {thought}", f"{meta['robots']} robots, {steps} steps, {meta['moving']:.1f} moving per step" + (f", every {every}th frame shown" if every > 1 else "") + "; left view from the front right corner, right view from the back left"]
    wrapped = [textwrap.fill(line, 118) for line in caption]
    frames, durations = [], []
    for t in shown:
        world = ep.world(t)
        a, b = frame(world, False), frame(world, True)
        lines = [f"step {t} / {steps}" + (" (start)" if t == 0 else " (end)" if t == steps else "")] + wrapped
        rows = sum(line.count("\n") + 1 for line in lines)
        canvas = Image.new("RGB", (a.width * 2, a.height + 14 + 17 * rows), "white")
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width, 0))
        ImageDraw.Draw(canvas).multiline_text((8, a.height + 6), "\n".join(lines), fill="black", font=font)
        frames.append(canvas)
        durations.append(1800 if t in (0, steps) else 140)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--per", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--families", default=",".join(L.FAMILIES))
    args = parser.parse_args()
    torch.set_num_threads(1)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 13)
    texts = L.Texts()
    out = Path(args.out)
    cards = []
    for family in args.families.split(","):
        for k in range(args.per):
            lo, hi = BINS[k % len(BINS)]
            row = None
            for attempt in range(6):
                seed = args.seed * 1000 + 100 * k + attempt
                row, info = L.generate(seed, [family], robots=(lo, hi))
                if row is not None:
                    break
            if row is None:
                print(family, k, "no episode", info, flush=True)
                continue
            ep = L.LiveEpisode(row)
            passed = bool(R.replay([ep.as_reverse()])["passed"][0])
            words = texts.words(row["meta"], np.random.default_rng(seed))
            name = f"live-{family}-{k}.gif"
            gif(ep, words, out / name, font)
            cards.append({"name": name, "family": family, "kind": row["meta"]["record"]["kind"], "spec": row["meta"]["record"]["spec"], "instruction": words[0], "thought": words[1], "rung": words[2], "robots": row["meta"]["robots"], "steps": ep.steps, "moving": round(row["meta"]["moving"], 2), "replay": passed, "cpu": round(info["cpu"], 2), "tries": info["tries"]})
            print(name, row["meta"]["robots"], ep.steps, "replay", passed, "|", words[0], flush=True)
    (out / "cards-live.json").write_text(json.dumps(cards, indent=0))


if __name__ == "__main__":
    main()
