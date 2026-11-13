"""GIFs of saved reverse episodes (scripts/reverse_v3_measure.py --save) for the browse page: two isometric views
(scripts/reverse_demos.frame), every k-th frame for long episodes, a caption under the picture with the instruction,
robots and steps. Picks `per` episodes per size bin (start as asked, replay passed; when a bin has fewer, replayed
ones whose start missed, said on the card), writes <prefix>-<bin>-<i>.gif and cards-<prefix>.json.

    nice -n 10 .venv/bin/python scripts/reverse_v3_browse.py --eps runs/reverse_v3/eps runs/reverse_v3/eps_final --out <browse dir> --per 2
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reverse_demos import frame  # noqa: E402

from beebot.vla.voxel import reverse as R  # noqa: E402

BINS = ((4, 115), (116, 250), (251, 500), (501, 800))


def gif(saved, path, max_frames=90, font=None):
    row, mass, walls = saved["row"], saved["mass"], saved["walls"]
    steps = mass.shape[0] - 1
    every = max(1, -(-steps // max_frames))  # every k-th frame (each a real state)
    shown = sorted(set(range(0, steps + 1, every)) | {steps})
    world = R.blank()
    world.walls[0] = walls
    caption = f"you: {saved['text']}\n{row['robots']} robots, {steps} steps, {row['moving']} moving per step on average" + (f", every {every}th frame" if every > 1 else "")
    frames, durations = [], []
    for t in shown:
        world.mass[0] = mass[t].float()
        a, b = frame(world, False), frame(world, True)
        lines = [f"step {t} / {steps}" + (" start" if t == 0 else (" goal" if t == steps else ""))] + [textwrap.fill(p, 110) for p in caption.split("\n")]
        canvas = Image.new("RGB", (a.width * 2, a.height + 22 + 17 * sum(line.count("\n") + 1 for line in lines)), "white")
        canvas.paste(a, (0, 0))
        canvas.paste(b, (a.width, 0))
        ImageDraw.Draw(canvas).multiline_text((8, a.height + 6), "\n".join(lines), fill="black", font=font)
        frames.append(canvas)
        durations.append(1500 if t in (0, steps) else 120)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", nargs="+", required=True, help="directories of saved episodes (a later one wins on the same goal)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--per", type=int, default=2)
    parser.add_argument("--prefix", default="reverse3")
    args = parser.parse_args()
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 13)
    by_id = {}
    for directory in args.eps:
        for path in sorted(Path(directory).glob("*.pt")):
            by_id[path.stem] = torch.load(path)
    saved = list(by_id.values())
    out = Path(args.out)
    cards = []
    for lo, hi in BINS:
        pool = [s for s in saved if lo <= s["row"]["robots"] <= hi and s["row"]["ok"] and s["row"]["steps"] > R.HOLD_STEPS]
        pool.sort(key=lambda s: (not s["row"]["guided"], s["row"]["id"]))  # one guided (A to B) first when there is one
        picked = pool[:1] + [s for s in pool[1:] if s["row"]["guided"] != pool[0]["row"]["guided"]][:1] if pool else []
        picked += [s for s in pool if s not in picked][: args.per - len(picked)]
        if len(picked) < args.per:  # too few with the start as asked: replayed ones whose start missed (said on the card)
            rest = [s for s in saved if lo <= s["row"]["robots"] <= hi and s["row"]["replay"] and not s["row"]["ok"]]
            rest.sort(key=lambda s: (s["row"]["guided"], s["row"]["id"]))
            picked += rest[: args.per - len(picked)]
        for i, s in enumerate(picked[: args.per]):
            name = f"{args.prefix}-{lo}-{hi}-{i}.gif"
            gif(s, out / name, font=font)
            row = s["row"]
            cards.append({"name": name, "bin": f"{lo}-{hi}", "kind": row["kind"], "source": row["source"], "text": s["text"], "robots": row["robots"], "steps": row["steps"], "moving": row["moving"], "guided": row["guided"], "end": row["end"], "ok": row["ok"], "out_of_band": row.get("out_of_band"), "distance": row.get("distance"), "kinds": row["kinds"], "replay": row["replay"]})
            print(name, row["robots"], row["steps"], row["kind"], flush=True)
    (out / f"cards-{args.prefix}.json").write_text(json.dumps(cards))


if __name__ == "__main__":
    main()
