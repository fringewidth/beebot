"""GIFs of domain-randomized live episodes (src/beebot/vla/voxel/randomize.py) for the browse page: per kind
(failure, weak, broken, drop, late, placement) `per` episodes with the item at strength 1 (picked so that the
effect shows), grey isometric views as scripts/reverse_live_browse.py (placement: the top camera of
views.scene_views, perfect cells left, off-grid robots right), caption under the picture. Writes
dr-<kind>-<k>.gif and cards-dr.json.

    OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python scripts/randomize_browse.py --out <browse dir> --per 3
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

from beebot.vla.voxel import randomize as DR  # noqa: E402
from beebot.vla.voxel import reverse_live as L  # noqa: E402
from beebot.vla.voxel import views as V  # noqa: E402
from beebot.vla.voxel import world as W  # noqa: E402
from beebot.vla.voxel.world import STAY  # noqa: E402

KINDS = {
    "failure": dict(failure=1.0, harm=1.0),
    "weak": dict(weak=1.0),
    "broken": dict(broken=1.0),
    "drop": dict(drop=1.0),
    "late": dict(late=1.0),
    "placement": dict(placement=1.0),
}
BOOST = {"drop": 0.12, "late": 0.12}  # the browse page shows the effect often: dropout rates raised for the picture only
FAMILIES = ("goto", "gather", "tower", "groups", "line", "spread")
BLURB = {
    "failure": "the MuJoCo table's failure rates times a per-episode multiplier; a harmful end (a cube knocked off or stuck) is followed by a guided re-plan from the disrupted state",
    "weak": "a few cubes fail 2-4 times as often as the table says (the table is on)",
    "broken": "a cube stops responding (stuck from the start; the goal holds its cell); it is commanded, nothing happens, the others work around it",
    "drop": "a commanded move is not executed (rate raised to 12 % for this picture); the cube makes the move again",
    "late": "a commanded move is executed one step late (rate raised to 12 % for this picture); everyone waits a step",
    "placement": "rendering only: robots drawn off their cells (up to 0.1 cell, 5 degrees) with slow drift; the state is on the grid",
}


def boosted(kind):
    if kind not in BOOST:
        return None
    original = DR.draw_physics

    def draw(rand, rng, *args, **kw):
        out = original(rand, rng, *args, **kw)
        if kind == "drop":
            out.p_drop = BOOST[kind]
        else:
            out.p_late = BOOST[kind]
        return out

    return draw


def effect(ep):
    """The steps where a commanded move did not happen (cells commanded to move and still occupied next frame)."""
    out = []
    for t in range(ep.steps):
        cmd = ep.codes(t)
        here = torch.from_numpy(ep.occupancy(t + 1))
        n = 0
        for z, y, x in (cmd != STAY).nonzero().tolist():
            if here[z, y, x]:
                n += 1
        out.append(n)
    return out


def picture(ep, t, kind, size=300):
    world = ep.world(t)
    if kind != "placement":
        return [frame(world, False), frame(world, True)]
    obs = W.observe(world)[0]
    spec = V.ViewSpec.clean(size=size, cameras=("top",))
    rng = np.random.default_rng(0)
    plain, _ = V.scene_views(obs, rng, spec, jitter=V.Placement(seed=0, offset=0.0, yaw=0.0))
    jitter = DR.placement_of(ep.meta, t)
    moved, _ = V.scene_views(obs, rng, spec, jitter=jitter)
    return [plain[0], moved[0]]


def gif(ep, kind, path, font, nominal_steps, max_frames=70):
    steps = ep.steps
    every = max(1, -(-steps // max_frames))
    shown = sorted(set(range(0, steps + 1, every)) | {steps})
    miss = effect(ep)
    meta = ep.meta
    physics = meta["physics"]
    extra = {"broken": f"broken cube(s) at {physics['broken']}" if physics["broken"] else "", "weak": f"{len(physics['weak'])} weak cubes", "failure": f"multiplier x{physics['multiplier']:.2f}, harmful outcomes {'on' if physics['harm'] else 'off'}"}.get(kind, "")
    caption = [f"{kind}: {BLURB[kind]}", f"{meta['robots']} robots, {steps} steps (nominal plan {nominal_steps}), {meta['repeats']} moves redone, {meta['replans']} re-plans" + (f", {extra}" if extra else "") + ("; left: perfect cells, right: as rendered" if kind == "placement" else "; left view from the front right corner, right view from the back left")]
    width = sum(p.width for p in picture(ep, 0, kind))
    wrapped = [textwrap.fill(line, max(50, width // 7)) for line in caption]
    frames, durations = [], []
    for t in shown:
        pics = picture(ep, t, kind)
        note = f"step {t} / {steps}" + (f": {miss[t]} commanded moves did not happen" if t < steps and miss[t] else "") + (" (start)" if t == 0 else " (end)" if t == steps else "")
        lines = [note] + wrapped
        rows = sum(line.count("\n") + 1 for line in lines)
        canvas = Image.new("RGB", (sum(p.width for p in pics), pics[0].height + 14 + 17 * rows), "white")
        x = 0
        for p in pics:
            canvas.paste(p, (x, 0))
            x += p.width
        ImageDraw.Draw(canvas).multiline_text((8, pics[0].height + 6), "\n".join(lines), fill="black", font=font)
        frames.append(canvas)
        durations.append(1800 if t in (0, steps) else 140)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--per", type=int, default=3)
    parser.add_argument("--kinds", default=",".join(KINDS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 13)
    out = Path(args.out)
    cards = []
    for kind in args.kinds.split(","):
        patched = boosted(kind)
        original = DR.draw_physics
        if patched is not None:
            DR.draw_physics = patched
        try:
            found = 0
            seed = 0
            while found < args.per and seed < 400:
                seed += 1
                family = FAMILIES[seed % len(FAMILIES)]
                row, _ = L.generate(seed, [family], robots=(6, 14))
                if row is None:
                    continue
                rand = DR.Randomization(**{**dict(strength=0.0, placement=0.0), **KINDS[kind]})
                made = DR.randomize_row(row, rand, seed)
                if made is None:
                    continue
                ep = L.LiveEpisode(made)
                if kind == "placement":
                    made = DR.randomize_row(row, DR.Randomization(strength=0.0, placement=1.0), seed)
                    ep = L.LiveEpisode(made)
                else:
                    if not (made["meta"]["physics"]["broken"] if kind == "broken" else made["meta"]["physics"]["weak"] if kind == "weak" else made["meta"]["repeats"] + made["meta"]["replans"] > 0):
                        continue
                    if kind == "failure" and made["meta"]["replans"] == 0:
                        continue
                    if kind == "late" and not any(sum(effect(ep)[t : t + 1]) for t in range(ep.steps)):
                        continue
                ok = bool(DR.replay_drawn(made)["passed"]) if "stuck" in made else True
                name = f"dr-{kind}-{found}.gif"
                gif(ep, kind, out / name, font, row["meta"]["steps"])
                cards.append({"name": name, "kind": kind, "family": family, "robots": made["meta"]["robots"], "steps": ep.steps, "nominal_steps": row["meta"]["steps"], "repeats": made["meta"]["repeats"], "replans": made["meta"]["replans"], "broken": made["meta"]["physics"]["broken"], "replay": ok})
                print(name, family, made["meta"]["robots"], ep.steps, "replay", ok, flush=True)
                found += 1
        finally:
            DR.draw_physics = original
    path = out / "cards-dr.json"
    old = json.loads(path.read_text()) if path.exists() else []
    ran = set(args.kinds.split(","))
    path.write_text(json.dumps([c for c in old if c["kind"] not in ran] + cards, indent=0))  # kinds not run now keep their cards


if __name__ == "__main__":
    main()
