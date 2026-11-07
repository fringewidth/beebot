"""Side-by-side strips of the recorded window (replay) and the policy's rollout from the same
start, for a few val windows, with the goal's facts and the rollout's facts. For eyeballing what
the policy does wrong.

    .venv/bin/python scripts/vla_rollout_strips.py runs/vla-r16-mean1 OUT.png [n_windows]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from beebot.vla import goal_eval
from beebot.vla.data import decode_field, make_windows, world_at
from beebot.vla.facts import window_facts
from beebot.vla.model import Policy
from beebot.vla.render import frame
from beebot.vla.train import load_or_generate
from beebot.vla.world import observe, step

run_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 6
cfg = json.loads((run_dir / "config.json").read_text())
device = torch.device("mps")
policy = Policy(cfg["grid"], width=cfg.get("width", 192), expert_width=cfg.get("expert_width", 128), layers=cfg.get("layers", 4), goal_tokens=cfg["goal_tokens"], goal_pool=cfg["goal_pool"]).to(device)
policy.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=device), strict=False)
policy.eval()
store = load_or_generate(cfg["grid"], cfg["val_episodes"], 10_000 + cfg["seed"], device)
windows = goal_eval.eventful_windows(store, make_windows(store, cfg["grid"]), device, n, torch.Generator().manual_seed(7))
e, w0, w1 = windows[:, 0], windows[:, 1], windows[:, 2]
length = (w1 - w0)
horizon = int(torch.ceil(goal_eval.SLACK * length.float()).max())


def record(mode):
    world = world_at(store, e, w0, device)
    goal = world_at(store, e, w1, device)
    z = policy.goal_z(policy.goal_encoder(torch.cat([observe(world), observe(goal)], 1)))
    frames = [world]
    with torch.no_grad():
        for t in range(horizon):
            if mode == "policy":
                field = policy.sample(observe(world), z)[:, 0]
            else:
                t_rec = (w0 + t).clamp(max=store["field"].shape[1] - 1)
                rec = decode_field(store["field"][e, t_rec], device)
                field = torch.where(((w0 + t) < w1).to(device).view(-1, 1, 1, 1), rec, torch.zeros_like(rec))
            world, _ = step(world, field)
            frames.append(world)
    return frames


def as_episode(frames, i):
    return {
        "walls": store["walls"][e[i]][None].numpy(),
        "box_rect": store["box_rect"][e[i]][None].numpy(),
        "box_color": store["box_color"][e[i]][None].numpy(),
        "cheese_color": store["cheese_color"][e[i]][None].numpy(),
        "mass": np.stack([f.mass[i].float().cpu().numpy() for f in frames])[None],
        "cheese_pos": np.stack([f.cheese_pos[i].cpu().numpy() for f in frames])[None],
        "cheese_alive": np.stack([f.cheese_alive[i].cpu().numpy() for f in frames])[None],
    }


def snap(ep, t):
    return {"mass": ep["mass"][0, t], "cheese_pos": ep["cheese_pos"][0, t], "cheese_alive": ep["cheese_alive"][0, t]}


replay, rollout = record("replay"), record("policy")
tile, pad, text_h = 160, 6, 64
cols = 6
rows_img = []
for i in range(len(windows)):
    L = int(length[i])
    h = int(np.ceil(goal_eval.SLACK * L))
    rep, pol = as_episode(replay, i), as_episode(rollout, i)
    static = {k: rep[k][0] for k in ("walls", "cheese_color", "box_rect", "box_color")}
    times = [0, L // 3, 2 * L // 3, L, (L + h) // 2, h]
    canvas = Image.new("RGB", (cols * (tile + pad), 2 * (tile + pad) + text_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    for r, ep in enumerate((rep, pol)):
        for c, t in enumerate(times):
            img = frame(ep, 0, t).resize((tile, tile))
            canvas.paste(img, (c * (tile + pad), r * (tile + pad)))
            draw.text((c * (tile + pad) + 3, r * (tile + pad) + 3), f"{'rec' if r == 0 else 'pol'} t={t}", fill=(255, 255, 0))
    goal_facts = window_facts(static, snap(rep, 0), snap(rep, L))
    best = max(range(h + 1), key=lambda t: len(window_facts(static, snap(pol, 0), snap(pol, t)).atoms & goal_facts.atoms))
    pol_facts = window_facts(static, snap(pol, 0), snap(pol, h))
    y = 2 * (tile + pad)
    draw.text((3, y), ("GOAL: " + " ".join(goal_facts.lines))[:230], fill=(0, 0, 0))
    draw.text((3, y + 20), ("POLICY at horizon: " + " ".join(pol_facts.lines))[:230], fill=(120, 0, 0))
    draw.text((3, y + 40), f"window {L} steps, horizon {h}", fill=(60, 60, 60))
    rows_img.append(canvas)
sheet = Image.new("RGB", (rows_img[0].width, sum(r.height + 10 for r in rows_img)), (200, 200, 200))
y = 0
for r in rows_img:
    sheet.paste(r, (0, y))
    y += r.height + 10
sheet.save(out)
print(out)
