"""Demo GIFs for the site: the same held-out goal played by the recorded play, the old flow
policy and the new readout policy, side by side with the goal frame, from the same start.

    .venv/bin/python scripts/vla_demo_media.py --out site/media/vla [--windows 96] [--pick 6]

Windows come from the frozen test set (held-out play programs, play data cut off at the frozen
table's time), restricted to ones whose goal includes eating a cheese. Each policy runs at its
best setting from the frozen table. Picks the windows where the readout does best relative to
flow, plus the readout's worst window (an honest failure). Writes one GIF per window and
demos.json (goal facts and every mode's facts score per window).
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from beebot.vla import goal_eval
from beebot.vla.data import decode_field, world_at
from beebot.vla.facts import window_facts
from beebot.vla.model import Policy
from beebot.vla.render import frame
from beebot.vla.train import Config, load_data
from beebot.vla.world import observe, step

CUTOFF = 1793833680  # the frozen table's play cutoff
PANEL = 240
HEADER = 30
MODES = [("replay", "Recorded play"), ("flow", "Before: flow expert"), ("readout", "Now: readout"), ("goal", "Goal")]


def load_policy(run, device):
    stored = json.loads((Path(run) / "config.json").read_text())
    cfg = Config(**{k: v for k, v in stored.items() if k in {f.name for f in dataclasses.fields(Config)}})
    cfg.play_cutoff = CUTOFF
    policy = Policy(cfg.grid, width=cfg.width, expert_width=cfg.expert_width, layers=cfg.layers, goal_tokens=cfg.goal_tokens, goal_pool=cfg.goal_pool, objective=cfg.objective, world_model=cfg.world_model, insulate=cfg.insulate).to(device)
    policy.load_state_dict(torch.load(Path(run) / "model_best.pt", map_location=device))
    return policy.eval(), cfg


def rollout(mode, policy, store, w, horizon, device, execute=1, guidance=2.0):
    """Worlds at steps 0..horizon (each a World on the CPU) for windows w (B, 3)."""
    e, w0, w1 = w[:, 0], w[:, 1], w[:, 2]
    world = world_at(store, e, w0, device)
    goal = world_at(store, e, w1, device)
    z = None
    if policy is not None:
        z = policy.goal_z(policy.goal_encoder(torch.cat([observe(world), observe(goal)], dim=1)))
    worlds, chunk = [world.to("cpu")], None
    with torch.no_grad():
        for t in range(horizon):
            if mode == "replay":
                t_rec = (w0 + t).clamp(max=store["field"].shape[1] - 1)
                recorded = decode_field(store["field"][e, t_rec], device)
                field = torch.where(((w0 + t) < w1).to(device).view(-1, 1, 1, 1), recorded, torch.zeros_like(recorded))
            else:
                if t % execute == 0:
                    chunk = policy.sample(observe(world), z, steps=10, guidance=guidance)
                field = chunk[:, t % execute]
            world, _ = step(world, field)
            worlds.append(world.to("cpu"))
    return worlds


def episode(worlds, i):
    """One window's rollout as render.frame's episode dict."""
    return {
        "walls": worlds[0].walls[i : i + 1].numpy(),
        "box_rect": worlds[0].box_rect[i : i + 1].numpy(),
        "box_color": worlds[0].box_color[i : i + 1].numpy(),
        "cheese_color": worlds[0].cheese_color[i : i + 1].numpy(),
        "mass": np.stack([w.mass[i].float().numpy() for w in worlds])[None],
        "cheese_pos": np.stack([w.cheese_pos[i].numpy() for w in worlds])[None],
        "cheese_alive": np.stack([w.cheese_alive[i].numpy() for w in worlds])[None],
    }


def snap(world, i):
    return {"mass": world.mass[i].float().numpy(), "cheese_pos": world.cheese_pos[i].numpy(), "cheese_alive": world.cheese_alive[i].numpy()}


def describe(atoms):
    """Plain words for a goal's fact atoms, hard ones first."""
    words = []
    for atom in sorted(atoms, key=lambda a: ["ate", "around", "pushed", "groups", "into box", "left box", "away", "move", "end region"].index(a[0])):
        kind = atom[0]
        if kind == "ate":
            words.append(f"eat the {atom[1]} cheese")
        elif kind == "around":
            words.append(f"surround the {atom[1]} cheese")
        elif kind == "pushed":
            words.append(f"push the {atom[1]} cheese {atom[2]}")
        elif kind == "groups":
            words.append("stay in one group" if atom[1] == 1 else f"split into {atom[1]}{'+' if atom[1] == 4 else ''} groups")
        elif kind == "into box":
            words.append(f"go into the {atom[1]} box")
        elif kind == "left box":
            words.append(f"leave the {atom[1]} box")
        elif kind == "away":
            words.append(f"move away from the {atom[1]} cheese")
        elif kind == "move":
            words.append(f"move {atom[1]}")
        elif kind == "end region":
            words.append(f"end up {atom[1]}")
    return words


def gif(path, panels, horizon):
    """panels: [(label, episode or None for a static goal frame, goal episode)]"""
    font = ImageFont.load_default(size=15)
    frames = []
    for t in range(horizon + 1 + 8):  # hold the last frame for 8 more
        canvas = Image.new("RGB", (PANEL * len(panels) + 6 * (len(panels) - 1), PANEL + HEADER), (20, 20, 24))
        draw = ImageDraw.Draw(canvas)
        for k, (label, ep) in enumerate(panels):
            tt = min(t, ep["mass"].shape[1] - 1)
            image = frame(ep, 0, tt).resize((PANEL, PANEL), Image.NEAREST)
            x = k * (PANEL + 6)
            canvas.paste(image, (x, HEADER))
            draw.text((x + 8, 6), label, fill=(232, 230, 224), font=font)
        draw.text((canvas.width - 64, 6), f"t={min(t, horizon):>3}", fill=(150, 148, 140), font=font)
        frames.append(canvas)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=110, loop=0, optimize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="site/media/vla")
    parser.add_argument("--flow", default="runs/vla-d-g32-play-goal")
    parser.add_argument("--readout", default="runs/vla-d-g32-play-readout-world")
    parser.add_argument("--windows", type=int, default=96)
    parser.add_argument("--pick", type=int, default=6)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--rollouts", default="", help="also write rollouts as JSON for the site (rollouts.json): the readout's best windows and its worst")
    parser.add_argument("--best", type=int, default=9, help="how many of the readout's best windows to write to --rollouts")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    flow, _ = load_policy(args.flow, device)
    readout, cfg = load_policy(args.readout, device)
    _, _, _, _, store, windows = load_data(cfg, device)

    statics, targets, chosen = [], [], []
    for row in windows.tolist():
        e, w0, w1 = row
        static = {k: store[k][e].numpy() for k in ("walls", "cheese_color", "box_rect", "box_color")}
        atoms = window_facts(static, goal_eval.stored_snapshot(store, e, w0), goal_eval.stored_snapshot(store, e, w1)).atoms
        if any(a[0] == "ate" for a in atoms):
            chosen.append(row)
            statics.append(static)
            targets.append(atoms)
        if len(chosen) == args.windows:
            break
    w = torch.tensor(chosen)
    horizon = int(torch.ceil(goal_eval.SLACK * (w[:, 2] - w[:, 1]).float()).max())
    runs = {
        "replay": rollout("replay", None, store, w, horizon, device),
        "flow": rollout("flow", flow, store, w, horizon, device, execute=4, guidance=2.0),
        "readout": rollout("readout", readout, store, w, horizon, device, execute=1, guidance=2.0),
    }
    goals = [world_at(store, w[:, 0], w[:, 2], device).to("cpu")]

    rows = []
    for i, (e, w0, w1) in enumerate(chosen):
        h = int(np.ceil(goal_eval.SLACK * (w1 - w0)))
        start = goal_eval.stored_snapshot(store, e, w0)
        end = goal_eval.stored_snapshot(store, e, w1)
        scores = {m: goal_eval.fact_scores(statics[i], start, end, [snap(world, i) for world in worlds[1 : h + 1]])[0] for m, worlds in runs.items()}
        rows.append({"index": i, "episode": e, "w0": w0, "w1": w1, "horizon": h, "goal": describe(targets[i]), "scores": scores})

    # Replaying recorded moves is not deterministic (conflicts between robots are settled at
    # random), so show windows where the replay itself reached the goal: a clean comparison.
    clean = [r for r in rows if r["scores"]["replay"] >= 0.8]
    ranked = sorted(clean, key=lambda r: (r["scores"]["readout"] - r["scores"]["flow"], r["scores"]["readout"]), reverse=True)
    picks = ranked[: args.pick] + [min(clean, key=lambda r: (r["scores"]["readout"], -r["scores"]["replay"]))]
    for k, r in enumerate(picks):
        i, h = r["index"], r["horizon"]
        panels = [(label, episode(runs[m], i) if m != "goal" else episode(goals, i)) for m, label in MODES]
        name = f"demo_{k}.gif" if k < args.pick else "demo_fail.gif"
        gif(out / name, panels, h)
        r["file"] = name
        print(name, r["goal"], {m: round(s, 2) for m, s in r["scores"].items()}, flush=True)
    if args.rollouts:
        # The readout's best windows (highest score, then the richest goals), and its worst one;
        # all from the rollouts scored above.
        best = sorted(clean, key=lambda r: (r["scores"]["readout"], len(r["goal"]), r["scores"]["replay"]), reverse=True)[: args.best]
        worst = min(clean, key=lambda r: (r["scores"]["readout"], -r["scores"]["replay"]))
        windows_json = [encode(runs, goals[0], r["index"], r, False) for r in best] + [encode(runs, goals[0], worst["index"], worst, True)]
        size = int(runs["replay"][0].walls.shape[-1])
        Path(args.rollouts).write_text(json.dumps({"grid": size, "cheese_side": max(2, size // 8), "windows": windows_json}, separators=(",", ":")))
        print("wrote", args.rollouts, Path(args.rollouts).stat().st_size // 1024, "KB", flush=True)
    summary = {m: round(float(np.mean([r["scores"][m] for r in rows])), 3) for m in runs}
    (out / "demos.json").write_text(json.dumps({"windows": len(rows), "mean_scores": summary, "picks": picks}, indent=1))
    print("mean over", len(rows), "eat windows:", summary)




def pack(bits):
    """Bool array -> base64 of its packed bits (row-major)."""
    import base64

    return base64.b64encode(np.packbits(np.asarray(bits, dtype=bool).ravel()).tobytes()).decode()


def cheeses(world, i):
    """[[row, col, alive], ...] for every cheese slot that holds a cheese."""
    return [[int(p[0]), int(p[1]), int(a)] for p, a, c in zip(world.cheese_pos[i].tolist(), world.cheese_alive[i].tolist(), world.cheese_color[i].tolist()) if c >= 0]


def encode(runs, goal, i, row, fail):
    """One window's rollouts as compact JSON: static scene, the goal frame, and per mode and step
    the robots (packed bits) and the cheeses; with the scores these exact rollouts got."""
    from beebot.vla.world import PALETTE_NAMES

    first, h = runs["replay"][0], row["horizon"]
    return {
        "goal": row["goal"],
        "fail": fail,
        "steps": h + 1,
        "walls": pack(first.walls[i].numpy()),
        "boxes": [[*r, PALETTE_NAMES[c][0]] for r, c in zip(first.box_rect[i].tolist(), first.box_color[i].tolist()) if c >= 0],
        "cheese_colors": [PALETTE_NAMES[c] for c in first.cheese_color[i].tolist() if c >= 0],
        "goal_frame": {"robots": pack(goal.mass[i].numpy() > 0.5), "cheeses": cheeses(goal, i)},
        "modes": {m: {"robots": [pack(world.mass[i].numpy() > 0.5) for world in worlds[: h + 1]], "cheeses": [cheeses(world, i) for world in worlds[: h + 1]]} for m, worlds in runs.items()},
        "scores": {m: round(v, 3) for m, v in row["scores"].items()},
    }


def export(args):
    """Re-roll the windows picked in demos.json (seeded) and write them as compact JSON for the
    site to draw: per window the static scene, and per mode and step the robots (packed bits)
    and the cheeses. Scores are recomputed for these rollouts, so the page shows what it plays."""
    from beebot.vla.world import PALETTE_NAMES

    device = torch.device(args.device)
    demos = json.loads(Path(args.demos).read_text())
    picks = demos["picks"]
    flow, _ = load_policy(args.flow, device)
    readout, cfg = load_policy(args.readout, device)
    _, _, _, _, store, _ = load_data(cfg, device)
    w = torch.tensor([[p["episode"], p["w0"], p["w1"]] for p in picks])
    horizon = max(p["horizon"] for p in picks)
    torch.manual_seed(args.seed)
    runs = {
        "replay": rollout("replay", None, store, w, horizon, device),
        "flow": rollout("flow", flow, store, w, horizon, device, execute=4, guidance=2.0),
        "readout": rollout("readout", readout, store, w, horizon, device, execute=1, guidance=2.0),
    }
    goal = world_at(store, w[:, 0], w[:, 2], device).to("cpu")
    out = []
    for i, p in enumerate(picks):
        e, w0, w1, h = p["episode"], p["w0"], p["w1"], p["horizon"]
        static = {k: store[k][e].numpy() for k in ("walls", "cheese_color", "box_rect", "box_color")}
        start, end = goal_eval.stored_snapshot(store, e, w0), goal_eval.stored_snapshot(store, e, w1)
        scores = {m: round(goal_eval.fact_scores(static, start, end, [snap(world, i) for world in worlds[1 : h + 1]])[0], 3) for m, worlds in runs.items()}
        first = runs["replay"][0]
        out.append({
            "goal": p["goal"],
            "fail": p.get("file") == "demo_fail.gif",
            "steps": h + 1,
            "walls": pack(first.walls[i].numpy()),
            "boxes": [[*r, PALETTE_NAMES[c][0]] for r, c in zip(first.box_rect[i].tolist(), first.box_color[i].tolist()) if c >= 0],
            "cheese_colors": [PALETTE_NAMES[c] for c in first.cheese_color[i].tolist() if c >= 0],
            "goal_frame": {"robots": pack(goal.mass[i].numpy() > 0.5), "cheeses": cheeses(goal, i)},
            "modes": {m: {"robots": [pack(world.mass[i].numpy() > 0.5) for world in worlds[: h + 1]], "cheeses": [cheeses(world, i) for world in worlds[: h + 1]]} for m, worlds in runs.items()},
            "scores": scores,
        })
        print(p["goal"], scores, flush=True)
    size = int(first.walls.shape[-1])
    Path(args.json).write_text(json.dumps({"grid": size, "cheese_side": max(2, size // 8), "windows": out}, separators=(",", ":")))
    print("wrote", args.json, Path(args.json).stat().st_size // 1024, "KB")


if __name__ == "__main__" and "--json" in __import__("sys").argv:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--demos", default="site/media/vla/demos.json")
    parser.add_argument("--flow", default="runs/vla-d-g32-play-goal")
    parser.add_argument("--readout", default="runs/vla-d-g32-play-readout-world")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="mps")
    export(parser.parse_args())


if __name__ == "__main__" and "--json" not in __import__("sys").argv:
    main()
