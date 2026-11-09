"""Live steering of the 2D LM policy: many independent swarms (one per visitor) advanced together
in one batched forward per step, each under its own instruction, and suggestions of commands
that rollouts show the policy follows.

Suggestions come from the task makers (tasks.MAKERS) filled in from the current scene, limited to
the intent keys (tasks.WORDINGS keys: "logic.single", "place.region", ...) that validate() found
reliable, both on a fresh scene and right after another command:

    uv run python -m beebot.vla.steer --init_run RUN --checkpoint policy_latest.pt --out scripts/steer_kinds.json
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import replace

import numpy as np
import torch

from beebot.vla import tasks as T
from beebot.vla.lm_policy import PATCH
from beebot.vla.lm_rl import RLConfig, lm_sampler, load_policy, tokens
from beebot.vla.world import COLOUR_NAMES, World, observe, step

SUGGEST_FAMILIES = ("logic", "place", "formation", "runner", "parallel")  # herd: the policy cannot push


def scene_of(world, runners, i=0):
    """The scene dict (tasks.sample_scene's format) of scene i's current state: alive cheeses only,
    so slot order matches tasks.stack of the result."""
    alive = world.cheese_alive[i].cpu().tolist()
    pos = world.cheese_pos[i].cpu().tolist()
    colour = world.cheese_color[i].cpu().tolist()
    run = runners[i].cpu().tolist()
    slots = [k for k, a in enumerate(alive) if a]
    boxes = [(rect, c) for rect, c in zip(world.box_rect[i].cpu().tolist(), world.box_color[i].cpu().tolist()) if c >= 0]
    return {
        "mass": world.mass[i].float().cpu().numpy(),
        "walls": world.walls[i].cpu().numpy(),
        "cheeses": [tuple(pos[k]) for k in slots],
        "cheese_colours": [colour[k] for k in slots],
        "boxes": [tuple(r) for r, _ in boxes],
        "box_colours": [c for _, c in boxes],
        "runners": [j for j, k in enumerate(slots) if run[k]],
    }


def candidates(scene, rng, tries=4):
    """(intent key, Task) pairs the makers can build for this scene, in plain training wordings."""
    keys = []
    original = T.say

    def capture(generator, key, held, **kw):  # tasks.say; its fill-ins include "r" (a region)
        keys.append(key)
        return original(generator, key, held, **kw)

    out = []
    T.say = capture
    try:
        for family in SUGGEST_FAMILIES:
            for _ in range(tries):
                keys.clear()
                _, make = T.MAKERS[family](rng, "")
                task = make(scene)
                if task is not None and keys and task.spec.get("form") != "absent":  # "absent": asks for a colour not there
                    out.append((keys[0], task))
    finally:
        T.say = original
    return out


def suggest(scene, reliable, rng, avoid=()):
    """A command text for this scene whose intent key is in reliable (key -> success), weighted
    by success, or None."""
    options = [(k, t) for k, t in candidates(scene, rng) if k in reliable and t.text not in avoid]
    if not options:
        return None
    weights = np.array([reliable[k] for k, _ in options])
    return options[int(rng.choice(len(options), p=weights / weights.sum()))][1].text


def first_scene(rng, reliable):
    """A fresh scene with a suggestion for it: (scene, text)."""
    while True:
        scene = None
        while scene is None:
            scene = T.sample_scene(rng, cheeses=(1, 5), boxes=(0, 2), runners=int(rng.random() < 0.25))
        text = suggest(scene, reliable, rng)
        if text:
            return scene, text


def concat(worlds):
    return World(*(torch.cat([getattr(w, f) for w in worlds]) for f in ("mass", "walls", "cheese_pos", "cheese_alive", "cheese_color", "box_rect", "box_color")))


def part(world, i):
    return World(*(getattr(world, f)[i : i + 1] for f in ("mass", "walls", "cheese_pos", "cheese_alive", "cheese_color", "box_rect", "box_color")))


class Swarms:
    """Independent swarms advanced together: advance(ids) steps those with an instruction once,
    in one batched forward. Not thread-safe: the caller serialises calls."""

    def __init__(self, policy, tokenizer, device, guidance=2.0):
        self.policy, self.tokenizer, self.device = policy, tokenizer, device
        self.sample = lm_sampler(policy, guidance, T.SIZE // PATCH)
        self.state = {}  # id -> dict(world, runners, text, t, eaten)

    def reset(self, sid, scene):
        world, runners = T.stack([scene])
        self.state[sid] = {"world": world.to(self.device), "runners": runners, "text": "", "t": 0, "eaten": []}

    def instruct(self, sid, text):
        self.state[sid]["text"] = text

    @torch.no_grad()
    def advance(self, ids):
        """One step for each swarm in ids that has an instruction; returns {id: colours eaten}."""
        ids = [i for i in ids if self.state[i]["text"]]
        if not ids:
            return {}
        states = [self.state[i] for i in ids]
        world = concat([s["world"] for s in states])
        before = world.cheese_alive.clone()
        _, field = self.sample(observe(world), tokens(self.tokenizer, [s["text"] for s in states], self.device))
        world, _ = step(world, field)
        eaten = {}
        for j, (sid, s) in enumerate(zip(ids, states)):
            w = part(world, j)
            s["t"] += 1
            if s["runners"].any() and s["t"] % T.RUNNER_PERIOD == 0:
                w = replace(w, cheese_pos=T.flee(w, s["runners"]))
            gone = (before[j] & ~w.cheese_alive[0]).nonzero().flatten().tolist()
            eaten[sid] = [COLOUR_NAMES[int(w.cheese_color[0, k])] for k in gone]
            s["eaten"] += eaten[sid]
            s["world"] = w
        return eaten


def validate(policy, tokenizer, device, episodes, steps, seed, batch=24):
    """Per intent key: success (full score, nothing unasked eaten) of its commands, on fresh
    scenes and right after another command (48 steps of a random first one)."""
    rng = np.random.default_rng(seed)
    sample = lm_sampler(policy, 2.0, T.SIZE // PATCH)
    condition = lambda texts: tokens(tokenizer, texts, device)
    stats = defaultdict(list)
    for protocol in ("fresh", "after"):
        for k in range(0, episodes, batch):
            scenes = []
            while len(scenes) < batch:
                s = T.sample_scene(rng, cheeses=(1, 5), boxes=(0, 2), runners=int(rng.random() < 0.25))
                if s is not None:
                    scenes.append(s)
            torch.manual_seed(seed + k)
            if protocol == "after":  # a first command from the same makers, then the scene as it ends up
                firsts = []
                for s in scenes:
                    options = candidates(s, rng)
                    firsts.append(options[int(rng.integers(len(options)))][1] if options else None)
                keep = [i for i, f in enumerate(firsts) if f is not None]
                scenes = [scenes[i] for i in keep]
                firsts = [firsts[i] for i in keep]
                world, runners = T.stack(scenes)
                world, runners = world.to(device), runners.to(device)
                ids, mask = condition([f.text for f in firsts])
                for t in range(48):
                    _, field = sample(observe(world), (ids, mask))
                    world, _ = step(world, field)
                    if runners.any() and (t + 1) % T.RUNNER_PERIOD == 0:
                        world = replace(world, cheese_pos=T.flee(world, runners))
                scenes = [scene_of(world, runners, i) for i in range(len(scenes))]
            picked = []
            for s in scenes:
                options = candidates(s, rng)
                by_key = defaultdict(list)
                for key, task in options:
                    by_key[key].append(task)
                if by_key:
                    key = str(rng.choice(sorted(by_key)))
                    picked.append((s, key, by_key[key][int(rng.integers(len(by_key[key])))]))
            if not picked:
                continue
            results = T.run(sample, condition, [p[0] for p in picked], [p[2] for p in picked], steps, device)[0]
            for (_, key, _), (score, unasked) in zip(picked, results):
                stats[(protocol, key)].append((score, unasked))
        print(protocol, "done", flush=True)
    out = {}
    for (protocol, key), rows in sorted(stats.items()):
        out.setdefault(key, {})[protocol] = {"n": len(rows), "success": round(float(np.mean([s >= 0.99 and not u for s, u in rows])), 3), "score": round(float(np.mean([s for s, _ in rows])), 3)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_run", required=True)
    parser.add_argument("--checkpoint", default="policy_latest.pt")
    parser.add_argument("--episodes", type=int, default=288, help="per protocol")
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--out", default="scripts/steer_kinds.json")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    policy, tokenizer = load_policy(RLConfig(init_run=args.init_run, checkpoint=args.checkpoint), device)
    policy.eval()
    t0 = time.time()
    kinds = validate(policy, tokenizer, device, args.episodes, args.steps, args.seed)
    for key, row in kinds.items():
        print(key, row, flush=True)
    with open(args.out, "w") as f:
        json.dump({"init_run": args.init_run, "checkpoint": args.checkpoint, "steps": args.steps, "minutes": round((time.time() - t0) / 60, 1), "kinds": kinds}, f, indent=1)


if __name__ == "__main__":
    main()
