"""Grounding tests: does a text policy do what it is told, or follow the scene? Controlled
instructions on held-out scenes, every one a counterfactual (the same scene under other words).

Tests (each from a window's start state):
  eat_colour    scenes with cheeses of 2+ colours; "eat the X cheese" for each colour present.
                hit: X eaten; wrong: another cheese eaten; first_right: X was the first eaten.
  box_colour    scenes with boxes of 2+ colours; "go into the X box" for each.
                hit: the swarm went into X (the scorer's "into box" event); wrong: into another.
  eat_gone      the eat_colour scenes with the X cheese removed before the start (as if just
                eaten); "eat the X cheese". unasked: another cheese eaten.
  eat_absent    scenes without any X cheese (X a palette colour), at least one cheese present;
                "eat the X cheese". unasked: a cheese eaten.
  control       the eat_colour scenes; "gather in the center". unasked: a cheese eaten (how much
                the scene alone makes the swarm eat).
Used by scripts/vla_grounding_eval.py and as the live scorecard of lm_rl.
"""

import torch

from beebot.vla import goal_eval
from beebot.vla.data import world_at
from beebot.vla.facts import window_facts
from beebot.vla.world import PALETTE_NAMES, World, observe, step


def start_worlds(store, trials, device):
    """The start worlds of trials (dicts with e, w0 and optionally remove: a cheese slot taken
    away before the start)."""
    e = torch.tensor([t["e"] for t in trials])
    w0 = torch.tensor([t["w0"] for t in trials])
    world = world_at(store, e, w0, device)
    alive = world.cheese_alive.clone()
    for i, t in enumerate(trials):
        if t.get("remove") is not None:
            alive[i, t["remove"]] = False
    return World(mass=world.mass, walls=world.walls, cheese_pos=world.cheese_pos, cheese_alive=alive, cheese_color=world.cheese_color, box_rect=world.box_rect, box_color=world.box_color)


@torch.no_grad()
def rollout(sample, condition, trials, store, steps, device, batch):
    """Per trial: the colours eaten in order and the scorer's events from start to end.
    condition(texts) -> a batch condition; sample(obs, condition) -> a chunk (first step used)."""
    out = []
    for k in range(0, len(trials), batch):
        part = trials[k : k + batch]
        world = start_worlds(store, part, device)
        start = world.to("cpu")
        cond = condition([t["text"] for t in part])
        eaten = [[] for _ in part]
        alive_before = world.cheese_alive.cpu()
        for _ in range(steps):
            world, _ = step(world, sample(observe(world), cond)[:, 0])
            alive_now = world.cheese_alive.cpu()
            for i, slot in (alive_before & ~alive_now).nonzero().tolist():
                eaten[i].append(PALETTE_NAMES[int(start.cheese_color[i, slot])])
            alive_before = alive_now
        end = world.to("cpu")
        for i, t in enumerate(part):
            static = {"walls": start.walls[i].numpy(), "cheese_color": start.cheese_color[i].numpy(), "box_rect": start.box_rect[i].numpy(), "box_color": start.box_color[i].numpy()}
            atoms = window_facts(static, goal_eval.snapshot(start, i), goal_eval.snapshot(end, i)).atoms
            out.append({**t, "eaten": eaten[i], "atoms": sorted(map(list, atoms), key=str)})
    return out


def make_trials(store, windows, n_scenes):
    trials = []
    count = {"eat_colour": 0, "box_colour": 0, "eat_absent": 0}
    for w in windows.tolist():
        e, w0 = w[0], w[1]
        colours = store["cheese_color"][e].tolist()
        alive = store["cheese_alive"][e, w0].tolist()
        present = {PALETTE_NAMES[c]: slot for slot, (c, a) in enumerate(zip(colours, alive)) if c >= 0 and a}
        anywhere = {PALETTE_NAMES[c] for c in colours if c >= 0}
        boxes = [PALETTE_NAMES[c] for c, rect in zip(store["box_color"][e].tolist(), store["box_rect"][e].tolist()) if c >= 0 and rect[2] > 0]
        base = {"e": e, "w0": w0}
        if len(present) >= 2 and count["eat_colour"] < n_scenes:
            count["eat_colour"] += 1
            for colour, slot in present.items():
                trials.append({**base, "test": "eat_colour", "target": colour, "text": f"eat the {colour} cheese"})
                trials.append({**base, "test": "eat_gone", "target": colour, "remove": slot, "text": f"eat the {colour} cheese"})
            trials.append({**base, "test": "control", "target": None, "text": "gather in the center"})
        if len(set(boxes)) >= 2 and count["box_colour"] < n_scenes:
            count["box_colour"] += 1
            for colour in sorted(set(boxes)):
                trials.append({**base, "test": "box_colour", "target": colour, "text": f"go into the {colour} box"})
        missing = [c for c in PALETTE_NAMES if c not in anywhere]
        if present and missing and count["eat_absent"] < n_scenes:
            count["eat_absent"] += 1
            trials.append({**base, "test": "eat_absent", "target": missing[0], "text": f"eat the {missing[0]} cheese"})
    return trials


def summarise(results):
    rows = {}
    for test in ("eat_colour", "box_colour", "eat_gone", "eat_absent", "control"):
        mine = [r for r in results if r["test"] == test]
        if not mine:
            continue
        n = len(mine)
        if test == "eat_colour":
            hit = sum(r["target"] in r["eaten"] for r in mine)
            wrong = sum(any(c != r["target"] for c in r["eaten"]) for r in mine)
            some = [r for r in mine if r["eaten"]]
            first = sum(r["eaten"][0] == r["target"] for r in some)
            rows[test] = {"n": n, "hit": round(hit / n, 3), "wrong": round(wrong / n, 3), "first_right": round(first / max(len(some), 1), 3)}
        elif test == "box_colour":
            hit = sum(["into box", r["target"]] in r["atoms"] for r in mine)
            wrong = sum(any(a[0] == "into box" and a[1] != r["target"] for a in r["atoms"]) for r in mine)
            rows[test] = {"n": n, "hit": round(hit / n, 3), "wrong": round(wrong / n, 3)}
        else:
            unasked = sum(bool(r["eaten"]) for r in mine)
            rows[test] = {"n": n, "unasked_eating": round(unasked / n, 3)}
    return rows
