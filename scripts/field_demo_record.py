"""Recordings for the field demo mock-up: three episodes on the real 2D physics (beebot.vla.world)
driven by a hand-coded DUMMY policy, written to site/field-demo/recordings.json.

The dummy policy stands in for the LM policy (beebot.vla.lm_policy, sampled by lm_rl.lm_sampler)
and produces the same kind of output: for every square a probability over the 9 actions (codes
0-7 are the directions of world.DIRS, code 8 is stay; data.CODE_VECTORS maps codes to vectors).
One code per square is sampled from it and decoded into the field exactly as lm_sampler does
(decode_field), then world.step moves the robots, pushes and eats the cheeses.

Dummy policy, per square: the geodesic distance to the target (the free squares of the cheese's
ring, the region's centre, or the free squares of the box) through open squares, Dijkstra
with 8-neighbour moves; each direction's logit is how much the move shortens that distance per unit
length, over a temperature; moves into walls or cheeses are penalised; staying has its own logit
(raised on the target); Gaussian noise on the logits. On the eat task, ring squares also favour
pressing into the cheese, which is what pushes it; on the other tasks other cheeses' rings carry a
toll, so nothing is eaten by accident.

    uv run python scripts/field_demo_record.py

Format (per episode, per step t: the state at t and the policy's output at that state; the sampled
moves of step t take the world to step t + 1):
  robots   flat [row, col, code, ...] ordered by robot id (ids are stable across steps)
  cheeses  [[row, col, alive], ...] by cheese slot; eaten / pushed: slots eaten / pushed on the way to t
  field    base64 int8 (2, 32, 32): expected vector sum_a p_a * CODE_VECTORS[a], (x, y), times 127
  pmax     base64 uint8 (32, 32): max_a p_a times 255
  obs      per channel an index into the episode's obs_blobs (base64 uint8 32x32, value times 255)
"""

import argparse
import base64
import heapq
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from beebot.vla import tasks as T
from beebot.vla.data import CODE_VECTORS, decode_field, encode_field
from beebot.vla.lm_rl import tokens
from beebot.vla.shapes import shape_mask, text_mask
from beebot.vla.steer import scene_of
from beebot.vla.world import COLOUR_NAMES, COLOUR_RGB, DIRS, STAY, cheese_masks, cheese_side, observe, ring_masks, step

OUT = Path("site/field-demo/recordings.json")
LM_OUT = Path("runs/field-demo/recordings.json")
CHANNELS = ("mass", "wall", "cheese mask", "cheese R", "cheese G", "cheese B", "box mask", "box R", "box G", "box B")
TEMPERATURE = 0.25
STAY_LOGIT = 0.0  # off the target
STAY_LOGIT_TARGET = 4.0  # on the target
STAY_LOGIT_WAIT = 6.0  # no target left (the cheese is eaten)
STAY_LOGIT_BOX = 2.0  # box task: in the box, lower so robots inside move on to free squares deeper in
PRESS_LOGIT = 3.0  # eat task: ring squares pressing into the cheese
BLOCKED_LOGIT = -6.0  # moving into a wall, a cheese (except pressing) or off the grid
RING_TOLL = 6.0  # place tasks: extra distance for passing a cheese's ring square
NOISE = 0.5  # standard deviation of the Gaussian noise on the logits
STEPS_AFTER_EAT = 20


def geodesic(passable, targets, toll):
    """Dijkstra distance (diagonal moves cost sqrt 2, plus toll at the square the move leaves)
    from the target squares through passable squares; inf where unreachable."""
    size = passable.shape[0]
    dist = np.full(passable.shape, np.inf)
    heap = []
    for y, x in zip(*np.nonzero(targets & passable)):
        dist[y, x] = 0.0
        heap.append((0.0, int(y), int(x)))
    heapq.heapify(heap)
    while heap:
        d, y, x = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        for dy, dx in DIRS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < size and 0 <= nx < size and passable[ny, nx]:
                nd = d + math.hypot(dy, dx) + toll[ny, nx]
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(heap, (nd, ny, nx))
    return dist


def target_squares(world, goal):
    """(targets bool (H, W), press bool (8, H, W) or None): where the swarm should end up and, for
    the eat task, which moves press into the cheese."""
    size = T.SIZE
    targets = np.zeros((size, size), bool)
    if goal["kind"] == "eat":
        slot = goal["slot"]
        if not bool(world.cheese_alive[0, slot]):
            return targets, None
        masks = cheese_masks(world)
        ring = ring_masks(world, masks)[0, slot].numpy()
        footprint = masks[0, slot].numpy()
        press = np.zeros((8, size, size), bool)
        for d, (dy, dx) in enumerate(DIRS):
            ahead = np.zeros_like(footprint)
            ys, xs = np.nonzero(ring)
            ny, nx = ys + dy, xs + dx
            inside = (ny >= 0) & (ny < size) & (nx >= 0) & (nx < size)
            hit = np.zeros(len(ys), bool)
            hit[inside] = footprint[ny[inside], nx[inside]]
            ahead[ys[hit], xs[hit]] = True
            press[d] = ahead
        return ring, press
    if goal["kind"] == "region":
        y, x = T.REGIONS[goal["region"]]
        targets[y, x] = True
        return targets, None
    y, x, h, w = goal["box"]
    targets[y : y + h, x : x + w] = True
    return targets, None


def dummy_policy(world, goal, rng):
    """(H, W, 9) probabilities over the action codes for every square."""
    size = T.SIZE
    walls = world.walls[0].numpy()
    on_cheese = cheese_masks(world)[0].any(0).numpy()
    passable = ~walls & ~on_cheese
    toll = np.zeros((size, size))
    if goal["kind"] != "eat":  # keep off the cheeses' rings, so nothing is eaten by accident
        toll[ring_masks(world)[0].any(0).numpy()] = RING_TOLL
    targets, press = target_squares(world, goal)
    free = targets & (world.mass[0].numpy() < 0.5)
    goals = free if free.any() else targets  # target squares no robot holds yet: the swarm wraps around a ring, fills a box from the back
    dist = geodesic(passable, goals, toll) if goals.any() else np.full((size, size), np.inf)
    reachable = np.isfinite(dist)
    near = np.where(reachable, dist, 0.0)

    logits = np.zeros((size, size, 9))
    for d, (dy, dx) in enumerate(DIRS):
        ahead_dist = np.full((size, size), np.inf)
        ahead_open = np.zeros((size, size), bool)
        ys = slice(max(-dy, 0), size - max(dy, 0))
        xs = slice(max(-dx, 0), size - max(dx, 0))
        ys_n = slice(max(dy, 0), size - max(-dy, 0))
        xs_n = slice(max(dx, 0), size - max(-dx, 0))
        ahead_dist[ys, xs] = dist[ys_n, xs_n]
        ahead_open[ys, xs] = passable[ys_n, xs_n] & (~targets[ys, xs] | targets[ys_n, xs_n])  # robots on the target stay on it
        gain = (near - np.where(np.isfinite(ahead_dist), ahead_dist, near)) / math.hypot(dy, dx)
        logits[..., d] = np.where(ahead_open, gain / TEMPERATURE, BLOCKED_LOGIT)
        if press is not None:
            logits[..., d] = np.where(press[d], PRESS_LOGIT, logits[..., d])
    on_target = STAY_LOGIT_BOX if goal["kind"] == "box" else STAY_LOGIT_TARGET
    logits[..., STAY] = np.where(targets & passable, on_target, STAY_LOGIT)
    logits[~reachable & passable, :STAY] = 0.0  # nowhere to go (the cheese is eaten): wait
    logits[~reachable & passable, STAY] = STAY_LOGIT_WAIT
    logits += rng.normal(0.0, NOISE, logits.shape)
    logits[walls] = 0.0
    logits[walls, STAY] = 8.0
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    return probs / probs.sum(-1, keepdims=True)


def sample_field(probs, generator):
    """codes (H, W) sampled per square and the field (1, 2, H, W), as lm_sampler builds it."""
    codes = torch.distributions.Categorical(probs=torch.from_numpy(probs).float()).sample()
    return codes, decode_field(codes.unsqueeze(0))


def match_robots(previous, intended, current):
    """Order current robot positions (N, 2) to follow previous ones: Hungarian matching on the
    distance to where each robot meant to go, plus its distance moved."""
    moved = ((previous[:, None, :] - current[None, :, :]) ** 2).sum(-1)
    off = ((intended[:, None, :] - current[None, :, :]) ** 2).sum(-1)
    rows, cols = linear_sum_assignment(off + 0.1 * moved)
    out = np.zeros_like(current)
    out[rows] = current[cols]
    return out


def b64(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode()


class Recorder:
    """One episode's recording: add(...) once per step, in order; episode() the finished dict."""

    def __init__(self, scene, texts, tokenizer):
        """texts: the commands given in order (one for a single instruction)."""
        self.scene, self.texts, self.tokenizer = scene, texts, tokenizer
        self.positions = np.argwhere(scene["mass"] > 0.5)
        self.codes = None
        self.blobs = []
        self.blob_index = {}
        self.steps = []

    def blob(self, channel):
        key = b64((channel * 255).round().astype(np.uint8))
        if key not in self.blob_index:
            self.blob_index[key] = len(self.blobs)
            self.blobs.append(key)
        return self.blob_index[key]

    def add(self, mass, cheese_pos, cheese_alive, cheese_color, obs, probs, codes, eaten, pushed, command=0):
        """The state at this step (numpy, one scene), the policy's probabilities (H, W, 9) and
        sampled codes (H, W) there, the cheese slots eaten / pushed on the way here, and the index
        of the command in force."""
        if self.steps:  # follow the robots from the previous step
            intended = self.positions + np.array([DIRS[c] if c != STAY else (0, 0) for c in self.codes])
            self.positions = match_robots(self.positions, intended, np.argwhere(mass > 0.5))
        self.codes = codes[self.positions[:, 0], self.positions[:, 1]]
        expected = probs @ CODE_VECTORS.numpy()  # (H, W, 2): x, y
        self.steps.append(
            {
                "t": len(self.steps),
                "command": command,
                "robots": np.column_stack([self.positions, self.codes]).flatten().tolist(),
                "cheeses": [[int(p[0]), int(p[1]), int(a)] for p, a, c in zip(cheese_pos.tolist(), cheese_alive.tolist(), cheese_color.tolist()) if c >= 0],
                "eaten": [int(k) for k in eaten],
                "pushed": [int(k) for k in pushed],
                "field": b64((expected.transpose(2, 0, 1) * 127).round().astype(np.int8)),
                "pmax": b64((probs.max(-1) * 255).round().astype(np.uint8)),
                "obs": [self.blob(obs[c]) for c in range(len(CHANNELS))],
            }
        )

    def episode(self, **extra):
        scene = self.scene
        commands = []
        for text in self.texts:
            ids = self.tokenizer(text).input_ids
            commands.append({"text": text, "token_ids": ids, "tokens": [self.tokenizer.decode([i]) for i in ids]})
        eaten_at = next((s["t"] for s in self.steps if s["eaten"]), None)
        return {
            "instruction": " -> ".join(self.texts),
            "commands": commands,
            "robot_count": int(len(self.positions)),
            "walls": b64(scene["walls"].astype(np.uint8)),
            "boxes": [{"rect": list(r), "colour": COLOUR_NAMES[c], "rgb": COLOUR_RGB[c].tolist()} for r, c in zip(scene["boxes"], scene["box_colours"])],
            "cheeses": [{"colour": COLOUR_NAMES[c], "rgb": COLOUR_RGB[c].tolist()} for c in scene["cheese_colours"]],
            "eaten_at": eaten_at,
            "pushes": sum(len(s["pushed"]) for s in self.steps),
            **extra,
            "obs_blobs": self.blobs,
            "steps": self.steps,
        }


def run_dummy_episode(scene, goal, instruction, tokenizer, max_steps, seed):
    rng = np.random.default_rng(seed)
    generator = torch.manual_seed(seed)
    world, _ = T.stack([scene])
    recorder = Recorder(scene, [instruction], tokenizer)
    eaten_at = None
    eaten, pushed = [], []
    for t in range(max_steps + 1):
        probs = dummy_policy(world, goal, rng)
        codes, field = sample_field(probs, generator)
        recorder.add(world.mass[0].numpy(), world.cheese_pos[0], world.cheese_alive[0], world.cheese_color[0], observe(world)[0].numpy(), probs, codes.numpy(), eaten, pushed)
        if t == max_steps or (eaten_at is not None and t >= eaten_at + STEPS_AFTER_EAT):
            break
        world, info = step(world, field)
        eaten = info.eaten[0].nonzero().flatten().tolist()
        pushed = info.pushed[0].nonzero().flatten().tolist()
        if goal["kind"] == "eat" and goal["slot"] in eaten:
            eaten_at = t + 1
    return recorder.episode(policy="dummy", goal=goal, seed=seed)


# ---------------------------------------------------------------------------------------------
# The trained LM policy: ROLLOUTS rollouts per instruction on fresh scenes, batched, each command
# judged by tasks.score (met: score >= 0.99 and no cheese eaten that the command did not ask for,
# as steer.validate counts success).


def present(scene):
    return {COLOUR_NAMES[c] for c in scene["cheese_colours"]}


def box_rect(scene, colour):
    colours = [COLOUR_NAMES[c] for c in scene["box_colours"]]
    return scene["boxes"][colours.index(colour)] if colour in colours else None


def eat_single(text, a):
    def make(scene):
        if a not in present(scene):
            return None
        return T.Task("logic", text, {"eat": {a}, "any_of": set(), "order": [], "forbid": present(scene) - {a}}, "")

    return make


def eat_either(text, a, b):
    def make(scene):
        if not {a, b} <= present(scene):
            return None
        return T.Task("logic", text, {"eat": set(), "any_of": {a, b}, "order": [], "forbid": present(scene) - {a, b}}, "")

    return make


def eat_all(text):
    def make(scene):
        if not present(scene):
            return None
        return T.Task("logic", text, {"eat": present(scene), "any_of": set(), "order": [], "forbid": set()}, "")

    return make


def eat_nearest_box(text, b):
    """As tasks.logic's nearest_box form: the cheese whose centre is nearest the box's centre."""

    def make(scene):
        rect = box_rect(scene, b)
        if rect is None or not scene["cheeses"]:
            return None
        side = cheese_side(T.SIZE)
        ref = (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)
        centres = {COLOUR_NAMES[c]: (y + side / 2, x + side / 2) for (y, x), c in zip(scene["cheeses"], scene["cheese_colours"])}
        target = min(centres, key=lambda c: np.hypot(centres[c][0] - ref[0], centres[c][1] - ref[1]))
        return T.Task("logic", text, {"eat": {target}, "any_of": set(), "order": [], "forbid": present(scene) - {target}}, "")

    return make


def catch(text, c):
    """As tasks.runner's catch form: eat the cheese c, which runs from the swarm (tasks.flee)."""

    def make(scene):
        if c not in present(scene):
            return None
        return T.Task("runner", text, {"eat": {c}, "any_of": set(), "order": [], "forbid": present(scene) - {c}, "form": "single"}, "")

    return make


def form(text, mask):
    """As tasks.formation: fill the mask (scored by F1 of occupied squares against it)."""

    def make(scene):
        if (mask & scene["walls"]).any():
            return None
        return T.Task("formation", text, {"mask": mask}, "")

    return make


def go_box(text, b):
    def make(scene):
        rect = box_rect(scene, b)
        return None if rect is None else T.Task("place", text, {"box": rect, "forbid": present(scene)}, "")

    return make


def go_region(text, r):
    return lambda scene: T.Task("place", text, {"region": r, "forbid": present(scene)}, "")


def go_direction(text, d):
    return lambda scene: T.Task("place", text, {"direction": d, "forbid": present(scene)}, "")


def scenes_with(cheeses, boxes, cheese_colours=(), box_colours=(), runner="", wall_share=0.5):
    """A scene sampler (tasks.sample_scene: random walls, swarm size and placement) whose scenes
    hold cheeses and boxes of the given colours, and with runner a colour, one runner cheese of
    that colour; None when a draw misses (the caller retries)."""

    def sample(rng):
        scene = T.sample_scene(rng, cheeses=cheeses, boxes=boxes, require=cheese_colours, runners=int(bool(runner)), wall_share=wall_share)
        if scene is None or any(box_rect(scene, b) is None for b in box_colours):
            return None
        if runner and [COLOUR_NAMES[scene["cheese_colours"][k]] for k in scene["runners"]] != [runner]:
            return None
        return scene

    return sample


# Each instruction: its intent keys (tasks.WORDINGS keys, one per command), the commands as
# (text, task maker) in the training wordings, and its scenes (counts as the task makers use).
INSTRUCTIONS = [
    {"keys": ["logic.single"], "commands": [("eat the red cheese", eat_single("eat the red cheese", "red"))], "scenes": scenes_with((1, 6), (0, 2), ("red",))},
    {"keys": ["logic.single"], "commands": [("gobble up the green cheese", eat_single("gobble up the green cheese", "green"))], "scenes": scenes_with((1, 6), (0, 2), ("green",))},
    {"keys": ["logic.or"], "commands": [("eat either yellow or purple", eat_either("eat either yellow or purple", "yellow", "purple"))], "scenes": scenes_with((2, 6), (0, 2), ("yellow", "purple"))},
    {"keys": ["logic.all"], "commands": [("clear the board of cheese", eat_all("clear the board of cheese"))], "scenes": scenes_with((1, 4), (0, 2))},
    {"keys": ["logic.nearest_box"], "commands": [("eat the cheese closest to the white box", eat_nearest_box("eat the cheese closest to the white box", "white"))], "scenes": scenes_with((2, 5), (1, 2), (), ("white",))},
    {"keys": ["place.box"], "commands": [("go into the blue box", go_box("go into the blue box", "blue"))], "scenes": scenes_with((0, 3), (1, 3), (), ("blue",))},
    {"keys": ["place.box"], "commands": [("get inside the orange box", go_box("get inside the orange box", "orange"))], "scenes": scenes_with((0, 3), (1, 3), (), ("orange",))},
    {"keys": ["place.move"], "commands": [("move left", go_direction("move left", "left"))], "scenes": scenes_with((0, 3), (0, 3))},
    {"keys": ["place.move"], "commands": [("head down", go_direction("head down", "down"))], "scenes": scenes_with((0, 3), (0, 3))},
    {"keys": ["place.region"], "commands": [("go to the top left", go_region("go to the top left", "top left"))], "scenes": scenes_with((0, 3), (0, 3))},
    {"keys": ["place.region"], "commands": [("gather in the bottom right", go_region("gather in the bottom right", "bottom right"))], "scenes": scenes_with((0, 3), (0, 3))},
    {"keys": ["formation.shape"], "commands": [("form a ring", form("form a ring", shape_mask("ring", T.SIZE)))], "scenes": scenes_with((0, 2), (0, 1), wall_share=0.0)},
    {"keys": ["formation.shape"], "commands": [("make a plus sign", form("make a plus sign", shape_mask("plus", T.SIZE)))], "scenes": scenes_with((0, 2), (0, 1), wall_share=0.0)},
    {"keys": ["formation.letter"], "commands": [("form the letter L", form("form the letter L", text_mask("L", T.SIZE)))], "scenes": scenes_with((0, 2), (0, 1), wall_share=0.0)},
    {"keys": ["runner.catch"], "commands": [("catch the red cheese", catch("catch the red cheese", "red"))], "scenes": scenes_with((1, 3), (0, 1), runner="red")},
    {
        "keys": ["place.box", "logic.all", "place.move"],
        "commands": [("go into the green box", go_box("go into the green box", "green")), ("eat everything", eat_all("eat everything")), ("move up", go_direction("move up", "up"))],
        "scenes": scenes_with((1, 3), (1, 2), (), ("green",)),
    },
]


def snapshot(world, i):
    return {"mass": world.mass[i].float().cpu().numpy(), "cheese_pos": world.cheese_pos[i].cpu().numpy(), "cheese_alive": world.cheese_alive[i].cpu().numpy()}


class Rollout:
    """One episode's progress through its commands: the command in force, its task (built from
    the scene as it stands when the command is given), when it was met and when it ends."""

    def __init__(self, commands, scene, first, cap, hold, final_hold):
        self.commands, self.cap, self.hold, self.final_hold = commands, cap, hold, final_hold
        self.segments = []
        self.finished_at = None
        self.begin(0, 0, scene, first)

    def begin(self, j, t, scene, first):
        self.j, self.start, self.scene, self.first = j, t, scene, first
        self.task = self.commands[j][1](scene)
        self.eaten = []
        self.met_at = None
        self.end = t + self.cap
        if self.task is None:  # the scene no longer allows the command (say, its cheese is gone)
            self.segments.append({"text": self.commands[j][0], "start": t, "met_at": None, "end": t, "success": False, "final_score": 0.0, "unasked": False, "eaten": [], "note": "not applicable"})
            self.finished_at = t

    def after_step(self, t, world, i, eaten_slots, runners):
        """Update after the step into state t (t >= 1); returns True when the episode ends at t."""
        for slot in eaten_slots:
            self.eaten.append(COLOUR_NAMES[int(world.cheese_color[i, slot])])
        score, unasked = T.score(self.task, self.scene, [self.first, snapshot(world, i)], self.eaten)
        last = self.j == len(self.commands) - 1
        if self.met_at is None and score >= 0.99 and not unasked:
            self.met_at = t
            self.end = min(t + (self.final_hold if last else self.hold), self.start + self.cap)
        if t < self.end:
            return False
        success = self.met_at is not None and score >= 0.99 and not unasked  # met, and still met at the end
        self.segments.append({"text": self.commands[self.j][0], "start": self.start, "met_at": self.met_at, "end": t, "success": bool(success), "final_score": round(float(score), 3), "unasked": bool(unasked), "eaten": list(self.eaten)})
        if last:
            self.finished_at = t
            return True
        self.begin(self.j + 1, t, scene_of(world, runners, i), snapshot(world, i))
        return self.finished_at is not None


@torch.no_grad()
def run_lm_batch(sample, tokenizer, instruction, scenes, seeds, args, log):
    """Roll out one instruction (its commands in turn) on each scene in one batch. A command runs
    until met plus args.hold steps (args.final_hold for the last), or args.cap steps. Returns the
    episodes, each with its segments and success (every command met and still met at its end)."""
    commands = instruction["commands"]
    device = torch.device(args.device)
    world, runners = T.stack(scenes)
    world = world.to(device)
    rollouts = [Rollout(commands, scene, snapshot(world, i), args.cap, args.hold, args.final_hold) for i, scene in enumerate(scenes)]
    recorders = [Recorder(scene, [text for text, _ in commands], tokenizer) for scene in scenes]
    eaten = torch.zeros(world.cheese_alive.shape, dtype=torch.bool)
    pushed = torch.zeros(world.cheese_alive.shape, dtype=torch.bool)
    t = 0
    while True:
        obs = observe(world)
        cond = tokens(tokenizer, [commands[r.j][0] for r in rollouts], device)
        _, field, probs = sample(obs, cond)
        codes = encode_field(field).cpu().numpy()
        obs_np, probs_np = obs.cpu().numpy(), probs.cpu().numpy()
        for i, (r, recorder) in enumerate(zip(rollouts, recorders)):
            if r.finished_at is None or t == r.finished_at:
                recorder.add(world.mass[i].cpu().numpy(), world.cheese_pos[i].cpu(), world.cheese_alive[i].cpu(), world.cheese_color[i].cpu(), obs_np[i], probs_np[i], codes[i], eaten[i].nonzero().flatten().tolist(), pushed[i].nonzero().flatten().tolist(), r.j)
        if all(r.finished_at is not None for r in rollouts):
            break
        world, info = step(world, field)
        eaten, pushed = info.eaten.cpu(), info.pushed.cpu()
        t += 1
        if runners.any() and t % T.RUNNER_PERIOD == 0:  # runner cheeses flee, as in tasks.run
            world = replace(world, cheese_pos=T.flee(world, runners))
        for i, r in enumerate(rollouts):
            if r.finished_at is None and r.after_step(t, world, i, eaten[i].nonzero().flatten().tolist(), runners):
                log(outcome(instruction, seeds[i], r, recorders[i]))
    return [recorder.episode(policy="lm", **outcome(instruction, seeds[i], r, recorder)) for i, (r, recorder) in enumerate(zip(rollouts, recorders))]


def outcome(instruction, seed, rollout, recorder):
    return {
        "instruction": " -> ".join(text for text, _ in instruction["commands"]),
        "intents": instruction["keys"],
        "seed": seed,
        "success": len(rollout.segments) == len(instruction["commands"]) and all(s["success"] for s in rollout.segments),
        "length": rollout.finished_at,  # steps
        "pushes": sum(len(s["pushed"]) for s in recorder.steps),
        "segments": rollout.segments,
    }


def centroid(mass):
    return np.argwhere(mass > 0.5).mean(0)


def find_scene(rng, kind):
    """A scene where the episode of this kind makes sense, and its goal and instruction."""
    while True:
        if kind == "eat":
            scene = T.sample_scene(rng, cheeses=(2, 3), boxes=(0, 1), require=("red",))
            if scene is None:
                continue
            slot = scene["cheese_colours"].index(COLOUR_NAMES.index("red"))
            y, x = scene["cheeses"][slot]
            away = np.array([y + 1.5, x + 1.5]) - centroid(scene["mass"])
            gap = np.hypot(*away)
            ahead = np.array([y + 1.5, x + 1.5]) + away / gap * 10  # where a push away from the swarm leads
            room = (ahead >= 4).all() and (ahead <= T.SIZE - 5).all()
            if 14 <= gap <= 24 and room and scene["walls"].mean() < 0.06:
                return scene, {"kind": "eat", "slot": slot}, "eat the red cheese"
        elif kind == "region":
            scene = T.sample_scene(rng, cheeses=(1, 2), boxes=(0, 1))
            if scene is None:
                continue
            far = np.hypot(*(centroid(scene["mass"]) - T.REGIONS["top left"])) >= 16
            clear = all(y > 13 or x > 13 for y, x in scene["cheeses"])
            if far and clear:
                return scene, {"kind": "region", "region": "top left"}, "go to the top left"
        else:
            scene = T.sample_scene(rng, cheeses=(0, 2), boxes=(1, 2))
            if scene is None:
                continue
            blue = COLOUR_NAMES.index("blue")
            if blue not in scene["box_colours"]:
                continue
            rect = scene["boxes"][scene["box_colours"].index(blue)]
            area = rect[2] * rect[3]
            gap = np.hypot(*(centroid(scene["mass"]) - (rect[0] + rect[2] / 2, rect[1] + rect[3] / 2)))
            if area >= scene["mass"].sum() and gap >= 10:
                return scene, {"kind": "box", "box": list(rect)}, "go into the blue box"


def rank(episode):
    """Order of preference among an instruction's runs: successes first, then runs that push a
    cheese, then the shortest."""
    return (episode["success"], episode["pushes"] > 0, -episode["length"])


def record_lm(args, log, save):
    """Roll out every instruction (args.only: a subset); save(results) after each, so a stopped run
    keeps what it finished."""
    from beebot.vla.lm_policy import PATCH
    from beebot.vla.lm_rl import RLConfig, lm_probs_sampler, load_policy

    policy, tokenizer = load_policy(RLConfig(init_run=args.init_run, checkpoint=args.checkpoint), torch.device(args.device))
    policy.eval()
    sample = lm_probs_sampler(policy, args.guidance, T.SIZE // PATCH)
    results = []
    for k, instruction in enumerate(INSTRUCTIONS):
        if args.only and str(k) not in args.only.split(","):
            continue
        scenes, seeds = [], []
        seed = 0
        while len(scenes) < args.rollouts:
            scene = instruction["scenes"](np.random.default_rng([k, seed]))
            if scene is not None and all(make(scene) is not None for _, make in instruction["commands"][:1]):
                scenes.append(scene)
                seeds.append(seed)
            seed += 1
        torch.manual_seed(k)
        episodes = run_lm_batch(sample, tokenizer, instruction, scenes, seeds, args, log)
        episodes.sort(key=rank, reverse=True)
        wins = sum(e["success"] for e in episodes)
        text = episodes[0]["instruction"]
        log({"instruction": text, "intents": instruction["keys"], "success_rate": f"{wins}/{len(episodes)}"})
        results.append(
            {
                "instruction": text,
                "intents": instruction["keys"],
                "successes": wins,
                "rollouts": len(episodes),
                "outcomes": [{key: e[key] for key in ("seed", "success", "length", "pushes", "segments")} for e in episodes],
                "runs": episodes[: args.keep],  # the best runs, best first
            }
        )
        save(results)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("dummy", "lm"), default="dummy")
    parser.add_argument("--seeds", default="7,0,0", help="dummy: scene and policy seed of the eat, region and box episodes")
    parser.add_argument("--max_steps", type=int, default=120, help="dummy: episode length")
    parser.add_argument("--out", default=None, help=f"default {OUT} (dummy), {LM_OUT} (lm)")
    parser.add_argument("--init_run", default="runs/kaggle/beebot-sft-long/runs/lm-expert-sft-long")
    parser.add_argument("--checkpoint", default="policy_latest.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--rollouts", type=int, default=16, help="lm: rollouts (fresh scenes) per instruction, one batch")
    parser.add_argument("--cap", type=int, default=140, help="lm: steps a command may take at most")
    parser.add_argument("--hold", type=int, default=10, help="lm: steps a command keeps going once met before the next command")
    parser.add_argument("--final_hold", type=int, default=15, help="lm: the same for the last command")
    parser.add_argument("--keep", type=int, default=6, help="lm: best runs kept per instruction")
    parser.add_argument("--only", default="", help="lm: comma-separated indices into INSTRUCTIONS (default all)")
    parser.add_argument("--log", default="logs/field-demo-record.out")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    out_path = Path(args.out or (LM_OUT if args.policy == "lm" else OUT))
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(args.log, "w")

    def log(row):
        line = json.dumps(row)
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    log({"pid": os.getpid(), "policy": args.policy, "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    note = "mock-up: driven by a hand-coded dummy policy (scripts/field_demo_record.py), not the trained model"
    def write(episodes, instructions):
        out = {
            "note": note,
            "policy": args.policy,
            "init_run": args.init_run if args.policy == "lm" else None,
            "checkpoint": args.checkpoint if args.policy == "lm" else None,
            "guidance": args.guidance if args.policy == "lm" else None,
            "size": T.SIZE,
            "cheese_side": 4,
            "codes": "0-7: world.DIRS (dy, dx) = " + json.dumps(DIRS) + "; 8: stay",
            "code_vectors": CODE_VECTORS.tolist(),
            "channels": CHANNELS,
            "episodes": episodes,  # dummy
            "instructions": instructions,  # lm
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, separators=(",", ":")))
        log({"wrote": str(out_path), "mb": round(out_path.stat().st_size / 1e6, 2)})

    episodes, instructions = [], []
    if args.policy == "lm":
        note = f"trained policy {args.init_run}/{args.checkpoint}, guidance {args.guidance}; per instruction every rollout's outcome and the best runs"
        instructions = record_lm(args, log, lambda results: write([], results))
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        episodes = []
        for kind, seed in zip(("eat", "region", "box"), (int(v) for v in args.seeds.split(","))):
            scene, goal, instruction = find_scene(np.random.default_rng(seed), kind)
            episode = run_dummy_episode(scene, goal, instruction, tokenizer, args.max_steps, seed)
            log({"instruction": instruction, "seed": seed, "steps": len(episode["steps"]) - 1, "robots": episode["robot_count"], "eaten_at": episode["eaten_at"], "pushes": episode["pushes"]})
            episodes.append(episode)
    write(episodes, instructions)


if __name__ == "__main__":
    main()
