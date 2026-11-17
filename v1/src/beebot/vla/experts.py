"""Task-conditioned expert demonstrations for the 2D task suite (vla/tasks.py): for a scene and
an instruction, a scripted expert drives the shared controller (vla/controller.py), and the
episode is kept only if the suite's own checker scores it a success. Behaviour cloning on these
gives the policy the skills hindsight-labelled play lacks (formation, herding, logic over the
whole scene) and the same-scene contrast it needs to read instructions: each scene is played
under its own instruction and, where another family fits, under a second one.

    uv run python -m beebot.vla.experts --out data/vla/experts_g32 --episodes 20000

Experts per family:
  place      one attractor: the box's centre, the region's point, or 12 squares along the direction
  logic      the cheeses to eat, one at a time (in order, or nearest first; or-forms: the nearest of
             the pair; eat-nothing forms: no attractor); an attractor on a cheese targets its ring
  runner     the same, chasing the runner's current position
  parallel   one attractor per job with half the swarm each (region point, box centre, cheese)
  herd       go behind the cheese (routed), then press straight through it along a cardinal
             direction; into a box: push along the axis with the larger gap, re-planned every push
  formation  a mask field: every robot off the target walks toward the nearest unfilled target
             square, robots on the target stay

Writes the play-store format (play_data.cache_play) so data.batch_from reads it: mass.npy,
field.npy (direction codes), small.pt, done, and labels.jsonl {"e", "family", "held_out",
"text", "score"}. Training split only: no held-out item is ever demonstrated.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from beebot.vla import tasks as T
from beebot.vla.controller import Controller, commands_to_tensors, walking_distance
from beebot.vla.data import encode_field
from beebot.vla.goal_eval import FACT_EVERY, snapshot
from beebot.vla.world import COLOUR_NAMES, DIRS, World, cheese_side, step

STEPS = 64
SUCCESS = 0.9
FORMATION_SUCCESS = 0.6  # F1 against the mask: a recognisable shape (exact fills are rare)
PUSH_REACH = 12  # how far past the cheese a push aims
BEHIND = 2  # squares behind the cheese's face where pushers gather
PUSH_SHARE = 0.3  # of the swarm that pushes; the rest parks PARK squares further back
PARK = 8
PUSHERS = 0  # > 0: at most this many robots push (a fixed count, not a share of the swarm)
CLEAR = 2  # herding starts by clearing everyone this close to the cheese away ...
CLEAR_STEPS = 12  # ... for at most this many steps
NUDGE_DONE = 5  # squares moved along the asked direction (the checker credits 4)
SURROUND_ROBOTS = 12  # robots sent to surround a cheese: enough to ring it, too few to eat it
CARDINAL = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def _centre(pos, side):
    return (pos[0] + (side - 1) / 2, pos[1] + (side - 1) / 2)


def _point(y, x):
    return {"row": int(round(min(max(y, 0), T.SIZE - 1))), "col": int(round(min(max(x, 0), T.SIZE - 1)))}


class Expert:
    """The expert for one (scene, task): commands(world, i, t) -> list of attractor dicts, or a
    mask (H, W) bool for formation."""

    def __init__(self, scene, task):
        self.scene, self.task, self.spec = scene, task, task.spec
        self.side = cheese_side(T.SIZE)
        self.slot = {COLOUR_NAMES[c]: k for k, c in enumerate(scene["cheese_colours"])}
        rows, cols = np.nonzero(scene["mass"] >= 0.5)
        self.start = (rows.mean(), cols.mean()) if len(rows) else (T.SIZE / 2, T.SIZE / 2)
        self.push = None  # (direction, phase, steps in phase) for herding
        self.origin = None  # the herded cheese's starting corner
        self.cleared, self.clear_steps = False, 0  # herding: the initial clearing phase

    def alive(self, world, i, colour):
        k = self.slot[colour]
        return bool(world.cheese_alive[i, k])

    def cheese_point(self, world, i, colour):
        y, x = world.cheese_pos[i, self.slot[colour]].tolist()
        return _point(*_centre((y, x), self.side))

    def commands(self, world, i, t):
        f, s = self.task.family, self.spec
        if f == "formation":
            return s["mask"]
        if f == "place":
            if "box" in s:
                y, x, h, w = s["box"]
                return [_point(y + (h - 1) / 2, x + (w - 1) / 2)]
            if "region" in s:
                return [_point(*T.REGIONS[s["region"]])]
            dy, dx = T.DIRECTIONS[s["direction"]]
            return [_point(self.start[0] + PUSH_REACH * dy, self.start[1] + PUSH_REACH * dx)]
        if f in ("logic", "runner"):
            return self.eat_commands(world, i)
        if f == "parallel":
            out = []
            for kind, x in s["jobs"]:
                if kind == "region":
                    out.append({**_point(*T.REGIONS[x]), "share": 0.5})
                elif kind == "box":
                    y, x0, h, w = s["rects"][x]
                    out.append({**_point(y + (h - 1) / 2, x0 + (w - 1) / 2), "share": 0.5})
                elif self.alive(world, i, x):
                    robots = max(float((world.mass[i] >= 0.5).sum()), 1.0)
                    share = min(0.5, SURROUND_ROBOTS / robots) if kind == "surround" else 0.5
                    out.append({**self.cheese_point(world, i, x), "share": share})
            return out
        if f == "herd":
            return self.herd_commands(world, i)
        raise ValueError(f)

    def eat_commands(self, world, i):
        s = self.spec
        if s["order"]:
            todo = [c for c in s["order"] if self.alive(world, i, c)]
        elif s["any_of"]:
            todo = [] if any(not self.alive(world, i, c) for c in s["any_of"]) else sorted(s["any_of"], key=lambda c: self.distance(world, i, c))[:1]
        else:
            todo = sorted((c for c in s["eat"] if self.alive(world, i, c)), key=lambda c: self.distance(world, i, c))
        return [self.cheese_point(world, i, todo[0])] if todo else []

    def distance(self, world, i, colour):
        mass = world.mass[i] >= 0.5
        rows, cols = torch.nonzero(mass, as_tuple=True)
        if not len(rows):
            return 0.0
        p = self.cheese_point(world, i, colour)
        return float(np.hypot(rows.float().mean().item() - p["row"], cols.float().mean().item() - p["col"]))

    def clearing(self, world, i, pos):
        """True while any robot is within CLEAR squares of the herded cheese, for at most
        CLEAR_STEPS steps from the start (a swarm that starts around a cheese near a wall
        covers its small open ring and eats it as soon as anyone moves)."""
        if self.cleared:
            return False
        y, x = pos
        mass = world.mass[i] >= 0.5
        near = mass[max(y - CLEAR, 0) : y + self.side + CLEAR, max(x - CLEAR, 0) : x + self.side + CLEAR].any().item()
        self.clear_steps += 1
        if not near or self.clear_steps > CLEAR_STEPS:
            self.cleared = True
            return False
        return True

    def herd_commands(self, world, i):
        s = self.spec
        if "nudge" in s:
            colour, direction = s["nudge"]
            d = CARDINAL[direction]
        else:
            colour, rect = s["into"][0][0], s["into"][1]
            y, x, h, w = rect
            cy, cx = _centre(world.cheese_pos[i, self.slot[colour]].tolist(), self.side)
            gy, gx = y + (h - 1) / 2 - cy, x + (w - 1) / 2 - cx
            d = (int(np.sign(gy)), 0) if abs(gy) >= abs(gx) else (0, int(np.sign(gx)))
            if d == (0, 0):
                return []
        if not self.alive(world, i, colour):
            return []
        pos = world.cheese_pos[i, self.slot[colour]].tolist()
        if self.origin is None:
            self.origin = pos
        cy, cx = _centre(pos, self.side)
        done = (pos[0] - self.origin[0]) * d[0] + (pos[1] - self.origin[1]) * d[1] >= NUDGE_DONE if "nudge" in s else T.cheese_in_box(pos, s["into"][1], self.side)
        if done:  # back off, so nobody keeps pressing or wraps around it
            return [_point(cy - d[0] * (PUSH_REACH + PARK), cx - d[1] * (PUSH_REACH + PARK))]
        if self.push is None or self.push[0] != d:
            self.push = [d, "gather", 0]
        self.push[2] += 1
        reach = self.side / 2 + BEHIND
        behind = (cy - d[0] * reach, cx - d[1] * reach)
        # A small pushing group; the rest park well behind it, so nobody wraps around the cheese
        # (covering most of its ring eats it).
        robots = max(float((world.mass[i] >= 0.5).sum()), 1.0)
        push_share = min(PUSH_SHARE, PUSHERS / robots) if PUSHERS else PUSH_SHARE
        # Park at the corner farthest from the cheese: "further behind" falls off the grid for
        # pushes toward an edge, and clamped back it lands beside the pushers.
        corners = [(2, 2), (2, T.SIZE - 3), (T.SIZE - 3, 2), (T.SIZE - 3, T.SIZE - 3)]
        far = max(corners, key=lambda c: (c[0] - cy) ** 2 + (c[1] - cx) ** 2)
        park = {**_point(*far), "share": 1 - push_share}
        if self.push[1] == "gather" and self.clearing(world, i, pos):
            return [_point(*far)]  # robots start touching the cheese: everyone clears off first
        if self.push[1] == "gather":
            mass = world.mass[i] >= 0.5
            near = mass[max(int(behind[0]) - 2, 0) : int(behind[0]) + 3, max(int(behind[1]) - 2, 0) : int(behind[1]) + 3].sum().item()
            if near >= 6 or self.push[2] > 24:
                self.push[1], self.push[2] = "press", 0
            return [{**_point(*behind), "share": push_share}, park]
        if self.push[2] > 16:  # re-plan: gather behind again from the cheese's new position
            self.push[1], self.push[2] = "gather", 0
        return [{**_point(cy + d[0] * PUSH_REACH, cx + d[1] * PUSH_REACH), "straight": True, "share": push_share}, park]


def mask_field(world, i, mask):
    """(2, H, W) field for one scene: robots off the target step toward the nearest unfilled target
    square (walking distance around walls and cheeses); robots on the target stay."""
    mass = world.mass[i] >= 0.5
    target = torch.as_tensor(mask, device=mass.device)
    blocked = world.walls[i].clone()
    side = cheese_side(T.SIZE)
    for (y, x), alive in zip(world.cheese_pos[i].tolist(), world.cheese_alive[i].tolist()):
        if alive:
            blocked[y : y + side, x : x + side] = True
    unfilled = target & ~mass & ~blocked
    field = torch.zeros(2, T.SIZE, T.SIZE, device=mass.device)
    if not unfilled.any():
        return field
    settled = mass & target
    # Paths run around settled robots (they stay put), not through them: through them, the robots
    # behind would press on robots that never move, and the formation jams.
    d = walking_distance(unfilled.unsqueeze(0), (~blocked).unsqueeze(0), (~blocked & ~settled).unsqueeze(0))[0]
    best = d.clone()
    d = torch.where(settled, float("inf"), d)  # never step onto a settled robot's square
    step_y = torch.zeros_like(d)
    step_x = torch.zeros_like(d)
    for dy, dx in DIRS:
        shifted = torch.full_like(d, float("inf"))
        ys = slice(max(-dy, 0), T.SIZE - max(dy, 0))
        yd = slice(max(dy, 0), T.SIZE - max(-dy, 0))
        xs = slice(max(-dx, 0), T.SIZE - max(dx, 0))
        xd = slice(max(dx, 0), T.SIZE - max(-dx, 0))
        shifted[ys, xs] = d[yd, xd]  # the distance at the neighbour (y + dy, x + dx)
        better = shifted < best
        best = torch.where(better, shifted, best)
        step_y = torch.where(better, torch.full_like(d, float(dy)), step_y)
        step_x = torch.where(better, torch.full_like(d, float(dx)), step_x)
    move = mass & ~target
    field[0] = torch.where(move, step_x, 0.0)
    field[1] = torch.where(move, step_y, 0.0)
    return field


@torch.no_grad()
def play(scenes, task_list, steps=STEPS, device="cpu", seed=0):
    """Expert episodes for (scene, task) pairs: per episode (score, unasked) and the recording
    {mass (B, T+1, H, W) float16, field (B, T, H, W) int8 codes, cheese_pos, cheese_alive}."""
    world, runners = T.stack(scenes)
    world, runners = world.to(device), runners.to(device)
    experts = [Expert(s, t) for s, t in zip(scenes, task_list)]
    controller = Controller()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    frames = [[snapshot(world, i)] for i in range(len(scenes))]
    eaten = [[] for _ in scenes]
    rec = {"mass": [world.mass.half().cpu()], "field": [], "cheese_pos": [world.cheese_pos.to(torch.int8).cpu()], "cheese_alive": [world.cheese_alive.cpu()]}
    alive_before = world.cheese_alive.cpu()
    for t in range(steps):
        orders = [e.commands(world, i, t) for i, e in enumerate(experts)]
        commands = [o if isinstance(o, list) else [] for o in orders]
        points, valid, straight, share = commands_to_tensors(commands, len(scenes), device)
        field = controller(world, points, valid, straight, share)
        for i, o in enumerate(orders):
            if not isinstance(o, list):
                field[i] = mask_field(world, i, o)
        codes = encode_field(field)
        world, _ = step(world, field, torch.rand(world.mass.shape, generator=generator).to(device))
        if runners.any() and (t + 1) % T.RUNNER_PERIOD == 0:
            world = World(mass=world.mass, walls=world.walls, cheese_pos=T.flee(world, runners), cheese_alive=world.cheese_alive, cheese_color=world.cheese_color, box_rect=world.box_rect, box_color=world.box_color)
        alive_now = world.cheese_alive.cpu()
        for i, slot in (alive_before & ~alive_now).nonzero().tolist():
            eaten[i].append(COLOUR_NAMES[int(world.cheese_color[i, slot])])
        alive_before = alive_now
        if (t + 1) % FACT_EVERY == 0 or t == steps - 1:
            for i in range(len(scenes)):
                frames[i].append(snapshot(world, i))
        rec["field"].append(codes.cpu())
        rec["mass"].append(world.mass.half().cpu())
        rec["cheese_pos"].append(world.cheese_pos.to(torch.int8).cpu())
        rec["cheese_alive"].append(world.cheese_alive.cpu())
    results = [T.score(task, scene, f, e) for task, scene, f, e in zip(task_list, scenes, frames, eaten)]
    rec = {k: torch.stack(v, 1) for k, v in rec.items()}
    rec.update({"walls": world.walls.cpu(), "cheese_color": world.cheese_color.cpu(), "box_rect": world.box_rect.cpu(), "box_color": world.box_color.cpu()})
    return results, rec


def succeeded(task, result):
    score, unasked = result
    if unasked:
        return False
    if task.family == "logic" and not task.spec["eat"] and not task.spec["any_of"]:  # eat nothing: not eating is the success
        return True
    return score >= (FORMATION_SUCCESS if task.family == "formation" else SUCCESS)


def draw(rng, n, contrast=0.5, families=None):
    """n (scene, task) pairs of the training split, families uniform; with probability contrast a
    scene also gets a second instruction from another family that fits it (same-scene contrast)."""
    out = []
    while len(out) < n:
        scene, task = T.sample_task(rng, "train", family=str(rng.choice(families)) if families else None)
        out.append((scene, task))
        if rng.random() < contrast and len(out) < n:
            for family in rng.permutation(T.FAMILIES):
                if family in (task.family, "runner"):
                    continue
                _, make = T.MAKERS[str(family)](rng, "")
                other = make(scene)
                if other is not None:
                    out.append((scene, other))
                    break
    return out


def write(out_dir, episodes, seed=0, batch=64, device="cpu", families=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    parts, labels, tried = [], [], {}
    t0 = time.time()
    kept = 0
    while kept < episodes:
        pairs = draw(rng, batch, families=families)
        results, rec = play([s for s, _ in pairs], [t for _, t in pairs], device=device, seed=int(rng.integers(1 << 30)))
        keep = [i for i, ((_, task), r) in enumerate(zip(pairs, results)) if succeeded(task, r)]
        for (_, task), r in zip(pairs, results):
            tried.setdefault(task.family, [0, 0])
            tried[task.family][0] += 1
            tried[task.family][1] += int(succeeded(task, r))
        if keep:
            idx = torch.tensor(keep)
            parts.append({k: v[idx] for k, v in rec.items()})
            for i in keep:
                labels.append({"e": kept, "family": pairs[i][1].family, "held_out": "", "text": pairs[i][1].text, "score": round(float(results[i][0]), 3)})
                kept += 1
        print(f"{kept}/{episodes} kept, {time.time() - t0:.0f} s, success by family {json.dumps({f: f'{s}/{n}' for f, (n, s) in sorted(tried.items())})}", flush=True)
    store = {k: torch.cat([p[k] for p in parts])[:episodes] for k in parts[0]}
    labels = labels[:episodes]
    np.save(out / "mass.npy", store.pop("mass").numpy())
    np.save(out / "field.npy", store.pop("field").numpy())
    torch.save({**store, "episode_weight": torch.ones(len(labels)), "phases": [[l["family"]] for l in labels], "source": [("expert", l["family"], l["e"]) for l in labels]}, out / "small.pt")
    with open(out / "labels.jsonl", "w") as f:
        for l in labels:
            f.write(json.dumps(l) + "\n")
    (out / "done").write_text(str(len(labels)))
    (out / "success.json").write_text(json.dumps(tried))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/vla/experts_g32")
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--families", default="", help="comma-separated: only these families' tasks (their scenes may still get a second instruction of another family)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    write(args.out, args.episodes, args.seed, args.batch, families=args.families.split(",") if args.families else None)


if __name__ == "__main__":
    main()
