"""The generalist task suite: generated scenes, instruction families with their checkers, held-out
splits and runner cheeses. lm_rl trains on it; the generalisation tests score held-out tasks with
the same checkers.

Families (each makes an instruction from what its scene holds, and scores an episode in [0, 1]):
  formation   spell a word or letter, or form a shape; F1 of occupied squares against the mask.
  logic       eat by colour with and / or / not / all / none / order / nearest-to / furthest;
              asking for an absent colour asks for nothing.
  place       go into a box, to a region, or move a direction; dense distance scores.
  parallel    two subtasks at once (region, box, eat, surround); split credit by group.
  herd        push a cheese into a box, or nudge it a direction without eating it.
  runner      catch a cheese that runs from the swarm (see flee).
Every family also charges the disrupt penalty (lm_rl) when the episode ends having eaten a cheese
it did not ask to eat.

Held out (never in training scenes or instructions): the colour pink; letters K, Q, Z and the
words HI, OK, GO, UP; the heart and triangle shapes; the logic forms "neither ... nor" and "all but
two"; parallel pairs surround+box and eat+region; whole families herd_parallel (two pushes at
once) and runner_logic (logic over runner cheeses). sample_task(split="train") never draws them;
split="test" draws in-distribution tasks; split=<a HELD_OUT key> forces that held-out item.
"""

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from beebot.vla import goal_eval
from beebot.vla.facts import AROUND_SHARE, BOX_IN, box_share, cheese_in_box, ring_share
from beebot.vla.shapes import FONT, SHAPES, formation_score, shape_mask, text_mask
from beebot.vla.world import COLOUR_NAMES, MASS_RANGE, World, _grow_blob, _place_boxes, _place_cheeses, cheese_side, largest_open_region, make_terrain, observe, step, zoom

SIZE = 32
CHEESE_SLOTS = 8
BOX_SLOTS = 3
HELD_OUT = {
    "colour": ("pink",),
    "letters": tuple("KQZ"),
    "words": ("HI", "OK", "GO", "UP"),
    "shapes": ("heart", "triangle"),
    "logic": ("neither", "all_but_two"),
    "pairs": (("box", "surround"), ("eat", "region")),
    "herd_parallel": (),
    "runner_logic": (),
    "wording": (),  # paraphrases of every intent that training never uses (scripts/vla_task_wordings.py)
}
TRAIN_COLOURS = tuple(c for c in COLOUR_NAMES if c not in HELD_OUT["colour"])
REGIONS = {
    "top left": (5, 5), "top": (5, 16), "top right": (5, 26),
    "left": (16, 5), "center": (16, 16), "right": (16, 26),
    "bottom left": (26, 5), "bottom": (26, 16), "bottom right": (26, 26),
}
DIRECTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1), "up and left": (-0.71, -0.71), "up and right": (-0.71, 0.71), "down and left": (0.71, -0.71), "down and right": (0.71, 0.71)}
SHAPE_WORDS = {"ring": "ring", "disc": "solid circle", "square": "square", "plus": "plus sign", "x": "big X", "horizontal line": "horizontal line", "vertical line": "vertical line", "diagonal line": "diagonal line", "triangle": "triangle", "heart": "heart"}
FAMILIES = ("formation", "logic", "place", "parallel", "herd", "runner")
RUNNER_RADIUS = 6  # squares a runner senses robots within
RUNNER_PERIOD = 2  # a runner moves at most once every this many steps: half the swarm's speed


WORDINGS = {
    'formation.letter': ['form the letter {w}', 'make a letter {w}', 'draw a big {w}', 'spell {w}'],
    'formation.shape': ['form a {s}', 'make a {s}', 'arrange yourselves into a {s}', 'make a {s} shape', 'everyone line up as a {s}'],
    'formation.word': ['spell {w}', 'write {w}', 'form the word {w}', 'spell out {w} with your bodies', 'make the letters {w}'],
    'herd.into': ['push the {x} cheese into the {b} box', 'herd the {x} cheese into the {b} box', 'get the {x} cheese inside the {b} box without eating it'],
    'herd.nudge': ['nudge the {c} cheese {d} without eating it', 'push the {c} cheese {d}', "move the {c} cheese {d}, don't eat it"],
    'herd.two': ['push the {x} and {y} cheeses into the {b} box', 'get both {x} and {y} into the {b} box'],
    'logic.all': ['eat all the cheese', 'eat everything', 'clear the board of cheese', 'eat every cheese you see'],
    'logic.all_but_two': ['eat all but the {a} and {b} ones', 'eat everything except {a} and {b}'],
    'logic.and': ['eat the {a} and {b} cheeses', 'eat both {a} and {b}', 'eat {a}, and also {b}'],
    'logic.except': ['eat everything except the {a} cheese', 'eat all the cheese but {a}', 'leave the {a} one alone and eat the rest', "eat every cheese that isn't {a}"],
    'logic.furthest': ['eat the cheese furthest from you', "eat the cheese that's farthest away", 'go for the most distant cheese and eat it'],
    'logic.nearest_box': ['eat the cheese closest to the {b} box', 'eat whichever cheese is nearest the {b} box'],
    'logic.neither': ['eat neither {a} nor {b}, but everything else', "don't eat {a} or {b}; eat the rest"],
    'logic.none': ["don't eat anything", 'leave the cheese alone', 'do not eat any cheese', 'stay away from the food'],
    'logic.or': ['eat the {a} or the {b} cheese', 'eat either {a} or {b}', 'eat one of the {a} and {b} cheeses'],
    'logic.ordered': ['eat {a} first, then {b}', 'eat the {a} cheese and after that the {b} one', 'first {a}, then {b}'],
    'logic.single': ['eat the {a} cheese', 'go eat the {a} one', 'eat {a}', 'gobble up the {a} cheese', 'find the {a} cheese and eat it'],
    'parallel.split': ['split up: {a}, and {b}', 'half of you {a}, the other half {b}', 'split into two groups: one should {a}, the other {b}', 'two teams: one team {a}, the other {b}'],
    'place.box': ['go into the {b} box', 'everyone into the {b} box', 'get inside the {b} box', 'move into the {b} box'],
    'place.move': ['move {d}', 'go {d}', 'shift {d}', 'head {d}'],
    'place.region': ['go to the {r}', 'gather in the {r}', 'move to the {r} of the grid', 'head for the {r}'],
    'runner.catch': ['catch the {c} cheese', 'chase down the {c} one and eat it', 'catch {c} before it gets away', 'hunt the {c} cheese'],
    'runner.except': ['catch every cheese except the {a} one', 'eat all of them but {a}'],
}
WORDINGS_PATH = Path(__file__).with_name("task_wordings.json")
EXTRA_WORDINGS = json.loads(WORDINGS_PATH.read_text()) if WORDINGS_PATH.exists() else {}  # scripts/vla_task_wordings.py


@dataclass
class Task:
    family: str
    text: str
    spec: dict  # what the checker needs: target sets, masks, slots, ...
    held_out: str  # "" or the HELD_OUT key it tests


# ---------------------------------------------------------------------------------------------
# Scenes


def sample_scene(rng, colours=TRAIN_COLOURS, cheeses=(0, 6), boxes=(0, 3), wall_share=0.5, runners=0, require=()):
    """One scene dict (numpy), or None if placement failed. cheeses / boxes: (low, high) counts.
    require: colour names that must be among the cheeses."""
    density = rng.uniform(0.03, 0.15) if rng.random() < wall_share else 0.0
    walls = ~largest_open_region(make_terrain(rng, SIZE, density))
    open_mask = ~walls
    if open_mask.mean() < 0.6:
        return None
    n_cheeses = int(rng.integers(cheeses[0], cheeses[1] + 1))
    n_boxes = int(rng.integers(boxes[0], boxes[1] + 1))
    placed = _place_cheeses(rng, open_mask, n_cheeses, SIZE)
    if len(placed) < n_cheeses:
        return None
    side = cheese_side(SIZE)
    near = np.zeros_like(open_mask)
    for y, x in placed:
        near[max(y - 1, 0) : y + side + 1, max(x - 1, 0) : x + side + 1] = True
    rects = _place_boxes(rng, open_mask, n_boxes, SIZE, near)
    if len(rects) < n_boxes:
        return None
    total = int(rng.integers(MASS_RANGE[0] * zoom(SIZE) ** 2, MASS_RANGE[1] * zoom(SIZE) ** 2 + 1))
    mass = np.zeros((SIZE, SIZE), np.float32)
    for share in np.array_split(np.arange(total), 1 if rng.random() < 0.7 else 2):
        allowed = open_mask & ~near & (mass == 0)
        if not allowed.any():
            return None
        for cell in _grow_blob(rng, allowed, len(share), SIZE):
            mass[cell] = 1.0
    pool = [COLOUR_NAMES.index(c) for c in colours]
    must = [COLOUR_NAMES.index(c) for c in require]
    if len(must) > n_cheeses or n_cheeses > len(pool):
        return None
    rest = [c for c in pool if c not in must]
    cheese_colours = must + [int(c) for c in rng.choice(rest, size=n_cheeses - len(must), replace=False)]
    rng.shuffle(cheese_colours)
    box_colours = [int(c) for c in rng.choice(pool, size=n_boxes, replace=False)]
    return {"mass": mass, "walls": walls, "cheeses": placed, "cheese_colours": cheese_colours, "boxes": rects, "box_colours": box_colours, "runners": list(range(min(runners, n_cheeses)))}


def stack(scenes):
    """A World of the scenes (CHEESE_SLOTS cheese slots, BOX_SLOTS boxes) and runners (B, K) bool."""
    b = len(scenes)
    cheese_pos = torch.zeros(b, CHEESE_SLOTS, 2, dtype=torch.long)
    cheese_alive = torch.zeros(b, CHEESE_SLOTS, dtype=torch.bool)
    cheese_color = torch.full((b, CHEESE_SLOTS), -1, dtype=torch.long)
    box_rect = torch.zeros(b, BOX_SLOTS, 4, dtype=torch.long)
    box_color = torch.full((b, BOX_SLOTS), -1, dtype=torch.long)
    runners = torch.zeros(b, CHEESE_SLOTS, dtype=torch.bool)
    for i, s in enumerate(scenes):
        for k, ((y, x), c) in enumerate(zip(s["cheeses"], s["cheese_colours"])):
            cheese_pos[i, k] = torch.tensor([y, x])
            cheese_alive[i, k] = True
            cheese_color[i, k] = c
        for k, (rect, c) in enumerate(zip(s["boxes"], s["box_colours"])):
            box_rect[i, k] = torch.tensor(rect)
            box_color[i, k] = c
        runners[i, s["runners"]] = True
    world = World(
        mass=torch.from_numpy(np.stack([s["mass"] for s in scenes])).float(),
        walls=torch.from_numpy(np.stack([s["walls"] for s in scenes])),
        cheese_pos=cheese_pos,
        cheese_alive=cheese_alive,
        cheese_color=cheese_color,
        box_rect=box_rect,
        box_color=box_color,
    )
    return world, runners


def flee(world, runners):
    """Each alive runner cheese moves one square toward the side (up, down, left, right) with the
    least robot mass within RUNNER_RADIUS of it, if it senses any robot and the squares it enters
    hold no wall, cheese or robot. Returns new cheese positions (B, K) on world's device."""
    side = cheese_side(SIZE)
    mass = world.mass.cpu().numpy()
    walls = world.walls.cpu().numpy()
    pos = world.cheese_pos.cpu().numpy().copy()
    alive = world.cheese_alive.cpu().numpy()
    run = runners.cpu().numpy() & alive
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    for b, k in zip(*np.nonzero(run)):
        y, x = pos[b, k]
        cy, cx = y + (side - 1) / 2, x + (side - 1) / 2
        dy, dx = yy - cy, xx - cx
        near = (np.abs(dy) <= RUNNER_RADIUS) & (np.abs(dx) <= RUNNER_RADIUS)
        cones = {(-1, 0): near & (dy < 0) & (np.abs(dx) <= -dy), (1, 0): near & (dy > 0) & (np.abs(dx) <= dy), (0, -1): near & (dx < 0) & (np.abs(dy) <= -dx), (0, 1): near & (dx > 0) & (np.abs(dy) <= dx)}
        sensed = {d: float(mass[b][m].sum()) for d, m in cones.items()}
        if sum(sensed.values()) == 0:
            continue
        others = np.zeros((SIZE, SIZE), bool)
        for j in np.nonzero(alive[b])[0]:
            if j != k:
                oy, ox = pos[b, j]
                others[oy : oy + side, ox : ox + side] = True
        for d in sorted(sensed, key=lambda d: (sensed[d], -sensed[(-d[0], -d[1])])):
            ny, nx = y + d[0], x + d[1]
            if ny < 0 or nx < 0 or ny + side > SIZE or nx + side > SIZE:
                continue
            new = np.zeros((SIZE, SIZE), bool)
            new[ny : ny + side, nx : nx + side] = True
            new[y : y + side, x : x + side] = False
            if (walls[b] & new).any() or (others & new).any() or (mass[b][new] >= 0.5).any():
                continue
            pos[b, k] = (ny, nx)
            break
    return torch.from_numpy(pos).to(world.cheese_pos.device)


# ---------------------------------------------------------------------------------------------
# Families: (rng, held) -> (scene requirements, a function scene -> Task or None)


def wordings(key, held):
    """The templates for intent key: the seed ones and the extra pool's training ones, or (held ==
    "wording") the pool's held-out ones (about a fifth, by a stable hash of the text)."""
    extra = EXTRA_WORDINGS.get(key, [])
    test = [t for t in extra if zlib.crc32(t.encode()) % 5 == 0]
    if held == "wording" and test:
        return test
    return WORDINGS[key] + [t for t in extra if t not in test]


def say(rng, key, held, **kw):
    return str(rng.choice(wordings(key, held))).format(**kw)


def names(scene, idx):
    return [COLOUR_NAMES[c] for c in idx]


def formation(rng, held):
    def make(scene):
        if held in ("letters", "words") or (held in ("", "wording") and rng.random() < 0.5):
            if held == "words":
                word = str(rng.choice(HELD_OUT["words"]))
            else:
                allowed = [ch for ch in FONT if ch not in HELD_OUT["letters"]]
                while True:
                    letters = [str(rng.choice(allowed)) for _ in range(int(rng.integers(1, 3)))]
                    if held == "letters":
                        letters[int(rng.integers(len(letters)))] = str(rng.choice(HELD_OUT["letters"]))
                    word = "".join(letters)
                    if held == "letters" or word not in HELD_OUT["words"]:
                        break
            mask = text_mask(word, SIZE)
            text = say(rng, "formation.letter" if len(word) == 1 else "formation.word", held, w=word)
        else:
            pool = HELD_OUT["shapes"] if held == "shapes" else [s for s in SHAPES if s not in HELD_OUT["shapes"]]
            name = str(rng.choice(pool))
            mask = shape_mask(name, SIZE)
            text = say(rng, "formation.shape", held, s=SHAPE_WORDS[name])
        if (mask & scene["walls"]).any():
            return None
        return Task("formation", text, {"mask": mask}, held)

    return {"cheeses": (0, 2), "boxes": (0, 1), "wall_share": 0.0}, make


def logic(rng, held):
    forms = HELD_OUT["logic"] if held == "logic" else ["single", "except", "or", "and", "all", "none", "ordered", "nearest_box", "furthest", "absent"]
    form = str(rng.choice(forms))

    def make(scene):
        present = names(scene, scene["cheese_colours"])
        p = list(present)
        rng.shuffle(p)
        everything = set(present)
        spec = {"eat": set(), "any_of": set(), "order": [], "forbid": set()}
        if form in ("except", "or", "and", "ordered", "neither", "all_but_two") and len(p) < (3 if form in ("neither", "all_but_two") else 2):
            return None
        if form == "single":
            a = p[0]
            spec.update(eat={a}, forbid=everything - {a})
            text = say(rng, "logic.single", held, a=a)
        elif form == "except":
            a = p[0]
            spec.update(eat=everything - {a}, forbid={a})
            text = say(rng, "logic.except", held, a=a)
        elif form == "or":
            a, b = p[:2]
            spec.update(any_of={a, b}, forbid=everything - {a, b})
            text = say(rng, "logic.or", held, a=a, b=b)
        elif form == "and":
            a, b = p[:2]
            spec.update(eat={a, b}, forbid=everything - {a, b})
            text = say(rng, "logic.and", held, a=a, b=b)
        elif form == "all":
            spec.update(eat=everything)
            text = say(rng, "logic.all", held)
        elif form == "none":
            spec.update(forbid=everything)
            text = say(rng, "logic.none", held)
        elif form == "ordered":
            a, b = p[:2]
            spec.update(eat={a, b}, order=[a, b], forbid=everything - {a, b})
            text = say(rng, "logic.ordered", held, a=a, b=b)
        elif form in ("nearest_box", "furthest"):
            if not p or (form == "nearest_box" and not scene["boxes"]):
                return None
            side = cheese_side(SIZE)
            centres = {COLOUR_NAMES[c]: (y + side / 2, x + side / 2) for (y, x), c in zip(scene["cheeses"], scene["cheese_colours"])}
            if form == "nearest_box":
                k = int(rng.integers(len(scene["boxes"])))
                by, bx, bh, bw = scene["boxes"][k]
                ref = (by + bh / 2, bx + bw / 2)
                target = min(centres, key=lambda c: np.hypot(centres[c][0] - ref[0], centres[c][1] - ref[1]))
                text = say(rng, "logic.nearest_box", held, b=COLOUR_NAMES[scene["box_colours"][k]])
            else:
                rows, cols = np.nonzero(scene["mass"])
                ref = (rows.mean(), cols.mean())
                target = max(centres, key=lambda c: np.hypot(centres[c][0] - ref[0], centres[c][1] - ref[1]))
                text = say(rng, "logic.furthest", held)
            spec.update(eat={target}, forbid=everything - {target})
        elif form == "absent":
            missing = [c for c in TRAIN_COLOURS if c not in everything]
            a = str(rng.choice(missing))
            spec.update(forbid=everything)
            text = say(rng, "logic.single", held, a=a)
        else:  # neither, all_but_two (held out)
            a, b = p[:2]
            spec.update(eat=everything - {a, b}, forbid={a, b})
            if form == "neither":
                text = say(rng, "logic.neither", held, a=a, b=b)
            else:
                text = say(rng, "logic.all_but_two", held, a=a, b=b)
        spec["form"] = form
        return Task("logic", text, spec, held)

    return {"cheeses": (1, 6), "boxes": (0, 2)}, make


def place(rng, held):
    kind = str(rng.choice(["box", "region", "move"]))

    def make(scene):
        if kind == "box":
            if not scene["boxes"]:
                return None
            k = int(rng.integers(len(scene["boxes"])))
            name = COLOUR_NAMES[scene["box_colours"][k]]
            text = say(rng, "place.box", held, b=name)
            return Task("place", text, {"box": scene["boxes"][k], "forbid": set(names(scene, scene["cheese_colours"]))}, held)
        if kind == "region":
            r = str(rng.choice(list(REGIONS)))
            text = say(rng, "place.region", held, r=r)
            return Task("place", text, {"region": r, "forbid": set(names(scene, scene["cheese_colours"]))}, held)
        d = str(rng.choice(list(DIRECTIONS)))
        text = say(rng, "place.move", held, d=d)
        return Task("place", text, {"direction": d, "forbid": set(names(scene, scene["cheese_colours"]))}, held)

    return {"cheeses": (0, 3), "boxes": (1, 3) if kind == "box" else (0, 3)}, make


SUBTASK_WORDS = {"region": "go to the {x}", "box": "go into the {x} box", "eat": "eat the {x} cheese", "surround": "surround the {x} cheese"}


def parallel(rng, held):
    if held == "pairs":
        pair = tuple(HELD_OUT["pairs"][int(rng.integers(len(HELD_OUT["pairs"])))])
    else:
        while True:
            pair = tuple(sorted(rng.choice(list(SUBTASK_WORDS), size=2, replace=True)))
            if pair not in HELD_OUT["pairs"]:
                break

    def make(scene):
        cheeses = names(scene, scene["cheese_colours"])
        box_names = names(scene, scene["box_colours"])
        jobs, used = [], set()
        for kind in pair:
            if kind == "region":
                options = [r for r in REGIONS if r not in used]
            elif kind == "box":
                options = [b for b in box_names if ("box", b) not in used]
            else:
                options = [c for c in cheeses if ("cheese", c) not in used]
            if not options:
                return None
            x = str(rng.choice(options))
            used.add(x if kind == "region" else ("box" if kind == "box" else "cheese", x))
            jobs.append((kind, x))
        a, b = (SUBTASK_WORDS[k].format(x=x) for k, x in jobs)
        text = say(rng, "parallel.split", held, a=a, b=b)
        eat = {x for k, x in jobs if k == "eat"}
        rects = dict(zip(box_names, scene["boxes"]))
        spec = {"jobs": jobs, "rects": rects, "forbid": set(cheeses) - eat}
        return Task("parallel", text, spec, held)

    needs_box = "box" in pair
    n_cheese = sum(k in ("eat", "surround") for k in pair)
    return {"cheeses": (max(n_cheese, 0), 4), "boxes": (2 if pair == ("box", "box") else 1, 3) if needs_box else (0, 2)}, make


def herd(rng, held):
    two = held == "herd_parallel"
    kind = "box" if two else str(rng.choice(["box", "nudge"]))

    def make(scene):
        cheeses = names(scene, scene["cheese_colours"])
        side = cheese_side(SIZE)
        if kind == "nudge":
            if not cheeses:
                return None
            c = str(rng.choice(cheeses))
            d = str(rng.choice(["up", "down", "left", "right"]))
            text = say(rng, "herd.nudge", held, c=c, d=d)
            return Task("herd", text, {"nudge": (c, d), "forbid": set(cheeses)}, held)
        if not scene["boxes"] or len(cheeses) < (2 if two else 1):
            return None
        k = int(rng.integers(len(scene["boxes"])))
        rect = scene["boxes"][k]
        free = [c for (pos, ci), c in zip(zip(scene["cheeses"], scene["cheese_colours"]), cheeses) if not cheese_in_box(pos, rect, side)]
        if len(free) < (2 if two else 1):
            return None
        targets = [str(t) for t in rng.choice(free, size=2 if two else 1, replace=False)]
        b = COLOUR_NAMES[scene["box_colours"][k]]
        if two:
            text = say(rng, "herd.two", held, x=targets[0], y=targets[1], b=b)
        else:
            text = say(rng, "herd.into", held, x=targets[0], b=b)
        return Task("herd", text, {"into": (targets, rect), "forbid": set(cheeses)}, held)

    return {"cheeses": (2 if two else 1, 3), "boxes": (1, 2) if kind == "box" else (0, 1)}, make


def runner(rng, held):
    def make(scene):
        cheeses = names(scene, scene["cheese_colours"])
        runs = [cheeses[k] for k in scene["runners"]]
        if not runs:
            return None
        if held == "runner_logic":
            if len(cheeses) < 2:
                return None
            a = str(rng.choice(cheeses))
            text = say(rng, "runner.except", held, a=a)
            return Task("runner", text, {"eat": set(cheeses) - {a}, "any_of": set(), "order": [], "forbid": {a}, "form": "except"}, held)
        c = str(rng.choice(runs))
        text = say(rng, "runner.catch", held, c=c)
        return Task("runner", text, {"eat": {c}, "any_of": set(), "order": [], "forbid": set(cheeses) - {c}, "form": "single"}, held)

    return {"cheeses": (2 if held == "runner_logic" else 1, 3), "boxes": (0, 1), "runners": 2 if held == "runner_logic" else int(rng.integers(1, 3))}, make


MAKERS = {"formation": formation, "logic": logic, "place": place, "parallel": parallel, "herd": herd, "runner": runner}
HELD_FAMILY = {"colour": "logic", "letters": "formation", "words": "formation", "shapes": "formation", "logic": "logic", "pairs": "parallel", "herd_parallel": "herd", "runner_logic": "runner"}


def sample_task(rng, split="train", family=None):
    """(scene, Task). split: "train", "test" (in distribution) or a HELD_OUT key."""
    held = split if split in HELD_OUT else ""
    family = HELD_FAMILY.get(held) or family or str(rng.choice(FAMILIES))
    for _ in range(100):
        needs, make = MAKERS[family](rng, held)
        colours = COLOUR_NAMES if held == "colour" else TRAIN_COLOURS
        require = ("pink",) if held == "colour" else ()
        scene = sample_scene(rng, colours=colours, require=require, **needs)
        if scene is None:
            continue
        task = make(scene)
        if task is None:
            continue
        if held == "colour":  # an instruction about the held-out colour
            task = Task("logic", say(rng, "logic.single", held, a="pink"), {"eat": {"pink"}, "any_of": set(), "order": [], "forbid": set(names(scene, scene["cheese_colours"])) - {"pink"}, "form": "single"}, held)
        return scene, task
    raise RuntimeError(f"could not make a {family} task for split {split}")


# ---------------------------------------------------------------------------------------------
# Checking


def groups(mass, min_share=0.15):
    """Connected groups (8-neighbourhood) of occupied squares, as (share of robots, cells mask),
    largest first, those under min_share dropped."""
    occupied = mass >= 0.5
    seen = np.zeros_like(occupied)
    out = []
    total = max(occupied.sum(), 1)
    for start in zip(*np.nonzero(occupied)):
        if seen[start]:
            continue
        stack, cells = [start], []
        seen[start] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < SIZE and 0 <= nx < SIZE and occupied[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        mask = np.zeros_like(occupied)
        mask[tuple(np.array(cells).T)] = True
        out.append((len(cells) / total, mask))
    out.sort(key=lambda g: -g[0])
    return [g for g in out if g[0] >= min_share]


def region_score(mask, region):
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return 0.0
    ry, rx = REGIONS[region]
    return float(max(0.0, 1 - np.hypot(rows.mean() - ry, cols.mean() - rx) / (SIZE / 2)))


def box_score(mask, rect):
    y, x, h, w = rect
    inside = mask[y : y + h, x : x + w].sum() / max(mask.sum(), 1)
    return float(min(1.0, inside / BOX_IN))


def score(task, scene, frames, eaten, found=None):
    """(score in [0, 1], unasked: a cheese the task did not ask for was eaten). frames: snapshots
    (goal_eval.snapshot) from the start to the end; eaten: colour names in the order eaten; found:
    groups(end mass) if already computed (scoring many tasks on one episode)."""
    spec, end = task.spec, frames[-1]
    mass = end["mass"]
    occupied = mass >= 0.5
    unasked = any(c in spec.get("forbid", set()) for c in eaten)
    side = cheese_side(SIZE)
    colour_slot = {COLOUR_NAMES[c]: k for k, c in enumerate(scene["cheese_colours"])}
    if task.family == "formation":
        return formation_score(mass, spec["mask"]), unasked
    if task.family in ("logic", "runner"):
        got = set(eaten)
        if spec["order"]:
            targets = [c for c in eaten if c in spec["eat"]]
            prefix = 0
            for want, have in zip(spec["order"], targets):
                if want != have:
                    break
                prefix += 1
            return prefix / len(spec["order"]), unasked
        if spec["any_of"]:
            return float(bool(got & spec["any_of"])), unasked
        if spec["eat"]:
            return len(got & spec["eat"]) / len(spec["eat"]), unasked
        return 0.0, unasked
    if task.family == "place":
        if "box" in spec:
            return box_score(occupied, spec["box"]), unasked
        if "region" in spec:
            return region_score(occupied, spec["region"]), unasked
        start = frames[0]["mass"] >= 0.5
        r0, c0 = np.nonzero(start)
        r1, c1 = np.nonzero(occupied)
        if not len(r1):
            return 0.0, unasked
        dy, dx = DIRECTIONS[spec["direction"]]
        along = (r1.mean() - r0.mean()) * dy + (c1.mean() - c0.mean()) * dx
        return float(np.clip(along / 6, 0, 1)), unasked
    if task.family == "parallel":
        found = groups(mass) if found is None else found
        local = [(k, x) for k, x in spec["jobs"] if k in ("region", "box")]
        scores = []
        for k, x in spec["jobs"]:
            if k == "eat":
                scores.append(float(x in eaten))
            elif k == "surround":
                slot = colour_slot[x]
                alive = end["cheese_alive"][slot]
                share = ring_share(mass, end["cheese_pos"][slot], scene["walls"]) if alive else 0.0
                scores.append(float(min(1.0, share / AROUND_SHARE)))
        def local_score(job, mask):
            k, x = job
            return region_score(mask, x) if k == "region" else box_score(mask, spec["rects"][x])
        if len(local) == 2:
            best = 0.0
            for i in range(len(found)):
                for j in range(len(found)):
                    if i != j:
                        best = max(best, (local_score(local[0], found[i][1]) + local_score(local[1], found[j][1])) / 2)
            if len(found) < 2:  # one group cannot be in two places: half credit at best
                best = 0.5 * (local_score(local[0], occupied) + local_score(local[1], occupied)) / 2
            scores += [best, best]
        elif len(local) == 1:
            scores.append(max([local_score(local[0], g[1]) for g in found] or [0.0]))
        split = 1.0 if len(found) >= 2 else 0.5
        return split * float(np.mean(scores)), unasked
    if task.family == "herd":
        if "nudge" in spec:
            c, d = spec["nudge"]
            slot = colour_slot[c]
            if not end["cheese_alive"][slot]:
                return 0.0, True
            dy, dx = DIRECTIONS[d]
            moved = (end["cheese_pos"][slot] - frames[0]["cheese_pos"][slot]) @ np.array([dy, dx])
            return float(np.clip(moved / 4, 0, 1)), unasked
        targets, rect = spec["into"]
        by, bx, bh, bw = rect
        centre = np.array([by + bh / 2, bx + bw / 2])
        parts = []
        for c in targets:
            slot = colour_slot[c]
            if not end["cheese_alive"][slot]:
                parts.append(0.0)
                unasked = True
                continue
            if cheese_in_box(end["cheese_pos"][slot], rect, side):
                parts.append(1.0)
                continue
            d0 = np.linalg.norm(frames[0]["cheese_pos"][slot] + side / 2 - centre)
            d1 = np.linalg.norm(end["cheese_pos"][slot] + side / 2 - centre)
            parts.append(float(0.8 * np.clip((d0 - d1) / max(d0, 1e-6), 0, 1)))
        return float(np.mean(parts)), unasked
    raise ValueError(task.family)


# ---------------------------------------------------------------------------------------------
# Episodes


@torch.no_grad()
def run(sample, condition, scenes, tasks, steps, device, keep_steps=(), outcomes=False):
    """Roll out a batch: sample(obs, cond) -> per-square moves (B, ...) and the field to apply,
    as (moves, field); condition(texts) -> cond. Returns per episode (score, unasked), and for the
    steps in keep_steps the (obs, moves) pairs (CPU); with outcomes, also per episode (frames,
    eaten) for scoring other instructions on the same episode (relabel)."""
    world, runners = stack(scenes)
    world, runners = world.to(device), runners.to(device)
    cond = condition([t.text for t in tasks])
    frames = [[goal_eval.snapshot(world, i)] for i in range(len(tasks))]
    eaten = [[] for _ in tasks]
    alive_before = world.cheese_alive.cpu()
    kept = []
    for t in range(steps):
        obs = observe(world)
        moves, field = sample(obs, cond)
        if t in keep_steps:
            kept.append((obs.cpu(), moves.cpu()))
        world, _ = step(world, field)
        if runners.any() and (t + 1) % RUNNER_PERIOD == 0:
            world = World(mass=world.mass, walls=world.walls, cheese_pos=flee(world, runners), cheese_alive=world.cheese_alive, cheese_color=world.cheese_color, box_rect=world.box_rect, box_color=world.box_color)
        alive_now = world.cheese_alive.cpu()
        for i, slot in (alive_before & ~alive_now).nonzero().tolist():
            eaten[i].append(COLOUR_NAMES[int(world.cheese_color[i, slot])])
        alive_before = alive_now
        if (t + 1) % goal_eval.FACT_EVERY == 0 or t == steps - 1:
            for i in range(len(tasks)):
                frames[i].append(goal_eval.snapshot(world, i))
    results = [score(task, scene, f, e) for task, scene, f, e in zip(tasks, scenes, frames, eaten)]
    if outcomes:
        return results, kept, list(zip(frames, eaten))
    return results, kept
