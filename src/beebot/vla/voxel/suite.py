"""The voxel generalist suite: instruction families at parity with the 2D suite (vla/tasks.py),
their checkers, held-out splits, an expert that demonstrates every training task through the
target controller, and an episode runner with fleeing ("runner") cheeses.

Families (a scene draws an instruction from what it holds; score in [0, 1] at the episode's end):
  formation     3D shapes (cube, sphere, ring, plus, wall; pyramid held out) and letters lying on
                the floor or standing up (K, Q, Z held out): F1 of occupied voxels against the mask.
  construction  a tower of height N, a bridge across the trench, filling the trench; stairs up to
                the plateau held out.
  logic         eat by colour: single, except, and, or, all, none, absent, ordered; neither and
                all-but-two held out.
  parallel      two groups, two jobs (region, tower, surround, eat); the pair eat + tower held out.
  herd          push a cheese into a floor region or into the trench without eating it; pushing
                two at once held out.
  runner        catch a cheese that flees; logic over runners held out.
Colours: the 2D extended palette; pink is held out. Wording: each intent's last template is held
out (and, for eat logic, the 2D paraphrase pool's held-out fifth).

Every task has a spec (what the checker needs, with "forbid": colours it must not eat) and, on
the training split, a plan: segments the expert runs (target voxel sets, eating, pushing,
surrounding), each until done or out of time. Held-out tasks have no plan: nothing demonstrates
them.

Coverage families (COVERAGE; drawn by sample_coverage, never by sample_tasks, so the draws of
FAMILIES are unchanged): who x what x when, README "Coverage".
  motion  everyone does one job: stay, go to a region, go next to a cheese, gather, spread out,
          split into groups, line up, climb onto a plateau, move a few steps, follow a runner.
  subset  a selector picks robots (the smaller or larger group, the group on the left, the group
          nearest a cheese, everyone except one of those, N of you, half of you, the robots on
          top) and they do a job; everyone else stays where they are.
  switch  the instruction changes mid-episode: task A from step 0, task B (often "stay") from
          step k (Task.parts, spec["steps"]).
canonical_record(task) and scene_summary(world, i) give every task and scene in plain JSON.
"""

import itertools
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from beebot.vla import tasks as T2
from beebot.vla.shapes import FONT, text_mask
from beebot.vla.voxel.controller import act
from beebot.vla.voxel.facts import BLOB_SHARE, bridges, groups, shell_share, solid_grid, tallest_run
from beebot.vla.voxel.world import CHEESE_SIDE, COLOUR_NAMES, DIRS, EAT_MIN_SHARE, EAT_SHARE, FRICTION, FULL_SHELL, MAX_CHEESES, SHELL_OFFSETS, STAY, VoxelWorld, _face, cheese_id_map, fail_moves, refuse, rule_moves, sample_scenes, shell, step

SIZE = 16
HELD_OUT = {
    "colour": ("pink",),
    "letters": tuple("KQZ"),
    "shapes": ("pyramid",),
    "logic": ("neither", "all_but_two"),
    "pairs": (("eat", "tower"),),
    "construction": ("stairs",),
    "herd_two": (),
    "runner_logic": (),
    "wording": (),
}
TRAIN_COLOURS = tuple(c for c in COLOUR_NAMES if c not in HELD_OUT["colour"])
FAMILIES = ("formation", "construction", "logic", "parallel", "herd", "runner")
HELD_FAMILY = {"colour": "logic", "letters": "formation", "shapes": "formation", "logic": "logic", "pairs": "parallel", "construction": "construction", "herd_two": "herd", "runner_logic": "runner"}
SHAPES = ("cube", "sphere", "ring", "plus", "wall", "pyramid")
SHAPE_WORDS = {"cube": "hollow cube", "sphere": "hollow sphere", "ring": "flat ring", "plus": "3D plus sign", "wall": "wall", "pyramid": "pyramid"}
FLOOR_REGIONS = {f"{a} {b}".strip() or "center": (i, j) for i, a in enumerate(("back", "", "front")) for j, b in enumerate(("left", "", "right"))}
FLOOR_REGIONS = {(k if k else "center"): v for k, v in FLOOR_REGIONS.items()}
PUSHERS = 8  # robots that push a cheese; the rest stay
RUNNER_RADIUS = 5
RUNNER_PERIOD = 2
EPISODE_STEPS = 48
SEGMENT_STEPS = 24
CUBE_STEPS = 72  # episode length of pivoting-cube demonstrations (world.CUBE_MOVES)
ROBOTS = (6, int(0.45 * SIZE * SIZE))  # the range of swarm sizes when scenes draw a robot count: a handful up to the densest sampled scene
COUNTS = ("uniform", "log")  # how a robot count is drawn from a range (draw_count)
SUCCESS = {"formation": 0.5}  # checker score an episode needs to count as done (a demonstration), per family
PASS_MARK = 0.75  # and for the families not named there
WORDINGS = {
    "letter_floor": ["write the letter {w} on the floor", "lie down in the shape of {w}", "spell {w} flat on the ground", "draw a big {w} on the floor", "form a letter {w} lying flat"],
    "letter_stand": ["stand up as the letter {w}", "form a standing letter {w}", "make a {w} standing upright", "build the letter {w} in the air"],
    "shape": ["form a {s}", "make a {s}", "arrange yourselves into a {s}", "build a {s} out of robots", "become a {s}"],
    "tower": ["build a tower {n} robots tall", "stack up {n} high", "make a column {n} robots high", "pile up into a tower of height {n}", "build a {n}-high tower"],
    "bridge": ["bridge the gap", "build a bridge across the trench", "span the gap with your bodies", "make a bridge over the trench", "connect the two plateaus with a bridge"],
    "fill": ["fill the trench", "climb down into the trench and fill it", "pack yourselves into the gap", "fill up the gap between the plateaus"],
    "stairs": ["build stairs up to the plateau", "make a staircase up the cliff"],
    "region": ["go to the {r} of the floor", "gather at the {r}", "head for the {r} area", "everyone to the {r}", "move over to the {r}"],
    "surround": ["surround the {c} cheese", "gather around the {c} cheese", "encircle the {c} cheese", "hug the {c} cheese without eating it"],
    "eat": ["eat the {c} cheese", "go eat the {c} one", "gobble the {c} cheese", "devour the {c} cheese"],
    "split": ["split up: {a}, and {b}", "two teams: one should {a}, the other {b}", "half of you {a}, the other half {b}", "divide into two groups; one group {a}, the other {b}", "in two groups at once: {a}; {b}"],
    "herd_region": ["push the {c} cheese to the {r}", "shove the {c} cheese over to the {r} without eating it", "move the {c} cheese to the {r} of the floor", "herd the {c} cheese to the {r}", "slide the {c} cheese into the {r} area"],
    "herd_trench": ["push the {c} cheese into the trench", "shove the {c} cheese down into the gap", "knock the {c} cheese into the trench without eating it", "drop the {c} cheese into the trench"],
    "herd_two": ["push the {c} and {d} cheeses to the {r}", "move both the {c} and {d} cheeses to the {r}"],
    "runner": ["catch the {c} cheese", "chase down the {c} cheese and eat it", "hunt the {c} cheese", "catch {c} before it gets away", "corner the {c} cheese and eat it"],
    "runner_except": ["catch every cheese except the {c} one", "eat all of them but {c}"],
}
SUBTASK = {"region": "go to the {x}", "tower": "build a tower {x} high", "surround": "surround the {x} cheese", "eat": "eat the {x} cheese"}
LOGIC_KEYS = {"single": "logic.single", "except": "logic.except", "or": "logic.or", "and": "logic.and", "all": "logic.all", "none": "logic.none", "ordered": "logic.ordered", "absent": "logic.single", "neither": "logic.neither", "all_but_two": "logic.all_but_two"}


@dataclass
class Task:
    family: str
    text: str
    spec: dict
    plan: list = field(default_factory=list)  # expert segments; empty for held-out tasks
    held_out: str = ""
    runners: tuple = ()  # cheese slots that flee
    parts: list = field(default_factory=list)  # a switch task: the instructions in force from spec["steps"][j] on


def say(rng, key, held, **kw):
    """A template of intent key: training ones (all but the last), or the last when held ==
    "wording"; eat-logic keys use the 2D pool (vla/tasks.wordings)."""
    if key.startswith("logic."):
        return T2.say(rng, key, "wording" if held == "wording" else "", **kw)
    pool = WORDINGS[key] if key in WORDINGS else COVERAGE_WORDINGS[key]
    choices = pool[-1:] if held == "wording" else pool[:-1]
    return str(rng.choice(choices)).format(**kw)


# ---------------------------------------------------------------------------------------------
# Masks


def shape_mask(name, size=SIZE, floor=0):
    """(D, H, W) bool target of a named shape, centred on the floor plan; shapes that rest on
    the ground (wall, pyramid) start at z = floor, the others float a voxel above it."""
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]
    c = (size - 1) / 2
    lift = floor + 1
    if name == "cube":
        lo, hi = int(c) - 1, int(c) + 3  # side 4
        inside = (zz >= lift) & (zz < lift + 4) & (yy >= lo) & (yy < hi) & (xx >= lo) & (xx < hi)
        core = (zz >= lift + 1) & (zz < lift + 3) & (yy >= lo + 1) & (yy < hi - 1) & (xx >= lo + 1) & (xx < hi - 1)
        return inside & ~core
    if name == "sphere":
        cz = lift + 3
        d = np.sqrt((zz - cz) ** 2 + (yy - c) ** 2 + (xx - c) ** 2)
        return np.abs(d - 2.7) < 0.55
    if name == "ring":
        d = np.sqrt((yy - c) ** 2 + (xx - c) ** 2)
        return (zz == lift) & (np.abs(d - 4.5) < 0.55)
    if name == "plus":
        cz = lift + 4
        bar = lambda a, b: (np.abs(a) < 0.6) & (np.abs(b) < 0.6)  # noqa: E731
        return ((bar(zz - cz, yy - c) & (np.abs(xx - c) <= 4)) | (bar(zz - cz, xx - c) & (np.abs(yy - c) <= 4)) | (bar(yy - c, xx - c) & (np.abs(zz - cz) <= 4))) & (zz >= lift)
    if name == "wall":
        return (yy == int(c)) & (np.abs(xx - c) <= 4) & (zz >= floor) & (zz < floor + 5)
    if name == "pyramid":
        out = np.zeros((size, size, size), bool)
        for k, half in enumerate((3, 2, 1, 0)):
            z = floor + k
            ring = (np.abs(yy - c).round() <= half) & (np.abs(xx - c).round() <= half)
            inner = (np.abs(yy - c).round() <= half - 1) & (np.abs(xx - c).round() <= half - 1)
            out |= (zz == z) & ring & ~inner
        return out
    raise ValueError(name)


def letter_mask(text, standing, size=SIZE, floor=0):
    """Letters (5 x 7 font) lying on the floor (rows along y) or standing in the x-z plane."""
    scale = 2 if len(text) == 1 else 1
    flat = text_mask(text, size, scale=scale)
    out = np.zeros((size, size, size), bool)
    if not standing:
        out[floor] = flat
        return out
    rows = np.nonzero(flat.any(axis=1))[0]
    glyph = flat[rows.min() : rows.max() + 1]  # (h, size): top row first
    h = glyph.shape[0]
    y = size // 2
    for r in range(h):
        z = floor + 1 + (h - 1 - r)
        if z < size:
            out[z, y] = glyph[r]
    return out


def f1(mass, mask):
    occupied = mass > 0.5
    hit = (occupied & mask).sum()
    if hit == 0:
        return 0.0
    precision, recall = hit / occupied.sum(), hit / mask.sum()
    return float(2 * precision * recall / (precision + recall))


# ---------------------------------------------------------------------------------------------
# Scenes


def scenes(rng, n, layout=None, gravity=None, cheeses=(0, 3), colours=TRAIN_COLOURS, require=(), robots=None, groups=None, layers=1):
    """(world, meta) of n scenes drawn until they match: layout ("flat", "hills", "gap" or None),
    gravity (bool or None), a cheese count in [low, high] (extra cheeses removed), colours drawn
    from `colours` with `require` among them. robots: the robot count of every scene (None: the
    world's density range). groups: (low, high), the swarm set down again as that many separate
    groups (place_groups) with `layers` layers at most; None leaves it as drawn."""
    picked, metas = [], []
    while len(picked) < n:
        world, meta = sample_scenes(rng, 32, SIZE, gravity_prob=0.5 if gravity is None else float(gravity), robots=robots)
        for i in range(32):
            if len(picked) == n:
                break
            if layout is not None and meta[i]["layout"] != layout:
                continue
            alive = world.cheese_alive[i].nonzero().squeeze(1).tolist()
            want = max(int(rng.integers(cheeses[0], cheeses[1] + 1)), len(require))
            if len(alive) < max(want, len(require)):
                continue
            keep = alive[:want]
            w = world.index(i)
            w.cheese_alive[0] = False
            w.cheese_alive[0, keep] = True
            pool = [COLOUR_NAMES.index(c) for c in colours if c not in require]
            chosen = [COLOUR_NAMES.index(c) for c in require] + [int(c) for c in rng.choice(pool, size=len(keep) - len(require), replace=False)]
            rng.shuffle(chosen)
            w.cheese_color[0] = -1
            for k, c in zip(keep, chosen):
                w.cheese_color[0, k] = c
            if groups is not None:
                count = int(rng.integers(groups[0], groups[1] + 1))
                mass = place_groups(rng, w, count, int(w.mass.sum()), layers)
                if mass is None:
                    continue
                w = replace(w, mass=mass)
            picked.append(w)
            metas.append(meta[i])
    fields = ("mass", "walls", "cheese_pos", "cheese_alive", "cheese_color", "gravity")
    return VoxelWorld(*(torch.cat([getattr(w, f) for w in picked]) for f in fields)), metas


def present(world, i):
    """[(slot, colour name)] of the scene's alive cheeses."""
    return [(k, COLOUR_NAMES[int(world.cheese_color[i, k])]) for k in range(MAX_CHEESES) if world.cheese_alive[i, k] and world.cheese_color[i, k] >= 0]


def region_box(name, size=SIZE):
    i, j = FLOOR_REGIONS[name]
    third = size / 3
    return (int(i * third), int((i + 1) * third), int(j * third), int((j + 1) * third))


def ground_point(meta, y, x, size=SIZE):
    return (min(int(meta["height"][y, x]), size - 1), y, x)


def region_targets(meta, name, n=9):
    y0, y1, x0, x1 = region_box(name)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    pts = [ground_point(meta, y, x) for y in range(cy - 1, cy + 2) for x in range(cx - 1, cx + 2)]
    return pts[:n]


def tower_label(n, points):
    """The sub-goal record of a tower segment: its height and the lowest corner of its column."""
    return {"kind": "tower", "height": n, "at": list(points[0])}


def tower_targets(meta, n, rng, size=SIZE):
    height = meta["height"]
    for _ in range(200):
        y, x = int(rng.integers(1, size - 2)), int(rng.integers(1, size - 2))
        foot = height[y : y + 2, x : x + 2]
        z0 = int(foot[0, 0])
        if (foot == z0).all() and z0 + n <= size:
            return [(z, y + a, x + c) for z in range(z0, z0 + n) for a in range(2) for c in range(2)]
    return None


def trench(meta, size=SIZE):
    """(axis 1 or 2 the trench runs across, start, width, top) or None."""
    if meta["gap"] is None:
        return None
    transposed, start, width, top = meta["gap"]
    return (1 if transposed else 2, start, width, top)


def trench_mask(meta, size=SIZE):
    axis, start, width, top = trench(meta)
    out = np.zeros((size, size, size), bool)
    if axis == 2:
        out[:top, :, start : start + width] = True
    else:
        out[:top, start : start + width, :] = True
    return out


def stairs_masks(meta, size=SIZE):
    """Every one-wide staircase against a plateau wall: step j (0 = next to the wall) holds the
    voxels below top - 1 - j, for each lane and both walls."""
    axis, start, width, top = trench(meta)
    out = []
    for lane in range(size):
        for side in (-1, 1):
            m = np.zeros((size, size, size), bool)
            for j in range(min(top - 1, width)):
                pos = start + j if side < 0 else start + width - 1 - j
                h = top - 1 - j
                if axis == 2:
                    m[:h, lane, pos] = True
                else:
                    m[:h, pos, lane] = True
            if m.any():
                out.append(m)
    return out


# ---------------------------------------------------------------------------------------------
# Families: (rng, world, meta, i, held) -> Task or None


def logic_spec(form, colours, rng):
    """(spec, text args, key) of an eat-logic form over the scene's colours, or None."""
    p = list(colours)
    rng.shuffle(p)
    every = set(p)
    need = 3 if form in ("neither", "all_but_two") else 2 if form in ("except", "or", "and", "ordered") else 1 if form in ("single", "all", "none") else 0
    if len(p) < need:
        return None
    spec = {"kind": "eat", "eat": set(), "any_of": set(), "order": [], "forbid": set()}
    kw = {}
    if form == "single":
        spec.update(eat={p[0]}, forbid=every - {p[0]})
        kw = {"a": p[0]}
    elif form == "except":
        spec.update(eat=every - {p[0]}, forbid={p[0]})
        kw = {"a": p[0]}
    elif form == "or":
        spec.update(any_of={p[0], p[1]}, forbid=every - {p[0], p[1]})
        kw = {"a": p[0], "b": p[1]}
    elif form == "and":
        spec.update(eat={p[0], p[1]}, forbid=every - {p[0], p[1]})
        kw = {"a": p[0], "b": p[1]}
    elif form == "ordered":
        spec.update(eat={p[0], p[1]}, order=[p[0], p[1]], forbid=every - {p[0], p[1]})
        kw = {"a": p[0], "b": p[1]}
    elif form == "all":
        spec.update(eat=every)
    elif form == "none":
        spec.update(forbid=every)
    elif form == "absent":
        missing = [c for c in TRAIN_COLOURS if c not in every]
        spec.update(forbid=every)
        kw = {"a": str(rng.choice(missing))}
    else:  # neither, all_but_two
        spec.update(eat=every - {p[0], p[1]}, forbid={p[0], p[1]})
        kw = {"a": p[0], "b": p[1]}
    spec["form"] = form
    return spec, kw


def eat_plan(world, i, colours_in_order):
    slot = {c: k for k, c in present(world, i)}
    return [{"kind": "eat", "cheese": slot[c]} for c in colours_in_order]


def make_formation(rng, held):
    def make(world, meta, i):
        if held in ("letters",) or (held in ("", "wording") and rng.random() < 0.5):
            allowed = [ch for ch in FONT if ch not in HELD_OUT["letters"]]
            if held == "letters":
                word = str(rng.choice(HELD_OUT["letters"]))
            elif rng.random() < 0.25:
                word = "".join(str(rng.choice(allowed)) for _ in range(2))
            else:
                word = str(rng.choice(allowed))
            standing = not bool(world.gravity[i]) and len(word) == 1 and rng.random() < 0.5
            mask = letter_mask(word, standing)
            label = {"kind": "letters", "word": word, "standing": standing}
            text = say(rng, "letter_stand" if standing else "letter_floor", held, w=word)
        else:
            pool = HELD_OUT["shapes"] if held == "shapes" else [s for s in SHAPES if s not in HELD_OUT["shapes"] and (s == "wall" or not bool(world.gravity[i]))]  # floating shapes need gravity off
            name = str(rng.choice(pool))
            mask = shape_mask(name)
            label = {"kind": "shape", "shape": name}
            text = say(rng, "shape", held, s=SHAPE_WORDS[name])
        if (mask & world.walls[i].numpy()).any() or not mask.any():
            return None
        plan = [] if held not in ("", "wording") else [{"kind": "targets", "points": [tuple(int(v) for v in p) for p in np.argwhere(mask)], "label": label}]
        return Task("formation", text, {"kind": "mask", "mask": mask, "forbid": set(c for _, c in present(world, i))}, plan, held)

    return {"layout": "flat", "cheeses": (0, 0)}, make


def make_construction(rng, held):
    kind = "stairs" if held == "construction" else str(rng.choice(["tower", "tower", "bridge", "fill"]))

    def make(world, meta, i):
        forbid = set(c for _, c in present(world, i))
        if kind == "tower":
            n = int(rng.integers(3, 9))
            pts = tower_targets(meta, n, rng)
            if pts is None:
                return None
            return Task("construction", say(rng, "tower", held, n=n), {"kind": "tower", "height": n, "forbid": forbid}, [{"kind": "targets", "points": pts, "label": tower_label(n, pts)}] if not held or held == "wording" else [], held)
        if trench(meta) is None:
            return None
        axis, start, width, top = trench(meta)
        if kind == "bridge":
            lane = int(rng.integers(SIZE - 1))
            pts = [(top - 1, lane + a, x) for a in range(2) for x in range(start, start + width)]
            if axis == 1:
                pts = [(z, x, y) for z, y, x in pts]
            return Task("construction", say(rng, "bridge", held), {"kind": "bridge", "width": width, "forbid": forbid}, [{"kind": "targets", "points": pts, "label": {"kind": "bridge"}}] if held in ("", "wording") else [], held)
        if kind == "fill":
            mask = trench_mask(meta)
            pts = sorted((tuple(int(v) for v in p) for p in np.argwhere(mask)), key=lambda p: p[0])
            return Task("construction", say(rng, "fill", held), {"kind": "fill", "mask": mask, "forbid": forbid}, [{"kind": "targets", "points": pts[:96], "label": {"kind": "fill"}}] if held in ("", "wording") else [], held)
        return Task("construction", say(rng, "stairs", held), {"kind": "stairs", "masks": stairs_masks(meta), "forbid": forbid}, [], held)

    needs = {"tower": {"gravity": True, "cheeses": (0, 1)}, "bridge": {"gravity": True, "layout": "gap", "cheeses": (0, 1)}, "fill": {"gravity": True, "layout": "gap", "cheeses": (0, 1)}, "stairs": {"gravity": True, "layout": "gap", "cheeses": (0, 1)}}[kind]
    return needs, make


def make_logic(rng, held):
    forms = HELD_OUT["logic"] if held == "logic" else ["single", "except", "or", "and", "all", "none", "ordered", "absent"]
    form = "single" if held == "colour" else str(rng.choice(forms))

    def make(world, meta, i):
        colours = [c for _, c in present(world, i)]
        if held == "colour":
            if "pink" not in colours:
                return None
            spec = {"kind": "eat", "eat": {"pink"}, "any_of": set(), "order": [], "forbid": set(colours) - {"pink"}, "form": "single"}
            return Task("logic", say(rng, "logic.single", held, a="pink"), spec, [], held)
        made = logic_spec(form, colours, rng)
        if made is None:
            return None
        spec, kw = made
        text = say(rng, LOGIC_KEYS[form], held, **kw)
        plan = []
        if held in ("", "wording"):
            order = spec["order"] or sorted(spec["eat"]) or ([sorted(spec["any_of"])[0]] if spec["any_of"] else [])
            plan = eat_plan(world, i, order)
        return Task("logic", text, spec, plan, held)

    needs = {"cheeses": (3, 3) if form in ("neither", "all_but_two") else (1, 3)}
    if held == "colour":
        needs["require"] = ("pink",)
        needs["colours"] = COLOUR_NAMES
    return needs, make


def make_parallel(rng, held):
    if held == "pairs":
        pair = tuple(HELD_OUT["pairs"][int(rng.integers(len(HELD_OUT["pairs"])))])
    else:
        while True:
            pair = tuple(sorted(rng.choice(list(SUBTASK), size=2, replace=True)))
            if pair not in HELD_OUT["pairs"] and pair != ("tower", "tower"):
                break

    def make(world, meta, i):
        colours = [c for _, c in present(world, i)]
        slot = {c: k for k, c in present(world, i)}
        jobs, plan_parts, used = [], [], set()
        for kind in pair:
            if kind == "region":
                options = [r for r in FLOOR_REGIONS if r not in used]
                x = str(rng.choice(options))
                plan_parts.append({"kind": "targets", "points": region_targets(meta, x), "label": {"kind": "region", "region": x}})
            elif kind == "tower":
                n = int(rng.integers(3, 7))
                pts = tower_targets(meta, n, rng)
                if pts is None:
                    return None
                x = n
                plan_parts.append({"kind": "targets", "points": pts, "label": tower_label(n, pts)})
            else:
                options = [c for c in colours if c not in used]
                if not options:
                    return None
                x = str(rng.choice(options))
                plan_parts.append({"kind": "eat" if kind == "eat" else "surround", "cheese": slot[x]})
            used.add(x)
            jobs.append((str(kind), x))
        words = [SUBTASK[k].format(x=x) for k, x in jobs]
        eat = {x for k, x in jobs if k == "eat"}
        spec = {"kind": "parallel", "jobs": jobs, "forbid": set(colours) - eat}
        plan = [{"kind": "union", "parts": plan_parts}] if held in ("", "wording") else []
        return Task("parallel", say(rng, "split", held, a=words[0], b=words[1]), spec, plan, held)

    n_cheese = sum(k in ("surround", "eat") for k in pair)
    return {"gravity": True if "tower" in pair else None, "cheeses": (n_cheese, max(n_cheese, 2)), "layout": "flat" if "tower" in pair else None}, make


def make_herd(rng, held):
    kind = "two" if held == "herd_two" else str(rng.choice(["region", "region", "trench"]))

    def make(world, meta, i):
        cheeses = present(world, i)
        colours = [c for _, c in cheeses]
        if kind == "trench":
            if trench(meta) is None or not cheeses:
                return None
            k, c = cheeses[int(rng.integers(len(cheeses)))]
            axis, start, width, top = trench(meta)
            pos = int(world.cheese_pos[i, k, axis])
            sign = 1 if pos < start else -1
            direction = [0, 0, 0]
            direction[axis] = sign
            spec = {"kind": "herd_trench", "cheese": c, "forbid": set(colours)}
            return Task("herd", say(rng, "herd_trench", held, c=c), spec, [{"kind": "push", "cheese": k, "direction": tuple(direction), "goal": spec}] if held in ("", "wording") else [], held)
        if len(cheeses) < (2 if kind == "two" else 1):
            return None
        pick = [cheeses[j] for j in rng.choice(len(cheeses), size=2 if kind == "two" else 1, replace=False)]
        here = {name for name, (y0, y1, x0, x1) in ((r, region_box(r)) for r in FLOOR_REGIONS) for k, _ in pick if y0 <= int(world.cheese_pos[i, k, 1]) + 1 < y1 and x0 <= int(world.cheese_pos[i, k, 2]) + 1 < x1}
        options = [r for r in FLOOR_REGIONS if r not in here]
        r = str(rng.choice(options))
        spec = {"kind": "herd_region", "cheeses": [c for _, c in pick], "region": r, "forbid": set(colours)}
        if kind == "two":
            return Task("herd", say(rng, "herd_two", held, c=pick[0][1], d=pick[1][1], r=r), spec, [], held)
        return Task("herd", say(rng, "herd_region", held, c=pick[0][1], r=r), spec, [{"kind": "push", "cheese": pick[0][0], "goal": spec}] if held in ("", "wording") else [], held)

    return ({"layout": "gap", "gravity": True, "cheeses": (1, 2)} if kind == "trench" else {"layout": "flat", "cheeses": (2, 3) if kind == "two" else (1, 3)}), make


def make_runner(rng, held):
    def make(world, meta, i):
        cheeses = present(world, i)
        if not cheeses:
            return None
        if held == "runner_logic":
            if len(cheeses) < 2:
                return None
            k, c = cheeses[int(rng.integers(len(cheeses)))]
            spec = {"kind": "eat", "eat": {x for _, x in cheeses} - {c}, "any_of": set(), "order": [], "forbid": {c}, "form": "except"}
            return Task("runner", say(rng, "runner_except", held, c=c), spec, [], held, tuple(s for s, _ in cheeses))
        runs = [cheeses[j] for j in range(len(cheeses)) if j < 2]
        k, c = runs[int(rng.integers(len(runs)))]
        spec = {"kind": "eat", "eat": {c}, "any_of": set(), "order": [], "forbid": {x for _, x in cheeses} - {c}, "form": "single"}
        return Task("runner", say(rng, "runner", held, c=c), spec, [{"kind": "eat", "cheese": k}] if held in ("", "wording") else [], held, tuple(s for s, _ in runs))

    return {"layout": "flat", "cheeses": (2, 3) if held == "runner_logic" else (1, 3)}, make


MAKERS = {"formation": make_formation, "construction": make_construction, "logic": make_logic, "parallel": make_parallel, "herd": make_herd, "runner": make_runner}


def draw_count(rng, low, high, counts="uniform"):
    """A robot count in [low, high]: "uniform", or "log" for a log-uniform draw (count k with
    probability proportional to log((k + 1) / k)), which gives every doubling of the swarm the
    same share: of 6 to 115, 31% under 15, 34% from 15 to 40 and 35% above, against 8%, 24% and
    68% for uniform."""
    if counts == "log":
        count = min(int(np.exp(rng.uniform(np.log(low), np.log(high + 1)))), high)
    else:
        count = int(rng.integers(low, high + 1))
    return count


def sample_tasks(rng, n, split="train", family=None, robots=None, check=None, counts="uniform"):
    """(world, metas, tasks) of n scenes with one task each. split: "train", "test" (in
    distribution) or a HELD_OUT key (forces that held-out item). robots: (low, high), each
    scene's robot count drawn from that range (counts: "uniform" or "log", draw_count); None:
    the world's density range. check: whether the task must be one the scene's swarm is large
    enough for (fits), which also leaves out a cheese that can never be eaten and a formation
    with over three robots per mask voxel; None: only when robots is given, so the default draws
    are as they always were. Families are redrawn for the same count until one fits, so the
    counts keep their distribution."""
    held = split if split in HELD_OUT else ""
    check = robots is not None if check is None else check
    worlds, metas, out = [], [], []
    count = None
    while len(out) < n:
        if robots is not None and count is None:
            count = draw_count(rng, robots[0], robots[1], counts)
        fam = HELD_FAMILY.get(held) or family or str(rng.choice(FAMILIES))
        needs, make = MAKERS[fam](rng, held)
        colours = needs.pop("colours", TRAIN_COLOURS)
        world, meta = scenes(rng, 1, colours=colours, robots=count, **needs)
        task = make(world, meta[0], 0)
        if task is None or (check and not fits(task, world, 0)):
            continue
        count = None
        worlds.append(world)
        metas.append(meta[0])
        out.append(task)
    fields = ("mass", "walls", "cheese_pos", "cheese_alive", "cheese_color", "gravity")
    return VoxelWorld(*(torch.cat([getattr(w, f) for w in worlds]) for f in fields)), metas, out


# ---------------------------------------------------------------------------------------------
# Checking


def snapshot(world, i):
    return {"mass": world.mass[i].numpy(), "cheese_pos": world.cheese_pos[i].numpy(), "cheese_alive": world.cheese_alive[i].numpy()}


def cheese_centre(snap, k):
    return snap["cheese_pos"][k].astype(float) + CHEESE_SIDE / 2


def score(task, static, start, end, eaten):
    """(score in [0, 1], unasked) of an episode: static (walls, cheese_color, gravity, meta),
    start and end snapshots, eaten the colour names in the order eaten."""
    spec = task.spec
    if "who" in spec:
        return job_score(task, static, start, end, eaten)
    unasked = any(c in spec["forbid"] for c in eaten)
    mass = end["mass"]
    slot = {COLOUR_NAMES[int(c)]: k for k, c in enumerate(static["cheese_color"]) if c >= 0}
    kind = spec["kind"]
    if kind == "mask":
        return f1(mass, spec["mask"]), unasked
    if kind == "tower":
        return min(1.0, tallest_run(mass) / spec["height"]), unasked
    if kind == "bridge":
        solid = solid_grid(static["walls"], end["cheese_pos"], end["cheese_alive"])
        return min(1.0, bridges(mass, solid) / spec["width"]), unasked
    if kind == "fill":
        inside = ((mass > 0.5) & spec["mask"]).sum() / max((mass > 0.5).sum(), 1)
        return float(min(1.0, inside / 0.6)), unasked
    if kind == "stairs":
        occupied = mass > 0.5
        return float(max(((occupied & m).sum() / m.sum()) for m in spec["masks"])), unasked
    if kind == "eat":
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
        return float(not eaten), unasked
    if kind == "herd_region":
        parts = []
        y0, y1, x0, x1 = region_box(spec["region"])
        target = np.array([(y0 + y1) / 2, (x0 + x1) / 2])
        for c in spec["cheeses"]:
            k = slot[c]
            if not end["cheese_alive"][k]:
                parts.append(0.0)
                unasked = True
                continue
            cy, cx = cheese_centre(end, k)[1:]
            if y0 <= cy < y1 and x0 <= cx < x1:
                parts.append(1.0)
                continue
            d0 = np.linalg.norm(cheese_centre(start, k)[1:] - target)
            d1 = np.linalg.norm(np.array([cy, cx]) - target)
            parts.append(float(0.8 * np.clip((d0 - d1) / max(d0, 1e-6), 0, 1)))
        return float(np.mean(parts)), unasked
    if kind == "herd_trench":
        k = slot[spec["cheese"]]
        if not end["cheese_alive"][k]:
            return 0.0, True
        axis, start_, width, top = trench(static["meta"])
        pos = end["cheese_pos"][k]
        inside = start_ <= pos[axis] and pos[axis] + CHEESE_SIDE <= start_ + width and pos[0] < top
        return float(inside), unasked
    if kind == "parallel":
        found = groups(mass)
        scores = []
        for job, x in spec["jobs"]:
            if job == "region":
                y0, y1, x0, x1 = region_box(x)
                scores.append(float(any(y0 <= g[1][1] < y1 and x0 <= g[1][2] < x1 for g in found)))
            elif job == "tower":
                scores.append(min(1.0, tallest_run(mass) / x))
            elif job == "eat":
                scores.append(float(x in eaten))
            else:  # surround
                k = slot[x]
                share = shell_share(mass, end["cheese_pos"][k], static["walls"]) if end["cheese_alive"][k] else 0.0
                scores.append(float(min(1.0, share / 0.5)))
        return float(np.mean(scores)), unasked
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------------
# Swarm size: which tasks a swarm of n robots can do


def task_kind(task):
    """A short name of what a task asks, finer than its family: the spec kind, with the form of
    an eat task ("eat:ordered") and the jobs of a parallel one ("parallel:region+tower")."""
    spec = task.spec
    if task.parts:
        return "switch:" + ">".join(task_kind(part) for part in task.parts)
    if "who" in spec:
        who = spec["who"]
        name = {"group": who.get("which", ""), "count": "count", "half": "half", "top": "top", "all": ""}[who["select"]]
        name = ("not_" if who.get("except") else "") + ("cheese" if who.get("by") == "cheese" else name)
        return f"{spec['kind']}@{name}" if name else spec["kind"]
    detail = spec.get("form", "") if spec["kind"] == "eat" else "+".join(job for job, _ in spec["jobs"]) if spec["kind"] == "parallel" else ""
    return f"{spec['kind']}:{detail}" if detail else spec["kind"]


def eat_need(world, i, k, runner=False):
    """The fewest robots that can eat scene i's cheese k where it stands: EAT_SHARE of its open
    shell and EAT_MIN_SHARE of the full shell (30 on open floor), infinite when its open shell
    is too small for the latter (a cheese in a corner). A runner moves, so it counts as the
    least any cheese takes."""
    least = int(np.ceil(EAT_MIN_SHARE * FULL_SHELL))
    open_ = int(shell(world.index(i))[1][0, k].sum())
    return least if runner else max(int(np.ceil(EAT_SHARE * open_)), least) if open_ >= least else float("inf")


def job_bound(job, x, n, total, world, i, slot):
    """The best score n of a swarm of `total` robots can give one job of a parallel task."""
    if job == "region":  # a group counts from BLOB_SHARE of the swarm
        bound = float(n >= max(1, int(np.ceil(BLOB_SHARE * total))))
    elif job == "tower":
        bound = min(1.0, n / (CHEESE_SIDE * CHEESE_SIDE * x))
    elif job == "eat":
        bound = float(n >= eat_need(world, i, slot[x]))
    else:  # surround: half the open shell is full marks
        open_ = int(shell(world.index(i))[1][0, slot[x]].sum())
        bound = min(1.0, n / max(open_, 1) / 0.5)
    return bound


def score_bound(task, world, i, n=None):
    """An upper bound on the checker score a swarm of n robots (default: scene i's) can reach on
    the task, from the rules and from the plan's target sets:
      mask      F1 of n robots against the mask, all on it or all of it covered.
      tower     the plan fills a 2 x 2 column, so a run of r takes 4 r robots; bridge: two lanes.
      eat       each cheese needs eat_need robots (robots stay after a meal, so the largest need
                decides); the score is the share of the asked cheeses the swarm can eat.
      herd      a cheese slides under a press of FRICTION robots.
      parallel  the best split of the swarm between the two jobs (job_bound).
      fill, stairs and eating nothing do not depend on the count."""
    spec = task.spec
    if task.parts:
        return min(score_bound(part, world, i, n) for part in task.parts)
    if "who" in spec:
        return coverage_bound(task, world, i, n)
    n = int(world.mass[i].sum()) if n is None else n
    slot = {c: k for k, c in present(world, i)}
    kind = spec["kind"]
    bound = 1.0
    if kind == "mask":
        m = int(spec["mask"].sum())
        bound = 2 * min(n, m) / (n + m)
    elif kind == "tower":
        bound = min(1.0, n / (CHEESE_SIDE * CHEESE_SIDE * spec["height"]))
    elif kind == "bridge":
        bound = min(1.0, n / (2 * spec["width"]))
    elif kind == "eat":
        able = {c for c in spec["eat"] | spec["any_of"] if n >= eat_need(world, i, slot[c], slot[c] in task.runners)}
        if spec["any_of"]:
            bound = float(bool(able))
        elif spec["order"]:
            bound = len(list(itertools.takewhile(lambda c: c in able, spec["order"]))) / len(spec["order"])
        elif spec["eat"]:
            bound = len(able) / len(spec["eat"])
    elif kind in ("herd_region", "herd_trench"):
        bound = float(n >= FRICTION)
    elif kind == "parallel":
        (job_a, a), (job_b, b) = spec["jobs"]
        bound = max((job_bound(job_a, a, part, n, world, i, slot) + job_bound(job_b, b, n - part, n, world, i, slot)) / 2 for part in range(n + 1))
    return bound


def fits(task, world, i):
    """Whether scene i's swarm is large enough for the task: score_bound reaches the family's
    pass mark (SUCCESS). Necessary, not sufficient: the expert can still fail a task that fits."""
    return score_bound(task, world, i) >= SUCCESS.get(task.family, PASS_MARK)


# ---------------------------------------------------------------------------------------------
# Episodes: runners, the expert, the runner loop


def flee(world, runners):
    """Each alive runner cheese slides one voxel sideways (y or x) away from the robots within
    RUNNER_RADIUS, if it senses any and the 2 x 2 x 2 block it moves into is clear and (under
    gravity) supported. runners: (B, K) bool. Returns new cheese_pos (B, K, 3)."""
    pos = world.cheese_pos.clone()
    mass = world.mass.numpy()
    walls = world.walls.numpy()
    b, d, h, w = world.mass.shape
    ids = cheese_id_map(world).numpy()
    for i, k in zip(*np.nonzero((runners & world.cheese_alive).numpy())):
        z, y, x = (int(v) for v in pos[i, k])
        cz, cy, cx = z + 1, y + 1, x + 1
        lo = lambda v, n: (max(v - RUNNER_RADIUS, 0), min(v + RUNNER_RADIUS + 1, n))  # noqa: E731
        (z0, z1), (y0, y1), (x0, x1) = lo(cz, d), lo(cy, h), lo(cx, w)
        near = mass[i, z0:z1, y0:y1, x0:x1]
        if near.sum() == 0:
            continue
        zz, yy, xx = np.nonzero(near > 0.5)
        dy, dx = (yy + y0 - cy + 0.5).mean(), (xx + x0 - cx + 0.5).mean()
        options = sorted([(0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], key=lambda m: m[1] * dy + m[2] * dx)
        for _, my, mx in options:
            ny, nx = y + my, x + mx
            if ny < 0 or nx < 0 or ny + CHEESE_SIDE > h or nx + CHEESE_SIDE > w:
                continue
            block = (slice(z, z + CHEESE_SIDE), slice(ny, ny + CHEESE_SIDE), slice(nx, nx + CHEESE_SIDE))
            if walls[i][block].any() or (mass[i][block] > 0.5).any() or ((ids[i][block] >= 0) & (ids[i][block] != k)).any():
                continue
            if bool(world.gravity[i]) and z > 0 and not walls[i, z - 1, ny : ny + CHEESE_SIDE, nx : nx + CHEESE_SIDE].all():
                continue
            pos[i, k] = torch.tensor([z, ny, nx])
            break
    return pos


def subgoal_at(entries, t):
    """The sub-goal entry ({"t", "index", "goal"}) in force at step t of an episode's record
    (Expert.subgoals[i]): the last one that starts at or before t."""
    return [entry for entry in entries if entry["t"] <= t][-1]


class Expert:
    """Runs each scene's plan: segments in order, each until done or SEGMENT_STEPS; the last
    segment holds to the end of the episode.

    moves: (27,) bool, the actions the robots can make (world.CUBE_MOVES for pivoting cubes; None
    for all 27). rule: a pivot rule (world.RULES); the expert then chooses, each step, only among
    the moves the rule lets each robot make from where it is. seed: the expert's own random
    generator for tie-breaks and jitter (None: torch's global one). subgoals[i] records what scene i's expert was after at every step, as entries
    {"t": first step, "index": plan segment, "goal": its content} added whenever either changes:
    the action at step t was chosen under subgoal_at(subgoals[i], t). index is -1 with goal
    {"kind": "stay"} for a task with no plan, and len(plan) with {"kind": "done"} once the last
    eat or push segment is finished and the robots idle."""

    def __init__(self, tasks, moves=None, seed=None, rule=None):
        self.tasks = tasks
        self.rule = rule
        self.index = [0] * len(tasks)
        self.since = [0] * len(tasks)
        self.surround = {}
        self.moves = moves
        self.generator = None if seed is None else torch.Generator().manual_seed(seed)
        self.subgoals = [[] for _ in tasks]
        self.part = [0] * len(tasks)  # the instruction in force of a switch task (Task.parts)
        self.jobs = {}  # scene -> (segment index, JobSetup) of the coverage job in force
        self.frozen = [None] * len(tasks)  # (D, H, W) bool per scene: robots that must stay this step

    def plan_of(self, i, t):
        """Scene i's plan at step t: its task's, or for a switch task that of the instruction in
        force, starting over (segment 0) on the step the instruction changes."""
        task = self.tasks[i]
        if not task.parts:
            return task.plan
        j = instruction_index(task, t)
        if j != self.part[i]:
            self.part[i] = j
            self.index[i] = 0
            self.since[i] = 0
            self.jobs.pop(i, None)
        return task.parts[j].plan

    def note(self, i, t, index, goal):
        """Record scene i's sub-goal at step t if it differs from the one in force."""
        entries = self.subgoals[i]
        if not entries or entries[-1]["index"] != index or entries[-1]["goal"] != goal:
            entries.append({"t": t, "index": index, "goal": goal})

    def push_direction(self, i, seg, world):
        """The unit step a push segment presses along now: its own, or toward the goal region
        along the larger horizontal offset."""
        direction = seg.get("direction")
        if direction is None:
            y0, y1, x0, x1 = region_box(seg["goal"]["region"])
            cy, cx = cheese_centre(snapshot(world, i), seg["cheese"])[1:]
            off = np.array([(y0 + y1) / 2 - cy, (x0 + x1) / 2 - cx])
            axis = 1 + int(np.abs(off).argmax())
            direction = [0, 0, 0]
            direction[axis] = 1 if off[axis - 1] > 0 else -1
        return tuple(int(v) for v in direction)

    def describe(self, i, seg, world):
        """A segment's content for the sub-goal record: its kind and what identifies its target
        (cheese colour, region, tower height, shape or word, push direction as (dz, dy, dx))."""
        kind = seg["kind"]
        if kind == "targets":
            goal = dict(seg.get("label", {"kind": "targets"}))
        elif kind == "job":
            goal = json_safe({k: v for k, v in seg["job"].items() if k != "forbid"})
        elif kind == "union":
            goal = {"kind": "union", "parts": [self.describe(i, part, world) for part in seg["parts"]]}
        else:
            goal = {"kind": kind, "colour": COLOUR_NAMES[int(world.cheese_color[i, seg["cheese"]])]}
            if kind == "push":
                goal["direction"] = list(self.push_direction(i, seg, world))
                goal["to"] = seg["goal"].get("region", "trench")
        return goal

    def done(self, i, seg, world):
        if seg["kind"] == "eat":
            return not bool(world.cheese_alive[i, seg["cheese"]])
        if seg["kind"] == "push":
            k = seg["cheese"]
            if not bool(world.cheese_alive[i, k]):
                return True
            goal = seg["goal"]
            snap = snapshot(world, i)
            if goal["kind"] == "herd_region":
                y0, y1, x0, x1 = region_box(goal["region"])
                cy, cx = cheese_centre(snap, k)[1:]
                return y0 <= cy < y1 and x0 <= cx < x1
            return False
        return False

    def targets_of(self, i, seg, world, shells):
        """(target points, press or None, cap): cap is how many robots (nearest the targets) take
        part, None for all; the others stay, so pushers and surrounders are not mobbed into
        eating the cheese."""
        coords, open_ = shells
        kind = seg["kind"]
        if kind == "targets":
            return seg["points"], None, None
        if kind == "job":
            setup = self.jobs[i][1]
            self.frozen[i] = setup.frozen
            if seg["job"]["kind"] == "eat":
                pts = [p for p in self.targets_of(i, {"kind": "eat", "cheese": setup.slot}, world, shells)[0] if not setup.frozen[p]]
                return pts, None, None
            return setup.points(world, i), None, None
        if kind == "eat":
            k = seg["cheese"]
            pts = [tuple(c) for c, o in zip(coords[i, k].tolist(), open_[i, k].tolist()) if o]
            return pts, None, len(pts) + 4
        if kind == "surround":
            k = seg["cheese"]
            pts = [tuple(c) for c, o in zip(coords[i, k].tolist(), open_[i, k].tolist()) if o]
            key = (i, k)
            if key not in self.surround:
                order = np.random.default_rng(i * 7 + k).permutation(len(pts))
                self.surround[key] = [pts[j] for j in order[: max(1, int(0.6 * len(pts)))]]
            return self.surround[key], None, len(self.surround[key])
        if kind == "push":
            k = seg["cheese"]
            direction = self.push_direction(i, seg, world)
            axis = int(np.abs(np.array(direction)).argmax())
            sign = direction[axis]
            pos = world.cheese_pos[i : i + 1, k]
            back = _face(pos, axis, torch.tensor([-1 if sign > 0 else CHEESE_SIDE]))[0]
            return [tuple(p) for p in back.tolist()], (k, tuple(direction)), PUSHERS
        if kind == "union":
            pts, caps = [], []
            for part in seg["parts"]:
                got = self.targets_of(i, part, world, shells)
                pts += got[0]
                caps.append(len(got[0]) + 6 if got[2] is None else got[2])  # a target set takes its size plus a few
            return pts, None, sum(caps)
        raise ValueError(kind)

    def segments(self, world, t):
        """Advance every scene's plan to step t and note its sub-goal. Returns, per scene, the
        target points, the press ((cheese, direction) or None), the cap on taking part (None: all
        robots) and the segment in force (None when the robots idle)."""
        shells = shell(world)
        lists, presses, caps, segs = [], [], [], []
        for i, task in enumerate(self.tasks):
            plan = self.plan_of(i, t)
            self.frozen[i] = None
            if plan and plan[min(self.index[i], len(plan) - 1)]["kind"] == "job" and self.jobs.get(i, (None,))[0] != self.index[i]:
                self.jobs[i] = (self.index[i], JobSetup(plan[self.index[i]]["job"], world, i))
            if not plan:
                self.note(i, t, -1, {"kind": "stay"})
                lists.append([])
                presses.append(None)
                caps.append(None)
                segs.append(None)
                continue
            seg = plan[self.index[i]]
            while self.index[i] < len(plan) - 1 and (self.done(i, seg, world) or self.since[i] >= SEGMENT_STEPS):
                self.index[i] += 1
                self.since[i] = 0
                seg = plan[self.index[i]]
            self.since[i] += 1
            if self.done(i, seg, world) and seg["kind"] in ("eat", "push"):
                self.note(i, t, len(plan), {"kind": "done"})
                lists.append([])
                presses.append(None)
                caps.append(None)
                segs.append(None)
                continue
            self.note(i, t, self.index[i], self.describe(i, seg, world))
            pts, press, cap = self.targets_of(i, seg, world, shells)
            lists.append(pts)
            presses.append(press)
            caps.append(cap)
            segs.append(seg)
        return lists, presses, caps, segs

    def tensors(self, lists, presses):
        """The controller's inputs from per-scene target lists and presses: targets (B, S, 3),
        valid (B, S), press_cheese (B,), press_dir (B, 3)."""
        b = len(self.tasks)
        s = max(1, max(len(p) for p in lists))
        targets = torch.zeros(b, s, 3, dtype=torch.long)
        valid = torch.zeros(b, s, dtype=torch.bool)
        press_cheese = torch.full((b,), -1, dtype=torch.long)
        press_dir = torch.zeros(b, 3, dtype=torch.long)
        for i, (pts, press) in enumerate(zip(lists, presses)):
            if pts:
                targets[i, : len(pts)] = torch.tensor(pts)
                valid[i, : len(pts)] = True
            if press is not None:
                press_cheese[i] = press[0]
                press_dir[i] = torch.tensor(press[1])
        return targets, valid, press_cheese, press_dir

    def __call__(self, world, t, generator=None):
        lists, presses, caps, _ = self.segments(world, t)
        targets, valid, press_cheese, press_dir = self.tensors(lists, presses)
        allowed = None if self.moves is None else self.moves.unsqueeze(0).expand(int((world.mass > 0.5).sum()), -1)
        if self.rule is not None:
            allowed = rule_moves(world, self.rule) if allowed is None else rule_moves(world, self.rule) & allowed
        codes = act(world, targets, valid, press_cheese, press_dir, generator=self.generator if generator is None else generator, allowed=allowed)
        for i, (pts, cap) in enumerate(zip(lists, caps)):
            if cap is None or not pts:
                continue
            here = world.mass[i].nonzero()
            if len(here) <= cap:
                continue
            dist = torch.cdist(here.float(), torch.tensor(pts, dtype=torch.float)).min(dim=1).values
            idle = here[dist.argsort()[cap:]]
            codes[i, idle[:, 0], idle[:, 1], idle[:, 2]] = STAY
        for i, frozen in enumerate(self.frozen):  # robots a coverage job leaves out stay, jitter or not
            if frozen is not None:
                codes[i][torch.from_numpy(frozen)] = STAY
        if self.rule is not None:  # of a cube and the cube it pivots on, one waits
            codes = refuse(world, codes, self.rule, priority=torch.rand(world.mass.shape, generator=self.generator if generator is None else generator))
        return codes


def run(world, metas, tasks, actor, steps=EPISODE_STEPS, seed=0, record=False, failure=None, rule=None):
    """Roll out a batch: actor(world, t) -> action codes (B, D, H, W). Returns per episode
    (score, unasked), and with record the store arrays (mass, actions, cheese_pos, cheese_alive)
    of the episodes. failure: (27,) per-action probability that a robot's move does not happen
    this step (world.failure_rates); the recorded action stays the one the actor chose. rule: a
    pivot rule (world.RULES) the world enforces: a move it forbids does not happen either."""
    generator = torch.Generator().manual_seed(seed)
    failures = torch.Generator().manual_seed(seed + 7_919)
    runners = torch.zeros(world.cheese_alive.shape, dtype=torch.bool)
    for i, task in enumerate(tasks):
        runners[i, list(task.runners)] = True
    start = world
    eaten = [[] for _ in tasks]
    since = [(snapshot(world, i), 0) for i in range(len(tasks))]  # a switch task is scored from the step its last instruction began: (snapshot, eaten so far)
    early = [False] * len(tasks)  # a switch task ate what an earlier instruction forbade
    masses, actions, positions, alive = [world.mass.to(torch.uint8)], [], [world.cheese_pos.to(torch.int8)], [world.cheese_alive]
    for t in range(steps):
        for i, task in enumerate(tasks):
            if task.parts and t in task.spec["steps"][1:]:
                part = task.parts[instruction_index(task, t) - 1]
                early[i] = early[i] or any(c in part.spec["forbid"] for c in eaten[i][since[i][1] :])
                since[i] = (snapshot(world, i), len(eaten[i]))
        codes = actor(world, t)
        before = world.cheese_alive.clone()
        effective_codes = codes if failure is None else fail_moves(codes, failure, failures)
        world, _ = step(world, effective_codes.long(), torch.rand(world.mass.shape, generator=generator), rule)
        if runners.any() and (t + 1) % RUNNER_PERIOD == 0:
            world = replace(world, cheese_pos=flee(world, runners))
        for i, k in (before & ~world.cheese_alive).nonzero().tolist():
            eaten[i].append(COLOUR_NAMES[int(world.cheese_color[i, k])])
        if record:
            actions.append(codes.to(torch.int8))
            masses.append(world.mass.to(torch.uint8))
            positions.append(world.cheese_pos.to(torch.int8))
            alive.append(world.cheese_alive)
    results = []
    for i, task in enumerate(tasks):
        static = {"walls": start.walls[i].numpy(), "cheese_color": start.cheese_color[i].tolist(), "gravity": bool(start.gravity[i]), "meta": metas[i]}
        if task.parts:
            last, unasked = score(task.parts[-1], static, since[i][0], snapshot(world, i), eaten[i][since[i][1] :])
            results.append((last, unasked or early[i]))
            continue
        results.append(score(task, static, snapshot(start, i), snapshot(world, i), eaten[i]))
    if not record:
        return results, None
    arrays = {"mass": torch.stack(masses, 1), "actions": torch.stack(actions, 1), "cheese_pos": torch.stack(positions, 1), "cheese_alive": torch.stack(alive, 1)}
    return results, arrays


# ---------------------------------------------------------------------------------------------
# Coverage: who x what x when (README, "Coverage")
#
# A coverage task's spec is {"kind": job, "who": selector, job parameters, "forbid": colours}.
# Selectors (spec["who"]):
#   {"select": "all"}                                   everyone
#   {"select": "group", "by": "size", "which": w}       w: smaller/larger (2 groups), smallest/largest (3)
#   {"select": "group", "by": "side", "which": side}    the group furthest left/right/back/front
#   {"select": "group", "by": "cheese", "which": c}     the group nearest the c cheese
#   {"select": "top"}                                   the robots standing on other robots
#   {"select": "count", "count": k}, {"select": "half"} any k of the robots, any half of them
# Group and top selectors take "except": True, which picks everyone else. The robots a selector
# leaves out must stay where they are. A group is a 26-connected set of robots (components).

COVERAGE = ("motion", "subset", "switch")
ALL = {"select": "all"}
MOTION_JOBS = ("stay", "goto", "approach", "gather", "spread", "groups", "line", "climb", "move", "follow")
SUBSET_JOBS = {  # job -> the selectors it is drawn with
    "goto": ("group", "except", "count", "half", "top"),
    "approach": ("group", "count", "half"),
    "gather": ("except",),
    "spread": ("group", "except", "top"),
    "groups": ("group",),
    "line": ("group", "count"),
    "move": ("group", "except"),
    "perch": ("count",),
    "tower": ("group",),
    "eat": ("group",),
}
SWITCH_FIRST = ("goto", "gather", "spread", "approach", "groups", "move", "logic", "construction", "herd")
SWITCH_THEN = ("stay", "stay", "stay", "goto", "gather", "spread")
SWITCH_AT = (8, 25)  # the step the second instruction comes, drawn in [low, high)
HELD_OUT.update({"selector_job": (("half", "approach"),), "switch_then": ()})
HELD_FAMILY.update({"selector_job": "subset", "switch_then": "switch"})
SIDE_AXES = {"left": (2, -1), "right": (2, 1), "back": (1, -1), "front": (1, 1)}  # FLOOR_REGIONS: back is low y
NEAR = 3  # floor-plan (Chebyshev) distance from a cheese's footprint that counts as next to it
MOVE_STEPS = 3
LINE_MAX = 14
LAYER_COST = 3.0  # target piles: one layer up costs as much as this squared distance on the floor
GROUP_GAP = 2  # cells between the groups of a multi-group scene
GROUP_PAIRS = {2: (("left", "right"), ("back", "front"), ("back left", "front right"), ("back right", "front left")), 3: (("back left", "back right", "front"), ("front left", "front right", "back"), ("back left", "front left", "right"), ("back right", "front right", "left"))}
COVERAGE_WORDINGS = {
    "stay": ["stay where you are", "stop moving", "freeze", "everyone hold still", "don't move"],
    "approach": ["go next to the {c} cheese without eating it", "get close to the {c} cheese but leave it alone", "move up beside the {c} cheese", "head over next to the {c} cheese, no eating"],
    "follow": ["follow the {c} cheese without eating it", "chase the {c} cheese but do not eat it", "stay close to the {c} cheese as it runs", "keep after the {c} cheese, no eating"],
    "gather": ["gather into one group", "come together in one group", "everyone get together", "form one single group", "merge into a single group"],
    "spread": ["spread out so that no two of you touch", "spread out", "move apart from each other", "keep your distance from one another", "scatter so nobody touches anybody"],
    "groups.two": ["split into two separate groups", "form two separate groups", "break up into two groups", "divide yourselves into two groups"],
    "groups.three": ["split into three separate groups", "form three separate groups", "break up into three groups", "divide yourselves into three groups"],
    "line.x": ["line up in one row from left to right", "form a line running left to right", "get in a row from left to right", "stand in a single row across, left to right"],
    "line.y": ["line up in one row from back to front", "form a line running back to front", "get in a row from back to front", "stand in single file from back to front"],
    "climb": ["climb onto the plateau on the {r}", "get up on the plateau on the {r}", "everyone up onto the plateau on the {r}", "scale the cliff to the plateau on the {r}"],
    "move": ["move three steps toward the {r}", "shift three steps toward the {r}", "take three steps toward the {r}", "step over three toward the {r}"],
    "subset": ["{who}: {job}; everyone else: stay where you are", "only {who} should move: {job}", "{who}, {job}; the rest of you stay put", "I want {who} to {job} while nobody else moves", "{who}: {job}. the others stay still"],
    "who.smaller": ["the smaller group", "the small group", "the group with fewer robots"],
    "who.larger": ["the larger group", "the big group", "the group with more robots"],
    "who.smallest": ["the smallest group", "the tiniest group", "the group with the fewest robots"],
    "who.largest": ["the largest group", "the biggest group", "the group with the most robots"],
    "who.side": ["the group on the {r}", "the group over on the {r}", "the robots on the {r}"],
    "who.cheese": ["the group nearest the {c} cheese", "the group closest to the {c} cheese", "the robots by the {c} cheese"],
    "who.count": ["{n} of you", "any {n} of you", "just {n} robots"],
    "who.half": ["half of you", "half the swarm", "one half of the robots"],
    "who.top": ["the robots standing on other robots", "the ones on top", "the top layer"],
    "who.not": ["everyone except {x}", "all but {x}", "everybody other than {x}"],
    "do.region": ["go to the {r} of the floor", "head to the {r}", "move over to the {r}"],
    "do.approach": ["go next to the {c} cheese without eating it", "get close to the {c} cheese", "move up beside the {c} cheese"],
    "do.gather": ["gather into one group", "come together in one group", "merge into a single group"],
    "do.spread": ["spread out so that no two of you touch", "spread out", "move apart from each other"],
    "do.groups": ["split into two separate groups", "form two separate groups", "break up into two groups"],
    "do.line_x": ["line up in one row from left to right", "form a line running left to right", "get in a row from left to right"],
    "do.line_y": ["line up in one row from back to front", "form a line running back to front", "get in a row from back to front"],
    "do.move": ["move three steps toward the {r}", "shift three steps toward the {r}", "take three steps toward the {r}"],
    "do.perch": ["climb on top of the {c} cheese", "get up onto the {c} cheese", "stand on top of the {c} cheese"],
    "do.tower": ["build a tower {n} robots tall", "stack up {n} high", "make a column {n} robots high"],
    "do.eat": ["eat the {c} cheese", "go eat the {c} one", "devour the {c} cheese"],
    "switch": ["{x}", "change of plan: {x}", "actually, {x}", "new instruction: {x}", "never mind that, {x}"],
}


def components(occupied):
    """26-connected groups of robots in a (D, H, W) bool grid: [(n, 3) int arrays], largest first
    (ties by their first voxel)."""
    cells = np.argwhere(occupied)
    index = {tuple(c): j for j, c in enumerate(cells.tolist())}
    seen = np.zeros(len(cells), bool)
    found = []
    for j in range(len(cells)):
        if seen[j]:
            continue
        seen[j] = True
        stack, members = [j], []
        while stack:
            a = stack.pop()
            members.append(a)
            z, y, x = cells[a]
            for dz, dy, dx in DIRS:
                b = index.get((z + dz, y + dy, x + dx))
                if b is not None and not seen[b]:
                    seen[b] = True
                    stack.append(b)
        found.append(cells[sorted(members)])
    return sorted(found, key=lambda g: (-len(g), tuple(g[0])))


def region_of(y, x):
    """The floor region (FLOOR_REGIONS) that holds floor point (y, x)."""
    y, x = min(max(float(y), 0.0), SIZE - 1e-6), min(max(float(x), 0.0), SIZE - 1e-6)
    return next(name for name in FLOOR_REGIONS if region_box(name)[0] <= y < region_box(name)[1] and region_box(name)[2] <= x < region_box(name)[3])


def cheese_centres(colours, snap):
    """{colour: (z, y, x) centre} of the alive cheeses of a snapshot; colours: cheese_color row."""
    return {COLOUR_NAMES[int(c)]: cheese_centre(snap, k) for k, c in enumerate(colours) if c >= 0 and snap["cheese_alive"][k]}


def take_count(who, total):
    """How many robots a count selector asks for out of `total`."""
    return int(who["count"]) if who["select"] == "count" else total // 2


def chosen(who, occupied, centres):
    """(D, H, W) bool: the robots a selector picks out of `occupied` (bool); None for "count" and
    "half", which any robots fulfil, and when the selector names nothing in the scene."""
    kind = who["select"]
    if kind in ("count", "half"):
        return None
    if kind == "all":
        return occupied.copy()
    if kind == "top":
        pick = np.zeros_like(occupied)
        pick[1:] = occupied[1:] & occupied[:-1]
    else:
        groups = components(occupied)
        by, which = who["by"], who["which"]
        if len(groups) < 2 or (by == "cheese" and which not in centres):
            return None
        if by == "size":
            group = groups[-1] if which in ("smaller", "smallest") else groups[0]
        elif by == "side":
            axis, sign = SIDE_AXES[which]
            group = max(groups, key=lambda g: sign * g[:, axis].mean())  # furthest that way
        else:
            centre = np.asarray(centres[which])
            group = min(groups, key=lambda g: np.linalg.norm(g.mean(axis=0) - centre))
        pick = np.zeros_like(occupied)
        pick[tuple(group.T)] = True
    return occupied & ~pick if who.get("except") else pick


def floor_distance(points_yx, pos):
    """Chebyshev distance in the floor plan from points (n, 2) to a cheese footprint at pos (z, y, x)."""
    lo = np.asarray(pos[1:], float)
    hi = lo + CHEESE_SIDE - 1
    gap = np.maximum(np.maximum(lo - points_yx, points_yx - hi), 0)
    return gap.max(axis=1)


def longest_line(occupied, axis):
    """The longest straight run of robots along floor axis 1 (y) or 2 (x)."""
    grid = np.moveaxis(occupied, axis, -1)
    run = np.zeros(grid.shape[:-1], int)
    best = 0
    for j in range(grid.shape[-1]):
        run = np.where(grid[..., j], run + 1, 0)
        best = max(best, int(run.max(initial=0)))
    return best


def face_alone(occupied):
    """(D, H, W) bool: robots with no robot on any of their six faces."""
    padded = np.pad(occupied, 1)
    touching = np.zeros_like(occupied)
    for axis in range(3):
        for sign in (-1, 1):
            touching |= np.roll(padded, sign, axis=axis)[1:-1, 1:-1, 1:-1]
    return occupied & ~touching


def job_value(spec, static, start, end, eaten, movers, n, pick):
    """How far the robots taking part (movers, at the end) did the job, in [0, 1]; n is how many
    take part."""
    kind = spec["kind"]
    n = max(n, 1)
    cells = np.argwhere(movers)
    slot = {COLOUR_NAMES[int(c)]: k for k, c in enumerate(static["cheese_color"]) if c >= 0}
    if kind in ("approach", "follow", "perch"):
        k = slot[spec["colour"]]
        if not end["cheese_alive"][k]:
            return 0.0
        pos = end["cheese_pos"][k]
        if kind == "perch":
            top = movers[pos[0] + CHEESE_SIDE, pos[1] : pos[1] + CHEESE_SIDE, pos[2] : pos[2] + CHEESE_SIDE] if pos[0] + CHEESE_SIDE < movers.shape[0] else np.zeros(1, bool)
            return min(1.0, float(top.sum()) / min(n, CHEESE_SIDE * CHEESE_SIDE))
        reach = NEAR if kind == "approach" else RUNNER_RADIUS
        return min(1.0, float((floor_distance(cells[:, 1:], pos) <= reach).sum()) / n) if len(cells) else 0.0
    if kind == "goto":
        y0, y1, x0, x1 = region_box(spec["region"])
        inside = (cells[:, 1] >= y0 - 1) & (cells[:, 1] < y1 + 1) & (cells[:, 2] >= x0 - 1) & (cells[:, 2] < x1 + 1)
        return min(1.0, float(inside.sum()) / n)
    if kind == "gather":
        groups = components(movers)
        return min(1.0, len(groups[0]) / n) if groups else 0.0
    if kind == "spread":
        return min(1.0, float((face_alone(end["mass"] > 0.5) & movers).sum()) / n)
    if kind == "groups":
        m = spec["count"]
        shares = [len(g) / n for g in components(movers)] + [0.0] * m
        return float(min(1.0, sum(shares[:m]) / 0.8) * min(1.0, shares[m - 1] / (0.5 / m)))
    if kind == "line":
        return min(1.0, longest_line(movers, 2 if spec["axis"] == "x" else 1) / n)
    if kind == "climb":
        y0, y1, x0, x1 = spec["plateau"]
        on = (cells[:, 0] >= spec["top"]) & (cells[:, 1] >= y0) & (cells[:, 1] < y1) & (cells[:, 2] >= x0) & (cells[:, 2] < x1)
        return min(1.0, float(on.sum()) / n)
    if kind == "move":
        axis, sign = SIDE_AXES[spec["toward"]]
        before = np.argwhere(pick if pick is not None else start["mass"] > 0.5)
        if not len(before) or not len(cells):
            return 0.0
        shift = sign * (cells[:, axis].mean() - before[:, axis].mean())
        return float(np.clip(shift / spec["steps"], 0.0, 1.0))
    if kind == "tower":
        return min(1.0, tallest_run(movers) / spec["height"])
    if kind == "eat":
        return float(spec["colour"] in eaten)
    raise ValueError(kind)


def job_score(task, static, start, end, eaten):
    """(score, unasked) of a coverage task: how far the robots taking part did the job, times the
    share of the robots left out that are still where they were. "stay" is the share of robots
    still on their start voxels. A count selector ("N of you", "half of you") picks no robots in
    advance: the robots off their start voxels are the ones that moved, the job is judged on them,
    and moving more than N costs N / moved."""
    spec = task.spec
    unasked = any(c in spec["forbid"] for c in eaten)
    before, after = start["mass"] > 0.5, end["mass"] > 0.5
    if spec["kind"] == "stay":
        return float((before & after).sum() / max(before.sum(), 1)), unasked
    who = spec["who"]
    pick = chosen(who, before, cheese_centres(static["cheese_color"], start))
    if pick is None and who["select"] not in ("count", "half"):
        return 0.0, unasked
    if pick is None:
        n = take_count(who, int(before.sum()))
        movers = after & ~before
        still = min(1.0, n / max(int(movers.sum()), 1))
    else:
        n = int(pick.sum())
        frozen = before & ~pick
        movers = after & ~frozen
        still = float((after & frozen).sum() / frozen.sum()) if frozen.any() else 1.0
    return float(job_value(spec, static, start, end, eaten, movers, n, pick) * still), unasked


def selected_count(task, world, i):
    """How many robots a coverage task's selector picks in scene i as it is now (None: none)."""
    occupied = world.mass[i].numpy() > 0.5
    who = task.spec["who"]
    if who["select"] in ("count", "half"):
        return take_count(who, int(occupied.sum()))
    pick = chosen(who, occupied, cheese_centres(world.cheese_color[i].tolist(), snapshot(world, i)))
    return None if pick is None else int(pick.sum())


def coverage_bound(task, world, i, n=None):
    """An upper bound on a coverage task's score for the scene's swarm (n: a swarm of that size
    for an "everyone" task): 0 when the selector picks nobody or more robots than there are; a
    tower takes 4 robots a level, eating eat_need robots, a line at most LINE_MAX robots, split
    into m groups 2 m robots, spreading out the spaced floor cells there are (spread_cells)."""
    spec = task.spec
    if spec["kind"] == "stay":
        return 1.0
    total = int(world.mass[i].sum())
    count = selected_count(task, world, i) if spec["who"]["select"] != "all" else (total if n is None else n)
    if not count or count > total:
        return 0.0
    kind = spec["kind"]
    bound = 1.0
    if kind == "tower":
        bound = min(1.0, count / (CHEESE_SIDE * CHEESE_SIDE * spec["height"]))
    elif kind == "eat":
        slot = {c: k for k, c in present(world, i)}
        bound = float(spec["colour"] in slot and count >= eat_need(world, i, slot[spec["colour"]]))
    elif kind == "line":
        bound = min(1.0, LINE_MAX / count)
    elif kind == "groups":
        bound = float(count >= 2 * spec["count"])
    elif kind == "spread":
        bound = min(1.0, len(spread_cells(world, i, np.zeros((SIZE,) * 3, bool))) / count)
    return bound


# Scenes with separate groups


def place_groups(rng, world, count, total, layers=1):
    """(1, D, H, W) mass: `total` robots of a one-scene world set down as `count` separate groups
    of different sizes (each at least 1.4 times the next smaller), compact piles of at most
    `layers` layers around the centres of far-apart floor regions (GROUP_PAIRS), GROUP_GAP cells
    apart, off cheeses and their shells. None when they do not fit."""
    if total < 3 * count:
        return None
    weights = np.sort(rng.uniform(1.0, 1.25, count) * 1.45 ** np.arange(count))
    sizes = np.maximum(np.floor(weights / weights.sum() * total).astype(int), 2)
    sizes[-1] += total - sizes.sum()
    if np.any(np.diff(sizes) < 1) or sizes.min() < 2 or np.any(sizes[1:] < 1.4 * sizes[:-1]):
        return None
    names = list(GROUP_PAIRS[count][int(rng.integers(len(GROUP_PAIRS[count])))])
    rng.shuffle(names)
    heights = world.walls[0].numpy().sum(0)
    avoid = world.walls[0].numpy() | (cheese_id_map(world)[0].numpy() >= 0) | shells_of(world, 0)
    mass = np.zeros(world.mass.shape[1:], bool)
    taken = np.zeros((SIZE, SIZE), bool)  # floor cells of the groups placed so far, grown by GROUP_GAP
    for size, name in zip(sizes, names):
        y0, y1, x0, x1 = region_box(name)
        centre = ((y0 + y1 - 1) / 2 + rng.uniform(-1, 1), (x0 + x1 - 1) / 2 + rng.uniform(-1, 1))
        points = stack_targets(heights, avoid, centre, int(size), cells=~taken, layers=layers)
        if len(points) < size:
            return None
        pile = np.zeros_like(mass)
        pile[tuple(np.array(points).T)] = True
        if len(components(pile)) != 1:
            return None
        mass |= pile
        floor = pile.any(axis=0)
        for _ in range(GROUP_GAP + 1):
            floor = floor | np.roll(floor, 1, 0) | np.roll(floor, -1, 0) | np.roll(floor, 1, 1) | np.roll(floor, -1, 1)
        taken |= floor
    if len(components(mass)) != count:
        return None
    return torch.from_numpy(mass).unsqueeze(0).to(world.mass.dtype)


def shells_of(world, i):
    """(D, H, W) bool: the shell voxels (inside the grid) of scene i's alive cheeses."""
    out = np.zeros(world.mass.shape[1:], bool)
    for k in range(MAX_CHEESES):
        if world.cheese_alive[i, k]:
            coords = world.cheese_pos[i, k].numpy() + SHELL_OFFSETS.numpy()
            inside = ((coords >= 0) & (coords < np.array(out.shape))).all(axis=1)
            out[tuple(coords[inside].T)] = True
    return out


def stack_targets(heights, avoid, centre, n, cells=None, layers=None):
    """The n voxels nearest a floor point centre (y, x), as a pile: (z, y, x) from the ground
    (heights, (H, W)) up, ranked by squared floor distance plus LAYER_COST per layer squared, so
    the pile grows outward and up and every voxel's lower neighbour comes first. avoid (D, H, W)
    bool voxels are left out together with what stands on them; cells (H, W) bool limits the
    floor plan; layers caps the height of the pile."""
    d, h, w = avoid.shape
    yy, xx = np.mgrid[0:h, 0:w]
    floor = (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2
    open_column = np.ones((h, w), bool) if cells is None else cells.copy()
    keys, points = [], []
    for layer in range(d if layers is None else layers):
        z = heights + layer
        inside = z < d
        open_column &= inside
        open_column[inside] &= ~avoid[z[inside], yy[inside], xx[inside]]
        if not open_column.any():
            break
        ys, xs = np.nonzero(open_column)
        keys.append(floor[ys, xs] + LAYER_COST * layer * layer)
        points += list(zip((heights[ys, xs] + layer).tolist(), ys.tolist(), xs.tolist()))
    if not points:
        return []
    order = np.argsort(np.concatenate(keys), kind="stable")[:n]
    return [points[j] for j in order]


def spread_cells(world, i, frozen):
    """Ground voxels a spread-out swarm can stand on: one floor cell in every 2 x 2 block (the
    parity with the most free cells), or the checkerboard when that holds more, off cheeses and
    their shells and away from the faces of frozen robots."""
    heights = world.walls[i].numpy().sum(0)
    avoid = world.walls[i].numpy() | (cheese_id_map(world.index(i))[0].numpy() >= 0) | shells_of(world, i)
    near = frozen.copy()
    for axis in range(3):
        for sign in (-1, 1):
            near |= np.roll(frozen, sign, axis=axis)
    avoid |= near
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    ok = (heights < SIZE)
    ok[ok] &= ~avoid[heights[ok], yy[ok], xx[ok]]
    best = []
    for mask in [(yy % 2 == a) & (xx % 2 == b) for a in (0, 1) for b in (0, 1)] + [(yy + xx) % 2 == a for a in (0, 1)]:
        ys, xs = np.nonzero(ok & mask)
        if len(ys) > len(best):
            best = list(zip(heights[ys, xs].tolist(), ys.tolist(), xs.tolist()))
    return best


class JobSetup:
    """A coverage job made concrete on the scene as it is when the job starts: who takes part (the
    others are frozen and stay) and the voxels they head for (points)."""

    def __init__(self, job, world, i):
        self.job = job
        occupied = world.mass[i].numpy() > 0.5
        self.heights = world.walls[i].numpy().sum(0)
        self.solid = world.walls[i].numpy() | (cheese_id_map(world.index(i))[0].numpy() >= 0)
        slots = {c: k for k, c in present(world, i)}
        self.slot = slots.get(job.get("colour"))
        if job["kind"] == "stay":
            self.pick = np.zeros_like(occupied)
        else:
            pick = chosen(job["who"], occupied, cheese_centres(world.cheese_color[i].tolist(), snapshot(world, i)))
            if pick is None:  # any k: those nearest where the job goes
                anchor = self.anchor(world, i, occupied)
                cells = np.argwhere(occupied)
                order = np.argsort(((cells[:, 1:] - anchor) ** 2).sum(axis=1) - 0.01 * cells[:, 0], kind="stable")
                pick = np.zeros_like(occupied)
                k = take_count(job["who"], len(cells)) if job["who"]["select"] in ("count", "half") else 0
                pick[tuple(cells[order[:k]].T)] = True
            self.pick = pick
        self.frozen = occupied & ~self.pick
        self.n = int(self.pick.sum())
        self.avoid = self.solid | self.frozen | (shells_of(world, i) if job["kind"] != "eat" else False)
        self.fixed = None if job["kind"] == "follow" else self.compute(world, i)

    def anchor(self, world, i, occupied):
        """The floor point (y, x) a job heads for, for choosing which robots take part."""
        job = self.job
        if self.slot is not None:
            return world.cheese_pos[i, self.slot, 1:].numpy() + CHEESE_SIDE / 2 - 0.5
        if job["kind"] == "goto":
            y0, y1, x0, x1 = region_box(job["region"])
            return np.array([(y0 + y1 - 1) / 2, (x0 + x1 - 1) / 2])
        if job["kind"] == "climb":
            y0, y1, x0, x1 = job["plateau"]
            return np.array([(y0 + y1 - 1) / 2, (x0 + x1 - 1) / 2])
        return np.argwhere(occupied)[:, 1:].mean(axis=0)

    def points(self, world, i):
        return self.compute(world, i) if self.fixed is None else self.fixed

    def centroid(self):
        return np.argwhere(self.pick)[:, 1:].mean(axis=0) if self.n else np.array([SIZE / 2, SIZE / 2])

    def compute(self, world, i):
        job, n = self.job, self.n
        kind = job["kind"]
        if kind in ("stay", "eat") or n == 0:  # eat: the shell of the cheese, as the eat segment (Expert.targets_of)
            return []
        if kind in ("approach", "follow"):
            pos = world.cheese_pos[i, self.slot].numpy()
            yy, xx = np.mgrid[0:SIZE, 0:SIZE]
            dist = floor_distance(np.stack([yy.ravel(), xx.ravel()], 1), pos).reshape(SIZE, SIZE)
            return stack_targets(self.heights, self.avoid, pos[1:] + 0.5, n, cells=(dist >= 1) & (dist <= NEAR))
        if kind == "perch":
            z, y, x = world.cheese_pos[i, self.slot].tolist()
            return [(z + CHEESE_SIDE, y + a, x + b) for a in range(CHEESE_SIDE) for b in range(CHEESE_SIDE) if z + CHEESE_SIDE < SIZE][:n]
        if kind == "goto":
            y0, y1, x0, x1 = region_box(job["region"])
            cells = np.zeros((SIZE, SIZE), bool)
            cells[max(y0 - 1, 0) : y1 + 1, max(x0 - 1, 0) : x1 + 1] = True
            return stack_targets(self.heights, self.avoid, ((y0 + y1 - 1) / 2, (x0 + x1 - 1) / 2), n, cells=cells)
        if kind == "climb":
            y0, y1, x0, x1 = job["plateau"]
            cells = np.zeros((SIZE, SIZE), bool)
            cells[y0:y1, x0:x1] = True
            cells &= self.heights >= job["top"]
            ys, xs = np.nonzero(cells)
            mine = self.centroid()
            near = int(np.argmin((ys - mine[0]) ** 2 + (xs - mine[1]) ** 2))
            return stack_targets(self.heights, self.avoid, (ys[near], xs[near]), n, cells=cells)
        if kind == "gather":
            return stack_targets(self.heights, self.avoid, self.centroid(), n)
        if kind == "spread":
            mine = self.centroid()
            options = spread_cells(world, i, self.frozen)
            options.sort(key=lambda p: (p[1] - mine[0]) ** 2 + (p[2] - mine[1]) ** 2)
            return options[:n]
        if kind == "groups":
            return self.group_points(job["count"])
        if kind == "line":
            return self.line_points(2 if job["axis"] == "x" else 1)
        if kind == "move":
            return self.move_points(world, i)
        if kind == "tower":
            return self.tower_points(job["height"])
        raise ValueError(kind)

    def group_points(self, m):
        """n targets as m piles around the centres of far-apart floor regions (GROUP_PAIRS), the
        set the robots taking part reach with the least travel; each pile keeps to the floor cells
        nearer its centre than any other by GROUP_GAP."""
        cells = np.argwhere(self.pick)[:, 1:]
        best = None
        for names in GROUP_PAIRS[m]:
            centres = np.array([[(region_box(r)[0] + region_box(r)[1] - 1) / 2, (region_box(r)[2] + region_box(r)[3] - 1) / 2] for r in names])
            cost = np.sqrt(((cells[:, None, :] - centres[None]) ** 2).sum(-1)).min(axis=1).sum()
            if best is None or cost < best[0]:
                best = (cost, centres)
        centres = best[1]
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        dist = np.sqrt((yy[None] - centres[:, 0, None, None]) ** 2 + (xx[None] - centres[:, 1, None, None]) ** 2)
        out = []
        for j in range(m):
            others = np.delete(dist, j, axis=0).min(axis=0)
            share = self.n // m + (1 if j < self.n % m else 0)
            out += stack_targets(self.heights, self.avoid, centres[j], share, cells=dist[j] + GROUP_GAP < others)
        return out

    def line_points(self, axis):
        """n ground voxels in one straight row along floor axis 1 or 2, level and clear, as near
        the robots taking part as such a row lies."""
        mine = self.centroid()
        other = 3 - axis  # the floor axis the row is fixed on (1 or 2)
        n = min(self.n, SIZE)
        rows = sorted(range(SIZE), key=lambda r: abs(r - mine[other - 1]))
        starts = sorted(range(SIZE - n + 1), key=lambda s: abs(s + (n - 1) / 2 - mine[axis - 1]))
        for r in rows:
            for s in starts:
                yx = [(r, s + j) if axis == 2 else (s + j, r) for j in range(n)]
                ground = [int(self.heights[y, x]) for y, x in yx]
                if len(set(ground)) == 1 and ground[0] < SIZE and not any(self.avoid[ground[0], y, x] for y, x in yx):
                    return [(ground[0], y, x) for y, x in yx]
        return []

    def move_points(self, world, i):
        """The robots taking part, shifted MOVE_STEPS cells toward the side, each at its height
        above the ground there; what lands on a wall or twice goes on a pile at the shifted centre."""
        axis, sign = SIDE_AXES[self.job["toward"]]
        out, seen = [], set()
        for z, y, x in np.argwhere(self.pick).tolist():
            yx = [y, x]
            yx[axis - 1] = min(max(yx[axis - 1] + sign * self.job["steps"], 0), SIZE - 1)
            above = z - int(self.heights[y, x])
            z2 = int(self.heights[yx[0], yx[1]]) + max(above, 0)
            point = (z2, yx[0], yx[1])
            if z2 < SIZE and not self.avoid[point] and point not in seen:
                seen.add(point)
                out.append(point)
        if len(out) < self.n:
            centre = self.centroid().copy()
            centre[axis - 1] += sign * self.job["steps"]
            more = [p for p in stack_targets(self.heights, self.avoid, centre, self.n + len(out)) if p not in seen]
            out += more[: self.n - len(out)]
        return out

    def tower_points(self, height):
        """A 2 x 2 column `height` high on level clear ground nearest the robots taking part."""
        mine = self.centroid()
        options = sorted(((y, x) for y in range(SIZE - 1) for x in range(SIZE - 1)), key=lambda p: (p[0] + 0.5 - mine[0]) ** 2 + (p[1] + 0.5 - mine[1]) ** 2)
        for y, x in options:
            foot = self.heights[y : y + 2, x : x + 2]
            z0 = int(foot[0, 0])
            if (foot == z0).all() and z0 + height <= SIZE and not self.avoid[z0 : z0 + height, y : y + 2, x : x + 2].any():
                return [(z, y + a, x + b) for z in range(z0, z0 + height) for a in range(2) for b in range(2)]
        return []


# Drawing coverage tasks


def coverage_task(family, text, spec, held, runners=()):
    """A coverage Task: one job segment as its plan (none when held out)."""
    return Task(family, text, spec, [{"kind": "job", "job": spec}] if held in ("", "wording") else [], held, runners)


def static_of(world, meta, i):
    return {"walls": world.walls[i].numpy(), "cheese_color": world.cheese_color[i].tolist(), "gravity": bool(world.gravity[i]), "meta": meta}


def already_done(task, world, meta, i):
    """Whether the scene as it starts already scores half marks on the task."""
    snap = snapshot(world, i)
    return job_score(task, static_of(world, meta, i), snap, snap, [])[0] >= 0.5


def plateaus(meta):
    """{side: ([y0, y1, x0, x1], top)} of a gap scene's two plateaus."""
    axis, start, width, top = trench(meta)
    names = ("left", "right") if axis == 2 else ("back", "front")
    boxes = ([0, SIZE, 0, start], [0, SIZE, start + width, SIZE]) if axis == 2 else ([0, start, 0, SIZE], [start + width, SIZE, 0, SIZE])
    return {name: (box, top) for name, box in zip(names, boxes)}


def job_spec(rng, job, world, meta, i, n):
    """(spec fields, wording key, wording arguments) of a job drawn for scene i, n robots taking
    part; None when the scene does not allow it. The key is the motion intent; subset tasks use
    the "do." form of the same job."""
    cheeses = present(world, i)
    colours = [c for _, c in cheeses]
    if job == "stay":
        return {}, "stay", {}
    if job == "goto":
        r = str(rng.choice(list(FLOOR_REGIONS)))
        return {"region": r}, "region", {"r": r}
    if job in ("approach", "follow", "perch", "eat"):
        if not colours:
            return None
        c = str(rng.choice(colours))
        return {"colour": c}, job, {"c": c}
    if job in ("gather", "spread"):
        return {}, job, {}
    if job == "groups":
        m = 3 if n >= 12 and rng.random() < 0.4 else 2
        return {"count": m}, "groups." + ("two" if m == 2 else "three"), {}
    if job == "line":
        axis = str(rng.choice(["x", "y"]))
        return {"axis": axis}, "line." + axis, {}
    if job == "climb":
        if trench(meta) is None:
            return None
        sides = plateaus(meta)
        side = str(rng.choice(sorted(sides)))
        box, top = sides[side]
        return {"side": side, "plateau": [int(v) for v in box], "top": int(top)}, "climb", {"r": side}
    if job == "move":
        r = str(rng.choice(list(SIDE_AXES)))
        return {"toward": r, "steps": MOVE_STEPS}, "move", {"r": r}
    if job == "tower":
        if n < 12:
            return None
        h = int(rng.integers(3, min(8, n // 4) + 1))
        return {"height": h}, "tower", {"n": h}
    raise ValueError(job)


MOTION_NEEDS = {
    "stay": {"cheeses": (0, 3)},
    "goto": {"cheeses": (0, 2)},
    "approach": {"cheeses": (1, 3), "layout": "flat"},
    "gather": {"cheeses": (0, 2), "groups": (2, 3)},
    "spread": {"cheeses": (0, 2)},
    "groups": {"cheeses": (0, 2)},
    "line": {"cheeses": (0, 1), "robots": (3, LINE_MAX)},
    "climb": {"cheeses": (0, 1), "layout": "gap", "gravity": True},
    "move": {"cheeses": (0, 2)},
    "follow": {"cheeses": (1, 2), "layout": "flat"},
}


def make_motion(rng, held, job=None):
    """Everyone does one job (MOTION_JOBS)."""
    job = job or str(rng.choice(MOTION_JOBS))

    def make(world, meta, i):
        drawn = job_spec(rng, job, world, meta, i, int(world.mass[i].sum()))
        if drawn is None:
            return None
        fields, key, kw = drawn
        spec = {"kind": job, "who": dict(ALL), **fields, "forbid": set(c for _, c in present(world, i))}
        runners = ()
        if job == "follow":
            runners = (next(k for k, c in present(world, i) if c == fields["colour"]),)
        task = coverage_task("motion", say(rng, key, held, **kw), spec, held, runners)
        if job != "stay" and already_done(task, world, meta, i):
            return None
        return task

    needs = dict(MOTION_NEEDS[job])
    if job == "gather" and rng.random() < 0.3:
        needs.pop("groups")  # a scattered swarm gathers too
    return needs, make


def who_text(rng, who, held):
    """The wording of a selector (COVERAGE_WORDINGS "who.")."""
    kind = who["select"]
    if kind == "group":
        key, kw = ("who.side", {"r": who["which"]}) if who["by"] == "side" else ("who.cheese", {"c": who["which"]}) if who["by"] == "cheese" else ("who." + who["which"], {})
    else:
        key, kw = ("who." + kind, {"n": who["count"]} if kind == "count" else {})
    text = say(rng, key, held, **kw)
    return say(rng, "who.not", held, x=text) if who.get("except") else text


def draw_who(rng, kind, world, i, most=8):
    """A selector of a kind ("group", "except", "count", "half", "top") that picks something in
    scene i, or None; a count selector asks for 3 to `most` robots."""
    occupied = world.mass[i].numpy() > 0.5
    total = int(occupied.sum())
    if kind in ("count", "half"):
        who = {"select": "count", "count": int(rng.integers(3, min(most, total - 1) + 1))} if kind == "count" and total > 4 else {"select": "half"} if kind == "half" and total >= 6 else None
        return who
    if kind == "top":
        return {"select": "top"} if (occupied[1:] & occupied[:-1]).sum() >= 3 else None
    groups = components(occupied)
    if len(groups) < 2:
        return None
    centres = cheese_centres(world.cheese_color[i].tolist(), snapshot(world, i))
    options = [("size", "smaller" if len(groups) == 2 else "smallest"), ("size", "larger" if len(groups) == 2 else "largest")]
    for side, (axis, sign) in SIDE_AXES.items():
        means = sorted(sign * g[:, axis].mean() for g in groups)
        if means[-1] - means[-2] >= 3:  # one group clearly furthest that way
            options.append(("side", side))
    for colour, centre in centres.items():
        dist = sorted(np.linalg.norm(g.mean(axis=0) - np.asarray(centre)) for g in groups)
        if dist[1] - dist[0] >= 3:
            options.append(("cheese", colour))
    if len(groups[-1]) * 1.4 > len(groups[-2]):  # sizes too close to tell apart
        options = [o for o in options if o[1] not in ("smaller", "smallest")]
    if len(groups[1]) * 1.4 > len(groups[0]):
        options = [o for o in options if o[1] not in ("larger", "largest")]
    if not options:
        return None
    by, which = options[int(rng.integers(len(options)))]
    who = {"select": "group", "by": by, "which": which}
    if kind == "except":
        who["except"] = True
    return who


def make_subset(rng, held):
    """A selector picks robots and they do a job; everyone else stays (SUBSET_JOBS)."""
    pairs = [(sel, job) for job, sels in SUBSET_JOBS.items() for sel in sels]
    if held == "selector_job":
        pairs = list(HELD_OUT["selector_job"])
    else:
        pairs = [p for p in pairs if p not in HELD_OUT["selector_job"]]
    sel, job = pairs[int(rng.integers(len(pairs)))]

    def make(world, meta, i):
        who = draw_who(rng, sel, world, i, CHEESE_SIDE * CHEESE_SIDE if job == "perch" else 8)
        if who is None:
            return None
        spec = {"kind": job, "who": who, "forbid": set(c for _, c in present(world, i))}
        n = selected_count(Task("subset", "", spec), world, i)
        if not n or n == int(world.mass[i].sum()):
            return None
        drawn = job_spec(rng, job, world, meta, i, n)
        if drawn is None:
            return None
        fields, key, kw = drawn
        spec.update(fields)
        if job == "eat":
            spec["forbid"] = spec["forbid"] - {fields["colour"]}
        key = {"region": "do.region", "groups.two": "do.groups", "groups.three": "do.groups", "line.x": "do.line_x", "line.y": "do.line_y"}.get(key, "do." + key)
        if job == "groups":
            spec["count"] = 2
        text = say(rng, "subset", held, who=who_text(rng, who, held), job=say(rng, key, held, **kw))
        task = coverage_task("subset", text, spec, held)
        if already_done(task, world, meta, i):
            return None
        return task

    needs = {"cheeses": (1, 3) if job in ("approach", "perch", "eat") else (0, 2), "layout": "flat" if job in ("perch", "line") else None}
    if sel in ("group", "except"):
        needs["groups"] = (2, 3)
    if sel == "top":
        needs["groups"] = (2, 3)
        needs["layers"] = 2
    if job in ("tower", "eat"):
        needs["gravity"] = True if job == "tower" else None
    if sel == "count" and job == "perch":
        needs["groups"] = None
    return needs, make


def make_switch(rng, held):
    """Task A from step 0, then task B from step k (SWITCH_AT): B is "stay" half the time, else a
    redirect (goto, gather, spread). The episode is scored on B from step k."""
    first = str(rng.choice(SWITCH_FIRST))
    then = str(rng.choice(SWITCH_THEN))
    at = int(rng.integers(*SWITCH_AT))
    if first in MAKERS:
        needs, make_first = MAKERS[first](rng, held if held in ("", "wording") else "")
    else:
        needs, make_first = make_motion(rng, held if held in ("", "wording") else "", first)
    needs.pop("robots", None)
    _, make_then = make_motion(rng, held if held in ("", "wording") else "", then)

    def make(world, meta, i):
        a = make_first(world, meta, i)
        if a is None:
            return None
        b = make_then(world, meta, i) if then != "goto" else None
        if then == "goto":
            taken = a.spec.get("region")
            r = str(rng.choice([name for name in FLOOR_REGIONS if name != taken]))
            spec = {"kind": "goto", "who": dict(ALL), "region": r, "forbid": set(c for _, c in present(world, i))}
            b = coverage_task("motion", say(rng, "region", held if held in ("", "wording") else "", r=r), spec, held)
        if b is None:
            return None
        b = replace(b, text=say(rng, "switch", held if held in ("", "wording") else "", x=b.text))
        return Task("switch", a.text, {"kind": "switch", "steps": [0, at], "forbid": set(a.spec["forbid"]) | set(b.spec["forbid"])}, a.plan, held, tuple(sorted(set(a.runners) | set(b.runners))), [a, b])

    return needs, make


COVERAGE_MAKERS = {"motion": make_motion, "subset": make_subset, "switch": make_switch}


def instruction_index(task, t):
    """Which of a switch task's instructions (Task.parts) is in force at step t."""
    return max(j for j, start in enumerate(task.spec["steps"]) if start <= t)


def instruction_at(task, t):
    """The task whose instruction is in force at step t: the task itself, or a switch task's part."""
    return task.parts[instruction_index(task, t)] if task.parts else task


def sample_coverage(rng, n, family=None, split="train", robots=None, counts="uniform", check=True):
    """(world, metas, tasks) of n scenes with one coverage task each (COVERAGE families), as
    sample_tasks draws the suite's: family None draws one of COVERAGE per scene; split "train",
    "test" or a HELD_OUT key of a coverage family; robots (low, high) draws each scene's robot
    count (counts: "uniform" or "log"), None the world's density range; check keeps only tasks
    whose score_bound reaches the pass mark."""
    held = split if split in HELD_OUT else ""
    worlds, metas, out = [], [], []
    while len(out) < n:
        fam = HELD_FAMILY.get(held) if held in ("selector_job", "switch_then") else family or str(rng.choice(COVERAGE))
        needs, make = COVERAGE_MAKERS[fam](rng, held)
        span = needs.pop("robots", None) or robots
        count = None if span is None else draw_count(rng, span[0], span[1], counts)
        colours = needs.pop("colours", TRAIN_COLOURS)
        needs = {k: v for k, v in needs.items() if v is not None or k in ("gravity",)}
        if needs.get("groups"):
            if count is not None and count < 4 * needs["groups"][0]:
                continue  # too few robots for separate groups
            if count is not None and count < 4 * needs["groups"][1]:
                needs["groups"] = (needs["groups"][0],) * 2
            if count is None or count > 60:
                needs["layers"] = max(needs.get("layers", 1), 2)
        world, meta = scenes(rng, 1, colours=colours, robots=count, **needs)
        task = make(world, meta[0], 0)
        if task is None or (check and score_bound(task, world, 0) < SUCCESS.get(task.family, PASS_MARK)):
            continue
        worlds.append(world)
        metas.append(meta[0])
        out.append(task)
    fields = ("mass", "walls", "cheese_pos", "cheese_alive", "cheese_color", "gravity")
    return VoxelWorld(*(torch.cat([getattr(w, f) for w in worlds]) for f in fields)), metas, out


# ---------------------------------------------------------------------------------------------
# Plain JSON for every task and scene (the labelling pipeline reads these two)


def json_safe(value):
    """A JSON-serialisable copy: sets become sorted lists, tuples lists, numpy scalars numbers,
    arrays a count of their true cells ({"voxels": n}), Tasks their canonical_record."""
    if isinstance(value, Task):
        return canonical_record(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(json_safe(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return {"voxels": int(np.count_nonzero(value))}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def subgoal_sentences(task):
    """Ordered plain sub-goal sentences of a task (canonical words, as wordings.canonical)."""
    from beebot.vla.voxel import wordings as W

    if task.parts:
        return [s for part in task.parts for s in subgoal_sentences(part)]
    spec = task.spec
    plain = W.canonical(W.slots_of(task))
    if "who" in spec:
        slots = W.slots_of(task)
        if slots["intent"] != "subset":
            return [plain]
        return [W.who_phrase(slots["who"]) + ": " + W.job_phrase(slots["job"]), "everyone else: stay where you are"]
    kind = spec["kind"]
    if kind == "eat":
        order = spec["order"] or sorted(spec["eat"])
        if spec["any_of"]:
            return [plain]
        return [W.INTENTS["sub.eat"]["canonical"].format(x=c) for c in order] or [plain]
    if kind == "parallel":
        return [W.INTENTS[W.SUB[job]]["canonical"].format(x=W.plain(W.INTENTS[W.SUB[job]]["slots"]["x"], x)) for job, x in spec["jobs"]]
    return [plain]


def canonical_record(task):
    """A task as plain JSON: {"family", "kind" (task_kind), "plain" (one canonical English
    sentence, wordings.canonical), "spec" (the checker's spec made JSON-safe, arrays as voxel
    counts, with the canonical slots under "slots"), "subgoals" (ordered plain sub-goal
    sentences)}. A switch task's plain sentence joins its instructions ("A; then, on a new
    instruction: B"), its spec lists them under "parts" as {"t": first step, "record": record}."""
    from beebot.vla.voxel import wordings as W

    if task.parts:
        records = [canonical_record(part) for part in task.parts]
        spec = {"kind": "switch", "parts": [{"t": int(t), "record": r} for t, r in zip(task.spec["steps"], records)]}
        plain = "; then, on a new instruction: ".join(r["plain"] for r in records)
        return {"family": task.family, "kind": task_kind(task), "plain": plain, "spec": spec, "subgoals": subgoal_sentences(task)}
    slots = W.slots_of(task)
    spec = json_safe({k: v for k, v in task.spec.items() if not isinstance(v, np.ndarray) and k != "masks"})
    spec["slots"] = json_safe(slots)
    return {"family": task.family, "kind": task_kind(task), "plain": W.canonical(slots), "spec": spec, "subgoals": subgoal_sentences(task)}


def terrain_summary(walls):
    """The terrain of a scene from its walls (D, H, W): flat, a trench between two plateaus, or
    raised ground (hills) with the regions it covers and its highest level."""
    heights = walls.sum(axis=0)
    top = int(heights.max())
    if top == 0:
        return {"kind": "flat"}
    for along, names, runs in ((0, ("left", "right"), "back to front"), (1, ("back", "front"), "left to right")):
        if not (heights.min(axis=along) == heights.max(axis=along)).all():  # level along the trench
            continue
        profile = heights.max(axis=along)
        low = np.nonzero(profile == 0)[0]
        if len(low) and (profile[profile > 0] == top).all() and low.max() - low.min() + 1 == len(low) and low.min() > 0 and low.max() < len(profile) - 1:
            centre = (low.min() + low.max()) / 2
            where = region_of(SIZE / 2, centre) if along == 0 else region_of(centre, SIZE / 2)
            return {"kind": "trench", "runs": runs, "region": where, "width": int(len(low)), "depth": top, "plateaus": list(names)}
    raised = sorted({region_of(y, x) for y, x in np.argwhere(heights > 0)})
    return {"kind": "hills", "highest": top, "regions": raised}


def scene_summary(world, i):
    """Scene i as plain JSON (coarse: floor regions of FLOOR_REGIONS, heights in voxels):
    {"robots": n, "gravity": bool, "terrain": terrain_summary, "cheeses": [{"colour", "region",
    "at": [z, y, x], "on": "floor" | "raised ground" | "trench floor"}], "groups": [{"size",
    "region", "centre": [y, x], "height"}] (26-connected groups of at least 2 robots and a tenth
    of the swarm, largest first), "scattered": robots outside those groups, "towers": [{"height",
    "region"}] (vertical runs of 3 or more, tallest first), "on_top": robots standing on robots}."""
    occupied = world.mass[i].numpy() > 0.5
    terrain = terrain_summary(world.walls[i].numpy())
    n = int(occupied.sum())
    cheeses = []
    for k, colour in present(world, i):
        z, y, x = (int(v) for v in world.cheese_pos[i, k])
        on = "floor" if z == 0 else "raised ground"
        if terrain["kind"] == "trench" and z == 0:
            on = "trench floor"
        cheeses.append({"colour": colour, "region": region_of(y + 1, x + 1), "at": [z, y, x], "on": on})
    found = components(occupied)
    big = [g for g in found if len(g) >= max(2, int(np.ceil(BLOB_SHARE * n)))]
    group_rows = [{"size": len(g), "region": region_of(*g[:, 1:].mean(axis=0)), "centre": [round(float(v), 1) for v in g[:, 1:].mean(axis=0)], "height": int(g[:, 0].max()) + 1} for g in big]
    towers = []
    run = np.zeros(occupied.shape[1:], int)
    tops = {}
    for z in range(occupied.shape[0]):
        run = np.where(occupied[z], run + 1, 0)
        for y, x in np.argwhere(run >= 3):
            tops[(y, x)] = int(run[y, x])
    for (y, x), height in sorted(tops.items(), key=lambda item: -item[1]):
        if not any(abs(y - t["yx"][0]) <= 1 and abs(x - t["yx"][1]) <= 1 for t in towers):
            towers.append({"height": height, "region": region_of(y, x), "yx": (y, x)})
    return {
        "robots": n,
        "gravity": bool(world.gravity[i]),
        "terrain": terrain,
        "cheeses": cheeses,
        "groups": group_rows,
        "scattered": n - sum(len(g) for g in big),
        "towers": [{"height": t["height"], "region": t["region"]} for t in towers],
        "on_top": int((occupied[1:] & occupied[:-1]).sum()),
    }
