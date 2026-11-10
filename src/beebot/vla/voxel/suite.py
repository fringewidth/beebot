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
"""

import itertools
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from beebot.vla import tasks as T2
from beebot.vla.shapes import FONT, text_mask
from beebot.vla.voxel.controller import act
from beebot.vla.voxel.facts import BLOB_SHARE, bridges, groups, shell_share, solid_grid, tallest_run
from beebot.vla.voxel.world import CHEESE_SIDE, COLOUR_NAMES, EAT_MIN_SHARE, EAT_SHARE, FRICTION, FULL_SHELL, MAX_CHEESES, STAY, VoxelWorld, _face, cheese_id_map, fail_moves, rule_moves, sample_scenes, shell, step

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


def say(rng, key, held, **kw):
    """A template of intent key: training ones (all but the last), or the last when held ==
    "wording"; eat-logic keys use the 2D pool (vla/tasks.wordings)."""
    if key.startswith("logic."):
        return T2.say(rng, key, "wording" if held == "wording" else "", **kw)
    pool = WORDINGS[key]
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


def scenes(rng, n, layout=None, gravity=None, cheeses=(0, 3), colours=TRAIN_COLOURS, require=(), robots=None):
    """(world, meta) of n scenes drawn until they match: layout ("flat", "hills", "gap" or None),
    gravity (bool or None), a cheese count in [low, high] (extra cheeses removed), colours drawn
    from `colours` with `require` among them. robots: the robot count of every scene (None: the
    world's density range)."""
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
            plan = task.plan
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
    masses, actions, positions, alive = [world.mass.to(torch.uint8)], [], [world.cheese_pos.to(torch.int8)], [world.cheese_alive]
    for t in range(steps):
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
        results.append(score(task, static, snapshot(start, i), snapshot(world, i), eaten[i]))
    if not record:
        return results, None
    arrays = {"mass": torch.stack(masses, 1), "actions": torch.stack(actions, 1), "cheese_pos": torch.stack(positions, 1), "cheese_alive": torch.stack(alive, 1)}
    return results, arrays
