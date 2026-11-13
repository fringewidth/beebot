"""Grounding example kinds for the think-then-act voxel policy: four ways to tie the cubes in the
pictures to the cubes the policy controls (runs/handoff/grounding-formats.md). Opt-in: ThinkData
gives them only with `ground=` (think_data.ThinkData), and without it its examples are byte for byte
what they were (tests/test_think_ground.py).

    ThinkData(stores, haiku_dirs, ..., ground={"dirs": ["data/vla/ground/v0/s0"], "shares": {"goal_image": 0.05, "point": 0.05, "views": 0.05, "predict": 0.05}})

Every kind keeps think_data's plain causal layout: one user turn with the images at its one
<IMAGES> place (all of the example's images, in a fixed order, as consecutive vision blocks) and the
words after them; the reply is the usual act reply (thought, act marker, then the action slots) or a
text reply (empty thought, answer, <|im_end|>). The user stands at the scene camera; obs, codes and
every direction word are in that camera's frame (camera_frame.py), as in think_data.

Kinds (images in order; P = perspective, T = top view, both 256 x 256, 64 visual tokens each):

1. goal_image  [scene P, scene T, goal P]  act. "build what the last picture shows": the episode's end
   state rendered from another side of the arena (a different quarter turn, its own lighting and
   materials). The first thought puts the picture in words (the goal's own text in the camera's
   words). Episodes: goal-library goals (gen_simple, gen_medium, gen_complex, gen_ground when it
   exists) walked backwards by reverse.take_apart (natural starts; gen_complex edits from their A).
1. point       [scene P with marks, scene T]  act. A ring on one cube (a tap on the camera feed) and
   either an arrow to a cross where it goes ("move this one there"), or words for where it goes ("put
   this one in the front left corner", "on top of the tallest stack"). The marks follow the cube
   while it walks (a selection that tracks it). Episodes: one robot moved from c to d by the guided
   walk (Goal.start = the scene), every other robot still.
1. views       [P from the user's camera, P from another side] in either order, act. No marks; the
   instruction names the cube and its destination relative to the viewer ("my view is the second
   picture. move the top cube of the stack nearest me to the back right corner"); the first thought
   names the same cube as seen from the other camera when that is unambiguous. Same episodes as point.
1. predict     text. "moved": [P at t with letters on 3-4 cubes, P at t + k from the same camera];
   "which of the lettered cubes moved?" -> "b." / "a and c.". "where": [P at t with a ring on one
   cube and letters on 3-4 cells it could go to]; "the circled cube moves one cell up and to the left.
   which letter marks where it ends up?" -> "c.". Scored by exact letter set. Episodes: the ground
   store's and ThinkData's own demo stores.

The ground store (build): a directory with episodes.pt (episodes as compact arrays) and
episodes.jsonl (their meta). Mass frames are bit-packed; codes are kept on robot cells only.

    nice -n 10 .venv/bin/python -m beebot.vla.voxel.think_ground build --out data/vla/ground/v0/s0 --goals 40 --moves 40 --seed 0
"""

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel import think_data as T
from beebot.vla.voxel.camera_frame import CODE_TURNS, SIDES, quarter_turns, turn_codes, turn_grid
from beebot.vla.voxel.world import CUBE_MOVES, DIR_TABLE, N_CHANNELS, STAY, observe

KINDS = ("goal_image", "point", "views", "predict")
SIZE = 16
CELLS = SIZE**3
RUNG = "ground"
SOURCES = ("gen_simple", "gen_medium", "gen_complex", "gen_ground")
MIN_AREA = 10  # pixels of a cube (or a cell's would-be cube) that must show for a mark on it
MARGIN = 2  # a viewer-relative extreme ("nearest me") must beat every other stack by this many cells
LETTERS = "abcd"
RING = (235, 30, 200)  # marks: magenta with a white edge, drawn as outlines (never a filled cube)
EDGE = (255, 255, 255)
VIEWER_WORDS = re.compile(r"\b(?:me|my|i look|i'm looking|as i see|viewer|nearer|farther|further from|closer to)\b", re.I)

GOAL_ASKS = (
    "build what the last picture shows",
    "make it look like the last picture",
    "copy the last picture",
    "match the last picture",
    "can you make this look like the last photo",
    "turn this into what's in the last picture",
    "the last picture is what i want, build it",
    "make the scene match the last picture please",
)
MARK_ASKS = ("move this one there", "put the circled one where the arrow points", "this one goes there", "drag this one over to the cross", "move the one i circled to the cross", "put this one where i marked")
MARK_TOP_ASKS = ("put this one on top of that one", "stack the circled one on the one the arrow points at", "put this one up there")
WORD_ASKS = ("move this one {dest}", "put the circled one {dest}", "take this one and put it {dest}", "this one, {dest} please")
VIEW_ASKS = ("move {ref} {dest}", "put {ref} {dest}", "take {ref} and put it {dest}")
VIEW_LEADS = ("my view is the {which} picture.", "i'm at the camera of the {which} picture.", "the {which} picture is what i see.")
ORDINAL = ("first", "second")


# ---------------------------------------------------------------------------------------------
# The ground store


def pack_episode(ep, meta):
    """A reverse.Episode as compact arrays: mass frames bit-packed (T + 1, 512), codes on robot cells
    only (concatenated per step, offsets (T + 1,)), the start world's static fields, meta."""
    mass = (ep.mass > 0).reshape(ep.mass.shape[0], -1).numpy()
    codes, offsets = [], [0]
    for t in range(ep.steps):
        here = mass[t].nonzero()[0]
        codes.append(ep.actions[t].reshape(-1).numpy()[here].astype(np.int8))
        offsets.append(offsets[-1] + len(here))
    return {
        "meta": dict(meta, steps=int(ep.steps), end=ep.end, natural=bool(ep.natural), touched=int(getattr(ep, "touched", 0)), robots=int(mass[0].sum())),
        "mass": np.packbits(mass, axis=1),
        "codes": np.concatenate(codes) if codes else np.zeros(0, np.int8),
        "offsets": np.asarray(offsets, np.int64),
        "cheese_pos": ep.cheese_pos.clone(),
        "cheese_alive": ep.cheese_alive.clone(),
        "world": ep.world,
    }


class GroundEpisode:
    """One stored episode: frames, codes and observations at any step (world frame)."""

    def __init__(self, row, held):
        self.row, self.meta, self.held = row, row["meta"], held
        self.steps = int(self.meta["steps"])

    def occupancy(self, t):
        return np.unpackbits(self.row["mass"][t])[:CELLS].reshape(SIZE, SIZE, SIZE).astype(bool)

    def codes(self, t):
        """(D, H, W) long, the codes recorded on frame t (STAY off robots)."""
        out = np.full(CELLS, STAY, np.int64)
        here = np.unpackbits(self.row["mass"][t])[:CELLS].nonzero()[0]
        o = self.row["offsets"]
        out[here] = self.row["codes"][o[t] : o[t + 1]]
        return torch.from_numpy(out.reshape(SIZE, SIZE, SIZE))

    def world(self, t):
        mass = torch.from_numpy(self.occupancy(t)).float().unsqueeze(0)
        return replace(self.row["world"], mass=mass, cheese_pos=self.row["cheese_pos"][t].unsqueeze(0), cheese_alive=self.row["cheese_alive"][t].unsqueeze(0))

    def observation(self, t, channels=N_CHANNELS):
        return observe(self.world(t), channels)[0]

    def mover_at(self, t):
        """move episodes: the moving robot's cell at frame t (every other robot stays)."""
        still = self.occupancy(0).copy()
        still[tuple(self.meta["mover"])] = False
        cells = np.argwhere(self.occupancy(t) & ~still)
        return tuple(int(v) for v in cells[0]) if len(cells) == 1 else None


def load_store(directories, held_share=0.05):
    """GroundEpisodes of the store directories; held out by a stable hash of the goal they come from
    (a move episode follows its parent goal), so validation never sees a goal training saw."""
    out = []
    for directory in directories:
        for row in torch.load(Path(directory) / "episodes.pt", weights_only=False):
            key = row["meta"].get("parent") or row["meta"]["source"]
            out.append(GroundEpisode(row, T.held_out("ground", key, held_share)))
    return out


# ---------------------------------------------------------------------------------------------
# Building the store (CPU; reverse.take_apart)


def library_goals(sources=SOURCES, max_robots=40, root=None):
    """[(meta, reverse.Goal)] of the generated goal libraries with at most max_robots robots:
    natural take-apart for end states, the guided walk from A for gen_complex edits and retargets
    (tricks and A = B skipped)."""
    from beebot.vla.voxel import goal_gen_complex as GC
    from beebot.vla.voxel import goal_library as G

    root = Path(root or G.LIBRARY)
    out = []
    for name in sources:
        if not (root / f"{name}.npz").exists():
            continue
        if (root / f"{name}_start.npz").exists():
            lines, cells, starts = GC.load(name, root)
        else:
            (lines, cells), starts = G.load(name, root), None
        for i, line in enumerate(lines):
            if len(cells[i]) > max_robots or line.get("kind") == "trick" or line.get("reply"):
                continue
            if T.large_counts(line["text"]):
                continue
            meta = {"source": line["id"], "generator": line.get("generator", name), "text": line["text"], "turns": line.get("turns") or [line["text"]]}
            if starts is not None:
                start = starts[i]
                if start is not None and {tuple(c) for c in start.tolist()} == {tuple(c) for c in cells[i].tolist()}:
                    continue
                record = dict(line, cells=cells[i].tolist(), start_cells=None if start is None else start.tolist())
                out.append((meta, GC.to_reverse_goal(record)))
            else:
                out.append((meta, G.to_goal(cells[i], line["text"])))
    return out


def build_goals(pairs, rng, batch=6, max_steps=None):
    """Packed episodes of the goals (reverse.take_apart in batches), kept when every step replays and
    the start is what was asked (natural, or exactly A)."""
    from beebot.vla.voxel import reverse as R

    rows = []
    for lo in range(0, len(pairs), batch):
        chunk = pairs[lo : lo + batch]
        try:
            episodes, _ = R.take_apart([g for _, g in chunk], rng, max_steps=max_steps)
            passed = R.replay(episodes)["passed"].tolist()
        except Exception as error:  # a goal reverse cannot take: skipped, counted by the caller
            print(json.dumps({"skipped": [m["source"] for m, _ in chunk], "error": f"{type(error).__name__}: {error}"[:200]}), flush=True)
            continue
        for (meta, _), ep, ok in zip(chunk, episodes, passed):
            if ok and ep.natural and ep.steps > 0:
                rows.append(pack_episode(ep, dict(meta, kind="goal")))
    return rows


def columns(occ):
    """{(y, x): top z} of the robot columns of an occupancy (D, H, W) bool."""
    zs, ys, xs = np.nonzero(occ)
    out = {}
    for z, y, x in zip(zs, ys, xs):
        out[(int(y), int(x))] = max(out.get((int(y), int(x)), -1), int(z))
    return out


def move_goal(occ, rng, world):
    """A one-robot move in a scene at rest: (mover cell c, destination d, destination kind, reverse.Goal)
    or None. c is the top robot of a column; d a floor corner, the top of another column, or a floor
    cell 2-6 columns away; the scene with c moved to d must be at rest."""
    from beebot.vla.voxel import reverse as R

    tops = columns(occ)
    if len(tops) < 2:
        return None
    keys = list(tops)
    named = referable_columns(occ)  # stacks some camera can name in words (views); half the movers and stack tops are drawn from them
    for _ in range(12):
        pool = [k for k in keys if k in named] if named and rng.random() < 0.5 else keys
        y, x = pool[int(rng.integers(len(pool)))]
        c = (tops[(y, x)], y, x)
        kind = str(rng.choice(["corner", "top", "floor"], p=[0.25, 0.4, 0.35]))
        if kind == "corner":
            options = [(0, cy, cx) for cy in (0, SIZE - 1) for cx in (0, SIZE - 1) if not occ[:, max(cy - 1, 0) : cy + 2, max(cx - 1, 0) : cx + 2].any()]
        elif kind == "top":
            options = [(z + 1, yy, xx) for (yy, xx), z in tops.items() if (yy, xx) != (y, x) and z + 1 < SIZE and max(abs(yy - y), abs(xx - x)) >= 2]
            if rng.random() < 0.5 and any((o[1], o[2]) in named for o in options):
                options = [o for o in options if (o[1], o[2]) in named]
        else:
            options = [(0, yy, xx) for yy in range(SIZE) for xx in range(SIZE) if 2 <= max(abs(yy - y), abs(xx - x)) <= 6 and not occ[0:2, max(yy - 1, 0) : yy + 2, max(xx - 1, 0) : xx + 2].any()]
        if not options:
            continue
        d = options[int(rng.integers(len(options)))]
        after = occ.copy()
        after[c] = False
        after[d] = True
        b_world = replace(world, mass=torch.from_numpy(after).float().unsqueeze(0))
        if not bool(R.fixed_point(b_world)[0]):
            continue
        target = torch.zeros(SIZE, SIZE, SIZE, dtype=torch.bool)
        target[d] = True
        goal = R.Goal(family="move", text="", world=b_world, target=target, start=torch.from_numpy(occ.copy()))
        return c, d, kind, goal
    return None


def build_moves(parents, n, rng, batch=8, max_steps=60):
    """Packed one-robot move episodes from the frames of parent episodes (their natural starts and
    their goals): kept when the walk reaches the scene with only that robot moved, every step replays
    and no other robot moved."""
    from beebot.vla.voxel import reverse as R

    rows, tries = [], 0
    while len(rows) < n and tries < 6 * n:
        chunk = []
        while len(chunk) < batch and tries < 6 * n:
            tries += 1
            parent = parents[int(rng.integers(len(parents)))]
            t = 0 if rng.random() < 0.5 else parent.steps
            world = parent.world(t)
            picked = move_goal(parent.occupancy(t), rng, world)
            if picked is not None:
                c, d, kind, goal = picked
                chunk.append(({"source": f"{parent.meta['source']}@{t}:{c}>{d}", "parent": parent.meta["source"], "generator": "move", "text": "", "mover": list(c), "dest": list(d), "dest_kind": kind, "frame": int(t)}, goal))
        if not chunk:
            break
        try:
            episodes, _ = R.take_apart([g for _, g in chunk], rng, max_steps=max_steps)
            passed = R.replay(episodes)["passed"].tolist()
        except Exception as error:
            print(json.dumps({"skipped_moves": len(chunk), "error": f"{type(error).__name__}: {error}"[:200]}), flush=True)
            continue
        for (meta, _), ep, ok in zip(chunk, episodes, passed):
            if ok and ep.natural and ep.end == "reached" and ep.steps > 0 and int(getattr(ep, "touched", 1)) == 1:
                rows.append(pack_episode(ep, dict(meta, kind="move")))
    return rows[:n]


def build(out, goals=40, moves=40, seed=0, max_robots=40, sources=SOURCES, root=None, max_steps=None):
    """Write a ground store: `goals` goal episodes drawn uniformly over the sources' goals, then
    `moves` one-robot moves from their frames. Returns the counts."""
    import time

    rng = np.random.default_rng(seed)
    t0 = time.time()
    by_source = {}
    for meta, goal in library_goals(sources, max_robots, root):
        by_source.setdefault(meta["source"].split("-")[0], []).append((meta, goal))
    names = sorted(by_source)
    pairs = []
    for k in range(goals):
        pool = by_source[names[k % len(names)]]
        pairs.append(pool[int(rng.integers(len(pool)))])
    rows = build_goals(pairs, rng, max_steps=max_steps)
    parents = [GroundEpisode(r, False) for r in rows]
    move_rows = build_moves(parents, moves, rng) if parents and moves else []
    write_store(out, rows + move_rows)
    counts = {"goal_episodes": len(rows), "goals_tried": len(pairs), "move_episodes": len(move_rows), "moves_asked": moves, "seconds": round(time.time() - t0, 1), "sources": {n: sum(r["meta"]["source"].startswith(n) for r in rows) for n in names}}
    (Path(out) / "counts.json").write_text(json.dumps(counts, indent=1))
    return counts


def write_store(out, rows):
    """episodes.pt (the packed episodes) and episodes.jsonl (their meta) in directory out."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(rows, out / "episodes.pt")
    with open(out / "episodes.jsonl", "w") as handle:
        for r in rows:
            handle.write(json.dumps(r["meta"]) + "\n")
    return out


# ---------------------------------------------------------------------------------------------
# Pictures: where a cell shows in a rendered view, and marks on it


def cell_spots(obs, look, size, cells, view=0):
    """Per cell (z, y, x), where it shows in view `view` of scene_views(obs, rng, spec, look) at
    spec.size `size`: (centre x, centre y, visible pixels, radius) in output pixels, or None when
    fewer than MIN_AREA pixels show. A robot cell shows its own faces; an empty cell the faces a cube
    there would show (the render is not changed). The view's lens warp, crop and resolution apply."""
    from beebot.vla.voxel import views as V

    scene = V.scene_of(obs)
    kind = scene.kind.copy()
    for c in cells:
        if kind[c] == 0:
            kind[c] = V.KIND_ROBOT
    faces = V.voxel_faces(kind)
    if look.clutter:
        faces = V._concat([faces, V.box_faces(look.clutter)])
    camera_look = look.views[view]
    d, h, w = kind.shape
    cam = V._camera(camera_look, d, h, w, size)
    origin, direction = V.camera_rays(cam)
    face_buf, depth, _, _ = V.splat(cam, faces, origin, direction)
    n = cam.width * cam.height
    dz = np.broadcast_to(direction[2], (n,))
    oz = np.broadcast_to(origin[2], (n,))
    t_floor = np.full(n, np.inf, np.float32)
    down = dz < -1e-6
    t_floor[down] = -oz[down] / dz[down]
    seen = (face_buf >= 0) & ~(t_floor < depth)
    index = np.where(seen, face_buf, 0)
    hit_cells = faces.cell[index]
    robotish = faces.kind[index] != V.KIND_CLUTTER
    res = cam.width
    warped = res != size or camera_look.k1 != 0 or camera_look.zoom != 1 or tuple(camera_look.shift) != (0.0, 0.0)
    out = []
    for c in cells:
        mask = (seen & robotish & (hit_cells == np.asarray(c)).all(axis=1)).reshape(res, res)
        if warped and mask.any():
            rgb = np.repeat(mask[..., None].astype(np.float32), 3, axis=-1)
            mask = V.warp(rgb, mask, size, camera_look.k1, camera_look.zoom, camera_look.shift, 0.0)[0][..., 0] > 0.5
        ys, xs = np.nonzero(mask)
        if len(xs) < MIN_AREA:
            out.append(None)
            continue
        out.append((float(xs.mean()) + 0.5, float(ys.mean()) + 0.5, int(len(xs)), float(max(5.0, 0.75 * np.sqrt(len(xs))))))
    return out


def _font(size=11):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_ring(image, spot):
    from PIL import ImageDraw

    x, y, _, r = spot
    r += 3
    draw = ImageDraw.Draw(image)
    draw.ellipse([x - r - 1, y - r - 1, x + r + 1, y + r + 1], outline=EDGE, width=4)
    draw.ellipse([x - r, y - r, x + r, y + r], outline=RING, width=2)


def draw_cross(image, spot):
    from PIL import ImageDraw

    x, y, _, _ = spot
    s = 5
    draw = ImageDraw.Draw(image)
    for a, b in (((x - s, y - s), (x + s, y + s)), ((x - s, y + s), (x + s, y - s))):
        draw.line([a, b], fill=EDGE, width=5)
    for a, b in (((x - s, y - s), (x + s, y + s)), ((x - s, y + s), (x + s, y - s))):
        draw.line([a, b], fill=RING, width=2)


def draw_arrow(image, start, end):
    """An arrow from the ring round `start` to just short of `end` (a drag on the screen)."""
    from PIL import ImageDraw

    (x0, y0, _, r0), (x1, y1, _, _) = start, end
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length < r0 + 12:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    a = (x0 + ux * (r0 + 4), y0 + uy * (r0 + 4))
    b = (x1 - ux * 8, y1 - uy * 8)
    head = [b, (b[0] - ux * 7 - uy * 4, b[1] - uy * 7 + ux * 4), (b[0] - ux * 7 + uy * 4, b[1] - uy * 7 - ux * 4)]
    draw = ImageDraw.Draw(image)
    draw.line([a, b], fill=EDGE, width=5)
    draw.polygon(head, fill=EDGE)
    draw.line([a, b], fill=RING, width=2)
    draw.polygon([(p[0] + ux * -1, p[1] + uy * -1) for p in head], fill=RING)


def draw_letter(image, spot, letter, away=None):
    """A dot on the spot and its letter in a small white box joined to it by a short line: above
    right, or (away: a point such as the circled cube) on the far side from that point, so letters
    on the cells round one cube fan out instead of piling up."""
    from PIL import ImageDraw

    x, y, _, _ = spot
    ux, uy = 0.6, -0.8
    if away is not None and np.hypot(x - away[0], y - away[1]) > 1e-3:
        ux, uy = (x - away[0]) / np.hypot(x - away[0], y - away[1]), (y - away[1]) / np.hypot(x - away[0], y - away[1])
    cx, cy = x + ux * 13, y + uy * 13
    bx, by = min(max(cx - 6, 0), image.width - 12), min(max(cy - 7, 0), image.height - 14)
    draw = ImageDraw.Draw(image)
    draw.line([(x, y), (bx + 6, by + 7)], fill=(0, 0, 0), width=1)
    draw.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=(255, 220, 0), outline=(0, 0, 0))
    draw.rectangle([bx, by, bx + 11, by + 13], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((bx + 3, by), letter, fill=(0, 0, 0), font=_font(11))


# ---------------------------------------------------------------------------------------------
# Words in the camera's frame


def turn_cell(cell, turns, h=SIZE):
    """A world cell (z, y, x) in the camera's frame (camera_frame.turn_grid)."""
    z, y, x = cell
    for _ in range(turns % 4):
        y, x = x, h - 1 - y
    return z, y, x


def corner_words(cell, turns):
    """The floor corner a corner cell is in, in the camera's words ("the front left corner")."""
    _, y, x = turn_cell(cell, turns)
    return f"the {'front' if y >= SIZE // 2 else 'back'} {'left' if x < SIZE // 2 else 'right'} corner"


REFS = {  # name: (key on (y', x', top z) in the camera's frame, sign: +1 the largest wins, phrase for a stack, phrase for a single cube)
    "near": (lambda y, x, z: y, 1, "the stack nearest me", "the cube nearest me"),
    "far": (lambda y, x, z: y, -1, "the stack farthest from me", "the cube farthest from me"),
    "left": (lambda y, x, z: x, -1, "the stack furthest to my left", "the cube furthest to my left"),
    "right": (lambda y, x, z: x, 1, "the stack furthest to my right", "the cube furthest to my right"),
    "tall": (lambda y, x, z: z, 1, "the tallest stack", None),
}
OTHER_CAMERA = {"near": "nearest that camera", "far": "farthest from that camera", "left": "furthest to that camera's left", "right": "furthest to that camera's right", "tall": "tallest"}


def unique_refs(occ_cam, column):
    """The names of REFS for which `column` (y', x') is the one extreme stack of a camera-frame
    occupancy: ahead of every other stack by MARGIN cells (1 for height)."""
    tops = columns(occ_cam)
    if column not in tops:
        return []
    out = []
    for name, (key, sign, _, _) in REFS.items():
        mine = sign * key(*column, tops[column])
        others = [sign * key(y, x, z) for (y, x), z in tops.items() if (y, x) != column]
        if others and mine - max(others) >= (1 if name == "tall" else MARGIN):
            out.append(name)
    return out


def referable_columns(occ):
    """The world (y, x) of the stacks that a camera on some side of the arena can name by a
    viewer-relative extreme (unique_refs; "tallest" only for a stack of two or more)."""
    out = set()
    for (y, x), z in columns(occ).items():
        for turns in range(4):
            _, ty, tx = turn_cell((z, y, x), turns)
            if [n for n in unique_refs(turn_grid(occ, turns), (ty, tx)) if n != "tall" or z > 0]:
                out.add((y, x))
                break
    return out


def ref_phrase(name, top_z, cube=True):
    """"the top cube of the stack nearest me" (a stack), "the cube nearest me" (a lone cube); cube=False
    names the stack itself."""
    stack, single = REFS[name][2], REFS[name][3]
    if top_z == 0 and single is not None:
        return single if cube else single
    return f"the top cube of {stack}" if cube else stack


def move_words(code, turns):
    """A world move code as said in the camera's frame ("up and to the left", "toward me")."""
    dz, dy, dx = (int(v) for v in DIR_TABLE[int(CODE_TURNS[turns % 4][code])])
    flat = {(0, 1): "to the right", (0, -1): "to the left", (1, 0): "toward me", (-1, 0): "away from me"}.get((dy, dx), "")
    up = {1: "up", -1: "down"}.get(dz, "")
    return f"{up} and {flat}" if up and flat else up or flat


# ---------------------------------------------------------------------------------------------
# Examples


@dataclass
class GroundExample(T.Example):
    answer: str = ""  # text kinds: the answer to score (exact letter set)
    ground: str = ""  # the grounding kind (and variant), e.g. "point:mark"


def _example(obs, images, instruction, reply, act, codes, ep_key, t, kind, family, store, turns, camera, system, memory=(), answer=""):
    return GroundExample(
        obs=obs,
        images=images,
        context_text=T.context(instruction, memory, system),
        reply_text=reply,
        act=act,
        codes=codes,
        seed=-1,
        index=int(hashlib.sha256(ep_key.encode()).hexdigest()[:8], 16),
        e=-1,
        t=t,
        rung=RUNG,
        family=family,
        kind=f"ground:{kind.split(':')[0]}",
        instruction=instruction,
        store=store,
        turns=turns,
        camera=camera,
        answer=answer,
        ground=kind,
    )


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


class Grounding:
    """The four grounding kinds over a ground store (and, for predict, the ThinkData's own stores).

    dirs     ground store directories (build)
    shares   {kind: share of all examples}; kinds absent or 0 are never drawn
    strength views' randomization (views.DEFAULT_STRENGTH as think_train's `scene`)
    thought_share  share of act examples drawn at the first step (the one with a thought)
    """

    def __init__(self, data=None, dirs=(), shares=None, strength=None, thought_share=1 / 3, held_share=None, predict_stores=True, channels=None, system=None):
        from beebot.vla.voxel import views as V

        self.data = data
        self.shares = {k: float(v) for k, v in (shares or {}).items() if float(v) > 0}
        unknown = set(self.shares) - set(KINDS)
        if unknown:
            raise ValueError(f"unknown grounding kinds {sorted(unknown)}; kinds are {KINDS}")
        self.total = sum(self.shares.values())
        if self.total > 1:
            raise ValueError(f"grounding shares sum to {self.total} > 1")
        self.spec = V.ViewSpec(strength=V.DEFAULT_STRENGTH if strength is None else float(strength))
        self.thought_share = thought_share
        self.channels = channels or (data.channels if data is not None else N_CHANNELS)
        self.system = system if system is not None else (data.system if data is not None else None)
        held_share = held_share if held_share is not None else (data.held_share if data is not None else 0.05)
        self.episodes = load_store(dirs, held_share) if dirs else []
        self.store = ",".join(str(d) for d in dirs)
        self.predict_stores = predict_stores and data is not None
        self.goals = {False: [e for e in self.episodes if e.meta["kind"] == "goal" and not e.held], True: [e for e in self.episodes if e.meta["kind"] == "goal" and e.held]}
        self.moves = {False: [e for e in self.episodes if e.meta["kind"] == "move" and not e.held], True: [e for e in self.episodes if e.meta["kind"] == "move" and e.held]}
        self.counts = {}
        # views: the move episodes some user side can say (the cube and where it goes both nameable), and those sides
        self.sayable = {held: [(ep, sides) for ep in self.moves[held] for sides in [self._sayable_sides(ep)] if sides] for held in (False, True)}

    # -- drawing

    def maybe(self, rng, held=False):
        """A grounding example with probability self.total (one rng draw when shares are set), else None."""
        if not self.total or rng.random() >= self.total:
            return None
        names = list(self.shares)
        p = np.array([self.shares[k] for k in names]) / self.total
        kind = names[int(rng.choice(len(names), p=p))]
        try:
            return self.sample(kind, rng, held)
        except ValueError:  # no usable episode of this kind on this side of the split (e.g. few held-out views): an ordinary example instead
            self.counts[f"unavailable:{kind}"] = self.counts.get(f"unavailable:{kind}", 0) + 1
            return None

    def sample(self, kind, rng, held=False, tries=20):
        """One example of `kind` (retried over episodes when a draw cannot be said or seen)."""
        make = {"goal_image": self.goal_image, "point": self.point, "views": self.views, "predict": self.predict}[kind]
        for _ in range(tries):
            ex = make(rng, held)
            if ex is not None:
                self.counts[ex.ground] = self.counts.get(ex.ground, 0) + 1
                return ex
        raise ValueError(f"no {kind} example in {tries} draws (held={held}): too few usable episodes in {self.store}")

    def _spec(self, cameras=None, side=None, rng=None):
        from beebot.vla.voxel import views as V

        spec = self.spec if cameras is None else replace(self.spec, cameras=cameras)
        return spec if side is None else replace(spec, azimuth=V.side_azimuth(side, rng))

    def _render(self, obs, rng, spec, look=None):
        from beebot.vla.voxel import views as V

        look = V.sample_look(rng, spec) if look is None else look
        images, pose = V.scene_views(obs, rng, spec, look=look)
        return images, pose, look

    def _step(self, ep, rng):
        """A step of an act episode: the first (it carries the thought) with thought_share, else any other."""
        if ep.steps <= 1 or rng.random() < self.thought_share:
            return 0
        return int(rng.integers(1, ep.steps))

    # -- 1. goal as a picture

    def goal_image(self, rng, held):
        pool = self.goals[held]
        if not pool:
            return None
        ep = _pick(rng, pool)
        t = self._step(ep, rng)
        text = ep.meta["text"]
        sayable = T.mappable(text) and not VIEWER_WORDS.search(text)
        obs = ep.observation(t, self.channels)
        images, pose, _ = self._render(obs, rng, self._spec(side=None if sayable else "front", rng=rng))
        turns = quarter_turns(pose["azimuth"])
        other = _pick(rng, [s for k, s in enumerate(SIDES) if k != turns])
        goal_spec = self._spec(("perspective",), other, rng)
        goal_spec = replace(goal_spec, off=goal_spec.off | {"cutout"})  # a cutout could hide the part of the goal that differs
        goal_images, _, _ = self._render(ep.observation(ep.steps, self.channels), rng, goal_spec)
        ask = _pick(rng, GOAL_ASKS)
        words = T.bare(T.in_user_frame(text, turns)).lower()
        said = f"the last picture, in words: {words}" if len(words.split()) >= 4 else "the last picture shows the end state"  # "fix it" says nothing
        thought = f"you asked: {ask}. {said}."
        memory = [thought] if t > 0 else []
        return _example(turn_grid(obs, turns), images + goal_images, ask, T.act_reply(thought if t == 0 else ""), True, turn_codes(ep.codes(t), turns), ep.meta["source"], t, "goal_image", ep.meta["generator"], self.store, turns, pose["azimuth"], self.system, memory)

    # -- moves: where the mover and its destination are, in words

    def _dest_words(self, ep, occ_cam, turns):
        """Words for a move episode's destination in the camera's frame, or None: a corner, or the top
        of a stack named by a viewer-relative extreme."""
        dest = tuple(ep.meta["dest"])
        if ep.meta["dest_kind"] == "corner":
            return "into " + corner_words(dest, turns), "into " + corner_words(dest, turns)
        if ep.meta["dest_kind"] == "top":
            z, y, x = turn_cell(dest, turns)
            names = unique_refs(occ_cam, (y, x))
            if names:
                stack = ref_phrase(names[0], z - 1, cube=False)
                return f"on top of {stack}", f"on top of {stack}"
        return None

    # -- 2. pointing on the image

    def point(self, rng, held):
        pool = self.moves[held]
        if not pool:
            return None
        ep = _pick(rng, pool)
        t = self._step(ep, rng)
        obs = ep.observation(t, self.channels)
        spec = self._spec()
        images, pose, look = self._render(obs, rng, spec)
        turns = quarter_turns(pose["azimuth"])
        mover, dest = ep.mover_at(t), tuple(ep.meta["dest"])
        if mover is None:
            return None
        spots = cell_spots(obs, look, spec.size, [mover, dest])
        if spots[0] is None:
            return None
        occ_cam = turn_grid(ep.occupancy(0), turns)
        words = self._dest_words(ep, occ_cam, turns)
        use_words = words is not None and rng.random() < 0.5
        too_close = spots[1] is not None and np.hypot(spots[1][0] - spots[0][0], spots[1][1] - spots[0][1]) < spots[0][3] + 14  # a cross inside or touching the ring reads as nothing
        if not use_words and (spots[1] is None or mover == dest or too_close):
            if words is None:
                return None
            use_words = True
        image = images[0].copy()
        draw_ring(image, spots[0])
        if use_words:
            ask = _pick(rng, WORD_ASKS).format(dest=words[0])
            said = f"the circled cube goes {words[1]}"
            variant = "point:words"
        else:
            if mover != dest:
                draw_arrow(image, spots[0], spots[1])
                draw_cross(image, spots[1])
            on_top = ep.meta["dest_kind"] == "top"
            ask = _pick(rng, MARK_TOP_ASKS if on_top else MARK_ASKS)
            said = "the circled cube goes on top of the one the arrow points at" if on_top else "the circled cube goes to the cross"
            variant = "point:mark"
        thought = f"you asked: {ask}. {said}."
        return _example(turn_grid(obs, turns), [image] + images[1:], ask, T.act_reply(thought if t == 0 else ""), True, turn_codes(ep.codes(t), turns), ep.meta["source"], t, variant, "move", self.store, turns, pose["azimuth"], self.system, [thought] if t > 0 else [])

    # -- 3. the same scene from several cameras, words relative to the viewer

    def _sayable_sides(self, ep):
        """The user's quarter turns from which both the cube and where it goes can be said (views)."""
        start = ep.occupancy(0)
        out = []
        for side_turns in range(4):
            z, y, x = turn_cell(tuple(ep.meta["mover"]), side_turns)
            names = [n for n in unique_refs(turn_grid(start, side_turns), (y, x)) if n != "tall" or z > 0]
            if names and self._dest_words(ep, turn_grid(start, side_turns), side_turns) is not None:
                out.append(side_turns)
        return out

    def views(self, rng, held):
        pool = self.sayable[held]
        if not pool:
            return None
        ep, sayable = _pick(rng, pool)
        start = ep.occupancy(0)
        t = self._step(ep, rng)
        obs = ep.observation(t, self.channels)
        mine, pose, _ = self._render(obs, rng, self._spec(("perspective",), SIDES[_pick(rng, sayable)], rng))
        turns = quarter_turns(pose["azimuth"])
        other_side = _pick(rng, [s for k, s in enumerate(SIDES) if k != turns])
        theirs, other_pose, _ = self._render(obs, rng, self._spec(("perspective",), other_side, rng))
        z, y, x = turn_cell(tuple(ep.meta["mover"]), turns)
        names = [n for n in unique_refs(turn_grid(start, turns), (y, x)) if n != "tall" or z > 0]
        words = self._dest_words(ep, turn_grid(start, turns), turns)
        relative = [n for n in names if n != "tall"]  # viewer-relative words first: the tallest needs no viewpoint
        name = _pick(rng, relative or names)
        ref = ref_phrase(name, z)
        mine_first = rng.random() < 0.5
        which = ORDINAL[0 if mine_first else 1]
        ask = f"{_pick(rng, VIEW_LEADS).format(which=which)} {_pick(rng, VIEW_ASKS).format(ref=ref, dest=words[0])}"
        other_turns = quarter_turns(other_pose["azimuth"])
        oz, oy, ox = turn_cell(tuple(ep.meta["mover"]), other_turns)
        seen_there = [n for n in unique_refs(turn_grid(start, other_turns), (oy, ox)) if n != "tall"]  # the tallest is the tallest from anywhere
        there = f" in the other picture it is the {'top cube of the stack' if oz > 0 else 'cube'} {OTHER_CAMERA[seen_there[0]]}." if seen_there else ""
        thought = f"you asked: {T.bare(ask)}. that is {ref}.{there} it goes {words[1]}."
        images = mine + theirs if mine_first else theirs + mine
        return _example(turn_grid(obs, turns), images, ask, T.act_reply(thought if t == 0 else ""), True, turn_codes(ep.codes(t), turns), ep.meta["source"], t, f"views:{which}", "move", self.store, turns, pose["azimuth"], self.system, [thought] if t > 0 else [])

    # -- 4. predicting the next picture (text)

    def _logged(self, rng, held):
        """(source key, family, obs_at(t), occupancy_at(t), codes_at(t), steps) of a logged episode:
        the ground store's or (predict_stores) one of ThinkData's demo stores."""
        ground = self.goals[held] + self.moves[held]
        use_store = self.predict_stores and (not ground or rng.random() < 0.5)
        if use_store:
            pool = self.data.held_pool if held else self.data.train_pool
            ep = pool.pick(pool.episodes, rng)
            store = self.data.data[ep.store]

            def obs_at(t):
                return self.data.observation(ep, t)

            def occ_at(t):
                return np.asarray(store["mass"][ep.e, t]) > 0

            def codes_at(t):
                return torch.as_tensor(np.asarray(store["actions"][ep.e, t])).long()

            return f"{ep.seed}/{ep.index}", ep.family, obs_at, occ_at, codes_at, self.data.steps(ep)
        if not ground:
            return None
        ep = _pick(rng, ground)
        return ep.meta["source"], ep.meta["generator"], lambda t: ep.observation(t, self.channels), ep.occupancy, ep.codes, ep.steps

    def predict(self, rng, held):
        logged = self._logged(rng, held)
        if logged is None:
            return None
        key, family, obs_at, occ_at, codes_at, steps = logged
        if steps < 1:
            return None
        variant = "moved" if rng.random() < 0.5 else "where"
        t = int(rng.integers(steps))
        occ, codes = occ_at(t), codes_at(t).numpy()
        obs = obs_at(t)
        spec = self._spec(("perspective",))
        images, pose, look = self._render(obs, rng, spec)
        turns = quarter_turns(pose["azimuth"])
        robots = [tuple(int(v) for v in c) for c in np.argwhere(occ)]
        moving = [c for c in robots if codes[c] != STAY]
        if not moving:
            return None
        image = images[0].copy()
        if variant == "moved":
            k = int(rng.integers(1, 4))
            k = min(k, steps - t)
            later = occ_at(t + k)
            window = [codes_at(s).numpy() for s in range(t, t + k)]
            # a candidate either left its cell (moved at t, the cell empty at t + k) or stayed put
            # through the window (STAY on every step, still there): nothing in between
            gone = [c for c in moving if not later[c]]
            kept = [c for c in robots if later[c] and all(w[c] == STAY for w in window)]
            if not gone or not kept:
                return None
            movers = [gone[i] for i in rng.permutation(len(gone))[: int(rng.integers(1, 3))]]
            stayers = [kept[i] for i in rng.permutation(len(kept))[: int(rng.integers(2, 4))]]
            picks = movers + stayers
            spots = cell_spots(obs, look, spec.size, picks)
            chosen = [(c, s) for c, s in zip(picks, spots) if s is not None]
            chosen = _apart(chosen)
            right = [c for c, _ in chosen if c in movers]
            if not right or len(chosen) < 3:
                return None
            order = [chosen[i] for i in rng.permutation(len(chosen))][: len(LETTERS)]
            letters = {c: LETTERS[i] for i, (c, _) in enumerate(order)}
            for c, s in order:
                draw_letter(image, s, letters[c])
            after, _, _ = self._render(obs_at(t + k), rng, spec, look=look)
            answer = " and ".join(sorted(letters[c] for c, _ in order if c in movers))
            ask = "which of the lettered cubes moved between the first and the second picture?"
            return _example(turn_grid(obs, turns), [image, after[0]], ask, T.text_reply("", answer + "."), False, None, key, t, "predict:moved", family, self.store, turns, pose["azimuth"], self.system, answer=answer)
        nxt = occ_at(t + 1)
        targets = {}
        for c in moving:
            d = tuple(int(v) for v in np.asarray(c) + DIR_TABLE[int(codes[c])].numpy())
            targets.setdefault(d, []).append(c)
        clean = []
        for c in moving:
            d = tuple(int(v) for v in np.asarray(c) + DIR_TABLE[int(codes[c])].numpy())
            if all(0 <= v < SIZE for v in d) and not occ[d] and nxt[d] and not nxt[c] and len(targets[d]) == 1:
                clean.append((c, d))
        if not clean:
            return None
        c, d = _pick(rng, clean)
        others = []
        for code in range(STAY):
            if code == int(codes[c]) or not bool(CUBE_MOVES[code]):
                continue
            e = tuple(int(v) for v in np.asarray(c) + DIR_TABLE[code].numpy())
            if all(0 <= v < SIZE for v in e) and not occ[e] and e != d:
                others.append(e)
        others = [others[i] for i in rng.permutation(len(others))[:5]]
        spots = cell_spots(obs, look, spec.size, [c, d] + others)
        if spots[0] is None or spots[1] is None:
            return None
        chosen = _apart([(e, s) for e, s in zip([d] + others, spots[1:]) if s is not None], gap=8.0, away=[spots[0]], away_gap=6.0)
        if not chosen or chosen[0][0] != d or len(chosen) < 3:
            return None
        chosen = chosen[: len(LETTERS)]
        order = [chosen[i] for i in rng.permutation(len(chosen))]
        draw_ring(image, spots[0])
        letter = ""
        for i, (e, s) in enumerate(order):
            draw_letter(image, s, LETTERS[i], away=spots[0])
            letter = LETTERS[i] if e == d else letter
        ask = f"the circled cube moves one cell {move_words(int(codes[c]), turns)}. which letter marks where it ends up?"
        return _example(turn_grid(obs, turns), [image], ask, T.text_reply("", letter + "."), False, None, key, t, "predict:where", family, self.store, turns, pose["azimuth"], self.system, answer=letter)


def _apart(chosen, gap=14.0, away=(), away_gap=None):
    """Keep (cell, spot) pairs whose spots are at least gap pixels from every kept one (and away_gap
    from the points `away`), in order."""
    kept = []
    away_gap = gap if away_gap is None else away_gap
    for c, s in chosen:
        if all(np.hypot(s[0] - k[0], s[1] - k[1]) >= gap for _, k in kept) and all(np.hypot(s[0] - o[0], s[1] - o[1]) >= away_gap for o in away):
            kept.append((c, s))
    return kept


def score_text(example, reply):
    """1.0 when a text reply names exactly the example's letters (predict kinds), else 0.0."""
    want = set(re.findall(r"\b[a-d]\b", example.answer))
    said = reply.split(T.UNTHINK)[-1]
    return float(set(re.findall(r"\b[a-d]\b", said.replace(T.END, ""))) == want)


# ---------------------------------------------------------------------------------------------
# Command line


def main():
    import argparse

    parser = argparse.ArgumentParser(description="build a ground store (goal and one-robot move episodes from the goal libraries)")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--goals", type=int, default=40)
    b.add_argument("--moves", type=int, default=40)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--max-robots", type=int, default=40)
    b.add_argument("--max-steps", type=int, default=None)
    b.add_argument("--sources", nargs="*", default=list(SOURCES))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "build":
        print(json.dumps(build(args.out, args.goals, args.moves, args.seed, args.max_robots, args.sources, max_steps=args.max_steps)))


if __name__ == "__main__":
    main()
