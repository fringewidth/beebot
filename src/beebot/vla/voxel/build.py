"""The build family: "build a chair", "make an arch", "build something you can sit on". The swarm
builds a named object of the shape library (voxel/shapes.py) part by part, or any training object
that serves a named use. README "Build".

Drawn like the coverage families and never by sample_tasks or sample_coverage (sample_build, and
demos --build F with a generator of its own), so every other draw is unchanged. A task's spec:

    {"kind": "build", "object": name, "use": None or a shapes.FUNCTIONS key, "program": the parts
     program's text (turned as built), "at": [z, y, x] of its lowest corner on the grid, "cells":
     (m, 3) target voxels, "voxels": m, "subgoals": one sentence per part group, "forbid": colours}

Plan: one segment per part group in program order, {"kind": "build", "points": the voxels of this
group and every earlier one, "label": {"kind": "build", "object", "part", "index", "of"},
"steps": its time limit}. BuildExpert (the suite's Expert with build segments) sends robots to
the voxels of the segment that can be placed now: under the support rule of shapes (a lever of at
most CANTILEVER from a column to the ground) on the voxels already filled, lowest layer first.
Robots on a filled voxel hold still. As many free robots as there are such voxels take part, the
nearest first; the others stay, and those standing on the build site or next to it move off
(park). Under the pivot rules the moves are the rule's; the world's gravity is unchanged (it is
more permissive than the support rule, so whatever the rule lets stand stands).

Score (score): a named object scores the best IoU of the final robots with its target voxels over
every placement on the grid and the four quarter turns (shapes.object_iou; robots away from the
object do not count); a use scores its function check (shapes.FUNCTIONS) on the final robots.
Pass: OBJECT_PASS and FUNCTION_PASS.

    PYTHONPATH=src OMP_NUM_THREADS=4 python -m beebot.vla.voxel.build --per-object 6 --out runs/build/expert.jsonl
"""

import argparse
import json
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel import shapes as H
from beebot.vla.voxel import suite as S
from beebot.vla.voxel.controller import act
from beebot.vla.voxel.free_cube import FreeCubeExpert, distances
from beebot.vla.voxel.world import CUBE_MOVES, STAY, VoxelWorld, cheese_id_map, failure_rates, flat_index, pivot_moves, refuse, rule_moves

FAMILY = "build"
OBJECT_PASS = 0.75  # IoU a demonstration of a named object needs
FUNCTION_PASS = H.FUNCTION_PASS
USE_SHARE = 0.2  # share of build draws that name a use instead of an object
SPLITS = ("train", "test", "objects", "wording")  # objects: held-out object names (shapes.HELD_NAMES)
WORDINGS = {
    "build": ["build {o}", "make {o} out of yourselves", "put together {o}", "assemble into {o}", "construct {o} from your bodies"],
    "build.use": ["build something {f}", "make something {f}", "put together something {f}", "I need something {f}", "construct something {f} from your bodies"],
}
NEEDS = {"layout": "flat", "gravity": True, "cheeses": (0, 1)}
MARGIN = 1  # free cells kept between the build site and the grid's edge
WINDOW = 4  # open voxels at a time, the first of the placing order
JITTER = 0.25  # a crew robot the planner leaves standing takes a random allowed move with this probability (jams)
CREW = 3  # robots taking part per open voxel (the nearest), so the next ones are on their way
PARK = 2  # robots not building that stand within this many cells of the site's box leave it
STAGE_BASE, STAGE_PER_VOXEL = 12, 4  # a part group's time limit: base + per voxel (then the next group starts anyway)
FIELDS = ("mass", "walls", "cheese_pos", "cheese_alive", "cheese_color", "gravity")


def say(rng, key, split_held, **kw):
    """A template of WORDINGS[key]: the training ones, or the last when held == "wording"."""
    pool = WORDINGS[key]
    return str(rng.choice(pool[-1:] if split_held == "wording" else pool[:-1])).format(**kw)


@lru_cache(maxsize=None)
def min_sizes(draws=32):
    """{name: the fewest robots any of `draws` draws of it needs} (seed 0); inf when none builds."""
    return {name: H.min_size(name, draws) for name in H.LIBRARY}


def names_for(split, robots, use=None):
    """The object names a draw of the split may use with that many robots (and that use)."""
    pool = H.HELD_NAMES if split == "objects" else H.TRAIN_NAMES
    sizes = min_sizes()
    return [n for n in pool if sizes[n] <= robots and (use is None or use in H.LIBRARY[n].functions)]


def site(rng, world, i, extent):
    """(y, x) of a random level spot for a footprint extent (h, d, w) MARGIN cells from the edge,
    its box grown by one cell clear of walls, cheeses and their shells; None if there is none."""
    h, d, w = extent
    size = world.mass.shape[-1]
    obstacles = world.walls[i].numpy() | (S.cheese_id_map(world.index(i))[0].numpy() >= 0) | S.shells_of(world, i)
    heights = world.walls[i].numpy().sum(0)
    options = []
    for y in range(MARGIN, size - d - MARGIN + 1):
        for x in range(MARGIN, size - w - MARGIN + 1):
            if (heights[y - 1 : y + d + 1, x - 1 : x + w + 1] != 0).any():
                continue
            if obstacles[: h + 1, y - 1 : y + d + 1, x - 1 : x + w + 1].any():
                continue
            options.append((y, x))
    return None if not options else options[int(rng.integers(len(options)))]


def stage_steps(n):
    return STAGE_BASE + STAGE_PER_VOXEL * n


def plan_of(name, parts, at, rule="switch_cube"):
    """The segments of a build: one per part group, its points the voxels of the group and every
    earlier one (absolute), in the placing order of shapes.climb_order (program order of the
    groups; shapes.build_order when no cube can walk everywhere, which fit_variant never draws)."""
    group_list = H.group_cells(parts)
    order = H.climb_order([cells for _, cells in group_list], rule) or H.build_order([cells for _, cells in group_list]) or [[tuple(c) for c in cells.tolist()] for _, cells in group_list]
    segments, so_far = [], []
    for k, ((label, _), cells) in enumerate(zip(group_list, order)):
        so_far += [tuple(int(v) for v in np.array(c) + np.array(at)) for c in cells]
        segments.append({"kind": "build", "points": list(so_far), "label": {"kind": "build", "object": name, "part": label, "index": k, "of": len(group_list)}, "steps": stage_steps(len(cells))})
    return segments


def build_task(name, use, parts, at, text, split_held, forbid, plan=None):
    """A build Task (plan None: made from the parts unless the task is held out)."""
    cells = H.program_cells(parts) + np.array(at)
    spec = {"kind": "build", "object": name, "use": use, "program": H.program_text(parts), "at": [int(v) for v in at], "cells": cells, "voxels": len(cells), "subgoals": H.subgoal_sentences(parts), "forbid": set(forbid)}
    if plan is None:
        plan = plan_of(name, parts, at) if split_held in ("", "wording") else []
    return S.Task(FAMILY, text, spec, plan, split_held)


def make_build(rng, split_held, name=None, use=None, demonstrate=False):
    """(needs, make) of a build task: a named object (only `name` when given), or with USE_SHARE
    (or when `use` is given) a use any training object of the library serves. demonstrate: give
    held-out tasks a plan too (to measure the expert on them)."""
    by_use = use is not None or (name is None and rng.random() < USE_SHARE)
    split = "objects" if split_held == "objects" else "train"

    def make(world, meta, i):
        n = int(world.mass[i].sum())
        use_key = use if not by_use or use is not None else str(rng.choice(sorted(H.USES)))
        options = [name] if name is not None else names_for(split, n, use_key if by_use else None)
        options = [o for o in options if min_sizes()[o] <= n]
        if not options:
            return None
        chosen_name = str(options[int(rng.integers(len(options)))])
        shape = H.fit_variant(chosen_name, rng, n)
        if shape is None:
            return None
        parts = H.rotate(shape.parts, int(rng.integers(4)))
        cells = H.program_cells(parts)
        spot = site(rng, world, i, tuple(cells.max(axis=0) + 1))
        if spot is None:
            return None
        at = (0, *spot)
        forbid = {c for _, c in S.present(world, i)}
        text = say(rng, "build.use", split_held, f=H.USES[use_key]) if by_use else say(rng, "build", split_held, o=H.phrase(chosen_name))
        task = build_task(chosen_name, use_key if by_use else None, parts, at, text, split_held, forbid, plan_of(chosen_name, parts, at) if demonstrate else None)
        snap = S.snapshot(world, i)
        value = score(task, S.static_of(world, meta, i), snap, snap, [])[0]
        if value >= (FUNCTION_PASS if by_use else 0.5):  # done before it starts
            return None
        return task

    return dict(NEEDS), make


def sample_build(rng, n, split="train", robots=None, counts="uniform", check=True, name=None, use=None, demonstrate=False):
    """(world, metas, tasks) of n scenes with one build task each, as suite.sample_coverage draws
    its families: split "train", "test", "wording" or "objects" (held-out names); robots (low,
    high) draws each scene's robot count (counts: "uniform" or "log"), None the world's density
    range. A count no object fits is drawn again. check: also the suite's bound (score_bound)."""
    split_held = split if split in ("objects", "wording") else ""
    worlds, metas, out = [], [], []
    while len(out) < n:
        needs, make = make_build(rng, split_held, name, use, demonstrate)
        count = None if robots is None else S.draw_count(rng, robots[0], robots[1], counts)
        world, meta = S.scenes(rng, 1, robots=count, **needs)
        task = make(world, meta[0], 0)
        if task is None or (check and S.score_bound(task, world, 0) < S.PASS_MARK):
            continue
        worlds.append(world)
        metas.append(meta[0])
        out.append(task)
    return VoxelWorld(*(torch.cat([getattr(w, f) for w in worlds]) for f in FIELDS)), metas, out


# ---------------------------------------------------------------------------------------------
# Checking


def score(task, static, start, end, eaten_colours):
    """(score, unasked): a named object's best IoU over placements and quarter turns; a use's
    function check (shapes.FUNCTIONS) on the final robots."""
    spec = task.spec
    off_limits = any(c in spec["forbid"] for c in eaten_colours)
    occupancy = end["mass"] > 0.5
    if spec.get("use"):
        return float(H.FUNCTIONS[spec["use"]](occupancy, static["walls"])), off_limits
    return H.object_iou(occupancy, spec["cells"]), off_limits


def passed(task, value, off_limits):
    return not off_limits and value >= (FUNCTION_PASS if task.spec.get("use") else OBJECT_PASS)


# ---------------------------------------------------------------------------------------------
# The expert


class BuildSegments:
    """Build segments for an expert (mixed into suite.Expert or free_cube.FreeCubeExpert): when a
    segment is done, what it is after, its target voxels, and who of the swarm takes part."""

    def done(self, i, seg, world):
        if seg["kind"] == "build":
            pts = np.array(seg["points"])
            return bool((world.mass[i].numpy()[tuple(pts.T)] > 0.5).all())
        return super().done(i, seg, world)

    def describe(self, i, seg, world):
        return dict(seg["label"]) if seg["kind"] == "build" else super().describe(i, seg, world)

    def targets_of(self, i, seg, world, shells):
        """The segment's filled voxels (their robots hold) and its open ones: the first WINDOW
        empty voxels of the placing order that the support rule lets a cube take now, skipping a
        voxel while a face neighbour in its layer that comes earlier in the order is still empty
        (filling it first could leave that neighbour a slot no pivot enters)."""
        if seg["kind"] != "build":
            return super().targets_of(i, seg, world, shells)
        occupancy = world.mass[i].numpy() > 0.5
        walls = world.walls[i].numpy()
        pts = [tuple(p) for p in seg["points"]]
        full = [p for p in pts if occupancy[p]]
        slots = self.slots(pts, set(full))
        full = [p for p in full if p not in slots]
        lever = H.levers(full, walls)
        open_, pending = [], set()
        for p in pts:
            if occupancy[p]:
                continue
            z, y, x = p
            if H.placeable(p, lever, walls) is not None and not any(n in pending for n in ((z, y + 1, x), (z, y - 1, x), (z, y, x + 1), (z, y, x - 1))):
                open_.append(p)
                if len(open_) == WINDOW:
                    break
            pending.add(p)
        self.frontier[i] = (full, open_)
        return full + open_, None, None

    @staticmethod
    def slots(pts, full):
        """Filled voxels to give up: an empty voxel of the order with filled voxels on both sides
        of it in its layer is a slot no pivot can enter (a robot that started there, or one that
        came early), so the one of the two later in the order lets its robot go, and is filled
        again after the gap."""
        index = {p: k for k, p in enumerate(pts)}
        out = set()
        for p in pts:
            if p in full:
                continue
            z, y, x = p
            for dy, dx in ((0, 1), (1, 0)):
                a, b = (z, y - dy, x - dx), (z, y + dy, x + dx)
                if a in full and b in full:
                    later = a if index[a] > index[b] else b
                    if index[later] > index[p]:
                        out.add(later)
        return out

    def crew(self, world, codes, segs, move_mask, generator=None, rank=None):
        """Robots on a filled voxel hold; as many free robots as there are open voxels keep their
        move (the nearest first: by rank(i) (D, H, W) when given, else straight-line distance); the
        rest stay, and those on or beside the site park outside it."""
        counts = (world.mass > 0.5).flatten(1).sum(dim=1).tolist()
        for i, seg in enumerate(segs):
            if seg is None or seg["kind"] != "build":
                continue
            full, open_ = self.frontier[i]
            occupancy = world.mass[i].numpy() > 0.5
            cells = np.argwhere(occupancy)
            final = np.array(self.tasks[i].plan[-1]["points"])
            low, high = final.min(axis=0), final.max(axis=0)
            filled_set = set(full)
            on_target = np.array([c in filled_set for c in map(tuple, cells.tolist())], bool)
            free = np.nonzero(~on_target)[0]
            workers = set()
            if open_ and len(free):
                if rank is not None:
                    order = rank(i)[tuple(cells[free].T)].astype(float)
                else:
                    order = ((cells[free][:, None, :] - np.array(open_)[None]) ** 2).sum(-1).min(axis=1).astype(float)
                if move_mask is not None:  # a robot that cannot move now (another on top of it) is passed over
                    start = sum(counts[:i])
                    movable = move_mask[start : start + counts[i]][:, :STAY].any(dim=1).numpy()
                    order = order + 1e6 * ~movable[free]
                workers = set(free[np.argsort(order, kind="stable")[: CREW * len(open_)]].tolist())
            box = lambda c, grow: (c[:, 1] >= low[1] - grow) & (c[:, 1] <= high[1] + grow) & (c[:, 2] >= low[2] - grow) & (c[:, 2] <= high[2] + grow)  # noqa: E731
            near = box(cells, PARK)
            parkers = [k for k in free.tolist() if k not in workers and near[k]]
            idle = [k for k in free.tolist() if k not in workers and not near[k]]
            stay = cells[np.concatenate([np.nonzero(on_target)[0], np.array(idle, int)]).astype(int)]
            codes[i, stay[:, 0], stay[:, 1], stay[:, 2]] = STAY
            jam = [k for k in workers if codes[i][tuple(cells[k])] == STAY]
            if jam and move_mask is not None:  # a jam of robots in each other's swing: some try another move
                start = sum(counts[:i])
                rows = move_mask[start : start + counts[i]]
                gen = self.generator if generator is None else generator
                for k, roll in zip(jam, torch.rand(len(jam), generator=gen).tolist()):
                    options = rows[k].clone()
                    options[STAY] = False
                    if roll < JITTER and options.any():
                        codes[i][tuple(cells[k])] = int(torch.multinomial(options.float(), 1, generator=gen))
            if parkers:
                heights = world.walls[i].numpy().sum(0)
                size = occupancy.shape[-1]
                yy, xx = np.mgrid[0:size, 0:size]
                ground = np.stack([heights.ravel(), yy.ravel(), xx.ravel()], 1)
                ring = ~box(ground, PARK + 1) & box(ground, PARK + 4) & (ground[:, 0] < occupancy.shape[0])
                spots = [tuple(p) for p in ground[ring].tolist() if not occupancy[tuple(p)]]
                park = cells[parkers]
                if spots:
                    start = sum(counts[:i])
                    rows = None if move_mask is None else move_mask[start : start + counts[i]]
                    targets = torch.tensor(spots).long().unsqueeze(0)
                    park_codes = act(world.index(i), targets, torch.ones(1, len(spots), dtype=torch.bool), jitter=0.0, generator=self.generator if generator is None else generator, allowed=rows)[0]
                    codes[i, park[:, 0], park[:, 1], park[:, 2]] = park_codes[park[:, 0], park[:, 1], park[:, 2]]
                else:
                    codes[i, park[:, 0], park[:, 1], park[:, 2]] = STAY
        return codes


class BuildExpert(BuildSegments, S.Expert):
    """The suite's attractor expert with build segments: the controller moves the crew (with its
    jitter), everyone else as crew() says. For runs without a pivot rule."""

    def __init__(self, tasks, moves=None, seed=None, rule=None):
        super().__init__(tasks, moves=moves, seed=seed, rule=rule)
        self.frontier = {}  # scene -> (filled voxels, voxels placeable now)

    def adjust_build(self, world, codes, segs, move_mask, generator=None):
        return self.crew(world, codes, segs, move_mask, generator)


class BuildPlanner(BuildSegments, FreeCubeExpert):
    """free_cube's planner with build segments, for the pivot rules: every robot off a filled
    voxel follows the breadth-first distance (in pivots, through the rule's own moves) to the
    open voxels; the crew are the robots with the fewest pivots to go. No jitter."""

    def __init__(self, tasks, moves=None, seed=None, rule="switch_cube"):
        super().__init__(tasks, seed=seed, rule=rule)
        self.frontier = {}

    def __call__(self, world, t, generator=None):
        generator = self.generator if generator is None else generator
        lists, presses, _, segs = self.segments(world, t)
        targets, valid, press_cheese, press_dir = self.tensors(lists, presses)
        head = torch.zeros(valid.shape, dtype=torch.int16)
        codes = self.plan(world, targets, valid, head, press_cheese, press_dir, generator)
        occupancy = world.mass > 0.5
        b, d, h, w = occupancy.shape
        flat, inside = flat_index(targets, (d, h, w))
        target_grid = torch.zeros(b, d * h * w, dtype=torch.bool)
        target_grid.scatter_(1, flat, valid & inside)
        target_grid = target_grid.view(b, d, h, w)
        scenery = occupancy & target_grid
        cheese = cheese_id_map(world) >= 0
        paths = pivot_moves(scenery, world.walls, cheese, release=self.rule == "switch_cube")
        dist = distances(paths, target_grid & ~occupancy & ~world.walls & ~cheese, torch.zeros_like(world.mass, dtype=torch.int16), ~scenery).numpy()
        move_mask = rule_moves(world, self.rule) & CUBE_MOVES
        codes = self.crew(world, codes, segs, move_mask, generator, rank=lambda i: dist[i])
        return refuse(world, codes, self.rule, priority=torch.rand(world.mass.shape, generator=generator))


class Mixed:
    """One player for a batch that mixes build tasks with others: `other` (an expert made for the
    other tasks alone, as without builds) plays those scenes and the build expert the build
    scenes, each on its own sub-batch, so the other scenes play exactly as they would alone.
    subgoals in the batch's order."""

    def __init__(self, tasks, other, build):
        self.tasks = tasks
        self.mine = [k for k, task in enumerate(tasks) if task.family == FAMILY]
        self.rest = [k for k, task in enumerate(tasks) if task.family != FAMILY]
        self.other = other([tasks[k] for k in self.rest]) if self.rest else None
        self.build = build([tasks[k] for k in self.mine]) if self.mine else None

    @property
    def subgoals(self):
        out = [None] * len(self.tasks)
        for indices, expert in ((self.rest, self.other), (self.mine, self.build)):
            for j, k in enumerate(indices):
                out[k] = expert.subgoals[j]
        return out

    def __call__(self, world, t, generator=None):
        codes = torch.full(world.mass.shape, STAY, dtype=torch.int8)
        for indices, expert in ((self.rest, self.other), (self.mine, self.build)):
            if indices:
                index = torch.tensor(indices)
                codes[index] = expert(world.index(index), t).to(torch.int8)
        return codes


def make_expert(tasks, moves=CUBE_MOVES, seed=None, rule="switch_cube"):
    """The build expert for a rule: the planner under a pivot rule, the attractor without."""
    if rule is None:
        return BuildExpert(tasks, moves=moves, seed=seed)
    return BuildPlanner(tasks, seed=seed, rule=rule)


def episode_steps(task):
    """Steps a build episode is given: every group's time limit, so a plan runs to its end."""
    return int(sum(seg["steps"] for seg in task.plan)) if task.plan else S.CUBE_STEPS


# ---------------------------------------------------------------------------------------------
# Measuring the expert on fresh seeds, object by object


def measure(names, per_object, seed=777, rule="switch_cube", robots=S.ROBOTS, counts="log", failure=0.15, steps=None, out=None, chunk=16, log=print):
    """Rows {"object", "use", "held_out", "robots", "voxels", "steps", "score", "passed",
    "done_at"} of the expert on per_object fresh scenes per name (or per use: names "use:sit"),
    cube moves, the rule, gravity on, failures. steps None: each task its episode_steps (tasks run
    in groups of equal length); done_at: the first step the score passed (None if never)."""
    rng = np.random.default_rng(seed)
    rates = failure_rates(failure)
    sink = open(out, "a") if out else None
    rows = []
    jobs = []
    for name in names:
        use = name[4:] if name.startswith("use:") else None
        split = "objects" if name in H.HELD_NAMES else "train"
        world, metas, tasks = sample_build(rng, per_object, split, robots=robots, counts=counts, name=None if use else name, use=use, demonstrate=True)
        for i in range(per_object):
            jobs.append((world.index(i), metas[i], tasks[i]))
    by_length = {}
    for job in jobs:
        by_length.setdefault(steps or episode_steps(job[2]), []).append(job)
    for length, group in sorted(by_length.items()):
        for c in range(0, len(group), chunk):
            part = group[c : c + chunk]
            world = VoxelWorld(*(torch.cat([getattr(w, f) for w, _, _ in part]) for f in FIELDS))
            world = replace(world, gravity=torch.ones_like(world.gravity))
            metas = [m for _, m, _ in part]
            tasks = [t for _, _, t in part]
            began = time.time()
            run_seed = seed * 1_000_003 + len(rows)
            expert = make_expert(tasks, seed=run_seed, rule=rule)
            results, arrays = S.run(world, metas, tasks, expert, steps=length, seed=run_seed, record=True, failure=rates, rule=rule)
            for i, ((value, off_limits), task) in enumerate(zip(results, tasks)):
                done_at = None
                static = S.static_of(world, metas[i], i)
                for t in range(0, length + 1, 4):
                    snap = {"mass": arrays["mass"][i, t].numpy().astype(float), "cheese_pos": arrays["cheese_pos"][i, t].numpy(), "cheese_alive": arrays["cheese_alive"][i, t].numpy()}
                    if passed(task, *score(task, static, snap, snap, [])):
                        done_at = t
                        break
                row = {"object": task.spec["object"], "use": task.spec["use"], "held_out": task.spec["object"] in H.HELD_NAMES, "robots": int(world.mass[i].sum()), "voxels": int(task.spec["voxels"]), "steps": length, "score": round(float(value), 4), "passed": bool(passed(task, value, off_limits)), "done_at": done_at}
                rows.append(row)
                if sink:
                    sink.write(json.dumps(row) + "\n")
            if sink:
                sink.flush()
            log(f"{len(rows)} / {len(jobs)} episodes ({length} steps, {len(part)} scenes, {time.time() - began:.0f} s)")
    if sink:
        sink.close()
    return rows


def report(rows):
    """Markdown tables: per object (and use) mean score, pass share, voxels and steps to pass."""
    lines = ["| object | held out | n | voxels | score | passed | steps given | steps to pass (median) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    keys = sorted({("use:" + r["use"]) if r["use"] else r["object"] for r in rows}, key=lambda k: (k.startswith("use:"), k in H.HELD_NAMES, k))
    for key in keys:
        part = [r for r in rows if (("use:" + r["use"]) if r["use"] else r["object"]) == key]
        done = [r["done_at"] for r in part if r["done_at"] is not None]
        lines.append(f"| {key} | {'yes' if key in H.HELD_NAMES else ''} | {len(part)} | {np.median([r['voxels'] for r in part]):.0f} | {np.mean([r['score'] for r in part]):.2f} | {np.mean([r['passed'] for r in part]):.2f} | {np.median([r['steps'] for r in part]):.0f} | {np.median(done) if done else '-'} |")
    by_name = [r for r in rows if not r["use"]]
    lines.append("")
    for label, part in (("named, training", [r for r in by_name if not r["held_out"]]), ("named, held out", [r for r in by_name if r["held_out"]]), ("uses", [r for r in rows if r["use"]])):
        if part:
            lines.append(f"{label}: n {len(part)}, mean score {np.mean([r['score'] for r in part]):.2f}, passed {np.mean([r['passed'] for r in part]):.2f}")
    for low, high in ((0, 14), (15, 40), (41, 10**6)):
        part = [r for r in rows if low <= r["voxels"] <= high]
        done = [r["done_at"] for r in part if r["done_at"] is not None]
        if part:
            lines.append(f"voxels {low} to {min(high, 999)}: n {len(part)}, passed {np.mean([r['passed'] for r in part]):.2f}, steps given (median) {np.median([r['steps'] for r in part]):.0f}, steps to pass (median, 90th percentile) {np.median(done) if done else '-'}, {np.percentile(done, 90) if done else '-'}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-object", type=int, default=6)
    parser.add_argument("--objects", nargs="*", default=None, help="names (default: every buildable name and every use)")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--rule", default="switch_cube")
    parser.add_argument("--steps", type=int, default=None, help="one episode length for all (default: each task's episode_steps)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None, help="print the tables of an existing .jsonl")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.report:
        rows = [json.loads(line) for line in open(args.report)]
    else:
        names = args.objects or [n for n in H.LIBRARY if min_sizes()[n] <= S.ROBOTS[1]] + ["use:" + u for u in H.USES]
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        rows = measure(names, args.per_object, args.seed, args.rule, steps=args.steps, out=args.out)
    print(report(rows))


if __name__ == "__main__":
    main()
