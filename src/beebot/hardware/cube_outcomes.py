"""What the voxel world's pivots do in MuJoCo: outcome distributions per move class, for the grid's
step (vla/voxel/world.py, OutcomeModel) and the gap between the two (grid_mujoco_outcome_gap).

The voxel world's pivot rule (world.pivot_moves) says which moves a cube can make. Here each such
move is run in MuJoCo (cube_swarm.Swarm.moves) and judged twice. MuJoCo's own detail is
cube_swarm's: reached, stayed, off (the mover ends off the lattice: tilted, between cells, or the
scene still moving), strayed (aligned in a third cell) or knock (another cube shifted more than
DISTURBED or tipped; knock takes precedence). What it leaves on the lattice (lattice_end) is read
once the scene is at rest (settle): knock when another cube is in another cell or off the lattice,
else the mover's reached, stayed, off or strayed, and for the harmful ones every such cube's start,
end and off flag in the move's frame. The lattice end is what the grid can show and what its table
holds (fit, world.OutcomeModel). Moves are classed on the grid (world.pivot_features: primitive,
bonds, hinges, rubs, lone), so a move gets the same class in both.

Sets of moves (`measure --set`):
1. isolated: one mover in a small scene (TEMPLATES), cube.py's primitives with free neighbours.
1. layouts: single moves drawn at random by kind in cube_swarm's ten-cube layouts, as
   cube_swarm.walk draws them; a walk ends at a harmful move.
1. stores: moves from the training stores (STORES), each in the scene cropped around the mover
   (CROP columns either way, cubes not grounded within the crop dropped, crops with terrain or
   objects skipped), drawn per class in proportion to how often the stores attempt it (`draw`
   writes the class counts and the scenes).
1. together: the pivots the grid carries out at once within TOGETHER of a mover in a store step,
   run at once, and the mover run alone from the same scene (same seed, so the same friction,
   magnets, masses and placement).

Magnets (`--magnets`): "release" is cube_swarm's Swarm(release=True), the switchable magnets of
the switch_cube rule; "permanent" is Swarm(release=False).

Cube settings (`measure --setting NAME`, rows tagged with it; `compare` and `fit --setting` select
by it): --design scales cube.Cube fields (magnet pulls), --speed scales cube_swarm.RPM per
primitive, --kicks gives a kick plan per class key or primitive (a factor, or a list of factors:
escalating retries, `escalated`), --escalate one schedule for every move. `kicks` fits a
per-class plan table from store rows (kick_table). HARDWARE.md, "Cutting harm" and "Kick plans
and retries".

    python -m beebot.hardware.cube_outcomes draw --root ~/beebot --out runs/cube_outcomes/draw
    python -m beebot.hardware.cube_outcomes measure --set stores --draw runs/cube_outcomes/draw --total 600 --out runs/cube_outcomes/stores-release.jsonl
    python -m beebot.hardware.cube_outcomes fit runs/cube_outcomes/stores-release.jsonl --out src/beebot/vla/voxel/outcomes/mujoco_release.json
    python -m beebot.hardware.cube_outcomes report runs/cube_outcomes/*-release.jsonl --draw runs/cube_outcomes/draw
"""

import argparse
import json
import math
import os
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from beebot.hardware.cube_swarm import DIRECTIONS, LAYOUTS, RPM, Swarm

SIZE = 16
CROP = 2  # columns either way of the mover kept from a store scene
TOGETHER = 2  # Chebyshev distance of the other movers run with a store mover
MAGNETS = ("permanent", "release")
DETAILS = ("reached", "stayed", "off", "strayed", "knock")
STORES = tuple(f"data/vla/voxel_full/s{i}" for i in range(8)) + ("data/vla/voxel_build/s0", "data/vla/voxel_build/s1")
RULE = "switch_cube"

# One mover in a small scene: (cells (z, y, x), mover index, direction (dz, dy, dx)).
TEMPLATES = {
    "floor_roll": ([(0, 0, 0)], 0, (0, 0, 1)),
    "floor_roll_edge": ([(0, 0, 0), (0, 1, -1)], 0, (0, 0, 1)),  # away from a cube touching it by a vertical edge
    "floor_roll_slot": ([(0, 0, 0), (0, 1, 1)], 0, (0, 0, 1)),  # into the cell beside a cube, sliding down its face
    "floor_peel": ([(0, 0, 0), (0, 0, -1)], 0, (0, 0, 1)),  # away from a face neighbour
    "traverse": ([(0, 0, 0), (0, 0, 1), (1, 0, 0)], 2, (0, 0, 1)),
    "convex_up_lone": ([(0, 0, 1), (0, 0, 0)], 1, (1, 0, 1)),  # onto a single cube
    "convex_up_backed": ([(0, 0, 1), (0, 0, 2), (0, 0, 0)], 2, (1, 0, 1)),  # onto the end of a row of two
    "convex_up_row": ([(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 0)], 3, (1, 0, 1)),
    "convex_up_hang": ([(0, 0, 1), (1, 0, 1), (0, 0, 2), (1, 0, 2), (1, 0, 0)], 4, (1, 0, 1)),  # from hanging on a wall
    "convex_down_lone": ([(0, 0, 0), (1, 0, 0)], 1, (-1, 0, 1)),  # off a lone stack of two
    "convex_down_row": ([(0, 0, 0), (0, 0, -1), (1, 0, 0)], 2, (-1, 0, 1)),  # off the end of a row
    "concave_up_lone": ([(0, 0, 1), (1, 0, 1), (0, 0, 0)], 2, (1, 0, 0)),  # up a lone stack of two
    "concave_up_backed": ([(0, 0, 1), (1, 0, 1), (0, 0, 2), (1, 0, 2), (0, 0, 0)], 4, (1, 0, 0)),
    "concave_down": ([(0, 0, 1), (1, 0, 1), (0, 0, 2), (1, 0, 2), (1, 0, 0)], 4, (-1, 0, 0)),  # from hanging, back down
    "wall_crawl": ([(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 0, 2), (1, 0, 2), (0, 1, 2), (1, 1, 2), (1, 0, 0)], 8, (0, 1, 0)),
}


# ---------------------------------------------------------------------------------------------
# MuJoCo


def settle(swarm, longest=1.0):
    """Run on until the scene is still, for at most `longest` s, so that a move starts from rest
    and what it leaves is read once the scene comes to rest (Swarm.rest)."""
    swarm.rest(longest)


def carry_out(swarm, orders, rpm=None):
    """Swarm.moves on orders [(cube, direction)] from a scene at rest, as a dict: per order the
    mover's own end ("mover": reached, stayed, off, strayed or refused, before any knock is
    charged) and its Move fields; the cubes knocked with where they end; the grid before and after
    and the cubes off the lattice before and after."""
    settle(swarm)
    before = swarm.lattice()
    moves = swarm.moves(orders, rpm)
    settle(swarm)
    after = swarm.lattice()
    knocked = moves[0].knocked if moves else ()
    rows = []
    for (cube, direction), m in zip(orders, moves):
        start, end = tuple(int(v) for v in before.cells[cube]), tuple(int(v) for v in after.cells[cube])
        pivot_target = None
        mover = "refused"
        if m.primitive:
            pivot_target = swarm.plan(cube, direction, before).target
            mover = "off" if after.off[cube] else "reached" if end == pivot_target else "stayed" if end == start else "strayed"
        rows.append({"cube": int(cube), "direction": [int(v) for v in direction], "primitive": m.primitive, "bonds": int(m.bonds), "hinges": int(m.hinges), "rubs": int(m.rubs), "target": [int(v) for v in pivot_target] if pivot_target else None, "start": list(start), "end": list(end), "mover": mover, "detail": m.detail, "rpm": int(m.rpm)})
    servo = {}
    for hinge in getattr(swarm, "hinges", ()):
        from beebot.hardware.cube_hinge import servo_row

        servo[int(hinge.cube)] = servo_row(hinge)
    for row in rows:
        row.update(servo.get(row["cube"], {}))
    return {
        "orders": rows,
        "knocked": [{"cube": int(j), "start": before.cells[j].tolist(), "end": after.cells[j].tolist(), "off": bool(after.off[j])} for j in knocked],
        "shift": moves[0].shift if moves else 0.0,
        "before": before.cells.tolist(),
        "after": after.cells.tolist(),
        "off_before": [int(j) for j in np.nonzero(before.off)[0]],
        "off_after": [int(j) for j in np.nonzero(after.off)[0]],
        "sim_seconds": moves[0].sim_seconds if moves else 0.0,
        "wall_seconds": moves[0].wall_seconds if moves else 0.0,
    }


# ---------------------------------------------------------------------------------------------
# The grid's view of a scene


def grid_world(cells, size=SIZE):
    """A one-scene VoxelWorld (gravity on, no terrain or objects) with cubes at cells (z, y, x)."""
    from beebot.vla.voxel.world import world_from_arrays

    mass = np.zeros((size, size, size))
    for c in cells:
        mass[tuple(c)] = 1
    return world_from_arrays(mass, gravity=True)


def grid_moves(world, orders, rule=RULE):
    """The grid's class of each order [(cell, code)] in a one-scene world: a dict per order
    (world.pivot_features as ints, plus h, s and half: world.pivot_geometry), or None when the
    rule does not allow it."""
    from beebot.vla.voxel.world import pivot_features, pivot_geometry, rule_table

    table, side = rule_table(world, rule, sides=True)
    out = []
    for cell, code in orders:
        z, y, x = cell
        if not bool(table[0, code, z, y, x]):
            out.append(None)
            continue
        here = torch.tensor([cell])
        sides = side[0, code, z, y, x].view(1)
        features = {k: int(v[0]) for k, v in pivot_features(world, torch.tensor([0]), here, torch.tensor([code]), sides).items()}
        h, s, half = pivot_geometry(torch.tensor([code]), sides)
        h, s, half = h[0].tolist(), s[0].tolist(), bool(half[0])
        out.append({**features, "h": h, "s": s, "half": half})
    return out


def class_key(features):
    """The class of a move from its features: primitive, then the hindrances it has."""
    from beebot.vla.voxel.world import PRIMITIVES

    name = PRIMITIVES[features["primitive"]] if isinstance(features["primitive"], int) else features["primitive"]
    tags = [tag for tag, on in (("bonded", features["bonds"]), ("hinged", features["hinges"]), ("rubs", features["rubs"]), ("lone", features["lone"])) if on]
    return "/".join([name] + tags)


# ---------------------------------------------------------------------------------------------
# Trials


def trial(job):
    """One scene, run once: job = {"set", "magnets", "seed", "cells", "orders": [(cell, direction)],
    "alone": also run the first order alone from the same scene, ...}. Returns a list of rows."""
    from beebot.vla.voxel.world import DIRS

    torch.set_num_threads(1)
    cells = [tuple(c) for c in job["cells"]]
    index = {c: i for i, c in enumerate(cells)}
    world = grid_world(cells)
    orders = [(tuple(cell), DIRS.index(tuple(direction))) for cell, direction in job["orders"]]
    grid = grid_moves(world, orders)
    kept = [k for k, g in enumerate(grid) if g is not None]
    if not kept:
        return []
    runs = [kept] + ([[kept[0]]] if job.get("alone") and len(kept) > 1 else [])
    rows = []
    rpm = job.get("rpm")
    if job.get("rpm_factor") or job.get("kicks"):  # one mover at its kick (kick) times the factor
        rpm = round(kick(job, grid[kept[0]])[0])
    for run, chosen in enumerate(runs):
        swarm = make_swarm(cells, job, release=job["magnets"] == "release")
        if len(chosen) == 1 and len(kick(job, grid[chosen[0]])) > 1:
            k = chosen[0]
            rows.append(escalated(swarm, (index[orders[k][0]], job["orders"][k][1]), job, grid[k], movers=1, alone=run == 1))
            continue
        result = carry_out(swarm, [(index[orders[k][0]], job["orders"][k][1]) for k in chosen], rpm)
        for k, row in zip(chosen, result["orders"]):
            rows.append(row_of(job, grid[k], row, result, movers=len(chosen), alone=run == 1))
    return rows


def kick(job, g):
    """The flywheel rpms a move of grid class g (a grid_moves dict) is tried at, in order: the
    primitive's calibrated speed (speeds_of job["speed"]) times job["rpm_factor"] (default 1)
    times each factor of the move's kick plan. The plan is job["kicks"]'s entry for the class key,
    else for the primitive's name (a per-class kick table, firmware only): a factor (one attempt)
    or a list of factors (escalating retries, escalated); without one, job["escalate"], else 1."""
    from beebot.vla.voxel.world import PRIMITIVES

    name = PRIMITIVES[g["primitive"]] if isinstance(g["primitive"], int) else g["primitive"]
    kicks = job.get("kicks") or {}
    plan = kicks.get(class_key(g), kicks.get(name, job.get("escalate") or 1.0))
    base = speeds_of(job.get("speed") or 1.0)[name] * (job.get("rpm_factor") or 1.0)
    return [base * f for f in (plan if isinstance(plan, list) else [plan])]


def escalated(swarm, order, job, g, **extra):
    """One move with escalating retries at the rpms of its kick plan (kick): try at the first;
    while an attempt leaves the scene as it was on the lattice (the mover stayed, no cube moved or
    newly off), try again at the next. Every attempt draws its own speed and brake jitter
    (Swarm.moves) from a scene the last attempt rocked, so no two are identical. Returns one row
    for the whole move: the scene before the first attempt, after the last; tries holds each
    attempt's lattice outcome, rpms their rpms, attempts their count, schedule the plan's rpms."""
    schedule = [round(rpm) for rpm in kick(job, g)]
    first, tries, rpms, knocked, sim, wall = None, [], [], [], 0.0, 0.0
    for rpm in schedule:
        result = carry_out(swarm, [order], rpm)
        first = first or result
        row = row_of(job, g, result["orders"][0], result, **extra)
        tries.append(lattice_end(row)[0])
        rpms.append(rpm)
        knocked += result["knocked"]
        sim += result["sim_seconds"]
        wall += result["wall_seconds"]
        if tries[-1] != "stayed":
            break
    whole = {**result, "before": first["before"], "off_before": first["off_before"], "knocked": knocked, "sim_seconds": sim, "wall_seconds": wall}
    return row_of(job, g, result["orders"][0], whole, **extra, schedule=schedule, attempts=len(tries), tries=tries, rpms=rpms)


def make_swarm(cells, job, release):
    """The scene a job runs in: cube_swarm.Swarm, or with job["drive"] (cube_hinge.Drive fields)
    cube_hinge.HingeSwarm, whose cubes pivot about a latched edge under a servo (switchable
    magnets always)."""
    if job.get("drive") is not None:
        from beebot.hardware.cube_hinge import Drive, HingeSwarm

        return HingeSwarm(cells, seed=job["seed"], drive=Drive(**job["drive"]), **swarm_options(job))
    return Swarm(cells, seed=job["seed"], release=release, **swarm_options(job))


def swarm_options(job):
    """Swarm keyword arguments a job sets: cube (a design varied by job["design"], a dict of
    cube.Cube fields scaled by factors), faces and dwell (job["faces"], job["dwell"]: Swarm's),
    speeds (job["speed"]: every primitive's cube_swarm.RPM times a factor, or a dict of factors
    per primitive, 1 for the ones not named)."""
    out = {}
    if job.get("design"):
        from dataclasses import replace

        from beebot.hardware.cube_swarm import DESIGN

        base = DESIGN.cube()
        out["cube"] = replace(base, **{name: getattr(base, name) * factor for name, factor in job["design"].items()})
    if job.get("faces"):
        out["faces"] = True
    if job.get("dwell"):
        out["dwell"] = float(job["dwell"])
    if job.get("speed"):
        out["speeds"] = speeds_of(job["speed"])
    return out


def speeds_of(speed):
    """cube_swarm.RPM scaled by a factor (a number) or by factors per primitive (a dict; 1 for the
    primitives it does not name; floor_peel, a floor roll away from a bond, follows floor_roll)."""
    if not isinstance(speed, dict):
        return {name: rpm * speed for name, rpm in RPM.items()}
    return {name: rpm * speed.get(name, speed.get("floor_roll", 1.0) if name == "floor_peel" else 1.0) for name, rpm in RPM.items()}


def row_of(job, g, row, result, **extra):
    """One output row: the job's tags, the grid class and features, MuJoCo's outcome."""
    return {
        **{key: job[key] for key in ("set", "magnets", "seed", "rpm", "rpm_factor", "kicks_name", "escalate", "design", "faces", "dwell", "speed", "setting", "drive") if key in job},
        **job.get("tags", {}),
        **extra,
        "key": class_key(g),
        **{f: g[f] for f in ("primitive", "bonds", "hinges", "rubs", "lone", "h", "s", "half")},
        "plan": [row["primitive"], row["bonds"], row["hinges"], row["rubs"]],
        **{f: row[f] for f in ("cube", "direction", "detail", "start", "target", "end", "mover")},
        **{f: row[f] for f in ("servo", "miss", "peak") if f in row},
        "rpm_used": row["rpm"],
        **{f: result[f] for f in ("knocked", "shift", "off_before", "off_after", "sim_seconds", "wall_seconds")},
        "cells": result["before"],
        "after": result["after"],
    }


def isolated_jobs(trials, magnets, seed=0):
    jobs = []
    for m in magnets:
        for name, (cells, mover, direction) in TEMPLATES.items():
            shifted = [(z, y + 6, x + 6) for z, y, x in cells]
            for t in range(trials):
                jobs.append({"set": "isolated", "magnets": m, "seed": seed + 1000 * t + zlib.crc32(name.encode()) % 997, "cells": shifted, "orders": [(shifted[mover], direction)], "tags": {"template": name}})
    return jobs


def layout_walk(job):
    """cube_swarm.walk on a Bench with grid classes: each move draws a kind (grid class) among
    those the grid offers, then a cube and direction of that kind; the walk ends at a harmful move
    or after job["moves"]. Returns one row per move."""
    from beebot.vla.voxel.world import DIRS

    torch.set_num_threads(1)
    layout = job["layout"]
    cells = [(z, y + 5, x + 3) for z, y, x in LAYOUTS[layout]]
    swarm = make_swarm(cells, job, release=job["magnets"] == "release")
    rng = np.random.default_rng([job["seed"], 1])
    rows = []
    over = False
    while len(rows) < job["moves"] and not over:
        reading = swarm.lattice()
        now = [tuple(int(v) for v in c) for c in reading.cells]
        if reading.off.any() or len(set(now)) < len(now) or min(min(c) for c in now) < 0 or max(max(c) for c in now) >= SIZE:
            break
        world = grid_world(now)
        options = [(i, d) for i in range(swarm.n) for d in DIRECTIONS]
        grid = grid_moves(world, [(now[i], DIRS.index(d)) for i, d in options])
        kinds = {}
        for (i, d), g in zip(options, grid):
            if g is not None and swarm.plan(i, d, reading) is not None:
                kinds.setdefault(class_key(g), []).append((i, d, g))
        if not kinds:
            break
        keys = sorted(kinds)
        pick = kinds[keys[rng.integers(len(keys))]]
        i, d, g = pick[rng.integers(len(pick))]
        if len(kick(job, g)) > 1:
            rows.append(escalated(swarm, (i, d), {**job, "set": "layouts"}, g, layout=layout, index=len(rows), movers=1, alone=False))
        else:
            rpm = round(kick(job, g)[0]) if job.get("kicks") or job.get("rpm_factor") else job.get("rpm")
            result = carry_out(swarm, [(i, d)], rpm)
            rows.append(row_of({**job, "set": "layouts"}, g, result["orders"][0], result, layout=layout, index=len(rows), movers=1, alone=False))
        over = rows[-1]["detail"] not in ("reached", "stayed")
    return rows


# ---------------------------------------------------------------------------------------------
# The training stores: how often each class is attempted, and scenes to replay


def store_world(store, e, frames=None):
    """The VoxelWorld of episode e of a loaded store at steps `frames` (default: every step with
    an action), one scene per step, gravity on, and the action codes (steps, D, H, W)."""
    from beebot.vla.voxel.world import VoxelWorld

    actions = store["actions"][e].long()
    steps = actions.shape[0]
    frames = list(range(steps)) if frames is None else list(frames)
    mass = store["mass"][e][frames].float()
    n = len(frames)
    extra = {}
    if "form" in store:
        extra = {name: store[name][e].unsqueeze(0).expand(n, *store[name][e].shape).clone() for name in ("form", "colour2", "pattern", "friction", "food", "zones")}
    world = VoxelWorld(
        mass=mass,
        walls=store["walls"][e].unsqueeze(0).expand(n, -1, -1, -1).clone(),
        cheese_pos=store["cheese_pos"][e][frames].long(),
        cheese_alive=store["cheese_alive"][e][frames],
        cheese_color=store["cheese_color"][e].unsqueeze(0).expand(n, -1).clone(),
        gravity=torch.ones(n, dtype=torch.bool),
        **extra,
    )
    return world, actions[frames]


def store_pivots(world, actions, rule=RULE):
    """The pivots the grid attempts in a batch of scenes under recorded actions: the actions the
    rule lets through (world.refuse, no priority, as the world's step) that are not presses.
    Returns (scene, here, code, side) tensors and world.pivot_features of each."""
    from beebot.vla.voxel.world import STAY, chain_press, cheese_id_map, flat_index, pivot_features, refuse, rule_table, DIR_TABLE

    ids = cheese_id_map(world)
    codes = refuse(world, actions, rule, ids)
    robots = world.mass > 0.5
    scene, z, y, x = (robots & (codes != STAY)).nonzero(as_tuple=True)
    here = torch.stack([z, y, x], dim=1)
    code = codes[scene, z, y, x]
    if world.friction is not None:
        press = chain_press(world, codes, ids)[scene, z, y, x] >= 0
    else:
        flat, inside = flat_index(here + DIR_TABLE[code], world.mass.shape[-3:])
        press = inside & (ids.reshape(ids.shape[0], -1)[scene, flat] >= 0)
    keep = ~press
    scene, here, code = scene[keep], here[keep], code[keep]
    _, side = rule_table(world, rule, sides=True)
    sides = side[scene, code, here[:, 0], here[:, 1], here[:, 2]]
    return scene, here, code, sides, pivot_features(world, scene, here, code, sides)


def grounded_crop(cubes, blocked, centre, radius):
    """Cells (z, y, x) of the cubes in the columns within `radius` of centre (y, x) that are face
    connected to the floor through cubes of the crop, or None if terrain or an object is there."""
    _, y0, x0 = centre
    ys, xs = slice(max(0, y0 - radius), y0 + radius + 1), slice(max(0, x0 - radius), x0 + radius + 1)
    if blocked[:, ys, xs].any():
        return None
    box = cubes[:, ys, xs]
    held = box.clone()
    held[1:] = False
    held &= box
    while True:
        grown = held.clone()
        grown[1:] |= held[:-1]
        grown[:-1] |= held[1:]
        grown[:, 1:] |= held[:, :-1]
        grown[:, :-1] |= held[:, 1:]
        grown[:, :, 1:] |= held[:, :, :-1]
        grown[:, :, :-1] |= held[:, :, 1:]
        grown &= box
        if (grown == held).all():
            break
        held = grown
    z, y, x = held.nonzero(as_tuple=True)
    return [(int(a), int(b) + ys.start, int(c) + xs.start) for a, b, c in zip(z, y, x)]


def store_draw(stores=STORES, episodes=40, per_class=300, together=200, seed=0, root="."):
    """Read `episodes` episodes of each store: the class counts of every pivot the grid attempts
    (Counter by class_key), and scenes to replay in MuJoCo: up to per_class single moves per class
    (the scene cropped around the mover, grounded_crop with CROP), counted as `crops`/`tried` per
    class, and up to `together` steps with several movers (a mover and the other pivots within
    TOGETHER of it, cropped with CROP + TOGETHER)."""
    from beebot.vla.voxel import play_store
    from beebot.vla.voxel.world import DIRS, cheese_id_map

    rng = np.random.default_rng(seed)
    counts = Counter()
    candidates = {}
    groups = []
    take = math.ceil(3 * per_class / max(1, episodes * len(stores))) + 2
    for path in stores:
        store = play_store.load(Path(root) / path)
        n = len(store["actions"])
        for e in rng.choice(n, size=min(episodes, n), replace=False):
            world, actions = store_world(store, int(e))
            scene, here, code, sides, features = store_pivots(world, actions)
            if not len(scene):
                continue
            keys = [class_key({k: int(v[i]) for k, v in features.items()}) for i in range(len(scene))]
            counts.update(keys)
            blocked = world.walls | (cheese_id_map(world) >= 0)
            cubes = world.mass > 0.5
            by_key = {}
            for i in rng.permutation(len(scene)).tolist():
                if len(by_key.setdefault(keys[i], [])) < take:
                    by_key[keys[i]].append(i)
            for key, picked in by_key.items():
                for i in picked:
                    t, cell = int(scene[i]), tuple(here[i].tolist())
                    cells = grounded_crop(cubes[t], blocked[t], cell, CROP)
                    candidates.setdefault(key, {"tried": 0, "jobs": []})["tried"] += 1
                    if cells is not None and cell in cells:
                        candidates[key]["jobs"].append({"cells": cells, "orders": [(cell, DIRS[int(code[i])])], "tags": {"store": path, "episode": int(e), "t": t, "store_key": key}})
            for t in rng.choice(int(scene.max()) + 1, size=min(3, int(scene.max()) + 1), replace=False).tolist():
                at = (scene == t).nonzero().squeeze(1).tolist()
                if len(at) < 2:
                    continue
                i = at[rng.integers(len(at))]
                centre = here[i]
                near = [j for j in at if j != i and int((here[j] - centre).abs().max()) <= TOGETHER]
                cells = grounded_crop(cubes[t], blocked[t], tuple(centre.tolist()), CROP + TOGETHER) if near else None
                if cells is None or not all(tuple(here[j].tolist()) in set(cells) for j in [i] + near):
                    continue
                groups.append({"cells": cells, "orders": [(tuple(here[j].tolist()), DIRS[int(code[j])]) for j in [i] + near], "tags": {"store": path, "episode": int(e), "t": t}})
    singles = []
    crops = {}
    for key, found in candidates.items():
        jobs = found["jobs"]
        crops[key] = {"tried": found["tried"], "crops": len(jobs)}
        singles += [jobs[k] for k in rng.permutation(len(jobs))[:per_class].tolist()]
    rng.shuffle(groups)
    return counts, crops, singles, groups[:together]


# ---------------------------------------------------------------------------------------------
# What a move leaves on the lattice, the grid's table, and the gap


def frame_of(row):
    """(3, 3) int: the move's frame as columns (h, s, h x s), (z, y, x) as world.pivot_geometry."""
    h, s = np.array(row["h"]), np.array(row["s"])
    return np.stack([h, s, np.cross(h, s)], axis=1)


def lattice_end(row):
    """What a single move left on the lattice: (outcome, aftermath). outcome: knock when another
    cube is in another cell or newly off the lattice afterwards (a cube off before the move does
    not count; rows of before 2026-11-11 have no off_before), else the mover's reached, stayed,
    off or strayed. aftermath (for the harmful ones): [[from, to, stuck]] for the mover and every
    such cube, cells relative to the mover's start in the move's frame (world.OutcomeModel)."""
    before, after = np.array(row["cells"]), np.array(row["after"])
    off = set(row["off_after"]) - set(row.get("off_before", ()))
    mover = row["cube"] if "cube" in row else [list(c) for c in row["cells"]].index(list(row["start"]))
    changed = [j for j in range(len(before)) if j != mover and (j in off or tuple(after[j]) != tuple(before[j]))]
    end = tuple(after[mover])
    if changed:
        outcome = "knock"
    elif mover in off:
        outcome = "off"
    else:
        outcome = "reached" if list(end) == list(row["target"]) else "stayed" if end == tuple(before[mover]) else "strayed"
    if outcome in ("reached", "stayed"):
        return outcome, None
    frame = frame_of(row)
    start = before[mover]
    local = lambda cell: [int(v) for v in (np.array(cell) - start) @ frame]  # noqa: E731
    return outcome, [[local(before[j]), local(after[j]), j in off] for j in [mover] + changed]


LATTICE = ("reached", "stayed", "off", "strayed", "knock")


def fit(rows, name, magnets):
    """The grid's table (world.OutcomeModel) from single-mover rows: per class the reached and
    stayed counts and the harmful aftermaths, plus MuJoCo's own details for reference."""
    classes = {}
    for r in rows:
        if r["detail"] == "refused" or r.get("movers", 1) != 1:
            continue
        outcome, aftermath = lattice_end(r)
        entry = classes.setdefault(r["key"], {"reached": 0, "stayed": 0, "harm": [], "lattice": Counter(), "detail": Counter()})
        entry["lattice"][outcome] += 1
        entry["detail"][r["detail"]] += 1
        if aftermath is None:
            entry[outcome] += 1
        else:
            entry["harm"].append(aftermath)
    return {"name": name, "magnets": magnets, "rows": sum(sum(e["lattice"].values()) for e in classes.values()), "classes": dict(sorted(classes.items()))}


def legal(row, rule=RULE):
    """Whether the grid's rule allows a single-mover row's move in its scene (world.rule_table,
    which forbids a move that leaves a cube held up only by the mover: world.overhangs). The walks
    of cube_swarm drew some moves the rule now forbids; a table fit on them (fit --legal) leaves
    them out, since the grid never makes them."""
    from beebot.vla.voxel.world import DIRS, rule_table

    world = grid_world(row["cells"])
    start = tuple(row["start"])
    code = DIRS.index(tuple(a - b for a, b in zip(row["target"], start)))
    return bool(rule_table(world, rule)[0, code, start[0], start[1], start[2]])


def grid_outcomes(rows, model=None, failure=None, runs=32, seed=0):
    """The grid's lattice outcomes of the moves of single-mover rows, each played `runs` times in
    its cropped scene under rule switch_cube: with an OutcomeModel, or with failure rates
    (world.fail_moves, the stores' old way). Returns a (len(rows), len(LATTICE)) array of shares."""
    from beebot.vla.voxel.world import DIRS, STAY, concat, fail_moves, step

    generator = torch.Generator().manual_seed(seed)
    out = np.zeros((len(rows), len(LATTICE)))
    for k, r in enumerate(rows):
        world = concat([grid_world(r["cells"])] * runs)
        start, target = tuple(r["start"]), tuple(r["target"])
        codes = torch.full(world.mass.shape, STAY, dtype=torch.long)
        codes[:, start[0], start[1], start[2]] = DIRS.index(tuple(a - b for a, b in zip(target, start)))
        if failure is not None:
            codes = fail_moves(codes, failure, generator)
        noise = torch.rand(world.mass.shape, generator=generator)
        after, _ = step(world, codes, torch.rand(world.mass.shape, generator=generator), RULE, model, noise)
        for b in range(runs):
            outcome = grid_end(world.mass[b] > 0.5, after.mass[b] > 0.5, after.stuck[b] if after.stuck is not None else None, start, target)
            out[k, LATTICE.index(outcome)] += 1 / runs
    return out


def grid_end(before, now, stuck, start, target):
    """lattice_end's outcome of a grid step, from occupancy before and after (D, H, W) bool and
    the stuck robots after: knock when a cube other than the mover changed cell or is stuck."""
    vacated = (before & ~now).nonzero().tolist()
    filled = (now & ~before).nonzero().tolist()
    stuck_cells = [] if stuck is None else (stuck & now).nonzero().tolist()
    mover_only = vacated in ([], [list(start)]) and len(filled) == len(vacated)
    end = tuple(filled[0]) if filled else start
    if not mover_only or any(tuple(c) != end for c in stuck_cells):
        return "knock"
    if stuck_cells:
        return "off"
    return "reached" if end == target else "stayed" if end == start else "strayed"


def outcome_gap(rows, grid, weights):
    """grid_mujoco_outcome_gap: per class the total variation distance between MuJoCo's lattice
    outcomes of single-mover rows and the grid's on the same scenes (grid: per row the grid's
    shares, grid_outcomes), and their mean over classes weighted by how often the stores attempt
    each class (weights: class -> count). Returns (gap, {class: (tv, n, weight, mujoco shares, grid
    shares)}, share of the stores' pivots in the classes measured)."""
    per = {}
    for r, g in zip(rows, grid):
        entry = per.setdefault(r["key"], {"mujoco": np.zeros(len(LATTICE)), "grid": np.zeros(len(LATTICE)), "n": 0})
        entry["mujoco"][LATTICE.index(lattice_end(r)[0])] += 1
        entry["grid"] += g
        entry["n"] += 1
    total = sum(weights.values())
    out = {}
    for c, e in per.items():
        m, g = e["mujoco"] / e["n"], e["grid"] / e["n"]
        out[c] = (0.5 * float(np.abs(m - g).sum()), e["n"], weights.get(c, 0) / total, m, g)
    covered = sum(w for _, _, w, _, _ in out.values())
    gap = sum(tv * w for tv, _, w, _, _ in out.values()) / max(covered, 1e-12)
    return gap, out, covered


def cross_validated(rows, magnets, extra=(), folds=5, runs=24, seed=0):
    """The grid's shares on each row from a table fit on the other folds and `extra` rows
    (world.OutcomeModel)."""
    from beebot.vla.voxel.world import OutcomeModel

    fold = np.random.default_rng(seed).permutation(len(rows)) % folds
    grid = np.zeros((len(rows), len(LATTICE)))
    for f in range(folds):
        train = [r for r, k in zip(rows, fold) if k != f] + list(extra)
        held = [i for i, k in enumerate(fold) if k == f]
        model = OutcomeModel(json.loads(json.dumps(fit(train, f"fold{f}", magnets))))
        grid[held] = grid_outcomes([rows[i] for i in held], model, runs=runs, seed=seed + f)
    return grid


def noise_floor(rows, weights, draws=400, seed=0):
    """The gap a perfect grid would still show at these sample sizes: per class the mean TV
    between the class's empirical outcome shares and a resample of the same size from them,
    weighted as outcome_gap."""
    rng = np.random.default_rng(seed)
    by = {}
    for r in rows:
        by.setdefault(r["key"], []).append(LATTICE.index(lattice_end(r)[0]))
    total = sum(weights.values())
    out, covered = 0.0, 0.0
    for c, outcomes in by.items():
        p = np.bincount(outcomes, minlength=len(LATTICE)) / len(outcomes)
        tv = np.mean([0.5 * np.abs(rng.multinomial(len(outcomes), p) / len(outcomes) - p).sum() for _ in range(draws)])
        out += tv * weights.get(c, 0) / total
        covered += weights.get(c, 0) / total
    return out / max(covered, 1e-12)


# ---------------------------------------------------------------------------------------------
# Can a cube that ended off the lattice move again?

RECOVERY_AXES = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]  # (x, y, z) world axes
RECOVERY_RPM = (5000, 10000, 15000, 20000)


def recover(job):
    """Replay a trial that left its mover off the lattice (same seed: the same end), then try every
    raw spin (Swarm.spin about each world axis at each RECOVERY_RPM, released) from that end, each
    from a fresh replay. Returns a row: whether the replay ended off again, how many seconds of
    waiting leave it still off, and per spin whether the cube ended aligned in a cell with no
    other cube disturbed."""
    torch.set_num_threads(1)
    cells = [tuple(c) for c in job["cells"]]
    index = {c: i for i, c in enumerate(cells)}
    cube = index[tuple(job["start"])]
    direction = tuple(job["direction"])

    def replay():
        swarm = make_swarm(cells, job, release=True)
        swarm.moves([(cube, direction)], job.get("rpm"))
        settle(swarm)
        return swarm

    swarm = replay()
    reading = swarm.lattice()
    row = {"seed": job["seed"], "key": job["key"], "off": bool(reading.off[cube]), "cell": reading.cells[cube].tolist()}
    if not row["off"]:
        return [row]
    swarm.run(1.0)
    row["off_after_1s"] = bool(swarm.lattice().off[cube])
    tries = []
    for axis in RECOVERY_AXES:
        for rpm in RECOVERY_RPM:
            swarm = replay()
            before, _ = swarm.poses()
            after = swarm.spin(cube, axis, rpm, release=True)
            centres, _ = swarm.poses()
            others = np.arange(swarm.n) != cube
            disturbed = bool(after.off[others].any() or (np.abs(centres - before).max(axis=1)[others] > 0.1).any())
            tries.append({"axis": axis, "rpm": rpm, "aligned": not bool(after.off[cube]), "cell": after.cells[cube].tolist(), "disturbed": disturbed})
    row["tries"] = tries
    return [row]


RIGHTING_RPM = (2000, 3500, 5000, 7000, 10000)


def righting_axis(rot):
    """The flywheel axis that turns a tilted cube back upright, from its own pose (what its IMU
    reads): the world vector of the body axis, among the two that are not nearest vertical, most
    nearly along the tilt axis (the axis turning the cube's most upright body axis onto +z), signed
    to turn it that way; and the tilt in degrees. (3, 3) rot: body axes as columns."""
    up = int(np.abs(rot[2]).argmax())
    u = rot[:, up] * np.sign(rot[2, up])
    tilt = np.cross(u, [0.0, 0.0, 1.0])
    size = np.linalg.norm(tilt)
    if size < 1e-9:
        return None, 0.0
    tilt /= size
    others = [j for j in range(3) if j != up]
    j = max(others, key=lambda k: abs(rot[:, k] @ tilt))
    axis = rot[:, j] * np.sign(rot[:, j] @ tilt)
    return axis, float(np.degrees(np.arccos(np.clip(u[2], -1.0, 1.0))))


def right(job):
    """Can one cheap move right a cube a move left off the lattice? Replay the trial (same seed:
    the same end), read the cube's pose, and from fresh replays spin its flywheel about
    righting_axis, toward upright and (for comparison) away, at each RIGHTING_RPM, released. Each
    spin is judged at rest: righted when the cube ends aligned in a cell (its start cell, the
    move's target or any other) and no other cube was newly put off or moved more than DISTURBED."""
    from beebot.hardware.cube import DISTURBED

    torch.set_num_threads(1)
    cells = [tuple(c) for c in job["cells"]]
    cube = cells.index(tuple(job["start"]))
    direction = tuple(job["direction"])

    def replay():
        swarm = make_swarm(cells, job, release=True)
        carry_out(swarm, [(cube, direction)], job.get("rpm"))
        return swarm

    swarm = replay()
    reading = swarm.lattice()
    _, rot = swarm.poses()
    axis, tilt = righting_axis(rot[cube])
    row = {"seed": job["seed"], "key": job["key"], "off": bool(reading.off[cube]), "tilt": round(tilt, 1), "cell": reading.cells[cube].tolist(), "start": list(job["start"]), "target": job.get("target")}
    if not row["off"] or axis is None:
        return [row]
    tries = []
    for sense in (1, -1):
        for rpm in RIGHTING_RPM:
            swarm = replay()
            before = swarm.lattice()
            centres_before, _ = swarm.poses()
            swarm.spin(cube, tuple(sense * axis), rpm, release=True)
            swarm.rest()
            after = swarm.lattice()
            centres, _ = swarm.poses()
            others = np.arange(swarm.n) != cube
            disturbed = bool((others & after.off & ~before.off).any() or (np.abs(centres - centres_before).max(axis=1)[others] > DISTURBED).any())
            tries.append({"sense": sense, "rpm": rpm, "aligned": not bool(after.off[cube]), "cell": after.cells[cube].tolist(), "disturbed": disturbed})
    row["tries"] = tries
    return [row]


# ---------------------------------------------------------------------------------------------
# Tables


def tally(rows, by=("magnets", "set", "key")):
    """{group: Counter of details} over rows (refused moves left out)."""
    out = {}
    for r in rows:
        if r["detail"] == "refused":
            continue
        out.setdefault(tuple(r[b] for b in by), Counter())[r["detail"]] += 1
    return out


def shares(counter, names=LATTICE):
    n = sum(counter.values())
    return " ".join(f"{counter[d] / max(n, 1):5.2f}" for d in names)


def where_left(rows):
    """What harmful moves leave behind, in the move's frame (h, s, h x s) relative to the mover's
    start: Counter of (outcome, who, from, to, off) over the participants of every aftermath."""
    out = Counter()
    for r in rows:
        outcome, aftermath = lattice_end(r)
        for k, (start, end, off) in enumerate(aftermath or []):
            out[(outcome, "mover" if k == 0 else "other", tuple(start), tuple(end), bool(off))] += 1
    return out


def report(paths, draw, runs=24, seed=0, setting="old"):
    """Print the measured tables of one cube setting (setting_of): outcomes per class and set,
    permanent against release, the harmful rate per pivot (stores-weighted and ten-cube), where
    harm comes from and what it leaves, simultaneous against single movers, and
    grid_mujoco_outcome_gap before and after."""
    from beebot.vla.voxel.world import OutcomeModel, failure_rates

    rows = [json.loads(line) for p in paths for line in open(p)]
    rows = [r for r in rows if r["detail"] != "refused" and setting_of(r) == setting]
    weights = json.loads((Path(draw) / "weights.json").read_text())["counts"]
    total = sum(weights.values())
    single = [r for r in rows if r.get("movers", 1) == 1 and r["set"] in ("isolated", "layouts", "stores")]
    print("## Lattice outcomes per class (reached stayed off strayed knock), n, MuJoCo knock incl. sub-cell shoves")
    for magnets in MAGNETS:
        for name in ("isolated", "layouts", "stores"):
            part = [r for r in single if r["magnets"] == magnets and r["set"] == name]
            if not part:
                continue
            print(f"### {magnets} / {name}: {len(part)} moves")
            by = {}
            for r in part:
                by.setdefault(r.get("template") if name == "isolated" else r["key"], []).append(r)
            for key in sorted(by, key=lambda k: -len(by[k])):
                c = Counter(lattice_end(r)[0] for r in by[key])
                raw = Counter(r["detail"] for r in by[key])
                share = weights.get(key, 0) / total if name == "stores" else 0
                print(f"{key:42s} n {len(by[key]):4d}  {shares(c)}   harmful {1 - (c['reached'] + c['stayed']) / len(by[key]):5.3f}  raw knock {raw['knock'] / len(by[key]):5.2f}" + (f"  store share {share:.3f}" if share else ""))
    stores = [r for r in single if r["set"] == "stores" and r["magnets"] == "release"]
    if stores:
        by = {}
        for r in stores:
            by.setdefault(r["key"], Counter())[lattice_end(r)[0]] += 1
        covered = sum(weights.get(k, 0) for k in by) / total
        mean = Counter()
        for k, c in by.items():
            n = sum(c.values())
            for d in LATTICE:
                mean[d] += weights.get(k, 0) / total / covered * c[d] / n
        print(f"\n## Store-weighted per pivot (release, {covered:.2f} of the stores' pivots in measured classes): " + " ".join(f"{d} {mean[d]:.3f}" for d in LATTICE))
        source = Counter({k: weights.get(k, 0) / total / covered * (1 - (c["reached"] + c["stayed"]) / sum(c.values())) for k, c in by.items()})
        harm = sum(source.values())
        print(f"harmful per pivot {harm:.3f}; by class (share of all harm):")
        for k, v in source.most_common(12):
            print(f"   {k:42s} {v / harm:5.2f}")
        by_primitive, by_tag = Counter(), Counter()
        for k, v in source.items():
            by_primitive[k.split("/")[0]] += v / harm
            by_tag["lone support" if "lone" in k else "other"] += v / harm
        print("   by primitive:", {k: round(v, 2) for k, v in by_primitive.most_common()}, "; lone supports:", round(by_tag["lone support"], 2), "of the harm from", round(sum(weights.get(k, 0) for k in by if "lone" in k) / total, 3), "of the pivots")
    layouts = [r for r in single if r["set"] == "layouts" and r["magnets"] == "release"]
    if layouts:
        c = Counter(lattice_end(r)[0] for r in layouts)
        print(f"\n## Ten-cube scene (release): {len(layouts)} pivots, " + " ".join(f"{d} {c[d] / len(layouts):.3f}" for d in LATTICE))
        src = Counter(r["key"] for r in layouts if lattice_end(r)[0] not in ("reached", "stayed"))
        print("harmful by class:", dict(src.most_common(10)))
    left = where_left([r for r in stores + layouts])
    print("\n## What harm leaves (release; frame h, s, h x s from the mover's start): count outcome who from to off")
    for k, v in left.most_common(25):
        print(f"   {v:4d} {k}")
    together = [r for r in rows if r["set"] == "together"]
    if together:
        first = {}
        for r in together:
            first.setdefault((r["seed"], r["alone"]), []).append(r)
        pair = Counter()
        for (s, alone), part in first.items():
            if alone:
                continue
            solo = first.get((s, True))
            if not solo:
                continue
            pair[(part[0]["mover"], solo[0]["mover"])] += 1
        both = [r for r in together if not r["alone"]]
        knock = Counter(bool(r["knocked"]) for r in both)
        print(f"\n## Together: {len(first)} runs; first mover own end (with company, alone): {dict(pair.most_common())}")
        print(f"   movers with company: {Counter(r['mover'] for r in both)}; steps with a knock {knock[True]} of {sum(knock.values())} mover rows")
        alone = [r for r in together if r["alone"]]
        print(f"   alone: {Counter(r['mover'] for r in alone)}; knock rows {sum(bool(r['knocked']) for r in alone)} of {len(alone)}")
    if stores:
        before, per_before, cov = outcome_gap(stores, grid_outcomes(stores, failure=failure_rates(0.10, up_over=0.23), runs=runs, seed=seed), weights)
        recipes = {
            "stores": [],
            "stores + ten-cube": [r for r in single if r["magnets"] == "release" and r["set"] == "layouts"],
            "stores + ten-cube + isolated": [r for r in single if r["magnets"] == "release" and r["set"] in ("layouts", "isolated")],
        }
        scores = {}
        for name, extra in recipes.items():
            scores[name] = outcome_gap(stores, cross_validated(stores, "release", extra, runs=runs, seed=seed), weights)
        best = min(scores, key=lambda k: scores[k][0])
        after, per_after, _ = scores[best]
        full = OutcomeModel(json.loads(json.dumps(fit(stores + recipes[best], "all", "release"))))
        inside, per_inside, _ = outcome_gap(stores, grid_outcomes(stores, full, runs=runs, seed=seed), weights)
        noise = noise_floor(stores, weights)
        print(f"\n## grid_mujoco_outcome_gap (weighted TV over {cov:.2f} of the stores' pivots): before {before:.3f}; after, 5-fold held out: " + "; ".join(f"fit on {k} {v[0]:.3f}" for k, v in scores.items()) + f"; best ({best}) in sample {inside:.3f}; noise floor at these counts {noise:.3f}")
        for k in sorted(per_before, key=lambda k: -per_before[k][2])[:24]:
            tv, n, w, m, g = per_before[k]
            print(f"   {k:42s} w {w:.3f} n {n:4d} before {tv:.3f} after {per_after[k][0]:.3f} shipped {per_inside[k][0]:.3f}  mujoco {np.round(m, 2)} old grid {np.round(g, 2)} new grid {np.round(per_inside[k][4], 2)}")
        coarse = {}
        for k, (tv, n, w, m, g) in per_before.items():
            name = k.split("/")[0]
            group = "hinged or bonded" if ("hinged" in k or "bonded" in k) else "free"
            for label in (name, f"{name} ({group})", "all"):
                entry = coarse.setdefault(label, {"w": 0.0, "m": 0.0, "old": 0.0, "new": 0.0, "n": 0})
                entry["w"] += w
                entry["m"] += w * m
                entry["old"] += w * g
                entry["new"] += w * per_after[k][4]
                entry["n"] += n
        print("\n## Per primitive (store-weighted mixtures; TV of the mixtures, before and after 5-fold)")
        for label, e in sorted(coarse.items(), key=lambda kv: -kv[1]["w"]):
            m, old, new = e["m"] / e["w"], e["old"] / e["w"], e["new"] / e["w"]
            print(f"   {label:32s} w {e['w']:.3f} n {e['n']:4d} before {0.5 * np.abs(m - old).sum():.3f} after {0.5 * np.abs(m - new).sum():.3f}  mujoco {np.round(m, 2)} old {np.round(old, 2)} new {np.round(new, 2)}")


def setting_of(row):
    """The cube setting a row was measured in: its "setting" tag, "old" for the rows of before
    settings were named (release, the mover's pairs only, cube_swarm.RPM, DESIGN, judged before
    the knock and rest fixes of 2026-11-11), "" for the lever rows of before (a design or
    rpm_factor and no name)."""
    if row.get("setting"):
        return row["setting"]
    return "" if row.get("design") or row.get("rpm_factor") else "old"


def weighted(rows, weights, groups=None):
    """Store-weighted lattice outcomes of single-mover rows: per class the shares of lattice_end
    outcomes, mixed by how often the stores attempt each class (weights: class -> count), over the
    classes measured. groups: {label: predicate on a class key} for mixtures over a part of the
    classes. Returns {label: (shares (len(LATTICE),), share of the stores' pivots covered, rows)}
    with "all" for every class."""
    by = {}
    for r in rows:
        by.setdefault(r["key"], np.zeros(len(LATTICE)))[LATTICE.index(lattice_end(r)[0])] += 1
    total = sum(weights.values())
    out = {}
    for label, keep in {"all": lambda k: True, **(groups or {})}.items():
        mix, covered, n = np.zeros(len(LATTICE)), 0.0, 0
        for key, counts in by.items():
            if not keep(key):
                continue
            w = weights.get(key, 0) / total
            mix += w * counts / counts.sum()
            covered += w
            n += int(counts.sum())
        out[label] = (mix / max(covered, 1e-12), covered, n)
    return out


GROUPS = {
    "floor roll, free": lambda k: k.split("/")[0] == "floor_roll" and "bonded" not in k and "hinged" not in k,
    "floor roll, held": lambda k: k.split("/")[0] == "floor_roll" and ("bonded" in k or "hinged" in k),
    "traverse": lambda k: k.split("/")[0] == "traverse",
    "wall crawl": lambda k: k.split("/")[0] == "wall_crawl",
    "convex up": lambda k: k.split("/")[0] == "convex_up",
    "convex up, hanging": lambda k: k.split("/")[0] == "convex_up_hang",
    "convex down": lambda k: k.split("/")[0] == "convex_down",
    "concave up": lambda k: k.split("/")[0] == "concave_up",
    "concave down": lambda k: k.split("/")[0] == "concave_down",
    "lone support": lambda k: "lone" in k,
}


def compare(paths, draw):
    """Per cube setting (setting_of): store moves mixed by how often the stores attempt each class
    (harm per pivot, reached, stayed, off, strayed, knock; per primitive reached and harm) and the
    ten-cube walks (the same, unweighted)."""
    rows = [json.loads(line) for p in paths for line in open(p)]
    rows = [r for r in rows if r["detail"] != "refused" and r.get("movers", 1) == 1 and r["magnets"] == "release" and setting_of(r)]
    weights = json.loads((Path(draw) / "weights.json").read_text())["counts"]
    settings = sorted({setting_of(r) for r in rows}, key=lambda s: (s != "old", s))
    print(f"{'setting':28s} {'set':7s} {'n':>4s} {'harm':>6s} " + " ".join(f"{d:>7s}" for d in LATTICE))
    per = {}
    for s in settings:
        part = [r for r in rows if setting_of(r) == s]
        stores = [r for r in part if r["set"] == "stores"]
        if stores:
            mixes = weighted(stores, weights, GROUPS)
            per[s] = mixes
            m, covered, n = mixes["all"]
            print(f"{s:28s} {'stores':7s} {n:4d} {1 - m[0] - m[1]:6.3f} " + " ".join(f"{v:7.3f}" for v in m) + f"   (covers {covered:.2f} of the stores' pivots)")
        walks = [r for r in part if r["set"] == "layouts"]
        if walks:
            c = Counter(lattice_end(r)[0] for r in walks)
            n = len(walks)
            print(f"{'':28s} {'10-cube':7s} {n:4d} {1 - (c['reached'] + c['stayed']) / n:6.3f} " + " ".join(f"{c[d] / n:7.3f}" for d in LATTICE))
    print("\nStore-weighted per primitive: reached / harmful (n)")
    print(f"{'':20s}" + "".join(f"{s[:22]:>24s}" for s in per))
    for label in GROUPS:
        cells = []
        for s in per:
            m, _, n = per[s][label]
            cells.append(f"{m[0]:.2f} / {1 - m[0] - m[1]:.2f} ({n:3d})" if n else "-")
        print(f"{label:20s}" + "".join(f"{c:>24s}" for c in cells))
    walked = {s: [r for r in rows if setting_of(r) == s and r["set"] == "layouts"] for s in settings}
    walked = {s: part for s, part in walked.items() if part}
    if not walked:
        return
    print("\nTen-cube per primitive: reached / harmful (n)")
    print(f"{'':20s}" + "".join(f"{s[:22]:>24s}" for s in walked))
    for label, member in GROUPS.items():
        cells = []
        for part in walked.values():
            c = Counter(lattice_end(r)[0] for r in part if member(r["key"]))
            n = sum(c.values())
            cells.append(f"{c['reached'] / n:.2f} / {1 - (c['reached'] + c['stayed']) / n:.2f} ({n:3d})" if n else "-")
        print(f"{label:20s}" + "".join(f"{c:>24s}" for c in cells))


def factor_of(row):
    """The kick a single-mover row was run at, as a factor of its primitive's calibrated speed
    (cube_swarm.RPM): rpm_factor times the setting's speed factor for the primitive."""
    name = row["key"].split("/")[0]
    speed = row.get("speed") or 1.0
    per = speed if not isinstance(speed, dict) else speed.get(name, 1.0)
    return round(per * (row.get("rpm_factor") or 1.0), 3)


def plan_outcomes(row):
    """[(kick plan, lattice outcome)] a single-mover row tells: a row run at one kick gives its
    factor (factor_of) and outcome; an escalated row (escalate, the factors it was tried at) gives
    every prefix of its schedule, a move still staying after the prefix counting as stayed."""
    end = lattice_end(row)[0]
    if not row.get("escalate"):
        return [((factor_of(row),), end)]
    mult = factor_of(row)
    schedule = tuple(round(mult * f, 3) for f in row["escalate"])
    return [(schedule[:k], end if row["attempts"] <= k else "stayed") for k in range(1, len(schedule) + 1)]


def kick_table(rows, weight=5.0, prior=4.0):
    """A per-class kick table from store rows run at several kicks or escalation schedules
    (plan_outcomes) with one magnet design (each scene counted once per plan): per class the plan that maximises reached - weight *
    harmful, each class's shares shrunk toward its primitive's under the same plan by `prior`
    pseudo-moves (most classes have a handful of moves per plan). A primitive's own entry (the
    fallback for classes not seen) is its best pooled plan. A plan is a factor or, for escalating
    retries, a list of factors (kick). Returns ({key: plan}, {key: (plan, n, reached, harm)})."""
    cells, seen = {}, set()
    for r in rows:
        if r["detail"] == "refused" or r.get("movers", 1) != 1:
            continue
        key = r["key"]
        scene = (r.get("store"), r.get("episode"), r.get("t"), r["seed"], tuple(r["start"]), tuple(r["direction"]))
        for plan, end in plan_outcomes(r):
            if (scene, plan) in seen:  # a replayed scene at the same plan replays identically (an escalation's first attempt is the single kick)
                continue
            seen.add((scene, plan))
            for name in (key, key.split("/")[0]):
                c = cells.setdefault((name, plan), Counter())
                c["n"] += 1
                c["reached"] += end == "reached"
                c["harm"] += end not in ("reached", "stayed")
    table, chosen = {}, {}
    names = sorted({name for name, _ in cells}, key=lambda k: ("/" in k, k))
    for name in names:
        prim = name.split("/")[0]
        options = []
        for (other, plan), c in cells.items():
            if other != name:
                continue
            pool = cells[(prim, plan)]
            m = 0.0 if name == prim else prior
            reach = (c["reached"] + m * pool["reached"] / pool["n"]) / (c["n"] + m)
            harm = (c["harm"] + m * pool["harm"] / pool["n"]) / (c["n"] + m)
            options.append((reach - weight * harm, -len(plan), plan, c["n"], reach, harm))
        best = max(options)
        plan = best[2][0] if len(best[2]) == 1 else list(best[2])
        table[name] = plan
        chosen[name] = (plan, *best[3:])
    return table, chosen


def print_tally(rows, by=("magnets", "set", "key")):
    table = tally(rows, by)
    print(f"{' / '.join(by):60s} {'n':>5s} " + " ".join(f"{d:>8s}" for d in DETAILS))
    for group in sorted(table):
        c = table[group]
        n = sum(c.values())
        print(f"{' / '.join(str(g) for g in group):60s} {n:5d} " + " ".join(f"{c[d] / n:8.3f}" for d in DETAILS))


def run_jobs(work, jobs, out, workers):
    """Run jobs through work() in a process pool, appending every row to `out` as it lands."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    print(f"{len(jobs)} jobs", flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(workers) as pool, open(out, "a") as f:
        futures = [pool.submit(work, job) for job in jobs]
        for future in as_completed(futures):
            try:
                f.writelines(json.dumps(r) + "\n" for r in future.result())
            except Exception as error:  # one bad scene must not lose the rest
                print(f"job failed: {error!r}", flush=True)
            f.flush()
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} jobs, {time.time() - t0:.0f} s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("draw", help="class counts of the stores' pivots and scenes to replay, into a directory")
    d.add_argument("--episodes", type=int, default=40, help="episodes read per store")
    d.add_argument("--per-class", type=int, default=300)
    d.add_argument("--together", type=int, default=300)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--root", default=".", help="where the stores' data/ is")
    d.add_argument("--out", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--set", choices=("isolated", "layouts", "stores", "together"), required=True)
    m.add_argument("--magnets", nargs="+", choices=MAGNETS, default=["release"])
    m.add_argument("--trials", type=int, default=200, help="isolated: trials per template; layouts: walks per layout")
    m.add_argument("--moves", type=int, default=8, help="layouts: moves per walk")
    m.add_argument("--draw", help="stores, together: the directory `draw` wrote")
    m.add_argument("--limit", type=int, default=None, help="stores, together: at most this many scenes")
    m.add_argument("--total", type=int, default=None, help="stores: about this many scenes, each class's share by how often the stores attempt it")
    m.add_argument("--per-class", type=int, default=150, help="stores with --total: at most this many per class")
    m.add_argument("--least", type=int, default=20, help="stores with --total: at least this many per class kept")
    m.add_argument("--min-share", type=float, default=0.002, help="stores with --total: classes attempted less often are left to the coarser classes")
    m.add_argument("--design", default=None, help='cube.Cube fields scaled, as JSON: {"edge_pull": 0.8}')
    m.add_argument("--rpm-factor", type=float, default=None, help="isolated, stores: every move at its calibrated speed (cube_swarm.RPM) times this")
    m.add_argument("--faces", action="store_true", help="the neighbours face to face with a mover switch off the face that touches it (Swarm faces)")
    m.add_argument("--dwell", type=float, default=None, help="seconds the magnets are off before the flywheel spins, beyond BRAKE_AT (Swarm dwell)")
    m.add_argument("--speed", default=None, help='flywheel speeds as cube_swarm.RPM times a factor, as JSON: 0.85, or per primitive {"convex_up": 0.85}')
    m.add_argument("--kicks", default=None, help="a per-class kick table: JSON {class key or primitive: factor} or a path to one (kick)")
    m.add_argument("--escalate", type=float, nargs="+", default=None, help="escalating retries: kick factors tried in order while a move stays (escalated)")
    m.add_argument("--setting", default=None, help="a name for the cube setting, recorded on every row (compare, report and fit select by it)")
    m.add_argument("--drive", default=None, help='cube_hinge: pivot about a latched edge under a servo instead of a flywheel kick, with these Drive fields as JSON ({} for the defaults)')
    m.add_argument("--replay", default=None, help="stores: run the scenes of these stores rows again, with their seeds (paired with them)")
    m.add_argument("--templates", nargs="+", default=None, help="isolated: only these TEMPLATES")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--workers", type=int, default=5)
    m.add_argument("--out", required=True)
    c = sub.add_parser("calibrate", help="single store moves at speeds scaled from cube_swarm.RPM")
    c.add_argument("--draw", required=True)
    c.add_argument("--factors", type=float, nargs="+", default=[0.8, 1.0, 1.25, 1.5])
    c.add_argument("--per", type=int, default=40, help="scenes per primitive and held/free")
    c.add_argument("--magnets", default="release", choices=MAGNETS)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--workers", type=int, default=5)
    c.add_argument("--out", required=True)
    q = sub.add_parser("kicks", help="a per-class kick table from store rows run at several kicks (kick_table)")
    q.add_argument("paths", nargs="+")
    q.add_argument("--weight", type=float, default=5.0, help="harm penalty: maximise reached - weight * harmful")
    q.add_argument("--prior", type=float, default=4.0, help="pseudo-moves shrinking a class toward its primitive")
    q.add_argument("--settings", nargs="+", default=None, help="only rows of these settings")
    q.add_argument("--out", required=True)
    t = sub.add_parser("tally")
    t.add_argument("paths", nargs="+")
    t.add_argument("--by", nargs="+", default=["magnets", "set", "key"])
    f = sub.add_parser("fit", help="the grid's table (world.OutcomeModel) from single-mover rows")
    f.add_argument("paths", nargs="+")
    f.add_argument("--name", default="mujoco_release")
    f.add_argument("--magnets", default="release", choices=MAGNETS)
    f.add_argument("--setting", default="old", help="fit the rows of this cube setting (setting_of)")
    f.add_argument("--sets", nargs="+", default=["isolated", "layouts", "stores"], help="the sets of moves fit on")
    f.add_argument("--legal", action="store_true", help="only moves the grid's rule allows in their scene (legal: no cube left held up only by the mover)")
    f.add_argument("--out", required=True)
    k = sub.add_parser("compare", help="harm and reached per cube setting, store-weighted and ten-cube")
    k.add_argument("paths", nargs="+")
    k.add_argument("--draw", required=True)
    v = sub.add_parser("recover", help="can a cube left off the lattice move again: raw spins from replayed off ends")
    v.add_argument("paths", nargs="+", help="isolated or stores rows (one move per scene, so a replay gives the same end)")
    v.add_argument("--limit", type=int, default=40)
    v.add_argument("--workers", type=int, default=8)
    v.add_argument("--out", required=True)
    g = sub.add_parser("right", help="a cheap righting move for a cube left off the lattice: spins about the body axis that turns it upright (right)")
    g.add_argument("paths", nargs="+", help="isolated or stores rows (one move per scene, so a replay gives the same end)")
    g.add_argument("--setting", default="v1", help="off ends of this cube setting (setting_of)")
    g.add_argument("--limit", type=int, default=40)
    g.add_argument("--workers", type=int, default=6)
    g.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("paths", nargs="+")
    r.add_argument("--draw", required=True)
    r.add_argument("--runs", type=int, default=24, help="grid plays per scene for the gap")
    r.add_argument("--setting", default="old", help="report the rows of this cube setting (setting_of)")
    args = parser.parse_args()
    if args.command == "tally":
        rows = [json.loads(line) for p in args.paths for line in open(p)]
        print_tally(rows, tuple(args.by))
        return
    if args.command == "kicks":
        rows = [json.loads(line) for p in args.paths for line in open(p)]
        rows = [r for r in rows if r["set"] == "stores" and (args.settings is None or setting_of(r) in args.settings)]
        table, chosen = kick_table(rows, args.weight, args.prior)
        for name, (plan, n, reach, harm) in chosen.items():
            print(f"{name:40s} {str(plan):28s} n {n:3d}  reached {reach:.2f}  harmful {harm:.2f}")
        Path(args.out).write_text(json.dumps(table, indent=1) + "\n")
        return
    if args.command == "compare":
        compare(args.paths, args.draw)
        return
    if args.command == "fit":
        rows = [json.loads(line) for p in args.paths for line in open(p)]
        rows = [x for x in rows if x["magnets"] == args.magnets and setting_of(x) == args.setting and x["set"] in args.sets]
        dropped = 0
        if args.legal:
            kept = [x for x in rows if x["detail"] == "refused" or x.get("movers", 1) != 1 or legal(x)]
            dropped, rows = len(rows) - len(kept), kept
        table = fit(rows, args.name, args.magnets)
        if args.legal:
            table["legal"] = {"dropped": dropped}
        table["sources"] = [str(p) for p in args.paths]
        table["setting"] = args.setting
        for entry in table["classes"].values():
            entry["lattice"], entry["detail"] = dict(entry["lattice"]), dict(entry["detail"])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(table, separators=(",", ":")) + "\n")
        print(f"{table['rows']} moves in {len(table['classes'])} classes -> {args.out}")
        return
    if args.command == "report":
        torch.set_num_threads(4)
        report(args.paths, args.draw, args.runs, setting=args.setting)
        return
    if args.command in ("recover", "right"):
        rows = [json.loads(line) for p in args.paths for line in open(p)]
        offs = [x for x in rows if x["magnets"] == "release" and x["detail"] != "refused" and x.get("movers", 1) == 1 and x["set"] in ("isolated", "stores") and not x.get("rpm_factor") and lattice_end(x)[0] == "off"]
        if args.command == "right":
            offs = [x for x in offs if setting_of(x) == args.setting]
        np.random.default_rng(0).shuffle(offs)
        options = ("design", "faces", "dwell", "speed")
        jobs = [{"cells": x["cells"], "seed": x["seed"], "start": x["start"], "target": x["target"], "direction": [a - b for a, b in zip(x["target"], x["start"])], "key": x["key"], **({"rpm": x["rpm"]} if x.get("rpm") else {}), **{k: x[k] for k in options if x.get(k)}} for x in offs[: args.limit]]
        run_jobs(recover if args.command == "recover" else right, jobs, args.out, args.workers)
        return
    if args.command == "draw":
        torch.set_num_threads(4)
        counts, crops, singles, groups = store_draw(episodes=args.episodes, per_class=args.per_class, together=args.together, seed=args.seed, root=args.root)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "weights.json").write_text(json.dumps({"stores": list(STORES), "episodes": args.episodes, "seed": args.seed, "counts": dict(counts.most_common()), "crops": crops}, indent=1))
        (out / "singles.jsonl").write_text("".join(json.dumps(j) + "\n" for j in singles))
        (out / "groups.jsonl").write_text("".join(json.dumps(j) + "\n" for j in groups))
        print(f"{sum(counts.values())} pivots in {len(counts)} classes; {len(singles)} single scenes, {len(groups)} group scenes")
        return
    if args.command == "calibrate":
        from beebot.vla.voxel.world import PRIMITIVES

        rng = np.random.default_rng(args.seed)
        singles = [json.loads(line) for line in open(Path(args.draw) / "singles.jsonl")]
        pools = {}
        for job in singles:
            key = job["tags"]["store_key"]
            name = key.split("/")[0]
            pools.setdefault((name, "bonded" in key or "hinged" in key), []).append(job)
        jobs = []
        for (name, is_held), pool in sorted(pools.items()):
            for k in rng.permutation(len(pool))[: args.per].tolist():
                for factor in args.factors:
                    jobs.append({**pool[k], "set": "calibrate", "magnets": args.magnets, "seed": args.seed * 100_003 + len(jobs), "rpm": round(RPM[name] * factor), "tags": {**pool[k]["tags"], "factor": factor, "held": is_held}})
        assert all(name in RPM for name in PRIMITIVES)
        run_jobs(trial, jobs, args.out, args.workers)
        return
    design = json.loads(args.design) if args.design else None
    extra = {"design": design} if design else {}
    if args.rpm_factor:
        extra["rpm_factor"] = args.rpm_factor
    if args.faces:
        extra["faces"] = True
    if args.dwell:
        extra["dwell"] = args.dwell
    if args.speed:
        extra["speed"] = json.loads(args.speed)
    if args.kicks:
        text = Path(args.kicks).read_text() if Path(args.kicks).is_file() else args.kicks
        extra["kicks"] = json.loads(text)
        extra["kicks_name"] = Path(args.kicks).name if Path(args.kicks).is_file() else "inline"
    if args.escalate:
        extra["escalate"] = args.escalate
    if args.setting:
        extra["setting"] = args.setting
    if args.drive is not None:
        extra["drive"] = json.loads(args.drive)
    if args.set == "stores" and args.replay:
        rows = [json.loads(line) for line in open(args.replay)]
        rows = [x for x in rows if x["set"] == "stores" and x.get("movers", 1) == 1 and x["detail"] != "refused"]
        tags = ("store", "episode", "t", "store_key")
        jobs = [{"set": "stores", "magnets": x["magnets"], "seed": x["seed"], "cells": x["cells"], "orders": [(x["start"], x["direction"])], "tags": {k: x[k] for k in tags if k in x}, **extra} for x in rows[: args.limit]]
        run_jobs(trial, jobs, args.out, args.workers)
        return
    if args.set == "isolated":
        work, jobs = trial, [{**job, **extra} for job in isolated_jobs(args.trials, args.magnets, args.seed) if not args.templates or job["tags"]["template"] in args.templates]
    elif args.set == "layouts":
        work = layout_walk
        jobs = [{"magnets": mg, "layout": name, "seed": args.seed + k, "moves": args.moves, **extra} for mg in args.magnets for name in LAYOUTS for k in range(args.trials)]
    else:
        name = "singles.jsonl" if args.set == "stores" else "groups.jsonl"
        picked = [json.loads(line) for line in open(Path(args.draw) / name)]
        if args.set == "stores" and args.total:  # scenes per class in proportion to how often the stores attempt it, within --per-class
            counts = json.loads((Path(args.draw) / "weights.json").read_text())["counts"]
            total = sum(counts.values())
            share = {key: n / total for key, n in counts.items()}
            by_key = {}
            for job in picked:
                by_key.setdefault(job["tags"]["store_key"], []).append(job)
            picked = []
            for key, jobs in by_key.items():
                if share.get(key, 0) >= args.min_share:
                    picked += jobs[: min(args.per_class, max(args.least, round(args.total * share[key])))]
            np.random.default_rng(args.seed).shuffle(picked)
        picked = picked[: args.limit]
        work = trial
        jobs = [{**job, "set": args.set, "magnets": mg, "seed": args.seed * 100_003 + k, "alone": args.set == "together", **extra} for mg in args.magnets for k, job in enumerate(picked)]
    run_jobs(work, jobs, args.out, args.workers)


if __name__ == "__main__":
    main()
