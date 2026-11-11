"""The voxel grid's moves carried out by pivoting cubes in MuJoCo, closed loop.

A grid actor chooses action codes on the voxel grid as in training: rollout.ExpertActor (the
scripted expert) or rollout.PolicyActor (the think-then-act policy). Each step, in place of the
grid's world.step, the moves are made by the simulated cubes of cube_swarm.Swarm, the lattice is
read back from the physics, and that reading is the grid the actor sees next. The actor never sees
MuJoCo: a policy renders its views (views.scene_views) from the read-back grid, as in training.
rollout.play runs the episode through its `advance` hook and scores it with the suite's checkers,
so an episode here and one on the grid are scored alike.

A step (Executor):

1. What is attempted. The codes go through the grid's rule (world.refuse, switch_cube in
   training), so MuJoCo is asked only for moves the grid would let happen this step.
1. Order and claims. Movers go in the order of the step's priority scores, highest first (the
   random scores the grid's own step uses for claims, from the episode's seed). Of movers aiming at
   one cell only the first goes and the others stay, as on the grid.
1. Waves. "sequential": one mover at a time, each planned (Swarm.plan) on the lattice as the
   moves before it left it; one whose support, swing or target an earlier mover of the step
   changed is refused by the plan and stays ("blocked"). "simultaneous": the movers whose cells
   do not overlap (start, swing, target) and that do not pivot on one another go at the same
   moment (move_together); one that clashes with an earlier mover goes in a later wave of the same
   step, planned again on the lattice that wave left. A wave of one is Swarm.move itself.
1. Read back. Swarm.lattice gives each cube's nearest cell. If the scene is still moving it is
   given up to SETTLE_MAX more seconds. A cube off the lattice (tilted, or between cells) is
   counted on its nearest cell and cannot move again (the plan refuses it); with `snap` it is put
   back upright on the nearest free supported cell, as a person would, and the step logs it. Two
   cubes reading as one cell: the later one in grid order takes the nearest free cell. The grid's
   gravity is not applied: the physics decided what stands.
1. Stopping. Every step the task's checker is run on the read-back grid (Checker, as
   rollout.play scores the end of an episode). With `stop`, a scene that passes makes no more
   moves; without it the step a scene first passed and whether it still passes at the end are
   recorded, which says how often later moves undo a finished task.

MuJoCo hosts a flat steel floor and cubes, nothing else: no terrain, no cheese or other object
(hostable). Floor zones are paint and are hosted. Which task kinds that leaves is KINDS.

Swarm options (`swarm`, a dict passed to cube_swarm.Swarm) select the cube model, e.g. a
release-bonds magnet mode once cube_swarm has one; without options the cubes have permanent
magnets, under which a cube with a second face neighbour never moves (HARDWARE.md).

    from beebot.hardware.closed_loop import hostable_episodes, play_mujoco
    from beebot.vla.voxel.rollout import ExpertActor, PolicyActor
    episodes = hostable_episodes(["motion:goto"], 1, seed=0)
    played, executor = play_mujoco(episodes, ExpertActor(episodes, episodes[0].seed), record=True)
    executor.save(0, "runs/closed_loop/goto.npz"); render("runs/closed_loop/goto.npz", "goto.mp4")

    python -m beebot.hardware.closed_loop measure --per-kind 3 --workers 4 --out runs/closed_loop/sequential.jsonl
    python -m beebot.hardware.closed_loop episode --kind build --video runs/closed_loop/build.mp4
    python -m beebot.hardware.closed_loop episode --kind motion:goto --checkpoint runs/X/policy.pt --device mps
    python -m beebot.hardware.closed_loop report runs/closed_loop/sequential.jsonl
"""

import argparse
import hashlib
import json
import math
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace

import mujoco
import numpy as np
import torch

from beebot.hardware.cube import ALIGNED, BRAKE_AT, BRAKE_HOLD, DISTURBED, DT, HORIZON, LANDED, SETTLE
from beebot.hardware.cube_swarm import DESIGN, RPM, Film, Move, Reading, Swarm, swarm_xml
from beebot.vla.voxel import rollout as R
from beebot.vla.voxel import suite as S
from beebot.vla.voxel.world import DIRS, STAY, refuse, step

MODES = ("sequential", "simultaneous")
SETTLE_MAX = 1.0  # s: longest wait after a step for a scene that is still moving
RECORD_HZ = 60  # trajectory frames per simulated second
ROBOTS = (10, 12)  # demo-sized scenes: about the ten cubes of cube_swarm's scenes
HARMFUL = ("off", "strayed", "knock")


# ---------------------------------------------------------------------------------------------
# Scenes MuJoCo can host


def hostable(world, i=0):
    """Whether MuJoCo can host scene i: no terrain and no object (cheese or block); floor zones
    are paint and pass."""
    return not bool(world.walls[i].any()) and not bool(world.cheese_alive[i].any())


def cells_of(world, i=0):
    """(N, 3) int (z, y, x) of scene i's robots, in grid order."""
    return (world.mass[i] > 0.5).nonzero().numpy()


# ---------------------------------------------------------------------------------------------
# Several movers at once


def turn_cells(cell, direction, pivot):
    """(swing, leans) of a mover at `cell` (z, y, x) making `pivot` (Swarm.plan) along direction
    (dz, dy, dx): the cells the turn passes through or ends in, and those it pivots on (its
    support and, for a quarter turn, the cell it comes to rest against), all (z, y, x). h and s
    come back from the pivot: axis = h x s, so s = axis x h for a quarter turn; a half turn up
    goes up (h) over a side support (s), one down goes sideways (h) round a support below (s)."""
    here = np.array(cell[::-1])
    d = np.array(direction[::-1])
    up = np.array([0, 0, 1])
    if np.count_nonzero(d) == 1:
        h, s, half = d, np.cross(np.array(pivot.axis), d), False
    elif d[2] > 0:
        h, s, half = up, d - up, True
    else:
        h, s, half = d + up, -up, True
    swing = [here + h, here - s, here + h - s] + ([here + h + s, here + 2 * h, here + 2 * h + s] if half else [])
    leans = [here + s] + ([] if half else [here + h + s])
    as_cells = lambda cs: {tuple(int(v) for v in c[::-1]) for c in cs}  # noqa: E731
    return as_cells(swing), as_cells(leans)


def read(swarm):
    """The lattice as Swarm.lattice reads it, but each cube judged on its own: off when turned
    more than 10 degrees from the axes, further than LANDED from its cell's centre, or itself
    still moving (Swarm.lattice calls every cube off while any one of them moves)."""
    centres, rot = swarm.poses()
    nearest = np.rint(centres)
    upright = (np.abs(rot).max(axis=2) > ALIGNED).all(axis=1)
    v = swarm.data.qvel[swarm.cube_dofs]
    moving = (np.abs(v[:, :3]).max(axis=1) >= 0.2 * swarm.cube.side) | (np.abs(v[:, 3:]).max(axis=1) >= 0.5)
    off = ~upright | (np.abs(centres - nearest).max(axis=1) > LANDED) | moving
    return Reading(cells=nearest[:, ::-1].astype(int), off=off)


@dataclass(frozen=True)
class Pose:
    """What judge() compares a move against: the cubes' centres (N, 3, x y z in L), whether each
    stood upright, and the reading, all as a wave begins."""

    centres: np.ndarray
    upright: np.ndarray
    reading: Reading


def pose(swarm):
    centres, rot = swarm.poses()
    return Pose(centres, (np.abs(rot).max(axis=2) > ALIGNED).all(axis=1), read(swarm))


def judge(swarm, before, orders, targets):
    """Details of moves made from `before` (Pose), as Swarm.move classifies one, judged on the scene
    now: knock if a cube that was not moving was displaced by more than DISTURBED or newly turned
    off the axes (a cube already turned is not knocked again), charged to the mover that started
    nearest it; else off (the mover is off the lattice), reached (in its target), stayed (in its
    start cell) or strayed (in a third cell). orders: [(cube, direction)]; targets: (z, y, x) each.
    Returns [(detail, shift)]: shift is the furthest displacement charged to that mover, in L."""
    after = read(swarm)
    centres, rot = swarm.poses()
    cubes = np.array([cube for cube, _ in orders])
    moving = np.zeros(swarm.n, dtype=bool)
    moving[cubes] = True
    displaced = np.abs(centres - before.centres).max(axis=1)
    turned = before.upright & ~(np.abs(rot).max(axis=2) > ALIGNED).all(axis=1)
    knocked = ~moving & ((displaced > DISTURBED) | turned)
    nearest = np.linalg.norm(before.centres[:, None, :] - before.centres[cubes][None, :, :], axis=-1).argmin(axis=1)
    out = []
    for j, ((cube, _), target) in enumerate(zip(orders, targets)):
        mine = ~moving & (nearest == j)
        end = tuple(int(v) for v in after.cells[cube])
        detail = "off"
        if knocked[mine].any():
            detail = "knock"
        elif not after.off[cube]:
            detail = "reached" if end == tuple(target) else "stayed" if end == tuple(int(v) for v in before.reading.cells[cube]) else "strayed"
        out.append((detail, float(displaced[mine].max()) if mine.any() else 0.0))
    return out


def move_together(swarm, orders, watch=None):
    """Swarm.move for several cubes at once: each move is planned on the scene as it is now (the
    other movers count as standing still), every mover's flywheel is spun up and braked at the
    same moment, and the scene runs until it is still. orders: [(cube, direction)] of distinct
    cubes. A lone order is Swarm.move itself. The flywheels are driven through Swarm.step: a
    brake limit of zero leaves a flywheel alone exactly as Swarm.move's `free` does. Moves are
    classified by judge(). Returns a Move per order; sim and wall seconds are those of the call."""
    if len(orders) == 1:
        return [swarm.move(*orders[0], watch=watch)]
    clock = time.perf_counter()
    began = swarm.data.time
    before = pose(swarm)
    pivots = [swarm.plan(cube, direction, before.reading) for cube, direction in orders]
    movers = [k for k, pivot in enumerate(pivots) if pivot is not None]
    speeds = np.zeros(len(orders))
    brakes = np.zeros(len(orders))
    details = ["refused"] * len(orders)
    shifts = [0.0] * len(orders)
    if movers:
        jitter = swarm.jitter
        for k in movers:
            speeds[k] = min(RPM[pivots[k].primitive], swarm.cube.max_rpm) * (1 + swarm.rng.normal(0, jitter.speed) if jitter.speed > 0 else 1.0)
            brakes[k] = swarm.cube.brake_torque * (1 + swarm.rng.uniform(-jitter.brake, jitter.brake) if jitter.brake > 0 else 1.0)
        cubes = np.array([orders[k][0] for k in movers])
        swarm.run(BRAKE_AT, watch)
        for k in movers:
            spin = np.array(pivots[k].axis) * speeds[k] * 2 * math.pi / 60
            swarm.data.qvel[swarm.fly_dofs[orders[k][0]]] = swarm.data.xmat[swarm.flies[orders[k][0]]].reshape(3, 3).T @ spin
        holding = brakes[movers][:, None]
        released = np.zeros((len(movers), 1))
        still = 0.0
        elapsed = 0.0
        while elapsed < HORIZON - BRAKE_AT and still < SETTLE:
            swarm.step(brake=(cubes, holding if elapsed < BRAKE_HOLD else released))
            elapsed += DT
            still = still + DT if elapsed > BRAKE_HOLD and swarm.calm() else 0.0
            if watch is not None:
                watch(swarm)
        judged = judge(swarm, before, [orders[k] for k in movers], [pivots[k].target for k in movers])
        for k, (detail, shift) in zip(movers, judged):
            details[k], shifts[k] = detail, shift
    final = read(swarm)
    out = []
    for k, (cube, direction) in enumerate(orders):
        pivot = pivots[k]
        out.append(
            Move(
                cube=cube,
                direction=tuple(direction),
                primitive=pivot.primitive if pivot else "",
                bonds=pivot.bonds if pivot else 0,
                hinges=pivot.hinges if pivot else 0,
                rubs=pivot.rubs if pivot else 0,
                rpm=round(float(speeds[k])),
                outcome={"reached": "reached", "stayed": "stayed", "refused": "stayed"}.get(details[k], "harmful"),
                detail=details[k],
                cell=tuple(int(v) for v in final.cells[cube]),
                shift=round(shifts[k], 3),
                sim_seconds=round(float(swarm.data.time - began), 3),
                wall_seconds=round(time.perf_counter() - clock, 3),
            )
        )
    return out


# ---------------------------------------------------------------------------------------------
# The task's checker every step, and the stop rule


class Checker:
    """Whether an episode's task passes on the scene as it is now, as rollout.play scores the end
    of an episode (a switch task on its last instruction, from the step that instruction came;
    nothing is eaten in a hostable scene). Call before(world, t) with the world as step t begins
    and after(world, t) with the world after it; after returns whether the task passes then, and
    False while an earlier instruction of a switch task is in force."""

    def __init__(self, episode):
        self.task = episode.task
        start = R.prepare_world(episode.world)
        self.static = {"walls": start.walls[0].numpy(), "cheese_color": start.cheese_color[0].tolist(), "gravity": bool(start.gravity[0]), "meta": episode.meta}
        self.last = int(self.task.spec["steps"][-1]) if self.task.parts else 0
        self.since = S.snapshot(start, 0)

    def before(self, world, i, t):
        if self.task.parts and t == self.last:
            self.since = S.snapshot(world, i)

    def after(self, world, i, t):
        if t < self.last:
            return False
        task = self.task.parts[-1] if self.task.parts else self.task
        value, unasked = S.score(task, self.static, self.since, S.snapshot(world, i), [])
        return R.passed(self.task, float(value), bool(unasked))


class Tracked:
    """The part of an `advance` (rollout.play's hook) shared by the grid and MuJoCo: each
    episode's checker every step (first_pass: the step after which it first passed, None if
    never; passes[i][t] per step) and, with stop, no more moves in a scene once it has passed."""

    def __init__(self, episodes, stop=False):
        self.episodes = episodes
        self.stop = stop
        self.checkers = [Checker(e) for e in episodes]
        self.first_pass = [None] * len(episodes)
        self.passes = [[] for _ in episodes]

    def frozen(self, i):
        return self.stop and self.first_pass[i] is not None

    def track(self, before, after, t):
        for i, checker in enumerate(self.checkers):
            if t >= self.episodes[i].steps:
                continue
            checker.before(before, i, t)
            ok = checker.after(after, i, t)
            self.passes[i].append(bool(ok))
            if ok and self.first_pass[i] is None:
                self.first_pass[i] = t

    def summary(self, i, passed_end):
        """{"first_pass", "ever_passed", "damaged"}: damaged is a scene that passed at some step and
        does not at the end (later moves undid it)."""
        first = self.first_pass[i]
        return {"first_pass": first, "ever_passed": first is not None, "damaged": first is not None and not passed_end}


class Grid(Tracked):
    """advance for rollout.play that steps the grid itself (world.step under the rule), with the
    checker every step and the stop rule: the grid arm matching an Executor run."""

    def __init__(self, episodes, rule=R.RULE, stop=False):
        super().__init__(episodes, stop)
        self.rule = rule

    def __call__(self, world, codes, priority, t):
        codes = codes.clone()
        for i in range(len(self.episodes)):
            if self.frozen(i):
                codes[i] = STAY
        after, _ = step(world, codes, priority, self.rule)
        self.track(world, after, t)
        return after


# ---------------------------------------------------------------------------------------------
# The executor


@dataclass(frozen=True)
class Order:
    """A move the grid lets happen this step: the robot at `cell` (z, y, x) takes action `code`
    toward `target`; score is its priority."""

    cell: tuple
    code: int
    direction: tuple
    target: tuple
    score: float


class Executor(Tracked):
    """advance(world, codes, priority, t) for rollout.play: carries out every scene's step with
    the cubes of one cube_swarm.Swarm per episode (module docstring). episodes: rollout.Episodes
    on hostable scenes. mode: MODES. swarm: options for cube_swarm.Swarm (jitter, magnets, ...);
    each Swarm is seeded with its episode's seed. rule: the grid rule that filters the codes (None:
    every code goes to MuJoCo's plan). stop: a scene that passes its checker makes no more moves.
    snap: put cubes off the lattice back on the nearest free cell. record: keep the cubes' poses
    at RECORD_HZ for a video (save, render).

    After a run: moves[i] (a row per mover the grid let move: what MuJoCo did), steps[i] (a row per
    step), first_pass, passes (Tracked)."""

    def __init__(self, episodes, mode="sequential", swarm=None, rule=R.RULE, stop=False, snap=False, record=False):
        super().__init__(episodes, stop)
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        for e in episodes:
            if not hostable(e.world):
                raise ValueError(f"MuJoCo cannot host episode {e.key}: terrain or objects")
        self.mode, self.rule, self.snap, self.record = mode, rule, snap, record
        options = dict(swarm or {})
        self.swarms = [Swarm(cells_of(e.world), seed=int(e.seed), **options) for e in episodes]
        self.moves = [[] for _ in episodes]
        self.steps = [[] for _ in episodes]
        self.frames = [[] for _ in episodes]
        self.t = 0
        self.qpos = [np.array([sw.model.joint(f"c{k}").qposadr[0] for k in range(sw.n)])[:, None] + np.arange(7) for sw in self.swarms]
        self.start_mismatch = []
        for i, sw in enumerate(self.swarms):
            reading = read(sw)
            self.start_mismatch.append(bool(reading.off.any()) or reading.occupied() != {tuple(int(v) for v in c) for c in cells_of(episodes[i].world)})
            self.frame(i, sw)

    # -- recording

    def frame(self, i, swarm):
        self.frames[i].append((self.t, float(swarm.data.time), swarm.data.qpos[self.qpos[i]].astype(np.float32).copy()))

    def watcher(self, i):
        """The watch for scene i's swarm: a frame every 1 / RECORD_HZ simulated seconds."""
        if not self.record:
            return None
        frames = self.frames[i]

        def watch(swarm):
            if swarm.data.time >= frames[-1][1] + 1 / RECORD_HZ - 1e-9:
                self.frame(i, swarm)

        return watch

    # -- a step

    def __call__(self, world, codes, priority, t):
        self.t = t
        mass = world.mass.clone()
        for i in range(len(self.swarms)):
            if t < self.episodes[i].steps and not self.frozen(i):
                mass[i] = self.scene_step(i, world.index(i), codes[i : i + 1], priority[i], t)
        after = replace(world, mass=mass)
        self.track(world, after, t)
        return after

    def scene_step(self, i, grid, codes, priority, t):
        """Scene i's step t in MuJoCo; returns the read-back (D, H, W) mass."""
        clock = time.perf_counter()
        cpu = time.process_time()
        swarm = self.swarms[i]
        began = swarm.data.time
        robots = grid.mass[0] > 0.5
        asked = int(((codes[0] != STAY) & robots).sum())
        allowed = codes if self.rule is None else refuse(grid, codes, self.rule)
        orders = []
        for z, y, x in ((allowed[0] != STAY) & robots).nonzero().tolist():
            code = int(allowed[0, z, y, x])
            d = DIRS[code]
            orders.append(Order((z, y, x), code, d, (z + d[0], y + d[1], x + d[2]), float(priority[z, y, x])))
        orders.sort(key=lambda o: -o.score)
        claimed, movers, lost = set(), [], 0
        for o in orders:
            if o.target in claimed:
                lost += 1
                continue
            claimed.add(o.target)
            movers.append(o)
        waves = self.waves(i, movers, t) if movers else 0
        if movers:
            settle(swarm, self.watcher(i))
        snapped = self.snap_back(i) if self.snap else 0
        reading = read(swarm)
        mass, crowded = occupancy(reading, swarm.poses()[0], tuple(grid.mass.shape[1:]))
        expected = step(grid, codes, priority.unsqueeze(0), self.rule)[0].mass[0] > 0.5
        self.steps[i].append(
            {
                "t": t,
                "asked": asked,
                "allowed": len(orders),
                "lost": lost,
                "movers": len(movers),
                "waves": waves,
                "differ": int(((mass > 0.5) != expected).sum()),
                "off": int(reading.off.sum()),
                "snapped": snapped,
                "crowded": crowded,
                "sim_seconds": round(float(swarm.data.time - began), 3),
                "wall_seconds": round(time.perf_counter() - clock, 4),
                "cpu_seconds": round(time.process_time() - cpu, 4),
                "cells": [list(map(int, c)) for c in reading.cells.tolist()],
            }
        )
        if self.record and movers:
            self.frame(i, swarm)
        return mass

    def waves(self, i, movers, t):
        """Carry out the movers (Orders, by priority) in waves; a row per mover in moves[i].
        Returns the number of waves. Simultaneous waves also keep apart movers that touch (by a
        face, an edge or a corner): two touching cubes turning at once twist each other."""
        swarm = self.swarms[i]
        start = read(swarm)
        planned_at_start = {}
        at_start = {tuple(int(v) for v in c): k for k, c in enumerate(start.cells) if not start.off[k]}
        for o in movers:
            cube = at_start.get(o.cell)
            planned_at_start[o.cell] = cube is not None and swarm.plan(cube, o.direction, start) is not None
        pending = movers
        count = 0
        while pending:
            if count:
                settle(swarm, self.watcher(i))
            before = pose(swarm)
            reading = before.reading
            on = {tuple(int(v) for v in c): k for k, c in enumerate(reading.cells) if not reading.off[k]}
            off = {tuple(int(v) for v in c) for k, c in enumerate(reading.cells) if reading.off[k]}
            batch, later, taken, starts, leaning = [], [], set(), set(), set()
            for o in pending:
                cube = on.get(o.cell)
                if cube is None:
                    self.moves[i].append(self.row(t, count, o, None, None, "cube off" if o.cell in off else "displaced"))
                    continue
                pivot = swarm.plan(cube, o.direction, reading)
                if pivot is None:
                    self.moves[i].append(self.row(t, count, o, cube, None, "blocked" if planned_at_start[o.cell] else "refused"))
                    continue
                swing, leans = turn_cells(o.cell, o.direction, pivot)
                touching = any(max(abs(a - b) for a, b in zip(o.cell, other)) <= 1 for other in starts)
                clash = bool(swing & taken) or o.cell in taken or o.cell in leaning or bool(leans & starts) or touching
                if batch and (self.mode == "sequential" or clash):
                    later.append(o)
                    continue
                batch.append((o, cube, pivot))
                taken |= swing | {o.cell}
                starts.add(o.cell)
                leaning |= leans
            if batch:
                orders = [(cube, o.direction) for o, cube, _ in batch]
                results = move_together(swarm, orders, self.watcher(i))
                settle(swarm, self.watcher(i))
                judged = judge(swarm, before, orders, [pivot.target for _, _, pivot in batch])
                for (o, cube, _), result, (detail, shift) in zip(batch, results, judged):
                    if result.detail == "refused":
                        detail = "refused"
                    self.moves[i].append(self.row(t, count, o, cube, replace(result, detail=detail, shift=round(shift, 3)), detail, len(batch)))
            pending = later
            count += 1
        return count

    def row(self, t, wave, order, cube, result, detail, together=1):
        out = {"t": t, "wave": wave, "together": together, "cube": cube, "cell": list(order.cell), "direction": list(order.direction), "target": list(order.target), "detail": detail}
        if result is not None:
            move = asdict(result)
            out.update({k: move[k] for k in ("primitive", "bonds", "hinges", "rubs", "rpm", "shift", "sim_seconds", "wall_seconds")})
            out["end"] = list(result.cell)
        else:
            out["primitive"] = ""
        out["outcome"] = "reached" if detail == "reached" else "harmful" if detail in HARMFUL else "stayed"
        return out

    def snap_back(self, i):
        """Put every cube off the lattice upright on the nearest free cell that stands on the
        floor or on a cube on the lattice (within two cells of where it is), still, and let the
        scene settle. Returns how many were moved."""
        swarm = self.swarms[i]
        reading = read(swarm)
        if not reading.off.any():
            return 0
        centres, _ = swarm.poses()
        taken = {tuple(int(v) for v in c) for k, c in enumerate(reading.cells) if not reading.off[k]}
        snapped = 0
        for k in np.nonzero(reading.off)[0]:
            here = np.array(centres[k][::-1])  # (z, y, x) in L
            base = np.rint(here).astype(int)
            options = []
            for dz in range(-2, 3):
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        c = (int(base[0] + dz), int(base[1] + dy), int(base[2] + dx))
                        if c[0] < 0 or c in taken or not (c[0] == 0 or (c[0] - 1, c[1], c[2]) in taken):
                            continue
                        options.append((float(np.linalg.norm(np.array(c) - here)), c))
            if not options:
                continue
            _, cell = min(options)
            taken.add(cell)
            adr = self.qpos[i][k]
            swarm.data.qpos[adr[:3]] = (np.array(cell[::-1], dtype=float) + [0, 0, 0.5]) * swarm.cube.side
            swarm.data.qpos[adr[3:]] = [1.0, 0.0, 0.0, 0.0]
            swarm.data.qvel[swarm.cube_dofs[k]] = 0.0
            swarm.data.qvel[swarm.fly_dofs[k]] = 0.0
            snapped += 1
        if snapped:
            mujoco.mj_forward(swarm.model, swarm.data)
            swarm.run(SETTLE, self.watcher(i))
            settle(swarm, self.watcher(i))
        return snapped

    # -- trajectories

    def save(self, i, path, extra=None):
        """Write scene i's trajectory: the scene's MJCF, the cubes' poses at RECORD_HZ (qpos of
        their free joints, (F, N, 7)) with the simulated time and grid step of each frame, and the
        step rows (with the read-back cells after every step) and moves as JSON."""
        swarm = self.swarms[i]
        e = self.episodes[i]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        centres = (cells_of(e.world)[:, ::-1] + np.array([0, 0, 0.5])) * swarm.cube.side
        info = {"key": e.key, "texts": e.texts, "starts": e.starts, "kind": report_kind(e.task), "cells": cells_of(e.world).tolist(), "steps": self.steps[i], "moves": self.moves[i], **(extra or {})}
        np.savez_compressed(
            path,
            xml=np.array(swarm_xml(swarm.cube, centres, swarm.mass, swarm.friction)),
            qpos=np.stack([f[2] for f in self.frames[i]]),
            time=np.array([f[1] for f in self.frames[i]]),
            step=np.array([f[0] for f in self.frames[i]]),
            info=np.array(json.dumps(info, default=str)),
        )


def settle(swarm, watch=None, longest=SETTLE_MAX):
    """Run braked until the scene is calm, at most `longest` seconds."""
    waited = 0.0
    while not swarm.calm() and waited < longest:
        swarm.run(0.05, watch)
        waited += 0.05


def occupancy(reading, centres, shape):
    """((D, H, W) float mass, crowded) of a lattice reading: each cube on its cell (clamped into the
    grid), cubes on the lattice first; a cube whose cell is taken goes to the free cell nearest its
    centre (centres: (N, 3) x, y, z in L), and crowded counts those."""
    d, h, w = shape
    mass = torch.zeros(shape)
    crowded = 0
    top = np.array([d - 1, h - 1, w - 1])
    for k in np.argsort(reading.off, kind="stable"):
        cell = tuple(int(v) for v in np.clip(reading.cells[k], 0, top))
        if mass[cell] > 0.5:
            here = np.array(centres[k][::-1])
            free = (mass < 0.5).nonzero().numpy()
            cell = tuple(int(v) for v in free[np.linalg.norm(free - here, axis=1).argmin()])
            crowded += 1
        mass[cell] = 1.0
    return mass, crowded


def play_mujoco(episodes, actor, mode="sequential", swarm=None, rule=R.RULE, stop=False, snap=False, record=False):
    """Play episodes (hostable) with actor(world, t) -> codes, every step carried out in MuJoCo
    (Executor). The actor is rollout.ExpertActor or rollout.PolicyActor (for a policy, copy its
    thoughts onto the Played as rollout.play_policy does). No synthetic failures: the physics
    decides. Returns (Played per episode, the Executor with its rows and trajectories)."""
    executor = Executor(episodes, mode, swarm, rule, stop, snap, record)
    played = R.play(episodes, actor, rule, failure=None, record=record, advance=executor)
    return played, executor


# ---------------------------------------------------------------------------------------------
# Video


class Replay:
    """A recorded trajectory as Film expects a swarm: model, data, cube and poses."""

    def __init__(self, xml, n):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.cube = DESIGN.cube()
        self.n = n
        self.bodies = np.array([self.model.body(f"c{k}").id for k in range(n)])
        self.qpos = np.array([self.model.joint(f"c{k}").qposadr[0] for k in range(n)])[:, None] + np.arange(7)

    def show(self, qpos, at):
        self.data.qpos[self.qpos] = qpos
        self.data.time = at
        mujoco.mj_forward(self.model, self.data)

    def poses(self):
        return self.data.xpos[self.bodies] / self.cube.side - [0, 0, 0.5], self.data.xmat[self.bodies].reshape(self.n, 3, 3)


def render(path, video, slow=2.0, fps=30, size=(1280, 720)):
    """An MP4 of a saved trajectory (Executor.save), `slow` times slower than the simulated time;
    idle steps take no time. Returns the video's length in seconds."""
    with np.load(path) as f:
        xml, qpos, times = str(f["xml"]), f["qpos"], f["time"]
    replay = Replay(xml, qpos.shape[1])
    replay.show(qpos[0], times[0])
    film = Film(replay, video, fps=fps, slow=slow, size=size)
    film.still(replay, 1.0)
    frames = 0
    for pose, at in zip(qpos, times):
        replay.show(pose, at)
        due = film.due
        film.frame(replay)
        frames += film.due > due
    film.still(replay, 2.0)
    film.close()
    return round(frames / fps + 3.0, 2)


# ---------------------------------------------------------------------------------------------
# The task kinds MuJoCo can host, drawn on demo-sized scenes

HOST = {"layout": "flat", "gravity": True, "cheeses": (0, 0)}
# family -> the jobs drawn one at a time (None: the family's own draw), and the kinds kept. The rest
# of the suite needs a cheese (logic, herd, runner, approach, follow, eat and surround jobs, the
# objects families but zones) or terrain (bridge, fill, stairs, climb), or more robots than a demo
# scene has (a subset's tower: 12 robots in the selected group).
DRAWS = {
    "formation": [None],
    "construction": [None],
    "parallel": [None],
    "motion": ["stay", "goto", "gather", "spread", "groups", "line", "move"],
    "subset": ["goto", "gather", "spread", "groups", "line", "move"],
    "switch": ["stay", "goto", "gather", "spread"],
    "zones": ["goto", "line"],
    "build": [None],
}
KINDS = (
    "formation:mask",
    "construction:tower",
    "parallel:region+region",
    "parallel:region+tower",
    *(f"motion:{job}" for job in DRAWS["motion"]),
    *(f"subset:{job}" for job in DRAWS["subset"]),
    *(f"switch:then {job}" for job in DRAWS["switch"]),
    "zones:goto_zone",
    "zones:line_zone",
    "build",
)


def report_kind(task):
    """The kind a task is reported under (KINDS): the family and the job; a switch task by its last
    instruction (what it is scored on), a subset task by its job whatever the selector, build
    tasks together (the object is in the episode row)."""
    if task.family == "build":
        return "build"
    if task.parts:
        return "switch:then " + task.parts[-1].spec["kind"]
    if task.family == "subset":
        return "subset:" + task.spec["kind"]
    kind = S.task_kind(task)
    return kind if kind.startswith(task.family + ":") else f"{task.family}:{kind}"


def draw_task(rng, family, job, count):
    """(world, meta, task) of one task of a family (and job: DRAWS) on a hostable scene with
    `count` robots, drawn as the family's makers draw it with the scene forced to HOST; None when
    the draw does not fit (no task, or the swarm too small by the suite's bound)."""
    from beebot.vla.voxel import build as B
    from beebot.vla.voxel import objects as O

    if family == "zones":
        needs, make = O.make_zones(rng, "", job)
        world, meta = O.object_scenes(rng, count, layout="flat", gravity=True, objects=(0, 0), zones=needs["zones"])
        task = make(world, meta, 0)
        ok = task is not None and O.bound(task, world, 0) >= S.SUCCESS.get(task.family, S.PASS_MARK)
    else:
        if family in S.MAKERS:
            needs, make = S.MAKERS[family](rng, "")
        elif family in S.COVERAGE_MAKERS:
            needs, make = S.COVERAGE_MAKERS[family](rng, "", job)
        else:
            needs, make = B.make_build(rng, "")
        needs = {k: v for k, v in needs.items() if k not in ("robots", "colours", "require")}
        needs.update(HOST)
        if needs.get("groups"):
            if count < 4 * needs["groups"][0]:
                return None
            if count < 4 * needs["groups"][1]:
                needs["groups"] = (needs["groups"][0],) * 2
        needs = {k: v for k, v in needs.items() if v is not None or k == "gravity"}
        world, metas = S.scenes(rng, 1, robots=count, **needs)
        meta = metas[0]
        task = make(world, meta, 0)
        ok = task is not None and S.score_bound(task, world, 0) >= S.SUCCESS.get(task.family, S.PASS_MARK)
    if not ok or not hostable(world) or not task.plan:
        return None
    return world, meta, task


def episode_seed(seed, kind, index):
    return int.from_bytes(hashlib.sha256(f"closed_loop/{seed}/{kind}/{index}".encode()).digest()[:4], "big") >> 1


def hostable_episodes(kinds=KINDS, per_kind=3, seed=0, robots=ROBOTS, tries=400):
    """per_kind rollout.Episodes of every kind (KINDS) on hostable scenes with robots (low, high)
    cubes (uniform), drawn from default_rng([seed, crc32(family + job)]); a kind that does not come
    up in `tries` draws is left short."""
    wanted = {k: per_kind for k in kinds}
    found = {k: [] for k in kinds}
    for family, jobs in DRAWS.items():
        for job in jobs:
            if not any(k.startswith(family + ":") or k == family for k in kinds):
                continue
            rng = np.random.default_rng([seed, zlib.crc32(f"{family}/{job}".encode())])
            for _ in range(tries):
                if all(len(found[k]) >= wanted[k] for k in kinds if k.startswith(family + ":") or k == family):
                    break
                drawn = draw_task(rng, family, job, int(rng.integers(robots[0], robots[1] + 1)))
                if drawn is None:
                    continue
                world, meta, task = drawn
                kind = report_kind(task)
                if kind not in found or len(found[kind]) >= wanted[kind]:
                    continue
                parts = task.parts or [task]
                index = len(found[kind])
                found[kind].append(
                    R.Episode(
                        world,
                        meta,
                        task,
                        [p.text for p in parts],
                        [int(v) for v in task.spec["steps"]] if task.parts else [0],
                        [S.canonical_record(p) for p in parts],
                        R.episode_steps(task),
                        episode_seed(seed, kind, index),
                        key={"kind": kind, "i": index, "seed": seed, "family": task.family, "robots": int(world.mass.sum()), "object": task.spec.get("object") or task.spec.get("use") or ""},
                    )
                )
    return [e for k in kinds for e in found[k]]


# ---------------------------------------------------------------------------------------------
# Measuring the expert: MuJoCo against the grid


def grid_arm(episode, failure, stop=False):
    """The expert on the grid alone (failure: rollout's failure rates or None)."""
    tracker = Grid([episode], stop=stop)
    clock = time.perf_counter()
    played = R.play([episode], R.ExpertActor([episode], int(episode.seed)), failure=failure, advance=tracker)[0]
    return {"score": round(played.score, 4), "passed": played.passed, **tracker.summary(0, played.passed), "moves": round(float(np.sum(played.moves)), 2), "wall_seconds": round(time.perf_counter() - clock, 2)}


def run_episode(job):
    """One episode in every arm: the grid with rollout's failures (the training condition), the
    grid without failures (what the expert would do on ideal cubes) and MuJoCo. Returns (episode
    row, move rows)."""
    episode, mode, swarm, stop, snap = job
    torch.set_num_threads(1)
    row = {**episode.key, "texts": episode.texts, "steps": episode.steps, "mode": mode, "swarm": {k: str(v) for k, v in (swarm or {}).items()}, "stop": stop, "snap": snap}
    row["grid"] = grid_arm(episode, R.FAILURE)
    row["ideal"] = grid_arm(episode, None)
    clock = time.perf_counter()
    actor = R.ExpertActor([episode], int(episode.seed))
    acting = [0.0]

    def timed(world, t):
        began = time.perf_counter()
        codes = actor(world, t)
        acting[0] += time.perf_counter() - began
        return codes

    played, executor = play_mujoco([episode], timed, mode, swarm, stop=stop, snap=snap)
    played = played[0]
    steps = executor.steps[0]
    moves = executor.moves[0]
    active = [s for s in steps if s["movers"]]
    row["mujoco"] = {
        "score": round(played.score, 4),
        "passed": played.passed,
        **executor.summary(0, played.passed),
        "start_mismatch": executor.start_mismatch[0],
        "attempted": len(moves),
        "details": {d: sum(m["detail"] == d for m in moves) for d in sorted({m["detail"] for m in moves})},
        "harmful": sum(m["detail"] in HARMFUL for m in moves),
        "off_at_end": steps[-1]["off"] if steps else 0,
        "off_steps": sum(s["off"] > 0 for s in steps),
        "snapped": sum(s["snapped"] for s in steps),
        "crowded": sum(s["crowded"] for s in steps),
        "lost_claims": sum(s["lost"] for s in steps),
        "differ_steps": sum(s["differ"] > 0 for s in steps),
        "active_steps": len(active),
        "waves": sum(s["waves"] for s in steps),
        "sim_seconds": round(sum(s["sim_seconds"] for s in steps), 3),
        "mujoco_wall_seconds": round(sum(s["wall_seconds"] for s in steps), 2),
        "mujoco_cpu_seconds": round(sum(s["cpu_seconds"] for s in steps), 2),
        "actor_wall_seconds": round(acting[0], 2),
        "wall_seconds": round(time.perf_counter() - clock, 2),
    }
    for m in moves:
        m.update({"kind": episode.key["kind"], "i": episode.key["i"], "robots": episode.key["robots"]})
    return row, moves


def measure(kinds=KINDS, per_kind=3, seed=0, robots=ROBOTS, mode="sequential", swarm=None, stop=False, snap=False, workers=4, out=None, steps=None, log=None):
    """Every hostable kind played by the expert in MuJoCo and on the grid (run_episode), in
    `workers` processes; rows appended to `out` (episodes) and out's .moves.jsonl (one row per
    mover) as they finish. steps: a cap on every episode's length. Returns (episode rows, move
    rows)."""
    log = log or (lambda line: print(line, flush=True))
    episodes = hostable_episodes(kinds, per_kind, seed, robots)
    if steps:
        episodes = [replace(e, steps=min(e.steps, steps)) for e in episodes]
    log(f"{len(episodes)} episodes: " + ", ".join(f"{k} {sum(e.key['kind'] == k for e in episodes)}" for k in kinds))
    rows, moves = [], []
    moves_path = None if out is None else out.replace(".jsonl", "") + ".moves.jsonl"
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    jobs = [(e, mode, swarm, stop, snap) for e in episodes]
    with ProcessPoolExecutor(workers) as pool:
        for row, rows_of_moves in pool.map(run_episode, jobs):
            rows.append(row)
            moves += rows_of_moves
            m = row["mujoco"]
            log(f"{row['kind']:24s} {row['i']} robots {row['robots']:2d}  grid {row['grid']['passed']!s:5s} ideal {row['ideal']['passed']!s:5s} mujoco {m['passed']!s:5s} ({m['score']:.2f})  moves {m['attempted']:3d} harmful {m['harmful']}  sim {m['sim_seconds']:6.1f} s  wall {m['wall_seconds']:6.1f} s")
            if out:
                with open(out, "a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
                with open(moves_path, "a") as f:
                    f.writelines(json.dumps(r, default=str) + "\n" for r in rows_of_moves)
    return rows, moves


# ---------------------------------------------------------------------------------------------
# Tables


def share(values):
    return f"{np.mean(values):.2f}" if len(values) else "-"


def report(rows, moves, emit=print):
    """Markdown tables: pass shares per kind in each arm; what MuJoCo did with the moves the grid
    let happen, by primitive and hindrance and by kind; the time a step and an episode take."""
    kinds = [k for k in KINDS if any(r["kind"] == k for r in rows)] + sorted({r["kind"] for r in rows} - set(KINDS))
    emit("| kind | n | grid, failures | grid, no failures | MuJoCo | MuJoCo, stop at pass | MuJoCo score | damaged after passing (grid / MuJoCo) | off the lattice (episodes) |")
    emit("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for kind in kinds + ["all"]:
        part = [r for r in rows if kind in ("all", r["kind"])]
        emit(
            f"| {kind} | {len(part)} | {share([r['grid']['passed'] for r in part])} | {share([r['ideal']['passed'] for r in part])} | {share([r['mujoco']['passed'] for r in part])} | {share([r['mujoco']['ever_passed'] for r in part])} | {share([r['mujoco']['score'] for r in part])} | {sum(r['ideal']['damaged'] for r in part)} / {sum(r['mujoco']['damaged'] for r in part)} | {sum(r['mujoco']['off_steps'] > 0 for r in part)} |"
        )
    details = ("reached", "stayed", "off", "strayed", "knock", "refused", "blocked", "cube off", "displaced")
    emit("")
    emit("| primitive | hindrance | moves | " + " | ".join(details) + " | harmful per pivot |")
    emit("| --- | --- | --- | " + " | ".join("---" for _ in details) + " | --- |")
    groups = sorted({(m["primitive"], hindrance(m)) for m in moves})
    for name, kind in groups + [("all", "")]:
        part = [m for m in moves if name == "all" or (m["primitive"], hindrance(m)) == (name, kind)]
        pivots = sum(m["primitive"] != "" for m in part)
        rate = f"{sum(m['detail'] in HARMFUL for m in part) / pivots:.3f}" if pivots else "-"
        emit(f"| {name or '(not attempted)'} | {kind} | {len(part)} | " + " | ".join(str(sum(m['detail'] == d for m in part)) for d in details) + f" | {rate} |")
    firsts = []
    for r in rows:
        harmful = [m["t"] for m in moves if m["kind"] == r["kind"] and m["i"] == r["i"] and m["detail"] in HARMFUL]
        firsts.append(min(harmful) if harmful else None)
    hit = [f for f in firsts if f is not None]
    emit("")
    emit(f"first harmful pivot: {len(hit)} of {len(rows)} episodes have one; step of the first: median {np.median(hit) if hit else '-'}, quartiles {np.percentile(hit, 25) if hit else '-'} to {np.percentile(hit, 75) if hit else '-'}")
    emit("")
    emit("| kind | moves | reached | stayed | harmful | not attempted | harmful per episode |")
    emit("| --- | --- | --- | --- | --- | --- | --- |")
    for kind in kinds:
        part = [m for m in moves if m["kind"] == kind]
        n = sum(r["kind"] == kind for r in rows)
        harmful = sum(m["detail"] in HARMFUL for m in part)
        emit(f"| {kind} | {len(part)} | {sum(m['detail'] == 'reached' for m in part)} | {sum(m['detail'] == 'stayed' for m in part)} | {harmful} | {sum(m['primitive'] == '' for m in part)} | {harmful / max(n, 1):.2f} |")
    mj = [r["mujoco"] for r in rows]
    active = sum(m["active_steps"] for m in mj)
    emit("")
    if all("mujoco_cpu_seconds" in m for m in mj):
        moved = sum(m["attempted"] - sum(m["details"].get(d, 0) for d in ("refused", "blocked", "cube off", "displaced")) for m in mj)
        emit(f"cpu: {sum(m['mujoco_cpu_seconds'] for m in mj) / max(active, 1):.2f} s per step with movers, {sum(m['mujoco_cpu_seconds'] for m in mj) / max(moved, 1):.2f} s per pivot ({moved} pivots), {np.mean([m['mujoco_cpu_seconds'] for m in mj]):.1f} s per episode")
    emit(f"time: {sum(m['sim_seconds'] for m in mj) / max(active, 1):.2f} s simulated and {sum(m['mujoco_wall_seconds'] for m in mj) / max(active, 1):.2f} s wall per step with movers ({active} steps, {sum(m['waves'] for m in mj) / max(active, 1):.2f} waves each); per episode {np.mean([m['sim_seconds'] for m in mj]):.1f} s simulated (median {np.median([m['sim_seconds'] for m in mj]):.1f}), {np.mean([m['mujoco_wall_seconds'] for m in mj]):.1f} s MuJoCo wall, {np.mean([m['wall_seconds'] for m in mj]):.1f} s wall with the expert")


def hindrance(move):
    if not move["primitive"]:
        return ""
    return "bonded" if move["bonds"] else "hinged" if move["hinges"] else "rubs" if move["rubs"] else "free"


def load(path):
    rows = [json.loads(line) for line in open(path)]
    moves_path = path.replace(".jsonl", "") + ".moves.jsonl"
    moves = [json.loads(line) for line in open(moves_path)] if os.path.exists(moves_path) else []
    return rows, moves


# ---------------------------------------------------------------------------------------------
# Command line


def swarm_options(pairs):
    """--swarm key=value pairs as Swarm options (values read as JSON when they parse)."""
    out = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["measure", "episode", "render", "report"])
    parser.add_argument("paths", nargs="*", help="report: episode rows (.jsonl); render: a trajectory (.npz)")
    parser.add_argument("--kinds", nargs="*", default=list(KINDS))
    parser.add_argument("--kind", default="motion:goto", help="episode: the kind to draw")
    parser.add_argument("--index", type=int, default=0, help="episode: which draw of the kind")
    parser.add_argument("--per-kind", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--robots", type=int, nargs=2, default=list(ROBOTS))
    parser.add_argument("--mode", choices=MODES, default="sequential")
    parser.add_argument("--swarm", nargs="*", help="cube_swarm.Swarm options, key=value")
    parser.add_argument("--stop", action="store_true", help="a scene that passes its checker makes no more moves")
    parser.add_argument("--snap", action="store_true", help="put cubes off the lattice back on the nearest free cell")
    parser.add_argument("--steps", type=int, help="a cap on the episode's steps (72, 144 for build, otherwise)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", help="measure: episode rows; episode: the trajectory (.npz)")
    parser.add_argument("--video", help="episode, render: write an MP4 here")
    parser.add_argument("--slow", type=float, default=2.0)
    parser.add_argument("--checkpoint", help="episode: drive it with this policy instead of the expert")
    parser.add_argument("--model", help="episode: a base model as the policy (new parts fresh)")
    parser.add_argument("--tiny", help="episode: a tiny random policy with this model's architecture (smoke runs)")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    args = parser.parse_args()
    if args.command == "report":
        rows, moves = [], []
        for path in args.paths:
            a, b = load(path)
            rows += a
            moves += b
        report(rows, moves)
    elif args.command == "render":
        print(f"{render(args.paths[0], args.video, args.slow)} s of video")
    elif args.command == "measure":
        rows, moves = measure(args.kinds, args.per_kind, args.seed, tuple(args.robots), args.mode, swarm_options(args.swarm), args.stop, args.snap, args.workers, args.out, args.steps)
        report(rows, moves)
    else:
        episode = hostable_episodes([args.kind], args.index + 1, args.seed, tuple(args.robots))[args.index]
        if args.steps:
            episode = replace(episode, steps=min(episode.steps, args.steps))
        policy = None
        if args.checkpoint or args.model or args.tiny:
            from beebot.vla.voxel.scorecard import load_model

            policy = load_model(args)
            actor = R.PolicyActor(policy, [episode])
        else:
            actor = R.ExpertActor([episode], int(episode.seed))
        clock = time.perf_counter()
        played, executor = play_mujoco([episode], actor, args.mode, swarm_options(args.swarm), stop=args.stop, snap=args.snap, record=True)
        played = played[0]
        thoughts = actor.thoughts[0] if policy is not None else []
        steps = executor.steps[0]
        print(f"{episode.key['kind']} ({episode.key['object'] or episode.texts[-1]}): score {played.score:.2f} passed {played.passed}; first pass at step {executor.first_pass[0]}; {sum(s['movers'] for s in steps)} moves, {sum(m['detail'] in HARMFUL for m in executor.moves[0])} harmful; {sum(s['sim_seconds'] for s in steps):.1f} s simulated in {time.perf_counter() - clock:.1f} s")
        path = args.out or os.path.join("runs", "closed_loop", f"{episode.key['kind'].replace(':', '_').replace(' ', '_')}_{args.index}.npz")
        executor.save(0, path, {"score": played.score, "passed": played.passed, "thoughts": thoughts})
        print(f"trajectory: {path}")
        if args.video:
            print(f"video: {args.video}, {render(path, args.video, args.slow)} s")


if __name__ == "__main__":
    main()
