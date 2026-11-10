"""Many pivoting cubes (hardware/cube.py) in one persistent MuJoCo scene, moved one at a time.

cube.py simulates one move as a short scene and judges it by where the mover lands. Here N cubes
live in one scene for a whole episode. A caller (a voxel-grid policy or a script) names a cube and
one of the 14 lattice moves; the cube carries it out with the same mechanism as cube.py (spin the
flywheel, brake, let it settle); the grid is then read back from the physics, so the caller can
re-plan when a pivot fails.

Model (everything not listed is cube.py's: time step, contacts, magnet law, brake, tolerances):

1. Bodies. Every cube is a full module: a box on a free joint plus a flywheel. The flywheel is one
   body of isotropic inertia on a ball joint, so it can be spun about any world axis whatever way
   the cube has come to lie. Spun about one axis it carries the same momentum and leaves the cube
   the same mass and inertia as cube.py's hinged ring, so cube.py's calibrated speeds carry over.
   It stands for three orthogonal flywheels; the mass of the two idle ones is not modelled.
1. Drive. A move sets the mover's flywheel to its speed about the pivot's axis, brakes it from
   BRAKE_AT for BRAKE_HOLD (cube.py's Coulomb brake) and runs until every cube has been still for
   SETTLE, or for HORIZON. Idle flywheels are kept still by their brakes.
1. Magnets. cube.py's Magnets, with one change in how the interacting site pairs are found: only
   cube pairs whose centres are closer than sqrt(3) L + CUTOFF d0 are searched. A magnet is at most
   sqrt(3)/2 L from its cube's centre, so no site pair within CUTOFF d0 is ever dropped: the forces
   are identical to cube.py's and the search costs O(N) cube pairs on a lattice instead of N^2.
1. Variation. As cube.py's Jitter, drawn where it belongs in a persistent scene: friction once per
   scene, magnet strength and mass once per cube, brake torque and flywheel speed once per move,
   and placement error at the start (cubes are set down with gaps of up to `offset` L and snap
   together).

Moves. A pivot is a turn about the edge the mover shares with a support: h is the way the mover
goes and s the side its support is on (the floor or a cube at cell + s). If cell + h + s is also
solid, the mover turns 90 degrees into cell + h (floor roll, traverse, climbing or descending a
wall, wall crawl). If it is empty, the mover turns 180 degrees around the support's edge into
cell + h + s (up and over, down and over). The flywheel spins about h x s. The cells the mover
sweeps through must be empty: cell - s and cell + h - s, and for 180 degrees also cell + 2h and
cell + 2h + s.

Cells are (z, y, x) integers, z up, floor cubes at z = 0. Units: SI unless a field says otherwise.

What holds a mover back, counted by plan() for every move: bonds (cubes face to face with it
other than its support), hinges (cubes touching it by an edge other than the one it pivots on) and
rubs (cubes whose face it slides down on the way in). With permanent magnets a bonded mover does
not move at all; measured outcomes are in HARDWARE.md.

    from beebot.hardware.cube_swarm import Swarm, patch
    swarm = Swarm(patch(1, 5), seed=0)
    swarm.move(4, (1, 0, -1))  # the end cube goes up and over its neighbour at x - 1
    swarm.lattice().cells

    python -m beebot.hardware.cube_swarm demo --video runs/cube_swarm/demo.mp4
    python -m beebot.hardware.cube_swarm mixed --moves 120 --workers 3
"""

import argparse
import json
import math
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from beebot.hardware.cube import ALIGNED, BRAKE_AT, BRAKE_HOLD, DISTURBED, DT, G, HORIZON, LANDED, SETTLE, SHAPE_PEAK, Jitter, Magnets
from beebot.hardware.cube_search import Build

# The 5 cm cube on steel of HARDWARE.md round 2b (runs/cube/climb-5cm-steel.jsonl, final row).
DESIGN = Build(side=0.05, brake_scale=2.56, steel_floor=0.3)

# Flywheel speed per primitive in rpm: the speeds cube_search tuned for DESIGN (the final test row
# of runs/cube/climb-5cm-steel.jsonl, neighbours anchored, which is what a bonded patch on steel is
# closest to), with four exceptions measured in this scene by the same recipe (speed grid, jittered
# trials; HARDWARE.md, swarm section):
# concave_up: 17800 instead of 20000. Beside a two-high stack at the end of a row, 17083 and 18500
#   reached 12 of 12 each and 20000 reached 9 of 12;
# floor_peel, a floor roll away from bonded neighbours: no speed works (as cube_search found), so
#   the floor roll's speed is kept, at which a failed peel is harmless;
# convex_up_hang, up and over from hanging on a wall with nothing below (cube.py only climbs from
#   the floor, where the steel holds the mover back): 14500, 8 of 10;
# convex_down: 12500 instead of 9838. Off the end of a row 9838 reached 7 of 10 and 11500 to 13500
#   reached 10 of 10.
RPM = {
    "floor_roll": 4839,
    "floor_peel": 4839,
    "traverse": 10645,
    "convex_up": 18852,
    "convex_up_hang": 14500,
    "convex_down": 12500,
    "concave_up": 17800,
    "concave_down": 9092,
    "wall_crawl": 11518,
}

# The 14 moves, as (dz, dy, dx): one horizontal axis at most, optionally with up or down.
DIRECTIONS = [(dz, dy, dx) for dz in (0, 1, -1) for dy, dx in ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)) if (dz, dy, dx) != (0, 0, 0)]
SUPPORTS = [(0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]  # (x, y, z), tried in this order
EDGES = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if abs(a) + abs(b) + abs(c) == 2])
COLOURS = ["0.90 0.30 0.25", "0.95 0.60 0.20", "0.95 0.85 0.30", "0.45 0.75 0.35", "0.25 0.65 0.60", "0.30 0.55 0.90", "0.50 0.40 0.85", "0.80 0.45 0.75", "0.60 0.60 0.60", "0.85 0.85 0.85"]


@dataclass(frozen=True)
class Reading:
    """The grid as read back from the physics."""

    cells: np.ndarray  # (N, 3) int, (z, y, x): the cell nearest each cube's centre
    off: np.ndarray  # (N,) bool: tilted, between cells or still moving

    def occupied(self):
        return {tuple(int(v) for v in c) for c in self.cells}


@dataclass(frozen=True)
class Pivot:
    primitive: str  # a key of RPM
    axis: tuple  # (x, y, z): the flywheel spins about this world axis, right-handed
    target: tuple  # (z, y, x)
    bonds: int  # cubes face to face with the mover, other than its support
    hinges: int  # cubes that share only an edge with the mover (their edge magnets hold it), other than the one it pivots on
    rubs: int  # cubes beside the cells the mover passes into, along the flywheel's axis: it slides down their faces


@dataclass(frozen=True)
class Move:
    cube: int
    direction: tuple
    primitive: str  # "" when no primitive applies
    bonds: int
    hinges: int
    rubs: int
    rpm: float
    outcome: str  # "reached", "stayed" (harmless) or "harmful"
    detail: str  # reached | stayed | refused (no primitive, nothing spun) | off | strayed (aligned in a third cell) | knock
    cell: tuple  # (z, y, x) where the mover's centre is afterwards
    shift: float  # L: the furthest any other cube was displaced
    sim_seconds: float
    wall_seconds: float


def patch(rows, columns, z=0):
    """Cells of a rows x columns patch on level z, row by row."""
    return [(z, y, x) for y in range(rows) for x in range(columns)]


def swarm_xml(cube, centres, masses, friction):
    """MJCF for full modules centred at `centres` (N, 3) in metres, with masses (N,) in kg."""
    h = cube.side / 2
    bodies = []
    for i, (c, mass) in enumerate(zip(centres, masses)):
        fly_mass = cube.flywheel_share * mass
        inertia = fly_mass * (cube.flywheel_radius * cube.side) ** 2
        bodies.append(
            f'<body name="c{i}" pos="{c[0]} {c[1]} {c[2]}"><freejoint name="c{i}"/>'
            f'<geom type="box" size="{h} {h} {h}" mass="{mass - fly_mass}" rgba="{COLOURS[i % len(COLOURS)]} 1"/>'
            f'<body name="fly{i}"><joint name="fly{i}" type="ball"/><inertial pos="0 0 0" mass="{fly_mass}" diaginertia="{inertia} {inertia} {inertia}"/></body></body>'
        )
    return f"""<mujoco>
  <option timestep="{DT}" integrator="implicitfast" cone="elliptic" impratio="10" gravity="0 0 {-G}"/>
  <visual><global offwidth="1920" offheight="1080"/><headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5"/></visual>
  <asset><texture name="grid" type="2d" builtin="checker" rgb1="0.24 0.26 0.29" rgb2="0.20 0.22 0.25" width="64" height="64"/>
    <material name="grid" texture="grid" texuniform="true" texrepeat="{1 / cube.side} {1 / cube.side}"/></asset>
  <default><geom friction="{friction} 0.005 0.0001" condim="4" solref="0.002 1" solimp="0.95 0.99 0.0005"/></default>
  <worldbody><geom type="plane" size="2 2 0.1" pos="{h} {h} 0" material="grid"/>
    <light pos="0 -1 2" dir="0 0.4 -1" diffuse="0.6 0.6 0.6"/>
    {''.join(bodies)}</worldbody>
</mujoco>"""


class SwarmMagnets(Magnets):
    """cube.py's magnets with the pair search culled by cube distance (module docstring, 3)."""

    def __init__(self, cube, n, strength_scale):
        super().__init__(cube, n, strength_scale)
        self.reach = math.sqrt(3) * cube.side + self.CUTOFF * self.d0

    def refresh(self, flat):
        sites = flat.reshape(self.n, -1, 3)
        centres = sites.mean(1)  # the sites are symmetric about the cube's centre
        gap = centres[None, :, :] - centres[:, None, :]
        a, b = np.nonzero(np.triu((gap * gap).sum(-1) < self.reach**2, 1))
        r = sites[b][:, None, :, :] - sites[a][:, :, None, :]  # (pairs, 30, 30, 3)
        pair, i, j = np.nonzero((r * r).sum(-1) < (self.CUTOFF * self.d0) ** 2)
        per = sites.shape[1]
        self.first, self.second = a[pair] * per + i, b[pair] * per + j
        self.coef = np.sqrt(self.strength[self.first] * self.strength[self.second]) / self.d0 / SHAPE_PEAK



class Swarm:
    """N cubes in one scene. cells: (N, 3) lattice cells (z, y, x), each resting on the floor or on
    another cube. cube: a cube.Cube (default: DESIGN). jitter: cube.Jitter (cube.NOMINAL for none)."""

    def __init__(self, cells, cube=None, seed=0, jitter=Jitter(), settle=0.3):
        self.cube = cube or DESIGN.cube()
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)
        rng = self.rng
        cells = np.asarray(cells, dtype=int)
        self.n = n = len(cells)
        lo, hi = jitter.friction
        self.friction = float(rng.uniform(lo, hi)) if hi > lo else lo
        self.mass = self.cube.mass * (1 + rng.uniform(-jitter.mass, jitter.mass, size=n) if jitter.mass > 0 else np.ones(n))
        strength = 1 + rng.uniform(-jitter.magnet, jitter.magnet, size=n) if jitter.magnet > 0 else np.ones(n)
        # placement error: a lone cube is set down up to `offset` L off its cell; cubes that touch are
        # set down with gaps of `offset` L about their group's middle and half that of scatter, and
        # the magnets close the gaps during the first settle
        xyz = cells[:, ::-1].astype(float)
        place = xyz.copy()
        adjacency = np.abs(cells[:, None, :] - cells[None, :, :]).sum(-1) <= 1
        group = np.arange(n)
        for _ in range(n):
            group = np.where(adjacency, group[None, :], n).min(axis=1)
        for g in np.unique(group) if jitter.offset > 0 else []:
            members = group == g
            middle = xyz[members, :2].mean(0)
            scatter = jitter.offset if members.sum() == 1 else jitter.offset / 2
            place[members, :2] = middle + (xyz[members, :2] - middle) * (1 + jitter.offset) + rng.uniform(-scatter, scatter, size=(members.sum(), 2))
        self.model = mujoco.MjModel.from_xml_string(swarm_xml(self.cube, (place + [0, 0, 0.5]) * self.cube.side, self.mass, self.friction))
        self.data = mujoco.MjData(self.model)
        self.magnets = SwarmMagnets(self.cube, n, strength)
        self.bodies = np.array([self.model.body(f"c{i}").id for i in range(n)])
        self.flies = np.array([self.model.body(f"fly{i}").id for i in range(n)])
        self.cube_dofs = np.array([self.model.joint(f"c{i}").dofadr[0] for i in range(n)])[:, None] + np.arange(6)  # linear then angular
        self.fly_dofs = np.array([self.model.joint(f"fly{i}").dofadr[0] for i in range(n)])[:, None] + np.arange(3)
        self.fly_inertia = self.cube.flywheel_share * self.mass * (self.cube.flywheel_radius * self.cube.side) ** 2
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)
        self.run(settle)

    # -----------------------------------------------------------------------------------------
    # physics

    def step(self, free=-1, brake=None):
        """One time step. The flywheel of cube `free` is left alone; every other one is braked."""
        data = self.data
        spin = data.qvel[self.fly_dofs]  # (n, 3), relative to the cube, in the flywheel's frame
        stop = 0.5 * self.fly_inertia[:, None] * spin / DT  # the torque that would halt it in about two steps
        size = np.linalg.norm(stop, axis=1, keepdims=True)
        limit = np.full((self.n, 1), self.cube.brake_torque)
        if brake is not None:
            limit[brake[0]] = brake[1]
        torque = -stop * np.minimum(1.0, limit / np.maximum(size, 1e-12))
        if free >= 0:
            torque[free] = 0.0
        data.qfrc_applied[self.fly_dofs] = torque
        data.xfrc_applied[self.bodies] = self.magnets.wrench(data.xpos[self.bodies], data.xmat[self.bodies].reshape(self.n, 3, 3))
        mujoco.mj_step(self.model, data)
        self.steps += 1

    def calm(self):
        v = self.data.qvel[self.cube_dofs]
        return bool(np.abs(v[:, :3]).max() < 0.2 * self.cube.side and np.abs(v[:, 3:]).max() < 0.5)

    def run(self, seconds, watch=None):
        """Advance with every flywheel braked."""
        for _ in range(int(round(seconds / DT))):
            self.step()
            if watch is not None:
                watch(self)

    def poses(self):
        """Centres (N, 3) as (x, y, z) in L with the floor layer at z = 0, and rotations (N, 3, 3)."""
        return self.data.xpos[self.bodies] / self.cube.side - [0, 0, 0.5], self.data.xmat[self.bodies].reshape(self.n, 3, 3)

    # -----------------------------------------------------------------------------------------
    # the grid

    def lattice(self):
        """Read every cube's pose back: its nearest cell, and whether it is off the lattice (turned
        more than 10 degrees from the axes, further than LANDED from the cell's centre, or the scene
        is still moving)."""
        centres, rot = self.poses()
        nearest = np.rint(centres)
        upright = (np.abs(rot).max(axis=2) > ALIGNED).all(axis=1)
        off = ~upright | (np.abs(centres - nearest).max(axis=1) > LANDED) | (not self.calm())
        return Reading(cells=nearest[:, ::-1].astype(int), off=off)

    def plan(self, cube, direction, reading=None):
        """The pivot that takes `cube` one step along direction (dz, dy, dx) given its surroundings,
        or None when no primitive applies (module docstring, Moves)."""
        reading = reading or self.lattice()
        cubes_at = {c[::-1] for c in reading.occupied()}  # (x, y, z)
        here = np.array(reading.cells[cube][::-1])
        step = np.array(direction[::-1])
        parts = [np.eye(3, dtype=int)[a] * step[a] for a in range(3) if step[a]]
        candidates = []
        if len(parts) == 1 and tuple(direction) in DIRECTIONS:
            candidates = [(parts[0], np.array(s), 1) for s in SUPPORTS if not np.dot(s, parts[0])]
        if len(parts) == 2 and tuple(direction) in DIRECTIONS:
            candidates = [(parts[0], parts[1], 2), (parts[1], parts[0], 2)]

        def solid(c):
            return c[2] < 0 or tuple(int(v) for v in c) in cubes_at

        pivot_found = None
        for h, s, quarters in candidates:
            sweep = [here + h, here - s, here + h - s] + ([here + h + s, here + 2 * h, here + 2 * h + s] if quarters == 2 else [])
            has_support = solid(here + s) and (quarters == 2 or solid(here + h + s))
            name = primitive_name(h, s, quarters, on_floor=here[2] == 0, hanging=not solid(here - [0, 0, 1]))
            if pivot_found is None and has_support and name and not any(solid(c) for c in sweep) and not reading.off[cube]:
                faces = [here + f for f in SUPPORTS if tuple(f) != tuple(s) and tuple(int(v) for v in here + f) in cubes_at]
                target = here + h + (s if quarters == 2 else 0)
                axis = np.cross(h, s)
                beside = [c + side for c in ([here + h, target] if quarters == 2 else [target]) for side in (axis, -axis)]
                edges = [here + e for e in EDGES if tuple(int(v) for v in here + e) in cubes_at and tuple(e) != tuple(h + s)]
                pivot_found = Pivot(
                    primitive="floor_peel" if name == "floor_roll" and faces else name,
                    axis=tuple(int(v) for v in axis),
                    target=tuple(int(v) for v in target[::-1]),
                    bonds=len(faces),
                    hinges=len(edges),
                    rubs=sum(solid(c) for c in beside),
                )
        return pivot_found

    def move(self, cube, direction, rpm=None, watch=None):
        """Carry out one lattice move of one cube and classify it from the poses before and after.
        rpm overrides the calibrated flywheel speed. watch(swarm), if given, is called every step."""
        clock = time.perf_counter()
        began = self.data.time
        before = self.lattice()
        centres_before, _ = self.poses()
        pivot = self.plan(cube, direction, before)
        detail = "refused"
        speed = 0.0
        shift = 0.0
        if pivot is not None:
            jitter = self.jitter
            speed = min(rpm or RPM[pivot.primitive], self.cube.max_rpm) * (1 + self.rng.normal(0, jitter.speed) if jitter.speed > 0 else 1.0)
            brake = self.cube.brake_torque * (1 + self.rng.uniform(-jitter.brake, jitter.brake) if jitter.brake > 0 else 1.0)
            self.run(BRAKE_AT, watch)
            spin = np.array(pivot.axis) * speed * 2 * math.pi / 60
            self.data.qvel[self.fly_dofs[cube]] = self.data.xmat[self.flies[cube]].reshape(3, 3).T @ spin
            still = 0.0
            elapsed = 0.0
            while elapsed < HORIZON - BRAKE_AT and still < SETTLE:
                self.step(free=-1 if elapsed < BRAKE_HOLD else cube, brake=(cube, brake))
                elapsed += DT
                still = still + DT if elapsed > BRAKE_HOLD and self.calm() else 0.0
                if watch is not None:
                    watch(self)
            after = self.lattice()
            centres, rot = self.poses()
            others = np.arange(self.n) != cube
            upright = (np.abs(rot).max(axis=2) > ALIGNED).all(axis=1)
            shift = float(np.abs(centres - centres_before)[others].max()) if self.n > 1 else 0.0
            neighbour_knock = shift > DISTURBED or not upright[others].all()
            end_cell = tuple(int(v) for v in after.cells[cube])
            detail = "off"
            if neighbour_knock:
                detail = "knock"
            elif not after.off[cube]:
                detail = "reached" if end_cell == pivot.target else "stayed" if end_cell == tuple(int(v) for v in before.cells[cube]) else "strayed"
        final = self.lattice()
        return Move(
            cube=cube,
            direction=tuple(direction),
            primitive=pivot.primitive if pivot else "",
            bonds=pivot.bonds if pivot else 0,
            hinges=pivot.hinges if pivot else 0,
            rubs=pivot.rubs if pivot else 0,
            rpm=round(float(speed)),
            outcome={"reached": "reached", "stayed": "stayed", "refused": "stayed"}.get(detail, "harmful"),
            detail=detail,
            cell=tuple(int(v) for v in final.cells[cube]),
            shift=round(shift, 3),
            sim_seconds=round(float(self.data.time - began), 3),
            wall_seconds=round(time.perf_counter() - clock, 3),
        )


def primitive_name(h, s, quarters, on_floor, hanging):
    """cube.py's name for the pivot with motion h and support side s, or "" when cube.py has no
    such primitive (hanging under a ceiling, passing under an overhang). convex_up_hang is convex_up
    started with nothing below the mover."""
    below, level = s[2] == -1, s[2] == 0
    names = {
        (1, True, 0): "floor_roll" if on_floor else "traverse",
        (1, False, 0): "wall_crawl" if level else "",
        (1, False, 1): "concave_up" if level else "",
        (1, False, -1): "concave_down" if level else "",
        (2, True, 0): "convex_down",
        (2, False, 1): ("convex_up_hang" if hanging else "convex_up") if level else "",
    }
    return names.get((quarters, bool(below), int(h[2])), "")


# ---------------------------------------------------------------------------------------------
# Scripted demonstration, mixed single moves, video


class Film:
    """Writes the scene to an MP4 through ffmpeg, `slow` times slower than real time. Pass
    film.frame as the `watch` of Swarm.move or Swarm.run."""

    def __init__(self, swarm, path, fps=30, slow=2.0, size=(1280, 720), crf=28):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.renderer = mujoco.Renderer(swarm.model, size[1], size[0])
        centres, _ = swarm.poses()
        low, high = centres[:, :2].min(0), centres[:, :2].max(0)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat = [*((low + high) / 2 * swarm.cube.side), 1.2 * swarm.cube.side]
        self.camera.distance = (max(high - low) + 5) * swarm.cube.side
        self.camera.azimuth, self.camera.elevation = 110, -30
        self.fps = fps
        self.every = 1 / (fps * slow)
        self.due = swarm.data.time
        command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-"]
        self.pipe = subprocess.Popen(command + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), path], stdin=subprocess.PIPE)

    def frame(self, swarm):
        if swarm.data.time >= self.due:
            self.due += self.every
            self.renderer.update_scene(swarm.data, camera=self.camera)
            self.pipe.stdin.write(self.renderer.render().tobytes())

    def still(self, swarm, seconds):
        """Hold the present view for `seconds` of video; the physics does not advance."""
        self.renderer.update_scene(swarm.data, camera=self.camera)
        self.pipe.stdin.write(self.renderer.render().tobytes() * int(seconds * self.fps))

    def close(self):
        self.pipe.stdin.close()
        self.pipe.wait()
        self.renderer.close()


def play(swarm, script, retries=3, film=None, log=None):
    """Run a script of (cube, direction) moves: after every move the grid is read back; a harmless
    failure is retried up to `retries` times; a harmful one (or a move that never succeeds) stops
    the script. Returns the list of Move rows, one per attempt, each with the grid after it."""
    rows = []
    on_track = True
    for index, (cube, direction) in enumerate(script):
        attempt = 0
        outcome = "stayed"
        while on_track and outcome == "stayed" and attempt <= retries:
            result = swarm.move(cube, direction, watch=film.frame if film else None)
            reading = swarm.lattice()
            row = {"step": index, "attempt": attempt, **asdict(result), "cells": reading.cells.tolist(), "off": reading.off.tolist()}
            rows.append(row)
            if log:
                with open(log, "a") as f:
                    f.write(json.dumps(row) + "\n")
            print(f"step {index:2d} try {attempt}  cube {cube} {tuple(direction)}  {result.primitive or '-':12s} bonds {result.bonds} hinges {result.hinges} rubs {result.rubs}  {result.rpm:5d} rpm  {result.detail:8s} at {result.cell}  sim {result.sim_seconds:.2f} s  wall {result.wall_seconds:.2f} s", flush=True)
            outcome = result.outcome
            attempt += 1
        on_track = on_track and outcome == "reached"
    return rows


# The demonstration starts from a row of five and five loose cubes (cubes 5 to 9) two cells away.
# A packed 2 x 5 patch cannot be used: no cube in it can move (HARDWARE.md, swarm section). The
# loose cubes travel so that they never touch another cube by a face or an edge until they meet the
# structure head on.
DEMO_CELLS = patch(1, 5) + [(0, 2, x) for x in (0, 2, 5, 7, 9)]
UP, DOWN, EAST, WEST, SOUTH = (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1), (0, -1, 0)
UP_WEST, UP_EAST = (1, 0, -1), (1, 0, 1)
# A three-high tower at x = 3 with a step on either side: heights 1, 2, 3, 2 along x = 1..4.
DEMO_SCRIPT = [
    (4, UP_WEST),  # the row's end cube climbs onto its neighbour: a two-high stack at x = 3
    (7, SOUTH),  # a loose cube rolls round to meet the foot of the stack head on
    (7, SOUTH),
    (7, WEST),
    (7, UP),  # climbs the stack's side and hangs beside its top cube
    (7, UP_WEST),  # up and over, onto the stack: three high
    (0, UP_EAST),  # the row's other end climbs on
    (0, EAST),  # and walks along the top to the tower
    (8, SOUTH),  # a second loose cube comes to the tower's foot
    (8, SOUTH),
    (8, WEST),
    (8, WEST),
    (8, WEST),
    (9, SOUTH),  # a third follows it
    (9, SOUTH),
    (9, WEST),
    (9, WEST),
    (9, WEST),
    (9, WEST),
    (9, UP_WEST),  # and climbs onto it
]

# Start layouts of ten cubes for the mixed single moves.
LAYOUTS = {
    "patch": patch(2, 5),
    "row": patch(1, 5) + [(0, 2, x) for x in (0, 2, 4, 6, 8)],
    "wall": [(z, 0, x) for z in (0, 1) for x in (0, 1, 2)] + [(0, 2, 0), (0, 2, 2), (0, 2, 4), (0, -2, 1)],
    "hang": [(z, 0, x) for z in (0, 1) for x in (0, 1, 2)] + [(1, 1, 0), (0, 0, 3), (0, 3, 2), (0, -2, 1)],
    "steps": patch(1, 4) + [(1, 0, 1), (1, 0, 2), (0, 2, 0), (0, 2, 2), (0, 2, 4), (0, 0, 6)],
}


def walk(args):
    """A random walk of single moves from a layout: each step draws a kind of move (primitive, with
    or without bonds) among those the grid offers, then a cube and direction of that kind. The
    walk ends at a harmful move. Returns one dict per move."""
    layout, seed, moves = args
    swarm = Swarm(LAYOUTS[layout], seed=seed)
    rng = np.random.default_rng([seed, 1])
    rows = []
    over = False
    while len(rows) < moves and not over:
        reading = swarm.lattice()
        kinds = {}
        for cube in range(swarm.n):
            for direction in DIRECTIONS:
                pivot = swarm.plan(cube, direction, reading)
                if pivot is not None:
                    kinds.setdefault((pivot.primitive, hindrance(pivot.bonds, pivot.hinges, pivot.rubs)), []).append((cube, direction))
        keys = sorted(kinds)
        over = not keys
        if keys:
            options = kinds[keys[rng.integers(len(keys))]]
            cube, direction = options[rng.integers(len(options))]
            result = swarm.move(cube, direction)
            rows.append({"layout": layout, "seed": seed, "index": len(rows), **asdict(result)})
            over = result.outcome == "harmful"
    return rows


def hindrance(bonds, hinges, rubs):
    """free: the mover touches only its support and the cube it pivots on, and slides past nothing."""
    return "bonded" if bonds else "hinged" if hinges else "rubs" if rubs else "free"


def tally(rows):
    """Print outcome counts by primitive and hindrance, and the time per move."""
    kinds = sorted({(r["primitive"], hindrance(r["bonds"], r["hinges"], r["rubs"])) for r in rows})
    details = ("reached", "stayed", "off", "strayed", "knock")
    print(f"{'primitive':17s} {'':6s} {'n':>4s} " + " ".join(f"{d:>8s}" for d in details))
    for name, kind in kinds:
        part = [r for r in rows if r["primitive"] == name and hindrance(r["bonds"], r["hinges"], r["rubs"]) == kind]
        print(f"{name:17s} {kind:6s} {len(part):4d} " + " ".join(f"{sum(r['detail'] == d for r in part):8d}" for d in details))
    counts = {o: sum(r["outcome"] == o for r in rows) for o in ("reached", "stayed", "harmful")}
    print(f"{len(rows)} moves: {counts}; per move: {np.mean([r['sim_seconds'] for r in rows]):.2f} s simulated, {np.mean([r['wall_seconds'] for r in rows]):.2f} s wall")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["demo", "mixed"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", help="demo: write an MP4 here")
    parser.add_argument("--slow", type=float, default=2.0, help="demo: the video runs this many times slower than real time")
    parser.add_argument("--out", help="append one JSON line per move here")
    parser.add_argument("--moves", type=int, default=120, help="mixed: total moves")
    parser.add_argument("--walk", type=int, default=8, help="mixed: moves per random walk")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.command == "demo":
        swarm = Swarm(DEMO_CELLS, seed=args.seed)
        film = Film(swarm, args.video, slow=args.slow) if args.video else None
        if film:
            film.still(swarm, 1.0)
        rows = play(swarm, DEMO_SCRIPT, film=film, log=args.out)
        if film:
            film.still(swarm, 2.0)
            film.close()
        print(f"{sum(r['outcome'] == 'reached' for r in rows)} of {len(DEMO_SCRIPT)} scripted moves reached their cell; friction {swarm.friction:.2f}")
        heights = {}
        for z, y, x in swarm.lattice().cells.tolist():
            heights[(y, x)] = max(heights.get((y, x), 0), z + 1)
        print("heights by (y, x):", dict(sorted(heights.items())))
    else:
        names = list(LAYOUTS)
        jobs = [(names[k % len(names)], args.seed + k, args.walk) for k in range(math.ceil(args.moves / args.walk))]
        with ProcessPoolExecutor(args.workers) as pool:
            rows = [r for part in pool.map(walk, jobs) for r in part]
        if args.out:
            with open(args.out, "a") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
    tally(rows)


if __name__ == "__main__":
    main()
