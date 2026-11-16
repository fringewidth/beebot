"""MuJoCo as the world for the 2D grid policy; the grid is only what the policy sees.

Minimum physics: robots are flat cylinders (diameter ROBOT_DIAMETER squares) on two planar slide
joints, x and y, each driven by a velocity actuator with a force limit (ROBOT_FORCE); a simple
controller sets the commanded velocity toward the robot's target square. Cheeses are 4 x 4-square
boxes on the same two slide joints with dry friction on each joint (CHEESE_FRICTION, a joint
frictionloss) between two and three robots' force, so one or two robots cannot move a cheese and
three can, as the grid's friction threshold (world.friction(32) = 3 robots' worth). Walls are
static boxes. All contacts are frictionless (condim 1); nothing rotates or tips. The PGS solver:
with the default Newton solver, robots pinched in a dense crowd against a wall were shot out at
hundreds of squares per second (the solver hit its iteration cap); PGS keeps them under the
controller's speed.

Units: one square is one length unit. Square (row, col) covers x in [col, col + 1] and y in
[row, row + 1], so x grows with the column and y with the row, as in the grid (the plane is seen
from below; it does not matter for planar physics).

One policy step (Swarm.step): every robot's target is the centre of the neighbouring square its
sampled move points to (its own square's centre for stay); MuJoCo runs for STEP_SECONDS. Then
snap() reads the positions back onto the grid: cheeses to the nearest whole square of their top
left corner, robots to distinct free squares (no wall, no live cheese) by a Hungarian assignment
on the squared distance from each robot to each square's centre, so two robots that end near one
square go to the nearest two free squares. Eating stays the grid's rule (world.eat_cheeses on the
snapped state); an eaten cheese loses its contacts and stays where it was.
"""

import mujoco
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from beebot.vla.world import DIRS, STAY, World, cheese_side

ROBOT_DIAMETER = 0.8  # squares
ROBOT_MASS = 0.05
ROBOT_FORCE = 1.0  # actuator force limit on each axis
ROBOT_SPEED = 2.5  # squares per second on each axis, the controller's cap
ROBOT_GAIN = 6.0  # 1/s: commanded velocity per square of distance to the target
VELOCITY_KV = 2.0  # velocity actuator gain (force per unit of velocity error)
CHEESE_MASS = 0.3
CHEESE_FRICTION = 2.5 * ROBOT_FORCE  # dry friction on each slide joint: between 2 and 3 robots
FRICTION_SOLREF = "0.01 1"  # stiff friction constraint: under the threshold a cheese does not creep
FRICTION_SOLIMP = "0.99 0.999 0.001"
TIMESTEP = 0.005
STEP_SECONDS = 1.0  # simulated time per policy step
FRAMES_PER_STEP = 6  # positions recorded per policy step, for the video


def wall_rectangles(walls):
    """Cover the wall squares with few rectangles (row, col, height, width): runs along each row,
    merged downward with identical runs in the rows below."""
    size = walls.shape[0]
    left = walls.copy()
    rects = []
    for r in range(size):
        c = 0
        while c < size:
            if not left[r, c]:
                c += 1
                continue
            end = c
            while end < size and left[r, end]:
                end += 1
            height = 1
            while r + height < size and left[r + height, c:end].all() and (c == 0 or not left[r + height, c - 1]) and (end == size or not left[r + height, end]):
                height += 1
            left[r : r + height, c:end] = False
            rects.append((r, c, height, end - c))
            c = end
    return rects


def scene_xml(walls, robots, cheeses, side):
    """robots: (N, 2) squares; cheeses: (K, 2) top-left squares."""
    size = walls.shape[0]
    geoms = []
    for r, c, h, w in wall_rectangles(walls):
        geoms.append(f'<geom type="box" pos="{c + w / 2} {r + h / 2} 0.5" size="{w / 2} {h / 2} 0.5"/>')
    for x, y, hx, hy in ((size / 2, -0.5, size / 2 + 1, 0.5), (size / 2, size + 0.5, size / 2 + 1, 0.5), (-0.5, size / 2, 0.5, size / 2 + 1), (size + 0.5, size / 2, 0.5, size / 2 + 1)):
        geoms.append(f'<geom type="box" pos="{x} {y} 0.5" size="{hx} {hy} 0.5"/>')
    bodies = []
    actuators = []
    for k in range(len(robots)):
        bodies.append(
            f'<body name="robot{k}" pos="0 0 0.2">'
            f'<joint name="robot{k}x" type="slide" axis="1 0 0"/><joint name="robot{k}y" type="slide" axis="0 1 0"/>'
            f'<geom name="robot{k}" type="cylinder" size="{ROBOT_DIAMETER / 2} 0.1" mass="{ROBOT_MASS}"/></body>'
        )
        for axis in "xy":
            actuators.append(f'<velocity joint="robot{k}{axis}" kv="{VELOCITY_KV}" forcelimited="true" forcerange="{-ROBOT_FORCE} {ROBOT_FORCE}"/>')
    half = side / 2
    for k in range(len(cheeses)):
        bodies.append(
            f'<body name="cheese{k}" pos="0 0 {half}">'
            f'<joint name="cheese{k}x" type="slide" axis="1 0 0" frictionloss="{CHEESE_FRICTION}" solreffriction="{FRICTION_SOLREF}" solimpfriction="{FRICTION_SOLIMP}"/>'
            f'<joint name="cheese{k}y" type="slide" axis="0 1 0" frictionloss="{CHEESE_FRICTION}" solreffriction="{FRICTION_SOLREF}" solimpfriction="{FRICTION_SOLIMP}"/>'
            f'<geom name="cheese{k}" type="box" size="{half} {half} {half}" mass="{CHEESE_MASS}"/></body>'
        )
    return f"""
<mujoco>
  <option timestep="{TIMESTEP}" integrator="implicitfast" solver="PGS"/>
  <default><geom condim="1"/></default>
  <worldbody>
    {"".join(geoms)}
    {"".join(bodies)}
  </worldbody>
  <actuator>{"".join(actuators)}</actuator>
</mujoco>"""


class Swarm:
    """One scene in MuJoCo. robots: (N, 2) start squares, in the order the caller tracks them by;
    cheeses: (K, 2) top-left squares, in slot order."""

    def __init__(self, walls, robots, cheeses, side):
        self.walls = np.asarray(walls, bool)
        self.side = side
        self.n_robots, self.n_cheeses = len(robots), len(cheeses)
        self.model = mujoco.MjModel.from_xml_string(scene_xml(self.walls, robots, cheeses, side))
        self.data = mujoco.MjData(self.model)
        self.robot_q = np.array([[self.model.joint(f"robot{k}{a}").qposadr[0] for a in "xy"] for k in range(self.n_robots)], int).reshape(-1, 2)
        self.cheese_q = np.array([[self.model.joint(f"cheese{k}{a}").qposadr[0] for a in "xy"] for k in range(self.n_cheeses)], int).reshape(-1, 2)
        self.cheese_geoms = [self.model.geom(f"cheese{k}").id for k in range(self.n_cheeses)]
        self.alive = np.ones(self.n_cheeses, bool)
        robots = np.asarray(robots, float).reshape(-1, 2)
        cheeses = np.asarray(cheeses, float).reshape(-1, 2)
        self.data.qpos[self.robot_q[:, 0]] = robots[:, 1] + 0.5
        self.data.qpos[self.robot_q[:, 1]] = robots[:, 0] + 0.5
        self.data.qpos[self.cheese_q[:, 0]] = cheeses[:, 1] + side / 2
        self.data.qpos[self.cheese_q[:, 1]] = cheeses[:, 0] + side / 2
        mujoco.mj_forward(self.model, self.data)

    def robot_xy(self):
        """(N, 2) robot centres (x, y)."""
        return self.data.qpos[self.robot_q]

    def cheese_xy(self):
        """(K, 2) cheese centres (x, y)."""
        return self.data.qpos[self.cheese_q]

    def positions(self):
        """(N + K, 2) continuous (row, col): robots as their square's (row, col) when centred on
        it, then cheeses as their top-left square's."""
        robots = self.robot_xy()[:, ::-1] - 0.5
        cheeses = self.cheese_xy()[:, ::-1] - self.side / 2
        return np.concatenate([robots, cheeses])

    def run(self, targets, seconds=STEP_SECONDS, frames=FRAMES_PER_STEP):
        """Drive every robot toward its target (N, 2) square for seconds. Returns the positions
        (frames, N + K, 2) at the start of each of frames equal slices of that time."""
        goal = np.asarray(targets, float)[:, ::-1] + 0.5  # (x, y) of the square centres
        substeps = round(seconds / TIMESTEP)
        every = substeps // frames
        out = []
        for s in range(substeps):
            if s % every == 0 and len(out) < frames:
                out.append(self.positions())
            velocity = np.clip(ROBOT_GAIN * (goal - self.robot_xy()), -ROBOT_SPEED, ROBOT_SPEED)
            self.data.ctrl[:] = velocity.reshape(-1)
            mujoco.mj_step(self.model, self.data)
        return np.stack(out)

    def snap(self):
        """(robot squares (N, 2) int, distinct and free; cheese top-left squares (K, 2) int)."""
        size = self.walls.shape[0]
        cheeses = np.rint(self.cheese_xy()[:, ::-1] - self.side / 2).astype(int)
        cheeses = np.clip(cheeses, 0, size - self.side)
        blocked = self.walls.copy()
        for (r, c), alive in zip(cheeses, self.alive):
            if alive:
                blocked[r : r + self.side, c : c + self.side] = True
        free = np.argwhere(~blocked)
        centres = free + 0.5
        where = self.robot_xy()[:, ::-1]  # (row, col) units: square (r, c) has centre (r + .5, c + .5)
        cost = ((where[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        rows, cols = linear_sum_assignment(cost)
        robots = np.zeros((self.n_robots, 2), int)
        robots[rows] = free[cols]
        return robots, cheeses

    def remove(self, k):
        """An eaten cheese: no more contacts; it stays where it is."""
        self.alive[k] = False
        self.model.geom_contype[self.cheese_geoms[k]] = 0
        self.model.geom_conaffinity[self.cheese_geoms[k]] = 0


def targets_of(squares, codes, walls):
    """Each robot's target square: its square plus the move of its sampled code (stay: itself).
    A move off the grid targets the robot's own square."""
    size = walls.shape[0]
    moves = np.array([DIRS[c] if c != STAY else (0, 0) for c in codes]).reshape(-1, 2)
    targets = squares + moves
    off = (targets < 0).any(1) | (targets >= size).any(1)
    targets[off] = squares[off]
    return targets


class Batch:
    """B scenes in MuJoCo behind a grid World: world() snaps every scene onto a World of the same
    layout as tasks.stack's (walls, boxes, cheese colours and slots), and step(codes) runs one
    policy step. squares[i] are scene i's robots' snapped squares, in the order of
    np.argwhere(scene mass), which is also the order the field-demo recorder tracks robots by."""

    def __init__(self, scenes, world):
        self.template = world  # tasks.stack(scenes)[0], on the CPU
        side = cheese_side(world.mass.shape[-1])
        self.swarms = []
        self.squares = []
        for scene in scenes:
            robots = np.argwhere(scene["mass"] > 0.5)
            self.swarms.append(Swarm(scene["walls"], robots, np.array(scene["cheeses"]).reshape(-1, 2), side))
            self.squares.append(robots)
        self.cheese_pos = world.cheese_pos.clone()

    def world(self, alive):
        """The snapped World; alive (B, K) bool, the grid's view of which cheeses are left."""
        mass = torch.zeros_like(self.template.mass)
        for i, swarm in enumerate(self.swarms):
            robots, cheeses = swarm.snap()
            self.squares[i] = robots
            mass[i, robots[:, 0], robots[:, 1]] = 1.0
            if len(cheeses):
                self.cheese_pos[i, : len(cheeses)] = torch.from_numpy(cheeses)
        return World(
            mass=mass,
            walls=self.template.walls,
            cheese_pos=self.cheese_pos.clone(),
            cheese_alive=alive.clone(),
            cheese_color=self.template.cheese_color,
            box_rect=self.template.box_rect,
            box_color=self.template.box_color,
        )

    def step(self, codes, active=None):
        """codes (B, H, W) sampled moves per square; active (B,) bool, the scenes still running
        (others are left as they are). Returns per scene the positions (frames, N + K, 2) over
        the step (None for an inactive scene)."""
        paths = []
        for i, swarm in enumerate(self.swarms):
            if active is not None and not active[i]:
                paths.append(None)
                continue
            squares = self.squares[i]
            targets = targets_of(squares, codes[i][squares[:, 0], squares[:, 1]], swarm.walls)
            paths.append(swarm.run(targets))
        return paths

    def remove(self, eaten):
        """eaten (B, K) bool: cheeses the grid rule ate this step."""
        for i, k in zip(*np.nonzero(np.asarray(eaten))):
            self.swarms[i].remove(int(k))


def push_test(robots, seconds=3):
    """Push one cheese with robots lined up against its left face, all driving +x at full force,
    on an open grid (they start one square off it, as on the grid, so they hit it at speed).
    Returns how far the cheese has moved (squares) after each second."""
    size, side = 32, cheese_side(32)
    cheese = (14, 12)
    squares = np.array([[cheese[0] + k, cheese[1] - 1] for k in range(robots)])
    swarm = Swarm(np.zeros((size, size), bool), squares, np.array([cheese]), side)
    start = swarm.cheese_xy()[0, 0]
    far = squares + np.array([0, 12])  # targets well inside the cheese: full force all along
    moved = []
    for _ in range(seconds):
        swarm.run(far, seconds=1.0, frames=1)
        moved.append(float(swarm.cheese_xy()[0, 0] - start))
    return moved
