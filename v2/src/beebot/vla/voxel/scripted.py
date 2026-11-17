"""Scripted play in the voxel world: random task sequences per episode, executed by the attractor
controller, recorded as a store like the 2D one (vla/data.generate_scripted).

Each scene runs its own sequence of segments of SEGMENT_STEPS steps; each segment draws an intent:

    goto     every robot heads for one random voxel (on the ground under gravity)
    gather   every robot heads for the swarm's centroid
    split    robots head for the nearer of two random voxels
    push     robots crowd behind a cheese and press it along one axis (horizontal under gravity)
    eat      robots fill a cheese's shell
    tower    robots fill a 2 x 2 column of a random height on level ground (gravity only)
    bridge   robots fill a span across the trench, level with the plateau tops (gravity, gap layout)
    stay     nobody moves

    from beebot.vla.voxel.scripted import generate
    store = generate(size=16, n_episodes=256, seed=0)
"""

import numpy as np
import torch

from beebot.vla.voxel.controller import act
from beebot.vla.voxel.world import CHEESE_SIDE, SIZE, _face, sample_scenes, shell, step

INTENTS = ("goto", "gather", "split", "push", "eat", "tower", "bridge", "stay")
INTENT_WEIGHTS = np.array([2, 1, 2, 3, 2, 2, 2, 0.5])
SEGMENT_STEPS = (8, 24)  # at the 16-grid
EPISODE_STEPS = 64  # at the 16-grid
MAX_TARGETS = 64
TOWER_HEIGHTS = (3, 8)


class ScriptedPlayer:
    def __init__(self, rng, world, meta):
        self.rng, self.meta = rng, meta
        b = world.mass.shape[0]
        self.size = world.mass.shape[-1]
        self.intent = ["stay"] * b
        self.ends = np.zeros(b, dtype=int)
        self.cheese = torch.full((b,), -1, dtype=torch.long)
        self.direction = torch.zeros(b, 3, dtype=torch.long)
        self.targets = torch.zeros(b, MAX_TARGETS, 3, dtype=torch.long)
        self.valid = torch.zeros(b, MAX_TARGETS, dtype=torch.bool)

    def ground(self, i, y, x, world):
        """The lowest free voxel above the terrain in column (y, x)."""
        return int(self.meta[i]["height"][y, x])

    def random_point(self, i, world):
        rng, n = self.rng, self.size
        for _ in range(100):
            y, x = int(rng.integers(n)), int(rng.integers(n))
            z = self.ground(i, y, x, world) if world.gravity[i] else int(rng.integers(n))
            if z < n and not world.walls[i, z, y, x]:
                return (z, y, x)
        return (n - 1, 0, 0)

    def set_points(self, i, points):
        self.valid[i] = False
        for j, p in enumerate(points[:MAX_TARGETS]):
            self.targets[i, j] = torch.tensor(p)
            self.valid[i, j] = True

    def begin(self, i, world, t):
        rng = self.rng
        alive = world.cheese_alive[i].nonzero().squeeze(1).tolist()
        gravity = bool(world.gravity[i])
        allowed = np.array([
            True,
            True,
            True,
            bool(alive),
            bool(alive),
            gravity,
            gravity and self.meta[i]["gap"] is not None,
            True,
        ])
        weights = INTENT_WEIGHTS * allowed
        intent = INTENTS[rng.choice(len(INTENTS), p=weights / weights.sum())]
        scale = self.size / SIZE
        self.ends[i] = t + round(rng.integers(SEGMENT_STEPS[0], SEGMENT_STEPS[1] + 1) * scale)
        self.intent[i] = intent
        self.cheese[i] = -1
        self.valid[i] = False
        if intent == "goto":
            self.set_points(i, [self.random_point(i, world)])
        elif intent == "split":
            self.set_points(i, [self.random_point(i, world), self.random_point(i, world)])
        elif intent == "gather":
            zyx = world.mass[i].nonzero().float().mean(dim=0).round().long().tolist()
            if gravity:
                zyx[0] = self.ground(i, zyx[1], zyx[2], world)
            self.set_points(i, [tuple(zyx)])
        elif intent in ("push", "eat"):
            self.cheese[i] = int(rng.choice(alive))
            axes = (1, 2) if gravity else (0, 1, 2)
            self.direction[i] = 0
            self.direction[i, int(rng.choice(axes))] = int(rng.choice((-1, 1)))
        elif intent == "tower":
            n = self.size
            height = self.meta[i]["height"]
            for _ in range(100):
                y, x = int(rng.integers(n - 1)), int(rng.integers(n - 1))
                foot = height[y : y + 2, x : x + 2]
                z0 = int(foot[0, 0])
                if (foot == z0).all() and z0 + TOWER_HEIGHTS[0] <= n:
                    break
            top = min(n, z0 + int(rng.integers(TOWER_HEIGHTS[0], TOWER_HEIGHTS[1] + 1)))
            self.set_points(i, [(z, y + a, x + c) for z in range(z0, top) for a in range(2) for c in range(2)])
        elif intent == "bridge":
            transposed, start, width, top = self.meta[i]["gap"]
            lane = int(rng.integers(self.size - 1))
            points = [(top - 1, lane + a, x) for a in range(2) for x in range(start, start + width)]
            if transposed:
                points = [(z, x, y) for z, y, x in points]
            self.set_points(i, points)

    def act(self, world, t, generator=None):
        """Action codes (B, D, H, W) and this step's intent label per scene."""
        b = world.mass.shape[0]
        for i in range(b):
            dead = self.cheese[i] >= 0 and not world.cheese_alive[i, self.cheese[i]]
            if t >= self.ends[i] or dead:
                self.begin(i, world, t)
        targets, valid = self.targets.clone(), self.valid.clone()
        press = torch.full((b,), -1, dtype=torch.long)
        k = self.cheese.clamp(min=0)
        rows = torch.arange(b)
        eating = torch.tensor([x == "eat" for x in self.intent])
        if eating.any():
            coords, open_ = shell(world)
            n = coords.shape[2]
            targets[eating, :n] = coords[rows, k][eating]
            valid[eating] = False
            valid[eating, :n] = open_[rows, k][eating]
        pushing = torch.tensor([x == "push" for x in self.intent])
        if pushing.any():
            pos = world.cheese_pos[rows, k]
            axis = self.direction.abs().argmax(dim=1)
            sign = self.direction.gather(1, axis.view(-1, 1)).squeeze(1)
            for a in range(3):
                sel = pushing & (axis == a)
                if sel.any():
                    back = _face(pos, a, torch.where(sign > 0, -1, CHEESE_SIDE))
                    targets[sel, :4] = back[sel]
                    valid[sel] = False
                    valid[sel, :4] = True
            press = torch.where(pushing, self.cheese, press)
        codes = act(world, targets, valid, press, self.direction, generator=generator)
        return codes, list(self.intent)


def episode_steps(size):
    return round(EPISODE_STEPS * size / SIZE)


@torch.no_grad()
def generate(size=SIZE, n_episodes=64, seed=0, batch_size=64, steps=None, gravity_prob=0.5):
    """Scripted play on freshly sampled scenes, on the CPU. The store:

    mass (E, T + 1, D, H, W) uint8, actions (E, T, D, H, W) int8 codes 0..26 (STAY = 26 where no
    robot is), cheese_pos (E, T + 1, K, 3) int8, cheese_alive (E, T + 1, K) bool, walls
    (E, D, H, W) bool, cheese_color (E, K), gravity (E,) bool, phases (E lists of T intents)."""
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    steps = steps or episode_steps(size)
    parts = []
    for start in range(0, n_episodes, batch_size):
        b = min(batch_size, n_episodes - start)
        world, meta = sample_scenes(rng, b, size, gravity_prob)
        player = ScriptedPlayer(rng, world, meta)
        masses, actions, positions, alive, phases = [world.mass.to(torch.uint8)], [], [world.cheese_pos.to(torch.int8)], [world.cheese_alive], []
        for t in range(steps):
            codes, labels = player.act(world, t, generator)
            score = torch.rand(world.mass.shape, generator=generator)
            world, _ = step(world, codes.long(), score)
            actions.append(codes)
            masses.append(world.mass.to(torch.uint8))
            positions.append(world.cheese_pos.to(torch.int8))
            alive.append(world.cheese_alive)
            phases.append(labels)
        parts.append(
            {
                "mass": torch.stack(masses, 1),
                "actions": torch.stack(actions, 1),
                "cheese_pos": torch.stack(positions, 1),
                "cheese_alive": torch.stack(alive, 1),
                "walls": world.walls,
                "cheese_color": world.cheese_color,
                "gravity": world.gravity,
                "phases": [[phases[t][i] for t in range(steps)] for i in range(b)],
            }
        )
    store = {key: torch.cat([p[key] for p in parts]) for key in parts[0] if key != "phases"}
    store["phases"] = [row for p in parts for row in p["phases"]]
    return store
