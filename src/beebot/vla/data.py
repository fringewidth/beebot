"""Play episodes as training data: generation (scripted play), windows, batches and augmentation.

An episode store is a dict of CPU tensors:
    mass         (E, T+1, G, G) float16
    field        (E, T, G, G) int8 direction code of the applied field: 0-7 for the unit vector
                 toward DIRS[k], 8 for zero (the controller only ever emits these, so it is lossless)
    cheese_pos   (E, T+1, K, 2) int8
    cheese_alive (E, T+1, K) bool
    walls        (E, G, G) bool
    cheese_color (E, K) long;  box_rect (E, NB, 4) long;  box_color (E, NB) long
    phases       list of E lists of T labels
Windows are (episode, w0, w1) triples cut on phase boundaries.
"""

import numpy as np
import torch
import torch.nn.functional as F

from beebot.vla.controller import Controller, commands_to_tensors
from beebot.vla.scripted import ScriptedPlayer
import math

from beebot.vla.world import DIRS, World, cheese_masks, observe, sample_world, step, zoom

CHUNK = 8  # action steps the policy generates per decision
EXECUTE = 4  # of which it executes this many before deciding again
WINDOW_STEPS = (8, 48)  # at the 16-grid
EPISODE_STEPS = 64  # at the 16-grid


CODE_VECTORS = torch.tensor([[dx / math.hypot(dx, dy), dy / math.hypot(dx, dy)] for dy, dx in DIRS] + [[0.0, 0.0]])


def encode_field(field):
    """(..., 2, H, W) controller field -> (..., H, W) int8 direction code."""
    vx, vy = field[..., 0, :, :], field[..., 1, :, :]
    k = torch.remainder(torch.round(torch.atan2(vy, vx) / (math.pi / 4)), 8)
    zero = torch.sqrt(vx * vx + vy * vy) < 0.5
    return torch.where(zero, 8, k).to(torch.int8)


def decode_field(code, device=None):
    """(..., H, W) int8 code -> (..., 2, H, W) float field."""
    table = CODE_VECTORS.to(code.device if device is None else device)
    return table[code.long().to(table.device)].movedim(-1, -3)


def episode_steps(size):
    return round(EPISODE_STEPS * zoom(size))


@torch.no_grad()
def perturb_field(field, noise, generator=None):
    """Replace each square's vector, with probability noise, by one of the 8 unit directions or zero
    (uniformly). Used to execute a noisy field while recording the clean one, so the data shows
    how the player corrects from states its own clean play wouldn't reach."""
    b, _, h, w = field.shape
    hit = torch.rand(b, h, w, generator=generator, device=field.device) < noise
    codes = torch.randint(9, (b, h, w), generator=generator, device=field.device)
    random_field = decode_field(codes)
    return torch.where(hit.unsqueeze(1), random_field, field)


def generate_scripted(size, n_episodes, seed, device, batch_size=256, noise=0.0):
    """Scripted play on freshly sampled scenes. With noise > 0, the executed field is perturbed
    (perturb_field) while the stored field is the player's clean one."""
    rng = np.random.default_rng(seed)
    steps = episode_steps(size)
    parts = []
    for start in range(0, n_episodes, batch_size):
        b = min(batch_size, n_episodes - start)
        world = sample_world(rng, b, size=size).to(device)
        player = ScriptedPlayer(rng, world)
        controller = Controller()
        masses = [world.mass.half().cpu()]
        positions = [world.cheese_pos.to(torch.int8).cpu()]
        alive = [world.cheese_alive.cpu()]
        fields, phases = [], []
        cmd_points, cmd_valid, cmd_straight, cmd_share = [], [], [], []
        for t in range(steps):
            commands, labels = player.act(world, t)
            points, valid, straight, share = commands_to_tensors(commands, b, device)
            cmd_points.append(points.to(torch.int16).cpu())
            cmd_valid.append(valid.cpu())
            cmd_straight.append(straight.cpu())
            cmd_share.append(share.half().cpu())
            field = controller(world, points, valid, straight, share)
            executed = perturb_field(field, noise) if noise > 0 else field
            world, _ = step(world, executed)
            fields.append(encode_field(field).cpu())
            masses.append(world.mass.half().cpu())
            positions.append(world.cheese_pos.to(torch.int8).cpu())
            alive.append(world.cheese_alive.cpu())
            phases.append(labels)
        parts.append(
            {
                "mass": torch.stack(masses, 1),
                "field": torch.stack(fields, 1),
                "cheese_pos": torch.stack(positions, 1),
                "cheese_alive": torch.stack(alive, 1),
                "walls": world.walls.cpu(),
                "cheese_color": world.cheese_color.cpu(),
                "box_rect": world.box_rect.cpu(),
                "box_color": world.box_color.cpu(),
                "phases": [[phases[t][i] for t in range(steps)] for i in range(b)],
                # The player's commands per step (attractor points, which are in use, straight
                # mode, share or NaN): only for the command-oracle diagnostic.
                "cmd_points": torch.stack(cmd_points, 1),
                "cmd_valid": torch.stack(cmd_valid, 1),
                "cmd_straight": torch.stack(cmd_straight, 1),
                "cmd_share": torch.stack(cmd_share, 1),
            }
        )
    store = {k: torch.cat([p[k] for p in parts]) for k in parts[0] if k != "phases"}
    store["phases"] = [row for p in parts for row in p["phases"]]
    return store


def phase_boundaries(phases, min_len):
    """Step indices where the phase changes (plus 0 and the end), with phases shorter than min_len merged into the next."""
    bounds = [0] + [t for t in range(1, len(phases)) if phases[t] != phases[t - 1]] + [len(phases)]
    merged = [0]
    for b in bounds[1:]:
        if b - merged[-1] >= min_len or b == len(phases):
            merged.append(b)
    if len(merged) > 2 and merged[-1] - merged[-2] < min_len:
        merged.pop(-2)
    return merged


def make_windows(store, size):
    low, high = (round(v * zoom(size)) for v in WINDOW_STEPS)
    windows = []
    for e, phases in enumerate(store["phases"]):
        bounds = phase_boundaries(phases, low)
        for i, w0 in enumerate(bounds):
            for w1 in bounds[i + 1 :]:
                if low <= w1 - w0 <= high:
                    windows.append((e, w0, w1))
    return torch.tensor(windows, dtype=torch.long)


def world_at(store, episodes, steps, device):
    """The World of each (episode, step) pair, on device."""
    return World(
        mass=store["mass"][episodes, steps].float().to(device),
        walls=store["walls"][episodes].to(device),
        cheese_pos=store["cheese_pos"][episodes, steps].long().to(device),
        cheese_alive=store["cheese_alive"][episodes, steps].to(device),
        cheese_color=store["cheese_color"][episodes].to(device),
        box_rect=store["box_rect"][episodes].to(device),
        box_color=store["box_color"][episodes].to(device),
    )


# ---------------------------------------------------------------------------------------------
# D4 transforms (rotations by 90 degrees, optionally followed by a left-right flip)


def transform_grid(x, k, flip):
    """Rotate (..., H, W) grids k quarter turns counterclockwise, then flip left-right if flip."""
    x = torch.rot90(x, k, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if flip else x


def transform_field(field, k, flip):
    """Fields (..., 2, H, W) with channel 0 = column (x) and 1 = row (y, downward) components."""
    field = transform_grid(field, k, flip)
    vx, vy = field[..., 0, :, :], field[..., 1, :, :]
    for _ in range(k % 4):
        vx, vy = vy, -vx  # a quarter turn counterclockwise: right becomes up
    if flip:
        vx = -vx
    return torch.stack([vx, vy], dim=-3)


def transform_world(world, k, flip):
    """The same scene seen through the transform (used by tests and evaluation)."""
    from beebot.vla.world import box_masks, cheese_masks

    def corner(mask):
        rows = mask.any(dim=-1).float().argmax(dim=-1)
        cols = mask.any(dim=-2).float().argmax(dim=-1)
        return rows, cols

    cm = transform_grid(cheese_masks(world), k, flip)
    rows, cols = corner(cm)
    pos = torch.where(world.cheese_alive.unsqueeze(-1), torch.stack([rows, cols], -1), world.cheese_pos)
    bm = transform_grid(box_masks(world), k, flip)
    brows, bcols = corner(bm)
    heights = bm.any(dim=-1).sum(-1)
    widths = bm.any(dim=-2).sum(-1)
    rect = torch.stack([brows, bcols, heights, widths], -1)
    rect = torch.where((world.box_color >= 0).unsqueeze(-1), rect, world.box_rect)
    return World(
        mass=transform_grid(world.mass, k, flip),
        walls=transform_grid(world.walls, k, flip),
        cheese_pos=pos,
        cheese_alive=world.cheese_alive,
        cheese_color=world.cheese_color,
        box_rect=rect,
        box_color=world.box_color,
    )


# ---------------------------------------------------------------------------------------------
# Batches


def reachable_weights(mass, walls, chunk=CHUNK, far_weight=0.1):
    """(B, chunk, G, G) loss weight: 1 on squares the step-0 mass could reach by chunk step k
    (dilated by k + 1), far_weight elsewhere, 0 on walls."""
    occupied = (mass > 1e-3).float().unsqueeze(1)
    weights = []
    for k in range(chunk):
        r = k + 1
        near = F.max_pool2d(occupied, 2 * r + 1, stride=1, padding=r) > 0
        weights.append(torch.where(near, 1.0, far_weight).squeeze(1))
    w = torch.stack(weights, 1)
    return w * (~walls).float().unsqueeze(1)


COMMAND_CHANNELS = 3


def command_map(store, e, t, size, device):
    """(B, 3, G, G) image of the player's commands at (episode, step): attractor points, their
    share (0 if none given) and their straight flag, each drawn on the attractor's square."""
    points = store["cmd_points"][e, t].long()  # B, A, 2
    valid = store["cmd_valid"][e, t]
    share = torch.nan_to_num(store["cmd_share"][e, t].float(), nan=0.0)
    straight = store["cmd_straight"][e, t].float()
    b, a = valid.shape
    flat = (points[..., 0].clamp(0, size - 1) * size + points[..., 1].clamp(0, size - 1))
    out = torch.zeros(b, COMMAND_CHANNELS, size * size)
    for c, value in enumerate((valid.float(), share * valid, straight * valid)):
        out[:, c].scatter_add_(1, flat, value)
    return out.view(b, COMMAND_CHANNELS, size, size).to(device)


def sample_batch(store, windows, batch_size, generator, device, augment=True, hold_prob=0.0, commands=False, window_weights=None):
    """A goal-conditioned training batch from randomly picked windows (uniformly, or in proportion
    to window_weights), each with a random colour permutation and a random rotation or reflection
    if augment. See batch_from for the shapes and hold_prob."""
    if window_weights is None:
        pick = windows[torch.randint(len(windows), (batch_size,), generator=generator)]
    else:
        pick = windows[torch.multinomial(window_weights, batch_size, replacement=True, generator=generator)]
    perm = group = None
    if augment:
        perm = torch.stack([torch.randperm(4, generator=generator) for _ in range(batch_size)])
        group = torch.randint(8, (batch_size,), generator=generator)
    hold = torch.rand(batch_size, generator=generator) < hold_prob if hold_prob > 0 else None
    return batch_from(store, pick, generator, device, perm, group, hold, commands)


FUTURE_EMPTY, FUTURE_ROBOT, FUTURE_CHEESE = 0, 1, 2
N_FUTURE = 3


def square_states(world):
    """(B, H, W) long: FUTURE_CHEESE under a cheese, else FUTURE_ROBOT where a robot is, else FUTURE_EMPTY."""
    robot = (world.mass > 0.5).long() * FUTURE_ROBOT
    return torch.where(cheese_masks(world).any(dim=1), torch.full_like(robot, FUTURE_CHEESE), robot)


def batch_from(store, pick, generator, device, perm=None, group=None, hold=None, commands=False):
    """Training samples for given windows pick (B, 3) = (episode, w0, w1), the current step drawn
    uniformly from [w0, w1 - 1].

    perm (B, 4): colour index c becomes perm[c] (None: unchanged). group (B,): g in 0..7 means
    g % 4 quarter turns counterclockwise, then a left-right flip if g >= 4 (None: unchanged).
    hold (B,) bool: those samples start at the goal itself (the window's end) and their whole
    chunk is labelled with the zero field: once the goal is reached, stay. commands: append the
    player's commands at the current step to obs (command_map; the command-oracle diagnostic).
    Returns obs (B, 10, G, G) at the current step, goal (B, 20, G, G) = window start and end
    frames, actions (B, CHUNK, 2, G, G), weights (B, CHUNK, G, G) (chunk steps past the window's
    end, or the episode's, get weight 0) and future (B, CHUNK, G, G) long: each square's state
    after each chunk step, FUTURE_EMPTY, FUTURE_ROBOT or FUTURE_CHEESE (the world-model target).
    """
    batch_size = len(pick)
    e, w0, w1 = pick[:, 0], pick[:, 1], pick[:, 2]
    s = w0 + (torch.rand(batch_size, generator=generator) * (w1 - w0).float()).long()
    if hold is not None:
        s = torch.where(hold, w1, s)
    steps = store["field"].shape[1]
    offsets = torch.arange(CHUNK)
    chunk_t = (s.unsqueeze(1) + offsets).clamp(max=steps - 1)
    inside = (s.unsqueeze(1) + offsets) < w1.unsqueeze(1)
    if hold is not None:
        inside = inside | hold.unsqueeze(1)

    worlds = []
    for t in (s, w0, w1):
        w = world_at(store, e, t, device)
        if perm is not None:
            p = perm.to(device)
            w.cheese_color = torch.where(w.cheese_color >= 0, p.gather(1, w.cheese_color.clamp(min=0)), w.cheese_color)
            w.box_color = torch.where(w.box_color >= 0, p.gather(1, w.box_color.clamp(min=0)), w.box_color)
        worlds.append(w)
    obs = observe(worlds[0])
    if commands:
        obs = torch.cat([obs, command_map(store, e, s.clamp(max=steps - 1), obs.shape[-1], device)], dim=1)
    goal = torch.cat([observe(worlds[1]), observe(worlds[2])], dim=1)
    future = torch.stack([square_states(world_at(store, e, (chunk_t[:, k] + 1).clamp(max=steps), device)) for k in range(CHUNK)], dim=1)
    if hold is not None:
        future = torch.where(hold.to(device).view(-1, 1, 1, 1), square_states(worlds[2]).unsqueeze(1).expand_as(future), future)
    actions = decode_field(store["field"][e.unsqueeze(1), chunk_t], device)
    if hold is not None:
        actions = torch.where(hold.to(device).view(-1, 1, 1, 1, 1), torch.zeros_like(actions), actions)
    weights = reachable_weights(worlds[0].mass, worlds[0].walls) * inside.to(device).float().view(batch_size, CHUNK, 1, 1)

    if group is not None:
        for g in range(1, 8):
            sel = (group == g).to(device)
            if not sel.any():
                continue
            k, flip = g % 4, g >= 4
            obs[sel] = transform_grid(obs[sel], k, flip)
            goal[sel] = transform_grid(goal[sel], k, flip)
            actions[sel] = transform_field(actions[sel], k, flip)
            weights[sel] = transform_grid(weights[sel], k, flip)
            future[sel] = transform_grid(future[sel], k, flip)
    return obs, goal, actions, weights, future
