"""Render rollouts as GIFs: mass in blue, cheese in yellow, walls in gray, ring squares tinted, pheromone in green."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from beebot.ca import rollout
from beebot.evaluate import FIRE_PROB, SETS, STEPS_PER_CELL, TAU, build_split

PIXELS_PER_SQUARE = 8
BACKGROUND = np.array([18, 18, 24])
RING = np.array([60, 45, 20])
CHEESE = np.array([250, 200, 40])
WALL = np.array([95, 92, 104])
MASS = np.array([90, 170, 255])
PHEROMONE = np.array([60, 200, 120])
BAR_IN_TIME = np.array([150, 150, 165])
BAR_OVER_TIME = np.array([220, 70, 70])
BAR_HEIGHT = 4
# Videos run this many times past the eval time limit, so a swarm that is slow
# can be told apart from one that never gathers. The bar along the bottom turns
# red once the eval limit has passed.
OVERRUN = 2


def frame(mass, cheese, walls, ring, pheromone, step, limit, total):
    image = np.where(ring[..., None], RING, BACKGROUND).astype(np.float32)
    trail = np.clip(pheromone / 8, 0, 0.35)[..., None]
    image = image * (1 - trail) + PHEROMONE * trail
    level = np.clip(mass, 0, 1)[..., None]
    image = image * (1 - level) + MASS * level
    image[cheese] = CHEESE
    image[walls] = WALL
    # Squares holding more than 1 unit of mass turn white, so crowding is visible.
    over = np.clip(mass - 1, 0, 1)[..., None]
    image = image * (1 - over) + 255 * over
    image = image.astype(np.uint8).repeat(PIXELS_PER_SQUARE, axis=0).repeat(PIXELS_PER_SQUARE, axis=1)
    bar = np.zeros((BAR_HEIGHT, image.shape[1], 3), dtype=np.uint8)
    bar[:] = BACKGROUND
    bar[:, : round(image.shape[1] * step / total)] = BAR_IN_TIME if step <= limit else BAR_OVER_TIME
    return Image.fromarray(np.concatenate([image, bar]))


def save_gif(path, initial, walls, cheeses, rings, masses, pheromones, limit, n_steps):
    """One layout's rollout as a GIF. cheeses and rings hold one mask per frame, the start included."""
    frames = [frame(initial, cheeses[0], walls, rings[0], np.zeros_like(initial), 0, limit, n_steps)]
    for t, (m, p) in enumerate(zip(masses, pheromones), start=1):
        frames.append(frame(m, cheeses[t], walls, rings[t], p, t, limit, n_steps))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=60, loop=0)


def render(policy, batch, device, path, n_examples=3):
    batch = batch.to(device)
    limit = STEPS_PER_CELL * batch.mass.shape[-1]
    n_steps = OVERRUN * limit
    torch.manual_seed(0)  # same firing pattern for every policy, so videos compare like for like
    with torch.no_grad():
        result = rollout(policy, batch, n_steps, FIRE_PROB, TAU, record=True)
    cheese = batch.cheese.cpu().numpy()
    walls = batch.walls.cpu().numpy()
    ring = batch.ring.cpu().numpy()
    initial = batch.mass.cpu().numpy()
    for i in range(min(n_examples, len(cheese))):
        masses = [m[i] for m in result.masses]
        pheromones = [p[i] for p in result.pheromones]
        static = [cheese[i]] * (n_steps + 1), [ring[i]] * (n_steps + 1)
        save_gif(path.with_name(f"{path.stem}_{i}.gif"), initial[i], walls[i], *static, masses, pheromones, limit, n_steps)


def render_examples(policy, device, out_dir):
    """A few val layouts from one interp size and one extrap size."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = SETS["val"]
    interp = build_split(spec["seed"], sizes=spec["interp"]["sizes"][:1], cheese_range=spec["interp"]["cheese"])
    extrap = build_split(spec["seed"] + 1, sizes=spec["extrap"]["sizes"][:1], cheese_range=spec["extrap"]["cheese"])
    render(policy, interp[0], device, out_dir / "interp")
    render(policy, extrap[0], device, out_dir / "extrap")


def render_successes(policy, batch, device, out_dir, prefix, steps_per_cell=STEPS_PER_CELL, n_examples=3):
    """The hardest layouts the policy solves within the limit: its successes with the farthest starts.

    Returns what was picked, for the page's caption.
    """
    unreached = torch.full_like(batch.distance, float("inf"))
    start = torch.where(batch.mass > 0, batch.distance, unreached).flatten(1).min(dim=1).values
    size = batch.mass.shape[-1]
    limit = round(steps_per_cell * size)
    n_steps = OVERRUN * limit
    torch.manual_seed(0)
    with torch.no_grad():
        result = rollout(policy, batch.to(device), n_steps, FIRE_PROB, TAU, record=True)
    eaten = result.eaten[:limit].cpu()
    solved = eaten.any(dim=0)
    first = eaten.float().argmax(dim=0) + 1
    order = sorted(np.flatnonzero(solved.numpy()), key=lambda i: -start[i].item())[:n_examples]

    cheese = batch.cheese.numpy()
    ring = batch.ring.numpy()
    clips = []
    for k, i in enumerate(order):
        name = f"{prefix}_{k}.gif"
        masses = [m[i] for m in result.masses]
        pheromones = [p[i] for p in result.pheromones]
        static = [cheese[i]] * (n_steps + 1), [ring[i]] * (n_steps + 1)
        save_gif(out_dir / name, batch.mass[i].numpy(), batch.walls[i].numpy(), *static, masses, pheromones, limit, n_steps)
        clips.append({"file": name, "start": int(start[i]), "steps": int(first[i]), "limit": limit})
    return {"split": prefix, "size": size, "layouts": len(solved), "solved": int(solved.sum()), "clips": clips}


def render_demo(policy, device, out_dir, steps_per_cell=STEPS_PER_CELL):
    """Demo clips for the page: the hardest val layouts solved, from one interp size and one extrap size."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = SETS["val"]
    interp = build_split(spec["seed"], sizes=spec["interp"]["sizes"][:1], cheese_range=spec["interp"]["cheese"])
    extrap = build_split(spec["seed"] + 1, sizes=spec["extrap"]["sizes"][:1], cheese_range=spec["extrap"]["cheese"])
    return [
        render_successes(policy, interp[0], device, out_dir, "interp", steps_per_cell),
        render_successes(policy, extrap[0], device, out_dir, "extrap", steps_per_cell),
    ]
