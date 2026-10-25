"""Render rollouts as GIFs: mass in blue, cheese in yellow, ring squares tinted, pheromone in green."""

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
MASS = np.array([90, 170, 255])
PHEROMONE = np.array([60, 200, 120])


def frame(mass, cheese, ring, pheromone):
    image = np.where(ring[..., None], RING, BACKGROUND).astype(np.float32)
    trail = np.clip(pheromone / 4, 0, 0.6)[..., None]
    image = image * (1 - trail) + PHEROMONE * trail
    level = np.clip(mass, 0, 1)[..., None]
    image = image * (1 - level) + MASS * level
    image[cheese] = CHEESE
    # Squares holding more than 1 unit of mass turn white, so crowding is visible.
    over = np.clip(mass - 1, 0, 1)[..., None]
    image = image * (1 - over) + 255 * over
    image = image.astype(np.uint8).repeat(PIXELS_PER_SQUARE, axis=0).repeat(PIXELS_PER_SQUARE, axis=1)
    return Image.fromarray(image)


def render(policy, batch, device, path, n_examples=3):
    batch = batch.to(device)
    n_steps = STEPS_PER_CELL * batch.mass.shape[-1]
    torch.manual_seed(0)  # same firing pattern for every policy, so videos compare like for like
    with torch.no_grad():
        result = rollout(policy, batch, n_steps, FIRE_PROB, TAU, record=True)
    cheese = batch.cheese.cpu().numpy()
    ring = batch.ring.cpu().numpy()
    initial = batch.mass.cpu().numpy()
    for i in range(min(n_examples, len(cheese))):
        frames = [frame(initial[i], cheese[i], ring[i], np.zeros_like(initial[i]))]
        frames += [frame(m[i], cheese[i], ring[i], p[i]) for m, p in zip(result.masses, result.pheromones)]
        frames[0].save(path.with_name(f"{path.stem}_{i}.gif"), save_all=True, append_images=frames[1:], duration=60, loop=0)


def render_examples(policy, device, out_dir):
    """A few val layouts from one interp size and one extrap size."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = SETS["val"]
    interp = build_split(spec["seed"], sizes=spec["interp"]["sizes"][:1], cheese_range=spec["interp"]["cheese"])
    extrap = build_split(spec["seed"] + 1, sizes=spec["extrap"]["sizes"][:1], cheese_range=spec["extrap"]["cheese"])
    render(policy, interp[0], device, out_dir / "interp")
    render(policy, extrap[0], device, out_dir / "extrap")
