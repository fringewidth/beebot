"""Isometric voxel rendering with PIL: one scene to an image, an episode to a GIF.

Each voxel is drawn as three shaded faces (top, left, right) in painter's order (far to near):
x increases to the lower right, y to the lower left, z up. Robots are grey, walls are sandy
(walls covered on all three visible sides are skipped), cheeses take their
palette colour.
"""

import numpy as np
from PIL import Image, ImageDraw

from beebot.vla.voxel.world import CHEESE_SIDE, PALETTE_RGB

ROBOT = (70, 80, 95)
WALL = (196, 178, 140)
BACKGROUND = (246, 246, 243)
SHADES = (1.0, 0.78, 0.62)  # top, left, right


def _shade(colour, k):
    return tuple(int(c * k) for c in colour)


def render_scene(mass, walls, cheese_pos=(), cheese_alive=(), cheese_color=(), scale=8):
    """mass, walls (D, H, W) numpy; cheeses as (K, 3), (K,), (K,). Returns a PIL image."""
    d, h, w = mass.shape
    colour = {}
    hidden = np.ones_like(walls)  # walls covered on the three visible sides (+z, +y, +x)
    hidden[:-1] &= walls[1:]
    hidden[:, :-1] &= walls[:, 1:]
    hidden[:, :, :-1] &= walls[:, :, 1:]
    hidden[-1] = False
    hidden[:, -1] = False
    hidden[:, :, -1] = False
    exposed = walls & ~hidden
    for z, y, x in np.argwhere(exposed):
        colour[(z, y, x)] = WALL
    for p, alive, c in zip(cheese_pos, cheese_alive, cheese_color):
        if alive and c >= 0:
            rgb = tuple(int(v * 255) for v in PALETTE_RGB[int(c)].tolist())
            z0, y0, x0 = (int(v) for v in p)
            for z in range(z0, z0 + CHEESE_SIDE):
                for y in range(y0, y0 + CHEESE_SIDE):
                    for x in range(x0, x0 + CHEESE_SIDE):
                        colour[(z, y, x)] = rgb
    for z, y, x in np.argwhere(mass > 0.5):
        colour[(z, y, x)] = ROBOT
    s = scale
    width = (w + h) * s + 2 * s
    height = (w + h) * s // 2 + d * s + 2 * s
    ox, oy = h * s + s, d * s + s
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    # The floor outline.
    corners = [(0, 0), (0, w), (h, w), (h, 0)]
    draw.polygon([(ox + (x - y) * s, oy + (x + y) * s / 2) for y, x in corners], outline=(200, 200, 195), fill=(236, 236, 230))
    for (z, y, x), rgb in sorted(colour.items(), key=lambda item: (sum(item[0]), item[0][0])):
        cx, cy = ox + (x - y) * s, oy + (x + y) * s / 2 - z * s
        top = [(cx, cy - s), (cx + s, cy - s / 2), (cx, cy), (cx - s, cy - s / 2)]
        left = [(cx - s, cy - s / 2), (cx, cy), (cx, cy + s), (cx - s, cy + s / 2)]
        right = [(cx, cy), (cx + s, cy - s / 2), (cx + s, cy + s / 2), (cx, cy + s)]
        draw.polygon(top, fill=_shade(rgb, SHADES[0]))
        draw.polygon(left, fill=_shade(rgb, SHADES[1]))
        draw.polygon(right, fill=_shade(rgb, SHADES[2]))
    return image


def render_episode(store, e, path, every=1, scale=8, duration=120):
    """Write episode e of a store (voxel/scripted.generate) as a GIF (or a PNG for one frame)."""
    frames = []
    for t in range(0, store["mass"].shape[1], every):
        frames.append(
            render_scene(
                store["mass"][e, t].numpy(),
                store["walls"][e].numpy(),
                store["cheese_pos"][e, t].numpy(),
                store["cheese_alive"][e, t].numpy(),
                store["cheese_color"][e].numpy(),
                scale,
            )
        )
    if str(path).endswith(".png") or len(frames) == 1:
        frames[-1].save(path)
    else:
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    return path
