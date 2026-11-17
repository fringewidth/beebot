"""Render recorded episodes as GIFs: terrain gray, boxes as tinted floor with a colored outline,
cheeses as solid colored blocks, swarm mass as white intensity, with an optional caption bar."""

import numpy as np
from PIL import Image, ImageDraw

from beebot.vla.world import COLOUR_RGB, cheese_side

PIXELS = 256  # image side, whatever the grid size
FLOOR = np.array([24, 24, 30], dtype=np.float32)
WALL = np.array([96, 92, 104], dtype=np.float32)
SWARM = np.array([240, 240, 245], dtype=np.float32)
CAPTION_HEIGHT = 18


def frame(episode, i, t, caption=None):
    walls = episode["walls"][i]
    size = walls.shape[0]
    side = cheese_side(size)
    SCALE = max(4, PIXELS // size)
    palette = (COLOUR_RGB.numpy() * 255).astype(np.float32)
    image = np.broadcast_to(FLOOR, (size, size, 3)).copy()
    outlines = []
    for rect, color in zip(episode["box_rect"][i], episode["box_color"][i]):
        if color < 0:
            continue
        y, x, h, w = rect
        image[y : y + h, x : x + w] = 0.75 * FLOOR + 0.25 * palette[color]
        outlines.append((y, x, h, w, tuple(int(v) for v in palette[color])))
    image[walls] = WALL
    level = np.clip(episode["mass"][i, t].astype(np.float32), 0, 1)[..., None]
    image = image * (1 - level) + SWARM * level
    for k, color in enumerate(episode["cheese_color"][i]):
        if color < 0 or not episode["cheese_alive"][i, t, k]:
            continue
        y, x = episode["cheese_pos"][i, t, k]
        image[y : y + side, x : x + side] = palette[color]
    big = Image.fromarray(image.astype(np.uint8)).resize((size * SCALE, size * SCALE), Image.NEAREST)
    draw = ImageDraw.Draw(big)
    for y, x, h, w, rgb in outlines:
        draw.rectangle([x * SCALE, y * SCALE, (x + w) * SCALE - 1, (y + h) * SCALE - 1], outline=rgb, width=2)
    if caption is None:
        return big
    canvas = Image.new("RGB", (big.width, big.height + CAPTION_HEIGHT), (12, 12, 16))
    canvas.paste(big, (0, 0))
    ImageDraw.Draw(canvas).text((4, big.height + 3), f"{caption}  t={t}", fill=(220, 220, 220))
    return canvas


def save_gif(path, episode, i, t0=0, t1=None, caption=None, ms_per_frame=120):
    """One episode (index i), steps t0..t1 inclusive, as a GIF."""
    t1 = episode["mass"].shape[1] - 1 if t1 is None else t1
    frames = [frame(episode, i, t, caption) for t in range(t0, t1 + 1)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms_per_frame, loop=0)
