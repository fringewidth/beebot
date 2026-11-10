"""Camera views of a voxel observation with visual domain randomization (README "Views and
randomization").

    from beebot.vla.voxel.views import scene_views, ViewSpec
    images, pose = scene_views(obs, rng, ViewSpec())

obs is one (C, D, H, W) observation (world.observe, any channel count from 6 to 10; numpy or
torch), rng a numpy Generator. images is one PIL image per camera of spec.cameras (default: the
oblique perspective view, then the top-down view), each spec.size square (default 256). pose
says where the cameras were (below). ViewSpec.clean() turns every randomization off.

Rendering is numpy only: every exposed voxel face that faces the camera is splatted over its
projected pixel box, the exact ray-plane hit of each pixel decides coverage and depth, and a
z-buffer (np.minimum.at) keeps the nearest face. The floor is one plane hit per pixel. That gives
per pixel the face (kind, voxel, normal), its (u, v) on the face and the world point, from which
materials, textures, lights, shadows (a shadow map from the main light, the same splatter) and
outlines are computed. Image-space effects (shadows of a hand or a cable, lens distortion, crop,
resolution, chromatic aberration, blur, noise, JPEG, cutout) follow.

World frame. x to the right, y to the front, z up, as seen by a person standing at the front
(world.py, suite.FLOOR_REGIONS: "back" is small y, "left" small x). A voxel (z, y, x) fills
[x, x+1] x [y, y+1] x [z, z+1]. The arena is the D x H x W grid.

Pose. pose["azimuth"] is the oblique camera's place around the arena in degrees: 0 at the front
(+y side, looking at the arena, as the person above), 90 at the right (+x), 180 at the back, 270
at the left. With camera randomization on it is drawn uniformly within 30 degrees of a side
(0, 90, 180, 270 +- 30), never within 15 degrees of a diagonal (45, 135, 225, 315), where front
and right are ambiguous to a person; pose["side"] names the nearest side. The top-down view is
turned with the same azimuth (plus its own small jitter): its image "up" points away from the
oblique camera, so both images agree on left and right. pose["views"] holds each camera's
projection, azimuth, elevation, field of view, position and target.

Clean mode (strength 0): oblique camera at azimuth 0, elevation 45 degrees, 50 degree field of
view, framed on the arena; orthographic top view at azimuth 0 (back at the top of the image);
lit style with one light and shadows; a dark slate sheet (CLEAN_FLOOR) whose colour is far (CIE
Lab distance at least 30) from every object colour (the zero-shot probe missed white cheeses on a
light floor); mid-grey robot cubes with dark edge lines; sandy terrain; object colours exact; no
noise. The top view shades faces by height (darker low, full brightness at 6 voxels and above)
and outlines every height step, so stacks and pyramids read from above.

Always drawn, randomized in appearance only, never removed: faint grid lines on every arena cell
boundary and the arena border, and four corner markers on the sheet just outside the arena's
corners, so pixels register to voxels from any viewpoint.

Randomizations. Each group has a strength in [0, 1] (ViewSpec field, None: the global
spec.strength); 0 turns the group off and gives the clean value. Every range below is given at
strength s and shrinks linearly to the clean value as s goes to 0; a probability p means p * s.
Any single item can be turned off by name with ViewSpec(off={"jpeg", ...}).

camera (episode-level, Look)
    azimuth           uniform within 30 deg of a side (see Pose); clean 0. Not scaled by s.
    elevation         45 +- 20 s deg (oblique camera above the arena, a phone on a stand)
    fov               50 +- 15 s deg (vertical); the camera distance is refit so the arena
                      (and FRAME_HEIGHT voxels above it) fills the image with a 7% margin
    distance          x U(1 - 0.05 s, 1 + 0.4 s) of the fitted distance
    look_at           target shifted by +-1.5 s cells in x and y, +-0.5 s in z
    roll              +-6 s deg about the optical axis
    aspect            focal lengths fx/fy differ by +-4 s %; principal point +-3 s % of the image
    distortion        p 0.6: radial k1 in U(-0.04 s, 0.22 s) (barrel mostly, some pincushion),
                      the image refit so its corners stay in view
    crop              zoom in x U(1, 1 + 0.1 s), shifted anywhere the crop stays inside the
                      render (zooming out is the distance item)
    resolution        rendered at U(1 - 0.5 s, 1) x size, upsampled to size (a low-res sensor)
    top_jitter        top view: rotation +-6 s deg on top of the azimuth, tilt up to 8 s deg
    top_perspective   p 0.5: the top view is a perspective camera high above (fov 20-45 deg),
                      else orthographic
lighting (episode-level)
    lights            1 light; p 0.5 each for a second and a third weaker light (0.2-0.6)
    light_direction   main light azimuth uniform, elevation 60 +- 30 s deg; clean: from the
                      camera's front left at 60 deg
    intensity         0.75 x U(1 - 0.35 s, 1 + 0.35 s)
    ambient           0.4 +- 0.22 s, tinted toward the light colour or a cool sky
    temperature       colour temperature 5500 +- 2700 s K for each light
    shadows           shadow map from the main light; p 0.35 off; 1 or 4 taps (clean 4)
    specular          Blinn-Phong highlights: robots ks U(0, 0.55 s) shininess 8-64, steel
                      floor ks U(0.1, 0.5), objects ks up to 0.15 s
style (episode-level, per view)
    style             p 0.3 "flat" (no lighting, fixed shade per face direction, as the old
                      isometric renderer), p 0.2 "toon" (three-band lighting, dark outlines), else
                      "lit"; outlines p 0.35 in "lit" and "flat" (always in clean)
materials (episode-level)
    floor             sheet material: steel (grey, fine anisotropic grain, smudges, sheen),
                      brushed metal (strong streaks), wood (planks, rings), paper (fine grain,
                      p 0.3 faint graph lines), noise (two-colour fractal), tiles (rotated,
                      grout); colour drawn per type, low saturation; resampled until its mean is
                      at least 20 Lab units from every object colour and 15 from the robots
    table             the surface beyond the sheet (same types, colours at least 25 Lab units
                      from every named colour, fading to the background with distance); sheet
                      margin 0.6-3 cells (clean 1.2)
    terrain           sand, wood, stone, concrete, cardboard, foam; side faces 0.75-1 of the
                      top; faint cell edges; at least 15 Lab units from every object colour
    robots            base grey 0.12-0.85 (tint up to 0.08 saturation), at least 15 Lab units
                      from every object colour; edge lines width 0.03-0.12 of a face, darker or
                      lighter than the base; p 0.5 face markings (dot, ring, cross, square,
                      stripe); per-cube brightness variation +-6 s %
    object_noise      object surface texture amplitude up to 0.12 s; food (channel 6) always
                      shows darker round holes (radius 0.1-0.2, darkness 0.5-0.8) so cheese reads
                      apart from passive blocks
    grid_lines        always drawn: colour lighter or darker than the sheet, opacity 0.15-0.5
                      (clean 0.35), width 0.04-0.1 cells
    markers           always drawn: L bracket, square, dot or cross, 0.4-1 cells, high contrast
                      to the sheet (clean: white L brackets)
colour (episode-level)
    colour_jitter     each named colour (objects and zones) gets a hue shift, saturation and
                      value factors within COLOUR_JITTER[name] x s, so the jittered colour's
                      nearest named colour (CIE Lab) is still its own (tested)
distractors
    clutter           Poisson(4 s) things on the table outside the sheet, never cube-like: papers
                      (1-4 cells, 0.02-0.08 high), books (0.2-0.6 high), cable- or pen-like bars
                      (2-7 long, 0.15-0.45 thick); they cast shadows
    stickers          Poisson(5 s) flat ellipses, rectangles and stripes on the table
    background        procedural backdrop where rays miss the floor: gradient plus fractal blobs
                      Every distractor, table and backdrop colour has saturation up to 0.3-0.4 and
                      is at least DISTRACTOR_GAP (25) Lab units from every named colour, so nothing
                      outside the arena looks like an object an instruction can name.
    hand_shadow       per image, p 0.3: a soft hand shadow (palm and fingers) entering from an
                      edge, darkening by 0.25-0.5
    cable_shadow      per image, p 0.3: a soft curved cable shadow, darkening by 0.25-0.5
corruptions (per image)
    gaussian_noise    p 0.8: sigma U(0, 0.035 s)
    shot_noise        p 0.6: variance k x intensity, k U(0, 0.02 s)
    motion_blur       p 0.25: a line kernel of 2 to 2 + 6 s pixels at any angle
    defocus           p 0.3: Gaussian blur radius U(0.3, 0.3 + 1.3 s) pixels
    jpeg              p 0.6: quality U(95 - 65 s, 95)
    vignetting        p 0.5: corners darkened by U(0, 0.45 s)
    exposure          p 0.8: x 2^U(-0.6 s, 0.6 s)
    white_balance     p 0.6: per-channel gains U(1 - 0.1 s, 1 + 0.1 s)
    gamma             p 0.6: U(1 - 0.25 s, 1 + 0.25 s)
    chromatic         p 0.4: red and blue sampled at radial scales 1 +- U(0, 0.006 s)
    cutout            p 0.4: 1-3 patches of 4-12% of the side, filled with noise or a flat
                      unnameable colour (as distractors), placed only where they cover no
                      object or zone pixel (so a target is never erased from any view)

State noise (perturb_state, a separate function on the voxel observation, default off: nothing
calls it unless the caller does): each robot is removed with probability `drop` (default
0.02), moved to a free face neighbour with probability `shift` (0.02), and a phantom robot is
added next to something solid for each robot with probability `phantom` (0.01), as imperfect
state estimation on real hardware would.

Episodes. sample_look(rng, spec) draws every episode-level choice (cameras, lights, materials,
colours, clutter); pass the same look to scene_views for every frame of an episode (a phone on a
stand does not move between frames). Per-image effects (hand and cable shadows, corruptions) are
drawn on every call. Without a look, scene_views draws a new one each call.

DEFAULT_STRENGTH is the strongest setting at which the zero-shot Qwen3.5-4B probe
(scripts/vla_view_probe.py) stays close to clean (README "Views and randomization").
"""

import io
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter

from beebot.vla.voxel.world import COLOUR_NAMES, COLOUR_RGB

DEFAULT_STRENGTH = 0.5
GROUPS = ("camera", "lighting", "style", "materials", "colour", "distractors", "corruptions")
CAMERAS = ("perspective", "top")
FRAME_HEIGHT = 4  # voxels above the floor the oblique camera keeps in frame
PALETTE = COLOUR_RGB.numpy().astype(np.float32)  # (9, 3) object and zone colours, COLOUR_NAMES order

CLEAN_FLOOR = np.array([0.20, 0.215, 0.24], np.float32)  # dark slate sheet
CLEAN_TABLE = np.array([0.80, 0.79, 0.76], np.float32)
CLEAN_ROBOT = np.array([0.56, 0.59, 0.64], np.float32)
CLEAN_TERRAIN = np.array([0.77, 0.70, 0.55], np.float32)
CLEAN_GRID = 0.35  # opacity of the grid lines (lighter than the sheet)

# Colour jitter bounds per named colour at strength 1: (hue shift in degrees, saturation factor
# range, value factor range, largest saturation added (white only)). Checked by
# test_colour_jitter_keeps_names: the nearest named colour of any jittered colour is its own.
COLOUR_JITTER = {
    "red": (6.0, (0.82, 1.1), (0.75, 1.08), 0.0),
    "green": (9.0, (0.78, 1.1), (0.72, 1.1), 0.0),
    "blue": (8.0, (0.75, 1.1), (0.75, 1.06), 0.0),
    "yellow": (5.0, (0.8, 1.1), (0.85, 1.06), 0.0),
    "orange": (4.0, (0.85, 1.05), (0.85, 1.03), 0.0),
    "purple": (8.0, (0.8, 1.1), (0.8, 1.08), 0.0),
    "white": (0.0, (1.0, 1.0), (0.86, 1.02), 0.06),
    "cyan": (7.0, (0.8, 1.1), (0.8, 1.06), 0.0),
    "pink": (5.0, (0.85, 1.1), (0.9, 1.02), 0.0),
}
ITEMS = {
    "camera": ("azimuth", "elevation", "fov", "distance", "look_at", "roll", "aspect", "distortion", "crop", "resolution", "top_jitter", "top_perspective"),
    "lighting": ("lights", "light_direction", "intensity", "ambient", "temperature", "shadows", "specular"),
    "style": ("style",),
    "materials": ("floor", "table", "terrain", "robots", "object_noise", "grid_lines", "markers"),
    "colour": ("colour_jitter",),
    "distractors": ("clutter", "stickers", "background", "hand_shadow", "cable_shadow"),
    "corruptions": ("gaussian_noise", "shot_noise", "motion_blur", "defocus", "jpeg", "vignetting", "exposure", "white_balance", "gamma", "chromatic", "cutout"),
}
KIND_ROBOT, KIND_WALL, KIND_OBJECT, KIND_CLUTTER = 1, 2, 3, 4
OTHER_AXES = ((1, 2), (0, 2), (0, 1))  # world axes (x 0, y 1, z 2) spanning a face of each normal axis


@dataclass
class ViewSpec:
    """Which cameras, the output size and how much randomization (module docstring)."""

    cameras: tuple = CAMERAS
    size: int = 256
    strength: float = DEFAULT_STRENGTH
    camera: float = None
    lighting: float = None
    style: float = None
    materials: float = None
    colour: float = None
    distractors: float = None
    corruptions: float = None
    off: frozenset = frozenset()
    azimuth: float = None  # fix the oblique camera's azimuth (degrees, world frame); None: drawn

    @classmethod
    def clean(cls, **kw):
        return cls(strength=0.0, **kw)

    def level(self, group):
        value = getattr(self, group)
        return float(self.strength if value is None else value)

    def amount(self, item):
        """The strength of one item: its group's, 0 if the item is turned off."""
        if item in self.off:
            return 0.0
        for group, names in ITEMS.items():
            if item in names:
                return self.level(group)
        raise KeyError(item)


# ---------------------------------------------------------------------------------------------
# Colour


def _linear(rgb):
    rgb = np.asarray(rgb, np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb):
    """sRGB in [0, 1] (..., 3) to CIE Lab (D65)."""
    lin = _linear(np.clip(rgb, 0, 1))
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


PALETTE_LAB = rgb_to_lab(PALETTE)


def lab_distance(a, b):
    return np.linalg.norm(rgb_to_lab(a) - rgb_to_lab(b), axis=-1)


def nearest_colour_name(rgb):
    """The named colour (COLOUR_NAMES) nearest to rgb in CIE Lab; (..., 3) gives an array of names."""
    d = np.linalg.norm(rgb_to_lab(rgb)[..., None, :] - PALETTE_LAB, axis=-1)
    index = d.argmin(axis=-1)
    return COLOUR_NAMES[int(index)] if np.ndim(index) == 0 else np.array(COLOUR_NAMES)[index]


def rgb_to_hsv(rgb):
    rgb = np.asarray(rgb, np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    top, low = rgb.max(-1), rgb.min(-1)
    span = top - low
    safe = np.where(span > 0, span, 1)
    h = np.where(top == r, (g - b) / safe % 6, np.where(top == g, (b - r) / safe + 2, (r - g) / safe + 4)) / 6
    h = np.where(span > 0, h, 0.0)
    s = np.where(top > 0, span / np.where(top > 0, top, 1), 0.0)
    return np.stack([h % 1.0, s, top], axis=-1)


def hsv_to_rgb(hsv):
    hsv = np.asarray(hsv, np.float64)
    h, s, v = hsv[..., 0] % 1.0, hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    table = np.stack([np.stack(c, -1) for c in ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))], 0)
    return np.take_along_axis(table, i[None, ..., None].repeat(3, -1), 0)[0]


def jitter_colour(name, rng, strength=1.0):
    """The named colour with brightness, saturation and a small hue shift drawn within
    COLOUR_JITTER[name] scaled by strength (0: the exact colour)."""
    rgb = PALETTE[COLOUR_NAMES.index(name)]
    if strength <= 0:
        return rgb.copy()
    hue, (s_lo, s_hi), (v_lo, v_hi), tint = COLOUR_JITTER[name]
    h, s, v = rgb_to_hsv(rgb)
    h = h + rng.uniform(-hue, hue) * strength / 360
    s = s * rng.uniform(1 - strength * (1 - s_lo), 1 + strength * (s_hi - 1))
    v = v * rng.uniform(1 - strength * (1 - v_lo), 1 + strength * (v_hi - 1))
    if tint > 0:
        h, s = rng.random(), s + rng.uniform(0, tint * strength)
    return np.clip(hsv_to_rgb(np.array([h, min(s, 1.0), min(v, 1.0)])), 0, 1).astype(np.float32)


def kelvin_rgb(kelvin):
    """Approximate RGB of a black body at `kelvin`, scaled to mean 1."""
    t = kelvin / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    rgb = np.clip(np.array([r, g, b], np.float32), 0, 255) / 255
    return rgb / rgb.mean()


# ---------------------------------------------------------------------------------------------
# Procedural noise


class Noise:
    """Value noise on a random periodic lattice: noise(x, y) in [0, 1], fbm for octaves."""

    def __init__(self, rng, n=64):
        self.flat = rng.random(n * n).astype(np.float32)  # n a power of two
        self.n = n

    def __call__(self, x, y):
        n = self.n
        x = np.asarray(x, np.float32)
        y = np.asarray(y, np.float32)
        xf, yf = np.floor(x), np.floor(y)
        fx, fy = x - xf, y - yf
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        x0, y0 = xf.astype(np.int32) & (n - 1), yf.astype(np.int32) & (n - 1)
        x1, row1 = (x0 + 1) & (n - 1), ((y0 + 1) & (n - 1)) * n
        row0 = y0 * n
        g = self.flat
        top = g[row0 + x0] * (1 - fx) + g[row0 + x1] * fx
        bottom = g[row1 + x0] * (1 - fx) + g[row1 + x1] * fx
        return top * (1 - fy) + bottom * fy

    def fbm(self, x, y, octaves=3):
        total, weight, scale, norm = 0.0, 1.0, 1.0, 0.0
        for k in range(octaves):
            total = total + weight * self(x * scale + 17.1 * k, y * scale - 9.7 * k)
            norm += weight
            weight *= 0.5
            scale *= 2.0
        return total / norm


def _hash01(a, b, c, seed):
    """A pseudo-random number in [0, 1) per integer triple (arrays)."""
    h = (np.asarray(a, np.int64) * 73856093) ^ (np.asarray(b, np.int64) * 19349663) ^ (np.asarray(c, np.int64) * 83492791) ^ int(seed)
    h = (h * 2654435761) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 2246822519) & 0xFFFFFFFF
    return ((h >> 8) % 65536).astype(np.float32) / 65536


# ---------------------------------------------------------------------------------------------
# Scene: the observation as voxel kinds and faces


@dataclass
class Scene:
    kind: np.ndarray  # (D, H, W) int8: 0 empty, KIND_ROBOT, KIND_WALL, KIND_OBJECT
    colour: np.ndarray  # (D, H, W) int8 palette index of object voxels, -1 elsewhere
    food: np.ndarray  # (D, H, W) bool
    zone: np.ndarray  # (H, W) int8 palette index of the zone painted on each floor cell, -1 none
    zone_level: np.ndarray  # (H, W) int: the height the zone paint lies at (its voxel's z)


def scene_of(obs):
    """A Scene from a (C, D, H, W) observation with 6 to 10 channels (world.CHANNELS)."""
    obs = obs.detach().cpu().numpy() if hasattr(obs, "detach") else np.asarray(obs)
    obs = obs.astype(np.float32)
    c, d, h, w = obs.shape
    if c < 6:
        raise ValueError(f"an observation needs at least 6 channels, got {c}")
    kind = np.zeros((d, h, w), np.int8)
    kind[obs[1] > 0.5] = KIND_WALL
    present = obs[2] > 0.5
    kind[present] = KIND_OBJECT
    kind[obs[0] > 0.5] = KIND_ROBOT
    colour = np.full((d, h, w), -1, np.int8)
    if present.any():
        rgb = np.moveaxis(obs[3:6], 0, -1)[present]
        colour[present] = np.linalg.norm(rgb[:, None, :] - PALETTE[None], axis=-1).argmin(axis=1)
    food = present & (obs[6] > 0.5) if c > 6 else present
    zone = np.full((h, w), -1, np.int8)
    zone_level = np.zeros((h, w), np.int64)
    if c >= 10:
        paint = np.moveaxis(obs[7:10], 0, -1)  # (D, H, W, 3)
        paint_mask = paint.max(-1) > 0.05
        z, y, x = np.nonzero(paint_mask)
        if len(z):
            zone[y, x] = np.linalg.norm(paint[z, y, x][:, None, :] - PALETTE[None], axis=-1).argmin(axis=1)
            zone_level[y, x] = z
    return Scene(kind, colour, food, zone, zone_level)


@dataclass
class Faces:
    """Axis-aligned rectangles: normal along world axis `axis` (0 x, 1 y, 2 z) with `sign`, in the
    plane at `c`, spanning [lo1, hi1] x [lo2, hi2] on OTHER_AXES[axis]."""

    axis: np.ndarray
    sign: np.ndarray
    c: np.ndarray
    lo1: np.ndarray
    hi1: np.ndarray
    lo2: np.ndarray
    hi2: np.ndarray
    kind: np.ndarray
    cell: np.ndarray  # (F, 3) voxel (z, y, x), or (clutter index, 0, 0)

    def __len__(self):
        return len(self.axis)

    def take(self, index):
        return Faces(*(getattr(self, f)[index] for f in ("axis", "sign", "c", "lo1", "hi1", "lo2", "hi2", "kind", "cell")))


def voxel_faces(kind):
    """The faces of solid voxels that border an empty voxel (or the outside of the grid, but not
    the floor under z = 0)."""
    d, h, w = kind.shape
    solid = kind > 0
    pad = np.zeros((d + 2, h + 2, w + 2), bool)
    pad[1:-1, 1:-1, 1:-1] = solid
    pad[0] = True
    parts = []
    for axis in range(3):
        for sign in (-1, 1):
            off = [0, 0, 0]  # (dz, dy, dx)
            off[2 - axis] = sign
            neighbour = pad[1 + off[0] : 1 + off[0] + d, 1 + off[1] : 1 + off[1] + h, 1 + off[2] : 1 + off[2] + w]
            z, y, x = np.nonzero(solid & ~neighbour)
            xyz = np.stack([x, y, z], 1).astype(np.float32)
            a1, a2 = OTHER_AXES[axis]
            n = len(z)
            parts.append(
                Faces(
                    axis=np.full(n, axis, np.int8),
                    sign=np.full(n, sign, np.int8),
                    c=xyz[:, axis] + (sign > 0),
                    lo1=xyz[:, a1],
                    hi1=xyz[:, a1] + 1,
                    lo2=xyz[:, a2],
                    hi2=xyz[:, a2] + 1,
                    kind=kind[z, y, x],
                    cell=np.stack([z, y, x], 1),
                )
            )
    return _concat(parts)


def box_faces(boxes):
    """Faces of axis-aligned boxes [(x0, x1, y0, y1, z0, z1)] (no bottoms), kind KIND_CLUTTER."""
    parts = []
    for i, (x0, x1, y0, y1, z0, z1) in enumerate(boxes):
        lo, hi = (x0, y0, z0), (x1, y1, z1)
        for axis in range(3):
            for sign in (-1, 1):
                if axis == 2 and sign < 0:
                    continue
                a1, a2 = OTHER_AXES[axis]
                parts.append(
                    Faces(
                        axis=np.array([axis], np.int8),
                        sign=np.array([sign], np.int8),
                        c=np.array([hi[axis] if sign > 0 else lo[axis]], np.float32),
                        lo1=np.array([lo[a1]], np.float32),
                        hi1=np.array([hi[a1]], np.float32),
                        lo2=np.array([lo[a2]], np.float32),
                        hi2=np.array([hi[a2]], np.float32),
                        kind=np.array([KIND_CLUTTER], np.int8),
                        cell=np.array([[i, 0, 0]]),
                    )
                )
    return _concat(parts)


def _concat(parts):
    names = ("axis", "sign", "c", "lo1", "hi1", "lo2", "hi2", "kind", "cell")
    if not parts:
        return Faces(*(np.zeros((0, 3) if n == "cell" else 0, np.float32 if n in ("c", "lo1", "hi1", "lo2", "hi2") else np.int64) for n in names))
    return Faces(*(np.concatenate([getattr(p, n) for p in parts]) for n in names))


# ---------------------------------------------------------------------------------------------
# Cameras


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def heading(azimuth_deg):
    """(toward the camera, the camera's right) on the floor for a camera at this azimuth."""
    a = math.radians(azimuth_deg)
    return np.array([math.sin(a), math.cos(a), 0.0]), np.array([math.cos(a), -math.sin(a), 0.0])


@dataclass
class Camera:
    position: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    width: int
    height: int
    fx: float  # pixels per unit at depth 1 (perspective) or pixels per unit (orthographic)
    fy: float
    cx: float
    cy: float
    ortho: bool

    @classmethod
    def looking(cls, position, target, azimuth, width, height, fov=50.0, roll=0.0, aspect=1.0, principal=(0.0, 0.0), ortho=False, scale=1.0):
        """A camera at position looking at target; azimuth (degrees) gives its right when it looks
        straight down; fov the vertical field of view (perspective) or scale (orthographic)."""
        forward = _unit(np.asarray(target, np.float64) - np.asarray(position, np.float64))
        if np.hypot(forward[0], forward[1]) > 1e-6:
            right = _unit([-forward[1], forward[0], 0.0])
        else:
            right = heading(azimuth)[1]
        up = np.cross(forward, right)
        r = math.radians(roll)
        right, up = math.cos(r) * right + math.sin(r) * up, -math.sin(r) * right + math.cos(r) * up
        f = scale if ortho else (height / 2) / math.tan(math.radians(fov) / 2)
        cx = width / 2 * (1 + principal[0])
        cy = height / 2 * (1 + principal[1])
        return cls(np.asarray(position, np.float64), right, up, forward, width, height, f * aspect, f, cx, cy, ortho)

    def project(self, points):
        """(N, 3) world points to pixel x, y and depth along the optical axis."""
        rel = np.asarray(points, np.float64) - self.position
        x, y, z = rel @ self.right, rel @ self.up, rel @ self.forward
        if self.ortho:
            return self.cx + self.fx * x, self.cy - self.fy * y, z
        zs = np.where(z > 1e-6, z, 1e-6)
        return self.cx + self.fx * x / zs, self.cy - self.fy * y / zs, z

    def rays(self):
        """Ray origins and directions of every pixel centre, row-major and channel first: (3, N)
        each, or (3,) where every pixel shares it (a perspective camera's origin, an orthographic
        camera's direction). t along a ray is depth along the optical axis."""
        xn = ((np.arange(self.width) + 0.5 - self.cx) / self.fx).astype(np.float32)
        yn = (-(np.arange(self.height) + 0.5 - self.cy) / self.fy).astype(np.float32)
        r, u, f = (v.astype(np.float32) for v in (self.right, self.up, self.forward))
        across = (r[:, None, None] * xn[None, None, :] + u[:, None, None] * yn[None, :, None]).reshape(3, -1)
        if self.ortho:
            return self.position.astype(np.float32)[:, None] + across, f
        return self.position.astype(np.float32), across + f[:, None]


def _box_corners(d, h, w, top):
    return np.array([(x, y, z) for x in (0, w) for y in (0, h) for z in (0, top)], np.float64)


def fit_distance(direction, target, corners, fov, width, height, azimuth, margin=0.07):
    """The smallest distance along direction (from target) at which the projected corners of a
    perspective camera of this fov span at most the image less `margin` on each side (the caller
    centres them with the principal point)."""
    lo, hi = 1.0, 500.0
    for _ in range(24):
        mid = (lo + hi) / 2
        cam = Camera.looking(target + direction * mid, target, azimuth, width, height, fov)
        px, py, z = cam.project(corners)
        inside = (z > 0.1).all() and px.max() - px.min() <= width * (1 - 2 * margin) and py.max() - py.min() <= height * (1 - 2 * margin)
        lo, hi = (lo, mid) if inside else (mid, hi)
    return hi


def fit_scale(cam, corners, margin=0.07):
    """Orthographic pixels per unit that keep every corner in view with margin."""
    rel = corners - cam.position
    x = np.abs(rel @ cam.right - ((corners.mean(0) - cam.position) @ cam.right))
    y = np.abs(rel @ cam.up - ((corners.mean(0) - cam.position) @ cam.up))
    return min(cam.width / 2 * (1 - margin) / max(x.max(), 1e-6), cam.height / 2 * (1 - margin) / max(y.max(), 1e-6))


# ---------------------------------------------------------------------------------------------
# Rasterizing faces


def _component(values, axis, pix, n):
    """values (3,) or (3, n): the axis-th component (per entry of axis) at pixels pix."""
    if values.ndim == 1:
        return values[axis]
    return values.ravel()[axis * n + pix]


def splat(cam, faces, origin, direction):
    """Nearest face per pixel: (face index (N,) int, -1 none; depth (N,), inf none; u, v (N,)).
    origin and direction are (3,) or (3, N) (Camera.rays)."""
    n = cam.width * cam.height
    face_buf = np.full(n, -1, np.int64)
    depth = np.full(n, np.inf, np.float32)
    u_buf = np.zeros(n, np.float32)
    v_buf = np.zeros(n, np.float32)
    if len(faces) == 0:
        return face_buf, depth, u_buf, v_buf
    axis = faces.axis.astype(np.int64)
    if cam.ortho:
        visible = faces.sign * cam.forward[axis] < -1e-6
    else:
        visible = faces.sign * (cam.position[axis] - faces.c) > 1e-6
    index = np.nonzero(visible)[0]
    if len(index) == 0:
        return face_buf, depth, u_buf, v_buf
    fs = faces.take(index)
    ax = fs.axis.astype(np.int64)
    corners = np.zeros((len(fs), 4, 3))
    rows = np.arange(len(fs))
    a1 = np.array(OTHER_AXES)[ax, 0]
    a2 = np.array(OTHER_AXES)[ax, 1]
    for k, (p, q) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        corners[rows, k, ax] = fs.c
        corners[rows, k, a1] = fs.hi1 if p else fs.lo1
        corners[rows, k, a2] = fs.hi2 if q else fs.lo2
    px, py, z = cam.project(corners.reshape(-1, 3))
    px, py, z = px.reshape(-1, 4), py.reshape(-1, 4), z.reshape(-1, 4)
    front = (z > 1e-3).all(1)
    x0 = np.clip(np.ceil(px.min(1) - 0.5) - 1, 0, cam.width).astype(np.int64)
    x1 = np.clip(np.floor(px.max(1) - 0.5) + 1, -1, cam.width - 1).astype(np.int64)
    y0 = np.clip(np.ceil(py.min(1) - 0.5) - 1, 0, cam.height).astype(np.int64)
    y1 = np.clip(np.floor(py.max(1) - 0.5) + 1, -1, cam.height - 1).astype(np.int64)
    bw = np.where(front, np.maximum(x1 - x0 + 1, 0), 0)
    bh = np.where(front, np.maximum(y1 - y0 + 1, 0), 0)
    count = bw * bh
    total = int(count.sum())
    if total == 0:
        return face_buf, depth, u_buf, v_buf
    f_rep = np.repeat(np.arange(len(fs)), count)
    off = np.arange(total) - np.repeat(np.cumsum(count) - count, count)
    w_rep = bw[f_rep]
    pix = (y0[f_rep] + off // w_rep) * cam.width + x0[f_rep] + off % w_rep
    ax_rep, b1, b2 = ax[f_rep], a1[f_rep], a2[f_rep]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (fs.c[f_rep] - _component(origin, ax_rep, pix, n)) / _component(direction, ax_rep, pix, n)
        h1 = _component(origin, b1, pix, n) + t * _component(direction, b1, pix, n)
        h2 = _component(origin, b2, pix, n) + t * _component(direction, b2, pix, n)
    eps = 1e-4
    lo1, hi1, lo2, hi2 = fs.lo1[f_rep], fs.hi1[f_rep], fs.lo2[f_rep], fs.hi2[f_rep]
    keep = np.nonzero((t > 1e-4) & (h1 >= lo1 - eps) & (h1 <= hi1 + eps) & (h2 >= lo2 - eps) & (h2 <= hi2 + eps))[0]
    pix, t, f_rep = pix[keep], t[keep].astype(np.float32), f_rep[keep]
    u = ((h1[keep] - lo1[keep]) / (hi1[keep] - lo1[keep])).astype(np.float32)
    v = ((h2[keep] - lo2[keep]) / (hi2[keep] - lo2[keep])).astype(np.float32)
    np.minimum.at(depth, pix, t)
    win = np.nonzero(t <= depth[pix])[0]
    pix_win = pix[win]
    face_buf[pix_win] = index[f_rep[win]]
    u_buf[pix_win] = u[win]
    v_buf[pix_win] = v[win]
    return face_buf, depth, u_buf, v_buf



# ---------------------------------------------------------------------------------------------
# The look: every episode-level choice


@dataclass
class Surface:
    """A procedural material on a plane: type (SURFACE_TYPES), two colours, texture amplitude,
    scale and angle, specular strength and shininess."""

    type: str
    rgb: np.ndarray
    rgb2: np.ndarray
    amp: float = 0.0
    scale: float = 1.0
    angle: float = 0.0
    ks: float = 0.0
    shininess: float = 16.0
    lines: float = 0.0  # opacity of graph-paper lines (paper)


SURFACE_TYPES = ("steel", "brushed", "wood", "paper", "noise", "tiles")
TERRAIN_TYPES = ("sand", "wood", "stone", "concrete", "cardboard", "foam")
CLUTTER_SHAPES = ("paper", "book", "bar")  # flat sheets, low slabs, cable- or pen-like bars: never cube-like
DISTRACTOR_GAP = 25.0  # Lab distance of every distractor, table and backdrop colour from every named colour


@dataclass
class Light:
    direction: np.ndarray  # unit vector toward the light
    rgb: np.ndarray  # colour times intensity


@dataclass
class CameraLook:
    name: str
    projection: str  # "perspective" or "orthographic"
    azimuth: float
    elevation: float
    fov: float
    distance_factor: float
    target_shift: np.ndarray
    roll: float
    aspect: float
    principal: tuple
    k1: float
    zoom: float
    shift: tuple
    resolution: float
    style: str
    outline: float
    height_cue: float


@dataclass
class Look:
    views: list
    lights: list
    ambient: np.ndarray
    shadows: bool
    shadow_taps: int
    floor: Surface
    table: Surface
    sheet_margin: float
    grid_rgb: np.ndarray
    grid_alpha: float
    grid_width: float
    marker: str
    marker_size: float
    marker_rgb: np.ndarray
    terrain: Surface
    terrain_side: float
    terrain_edge: float
    robot_rgb: np.ndarray
    robot_edge_rgb: np.ndarray
    robot_edge_width: float
    robot_variation: float
    robot_ks: float
    robot_shininess: float
    marking: str
    marking_size: float
    marking_rgb: np.ndarray
    object_rgb: np.ndarray  # (9, 3) jittered palette for objects
    zone_rgb: np.ndarray  # (9, 3) jittered palette for zones
    object_noise: float
    object_ks: float
    hole_radius: float
    hole_dark: float
    clutter: list  # boxes (x0, x1, y0, y1, z0, z1)
    clutter_rgb: np.ndarray
    stickers: list  # (shape, cx, cy, rx, ry, angle, rgb)
    background: tuple  # (rgb top, rgb bottom, rgb blobs, blob amount)
    seed: int
    noise: Noise = field(repr=False, default=None)
    cache: dict = field(repr=False, default_factory=dict)  # the shadow camera and its rays


def _between(rng, clean, low, high, s):
    """A draw in [low, high] pulled toward clean by 1 - s."""
    return clean + s * (rng.uniform(low, high) - clean)


def _neutral(rng, low, high, tint):
    g = rng.uniform(low, high)
    rgb = hsv_to_rgb(np.array([rng.random(), rng.uniform(0, tint), 1.0])) * g
    return np.clip(rgb, 0, 1).astype(np.float32)


def _far_from_palette(rgb, gap):
    return bool((lab_distance(rgb[None], PALETTE) >= gap).all())


def unnameable_colour(rng, saturation=0.3):
    """A distractor colour: low saturation and at least DISTRACTOR_GAP Lab units from every named
    colour, so nothing outside the arena looks like an object an instruction can name."""
    for _ in range(60):
        rgb = hsv_to_rgb(np.array([rng.random(), rng.uniform(0, saturation), rng.uniform(0.1, 0.95)])).astype(np.float32)
        if _far_from_palette(rgb, DISTRACTOR_GAP):
            return rgb
    return np.full(3, 0.45, np.float32)


def _surface(rng, s, kinds=SURFACE_TYPES, any_colour=False):
    kind = kinds[int(rng.integers(len(kinds)))]
    angle = rng.uniform(0, math.pi)
    if kind == "steel":
        rgb = _neutral(rng, 0.3, 0.72, 0.06)
        return Surface(kind, rgb, rgb * 0.9, amp=rng.uniform(0.04, 0.12), scale=rng.uniform(0.6, 1.6), angle=angle, ks=rng.uniform(0.1, 0.5), shininess=rng.uniform(10, 40))
    if kind == "brushed":
        rgb = _neutral(rng, 0.3, 0.75, 0.05)
        return Surface(kind, rgb, rgb * 0.8, amp=rng.uniform(0.12, 0.25), scale=rng.uniform(1.0, 3.0), angle=angle, ks=rng.uniform(0.2, 0.5), shininess=rng.uniform(8, 30))
    if kind == "wood":
        rgb = hsv_to_rgb(np.array([rng.uniform(0.05, 0.11), rng.uniform(0.3, 0.6), rng.uniform(0.3, 0.75)])).astype(np.float32)
        return Surface(kind, rgb, (rgb * rng.uniform(0.65, 0.85)).astype(np.float32), amp=rng.uniform(0.3, 0.8), scale=rng.uniform(1.5, 4.0), angle=angle, ks=rng.uniform(0.0, 0.15), shininess=12)
    if kind == "paper":
        rgb = hsv_to_rgb(np.array([rng.random(), rng.uniform(0, 0.12), rng.uniform(0.5, 0.95)])).astype(np.float32)
        lines = rng.uniform(0.1, 0.3) if rng.random() < 0.3 else 0.0
        return Surface(kind, rgb, rgb * 0.8, amp=rng.uniform(0.02, 0.07), scale=rng.uniform(0.4, 1.3), angle=angle, ks=0.02, lines=lines)
    if kind == "noise":
        sat = 0.7 if any_colour else 0.25
        rgb = hsv_to_rgb(np.array([rng.random(), rng.uniform(0, sat), rng.uniform(0.15, 0.85)])).astype(np.float32)
        rgb2 = np.clip(rgb * rng.uniform(0.6, 1.4) + rng.normal(0, 0.05, 3), 0, 1).astype(np.float32)
        return Surface(kind, rgb, rgb2, amp=rng.uniform(0.3, 1.0), scale=rng.uniform(0.1, 0.6), angle=angle, ks=rng.uniform(0, 0.1))
    sat = 0.7 if any_colour else 0.2
    rgb = hsv_to_rgb(np.array([rng.random(), rng.uniform(0, sat), rng.uniform(0.15, 0.85)])).astype(np.float32)
    rgb2 = np.clip(rgb * rng.uniform(0.75, 1.25), 0, 1).astype(np.float32)
    return Surface("tiles", rgb, rgb2, amp=rng.uniform(0.1, 0.4), scale=rng.uniform(1.3, 5.0), angle=angle, ks=rng.uniform(0, 0.3), shininess=30)


def _terrain(rng):
    kind = TERRAIN_TYPES[int(rng.integers(len(TERRAIN_TYPES)))]
    hsv = {
        "sand": (rng.uniform(0.08, 0.13), rng.uniform(0.2, 0.4), rng.uniform(0.6, 0.85)),
        "wood": (rng.uniform(0.05, 0.11), rng.uniform(0.3, 0.6), rng.uniform(0.35, 0.75)),
        "stone": (rng.random(), rng.uniform(0, 0.1), rng.uniform(0.35, 0.7)),
        "concrete": (rng.random(), rng.uniform(0, 0.06), rng.uniform(0.45, 0.8)),
        "cardboard": (rng.uniform(0.07, 0.1), rng.uniform(0.35, 0.55), rng.uniform(0.5, 0.75)),
        "foam": (rng.random(), rng.uniform(0.05, 0.3), rng.uniform(0.3, 0.8)),
    }[kind]
    rgb = hsv_to_rgb(np.array(hsv)).astype(np.float32)
    amp = {"sand": 0.15, "wood": 0.3, "stone": 0.3, "concrete": 0.15, "cardboard": 0.1, "foam": 0.06}[kind] * rng.uniform(0.5, 1.5)
    return Surface(kind, rgb, (rgb * 0.8).astype(np.float32), amp=amp, scale=rng.uniform(0.5, 2.0), angle=rng.uniform(0, math.pi))


def _allowed_azimuth(rng):
    """Within 30 degrees of a side (0, 90, 180, 270): never within 15 of a diagonal."""
    return float((90 * int(rng.integers(4)) + rng.uniform(-30, 30)) % 360)


def side_of(azimuth):
    return ("front", "right", "back", "left")[int(((azimuth + 45) % 360) // 90)]


def _camera_look(rng, spec, name):
    def amt(item):
        return spec.amount(item)

    style_level = spec.amount("style")
    pick = rng.random()
    style = "lit"
    if pick < 0.3 * style_level:
        style = "flat"
    elif pick < 0.5 * style_level:
        style = "toon"
    outline = 0.5 if style_level == 0 else (0.55 if style == "toon" else (rng.uniform(0.25, 0.6) if rng.random() < 0.35 * style_level else 0.0))
    k1 = 0.0
    if rng.random() < 0.6 * amt("distortion"):
        k1 = rng.uniform(-0.04, 0.22) * amt("distortion")
    resolution = _between(rng, 1.0, 0.5, 1.0, amt("resolution"))
    zoom = _between(rng, 1.0, 1.0, 1.1, amt("crop"))
    room = (1 - 1 / zoom) / 2  # the crop stays inside the render
    shift = (rng.uniform(-room, room), rng.uniform(-room, room))
    aspect = _between(rng, 1.0, 0.96, 1.04, amt("aspect"))
    principal = (rng.uniform(-0.03, 0.03) * amt("aspect"), rng.uniform(-0.03, 0.03) * amt("aspect"))
    if name == "top":
        perspective = rng.random() < 0.5 * amt("top_perspective")
        return CameraLook(
            name=name,
            projection="perspective" if perspective else "orthographic",
            azimuth=rng.uniform(-6, 6) * amt("top_jitter"),
            elevation=90 - rng.uniform(0, 8) * amt("top_jitter"),
            fov=rng.uniform(20, 45) if perspective else 0.0,
            distance_factor=1.0,
            target_shift=np.zeros(3),
            roll=0.0,
            aspect=aspect,
            principal=principal,
            k1=k1,
            zoom=zoom,
            shift=shift,
            resolution=resolution,
            style=style,
            outline=max(outline, 0.45),
            height_cue=_between(rng, 0.4, 0.28, 0.5, spec.level("materials")),
        )
    return CameraLook(
        name=name,
        projection="perspective",
        azimuth=0.0,
        elevation=_between(rng, 45, 25, 65, amt("elevation")),
        fov=_between(rng, 50, 35, 65, amt("fov")),
        distance_factor=_between(rng, 1.0, 0.95, 1.4, amt("distance")),
        target_shift=np.array([rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), rng.uniform(-0.5, 0.5)]) * amt("look_at"),
        roll=rng.uniform(-6, 6) * amt("roll"),
        aspect=aspect,
        principal=principal,
        k1=k1,
        zoom=zoom,
        shift=shift,
        resolution=resolution,
        style=style,
        outline=outline,
        height_cue=0.0,
    )


def sample_look(rng, spec=None):
    """Every episode-level choice of the views (module docstring): cameras, lights, materials,
    colours, clutter. Reuse it for every frame of an episode."""
    spec = ViewSpec() if spec is None else spec
    views = [_camera_look(rng, spec, name) for name in spec.cameras]
    azimuth = spec.azimuth
    if azimuth is None:
        azimuth = _allowed_azimuth(rng) if spec.amount("azimuth") > 0 else 0.0
    for view in views:
        view.azimuth = float((azimuth + view.azimuth) % 360)

    light_level, dir_level = spec.amount("lights"), spec.amount("light_direction")
    if dir_level > 0:
        light_az = rng.uniform(0, 360)
        light_el = _between(rng, 60, 30, 90, dir_level)
    else:
        light_az, light_el = azimuth - 35, 60.0
    temperature = spec.amount("temperature")
    intensity = spec.amount("intensity")

    def light(az, el, power):
        a, e = math.radians(az), math.radians(el)
        direction = np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)], np.float32)
        kelvin = _between(rng, 5500, 2800, 8200, temperature)
        return Light(direction, (kelvin_rgb(kelvin) * power).astype(np.float32))

    lights = [light(light_az, light_el, 0.75 * _between(rng, 1.0, 0.65, 1.35, intensity))]
    for _ in range(2):
        if rng.random() < 0.5 * light_level:
            lights.append(light(rng.uniform(0, 360), rng.uniform(25, 80), rng.uniform(0.2, 0.6)))
    ambient_level = _between(rng, 0.4, 0.18, 0.62, spec.amount("ambient"))
    sky = np.array([0.85, 0.92, 1.1], np.float32) if rng.random() < 0.5 else lights[0].rgb / max(lights[0].rgb.mean(), 1e-6)
    tint = spec.amount("ambient") * rng.uniform(0, 0.6)
    ambient = (ambient_level * ((1 - tint) + tint * sky)).astype(np.float32)
    shadows = not (rng.random() < 0.35 * spec.amount("shadows"))
    shadow_taps = (1, 4)[int(rng.integers(2))] if spec.amount("shadows") > 0 else 4

    mat = spec.level("materials")
    robot_level = spec.amount("robots")
    robot_rgb = CLEAN_ROBOT.copy()
    if robot_level > 0:
        for _ in range(30):
            rgb = _neutral(rng, 0.12, 0.85, 0.08)
            if _far_from_palette(rgb, 15):
                robot_rgb = rgb
                break
    floor = Surface("slate", CLEAN_FLOOR.copy(), CLEAN_FLOOR.copy(), amp=0.0, scale=1.0)
    if spec.amount("floor") > 0:
        for _ in range(40):
            option = _surface(rng, mat)
            if _far_from_palette(option.rgb, 20) and lab_distance(option.rgb, robot_rgb) >= 15:
                floor = option
                break
    table = Surface("plain", CLEAN_TABLE.copy(), CLEAN_TABLE.copy(), amp=0.0, scale=1.0)
    if spec.amount("table") > 0:
        for _ in range(30):
            option = _surface(rng, mat, any_colour=True)
            if _far_from_palette(option.rgb, DISTRACTOR_GAP) and _far_from_palette(option.rgb2, DISTRACTOR_GAP):
                table = option
                break
    terrain = Surface("sand", CLEAN_TERRAIN.copy(), CLEAN_TERRAIN * 0.8, amp=0.08, scale=1.0)
    if spec.amount("terrain") > 0:
        for _ in range(30):
            option = _terrain(rng)
            if _far_from_palette(option.rgb, 15) and lab_distance(option.rgb, floor.rgb) >= 10:
                terrain = option
                break
    floor_light = rgb_to_lab(floor.rgb)[0] > 50
    grid_level = spec.amount("grid_lines")
    grid_lighter = (not floor_light) if grid_level == 0 else bool(rng.random() < (0.25 if floor_light else 0.75))
    grid_rgb = np.ones(3, np.float32) if grid_lighter else np.zeros(3, np.float32)
    if grid_level > 0:
        grid_rgb = np.clip(grid_rgb * rng.uniform(0.85, 1.0) + rng.uniform(0, 0.15) + rng.normal(0, 0.04, 3), 0, 1).astype(np.float32)
    marker_level = spec.amount("markers")
    marker_rgb = (np.zeros(3) if floor_light else np.ones(3)).astype(np.float32) * (0.95 if not floor_light else 1.0)
    marker = "L"
    marker_size = 0.7
    if marker_level > 0:
        marker = ("L", "square", "dot", "cross")[int(rng.integers(4))]
        marker_size = rng.uniform(0.4, 1.0)
        base = rng.uniform(0.0, 0.18) if floor_light else rng.uniform(0.82, 1.0)
        marker_rgb = np.clip(base + rng.normal(0, 0.05, 3), 0, 1).astype(np.float32)

    edge_dark = robot_rgb.mean() > 0.3
    if robot_level > 0:
        edge_factor = rng.uniform(0.3, 0.7) if edge_dark else rng.uniform(1.6, 2.6)
        markings = ("none", "dot", "ring", "cross", "square", "stripe")
        marking = markings[int(rng.integers(1, len(markings)))] if rng.random() < 0.5 else "none"
    else:
        edge_factor, marking = 0.45, "none"
    robot_edge_rgb = np.clip(robot_rgb * edge_factor + (0 if edge_dark else 0.05), 0, 1).astype(np.float32)
    marking_rgb = np.clip(robot_rgb * (rng.uniform(0.35, 0.7) if edge_dark else rng.uniform(1.8, 3.0)), 0, 1).astype(np.float32)

    colour_level = spec.amount("colour_jitter")
    object_rgb = np.stack([jitter_colour(name, rng, colour_level) for name in COLOUR_NAMES])
    zone_rgb = np.stack([jitter_colour(name, rng, colour_level) for name in COLOUR_NAMES])

    clutter, clutter_rgb, stickers = [], [], []
    margin = _between(rng, 1.2, 0.6, 3.0, spec.amount("table") if spec.amount("table") > 0 else 0.0) if mat > 0 else 1.2
    reach = 16.0 + margin
    for _ in range(int(rng.poisson(4 * spec.amount("clutter")))):
        shape = CLUTTER_SHAPES[int(rng.integers(len(CLUTTER_SHAPES)))]
        if shape == "paper":
            sx, sy, sz = rng.uniform(1, 4), rng.uniform(1, 3), rng.uniform(0.02, 0.08)
        elif shape == "book":
            sx, sy, sz = rng.uniform(1.5, 4), rng.uniform(1, 3), rng.uniform(0.2, 0.6)
        else:
            sx, sy, sz = rng.uniform(2, 7), rng.uniform(0.15, 0.45), rng.uniform(0.15, 0.45)
        if rng.random() < 0.5:
            sx, sy = sy, sx
        side = int(rng.integers(4))
        along = rng.uniform(-margin - 6, 16 + margin + 6)
        out = rng.uniform(0.3, 8)
        x0, y0 = {0: (along, -margin - out - sy), 1: (along, reach + out), 2: (-margin - out - sx, along), 3: (reach + out, along)}[side]
        clutter.append((x0, x0 + sx, y0, y0 + sy, 0.0, sz))
        clutter_rgb.append(unnameable_colour(rng))
    for _ in range(int(rng.poisson(5 * spec.amount("stickers")))):
        shape = ("ellipse", "rect", "stripe")[int(rng.integers(3))]
        side = int(rng.integers(4))
        along = rng.uniform(-margin - 4, 16 + margin + 4)
        out = rng.uniform(0.5, 6)
        cx, cy = {0: (along, -margin - out), 1: (along, reach + out), 2: (-margin - out, along), 3: (reach + out, along)}[side]
        stickers.append((shape, cx, cy, rng.uniform(0.3, 2.5), rng.uniform(0.2, 1.5), rng.uniform(0, math.pi), unnameable_colour(rng)))
    if spec.amount("background") > 0:
        background = (unnameable_colour(rng, 0.4), unnameable_colour(rng, 0.4), unnameable_colour(rng, 0.4), rng.uniform(0, 0.6))
    else:
        background = (np.array([0.9, 0.9, 0.88], np.float32), np.array([0.7, 0.7, 0.68], np.float32), np.zeros(3, np.float32), 0.0)
    seed = int(rng.integers(1 << 30))
    if all(spec.level(group) == 0 for group in GROUPS):
        seed = 0  # clean: the same textures every time
    noise_rng = np.random.default_rng(seed)
    return Look(
        views=views,
        lights=lights,
        ambient=ambient,
        shadows=shadows,
        shadow_taps=shadow_taps,
        floor=floor,
        table=table,
        sheet_margin=margin,
        grid_rgb=grid_rgb,
        grid_alpha=_between(rng, CLEAN_GRID, 0.15, 0.5, grid_level),
        grid_width=_between(rng, 0.06, 0.04, 0.1, grid_level),
        marker=marker,
        marker_size=marker_size,
        marker_rgb=marker_rgb,
        terrain=terrain,
        terrain_side=_between(rng, 0.9, 0.75, 1.0, spec.amount("terrain")),
        terrain_edge=_between(rng, 0.2, 0.05, 0.35, spec.amount("terrain")),
        robot_rgb=robot_rgb,
        robot_edge_rgb=robot_edge_rgb,
        robot_edge_width=_between(rng, 0.07, 0.03, 0.12, robot_level),
        robot_variation=0.06 * robot_level,
        robot_ks=_between(rng, 0.15, 0.0, 0.55, spec.amount("specular")),
        robot_shininess=_between(rng, 24, 8, 64, spec.amount("specular")),
        marking=marking,
        marking_size=rng.uniform(0.15, 0.35),
        marking_rgb=marking_rgb,
        object_rgb=object_rgb,
        zone_rgb=zone_rgb,
        object_noise=rng.uniform(0, 0.12) * spec.amount("object_noise"),
        object_ks=rng.uniform(0, 0.15) * spec.amount("specular"),
        hole_radius=_between(rng, 0.16, 0.1, 0.2, mat),
        hole_dark=_between(rng, 0.6, 0.5, 0.8, mat),
        clutter=clutter,
        clutter_rgb=np.array(clutter_rgb, np.float32).reshape(-1, 3),
        stickers=stickers,
        background=background,
        seed=seed,
        noise=Noise(noise_rng),
    )


# ---------------------------------------------------------------------------------------------
# Materials


def _rotate(x, y, angle):
    c, s = math.cos(angle), math.sin(angle)
    return c * x - s * y, s * x + c * y


def _mix(rgb, rgb2, amount):
    """(3, n): rgb where amount is 0, rgb2 where it is 1."""
    return rgb[:, None] + (rgb2 - rgb)[:, None] * amount


def surface_albedo(surface, x, y, noise):
    """(3, n) albedo of a Surface at plane coordinates x, y (world units)."""
    rgb, rgb2 = surface.rgb, surface.rgb2
    k = surface.scale
    if surface.type in ("steel", "brushed", "wood", "paper", "tiles"):
        xr, yr = _rotate(x, y, surface.angle)
    if surface.type in ("steel", "brushed"):
        streak = noise(xr * 0.5 * k, yr * 18 * k) * 0.6 + noise(xr * 1.3 * k + 40, yr * 45 * k) * 0.4
        smudge = noise(x * 0.25 + 9, y * 0.25 - 3)
        return rgb[:, None] * (1 + surface.amp * (streak - 0.5) * 2 + 0.6 * surface.amp * (smudge - 0.5))
    if surface.type == "wood":
        width = 1.5 * k
        plank = np.floor(yr / width)
        ring = np.sin(2 * math.pi * (xr * 0.08 / k + noise(xr * 0.15, yr * 1.5) * 1.2 + _hash01(plank, 1, 2, 7) * 3) * 3)
        shade = (0.92 + 0.16 * _hash01(plank, 3, 4, 11)) * np.where(np.abs(yr / width - plank - 0.5) > 0.47, 0.7, 1.0)
        return _mix(rgb, rgb2, (0.5 + 0.5 * ring) * surface.amp) * shade
    if surface.type == "paper":
        out = rgb[:, None] * (1 + surface.amp * (noise(x * 25 * k, y * 25 * k) - 0.5) * 2)
        if surface.lines > 0:
            line = (np.minimum(np.abs(xr * 2 - np.round(xr * 2)), np.abs(yr * 2 - np.round(yr * 2))) < 0.03) * surface.lines
            out = out * (1 - line) + np.array([0.4, 0.55, 0.8], np.float32)[:, None] * line
        return out
    if surface.type == "noise":
        return _mix(rgb, rgb2, np.clip(noise.fbm(x * k, y * k, 3) * surface.amp * 2 - surface.amp + 0.5, 0, 1))
    if surface.type == "tiles":
        tx, ty = xr / k, yr / k
        checker = (np.floor(tx) + np.floor(ty)) % 2
        grout = (np.minimum(np.abs(tx - np.round(tx)), np.abs(ty - np.round(ty))) * k) < 0.05
        return _mix(rgb, rgb2, checker) * ((1 + surface.amp * (noise(x * 2, y * 2) - 0.5)) * np.where(grout, 0.6, 1.0))
    if surface.amp == 0:
        return np.repeat(rgb[:, None], len(x), 1)
    return rgb[:, None] * (1 + surface.amp * (noise.fbm(x * 0.7, y * 0.7, 2) - 0.5) * 2)


def terrain_albedo(surface, a, b, noise):
    """(3, n) terrain blocks: a solid texture on face coordinates (a, b)."""
    k = surface.scale
    if surface.type == "wood":
        ring = np.sin(2 * math.pi * (a * 0.3 / k + noise(a * 0.4, b * 2.5)) * 3)
        return _mix(surface.rgb, surface.rgb2, (0.5 + 0.5 * ring) * surface.amp)
    if surface.type == "stone":
        cells = noise(a * 1.7 * k + 3, b * 1.7 * k) * 0.6 + noise(a * 6 * k, b * 6 * k) * 0.4
        return surface.rgb[:, None] * (1 + surface.amp * (cells - 0.5) * 2)
    if surface.type == "cardboard":
        ribs = 0.5 + 0.5 * np.sin(a * 2 * math.pi * 3 / k)
        return surface.rgb[:, None] * (1 + surface.amp * (ribs - 0.5) + 0.05 * (noise(a * 9, b * 9) - 0.5))
    grain = noise(a * 8 * k + 5, b * 8 * k) * 0.5 + noise(a * 0.8, b * 0.8) * 0.5
    return surface.rgb[:, None] * (1 + surface.amp * (grain - 0.5) * 2)


def _marking_mask(kind, u, v, size):
    du, dv = u - 0.5, v - 0.5
    r = np.sqrt(du * du + dv * dv)
    if kind == "dot":
        return r < size * 0.6
    if kind == "ring":
        return np.abs(r - size) < 0.05
    if kind == "cross":
        return ((np.abs(du) < 0.05) | (np.abs(dv) < 0.05)) & (np.maximum(np.abs(du), np.abs(dv)) < size)
    if kind == "square":
        return np.abs(np.maximum(np.abs(du), np.abs(dv)) - size) < 0.04
    if kind == "stripe":
        return np.abs(dv) < size * 0.3
    return np.zeros_like(u, bool)


# ---------------------------------------------------------------------------------------------
# Rendering one view


def _camera(view, d, h, w, size):
    """The Camera of a CameraLook for a (d, h, w) grid at render resolution."""
    res = max(32, int(round(size * view.resolution)))
    centre = np.array([w / 2, h / 2, 0.0])
    toward, _ = heading(view.azimuth)
    if view.name == "top":
        tilt = math.radians(90 - view.elevation)
        direction = _unit(np.array([0, 0, 1.0]) * math.cos(tilt) + toward * math.sin(tilt))
        if view.projection == "orthographic":
            position = centre + direction * (d + 20)
            cam = Camera.looking(position, centre, view.azimuth, res, res, ortho=True, aspect=view.aspect, principal=view.principal)
            scale = fit_scale(cam, _box_corners(d, h, w, 0), margin=0.06)
            return Camera.looking(position, centre, view.azimuth, res, res, ortho=True, scale=scale, aspect=view.aspect, principal=view.principal)
        dist = fit_distance(direction, centre, _box_corners(d, h, w, min(4, d)), view.fov, res, res, view.azimuth, margin=0.06)
        return Camera.looking(centre + direction * dist, centre, view.azimuth, res, res, fov=view.fov, aspect=view.aspect, principal=view.principal)
    el = math.radians(view.elevation)
    direction = _unit(toward * math.cos(el) + np.array([0, 0, 1.0]) * math.sin(el))
    target = centre + np.array([0, 0, 1.0])
    corners = _box_corners(d, h, w, min(FRAME_HEIGHT, d))
    dist = fit_distance(direction, target, corners, view.fov, res, res, view.azimuth) * view.distance_factor
    cam = Camera.looking(target + direction * dist, target, view.azimuth, res, res, fov=view.fov)
    px, py, _ = cam.project(corners)
    principal = (view.principal[0] - (px.min() + px.max() - res) / res, view.principal[1] - (py.min() + py.max() - res) / res)  # the arena's box principal in the image
    return Camera.looking(cam.position, target + view.target_shift, view.azimuth, res, res, fov=view.fov, roll=view.roll, aspect=view.aspect, principal=principal)


_RAYS = {}


def camera_rays(cam):
    """cam.rays(), cached by camera (a fixed camera renders every frame of an episode)."""
    key = (cam.ortho, cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, np.concatenate([cam.position, cam.right, cam.up]).round(9).tobytes())
    rays = _RAYS.get(key)
    if rays is None:
        if len(_RAYS) >= 24:
            _RAYS.clear()
        rays = cam.rays()
        _RAYS[key] = rays
    return rays


def _shadow_camera(look, d, h, w):
    key = ("shadow", d, h, w)
    if key not in look.cache:
        toward = _unit(look.lights[0].direction)
        centre = np.array([w / 2, h / 2, d / 4])
        position = centre + toward * (2 * max(d, h, w) + 10)
        azimuth = math.degrees(math.atan2(toward[0], toward[1])) if np.hypot(toward[0], toward[1]) > 1e-6 else 0.0
        corners = np.array([(x, y, z) for x in (-3, w + 3) for y in (-3, h + 3) for z in (0, min(d, 8))], np.float64)
        res = SHADOW_RES
        cam = Camera.looking(position, centre, azimuth, res, res, ortho=True)
        cam = Camera.looking(position, centre, azimuth, res, res, ortho=True, scale=fit_scale(cam, corners, margin=0.01))
        look.cache[key] = (cam, cam.rays())
    return look.cache[key]


def _shadow_lookup(look, faces, points, nid, d, h, w):
    """(n,) visibility in [0, 1] of world points (3, n) with normal ids nid from the main light:
    a shadow map (the faces seen from the light), a normal offset and a slope bias against
    self-shadowing, 1 or 4 taps."""
    cam, (origin, direction) = _shadow_camera(look, d, h, w)
    _, depth_map, _, _ = splat(cam, faces, origin, direction)
    texel = 1.0 / cam.fy
    cos = np.clip(NORMALS @ look.lights[0].direction, 0.05, 1)
    slope = np.minimum(np.sqrt(1 - cos * cos) / cos, 4.0)
    basis = np.stack([cam.right * cam.fx, -cam.up * cam.fy, cam.forward]).astype(np.float32)  # world to (px, py, depth)
    shift = (NORMALS * (1.5 * texel)) @ basis.T  # (7, 3): the normal offset per normal id
    shift[:, 2] -= texel * (0.5 + slope) + 0.01  # minus the bias
    shift += np.array([cam.cx, cam.cy, 0.0], np.float32) - (basis @ cam.position).astype(np.float32)
    coords = basis @ points
    px = coords[0] + shift[:, 0][nid]
    py = coords[1] + shift[:, 1][nid]
    z = coords[2] + shift[:, 2][nid]
    res = cam.width
    if look.shadow_taps <= 1:
        i = np.clip(px.astype(np.int32), 0, res - 1)
        j = np.clip(py.astype(np.int32), 0, res - 1)
        return (z <= depth_map[j * res + i]).astype(np.float32)
    i0 = np.clip((px - 0.5).astype(np.int32), 0, res - 2)
    k0 = np.clip((py - 0.5).astype(np.int32), 0, res - 2) * res + i0
    light_hits = np.zeros(len(nid), np.float32)
    for k in (k0, k0 + 1, k0 + res, k0 + res + 1):
        light_hits += z <= depth_map[k]
    return light_hits * 0.25


SHADOW_RES = 160
NORMALS = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1], [0, 0, 1]], np.float32)  # normal id: 2 axis + (sign > 0); 6 the floor
FLAT_SHADE = np.array([0.8, 0.8, 0.66, 0.66, 1.0, 1.0, 1.0], np.float32)  # flat style per normal id


def _rows(values, index):
    """values (3,) or (3, N): the columns at index, (3, len(index)) (broadcast when (3,))."""
    if values.ndim == 1:
        return values[:, None]
    return values[:, index]


def render_view(scene, faces, look, view, size):
    """(rgb (res, res, 3) float, protect (res, res) bool: object and zone pixels, Camera)."""
    d, h, w = scene.kind.shape
    slot = ("camera", view.name, d, h, w, size)
    if slot not in look.cache:
        look.cache[slot] = _camera(view, d, h, w, size)
    cam = look.cache[slot]
    origin, direction = camera_rays(cam)
    face_buf, depth, u, v = splat(cam, faces, origin, direction)
    n = cam.width * cam.height
    dz = np.broadcast_to(direction[2], (n,))
    oz = np.broadcast_to(origin[2], (n,))
    down = np.nonzero(dz < -1e-6)[0]
    t_floor = np.full(n, np.inf, np.float32)
    t_floor[down] = -oz[down] / dz[down]
    on_floor = t_floor < depth
    fi = np.nonzero((face_buf >= 0) & ~on_floor)[0]
    fl = np.nonzero(on_floor)[0]
    hit = np.concatenate([fi, fl])  # every pixel that sees a surface: faces, then floor
    t_hit = np.concatenate([depth[fi], t_floor[fl]])
    ray = _rows(direction, hit)
    point = _rows(origin, hit) + ray * t_hit
    nf = len(fi)
    nid = np.full(len(hit), 6, np.int64)
    albedo = np.empty((3, len(hit)), np.float32)
    ks = np.zeros(len(hit), np.float32)
    shininess = np.full(len(hit), 16.0, np.float32)
    key = np.zeros(n, np.int32)  # segment id for outlines: kind and colour
    protect = np.zeros(n, bool)
    noise = look.noise

    if nf:
        f = face_buf[fi]
        axis, sign, kind, cell = faces.axis[f].astype(np.int64), faces.sign[f], faces.kind[f], faces.cell[f]
        nid[:nf] = 2 * axis + (sign > 0)
        fu, fv = u[fi], v[fi]
        px, py, pz = point[0, :nf], point[1, :nf], point[2, :nf]
        pa = np.where(axis == 0, py, px)  # plane coordinates of the face (OTHER_AXES)
        pb = np.where(axis == 2, py, pz)
        key[fi] = kind
        out = albedo[:, :nf]
        z, y, x = cell[:, 0], cell[:, 1], cell[:, 2]

        r = np.nonzero(kind == KIND_ROBOT)[0]
        if len(r):
            base = np.repeat(look.robot_rgb[:, None], len(r), 1)
            if look.robot_variation > 0:
                base *= 1 + look.robot_variation * (2 * _hash01(z[r], y[r], x[r], look.seed) - 1)
            ru, rv = fu[r], fv[r]
            if look.marking != "none":
                base[:, _marking_mask(look.marking, ru, rv, look.marking_size)] = look.marking_rgb[:, None]
            base[:, np.minimum(np.minimum(ru, 1 - ru), np.minimum(rv, 1 - rv)) < look.robot_edge_width] = look.robot_edge_rgb[:, None]
            out[:, r] = base
            ks[r] = look.robot_ks
            shininess[r] = look.robot_shininess

        wl = np.nonzero(kind == KIND_WALL)[0]
        if len(wl):
            base = terrain_albedo(look.terrain, pa[wl] + 31.0 * axis[wl], pb[wl], noise)
            side = axis[wl] != 2
            wu, wv = fu[wl], fv[wl]
            edge = np.minimum(np.minimum(wu, 1 - wu), np.minimum(wv, 1 - wv)) < 0.04
            base *= np.where(side, look.terrain_side, 1.0) * np.where(edge, 1 - look.terrain_edge, 1.0)
            zw, yw, xw = z[wl], y[wl], x[wl]
            paint_mask = np.nonzero(~side & (sign[wl] > 0) & (scene.zone[yw, xw] >= 0) & (scene.zone_level[yw, xw] == zw + 1))[0]
            if len(paint_mask):
                colour = scene.zone[yw[paint_mask], xw[paint_mask]]
                paint = look.zone_rgb[colour].T * (1 + 0.04 * (noise(pa[wl][paint_mask] * 6, pb[wl][paint_mask] * 6) - 0.5))
                base[:, paint_mask] = paint * 0.92 + base[:, paint_mask] * 0.08
                key[fi[wl[paint_mask]]] = 100 + colour
                protect[fi[wl[paint_mask]]] = True
            out[:, wl] = base

        ob = np.nonzero(kind == KIND_OBJECT)[0]
        if len(ob):
            zo, yo, xo = z[ob], y[ob], x[ob]
            colour = np.clip(scene.colour[zo, yo, xo].astype(np.int64), 0, len(PALETTE) - 1)
            base = look.object_rgb[colour].T.copy()
            if look.object_noise > 0:
                base *= 1 + look.object_noise * (noise(pa[ob] * 3 + 11 * axis[ob], pb[ob] * 3) - 0.5) * 2
            food = scene.food[zo, yo, xo]
            if food.any():
                ou, ov = fu[ob], fv[ob]
                seed_a = _hash01(zo * 7 + axis[ob], yo, xo, look.seed + 1)
                seed_b = _hash01(zo, yo * 5 + axis[ob], xo, look.seed + 2)
                hole = np.zeros(len(zo), bool)
                for cu, cv, rad in ((0.25 + 0.5 * seed_a, 0.25 + 0.5 * seed_b, look.hole_radius), (1 - 0.5 * seed_b, 0.5 * seed_a, look.hole_radius * 0.7)):
                    hole |= (ou - cu) ** 2 + (ov - cv) ** 2 < rad * rad
                base *= np.where(hole & food, look.hole_dark, 1.0)
            out[:, ob] = base
            ks[ob] = look.object_ks
            key[fi[ob]] = 10 + colour
            protect[fi[ob]] = True

        cl = np.nonzero(kind == KIND_CLUTTER)[0]
        if len(cl):
            base = look.clutter_rgb[np.clip(cell[cl, 0], 0, max(len(look.clutter_rgb) - 1, 0))].T
            out[:, cl] = base * (1 + 0.1 * (noise(pa[cl] * 2, pb[cl] * 2) - 0.5))
            key[fi[cl]] = 50
        if view.height_cue > 0:
            cue = view.height_cue * np.where(kind == KIND_OBJECT, 0.4, 1.0)
            out *= 1 - cue + cue * np.clip(pz, 0, 6) / 6

    if len(fl):
        x, y = point[0, nf:], point[1, nf:]
        out = albedo[:, nf:]
        m = look.sheet_margin
        sheet = (x >= -m) & (x <= w + m) & (y >= -m) & (y <= h + m)
        si = np.nonzero(sheet)[0]
        if len(si):
            out[:, si] = surface_albedo(look.floor, x[si], y[si], noise)
            ks[nf + si] = look.floor.ks
            shininess[nf + si] = look.floor.shininess
        ti = np.nonzero(~sheet)[0]
        if len(ti):
            tx, ty = x[ti], y[ti]
            part = surface_albedo(look.table, tx, ty, noise)
            for shape, cx, cy, rx, ry, angle, rgb in look.stickers:
                lx, ly = _rotate(tx - cx, ty - cy, -angle)
                if shape == "ellipse":
                    inside = (lx / rx) ** 2 + (ly / ry) ** 2 < 1
                elif shape == "rect":
                    inside = (np.abs(lx) < rx) & (np.abs(ly) < ry)
                else:
                    inside = (np.abs(lx) < rx * 2) & (np.abs(ly) < 0.12)
                part[:, inside] = rgb[:, None]
            far = np.maximum(np.maximum(-tx - m, tx - w - m), np.maximum(-ty - m, ty - h - m))
            fade = np.clip((far - 10) / 30, 0, 1)
            out[:, ti] = part * (1 - fade) + look.background[1][:, None] * fade
            ks[nf + ti] = look.table.ks
        xi = np.clip(np.floor(x).astype(np.int64), 0, w - 1)
        yi = np.clip(np.floor(y).astype(np.int64), 0, h - 1)
        zi = np.nonzero((x >= 0) & (x <= w) & (y >= 0) & (y <= h) & (scene.zone[yi, xi] >= 0) & (scene.zone_level[yi, xi] == 0))[0]
        if len(zi):
            colour = scene.zone[yi[zi], xi[zi]]
            paint = look.zone_rgb[colour].T * (1 + 0.04 * (noise(x[zi] * 6, y[zi] * 6) - 0.5))
            out[:, zi] = paint * 0.92 + out[:, zi] * 0.08
            key[fl[zi]] = 100 + colour
            protect[fl[zi]] = True
        gw = look.grid_width / 2
        grid = ((np.abs(x - np.round(x)) < gw) | (np.abs(y - np.round(y)) < gw)) & (x >= -gw) & (x <= w + gw) & (y >= -gw) & (y <= h + gw)
        out *= np.where(grid, 1 - look.grid_alpha, 1.0)
        out += look.grid_rgb[:, None] * (grid * look.grid_alpha)
        mi = np.nonzero(sheet & ((x < 0) | (x > w) | (y < 0) | (y > h)))[0]
        if len(mi):
            marker = mi[_corner_markers(look, x[mi], y[mi], w, h)]
            out[:, marker] = look.marker_rgb[:, None]
            key[fl[marker]] = 60

    colour = np.empty((3, n), np.float32)
    if len(hit):
        if view.style == "flat":
            shaded = albedo * FLAT_SHADE[nid]
        else:
            shaded = _light(look, view, faces, albedo, nid, point, ray, ks, shininess, d, h, w)
        for c in range(3):
            colour[c][hit] = shaded[c]
    sky = np.ones(n, bool)
    sky[hit] = False
    sky_px = np.nonzero(sky)[0]
    if len(sky_px):
        background = _background(look, _rows(direction, sky_px), len(sky_px))
        for c in range(3):
            colour[c][sky_px] = background[c]
    colour = colour.reshape(3, cam.height, cam.width)
    if view.outline > 0:
        depth_all = np.full(n, 1e4, np.float32)
        depth_all[hit] = t_hit
        colour *= _outline(key.reshape(cam.height, cam.width), depth_all.reshape(cam.height, cam.width), cam.ortho, view.outline)
    return np.ascontiguousarray(colour.transpose(1, 2, 0)), protect.reshape(cam.height, cam.width), cam


def _corner_markers(look, x, y, w, h):
    """bool per floor point: on one of the four corner markers just outside the arena."""
    size_m, gap = look.marker_size, 0.15
    marker = np.zeros(len(x), bool)
    for cx, ox in ((0.0, -1), (float(w), 1)):
        for cy, oy in ((0.0, -1), (float(h), 1)):
            a, b = (x - cx) * ox, (y - cy) * oy
            if look.marker == "L":
                th = 0.22 * size_m + 0.08
                marker |= ((a >= gap) & (a <= gap + th) & (b >= -size_m) & (b <= gap + th)) | ((b >= gap) & (b <= gap + th) & (a >= -size_m) & (a <= gap + th))
            elif look.marker == "square":
                marker |= (a >= gap) & (a <= gap + size_m) & (b >= gap) & (b <= gap + size_m)
            elif look.marker == "dot":
                rad = size_m / 2
                marker |= (a - gap - rad) ** 2 + (b - gap - rad) ** 2 < rad * rad
            else:
                c0, arm = gap + size_m / 2, 0.1 + 0.05 * size_m
                marker |= ((np.abs(a - c0) < arm) & (np.abs(b - c0) < size_m / 2)) | ((np.abs(b - c0) < arm) & (np.abs(a - c0) < size_m / 2))
    return marker


def _light(look, view, faces, albedo, nid, point, ray, ks, shininess, d, h, w):
    """Lit colour (3, n) of surface pixels: ambient, Lambert per light (toon: three bands), the
    main light shadowed, Blinn-Phong highlights where ks > 0 (lit style)."""
    first, rest = None, np.repeat(look.ambient[:, None], len(NORMALS), 1)  # (3, normal ids)
    lams = []
    for k, light in enumerate(look.lights):
        lam = np.clip(NORMALS @ light.direction, 0, 1)
        if view.style == "toon":
            lam = np.where(lam > 0.66, 1.0, np.where(lam > 0.25, 0.6, 0.25)).astype(np.float32)
        lams.append(lam)
        if k == 0:
            first = light.rgb[:, None] * lam
        else:
            rest = rest + light.rgb[:, None] * lam
    visibility = None
    colour = np.empty_like(albedo)
    if look.shadows:
        visibility = np.zeros(len(nid), np.float32)
        facing = np.nonzero((lams[0] > 0)[nid])[0]  # only these can be shadowed
        visibility[facing] = _shadow_lookup(look, faces, point[:, facing], nid[facing], d, h, w)
        for c in range(3):
            colour[c] = albedo[c] * (rest[c][nid] + visibility * first[c][nid])
    else:
        both = rest + first
        for c in range(3):
            colour[c] = albedo[c] * both[c][nid]
    if view.style != "lit":
        return colour
    s = np.nonzero(ks > 0)[0]
    if len(s):
        view_dir = -(ray[:, s] if ray.shape[1] > 1 else ray)
        view_dir = view_dir / np.sqrt((view_dir * view_dir).sum(0))
        ns = nid[s]
        normal = NORMALS.T[:, ns]
        weight = ks[s]
        for k, light in enumerate(look.lights):
            half = view_dir + light.direction[:, None]
            nh = np.clip((half * normal).sum(0) / np.sqrt((half * half).sum(0)), 1e-6, 1)
            spec = np.exp(shininess[s] * np.log(nh)) * (lams[k][ns] > 0) * weight
            if k == 0 and visibility is not None:
                spec = spec * visibility[s]
            for c in range(3):
                colour[c][s] += spec * light.rgb[c]
    return colour


def _background(look, ray, n):
    """(3, n) backdrop colour of rays (3, n) or (3, 1) that miss the floor."""
    top_rgb, bottom_rgb, blob_rgb, blob = look.background
    dd = np.broadcast_to(ray / np.sqrt((ray * ray).sum(0)), (3, n))
    up = np.clip(dd[2] * 2 + 0.5, 0, 1)
    bg = _mix(bottom_rgb, top_rgb, up)
    if blob > 0:
        az = np.arctan2(dd[0], dd[1])
        amount = np.clip((look.noise.fbm(az * 4 + 50, dd[2] * 8, 3) - 0.45) * 4, 0, 1) * blob
        bg = bg * (1 - amount) + blob_rgb[:, None] * amount
    return bg


def _outline(key, depth, ortho, strength):
    """(h, w) factor darkening pixels on a change of segment (kind, colour) or a depth step (the
    nearer side)."""
    edge = np.zeros(key.shape, bool)
    for a, b, da, db, ea, eb in (
        (key[:-1], key[1:], depth[:-1], depth[1:], edge[:-1], edge[1:]),
        (key[:, :-1], key[:, 1:], depth[:, :-1], depth[:, 1:], edge[:, :-1], edge[:, 1:]),
    ):
        limit = 0.45 if ortho else 0.025 * np.minimum(da, db)
        step = (a != b) | (np.abs(da - db) > limit)
        nearer = da <= db
        ea |= step & nearer
        eb |= step & ~nearer
    return np.where(edge, np.float32(1 - strength), np.float32(1))



# ---------------------------------------------------------------------------------------------
# Image-space effects


def _soft_mask(rng, kind, res):
    """A soft (res, res) mask in [0, 1] of a hand or a cable shadow, drawn at 64 x 64."""
    m = 64
    j, i = np.mgrid[0:m, 0:m].astype(np.float32) / m
    mask = np.zeros((m, m), np.float32)

    def capsule(ax, ay, bx, by, radius):
        px, py = i - ax, j - ay
        vx, vy = bx - ax, by - ay
        t = np.clip((px * vx + py * vy) / max(vx * vx + vy * vy, 1e-6), 0, 1)
        return np.hypot(px - t * vx, py - t * vy) < radius

    edge = int(rng.integers(4))
    entry = rng.uniform(0.2, 0.8)
    start = {0: (entry, -0.1), 1: (1.1, entry), 2: (entry, 1.1), 3: (-0.1, entry)}[edge]
    if kind == "hand":
        reach = rng.uniform(0.3, 0.6)
        inward = {0: (0, 1), 1: (-1, 0), 2: (0, -1), 3: (1, 0)}[edge]
        angle = rng.uniform(-0.5, 0.5)
        dx = inward[0] * math.cos(angle) - inward[1] * math.sin(angle)
        dy = inward[0] * math.sin(angle) + inward[1] * math.cos(angle)
        palm = (start[0] + dx * reach, start[1] + dy * reach)
        mask[capsule(start[0], start[1], palm[0], palm[1], rng.uniform(0.08, 0.13))] = 1
        for k in range(rng.integers(3, 6)):
            spread = (k - 2) * rng.uniform(0.25, 0.4)
            fx = dx * math.cos(spread) - dy * math.sin(spread)
            fy = dx * math.sin(spread) + dy * math.cos(spread)
            length = rng.uniform(0.12, 0.25)
            mask[capsule(palm[0], palm[1], palm[0] + fx * length, palm[1] + fy * length, rng.uniform(0.025, 0.04))] = 1
    else:
        end = (rng.uniform(0, 1), rng.uniform(0, 1))
        bend = (rng.uniform(0, 1), rng.uniform(0, 1))
        radius = rng.uniform(0.012, 0.035)
        steps = np.linspace(0, 1, 12)
        pts = [((1 - s) ** 2 * start[0] + 2 * (1 - s) * s * bend[0] + s * s * end[0], (1 - s) ** 2 * start[1] + 2 * (1 - s) * s * bend[1] + s * s * end[1]) for s in steps]
        for a, b in zip(pts[:-1], pts[1:]):
            mask[capsule(a[0], a[1], b[0], b[1], radius)] = 1
    small = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 2.5)))
    return np.asarray(small.resize((res, res), Image.BILINEAR), np.float32) / 255


def _bilinear(img, xs, ys):
    """Sample img (R, R, C) at pixel coordinates xs, ys (S, S) with edge clamping: (S, S, C)."""
    rh, rw, c = img.shape
    xs = np.clip(xs - 0.5, 0, rw - 1.001)
    ys = np.clip(ys - 0.5, 0, rh - 1.001)
    x0, y0 = xs.astype(np.int32), ys.astype(np.int32)
    fx, fy = (xs - x0).ravel(), (ys - y0).ravel()
    i00 = (y0 * rw + x0).ravel()
    out = np.empty((c, i00.size), np.float32)
    for k in range(c):
        flat = np.ascontiguousarray(img[..., k]).ravel()
        top = flat[i00] * (1 - fx) + flat[i00 + 1] * fx
        bottom = flat[i00 + rw] * (1 - fx) + flat[i00 + rw + 1] * fx
        out[k] = top * (1 - fy) + bottom * fy
    return out.T.reshape(xs.shape + (c,))


def warp(rgb, protect, size, k1=0.0, zoom=1.0, shift=(0.0, 0.0), chromatic=0.0):
    """Resample rgb (R, R, 3) and protect (R, R) to size x size through radial distortion k1
    (refit so the corners stay in view), a zoom (>= 1) and a shift (fractions of the image,
    keeping the crop inside the render) and chromatic aberration (red sampled at radial scale
    1 + chromatic, blue at 1 - chromatic). Returns (rgb float, protect bool)."""
    res = rgb.shape[0]
    if k1 == 0 and chromatic == 0:
        half = res / 2 / zoom
        cx, cy = res / 2 * (1 + 2 * shift[0]), res / 2 * (1 + 2 * shift[1])
        box = (max(cx - half, 0), max(cy - half, 0), min(cx + half, res), min(cy + half, res))
        image = Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)).resize((size, size), Image.BILINEAR, box=box)
        mask = Image.fromarray(protect.astype(np.uint8) * 255).resize((size, size), Image.BILINEAR, box=box)
        return np.asarray(image, np.float32) / 255, np.asarray(mask) > 0
    j, i = np.mgrid[0:size, 0:size].astype(np.float32)
    xn = (i + 0.5) / size * 2 - 1
    yn = (j + 0.5) / size * 2 - 1
    factor = (1 + k1 * (xn * xn + yn * yn)) / (1 + 2 * max(k1, 0.0)) / zoom
    xs, ys = xn * factor + 2 * shift[0], yn * factor + 2 * shift[1]
    channels = np.concatenate([rgb, protect[..., None].astype(np.float32)], -1)
    out = _bilinear(channels, (xs + 1) / 2 * res, (ys + 1) / 2 * res)
    if chromatic > 0:
        for channel, scale in ((0, 1 + chromatic), (2, 1 - chromatic)):
            out[..., channel] = _bilinear(rgb[..., channel : channel + 1], (xs * scale + 1) / 2 * res, (ys * scale + 1) / 2 * res)[..., 0]
    return out[..., :3], out[..., 3] > 0.05


def corrupt(rgb, protect, rng, spec):
    """The per-image corruptions of the module docstring on rgb (S, S, 3) float in [0, 1]; protect
    (S, S) bool marks pixels a cutout must not cover. Returns a PIL image."""
    size = rgb.shape[0]

    def amt(item):
        return spec.amount(item)

    def chance(item, p):
        level = amt(item)
        return level > 0 and rng.random() < p * level

    if chance("motion_blur", 0.25):
        length = rng.uniform(2, 2 + 6 * amt("motion_blur"))
        angle = rng.uniform(0, math.pi)
        taps = max(2, int(round(length)))
        acc = np.zeros_like(rgb)
        for t in np.linspace(-length / 2, length / 2, taps):
            dx, dy = int(round(t * math.cos(angle))), int(round(t * math.sin(angle)))
            acc += np.roll(rgb, (dy, dx), axis=(0, 1))
        rgb = acc / taps
    if chance("defocus", 0.3):
        image = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.3 + 1.3 * amt("defocus"))))
        rgb = np.asarray(image, np.float32) / 255
    if chance("vignetting", 0.5):
        j, i = np.mgrid[0:size, 0:size].astype(np.float32)
        r2 = ((i / size - 0.5) ** 2 + (j / size - 0.5) ** 2) * 2
        rgb = rgb * (1 - rng.uniform(0, 0.45 * amt("vignetting")) * r2)[..., None]
    if chance("exposure", 0.8):
        rgb = rgb * 2 ** rng.uniform(-0.6 * amt("exposure"), 0.6 * amt("exposure"))
    sigma = rng.uniform(0, 0.035 * amt("gaussian_noise")) if chance("gaussian_noise", 0.8) else 0.0
    shot = rng.uniform(0, 0.02 * amt("shot_noise")) if chance("shot_noise", 0.6) else 0.0
    if sigma > 0 or shot > 0:
        std = np.sqrt(sigma * sigma + shot * np.clip(rgb, 0, None))
        rgb = rgb + std * rng.standard_normal(rgb.shape, dtype=np.float32)
    if chance("white_balance", 0.6):
        level = 0.1 * amt("white_balance")
        rgb = rgb * rng.uniform(1 - level, 1 + level, 3).astype(np.float32)
    rgb = np.clip(rgb, 0, 1)
    if chance("gamma", 0.6):
        rgb = rgb ** rng.uniform(1 - 0.25 * amt("gamma"), 1 + 0.25 * amt("gamma"))
    if chance("cutout", 0.4):
        rgb = rgb.copy()
        for _ in range(int(rng.integers(1, 4))):
            side = max(2, int(size * rng.uniform(0.04, 0.12)))
            for _ in range(12):
                y0, x0 = int(rng.integers(0, size - side)), int(rng.integers(0, size - side))
                if not protect[y0 : y0 + side, x0 : x0 + side].any():
                    rgb[y0 : y0 + side, x0 : x0 + side] = unnameable_colour(rng) if rng.random() < 0.5 else rng.random((side, side, 3))  # a flat patch must not look like a named object
                    break
    image = Image.fromarray((rgb * 255 + 0.5).astype(np.uint8))
    if chance("jpeg", 0.6):
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=int(rng.uniform(95 - 65 * amt("jpeg"), 95)))
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    return image


# ---------------------------------------------------------------------------------------------
# Public


def pose_of(look, cameras):
    """The pose dict of a look (module docstring "Pose")."""
    views = []
    for view, cam in zip(look.views, cameras):
        views.append(
            {
                "camera": view.name,
                "projection": "orthographic" if cam.ortho else "perspective",
                "azimuth": round(view.azimuth, 2),
                "elevation": round(view.elevation, 2),
                "fov": round(view.fov, 2) if not cam.ortho else None,
                "position": [round(float(c), 3) for c in cam.position],
                "style": view.style,
            }
        )
    main = next((v for v in look.views if v.name == "perspective"), look.views[0])
    return {"azimuth": round(main.azimuth, 2), "side": side_of(main.azimuth), "elevation": round(main.elevation, 2), "views": views}


_CLEAN_LOOKS = {}


def _clean_look(spec):
    """The look of a clean spec (deterministic, so kept)."""
    slot = (tuple(spec.cameras), spec.size, spec.azimuth)
    if slot not in _CLEAN_LOOKS:
        _CLEAN_LOOKS[slot] = sample_look(np.random.default_rng(0), spec)
    return _CLEAN_LOOKS[slot]


def scene_views(obs, rng, spec=None, look=None):
    """(images, pose): one PIL image (spec.size square) per camera of spec.cameras and the pose
    of the cameras (module docstring). look: the episode-level draw (sample_look) to reuse; None
    draws a new one."""
    spec = ViewSpec() if spec is None else spec
    if look is None:
        look = _clean_look(spec) if all(spec.level(group) == 0 for group in GROUPS) else sample_look(rng, spec)
    scene = scene_of(obs)
    faces = voxel_faces(scene.kind)
    if look.clutter:
        faces = _concat([faces, box_faces(look.clutter)])
    images, cameras = [], []
    for view in look.views:
        rgb, protect, cam = render_view(scene, faces, look, view, spec.size)
        res = rgb.shape[0]
        for item in ("hand_shadow", "cable_shadow"):
            level = spec.amount(item)
            if level > 0 and rng.random() < 0.3 * level:
                mask = _soft_mask(rng, "hand" if item == "hand_shadow" else "cable", res)
                rgb = rgb * (1 - rng.uniform(0.25, 0.5) * mask)[..., None]
        chromatic = rng.uniform(0, 0.006 * spec.amount("chromatic")) if rng.random() < 0.4 * spec.amount("chromatic") else 0.0
        if res != spec.size or view.k1 != 0 or view.zoom != 1 or view.shift != (0.0, 0.0) or chromatic > 0:
            rgb, protect = warp(rgb, protect, spec.size, view.k1, view.zoom, view.shift, chromatic)
        images.append(corrupt(rgb.astype(np.float32), protect, rng, spec))
        cameras.append(cam)
    return images, pose_of(look, cameras)


@dataclass
class StateNoise:
    """Per-robot rates of state-estimation errors (perturb_state)."""

    drop: float = 0.02
    shift: float = 0.02
    phantom: float = 0.01


def perturb_state(obs, rng, noise=None):
    """A copy of obs (C, D, H, W) with the robot channel corrupted as imperfect state estimation
    would: robots missing, moved one voxel to a free face neighbour, phantom robots next to
    something solid (StateNoise rates). Only channel 0 changes. Off unless called."""
    noise = StateNoise() if noise is None else noise
    torch_input = hasattr(obs, "detach")
    arr = obs.detach().cpu().numpy().copy() if torch_input else np.array(obs, copy=True)
    robot = arr[0] > 0.5
    occupied_cells = (arr[1] > 0.5) | (arr[2] > 0.5)
    d, h, w = robot.shape
    zs, ys, xs = np.nonzero(robot)
    order = rng.permutation(len(zs))
    steps = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
    out = robot.copy()
    for k in order:
        cell = (zs[k], ys[k], xs[k])
        roll = rng.random()
        if roll < noise.drop:
            out[cell] = False
        elif roll < noise.drop + noise.shift:
            for step in steps[rng.permutation(6)]:
                z, y, x = cell[0] + step[0], cell[1] + step[1], cell[2] + step[2]
                if 0 <= z < d and 0 <= y < h and 0 <= x < w and not out[z, y, x] and not occupied_cells[z, y, x]:
                    out[cell] = False
                    out[z, y, x] = True
                    break
    phantoms = int(rng.binomial(len(zs), noise.phantom)) if len(zs) else 0
    if phantoms:
        solid = out | occupied_cells
        support = np.zeros_like(solid)
        support[0] = True
        support[1:] |= solid[:-1]
        for axis in (1, 2):
            support |= np.roll(solid, 1, axis) | np.roll(solid, -1, axis)
        free = np.argwhere(support & ~solid)
        for k in rng.choice(len(free), size=min(phantoms, len(free)), replace=False):
            out[tuple(free[k])] = True
    arr[0] = out.astype(arr.dtype)
    if torch_input:
        import torch

        return torch.from_numpy(arr).to(obs.device)
    return arr
