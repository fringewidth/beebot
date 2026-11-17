"""Camera views with domain randomization (vla/voxel/views.py, README "Views and randomization"):
the interface and pose, the world frame of both cameras, clean mode, colour jitter that keeps
colour names, cutouts that spare targets, and state noise."""

import numpy as np
import pytest
import torch

from beebot.vla.voxel import views as V
from beebot.vla.voxel.world import COLOUR_NAMES, MAX_OBJECTS, N_CHANNELS, OLD_CHANNELS, VoxelWorld, observe, sample_object_scenes, with_objects

SIZE = 16


@pytest.fixture(scope="module")
def object_obs():
    world, _ = sample_object_scenes(np.random.default_rng(3), 2)
    return observe(world)


def one_red_object(y, x):
    """A world with one red 2 x 2 x 2 object at floor cell (y, x) and no robots."""
    k = MAX_OBJECTS
    world = VoxelWorld(
        torch.zeros(1, SIZE, SIZE, SIZE),
        torch.zeros(1, SIZE, SIZE, SIZE, dtype=torch.bool),
        torch.zeros(1, k, 3, dtype=torch.long),
        torch.zeros(1, k, dtype=torch.bool),
        torch.full((1, k), -1, dtype=torch.long),
        torch.tensor([False]),
    )
    world = with_objects(world)
    world.cheese_pos[0, 0] = torch.tensor([0, y, x])
    world.cheese_alive[0, 0] = True
    world.cheese_color[0, 0] = COLOUR_NAMES.index("red")
    world.food[0, 0] = False
    return observe(world)[0]


def red_centroid(image):
    a = np.asarray(image, np.float32) / 255
    mask = (a[..., 0] > 0.55) & (a[..., 1] < 0.35) & (a[..., 2] < 0.35)
    assert mask.sum() > 4
    ys, xs = np.nonzero(mask)
    return xs.mean() / a.shape[1], ys.mean() / a.shape[0]


def test_default_views_and_pose(object_obs):
    images, pose = V.scene_views(object_obs[0], np.random.default_rng(0), V.ViewSpec(size=80))
    assert len(images) == 2 and all(im.size == (80, 80) and im.mode == "RGB" for im in images)
    assert [v["camera"] for v in pose["views"]] == ["perspective", "top"]
    az = pose["azimuth"]
    assert min(abs((az - c + 180) % 360 - 180) for c in (0, 90, 180, 270)) <= 30
    assert pose["side"] == V.side_of(az)
    one, _ = V.scene_views(object_obs[0], np.random.default_rng(0), V.ViewSpec(size=64, cameras=("top",)))
    assert len(one) == 1


def test_azimuths_stay_away_from_diagonals():
    rng = np.random.default_rng(0)
    az = np.array([V._allowed_azimuth(rng) for _ in range(4000)])
    gap = np.min([np.abs((az - d + 180) % 360 - 180) for d in (45, 135, 225, 315)], axis=0)
    assert gap.min() >= 15
    assert np.histogram((az + 45) % 360, bins=4, range=(0, 360))[0].min() > 800  # every side drawn
    assert [V.side_of(a) for a in (0, 80, 190, 280, 350)] == ["front", "right", "back", "left", "front"]


def test_channel_counts_and_inputs(object_obs):
    spec = V.ViewSpec.clean(size=64)
    reference = None
    for channels in (OLD_CHANNELS, 7, N_CHANNELS):
        obs = object_obs[0, :channels]
        for value in (obs, obs.numpy()):
            images, _ = V.scene_views(value, np.random.default_rng(1), spec)
            assert len(images) == 2
        if channels == OLD_CHANNELS:
            reference = np.asarray(images[0])
    with pytest.raises(ValueError):
        V.scene_of(object_obs[0, :5])
    assert reference is not None


def test_clean_is_deterministic(object_obs):
    spec = V.ViewSpec.clean(size=64)
    a, pose_a = V.scene_views(object_obs[1], np.random.default_rng(1), spec)
    b, pose_b = V.scene_views(object_obs[1], np.random.default_rng(2), spec)
    assert all((np.asarray(x) == np.asarray(y)).all() for x, y in zip(a, b))
    assert pose_a == pose_b and pose_a["azimuth"] == 0


def test_randomized_views_differ_and_items_turn_off(object_obs):
    spec = V.ViewSpec(strength=1.0, size=64)
    a, _ = V.scene_views(object_obs[0], np.random.default_rng(1), spec)
    b, _ = V.scene_views(object_obs[0], np.random.default_rng(2), spec)
    assert (np.asarray(a[0]) != np.asarray(b[0])).any()
    _, pose = V.scene_views(object_obs[0], np.random.default_rng(1), V.ViewSpec(strength=1.0, size=48, off=frozenset({"azimuth"})))
    assert pose["azimuth"] == pytest.approx(0) or pose["views"][0]["azimuth"] == 0
    assert V.ViewSpec(strength=1.0, camera=0.0).amount("fov") == 0 and V.ViewSpec(strength=0.3).amount("jpeg") == 0.3


@pytest.mark.parametrize("azimuth, perspective_side, top_corner", [(0, "left", (0, 0)), (180, "right", (1, 1)), (90, None, (1, 0))])
def test_world_frame_of_both_cameras(azimuth, perspective_side, top_corner):
    """A red block at the back left (small x and y): left in the front camera's image, top left
    in the top view; a camera at the back sees it on the right and the top view turns with it."""
    obs = one_red_object(1, 1)
    images, pose = V.scene_views(obs, np.random.default_rng(0), V.ViewSpec.clean(size=96, azimuth=azimuth))
    assert pose["azimuth"] == azimuth and pose["views"][1]["azimuth"] == azimuth
    if perspective_side:
        cx, _ = red_centroid(images[0])
        assert (cx < 0.5) == (perspective_side == "left")
    tx, ty = red_centroid(images[1])
    assert (tx > 0.5, ty > 0.5) == (bool(top_corner[0]), bool(top_corner[1]))


def test_colour_jitter_keeps_names():
    assert set(V.COLOUR_JITTER) == set(COLOUR_NAMES)
    rng = np.random.default_rng(0)
    for name in COLOUR_NAMES:
        assert V.nearest_colour_name(V.jitter_colour(name, rng, 0.0)) == name
        draws = np.stack([V.jitter_colour(name, rng, 1.0) for _ in range(400)])
        assert (V.nearest_colour_name(draws) == name).all(), name
        hue, (s_lo, s_hi), (v_lo, v_hi), tint = V.COLOUR_JITTER[name]
        h, s, v = V.rgb_to_hsv(V.PALETTE[COLOUR_NAMES.index(name)])
        for dh in np.linspace(-hue, hue, 5):
            for fs in np.linspace(s_lo, s_hi, 5):
                for fv in np.linspace(v_lo, v_hi, 5):
                    corner = V.hsv_to_rgb(np.array([h + dh / 360, min(s * fs + tint, 1), min(v * fv, 1)]))
                    assert V.nearest_colour_name(corner) == name, (name, dh, fs, fv)


def test_clean_colours_contrast():
    """The clean sheet is far from every object colour (white cheeses were missed on a light
    floor), robots and terrain are apart from them too."""
    assert (V.lab_distance(V.CLEAN_FLOOR[None], V.PALETTE) >= 30).all()
    assert (V.lab_distance(V.CLEAN_ROBOT[None], V.PALETTE) >= 15).all()
    assert (V.lab_distance(V.CLEAN_TERRAIN[None], V.PALETTE) >= 15).all()
    rng = np.random.default_rng(0)
    for _ in range(20):
        look = V.sample_look(rng, V.ViewSpec(strength=1.0))
        assert look.grid_alpha >= 0.15 and look.marker_size >= 0.4  # never removed
        assert V.lab_distance(look.marker_rgb, look.floor.rgb) > 20


def test_distractors_are_unnameable():
    """Clutter, stickers, the table and the backdrop never take a colour near a named one, and
    clutter is never cube-like (a flat paper or book, or a thin bar)."""
    rng = np.random.default_rng(0)
    colours, boxes = [], []
    for _ in range(40):
        look = V.sample_look(rng, V.ViewSpec(strength=1.0))
        colours += list(look.clutter_rgb) + [s[-1] for s in look.stickers] + list(look.background[:3]) + [look.table.rgb, look.table.rgb2]
        boxes += look.clutter
    assert len(boxes) > 50 and len(colours) > 200
    colours = np.stack(colours)
    assert (V.lab_distance(colours[:, None], V.PALETTE[None]) >= V.DISTRACTOR_GAP).all()
    for x0, x1, y0, y1, z0, z1 in boxes:
        sides = sorted((x1 - x0, y1 - y0, z1 - z0))
        assert sides[0] <= 0.6 and sides[0] < 0.5 * sides[2]


def test_cutout_spares_protected_pixels():
    spec = V.ViewSpec(strength=1.0, off=frozenset(i for g in V.ITEMS.values() for i in g if i != "cutout"))
    rgb = np.full((64, 64, 3), 0.5, np.float32)
    protect = np.ones((64, 64), bool)
    protect[:, 40:] = False
    any_change = False
    for seed in range(30):
        out = np.asarray(V.corrupt(rgb, protect, np.random.default_rng(seed), spec), np.float32) / 255
        assert np.abs(out[protect] - 0.5).max() < 0.01
        any_change |= bool(np.abs(out[~protect] - 0.5).max() > 0.01)
    assert any_change


def test_state_noise():
    world, _ = sample_object_scenes(np.random.default_rng(5), 1)
    obs = observe(world)[0]
    rng = np.random.default_rng(0)
    robots = int(obs[0].sum())
    no_robots = V.perturb_state(obs, rng, V.StateNoise(drop=1.0, shift=0.0, phantom=0.0))
    assert isinstance(no_robots, torch.Tensor) and no_robots[0].sum() == 0 and torch.equal(no_robots[1:], obs[1:])
    after_shift = V.perturb_state(obs.numpy(), rng, V.StateNoise(drop=0.0, shift=1.0, phantom=0.0))
    assert after_shift[0].sum() == robots and (after_shift[0] != obs[0].numpy()).any()
    assert not ((after_shift[0] > 0.5) & ((after_shift[1] > 0.5) | (after_shift[2] > 0.5))).any()
    extra = V.perturb_state(obs, rng, V.StateNoise(drop=0.0, shift=0.0, phantom=1.0))
    assert extra[0].sum() > robots
    same = V.perturb_state(obs, rng, V.StateNoise(0.0, 0.0, 0.0))
    assert torch.equal(same, obs)
