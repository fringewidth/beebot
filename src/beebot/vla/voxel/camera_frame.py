"""The camera's frame: the world turned about the vertical axis so that the camera stands at its
front. The user always stands at the camera, and the policy works entirely in this frame: its
voxel observation, its action codes and every word it reads or writes.

Why. The policy sees rendered images (views.scene_views) and the voxel grid. Turning the scene and
the camera together by a quarter turn gives the same images (the four corner markers look alike
and the top view turns with the camera), so the board's own sides cannot be seen. What can be
seen is where things are relative to the camera, so that is the frame the policy gets.

Convention (views.py's world frame: x to the right, y to the front, z up, as a person at the front
sees it; a voxel is (z, y, x)). The camera's quarter turn k = quarter_turns(azimuth) names the board
side it looks from (views.side_of): 0 front (+y), 1 right (+x), 2 back (-y), 3 left (-x). In the
camera's frame the camera stands at the front (+y'), its right is +x', up is still z:

    a world direction (dx, dy) is (dx', dy') = (dx cos a - dy sin a, dx sin a + dy cos a), a = 90 k
    degrees; k = 1 (camera at the world's right): dx' = -dy, dy' = dx, so the world's front (+y)
    is the camera's left and the world's right (+x, nearest the camera) is the camera's front.
    a voxel (z, y, x) of a D x H x W grid, k = 1: (z, x, H - 1 - y) of the D x W x H turned grid.

Words in the camera's frame: "left" and "right" are the camera's; "front" / "near side" the side
nearest the camera (+y'), "back" / "far side" the side away from it (-y'); "forward", "ahead",
"away" and "north" point away from the camera (-y'), "toward me", "backward" and "south" toward it
(+y'); east is the camera's right (+x'), west its left (-x'); "up" and "down" are height. A world
direction word w is camera_word(w, k) there (think_data.in_user_frame turns whole texts).

Policy actions. turn_codes maps world-frame action codes (world.DIR_TABLE, STAY 26) into the camera's
frame (the training targets); unturn_codes maps the policy's camera-frame codes back into the world
frame before the world (or the MuJoCo cubes) steps:

    obs_policy = policy_observation(observe(world)[i], pose["azimuth"])  # what the policy sees
    codes_world = world_actions(codes_policy, pose["azimuth"])           # what the world (or MuJoCo) steps with

(policy_observation and world_actions are turn_grid and unturn_codes with k = quarter_turns of the
azimuth; rollout.PolicyActor already returns world-frame codes.)

CODE_TURNS[k][c] is the camera-frame code of world code c (STAY stays STAY); CODE_UNTURNS[k] is its
inverse. turn_world turns a whole VoxelWorld (for tests: world.step commutes with the turn, up to
the step's own axis tie-breaks, see tests/test_vla_voxel_camera_frame.py).
"""

import numpy as np
import torch

from beebot.vla.voxel.world import BOX, CHEESE_SIDE, DIR_TABLE, STAY

SIDES = ("front", "right", "back", "left")  # board sides counterclockwise from the default viewer's (+y), as views.side_of names them


def quarter_turns(degrees):
    """The quarter turn (0 front, 1 right, 2 back, 3 left) of a place around the arena at this
    azimuth, nearest first (a place half way between two goes to the later one)."""
    return int(((float(degrees) + 45.0) % 360.0) // 90.0)


def camera_word(word, turns):
    """A world direction word (front, right, back, left) as said in the frame of a camera at
    quarter turn `turns`."""
    return SIDES[(SIDES.index(word) - turns) % 4]


def world_word(word, turns):
    """A camera-frame direction word (front, right, back, left) as the world's own side."""
    return SIDES[(SIDES.index(word) + turns) % 4]


def turn_vector(dz, dy, dx, turns):
    """A world direction (dz, dy, dx) in the camera's frame."""
    for _ in range(turns % 4):
        dy, dx = dx, -dy
    return dz, dy, dx


def _code_turns():
    table = [tuple(int(v) for v in row) for row in DIR_TABLE.tolist()]
    out = torch.empty(4, len(table), dtype=torch.long)
    for k in range(4):
        for code, (dz, dy, dx) in enumerate(table):
            out[k, code] = table.index(turn_vector(dz, dy, dx, k))
    return out


CODE_TURNS = _code_turns()  # (4, 27): CODE_TURNS[k][c] the camera-frame code of world code c
CODE_UNTURNS = torch.stack([torch.argsort(row) for row in CODE_TURNS])  # (4, 27): its inverse, camera code -> world code
assert all(int(CODE_TURNS[k, STAY]) == STAY for k in range(4))


def turn_grid(grid, turns):
    """grid (..., H, W) (any leading dims, e.g. (C, D, H, W); torch or numpy) in the camera's
    frame: voxel (y, x) goes to (x, H - 1 - y) per quarter turn (module docstring)."""
    k = turns % 4
    if isinstance(grid, np.ndarray):
        return np.ascontiguousarray(np.rot90(grid, k, axes=(-1, -2))) if k else grid
    return torch.rot90(grid, k, dims=(-1, -2)) if k else grid


def unturn_grid(grid, turns):
    """The inverse of turn_grid: a camera-frame grid back in the world's frame."""
    return turn_grid(grid, -turns)


def _recode(codes, table):
    if isinstance(codes, np.ndarray):
        return table.numpy()[codes.astype(np.int64)].astype(codes.dtype)
    return table.to(codes.device)[codes.long()].to(codes.dtype)


def turn_codes(codes, turns):
    """World-frame action codes (..., H, W) in the camera's frame: the grid turned and each code
    replaced by its turned direction (CODE_TURNS)."""
    k = turns % 4
    return _recode(turn_grid(codes, k), CODE_TURNS[k]) if k else codes


def unturn_codes(codes, turns):
    """The policy's camera-frame action codes (..., H, W) back in the world's frame: what the world
    steps with. The inverse of turn_codes."""
    k = turns % 4
    return _recode(unturn_grid(codes, k), CODE_UNTURNS[k]) if k else codes


def policy_observation(obs, azimuth):
    """A world-frame observation (C, D, H, W) (world.observe of one scene; torch or numpy) as the
    policy sees it: in the frame of the camera at `azimuth` (degrees, pose["azimuth"] of the
    episode's views)."""
    return turn_grid(obs, quarter_turns(azimuth))


def world_actions(codes, azimuth):
    """The policy's action codes (camera frame, (..., D, H, W) long in 0..26) turned back into the
    world's frame, for the camera at `azimuth` (degrees: pose["azimuth"] of the episode's views,
    rollout.Played.camera): what world.step, or the MuJoCo cubes, execute. The inverse of the turn
    that made the training targets (turn_codes)."""
    return unturn_codes(codes, quarter_turns(azimuth))


def _turn_points(y, x, h, w, turns):
    """Voxel coordinates (y, x) of an h x w grid in the turned grid, and its new (h, w)."""
    for _ in range(turns % 4):
        y, x, h, w = x, h - 1 - y, w, h
    return y, x, h, w


def _pack(form, pos):
    """Forms (B, K, BOX, BOX, BOX) shifted so that each object's voxels start at the box's low
    corner in y and x (as world.object_form makes them), pos moved to keep them in place."""
    form, pos = form.clone(), pos.clone()
    for b, k in (form.flatten(2).any(dim=2)).nonzero().tolist():
        _, ys, xs = form[b, k].nonzero(as_tuple=True)
        dy, dx = int(ys.min()), int(xs.min())
        if dy or dx:
            form[b, k] = torch.roll(form[b, k], shifts=(-dy, -dx), dims=(1, 2))
            pos[b, k, 1] += dy
            pos[b, k, 2] += dx
    return form, pos


def turn_world(world, turns):
    """A VoxelWorld in the camera's frame: robots, terrain, zones and objects turned (each object's
    form turned in its box and packed to the box's low corner; plain cheeses keep their 2 x 2 x 2
    form). Slot properties (colours, pattern, friction, food) stay with their slot. A patterned
    object's paint follows its form's axes (world.paint_of), so its colours may lie differently
    from the turned observation's; motion, eating and occupancy turn exactly."""
    from dataclasses import replace

    k = turns % 4
    if not k:
        return world
    h, w = world.mass.shape[-2:]
    pos = world.cheese_pos
    side = CHEESE_SIDE if world.form is None else BOX
    y0, x0, _, _ = _turn_points(pos[..., 1], pos[..., 2], h, w, k)
    y1, x1, _, _ = _turn_points(pos[..., 1] + side - 1, pos[..., 2] + side - 1, h, w, k)
    new_pos = torch.stack([pos[..., 0], torch.minimum(y0, y1), torch.minimum(x0, x1)], dim=-1)
    form = None
    if world.form is not None:
        form, new_pos = _pack(turn_grid(world.form, k), new_pos)
    zones = None if world.zones is None else turn_grid(world.zones, k)
    return replace(world, mass=turn_grid(world.mass, k), walls=turn_grid(world.walls, k), cheese_pos=new_pos, form=form, zones=zones)
