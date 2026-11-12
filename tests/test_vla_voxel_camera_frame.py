"""The camera's frame (vla/voxel/camera_frame.py): the grid and the action codes turn together, the
world's step commutes with the turn, and the camera-frame words name the turned grid's sides."""

import numpy as np
import torch

from beebot.vla.voxel import camera_frame as C
from beebot.vla.voxel import suite as S
from beebot.vla.voxel import world as W
from beebot.vla.voxel.think_data import in_user_frame


def test_the_convention_on_one_voxel_and_one_move():
    grid = torch.zeros(2, 4, 5)  # D, H 4, W 5
    grid[1, 0, 3] = 1  # at the world's back (y 0)
    turned = C.turn_grid(grid, 1)  # a camera at the world's right (+x)
    assert turned.shape == (2, 5, 4) and (turned == 1).nonzero().tolist() == [[1, 3, 3]]  # (z, x, H - 1 - y)
    assert np.array_equal(C.turn_grid(grid.numpy(), 1), turned.numpy())
    east = W.DIRS.index((0, 0, 1))
    assert W.DIR_TABLE[C.CODE_TURNS[1][east]].tolist() == [0, 1, 0]  # toward that camera: the camera frame's front (+y')
    assert W.DIR_TABLE[C.CODE_TURNS[1][W.DIRS.index((0, 1, 0))]].tolist() == [0, 0, -1]  # the world's front: the camera's left
    assert C.quarter_turns(0) == 0 and C.quarter_turns(90) == 1 and C.quarter_turns(185) == 2 and C.quarter_turns(-80) == 3
    assert [C.camera_word("front", k) for k in range(4)] == ["front", "left", "back", "right"]
    assert all(C.world_word(C.camera_word(w, k), k) == w for w in C.SIDES for k in range(4))


def test_world_actions_turn_the_policys_codes_back_with_the_camera_pose():
    """The closed loop's helpers: a camera at azimuth 100 (the world's right, quarter turn 1); the
    policy's move to the camera's right (+x') is the world's back (-y)."""
    right = torch.full((2, 3, 3), W.DIRS.index((0, 0, 1)))
    assert (C.world_actions(right, 100.0) == W.DIRS.index((0, -1, 0))).all()
    assert (C.world_actions(right, 350.0) == W.DIRS.index((0, 0, 1))).all()  # the default camera: unchanged
    obs = torch.rand(4, 2, 3, 3)
    for azimuth in (10.0, 100.0, 190.0, 280.0):
        codes = torch.randint(0, W.N_ACTIONS, (2, 3, 3), generator=torch.Generator().manual_seed(int(azimuth)))
        k = C.quarter_turns(azimuth)
        assert torch.equal(C.world_actions(C.turn_codes(codes, k), azimuth), codes)
        assert torch.equal(C.policy_observation(obs, azimuth), C.turn_grid(obs, k))


def test_codes_permute_with_stay_fixed_and_unturn_inverts():
    for k in range(4):
        table = C.CODE_TURNS[k]
        assert sorted(table.tolist()) == list(range(W.N_ACTIONS)) and int(table[W.STAY]) == W.STAY
        assert torch.equal(W.DIR_TABLE[table][:, 0], W.DIR_TABLE[:, 0])  # height never turns
        assert torch.equal(C.CODE_TURNS[k][C.CODE_UNTURNS[k]], torch.arange(W.N_ACTIONS))
        codes = torch.randint(0, W.N_ACTIONS, (3, 6, 6), generator=torch.Generator().manual_seed(k))
        assert torch.equal(C.unturn_codes(C.turn_codes(codes, k), k), codes)
        assert np.array_equal(C.unturn_codes(C.turn_codes(codes.numpy(), k), k), codes.numpy())
    for k in range(4):  # a turn of k then of 1 is a turn of k + 1
        assert torch.equal(C.CODE_TURNS[1][C.CODE_TURNS[k]], C.CODE_TURNS[(k + 1) % 4])


def test_one_robot_moves_where_its_turned_code_says_in_the_turned_grid():
    for code in range(W.N_ACTIONS):
        mass = torch.zeros(5, 5, 5)
        mass[2, 2, 2] = 1
        world = W.world_from_arrays(mass)
        codes = torch.full((1, 5, 5, 5), W.STAY, dtype=torch.long)
        codes[0, 2, 2, 2] = code
        after, _ = W.step(world, codes)
        for k in range(4):
            turned, _ = W.step(C.turn_world(world, k), C.turn_codes(codes, k))
            assert torch.equal(turned.mass, C.turn_grid(after.mass, k)), (code, k)


def horizontal_ties(world, codes, score):
    """(B,) scenes where some object's press is as strong along y as along x (and stronger than
    along z) and reaches its friction: world.push_cheeses then takes y (the first axis), a choice
    tied to the world's axes that no turn can carry over."""
    _, force = W.move_robots(world, codes, W.cheese_id_map(world), score)
    a = force.abs()
    friction = world.friction if world.friction is not None else torch.full(a.shape[:2], W.FRICTION)
    return ((a[..., 1] == a[..., 2]) & (a[..., 1] > a[..., 0]) & (a[..., 1] >= friction)).any(dim=1)


def same_world(a, b):
    return torch.equal(a.mass, b.mass) and torch.equal(W.cheese_masks(a), W.cheese_masks(b)) and torch.equal(a.cheese_alive, b.cheese_alive)


def test_a_turned_scene_with_turned_codes_steps_as_the_turned_step():
    """For every quarter turn: step(turn(world), turn(codes), turn(move order)) == turn(step(world,
    codes, move order)), on random scenes (terrain, cheeses, objects of every shape, zones, gravity
    on and off) with a random action for every robot. The only exceptions are the step's own ties
    of a press equally strong along y and x (horizontal_ties), which are skipped and rare."""
    rng = np.random.default_rng(0)
    compared = skipped = 0
    for draw in range(4):
        world = W.sample_scenes(rng, 12)[0] if draw % 2 == 0 else W.sample_object_scenes(rng, 12)[0]
        codes = torch.randint(0, W.N_ACTIONS, world.mass.shape, generator=torch.Generator().manual_seed(draw))
        score = torch.rand(world.mass.shape, generator=torch.Generator().manual_seed(100 + draw))
        after, info = W.step(world, codes, score)
        tied = horizontal_ties(world, codes, score)
        for k in range(4):
            turned = C.turn_world(world, k)
            assert torch.equal(W.observe(turned)[:, :3], C.turn_grid(W.observe(world)[:, :3], k))  # robots, walls, objects
            stepped, turned_info = W.step(turned, C.turn_codes(codes, k), C.turn_grid(score, k))
            expected = C.turn_world(after, k)
            for i in range(world.mass.shape[0]):
                if tied[i]:
                    skipped += 1
                    continue
                compared += 1
                assert same_world(stepped.index(i), expected.index(i)), (draw, k, i)
                assert torch.equal(turned_info.eaten[i], info.eaten[i]) and torch.equal(turned_info.pushed[i], info.pushed[i])
    assert compared >= 0.9 * (compared + skipped)


def test_camera_words_name_the_turned_grids_sides():
    """A world region said in the camera's frame (think_data.in_user_frame with the camera's quarter
    turn) is where that region lies in the turned grid: "my left" is the turned grid's small-x side,
    "the front" (near side) its large-y side, and the side the camera looks from is its front."""

    def region_of(y, x):
        return next(name for name in S.FLOOR_REGIONS if S.region_box(name)[0] <= y < S.region_box(name)[1] and S.region_box(name)[2] <= x < S.region_box(name)[3])

    for k in range(4):
        for name in S.FLOOR_REGIONS:
            y0, y1, x0, x1 = S.region_box(name)
            mask = torch.zeros(1, 16, 16)
            mask[0, y0:y1, x0:x1] = 1
            ys, xs = C.turn_grid(mask, k)[0].nonzero(as_tuple=True)
            said = in_user_frame(name, k)
            assert region_of(float(ys.float().mean()), float(xs.float().mean())) == said, (k, name, said)
        side = C.SIDES[k]  # the camera looks from the world's side k: in its own frame, the front
        assert in_user_frame(side, k) == "front" and C.camera_word(side, k) == "front"
    mask = torch.zeros(1, 16, 16)
    mask[0, :, :3] = 1  # the world's left strip
    assert C.turn_grid(mask, 3)[0, 13:, :].all() and in_user_frame("left", 3) == "front"  # a camera at the world's left: that strip is nearest it
