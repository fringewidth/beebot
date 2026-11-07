"""Checks for the language-conditioned world: mass moves as described, cheeses push and get eaten by the rules."""

import numpy as np
import pytest
import torch

from beebot.vla.world import (
    CAPACITY,
    DIRS,
    FRICTION,
    MAX_BOXES,
    MAX_CHEESES,
    SIZE,
    cheese_masks,
    largest_open_region,
    observe,
    ring_masks,
    sample_world,
    STAY,
    robot_directions,
    step,
    world_from_arrays as make_world,
)


def uniform_field(vx, vy, size=SIZE):
    field = torch.zeros(1, 2, size, size)
    field[:, 0] = vx
    field[:, 1] = vy
    return field


def test_robots_take_the_nearest_of_8_directions():
    def code(vx, vy):
        return robot_directions(torch.tensor([vx, vy], dtype=torch.float32).view(1, 2, 1, 1)).item()

    assert code(1.0, 1.0) == 1  # 45 degrees, toward (dy, dx) = (1, 1)
    assert code(np.cos(0.3), np.sin(0.3)) == 0  # about 17 degrees rounds to straight right
    assert code(0.0, -1.0) == 6  # up
    assert code(0.3, 0.0) == STAY  # too short to move


def test_robot_moves_one_square():
    mass = np.zeros((SIZE, SIZE))
    mass[5, 5] = 1.0
    world, _ = step(make_world(mass), uniform_field(1.0, 0.0))
    assert world.mass[0, 5, 6].item() == 1.0
    assert world.mass.sum().item() == 1.0


def test_edges_and_walls_keep_robots_in_place():
    mass = np.zeros((SIZE, SIZE))
    mass[5, SIZE - 1] = 1.0
    mass[8, 3] = 1.0
    walls = np.zeros((SIZE, SIZE), dtype=bool)
    walls[8, 4] = True
    world, _ = step(make_world(mass, walls), uniform_field(1.0, 0.0))
    assert world.mass[0, 5, SIZE - 1].item() == 1.0
    assert world.mass[0, 8, 3].item() == 1.0


def test_two_robots_wanting_one_square_one_wins():
    mass = np.zeros((SIZE, SIZE))
    mass[5, 4] = mass[5, 6] = 1.0
    field = torch.zeros(1, 2, SIZE, SIZE)
    field[0, 0, 5, 4] = 1.0
    field[0, 0, 5, 6] = -1.0
    world, _ = step(make_world(mass), field)
    assert world.mass[0, 5, 5].item() == 1.0
    assert world.mass.sum().item() == 2.0
    assert world.mass[0, 5, 4].item() + world.mass[0, 5, 6].item() == 1.0


def test_a_line_advances_together_and_swaps_do_not_happen():
    mass = np.zeros((SIZE, SIZE))
    mass[5, 2:8] = 1.0
    world, _ = step(make_world(mass), uniform_field(1.0, 0.0))
    assert world.mass[0, 5].nonzero().flatten().tolist() == list(range(3, 9))
    mass = np.zeros((SIZE, SIZE))
    mass[5, 4] = mass[5, 5] = 1.0
    field = torch.zeros(1, 2, SIZE, SIZE)
    field[0, 0, 5, 4] = 1.0
    field[0, 0, 5, 5] = -1.0
    world, _ = step(make_world(mass), field)
    assert world.mass[0, 5, 4].item() == 1.0 and world.mass[0, 5, 5].item() == 1.0


def test_random_fields_keep_one_robot_per_square():
    rng = np.random.default_rng(0)
    world = sample_world(rng, 16)
    start = world.mass.sum(dim=(-2, -1))
    torch.manual_seed(0)
    for _ in range(40):
        world, _ = step(world, torch.randn(16, 2, SIZE, SIZE))
    assert torch.equal(world.mass.sum(dim=(-2, -1)), start)
    assert ((world.mass == 0) | (world.mass == 1)).all()
    assert world.mass[world.walls].abs().max().item() == 0
    masks = cheese_masks(world).any(dim=1)
    assert world.mass[masks].abs().max().item() == 0


def pusher(rows, cols, amount):
    mass = np.zeros((SIZE, SIZE))
    for r in rows:
        for c in cols:
            mass[r, c] = amount
    return mass


def test_push_moves_cheese_and_shoves_a_robot_ahead():
    mass = pusher([6, 7], [4], 1.0)  # two robots pressing on the cheese's left side
    mass[6, 7] = 1.0  # in the way, will be shoved to column 8
    world = make_world(mass, cheeses=[(6, 5, 0)])
    field = torch.zeros(1, 2, SIZE, SIZE)
    field[0, 0, :, 4] = 1.0  # only the pushers move
    after, info = step(world, field)
    assert info.pushed[0, 0]
    assert after.cheese_pos[0, 0].tolist() == [6, 6]
    assert after.mass[0, 6, 8].item() == 1.0
    assert after.mass.sum().item() == world.mass.sum().item()


def test_weak_push_does_not_move():
    mass = pusher([6], [4], FRICTION * 0.5)
    world = make_world(mass, cheeses=[(6, 5, 0)])
    after, info = step(world, uniform_field(1.0, 0.0))
    assert not info.pushed[0, 0]
    assert after.cheese_pos[0, 0].tolist() == [6, 5]


def test_opposing_pushes_cancel():
    mass = pusher([6, 7], [4], 1.0) + pusher([6, 7], [7], 1.0)
    field = torch.zeros(1, 2, SIZE, SIZE)
    field[0, 0, :, 4] = 1.0
    field[0, 0, :, 7] = -1.0
    after, info = step(make_world(mass, cheeses=[(6, 5, 0)]), field)
    assert not info.pushed[0, 0]


def test_push_blocked_by_wall():
    walls = np.zeros((SIZE, SIZE), dtype=bool)
    walls[6:8, 7] = True
    after, info = step(make_world(pusher([6, 7], [4], 1.0), walls, cheeses=[(6, 5, 0)]), uniform_field(1.0, 0.0))
    assert not info.pushed[0, 0]


def test_shove_blocked_when_the_robot_has_nowhere_to_go():
    field = torch.zeros(1, 2, SIZE, SIZE)
    field[0, 0, :, 4] = 1.0
    mass = pusher([6, 7], [4], 1.0)
    mass[6, 7] = 1.0
    walls = np.zeros((SIZE, SIZE), dtype=bool)
    walls[6, 8] = True
    after, info = step(make_world(mass, walls, cheeses=[(6, 5, 0)]), field)
    assert not info.pushed[0, 0]
    mass[6, 8] = 1.0  # another robot behind it instead of a wall
    after, info = step(make_world(mass, cheeses=[(6, 5, 0)]), field)
    assert not info.pushed[0, 0]


def ring_cover(y, x, value):
    mass = np.zeros((SIZE, SIZE))
    for r in range(y - 1, y + 3):
        for c in range(x - 1, x + 3):
            if 0 <= r < SIZE and 0 <= c < SIZE and not (y <= r < y + 2 and x <= c < x + 2):
                mass[r, c] = value
    return mass


def test_surrounded_cheese_is_eaten():
    world = make_world(ring_cover(6, 6, 1.0), cheeses=[(6, 6, 0)])
    after, info = step(world, uniform_field(0.0, 0.0))
    assert info.eaten[0, 0]
    assert not after.cheese_alive[0, 0]
    assert after.mass.sum().item() == pytest.approx(world.mass.sum().item())


def test_partial_ring_is_not_eaten():
    mass = ring_cover(6, 6, 1.0)
    mass[5, 5:9] = 0  # clear the top row of the ring: 8 of 12 covered is under 75%
    after, info = step(make_world(mass, cheeses=[(6, 6, 0)]), uniform_field(0.0, 0.0))
    assert not info.eaten[0, 0]


def test_cheese_in_a_corner_cannot_be_eaten():
    mass = ring_cover(0, 0, CAPACITY)
    after, info = step(make_world(mass, cheeses=[(0, 0, 0)]), uniform_field(0.0, 0.0))
    assert not info.eaten[0, 0]


def test_ring_excludes_walls_and_other_cheese():
    walls = np.zeros((SIZE, SIZE), dtype=bool)
    walls[5, 6] = True
    world = make_world(np.zeros((SIZE, SIZE)), walls, cheeses=[(6, 6, 0)])
    rings = ring_masks(world)
    assert rings[0, 0].sum().item() == 11
    assert not rings[0, 0, 5, 6]


def test_sampled_scenes_are_valid():
    rng = np.random.default_rng(1)
    world = sample_world(rng, 64)
    open_region = [largest_open_region(w.numpy()) for w in world.walls]
    for i in range(64):
        assert (open_region[i] == ~world.walls[i].numpy()).all()  # every open square is reachable
        assert world.cheese_alive[i].any()
        masks = cheese_masks(world.index(i))[0]
        assert not (masks & world.walls[i]).any()
        assert world.mass[i][world.walls[i]].sum() == 0
        assert ((world.mass[i] == 0) | (world.mass[i] == 1)).all()  # one robot per square
    cluttered = ((world.cheese_alive.sum(1) >= 2) & ((world.box_color >= 0).sum(1) == 2)).float().mean()
    assert 0.3 < cluttered.item() < 0.9
    assert observe(world).shape == (64, 10, SIZE, SIZE)


def test_dirs_are_ordered_by_angle():
    angles = [np.arctan2(dy, dx) % (2 * np.pi) for dy, dx in DIRS]
    assert angles == sorted(angles)
