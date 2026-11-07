"""Checks for the attractor controller: it gathers, splits, routes around walls, pushes and surrounds."""

import numpy as np
import pytest
import torch

from beebot.vla.controller import Controller, attractor_field, commands_to_tensors, walking_distance
from beebot.vla.world import SIZE, step
from beebot.vla.world import world_from_arrays as make_world


def run(world, commands, steps):
    eaten_any = torch.zeros_like(world.cheese_alive)
    controller = Controller()
    for _ in range(steps):
        points, valid, straight, share = commands_to_tensors([commands], 1)
        world, info = step(world, controller(world, points, valid, straight, share))
        eaten_any |= info.eaten
    return world, eaten_any


def blob(rows, cols):
    mass = np.zeros((SIZE, SIZE))
    mass[rows[0] : rows[1], cols[0] : cols[1]] = 1.0
    return mass


def test_walking_distance_goes_around_walls():
    walls = torch.zeros(1, 5, 5, dtype=torch.bool)
    walls[0, 0:4, 2] = True
    sources = torch.zeros(1, 5, 5, dtype=torch.bool)
    sources[0, 0, 0] = True
    d = walking_distance(sources, ~walls)
    assert d[0, 0, 4].item() == 8  # down to the wall's end at row 4 and back up
    assert torch.isinf(d[0, 0, 2])


def test_single_attractor_gathers_the_swarm():
    world = make_world(blob((10, 14), (10, 14)))
    world, _ = run(world, [{"row": 2, "col": 2}], 30)
    near = world.mass[0, 0:6, 0:6].sum().item()
    assert near > 0.95 * world.mass.sum().item()


def test_two_attractors_split_the_swarm():
    world = make_world(blob((6, 10), (6, 10)))
    world, _ = run(world, [{"row": 7, "col": 0}, {"row": 8, "col": 15}], 30)
    left = world.mass[0, :, :5].sum().item()
    right = world.mass[0, :, 11:].sum().item()
    assert left > 4 and right > 4


def test_field_is_dense_and_zero_on_walls():
    walls = np.zeros((SIZE, SIZE), dtype=bool)
    walls[3, 3:9] = True
    world = make_world(np.zeros((SIZE, SIZE)), walls)
    points, valid, straight, share = commands_to_tensors([[{"row": 0, "col": 0}]], 1)
    field, _ = attractor_field(world, points, valid, straight, share)
    moving = field.abs().sum(dim=1)[0] > 0
    assert moving[~torch.tensor(walls)].sum().item() == SIZE * SIZE - 6 - 1  # all open squares but the attractor
    assert not moving[torch.tensor(walls)].any()


def test_straight_mode_pushes_a_cheese():
    world = make_world(blob((7, 9), (1, 4)), cheeses=[(7, 6, 0)])
    world, _ = run(world, [{"row": 8, "col": 15, "straight": True}], 20)
    assert world.cheese_pos[0, 0, 1].item() >= 10


def test_routed_attractor_on_a_cheese_surrounds_and_eats_it():
    world = make_world(blob((1, 6), (1, 5)), cheeses=[(9, 9, 0)])
    world, eaten = run(world, [{"row": 9, "col": 9}], 40)
    assert eaten[0, 0]


def test_shares_split_one_blob_into_four_corners():
    # 36 robots: with discrete robots, the auction's boundary squares shift a robot or two between
    # neighbouring groups from step to step, so a small swarm can't be held to a tight split.
    world = make_world(blob((6, 12), (4, 10)))
    corners = [{"row": r, "col": c, "share": 0.25} for r in (0, 15) for c in (0, 15)]
    world, _ = run(world, corners, 50)
    m = world.mass[0]
    total = m.sum().item()
    quarters = [m[:8, :8].sum(), m[:8, 8:].sum(), m[8:, :8].sum(), m[8:, 8:].sum()]
    assert all(0.15 * total < q.item() < 0.35 * total for q in quarters), [round(q.item(), 1) for q in quarters]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_controller_matches_between_cpu_and_mps():
    """MPS once produced distances of 0 or 1 everywhere (inf handling in max_pool2d), silently
    turning the controller's field almost entirely to zero during data generation."""
    import numpy as np
    from beebot.vla.world import sample_world

    world = sample_world(np.random.default_rng(3), 8, size=24)
    commands = [[{"row": 2, "col": 3}, {"row": 20, "col": 18, "share": 0.4}]] * 8
    fields = []
    for device in ("cpu", "mps"):
        points, valid, straight, share = commands_to_tensors(commands, 8, device)
        field, _ = attractor_field(world.to(device), points, valid, straight, share)
        fields.append(field.cpu())
    assert torch.allclose(fields[0], fields[1], atol=1e-5)
    assert (fields[0].abs().sum(1) > 0).float().mean() > 0.8
