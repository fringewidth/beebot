"""The free-cube rule (world.pivot_moves): a cube pivots on a support and moves only when no other
cube touches it face to face; the world refuses anything else and the expert chooses within it."""

import numpy as np
import torch

from beebot.vla.voxel import controller
from beebot.vla.voxel import suite as S
from beebot.vla.voxel.world import CUBE_MOVES, DIRS, STAY, failure_rates, refuse, rule_moves, step, world_from_arrays


def code(dz, dy, dx):
    return DIRS.index((dz, dy, dx))


def scene(cells, walls=(), cheeses=(), gravity=True):
    mass = np.zeros((16, 16, 16))
    for c in cells:
        mass[c] = 1
    solid = np.zeros((16, 16, 16), bool)
    for c in walls:
        solid[c] = True
    return world_from_arrays(mass, solid, cheeses, gravity)


def moves_of(world, cell, rule="free_cube"):
    """The set of (dz, dy, dx) the robot at cell may make."""
    robots = [tuple(c) for c in (world.mass[0] > 0.5).nonzero().tolist()]
    row = rule_moves(world, rule)[robots.index(tuple(cell))]
    assert bool(row[STAY])
    return {DIRS[c] for c in row[:STAY].nonzero().squeeze(1).tolist()}


ROLLS = {(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0)}


def test_a_lone_cube_rolls_on_the_floor_and_nothing_else():
    world = scene([(0, 5, 5)])
    assert moves_of(world, (0, 5, 5)) == ROLLS
    assert moves_of(scene([(0, 0, 0)]), (0, 0, 0)) == {(0, 0, 1), (0, 1, 0)}  # the grid's edge


def test_a_second_face_neighbour_freezes_a_cube():
    pair = scene([(0, 5, 5), (0, 5, 6)])
    assert moves_of(pair, (0, 5, 5)) == {(1, 0, 1)}  # it cannot roll away, only climb its neighbour
    row = scene([(0, 5, 4), (0, 5, 5), (0, 5, 6)])
    assert moves_of(row, (0, 5, 5)) == set()  # a cube on either side
    assert moves_of(row, (0, 5, 4)) == {(1, 0, 1)}
    corner = scene([(0, 5, 5), (0, 5, 6), (0, 6, 5)])
    assert moves_of(corner, (0, 5, 5)) == set()
    stack = scene([(0, 5, 5), (1, 5, 5)])
    assert moves_of(stack, (0, 5, 5)) == set()  # loaded from above
    assert moves_of(stack, (1, 5, 5)) == {(-1, 0, 1), (-1, 0, -1), (-1, 1, 0), (-1, -1, 0)}  # the top one steps down any side
    assert moves_of(row, (0, 5, 5), "switch_cube") == {(0, 1, 0), (0, -1, 0)}  # with bonds released it rolls out sideways
    assert moves_of(pair, (0, 5, 5), "switch_cube") == {(1, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0)}


def test_the_measured_route_up_a_wall_of_cubes():
    wall = [(0, 5, 6), (1, 5, 6)]
    foot = scene([(0, 5, 5)] + wall)
    assert moves_of(foot, (0, 5, 5)) == {(1, 0, 0)}  # concave up; the top is two up, out of a half turn's reach
    hanging = scene([(1, 5, 5)] + wall)
    assert moves_of(hanging, (1, 5, 5)) == {(1, 0, 1), (-1, 0, 0)}  # convex up from hanging, or back down
    top = scene([(2, 5, 6)] + wall)
    assert (-1, 0, -1) in moves_of(top, (2, 5, 6))  # convex down to hang again
    crowd = scene([(1, 5, 5), (1, 6, 5)] + wall)
    assert moves_of(crowd, (1, 5, 5)) == set()  # hanging beside another cube: held
    onto = scene([(1, 5, 5), (0, 5, 5)] + wall)
    assert moves_of(onto, (1, 5, 5)) == set()  # no cube climbs from on top of another
    assert moves_of(onto, (0, 5, 5)) == set()


def test_terrain_and_cheese_do_not_bond_and_a_cheese_is_no_hinge():
    walls = [(0, 5, 6), (1, 5, 6)]
    beside = scene([(0, 5, 5)], walls)
    assert moves_of(beside, (0, 5, 5)) == {(0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0)}  # rolls away or climbs the terrain
    on_top = scene([(2, 5, 6)], walls)
    assert moves_of(on_top, (2, 5, 6)) == {(-1, 0, 1), (-1, 0, -1), (-1, 1, 0), (-1, -1, 0)}
    cheese = scene([(0, 5, 5)], cheeses=[(0, 5, 6, 0)])
    assert moves_of(cheese, (0, 5, 5)) == {(0, 0, -1), (0, 1, 0), (0, -1, 0)}  # no climbing a cheese
    riding = scene([(2, 5, 6)], cheeses=[(0, 5, 6, 0)])
    assert (0, 0, 1) in moves_of(riding, (2, 5, 6)) and (-1, 0, -1) in moves_of(riding, (2, 5, 6))  # it rolls over its top


def test_the_world_refuses_what_the_rule_forbids():
    world = scene([(0, 5, 5), (0, 5, 6), (0, 9, 9)])
    away = code(0, 0, -1)
    codes = torch.full((1, 16, 16, 16), STAY)
    codes[0, 0, 5, 5] = away
    codes[0, 0, 9, 9] = away
    assert refuse(world, codes, "free_cube")[0, 0, 5, 5] == STAY and refuse(world, codes, "free_cube")[0, 0, 9, 9] == away
    after, _ = step(world, codes, rule="free_cube")
    assert after.mass[0, 0, 5, 5] == 1 and after.mass[0, 0, 9, 8] == 1 and after.mass[0, 0, 9, 9] == 0  # the free cube rolled, the other stayed
    loose, _ = step(world, codes)
    assert loose.mass[0, 0, 5, 4] == 1  # without the rule it moves
    released, _ = step(world, codes, rule="switch_cube")
    assert released.mass[0, 0, 5, 4] == 1
    press = scene([(0, 5, 5), (0, 6, 5)], cheeses=[(0, 5, 6, 0)])
    codes = torch.full((1, 16, 16, 16), STAY)
    codes[0, 0, 5, 5] = codes[0, 0, 6, 5] = code(0, 0, 1)
    assert (refuse(press, codes, "free_cube") == codes).all()  # a press is not a pivot
    pushed, info = step(press, codes, rule="free_cube")
    assert bool(info.pushed[0, 0])


def test_recorded_actions_are_the_expert_choice_and_obey_the_rule():
    rng = np.random.default_rng(2)
    world, metas, tasks = S.sample_tasks(rng, 10, "train", robots=(6, 60))
    steps = 24
    for failure in (None, failure_rates(0.5)):
        expert = S.Expert(tasks, moves=CUBE_MOVES, seed=0, rule="free_cube")
        chosen = []

        def actor(w, t):
            codes = expert(w, t)
            chosen.append((w, codes.clone()))
            return codes

        _, arrays = S.run(world, metas, tasks, actor, steps=steps, seed=0, record=True, failure=failure, rule="free_cube")
        n_moves = 0
        for t, (w, codes) in enumerate(chosen):
            assert (arrays["actions"][:, t] == codes).all()  # the label is the choice, moved or not
            assert (arrays["mass"][:, t].float() == w.mass).all()
            assert (refuse(w, codes.long(), "free_cube") == codes).all()  # and the rule allows it there
            n_moves += int((codes != STAY).sum())
        assert n_moves > 50


def test_without_a_rule_the_expert_asks_for_moves_cubes_cannot_make():
    rng = np.random.default_rng(2)
    world, metas, tasks = S.sample_tasks(rng, 10, "train", robots=(40, 115))
    expert = S.Expert(tasks, moves=CUBE_MOVES, seed=0)
    codes = expert(world, 0).long()
    asked = int((codes != STAY).sum())
    refused = int((refuse(world, codes, "free_cube") != codes).sum())
    assert refused > 0.3 * asked
    assert (controller.act(world, torch.zeros(10, 1, 3, dtype=torch.long), torch.ones(10, 1, dtype=torch.bool), jitter=1.0, generator=torch.Generator().manual_seed(0), allowed=rule_moves(world, "free_cube"))[world.mass < 0.5] == STAY).all()
