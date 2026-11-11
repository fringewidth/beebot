"""Move outcomes measured in MuJoCo, in the voxel world's step (world.OutcomeModel): each pivot
reaches, stays, or ends as a MuJoCo trial of its class ended, replayed in the move's frame; a cube
left off the lattice is stuck."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from beebot.vla.voxel import suite as S
from beebot.vla.voxel.world import DIRS, N_CLASSES, OUTCOME_MODELS, PRIMITIVES, STAY, OutcomeModel, class_name, concat, pivot_class, pivot_features, rule_moves, rule_table, step, world_from_arrays


def code(dz, dy, dx):
    return DIRS.index((dz, dy, dx))


def scene(cells, walls=(), cheeses=()):
    mass = np.zeros((16, 16, 16))
    for c in cells:
        mass[c] = 1
    solid = np.zeros((16, 16, 16), bool)
    for c in walls:
        solid[c] = True
    return world_from_arrays(mass, solid, cheeses, gravity=True)


def table(**classes):
    """A model from {key: (reached, stayed, [aftermath, ...])}."""
    return OutcomeModel({"name": "test", "classes": {key.replace("__", "/"): {"reached": r, "stayed": s, "harm": h} for key, (r, s, h) in classes.items()}})


def one_move(world, cell, move):
    codes = torch.full(world.mass.shape, STAY)
    codes[0][cell] = code(*move)
    return codes


def test_classes_name_the_pivot_and_what_holds_it():
    world = scene([(0, 5, 5), (0, 5, 6), (0, 6, 4)])
    _, side = rule_table(world, "switch_cube", sides=True)
    here = torch.tensor([[0, 5, 5]])
    c = code(0, 0, -1)  # rolls away from its face neighbour, past the cube touching it by an edge
    features = pivot_features(world, torch.tensor([0]), here, torch.tensor([c]), side[0, c, 0, 5, 5].view(1))
    index = int(pivot_class(features)[0])
    assert class_name(index) == "floor_roll/bonded/hinged/rubs"
    lone = scene([(0, 5, 5), (0, 5, 6)])
    _, side = rule_table(lone, "switch_cube", sides=True)
    c = code(1, 0, 1)
    features = pivot_features(lone, torch.tensor([0]), here, torch.tensor([c]), side[0, c, 0, 5, 5].view(1))
    assert class_name(int(pivot_class(features)[0])) == "convex_up/lone"
    assert N_CLASSES == len(PRIMITIVES) * 16


def test_without_a_model_step_is_unchanged():
    world = scene([(0, 5, 5), (0, 9, 9)])
    codes = one_move(world, (0, 9, 9), (0, 1, 0))
    plain, _ = step(world, codes, rule="switch_cube")
    assert plain.stuck is None and plain.mass[0, 0, 10, 9] == 1


def test_outcome_shares_follow_the_table():
    model = table(floor_roll=(70, 30, []))
    world = concat([scene([(0, 5, 5)])] * 400)
    codes = torch.full(world.mass.shape, STAY)
    codes[:, 0, 5, 5] = code(0, 0, 1)
    after, _ = step(world, codes, rule="switch_cube", outcomes=model, noise=torch.rand(world.mass.shape, generator=torch.Generator().manual_seed(0)))
    reached = float(after.mass[:, 0, 5, 6].mean())
    assert abs(reached - 0.7) < 0.07
    assert (after.mass.sum(dim=(1, 2, 3)) == 1).all()


def test_a_mover_left_off_the_lattice_is_stuck_for_good():
    model = table(floor_roll=(0, 0, [[[[0, 0, 0], [1, 0, 0], True]]]))  # ends one step along h, off the lattice
    world = scene([(0, 5, 5)])
    after, _ = step(world, one_move(world, (0, 5, 5), (0, 0, 1)), rule="switch_cube", outcomes=model)
    assert after.mass[0, 0, 5, 6] == 1 and bool(after.stuck[0, 0, 5, 6])
    assert bool(rule_moves(after, "switch_cube")[0, STAY]) and not rule_moves(after, "switch_cube")[0, :STAY].any()
    again, _ = step(after, one_move(after, (0, 5, 6), (0, 1, 0)), rule="switch_cube", outcomes=table(floor_roll=(100, 0, [])))
    assert again.mass[0, 0, 5, 6] == 1 and bool(again.stuck[0, 0, 5, 6])


def test_a_knock_is_replayed_in_the_moves_frame():
    # the mover stays (aligned); its support (one cell along s) is shoved a cell further along s, off the lattice
    knock = [[[0, 0, 0], [0, 0, 0], False], [[0, 1, 0], [0, 2, 0], True]]
    model = table(convex_up__lone=(0, 0, [knock]))
    for move, support, shoved in (((1, 0, 1), (0, 5, 6), (0, 5, 7)), ((1, -1, 0), (0, 4, 5), (0, 3, 5))):
        world = scene([(0, 5, 5), support])
        after, _ = step(world, one_move(world, (0, 5, 5), move), rule="switch_cube", outcomes=model)
        assert after.mass[0, 0, 5, 5] == 1 and after.mass[0][support] == 0 and after.mass[0][shoved] == 1
        assert bool(after.stuck[0][shoved]) and not bool(after.stuck[0, 0, 5, 5])


def test_an_aftermath_into_a_taken_cell_leaves_the_cube_where_it_was():
    model = table(floor_roll=(0, 0, [[[[0, 0, 0], [2, 0, 0], False]]]))  # strays two cells along h
    world = scene([(0, 5, 5), (0, 5, 7)])
    after, _ = step(world, one_move(world, (0, 5, 5), (0, 0, 1)), rule="switch_cube", outcomes=model)
    assert after.mass[0, 0, 5, 5] == 1 and after.mass[0, 0, 5, 7] == 1 and after.mass.sum() == 2


def test_stuck_cubes_fall_with_their_column_and_block_a_cheese():
    model = table(convex_down=(0, 0, [[[[0, 0, 0], [0, 0, 0], True]]]))  # off where it stood
    tower = scene([(0, 5, 5), (1, 5, 5)])
    after, _ = step(tower, one_move(tower, (1, 5, 5), (-1, 0, 1)), rule="switch_cube", outcomes=model)
    assert bool(after.stuck[0, 1, 5, 5])
    gone = after.mass.clone()
    gone[0, 0, 5, 5] = 0
    fallen, _ = step(replace(after, mass=gone), torch.full(gone.shape, STAY), rule="switch_cube")
    assert fallen.mass[0, 0, 5, 5] == 1 and bool(fallen.stuck[0, 0, 5, 5]) and not fallen.stuck[0, 1].any()
    for blocking in (False, True):
        pressed = scene([(0, 5, 4), (0, 6, 4), (0, 5, 7)], cheeses=[(0, 5, 5, 0)])  # two press a cheese toward a cube in its way
        pressed.stuck = torch.zeros(pressed.mass.shape, dtype=torch.bool)
        pressed.stuck[0, 0, 5, 7] = blocking
        codes = torch.full(pressed.mass.shape, STAY)
        codes[0, 0, 5, 4] = codes[0, 0, 6, 4] = code(0, 0, 1)
        pushed, info = step(pressed, codes, rule="switch_cube")
        assert bool(info.pushed[0, 0]) != blocking  # a robot is shoved along; a stuck one stops the cheese
        assert pushed.mass[0, 0, 5, 7 if blocking else 8] == 1


@pytest.mark.parametrize("name", OUTCOME_MODELS)
def test_measured_tables_load_and_cover_every_class(name):
    model = OutcomeModel.load(name)
    assert model.name == name
    floor = [i for i in range(N_CLASSES) if class_name(i) == "floor_roll"][0]
    assert model.source[floor] is not None
    reach, stay, harm = model.probabilities(floor)
    assert abs(reach + stay + harm - 1) < 1e-6
    assert ((model.reach + model.stay) <= 1 + 1e-6).all()


def test_a_store_drawn_with_a_model_records_it(tmp_path):
    from beebot.vla.voxel import demos as D
    from beebot.vla.voxel.play_store import load

    n = D.write(tmp_path, 2, seed=1, chunk=2, steps=12, moves="cube", robots=S.ROBOTS, compact=True, counts="log", rule="switch_cube", outcomes="mujoco_release")
    store = load(tmp_path)
    assert store["config"]["outcomes"] == "mujoco_release" and store["config"]["failure"] is None
    assert len(store["stuck"]) == n
    for e, cells in enumerate(store["stuck"]):
        for t, z, y, x in cells:
            assert store["mass"][e, t][z, y, x] == 1
    with pytest.raises(ValueError):
        D.write(tmp_path / "both", 1, failure=torch.zeros(27), rule="switch_cube", outcomes="mujoco_release")


def test_run_with_a_model_records_stuck_robots_and_refuses_failure_rates():
    rng = np.random.default_rng(3)
    world, metas, tasks = S.sample_tasks(rng, 4, "train", robots=(10, 30))
    world = replace(world, gravity=torch.ones_like(world.gravity))
    model = table(floor_roll=(50, 0, [[[[0, 0, 0], [0, 0, 0], True]]]))
    expert = S.Expert(tasks, seed=0, rule="switch_cube")
    _, arrays = S.run(world, metas, tasks, expert, steps=6, seed=0, record=True, rule="switch_cube", outcomes=model)
    assert arrays["stuck"].shape == arrays["mass"].shape
    assert (arrays["stuck"] <= (arrays["mass"] > 0)).all()
    with pytest.raises(ValueError):
        S.run(world, metas, tasks, expert, steps=1, failure=torch.zeros(27), rule="switch_cube", outcomes=model)


def test_a_measured_knock_fits_and_replays_as_a_knock_in_any_direction():
    from beebot.hardware import cube_outcomes as C

    # MuJoCo: convex up onto a lone cube shoved it a cell further along s, off the lattice; the mover stayed
    row = {"cells": [[0, 5, 5], [0, 5, 6]], "after": [[0, 5, 5], [0, 5, 7]], "off_after": [1], "cube": 0, "start": [0, 5, 5], "target": [1, 5, 6], "h": [1, 0, 0], "s": [0, 0, 1], "key": "convex_up/lone", "detail": "knock", "movers": 1}
    outcome, aftermath = C.lattice_end(row)
    assert outcome == "knock" and aftermath == [[[0, 0, 0], [0, 0, 0], False], [[0, 1, 0], [0, 2, 0], True]]
    model = OutcomeModel(C.fit([row], "one", "release"))
    turned = {**row, "cells": [[0, 5, 5], [0, 4, 5]], "target": [1, 4, 5]}  # the same move toward -y
    for r in (row, turned):
        assert C.grid_outcomes([r], model, runs=4)[0, C.LATTICE.index("knock")] == 1.0
    assert C.grid_outcomes([row], failure=torch.zeros(27), runs=4)[0, C.LATTICE.index("reached")] == 1.0
