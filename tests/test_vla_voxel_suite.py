"""The voxel generalist suite (vla/voxel/suite.py): training never draws held-out items, checkers
credit the right outcomes, runners flee, the expert demonstrates training tasks."""

import numpy as np
import torch

from beebot.vla.voxel import suite as S
from beebot.vla.voxel.world import COLOUR_NAMES, world_from_arrays


def test_training_never_draws_held_out_items():
    rng = np.random.default_rng(0)
    world, metas, tasks = S.sample_tasks(rng, 60, "train")
    for i, task in enumerate(tasks):
        assert task.held_out == ""
        assert "pink" not in task.text and "pyramid" not in task.text and "stairs" not in task.text and "neither" not in task.text
        assert all(COLOUR_NAMES[c] != "pink" for c in world.cheese_color[i].tolist() if c >= 0)
        if task.family == "parallel":
            assert tuple(sorted(k for k, _ in task.spec["jobs"])) not in S.HELD_OUT["pairs"]
        assert task.plan or task.spec.get("form") in ("none", "absent")


def test_held_out_splits_force_their_item_and_have_no_plan():
    rng = np.random.default_rng(1)
    for split in ("colour", "letters", "shapes", "logic", "pairs", "construction", "herd_two", "runner_logic"):
        _, _, tasks = S.sample_tasks(rng, 2, split)
        assert all(t.held_out == split and not t.plan for t in tasks)


def test_mask_checker_is_f1_and_tower_checker_is_height():
    mask = S.shape_mask("wall")
    task = S.Task("formation", "", {"kind": "mask", "mask": mask, "forbid": set()})
    static = {"walls": np.zeros((16, 16, 16), bool), "cheese_color": [-1, -1, -1], "gravity": True, "meta": None}
    snap = lambda m: {"mass": m.astype(np.float32), "cheese_pos": np.zeros((3, 3), int), "cheese_alive": np.zeros(3, bool)}  # noqa: E731
    assert S.score(task, static, snap(mask), snap(mask), [])[0] == 1.0
    assert S.score(task, static, snap(mask), snap(np.zeros_like(mask)), [])[0] == 0.0
    column = np.zeros((16, 16, 16), bool)
    column[:5, 3, 3] = True
    tower = S.Task("construction", "", {"kind": "tower", "height": 5, "forbid": set()})
    assert S.score(tower, static, snap(column), snap(column), [])[0] == 1.0


def test_runner_slides_away_from_robots():
    mass = np.zeros((16, 16, 16))
    mass[0, 8, 3:6] = 1  # robots to the left of the cheese
    world = world_from_arrays(mass, cheeses=[(0, 8, 7, 0)], gravity=False)
    pos = S.flee(world, torch.tensor([[True, False, False]]))
    assert pos[0, 0].tolist() == [0, 8, 8]


def test_expert_solves_training_logic_and_construction():
    rng = np.random.default_rng(2)
    for family in ("logic", "construction"):
        world, metas, tasks = S.sample_tasks(rng, 6, "train", family)
        results, _ = S.run(world, metas, tasks, S.Expert(tasks, seed=0), seed=0)
        assert np.mean([s for s, _ in results]) > 0.6
        assert not any(u for _, u in results)
