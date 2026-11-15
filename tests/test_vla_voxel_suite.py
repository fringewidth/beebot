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
    for family, least in (("logic", 0.6), ("construction", 0.5)):  # construction under the strict judges: a free-standing tower, a full trench (out of reach of these swarms), a bridge in the plateaus' top layer
        world, metas, tasks = S.sample_tasks(rng, 6, "train", family)
        results, _ = S.run(world, metas, tasks, S.Expert(tasks, seed=0), seed=0)
        assert np.mean([s for s, _ in results]) > least
        assert not any(u for _, u in results)


# Strict judges (runs/handoff/judges-v2.md): a pass is what a stranger watching the grey video would call done.

STATIC = {"walls": np.zeros((16, 16, 16), bool), "cheese_color": [-1, -1, -1], "gravity": True, "meta": None}


def snap(occupied):
    return {"mass": occupied.astype(np.float32), "cheese_pos": np.zeros((3, 3), int), "cheese_alive": np.zeros(3, bool)}


def test_a_tower_is_a_free_standing_column_of_the_height_asked():
    tower = S.Task("construction", "", {"kind": "tower", "height": 5, "forbid": set()})
    column = np.zeros((16, 16, 16), bool)
    column[:5, 3, 3] = True
    column[0, 3, 4] = True  # a robot on the floor next to it is fine
    assert S.score(tower, STATIC, snap(column), snap(column), [])[0] == 1.0
    hanging = column.copy()
    hanging[3, 4, 3] = True  # a robot stuck on its side
    assert S.score(tower, STATIC, snap(hanging), snap(hanging), [])[0] < 0.5
    pile = np.zeros((16, 16, 16), bool)
    pile[:5, 3, 3] = pile[:4, 3, 4] = pile[:3, 4, 3] = pile[:2, 4, 4] = True  # a 5-high pile, a staircase
    assert S.score(tower, STATIC, snap(pile), snap(pile), [])[0] < 0.5
    taller = np.zeros((16, 16, 16), bool)
    taller[:6, 3, 3] = True
    assert S.score(tower, STATIC, snap(taller), snap(taller), [])[0] < 0.5
    square = np.zeros((16, 16, 16), bool)
    square[:5, 3:5, 3:5] = True  # a 2 x 2 column
    assert S.score(tower, STATIC, snap(square), snap(square), [])[0] == 1.0


def test_a_shape_passes_only_complete_and_clean():
    mask = S.shape_mask("wall", half=True)
    task = S.Task("formation", "", {"kind": "mask", "mask": mask, "forbid": set()})
    assert S.score(task, STATIC, snap(mask), snap(mask), [])[0] == 1.0
    partial = mask.copy()
    partial[tuple(np.argwhere(mask)[0])] = False
    assert S.score(task, STATIC, snap(partial), snap(partial), [])[0] < S.SUCCESS["formation"]
    stray = mask.copy()
    z, y, x = np.argwhere(mask)[0]
    stray[0, y + 1, x] = True  # a robot touching the wall
    assert S.score(task, STATIC, snap(stray), snap(stray), [])[0] < S.SUCCESS["formation"]
    far = mask.copy()
    far[0, 0, 0] = True  # a robot well away from it is fine
    assert S.score(task, STATIC, snap(far), snap(far), [])[0] == 1.0


def test_parallel_jobs_need_two_separate_groups():
    task = S.Task("parallel", "", {"kind": "parallel", "jobs": [("region", "back left"), ("tower", 3)], "forbid": set()})
    one = np.zeros((16, 16, 16), bool)
    one[:3, 2, 2] = True  # a tower inside the back left region, no second group
    one[0, 1, 1] = one[0, 1, 3] = True
    assert S.score(task, STATIC, snap(one), snap(one), [])[0] < 0.75
    two = np.zeros((16, 16, 16), bool)
    two[0, 1:3, 1:3] = True  # a group at the back left
    two[:3, 10, 10] = True  # a tower in the front right, far from it
    assert S.score(task, STATIC, snap(two), snap(two), [])[0] == 1.0
    near = two.copy()
    near[0, 1:3, 1:3] = False
    near[0, 4, 4] = near[0, 5, 5] = True  # a group in the back left box grown by a cell, its centre outside the box
    assert S.score(task, STATIC, snap(near), snap(near), [])[0] < 0.75


def test_groups_are_visibly_apart_and_spread_means_nobody_touches():
    occupied = np.zeros((16, 16, 16), bool)
    occupied[0, 2, 2:5] = True
    occupied[0, 2, 6:9] = True  # one empty cell between: one group to a viewer
    assert len(S.visible_groups(occupied)) == 1
    occupied[0, 2, 6:9] = False
    occupied[0, 2, 7:10] = True  # two empty cells between
    assert len(S.visible_groups(occupied)) == 2
    diagonal = np.zeros((16, 16, 16), bool)
    diagonal[0, 2, 2] = diagonal[0, 3, 3] = True  # corner to corner: they touch
    assert S.apart(diagonal).sum() == 0
    diagonal[0, 3, 3] = False
    diagonal[0, 4, 4] = True
    assert S.apart(diagonal).sum() == 2


def test_plain_words_name_things_as_a_stranger_sees_them():
    assert S.plain_words("eat the red cheese if there is one") == "eat the red food block if there is one"
    assert S.plain_words("eat all the cheese") == "eat all the food"
    assert S.plain_words("push the red and blue cheeses to the left") == "push the red and blue food blocks to the left"
    assert S.plain_words("line up along the green stripe") == "line up along the green line on the floor"
    assert S.plain_words("go to the red zone") == "go to the red patch of floor"
    assert S.plain_words("build a tower 4 robots tall") == "build a tower 4 robots tall"
