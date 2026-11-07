"""The generalist task suite (vla/tasks.py): checkers credit the right outcomes, held-out items
stay out of training, runners flee."""

import numpy as np
import torch

from beebot.vla import tasks as T
from beebot.vla.shapes import formation_score, text_mask
from beebot.vla.world import COLOUR_NAMES, cheese_side


def _scene(cheeses=(), boxes=(), mass=None):
    """cheeses: [(row, col, colour name)], boxes: [(row, col, h, w, colour name)]."""
    return {
        "mass": np.zeros((T.SIZE, T.SIZE), np.float32) if mass is None else mass,
        "walls": np.zeros((T.SIZE, T.SIZE), bool),
        "cheeses": [(y, x) for y, x, _ in cheeses],
        "cheese_colours": [COLOUR_NAMES.index(c) for _, _, c in cheeses],
        "boxes": [b[:4] for b in boxes],
        "box_colours": [COLOUR_NAMES.index(b[4]) for b in boxes],
        "runners": [],
    }


def _frame(mass, scene, alive=None, pos=None):
    k = len(scene["cheeses"])
    return {"mass": mass, "cheese_pos": np.array(pos if pos is not None else scene["cheeses"]).reshape(k, 2), "cheese_alive": np.array(alive if alive is not None else [True] * k)}


def test_training_never_draws_held_out_items():
    rng = np.random.default_rng(0)
    for _ in range(300):
        scene, task = T.sample_task(rng, "train")
        assert task.held_out == ""
        assert "pink" not in task.text
        assert all(COLOUR_NAMES[c] != "pink" for c in scene["cheese_colours"] + scene["box_colours"])
        for word in T.HELD_OUT["words"]:
            assert f" {word}" not in task.text
        assert "heart" not in task.text and "triangle" not in task.text and "neither" not in task.text


def test_formation_scores_the_exact_shape_one():
    mask = text_mask("AB", T.SIZE)
    assert formation_score(mask.astype(np.float32), mask) == 1.0
    assert formation_score(np.zeros((T.SIZE, T.SIZE), np.float32), mask) == 0.0


def test_logic_counts_targets_order_and_forbidden():
    scene = _scene(cheeses=[(4, 4, "red"), (4, 20, "blue"), (20, 4, "green")])
    frames = [_frame(np.zeros((T.SIZE, T.SIZE)), scene)] * 2
    except_blue = T.Task("logic", "", {"eat": {"red", "green"}, "any_of": set(), "order": [], "forbid": {"blue"}}, "")
    assert T.score(except_blue, scene, frames, ["red", "green"]) == (1.0, False)
    assert T.score(except_blue, scene, frames, ["red", "blue"]) == (0.5, True)
    ordered = T.Task("logic", "", {"eat": {"red", "blue"}, "any_of": set(), "order": ["red", "blue"], "forbid": set()}, "")
    assert T.score(ordered, scene, frames, ["red", "blue"])[0] == 1.0
    assert T.score(ordered, scene, frames, ["blue", "red"])[0] == 0.0
    either = T.Task("logic", "", {"eat": set(), "any_of": {"red", "blue"}, "order": [], "forbid": {"green"}}, "")
    assert T.score(either, scene, frames, ["blue"]) == (1.0, False)


def test_parallel_needs_two_groups_in_the_two_places():
    scene = _scene(boxes=[(2, 2, 8, 8, "blue")])
    mass = np.zeros((T.SIZE, T.SIZE), np.float32)
    mass[3:9, 3:9] = 1  # a group inside the blue box (top left)
    mass[23:29, 23:29] = 1  # a group at the bottom right
    task = T.Task("parallel", "", {"jobs": [("box", "blue"), ("region", "bottom right")], "rects": {"blue": (2, 2, 8, 8)}, "forbid": set()}, "")
    good, _ = T.score(task, scene, [_frame(mass, scene)] * 2, [])
    one = np.zeros_like(mass)
    one[3:9, 3:9] = 1
    one[9:12, 3:9] = 1
    bad, _ = T.score(task, scene, [_frame(one, scene)] * 2, [])
    assert good > 0.9 and bad < 0.5


def test_herd_credits_a_cheese_in_the_box_and_charges_eating_it():
    side = cheese_side(T.SIZE)
    scene = _scene(cheeses=[(20, 20, "red")], boxes=[(2, 2, 8, 8, "blue")])
    task = T.Task("herd", "", {"into": (["red"], (2, 2, 8, 8)), "forbid": {"red"}}, "")
    start = _frame(np.zeros((T.SIZE, T.SIZE)), scene)
    inside = _frame(np.zeros((T.SIZE, T.SIZE)), scene, pos=[(3, 3)])
    assert T.score(task, scene, [start, inside], [])[0] == 1.0
    eaten = _frame(np.zeros((T.SIZE, T.SIZE)), scene, alive=[False])
    assert T.score(task, scene, [start, eaten], ["red"]) == (0.0, True)
    assert side == 4


def test_runner_moves_away_from_robots():
    mass = np.zeros((T.SIZE, T.SIZE), np.float32)
    mass[14:18, 8:12] = 1  # robots to the left of the cheese
    scene = _scene(cheeses=[(14, 14, "red")], mass=mass)
    scene["runners"] = [0]
    world, runners = T.stack([scene])
    pos = T.flee(world, runners)
    assert pos[0, 0, 1].item() == 15 and pos[0, 0, 0].item() == 14  # one square right, away from them
    alone = _scene(cheeses=[(14, 14, "red")])
    alone["runners"] = [0]
    world, runners = T.stack([alone])
    assert torch.equal(T.flee(world, runners)[0, 0], torch.tensor([14, 14]))  # senses nobody: stays
