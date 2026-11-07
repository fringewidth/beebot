"""Hindsight task labels (vla/relabel.py): only instructions the window's outcome satisfies, and
nothing held out."""

import numpy as np

from beebot.vla import relabel as R
from beebot.vla import tasks as T
from beebot.vla.world import COLOUR_NAMES


def _scene():
    mass = np.zeros((T.SIZE, T.SIZE), np.float32)
    mass[14:18, 14:18] = 1
    return {
        "mass": mass,
        "walls": np.zeros((T.SIZE, T.SIZE), bool),
        "cheeses": [(4, 4), (4, 24), (24, 4)],
        "cheese_colours": [COLOUR_NAMES.index(c) for c in ("red", "blue", "green")],
        "boxes": [(22, 20, 8, 8)],
        "box_colours": [COLOUR_NAMES.index("yellow")],
        "runners": [],
    }


def _frame(scene, alive, pos=None):
    return {"mass": scene["mass"], "cheese_pos": np.array(pos or scene["cheeses"]), "cheese_alive": np.array(alive)}


def _satisfied(scene, frames, eaten):
    rng = np.random.default_rng(0)
    return {(f, text) for f, items in R.candidates(scene, rng).items() for _, text, spec in items if R.satisfied(f, spec, scene, frames, eaten)}


def test_eating_red_and_green_satisfies_except_blue_and_not_eat_blue():
    scene = _scene()
    frames = [_frame(scene, [True] * 3), _frame(scene, [False, True, False])]
    rng = np.random.default_rng(0)
    for family, items in R.candidates(scene, rng).items():
        for _, text, spec in items:
            ok = R.satisfied(family, spec, scene, frames, ["red", "green"])
            if family == "logic" and spec["eat"] == {"red", "green"} and not spec["order"]:
                assert ok, text
            if family == "logic" and "blue" in spec["eat"] | spec["any_of"]:
                assert not ok, text
            if family in ("place", "herd"):  # these forbid every cheese: eating any fails them
                assert not ok, text


def test_nothing_eaten_satisfies_none_and_a_pushed_cheese_satisfies_herding():
    scene = _scene()
    pushed = [(4, 4), (4, 24), (24, 22)]  # green pushed right into the yellow box
    frames = [_frame(scene, [True] * 3), _frame(scene, [True] * 3, pushed)]
    found = _satisfied(scene, frames, [])
    families = {f for f, _ in found}
    assert "herd" in families and "logic" in families
    assert any("green" in text and "yellow" in text for f, text in found if f == "herd")


def test_no_held_out_items_in_candidates():
    scene = _scene()
    rng = np.random.default_rng(1)
    for _ in range(20):
        for family, items in R.candidates(scene, rng).items():
            for _, text, spec in items:
                assert "pink" not in text and "neither" not in text
                if family == "parallel":
                    assert tuple(sorted(k for k, _ in spec["jobs"])) not in T.HELD_OUT["pairs"]
