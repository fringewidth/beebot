"""Checks for the fact extractor: it names directions, regions, splits, boxes, pushes and eats correctly."""

import numpy as np

from beebot.vla.facts import direction_name, region_name, window_facts

SIZE = 16


def static(cheese_color=(0, -1, -1), box_rect=((0, 0, 0, 0), (0, 0, 0, 0)), box_color=(-1, -1)):
    return {
        "walls": np.zeros((SIZE, SIZE), dtype=bool),
        "cheese_color": np.array(cheese_color),
        "box_rect": np.array(box_rect),
        "box_color": np.array(box_color),
    }


def snap(mass, cheese_pos=((7, 7), (0, 0), (0, 0)), alive=(True, False, False)):
    return {"mass": mass, "cheese_pos": np.array(cheese_pos), "cheese_alive": np.array(alive)}


def square(rows, cols):
    m = np.zeros((SIZE, SIZE))
    m[rows[0] : rows[1], cols[0] : cols[1]] = 1.0
    return m


def test_names_follow_row_zero_at_top():
    assert direction_name(-3, 0) == "up"
    assert direction_name(2, 2) == "down and right"
    assert direction_name(0, -1) == "left"
    assert region_name(1, 14, SIZE) == "top right"
    assert region_name(8, 8, SIZE) == "center"
    assert region_name(14, 8, SIZE) == "bottom"


def test_move_to_a_corner():
    f = window_facts(static(), snap(square((12, 15), (12, 15))), snap(square((0, 3), (0, 3))))
    assert f.types == {"move"}
    assert "up and left" in f.lines[0] and "top left" in f.lines[0]


def test_split_into_two():
    end = square((0, 3), (0, 3)) + square((13, 16), (13, 16))
    f = window_facts(static(), snap(square((6, 9), (1, 7))), snap(end))
    assert "split" in f.types
    assert "into 2" in " ".join(f.lines)


def test_box_entry():
    s = static(box_rect=((11, 11, 5, 5), (0, 0, 0, 0)), box_color=(2, -1))
    f = window_facts(s, snap(square((0, 3), (0, 3))), snap(square((12, 15), (12, 15))))
    assert "box" in f.types
    assert any("blue box" in line for line in f.lines)


def test_push_and_eat():
    mass = square((0, 2), (0, 4))
    f = window_facts(static(), snap(mass), snap(mass, cheese_pos=((7, 11), (0, 0), (0, 0))))
    assert "push" in f.types
    assert any("red cheese right by 4" in line for line in f.lines)
    f = window_facts(static(), snap(mass), snap(mass, alive=(False, False, False)))
    assert f.types == {"eat"}


def test_nothing_happened_is_not_eventful():
    mass = square((2, 5), (2, 5))
    assert not window_facts(static(), snap(mass), snap(mass)).eventful
