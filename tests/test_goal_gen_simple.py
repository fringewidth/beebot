import re

import numpy as np

from beebot.vla.voxel import goal_gen_simple as S
from beebot.vla.voxel import goal_library as G


def test_every_generator_gives_valid_goals_for_a_few_seeds():
    for name in S.GENERATORS:
        for seed in range(3):
            rng = np.random.default_rng(seed)
            goal = None
            for _ in range(60):
                goal, why = S.draw(name, rng)
                if goal is not None:
                    break
            assert goal is not None, f"{name} seed {seed}: no accepted draw ({why})"
            cells = goal["cells"]
            assert cells.shape[1] == 3 and G.MIN_CELLS <= len(cells) <= G.MAX_CELLS
            assert (cells.min(axis=0) == 0).all() and (cells.max(axis=0) < 16).all()
            assert S.check(cells, goal["text"]) is None
            assert goal["overhang"] <= S.MAX_OVERHANG


def test_texts_avoid_internal_words_and_large_numbers():
    rng = np.random.default_rng(7)
    for name in S.GENERATORS:
        for _ in range(40):
            goal, _ = S.draw(name, rng)
            if goal is None:
                continue
            text = goal["text"].lower()
            for word in S.FORBIDDEN:
                assert word not in text, (name, text)
            assert not S.LARGE_WORD.search(text), (name, text)
            assert all(int(v) < 10 for v in re.findall(r"\d+", text)), (name, text)


def test_overhang_counts_distance_to_the_support_below():
    bridge = np.array([(0, 0, 0), (0, 0, 1), (0, 0, 2), (1, 0, 0), (1, 0, 2)])
    assert S.overhang(bridge) == 0
    cantilever = np.array([(0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 0, 3)])
    assert S.overhang(cantilever) == 3
    assert S.check(cantilever, "a shelf") is not None


def test_check_rejects_a_floating_shape():
    floating = np.array([(0, 0, 0), (2, 0, 0), (2, 0, 1), (3, 0, 0)])
    assert S.check(floating, "a floating bit") == "grounded"


def test_check_rejects_forbidden_words_and_large_numbers():
    solid = np.array([(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)])
    assert S.check(solid, "a voxel block") == "forbidden word"
    assert S.check(solid, "twelve cubes in a row") == "large number"
    assert S.check(solid, "a row of 12 cubes") == "large number"
    assert S.check(solid, "a flat square, two on each side") is None


def test_sample_is_reproducible_and_returns_n_goals():
    first = S.sample(30, seed=3)
    second = S.sample(30, seed=3)
    assert len(first) == 30
    assert [g["text"] for g in first] == [g["text"] for g in second]
    assert all(g["generator"] in S.GENERATORS for g in first)


def test_label_audit_wordings_are_gone():
    rng = np.random.default_rng(3)
    for name in ("tower_row", "floor_cross", "wall", "frame_standing"):
        for _ in range(60):
            goal, _ = S.draw(name, rng)
            if goal is None:
                continue
            text = goal["text"]
            assert "one one" not in text and "one way" not in text, text
            assert "across, " not in text, text
            assert not re.search(r"(?<!wide )(?:left to right|front to back) and \w+ high", text), text
