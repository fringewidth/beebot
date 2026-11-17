import numpy as np
import pytest

from beebot.vla.voxel import goal_gen_complex as C
from beebot.vla.voxel import goal_library as G
from beebot.vla.voxel import reverse


@pytest.fixture(scope="module")
def drawn():
    return {name: C.sample(12, seed=3, names=[name]) for name in C.GENERATORS}


@pytest.mark.parametrize("name", list(C.GENERATORS))
def test_every_state_is_a_valid_goal(drawn, name):
    goals = drawn[name]
    assert len(goals) == 12
    for g in goals:
        assert g["generator"] == name and g["kind"] in ("scene", "edit", "retarget", "trick")
        states = [g["cells"]] + ([g["start_cells"]] if g["start_cells"] is not None else [])
        for cells in states:
            assert C.geometry_problem(cells, g["params"]["scale"]) is None
            assert len(set(map(tuple, cells))) == len(cells)
        assert all(C.at_rest(states))
        if g["start_cells"] is not None:
            assert len(g["start_cells"]) == len(g["cells"])
            a, b = set(map(tuple, g["start_cells"])), set(map(tuple, g["cells"]))
            assert set(map(tuple, g["params"]["changed"]["fill"])) == b - a
            assert set(map(tuple, g["params"]["changed"]["empty"])) == a - b
            if g["kind"] != "trick":
                assert a != b
            assert len(g["turns"]) >= 2  # the turn that built A, then the request
        for text in g["turns"] + [g["reply"]]:
            assert C.text_ok(text)
        scale = g["params"]["scale"]
        assert g["params"]["overhang"] <= C.MAX_OVERHANG * scale and g["params"]["span"] <= C.MAX_SPAN * scale
        assert len(g["cells"]) <= C.MAX_ROBOTS


def test_tricks_say_what_they_did(drawn):
    tricks = [g for name in C.GENERATORS if name.startswith("trick") for g in drawn[name]]
    for g in tricks:
        if g["params"].get("behaviour") == "nothing":
            assert g["start_cells"] == g["cells"] and g["reply"]
        if g["params"].get("behaviour") == "closest":
            assert g["start_cells"] is None and g["reply"]


def test_parked_rows_share_a_prefix(drawn):
    for name in ("edit_house", "retarget_part", "edit_towers"):
        for g in drawn[name]:
            park = g["params"]["parked"]
            if not park["side"]:
                continue
            slots = C.park_slots(park["side"], C.rows_for(max(park["a"], park["b"])))
            a, b = set(map(tuple, g["start_cells"])), set(map(tuple, g["cells"]))
            assert set(slots[: park["a"]]) <= a and set(slots[: park["b"]]) <= b
            assert not set(slots[max(park["a"], park["b"]) :]) & (a | b)


def test_words_rules():
    assert C.text_ok("build a tower nine high next to the house")
    for bad in ("build a tower ten high", "use 12 of you", "push the cheese", "fill the cells", "a striped wall", "a dozen towers"):
        assert not C.text_ok(bad)


def test_overhang_and_span():
    assert C.overhang_span(C.b_tower(5)) == (0, 0)
    assert C.overhang_span(C.b_post(4, 3)) == (3, 3)  # an arm of three at the top
    assert C.overhang_span(C.b_arch(2, 3)) == (1, 2)  # a beam on top over a gap of two
    hang = C.b_post(4, 2) | {(2, 2, 0)}  # a cell hanging under the arm's tip: lever of the tip, no more
    assert C.overhang_span(hang)[0] == 2


def test_turns_follow_the_viewer():
    chair = C.Part("chair", {"a": 3, "legs": 1, "back": 2}, 3)  # k = 3: facing right, back on the left
    assert chair.facing == "right"
    back = [c for c in chair.cells if c[0] >= 2]
    assert max(c[2] for c in back) == min(c[2] for c in chair.cells)
    stairs = C.Part("stairs", {"width": 1, "steps": 3}, 0)  # low end toward the viewer, rising away
    top = max(stairs.cells)
    assert top[1] == min(c[1] for c in stairs.cells)
    fig = C.Part("figure", {"left": "up", "right": "out", "legs": 2}, 0)
    left_arm = [c for c in fig.cells if c[2] < -1]
    right_arm = [c for c in fig.cells if c[2] > 1]
    assert max(c[0] for c in left_arm) > max(c[0] for c in right_arm)  # the raised arm is on the viewer's left
    assert C.arm_side(-1, 0) == "left" and C.arm_side(1, 0) == "right"


def test_relations_hold():
    rng = np.random.default_rng(5)
    for _ in range(40):
        try:
            xp, yp, rel, gap = C.pair_scene(rng)
        except C.Reject:
            continue
        xy0, xy1, xx0, xx1 = xp.foot
        yy0, yy1, yx0, yx1 = yp.foot
        if rel == "left":
            assert xx1 + gap + 1 == yx0
        elif rel == "right":
            assert yx1 + gap + 1 == xx0
        elif rel == "behind":
            assert xy1 + gap + 1 == yy0
        else:
            assert yy1 + gap + 1 == xy0
        assert C.overlap_ok(xp, yp, rel)


def test_save_and_load(tmp_path, drawn):
    goals = drawn["edit_figure"][:3] + drawn["scene_pair"][:3]
    C.save(goals, "gen_complex_test", root=tmp_path)
    lines, cells, starts = C.load("gen_complex_test", root=tmp_path)
    assert len(lines) == 6
    for g, line, b, a in zip(goals, lines, cells, starts):
        assert sorted(map(tuple, b.tolist())) == [tuple(c) for c in g["cells"]]
        assert (a is None) == (g["start_cells"] is None) == (not line["has_start"])
        assert line["turns"] == g["turns"]


def test_reverse_walks_an_edit_and_a_scene():
    import torch

    torch.set_num_threads(1)
    goals = C.sample(1, seed=0, names=["edit_towers"]) + C.sample(1, seed=0, names=["scene_compare"])
    episodes, _ = reverse.take_apart([C.to_reverse_goal(g) for g in goals], np.random.default_rng(0))
    assert reverse.replay(episodes)["passed"].all()
    assert episodes[0].end == "reached" and episodes[0].distance == 0
    assert episodes[1].end == "natural"
    end = (episodes[0].mass[-1] > 0).nonzero().tolist()
    assert sorted(map(tuple, end)) == [tuple(c) for c in goals[0]["cells"]]


def test_library_grounded_matches():
    g = C.sample(1, seed=1, names=["scene_big"])[0]
    vox = np.zeros((16, 16, 16), dtype=bool)
    arr = np.asarray(g["cells"])
    vox[arr[:, 0], arr[:, 1], arr[:, 2]] = True
    assert G.grounded(vox)


def test_scaled_parts_are_blocks_and_worded_without_counts():
    rng = np.random.default_rng(2)
    small = C.Part("table", {"a": 3, "b": 4, "legs": 2})
    big = small.scaled(2)
    assert len(big.cells) == 8 * len(small.cells)
    for _ in range(20):
        text = C.describe(big, rng).replace("four legs", "")  # the legs are four at any scale
        assert not any(w in text for w in C.NUMBERS[1:]) and "table" in text
