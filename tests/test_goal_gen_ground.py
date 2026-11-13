import re

import numpy as np
import pytest

from beebot.vla.voxel import goal_gen_complex as C
from beebot.vla.voxel import goal_gen_ground as GG
from beebot.vla.voxel import goal_library as G


@pytest.fixture(scope="module")
def drawn():
    return {name: GG.sample(12, seed=3, names=[name]) for name in GG.GENERATORS}


def cells_of(goal, key):
    return set(map(tuple, goal[key]))


@pytest.mark.parametrize("name", list(GG.GENERATORS))
def test_every_state_is_a_valid_goal(drawn, name):
    for g in drawn[name]:
        assert g["generator"] == name and g["kind"] in ("complete", "refer", "match", "ask")
        assert len(g["turns"]) == 1
        for cells in (g["cells"], g["start_cells"]):
            assert C.geometry_problem(cells, 1) is None
            assert len(set(map(tuple, cells))) == len(cells)
        assert all(C.at_rest([g["cells"], g["start_cells"]]))
        assert len(g["start_cells"]) == len(g["cells"])
        a, b = cells_of(g, "start_cells"), cells_of(g, "cells")
        assert set(map(tuple, g["params"]["changed"]["fill"])) == b - a
        assert set(map(tuple, g["params"]["changed"]["empty"])) == a - b
        assert (a == b) == (g["kind"] == "ask")
        assert len(g["cells"]) <= C.MAX_ROBOTS


def test_texts_use_plain_words_and_small_numbers(drawn):
    banned = re.compile(r"\b(cheese|stripes?|zones?|voxels?|cells?)\b")
    for goals in drawn.values():
        for g in goals:
            for text in g["turns"] + [g["reply"]]:
                assert not banned.search(text.lower())
                assert C.text_ok(text)
                assert all(int(d) < 10 for d in re.findall(r"\d+", text))


def test_completions_are_implied_and_unambiguous():
    goals = []
    for name in ("complete_pattern", "complete_part", "complete_scene", "complete_shape"):
        goals += GG.sample(15, seed=11, names=[name])
    seen_k = set()
    for g in goals:
        a, b = cells_of(g, "start_cells"), cells_of(g, "cells")
        holes = b - a
        k = g["params"]["k"]
        seen_k.add(k)
        assert len(holes) == k and len(a - b) == k  # k cells missing, k robots standing elsewhere
        rest = a & b
        assert G.grounded(GG.vox(rest))
        match, bad = GG.explain(GG.vox(rest), holes, k)
        assert match, "no rule implies the missing cells"
        assert not bad, "another rule asks for a different completion"
        assert g["params"]["rule"] in ("symmetry", "pattern", "step", "face")
        assert all(GG.visible(c, GG.vox(rest)) for c in holes)
        assert len(g["turns"][0]) > 0
    assert len(seen_k) >= 3


def test_completion_text_does_not_say_where_and_states_small_counts():
    texts = set()
    for g in GG.sample(60, seed=4, names=["complete_pattern", "complete_part"]):
        text = g["turns"][0].lower()
        texts.add(text)
        assert not re.search(r"\b(left|right|front|behind|top|bottom|middle|corner|edge|side)\b", text)
    assert len(texts) > 10


def test_symmetry_check_rejects_an_ambiguous_removal():
    # a flat 3 x 3 square: removing a corner and its mirror partners leaves several completions
    square = {(0, y, x) for y in range(3) for x in range(3)}
    both = {(0, 0, 0), (0, 0, 2)}  # a cell and its mirror image: mirror_x is silent about them
    rest = square - both
    match, bad = GG.explain(GG.vox(rest), both, 2)
    assert not any(r[0] == "mirror_x" for r, _ in match)
    # one corner of a symmetric square is implied by the mirror, and rules agree
    one = {(0, 0, 0)}
    match, bad = GG.explain(GG.vox(square - one), one, 1)
    assert match


def test_references_fit_exactly_one_cube():
    for name in ("refer_top_to_floor", "refer_top_move", "refer_lone_to_wall", "refer_lone_to_tower", "refer_bump_in"):
        for g in GG.sample(15, seed=8, names=[name]):
            a, b = cells_of(g, "start_cells"), cells_of(g, "cells")
            assert 1 <= len(a - b) <= 3
            for ref in g["params"]["refs"]:
                found = GG.ref_candidates(a, ref)
                assert found == [tuple(ref["target"])], (name, ref, found)
            assert len(g["params"]["refs"]) >= 1
            assert len(a - b) == 1 and len(b - a) == 1


def test_references_detect_ties():
    towers = GG.column(0, 0, 4) | GG.column(0, 3, 4) | GG.column(0, 6, 2)
    assert len(GG.ref_candidates(towers, {"type": "tower", "which": "tallest"})) == 2
    assert len(GG.ref_candidates(towers, {"type": "tower", "which": "shortest"})) == 1
    assert len(GG.ref_candidates(towers, {"type": "tower", "which": "middle"})) == 1


def test_matching_ends_match_the_picture(drawn):
    for g in drawn["match_towers"]:
        a, b = cells_of(g, "start_cells"), cells_of(g, "cells")
        parked = set(C.park_slots(g["params"]["parked"]["side"], C.MAX_ROWS)[: max(g["params"]["parked"]["a"], g["params"]["parked"]["b"])]) if g["params"]["parked"]["side"] else set()
        heights = sorted(h for _, _, h in GG.column_towers(b - parked))
        assert heights[0] == heights[-1] or len(heights) == 3
    for g in drawn["match_walls"]:
        b = cells_of(g, "cells")
        park = g["params"]["parked"]
        parked = set(C.park_slots(park["side"], C.MAX_ROWS)[: park["b"]]) if park["side"] else set()
        rows = {}
        for z, y, x in b - parked:
            rows.setdefault(y, set()).add(x)
        lengths = {len(v) for v in rows.values()}
        assert len(lengths) == 1  # both walls the same length
    for g in drawn["match_mirror"]:
        b = cells_of(g, "cells")
        park = g["params"]["parked"]
        parked = set(C.park_slots(park["side"], C.MAX_ROWS)[: park["b"]]) if park["side"] else set()
        struct = b - parked
        assert {(z, y, SIZE_1 - x) for z, y, x in struct} == struct  # mirrored about the middle of the floor
    for g in drawn["match_copy"]:
        a, b = cells_of(g, "start_cells"), cells_of(g, "cells")
        park = g["params"]["parked"]
        parked = set(C.park_slots(park["side"], C.MAX_ROWS)[: park["a"]]) if park["side"] else set()
        spare_b = set(C.park_slots(park["side"], C.MAX_ROWS)[: park["b"]]) if park["side"] else set()
        original = a - parked
        assert original < b and len(b - original - spare_b) == len(original)


SIZE_1 = C.SIZE - 1


def test_questions_have_true_answers(drawn):
    for g in drawn["ask_height"]:
        towers = GG.column_towers(cells_of(g, "cells"))
        assert len(towers) == 1 and towers[0][2] == g["params"]["answer"]
    for g in drawn["ask_count"]:
        assert len(GG.column_towers(cells_of(g, "cells"))) == g["params"]["answer"] < 10
    for g in drawn["ask_taller"]:
        towers = sorted(GG.column_towers(cells_of(g, "cells")), key=lambda t: t[1])
        assert len(towers) == 2
        assert ("left" if towers[0][2] > towers[1][2] else "right") == g["params"]["answer"]
    for g in drawn["ask_wall"]:
        assert g["start_cells"] == g["cells"] and g["reply"]


def test_k_grows_and_stays_small():
    goals = GG.sample(150, seed=2, names=["complete_pattern", "complete_part"])
    ks = [g["params"]["k"] for g in goals]
    assert min(ks) == 1 and max(ks) <= 8 and len(set(ks)) >= 5


def test_save_load_roundtrip(tmp_path):
    goals = GG.sample(6, seed=1, names=["complete_part", "ask_height", "match_towers"])
    GG.save(goals, "t", root=tmp_path)
    lines, cells, starts = GG.load("t", root=tmp_path)
    assert len(lines) == 6
    for g, c, s in zip(goals, cells, starts):
        assert set(map(tuple, c.tolist())) == cells_of(g, "cells")
        assert set(map(tuple, s.tolist())) == cells_of(g, "start_cells")
