import random

import torch

from beebot.vla.caption import parse_reply, strip_tags, tag_check, text_grid
from beebot.vla.caption_set import variants


def tiny_store():
    size = 16
    mass = torch.zeros(1, 2, size, size, dtype=torch.float16)
    mass[0, 0, 0, 0] = 1.0
    walls = torch.zeros(1, size, size, dtype=torch.bool)
    walls[0, 5, 5] = True
    return {
        "mass": mass,
        "walls": walls,
        "cheese_pos": torch.tensor([[[[8, 8], [0, 0]]] * 2], dtype=torch.int8),
        "cheese_alive": torch.tensor([[[True, False]] * 2]),
        "cheese_color": torch.tensor([[0, 1]]),
        "box_rect": torch.tensor([[[10, 10, 2, 3], [0, 0, 0, 0]]]),
        "box_color": torch.tensor([[2, -1]]),
    }


def test_text_grid():
    rows = text_grid(tiny_store(), 0, 0).split("\n")
    assert len(rows) == 16 and all(len(r) == 16 for r in rows)
    assert rows[0][0] == "o" and rows[5][5] == "#"
    assert rows[8][8] == rows[9][9] and rows[8][8].isupper()
    assert rows[10][10].islower() and rows[11][12].islower() and rows[12][12] == "."


def test_parse_reply_and_tags():
    reply = 'Sure.\n{"windows": [{"id": "a", "captions": ["x", "y", "z", "w"]}, {"id": "b", "captions": ["x", "y", "z"]}]}'
    assert parse_reply(reply, ["a", "b"]) == {"a": ["x", "y", "z", "w"]}  # one caption per tier, or the window is dropped
    assert parse_reply("not json", ["a"]) == {}
    assert tag_check("Move <d>up</d> to the <c>red</c> cheese") == []
    assert tag_check("Move up to the red cheese") == ["untagged direction: up", "untagged colour: red"]
    assert strip_tags("Go <d>left</d> to <c>red</c> <u>in a line</u>") == "Go left to red in a line"


def test_variants_identity_first_and_distinct():
    found = variants("Push the <c>red</c> cheese to the <d>top left corner</d>", random.Random(0))
    assert found[0] == ("Push the red cheese to the top left corner", (0, 1, 2, 3), 0)
    assert len({text for text, _, _ in found}) == len(found) > 1
    untaggable = variants("Move up", random.Random(0))
    assert untaggable == [("Move up", (0, 1, 2, 3), 0)]


def test_invariances():
    from beebot.vla.caption_set import invariances

    assert invariances("Eat the <c>red</c> cheese") == (True, False)
    assert invariances("Move <d>left</d>") == (False, True)
    assert invariances("Gather in the <d>center</d>") == (True, True)
    assert invariances("Move up") == (False, True)


def test_caption_set_tags_bare_colours_and_drops_distances():
    from beebot.vla.caption_set import MEASURED, invariances, tag_colours

    assert tag_colours("Eat red, push the <c>green</c> one") == "Eat <c>red</c>, push the <c>green</c> one"
    assert invariances(tag_colours("Eat the blue one")) == (True, False)
    assert tag_colours("<u>a ring around the red one</u>") == "<u>a ring around the red one</u>"
    assert MEASURED.search("move about six squares up") and MEASURED.search("push it 3 steps")
    assert not MEASURED.search("split into 3 groups") and not MEASURED.search("push the blue square up")
    assert MEASURED.search("push red six squares to the right") and MEASURED.search("one step to the left")
    assert not MEASURED.search("eat two squares") and not MEASURED.search("eat the two squares near the wall")
