import torch

from beebot.vla.caption_aug import WEST, NORTH, EAST, SOUTH, rewrite, turn
from beebot.vla.data import transform_grid
from beebot.vla.world import PALETTE_NAMES


def plain(caption):
    import re

    return re.sub(r"</?[dcu]>", "", caption)


def test_turn_matches_transform_grid():
    for k in range(4):
        for flip in (False, True):
            for direction in (NORTH, SOUTH, WEST, EAST):
                grid = torch.zeros(5, 5)
                grid[2 + direction[0], 2 + direction[1]] = 1
                out = transform_grid(grid, k, flip)
                row, col = (out == 1).nonzero()[0].tolist()
                assert (row - 2, col - 2) == turn(direction, k, flip)


def test_quarter_turn_places_and_motions():
    assert plain(rewrite("Go to the <d>top left corner</d>", k=1)) == "Go to the bottom left corner"
    assert plain(rewrite("Move <d>left</d>", k=1)) == "Move down"
    assert plain(rewrite("Gather on <d>the left</d>", k=1)) == "Gather on the bottom"
    assert plain(rewrite("Gather on the <d>left</d>", k=1)) == "Gather on the bottom"
    assert plain(rewrite("Move <d>up and to the left</d>", k=1)) == "Move down and to the left"
    assert plain(rewrite("Move <d>up and left</d>", k=1)) == "Move down and left"
    assert plain(rewrite("Push it to the <d>upper right</d>", k=1)) == "Push it to the upper left"
    assert rewrite("Line up <d>horizontally</d>", k=1) is None  # "up" outside a tag
    assert plain(rewrite("Spread <d>horizontally</d>", k=1)) == "Spread vertically"


def test_flip_and_identity():
    assert plain(rewrite("Go <d>right</d> to the <d>bottom right</d>", flip=True)) == "Go left to the bottom left"
    assert rewrite("Go <d>Up</d>", k=0) == "Go <d>Up</d>"
    assert plain(rewrite("<d>Up</d> we go", k=2)) == "Down we go"


def test_fails_closed():
    assert rewrite("Move left", k=1) is None
    assert rewrite("Move <d>above</d> the cheese", k=1) is None
    assert rewrite("Eat the red cheese", colour_map=[1, 0, 2, 3]) is None
    assert rewrite("Eat the <c>red</c> cheese <d>left", k=1) is None


def test_colours():
    perm = [1, 2, 3, 0]
    out = rewrite("Eat the <c>Red</c> cheese, not the <c>blue</c> one", colour_map=perm)
    red, blue = PALETTE_NAMES.index("red"), PALETTE_NAMES.index("blue")
    assert plain(out) == f"Eat the {PALETTE_NAMES[perm[red]].capitalize()} cheese, not the {PALETTE_NAMES[perm[blue]]} one"


def test_unverified_claims_are_rewritten_inside():
    assert plain(rewrite("<u>form a ring <d>left</d> of it</u>", k=2)) == "form a ring right of it"
