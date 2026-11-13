import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from kaggle_follow import new_lines


def test_holds_back_the_unfinished_line_and_never_repeats():
    lines, seen = new_lines("a\nb\nc", 0, final=False)
    assert (lines, seen) == (["a", "b"], 2)
    lines, seen = new_lines("a\nb\nc\nd\n", seen, final=False)
    assert (lines, seen) == (["c", "d"], 4)
    lines, seen = new_lines("a\nb\nc\nd\n", seen, final=False)
    assert (lines, seen) == ([], 4)


def test_final_read_takes_the_last_line():
    assert new_lines("a\nb", 1, final=True) == (["b"], 2)
    assert new_lines("a\nb\n", 1, final=True) == (["b"], 2)


def test_a_shorter_read_changes_nothing():
    assert new_lines("a\n", 5, final=False) == ([], 5)
