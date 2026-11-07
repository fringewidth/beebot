"""The voxel task suite (vla/voxel/tasks.py): checkers on facts, no credit for doing nothing,
held-out pairs kept out of training."""

import numpy as np

from beebot.vla.voxel import tasks as V
from beebot.vla.voxel.facts import Facts


def _facts(*atoms):
    return Facts(atoms=set(atoms))


def _find(options, family, needle):
    return next(spec for text, spec in options[family] if needle in text)


def test_tower_checker_accepts_taller_than_asked_and_rejects_nothing():
    options = V.candidates({"colours": ["red"], "gravity": True}, np.random.default_rng(0))
    medium = next(spec for text, spec in options["tower"] if "medium" in text)
    assert V.check(medium, _facts(("tower", "tall"))) == (1.0, False)
    assert V.check(medium, _facts(("tower", "low"))) == (0.0, False)
    assert V.check(medium, _facts()) == (0.0, False)


def test_surround_needs_the_cheese_kept_and_doing_nothing_scores_zero():
    options = V.candidates({"colours": ["red", "blue"], "gravity": False}, np.random.default_rng(0))
    spec = next(spec for text, spec in options["around"] if "red" in text)
    assert V.check(spec, _facts()) == (0.0, False)
    assert V.check(spec, _facts(("around", "red"))) == (1.0, False)
    assert V.check(spec, _facts(("around", "red"), ("ate", "red"))) == (0.0, True)


def test_eat_logic_except_and_none():
    options = V.candidates({"colours": ["red", "blue", "green"], "gravity": False}, np.random.default_rng(1))
    except_blue = next(spec for _, spec in options["eat"] if spec["form"] == "except" and spec["forbid"] == {"blue"})
    assert V.check(except_blue, _facts(("ate", "red"), ("ate", "green"))) == (1.0, False)
    assert V.check(except_blue, _facts(("ate", "red"), ("ate", "blue")))[1] is True
    none = next(spec for _, spec in options["eat"] if spec["form"] == "none")
    assert V.check(none, _facts()) == (1.0, False)


def test_training_pairs_exclude_held_out_family_pairs():
    rng = np.random.default_rng(2)
    options = V.candidates({"colours": ["red", "blue"], "gravity": True}, rng)
    for _ in range(20):
        for _, spec in V.pair_options(options, rng):
            assert tuple(sorted(spec["parts"])) not in V.HELD_PAIRS
        for _, spec in V.pair_options(options, rng, held=True):
            assert tuple(sorted(spec["parts"])) in V.HELD_PAIRS
