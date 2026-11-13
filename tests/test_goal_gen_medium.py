import re

import numpy as np
import pytest

from beebot.vla.voxel import goal_gen_medium as M
from beebot.vla.voxel import goal_library as G
from beebot.vla.voxel import reverse


@pytest.mark.parametrize("name", sorted(M.GENERATORS))
def test_generator_valid(name):
    got = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        for _ in range(5):
            item = M.draw(name, rng)
            if item is not None:
                got.append(item)
    assert len(got) >= 20
    for cells, text, params in got:
        assert cells.shape[1] == 3
        assert 4 <= len(cells) <= 115
        assert cells.min() == 0 or cells[:, 0].min() == 0
        assert (cells.max(axis=0) < 16).all()
        assert G.grounded(G.cells_to_vox(cells))
        assert "overhang" in params and "span" in params
        low = text.lower()
        for word in M.FORBIDDEN:
            assert word not in low
        assert all(int(n) < 10 for n in re.findall(r"\d+", low))
    flags = M.at_rest([g[0] for g in got[:16]])
    assert all(flags)


def test_texts_vary():
    for name in M.GENERATORS:
        rng = np.random.default_rng(1)
        texts = {t for _, t, _ in (g for g in (M.draw(name, rng) for _ in range(60)) if g)}
        assert len(texts) >= 5, name


def test_sample_distinct_and_deterministic():
    a = M.sample(60, seed=3)
    b = M.sample(60, seed=3)
    assert len(a) == 60
    assert [x["text"] for x in a] == [x["text"] for x in b]
    keys = {(x["generator"], M.shape_key(x["cells"])) for x in a}
    assert len(keys) == 60


def test_overhang_stats():
    # a one-cell column with a 2-cell arm: overhang 2
    cells = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 0, 2]])
    over, span, hanging = M.overhang_stats(cells)
    assert over == 2
    assert span == 2
    assert hanging == 0
