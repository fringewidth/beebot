import numpy as np
import torch

from beebot.vla.voxel import goal_library as G
from beebot.vla.voxel import world as W


def test_parse_bricks_covers_each_brick():
    vox = G.parse_bricks("2x4 (1,2,0)\n1x1 (0,0,3)\n")
    assert vox.shape == (20, 20, 20)
    assert vox[0, 2:6, 1:3].all() and vox[0].sum() == 8  # h = 2 along x, w = 4 along y
    assert vox[3, 0, 0] and vox.sum() == 9


def test_parse_blueprint_skips_air_and_fluid():
    blueprint = {
        "size": [3, 2, 2],
        "palette": {"a": "minecraft:stone", "b": "minecraft:air", "c": "minecraft:water"},
        "layers": [["aab", "bcb"], ["bba", "bbb"]],
    }
    vox = G.parse_blueprint(blueprint)
    assert vox.shape == (2, 2, 3)
    assert vox.sum() == 3 and vox[0, 0, 0] and vox[0, 0, 1] and vox[1, 0, 2]


def test_parse_blocks_y_up_and_crop():
    vox = G.parse_blocks([[5, 63, -2, 1], [5, 64, -2, 1], [6, 63, -2, 2]])
    assert vox.shape == (2, 1, 2)
    assert vox[0, 0, 0] and vox[1, 0, 0] and vox[0, 0, 1] and not vox[1, 0, 1]
    assert G.parse_blocks([]) is None


def test_craftassist_occupancy_ignores_meta():
    schematic = np.zeros((2, 2, 2, 2), dtype=np.uint8)
    schematic[0, 1, 1] = (5, 3)
    schematic[1, 0, 0] = (0, 7)
    assert G.craftassist_occupancy(schematic).sum() == 1


def test_rescale_down_and_up():
    vox = np.ones((8, 8, 8), dtype=bool)
    assert G.rescale(vox, 0.5).shape == (4, 4, 4) and G.rescale(vox, 0.5).all()
    up = G.rescale(np.ones((2, 3, 4), dtype=bool), 2.0)
    assert up.shape == (4, 6, 8) and up.all()
    thin = np.zeros((6, 6, 6), dtype=bool)
    thin[:, 0, :] = True  # one-thick wall: a third of the voxels under each coarse cell
    assert G.rescale(thin, 1 / 3).any()


def test_rotate_keeps_cells_and_four_turns_return():
    vox = np.zeros((2, 3, 4), dtype=bool)
    vox[0, 0, 0] = vox[1, 2, 3] = True
    assert G.rotate(vox, 1).sum() == 2 and G.rotate(vox, 1).shape == (2, 4, 3)
    assert (G.rotate(vox, 4) == vox).all()


def test_grounded_and_connected():
    tower = np.zeros((4, 1, 1), dtype=bool)
    tower[:, 0, 0] = True
    assert G.grounded(tower) and G.connected(tower)
    floating = tower.copy()
    floating[1, 0, 0] = False
    assert not G.grounded(floating) and not G.connected(floating)
    overhang = np.zeros((3, 1, 3), dtype=bool)
    overhang[0, 0, 0] = overhang[1, 0, 0] = overhang[1, 0, 1] = overhang[1, 0, 2] = True  # face-connected, held by a magnet
    assert G.grounded(overhang)
    two = np.zeros((1, 1, 3), dtype=bool)
    two[0, 0, 0] = two[0, 0, 2] = True
    assert G.grounded(two) and not G.connected(two)


def test_grounded_agrees_with_world_supported():
    from beebot.vla.voxel import reverse

    rng = np.random.default_rng(3)
    for _ in range(25):
        vox = rng.random((4, 4, 4)) < 0.5
        if not vox.any():
            continue
        goal = G.to_goal(G.vox_to_cells(vox))
        held = W.supported(goal.world)[0]
        want = bool((held == (goal.world.mass[0] > 0.5)).all())
        assert G.grounded(vox) == want
        assert goal.target.sum() == vox.sum() and bool(reverse.blank().mass.sum() == 0)


def test_variants_fit_world_and_budget():
    vox = np.zeros((20, 20, 20), dtype=bool)
    vox[:12, 4:16, 4:16] = True
    found, tally = G.variants(vox, np.random.default_rng(0))
    assert found
    for scale, rot, out in found:
        assert max(out.shape) <= G.SIZE
        assert G.MIN_CELLS <= out.sum() <= G.MAX_CELLS
        assert G.grounded(out)
    assert sum(tally.values()) > 0


def test_variants_parameters_cap_and_targets():
    vox = np.ones((16, 16, 16), dtype=bool)
    found, _ = G.variants(vox, np.random.default_rng(0), keep=(100, 800), max_cells=800, rotations=1)
    counts = sorted(int(v.sum()) for _, _, v in found)
    assert counts and counts[-1] > 115 and counts[-1] <= 800
    capped, tally = G.variants(vox, np.random.default_rng(0), keep=(100, 800), max_cells=300, rotations=1)
    assert all(int(v.sum()) <= 300 for _, _, v in capped) and tally["big"] > 0
    assert G.MAX_CELLS == 256 and G.KEEP_TARGETS == (64, 128, 256)
    assert max(G.KEEP_TARGETS) <= G.MAX_CELLS


def test_variants_reject_floating_structure(monkeypatch):
    monkeypatch.setattr(G, "SCALES", (1.0,))
    vox = np.zeros((6, 3, 3), dtype=bool)
    vox[0, 1, 1] = True
    vox[4:6, :, :] = True  # a floating slab over a single cube
    found, tally = G.variants(vox, np.random.default_rng(0))
    assert not found and tally["ungrounded"] > 0


def test_crops_rest_on_bottom_and_are_inside():
    vox = np.zeros((10, 30, 30), dtype=bool)
    vox[:, 2:28, 2:28] = True
    pieces = G.crops(vox, np.random.default_rng(1), n=3)
    assert len(pieces) == 3
    for piece in pieces:
        assert max(piece.shape) <= G.SIZE and piece.sum() >= G.MIN_CELLS


def test_save_load_roundtrip(tmp_path):
    item = G.Item("src", "a", "whole", np.ones((2, 2, 2), dtype=bool), text="a cube", label="x")
    goals, _ = G.process(item)
    assert goals
    lines = [G.entry_line(item, g, f"src:{i}") for i, g in enumerate(goals)]
    G.save("src", lines, [G.vox_to_cells(g["vox"]) for g in goals], root=tmp_path)
    back_lines, back_cells = G.load("src", root=tmp_path)
    assert back_lines == lines
    for g, cells in zip(goals, back_cells):
        assert (G.cells_to_vox(cells) == g["vox"]).all()
    assert back_lines[0]["text"] == "a cube" and back_lines[0]["cells"] == int(goals[0]["vox"].sum())


def test_to_goal_is_centred_and_reversible_shape():
    from beebot.vla.voxel import reverse

    cells = G.vox_to_cells(np.ones((1, 2, 2), dtype=bool))
    goal = G.to_goal(cells, "a tile")
    assert goal.family == "library" and goal.text == "a tile"
    assert int(goal.world.mass.sum()) == 4 and bool(goal.target.any())
    assert reverse.fixed_point(goal.world).all()
    episodes, _ = reverse.take_apart([goal], np.random.default_rng(0), max_steps=8)
    assert torch.is_tensor(reverse.replay(episodes)["passed"])


def test_histogram_bins():
    assert G.histogram([4, 7, 8, 115, 116])["4-7"] == 2
    assert G.histogram([4, 7, 8, 115, 116])["100-115"] == 1
    assert G.histogram([116, 250, 251, 500, 501, 800])["116-250"] == 2


def _table():
    vox = np.zeros((10, 20, 20), dtype=bool)
    vox[9] = True
    for y in (0, 19):
        for x in (0, 19):
            vox[:9, y, x] = True
    return vox


def test_search_keeps_legs_old_rule_loses_them():
    base = G.crop_bbox(_table())
    old = G.best_variant(base, 0.5, 800, search=False)
    new = G.best_variant(base, 0.5, 800, search=True)
    assert not isinstance(new, str)
    assert new["vox"].shape[0] > 1 and new["vox"][0].sum() >= 4  # legs reach the floor
    assert isinstance(old, str) or old["vox"].shape[0] == 1  # old: a legless slab, cropped down to z = 0
    assert new["iou"] >= 0.9


def test_repair_adds_only_cells_with_source_voxels_and_leaves_interior_hollow():
    fill = np.zeros((6, 5, 5), dtype=np.float32)
    mask = np.zeros((6, 5, 5), dtype=bool)
    mask[5] = True
    fill[5] = 1.0
    fill[:5, 2, 2] = 0.1  # a thin post
    out = G.repair(mask, fill)
    assert G.grounded(out)
    assert not (out & ~mask & (fill == 0)).any()
    assert int(out.sum()) == int(mask.sum()) + 5


def test_variants_search_flag():
    found, _ = G.variants(_table(), np.random.default_rng(0), search=False, rotations=1)
    assert found


def test_load_drops_goals_above_cap(tmp_path):
    small = np.argwhere(np.ones((2, 2, 2), dtype=bool)).astype(np.uint8)
    big = np.argwhere(np.ones((6, 6, 8), dtype=bool)).astype(np.uint8)
    assert len(big) > G.MAX_CELLS
    G.save("t", [{"id": "a"}, {"id": "b"}], [small, big], root=tmp_path)
    lines, cells = G.load("t", tmp_path)
    assert [line["id"] for line in lines] == ["a"] and len(cells) == 1
    lines, cells = G.load("t", tmp_path, max_cells=None)
    assert len(lines) == 2 and len(cells[1]) == len(big)


def test_variants_tolerance_skips_far_targets():
    vox = np.ones((4, 4, 4), dtype=bool)
    near, _ = G.variants(vox, np.random.default_rng(0), keep=(8, 64), rotations=1, tolerance=2)
    counts = sorted(int(v.sum()) for _, _, v in near)
    assert counts and all(4 <= c <= 128 for c in counts)
    far, _ = G.variants(vox, np.random.default_rng(0), keep=(1000,), rotations=1, tolerance=2)
    assert far == []


def test_dropped_sources_load_empty_unless_asked(tmp_path):
    small = np.argwhere(np.ones((2, 2, 2), dtype=bool)).astype(np.uint8)
    for name in G.DROPPED_SOURCES:
        G.save(name, [{"id": "a"}], [small], root=tmp_path)
        assert G.load(name, tmp_path) == ([], [])
        lines, cells = G.load(name, tmp_path, include_dropped=True)
        assert len(lines) == 1 and len(cells) == 1
    assert set(G.DROPPED_SOURCES) == {"oga_vox", "itch_vox", "fusion360"}
