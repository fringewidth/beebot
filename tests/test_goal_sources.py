"""Coloured goals: colour follows the cells through scaling, rotation and storage; converter helpers."""

import gzip
import importlib.util
import io

import numpy as np

from beebot.vla.voxel import goal_library as G
from beebot.vla.voxel import goal_sources as GS


def two_colour_tower():
    """A 24-high, 6 x 6 tower: red lower half, blue upper half (rgb per voxel)."""
    occupied = np.ones((24, 6, 6), dtype=bool)
    rgb = np.zeros((24, 6, 6, 3), dtype=np.uint8)
    rgb[:12] = (255, 0, 0)
    rgb[12:] = (0, 0, 255)
    return occupied, rgb


def test_colours_align_with_cells_and_follow_height():
    occupied, rgb = two_colour_tower()
    found, _ = GS.coloured_variants(rgb, occupied, np.random.default_rng(0), keep=(100,))
    assert found
    for scale, rot, vox, paint in found:
        assert paint.shape == vox.shape + (3,)
        assert (paint[~vox] == 0).all()
        zs = np.nonzero(vox)[0]
        colours = paint[np.nonzero(vox)]
        low = colours[zs <= zs.min() + 1]
        high = colours[zs >= zs.max() - 1]
        assert (low[:, 0] == 255).all() and (low[:, 2] == 0).all()  # rotation about the vertical axis keeps height colours
        assert (high[:, 2] == 255).all() and (high[:, 0] == 0).all()
        assert G.grounded(vox)


def test_upscaled_small_source_keeps_colours():
    occupied = np.zeros((2, 2, 2), dtype=bool)
    occupied[0] = True
    occupied[1, 0, 0] = True
    rgb = np.zeros(occupied.shape + (3,), dtype=np.uint8)
    rgb[0] = (10, 20, 30)
    rgb[1, 0, 0] = (200, 100, 50)
    found, _ = GS.coloured_variants(rgb, occupied, np.random.default_rng(1), keep=(40,), rotations=1)
    assert found
    scale, rot, vox, paint = found[0]
    assert scale >= 1
    colours = {tuple(c) for c in paint[vox]}
    assert colours == {(10, 20, 30), (200, 100, 50)}


def test_reconnected_cells_take_neighbour_colour_not_black():
    occupied = np.zeros((20, 20, 20), dtype=bool)
    occupied[0:3, 8:12, 8:12] = True  # base
    occupied[3:20, 9, 9] = True  # a one voxel post up to a head
    occupied[16:20, 5:14, 5:14] = True
    rgb = np.zeros(occupied.shape + (3,), dtype=np.uint8)
    rgb[occupied] = (120, 130, 140)
    found, _ = GS.coloured_variants(rgb, occupied, np.random.default_rng(2), keep=(60,), rotations=1)
    for _, _, vox, paint in found:
        assert (paint[vox].sum(axis=1) > 0).all()


def test_save_load_rgb_roundtrip_and_old_loader(tmp_path):
    cells = [np.array([[0, 0, 0], [1, 0, 0]], dtype=np.uint8), np.array([[0, 0, 0]], dtype=np.uint8)]
    rgb = [np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8), np.array([[7, 8, 9]], dtype=np.uint8)]
    lines = [{"id": "a:0", "cells": 2, "text": "x"}, {"id": "a:1", "cells": 1, "text": "y"}]
    GS.save_coloured("a", lines, cells, rgb, tmp_path)
    loaded_lines, loaded_cells = G.load("a", root=tmp_path)
    assert [line["id"] for line in loaded_lines] == ["a:0", "a:1"]
    assert (loaded_cells[0] == cells[0]).all()
    colours = GS.load_rgb("a", tmp_path)
    assert (colours[0] == rgb[0]).all() and (colours[1] == rgb[1]).all()
    GS.save_coloured("b", lines, cells, None, tmp_path)
    assert GS.load_rgb("b", tmp_path) is None


def test_cell_colours_order_matches_vox_to_cells():
    vox = np.zeros((2, 2, 2), dtype=bool)
    vox[0, 1, 0] = vox[1, 0, 1] = True
    paint = np.zeros(vox.shape + (3,), dtype=np.uint8)
    paint[0, 1, 0] = (1, 1, 1)
    paint[1, 0, 1] = (2, 2, 2)
    cells = G.vox_to_cells(vox)
    colours = GS.cell_colours(vox, paint)
    for cell, colour in zip(cells, colours):
        assert (paint[tuple(cell)] == colour).all()


def test_render_coloured_draws_the_colour():
    cells = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.uint8)
    image = GS.render_coloured(cells, np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8), scale=6)
    pixels = np.asarray(image).reshape(-1, 3)
    assert ((pixels[:, 0] > 150) & (pixels[:, 1] < 60)).any()
    assert ((pixels[:, 1] > 150) & (pixels[:, 0] < 60)).any()


def test_words():
    assert GS.words("Watch_Tower-02.vox") == "watch tower"
    assert GS.words("bamboo 01_32.vox") == "bamboo"


def load_converter():
    spec = importlib.util.spec_from_file_location("more", "scripts/goals_sources_more.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_oga_text_rules():
    more = load_converter()
    assert more.oga_text("alphabet-3d-voxel", "A.vox", "A.vox") == "the letter a"
    assert more.oga_text("voxel-numbers", "7.vox", "7.vox") == "the number 7"
    assert more.oga_text("voxel-buildings", "3.vox", "3.vox") == "building"
    assert more.oga_text("medieval", "Models.vox", "Decoration - Barrel - Top") == "barrel top"
    assert more.oga_text("farm-1", "CASA.vox", "CASA.vox") == "casa"


def test_parse_nrrd_orientation():
    more = load_converter()
    sizes = (4, 3, 3, 3)  # channel fastest, then up, depth, x
    memory = np.zeros((3, 3, 3, 4), dtype=np.uint8)  # (x, depth, up, channel)
    memory[2, 1, 0] = (9, 8, 7, 255)  # x = 2, depth = 1, up = 0
    header = "NRRD0004\ntype: uint8\ndimension: 4\nsizes: 4 3 3 3\nencoding: gzip\nendian: little\n\n".encode()
    data = header + gzip.compress(memory.tobytes())
    grid = more.parse_nrrd(data)
    assert grid.shape == (3, 3, 3, 4)
    assert tuple(grid[0, 1, 2]) == (9, 8, 7, 255)  # (z up, y depth, x)


def test_mesh_to_occupancy_box_is_upright_and_closed():
    import trimesh

    more = load_converter()
    box = trimesh.creation.box(extents=(1.0, 4.0, 1.0))  # tall along y; glTF y is up
    data = trimesh.Scene(box).export(file_type="glb")
    occupied = more.mesh_to_occupancy(data, res=16)
    assert occupied.shape[0] in (16, 17) and occupied.shape[1] <= 5 and occupied.shape[2] <= 5
    assert occupied.sum() >= 16 * 3 * 3 - 20  # a closed mesh is filled
