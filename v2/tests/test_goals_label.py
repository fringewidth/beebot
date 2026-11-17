import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

from beebot.vla.voxel import goal_library as G

spec = importlib.util.spec_from_file_location("goals_label", Path(__file__).parent.parent / "scripts" / "goals_label.py")
L = importlib.util.module_from_spec(spec)
sys.modules["goals_label"] = L
spec.loader.exec_module(L)


def fake_library(root):
    lines = []
    cells = []
    for i in range(40):
        gen = "alpha" if i % 2 else "beta"
        lines.append({"id": f"s:{i}", "source": "s", "generator": gen, "cells": 4 + i, "text": f"shape number {i} of {gen}"})
        cells.append(np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.uint8))
    G.save("s", lines, cells, root=root)
    return lines


def test_sample_spreads_over_generators_and_sizes(tmp_path):
    lines = fake_library(tmp_path)
    chosen = L.sample_indices(lines, 12)
    assert len(chosen) == 12 and len(set(chosen)) == 12
    assert {lines[i]["generator"] for i in chosen} == {"alpha", "beta"}
    assert len({min(lines[i]["cells"] // 16, 3) for i in chosen}) > 1


def test_controls_come_from_other_groups(tmp_path):
    lines = fake_library(tmp_path)
    rng = random.Random(0)
    pool = list(range(len(lines)))
    for index in (0, 1, 7):
        picks = L.pick_others(lines, index, 3, rng, pool)
        assert len(picks) == 3 and index not in picks
        assert all(lines[j]["generator"] != lines[index]["generator"] for j in picks)


def test_stub_run_writes_resumable_labels_and_progress(tmp_path):
    fake_library(tmp_path)
    args = L.argparse.Namespace(library=str(tmp_path), root=str(tmp_path), per_source=6, seed=0, limit=4, method="both", deadline=0, log=str(tmp_path / "log.out"))
    rows, _ = L.run_source(L.Stub(), "s", args, random.Random(0))
    assert len(rows) == 4
    first = json.loads(open(tmp_path / "labels_s.jsonl").readline())
    assert first["own_yes"] is True and first["control_yes"] is True and first["control_id"] != first["id"]
    assert first["mc"]["true_letter"] in "ABCD" and first["revision"] == L.REVISION
    args.limit = 0
    rows, _ = L.run_source(L.Stub(), "s", args, random.Random(0))
    assert len(rows) == 6
    assert len(open(tmp_path / "labels_s.jsonl").readlines()) == 6


def test_what_of_cuts_copyright_tail():
    assert L.what_of({"text": "House. © Someone 2018. All rights reserved."}) == "House"
    assert L.what_of({"text": "", "label": "roof"}) == "roof"
