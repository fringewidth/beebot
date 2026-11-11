"""The grant demo's case (README "Demo: ten cubes on an empty floor"): task sizes fitted to the
swarm (scaled), scenes that are an empty flat floor (empty), and the demos --demo preset."""

import json
import sys

import numpy as np
import pytest
import torch

from beebot.vla.voxel import demos
from beebot.vla.voxel import objects as O
from beebot.vla.voxel import suite as S
from beebot.vla.voxel.play_store import load
from beebot.vla.voxel.world import CUBE_MOVES, failure_rates, world_from_arrays


def floor(n, row=0):
    """A world of n robots in a row along x on an empty flat floor, gravity on."""
    mass = np.zeros((16, 16, 16))
    mass[0, 2 + row, 1 : 1 + n] = 1
    return world_from_arrays(mass, gravity=True)


def test_a_tower_the_swarm_cannot_fill_two_by_two_is_one_column():
    assert S.column_side(10, 3, scaled=True) == 1 and S.column_side(12, 3, scaled=True) == 2
    assert all(S.column_side(n, h, scaled=False) == 2 for n in (3, 10, 40) for h in (3, 8))
    world = floor(10)
    meta = {"layout": "flat", "height": torch.zeros(16, 16, dtype=torch.long), "gap": None}
    tower = lambda points, h: S.Task("construction", "", {"kind": "tower", "height": h, "forbid": set()}, [{"kind": "targets", "points": points, "label": S.tower_label(h, points)}])  # noqa: E731
    one = S.tower_targets(meta, 6, np.random.default_rng(0), side=1)
    two = S.tower_targets(meta, 6, np.random.default_rng(0))
    assert len(one) == 6 and len({p[1:] for p in one}) == 1 and len(two) == 24
    assert S.score_bound(tower(one, 6), world, 0) == 1.0 and S.fits(tower(one, 6), world, 0)
    assert S.score_bound(tower(two, 6), world, 0) == 10 / 24 and not S.fits(tower(two, 6), world, 0)
    no_plan = S.Task("construction", "", {"kind": "tower", "height": 6, "forbid": set()})
    assert S.score_bound(no_plan, world, 0) == 10 / 24  # held out: as before
    made = [S.make_construction(np.random.default_rng(s), "", scaled=True) for s in range(40)]
    towers = [t for t in (make(world, meta, 0) for _, make in made) if t is not None and t.spec["kind"] == "tower"]
    assert towers and all(len(t.plan[0]["points"]) == t.spec["height"] and S.fits(t, world, 0) for t in towers)
    unscaled = [t for t in (make(world, meta, 0) for _, make in (S.make_construction(np.random.default_rng(s), "") for s in range(40))) if t is not None and t.spec["kind"] == "tower"]
    assert all(len(t.plan[0]["points"]) == 4 * t.spec["height"] for t in unscaled)


def test_a_parallel_tower_job_is_bounded_by_its_planned_column():
    world = floor(10)
    meta = {"layout": "flat", "height": torch.zeros(16, 16, dtype=torch.long), "gap": None}
    tasks = []
    for s in range(200):
        _, make = S.make_parallel(np.random.default_rng(s), "", scaled=True)
        task = make(world, meta, 0)
        if task is not None and any(job == "tower" for job, _ in task.spec["jobs"]):
            tasks.append(task)
    assert tasks
    for task in tasks:
        (part,) = [p for p, (job, _) in zip(task.plan[0]["parts"], task.spec["jobs"]) if job == "tower"]
        height = next(x for job, x in task.spec["jobs"] if job == "tower")
        assert len(part["points"]) == height  # 10 robots: never a full 2 x 2 column of 3 or more
        assert S.score_bound(task, world, 0) == 1.0


def test_letters_and_the_wall_come_at_half_size_when_the_swarm_cannot_do_the_full_one():
    assert int(S.shape_mask("wall", half=True).sum()) == 12 and int(S.shape_mask("wall").sum()) == 40
    with pytest.raises(ValueError):
        S.shape_mask("ring", half=True)
    assert int(S.letter_mask("L", False, scale=1).sum()) == 11 and int(S.letter_mask("L", False).sum()) == 44
    meta = {"layout": "flat", "height": torch.zeros(16, 16, dtype=torch.long), "gap": None}
    crowd = np.zeros((16, 16, 16))
    crowd[0, :6, :10] = 1  # 60 robots
    for world, robots, half in ((floor(9), 9, True), (world_from_arrays(crowd, gravity=True), 60, False)):
        seen = set()
        for s in range(60):
            _, make = S.make_formation(np.random.default_rng(s), "", scaled=True)
            task = make(world, meta, 0)
            label = task.plan[0]["label"]
            m = int(task.spec["mask"].sum())
            if label["kind"] == "letters" and len(label["word"]) == 1:
                full = int(S.letter_mask(label["word"], False).sum())
                assert (m < full) == half and (not half or m == int(S.letter_mask(label["word"], False, scale=1).sum()))
                seen.add("letter")
            if label["kind"] == "shape":
                assert label["shape"] == "wall" and (m == 12) == half  # gravity on: the wall only
                seen.add("wall")
            if half:
                assert S.mask_fits(task.spec["mask"], robots) or len(label.get("word", "")) == 2
        assert seen == {"letter", "wall"}


def test_a_group_tower_takes_a_robot_a_level_when_scaled():
    world = floor(6)
    meta = {"layout": "flat", "height": torch.zeros(16, 16, dtype=torch.long), "gap": None}
    assert S.job_spec(np.random.default_rng(0), "tower", world, meta, 0, 6) is None  # unscaled: 12 robots at least
    heights = {S.job_spec(np.random.default_rng(s), "tower", world, meta, 0, 6, scaled=True)[0]["height"] for s in range(40)}
    assert heights == {3, 4, 5, 6}
    spec = {"kind": "tower", "who": {"select": "all"}, "height": 5, "forbid": set()}
    setup = S.JobSetup(spec, world, 0)
    points = setup.tower_points(5)
    assert len(points) == 5 and len({p[1:] for p in points}) == 1
    assert S.coverage_bound(S.Task("subset", "", spec), world, 0) == 1.0
    assert len(S.JobSetup(spec, floor(12), 0).tower_points(3)) == 12  # enough robots: the 2 x 2 column as before


def test_the_expert_builds_a_one_wide_tower_under_switch_cube():
    world = floor(6)
    meta = {"layout": "flat", "height": torch.zeros(16, 16, dtype=torch.long), "gap": None}
    points = S.tower_targets(meta, 5, np.random.default_rng(1), side=1)
    task = S.Task("construction", "", {"kind": "tower", "height": 5, "forbid": set()}, [{"kind": "targets", "points": points, "label": S.tower_label(5, points)}])
    results, _ = S.run(world, [meta], [task], S.Expert([task], moves=CUBE_MOVES, seed=0, rule="switch_cube"), steps=72, seed=0, rule="switch_cube")
    assert results[0][0] == 1.0  # five of six cubes in one column


def test_object_towers_follow_the_swarm_when_scaled():
    world, meta = O.object_scenes(np.random.default_rng(0), robots=9, empty=True)
    assert len(O.tower_points(world, 0, 5, np.random.default_rng(0), scaled=True)) == 5
    assert len(O.tower_points(world, 0, 5, np.random.default_rng(0))) == 20
    _, _, tasks = O.sample_objects(np.random.default_rng(2), 12, family="reasoning", kind="conditional", robots=(8, 12), keep=True, empty=True, scaled=True)
    assert tasks and {O.task_kind(t) for t in tasks} <= {"conditional:eat_tower@else", "conditional:surround_goto@else"}
    for t in tasks:
        if t.spec["else"]["kind"] == "tower":
            assert len(t.plan[0]["points"]) == t.spec["else"]["height"]


def test_empty_scenes_are_a_bare_floor_with_gravity():
    world, metas = S.scenes(np.random.default_rng(0), 6, layout="gap", cheeses=(2, 3), robots=10, empty=True)
    assert not world.walls.any() and not world.cheese_alive.any() and world.gravity.all()
    assert (world.mass.sum(dim=(1, 2, 3)) == 10).all() and all(m["layout"] == "flat" and m["gap"] is None for m in metas)
    grouped, _ = S.scenes(np.random.default_rng(1), 4, robots=11, groups=(2, 2), empty=True)
    assert all(len(S.components(grouped.mass[i].numpy() > 0.5)) == 2 for i in range(4))


def test_an_empty_draw_gives_up_what_an_empty_floor_does_not_allow():
    rng = np.random.default_rng(0)
    world, metas, tasks = S.sample_tasks(rng, 12, "train", family="herd", robots=(8, 12), keep=True, empty=True)
    assert world is None and tasks == []  # every herd task needs a cheese
    world, metas, tasks = S.sample_tasks(rng, 40, "train", family="logic", robots=(8, 12), keep=True, empty=True)
    assert 0 < len(tasks) < 20 and {S.task_kind(t) for t in tasks} == {"eat:absent"}  # eat a colour that is not there
    world, _, tasks = S.sample_tasks(rng, 24, "train", robots=(8, 12), keep=True, empty=True, scaled=True)
    assert not world.walls.any() and not world.cheese_alive.any()
    assert {t.family for t in tasks} <= {"formation", "construction", "logic", "parallel"}
    world, _, lines = S.sample_coverage(np.random.default_rng(1), 6, family="motion", job="line", robots=(8, 12), keep=True, empty=True, scaled=True)
    assert all(8 <= c <= 12 for c in world.mass.sum(dim=(1, 2, 3)).long().tolist())  # the line's own 3 to 14 cut to the draw's
    unscaled, _, _ = S.sample_coverage(np.random.default_rng(1), 12, family="motion", job="line", robots=(8, 12))
    assert min(unscaled.mass.sum(dim=(1, 2, 3)).long().tolist()) < 8  # as before: the line's own range replaces the draw's
    with pytest.raises(ValueError):
        next(demos.draws(4, 0, 4, robots=(8, 12), empty=True))


def test_the_demo_preset_writes_an_empty_floor_store(tmp_path, monkeypatch):
    out = tmp_path / "demo"
    monkeypatch.setattr(sys, "argv", ["demos", "--out", str(out), "--demo", "--scenes", "8", "--chunk", "8", "--build", "0", "--steps", "24", "--compact"])
    demos.main()
    store = load(out)
    config = store["config"]
    assert config["preset"] == "demo" and config["empty"] and config["scaled"] and config["keep"] and config["fit"]
    assert config["robots"] == [8, 12] and config["rule"] == "switch_cube" and config["moves"] == "cube" and config["expert"] == "attractor"
    assert config["coverage"] == 1.0 and config["objects"] == 1.0 and config["build"] == 0.0 and config["steps"] == 24  # given options win
    if "outcomes" in demos.PRESETS["demo"]:  # world.py's measured outcome model, once there is one
        assert config["outcomes"] == demos.PRESETS["demo"]["outcomes"]
    else:
        assert np.allclose(config["failure"], failure_rates(0.1, up_over=0.23).tolist())
    attempts = [json.loads(line) for line in open(out / "attempts.jsonl")]
    assert attempts and all(8 <= a["robots"] <= 12 for a in attempts)
    assert store["count"] > 0 and not store["walls"].any() and not store["cheese_alive"].any() and store["gravity"].all()
    from beebot.vla.voxel import haiku

    options, rows = haiku.store_options(out)
    assert options["empty"] and options["scaled"]
    items = haiku.drawn_items({**options, "chunk": 8}, None, rows)  # a Haiku export of the store joins to it
    assert len(items) == store["count"]
