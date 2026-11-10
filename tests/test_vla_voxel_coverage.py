"""Coverage of the voxel suite (vla/voxel/suite.py, README "Coverage"): selectors, the new jobs,
scenes with separate groups, instruction switches, canonical records and scene summaries, and
the draws of the existing families left as they were."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from beebot.vla.voxel import demos
from beebot.vla.voxel import suite as S
from beebot.vla.voxel import wordings as W
from beebot.vla.voxel.world import COLOUR_NAMES, CUBE_MOVES, STAY, failure_rates, world_from_arrays


def digest(seed, n, **kw):
    rng = np.random.default_rng(seed)
    world, metas, tasks = S.sample_tasks(rng, n, "train", **kw)
    world, metas, tasks = demos.contrast(rng, world, metas, tasks, check=kw.get("robots") is not None)
    h = hashlib.sha256()
    for f in ("mass", "walls", "cheese_pos", "cheese_alive", "cheese_color", "gravity"):
        h.update(getattr(world, f).numpy().tobytes())
    for t in tasks:
        h.update((t.family + "|" + t.text + "|" + S.task_kind(t)).encode())
    h.update(str(rng.integers(1 << 30)).encode())
    return h.hexdigest()[:16]


def test_existing_draws_are_unchanged():
    """Digests of sample_tasks + demos.contrast taken before the coverage families were added."""
    assert digest(0, 12) == "223468c15763e35a"
    assert digest(1, 12, robots=S.ROBOTS) == "0fba3776bc1a3336"
    assert digest(2, 12, robots=S.ROBOTS, counts="log") == "8e52662f26b3cd06"


def two_groups(gravity=True, cheese=None):
    """A world with a group of 3 robots on the left and one of 6 on the right (all on the floor)."""
    mass = np.zeros((16, 16, 16))
    mass[0, 7, 1:4] = 1
    mass[0, 7:9, 10:13] = 1
    cheeses = [] if cheese is None else [cheese]
    return world_from_arrays(mass, cheeses=cheeses, gravity=gravity)


def test_selectors_pick_the_named_robots():
    world = two_groups(cheese=(0, 1, 11, COLOUR_NAMES.index("red")))
    occupied = world.mass[0].numpy() > 0.5
    centres = S.cheese_centres(world.cheese_color[0].tolist(), S.snapshot(world, 0))
    left = np.zeros_like(occupied)
    left[0, 7, 1:4] = True
    pick = lambda **who: S.chosen(who, occupied, centres)  # noqa: E731
    assert (pick(select="group", by="size", which="smaller") == left).all()
    assert (pick(select="group", by="size", which="larger") == (occupied & ~left)).all()
    assert (pick(select="group", by="side", which="left") == left).all()
    assert (pick(select="group", by="side", which="right") == (occupied & ~left)).all()
    assert (pick(select="group", by="cheese", which="red") == (occupied & ~left)).all()
    assert (pick(select="group", by="size", which="smaller", **{"except": True}) == (occupied & ~left)).all()
    assert pick(select="count", count=3) is None and pick(select="all").sum() == 9
    stacked = occupied.copy()
    stacked[1, 7, 1] = True
    top = S.chosen({"select": "top"}, stacked, centres)
    assert top.sum() == 1 and top[1, 7, 1]


def test_stay_scores_the_share_still_and_the_expert_stays_under_every_rule():
    world = two_groups()
    task = S.coverage_task("motion", "stay where you are", {"kind": "stay", "who": dict(S.ALL), "forbid": set()}, "")
    static = S.static_of(world, None, 0)
    start = S.snapshot(world, 0)
    moved = {**start, "mass": start["mass"].copy()}
    moved["mass"][0, 7, 1] = 0
    moved["mass"][0, 2, 2] = 1
    assert S.score(task, static, start, start, [])[0] == 1.0
    assert abs(S.score(task, static, start, moved, [])[0] - 8 / 9) < 1e-6
    for rule in (None, "switch_cube", "free_cube"):
        results, arrays = S.run(world, [None], [task], S.Expert([task], moves=CUBE_MOVES, seed=0, rule=rule), steps=12, seed=0, record=True, failure=failure_rates(), rule=rule)
        assert results[0] == (1.0, False)
        assert (arrays["actions"] == STAY).all()


def test_robots_left_out_stay_and_moving_them_costs():
    world = two_groups()
    spec = {"kind": "goto", "who": {"select": "group", "by": "size", "which": "smaller"}, "region": "front left", "forbid": set()}
    task = S.coverage_task("subset", "", spec, "")
    expert = S.Expert([task], moves=CUBE_MOVES, seed=0, rule="switch_cube")
    results, arrays = S.run(world, [None], [task], expert, steps=40, seed=0, record=True, rule="switch_cube")
    right = np.zeros((16, 16, 16), bool)
    right[0, 7:9, 10:13] = True
    assert (arrays["actions"][0][:, torch.from_numpy(right)] == STAY).all()  # the larger group never asked to move
    assert (arrays["mass"][0, -1][torch.from_numpy(right)] == 1).all()
    assert results[0][0] >= 0.75
    static = S.static_of(world, None, 0)
    start = S.snapshot(world, 0)
    end = {**start, "mass": arrays["mass"][0, -1].numpy().astype(np.float32)}
    end["mass"][0, 8, 12] = 0  # one of the six moved away
    end["mass"][2, 2, 2] = 1
    assert abs(S.score(task, static, start, end, [])[0] - results[0][0] * 5 / 6) < 1e-6


def test_count_selectors_judge_the_robots_that_moved():
    world = two_groups()
    spec = {"kind": "goto", "who": {"select": "count", "count": 3}, "region": "front left", "forbid": set()}
    task = S.coverage_task("subset", "", spec, "")
    static = S.static_of(world, None, 0)
    start = S.snapshot(world, 0)
    end = {**start, "mass": start["mass"].copy()}
    end["mass"][0, 7, 1:4] = 0
    end["mass"][0, 12, 1:4] = 1  # three robots went to the front left
    assert S.score(task, static, start, end, [])[0] == 1.0
    end["mass"][0, 7:9, 10:13] = 0
    end["mass"][0, 13, 0:6] = 1  # six more went there too: nine moved where three were asked
    assert abs(S.score(task, static, start, end, [])[0] - 3 / 9) < 1e-6


def test_multi_group_scenes_hold_separate_groups_of_different_sizes():
    rng = np.random.default_rng(3)
    world, metas = S.scenes(rng, 12, groups=(2, 3), robots=40)
    for i in range(12):
        found = S.components(world.mass[i].numpy() > 0.5)
        sizes = sorted(len(g) for g in found)
        assert len(found) in (2, 3) and sum(sizes) == 40
        assert all(b >= 1.4 * a for a, b in zip(sizes, sizes[1:]))
        robots = S.scene_summary(world, i)["robots"]
        assert len(robots["groups"]) == len(found) and robots["alone"] == 0 and sorted(g["size"] for g in robots["groups"]) == sizes
        assert S.shells_of(world, i)[world.mass[i].numpy() > 0.5].sum() == 0  # no robot starts on a cheese's shell


@pytest.mark.parametrize("family", S.COVERAGE)
def test_coverage_tasks_have_records_that_parse_back(family):
    rng = np.random.default_rng(11)
    world, metas, tasks = S.sample_coverage(rng, 24, family=family, robots=S.ROBOTS, counts="log")
    for i, task in enumerate(tasks):
        assert task.family == family and (task.plan or (task.parts and task.parts[0].spec.get("form") in ("none", "absent")))
        record = S.canonical_record(task)
        assert set(record) == {"family", "kind", "plain", "spec", "subgoals"}
        json.dumps(record)
        json.dumps(S.scene_summary(world, i))
        assert record["subgoals"] and all(isinstance(s, str) for s in record["subgoals"])
        parts = task.parts or [task]
        for part in parts:
            slots = W.slots_of(part)
            assert W.parse_canonical(W.canonical(slots)) == json.loads(json.dumps(slots))
        if task.parts:
            assert [p["t"] for p in record["spec"]["parts"]] == task.spec["steps"] and S.instruction_at(task, task.spec["steps"][1]) is task.parts[1]
            assert record["plain"] == "; then, on a new instruction: ".join(S.canonical_record(p)["plain"] for p in task.parts)


def test_every_old_task_has_a_record():
    rng = np.random.default_rng(12)
    _, _, tasks = S.sample_tasks(rng, 40, "train")
    for split in ("colour", "letters", "shapes", "logic", "pairs", "construction", "herd_two", "runner_logic"):
        tasks += S.sample_tasks(rng, 1, split)[2]
    for task in tasks:
        record = S.canonical_record(task)
        json.dumps(record)
        assert record["plain"] == W.canonical(W.slots_of(task)) and len(record["subgoals"]) == len(task.plan)
    eat = [t for t in S.sample_tasks(rng, 30, "train", "logic")[2] if t.spec["order"]]
    assert eat and all(S.canonical_record(t)["subgoals"] == [f"eat the {c} cheese" for c in t.spec["order"]] for t in eat)


def test_scene_summary_names_the_terrain_and_cheeses():
    rng = np.random.default_rng(5)
    world, metas = S.scenes(rng, 6, layout="gap", cheeses=(1, 2))
    for i in range(6):
        summary = S.scene_summary(world, i)
        axis, start, width, top = S.trench(metas[i])
        trench = summary["terrain"]["trench"]
        assert summary["terrain"]["layout"] == "trench between two plateaus" and trench["width"] == width and trench["depth"] == top
        assert trench["runs"] == ("back to front" if axis == 2 else "left to right") and trench["from_x" if axis == 2 else "from_y"] == start
        assert sorted(c["colour"] for c in summary["cheeses"]) == sorted(c for _, c in S.present(world, i))
        assert summary["robots"]["count"] == int(world.mass[i].sum())
    flat, _ = S.scenes(rng, 1, layout="flat")
    assert S.scene_summary(flat, 0)["terrain"] == {"layout": "flat floor"}


def test_records_and_summaries_match_the_haiku_stand_ins():
    """suite.canonical_record and scene_summary give what haiku.local_canonical_record and
    local_scene_summary gave for the six families; the summary only adds fields."""
    from beebot.vla.voxel import haiku

    rng = np.random.default_rng(15)
    world, _, tasks = S.sample_tasks(rng, 60, "train", robots=S.ROBOTS, counts="log")
    for i, task in enumerate(tasks):
        assert S.canonical_record(task) == haiku.local_canonical_record(task)
        mine, theirs = S.scene_summary(world, i), haiku.local_scene_summary(world, i)
        assert {k: v for k, v in mine.items() if k != "robots"} == {k: v for k, v in theirs.items() if k != "robots"}
        assert {k: mine["robots"][k] for k in theirs["robots"]} == theirs["robots"]
    assert set(haiku.LOCAL_NOTES.items()) <= set(S.SPEC_NOTES.items())


def test_a_switch_changes_the_expert_at_its_step_and_is_scored_from_there():
    rng = np.random.default_rng(9)
    world, metas, tasks = S.sample_coverage(rng, 8, family="switch")
    expert = S.Expert(tasks, seed=0)
    results, arrays = S.run(world, metas, tasks, expert, steps=40, seed=0, record=True)
    for i, task in enumerate(tasks):
        at = task.spec["steps"][1]
        assert any(entry["t"] == at for entry in expert.subgoals[i]), (task.spec, expert.subgoals[i])
        if task.parts[1].spec["kind"] == "stay":
            assert (arrays["actions"][i, at:] == STAY).all()
            static = S.static_of(world, metas[i], i)
            mid = {"mass": arrays["mass"][i, at].numpy(), "cheese_pos": arrays["cheese_pos"][i, at].numpy(), "cheese_alive": arrays["cheese_alive"][i, at].numpy()}
            end = {"mass": arrays["mass"][i, -1].numpy(), "cheese_pos": arrays["cheese_pos"][i, -1].numpy(), "cheese_alive": arrays["cheese_alive"][i, -1].numpy()}
            assert abs(results[i][0] - S.score(task.parts[1], static, mid, end, [])[0]) < 1e-6


@pytest.mark.parametrize("family", S.COVERAGE)
def test_the_expert_passes_most_coverage_tasks(family):
    rng = np.random.default_rng(21)
    world, metas, tasks = S.sample_coverage(rng, 16, family=family, robots=S.ROBOTS, counts="log")
    world = replace(world, gravity=torch.ones_like(world.gravity))
    results, _ = S.run(world, metas, tasks, S.Expert(tasks, moves=CUBE_MOVES, seed=0, rule="switch_cube"), steps=S.CUBE_STEPS, seed=0, rule="switch_cube")
    passed = [not u and s >= S.PASS_MARK for s, u in results]
    assert np.mean(passed) >= 0.6, [(S.task_kind(t), round(s, 2)) for t, (s, _) in zip(tasks, results)]


def test_held_out_coverage_splits_force_their_item():
    rng = np.random.default_rng(4)
    _, _, tasks = S.sample_coverage(rng, 3, split="selector_job")
    assert all(t.held_out == "selector_job" and not t.plan and S.task_kind(t) == "approach@half" for t in tasks)
    _, _, train = S.sample_coverage(rng, 30, family="subset")
    assert not any(S.task_kind(t) == "approach@half" for t in train)


def test_coverage_wordings_round_trip_in_every_split():
    rng = np.random.default_rng(13)
    world, metas, tasks = S.sample_coverage(rng, 30, robots=S.ROBOTS, counts="log")
    choice = np.random.default_rng(14)
    for i, task in enumerate(tasks):
        present = [c for _, c in S.present(world, i)]
        for part in task.parts or [task]:
            slots = W.slots_of(part, present)
            for split in W.SPLITS:
                made = W.render(part.text, slots, choice, split)
                assert W.parse_canonical(made.canonical)["intent"] == slots["intent"]
                assert "{" not in made.instruction
