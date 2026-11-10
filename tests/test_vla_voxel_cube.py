"""Pivoting-cube demonstrations of the voxel suite: no horizontal diagonal moves, move failures
with the expert's choice as the label, swarms of any size with tasks that fit them, the expert's
sub-goal recorded at every step, and the store that holds all of it."""

import json
import random

import numpy as np
import torch

from beebot.vla.voxel import controller, demos
from beebot.vla.voxel import suite as S
from beebot.vla.voxel.play_store import load
from beebot.vla.voxel.world import CUBE_MOVES, DIR_TABLE, MOVE_CLASS, MOVE_CLASSES, STAY, fail_moves, failure_rates, world_from_arrays


def _diagonal(codes):
    """Whether each action code moves along both horizontal axes."""
    step = DIR_TABLE[codes.long()]
    return (step[..., 1] != 0) & (step[..., 2] != 0)


def test_cube_moves_are_the_fourteen_without_a_horizontal_diagonal():
    assert int(CUBE_MOVES[:STAY].sum()) == 14 and bool(CUBE_MOVES[STAY])
    assert not _diagonal(CUBE_MOVES.nonzero().squeeze(1)).any()
    assert _diagonal((~CUBE_MOVES).nonzero().squeeze(1)).all()


def test_jittered_moves_respect_the_allowed_mask():
    rng = np.random.default_rng(0)
    world, _ = S.scenes(rng, 4)
    n = int(world.mass.sum())
    targets = torch.full((4, 1, 3), 8)
    codes = controller.act(world, targets, torch.ones(4, 1, dtype=torch.bool), jitter=1.0, generator=torch.Generator().manual_seed(0), allowed=CUBE_MOVES.unsqueeze(0).expand(n, -1))
    chosen = codes[world.mass > 0.5]
    assert not _diagonal(chosen).any()
    assert len(chosen.unique()) > 10  # jitter still spreads over the allowed moves


def test_no_recorded_expert_action_is_a_diagonal():
    rng = np.random.default_rng(1)
    world, metas, tasks = S.sample_tasks(rng, 12, "train")
    _, arrays = S.run(world, metas, tasks, S.Expert(tasks, moves=CUBE_MOVES, seed=0), steps=30, seed=0, record=True)
    assert not _diagonal(arrays["actions"]).any()
    assert (arrays["actions"] != STAY).any()
    _, free = S.run(world, metas, tasks, S.Expert(tasks, seed=0), steps=30, seed=0, record=True)
    assert _diagonal(free["actions"]).any()  # the restriction is what removes them


def test_failure_rates_follow_the_move_classes():
    rates = failure_rates(0.15, up_over=0.4, vertical=0.0)
    assert rates[STAY] == 0
    for code in range(STAY):
        dz, dy, dx = DIR_TABLE[code].tolist()
        sideways = dy != 0 or dx != 0
        name = "level" if dz == 0 else "vertical" if not sideways else "up_over" if dz > 0 else "down_over"
        assert MOVE_CLASSES[int(MOVE_CLASS[code])] == name
        assert abs(float(rates[code]) - {"level": 0.15, "up_over": 0.4, "down_over": 0.15, "vertical": 0.0}[name]) < 1e-6
    codes = torch.arange(27, dtype=torch.int8).repeat(20000, 1)
    outcome = fail_moves(codes, rates, torch.Generator().manual_seed(0))
    drop_share = ((outcome == STAY) & (codes != STAY)).float().mean(dim=0)
    assert (drop_share - rates).abs().max() < 0.015
    assert ((outcome == codes) | (outcome == STAY)).all()


def test_failed_moves_do_not_happen_and_the_label_is_the_chosen_action():
    mass = np.zeros((16, 16, 16))
    mass[4:12, 2:14:2, 3] = 1  # 48 robots in the air, none in another's way along x
    world = world_from_arrays(mass, gravity=False)
    right = int(controller.direction_code(torch.tensor([0, 0, 1])))
    task = S.Task("formation", "", {"kind": "mask", "mask": mass > 0.5, "forbid": set()})

    def actor(w, t):
        return torch.where(w.mass > 0.5, right, STAY).to(torch.int8)

    steps = 10
    _, arrays = S.run(world, [None], [task], actor, steps=steps, seed=0, record=True, failure=failure_rates(0.25))
    x = torch.arange(16).view(1, 1, 16)
    distance = [float((arrays["mass"][0, t].float() * x).sum() - (arrays["mass"][0, 0].float() * x).sum()) for t in range(steps + 1)]
    moved_share = distance[-1] / (48 * steps)
    assert 0.65 < moved_share < 0.85  # about a quarter of the moves failed
    for t in range(steps):  # every robot's recorded action is the one it chose, moved or not
        here = arrays["mass"][0, t] > 0
        assert (arrays["actions"][0, t][here] == right).all()
        assert (arrays["actions"][0, t][~here] == STAY).all()
    _, all_fail = S.run(world, [None], [task], actor, steps=3, seed=0, record=True, failure=failure_rates(1.0))
    assert (all_fail["mass"][0, -1] == all_fail["mass"][0, 0]).all() and (all_fail["actions"][0][:, mass > 0.5] == right).all()
    _, no_fail = S.run(world, [None], [task], actor, steps=3, seed=0, record=True)
    assert float((no_fail["mass"][0, -1].float() * x).sum() - (no_fail["mass"][0, 0].float() * x).sum()) == 48 * 3


def test_score_bound_tracks_the_swarm_size():
    mass = np.zeros((16, 16, 16))
    mass[0, 0, :10] = 1
    world = world_from_arrays(mass, cheeses=[(0, 7, 7, 0), (0, 0, 14, 1)], gravity=True)
    red, blue = (c for _, c in S.present(world, 0))
    eat = lambda colour: S.Task("logic", "", {"kind": "eat", "eat": {colour}, "any_of": set(), "order": [], "forbid": set(), "form": "single"})  # noqa: E731
    assert S.eat_need(world, 0, 0) == 30 and S.eat_need(world, 0, 1) == float("inf")  # open floor; a corner
    assert S.score_bound(eat(red), world, 0, 29) == 0.0 and S.score_bound(eat(red), world, 0, 30) == 1.0
    assert S.score_bound(eat(blue), world, 0, 200) == 0.0
    assert not S.fits(eat(red), world, 0)  # ten robots
    nothing = S.Task("logic", "", {"kind": "eat", "eat": set(), "any_of": set(), "order": [], "forbid": {red, blue}, "form": "none"})
    assert S.fits(nothing, world, 0)
    tower = lambda n: S.Task("construction", "", {"kind": "tower", "height": n, "forbid": set()})  # noqa: E731
    assert not S.fits(tower(4), world, 0) and S.fits(tower(3), world, 0)  # 10 robots: three quarters of a 2 x 2 x 3 column
    mask = S.shape_mask("wall")  # 40 voxels
    shape = S.Task("formation", "", {"kind": "mask", "mask": mask, "forbid": set()})
    assert not S.fits(shape, world, 0) and S.score_bound(shape, world, 0, 13) < 0.5 <= S.score_bound(shape, world, 0, 14) and S.score_bound(shape, world, 0, 121) < 0.5
    herd = S.Task("herd", "", {"kind": "herd_region", "cheeses": [red], "region": "left", "forbid": {red, blue}})
    assert S.fits(herd, world, 0)
    pair = lambda *jobs: S.Task("parallel", "", {"kind": "parallel", "jobs": list(jobs), "forbid": set()})  # noqa: E731
    assert S.fits(pair(("region", "left"), ("region", "right")), world, 0)
    assert not S.fits(pair(("region", "left"), ("eat", red)), world, 0)
    assert S.score_bound(pair(("region", "left"), ("surround", red)), world, 0, 23) == 1.0 > S.score_bound(pair(("region", "left"), ("surround", red)), world, 0, 22)  # 3 robots (a tenth of the swarm) there, 20 on half of the 40-voxel shell


def test_small_swarms_get_only_tasks_they_can_do():
    rng = np.random.default_rng(3)
    world, metas, tasks = S.sample_tasks(rng, 40, "train", robots=(6, 14))
    world, metas, tasks = demos.contrast(rng, world, metas, tasks, check=True)
    counts = world.mass.sum(dim=(1, 2, 3)).long().tolist()
    assert min(counts) >= 6 and max(counts) <= 14 and len(set(counts)) > 5
    assert len(tasks) > 60  # scenes still get several instructions
    for i, task in enumerate(tasks):
        assert S.fits(task, world, i)
        assert task.family != "runner"
        assert all(seg["kind"] != "eat" and not (seg["kind"] == "union" and any(p["kind"] == "eat" for p in seg["parts"])) for seg in task.plan)
        if task.spec["kind"] == "tower":
            assert 4 * task.spec["height"] * 0.75 <= counts[i]
        if task.spec["kind"] == "mask":
            assert task.spec["mask"].sum() <= 3 * counts[i]
    assert {"herd", "construction", "logic"} <= {t.family for t in tasks}
    loose_world, _, loose = S.sample_tasks(np.random.default_rng(3), 40, "train", robots=(6, 14), check=False)
    assert not all(S.fits(t, loose_world, i) for i, t in enumerate(loose))  # the check is what removes them


def test_default_scenes_keep_the_density_range_and_large_swarms_all_families():
    world, _ = S.scenes(np.random.default_rng(0), 8)
    assert int(world.mass.sum(dim=(1, 2, 3)).min()) >= 40
    _, _, tasks = S.sample_tasks(np.random.default_rng(4), 60, "train", robots=(90, 115))
    assert {t.family for t in tasks} == set(S.FAMILIES)


def test_subgoals_line_up_with_segment_switches():
    rng = np.random.default_rng(5)
    world, metas, tasks = S.sample_tasks(rng, 16, "train", "logic")
    assert any(len(t.plan) > 1 for t in tasks)
    expert = S.Expert(tasks, moves=CUBE_MOVES, seed=0)
    trace = []

    def actor(w, t):
        codes = expert(w, t)
        trace.append(list(expert.index))
        return codes

    steps = 72
    _, arrays = S.run(world, metas, tasks, actor, steps=steps, seed=0, record=True, failure=failure_rates())
    n_switches = 0
    for i, task in enumerate(tasks):
        entries = expert.subgoals[i]
        assert entries[0]["t"] == 0
        assert all(a["t"] < b["t"] for a, b in zip(entries, entries[1:]))
        if not task.plan:
            assert entries == [{"t": 0, "index": -1, "goal": {"kind": "stay"}}]
            continue
        for t in range(steps):
            entry = S.subgoal_at(entries, t)
            seg = task.plan[trace[t][i]]
            alive = bool(arrays["cheese_alive"][i, t, seg["cheese"]])  # the state the action at t was chosen in
            if entry["goal"]["kind"] == "done":
                assert entry["index"] == len(task.plan) and trace[t][i] == len(task.plan) - 1 and not alive
            else:
                assert entry["index"] == trace[t][i]
                assert entry["goal"] == {"kind": "eat", "colour": S.COLOUR_NAMES[int(world.cheese_color[i, seg["cheese"]])]}
        for before, after in zip(entries, entries[1:]):  # a switch happens the step the cheese is gone, or on the segment's time limit
            n_switches += 1
            cheese = task.plan[before["index"]]["cheese"]
            t = after["t"]
            eaten_now = bool(arrays["cheese_alive"][i, t - 1, cheese]) and not bool(arrays["cheese_alive"][i, t, cheese])
            timed_out = t - before["t"] == S.SEGMENT_STEPS and bool(arrays["cheese_alive"][i, t, cheese])
            assert eaten_now or timed_out
    assert n_switches > 5


def test_subgoals_name_the_target():
    rng = np.random.default_rng(6)
    for family in ("construction", "formation", "herd", "parallel"):
        world, metas, tasks = S.sample_tasks(rng, 8, "train", family)
        expert = S.Expert(tasks, seed=0)
        S.run(world, metas, tasks, expert, steps=30, seed=0)
        for i, task in enumerate(tasks):
            first = expert.subgoals[i][0]
            goal = first["goal"]
            assert first["t"] == 0 and first["index"] == 0
            json.dumps(expert.subgoals[i])  # plain data
            if task.spec["kind"] == "tower":
                assert goal["kind"] == "tower" and goal["height"] == task.spec["height"] and len(expert.subgoals[i]) == 1
            if task.spec["kind"] in ("bridge", "fill"):
                assert goal == {"kind": task.spec["kind"]}
            if family == "formation":
                assert goal["kind"] in ("letters", "shape") and (goal.get("word") or goal.get("shape")) and len(expert.subgoals[i]) == 1
            if task.spec["kind"] == "herd_region":
                assert goal["kind"] == "push" and goal["colour"] == task.spec["cheeses"][0] and goal["to"] == task.spec["region"]
                assert sum(abs(v) for v in goal["direction"]) == 1 and goal["direction"][0] == 0
                for a, b in zip(expert.subgoals[i], expert.subgoals[i][1:]):  # a new entry within the segment is a new push direction
                    assert b["goal"]["kind"] == "done" or (b["index"] == a["index"] and b["goal"]["direction"] != a["goal"]["direction"])
            if task.spec["kind"] == "herd_trench":
                assert goal["kind"] == "push" and goal["to"] == "trench"
            if family == "parallel":
                assert goal["kind"] == "union" and len(goal["parts"]) == 2
                target_names = sorted(str(part.get("region") or part.get("height") or part.get("colour")) for part in goal["parts"])
                assert target_names == sorted(str(x) for _, x in task.spec["jobs"])


def test_cube_store_round_trip(tmp_path):
    steps = 24
    count = demos.write(tmp_path, 6, seed=0, chunk=6, steps=steps, moves="cube", robots=(6, 115), failure=demos.parse_failure("0.15,up_over=0.3"))
    store = load(tmp_path)
    labels = [json.loads(line) for line in open(tmp_path / "labels.jsonl")]
    attempts = [json.loads(line) for line in open(tmp_path / "attempts.jsonl")]
    assert count == store["count"] == len(labels) == len(store["subgoals"]) == len(store["robots"]) > 0
    assert sorted(a["e"] for a in attempts if a["e"] >= 0) == list(range(count)) and len(attempts) >= count
    assert all(set(r) == {"e", "w0", "w1", "family", "form", "text"} and (r["w0"], r["w1"]) == (0, steps) for r in labels)
    assert store["mass"].shape[1:] == (steps + 1, 16, 16, 16) and store["actions"].shape[1:] == (steps, 16, 16, 16)
    actions = torch.from_numpy(np.asarray(store["actions"][:count]))
    assert not _diagonal(actions).any()
    assert (torch.from_numpy(np.asarray(store["mass"][:count, 0])).sum(dim=(1, 2, 3)) == store["robots"]).all()
    assert store["config"]["moves"] == "cube" and abs(store["config"]["failure"][int(controller.direction_code(torch.tensor([1, 0, 1])))] - 0.3) < 1e-6
    for e in range(count):
        assert store["subgoals"][e][0]["t"] == 0 and all(entry["t"] < steps for entry in store["subgoals"][e])
    from beebot.vla.voxel.lm import batch_of

    obs, codes, texts = batch_of(store, labels, 5, random.Random(0), "cpu")
    assert obs.shape == (5, 6, 16, 16, 16) and codes.shape == (5, 16, 16, 16) and len(texts) == 5
    again = tmp_path / "again"
    demos.write(again, 6, seed=0, chunk=6, steps=steps, moves="cube", robots=(6, 115), failure=demos.parse_failure("0.15,up_over=0.3"))
    assert (np.load(again / "actions.npy", mmap_mode="r")[:count] == np.asarray(store["actions"][:count])).all()  # deterministic under the seed
