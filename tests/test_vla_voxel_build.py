"""The build family (vla/voxel/build.py) and its shape library (vla/voxel/shapes.py): parts
programs and their text form, the support and climbing rules, functions, the overlap score, the
draw, records and wordings, the expert, and the mesh interface."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from beebot.vla.voxel import build as B
from beebot.vla.voxel import demos
from beebot.vla.voxel import shapes as H
from beebot.vla.voxel import suite as S
from beebot.vla.voxel import wordings as W
from beebot.vla.voxel.world import fall, failure_rates

BUILD_DIGEST = "pinned below"  # draw_digest of test_the_build_draw_is_pinned


def draws_of(name, n=4, seed=0):
    rng = np.random.default_rng(seed)
    return [H.variant(name, rng) for _ in range(n)]


# ---------------------------------------------------------------------------------------------
# Parts programs


@pytest.mark.parametrize("name", sorted(H.LIBRARY))
def test_every_program_round_trips_through_its_text(name):
    for shape in draws_of(name, 3):
        text = shape.text()
        assert H.parse_program(text) == shape.parts
        assert (H.program_mask(text) == H.to_grid(shape.cells())).all()
        assert len(H.subgoal_sentences(shape.parts)) == len(H.groups(shape.parts))


def test_the_text_form_is_compact_and_refuses_what_it_cannot_read():
    stool = (H.Part("col", (0, 0, 0), (3, 1, 1), "legs"), H.Part("col", (0, 0, 2), (3, 1, 1), "legs"), H.Part("col", (0, 2, 0), (3, 1, 1), "legs"), H.Part("col", (0, 2, 2), (3, 1, 1), "legs"), H.Part("slab", (3, 0, 0), (1, 3, 3), "seat"))
    text = H.program_text(stool)
    assert text == "legs: col 0 0 0 3 1 1, col 0 0 2 3 1 1, col 0 2 0 3 1 1, col 0 2 2 3 1 1; seat: slab 3 0 0 1 3 3"
    assert H.subgoal_sentences(stool) == ["four legs 3 tall", "a 3 by 3 seat on top"]
    assert H.to_grid(H.program_cells(stool)).sum() == 4 * 3 + 9
    for bad in ("legs col 0 0 0 3 1 1", "legs: tower 0 0 0 3 1 1", "legs: col 0 0 3 1 1", "letter: glyph 0 0 0 7 1 5 +x", ""):
        with pytest.raises(ValueError):
            H.parse_program(bad)
    written = "back: wall 0 0 0 3 1 4; seat: slab 0 1 0 1 3 4"  # a new object written as parts
    assert H.program_mask(written).sum() == 12 + 12 - 0


def test_primitives_render_inside_their_box():
    for prim in H.PRIMS:
        part = H.Part(prim, (0, 0, 0), (4, 5, 5), "p", "+x" if prim in ("ramp", "glyph") else "x" if prim in ("arch", "roof") else "", "A" if prim == "glyph" else "")
        if prim == "glyph":
            part = replace(part, size=H.glyph_size("A"))
        grid = H.part_grid(part)
        assert grid.shape == part.size and grid.any(), prim
    arch = H.part_grid(H.Part("arch", (0, 0, 0), (3, 1, 5), "a", "x"))
    assert arch[2].all() and not arch[:2, 0, 1:4].any() and arch[:, 0, 0].all() and arch[:, 0, 4].all()
    ramp = H.part_grid(H.Part("ramp", (0, 0, 0), (4, 1, 4), "r", "+x"))
    assert ramp.sum(axis=0)[0].tolist() == [1, 2, 3, 4]
    ring = H.part_grid(H.Part("ring", (0, 0, 0), (2, 4, 4), "r"))
    assert ring.sum() == 2 * 12 and not ring[:, 1:3, 1:3].any()


@pytest.mark.parametrize("k", [1, 2, 3])
def test_turning_a_program_turns_its_voxels(k):
    for name in ("chair", "slide", "letter J", "house", "igloo"):
        shape = draws_of(name, 1, seed=k)[0]
        grid = H.to_grid(shape.cells())
        turned = H.to_grid(H.program_cells(H.rotate(shape.parts, k)))
        assert (turned == np.rot90(grid, -k, axes=(1, 2))).all(), name


# ---------------------------------------------------------------------------------------------
# Support and climbing


def test_a_bond_holds_an_overhang_up_to_the_cantilever_limit():
    column = [(0, 0, 0), (1, 0, 0)]
    arm = lambda n: column + [(1, 0, x) for x in range(1, n + 1)]  # noqa: E731
    assert H.stands(arm(3)) and not H.stands(arm(4))
    assert H.levers(arm(3))[(1, 0, 3)] == 3
    assert H.stands(arm(4), limit=4)
    lamp = arm(2) + [(0, 0, 2)]  # hanging under the arm's end
    assert H.levers(lamp)[(0, 0, 2)] == 0  # it rests on the floor: grounded
    hanging = column + [(2, 0, 0), (2, 0, 1), (2, 0, 2), (1, 0, 2)]
    assert H.levers(hanging)[(1, 0, 2)] == 3 and not H.stands(hanging, limit=2)
    span = [(0, 0, 0), (1, 0, 0)] + [(1, 0, x) for x in range(1, 7)] + [(0, 0, 7), (1, 0, 7)]
    assert H.stands(span)  # 6 cells between two columns: never more than 3 from one
    assert not H.stands([(0, 0, 0), (2, 0, 0)])  # a gap


def test_the_build_order_keeps_every_partial_structure_standing():
    for name in ("table", "arch", "bridge", "house", "letter A"):
        for shape in draws_of(name, 4):
            groups = [cells for _, cells in H.group_cells(shape.parts)]
            order = H.build_order(groups)
            if order is None:
                continue
            placed = []
            for cell in [c for group in order for c in group]:
                placed.append(cell)
                assert H.stands(placed), (name, cell)
            assert sorted(placed) == sorted(map(tuple, shape.cells().tolist()))


def test_a_cube_climbs_a_column_but_not_under_an_overhang():
    structure = np.zeros((5, 7, 7), bool)
    structure[0:2, 3, 3] = True
    got = H.reach(structure)
    assert got[2, 3, 3] and got[1, 3, 4] and not got[3, 3, 3]
    mushroom = H.parse_program("stem: col 0 2 2 3 1 1; cap: disc 3 0 0 1 5 5")
    assert H.buildable(mushroom) and not H.climbable(mushroom)


def test_a_part_that_fills_a_slot_must_come_before_the_part_that_closes_it():
    sides_first = H.parse_program("left side: wall 0 0 0 4 1 1, wall 0 0 4 4 1 1; shelf: slab 2 0 1 1 1 3")
    shelf_first = H.parse_program("left side: wall 0 0 0 4 1 1; shelf: slab 2 0 1 1 1 3; right side: wall 0 0 4 4 1 1")
    assert H.buildable(sides_first) and not H.climbable(sides_first)
    assert H.climbable(shelf_first)


# ---------------------------------------------------------------------------------------------
# Functions


def on_grid(cells, at=(0, 5, 5)):
    grid = np.zeros((16, 16, 16), bool)
    grid[tuple((np.asarray(cells) + np.array(at)).T)] = True
    return grid


@pytest.mark.parametrize("use", sorted(H.USES))
def test_objects_for_a_use_pass_its_check_and_most_others_do_not(use):
    rng = np.random.default_rng(3)
    for name in [n for n in H.TRAIN_NAMES if use in H.LIBRARY[n].functions]:
        shapes = [s for s in (H.variant(name, rng) for _ in range(6)) if H.buildable(s.parts)]
        hits = [H.FUNCTIONS[use](on_grid(s.cells())) for s in shapes]
        assert np.mean([h >= H.FUNCTION_PASS for h in hits]) >= 0.8, (name, hits)
    assert H.FUNCTIONS[use](np.zeros((16, 16, 16), bool)) == 0.0
    flat = np.zeros((16, 16, 16), bool)
    flat[0, 4:9, 4:9] = True
    assert H.FUNCTIONS[use](flat) < H.FUNCTION_PASS  # a mat on the floor is good for nothing


def test_function_checks_on_simple_structures():
    stool = on_grid(H.program_cells(H.parse_program("legs: col 0 0 0 2 1 1, col 0 0 2 2 1 1, col 0 2 0 2 1 1, col 0 2 2 2 1 1; seat: slab 2 0 0 1 3 3")))
    assert H.sit(stool) == 1.0 and H.table(stool) == 1.0 and H.contain(stool) == 0.0
    cup = on_grid(H.program_cells(H.parse_program("base: slab 0 0 0 1 3 3; wall: ring 1 0 0 2 3 3")))
    assert H.contain(cup) == 1.0 and H.shelter(cup) == 0.0
    ring_on_floor = on_grid(H.program_cells(H.parse_program("wall: ring 0 0 0 2 3 3")))
    assert H.contain(ring_on_floor) == 0.0  # no bottom of its own
    arch = on_grid(H.program_cells(H.parse_program("arch: arch 0 0 0 3 1 5 x")))
    assert H.span(arch) == 1.0
    column = on_grid([(z, 0, 0) for z in range(5)])
    assert H.span(column) == 0.0 and H.sit(column) == 0.0


def test_scattered_swarms_are_rarely_anything():
    rng = np.random.default_rng(4)
    world, _ = S.scenes(rng, 24, layout="flat", gravity=True, cheeses=(0, 0))
    for _ in range(3):  # sampled stacks can start in the air; let them settle
        world = fall(world)
    for use, check in H.FUNCTIONS.items():
        hits = [check(world.mass[i].numpy() > 0.5) >= H.FUNCTION_PASS for i in range(24)]
        assert np.mean(hits) <= 0.25, use


# ---------------------------------------------------------------------------------------------
# The overlap score


def test_the_score_finds_the_object_wherever_and_however_it_was_built():
    chair = draws_of("chair", 1)[0].cells()
    for k, at in ((0, (0, 2, 3)), (1, (0, 8, 1)), (3, (0, 0, 0))):
        built = H.program_cells(H.rotate(draws_of("chair", 1)[0].parts, k))
        assert H.object_iou(on_grid(built, at), chair) == 1.0
    grid = on_grid(chair, (0, 3, 3))
    grid[0, 15, 15] = grid[0, 14, 0] = True  # idle robots far away cost nothing
    assert H.object_iou(grid, chair) == 1.0
    grid[0, 2, 3] = True  # one beside it does
    assert H.object_iou(grid, chair) == pytest.approx(len(chair) / (len(chair) + 1))
    assert H.object_iou(np.zeros((16, 16, 16), bool), chair) == 0.0
    half = on_grid(chair[: len(chair) // 2], (0, 3, 3))
    assert 0.3 < H.object_iou(half, chair) <= 0.55


# ---------------------------------------------------------------------------------------------
# The library and what is held out


def test_held_out_objects_are_built_from_trained_parts():
    trained = {p.prim for name in H.TRAIN_NAMES for s in draws_of(name, 6) for p in s.parts}
    for name in H.HELD_NAMES:
        used = {p.prim for s in draws_of(name, 6) for p in s.parts}
        assert used <= trained, (name, used - trained)
    assert all(H.LIBRARY[n].held_out for n in H.LIBRARY if H.LIBRARY[n].category in H.HELD_CATEGORIES)
    drawable = [n for n in H.LIBRARY if B.min_sizes()[n] <= S.ROBOTS[1]]
    held = [n for n in drawable if n in H.HELD_NAMES]
    assert 0.15 <= len(held) / len(drawable) <= 0.3
    assert {"pyramid", "staircase", "letter K", "letter Q"} <= set(H.HELD_NAMES)  # the suite's held-out shape, construction and letters stay held out


def test_the_library_spans_the_swarm_sizes_and_most_of_it_can_be_built():
    sizes = B.min_sizes()
    drawable = [n for n in H.TRAIN_NAMES if sizes[n] <= S.ROBOTS[1]]
    assert len(drawable) >= 40
    assert min(sizes[n] for n in drawable) <= 8 and len([n for n in drawable if sizes[n] <= 15]) >= 10
    categories = {H.LIBRARY[n].category for n in drawable}
    assert {"furniture", "structure", "building", "container", "symbol", "letter", "number", "animal"} <= categories
    rng = np.random.default_rng(5)
    for robots in (8, 20, 60, 115):
        for name in [n for n in drawable if sizes[n] <= robots][:12]:
            shape = H.fit_variant(name, rng, robots)
            if shape is not None:
                assert len(shape.cells()) <= robots and H.fits_grid(shape.cells()) and H.climbable(shape.parts)


# ---------------------------------------------------------------------------------------------
# The family: draws, records, wordings


def draw_digest(seed, n, **kw):
    rng = np.random.default_rng(seed)
    world, metas, tasks = B.sample_build(rng, n, **kw)
    h = hashlib.sha256()
    for f in B.FIELDS:
        h.update(getattr(world, f).numpy().tobytes())
    for task in tasks:
        h.update((task.text + "|" + S.task_kind(task) + "|" + task.spec["program"] + "|" + json.dumps(task.spec["at"])).encode())
    return h.hexdigest()[:16]


def test_build_tasks_have_records_that_parse_back():
    rng = np.random.default_rng(6)
    world, metas, tasks = B.sample_build(rng, 16, robots=S.ROBOTS, counts="log")
    assert any(t.spec["use"] for t in tasks) and any(not t.spec["use"] for t in tasks)
    for i, task in enumerate(tasks):
        record = S.canonical_record(task)
        json.dumps(record)
        spec = record["spec"]
        assert spec["kind"] == "build" and record["family"] == "build"
        slots = W.parse_canonical(record["plain"])
        if task.spec["use"]:
            assert slots == {"intent": "build.use", "f": task.spec["use"]} and spec["use"] == task.spec["use"] and "object" not in spec
            assert task.spec["object"] in H.TRAIN_NAMES and task.spec["use"] in H.LIBRARY[task.spec["object"]].functions
        else:
            assert slots == {"intent": "build", "o": task.spec["object"]} and spec["object"] == task.spec["object"]
        assert H.parse_program(spec["program"]) and spec["voxels"] == len(task.spec["cells"]) <= int(world.mass[i].sum())
        assert record["subgoals"] == H.subgoal_sentences(H.parse_program(spec["program"])) and len(record["subgoals"]) == len(task.plan)
        assert task.plan[-1]["points"] and sorted(task.plan[-1]["points"]) == sorted(map(tuple, task.spec["cells"].tolist()))
        assert H.object_iou(S.snapshot(world, i)["mass"] > 0.5, task.spec["cells"]) < 0.5
        summary = S.scene_summary(world, i)
        assert summary["gravity"] and summary["terrain"] == {"layout": "flat floor"}


def test_the_verifier_ignores_the_plan_in_a_build_spec():
    from beebot.vla.voxel import haiku

    task = B.sample_build(np.random.default_rng(7), 1)[2][0]
    spec = S.canonical_record(task)["spec"]
    read_back = {k: v for k, v in spec.items() if k not in ("program", "voxels")}
    assert haiku.same_spec(read_back, spec)
    menu = haiku.menu_from([S.canonical_record(task)])
    assert '"program":' not in menu.split("Example")[0] and '"build"' in menu


def test_held_out_and_wording_splits():
    rng = np.random.default_rng(8)
    _, _, tasks = B.sample_build(rng, 8, split="objects", robots=S.ROBOTS)
    assert all(t.spec["object"] in H.HELD_NAMES and not t.plan and t.held_out == "objects" for t in tasks)
    _, _, tasks = B.sample_build(rng, 8, split="wording")
    assert all(t.text.startswith("construct") and t.plan for t in tasks)
    _, _, tasks = B.sample_build(rng, 12)
    assert all(t.spec["object"] in H.TRAIN_NAMES and not t.text.startswith("construct") for t in tasks)


def test_build_wordings_round_trip_in_every_split():
    rng = np.random.default_rng(9)
    world, metas, tasks = B.sample_build(rng, 12, robots=S.ROBOTS, counts="log")
    choice = np.random.default_rng(10)
    for i, task in enumerate(tasks):
        slots = W.slots_of(task, [c for _, c in S.present(world, i)])
        for split in W.SPLITS:
            made = W.render(task.text, slots, choice, split)
            assert W.parse_canonical(made.canonical)["intent"] == slots["intent"]
            assert "{" not in made.instruction
            if split == "wording":
                assert made.template in W.INTENTS[slots["intent"]]["held_out"]


def test_the_build_draw_is_pinned():
    """Labels join to draws by (seed, index): what sample_build draws must not change."""
    assert draw_digest(11, 6, robots=S.ROBOTS, counts="log") == BUILD_DIGEST


def test_build_tasks_come_after_the_other_draws_and_leave_them_alone():
    plain = [[S.task_kind(t) for t in tasks] for _, _, _, tasks, _ in demos.draws(12, 5, 12, robots=S.ROBOTS, check=True, coverage=0.5)]
    mixed = [[S.task_kind(t) for t in tasks] for _, _, _, tasks, _ in demos.draws(12, 5, 12, robots=S.ROBOTS, check=True, coverage=0.5, build=0.25)]
    for a, b in zip(plain, mixed):
        assert b[: len(a)] == a and len(b) == len(a) + 3 and all(kind.startswith("build:") for kind in b[len(a) :])


# ---------------------------------------------------------------------------------------------
# The expert


def test_the_expert_builds_a_stool_part_by_part_under_switch_cube():
    rng = np.random.default_rng(12)
    world, metas, tasks = B.sample_build(rng, 2, robots=(20, 30), name="stool", demonstrate=True)
    world = replace(world, gravity=torch.ones_like(world.gravity))
    expert = B.make_expert(tasks, seed=0, rule="switch_cube")
    steps = max(B.episode_steps(t) for t in tasks)
    results, _ = S.run(world, metas, tasks, expert, steps=steps, seed=0, failure=failure_rates(0.15), rule="switch_cube")
    for (value, unasked), task, entries in zip(results, tasks, expert.subgoals):
        assert B.passed(task, value, unasked), value
        parts = [e["goal"]["part"] for e in entries]
        assert parts[0] == "legs" and parts[-1] == "seat" and sorted(set(parts)) == ["legs", "seat"]
        assert all(e["goal"]["kind"] == "build" for e in entries)


def test_a_mixed_batch_plays_the_other_tasks_as_without_builds():
    rng = np.random.default_rng(13)
    world, metas, tasks = S.sample_coverage(rng, 3, robots=S.ROBOTS, counts="log")
    more, more_metas, more_tasks = B.sample_build(rng, 2, robots=S.ROBOTS, counts="log")
    both = S.VoxelWorld(*(torch.cat([getattr(world, f), getattr(more, f)]) for f in B.FIELDS))
    both = replace(both, gravity=torch.ones_like(both.gravity))
    alone_world = replace(world, gravity=torch.ones_like(world.gravity))
    make_other = lambda some: S.Expert(some, moves=B.CUBE_MOVES, seed=3, rule="switch_cube")  # noqa: E731
    alone, alone_arrays = S.run(alone_world, metas, tasks, make_other(tasks), steps=12, seed=3, record=True, rule="switch_cube")
    player = B.Mixed(tasks + more_tasks, make_other, lambda some: B.make_expert(some, seed=3))
    mixed, mixed_arrays = S.run(both, metas + more_metas, tasks + more_tasks, player, steps=12, seed=3, record=True, rule="switch_cube")
    assert (mixed_arrays["actions"][:3] == alone_arrays["actions"]).all()
    assert len(player.subgoals) == 5 and all(player.subgoals)


def test_demos_write_build_episodes(tmp_path):
    from beebot.vla.voxel.play_store import instructions_of, load

    count = demos.write(tmp_path, 4, seed=2, chunk=4, steps=40, moves="cube", robots=S.ROBOTS, compact=True, rule="switch_cube", build=1.0)
    attempts = [json.loads(line) for line in open(tmp_path / "attempts.jsonl")]
    assert sum(a["kind"].startswith("build:") for a in attempts) == 4
    store = load(tmp_path)
    assert store["config"]["build"] == 1.0 and len(store["instructions"]) == count
    for e in range(count):
        if instructions_of(store, e)[0]["record"]["family"] == "build":
            assert all(entry["goal"]["kind"] == "build" for entry in store["subgoals"][e])


# ---------------------------------------------------------------------------------------------
# Meshes


def box_mesh(h, d, w):
    """A closed box mesh (vertices, triangles), y up as in ModelNet."""
    corners = np.array([[x, y, z] for x in (0, w) for y in (0, h) for z in (0, d)], float)
    faces = []
    for axis in range(3):
        for side in (0, 1):
            quad = [k for k in range(8) if (corners[k, axis] > 0) == bool(side)]
            a, b, c, e = quad
            faces += [(a, b, e), (a, e, c)]
    return corners, np.array(faces)


def test_meshes_are_voxelised_hollowed_and_kept_only_if_they_build(tmp_path):
    vertices, faces = box_mesh(3.0, 4.0, 6.0)
    solid = H.voxelise(vertices, faces, 6, up=1, fill=True)
    assert solid.all() and solid.shape[0] < solid.shape[2]
    shape = H.mesh_shape(*box_mesh(2.0, 3.0, 4.0), "brick", "test", robots=60)
    assert shape is not None and len(shape.cells()) <= 60 and H.buildable(shape.parts)
    assert H.parse_program(shape.text()) == shape.parts
    tiny = H.mesh_shape(vertices, faces, "brick", "test", robots=5)
    assert tiny is None or len(tiny.cells()) <= 5
    off = tmp_path / "box.off"
    off.write_text("OFF\n8 12 0\n" + "\n".join(" ".join(map(str, v)) for v in vertices) + "\n" + "\n".join("3 " + " ".join(map(str, f)) for f in faces) + "\n")
    v2, f2 = H.read_off(off)
    assert np.allclose(v2, vertices) and (f2 == faces).all()
    obj = tmp_path / "box.obj"
    obj.write_text("\n".join("v " + " ".join(map(str, v)) for v in vertices) + "\n" + "\n".join("f " + " ".join(str(k + 1) for k in f) for f in faces) + "\n")
    v3, f3 = H.read_obj(obj)
    assert np.allclose(v3, vertices) and (f3 == faces).all()
    parts = H.boxes_of(solid)
    assert (H.to_grid(H.program_cells(parts), solid.shape) == solid).all() and len(parts) == 1
