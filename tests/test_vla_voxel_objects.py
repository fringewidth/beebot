"""Objects in the voxel world (vla/voxel/world.py, objects.py, README "Objects"): shapes, patterns,
weight and food as object fields with physics that reduces to the old one for 2 x 2 x 2 cheeses,
the observation's channel layout, floor zones, references, the four object families, their
records and wordings, the draw (pinned) and stores with objects."""

import hashlib
import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from beebot.vla.voxel import demos
from beebot.vla.voxel import objects as O
from beebot.vla.voxel import suite as S
from beebot.vla.voxel import wordings as W
from beebot.vla.voxel.world import BOX, CHANNELS, COLOUR_NAMES, COLOUR_RGB, CUBE_MOVES, FRICTION, HEAVY_FRICTION, MAX_OBJECTS, N_CHANNELS, OLD_CHANNELS, PATTERNS, STAY, VoxelWorld, concat, failure_rates, object_form, observe, paint_of, sample_object_scenes, sample_scenes, step, with_objects

OBJECT_DIGESTS = ("PLACEHOLDER_A", "PLACEHOLDER_B")  # draw_digest of test_the_object_draw_is_pinned


def one_object(form, pos, colour="red", colour2=None, pattern="plain", heavy=False, food=True, size=16, mass=None, gravity=False, zones=None):
    """A one-scene world with one object (the first slot) and robots `mass` (D, H, W)."""
    k = MAX_OBJECTS
    world = VoxelWorld(
        torch.zeros(1, size, size, size) if mass is None else torch.as_tensor(mass, dtype=torch.float32).unsqueeze(0),
        torch.zeros(1, size, size, size, dtype=torch.bool),
        torch.zeros(1, k, 3, dtype=torch.long),
        torch.zeros(1, k, dtype=torch.bool),
        torch.full((1, k), -1, dtype=torch.long),
        torch.tensor([gravity]),
    )
    world = with_objects(world)
    world.cheese_pos[0, 0] = torch.tensor(pos)
    world.cheese_alive[0, 0] = True
    world.cheese_color[0, 0] = COLOUR_NAMES.index(colour)
    world.form[0, 0] = form
    world.colour2[0, 0] = -1 if colour2 is None else COLOUR_NAMES.index(colour2)
    world.pattern[0, 0] = PATTERNS.index(pattern)
    world.friction[0, 0] = HEAVY_FRICTION if heavy else FRICTION
    world.food[0, 0] = food
    if zones is not None:
        world.zones[0] = torch.as_tensor(zones)
    return world


def press(world, cells, direction):
    """Codes (1, D, H, W): robots at cells move by direction, everyone else stays."""
    from beebot.vla.voxel.controller import direction_code

    codes = torch.full(world.mass.shape, STAY, dtype=torch.long)
    code = int(direction_code(torch.tensor(direction)))
    for cell in cells:
        codes[(0,) + tuple(cell)] = code
    return codes


def test_default_objects_step_and_observe_as_before():
    """A world with every object field at its default (2 x 2 x 2 light plain food) steps and is
    observed exactly as the world without them, with and without the pivot rule."""
    rng = np.random.default_rng(0)
    world, _ = sample_scenes(rng, 16, gravity_prob=0.5)
    full = with_objects(world, MAX_OBJECTS)
    generator = torch.Generator().manual_seed(1)
    pushes = 0
    for t in range(24):
        codes = torch.randint(0, 27, world.mass.shape, generator=generator)
        score = torch.rand(world.mass.shape, generator=generator)
        rule = "switch_cube" if t % 2 else None
        world, info = step(world, codes, score, rule)
        full, full_info = step(full, codes, score, rule)
        assert torch.equal(world.mass, full.mass) and torch.equal(world.cheese_pos, full.cheese_pos[:, :3]) and torch.equal(world.cheese_alive, full.cheese_alive[:, :3])
        assert torch.equal(info.pushed, full_info.pushed[:, :3]) and torch.equal(info.eaten, full_info.eaten[:, :3])
        assert torch.equal(observe(world), observe(full))
        pushes += int(info.pushed.sum())
    assert pushes > 0


def test_channel_layout():
    world, _ = sample_scenes(np.random.default_rng(2), 4)
    obs = observe(world)
    assert obs.shape[1] == N_CHANNELS == len(CHANNELS) == 10 and OLD_CHANNELS == 6
    assert CHANNELS[:OLD_CHANNELS] == ("robot", "wall", "object", "object red", "object green", "object blue")
    assert torch.equal(obs[:, 0], world.mass) and torch.equal(obs[:, 1], world.walls.float())
    assert torch.equal(obs[:, 6], obs[:, 2]) and not obs[:, 7:].any()  # old worlds: every cheese is food, no zones
    assert torch.equal(observe(world, OLD_CHANNELS), obs[:, :OLD_CHANNELS])


def test_shapes_patterns_and_colours_per_voxel():
    assert int(object_form("cube", 3).sum()) == 27 and int(object_form("slab", 4).sum()) == 16
    assert int(object_form("rod", 4, along="y").sum()) == 4 and object_form("rod", 4, along="y")[0, :, 0].all()
    assert int(object_form("L", 3, arm=4).sum()) == 3 + 4 - 1
    assert [O.shape_of(object_form(s, 3)) for s in ("cube", "slab", "rod", "L")] == ["cube", "slab", "rod", "L"]
    rod = object_form("rod", 4)
    striped = paint_of(rod.view(1, 1, BOX, BOX, BOX), torch.tensor([[PATTERNS.index("striped")]]))[0, 0, 0, 0]
    halves = paint_of(rod.view(1, 1, BOX, BOX, BOX), torch.tensor([[PATTERNS.index("two-tone")]]))[0, 0, 0, 0]
    assert striped.tolist() == [False, True, False, True] and halves.tolist() == [False, False, True, True]
    world = one_object(rod, (0, 5, 5), colour="red", colour2="blue", pattern="striped", food=False)
    obs = observe(world)
    assert torch.allclose(obs[0, 3:6, 0, 5, 5], COLOUR_RGB[COLOUR_NAMES.index("red")]) and torch.allclose(obs[0, 3:6, 0, 5, 6], COLOUR_RGB[COLOUR_NAMES.index("blue")])
    assert obs[0, 2].sum() == 4 and obs[0, 6].sum() == 0  # a block is an object but not food


def test_light_needs_two_pressers_and_heavy_six_with_robots_in_line():
    cube = object_form("cube", 2)
    face = [(0, 5, 4), (1, 5, 4), (0, 6, 4), (1, 6, 4)]  # the four voxels behind the cube's x face
    mass = np.zeros((16, 16, 16))
    for cell in face[:2]:
        mass[cell] = 1
    light = one_object(cube, (0, 5, 5), mass=mass)
    moved, info = step(light, press(light, face[:2], (0, 0, 1)), rule="switch_cube")
    assert info.pushed[0, 0] and moved.cheese_pos[0, 0].tolist() == [0, 5, 6]
    mass = np.zeros((16, 16, 16))
    mass[0, 5, 4] = mass[0, 5, 3] = 1  # one presser and one robot in line behind it
    light = one_object(cube, (0, 5, 5), mass=mass)
    _, info = step(light, press(light, [(0, 5, 4), (0, 5, 3)], (0, 0, 1)))
    assert not info.pushed[0, 0]  # a light object counts only robots moving into it
    mass = np.zeros((16, 16, 16))
    for cell in face:
        mass[cell] = 1
    heavy = one_object(cube, (0, 5, 5), heavy=True, mass=mass)
    _, info = step(heavy, press(heavy, face, (0, 0, 1)), rule="switch_cube")
    assert not info.pushed[0, 0]  # four on the face are not enough
    behind = [(0, 5, 3), (0, 6, 3)]
    for cell in behind:
        mass[cell] = 1
    heavy = one_object(cube, (0, 5, 5), heavy=True, mass=mass)
    moved, info = step(heavy, press(heavy, face + behind, (0, 0, 1)), rule="switch_cube")
    assert info.pushed[0, 0] and moved.cheese_pos[0, 0].tolist() == [0, 5, 6]


def test_any_shape_slides_falls_and_is_eaten_by_its_shell():
    ell = object_form("L", 3, corner=(0, 0), arm=3)  # row y=0 x=0..2 and column x=0 y=0..2
    behind = [(0, 4, 4), (0, 4, 5)]  # two robots behind the row, pressing toward +y
    mass = np.zeros((16, 16, 16))
    for cell in behind:
        mass[cell] = 1
    world = one_object(ell, (0, 5, 4), mass=mass)
    moved, info = step(world, press(world, behind, (0, 1, 0)))
    assert info.pushed[0, 0] and moved.cheese_pos[0, 0].tolist() == [0, 6, 4]
    floating = one_object(object_form("slab", 3), (4, 5, 5), gravity=True)
    fallen, _ = step(floating, torch.full(floating.mass.shape, STAY, dtype=torch.long))
    assert fallen.cheese_pos[0, 0, 0] == 3
    for side, need in ((1, 13), (2, 30), (3, 55), (4, 87)):  # three quarters of the open shell on the floor: 17, 40, 73, 116 voxels
        cube = object_form("cube", side)
        thing = O.things_of(one_object(cube, (0, 6, 6)), 0)[0]
        assert O.eat_need(thing, np.zeros((16, 16, 16), bool)) == need
        cells, _ = O.shell_cells(thing, np.zeros((16, 16, 16), bool))
        mass = np.zeros((16, 16, 16))
        mass[tuple(cells.T)] = 1
        for food in (True, False):
            world = one_object(cube, (0, 6, 6), mass=mass, food=food)
            _, info = step(world, torch.full(world.mass.shape, STAY, dtype=torch.long))
            assert bool(info.eaten[0, 0]) == food  # blocks are never eaten


def test_zones_are_observed_where_a_robot_stands_on_them():
    zones = -torch.ones(16, 16, dtype=torch.long)
    zones[2:5, 3:6] = COLOUR_NAMES.index("blue")
    world = one_object(object_form("cube", 1), (0, 10, 10), zones=zones)
    world.walls[0, 0, 2:5, 3:6] = True  # the zone lies on a raised patch: it shows one voxel up
    obs = observe(world)
    assert torch.allclose(obs[0, 7:10, 1, 3, 4], COLOUR_RGB[COLOUR_NAMES.index("blue")]) and obs[0, 7:10, 0].sum() == 0
    assert obs[0, 7:10].abs().sum(dim=0).gt(0).sum() == 9


def test_sampled_object_scenes_are_valid():
    world, metas = sample_object_scenes(np.random.default_rng(4), 12, robots=40)
    assert world.cheese_alive.shape[1] == MAX_OBJECTS and world.form.shape[2:] == (BOX,) * 3
    ids = __import__("beebot.vla.voxel.world", fromlist=["cheese_id_map"]).cheese_id_map(world)
    for i in range(12):
        assert not ((ids[i] >= 0) & (world.mass[i] > 0)).any() and not ((ids[i] >= 0) & world.walls[i]).any()
        things = O.things_of(world, i)
        for a in things:
            for b in things:
                if a["slot"] < b["slot"]:
                    assert O.cell_gap(a["cells"], b["cells"]) >= 2 or abs(a["pos"][0] - b["pos"][0]) > 0  # a voxel apart
        assert {z["colour"] for z in O.zone_list(world.zones[i].numpy())} == {COLOUR_NAMES[m["colour"]] for m in metas[i]["zones"]}


def scene(things_spec, zones=None, layout_meta=None):
    """A one-scene world from [(shape, size, pos, colour, food, heavy)] and its meta (with props)."""
    k = MAX_OBJECTS
    world = with_objects(VoxelWorld(torch.zeros(1, 16, 16, 16), torch.zeros(1, 16, 16, 16, dtype=torch.bool), torch.zeros(1, k, 3, dtype=torch.long), torch.zeros(1, k, dtype=torch.bool), torch.full((1, k), -1, dtype=torch.long), torch.tensor([True])))
    for slot, (shape, size, pos, colour, food, heavy) in enumerate(things_spec):
        world.cheese_pos[0, slot] = torch.tensor(pos)
        world.cheese_alive[0, slot] = True
        world.cheese_color[0, slot] = COLOUR_NAMES.index(colour)
        world.form[0, slot] = object_form(shape, size)
        world.food[0, slot] = food
        world.friction[0, slot] = HEAVY_FRICTION if heavy else FRICTION
    if zones is not None:
        world.zones[0] = torch.as_tensor(zones)
    meta = {"layout": "flat", "height": torch.zeros(16, 16, dtype=torch.long), "gap": None, "zones": [], **(layout_meta or {})}
    meta["props"] = O.props_of(world, 0)
    return world, meta


def test_references_pick_out_one_object():
    zones = -np.ones((16, 16), int)
    zones[12:15, 12:15] = COLOUR_NAMES.index("green")
    world, meta = scene([("cube", 1, (0, 2, 2), "blue", False, False), ("cube", 3, (0, 2, 8), "blue", False, True), ("slab", 2, (0, 9, 3), "red", True, False), ("cube", 2, (0, 12, 8), "yellow", True, False)], zones=zones)
    things = O.things_of(world, 0)
    ref = lambda **kw: O.resolve(kw, things, meta)  # noqa: E731
    assert ref(by="compare", noun="block", attrs={"colour": "blue"}, which="smaller") == 0
    assert ref(by="compare", noun="block", attrs={"colour": "blue"}, which="larger") == 1
    assert ref(by="super", noun="cheese", attrs={}, which="biggest") == 3  # 8 voxels against 4
    assert ref(by="attrs", noun="block", attrs={"colour": "blue"}) is None  # two of them
    assert ref(by="near", noun="cheese", attrs={}, which="nearest", anchor={"zone": "green", "kind": "zone"}) == 3
    assert ref(by="near", noun="cheese", attrs={}, which="furthest from", anchor={"zone": "green", "kind": "zone"}) == 2
    assert O.phrase({"by": "compare", "noun": "block", "attrs": {"colour": "blue"}, "which": "smaller"}) == "the smaller of the two blue blocks"
    assert O.phrase({"by": "next_to", "noun": "cheese", "attrs": {"size": "small"}, "anchor": {"by": "attrs", "noun": "block", "attrs": {"colour": "yellow"}}}) == "the small cheese next to the yellow block"
    attrs = O.describe(np.random.default_rng(0), things, 2)
    assert O.resolve({"by": "attrs", "noun": "cheese", "attrs": attrs}, things, meta) == 2 and len(attrs) == 1


@pytest.mark.parametrize("family", O.OBJECT_FAMILIES)
def test_object_tasks_have_records_wordings_and_summaries(family):
    rng = np.random.default_rng(31)
    world, metas, tasks = O.sample_objects(rng, 16, family=family, robots=S.ROBOTS, counts="log")
    choice = np.random.default_rng(32)
    for i, task in enumerate(tasks):
        assert task.family == family and task.plan and task.spec["objects"]
        record = S.canonical_record(task)
        assert set(record) == {"family", "kind", "plain", "spec", "subgoals"} and len(record["subgoals"]) == len(task.plan)
        json.dumps(record)
        summary = S.scene_summary(world, i, task)
        json.dumps(summary)
        ids = {o["id"] for o in summary["objects"]}
        spec = record["spec"]
        target = spec.get("target", spec.get("do", {}).get("target"))
        assert target is None or target in ids
        assert all(("weight" in o) == (o["id"] in O.told(task)) for o in summary["objects"])
        assert {c["colour"] for c in summary["cheeses"]} <= {o["colour"] for o in summary["objects"] if o["kind"] == "cheese"}
        assert W.canonical(W.parse_canonical(record["plain"])) == record["plain"]  # a bare colour phrase reads back as the colour intent of the same sentence
        slots = W.slots_of(task)
        for split in W.SPLITS:
            made = W.render(task.text, slots, choice, split)
            assert made.canonical == record["plain"] and "{" not in made.instruction
        assert O.score(task, S.static_of(world, metas[i], i), S.snapshot(world, i), S.snapshot(world, i), [])[0] < 0.5  # not done at the start


def test_held_out_property_words_stay_out_of_training_renders():
    rng = np.random.default_rng(33)
    _, _, tasks = O.sample_objects(rng, 40, robots=S.ROBOTS, counts="log")
    held = {text for group in W.PROPERTIES.values() for entry in group.values() for text in entry["held_out"]}
    choice = np.random.default_rng(34)
    words_used = set()
    for task in tasks:
        slots = W.slots_of(task)
        for _ in range(3):
            made = W.render(task.text, slots, choice, "train")
            assert not [h for h in held if f" {h} " in f" {made.instruction} "], made.instruction
        made = W.render(task.text, slots, choice, "words")
        words_used.update(h for h in held if f" {h} " in f" {made.instruction} ")
    assert words_used  # the words split reaches the held-out property words


def test_held_out_compositions_force_their_item_and_stay_out_of_training():
    rng = np.random.default_rng(35)
    for split, kinds in (("ref_job", ("push_thing:compare", "surround_thing:next_to")), ("condition", ("conditional:eat_goto",)), ("count", ("count:groups",))):
        _, _, tasks = O.sample_objects(rng, 3, split=split, check=False)
        assert all(t.held_out == split and not t.plan and any(S.task_kind(t).startswith(k) for k in kinds) for t in tasks), [S.task_kind(t) for t in tasks]
    _, _, train = O.sample_objects(rng, 40, family="reasoning")
    assert not any(S.task_kind(t).startswith(("push_thing:compare", "surround_thing:next_to", "conditional:eat_goto", "count:groups")) for t in train)


@pytest.mark.parametrize("family", O.OBJECT_FAMILIES)
def test_the_expert_passes_most_object_tasks(family):
    rng = np.random.default_rng(41)
    world, metas, tasks = O.sample_objects(rng, 16, family=family, robots=S.ROBOTS, counts="log")
    world = replace(world, gravity=torch.ones_like(world.gravity))
    results, _ = S.run(world, metas, tasks, O.ObjectExpert(tasks, moves=CUBE_MOVES, seed=0, rule="switch_cube"), steps=S.CUBE_STEPS, seed=0, failure=failure_rates(), rule="switch_cube")
    passed = [not u and s >= S.PASS_MARK for s, u in results]
    assert np.mean(passed) >= 0.5, [(S.task_kind(t), round(s, 2)) for t, (s, _) in zip(tasks, results)]


def test_the_heavy_one_is_found_by_trying():
    world, meta = scene([("cube", 2, (0, 4, 4), "red", False, False), ("cube", 2, (0, 4, 10), "red", False, True)])
    mass = np.zeros((16, 16, 16))
    mass[0, 9:12, 5:11] = 1
    world = replace(world, mass=torch.as_tensor(mass, dtype=torch.float32).unsqueeze(0))
    things = O.things_of(world, 0)
    task = O.job_task(np.random.default_rng(0), "blocks", "", world, meta, 0, {"region": "front left"}, {"by": "attrs", "noun": "block", "attrs": {"weight": "heavy"}}, 1, things, find=[0, 1])
    expert = O.ObjectExpert([task], moves=CUBE_MOVES, seed=0, rule="switch_cube")
    results, _ = S.run(world, [meta], [task], expert, steps=S.CUBE_STEPS, seed=0, rule="switch_cube")
    assert expert.probes[0]["found"] == 1 and expert.probes[0]["tried"][:2] == [0, 1]
    assert any(entry["goal"].get("found") == 1 for entry in expert.subgoals[0])


def draw_digest(seed, n_scenes, **kw):
    """Digest of demos.draws with objects: scenes, object fields, tasks, canonical sentences."""
    h = hashlib.sha256()
    count = 0
    for c, world, metas, tasks, run_seed in demos.draws(n_scenes, seed, 12, **kw):
        for f in ("mass", "walls", "cheese_pos", "cheese_alive", "cheese_color", "gravity", "form", "colour2", "pattern", "friction", "food", "zones"):
            h.update(getattr(world, f).numpy().tobytes())
        for t in tasks:
            h.update((t.family + "|" + t.text + "|" + S.task_kind(t) + "|" + S.canonical_record(t)["plain"]).encode())
            count += t.family in O.OBJECT_FAMILIES
    return h.hexdigest()[:16], count


def test_the_object_draw_is_pinned():
    """Haiku labels join to draws by (seed, index): what demos.draws --objects draws must not
    change; it depends on the makers, the sampler and the draw-time checks of objects.py."""
    assert draw_digest(3, 24, objects=0.5) == (OBJECT_DIGESTS[0], 12)
    assert draw_digest(4, 24, robots=S.ROBOTS, counts="log", check=True, rule="switch_cube", coverage=0.5, objects=0.5) == (OBJECT_DIGESTS[1], 12)


def test_objects_come_after_the_other_draws_and_leave_them_alone():
    plain = [[S.task_kind(t) for t in tasks] for _, _, _, tasks, _ in demos.draws(24, 5, 12, robots=S.ROBOTS, check=True, coverage=0.5)]
    mixed = [(w, [S.task_kind(t) for t in tasks]) for _, w, _, tasks, _ in demos.draws(24, 5, 12, robots=S.ROBOTS, check=True, coverage=0.5, objects=0.5)]
    for a, (world, b) in zip(plain, mixed):
        assert b[: len(a)] == a and len(b) == len(a) + 6 and world.form is not None and world.cheese_alive.shape[1] == MAX_OBJECTS


def test_a_store_with_objects_round_trips(tmp_path):
    from beebot.vla.voxel.lm import batch_of, world_at
    from beebot.vla.voxel.play_store import load

    count = demos.write(tmp_path, 6, seed=2, chunk=6, steps=30, moves="cube", robots=S.ROBOTS, compact=True, rule="switch_cube", objects=1.0)
    store = load(tmp_path)
    labels = [json.loads(line) for line in open(tmp_path / "labels.jsonl")]
    attempts = [json.loads(line) for line in open(tmp_path / "attempts.jsonl")]
    assert store["config"]["objects"] == 1.0 and count > 0 and any(a["family"] in O.OBJECT_FAMILIES for a in attempts)
    assert store["form"].shape == (count, MAX_OBJECTS, BOX, BOX, BOX) and store["cheese_pos"].shape[2] == MAX_OBJECTS
    kept = [a for a in attempts if a["e"] >= 0 and a["family"] in O.OBJECT_FAMILIES]
    if kept:
        world = world_at(store, [(kept[0]["e"], 0)])
        assert world.form is not None and world.zones is not None
    obs, codes, texts = batch_of(store, labels, 4, random.Random(0), "cpu")
    assert obs.shape == (4, N_CHANNELS, 16, 16, 16) and len(texts) == 4
