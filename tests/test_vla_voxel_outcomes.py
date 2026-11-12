"""Move outcomes measured in MuJoCo, in the voxel world's step (world.OutcomeModel): each pivot
reaches, stays, or ends as a MuJoCo trial of its class ended, replayed in the move's frame; a cube
left off the lattice is stuck."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from beebot.vla.voxel import suite as S
from beebot.vla.voxel.world import DIRS, N_CLASSES, OUTCOME_MODELS, PRIMITIVES, STAY, OutcomeModel, class_name, concat, pivot_class, pivot_features, rule_moves, rule_table, step, world_from_arrays


def code(dz, dy, dx):
    return DIRS.index((dz, dy, dx))


def scene(cells, walls=(), cheeses=()):
    mass = np.zeros((16, 16, 16))
    for c in cells:
        mass[c] = 1
    solid = np.zeros((16, 16, 16), bool)
    for c in walls:
        solid[c] = True
    return world_from_arrays(mass, solid, cheeses, gravity=True)


def table(**classes):
    """A model from {key: (reached, stayed, [aftermath, ...])}."""
    return OutcomeModel({"name": "test", "classes": {key.replace("__", "/"): {"reached": r, "stayed": s, "harm": h} for key, (r, s, h) in classes.items()}})


def one_move(world, cell, move):
    codes = torch.full(world.mass.shape, STAY)
    codes[0][cell] = code(*move)
    return codes


def test_classes_name_the_pivot_and_what_holds_it():
    world = scene([(0, 5, 5), (0, 5, 6), (0, 6, 4)])
    _, side = rule_table(world, "switch_cube", sides=True)
    here = torch.tensor([[0, 5, 5]])
    c = code(0, 0, -1)  # rolls away from its face neighbour, past the cube touching it by an edge
    features = pivot_features(world, torch.tensor([0]), here, torch.tensor([c]), side[0, c, 0, 5, 5].view(1))
    index = int(pivot_class(features)[0])
    assert class_name(index) == "floor_roll/bonded/hinged/rubs"
    lone = scene([(0, 5, 5), (0, 5, 6)])
    _, side = rule_table(lone, "switch_cube", sides=True)
    c = code(1, 0, 1)
    features = pivot_features(lone, torch.tensor([0]), here, torch.tensor([c]), side[0, c, 0, 5, 5].view(1))
    assert class_name(int(pivot_class(features)[0])) == "convex_up/lone"
    assert N_CLASSES == len(PRIMITIVES) * 16


def test_without_a_model_step_is_unchanged():
    world = scene([(0, 5, 5), (0, 9, 9)])
    codes = one_move(world, (0, 9, 9), (0, 1, 0))
    plain, _ = step(world, codes, rule="switch_cube")
    assert plain.stuck is None and plain.mass[0, 0, 10, 9] == 1


def test_outcome_shares_follow_the_table():
    model = table(floor_roll=(70, 30, []))
    world = concat([scene([(0, 5, 5)])] * 400)
    codes = torch.full(world.mass.shape, STAY)
    codes[:, 0, 5, 5] = code(0, 0, 1)
    after, _ = step(world, codes, rule="switch_cube", outcomes=model, noise=torch.rand(world.mass.shape, generator=torch.Generator().manual_seed(0)))
    reached = float(after.mass[:, 0, 5, 6].mean())
    assert abs(reached - 0.7) < 0.07
    assert (after.mass.sum(dim=(1, 2, 3)) == 1).all()


def test_a_mover_left_off_the_lattice_is_stuck_for_good():
    model = table(floor_roll=(0, 0, [[[[0, 0, 0], [1, 0, 0], True]]]))  # ends one step along h, off the lattice
    world = scene([(0, 5, 5)])
    after, _ = step(world, one_move(world, (0, 5, 5), (0, 0, 1)), rule="switch_cube", outcomes=model)
    assert after.mass[0, 0, 5, 6] == 1 and bool(after.stuck[0, 0, 5, 6])
    assert bool(rule_moves(after, "switch_cube")[0, STAY]) and not rule_moves(after, "switch_cube")[0, :STAY].any()
    again, _ = step(after, one_move(after, (0, 5, 6), (0, 1, 0)), rule="switch_cube", outcomes=table(floor_roll=(100, 0, [])))
    assert again.mass[0, 0, 5, 6] == 1 and bool(again.stuck[0, 0, 5, 6])


def test_a_knock_is_replayed_in_the_moves_frame():
    # the mover stays (aligned); its support (one cell along s) is shoved a cell further along s, off the lattice
    knock = [[[0, 0, 0], [0, 0, 0], False], [[0, 1, 0], [0, 2, 0], True]]
    model = table(convex_up__lone=(0, 0, [knock]))
    for move, support, shoved in (((1, 0, 1), (0, 5, 6), (0, 5, 7)), ((1, -1, 0), (0, 4, 5), (0, 3, 5))):
        world = scene([(0, 5, 5), support])
        after, _ = step(world, one_move(world, (0, 5, 5), move), rule="switch_cube", outcomes=model)
        assert after.mass[0, 0, 5, 5] == 1 and after.mass[0][support] == 0 and after.mass[0][shoved] == 1
        assert bool(after.stuck[0][shoved]) and not bool(after.stuck[0, 0, 5, 5])


def test_an_aftermath_into_a_taken_cell_leaves_the_cube_where_it_was():
    model = table(floor_roll=(0, 0, [[[[0, 0, 0], [2, 0, 0], False]]]))  # strays two cells along h
    world = scene([(0, 5, 5), (0, 5, 7)])
    after, _ = step(world, one_move(world, (0, 5, 5), (0, 0, 1)), rule="switch_cube", outcomes=model)
    assert after.mass[0, 0, 5, 5] == 1 and after.mass[0, 0, 5, 7] == 1 and after.mass.sum() == 2


def test_stuck_cubes_fall_with_their_column_and_block_a_cheese():
    model = table(convex_down=(0, 0, [[[[0, 0, 0], [0, 0, 0], True]]]))  # off where it stood
    tower = scene([(0, 5, 5), (1, 5, 5)])
    after, _ = step(tower, one_move(tower, (1, 5, 5), (-1, 0, 1)), rule="switch_cube", outcomes=model)
    assert bool(after.stuck[0, 1, 5, 5])
    gone = after.mass.clone()
    gone[0, 0, 5, 5] = 0
    fallen, _ = step(replace(after, mass=gone), torch.full(gone.shape, STAY), rule="switch_cube")
    assert fallen.mass[0, 0, 5, 5] == 1 and bool(fallen.stuck[0, 0, 5, 5]) and not fallen.stuck[0, 1].any()
    for blocking in (False, True):
        pressed = scene([(0, 5, 4), (0, 6, 4), (0, 5, 7)], cheeses=[(0, 5, 5, 0)])  # two press a cheese toward a cube in its way
        pressed.stuck = torch.zeros(pressed.mass.shape, dtype=torch.bool)
        pressed.stuck[0, 0, 5, 7] = blocking
        codes = torch.full(pressed.mass.shape, STAY)
        codes[0, 0, 5, 4] = codes[0, 0, 6, 4] = code(0, 0, 1)
        pushed, info = step(pressed, codes, rule="switch_cube")
        assert bool(info.pushed[0, 0]) != blocking  # a robot is shoved along; a stuck one stops the cheese
        assert pushed.mass[0, 0, 5, 7 if blocking else 8] == 1


@pytest.mark.parametrize("name", OUTCOME_MODELS)
def test_measured_tables_load_and_cover_every_class(name):
    model = OutcomeModel.load(name)
    assert model.name == name
    floor = [i for i in range(N_CLASSES) if class_name(i) == "floor_roll"][0]
    assert model.source[floor] is not None
    reach, stay, harm = model.probabilities(floor)
    assert abs(reach + stay + harm - 1) < 1e-6
    assert ((model.reach + model.stay) <= 1 + 1e-6).all()


def test_a_store_drawn_with_a_model_records_it(tmp_path):
    from beebot.vla.voxel import demos as D
    from beebot.vla.voxel.play_store import load

    n = D.write(tmp_path, 2, seed=1, chunk=2, steps=12, moves="cube", robots=S.ROBOTS, compact=True, counts="log", rule="switch_cube", outcomes="mujoco_release")
    store = load(tmp_path)
    assert store["config"]["outcomes"] == "mujoco_release" and store["config"]["failure"] is None
    assert len(store["stuck"]) == n
    for e, cells in enumerate(store["stuck"]):
        for t, z, y, x in cells:
            assert store["mass"][e, t][z, y, x] == 1
    with pytest.raises(ValueError):
        D.write(tmp_path / "both", 1, failure=torch.zeros(27), rule="switch_cube", outcomes="mujoco_release")


def test_run_with_a_model_records_stuck_robots_and_refuses_failure_rates():
    rng = np.random.default_rng(3)
    world, metas, tasks = S.sample_tasks(rng, 4, "train", robots=(10, 30))
    world = replace(world, gravity=torch.ones_like(world.gravity))
    model = table(floor_roll=(50, 0, [[[[0, 0, 0], [0, 0, 0], True]]]))
    expert = S.Expert(tasks, seed=0, rule="switch_cube")
    _, arrays = S.run(world, metas, tasks, expert, steps=6, seed=0, record=True, rule="switch_cube", outcomes=model)
    assert arrays["stuck"].shape == arrays["mass"].shape
    assert (arrays["stuck"] <= (arrays["mass"] > 0)).all()
    with pytest.raises(ValueError):
        S.run(world, metas, tasks, expert, steps=1, failure=torch.zeros(27), rule="switch_cube", outcomes=model)


def test_odds_give_each_move_its_class_odds():
    model = table(concave_up__lone=(0, 0, [[[[0, 0, 0], [0, 0, 0], True]]] * 10), floor_roll=(9, 1, []))
    world = scene([(0, 5, 5), (0, 5, 6), (1, 5, 6)])  # a cube beside a lone stack of two
    odds = model.odds(world, "switch_cube")
    robot = 0  # grid order: (0, 5, 5) first
    up = code(1, 0, 0)
    assert odds.shape == (3, 27, 3)
    assert torch.allclose(odds[robot, up], torch.tensor(model.probabilities(model_index(world, (0, 5, 5), up))))
    assert odds[robot, up, 2] > 0.5 and odds[robot, code(0, 1, 0), 2] == 0
    assert odds[robot, STAY].tolist() == [0.0, 1.0, 0.0]
    assert odds[robot, code(0, 0, 1)].tolist() == [0.0, 1.0, 0.0]  # into the stack: not a move the rule allows


def model_index(world, cell, move):
    _, side = rule_table(world, "switch_cube", sides=True)
    features = pivot_features(world, torch.tensor([0]), torch.tensor([cell]), torch.tensor([move]), side[0, move][cell].view(1))
    return int(pivot_class(features)[0])


def test_risky_leaves_out_moves_a_safer_one_can_replace():
    from beebot.vla.voxel.controller import risky

    odds = torch.zeros(1, 27, 3)
    odds[..., 1] = 1.0
    a, b, c = code(0, 0, 1), code(0, 1, 0), code(1, 0, 1)
    odds[0, a] = torch.tensor([0.3, 0.6, 0.1])
    odds[0, b] = torch.tensor([0.5, 0.5, 0.0])
    odds[0, c] = torch.tensor([0.2, 0.0, 0.8])
    progress = torch.zeros(1, 27, dtype=torch.bool)
    progress[0, [a, b, c]] = True
    out = risky(odds, progress)
    assert out[0, a] and not out[0, b] and out[0, c]  # a: 0.1 above b's 0; c: more likely to harm than land
    progress[0, b] = False  # b no longer gets closer: a is the safest move that does
    out = risky(odds, progress)
    assert not out[0, a] and out[0, c] and not out[0, STAY]


def test_the_controller_waits_rather_than_take_a_move_likely_to_harm():
    from beebot.vla.voxel.controller import act

    world = scene([(0, 5, 5)])
    targets = torch.tensor([[[0, 5, 9]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    allowed = rule_moves(world, "switch_cube")
    east = code(0, 0, 1)
    for harm, expected in ((0.0, east), (0.1, east), (0.7, STAY)):
        odds = torch.zeros(1, 27, 3)
        odds[..., 0] = 1.0  # every other move lands
        odds[0, STAY] = torch.tensor([0.0, 1.0, 0.0])
        odds[0, east] = torch.tensor([0.3, 0.7 - harm, harm]) if harm < 0.7 else torch.tensor([0.3, 0.0, 0.7])
        got = act(world, targets, valid, jitter=0.0, allowed=allowed, odds=odds)[0, 0, 5, 5]
        assert int(got) == expected


def test_a_risk_weighing_expert_never_takes_a_move_more_likely_to_harm_than_land():
    rng = np.random.default_rng(5)
    world, metas, tasks = S.sample_tasks(rng, 6, "train", robots=(8, 12))
    world = replace(world, gravity=torch.ones_like(world.gravity))
    model = OutcomeModel.load(OUTCOME_MODELS[-1])
    blind, weighing = (S.Expert(tasks, seed=0, rule="switch_cube") for _ in range(2))
    weighing.outcomes = model
    chosen = {False: 0, True: 0}
    for t in range(16):
        odds = model.odds(world, "switch_cube")
        bad = odds[..., 2] > odds[..., 0]
        robots = world.mass > 0.5
        for expert, weigh in ((blind, False), (weighing, True)):
            codes = expert(world, t)
            mine = codes[robots].long()
            chosen[weigh] += int(bad[torch.arange(len(mine)), mine].sum())
        world, _ = step(world, codes.long(), rule="switch_cube", outcomes=model, noise=torch.rand(world.mass.shape, generator=torch.Generator().manual_seed(t)))
    assert chosen[True] == 0


def test_a_cube_off_before_the_move_is_no_knock():
    from beebot.hardware import cube_outcomes as C

    row = {"cells": [[0, 5, 5], [0, 5, 8]], "after": [[0, 5, 6], [0, 5, 8]], "off_before": [1], "off_after": [1], "cube": 0, "start": [0, 5, 5], "target": [0, 5, 6], "h": [0, 0, 1], "s": [-1, 0, 0], "key": "floor_roll", "detail": "reached", "movers": 1}
    assert C.lattice_end(row)[0] == "reached"
    assert C.lattice_end({**row, "off_before": []})[0] == "knock"


def test_a_measured_knock_fits_and_replays_as_a_knock_in_any_direction():
    from beebot.hardware import cube_outcomes as C

    # MuJoCo: convex up onto a lone cube shoved it a cell further along s, off the lattice; the mover stayed
    row = {"cells": [[0, 5, 5], [0, 5, 6]], "after": [[0, 5, 5], [0, 5, 7]], "off_after": [1], "cube": 0, "start": [0, 5, 5], "target": [1, 5, 6], "h": [1, 0, 0], "s": [0, 0, 1], "key": "convex_up/lone", "detail": "knock", "movers": 1}
    outcome, aftermath = C.lattice_end(row)
    assert outcome == "knock" and aftermath == [[[0, 0, 0], [0, 0, 0], False], [[0, 1, 0], [0, 2, 0], True]]
    model = OutcomeModel(C.fit([row], "one", "release"))
    turned = {**row, "cells": [[0, 5, 5], [0, 4, 5]], "target": [1, 4, 5]}  # the same move toward -y
    for r in (row, turned):
        assert C.grid_outcomes([r], model, runs=4)[0, C.LATTICE.index("knock")] == 1.0
    assert C.grid_outcomes([row], failure=torch.zeros(27), runs=4)[0, C.LATTICE.index("reached")] == 1.0


def test_a_kick_plan_gives_a_class_its_rpms():
    from beebot.hardware import cube_outcomes as C
    from beebot.hardware.cube_swarm import RPM

    g = {"primitive": "convex_up", "bonds": 1, "hinges": 1, "rubs": 0, "lone": 0}
    assert C.kick({}, g) == [RPM["convex_up"]]
    assert C.kick({"kicks": {"convex_up": [0.8, 1.0]}}, g) == [0.8 * RPM["convex_up"], RPM["convex_up"]]
    assert C.kick({"kicks": {"convex_up/bonded/hinged": 0.9, "convex_up": [0.8, 1.0]}}, g) == [0.9 * RPM["convex_up"]]  # the class before the primitive
    assert C.kick({"escalate": [0.7, 0.9], "speed": {"convex_up": 0.5}}, g) == [0.35 * RPM["convex_up"], 0.45 * RPM["convex_up"]]


def test_escalated_rows_give_every_prefix_of_their_schedule():
    from beebot.hardware import cube_outcomes as C

    row = {"cells": [[0, 5, 5]], "after": [[0, 5, 6]], "off_after": [], "cube": 0, "start": [0, 5, 5], "target": [0, 5, 6], "h": [0, 0, 1], "s": [-1, 0, 0], "key": "floor_roll", "detail": "reached", "movers": 1, "escalate": [0.7, 0.9, 1.1], "attempts": 2, "seed": 0, "direction": [0, 0, 1]}
    assert C.plan_outcomes(row) == [((0.7,), "stayed"), ((0.7, 0.9), "reached"), ((0.7, 0.9, 1.1), "reached")]
    single = {**row, "escalate": None, "rpm_factor": 1.2}
    assert C.plan_outcomes(single) == [((1.2,), "reached")]
    table, _ = C.kick_table([row], weight=5.0)
    assert table["floor_roll"] == [0.7, 0.9]  # as good as the longer plan and shorter


def test_a_move_that_leaves_a_cube_held_up_only_by_the_mover_is_forbidden():
    # C on A traverses toward D; B hangs on C's side, so C leaving would drop it (world.overhangs)
    a, c, d, b = (0, 5, 5), (1, 5, 5), (0, 5, 4), (1, 5, 6)
    left = code(0, 0, -1)
    for rule in ("switch_cube", "free_cube"):
        assert not bool(rule_table(scene([a, c, d, b]), rule)[0, left, 1, 5, 5])
    assert bool(rule_table(scene([a, c, d]), "switch_cube")[0, left, 1, 5, 5])
    off = replace(scene([a, c, d, b]), gravity=torch.zeros(1, dtype=torch.bool))  # without gravity nothing drops
    assert bool(rule_table(off, "switch_cube")[0, left, 1, 5, 5])
