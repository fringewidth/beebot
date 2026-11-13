"""Domain randomization (vla/voxel/randomize.py, views.Placement): every item works on its own and is
off by default, randomized episodes are valid (replay_drawn: frames follow DrawnPhysics.step, no forbidden
command, the goal is reached) and keep the unrandomized frames' kind, broken cubes never move, placement
changes the picture only when asked."""

import numpy as np
import pytest
import torch

from beebot.vla.voxel import randomize as DR
from beebot.vla.voxel import reverse_live as L
from beebot.vla.voxel import views as V
from beebot.vla.voxel import world as W
from beebot.vla.voxel.world import STAY


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def nominal():
    """A few small nominal episodes (seed, row)."""
    rows = []
    for seed in range(40):
        row, _ = L.generate(seed, ["gather" if seed % 2 else "goto"], robots=(6, 12))
        if row is not None and row["meta"]["steps"] >= 4:
            rows.append((seed, row))
        if len(rows) == 4:
            break
    assert len(rows) == 4
    return rows


@pytest.fixture(scope="module")
def overlapping():
    """Episodes whose starts share cells with their goals (a broken cube must sit on a goal cell)."""
    rows = []
    for seed in range(60):
        row, _ = L.generate(seed, [("move", "spread", "line", "groups")[seed % 4]], robots=(6, 14))
        if row is not None and row["meta"]["steps"] >= 3:
            rows.append((seed, row))
        if len(rows) == 8:
            break
    return rows


def kept(rows, rand, tries=12):
    """The first randomized row of each nominal row that is not dropped."""
    out = []
    for seed, row in rows:
        for k in range(tries):
            made = DR.randomize_row(row, rand, seed * 100 + k)
            if made is not None:
                out.append((row, made))
                break
    return out


def test_everything_is_off_by_default():
    rand = DR.Randomization(strength=0.0, placement=0.0)
    assert not rand.any()
    draw = DR.draw_physics(rand, np.random.default_rng(0))
    assert draw.outcomes is None and draw.p_drop == 0 and draw.p_late == 0 and draw.offset == 0 and not draw.broken


def test_a_row_without_physics_is_unchanged(nominal):
    seed, row = nominal[0]
    made = DR.randomize_row(row, DR.Randomization(strength=0.0, placement=0.0), seed)
    assert made["mass"] is row["mass"] and made["meta"]["steps"] == row["meta"]["steps"]
    assert "stuck" not in made


def test_failure_scaling_keeps_probabilities_valid():
    reach, stay = torch.tensor([0.9, 0.6, 1.0]), torch.tensor([0.06, 0.2, 0.0])
    r2, s2 = DR._scaled(reach, stay, 2.0)
    assert torch.allclose(1 - r2, (1 - reach) * 2) or bool((r2 >= 0).all())
    assert float(1 - r2[0]) == pytest.approx(0.2) and float(s2[0]) == pytest.approx(0.12)
    assert float(r2[2]) == 1.0
    r0, s0 = DR._scaled(reach, stay, 2.0, 0.0)  # harmless only: the harmful share is gone
    assert torch.allclose(1 - r0, s0)
    assert bool((r2 >= 0).all()) and bool((s2 <= 1 - r2 + 1e-6).all())
    big, _ = DR._scaled(reach, stay, 100.0)
    assert bool((big >= -1e-6).all())


def test_per_episode_multiplier_is_drawn_in_range():
    rand = DR.Randomization(strength=1.0, failure=1.0)
    values = [DR.draw_physics(rand, np.random.default_rng(i)).multiplier for i in range(200)]
    assert 0.5 <= min(values) < 0.7 and 1.5 < max(values) <= 2.0


def test_dropped_and_late_commands():
    world = W.world_from_arrays(torch.zeros(1, 16, 16, 16))
    mass = torch.zeros(1, 16, 16, 16)
    mass[0, 0, 8, 8] = 1
    world = W.world_from_arrays(mass[0], gravity=True)
    codes = torch.full(mass.shape, STAY, dtype=torch.long)
    codes[0, 0, 8, 8] = W.DIRS.index((0, 0, 1))
    dropped = DR.DrawnPhysics(DR.PhysicsDraw(seed=1, p_drop=1.0), world)
    after, realized = dropped.step(world, codes, 0)
    assert bool(after.mass[0, 0, 8, 8] > 0.5) and int(realized[0, 0, 8, 8]) == STAY
    late = DR.DrawnPhysics(DR.PhysicsDraw(seed=1, p_late=1.0), world)
    mid, _ = late.step(world, codes, 0)
    assert bool(mid.mass[0, 0, 8, 8] > 0.5) and late.waiting
    nothing = torch.full(mass.shape, STAY, dtype=torch.long)
    done, _ = late.step(mid, nothing, 1)
    assert bool(done.mass[0, 0, 8, 9] > 0.5) and not bool(done.mass[0, 0, 8, 8] > 0.5)


def test_weak_cube_factor_follows_the_cube():
    scale = torch.ones(1, 4, 4, 4)
    scale[0, 0, 1, 1] = 3.0
    realized = torch.full((1, 4, 4, 4), STAY, dtype=torch.long)
    realized[0, 0, 1, 1] = W.DIRS.index((0, 0, 1))
    after = torch.zeros(1, 4, 4, 4)
    after[0, 0, 1, 2] = 1
    moved = DR.follow(scale, realized, after)
    assert float(moved[0, 0, 1, 2]) == 3.0 and float(moved.sum()) == 3.0 + 63
    still = torch.zeros(1, 4, 4, 4)
    still[0, 0, 1, 1] = 1
    assert float(DR.follow(scale, realized, still)[0, 0, 1, 1]) == 3.0


@pytest.mark.parametrize("config", [dict(failure=1.0, harm=0.0), dict(failure=1.0, harm=1.0), dict(weak=1.0), dict(drop=1.0), dict(late=1.0), dict(drop=1.0, late=1.0, failure=1.0)])
def test_randomized_episodes_are_valid_and_reach_the_goal(nominal, config):
    rand = DR.Randomization(strength=0.0, placement=0.0, **config)
    made = kept(nominal, rand)
    assert made
    for row, out in made:
        ep = L.LiveEpisode(out)
        assert DR.replay_drawn(out)["passed"]
        assert np.array_equal(ep.occupancy(ep.steps), L.LiveEpisode(row).occupancy(row["meta"]["steps"]))  # the goal
        assert int(out["meta"]["robots"]) == int(ep.occupancy(0).sum())
        for t in range(ep.steps):  # robots are conserved on every frame
            assert int(ep.occupancy(t).sum()) == int(out["meta"]["robots"])


def test_failures_lengthen_episodes_and_are_recorded(nominal):
    rand = DR.Randomization(strength=0.0, placement=0.0, drop=1.0)
    made = kept(nominal, rand)
    assert any(out["meta"]["repeats"] + out["meta"]["replans"] > 0 for _, out in made)
    assert all(out["meta"]["steps"] >= row["meta"]["steps"] for row, out in made)


def test_a_broken_cube_never_moves_and_the_goal_holds_it(overlapping):
    rand = DR.Randomization(strength=0.0, placement=0.0, broken=1.0)
    found = 0
    for seed, row in overlapping:
        for k in range(40):
            out = DR.randomize_row(row, rand, seed * 100 + k)
            if out is None or not out["meta"]["physics"]["broken"]:
                continue
            ep = L.LiveEpisode(out)
            for z, y, x in out["meta"]["physics"]["broken"]:
                for t in range(ep.steps + 1):
                    assert ep.occupancy(t)[z, y, x]
                    assert bool(ep.world(t).stuck[0, z, y, x])
            assert DR.replay_drawn(out)["passed"]
            found += 1
            break
    assert found


def test_a_broken_cube_does_not_respond_to_commands():
    mass = torch.zeros(1, 16, 16, 16)
    mass[0, 0, 8, 8] = 1
    world = W.world_from_arrays(mass[0], gravity=True)
    physics = DR.DrawnPhysics(DR.PhysicsDraw(seed=1, broken=[[0, 8, 8]]), world)
    world = physics.start(world)
    codes = torch.full(mass.shape, STAY, dtype=torch.long)
    codes[0, 0, 8, 8] = W.DIRS.index((0, 0, 1))
    after, realized = physics.step(world, codes, 0)
    assert bool(after.mass[0, 0, 8, 8] > 0.5) and not bool(after.mass[0, 0, 8, 9] > 0.5)
    assert bool(after.stuck[0, 0, 8, 8])


def test_a_broken_cube_off_the_goal_makes_no_plan():
    from beebot.vla.voxel import reverse as R  # noqa: F401

    mass = torch.zeros(1, 16, 16, 16)
    mass[0, 0, 8, 8] = 1
    start = W.world_from_arrays(mass[0], gravity=True)
    goal = mass.clone()
    goal[0, 0, 8, 8], goal[0, 0, 8, 10] = 0, 1
    stuck = torch.zeros(1, 16, 16, 16, dtype=torch.bool)
    stuck[0, 0, 8, 8] = True
    found, why = DR._plan_from(DR.replace(start, stuck=stuck), W.world_from_arrays(goal[0], gravity=True), np.random.default_rng(0), 20)
    assert found is None and why == "stuck cell not in the goal"


def test_placement_moves_robots_off_the_grid_only_when_asked():
    mass = torch.zeros(1, 6, 6, 6)
    mass[0, 0, 2:4, 2:4] = 1
    obs = W.observe(W.world_from_arrays(mass[0], gravity=True))[0]
    spec = V.ViewSpec.clean(size=96)
    rng = np.random.default_rng(0)
    plain, _ = V.scene_views(obs, rng, spec)
    off, _ = V.scene_views(obs, rng, spec, jitter=V.Placement(seed=3, offset=0.0, yaw=0.0))
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(plain, off))
    moved, _ = V.scene_views(obs, rng, spec, jitter=V.Placement(seed=3, offset=0.2, yaw=8.0))
    assert any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(plain, moved))
    with V.placement(V.Placement(seed=3, offset=0.2, yaw=8.0)):
        again, _ = V.scene_views(obs, rng, spec)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(moved, again))
    after, _ = V.scene_views(obs, rng, spec)  # the context is gone
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(plain, after))


def test_placement_is_consistent_across_frames_with_slow_drift():
    kind = np.zeros((4, 4, 4), np.int8)
    kind[0, 1, 1] = V.KIND_ROBOT
    faces = V.voxel_faces(kind)
    a = V.placement_faces(faces, V.Placement(seed=5, offset=0.1, yaw=5.0, t=0.0))
    b = V.placement_faces(faces, V.Placement(seed=5, offset=0.1, yaw=5.0, t=1.0))
    c = V.placement_faces(faces, V.Placement(seed=6, offset=0.1, yaw=5.0, t=0.0))
    assert np.abs(a.c - faces.c).max() <= 0.1 + 1e-6 and np.abs(np.degrees(a.yaw)).max() <= 5.0 + 1e-6
    assert np.abs(a.c - b.c).max() < 0.1 * 0.3 * 0.2  # one step moves it a little
    assert np.abs(a.c - c.c).max() > 0  # another episode, another draw
    assert np.allclose(V.placement_faces(faces, V.Placement(seed=5, offset=0.1, yaw=5.0, t=0.0)).c, a.c)  # fixed per (seed, cell, t)


def test_live_with_physics_gives_examples_and_drops_nothing_unfinished():
    live = L.Live(share=1.0, families=("gather", "goto"), physics={"strength": 1.0}, robots=(6, 12), workers=0, held=2)
    rng = np.random.default_rng(0)
    for _ in range(4):
        example = live.sample(rng)
        assert example.live in ("gather", "goto")
    for ep in live.cache:
        assert DR.replay_drawn(ep.row)["passed"] if "stuck" in ep.row else True
    assert live.physics == {"strength": 1.0}


def test_live_default_has_no_physics():
    assert L.Live(share=0.5).physics is None
