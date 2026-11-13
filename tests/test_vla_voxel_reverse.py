import numpy as np
import pytest
import torch

from beebot.vla.voxel import reverse as R
from beebot.vla.voxel import world as W
from beebot.vla.voxel.lm import world_at


@pytest.fixture(scope="module")
def walked():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    out = {}
    for family in R.FAMILIES:
        goals = [R.make_goal(family, rng) for _ in range(4)]
        out[family] = R.take_apart(goals, rng)
    return out


@pytest.mark.parametrize("family", R.FAMILIES)
def test_goals_are_fixed_points(family):
    rng = np.random.default_rng(1)
    goals = [R.make_goal(family, rng) for _ in range(6)]
    world = W.concat([g.world for g in goals])
    assert R.fixed_point(world).all()
    assert all(g.text and g.target.any() for g in goals)
    assert all(bool(world.gravity.all()) for g in goals)
    for g in goals:  # the instruction speaks a stranger's words, not the sim's
        assert not any(word in g.text for word in ("cheese", "zone", "stripes"))


@pytest.mark.parametrize("family", R.FAMILIES)
def test_reversed_episodes_replay_forward(walked, family):
    episodes, stats = walked[family]
    result = R.replay(episodes)
    assert result["passed"].all(), result["first_bad"]
    assert sum(stats["accepted"]) > 0 and sum(stats["proposed"]) >= sum(stats["accepted"])
    for ep in episodes:
        assert ep.steps >= 1
        goal = ep.goal.world
        assert torch.equal(ep.mass[-1].bool(), goal.mass[0] > 0.5)  # ends exactly in its goal
        occupied = ep.mass[:-1].bool()
        assert (ep.actions[~occupied] == W.STAY).all()  # codes only where robots are (the store's format)
        moves = ep.actions[occupied].long()
        assert W.CUBE_MOVES[moves].all()  # moves a pivoting cube makes


def test_push_moves_the_object_onto_its_patch(walked):
    episodes, _ = walked["push"]
    for ep in episodes:
        slot = ep.goal.push["slot"]
        start, end = ep.cheese_pos[0, slot], ep.cheese_pos[-1, slot]
        assert int((end - start).abs().sum()) >= R.MIN_PUSHES
        zone = ep.world.zones[0] >= 0
        cells = W.cheese_id_map(world_at_ep(ep, ep.steps))[0] == slot
        assert zone[cells.any(dim=0)].all()  # every footprint cell on the patch


def world_at_ep(ep, t):
    from dataclasses import replace

    return replace(ep.world, mass=ep.mass[t].unsqueeze(0).float(), cheese_pos=ep.cheese_pos[t].unsqueeze(0), cheese_alive=ep.cheese_alive[t].unsqueeze(0))


def test_verify_rejects_a_step_with_a_fall_or_a_refused_move():
    world = R.blank()
    world.mass[0, :3, 5, 5] = 1.0  # a tower of three
    later = world
    # a robot one up in the air beside the tower with STAY would fall into place: not at rest
    earlier = R.blank()
    earlier.mass[0, :2, 5, 5] = 1.0
    earlier.mass[0, 3, 5, 5] = 1.0
    codes = torch.full(earlier.mass.shape, W.STAY, dtype=torch.long)
    assert not bool(R.verify(earlier, codes, later)[0])
    # the top robot arriving from beside the tower at height 1: a legal climb, verifies
    earlier = R.blank()
    earlier.mass[0, :2, 5, 5] = 1.0
    earlier.mass[0, 1, 6, 5] = 1.0
    codes = torch.full(earlier.mass.shape, W.STAY, dtype=torch.long)
    codes[0, 1, 6, 5] = R.code_of((1, -1, 0))
    assert bool(R.verify(earlier, codes, later)[0])
    # a horizontal diagonal no cube makes: refused, so never verifies
    earlier = R.blank()
    earlier.mass[0, :2, 5, 5] = 1.0
    earlier.mass[0, 2, 6, 6] = 1.0
    codes = torch.full(earlier.mass.shape, W.STAY, dtype=torch.long)
    codes[0, 2, 6, 6] = R.code_of((0, -1, -1))
    assert not bool(R.verify(earlier, codes, later)[0])


def test_store_layout_reads_back(walked):
    episodes = walked["letter"][0] + walked["push"][0]
    store, labels = R.to_store(episodes)
    assert len(labels) == len(episodes)
    rows = [(i, t) for i, ep in enumerate(episodes) for t in (0, ep.steps)]
    worlds = world_at(store, rows)
    for j, (i, t) in enumerate(rows):
        assert torch.equal(worlds.mass[j] > 0.5, episodes[i].mass[t].bool())
        assert torch.equal(worlds.cheese_pos[j], episodes[i].cheese_pos[t])
    assert W.observe(worlds).shape[1] == W.N_CHANNELS
    steps = store["actions"].shape[1]
    assert store["mass"].shape[1] == steps + 1
    for i, ep in enumerate(episodes):  # padding: the goal held with STAY
        assert (store["actions"][i, ep.steps :] == W.STAY).all()


# ---------------------------------------------------------------------------------------------
# v2: extraction, scaffolding, guided walks, natural bands (runs/handoff/reverse-v2.md)


def test_local_sources_match_the_full_grid():
    rng = np.random.default_rng(3)
    world, _ = W.sample_object_scenes(rng, 3, gravity_prob=1.0)
    occ = world.mass > 0.5
    cheese = W.cheese_id_map(world) >= 0
    scene, here, src, legal = R.single_moves(world)
    for n in range(0, len(scene), 7):
        cubes = occ[scene[n]].clone().unsqueeze(0)
        cubes[0, here[n, 0], here[n, 1], here[n, 2]] = False
        table = W.pivot_moves(cubes, world.walls[scene[n]].unsqueeze(0), cheese[scene[n]].unsqueeze(0), release=True)[0]
        for k, code in enumerate(R._CUBE_CODES.tolist()):
            s = src[n, k]
            inside = bool(((s >= 0) & (s < W.SIZE)).all())
            want = inside and bool(table[code, s[0], s[1], s[2]]) and not bool((occ | world.walls | cheese)[scene[n], s[0], s[1], s[2]])
            assert bool(legal[n, k]) == want


@pytest.fixture(scope="module")
def walked_v2():
    torch.manual_seed(0)
    rng = np.random.default_rng(5)
    out = {}
    for family in R.NEW_FAMILIES + R.GUIDED_FAMILIES:
        goals = [R.make_goal(family, rng) for _ in range(1 if family == "big" else 3)]
        out[family] = R.take_apart(goals, rng)
    return out


@pytest.mark.parametrize("family", R.NEW_FAMILIES + R.GUIDED_FAMILIES)
def test_v2_episodes_replay_forward(walked_v2, family):
    episodes, _ = walked_v2[family]
    result = R.replay(episodes)
    assert result["passed"].all(), result["first_bad"]
    for ep in episodes:
        assert ep.steps >= 1 and len(ep.kinds) == ep.steps
        assert torch.equal(ep.mass[-1].bool(), ep.goal.world.mass[0] > 0.5)
        occupied = ep.mass[:-1].bool()
        assert (ep.actions[~occupied] == W.STAY).all()
        assert W.CUBE_MOVES[ep.actions[occupied].long()].all()


def test_big_goals_have_60_to_115_robots():
    rng = np.random.default_rng(2)
    for _ in range(6):
        g = R.make_goal("big", rng)
        assert 60 <= int((g.world.mass > 0.5).sum()) <= 115


def test_structures_reach_a_natural_start(walked_v2):
    for family in ("table", "chair", "cantilever", "arch", "span"):
        episodes, _ = walked_v2[family]
        assert sum(ep.natural for ep in episodes) >= 2, [ep.end for ep in episodes]
        for ep in episodes:
            if ep.natural:
                assert ep.stats[R.STATS.index("fill")] < 0.5  # little of the structure left standing


def test_span_is_closed_with_helpers():
    rng = np.random.default_rng(0)
    goal = R.make_goal("span", rng)
    world = goal.world
    scene, here, src, legal = R.single_moves(world)
    deck = goal.target[here[:, 0], here[:, 1], here[:, 2]]
    assert not legal[deck].any()  # no deck robot can be the last one in without support beside it
    episodes, stats = R.take_apart([goal], rng)
    ep = episodes[0]
    assert R.replay(episodes)["passed"].all()
    assert "scaffold" in ep.kinds and stats["scaffolds"][0] >= 1


def test_guided_walks_reach_their_start_exactly(walked_v2):
    for family in R.GUIDED_FAMILIES:
        episodes, _ = walked_v2[family]
        for ep in episodes:
            assert ep.goal.start is not None
            if ep.end == "reached":
                assert ep.distance == 0 and torch.equal(ep.mass[0].bool(), ep.goal.start)
        assert sum(ep.end == "reached" for ep in episodes) >= 2


def test_edits_touch_few_robots(walked_v2):
    episodes, _ = walked_v2["edit"]
    for ep in episodes:
        if ep.end == "reached":
            assert 1 <= ep.touched <= 5  # the arm (at most three) and the robot that changes place, nothing else


def test_natural_bands_pass_natural_scenes_and_fail_goals():
    rng = np.random.default_rng(4)
    passed = []
    for family in ("table", "tower", "letter", "big"):
        g = R.make_goal(family, rng)
        band = R.natural_bands(g.world, g.target, rng)
        assert not R.in_band(R.start_stats(g.world, g.target.unsqueeze(0))[0], band)  # the goal itself is no natural scene
        scenes = R.natural_scenes(g.world.walls[0], int((g.world.mass > 0.5).sum()), rng, 16)
        ref = R.replace(g.world.index([0] * 16), mass=scenes.float())
        passed += [R.in_band(row, band) for row in R.start_stats(ref, g.target.unsqueeze(0).expand(16, -1, -1, -1))]
    assert np.mean(passed) > 0.6


def test_parked_robots_stand_in_a_row_on_an_edge():
    rng = np.random.default_rng(6)
    for _ in range(20):
        g = R.make_goal("table", rng)
        spare = (g.world.mass[0] > 0.5) & ~g.target
        if g.unused == "parked" and spare.sum() >= 2:
            break
    cells = spare.nonzero()
    assert (cells[:, 0] == 0).all()
    on_edge = (cells[:, 1] == 0).all() | (cells[:, 1] == W.SIZE - 1).all() | (cells[:, 2] == 0).all() | (cells[:, 2] == W.SIZE - 1).all()
    assert bool(on_edge) and "edge" in g.text
