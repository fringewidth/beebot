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
