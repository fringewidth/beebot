"""Checks for the policy, the data pipeline and the goal-reaching evaluation."""

import numpy as np
import pytest
import torch

from beebot.vla import goal_eval
from beebot.vla.controller import Controller, commands_to_tensors
from beebot.vla.data import CHUNK, generate_scripted, make_windows, phase_boundaries, sample_batch, transform_field, transform_grid, transform_world
from beebot.vla.model import Policy, chunk_to_tokens, tokens_to_chunk
from beebot.vla.world import cheese_side, observe, ring_masks, sample_world, step


@pytest.mark.parametrize("grid", [16, 24])
def test_tokens_round_trip(grid):
    actions = torch.randn(2, CHUNK, 2, grid, grid)
    patch = grid // 8
    assert torch.equal(tokens_to_chunk(chunk_to_tokens(actions, patch), patch, grid), actions)


@pytest.mark.parametrize("grid", [16, 24])
def test_physics_is_equivariant_under_rotations_and_flips(grid):
    torch.manual_seed(0)
    world = sample_world(np.random.default_rng(grid), 6, size=grid)
    field = torch.randn(6, 2, grid, grid)
    score = torch.rand(6, grid, grid)  # robot priorities, rotated along with the world
    after, _ = step(world, field, score)
    for g in range(1, 8):
        k, flip = g % 4, g >= 4
        moved, _ = step(transform_world(world, k, flip), transform_field(field, k, flip), transform_grid(score, k, flip))
        expected = transform_grid(observe(after), k, flip)
        assert torch.allclose(observe(moved), expected, atol=1e-4), (k, flip)


def test_scaled_world_surround_eats_a_bigger_cheese():
    grid = 32
    world = sample_world(np.random.default_rng(5), 4, size=grid)
    assert cheese_side(grid) == 4
    controller = Controller()
    eaten = torch.zeros_like(world.cheese_alive)
    for _ in range(90):
        commands = [[{"row": int(world.cheese_pos[i, 0, 0]), "col": int(world.cheese_pos[i, 0, 1])}] for i in range(4)]
        points, valid, straight, share = commands_to_tensors(commands, 4)
        world, info = step(world, controller(world, points, valid, straight, share))
        eaten |= info.eaten
    assert eaten[:, 0].all()


def test_phase_boundaries_merge_short_phases():
    phases = ["a"] * 10 + ["b"] * 3 + ["c"] * 12 + ["d"] * 2
    bounds = phase_boundaries(phases, 8)
    assert bounds[0] == 0 and bounds[-1] == len(phases)
    assert all(b - a >= 8 for a, b in zip(bounds, bounds[1:]))


@pytest.mark.parametrize("grid,goal_tokens,goal_pool", [(16, 1, "mean"), (24, 1, "mean"), (16, 4, "attention"), (16, 4, "mean")])
def test_policy_shapes_and_backward(grid, goal_tokens, goal_pool):
    policy = Policy(grid, goal_tokens=goal_tokens, goal_pool=goal_pool)
    obs = torch.randn(3, 10, grid, grid)
    goal = torch.randn(3, 20, grid, grid)
    z = policy.goal_z(policy.goal_encoder(goal), torch.tensor([False, True, False]))
    actions = torch.randn(3, CHUNK, 2, grid, grid)
    weights = torch.rand(3, CHUNK, grid, grid)
    loss = policy.flow_loss(obs, z, actions, weights)
    loss.backward()
    assert policy.goal_encoder.out.weight.grad is not None
    assert policy.expert[0].qkv.weight.grad is not None
    policy.eval()
    assert policy.sample(obs, z, guidance=2.0).shape == (3, CHUNK, 2, grid, grid)


def test_data_pipeline_and_replay_reaches_every_goal():
    grid = 16
    store = generate_scripted(grid, 12, seed=0, device="cpu", batch_size=6)
    windows = make_windows(store, grid)
    assert len(windows) > 0
    assert ((windows[:, 2] - windows[:, 1]) >= 8).all()
    gen = torch.Generator().manual_seed(0)
    obs, goal, actions, weights, future = sample_batch(store, windows, 16, gen, "cpu")
    assert obs.shape == (16, 10, grid, grid) and goal.shape == (16, 20, grid, grid)
    assert actions.shape == (16, CHUNK, 2, grid, grid) and weights.shape == (16, CHUNK, grid, grid)
    assert actions.abs().max() <= 1.0 + 1e-6
    torch.manual_seed(0)
    replay = goal_eval.summary(goal_eval.run(None, store, windows[:64], "cpu", mode="replay"))
    # Robots that want the same square are ordered at random each step, so replaying the stored
    # fields open loop does not reproduce the episode exactly: a robot that loses one conflict
    # arrives a step late, and an eat (a closed ring) can then fail. Measured over seeds: facts
    # 0.92 to 0.95, reached 0.86 to 0.91, eat recall 0.73 to 0.87; do-nothing scores 0 facts.
    assert replay["fact_score"] >= 0.85 and replay["reached"] >= 0.75, replay
    stats = {}
    for name in ("goto", "split", "surround", "push", "stay"):
        stats[name] = sum(label.endswith(name) for row in store["phases"] for label in row)
    assert all(v > 0 for v in stats.values()), stats


def test_masked_head_fits_a_batch_and_samples_valid_actions():
    """The discrete action head (objective "masked") can memorise one batch: after training, its
    samples reproduce the batch's per-square actions, and every sampled vector is one of the 9."""
    from beebot.vla.data import CODE_VECTORS, FUTURE_ROBOT, encode_field

    torch.manual_seed(0)
    grid = 16
    store = generate_scripted(grid, 6, seed=0, device="cpu", batch_size=6)
    windows = make_windows(store, grid)
    obs, goal, actions, weights, future = sample_batch(store, windows, 8, torch.Generator().manual_seed(0), "cpu", augment=False)
    policy = Policy(grid, width=64, expert_width=64, layers=2, objective="masked")
    z = policy.goal_z(policy.goal_encoder(goal))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    first = None
    for _ in range(300):
        z = policy.goal_z(policy.goal_encoder(goal))
        loss = policy.flow_loss(obs, z, actions, weights, future=future)
        first = first if first is not None else loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.3 * first, (first, loss.item())
    policy.eval()
    with torch.no_grad():
        chunk = policy.sample(obs, policy.goal_z(policy.goal_encoder(goal)), steps=8)
    assert chunk.shape == actions.shape
    vectors = chunk.movedim(2, -1).reshape(-1, 2)
    assert (torch.cdist(vectors, CODE_VECTORS.float()).min(dim=1).values < 1e-5).all()
    on_robots = torch.cat([obs[:, :1] > 0.5, future[:, :-1] == FUTURE_ROBOT], dim=1) & (weights > 0)
    agree = (encode_field(chunk) == encode_field(actions))[on_robots].float().mean().item()
    assert agree > 0.8, agree


def test_world_model_head_fits_actions_and_future():
    """objective "masked" with world_model: the expert also fills in each square's future state;
    on one memorised batch both the actions and the loss come out right."""
    from beebot.vla.data import FUTURE_ROBOT, encode_field

    torch.manual_seed(0)
    grid = 16
    store = generate_scripted(grid, 6, seed=0, device="cpu", batch_size=6)
    windows = make_windows(store, grid)
    obs, goal, actions, weights, future = sample_batch(store, windows, 8, torch.Generator().manual_seed(0), "cpu", augment=False)
    assert future.shape == weights.shape and future.max() <= 2
    policy = Policy(grid, width=64, expert_width=64, layers=2, objective="masked", world_model=True)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    first = None
    for _ in range(300):
        loss = policy.flow_loss(obs, policy.goal_z(policy.goal_encoder(goal)), actions, weights, future=future)
        first = first if first is not None else loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.3 * first, (first, loss.item())
    policy.eval()
    with torch.no_grad():
        chunk = policy.sample(obs, policy.goal_z(policy.goal_encoder(goal)), steps=8)
    on_robots = torch.cat([obs[:, :1] > 0.5, future[:, :-1] == FUTURE_ROBOT], dim=1) & (weights > 0)
    agree = (encode_field(chunk) == encode_field(actions))[on_robots].float().mean().item()
    assert agree > 0.8, agree


def test_insulated_backbone_gets_no_gradient_from_the_expert():
    """insulate: the action expert's loss leaves the backbone untouched; only the readout trains it."""
    torch.manual_seed(0)
    grid = 16
    store = generate_scripted(grid, 4, seed=0, device="cpu", batch_size=4)
    windows = make_windows(store, grid)
    obs, goal, actions, weights, future = sample_batch(store, windows, 4, torch.Generator().manual_seed(0), "cpu", augment=False)
    policy = Policy(grid, width=64, expert_width=64, layers=2, objective="masked", insulate=True)
    kvs, readout = policy.encode(obs, policy.goal_z(policy.goal_encoder(goal)), actions, weights)
    assert all(not k.requires_grad and not v.requires_grad for k, v in kvs)
    loss = policy.flow_loss(obs, policy.goal_z(policy.goal_encoder(goal)), actions, weights, future=future)
    loss.backward()
    assert policy.backbone[0].qkv.weight.grad.abs().sum() > 0  # from the readout
    assert policy.expert[0].qkv.weight.grad.abs().sum() > 0


def test_masked_loss_ignores_the_field_on_empty_squares():
    """objective "masked": only robots' actions are context or targets, so a program's field on
    empty squares (which would leak its intent through revealed slots) cannot change the loss."""
    from beebot.vla.data import FUTURE_ROBOT

    torch.manual_seed(0)
    grid = 16
    store = generate_scripted(grid, 4, seed=0, device="cpu", batch_size=4)
    windows = make_windows(store, grid)
    obs, goal, actions, weights, future = sample_batch(store, windows, 4, torch.Generator().manual_seed(0), "cpu", augment=False)
    policy = Policy(grid, width=64, expert_width=64, layers=2, objective="masked")
    occupied = torch.cat([obs[:, :1] > 0.5, future[:, :-1] == FUTURE_ROBOT], dim=1)
    scrambled = torch.where(occupied.unsqueeze(2), actions, torch.randn_like(actions))
    z = policy.goal_z(policy.goal_encoder(goal))
    t, e = policy.loss_noise(4, "cpu")
    with torch.no_grad():
        a = policy.flow_loss(obs, z, actions, weights, t=t, e=e, future=future)
        b = policy.flow_loss(obs, z, scrambled, weights, t=t, e=e, future=future)
    assert torch.allclose(a, b), (a, b)


def test_readout_decode_inverts_the_readout_layout():
    """decode "readout": a readout that has memorised one batch gives back that batch's actions,
    so the unpatchify in sample_readout matches the patchify the readout loss uses."""
    from beebot.vla.data import encode_field

    torch.manual_seed(0)
    grid = 16
    store = generate_scripted(grid, 4, seed=0, device="cpu", batch_size=4)
    windows = make_windows(store, grid)
    obs, goal, actions, weights, future = sample_batch(store, windows, 4, torch.Generator().manual_seed(0), "cpu", augment=False)
    policy = Policy(grid, width=64, expert_width=64, layers=2, objective="masked", insulate=True)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    for _ in range(300):
        z = policy.goal_z(policy.goal_encoder(goal))
        _, readout = policy.encode(obs, z, actions, weights)
        optimizer.zero_grad()
        readout.backward()
        optimizer.step()
    policy.eval()
    policy.decode = "readout"
    with torch.no_grad():
        chunk = policy.sample(obs, policy.goal_z(policy.goal_encoder(goal)))
    agree = (encode_field(chunk) == encode_field(actions))[weights >= 1.0].float().mean().item()
    assert agree > 0.8, agree


def test_readout_objective_fits_actions_and_future_without_an_expert():
    """objective "readout" (with world_model: a future-state readout too): no expert is built, the
    backbone alone memorises one batch, and its one-shot samples reproduce the robots' actions."""
    from beebot.vla.data import FUTURE_ROBOT, encode_field

    torch.manual_seed(0)
    grid = 16
    store = generate_scripted(grid, 4, seed=0, device="cpu", batch_size=4)
    windows = make_windows(store, grid)
    obs, goal, actions, weights, future = sample_batch(store, windows, 4, torch.Generator().manual_seed(0), "cpu", augment=False)
    policy = Policy(grid, width=64, expert_width=64, layers=2, objective="readout", world_model=True)
    assert len(policy.expert) == 0 and policy.decode == "readout"
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    first = None
    for _ in range(300):
        loss = policy.flow_loss(obs, policy.goal_z(policy.goal_encoder(goal)), actions, weights, future=future)
        first = first if first is not None else loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.3 * first, (first, loss.item())
    policy.eval()
    with torch.no_grad():
        chunk = policy.sample(obs, policy.goal_z(policy.goal_encoder(goal)), guidance=2.0)
    on_robots = torch.cat([obs[:, :1] > 0.5, future[:, :-1] == FUTURE_ROBOT], dim=1) & (weights > 0)
    agree = (encode_field(chunk) == encode_field(actions))[on_robots].float().mean().item()
    assert agree > 0.8, agree
