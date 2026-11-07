"""The relative-position policies (vla/swarm.py): crops, positions, memorisation, size freedom."""

import pytest
import torch
import torch.nn.functional as F

from beebot.vla.data import encode_field, make_windows, sample_batch
from beebot.vla.swarm import RelativePolicy, robot_crops, rope_angles, rotate
from beebot.vla.train import generate_scripted


def test_rotary_attention_depends_only_on_offsets():
    """Shifting every position by the same amount leaves every query-key product unchanged."""
    torch.manual_seed(0)
    q, k = torch.randn(1, 2, 5, 32), torch.randn(1, 2, 5, 32)
    pos = torch.rand(1, 5, 2) * 20
    a = rope_angles(pos)
    b = rope_angles(pos + torch.tensor([7.0, -3.0]))
    scores_a = rotate(q, a) @ rotate(k, a).transpose(-1, -2)
    scores_b = rotate(q, b) @ rotate(k, b).transpose(-1, -2)
    assert torch.allclose(scores_a, scores_b, atol=1e-4)


def test_robot_crops_center_on_each_robot_and_see_walls_outside():
    obs = torch.zeros(1, 10, 8, 8)
    obs[0, 0, 0, 0] = 1.0  # a robot in the corner
    obs[0, 0, 5, 3] = 1.0
    obs[0, 2, 5, 4] = 1.0  # something to its right
    crops, positions, owner, valid = robot_crops(obs, 3)
    assert valid.all() and crops.shape == (1, 2, 10 * 9)
    assert sorted(owner[0].tolist()) == [0, 5 * 8 + 3]
    view = crops.view(1, 2, 10, 3, 3)
    corner = owner[0].tolist().index(0)
    assert view[0, corner, 0, 1, 1] == 1.0 and view[0, corner, 1, 0, 0] == 1.0  # itself; wall beyond the edge
    other = 1 - corner
    assert view[0, other, 2, 1, 2] == 1.0 and positions[0, other].tolist() == [5.0, 3.0]


def _fits(view):
    torch.manual_seed(0)
    grid = 16
    store = generate_scripted(grid, 4, seed=0, device="cpu", batch_size=4)
    windows = make_windows(store, grid)
    obs, goal, actions, weights, future = sample_batch(store, windows, 4, torch.Generator().manual_seed(0), "cpu", augment=False)
    policy = RelativePolicy(view, width=64, layers=2)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    z = policy.goal_z(policy.goal_encoder(goal))
    first = None
    for _ in range(250):
        loss = policy.flow_loss(obs, z, actions, weights)
        first = first if first is not None else loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.3 * first, (first, loss.item())
    chunk = policy.sample(obs, z, guidance=2.0)
    robots = (obs[:, 0] > 0.5) & (weights[:, 0] > 0)
    agree = (encode_field(chunk[:, 0]) == encode_field(actions[:, 0]))[robots].float().mean().item()
    assert agree > 0.8, agree
    # Any grid size: a bigger world runs through the same weights.
    big = torch.zeros(2, 10, 32, 32)
    big[:, 0, 10:14, 10:14] = 1.0
    assert policy.sample(big, policy.goal_encoder(torch.cat([big, big], dim=1))).shape == (2, 8, 2, 32, 32)


def test_global_view_memorises_and_runs_on_any_size():
    _fits("global")


def test_swarm_view_memorises_and_runs_on_any_size():
    _fits("swarm")


def _text_fits(view, text_mode="add"):
    """With text, a caption embedding steers the moves: distinct embeddings memorise distinct
    batches, the null (all-zero) embedding samples, and guidance runs."""
    torch.manual_seed(0)
    grid = 16
    store = generate_scripted(grid, 4, seed=0, device="cpu", batch_size=4)
    windows = make_windows(store, grid)
    obs, goal, actions, weights, future = sample_batch(store, windows, 4, torch.Generator().manual_seed(0), "cpu", augment=False)
    policy = RelativePolicy(view, width=64, layers=2, text_dim=16, text_mode=text_mode)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    emb = torch.randn(4, 16)
    null = torch.tensor([False, True, False, False])
    first = None
    for _ in range(250):
        z = policy.goal_z(policy.text_head(emb), null)
        loss = policy.flow_loss(obs, policy.goal_z(policy.text_head(emb)), actions, weights) + 0.1 * policy.flow_loss(obs, z, actions, weights)
        first = first if first is not None else loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.3 * first, (first, loss.item())
    assert policy.contrastive_loss(goal, policy.text_head(emb), None).item() == 0.0
    chunk = policy.sample(obs, policy.goal_z(policy.text_head(emb)), guidance=2.0)
    assert chunk.shape == (4, 8, 2, grid, grid)
    # Goal frames still work in the same model, and goal-only models have no text weights.
    assert policy.sample(obs, policy.goal_z(policy.goal_encoder(goal))).shape == chunk.shape
    assert not any(name.startswith("text") for name, _ in RelativePolicy(view, width=64, layers=2).named_parameters())


def test_global_view_follows_text():
    _text_fits("global")


def test_swarm_view_follows_text():
    _text_fits("swarm")


def test_swarm_view_follows_text_through_cross_attention():
    _text_fits("swarm", "cross")


def test_distillation_copies_the_goal_conditioned_moves():
    """The text student's moves approach the goal teacher's; rows not kept contribute nothing."""
    torch.manual_seed(0)
    grid = 16
    store = generate_scripted(grid, 4, seed=0, device="cpu", batch_size=4)
    obs, goal, actions, weights, future = sample_batch(store, make_windows(store, grid), 4, torch.Generator().manual_seed(0), "cpu", augment=False)
    policy = RelativePolicy("swarm", width=64, layers=2, text_dim=16, text_mode="cross")
    emb, keep = torch.randn(4, 16), torch.ones(4, dtype=torch.bool)
    teacher_z = policy.goal_encoder(goal)
    with torch.no_grad():
        teacher, _, _ = policy.logits(obs, teacher_z)
    entropy = -(F.softmax(teacher, -1) * F.log_softmax(teacher, -1)).sum(-1)
    text_params = [p for n, p in policy.named_parameters() if "text" in n or "cross" in n]
    optimizer = torch.optim.AdamW(text_params, lr=3e-3)
    first = None
    for _ in range(200):
        loss = policy.distill_loss(obs, policy.text_head(emb), teacher_z, weights, keep)
        first = first if first is not None else loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Cross-entropy falls towards the teacher's own entropy (its floor); only text weights moved.
    assert loss.item() < first, (first, loss.item())
    assert loss.item() - entropy.mean().item() < 0.5 * (first - entropy.mean().item())
    assert policy.distill_loss(obs, policy.text_head(emb), teacher_z, weights, torch.zeros(4, dtype=torch.bool)).item() == 0.0


def test_frequency_aware_scale_keeps_short_offsets_and_squeezes_long_ones():
    from beebot.vla.swarm import frequency_aware_scale

    assert torch.allclose(frequency_aware_scale(32, 32), torch.ones(8))
    scale = frequency_aware_scale(32, 64)
    assert scale[0] == 1.0 and scale[-1] == 0.5  # fastest exact, slowest squeezed by 32 / 64
    assert (scale[1:] <= scale[:-1]).all()  # never squeezes a faster frequency more than a slower one


@pytest.mark.parametrize("text_mode", ["add", "cross"])
def test_swarm_view_with_text_has_no_absolute_position(text_mode):
    """Text goals add nothing positional: shifting the whole scene leaves every robot's move
    distribution unchanged (away from the edges, where the crops would see the outside wall)."""
    torch.manual_seed(0)
    policy = RelativePolicy("swarm", width=32, layers=2, text_dim=8, text_mode=text_mode).eval()
    for name, param in policy.named_parameters():
        if "cross.out" in name:  # zero at init: give the text a say before checking
            torch.nn.init.normal_(param, std=0.1)
    obs = torch.zeros(1, 10, 32, 32)
    obs[0, 0, 12:15, 10:13] = 1.0  # robots
    obs[0, 2, 13, 16] = 1.0  # a cheese nearby
    shifted = torch.roll(obs, shifts=(5, 7), dims=(2, 3))
    z = policy.text_head(torch.randn(1, 8))
    with torch.no_grad():
        a, _, _ = policy.logits(obs, z)
        b, _, _ = policy.logits(shifted, z)
    assert torch.allclose(a, b, atol=1e-5)
