"""Checks for the Rivals stage: two species in one world, with different payoffs."""

import numpy as np
import pytest
import torch

from beebot.ca import N_NOTES, OFFSETS
from beebot.layout import UNREACHABLE, walking_distance
from beebot.model import RivalNCA
from beebot.rivals import GAMES, TAU, RivalBatch, losses, rollout, sample_batch, push


class Stay:
    def __call__(self, mass, other, cheese, walls, notes, other_notes):
        b, h, w = mass.shape
        logits = torch.full((b, len(OFFSETS), h, w), -50.0)
        logits[:, 0] = 50.0
        return logits, torch.zeros(b, N_NOTES, h, w)


class Wander:
    def __init__(self, seed):
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, mass, other, cheese, walls, notes, other_notes):
        b, h, w = mass.shape
        logits = torch.randn(b, len(OFFSETS), h, w, generator=self.generator) * 3
        return logits, torch.randn(b, N_NOTES, h, w, generator=self.generator)


def one_cheese_world(size=12):
    walls = torch.zeros(1, size, size, dtype=torch.bool)
    cheese = torch.zeros(1, 1, size, size, dtype=torch.bool)
    cheese[0, 0, 6, 6] = True
    rings = torch.zeros_like(cheese)
    rings[0, 0, 5:8, 5:8] = True
    rings[0, 0, 6, 6] = False
    return walls, cheese, rings


@pytest.mark.parametrize("game", GAMES)
def test_layout_invariants(game):
    batch = sample_batch(np.random.default_rng(1), 20, game, 8)
    for i in range(batch.mass_a.shape[0]):
        a = batch.mass_a[i].numpy() > 0
        b = batch.mass_b[i].numpy() > 0
        walls = batch.walls[i].numpy()
        cheese = batch.cheese[i].numpy().any(axis=0)
        rings = batch.rings[i].numpy().any(axis=0)
        assert not (a & b).any(), "the nests don't overlap"
        assert not ((a | b) & (cheese | walls | rings)).any()
        for nest in (a, b):
            assert (walking_distance(nest, ~walls & ~cheese)[rings] < UNREACHABLE).all()
        assert batch.mass_a[i].sum() == batch.mass_b[i].sum(), "both swarms get the same mass"
    if game == "staghunt":
        stag_ring = batch.rings[:, -1].flatten(1).sum(dim=1).float()
        each = batch.mass_a.flatten(1).sum(dim=1)
        assert (each < TAU * stag_ring).all(), "neither swarm can fill the stag's ring alone"
        assert (2 * each > TAU * stag_ring).all(), "together they can"


def test_mass_is_conserved_per_species():
    batch = sample_batch(np.random.default_rng(2), 16, "shared", 6)
    result = rollout(Wander(0), Wander(1), batch, 40, 0.75, TAU, record=True)
    for masses, start in ((result.masses_a, batch.mass_a), (result.masses_b, batch.mass_b)):
        total = start.sum(dim=(1, 2)).numpy()
        for m in masses:
            assert np.allclose(m.sum(axis=(1, 2)), total, atol=1e-3)
            assert np.abs(m[batch.walls.numpy()]).max() == 0


def test_eating_needs_both_species_combined_and_splits_by_ownership():
    walls, cheese, rings = one_cheese_world()
    ring = rings[0, 0]
    mass_a = torch.zeros(1, 12, 12)
    mass_b = torch.zeros(1, 12, 12)
    mass_a[0][ring] = 0.3  # neither alone holds more than TAU on the ring
    mass_b[0][ring] = 0.3
    mass_b[0, 5, 5] = 0.9  # but B holds more of it (0.9 instead of 0.3 on one square)
    batch = RivalBatch(mass_a=mass_a, mass_b=mass_b, cheese=cheese, value=torch.ones(1, 1), walls=walls, rings=rings)
    result = rollout(Stay(), Stay(), batch, 2, 1.0, TAU)
    assert result.eaten_value[0, 0] == 1.0
    a, b = result.payoff_a.item(), result.payoff_b.item()
    assert a + b == pytest.approx(1.0)
    assert b > a
    assert b == pytest.approx((0.3 * 7 + 0.9) / (0.3 * 8 + 0.3 * 7 + 0.9))


def test_payoffs_by_game():
    walls, cheese, rings = one_cheese_world()
    ring = rings[0, 0]
    mass_a = torch.zeros(1, 12, 12)
    mass_b = torch.zeros(1, 12, 12)
    mass_a[0][ring] = 1.0  # A alone fills the ring
    mass_b[0, 0, 0] = 8.0
    batch = RivalBatch(mass_a=mass_a, mass_b=mass_b, cheese=cheese, value=torch.ones(1, 1), walls=walls, rings=rings)
    result = rollout(Stay(), Stay(), batch, 3, 1.0, TAU)
    shared = losses(result, "shared", 0.0)
    zerosum = losses(result, "zerosum", 0.0)
    stag = losses(result, "staghunt", 0.0)
    assert shared[0] == shared[1] == pytest.approx(-1.0)
    assert zerosum[0] == pytest.approx(-1.0) and zerosum[1] == pytest.approx(1.0)
    assert stag[0] == pytest.approx(-1.0) and stag[1] == pytest.approx(0.0)


def test_each_loss_reaches_both_rules_but_trains_only_its_own():
    batch = sample_batch(np.random.default_rng(3), 16, "staghunt", 4)
    a, b = RivalNCA(), RivalNCA()
    loss_a, loss_b = losses(rollout(a, b, batch, 12, 0.75, TAU), "staghunt", 1.0, 0.1, 0.1)
    grads_a = torch.autograd.grad(loss_a, list(a.parameters()), retain_graph=True)
    grads_b = torch.autograd.grad(loss_b, list(b.parameters()))
    assert all(torch.isfinite(g).all() for g in grads_a + grads_b)
    assert a.head.weight.grad is None and b.head.weight.grad is None


def test_staghunt2_stag_needs_both_species():
    walls, cheese, rings = one_cheese_world()
    ring = rings[0, 0]
    joint = torch.ones(1, 1, dtype=torch.bool)
    mass_a = torch.zeros(1, 12, 12)
    mass_b = torch.zeros(1, 12, 12)
    mass_a[0][ring] = 1.0  # A alone fills the ring
    mass_b[0, 0, 0] = 8.0
    alone = RivalBatch(mass_a=mass_a, mass_b=mass_b, cheese=cheese, value=torch.ones(1, 1), walls=walls, rings=rings, joint=joint)
    result = rollout(Stay(), Stay(), alone, 3, 1.0, TAU)
    assert result.eaten_value[-1, 0] == 0.0
    assert losses(result, "staghunt2", 0.0)[0] == pytest.approx(0.0)

    mass_b = torch.zeros(1, 12, 12)
    mass_a[0][ring] = 0.6  # both hold half the ring's mass
    mass_b[0][ring] = 0.6
    together = RivalBatch(mass_a=mass_a, mass_b=mass_b, cheese=cheese, value=torch.ones(1, 1), walls=walls, rings=rings, joint=joint)
    result = rollout(Stay(), Stay(), together, 3, 1.0, TAU)
    assert result.eaten_value[-1, 0] == 1.0
    assert result.payoff_a.item() == pytest.approx(0.5)


def test_staghunt2_swarms_have_the_other_games_density():
    rng = np.random.default_rng(4)
    for size in (22, 36):
        old = sample_batch(rng, size, "staghunt", 4).mass_a.sum(dim=(1, 2)).mean()
        new = sample_batch(rng, size, "staghunt2", 4)
        assert new.mass_a.sum(dim=(1, 2)).mean() > old
        assert new.joint[:, -1].all() and not new.joint[:, :-1].any()


def test_push_moves_cheese_away_from_touching_robots_and_conserves_mass():
    walls, cheese, rings = one_cheese_world()
    mass_a = torch.zeros(1, 12, 12)
    mass_b = torch.zeros(1, 12, 12)
    ys, xs = cheese[0, 0].nonzero(as_tuple=True)
    mass_a[0, ys.min() - 1, xs.min()] = 0.3  # touching the cheese from the north only
    present = torch.ones(1, 1, dtype=torch.bool)
    new_a, new_b, moved, soft = push(mass_a, mass_b, cheese, cheese.float(), present, walls, TAU)
    assert torch.equal(moved, torch.roll(cheese, 1, dims=2))  # one square south
    assert torch.equal(soft, moved.float())
    assert new_a.sum() == pytest.approx(mass_a.sum().item())


def test_full_ring_holds_a_pushable_cheese_and_it_is_eaten():
    walls, cheese, rings = one_cheese_world()
    mass_a = torch.zeros(1, 12, 12)
    mass_b = torch.zeros(1, 12, 12)
    mass_a[0][rings[0, 0]] = 0.6
    batch = RivalBatch(mass_a=mass_a, mass_b=mass_b, cheese=cheese, value=torch.ones(1, 1), walls=walls, rings=rings)
    result = rollout(Stay(), Stay(), batch, 4, 1.0, TAU, record=True, pushable=True)
    assert result.eaten_value[0, 0] == 1.0


def test_push_games_play_and_train():
    for game in ("shared-push", "zerosum-push"):
        batch = sample_batch(np.random.default_rng(5), 16, game, 3)
        a, b = RivalNCA(), RivalNCA()
        result = rollout(a, b, batch, 12, 0.75, TAU, record=True, pushable=True)
        loss_a, _ = losses(result, game, 1.0)
        grads = torch.autograd.grad(loss_a, list(a.parameters()), allow_unused=True)
        assert any(g is not None and g.abs().sum() > 0 for g in grads)
        total = (batch.mass_a.sum() + batch.mass_b.sum()).item()
        assert sum(m.sum() for m in result.masses_a[-1:]) + sum(m.sum() for m in result.masses_b[-1:]) == pytest.approx(total, rel=1e-4)
