"""Checks for the Feast stage: several cheeses that vanish once their ring is full."""

import numpy as np
import pytest
import torch

from beebot.ca import N_NOTES, OFFSETS
from beebot.feast import FeastBatch, evaluate, first_done_step, loss, rollout, sample_batch
from beebot.layout import MASS_PER_RING_SQUARE, NEIGHBORS_8, UNREACHABLE, walking_distance
from beebot.model import NCA, StickyBaseline


class RandomPolicy:
    def __init__(self, seed=0):
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, mass, cheese, walls, notes):
        b, h, w = mass.shape
        logits = torch.randn(b, len(OFFSETS), h, w, generator=self.generator) * 3
        writes = torch.randn(b, N_NOTES, h, w, generator=self.generator) * 3
        return logits, writes


class StayPolicy:
    def __call__(self, mass, cheese, walls, notes):
        b, h, w = mass.shape
        logits = torch.full((b, len(OFFSETS), h, w), -50.0)
        logits[:, 0] = 50.0
        return logits, torch.zeros(b, N_NOTES, h, w)


def touching(mask):
    out = np.zeros_like(mask)
    for dy, dx in NEIGHBORS_8:
        out |= np.roll(mask, (dy, dx), axis=(0, 1))
    return out


@pytest.fixture(scope="module")
def batch():
    return sample_batch(np.random.default_rng(3), 24, (2, 4), 8)


def test_layout_invariants(batch):
    for i in range(batch.mass.shape[0]):
        mass = batch.mass[i].numpy()
        walls = batch.walls[i].numpy()
        cheeses = [c for c in batch.cheese[i].numpy() if c.any()]
        rings = batch.rings[i].numpy()
        assert 2 <= len(cheeses) <= 4
        any_cheese = np.logical_or.reduce(cheeses)
        all_rings = rings.any(axis=0)

        for j, c in enumerate(cheeses):
            assert not (c & walls).any()
            others = any_cheese & ~c
            assert not (touching(c) & others).any(), "cheeses never touch"
            assert (rings[j] == (touching(c) & ~any_cheese & ~walls)).all()
            assert rings[j].any()
        assert not (mass[any_cheese | walls | all_rings] > 0).any(), "the nest starts off the cheese, walls and rings"
        assert mass.sum() >= MASS_PER_RING_SQUARE * all_rings.sum()
        from_nest = walking_distance(mass > 0, ~walls & ~any_cheese)
        assert (from_nest[all_rings] < UNREACHABLE).all(), "every ring square is reachable from the nest"


def test_random_policy_conserves_mass_and_respects_obstacles(batch):
    result = rollout(RandomPolicy(), batch, 60, 0.75, 0.5, record=True)
    total = batch.mass.sum(dim=(1, 2)).numpy()
    walls = batch.walls.numpy()
    for t, m in enumerate(result.masses):
        assert np.allclose(m.sum(axis=(1, 2)), total, atol=1e-3)
        assert (m >= 0).all()
        assert np.abs(m[walls]).max() == 0
        # Mass never sits on cheese that was present while it moved.
        present_before = batch.cheese.any(dim=1).numpy() if t == 0 else result.cheeses[t - 1]
        assert np.abs(m[present_before]).max() == 0


def test_cheese_vanishes_once_its_ring_is_full():
    size = 12
    walls = torch.zeros(1, size, size, dtype=torch.bool)
    cheese = torch.zeros(1, 2, size, size, dtype=torch.bool)
    cheese[0, 0, 3, 3] = True
    cheese[0, 1, 8, 8] = True
    rings = torch.zeros_like(cheese)
    for k, (y, x) in enumerate([(3, 3), (8, 8)]):
        rings[0, k, y - 1 : y + 2, x - 1 : x + 2] = True
        rings[0, k, y, x] = False
    mass = torch.zeros(1, size, size)
    mass[rings[:, 0]] = 1.0  # the first ring is full from the start; the second is empty
    batch = FeastBatch(mass=mass, cheese=cheese, walls=walls, rings=rings)

    result = rollout(StayPolicy(), batch, 3, 1.0, 0.5, record=True)
    assert result.left[:, 0].tolist() == [1, 1, 1]
    assert not result.cheeses[0][0, 3, 3], "the eaten cheese is gone"
    assert result.cheeses[0][0, 8, 8], "the other cheese is still there"
    # Half the cheeses are left unsurrounded, so the deficit is the second ring's whole share.
    assert torch.allclose(result.deficit[:, 0], torch.full((3,), 0.5))
    assert first_done_step(result).tolist() == [0]

    mass[rings[:, 1]] = 1.0
    result = rollout(StayPolicy(), FeastBatch(mass=mass, cheese=cheese, walls=walls, rings=rings), 3, 1.0, 0.5)
    assert result.left[:, 0].tolist() == [0, 0, 0]
    assert first_done_step(result).tolist() == [1]
    assert result.deficit.abs().max() < 1e-6, "nothing is left to surround after the first step"


def test_loss_is_finite_and_trains_every_layer(batch):
    model = NCA()
    value = loss(rollout(model, batch, 12, 0.75, 0.5), 1.0)
    assert torch.isfinite(value)
    value.backward()
    assert all(p.grad is not None for p in model.parameters())
    assert model.head.weight.grad.abs().sum() > 0


def test_evaluation_is_deterministic():
    first = evaluate(StickyBaseline(), torch.device("cpu"), "val")
    second = evaluate(StickyBaseline(), torch.device("cpu"), "val")
    assert first == second
