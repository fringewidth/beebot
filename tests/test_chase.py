"""Checks for the Chase stage: one cheese that runs from the swarm."""

import numpy as np
import pytest
import torch

from beebot.ca import N_NOTES, OFFSETS
from beebot.chase import ChaseBatch, cone_masses, evaluate, move_cheese, ring_of, rollout, sample_batch
from beebot.feast import first_done_step, loss
from beebot.layout import UNREACHABLE, walking_distance
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


@pytest.fixture(scope="module")
def batch():
    return sample_batch(np.random.default_rng(5), 16, 20)


def empty_world(size=12):
    walls = torch.zeros(1, size, size, dtype=torch.bool)
    cheese = torch.zeros(1, size, size, dtype=torch.bool)
    cheese[0, 6, 6] = True
    return walls, cheese


def test_layout_invariants(batch):
    for i in range(batch.mass.shape[0]):
        mass = batch.mass[i].numpy()
        cheese = batch.cheese[i].numpy()
        walls = batch.walls[i].numpy()
        ring = ring_of(batch.cheese[i : i + 1], batch.walls[i : i + 1])[0].numpy()
        assert 1 <= cheese.sum() <= 5
        assert not (cheese & walls).any()
        assert ring.any()
        assert not (mass[cheese | walls | ring] > 0).any(), "the nest starts off the cheese, walls and ring"
        from_nest = walking_distance(mass > 0, ~walls & ~cheese)
        assert (from_nest[ring] < UNREACHABLE).all()


def test_mass_is_conserved_and_never_on_walls_or_cheese(batch):
    result = rollout(RandomPolicy(), batch, 60, 0.75, 0.5, record=True)
    total = batch.mass.sum(dim=(1, 2)).numpy()
    walls = batch.walls.numpy()
    for m, c in zip(result.masses, result.cheeses):
        assert np.allclose(m.sum(axis=(1, 2)), total, atol=1e-3)
        assert (m >= -1e-6).all()
        assert np.abs(m[walls]).max() == 0
        assert np.abs(m[c]).max() < 1e-5


def test_cheese_stays_when_it_senses_no_robots():
    walls, cheese = empty_world()
    mass = torch.zeros(1, 12, 12)
    mass[0, 0, 0] = 3.0  # far outside the sensing radius
    _, moved = move_cheese(mass, cheese, walls, 0.5)
    assert torch.equal(moved, cheese)


def test_cheese_runs_from_the_robots():
    walls, cheese = empty_world()
    mass = torch.zeros(1, 12, 12)
    mass[0, 4, 6] = 1.0  # robots to the north
    _, moved = move_cheese(mass, cheese, walls, 0.5)
    assert moved[0, 7, 6] and moved.sum() == 1, "it moves south, away from the robots"
    sensed = cone_masses(mass, cheese)
    assert sensed[0, 0] == 1.0 and sensed[0, 1:].sum() == 0


def test_cheese_is_blocked_by_walls_and_by_robots():
    walls, cheese = empty_world()
    walls[0, 7, 6] = True  # wall to the south
    mass = torch.zeros(1, 12, 12)
    mass[0, 4, 6] = 1.0
    mass[0, 6, 7] = 0.6  # a robot to the east, too solid to push
    mass[0, 6, 5] = 0.6  # and to the west
    new_mass, moved = move_cheese(mass, cheese, walls, 0.5)
    assert moved[0, 5, 6], "the only open way is north, toward the robots"
    assert torch.isclose(new_mass.sum(), mass.sum())


def test_thin_mass_is_pushed_into_the_vacated_square():
    walls, cheese = empty_world()
    mass = torch.zeros(1, 12, 12)
    mass[0, 3, 6] = 1.0  # robots to the north, south and west
    mass[0, 9, 6] = 1.0
    mass[0, 6, 3] = 1.0
    mass[0, 6, 7] = 0.2  # a thin film to the east, the emptiest side
    new_mass, moved = move_cheese(mass, cheese, walls, 0.5)
    assert moved[0, 6, 7]
    assert new_mass[0, 6, 7] == 0 and torch.isclose(new_mass[0, 6, 6], torch.tensor(0.2))


def test_cheese_is_eaten_once_its_ring_is_full():
    walls, cheese = empty_world()
    mass = torch.zeros(1, 12, 12)
    mass[0, 5:8, 5:8] = 1.0
    mass[0, 6, 6] = 0.0
    result = rollout(StayPolicy(), ChaseBatch(mass=mass, cheese=cheese, walls=walls), 4, 1.0, 0.5)
    assert first_done_step(result).tolist() == [1]
    assert result.deficit.abs().max() < 1e-6


def test_loss_trains_every_layer(batch):
    model = NCA()
    value = loss(rollout(model, batch, 12, 0.75, 0.5), 1.0, 0.1, 0.5)
    assert torch.isfinite(value)
    value.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_evaluation_is_deterministic():
    first = evaluate(StickyBaseline(), torch.device("cpu"), "val")
    second = evaluate(StickyBaseline(), torch.device("cpu"), "val")
    assert first == second
