"""Checks that the world behaves as described, so results aren't artifacts of a bug.

    uv run pytest
"""

from collections import deque

import numpy as np
import pytest
import torch

from beebot.ca import (
    GATHER_PACKING,
    N_NOTES,
    OFFSETS,
    PHEROMONE_DECAY,
    first_eaten_step,
    gather_radius,
    look,
    loss,
    rollout,
    step,
    blocked_moves,
)
from beebot.evaluate import evaluate
from beebot.layout import (
    NEIGHBORS_8,
    UNREACHABLE,
    make_terrain,
    sample_batch,
    swarm_mass,
    walking_distance,
)
from beebot.model import NCA, StickyBaseline


def reference_distance(sources, passable):
    """Plain breadth-first search on the torus with 8 neighbors."""
    size = sources.shape[0]
    distance = np.full(sources.shape, UNREACHABLE, dtype=np.int64)
    queue = deque()
    for y, x in np.argwhere(sources):
        distance[y, x] = 0
        queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy, dx in NEIGHBORS_8:
            ny, nx = (y + dy) % size, (x + dx) % size
            if passable[ny, nx] and distance[ny, nx] == UNREACHABLE:
                distance[ny, nx] = distance[y, x] + 1
                queue.append((ny, nx))
    return distance


class RandomPolicy:
    """Random destination scores and pheromone writes, to stress the physics."""

    def __init__(self, seed=0):
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, mass, cheese, walls, notes):
        b, h, w = mass.shape
        logits = torch.randn(b, len(OFFSETS), h, w, generator=self.generator) * 3
        writes = torch.randn(b, N_NOTES, h, w, generator=self.generator) * 3
        return logits, writes


class ConstantPolicy:
    """Sends all mass to one destination and writes the same value everywhere."""

    def __init__(self, destination, write=1.0):
        self.destination = destination
        self.write = write

    def __call__(self, mass, cheese, walls, notes):
        b, h, w = mass.shape
        logits = torch.full((b, len(OFFSETS), h, w), -50.0)
        logits[:, self.destination] = 50.0
        return logits, torch.full((b, N_NOTES, h, w), self.write)


@pytest.fixture(scope="module")
def batch():
    return sample_batch(np.random.default_rng(7), 24, (1, 9), (2, 12), 8, (0.0, 0.35))


def test_look_reads_the_neighbor():
    field = torch.arange(25.0).reshape(1, 5, 5)
    for dy, dx in OFFSETS:
        looked = look(field, dy, dx)
        for y in range(5):
            for x in range(5):
                assert looked[0, y, x] == field[0, (y + dy) % 5, (x + dx) % 5]


@pytest.mark.parametrize("seed", range(5))
def test_walking_distance_matches_reference(seed):
    rng = np.random.default_rng(seed)
    walls = make_terrain(rng, 20, 0.3)
    sources = np.zeros_like(walls)
    sources[tuple(np.argwhere(~walls)[0])] = True
    passable = ~walls & ~sources
    assert (walking_distance(sources, passable) == reference_distance(sources, passable)).all()


def test_layout_invariants(batch):
    for i in range(batch.mass.shape[0]):
        mass = batch.mass[i].numpy()
        cheese = batch.cheese[i].numpy()
        walls = batch.walls[i].numpy()
        ring = batch.ring[i].numpy()
        distance = batch.distance[i].numpy()

        assert not (cheese & walls).any()
        assert not (mass[cheese | walls | ring] > 0).any(), "the nest starts off the cheese, walls and ring"
        assert set(np.unique(mass)) <= {0.0, 1.0}

        touching = np.zeros_like(cheese)
        for dy, dx in NEIGHBORS_8:
            touching |= np.roll(cheese, (dy, dx), axis=(0, 1))
        # The ring is every open square touching the cheese that the swarm can walk to.
        assert (ring == (touching & ~cheese & ~walls & (distance == 1))).all()

        passable = ~walls & ~cheese
        assert (distance == reference_distance(cheese, passable)).all()
        from_nest = reference_distance(mass > 0, passable)
        assert (from_nest[ring] < UNREACHABLE).all(), "every ring square is reachable from the nest"
        reachable = passable & (distance < UNREACHABLE)
        assert mass.sum() == swarm_mass(ring, reachable.sum())


def test_random_policy_conserves_mass_and_respects_obstacles(batch):
    mass = batch.mass.clone()
    notes = torch.zeros(mass.shape[0], N_NOTES, *mass.shape[1:])
    blocked = blocked_moves(batch.cheese | batch.walls)
    total = mass.sum(dim=(1, 2))
    policy = RandomPolicy()
    for _ in range(60):
        fire = (torch.rand(mass.shape[0], 1, *mass.shape[1:]) < 0.75).float()
        mass, notes = step(policy, mass, notes, batch.cheese.float(), batch.walls.float(), blocked, fire)
        assert torch.allclose(mass.sum(dim=(1, 2)), total, atol=1e-3)
        assert (mass >= 0).all()
        assert mass[batch.cheese | batch.walls].abs().max() == 0


def test_mass_moves_to_the_chosen_neighbor():
    size = 6
    mass = torch.zeros(1, size, size)
    mass[0, 2, 2] = 1.0
    nothing = torch.zeros(1, size, size, dtype=torch.bool)
    blocked = blocked_moves(nothing)
    notes = torch.zeros(1, N_NOTES, size, size)
    fire = torch.ones(1, 1, size, size)
    east = OFFSETS.index((0, 1))
    new_mass, _ = step(ConstantPolicy(east), mass, notes, nothing.float(), nothing.float(), blocked, fire)
    assert new_mass[0, 2, 3] == pytest.approx(1.0, abs=1e-6)
    assert new_mass.sum() == pytest.approx(1.0, abs=1e-6)


def test_partial_robots_move_in_proportion():
    size = 6
    mass = torch.zeros(1, size, size)
    mass[0, 2, 2] = 0.25
    nothing = torch.zeros(1, size, size, dtype=torch.bool)
    notes = torch.zeros(1, N_NOTES, size, size)
    fire = torch.ones(1, 1, size, size)
    east = OFFSETS.index((0, 1))
    new_mass, _ = step(ConstantPolicy(east), mass, notes, nothing.float(), nothing.float(), blocked_moves(nothing), fire)
    # A quarter of a robot acts at a quarter strength: a quarter of its mass moves.
    assert new_mass[0, 2, 3] == pytest.approx(0.25 * 0.25, abs=1e-6)
    assert new_mass[0, 2, 2] == pytest.approx(0.25 * 0.75, abs=1e-6)


def test_empty_floor_and_resting_squares_write_nothing():
    size = 6
    mass = torch.zeros(1, size, size)
    mass[0, 1, 1] = 1.0
    nothing = torch.zeros(1, size, size, dtype=torch.bool)
    notes = torch.rand(1, N_NOTES, size, size)
    stay = OFFSETS.index((0, 0))
    fire = torch.ones(1, 1, size, size)
    _, new_notes = step(ConstantPolicy(stay), mass, notes, nothing.float(), nothing.float(), blocked_moves(nothing), fire)
    empty = mass[0] == 0
    assert torch.allclose(new_notes[0][:, empty], PHEROMONE_DECAY * notes[0][:, empty])
    assert not torch.allclose(new_notes[0][:, 1, 1], PHEROMONE_DECAY * notes[0][:, 1, 1])

    resting = torch.zeros(1, 1, size, size)
    still_mass, still_notes = step(
        ConstantPolicy(OFFSETS.index((0, 1))), mass, notes, nothing.float(), nothing.float(), blocked_moves(nothing), resting
    )
    assert torch.equal(still_mass, mass), "a square that doesn't fire keeps its mass"
    assert torch.allclose(still_notes, PHEROMONE_DECAY * notes)


def test_gather_radius_matches_brute_force(batch):
    open_squares = ~batch.cheese & ~batch.walls
    total = batch.mass.sum(dim=(1, 2))
    radius = gather_radius(batch.distance, open_squares, total)
    for i in range(batch.mass.shape[0]):
        d = batch.distance[i][open_squares[i]]
        r = 1
        while (d <= r).sum() * GATHER_PACKING < total[i]:
            r += 1
        assert radius[i].item() == r


def test_eaten_needs_ring_and_gathering():
    batch = sample_batch(np.random.default_rng(3), 20, (1, 1), (6, 8), 1, (0.0, 0.0))
    total = batch.mass.sum().item()
    ring = batch.ring[0]
    distance = batch.distance[0]
    far = (distance == distance[distance < UNREACHABLE].max()).nonzero()[0]

    def eaten_with(mass):
        assert mass.sum().item() == pytest.approx(total, abs=1e-4)
        b = type(batch)(mass=mass, cheese=batch.cheese, walls=batch.walls, ring=batch.ring, distance=batch.distance)
        return bool(rollout(ConstantPolicy(OFFSETS.index((0, 0))), b, 1, 1.0, 0.5).eaten[0, 0])

    away = torch.zeros_like(batch.mass)
    away[0][ring] = 0.6
    away[0, far[0], far[1]] = total - away.sum()
    assert not eaten_with(away), "a full ring isn't enough while most of the swarm is away"

    home = torch.zeros_like(batch.mass)
    home[0][ring] = total / ring.sum()
    assert eaten_with(home)


def test_loss_is_finite_and_trains_every_layer(batch):
    model = NCA()
    result = rollout(model, batch, 12, 0.75, 0.5)
    value = loss(result, 0.5, 0.5, 1.0, 0.5)
    assert torch.isfinite(value)
    value.backward()
    assert all(p.grad is not None for p in model.parameters())
    assert model.head.weight.grad.abs().sum() > 0


def test_first_eaten_step():
    class Fake:
        eaten = torch.tensor([[False, False], [True, False], [True, False]])

    assert first_eaten_step(Fake()).tolist() == [2, 0]


def test_evaluation_is_deterministic():
    first = evaluate(StickyBaseline(), torch.device("cpu"), "val")
    second = evaluate(StickyBaseline(), torch.device("cpu"), "val")
    assert first == second


def test_privileged_swarm_smells_walking_distance(batch):
    model = NCA(privileged=True)
    rollout(model, batch, 2, 0.75, 0.5)
    assert model.scent.shape == batch.distance.shape
    assert torch.allclose(model.scent[batch.cheese], torch.ones(()))
    blind = NCA()
    rollout(blind, batch, 2, 0.75, 0.5)
    assert blind.scent is None


def test_blind_model_can_start_from_privileged_weights(tmp_path):
    from beebot.model import load_weights

    teacher = NCA(privileged=True)
    torch.nn.init.normal_(teacher.body[0].weight)
    path = tmp_path / "teacher.pt"
    torch.save(teacher.state_dict(), path)
    student = NCA()
    load_weights(student, path, "cpu")
    keep = [c for c in range(teacher.body[0].weight.shape[1]) if c != 3]
    assert torch.equal(student.body[0].weight, teacher.body[0].weight[:, keep])
    assert torch.equal(student.head.weight, teacher.head.weight)


def test_wrapped_privileged_policy_smells_the_current_batch():
    from beebot.probe import PheromoneBlackout

    rng = np.random.default_rng(11)
    model = NCA(privileged=True)
    wrapped = PheromoneBlackout(model)
    for size in (16, 16, 20):
        b = sample_batch(rng, size, (1, 3), (2, 6), 2, (0.0, 0.2))
        rollout(wrapped, b, 1, 0.75, 0.5)
        assert torch.equal(model.scent, torch.exp(-b.distance.clamp(max=1000) / 8.0))
