import numpy as np

from beebot.hardware.cube import NOMINAL, Magnets
from beebot.hardware.cube_swarm import DESIGN, DIRECTIONS, Swarm, SwarmMagnets, patch


def test_fourteen_moves_without_horizontal_diagonals():
    assert len(DIRECTIONS) == 14
    assert all(dy == 0 or dx == 0 for _, dy, dx in DIRECTIONS)


def test_ten_cubes_rest_on_their_cells_and_lattice_round_trips():
    cells = patch(2, 5)
    swarm = Swarm(cells, seed=0)
    start, _ = swarm.poses()
    swarm.run(1.0)
    rest, _ = swarm.poses()
    reading = swarm.lattice()
    assert np.abs(rest - start).max() < 0.01  # L
    assert not reading.off.any()
    assert [tuple(c) for c in reading.cells.tolist()] == cells
    assert reading.occupied() == set(cells)


def test_lone_cube_rolls_to_the_next_cell_and_back():
    swarm = Swarm([(0, 0, 0)], seed=0, jitter=NOMINAL)
    there = swarm.move(0, (0, 0, 1))
    assert (there.primitive, there.outcome, there.cell) == ("floor_roll", "reached", (0, 0, 1))
    back = swarm.move(0, (0, -1, 0))
    assert (back.outcome, back.cell) == ("reached", (0, -1, 1))
    assert [tuple(c) for c in swarm.lattice().cells.tolist()] == [(0, -1, 1)]


def test_plan_picks_the_primitive_from_the_surroundings():
    swarm = Swarm([(0, 0, 0), (0, 0, 1), (1, 0, 1), (0, 0, 4)], seed=0, jitter=NOMINAL, settle=0.05)
    assert swarm.plan(3, (0, 0, 1)).primitive == "floor_roll"
    assert swarm.plan(3, (0, 0, -1)).primitive == "floor_roll"
    assert swarm.plan(3, (1, 0, 0)) is None  # nothing to climb
    assert swarm.plan(3, (0, 1, 1)) is None  # a horizontal diagonal is not a move
    assert swarm.plan(0, (1, 0, 0)).primitive == "concave_up"  # up the wall at x = 1
    assert swarm.plan(0, (0, 0, -1)).primitive == "floor_peel"  # away from a bonded neighbour
    up = swarm.plan(2, (-1, 0, 1))  # off the top of the stack, down and over
    assert (up.primitive, up.target, up.axis) == ("convex_down", (0, 0, 2), (0, 1, 0))
    assert swarm.plan(1, (0, 0, 1)) is None  # a cube sits on it
    assert swarm.move(1, (0, 0, 1)).detail == "refused"


def test_culled_pair_search_gives_the_same_forces():
    cube = DESIGN.cube()
    rng = np.random.default_rng(0)
    cells = np.array([(x, y, z) for x in range(3) for y in range(2) for z in range(2)], dtype=float)
    pos = (cells + rng.uniform(-0.05, 0.05, cells.shape)) * cube.side
    rot = np.stack([np.eye(3)] * len(pos))
    strength = rng.uniform(0.75, 1.25, len(pos))
    assert np.allclose(Magnets(cube, len(pos), strength).wrench(pos, rot), SwarmMagnets(cube, len(pos), strength).wrench(pos, rot))
