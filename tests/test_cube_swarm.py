import numpy as np

from beebot.hardware.cube import NOMINAL, Magnets
from beebot.hardware.cube_swarm import DESIGN, DIRECTIONS, Swarm, SwarmMagnets, disturbed, patch


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


def test_plan_counts_what_holds_the_mover_back():
    swarm = Swarm([(0, 0, 0), (0, 1, 1), (0, 0, -1)], seed=0, jitter=NOMINAL, settle=0.05)
    east = swarm.plan(0, (0, 0, 1))  # away from the cube behind, past one touching by an edge, into the cell beside it
    assert (east.primitive, east.bonds, east.hinges, east.rubs) == ("floor_peel", 1, 1, 1)
    south = swarm.plan(1, (0, -1, 0))  # into the slot beside cube 0, sliding down its face
    assert (south.primitive, south.bonds, south.hinges, south.rubs) == ("floor_roll", 0, 1, 1)
    west = swarm.plan(2, (0, 0, -1))  # away from cube 0: bonded behind, nothing else
    assert (west.primitive, west.bonds, west.hinges, west.rubs) == ("floor_peel", 1, 0, 0)


def test_culled_pair_search_gives_the_same_forces():
    cube = DESIGN.cube()
    rng = np.random.default_rng(0)
    cells = np.array([(x, y, z) for x in range(3) for y in range(2) for z in range(2)], dtype=float)
    pos = (cells + rng.uniform(-0.05, 0.05, cells.shape)) * cube.side
    rot = np.stack([np.eye(3)] * len(pos))
    strength = rng.uniform(0.75, 1.25, len(pos))
    assert np.allclose(Magnets(cube, len(pos), strength).wrench(pos, rot), SwarmMagnets(cube, len(pos), strength).wrench(pos, rot))


def test_switched_off_pairs_exert_no_force():
    cube = DESIGN.cube()
    cells = np.array([(0, 0, 10), (1, 0, 10), (2, 0, 10)], dtype=float)  # (x, y, z) in L, far above the steel floor
    pos = cells * cube.side
    rot = np.stack([np.eye(3)] * 3)
    magnets = SwarmMagnets(cube, 3, np.ones(3))
    cut = np.zeros((3, 3), dtype=bool)
    cut[0, 1] = cut[1, 0] = True
    magnets.switch(cut)
    alone = magnets.wrench(pos, rot)
    assert np.allclose(alone[0], 0, atol=1e-5) and not np.allclose(alone[2], 0, atol=1e-5)  # cube 0 feels only the far floor; cubes 1 and 2 still pull together
    magnets.switch(None)
    assert not np.allclose(magnets.wrench(pos, rot)[0], 0, atol=1e-5)


def test_release_frees_a_mover_held_by_a_face_bond():
    for seed in (0, 1):
        held = Swarm([(0, 0, 0), (0, 0, 1)], seed=seed)
        assert held.move(0, (0, 0, -1), rpm=6000).detail == "stayed"  # floor peel: the bond holds it
        free = Swarm([(0, 0, 0), (0, 0, 1)], seed=seed, release=True)
        moved = free.move(0, (0, 0, -1), rpm=6000)
        assert (moved.detail, moved.cell) == ("reached", (0, 0, -1))
        assert free.magnets.cut is None  # switched back on after the move


def test_released_pairs_keep_what_the_mover_leans_on():
    swarm = Swarm([(0, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 0)], seed=0, jitter=NOMINAL, settle=0.05)
    pivot = swarm.plan(0, (1, 0, 0))  # concave up the stack at x = 1, beside the cube at y = 1
    assert pivot.primitive == "concave_up" and set(pivot.leans) == {(0, 0, 1), (1, 0, 1)}
    cut, quiet = swarm.released([0], [pivot], swarm.lattice())
    assert cut[0].tolist() == [False, False, False, True] and cut[3, 0] and not cut[1:3, 1:].any()
    assert quiet is None  # neighbours' faces switch only with faces=True


def test_faces_switch_off_the_neighbours_face_toward_the_mover():
    swarm = Swarm([(0, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 0)], seed=0, jitter=NOMINAL, settle=0.05, release=True, faces=True)
    pivot = swarm.plan(0, (1, 0, 0))
    cut, quiet = swarm.released([0], [pivot], swarm.lattice())
    per = quiet.reshape(swarm.n, -1)
    assert per.sum(axis=1).tolist() == [0, 0, 0, 9]  # the support and the cube above it lean; the cube at y = 1 quiets its face's 8 edge magnets and its centre
    local = swarm.magnets.local[per[3]]  # (x, y, z) in m: the face toward the mover is the cube's -y face
    assert np.allclose(local[:, 1], local[:, 1].min()) and local[:, 1].min() < -0.4 * swarm.cube.side


def test_a_knock_counts_only_cubes_this_move_disturbed():
    rot = np.stack([np.eye(3)] * 3)
    tilted = np.array([[1, 0, 0], [0, np.cos(0.3), -np.sin(0.3)], [0, np.sin(0.3), np.cos(0.3)]])
    before = rot.copy()
    before[1] = tilted  # cube 1 was already tilted before the move
    after = before.copy()
    after[2] = tilted  # the move tilts cube 2
    assert disturbed(before, after).tolist() == [False, False, True]
    after[1] = tilted @ tilted  # and turns cube 1 further
    assert disturbed(before, after).tolist() == [False, True, True]


def test_a_cube_still_moving_does_not_put_the_others_off():
    swarm = Swarm([(0, 0, 0), (0, 0, 3)], seed=0, jitter=NOMINAL, settle=0.05)
    swarm.data.qvel[swarm.cube_dofs[1, 3]] = 5.0  # cube 1 spins
    reading = swarm.lattice()
    assert reading.off.tolist() == [False, True]


def test_several_cubes_move_at_once():
    swarm = Swarm([(0, 0, 0), (0, 0, 4), (0, 3, 2)], seed=0)
    moves = swarm.moves([(0, (0, 0, 1)), (1, (0, 0, -1)), (2, (0, -1, 0))])
    assert [m.detail for m in moves] == ["reached"] * 3
    assert [m.cell for m in moves] == [(0, 0, 1), (0, 0, 3), (0, 2, 2)]
    assert swarm.moves([(0, (0, 0, 1)), (1, (1, 0, 0))])[1].detail == "refused"
