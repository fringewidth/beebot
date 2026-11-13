import math

import numpy as np

from beebot.hardware.cube import G, NOMINAL
from beebot.hardware.cube_swarm import DESIGN, SwarmMagnets
from beebot.hardware.cube_turning import CRANK, MASS, Face, FreeHinge, TurningMagnets, TurningSwarm, module, quarter_turn


def test_the_module_keeps_the_hinge_cubes_magnets_in_newtons():
    base, heavy = DESIGN.cube(), module()
    assert heavy.mass == MASS and heavy.side == base.side
    assert abs(heavy.edge_pull * heavy.mass - base.edge_pull * base.mass) < 1e-12
    closed = (8 * heavy.edge_pull + heavy.face_pull) * heavy.mass * G
    assert 52 < closed < 54  # N: HARDWARE.md's 53 N a face


def test_with_no_inner_share_the_magnets_are_cube_swarms():
    cube = module()
    rng = np.random.default_rng(0)
    pos = np.array([[0, 0, 0.025], [0.0503, 0.001, 0.0255], [0.0, 0.0508, 0.0252]])
    rot = np.stack([np.eye(3)] * 3)
    scale = 1 + rng.uniform(-0.2, 0.2, size=3)
    theirs = SwarmMagnets(cube, 3, scale).wrench(pos, rot)
    ours, discs = TurningMagnets(cube, 3, scale, Face(share=0.0), [(0, 1)] * 3).wrench(pos, rot, rot)
    assert np.allclose(ours, theirs, atol=1e-9) and np.allclose(discs, 0.0)


def test_a_closed_face_pulls_the_same_whatever_the_share():
    cube = module()
    pos = np.array([[0, 0, 0.5], [0.05, 0, 0.5]])  # far above the steel
    rot = np.stack([np.eye(3)] * 2)
    pulls = []
    for share in (0.0, 0.5):
        body, disc = TurningMagnets(cube, 2, np.ones(2), Face(share=share, proud=0.0), [(0, 1)] * 2).wrench(pos, rot, rot)
        pulls.append(body[0, 0] + disc[0, 0])
    assert pulls[0] > 45 and abs(pulls[1] - pulls[0]) < 0.12 * pulls[0]
    assert disc[0, 0] > 0.4 * pulls[1]  # half of it holds the neighbour by the disc


def test_quarter_turn_of_a_cell_offset():
    assert quarter_turn((0, 0, 1), (1, 0, 0), 1).tolist() == [0, -1, 0]
    assert quarter_turn((0, 0, 1), (1, 0, 0), -1).tolist() == [0, 1, 0]
    assert quarter_turn((2, 1, 0), (1, 0, 0), 4).tolist() == [2, 1, 0]


def test_the_face_turns_a_bonded_cube_a_quarter_and_relocks():
    swarm = TurningSwarm([(0, 0, 0), (1, 0, 0), (1, 0, 1)], seed=0, jitter=NOMINAL, settle=0.1)
    row = swarm.turn(1, 1, carried=[2])
    assert (row["servo"], row["detail"]) == ("arrived", "reached") and abs(row["error"]) < 1.0
    _, rot = swarm.poses()
    assert abs(rot[2][2, 1] - 1) < 0.01  # the carried cube's y axis now points up: a quarter turn about +x
    assert np.abs(rot[1] - np.eye(3)).max() < 0.02  # the driver's body has not turned
    assert swarm.lattice().cells.tolist() == [[0, 0, 0], [1, 0, 0], [1, 0, 1]]


def test_too_weak_a_face_motor_turns_back_and_leaves_the_arm_where_it_was():
    cells = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 0, 1), (2, 1, 1)]  # an arm cube out to the side: 0.1 N m of gravity
    swarm = TurningSwarm(cells, seed=0, jitter=NOMINAL, settle=0.1, face=Face(torque=0.05))
    row = swarm.turn(2, 1, carried=[3, 4])  # lifting it
    assert row["servo"] == "returned" and row["outcome"] == "stayed"
    assert swarm.lattice().cells.tolist() == [list(c) for c in cells]


def test_cubes_with_a_turning_face_still_pivot_about_an_edge():
    swarm = TurningSwarm([(0, 0, 0), (0, 0, 1)], seed=0, jitter=NOMINAL, settle=0.1)
    climb = swarm.move(1, (1, 0, -1))
    assert (climb.primitive, climb.detail, climb.cell) == ("convex_up", "reached", (1, 0, 0))


def test_the_plain_hinge_cube_may_not_pass_under_the_arm_and_the_free_one_may():
    from beebot.hardware.cube_hinge import HingeSwarm

    cells = CRANK[:4] + [(2, -1, 1)]  # the load hangs on the arm's side
    assert HingeSwarm(cells, seed=0, jitter=NOMINAL, settle=0.05).plan(4, (-1, 1, 0)) is None
    pivot = FreeHinge(cells, seed=0, jitter=NOMINAL, settle=0.05).plan(4, (-1, 1, 0))
    assert pivot.primitive == "under" and pivot.target == (1, 0, 1)


def test_detent_holds_within_its_capture_only():
    swarm = TurningSwarm([(0, 0, 0)], seed=0, jitter=NOMINAL, settle=0.05)
    width = math.radians(swarm.face.capture)
    swarm.data.qpos[swarm.disc_qpos[0]] = width / 2
    assert abs(swarm.passive()[0] + swarm.face.detent) < 1e-9  # the peak, pushing back
    swarm.data.qpos[swarm.disc_qpos[0]] = 3 * width
    assert abs(swarm.passive()[0]) < 1e-9  # only the drag, and the disc is still
