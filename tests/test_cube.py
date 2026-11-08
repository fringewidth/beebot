import math

import numpy as np
import pytest

from beebot.hardware.cube import G, NOMINAL, Cube, Magnets, magnet_sites, trial
from beebot.hardware.cube_search import REF, Build


def test_sites_and_twin_spacing():
    inset = 0.01
    sites, edge = magnet_sites(inset)
    assert sites.shape == (30, 3) and edge.sum() == 24
    # the twin of a +x face magnet on the next cube over sits 2 * inset away, straight across
    plus_x = sites[(~edge) & (sites[:, 0] > 0)][0]
    minus_x = sites[(~edge) & (sites[:, 0] < 0)][0] + np.array([1.0, 0, 0])
    assert np.allclose(minus_x - plus_x, [2 * inset, 0, 0])


def test_face_contact_holds_with_the_sum_of_twin_pulls():
    cube = Cube()
    magnets = Magnets(cube, 2, np.ones(2))
    pos = np.array([[0.0, 0, 0], [cube.side, 0, 0]])
    rot = np.stack([np.eye(3)] * 2)
    pull = magnets.wrench(pos, rot)[0, 0] / (cube.mass * G)  # on cube 0, toward +x
    # four shared edges with two twin pairs each, one face pair; every twin pair at its peak
    assert pull == pytest.approx(8 * cube.edge_pull + cube.face_pull, rel=0.05)


def test_build_at_reference_size_is_the_reference():
    cube = Build(side=0.05).cube()
    for field in ("mass", "flywheel_share", "brake_torque", "edge_pull", "face_pull"):
        assert getattr(cube, field) == pytest.approx(getattr(REF, field))


def test_build_scaling():
    small = Build(side=0.03).cube()
    assert small.brake_torque == pytest.approx(REF.brake_torque * 0.6**3)
    assert small.edge_pull == pytest.approx(REF.edge_pull / 0.6)  # area over volume


def rpm(value):
    return value * 2 * math.pi / 60


def test_floor_roll_needs_the_right_speed():
    cube = Cube()
    rng = np.random.default_rng(0)
    assert trial(cube, "floor_roll", rpm(1500), rng, NOMINAL)["outcome"] == "stay"
    assert trial(cube, "floor_roll", rpm(3400), rng, NOMINAL)["outcome"] == "success"
    assert trial(cube, "floor_roll", rpm(15000), rng, NOMINAL)["outcome"] == "off"
