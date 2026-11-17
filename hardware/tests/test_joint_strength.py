"""hardware/joint_strength.py against hand calculations (units: module weights, L)."""

import numpy as np
import pytest

from beebot.hardware.joint_strength import Joints, check, check_grid, check_many

WALL = [(0, 0, -1), (1, 0, -1)]
J = Joints()


def test_face_moment_is_four_edge_pulls():
    assert J.moment() == pytest.approx(4 * 4.5 + 0.5 * 0.3)
    assert 0.85 < J.edge_share < 0.95


@pytest.mark.parametrize("n", [3, 6, 7])
def test_cantilever_from_wall(n):
    """The root face holds n^2 / 2 m g L of moment with 4 edge_pull + 0.5 face_pull."""
    r = check([(1, 0, x) for x in range(n)], anchored=WALL)
    assert r.factor == pytest.approx(2 * J.moment() / n**2, rel=1e-4)
    assert r.stands == (n <= 6)
    if not r.stands:
        assert len(r.moving) == n
        assert r.opening == [((1, 0, 0), "fixed", "face")]


def test_free_l_overturns_on_steel():
    """Column of 3 on the steel floor with an arm of n at the top: about the base's edge the arm
    turns it over with n^2 / 2, the column holds it with 1.5 and the steel with 0.3 of a face's
    magnet moment, unscaled: factor = 0.3 M / (n^2 / 2 - 1.5)."""
    n = 4
    cells = [(z, 0, 0) for z in range(3)] + [(2, 0, x) for x in range(1, n + 1)]
    r = check(cells)
    assert r.factor == pytest.approx(0.3 * J.moment() / (n**2 / 2 - 1.5), rel=1e-4)
    assert r.opening == [((0, 0, 0), "fixed", "face")]


@pytest.mark.parametrize("h", [3, 15])
def test_tower_side_load(h):
    """A side load P at the top cube of a tower on steel: P (h - 0.5) = 0.5 h + 0.3 M."""
    load = np.zeros((h, 3))
    load[-1, 0] = 1.0
    r = check([(z, 0, 0) for z in range(h)], load=load, mode="live")
    assert r.factor == pytest.approx((0.5 * h + 0.3 * J.moment()) / (h - 0.5), rel=1e-4)


def test_friction_holds_a_cube_on_a_wall():
    """A cube on the side of a wall slides when its weight passes mu times the magnets' clamp."""
    r = check([(1, 0, 0)], anchored=[(1, 0, -1)], joints=Joints(edges=False, cap=50.0))
    clamp = 8 * 4.5 + 0.3
    assert r.factor == pytest.approx(0.5 * clamp, rel=1e-4)


def test_floating_and_resting():
    assert check([(0, 0, 0)]).factor == pytest.approx(J.cap, rel=1e-4)
    r = check([(3, 0, 0)])
    assert r.factor == pytest.approx(0.0, abs=1e-6) and not r.stands
    assert r.moving.tolist() == [[3, 0, 0]]


def test_edge_contact_is_a_pin():
    """A cube held only by an edge to a fixed cube turns about that edge."""
    r = check([(2, 0, 1)], anchored=[(1, 0, 0)])
    assert r.factor == pytest.approx(0.0, abs=1e-5)


def test_live_load_on_a_structure_that_cannot_hold_itself():
    load = np.zeros((8, 3))
    load[-1, 2] = -1.0
    r = check([(1, 0, x) for x in range(8)], anchored=WALL, load=load, mode="live")
    assert r.factor == 0.0 and not r.stands


def test_batch_matches_single_and_grid():
    shapes = [{"cells": [(1, 0, x) for x in range(n)], "anchored": WALL} for n in range(1, 9)]
    shapes.append({"cells": [(z, 0, 0) for z in range(3)] + [(2, 0, x) for x in range(1, 5)]})
    batch = check_many(shapes)
    for spec, r in zip(shapes, batch):
        one = check(spec["cells"], anchored=spec.get("anchored", ()))
        assert r.factor == pytest.approx(one.factor, rel=1e-5)
        assert r.converged
    occupied = np.zeros((4, 2, 6), dtype=bool)
    for z, y, x in shapes[-1]["cells"]:
        occupied[z, y, x] = True
    assert check_grid(occupied).factor == pytest.approx(batch[-1].factor, rel=1e-5)


def test_two_components_take_the_weaker():
    cells = [(0, 0, 0)] + [(z, 0, 5) for z in range(3)] + [(2, 0, x) for x in range(6, 10)]
    r = check(cells)
    assert r.factor == pytest.approx(0.3 * J.moment() / (16 / 2 - 1.5), rel=1e-4)
    assert [0, 0, 0] not in r.moving.tolist()


def test_gusset_moves_the_weak_joint_out():
    """An arm of 8 from a wall fails at the wall; a one-cube gusset under its root moves the
    failure to the joint past the gusset, where the arm beyond is 7 long."""
    arm = [(4, 0, x) for x in range(8)]
    wall = [(z, 0, -1) for z in range(5)]
    assert check(arm, anchored=wall).factor == pytest.approx(2 * J.moment() / 64, rel=1e-4)
    r = check(arm + [(3, 0, 0)], anchored=wall)
    assert r.factor == pytest.approx(2 * J.moment() / 49, rel=1e-4)
    assert r.opening == [((4, 0, 1), (4, 0, 0), "face")]
