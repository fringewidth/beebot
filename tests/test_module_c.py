import math

import numpy as np

from beebot.hardware.module_c import (
    DISC_TORQUE,
    MASS,
    PROUD,
    SIDE,
    WHEEL_TORQUE,
    Placement,
    Planner,
    World,
    nearest_lattice,
    orientation,
    packing,
    unit,
)


def test_a_cube_builds_at_120_g_with_clipped_disc_and_wheel_motors():
    w = World([Placement((0.0, 0.0))])
    m = w.model
    assert abs(m.body_subtreemass[w.body[0]] - 0.120) < 1e-9 and abs(MASS - 0.120) < 1e-9
    ranges = {m.actuator(a).name[0]: tuple(m.actuator_ctrlrange[a]) for a in range(m.nu)}
    assert ranges == {"d": (-DISC_TORQUE, DISC_TORQUE), "w": (-WHEEL_TORQUE, WHEEL_TORQUE)}
    assert tuple(m.jnt_range[m.joint("c0s").id]) == (0.0, PROUD)


def test_orientation_puts_the_named_face_down():
    for down in ("+x", "-x", "+y", "-y", "+z", "-z"):
        rot = orientation(down, 0.7)
        assert np.allclose(rot @ unit(down), [0, 0, -1])
        assert abs(np.linalg.det(rot) - 1) < 1e-9


def test_a_lone_cube_rests_on_its_wheel_and_one_edge_and_drives_straight():
    w = World([Placement((0.0, 0.0), yaw=0.4)])
    P = Planner(w)
    w.run(0.3)
    P.tip([0], w.rolling(0))
    assert abs(w.data.qpos[w.slide_q[0]] - PROUD) < 0.0002  # its own weight does not push the wheel in
    assert abs(w._tilt(0) - math.degrees(math.atan(PROUD / (SIDE / 2)))) < 0.4  # leans on one bottom edge
    d, start = w.rolling(0), w.pos(0)
    w.set_wheels({0: 15.0})
    w.run(1.0)
    w.stop([0])
    moved = w.pos(0) - start
    assert moved @ d > 0.13 and abs(np.cross(d, moved)[2]) < 0.002


def test_a_bond_forms_only_inside_the_capture_range():
    w = World([Placement((0.0, 0.0)), Placement((0.0, SIDE + 0.006))])  # 6 mm gap: out of range
    w.run(0.2)
    assert not w.bond(0, 1) and w.failed_docks == 1 and not w.bonds
    w = World([Placement((0.0, 0.0)), Placement((0.0, SIDE + 0.001), yaw=math.radians(2))])
    w.run(0.2)
    assert w.bond(0, 1) and w.failed_docks == 0
    w.run(0.5)
    assert nearest_lattice(w.rot(0).T @ w.rot(1))[1] < 1.0  # the weld pulls it flush


def test_a_latch_refuses_edges_more_than_3_mm_apart():
    w = World([Placement((0.0, 0.0)), Placement((0.0, SIDE + 0.008))])
    w.run(0.2)
    edge = w.pos(0) + np.array([0, SIDE / 2, SIDE / 2])
    assert w.pivot(1, edge, [1, 0, 0], math.pi, support=0) == "unlatched"


def test_floor_roll_and_climb_onto_a_bonded_neighbour_and_back():
    pls = [Placement((0.0, k * (SIDE + 0.0005))) for k in range(3)]
    w = World(pls, bonds=[(0, 1, None), (1, 2, None)])
    P = Planner(w)
    w.run(0.3)
    assert P.climb(2, 1) == "arrived"
    assert abs(w.pos(2)[2] - w.pos(1)[2] - SIDE) < 0.003 and frozenset((1, 2)) in w.bonds
    out = P.climb_down(2, 1, w.pos(1) - w.pos(0))
    assert out == "arrived" and abs(w.pos(2)[2] - w.pos(1)[2]) < 0.003 and frozenset((1, 2)) in w.bonds
    assert max(p[1] for p in w.pivot_peaks) <= 0.1 + 1e-9
    lone = World([Placement((0.0, 0.0), down="+y")])
    lone.run(0.2)
    start = lone.pos(0)
    assert lone.floor_roll(0, [1, 0, 0]) == "arrived"
    assert abs(lone.pos(0)[0] - start[0] - SIDE) < 0.003


def test_packing_boxes_fit_without_overlap():
    result = packing()
    assert result["fits"] and not result["overlaps"] and not result["outside"]
    assert 0 < result["free_cm3"] < result["interior_cm3"]
