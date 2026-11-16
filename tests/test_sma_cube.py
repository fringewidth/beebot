"""Fast checks of the SMA cube model (src/beebot/hardware/sma_cube.py). The behaviour Monte Carlo
lives in scripts/module_phys/sma_cube_sim.py; these only pin the physics and the move rules."""

import math

import numpy as np
import pytest

from beebot.hardware.sma_cube import (
    CARRIERS,
    LATTICE_ROTATIONS,
    Design,
    Params,
    World,
    bfs_lone,
    key,
    magnet_forces,
    moves,
    rot90,
)

D = Design(L=14e-3, storage="lipo401012", joint_route="gear", leg_t=0.5e-3, pulse_caps=6)


def two_cubes(R1=np.eye(3)):
    h = D.h
    return World(D, [(np.array([0.0, 0.0, h]), np.eye(3)), (np.array([0.0, D.L, h]), R1)], Params(seed=1))


def test_latch_far_stronger_than_magnets_and_holds_cube_weights():
    b = D.bonds()
    w = D.mass * 9.81
    assert b["plain"]["tension_N"] > 20 * b["mag_plain"]["tension_N"]
    assert b["plain"]["tension_N"] > 500 * w
    assert b["disc"]["peel_Nm"] > 10 * b["mag_disc_plain"]["peel_Nm"]


def test_two_point_dipoles_attract_when_aligned():
    P = np.array([[0.0, 0, 0], [0.0, 0, 2e-3]])
    U = np.array([[0.0, 0, 1], [0.0, 0, 1]])
    F, _ = magnet_forces(P, U, np.array([1e-4, 1e-4]), np.array([0, 1], np.int64), np.array([False, False]), 1.0, 1.0)
    assert F[0, 2] > 0 > F[1, 2]
    assert F[0, 2] == pytest.approx(-F[1, 2])


def test_lone_cube_rights_itself_from_every_orientation():
    res = bfs_lone(radius=3)
    assert len(res) == 24
    assert all(r["home_any"] is not None and r["home_any"] <= 6 for r in res.values())


def test_structure_moves_use_x_legs_only_and_no_descend_by_side_leg():
    occ = {(0, 1, 0), (0, 2, 0)}
    seen = set()
    for R in LATTICE_ROTATIONS:
        for mv in moves(R.round().astype(int), (0, 0, 0), occ):
            seen.add(mv.kind)
            assert mv.kind != "DESCEND"
            if mv.kind == "CLIMB":
                assert mv.leg in ("XP", "XN")
                assert mv.delta == (0, 1, 1)
    assert "CLIMB" in seen


def test_climb_needs_a_braced_support():
    lone = {(0, 1, 0)}
    climbs = [mv for R in LATTICE_ROTATIONS for mv in moves(R.round().astype(int), (0, 0, 0), lone) if mv.kind == "CLIMB"]
    assert not climbs


def test_latch_only_between_parallel_x_plain_faces():
    w = two_cubes()
    w.run(0.2)
    assert w.latch(0, 1)
    crossed = two_cubes(rot90(np.array([0, 1, 0]), 1))  # second cube's x turned to vertical
    crossed.run(0.2)
    assert not crossed.can_latch(0, 1)


def test_retracted_corner_spheres_leave_less_than_one_weight_of_pull():
    w = two_cubes()
    w.run(0.2)
    full = w._apply_magnets()[1][w.mag_cube == 0].sum(0)[1]
    for c in ("pp", "pn"):  # the +y face's two bolts
        w.release(0, [c], True, level="full")
    assert w.run_until(lambda: np.all(w.rel[0, [0, 1]] >= D.rel_stroke - 1e-9), 30.0)
    rest = w._apply_magnets()[1][w.mag_cube == 0].sum(0)[1]
    assert full > 10 * D.mass * 9.81
    assert rest < D.mass * 9.81


def test_bolt_drive_and_tooth_count():
    r = D.release_actuator()
    assert r.drive > 0.3  # N: frees hooks against the magnets' pull along the stroke
    assert D.rel_steps == math.ceil(D.rel_stroke / 0.2e-3 - 1e-9)


def test_cost_and_wiring_are_reported_at_three_volumes():
    c = D.cost()
    assert c["10"] > c["1k"] > c["1M"] > 0
