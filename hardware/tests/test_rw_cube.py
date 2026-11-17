"""Fast checks of the reaction-wheel cube model (hardware/src/beebot/hardware/rw_cube.py)."""
import math

import magpylib as mp
import numpy as np
import pytest

from beebot.hardware import rw_cube as rc
from beebot.hardware.mmsim.mjcoupling import DipoleCoupling, dipole_pairs, total_momentum


@pytest.fixture(scope="module")
def design():
    return rc.Design(catch_T=1e-3)


def test_dipole_pairs_match_magpylib_spheres():
    rng = np.random.default_rng(1)
    r, br = 0.5e-3, 1.45
    m = br / rc.MU0 * 4 / 3 * math.pi * r**3
    P = np.array([[0, 0, 0], [1.1e-3, 0.2e-3, 0.1e-3], [0.3e-3, 1.4e-3, -0.5e-3]])
    ax = rng.normal(size=(3, 3))
    ax /= np.linalg.norm(ax, axis=1)[:, None]
    F, T, _ = dipole_pairs(P, ax * m)
    for j in range(3):
        tgt = mp.magnet.Sphere(polarization=br * ax[j], diameter=2 * r, position=P[j])
        src = [mp.magnet.Sphere(polarization=br * ax[i], diameter=2 * r, position=P[i]) for i in range(3) if i != j]
        Fm, Tm = mp.getFT(src, tgt, pivot=P[j])
        Fm = np.asarray(Fm).reshape(-1, 3).sum(0)
        Tm = np.asarray(Tm).reshape(-1, 3).sum(0)
        assert np.abs(F[j] - Fm).max() < 1e-3 * np.abs(Fm).max()
        assert np.abs(T[j] - Tm).max() < 1e-6 * np.abs(Tm).max()
    # pairwise construction conserves momentum
    assert np.abs(F.sum(0)).max() < 1e-12
    assert np.abs((T + np.cross(P, F)).sum(0)).max() < 1e-15


def test_numba_kernel_matches_numpy_coupling(design):
    var = rc.nominal_variation(design, 2)
    sim = rc.Sim(design, [rc.CubeSpec((0, 0, 0), np.eye(3)), rc.CubeSpec((1, 0, 0), rc.ROTS[5])], var)
    sim.dip.compute()
    xf = np.zeros((sim.m.nbody, 6))
    sim.dip.accumulate(xf)
    xf2 = np.zeros((sim.m.nbody, 6))
    F = np.zeros((len(sim.dip.bodies), 3))
    B = F.copy()
    d = sim.data
    rc._dipole_kernel(d.xpos, d.xmat, d.xipos, sim.dip.bodies, sim.dip.local_axis, sim.dip.moment, sim.dip.ii, sim.dip.jj, xf2, F, B)
    assert np.abs(xf - xf2).max() < 1e-9 * np.abs(xf).max()


def test_mass_and_flywheel_match_concepts(design):
    der = rc.derive(design)
    assert 0.80e-3 < der.m_total < 0.90e-3  # concepts.md: 0.85 g
    assert 3.5e-10 < der.rotor_i_axial < 4.2e-10  # scaling.json i_f 3.91e-10
    assert der.feasible
    assert der.motor["k_t"] > 0 and der.motor["R_m"] > 0


def test_rotor_core_far_field_dipole_within_15_percent(design):
    der = rc.derive(design)
    rcore = design.core_frac * design.rotor_r
    mag = mp.magnet.Cylinder(polarization=(rc.BR_N52, 0, 0), dimension=(2 * rcore, design.rotor_t))
    p = np.array([[2.0e-3, 2.5e-3, 1.5e-3]])  # a corner sphere seen from rotor X's centre, about 3.5 mm away
    b_true = np.asarray(mag.getB(p[0])).reshape(3)
    r = p[0]
    d = np.linalg.norm(r)
    u = r / d
    m = np.array([der.core_moment, 0, 0])
    b_dip = 1e-7 * (3 * u * (m @ u) - m) / d**3
    assert np.linalg.norm(b_dip - b_true) / np.linalg.norm(b_true) < 0.15


def test_catch_conserves_angular_momentum_in_free_space(design):
    var = rc.nominal_variation(design, 1)
    sim = rc.Sim(design, [rc.CubeSpec((0, 0, 2), np.eye(3))], var, floor=False)
    sim.m.opt.gravity[:] = 0
    sim.run(0.002)
    sim.spin(0, "y", 3000.0)
    _, L0 = total_momentum(sim.m, sim.data)
    sim.catch(0, "y", 0.1e-3)
    sim.run(0.016)  # the pawl locks 12 catch durations after the hit
    _, L1 = total_momentum(sim.m, sim.data)
    assert sim.pawl is None  # caught and locked
    assert abs(sim.data.qvel[sim.rot_dof[(0, "y")]]) < 1.0
    assert np.linalg.norm(L1 - L0) < 0.01 * np.linalg.norm(L0)


def test_nominal_floor_roll_reaches_one_cell(design):
    tr = rc.run_move(design, "roll_floor", 0, nominal=True, retries=0, k_override=1.1, rot_index=0)
    assert tr.cls == "reached", tr.metrics
    assert tr.metrics["offset_mm"] < 0.25 * design.L * 1e3


def test_too_little_impulse_stays(design):
    tr = rc.run_move(design, "roll_floor", 0, nominal=True, retries=0, k_override=0.8, rot_index=0)
    assert tr.cls == "stayed"


def test_spinup_power_limited_and_reaches_speed(design):
    der = rc.derive(design)
    out = rc.spinup(design, der, 3000.0)
    assert out["ok"] and 0 < out["t"] < 60 and out["energy"] > 0.5 * der.rotor_i_axial * 3000.0**2
    assert not rc.spinup(design, der, 1.5 * design.omega_cap)["ok"]


def test_bond_closes_from_small_gap(design):
    tr = rc.run_bond(design, 0, nominal=True, gap=0.05, lateral=0.0, yaw=0.0)
    assert tr.cls == "closed"
    tr = rc.run_bond(design, 0, nominal=True, gap=0.8, lateral=0.0, yaw=0.0)
    assert tr.cls == "gap_left"


def test_face_down_and_rotations():
    assert len(rc.ROTS) == 24
    assert rc.face_down(np.eye(3)) == "-z"
    assert all(rc.rotor_along(Rm, np.array([0, 1.0, 0]))[0] is not None for Rm in rc.orientations_for_axis([0, 1.0, 0]))
    assert len(rc.orientations_for_axis([0, 1.0, 0])) == 16
