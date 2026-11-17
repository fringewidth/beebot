"""Tests for the millimetre-scale simulation toolchain (beebot.hardware.mmsim). CPU only, about two minutes."""
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from beebot.hardware.mmsim import circuit, coil, demo, opt, toys, validation
from beebot.hardware.mmsim.fem import pull_force_axisym
from beebot.hardware.mmsim.magnetics import (Coil, Magnet, force_coil_coil, force_coil_magnet,
                                             pull_force_magnet_magnet, pull_force_to_steel, wrench)
from beebot.hardware.mmsim.mjcoupling import CylPairTable, MagneticCoupling, SurfaceForces, hybrid_wrench

MU0 = 4e-7 * np.pi


# ------------------------------------------------------------------ magnetics

def test_pole_model_matches_getft():
    for d, h, br in [(1e-3, 1e-3, 1.45), (3e-3, 2e-3, 1.45)]:
        ref = validation.pole_model_force(d, h, br)
        sim = pull_force_magnet_magnet("cyl", (d, h), br, 0.0, mag_mesh=(24, 4, 8))
        assert abs(sim - ref) / ref < 0.04


def test_image_force_attractive_and_decays_with_gap():
    f0 = pull_force_to_steel("cyl", (3e-3, 2e-3), 1.4, 0.0, mag_mesh=(12, 3, 4))
    f1 = pull_force_to_steel("cyl", (3e-3, 2e-3), 1.4, 0.5e-3, mag_mesh=(12, 3, 4))
    assert f0 > f1 > 0


def test_coil_on_magnet_matches_dipole_formula():
    radius, current, z = 5e-3, 1.0, 6e-3
    theta = np.linspace(0, 2 * np.pi, 361)
    verts = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)])
    c = Coil(verts, current)
    side, br = 0.4e-3, 1.4
    m = Magnet("cuboid", (side,) * 3, br, (0, 0, z))
    F, _ = force_coil_magnet(c, m, mag_mesh=(3, 3, 3))
    moment = br * side**3 / MU0
    dbz = -3 * MU0 * current * radius**2 * z / (2 * (radius**2 + z**2) ** 2.5)
    assert abs(F[2] - moment * dbz) / abs(moment * dbz) < 0.03


def test_coil_coil_forces_nearly_reciprocal():
    sp = coil.Spiral(3, 3e-3, 100e-6, 100e-6, 18e-6)
    v = coil.spiral_vertices(sp)
    a = Coil(v, 1.0, (0, 0, 0))
    b = Coil(v, 1.0, (0.2e-3, 0, 1e-3))
    Fb, _ = force_coil_coil(a, b, pivot=b.pos, coil_mesh=120)
    Fa, _ = force_coil_coil(b, a, pivot=a.pos, coil_mesh=120)
    assert np.linalg.norm(Fa + Fb) < 0.1 * np.linalg.norm(Fb)


# ------------------------------------------------------------------ fem

def test_fem_linear_steel_close_to_image_method():
    res = pull_force_axisym(5e-3, 2e-3, 1.45, 100e-6, mu_r=1000.0, maxh_fine=2e-3 / 24, mu_magnet=1.0)
    img = pull_force_to_steel("cyl", (5e-3, 2e-3), 1.45, 100e-6, mu_r=1000.0, mag_mesh=(24, 4, 8))
    assert abs(res["force"] - img) / img < 0.08


def test_fem_saturation_reduces_force():
    lin = pull_force_axisym(5e-3, 2e-3, 1.45, 100e-6, mu_r=1000.0, maxh_fine=2e-3 / 12)
    nl = pull_force_axisym(5e-3, 2e-3, 1.45, 100e-6, mu_r=None, maxh_fine=2e-3 / 12)
    assert nl["force"] < 0.9 * lin["force"]


# ------------------------------------------------------------------ coil

def test_track_length_matches_aebischer():
    sp = coil.Spiral(4, 50e-3, 12 * 25.4e-6, 28 * 25.4e-6, 35e-6)
    ref = 4 * 4 * sp.inner() + (4 * 16 - 20) * sp.pitch
    assert abs(coil.track_length(sp) - ref) / ref < 0.01


def test_inductance_against_published_subset():
    rows = validation.inductance_rows(methods=("flux", "mohan"))
    sub = [r for r in rows if r.id.startswith(("L_aeb01", "L_aeb07", "L_aeb13"))]
    assert len(sub) == 6
    assert all(r.passed for r in sub)


def test_resistance_scales_with_layers_and_thickness():
    a = coil.Spiral(5, 4e-3, 75e-6, 75e-6, 18e-6, layers=1)
    b = coil.Spiral(5, 4e-3, 75e-6, 75e-6, 18e-6, layers=2)
    c = coil.Spiral(5, 4e-3, 75e-6, 75e-6, 35e-6, layers=1)
    assert coil.resistance(b) == pytest.approx(2 * coil.resistance(a), rel=1e-9)
    assert coil.resistance(c) == pytest.approx(coil.resistance(a) * 18 / 35, rel=1e-9)
    assert coil.ampere_turns(b, 3.0)["NI"] > 0


def test_onderdonk_adiabatic_limit():
    rows = validation.thermal_rows()
    assert all(r.passed for r in rows)


def test_safe_current_decreases_with_duration():
    sp = coil.Spiral(5, 4e-3, 75e-6, 75e-6, 18e-6, layers=2)
    cur = [c for _, c in coil.safe_current_table(sp, [1e-3, 1e-2, 1e-1, 1.0], 100.0)]
    assert all(cur[i] > cur[i + 1] for i in range(3))


# ------------------------------------------------------------------ circuit

def test_circuit_energy_balance_and_peak():
    rows = validation.circuit_rows()
    assert all(r.passed for r in rows)


def test_cell_droop_follows_internal_resistance():
    cell = circuit.ceracharge()
    res = circuit.simulate_pulse(cell, 10.0, 1e-6, 1e-3, circuit.Driver(0.1), t_total=2e-3)
    assert res["droop_at_peak"] == pytest.approx(res["i_peak"] * 200.0, rel=0.02)


# ------------------------------------------------------------------ coupling

@pytest.fixture(scope="module")
def small_table():
    return CylPairTable(1e-3, 1e-3, 1.45, n_rho=12, n_gap=14, n_tilt=3, mag_mesh=(12, 3, 4))


def test_lookup_agrees_with_direct(small_table):
    a = demo.edge_magnets(+1)
    b = [m.with_pose(np.array(m.pos) + [5e-3 + 1.2e-3, 0.3e-3, 0], R.from_euler("z", 3, degrees=True) * m.rot)
         for m in demo.edge_magnets(-1)]
    com = np.array([6.2e-3, 0.3e-3, 0])
    Fd, Td = wrench(a, b, pivot=com, mag_mesh=(12, 3, 4))
    Fl, Tl = hybrid_wrench(small_table, a, b, com, mag_mesh=(12, 3, 4))
    assert np.linalg.norm(Fl - Fd) < 0.1 * np.linalg.norm(Fd)
    assert np.linalg.norm(Tl - Td) < 0.15 * np.linalg.norm(Td) + 1e-7


def test_wrench_balance_exact_and_dynamic_momentum():
    model = demo.build_model(gap=1.0e-3, offset_y=0.5e-3, yaw_deg=8.0, gravity=False, floor=False)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    cpl = MagneticCoupling(model, data, demo.attached_pair())
    cpl.apply()
    f = data.xfrc_applied
    assert np.linalg.norm(f[1, :3] + f[2, :3]) < 1e-12 * np.linalg.norm(f[1, :3])
    # angular momentum rate about world origin: sum(tau + r x F) = 0
    tot = sum(f[b, 3:] + np.cross(data.xipos[b], f[b, :3]) for b in (1, 2))
    assert np.linalg.norm(tot) < 1e-10 * np.linalg.norm(f[2, 3:] + 1e-12) + 1e-14
    res = demo.run_momentum_test(n_steps=300, dt=1e-5)
    assert res["dp_max"] < 1e-9 * res["p_scale"]
    assert res["dL_max"] < 0.03 * (res["p_scale"] * 6e-3)


def test_seating_flush_and_stable_over_timesteps():
    surf = SurfaceForces(adhesion=2e-4, mu_s=0.5, mu_k=0.3)
    finals = []
    for dt in (5e-5, 2e-5):
        r = demo.run_seating(dt=dt, t_end=0.02, surface=surf)
        assert r["finite"]
        assert abs(r["gap_mm"]) < 0.02
        assert r["twist_deg"] < 0.1
        finals.append(r["gap_mm"])
    assert abs(finals[0] - finals[1]) < 0.02


def test_stiction_holds_cube_on_floor():
    model = demo.build_model(gap=20e-3, gravity=True)
    data = mujoco.MjData(model)
    surf = SurfaceForces(adhesion=0.0, mu_s=0.5, mu_k=0.3)
    mujoco.mj_forward(model, data)
    cpl = MagneticCoupling(model, data, [], surface=surf)
    cpl.bodies = []
    ib = 2
    x0 = data.xpos[ib].copy()
    for _ in range(200):
        cpl.apply()
        mujoco.mj_step(model, data)
    # push below the static limit: 0.3 of weight
    push = 0.3 * demo.MASS * 9.81
    for _ in range(2000):
        cpl.apply()
        data.xfrc_applied[ib, 0] += push
        mujoco.mj_step(model, data)
    held = abs(data.xpos[ib][0] - x0[0])
    assert held < 0.5e-3
    for _ in range(2000):
        cpl.apply()
        data.xfrc_applied[ib, 0] += 0.9 * demo.MASS * 9.81
        mujoco.mj_step(model, data)
    assert data.xpos[ib][0] - x0[0] > 3e-3


# ------------------------------------------------------------------ optimisation and Monte Carlo

def test_wilson_interval_known_values():
    lo, hi = opt.wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = opt.wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)


def test_workers_capped_at_six():
    assert opt._cap(64) == 6


def test_parallel_search_all_algorithms():
    r = opt.run_search(toys.zdt1, [0] * 3, [1] * 3, n_obj=2, algo="nsga2", pop=16, n_gen=8, workers=3)
    assert r.F.shape[1] == 2 and len(r.F) >= 2
    r = opt.run_search(toys.sphere, [0] * 3, [1] * 3, algo="ga", pop=16, n_gen=15, workers=3)
    assert r.F.min() < 0.05
    r = opt.run_search(toys.sphere, [0] * 3, [1] * 3, algo="cma", pop=12, n_gen=20, workers=3)
    assert r.F.min() < 0.01


def test_monte_carlo_toy_reproducible():
    p = {"magnet_pos_x": 0.0, "magnet_pos_y": 0.0, "magnetisation": 1.0, "print_dim_gap": 0.3e-3,
         "threshold": 0.165, "friction_mu": 0.5, "load": 0.075}
    a = opt.monte_carlo(toys.toy_latch, p, n=48, workers=3, seed=5)
    b = opt.monte_carlo(toys.toy_latch, p, n=48, workers=3, seed=5)
    assert a == b
    assert a["wilson_lo"] <= a["rate"] <= a["wilson_hi"]
