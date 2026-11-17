import math

import pytest

from beebot.hardware.geared_joint import (
    GearedRig,
    CubeSpec,
    lattice,
    nominal_bonds,
    nominal_design,
    sma_design,
    sphere_pair_force,
    stepper_params,
    wolfrom_train,
    worm_last_train,
)


def test_worm_self_locks_only_below_the_friction_angle():
    worm = next(m for m in worm_last_train(14e-3).meshes if m.kind == "worm")
    assert worm.self_locking(mu=0.15)
    assert not worm.self_locking(mu=0.03)
    assert worm.back_efficiency(mu=0.15) <= 0.0 < worm.efficiency(mu=0.15) < 0.5


def test_wolfrom_efficiency_follows_the_ratio_loss_rule():
    train = wolfrom_train(14e-3)
    stage = train.meshes[-1]
    eps = 0.025
    assert stage.efficiency(eps=eps) == pytest.approx(1 / (1 + (stage.ratio - 1) * eps), rel=1e-9)
    assert train.ratio == pytest.approx(math.prod(m.ratio for m in train.meshes), rel=1e-9)


def test_nominal_gear_train_is_slow_and_lossy_at_14_mm():
    d = nominal_design(14e-3)
    table = d.table()
    assert d.train.ratio > 500
    assert 0.03 < d.train.efficiency() < 0.3
    assert table is not None


def test_sphere_pair_force_matches_the_dipole_limit_at_large_gap():
    d, gap = 1.0e-3, 5.0e-3
    r = d / 2
    m = 1.43 / (4e-7 * math.pi) * 4 / 3 * math.pi * r**3  # N52, Br about 1.43 T
    dist = d + gap
    dipole = 3 * 4e-7 * math.pi * m * m / (2 * math.pi * dist**4)
    f = sphere_pair_force(d, gap, derate=1.0, use_magpylib=False)
    assert f == pytest.approx(dipole, rel=0.1)


def test_bond_capacities_order_plain_over_disc():
    b = nominal_bonds(8.25e-3)
    plain, disc, aligned = b.plain(), b.disc(), b.disc_aligned()
    assert plain["peel_Nm"] > aligned["peel_Nm"] > disc["peel_Nm"] > 0
    assert disc["twist_Nm"] > plain["twist_Nm"]  # discs twist through pins


def test_sma_stepper_numbers_at_8_25_mm():
    st = sma_design(8.25e-3)
    t = st.table()
    assert 3.0 < t["step_deg"] < 6.0
    assert 15 < t["speed_deg_s"] < 40
    assert st.hold(True) > 1.5 * st.drive  # the lock holds more than the wire can drive
    assert 0.002 < st.efficiency < 0.03
    assert t["wire_stress_MPa"] == 172


def _rig(cells, L=8.25e-3, **kw):
    spec = CubeSpec(L=L, mass=1.29e-3, disc_radius_frac=0.45)
    bonds = nominal_bonds(L)
    joint = stepper_params(sma_design(L), bonds)
    return GearedRig(lattice(cells, spec.pitch), spec, joint, bonds, **kw), joint


def test_lattice_builds_and_bonds_have_the_right_kinds():
    cells = [
        {"at": (0, 0, 0), "axis": "+z", "mount": "n", "z0": 0.6 * 8.25e-3},
        {"at": (0, 0, 1), "axis": "+x", "parent": 0},
        {"at": (1, 0, 1), "axis": "+x", "parent": 1},
    ]
    rig, _ = _rig(cells)
    assert rig.model.nu == 6
    assert rig.bond_kind == ["disc", "disc"]


def test_stepper_moves_in_whole_teeth_and_holds_with_no_power():
    cells = [
        {"at": (0, 0, 0), "axis": "+z", "mount": "n", "z0": 0.6 * 8.25e-3},
        {"at": (0, 0, 1), "axis": "+x", "parent": 0},
        {"at": (1, 0, 1), "axis": "+y", "parent": 1},
    ]
    rig, joint = _rig(cells)
    rig.run(0.3)
    target = 4 * joint.step
    rig.command(targets={"0p": target}, seconds=0.5)
    rig.run(3.0)
    q = rig.angle("0p")
    assert abs(q - target) < joint.step
    assert rig.step_count >= 3 and rig.failed_steps == 0
    e0 = rig.energy
    rig.run(1.0)
    assert rig.energy == e0  # holding draws nothing
    assert abs(rig.angle("0p") - q) < 2 * joint.backlash + 1e-3


def test_axial_play_adds_slide_and_tilt_joints():
    cells = [
        {"at": (0, 0, 1), "axis": "+z", "mount": "b", "z0": 0.5 * 8.25e-3},
        {"at": (0, 1, 1), "axis": "+z", "parent": 0, "play": "n"},
        {"at": (0, 1, 0), "axis": "+z", "parent": 1},
    ]
    rig, _ = _rig(cells)
    names = {rig.model.joint(j).name for j in range(rig.model.njnt)}
    assert {"s1n", "t1ny", "t1nz"} <= names
