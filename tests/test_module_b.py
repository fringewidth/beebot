import math

import mujoco
import numpy as np

from beebot.hardware.module_b import HOLD, PITCH, Module, Rig, bond_rotation, build, lattice, poses


def test_a_lone_module_builds_with_two_torque_limited_drives():
    model = mujoco.MjModel.from_xml_string(build([Module()]))
    assert model.nu == 2 and model.njnt == 3  # free joint and two disc hinges
    assert np.allclose(model.actuator_forcerange, [[-HOLD, HOLD]] * 2)
    assert np.all(model.actuator_forcelimited) and np.all(model.actuator_ctrllimited)
    assert abs(model.body_subtreemass[1] - 0.120) < 1e-9  # 110 g body and two 5 g discs


def test_bond_rotation_puts_the_mate_face_against_the_parent_face():
    for face in ("+x", "-y", "+z"):
        for mate in ("-x", "+y", "-z"):
            for turn in range(4):
                rot = bond_rotation(face, mate, turn)
                assert abs(np.linalg.det(rot) - 1) < 1e-9
                normal = {"x": 0, "y": 1, "z": 2}
                n_p = np.zeros(3)
                n_p[normal[face[1]]] = 1 if face[0] == "+" else -1
                n_c = np.zeros(3)
                n_c[normal[mate[1]]] = 1 if mate[0] == "+" else -1
                assert np.allclose(rot @ n_c, -n_p)


def test_lattice_cells_build_where_they_were_asked_for():
    cells = [
        {"at": (0, 0, 0), "axis": "+z", "mount": "n"},
        {"at": (0, 0, 1), "axis": "+y", "parent": 0},
        {"at": (0, 1, 1), "axis": "+z", "parent": 1},
        {"at": (1, 1, 1), "axis": "+y", "parent": 2},
    ]
    modules = lattice(cells)
    rots, centres = poses(modules)
    for cell, rot, centre in zip(cells, rots, centres):
        assert np.allclose((centre - centres[0]) / PITCH, cell["at"])
    assert np.allclose(rots[3][:, 0], [0, 1, 0])  # module 3's disc axis points along world y
    rig = Rig(modules)
    rig.run(0.2)
    for k in range(4):
        assert np.allclose(rig.position(f"m{k}b"), centres[k], atol=2e-3)  # it holds its shape under gravity


def test_the_drive_never_exceeds_its_holding_torque():
    cells = [{"at": (0, 0, 0), "axis": "+y", "mount": "b"}] + [{"at": (k, 0, 0), "axis": "+z", "parent": k - 1} for k in range(1, 6)]
    cells[1]["axis"] = "+x"  # module 1's disc carries the rest: a five-module arm out sideways
    cells[2] = {"at": (2, 0, 0), "axis": "+z", "parent": 1}
    rig = Rig(lattice(cells))
    rig.command(targets={"1p": 0.5}, seconds=0.5)
    rig.run(1.0)
    assert rig.peak.max() <= HOLD + 1e-9
    assert rig.peak[rig.index["1p"]] > 0.79  # 4 x 120 g at 52-210 mm needs about 0.65 N m, plus the swing


def test_a_lone_module_rolls_and_turns_on_the_spot():
    rig = Rig([Module()])
    rig.run(0.3)
    start = rig.position("m0b")
    rig.command(speeds={"0p": 5.8, "0n": 5.8}, ramp=0.3)
    rig.run(1.5)
    travelled = np.linalg.norm((rig.position("m0b") - start)[:2])
    assert 0.17 < travelled < 0.23  # 55 rpm on a 52 mm disc is 0.15 m/s
    yaw = rig.heading("m0b")
    rig.command(speeds={"0p": 0.0, "0n": 0.0}, ramp=0.3)
    rig.run(0.5)
    rig.command(targets={"0p": rig.angle("0p") - 1.0, "0n": rig.angle("0n") + 1.0}, seconds=0.8)
    rig.run(1.0)
    assert abs(rig.heading("m0b") - yaw) > math.radians(45)


def test_with_side_plates_a_lone_module_cannot_roll():
    rig = Rig(lattice([{"at": (0, 0, 0), "axis": "+x", "z0": 0.0265}]), plates=True)
    rig.run(0.3)
    start = rig.position("m0b")
    rig.command(speeds={"0p": 5.8, "0n": 5.8}, ramp=0.3)
    rig.run(1.0)
    assert np.linalg.norm((rig.position("m0b") - start)[:2]) < 0.005
