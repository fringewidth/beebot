"""Cube geometry rules that the design rests on (hardware/cube50/DESIGN.md). The CAD is the 60 mm cube;
the module keeps the cube50 name."""

import importlib.util
import math
from pathlib import Path

import pytest

CAD = Path(__file__).resolve().parents[1] / "cube50" / "cad" / "cube50_cad.py"


@pytest.fixture(scope="module")
def cad():
    pytest.importorskip("cadquery")
    spec = importlib.util.spec_from_file_location("cube50_cad", CAD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pins_meet_slots_at_every_quarter_turn(cad):
    """Two faces meet mirrored: a partner's angle phi lands on alpha - phi. Every pin must land on a
    slot and never on a pin, for all four relative quarter turns."""
    slots = {round(a % 360, 6) for a in cad.SLOT_ANGLES}
    pins = {round(a % 360, 6) for a in cad.PIN_ANGLES}
    for alpha in (0, 90, 180, 270):
        landed = {round((alpha - a) % 360, 6) for a in cad.PIN_ANGLES}
        assert landed == slots and not landed & pins


def test_sled_stroke_puts_the_axle_on_the_edge(cad):
    assert cad.PR[0] + cad.SLIDE[0] == pytest.approx(cad.P[0])
    assert cad.PR[1] + cad.SLIDE[1] == pytest.approx(cad.P[1])
    assert cad.P == (cad.H, cad.H)


def test_hinge_parts_stay_inside_the_edge_channel(cad):
    """Deployed, the knuckle and the axle gear turn inside the channel four chamfers leave."""
    channel = cad.C / math.sqrt(2)
    assert cad.KR < channel
    assert cad.tip_r(cad.N_AXLE) < channel


def test_stowed_hinge_is_not_proud(cad):
    r = max(cad.KR, cad.tip_r(cad.N_AXLE))
    assert cad.PR[0] + r <= cad.H - 0.2
    assert cad.PR[1] + r <= cad.H - 0.2
    assert max(cad.KNUCKLE_Y, cad.AXLE_Y[1], -cad.AXLE_Y[0], cad.L1_Y[1]) <= cad.EDGE_END


def test_hinge_train_meshes_and_gives_the_torque(cad):
    pr = cad.pitch_r
    assert math.dist(cad.G2, cad.GC) == pytest.approx(pr(cad.N_MOTOR) + pr(cad.N_BIG), abs=1e-6)
    assert math.dist(cad.GC, cad.GI) == pytest.approx(pr(cad.N_SMALL) + pr(cad.N_IDLER), abs=1e-6)
    assert math.dist(cad.GI, cad.PR) == pytest.approx(pr(cad.N_IDLER) + pr(cad.N_AXLE), abs=1e-6)
    assert cad.HINGE_RATIO == pytest.approx(25 / 12)
    assert 0.245 * cad.HINGE_RATIO * 0.95**3 >= 0.42  # 1.5 x the 0.278 N m side-neighbour pivot peak
    # deployed, the compound's big gear and its pocket stay 0.8 mm under the vertical-edge chamfer
    x_tip = cad.GC[0] + cad.RETRACT + cad.tip_r(cad.N_BIG) + 0.4
    assert x_tip + cad.L1_Y[1] + 0.5 <= 2 * cad.H - cad.C - 0.8 * math.sqrt(2)


def test_disc_train_meshes_and_carries_the_cap(cad):
    m0, m1, m2 = cad.DT_M
    n = cad.DT
    pr = cad.pitch_r
    assert math.hypot(*cad.C2_XY) == pytest.approx(pr(n[4], m2) + pr(n[5], m2), abs=1e-6)
    assert math.dist(cad.C1_XY, cad.C2_XY) == pytest.approx(pr(n[2], m1) + pr(n[3], m1), abs=1e-6)
    assert math.dist(cad.C1_XY, cad.DISC_MOTOR_XY) == pytest.approx(pr(n[0], m0) + pr(n[1], m0), abs=1e-6)
    assert cad.DISC_RATIO == pytest.approx(14.29, abs=0.01)
    cap = cad.DISC_MOTOR["rating_Nm"] * cad.DISC_RATIO * 0.95**3
    assert cap >= 2.2
    # the gears that share a layer with the centre shaft clear it
    assert math.hypot(*cad.C1_XY) - cad.tip_r(n[1], m0) - cad.SHAFT_D / 2 >= 0.5
    assert math.hypot(*cad.C2_XY) - cad.tip_r(n[3], m1) - cad.SHAFT_D / 2 >= 0.5
    # Lewis bending at the cap stays inside hardened steel (400 MPa, guess)
    rating = cad.DISC_MOTOR["rating_Nm"]
    assert cad.lewis(rating, n[0], m0, cad.LC[1] - cad.LC[0]) <= 400
    assert cad.lewis(rating * n[1] / n[0] * 0.95, n[2], m1, cad.LB[1] - cad.LB[0]) <= 400
    assert cad.lewis(cap / (n[5] / n[4]) / 0.95, n[4], m2, cad.LA[1] - cad.LA[0]) <= 400


def test_drive_layers_fit_under_the_foot(cad):
    assert cad.BRIDGE_Z[1] < cad.LA[0] < cad.LA[1] < cad.LB[0] < cad.LB[1] < cad.LC[0] < cad.LC[1] < cad.PLATE_Z[0]
    assert cad.PLATE_Z[1] <= cad.ZC
    assert cad.BRIDGE_Z[0] - 1.0 >= cad.CELL[0] / 2 + 0.5  # the shaft's E-clip clears the cell


def test_mechanism_stack_fits_the_disc_plate(cad):
    assert cad.BAY_Z0 > cad.DISC_Z0
    assert cad.BEARING_Z[1] <= cad.DISC_Z0
    assert cad.ZC + cad.GAP == pytest.approx(cad.ZF)
    assert cad.CAM_R[0] > cad.BEARING_LIP[1] + 0.3  # the cam ring clears the bearing's lip


def test_lock_holds_the_bond(cad):
    """3.2 N m at the lock, carried by the pin bearing on the 0.8 mm ring (304 yields at 215 MPa)."""
    engaged = cad.LOCK_RING[2] - 0.05
    force = 215 * cad.LOCK_PIN_D * engaged
    assert force * cad.LOCK_R / 1000 >= 1.5 * 3.2


def test_face_magnet_pockets_keep_a_printable_wall(cad):
    wall = 2 * cad.MAG_R * math.sin(math.radians(min(cad.MAG_DELTA, 45 - cad.MAG_DELTA))) - cad.MAG_D - 0.1
    assert wall >= 0.6
