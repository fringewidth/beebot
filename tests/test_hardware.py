import numpy as np
import torch

from beebot.hardware import physics, structure
from beebot.hardware.design import Capability, Design
from beebot.vla.voxel.world import STAY, world_from_arrays


def overhang(strength, on, off):
    """A row of modules on a plateau (x < 12) running `off` modules past its edge; returns which
    modules come loose, as a string over the row."""
    occupied = np.zeros((4, 4, 24), dtype=bool)
    solid = np.zeros_like(occupied)
    solid[0, :, :12] = True
    occupied[1, 1, 12 - on : 12 + off] = True
    loose = structure.loose_modules(occupied, solid, np.zeros(occupied.shape), strength, False, 0.0)
    return "".join(str(int(v)) for v in loose[1, 1, 12 - on : 12 + off])


def test_overhang_tips_without_enough_counterweight():
    assert overhang(100, 6, 4) == "0" * 10
    assert overhang(100, 5, 4) == "0" * 9  # 12.5 against 8 module-lengths of overturning moment
    assert overhang(100, 3, 4) == "1" * 7  # 4.5 against 8: the whole row tips


def test_root_connector_breaks_past_n_squared():
    # 4 modules past the edge put a moment of 8 on the root face; its magnet, half a module from
    # the pivot edge, then pulls 16
    assert overhang(17, 6, 4) == "0" * 10
    assert overhang(15, 6, 4) == "0" * 6 + "1" * 4


def test_no_connector_no_overhang():
    assert overhang(0, 6, 1) == "0" * 6 + "1"


def test_tower_stands_and_hovering_needs_thrust():
    occupied = np.zeros((6, 3, 3), dtype=bool)
    occupied[:4, 1, 1] = True
    solid = np.zeros_like(occupied)
    assert not structure.loose_modules(occupied, solid, np.zeros(occupied.shape), 0.0, False, 0.0).any()
    floating = np.zeros_like(occupied)
    floating[3, 1, 1] = True
    assert structure.loose_modules(floating, solid, np.zeros(occupied.shape), 0.0, False, 0.0)[3, 1, 1]
    assert not structure.loose_modules(floating, solid, np.zeros(occupied.shape), 0.0, True, 1.5).any()
    heavy = np.zeros(occupied.shape)
    heavy[3, 1, 1] = 1.0  # carrying one more module weight than its thrust of 1.5 can lift
    assert structure.loose_modules(floating, solid, heavy, 0.0, True, 1.5)[3, 1, 1]


def crawler(strength=50.0, torque=50.0, switchable=False):
    return Capability(strength=strength, torque=torque, thrust=0.0, cheese=4.0, switchable=switchable)


def test_crawler_needs_a_surface_and_torque():
    mass = np.zeros((4, 5, 5))
    mass[0, 2, 2] = 1.0
    world = world_from_arrays(mass, gravity=True)
    allowed = physics.allowed_moves(world, crawler())
    up = physics.DIR_TABLE.tolist().index([1, 0, 0])
    side = physics.DIR_TABLE.tolist().index([0, 0, 1])
    assert allowed[0, side] and not allowed[0, up] and allowed[0, STAY]  # nothing to climb on
    weak = physics.allowed_moves(world, crawler(torque=0.4))
    assert not weak[0, side]


def test_permanent_bonds_need_prying():
    mass = np.zeros((4, 5, 5))
    mass[0, 2, 1:4] = 1.0  # the middle module has two bonds
    world = world_from_arrays(mass, gravity=True)
    side = physics.DIR_TABLE.tolist().index([0, 1, 0])
    middle = 1  # robot_list order is flat index order
    assert not physics.allowed_moves(world, crawler(strength=50, torque=30))[middle, side]
    assert physics.allowed_moves(world, crawler(strength=50, torque=60))[middle, side]
    assert physics.allowed_moves(world, crawler(strength=50, torque=1, switchable=True))[middle, side]


def test_designs_scale_like_square_cube():
    small = Design(0.01, "magnet", 0.05, 0.1).capability()
    large = Design(0.10, "magnet", 0.05, 0.1).capability()
    assert abs(small.strength / large.strength - 10) < 1e-6  # strength per weight goes as 1/L
    assert abs(small.cheese - large.cheese) < 1e-9


def test_step_keeps_mass():
    mass = np.zeros((6, 6, 6))
    mass[0, 1:5, 1:5] = 1.0
    world = world_from_arrays(mass, gravity=True)
    codes = torch.randint(0, STAY + 1, world.mass.shape)
    after, _ = physics.step(world, codes, crawler(), torch.rand(world.mass.shape))
    assert after.mass.sum() == world.mass.sum()
