"""forms.py: the statics behind the form search, against sums done by hand."""

import math

import numpy as np

from beebot.hardware import forms

UNIT = forms.Unit(side=0.05, mass=0.2, twist=1.2, bend=5.0, pull=300.0)
TWIST, BEND, _ = UNIT.caps
BEND *= math.cos(math.pi / 16)  # bending is checked along 16 directions, inside the circle


def held(design, live, feet=None, angles=()):
    body = forms.Body(design, 0)
    p, r = body.pose(angles)
    return body, body.spare(p, r, UNIT.caps, np.eye(len(design))[live], feet(body, p, r) if feet else None)


def test_cantilever_fails_at_the_floor_bond():
    for n in (3, 6):
        design = ((0, 0, 0, 2), *((k, 0, 0, 0) for k in range(1, n + 1)))
        body, spare = held(design, n)
        assert body.joints == [] and body.loops == []  # cubes on their own twist axis move nothing
        assert math.isclose(spare, (BEND - n * (n + 1) / 2) / n)  # cube k sits k sides from the floor bond's centre
    assert held(((0, 0, 0, 2), *((k, 0, 0, 0) for k in range(1, 30))), 29)[1] is None


def test_shoulder_twist_limits_an_elbowed_arm():
    design = ((0, 0, 0, 2), (0, 0, 1, 1), (0, 1, 1, 0), (1, 1, 1, 0), (2, 1, 1, 0))
    body, spare = held(design, 4)
    assert [design[j] for j in body.joints] == [(0, 0, 1, 1), (0, 1, 1, 0)]  # base yaw and shoulder
    assert math.isclose(spare, (TWIST - 3) / 2)  # cubes at 1 and 2 sides from the shoulder axis, the load at 2
    assert math.isclose(held(design, 4, angles={2: math.pi})[1], spare)  # the same arm swung over
    p, r = body.pose({2: math.pi / 2})
    assert body.clash(p, r)  # swung down, it is under the floor


def test_free_standing_tips_instead_of_pulling_on_the_floor():
    stand = lambda body, p, r: body.lowest(p, r, np.array([0]))
    assert held(((0, 0, 0, 2), (0, 0, 1, 0), (1, 0, 1, 0), (2, 0, 1, 0)), 3, stand)[1] is None
    _, spare = held(((0, 0, 0, 2), (0, 0, 1, 0), (1, 0, 1, 0)), 2, stand)
    assert math.isclose(spare, (0.35 * 3 - 1) / (1 - 0.35))  # moments about the foot's edge at x = 0.35


def test_a_loop_bond_shares_the_load():
    ring = ((0, 0, 0, 2), (0, 0, 1, 0), (1, 0, 1, 2), (1, 0, 2, 0), (0, 0, 2, 2))
    body, spare = held(ring, 3)
    assert body.loops == [(4, 3)] and [ring[j] for j in body.joints] == [(0, 0, 1, 0)]
    cut = ring[:4] + ((0, 0, 2, 0),)  # the same cubes with the last bond missing
    assert spare >= held(cut, 3)[1] - 1e-6


def test_a_bond_needs_a_disc():
    assert not forms.Body(((0, 0, 0, 1), (1, 0, 0, 1)), 0).whole  # two y-axis cubes side by side along x
    assert forms.Body(((0, 0, 0, 0), (1, 0, 0, 1)), 0).whole


def test_grow_keeps_mirror_images_and_one_name():
    mirrors = ((0, 1), (1, 1))
    seed = forms.settle(forms.images((0, 0, 0, 2), mirrors), mirrors)
    assert len(seed) == 4
    assert all(len(d) == 8 and forms.settle(d, mirrors) == d for d in forms.grow(seed, mirrors))
    line = ((0, 0, 0, 2), (1, 0, 0, 0))
    assert forms.settle(line) == forms.settle(((0, 0, 0, 2), (0, -1, 0, 1)))  # turned a quarter: the same design


def test_search_finds_an_arm_and_a_stronger_cube_reaches_further():
    weak = forms.search(UNIT, forms.FORMS["arm"], cubes=5, beam=4)
    strong = forms.search(forms.replace(UNIT, twist=3 * UNIT.twist), forms.FORMS["arm"], cubes=5, beam=4)
    assert weak and max(s[0] for _, s in strong) >= max(s[0] for _, s in weak) and all(len(s) == 3 for _, s in weak)
    design = next(iter(weak))[0]
    assert [row.get("parent") for row in forms.cells(design)][0] is None


def test_a_loaded_foot_drags_so_a_leg_needs_far_more_than_its_own_weight():
    mirrors = ((0, 3), (1, 0))  # body of four along x, a hub and a foot at each corner
    design = forms.settle({(x, 0, 1, 0) for x in (1, 2)} | {(x, 0, 1, 1) for x in (0, 3)} | {(x, s, 1, 2) for x in (0, 3) for s in (1, -1)} | {(x, s, 0, 0) for x in (0, 3) for s in (1, -1)}, mirrors)
    hold_leg_out = UNIT.mass * forms.GRAVITY * UNIT.side  # N m: the foot cube one side from the hip axis
    walks = lambda twist, grip: bool(forms.walker(design, forms.replace(UNIT, twist=twist, margin=1.0, grip=grip)))
    assert not walks(3 * hold_leg_out, 0.6)  # holding the leg out is not the limit
    assert walks(9 * hold_leg_out, 0.6)  # half the robot on the foot's edge, dragged: 6 x (0.35 + 0.6 x 1.5) = 7.5
    assert not walks(9 * hold_leg_out, 1.2)  # a stickier floor asks for more
