from beebot.hardware.cube import NOMINAL
from beebot.hardware.cube_hinge import Drive, HingeSwarm, smooth


def test_profile_runs_from_zero_to_one_and_rests_at_both_ends():
    assert smooth(0.0) == (0.0, 0.0)
    assert smooth(1.0) == (1.0, 0.0)
    assert smooth(2.0) == (1.0, 0.0)
    assert abs(smooth(0.5)[0] - 0.5) < 1e-9


def test_lone_cube_rolls_on_a_latch_floor_and_climbs_a_lone_cube():
    swarm = HingeSwarm([(0, 0, 1)], seed=0, jitter=NOMINAL)
    roll = swarm.move(0, (0, 0, 1))
    assert (roll.primitive, roll.detail, roll.cell) == ("floor_roll", "reached", (0, 0, 2))
    assert swarm.hinges[0].outcome == "arrived"
    swarm = HingeSwarm([(0, 0, 1), (0, 0, 0)], seed=0, jitter=NOMINAL)
    climb = swarm.move(1, (1, 0, 1))  # up and over a lone cube: the ballistic flip shoves it away
    assert (climb.primitive, climb.detail, climb.cell) == ("convex_up", "reached", (1, 0, 1))


def test_too_weak_a_motor_turns_back_and_leaves_the_cube_where_it_was():
    swarm = HingeSwarm([(0, 0, 1), (0, 0, 0)], seed=0, jitter=NOMINAL, drive=Drive(torque=0.02))
    climb = swarm.move(1, (1, 0, 1))
    assert swarm.hinges[0].outcome in ("returned", "stalled")
    assert (climb.detail, climb.cell) == ("stayed", (0, 0, 0))


def test_no_floor_latch_means_no_floor_pivot():
    swarm = HingeSwarm([(0, 0, 1)], seed=0, jitter=NOMINAL, drive=Drive(floor_latch=False))
    roll = swarm.move(0, (0, 0, 1))
    assert (roll.detail, roll.cell) == ("stayed", (0, 0, 1))
    assert not swarm.hinges[0].latched
