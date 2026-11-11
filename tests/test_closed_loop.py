import numpy as np
import torch

from beebot.hardware.closed_loop import Executor, Grid, move_together, occupancy, play_mujoco, turn_cells
from beebot.hardware.cube import NOMINAL
from beebot.hardware.cube_swarm import Reading, Swarm
from beebot.vla.voxel import rollout as R
from beebot.vla.voxel import suite as S
from beebot.vla.voxel.world import DIRS, STAY, world_from_arrays

EAST, WEST, NORTH = DIRS.index((0, 0, 1)), DIRS.index((0, 0, -1)), DIRS.index((0, 1, 0))


def scene(cells, size=8):
    mass = np.zeros((size, size, size))
    for c in cells:
        mass[c] = 1.0
    return world_from_arrays(mass, gravity=True)


def episode(cells, task=None, steps=3, seed=0):
    task = task or S.Task("construction", "build a tower 9 high", {"kind": "tower", "height": 9, "forbid": set()})
    return R.Episode(scene(cells), {}, task, [task.text], [0], [{}], steps, seed)


def scripted(moves):
    """An actor that gives, every step, each robot standing on a cell of `moves` that cell's code."""

    def actor(world, t):
        codes = torch.full(world.mass.shape, STAY, dtype=torch.long)
        for (z, y, x), code in moves.items():
            if world.mass[0, z, y, x] > 0.5:
                codes[0, z, y, x] = code
        return codes

    return actor


def test_turn_cells_are_the_plans_swing_and_supports():
    swarm = Swarm([(0, 2, 2), (0, 2, 3), (0, 2, 6)], seed=0, jitter=NOMINAL, settle=0.05)
    roll = swarm.plan(2, (0, 0, 1))
    assert turn_cells((0, 2, 6), (0, 0, 1), roll) == ({(0, 2, 7), (1, 2, 6), (1, 2, 7)}, {(-1, 2, 6), (-1, 2, 7)})
    climb = swarm.plan(0, (1, 0, 1))  # up and over the cube at x + 1
    swing, leans = turn_cells((0, 2, 2), (1, 0, 1), climb)
    assert climb.target in swing and leans == {(0, 2, 3)}
    assert swing == {(1, 2, 2), (0, 2, 1), (1, 2, 1), (1, 2, 3), (2, 2, 2), (2, 2, 3)}


def test_two_far_apart_rolls_at_once_both_arrive():
    swarm = Swarm([(0, 0, 0), (0, 0, 5)], seed=0, jitter=NOMINAL, settle=0.05)
    moves = move_together(swarm, [(0, (0, 0, 1)), (1, (0, 1, 0))])
    assert [(m.detail, m.cell) for m in moves] == [("reached", (0, 0, 1)), ("reached", (0, 1, 5))]
    assert moves[0].sim_seconds == moves[1].sim_seconds < 0.6  # one turn's time for both


def test_executor_steps_the_grid_from_what_the_cubes_did():
    ep = episode([(0, 1, 1), (0, 5, 5)])
    played, executor = play_mujoco([ep], scripted({(0, 1, 1): EAST, (0, 1, 2): EAST}), swarm={"jitter": NOMINAL})
    assert [m["detail"] for m in executor.moves[0]] == ["reached", "reached"]
    assert executor.steps[0][-1]["cells"] == [[0, 1, 3], [0, 5, 5]]  # cube by cube, in grid order at the start
    assert [s["differ"] for s in executor.steps[0]] == [0, 0, 0]  # the grid would have done the same
    assert not executor.start_mismatch[0]
    assert not played[0].passed


def test_of_two_cubes_aiming_at_one_cell_one_goes():
    ep = episode([(0, 2, 1), (0, 2, 3)], steps=1)
    _, executor = play_mujoco([ep], scripted({(0, 2, 1): EAST, (0, 2, 3): WEST}), swarm={"jitter": NOMINAL})
    assert executor.steps[0][0]["lost"] == 1 and executor.steps[0][0]["movers"] == 1
    assert len(executor.moves[0]) == 1 and executor.moves[0][0]["detail"] == "reached"


def test_simultaneous_takes_one_wave_where_sequential_takes_two():
    cells = [(0, 1, 1), (0, 5, 5)]
    actor = {(0, 1, 1): EAST, (0, 5, 5): NORTH}
    _, one = play_mujoco([episode(cells, steps=1)], scripted(actor), mode="simultaneous", swarm={"jitter": NOMINAL})
    _, two = play_mujoco([episode(cells, steps=1)], scripted(actor), mode="sequential", swarm={"jitter": NOMINAL})
    assert one.steps[0][0]["waves"] == 1 and two.steps[0][0]["waves"] == 2
    assert sorted(one.steps[0][0]["cells"]) == sorted(two.steps[0][0]["cells"]) == [[0, 1, 2], [0, 6, 5]]
    assert one.steps[0][0]["sim_seconds"] < two.steps[0][0]["sim_seconds"]


def test_touching_movers_go_in_separate_waves():
    apart = {(0, 1, 1): WEST, (0, 3, 2): EAST}
    corner = {(0, 1, 1): WEST, (0, 2, 2): EAST}  # these two touch by an edge
    _, one = play_mujoco([episode(list(apart), steps=1)], scripted(apart), mode="simultaneous", swarm={"jitter": NOMINAL})
    _, two = play_mujoco([episode(list(corner), steps=1)], scripted(corner), mode="simultaneous", swarm={"jitter": NOMINAL})
    assert one.steps[0][0]["waves"] == 1 and two.steps[0][0]["waves"] == 2


def test_stop_rule_freezes_a_scene_once_its_checker_passes():
    tower = S.Task("construction", "build a tower 1 high", {"kind": "tower", "height": 1, "forbid": set()})
    actor = {(0, 1, x): EAST for x in range(1, 7)}
    _, stopped = play_mujoco([episode([(0, 1, 1)], tower, steps=3)], scripted(actor), stop=True, swarm={"jitter": NOMINAL})
    _, going = play_mujoco([episode([(0, 1, 1)], tower, steps=3)], scripted(actor), stop=False, swarm={"jitter": NOMINAL})
    assert stopped.first_pass[0] == 0 and len(stopped.moves[0]) == 1
    assert len(going.moves[0]) == 3 and going.passes[0] == [True, True, True]


def test_grid_tracker_matches_the_grid_step():
    ep = episode([(0, 1, 1)], steps=2)
    tracker = Grid([ep])
    played = R.play([ep], scripted({(0, 1, 1): EAST, (0, 1, 2): EAST}), failure=None, advance=tracker)
    plain = R.play([ep], scripted({(0, 1, 1): EAST, (0, 1, 2): EAST}), failure=None)
    assert played[0].score == plain[0].score and tracker.passes[0] == [False, False]


def test_occupancy_moves_a_crowding_cube_to_the_nearest_free_cell():
    reading = Reading(cells=np.array([(0, 1, 1), (0, 1, 1)]), off=np.array([True, False]))
    centres = np.array([(1.4, 1.0, 0.0), (1.0, 1.0, 0.0)])  # x, y, z: the off cube leans toward x + 1
    mass, crowded = occupancy(reading, centres, (4, 4, 4))
    assert crowded == 1 and mass[0, 1, 1] == 1 and mass[0, 1, 2] == 1 and mass.sum() == 2


def test_snap_puts_a_cube_off_the_lattice_back_on_a_cell():
    ep = episode([(0, 1, 1), (0, 4, 4)], steps=1)
    executor = Executor([ep], swarm={"jitter": NOMINAL}, snap=True)
    swarm = executor.swarms[0]
    adr = executor.qpos[0][0]
    swarm.data.qpos[adr[3:]] = [np.cos(np.pi / 8), np.sin(np.pi / 8), 0, 0]  # tilted 45 degrees about x
    swarm.data.qpos[adr[:3]] += [0.4 * swarm.cube.side, 0, 0.3 * swarm.cube.side]
    import mujoco

    mujoco.mj_forward(swarm.model, swarm.data)
    swarm.run(0.2)
    assert swarm.lattice().off[0]
    assert executor.snap_back(0) == 1
    reading = swarm.lattice()
    assert not reading.off.any() and tuple(reading.cells[1]) == (0, 4, 4)
