import numpy as np

from beebot.vla import mujoco_world as M


def test_push_threshold_matches_grid():
    """One or two robots cannot move a cheese (at most the bump of hitting it); three can, as the
    grid's friction threshold at size 32 (3 robots' worth)."""
    moved = {n: M.push_test(n, seconds=2) for n in (1, 2, 3, 4)}
    assert moved[1][-1] < 0.05 and moved[2][-1] < 0.2
    assert moved[2][-1] - moved[2][0] < 0.01  # no creep under the threshold
    assert moved[3][0] > 0.8 and moved[4][0] > moved[3][0]


def test_robot_reaches_next_square_in_one_step():
    swarm = M.Swarm(np.zeros((32, 32), bool), np.array([[10, 10], [20, 20]]), np.zeros((0, 2)), 4)
    swarm.run(np.array([[10, 11], [21, 21]]))
    squares, _ = swarm.snap()
    assert (squares == [[10, 11], [21, 21]]).all()
    assert np.abs(swarm.positions() - [[10, 11], [21, 21]]).max() < 0.05


def test_wall_rectangles_cover_walls_exactly():
    rng = np.random.default_rng(0)
    walls = rng.random((32, 32)) < 0.3
    cover = np.zeros((32, 32), int)
    for r, c, h, w in M.wall_rectangles(walls):
        cover[r : r + h, c : c + w] += 1
    assert (cover == walls).all()


def test_snap_gives_distinct_free_squares():
    """Two robots squeezed toward one square snap to two distinct free squares, never onto a wall
    or a cheese."""
    walls = np.zeros((32, 32), bool)
    walls[5, 6] = True
    swarm = M.Swarm(walls, np.array([[5, 4], [5, 5], [4, 5]]), np.array([[6, 6]]), 4)
    swarm.data.qpos[swarm.robot_q[:, 0]] = 5.5  # every robot at the centre of square (5, 5)
    swarm.data.qpos[swarm.robot_q[:, 1]] = 5.5
    squares, cheeses = swarm.snap()
    assert len({tuple(s) for s in squares}) == 3
    assert not walls[squares[:, 0], squares[:, 1]].any()
    assert ((squares < 6) | (squares >= 10)).any(1).all()  # off the cheese at rows / cols 6-9
    assert (cheeses == [[6, 6]]).all()


def test_swarm_herded_into_walls_stays_calm():
    """Whole swarms driven one way into walls and cheeses for 15 policy steps: no robot is shot
    out of the crowd (with the Newton solver one reached 41 squares per second here)."""
    import mujoco

    from beebot.vla import tasks as T

    rng = np.random.default_rng(0)
    peak = 0.0
    for _ in range(3):
        scene = None
        while scene is None:
            scene = T.sample_scene(rng, cheeses=(2, 3), boxes=(0, 0), wall_share=1.0)
        swarm = M.Swarm(scene["walls"], np.argwhere(scene["mass"] > 0.5), np.array(scene["cheeses"]), 4)
        for _ in range(15):
            squares, _ = swarm.snap()
            code = int(rng.integers(0, 2)) * 2  # everyone right (0) or down (2)
            goal = M.targets_of(squares, np.full(len(squares), code), scene["walls"])[:, ::-1] + 0.5
            for _ in range(round(1.0 / M.TIMESTEP)):
                swarm.data.ctrl[:] = np.clip(M.ROBOT_GAIN * (goal - swarm.robot_xy()), -M.ROBOT_SPEED, M.ROBOT_SPEED).reshape(-1)
                mujoco.mj_step(swarm.model, swarm.data)
                peak = max(peak, float(np.abs(swarm.data.qvel).max()))
    assert peak < 2 * M.ROBOT_SPEED
