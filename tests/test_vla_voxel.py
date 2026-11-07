import numpy as np
import torch

from beebot.vla.voxel import facts as F
from beebot.vla.voxel.controller import act, direction_code
from beebot.vla.voxel.render import render_scene
from beebot.vla.voxel.scripted import generate
from beebot.vla.voxel.world import (
    FULL_SHELL,
    N_ACTIONS,
    PALETTE_RGB,
    STAY,
    SHELL_OFFSETS,
    cheese_id_map,
    observe,
    sample_scenes,
    step,
    world_from_arrays,
)

N = 8


def code(dz, dy, dx):
    return int(direction_code(torch.tensor([dz, dy, dx])))


def grid():
    return np.zeros((N, N, N), dtype=np.float32)


def codes_for(world, moves):
    """moves: {(z, y, x): (dz, dy, dx)}; everyone else stays."""
    codes = torch.full(world.mass.shape, STAY, dtype=torch.long)
    for (z, y, x), d in moves.items():
        codes[0, z, y, x] = code(*d)
    return codes


def robots(world):
    return {tuple(c) for c in world.mass[0].nonzero().tolist()}


def test_direction_codes_cover_all_actions():
    assert N_ACTIONS == 27 and STAY == 26
    assert code(0, 0, 0) == STAY
    assert len({code(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}) == 27


def test_robots_conserved_under_random_play():
    rng = np.random.default_rng(0)
    world, _ = sample_scenes(rng, 16, size=12)
    n0 = world.mass.sum(dim=(1, 2, 3))
    for _ in range(25):
        world, _ = step(world, torch.randint(N_ACTIONS, world.mass.shape))
        assert torch.equal(world.mass.sum(dim=(1, 2, 3)), n0)
        assert set(world.mass.unique().tolist()) <= {0.0, 1.0}
        assert not (world.walls & (world.mass > 0)).any()
        assert not ((cheese_id_map(world) >= 0) & (world.mass > 0)).any()


def test_conflict_one_winner_by_priority():
    m = grid()
    m[3, 3, 2] = m[3, 3, 4] = 1
    world = world_from_arrays(m)
    codes = codes_for(world, {(3, 3, 2): (0, 0, 1), (3, 3, 4): (0, 0, -1)})
    score = torch.zeros(world.mass.shape)
    score[0, 3, 3, 4] = 0.9
    score[0, 3, 3, 2] = 0.1
    after, _ = step(world, codes, score)
    assert robots(after) == {(3, 3, 2), (3, 3, 3)}
    score[0, 3, 3, 2] = 0.95
    after, _ = step(world, codes, score)
    assert robots(after) == {(3, 3, 3), (3, 3, 4)}


def test_line_advances_and_swap_stays():
    m = grid()
    for x in range(1, 5):
        m[2, 2, x] = 1
    world = world_from_arrays(m)
    after, _ = step(world, codes_for(world, {(2, 2, x): (0, 0, 1) for x in range(1, 5)}))
    assert robots(after) == {(2, 2, x) for x in range(2, 6)}
    m = grid()
    m[2, 2, 2] = m[2, 2, 3] = 1
    world = world_from_arrays(m)
    after, _ = step(world, codes_for(world, {(2, 2, 2): (0, 0, 1), (2, 2, 3): (0, 0, -1)}))
    assert robots(after) == robots(world)


def test_walls_block_and_edges_block():
    m, w = grid(), np.zeros((N, N, N), dtype=bool)
    m[0, 0, 0] = 1
    w[0, 0, 1] = True
    world = world_from_arrays(m, w)
    after, _ = step(world, codes_for(world, {(0, 0, 0): (0, 0, 1)}))
    assert robots(after) == {(0, 0, 0)}
    after, _ = step(world, codes_for(world, {(0, 0, 0): (-1, -1, -1)}))
    assert robots(after) == {(0, 0, 0)}


def test_gravity_unsupported_fall_one_voxel_per_step():
    m = grid()
    m[4, 3, 3] = m[5, 3, 3] = 1  # a floating pair
    world = world_from_arrays(m, gravity=True)
    stay = codes_for(world, {})
    world, _ = step(world, stay)
    assert robots(world) == {(3, 3, 3), (4, 3, 3)}
    for _ in range(5):
        world, _ = step(world, stay)
    assert robots(world) == {(0, 3, 3), (1, 3, 3)}
    flying = world_from_arrays(m, gravity=False)
    assert robots(step(flying, stay)[0]) == robots(flying)


def test_gravity_towers_stand_and_walls_hold():
    m, w = grid(), np.zeros((N, N, N), dtype=bool)
    for z in range(6):
        m[z, 2, 2] = 1  # a tower on the floor
    w[:, 5, 5] = True  # a pillar
    m[4, 5, 6] = m[4, 5, 7] = 1  # clinging to the pillar and to each other
    world = world_from_arrays(m, w, gravity=True)
    after, _ = step(world, codes_for(world, {}))
    assert robots(after) == robots(world)
    # Climbing: a robot steps up onto the tower's side; it is held by the tower.
    m2 = m.copy()
    m2[0, 2, 3] = 1
    world = world_from_arrays(m2, w, gravity=True)
    after, _ = step(world, codes_for(world, {(0, 2, 3): (1, 0, 0)}))
    assert (1, 2, 3) in robots(after)
    # A robot that steps off the tower top into the air falls back the next step.
    after, _ = step(world, codes_for(world, {(5, 2, 2): (1, 0, 1)}))
    assert (5, 2, 3) in robots(after)  # moved up and over, fell one voxel, now beside the tower
    assert (6, 2, 3) not in robots(after)


def test_gravity_cheese_falls():
    m = grid()
    world = world_from_arrays(m, cheeses=[(3, 2, 2, 0)], gravity=True)
    for _ in range(3):
        world, _ = step(world, codes_for(world, {}))
    assert world.cheese_pos[0, 0].tolist() == [0, 2, 2]


def push_scene(n_pushers, wall_ahead=False, robot_ahead=False):
    m, w = grid(), np.zeros((N, N, N), dtype=bool)
    pushers = [(z, y, 2) for z in range(2) for y in range(3, 5)][:n_pushers]
    for p in pushers:
        m[p] = 1
    if wall_ahead:
        w[0, 3, 5] = True
    if robot_ahead:
        m[0, 3, 5] = 1
    world = world_from_arrays(m, w, cheeses=[(0, 3, 3, 1)])
    return world, codes_for(world, {p: (0, 0, 1) for p in pushers})


def test_push_needs_friction_and_free_lead():
    world, codes = push_scene(4)
    after, info = step(world, codes)
    assert info.pushed[0, 0] and after.cheese_pos[0, 0].tolist() == [0, 3, 4]
    world, codes = push_scene(1)
    after, info = step(world, codes)
    assert not info.pushed[0, 0] and after.cheese_pos[0, 0].tolist() == [0, 3, 3]
    world, codes = push_scene(4, wall_ahead=True)
    after, info = step(world, codes)
    assert not info.pushed[0, 0]
    world, codes = push_scene(4, robot_ahead=True)
    after, info = step(world, codes)
    assert info.pushed[0, 0]
    assert (0, 3, 6) in robots(after) and (0, 3, 5) not in robots(after)  # shoved one further
    assert after.mass.sum() == world.mass.sum()


def eat_scene(fill):
    m = grid()
    coords = (np.array([0, 3, 3]) + SHELL_OFFSETS.numpy()).tolist()
    coords = [c for c in coords if all(0 <= v < N for v in c)]
    for c in coords[:fill]:
        m[tuple(c)] = 1
    return world_from_arrays(m, cheeses=[(0, 3, 3, 2)]), len(coords)


def test_eat_by_shell_coverage():
    world, n_open = eat_scene(10**6)
    assert n_open == FULL_SHELL - 16  # the layer under the floor is outside the grid
    after, info = step(world, torch.full(world.mass.shape, STAY))
    assert info.eaten[0, 0] and not after.cheese_alive[0, 0]
    world, _ = eat_scene(20)
    after, info = step(world, torch.full(world.mass.shape, STAY))
    assert not info.eaten[0, 0] and after.cheese_alive[0, 0]


def test_observe_channels():
    world = world_from_arrays(grid(), cheeses=[(0, 0, 0, 3)])
    obs = observe(world)
    assert obs.shape == (1, 6, N, N, N)
    assert obs[0, 2].sum() == 8 and obs[0, 1].sum() == 0
    assert torch.allclose(obs[0, 3:, 1, 1, 1], PALETTE_RGB[3])


def test_controller_heads_for_targets():
    m = grid()
    m[0, 0, 0] = m[0, 7, 7] = 1
    world = world_from_arrays(m)
    targets = torch.tensor([[[4, 4, 4]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    before = sum(np.linalg.norm(np.subtract(r, (4, 4, 4))) for r in robots(world))
    for _ in range(3):
        world, _ = step(world, act(world, targets, valid, jitter=0.0).long())
    after = sum(np.linalg.norm(np.subtract(r, (4, 4, 4))) for r in robots(world))
    assert after < before - 4
    codes = act(world, targets, valid, jitter=0.0)
    assert (codes[world.mass == 0] == STAY).all()


# Facts on hand-built scenes.


def snapshot(m, cheeses=(), alive=None):
    pos = np.array([c[:3] for c in cheeses] + [(0, 0, 0)] * (3 - len(cheeses)))
    alive = np.array([True] * len(cheeses) + [False] * (3 - len(cheeses))) if alive is None else np.array(alive)
    return {"mass": m, "cheese_pos": pos, "cheese_alive": alive}


def static(colours=(), walls=None, gravity=True):
    return {"walls": np.zeros((N, N, N), dtype=bool) if walls is None else walls, "cheese_color": list(colours) + [-1] * (3 - len(colours)), "gravity": gravity}


def test_direction_and_region_names():
    assert F.direction_name(0, 0, 3) == "right"
    assert F.direction_name(2, 0, -2) == "up left"
    assert F.direction_name(0, 3, 0.5) == "forward"
    assert F.region_name(0, 7, 0, 8) == "low front left"
    assert F.region_name(4, 4, 4, 8) == "center"


def test_facts_move_and_groups():
    a, b = grid(), grid()
    a[0, 1:3, 1:3] = 1
    b[0, 1:3, 5:7] = 1
    f = F.window_facts(static(), snapshot(a), snapshot(b))
    assert ("move", "right") in f.atoms and ("end region", "low back right") in f.atoms
    c = grid()
    c[0, 1:3, 0:2] = 1
    c[0, 1:3, 6:8] = 1
    f = F.window_facts(static(), snapshot(a), snapshot(c))
    assert ("groups", 2) in f.atoms


def test_facts_push_ate_around():
    m = grid()
    f = F.window_facts(static([0]), snapshot(m, [(0, 3, 3)]), snapshot(m, [(0, 3, 5)]))
    assert ("pushed", "red", "right") in f.atoms
    f = F.window_facts(static([0]), snapshot(m, [(2, 3, 3)]), snapshot(m, [(0, 3, 3)]))
    assert not any(a[0] == "pushed" for a in f.atoms)  # a fall under gravity is not a push
    f = F.window_facts(static([1]), snapshot(m, [(0, 3, 3)]), snapshot(m, [(0, 3, 3)], alive=[False, False, False]))
    assert ("ate", "green") in f.atoms
    world, _ = eat_scene(25)
    full = world.mass[0].numpy()
    f = F.window_facts(static([2]), snapshot(m, [(0, 3, 3)]), snapshot(full, [(0, 3, 3)]))
    assert ("around", "blue") in f.atoms


def test_facts_tower():
    a, b = grid(), grid()
    a[0, 0, :6] = 1
    b[:6, 0, 0] = 1
    f = F.window_facts(static(), snapshot(a), snapshot(b))
    assert ("tower", "tall") in f.atoms  # thresholds scale with the grid: 8 voxels at 16, 4 at 8
    f = F.window_facts(static(gravity=False), snapshot(a), snapshot(b))
    assert not any(x[0] == "tower" for x in f.atoms)


def test_facts_bridge():
    w = np.zeros((N, N, N), dtype=bool)
    w[:3, :, :2] = True
    w[:3, :, 6:] = True  # two plateaus of height 3, a trench x in 2..5
    a, b = grid(), grid()
    a[3, 0, 0:4] = 1  # lying on the left plateau
    b[2, 0, 2:6] = 1  # spanning the trench at the plateaus' top layer
    f = F.window_facts(static(walls=w), snapshot(a), snapshot(b))
    assert ("bridge",) in f.atoms
    c = grid()
    c[2, 0, 2:5] = 1  # one short of the far side
    f = F.window_facts(static(walls=w), snapshot(a), snapshot(c))
    assert ("bridge",) not in f.atoms


def test_generator_shapes_and_determinism():
    store = generate(size=12, n_episodes=6, seed=3, batch_size=4, steps=10)
    assert store["mass"].shape == (6, 11, 12, 12, 12) and store["mass"].dtype == torch.uint8
    assert store["actions"].shape == (6, 10, 12, 12, 12) and store["actions"].dtype == torch.int8
    assert store["cheese_pos"].shape == (6, 11, 3, 3) and store["cheese_alive"].shape == (6, 11, 3)
    assert store["walls"].shape == (6, 12, 12, 12) and store["cheese_color"].shape == (6, 3)
    assert len(store["phases"]) == 6 and len(store["phases"][0]) == 10
    assert int(store["actions"].min()) >= 0 and int(store["actions"].max()) <= STAY
    assert (store["actions"][store["mass"][:, :-1] == 0] == STAY).all()
    counts = store["mass"].flatten(2).sum(-1)
    assert (counts == counts[:, :1]).all()
    again = generate(size=12, n_episodes=6, seed=3, batch_size=4, steps=10)
    for key in ("mass", "actions", "cheese_pos", "cheese_alive", "walls", "cheese_color", "gravity"):
        assert torch.equal(store[key], again[key]), key
    assert store["phases"] == again["phases"]
    other = generate(size=12, n_episodes=6, seed=4, batch_size=4, steps=10)
    assert not torch.equal(store["mass"], other["mass"])


def test_store_facts_and_render(tmp_path):
    store = generate(size=12, n_episodes=2, seed=0, steps=12)
    f = F.store_facts(store, 0, 0, 12)
    assert isinstance(f.atoms, set)
    image = render_scene(store["mass"][0, -1].numpy(), store["walls"][0].numpy(), store["cheese_pos"][0, -1].numpy(), store["cheese_alive"][0, -1].numpy(), store["cheese_color"][0].numpy())
    assert image.size[0] > 0
