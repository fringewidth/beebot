"""The compact voxel store (robots by voxel index instead of dense grids): it gives back the dense
arrays exactly, old stores still load, the trainer's batches are the same from both, and the
demonstration generator's options that go with it (log-uniform robot counts, tasks checked
against the swarm in every mode)."""

import json
import random

import numpy as np
import pytest
import torch

from beebot.vla.voxel import demos, play_store
from beebot.vla.voxel import suite as S
from beebot.vla.voxel.lm import batch_of, world_at
from beebot.vla.voxel.play_store import CompactGrids, CompactWriter, convert, load
from beebot.vla.voxel.world import STAY


def _episodes(rng, n, steps, dims):
    """Dense (mass, actions) of n episodes whose robot count changes from frame to frame, with
    an empty frame and a full one among them."""
    mass = (rng.random((n, steps + 1, *dims)) < rng.random((n, steps + 1, 1, 1, 1)) * 0.2).astype(np.uint8)
    mass[0, 1] = 0
    mass[-1, 0] = 1
    actions = np.where(mass[:, :-1] > 0, rng.integers(0, STAY + 1, (n, steps, *dims)), STAY).astype(np.int8)
    return mass, actions


def _write(out, mass, actions):
    writer = CompactWriter(out, mass.shape[2:])
    for e in range(len(mass)):
        writer.add(mass[e], actions[e])
    writer.close()
    return CompactGrids(out, "mass"), CompactGrids(out, "actions")


def test_compact_grids_equal_the_dense_arrays(tmp_path):
    mass, actions = _episodes(np.random.default_rng(0), 5, 7, (6, 5, 4))
    grids = dict(zip(("mass", "actions"), _write(tmp_path, mass, actions)))
    for name, dense in (("mass", mass), ("actions", actions)):
        got = grids[name]
        assert got.shape == dense.shape and len(got) == len(dense) and got.dtype == torch.from_numpy(dense).dtype
        assert got[:].dtype == got.dtype and np.array_equal(got[:].numpy(), dense)
        for index in [(3, 2), (0, 1), (4, 0), 2, (slice(1, 4), 0), (slice(None), slice(2, 5)), (1, slice(None, None, 2)), (2, 3, 1), (slice(0, 2), slice(0, 2), slice(None), 2, 1), (-1, -1), ([0, 3], 1)]:
            assert np.array_equal(got[index].numpy(), dense[index]), index
        assert got[5:].shape == dense[5:].shape
    assert int(grids["mass"][0, 1].sum()) == 0 and int(grids["actions"][0, 1].min()) == STAY  # the empty frame


def test_compact_writer_refuses_what_it_cannot_give_back(tmp_path):
    mass, actions = _episodes(np.random.default_rng(1), 1, 3, (4, 4, 4))
    writer = CompactWriter(tmp_path, mass.shape[2:])
    heavy = mass[0].copy()
    heavy[0, 0, 0, 0] = 2
    with pytest.raises(ValueError):
        writer.add(heavy, actions[0])
    stray = actions[0].copy()
    stray[1][mass[0, 1] == 0] = 3  # frame 1 has no robots
    with pytest.raises(ValueError):
        writer.add(mass[0], stray)
    with pytest.raises(ValueError):
        writer.add(mass[0, :-1], actions[0])
    writer.close()
    assert CompactGrids(tmp_path, "mass").shape == (0, 0, 4, 4, 4)


def test_large_grids_use_wider_voxel_indices(tmp_path):
    mass = np.zeros((1, 2, 41, 40, 41), np.uint8)
    mass[0, :, -1, -1, -1] = 1
    actions = np.full((1, 1, 41, 40, 41), STAY, np.int8)
    actions[0, 0, -1, -1, -1] = 4
    got_mass, got_actions = _write(tmp_path, mass, actions)
    assert json.loads((tmp_path / "compact.json").read_text())["voxel_dtype"] == "uint32"
    assert np.array_equal(got_mass[:].numpy(), mass) and np.array_equal(got_actions[:].numpy(), actions)


def _same_batches(dense, compact, labels):
    for seed in range(3):
        a = batch_of(dense, labels, 6, random.Random(seed), "cpu")
        b = batch_of(compact, labels, 6, random.Random(seed), "cpu")
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2] == b[2]
        assert a[0].dtype == b[0].dtype and a[1].dtype == b[1].dtype


def test_demo_store_converts_and_trains_the_same(tmp_path):
    steps = 16
    options = dict(seed=0, chunk=6, steps=steps, moves="cube", robots=(6, 115), failure=demos.parse_failure("0.15"))
    count = demos.write(tmp_path / "dense", 6, **options)
    before = {p.name: p.stat().st_size for p in (tmp_path / "dense").iterdir()}
    assert convert(tmp_path / "dense", tmp_path / "converted") == count > 0
    assert before == {p.name: p.stat().st_size for p in (tmp_path / "dense").iterdir()}  # the source is only read
    dense, converted = load(tmp_path / "dense"), load(tmp_path / "converted")
    assert isinstance(dense["mass"], torch.Tensor) and isinstance(converted["mass"], CompactGrids)  # the old store still loads as it did
    assert converted["mass"].shape == (count, steps + 1, 16, 16, 16) and converted["actions"].shape == (count, steps, 16, 16, 16)  # only the rows in use
    for k in ("mass", "actions"):
        assert torch.equal(converted[k][:], dense[k][:count])
    for k in ("walls", "cheese_pos", "cheese_alive", "cheese_color", "gravity", "robots"):
        assert torch.equal(converted[k], dense[k])
    assert converted["subgoals"] == dense["subgoals"] and converted["config"] == dense["config"]
    for name in ("labels.jsonl", "attempts.jsonl", "done"):
        assert (tmp_path / "converted" / name).read_text() == (tmp_path / "dense" / name).read_text()
    labels = [json.loads(line) for line in open(tmp_path / "converted" / "labels.jsonl")]
    _same_batches(dense, converted, labels)
    rows = [(count - 1, steps - 1), (0, 0), (count // 2, 5)]
    assert all(torch.equal(getattr(world_at(dense, rows), f), getattr(world_at(converted, rows), f)) for f in demos.FIELDS)
    with pytest.raises(ValueError):
        convert(tmp_path / "dense", tmp_path / "converted")  # never over an existing store

    assert demos.write(tmp_path / "direct", 6, compact=True, **options) == count  # written compact from the start: the same store
    direct = load(tmp_path / "direct")
    assert not (tmp_path / "direct" / "mass.npy").exists() and isinstance(direct["actions"], CompactGrids)
    for name in ("voxels.bin", "codes.bin", "offsets.npy", "compact.json", "labels.jsonl"):
        assert (tmp_path / "direct" / name).read_bytes() == (tmp_path / "converted" / name).read_bytes()
    dense_bytes = sum((tmp_path / "dense" / f"{k}.npy").stat().st_size for k in ("mass", "actions")) * count / len(dense["mass"])
    compact_bytes = sum((tmp_path / "direct" / name).stat().st_size for name in ("voxels.bin", "codes.bin", "offsets.npy"))
    assert compact_bytes * 8 < dense_bytes


def test_play_store_still_loads_and_converts(tmp_path):
    play_store.write(tmp_path / "play", size=12, n_episodes=4, seed=1, chunk=4)
    dense = load(tmp_path / "play")
    assert isinstance(dense["mass"], torch.Tensor) and dense["mass"].shape[0] == 4 and dense["mass"].dtype == torch.uint8
    assert convert(tmp_path / "play", tmp_path / "compact") == 4  # no "count": every episode
    converted = load(tmp_path / "compact")
    assert torch.equal(converted["mass"][:], dense["mass"]) and torch.equal(converted["actions"][:], dense["actions"])
    assert converted["mass"][2, 3].dtype == dense["mass"][2, 3].dtype and converted["actions"][2, 3].dtype == dense["actions"][2, 3].dtype
    steps = dense["actions"].shape[1]
    labels = [{"e": e, "w0": 0, "w1": steps, "text": f"episode {e}"} for e in range(4)]
    _same_batches(dense, converted, labels)


def _never(task, world, i):
    """Whether no expert can pass the task in scene i: it must eat a cheese whose open shell is
    too small to be eaten, or it is a formation with over three robots per mask voxel."""
    spec = task.spec
    slot = {c: k for k, c in S.present(world, i)}
    if spec["kind"] == "eat":
        hopeless = {c for c in spec["eat"] | spec["any_of"] if slot[c] not in task.runners and S.eat_need(world, i, slot[c]) == float("inf")}
        return bool(hopeless) and (not spec["any_of"] or hopeless == spec["any_of"])
    return spec["kind"] == "mask" and int(world.mass[i].sum()) > 3 * int(spec["mask"].sum())


def test_checked_draws_leave_out_tasks_no_swarm_can_do():
    default_world, _, default = S.sample_tasks(np.random.default_rng(1), 40, "train", "formation")
    assert any(_never(t, default_world, i) for i, t in enumerate(default))  # the density range holds swarms too large for small masks
    rng = np.random.default_rng(1)
    world, metas, tasks = S.sample_tasks(rng, 24, "train", check=True)  # no robots range: the check is opt-in
    world, metas, tasks = demos.contrast(rng, world, metas, tasks, check=True)
    assert int(world.mass.sum(dim=(1, 2, 3)).min()) >= 40 and len(tasks) > 40
    assert all(S.fits(t, world, i) and not _never(t, world, i) for i, t in enumerate(tasks))


def test_a_corner_cheese_is_never_an_eat_task_when_checked():
    world, metas = S.scenes(np.random.default_rng(0), 1, layout="flat", cheeses=(1, 1), robots=60)
    world.cheese_pos[0, world.cheese_alive[0].nonzero()[0, 0]] = torch.tensor([0, 0, 14])  # into a corner of the floor
    world.mass[0, :2, :3, 12:] = 0
    slot, colour = S.present(world, 0)[0]
    assert S.eat_need(world, 0, slot) == float("inf")
    rng = np.random.default_rng(0)
    made = [S.MAKERS["logic"](rng, "")[1](world, metas[0], 0) for _ in range(40)]
    eats = [t for t in made if t is not None and t.spec["eat"] | t.spec["any_of"]]
    assert eats and all(_never(t, world, 0) and not S.fits(t, world, 0) for t in eats)  # drawn by the maker, turned away by the check
    assert all(S.fits(t, world, 0) for t in made if t is not None and not (t.spec["eat"] | t.spec["any_of"]))  # eating nothing stays


def test_log_counts_oversample_small_swarms():
    rng = np.random.default_rng(0)
    low, high = S.ROBOTS
    draws = np.array([S.draw_count(rng, low, high, "log") for _ in range(20000)])
    assert draws.min() == low and draws.max() == high
    shares = [(draws < 15).mean(), ((draws >= 15) & (draws <= 40)).mean(), (draws > 40).mean()]
    assert all(abs(share - want) < 0.02 for share, want in zip(shares, (0.309, 0.339, 0.351)))
    uniform = np.array([S.draw_count(rng, low, high) for _ in range(20000)])
    assert abs((uniform < 15).mean() - 9 / 110) < 0.01
    assert S.draw_count(np.random.default_rng(7), low, high) == int(np.random.default_rng(7).integers(low, high + 1))  # the uniform draw is the one sample_tasks always made
    world, _, tasks = S.sample_tasks(np.random.default_rng(2), 30, "train", robots=S.ROBOTS, counts="log")
    counts = world.mass.sum(dim=(1, 2, 3))
    assert int((counts < 15).sum()) >= 4 and all(S.fits(t, world, i) for i, t in enumerate(tasks))


def test_demo_config_records_the_new_options(tmp_path):
    demos.write(tmp_path, 4, seed=1, chunk=4, steps=8, robots=(6, 40), counts="log", compact=True)
    store = load(tmp_path)
    assert store["config"]["counts"] == "log" and store["config"]["fit"] is True
    assert store["count"] == len(store["mass"]) == len(store["robots"])
    assert (store["mass"][:, 0].sum(dim=(1, 2, 3)) == store["robots"]).all() if store["count"] else True
