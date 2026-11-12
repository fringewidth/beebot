"""A demo store (demos.write) and a fresh draw (demos.draws, as voxel/haiku.py exports it) made in
two processes agree on every attempt: family, kind, robot count and the hash of task, meta and
scene (demos.draw_digest). Each process gets its own PYTHONHASHSEED, as two runs of the CLI do: a
draw that iterates a set of strings picks differently under another string hashing. The stores
voxel_full/s0..s7 and their Haiku exports of 2026-11-10 parted ways that way (s0 at attempt 715,
in the 7th chunk of 24 scenes, at objects.draw_ref's compare options)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import beebot

SRC = Path(beebot.__file__).resolve().parents[1]  # the code under test, also in the subprocesses

CHECK = """
import json, sys
from beebot.vla.voxel import demos as D, haiku as H, suite as S

options, attempts = H.store_options(sys.argv[1])
robots = None if options["robots"] is None else tuple(options["robots"])
check = options["fit"] or robots is not None
index = 0
for _, world, metas, tasks, _ in D.draws(10**9, options["seed"], options["chunk"], robots, options["counts"], check, options["rule"], options["apart"], options["coverage"], options["objects"], options["build"], options["only"], options["keep"]):
    for i, task in enumerate(tasks):
        if index == len(attempts):
            print(json.dumps({"checked": index}))
            sys.exit()
        drawn = {"family": task.family, "kind": S.task_kind(task), "robots": int(world.mass[i].sum()), "draw": D.draw_digest(world, metas, tasks, i)}
        stored = {key: attempts[index].get(key) for key in drawn}
        if stored != drawn:
            print(json.dumps({"checked": index, "index": index, "store": stored, "drawn": drawn}))
            sys.exit()
        index += 1
"""

CUBE = ["--cube", "--rule", "switch_cube", "--robots", "6", "115", "--counts", "log", "--fit", "--compact", "--threads", "1"]


def python(args, hash_seed):
    env = dict(os.environ, PYTHONHASHSEED=str(hash_seed), PYTHONPATH=os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]), OMP_NUM_THREADS="1")
    return subprocess.run([sys.executable, *args], env=env, check=True, capture_output=True, text=True).stdout


def check_store(store, hash_seed):
    """The first attempt of the store that differs from a fresh draw under hash_seed (None: none)
    and the number of attempts checked."""
    result = json.loads(python(["-c", CHECK, str(store)], hash_seed).strip().splitlines()[-1])
    return result.get("index"), result


# Few steps: the draw does not depend on them and playing is quick (builds get enough for some
# demonstrations, as a store with none fails to save). A build draw spends about a minute per
# process on build.min_sizes, so the build store is checked under one other hashing only.
@pytest.mark.parametrize(
    "name, scenes, chunk, options, hash_seeds",
    [
        ("full", 168, 24, ["--steps", "4", "--coverage", "0.3", "--objects", "1.0"], (1, 2)),  # through s0's attempt 715
        ("keep", 48, 8, ["--steps", "4", "--coverage", "0.3", "--objects", "1.0", "--keep"], (1, 2)),
        ("build", 8, 4, ["--steps", "36", "--build", "1.0", "--only", "build"], (1,)),
    ],
)
def test_store_matches_a_draw_in_another_process(tmp_path, name, scenes, chunk, options, hash_seeds):
    store = tmp_path / name
    python(["-m", "beebot.vla.voxel.demos", "--out", str(store), "--seed", "0", "--scenes", str(scenes), "--chunk", str(chunk), *CUBE, *options], hash_seed=0)
    attempts = [json.loads(line) for line in (store / "attempts.jsonl").read_text().splitlines()]
    assert attempts and all("draw" in row for row in attempts)
    for hash_seed in hash_seeds:
        first, result = check_store(store, hash_seed)
        assert first is None, f"{name}: attempt {first} differs from a draw under PYTHONHASHSEED={hash_seed}: {result}"
        assert result["checked"] == len(attempts)
