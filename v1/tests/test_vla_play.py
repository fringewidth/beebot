"""Checks for the play-program runner: programs drive the swarm, and bad programs fail alone and contained."""

import numpy as np

from beebot.vla.play import run_program
from beebot.vla.world import sample_world

GATHER_TOP_LEFT = """
def act(view, memory):
    memory["phase"] = "corner"
    return [{"row": 0, "col": 0}]
"""

PHASED = """
def act(view, memory):
    if view["step"] < 5:
        memory["phase"] = "first"
        return [{"row": 0, "col": 15}]
    memory["phase"] = "second"
    return [{"row": 15, "col": 0}]
"""

HANG_ON_HEAVY = """
def act(view, memory):
    while view["total_mass"] > THRESHOLD:
        pass
    return [{"row": 8, "col": 8}]
"""

WRITES_A_FILE = """
def act(view, memory):
    np.save("/tmp/beebot_sandbox_probe.npy", view["mass"])
    return []
"""

IMPORTS_OS = """
import os
def act(view, memory):
    return []
"""


def test_program_gathers_and_records():
    world = sample_world(np.random.default_rng(0), 3)
    out = run_program(GATHER_TOP_LEFT, world, steps=40)
    assert out["errors"] == [None, None, None]
    assert out["mass"].shape == (3, 41, 16, 16)
    assert out["field"].shape == (3, 40, 2, 16, 16)
    total = out["mass"][:, -1].astype(np.float64).sum(axis=(-2, -1))
    near = out["mass"][:, -1, :8, :8].astype(np.float64).sum(axis=(-2, -1))
    assert (near > 0.8 * total).all()
    assert out["phases"][0][0] == "corner"


def test_phases_are_logged_per_step():
    out = run_program(PHASED, sample_world(np.random.default_rng(1), 1), steps=8)
    assert out["phases"][0] == ["first"] * 5 + ["second"] * 3


def test_timeout_fails_only_that_episode():
    world = sample_world(np.random.default_rng(2), 4)
    masses = sorted(world.mass.sum(dim=(-2, -1)).tolist())
    threshold = (masses[-1] + masses[-2]) / 2  # only the heaviest scene hangs
    out = run_program(HANG_ON_HEAVY.replace("THRESHOLD", str(threshold)), world, steps=3)
    assert sum(e == "timeout" for e in out["errors"]) == 1
    assert sum(e is None for e in out["errors"]) == 3


def test_sandbox_blocks_file_writes_and_imports():
    world = sample_world(np.random.default_rng(3), 1)
    out = run_program(WRITES_A_FILE, world, steps=2)
    assert out["errors"][0] is not None
    out = run_program(IMPORTS_OS, world, steps=2)
    assert out["errors"][0].startswith("load")
