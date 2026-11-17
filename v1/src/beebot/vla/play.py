"""Run play programs on batches of scenes and record the episodes.

A play program is Python source defining act(view, memory) -> list of up to 4 attractors
{"row", "col", "straight"}. The view is a dict:
    step:        int, 0-based
    size:        the grid's side N (16 to 48)
    cheese_side: a cheese's side in squares (N / 8)
    mass:        NxN numpy array of swarm mass
    walls:       NxN numpy bool array
    total_mass: float
    cheeses:    [{"index", "color", "row", "col", "eaten"}], row and col of the top-left square
    boxes:      [{"color", "row", "col", "height", "width"}]
memory is a dict kept across one episode's steps; setting memory["phase"] to a short string marks
the program's phases (used only to cut windows, never as labels).

Each episode's program runs in its own subprocess under sandbox-exec (no network, no file
writes), with a time limit per call. A crash or timeout ends only that episode.
"""

import json
import os
import resource
import selectors
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from beebot.vla.controller import Controller, commands_to_tensors
from beebot.vla.world import PALETTE_NAMES, cheese_side, step

CALL_TIMEOUT = 1.0  # seconds per act() call on a 16x16 grid; scaled with the number of squares
CPU_LIMIT = 300  # seconds of CPU per episode process
SANDBOX_PROFILE = "(version 1)(allow default)(deny network*)(deny file-write*)"
CHILD = Path(__file__).with_name("child.py")


def _limit_cpu():
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT, CPU_LIMIT))


def already_sandboxed():
    """True if this process is inside a macOS sandbox: applying a nested one is refused."""
    probe = subprocess.run(["/usr/bin/sandbox-exec", "-p", SANDBOX_PROFILE, "/usr/bin/true"], capture_output=True)
    return probe.returncode != 0 and b"Operation not permitted" in probe.stderr


def _spawn(inside_sandbox=False):
    """inside_sandbox: the caller is already sandboxed (macOS can't nest sandboxes), so the child
    runs under the outer sandbox only. Callers must have checked already_sandboxed()."""
    command = [sys.executable, "-I", "-B", str(CHILD)]
    if not inside_sandbox:
        command = ["/usr/bin/sandbox-exec", "-p", SANDBOX_PROFILE] + command
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        preexec_fn=_limit_cpu,
        env={"PATH": os.environ.get("PATH", ""), "OMP_NUM_THREADS": "1"},
    )


def scene_view(world, i, t):
    cheeses = []
    for k in range(world.cheese_color.shape[1]):
        color = int(world.cheese_color[i, k])
        if color < 0:
            continue
        y, x = world.cheese_pos[i, k].tolist()
        cheeses.append({"index": k, "color": PALETTE_NAMES[color], "row": y, "col": x, "eaten": not bool(world.cheese_alive[i, k])})
    boxes = []
    for k in range(world.box_color.shape[1]):
        color = int(world.box_color[i, k])
        if color < 0:
            continue
        y, x, h, w = world.box_rect[i, k].tolist()
        boxes.append({"color": PALETTE_NAMES[color], "row": y, "col": x, "height": h, "width": w})
    mass = world.mass[i]
    return {
        "step": t,
        "size": mass.shape[0],
        "cheese_side": cheese_side(mass.shape[0]),
        "mass": np.round(mass.numpy(), 3).tolist(),
        "walls": world.walls[i].int().tolist(),
        "total_mass": round(float(mass.sum()), 3),
        "cheeses": cheeses,
        "boxes": boxes,
    }


class ProgramPlayers:
    """One sandboxed process per episode, all running the same program source."""

    def __init__(self, source, n, inside_sandbox=False, timeout=CALL_TIMEOUT):
        self.timeout = timeout
        self.processes = [_spawn(inside_sandbox) for _ in range(n)]
        self.errors = [None] * n
        for p in self.processes:
            p.stdin.write(json.dumps({"source": source}) + "\n")

    def act(self, views):
        """Send each live episode its view; returns (commands, phases), None commands for failed episodes."""
        selector = selectors.DefaultSelector()
        pending = {}
        for i, (p, view) in enumerate(zip(self.processes, views)):
            if self.errors[i] is not None:
                continue
            try:
                p.stdin.write(json.dumps(view) + "\n")
                p.stdin.flush()
            except (BrokenPipeError, OSError):
                self._fail(i, "process died")
                continue
            selector.register(p.stdout, selectors.EVENT_READ, i)
            pending[i] = time.monotonic() + self.timeout
        replies = {}
        while pending:
            remaining = max(min(pending.values()) - time.monotonic(), 0)
            for key, _ in selector.select(timeout=remaining):
                i = key.data
                line = key.fileobj.readline()
                selector.unregister(key.fileobj)
                pending.pop(i, None)
                replies[i] = json.loads(line) if line else {"error": "process died"}
            now = time.monotonic()
            for i in [i for i, deadline in pending.items() if deadline <= now]:
                selector.unregister(self.processes[i].stdout)
                pending.pop(i)
                replies[i] = {"error": "timeout"}
        selector.close()

        commands, phases = [], []
        for i in range(len(self.processes)):
            reply = replies.get(i)
            if reply is not None and "error" in reply:
                self._fail(i, reply["error"])
            if self.errors[i] is not None:
                commands.append(None)
                phases.append(None)
                continue
            commands.append(reply["commands"])
            phases.append(reply["phase"])
        return commands, phases

    def _fail(self, i, message):
        if self.errors[i] is None:
            self.errors[i] = message
            self.processes[i].kill()

    def close(self):
        for p in self.processes:
            if p.poll() is None:
                p.kill()
            p.wait()


def run_program(source, world, steps=64, inside_sandbox=False):
    """Play `source` on every scene in `world` for `steps` steps.

    Returns a dict of numpy arrays (mass (B, T+1, H, W) float16, field (B, T, 2, H, W) float16,
    cheese_pos (B, T+1, K, 2), cheese_alive (B, T+1, K), the static scene, phase labels per step,
    and per-episode errors (None for episodes that ran to the end).
    """
    b = world.mass.shape[0]
    players = ProgramPlayers(source, b, inside_sandbox, CALL_TIMEOUT * max(1.0, world.mass[0].numel() / 256))
    controller = Controller()
    masses = [world.mass.clone()]
    positions = [world.cheese_pos.clone()]
    alive = [world.cheese_alive.clone()]
    fields = []
    phases = []
    try:
        for t in range(steps):
            views = [scene_view(world, i, t) for i in range(b)]
            commands, step_phases = players.act(views)
            points, valid, straight, share = commands_to_tensors(commands, b)
            field = controller(world, points, valid, straight, share)
            world, _ = step(world, field)
            fields.append(field)
            masses.append(world.mass.clone())
            positions.append(world.cheese_pos.clone())
            alive.append(world.cheese_alive.clone())
            phases.append(step_phases)
    finally:
        players.close()
    return {
        "mass": torch.stack(masses, dim=1).numpy().astype(np.float16),
        "field": torch.stack(fields, dim=1).numpy().astype(np.float16),
        "cheese_pos": torch.stack(positions, dim=1).numpy().astype(np.int8),
        "cheese_alive": torch.stack(alive, dim=1).numpy(),
        "walls": world.walls.numpy(),
        "cheese_color": world.cheese_color.numpy(),
        "box_rect": world.box_rect.numpy(),
        "box_color": world.box_color.numpy(),
        "phases": [[phases[t][i] for t in range(steps)] for i in range(b)],
        "errors": list(players.errors),
    }
