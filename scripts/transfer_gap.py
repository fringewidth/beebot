"""Sim-to-sim transfer gap of a policy checkpoint: the same tasks in the voxel grid and in MuJoCo.

    python scripts/transfer_gap.py --checkpoint runs/<run>/policy_step<k>.pt [--per-category 4] [--steps 72]
    python scripts/transfer_gap.py --actor expert        # no LM: the scripted expert through the same code
    python scripts/transfer_gap.py --actor random        # no LM: random codes (a smoke run)

The episodes are basics' (beebot.vla.voxel.basics: the same phrases, camera sides, board tasks and
feasibility filter, scored by the suite's checkers) on scenes MuJoCo can host: a flat floor, no terrain,
no cheese, 10-12 robots (closed_loop.hostable). Basics' own scenes are mostly hosted terrain, which
MuJoCo does not have (about 1 in 9 is hostable), so these are fresh draws and the grid score here is
not basics' score; compare a checkpoint's grid number here with its MuJoCo number, and checkpoints'
gaps with each other. The draws are fixed by --seed (default 4243), the same every run.

Each episode is played twice with the same policy: on the grid exactly as basics plays it (rollout.play,
the training failure rates, rule switch_cube) and with its moves carried out by the hinge cube in
MuJoCo (closed_loop.Executor, the demo's drive; the actor sees only the grid read back from the physics).

Output, JSON lines (stdout; --out appends the per-episode rows):
  {"run": "transfer_gap", "category": ..., "n", "grid_score", "mujoco_score", "gap", "grid_pass", "mujoco_pass", "pass_gap"} per category
  {"run": "transfer", "checkpoint", "step", "transfer_grid_score", "transfer_mujoco_score", "transfer_gap",
   "transfer_grid_pass", "transfer_mujoco_pass", "transfer_pass_gap", "transfer_harmful_per_move",
   "transfer_disagree_share", "moves", "disagreements": {primitive: {"moves", "disagree", "harmful"}}, "wall_seconds", ...}

gap = grid score - MuJoCo score (positive: the policy does worse when its moves meet physics). It exits 0
whatever happens (a failure is a row with "error"), and a watchdog ends the process after --deadline minutes.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from dataclasses import replace

import numpy as np
import torch

HARMFUL = ("off", "strayed", "knock")
DEMO_DRIVE = {"torque": 0.3}
SEED = 4243
ROBOTS = (10, 12)


def log(message):
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------
# The episodes


def draw_episodes(per_category=4, seed=SEED, robots=ROBOTS, steps=None, redraws=400):
    """per_category basics episodes of every category on hostable scenes: the category's phrases in turn,
    the camera side drawn, the board task basics builds, a scene drawn again until the task is feasible
    (basics.feasible). Same rows, same order, for the same arguments."""
    from beebot.hardware.closed_loop import hostable, settled
    from beebot.vla.voxel import basics as B
    from beebot.vla.voxel import rollout as R
    from beebot.vla.voxel import suite as S
    from beebot.vla.voxel.think_data import ORDER

    rng = np.random.default_rng(seed)
    out = []
    for category in B.CATEGORIES:
        phrases = [p for p in B.PHRASES if p[0] == category]
        for k in range(per_category):
            _, text, directions, job = phrases[k % len(phrases)]
            camera = ORDER[int(rng.integers(4))]
            region = B.board_region(directions, camera)
            for tries in range(redraws + 1):
                count = int(rng.integers(robots[0], robots[1] + 1))
                world, metas = S.scenes(rng, 1, layout="flat", gravity=True, cheeses=(0, 0), robots=count)
                board = settled(world.index(0))
                task = B.board_task(board, 0, text, region, job)
                if hostable(board) and B.feasible(task, board, metas[0], 0):
                    break
            else:
                raise ValueError(f"no feasible hostable scene for {text!r} in {redraws} draws")
            index = len(out)
            key = {"category": category, "text": text, "camera": camera, "region": region, "i": index, "redraws": tries, "robots": int(board.mass.sum())}
            episode = R.Episode(board, metas[0], task, [text], [0], [S.canonical_record(task)], R.STEPS, seed * 1_000 + index, key=key, camera=camera, as_said=True)
            if steps:
                episode = replace(episode, steps=min(episode.steps, steps))
            out.append(episode)
    return out


# ---------------------------------------------------------------------------------------------
# Actors


class RandomActor:
    """Random action codes for every cell (a smoke arm: the code path without an LM)."""

    def __init__(self, episodes, seed=0):
        self.draws = [torch.Generator().manual_seed(int(e.seed) + seed) for e in episodes]

    def __call__(self, world, t):
        from beebot.vla.voxel.world import N_ACTIONS

        codes = torch.empty(world.mass.shape, dtype=torch.long)
        for i, g in enumerate(self.draws):
            codes[i] = torch.randint(0, N_ACTIONS, world.mass.shape[1:], generator=g)
        return codes


def make_actors(kind, episodes, policy):
    """(grid actor, MuJoCo actor), each fresh for these episodes (the actors keep memory per episode)."""
    from beebot.vla.voxel import rollout as R

    if kind == "policy":
        return R.PolicyActor(policy, episodes), R.PolicyActor(policy, episodes)
    if kind == "expert":
        return R.ExpertActor(episodes, 4242), R.ExpertActor(episodes, 4242)
    return RandomActor(episodes), RandomActor(episodes)


def _shard_worker(conn, episodes, mode, swarm):
    """A process holding the Executor of a shard of the episodes (its Swarms live here)."""
    torch.set_num_threads(1)
    from beebot.hardware.closed_loop import Executor

    executor = Executor(episodes, mode, swarm)
    while True:
        message = conn.recv()
        if message is None:
            conn.send((executor.moves, executor.steps, executor.first_pass))
            return
        world, codes, priority, t = message
        conn.send(executor(world, codes, priority, t).mass)


class Sharded:
    """advance for rollout.play: the episodes spread over `workers` processes, each stepping its own
    scenes in MuJoCo (closed_loop.Executor) while the actor stays in this process. MuJoCo in threads
    does not scale (Python holds the GIL between its calls), so processes. After finish(): moves,
    steps, first_pass per episode, as Executor's."""

    def __init__(self, episodes, mode, swarm, workers):
        import multiprocessing as mp

        context = mp.get_context("spawn")
        n = len(episodes)
        self.shards = [list(range(w, n, workers)) for w in range(min(workers, n))]
        self.pipes, self.procs = [], []
        for idx in self.shards:
            parent, child = context.Pipe()
            proc = context.Process(target=_shard_worker, args=(child, [episodes[i] for i in idx], mode, swarm), daemon=True)
            proc.start()
            self.pipes.append(parent)
            self.procs.append(proc)
        self.n = n
        self.moves = self.steps = self.first_pass = None

    def __call__(self, world, codes, priority, t):
        from beebot.vla.voxel.rollout import concat

        for idx, pipe in zip(self.shards, self.pipes):
            pipe.send((concat([world.index(i) for i in idx]), codes[idx], priority[idx], t))
        mass = world.mass.clone()
        for idx, pipe in zip(self.shards, self.pipes):
            mass[idx] = pipe.recv()
        return replace(world, mass=mass)

    def finish(self):
        self.moves, self.steps, self.first_pass = [None] * self.n, [None] * self.n, [None] * self.n
        for idx, pipe, proc in zip(self.shards, self.pipes, self.procs):
            pipe.send(None)
            moves, steps, first = pipe.recv()
            for k, i in enumerate(idx):
                self.moves[i], self.steps[i], self.first_pass[i] = moves[k], steps[k], first[k]
            proc.join(30)
        return self


def play_both(episodes, kind, policy, mode, drive, threads=1):
    """(grid Played list, MuJoCo Played list, Executor, wall seconds {grid, mujoco}); all episodes in one batch."""
    from beebot.vla.voxel import rollout as R

    grid_actor, mujoco_actor = make_actors(kind, episodes, policy)
    clock = time.perf_counter()
    with torch.no_grad():
        grid = R.play(episodes, grid_actor)
    grid_wall = time.perf_counter() - clock
    clock = time.perf_counter()
    with torch.no_grad():
        executor = Sharded(episodes, mode, {"drive": dict(drive)} if drive is not None else None, threads)
        try:
            mujoco = R.play(episodes, mujoco_actor, R.RULE, failure=None, advance=executor)
            executor.finish()
        finally:
            for proc in executor.procs:
                if proc.is_alive():
                    proc.terminate()
    return grid, mujoco, executor, {"grid": grid_wall, "mujoco": time.perf_counter() - clock}


# ---------------------------------------------------------------------------------------------
# The comparison (pure functions of plain rows; tested on synthetic ones)


def mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


def compare(rows):
    """Per category and overall: n, grid and MuJoCo score and pass, and the gaps (grid minus MuJoCo).
    rows: [{"category", "grid_score", "mujoco_score", "grid_pass", "mujoco_pass"}]."""
    out = {}
    categories = list(dict.fromkeys(r["category"] for r in rows))
    for name in categories + ["all"]:
        part = [r for r in rows if name in ("all", r["category"])]
        g, m = mean([r["grid_score"] for r in part]), mean([r["mujoco_score"] for r in part])
        gp, mp = mean([float(r["grid_pass"]) for r in part]), mean([float(r["mujoco_pass"]) for r in part])
        out[name] = {"n": len(part), "grid_score": g, "mujoco_score": m, "gap": g - m, "grid_pass": gp, "mujoco_pass": mp, "pass_gap": gp - mp}
    return out


def disagreements(moves):
    """What MuJoCo did with the moves the grid let happen, by pivot class (the primitive MuJoCo planned;
    "" for a move its plan refused or that found the cube off its cell). The grid's own step carries
    out every move it allows, so a move whose detail is not "reached" is one the worlds disagree on.
    Returns (per primitive {"moves", "disagree", "harmful", "detail": {...}}, totals {"moves",
    "disagree", "harmful"}). moves: rows with "primitive" and "detail" (closed_loop.Executor.moves)."""
    per = defaultdict(lambda: {"moves": 0, "disagree": 0, "harmful": 0, "detail": defaultdict(int)})
    for m in moves:
        cls = m.get("primitive", "") or "none"
        row = per[cls]
        row["moves"] += 1
        row["disagree"] += m["detail"] != "reached"
        row["harmful"] += m["detail"] in HARMFUL
        row["detail"][m["detail"]] += 1
    per = {k: {**v, "detail": dict(v["detail"])} for k, v in sorted(per.items())}
    total = {"moves": sum(v["moves"] for v in per.values()), "disagree": sum(v["disagree"] for v in per.values()), "harmful": sum(v["harmful"] for v in per.values())}
    return per, total


def step_of(path):
    found = re.search(r"step(\d+)", os.path.basename(path or ""))
    return int(found.group(1)) if found else None


def summary_rows(rows, moves, checkpoint, wall, extra=None):
    """The JSON lines of a run: one per category, then the overall `transfer` row."""
    table = compare(rows)
    per, total = disagreements(moves)
    lines = []
    for name, v in table.items():
        if name == "all":
            continue
        lines.append({"run": "transfer_gap", "category": name, **{k: round(x, 4) if isinstance(x, float) else x for k, x in v.items()}})
    a = table["all"]
    overall = {
        "run": "transfer",
        "checkpoint": checkpoint,
        "step": step_of(checkpoint),
        "episodes": a["n"],
        "transfer_grid_score": round(a["grid_score"], 4),
        "transfer_mujoco_score": round(a["mujoco_score"], 4),
        "transfer_gap": round(a["gap"], 4),
        "transfer_grid_pass": round(a["grid_pass"], 4),
        "transfer_mujoco_pass": round(a["mujoco_pass"], 4),
        "transfer_pass_gap": round(a["pass_gap"], 4),
        "moves": total["moves"],
        "harmful": total["harmful"],
        "transfer_harmful_per_move": round(total["harmful"] / max(total["moves"], 1), 4),
        "transfer_disagree_share": round(total["disagree"] / max(total["moves"], 1), 4),
        "disagreements": per,
        "wall_seconds": {k: round(v, 1) for k, v in wall.items()},
    }
    overall.update(extra or {})
    lines.append(overall)
    return lines


# ---------------------------------------------------------------------------------------------


def watchdog(minutes, checkpoint):
    """End the process (exit 0, with an error row) when the measurement runs past `minutes`."""

    def stop():
        print(json.dumps({"run": "transfer", "checkpoint": checkpoint, "step": step_of(checkpoint), "error": f"deadline: stopped after {minutes} minutes"}), flush=True)
        os._exit(0)

    timer = threading.Timer(minutes * 60, stop)
    timer.daemon = True
    timer.start()
    return timer


def run(args):
    started = time.perf_counter()
    torch.set_num_threads(max(1, args.threads))
    episodes = draw_episodes(args.per_category, args.seed, tuple(args.robots), args.steps)
    log(f"{len(episodes)} episodes drawn in {time.perf_counter() - started:.1f} s; robots {[e.key['robots'] for e in episodes]}")
    policy = None
    if args.actor == "policy":
        if not args.checkpoint:
            raise ValueError("the policy actor needs --checkpoint")
        from beebot.vla.voxel import think

        policy, _ = think.load_policy(args.checkpoint, args.device, torch.float32)
        policy.eval()
        log(f"policy loaded on {args.device}")
    drive = json.loads(args.drive) if args.drive else None
    grid, mujoco, executor, wall = play_both(episodes, args.actor, policy, args.mode, drive, args.workers)
    rows = []
    for k, e in enumerate(episodes):
        rows.append({**e.key, "grid_score": grid[k].score, "grid_pass": bool(grid[k].passed), "mujoco_score": mujoco[k].score, "mujoco_pass": bool(mujoco[k].passed), "mujoco_first_pass": executor.first_pass[k], "mujoco_moves": len(executor.moves[k]), "mujoco_off_at_end": executor.steps[k][-1]["off"] if executor.steps[k] else 0})
    moves = [m for ms in executor.moves for m in ms]
    total_wall = time.perf_counter() - started
    wall = {**wall, "total": total_wall}
    lines = summary_rows(rows, moves, args.checkpoint, wall, {"actor": args.actor, "mode": args.mode, "steps": episodes[0].steps, "seed": args.seed})
    for line in lines:
        print(json.dumps(line), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps({"checkpoint": args.checkpoint, "actor": args.actor, **r}) + "\n")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--actor", choices=["policy", "expert", "random"], default="policy")
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--steps", type=int, default=72, help="cap on every episode's steps, both worlds (72 is basics')")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--robots", type=int, nargs=2, default=list(ROBOTS))
    parser.add_argument("--mode", choices=["sequential", "simultaneous"], default="simultaneous")
    parser.add_argument("--workers", type=int, default=8, help="processes stepping MuJoCo scenes")
    parser.add_argument("--drive", default=json.dumps(DEMO_DRIVE), help="hinge cube Drive fields as JSON ('' for the flywheel cube)")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--deadline", type=float, default=40.0, help="minutes after which the process ends itself (exit 0, error row)")
    parser.add_argument("--out", default="", help="append the per-episode rows here (JSON lines)")
    args = parser.parse_args(argv)
    watchdog(args.deadline, args.checkpoint)
    try:
        return run(args)
    except Exception as error:  # the chain must go on training
        print(json.dumps({"run": "transfer", "checkpoint": args.checkpoint, "step": step_of(args.checkpoint), "error": f"{type(error).__name__}: {error}"}), flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
