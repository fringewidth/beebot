"""Expert demonstrations of the voxel suite (voxel/suite.py), written as a store with labels for
voxel/lm.py: each scene is played under several training instructions from the same start (one
per family the scene allows), by the expert, and an episode is kept only if the checker says the
expert did what the instruction asks (and ate nothing it forbids). The label is the instruction.

    OMP_NUM_THREADS=4 uv run python -m beebot.vla.voxel.demos --out data/vla/voxel_demos_g16 --scenes 6000

Store fields as voxel/play_store (mass, actions memory-mapped; small.pt with the rest); labels.jsonl
records {"e", "w0", "w1", "family", "form", "text"} covering whole episodes. small.pt also holds,
per kept episode, "robots" (the swarm's size) and "subgoals" (what the expert was after at every
step: Expert.subgoals), and "config" (the options below). attempts.jsonl has one row per episode
played, kept or not: {"family", "kind", "robots", "score", "unasked", "e"} with e = -1 when dropped.

Demonstrations a pivoting cube can carry out (--cube): only the 14 moves without a horizontal
diagonal (--moves cube), 72 steps (--steps), swarms of 6 robots up to the usual maximum with only
tasks that size can do (--robots 6 115), and moves that fail with probability 0.15 (--failure;
per move class as in --failure 0.15,up_over=0.3). The recorded action is always the expert's
choice, whether or not the move happened.

    OMP_NUM_THREADS=4 uv run python -m beebot.vla.voxel.demos --out data/vla/voxel_demos_cube --scenes 6000 --cube

Demonstrations real cubes can carry out (--free-cube): --cube, and the world enforces the
free-cube rule (--rule free_cube, world.pivot_moves: a cube pivots on a support and moves only
when no other cube touches it face to face), gravity is on in every scene, and the expert is the
planner for lone cubes (--expert planner, voxel/free_cube.py). --rule switch_cube is the same for
cubes that can release their bonds. No recorded action breaks the rule. --apart sets every
swarm down again before the episode so that no two robots touch (free_cube.apart): under the rule
a cube that starts in a group can never leave it.

    OMP_NUM_THREADS=4 uv run python -m beebot.vla.voxel.demos --out data/vla/voxel_demos_free_cube --scenes 6000 --free-cube --counts log --compact

Further options, all off unless given: --compact writes the grids by robot (play_store's compact
format, about 40 times smaller; load() reads both); --counts log draws the robot count
log-uniformly over --robots, so small swarms are as common as large ones; --fit draws only tasks
the scene's swarm can do (suite.fits) also when --robots is not given.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel import suite as S
from beebot.vla.voxel.play_store import CompactWriter, DenseWriter
from beebot.vla.voxel.free_cube import FreeCubeExpert, apart
from beebot.vla.voxel.world import CUBE_MOVES, FAILURE_RATE, MAX_OBJECTS, RULES, VoxelWorld, concat, failure_rates, with_objects

SUCCESS = S.SUCCESS  # checker score an episode needs to count as a demonstration
FIELDS = ("mass", "walls", "cheese_pos", "cheese_alive", "cheese_color", "gravity")
OBJECT_FIELDS = ("form", "colour2", "pattern", "friction", "food", "zones")  # per episode in small.pt when the draw has objects (README "Objects")
MOVES = {"all": None, "cube": CUBE_MOVES}
EXPERTS = ("attractor", "planner")  # suite.Expert; free_cube.FreeCubeExpert (needs a rule)


def contrast(rng, world, metas, tasks, check=False):
    """Each scene's own task plus one training task of every other family its scene allows
    (check: and its swarm is large enough for, suite.fits)."""
    worlds, out_metas, out = [], [], []
    for i, task in enumerate(tasks):
        w = world.index(i)
        worlds.append(w)
        out_metas.append(metas[i])
        out.append(task)
        for family in S.FAMILIES:
            if family == task.family:
                continue
            _, make = S.MAKERS[family](rng, "")
            other = make(w, metas[i], 0)
            if other is not None and check and not S.fits(other, w, 0):
                continue
            if other is not None and other.plan or (other is not None and other.spec.get("form") in ("none", "absent")):
                worlds.append(w)
                out_metas.append(metas[i])
                out.append(other)
    return VoxelWorld(*(torch.cat([getattr(x, f) for x in worlds]) for f in FIELDS)), out_metas, out


ONLY = ("build",)  # families that can be drawn alone (draws only=...)


def draws(n_scenes, seed=0, chunk=24, robots=None, counts="uniform", check=False, rule=None, spaced=False, coverage=0.0, objects=0.0, build=0.0, only=None):
    """(c, world, metas, tasks, run_seed) of each chunk of scenes, in the order write() plays them:
    the scene sample, its contrast tasks, then round(coverage x the chunk's scenes) coverage tasks
    (suite.sample_coverage, one per scene, from a generator of their own seeded by (seed, 7), so
    the suite's draws are the same with or without them), then round(objects x the chunk's scenes)
    object tasks (objects.sample_objects, from a generator seeded by (seed, 11); with objects every
    world of the draw carries the object fields and world.MAX_OBJECTS slots), gravity on under a
    rule and the swarms set apart. The
    episodes of chunk c are the next len(tasks) rows of attempts.jsonl, so the n-th task drawn
    here (counting from 0 over all chunks) is attempts.jsonl's line n (voxel/haiku.py joins
    language labels to a store this way). build: round(build x the chunk's scenes) build tasks
    (voxel/build.py) after those, from a generator seeded by (seed, 13).
    only="build": the build tasks alone (no suite, coverage or object tasks), drawn from the same
    (seed, 13) generator, so chunk c holds exactly the build tasks of chunk c of the full draw with
    the same seed, chunk and build share, played with the same run seed. Indices (attempts.jsonl
    lines, haiku item ids) then count build tasks only: a store and a Haiku export join only when
    both were made with --only build (haiku.store_options reads it from the store)."""
    if only is not None and only not in ONLY:
        raise ValueError(f"only: one of {ONLY}")
    if only == "build" and not build:
        raise ValueError("only build needs a build share above 0")
    rng = np.random.default_rng(seed)
    extra = np.random.default_rng((seed, 7))
    things = np.random.default_rng((seed, 11))
    builds = np.random.default_rng((seed, 13))
    for c in range(0, n_scenes, chunk):
        if only == "build":
            world, metas, tasks = None, [], []
            n_build = int(round(build * min(chunk, n_scenes - c)))
            if n_build:
                from beebot.vla.voxel.build import sample_build

                world, metas, tasks = sample_build(builds, n_build, robots=robots, counts=counts)
            else:
                continue
            run_seed = seed * 1_000_003 + c
            if rule is not None:
                world = replace(world, gravity=torch.ones_like(world.gravity))
            if spaced:
                world = apart(world, np.random.default_rng(run_seed))
            yield c, world, metas, tasks, run_seed
            continue
        world, metas, tasks = S.sample_tasks(rng, min(chunk, n_scenes - c), "train", robots=robots, check=check, counts=counts)
        world, metas, tasks = contrast(rng, world, metas, tasks, check=check)
        n_coverage = int(round(coverage * min(chunk, n_scenes - c)))
        if n_coverage:
            more, more_metas, more_tasks = S.sample_coverage(extra, n_coverage, robots=robots, counts=counts)
            world = VoxelWorld(*(torch.cat([getattr(world, f), getattr(more, f)]) for f in FIELDS))
            metas, tasks = metas + more_metas, tasks + more_tasks
        if objects:
            from beebot.vla.voxel import objects as O

            world = with_objects(world, MAX_OBJECTS)
            n_objects = int(round(objects * min(chunk, n_scenes - c)))
            if n_objects:
                more, more_metas, more_tasks = O.sample_objects(things, n_objects, robots=robots, counts=counts)
                world = concat([world, more])
                metas, tasks = metas + more_metas, tasks + more_tasks
        n_build = int(round(build * min(chunk, n_scenes - c)))
        if n_build:
            from beebot.vla.voxel.build import sample_build

            more, more_metas, more_tasks = sample_build(builds, n_build, robots=robots, counts=counts)
            world = concat([world, more])
            metas, tasks = metas + more_metas, tasks + more_tasks
        run_seed = seed * 1_000_003 + c
        if rule is not None:
            world = replace(world, gravity=torch.ones_like(world.gravity))
        if spaced:
            world = apart(world, np.random.default_rng(run_seed))
        yield c, world, metas, tasks, run_seed


def write(out_dir, n_scenes, seed=0, chunk=24, steps=S.EPISODE_STEPS, moves="all", robots=None, failure=None, compact=False, counts="uniform", fit=False, rule=None, expert="attractor", spaced=False, coverage=0.0, objects=0.0, build=0.0, only=None):
    """Write the store. moves: "all" or "cube" (MOVES). robots: (low, high) robot counts, with
    tasks fitted to the swarm (None: the world's density range). failure: (27,) per-action
    failure probabilities (world.failure_rates) or None. compact: write mass and actions in
    play_store's compact format instead of the dense .npy files. counts: how a scene's robot
    count is drawn from `robots` ("uniform" or "log", suite.draw_count). fit: draw only tasks
    the swarm can do also when robots is None (with robots that is always so). rule: a pivot
    rule (world.RULES) that the world enforces and the expert chooses within, with gravity on in
    every scene; None: the voxel world's own moves. expert: "attractor", or with a rule
    "planner" (free_cube.FreeCubeExpert). spaced: every swarm is set down again with no two
    robots touching before it is played (free_cube.apart). coverage: coverage tasks per scene
    drawn (draws). objects: object tasks per scene drawn (draws), played by objects.ObjectExpert
    (the attractor expert only); small.pt then holds each episode's object fields (OBJECT_FIELDS).
    build: build tasks per scene drawn (draws), played by build.make_expert while the others play
    as without them (build.Mixed). only: "build" plays the build tasks alone (draws).
    Returns the number of demonstrations."""
    if objects and expert != "attractor":
        raise ValueError("object tasks are demonstrated by the attractor expert (objects.ObjectExpert) only")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dims = (S.SIZE,) * 3
    writer = CompactWriter(out, dims) if compact else DenseWriter(out, dims, int(np.ceil(n_scenes * (len(S.FAMILIES) + coverage + objects + build))) + 1, steps)
    check = fit or robots is not None
    small = {k: [] for k in ("cheese_pos", "cheese_alive", "walls", "cheese_color", "gravity") + (OBJECT_FIELDS if objects else ())}
    subgoals, swarm, instructions = [], [], []
    labels = open(out / "labels.jsonl", "w")
    attempts = open(out / "attempts.jsonl", "w")
    row, tried, kept = 0, {}, {}
    t0 = time.time()
    for c, world, metas, tasks, run_seed in draws(n_scenes, seed, chunk, robots, counts, check, rule, spaced, coverage, objects, build, only):
        if objects:
            from beebot.vla.voxel.objects import ObjectExpert

            other = lambda some: ObjectExpert(some, moves=MOVES[moves], seed=run_seed, rule=rule)  # noqa: E731
        elif expert == "planner":
            other = lambda some: FreeCubeExpert(some, seed=run_seed, rule=rule)  # noqa: E731
        else:
            other = lambda some: S.Expert(some, moves=MOVES[moves], seed=run_seed, rule=rule)  # noqa: E731
        if any(task.family == "build" for task in tasks):
            from beebot.vla.voxel import build as B

            player = B.Mixed(tasks, other, lambda some: B.make_expert(some, moves=MOVES[moves], seed=run_seed, rule=rule))
        else:
            player = other(tasks)
        results, arrays = S.run(world, metas, tasks, player, steps=steps, seed=run_seed, record=True, failure=failure, rule=rule)
        for i, ((score, unasked), task) in enumerate(zip(results, tasks)):
            tried[task.family] = tried.get(task.family, 0) + 1
            is_demo = not unasked and score >= SUCCESS.get(task.family, 0.75)
            n_robots = int(world.mass[i].sum())
            attempts.write(json.dumps({"family": task.family, "kind": S.task_kind(task), "robots": n_robots, "score": round(float(score), 4), "unasked": bool(unasked), "e": row if is_demo else -1}) + "\n")
            if not is_demo:
                continue
            kept[task.family] = kept.get(task.family, 0) + 1
            writer.add(arrays["mass"][i].numpy(), arrays["actions"][i].numpy())
            small["cheese_pos"].append(arrays["cheese_pos"][i])
            small["cheese_alive"].append(arrays["cheese_alive"][i])
            small["walls"].append(world.walls[i])
            small["cheese_color"].append(world.cheese_color[i])
            small["gravity"].append(world.gravity[i])
            for name in OBJECT_FIELDS if objects else ():
                small[name].append(getattr(world, name)[i])
            subgoals.append(player.subgoals[i])
            swarm.append(n_robots)
            instructions.append(instruction_entries(task, steps))
            for j, entry in enumerate(instructions[-1]):  # one label row per instruction, over the steps it is in force
                end = instructions[-1][j + 1]["t"] if j + 1 < len(instructions[-1]) else steps
                part = S.instruction_at(task, entry["t"])
                labels.write(json.dumps({"e": row, "w0": entry["t"], "w1": end, "family": part.family, "form": part.spec.get("form", ""), "text": entry["text"]}) + "\n")
            row += 1
        labels.flush()
        attempts.flush()
        print(f"{min(c + chunk, n_scenes)} scenes, {row} demos, {time.time() - t0:.0f} s; kept/tried {json.dumps({f: f'{kept.get(f, 0)}/{tried[f]}' for f in sorted(tried)})}", flush=True)
    writer.close()
    config = {"seed": seed, "steps": steps, "moves": moves, "robots": None if robots is None else list(robots), "failure": None if failure is None else failure.tolist(), "counts": counts, "fit": check, "rule": rule, "expert": expert, "apart": spaced, "coverage": coverage, "objects": objects, "build": build, "only": only}
    torch.save({**{k: torch.stack(v) for k, v in small.items()}, "phases": [[] for _ in range(row)], "count": row, "robots": torch.tensor(swarm), "subgoals": subgoals, "instructions": instructions, "config": config}, out / "small.pt")
    (out / "done").write_text(str(row))
    return row


def instruction_entries(task, steps):
    """The instructions in force over an episode: [{"t": first step, "text": the instruction,
    "record": suite.canonical_record of the task it asks}], one entry, or one per part of a switch
    task (steps past the episode's end left out)."""
    parts = task.parts or [task]
    starts = task.spec["steps"] if task.parts else [0]
    return [{"t": int(t), "text": part.text, "record": S.canonical_record(part)} for t, part in zip(starts, parts) if t < steps]


def parse_failure(text):
    """(27,) failure probabilities from "0.15" or "0.15,up_over=0.3,vertical=0.2": the rate of
    every move, then rates of single move classes (world.MOVE_CLASSES)."""
    first, *rest = text.split(",")
    by_class = {name: float(value) for name, value in (part.split("=") for part in rest)}
    return failure_rates(float(first), **by_class)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenes", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cube", action="store_true", help=f"pivoting-cube demonstrations: --moves cube --steps {S.CUBE_STEPS} --robots {S.ROBOTS[0]} {S.ROBOTS[1]} --failure {FAILURE_RATE}, each unless given")
    parser.add_argument("--free-cube", action="store_true", help="demonstrations real cubes can carry out: --cube with --rule free_cube --expert planner, each unless given")
    parser.add_argument("--rule", choices=RULES, default=None, help="a pivot rule the world enforces and the expert chooses within (gravity on in every scene)")
    parser.add_argument("--expert", choices=EXPERTS, default=None, help="planner needs --rule")
    parser.add_argument("--apart", action="store_true", help="set every swarm down again so that no two robots touch at the start")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--moves", choices=sorted(MOVES), default=None)
    parser.add_argument("--robots", type=int, nargs=2, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--failure", default=None, help="move failure probability, optionally per class: 0.15,up_over=0.3")
    parser.add_argument("--compact", action="store_true", help="store mass and actions by robot (play_store's compact format) instead of dense grids")
    parser.add_argument("--counts", choices=S.COUNTS, default="uniform", help="how the robot count is drawn over --robots; log makes small swarms as common as large ones")
    parser.add_argument("--coverage", type=float, default=0.0, help="coverage tasks (suite.sample_coverage: stay, selectors, switches, ...) per scene, drawn after the suite's own")
    parser.add_argument("--objects", type=float, default=0.0, help="object tasks (objects.sample_objects: properties, blocks, zones, reasoning) per scene, drawn after the coverage tasks")
    parser.add_argument("--build", type=float, default=0.0, help="build tasks (voxel/build.py: build a named object or something for a use) per scene, drawn after the coverage tasks; they need long episodes (build.episode_steps), so give --steps")
    parser.add_argument("--only", choices=ONLY, default=None, help="draw only this family's tasks (with --build: the build tasks of the full draw, nothing else); attempts and labels then count those tasks only")
    parser.add_argument("--fit", action="store_true", help="draw only tasks the scene's swarm can do, also without --robots (with --robots or --cube this is always on)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cube = args.cube or args.free_cube
    steps = args.steps or (S.CUBE_STEPS if cube else S.EPISODE_STEPS)
    moves = args.moves or ("cube" if cube else "all")
    robots = args.robots or (S.ROBOTS if cube else None)
    failure = parse_failure(args.failure) if args.failure else failure_rates() if cube else None
    rule = args.rule or ("free_cube" if args.free_cube else None)
    expert = args.expert or ("planner" if args.free_cube else "attractor")
    if expert == "planner" and rule is None:
        parser.error("--expert planner needs --rule")
    write(args.out, args.scenes, args.seed, args.chunk, steps, moves, robots, failure, args.compact, args.counts, args.fit, rule, expert, args.apart, args.coverage, args.objects, args.build, args.only)


if __name__ == "__main__":
    main()
