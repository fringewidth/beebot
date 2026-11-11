"""What 8 to 12 cubes on an empty flat floor can do (README "Demo: ten cubes on an empty floor"):
every family of the suite, coverage, object and build makers drawn on empty floors (demos --empty
--keep --fit, --scaled unless --unscaled) and played by the expert as demos --demo plays them
(switch_cube, cube moves, failures 0.1 and 0.23 up and over, 72 steps, gravity on).

    PYTHONPATH=src OMP_NUM_THREADS=2 nice python scripts/vla_voxel_demo_feasibility.py --n 48 --seed 31 --out runs/demo/feasibility.jsonl
    PYTHONPATH=src OMP_NUM_THREADS=2 nice python scripts/vla_voxel_demo_feasibility.py --n 96 --seed 33 --jobs motion:goto,motion:move,subset:move --out runs/demo/directions.jsonl
    PYTHONPATH=src OMP_NUM_THREADS=2 nice python scripts/vla_voxel_demo_feasibility.py --n 48 --seed 32 --unscaled --no-fit --only suite,objects --out runs/demo/unscaled.jsonl

Rows: one per draw {"draw", "label", "requested", "made"} (tries and the tasks they gave), then one
per episode {"sampler", "label", "family", "kind", "detail", "robots", "score", "passed",
"problem"}: passed is a demonstration (checker pass and demos.maker_problem None). Prints a table
by detail (letter size, tower column and height, region, direction, ...) at the end.
"""

import argparse
import collections
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel import build as B
from beebot.vla.voxel import demos as D
from beebot.vla.voxel import objects as O
from beebot.vla.voxel import suite as S
from beebot.vla.voxel.world import CUBE_MOVES

RULE = "switch_cube"
ROBOTS = tuple(D.PRESETS["demo"]["robots"])
FAILURE = D.parse_failure("0.1,up_over=0.23")
STEPS = S.CUBE_STEPS


def detail(task):
    """A finer name than suite.task_kind: letter or wall size, tower column and height, region,
    direction, axis, group count."""
    kind = S.task_kind(task)
    spec = task.spec
    if task.family == "formation":
        label = task.plan[0]["label"] if task.plan else {}
        m = int(spec["mask"].sum())
        if label.get("kind") == "letters":
            full = int(S.letter_mask(label["word"], label["standing"]).sum())
            return f"letters:{len(label['word'])}{'-standing' if label['standing'] else ''}:{'half' if m != full else 'full'}"
        full = int(S.shape_mask(label.get("shape", "wall")).sum())
        return f"shape:{label.get('shape')}:{'half' if m != full else 'full'}"
    if kind == "tower":
        points = task.plan[0]["points"] if task.plan else []
        return f"tower:{'1-wide' if points and len(points) == spec['height'] else '2x2'}:h{spec['height']}"
    if kind.startswith("parallel") and "tower" in kind and task.plan:
        side = next("1-wide" if len(part["points"]) == x else "2x2" for part, (job, x) in zip(task.plan[0]["parts"], spec["jobs"]) if job == "tower")
        return f"{kind}:{side}"
    if task.parts or "who" not in spec:
        return kind
    field = {"tower": "height", "goto": "region", "move": "toward", "line": "axis", "groups": "count"}.get(spec["kind"])
    return f"{kind}:{spec[field]}" if field else kind


def play(world, metas, tasks, seed, objects=False):
    """Rows of one batch, played as demos.write plays it."""
    if objects:
        other = lambda some: O.ObjectExpert(some, moves=CUBE_MOVES, seed=seed, rule=RULE)  # noqa: E731
    else:
        other = lambda some: S.Expert(some, moves=CUBE_MOVES, seed=seed, rule=RULE)  # noqa: E731
    if any(t.family == B.FAMILY for t in tasks):
        player = B.Mixed(tasks, other, lambda some: B.make_expert(some, moves=CUBE_MOVES, seed=seed, rule=RULE))
    else:
        player = other(tasks)
    world = replace(world, gravity=torch.ones_like(world.gravity))
    results, arrays = S.run(world, metas, tasks, player, steps=STEPS, seed=seed, record=True, failure=FAILURE, rule=RULE)
    rows = []
    for i, ((score, unasked), task) in enumerate(zip(results, tasks)):
        passed = not unasked and score >= S.SUCCESS.get(task.family, S.PASS_MARK)
        problem = D.maker_problem(task, world, metas[i], i, arrays, RULE, float(score)) if passed else None
        rows.append({"family": task.family, "kind": S.task_kind(task), "detail": detail(task), "robots": int(world.mass[i].sum()), "score": round(float(score), 3), "passed": bool(passed and problem is None), "problem": problem})
    return rows


def table(rows):
    """Lines: made / requested per draw, then passed per (label, detail)."""
    lines = ["made / tries:"]
    lines += [f"  {r['draw']:9s} {r['label']:24s} {r['made']}/{r['requested']}" for r in rows if "draw" in r]
    found = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for r in rows:
        if "passed" in r:
            entry = found[(r["label"], r["detail"])]
            entry[0] += r["passed"]
            entry[1] += 1
            entry[2][r["problem"]] += 1
    lines.append("passed:")
    for (label, name), (passed, n, problems) in sorted(found.items()):
        why = {k: v for k, v in problems.items() if k}
        lines.append(f"  {label:24s} {name:40s} {passed:3d}/{n:<3d} {passed / n:.2f} {why if why else ''}")
    total = sum(r["passed"] for r in rows if "passed" in r)
    episodes = sum("passed" in r for r in rows)
    lines.append(f"all: {total}/{episodes}")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=48, help="tries per suite family, object family and switch job; half that per motion and subset job (and per object family); twice that for build")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--unscaled", action="store_true", help="the makers' sizes as before (no --scaled)")
    parser.add_argument("--no-fit", action="store_true", help="no fit check: also what the swarm cannot do")
    parser.add_argument("--only", default=None, help="comma list of suite, coverage, objects, build")
    parser.add_argument("--jobs", default=None, help="only these coverage jobs, --n tries each: motion:goto,subset:move,...")
    args = parser.parse_args()
    torch.set_num_threads(2)
    scaled, check = not args.unscaled, not args.no_fit
    common = {"robots": ROBOTS, "check": check, "keep": True, "empty": True}
    only = set() if args.jobs else None if args.only is None else set(args.only.split(","))
    rng = np.random.default_rng(args.seed)
    drawn = []  # (sampler, label, tries, (world, metas, tasks))
    if args.jobs:
        for item in args.jobs.split(","):
            family, job = item.split(":")
            drawn.append(("coverage", item, args.n, S.sample_coverage(rng, args.n, family=family, job=job, scaled=scaled, **common)))
    if only is None or "suite" in only:
        for family in S.FAMILIES:
            drawn.append(("suite", family, args.n, S.sample_tasks(rng, args.n, "train", family=family, scaled=scaled, **common)))
    if only is None or "coverage" in only:
        for family, jobs in (("motion", S.MOTION_JOBS), ("subset", tuple(S.SUBSET_JOBS))):
            for job in jobs:
                drawn.append(("coverage", f"{family}:{job}", args.n // 2, S.sample_coverage(rng, args.n // 2, family=family, job=job, scaled=scaled, **common)))
        for then in sorted(set(S.SWITCH_THEN)):
            drawn.append(("coverage", f"switch:>{then}", args.n, S.sample_coverage(rng, args.n, family="switch", job=then, scaled=scaled, **common)))
    if only is None or "objects" in only:
        for family in O.OBJECT_FAMILIES:
            drawn.append(("objects", family, args.n // 2, O.sample_objects(rng, args.n // 2, family=family, scaled=scaled, **common)))
        drawn.append(("objects", "reasoning:conditional", args.n, O.sample_objects(rng, args.n, family="reasoning", kind="conditional", scaled=scaled, **common)))
    if only is None or "build" in only:
        drawn.append(("build", "build", args.n * 2, B.sample_build(rng, args.n * 2, **common)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"draw": sampler, "label": label, "requested": tries, "made": len(tasks)} for sampler, label, tries, (_, _, tasks) in drawn]
    t0 = time.time()
    seed = 1000 + args.seed
    for sampler, label, _, (world, metas, tasks) in drawn:
        for start in range(0, len(tasks), 48):
            index = list(range(start, min(start + 48, len(tasks))))
            seed += 1
            for row in play(world.index(torch.tensor(index)), [metas[k] for k in index], [tasks[k] for k in index], seed, objects=sampler == "objects"):
                rows.append({"sampler": sampler, "label": label, **row})
        print(f"played {label} ({len(tasks)}) at {time.time() - t0:.0f} s", flush=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("\n".join(table(rows)))


if __name__ == "__main__":
    main()
