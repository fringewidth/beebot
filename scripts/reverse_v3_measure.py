"""Reverse demos at scale (runs/handoff/reverse-v3.md): take_apart on goals drawn from a library by robot count,
one goal at a time, CPU only, one torch thread, a CPU-time budget per episode.

    nice -n 10 .venv/bin/python scripts/reverse_v3_measure.py --source gen_complex --lo 251 --hi 500 --n 5 \
        --out runs/reverse_v3/after-gen_complex-251.jsonl [--impl path/to/reverse.py] [--save runs/reverse_v3/eps]

Sources: gen_complex (end states B, starts A when the goal has one; tricks with A = B included; kinds scene, edit,
retarget, trick drawn in turn), gen_medium, v2 (the libraries in data/goal_library_v2 named in V2_SOURCES, pooled). Goals are drawn with
--seed from those whose robot count is in [lo, hi], the same draw for every --impl. One JSON line per goal: ok (replay passed and the start is natural or
exactly A), end, steps, cpu seconds of the walk and of the replay, robots moving per step (mean over steps), or
timeout. --impl loads reverse.py from a file (the version before, for a before/after table).
"""

import argparse
import importlib.util
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel import goal_gen_complex as C
from beebot.vla.voxel import goal_library as G

V2_ROOT = Path("data/goal_library_v2")
V2_SOURCES = ("craftassist_houses", "craftassist_parts", "hack337", "iglu_multiturn", "iglu_single", "stabletext2brick")  # complete on 2026-11-13 (more were being written)


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def load_impl(path):
    if path is None:
        from beebot.vla.voxel import reverse

        return reverse
    spec = importlib.util.spec_from_file_location("reverse_impl", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["reverse_impl"] = module
    spec.loader.exec_module(module)
    return module


def draw(source, lo, hi, n, seed, libraries=V2_SOURCES):
    """[(id, kind, text, B cells, A cells or None, meta)] drawn from a source by robot count (v2: the
    libraries named, pooled)."""
    rng = np.random.default_rng(seed)
    if source == "gen_complex":  # kinds in turn (scene, edit, retarget, trick, in a drawn order), a random goal of each
        lines, cells, starts = C.load()
        pools = {}
        for i, line in enumerate(lines):
            if lo <= len(cells[i]) <= hi:
                pools.setdefault(line["kind"], []).append(i)
        kinds = [sorted(pools)[k] for k in rng.permutation(len(pools))]
        for kind in kinds:
            pools[kind] = rng.permutation(pools[kind]).tolist()
        pick = []
        while len(pick) < n and any(pools.values()):
            for kind in kinds:
                if pools[kind] and len(pick) < n:
                    pick.append(pools[kind].pop())
        return [(lines[i]["id"], lines[i]["generator"], lines[i]["turns"][-1], cells[i], starts[i], lines[i]) for i in pick]
    if source == "gen_medium":
        lines, cells = G.load("gen_medium")
        pool = [i for i, line in enumerate(lines) if lo <= len(cells[i]) <= hi]
        pick = rng.choice(pool, size=min(n, len(pool)), replace=False).tolist()
        return [(lines[i]["id"], lines[i]["generator"], lines[i]["text"], cells[i], None, lines[i]) for i in pick]
    rows = []
    for name in libraries:
        lines, cells = G.load(name, root=V2_ROOT)
        rows += [(lines[i]["id"], name, lines[i]["text"], cells[i], None, lines[i]) for i in range(len(lines)) if lo <= len(cells[i]) <= hi]
    rows.sort(key=lambda r: r[0])
    pick = rng.choice(len(rows), size=min(n, len(rows)), replace=False).tolist()
    return [rows[i] for i in pick]


def to_goal(reverse, source, cells, start, meta, text):
    if source == "gen_complex":
        goal = {"cells": cells.tolist(), "start_cells": None if start is None else start.tolist(), "params": meta["params"], "turns": meta["turns"], "generator": meta["generator"]}
        g = C.to_reverse_goal(goal)
    else:
        g = G.to_goal(cells, text)
    # rebuild through the implementation under test (its own Goal and blank)
    world = reverse.blank()
    world.mass[0] = g.world.mass[0]
    return reverse.Goal(family=g.family, text=g.text, world=world, target=g.target, start=g.start, unused=g.unused)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["gen_complex", "gen_medium", "v2"])
    parser.add_argument("--lo", type=int, required=True)
    parser.add_argument("--hi", type=int, required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=900, help="CPU seconds per episode (walk only)")
    parser.add_argument("--impl", default=None)
    parser.add_argument("--pick", type=int, nargs="*", default=None, help="only these positions of the draw")
    parser.add_argument("--save", default=None, help="directory: save each episode (torch) for pictures")
    parser.add_argument("--libraries", default="v3", help="v2 source: v3 (V2_SOURCES, the six of reverse-v3.md) or all (every library in V2_ROOT, reverse-v4.md)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    libraries = V2_SOURCES if args.libraries == "v3" else tuple(sorted(p.stem for p in V2_ROOT.glob("*.jsonl") if p.stem not in G.DROPPED_SOURCES))
    torch.set_num_threads(1)
    reverse = load_impl(args.impl)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGVTALRM, _alarm)
    with open(args.out, "w") as handle:
        drawn = draw(args.source, args.lo, args.hi, args.n, args.seed, libraries)
        for gid, kind, text, cells, start, meta in [drawn[k] for k in args.pick] if args.pick is not None else drawn:
            row = {"id": gid, "kind": kind, "source": args.source, "bin": f"{args.lo}-{args.hi}", "robots": int(len(cells)), "guided": start is not None, "text": text}
            goal = to_goal(reverse, args.source, cells, start, meta, text)
            rng = np.random.default_rng(args.seed)
            cpu, wall = time.process_time(), time.time()
            signal.setitimer(signal.ITIMER_VIRTUAL, args.timeout)
            try:
                episodes, stats = reverse.take_apart([goal], rng)
                signal.setitimer(signal.ITIMER_VIRTUAL, 0)
            except Timeout:
                row.update(ok=False, timeout=True, cpu=round(time.process_time() - cpu, 1), wall=round(time.time() - wall, 1))
                print(json.dumps(row), flush=True)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                continue
            walk_cpu = time.process_time() - cpu
            ep = episodes[0]
            c1 = time.process_time()
            passed = bool(reverse.replay(episodes)["passed"][0])
            moving = (ep.actions != reverse.STAY).flatten(1).sum(dim=1).float() if ep.steps else torch.zeros(1)
            row.update(
                ok=passed and bool(ep.natural),
                replay=passed,
                end=ep.end,
                steps=int(ep.steps),
                distance=int(ep.distance),
                cpu=round(walk_cpu, 1),
                replay_cpu=round(time.process_time() - c1, 1),
                wall=round(time.time() - wall, 1),
                moving=round(float(moving.mean()), 2),
                moving_max=int(moving.max()),
                kinds={k: ep.kinds.count(k) for k in sorted(set(ep.kinds or []))},
                scaffolds=int(stats["scaffolds"][0]),
                digs=int(stats.get("digs", [0])[0]),
            )
            if goal.start is None and hasattr(reverse, "natural_bands"):  # natural: which statistics of the start are out of band
                low, high = reverse.natural_bands(goal.world, goal.target, np.random.default_rng(args.seed))
                stats = torch.tensor(ep.stats)
                row["out_of_band"] = [name for name, v, lo_, hi_ in zip(reverse.STATS, stats.tolist(), low.tolist(), high.tolist()) if not lo_ <= v <= hi_]
            if args.save:
                Path(args.save).mkdir(parents=True, exist_ok=True)
                torch.save({"row": row, "mass": ep.mass, "actions": ep.actions, "walls": ep.world.walls[0], "text": text}, Path(args.save) / f"{gid}.pt")
            print(json.dumps(row), flush=True)
            handle.write(json.dumps(row) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
