"""The one command an LLM play session may run: play a program on fresh scenes, save the episodes,
and print what happened.

    play-run PROGRAM.py --player haiku-1 [--scenes 10] [--size 16]

Episodes go to <data>/<program id>/ (episodes.npz, program.py, meta.json). The printed summary
gives, per scene, any error and the facts for each phase the program marked.
"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from beebot.vla.data import episode_steps
from beebot.vla.facts import window_facts
from beebot.vla.play import already_sandboxed, run_program
from beebot.vla.world import PHYSICS, sample_world

DEFAULT_DATA = Path("/Users/hrishik/beebot/data/vla/play")


def phase_segments(phases):
    """[(phase, t0, t1)] runs of equal phase labels; t1 is the snapshot index after the run's last step."""
    segments = []
    start = 0
    for t in range(1, len(phases) + 1):
        if t == len(phases) or phases[t] != phases[start]:
            segments.append((phases[start], start, t))
            start = t
    return segments


def summarize(episodes, i):
    static = {k: episodes[k][i] for k in ("walls", "cheese_color", "box_rect", "box_color")}

    def snap(t):
        return {"mass": episodes["mass"][i, t], "cheese_pos": episodes["cheese_pos"][i, t], "cheese_alive": episodes["cheese_alive"][i, t]}

    lines = []
    for phase, t0, t1 in phase_segments(episodes["phases"][i]):
        facts = window_facts(static, snap(t0), snap(t1))
        what = " ".join(facts.lines) if facts.eventful else "Nothing notable happened."
        lines.append(f"  steps {t0}-{t1} [{phase or 'no phase'}]: {what}")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("program")
    parser.add_argument("--player", required=True)
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=0, help="default: 64 at size 16, scaled with the size")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--program-id", default="", help="name for the saved run (default: time, player and a random tag)")
    parser.add_argument("--quiet", action="store_true", help="print only the one-line result, not the per-scene facts")
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=60.0,
        help="pace: a run takes at least this long, so a play session makes at most one request a minute",
    )
    parser.add_argument(
        "--inside-sandbox",
        action="store_true",
        help="this command runs inside another sandbox (e.g. codex's); programs then run under that one",
    )
    args = parser.parse_args()
    if args.inside_sandbox and not already_sandboxed():
        raise SystemExit("--inside-sandbox given, but this process is not sandboxed; refusing to run programs unsandboxed")

    started = time.monotonic()
    source = Path(args.program).read_text()
    program_id = args.program_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{args.player}-{np.random.default_rng().integers(1 << 30):08x}"
    seed = np.random.default_rng().integers(1 << 62)
    steps = args.steps or episode_steps(args.size)
    world = sample_world(np.random.default_rng(seed), args.scenes, size=args.size)
    episodes = run_program(source, world, steps=steps, inside_sandbox=args.inside_sandbox)

    out = Path(args.data) / program_id
    out.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in episodes.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(out / "episodes.npz", **arrays)
    shutil.copy(args.program, out / "program.py")
    meta = {
        "program_id": program_id,
        "player": args.player,
        "seed": int(seed),
        "size": args.size,
        "physics": PHYSICS,
        "steps": steps,
        "phases": episodes["phases"],
        "errors": episodes["errors"],
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "meta.json").write_text(json.dumps(meta))

    time.sleep(max(0.0, args.min_seconds - (time.monotonic() - started)))
    ok = sum(e is None for e in episodes["errors"])
    print(f"program {program_id}: {ok}/{args.scenes} scenes ran to the end")
    if args.quiet:
        return
    for i, error in enumerate(episodes["errors"]):
        if error is not None:
            print(f"scene {i}: error: {error}")
            continue
        print(f"scene {i}:")
        print("\n".join(summarize(episodes, i)))


if __name__ == "__main__":
    main()
