"""Coverage-steered nudges for LLM play sessions.

    uv run play-coverage [--recent 100] [--out logs/vla-play-nudge.txt]

Compares how often each behaviour type (window_facts types over whole episodes) shows up in the
most recent programs with how often it showed up in the first programs collected (the sessions'
unsteered mix), and writes one nudge line naming the two types that have fallen furthest behind.
vla_play_nudge.sh types that line into idle sessions. Run it every few minutes in a loop.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from beebot.vla.facts import window_facts
from beebot.vla.play_data import PLAY_ROOT, run_dirs

PHRASES = {
    "split": "splitting into several groups (two, three, four) that go to different places",
    "merge": "bringing separate groups back together",
    "box": "going into and out of boxes, or pushing cheeses into boxes",
    "push": "pushing cheeses to particular places",
    "eat": "eating some cheeses and leaving others",
    "around": "surrounding a cheese without eating it",
    "away": "keeping away from cheeses or walls",
    "move": "travelling to particular regions of the grid",
}
FALLBACK = "Keep going: write and run the next program, something you haven't tried yet."


def episode_types(run_dir):
    meta = json.loads((run_dir / "meta.json").read_text())
    episodes = np.load(run_dir / "episodes.npz")
    counts, ok = Counter(), 0
    last = episodes["mass"].shape[1] - 1
    for i, error in enumerate(meta["errors"]):
        if error is not None:
            continue
        ok += 1
        static = {k: episodes[k][i] for k in ("walls", "cheese_color", "box_rect", "box_color")}

        def snap(t):
            return {"mass": episodes["mass"][i, t], "cheese_pos": episodes["cheese_pos"][i, t], "cheese_alive": episodes["cheese_alive"][i, t]}

        counts.update(window_facts(static, snap(0), snap(last)).types)
    return counts, ok


def rates(run_dirs):
    total, episodes = Counter(), 0
    for run_dir in run_dirs:
        counts, ok = episode_types(run_dir)
        total.update(counts)
        episodes += ok
    return {kind: total[kind] / max(episodes, 1) for kind in PHRASES}


def nudge_line(root=PLAY_ROOT, recent=100):
    dirs = sorted(run_dirs(root), key=lambda d: json.loads((d / "meta.json").read_text())["time"])
    if len(dirs) < 2 * recent:
        return FALLBACK
    baseline, now = rates(dirs[:recent]), rates(dirs[-recent:])
    behind = sorted(PHRASES, key=lambda k: now[k] / max(baseline[k], 1e-6))[:2]
    return (
        "Keep going: write and run the next program, something you haven't tried yet. "
        f"Across all players, these have become rare lately: {PHRASES[behind[0]]}; and {PHRASES[behind[1]]}. "
        "Build something around one of them, combined with anything else."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recent", type=int, default=100)
    parser.add_argument("--out", default="logs/vla-play-nudge.txt")
    args = parser.parse_args()
    line = nudge_line(recent=args.recent)
    Path(args.out).write_text(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
