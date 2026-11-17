"""How much and how varied the LLM play is, per player.

    uv run play-stats [/Users/hrishik/beebot-play]

Per player: programs and programs per hour, the share of scenes that errored, how often each fact
type occurs per 100 episodes (over whole episodes and over the phases the programs marked), and
how spread the phases are over combinations of fact types (distinct combinations and the entropy
of their distribution, in bits). A play set that keeps doing the same few things shows up as few
combinations and low entropy.
"""

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from beebot.vla.facts import window_facts
from beebot.vla.play_cli import phase_segments

DEFAULT_ROOT = Path("/Users/hrishik/beebot-play")
TYPES = ("move", "split", "merge", "box", "push", "eat", "around", "away")


class PlayerStats:
    def __init__(self, name):
        self.name = name
        self.programs = 0
        self.scenes = 0
        self.errors = 0
        self.times = []
        self.episode_types = Counter()
        self.phase_types = Counter()
        self.phases = 0
        self.combinations = Counter()

    def add(self, run_dir):
        meta = json.loads((run_dir / "meta.json").read_text())
        episodes = np.load(run_dir / "episodes.npz")
        self.programs += 1
        self.times.append(datetime.strptime(meta["time"], "%Y-%m-%dT%H:%M:%S%z"))
        for i, error in enumerate(meta["errors"]):
            self.scenes += 1
            if error is not None:
                self.errors += 1
                continue
            static = {k: episodes[k][i] for k in ("walls", "cheese_color", "box_rect", "box_color")}

            def snap(t):
                return {"mass": episodes["mass"][i, t], "cheese_pos": episodes["cheese_pos"][i, t], "cheese_alive": episodes["cheese_alive"][i, t]}

            last = episodes["mass"].shape[1] - 1
            self.episode_types.update(window_facts(static, snap(0), snap(last)).types)
            for _, t0, t1 in phase_segments(meta["phases"][i]):
                types = window_facts(static, snap(t0), snap(t1)).types
                self.phases += 1
                self.phase_types.update(types)
                self.combinations["+".join(sorted(types)) or "nothing"] += 1

    def report(self):
        hours = (max(self.times) - min(self.times)).total_seconds() / 3600 if len(self.times) > 1 else 0.0
        rate = f"{(self.programs - 1) / hours:.1f}/h" if hours > 0 else "n/a"
        ok = self.scenes - self.errors
        lines = [f"{self.name}: {self.programs} programs ({rate}), {self.scenes} scenes, {self.errors / max(self.scenes, 1):.0%} errored, {self.phases} phases"]
        per_episode = " ".join(f"{t} {100 * self.episode_types[t] / max(ok, 1):.0f}" for t in TYPES)
        per_phase = " ".join(f"{t} {100 * self.phase_types[t] / max(self.phases, 1):.0f}" for t in TYPES)
        lines.append(f"  per 100 episodes: {per_episode}")
        lines.append(f"  per 100 phases:   {per_phase}")
        total = sum(self.combinations.values())
        entropy = -sum(c / total * math.log2(c / total) for c in self.combinations.values()) if total else 0.0
        common = ", ".join(f"{k} {c}" for k, c in self.combinations.most_common(6))
        lines.append(f"  phase fact combinations: {len(self.combinations)} distinct, entropy {entropy:.2f} bits; most common: {common}")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    for workspace in sorted(Path(args.root).iterdir()):
        data = workspace / "data"
        if not data.is_dir():
            continue
        stats = PlayerStats(workspace.name)
        for run_dir in sorted(data.iterdir()):
            if (run_dir / "meta.json").exists():
                stats.add(run_dir)
        if stats.programs:
            print(stats.report())


if __name__ == "__main__":
    main()
