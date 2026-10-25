"""Collect every finished run into site/ for the hill-climb page.

    uv run report

Writes site/data.json and copies the GIFs the page shows into site/media/.
Scores the hand-written baseline once and keeps it in runs/baseline/.
"""

import dataclasses
import json
import shutil
from pathlib import Path

import torch

from beebot.evaluate import SMELL_LENGTH, evaluate
from beebot.model import GreedyBaseline
from beebot.render import render_examples
from beebot.train import LEVELS, Config

RUNS = Path("runs")
SITE = Path("site")
# Run settings that don't change what the model learns; left out of the "change" column.
NOT_A_CHANGE = {"name", "device", "log_every", "init"}


def ensure_baseline(device):
    run_dir = RUNS / "baseline"
    if (run_dir / "summary.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline = GreedyBaseline(SMELL_LENGTH)
    summary = {"name": "baseline", "val": evaluate(baseline, device, "val")}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    render_examples(baseline, device, run_dir / "videos")


def config_changes(config):
    defaults = dataclasses.asdict(Config())
    return {k: v for k, v in config.items() if k not in NOT_A_CHANGE and defaults.get(k) != v}


def curve(run_dir):
    points = []
    for line in (run_dir / "log.jsonl").read_text().splitlines():
        entry = json.loads(line)
        points.append([entry["minutes"], entry["level"], entry["success_ema"]])
    return points


def copy_videos(run_dir, name):
    videos = []
    for gif in sorted((run_dir / "videos").glob("*.gif")):
        target = SITE / "media" / name / gif.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(gif, target)
        videos.append(f"media/{name}/{gif.name}")
    return videos


def main():
    device = torch.device("mps")
    ensure_baseline(device)
    if (SITE / "media").exists():
        shutil.rmtree(SITE / "media")

    runs = []
    for run_dir in sorted(RUNS.iterdir()):
        if run_dir.name == "baseline" or not (run_dir / "summary.json").exists():
            continue
        summary = json.loads((run_dir / "summary.json").read_text())
        config = json.loads((run_dir / "config.json").read_text())
        runs.append(
            {
                "name": run_dir.name,
                "changes": config_changes(config),
                "minutes": config["minutes"],
                "iterations": summary["iterations"],
                "final_level": summary["final_level"],
                "val": summary["val"],
                "curve": curve(run_dir),
                "videos": copy_videos(run_dir, run_dir.name),
            }
        )

    baseline = json.loads((RUNS / "baseline" / "summary.json").read_text())
    data = {
        "levels": LEVELS,
        "baseline": {"val": baseline["val"], "videos": copy_videos(RUNS / "baseline", "baseline")},
        "runs": runs,
    }
    SITE.mkdir(exist_ok=True)
    (SITE / "data.json").write_text(json.dumps(data))
    print(f"{len(runs)} runs written to {SITE / 'data.json'}")


if __name__ == "__main__":
    main()
