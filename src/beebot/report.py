"""Collect every stage's finished runs into site/ for the project page.

    uv run report

Writes site/data.json, copies the GIFs the page shows into site/media/, and
copies the written logs (JOURNAL.md, HILLCLIMB.md) next to it.
Scores the current stage's hand-written baseline once and keeps it in its
baseline/ folder.
"""

import dataclasses
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import torch

from beebot.evaluate import evaluate
from beebot.model import StickyBaseline
from beebot.render import render_examples
from beebot.train import LEVELS, Config

SITE = Path("site")
DOCUMENTS = ["JOURNAL.md", "HILLCLIMB.md"]
# Run settings that don't change what the model learns; left out of the "change" column.
NOT_A_CHANGE = {"name", "device", "log_every", "init", "use_checkpoint"}
# Only each stage's best runs get videos on the page, to keep the site small.
RUNS_WITH_VIDEOS = {"current": 2, "closed": 1}

# Newest first. "reference" is the run whose config counts as that stage's
# defaults; None means the current Config defaults.
STAGES = [
    {
        "id": "v3",
        "title": "Terrain",
        "folder": "runs",
        "reference": None,
        "status": "current",
        "summary": "No smell, pheromones, the whole swarm must gather, and random cave-like walls covering up to 30% of the world.",
    },
    {
        "id": "v2",
        "title": "No smell",
        "folder": "runs_v2",
        "reference": "v2r1-lr1e3",
        "status": "closed",
        "summary": "Touch only. Robots lay pheromones, only robots write them, and a scrap of a robot moves in proportion to its mass. Late in the stage, success also required the whole swarm to gather.",
    },
    {
        "id": "v1",
        "title": "Smell",
        "folder": "runs_v1",
        "reference": "sanity",
        "status": "closed",
        "summary": "Every square smells the cheese, so the environment hands each robot an arrow. Saturated near the physical limit; final test 0.267 vs baseline 0.405.",
    },
]


def finite(value):
    """Replace NaN and infinity with None: they aren't valid JSON, and the page can't load them."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    return value


def ensure_baseline(folder, device):
    run_dir = folder / "baseline"
    if (run_dir / "summary.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline = StickyBaseline()
    summary = {"name": "baseline", "val": evaluate(baseline, device, "val")}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    render_examples(baseline, device, run_dir / "videos")


def config_changes(config, defaults):
    return {k: v for k, v in config.items() if k not in NOT_A_CHANGE and k in defaults and defaults[k] != v}


def curve(run_dir):
    points = []
    for line in (run_dir / "log.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if "level" in entry:
            points.append([entry["minutes"], entry["level"], entry["success_ema"]])
    return points


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def copy_videos(run_dir, stage_id, name):
    videos = []
    for gif in sorted((run_dir / "videos").glob("*.gif")):
        target = SITE / "media" / stage_id / name / gif.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(gif, target)
        videos.append(f"media/{stage_id}/{name}/{gif.name}")
    return videos


def collect_stage(stage):
    folder = Path(stage["folder"])
    if stage["reference"]:
        defaults = read_json(folder / stage["reference"] / "config.json")
    else:
        defaults = dataclasses.asdict(Config())

    runs = []
    for run_dir in sorted(folder.iterdir()) if folder.exists() else []:
        if run_dir.name == "baseline" or not (run_dir / "summary.json").exists():
            continue
        summary = read_json(run_dir / "summary.json")
        config = read_json(run_dir / "config.json")
        runs.append(
            {
                "name": run_dir.name,
                "changes": config_changes(config, defaults),
                "minutes": config["minutes"],
                "iterations": summary["iterations"],
                "final_level": summary["final_level"],
                "diverged": summary.get("diverged", False),
                "val": summary["val"],
                "val_ring_only": summary.get("val_ring_only"),
                "final_test": read_json(run_dir / "final_test.json"),
                "curve": curve(run_dir),
                "probes": read_json(run_dir / "probes.json"),
                "run_dir": run_dir,
            }
        )

    runs.sort(key=lambda r: r["val"]["score"])
    # A privileged control can outscore every real run, so the best run without it always gets videos too.
    best_blind = next((r["name"] for r in runs if not r["changes"].get("privileged")), None)
    for rank, run in enumerate(runs):
        run_dir = run.pop("run_dir")
        keep = rank < RUNS_WITH_VIDEOS[stage["status"]] or run["name"] == best_blind
        run["videos"] = copy_videos(run_dir, stage["id"], run["name"]) if keep else []

    baseline_dir = folder / "baseline"
    baseline = read_json(baseline_dir / "summary.json")
    return {
        "id": stage["id"],
        "title": stage["title"],
        "status": stage["status"],
        "summary": stage["summary"],
        "levels": len(LEVELS),
        "baseline": {
            "val": baseline["val"] if baseline else None,
            "videos": copy_videos(baseline_dir, stage["id"], "baseline") if baseline else [],
            "probes": read_json(baseline_dir / "probes.json"),
        },
        "runs": runs,
    }


def main():
    device = torch.device("mps")
    current = Path(STAGES[0]["folder"])
    current.mkdir(exist_ok=True)
    ensure_baseline(current, device)
    if (SITE / "media").exists():
        shutil.rmtree(SITE / "media")

    stages = [collect_stage(stage) for stage in STAGES]
    for document in DOCUMENTS:
        if Path(document).exists():
            shutil.copyfile(document, SITE / document)

    data = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "stages": stages}
    SITE.mkdir(exist_ok=True)
    (SITE / "data.json").write_text(json.dumps(finite(data), allow_nan=False))
    for stage in stages:
        print(f"{stage['id']}: {len(stage['runs'])} runs")


if __name__ == "__main__":
    main()
