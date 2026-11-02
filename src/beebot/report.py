"""Collect every stage's runs into site/ for the project page.

    uv run report

Writes site/data.json, copies the GIFs the page shows into site/media/, and
copies the written logs (JOURNAL.md, HILLCLIMB.md) next to it.
Scores each current stage's hand-written baseline once and keeps it in its
baseline folder. Runs still training (no summary.json yet) show their best val
check so far, under the fog cap when one passes it, marked as in progress.
Everything runs on the CPU, so the page can be rebuilt while training holds the GPU.
"""

import dataclasses
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import torch

from beebot import chase, feast, rivals
from beebot.evaluate import evaluate
from beebot.model import NCA, StickyBaseline
from beebot.render import render_demo, render_examples
from beebot.train import LEVELS, Config, fog_ok
from beebot.train_feast import FeastConfig
from beebot.train_rivals import RivalsConfig, load_pair

# Stages whose task has its own module (sampling, evaluation, demo rendering).
TASKS = {"feast": feast, "chase": chase}
SITE = Path("site")
DOCUMENTS = ["JOURNAL.md", "HILLCLIMB.md"]
# Run settings that don't change what the model learns; left out of the "change" column.
NOT_A_CHANGE = {"name", "device", "log_every", "init", "use_checkpoint"}
# The page's demo is the stage's best run (best val score without the smelling
# control), on the hardest val layouts it solved. Stages whose physics the
# current code no longer reproduces ("demo": "fixed") keep the videos rendered
# when they ran.

# Newest first. "reference" is the run whose config counts as that stage's
# defaults; None means the current Config defaults.
STAGES = [
    {
        "id": "v6-shared",
        "title": "Rivals: shared",
        "task": "rivals",
        "game": "shared",
        "folder": "runs",
        "prefix": "v6-rivals-shared-",
        "max_fog": 0.45,
        "reference": None,
        "status": "current",
        "metric": "eaten",
        "summary": "Two species of swarm, A (blue) and B (orange), in one world, each with its own rule, mass and pheromones, sensing the other's. A cheese is eaten once their combined robots fill its ring, and each swarm owns the share of the ring it holds. Both swarms are paid for every cheese eaten, whoever eats it: the cooperative control, the same objective as Feast split across two species. Each swarm learns only from its own payoff and updates only its own rule. Scored on the share of the cheese value eaten within the time limit, each layout played twice with the nests swapped. Fog cap 0.45 for both swarms.",
    },
    {
        "id": "v6-zerosum",
        "title": "Rivals: zero-sum",
        "task": "rivals",
        "game": "zerosum",
        "folder": "runs",
        "prefix": "v6-rivals-zerosum-",
        "max_fog": 0.45,
        "reference": None,
        "status": "current",
        "metric": "eaten",
        "summary": "Two species of swarm, A (blue) and B (orange), in one world, each with its own rule, mass and pheromones, sensing the other's. A cheese is eaten once their combined robots fill its ring, and each swarm owns the share of the ring it holds. Each swarm is paid its share of every cheese minus the other's, so one swarm's gain is the other's loss: the competitive control. Each swarm learns only from its own payoff and updates only its own rule. Scored on the share of the cheese value eaten within the time limit, each layout played twice with the nests swapped. Fog cap 0.45 for both swarms.",
    },
    {
        "id": "v6-staghunt",
        "title": "Rivals: stag hunt",
        "task": "rivals",
        "game": "staghunt",
        "folder": "runs",
        "prefix": "v6-rivals-staghunt-",
        "max_fog": 0.45,
        "reference": None,
        "status": "current",
        "metric": "eaten",
        "summary": "Two species of swarm, A (blue) and B (orange), in one world, each with its own rule, mass and pheromones, sensing the other's. A cheese is eaten once their combined robots fill its ring, and each swarm owns the share of the ring it holds. Two small cheeses either swarm can eat alone (worth 1 each) and one big cheese, the stag (worth 4), whose ring needs more robots than either swarm has. Each swarm is paid its own share. Hunting the stag together pays more, but only if the other swarm comes too. Each swarm learns only from its own payoff and updates only its own rule. Scored on the share of the cheese value eaten within the time limit, each layout played twice with the nests swapped. Fog cap 0.45 for both swarms.",
    },
    {
        "id": "v5",
        "title": "Chase",
        "task": "chase",
        "folder": "runs",
        "prefix": "v5-",
        "baseline": "baseline-chase",
        "max_fog": 0.45,
        "reference": None,
        "status": "current",
        "summary": "One cheese on terrain that runs from the swarm: every 2 steps it moves one square toward the side with the least robot mass within 4 squares, blocked by walls and by squares holding half a robot or more. Eaten once its ring is full. A scripted swarm that can see the cheese catches it in 56 to 81% of layouts, so the task is fair; the blind swarm has to find it by touch and pin it. Training speeds the cheese up level by level (still, then every 16, 8, 4, 2 steps); val always uses full speed. The second run (slower curriculum, an earlier fog limit, credit for ground covered by solid robots) caught the full-speed cheese in up to 16/22/8% of val layouts with fog, but under the 0.45 fog cap only 5/3/6% (baseline 0/2/0%), and none at all once the swarm was fully compact. Not solved; the stage stays open.",
    },
    {
        "id": "v4",
        "title": "Feast",
        "task": "feast",
        "folder": "runs_v4",
        "prefix": "v4-",
        "baseline": "baseline",
        "max_fog": 0.45,
        "reference": None,
        "status": "closed",
        "summary": "Two to four cheeses on terrain; each vanishes once its ring is full, and the swarm must eat them all. No smell, pheromones only. Trained from scratch, no curriculum. The loss is the task (unfilled rings, plus crowding because robots take up space), a cost per step until everything is eaten, credit for floor that solid robots have covered, and a limit on fog (see below) that starts wide open and tightens once the swarm can eat. A run only counts at val fog of 0.45 or less on every split, the level of the accepted terrain swarm. Closed as a weak proof of concept on the 352-minute checkpoint of v4-feast-explore2: test 0.846, eating everything in 48% of unseen sizes, 6% of bigger worlds and 27% of damaged swarms, at fog 0.36 to 0.39, against 0% for the hand-written baseline. Bigger worlds are the weak spot: the swarm cannot search them without spreading thin.",
        "aside": {
            "folder": "fog_hack",
            "title": "The fog hack",
            "text": "With the loss being the task alone, the swarm found a shortcut. Robot mass in the simulation is continuous, so a square can hold a few hundredths of a robot. The swarm dissolves into a thin film over the whole world, and the film thickens wherever it touches a cheese's ring until the ring is full. It never searches, because it is already everywhere. No swarm of whole robots could do this: the objective said 'fill every ring' and never said the swarm has to stay made of robots. It also outscores every fog-free swarm trained so far. Run v4-feast-ema at 60 minutes, on the same held-out layouts; fog is the share of the swarm sitting on squares with less than half a robot.",
        },
    },
    {
        "id": "v3b",
        "title": "Terrain, 6x time",
        "folder": "runs_v3b",
        "reference": None,
        "status": "closed",
        "summary": "v3a with the time limit doubled to 6 steps per grid square, trained from scratch. Closed as a proof of concept: test 0.695 (52/38/44% of layouts) against 0% for the hand-written baseline; blocking pheromones drops it to 0%. The main weakness is recruiting the whole swarm and holding the formation, not time: with 4x the time limit, val success rises only from 46/44% to 58/62%. Reuniting separate nests is also weak.",
    },
    {
        "id": "v3a",
        "title": "Terrain",
        "folder": "runs_v3a",
        "reference": "v3r1-blind",
        "steps_per_cell": 3,
        "status": "closed",
        "summary": "No smell, pheromones, the whole swarm must gather, and random cave-like walls covering up to 30% of the world. Time limit 3 steps per grid square. Best blind val 0.724 (44/39/42%); the scent-following control reached 96/95/89%.",
    },
    {
        "id": "v2",
        "title": "No smell",
        "folder": "runs_v2",
        "reference": "v2r1-lr1e3",
        "demo": "fixed",
        "status": "closed",
        "summary": "Touch only. Robots lay pheromones, only robots write them, and a scrap of a robot moves in proportion to its mass. Late in the stage, success also required the whole swarm to gather.",
    },
    {
        "id": "v1",
        "title": "Smell",
        "folder": "runs_v1",
        "reference": "sanity",
        "demo": "fixed",
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


def rivals_probes(stage):
    """Strangers and blackout probes for a Rivals game (see rivals_probes.py), on the
    fog-capped pairs and on the peak (fog allowed) pairs."""
    if stage.get("task") != "rivals":
        return None
    prefix = Path(stage["folder"]) / f"v6-rivals-{stage['game']}-probes"
    found = {
        "capped": read_json(Path(f"{prefix}.json")),
        "peak": read_json(Path(f"{prefix}-peak.json")),
    }
    return found if any(found.values()) else None


def baseline_dir(stage):
    return Path(stage["folder"]) / stage.get("baseline", "baseline")


def ensure_baseline(stage, device):
    run_dir = baseline_dir(stage)
    if (run_dir / "summary.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline = StickyBaseline()
    if stage.get("task") in TASKS:
        task = TASKS[stage["task"]]
        summary = {"name": "baseline", "val": task.evaluate(baseline, device, "val")}
        task.render_examples(baseline, device, run_dir / "videos")
    else:
        summary = {"name": "baseline", "val": evaluate(baseline, device, "val")}
        render_examples(baseline, device, run_dir / "videos")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def config_changes(config, defaults):
    return {k: v for k, v in config.items() if k not in NOT_A_CHANGE and k in defaults and defaults[k] != v}


def curve(run_dir):
    points = []
    for line in (run_dir / "log.jsonl").read_text().splitlines():
        entry = json.loads(line)
        smoothed = entry.get("success_ema", entry.get("eaten_ema"))
        if smoothed is not None:
            points.append([entry["minutes"], entry.get("level", 0), smoothed])
    return points


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def copy_videos(source, stage_id, name):
    videos = []
    for gif in sorted(source.glob("*.gif")):
        target = SITE / "media" / stage_id / name / gif.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(gif, target)
        videos.append(f"media/{stage_id}/{name}/{gif.name}")
    return videos


def demo(stage, run_dir, device, weights):
    """Videos and captions for the stage's best run, rendered once and kept in its demo/ folder."""
    if stage.get("demo") == "fixed":
        return {"run": run_dir.name, "videos": copy_videos(run_dir / "videos", stage["id"], run_dir.name), "picks": None}
    picks_path = run_dir / "demo" / "picks.json"
    if not picks_path.exists():
        config = read_json(run_dir / "config.json")
        if stage.get("task") == "rivals":
            a, b = load_pair(weights.parent, weights.name, config, device)
            picks = rivals.render_demo(a, b, device, run_dir / "demo", stage["game"])
            picks_path.write_text(json.dumps(picks, indent=2))
            return {"run": run_dir.name, "videos": copy_videos(run_dir / "demo", stage["id"], run_dir.name), "picks": picks}
        model = NCA(hidden=config["hidden"], n_hidden_layers=config["n_hidden_layers"], privileged=config.get("privileged", False))
        model.load_state_dict(torch.load(weights, map_location=device))
        model.to(device).eval()
        if stage.get("task") in TASKS:
            picks = TASKS[stage["task"]].render_demo(model, device, run_dir / "demo")
        else:
            picks = render_demo(model, device, run_dir / "demo", stage.get("steps_per_cell", 6))
        picks_path.write_text(json.dumps(picks, indent=2))
    return {"run": run_dir.name, "videos": copy_videos(run_dir / "demo", stage["id"], run_dir.name), "picks": read_json(picks_path)}


def collect_aside(stage):
    """A run kept outside the stage's ranking for what it shows, with its clips and val scores."""
    aside = stage.get("aside")
    if not aside:
        return None
    folder = Path(aside["folder"])
    return {
        "title": aside["title"],
        "text": aside["text"],
        "videos": copy_videos(folder / "videos", stage["id"], folder.name),
        "picks": read_json(folder / "picks.json"),
        "val": read_json(folder / "val.json"),
    }


def provisional_summary(run_dir, cap):
    """For a run still training: its best val check so far, under the stage's fog cap if one passes it."""
    history = [json.loads(line) for line in (run_dir / "val_history.jsonl").read_text().splitlines()]
    capped = [h for h in history if fog_ok(h["val"], cap)]
    best = min(capped or history, key=lambda h: h["val"]["score"])
    weights = run_dir / f"model_{best['iteration']}.pt"
    if not weights.exists():
        weights = run_dir / "model.pt"
    if (run_dir / f"val_{best['iteration']}_a.pt").exists():
        weights = run_dir / f"val_{best['iteration']}"  # a Rivals pair: the prefix of its two files
    checkpoint = read_json(run_dir / "checkpoint.json")
    return {
        "iterations": checkpoint["iteration"],
        "final_level": checkpoint.get("level", 0),
        "val": best["val"],
        "in_progress": True,
        "under_fog_cap": bool(capped),
        "best_at_minutes": best["minutes"],
        "weights": weights,
    }


def collect_stage(stage, device):
    folder = Path(stage["folder"])
    if stage["reference"]:
        defaults = read_json(folder / stage["reference"] / "config.json")
    elif stage.get("task") == "rivals":
        defaults = dataclasses.asdict(RivalsConfig())
    elif stage.get("task") in TASKS:
        defaults = dataclasses.asdict(FeastConfig())
    else:
        defaults = dataclasses.asdict(Config())

    runs = []
    for run_dir in sorted(folder.iterdir()) if folder.exists() else []:
        if run_dir.name.startswith("baseline") or not run_dir.name.startswith(stage.get("prefix", "")):
            continue
        config = read_json(run_dir / "config.json")
        if config is None or (stage.get("task") in TASKS and config.get("task", "feast") != stage["task"]):
            continue
        if stage.get("task") == "rivals" and config.get("game") != stage["game"]:
            continue
        summary = read_json(run_dir / "summary.json")
        if summary is None:
            # Only a current stage shows runs still training; a closed stage's unfinished runs were stopped.
            if stage["status"] != "current" or not (run_dir / "val_history.jsonl").exists():
                continue
            summary = provisional_summary(run_dir, stage.get("max_fog"))
        runs.append(
            {
                "name": run_dir.name,
                "changes": config_changes(config, defaults),
                "minutes": config["minutes"],
                "iterations": summary["iterations"],
                "final_level": summary.get("final_level", 0),
                "diverged": summary.get("diverged", False),
                "val": summary["val"],
                "val_ring_only": summary.get("val_ring_only"),
                "final_test": read_json(run_dir / "final_test.json"),
                "curve": curve(run_dir),
                "probes": read_json(run_dir / "probes.json"),
                "in_progress": summary.get("in_progress", False),
                "under_fog_cap": summary.get("under_fog_cap"),
                "best_at_minutes": summary.get("best_at_minutes"),
                "run_dir": run_dir,
                "weights": summary.get("weights", run_dir / ("best" if stage.get("task") == "rivals" else "model.pt")),
            }
        )

    runs.sort(key=lambda r: r["val"]["score"])
    # A privileged control can outscore every real run, so the demo is the best run without it.
    best = next((r for r in runs if not r["changes"].get("privileged")), None)
    stage_demo = demo(stage, best["run_dir"], device, best["weights"]) if best else None
    for run in runs:
        run.pop("run_dir")
        run.pop("weights")

    baseline = read_json(baseline_dir(stage) / "summary.json")
    return {
        "id": stage["id"],
        "title": stage["title"],
        "status": stage["status"],
        "summary": stage["summary"],
        "levels": 0 if stage.get("task") in TASKS or stage.get("task") == "rivals" else len(LEVELS),
        "metric": stage.get("metric", "success"),
        "baseline": {
            "val": baseline["val"] if baseline else None,
            "probes": read_json(baseline_dir(stage) / "probes.json"),
        },
        "demo": stage_demo,
        "rivals_probes": rivals_probes(stage),
        "aside": collect_aside(stage),
        "runs": runs,
    }


def main():
    device = torch.device("cpu")
    for stage in STAGES:
        if stage["status"] == "current" and stage.get("task") != "rivals":
            Path(stage["folder"]).mkdir(exist_ok=True)
            ensure_baseline(stage, device)
    if (SITE / "media").exists():
        shutil.rmtree(SITE / "media")

    stages = [collect_stage(stage, device) for stage in STAGES]
    # A stage that hasn't started training yet gets no tab.
    stages = [stage for stage in stages if stage["runs"] or stage["status"] != "current"]
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
