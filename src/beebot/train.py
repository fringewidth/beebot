"""Train the NCA on random layouts with a curriculum, then score it on the frozen val set.

    uv run train --name my-run --minutes 30 --lr 2e-3

Every field of Config is a command-line flag. Outputs go to runs/<name>/.
"""

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from beebot.ca import loss, rollout
from beebot.evaluate import evaluate
from beebot.layout import sample_batch
from beebot.model import NCA, load_weights
from beebot.render import render_examples

# Curriculum levels: (largest start distance, largest cheese, largest wall share).
# Training moves up a level once the hard success rate on training batches stays
# above LEVEL_UP_SUCCESS. Wall share is sampled uniformly from 0 up to the level's largest.
LEVELS = [
    (3, 1, 0.0),
    (5, 1, 0.0),
    (5, 3, 0.05),
    (8, 3, 0.1),
    (8, 5, 0.15),
    (12, 5, 0.2),
    (12, 9, 0.25),
    (16, 9, 0.3),
    (20, 9, 0.3),
]
LEVEL_UP_SUCCESS = 0.8
MIN_ITERATIONS_PER_LEVEL = 50


@dataclass
class Config:
    name: str = "run"
    seed: int = 0
    device: str = "mps"
    minutes: float = 30.0
    init: str = ""  # path to a model.pt to start from

    hidden: int = 96
    n_hidden_layers: int = 2
    stay_bias: float = 4.0
    privileged: bool = False  # control: also smell the walking distance to the cheese

    lr: float = 1e-3
    batch_size: int = 32
    train_sizes: str = "16,20,24,28,32,40"
    steps_per_cell: float = 3.0  # rollout length = steps_per_cell * grid size
    fire_prob: float = 0.75
    tau: float = 0.5
    use_checkpoint: bool = True

    w_surround: float = 0.5
    w_distance: float = 0.5
    w_crowding: float = 1.0
    w_gather: float = 0.5
    distance_decay: float = 1.0  # w_distance is multiplied by this at every level up

    log_every: int = 20


def parse_config():
    parser = argparse.ArgumentParser()
    for field in dataclasses.fields(Config):
        if field.type is bool:
            parser.add_argument(f"--{field.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=field.default)
        else:
            parser.add_argument(f"--{field.name}", type=field.type, default=field.default)
    return Config(**vars(parser.parse_args()))


def normalize_gradients(model):
    """Scale each parameter's gradient to unit norm, as in Growing NCA. Keeps long unrolls stable."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad /= p.grad.norm() + 1e-8


def train(cfg):
    run_dir = Path("runs") / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    sizes = [int(s) for s in cfg.train_sizes.split(",")]

    model = NCA(
        hidden=cfg.hidden,
        n_hidden_layers=cfg.n_hidden_layers,
        stay_bias=cfg.stay_bias,
        privileged=cfg.privileged,
    ).to(device)
    if cfg.init:
        load_weights(model, cfg.init, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    level = 0
    iterations_at_level = 0
    success_ema = 0.0
    w_distance = cfg.w_distance
    start = time.time()
    iteration = 0
    diverged = False
    log = open(run_dir / "log.jsonl", "w")

    while time.time() - start < cfg.minutes * 60:
        size = int(rng.choice(sizes))
        max_distance, max_cheese, max_walls = LEVELS[level]
        batch = sample_batch(rng, size, (1, max_cheese), (2, max_distance), cfg.batch_size, (0.0, max_walls)).to(device)
        n_steps = round(cfg.steps_per_cell * size)

        result = rollout(model, batch, n_steps, cfg.fire_prob, cfg.tau, use_checkpoint=cfg.use_checkpoint)
        value = loss(result, cfg.w_surround, w_distance, cfg.w_crowding, cfg.w_gather)
        if not torch.isfinite(value):
            diverged = True
            print(json.dumps({"iteration": iteration, "diverged": True}), flush=True)
            break
        optimizer.zero_grad()
        value.backward()
        normalize_gradients(model)
        optimizer.step()

        success = result.eaten.any(dim=0).float().mean().item()
        success_ema = 0.95 * success_ema + 0.05 * success
        iteration += 1
        iterations_at_level += 1

        leveled_up = False
        if level < len(LEVELS) - 1 and iterations_at_level >= MIN_ITERATIONS_PER_LEVEL and success_ema > LEVEL_UP_SUCCESS:
            level += 1
            iterations_at_level = 0
            success_ema = 0.0
            w_distance *= cfg.distance_decay
            leveled_up = True

        if iteration % cfg.log_every == 0 or leveled_up:
            entry = {
                "iteration": iteration,
                "minutes": round((time.time() - start) / 60, 2),
                "loss": round(value.item(), 4),
                "success": round(success, 3),
                "success_ema": round(success_ema, 3),
                "level": level,
                "size": size,
            }
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(json.dumps(entry), flush=True)

    log.close()
    torch.save(model.state_dict(), run_dir / "model.pt")

    model.eval()
    scores = evaluate(model, device, "val")
    summary = {
        "name": cfg.name,
        "iterations": iteration,
        "final_level": level,
        "diverged": diverged,
        "levels_total": len(LEVELS),
        "train_success_ema": success_ema,
        "val": scores,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    render_examples(model, device, run_dir / "videos")


def main():
    train(parse_config())


if __name__ == "__main__":
    main()
