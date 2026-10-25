"""Train the NCA on random layouts with a curriculum, then score it on the frozen val set.

    uv run train --name my-run --minutes 30 --lr 2e-3

Every field of Config is a command-line flag. Outputs go to runs/<name>/.

While training, the model and its curriculum level are saved every
save_every_minutes, so another run can start from them with --init and
--start_level. Creating runs/<name>/STOP ends the run early but still saves,
scores and renders it.
"""

import argparse
import dataclasses
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from beebot.ca import loss, rollout
from beebot.evaluate import evaluate
from beebot.layout import sample_batch_seeded
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

# Adaptive curriculum: instead of levels, every batch uses the full range of
# cheese sizes and walls, and one start distance. Distances are picked in
# proportion to how close their success rate is to 50%, so training spends its
# time at the frontier of what the swarm can do.
ADAPTIVE_DISTANCES = list(range(2, 21))
ADAPTIVE_CHEESE = (1, 9)
ADAPTIVE_WALLS = (0.0, 0.3)
ADAPTIVE_FLOOR = 0.05  # every distance keeps at least this much weight


def adaptive_weights(success_by_distance):
    weights = np.array([s * (1 - s) + ADAPTIVE_FLOOR for s in success_by_distance.values()])
    return weights / weights.sum()


@dataclass
class Config:
    name: str = "run"
    seed: int = 0
    device: str = "mps"
    minutes: float = 30.0
    init: str = ""  # path to a model.pt to start from
    start_level: int = 0
    save_every_minutes: float = 10.0
    eval_every_minutes: float = 30.0  # score on val and keep the best checkpoint
    lr_schedule: str = "constant"  # or "cosine": decays to lr_floor * lr over the run
    lr_floor: float = 0.1

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
    w_explore: float = 0.0  # reward covering the floor until first contact with the cheese
    distance_decay: float = 1.0  # w_distance is multiplied by this at every level up
    replay: float = 0.3  # share of batches drawn from an earlier level, so easy cases aren't forgotten
    adaptive: bool = False  # adaptive distance curriculum instead of levels (see ADAPTIVE_DISTANCES)

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


def save_checkpoint(run_dir, model, iteration, level, minutes):
    torch.save(model.state_dict(), run_dir / "model.pt")
    state = {"iteration": iteration, "level": level, "minutes": round(minutes, 2)}
    (run_dir / "checkpoint.json").write_text(json.dumps(state))


def keep_if_best(run_dir, model, device, best_score, iteration, minutes):
    """Score the model on val; if it's the best so far, save it as model_best.pt. Returns the best score."""
    # Evaluation reseeds the random generators for determinism; keep training's own randomness intact.
    cpu_state = torch.get_rng_state()
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    model.eval()
    with torch.no_grad():
        scores = evaluate(model, device, "val")
    model.train()
    torch.set_rng_state(cpu_state)
    if mps_state is not None:
        torch.mps.set_rng_state(mps_state)
    with open(run_dir / "val_history.jsonl", "a") as f:
        f.write(json.dumps({"iteration": iteration, "minutes": round(minutes, 2), "val": scores}) + "\n")
    if scores["score"] < best_score:
        torch.save(model.state_dict(), run_dir / "model_best.pt")
        return scores["score"]
    return best_score


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

    level = cfg.start_level
    iterations_at_level = 0
    success_ema = 0.0
    w_distance = cfg.w_distance * cfg.distance_decay**level
    start = time.time()
    last_save = start
    last_eval = start
    best_score = float("inf")
    iteration = 0
    diverged = False
    log = open(run_dir / "log.jsonl", "w")

    # Batches are generated in a worker process while the GPU trains on the previous one.
    workers = ProcessPoolExecutor(max_workers=1)

    success_by_distance = {d: 0.5 for d in ADAPTIVE_DISTANCES}

    def request_batch():
        size = int(rng.choice(sizes))
        if cfg.adaptive:
            # A start distance the grid can hold, chosen at the frontier of current success.
            usable = {d: v for d, v in success_by_distance.items() if d <= size // 2}
            distance = int(rng.choice(list(usable), p=adaptive_weights(usable)))
            seed = int(rng.integers(2**31))
            future = workers.submit(
                sample_batch_seeded, seed, size, ADAPTIVE_CHEESE, (distance, distance), cfg.batch_size, ADAPTIVE_WALLS
            )
            return future, size, False, distance
        replayed = level > 0 and rng.random() < cfg.replay
        batch_level = int(rng.integers(level)) if replayed else level
        max_distance, max_cheese, max_walls = LEVELS[batch_level]
        seed = int(rng.integers(2**31))
        future = workers.submit(
            sample_batch_seeded, seed, size, (1, max_cheese), (2, max_distance), cfg.batch_size, (0.0, max_walls)
        )
        return future, size, replayed, None

    pending = request_batch()
    while time.time() - start < cfg.minutes * 60 and not (run_dir / "STOP").exists():
        future, size, replayed, distance = pending
        batch = future.result().to(device)
        pending = request_batch()
        n_steps = round(cfg.steps_per_cell * size)

        result = rollout(model, batch, n_steps, cfg.fire_prob, cfg.tau, use_checkpoint=cfg.use_checkpoint)
        value = loss(result, cfg.w_surround, w_distance, cfg.w_crowding, cfg.w_gather, cfg.w_explore)
        if not torch.isfinite(value):
            diverged = True
            print(json.dumps({"iteration": iteration, "diverged": True}), flush=True)
            break
        if cfg.lr_schedule == "cosine":
            progress = min(1.0, (time.time() - start) / (cfg.minutes * 60))
            scale = cfg.lr_floor + (1 - cfg.lr_floor) * 0.5 * (1 + np.cos(np.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr * scale
        optimizer.zero_grad()
        value.backward()
        normalize_gradients(model)
        optimizer.step()

        success = result.eaten.any(dim=0).float().mean().item()
        iteration += 1
        if cfg.adaptive:
            success_by_distance[distance] = 0.9 * success_by_distance[distance] + 0.1 * success
        if not replayed:
            success_ema = 0.95 * success_ema + 0.05 * success
            iterations_at_level += 1

        leveled_up = False
        if not cfg.adaptive and level < len(LEVELS) - 1 and iterations_at_level >= MIN_ITERATIONS_PER_LEVEL and success_ema > LEVEL_UP_SUCCESS:
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
                "replayed": replayed,
            }
            if cfg.adaptive:
                entry["distance"] = distance
                entry["success_by_distance"] = {d: round(v, 2) for d, v in success_by_distance.items()}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(json.dumps(entry), flush=True)

        if time.time() - last_save > cfg.save_every_minutes * 60:
            save_checkpoint(run_dir, model, iteration, level, (time.time() - start) / 60)
            last_save = time.time()

        if time.time() - last_eval > cfg.eval_every_minutes * 60:
            best_score = keep_if_best(run_dir, model, device, best_score, iteration, (time.time() - start) / 60)
            last_eval = time.time()

    workers.shutdown(cancel_futures=True)
    log.close()
    save_checkpoint(run_dir, model, iteration, level, (time.time() - start) / 60)

    # The run's result is its best val checkpoint, final model included.
    best_score = keep_if_best(run_dir, model, device, best_score, iteration, (time.time() - start) / 60)
    model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=device))
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
