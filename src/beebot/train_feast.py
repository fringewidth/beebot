"""Train the NCA on Feast from scratch, then score it on Feast's frozen val set.

    uv run train-feast --name my-run --minutes 200

Every field of FeastConfig is a command-line flag. Outputs go to runs/<name>/.
No curriculum: every batch samples the full range of grid sizes, cheese counts
and walls. The loss terms are described in feast.py. Saving, the STOP file and best-checkpoint keeping work as in train.py.
"""

import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from beebot import feast
from beebot.model import NCA, load_weights
from beebot.train import awake_seconds, keep_if_best, parse_config, save_checkpoint

TRAIN_CHEESE = (2, 4)


@dataclass
class FeastConfig:
    task: str = "feast"
    name: str = "run"
    seed: int = 0
    device: str = "mps"
    minutes: float = 200.0
    init: str = ""  # only for resuming this same run from its own model.pt
    save_every_minutes: float = 10.0
    eval_every_minutes: float = 30.0
    lr_schedule: str = "cosine"
    lr_floor: float = 0.1

    hidden: int = 96
    n_hidden_layers: int = 2
    stay_bias: float = 4.0

    lr: float = 3e-4
    # Val checks, checkpoints and the final model use an exponential moving average of the
    # weights. The raw weights swing between good and bad within 10 to 20 steps, so a raw
    # snapshot can land in a dip (v4-feast: val 4% while training batches showed 75%).
    ema_decay: float = 0.995
    max_grad_norm: float = 1.0
    batch_size: int = 32
    train_sizes: str = "16,20,24,28,32,40"
    steps_per_cell: float = 6.0
    fire_prob: float = 0.75
    tau: float = 0.5
    use_checkpoint: bool = True
    w_crowding: float = 1.0  # robots take up space
    w_fog: float = 0.1  # squares holding a thin film of robot (see feast.py)
    # If set, w_fog ramps linearly to this over the first fog_ramp share of training. At a
    # flat 1.0 from step 0 every move costs more than it earns and the swarm freezes
    # (v4-feast-fog1); at a flat 0.1 the swarm learns to eat by fog again (v4-feast-shaped).
    w_fog_final: float = 0.0
    fog_ramp: float = 0.5
    w_time: float = 0.5  # every step until the last cheese is eaten

    log_every: int = 20


def train(cfg):
    run_dir = Path("runs") / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)
    sizes = [int(s) for s in cfg.train_sizes.split(",")]

    model = NCA(hidden=cfg.hidden, n_hidden_layers=cfg.n_hidden_layers, stay_bias=cfg.stay_bias).to(device)
    if cfg.init:
        load_weights(model, cfg.init, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    averaged = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay))

    start = awake_seconds()
    last_save = start
    last_eval = start
    best_score = float("inf")
    iteration = 0
    success_ema = 0.0
    diverged = False
    log = open(run_dir / "log.jsonl", "a" if cfg.init else "w")
    workers = ProcessPoolExecutor(max_workers=1)

    def request_batch():
        size = int(rng.choice(sizes))
        seed = int(rng.integers(2**31))
        return workers.submit(feast.sample_batch_seeded, seed, size, TRAIN_CHEESE, cfg.batch_size), size

    if cfg.init:
        best_score = keep_if_best(run_dir, averaged.module, device, best_score, iteration, 0.0, evaluate=feast.evaluate)

    pending = request_batch()
    while awake_seconds() - start < cfg.minutes * 60 and not (run_dir / "STOP").exists():
        future, size = pending
        batch = future.result().to(device)
        pending = request_batch()
        n_steps = round(cfg.steps_per_cell * size)

        result = feast.rollout(model, batch, n_steps, cfg.fire_prob, cfg.tau, use_checkpoint=cfg.use_checkpoint)
        progress = min(1.0, (awake_seconds() - start) / (cfg.minutes * 60))
        w_fog = cfg.w_fog
        if cfg.w_fog_final:
            w_fog += (cfg.w_fog_final - cfg.w_fog) * min(1.0, progress / cfg.fog_ramp)
        value = feast.loss(result, cfg.w_crowding, w_fog, cfg.w_time)
        if not torch.isfinite(value):
            diverged = True
            print(json.dumps({"iteration": iteration, "diverged": True}), flush=True)
            break
        if cfg.lr_schedule == "cosine":
            scale = cfg.lr_floor + (1 - cfg.lr_floor) * 0.5 * (1 + np.cos(np.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr * scale
        optimizer.zero_grad()
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        averaged.update_parameters(model)

        success = (result.left[-1] == 0).float().mean().item()
        eaten_share = (1 - result.left[-1].float() / result.n_cheese.float()).mean().item()
        success_ema = 0.95 * success_ema + 0.05 * success
        iteration += 1

        if iteration % cfg.log_every == 0:
            entry = {
                "iteration": iteration,
                "minutes": round((awake_seconds() - start) / 60, 2),
                "loss": round(value.item(), 4),
                "success": round(success, 3),
                "eaten_share": round(eaten_share, 3),
                "fog": round(result.fog_share.mean().item(), 3),
                "success_ema": round(success_ema, 3),
                "w_fog": round(w_fog, 3),
                "size": size,
            }
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(json.dumps(entry), flush=True)

        if awake_seconds() - last_save > cfg.save_every_minutes * 60:
            save_checkpoint(run_dir, averaged.module, iteration, 0, (awake_seconds() - start) / 60)
            last_save = awake_seconds()

        if awake_seconds() - last_eval > cfg.eval_every_minutes * 60:
            best_score = keep_if_best(run_dir, averaged.module, device, best_score, iteration, (awake_seconds() - start) / 60, evaluate=feast.evaluate)
            last_eval = awake_seconds()

    workers.shutdown(cancel_futures=True)
    log.close()
    save_checkpoint(run_dir, averaged.module, iteration, 0, (awake_seconds() - start) / 60)

    # The run's result is its best val checkpoint, final averaged model included.
    best_score = keep_if_best(run_dir, averaged.module, device, best_score, iteration, (awake_seconds() - start) / 60, evaluate=feast.evaluate)
    model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=device))
    torch.save(model.state_dict(), run_dir / "model.pt")
    model.eval()
    scores = feast.evaluate(model, device, "val")
    summary = {
        "name": cfg.name,
        "iterations": iteration,
        "final_level": 0,
        "diverged": diverged,
        "train_success_ema": success_ema,
        "val": scores,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    feast.render_examples(model, device, run_dir / "videos")


def main():
    train(parse_config(FeastConfig))


if __name__ == "__main__":
    main()
