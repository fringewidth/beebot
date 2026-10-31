"""Train the NCA on Feast (or Chase, with --task chase) from scratch, then score it on that task's frozen val set.

    uv run train-feast --name my-run --minutes 200
    uv run train-feast --task chase --name my-run --minutes 200

Every field of FeastConfig is a command-line flag. Outputs go to runs/<name>/.
No curriculum: every batch samples the full range of grid sizes, cheese counts
and walls. The loss terms are described in feast.py. Saving, the STOP file and best-checkpoint keeping work as in train.py.
"""

import dataclasses
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from beebot import chase, feast
from beebot.model import NCA, load_weights
from beebot.train import awake_seconds, keep_if_best, parse_config, save_checkpoint

# Each task module provides sample_training_batch, rollout, evaluate and render_examples.
# The loss is Feast's for both.
TASKS = {"feast": feast, "chase": chase}


@dataclass
class FeastConfig:
    task: str = "feast"
    name: str = "run"
    seed: int = 0
    device: str = "mps"
    minutes: float = 200.0
    init: str = ""  # only for resuming this same run from its own model.pt
    # Pick this run up where its last checkpoint left off: weights from model.pt, and the
    # clock and iteration from checkpoint.json, so the lr and fog schedules continue. The
    # fog multiplier comes from dual.json; the optimizer state restarts.
    resume: bool = False
    save_every_minutes: float = 5.0
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
    # If set, fog is a constraint instead: w_fog is a Lagrange multiplier that grows while the
    # batch's fog share is above fog_target and shrinks while it is below (dual ascent), so the
    # swarm can't stay frozen once fog is under control. Overrides the ramp.
    fog_target: float = 0.0
    dual_lr: float = 0.02
    # If set, the fog target starts here and tightens linearly to fog_target over the first
    # fog_ramp share of training. Fog is how an untrained swarm explores (spread-out mass
    # reaches the rings, so the loss has a gradient); a tight limit from step 0 freezes it.
    fog_target_start: float = 0.0
    w_time: float = 0.5  # every step until the last cheese is eaten

    log_every: int = 20


def train(cfg):
    run_dir = Path("runs") / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    resumed = {"minutes": 0.0, "iteration": 0}
    if cfg.resume:
        resumed = json.loads((run_dir / "checkpoint.json").read_text())
        if (run_dir / "dual.json").exists():
            cfg.w_fog = json.loads((run_dir / "dual.json").read_text())["w_fog"]
        # Keep a copy of the starting weights: model.pt is overwritten at the next save.
        start_weights = run_dir / f"model_resume_{resumed['iteration']}.pt"
        if not start_weights.exists():
            shutil.copy(run_dir / "model.pt", start_weights)
        cfg.init = str(start_weights)
    config_name = f"config_resume_{resumed['iteration']}.json" if cfg.resume else "config.json"
    (run_dir / config_name).write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    # A resumed run draws fresh batches rather than replaying the ones it started with.
    torch.manual_seed(cfg.seed + resumed["iteration"])
    rng = np.random.default_rng([cfg.seed, resumed["iteration"]])
    device = torch.device(cfg.device)
    sizes = [int(s) for s in cfg.train_sizes.split(",")]
    env = TASKS[cfg.task]

    model = NCA(hidden=cfg.hidden, n_hidden_layers=cfg.n_hidden_layers, stay_bias=cfg.stay_bias).to(device)
    if cfg.init:
        load_weights(model, cfg.init, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    averaged = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay))

    start = awake_seconds() - resumed["minutes"] * 60
    last_save = awake_seconds()
    last_eval = awake_seconds()
    best_score = float("inf")
    if cfg.resume and (run_dir / "val_history.jsonl").exists():
        best_score = min(json.loads(line)["val"]["score"] for line in open(run_dir / "val_history.jsonl"))
    iteration = resumed["iteration"]
    success_ema = 0.0
    multiplier = cfg.w_fog
    target = cfg.fog_target
    diverged = False
    log = open(run_dir / "log.jsonl", "a" if cfg.init else "w")
    workers = ProcessPoolExecutor(max_workers=1)

    def request_batch():
        size = int(rng.choice(sizes))
        seed = int(rng.integers(2**31))
        return workers.submit(env.sample_training_batch, seed, size, cfg.batch_size), size

    if cfg.init and not cfg.resume:
        best_score = keep_if_best(run_dir, averaged.module, device, best_score, iteration, 0.0, evaluate=env.evaluate)

    pending = request_batch()
    while awake_seconds() - start < cfg.minutes * 60 and not (run_dir / "STOP").exists():
        future, size = pending
        batch = future.result().to(device)
        pending = request_batch()
        n_steps = round(cfg.steps_per_cell * size)

        result = env.rollout(model, batch, n_steps, cfg.fire_prob, cfg.tau, use_checkpoint=cfg.use_checkpoint)
        progress = min(1.0, (awake_seconds() - start) / (cfg.minutes * 60))
        w_fog = cfg.w_fog
        if cfg.w_fog_final:
            w_fog += (cfg.w_fog_final - cfg.w_fog) * min(1.0, progress / cfg.fog_ramp)
        if cfg.fog_target:
            w_fog = multiplier
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
        if cfg.fog_target:
            target = cfg.fog_target
            if cfg.fog_target_start:
                target += (cfg.fog_target_start - cfg.fog_target) * max(0.0, 1 - progress / cfg.fog_ramp)
            violation = result.fog_share.mean().item() - target
            multiplier = max(0.0, multiplier + cfg.dual_lr * violation)

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
                "fog_target": round(target, 3) if cfg.fog_target else None,
                "size": size,
            }
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(json.dumps(entry), flush=True)

        if awake_seconds() - last_save > cfg.save_every_minutes * 60:
            save_checkpoint(run_dir, averaged.module, iteration, 0, (awake_seconds() - start) / 60)
            (run_dir / "dual.json").write_text(json.dumps({"w_fog": multiplier}))
            last_save = awake_seconds()

        if awake_seconds() - last_eval > cfg.eval_every_minutes * 60:
            best_score = keep_if_best(run_dir, averaged.module, device, best_score, iteration, (awake_seconds() - start) / 60, evaluate=env.evaluate)
            last_eval = awake_seconds()

    workers.shutdown(cancel_futures=True)
    log.close()
    save_checkpoint(run_dir, averaged.module, iteration, 0, (awake_seconds() - start) / 60)

    # The run's result is its best val checkpoint, final averaged model included.
    best_score = keep_if_best(run_dir, averaged.module, device, best_score, iteration, (awake_seconds() - start) / 60, evaluate=env.evaluate)
    model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=device))
    torch.save(model.state_dict(), run_dir / "model.pt")
    model.eval()
    scores = env.evaluate(model, device, "val")
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
    env.render_examples(model, device, run_dir / "videos")


def main():
    train(parse_config(FeastConfig))


if __name__ == "__main__":
    main()
