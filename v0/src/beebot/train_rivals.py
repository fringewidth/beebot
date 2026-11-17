"""Train a pair of rival swarms (A and B) from scratch on one Rivals game, then score the pair on val.

    uv run train-rivals --game staghunt --name my-pair --seed 0 --minutes 300

Both rules start fresh. Each swarm's gradient comes only from its own payoff and updates
only its own rule. Fog is a constraint per swarm, as in Feast: the limit starts open,
waits until the pair eats fog_gate of the value on average, then tightens to fog_target
over fog_tighten_minutes, and each swarm has its own multiplier. Checkpoints, resuming,
the STOP file and rollback work as in train_feast.py. Two pairs trained apart with
different seeds can be paired up afterwards as strangers (see rivals_probes.py).
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

from beebot import rivals
from beebot.model import RivalNCA
from beebot.train import awake_seconds, fog_ok, parse_config


@dataclass
class RivalsConfig:
    game: str = "shared"
    name: str = "pair"
    seed: int = 0
    device: str = "mps"
    minutes: float = 300.0
    resume: bool = False
    save_every_minutes: float = 5.0
    eval_every_minutes: float = 30.0
    lr_floor: float = 0.1

    hidden: int = 96
    n_hidden_layers: int = 2
    stay_bias: float = 4.0

    lr: float = 3e-4
    ema_decay: float = 0.995
    max_grad_norm: float = 1.0
    batch_size: int = 32
    train_sizes: str = "16,20,24,28,32,40"
    steps_per_cell: float = 6.0
    fire_prob: float = 0.75
    tau: float = 0.5
    use_checkpoint: bool = True
    w_crowding: float = 1.0

    fog_target: float = 0.35
    fog_target_start: float = 1.0
    fog_gate: float = 0.3
    fog_tighten_minutes: float = 180.0
    dual_lr: float = 0.01
    max_fog: float = 0.45

    rollback: bool = True
    rollback_from: float = 0.3
    rollback_share: float = 0.33
    # The best average fades by rollback_memory per iteration, so a slow decline (a tightening
    # fog limit costs eating) doesn't count as a collapse; a sudden drop still does. With 1.0
    # the best never fades: a run held below a third of its fog-free best by the fog limit
    # then rolls back every few dozen iterations and stops learning (v6-rivals-shared-s0,
    # 13 rollbacks from 258 to 305 minutes).
    rollback_memory: float = 0.999

    log_every: int = 20


def save_pair(run_dir, prefix, a, b):
    torch.save(a.state_dict(), run_dir / f"{prefix}_a.pt")
    torch.save(b.state_dict(), run_dir / f"{prefix}_b.pt")


def load_pair(run_dir, prefix, cfg, device):
    pair = []
    for side in ("a", "b"):
        model = RivalNCA(hidden=cfg["hidden"], n_hidden_layers=cfg["n_hidden_layers"], stay_bias=cfg["stay_bias"]).to(device)
        model.load_state_dict(torch.load(run_dir / f"{prefix}_{side}.pt", map_location=device))
        model.eval()
        pair.append(model)
    return pair


def keep_if_best(run_dir, a, b, device, game, best_score, iteration, minutes, max_fog):
    """Score the averaged pair on val; keep it as best_a.pt/best_b.pt if it's the best under the fog cap."""
    cpu_state = torch.get_rng_state()
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    a.eval()
    b.eval()
    with torch.no_grad():
        scores = rivals.evaluate(a, b, device, game, "val")
    a.train()
    b.train()
    torch.set_rng_state(cpu_state)
    if mps_state is not None:
        torch.mps.set_rng_state(mps_state)
    with open(run_dir / "val_history.jsonl", "a") as f:
        f.write(json.dumps({"iteration": iteration, "minutes": round(minutes, 2), "val": scores}) + "\n")
    if scores["score"] < best_score and fog_ok(scores, max_fog):
        save_pair(run_dir, "best", a, b)
        return scores["score"]
    return best_score


def train(cfg):
    run_dir = Path("runs") / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    resumed = {"minutes": 0.0, "iteration": 0, "w_fog": [0.0, 0.0], "gate_minutes": None}
    if cfg.resume:
        resumed = json.loads((run_dir / "checkpoint.json").read_text())
    config_name = f"config_resume_{resumed['iteration']}.json" if cfg.resume else "config.json"
    (run_dir / config_name).write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    torch.manual_seed(cfg.seed + resumed["iteration"])
    rng = np.random.default_rng([cfg.seed, resumed["iteration"]])
    device = torch.device(cfg.device)
    sizes = [int(s) for s in cfg.train_sizes.split(",")]

    models = [RivalNCA(hidden=cfg.hidden, n_hidden_layers=cfg.n_hidden_layers, stay_bias=cfg.stay_bias).to(device) for _ in range(2)]
    if cfg.resume:
        for side, model in zip(("a", "b"), models):
            start_weights = run_dir / f"resume_{resumed['iteration']}_{side}.pt"
            if not start_weights.exists():
                shutil.copy(run_dir / f"model_{side}.pt", start_weights)
            model.load_state_dict(torch.load(start_weights, map_location=device))
    optimizers = [torch.optim.Adam(m.parameters(), lr=cfg.lr) for m in models]
    averaged = [AveragedModel(m, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay)) for m in models]

    start = awake_seconds() - resumed["minutes"] * 60
    last_save = awake_seconds()
    last_eval = awake_seconds()
    best_score = float("inf")
    if cfg.resume and (run_dir / "val_history.jsonl").exists():
        history = [json.loads(line)["val"] for line in open(run_dir / "val_history.jsonl")]
        best_score = min([v["score"] for v in history if fog_ok(v, cfg.max_fog or None)], default=float("inf"))
    iteration = resumed["iteration"]
    multipliers = list(resumed["w_fog"])
    gate_minutes = resumed["gate_minutes"]
    eaten_ema = 0.0
    best_eaten_ema = 0.0
    rollbacks = 0
    target = cfg.fog_target_start
    diverged = False
    log = open(run_dir / "log.jsonl", "a" if cfg.resume else "w")
    workers = ProcessPoolExecutor(max_workers=1)

    def request_batch():
        size = int(rng.choice(sizes))
        seed = int(rng.integers(2**31))
        return workers.submit(rivals.sample_training_batch, seed, size, cfg.game, cfg.batch_size), size

    def checkpoint_now():
        save_pair(run_dir, "model", averaged[0].module, averaged[1].module)
        state = {"iteration": iteration, "minutes": round((awake_seconds() - start) / 60, 2), "w_fog": multipliers, "gate_minutes": gate_minutes}
        (run_dir / "checkpoint.json").write_text(json.dumps(state))

    pending = request_batch()
    while awake_seconds() - start < cfg.minutes * 60 and not (run_dir / "STOP").exists():
        future, size = pending
        batch = future.result().to(device)
        pending = request_batch()
        n_steps = round(cfg.steps_per_cell * size)
        minutes = (awake_seconds() - start) / 60
        progress = min(1.0, minutes / cfg.minutes)

        # Every other batch swaps the two nests, so neither swarm learns to play from one side only.
        if iteration % 2:
            batch = batch.swapped()
        result = rivals.rollout(models[0], models[1], batch, n_steps, cfg.fire_prob, cfg.tau, use_checkpoint=cfg.use_checkpoint, pushable=cfg.game in rivals.PUSH_GAMES)
        loss_a, loss_b = rivals.losses(result, cfg.game, cfg.w_crowding, multipliers[0], multipliers[1])
        if not (torch.isfinite(loss_a) and torch.isfinite(loss_b)):
            diverged = True
            print(json.dumps({"iteration": iteration, "diverged": True}), flush=True)
            break
        scale = cfg.lr_floor + (1 - cfg.lr_floor) * 0.5 * (1 + np.cos(np.pi * progress))
        grads = [
            torch.autograd.grad(loss_a, list(models[0].parameters()), retain_graph=True),
            torch.autograd.grad(loss_b, list(models[1].parameters())),
        ]
        for model, optimizer, grad, avg in zip(models, optimizers, grads, averaged):
            for p, g in zip(model.parameters(), grad):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr * scale
            optimizer.step()
            avg.update_parameters(model)

        eaten = result.eaten_value[-1].mean().item()
        eaten_ema = 0.95 * eaten_ema + 0.05 * eaten
        fogs = [result.fog_share_a.mean().item(), result.fog_share_b.mean().item()]
        if gate_minutes is None and eaten_ema >= cfg.fog_gate:
            gate_minutes = minutes
        tightened = 0.0 if gate_minutes is None else (minutes - gate_minutes) / cfg.fog_tighten_minutes
        target = cfg.fog_target + (cfg.fog_target_start - cfg.fog_target) * max(0.0, 1 - tightened)
        multipliers = [max(0.0, w + cfg.dual_lr * (fog - target)) for w, fog in zip(multipliers, fogs)]

        best_eaten_ema = max(best_eaten_ema * cfg.rollback_memory, eaten_ema)
        if cfg.rollback and best_eaten_ema >= cfg.rollback_from and eaten_ema < cfg.rollback_share * best_eaten_ema:
            for i, model in enumerate(models):
                model.load_state_dict(averaged[i].module.state_dict())
                optimizers[i] = torch.optim.Adam(model.parameters(), lr=cfg.lr * scale)
            rollbacks += 1
            print(json.dumps({"iteration": iteration, "rollback": rollbacks, "eaten_ema": round(eaten_ema, 3), "best": round(best_eaten_ema, 3)}), flush=True)
            eaten_ema = best_eaten_ema
        iteration += 1

        if iteration % cfg.log_every == 0:
            entry = {
                "iteration": iteration,
                "minutes": round(minutes, 2),
                "loss_a": round(loss_a.item(), 4),
                "loss_b": round(loss_b.item(), 4),
                "eaten": round(eaten, 3),
                "payoff_a": round(result.payoff_a.mean().item(), 3),
                "payoff_b": round(result.payoff_b.mean().item(), 3),
                "fog_a": round(fogs[0], 3),
                "fog_b": round(fogs[1], 3),
                "eaten_ema": round(eaten_ema, 3),
                "w_fog_a": round(multipliers[0], 3),
                "w_fog_b": round(multipliers[1], 3),
                "fog_target": round(target, 3),
                "rollbacks": rollbacks,
                "size": size,
            }
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(json.dumps(entry), flush=True)

        if awake_seconds() - last_save > cfg.save_every_minutes * 60:
            checkpoint_now()
            last_save = awake_seconds()

        if awake_seconds() - last_eval > cfg.eval_every_minutes * 60:
            save_pair(run_dir, f"val_{iteration}", averaged[0].module, averaged[1].module)
            best_score = keep_if_best(run_dir, averaged[0].module, averaged[1].module, device, cfg.game, best_score, iteration, (awake_seconds() - start) / 60, cfg.max_fog or None)
            last_eval = awake_seconds()

    workers.shutdown(cancel_futures=True)
    log.close()
    checkpoint_now()
    best_score = keep_if_best(run_dir, averaged[0].module, averaged[1].module, device, cfg.game, best_score, iteration, (awake_seconds() - start) / 60, cfg.max_fog or None)
    if not (run_dir / "best_a.pt").exists():
        # No val checkpoint met the fog cap; the summary's fog numbers say so.
        save_pair(run_dir, "best", averaged[0].module, averaged[1].module)
    a, b = load_pair(run_dir, "best", dataclasses.asdict(cfg), device)
    summary = {
        "name": cfg.name,
        "game": cfg.game,
        "iterations": iteration,
        "diverged": diverged,
        "rollbacks": rollbacks,
        "val": rivals.evaluate(a, b, device, cfg.game, "val"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    train(parse_config(RivalsConfig))


if __name__ == "__main__":
    main()
