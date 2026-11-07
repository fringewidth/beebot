"""Inference-only sweep on a finished run: chunk steps executed per decision, guidance, and sampling
steps (Euler steps for flow, unmasking rounds for masked; 1 is one-shot decoding), scored closed
loop on the run's own test windows (the ones training reported).

    .venv/bin/python scripts/vla_inference_sweep.py runs/vla-d-g32-play-masked [--settings 4,1,10 1,1,1 ...]

Writes rows to <run>/inference_sweep.jsonl.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from beebot.vla import goal_eval
from beebot.vla.model import Policy
from beebot.vla.swarm import RelativePolicy
from beebot.vla.train import Config, load_data

DEFAULT_SETTINGS = ["4,1,10", "4,1,1", "4,1,4", "1,1,10", "2,1,10", "8,1,10", "4,2,10"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--play-cutoff", type=float, default=0.0, help="Unix time: ignore play runs recorded after it")
    parser.add_argument("--decode", default=None, help="expert or readout (default: the model's own; readout on insulate runs is the backbone's one-shot readout)")
    parser.add_argument("--settings", nargs="*", default=DEFAULT_SETTINGS, help="execute,guidance,sample_steps")
    args = parser.parse_args()
    run_dir = Path(args.run)
    stored = json.loads((run_dir / "config.json").read_text())
    names = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in stored.items() if k in names})
    cfg.play_cutoff = args.play_cutoff
    device = torch.device(args.device)
    if cfg.view:
        policy = RelativePolicy(cfg.view, width=cfg.width, layers=cfg.layers, crop=cfg.crop).to(device)
    else:
        policy = Policy(cfg.grid, width=cfg.width, expert_width=cfg.expert_width, layers=cfg.layers, goal_tokens=cfg.goal_tokens, goal_pool=cfg.goal_pool, objective=cfg.objective, world_model=cfg.world_model, insulate=cfg.insulate).to(device)
    policy.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=device))
    policy.eval()
    policy.decode = args.decode or policy.decode
    _, _, _, _, test_store, test_windows = load_data(cfg, device)
    out = run_dir / "inference_sweep.jsonl"
    for setting in args.settings:
        execute, guidance, steps = setting.split(",")
        with torch.no_grad():
            result = goal_eval.run(policy, test_store, test_windows, device, guidance=float(guidance), execute=int(execute), sample_steps=int(steps))
        row = {"decode": policy.decode, "execute": int(execute), "guidance": float(guidance), "sample_steps": int(steps), **goal_eval.summary(result)}
        print(json.dumps(row), flush=True)
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
