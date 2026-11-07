"""Calibrate the closed-loop goal metrics: do-nothing, the goal-reading oracle (no learning),
replay, and trained checkpoints, all on the same 256 val windows.

    .venv/bin/python scripts/vla_goal_oracle.py runs/vla-r16-mean1 [runs/...]
"""

import json
import sys
from pathlib import Path

import torch

from beebot.vla import goal_eval
from beebot.vla.data import make_windows
from beebot.vla.model import Policy
from beebot.vla.train import load_or_generate

device = torch.device("mps")
first = json.loads((Path(sys.argv[1]) / "config.json").read_text())
store = load_or_generate(first["grid"], first["val_episodes"], 10_000 + first["seed"], device)
windows = goal_eval.eventful_windows(store, make_windows(store, first["grid"]), device, 256, torch.Generator().manual_seed(1))
for mode in ("zero", "oracle", "replay"):
    print(mode, json.dumps(goal_eval.summary(goal_eval.run(None, store, windows, device, mode=mode))), flush=True)
for run_dir in map(Path, sys.argv[1:]):
    cfg = json.loads((run_dir / "config.json").read_text())
    policy = Policy(cfg["grid"], width=cfg.get("width", 192), expert_width=cfg.get("expert_width", 128), layers=cfg.get("layers", 4), goal_tokens=cfg["goal_tokens"], goal_pool=cfg["goal_pool"]).to(device)
    policy.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=device), strict=False)
    policy.eval()
    for execute in (4, 1):
        scores = goal_eval.summary(goal_eval.run(policy, store, windows, device, execute=execute))
        print(run_dir.name, f"execute {execute}", json.dumps(scores), flush=True)
