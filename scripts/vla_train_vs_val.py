"""Closed-loop scores on training windows vs val windows for a finished scripted run, to tell
overfitting (train well above val) from underfitting or ambiguity (train about equal to val).

    .venv/bin/python scripts/vla_train_vs_val.py runs/vla-r16-mean1
"""

import json
import sys
from pathlib import Path

import torch

from beebot.vla import goal_eval
from beebot.vla.data import make_windows
from beebot.vla.model import Policy
from beebot.vla.train import load_or_generate

run_dir = Path(sys.argv[1])
cfg = json.loads((run_dir / "config.json").read_text())
device = torch.device("mps")
policy = Policy(cfg["grid"], width=cfg.get("width", 192), expert_width=cfg.get("expert_width", 128), layers=cfg.get("layers", 4), goal_tokens=cfg["goal_tokens"], goal_pool=cfg["goal_pool"]).to(device)
policy.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=device), strict=False)
policy.eval()
for name, episodes, seed in (("train", cfg["episodes"], cfg["seed"]), ("val", cfg["val_episodes"], 10_000 + cfg["seed"])):
    store = load_or_generate(cfg["grid"], episodes, seed, device, cfg.get("data_noise", 0.0) if name == "train" else 0.0)
    windows = goal_eval.eventful_windows(store, make_windows(store, cfg["grid"]), device, 256, torch.Generator().manual_seed(1))
    print(name, json.dumps(goal_eval.summary(goal_eval.run(policy, store, windows, device))), flush=True)
