"""Rescore the fog run on Feast val (with the fog measurement it predates) and render demo clips, on the CPU.

    uv run python fog_hack/make.py
"""

import json
from pathlib import Path

import torch

from beebot import feast
from beebot.model import NCA

here = Path(__file__).parent
device = torch.device("cpu")
model = NCA(hidden=96, n_hidden_layers=2, stay_bias=4.0)
model.load_state_dict(torch.load(here / "model_best.pt", map_location=device))
model.eval()
picks = feast.render_demo(model, device, here / "videos")
(here / "picks.json").write_text(json.dumps(picks, indent=2))
scores = feast.evaluate(model, device, "val")
(here / "val.json").write_text(json.dumps(scores, indent=2))
print(json.dumps(scores, indent=2))
