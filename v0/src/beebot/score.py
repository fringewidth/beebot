"""Score runs' latest checkpoints on the val set only, for choosing between runs mid-training.

    uv run score runs/<name> [runs/<name> ...]

Never touches the test set.
"""

import json
import sys
from pathlib import Path

import torch

from beebot.evaluate import evaluate
from beebot.model import NCA


def main():
    device = torch.device("mps")
    for arg in sys.argv[1:]:
        run_dir = Path(arg)
        config = json.loads((run_dir / "config.json").read_text())
        checkpoint = json.loads((run_dir / "checkpoint.json").read_text())
        model = NCA(
            hidden=config["hidden"],
            n_hidden_layers=config["n_hidden_layers"],
            privileged=config.get("privileged", False),
        ).to(device)
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
        model.eval()
        val = evaluate(model, device, "val")
        result = {"checkpoint": checkpoint, "val": val}
        with open(run_dir / "checkpoint_scores.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
        splits = " ".join(f"{k} {val[k]['success']:.2f}" for k in ("interp", "extrap", "damage"))
        print(f"{run_dir.name}: minute {checkpoint['minutes']} level {checkpoint['level']} val {val['score']:.3f} | {splits}", flush=True)


if __name__ == "__main__":
    main()
