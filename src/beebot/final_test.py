"""Score a trained model, and the hand-written baseline, on val and on the untouched test set.

    uv run final-test runs/<name>

Run this once per finalist, after the hill-climb is over. Re-running it to pick
between candidates turns the test set into a second val set.
"""

import json
import sys
from pathlib import Path

import torch

from beebot.evaluate import SMELL_LENGTH, evaluate
from beebot.model import NCA, GreedyBaseline


def main():
    run_dir = Path(sys.argv[1])
    device = torch.device(sys.argv[2] if len(sys.argv) > 2 else "mps")
    config = json.loads((run_dir / "config.json").read_text())
    model = NCA(hidden=config["hidden"], n_hidden_layers=config["n_hidden_layers"]).to(device)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.eval()
    baseline = GreedyBaseline(SMELL_LENGTH)

    results = {
        "model": {"val": evaluate(model, device, "val"), "test": evaluate(model, device, "test")},
        "baseline": {"val": evaluate(baseline, device, "val"), "test": evaluate(baseline, device, "test")},
    }
    (run_dir / "final_test.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
