"""Run a trained relative-position policy (vla/swarm.py) unchanged on bigger worlds: scripted
test scenes at each grid size, eventful windows only, re-planning every step.

    .venv/bin/python scripts/vla_size_transfer.py runs/vla-rel-g32s-global --sizes 32 48 64

Writes rows to <run>/size_transfer.jsonl: size, guidance, the facts score and its recalls,
and the do-nothing and replay baselines at that size.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import torch

from beebot.vla import goal_eval
from beebot.vla.data import make_windows
from beebot.vla.swarm import RelativePolicy, frequency_aware_scale
from beebot.vla.train import Config, load_or_generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 48, 64])
    parser.add_argument("--episodes", type=int, default=128)  # g64 episodes are slow to generate (257 steps, about 380 robots)
    parser.add_argument("--windows", type=int, default=256)
    parser.add_argument("--guidance", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--q-scale", type=float, nargs="+", default=[1.0], help="multiply attention queries (sharper attention over more tokens than in training)")
    parser.add_argument("--pos-scale", type=float, nargs="+", default=[1.0], help="multiply rotary positions (position interpolation: training size / test size)")
    parser.add_argument("--interpolation", nargs="+", default=["plain"], choices=["plain", "frequency"], help="plain: --pos-scale on every frequency; frequency: fast (short-offset) frequencies exact, slow ones squeezed by trained / test size (the YaRN ramp), --pos-scale ignored")
    args = parser.parse_args()
    run = Path(args.run)
    stored = json.loads((run / "config.json").read_text())
    cfg = Config(**{k: v for k, v in stored.items() if k in {f.name for f in dataclasses.fields(Config)}})
    device = torch.device(args.device)
    policy = RelativePolicy(cfg.view, width=cfg.width, layers=cfg.layers, crop=cfg.crop).to(device)
    policy.load_state_dict(torch.load(run / "model_best.pt", map_location=device))
    policy.eval()
    out = run / "size_transfer.jsonl"
    for size in args.sizes:
        # Seed 20_000 + seed is the test split of scripted data at every size (train.load_data).
        store = load_or_generate(size, args.episodes, 20_000 + cfg.seed, device)
        windows = goal_eval.eventful_windows(store, make_windows(store, size), device, args.windows, torch.Generator().manual_seed(1))
        baselines = {mode: goal_eval.summary(goal_eval.run(None, store, windows, device, mode=mode))["fact_score"] for mode in ("zero", "replay")}
        settings = [(q, p, g, "plain") for q in args.q_scale for p in args.pos_scale for g in args.guidance] if "plain" in args.interpolation else []
        settings += [(q, 1.0, g, "frequency") for q in args.q_scale for g in args.guidance] if "frequency" in args.interpolation else []
        for q_scale, pos_scale, guidance, interpolation in settings:
            for module in policy.modules():
                if hasattr(module, "q_scale"):
                    module.q_scale = q_scale
            policy.pos_scale = pos_scale
            policy.freq_scale = frequency_aware_scale(cfg.grid, size) if interpolation == "frequency" else None
            with torch.no_grad():
                scores = goal_eval.summary(goal_eval.run(policy, store, windows, device, guidance=guidance, execute=1))
            row = {"size": size, "trained_on": cfg.grid, "guidance": guidance, "q_scale": q_scale, "pos_scale": pos_scale, "interpolation": interpolation, **scores, "zero_facts": baselines["zero"], "replay_facts": baselines["replay"]}
            print(json.dumps(row), flush=True)
            with open(out, "a") as f:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
