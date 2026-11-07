"""Does a text policy ground what it is told, or follow the scene? Controlled instructions on
held-out test scenes, every one a counterfactual: the same scene under different words.

    uv run python scripts/vla_grounding_eval.py RUN [RUN ...] [--scenes 128] [--steps 64]

Tests (beebot.vla.grounding; each from a test window's start state, STEPS steps):
  eat_colour    scenes with cheeses of 2+ colours; "eat the X cheese" for each colour present.
                hit: X eaten; wrong: another cheese eaten; first_right: X was the first eaten.
  box_colour    scenes with boxes of 2+ colours; "go into the X box" for each.
                hit: the swarm went into X (the scorer's "into box" event); wrong: into another.
  eat_gone      the eat_colour scenes with the X cheese removed before the start (as if just
                eaten); "eat the X cheese". unasked: another cheese eaten.
  eat_absent    scenes without any X cheese (X a palette colour), at least one cheese present;
                "eat the X cheese". unasked: a cheese eaten.
  control       the eat_colour scenes; "gather in the center". unasked: a cheese eaten (how much
                the scene alone makes the swarm eat).
A run is a relative-position text run (train.py) or a Qwen LM policy (lm_train.py). Rows are
printed and appended to RUN/grounding_eval.jsonl.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from vla_swarm_demo import QwenText, SwarmText  # noqa: E402

from beebot.vla.grounding import make_trials, rollout, summarise  # noqa: E402
from beebot.vla.play_data import load_play_cache  # noqa: E402
from beebot.vla.train import Config, load_data  # noqa: E402


def batched(policy, texts, device):
    """The per-row condition for a batch of instructions."""
    if isinstance(policy, QwenText):
        enc = policy.tokenizer(texts, return_tensors="pt", padding=True)
        return enc.input_ids.to(device), enc.attention_mask.to(device)
    return policy.policy.text_head(policy.encoder.embed(texts).to(device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--scenes", type=int, default=128, help="scenes per test")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--guidance", type=float, default=0, help="0: the model's own default (Qwen 2, swarm 4)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--play_cache", default="data/vla/play_cache_g32")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)

    store = load_play_cache(args.play_cache)
    windows = load_data(Config(grid=32, data="play", play_cache=args.play_cache, val_windows=128, test_windows=512), "cpu")[5]
    trials = make_trials(store, windows, args.scenes)
    print(f"{len(trials)} trials from {len(windows)} test scenes", flush=True)
    for run in args.runs:
        is_lm = "lora_rank" in json.loads((Path(run) / "config.json").read_text())
        policy = (QwenText if is_lm else SwarmText)(run, device)
        guidance = args.guidance or (2.0 if is_lm else 4.0)
        results = rollout(lambda obs, cond: policy.sample(obs, cond, guidance), lambda texts: batched(policy, texts, device), trials, store, args.steps, device, args.batch)
        row = {"run": Path(run).name, "guidance": guidance, "steps": args.steps, **summarise(results)}
        print(json.dumps(row), flush=True)
        with open(Path(run) / "grounding_eval.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        del policy
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
