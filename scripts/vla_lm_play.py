"""Run a trained LM policy (lm_train's policy_best.pt) locally on held-out test windows and write
a GIF per window: the recorded play, the policy's rollout and the goal frame, side by side.

    uv run python scripts/vla_lm_play.py --run RUN_DIR --play_cache CACHE [--text "eat the red cheese"]
        [--windows 4] [--seed 0] [--guidance 2] [--out OUT_DIR] [--device mps]

Without --text the policy gets each window's goal frame; with it, the typed instruction instead
(the same text for every window). Prints each window's events in words and the rollout's facts
score against them (a typed instruction that asks for something else will score low by design).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from vla_demo_media import describe, episode, gif, snap  # noqa: E402

from beebot.vla import goal_eval  # noqa: E402
from beebot.vla.data import world_at  # noqa: E402
from beebot.vla.facts import window_facts  # noqa: E402
from beebot.vla.lm_policy import LMPolicy  # noqa: E402
from beebot.vla.train import Config, load_data  # noqa: E402
from beebot.vla.world import observe, step  # noqa: E402


def load(run, device, checkpoint="policy_best.pt"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = json.loads((Path(run) / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    lm = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32)
    policy = LMPolicy(lm, lora_rank=cfg["lora_rank"], freeze_embeddings=cfg.get("freeze_embeddings", True))
    saved = torch.load(Path(run) / checkpoint, map_location="cpu")
    weights = saved.get("weights", saved)
    missing = [n for n, p in policy.named_parameters() if p.requires_grad and n not in weights]
    assert not missing, f"checkpoint lacks trainable weights: {missing[:5]}"
    policy.load_state_dict({k: v.float() for k, v in weights.items()}, strict=False)
    return policy.to(device).eval(), tokenizer


def rollout(policy, store, w, device, guidance, text_ids=None, text_mask=None):
    """Worlds at steps 0..horizon (on the CPU) for windows w (B, 3), all with the same horizon."""
    e, w0, w1 = w[:, 0], w[:, 1], w[:, 2]
    world = world_at(store, e, w0, device)
    goal = None if text_ids is not None else policy.goal_encoder(torch.cat([observe(world), observe(world_at(store, e, w1, device))], dim=1))
    worlds = [world.to("cpu")]
    with torch.no_grad():
        for _ in range(int((w1 - w0).max())):
            chunk = policy.sample(observe(world), goal=goal, text_ids=text_ids, text_mask=text_mask, guidance=guidance)
            world, _ = step(world, chunk[:, 0])
            worlds.append(world.to("cpu"))
    return worlds


def recorded(store, w, device):
    """The recorded play over the same windows, as Worlds on the CPU."""
    e, w0, w1 = w[:, 0], w[:, 1], w[:, 2]
    return [world_at(store, e, (w0 + t).clamp(max=w1), device).to("cpu") for t in range(int((w1 - w0).max()) + 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="a run directory with config.json and policy_best.pt")
    parser.add_argument("--play_cache", required=True)
    parser.add_argument("--play_root", default="")
    parser.add_argument("--text", default="", help="an instruction to give instead of the goal frame")
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--out", default="runs/lm_play")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    policy, tokenizer = load(args.run, device)
    data_cfg = Config(grid=32, data="play", play_root=args.play_root, play_cache=args.play_cache, val_windows=128, test_windows=512)
    store, _, _, _, _, test_windows = load_data(data_cfg, "cpu")
    pick = torch.randperm(len(test_windows), generator=torch.Generator().manual_seed(args.seed))[: args.windows]
    for k, w in enumerate(test_windows[pick].split(1)):
        text_ids = text_mask = None
        if args.text:
            enc = tokenizer([args.text], return_tensors="pt")
            text_ids, text_mask = enc.input_ids.to(device), enc.attention_mask.to(device)
        worlds = rollout(policy, store, w, device, args.guidance, text_ids, text_mask)
        replay = recorded(store, w, device)
        e = int(w[0, 0])
        static = {key: store[key][e].numpy() for key in ("walls", "cheese_color", "box_rect", "box_color")}
        start, goal = snap(replay[0], 0), snap(replay[-1], 0)
        target = window_facts(static, start, goal).atoms
        scores = goal_eval.fact_scores(static, start, goal, [snap(world, 0) for world in worlds])
        label = f"Qwen: {args.text[:28]}" if args.text else "Qwen: goal image"
        path = out / f"window{k}.gif"
        gif(path, [("Recorded play", episode(replay, 0)), (label, episode(worlds, 0)), ("Goal", episode(replay[-1:], 0))], len(worlds) - 1)
        print(f"window {k} ({path}): events {describe(target)}; facts score {'n/a' if scores is None else round(scores[0], 3)}", flush=True)


if __name__ == "__main__":
    main()
