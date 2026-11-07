"""Type instructions to a trained swarm text policy and watch what it does: the score-to-vibes check.

    uv run python scripts/vla_swarm_demo.py --run runs/vla-rel-g32-swarm-text [--guidance 4]
        [--scene 0] [--steps 64] [--out runs/swarm_demo] [--text "eat the red cheese" ...]
    uv run python scripts/vla_swarm_demo.py --run runs/lm-qwen-lora-c1 --guidance 2   # a Qwen LM policy

The run is either a relative-position text run (train.py: config.json and model_best.pt; the
instruction goes through the frozen caption embedding) or a Qwen LM policy (lm_train.py:
config.json with "lora_rank" and policy_best.pt; the instruction goes in as tokens).

Without --text it reads instructions from the terminal, one per line, on a held-out test scene:
each writes a GIF (the scene at the start and the swarm's rollout) to --out, opens it, and prints
the scene and the events that happened, in the scorer's words (what requested-events scoring
would credit). Commands: ":scene N" picks test scene N, ":random" a random one, ":again" reruns
the last instruction (rollouts are sampled), ":quit" ends.
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from vla_demo_media import describe, episode, gif  # noqa: E402
from vla_lm_play import load as load_lm  # noqa: E402
from vla_text_eval import load_policy  # noqa: E402

from beebot.vla import goal_eval  # noqa: E402
from beebot.vla.caption import PALETTE_NAMES  # noqa: E402
from beebot.vla.data import world_at  # noqa: E402
from beebot.vla.facts import window_facts  # noqa: E402
from beebot.vla.play_data import load_play_cache  # noqa: E402
from beebot.vla.text import TextEncoder  # noqa: E402
from beebot.vla.train import Config, load_data  # noqa: E402
from beebot.vla.world import observe, step  # noqa: E402


def scene_words(store, e, t):
    boxes = [f"{PALETTE_NAMES[c]} box" for (r0, c0, h, w), c in zip(store["box_rect"][e].tolist(), store["box_color"][e].tolist()) if h > 0 and w > 0]
    cheeses = [f"{PALETTE_NAMES[c]} cheese" for c, alive in zip(store["cheese_color"][e].tolist(), store["cheese_alive"][e, t].tolist()) if alive]
    return ", ".join(cheeses + boxes) or "nothing but the swarm"


class SwarmText:
    """The relative-position policy with the frozen caption embedding."""

    def __init__(self, run, device):
        self.policy, _ = load_policy(run, device)
        self.encoder, self.device = TextEncoder(device="cpu"), device

    def condition(self, text):
        return self.policy.text_head(self.encoder.embed([text]).to(self.device))

    def sample(self, obs, condition, guidance):
        return self.policy.sample(obs, condition, guidance=guidance)


class QwenText:
    """A Qwen LM policy reading the instruction as tokens."""

    def __init__(self, run, device):
        self.policy, self.tokenizer = load_lm(run, device)
        self.device = device

    def condition(self, text):
        enc = self.tokenizer([text], return_tensors="pt")
        return enc.input_ids.to(self.device), enc.attention_mask.to(self.device)

    def sample(self, obs, condition, guidance):
        ids, mask = condition
        return self.policy.sample(obs, text_ids=ids, text_mask=mask, guidance=guidance)


@torch.no_grad()
def act(policy, store, window, z, steps, guidance, device):
    """The worlds of a rollout from the window's start (on the CPU) and the events at its end and
    at any point, from the start frame."""
    e, w0 = int(window[0]), int(window[1])
    world = world_at(store, torch.tensor([e]), torch.tensor([w0]), device)
    worlds = [world.to("cpu")]
    for _ in range(steps):
        chunk = policy.sample(observe(world), z, guidance)
        world, _ = step(world, chunk[:, 0])
        worlds.append(world.to("cpu"))
    static = {k: store[k][e].numpy() for k in ("walls", "cheese_color", "box_rect", "box_color")}
    start = goal_eval.stored_snapshot(store, e, w0)
    found = [window_facts(static, start, goal_eval.snapshot(w, 0)).atoms for w in worlds[goal_eval.FACT_EVERY :: goal_eval.FACT_EVERY]]
    return worlds, found[-1], set().union(*found)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--play_cache", default="data/vla/play_cache_g32")
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--scene", type=int, default=0, help="index into the held-out test scenes")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--text", nargs="*", default=[], help="instructions to run without prompting")
    parser.add_argument("--out", default="runs/swarm_demo")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--no_open", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    is_lm = "lora_rank" in json.loads((Path(args.run) / "config.json").read_text())
    policy = (QwenText if is_lm else SwarmText)(args.run, device)
    store = load_play_cache(args.play_cache)
    data_cfg = Config(grid=32, data="play", play_cache=args.play_cache, val_windows=128, test_windows=512)
    test_windows = load_data(data_cfg, "cpu")[5]
    scene = args.scene
    count = 0

    def run_one(text):
        nonlocal count
        window = test_windows[scene]
        z = policy.condition(text)
        worlds, at_end, at_any = act(policy, store, window, z, args.steps, args.guidance, device)
        count += 1
        path = out / f"{count:03d}-scene{scene}-{'-'.join(text.lower().split())[:40]}.gif"
        gif(path, [(text[:34], episode(worlds, 0))], len(worlds) - 1)
        print(f"  at the end: {', '.join(describe(at_end)) or 'nothing'}")
        print(f"  at some point: {', '.join(describe(at_any - at_end)) or '(nothing else)'}")
        print(f"  {path}", flush=True)
        if not args.no_open:
            subprocess.run(["open", str(path)], check=False)

    def show_scene():
        e, w0 = int(test_windows[scene][0]), int(test_windows[scene][1])
        print(f"scene {scene}: {scene_words(store, e, w0)}", flush=True)

    show_scene()
    for text in args.text:
        print(f"> {text}")
        run_one(text)
    if args.text:
        return
    last = None
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line == ":quit":
            break
        if line.startswith(":scene"):
            scene = int(line.split()[1]) % len(test_windows)
            show_scene()
            continue
        if line == ":random":
            scene = random.randrange(len(test_windows))
            show_scene()
            continue
        if line == ":again":
            line = last
            if line is None:
                continue
        last = line
        run_one(line)


if __name__ == "__main__":
    main()
