"""2D under the voxel pilot's recipe, for an equal-compute comparison of how hard each world is:
a fresh LoRA on Qwen, text-only imitation of play windows under their hindsight task labels
(vla/relabel.py's captions_tasks_g32.jsonl), the same steps and batch, then a scorecard with the
same shuffled-instruction control (another scene's instruction of the same family) on the 2D
families that match the voxel ones: place (move, region, box), herd (push), logic (eat), parallel
(split up).

    uv run python scripts/vla_lm_fair_2d.py --name lm2d-fair --steps 3586
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from beebot.vla import tasks
from beebot.vla.lm_policy import PATCH, LMPolicy
from beebot.vla.lm_rl import lm_sampler, tokens
from beebot.vla.lm_train import Captions
from beebot.vla.play_data import load_play_cache

FAMILIES = ("place", "herd", "logic", "parallel")


def scorecard(policy, tokenizer, per_family, steps, guidance, device, seed=4242):
    policy.eval()
    rng = np.random.default_rng(seed)
    rows = {}
    for family in FAMILIES:
        pairs = [tasks.sample_task(rng, "test", family=family) for _ in range(per_family)]
        scenes, batch = [s for s, _ in pairs], [t for _, t in pairs]
        shuffled = [tasks.Task(t.family, batch[(i + 1) % len(batch)].text, t.spec, t.held_out) for i, t in enumerate(batch)]
        out = {}
        for name, these in (("score", batch), ("shuffled", shuffled)):
            torch.manual_seed(seed)
            results, _ = tasks.run(lm_sampler(policy, guidance, tasks.SIZE // PATCH), lambda texts: tokens(tokenizer, texts, device), scenes, these, steps, device)
            out[name] = results
        rows[family] = {"n": per_family, "score": round(float(np.mean([s for s, _ in out["score"]])), 3), "shuffled": round(float(np.mean([s for s, _ in out["shuffled"]])), 3), "unasked": round(float(np.mean([u for _, u in out["score"]])), 3)}
    policy.train()
    return rows


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="lm2d-fair")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--play_cache", default="data/vla/play_cache_g32")
    parser.add_argument("--labels", default="data/vla/kaggle_play/captions_tasks_g32.jsonl")
    parser.add_argument("--steps", type=int, default=3586)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lr_lm", type=float, default=2e-4)
    parser.add_argument("--lr_new", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--null_prob", type=float, default=0.1)
    parser.add_argument("--per_family", type=int, default=12)
    parser.add_argument("--eval_steps", type=int, default=48)
    parser.add_argument("--eval_guidance", type=float, default=2.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    run_dir = Path("runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=1))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    policy = LMPolicy(AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32), lora_rank=args.lora_rank).to(device)
    store = load_play_cache(args.play_cache)
    captions = Captions(args.labels, store, tokenizer, 128, 128)
    optimizer = torch.optim.AdamW([{"params": policy.lm_parameters(), "lr": args.lr_lm}, {"params": policy.new_parameters(), "lr": args.lr_new}], weight_decay=0.01)
    generator = torch.Generator().manual_seed(args.seed)
    history = open(run_dir / "history.jsonl", "a")

    def log(row):
        print(json.dumps(row), flush=True)
        history.write(json.dumps(row) + "\n")
        history.flush()

    log({"step": 0, "card": scorecard(policy, tokenizer, args.per_family, args.eval_steps, args.eval_guidance, device)})
    t0, losses = time.time(), []
    for step in range(1, args.steps + 1):
        fraction = step / args.steps
        for g, base in zip(optimizer.param_groups, (args.lr_lm, args.lr_new)):
            g["lr"] = base * min(1.0, step / args.warmup_steps) * 0.5 * (1 + math.cos(math.pi * fraction))
        obs, actions, weights, ids, mask = captions.sample(store, args.batch, generator, device)
        null = torch.rand(args.batch, device=device) < args.null_prob
        loss = policy.policy_loss(obs, actions, weights, text_ids=ids, text_mask=mask, null=null)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
        optimizer.step()
        losses.append(loss.item())
        if step % 50 == 0:
            log({"step": step, "minutes": round((time.time() - t0) / 60, 1), "loss": round(float(np.mean(losses[-50:])), 4)})
    log({"step": args.steps, "card": scorecard(policy, tokenizer, args.per_family, args.eval_steps, args.eval_guidance, device)})
    trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
    torch.save({k: v.half() for k, v in policy.state_dict().items() if k in trainable}, run_dir / "policy_latest.pt")


if __name__ == "__main__":
    main()
