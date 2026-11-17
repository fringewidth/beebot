"""Local learning gate for the voxel policy, before any Kaggle run: an overfit test. One scene,
several training instructions on it (one per family it allows, voxel/demos.contrast); the expert
demonstrates each a few times; the policy trains on only those demonstrations; then it must solve
each instruction from that same start, and not when given another instruction of the set (the
scene is the same, so ignoring the text cannot pass).

    uv run python -m beebot.vla.voxel.gate --steps 400
"""

import argparse
import json
import random
import time

import numpy as np
import torch

from beebot.vla.voxel import suite as S
from beebot.vla.voxel.demos import FIELDS, contrast
from beebot.vla.voxel.lm import VoxelLMPolicy, batch_of, tokens
from beebot.vla.voxel.world import VoxelWorld, observe


def repeat(world, metas, tasks, k):
    idx = [i for i in range(len(tasks)) for _ in range(k)]
    return VoxelWorld(*(getattr(world, f)[idx] for f in FIELDS)), [metas[i] for i in idx], [tasks[i] for i in idx]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--demos", type=int, default=12, help="expert runs per instruction")
    parser.add_argument("--evals", type=int, default=4, help="policy runs per instruction")
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    world, metas, tasks = S.sample_tasks(rng, 1, "train", family="herd")
    world, metas, tasks = contrast(rng, world, metas, tasks)
    print(f"{len(tasks)} instructions on one scene: {[t.text for t in tasks]}", flush=True)
    w, m, t = repeat(world, metas, tasks, args.demos)
    results, arrays = S.run(w, m, t, S.Expert(t), seed=args.seed, record=True)
    keep = [i for i, (s, u) in enumerate(results) if not u and s >= success_threshold(t[i])]
    store = {"mass": arrays["mass"][keep], "actions": arrays["actions"][keep], "cheese_pos": arrays["cheese_pos"][keep], "cheese_alive": arrays["cheese_alive"][keep], **{f: getattr(w, f)[keep] for f in ("walls", "cheese_color", "gravity")}}
    labels = [{"e": j, "w0": 0, "w1": S.EPISODE_STEPS, "text": t[i].text} for j, i in enumerate(keep)]
    expert = {task.text: float(np.mean([results[i][0] for i in range(len(results)) if t[i].text == task.text])) for task in tasks}
    print(f"expert success per instruction {json.dumps({k[:40]: round(v, 2) for k, v in expert.items()})}; {len(keep)} demos kept", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    tokenizer.padding_side = "right"
    policy = VoxelLMPolicy(AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32), lora_rank=64).to(device)
    optimizer = torch.optim.AdamW([{"params": policy.lm_parameters(), "lr": 2e-4}, {"params": policy.new_parameters(), "lr": 3e-4}])
    pyrng = random.Random(0)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        obs, codes, texts = batch_of(store, labels, args.batch, pyrng, device)
        ids, mask = tokens(tokenizer, texts, device)
        loss = policy.loss(obs, codes, ids, mask, torch.rand(args.batch, device=device) < 0.1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.3f} ({time.time() - t0:.0f} s)", flush=True)

    policy.eval()
    w, m, t = repeat(world, metas, tasks, args.evals)
    texts = [x.text for x in t]
    shifted = [tasks[(tasks.index(x) + 1) % len(tasks)].text for x in t]  # another instruction of the set
    out = {}
    for name, these in (("own", texts), ("other", shifted)):
        ids, mask = tokens(tokenizer, these, device)
        actor = lambda wd, step_t: policy.act(observe(wd).to(device), ids, mask, args.guidance).cpu()  # noqa: E731
        out[name] = S.run(w, m, t, actor, seed=args.seed)[0]
    rows = {}
    for task in tasks:
        idx = [i for i in range(len(t)) if t[i].text == task.text]
        rows[f"{task.family}: {task.text[:40]}"] = {"expert": round(expert[task.text], 2), "own": round(float(np.mean([out["own"][i][0] for i in idx])), 2), "other": round(float(np.mean([out["other"][i][0] for i in idx])), 2)}
    print(json.dumps(rows, indent=1), flush=True)


def success_threshold(task):
    return 0.5 if task.family == "formation" else 0.75


if __name__ == "__main__":
    main()
