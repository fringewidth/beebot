"""Score the step-named checkpoints of an expert_sft run (trained elsewhere, e.g. on Kaggle with
--val_batches 0) on the Mac: the held-out and fixed-training demonstration loss of each, computed the
same way as expert_sft's own evaluations, with the same held-out split.

    uv run python v1/scripts/scale_curve_score.py runs/kaggle/<slug>/runs/scale-curve-4b --arm 4b

Appends one JSON line per checkpoint ({"arm", "step", "demo_val_loss", "demo_train_loss_fixed"}) and
the run's last training-progress row (for seconds per step and peak memory) to --log; steps already
in --log for this arm are skipped, so a stopped scoring resumes. Plot with scale_curve_plot.py.
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch

from beebot.vla.expert_sft import fresh_policy, held_out_losses, split_labels
from beebot.vla.play_data import load_play_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--log", default="logs/scale-curve-kaggle.out")
    parser.add_argument("--stores", default="data/vla", help="local directory holding the run's stores (by name)")
    parser.add_argument("--val_batches", type=int, default=16)
    parser.add_argument("--every", type=int, default=100, help="score the checkpoints at multiples of this step")
    parser.add_argument("--weights_dtype", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    run = Path(args.run_dir)
    config = json.loads((run / "config.json").read_text())
    sft = config["expert_sft"]
    device = torch.device(args.device)
    log = Path(args.log)
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            row = json.loads(line) if line.startswith("{") else {}
            if row.get("arm") == args.arm and "demo_val_loss" in row:
                done.add(row["step"])
    steps = sorted(int(m.group(1)) for p in run.glob("policy_step*.pt") if (m := re.fullmatch(r"policy_step(\d+)\.pt", p.name)))
    todo = [s for s in steps if s % args.every == 0 and s not in done]
    print(f"{args.arm}: checkpoints at {steps}; scoring {todo}", flush=True)

    store_paths = [Path(args.stores) / Path(p).name for p in sft["store"]]
    stores = [load_play_cache(path) for path in store_paths]
    labels = [{**json.loads(l), "store": k} for k, path in enumerate(store_paths) for l in open(path / "labels.jsonl")]
    train_labels, val_labels = split_labels(labels, sft["val_share"])
    print(f"{len(train_labels)} training and {len(val_labels)} held-out demonstrations", flush=True)

    torch.manual_seed(sft["seed"])
    policy, tokenizer = fresh_policy(config["model"], config["lora_rank"], device, args.weights_dtype, config.get("random_init", False))
    if config.get("random_init"):  # the run's own random frozen weights
        base = torch.load(run / "base_weights.pt", map_location="cpu")
        missing, unexpected = policy.load_state_dict(base, strict=False)
    trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
    if config.get("random_init"):
        assert not unexpected and set(missing) == trainable, (unexpected[:5], sorted(set(missing) ^ trainable)[:5])
    out = open(log, "a")
    for step in todo:
        t0 = time.time()
        weights = torch.load(run / f"policy_step{step}.pt", map_location="cpu")
        assert set(weights) == trainable, (sorted(set(weights) ^ trainable))[:5]
        policy.load_state_dict(weights, strict=False)
        val, train_fixed = held_out_losses(policy, tokenizer, stores, train_labels, val_labels, sft["batch"], args.val_batches, device)
        row = {"arm": args.arm, "step": step, "demo_val_loss": round(val, 4), "demo_train_loss_fixed": round(train_fixed, 4), "scored_seconds": round(time.time() - t0, 1)}
        print(json.dumps(row), flush=True)
        out.write(json.dumps(row) + "\n")
        out.flush()
    history = [json.loads(l) for l in open(run / "history.jsonl") if l.startswith("{")]
    progress = [r for r in history if "demo_loss" in r]
    if progress:  # training speed and memory where it trained
        row = {**progress[-1], "arm": args.arm}
        out.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
