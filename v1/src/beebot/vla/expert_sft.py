"""Behaviour cloning of the LM policy on task-conditioned expert demonstrations (vla/experts.py),
mixed with the play captions it was trained on, starting from a trained policy (c1).

    uv run python -m beebot.vla.expert_sft overfit --tasks 4 --steps 300      # the learning gate
    uv run python -m beebot.vla.expert_sft train --name lm-expert-sft --minutes 90

overfit: a few fixed (scene, instruction) pairs from different families, their expert episodes
recorded in memory, the policy trained on exactly those, then rolled out on exactly those: if its
score on them does not rise well above the starting policy's, a larger run will not help.
train: demonstrations from expert stores plus play captions (caption_share of each batch), with
held-out demonstrations, weight decay and distillation of the frozen LM on general text; every
evaluation logs the held-out demonstration loss, the generalist scorecard, the grounding tests and
the language scores, and keeps a checkpoint.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from beebot.vla import describe, experts, tasks
from beebot.vla.data import batch_from
from beebot.vla.lm_policy import PATCH
from beebot.vla.lm_rl import RLConfig, generalist_card, lm_sampler, load_policy, make_suite, tokens
from beebot.vla.play_data import load_play_cache

C1 = "runs/lm-qwen-lora-c1"


def demo_batch(stores, labels, n, rng, generator, device):
    """n demonstration samples drawn uniformly over all labels; stores: one store (overfit) or a
    list of stores, each label's "store" naming its own (default 0)."""
    stores = stores if isinstance(stores, list) else [stores]
    picks = [labels[rng.randrange(len(labels))] for _ in range(n)]
    parts = []
    for k, store in enumerate(stores):
        mine = [p for p in picks if p.get("store", 0) == k]
        if not mine:
            continue
        steps = store["field"].shape[1]
        obs, _, actions, weights, _ = batch_from(store, torch.tensor([[p["e"], 0, steps] for p in mine]), generator, device)
        parts.append((obs, actions, weights, [p["text"] for p in mine]))
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]), torch.cat([p[2] for p in parts]), [t for p in parts for t in p[3]]


def sft_step(policy, optimizer, tokenizer, batch, device, null_prob=0.1):
    obs, actions, weights, texts = batch
    ids, mask = tokens(tokenizer, texts, device)
    null = torch.rand(len(texts), device=device) < null_prob
    loss = policy.policy_loss(obs, actions, weights, text_ids=ids, text_mask=mask, null=null)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
    optimizer.step()
    return loss.item()


def optimizer_for(policy, lr_lm, lr_new):
    return torch.optim.AdamW([{"params": policy.lm_parameters(), "lr": lr_lm}, {"params": policy.new_parameters(), "lr": lr_new}], weight_decay=0.0)


def greedy_sampler(policy, guidance, n):
    """lm_sampler with the most likely move per square instead of a sample."""
    from beebot.vla.data import decode_field

    def sample(obs, cond):
        ids, mask = cond
        out = policy.logits(obs, text_ids=ids, text_mask=mask)
        if guidance != 1.0:
            null = policy.logits(obs, text_ids=ids, text_mask=mask, null=torch.ones(len(obs), dtype=torch.bool, device=obs.device))
            out = null + guidance * (out - null)
        moves = out.float().argmax(-1)
        b, g = len(obs), obs.shape[-1]
        codes = moves.view(b, n, n, PATCH, PATCH).permute(0, 1, 3, 2, 4).reshape(b, g, g)
        return moves, decode_field(codes, obs.device)

    return sample


def rollout_scores(policy, tokenizer, pairs, device, guidance=2.0, steps=experts.STEPS, repeats=4, greedy=False):
    policy.eval()
    scenes = [s for s, _ in pairs for _ in range(repeats)]
    batch = [t for _, t in pairs for _ in range(repeats)]
    sampler = (greedy_sampler if greedy else lm_sampler)(policy, guidance, tasks.SIZE // PATCH)
    results, _ = tasks.run(sampler, lambda texts: tokens(tokenizer, texts, device), scenes, batch, steps, device)
    policy.train()
    return [round(float(np.mean([results[i * repeats + r][0] for r in range(repeats)])), 3) for i in range(len(pairs))]


def overfit(args):
    device = torch.device(args.device)
    policy, tokenizer = load_policy(RLConfig(init_run=C1, checkpoint="policy_best.pt"), device)
    rng = np.random.default_rng(args.seed)
    families = (args.families.split(",") if args.families else ["place", "logic", "formation", "parallel", "herd", "runner"])[: args.tasks]
    pairs = []
    for family in families:  # one task per family whose expert succeeds on it
        for _ in range(20):
            scene, task = tasks.sample_task(rng, "train", family=family)
            results, _ = experts.play([scene], [task])
            if experts.succeeded(task, results[0]):
                pairs.append((scene, task))
                break
    print("tasks:", [t.text for _, t in pairs], flush=True)
    # several expert episodes per task (the world's random move order differs between them)
    reps = 8
    results, rec = experts.play([s for s, _ in pairs for _ in range(reps)], [t for _, t in pairs for _ in range(reps)], seed=args.seed)
    labels = [{"e": i, "text": pairs[i // reps][1].text} for i, (t, r) in enumerate(zip([t for _, t in pairs for _ in range(reps)], results)) if experts.succeeded(t, r)]
    store = {**rec, "episode_weight": torch.ones(len(results))}
    print(f"{len(labels)} expert episodes kept of {len(results)}", flush=True)
    before = rollout_scores(policy, tokenizer, pairs, device)
    print("before:", dict(zip([t.family for _, t in pairs], before)), flush=True)
    optimizer = optimizer_for(policy, args.lr_lm, args.lr_new)
    py_rng, generator = random.Random(args.seed), torch.Generator().manual_seed(args.seed)
    for step in range(1, args.steps + 1):
        loss = sft_step(policy, optimizer, tokenizer, demo_batch(store, labels, args.batch, py_rng, generator, device), device)
        if step % 50 == 0:
            print(json.dumps({"step": step, "loss": round(loss, 4)}), flush=True)
    after = rollout_scores(policy, tokenizer, pairs, device)
    print("after:", dict(zip([t.family for _, t in pairs], after)), flush=True)
    print(json.dumps({"overfit": {t.family: {"before": b, "after": a} for (_, t), b, a in zip(pairs, before, after)}}), flush=True)
    if args.probe:  # how the trained policy fares under other sampling settings
        for guidance, steps in ((1.0, 64), (2.0, 96), (1.0, 96)):
            print(f"guidance {guidance}, {steps} steps:", dict(zip([t.family for _, t in pairs], rollout_scores(policy, tokenizer, pairs, device, guidance, steps))), flush=True)
        greedy = rollout_scores(policy, tokenizer, pairs, device, 2.0, 64, greedy=True)
        print("greedy (argmax moves):", dict(zip([t.family for _, t in pairs], greedy)), flush=True)


def fresh_policy(model, lora_rank, device):
    """A new LoRA policy on a pretrained LM (no earlier behaviour training): for size comparisons."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from beebot.vla.lm_policy import LMPolicy

    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.padding_side = "right"
    lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float16 if device.type == "cuda" else torch.float32)
    return LMPolicy(lm, lora_rank=lora_rank, freeze_embeddings=True).to(device), tokenizer


def split_labels(labels, val_share):
    """(training, held-out) labels: the last val_share of each store's episodes are held out."""
    last = {}
    for l in labels:
        last[l["store"]] = max(last.get(l["store"], 0), l["e"])
    cut = {k: (1 - val_share) * (n + 1) for k, n in last.items()}
    return [l for l in labels if l["e"] < cut[l["store"]]], [l for l in labels if l["e"] >= cut[l["store"]]]


def train(args):
    """Long behaviour cloning with what it takes to see the whole curve and not lose old skills:
    held-out demonstrations (their loss next to the training loss), weight decay, rehearsal of the
    play captions, distillation of the frozen LM on general text, and at every evaluation the task
    scorecard, the grounding tests and the language scores, with a checkpoint each time."""
    from types import SimpleNamespace

    from beebot.vla import grounding
    from beebot.vla.lm_probe import probe_items
    from beebot.vla.lm_rl import scorecard
    from beebot.vla.lm_train import Captions, distill_loss, general_text, language
    from beebot.vla.train import Config, load_data

    t_start = time.time()  # wall clock, setup and evaluations included (for wall_minutes)
    device = torch.device(args.device)
    amp = device.type == "cuda"
    cfg = RLConfig(init_run=args.init_run, checkpoint=args.checkpoint, eval_tasks=args.eval_tasks, eval_scenes=args.eval_scenes)
    if args.fresh_model:
        policy, tokenizer = fresh_policy(args.fresh_model, args.lora_rank, device)
        base = {"model": args.fresh_model, "lora_rank": args.lora_rank, "freeze_embeddings": True}
    else:
        policy, tokenizer = load_policy(cfg, device)
        base = {k: v for k, v in json.loads((Path(args.init_run) / "config.json").read_text()).items() if k != "expert_sft"}
    run_dir = Path("runs") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**base, "expert_sft": vars(args)}, indent=1))
    stores = [load_play_cache(path) for path in args.store]
    labels = [{**json.loads(l), "store": k} for k, path in enumerate(args.store) for l in open(Path(path) / "labels.jsonl")]
    train_labels, val_labels = split_labels(labels, args.val_share)
    play = load_play_cache(args.play_cache)
    captions = Captions(args.captions, play, tokenizer, 128, 128) if args.caption_share else None
    _, _, _, _, _, test_windows = load_data(Config(grid=32, data="play", play_cache=args.play_cache, val_windows=128, test_windows=512), "cpu")
    trials = grounding.make_trials(play, test_windows, args.eval_scenes)
    text_cfg = SimpleNamespace(general=args.general, lm_length=args.lm_length, gsm8k_share=0.2, instruct_share=0.3, seed=args.seed, gsm8k_problems=50)
    text_chunks, held_text, _ = general_text(text_cfg, tokenizer)  # held-out text: perplexity is the forgetting signal
    probe = {"world_probe": probe_items()}
    suite = make_suite(cfg)
    optimizer = torch.optim.AdamW([{"params": policy.lm_parameters(), "lr": args.lr_lm}, {"params": policy.new_parameters(), "lr": args.lr_new}], weight_decay=args.weight_decay)
    params = [p for g in optimizer.param_groups for p in g["params"]]
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    py_rng, generator = random.Random(args.seed), torch.Generator().manual_seed(args.seed)
    history = open(run_dir / "history.jsonl", "a")
    print(f"{len(train_labels)} training and {len(val_labels)} held-out demonstrations over {len(stores)} store(s)", flush=True)

    def autocast():
        return torch.autocast(device.type, dtype=torch.float16, enabled=amp)

    def log(row):
        print(json.dumps(row), flush=True)
        history.write(json.dumps(row) + "\n")
        history.flush()

    def demo_loss(batch, null=None):
        obs, actions, weights, texts = batch
        ids, mask = tokens(tokenizer, texts, device)
        with autocast():
            return policy.policy_loss(obs, actions, weights, text_ids=ids, text_mask=mask, null=null)

    @torch.no_grad()
    def held_out_loss():
        policy.eval()
        out = float(np.mean([demo_loss(demo_batch(stores, val_labels, args.batch, random.Random(k), torch.Generator().manual_seed(k), device)).item() for k in range(args.val_batches)]))
        train_part = float(np.mean([demo_loss(demo_batch(stores, train_labels, args.batch, random.Random(1000 + k), torch.Generator().manual_seed(1000 + k), device)).item() for k in range(args.val_batches)]))
        policy.train()
        return out, train_part

    def evaluate(step, minutes):
        val, train_fixed = held_out_loss()
        with autocast():
            card = generalist_card(policy, tokenizer, cfg, device, suite)
            ground = scorecard(policy, tokenizer, play, trials, cfg, device)
        lang = language(policy, tokenizer, held_text, probe, None, device, with_gsm=False)
        scene_qa = {"describe": describe.evaluate(policy, tokenizer, stores, val_labels, device)} if args.describe_batch else {}
        log({"step": step, "minutes": round(minutes, 1), "demo_val_loss": round(val, 4), "demo_train_loss_fixed": round(train_fixed, 4), "language": lang, "grounding": ground, "general": card, **scene_qa})
        trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
        weights = {k: v.half() for k, v in policy.state_dict().items() if k in trainable}
        torch.save(weights, run_dir / "policy_latest.pt")
        torch.save(weights, run_dir / f"policy_step{step}.pt")

    evaluate(0, 0.0)
    t0, eval_seconds, next_eval = time.time(), 0.0, args.eval_every_minutes * 60
    losses, kls, d_losses = [], [], []
    step = 0
    while True:
        elapsed = time.time() - t0 - eval_seconds
        if elapsed > args.minutes * 60 or (args.wall_minutes and time.time() - t_start > args.wall_minutes * 60):
            break
        step += 1
        fraction = min(elapsed / (args.minutes * 60), 1.0)
        for g, base_lr in zip(optimizer.param_groups, (args.lr_lm, args.lr_new)):
            g["lr"] = base_lr * min(1.0, step / args.warmup_steps) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * fraction)))
        optimizer.zero_grad(set_to_none=True)
        n_caption = int(round(args.batch * args.caption_share))
        null = torch.rand(args.batch - n_caption, device=device) < args.null_prob
        loss = demo_loss(demo_batch(stores, train_labels, args.batch - n_caption, py_rng, generator, device), null)
        scaler.scale(loss).backward()
        losses.append(loss.item())
        if n_caption:
            obs, actions, weights, ids, mask = captions.sample(play, n_caption, generator, device)
            with autocast():
                c_loss = policy.policy_loss(obs, actions, weights, text_ids=ids, text_mask=mask)
            scaler.scale(c_loss * n_caption / (args.batch - n_caption)).backward()
        if args.describe_batch:
            with autocast():
                d_loss = describe.loss(policy, tokenizer, describe.sample(stores, train_labels, args.describe_batch, py_rng, device), device)
            scaler.scale(args.describe_weight * d_loss).backward()
            d_losses.append(d_loss.item())
        if args.distill_weight:
            chunk = torch.stack([next(text_chunks) for _ in range(args.lm_batch)]).to(device)
            with autocast():
                kl = distill_loss(policy.lm, None, chunk)
            scaler.scale(args.distill_weight * kl).backward()
            kls.append(kl.item())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % 50 == 0:
            log({"step": step, "minutes": round(elapsed / 60, 1), "demo_loss": round(float(np.mean(losses[-50:])), 4), **({"distill_kl": round(float(np.mean(kls[-50:])), 5)} if kls else {}), **({"describe_loss": round(float(np.mean(d_losses[-50:])), 4)} if d_losses else {})})
        if elapsed >= next_eval:
            t_eval = time.time()
            evaluate(step, elapsed / 60)
            eval_seconds += time.time() - t_eval
            next_eval = elapsed + args.eval_every_minutes * 60
    evaluate(step, (time.time() - t0 - eval_seconds) / 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["overfit", "train"])
    parser.add_argument("--name", default="lm-expert-sft")
    parser.add_argument("--store", nargs="+", default=["data/vla/experts_g32"], help="train: one or more expert stores")
    parser.add_argument("--init_run", default=C1, help="train: the run to start from (its config.json and checkpoint)")
    parser.add_argument("--checkpoint", default="policy_best.pt")
    parser.add_argument("--fresh_model", default="", help="train: start from this pretrained LM with a new LoRA instead of init_run (size comparisons)")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--play_cache", default="data/vla/play_cache_g32")
    parser.add_argument("--play_root", default="", help="accepted for the Kaggle kernel's arguments; the cache is what is read")
    parser.add_argument("--captions", default="data/vla/kaggle_play/captions_g32.jsonl")
    parser.add_argument("--caption_share", type=float, default=0.25)
    parser.add_argument("--general", default="hf", help="general text for distillation: hf (FineWeb-Edu, GSM8K, chats) or a local text file")
    parser.add_argument("--distill_weight", type=float, default=1.0)
    parser.add_argument("--lm_batch", type=int, default=2)
    parser.add_argument("--lm_length", type=int, default=256)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--describe_batch", type=int, default=0, help="train: scene questions per step (describe.py): the text attends to the scene tokens and answers which cheeses are present or were eaten")
    parser.add_argument("--describe_weight", type=float, default=1.0)
    parser.add_argument("--null_prob", type=float, default=0.1)
    parser.add_argument("--val_share", type=float, default=0.05, help="train: last share of each store's episodes held out")
    parser.add_argument("--val_batches", type=int, default=8)
    parser.add_argument("--eval_scenes", type=int, default=32, help="train: grounding test scenes")
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--families", default="", help="overfit: comma-separated families (default: the first --tasks of place, logic, formation, parallel, herd, runner)")
    parser.add_argument("--probe", action="store_true", help="overfit: also score under other guidance, lengths and greedy moves")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--minutes", type=float, default=90.0, help="training time, evaluations excluded")
    parser.add_argument("--wall_minutes", type=float, default=0.0, help="train: also stop when this much wall time has passed since start (0: off), before a final evaluation")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr_lm", type=float, default=1e-4)
    parser.add_argument("--lr_new", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_every_minutes", type=float, default=30.0)
    parser.add_argument("--eval_tasks", type=int, default=24)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    overfit(args) if args.command == "overfit" else train(args)


if __name__ == "__main__":
    main()
