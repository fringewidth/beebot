"""Fine-tune a pretrained LM (Qwen3) as the readout policy (vla/lm_policy.py), on goal frames and
on captions as text tokens, while distilling the frozen original on general text and GSM8K train
problems so the LM keeps its language and world knowledge.

    python -m beebot.vla.lm_train --name lm-full --play_root PLAY --captions CAPTIONS.jsonl \
        --general hf --device cuda [--lora_rank 64]
    torchrun --standalone --nproc_per_node 2 -m beebot.vla.lm_train ...   # one run on two GPUs

On several GPUs (torchrun) every process trains on its own batches and the gradients are averaged
after each backward pass; process 0 keeps the clock (learning rate, evaluations, stopping) and
broadcasts it, evaluates and saves, while the others wait.

Every eval: goal facts and caption facts (plus a shuffled-caption baseline) on held-out programs,
re-planning every step; perplexity on held-out general text and the world-knowledge probe; GSM8K
accuracy every gsm8k_every evals. Writes runs/<name>/ (config, history, best checkpoint, summary).
"""

import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from beebot.vla import goal_eval
from beebot.vla.caption_set import resolve, variant_rows
from beebot.vla.data import batch_from, sample_batch
from beebot.vla.lm_policy import lora_disabled, EvalAdapter, LMPolicy
from beebot.vla.lm_probe import benchmark_items, gsm8k_accuracy, mc_accuracy, perplexity, probe_items
from beebot.vla.tpu import checkpoint  # torch_xla's on XLA tensors, torch's non-reentrant elsewhere

from beebot.vla.train import Config, load_data, parse_config, program_split


@dataclass
class LMConfig:
    name: str = "lm"
    model: str = "Qwen/Qwen3-0.6B"
    play_root: str = ""
    play_cache: str = ""  # a directory written by play_data.cache_play (shared between runs)
    captions: str = ""  # caption records (jsonl); their variants are built here
    general: str = "hf"  # "hf" (FineWeb-Edu and GSM8K from the Hub) or a local text file for smoke tests
    lora_rank: int = 0  # 0: full fine-tuning
    steps: int = 12000
    minutes: float = 0.0  # training-time budget; 0 means steps only
    goal_batch: int = 16
    text_batch: int = 8
    lm_batch: int = 4
    lm_length: int = 256
    gsm8k_share: float = 0.1  # shares of the distillation chunks: GSM8K train, SmolTalk chats, the rest FineWeb-Edu
    instruct_share: float = 0.3
    bench_items: int = 150  # per benchmark (MMLU, ARC-Challenge, HellaSwag)
    distill_weight: float = 1.0
    lr_new: float = 3e-4
    lr_lm: float = 2e-5  # pretrained weights (LoRA: use about 2e-4)
    weight_decay: float = 0.01
    warmup_steps: int = 300
    null_prob: float = 0.1
    eval_every_minutes: float = 45.0
    log_every: int = 50  # training steps between progress lines
    wall_minutes: float = 0.0  # > 0: stop training this long after the process started, evaluations and setup included (Kaggle's 12 h cap)
    init: str = ""  # a policy_best.pt / policy_latest.pt to start from (trainable weights; optimizer too if present)
    schedule_offset: float = 0.0  # resuming: share of the learning-rate decay already done (the rest is spread over this run's budget)
    eval_only: bool = False  # with init: skip training, run the final language and test evaluation
    final_test: bool = True  # False: end after training (a session that a resumed one will continue)
    save_latest: bool = True  # at every evaluation also save the latest weights and optimizer state, to resume from
    first_eval_step: int = 20  # one early evaluation, so a broken evaluation path fails in minutes, not hours
    freeze_embeddings: bool = True  # full fine-tuning: keep the (tied) token embeddings and output head frozen
    eval_windows: int = 128
    test_windows: int = 512
    gsm8k_problems: int = 100
    gsm8k_every: int = 3
    gsm8k_max_tokens: int = 320
    checkpointing: bool = True
    half_base: bool = True  # LoRA: keep the frozen LM weights in fp16 instead of casting fp32 copies every step
    device: str = "cuda"
    seed: int = 0


class Captions:
    """Caption variants (texts) on training windows, and one identity caption per held-out window."""

    def __init__(self, path, store, tokenizer, val_limit, test_limit):
        rows = variant_rows(path, seed=0)
        windows, program = resolve(rows, store)
        split = torch.tensor([("train", "val", "test").index(program_split(p)) if e >= 0 else -1 for (p, _, _, _), e in zip(rows["keys"], windows[:, 0].tolist())])
        variant_split = split[rows["window"]]
        self.rows, self.windows, self.tokenizer = rows, windows, tokenizer
        self.train = (variant_split == 0).nonzero().flatten()
        identity = (rows["group"] == 0) & (rows["perm"] == torch.arange(4)).all(1)
        self.held_out = {}
        for s, name, limit in ((1, "val", val_limit), (2, "test", test_limit)):
            by_window = {}
            for v in (identity & (variant_split == s)).nonzero().flatten().tolist():
                by_window.setdefault(int(rows["window"][v]), v)
            chosen = random.Random(0).sample(sorted(by_window), min(limit, len(by_window)))
            self.held_out[name] = [by_window[w] for w in chosen]
        print(f"captions: {len(self.train)} training variants; {len(self.held_out['val'])} val, {len(self.held_out['test'])} test windows", flush=True)

    def tokens(self, texts, device):
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=96)
        return enc.input_ids.to(device), enc.attention_mask.to(device)

    def sample(self, store, n, generator, device):
        v = self.train[torch.randint(len(self.train), (n,), generator=generator)]
        r = self.rows
        perm, group = r["perm"][v], r["group"][v]
        random_perm = torch.stack([torch.randperm(4, generator=generator) for _ in range(n)])
        random_group = torch.randint(8, (n,), generator=generator)
        perm = torch.where(r["colour_free"][v].view(-1, 1), random_perm, perm)
        group = torch.where(r["rotation_free"][v], random_group, group)
        obs, goal, actions, weights, future = batch_from(store, self.windows[r["window"][v]], generator, device, perm, group)
        ids, mask = self.tokens([r["texts"][i] for i in v.tolist()], device)
        return obs, actions, weights, ids, mask

    def eval_set(self, name):
        """(windows (N, 3), padded token ids as floats with -1 for padding) for goal_eval.run."""
        v = self.held_out[name]
        ids, mask = self.tokens([self.rows["texts"][i] for i in v], "cpu")
        return self.windows[self.rows["window"][v]], torch.where(mask.bool(), ids, torch.full_like(ids, -1)).float()


def general_text(cfg, tokenizer, held_out=64, rank=0, world=1):
    """(training chunk iterator, held-out chunks for perplexity, GSM8K test problems). With several
    processes each takes every world-th chunk of every stream, so none trains on another's text;
    the held-out chunks are the same for all."""
    if cfg.general == "hf":
        from datasets import load_dataset

        web = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        gsm_train = list(load_dataset("openai/gsm8k", "main", split="train"))
        gsm_test = list(load_dataset("openai/gsm8k", "main", split="test"))
        chats = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
        web_texts = (row["text"] for row in web)
        chat_texts = (tokenizer.apply_chat_template(row["messages"], tokenize=False) for row in chats)
    else:
        lines = [l for l in Path(cfg.general).read_text().split("\n\n") if len(l) > 200]
        web_texts = itertools_cycle(lines)
        gsm_train = [{"question": "Tom has 3 apples and buys 4 more. How many apples does he have?", "answer": "3 + 4 = 7\n#### 7"}] * 8
        gsm_test = gsm_train[:4]
        chat_texts = itertools_cycle([tokenizer.apply_chat_template([{"role": "user", "content": l[:200]}, {"role": "assistant", "content": l[200:600]}], tokenize=False) for l in lines])

    def chunks(texts):
        buffer = []
        for text in texts:
            buffer.extend(tokenizer(text).input_ids + [tokenizer.eos_token_id])
            while len(buffer) >= cfg.lm_length:
                yield torch.tensor(buffer[: cfg.lm_length])
                buffer = buffer[cfg.lm_length :]

    def share(stream):
        for i, chunk in enumerate(stream):
            if i % world == rank:
                yield chunk

    web_chunks = chunks(web_texts)
    held = [next(web_chunks) for _ in range(held_out)]
    web_chunks = share(web_chunks)
    rng = random.Random(cfg.seed + rank)
    gsm_chunks = share(chunks(f"Question: {p['question']}\nAnswer: {p['answer']}" for p in itertools_cycle(gsm_train, random.Random(cfg.seed))))
    chat_chunks = share(chunks(chat_texts))

    def mixed():
        while True:
            u = rng.random()
            yield next(gsm_chunks) if u < cfg.gsm8k_share else next(chat_chunks) if u < cfg.gsm8k_share + cfg.instruct_share else next(web_chunks)

    subset = random.Random(0).sample(gsm_test, min(cfg.gsm8k_problems, len(gsm_test)))
    return mixed(), held, subset


def itertools_cycle(items, rng=None):
    items = list(items)
    while True:
        if rng is not None:
            rng.shuffle(items)
        yield from items


def setup_distributed(cfg):
    """(rank, world size, device). Under torchrun (WORLD_SIZE > 1) the process joins the process
    group and takes the GPU of its local rank; otherwise it is the only process, on cfg.device."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    device = torch.device(cfg.device)
    if world > 1:
        if device.type == "cuda":
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
            torch.cuda.set_device(device)
        # Long timeout: the other processes wait at a barrier while process 0 evaluates.
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(hours=3))
    return rank, world, device


def average_gradients(params, world):
    """All-reduce the gradients of params as one flat buffer and divide by world, so every process
    steps with the same mean gradient. An inf or nan on one process reaches all of them, so their
    loss scalers skip the same steps and the weights stay identical."""
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat)
    flat.div_(world)
    offset = 0
    for p in params:
        p.grad = flat[offset : offset + p.numel()].view_as(p)
        offset += p.numel()


def distill_loss(student, teacher, ids, chunk=128):
    """KL(teacher || student) over next-token distributions, mean over tokens. teacher is a frozen
    copy of the LM, or None for LoRA: the student with its adapters switched off is the original
    model. The vocabulary-wide logits are made chunk tokens at a time and recomputed in the
    backward pass, so no (tokens, vocabulary) tensor of the whole batch is ever stored."""
    with torch.no_grad():
        if teacher is None:
            with lora_disabled(student):
                t_hidden = student.model(input_ids=ids).last_hidden_state
            head = student.lm_head
        else:
            t_hidden = teacher.model(input_ids=ids).last_hidden_state
            head = teacher.lm_head
    s_hidden = student.model(input_ids=ids).last_hidden_state
    s_flat, t_flat = s_hidden.reshape(-1, s_hidden.shape[-1]), t_hidden.reshape(-1, t_hidden.shape[-1])

    def part(s_part, t_part):
        with torch.no_grad():
            t = F.log_softmax(head(t_part).float(), -1)
        s = F.log_softmax(student.lm_head(s_part).float(), -1)
        # F.kl_div(s, t, log_target=True, reduction="sum") written out: torch_xla has no autograd
        # kernel for aten::kl_div, and the fallback gives the student no gradient (silently)
        return (t.exp() * (t - s)).sum()

    total = sum(checkpoint(part, s_flat[i : i + chunk], t_flat[i : i + chunk]) for i in range(0, s_flat.shape[0], chunk))
    return total / s_flat.shape[0]


def evaluate(policy, store, goal_windows, captions, device, guidance=1.0, split="val"):
    adapter = EvalAdapter(policy)
    out = {}
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        goal = goal_eval.summary(goal_eval.run(adapter, store, goal_windows, device, guidance=guidance, execute=1, batch_size=64))
        out.update({f"goal_{k}": v for k, v in goal.items() if k in ("fact_score", "reached", "recall_ate", "recall_around", "recall_pushed", "recall_groups") or k.startswith(("fact_events_", "n_events_"))})
        if captions is not None:
            windows, ids = captions.eval_set(split)
            text = goal_eval.summary(goal_eval.run(adapter, store, windows, device, guidance=guidance, execute=1, batch_size=64, text_emb=ids))
            shuffled = goal_eval.summary(goal_eval.run(adapter, store, windows, device, guidance=guidance, execute=1, batch_size=64, text_emb=ids.roll(1, dims=0)))
            out.update({"text_fact_score": text["fact_score"], "text_reached": text["reached"], "text_shuffled_fact_score": shuffled["fact_score"]})
            out.update({f"text_{k}": v for k, v in text.items() if k.startswith(("fact_events_", "n_events_"))})
            out.update({f"text_shuffled_{k}": v for k, v in shuffled.items() if k.startswith("fact_events_")})
    return out


def language(policy, tokenizer, held, probe, gsm, device, with_gsm, max_tokens=320):
    """probe: {name: (prompt, options, answer) items}, scored by the LM's own head."""
    lm = policy.lm
    was_training = lm.training
    lm.eval()
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        out = {"perplexity": round(perplexity(lm, held, device), 3)}
        out.update({name: round(mc_accuracy(lm, tokenizer, items, device), 3) for name, items in probe.items()})
        if with_gsm:
            out["gsm8k"] = round(gsm8k_accuracy(lm, tokenizer, gsm, device, max_new_tokens=max_tokens), 3)
    lm.train(was_training)
    return out


def run():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = parse_config(LMConfig)
    rank, world, device = setup_distributed(cfg)
    lead = rank == 0
    torch.manual_seed(cfg.seed)  # the same on every process: identical initial weights
    run_dir = Path("runs") / cfg.name
    if lead:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps({**asdict(cfg), "world": world}, indent=1))
    generator = torch.Generator().manual_seed(cfg.seed + rank)  # each process its own batches

    t_start = time.time()

    def stage(name):
        if lead:
            print(f"[{(time.time() - t_start) / 60:.1f} min] {name}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.padding_side = "right"
    half = bool(cfg.lora_rank) and cfg.half_base and device.type == "cuda"
    lm = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.float16 if half else torch.float32)
    teacher = None  # LoRA: the student with its adapters off is the original model
    if not cfg.lora_rank:
        teacher = copy.deepcopy(lm).half().to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    policy = LMPolicy(lm, lora_rank=cfg.lora_rank, freeze_embeddings=cfg.freeze_embeddings).to(device)
    if cfg.checkpointing:
        policy.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        policy.lm.config.use_cache = False
    stage("model and teacher loaded")

    data_cfg = Config(grid=32, data="play", play_root=cfg.play_root, play_cache=cfg.play_cache, val_windows=cfg.eval_windows, test_windows=cfg.test_windows)
    store, windows, _, val_windows, _, test_windows = load_data(data_cfg, "cpu")
    stage("play data loaded")
    captions = Captions(cfg.captions, store, tokenizer, cfg.eval_windows, cfg.test_windows) if cfg.captions else None
    stage("captions resolved")
    text_stream, held_text, gsm = general_text(cfg, tokenizer, rank=rank, world=world)
    stage("general text streams open")
    probe = {"world_probe": probe_items(), **(benchmark_items(cfg.bench_items) if cfg.bench_items else {})} if lead else None
    stage("benchmarks loaded")

    groups = [{"params": policy.new_parameters(), "lr": cfg.lr_new}, {"params": policy.lm_parameters(), "lr": cfg.lr_lm}]
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    base_lrs = [g["lr"] for g in groups]
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n_new = sum(p.numel() for p in policy.new_parameters())
    n_lm = sum(p.numel() for p in policy.lm_parameters())
    stage(f"trainable: {n_new / 1e6:.2f}M new, {n_lm / 1e6:.1f}M in the LM; {world} process(es), {cfg.goal_batch * world} goal and {cfg.text_batch * world} text examples per step")
    params = [p for g in groups for p in g["params"]]

    history = open(run_dir / "history.jsonl", "a") if lead else None
    baseline = language(policy, tokenizer, held_text, probe, gsm, device, with_gsm=True, max_tokens=cfg.gsm8k_max_tokens) if lead else None
    stage(f"language before training: {baseline}")
    step = 0
    fresh_optimizer = True
    if cfg.init:
        saved = torch.load(cfg.init, map_location=device)
        weights = saved.get("weights", saved)
        missing = [n for n, p in policy.named_parameters() if p.requires_grad and n not in weights]
        assert not missing, f"init lacks trainable weights: {missing[:5]}"
        policy.load_state_dict({k: v.float() for k, v in weights.items()}, strict=False)
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            fresh_optimizer = False
        step = saved.get("step", 0)
        stage(f"started from {cfg.init} (step {step}, optimizer {'restored' if 'optimizer' in saved else 'fresh'})")
    if world > 1:
        for p in params:  # identical already (same seed, same init); made certain
            dist.broadcast(p.data, 0)
    policy.train()  # from_pretrained returns eval mode, where gradient checkpointing is off
    if lead:
        history.write(json.dumps({"step": 0, **baseline}) + "\n")
        history.flush()

    def save_latest(step, fraction):
        """Trainable weights, optimizer and scaler state, step and schedule position: a resume point."""
        trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
        weights = {k: v for k, v in policy.state_dict().items() if k in trainable}
        torch.save({"weights": weights, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "step": step, "fraction": fraction}, run_dir / "policy_latest.pt.tmp")
        (run_dir / "policy_latest.pt.tmp").replace(run_dir / "policy_latest.pt")
        stage(f"saved policy_latest.pt at step {step} (schedule fraction {fraction:.3f})")

    start = time.time()
    eval_seconds, next_eval, n_evals, best = 0.0, cfg.eval_every_minutes * 60, 0, -1.0
    fraction = cfg.schedule_offset
    first_step = step
    while not cfg.eval_only:
        elapsed = time.time() - start - eval_seconds
        stop = step >= cfg.steps or (cfg.minutes and elapsed >= cfg.minutes * 60) or (cfg.wall_minutes and time.time() - t_start >= cfg.wall_minutes * 60)
        if world > 1:  # process 0's clock for everyone: same learning rate, evaluations and stop
            clock = torch.tensor([elapsed, float(bool(stop))], dtype=torch.float64, device=device)
            dist.broadcast(clock, 0)
            elapsed, stop = clock[0].item(), bool(clock[1].item())
        if stop:
            break
        progress = max((step - first_step) / max(cfg.steps - first_step, 1), elapsed / (cfg.minutes * 60) if cfg.minutes else 0.0)  # whichever budget runs out first
        fraction = cfg.schedule_offset + (1 - cfg.schedule_offset) * progress
        warm = step - first_step + 1 if fresh_optimizer else cfg.warmup_steps  # a fresh optimizer warms up again, also when resuming
        factor = min(1.0, warm / cfg.warmup_steps) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * fraction)))
        for g, lr in zip(optimizer.param_groups, base_lrs):
            g["lr"] = lr * factor

        # One backward per loss: each loss's activations are freed before the next one's forward
        # pass, so peak memory is the largest loss's rather than their sum. Same gradients.
        optimizer.zero_grad(set_to_none=True)
        amp = torch.autocast(device.type, dtype=torch.float16, enabled=use_amp)
        with amp:
            obs, goal_frames, actions, weights, _ = sample_batch(store, windows, cfg.goal_batch, generator, "cpu", True)
            obs, goal_frames, actions, weights = (t.to(device) for t in (obs, goal_frames, actions, weights))
            null = (torch.rand(cfg.goal_batch, generator=generator) < cfg.null_prob).to(device)
            goal_loss = policy.policy_loss(obs, actions, weights, goal=policy.goal_encoder(goal_frames), null=null)
        scaler.scale(goal_loss).backward()
        record = {"goal_loss": round(goal_loss.item(), 4)}
        if captions is not None:
            with amp:
                t_obs, t_actions, t_weights, ids, mask = captions.sample(store, cfg.text_batch, generator, "cpu")
                t_obs, t_actions, t_weights, ids, mask = (t.to(device) for t in (t_obs, t_actions, t_weights, ids, mask))
                text_loss = policy.policy_loss(t_obs, t_actions, t_weights, text_ids=ids, text_mask=mask)
            scaler.scale(text_loss).backward()
            record["text_loss"] = round(text_loss.item(), 4)
        if cfg.distill_weight > 0:
            with amp:
                chunk = torch.stack([next(text_stream) for _ in range(cfg.lm_batch)]).to(device)
                kl = distill_loss(policy.lm, teacher, chunk)
            scaler.scale(cfg.distill_weight * kl).backward()
            record["distill_kl"] = round(kl.item(), 5)
        if world > 1:
            average_gradients(params, world)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        examples = {"goal_examples": (step - first_step) * cfg.goal_batch * world, "text_examples": (step - first_step) * cfg.text_batch * world}  # this session's
        if lead and (step % cfg.log_every == 0 or step in (1, 10)):
            memory = {"gpu_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 2)} if device.type == "cuda" else {}
            print(json.dumps({"step": step, "minutes": round(elapsed / 60, 1), "seconds_per_step": round(elapsed / max(step - first_step, 1), 2), **examples, **memory, **record}), flush=True)

        if elapsed >= next_eval or step == first_step + cfg.first_eval_step:
            t0 = time.time()
            n_evals += 1
            if lead:
                stage(f"evaluating at step {step}")
                policy.eval()
                scores = {**evaluate(policy, store, val_windows, captions, device), **language(policy, tokenizer, held_text, probe, gsm, device, with_gsm=n_evals % cfg.gsm8k_every == 0, max_tokens=cfg.gsm8k_max_tokens)}
                policy.train()
                row = {"step": step, "minutes": round(elapsed / 60, 1), **examples, **scores}
                print(f"val: {row}", flush=True)
                history.write(json.dumps(row) + "\n")
                history.flush()
                selection = scores["goal_fact_score"] + scores.get("text_fact_score", scores["goal_fact_score"])
                trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
                if selection > best:
                    best = selection
                    torch.save({k: v.half() for k, v in policy.state_dict().items() if k in trainable}, run_dir / "policy_best.pt")
                if cfg.save_latest:
                    save_latest(step, fraction)
            if world > 1:
                dist.barrier()  # the others wait for process 0's evaluation
            eval_seconds += time.time() - t0
            next_eval = elapsed + cfg.eval_every_minutes * 60

    if lead and cfg.save_latest and not cfg.eval_only:
        save_latest(step, fraction)  # where the next session resumes, however long since the last evaluation
    if world > 1:
        # Every process prints this; they must agree (same mean gradients, same steps).
        print(f"process {rank}: weight checksum {sum(p.detach().double().sum().item() for p in params):.10e}", flush=True)
        dist.destroy_process_group()
    if not lead or not cfg.final_test:
        return  # the final evaluation runs on process 0 alone, and only in a run's last session
    stage(f"training done at step {step}; test evaluation")
    policy.eval()
    summary = {"steps": step, "language_before": baseline}
    # Each part is written as soon as it exists, so a session cut short keeps what finished.
    summary["language_after"] = language(policy, tokenizer, held_text, probe, gsm, device, with_gsm=True, max_tokens=cfg.gsm8k_max_tokens)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    stage(f"language after training: {summary['language_after']}")
    for guidance in (2.0, 1.0):
        summary[f"test_w{guidance:g}"] = evaluate(policy, store, test_windows, captions, device, guidance, split="test")
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        stage(f"test at guidance {guidance:g}: {summary[f'test_w{guidance:g}']}")
    print(json.dumps(summary, indent=1), flush=True)


def main():
    run()
    # Exit hard once done: on Kaggle the interpreter did not exit after a finished run (a test
    # kernel and a two-GPU smoke both sat for hours after their last line), most likely held by
    # non-daemon threads of the streamed text datasets. Everything is written and flushed by now.
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
