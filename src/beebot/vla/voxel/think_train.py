"""Train the think-then-act voxel policy (voxel/think.py) on ThinkData examples (voxel/think_data.py),
distilling the frozen original LM on general text so the policy keeps its language.

    python -m beebot.vla.voxel.think_train --name vox-think --stores S1,S2 --haiku_dirs H1 --device cuda
    torchrun --standalone --nproc_per_node 2 -m beebot.vla.voxel.think_train ...      # two GPUs
    python -m beebot.vla.voxel.think_train --name think-smoke --data fake --device mps --steps 40 \\
        --distill_weight 0 --val_every_minutes 1                                       # a Mac smoke run

A step: `accumulate` micro-batches of `batch` examples on every process (text and action losses,
think.py; the step's examples sorted by length before they are split into micro-batches, so each
micro-batch pads little), one distillation batch of general text (KL to the frozen original; for
LoRA the LM with its adapters off), gradients averaged over processes, clipped to norm 1, AdamW
with one learning rate for the LM's weights and another for the new ones (scene and slot
embeddings, positions, readout), linear warmup then cosine down to min_lr_ratio over `steps`.

No rollouts and no scorecards here (the Mac scores the checkpoints): every val_every_minutes the
held-out losses on ThinkData.validation, every save_every_minutes a step-named checkpoint
runs/<name>/policy_step{N}.pt (trainable weights in bf16 and what think.load_policy needs to rebuild
the policy) and runs/<name>/resume.pt (fp32 weights, optimizer and scaler state, step) for --resume.
Progress lines are JSON on stdout and in runs/<name>/history.jsonl.

Precision (--precision auto): CUDA with bf16 (compute capability 8+, e.g. GH200) bf16 autocast;
older CUDA (T4) fp16 autocast with a gradient scaler; MPS and CPU fp32. The LM's weights and the
optimizer state are fp32 unless --weights_dtype bf16 (the LM stored in bf16: for LoRA or frozen
parts, or to fit a large LM; full fine-tuning on bf16 weights loses small updates). The new weights
are always fp32.

Vision-language LMs (Qwen3.5): the examples' images go through the vision encoder (frozen unless
--train_vision; its projector trains unless --train_projector false). For hybrid LMs the log says
which linear-attention kernels are in use (fla and causal_conv1d when installed, else transformers'
torch fallback).

On several GPUs (torchrun) every process trains on its own examples; process 0 keeps the clock
(stop, validation and saving) and broadcasts it, validates and saves while the others wait.
"""

import copy
import dataclasses
import json
import math
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from beebot.train import parse_config
from beebot.vla import tpu
from beebot.vla.lm_train import distill_loss, general_text, setup_distributed
from beebot.vla.voxel.think import ThinkVoxelPolicy, checkpoint_meta, linear_attention_kernels, load_lm, trainable_state, use_matmul_linear_attention
from beebot.vla.voxel.world import N_CHANNELS


@dataclass
class ThinkConfig:
    name: str = "vox-think"
    model: str = "Qwen/Qwen3.5-4B"  # from the local Hugging Face cache: a vision-language LM (Qwen3.5) or a text LM (Qwen3)
    data: str = "think"  # "think": think_data.ThinkData; "fake": think_fake.FakeThinkData (smoke runs, throughput)
    stores: str = ""  # comma-separated demonstration stores for ThinkData
    haiku_dirs: str = ""  # comma-separated Haiku label directories for ThinkData
    data_kwargs: str = "{}"  # JSON: further ThinkData (or FakeThinkData) keyword arguments
    views: str = "scene"  # ThinkData's views: "scene" (views.scene_views, randomized at DEFAULT_STRENGTH) or "stand_in" (one isometric image)
    val_seed: int = 0
    val_examples: int = 256
    channels: int = N_CHANNELS
    lora_rank: int = 0  # 0: full fine-tuning
    train_embeddings: bool = False  # also train the token embeddings (tied to the output head in Qwen3 and Qwen3.5-4B)
    train_vision: bool = False  # vision-language LMs: train the vision encoder
    train_projector: bool = True  # vision-language LMs: train the vision projector (merger)
    weights_dtype: str = "fp32"  # the LM's weights: fp32 or bf16
    init: str = ""  # weights to start from: a think checkpoint or any dict of policy weights (matching names load, the rest stay)
    resume: str = ""  # a resume.pt: weights, optimizer, scaler and step
    batch: int = 16  # examples per micro-batch per process
    accumulate: int = 1  # micro-batches per optimizer step
    bucket: bool = True  # sort each step's examples by length before splitting them into micro-batches
    text_weight: float = 1.0
    action_weight: float = 1.0
    ce_chunk: int = 512  # reply positions per chunk of vocabulary-wide logits
    checkpointing: bool = False  # gradient checkpointing of the LM's layers
    precision: str = "auto"  # auto, fp32, bf16, fp16
    lr_lm: float = 2e-5
    lr_new: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    steps: int = 20000  # the schedule's length; a session may stop earlier (minutes, wall_minutes) and be resumed
    minutes: float = 0.0  # > 0: stop after this much training time (validation and saving excluded)
    wall_minutes: float = 0.0  # > 0: stop this long after the process started, everything included (Kaggle's 12 h)
    distill_weight: float = 1.0
    general: str = "hf"  # "hf" (FineWeb-Edu, GSM8K, SmolTalk from the Hub) or a local text file
    lm_batch: int = 2
    lm_length: int = 256
    log_every: int = 20
    val_every_minutes: float = 30.0
    save_every_minutes: float = 60.0
    save_resume: bool = True
    prefetch: int = 4  # steps of examples made ahead by a background thread; 0: in the training loop
    device: str = "cuda"  # cuda, mps, cpu, or xla (a TPU through torch_xla)
    chips: int = 1  # xla: processes, one per TPU chip, data-parallel (torch_xla.launch)
    widths: str = "384,512"  # xla: the row widths batches are padded to (fixed shapes; think.ThinkVoxelPolicy.pack)
    seed: int = 0


def precision_of(name, device):
    """(precision name, autocast dtype or None) for --precision on this device."""
    if name == "auto":
        name = ("bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16") if device.type == "cuda" else "bf16" if device.type == "xla" else "fp32"
    return name, {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def make_data(cfg, tokenizer):
    split = lambda s: [p for p in s.split(",") if p]  # noqa: E731
    kwargs = json.loads(cfg.data_kwargs)
    if cfg.data == "fake":
        from beebot.vla.voxel.think_fake import FakeThinkData

        source = FakeThinkData(tokenizer, cfg.seed, channels=cfg.channels, **kwargs)
    else:
        from beebot.vla.voxel.think_data import ThinkData

        from beebot.vla.voxel.think_data import stand_in_views
        from beebot.vla.voxel.views import ViewSpec, scene_views, side_azimuth

        spec = ViewSpec()
        def scene(obs, rng, side=None):  # side: the board side the camera must look from (frame labels)
            return scene_views(obs, rng, spec if side is None else dataclasses.replace(spec, azimuth=side_azimuth(side, rng)))

        views = {"scene": scene, "stand_in": stand_in_views}[cfg.views]
        source = ThinkData(split(cfg.stores), split(cfg.haiku_dirs), tokenizer, cfg.seed, views=views, **kwargs)
    return source


class Prefetch:
    """Each step's examples (count of them) as prepared rows, made by a background thread depth
    steps ahead (depth 0: made on demand). Tokenizing and image processing happen there too, and
    finish(rows) when given (XLA: the step's micro-batches packed in fixed shapes)."""

    def __init__(self, data, prepare, count, rng, depth, finish=None):
        self.data, self.prepare, self.count, self.rng, self.finish = data, prepare, count, rng, finish
        self.queue = queue.Queue(depth) if depth else None
        if depth:
            threading.Thread(target=self.fill, daemon=True).start()

    def make(self):
        rows = [self.prepare(self.data.sample(self.rng)) for _ in range(self.count)]
        return self.finish(rows) if self.finish else rows

    def fill(self):
        try:
            while True:
                self.queue.put(self.make())
        except Exception as error:  # surfaces in the training loop
            self.queue.put(error)

    def next(self):
        item = self.queue.get() if self.queue is not None else self.make()
        if isinstance(item, Exception):
            raise item
        return item


def average_gradients(params, world, bucket=1 << 25):
    """All-reduce the gradients of params in flat buckets of about `bucket` elements and divide by
    world; a parameter without a gradient counts as zero, so every process steps the same."""
    group, size = [], 0
    for i, p in enumerate(params):
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        group.append(p)
        size += p.numel()
        if size >= bucket or i == len(params) - 1:
            flat = torch.cat([q.grad.reshape(-1) for q in group])
            dist.all_reduce(flat)
            flat.div_(world)
            offset = 0
            for q in group:
                q.grad.copy_(flat[offset : offset + q.numel()].view_as(q))
                offset += q.numel()
            group, size = [], 0


def memory_gb(device):
    """Peak allocated CUDA memory, the MPS driver's current allocation, or the TPU chip's HBM in use, in GB."""
    used = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else torch.mps.driver_allocated_memory() / 1e9 if device.type == "mps" else tpu.xla_memory_gb(device) if device.type == "xla" else 0
    return round(used, 2)


def lr_factor(step, cfg):
    warm = min(1.0, (step + 1) / max(cfg.warmup_steps, 1))
    progress = min(step / max(cfg.steps, 1), 1.0)
    return warm * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def run(cfg=None, index=0):
    """One training session (cfg: from the command line when None; index: the process's TPU chip under --chips)."""
    t_start = time.time()
    cfg = cfg or parse_config(ThinkConfig)
    xla = tpu.is_xla(cfg.device)
    rank, world, device = tpu.xla_setup(index) if xla else setup_distributed(cfg)
    lead = rank == 0
    if xla:
        use_matmul_linear_attention()
    torch.manual_seed(cfg.seed)  # the same on every process: identical new weights
    run_dir = Path("runs") / cfg.name
    precision, amp_dtype = precision_of(cfg.precision, device)

    def stage(text):
        if lead:
            print(f"[{(time.time() - t_start) / 60:.1f} min] {text}", flush=True)

    if lead:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps({**asdict(cfg), "world": world, "precision_used": precision}, indent=1))
    lm, tokenizer, image_processor = load_lm(cfg.model, {"fp32": torch.float32, "bf16": torch.bfloat16}[cfg.weights_dtype])
    teacher = None  # LoRA: the student with its adapters off is the original model
    if cfg.distill_weight > 0 and not cfg.lora_rank:
        teacher = copy.deepcopy(lm)
        if hasattr(teacher.model, "visual"):
            del teacher.model.visual  # distillation is on text alone
        teacher = teacher.to(device, amp_dtype or lm.dtype).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    policy = ThinkVoxelPolicy(lm, tokenizer, image_processor, lora_rank=cfg.lora_rank, freeze_embeddings=not cfg.train_embeddings, train_vision=cfg.train_vision, train_projector=cfg.train_projector, channels=cfg.channels, text_weight=cfg.text_weight, action_weight=cfg.action_weight, ce_chunk=cfg.ce_chunk)
    if cfg.init:
        saved = torch.load(cfg.init, map_location="cpu")
        weights = saved.get("weights", saved)
        own = dict(policy.named_parameters())
        matching = {k: v.to(own[k].dtype) for k, v in weights.items() if k in own and own[k].shape == v.shape}
        assert matching, f"init: no matching weights in {cfg.init}"
        policy.load_state_dict(matching, strict=False)
        stage(f"init from {cfg.init}: {len(matching)} of {len(weights)} tensors")
    policy = policy.to(device)
    if cfg.checkpointing:
        if xla:  # torch_xla's checkpoint, so XLA does not fold the recomputation back into the forward pass
            import transformers.modeling_utils

            transformers.modeling_utils.checkpoint = tpu.xla_checkpoint_function()
        policy.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": xla})
        policy.lm.config.use_cache = False
    groups = [{"params": policy.lm_parameters(), "lr": cfg.lr_lm}, {"params": policy.new_parameters(), "lr": cfg.lr_new}]
    base_lrs = [g["lr"] for g in groups]
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    assert not (xla and amp_dtype == torch.float16), "xla: bf16 or fp32"
    scaler = torch.amp.GradScaler("cpu" if xla else device.type, enabled=amp_dtype == torch.float16)
    params = [p for g in groups for p in g["params"]]
    step = 0
    if cfg.resume:
        saved = torch.load(cfg.resume, map_location="cpu" if xla else device)
        policy.load_state_dict(saved["weights"], strict=False)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        step = saved["step"]
        stage(f"resumed from {cfg.resume} at step {step}")
    if world > 1 and not xla:  # on XLA every process starts from the same file and seed
        for p in params:
            dist.broadcast(p.data, 0)
    stage(f"trainable: {sum(p.numel() for p in groups[1]['params']) / 1e6:.2f}M new, {sum(p.numel() for p in groups[0]['params']) / 1e6:.1f}M in the LM; precision {precision}, LM weights {cfg.weights_dtype}; {world} process(es) x {cfg.batch} x {cfg.accumulate} examples per step")
    kernels = linear_attention_kernels(lm)
    if kernels:
        stage(f"linear-attention kernels: {json.dumps(kernels)}")

    data = make_data(cfg, tokenizer)
    finish = None
    if xla:  # each step's micro-batches packed in fixed shapes on the CPU, in the prefetch thread
        widths = [int(w) for w in cfg.widths.split(",")]

        def finish(rows):
            rows = sorted(rows, key=lambda p: p.length) if cfg.bucket else rows
            packs = [policy.pack(rows[k * cfg.batch : (k + 1) * cfg.batch], widths) for k in range(cfg.accumulate)]
            totals = torch.tensor([float(sum(p.counts["text_tokens"] for p in packs)), float(sum(p.counts["robots"] for p in packs))])
            for p in packs:  # the step's totals (a tensor, so the graph does not change with them)
                p.totals = totals
            return packs

    batches = Prefetch(data, policy.prepare_example, cfg.batch * cfg.accumulate, np.random.default_rng((cfg.seed, rank, step)), cfg.prefetch, finish)
    val_set = sorted((policy.prepare_example(e) for e in data.validation(cfg.val_examples, cfg.val_seed)), key=lambda p: p.length) if lead else None
    if lead and xla:
        val_set = [policy.pack(val_set[i : i + cfg.batch], widths) for i in range(0, len(val_set), cfg.batch)]
    text_stream = None
    if cfg.distill_weight > 0:
        text_cfg = SimpleNamespace(general=cfg.general, lm_length=cfg.lm_length, gsm8k_share=0.1, instruct_share=0.3, seed=cfg.seed, gsm8k_problems=0)
        text_stream, _, _ = general_text(text_cfg, tokenizer, held_out=1, rank=rank, world=world)
    stage("data ready")

    def autocast():
        return torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None)

    history = open(run_dir / "history.jsonl", "a") if lead else None

    def log(row):
        print(json.dumps(row), flush=True)
        history.write(json.dumps(row) + "\n")
        history.flush()

    if lead and kernels:
        log({"step": step, "linear_attention_kernels": kernels})

    @torch.no_grad()
    def validate():
        """Held-out losses over the whole validation list: text per reply token, action per robot voxel."""
        policy.eval()
        text = action = tokens = robots = 0.0
        groups = val_set if xla else [val_set[i : i + cfg.batch] for i in range(0, len(val_set), cfg.batch)]  # xla: packed already
        for group in groups:
            with autocast():
                out = policy(group.to(device) if xla else group)
            text += out["text"].item() * out["text_tokens"]
            action += out["action"].item() * out["robots"]
            tokens += out["text_tokens"]
            robots += out["robots"]
        policy.train()
        text, action = text / max(tokens, 1), action / max(robots, 1)
        return {"val_text": round(text, 4), "val_action": round(action, 4), "val_loss": round(cfg.text_weight * text + cfg.action_weight * action, 4)}

    def save(step):
        meta = {**checkpoint_meta(policy, cfg.model), "step": step}
        host = tpu.to_cpu if xla else (lambda value: value)
        torch.save({"weights": host(trainable_state(policy, torch.bfloat16)), "meta": meta}, run_dir / f"policy_step{step}.pt")
        if cfg.save_resume:
            torch.save({"weights": host(trainable_state(policy)), "optimizer": host(optimizer.state_dict()), "scaler": scaler.state_dict(), "step": step, "meta": meta}, run_dir / "resume.pt.tmp")
            (run_dir / "resume.pt.tmp").replace(run_dir / "resume.pt")
        stage(f"saved policy_step{step}.pt")

    policy.train()
    first_step = step
    t0 = time.time()
    paused = 0.0  # validation and saving time, kept out of the training clock
    next_val, next_save = cfg.val_every_minutes * 60, cfg.save_every_minutes * 60
    window = {"loss": [], "text": [], "action": [], "distill_kl": []}
    counts = {"tokens": 0, "examples": 0, "wait": 0.0}
    t_window = time.time()
    while True:
        elapsed = time.time() - t0 - paused
        flags = [step >= cfg.steps or bool(cfg.minutes and elapsed >= cfg.minutes * 60) or bool(cfg.wall_minutes and time.time() - t_start >= cfg.wall_minutes * 60), elapsed >= next_val, elapsed >= next_save]
        if world > 1 and xla:  # the processes agree at log steps only (a host sync); in between only the step count stops
            flags = tpu.xla_agree(flags, device) if step % cfg.log_every == 0 else [step >= cfg.steps, False, False]
        elif world > 1:  # process 0's clock for everyone
            clock = torch.tensor([float(f) for f in flags], device=device)
            dist.broadcast(clock, 0)
            flags = [bool(f) for f in clock.tolist()]
        stop, do_val, do_save = flags
        if stop:
            break
        if do_val or do_save:
            t_pause = time.time()
            if lead and do_val:
                log({"step": step, "minutes": round(elapsed / 60, 2), **validate()})
            if lead and do_save:
                save(step)
            if world > 1 and xla:
                tpu.xla_barrier("pause")
            elif world > 1:
                dist.barrier()
            next_val = elapsed + cfg.val_every_minutes * 60 if do_val else next_val
            next_save = elapsed + cfg.save_every_minutes * 60 if do_save else next_save
            paused += time.time() - t_pause
        for g, lr in zip(optimizer.param_groups, base_lrs):
            g["lr"] = lr * lr_factor(step, cfg)
        optimizer.zero_grad(set_to_none=True)
        t_wait = time.time()
        rows = batches.next()
        counts["wait"] += time.time() - t_wait
        if cfg.bucket and not xla:  # xla: sorted and packed in the prefetch thread
            rows = sorted(rows, key=lambda p: p.length)
        # the step's loss is the mean over all its reply tokens and robot voxels, whichever
        # micro-batch they land in (a mean of micro-batch means overweights the short replies
        # that bucketing groups together, about 3x per token); xla batches carry their totals
        if not xla:
            text_total = sum(p.reply_len for p in rows)
            robot_total = max(1.0, float(sum((p.obs[0] > 0.5).sum().item() for p in rows if p.act)))
        step_losses = {}
        for k in range(cfg.accumulate):
            with autocast():
                out = policy(rows[k].to(device)) if xla else policy(rows[k * cfg.batch : (k + 1) * cfg.batch], text_total=text_total, robot_total=robot_total)
            scaler.scale(out["loss"]).backward()
            for key in ("loss", "text", "action"):
                part = out[key].detach() if xla else out[key].item()  # xla: read at log time
                step_losses[key] = step_losses[key] + part if key in step_losses else part
            counts["tokens"] += out["tokens"]
            counts["examples"] += out["examples"]
        for key, value in step_losses.items():
            window[key].append(value)
        if text_stream is not None:
            chunk = torch.stack([next(text_stream) for _ in range(cfg.lm_batch)]).to(device)
            with autocast():
                kl = distill_loss(policy.lm, teacher, chunk)
            scaler.scale(cfg.distill_weight * kl).backward()
            window["distill_kl"].append(kl.detach() if xla else kl.item())
        if world > 1 and xla:
            tpu.xla_average_gradients(params, world)
        elif world > 1:
            average_gradients(params, world)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(optimizer)
        scaler.update()
        if xla:
            tpu.xla_mark_step()
        step += 1
        if lead and (step % cfg.log_every == 0 or step - first_step in (1, 10)):
            values = {k: [float(x) for x in v] for k, v in window.items() if v}  # xla: waits for the step to finish
            seconds = time.time() - t_window
            row = {"step": step, "minutes": round((time.time() - t0 - paused) / 60, 2), **{k: round(float(np.mean(v)), 5) for k, v in values.items()}}
            row.update({"lr_lm": optimizer.param_groups[0]["lr"], "tokens_per_s": round(counts["tokens"] * world / seconds, 1), "examples_per_s": round(counts["examples"] * world / seconds, 2), "data_wait": round(counts["wait"] / seconds, 3), "gpu_gb": memory_gb(device)})
            if xla:
                row.update({"step_s": round(seconds / len(values["loss"]), 3), **tpu.xla_report()})
            log(row)
            window = {k: [] for k in window}
            counts = {"tokens": 0, "examples": 0, "wait": 0.0}
            t_window = time.time()
    if lead:
        log({"step": step, "minutes": round((time.time() - t0 - paused) / 60, 2), **validate()})
        save(step)
    if world > 1 and xla:
        tpu.xla_barrier("done")
    elif world > 1:
        dist.barrier()
        print(f"process {rank}: weight checksum {sum(p.detach().double().sum().item() for p in params):.10e}", flush=True)
        dist.destroy_process_group()
    stage(f"done at step {step}")


def finish_process(*_):
    """Exit hard once done, as lm_train does: the streamed text datasets' threads can hold the
    interpreter open (Kaggle kernels sat for hours after their last line)."""
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def xla_worker(index, cfg):
    """One TPU chip's process (--device xla --chips N)."""
    run(cfg, index)
    finish_process()


def main():
    cfg = parse_config(ThinkConfig)
    if tpu.is_xla(cfg.device) and cfg.chips > 1:
        tpu.xla_launch(xla_worker, (cfg,))
    else:
        run(cfg)
    finish_process()


if __name__ == "__main__":
    main()
