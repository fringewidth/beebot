"""Train the policy.

    uv run train-vla --name vla-probe-g32 --grid 32 --minutes 45
    uv run train-vla --name vla-main --data play --captions data/vla/captions/play_set.pt

Data: --data scripted (scripted play, the grid-size probe and the scripted ablation) or
--data play (LLM play from the play workspaces, split by program into train, val and test).
With --captions (a caption set from vla-caption-set over the same play), each batch is half
goal-conditioned (z from the window's frames) and half caption-conditioned (z from the text
head), plus a sigmoid contrastive loss between the two z of each caption-half window.

Every Config field is a flag. Writes runs/<name>/: config.json, log.jsonl, val_history.jsonl,
model_best.pt (best closed-loop val), model.pt (last), summary.json (test numbers for the best
checkpoint, next to the do-nothing and replay baselines). A file named STOP in the run directory
ends training early, as in the other trainers.
"""

import dataclasses
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from beebot.train import awake_seconds, parse_config
from beebot.vla import goal_eval
from beebot.vla.caption import TIERS
from beebot.vla.caption_set import resolve
from beebot.vla.data import COMMAND_CHANNELS, batch_from, generate_scripted, make_windows, sample_batch, world_at
from beebot.vla.model import Policy
from beebot.vla.swarm import RelativePolicy
from beebot.vla.play_data import PLAY_ROOT, load_play, load_play_cache, run_dirs
from beebot.vla.tpu import cpu_view, devices, step_done, to_device
from beebot.vla.world import N_CHANNELS, PHYSICS, observe


@dataclass
class Config:
    name: str = "vla-probe"
    grid: int = 16
    device: str = "mps"
    seed: int = 0
    minutes: float = 45.0
    # If set, training stops after this many steps and the learning-rate schedule runs over steps,
    # not minutes, so runs sharing the GPU (fewer steps per minute) still compare like for like.
    max_steps: int = 0
    wall_minutes: float = 0.0  # > 0: also stop training after this much wall time, evaluation included (room for the final test inside a capped session)

    goal_tokens: int = 1
    goal_pool: str = "mean"
    width: int = 192
    expert_width: int = 128
    layers: int = 4

    data: str = "scripted"  # "scripted" or "play"
    play_root: str = str(PLAY_ROOT)  # data = "play": runs under this root at this grid size
    play_cutoff: float = 0.0  # Unix time; > 0 ignores play runs recorded after it (frozen evaluations)
    play_cache: str = ""  # a directory written by play_data.cache_play: the store memory-mapped from disk
    captions: str = ""  # caption set path; empty means goal-conditioned only
    requested: str = ""  # requested-events labels of the held-out captions (scripts/vla_requested_events.py): text scored, and checkpoints chosen, on what each caption asks for
    text_fraction: float = 0.5
    contrastive_weight: float = 0.1

    episodes: int = 2500  # scripted only
    data_noise: float = 0.0  # scripted only: share of squares whose executed action is randomized (training data only)
    val_episodes: int = 300  # scripted only
    test_episodes: int = 600  # scripted only
    val_windows: int = 128
    text_val_windows: int = 512  # held-out caption windows scored per val (each costs two closed-loop rollouts)
    test_windows: int = 512

    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 300
    lr_floor: float = 0.1
    max_grad_norm: float = 1.0
    null_prob: float = 0.1
    # Diagnostic: the policy also sees the scripted player's commands at the current step (perfect
    # knowledge of intent), to tell an action-generation limit from goal ambiguity. Scripted only.
    command_oracle: bool = False
    insulate: bool = False  # knowledge insulation: backbone action readout, stop-gradient into the expert
    world_model: bool = False  # objective "masked" only: also fill in each square's future state (empty, robot, cheese)
    view: str = ""  # "global" or "swarm": the relative-position policy in vla/swarm.py (readout, one step; use execute 1)
    crop: int = 9  # view "swarm": each robot's local view, crop x crop cells
    execute: int = 4  # chunk steps run per decision in evaluation
    text_mode: str = "add"  # relative-position policies: "add" (one vector added at the input) or "cross" (text tokens read in every block)
    text_tokens: int = 8  # text tokens for text_mode "cross"
    distill_weight: float = 0.0  # > 0: text examples also copy the goal-conditioned policy's moves (same window, goal frame as z)
    objective: str = "flow"  # "flow" (flow matching), "masked" (discrete actions, masked diffusion), "readout" (no expert: one-shot per-square categoricals from the backbone) or "mse" (an ablation)
    hold_prob: float = 0.0  # share of goal-half samples that start at the goal and are labelled "stay"
    augment: bool = True

    eval_every_minutes: float = 15.0
    log_every: int = 50
    gap_every: int = 500


def load_or_generate(grid, episodes, seed, device, noise=0.0, commands=False):
    suffix = (f"_noise{noise:g}" if noise > 0 else "") + ("_cmd" if commands else "")
    path = Path("data/vla/scripted") / PHYSICS / f"g{grid}_n{episodes}_s{seed}{suffix}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    store = generate_scripted(grid, episodes, seed, device, noise=noise)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, path)
    return store


def program_split(program_id):
    """'val', 'test' or 'train', fixed per program: about 10%, 10% and 80%."""
    bucket = int(hashlib.sha256(program_id.encode()).hexdigest(), 16) % 10
    return "val" if bucket == 0 else "test" if bucket == 1 else "train"


def split_windows(store, windows):
    """{split: windows whose episode's program is in that split}."""
    split_of = torch.tensor([("train", "val", "test").index(program_split(p)) for p, _, _ in store["source"]])
    episode_split = split_of[windows[:, 0]]
    return {name: windows[episode_split == i] for i, name in enumerate(("train", "val", "test"))}


@dataclass
class HeldOutCaptions:
    windows: torch.Tensor  # (N, 3) rows of the store
    emb: torch.Tensor  # (N, D) caption embeddings
    tier: torch.Tensor  # (N,) index into caption.TIERS
    texts: list
    requested: list = None  # per window, the atoms its caption asks for (Config.requested), or None


def held_out_captions(caption_set, windows, variant_split, split, limit):
    """One caption per held-out window of a split (1 val, 2 test), at most `limit` windows drawn
    with a fixed seed (the same sample for every run on this caption set). The caption types are
    taken in turn across windows (short, medium, detailed, casual, ...), so one closed-loop run
    also scores each type on its own."""
    identity = (caption_set["group"] == 0) & (caption_set["perm"] == torch.arange(4)).all(1)
    tier = caption_set.get("tier", torch.zeros_like(caption_set["window"]))
    by_window = {}
    for v in (identity & (variant_split == split)).nonzero().flatten().tolist():
        by_window.setdefault(int(caption_set["window"][v]), {}).setdefault(int(tier[v]), v)
    chosen = random.Random(0).sample(sorted(by_window), min(limit, len(by_window)))
    order = []
    for i, levels in enumerate(by_window[w] for w in chosen):
        choices = sorted(levels)
        order.append(levels[choices[i % len(choices)]])
    return HeldOutCaptions(
        windows=windows[caption_set["window"][order]],
        emb=caption_set["emb"][order],
        tier=tier[order],
        texts=[caption_set["texts"][v] for v in order],
    )


class CaptionHalf:
    """Training variants of a caption set whose windows are in the training split, and the
    held-out windows' identity captions for evaluation."""

    def __init__(self, path, store, val_limit=512):
        caption_set = torch.load(path, weights_only=False)
        windows, program = resolve(caption_set, store)
        split = torch.tensor([("train", "val", "test").index(program_split(p)) if e >= 0 else -1 for (p, _, _, _), e in zip(caption_set["keys"], windows[:, 0].tolist())])
        variant_split = split[caption_set["window"]]
        train = variant_split == 0
        self.windows, self.program = windows, program
        self.emb = caption_set["emb"][train]
        self.window = caption_set["window"][train]
        self.caption = caption_set["caption"][train]
        self.perm = caption_set["perm"][train]
        self.group = caption_set["group"][train]
        self.rotation_free = caption_set["rotation_free"][train]
        self.colour_free = caption_set["colour_free"][train]
        print(f"captions: {self.rotation_free.float().mean():.0%} rotation-free, {self.colour_free.float().mean():.0%} colour-free", flush=True)
        self.val = held_out_captions(caption_set, windows, variant_split, 1, val_limit)
        self.test = held_out_captions(caption_set, windows, variant_split, 2, val_limit)
        print(f"captions: {int(train.sum())} training variants over {len(self.window.unique())} windows; {len(self.val.windows)} val and {len(self.test.windows)} test windows", flush=True)

    def sample(self, store, n, generator, device):
        """obs, goal frames, actions, weights, text embeddings and the contrastive negative mask."""
        v = torch.randint(len(self.emb), (n,), generator=generator)
        window = self.window[v]
        perm, group = self.perm[v], self.group[v]
        random_perm = torch.stack([torch.randperm(4, generator=generator) for _ in range(n)])
        random_group = torch.randint(8, (n,), generator=generator)
        perm = torch.where(self.colour_free[v].view(-1, 1), random_perm, perm)
        group = torch.where(self.rotation_free[v], random_group, group)
        obs, goal, actions, weights, future = batch_from(store, self.windows[window], generator, device, perm, group)
        program = self.program[window]
        negative = (program.view(-1, 1) != program.view(1, -1)) & (self.caption[v].view(-1, 1) != self.caption[v].view(1, -1))
        return obs, goal, actions, weights, future, self.emb[v].float().to(device), negative.to(device)


def lr_factor(cfg, step, fraction):
    warm = min(1.0, (step + 1) / cfg.warmup_steps)
    cosine = cfg.lr_floor + (1 - cfg.lr_floor) * 0.5 * (1 + math.cos(math.pi * min(fraction, 1.0)))
    return warm * cosine


def null_mask(n, null_prob, generator, device):
    return (torch.rand(n, generator=generator) < null_prob).to(device)


def load_data(cfg, device):
    """(train store, train windows, val store, val windows, test store, test windows); val and test
    windows are ones the do-nothing baseline fails."""
    eval_gen = torch.Generator().manual_seed(1)
    if cfg.data == "scripted":
        store = load_or_generate(cfg.grid, cfg.episodes, cfg.seed, device, cfg.data_noise, cfg.command_oracle)
        val_store = load_or_generate(cfg.grid, cfg.val_episodes, 10_000 + cfg.seed, device, commands=cfg.command_oracle)
        test_store = load_or_generate(cfg.grid, cfg.test_episodes, 20_000 + cfg.seed, device, commands=cfg.command_oracle)
        windows = make_windows(store, cfg.grid)
        val_all, test_all = make_windows(val_store, cfg.grid), make_windows(test_store, cfg.grid)
    else:
        if cfg.play_cache:
            store = load_play_cache(cfg.play_cache)
        else:
            store = load_play(run_dirs(cfg.play_root, size=cfg.grid, cutoff=cfg.play_cutoff))
        val_store = test_store = store
        parts = split_windows(store, make_windows(store, cfg.grid))
        windows, val_all, test_all = parts["train"], parts["val"], parts["test"]
    val_windows = goal_eval.eventful_windows(val_store, val_all, device, cfg.val_windows, eval_gen)
    test_windows = goal_eval.eventful_windows(test_store, test_all, device, cfg.test_windows, eval_gen)
    print(f"grid {cfg.grid}, {cfg.data}: {len(store['phases'])} episodes, {len(windows)} training windows; val {len(val_windows)}, test {len(test_windows)}", flush=True)
    return store, windows, val_store, val_windows, test_store, test_windows


def train(cfg):
    assert not cfg.view or cfg.execute == 1, "the relative-position policies plan one step at a time: use --execute 1"
    run_dir = Path("runs") / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    # On XLA (--device xla) the model trains on the TPU and everything else stays on the CPU
    # (beebot.vla.tpu); elsewhere device and data_device are the same.
    device, data_device = devices(cfg.device)
    torch.manual_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)

    store, windows, val_store, val_windows, test_store, test_windows = load_data(cfg, data_device)
    # Play: near-duplicate programs share one program's weight (play_data.program_weights).
    window_weights = store["episode_weight"][windows[:, 0]] if "episode_weight" in store else None
    captions = CaptionHalf(cfg.captions, store, cfg.text_val_windows) if cfg.captions else None
    if captions is not None and cfg.requested:
        labels = json.loads(Path(cfg.requested).read_text())
        for split in ("val", "test"):
            held = getattr(captions, split)
            assert len(labels[split]) == len(held.windows), f"{cfg.requested} labels {split} for another sample of windows"
            held.requested = labels[split]
    n_text = round(cfg.batch_size * cfg.text_fraction) if captions else 0
    n_goal = cfg.batch_size - n_text

    obs_channels = N_CHANNELS + (COMMAND_CHANNELS if cfg.command_oracle else 0)
    def build():
        if cfg.view:
            return RelativePolicy(cfg.view, width=cfg.width, layers=cfg.layers, obs_channels=obs_channels, crop=cfg.crop, text_dim=1024 if cfg.captions else 0, text_mode=cfg.text_mode, text_tokens=cfg.text_tokens)
        return Policy(cfg.grid, width=cfg.width, expert_width=cfg.expert_width, layers=cfg.layers, goal_tokens=cfg.goal_tokens, goal_pool=cfg.goal_pool, obs_channels=obs_channels, objective=cfg.objective, world_model=cfg.world_model, insulate=cfg.insulate)

    policy = build().to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(f"params {sum(p.numel() for p in policy.parameters()) / 1e6:.2f}M", flush=True)

    # The budget counts training time only: evaluation is slower on bigger grids (longer horizons),
    # and counting it would hand bigger grids less training.
    budget = cfg.minutes * 60
    start = awake_seconds()
    eval_seconds = 0.0
    next_eval = cfg.eval_every_minutes * 60
    best = -1.0
    step_i = 0
    log = open(run_dir / "log.jsonl", "a")
    while True:
        elapsed = awake_seconds() - start - eval_seconds
        out_of_budget = step_i >= cfg.max_steps if cfg.max_steps else elapsed >= budget
        out_of_budget = out_of_budget or bool(cfg.wall_minutes and awake_seconds() - start >= cfg.wall_minutes * 60)
        if out_of_budget or (run_dir / "STOP").exists():
            break
        fraction = step_i / cfg.max_steps if cfg.max_steps else elapsed / budget
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr * lr_factor(cfg, step_i, fraction)

        batch = sample_batch(store, windows, n_goal, generator, data_device, cfg.augment, cfg.hold_prob, cfg.command_oracle, window_weights)
        obs, goal, actions, weights, future = to_device(device, *batch)
        z = policy.goal_z(policy.goal_encoder(goal), null_mask(n_goal, cfg.null_prob, generator, data_device).to(device))
        loss = policy.flow_loss(obs, z, actions, weights, future=future)
        record = {}
        if captions:
            t_obs, t_goal, t_actions, t_weights, t_future, emb, negative = to_device(device, *captions.sample(store, n_text, generator, data_device))
            raw_text = policy.text_head(emb)
            text_null = null_mask(n_text, cfg.null_prob, generator, data_device).to(device)
            t_z = policy.goal_z(raw_text, text_null)
            text_loss = policy.flow_loss(t_obs, t_z, t_actions, t_weights, future=t_future)
            contrastive = policy.contrastive_loss(policy.goal_encoder(t_goal), raw_text, negative)
            record = {"goal_loss": round(loss.item(), 5), "text_loss": round(text_loss.item(), 5), "contrastive": round(contrastive.item(), 5)}
            if cfg.distill_weight:
                # Rows whose caption was dropped (the null goal) have nothing to copy the teacher from.
                distill = policy.distill_loss(t_obs, t_z, policy.goal_encoder(t_goal), t_weights, ~text_null)
                record["distill"] = round(distill.item(), 5)
                text_loss = text_loss + cfg.distill_weight * distill
            loss = (n_goal * loss + n_text * text_loss) / cfg.batch_size + cfg.contrastive_weight * contrastive
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
        optimizer.step()
        step_done(device)
        step_i += 1

        if step_i % cfg.log_every == 0:
            record = {"step": step_i, "minutes": round(elapsed / 60, 2), "loss": round(loss.item(), 5), "lr": optimizer.param_groups[0]["lr"], **record}
            if step_i % cfg.gap_every == 0:
                with torch.no_grad():
                    record["z_gap"] = z_gap(policy, obs, z, actions, weights, future)
                    if captions:
                        record["text_z_gap"] = z_gap(policy, t_obs, t_z, t_actions, t_weights, t_future)
            log.write(json.dumps(record) + "\n")
            log.flush()

        if elapsed >= next_eval:
            next_eval += cfg.eval_every_minutes * 60
            eval_start = awake_seconds()
            best = evaluate_and_keep(cpu_view(policy, build, device), val_store, val_windows, captions, data_device, run_dir, step_i, elapsed, best, cfg.execute)
            eval_seconds += awake_seconds() - eval_start
    log.close()
    # From here on (final val, test) everything runs on the data device (the CPU on XLA).
    policy, device = cpu_view(policy, build, device), data_device
    torch.save(policy.state_dict(), run_dir / "model.pt")
    best = evaluate_and_keep(policy, val_store, val_windows, captions, device, run_dir, step_i, awake_seconds() - start - eval_seconds, best, cfg.execute)

    policy.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=device))
    policy.eval()
    results = {"grid": cfg.grid, "data": cfg.data, "steps": step_i, "best_val_selection_score": best}
    for guidance in (1.0, 2.0):
        results[f"test_policy_w{guidance:g}"] = goal_eval.summary(goal_eval.run(policy, test_store, test_windows, device, guidance=guidance, execute=cfg.execute))
    agree, agree_shuffled = goal_eval.direction_agreement(policy, test_store, test_windows[:256], device)
    results["test_direction_agreement"] = {"true_goal": agree, "shuffled_goal": agree_shuffled}
    results["test_zero"] = goal_eval.summary(goal_eval.run(None, test_store, test_windows, device, mode="zero"))
    results["test_replay"] = goal_eval.summary(goal_eval.run(None, test_store, test_windows, device, mode="replay"))
    if captions is not None and len(captions.test.windows):
        for guidance in (1.0, 2.0):
            results[f"test_text_w{guidance:g}"] = text_scores(policy, test_store, captions.test, device, guidance, cfg.execute)
    (run_dir / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


def z_gap(policy, obs, z, actions, weights, future):
    """Flow loss with z shuffled across the batch minus with the true z, on identical noise."""
    t, e = policy.loss_noise(len(obs), obs.device)
    true_loss = policy.flow_loss(obs, z, actions, weights, t=t, e=e, future=future).item()
    shuffled_loss = policy.flow_loss(obs, z.roll(1, dims=0), actions, weights, t=t, e=e, future=future).item()
    return round(shuffled_loss - true_loss, 5)


@torch.no_grad()
def text_scores(policy, store, held_out, device, guidance=1.0, execute=goal_eval.EXECUTE):
    """Held-out captions: closed-loop reaching of the window's end when conditioned on the caption,
    the same with captions shuffled across windows, and text-to-goal retrieval top-1 and top-5;
    with requested-events labels (held_out.requested), also the fraction of what each caption
    asks for that the rollout does (text_requested_best; goal_eval.requested_scores)."""
    windows, emb, requested = held_out.windows, held_out.emb, held_out.requested
    true_result = goal_eval.run(policy, store, windows, device, guidance=guidance, text_emb=emb, execute=execute, requested=requested)
    true = goal_eval.summary(true_result)
    shuffled = goal_eval.summary(goal_eval.run(policy, store, windows, device, guidance=guidance, text_emb=emb.roll(1, dims=0), execute=execute, requested=requested))
    asked = {}
    if requested is not None:
        asked = {f"text_{k}": true[k] for k in ("requested_best", "requested_end", "unasked_disruptive")}
        asked["text_shuffled_requested_best"] = shuffled["requested_best"]
        for level, name in enumerate(TIERS):
            mine = (held_out.tier == level).numpy()
            if mine.any():
                asked[f"text_requested_{name}"] = round(np.nanmean(true_result["requested_best"].numpy()[mine]).item(), 4)
    per_tier = {}
    for level, name in enumerate(TIERS):
        mine = (held_out.tier == level).numpy()
        if mine.any():
            per_tier[f"text_fact_{name}"] = round(np.nanmean(true_result["fact_score"].numpy()[mine]).item(), 4)
    retrieval = text_retrieval(policy, store, windows, emb, device) if hasattr(policy, "z_norm") else {}
    return {
        "text_fact_score": true["fact_score"],
        "text_shuffled_fact_score": shuffled["fact_score"],
        **per_tier,
        **{f"text_{k}": v for k, v in true.items() if k.startswith(("fact_events_", "n_events_"))},
        "text_reached": true["reached"],
        "text_overlap": true["overlap"],
        "text_shuffled_reached": shuffled["reached"],
        "text_shuffled_overlap": shuffled["overlap"],
        **asked,
        **retrieval,
    }


def text_retrieval(policy, store, windows, emb, device):
    """Text-to-goal retrieval top-1 and top-5 between the shared goal and text z (policies with a
    shared z space only; the relative-position policies have none)."""
    n = min(len(windows), 256)
    e, w0, w1 = windows[:n, 0], windows[:n, 1], windows[:n, 2]
    frames = torch.cat([observe(world_at(store, e, w0, device)), observe(world_at(store, e, w1, device))], dim=1)
    g = F.normalize(policy.z_norm(policy.goal_encoder(frames)).flatten(1), dim=-1)
    t = F.normalize(policy.z_norm(policy.text_head(emb[:n].float().to(device))).flatten(1), dim=-1)
    rank = (t @ g.T).argsort(dim=1, descending=True)
    hits = rank == torch.arange(n, device=device).view(-1, 1)
    return {"retrieval_top1": round(hits[:, :1].any(1).float().mean().item(), 4), "retrieval_top5": round(hits[:, :5].any(1).float().mean().item(), 4), "retrieval_n": n}


def evaluate_and_keep(policy, store, windows, captions, device, run_dir, step_i, elapsed, best, execute=4):
    policy.eval()
    scores = goal_eval.summary(goal_eval.run(policy, store, windows, device, execute=execute))
    if captions is not None and len(captions.val.windows):
        scores.update(text_scores(policy, store, captions.val, device, execute=execute))
    # Checkpoints are chosen on the goal facts score, averaged with the text facts score when the
    # run trains a caption half (the text half is that run's point).
    # With requested-events labels, the text half is judged on what each caption asks for.
    text_key = "text_requested_best" if "text_requested_best" in scores else "text_fact_score"
    scores["selection_score"] = round((scores["fact_score"] + scores[text_key]) / 2, 4) if text_key in scores else scores["fact_score"]
    policy.train()
    with open(run_dir / "val_history.jsonl", "a") as f:
        f.write(json.dumps({"step": step_i, "minutes": round(elapsed / 60, 2), **scores}) + "\n")
    print(f"val at {elapsed / 60:.1f} min, step {step_i}: {scores}", flush=True)
    # Checkpoints are chosen on the semantic facts score: "reached" turned out too strict to
    # rank with (a goal-reading oracle reaches only 7% of val windows); see VLA_HILLCLIMB.md.
    if scores["selection_score"] > best:
        torch.save(policy.state_dict(), run_dir / "model_best.pt")
    return max(best, scores["selection_score"])


def main():
    train(parse_config(Config))


if __name__ == "__main__":
    main()
