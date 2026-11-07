"""RL fine-tuning of a Qwen LM policy (lm_train) on instructions sampled per scene: the
counterfactual coverage that play data with hindsight captions lacks (the same scene under other
words, instructions already done, impossible ones).

    uv run python -m beebot.vla.lm_rl --name lm-rl-pilot --init_run runs/lm-qwen-lora-c1 [--updates 150]

Each update: `prompts` training scenes, each with an instruction drawn from its own contents
(eat a cheese that is there, one just removed, one of a colour that is absent; go into a box;
move a direction; end up in a region), `samples` rollouts of `steps` steps per prompt from the
policy itself (guidance 1). Reward: the fraction of the requested events (goal_eval.
requested_scores) met at the best moment, minus disrupt_penalty if it ends having eaten or
pushed a cheese nobody asked about; an instruction that cannot be done asks for nothing, so only
the penalty applies. Advantage: reward minus its prompt's mean (a group baseline, as in GRPO).
The policy gradient uses grad_states random steps of each rollout; an imitation loss on play
data (goal frames and captions, as lm_train) keeps what the policy already does. Instruction
wordings differ from the grounding tests' (beebot.vla.grounding), which are the live scorecard.
"""

import contextlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from beebot.train import parse_config
from beebot.vla import caption, goal_eval, grounding, relabel, tasks
from beebot.vla.data import decode_field, sample_batch
from beebot.vla.lm_policy import PATCH, LMPolicy
import torch.distributed as dist

from beebot.vla.lm_train import Captions, average_gradients, setup_distributed
from beebot.vla.model import patchify
from beebot.vla.play_data import load_play_cache
from beebot.vla.train import Config, load_data
from beebot.vla.world import PALETTE_NAMES, observe, step

DIRECTIONS = ["right",  # facts.direction_name's
    "up and right", "up", "up and left", "left", "down and left", "down", "down and right"]
REGIONS = ["top left", "top", "top right", "left", "center", "right", "bottom left", "bottom", "bottom right"]  # facts.region_name's
WORDINGS = {
    "eat": ["go eat the {} cheese", "eat the {} one", "gobble up the {} cheese", "find the {} cheese and eat it", "munch the {} cheese", "the {} cheese: eat it"],
    "box": ["get into the {} box", "head into the {} box", "move inside the {} box", "enter the {} box", "everyone into the {} box"],
    "move": ["move {}", "go {}", "head {}", "shift the swarm {}"],
    "region": ["go to the {}", "end up in the {}", "head to the {}", "move to the {}"],
}


@dataclass
class RLConfig:
    name: str = "lm-rl"
    init_run: str = "runs/lm-qwen-lora-c1"  # the directory with lm_train's config.json and the checkpoint (or that config.json itself)
    checkpoint: str = "policy_best.pt"
    play_cache: str = "data/vla/play_cache_g32"
    captions: str = "data/vla/kaggle_play/captions_g32.jsonl"
    play_root: str = ""  # accepted for the Kaggle kernel's arguments; the cache is what is read
    device: str = "mps"
    wall_minutes: float = 0.0  # > 0: stop training after this much wall time since start (room for the last scorecard)
    seed: int = 0
    updates: int = 150
    minutes: float = 0.0  # > 0: also stop after this much wall time
    prompts: int = 8  # scenes, one instruction each, per update
    samples: int = 4  # rollouts per prompt (the group baseline)
    steps: int = 48
    grad_states: int = 8  # steps of each rollout in the policy gradient
    chunk: int = 32  # states per forward and backward
    lr_lm: float = 5e-5
    lr_new: float = 5e-5
    disrupt_penalty: float = 1.0
    imitation_weight: float = 0.5
    imitation_batch: int = 16  # goal and caption examples each
    eval_every: int = 25
    eval_scenes: int = 32
    eval_steps: int = 48
    eval_guidance: float = 2.0
    max_grad_norm: float = 1.0
    mode: str = "grounding"  # "grounding": instructions on play scenes (instruction()); "generalist": the task suite (beebot.vla.tasks)
    eval_tasks: int = 16  # generalist scorecard: tasks per family (in distribution) and per held-out split
    relabel_weight: float = 0.0  # generalist: also imitate each rollout under an instruction it satisfied (relabel), with this weight
    idle_keep: float = 0.25  # relabelling: chance of keeping an eat-nothing logic label (most rollouts satisfy one; see relabel.thin_idle)


def init_dir(cfg):
    path = Path(cfg.init_run)
    return path.parent if path.is_file() else path


def load_policy(cfg, device):
    """The LM policy from init_run. On CUDA the frozen LM weights are fp16 (as lm_train's half_base)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run = json.loads((init_dir(cfg) / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(run["model"])
    tokenizer.padding_side = "right"
    lm = AutoModelForCausalLM.from_pretrained(run["model"], dtype=torch.float16 if device.type == "cuda" else torch.float32)
    policy = LMPolicy(lm, lora_rank=run["lora_rank"], freeze_embeddings=run.get("freeze_embeddings", True))
    saved = torch.load(init_dir(cfg) / cfg.checkpoint, map_location="cpu")
    weights = saved.get("weights", saved)
    missing = [n for n, p in policy.named_parameters() if p.requires_grad and n not in weights]
    assert not missing, f"checkpoint lacks trainable weights: {missing[:5]}"
    policy.load_state_dict({k: v.float() for k, v in weights.items()}, strict=False)
    return policy.to(device), tokenizer


def instruction(store, e, w0, rng):
    """A training instruction for the scene at (e, w0): (text, requested atoms, cheese slot to
    remove before the start or None). Kinds are drawn uniformly among those the scene allows."""
    colours = store["cheese_color"][e].tolist()
    alive = store["cheese_alive"][e, w0].tolist()
    present = {PALETTE_NAMES[c]: slot for slot, (c, a) in enumerate(zip(colours, alive)) if c >= 0 and a}
    absent = [c for c in PALETTE_NAMES if c not in present]
    boxes = sorted({PALETTE_NAMES[c] for c, rect in zip(store["box_color"][e].tolist(), store["box_rect"][e].tolist()) if c >= 0 and rect[2] > 0})
    kinds = ["move", "region"] + (["eat", "eat_gone"] if present else []) + (["eat_absent"] if absent else []) + (["box"] if boxes else [])
    kind = rng.choice(kinds)
    if kind == "eat":
        c = rng.choice(sorted(present))
        return rng.choice(WORDINGS["eat"]).format(c), {("ate", c)}, None
    if kind == "eat_gone":
        c = rng.choice(sorted(present))
        return rng.choice(WORDINGS["eat"]).format(c), set(), present[c]
    if kind == "eat_absent":
        return rng.choice(WORDINGS["eat"]).format(rng.choice(absent)), set(), None
    if kind == "box":
        b = rng.choice(boxes)
        return rng.choice(WORDINGS["box"]).format(b), {("into box", b)}, None
    if kind == "move":
        d = rng.choice(DIRECTIONS)
        return rng.choice(WORDINGS["move"]).format(d), {("move", d)}, None
    r = rng.choice(REGIONS)
    return rng.choice(WORDINGS["region"]).format(r), {("end region", r)}, None


def tokens(tokenizer, texts, device):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=96)
    return enc.input_ids.to(device), enc.attention_mask.to(device)


@torch.no_grad()
def collect(policy, tokenizer, store, windows, cfg, rng, device):
    """Rollouts for one update: per rollout its reward and grad_states (obs, moves) pairs."""
    trials = []
    for _ in range(cfg.prompts):
        e, w0, _ = windows[rng.randrange(len(windows))].tolist()
        text, requested, remove = instruction(store, e, w0, rng)
        trials += [{"e": e, "w0": w0, "text": text, "requested": requested, "remove": remove}] * cfg.samples
    world = grounding.start_worlds(store, trials, device)
    start = world.to("cpu")
    ids, mask = tokens(tokenizer, [t["text"] for t in trials], device)
    keep = sorted(rng.sample(range(cfg.steps), cfg.grad_states))
    states, snaps = [], []
    b, g = len(trials), world.mass.shape[-1]
    n = g // PATCH
    for t in range(cfg.steps):
        obs = observe(world)
        moves = torch.distributions.Categorical(logits=policy.logits(obs, text_ids=ids, text_mask=mask).float()).sample()
        if t in keep:
            states.append((obs.cpu(), moves.cpu()))
        codes = moves.view(b, n, n, PATCH, PATCH).permute(0, 1, 3, 2, 4).reshape(b, g, g)
        world, _ = step(world, decode_field(codes, device))
        if (t + 1) % goal_eval.FACT_EVERY == 0:
            snaps.append(world.to("cpu"))
    rewards, met, disrupted = [], [], []
    for i, tr in enumerate(trials):
        static = {"walls": start.walls[i].numpy(), "cheese_color": start.cheese_color[i].numpy(), "box_rect": start.box_rect[i].numpy(), "box_color": start.box_color[i].numpy()}
        best, _, unasked = goal_eval.requested_scores(static, goal_eval.snapshot(start, i), [goal_eval.snapshot(s, i) for s in snaps], tr["requested"])
        rewards.append((best or 0.0) - cfg.disrupt_penalty * float(unasked))
        met.append(best)
        disrupted.append(unasked)
    r = torch.tensor(rewards)
    advantage = r - r.view(cfg.prompts, cfg.samples).mean(1).repeat_interleave(cfg.samples)
    stats = {
        "reward": round(r.mean().item(), 4),
        "met": round(sum(m for m in met if m is not None) / max(sum(m is not None for m in met), 1), 4),
        "unasked": round(sum(disrupted) / len(disrupted), 4),
    }
    return states, ids.cpu(), mask.cpu(), advantage, stats


def lm_sampler(policy, guidance, n):
    """tasks.run's sample(obs, cond): per-square moves and the field, from the LM policy."""

    def sample(obs, cond):
        ids, mask = cond
        out = policy.logits(obs, text_ids=ids, text_mask=mask)
        if guidance != 1.0:
            null = policy.logits(obs, text_ids=ids, text_mask=mask, null=torch.ones(len(obs), dtype=torch.bool, device=obs.device))
            out = null + guidance * (out - null)
        moves = torch.distributions.Categorical(logits=out.float()).sample()
        b, g = len(obs), obs.shape[-1]
        codes = moves.view(b, n, n, PATCH, PATCH).permute(0, 1, 3, 2, 4).reshape(b, g, g)
        return moves, decode_field(codes, obs.device)

    return sample


@torch.no_grad()
def collect_tasks(policy, tokenizer, cfg, rng, np_rng, device):
    """Rollouts for one update on the generalist task suite (training split)."""
    pairs = [tasks.sample_task(np_rng, "train") for _ in range(cfg.prompts)]
    scenes = [s for s, _ in pairs for _ in range(cfg.samples)]
    batch = [t for _, t in pairs for _ in range(cfg.samples)]
    keep = set(rng.sample(range(cfg.steps), cfg.grad_states))
    results, kept, outcomes = tasks.run(lm_sampler(policy, 1.0, tasks.SIZE // PATCH), lambda texts: tokens(tokenizer, texts, device), scenes, batch, cfg.steps, device, keep, outcomes=True)
    r = torch.tensor([s - cfg.disrupt_penalty * float(u) for s, u in results])
    advantage = r - r.view(cfg.prompts, cfg.samples).mean(1).repeat_interleave(cfg.samples)
    ids, mask = tokens(tokenizer, [t.text for t in batch], "cpu")
    by_family = {}
    for t, (s, _) in zip(batch, results):
        by_family.setdefault(t.family, []).append(s)
    flat = r.view(cfg.prompts, cfg.samples)
    stats = {"reward": round(r.mean().item(), 4), "met": round(float(np.mean([s for s, _ in results])), 4), "unasked": round(float(np.mean([u for _, u in results])), 4), "flat_groups": round(float((flat.max(1).values - flat.min(1).values < 1e-6).float().mean()), 3), **{f"met_{f}": round(float(np.mean(v)), 3) for f, v in sorted(by_family.items())}}
    hindsight = None
    if cfg.relabel_weight:
        texts, weights, families = [], [], {}
        for scene, (frames, eaten) in zip(scenes, outcomes):
            options = relabel.options_of(scene, frames, eaten, np_rng)
            if "logic" in options:
                active = [o for o in options["logic"] if o[1]["form"] not in relabel.IDLE_FORMS]
                options["logic"] = active if active or np_rng.random() >= cfg.idle_keep else options["logic"]
                if not options["logic"]:
                    del options["logic"]
            if not options:
                texts.append("")
                weights.append(0.0)
                continue
            family = str(np_rng.choice(sorted(options)))
            texts.append(relabel.pick(family, options[family], np_rng)[0])
            weights.append(1.0)
            families[family] = families.get(family, 0) + 1
        h_ids, h_mask = tokens(tokenizer, [caption.strip_tags(t) for t in texts], "cpu")
        hindsight = (h_ids, h_mask, torch.tensor(weights))
        stats.update({"relabelled": round(float(np.mean(weights)), 3), **{f"relabel_{f}": n for f, n in sorted(families.items())}})
    return kept, ids, mask, advantage, stats, hindsight


def generalist_card(policy, tokenizer, cfg, device, suite):
    """Mean score (and unasked-eating rate) per family on in-distribution test tasks and per
    held-out split, at eval_guidance."""
    policy.eval()
    rows = {}
    for name, pairs in suite.items():
        scenes, batch = [s for s, _ in pairs], [t for _, t in pairs]
        results, _ = tasks.run(lm_sampler(policy, cfg.eval_guidance, tasks.SIZE // PATCH), lambda texts: tokens(tokenizer, texts, device), scenes, batch, cfg.eval_steps, device)
        rows[name] = {"score": round(float(np.mean([s for s, _ in results])), 3), "unasked": round(float(np.mean([u for _, u in results])), 3)}
    policy.train()
    return rows


def make_suite(cfg, seed=12345):
    """Fixed evaluation tasks: eval_tasks per family (test split) and per held-out split."""
    np_rng = np.random.default_rng(seed)
    suite = {f"test_{f}": [tasks.sample_task(np_rng, "test", family=f) for _ in range(cfg.eval_tasks)] for f in tasks.FAMILIES}
    suite.update({f"held_{h}": [tasks.sample_task(np_rng, h) for _ in range(cfg.eval_tasks)] for h in tasks.HELD_OUT})
    return suite


def policy_gradient(policy, states, ids, mask, advantage, cfg, device, scale=lambda loss: loss, amp=contextlib.nullcontext):
    """Accumulates the gradient of -advantage * mean log-probability of the taken moves over each
    state's robots, averaged over all states; returns the loss value."""
    obs = torch.cat([s[0] for s in states])
    moves = torch.cat([s[1] for s in states])
    k = len(states)
    ids_all, mask_all, adv_all = ids.repeat(k, 1), mask.repeat(k, 1), advantage.repeat(k)
    total = len(obs)
    loss_value = 0.0
    for s in range(0, total, cfg.chunk):
        o = obs[s : s + cfg.chunk].to(device)
        with amp():
            out = policy.logits(o, text_ids=ids_all[s : s + cfg.chunk].to(device), text_mask=mask_all[s : s + cfg.chunk].to(device))
        logp = F.log_softmax(out.float(), dim=-1).gather(-1, moves[s : s + cfg.chunk].to(device).unsqueeze(-1)).squeeze(-1)
        robots = patchify((o[:, :1] > 0.5).float(), PATCH)
        per_state = (logp * robots).sum((1, 2)) / robots.sum((1, 2)).clamp(min=1.0)
        loss = -(adv_all[s : s + cfg.chunk].to(device) * per_state).sum() / total
        scale(loss).backward()
        loss_value += loss.item()
    return loss_value


def scorecard(policy, tokenizer, store, trials, cfg, device):
    policy.eval()

    def sample(obs, cond):
        return policy.sample(obs, text_ids=cond[0], text_mask=cond[1], guidance=cfg.eval_guidance)

    rows = grounding.summarise(grounding.rollout(sample, lambda texts: tokens(tokenizer, texts, device), trials, store, cfg.eval_steps, device, 32))
    policy.train()
    return rows


def run():
    cfg = parse_config(RLConfig)
    rank, world, device = setup_distributed(cfg)
    lead = rank == 0
    t_start = time.time()
    torch.manual_seed(cfg.seed)  # the same on every process: identical weights from the same init
    rng = random.Random(cfg.seed + 1000 * rank)  # but each samples its own scenes and rollouts
    generator = torch.Generator().manual_seed(cfg.seed + 1000 * rank)
    np_rng = np.random.default_rng(cfg.seed + 1000 * rank)
    run_dir = Path("runs") / cfg.name
    init = json.loads((init_dir(cfg) / "config.json").read_text())
    if lead:
        run_dir.mkdir(parents=True, exist_ok=True)
        # lm_train's config of the policy (what vla_lm_play.load and the demos read), with this run's.
        (run_dir / "config.json").write_text(json.dumps({**init, "rl": asdict(cfg)}, indent=2))

    policy, tokenizer = load_policy(cfg, device)
    store = load_play_cache(cfg.play_cache)
    data_cfg = Config(grid=32, data="play", play_cache=cfg.play_cache, val_windows=128, test_windows=512)
    _, windows, _, _, _, test_windows = load_data(data_cfg, "cpu")
    captions = Captions(cfg.captions, store, tokenizer, 128, 128)
    trials = grounding.make_trials(store, test_windows, cfg.eval_scenes)
    suite = make_suite(cfg) if cfg.mode == "generalist" else None
    use_amp = device.type == "cuda"

    def amp():
        return torch.autocast(device.type, dtype=torch.float16, enabled=use_amp)

    def card():
        with amp():
            out = scorecard(policy, tokenizer, store, trials, cfg, device)
            if suite is not None:
                out["general"] = generalist_card(policy, tokenizer, cfg, device, suite)
        return out

    optimizer = torch.optim.AdamW([{"params": policy.lm_parameters(), "lr": cfg.lr_lm}, {"params": policy.new_parameters(), "lr": cfg.lr_new}], weight_decay=0.0)
    params = [p for g in optimizer.param_groups for p in g["params"]]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = open(run_dir / "history.jsonl", "a") if lead else None

    def log(row):
        if lead:
            print(json.dumps(row), flush=True)
            history.write(json.dumps(row) + "\n")
            history.flush()

    def save():
        if lead:
            trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
            torch.save({k: v.half() for k, v in policy.state_dict().items() if k in trainable}, run_dir / "policy_best.pt")

    def evaluate(update):
        if lead:
            log({"update": update, **card()})
            save()
        if world > 1:
            dist.barrier()  # the others wait for process 0's scorecard

    evaluate(0)
    policy.train()
    t0 = time.time()
    done = 0  # the last completed update (--updates 0: only the starting scorecard)
    for update in range(1, cfg.updates + 1):
        stop = bool((cfg.minutes and time.time() - t0 > cfg.minutes * 60) or (cfg.wall_minutes and time.time() - t_start > cfg.wall_minutes * 60))
        if world > 1:  # process 0's clock for everyone
            flag = torch.tensor([float(stop)], device=device)
            dist.broadcast(flag, 0)
            stop = bool(flag.item())
        if stop:
            break
        t_update = time.time()
        policy.eval()
        with amp():
            hindsight = None
            if cfg.mode == "generalist":
                states, ids, mask, advantage, stats, hindsight = collect_tasks(policy, tokenizer, cfg, rng, np_rng, device)
            else:
                states, ids, mask, advantage, stats = collect(policy, tokenizer, store, windows, cfg, rng, device)
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        pg = policy_gradient(policy, states, ids, mask, advantage, cfg, device, scale=scaler.scale, amp=amp)
        if hindsight is not None:  # imitation of each rollout under what it did (a constant positive advantage)
            h_ids, h_mask, h_weights = hindsight
            stats["hindsight"] = round(policy_gradient(policy, states, h_ids, h_mask, cfg.relabel_weight * h_weights, cfg, device, scale=scaler.scale, amp=amp), 5)
        imitation = 0.0
        if cfg.imitation_weight:
            obs, goal, actions, weights, _ = sample_batch(store, windows, cfg.imitation_batch, generator, device, True)
            with amp():
                goal_loss = policy.policy_loss(obs, actions, weights, goal=policy.goal_encoder(goal))
            scaler.scale(cfg.imitation_weight * goal_loss).backward()
            t_obs, t_actions, t_weights, t_ids, t_mask = captions.sample(store, cfg.imitation_batch, generator, device)
            with amp():
                text_loss = policy.policy_loss(t_obs, t_actions, t_weights, text_ids=t_ids, text_mask=t_mask)
            scaler.scale(cfg.imitation_weight * text_loss).backward()
            imitation = goal_loss.item() + text_loss.item()
        if world > 1:
            average_gradients(params, world)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm).item()
        scaler.step(optimizer)
        scaler.update()
        log({"update": update, "minutes": round((time.time() - t0) / 60, 1), "seconds": round(time.time() - t_update, 1), **stats, "pg": round(pg, 5), "imitation": round(imitation, 4), "grad_norm": round(grad_norm, 3)})
        done = update
        if update % cfg.eval_every == 0:
            evaluate(update)
    if done % cfg.eval_every:
        evaluate(done)
    if world > 1:
        print(f"process {rank}: weight checksum {sum(p.detach().double().sum().item() for p in params):.10e}", flush=True)
        dist.destroy_process_group()


def main():
    run()
    # As lm_train: exit hard, so no lingering thread keeps a finished Kaggle session alive.
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
