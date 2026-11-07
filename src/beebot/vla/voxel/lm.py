"""The Qwen LM policy in the voxel world: lm_policy.LMPolicy with a 3D input and readout (a 4 x 4
x 4 patch per scene token, so a 16^3 grid is 64 tokens as a 32 x 32 grid is in 2D; 27 actions per
voxel), trained on hindsight-labelled scripted play (voxel/relabel.py) and scored on the voxel task
suite (voxel/tasks.py) against a shuffled-instruction control.

    uv run python -m beebot.vla.voxel.lm --name vox-lm-pilot --minutes 60

Training: text-conditioned imitation of each robot's next action at a random step of a labelled
window (classifier-free: null_prob of rows drop the text). Evaluation: fresh scenes, one
instruction each from every family the scene allows (and held-out pairs), steps rollouts, the
instruction's checker on the facts from start to end; "shuffled" gives each scene another scene's
instruction of the same family, so score above it is instruction-following.
"""

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from beebot.train import parse_config
from beebot.vla.lm_policy import MAX_SIDE, LMPolicy
from beebot.vla.voxel import tasks as V
from beebot.vla.voxel.facts import window_facts
from beebot.vla.voxel.play_store import load
from beebot.vla.voxel.world import N_ACTIONS, N_CHANNELS, PALETTE_NAMES, VoxelWorld, observe, sample_scenes, step

PATCH = 4


def patchify3(x, p=PATCH):
    """(B, C, D, H, W) -> (B, (D/p)(H/p)(W/p), C p^3), patches in (z, y, x) order."""
    b, c, d, h, w = x.shape
    return x.reshape(b, c, d // p, p, h // p, p, w // p, p).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, (d // p) * (h // p) * (w // p), c * p**3)


def unpatchify3(v, d, p=PATCH):
    """(B, n^3, p^3) per-patch values -> (B, D, H, W) for a cube of side d."""
    n = d // p
    return v.reshape(v.shape[0], n, n, n, p, p, p).permute(0, 1, 4, 2, 5, 3, 6).reshape(v.shape[0], d, d, d)


class VoxelLMPolicy(LMPolicy):
    def __init__(self, lm, lora_rank=0, freeze_embeddings=True):
        super().__init__(lm, obs_channels=N_CHANNELS, lora_rank=lora_rank, freeze_embeddings=freeze_embeddings)
        hidden = lm.config.hidden_size
        del self.goal_embed  # text goals only
        self.scene_embed = nn.Linear(N_CHANNELS * PATCH**3, hidden)
        self.depth = nn.Embedding(MAX_SIDE, hidden)
        self.readout = nn.Linear(hidden, PATCH**3 * N_ACTIONS)
        scale = lm.get_input_embeddings().weight.std().item()
        nn.init.normal_(self.scene_embed.weight, std=scale / math.sqrt(self.scene_embed.in_features) * 4)
        nn.init.zeros_(self.scene_embed.bias)
        nn.init.normal_(self.depth.weight, std=scale)

    def positions3(self, d, device):
        n = torch.arange(d // PATCH, device=device)
        return (self.depth(n).view(-1, 1, 1, self.depth.embedding_dim) + self.row(n).view(1, -1, 1, self.depth.embedding_dim) + self.col(n).view(1, 1, -1, self.depth.embedding_dim)).reshape(-1, self.depth.embedding_dim)

    def scene_tokens(self, obs):
        pos = self.positions3(obs.shape[-1], obs.device)
        return (self.scene_embed(patchify3(obs)) + pos + self.kind[0]).to(self.lm.get_input_embeddings().weight.dtype)

    def logits(self, obs, text_ids=None, text_mask=None, null=None):
        """(B, n^3, PATCH^3, 27)."""
        h = self.hidden(obs, None, text_ids, text_mask, null)
        return self.readout(h.float()).view(h.shape[0], h.shape[1], PATCH**3, N_ACTIONS)

    def loss(self, obs, codes, text_ids, text_mask, null=None):
        """Cross-entropy on each robot's action code (codes (B, D, H, W))."""
        out = self.logits(obs, text_ids, text_mask, null)
        target = patchify3(codes.unsqueeze(1).float()).round().long()
        robots = patchify3(obs[:, :1])
        ce = F.cross_entropy(out.reshape(-1, N_ACTIONS), target.reshape(-1), reduction="none").view_as(robots)
        return (ce * robots).sum() / robots.sum().clamp(min=1.0)

    @torch.no_grad()
    def act(self, obs, text_ids, text_mask, guidance=1.0):
        """Sampled action codes (B, D, H, W)."""
        out = self.logits(obs, text_ids, text_mask)
        if guidance != 1.0:
            null = self.logits(obs, text_ids, text_mask, null=torch.ones(len(obs), dtype=torch.bool, device=obs.device))
            out = null + guidance * (out - null)
        moves = torch.distributions.Categorical(logits=out.float()).sample()
        return unpatchify3(moves, obs.shape[-1])


@dataclass
class VoxConfig:
    name: str = "vox-lm"
    model: str = "Qwen/Qwen3-0.6B"
    store: str = "data/vla/voxel_play_g16"
    labels: str = "data/vla/voxel_play_g16/labels.jsonl"
    episodes: int = 0  # 0: all; else train on the first N episodes (data scaling)
    lora_rank: int = 64
    init_lora: str = ""  # a 2D LM policy checkpoint (lm_train's policy_best.pt): start from its LoRA weights (2D -> 3D transfer)
    batch: int = 16
    lr_lm: float = 2e-4
    lr_new: float = 3e-4
    warmup_steps: int = 100
    steps: int = 100000000
    minutes: float = 60.0
    null_prob: float = 0.1
    eval_every_minutes: float = 30.0
    eval_per_family: int = 12  # per evaluation: scenes per instruction family (and of held-out pairs), one instruction each
    eval_steps: int = 48
    eval_guidance: float = 2.0
    test_share: float = 0.05  # the last share of episodes: held-out play for the validation loss
    device: str = "mps"
    seed: int = 0


def tokens(tokenizer, texts, device):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=48)
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def world_at(store, rows):
    """VoxelWorld (CPU) of the store's (e, t) rows."""
    e = torch.tensor([r[0] for r in rows])
    t = torch.tensor([r[1] for r in rows])
    mass = torch.stack([torch.from_numpy(np.asarray(store["mass"][i, j])) for i, j in zip(e.tolist(), t.tolist())]).float()
    return VoxelWorld(mass, store["walls"][e], store["cheese_pos"][e, t].long(), store["cheese_alive"][e, t], store["cheese_color"][e], store["gravity"][e])


def batch_of(store, labels, n, rng, device):
    picks = [labels[rng.randrange(len(labels))] for _ in range(n)]
    rows = [(r["e"], rng.randrange(r["w0"], r["w1"])) for r in picks]
    obs = observe(world_at(store, rows)).to(device)
    codes = torch.stack([torch.from_numpy(np.asarray(store["actions"][e, t])) for e, t in rows]).long().to(device)
    return obs, codes, [r["text"] for r in picks]


EVAL_FAMILIES = ("move", "region", "split", "merge", "push", "eat", "around", "tower", "bridge", "pair")


def eval_tasks(per_family, rng, held=False):
    """(world, [(family, text, spec)]): per_family fresh scenes for each family (held: the
    held-out pairs only), each scene drawn until it allows that family, one instruction each."""
    worlds, out = [], []
    for family in ("held_pair",) if held else EVAL_FAMILIES:
        found = 0
        while found < per_family:
            world, meta = sample_scenes(rng, 16, gravity_prob=0.5)
            for i in range(16):
                if found == per_family:
                    break
                scene = {"colours": [PALETTE_NAMES[c] for c in world.cheese_color[i].tolist() if c >= 0], "gravity": bool(world.gravity[i]), "gap": meta[i]["layout"] == "gap"}
                options = V.candidates(scene, rng)
                if family in ("pair", "held_pair"):
                    choices = V.pair_options(options, rng, held=held)
                else:
                    choices = options.get(family, [])
                if not choices:
                    continue
                text, spec = choices[int(rng.integers(len(choices)))]
                worlds.append(world.index(i))
                out.append((family, text, spec))
                found += 1
    fields = ("mass", "walls", "cheese_pos", "cheese_alive", "cheese_color", "gravity")
    return VoxelWorld(*(torch.cat([getattr(w, f) for w in worlds]) for f in fields)), out


@torch.no_grad()
def rollout(policy, tokenizer, world, texts, steps, guidance, device, generator):
    """Facts from start to end of each scene's rollout; the world steps on the CPU."""
    ids, mask = tokens(tokenizer, texts, device)
    start = world
    for _ in range(steps):
        codes = policy.act(observe(world).to(device), ids, mask, guidance).cpu()
        world, _ = step(world, codes, torch.rand(world.mass.shape, generator=generator))
    facts = []
    for i in range(len(texts)):
        static = {"walls": start.walls[i].cpu().numpy(), "cheese_color": start.cheese_color[i].tolist(), "gravity": bool(start.gravity[i])}

        def snap(w):
            return {"mass": w.mass[i].cpu().numpy(), "cheese_pos": w.cheese_pos[i].cpu().numpy(), "cheese_alive": w.cheese_alive[i].cpu().numpy()}

        facts.append(window_facts(static, snap(start), snap(world)))
    return facts


def scorecard(policy, tokenizer, cfg, device, seed=4242):
    """Per family (and held-out pairs): mean score with the instruction, with a shuffled one (same
    family, another scene's), and the unasked-eating rate."""
    policy.eval()
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    rows = {}
    for held in (False, True):
        world, items = eval_tasks(cfg.eval_per_family, rng, held)
        texts = [t for _, t, _ in items]
        by_family = {}
        for i, (f, _, _) in enumerate(items):
            by_family.setdefault(f, []).append(i)
        shuffled = list(texts)
        for idx in by_family.values():  # another scene's instruction of the same family
            perm = list(np.roll(idx, 1))
            for a, b in zip(idx, perm):
                shuffled[a] = texts[b]
        results = {}
        for name, these in (("score", texts), ("shuffled", shuffled)):
            facts = []
            for s in range(0, len(these), 32):
                part = world.index(slice(s, s + 32))
                facts += rollout(policy, tokenizer, part, these[s : s + 32], cfg.eval_steps, cfg.eval_guidance, device, generator)
            results[name] = [V.check(spec, f) for (_, _, spec), f in zip(items, facts)]
        for f, idx in sorted(by_family.items()):
            rows[f] = {"n": len(idx), "score": round(float(np.mean([results["score"][i][0] for i in idx])), 3), "shuffled": round(float(np.mean([results["shuffled"][i][0] for i in idx])), 3), "unasked": round(float(np.mean([results["score"][i][1] for i in idx])), 3)}
    policy.train()
    return rows


def run():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = parse_config(VoxConfig)
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    run_dir = Path("runs") / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.padding_side = "right"
    lm = AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.float32)
    policy = VoxelLMPolicy(lm, lora_rank=cfg.lora_rank)
    if cfg.init_lora:
        saved = torch.load(cfg.init_lora, map_location="cpu")
        saved = saved.get("weights", saved)
        lora = {k: v.float() for k, v in saved.items() if k.startswith("lm.") and k.rsplit(".", 1)[-1] in ("a", "b")}  # LoRALinear's a and b
        own = dict(policy.named_parameters())
        assert lora and all(k in own and own[k].shape == v.shape for k, v in lora.items()), "init_lora: no matching LoRA weights"
        policy.load_state_dict(lora, strict=False)
        print(f"LoRA from {cfg.init_lora}: {len(lora)} tensors", flush=True)
    policy = policy.to(device)
    store = load(cfg.store)
    n_episodes = cfg.episodes or len(store["gravity"])
    test_from = int(len(store["gravity"]) * (1 - cfg.test_share))
    labels = [json.loads(l) for l in open(cfg.labels)]
    train = [r for r in labels if r["e"] < min(n_episodes, test_from)]
    test = [r for r in labels if r["e"] >= test_from]
    print(f"{len(train)} training labels on {min(n_episodes, test_from)} episodes, {len(test)} test labels", flush=True)
    optimizer = torch.optim.AdamW([{"params": policy.lm_parameters(), "lr": cfg.lr_lm}, {"params": policy.new_parameters(), "lr": cfg.lr_new}], weight_decay=0.01)
    history = open(run_dir / "history.jsonl", "a")

    def log(row):
        print(json.dumps(row), flush=True)
        history.write(json.dumps(row) + "\n")
        history.flush()

    def evaluate(step_i, minutes):
        val = []
        policy.eval()
        with torch.no_grad():
            for k in range(8):  # the same held-out batches every time
                obs, codes, texts = batch_of(store, test, cfg.batch, random.Random(k), device)
                val.append(policy.loss(obs, codes, *tokens(tokenizer, texts, device)).item())
        policy.train()
        log({"step": step_i, "minutes": round(minutes, 1), "val_loss": round(float(np.mean(val)), 4), "card": scorecard(policy, tokenizer, cfg, device)})
        trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
        torch.save({k: v.half() for k, v in policy.state_dict().items() if k in trainable}, run_dir / "policy_latest.pt")

    evaluate(0, 0.0)
    t0 = time.time()  # the training clock starts after the first scorecard
    eval_seconds, next_eval = 0.0, cfg.eval_every_minutes * 60
    losses = []
    for step_i in range(1, cfg.steps + 1):
        elapsed = time.time() - t0 - eval_seconds
        if cfg.minutes and elapsed > cfg.minutes * 60:
            break
        fraction = min(elapsed / (cfg.minutes * 60), 1.0) if cfg.minutes else step_i / cfg.steps
        warm = min(1.0, step_i / cfg.warmup_steps)
        for g, base in zip(optimizer.param_groups, (cfg.lr_lm, cfg.lr_new)):
            g["lr"] = base * warm * 0.5 * (1 + math.cos(math.pi * fraction))
        obs, codes, texts = batch_of(store, train, cfg.batch, rng, device)
        ids, mask = tokens(tokenizer, texts, device)
        null = torch.rand(cfg.batch, device=device) < cfg.null_prob
        loss = policy.loss(obs, codes, ids, mask, null)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
        optimizer.step()
        losses.append(loss.item())
        if step_i % 50 == 0:
            log({"step": step_i, "minutes": round(elapsed / 60, 1), "loss": round(float(np.mean(losses[-50:])), 4)})
        if elapsed >= next_eval:
            t_eval = time.time()
            evaluate(step_i, elapsed / 60)
            eval_seconds += time.time() - t_eval
            next_eval = elapsed + cfg.eval_every_minutes * 60
    evaluate(step_i, (time.time() - t0 - eval_seconds) / 60)


def score_checkpoint():
    """The scorecard of a saved run: uv run python -c 'from beebot.vla.voxel.lm import score_checkpoint; score_checkpoint()' --name RUN"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = parse_config(VoxConfig)
    device = torch.device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.padding_side = "right"
    saved = json.loads((Path("runs") / cfg.name / "config.json").read_text())
    policy = VoxelLMPolicy(AutoModelForCausalLM.from_pretrained(cfg.model, dtype=torch.float32), lora_rank=saved["lora_rank"]).to(device)
    weights = torch.load(Path("runs") / cfg.name / "policy_latest.pt", map_location="cpu")
    policy.load_state_dict({k: v.float() for k, v in weights.items()}, strict=False)
    print(json.dumps(scorecard(policy, tokenizer, cfg, device)), flush=True)


if __name__ == "__main__":
    run()
