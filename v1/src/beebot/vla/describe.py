"""Scene questions for the LM policy: the same scene tokens the policy acts on, placed inline in a
chat prompt the text can attend to, with answers read off the demonstration stores. Trained next
to the action loss (expert_sft.py, --describe_share) so describing the scene emerges in the
shared weights (scene embedding, positions, LoRA) rather than in a separate module.

    present: one frame,  "Which cheeses are on the grid?"        -> "blue, red" / "none"
    eaten:   two frames, "Which cheeses were eaten in between?"   -> "green" / "none"

Phrasings: TRAIN ones are trained on; HELD ones only appear in evaluation (does the model answer
the question or a template?).
"""

import random

import torch
import torch.nn.functional as F

from beebot.vla.data import world_at
from beebot.vla.world import COLOUR_NAMES, observe

PHRASINGS = {
    "present": {
        "train": [
            "Which cheeses are on the grid?",
            "What colours of cheese can you see?",
            "List the cheeses in this scene by colour.",
            "Which cheese colours are present?",
        ],
        "held": [
            "Name every cheese colour you can spot here.",
            "What cheeses remain in the arena?",
        ],
    },
    "eaten": {
        "train": [
            "Which cheeses were eaten in between?",
            "What cheese colours did the swarm eat?",
            "Which cheeses disappeared between the two grids?",
            "List the cheeses that got eaten.",
        ],
        "held": [
            "Which colours of cheese were consumed from the first grid to the second?",
            "What did the robots eat?",
        ],
    },
}
SCENE = "<scene>"


def colours(names):
    """Canonical answer text: palette order, each colour once, or "none"."""
    found = sorted(set(names), key=COLOUR_NAMES.index)
    return ", ".join(found) if found else "none"


def facts(store, e, t0, t1):
    """(colours present at t1, colours eaten between t0 and t1) of a stored episode."""
    alive0, alive1 = store["cheese_alive"][e, t0], store["cheese_alive"][e, t1]
    colour = store["cheese_color"][e]
    present = [COLOUR_NAMES[int(c)] for c, a in zip(colour, alive1) if a and c >= 0]
    eaten = [COLOUR_NAMES[int(c)] for c, a, b in zip(colour, alive0, alive1) if a and not b and c >= 0]
    return colours(present), colours(eaten)


def sample(stores, labels, n, rng, device, split="train"):
    """n questions: (kind, question, observations, answer), observations a list of (C, G, G)."""
    items = []
    for _ in range(n):
        label = labels[rng.randrange(len(labels))]
        store = stores[label.get("store", 0)]
        e = label["e"]
        steps = store["mass"].shape[1]
        kind = rng.choice(("present", "eaten"))
        if kind == "eaten" and rng.random() < 0.5:  # the whole episode: most short spans eat nothing
            t0, t1 = 0, steps - 1
        else:
            t0 = rng.randrange(steps)
            t1 = rng.randrange(t0, steps)
        present, eaten = facts(store, e, t0, t1)
        question = rng.choice(PHRASINGS[kind][split])
        frames = [t1] if kind == "present" else [t0, t1]
        world = world_at(store, torch.tensor([e] * len(frames)), torch.tensor(frames), device)
        obs = list(observe(world))
        items.append((kind, question, obs, present if kind == "present" else eaten))
    return items


def prompt_text(tokenizer, kind, question):
    """The chat prompt with SCENE placeholders, up to the assistant's answer."""
    scene = f"Grid: {SCENE}" if kind == "present" else f"First grid: {SCENE}\nSecond grid: {SCENE}"
    return tokenizer.apply_chat_template([{"role": "user", "content": f"{scene}\n{question}"}], add_generation_prompt=True, enable_thinking=False, tokenize=False)


def embed(policy, tokenizer, text, obs, device):
    """Input embeddings of text with each SCENE placeholder replaced by the next observation's
    scene tokens (the policy's own scene embedding)."""
    table = policy.lm.get_input_embeddings()
    pieces = text.split(SCENE)
    parts = []
    for k, piece in enumerate(pieces):
        if piece:
            parts.append(table(tokenizer(piece, return_tensors="pt", add_special_tokens=False).input_ids.to(device))[0])
        if k < len(obs):
            parts.append(policy.scene_tokens(obs[k].unsqueeze(0).to(device))[0])
    return torch.cat(parts)


def loss(policy, tokenizer, items, device):
    """Mean cross-entropy of the answer tokens (and the end-of-turn token)."""
    table = policy.lm.get_input_embeddings()
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    seqs, targets = [], []
    for kind, question, obs, answer in items:
        context = embed(policy, tokenizer, prompt_text(tokenizer, kind, question), obs, device)
        ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist() + [end]
        answer_ids = torch.tensor(ids, device=device)
        seqs.append(torch.cat([context, table(answer_ids[:-1])]))
        targets.append(torch.cat([torch.full((len(context) - 1,), -100, device=device), answer_ids]))
    length = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), length, seqs[0].shape[-1], dtype=seqs[0].dtype, device=device)
    y = torch.full((len(seqs), length), -100, dtype=torch.long, device=device)
    mask = torch.zeros(len(seqs), length, dtype=torch.long, device=device)
    for i, (s, t) in enumerate(zip(seqs, targets)):  # right padding
        x[i, : len(s)] = s
        y[i, : len(t)] = t
        mask[i, : len(s)] = 1
    logits = policy.lm(inputs_embeds=x, attention_mask=mask).logits
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)


@torch.no_grad()
def answer(policy, tokenizer, text, obs, device, max_new_tokens=24):
    """Greedy reply to a prompt with scene placeholders."""
    x = embed(policy, tokenizer, text, obs, device).unsqueeze(0)
    out = policy.lm.generate(inputs_embeds=x, attention_mask=torch.ones(x.shape[:2], dtype=torch.long, device=device), max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def parsed(reply):
    """The set of colour names in a reply (empty for "none" or anything without colours)."""
    words = reply.lower().replace(",", " ").replace(".", " ").split()
    return {w for w in words if w in COLOUR_NAMES}


@torch.no_grad()
def evaluate(policy, tokenizer, stores, labels, device, n=48, seed=0):
    """Exact-set accuracy per kind on held-out episodes, with trained and with held-out phrasings,
    and a few free-form replies to look at."""
    policy.eval()
    out = {}
    for split in ("train", "held"):
        rng = random.Random(seed)
        items = sample(stores, labels, n, rng, device, split)
        for kind in ("present", "eaten"):
            mine = [it for it in items if it[0] == kind]
            right = [parsed(answer(policy, tokenizer, prompt_text(tokenizer, k, q), obs, device)) == parsed(a) for k, q, obs, a in mine]
            out[f"{kind}_{split}"] = round(sum(right) / max(len(right), 1), 3)
    rng = random.Random(seed + 1)
    _, _, obs, _ = sample(stores, labels, 1, rng, device)[0]
    free = tokenizer.apply_chat_template([{"role": "user", "content": f"First grid: {SCENE}\nSecond grid: {SCENE}\nDescribe what happened between these two grids."}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
    frames = obs if len(obs) == 2 else obs * 2
    out["free_form"] = answer(policy, tokenizer, free, frames, device, max_new_tokens=48)
    policy.train()
    return out
