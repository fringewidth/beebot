"""Forgetting measures for the LM policy, scored with the LM's own head on the same weights:
perplexity on held-out general text, a multiple-choice probe of the world knowledge our
instructions lean on, and GSM8K accuracy on a fixed subset of the test split.
"""

import itertools
import random
import re

import torch
import torch.nn.functional as F

PRIMARY = ["red", "blue", "yellow"]
SECONDARY = ["green", "orange", "purple"]
WARM = ["red", "orange", "yellow"]
COOL = ["blue", "green", "purple"]
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left", "top left": "bottom right", "top right": "bottom left", "bottom left": "top right", "bottom right": "top left"}
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def probe_questions(seed=0):
    """About 120 (question, options, answer index): colours, categories, negation, spatial words,
    counting and ordering, the knowledge our instructions lean on."""
    rng = random.Random(seed)
    qs = []
    for c in PRIMARY + SECONDARY:
        qs.append((f"Is {c} a primary colour (in the red, yellow, blue model)?", ["yes", "no"], 0 if c in PRIMARY else 1))
        qs.append((f"Is {c} a warm colour or a cool colour?", ["warm", "cool"], 0 if c in WARM else 1))
    for a, b in itertools.permutations(PRIMARY, 2):
        mix = {frozenset(["red", "blue"]): "purple", frozenset(["red", "yellow"]): "orange", frozenset(["blue", "yellow"]): "green"}[frozenset([a, b])]
        options = rng.sample(SECONDARY, 3)
        qs.append((f"Mixing {a} and {b} paint gives which colour?", options, options.index(mix)))
    for word, opposite in OPPOSITE.items():
        options = [opposite] + rng.sample([w for w in OPPOSITE if w not in (word, opposite)], 2)
        rng.shuffle(options)
        qs.append((f"What is the opposite side of the {word} of a square?", options, options.index(opposite)))
    colours = PRIMARY + SECONDARY
    for _ in range(30):
        shown = rng.sample(colours, rng.randint(3, 5))
        excluded = rng.choice(shown)
        keep = [c for c in shown if c != excluded]
        target = rng.choice(keep)
        options = [target, excluded]
        rng.shuffle(options)
        qs.append((f"The cheeses are {', '.join(shown)}. You must eat every cheese except the {excluded} one. Which of these should you eat?", options, options.index(target)))
        n = sum(c in PRIMARY for c in shown)
        options = rng.sample([v for v in range(6) if v != n], 2) + [n]
        rng.shuffle(options)
        qs.append((f"How many of these colours are primary colours: {', '.join(shown)}?", [NUMBER_WORDS[v] for v in options], options.index(n)))
    for _ in range(20):
        pair = rng.sample(colours, 2)
        if (pair[0] in WARM) == (pair[1] in WARM):
            continue
        cool_first = pair[0] if pair[0] in COOL else pair[1]
        qs.append((f"Sorting from cool to warm, which comes first: {pair[0]} or {pair[1]}?", pair, pair.index(cool_first)))
    for word, n in (("pair", 2), ("dozen", 12), ("trio", 3), ("quartet", 4), ("half a dozen", 6)):
        options = rng.sample([v for v in (2, 3, 4, 6, 8, 10, 12) if v != n], 2) + [n]
        rng.shuffle(options)
        qs.append((f"How many things are in a {word}?", [str(v) for v in options], options.index(n)))
    return qs


@torch.no_grad()
def mc_accuracy(lm, tokenizer, items, device):
    """Share answered right over (prompt, options, answer index): the option with the highest
    mean log-likelihood per token after the prompt."""
    right = 0
    for prompt, options, answer in items:
        n_prompt = len(tokenizer(prompt).input_ids)
        scores = []
        for option in options:
            ids = tokenizer(prompt + " " + option, return_tensors="pt").input_ids.to(device)
            logits = lm(input_ids=ids).logits[0, :-1].float()
            logp = F.log_softmax(logits, -1).gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)
            scores.append(logp[n_prompt - 1 :].mean().item())
        right += int(max(range(len(options)), key=scores.__getitem__) == answer)
    return right / max(len(items), 1)


def probe_items(seed=0):
    return [(f"Question: {q}\nAnswer:", options, answer) for q, options, answer in probe_questions(seed)]


def benchmark_items(n=150, seed=0):
    """Fixed subsets of MMLU (knowledge), ARC-Challenge (reasoning) and HellaSwag (commonsense
    language), as (prompt, options, answer index), from the Hugging Face Hub."""
    from datasets import load_dataset

    rng = random.Random(seed)
    out = {}
    mmlu = list(load_dataset("cais/mmlu", "all", split="test"))
    out["mmlu"] = [(f"Question: {r['question']}\nAnswer:", r["choices"], r["answer"]) for r in rng.sample(mmlu, n)]
    arc = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"))
    out["arc"] = [(f"Question: {r['question']}\nAnswer:", r["choices"]["text"], r["choices"]["label"].index(r["answerKey"])) for r in rng.sample(arc, n) if r["answerKey"] in r["choices"]["label"]]
    hella = list(load_dataset("Rowan/hellaswag", split="validation"))
    out["hellaswag"] = [(r["ctx"], r["endings"], int(r["label"])) for r in rng.sample(hella, n)]
    return out


@torch.no_grad()
def perplexity(lm, chunks, device, batch=4):
    """Perplexity over token chunks (list of 1-D id tensors of equal length)."""
    total, count = 0.0, 0
    for i in range(0, len(chunks), batch):
        ids = torch.stack(chunks[i : i + batch]).to(device)
        logits = lm(input_ids=ids).logits[:, :-1].float()
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")
        total += nll.item()
        count += ids[:, 1:].numel()
    return float(torch.exp(torch.tensor(total / count)))


def gsm8k_answer(text):
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return numbers[-1].replace(",", "").rstrip(".") if numbers else None


@torch.no_grad()
def gsm8k_accuracy(lm, tokenizer, problems, device, batch=16, max_new_tokens=320):
    """Greedy, no thinking, final number compared with the reference after '####'."""
    right = 0
    tokenizer.padding_side = "left"
    for i in range(0, len(problems), batch):
        part = problems[i : i + batch]
        prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": p["question"] + "\nSolve step by step, then give the final number on its own line."}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for p in part
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        out = lm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        for p, ids in zip(part, out[:, enc.input_ids.shape[1] :]):
            got = gsm8k_answer(tokenizer.decode(ids, skip_special_tokens=True))
            ref = p["answer"].split("####")[-1].strip().replace(",", "")
            right += int(got is not None and float(got) == float(ref))
    tokenizer.padding_side = "right"
    return right / len(problems)
