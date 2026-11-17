"""Does the policy's language side know what the swarm just did? The LM policy rolls out tasks;
then the same model (its LM head, LoRA on, and the original LM with LoRA off) is shown the
episode's start and end frames as scene tokens inside a text prompt and asked yes/no questions
about them, scored on p(yes) - p(no) against what happened:

    eaten:   "Was the red cheese eaten?"   (start and end frames; colours present at the start)
    present: "Is there a red cheese?"      (end frame; every training colour)

Controls: the same questions with another episode's frames (so the answers cannot depend on the
scene), and with the frames replaced by a text description of the cheeses (the language-only
upper bound). Then linear probes (logistic regression, split by episode) on the hidden state of
the question's last token, against the same probe with the shuffled frames (what the question
text alone predicts): is the fact in the representation even when the head does not say it?

In training, text never attends to scene tokens (lm_policy.LMPolicy.hidden), and the policy sees
one frame at a time, so the zero-shot answers are expected near chance.

    uv run python -m beebot.vla.self_report --init_run RUN --checkpoint policy_latest.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from beebot.vla import tasks as T
from beebot.vla.lm_policy import PATCH, lora_disabled
from beebot.vla.lm_rl import RLConfig, lm_sampler, load_policy, tokens

SPLITS = ("test_logic", "logic", "test_runner", "test_parallel")


def rollouts(policy, tokenizer, n, steps, guidance, device, seed, batch=24):
    """Per episode: start and end observations (C, G, G), the colours present at the start, and
    the colours eaten."""
    rng = np.random.default_rng(seed)
    sample = lm_sampler(policy, guidance, T.SIZE // PATCH)
    condition = lambda texts: tokens(tokenizer, texts, device)
    episodes = []
    per_split = n // len(SPLITS)
    for split in SPLITS:
        pairs = [T.sample_task(rng, "test" if split.startswith("test_") else split, family=split[5:] if split.startswith("test_") else None) for _ in range(per_split)]
        for k in range(0, len(pairs), batch):
            chunk = pairs[k : k + batch]
            scenes, task_list = [s for s, _ in chunk], [t for _, t in chunk]
            torch.manual_seed(seed + k)
            _, kept, outcomes = T.run(sample, condition, scenes, task_list, steps, device, keep_steps=(0, steps - 1), outcomes=True)
            for i, (scene, task) in enumerate(chunk):
                start = T.names(scene, scene["cheese_colours"])
                eaten = outcomes[i][1]
                episodes.append({"split": split, "text": task.text, "start_obs": kept[0][0][i], "end_obs": kept[1][0][i], "start": start, "eaten": eaten, "end": [c for c in start if c not in eaten] if len(set(start)) == len(start) else None})
    return [e for e in episodes if e["end"] is not None]  # a repeated colour makes "the red cheese" ambiguous


def questions(episodes):
    """(episode index, kind, colour, truth) for every question."""
    out = []
    for j, e in enumerate(episodes):
        for c in e["start"]:
            out.append((j, "eaten", c, c in e["eaten"]))
        for c in T.TRAIN_COLOURS:
            out.append((j, "present", c, c in e["end"]))
    return out


def describe(colours):
    return "cheeses: " + (", ".join(colours) if colours else "none")


def embed(policy, tokenizer, pieces, device):
    """One sequence of input embeddings from text strings and observation tensors (scene tokens)."""
    table = policy.lm.get_input_embeddings()
    parts = []
    for p in pieces:
        if isinstance(p, str):
            ids = tokenizer(p, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            parts.append(table(ids)[0])
        else:
            parts.append(policy.scene_tokens(p.unsqueeze(0).to(device))[0])
    return torch.cat(parts)


def prompt(kind, colour, before, after):
    """Prompt pieces: before/after are observations or text descriptions."""
    question = f"Was the {colour} cheese eaten?" if kind == "eaten" else f"Is there a {colour} cheese now?"
    pieces = ["A swarm of robots moves on a grid with coloured cheeses.\nBefore: "]
    pieces += [before, "\nAfter: ", after] if kind == "eaten" else ["(not shown)", "\nAfter: ", after]
    return pieces + [f"\nQuestion: {question} Answer yes or no.\nAnswer:"]


@torch.no_grad()
def answers(policy, tokenizer, episodes, qs, condition, device, batch=32):
    """Per question: p(yes) - p(no) from the LM head at the last position, and that position's
    final hidden state. condition: "scene", "shuffled" (another episode's frames) or "text"."""
    yes = [tokenizer(w, add_special_tokens=False).input_ids[0] for w in (" yes", " Yes")]
    no = [tokenizer(w, add_special_tokens=False).input_ids[0] for w in (" no", " No")]
    other = np.roll(np.arange(len(episodes)), len(episodes) // 2)
    margins, states = [], []
    for k in range(0, len(qs), batch):
        seqs = []
        for j, kind, colour, _ in qs[k : k + batch]:
            e = episodes[other[j]] if condition == "shuffled" else episodes[j]
            if condition == "text":
                before, after = describe(e["start"]), describe(e["end"])
            else:
                before, after = e["start_obs"], e["end_obs"]
            seqs.append(embed(policy, tokenizer, prompt(kind, colour, before, after), device))
        length = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), length, seqs[0].shape[-1], dtype=seqs[0].dtype, device=device)
        mask = torch.zeros(len(seqs), length, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):  # left padding: the last position is every row's answer slot
            x[i, length - len(s) :] = s
            mask[i, length - len(s) :] = 1
        positions = (mask.cumsum(1) - 1).clamp(min=0)
        out = policy.lm(inputs_embeds=x, attention_mask=mask, position_ids=positions, output_hidden_states=True)
        logp = out.logits[:, -1].float().log_softmax(-1)
        margins += (logp[:, yes].logsumexp(-1) - logp[:, no].logsumexp(-1)).tolist()
        states.append(out.hidden_states[-1][:, -1].float().cpu())
    return np.array(margins), torch.cat(states).numpy()


def auc(scores, truth):
    """Area under the ROC curve: the chance a true question outscores a false one (0.5: chance)."""
    truth = np.asarray(truth, dtype=bool)
    pos, neg = scores[truth], scores[~truth]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def probe(features, truth, groups, seed=0, l2=1e-2):
    """Held-out AUC of a logistic regression (L2, full-batch L-BFGS), split by episode (a fifth
    held out)."""
    rng = np.random.default_rng(seed)
    held = set(rng.choice(np.unique(groups), size=max(1, len(np.unique(groups)) // 5), replace=False).tolist())
    test = np.array([g in held for g in groups])
    x = torch.tensor(features, dtype=torch.float64)
    y = torch.tensor(truth, dtype=torch.float64)
    mean, std = x[~test].mean(0), x[~test].std(0).clamp(min=1e-6)
    x = (x - mean) / std
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    train_x, train_y = x[torch.tensor(~test)], y[torch.tensor(~test)]
    optimizer = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(train_x @ w + b, train_y) + l2 * (w * w).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return auc((x[torch.tensor(test)] @ w + b).numpy(), truth[test])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_run", required=True)
    parser.add_argument("--checkpoint", default="policy_latest.pt")
    parser.add_argument("--episodes", type=int, default=192)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--out", default="logs/self-report.json")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    policy, tokenizer = load_policy(RLConfig(init_run=args.init_run, checkpoint=args.checkpoint), device)
    policy.eval()
    episodes = rollouts(policy, tokenizer, args.episodes, args.steps, args.guidance, device, args.seed)
    qs = questions(episodes)
    truth = np.array([q[3] for q in qs])
    groups = np.array([q[0] for q in qs])
    kinds = np.array([q[1] for q in qs])
    print(f"{len(episodes)} episodes, {len(qs)} questions ({truth.mean():.2f} true)", flush=True)
    result = {"init_run": args.init_run, "checkpoint": args.checkpoint, "episodes": len(episodes), "questions": len(qs)}
    for model in ("policy", "base"):
        for condition in ("scene", "shuffled", "text"):
            if model == "base":
                with lora_disabled(policy.lm):
                    margins, states = answers(policy, tokenizer, episodes, qs, condition, device)
            else:
                margins, states = answers(policy, tokenizer, episodes, qs, condition, device)
            for kind in ("eaten", "present"):
                sel = kinds == kind
                row = {"zero_shot_auc": round(auc(margins[sel], truth[sel]), 3), "says_yes": round(float((margins[sel] > 0).mean()), 3), "probe_auc": round(probe(states[sel], truth[sel], groups[sel]), 3)}
                result[f"{model}/{condition}/{kind}"] = row
                print(model, condition, kind, row, flush=True)
    for kind in ("eaten", "present"):
        result[f"base_rate/{kind}"] = round(float(truth[kinds == kind].mean()), 3)
    Path(args.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
