"""Text pre-flight: can one pooled embedding carry what the policy needs to read from a command?

For each pair type, sentences come from templates that vary one labelled attribute and some
nuisance slots. A linear probe is trained on the embeddings of some templates and tested on
held-out templates (new surface forms, same words). A type passes at 90% held-out accuracy.
A synonym check (train on base words, test on synonyms the probe never saw) is reported
alongside but doesn't gate.

    uv run python -m beebot.vla.preflight [--model NAME]
"""

import argparse
import itertools
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from beebot.vla.text import DEFAULT_MODEL, TextEncoder

PASS = 0.9
COLORS = ["red", "green", "blue", "yellow"]
REGIONS = ["top left", "top", "top right", "left", "center", "right", "bottom left", "bottom", "bottom right"]
VERBS = {"push": "push", "eat": "eat", "surround": "surround", "avoid": "stay away from"}
VERB_SYNONYMS = {"push": "shove", "eat": "devour", "surround": "encircle", "avoid": "keep clear of"}
QUANTITIES = {"all": "all of", "half": "half of", "quarter": "a quarter of"}


def colour_items():
    train = ["{v} the {c} cheese", "{v} the {c} cheese toward the {r}", "the swarm should {v} the {c} cheese", "go and {v} the {c} one"]
    test = ["make sure you {v} the {c} cheese", "your job is to {v} that {c} cheese near the {r}"]
    return [
        (split, t.format(v=v, c=c, r=r), c)
        for split, templates in (("train", train), ("test", test))
        for t in templates
        for v, c, r in itertools.product(VERBS.values(), COLORS, REGIONS[::2])
    ]


def direction_items():
    train = ["go to the {r}", "gather in the {r}", "move the swarm to the {r}", "push the {c} cheese to the {r}"]
    test = ["regroup at the {r} of the map", "the {r} is where everyone should end up"]
    return [
        (split, t.format(r=r, c=c), r)
        for split, templates in (("train", train), ("test", test))
        for t in templates
        for r, c in itertools.product(REGIONS, COLORS)
    ]


def verb_items(words):
    train = ["{v} the {c} cheese", "{v} the cheese in the {r}", "now {v} the {c} cheese please"]
    test = ["the swarm must {v} the {c} cheese", "{v} whichever cheese is in the {r}"]
    return [
        (split, t.format(v=words[label], c=c, r=r), label)
        for split, templates in (("train", train), ("test", test))
        for t in templates
        for label, c, r in itertools.product(VERBS, COLORS, REGIONS[::2])
    ]


def object_items():
    train = ["go to the {c} {o}", "move next to the {c} {o}", "head for the {o} in the {r}"]
    test = ["the {c} {o} is your destination", "end up beside that {o} near the {r}"]
    return [
        (split, t.format(c=c, o=o, r=r), o)
        for split, templates in (("train", train), ("test", test))
        for t in templates
        for o, c, r in itertools.product(["cheese", "box"], COLORS, REGIONS[::2])
    ]


def role_items(role):
    # Both orders appear in training, so naming order alone can't predict the role.
    train = ["push the {a} cheese into the {b} box", "get the {a} cheese to the {b} box", "into the {b} box, push the {a} cheese", "fill the {b} box with the {a} cheese"]
    test = ["the {a} cheese should end up in the {b} box", "into the {b} box with the {a} cheese"]
    return [
        (split, t.format(a=a, b=b), a if role == "cheese" else b)
        for split, templates in (("train", train), ("test", test))
        for t in templates
        for a, b in itertools.permutations(COLORS, 2)
    ]


def quantity_items():
    train = ["send {q} the swarm to the {r}", "move {q} the robots to the {r}", "{q} the swarm should gather at the {r}"]
    test = ["I want {q} your swarm in the {r}", "put {q} the robots over at the {r}"]
    return [
        (split, t.format(q=QUANTITIES[label], r=r), label)
        for split, templates in (("train", train), ("test", test))
        for t in templates
        for label, r in itertools.product(QUANTITIES, REGIONS)
    ]


def synonym_items():
    base = verb_items(VERBS)
    train = [("train", text, label) for split, text, label in base]
    test = [("test", text, label) for split, text, label in verb_items(VERB_SYNONYMS)]
    return train + test


PAIR_TYPES = {
    "colour": colour_items,
    "direction": direction_items,
    "verb": lambda: verb_items(VERBS),
    "object": object_items,
    "role_cheese": lambda: role_items("cheese"),
    "role_box": lambda: role_items("box"),
    "quantity": quantity_items,
}
INFO_TYPES = {"verb_synonyms": synonym_items}


def probe(x_train, y_train, x_test, y_test, n_classes, steps=300, weight_decay=1e-3):
    """Multinomial logistic regression on standardized features; returns held-out accuracy."""
    mean, std = x_train.mean(0), x_train.std(0) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    weight = torch.zeros(x_train.shape[1], n_classes, requires_grad=True)
    bias = torch.zeros(n_classes, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(x_train @ weight + bias, y_train) + weight_decay * weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return (x_test @ weight + bias).argmax(1).eq(y_test).float().mean().item()


def run(model_name, out_path):
    random.seed(0)
    encoder = TextEncoder(model_name)
    results = {"model": model_name, "pass_at": PASS, "types": {}}
    for name, make in list(PAIR_TYPES.items()) + list(INFO_TYPES.items()):
        items = make()
        labels = sorted({label for _, _, label in items})
        index = {label: i for i, label in enumerate(labels)}
        texts = [text for _, text, _ in items]
        x = encoder.embed(texts)
        y = torch.tensor([index[label] for _, _, label in items])
        is_train = torch.tensor([split == "train" for split, _, _ in items])
        accuracy = probe(x[is_train], y[is_train], x[~is_train], y[~is_train], len(labels))
        gating = name in PAIR_TYPES
        results["types"][name] = {
            "accuracy": round(accuracy, 3),
            "n_train": int(is_train.sum()),
            "n_test": int((~is_train).sum()),
            "classes": len(labels),
            "gating": gating,
            "pass": accuracy >= PASS if gating else None,
        }
        print(f"{name:14s} held-out accuracy {accuracy:.3f} ({len(labels)} classes){'' if gating else '  [info only]'}")
    results["all_pass"] = all(r["pass"] for r in results["types"].values() if r["gating"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    print("all gating types pass" if results["all_pass"] else "some gating types fail: use the token-level fallback")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="runs/vla-preflight/pooled.json")
    args = parser.parse_args()
    run(args.model, args.out)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------
# Role-binding deep dive: more templates, per-template accuracy, and the token-level fallback.

# Half the training templates name the box first, so naming order alone can't predict the role.
ROLE_TRAIN = [
    "push the {a} cheese into the {b} box",
    "move the {a} cheese inside the {b} box",
    "get the {a} cheese to the {b} box",
    "take the {a} cheese over to the {b} box",
    "roll the {a} cheese until it sits in the {b} box",
    "steer the {a} cheese toward the {b} box",
    "into the {b} box, push the {a} cheese",
    "fill the {b} box with the {a} cheese",
    "the {b} box needs the {a} cheese inside it",
    "at the {b} box, drop off the {a} cheese",
    "make the {b} box hold the {a} cheese",
    "toward the {b} box goes the {a} cheese",
]
ROLE_TEST = [
    "the {a} cheese should end up in the {b} box",
    "deliver the {a} cheese to the {b} box",
    "I want the {a} cheese parked in the {b} box",
    "into the {b} box with the {a} cheese",
    "the {b} box is where the {a} cheese goes",
    "put inside the {b} box the {a} cheese",
]


class AttentionPool(torch.nn.Module):
    """A learned query attends over projected token states; two linear heads read the pooled vector."""

    def __init__(self, dim, n_classes):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(dim) / dim**0.5)
        self.key = torch.nn.Linear(dim, dim)
        self.heads = torch.nn.ModuleList([torch.nn.Linear(dim, n_classes) for _ in range(2)])

    def forward(self, tokens, mask):
        scores = (self.key(tokens) @ self.query).masked_fill(~mask, float("-inf"))
        pooled = (torch.softmax(scores, dim=1).unsqueeze(-1) * tokens).sum(dim=1)
        return [head(pooled) for head in self.heads]


def role_deep_dive(model_name, out_path, projected_dim=128, seed=0, device="cpu"):
    torch.manual_seed(seed)
    encoder = TextEncoder(model_name, device=device)
    index = {c: i for i, c in enumerate(COLORS)}
    items = [
        (split, ti, t.format(a=a, b=b), index[a], index[b])
        for split, templates in (("train", ROLE_TRAIN), ("test", ROLE_TEST))
        for ti, t in enumerate(templates)
        for a, b in itertools.permutations(COLORS, 2)
    ]
    texts = [text for _, _, text, _, _ in items]
    is_train = torch.tensor([s == "train" for s, _, _, _, _ in items])
    y_cheese = torch.tensor([a for *_, a, _ in items])
    y_box = torch.tensor([b for *_, b in items])
    test_template = torch.tensor([ti for s, ti, *_ in items])

    pooled = encoder.embed(texts)
    report = {"model": model_name, "pooled": {}, "token_level": {}}
    for role, y in (("cheese", y_cheese), ("box", y_box)):
        accuracy = probe(pooled[is_train], y[is_train], pooled[~is_train], y[~is_train], len(COLORS))
        report["pooled"][role] = round(accuracy, 3)

    states = encoder.token_states(texts)
    projection = torch.randn(states[0].shape[1], projected_dim) / states[0].shape[1] ** 0.5
    longest = max(s.shape[0] for s in states)
    tokens = torch.zeros(len(states), longest, projected_dim)
    mask = torch.zeros(len(states), longest, dtype=torch.bool)
    for i, s in enumerate(states):
        tokens[i, : s.shape[0]] = s @ projection
        mask[i, : s.shape[0]] = True
    tokens = tokens / tokens[mask].std()
    model = AttentionPool(projected_dim, len(COLORS))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    for _ in range(1500):
        logits = model(tokens[is_train], mask[is_train])
        loss = F.cross_entropy(logits[0], y_cheese[is_train]) + F.cross_entropy(logits[1], y_box[is_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model(tokens[~is_train], mask[~is_train])
    for role, y, out in (("cheese", y_cheese, logits[0]), ("box", y_box, logits[1])):
        correct = out.argmax(1).eq(y[~is_train])
        per_template = {
            ROLE_TEST[ti]: round(correct[test_template[~is_train] == ti].float().mean().item(), 3) for ti in range(len(ROLE_TEST))
        }
        report["token_level"][role] = {"accuracy": round(correct.float().mean().item(), 3), "per_template": per_template}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
