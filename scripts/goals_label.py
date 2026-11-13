"""Label library goals with a small vision model (Moondream 2): a short caption of the rendered goal and
whether it still looks like its source label. NOT run by the goal-library builder; the manager schedules it.

    # local (weights are downloaded by transformers on first use; device defaults to mps)
    nice -n 10 .venv/bin/python scripts/goals_label.py stabletext2brick --backend local --device mps --limit 2000
    # hosted API (reads MOONDREAM_API_KEY from the environment)
    MOONDREAM_API_KEY=... .venv/bin/python scripts/goals_label.py hack337 --backend api
    # plumbing check without any model
    .venv/bin/python scripts/goals_label.py iglu_multiturn --backend stub --limit 5

For each goal it renders two views (goal_library.render_views), asks for `caption(length="short")`, and asks
"Does this look like <source text or label>? Answer yes or no." (the question text is the source's caption,
instruction, title or part label). Output: data/goal_library/labels_<source>.jsonl with id, caption, looks_like
(true/false/null), answer, backend. Resumable: ids already present are skipped. Moondream sees a robot-grey
isometric picture; the question is about shape, so judge yes rates against the label's difficulty.
"""

import argparse
import json
import os
from pathlib import Path

from beebot.vla.voxel import goal_library as G

MODEL_ID = "vikhyatk/moondream2"
QUESTION = "Does this look like {what}? Answer yes or no."


class Stub:
    name = "stub"

    def caption(self, image):
        return "a grey block structure"

    def ask(self, image, question):
        return "yes"


class Local:
    """transformers backend: AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)."""

    name = "local"

    def __init__(self, device="mps", revision=None):
        import torch
        from transformers import AutoModelForCausalLM

        kwargs = {"trust_remote_code": True}
        if revision:
            kwargs["revision"] = revision
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16 if device != "cpu" else torch.float32, **kwargs).to(device)
        self.model.eval()

    def caption(self, image):
        return self.model.caption(image, length="short")["caption"]

    def ask(self, image, question):
        return self.model.query(image, question)["answer"]


class Api:
    """Moondream cloud API (pip package `moondream`), key from MOONDREAM_API_KEY."""

    name = "api"

    def __init__(self):
        import moondream as md

        key = os.environ.get("MOONDREAM_API_KEY")
        if not key:
            raise SystemExit("MOONDREAM_API_KEY is not set")
        self.model = md.vl(api_key=key)

    def caption(self, image):
        return self.model.caption(image, length="short")["caption"]

    def ask(self, image, question):
        return self.model.query(image, question)["answer"]


def make_backend(args):
    if args.backend == "stub":
        return Stub()
    if args.backend == "api":
        return Api()
    return Local(device=args.device, revision=args.revision)


def yes_no(answer):
    text = answer.strip().lower()
    if text.startswith("yes"):
        return True
    if text.startswith("no"):
        return False
    return None


def what_of(line):
    """The phrase the question asks about: the source text, else the part label or category id."""
    return (line["text"] or line["label"] or "").strip().rstrip(".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--backend", choices=("local", "api", "stub"), default="local")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--revision", default=None, help="pin a moondream2 revision for the local backend")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=str(G.LIBRARY))
    args = parser.parse_args()
    lines, cells = G.load(args.source)
    path = Path(args.out) / f"labels_{args.source}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(row)["id"] for row in open(path)}
    backend = make_backend(args)
    count = 0
    with open(path, "a") as handle:
        for line, cell in zip(lines, cells):
            if line["id"] in done:
                continue
            if args.limit and count >= args.limit:
                break
            image = G.render_views(cell)
            row = {"id": line["id"], "caption": backend.caption(image), "backend": backend.name}
            what = what_of(line)
            if what:
                answer = backend.ask(image, QUESTION.format(what=what))
                row["answer"] = answer
                row["looks_like"] = yes_no(answer)
            else:
                row["answer"] = None
                row["looks_like"] = None
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            count += 1
    print(f"labelled {count} goals of {args.source} -> {path}")


if __name__ == "__main__":
    main()
