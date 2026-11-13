"""Label library goals with a small vision model (Moondream 2): a short caption of the rendered goal and
whether it still looks like its source text, with a wrong-text control on every goal.

    # local (weights come from the Hugging Face cache; device mps; pinned revision below)
    nice -n 5 .venv/bin/python scripts/goals_label.py stabletext2brick --backend local --method yesno --per-source 500
    # calibration across sources: both methods, own and control text, no label files
    nice -n 5 .venv/bin/python scripts/goals_label.py --calibrate 200 --sources stabletext2brick,gen_medium,hack337
    # speed test (captions and questions timed separately)
    nice -n 5 .venv/bin/python scripts/goals_label.py --speed 20 stabletext2brick
    # hosted API (reads MOONDREAM_API_KEY from the environment)
    MOONDREAM_API_KEY=... .venv/bin/python scripts/goals_label.py hack337 --backend api
    # plumbing check without any model
    .venv/bin/python scripts/goals_label.py iglu_multiturn --backend stub --per-source 5

Per goal: two views (goal_library.render_views), `caption(length="short")`, and a question about the goal's own
text and the same question about the text of a different goal (the wrong-text control, drawn from another
generator or category of the same source, else any other text). Methods:

1. yesno: "Does this look like <text>? Answer yes or no." for own and for control. A good question has a high own
   yes rate and a low control yes rate.
1. mc: the picture with 4 candidate texts (the true one in a random slot, 3 other goals' texts), "which letter
   fits best". Accuracy against chance 0.25.

Sampling: up to --per-source goals, spread over cell-count bins and over generators (or part labels or kinds),
round robin, fixed seed. Output: data/goal_library/labels_<source>.jsonl; resumable (ids present are skipped).
Progress: one JSON line per 50 goals appended to logs/moondream-label.out and printed, plus a final summary.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from beebot.vla.voxel import goal_library as G

MODEL_ID = "vikhyatk/moondream2"
REVISION = "5d6c926f44e26b07957b0dd315bbedcb4c17a5fe"
QUESTION = "Does this look like {what}? Answer yes or no."
CHOICE = "Which description fits this picture best?\n{options}\nAnswer with the letter only."
BINS = (4, 8, 16, 32, 64, 10**9)
LETTERS = "ABCD"
LOG = "logs/moondream-label.out"
SCALE = 9  # pixels per cell in render_views


class Stub:
    name = "stub"

    def encode(self, image):
        return image

    def caption(self, image):
        return "a grey block structure"

    def ask(self, image, question):
        return "A" if "Which description" in question else "yes"


class Local:
    """transformers backend: AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)."""

    name = "local"

    def __init__(self, device="mps", revision=REVISION):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.device = device
        dtype = torch.float16 if device != "cpu" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=revision, trust_remote_code=True, dtype=dtype).to(device)
        self.model.eval()

    def encode(self, image):
        return self.model.encode_image(image)

    def caption(self, image):
        return self.model.caption(image, length="short")["caption"]

    def ask(self, image, question):
        return self.model.query(image, question)["answer"]

    def memory_gb(self):
        return self.torch.mps.driver_allocated_memory() / 2**30


class Api:
    """Moondream cloud API (pip package `moondream`), key from MOONDREAM_API_KEY."""

    name = "api"

    def __init__(self):
        import moondream as md

        key = os.environ.get("MOONDREAM_API_KEY")
        if not key:
            raise SystemExit("MOONDREAM_API_KEY is not set")
        self.model = md.vl(api_key=key)

    def encode(self, image):
        return image

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
    """The phrase the question asks about: the source text, else the part label. Copyright tails (Hack337) and
    very long texts are cut."""
    text = (line.get("text") or line.get("label") or "").split("©")[0].strip()
    return text[:200].strip().rstrip(".")


def group_of(line):
    return line.get("generator") or line.get("label") or line.get("kind") or ""


def sample_indices(lines, per_source, seed=0):
    """Round robin over (group, cell bin) strata, so every generator and size range is represented."""
    rng = random.Random(seed)
    strata = {}
    for i, line in enumerate(lines):
        if not what_of(line):
            continue
        size = line["cells"]
        bin_index = next(k for k in range(len(BINS) - 1) if size < BINS[k + 1])
        strata.setdefault((group_of(line), bin_index), []).append(i)
    for members in strata.values():
        rng.shuffle(members)
    keys = sorted(strata)
    rng.shuffle(keys)
    chosen = []
    while len(chosen) < per_source and any(strata[key] for key in keys):
        for key in keys:
            if strata[key] and len(chosen) < per_source:
                chosen.append(strata[key].pop())
    return sorted(chosen)


def pick_others(lines, index, count, rng, pool):
    """`count` texts of other goals, from groups different from the goal's own where the pool has them."""
    own = lines[index]
    own_text = what_of(own)
    other_group = [j for j in pool if group_of(lines[j]) != group_of(own) and what_of(lines[j]) != own_text]
    anywhere = [j for j in pool if what_of(lines[j]) != own_text]
    candidates = other_group if len(other_group) >= count else anywhere
    picks = []
    seen = {own_text}
    for _ in range(50 * count):
        if len(picks) == count:
            break
        j = rng.choice(candidates)
        text = what_of(lines[j])
        if text not in seen:
            seen.add(text)
            picks.append(j)
    return picks


def ask_yesno(backend, enc, lines, index, other, timing=None):
    row = {}
    for tag, j in (("own", index), ("control", other)):
        start = time.time()
        answer = backend.ask(enc, QUESTION.format(what=what_of(lines[j])))
        if timing is not None:
            timing.append(time.time() - start)
        row[tag + "_answer"] = answer
        row[tag + "_yes"] = yes_no(answer)
    row["control_id"] = lines[other]["id"]
    return row


def ask_choice(backend, enc, lines, index, others, rng, timing=None):
    """Four candidate texts, the true one in a random slot; correct when the model picks it."""
    slot = rng.randrange(4)
    order = list(others[:3])
    order.insert(slot, index)
    options = "\n".join(f"{LETTERS[k]}. {what_of(lines[j])}" for k, j in enumerate(order))
    start = time.time()
    answer = backend.ask(enc, CHOICE.format(options=options))
    if timing is not None:
        timing.append(time.time() - start)
    letter = next((c for c in answer.strip().upper() if c in LETTERS), None)
    return {
        "answer": answer,
        "choice": letter,
        "true_letter": LETTERS[slot],
        "correct": letter == LETTERS[slot] if letter else None,
        "option_ids": [lines[j]["id"] for j in order],
    }


def label_goal(backend, cell, lines, index, pool, method, rng, timing=None):
    image = G.render_views(cell, scale=SCALE)
    enc = backend.encode(image)
    row = {"id": lines[index]["id"], "backend": backend.name, "method": method, "revision": REVISION}
    start = time.time()
    row["caption"] = backend.caption(enc)
    if timing is not None:
        timing.append(("caption", time.time() - start))
    others = pick_others(lines, index, 3, rng, pool)
    if method in ("yesno", "both"):
        times = []
        row.update(ask_yesno(backend, enc, lines, index, others[0], times))
        if timing is not None:
            timing.append(("yesno", sum(times) / len(times)))
    if method in ("mc", "both"):
        times = []
        sub = ask_choice(backend, enc, lines, index, others, rng, times)
        if timing is not None:
            timing.append(("mc", sum(times)))
        if method == "mc":
            row.update(sub)
        else:
            row["mc"] = sub
    return row


def rate(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def summarize(rows, method):
    out = {}
    if method in ("yesno", "both"):
        out["own_yes_rate"] = rate([r.get("own_yes") for r in rows])
        out["control_yes_rate"] = rate([r.get("control_yes") for r in rows])
    if method in ("mc", "both"):
        out["mc_accuracy"] = rate([(r.get("mc") or r).get("correct") for r in rows])
    return out


def emit(record, log):
    text = json.dumps(record)
    print(text, flush=True)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as handle:
        handle.write(text + "\n")


def run_source(backend, source, args, rng):
    lines, cells = G.load(source, args.library)
    chosen = sample_indices(lines, args.per_source, args.seed)
    pool = [i for i in range(len(lines)) if what_of(lines[i])]
    path = Path(args.root) / f"labels_{source}.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(row)["id"] for row in open(path)}
    rows = [json.loads(row) for row in open(path)] if path.exists() else []
    start = time.time()
    count = 0
    with open(path, "a") as handle:
        for i in chosen:
            if lines[i]["id"] in done:
                continue
            if args.limit and count >= args.limit:
                break
            if args.deadline and time.time() > args.deadline:
                emit({"source": source, "stopped": "deadline", "done": len(rows)}, args.log)
                return rows, True
            row = label_goal(backend, cells[i], lines, i, pool, args.method, rng)
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            rows.append(row)
            count += 1
            if count % 50 == 0:
                emit({"source": source, "done": len(rows), **summarize(rows, args.method), "s_per_goal": round((time.time() - start) / count, 2)}, args.log)
    emit({"source": source, "done": len(rows), "final": True, **summarize(rows, args.method), "s_per_goal": round((time.time() - start) / max(count, 1), 2)}, args.log)
    return rows, False


def speed(backend, args):
    lines, cells = G.load(args.source, args.library)
    chosen = sample_indices(lines, args.speed, args.seed)
    pool = [i for i in range(len(lines)) if what_of(lines[i])]
    rng = random.Random(args.seed)
    timing = []
    for i in chosen:
        label_goal(backend, cells[i], lines, i, pool, "both", rng, timing)
    out = {"speed_goals": len(chosen)}
    for kind in ("caption", "yesno", "mc"):
        values = [t for k, t in timing if k == kind]
        out[f"s_per_{kind}"] = round(sum(values) / len(values), 3)
    if hasattr(backend, "memory_gb"):
        out["peak_mps_gb"] = round(backend.memory_gb(), 2)
    emit(out, args.log)


def calibrate(backend, args):
    """Both methods on a mixed sample of the given sources; own and control rates. Wrong texts come from the
    union of the sources, so a control is usually another source or generator."""
    sources = args.sources.split(",")
    per = max(1, args.calibrate // len(sources))
    rng = random.Random(args.seed)
    rows = []
    path = Path(args.root) / f"calibration_scale{SCALE}.jsonl"
    with open(path, "w") as handle:
        for source in sources:
            lines, cells = G.load(source, args.library)
            pool = [i for i in range(len(lines)) if what_of(lines[i])]
            for i in sample_indices(lines, per, args.seed):
                row = label_goal(backend, cells[i], lines, i, pool, "both", rng)
                row["source"] = source
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
                handle.flush()
        emit({"calibration": True, "n": len(rows), **summarize(rows, "both")}, args.log)
        for source in sources:
            sub = [r for r in rows if r["source"] == source]
            emit({"calibration": True, "source": source, "n": len(sub), **summarize(sub, "both")}, args.log)


def main():
    global SCALE
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default=None)
    parser.add_argument("--backend", choices=("local", "api", "stub"), default="local")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--revision", default=REVISION, help="moondream2 revision for the local backend")
    parser.add_argument("--method", choices=("yesno", "mc", "both"), default="yesno")
    parser.add_argument("--per-source", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0, help="stop after this many new goals")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=int, default=0, help="time this many goals of SOURCE and exit")
    parser.add_argument("--calibrate", type=int, default=0, help="both methods on this many goals over --sources")
    parser.add_argument("--sources", default="")
    parser.add_argument("--hours", type=float, default=0, help="stop labelling after this many hours (resumable)")
    parser.add_argument("--library", default=str(G.LIBRARY), help="where the goal files are read from")
    parser.add_argument("--root", default=str(G.LIBRARY), help="where labels_<source>.jsonl and calibration.jsonl are written")
    parser.add_argument("--log", default=LOG)
    parser.add_argument("--scale", type=int, default=SCALE)
    args = parser.parse_args()
    SCALE = args.scale
    args.deadline = time.time() + 3600 * args.hours if args.hours else 0
    backend = make_backend(args)
    if args.speed:
        args.source = args.source or "stabletext2brick"
        speed(backend, args)
        return
    if args.calibrate:
        calibrate(backend, args)
        return
    sources = args.source.split(",")
    allrows = []
    for source in sources:
        rows, stopped = run_source(backend, source, args, random.Random(args.seed))
        allrows += rows
        if stopped:
            break
    summary = {"summary": True, "goals": len(allrows), **summarize(allrows, args.method)}
    summary["goal_looks_like_rate"] = summary.get("own_yes_rate", summary.get("mc_accuracy"))
    emit(summary, args.log)


if __name__ == "__main__":
    main()
