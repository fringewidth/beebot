"""Re-caption text2shape and objaverse_cap3d shapes with Claude Haiku through `claude -p` (label audit
2026-11-13, verdict FIX; user approved 2026-11-15).

Reads the requests v2/scripts/recaption_prepare.py wrote (runs/label_audit/recaption_batch.jsonl: one per
distinct shape, with its category and original captions) and sends them in batches of --batch shapes per
call, --workers calls at a time, with the rules of recaption_prepare.PROMPT stated once per call. Haiku
answers one JSON object {item number: description}. A shape whose captions disagree with each other or
with the category is answered UNCLEAR (this is also the objaverse category check of the audit).

    nice .venv/bin/python v2/scripts/recaption_run.py                 # everything, resumable
    nice .venv/bin/python v2/scripts/recaption_run.py --limit 60      # a pilot

Output (appended one row per shape, resumable: answered ids are skipped):
  data/goal_library_v2/recaptions.jsonl   {custom_id, source, source_id, label, text, unclear, call}
  runs/label_audit/recaption/             calls.jsonl, failures.jsonl, throughput.jsonl (one line a minute)
"""

import argparse
import json
from pathlib import Path

from beebot.vla import caption
from beebot.vla.voxel import haiku as H

REQUESTS = Path("runs/label_audit/recaption_batch.jsonl")
OUT = Path("data/goal_library_v2/recaptions.jsonl")
LOGS = Path("runs/label_audit/recaption")
MAX_WORDS = 25

RULES = """Below are numbered 3D objects. For each one you get its category and captions people wrote for it. Write one plain description of each object's SHAPE only, in one sentence of at most 25 words.

Rules:
1. Describe only geometry a person could check on a voxel model: overall form, parts (legs, back, arms, handle, top), how many, proportions, and openings.
1. Do not mention colours, materials, textures, brands, names, logos, printed or written text, price, age, style words ("modern", "cute") or what the object is made of.
1. Use plain words, no jargon, no numbers above nine, and never repeat a word twice in a row.
1. If the captions disagree with each other or with the category, describe only what they agree on; if nothing is left, answer exactly UNCLEAR.
1. The object will be built from plain grey cubes, at most 16 on a side, with no colour or texture. If it would only be recognisable by its colour, texture, printed text or picture, or by details smaller than about a tenth of its size (for example most fruit, signs, books, screens, coins), answer exactly UNCLEAR.

Answer with one JSON object and nothing else: keys are the item numbers as strings, values are the descriptions, e.g. {"1": "...", "2": "UNCLEAR"}.

"""


def item_block(number, request):
    captions = "\n".join(f"  - {c.strip()}" for c in request["captions"][:5])
    return f"{number}. Category: {request['label'].replace('_', ' ')}\n  Captions:\n{captions}"


def prompt_of(batch):
    return RULES + "\n\n".join(item_block(i + 1, r) for i, r in enumerate(batch))


def interleave(requests):
    """Round robin over sources, so a pilot (--limit) sees both."""
    by_source = {}
    for request in requests:
        by_source.setdefault(request["source"], []).append(request)
    lists = list(by_source.values())
    return [lst[i] for i in range(max(map(len, lists), default=0)) for lst in lists if i < len(lst)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=30)
    parser.add_argument("--workers", type=int, default=H.WORKERS)
    parser.add_argument("--limit", type=int, default=None, help="only the first N pending shapes (a pilot)")
    parser.add_argument("--passes", type=int, default=2, help="re-send shapes a reply left out")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    requests = interleave(H.read_jsonl(REQUESTS))
    for npass in range(args.passes):
        done = {row["custom_id"] for row in H.read_jsonl(out)}
        todo = [r for r in requests if r["custom_id"] not in done]
        if args.limit is not None:
            todo = todo[: max(0, args.limit - len(done))]
        if not todo:
            break
        print(json.dumps({"pass": npass, "pending": len(todo), "done": len(done), "of": len(requests)}), flush=True)

        def on_answers(batch, answers, call):
            for i, request in enumerate(batch):
                text = H.clean(answers.get(str(i + 1)))
                if not text:
                    continue
                unclear = text.upper().rstrip(".") == "UNCLEAR"
                H.append(out, {"custom_id": request["custom_id"], "source": request["source"], "source_id": request["custom_id"].split(":", 1)[1], "label": request["label"], "text": "" if unclear else text, "unclear": unclear, "words": len(text.split()), "call": call})

        def ids_of(batch):
            return {str(i + 1) for i in range(len(batch))}

        with H.Meter("recaption", len(todo), LOGS / "throughput.jsonl") as meter:
            H.run_batches(H.chunks(todo, args.batch), prompt_of, ids_of, lambda p: caption.call_provider("haiku", p), on_answers, meter, LOGS / "calls.jsonl", LOGS / "failures.jsonl", args.workers)
    rows = H.read_jsonl(out)
    summary = {"answered": len(rows), "of": len(requests), "unclear": sum(r["unclear"] for r in rows), "over_max_words": sum(r["words"] > MAX_WORDS for r in rows)}
    for source in ("text2shape", "objaverse_cap3d"):
        mine = [r for r in rows if r["source"] == source]
        summary[source] = {"answered": len(mine), "unclear": sum(r["unclear"] for r in mine)}
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
