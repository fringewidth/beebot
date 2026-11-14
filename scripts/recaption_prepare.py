"""Prepare (never send) a re-captioning batch for text2shape and objaverse_cap3d (label audit 2026-11-13, verdict FIX).

Reads data/goal_library_v2/{text2shape,objaverse_cap3d}.jsonl, takes each distinct shape once (source_id), and writes
one request per shape to --out (JSON lines: custom_id, prompt, source, original captions). It calls no API. It prints
item counts, a token estimate and the cost at two Haiku price points. It also lists objaverse rows whose category
label shares no word with the caption (the audit found about 15 rows where they disagree).

    uv run python scripts/recaption_prepare.py --out runs/label_audit/recaption_batch.jsonl

Token counts are estimated as characters / 3.5 (an assumption; count_tokens on a sample would be exact). Prices are
per million tokens from the claude-api skill table cached 2026-11-03: claude-haiku-5-5 $0.10 in / $0.50 out (prompts
under 100K), claude-haiku-4-5 $1.00 in / $5.00 out. The Message Batches API halves both; shown as the batch column.
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path("data/goal_library_v2")
CHARS_PER_TOKEN = 3.5
OUT_TOKENS_PER_ITEM = 40  # a short plain description
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-haiku-4-5": (1.00, 5.00)}

PROMPT = """Below are captions written for one 3D object, plus its category. Write one plain description of the object's SHAPE only, in one sentence of at most 25 words.

Rules:
1. Describe only geometry a person could check on a voxel model: overall form, parts (legs, back, arms, handle, top), how many, proportions, and openings.
1. Do not mention colours, materials, textures, brands, names, logos, printed or written text, price, age, style words ("modern", "cute") or what the object is made of.
1. Use plain words, no jargon, no numbers above nine, and never repeat a word twice in a row.
1. If the captions disagree with each other or with the category, describe only what they agree on; if nothing is left, answer exactly UNCLEAR.

Category: {label}
Captions:
{captions}

Answer with the description only."""

STOP = {"a", "an", "the", "of", "and", "with", "in", "on", "for", "to", "or", "computer", "equipment", "object"}


def words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP}


def stem(word):
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def shapes(name):
    seen = {}
    for line in open(ROOT / f"{name}.jsonl"):
        row = json.loads(line)
        if row["source_id"] in seen:
            continue
        captions = row.get("extra", {}).get("descriptions") or [row["text"]]
        seen[row["source_id"]] = {"source": name, "source_id": row["source_id"], "label": row["label"], "captions": captions}
    return list(seen.values())


def mismatched(items):
    """Objaverse: the category label shares no word (after a plural stem) with any caption."""
    out = []
    for item in items:
        label_words = {stem(w) for w in words(item["label"].replace("_", " "))}
        caption_words = {stem(w) for c in item["captions"] for w in words(c)}
        if label_words and not label_words & caption_words:
            out.append(item)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    totals = {"items": 0, "in_tokens": 0}
    with open(args.out, "w") as handle:
        for name in ("text2shape", "objaverse_cap3d"):
            items = shapes(name)
            in_chars = 0
            for item in items:
                captions = "\n".join(f"- {c.strip()}" for c in item["captions"][:5])
                prompt = PROMPT.format(label=item["label"].replace("_", " "), captions=captions)
                in_chars += len(prompt)
                handle.write(json.dumps({"custom_id": f"{name}:{item['source_id']}", "source": name, "prompt": prompt, "captions": item["captions"], "label": item["label"]}) + "\n")
            in_tokens = int(in_chars / CHARS_PER_TOKEN)
            totals["items"] += len(items)
            totals["in_tokens"] += in_tokens
            print(json.dumps({"source": name, "items": len(items), "input_tokens_est": in_tokens}))
            if name == "objaverse_cap3d":
                bad = mismatched(items)
                print(json.dumps({"source": name, "category_caption_no_shared_word": len(bad), "examples": [(b["label"], b["captions"][0][:80]) for b in bad[:8]]}))
    out_tokens = totals["items"] * OUT_TOKENS_PER_ITEM
    for model, (price_in, price_out) in PRICES.items():
        cost = totals["in_tokens"] / 1e6 * price_in + out_tokens / 1e6 * price_out
        print(json.dumps({"model": model, "items": totals["items"], "input_tokens_est": totals["in_tokens"], "output_tokens_est": out_tokens, "usd": round(cost, 2), "usd_batch_api": round(cost / 2, 2)}))
    print(f"wrote {totals['items']} requests to {args.out}; nothing was sent")


if __name__ == "__main__":
    main()
