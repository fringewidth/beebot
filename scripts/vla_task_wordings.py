"""Paraphrases for every instruction intent of the task suite (beebot.vla.tasks.WORDINGS), written
by Haiku, so the policy meets many ways of saying each thing. tasks.wordings() holds out about a
fifth of them (a stable hash) for the "wording" generalisation test.

    uv run python scripts/vla_task_wordings.py [--per 40] [--out src/beebot/vla/task_wordings.json]

One call per intent. A paraphrase is kept only if it uses exactly the seed templates' placeholders.
"""

import argparse
import json
import re
import string
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from beebot.vla.caption import call_provider
from beebot.vla.tasks import WORDINGS

PROMPT = """People will type instructions to a swarm of small robots on a grid world with coloured
cheeses (which the swarm can eat by surrounding them, or push), coloured boxes (floor areas the
swarm can enter) and walls. Write {per} different ways a person might phrase the instruction below.

Seed phrasings (same meaning):
{seeds}

Rules:
- {fields_rule}
- Same meaning as the seeds, nothing added or dropped.
- Vary a lot: terse commands, polite requests, casual or slangy, longer sentences, indirect ones
  ("I'd like ..."), different verbs and word order. Lowercase is fine. No numbering.

Reply with JSON only: {{"phrasings": ["...", ...]}}"""


def fields_of(template):
    return sorted({f for _, f, _, _ in string.Formatter().parse(template) if f})


def generate(key, per):
    seeds = WORDINGS[key]
    fields = fields_of(seeds[0])
    fields_rule = (
        "Keep exactly these placeholders, each once, spelled the same: " + ", ".join("{" + f + "}" for f in fields) + ". They will be filled in later (a colour, a word, a region, a direction, a shape)."
        if fields
        else "There are no placeholders: write plain sentences, with no braces or brackets."
    )
    prompt = PROMPT.format(per=per, seeds="\n".join(f"- {s}" for s in seeds), fields_rule=fields_rule)
    reply = call_provider("haiku", prompt)
    start, end = reply.find("{"), reply.rfind("}")
    try:
        found = json.loads(reply[start : end + 1])["phrasings"]
    except (ValueError, KeyError):
        return key, []
    kept = []
    for t in found:
        t = re.sub(r"\s+", " ", str(t)).strip()
        try:
            ok = fields_of(t) == fields and all(t.count("{" + f + "}") == 1 for f in fields)
        except ValueError:
            ok = False
        if ok and "(none)" not in t and t not in seeds and t not in kept:
            kept.append(t)
    return key, kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per", type=int, default=40)
    parser.add_argument("--out", default="src/beebot/vla/task_wordings.json")
    parser.add_argument("--keys", nargs="*", help="only these intents (the rest of --out is kept)")
    args = parser.parse_args()
    keys = args.keys or sorted(WORDINGS)
    with ThreadPoolExecutor(4) as pool:
        results = dict(pool.map(lambda k: generate(k, args.per), keys))
    out = Path(args.out)
    merged = {**(json.loads(out.read_text()) if out.exists() and args.keys else {}), **results}
    Path(args.out).write_text(json.dumps(merged, indent=1))
    for key, kept in sorted(results.items()):
        print(f"{key}: {len(kept)} kept, e.g. {kept[:2]}")


if __name__ == "__main__":
    main()
