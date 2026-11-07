"""Label which events each held-out caption asks for, so text following can be scored on what
was asked rather than on every event of the recorded window (a short caption names one event of
about four; the window facts score charges it for the rest).

    uv run python scripts/vla_requested_events.py --play_cache data/vla/play_cache_g32 \
        --captions data/vla/captions/play_g32_set.pt [--out data/vla/captions/requested_g32.json]

For the val and test captions that train.py evaluates (CaptionHalf's held-out sets), Haiku reads
the caption and the window's events (the scorer's fact atoms, in words) and lists the ones the
caption requests. Output: {"val": [[atom, ...] per held-out window], "test": [...]}, in the order
of CaptionHalf.val / .test, each atom as a list (JSON has no tuples). Re-running resumes: windows
already labelled are kept.
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vla_demo_media import describe  # noqa: E402

from beebot.vla import goal_eval  # noqa: E402
from beebot.vla.caption import call_provider  # noqa: E402
from beebot.vla.facts import window_facts  # noqa: E402
from beebot.vla.play_data import cache_play, load_play_cache, run_dirs  # noqa: E402
from beebot.vla.train import CaptionHalf  # noqa: E402

PROMPT = """You label instructions given to a swarm of robots in a grid world with cheeses and boxes.

For each item below you get an instruction and a numbered list of events that happened in the
recorded episode the instruction was written for. List the numbers of the events the instruction
asks for. Be strict: an event counts only if the instruction states it, in any wording, or if
doing what it says is impossible without that event. An event that merely happened in the
episode, or that is compatible with the instruction, does not count.

- Paraphrases count: "gobble the red one" asks for "eat the red cheese", "crowd the red one" for
  "surround the red cheese", "break into a few groups" for a split into groups.
- A place to end up does not ask for a direction of travel: "gather in the center" asks for
  "end up center" (and "stay in one group"), not for "move down and left".
- A direction of travel does not ask for a place to end up: "go up" asks for "move up", not for
  "end up center"; "drift left a bit" asks for "move left", not for "end up top left".
- Naming one box or cheese does not ask for events about other ones: "go into the yellow box"
  does not ask for "leave the blue box"; "away from the green one" does not ask for "move away
  from the red cheese".
- "Gather", "together", "as one" ask for "stay in one group"; otherwise do not count it.
- Any request to split ("split", "break up", "a few groups", "several groups", "spread out into
  groups") asks for the listed "split into N groups" event, whatever N is; "split in two" or
  "three groups" ask for it only if N matches.
- A direction in the instruction must match the event's direction.

Reply with JSON only, no prose: {{"items": [{{"id": <id>, "requested": [<event numbers>]}}, ...]}}

{items}
"""


def strip_tags(text):
    return re.sub(r"</?[a-z]>", "", text)


def atoms_of(store, window):
    e, w0, w1 = (int(x) for x in window)
    static = {k: store[k][e].numpy() for k in ("walls", "cheese_color", "box_rect", "box_color")}
    atoms = window_facts(static, goal_eval.stored_snapshot(store, e, w0), goal_eval.stored_snapshot(store, e, w1)).atoms
    return sorted(atoms, key=str)


def label_batch(batch):
    """batch: [(id, caption, atoms)] -> {id: [atom, ...]} for the items the reply got right."""
    items = []
    for i, caption, atoms in batch:
        events = "\n".join(f"  {k + 1}. {describe([a])[0]}" for k, a in enumerate(atoms))
        items.append(f"Item {i}\nInstruction: {caption}\nEvents:\n{events}")
    reply = call_provider("haiku", PROMPT.format(items="\n\n".join(items)), effort="medium")
    start, end = reply.find("{"), reply.rfind("}")
    try:
        parsed = json.loads(reply[start : end + 1])["items"]
    except (ValueError, KeyError, TypeError):
        return {}
    atoms_by_id = {i: atoms for i, _, atoms in batch}
    out = {}
    for item in parsed:
        i, numbers = item.get("id"), item.get("requested")
        if i in atoms_by_id and isinstance(numbers, list) and all(isinstance(n, int) and 1 <= n <= len(atoms_by_id[i]) for n in numbers):
            out[i] = [list(atoms_by_id[i][n - 1]) for n in sorted(set(numbers))]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--play_cache", required=True)
    parser.add_argument("--play_root", default="/Users/hrishik/beebot-play-g32", help="to build the cache if it is missing")
    parser.add_argument("--captions", required=True)
    parser.add_argument("--out", default="data/vla/captions/requested_g32.json")
    parser.add_argument("--limit", type=int, default=512, help="held-out windows per split (train.py's text_val_windows)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not (Path(args.play_cache) / "small.pt").exists():
        print(f"play cache: {cache_play(run_dirs(args.play_root, size=32), args.play_cache)} episodes", flush=True)
    store = load_play_cache(args.play_cache)
    caps = CaptionHalf(args.captions, store, args.limit)
    out_path = Path(args.out)
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    for split in ("val", "test"):
        held = getattr(caps, split)
        labels = done.get(split) or [None] * len(held.windows)
        todo = [(i, strip_tags(held.texts[i]), atoms_of(store, held.windows[i])) for i in range(len(held.windows)) if labels[i] is None]
        for attempt in range(3):
            if not todo:
                break
            batches = [todo[k : k + args.batch] for k in range(0, len(todo), args.batch)]
            with ThreadPoolExecutor(args.workers) as pool:
                for got in pool.map(label_batch, batches):
                    for i, atoms in got.items():
                        labels[i] = atoms
            todo = [t for t in todo if labels[t[0]] is None]
            done[split] = labels
            out_path.write_text(json.dumps(done))
            print(f"{split}: {sum(x is not None for x in labels)} of {len(labels)} labelled after pass {attempt + 1}", flush=True)
        requested = [len(x) for x in labels if x is not None]
        events = [len(atoms_of(store, w)) for w in held.windows]
        print(f"{split}: requested {sum(requested) / max(len(requested), 1):.2f} events per caption of {sum(events) / len(events):.2f} in the window", flush=True)


if __name__ == "__main__":
    main()
