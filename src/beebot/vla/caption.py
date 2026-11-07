"""Caption play windows with LLM-written instructions (Claude Haiku by default; Luna is optional).

    uv run vla-caption --calls 20 [--size 16] [--out data/vla/captions/play.jsonl]

Each call gets CAPTION_BRIEF.md plus 5 windows (facts and three text grids) and returns four
tagged instructions per window (short, medium, detailed, casual). Providers are Claude Haiku
(claude -p, low effort) and Luna (codex exec, low reasoning effort), --per-provider calls in
flight each; Luna read our grids poorly in scripts/vla_grid_reading_probe.py, so --providers
defaults to haiku. Results are appended
to a JSONL file, one line per window; windows already in the file are skipped, so the command
can be re-run to continue. Each line keeps the facts, the provider and tag checks, so audits read
straight from it.
"""

import argparse
import json
import random
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from beebot.vla.data import make_windows
from beebot.vla.facts import window_facts
from beebot.vla.play_data import PLAY_ROOT, load_play, run_dirs
from beebot.vla.world import PALETTE_NAMES, cheese_side

BRIEF = Path(__file__).with_name("CAPTION_BRIEF.md")
DEFAULT_OUT = Path("data/vla/captions/play.jsonl")
PER_CALL = 5
CALL_TIMEOUT = 300
PROVIDERS = ("haiku", "luna")
TIERS = ("short", "medium", "detailed", "casual")  # caption order in every record (records before casual have three)

# Words whose meaning changes under rotations and reflections; relational words that don't
# ("toward", "near", "corner", "center") need no tag for augmentation to be safe.
DIRECTION_WORDS = "top bottom left right up down upper lower north south east west above below leftmost rightmost topmost".split()
COLOUR_WORDS = "red green blue yellow orange purple pink black white gray grey brown cyan".split()
TAG = re.compile(r"<(d|c|u)>(.*?)</\1>")


# ---------------------------------------------------------------------------------------------
# Windows


def text_grid(store, e, t):
    """One frame as text at full resolution, one character per square: '#' wall, R/G/B/Y cheese,
    'o' robot, r/g/b/y box floor (where nothing covers it), '.' empty floor."""
    size = store["mass"].shape[-1]
    rows = [["." for _ in range(size)] for _ in range(size)]
    for (r0, c0, h, w), colour in zip(store["box_rect"][e].tolist(), store["box_color"][e].tolist()):
        if h <= 0 or w <= 0:
            continue
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                rows[r][c] = PALETTE_NAMES[colour][0].lower()
    for r, c in zip(*np.nonzero(store["walls"][e].numpy())):
        rows[r][c] = "#"
    for r, c in zip(*np.nonzero(store["mass"][e, t].float().numpy() > 0.5)):
        rows[r][c] = "o"
    side = cheese_side(size)
    for k, alive in enumerate(store["cheese_alive"][e, t].tolist()):
        if not alive:
            continue
        r0, c0 = store["cheese_pos"][e, t, k].tolist()
        letter = PALETTE_NAMES[store["cheese_color"][e, k]][0].upper()
        for r in range(r0, r0 + side):
            for c in range(c0, c0 + side):
                rows[r][c] = letter
    return "\n".join("".join(row) for row in rows)


def facts_for(store, e, w0, w1):
    static = {k: store[k][e].numpy() for k in ("walls", "cheese_color", "box_rect", "box_color")}

    def snap(t):
        return {"mass": store["mass"][e, t].float().numpy(), "cheese_pos": store["cheese_pos"][e, t].numpy(), "cheese_alive": store["cheese_alive"][e, t].numpy()}

    return window_facts(static, snap(w0), snap(w1))


def window_id(store, e, w0, w1):
    program_id, _, scene = store["source"][e]
    return f"{program_id}:{scene}:{w0}-{w1}"


def eventful_windows(store, seed):
    """[(id, e, w0, w1, facts)] for every window where something happened, shuffled."""
    size = store["mass"].shape[-1]
    found = []
    for e, w0, w1 in make_windows(store, size).tolist():
        facts = facts_for(store, e, w0, w1)
        if facts.eventful:
            found.append((window_id(store, e, w0, w1), e, w0, w1, facts))
    random.Random(seed).shuffle(found)
    return found


def window_text(store, entry):
    wid, e, w0, w1, facts = entry
    middle = (w0 + w1) // 2
    boxes = [f"{PALETTE_NAMES[c]} box" for (r0, c0, h, w), c in zip(store["box_rect"][e].tolist(), store["box_color"][e].tolist()) if h > 0 and w > 0]
    cheeses = [PALETTE_NAMES[c] for c, alive in zip(store["cheese_color"][e].tolist(), store["cheese_alive"][e, w0].tolist()) if alive]
    parts = [
        f"### Window {wid}",
        f"Scene: cheeses at the start {', '.join(cheeses) or 'none'}; boxes {', '.join(boxes) or 'none'}.",
        "Facts:",
        *[f"- {line}" for line in facts.lines],
    ]
    for name, t in (("First", w0), ("Middle", middle), ("Last", w1)):
        parts += [f"{name} frame:", "```", text_grid(store, e, t), "```"]
    return "\n".join(parts)


# ---------------------------------------------------------------------------------------------
# Calls


def call_provider(provider, prompt, effort="low"):
    """The model's reply text; raises on failure."""
    if provider == "haiku":
        command = ["claude", "-p", "--model", "haiku", "--effort", effort, "--tools", "", "--no-session-persistence", "--setting-sources", ""]
        done = subprocess.run(command, input=prompt, capture_output=True, text=True, timeout=CALL_TIMEOUT)
        if done.returncode != 0:
            raise RuntimeError(f"claude exited {done.returncode}: {done.stderr[-500:]}")
        return done.stdout
    with tempfile.TemporaryDirectory() as tmp:
        reply = Path(tmp) / "reply.txt"
        command = ["codex", "exec", "-m", "gpt-6-luna", "-s", "read-only", "--skip-git-repo-check", "--ephemeral", "-c", f"model_reasoning_effort={effort}", "-C", tmp, "-o", str(reply), "-"]
        done = subprocess.run(command, input=prompt, capture_output=True, text=True, timeout=CALL_TIMEOUT)
        if done.returncode != 0 or not reply.exists():
            raise RuntimeError(f"codex exited {done.returncode}: {done.stderr[-500:]}")
        return reply.read_text()


def parse_reply(text, ids):
    """{id: [short, medium, detailed, casual captions]} for the windows the reply got right."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        windows = json.loads(text[start : end + 1]).get("windows", [])
    except json.JSONDecodeError:
        return {}
    wanted = set(ids)
    parsed = {}
    for w in windows:
        captions = w.get("captions") if isinstance(w, dict) else None
        if w.get("id") in wanted and isinstance(captions, list) and len(captions) == len(TIERS) and all(isinstance(c, str) and c.strip() for c in captions):
            parsed[w["id"]] = [c.strip() for c in captions]
    return parsed


def tag_check(caption):
    """Tag problems in one caption: unbalanced tags, and direction or colour words left outside tags."""
    outside = TAG.sub(" ", caption)
    problems = []
    if re.search(r"</?[dcu]>", outside):
        problems.append("unbalanced tags")
    words = re.findall(r"[a-z]+", outside.lower())
    loose_direction = sorted({w for w in words if w in DIRECTION_WORDS})
    loose_colour = sorted({w for w in words if w in COLOUR_WORDS})
    if loose_direction:
        problems.append("untagged direction: " + " ".join(loose_direction))
    if loose_colour:
        problems.append("untagged colour: " + " ".join(loose_colour))
    return problems


def strip_tags(caption):
    return re.sub(r"</?[dcu]>", "", caption)


def run(store, entries, calls, out, log, per_provider=1, providers=PROVIDERS):
    """Caption entries in groups of PER_CALL, alternating providers, per_provider calls in flight each."""
    brief = BRIEF.read_text()
    groups = [entries[i : i + PER_CALL] for i in range(0, len(entries), PER_CALL)][:calls]
    lock = threading.Lock()
    queue = list(enumerate(groups))

    def worker(provider):
        while True:
            with lock:
                if not queue:
                    return
                index, group = queue.pop(0)
            prompt = brief + "\n\n## Windows\n\n" + "\n\n".join(window_text(store, entry) for entry in group)
            started = time.time()
            try:
                parsed = parse_reply(call_provider(provider, prompt), [entry[0] for entry in group])
                error = None
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                parsed, error = {}, str(exc)[-300:]
            seconds = round(time.time() - started, 1)
            with lock:
                with open(out, "a") as f:
                    for wid, e, w0, w1, facts in group:
                        if wid not in parsed:
                            continue
                        captions = parsed[wid]
                        record = {
                            "id": wid,
                            "provider": provider,
                            "facts": facts.lines,
                            "types": sorted(facts.types),
                            "captions": captions,
                            "plain": [strip_tags(c) for c in captions],
                            "tag_problems": [tag_check(c) for c in captions],
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        }
                        f.write(json.dumps(record) + "\n")
                line = {"call": index, "provider": provider, "seconds": seconds, "windows": len(group), "captioned": len(parsed), "error": error}
                print(json.dumps(line), flush=True)
                with open(log, "a") as f:
                    f.write(json.dumps({**line, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")

    threads = [threading.Thread(target=worker, args=(p,)) for _ in range(per_provider) for p in providers]
    for thread in threads:
        thread.start()
        time.sleep(1.0)
    for thread in threads:
        thread.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--play-root", default=str(PLAY_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-provider", type=int, default=1)
    parser.add_argument("--providers", default="haiku", help="comma-separated; luna read our grids poorly (scripts/vla_grid_reading_probe.py), so haiku only by default")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text().splitlines() if line.strip()}
    store = load_play(run_dirs(args.play_root, size=args.size))
    entries = [entry for entry in eventful_windows(store, args.seed) if entry[0] not in done]
    print(f"{len(store['phases'])} scenes, {len(entries)} eventful windows not yet captioned", flush=True)
    run(store, entries, args.calls, out, out.with_suffix(".calls.jsonl"), args.per_provider, tuple(args.providers.split(",")))


if __name__ == "__main__":
    main()
