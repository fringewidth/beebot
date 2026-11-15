"""Check re-captions against the grey voxel render with Claude Haiku (vision, through `claude -p` with the Read
tool), and draw review sheets for a person to check Haiku's verdicts.

Per shape: its two-view grey render (goal_library.render_views) and two descriptions, its own re-caption and the
re-caption of a shape of another category (the control), in a random order as A and B. Haiku picks A, B or
neither ("neither" when a person who was not told would not recognise the shape from either). A shape is kept
when Haiku picks its own description. Renders are tiled 10 to an image, numbered, two images per call.

    nice .venv/bin/python scripts/recaption_vision.py --recaptions runs/label_audit/recaption/pilot.jsonl --out runs/label_audit/recaption/vision_pilot

Output under --out: tiles/ (what Haiku saw), vision.jsonl (one row per shape), review/sheet_NN.png (render,
category, original caption, new caption, Haiku's pick; UNCLEAR shapes too) and summary.json.
"""

import argparse
import json
import random
import subprocess
import textwrap
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from beebot.vla.voxel import goal_library as G
from beebot.vla.voxel import haiku as H

ROOT = Path("data/goal_library_v2")
PER_IMAGE = 10
IMAGES_PER_CALL = 2
COLS = 2
CALL_TIMEOUT = 600
FONT = "/System/Library/Fonts/Helvetica.ttc"

PROMPT = """Read these images: {paths}

Each image holds numbered panels. Each panel shows ONE object built from plain grey cubes, seen from two sides (the left and right halves of the panel are the same object from opposite corners). There is no colour or texture.

For each panel number below, decide which description fits the object: A, B, or neither. Answer neither when a person who had not been told what it is would not recognise the object from either description. Judge the shape only.

{items}

Answer with one JSON object and nothing else: keys are the panel numbers as strings, values "A", "B" or "neither", e.g. {{"1": "A", "2": "neither"}}."""


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def first_cells(sources):
    """source -> {source_id: cells} of the first stored variant within the robot cap."""
    out = {}
    for source in sources:
        lines, arrays = G.load(source, root=ROOT)
        table = {}
        for line, cells in zip(lines, arrays):
            table.setdefault(line["source_id"], cells)
        out[source] = table
    return out


def tile(renders, numbers, path):
    w, h = renders[0].size
    rows = (len(renders) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * w, rows * h), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (render, number) in enumerate(zip(renders, numbers)):
        x, y = (k % COLS) * w, (k // COLS) * h
        sheet.paste(render, (x, y))
        draw.rectangle([x, y, x + w - 1, y + h - 1], outline="black", width=2)
        draw.text((x + 8, y + 4), str(number), fill="black", font=font(36))
    sheet.save(path)


def call(prompt):
    command = ["claude", "-p", "--model", "haiku", "--effort", "medium", "--tools", "Read", "--allowedTools", "Read", "--no-session-persistence", "--setting-sources", ""]
    done = subprocess.run(command, input=prompt, capture_output=True, text=True, timeout=CALL_TIMEOUT)
    if done.returncode != 0:
        raise RuntimeError(f"claude exited {done.returncode}: {done.stderr[-500:]}")
    return done.stdout


def review_sheet(rows, renders, path):
    w, h = renders[0].size
    text_w = 640
    sheet = Image.new("RGB", (w + text_w, h * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (row, render) in enumerate(zip(rows, renders)):
        y = k * h
        sheet.paste(render, (0, y))
        draw.line([0, y, w + text_w, y], fill="black", width=2)
        verdict = "UNCLEAR (dropped by the text pass)" if row["unclear"] else f"Haiku picked: {row['pick']}  ->  {'KEEP' if row['keep'] else 'DROP'}"
        lines = [f"#{row['n']}  {row['source']}  category: {row['label']}", "", "original: " + row["original"], "", "new: " + (row["text"] or "UNCLEAR"), ""]
        if not row["unclear"]:
            lines += [f"A: {row['A']}", f"B: {row['B']}", ""]
        lines.append(verdict)
        out, ty = [], y + 8
        for line in lines:
            out += textwrap.wrap(line, 78) or [""]
        for line in out[:17]:
            draw.text((w + 10, ty), line, fill="black", font=font(15))
            ty += 17
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recaptions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    for sub in ("tiles", "review"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    recaptions = H.read_jsonl(args.recaptions)
    requests = {r["custom_id"]: r for r in H.read_jsonl("runs/label_audit/recaption_batch.jsonl")}
    cells = first_cells(sorted({r["source"] for r in recaptions}))
    rows = []
    for row in recaptions:
        shape = cells[row["source"]].get(row["source_id"])
        if shape is None:
            continue  # every variant is over the robot cap: the loader never shows it
        rows.append({**row, "n": len(rows) + 1, "original": " | ".join(c.strip() for c in requests[row["custom_id"]]["captions"][:2]), "cells": shape})
    described = [r for r in rows if not r["unclear"]]
    for r in described:
        others = [o for o in described if o["label"] != r["label"]] or [o for o in described if o is not r]
        control = rng.choice(others)["text"]
        r["own_slot"] = rng.choice("AB")
        r["A"], r["B"] = (r["text"], control) if r["own_slot"] == "A" else (control, r["text"])
    renders = {r["n"]: G.render_views(r["cells"], scale=9) for r in rows}
    jobs = []
    per_call = PER_IMAGE * IMAGES_PER_CALL
    for c in range(0, len(described), per_call):
        group = described[c : c + per_call]
        paths = []
        for i in range(0, len(group), PER_IMAGE):
            part = group[i : i + PER_IMAGE]
            path = out / "tiles" / f"call{c // per_call:03d}_{i // PER_IMAGE}.png"
            tile([renders[r["n"]] for r in part], [r["n"] for r in part], path)
            paths.append(str(path.resolve()))
        items = "\n".join(f"{r['n']}. A: {r['A']}\n   B: {r['B']}" for r in group)
        jobs.append((group, PROMPT.format(paths=", ".join(paths), items=items)))

    def work(job):
        group, prompt = job
        for attempt in range(3):
            try:
                answer = H.parse_json(call(prompt))
                if answer:
                    return group, answer
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                print(json.dumps({"error": str(exc)[-300:], "attempt": attempt}), flush=True)
        return group, {}

    with ThreadPoolExecutor(args.workers) as pool:
        for group, answer in pool.map(work, jobs):
            for r in group:
                pick = str(answer.get(str(r["n"]), "missing")).strip()
                r["pick"] = pick
                r["keep"] = pick == r["own_slot"]
    for r in rows:
        r.setdefault("pick", None)
        r.setdefault("keep", False)
    with open(out / "vision.jsonl", "w") as handle:
        for r in rows:
            handle.write(json.dumps({k: v for k, v in r.items() if k != "cells"}) + "\n")
    for s in range(0, len(rows), 10):
        part = rows[s : s + 10]
        review_sheet(part, [renders[r["n"]] for r in part], out / "review" / f"sheet_{s // 10:02d}.png")
    picks = [r["pick"] for r in described]
    summary = {"shapes": len(rows), "unclear_text_pass": len(rows) - len(described), "judged": len(described), "own": sum(r["keep"] for r in described), "control": sum(r["pick"] in ("A", "B") and not r["keep"] for r in described), "neither": picks.count("neither"), "missing": picks.count("missing")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
