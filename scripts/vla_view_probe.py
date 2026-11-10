"""Readability of the randomized views (vla/voxel/views.py, README "Views and randomization").

    PYTHONPATH=src python scripts/vla_view_probe.py --out runs/views --sheets --speed
    PYTHONPATH=src python scripts/vla_view_probe.py --out runs/views --qwen --n 30

--sheets writes contact sheets of the probe items (colours, letters, shapes, as the zero-shot VLM
probe) under each strength: <out>/sheet_<strength>_<probe>.jpg, every item as its oblique view
above its top view. --speed times scene_views (pairs per second on this process, one thread).
--qwen asks Qwen3.5-4B (local weights, MPS) the probe questions zero-shot on n items per probe and
strength and prints the accuracy table (<out>/qwen.jsonl, <out>/qwen.txt). The Mac GPU must be
free (one GPU job at a time).
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from beebot.vla.voxel import suite
from beebot.vla.voxel.views import ViewSpec, sample_look, scene_views
from beebot.vla.voxel.world import CHEESE_SIDE, COLOUR_NAMES, observe, sample_object_scenes, world_from_arrays

SIZE = 16
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
STRENGTHS = {"clean": 0.0, "s25": 0.25, "medium": 0.5, "s75": 0.75, "strong": 1.0}
SHAPE_OPTIONS = [suite.SHAPE_WORDS[s] for s in suite.SHAPES]


def colour_items(n, seed=0):
    rng = np.random.default_rng(seed)
    world, _ = suite.scenes(rng, n, cheeses=(1, 3), colours=COLOUR_NAMES)
    items = []
    for i in range(n):
        w = world.index(i)
        names = []
        for k, name in suite.present(world, i):
            z0, y0, x0 = (int(v) for v in world.cheese_pos[i, k])
            w.mass[0, z0:, y0 : y0 + CHEESE_SIDE, x0 : x0 + CHEESE_SIDE] = 0  # no robot on top hides a cheese
            names.append(name)
        items.append({"probe": "colours", "id": f"colours-{i}", "answer": sorted(names), "obs": observe(w)[0]})
    return items


def shift_mask(mask, rng, rotate):
    if rotate:
        mask = np.rot90(mask, int(rng.integers(4)), axes=(1, 2)).copy()
    ys, xs = np.nonzero(mask.any(axis=0))
    dy = int(rng.integers(-ys.min(), SIZE - ys.max()))
    dx = int(rng.integers(-xs.min(), SIZE - xs.max()))
    return np.roll(mask, (dy, dx), axis=(1, 2))


def mask_item(probe, i, answer, mask):
    return {"probe": probe, "id": f"{probe}-{i}", "answer": answer, "obs": observe(world_from_arrays(mask.astype(np.float32)))[0]}


def letter_items(n, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        letter = LETTERS[int(rng.integers(len(LETTERS)))]
        out.append(mask_item("letters", i, letter, shift_mask(suite.letter_mask(letter, standing=False), rng, rotate=False)))
    return out


def shape_items(n, seed=2):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        name = suite.SHAPES[i % len(suite.SHAPES)]
        out.append(mask_item("shapes", i, suite.SHAPE_WORDS[name], shift_mask(suite.shape_mask(name), rng, rotate=True)))
    rng.shuffle(out)
    return out


def render(items, strength, seed):
    rng = np.random.default_rng(seed)
    spec = ViewSpec(strength=strength)
    for item in items:
        item["images"], item["pose"] = scene_views(item["obs"], rng, spec)


def sheets(items_by_probe, out, per_sheet=8):
    for name, strength in STRENGTHS.items():
        for probe, items in items_by_probe.items():
            render(items[:per_sheet], strength, seed=11)
            sheet = Image.new("RGB", (256 * per_sheet, 512), (255, 255, 255))
            for k, item in enumerate(items[:per_sheet]):
                sheet.paste(item["images"][0], (256 * k, 0))
                sheet.paste(item["images"][1], (256 * k, 256))
            sheet.save(out / f"sheet_{name}_{probe}.jpg", quality=85)


def speed(out):
    torch.set_num_threads(1)
    world, _ = sample_object_scenes(np.random.default_rng(0), 16)
    obs = observe(world).numpy()
    lines = []
    for name in ("clean", "medium", "strong"):
        spec = ViewSpec(strength=STRENGTHS[name])
        for reuse in (False, True):
            rng = np.random.default_rng(0)
            look = sample_look(rng, spec) if reuse else None
            scene_views(obs[0], rng, spec, look)
            best = 0.0
            for _ in range(3):
                start = time.perf_counter()
                for i in range(16):
                    scene_views(obs[i], rng, spec, look)
                best = max(best, 16 / (time.perf_counter() - start))
            line = f"{name:7s} look {'reused' if reuse else 'fresh '}  {best:5.1f} pairs/s (256 x 256, one thread, best of 3 x 16)"
            print(line, flush=True)
            lines.append(line)
    (out / "speed.txt").write_text("\n".join(lines) + "\n")


WORLD = "A 16 x 16 x 16 voxel world on a sheet. Robots are small cubes (all one colour), terrain is blocks, cheeses are coloured blocks."
QUESTIONS = {
    "colours": "Which cheese colours are in the scene? Possible colours: " + ", ".join(COLOUR_NAMES) + ". Answer with the colours present, comma-separated, nothing else.",
    "letters": "The robots stand on the floor in the shape of one capital letter of the alphabet (it may be seen rotated). Which letter? Answer with the single letter only.",
    "shapes": "The robots form one of these shapes: " + "; ".join(SHAPE_OPTIONS) + ". Which one? Answer with the shape name exactly as listed, nothing else.",
}
QUESTIONS = {k: v + " Reply with the answer only, no explanation." for k, v in QUESTIONS.items()}


def prompt(probe):
    return f"{WORLD}\nThe two images show the same scene: the first from a camera above one side of the arena, the second from directly above.\n\n{QUESTIONS[probe]}"


def parse(probe, text):
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    low = t.lower()
    if probe == "colours":
        return sorted({c for c in COLOUR_NAMES if re.search(rf"\b{c}\b", low)})
    if probe == "letters":
        m = re.search(r"\b([A-Z])\b", t)
        return m.group(1) if m else t[:1].upper()
    for short, word in (("sphere", "hollow sphere"), ("ring", "flat ring"), ("plus", "3D plus sign"), ("pyramid", "pyramid"), ("cube", "hollow cube"), ("wall", "wall")):
        if short in low:
            return word
    return t


class ManualProcessor:
    """apply_chat_template for text and PIL images without torchvision (as the VLM probe)."""

    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        images = [c["image"] for m in messages for c in m["content"] if c["type"] == "image"]
        pixels = self.image_processor(images=images, return_tensors="pt")
        merge = self.image_processor.merge_size**2
        pieces = text.split("<|image_pad|>")
        text = pieces[0]
        for thw, rest in zip(pixels["image_grid_thw"], pieces[1:]):
            text += "<|image_pad|>" * int(thw.prod() // merge) + rest
        out = dict(pixels)
        out.update(self.tokenizer(text, return_tensors="pt"))
        return out


@torch.no_grad()
def qwen(items_by_probe, out, n):
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from transformers.models.qwen2_vl import Qwen2VLImageProcessorPil

    name = "Qwen/Qwen3.5-4B"
    proc = ManualProcessor(AutoTokenizer.from_pretrained(name, local_files_only=True), Qwen2VLImageProcessorPil.from_pretrained(name, local_files_only=True))
    model = AutoModelForImageTextToText.from_pretrained(name, local_files_only=True, dtype=torch.bfloat16).to("mps").eval()
    log = open(out / "qwen.jsonl", "w")
    rows = []
    for setting, strength in STRENGTHS.items():
        for probe, items in items_by_probe.items():
            render(items[:n], strength, seed=23)
            for item in items[:n]:
                content = [{"type": "image", "image": item["images"][0]}, {"type": "image", "image": item["images"][1]}, {"type": "text", "text": prompt(probe)}]
                inputs = {k: v.to("mps") for k, v in proc([{"role": "user", "content": content}]).items()}
                reply = model.generate(**inputs, max_new_tokens=32, do_sample=False)
                text = proc.tokenizer.decode(reply[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                guess = parse(probe, text)
                row = {"setting": setting, "strength": strength, "probe": probe, "id": item["id"], "answer": item["answer"], "guess": guess, "reply": text, "correct": guess == item["answer"], "azimuth": item["pose"]["azimuth"]}
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
            acc = np.mean([r["correct"] for r in rows if r["setting"] == setting and r["probe"] == probe])
            print(f"{setting:7s} {probe:8s} {acc:.2f}", flush=True)
    lines = ["setting  strength  colours  letters  shapes  mean"]
    for setting, strength in STRENGTHS.items():
        accs = [np.mean([r["correct"] for r in rows if r["setting"] == setting and r["probe"] == p]) for p in items_by_probe]
        lines.append(f"{setting:7s}  {strength:8.2f}  " + "  ".join(f"{a:7.2f}" for a in accs) + f"  {np.mean(accs):4.2f}")
    text = "\n".join(lines)
    print(text)
    (out / "qwen.txt").write_text(text + f"\n\n{n} items per probe and setting, zero-shot, greedy, Qwen3.5-4B bf16 on MPS\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/views")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--sheets", action="store_true")
    parser.add_argument("--speed", action="store_true")
    parser.add_argument("--qwen", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items_by_probe = {"colours": colour_items(args.n), "letters": letter_items(args.n), "shapes": shape_items(args.n)}
    if args.sheets:
        sheets(items_by_probe, out)
    if args.speed:
        speed(out)
    if args.qwen:
        qwen(items_by_probe, out, args.n)


if __name__ == "__main__":
    main()
