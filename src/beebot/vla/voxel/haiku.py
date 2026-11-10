"""Open-vocabulary instructions and thought labels for the voxel suite, written by Claude Haiku and
checked by a second Haiku call that sees only the instruction and the scene.

    PYTHONPATH=src python -m beebot.vla.voxel.haiku <command> [--name NAME] [--dry-run] ...

Commands (each takes --dry-run: a fake provider, the exact prompts and the parsed fake outputs
printed, nothing written under --root):
  export    draw tasks exactly as voxel/demos.py draws them and write items.jsonl (--from-items: the
            items of an earlier export, with start voxels attached and a fresh menu)
  generate  three instructions per item on a ladder of indirection (close, indirect, casual), each
            with a first thought and one line per later sub-goal; about 10% of items also get an
            instruction that cannot or should not be followed (kind refuse or clarify)
  verify    a separate call per batch of instructions that sees only the instruction and the scene
            summary and writes the task back as structured JSON; a pair is kept only if that spec
            equals the item's own (same_spec); keep rates per family and per rung
  testset   a frozen held-out test set: its own seed, a separately worded prompt that offers only
            held-out words and categories (wordings.json splits), verified the same way, written
            to <root>/test/ with a HELD_OUT marker; the training loader refuses it
  pilot     export + generate + verify of 50 items, then a review of every pair to read by hand
  report    keep rates of a verified set

Files under <root>/<name>/ (root: data/vla/haiku), all JSONL appended one row at a time:
  config.json      the draw (seed, chunk, robots, counts, fit, rule, apart, store) and the items count
  menu.json        the verifier's menu: spec kinds, their fields and values, a plain example of each
  items.jsonl      {id, seed, index, e, record, scene}, and for the coverage families voxels (the
                   start robots and cheese centres a robot selector is resolved on, start_voxels)
  generated.jsonl  one row per item answered: {id, seed, index, family, kind, pairs, rejects, call}
                   pairs: {pair_id, rung, instruction, thought, switches, done} (and kind, reply for
                   the refusal rung); rejects: {rung, reason, answer} of answers that failed checks
  verified.jsonl   kept pairs: the pair, its item's keys and the verifier's spec
  rejected.jsonl   pairs the verifier read differently, with the reason and its spec
  failures.jsonl   calls that failed after every retry, items missing from a reply
  calls.jsonl      one row per call; throughput.jsonl one row per minute (items/min, calls/min)
  report.json      keep rates per family and per rung (verify)
Every command is resumable: ids already answered are skipped, so a run that stops is re-run as is.

Switch episodes (an instruction change at step t) are separate user messages: generate and verify
split each into one item per part (instruction_items: "<id>#p<j>", the part's record and t, the
episode's start scene, the later parts marked so that Haiku does not refer to start positions),
and Labels.switch_parts gives each part's instructions with its step. Robot selectors (who, by,
which, count, except) are compared by the robots they pick at the start (same_task), so that
"the larger group" verifies as "everyone except the group at the back" when they are the same robots.

Records and scenes. The pipeline only reads canonical records {family, kind, plain, spec, subgoals}
(subgoals: the plain sentence of each expert plan segment, in plan order) and scene summaries
(JSON); it never names a family. Until suite.py has its own canonical_record(task) and
scene_summary(world, i), local_canonical_record and local_scene_summary below stand in for them;
switching is the one line that binds canonical_record and scene_summary.

Joining labels to demonstrations. Items are keyed by (seed, index): index n is the n-th task that
demos.draws() yields for that seed, counted over all chunks, which is line n of the store's
attempts.jsonl; that line's "e" is the episode's row in the store (-1: the expert failed and the
episode was not kept). The join holds when the store was written with the same --seed, --chunk,
--robots, --counts, --fit, --rule and --apart (the cube presets set robots, rule and apart) and
by the same suite code. export --store DIR reads those options from the store's small.pt, keeps
only the attempts that became episodes, records each item's "e" and checks every family and kind
against attempts.jsonl; Labels.by_episode() then maps store rows to their labels.
"""

import argparse
import functools
import hashlib
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from beebot.vla import caption
from beebot.vla.voxel import demos as D
from beebot.vla.voxel import suite as S
from beebot.vla.voxel import wordings as W
from beebot.vla.voxel.facts import groups
from beebot.vla.world import COLOUR_NAMES

ROOT = Path("data/vla/haiku")
TEST_NAME = "test"
TEST_SEED = 900_001  # the test set's draw; training exports may not use it
MARKER = "HELD_OUT"  # file in the test set's directory
FROZEN = "FROZEN"  # written once the test set is built; testset then refuses to run again
RUNGS = ("close", "indirect", "casual")
TEST_RUNGS = ("direct", "indirect")
REFUSAL = "refusal"  # the rung of an instruction that cannot or should not be followed
REFUSAL_KINDS = ("refuse", "clarify")
REFUSAL_SHARE = 0.10
DONE_LINE = "done."
MAX_THOUGHT_WORDS = 45  # besides the quoted instruction: about 60 tokens; the prompt asks for 35
MAX_SWITCH_WORDS = 12
GENERATE_BATCH = 20
VERIFY_BATCH = 25
TEST_BATCH = 15
WORKERS = 8
RETRIES = 6
BACKOFF = (5.0, 300.0)  # first delay and cap, seconds; doubled per attempt, with jitter
LOG_EVERY = 60.0
MENU_SEED = 4_242  # the draw the verifier's menu is read from (not the items')
MENU_SCENES = 200
ORDERED_KEYS = frozenset({"order", "sequence", "steps"})  # spec lists whose order matters; every other list is a set
IGNORED_KEYS = frozenset({"note", "program", "voxels"})  # spec keys same_spec does not compare (a build task's plan is not in its instruction)
EXTRA_COLOURS = ("black", "brown", "grey", "gray", "magenta", "indigo", "gold", "golden", "silver", "teal", "turquoise", "maroon", "beige", "violet", "lavender", "crimson", "scarlet", "pink")
TERRAIN_WORDS = {"trench": ("trench", "trenches", "gap", "ditch", "channel", "chasm", "ravine", "canyon", "gorge"), "plateau": ("plateau", "plateaus", "platform", "platforms", "mesa", "terrace"), "hill": ("hill", "hills", "hilly", "bump", "bumps", "mound")}
SCRIPTS = re.compile(r"[ऀ-ॿ぀-ヿ㐀-䶿一-鿿가-힯]")  # Devanagari, kana, CJK, Hangul: held-out languages
HELD_EXTRA = ("autumn", "japan", "france", "ukraine", "ukrainian", "rubik", "same colour as", "same color as", "coloured like", "colored like")


# ---------------------------------------------------------------------------------------------
# Adapters: canonical records and scene summaries of the existing families


def colour_order(spec):
    """The colours an eat task's plan eats, in plan order (suite.make_logic, make_runner)."""
    return list(spec.get("order") or sorted(spec.get("eat", ())) or sorted(spec.get("any_of", ()))[:1])


def local_spec(task):
    """The JSON spec of a task: what an instruction has to pin down, in this scene. Lists are sets
    unless their key is in ORDERED_KEYS. An eat task's forbid set is left out: it is every other
    cheese in the scene. An eat task whose colour is not in the scene is a refusal."""
    spec = task.spec
    kind = spec["kind"]
    if kind == "mask":
        slots = W.slots_of(task)
        out = {"kind": "shape", "shape": slots["s"]} if slots["intent"] == "shape" else {"kind": "letters", "word": slots["w"], "standing": slots["intent"] == "letter_stand"}
    elif kind == "tower":
        out = {"kind": "tower", "height": int(spec["height"])}
    elif kind in ("bridge", "fill", "stairs"):
        out = {"kind": kind}
    elif kind == "parallel":
        out = {"kind": "parallel", "jobs": [[job, x] for job, x in spec["jobs"]]}
    elif kind == "herd_trench":
        out = {"kind": "herd_trench", "cheese": spec["cheese"]}
    elif kind == "herd_region":
        out = {"kind": "herd_region", "cheeses": sorted(spec["cheeses"]), "region": spec["region"]}
    elif kind == "eat" and spec.get("form") == "absent":
        out = {"kind": "refuse"}
    elif kind == "eat":
        out = {"kind": "eat", "eat": sorted(spec["eat"]), "any_of": sorted(spec["any_of"]), "order": list(spec["order"])}
    else:
        raise ValueError(kind)
    return out


def local_subgoals(task, plain):
    """The plain sentence of each plan segment, in plan order: an eat segment eats the next colour
    of colour_order, every other segment is the whole task."""
    eat = W.INTENTS["sub.eat"]["canonical"]
    colours = colour_order(task.spec)
    out = []
    for seg in task.plan:
        if seg["kind"] == "eat":
            out.append(eat.format(x=colours[sum(s["kind"] == "eat" for s in task.plan[: len(out)])]))
        else:
            out.append(plain)
    return out


def local_canonical_record(task):
    """{family, kind, plain, spec, subgoals} of a suite task (stands in for suite.canonical_record)."""
    plain = W.canonical(W.slots_of(task))
    return {"family": task.family, "kind": S.task_kind(task), "plain": plain, "spec": local_spec(task), "subgoals": local_subgoals(task, plain)}


def region_of(y, x):
    """The floor area (suite.FLOOR_REGIONS) a point (y, x) lies in."""
    for name in S.FLOOR_REGIONS:
        y0, y1, x0, x1 = S.region_box(name)
        if y0 <= y < y1 and x0 <= x < x1:
            return name
    raise ValueError((y, x))


def terrain_of(height):
    """The terrain of a heightmap (H, W) in words: flat, a trench between two plateaus, or hills."""
    if height.max() == 0:
        return {"layout": "flat floor"}
    low = height == 0
    tops = np.unique(height[~low])
    for axis, runs in ((0, "back to front"), (1, "left to right")):
        band = low.all(axis=axis)
        if len(tops) == 1 and band.any() and (low == (band[None, :] if axis == 0 else band[:, None])).all():
            where = np.nonzero(band)[0]
            if where.max() - where.min() + 1 == len(where):
                along = "x" if axis == 0 else "y"
                return {"layout": "trench between two plateaus", "trench": {"runs": runs, f"from_{along}": int(where.min()), "width": len(where), "depth": int(tops[0])}, "plateau_height": int(tops[0])}
    return {"layout": "low hills", "highest": int(height.max())}


def local_scene_summary(world, i):
    """JSON of scene i at the start: gravity, terrain, the swarm and the cheeses with their floor
    area and lowest corner (x left to right, y back to front, z up). Stands in for
    suite.scene_summary."""
    walls = world.walls[i].numpy()
    mass = world.mass[i].numpy() > 0.5
    height = walls.sum(0)
    terrain = terrain_of(height)
    cells = np.argwhere(mass)
    robots = {"count": int(len(cells))}
    if len(cells):
        cy, cx = cells[:, 1].mean(), cells[:, 2].mean()
        robots.update(mostly_around=region_of(int(cy), int(cx)), clumps=len(groups(mass)))
    cheeses = []
    for k, colour in S.present(world, i):
        z, y, x = (int(v) for v in world.cheese_pos[i, k])
        on = "floor" if z == 0 else "plateau" if "trench" in terrain else "hill"
        cheeses.append({"colour": colour, "area": region_of(y + 1, x + 1), "on": on, "x": x, "y": y, "z": z})
    return {"gravity": bool(world.gravity[i]), "terrain": terrain, "robots": robots, "cheeses": cheeses}


# The suite's own versions: the same records for the six families (tests/test_vla_voxel_coverage.py
# checks it against the stand-ins above), records for the coverage families, and the scene with its
# robots' groups, towers and robots standing on robots added.
canonical_record, scene_summary = S.canonical_record, S.scene_summary

LOCAL_NOTES = {
    "eat": "eat: colours that must all be eaten. any_of: eat one of these (either/or). order: the colours in the order they must be eaten, only when the order matters, else []. No other cheese may be eaten; an instruction to eat nothing has all three lists empty.",
    "letters": "word: the capital letters to write. standing: true if they stand upright, false if they lie on the floor.",
    "shape": "shape: the shape the robots form.",
    "tower": "height: how many robots tall.",
    "parallel": "jobs: the two groups' jobs, [job, target] each: region with a floor area, tower with a height, surround or eat with a colour.",
    "herd_region": "cheeses: the cheeses to push (without eating them). region: the floor area to push them to.",
    "herd_trench": "cheese: the cheese to push into the trench (without eating it).",
    "bridge": "build a bridge across the trench.",
    "fill": "fill the trench with robots.",
    "stairs": "build stairs up to the plateau.",
    "refuse": "the instruction cannot or should not be followed in this scene: it asks for a colour or thing the scene does not have, or makes no sense. The swarm does nothing.",
    "clarify": "the instruction could mean two different tasks in this scene, so the swarm asks before acting.",
}
SPEC_NOTES = getattr(S, "SPEC_NOTES", LOCAL_NOTES)


# ---------------------------------------------------------------------------------------------
# Export


def item_id(seed, index):
    return f"s{seed}-{index:07d}"


def store_options(store):
    """The draw options a demo store was written with (small.pt's config) and its attempts."""
    import torch

    config = torch.load(Path(store) / "small.pt", map_location="cpu", weights_only=False)["config"]
    attempts = [json.loads(line) for line in (Path(store) / "attempts.jsonl").read_text().splitlines() if line.strip()]
    options = {"seed": config["seed"], "robots": config["robots"], "counts": config.get("counts", "uniform"), "fit": bool(config.get("fit")), "rule": config.get("rule"), "apart": bool(config.get("apart")), "coverage": float(config.get("coverage", 0.0)), "objects": float(config.get("objects", 0.0)), "build": float(config.get("build", 0.0)), "only": config.get("only")}
    return options, attempts


def drawn_items(options, limit, attempts=None, keep=None):
    """Items {id, seed, index, e, record, scene} in demos.draws order; with attempts (a store's
    attempts.jsonl rows) only those that became episodes, each checked against its row; with keep
    only the items it accepts."""
    seed = options["seed"]
    robots = None if options.get("robots") is None else tuple(options["robots"])
    scenes = 10**9  # full chunks until the limit or the store's attempts run out
    index = 0
    out = []
    for _, world, _, tasks, _ in D.draws(scenes, seed, options.get("chunk", 24), robots, options.get("counts", "uniform"), bool(options.get("fit")) or robots is not None, options.get("rule"), bool(options.get("apart")), float(options.get("coverage", 0.0)), float(options.get("objects", 0.0)), float(options.get("build", 0.0)), options.get("only")):
        for i, task in enumerate(tasks):
            e = None
            if attempts is not None:
                if index >= len(attempts):
                    return out
                row = attempts[index]
                if (row["family"], row["kind"]) != (task.family, S.task_kind(task)):
                    raise ValueError(f"attempt {index} is {row['family']}/{row['kind']} in the store but {task.family}/{S.task_kind(task)} here: different options, chunk or suite code")
                e = row["e"]
            if e is None or e >= 0:
                item = {"id": item_id(seed, index), "seed": seed, "index": index, "e": e, "record": canonical_record(task), "scene": scene_summary(world, i, task)}
                if task.family in S.COVERAGE:
                    item["voxels"] = start_voxels(world, i)
                if keep is None or keep(item):
                    out.append(item)
            index += 1
            if limit is not None and len(out) >= limit:
                return out
    return out


def start_voxels(world, i):
    """What a robot selector is resolved on (selection): the robots' voxels (z, y, x) and the
    centres of the alive cheeses in scene i at the start."""
    occupied = world.mass[i].numpy() > 0.5
    centres = S.cheese_centres(world.cheese_color[i].tolist(), S.snapshot(world, i))
    return {"shape": list(occupied.shape), "robots": np.argwhere(occupied).tolist(), "cheeses": {c: [float(v) for v in centre] for c, centre in centres.items()}}


def attach_voxels(items, options):
    """Items of an earlier export with start_voxels added to those of the coverage families, read
    off a fresh draw with the export's options (config.json); a record that differs from the
    draw's means other options or suite code and raises."""
    wanted = {item["index"]: item for item in items if item["record"]["family"] in S.COVERAGE and "voxels" not in item}
    if wanted:
        drawn = {item["index"]: item for item in drawn_items(options, max(wanted) + 1)}
        for index, item in wanted.items():
            if json.loads(json.dumps(drawn[index]["record"])) != item["record"]:
                raise ValueError(f"item {item['id']} is {item['record']['kind']} here but {drawn[index]['record']['kind']} in the draw: different options or suite code")
    return [{**item, "voxels": drawn[item["index"]]["voxels"]} if item["index"] in wanted else item for item in items]


def instruction_items(items):
    """One item per instruction: a switch episode (spec kind switch) becomes one item per part,
    "<id>#p<j>", with the episode's seed, index, e, scene (the start: the state at step t is not
    known before the rollout) and voxels, the part's record (family kept as the episode's), its
    step t, its number and the number of parts; every other item as it is."""
    out = []
    for item in items:
        record = item["record"]
        if record["spec"]["kind"] != "switch":
            out.append(item)
            continue
        parts = record["spec"]["parts"]
        for j, part in enumerate(parts):
            out.append({**item, "id": f"{item['id']}#p{j}", "record": {**part["record"], "family": record["family"]}, "part": j, "parts": len(parts), "t": int(part["t"])})
    return out


SELECTOR_KEYS = ("who", "by", "which", "except")  # with count when who is count (suite.who_fields)


def job_fields(spec):
    """A spec without its robot selector fields."""
    drop = set(SELECTOR_KEYS) | ({"count"} if spec.get("who") == "count" else set())
    return {k: v for k, v in spec.items() if k not in drop}


def side_group(groups, which):
    """The group furthest toward a side or a corner ("front right": furthest front plus right)."""
    words = which.split()
    if not words or not all(w in S.SIDE_AXES for w in words):
        return None
    return max(groups, key=lambda g: sum(S.SIDE_AXES[w][1] * g[:, S.SIDE_AXES[w][0]].mean() for w in words))


def selection(spec, voxels):
    """The robots a spec's selector fields pick at the start (suite.chosen): a frozenset of voxels,
    ("any", n) for count and half (any n robots do), or None when the selector names nothing in
    the scene or cannot be read. No selector: every robot."""
    occupied = np.zeros(voxels["shape"], bool)
    cells = np.asarray(voxels["robots"], int).reshape(-1, 3)
    occupied[tuple(cells.T)] = True
    who = str(spec.get("who", "all")).strip().lower()
    try:
        if who in ("count", "half"):
            return ("any", int(spec["count"]) if who == "count" else len(cells) // 2)
        selector = {"select": who, "by": str(spec.get("by", "")).strip().lower(), "which": str(spec.get("which", "")).strip().lower(), "except": spec.get("except") is True}
        if who == "group" and selector["by"] == "side" and selector["which"] not in S.SIDE_AXES:
            groups = S.components(occupied)
            group = side_group(groups, selector["which"]) if len(groups) >= 2 else None
            if group is None:
                return None
            pick = np.zeros_like(occupied)
            pick[tuple(group.T)] = True
            pick = occupied & ~pick if selector["except"] else pick
        else:
            pick = S.chosen(selector, occupied, {c: tuple(v) for c, v in voxels["cheeses"].items()})
    except (KeyError, ValueError, TypeError, IndexError):
        return None
    return None if pick is None else frozenset(map(tuple, np.argwhere(pick).tolist()))


def same_task(answer, want, voxels=None):
    """same_spec, or the same job fields and a selector that picks the same robots in the start
    scene's voxels (start_voxels): another selector for the same robots, "the larger group" for
    "everyone except the group at the back"."""
    if same_spec(answer, want):
        return True
    if not isinstance(answer, dict) or voxels is None or not ("who" in answer or "who" in want):
        return False
    if normal(job_fields(answer)) != normal(job_fields(want)):
        return False
    picked = selection(answer, voxels)
    return picked is not None and picked == selection(want, voxels)


def values_of(value):
    """The scalar values in a JSON value, flattened."""
    if isinstance(value, list):
        return [v for x in value for v in values_of(x)]
    return [value]


def describe(values):
    """Words for the values a field took: a whole number range, true or false, or the choices."""
    if all(isinstance(v, bool) for v in values):
        return "true or false"
    if all(isinstance(v, int) for v in values):
        return f"a whole number ({min(values)} to {max(values)})"
    distinct = sorted({str(v) for v in values})
    return "one of: " + ", ".join(distinct) if len(distinct) <= 12 else "text or a number, e.g. " + ", ".join(distinct[:: max(1, len(distinct) // 10)])  # many values: an open set, shown by example


def menu_from(records):
    """The verifier's menu: per spec kind its fields with the values seen, a note (SPEC_NOTES) and
    one plain example; refuse and clarify always. Read off records, so it names no family. A
    switch record contributes its parts (each is verified as its own instruction); the robot
    selector fields (SELECTOR_KEYS) are left out of the kinds and described once, after them,
    with the note SPEC_NOTES["who"] and an example."""
    records = [part["record"] for record in records for part in (record["spec"]["parts"] if record["spec"]["kind"] == "switch" else [{"record": record}])]
    kinds = {}
    selector_example = None
    fullness = lambda s: sum(bool(v) or v is False for v in s.values())  # noqa: E731
    for record in records:
        spec = record["spec"]
        if "who" in spec and record.get("plain") and (selector_example is None or fullness(spec) > fullness(selector_example[1])):
            selector_example = [record["plain"], spec]
        spec = job_fields(spec)
        entry = kinds.setdefault(spec["kind"], {"fields": {}, "example": None})
        if record.get("plain") and "who" not in record["spec"] and spec["kind"] not in REFUSAL_KINDS and (entry["example"] is None or fullness(spec) > fullness(entry["example"][1])):  # the fullest example of everyone acting
            entry["example"] = [record["plain"], spec]
        for key, value in spec.items():
            if key == "kind" or key in IGNORED_KEYS:
                continue
            field = entry["fields"].setdefault(key, {"list": isinstance(value, list), "pair": False, "values": []})
            if isinstance(value, list) and value and isinstance(value[0], list):
                field["pair"] = True
                field.setdefault("positions", [[] for _ in value[0]])
                for pair in value:
                    for p, v in enumerate(pair):
                        field["positions"][p].append(v)
            field["values"] += values_of(value)
    for kind in REFUSAL_KINDS:
        kinds.setdefault(kind, {"fields": {}, "example": None})
    pools = [set(field["values"]) for entry in kinds.values() for field in entry["fields"].values() if not field["pair"] and all(isinstance(v, str) for v in field["values"])]

    def widest(values):
        """The values of the largest field whose values include these (a rare field such as
        any_of then lists every colour, not only the ones a sample happened to draw)."""
        if not values or not all(isinstance(v, str) for v in values):
            return values
        return sorted(max((pool for pool in pools if set(values) <= pool), key=len, default=set(values)))

    lines = []
    for kind, entry in sorted(kinds.items()):
        parts = []
        for key, field in entry["fields"].items():
            if field["pair"]:
                words = "a list of [" + "; ".join(describe(widest(position)) for position in field["positions"]) + "] pairs"
            elif field["list"]:
                words = "a list, each " + describe(widest(field["values"])) if field["values"] else "a list"
            else:
                words = describe(widest(field["values"]))
            parts.append(f'"{key}": {words}{" (order matters)" if key in ORDERED_KEYS and field["list"] else ""}')
        line = f'- "{kind}"' + (" with " + "; ".join(parts) if parts else " (no fields)")
        if kind in SPEC_NOTES:
            line += ". " + SPEC_NOTES[kind]
        if entry["example"] is not None:
            line += f' Example: "{entry["example"][0]}" -> {json.dumps(entry["example"][1])}'
        lines.append(line)
    if selector_example is not None:
        lines.append('\nOnly some robots: any task kind above may add "who", "by", "which", "count" and "except". ' + SPEC_NOTES.get("who", "") + f' Example: "{selector_example[0]}" -> {json.dumps(selector_example[1])}')
    return "\n".join(lines)


@functools.lru_cache(maxsize=1)
def menu_records():
    return tuple(item["record"] for item in drawn_items({"seed": MENU_SEED}, MENU_SCENES * 3))


def export(directory, options, limit, store=None, dry_run=False, keep=None):
    """Write items.jsonl, config.json and menu.json; returns the items. keep: a predicate on items
    (the test set keeps those with held-out expressions to offer); the limit counts kept items."""
    attempts = None
    if store is not None:
        store_draw, attempts = store_options(store)
        options = {**options, **store_draw}
    if options["seed"] == TEST_SEED and directory.name != TEST_NAME:
        raise ValueError(f"seed {TEST_SEED} is the test set's")
    items = drawn_items(options, limit, attempts, keep)
    menu = menu_from(list(menu_records()) + [item["record"] for item in items])
    if dry_run:
        for item in items[:3]:
            print(json.dumps(item))
        print(f"(dry run) {len(items)} items would go to {directory / 'items.jsonl'}\n\nverifier menu:\n{menu}")
        return items
    if (directory / "items.jsonl").exists():
        raise FileExistsError(f"{directory / 'items.jsonl'} exists; answers key off its ids, so use a new --name")
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "items.jsonl", "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    (directory / "config.json").write_text(json.dumps({**options, "store": None if store is None else str(store), "items": len(items), "time": now()}, indent=1))
    (directory / "menu.json").write_text(json.dumps({"menu": menu}))
    print(f"{len(items)} items -> {directory / 'items.jsonl'}; families {count_by(items, lambda x: x['record']['family'])}")
    return items


def export_from(directory, path, dry_run=False):
    """A new dataset from the items of an earlier export (path: its items.jsonl or a subset in the
    same format, next to its config.json): start voxels attached by a fresh draw with that
    config's options (attach_voxels), the menu written anew."""
    path = Path(path)
    options = json.loads((path.parent / "config.json").read_text())
    items = attach_voxels(read_jsonl(path), options)
    menu = menu_from(list(menu_records()) + [item["record"] for item in items])
    if dry_run:
        print(f"(dry run) {len(items)} items would go to {directory / 'items.jsonl'}\n\nverifier menu:\n{menu}")
        return items
    if (directory / "items.jsonl").exists():
        raise FileExistsError(f"{directory / 'items.jsonl'} exists; answers key off its ids, so use a new --name")
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "items.jsonl", "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    (directory / "config.json").write_text(json.dumps({**options, "from_items": str(path), "items": len(items), "time": now()}, indent=1))
    (directory / "menu.json").write_text(json.dumps({"menu": menu}))
    print(f"{len(items)} items from {path} -> {directory / 'items.jsonl'}; families {count_by(items, lambda x: x['record']['family'])}")
    return items


# ---------------------------------------------------------------------------------------------
# Words: held-out terms, slot values, mentions


def mentions(text, terms):
    """The terms that occur in text as whole words or phrases, ignoring case."""
    low = text.lower()
    return sorted({t for t in terms if re.search(r"(?<![\w])" + re.escape(t.lower()) + r"(?![\w])", low)})


def held_terms():
    """Every held-out slot expression (wordings.slot_expressions), the bare held-out things, every
    expression of the colours and shapes the suite holds out, the held-out categories' names and
    the held-out frames (HELD_EXTRA)."""
    terms = {text for _, _, text in W.slot_expressions()["held_out"]}
    for name, entry in W.SLOTS["colour"]["values"].items():
        terms.update(bare for bare, _ in entry["things"]["held_out"])
        if W.suite_held(entry):
            terms.add(name)
            terms.update(text for split in ("train", "words", "languages", "frames") for text in W.all_forms("colour", name, split))
    for name, entry in W.SLOTS["shape"]["values"].items():
        if W.suite_held(entry):
            terms.update([entry["canonical"]] + entry["train"] + entry["held_out"])
    terms.update(cat["name"] for cat in W.CATEGORIES if cat["split"] == "held_out")
    terms.update(HELD_EXTRA)
    terms.update(t.replace("-", " ") for t in list(terms) if "-" in t)
    return sorted(t for t in terms if t.strip())


def avoid_list():
    """The held-out terms as the prompt lists them: no -colored/-coloured or colour-of variants."""
    terms = [t for t in held_terms() if not re.search(r"colou?red$|colou?r of|colou?r on|colou?rs? (leaves|you)|^an? [\w' ]+ colou?r$|^one of the", t) and "-" not in t]
    return sorted(set(terms), key=str.lower)


def leaks(text):
    """Held-out material in a training text: held-out terms, and scripts of held-out languages."""
    hits = mentions(text, HELD)
    if SCRIPTS.search(text):
        hits.append("held-out script")
    return hits


def slot_values(plain):
    """[(kind, value)] of the slot words in a plain sentence: colours, floor areas, shapes, counts,
    letters and lexical words (trench, plateau), read with the wordings vocabulary."""
    out = []
    rest = plain
    for kind, words in (("region", sorted(W.SLOTS["region"]["values"], key=len, reverse=True)), ("shape", sorted((e["canonical"] for e in W.SLOTS["shape"]["values"].values()), key=len, reverse=True))):
        for word in words:
            if re.search(r"(?<![\w])" + re.escape(word) + r"(?![\w])", rest):
                out.append((kind, word if kind == "region" else W.SHAPE_NAMES[word]))
                rest = re.sub(r"(?<![\w])" + re.escape(word) + r"(?![\w])", " ", rest)
    out += [("colour", c) for c in mentions(rest, list(W.SLOTS["colour"]["values"]))]
    out += [("count", int(n)) for n in re.findall(r"(?<![\w])(\d+)(?![\w])", rest)]
    out += [("letter", w) for w in re.findall(r"(?<![\w])([A-Z]{1,2})(?![\w])", rest)]
    out += [(kind, kind) for kind in W.LEXICAL if mentions(rest, [W.LEXICAL[kind]["canonical"]])]
    return out


def scene_words(scene):
    """Every word in a scene summary, keys and values."""
    words = set()

    def walk(value):
        if isinstance(value, dict):
            for key, inner in value.items():
                words.update(re.findall(r"[a-z]+", str(key).lower()))
                walk(inner)
        elif isinstance(value, list):
            for inner in value:
                walk(inner)
        else:
            words.update(re.findall(r"[a-z]+", str(value).lower()))

    walk(scene)
    return words


def scene_problems(text, scene, plain):
    """Colours and terrain a thought names that neither the scene nor the task's sentence has."""
    have = scene_words(scene) | set(re.findall(r"[a-z]+", plain.lower()))
    problems = [f"colour {c} not in scene" for c in mentions(text, list(COLOUR_NAMES) + list(EXTRA_COLOURS)) if c not in have]
    for name, words in TERRAIN_WORDS.items():
        if mentions(text, words) and name not in have and not (have & set(words)):
            problems.append(f"{name} not in scene")
    return problems


HELD = held_terms()
AVOID = avoid_list()


# ---------------------------------------------------------------------------------------------
# Prompts

GENERATE_PROMPT = """You write training text for a swarm of small cube robots in a 16 x 16 x 16 voxel world. The swarm can form shapes and letters, build towers and bridges, fill trenches, go to areas of the floor, surround, push or eat coloured 2 x 2 x 2 cheeses, and split into two groups that do two jobs. A model will learn from your text to understand what people ask and to think before it acts.

Each item below is one task: its plain sentence ("task"), the sub-goals the swarm works through, in this order ("sub_goals"), and the scene it starts in ("scene"). In the scene, x runs from left (0) to right (15), y from back (0) to front (15) and z is the height above the floor; the floor has nine areas: back left, back, back right, left, center, right, front left, front, front right.

For each item write three instructions a person might type to ask for exactly this task in this scene, from plain to indirect:
1. "close": a close paraphrase of the task in different words.
2. "indirect": it names the targets indirectly, so that working out what is meant takes a step of thought: a category of colours, a description (the colour of the sky), a relation in this scene (the cheese nearest the back, the one on the plateau), a count (both of them, all three) or a negation (everything but ...). It must pick out exactly the task's targets in this scene and nothing else.
3. "casual": the way people type in a chat: lower case, loose grammar, a typo or two, slang; now and then in Spanish or French.
Every instruction asks for the same thing as the task: the same targets, the same order where the task has one, nothing added and nothing left out. The sub-goals are only the order the swarm will choose to work in: unless the task sentence itself says an order (first, then, before, after), never put an order into the instruction. Vary the wording from item to item.

For each instruction write the swarm's first thought, in lower case and at most 35 words besides the quoted instruction, in this shape:
"you asked: <the instruction>. i see <the scene contents that matter>. <what the instruction means here, resolved to concrete colours, places or numbers>. first: <the first sub-goal>."
Keep the four parts in this order but word them freely, for example: "you asked: eat the cool coloured ones. i see red, green, blue and yellow cheeses. cool means green and blue here. first: the blue one."
If the task has no sub-goals, end with what the swarm does instead (it stays where it is).
Then "switches": one short lower-case line for each later sub-goal, in order, said when the swarm moves on to it ("now the green one", "green next"): one fewer than the sub-goals, none for a single sub-goal.
Thoughts mention only colours, cheeses and terrain that the scene has, by their plain names (red, green, blue, yellow, orange, purple, white, cyan, trench, plateau, hills); never invent anything.

Items with "later": true are a new instruction that arrives later, mid-task: the swarm has been working on an earlier instruction for a while and is then told this one as a separate message. The scene shows the start, so the robots (and anything they pushed) are no longer where it shows them: never refer to where the robots or things are or were in the instruction or the thought (no "the group at the back", "the cheese on the left"); "i see" names only what cannot have moved (the terrain, which cheeses there are). Write the instruction as a message of its own, without mentioning any earlier task.

Never use any of these words or phrases in an instruction or a thought (they are kept for testing): @AVOID@. Other languages: Spanish or French only.

Items with "refusal": true get one more instruction, one that cannot or should not be followed in this scene: it asks for a colour or thing the scene does not have, it could mean two different tasks here, or it makes no sense. Give its "kind" ("refuse" when the swarm should do nothing, "clarify" when it should ask), its "text", its "thought" (same shape, ending in why the swarm will not act or what is unclear) and its "reply": what the swarm says back, a short statement that it will do nothing and why, or a clarifying question.

Reply with JSON only, one object per item id and no other text:
{"<id>": {"instructions": [{"rung": "close", "text": "...", "thought": "...", "switches": []}, {"rung": "indirect", "text": "...", "thought": "...", "switches": []}, {"rung": "casual", "text": "...", "thought": "...", "switches": []}], "refusal": {"kind": "refuse", "text": "...", "thought": "...", "reply": "..."}}}
Leave "refusal" out for items without "refusal": true.

Items (one JSON object per line):
@ITEMS@
"""

TEST_PROMPT = """We are writing an exam for a robot swarm that follows typed instructions. The swarm is made of small cube robots in a 16 x 16 x 16 voxel world, with coloured 2 x 2 x 2 cheeses it can eat, push or surround, floor areas it can go to and shapes, letters and towers it can build. The exam checks whether it understands words it has never been taught.

Each item gives a task in plain words ("task"), the scene ("scene"; x runs left to right, y back to front, z up), the unfamiliar expressions to use ("use": for each plain word of the task, expressions that mean the same, and colour groups that pick out the right cheeses in this scene) and the plain words to keep out ("avoid").

For each item write two requests for exactly this task in this scene:
- "direct": a plain request in which every word listed under "avoid" is replaced by one of its expressions under "use".
- "indirect": a less direct request (a question, a polite or roundabout wish, a negation, a relation in the scene) that still uses only the expressions under "use" for those words.
Same targets, same order where the task has one, nothing added and nothing left out. Never write a word listed under "avoid". Keep the expressions as given (you may change their spelling of colour/color or add an article).

Reply with JSON only, one object per item id and no other text:
{"<id>": {"instructions": [{"rung": "direct", "text": "..."}, {"rung": "indirect", "text": "..."}]}}

Items (one JSON object per line):
@ITEMS@
"""

VERIFY_PROMPT = """Someone typed an instruction to a swarm of small cube robots in a 16 x 16 x 16 voxel world with coloured 2 x 2 x 2 cheeses. For each instruction below, work out which task it asks for in the scene shown with it, and write that task in the structured form below. In the scene, x runs from left (0) to right (15), y from back (0) to front (15) and z is the height; the floor has nine areas: back left, back, back right, left, center, right, front left, front, front right.

Task kinds and their fields:
@MENU@

Rules:
- Resolve the instruction in its own scene: every colour, area and number you write is a concrete value from the lists above, and colours are cheeses that are in the scene.
- Lists are sets unless the field says the order matters.
- A task needs only what its note says: line, gather, spread, stay, groups and move need nothing in the scene but the robots.
- Add "who" and its fields only when the instruction names some of the robots; "all of you", "everyone" or a count of every robot means no "who".
- "refuse" when the instruction cannot or should not be followed in its scene (it asks for a colour or thing the scene does not have, or it makes no sense); "clarify" when it could mean two different tasks there.
- Instructions may be casual, misspelt or in another language.

Reply with JSON only, one task per id and no other text, e.g. {"<id>": {"kind": "tower", "height": 4}}

Instructions (one JSON object per line):
@ITEMS@
"""


def generate_entry(item, refusal_share):
    entry = {"id": item["id"], "task": item["record"]["plain"], "sub_goals": item["record"]["subgoals"], "scene": item["scene"]}
    if item.get("part", 0) > 0:
        entry["later"] = True
    if refusal_wanted(item, refusal_share):
        entry["refusal"] = True
    return entry


def wants_refusal(identifier, share):
    return int(hashlib.sha1(identifier.encode()).hexdigest()[:8], 16) % 1000 < share * 1000


def refusal_wanted(item, share):
    """Whether an item also gets a refusal instruction: never a switch's later part (a refusal
    pairs with the episode's first frame)."""
    return item.get("part", 0) == 0 and wants_refusal(item["id"], share)


def test_offers(item, rng, per=4):
    """(use, avoid) of a test item: held-out expressions for each slot word of its sentence
    (wordings.entries of the words, languages and frames splits) and held-out categories that pick
    out its colours in the scene; the plain words that have such expressions are to be avoided,
    with their training synonyms."""
    present = sorted({c for c in W.SLOTS["colour"]["values"] if c in scene_words(item["scene"])})
    use, avoid = {}, set()
    colours = []
    for kind, value in slot_values(item["record"]["plain"]):
        splits = ("words", "languages", "frames") if kind == "colour" else ("words", "languages")
        forms = sorted({text for split in splits for _, group in W.entries(kind, value, split) for text, _ in group})
        if kind == "colour":
            colours.append(value)
        if not forms:
            continue
        word = W.plain(kind, value)
        use[word] = [forms[j] for j in sorted(rng.choice(len(forms), size=min(per, len(forms)), replace=False))]
        if kind != "letter":  # a letter's held-out frames keep the letter ("uppercase S")
            avoid.add(word)
            avoid.update(text for _, group in W.entries(kind, value, "train") for text, _ in group)
    targets = [colours] + [[c] for c in colours] if colours else []
    category_forms = sorted({text for target in targets for _, group in W.category_entries(target, present, "categories") for text, _ in group})
    if category_forms:
        use["colour groups"] = category_forms[:per]
    return use, sorted(avoid)


def test_entry(item):
    use, avoid = test_offers(item, item_rng(item))
    return {"id": item["id"], "task": item["record"]["plain"], "scene": item["scene"], "use": use, "avoid": avoid}


def items_block(entries):
    return "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries)


def generate_prompt(entries):
    return GENERATE_PROMPT.replace("@AVOID@", ", ".join(AVOID)).replace("@ITEMS@", items_block(entries))


def test_prompt(entries):
    return TEST_PROMPT.replace("@ITEMS@", items_block(entries))


def verify_prompt(entries, menu):
    return VERIFY_PROMPT.replace("@MENU@", menu).replace("@ITEMS@", items_block(entries))


# ---------------------------------------------------------------------------------------------
# Replies


def parse_json(text):
    """The JSON object a reply holds (code fences and text around it allowed), or None. A reply
    that gives one object per item (several top-level objects) is merged into one."""
    decoder, merged, at = json.JSONDecoder(), {}, 0
    while (start := text.find("{", at)) >= 0:
        try:
            part, end = decoder.raw_decode(text, start)
        except ValueError:
            at = start + 1
            continue
        if isinstance(part, dict):
            merged.update(part)
        at = end
    return merged or None


def clean(text):
    return re.sub(r"\s+", " ", text).strip() if isinstance(text, str) else ""


def item_rng(item):
    return np.random.default_rng(int(hashlib.sha1(item["id"].encode()).hexdigest()[:8], 16))


def text_problem(answer):
    """Why an instruction answer has no usable text, or None."""
    if not isinstance(answer, dict):
        return "not an object"
    return None if clean(answer.get("text")) else "no text"


def check_test(answer, item, rung):
    """(pair, None) or (None, reason) of a test-set instruction: it uses a held-out expression
    offered for the item and none of the plain or training words it was to avoid."""
    problem = text_problem(answer)
    text = clean(answer.get("text")) if problem is None else ""
    if problem is None:
        use, avoid = test_offers(item, item_rng(item))
        if not any(form.lower() in text.lower() for forms in use.values() for form in forms):
            problem = "uses no held-out expression"
        elif mentions(text, avoid):
            problem = f"uses training words {mentions(text, avoid)}"
    return (None, problem) if problem else ({"rung": rung, "instruction": text}, None)


def thought_problem(answer, rung, want):
    """Why the text, thought and switch lines of a training answer fail the shared checks (length,
    number of switches, held-out words), or None."""
    problem = text_problem(answer)
    if problem is not None:
        return problem
    text, thought = clean(answer.get("text")), clean(answer.get("thought")).lower()
    switches = answer.get("switches", [])
    reasoning = thought.replace(text.lower(), " ")
    if not thought:
        problem = "no thought"
    elif not isinstance(switches, list) or not all(isinstance(s, str) and s.strip() for s in switches):
        problem = "switches not a list of lines"
    elif rung != REFUSAL and len(switches) != want:
        problem = f"{len(switches)} switches for {want} later sub-goals"
    elif len(reasoning.split()) > MAX_THOUGHT_WORDS:
        problem = f"thought of {len(reasoning.split())} words besides the instruction"
    elif any(len(s.split()) > MAX_SWITCH_WORDS for s in switches):
        problem = "switch line too long"
    elif leaks(" ".join([text, thought, clean(answer.get("reply"))] + [clean(s) for s in switches])):
        problem = f"held-out words {leaks(' '.join([text, thought, clean(answer.get('reply'))] + [clean(s) for s in switches]))}"
    return problem


def training_pair(answer, rung):
    thought = clean(answer.get("thought")).lower()
    return {"rung": rung, "instruction": clean(answer.get("text")), "thought": thought, "switches": [clean(s).lower() for s in answer.get("switches", [])], "done": DONE_LINE}


def check_follow(answer, item, rung):
    """(pair, None) or (None, reason) of an instruction the swarm follows: the shared checks, then
    the thought names nothing the scene lacks and names each sub-goal's colours in order."""
    record = item["record"]
    problem = thought_problem(answer, rung, max(len(record["subgoals"]) - 1, 0))
    pair = training_pair(answer, rung) if problem is None else None
    if pair is not None:
        reasoning = pair["thought"].replace(pair["instruction"].lower(), " ")
        problems = scene_problems(" ".join([reasoning] + pair["switches"]), item["scene"], record["plain"])
        goals = record["subgoals"]
        unnamed_colours = [c for c in mentions(goals[0], COLOUR_NAMES) if not mentions(reasoning, [c])] if goals else []
        unnamed_colours += [c for goal, line in zip(goals[1:], pair["switches"]) for c in mentions(goal, COLOUR_NAMES) if not mentions(line, [c])]
        if unnamed_colours:
            problems.append(f"sub-goal colours {unnamed_colours} not named in order")
        problem = "; ".join(problems) or None
    return (None, problem) if problem else (pair, None)


def check_refusal(answer, item):
    """(pair, None) or (None, reason) of an instruction that cannot or should not be followed: the
    shared checks, a kind (refuse or clarify) and a reply. Its thought may name what is absent."""
    problem = thought_problem(answer, REFUSAL, 0)
    if problem is None and (answer.get("kind") not in REFUSAL_KINDS or not clean(answer.get("reply"))):
        problem = "refusal without a kind or reply"
    pair = None if problem else {**training_pair(answer, REFUSAL), "switches": [], "kind": answer["kind"], "reply": clean(answer["reply"])}
    return pair, problem


def check_item(item, answer, mode, refusal_share):
    """(pairs, rejects) of an item's answer; an answer that is not an object rejects every rung."""
    rungs = TEST_RUNGS if mode == "test" else RUNGS
    if not isinstance(answer, dict) or not isinstance(answer.get("instructions"), list):
        return [], [{"rung": "*", "reason": "malformed item", "answer": answer}]
    by_rung = {}
    for entry in answer["instructions"]:
        if isinstance(entry, dict) and entry.get("rung") in rungs and entry["rung"] not in by_rung:
            by_rung[entry["rung"]] = entry
    pairs, rejects = [], []
    checks = [(rung, by_rung.get(rung), (lambda a, r=rung: check_test(a, item, r)) if mode == "test" else (lambda a, r=rung: check_follow(a, item, r))) for rung in rungs]
    if mode != "test" and refusal_wanted(item, refusal_share):
        checks.append((REFUSAL, answer.get("refusal"), lambda a: check_refusal(a, item)))
    for rung, entry, check in checks:
        pair, reason = check(entry) if entry is not None else (None, "missing")
        if pair is None:
            rejects.append({"rung": rung, "reason": reason, "answer": entry})
        else:
            pairs.append({**pair, "pair_id": f"{item['id']}:{rung}"})
    return pairs, rejects


# ---------------------------------------------------------------------------------------------
# Calls: batches over workers, retries with backoff, a throughput line every minute


def haiku(prompt):
    return caption.call_provider("haiku", prompt)


class Echo:
    """A provider that prints every prompt and reply (dry runs)."""

    def __init__(self, provider):
        self.provider = provider
        self.lock = threading.Lock()

    def __call__(self, prompt):
        reply = self.provider(prompt)
        with self.lock:
            print("=" * 100 + "\nPROMPT\n" + "=" * 100 + "\n" + prompt + "\n" + "-" * 100 + "\nREPLY (fake)\n" + "-" * 100 + "\n" + reply + "\n", flush=True)
        return reply


def backoff(attempt, rng):
    first, cap = BACKOFF
    return min(cap, first * 2**attempt) * (0.5 + rng.random())


class Meter:
    """Counts items, calls and failures; a thread prints a throughput line every `every` seconds
    and appends it to log."""

    def __init__(self, stage, total, log, every=LOG_EVERY):
        self.stage, self.total, self.log, self.every = stage, total, log, every
        self.items = self.calls = self.failures = 0
        self.mark = (0, 0, time.time())
        self.start = time.time()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def add(self, items=0, calls=0, failures=0):
        with self.lock:
            self.items += items
            self.calls += calls
            self.failures += failures

    def line(self):
        with self.lock:
            items, calls, at = self.mark
            minutes = max(time.time() - at, 1e-9) / 60
            out = {"stage": self.stage, "minutes": round((time.time() - self.start) / 60, 2), "items_per_min": round((self.items - items) / minutes, 2), "calls_per_min": round((self.calls - calls) / minutes, 2), "items": self.items, "of": self.total, "calls": self.calls, "failures": self.failures, "time": now()}
            self.mark = (self.items, self.calls, time.time())
        return out

    def emit(self):
        out = self.line()
        print(json.dumps(out), flush=True)
        if self.log is not None:
            append(self.log, out)

    def loop(self):
        while not self.stop.wait(self.every):
            self.emit()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.emit()


def run_batches(batches, prompt_of, ids_of, provider, on_answers, meter, calls_log, failures_log, workers=WORKERS, retries=RETRIES, sleep=time.sleep, seed=0):
    """Send each batch's prompt, retrying with exponential backoff on an error or a reply with no
    JSON in it; on_answers(batch, answers, call) gets the reply's objects by id (under a lock).
    Ids a reply leaves out go to failures_log and stay pending for the next pass."""
    lock = threading.Lock()
    rng = random.Random(seed)

    def work(job):
        call, batch = job
        prompt = prompt_of(batch)
        error = None
        for attempt in range(retries + 1):
            start_time = time.time()
            try:
                answer = parse_json(provider(prompt))
                if answer is None:
                    raise ValueError("no JSON object in the reply")
                error = None
                break
            except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
                error = f"{type(exc).__name__}: {str(exc)[-300:]}"
                meter.add(failures=1)
                append(calls_log, {"call": call, "attempt": attempt, "items": len(batch), "seconds": round(time.time() - start_time, 1), "error": error, "time": now()})
                if attempt < retries:
                    with lock:
                        delay = backoff(attempt, rng)
                    sleep(delay)
        meter.add(calls=1)
        if error is not None:
            for identifier in ids_of(batch):
                append(failures_log, {"id": identifier, "call": call, "error": error, "time": now()})
            return
        asked_ids = ids_of(batch)
        answers = {k: v for k, v in answer.items() if k in asked_ids}
        with lock:
            on_answers(batch, answers, call)
            for identifier in asked_ids:
                if identifier not in answers:
                    append(failures_log, {"id": identifier, "call": call, "error": "missing from the reply", "time": now()})
        meter.add(items=len(answers))
        append(calls_log, {"call": call, "attempt": attempt, "items": len(batch), "answered": len(answers), "seconds": round(time.time() - start_time, 1), "error": None, "time": now()})

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(work, list(enumerate(batches))))


# ---------------------------------------------------------------------------------------------
# Stages


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line cut short by a killed run
    return rows


def append(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def count_by(rows, key):
    out = {}
    for row in rows:
        out[key(row)] = out.get(key(row), 0) + 1
    return dict(sorted(out.items()))


def chunks(rows, size):
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def generate(directory, provider, mode="train", batch=None, workers=WORKERS, limit=None, passes=2, refusal_share=REFUSAL_SHARE, log_every=LOG_EVERY, sleep=time.sleep, verbose=False):
    """Answer every item of items.jsonl not yet in generated.jsonl; returns the rows written."""
    items = instruction_items(read_jsonl(directory / "items.jsonl"))[:limit]
    batch = batch or (TEST_BATCH if mode == "test" else GENERATE_BATCH)
    entry_of = test_entry if mode == "test" else (lambda item: generate_entry(item, refusal_share))
    prompt_of = test_prompt if mode == "test" else generate_prompt
    by_id = {item["id"]: item for item in items}
    out_path = directory / "generated.jsonl"
    rows = []

    def on_answers(entries, answers, call):
        for identifier, answer in answers.items():
            item = by_id[identifier]
            pairs, rejects = check_item(item, answer, mode, refusal_share)
            row = {"id": identifier, "seed": item["seed"], "index": item["index"], "e": item.get("e"), **part_keys(item), "family": item["record"]["family"], "kind": item["record"]["kind"], "pairs": pairs, "rejects": rejects, "call": call, "time": now()}
            append(out_path, row)
            rows.append(row)
            if verbose:
                print("PARSED " + json.dumps(row, ensure_ascii=False, indent=1), flush=True)

    for _ in range(passes):
        answered_ids = {row["id"] for row in read_jsonl(out_path)}
        todo = [item for item in items if item["id"] not in answered_ids]
        if not todo:
            break
        random.Random(len(todo)).shuffle(todo)  # mixes families and scenes within a call (contrast tasks share a scene)
        with Meter("generate" if mode == "train" else "testset-generate", len(todo), directory / "throughput.jsonl", log_every) as meter:
            run_batches(chunks([entry_of(item) for item in todo], batch), prompt_of, lambda entries: {e["id"] for e in entries}, provider, on_answers, meter, directory / "calls.jsonl", directory / "failures.jsonl", workers, sleep=sleep)
    return rows


def normal(value, key=None):
    """A spec value in a form two equal specs share: lower-case trimmed text, whole numbers from
    numeric text, and lists sorted unless their key is in ORDERED_KEYS."""
    if isinstance(value, dict):
        return {k: normal(v, k) for k, v in value.items() if k not in IGNORED_KEYS}
    if isinstance(value, (list, tuple)):
        out = [normal(v) for v in value]
        return out if key in ORDERED_KEYS else sorted(out, key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, str):
        text = value.strip().lower()
        return int(text) if re.fullmatch(r"-?\d+", text) else text
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def same_spec(answer, want):
    return isinstance(answer, dict) and normal(answer) == normal(want)


def part_keys(item):
    """A switch part's place in its episode (instruction_items): part, parts and t; {} otherwise."""
    return {key: item[key] for key in ("part", "parts", "t") if key in item}


def pair_record(item, pair):
    """The record a pair must verify to: its item's, or refuse/clarify for the refusal rung."""
    if pair["rung"] == REFUSAL:
        return {"family": item["record"]["family"], "kind": pair["kind"], "plain": "", "spec": {"kind": pair["kind"]}, "subgoals": []}
    return item["record"]


def verify(directory, provider, batch=VERIFY_BATCH, workers=WORKERS, passes=2, log_every=LOG_EVERY, sleep=time.sleep, verbose=False, seed=0):
    """Read back every generated pair not yet decided; keep it if the spec matches."""
    items = {item["id"]: item for item in instruction_items(read_jsonl(directory / "items.jsonl"))}
    menu = json.loads((directory / "menu.json").read_text())["menu"]
    kept_path, rejected_path = directory / "verified.jsonl", directory / "rejected.jsonl"
    pairs = {}
    for row in read_jsonl(directory / "generated.jsonl"):
        for pair in row["pairs"]:
            pairs[pair["pair_id"]] = (items[row["id"]], pair)

    def on_answers(entries, answers, call):
        for pair_id, answer in answers.items():
            item, pair = pairs[pair_id]
            record = pair_record(item, pair)
            row = {**pair, "id": item["id"], "seed": item["seed"], "index": item["index"], "e": item.get("e"), **part_keys(item), "family": record["family"], "kind": record["kind"], "spec": record["spec"], "read_as": answer, "call": call, "time": now()}
            kept = same_task(answer, record["spec"], item.get("voxels"))
            if kept:
                append(kept_path, row)
            else:
                append(rejected_path, {**row, "reason": "read as a different task" if isinstance(answer, dict) else "malformed answer"})
            if verbose:
                print(("KEPT " if kept else "REJECTED ") + json.dumps({"pair_id": pair_id, "instruction": pair["instruction"], "spec": record["spec"], "read_as": answer}, ensure_ascii=False), flush=True)

    for _ in range(passes):
        verdict_ids = {row["pair_id"] for row in read_jsonl(kept_path) + read_jsonl(rejected_path)}
        todo = [key for key in pairs if key not in verdict_ids]
        if not todo:
            break
        rng = random.Random(seed)
        batches = []
        for rung in dict.fromkeys(RUNGS + TEST_RUNGS + (REFUSAL,)):  # one rung per batch: an item's instructions never share a call
            group = [key for key in todo if pairs[key][1]["rung"] == rung]
            rng.shuffle(group)
            batches += chunks([{"id": key, "instruction": pairs[key][1]["instruction"], "scene": pairs[key][0]["scene"]} for key in group], batch)
        with Meter("verify", len(todo), directory / "throughput.jsonl", log_every) as meter:
            run_batches(batches, lambda entries: verify_prompt(entries, menu), lambda entries: {e["id"] for e in entries}, provider, on_answers, meter, directory / "calls.jsonl", directory / "failures.jsonl", workers, sleep=sleep)
    return report(directory)


def report(directory, show=True):
    """Keep rates per family and per rung: of the instructions asked for (generated), of those that
    passed the programmatic checks, and of those the verifier read back the same."""
    generated_rows = read_jsonl(directory / "generated.jsonl")
    keeps = read_jsonl(directory / "verified.jsonl")
    rejects = read_jsonl(directory / "rejected.jsonl")
    out = {"items": len(instruction_items(read_jsonl(directory / "items.jsonl"))), "answered": len(generated_rows), "pairs": sum(len(r["pairs"]) for r in generated_rows), "kept": len(keeps), "by_family": {}, "by_rung": {}, "check_rejects": {}, "time": now()}
    for name, key in (("by_family", lambda r: r["family"]), ("by_rung", lambda r: r["rung"])):
        table = {}
        for row in generated_rows:
            for pair in row["pairs"]:
                table.setdefault(key({**pair, "family": row["family"]}), {"checked": 0, "kept": 0, "read_differently": 0, "check_rejects": 0})["checked"] += 1
            for reject in row["rejects"]:
                table.setdefault(key({**reject, "family": row["family"]}), {"checked": 0, "kept": 0, "read_differently": 0, "check_rejects": 0})["check_rejects"] += 1
        for row in keeps:
            table.setdefault(key(row), {"checked": 0, "kept": 0, "read_differently": 0, "check_rejects": 0})["kept"] += 1
        for row in rejects:
            table.setdefault(key(row), {"checked": 0, "kept": 0, "read_differently": 0, "check_rejects": 0})["read_differently"] += 1
        for entry in table.values():
            verdicts = entry["kept"] + entry["read_differently"]
            entry["verify_keep_rate"] = round(entry["kept"] / verdicts, 3) if verdicts else None
            entry["overall_keep_rate"] = round(entry["kept"] / (entry["checked"] + entry["check_rejects"]), 3) if entry["checked"] + entry["check_rejects"] else None
        out[name] = dict(sorted(table.items()))
    for row in generated_rows:
        for reject in row["rejects"]:
            reason = re.sub(r"\[.*\]|\d+", "#", str(reject["reason"]))
            out["check_rejects"][reason] = out["check_rejects"].get(reason, 0) + 1
    (directory / "report.json").write_text(json.dumps(out, indent=1))
    if show:
        print(f"{out['answered']}/{out['items']} items answered, {out['pairs']} pairs passed the checks, {out['kept']} kept by the verifier")
        for name in ("by_family", "by_rung"):
            print(f"\n{name[3:]:<14} {'checked':>8} {'kept':>6} {'differs':>8} {'chk-rej':>8} {'verify%':>8} {'overall%':>9}")
            for key, entry in out[name].items():
                pct = lambda v: "-" if v is None else f"{100 * v:.0f}"  # noqa: E731
                print(f"{key:<14} {entry['checked']:>8} {entry['kept']:>6} {entry['read_differently']:>8} {entry['check_rejects']:>8} {pct(entry['verify_keep_rate']):>8} {pct(entry['overall_keep_rate']):>9}")
        if out["check_rejects"]:
            print("\ncheck rejects by reason: " + json.dumps(dict(sorted(out["check_rejects"].items(), key=lambda kv: -kv[1]))))
    return out


def review(directory, path=None):
    """Every item with its pairs and verdicts, as text to read by hand."""
    items = instruction_items(read_jsonl(directory / "items.jsonl"))
    answers_by_id ={row["id"]: row for row in read_jsonl(directory / "generated.jsonl")}
    verdicts = {row["pair_id"]: ("KEPT", row) for row in read_jsonl(directory / "verified.jsonl")}
    verdicts.update({row["pair_id"]: ("READ AS " + json.dumps(row["read_as"]), row) for row in read_jsonl(directory / "rejected.jsonl")})
    lines = []
    for item in items:
        record, scene = item["record"], item["scene"]
        lines += ["", "#" * 100, f"{item['id']}  {record['family']}/{record['kind']}  task: {record['plain']}", f"  sub-goals: {record['subgoals']}", f"  spec: {json.dumps(record['spec'])}", f"  scene: {json.dumps(scene)}"]
        row = answers_by_id.get(item["id"])
        if row is None:
            lines.append("  (not answered)")
            continue
        for pair in row["pairs"]:
            verdict = verdicts.get(pair["pair_id"], ("(not verified)", None))[0]
            lines += [f"  [{pair['rung']}{'/' + pair['kind'] if 'kind' in pair else ''}] {pair['instruction']}", f"      -> {verdict}"]
            if "thought" in pair:
                lines.append(f"      thought: {pair['thought']}")
            if pair.get("switches"):
                lines.append(f"      switches: {pair['switches']}")
            if pair.get("reply"):
                lines.append(f"      reply: {pair['reply']}")
        for reject in row["rejects"]:
            lines.append(f"  [{reject['rung']}] CHECK REJECT ({reject['reason']}): {json.dumps(reject['answer'], ensure_ascii=False)}")
    text = "\n".join(lines)
    if path is not None:
        Path(path).write_text(text)
    return text


# ---------------------------------------------------------------------------------------------
# A fake provider for tests and dry runs


class FakeHaiku:
    """Answers every prompt of this module without a model: generate rewrites the plain sentence,
    testset swaps in the first offered expression, verify reads back what it wrote (instructions it
    did not write read as refuse). fail_calls: the first n calls raise a rate-limit error;
    malformed_ids: answered with a broken object; dropped_ids: left out of replies."""

    def __init__(self, fail_calls=0, malformed_ids=(), dropped_ids=(), garbage_calls=0):
        self.fail_calls, self.garbage_calls = fail_calls, garbage_calls
        self.malformed_ids, self.dropped_ids = set(malformed_ids), set(dropped_ids)
        self.memory = {}
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self, prompt):
        with self.lock:
            self.calls += 1
            calls = self.calls
        if calls <= self.fail_calls:
            raise RuntimeError("claude exited 1: rate limit reached (fake)")
        if calls <= self.fail_calls + self.garbage_calls:
            return "Sorry, I can't help with that."
        entries = [json.loads(line) for line in prompt.split("one JSON object per line):\n", 1)[1].splitlines() if line.strip()]
        out = {}
        for entry in entries:
            if entry["id"] in self.dropped_ids:
                continue
            if entry["id"] in self.malformed_ids:
                out[entry["id"]] = {"instructions": "oops"}
                continue
            out[entry["id"]] = self.verify(entry) if "instruction" in entry else self.test(entry) if "use" in entry else self.generate(entry)
        return "```json\n" + json.dumps(out, ensure_ascii=False) + "\n```"

    def remember(self, text, spec):
        with self.lock:
            self.memory[text] = spec

    def generate(self, entry):
        task, goals = entry["task"], entry["sub_goals"]
        colours = [c["colour"] for c in entry["scene"].get("cheeses", [])]
        sight = ("i see " + ", ".join(colours) + " cheeses") if colours else "i see the swarm"
        first = goals[0] if goals else "nothing to do, i stay where i am"
        spec = entry.get("_spec")
        instructions = []
        for rung, text in zip(RUNGS, (task, "could you " + task + " please", task.lower() + " pls")):
            self.remember(text, spec)
            instructions.append({"rung": rung, "text": text, "thought": f"you asked: {text}. {sight}. that means: {task}. first: {first}.", "switches": [goal.replace("eat the ", "now the ") for goal in goals[1:]]})
        answer = {"instructions": instructions}
        if entry.get("refusal"):
            text = "eat the black cheese"
            self.remember(text, {"kind": "refuse"})
            answer["refusal"] = {"kind": "refuse", "text": text, "thought": f"you asked: {text}. {sight}. there is no black cheese. so i stay put.", "reply": "there is no black cheese here, so i will stay put."}
        return answer

    def test(self, entry):
        text = entry["task"]
        for word, forms in entry["use"].items():
            if word in text:
                text = re.sub(r"(?<![\w])" + re.escape(word) + r"(?![\w])", forms[0], text)
        self.remember(text, entry.get("_spec"))
        self.remember("please, " + text, entry.get("_spec"))
        return {"instructions": [{"rung": "direct", "text": text}, {"rung": "indirect", "text": "please, " + text}]}

    def verify(self, entry):
        spec = self.memory.get(entry["instruction"], entry.get("_spec"))
        return spec if spec is not None else {"kind": "refuse"}


class SpecHints:
    """Wraps FakeHaiku so that it knows each item's spec: the prompt it is called with is the real
    one, and the hints reach only the fake (added to its copy of the entries). A pair id
    "<item id>:<rung>" gets its item's spec, or refuse for the refusal rung."""

    def __init__(self, fake, items):
        self.fake = fake
        self.specs = {item["id"]: item["record"]["spec"] for item in instruction_items(items)}

    def __call__(self, prompt):
        head, sep, body = prompt.partition("one JSON object per line):\n")
        lines = []
        for line in body.splitlines():
            if line.strip():
                entry = json.loads(line)
                identifier, _, rung = entry["id"].partition(":")
                if identifier in self.specs:
                    entry["_spec"] = {"kind": "refuse"} if rung == REFUSAL else self.specs[identifier]
                line = json.dumps(entry, ensure_ascii=False)
            lines.append(line)
        return self.fake(head + sep + "\n".join(lines))


# ---------------------------------------------------------------------------------------------
# The loader for training


class Labels:
    """Verified labels of a training set: choices(seed, index) -> [{rung, instruction, thought,
    switches, done}], refusals(seed, index) -> [{kind, instruction, thought, reply}] (these pair
    with the episode's first frame and doing nothing, not with its actions), by_episode(store)
    -> {store row e: choices}. A switch episode's instructions are separate user messages:
    switch_parts(seed, index) -> [{part, t, choices}] (part j's instruction is given at step t;
    choices() has none for it) and switches_by_episode(store) -> {e: switch_parts}. Refuses the
    test set, and checks that no training pair shares an episode key or an instruction with it."""

    def __init__(self, name, root=ROOT):
        self.directory = Path(root) / name
        assert not is_test(self.directory), f"{self.directory} is the held-out test set; it never goes into training"
        self.rows = read_jsonl(self.directory / "verified.jsonl")
        test = Path(root) / TEST_NAME
        if test.exists():
            test_rows = read_jsonl(test / "verified.jsonl") + read_jsonl(test / "items.jsonl")
            keys = {(row["seed"], row["index"]) for row in test_rows}
            texts = {row["instruction"].lower() for row in test_rows if "instruction" in row}
            assert not keys & {(row["seed"], row["index"]) for row in self.rows}, "training labels share episodes with the test set"
            assert not texts & {row["instruction"].lower() for row in self.rows}, "training labels share instructions with the test set"
        self.by_key, self.part_key = {}, {}
        for row in self.rows:
            self.by_key.setdefault((row["seed"], row["index"]), []).append(row)
            if "part" in row:
                self.part_key.setdefault((row["seed"], row["index"]), []).append(row)

    def choices(self, seed, index):
        return [choice_of(r) for r in self.by_key.get((seed, index), []) if r["rung"] != REFUSAL and "part" not in r]

    def refusals(self, seed, index):
        return [{"kind": r["kind"], "instruction": r["instruction"], "thought": r["thought"], "reply": r["reply"]} for r in self.by_key.get((seed, index), []) if r["rung"] == REFUSAL]

    def switch_parts(self, seed, index):
        """[{part, t, choices}] of a switch episode in part order; [] unless every part has a
        verified instruction."""
        rows = [r for r in self.part_key.get((seed, index), []) if r["rung"] != REFUSAL]
        if not rows:
            return []
        parts = [{"part": j, "t": next((r["t"] for r in rows if r["part"] == j), None), "choices": [choice_of(r) for r in rows if r["part"] == j]} for j in range(rows[0]["parts"])]
        return parts if all(part["choices"] for part in parts) else []

    def by_episode(self, store, seed=None):
        """{e: choices} for a demo store written from the same draw: row n of attempts.jsonl is index n."""
        seed, attempts = self.store_rows(store, seed)
        return {row["e"]: self.choices(seed, n) for n, row in enumerate(attempts) if row["e"] >= 0 and (seed, n) in self.by_key and (seed, n) not in self.part_key}

    def switches_by_episode(self, store, seed=None):
        """{e: switch_parts} of the switch episodes of a demo store written from the same draw."""
        seed, attempts = self.store_rows(store, seed)
        return {row["e"]: self.switch_parts(seed, n) for n, row in enumerate(attempts) if row["e"] >= 0 and self.switch_parts(seed, n)}

    def store_rows(self, store, seed):
        seed = json.loads((self.directory / "config.json").read_text())["seed"] if seed is None else seed
        return seed, read_jsonl(Path(store) / "attempts.jsonl")


def choice_of(row):
    return {"rung": row["rung"], "instruction": row["instruction"], "thought": row["thought"], "switches": row["switches"], "done": row["done"]}


def is_test(directory):
    directory = Path(directory)
    return (directory / MARKER).exists() or directory.name == TEST_NAME


# ---------------------------------------------------------------------------------------------
# Commands


def options_of(args):
    cube = args.cube or args.free_cube
    return {
        "seed": args.seed,
        "chunk": args.chunk,
        "robots": args.robots or (list(S.ROBOTS) if cube else None),
        "counts": args.counts,
        "fit": args.fit,
        "rule": args.rule or ("free_cube" if args.free_cube else None),
        "apart": args.apart,
        "coverage": getattr(args, "coverage", 0.0),
        "objects": getattr(args, "objects", 0.0),
        "build": getattr(args, "build", 0.0),
        "only": getattr(args, "only", None),
    }


def dry_copy(directory, dry_run, files=("items.jsonl", "config.json", "menu.json", "generated.jsonl")):
    """A scratch copy of the inputs for a dry run (nothing under --root is written)."""
    if not dry_run:
        return directory
    scratch = Path(tempfile.mkdtemp(prefix="haiku-dry-"))
    for name in files:
        if (directory / name).exists():
            shutil.copy(directory / name, scratch / name)
    print(f"(dry run) working in {scratch}", flush=True)
    return scratch


def provider_for(args, items):
    if args.dry_run:
        return Echo(SpecHints(FakeHaiku(), items))
    return haiku


def build_testset(directory, args, provider_of):
    """Export, generate (the test prompt) and verify the held-out test set; freeze it."""
    if (directory / FROZEN).exists() and not args.dry_run:
        raise FileExistsError(f"{directory} is frozen; the test set is built once")
    options = {**options_of(args), "seed": args.seed if args.seed != 0 else TEST_SEED}
    if args.dry_run:
        scratch = Path(tempfile.mkdtemp(prefix="haiku-dry-")) / TEST_NAME
        print(f"(dry run) working in {scratch}", flush=True)
        directory = scratch
    if not (directory / "items.jsonl").exists():
        export(directory, options, args.dry_items if args.dry_run else args.items, keep=lambda item: bool(test_entry(item)["use"]))
    (directory / MARKER).write_text("held-out test set: never train on it (haiku.Labels refuses it)\n")
    items = read_jsonl(directory / "items.jsonl")
    provider = provider_of(items)
    generate(directory, provider, "test", args.batch, args.workers, verbose=args.dry_run, log_every=args.log_every)
    verify(directory, provider, args.verify_batch, args.workers, verbose=args.dry_run, log_every=args.log_every)
    if not args.dry_run:
        (directory / FROZEN).write_text(now() + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("export", "generate", "verify", "testset", "pilot", "report"))
    parser.add_argument("--name", default=None, help="dataset directory under --root (pilot: pilot, testset: test)")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--dry-run", action="store_true", help="fake provider; print the exact prompts and parsed fake outputs; write nothing under --root")
    parser.add_argument("--dry-items", type=int, default=4, help="items a dry run uses")
    parser.add_argument("--items", type=int, default=None, help="how many items to export (export 1000, pilot 50, testset 300)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=24, help="demos --chunk the store was or will be written with")
    parser.add_argument("--robots", type=int, nargs=2, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--counts", choices=S.COUNTS, default="uniform")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--rule", default=None)
    parser.add_argument("--apart", action="store_true")
    parser.add_argument("--coverage", type=float, default=0.0, help="as demos --coverage: coverage tasks (suite.sample_coverage) per scene, after the suite's own")
    parser.add_argument("--objects", type=float, default=0.0, help="as demos --objects: object tasks (objects.sample_objects) per scene, after the coverage tasks")
    parser.add_argument("--build", type=float, default=0.0, help="as demos --build: build tasks (voxel/build.py) per scene, after the coverage tasks")
    parser.add_argument("--only", choices=("build",), default=None, help="as demos --only: draw only the build tasks (item indices count build tasks only; join only to a store written with --only build)")
    parser.add_argument("--cube", action="store_true", help="as demos --cube: robots in suite.ROBOTS")
    parser.add_argument("--free-cube", action="store_true", help="as demos --free-cube: --cube and --rule free_cube")
    parser.add_argument("--store", default=None, help="export: a demo store to join to; its options are read from small.pt and only its kept episodes exported")
    parser.add_argument("--from-items", default=None, help="export: the items of an earlier export (its items.jsonl or a file of its items, next to its config.json) into a new --name, with start voxels and a fresh menu")
    parser.add_argument("--batch", type=int, default=None, help=f"items per generate call (default {GENERATE_BATCH}; testset {TEST_BATCH})")
    parser.add_argument("--verify-batch", type=int, default=VERIFY_BATCH)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--refusals", type=float, default=REFUSAL_SHARE, help="share of items that also get a refuse/clarify instruction")
    parser.add_argument("--limit", type=int, default=None, help="generate: only the first n items")
    parser.add_argument("--log-every", type=float, default=LOG_EVERY, help="seconds between throughput lines")
    args = parser.parse_args(argv)
    args.items = args.items or {"pilot": 50, "testset": 300}.get(args.command, 1000)
    root = Path(args.root)
    name = args.name or {"pilot": "pilot", "testset": TEST_NAME}.get(args.command)
    if name is None:
        parser.error("--name is needed")
    directory = root / name
    if args.command != "testset" and is_test(directory) and args.command not in ("report",):
        parser.error(f"{directory} is the test set; build it with the testset command")

    if args.command == "export" and args.from_items:
        export_from(directory, args.from_items, args.dry_run)
    elif args.command == "export":
        export(directory, options_of(args), args.dry_items if args.dry_run else args.items, args.store, args.dry_run)
    elif args.command == "generate":
        work = dry_copy(directory, args.dry_run)
        items = read_jsonl(work / "items.jsonl")
        generate(work, provider_for(args, items), "train", args.batch, args.workers, args.dry_items if args.dry_run else args.limit, refusal_share=args.refusals, log_every=args.log_every, verbose=args.dry_run)
    elif args.command == "verify":
        work = dry_copy(directory, args.dry_run)
        if args.dry_run:
            generated_rows = read_jsonl(work / "generated.jsonl")[: args.dry_items]
            (work / "generated.jsonl").write_text("".join(json.dumps(row) + "\n" for row in generated_rows))
        verify(work, provider_for(args, read_jsonl(work / "items.jsonl")), args.verify_batch, args.workers, log_every=args.log_every, verbose=args.dry_run)
    elif args.command == "testset":
        build_testset(directory, args, lambda items: provider_for(args, items))
    elif args.command == "pilot":
        if args.dry_run:
            directory = Path(tempfile.mkdtemp(prefix="haiku-dry-")) / name
            print(f"(dry run) working in {directory}", flush=True)
        if not (directory / "items.jsonl").exists():
            export(directory, options_of(args), args.dry_items if args.dry_run else args.items, args.store)
        items = read_jsonl(directory / "items.jsonl")
        provider = provider_for(args, items)
        generate(directory, provider, "train", args.batch, args.workers, refusal_share=args.refusals, log_every=args.log_every, verbose=args.dry_run)
        verify(directory, provider, args.verify_batch, args.workers, log_every=args.log_every, verbose=args.dry_run)
        print(review(directory, directory / "review.txt"))
        print(f"\nreview written to {directory / 'review.txt'}")
    else:
        report(directory)


if __name__ == "__main__":
    sys.exit(main())
