"""Hindsight task instructions for play windows: every instruction from the task suite's training
families (logic, place, parallel, herd) that the window's own outcome satisfies, as scored by
tasks.score. The labels are true by construction, so supervised training on them teaches the
suite's language (except, or, ordered, nudge, herd into a box, split up) on behaviour the play data
already contains, before RL.

    uv run python -m beebot.vla.relabel --out data/vla/kaggle_play/captions_tasks_g32.jsonl

Writes caption records in the captions jsonl format (id "program:scene:w0-w1", captions), with
direction and region words inside <d> tags so the rotation augmentation can rewrite them; colour
words are tagged when the set is built. Only training wordings and training colours are used, and
no held-out form (neither, all but two, held-out pairs, two pushes at once). Per window, one record
for each family with a satisfied instruction, holding one of its satisfied instructions at random
(logic: a satisfied form first, uniformly, then an instruction of it). Families appear as often as
the play data satisfies them; the eat-nothing logic forms are thinned to their share of the suite
(thin_idle), since otherwise idle windows make them most of the logic labels.
"""

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from beebot.vla import tasks as T
from beebot.vla.goal_eval import FACT_EVERY
from beebot.vla.play_data import load_play_cache
from beebot.vla.world import COLOUR_NAMES, cheese_side

THRESHOLD = 0.9  # an instruction counts as satisfied at or above this score, with nothing unasked eaten
WINDOWS = (32, 48, 64)  # window lengths drawn per episode; RL episodes are 48 to 64 steps
PER_EPISODE = 3


def tagged(phrase):
    return f"<d>{phrase}</d>"


def window_scene(store, e, w0):
    """The task-suite scene dict of episode e at step w0 (alive cheeses only) and their slots."""
    pos = store["cheese_pos"][e, w0].long().numpy()
    alive = store["cheese_alive"][e, w0].numpy()
    colour = store["cheese_color"][e].numpy()
    slots = [k for k in range(len(alive)) if alive[k] and colour[k] >= 0]
    boxes = [(r, int(c)) for r, c in zip(store["box_rect"][e].tolist(), store["box_color"][e].tolist()) if c >= 0 and r[2] > 0]
    scene = {
        "mass": store["mass"][e, w0].float().numpy(),
        "walls": store["walls"][e].numpy(),
        "cheeses": [tuple(pos[k]) for k in slots],
        "cheese_colours": [int(colour[k]) for k in slots],
        "boxes": [tuple(r) for r, _ in boxes],
        "box_colours": [c for _, c in boxes],
        "runners": [],
    }
    return scene, slots


def window_frames(store, e, w0, w1, slots):
    """Snapshots every FACT_EVERY steps from w0 to w1 over the given cheese slots, and the colours
    eaten in order."""
    times = list(range(w0, w1, FACT_EVERY)) + [w1]
    frames = [{"mass": store["mass"][e, t].float().numpy(), "cheese_pos": store["cheese_pos"][e, t, slots].long().numpy(), "cheese_alive": store["cheese_alive"][e, t, slots].numpy()} for t in times]
    alive = store["cheese_alive"][e, w0 : w1 + 1][:, slots].numpy()
    colour = store["cheese_color"][e].numpy()
    eaten = []
    for t in range(1, len(alive)):
        for j in np.nonzero(alive[t - 1] & ~alive[t])[0]:
            eaten.append(COLOUR_NAMES[int(colour[slots[j]])])
    return frames, eaten


def candidates(scene, rng):
    """Every training-split task the scene allows, by family: [(family, text, spec)]. Texts use the
    training wordings; direction and region fillers are <d>-tagged."""
    out = {f: [] for f in ("logic", "place", "parallel", "herd")}
    present = [COLOUR_NAMES[c] for c in scene["cheese_colours"]]
    everything = set(present)
    box_names = [COLOUR_NAMES[c] for c in scene["box_colours"]]
    side = cheese_side(T.SIZE)

    def logic(key, form, spec, **kw):
        out["logic"].append(("logic", T.say(rng, key, "", **kw), {"eat": set(), "any_of": set(), "order": [], "forbid": set(), **spec, "form": form}))

    for a in present:
        logic("logic.single", "single", {"eat": {a}, "forbid": everything - {a}}, a=a)
        if len(present) >= 2:
            logic("logic.except", "except", {"eat": everything - {a}, "forbid": {a}}, a=a)
    for a, b in itertools.permutations(present, 2):
        logic("logic.ordered", "ordered", {"eat": {a, b}, "order": [a, b], "forbid": everything - {a, b}}, a=a, b=b)
    for a, b in itertools.combinations(present, 2):
        logic("logic.and", "and", {"eat": {a, b}, "forbid": everything - {a, b}}, a=a, b=b)
        logic("logic.or", "or", {"any_of": {a, b}, "forbid": everything - {a, b}}, a=a, b=b)
    if present:
        logic("logic.all", "all", {"eat": everything})
        logic("logic.none", "none", {"forbid": everything})
        missing = [c for c in T.TRAIN_COLOURS if c not in everything]
        logic("logic.single", "absent", {"forbid": everything}, a=str(rng.choice(missing)))  # absent: asks for nothing
        centres = {COLOUR_NAMES[c]: (y + side / 2, x + side / 2) for (y, x), c in zip(scene["cheeses"], scene["cheese_colours"])}
        rows, cols = np.nonzero(scene["mass"] >= 0.5)
        if len(rows):
            far = max(centres, key=lambda c: np.hypot(centres[c][0] - rows.mean(), centres[c][1] - cols.mean()))
            logic("logic.furthest", "furthest", {"eat": {far}, "forbid": everything - {far}})
        for (by, bx, bh, bw), name in zip(scene["boxes"], box_names):
            near = min(centres, key=lambda c: np.hypot(centres[c][0] - by - bh / 2, centres[c][1] - bx - bw / 2))
            logic("logic.nearest_box", "nearest_box", {"eat": {near}, "forbid": everything - {near}}, b=name)

    for rect, name in zip(scene["boxes"], box_names):
        out["place"].append(("place", T.say(rng, "place.box", "", b=name), {"box": rect, "forbid": everything}))
    for r in T.REGIONS:
        out["place"].append(("place", T.say(rng, "place.region", "", r=tagged(r)), {"region": r, "forbid": everything}))
    for d in T.DIRECTIONS:
        out["place"].append(("place", T.say(rng, "place.move", "", d=tagged(d)), {"direction": d, "forbid": everything}))

    jobs = [("region", r) for r in T.REGIONS] + [("box", b) for b in box_names] + [("eat", c) for c in present] + [("surround", c) for c in present]
    rects = dict(zip(box_names, scene["boxes"]))
    for j1, j2 in itertools.combinations(jobs, 2):
        if tuple(sorted((j1[0], j2[0]))) in T.HELD_OUT["pairs"] or (j1[0] in ("eat", "surround") and j2[0] in ("eat", "surround") and j1[1] == j2[1]):
            continue
        words = [T.SUBTASK_WORDS[k].format(x=tagged(x) if k == "region" else x) for k, x in (j1, j2)]
        if rng.random() < 0.5:
            words.reverse()
        eat = {x for k, x in (j1, j2) if k == "eat"}
        out["parallel"].append(("parallel", T.say(rng, "parallel.split", "", a=words[0], b=words[1]), {"jobs": [j1, j2], "rects": rects, "forbid": everything - eat}))

    for (pos, c) in zip(scene["cheeses"], present):
        for d in ("up", "down", "left", "right"):
            out["herd"].append(("herd", T.say(rng, "herd.nudge", "", c=c, d=tagged(d)), {"nudge": (c, d), "forbid": everything}))
        for rect, name in zip(scene["boxes"], box_names):
            if not T.cheese_in_box(pos, rect, side):
                out["herd"].append(("herd", T.say(rng, "herd.into", "", x=c, b=name), {"into": ([c], rect), "forbid": everything}))
    return out


def satisfied(family, spec, scene, frames, eaten):
    task = T.Task(family, "", spec, "")
    s, unasked = T.score(task, scene, frames, eaten)
    if unasked:
        return False
    if family == "logic" and not spec["eat"] and not spec["any_of"]:  # none / absent: nothing eaten is the success
        return not eaten
    return s >= THRESHOLD


def label_episode(store, e, rng):
    records = []
    t_max = store["mass"].shape[1] - 1
    program_id, _, scene_index = store["source"][e]
    for _ in range(PER_EPISODE):
        length = int(rng.choice(WINDOWS))
        w0 = int(rng.integers(0, t_max - length + 1))
        w1 = w0 + length
        scene, slots = window_scene(store, e, w0)
        if not (scene["mass"] >= 0.5).any():
            continue
        frames, eaten = window_frames(store, e, w0, w1, slots)
        end_groups = GROUPS(frames[-1]["mass"])
        T.groups = lambda mass, min_share=0.15: end_groups  # score() only asks for the end frame's groups
        options = {}
        for family, items in candidates(scene, rng).items():
            ok = [(text, spec) for _, text, spec in items if satisfied(family, spec, scene, frames, eaten)]
            if ok:
                options[family] = ok
        if not options:
            continue
        for family, ok in sorted(options.items()):
            form = ""
            if family == "logic":  # a form first, uniformly among the satisfied ones, as the suite draws forms
                form = str(rng.choice(sorted({spec["form"] for _, spec in ok})))
                ok = [(text, spec) for text, spec in ok if spec["form"] == form]
            text, _ = ok[int(rng.integers(len(ok)))]
            records.append({"id": f"{program_id}:{scene_index}:{w0}-{w1}", "provider": "relabel", "types": [family], "form": form, "captions": [text]})
    return records


IDLE_FORMS = ("none", "absent")  # logic forms satisfied by eating nothing
IDLE_SHARE = 0.2  # their share of the suite's logic forms (2 of 10)


def thin_idle(records, rng):
    """Most play windows eat nothing, and the only logic labels true of those say "eat nothing":
    left alone they are most of the logic labels and teach the policy to ignore eating requests.
    Keep a random subset of them so they are IDLE_SHARE of the logic labels, as in the suite."""
    idle = [i for i, r in enumerate(records) if r["form"] in IDLE_FORMS]
    active = sum(1 for r in records if r["form"] and r["form"] not in IDLE_FORMS)
    keep = int(active * IDLE_SHARE / (1 - IDLE_SHARE))
    drop = set(rng.choice(idle, size=max(len(idle) - keep, 0), replace=False).tolist()) if len(idle) > keep else set()
    return [r for i, r in enumerate(records) if i not in drop]


STORE = None
GROUPS = T.groups


def _init(play_cache):
    global STORE
    STORE = load_play_cache(play_cache)


def _work(args):
    start, stop, seed = args
    rng = np.random.default_rng(seed)
    return [r for e in range(start, stop) for r in label_episode(STORE, e, rng)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--play_cache", default="data/vla/play_cache_g32")
    parser.add_argument("--out", default="data/vla/kaggle_play/captions_tasks_g32.jsonl")
    parser.add_argument("--episodes", type=int, default=0, help="0: all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    store = load_play_cache(args.play_cache)
    n = args.episodes or len(store["source"])
    del store
    chunks = [(s, min(s + 500, n), args.seed * 100003 + s) for s in range(0, n, 500)]
    records = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.play_cache,)) as pool:
        for part in pool.map(_work, chunks):
            records += part
            print(f"{len(records)} labels", flush=True)
    records = thin_idle(records, np.random.default_rng(args.seed))
    counts, forms = {}, {}
    with open(args.out, "w") as f:
        for r in records:
            counts[r["types"][0]] = counts.get(r["types"][0], 0) + 1
            if r["form"]:
                forms[r["form"]] = forms.get(r["form"], 0) + 1
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} labels to {args.out}: {counts}; logic forms {forms}")

if __name__ == "__main__":
    main()
