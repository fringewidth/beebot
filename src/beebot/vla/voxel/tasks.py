"""Instructions for the voxel world, checked by its window facts (voxel/facts.py): the 3D task
suite. One instruction family per fact kind (move, region, split, merge, push, eat logic,
surround, tower, bridge) plus pairs of them ("X, and Y"). The same candidates serve hindsight
labelling of play windows (relabel) and, sampled on fresh scenes, RL and evaluation.

An instruction is (family, text, spec). spec["need"] is a list of clauses, each a predicate on the
episode's facts; the score is the share of clauses met, and the episode is "unasked" when it ate a
colour in spec["forbid"]. Eat logic uses the 2D forms (single, except, and, or, all, none, absent)
and their wordings; the order of eating is not checked here, so there is no ordered form.

Held out of training: the pairs (tower, push) and (bridge, eat), and the logic forms neither and
all_but_two, as in 2D.
"""

import itertools

import numpy as np

from beebot.vla import tasks as T2
from beebot.vla.voxel.facts import TOWER_BUCKETS, region_name
from beebot.vla.voxel.world import PALETTE_NAMES

TOWER_RANK = {label: i + 1 for i, (_, label) in enumerate(TOWER_BUCKETS)}
COUNT_WORDS = {2: "two", 3: "three", 4: "four"}
AXES = (("up", "down"), ("forward", "back"), ("left", "right"))
MOVES = [w for pair in AXES for w in pair] + [f"{a} {b}" for a, b in itertools.product(("up", "down"), ("forward", "back", "left", "right"))] + [f"{a} {b}" for a, b in itertools.product(("forward", "back"), ("left", "right"))]
REGIONS = sorted({region_name(z, y, x, 15) for z in (0, 7, 14) for y in (0, 7, 14) for x in (0, 7, 14)})
HELD_PAIRS = (("push", "tower"), ("bridge", "eat"))
WORDINGS = {
    "move": ["move {d}", "go {d}", "head {d}", "shift the swarm {d}", "everyone move {d}"],
    "region": ["go to the {r} of the space", "gather in the {r} area", "head for the {r} part of the grid", "end up {r}"],
    "split": ["split into {n} groups", "break up into {n} groups", "form {n} separate groups", "divide into {n} teams"],
    "merge": ["gather into one group", "come together", "merge into a single group", "regroup into one blob"],
    "push": ["push the {c} cheese {d}", "shove the {c} cheese {d}", "move the {c} cheese {d} without eating it", "slide the {c} cheese {d}"],
    "push_any": ["push the {c} cheese", "move the {c} cheese without eating it", "give the {c} cheese a shove"],
    "around": ["surround the {c} cheese", "gather around the {c} cheese", "encircle the {c} cheese"],
    "tower": ["build a {b} tower", "stack up into a {b} tower", "make a {b} column of robots", "pile up into a {b} tower"],
    "tower_any": ["build a tower", "stack yourselves up", "make a tower"],
    "bridge": ["bridge the gap", "build a bridge across the trench", "span the gap with your bodies", "make a bridge over the gap"],
    "pair": ["{a}, and {b}", "{a}; also {b}", "{a} and {b}", "do two things: {a}, and {b}"],
}


def say(rng, key, **kw):
    return str(rng.choice(WORDINGS[key])).format(**kw)


def clause_met(clause, atoms, eaten):
    kind = clause[0]
    if kind == "atom":
        return clause[1] in atoms
    if kind == "tower_at_least":
        return any(a[0] == "tower" and TOWER_RANK[a[1]] >= clause[1] for a in atoms)
    if kind == "pushed_any":
        return any(a[0] == "pushed" and a[1] == clause[1] for a in atoms)
    if kind == "ate":
        return clause[1] in eaten
    if kind == "ate_any":
        return bool(eaten & clause[1])
    if kind == "kept":  # alive at the end: needed by push and surround (eating it fails them)
        return clause[1] not in eaten
    raise ValueError(kind)


def check(spec, facts):
    """(score in [0, 1], unasked) of an episode's facts (window_facts from its start to its end)."""
    atoms = facts.atoms
    eaten = {a[1] for a in atoms if a[0] == "ate"}
    unasked = bool(eaten & spec["forbid"])
    gates = [c for c in spec["need"] if c[0] == "kept"]  # conditions, not credit: doing nothing must not score
    need = [c for c in spec["need"] if c[0] != "kept"]
    if not need:  # eat nothing: met when nothing was eaten
        return float(not eaten), unasked
    if not all(clause_met(c, atoms, eaten) for c in gates):
        return 0.0, unasked
    return float(np.mean([clause_met(c, atoms, eaten) for c in need])), unasked


def candidates(scene, rng):
    """Every training instruction the scene allows, by family: {family: [(text, spec)]}.
    scene: {"colours": names of the cheeses alive at the start, "gravity": bool, optionally "gap":
    bool, whether the terrain has a trench to bridge}."""
    present = list(scene["colours"])
    everything = set(present)
    out = {f: [] for f in ("move", "region", "split", "merge", "push", "eat", "around", "tower", "bridge")}

    def add(family, text, need, forbid, form=""):
        out[family].append((text, {"need": need, "forbid": set(forbid), "form": form, "family": family}))

    for d in MOVES:
        add("move", say(rng, "move", d=d), [("atom", ("move", d))], everything)
    for r in REGIONS:
        add("region", say(rng, "region", r=r), [("atom", ("end region", r))], everything)
    for n in (2, 3, 4):
        add("split", say(rng, "split", n=COUNT_WORDS[n]), [("atom", ("groups", n))], everything)
    add("merge", say(rng, "merge"), [("atom", ("groups", 1))], everything)
    for c in present:
        horizontal = [d for d in MOVES if "up" not in d and "down" not in d] if scene["gravity"] else MOVES
        for d in horizontal:
            add("push", say(rng, "push", c=c, d=d), [("atom", ("pushed", c, d)), ("kept", c)], everything)
        add("push", say(rng, "push_any", c=c), [("pushed_any", c), ("kept", c)], everything)
        add("around", say(rng, "around", c=c), [("atom", ("around", c)), ("kept", c)], everything)
    for a in present:
        add("eat", T2.say(rng, "logic.single", "", a=a), [("ate", a)], everything - {a}, "single")
        if len(present) >= 2:
            add("eat", T2.say(rng, "logic.except", "", a=a), [("ate", c) for c in everything - {a}], {a}, "except")
    for a, b in itertools.combinations(present, 2):
        add("eat", T2.say(rng, "logic.and", "", a=a, b=b), [("ate", a), ("ate", b)], everything - {a, b}, "and")
        add("eat", T2.say(rng, "logic.or", "", a=a, b=b), [("ate_any", {a, b})], everything - {a, b}, "or")
    if present:
        add("eat", T2.say(rng, "logic.all", ""), [("ate", c) for c in present], set(), "all")
        add("eat", T2.say(rng, "logic.none", ""), [], everything, "none")
        missing = [c for c in PALETTE_NAMES if c not in everything]
        if missing:
            add("eat", T2.say(rng, "logic.single", "", a=str(rng.choice(missing))), [], everything, "absent")
    if scene["gravity"]:
        for label, rank in TOWER_RANK.items():
            add("tower", say(rng, "tower", b=label), [("tower_at_least", rank)], everything)
        add("tower", say(rng, "tower_any"), [("tower_at_least", 1)], everything)
        if scene.get("gap", True):  # eval scenes say whether there is a trench; play windows don't (their labels are facts already)
            add("bridge", say(rng, "bridge"), [("atom", ("bridge",))], everything)
    return {f: v for f, v in out.items() if v}


def pair_options(options, rng, held=False):
    """Satisfied pairs from two different families' satisfied instructions: [(text, spec)]. held:
    only the held-out family pairs (HELD_PAIRS), else only the others."""
    out = []
    families = sorted(options)
    for f1, f2 in itertools.combinations(families, 2):
        if (tuple(sorted((f1, f2))) in HELD_PAIRS) != held:
            continue
        (t1, s1), (t2, s2) = options[f1][int(rng.integers(len(options[f1])))], options[f2][int(rng.integers(len(options[f2])))]
        if s1["form"] in ("none", "absent") or s2["form"] in ("none", "absent"):
            continue
        out.append((say(rng, "pair", a=t1, b=t2), {"need": s1["need"] + s2["need"], "forbid": s1["forbid"] & s2["forbid"], "form": "", "family": "pair", "parts": (f1, f2)}))
    return out


def satisfied_options(scene, facts, rng, threshold=1.0):
    """{family: [(text, spec)]} of the candidates the episode's facts meet in full (nothing
    forbidden eaten), with "pair" for satisfied pairs of two families."""
    options = {}
    for family, items in candidates(scene, rng).items():
        ok = [(text, spec) for text, spec in items if (lambda r: r[0] >= threshold and not r[1])(check(spec, facts))]
        if ok:
            options[family] = ok
    pairs = pair_options(options, rng)
    if pairs:
        options["pair"] = pairs
    return options
