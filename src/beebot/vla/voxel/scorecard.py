"""The 3D scorecard of the think-then-act voxel policy: closed-loop episodes (voxel/rollout.py) over
every family and the held-out splits, with controls, thought checks and an oracle-thought arm.

    PYTHONPATH=src python -m beebot.vla.voxel.scorecard --checkpoint runs/X/policy_step800.pt --device mps --out runs/X/card_step800
    PYTHONPATH=src python -m beebot.vla.voxel.scorecard --model Qwen/Qwen3.5-0.8B --device mps --out runs/card_base
    PYTHONPATH=src python -m beebot.vla.voxel.scorecard --arms expert --out runs/card_expert     # no model
    PYTHONPATH=src python -m beebot.vla.voxel.scorecard compare runs/A/episodes.jsonl runs/B/episodes.jsonl

Splits (--splits; each scene drawn as the training stores draw theirs: robots 6 to 115 log-uniform,
tasks the swarm can do, gravity on, object fields; rollout.py has the world's conditions):
  test      in-distribution: --per-family (48) episodes of each of the 14 families (FAMILIES), with
            the training templates (sample_tasks / sample_coverage / sample_objects / sample_build,
            split "test")
  wording   the held-out templates (split "wording"), --per-held (16) per family
  held      held-out compositions, --per-held each (HELD: suite colour, letters, shapes, logic,
            pairs, construction (stairs), selector_job; objects ref_job, condition, count; build
            objects (held-out names)); split names "held:<key>"
  testset   the frozen Haiku test set (data/vla/haiku/test): every verified instruction (held-out
            words and categories) on its own episode, the scene redrawn with demos.draws from the
            test set's config and checked against the item's record (a mismatch is skipped and
            counted); a switch instruction is cut into its two parts at "; then, on a new
            instruction"
Arms (--arms; ARM_SPLITS says which splits each runs on):
  policy    the episode's instruction, the policy thinks and acts
  shuffled  the same scene and seed with another episode's instruction of the same split and family
            (the next one with a different text); test split only by default
  plain     the canonical plain sentence of the task (canonical_record "plain") instead of the
            instruction; the held-out splits (gate b's denominator)
  oracle    the episode's instruction with the canonical first thought fed on the first step of each
            instruction (rollout.canonical_thought); gate b's numerator on the held-out splits
  expert    the scripted expert (rollout.ExpertActor) on the same scenes: the ceiling; held-out
            compositions have no plan, so there it is the robots staying still
  still     no robot ever moves: the floor. Some tasks pass so (a switch to "stay", "eat no
            cheese"), so read every arm, and gate b's ratio, against it
Metrics per (split, family, arm), each with its standard error (std / sqrt(n)):
  score     the suite's checker score at the episode's end (suite.score; a switch task on its last
            instruction), 0 when the swarm ate a cheese the instruction forbids (unasked)
  pass      rollout.passed: the family's pass mark (suite.SUCCESS, else 0.75; build.passed) and
            nothing forbidden eaten
  unasked, spoke (the policy answered in words and stopped), unfinished (steps out of tokens)
  thought_parse     the first thought of each instruction exists and has the approved shape:
                    "you asked: ..." and a non-empty "first: ..." clause (for a task without
                    sub-goals, whose labels end with what the swarm does instead, any last sentence)
  thought_first     it names the canonical first sub-goal (canonical_record "subgoals"[0], "stay
                    where you are" when there is none): every key fact of the sub-goal (colours,
                    floor areas, numbers 2 and up, shapes, letters, a build object's name; facts())
                    is in the thought (the instruction it quotes left out), and the "first:" clause
                    names no colour the sub-goal does not; a sub-goal without key facts is matched by
                    its kind of move (KINDS: stay, gather, spread, ...), failing that by half its
                    content words
  thought_resolves  the thought holds every key fact of what the swarm will do (all sub-goals; a
                    build task's plain sentence); a refuse, stay or eat-nothing task: it says the
                    swarm stays or eats nothing
                    On 25,378 verified Haiku training labels: parse 1.00, first 0.97, resolves 0.97.
  thought metrics are the mean over the episode's instructions; the oracle arm's are not measured
Paired, per (split, family), on the same episodes:
  gap           policy - shuffled (score and pass): does the policy act on the instruction
  oracle_ratio  pass(oracle) / pass(plain), delta-method standard error: gate b (>= 0.9)
  oracle_gain   oracle - policy (score)
Macro means over families (each family weighted equally; se = sqrt(sum se^2) / families) per
split and arm. Gates (summary "gates"): a) on the test split, thought_parse and thought_first of
the policy arm (>= 0.9 each), the shuffled gap, and the pass share of episodes whose first thought
matched against those whose did not (acting on the thought); b) per held-out split, the macro
oracle_ratio (>= 0.9).

Seeds (disjoint from training): scenes of split s and family f come from
numpy.random.default_rng([SEED, crc32(s), crc32(f)]) with SEED = 700_001; the training stores
draw from default_rng(k), (k, 7), (k, 11), (k, 13) for store seeds k = 0..7, the Haiku test set
from 900_001 and the verifier's menu from 4_242. An episode's seed (its world noise, failures,
action sampling, views) is sha256("700001/<split>/<family>/<i>"), the same in every arm, so arms
and checkpoints are compared on the same episodes (compare).

Live numbers for ops (JSON lines appended while it runs, each with "run" (the --out directory's
name) and "time"):
  --rollout-log (logs/think_act_loop.jsonl): after every (arm, split, family) group
            {"arm", "rollout_success" (pass share), "n", "split", "family", "repeat"}, and when an
            arm finishes a repeat, per split, the macro over families with "family": "all" and its
            "se"; arm names model (policy), oracle_thought (oracle), expert, plain, shuffled, still
  --log (logs/scorecard3d.jsonl): when an arm finishes a repeat, per split {"arm", "split",
            "scorecard3d_mean" (macro score), "scorecard3d_se"}; at the end per (arm, split)
            {"scorecard3d_noise", "noise_from", "repeats", "scorecard3d_mean"}: the standard
            deviation of the scorecard mean across --repeats cards (seeds SEED + r: fresh scenes
            and episode seeds; the test set's scenes are fixed, only its episode seeds change), or
            with one card its macro standard error, which estimates the same spread. A re-run with
            the same seed repeats the card exactly, so the noise is over scenes and seeds.
Output (--out DIR): episodes.jsonl (one JSON line per episode and arm, as it finishes, then one
line {"summary": ...}), card.md (the tables). Ranking two checkpoints: `compare` pairs their
episodes by (split, family, i, arm) and gives each difference with its paired standard error.
"""

import argparse
import hashlib
import json
import re
import sys
import time
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel import rollout as R
from beebot.vla.voxel import suite as S
from beebot.vla.voxel.world import COLOUR_NAMES

SEED = 700_001
FAMILIES = ("formation", "construction", "logic", "parallel", "herd", "runner", "motion", "subset", "switch", "things", "blocks", "zones", "reasoning", "build")
TESTSET = Path("data/vla/haiku/test")
ARMS = ("policy", "shuffled", "plain", "oracle", "expert", "still")
SPLIT_GROUPS = ("test", "wording", "held", "testset")
ARM_SPLITS = {"policy": SPLIT_GROUPS, "shuffled": ("test",), "plain": ("wording", "held", "testset"), "oracle": SPLIT_GROUPS, "expert": SPLIT_GROUPS, "still": SPLIT_GROUPS}
THINKING_ARMS = ("policy", "plain")  # arms whose thoughts are checked
GATE = 0.9


def held_splits():
    """[(split key, family)] of the held-out compositions."""
    from beebot.vla.voxel import objects as O

    suite = [(key, S.HELD_FAMILY[key]) for key, items in S.HELD_OUT.items() if key != "wording" and items]
    return suite + [(key, O.HELD_FAMILY[key]) for key in O.HELD_OUT] + [("objects", "build")]


# ---------------------------------------------------------------------------------------------
# Thought checks


NUMBER_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()) if i >= 2}
REGIONS = sorted(S.FLOOR_REGIONS, key=len, reverse=True)
REGION_ALIASES = {"centre": "center", "middle": "center", "rear": "back"}
KINDS = (  # (kind of move, words that say it), tried in order on a sub-goal without key facts
    ("none", ("do not eat", "don't eat", "not eat", "no cheese", "eat nothing", "eats nothing", "nothing gets eaten", "nothing is eaten", "left alone", "leave every", "leave all", "leave them", "leave the cheese")),
    ("stay", ("stay", "hold", "still", "freeze", "remain", "stop", "wait", "don't move", "do not move", "not move", "nothing to do", "in place", "nobody moves", "no one moves")),
    ("gather", ("gather", "together", "one group", "merge", "bunch", "huddle", "clump", "regroup", "join")),
    ("spread", ("spread", "apart", "scatter", "no two", "space out")),
    ("line", ("line", "row", "single file")),
    ("fill", ("fill", "pack")),
    ("bridge", ("bridge", "span")),
    ("stairs", ("stair", "steps up")),
    ("eat", ("eat", "devour", "gobble", "catch", "munch", "consume")),
)
STOPWORDS = set("a an the of to on in into at by and or with without it its them you your up out all every each one is are be that this so as from for then first".split())
WORD = re.compile(r"[a-z0-9][a-z0-9'\-]*")


def words(text):
    return WORD.findall(text.lower())


def record_keys(record):
    """Words a record's sentences may hold that are key facts of this task only: a letters task's
    word, a build task's object name."""
    spec = record.get("spec", {})
    keys = set()
    if spec.get("kind") == "letters":
        keys.add(str(spec.get("word", "")).lower())
    if spec.get("kind") == "build" and spec.get("object"):
        keys.update(w for w in words(str(spec["object"]).replace("_", " ")) if w not in STOPWORDS)
    return keys


COORDINATES = re.compile(r"\b[xyz]\s*=?\s*\d+(?:\s*(?:to|-)\s*\d+)?")  # positions, which think_data cuts from thoughts (a viewer cannot read them)
NOT_COUNTS = re.compile(r"\bno two\b")


def facts(text, keys=()):
    """Key facts of a text: {(type, value)} for colours, floor areas, numbers from 2 up (digits or
    words; not coordinates, not "no two"), the suite's shape names and the record's own keys
    (record_keys)."""
    low = NOT_COUNTS.sub(" ", COORDINATES.sub(" ", text.lower()))
    low = " " + re.sub(r"[^a-z0-9'\- ]", " ", low) + " "
    for alias, name in REGION_ALIASES.items():
        low = low.replace(f" {alias} ", f" {name} ")
    out = set()
    for region in REGIONS:
        if f" {region} " in low:
            out.add(("region", region))
            low = low.replace(f" {region} ", " ")
    for w in words(low):
        if w in COLOUR_NAMES:
            out.add(("colour", w))
        elif w.isdigit() and int(w) >= 2:
            out.add(("number", int(w)))
        elif w in NUMBER_WORDS:
            out.add(("number", NUMBER_WORDS[w]))
        elif w in S.SHAPES:
            out.add(("shape", w))
        elif w in keys:
            out.add(("key", w))
    return out


def says(text, phrase):
    """Whether text holds phrase at a word start (so "eat" is not found in "seat")."""
    return re.search(r"\b" + re.escape(phrase), text.lower()) is not None


def kind_of(text):
    """The first KINDS entry whose words the text holds, or None."""
    return next((kind for kind, said in KINDS if any(says(text, s) for s in said)), None)


def thought_parts(thought, instruction, record):
    """(well formed, the "first:" clause, the reasoning) of a first thought. Well formed: it starts
    "you asked:" and has a non-empty "first:" clause, or (a task without sub-goals, which the
    labels end with what the swarm does instead) any last sentence. The clause is the text after
    the last "first:", else the last sentence; the reasoning is the thought without the quoted
    instruction."""
    low = thought.lower().strip()
    quoted = instruction.lower().strip().rstrip(".!?")
    reasoning = low.replace(quoted, " ", 1) if quoted else low
    cut = low.rfind("first:")
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", low) if s.strip()]
    clause = low[cut + len("first:") :].strip().rstrip(".") if cut >= 0 else (sentences[-1].rstrip(".") if sentences else "")
    well = low.startswith(R.FIRST_SHAPE) and bool(clause) and (cut >= 0 or not record.get("subgoals"))
    return well, clause, reasoning


def kinds_in(text):
    return {kind for kind, said in KINDS if any(says(text, s) for s in said)}


def matches_subgoal(clause, reasoning, record):
    """Whether a first thought names the record's first sub-goal (canonical "subgoals"[0], or
    STAY_LINE): every key fact of the sub-goal is in the reasoning (the labels often point back
    from the clause: "first: push it to the back"), the clause names no colour the sub-goal does
    not (the task's when the sub-goal names none); a sub-goal without key facts needs its kind of
    move in the clause or the reasoning, failing that half its content words in the reasoning."""
    subgoals = record.get("subgoals") or []
    goal = subgoals[0] if subgoals else R.STAY_LINE
    keys = record_keys(record)
    need = facts(goal, keys)
    colours = {f for f in facts(goal, keys) if f[0] == "colour"} or {f for f in facts(" ".join([record.get("plain", "")] + list(subgoals)), keys) if f[0] == "colour"}
    wrong = {f for f in facts(clause, keys) if f[0] == "colour"} - colours
    if need:
        ok = need <= facts(reasoning, keys)
    elif kind_of(goal) is not None:
        ok = kind_of(goal) in kinds_in(clause) | kinds_in(reasoning)
    else:
        content = {w for w in words(goal) if w not in STOPWORDS}
        ok = len(content & set(words(reasoning))) >= 0.5 * len(content)
    return bool(clause) and ok and not wrong


def check_thought(thought, instruction, record):
    """{"parse", "first", "resolves"} of the first thought of an instruction ("" or None: no thought).
    resolves: every key fact of what the swarm will do (all its sub-goals; a build task's or a task
    without sub-goals: its plain sentence) is in the reasoning; without key facts, its kind of move;
    a refuse, stay or eat-nothing record: the thought says the swarm stays or eats nothing."""
    thought = thought or ""
    well, clause, reasoning = thought_parts(thought, instruction, record)
    spec_kind = record.get("spec", {}).get("kind")
    subgoals = record.get("subgoals") or []
    resolved = " ".join(subgoals) if subgoals and spec_kind != "build" else record.get("plain", "")
    need = facts(resolved, record_keys(record))
    plain_kind = kind_of(record.get("plain", ""))
    spec = record.get("spec", {})
    idle = spec_kind in ("refuse", "stay") or (spec_kind == "eat" and not (spec.get("eat") or spec.get("any_of") or spec.get("order")))
    if idle:  # the thought says the swarm will not move or eat
        resolves = bool(kinds_in(reasoning) & {"stay", "none"})
    elif need:
        resolves = need <= facts(reasoning, record_keys(record))
    else:
        resolves = plain_kind is None or plain_kind in kinds_in(reasoning)
    return {"parse": bool(thought) and well, "first": bool(thought) and matches_subgoal(clause, reasoning, record), "resolves": bool(thought) and bool(resolves)}


def thought_checks(episode, played):
    """Per instruction j: the check of the first thought given on its first step."""
    out = []
    for j, (text, start, record) in enumerate(zip(episode.texts, episode.starts, episode.records)):
        if start >= episode.steps:
            continue
        thought = next((t["text"] for t in played.thoughts if t["part"] == j and t["t"] == start), "")
        out.append({"part": j, "thought": thought, **check_thought(thought, text, record)})
    return out


# ---------------------------------------------------------------------------------------------
# Episodes


def scene_rng(split, family, seed=SEED):
    return np.random.default_rng([seed, zlib.crc32(split.encode()), zlib.crc32(family.encode())])


def episode_seed(split, family, i, seed=SEED):
    return int.from_bytes(hashlib.sha256(f"{seed}/{split}/{family}/{i}".encode()).digest()[:4], "big") >> 1


TRIES = 20  # scenes tried per robot count before another count is drawn (suite_tasks)


def suite_tasks(rng, n, split, family, robots, counts):
    """suite.sample_tasks for one family: with a fixed family, sample_tasks keeps a robot count it
    cannot use forever (a runner needs 23 robots), so here the count is drawn again after TRIES
    scenes that do not fit; robot counts are then those the family fits, as in the training draws."""
    held = split if split in S.HELD_OUT else ""
    worlds, metas, out = [], [], []
    fields = ("mass", "walls", "cheese_pos", "cheese_alive", "cheese_color", "gravity")
    while len(out) < n:
        count = S.draw_count(rng, robots[0], robots[1], counts)
        for _ in range(TRIES):
            needs, make = S.MAKERS[S.HELD_FAMILY.get(held) or family](rng, held)
            colours = needs.pop("colours", S.TRAIN_COLOURS)
            world, meta = S.scenes(rng, 1, colours=colours, robots=count, **needs)
            task = make(world, meta[0], 0)
            if task is not None and S.fits(task, world, 0):
                worlds.append(world)
                metas.append(meta[0])
                out.append(task)
                break
    return S.VoxelWorld(*(torch.cat([getattr(w, f) for w in worlds]) for f in fields)), metas, out


def draw(split, family, n, seed=SEED):
    """(world, metas, tasks) of n scenes of a family on a split ("test", "wording" or a held-out key)."""
    from beebot.vla.voxel import build as B
    from beebot.vla.voxel import objects as O

    rng = scene_rng(split, family, seed)
    common = {"robots": S.ROBOTS, "counts": "log"}
    if family in S.FAMILIES:
        drawn = suite_tasks(rng, n, split, family, **common)
    elif family in S.COVERAGE:
        drawn = S.sample_coverage(rng, n, family=family, split=split, **common)
    elif family in O.OBJECT_FAMILIES:
        drawn = O.sample_objects(rng, n, family=family, split=split, **common)
    else:
        drawn = B.sample_build(rng, n, split=split, **common)
    return drawn


def parts_of(task):
    return task.parts or [task]


def starts_of(task):
    return [int(t) for t in task.spec["steps"]] if task.parts else [0]


def make_episode(world, meta, task, texts, split, family, i, extra=None, seed=SEED):
    key = {"split": split, "family": family, "i": i, **(extra or {})}
    if seed != SEED:
        key["repeat"] = seed - SEED
    return R.Episode(world, meta, task, list(texts), starts_of(task), [S.canonical_record(p) for p in parts_of(task)], R.episode_steps(task), episode_seed(split, family, i, seed), key=key)


def drawn_episodes(split, family, n, name=None, seed=SEED):
    """Episodes of a drawn split, each with its task's own instruction(s)."""
    world, metas, tasks = draw(split, family, n, seed)
    name = name or split
    return [make_episode(world.index(i), metas[i], task, [p.text for p in parts_of(task)], name, family, i, seed=seed) for i, task in enumerate(tasks)]


SWITCH_CUT = re.compile(r"\s*;\s*then,?\s*(?:on|upon)\s+(?:a|the)\s+new\s+instruction\s*[:,]?\s*", re.IGNORECASE)


def testset_episodes(directory=TESTSET, per_family=None, families=FAMILIES, seed=SEED):
    """(episodes, counts): one episode per verified test-set instruction, its scene redrawn from the
    test set's draw options (demos.draws) and checked against the item's record. The scenes are the
    test set's own; seed sets only the episodes' seeds (world noise, failures, sampling)."""
    from beebot.vla.voxel import demos as D

    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    items = {json.loads(line)["index"]: json.loads(line) for line in (directory / "items.jsonl").read_text().splitlines() if line.strip()}
    pairs = sorted((json.loads(line) for line in (directory / "verified.jsonl").read_text().splitlines() if line.strip()), key=lambda r: r["pair_id"])
    pairs = [p for p in pairs if p["family"] in families]
    wanted = {p["index"] for p in pairs}
    found = {}
    index = 0
    robots = tuple(config["robots"]) if config.get("robots") else None
    for _, world, metas, tasks, _ in D.draws(10**9, config["seed"], config.get("chunk", 24), robots, config.get("counts", "uniform"), bool(config.get("fit")) or robots is not None, config.get("rule"), bool(config.get("apart")), float(config.get("coverage", 0.0)), float(config.get("objects", 0.0)), float(config.get("build", 0.0)), config.get("only")):
        for i, task in enumerate(tasks):
            if index in wanted:
                found[index] = (world.index(i), metas[i], task)
            index += 1
        if not wanted or index > max(wanted):
            break
    counts = {"pairs": len(pairs), "mismatched": 0, "uncut": 0}
    by_family = {}
    for pair in pairs:
        world, meta, task = found[pair["index"]]
        if json.loads(json.dumps(S.canonical_record(task))) != items[pair["index"]]["record"]:
            counts["mismatched"] += 1
            continue
        texts = [pair["instruction"]]
        if task.parts:
            texts = [t for t in SWITCH_CUT.split(pair["instruction"]) if t.strip()]
            if len(texts) != len(task.parts):
                counts["uncut"] += 1
                continue
        family_list = by_family.setdefault(pair["family"], [])
        if per_family is None or len(family_list) < per_family:
            family_list.append((world, meta, task, texts, pair))
    episodes = []
    for family, rows in by_family.items():
        for i, (world, meta, task, texts, pair) in enumerate(rows):
            episodes.append(make_episode(world, meta, task, texts, "testset", family, i, {"pair_id": pair["pair_id"], "rung": pair["rung"]}, seed))
    counts["episodes"] = len(episodes)
    return episodes, counts


def donors(episodes):
    """For each episode of a (split, family) group, the episode whose instruction it gets in the
    shuffled control: the next one (cyclically) with a different first instruction."""
    out = []
    for i, e in enumerate(episodes):
        n = len(episodes)
        pick = next((episodes[(i + k) % n] for k in range(1, n) if episodes[(i + k) % n].texts[0] != e.texts[0]), episodes[(i + 1) % n])
        out.append(pick)
    return out


def arm_episodes(arm, episodes):
    """The episodes of one arm from a (split, family) group's base episodes."""
    if arm == "shuffled":
        out = [replace(e, texts=[d.texts[min(j, len(d.texts) - 1)] for j in range(len(e.texts))], key={**e.key, "donor": d.key["i"]}) for e, d in zip(episodes, donors(episodes))]
    elif arm == "plain":
        out = [replace(e, texts=[r["plain"] for r in e.records]) for e in episodes]
    elif arm == "oracle":
        out = [replace(e, oracle=True) for e in episodes]
    else:
        out = list(episodes)
    return out


# ---------------------------------------------------------------------------------------------
# Running


def episode_row(arm, episode, played, seconds, thinking):
    row = {
        **episode.key,
        "arm": arm,
        "kind": S.task_kind(episode.task),
        "seed": episode.seed,
        "robots": int(episode.world.mass.sum()),
        "texts": episode.texts,
        "starts": episode.starts,
        "steps": episode.steps,
        "score": 0.0 if played.unasked else round(played.score, 4),
        "raw_score": round(played.score, 4),
        "unasked": played.unasked,
        "passed": played.passed,
        "spoke": played.spoke,
        "unfinished": played.unfinished,
        "moved": round(float(np.mean(played.moves)) if played.moves else 0.0, 4),
        "thoughts": played.thoughts,
        "seconds": round(seconds, 3),
    }
    if thinking:
        row["thought_checks"] = thought_checks(episode, played)
    return row


def run_arm(arm, episodes, policy=None, batch=8, expert_batch=48, views=R.stand_in_views, sample=False, max_new_tokens=R.MAX_NEW_TOKENS, emit=print):
    """Rows of one arm's episodes, emitted batch by batch."""
    rows = [None] * len(episodes)
    size = expert_batch if arm in ("expert", "still") else batch
    for indices in R.batches(episodes, size):
        part = [episodes[i] for i in indices]
        t0 = time.time()
        if arm == "expert":
            played = R.play_expert(part, seed=part[0].seed)
        elif arm == "still":
            played = R.play(part, lambda world, t: torch.full(world.mass.shape, R.STAY, dtype=torch.long))
        else:
            played = R.play_policy(policy, part, views, sample, max_new_tokens)
        seconds = (time.time() - t0) / len(part)
        for i, e, p in zip(indices, part, played):
            rows[i] = episode_row(arm, e, p, seconds, arm in THINKING_ARMS)
            emit(rows[i])
    return rows


def mean_se(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    se = float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return (float(v.mean()) if n else float("nan")), se


def ratio_se(top, bottom):
    """(mean(top) / mean(bottom), delta-method standard error) of paired samples."""
    a, b = np.asarray(top, float), np.asarray(bottom, float)
    n = len(a)
    r = se = float("nan")
    if n >= 2 and b.mean() != 0:
        r = a.mean() / b.mean()
        cov = np.cov(a, b, ddof=1)
        se = np.sqrt(max((cov[0, 0] + r * r * cov[1, 1] - 2 * r * cov[0, 1]) / (n * b.mean() ** 2), 0.0))
    return float(r), float(se)


def macro(entries, key):
    """(mean over families of entries' key, sqrt(sum se^2) / families) of a list of group entries."""
    usable = [e for e in entries if e.get(key) is not None and not np.isnan(e[key])]
    mean = float(np.mean([e[key] for e in usable])) if usable else float("nan")
    se = float(np.sqrt(sum(e[key + "_se"] ** 2 for e in usable)) / len(usable)) if usable else float("nan")
    return mean, se


def thought_values(rows, name):
    return [float(np.mean([c[name] for c in r["thought_checks"]])) for r in rows if r.get("thought_checks")]


def summarize(rows):
    """The summary of episode rows: groups, paired comparisons, macro means and gates."""
    groups = {}
    for r in rows:
        groups.setdefault((r["split"], r["family"], r["arm"]), []).append(r)
    entries = []
    for (split, family, arm), these in sorted(groups.items()):
        entry = {"split": split, "family": family, "arm": arm, "n": len(these)}
        for name, values in (("score", [r["score"] for r in these]), ("pass", [float(r["passed"]) for r in these]), ("unasked", [float(r["unasked"]) for r in these]), ("spoke", [float(r["spoke"] is not None) for r in these])):
            entry[name], entry[name + "_se"] = mean_se(values)
        entry["unfinished"] = float(np.mean([r["unfinished"] for r in these]))
        entry["seconds_per_episode"] = float(np.mean([r["seconds"] for r in these]))
        if arm in THINKING_ARMS:
            for name in ("parse", "first", "resolves"):
                entry["thought_" + name], entry["thought_" + name + "_se"] = mean_se(thought_values(these, name))
        entries.append(entry)
    paired = []
    for split, family in sorted({(s, f) for s, f, _ in groups}):
        by_arm = {arm: {r["i"]: r for r in groups.get((split, family, arm), [])} for arm in ARMS}
        entry = {"split": split, "family": family}
        for name, a, b in (("gap", "policy", "shuffled"), ("oracle_gain", "oracle", "policy")):
            common = sorted(set(by_arm[a]) & set(by_arm[b]))
            if common:
                entry[name + "_n"] = len(common)
                entry[name + "_score"], entry[name + "_score_se"] = mean_se([by_arm[a][i]["score"] - by_arm[b][i]["score"] for i in common])
                entry[name + "_pass"], entry[name + "_pass_se"] = mean_se([float(by_arm[a][i]["passed"]) - float(by_arm[b][i]["passed"]) for i in common])
        common = sorted(set(by_arm["oracle"]) & set(by_arm["plain"]))
        if common:
            entry["oracle_ratio_n"] = len(common)
            entry["oracle_ratio"], entry["oracle_ratio_se"] = ratio_se([float(by_arm["oracle"][i]["passed"]) for i in common], [float(by_arm["plain"][i]["passed"]) for i in common])
            entry["oracle_ratio_score"], entry["oracle_ratio_score_se"] = ratio_se([by_arm["oracle"][i]["score"] for i in common], [by_arm["plain"][i]["score"] for i in common])
        policy_rows = [r for r in by_arm["policy"].values() if r.get("thought_checks")]
        right = [float(r["passed"]) for r in policy_rows if all(c["first"] for c in r["thought_checks"])]
        wrong = [float(r["passed"]) for r in policy_rows if not all(c["first"] for c in r["thought_checks"])]
        entry["pass_right_thought"], entry["pass_wrong_thought"], entry["right_thought_n"] = (mean_se(right)[0] if right else None), (mean_se(wrong)[0] if wrong else None), len(right)
        paired.append(entry)
    macros = {}
    for split in sorted({e["split"] for e in entries}):
        for arm in ARMS:
            these = [e for e in entries if e["split"] == split and e["arm"] == arm]
            if these:
                out = {"families": len(these)}
                for name in ("score", "pass") + (("thought_parse", "thought_first", "thought_resolves") if arm in THINKING_ARMS else ()):
                    out[name], out[name + "_se"] = macro(these, name)
                macros.setdefault(split, {})[arm] = out
        these = [e for e in paired if e["split"] == split]
        for name in ("gap_score", "gap_pass", "oracle_gain_score", "oracle_ratio", "oracle_ratio_score"):
            if any(name in e for e in these):
                value, se = macro([e for e in these if name in e], name)
                macros.setdefault(split, {})[name] = {"mean": value, "se": se}
    gates = {}
    test = macros.get("test", {})
    if "policy" in test:
        rows_test = [r for r in rows if r["split"] == "test" and r["arm"] == "policy" and r.get("thought_checks")]
        right = [float(r["passed"]) for r in rows_test if all(c["first"] for c in r["thought_checks"])]
        wrong = [float(r["passed"]) for r in rows_test if not all(c["first"] for c in r["thought_checks"])]
        gates["a"] = {
            "thought_parse": test["policy"].get("thought_parse"),
            "thought_first": test["policy"].get("thought_first"),
            "gap_score": test.get("gap_score"),
            "pass_right_thought": mean_se(right) if right else None,
            "pass_wrong_thought": mean_se(wrong) if wrong else None,
            "passes": bool((test["policy"].get("thought_parse") or 0) >= GATE and (test["policy"].get("thought_first") or 0) >= GATE),
        }
    gates["b"] = {}
    for split, values in macros.items():
        if "oracle_ratio" in values and not np.isnan(values["oracle_ratio"]["mean"]):
            context = {f"{arm}_pass": values[arm]["pass"] for arm in ("plain", "oracle", "still") if arm in values}  # read the ratio against the floor (still)
            gates["b"][split] = {**values["oracle_ratio"], **context, "passes": bool(values["oracle_ratio"]["mean"] >= GATE)}
    return {"groups": entries, "paired": paired, "macro": macros, "gates": gates}


def fmt(value, se=None, digits=2):
    """value (± se) with `digits` decimals; "-" when there is none."""
    missing = value is None or (isinstance(value, float) and np.isnan(value))
    shown = "-" if missing else f"{value:.{digits}f}"
    if not missing and se is not None and not np.isnan(se):
        shown += f" ± {se:.{digits}f}"
    return shown


def markdown(summary):
    """The scorecard's tables: per split, one row per family (score and pass per arm, the paired
    comparisons and the thought checks), then the macro means and the gates."""
    groups = {(e["split"], e["family"], e["arm"]): e for e in summary["groups"]}
    paired = {(e["split"], e["family"]): e for e in summary["paired"]}
    lines = []
    for split in sorted({e["split"] for e in summary["groups"]}, key=lambda s: (SPLIT_GROUPS.index(s) if s in SPLIT_GROUPS else 2, s)):
        arms = [a for a in ARMS if any(k[0] == split and k[2] == a for k in groups)]
        lines += [f"### {split}", ""]
        head = ["family", "n"] + [f"{a} score" for a in arms] + [f"{a} pass" for a in arms]
        extra = []
        if "shuffled" in arms:
            extra.append(("gap (score)", "gap_score"))
        if "plain" in arms and "oracle" in arms:
            extra.append(("oracle/plain (pass)", "oracle_ratio"))
        thinking = [a for a in THINKING_ARMS if a in arms][:1]
        head += [h for h, _ in extra] + ([f"{thinking[0]} thought parse / first / resolves"] if thinking else [])
        lines += ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
        families = sorted({k[1] for k in groups if k[0] == split}, key=lambda f: FAMILIES.index(f) if f in FAMILIES else 99)
        for family in families:
            n = max(groups[(split, family, a)]["n"] for a in arms if (split, family, a) in groups)
            cells = [family, str(n)]
            cells += [fmt(groups[(split, family, a)]["score"], groups[(split, family, a)]["score_se"]) if (split, family, a) in groups else "-" for a in arms]
            cells += [fmt(groups[(split, family, a)]["pass"], groups[(split, family, a)]["pass_se"]) if (split, family, a) in groups else "-" for a in arms]
            p = paired.get((split, family), {})
            cells += [fmt(p.get(key), p.get(key + "_se")) for _, key in extra]
            if thinking:
                g = groups.get((split, family, thinking[0]), {})
                cells.append(" / ".join(fmt(g.get(f"thought_{k}")) for k in ("parse", "first", "resolves")))
            lines.append("| " + " | ".join(cells) + " |")
        m = summary["macro"].get(split, {})
        cells = ["macro", ""] + [fmt(m.get(a, {}).get("score"), m.get(a, {}).get("score_se")) for a in arms] + [fmt(m.get(a, {}).get("pass"), m.get(a, {}).get("pass_se")) for a in arms]
        cells += [fmt(m.get(key, {}).get("mean"), m.get(key, {}).get("se")) for _, key in extra]
        if thinking:
            cells.append(" / ".join(fmt(m.get(thinking[0], {}).get(f"thought_{k}")) for k in ("parse", "first", "resolves")))
        lines += ["| " + " | ".join(cells) + " |", ""]
    gates = summary.get("gates", {})
    if gates:
        lines += ["### gates", ""]
        if "a" in gates:
            a = gates["a"]
            lines.append(f"1. a) thoughts parse {fmt(a['thought_parse'])} and match their sub-goal {fmt(a['thought_first'])} (need >= {GATE}); shuffled gap {fmt((a['gap_score'] or {}).get('mean'), (a['gap_score'] or {}).get('se'))}; pass with a right first thought {fmt(*(a['pass_right_thought'] or (None, None)))}, with a wrong one {fmt(*(a['pass_wrong_thought'] or (None, None)))}: {'passes' if a['passes'] else 'fails'}")
        for split, b in gates.get("b", {}).items():
            lines.append(f"1. b) {split}: pass(oracle thought) / pass(plain instruction) = {fmt(b['mean'], b['se'])} (need >= {GATE}; pass shares: plain {fmt(b.get('plain_pass'))}, oracle {fmt(b.get('oracle_pass'))}, no moves {fmt(b.get('still_pass'))}): {'passes' if b['passes'] else 'fails'}")
        lines.append("")
    return "\n".join(lines)


def compare(path_a, path_b):
    """Markdown: per (split, arm), B - A over the episodes both scored (paired by split, family, i,
    arm), macro over families with standard errors, and the per-family differences."""

    def rows_of(path):
        return {(r["split"], r["family"], r["i"], r["arm"], r.get("repeat", 0)): r for r in (json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()) if "summary" not in r}

    a, b = rows_of(path_a), rows_of(path_b)
    common = sorted(set(a) & set(b))
    groups = {}
    for key in common:
        groups.setdefault((key[0], key[3]), {}).setdefault(key[1], []).append(key)
    lines = [f"B = {path_b} minus A = {path_a}, {len(common)} paired episodes", "", "| split | arm | families | score B - A | pass B - A | per family (score) |", "| --- | --- | --- | --- | --- | --- |"]
    for (split, arm), families in sorted(groups.items()):
        entries = []
        for family, keys in families.items():
            entry = {"family": family}
            entry["score"], entry["score_se"] = mean_se([b[k]["score"] - a[k]["score"] for k in keys])
            entry["pass"], entry["pass_se"] = mean_se([float(b[k]["passed"]) - float(a[k]["passed"]) for k in keys])
            entries.append(entry)
        per = ", ".join(f"{e['family']} {e['score']:+.2f}" for e in entries)
        lines.append(f"| {split} | {arm} | {len(entries)} | {fmt(*macro(entries, 'score'))} | {fmt(*macro(entries, 'pass'))} | {per} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Models


def tiny_policy(name="Qwen/Qwen3.5-0.8B", seed=0):
    """A tiny randomly initialised policy with a cached model's architecture, tokenizer and image
    processor (for tests, smoke runs and speed checks; nothing downloaded)."""
    import transformers

    from beebot.vla.voxel.think import ThinkVoxelPolicy

    config = transformers.AutoConfig.from_pretrained(name, local_files_only=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(name, local_files_only=True)
    torch.manual_seed(seed)
    image_processor = None
    if hasattr(config, "vision_config"):
        from transformers.models.qwen2_vl import Qwen2VLImageProcessorPil

        image_processor = Qwen2VLImageProcessorPil.from_pretrained(name, local_files_only=True)
        text, vision = config.text_config, config.vision_config
        text.hidden_size, text.intermediate_size, text.num_hidden_layers = 32, 64, 4
        text.layer_types = ["linear_attention"] * 3 + ["full_attention"]
        text.num_attention_heads, text.num_key_value_heads, text.head_dim = 2, 1, 32
        text.linear_num_key_heads, text.linear_num_value_heads, text.linear_key_head_dim, text.linear_value_head_dim = 2, 4, 8, 8
        text.rope_parameters = {**text.rope_parameters, "partial_rotary_factor": 0.5, "mrope_section": [3, 3, 2]}
        vision.depth, vision.hidden_size, vision.intermediate_size, vision.num_heads, vision.out_hidden_size = 2, 32, 64, 2, 32
        lm = transformers.AutoModelForImageTextToText.from_config(config).float()
    else:
        config.hidden_size, config.intermediate_size, config.num_hidden_layers = 32, 64, 2
        config.num_attention_heads, config.num_key_value_heads, config.head_dim = 4, 2, 8
        lm = transformers.AutoModelForCausalLM.from_config(config).float()
    return ThinkVoxelPolicy(lm, tokenizer, image_processor).eval()


def load_model(args):
    """The policy of --checkpoint, --model (base weights, new parts freshly initialised) or --tiny."""
    from beebot.vla.voxel import think

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    if args.checkpoint:
        policy, _ = think.load_policy(args.checkpoint, args.device, dtype)
    elif args.model:
        torch.manual_seed(0)
        lm, tokenizer, image_processor = think.load_lm(args.model, dtype)
        policy = think.ThinkVoxelPolicy(lm, tokenizer, image_processor).to(args.device).eval()
        policy.lm.to(dtype)
    else:
        policy = tiny_policy(args.tiny).to(args.device)
    return policy


class SceneViews:
    """views.scene_views with the default ViewSpec (voxel/views.py, once it lands), one look
    (cameras, lights, materials: views.sample_look) per episode, kept for all its steps; the
    rollout gives each episode its own rng, which keys the look."""

    def __init__(self):
        from beebot.vla.voxel import views as V

        self.V, self.spec, self.looks = V, V.ViewSpec(), {}

    def __call__(self, obs, rng):
        if id(rng) not in self.looks:
            self.looks[id(rng)] = (rng, self.V.sample_look(rng, self.spec))  # the rng is kept so its id is not reused
        return self.V.scene_views(obs, rng, self.spec, look=self.looks[id(rng)][1])


def views_of(name):
    """The views function: "stand_in" (rollout.stand_in_views) or "scene" (SceneViews)."""
    return SceneViews() if name == "scene" else R.stand_in_views


# ---------------------------------------------------------------------------------------------
# Live numbers for ops (JSON lines, appended while the card runs)

OPS_ARMS = {"policy": "model", "oracle": "oracle_thought"}  # rollout_success's arm names; the others keep theirs
SCORECARD_LOG = Path("logs/scorecard3d.jsonl")
ROLLOUT_LOG = Path("logs/think_act_loop.jsonl")


class OpsLog:
    """Appends JSON lines, each with the run's name and the time, to a log file."""

    def __init__(self, path, run):
        self.path, self.run = Path(path), run
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, row):
        with open(self.path, "a") as f:
            f.write(json.dumps({**row, "run": self.run, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")


def arm_lines(arm, rows, repeat):
    """When an arm has finished its episodes of one repeat: per split, its rollout_success (macro
    pass share over families) and its scorecard3d_mean (macro score) with the standard error."""
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], {}).setdefault(r["family"], []).append(r)
    rollout, card_lines = [], []
    for split, families in sorted(by_split.items()):
        entries = []
        for these in families.values():
            entry = {}
            entry["score"], entry["score_se"] = mean_se([r["score"] for r in these])
            entry["pass"], entry["pass_se"] = mean_se([float(r["passed"]) for r in these])
            entries.append(entry)
        n = sum(len(v) for v in families.values())
        success, success_se = macro(entries, "pass")
        value, se = macro(entries, "score")
        rollout.append({"arm": OPS_ARMS.get(arm, arm), "rollout_success": round(success, 4), "se": round(success_se, 4), "n": n, "split": split, "family": "all", "repeat": repeat})
        card_lines.append({"arm": arm, "split": split, "scorecard3d_mean": round(value, 4), "scorecard3d_se": round(se, 4), "families": len(entries), "n": n, "repeat": repeat})
    return rollout, card_lines


def noise_lines(means, standard_errors):
    """scorecard3d_noise per (arm, split): the standard deviation of the scorecard mean over the
    repeats (fresh scenes and seeds each) when there are two or more, else the macro standard error
    of the one card, which estimates the same spread."""
    out = []
    for key, values in sorted(means.items()):
        arm, split = key
        repeated = len(values) >= 2
        noise = float(np.std(values, ddof=1)) if repeated else standard_errors[key][0]
        out.append({"arm": arm, "split": split, "scorecard3d_noise": round(noise, 4), "noise_from": "repeats" if repeated else "standard_error", "repeats": len(values), "scorecard3d_mean": round(float(np.mean(values)), 4), "standard_error": round(float(np.mean(standard_errors[key])), 4)})
    return out


# ---------------------------------------------------------------------------------------------
# The command


def base_episodes(args, log=print, seed=SEED):
    """{(split, family): base episodes} of the chosen splits and families, scenes from seed."""
    out = {}
    families = [f for f in FAMILIES if f in args.families]
    if "test" in args.splits:
        for family in families:
            out[("test", family)] = drawn_episodes("test", family, args.per_family, seed=seed)
    if "wording" in args.splits:
        for family in families:
            out[("wording", family)] = drawn_episodes("wording", family, args.per_held, seed=seed)
    if "held" in args.splits:
        for key, family in held_splits():
            if family in families:
                out[(f"held:{key}", family)] = drawn_episodes(key, family, args.per_held, name=f"held:{key}", seed=seed)
    if "testset" in args.splits:
        episodes, counts = testset_episodes(args.testset, args.testset_per_family, families, seed)
        log(json.dumps({"testset": counts}))
        for e in episodes:
            out.setdefault(("testset", e.key["family"]), []).append(e)
    return out


def main(argv=None):
    """`compare A B` prints the comparison; anything else runs the scorecard (card) and returns its summary."""
    argv = list(sys.argv[1:] if argv is None else argv)
    summary = None
    if argv[:1] == ["compare"]:
        print(compare(argv[1], argv[2]))
    else:
        summary = card(argv)
    return summary


def card(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default=None, help="a think_train checkpoint (policy_step*.pt)")
    parser.add_argument("--model", default=None, help="base weights from the local cache, e.g. Qwen/Qwen3.5-0.8B (new parts random)")
    parser.add_argument("--tiny", default=None, help="a tiny random model with this cached model's architecture (smoke tests, speed)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--splits", default=",".join(SPLIT_GROUPS))
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--per-family", type=int, default=48)
    parser.add_argument("--per-held", type=int, default=16)
    parser.add_argument("--testset", default=str(TESTSET))
    parser.add_argument("--testset-per-family", type=int, default=None)
    parser.add_argument("--batch", type=int, default=8, help="policy episodes per forward")
    parser.add_argument("--expert-batch", type=int, default=48)
    parser.add_argument("--sample", action="store_true", help="sample actions from the softmax (default argmax)")
    parser.add_argument("--max-new-tokens", type=int, default=R.MAX_NEW_TOKENS)
    parser.add_argument("--views", choices=("stand_in", "scene"), default="stand_in")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None, help="cap every episode at this many steps (smoke runs; the card then differs from the default one)")
    parser.add_argument("--repeats", type=int, default=1, help="whole cards on fresh scenes and seeds (seed SEED + r); two or more measure scorecard3d_noise directly")
    parser.add_argument("--log", default=str(SCORECARD_LOG), help="scorecard3d_mean and scorecard3d_noise lines are appended here")
    parser.add_argument("--rollout-log", default=str(ROLLOUT_LOG), help="rollout_success lines are appended here")
    args = parser.parse_args(argv)
    args.arms = [a for a in args.arms.split(",") if a]
    args.splits = [s for s in args.splits.split(",") if s]
    args.families = [f for f in args.families.split(",") if f]
    if args.threads:
        torch.set_num_threads(args.threads)
    needs_model = any(a not in ("expert", "still") for a in args.arms)
    if needs_model and not (args.checkpoint or args.model or args.tiny):
        parser.error("arms other than expert and still need --checkpoint, --model or --tiny")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy = load_model(args) if needs_model else None
    views = views_of(args.views)
    t0 = time.time()
    card_log, rollout_log = OpsLog(args.log, out.name), OpsLog(args.rollout_log, out.name)
    rows = []
    means, errors = {}, {}
    with open(out / "episodes.jsonl", "w") as sink:

        def emit(row):
            sink.write(json.dumps(row) + "\n")
            sink.flush()

        for repeat in range(args.repeats):
            groups = base_episodes(args, seed=SEED + repeat)
            if args.steps:
                groups = {key: [replace(e, steps=min(e.steps, args.steps)) for e in episodes] for key, episodes in groups.items()}
            print(json.dumps({"repeat": repeat, "drawn": {f"{s}/{f}": len(v) for (s, f), v in groups.items()}, "seconds": round(time.time() - t0, 1)}), flush=True)
            for arm in args.arms:
                arm_rows = []
                for (split, family), episodes in groups.items():
                    group = "held" if split.startswith("held:") else split
                    if group not in ARM_SPLITS[arm]:
                        continue
                    t1 = time.time()
                    done = run_arm(arm, arm_episodes(arm, episodes), policy, args.batch, args.expert_batch, views, args.sample, args.max_new_tokens, emit)
                    for r in done:
                        r["repeat"] = repeat
                    arm_rows += done
                    success = float(np.mean([r["passed"] for r in done]))
                    rollout_log({"arm": OPS_ARMS.get(arm, arm), "rollout_success": round(success, 4), "n": len(done), "split": split, "family": family, "repeat": repeat})
                    print(json.dumps({"repeat": repeat, "arm": arm, "split": split, "family": family, "n": len(done), "score": round(float(np.mean([r["score"] for r in done])), 3), "pass": round(success, 3), "seconds": round(time.time() - t1, 1)}), flush=True)
                rollout_lines, card_lines = arm_lines(arm, arm_rows, repeat)
                for line in rollout_lines:
                    rollout_log(line)
                for line in card_lines:
                    card_log(line)
                    means.setdefault((arm, line["split"]), []).append(line["scorecard3d_mean"])
                    errors.setdefault((arm, line["split"]), []).append(line["scorecard3d_se"])
                rows += arm_rows
        for line in noise_lines(means, errors):
            card_log(line)
            print(json.dumps(line), flush=True)
        summary = summarize([r for r in rows if r["repeat"] == 0])
        summary["repeats"] = noise_lines(means, errors)
        config = {k: v for k, v in vars(args).items()}
        config.update(seed=SEED, rule=R.RULE, failure=R.FAILURE.tolist(), steps=R.STEPS, build_steps=R.BUILD_STEPS, seconds=round(time.time() - t0, 1))
        sink.write(json.dumps({"summary": {**summary, "config": config}}) + "\n")
    table = markdown(summary)
    (out / "card.md").write_text(table + "\n")
    print(table, flush=True)
    return summary


if __name__ == "__main__":
    main()
