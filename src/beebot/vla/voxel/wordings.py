"""The wording bank of the voxel suite (wordings.json): many phrasings per intent, open slot words
that map back to canonical values, and categories that denote sets of colours.

The policy restates an instruction in plain canonical words before it acts, so every training
instruction comes as a pair: the instruction as a person might say it, and one fixed canonical
sentence per intent. rewrite() makes that pair from a suite task's canonical slots (slots_of).

    slots = slots_of(task, present=[c for _, c in suite.present(world, i)])
    instruction, plain = rewrite(task.text, slots, rng, "train")

Splits (SPLITS). "train" draws only training material. Each other split holds out one thing and
keeps the rest plain, so a score on it has one cause:
  wording     held-out templates, canonical slot words
  words       held-out synonyms and descriptions (pink and pyramid: all of theirs) in training templates
  languages   colour and count words of the held-out languages
  categories  held-out colour categories ("warm"): one that picks out the task's cheeses in the scene
  frames      training descriptions in a held-out frame ("the same colour as a tomato"): a new
              combination of seen words
A slot with nothing to offer in a split stays canonical; render() lists what was replaced.

Three rules decide which expression may fill which placeholder:
  1. a phrase ("the colour of grass", "a warm colour") only goes where the template has a
     predicative frame just before the placeholder (PREDICATIVE: "the cheese that is {c}");
  2. a template that says "letter" takes single letters only;
  3. shape expressions follow "a" and region expressions follow "the" (tests check the lists).

Token separation (token_clashes): in the form the policy reads (lowercase after a space) no
held-out slot expression shares a token with any training slot expression of any value, apart
from SHARED_TOKENS (function and frame words, listed in the file).
"""

import json
import re
import string
from dataclasses import dataclass, field
from pathlib import Path

from beebot.vla import tasks as T2
from beebot.vla.voxel import suite as S

PATH = Path(__file__).with_name("wordings.json")
DATA = json.loads(PATH.read_text())
INTENTS = DATA["intents"]
SLOTS = DATA["slots"]
LEXICAL = DATA["lexical"]
CATEGORIES = DATA["categories"]
LANGUAGES = DATA["languages"]
PREDICATIVE = tuple(DATA["predicative"])
SHARED_TOKENS = tuple(DATA["shared_tokens"])
SPLITS = ("train", "wording", "words", "languages", "categories", "frames")
KEEP = 0.3  # training: share of slots that keep their canonical word
CATEGORY = 0.5  # training: share of the tasks a set category fits that are worded with it
SUB = {"region": "sub.region", "tower": "sub.tower", "surround": "sub.surround", "eat": "sub.eat"}
SETS = {"logic.and": ("set.eat", "eat"), "logic.all": ("set.eat", "eat"), "logic.neither": ("set.except", "forbid"), "logic.all_but_two": ("set.except", "forbid"), "logic.or": ("set.or", "any_of")}
ALIASES = {"eat": "logic.single", "logic.all_but_two": "logic.neither"}  # intents whose canonical sentence is the same
SHAPE_NAMES = {words: name for name, words in S.SHAPE_WORDS.items()}
COMPOSITE = ("sub.", "set.", "who.", "do.")  # parts of other intents, never a whole instruction
DO = {"goto": ("do.region", "region"), "approach": ("do.approach", "colour"), "gather": ("do.gather", None), "spread": ("do.spread", None), "groups": ("do.groups", None), "move": ("do.move", "toward"), "perch": ("do.perch", "colour"), "climb": ("do.climb", "side"), "tower": ("do.tower", "height"), "eat": ("do.eat", "colour")}
MOTION = {"stay": ("stay", None), "goto": ("region", "region"), "approach": ("approach", "colour"), "follow": ("follow", "colour"), "gather": ("gather", None), "spread": ("spread", None), "climb": ("climb", "side"), "move": ("move", "toward")}
MOTION_SLOT = {"region": "r", "approach": "c", "follow": "c", "climb": "r", "move": "r"}


@dataclass
class Choice:
    """One replacement rewrite made: the slot, its canonical value (a list of colours for a
    category), the text used and where it came from (synonym, thing, language, category, ...)."""

    slot: str
    value: object
    text: str
    source: str


@dataclass
class Rewrite:
    instruction: str
    canonical: str
    template: str
    choices: list = field(default_factory=list)


def fields_of(template):
    return sorted({name for _, name, _, _ in string.Formatter().parse(template) if name})


def pick(rng, items):
    """A uniform draw that works with numpy generators and random.Random alike."""
    return items[int(rng.random() * len(items))]


def templates(intent, split="train"):
    """The templates of an intent: the held-out ones for the "wording" split, else the training ones."""
    return INTENTS[intent]["held_out" if split == "wording" else "train"]


def suite_held(entry):
    return bool(entry.get("suite_held_out"))


# ---------------------------------------------------------------------------------------------
# The lexicon: canonical value -> expressions


def plain(kind, value):
    """The canonical word of a slot value, as the canonical sentence spells it."""
    return SLOTS["shape"]["values"][value]["canonical"] if kind == "shape" else str(value)


def colour_entries(value, split):
    """[(source, [(text, is_phrase)])] of a colour: each entry is one word or thing with the
    surface forms it can take."""
    entry = SLOTS["colour"]["values"][value]
    frames = SLOTS["colour"]["frames"]
    part = "train" if split in ("train", "frames") else "held_out"
    tongues = [] if suite_held(entry) and split == "train" else list(entry["languages"]) if suite_held(entry) else [name for name in LANGUAGES[part] if name in entry["languages"]]
    out = []
    if split in ("train", "words"):
        out += [("synonym", [(word, False)]) for word in entry["synonyms"][part]]
    if split in ("train", "words"):
        for bare, noun in entry["things"][part]:
            forms = [(frame.format(thing=bare), False) for frame in frames["adjective"]["train"]] + [(frame.format(np=noun), True) for frame in frames["phrase"]["train"]]
            out.append(("thing", forms))
    if split == "frames":
        for bare, noun in entry["things"]["train"]:
            out.append(("frame", [(frame.format(np=noun), True) for frame in frames["phrase"]["held_out"]]))
    if split in ("train", "languages"):
        out += [("language", [(entry["languages"][name], False)]) for name in tongues]
    return out


def entries(kind, value, split):
    """[(source, [(text, is_phrase)])] of the expressions a slot value has in a split ("train",
    "words", "languages", "frames"); categories are added by the rewriter, which knows the scene."""
    out = []
    if kind == "colour":
        out = colour_entries(value, split)
    elif kind == "count":
        entry = SLOTS["count"]["values"][str(value)]
        if split in ("train", "words"):
            out += [("synonym", [(word, False)]) for word in entry["train" if split == "train" else "held_out"]]
        if split in ("train", "languages"):
            out += [("language", [(entry["languages"][name], False)]) for name in LANGUAGES["train" if split == "train" else "held_out"] if name in entry["languages"]]
    elif kind in ("region", "shape"):
        entry = SLOTS[kind]["values"][value]
        if split in ("train", "words"):
            out += [("synonym", [(word, False)]) for word in entry["train" if split == "train" else "held_out"]]
    elif kind == "letter":
        if split in ("train", "words") and len(value) == 1:
            out += [("synonym", [(frame.format(w=value), False)]) for frame in SLOTS["letter"]["train" if split == "train" else "held_out"]]
    elif kind in LEXICAL:
        if split in ("train", "words"):
            out += [("synonym", [(word, False)]) for word in LEXICAL[kind]["train" if split == "train" else "held_out"]]
    else:
        raise ValueError(kind)
    return out


def category_entries(target, present, split):
    """[(source, forms, colours)] of the categories that pick out exactly the colours `target`
    among the cheeses present, skipping scenes that hold a colour the category is unclear about."""
    part = "held_out" if split == "categories" else "train"
    out = []
    for cat in CATEGORIES:
        if cat["split"] == part and set(cat["set"]) & set(present) == set(target) and not set(cat["unclear"]) & set(present):
            out.append(("category", [(text, False) for text in cat["adjective"]] + [(text, True) for text in cat["predicate"]]))
    return out


def fitting(forms, template, name):
    """The surface forms that may fill placeholder `name` of the template (rule 1)."""
    predicative = template.split("{" + name + "}")[0].endswith(PREDICATIVE)
    return [text for text, phrase in forms if predicative or not phrase]


def takes(template, slots):
    """Rule 2: a template that says "letter" needs a single letter."""
    return len(str(slots.get("w", "A"))) == 1 or not re.search(r"\bletter\b", template)


# ---------------------------------------------------------------------------------------------
# Canonical sentences and their inverse


def canonical(slots):
    """The plain sentence of a task's canonical slots: one fixed template per intent."""
    intent = slots["intent"]
    spec = INTENTS[intent]
    if intent == "split":
        words = {name: INTENTS[SUB[kind]]["canonical"].format(x=plain(INTENTS[SUB[kind]]["slots"]["x"], x)) for name, (kind, x) in zip("ab", slots["jobs"])}
    elif intent == "subset":
        words = {"who": who_phrase(slots["who"]), "job": job_phrase(slots["job"])}
    else:
        words = {name: plain(kind, slots[name]) for name, kind in spec["slots"].items()}
    return spec["canonical"].format(**words)


def part_phrase(part):
    """The canonical phrase of a [sub-intent, value] part of a composite intent ("who." or "do."):
    the sub-intent's canonical with its slot x filled, recursively for "who.not"."""
    intent, value = part
    spec = INTENTS[intent]
    if intent == "who.not":
        return spec["canonical"].format(x=who_phrase(value))
    return spec["canonical"].format(x=plain(spec["slots"]["x"], value)) if spec["slots"] else spec["canonical"]


who_phrase = part_phrase
job_phrase = part_phrase


def part_words(part):
    """The canonical slot words inside a composite part (for check_text)."""
    intent, value = part
    if intent == "who.not":
        return part_words(value)
    return [plain(INTENTS[intent]["slots"]["x"], value)] if INTENTS[intent]["slots"] else []


def parse_part(text, prefix):
    """The [sub-intent, value] of a canonical "who." or "do." phrase, or None."""
    if prefix == "who.":
        match = re.fullmatch(INTENTS["who.not"]["canonical"].replace("{x}", "(?P<x>.+)"), text)
        if match is not None:
            inner = parse_part(match["x"], "who.")
            return None if inner is None else ["who.not", inner]
    for intent, spec in INTENTS.items():
        if not intent.startswith(prefix) or intent == "who.not":
            continue
        match = pattern(spec["canonical"], spec["slots"]).match(text)
        if match is not None:
            return [intent, typed(spec["slots"]["x"], match["x"]) if spec["slots"] else None]
    return None


def vocabulary(kind):
    """A regular expression of the canonical words a slot kind can take."""
    if kind == "colour":
        options = list(SLOTS["colour"]["values"])
    elif kind == "region":
        options = list(SLOTS["region"]["values"])
    elif kind == "shape":
        options = [entry["canonical"] for entry in SLOTS["shape"]["values"].values()]
    elif kind == "count":
        options = [r"\d+"]
    elif kind == "letter":
        options = ["[A-Z]{1,2}"]
    else:
        options = [".+"]
    return "|".join(sorted((o if kind in ("count", "letter", "job", "who", "do") else re.escape(o) for o in options), key=len, reverse=True))


def pattern(template, kinds):
    """An anchored regular expression of a template whose placeholders only match the canonical
    vocabulary of their kinds."""
    parts = re.split(r"(\{\w+\})", template)
    return re.compile("^" + "".join(f"(?P<{p[1:-1]}>{vocabulary(kinds[p[1:-1]])})" if p.startswith("{") else re.escape(p) for p in parts) + "$")


def typed(kind, word):
    return int(word) if kind == "count" else SHAPE_NAMES[word] if kind == "shape" else word


def parse_canonical(text):
    """The canonical slots of a canonical sentence: the inverse of canonical(), up to ALIASES
    (two intents that say the same thing share a sentence). Raises ValueError on anything else."""
    found = None
    for intent, spec in INTENTS.items():
        if found is not None or intent in ALIASES or intent.startswith(COMPOSITE):
            continue
        match = pattern(spec["canonical"], spec["slots"]).match(text)
        if match is None:
            continue
        if intent == "subset":
            who, job = parse_part(match["who"], "who."), parse_part(match["job"], "do.")
            if who is not None and job is not None:
                found = {"intent": intent, "who": who, "job": job}
        elif intent == "split":
            jobs = []
            for name in "ab":
                for kind, sub in SUB.items():
                    inner = pattern(INTENTS[sub]["canonical"], INTENTS[sub]["slots"]).match(match[name])
                    if inner is not None:
                        jobs.append((kind, typed(INTENTS[sub]["slots"]["x"], inner["x"])))
            if len(jobs) == 2:
                found = {"intent": intent, "jobs": jobs}
        else:
            found = {"intent": intent, **{name: typed(kind, match[name]) for name, kind in spec["slots"].items()}}
    if found is None:
        raise ValueError(f"not a canonical sentence: {text!r}")
    return found


# ---------------------------------------------------------------------------------------------
# From a suite task to its canonical slots


def invert(text, keys):
    """(key, slot words) of a suite instruction, by matching it against every template the suite
    can have used for the keys, with placeholders restricted to the suite's vocabulary. Exact:
    raises ValueError unless the matches agree on one reading."""
    kinds = {"w": "letter", "s": "shape", "a": "colour", "c": "colour"}
    readings = set()
    for key in keys:
        pool = T2.wordings(key, "") + T2.wordings(key, "wording") if key.startswith("logic.") else S.WORDINGS[key]
        for template in pool:
            match = pattern(template, kinds).match(text)
            if match is not None:
                readings.add((key, tuple(sorted(match.groupdict().items()))))
    if len(readings) != 1:
        raise ValueError(f"{len(readings)} readings of {text!r} as {keys}")
    key, words = readings.pop()
    return key, dict(words)


def slots_of(task, present=None):
    """The canonical slots of a suite task: {"intent": key, slot: value, ...}, a parallel task as
    {"intent": "split", "jobs": [(kind, value), (kind, value)]}. present: the colours of the
    scene's cheeses (suite.present); with it the slots carry "present" and, for eat tasks, the
    colour sets, which is what lets categories be used.

    Read off task.spec. The spec does not say two things, which are recovered from task.text by
    exact template inversion (invert): what a formation spells or shapes, and the colour named by
    an eat task whose cheese is absent from the scene. If Task carried the wording key and
    arguments its maker passed to say(), neither would be needed."""
    spec = task.spec
    kind = spec["kind"]
    if task.parts:
        raise ValueError("a switch task has one instruction per part: slots_of(part) for each of task.parts")
    if "who" in spec:
        out = coverage_slots(spec)
    elif kind == "mask":
        key, words = invert(task.text, ("letter_floor", "letter_stand", "shape"))
        out = {"intent": key, "s": SHAPE_NAMES[words["s"]]} if key == "shape" else {"intent": key, "w": words["w"]}
    elif kind == "tower":
        out = {"intent": "tower", "n": int(spec["height"])}
    elif kind in ("bridge", "fill", "stairs"):
        out = {"intent": kind}
    elif kind == "parallel":
        out = {"intent": "split", "jobs": [(job, x) for job, x in spec["jobs"]]}
    elif kind == "herd_trench":
        out = {"intent": "herd_trench", "c": spec["cheese"]}
    elif kind == "herd_region":
        cheeses = list(spec["cheeses"])
        out = {"intent": "herd_region", "c": cheeses[0], "r": spec["region"]} if len(cheeses) == 1 else {"intent": "herd_two", "c": cheeses[0], "d": cheeses[1], "r": spec["region"]}
    elif kind == "eat":
        form = spec["form"]
        eat, forbid, either = sorted(spec["eat"]), sorted(spec["forbid"]), sorted(spec["any_of"])
        if task.family == "runner":
            out = {"intent": "runner", "c": eat[0]} if form == "single" else {"intent": "runner_except", "c": forbid[0]}
        elif form == "single":
            out = {"intent": "logic.single", "a": eat[0]}
        elif form == "absent":
            out = {"intent": "logic.single", "a": invert(task.text, ("logic.single",))[1]["a"]}
        elif form == "except":
            out = {"intent": "logic.except", "a": forbid[0]}
        elif form == "or":
            out = {"intent": "logic.or", "a": either[0], "b": either[1]}
        elif form == "and":
            out = {"intent": "logic.and", "a": eat[0], "b": eat[1]}
        elif form == "ordered":
            out = {"intent": "logic.ordered", "a": spec["order"][0], "b": spec["order"][1]}
        elif form in ("all", "none"):
            out = {"intent": "logic." + form}
        else:  # neither, all_but_two
            out = {"intent": "logic." + form, "a": forbid[0], "b": forbid[1]}
        if present is not None and task.family == "logic":
            out.update(eat=eat, forbid=forbid, any_of=either)
    else:
        raise ValueError(kind)
    if present is not None:
        out["present"] = list(present)
    return out


def who_part(who):
    """[sub-intent, value] of a suite selector (suite.spec["who"])."""
    kind = who["select"]
    if kind == "group":
        part = ["who.side", who["which"]] if who["by"] == "side" else ["who.cheese", who["which"]] if who["by"] == "cheese" else ["who." + who["which"], None]
    else:
        part = ["who.count", int(who["count"])] if kind == "count" else ["who." + kind, None]
    return ["who.not", part] if who.get("except") else part


def job_part(spec):
    """[sub-intent, value] of a coverage job done by a selected part of the swarm."""
    kind = spec["kind"]
    if kind == "line":
        return ["do.line_" + spec["axis"], None]
    intent, field_name = DO[kind]
    return [intent, None if field_name is None else spec[field_name]]


def coverage_slots(spec):
    """Canonical slots of a coverage task (suite: "who" in its spec): a motion intent for
    everyone, {"intent": "subset", "who": part, "job": part} for a selected part of the swarm."""
    kind = spec["kind"]
    if spec["who"]["select"] != "all":
        return {"intent": "subset", "who": who_part(spec["who"]), "job": job_part(spec)}
    if kind == "groups":
        return {"intent": "groups." + ("two" if spec["count"] == 2 else "three")}
    if kind == "line":
        return {"intent": "line." + spec["axis"]}
    intent, field_name = MOTION[kind]
    return {"intent": intent, MOTION_SLOT[intent]: spec[field_name]} if field_name else {"intent": intent}


def check_text(text, slots):
    """Raises ValueError unless every canonical slot word occurs in the suite's instruction."""
    if slots["intent"] == "subset":
        words = part_words(slots["who"]) + part_words(slots["job"])
    elif slots["intent"] == "split":
        words = [plain(INTENTS[SUB[kind]]["slots"]["x"], x) for kind, x in slots["jobs"]]
    else:
        words = [plain(kind, slots[name]) for name, kind in INTENTS[slots["intent"]]["slots"].items()]
    missing = [word for word in words if word not in text]
    if missing:
        raise ValueError(f"{missing} not in the instruction {text!r}")


# ---------------------------------------------------------------------------------------------
# The rewriter


def choose(rng, kind, value, template, name, split, present, keep, choices, slot):
    """The text for one placeholder: the canonical word, or an expression of the split that fits."""
    part = "train" if split == "wording" else split
    options = [] if split == "wording" else entries(kind, value, "words" if part == "categories" and kind != "colour" else part) if part != "categories" else []
    if kind == "colour" and present is not None and value in present and len(set(present)) > 1 and split in ("train", "categories"):  # "the warm-coloured cheese" only where it tells cheeses apart
        options = options + category_entries([value], present, split)
    options = [(source, fitting(forms, template, name)) for source, forms in options]
    options = [(source, forms) for source, forms in options if forms]
    word = plain(kind, value)
    if options and (split != "train" or rng.random() >= keep):
        source, forms = pick(rng, options)
        word = pick(rng, forms)
        choices.append(Choice(slot, value, word, source))
    return word


def fill(rng, intent, values, split, present, keep, choices, prefix=""):
    """One template of the intent with its placeholders filled; values: {placeholder: value}."""
    spec = INTENTS[intent]
    pool = [t for t in templates(intent, split) if takes(t, values)]
    if split == "frames":  # a held-out frame is a phrase, so it needs a predicative template
        pool = [t for t in pool if any(t.split("{" + name + "}")[0].endswith(PREDICATIVE) for name, kind in spec["slots"].items() if kind == "colour")] or pool
    template = pick(rng, pool)
    words = {}
    for name in fields_of(template):
        kind = spec["slots"].get(name, name)  # a placeholder that is not a slot is a lexical word (trench, plateau)
        value = values.get(name, LEXICAL[kind]["canonical"] if kind in LEXICAL else None)
        words[name] = choose(rng, kind, value, template, name, split, present, keep, choices, prefix + name)
    return template, template.format(**words)


def fill_part(rng, part, split, present, keep, choices, prefix):
    """A wording of a [sub-intent, value] part of a composite intent, "who.not" around its inner
    selector."""
    intent, value = part
    if intent == "who.not":
        return pick(rng, templates(intent, split)).format(x=fill_part(rng, value, split, present, keep, choices, prefix))
    return fill(rng, intent, {"x": value} if INTENTS[intent]["slots"] else {}, split, present, keep, choices, prefix=prefix)[1]


def set_category(rng, slots, split, category):
    """(set intent, colours, category forms) when a category names the task's whole colour set in
    this scene and the draw says to use it; None otherwise."""
    intent = slots["intent"]
    present = slots.get("present")
    found = None
    if intent in SETS and present is not None and split in ("train", "wording", "categories"):
        key, role = SETS[intent]
        target = slots[role]
        options = category_entries(target, present, split) if len(target) >= 2 else []
        if options and (split == "categories" or rng.random() < category):
            found = (key, target, pick(rng, options)[1])
    return found


def render(text, canonical_slots, rng, split="train", keep=KEEP, category=CATEGORY):
    """A Rewrite of a suite instruction: the reworded instruction, the canonical sentence, the
    template used and the replacements made. text is the suite's instruction: it is only checked
    against the slots (check_text), never parsed."""
    if split not in SPLITS:
        raise ValueError(split)
    slots = canonical_slots
    check_text(text, slots)
    intent = slots["intent"]
    present = slots.get("present")
    choices = []
    whole = set_category(rng, slots, split, category)
    if whole is not None:
        key, target, forms = whole
        template = pick(rng, templates(key, split))
        word = pick(rng, fitting(forms, template, "k") or [f for f, _ in forms])
        choices.append(Choice("k", list(target), word, "category"))
        instruction = template.format(k=word)
    elif intent == "split":
        words = {}
        for name, (kind, x) in zip("ab", slots["jobs"]):
            words[name] = fill(rng, SUB[kind], {"x": x}, split, present, keep, choices, prefix=name + ".")[1]
        template = pick(rng, templates(intent, split))
        instruction = template.format(**words)
    elif intent == "subset":
        words = {"who": fill_part(rng, slots["who"], split, present, keep, choices, "who."), "job": fill_part(rng, slots["job"], split, present, keep, choices, "job.")}
        template = pick(rng, templates(intent, split))
        instruction = template.format(**words)
    else:
        template, instruction = fill(rng, intent, {name: slots[name] for name in INTENTS[intent]["slots"]}, split, present, keep, choices)
    return Rewrite(instruction, canonical(slots), template, choices)


def rewrite(text, canonical_slots, rng, split="train"):
    """(reworded instruction, plain canonical instruction) of a suite instruction and its
    canonical slots (slots_of). The second is what the policy learns to write as its restatement."""
    made = render(text, canonical_slots, rng, split)
    return made.instruction, made.canonical


# ---------------------------------------------------------------------------------------------
# Held-out sets and the token check


def all_forms(kind, value, split):
    return [text for _, forms in entries(kind, value, split) for text, _ in forms]


def slot_expressions():
    """{"train": [(kind, value, text)], "held_out": [...]}: every slot expression by the side of
    the split it is on. Canonical words count as training, except those of values the suite holds
    out (pink, pyramid), whose every expression is held out. Letter expressions are listed by
    their frame alone. The "frames" split is not here: its words are training words by design."""
    out = {"train": [], "held_out": []}
    for kind in ("colour", "count", "region", "shape"):
        for value, entry in SLOTS[kind]["values"].items():
            value = int(value) if kind == "count" else value
            if not suite_held(entry):
                out["train"].append((kind, value, plain(kind, value)))
            out["train"] += [(kind, value, text) for text in all_forms(kind, value, "train")]
            out["held_out"] += [(kind, value, text) for split in ("words", "languages") for text in all_forms(kind, value, split)]
    for kind, entry in LEXICAL.items():
        out["train"] += [(kind, kind, text) for text in [entry["canonical"]] + entry["train"]]
        out["held_out"] += [(kind, kind, text) for text in entry["held_out"]]
    for side in out:
        out[side] += [("letter", "", frame.replace("{w}", "").strip()) for frame in SLOTS["letter"][side]]
    for cat in CATEGORIES:
        out[cat["split"]] += [("category", cat["name"], text) for text in cat["adjective"] + cat["predicate"]]
    return out


def held_out_sets():
    """What is fixed as held out before any training, by kind: words per slot value, languages,
    categories, frames, templates per intent, and the suite's own held-out items and compositions."""
    sides = slot_expressions()
    words = {}
    for kind, value, text in sides["held_out"]:
        if kind != "category":
            words.setdefault(kind, {}).setdefault(str(value), []).append(text)
    return {
        "words": words,
        "languages": LANGUAGES["held_out"],
        "categories": {cat["name"]: cat["set"] for cat in CATEGORIES if cat["split"] == "held_out"},
        "frames": SLOTS["colour"]["frames"]["phrase"]["held_out"],
        "wording": {intent: spec["held_out"] for intent, spec in INTENTS.items()},
        "suite": {key: list(value) for key, value in S.HELD_OUT.items()},
    }


def tokens_of(tokenizer, text):
    """The tokens of an expression as the policy reads it inside an instruction: lowercase, after a space."""
    return [tokenizer.decode([i]) for i in tokenizer(" " + text.lower(), add_special_tokens=False).input_ids]


def token_clashes(tokenizer):
    """[(held-out expression, shared tokens, training expressions that have them)]: held-out slot
    expressions that share a token outside SHARED_TOKENS with a training slot expression of any
    slot and value. Empty when the split is clean."""
    sides = slot_expressions()
    seen = {}
    for _, _, text in sides["train"]:
        for token in tokens_of(tokenizer, text):
            seen.setdefault(token, set()).add(text)
    clashes = []
    for _, _, text in sides["held_out"]:
        hit = sorted({t for t in tokens_of(tokenizer, text) if t in seen and t not in SHARED_TOKENS})
        if hit:
            clashes.append((text, hit, sorted({source for t in hit for source in seen[t]})))
    return clashes


def name_leaks():
    """[(held-out expression, trained word)]: held-out expressions in which a word starts with a
    trained single-word expression ("blueberry" starts with "blue"), which a tokenizer may or may
    not split there; and held-out expressions that occur whole in a training template or a
    canonical sentence."""
    sides = slot_expressions()
    names = {text for _, _, text in sides["train"] if re.fullmatch(r"[a-z]{3,}", text)}
    allowed = {token.strip() for token in SHARED_TOKENS}
    leaks = []
    for _, _, text in sides["held_out"]:
        for word in re.findall(r"[a-z]+", text.lower()):
            leaks += [(text, name) for name in names if word.startswith(name) and word not in allowed and name not in allowed]
    sentences = [t for spec in INTENTS.values() for t in spec["train"] + [spec["canonical"]]]
    for kind, _, text in sides["held_out"]:
        if kind != "letter":
            leaks += [(text, sentence) for sentence in sentences if re.search(r"(?<!\w)" + re.escape(text.lower()) + r"(?!\w)", sentence.lower())]
    return leaks


def denotes():
    """{(kind, expression): canonical value} over every split, for checking a rewrite and for
    explaining it ("warm means red"). Raises ValueError if an expression has two values."""
    index = {}
    for kind in ("colour", "count", "region", "shape"):
        for value in SLOTS[kind]["values"]:
            value = int(value) if kind == "count" else value
            texts = [plain(kind, value)] + [text for split in ("train", "words", "languages", "frames") for text in all_forms(kind, value, split)]
            for text in texts:
                if index.setdefault((kind, text), value) != value:
                    raise ValueError(f"{text!r} is both {index[(kind, text)]} and {value}")
    return index
