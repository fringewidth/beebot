"""The voxel wording bank (vla/voxel/wordings.py, wordings.json): templates format and keep the
suite's placeholders, training and held-out lists are disjoint down to the token, and a rewrite
keeps a task's canonical values."""

import random
import re

import numpy as np
import pytest

from beebot.vla import tasks as T2
from beebot.vla.voxel import suite as S
from beebot.vla.voxel import wordings as W

SUITE_KEYS = {**{key: key for key in S.WORDINGS}, **{key: key for key in set(S.LOGIC_KEYS.values())}}
EXTRA = set(W.LEXICAL)  # placeholders a template may add: lexical words, filled from the lexicon


def original(key):
    return S.WORDINGS[key][0] if key in S.WORDINGS else T2.WORDINGS[key][0]


def all_templates(spec):
    return spec["train"] + spec["held_out"]


@pytest.mark.parametrize("key", sorted(SUITE_KEYS))
def test_every_suite_intent_has_forty_templates_with_the_original_placeholders(key):
    spec = W.INTENTS[key]
    assert len(spec["train"]) + len(spec["held_out"]) >= 40
    assert len(spec["held_out"]) == 6
    fields = W.fields_of(original(key))
    assert sorted(spec["slots"]) == fields
    assert W.fields_of(spec["canonical"]) == fields
    for template in all_templates(spec):
        assert sorted(set(W.fields_of(template)) - EXTRA) == fields, template
        assert all(template.count("{" + name + "}") == 1 for name in fields), template


def test_templates_format_and_are_distinct_and_the_splits_are_disjoint():
    for key, spec in W.INTENTS.items():
        every = all_templates(spec)
        assert len({t.lower() for t in every}) == len(every), key
        assert not set(spec["train"]) & set(spec["held_out"]), key
        for template in every:
            filled = template.format(**{name: "X" for name in W.fields_of(template)})
            assert "{" not in filled and "}" not in filled
            assert set(W.fields_of(template)) <= set(spec["slots"]) | EXTRA, template
            assert template == template.strip() and "  " not in template


def test_articles_and_frames_suit_every_expression():
    """Rule 3 of the module: nothing but a shape follows "a" or "an"; shapes never follow "an" and
    start with a consonant sound; regions and lexical words follow "the" or "that"; every colour
    intent has predicative templates on both sides so phrases have somewhere to go."""
    for key, spec in W.INTENTS.items():
        for template in all_templates(spec):
            for name in W.fields_of(template):
                before = template.split("{" + name + "}")[0]
                kind = spec["slots"].get(name, name)
                if kind == "shape":
                    assert before.endswith("a "), template
                else:
                    assert not re.search(r"\b(a|an) $", before), template
                if kind in ("region", "trench", "plateau"):
                    assert re.search(r"\b(the|that|two|one) $", before), template
        colours = [name for name, kind in spec["slots"].items() if kind in ("colour", "category")]
        if colours:
            for side in ("train", "held_out"):
                assert any(t.split("{" + name + "}")[0].endswith(W.PREDICATIVE) for t in spec[side] for name in colours), (key, side)
    for entry in W.SLOTS["shape"]["values"].values():
        for text in [entry["canonical"]] + entry["train"] + entry["held_out"]:
            assert text[0].lower() not in "aeiou", text


def test_training_templates_are_not_held_out_by_the_2d_pool():
    for key in set(S.LOGIC_KEYS.values()):
        assert not set(W.INTENTS[key]["train"]) & set(T2.wordings(key, "wording")) - set(T2.WORDINGS[key]), key


def test_lexicon_lists_are_disjoint_and_every_expression_has_one_value():
    sides = W.slot_expressions()
    train = {(kind, text.lower()) for kind, _, text in sides["train"]}
    held_out = {(kind, text.lower()) for kind, _, text in sides["held_out"]}
    assert not train & held_out
    assert not {text for _, text in train} & {text for _, text in held_out}
    W.denotes()  # raises if an expression maps to two canonical values
    for kind, value, text in sides["held_out"]:
        assert kind != "colour" or value == "pink" or text != "pink"
    assert not [text for kind, value, text in sides["train"] if value in ("pink", "pyramid")]
    assert set(W.SLOTS["colour"]["values"]) == set(S.COLOUR_NAMES)
    assert set(W.SLOTS["region"]["values"]) == set(S.FLOOR_REGIONS)
    assert {name: entry["canonical"] for name, entry in W.SLOTS["shape"]["values"].items()} == S.SHAPE_WORDS


def test_categories_are_sets_of_colours_with_reasons():
    names = set(S.COLOUR_NAMES)
    assert len(W.CATEGORIES) >= 15
    assert sum(cat["split"] == "held_out" for cat in W.CATEGORIES) >= 5
    for cat in W.CATEGORIES:
        assert cat["split"] in ("train", "held_out")
        assert len(set(cat["set"])) == len(cat["set"]) >= 2 and set(cat["set"]) <= names
        assert set(cat["unclear"]) <= names - set(cat["set"])
        assert len(cat["why"]) > 20 and cat["predicate"]
    assert "pink" not in {c for cat in W.CATEGORIES if cat["split"] == "train" for c in cat["set"]}


def test_no_held_out_expression_shares_a_token_with_training():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except OSError:
        pytest.skip("the Qwen tokenizer is not in the local cache")
    assert W.tokens_of(tokenizer, "the colour of grass") == [" the", " colour", " of", " grass"]
    assert W.token_clashes(tokenizer) == []
    used = {token for side in W.slot_expressions().values() for _, _, text in side for token in W.tokens_of(tokenizer, text)}
    assert set(W.SHARED_TOKENS) <= used  # the allowance lists nothing that is not needed


def test_no_held_out_expression_leaks_into_training_text():
    assert W.name_leaks() == []


def sample(n, seed, split="train", robots=None):
    rng = np.random.default_rng(seed)
    world, _, tasks = S.sample_tasks(rng, n, split, robots=robots)
    return [(task, [c for _, c in S.present(world, i)]) for i, task in enumerate(tasks)]


def same(parsed, slots):
    """Whether parsed canonical slots carry a task's canonical values (intents up to ALIASES)."""
    keys = [k for k in slots if k not in ("present", "eat", "forbid", "any_of", "intent")]
    return W.ALIASES.get(slots["intent"], slots["intent"]) == parsed["intent"] and all(parsed[k] == (list(map(tuple, slots[k])) if k == "jobs" else slots[k]) for k in keys)


def test_slots_agree_with_the_expert_labels():
    for task, present in sample(120, 3):
        slots = W.slots_of(task, present)
        label = task.plan[0].get("label", {}) if task.plan else {}
        if label.get("kind") == "letters":
            assert slots == {"intent": "letter_stand" if label["standing"] else "letter_floor", "w": label["word"], "present": present}
        if label.get("kind") == "shape":
            assert slots["s"] == label["shape"]
        if task.spec.get("form") == "absent":
            assert slots["a"] not in present and slots["a"] in task.text


def test_rewrite_round_trips_canonical_values():
    index = W.denotes()
    kinds = {"colour", "count", "region", "shape"}
    count = 0
    cases = [(case, "train") for case in sample(240, 4)] + [(case, "train") for case in sample(60, 5, robots=S.ROBOTS)]
    cases += [(case, held) for held in ("colour", "letters", "shapes", "logic", "pairs", "construction", "herd_two", "runner_logic") for case in sample(6, 6, held)]
    rng = np.random.default_rng(7)
    for (task, present), _ in cases:
        slots = W.slots_of(task, present)
        for split in W.SPLITS:
            made = W.render(task.text, slots, rng, split)
            assert W.rewrite(task.text, slots, random.Random(count), split)[1] == made.canonical
            parsed = W.parse_canonical(made.canonical)
            assert same(parsed, slots), (made.canonical, slots)
            assert W.canonical(parsed) == made.canonical
            assert "{" not in made.instruction and made.instruction.strip() == made.instruction
            for choice in made.choices:
                assert choice.text in made.instruction
                kind = "colour" if choice.source in ("thing", "frame") else None
                if choice.source == "category":
                    target = choice.value if isinstance(choice.value, list) else [choice.value]
                    cat = next(c for c in W.CATEGORIES if choice.text in c["adjective"] + c["predicate"])
                    assert set(cat["set"]) & set(present) == set(target) and not set(cat["unclear"]) & set(present)
                    assert cat["split"] == ("held_out" if split == "categories" else "train")
                elif choice.source in ("synonym", "language", "thing", "frame"):
                    found = [index[(k, choice.text)] for k in kinds if (k, choice.text) in index]
                    assert kind is None or found
                    assert not found or found == [choice.value], (choice, found)
            if split == "wording":
                assert made.template in W.INTENTS[slots["intent"]]["held_out"] + [t for key in ("set.eat", "set.except", "set.or") for t in W.INTENTS[key]["held_out"]]
            else:
                assert not any(made.template in spec["held_out"] for spec in W.INTENTS.values())
            count += 1
    assert count >= 200 * len(W.SPLITS)


def test_training_rewrites_never_use_held_out_material():
    sides = W.slot_expressions()
    banned = sorted({text for _, _, text in sides["held_out"]} - {"pink", "pyramid"}, key=len, reverse=True)
    rng = np.random.default_rng(8)
    seen = set()
    for task, present in sample(150, 9):
        slots = W.slots_of(task, present)
        for _ in range(4):
            made = W.render(task.text, slots, rng, "train")
            seen.update(choice.source for choice in made.choices)
            assert not [text for text in banned if re.search(r"(?<![\w-])" + re.escape(text) + r"(?![\w-])", made.instruction)], made.instruction
    assert {"synonym", "thing", "language", "category"} <= seen


def test_each_held_out_split_reaches_its_material():
    rng = np.random.default_rng(10)
    reached = {split: set() for split in W.SPLITS}
    for task, present in sample(200, 11):
        slots = W.slots_of(task, present)
        for split in W.SPLITS:
            reached[split].update(choice.source for choice in W.render(task.text, slots, rng, split).choices)
    assert reached["wording"] <= {"category"}
    assert reached["words"] == {"synonym", "thing"}
    assert reached["languages"] == {"language"}
    assert reached["categories"] == {"category"}
    assert reached["frames"] == {"frame"}


def test_a_wrong_slot_is_refused():
    task, present = sample(1, 12)[0]
    slots = W.slots_of(task, present)
    with pytest.raises(ValueError):
        W.render("something else entirely", {"intent": "tower", "n": 5}, np.random.default_rng(0))
    with pytest.raises(ValueError):
        W.parse_canonical(task.text + " and then some")
    assert W.canonical(slots) == W.rewrite(task.text, slots, np.random.default_rng(0))[1]
