"""Think-then-act training examples (vla/voxel/think_data.py): thought steps from the stored
sub-goal record, memory of earlier thoughts, instruction changes, Haiku labels joined by
(seed, index) with templates otherwise, refusals, sampling shares, the held-out split and the
token layout. Most tests use a small synthetic store; one writes a tiny real store."""

import json
import re

import numpy as np
import pytest
import torch

from beebot.vla.voxel import demos as D
from beebot.vla.voxel import suite as S
from beebot.vla.voxel import think_data as T
from beebot.vla.voxel.lm import world_at
from beebot.vla.voxel.play_store import CompactWriter, load
from beebot.vla.voxel.world import COLOUR_NAMES, STAY, failure_rates, observe

STEPS = 12
SEED = 3
OPTIONS = {"robots": [6, 115], "counts": "log", "rule": "switch_cube", "apart": False, "coverage": 0.3, "objects": 1.0}


def record(family, kind, plain, spec, subgoals):
    return {"family": family, "kind": kind, "plain": plain, "spec": spec, "subgoals": subgoals}


EAT_TWO = record("logic", "eat:ordered", "eat the red cheese, then the blue cheese", {"kind": "eat", "eat": ["blue", "red"], "any_of": [], "order": ["red", "blue"]}, ["eat the red cheese", "eat the blue cheese"])
GOTO = record("switch", "goto", "go to the front of the floor", {"kind": "goto", "region": "front"}, ["go to the front of the floor"])
STAY_RECORD = record("switch", "stay", "stay where you are", {"kind": "stay"}, ["stay where you are"])
WALL = record("formation", "mask", "form a wall", {"kind": "shape", "shape": "wall"}, ["form a wall"])
ABSENT = record("logic", "eat:absent", "eat the green cheese", {"kind": "refuse"}, [])

# (family, attempts kind, instructions, sub-goal entries) of the four episode patterns
PATTERNS = [
    ("logic", "eat:ordered", [{"t": 0, "text": "eat red, then blue", "record": EAT_TWO}], [{"t": 0, "index": 0, "goal": {"kind": "eat", "colour": "red"}}, {"t": 2, "index": 0, "goal": {"kind": "eat", "colour": "red", "note": "same segment"}}, {"t": 4, "index": 1, "goal": {"kind": "eat", "colour": "blue"}}, {"t": 9, "index": 2, "goal": {"kind": "done"}}]),
    ("switch", "switch:goto>stay", [{"t": 0, "text": "go to the front", "record": GOTO}, {"t": 6, "text": "stay where you are", "record": STAY_RECORD}], [{"t": 0, "index": 0, "goal": {"kind": "goto"}}, {"t": 6, "index": 0, "goal": {"kind": "stay"}}]),
    ("formation", "mask", [{"t": 0, "text": "build a wall", "record": WALL}], [{"t": 0, "index": 0, "goal": {"kind": "shape", "shape": "wall"}}]),
    ("logic", "eat:absent", [{"t": 0, "text": "eat the green cheese", "record": ABSENT}], [{"t": 0, "index": -1, "goal": {"kind": "stay"}}]),
]
EPISODES = 40  # pattern e % 4; attempts hold a dropped row before every fifth kept one


def synthetic_store(path, seed=SEED, episodes=EPISODES, options=OPTIONS):
    """A compact demo store with hand-made instructions and sub-goals over random robots."""
    rng = np.random.default_rng(seed)
    writer = CompactWriter(path, (16, 16, 16))
    small = {"cheese_pos": [], "cheese_alive": [], "walls": [], "cheese_color": [], "gravity": []}
    instructions, subgoals, attempts = [], [], []
    for e in range(episodes):
        family, kind, entries, goals = PATTERNS[e % len(PATTERNS)]
        mass = np.zeros((STEPS + 1, 16, 16, 16), np.uint8)
        cells = rng.choice(16 * 16, size=8, replace=False)
        for t in range(STEPS + 1):
            mass[t, 0].reshape(-1)[cells] = 1
        actions = np.full((STEPS, 16, 16, 16), STAY, np.int8)
        for t in range(STEPS):
            actions[t][mass[t] > 0] = rng.integers(0, 26, size=int(mass[t].sum()))
        writer.add(mass, actions)
        pos = torch.zeros(STEPS + 1, 3, 3, dtype=torch.long)
        pos[:, 0] = torch.tensor([0, 2, 2])
        pos[:, 1] = torch.tensor([0, 12, 12])
        small["cheese_pos"].append(pos)
        small["cheese_alive"].append(torch.tensor([[True, True, False]] * (STEPS + 1)))
        small["walls"].append(torch.zeros(16, 16, 16, dtype=torch.bool))
        small["cheese_color"].append(torch.tensor([COLOUR_NAMES.index("red"), COLOUR_NAMES.index("blue"), -1]))
        small["gravity"].append(torch.tensor(True))
        instructions.append(entries)
        subgoals.append(goals)
        if e % 5 == 0:
            attempts.append({"family": "herd", "kind": "herd_region", "robots": 8, "score": 0.1, "unasked": False, "e": -1})
        attempts.append({"family": family, "kind": kind, "robots": 8, "score": 1.0, "unasked": False, "e": e})
    writer.close()
    config = {"seed": seed, "steps": STEPS, **options}
    torch.save({**{k: torch.stack(v) for k, v in small.items()}, "phases": [[] for _ in range(episodes)], "count": episodes, "robots": torch.full((episodes,), 8), "subgoals": subgoals, "instructions": instructions, "config": config}, path / "small.pt")
    (path / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in attempts))
    return path


def index_of(store_dir, e):
    """The attempts.jsonl line (the draw index) of store row e."""
    rows = [json.loads(line) for line in (store_dir / "attempts.jsonl").read_text().splitlines()]
    return next(n for n, row in enumerate(rows) if row["e"] == e)


def label_dir(path, rows, seed=SEED, options=OPTIONS):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"seed": seed, "chunk": 24, **options}))
    (path / "verified.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


SPECS = {r["kind"]: r["spec"] for r in (EAT_TWO, GOTO, STAY_RECORD, WALL)} | {"refuse": {"kind": "refuse"}}


def pair(seed, index, family, kind, rung, instruction, thought, switches=(), **extra):
    """A verified label row as haiku.verify writes it (spec: the item's record spec)."""
    return {"rung": rung, "instruction": instruction, "thought": thought, "switches": list(switches), "done": "done.", "pair_id": f"s{seed}-{index}:{rung}", "seed": seed, "index": index, "family": family, "kind": kind, "spec": SPECS[kind], **extra}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return synthetic_store(tmp_path_factory.mktemp("think") / "store")


@pytest.fixture(scope="module")
def labels(store, tmp_path_factory):
    """Haiku labels for episode 0 (eat two, two rungs), 1 (switch, both parts) and 2 (wall, with a
    refusal); every other episode stays on its template."""
    n0, n1, n2 = (index_of(store, e) for e in (0, 1, 2))
    rows = [
        pair(SEED, n0, "logic", "eat:ordered", "close", "eat the red one and then the blue one", "you asked: eat the red one and then the blue one. i see red and blue cheeses. red first. first: eat the red cheese.", ["now the blue one"]),
        pair(SEED, n0, "logic", "eat:ordered", "casual", "red then blue pls", "you asked: red then blue pls. i see red and blue cheeses. red first. first: eat the red cheese.", ["now the blue one"]),
        pair(SEED, n1, "switch", "goto", "close", "head to the front", "you asked: head to the front. i see a flat floor. first: go to the front of the floor.", part=0, parts=2, t=0),
        pair(SEED, n1, "switch", "stay", "indirect", "could you all freeze", "you asked: could you all freeze. nobody moves. first: stay where you are.", part=1, parts=2, t=6),
        pair(SEED, n2, "formation", "mask", "close", "make a wall", "you asked: make a wall. i see robots. first: form a wall."),
        pair(SEED, n2, "formation", "refuse", "refusal", "make the blue cheese fly", "you asked: make the blue cheese fly. gravity is on, so i will not act.", reply="the blue cheese cannot fly, so i will do nothing."),
    ]
    return label_dir(tmp_path_factory.mktemp("haiku") / "run-test", rows)


def world_views(obs, rng):
    """The stand-in's images without a pose: the camera counts as the default viewer's (azimuth 0),
    so a user behind it speaks in the world's frame."""
    return T.stand_in_views(obs)[0]


@pytest.fixture(scope="module")
def data(store, labels):
    return T.ThinkData([store], [labels], held_share=0.0, views=world_views)


def ep_of(data, e):
    return next(ep for ep in data.episodes if ep.e == e)


@pytest.fixture(scope="module")
def tokenizer():
    try:
        import transformers

        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    except Exception:
        pytest.skip("the Qwen3.5 tokenizer is not cached")


# ---------------------------------------------------------------------------------------------
# Events and memory


def test_events_come_from_segment_changes_instruction_changes_and_done():
    events = T.episode_events(PATTERNS[0][2], PATTERNS[0][3], STEPS)
    assert [(ev.t, ev.kind, ev.part, ev.index) for ev in events] == [(0, "first", 0, 0), (4, "switch", 0, 1), (9, "done", 0, 2)]  # t 2 changes the goal, not the segment
    events = T.episode_events(PATTERNS[1][2], PATTERNS[1][3], STEPS)
    assert [(ev.t, ev.kind, ev.part) for ev in events] == [(0, "first", 0), (6, "change", 1)]
    skipping = [{"t": 0, "index": 0, "goal": {}}, {"t": 3, "index": 2, "goal": {}}, {"t": 5, "index": 3, "goal": {}}]
    assert [(ev.t, ev.index) for ev in T.episode_events([{"t": 0}], skipping, STEPS)] == [(0, 0), (3, 2), (5, 3)]
    assert [ev.t for ev in T.episode_events([{"t": 0}, {"t": 20}], skipping + [{"t": 20, "index": 0, "goal": {}}], STEPS)] == [0, 3, 5]  # past the end


def test_memory_holds_the_latest_first_thought_and_the_latest_line_before_t():
    events = T.episode_events(PATTERNS[0][2], PATTERNS[0][3], STEPS)
    assert T.memory_before(events, 0) == []
    assert [ev.t for ev in T.memory_before(events, 1)] == [0]
    assert [ev.t for ev in T.memory_before(events, 4)] == [0]
    assert [ev.t for ev in T.memory_before(events, 5)] == [0, 4]
    assert [ev.t for ev in T.memory_before(events, 11)] == [0, 9]
    events = T.episode_events(PATTERNS[1][2], PATTERNS[1][3], STEPS)
    assert [ev.t for ev in T.memory_before(events, 6)] == [0]  # at the change: the old instruction's thought
    assert [ev.t for ev in T.memory_before(events, 7)] == [6]


# ---------------------------------------------------------------------------------------------
# Examples


def test_examples_at_each_kind_of_step(data, store):
    ep = ep_of(data, 4)  # eat two, template
    parts = data.labelling(ep, np.random.default_rng(0))
    assert parts[0]["rung"] == "template" and parts[0]["instruction"] == "eat red, then blue"
    first = data.example(ep, 0, parts)
    assert first.kind == "first" and first.act and first.rung == "template" and first.family == "logic"
    assert first.context_text == "<|im_start|>user\n<IMAGES>\n(i'm standing behind the camera)\neat red, then blue<|im_end|>\n<|im_start|>assistant\n"
    assert re.fullmatch(r"<think>\nyou asked: eat red, then blue\. i see .+\. that means: .+\. first: eat the red cheese\.\n</think>\n\n<tool_call>", first.reply_text)
    plain = data.example(ep, 2, parts)
    assert plain.kind == "plain" and plain.reply_text == "<think>\n\n</think>\n\n<tool_call>"
    thought = first.reply_text.split("\n")[1]
    assert plain.context_text == f"<|im_start|>assistant\n{thought}<|im_end|>\n" + first.context_text
    switch = data.example(ep, 4, parts)
    assert switch.kind == "switch" and switch.reply_text == "<think>\neat the blue cheese next.\n</think>\n\n<tool_call>"
    after = data.example(ep, 6, parts)
    assert after.context_text.startswith(f"<|im_start|>assistant\n{thought}\neat the blue cheese next.<|im_end|>\n")
    done = data.example(ep, 9, parts)
    assert done.kind == "done" and "<think>\ndone.\n</think>" in done.reply_text
    assert data.example(ep, 11, parts).context_text.startswith(f"<|im_start|>assistant\n{thought}\ndone.<|im_end|>\n")
    raw = load(store)
    assert torch.equal(switch.codes, raw["actions"][4, 4].long()) and switch.codes.dtype == torch.long
    assert torch.equal(switch.obs, observe(world_at(raw, [(4, 4)]))[0]) and switch.obs.shape == (10, 16, 16, 16)
    for example in (first, plain, switch, done):
        assert example.context_text.count(T.IMAGES) == 1 and len(example.images) == 1 and example.images[0].size == (272, 272)


def test_an_instruction_change_puts_the_new_instruction_in_the_user_turn(data):
    ep = ep_of(data, 5)  # switch, template
    parts = data.labelling(ep, np.random.default_rng(0))
    before = data.example(ep, 5, parts)
    change = data.example(ep, 6, parts)
    later = data.example(ep, 8, parts)
    assert "<IMAGES>\n(i'm standing behind the camera)\ngo to the front<|im_end|>" in before.context_text
    assert change.kind == "change" and "<IMAGES>\n(i'm standing behind the camera)\nstay where you are<|im_end|>" in change.context_text
    assert change.reply_text.startswith("<think>\nyou asked: stay where you are. i see") and "first: stay where you are." in change.reply_text
    assert "you asked: go to the front." in change.context_text  # memory: the old instruction's thought
    assert "you asked: stay where you are." in later.context_text and "you asked: go to the front." not in later.context_text
    assert "<IMAGES>\n(i'm standing behind the camera)\nstay where you are<|im_end|>" in later.context_text and later.kind == "plain"


def test_haiku_labels_replace_the_template_and_rungs_are_drawn(data):
    ep = ep_of(data, 0)
    assert ep.labelled and len(ep.choices) == 2
    rungs = set()
    for seed in range(12):
        parts = data.labelling(ep, np.random.default_rng(seed))
        rungs.add(parts[0]["rung"])
        first = data.example(ep, 0, parts)
        assert first.reply_text == f"<think>\n{parts[0]['thought']}\n</think>\n\n<tool_call>"
        assert f"<IMAGES>\n(i'm standing behind the camera)\n{parts[0]['instruction']}<|im_end|>" in first.context_text and first.rung == parts[0]["rung"]
        assert data.example(ep, 4, parts).reply_text == "<think>\nnow the blue one\n</think>\n\n<tool_call>"
        assert parts[0]["thought"] in data.example(ep, 7, parts).context_text
    assert rungs == {"close", "casual"}


def test_haiku_switch_parts_label_each_instruction(data):
    ep = ep_of(data, 1)
    assert ep.parts and not ep.choices
    parts = data.labelling(ep, np.random.default_rng(0))
    assert [p["instruction"] for p in parts] == ["head to the front", "could you all freeze"]
    change = data.example(ep, 6, parts)
    assert "<IMAGES>\n(i'm standing behind the camera)\ncould you all freeze<|im_end|>" in change.context_text and change.rung == "indirect"
    assert change.reply_text.startswith("<think>\nyou asked: could you all freeze.")
    assert "you asked: head to the front." in change.context_text
    assert data.example(ep, 3, parts).rung == "close"


def test_refusals_that_refuse_act_with_every_robot_staying_and_clarifications_speak(data, store, tokenizer):
    """A request the scene cannot satisfy is met by doing nothing, an action like any other (every
    robot STAY) after the thought; only a clarification is a spoken question."""
    ep = ep_of(data, 2)
    example = data.refusal(ep, ep.refusals[0])
    assert example.act and example.t == 0 and example.kind == "refusal" and example.rung == "refusal"
    assert example.reply_text == "<think>\nyou asked: make the blue cheese fly. gravity is on, so i will not act.\n</think>\n\n" + T.ACT
    assert example.codes.shape == example.obs.shape[-3:] and (example.codes == STAY).all()
    assert "<IMAGES>\n(i'm standing behind the camera)\nmake the blue cheese fly<|im_end|>" in example.context_text
    asking = data.refusal(ep, dict(ep.refusals[0], kind="clarify", reply="which cheese?"))
    assert not asking.act and asking.codes is None and asking.reply_text.endswith("which cheese?<|im_end|>") and T.ACT not in asking.reply_text
    assert torch.equal(example.obs, observe(world_at(load(store), [(2, 0)]))[0])
    assert (T.still(example.obs) == STAY).all()
    assert np.array_equal(np.asarray(example.images[0]), np.asarray(T.stand_in_views(example.obs)[0][0]))
    ids, labels = T.tokenize(example, tokenizer)
    assert ids[-1].item() == tokenizer.convert_tokens_to_ids(T.ACT) and labels[-1].item() == ids[-1].item()


def test_an_absent_target_thinks_that_nothing_matches_and_acts(data):
    ep = ep_of(data, 3)
    first = data.example(ep, 0, data.labelling(ep, np.random.default_rng(0)))
    assert first.act and "nothing here matches it, so there is nothing to do. first: stay where you are." in first.reply_text


# ---------------------------------------------------------------------------------------------
# Joining labels


def test_the_join_refuses_a_wrong_draw_a_wrong_family_and_the_test_set(store, tmp_path):
    n0 = index_of(store, 0)
    good = pair(SEED, n0, "logic", "eat:ordered", "close", "eat red", "you asked: eat red. first: eat the red cheese.")
    with pytest.raises(ValueError, match="objects"):
        T.ThinkData([store], [label_dir(tmp_path / "other-draw", [good], options={**OPTIONS, "objects": 0.3})])
    with pytest.raises(ValueError, match="join is broken"):
        T.ThinkData([store], [label_dir(tmp_path / "shifted", [dict(good, index=n0 + 1)])])
    with pytest.raises(ValueError, match="spec"):  # same family and kind, another task: the record differs
        T.ThinkData([store], [label_dir(tmp_path / "other-spec", [dict(good, spec=dict(good["spec"], order=["blue", "red"]))])])
    with pytest.raises(ValueError, match="kind"):
        T.ThinkData([store], [label_dir(tmp_path / "other-kind", [dict(good, kind="eat:and")])])
    switch_part = pair(SEED, index_of(store, 1), "switch", "stay", "close", "freeze", "you asked: freeze. first: stay where you are.", part=0, parts=2, t=0)
    with pytest.raises(ValueError, match="spec"):  # part 0 of that episode is the goto, not the stay
        T.ThinkData([store], [label_dir(tmp_path / "wrong-part", [switch_part])])
    data = T.ThinkData([store], [label_dir(tmp_path / "other-seed", [dict(good, seed=SEED + 1)], seed=SEED + 1)])
    assert not any(ep.labelled for ep in data.episodes)  # labels of another seed join nothing
    with pytest.raises(AssertionError, match="test set"):
        T.ThinkData([store], [label_dir(tmp_path / "test", [good])])


def test_the_exported_item_must_match_the_attempt_and_drop_mode_falls_back_to_templates(store, tmp_path):
    n0 = index_of(store, 0)
    good = pair(SEED, n0, "logic", "eat:ordered", "close", "eat red", "you asked: eat red. first: eat the red cheese.")

    def with_item(name, robots):
        directory = label_dir(tmp_path / name, [good])
        item = {"id": f"s{SEED}-{n0}", "seed": SEED, "index": n0, "e": None, "record": EAT_TWO, "scene": {"robots": {"count": robots}}}
        (directory / "items.jsonl").write_text(json.dumps(item) + "\n")
        return directory

    assert ep_of(T.ThinkData([store], [with_item("same-scene", 8)], held_share=0.0), 0).labelled
    with pytest.raises(ValueError, match="30 robots"):  # same task, another scene: a draw that slipped
        T.ThinkData([store], [with_item("other-scene", 30)])
    data = T.ThinkData([store], [with_item("other-scene-drop", 30)], held_share=0.0, mismatch="drop")
    assert not ep_of(data, 0).labelled and data.counts()["mismatched_episodes"] == 1
    assert data.mismatches[0][1] == n0 and "30 robots" in data.mismatches[0][2]


def test_index_counts_dropped_attempts(data, store):
    for ep in data.episodes:
        assert ep.index == index_of(store, ep.e) and ep.seed == SEED
    assert data.episodes[1].index > 1  # dropped rows before it


def test_counts_split_haiku_template_and_refusals(data):
    counts = data.counts()
    assert counts["episodes"] == EPISODES and counts["haiku"] == 2 and counts["haiku_switch"] == 1
    assert counts["template"] == EPISODES - 3 and counts["refusal_pairs"] == 1 and counts["with_refusals"] == 1
    assert counts["families"]["switch"] == {"episodes": 10, "haiku": 1, "template": 9}


# ---------------------------------------------------------------------------------------------
# Sampling


def test_sampling_shares_and_the_held_out_split(store, labels):
    data = T.ThinkData([store], [labels], held_share=0.25, refusal_share=0.1, thought_share=1 / 3)
    held = {(ep.seed, ep.index) for ep in data.held}
    stored = load(store)
    assert held and len(held) < EPISODES and held == {(ep.seed, ep.index) for ep in data.episodes if T.held_out_scene(T.scene_key(stored, ep.e), 0.25)}
    scenes = {}
    for ep in data.episodes:  # every episode of a start scene is on the same side
        scenes.setdefault(T.scene_key(stored, ep.e), set()).add(ep.held)
    assert all(len(sides) == 1 for sides in scenes.values())
    rng = np.random.default_rng(0)
    drawn = [data.sample(rng) for _ in range(1500)]
    assert not {(ex.seed, ex.index) for ex in drawn} & held
    refusals = sum(ex.kind == "refusal" for ex in drawn) / len(drawn)
    acting = [ex for ex in drawn if ex.act and ex.kind != "refusal"]
    thinking = sum(ex.kind != "plain" for ex in acting) / len(acting)
    assert 0.07 < refusals < 0.13 and 0.29 < thinking < 0.38
    assert all((ex.kind == "plain") == ("<think>\n\n</think>" in ex.reply_text) for ex in acting)
    first = data.validation(30, seed=1)
    assert [(ex.e, ex.t, ex.rung) for ex in first] == [(ex.e, ex.t, ex.rung) for ex in T.ThinkData([store], [labels], held_share=0.25, refusal_share=0.1).validation(30, seed=1)]
    assert {(ex.seed, ex.index) for ex in first} <= held


def randomized(obs, rng):
    """A fake views function: one 'image' per view, its randomization drawn from rng."""
    return [("oblique", int(obs[0].sum()), int(rng.integers(1 << 30))), ("top", int(obs[0].sum()), None)]


def clean(obs, rng):
    return [("clean", int(obs[0].sum()))]


def test_views_get_the_observation_and_the_samplers_rng_and_validation_is_fixed(store):
    data = T.ThinkData([store], [], held_share=0.25, views=randomized)
    rng = np.random.default_rng(0)
    drawn = [data.sample(rng) for _ in range(40)]
    for example in drawn:
        assert [view[0] for view in example.images] == ["oblique", "top"] and example.images[0][1] == int(example.obs[0].sum())
    assert len({example.images[0][2] for example in drawn}) == len(drawn)  # fresh randomization every draw
    again = T.ThinkData([store], [], held_share=0.25, views=randomized).validation(20, seed=3)
    assert [ex.images for ex in data.validation(20, seed=3)] == [ex.images for ex in again]  # a fixed seed
    plain = T.ThinkData([store], [], held_share=0.25, views=randomized, validation_views=clean).validation(20, seed=3)
    assert all(ex.images == [("clean", int(ex.obs[0].sum()))] for ex in plain)


def test_haiku_share_sets_the_share_of_labelled_episodes(store, labels):
    data = T.ThinkData([store], [labels], held_share=0.0, refusal_share=0.0, haiku_share=0.5)
    rng = np.random.default_rng(0)
    drawn = [data.sample(rng) for _ in range(600)]
    share = sum(ex.rung != "template" for ex in drawn) / len(drawn)
    assert 0.42 < share < 0.58  # 3 of 40 episodes are labelled; uniform drawing would give 0.075
    natural = T.ThinkData([store], [labels], held_share=0.0, refusal_share=0.0)
    drawn = [natural.sample(rng) for _ in range(600)]
    assert sum(ex.rung != "template" for ex in drawn) / len(drawn) < 0.15


# ---------------------------------------------------------------------------------------------
# Templates


def test_template_thought_names_what_the_task_names_first_and_reads_terrain():
    summary = {
        "terrain": {"layout": "trench between two plateaus", "trench": {"runs": "left to right"}},
        "robots": {"count": 12},
        "cheeses": [{"colour": "white", "area": "center", "on": "floor"}, {"colour": "orange", "area": "back left", "on": "plateau"}],
    }
    rec = record("logic", "eat:single", "eat the orange cheese", {"kind": "eat", "eat": ["orange"], "any_of": [], "order": []}, ["eat the orange cheese"])
    thought = T.template_thought("Eat the orange cheese!", rec, summary)
    assert thought == "you asked: eat the orange cheese. i see the swarm, a trench running left to right between two plateaus, an orange cheese on a plateau at the back left and a white cheese in the center. that means: eat the orange cheese. first: eat the orange cheese."
    many = dict(summary, terrain={"layout": "flat floor"}, cheeses=[{"colour": c, "area": "left", "on": "floor"} for c in ("red", "blue", "green", "white", "cyan", "purple")])
    assert T.sight(many, rec).endswith("and other things")
    assert T.template_line(EAT_TWO, 1) == "eat the blue cheese next."


def test_visible_drops_counts_and_coordinates_but_keeps_what_the_task_states():
    tower = record("construction", "tower", "build a tower 4 robots tall", {"kind": "tower", "height": 4}, ["build a tower 4 robots tall"])
    assert T.visible("you asked: make a 4-robot tower. i see 47 robots and no towers yet. first: build a tower 4 robots tall.", "make a 4-robot tower", tower) == "you asked: make a 4-robot tower. i see the robots and no towers yet. first: build a tower 4 robots tall."
    blocks = record("blocks", "line_blocks", "line the blocks up", {"kind": "line_blocks", "axis": "x"}, ["push the red block into the row at x = 7"])
    assert T.visible("you asked: line the blocks up. i see a red block at x 3. first: push the red block into the row at x = 7.", "line the blocks up", blocks) == "you asked: line the blocks up. i see a red block. first: push the red block into the row running back to front."
    subset = record("subset", "climb@count", "4 of you: climb", {"kind": "climb", "who": "count", "count": 4}, ["4 of you: climb"])
    assert T.visible("you asked: four of you climb. i see 30 robots and a plateau. the other 26 hold. first: 4 of you: climb.", "four of you climb", subset) == "you asked: four of you climb. i see the robots and a plateau. the others hold. first: 4 of you: climb."
    assert T.visible("you asked: all 24 of you hold. i see 24 robots. first: hold still.", "all 24 of you hold", None) == "you asked: all 24 of you hold. i see 24 robots. first: hold still."  # the instruction states it
    crowds = T.visible("you asked: the right group moves. i see three crowds: 15 at the front left, 9 on the right. the 9 move, depth 4. first: the right group moves.", "the right group moves", None)
    assert not re.search(r"\d", crowds) and crowds.endswith("first: the right group moves.")
    assert T.visible("i see 54 robots on a flat floor. first: form a wall.", "form a wall") == "i see the robots on a flat floor. first: form a wall."  # no quote


def test_template_thoughts_state_no_robot_count(data):
    for ep in data.episodes:
        for part in data.template_parts(ep):
            assert "i see the swarm" in part["thought"] and not re.search(r"\d+ robots", part["thought"])
    assert data.counts()["rewritten"]["template_thoughts"] > 0


@pytest.fixture(scope="module")
def real_store(tmp_path_factory):
    """A tiny store written by demos.write with the real stores' options (short episodes)."""
    path = tmp_path_factory.mktemp("real") / "store"
    D.write(path, 2, seed=1, chunk=2, steps=16, moves="cube", robots=S.ROBOTS, failure=failure_rates(), compact=True, counts="log", rule="switch_cube", coverage=1.0, objects=1.0)
    return path


def test_a_real_store_gives_an_example_at_every_step_of_every_episode(real_store):
    data = T.ThinkData([real_store], [], held_share=0.0)
    assert data.episodes and all(not ep.labelled for ep in data.episodes)
    pattern = re.compile(r"you asked: .+\. i see the swarm[ ,].*\. (that means: .+|nothing here matches it, so there is nothing to do)\. first: .+\.")
    for ep in data.episodes:
        parts = data.labelling(ep, np.random.default_rng(0))
        assert len(parts) == len(data.instructions(ep))
        for part in parts:
            assert pattern.fullmatch(part["thought"]) and part["thought"] == part["thought"].lower()
        for t in range(data.steps(ep)):
            example = data.example(ep, t, parts)
            assert example.codes.shape == (16, 16, 16) and example.obs.shape == (10, 16, 16, 16)
            assert example.context_text.count(T.IMAGES) == 1 and example.reply_text.endswith(T.ACT)
            assert (example.codes[example.obs[0] == 0] == STAY).all()
    kinds = {kind.split(":")[0] for kind in T.render_examples(data)}
    assert {"first", "plain"} <= kinds


# ---------------------------------------------------------------------------------------------
# The user's frame


def test_direction_words_turn_into_the_users_frame():
    text = "go to the front left, then push the red cheese in front of the blue one toward the right, side by side from left to right"
    assert T.in_user_frame(text, 0) == text
    assert T.in_user_frame(text, 1) == "go to the back left, then push the red cheese to the left of the blue one toward the front, one behind the other from back to front"
    assert T.in_user_frame(text, 2) == "go to the back right, then push the red cheese behind the blue one toward the left, side by side from right to left"
    assert T.in_user_frame("a cheese on the left and the back-right corner", 3) == "a cheese at the front and the back-left corner"
    assert T.in_user_frame("move three steps forward, then step back", 1) == "move three steps to the left, then step right"
    idioms = "eat it right now, that's right; nothing left, come back together"
    assert T.in_user_frame(idioms, 1) == idioms
    for turns in range(4):  # four quarter turns are the identity, word by word
        once = text
        for _ in range(4):
            once = T.in_user_frame(once, 1)
        assert once == text
    assert not T.mappable("go to the far side") and not T.mappable("head north") and T.mappable("go to the far left")


def test_the_users_place_is_the_camera_turned_by_the_stated_offset():
    assert T.quarter_turns(0) == 0 and T.quarter_turns(90) == 1 and T.quarter_turns(180) == 2 and T.quarter_turns(-90) == 3
    assert [T.user_turns(90, name) for name in ("behind", "left", "right", "opposite")] == [1, 0, 2, 3]  # a camera at the world's right
    assert [T.user_turns(T.STAND_IN_AZIMUTH, name) for name in ("behind", "left", "right", "opposite")] == [1, 0, 2, 3]  # half way goes to the later quarter


def turned_views(obs, rng):
    """Fake views: one image, the camera at the world's right."""
    return ["iso"], {"azimuth": 90.0}


def test_examples_state_where_the_user_stands_and_speak_in_that_frame(store, labels):
    data = T.ThinkData([store], [labels], held_share=0.0, views=turned_views, refusal_share=0.0)
    ep = ep_of(data, 5)  # switch, template: "go to the front" then "stay where you are"
    seen = set()
    for seed in range(40):
        rng = np.random.default_rng(seed)
        example = data.example(ep, 7, data.labelling(ep, rng), rng)
        seen.add(example.user)
        assert example.camera == 90.0 and example.turns == T.user_turns(90.0, example.user)
        assert T.USER_POSITIONS[example.user][1] + "\nstay where you are<|im_end|>" in example.context_text
        assert f"you asked: stay where you are. i see the swarm on a flat floor, a red cheese {T.in_user_frame('at the back left', example.turns)}" in example.context_text
        assert example.context_text.endswith("<|im_start|>assistant\n") and example.reply_text.startswith("<think>\n") and example.reply_text.endswith("<tool_call>")
        before = data.example(ep, 2, data.labelling(ep, rng), rng)
        assert f"\n{T.in_user_frame('go to the front', before.turns)}<|im_end|>" in before.context_text
    assert seen == set(T.USER_POSITIONS)


def test_unturnable_labels_are_used_only_in_the_worlds_frame(store, tmp_path):
    n2 = index_of(store, 2)
    far = pair(SEED, n2, "formation", "mask", "close", "make a wall on the far side", "you asked: make a wall on the far side. first: form a wall.")
    near = pair(SEED, n2, "formation", "mask", "casual", "wall pls", "you asked: wall pls. first: form a wall.")
    both = T.ThinkData([store], [label_dir(tmp_path / "both", [far, near])], held_share=0.0, views=turned_views)
    alone = T.ThinkData([store], [label_dir(tmp_path / "alone", [far])], held_share=0.0, views=turned_views)
    for seed in range(40):
        rng = np.random.default_rng(seed)
        for data, fallback in ((both, "wall pls"), (alone, "build a wall")):
            ep = ep_of(data, 2)
            example = data.example(ep, 0, data.labelling(ep, rng), rng)
            if example.turns:
                assert example.instruction == fallback and "far side" not in example.context_text + example.reply_text
    assert both.frames["rung_swapped"] > 0 and alone.frames["template_instead"] > 0


# ---------------------------------------------------------------------------------------------
# Tokens


def test_fixed_tokens_are_single_tokens_and_the_thought_forms_are_fixed(tokenizer):
    ids = T.check_tokens(tokenizer)
    assert ids == {"<think>": 248068, "</think>": 248069, "<tool_call>": 248058, "<|im_start|>": 248045, "<|im_end|>": 248046, "<|image_pad|>": 248056, "<|vision_start|>": 248053, "<|vision_end|>": 248054, "<|video_pad|>": 248057}
    assert tokenizer(T.act_reply(""), add_special_tokens=False).input_ids == [248068, 271, 248069, 271, 248058]
    thought = tokenizer(T.act_reply("first: eat the red cheese."), add_special_tokens=False).input_ids
    assert thought[:2] == [248068, 198] and thought[-3:] == [248069, 271, 248058]


def test_tokenize_expands_the_images_and_labels_only_the_reply(data, tokenizer):
    ep = ep_of(data, 0)
    parts = data.labelling(ep, np.random.default_rng(0))
    assert T.image_tokens(data.example(ep, 0, parts).images[0]) == 64  # 272 x 272 resized to 256 x 256: 8 x 8 tokens of 32 pixels
    for t in (0, 5):
        example = data.example(ep, t, parts)
        ids, labels = T.tokenize(example, tokenizer)
        vision = "<|vision_start|>" + "<|image_pad|>" * 64 + "<|vision_end|>"
        joint = tokenizer(example.context_text.replace(T.IMAGES, vision) + example.reply_text, add_special_tokens=False).input_ids
        assert ids.tolist() == joint
        reply = tokenizer(example.reply_text, add_special_tokens=False).input_ids
        assert labels.tolist() == [-100] * (len(ids) - len(reply)) + reply
        assert ids[-1].item() == tokenizer.convert_tokens_to_ids(T.ACT)
        two, _ = T.tokenize(example, tokenizer, counts=[4, 6])
        assert two.tolist().count(tokenizer.convert_tokens_to_ids(T.IMAGE_PAD)) == 10 and two.tolist().count(tokenizer.convert_tokens_to_ids(T.VISION_START)) == 2
        with_slots, slot_labels = T.tokenize(example, tokenizer, slots=64)
        assert with_slots.tolist() == ids.tolist() + [tokenizer.convert_tokens_to_ids(T.SLOT)] * 64 and (slot_labels[-64:] == -100).all()
    broken = T.Example(**{**example.__dict__, "context_text": example.context_text + T.IMAGES})
    with pytest.raises(ValueError, match="IMAGES"):
        T.tokenize(broken, tokenizer)


def test_pool_picks_in_proportion_to_weight():
    episodes = [T.Episode(0, e, 0, e, "logic", False, [], [], [], "eat:absent" if e < 2 else "eat:single", 0.1 if e < 2 else 1.0) for e in range(6)]
    pool = T.Pool(episodes)
    rng = np.random.default_rng(0)
    picks = [pool.pick(pool.episodes, rng).e for _ in range(20000)]
    share = sum(p < 2 for p in picks) / len(picks)
    assert abs(share - 0.2 / 4.2) < 0.01  # two episodes at 0.1 among four at 1.0


def test_no_steps_are_drawn_after_the_task_is_done(store, labels):
    """Steps after the last instruction's "done." (a finished swarm standing still) are never drawn."""
    data = T.ThinkData([store], [labels], held_share=0.0, refusal_share=0.0)
    done = {}
    for ep in data.episodes:
        last = len(data.instructions(ep)) - 1
        ends = [ev.t for ev in data.episode_events(ep) if ev.kind == "done" and ev.part == last]
        if ends:
            done[(ep.seed, ep.index)] = ends[-1]
    assert done
    rng = np.random.default_rng(3)
    drawn = [data.sample(rng) for _ in range(600)]
    assert all(ex.t <= done[(ex.seed, ex.index)] for ex in drawn if (ex.seed, ex.index) in done)
