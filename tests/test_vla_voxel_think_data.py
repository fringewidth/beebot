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


def pair(seed, index, family, kind, rung, instruction, thought, switches=(), **extra):
    return {"rung": rung, "instruction": instruction, "thought": thought, "switches": list(switches), "done": "done.", "pair_id": f"s{seed}-{index}:{rung}", "seed": seed, "index": index, "family": family, "kind": kind, **extra}


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


@pytest.fixture(scope="module")
def data(store, labels):
    return T.ThinkData([store], [labels], held_share=0.0)


def ep_of(data, e):
    return next(ep for ep in data.episodes if ep.e == e)


@pytest.fixture(scope="module")
def tokenizer():
    try:
        import transformers

        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception:
        pytest.skip("the Qwen tokenizer is not cached")


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
    assert first.context_text == "<|im_start|>user\neat red, then blue\n<|image_pad|><|im_end|>\n<|im_start|>assistant\n"
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
        assert example.context_text.count(T.SCENE) == 1


def test_an_instruction_change_puts_the_new_instruction_in_the_user_turn(data):
    ep = ep_of(data, 5)  # switch, template
    parts = data.labelling(ep, np.random.default_rng(0))
    before = data.example(ep, 5, parts)
    change = data.example(ep, 6, parts)
    later = data.example(ep, 8, parts)
    assert "user\ngo to the front\n" in before.context_text
    assert change.kind == "change" and "user\nstay where you are\n" in change.context_text
    assert change.reply_text.startswith("<think>\nyou asked: stay where you are. i see") and "first: stay where you are." in change.reply_text
    assert "you asked: go to the front." in change.context_text  # memory: the old instruction's thought
    assert "you asked: stay where you are." in later.context_text and "you asked: go to the front." not in later.context_text
    assert "user\nstay where you are\n" in later.context_text and later.kind == "plain"


def test_haiku_labels_replace_the_template_and_rungs_are_drawn(data):
    ep = ep_of(data, 0)
    assert ep.labelled and len(ep.choices) == 2
    rungs = set()
    for seed in range(12):
        parts = data.labelling(ep, np.random.default_rng(seed))
        rungs.add(parts[0]["rung"])
        first = data.example(ep, 0, parts)
        assert first.reply_text == f"<think>\n{parts[0]['thought']}\n</think>\n\n<tool_call>"
        assert f"user\n{parts[0]['instruction']}\n" in first.context_text and first.rung == parts[0]["rung"]
        assert data.example(ep, 4, parts).reply_text == "<think>\nnow the blue one\n</think>\n\n<tool_call>"
        assert parts[0]["thought"] in data.example(ep, 7, parts).context_text
    assert rungs == {"close", "casual"}


def test_haiku_switch_parts_label_each_instruction(data):
    ep = ep_of(data, 1)
    assert ep.parts and not ep.choices
    parts = data.labelling(ep, np.random.default_rng(0))
    assert [p["instruction"] for p in parts] == ["head to the front", "could you all freeze"]
    change = data.example(ep, 6, parts)
    assert "user\ncould you all freeze\n" in change.context_text and change.rung == "indirect"
    assert change.reply_text.startswith("<think>\nyou asked: could you all freeze.")
    assert "you asked: head to the front." in change.context_text
    assert data.example(ep, 3, parts).rung == "close"


def test_refusals_are_a_thought_and_a_text_answer_at_the_first_frame(data, store, tokenizer):
    ep = ep_of(data, 2)
    example = data.refusal(ep, ep.refusals[0])
    assert not example.act and example.codes is None and example.t == 0 and example.kind == "refusal" and example.rung == "refusal"
    assert example.reply_text == "<think>\nyou asked: make the blue cheese fly. gravity is on, so i will not act.\n</think>\n\nthe blue cheese cannot fly, so i will do nothing.<|im_end|>"
    assert T.ACT not in example.reply_text and "user\nmake the blue cheese fly\n" in example.context_text
    assert torch.equal(example.obs, observe(world_at(load(store), [(2, 0)]))[0])
    assert (T.still(example.obs) == STAY).all()
    ids, labels = T.tokenize(example, tokenizer)
    assert ids[-1].item() == tokenizer.convert_tokens_to_ids(T.END) and labels[-1].item() == ids[-1].item()


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
    data = T.ThinkData([store], [label_dir(tmp_path / "other-seed", [dict(good, seed=SEED + 1)], seed=SEED + 1)])
    assert not any(ep.labelled for ep in data.episodes)  # labels of another seed join nothing
    with pytest.raises(AssertionError, match="test set"):
        T.ThinkData([store], [label_dir(tmp_path / "test", [good])])


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
    assert held and len(held) < EPISODES and held == {(ep.seed, ep.index) for ep in data.episodes if T.held_out(ep.seed, ep.index, 0.25)}
    rng = np.random.default_rng(0)
    drawn = [data.sample(rng) for _ in range(1500)]
    assert not {(ex.seed, ex.index) for ex in drawn} & held
    refusals = sum(ex.kind == "refusal" for ex in drawn) / len(drawn)
    acting = [ex for ex in drawn if ex.act]
    thinking = sum(ex.kind != "plain" for ex in acting) / len(acting)
    assert 0.07 < refusals < 0.13 and 0.29 < thinking < 0.38
    assert all((ex.kind == "plain") == ("<think>\n\n</think>" in ex.reply_text) for ex in acting)
    first = data.validation(30, seed=1)
    assert [(ex.e, ex.t, ex.rung) for ex in first] == [(ex.e, ex.t, ex.rung) for ex in T.ThinkData([store], [labels], held_share=0.25, refusal_share=0.1).validation(30, seed=1)]
    assert {(ex.seed, ex.index) for ex in first} <= held


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
    assert thought == "you asked: eat the orange cheese. i see 12 robots, a trench running left to right between two plateaus, an orange cheese on a plateau at the back left and a white cheese in the center. that means: eat the orange cheese. first: eat the orange cheese."
    many = dict(summary, terrain={"layout": "flat floor"}, cheeses=[{"colour": c, "area": "left", "on": "floor"} for c in ("red", "blue", "green", "white", "cyan", "purple")])
    assert T.sight(many, rec).endswith("and 2 more things")
    assert T.template_line(EAT_TWO, 1) == "eat the blue cheese next."


@pytest.fixture(scope="module")
def real_store(tmp_path_factory):
    """A tiny store written by demos.write with the real stores' options (short episodes)."""
    path = tmp_path_factory.mktemp("real") / "store"
    D.write(path, 2, seed=1, chunk=2, steps=16, moves="cube", robots=S.ROBOTS, failure=failure_rates(), compact=True, counts="log", rule="switch_cube", coverage=1.0, objects=1.0)
    return path


def test_a_real_store_gives_an_example_at_every_step_of_every_episode(real_store):
    data = T.ThinkData([real_store], [], held_share=0.0)
    assert data.episodes and all(not ep.labelled for ep in data.episodes)
    pattern = re.compile(r"you asked: .+\. i see \d+ robots?[ ,].*\. (that means: .+|nothing here matches it, so there is nothing to do)\. first: .+\.")
    for ep in data.episodes:
        parts = data.labelling(ep, np.random.default_rng(0))
        assert len(parts) == len(data.instructions(ep))
        for part in parts:
            assert pattern.fullmatch(part["thought"]) and part["thought"] == part["thought"].lower()
        for t in range(data.steps(ep)):
            example = data.example(ep, t, parts)
            assert example.codes.shape == (16, 16, 16) and example.obs.shape == (10, 16, 16, 16)
            assert example.context_text.count(T.SCENE) == 1 and example.reply_text.endswith(T.ACT)
            assert (example.codes[example.obs[0] == 0] == STAY).all()
    kinds = {kind.split(":")[0] for kind in T.render_examples(data)}
    assert {"first", "plain"} <= kinds


# ---------------------------------------------------------------------------------------------
# Tokens


def test_fixed_tokens_are_single_tokens_and_the_thought_forms_are_fixed(tokenizer):
    ids = T.check_tokens(tokenizer)
    assert ids == {"<think>": 151667, "</think>": 151668, "<tool_call>": 151657, "<|im_start|>": 151644, "<|im_end|>": 151645, "<|image_pad|>": 151655, "<|video_pad|>": 151656}
    assert tokenizer(T.act_reply(""), add_special_tokens=False).input_ids == [151667, 271, 151668, 271, 151657]
    thought = tokenizer(T.act_reply("first: eat the red cheese."), add_special_tokens=False).input_ids
    assert thought[:2] == [151667, 198] and thought[-3:] == [151668, 271, 151657]


def test_tokenize_expands_the_scene_and_labels_only_the_reply(data, tokenizer):
    ep = ep_of(data, 0)
    parts = data.labelling(ep, np.random.default_rng(0))
    for t in (0, 5):
        example = data.example(ep, t, parts)
        ids, labels = T.tokenize(example, tokenizer, scene_tokens=64)
        joint = tokenizer(example.context_text + example.reply_text, add_special_tokens=False).input_ids
        scene = tokenizer.convert_tokens_to_ids(T.SCENE)
        at = joint.index(scene)
        assert ids.tolist() == joint[:at] + [scene] * 64 + joint[at + 1 :]
        reply = tokenizer(example.reply_text, add_special_tokens=False).input_ids
        assert labels.tolist() == [-100] * (len(ids) - len(reply)) + reply
        assert ids[-1].item() == tokenizer.convert_tokens_to_ids(T.ACT)
        with_slots, slot_labels = T.tokenize(example, tokenizer, scene_tokens=64, slots=True)
        assert with_slots.tolist() == ids.tolist() + [tokenizer.convert_tokens_to_ids(T.SLOT)] * 64 and (slot_labels[-64:] == -100).all()
    broken = T.Example(**{**example.__dict__, "context_text": example.context_text + T.SCENE})
    with pytest.raises(ValueError, match="SCENE"):
        T.tokenize(broken, tokenizer)
