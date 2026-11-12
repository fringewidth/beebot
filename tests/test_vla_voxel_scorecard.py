"""The closed-loop rollout (voxel/rollout.py) and the 3D scorecard (voxel/scorecard.py): a scripted
stand-in policy for the bookkeeping (switches, refusals, oracle thoughts, memory) and a tiny random
Qwen3.5 for the real think_then_act path."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from beebot.vla.voxel import rollout as R
from beebot.vla.voxel import scorecard as C
from beebot.vla.voxel import suite as S
from beebot.vla.voxel import think_data as T
from beebot.vla.voxel.camera_frame import SIDES, turn_grid
from beebot.vla.voxel.lm import PATCH, world_at
from beebot.vla.voxel.play_store import load
from beebot.vla.voxel.think import ThinkStep
from beebot.vla.voxel.world import DIRS, N_ACTIONS, STAY, observe


def short(episodes, steps=4):
    return [replace(e, steps=steps) for e in episodes]


@pytest.fixture(scope="module")
def motion():
    return C.drawn_episodes("test", "motion", 3)


@pytest.fixture(scope="module")
def switch():
    return C.drawn_episodes("test", "switch", 2)


class Scripted:
    """A stand-in policy: think_then_act answers from script(context, row) -> (thought, act, answer),
    acting rows choose `move` everywhere; every context and observation is kept."""

    visual = None

    def __init__(self, script, move=STAY):
        self.script, self.move = script, move
        self.calls = []
        self.seen = []

    def think_then_act(self, obs, contexts, images=None, max_new_tokens=0):
        self.calls.append(list(contexts))
        self.seen.append(obs.clone())
        replies = [self.script(c, k) for k, c in enumerate(contexts)]
        b = len(contexts)
        n = (obs.shape[-1] // PATCH) ** 3
        logits = torch.zeros(b, n, PATCH**3, N_ACTIONS)
        logits[..., self.move] = 1.0
        acts = torch.tensor([a for _, a, _ in replies])
        return ThinkStep([[] for _ in replies], [t for t, _, _ in replies], [w for _, _, w in replies], acts, torch.ones(b, dtype=torch.bool), logits)


def thinking_once(context, k):
    """A first thought when the memory is empty, a switch line on the step after, silence later."""
    if "<|im_start|>assistant\n" in context.split("<|im_start|>user")[0]:
        return ("", True, "")
    return ("you asked: go. i see the swarm. first: gather into one group.", True, "")


def test_instruction_switch_at_its_step(switch):
    e = short(switch, steps=30)[0]
    policy = Scripted(thinking_once)
    played = R.play_policy(policy, [e])[0]
    k = e.starts[1]
    first, second = (T.in_user_frame(text, played.turns) for text in e.texts)  # as said in the episode's drawn frame
    user = [calls[0].split("<|im_start|>user\n")[1] for calls in policy.calls]
    assert all(first in u for u in user[:k]) and all(second in u for u in user[k:])
    assert all(second not in u for u in user[:k])
    # memory: the first thought from step 1 on, still there on the step the instruction changes
    assert "<|im_start|>assistant\nyou asked: go." in policy.calls[k][0]


def test_refusal_ends_the_episode_with_no_moves(motion):
    episodes = short(motion, steps=6)
    policy = Scripted(lambda c, k: ("you asked: sing. i see robots. first: nothing.", False, "i only move robots."), move=0)
    played = R.play_policy(policy, episodes, record=True)
    assert len(policy.calls) == 1  # every row spoke on the first step: no further calls
    for p in played:
        assert p.spoke == {"t": 0, "text": "i only move robots."}
        assert (p.actions == STAY).all() and max(p.moves) == 0.0


def test_acting_moves_robots(motion):
    played = R.play_policy(Scripted(lambda c, k: ("", True, ""), move=0), short(motion, steps=2), record=True)
    assert all(p.spoke is None and p.moves[0] > 0 for p in played)


def test_oracle_arm_feeds_the_canonical_thought(motion):
    episodes = C.arm_episodes("oracle", short(motion, steps=3))
    policy = Scripted(lambda c, k: ("", True, ""))
    played = R.play_policy(policy, episodes)
    for k, (e, p) in enumerate(zip(episodes, played)):
        thought = R.oracle_thought(e.texts[0], e.records[0], S.scene_summary(R.prepare_world(e.world), 0, e.task), p.turns)
        assert policy.calls[0][k].endswith(f"<|im_start|>assistant\n<think>\n{thought}\n</think>\n\n")
        assert p.thoughts[0] == {"t": 0, "part": 0, "text": thought, "forced": True}
        assert policy.calls[1][k].startswith(f"<|im_start|>assistant\n{thought}<|im_end|>")  # in memory on the next step
        assert "<think>" not in policy.calls[1][k]  # fed only on the instruction's first step
    checks = C.thought_checks(episodes[0], played[0])
    assert checks[0]["parse"] and checks[0]["first"] and checks[0]["resolves"]


def test_canonical_thoughts_pass_the_checks():
    """The checker accepts the canonical thought of every family's task (its labels' own form);
    build is left out here: its first draw in a process costs a minute (build.min_sizes)."""
    for family in C.FAMILIES[:-1]:
        e = C.drawn_episodes("test", family, 1)[0]
        world = R.prepare_world(e.world)
        for text, record in zip(e.texts, e.records):
            thought = R.oracle_thought(text, record, S.scene_summary(world, 0, e.task))
            assert all(C.check_thought(thought, text, record).values()), (family, thought, record)
    record = {"plain": "eat the red cheese first, then the blue cheese", "subgoals": ["eat the red cheese", "eat the blue cheese"], "spec": {"kind": "eat"}}
    assert not C.check_thought("you asked: x. i see red and blue. first: the blue one.", "x", record)["first"]
    assert C.check_thought("you asked: x. i see red and blue. red then blue. first: the red one.", "x", record)["first"]
    assert not C.check_thought("eat it", "x", record)["parse"] and not any(C.check_thought("", "x", record).values())


def test_shuffled_control_pairs_differ(motion):
    shuffled = C.arm_episodes("shuffled", motion)
    for e, s in zip(motion, shuffled):
        assert s.texts[0] != e.texts[0] and s.seed == e.seed and s.task is e.task and s.key["donor"] != e.key["i"]


def test_determinism_and_batch_independence():
    policy = C.tiny_policy("Qwen/Qwen3.5-0.8B")
    episodes = short(C.drawn_episodes("test", "logic", 2), steps=2)
    first = R.play_policy(policy, episodes, max_new_tokens=6, record=True)
    again = R.play_policy(policy, episodes, max_new_tokens=6, record=True)
    alone = R.play_policy(policy, episodes[1:], max_new_tokens=6, record=True)
    sampled = [R.play_policy(policy, episodes, sample=True, max_new_tokens=6, record=True) for _ in range(2)]
    for a, b in zip(first, again):
        assert a.thoughts == b.thoughts and torch.equal(a.actions, b.actions) and a.score == b.score
    assert torch.equal(first[1].actions, alone[0].actions) and first[1].score == alone[0].score
    assert all(torch.equal(a.actions, b.actions) for a, b in zip(*sampled))


def test_expert_arm_scores_and_is_deterministic():
    episodes = short(C.drawn_episodes("test", "construction", 3), steps=8)
    first = R.play_expert(episodes, seed=1)
    again = R.play_expert(episodes, seed=1)
    assert [p.score for p in first] == [p.score for p in again]
    assert all(0.0 <= p.score <= 1.0 and isinstance(p.passed, bool) for p in first)
    assert any(max(p.moves) > 0 for p in first)


def test_summary_and_table(motion, tmp_path):
    episodes = {("test", "motion"): short(motion, steps=3), ("wording", "motion"): short(C.drawn_episodes("wording", "motion", 3), steps=3)}
    policy = Scripted(thinking_once, move=0)
    rows = []
    for arm in C.ARMS:
        for (split, family), base in episodes.items():
            if split in C.ARM_SPLITS[arm]:
                rows += C.run_arm(arm, C.arm_episodes(arm, base), policy, batch=2, emit=lambda row: None)
    summary = C.summarize(rows)
    groups = {(g["split"], g["arm"]): g for g in summary["groups"]}
    assert set(groups) == {("test", a) for a in ("policy", "shuffled", "oracle", "expert", "still")} | {("wording", a) for a in ("policy", "plain", "oracle", "expert", "still")}
    assert all(r["moved"] == 0.0 for r in rows if r["arm"] == "still")
    for g in groups.values():
        assert {"n", "score", "score_se", "pass", "pass_se", "unasked", "spoke", "unfinished", "seconds_per_episode"} <= set(g)
    assert {"thought_parse", "thought_first", "thought_resolves"} <= set(groups[("test", "policy")])
    paired = {p["split"]: p for p in summary["paired"]}
    assert {"gap_score", "gap_score_se", "gap_pass", "oracle_gain_score"} <= set(paired["test"])
    assert {"oracle_ratio", "oracle_ratio_se"} <= set(paired["wording"])
    assert {"policy", "gap_score"} <= set(summary["macro"]["test"]) and "oracle_ratio" in summary["macro"]["wording"]
    assert {"thought_parse", "thought_first", "passes"} <= set(summary["gates"]["a"]) and "wording" in summary["gates"]["b"]
    row = rows[0]
    assert {"split", "family", "i", "arm", "seed", "texts", "score", "passed", "thoughts", "spoke"} <= set(row)
    table = C.markdown(summary)
    assert "### test" in table and "### wording" in table and "### gates" in table


def test_stats():
    mean, se = C.mean_se([0.0, 1.0, 1.0, 0.0])
    assert mean == 0.5 and abs(se - np.std([0, 1, 1, 0], ddof=1) / 2) < 1e-9
    r, se = C.ratio_se([1, 1, 0, 1], [1, 1, 1, 1])
    assert r == 0.75 and se > 0
    assert C.macro([{"x": 0.2, "x_se": 0.1}, {"x": 0.4, "x_se": 0.1}], "x") == pytest.approx((0.3, np.sqrt(0.02) / 2))


def test_command_writes_ops_lines(tmp_path):
    """The command (expert and still arms, two repeats) writes the episode rows, the card and the
    live lines ops follows: rollout_success per group and per arm, scorecard3d_mean, scorecard3d_noise."""
    import json

    card_log, rollout_log = tmp_path / "card.jsonl", tmp_path / "rollout.jsonl"
    summary = C.main(["--arms", "expert,still", "--splits", "test", "--families", "motion,logic", "--per-family", "3", "--steps", "3", "--repeats", "2", "--out", str(tmp_path / "out"), "--log", str(card_log), "--rollout-log", str(rollout_log)])
    rollout = [json.loads(line) for line in rollout_log.read_text().splitlines()]
    card = [json.loads(line) for line in card_log.read_text().splitlines()]
    assert {(r["arm"], r["family"]) for r in rollout} >= {("expert", "motion"), ("still", "logic"), ("expert", "all")}
    assert all({"rollout_success", "n", "run", "time"} <= set(r) for r in rollout)
    means = [r for r in card if "scorecard3d_mean" in r and "scorecard3d_noise" not in r]
    noise = [r for r in card if "scorecard3d_noise" in r]
    assert len(means) == 4 and {r["repeat"] for r in means} == {0, 1}
    assert {(r["arm"], r["noise_from"], r["repeats"]) for r in noise} == {("expert", "repeats", 2), ("still", "repeats", 2)}
    rows = [json.loads(line) for line in (tmp_path / "out" / "episodes.jsonl").read_text().splitlines()]
    assert "summary" in rows[-1] and len(rows) == 1 + 2 * 2 * 2 * 3 and summary["repeats"] == rows[-1]["summary"]["repeats"]
    assert (tmp_path / "out" / "card.md").exists()


def think_data_tests():
    """tests/test_vla_voxel_think_data.py as a module (its synthetic store)."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("think_data_tests", Path(__file__).parent / "test_vla_voxel_think_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def drawn_stand_in(obs, rng=None, side=None):
    """The stand-in views from the side asked for, else from a side drawn from rng."""
    return T.stand_in_views(obs, side=side or SIDES[int(rng.integers(4))])


def test_rollout_contexts_and_observations_equal_think_data_examples(tmp_path):
    """For training episodes (an eat-two template with a switch line and done, and an instruction
    change), the context the rollout builds at every step, given the thoughts training emits, is the
    context ThinkData produced for that step, and the observation the policy gets is the example's
    (both in the camera's frame), for cameras on several sides."""
    TD = think_data_tests()
    store = TD.synthetic_store(tmp_path / "store")
    data = T.ThinkData([store], [], held_share=0.0, views=drawn_stand_in)
    raw = load(store)
    frames = set()
    for e in (4, 5):
        ep = next(x for x in data.episodes if x.e == e)
        parts = data.labelling(ep, np.random.default_rng(0))
        entries = data.instructions(ep)
        for seed in (0, 1, 4, 11):  # cameras at the world's left, right, back and front
            examples = [data.example(ep, t, parts, np.random.default_rng(seed)) for t in range(TD.STEPS)]
            thoughts = [x.reply_text[len("<think>\n") : x.reply_text.index("\n</think>")] for x in examples]
            episode = R.Episode(world_at(raw, [(e, 0)]), {}, None, [x["text"] for x in entries], [int(x["t"]) for x in entries], [x["record"] for x in entries], TD.STEPS, seed, camera=SIDES[examples[0].turns])
            policy = Scripted(lambda c, k: (thoughts[len(policy.calls) - 1], True, ""))
            actor = R.PolicyActor(policy, [episode], views=drawn_stand_in)
            for t in range(TD.STEPS):
                actor(world_at(raw, [(e, t)]), t)
            assert [calls[0] for calls in policy.calls] == [x.context_text for x in examples]
            assert all(torch.equal(seen[0], x.obs) for seen, x in zip(policy.seen, examples))
            assert actor.frames[0][0] == examples[0].turns
            frames.add(examples[0].turns)
    assert frames == {0, 1, 2, 3}  # the world's frame and every turned one were compared


def test_the_camera_frame_is_set_per_episode_and_turns_the_instruction(motion):
    base = replace(short(motion, steps=2)[0], texts=["go to the front left of the floor"])
    turned = R.play_policy(Scripted(lambda c, k: ("", True, "")), [base], views=T.stand_in_views)[0]
    assert turned.turns == 1  # the stand-in camera is at azimuth 45
    policy = Scripted(lambda c, k: ("", True, ""))
    R.play_policy(policy, [base], views=T.stand_in_views)
    said = T.in_user_frame("go to the front left of the floor", 1)
    assert said != "go to the front left of the floor" and policy.calls[0][0] == f"<|im_start|>user\n<IMAGES>\n{said}<|im_end|>\n<|im_start|>assistant\n"
    world = R.play_policy(Scripted(lambda c, k: ("", True, "")), [replace(base, texts=["head north"])], views=T.stand_in_views)[0]
    assert world.turns == 0  # cannot be turned: the camera looks from the world's front
    turns = {R.play_policy(Scripted(lambda c, k: ("", True, "")), [replace(base, seed=s)])[0].turns for s in range(12)}
    assert len(turns) > 1  # the camera is drawn per episode from its seed (SceneViews)
    record = {**base.records[0], "plain": "go to the front left of the floor", "subgoals": ["go to the front left of the floor"]}
    thought = R.oracle_thought(base.texts[0], record, {"terrain": {"layout": "flat floor"}, "cheeses": []}, 1)
    played = R.Played(0.0, False, False, thoughts=[{"t": 0, "part": 0, "text": thought, "forced": True}], turns=1)
    checks = C.thought_checks(replace(base, records=[record]), played)[0]
    assert checks["parse"] and checks["first"] and checks["resolves"], (thought, checks)


def camera_at(azimuths):
    """Fake views: one image, the camera at azimuths[side] (side None: drawn by the views); the
    sides asked for are kept."""
    asked = []

    def views(obs, rng, side=None):
        asked.append(side)
        return ["image"], {"azimuth": azimuths[side]}

    views.asked = asked
    return views


def test_the_policy_sees_its_cameras_frame_and_its_moves_are_turned_back(motion):
    """A camera at the world's right (quarter turn 1): the policy gets the observation turned into
    that camera's frame, and its move to the camera's right (+x') is the world's back (-y) when the
    world steps."""
    e = short(motion, steps=1)[0]
    policy = Scripted(lambda c, k: ("", True, ""), move=DIRS.index((0, 0, 1)))
    played = R.play_policy(policy, [e], views=camera_at({None: 100.0}), rule=None, failure=None, record=True)[0]
    assert played.turns == 1 and played.camera == 100.0
    assert torch.equal(policy.seen[0][0], turn_grid(observe(R.prepare_world(e.world))[0], 1))
    assert (played.actions[0] == DIRS.index((0, -1, 0))).all()  # what the world stepped with, everywhere
    behind = R.play_policy(Scripted(lambda c, k: ("", True, ""), move=DIRS.index((0, 0, 1))), [e], views=camera_at({None: 5.0}), rule=None, failure=None, record=True)[0]
    assert behind.turns == 0 and (behind.actions[0] == DIRS.index((0, 0, 1))).all()  # the default camera: the world's own frame


def test_the_camera_side_is_pinned_drawn_or_the_front_for_words_that_cannot_turn(motion):
    """Episode.camera pins the camera's board side; an instruction whose direction words cannot be
    turned (north, the far side) puts the camera at the world's front; otherwise the views draw it.
    The user is at the camera: the user turn carries the instruction only."""
    base = short(motion, steps=1)[0]
    azimuths = {None: 190.0, "front": 20.0, "left": 280.0}
    for texts, camera, side, turns in ((["go to the front left of the floor"], None, None, 2), (["head north"], None, "front", 0), (["go to the front left of the floor"], "left", "left", 3)):
        views = camera_at(azimuths)
        policy = Scripted(lambda c, k: ("", True, ""))
        played = R.play_policy(policy, [replace(base, texts=texts, camera=camera)], views=views)[0]
        assert views.asked[0] == side and played.turns == turns
        said = T.in_user_frame(texts[0], turns)
        assert policy.calls[0][0] == f"<|im_start|>user\n<IMAGES>\n{said}<|im_end|>\n<|im_start|>assistant\n"
    assert SIDES[3] == "left" and T.in_user_frame("go to the front left of the floor", 2) == "go to the back right of the floor"
