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
from beebot.vla.voxel.lm import PATCH
from beebot.vla.voxel.think import ThinkStep
from beebot.vla.voxel.world import N_ACTIONS, STAY


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
    acting rows choose `move` everywhere; every context is kept."""

    visual = None

    def __init__(self, script, move=STAY):
        self.script, self.move = script, move
        self.calls = []

    def think_then_act(self, obs, contexts, images=None, max_new_tokens=0):
        self.calls.append(list(contexts))
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
    R.play_policy(policy, [e])
    k = e.starts[1]
    user = [calls[0].split("<|im_start|>user\n")[1] for calls in policy.calls]
    assert all(e.texts[0] in u for u in user[:k]) and all(e.texts[1] in u for u in user[k:])
    assert all(e.texts[1] not in u for u in user[:k])
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
        thought = R.canonical_thought(e.texts[0], e.records[0], S.scene_summary(R.prepare_world(e.world), 0, e.task))
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
            thought = R.canonical_thought(text, record, S.scene_summary(world, 0, e.task))
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
