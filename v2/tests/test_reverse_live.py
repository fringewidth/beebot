"""Live reverse examples (vla/voxel/reverse_live.py): every family gives an episode that replays under
world.step and does what its words say, the words never carry the simulator's own nouns or anything
the scene may lack, ThinkData mixes live examples in at their share, and ThinkData without `live`
gives today's examples byte for byte."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

import think_data_digest as DIGEST  # noqa: E402

from beebot.vla.voxel import reverse as R  # noqa: E402
from beebot.vla.voxel import reverse_live as L  # noqa: E402
from beebot.vla.voxel import think_data as T  # noqa: E402
from beebot.vla.voxel.world import STAY  # noqa: E402

SMALL = (6, 14)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def texts():
    return L.Texts()


@pytest.fixture(scope="module")
def episodes():
    """One small episode per family (seed 0, else the next seeds)."""
    out = {}
    for family in L.FAMILIES:
        for seed in range(4):
            row, _ = L.generate(seed, [family], robots=SMALL, replay=True)
            if row is not None:
                out[family] = L.LiveEpisode(row)
                break
    return out


# ---------------------------------------------------------------------------------------------
# Defaults unchanged


def test_default_examples_are_byte_identical(tmp_path):
    """live=None and a zero share draw no extra random numbers; with the tokenizer cached, the
    examples equal the fixture written before the grounding kinds and live examples existed."""
    plain = DIGEST.digests(DIGEST.build(tmp_path / "a"), n=24)
    off = DIGEST.digests(DIGEST.build(tmp_path / "b", live=None), n=24)
    zero = DIGEST.digests(DIGEST.build(tmp_path / "c", live={"share": 0.0}), n=24)
    assert plain == off == zero
    tokenizer = DIGEST.tokenizer_or_none()
    if tokenizer is None:
        pytest.skip("the Qwen3.5 tokenizer is not cached")
    want = json.loads(DIGEST.FIXTURE.read_text())["digests"]
    assert DIGEST.digests(DIGEST.build(tmp_path / "d", live=None), tokenizer) == want


# ---------------------------------------------------------------------------------------------
# Episodes


def test_every_family_gives_an_episode_that_replays(episodes):
    assert set(episodes) == set(L.FAMILIES)
    for family, ep in episodes.items():
        report = R.replay([ep.as_reverse()])
        assert bool(report["passed"][0]), family
        start, end = ep.occupancy(0), ep.occupancy(ep.steps)
        assert start.sum() == end.sum() == ep.meta["robots"]
        if family == "stay":
            assert (start == end).all() and ep.steps == R.HOLD_STEPS
        else:
            assert not (start == end).all(), family
        assert (ep.codes(ep.steps) == STAY).all()  # the goal held: every robot stays


def test_end_states_do_what_the_words_say(episodes):
    goto = episodes["goto"]
    box = L.region_columns(goto.meta["info"]["region"])
    assert L.share_in(goto.occupancy(goto.steps), box) == 1.0
    assert L.share_in(goto.occupancy(0), box) <= 1 / 3
    tower = episodes["tower"]
    assert L.tallest(tower.occupancy(tower.steps)) == tower.meta["info"]["height"]
    assert L.tallest(tower.occupancy(0)) < tower.meta["info"]["height"]
    assert len(L.components(episodes["gather"].occupancy(episodes["gather"].steps))) == 1
    groups = episodes["groups"]
    assert len(L.components(groups.occupancy(groups.steps))) == groups.meta["info"]["count"]
    assert L.touching_any(episodes["spread"].occupancy(episodes["spread"].steps)) == 0
    move = episodes["move"]
    side = move.meta["info"]["toward"]
    axis, sign = L.S.SIDE_AXES[side]
    shift = np.argwhere(move.occupancy(move.steps))[:, axis].mean() - np.argwhere(move.occupancy(0))[:, axis].mean()
    assert shift * sign == pytest.approx(3.0)


def test_checks_refuse_ends_that_do_not_match():
    rng = np.random.default_rng(3)
    draw = L.draw_goto(rng)
    end = (draw.goal.world.mass[0] > 0.5).numpy()
    assert draw.check(np.zeros_like(end), end) == ""
    moved = np.zeros_like(end)
    moved[0, 8, 8] = True  # a robot outside the region
    assert draw.check(np.zeros_like(end), end | moved) != ""
    assert draw.check(end, end) != ""  # the start already in the region


# ---------------------------------------------------------------------------------------------
# Words


def test_text_filter():
    rec = L.record("goto", "go to the front left of the floor", {"kind": "goto", "region": "front left"})
    assert L.text_ok("everyone to the front left corner", rec)
    assert not L.text_ok("go to the front left, away from the red cheese", rec)  # a colour and a noun the scene may lack
    assert not L.text_ok("all 38 of you go to the front left", rec)  # a robot count
    assert not L.text_ok("go to the back right", rec)
    assert not L.text_ok("go to the row with the smallest y", rec)
    tower = L.record("tower", "build a tower 5 robots tall", {"kind": "tower", "height": 5})
    assert L.text_ok("stack five of you into a tower", tower)
    assert not L.text_ok("stack six of you into a tower", tower)


def test_words_never_carry_internal_nouns(episodes, texts):
    rng = np.random.default_rng(0)
    for family, ep in episodes.items():
        for _ in range(20):
            instruction, thought, rung = texts.words(ep.meta, rng)
            assert not L.INTERNAL.search(instruction), (family, instruction)
            assert not L.re.search(r"\b(?:zones?|stripes?|cheeses?)\b", thought), (family, thought)
            assert thought.startswith("you asked: ")
            assert rung in L.RUNGS + (T.TEMPLATE,)


def test_haiku_texts_are_reused_where_specs_match(texts):
    rec = L.record("tower", "build a tower 4 robots tall", {"kind": "tower", "height": 4})
    assert len(texts.choices(rec, "tower")) > 50
    rec = L.record("gather", "gather into one group", {"kind": "gather"})
    assert len(texts.choices(rec, "gather")) > 50


# ---------------------------------------------------------------------------------------------
# ThinkData


def test_think_data_mixes_live_examples_at_their_share(tmp_path):
    live = {"share": 0.5, "families": ["stay", "line", "tower"], "robots": SMALL, "reuse": 4, "held": 3}
    data = DIGEST.build(tmp_path, live=live)
    rng = np.random.default_rng(0)
    examples = [data.sample(rng) for _ in range(40)]
    mine = [ex for ex in examples if isinstance(ex, L.LiveExample)]
    assert 10 <= len(mine) <= 30
    for ex in mine:
        assert ex.act and ex.store == L.RUNG and ex.family.startswith("live:")
        assert ex.context_text.count(T.IMAGES) == 1 and ex.reply_text.startswith(T.THINK)
        assert ex.codes.shape == ex.obs.shape[1:]
        moving = ex.codes != STAY
        assert not (moving & ~(ex.obs[0] > 0.5)).any()  # codes only on robots
        assert (ex.kind == "first") == (ex.t == 0)
    held = data.validation(12, seed=1)
    assert held == data.validation(12, seed=1)
    assert data.counts()["live"]["examples"] >= len(mine)


def test_background_workers_fill_the_cache():
    live = L.Live(None, share=1.0, families=["stay"], workers=1, robots=SMALL)
    try:
        ex = live.sample(np.random.default_rng(0))
        assert ex.live == "stay" and live.generated >= 1
    finally:
        live.close()


def test_zone_rewordings_are_dropped_by_default():
    index = L.haiku_index()
    assert not any('"goto_zone"' in key or '"line_zone"' in key for key in index)
    assert L.Texts().index == index
