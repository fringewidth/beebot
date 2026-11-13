"""Grounding example kinds (vla/voxel/think_ground.py) and the guarantee that ThinkData without
`ground` gives today's examples byte for byte (token ids, images, observations, codes, texts)."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

import think_data_digest as DIGEST  # noqa: E402

from beebot.vla.voxel import reverse as R  # noqa: E402
from beebot.vla.voxel import think_data as T  # noqa: E402
from beebot.vla.voxel import think_ground as G  # noqa: E402
from beebot.vla.voxel import views as V  # noqa: E402
from beebot.vla.voxel import world as W  # noqa: E402
from beebot.vla.voxel.camera_frame import turn_codes, turn_grid  # noqa: E402

@pytest.fixture(scope="module")
def tokenizer():
    tok = DIGEST.tokenizer_or_none()
    if tok is None:
        pytest.skip("the Qwen3.5 tokenizer is not cached")
    return tok


@pytest.fixture(scope="module")
def ground_dir(tmp_path_factory):
    """A small ground store: tower and letter goals walked backwards, then one-robot moves from their frames."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    pairs = []
    for k in range(8):
        goal = R.make_goal("tower" if k % 2 else "letter", rng)
        pairs.append(({"source": f"test-{k}", "generator": goal.family, "text": goal.text, "turns": [goal.text]}, goal))
    rows = G.build_goals(pairs, rng)
    assert len(rows) >= 6
    parents = [G.GroundEpisode(r, False) for r in rows]
    moves = G.build_moves(parents, 10, rng)
    assert len(moves) >= 6
    return G.write_store(tmp_path_factory.mktemp("ground") / "s0", rows + moves)


@pytest.fixture(scope="module")
def grounding(ground_dir):
    return G.Grounding(None, dirs=[ground_dir], shares={k: 0.25 for k in G.KINDS}, held_share=0.0)


# ---------------------------------------------------------------------------------------------
# Defaults unchanged


def test_default_examples_are_byte_identical_to_before_grounding(tmp_path, tokenizer):
    """The fixture was written from think_data.py before think_ground existed (tests/think_data_digest.py)."""
    want = json.loads(DIGEST.FIXTURE.read_text())["digests"]
    assert DIGEST.digests(DIGEST.build(tmp_path), tokenizer) == want


def test_ground_none_or_zero_shares_leave_scene_view_examples_unchanged(tmp_path, ground_dir):
    """With the randomized scene views (views.scene_views) too: no grounding draws no extra random numbers."""

    def scene(obs, rng, side=None):
        spec = V.ViewSpec() if side is None else V.ViewSpec(azimuth=V.side_azimuth(side, rng))
        return V.scene_views(obs, rng, spec)

    plain = DIGEST.digests(DIGEST.build(tmp_path / "a", views=scene), n=16)
    off = DIGEST.digests(DIGEST.build(tmp_path / "b", views=scene, ground=None), n=16)
    zero = DIGEST.digests(DIGEST.build(tmp_path / "c", views=scene, ground={"dirs": [ground_dir], "shares": {}}), n=16)
    assert plain == off == zero


def test_think_data_draws_grounding_kinds_at_their_share(tmp_path, ground_dir):
    data = DIGEST.build(tmp_path, ground={"dirs": [ground_dir], "shares": {"predict": 0.5, "point": 0.2}, "held_share": 0.0})
    rng = np.random.default_rng(0)
    kinds = [data.sample(rng).kind for _ in range(120)]
    share = sum(k.startswith("ground:") for k in kinds) / len(kinds)
    assert 0.55 < share < 0.85
    assert {k for k in kinds if k.startswith("ground:")} == {"ground:predict", "ground:point"}
    assert data.counts()["ground"]["episodes"] > 0
    with pytest.raises(ValueError):
        DIGEST.build(tmp_path / "x", ground={"dirs": [ground_dir], "shares": {"nonsense": 0.1}})


# ---------------------------------------------------------------------------------------------
# The store


def test_packed_episodes_give_back_frames_and_codes(ground_dir):
    episodes = G.load_store([ground_dir], 0.0)
    for ep in episodes[:6]:
        row = ep.row
        for t in (0, ep.steps // 2, ep.steps - 1):
            occ = ep.occupancy(t)
            codes = ep.codes(t)
            assert (codes[~torch.from_numpy(occ)] == W.STAY).all()
            world = ep.world(t)
            assert world.mass.shape == (1, 16, 16, 16) and bool((world.mass[0] > 0.5).numpy().sum() == occ.sum())
        assert row["meta"]["steps"] == ep.steps
    replay = R.replay([_as_reverse_episode(ep) for ep in episodes])
    assert bool(replay["passed"].all())


def _as_reverse_episode(ep):
    mass = torch.stack([torch.from_numpy(ep.occupancy(t)).to(torch.uint8) for t in range(ep.steps + 1)])
    actions = torch.stack([ep.codes(t).to(torch.int8) for t in range(ep.steps)])
    return R.Episode(family="x", text="", world=ep.world(0), mass=mass, cheese_pos=ep.row["cheese_pos"], cheese_alive=ep.row["cheese_alive"], actions=actions, natural=True, end="")


def test_move_episodes_move_one_robot_from_its_cell_to_the_destination(ground_dir):
    moves = [ep for ep in G.load_store([ground_dir], 0.0) if ep.meta["kind"] == "move"]
    for ep in moves:
        start, end = ep.occupancy(0), ep.occupancy(ep.steps)
        assert start[tuple(ep.meta["mover"])] and not end[tuple(ep.meta["mover"])]
        assert end[tuple(ep.meta["dest"])] and not start[tuple(ep.meta["dest"])]
        assert (start ^ end).sum() == 2
        assert all(ep.mover_at(t) is not None for t in range(ep.steps + 1))
        assert ep.mover_at(ep.steps) == tuple(ep.meta["dest"])


# ---------------------------------------------------------------------------------------------
# Pictures and words


def test_cell_spots_find_a_cube_where_it_is_drawn_and_a_cell_where_its_cube_would_be():
    world = R.blank()
    world.mass[0, 0, 8, 3] = 1.0
    world.mass[0, 0:3, 4, 12] = 1.0
    obs = W.observe(world)[0]
    spec = V.ViewSpec.clean()
    rng = np.random.default_rng(0)
    look = V.sample_look(rng, spec)
    images, _ = V.scene_views(obs, rng, spec, look=look)
    lone, empty = G.cell_spots(obs, look, spec.size, [(0, 8, 3), (0, 12, 6)])
    without = R.blank()
    without.mass[0, 0:3, 4, 12] = 1.0
    bare_images, _ = V.scene_views(W.observe(without)[0], rng, spec, look=look)
    changed = np.abs(np.asarray(images[0]).astype(int) - np.asarray(bare_images[0]).astype(int)).sum(-1) > 30
    ys, xs = np.nonzero(changed)
    assert changed[int(lone[1]), int(lone[0])] and abs(xs.mean() - lone[0]) < 4 and abs(ys.mean() - lone[1]) < 6
    filled = R.blank()
    filled.mass[0] = world.mass[0]
    filled.mass[0, 0, 12, 6] = 1.0
    again = G.cell_spots(W.observe(filled)[0], look, spec.size, [(0, 12, 6)])[0]
    assert abs(again[0] - empty[0]) < 1 and abs(again[1] - empty[1]) < 1
    assert lone[0] < 128 < G.cell_spots(obs, look, spec.size, [(2, 4, 12)])[0][0]  # left is low x from the front
    hidden = G.cell_spots(obs, look, spec.size, [(0, 3, 12)])[0]  # behind the tower, seen from the front
    assert hidden is None or hidden[2] < 40


def test_words_are_in_the_cameras_frame():
    assert G.corner_words((0, 15, 0), 0) == "the front left corner"
    assert G.corner_words((0, 15, 0), 2) == "the back right corner"
    assert G.corner_words((0, 15, 0), 1) == "the back left corner"  # a camera at the world's right: the world's front is its left, the world's left its far side
    left = W.DIRS.index((0, 0, -1))
    front = W.DIRS.index((0, 1, 0))
    climb = W.DIRS.index((1, 0, 1))
    assert G.move_words(left, 0) == "to the left"
    assert G.move_words(front, 0) == "toward me"
    assert G.move_words(front, 1) == "to the left"
    assert G.move_words(climb, 0) == "up and to the right"
    occ = np.zeros((16, 16, 16), bool)
    occ[0:3, 2, 2] = True
    occ[0, 14, 8] = True
    occ[0, 6, 13] = True
    assert set(G.unique_refs(occ, (14, 8))) == {"near"}
    assert set(G.unique_refs(occ, (2, 2))) == {"far", "left", "tall"}
    assert G.ref_phrase("near", 0) == "the cube nearest me"
    assert G.ref_phrase("left", 2) == "the top cube of the stack furthest to my left"


# ---------------------------------------------------------------------------------------------
# The kinds


def _ring_pixels(image):
    a = np.asarray(image).astype(int)
    return int((np.abs(a - np.array(G.RING)).sum(-1) < 40).sum())


def test_goal_image_shows_the_goal_last_and_acts_in_the_cameras_frame(grounding):
    rng = np.random.default_rng(1)
    for _ in range(6):
        ex = grounding.sample("goal_image", rng)
        assert len(ex.images) == 3 and ex.act and ex.context_text.count(T.IMAGES) == 1
        assert ex.instruction in G.GOAL_ASKS
        assert ex.codes.shape == (16, 16, 16) and ex.obs.shape[-3:] == (16, 16, 16)
        if ex.t == 0:
            assert ex.reply_text.startswith(T.THINK + "\nyou asked: ") and "the last picture" in ex.reply_text
        else:
            assert ex.reply_text == T.act_reply("") and "you asked: " in ex.context_text


def test_goal_image_codes_and_obs_are_the_episodes_turned(ground_dir):
    g = G.Grounding(None, dirs=[ground_dir], shares={"goal_image": 1.0}, held_share=0.0, thought_share=0.0)
    episodes = {e.meta["source"]: e for e in g.episodes}
    rng = np.random.default_rng(3)
    for _ in range(4):
        ex = g.sample("goal_image", rng)
        matches = [e for e in episodes.values() if e.meta["kind"] == "goal" and ex.t < e.steps and torch.equal(turn_codes(e.codes(ex.t), ex.turns), ex.codes) and torch.equal(turn_grid(e.observation(ex.t), ex.turns), ex.obs)]
        assert matches


def test_point_marks_the_cube_and_says_where_it_goes(grounding):
    rng = np.random.default_rng(2)
    seen = set()
    for _ in range(12):
        ex = grounding.sample("point", rng)
        seen.add(ex.ground)
        assert len(ex.images) == 2 and ex.act
        assert _ring_pixels(ex.images[0]) >= _ring_pixels(ex.images[1]) + 30
        if ex.ground == "point:words":
            assert re.search(r"corner|on top of", ex.instruction)
    assert seen <= {"point:mark", "point:words"} and "point:mark" in seen


def test_views_name_the_cube_relative_to_the_viewer(grounding):
    rng = np.random.default_rng(3)
    for _ in range(6):
        ex = grounding.sample("views", rng, tries=400)
        which = ex.ground.split(":")[1]
        assert len(ex.images) == 2 and ex.act and which in ex.instruction
        assert re.search(r"nearest me|farthest from me|my left|my right|tallest", ex.instruction)
        assert _ring_pixels(ex.images[0]) < 30 and _ring_pixels(ex.images[1]) < 30


def test_predict_answers_with_the_right_letters(grounding, tokenizer):
    rng = np.random.default_rng(4)
    seen = set()
    for _ in range(12):
        ex = grounding.sample("predict", rng)
        seen.add(ex.ground)
        assert not ex.act and ex.codes is None and ex.answer
        assert ex.reply_text == T.text_reply("", ex.answer + ".")
        assert G.score_text(ex, ex.reply_text) == 1.0 and G.score_text(ex, T.text_reply("", "z.")) == 0.0
        assert len(ex.images) == (2 if ex.ground == "predict:moved" else 1)
        ids, labels = T.tokenize(ex, tokenizer)
        assert (ids == tokenizer.convert_tokens_to_ids(T.VISION_START)).sum() == len(ex.images)
        assert tokenizer.decode(ids[labels != -100]).endswith(ex.answer + "." + T.END)
    assert seen == {"predict:moved", "predict:where"}


def test_predict_where_letters_the_cell_the_cube_lands_in(ground_dir):
    g = G.Grounding(None, dirs=[ground_dir], shares={"predict": 1.0}, held_share=0.0)
    rng = np.random.default_rng(5)
    found = 0
    while found < 4:
        ex = g.sample("predict", rng)
        if ex.ground != "predict:where":
            continue
        found += 1
        words = re.search(r"moves one cell (.+)\. which", ex.instruction).group(1)
        assert words and words in {G.move_words(c, ex.turns) for c in range(W.STAY) if bool(W.CUBE_MOVES[c])}


def test_tokenize_expands_three_images_in_order(grounding, tokenizer):
    ex = grounding.sample("goal_image", np.random.default_rng(6))
    ids, labels = T.tokenize(ex, tokenizer)
    start, pad = tokenizer.convert_tokens_to_ids(T.VISION_START), tokenizer.convert_tokens_to_ids(T.IMAGE_PAD)
    assert int((ids == start).sum()) == 3 and int((ids == pad).sum()) == sum(T.image_tokens(i) for i in ex.images)
    assert labels[-1].item() == tokenizer.convert_tokens_to_ids(T.ACT)
