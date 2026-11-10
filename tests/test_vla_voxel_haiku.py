"""The Haiku label pipeline (vla/voxel/haiku.py) with a fake provider: no model is ever called.
Items join to demo stores by (seed, index), answers are checked and verified, runs resume, calls
back off, the test set stays out of training, and dry runs write nothing."""

import argparse
import json

import numpy as np
import pytest

from beebot.vla.voxel import demos as D
from beebot.vla.voxel import haiku as H
from beebot.vla.voxel import suite as S


@pytest.fixture(scope="module")
def items():
    return H.drawn_items({"seed": 3}, 40)


def no_sleep(delays):
    return lambda seconds: delays.append(seconds)


def build(tmp_path, items, name="train"):
    directory = tmp_path / name
    directory.mkdir(parents=True)
    with open(directory / "items.jsonl", "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    (directory / "config.json").write_text(json.dumps({"seed": items[0]["seed"]}))
    (directory / "menu.json").write_text(json.dumps({"menu": H.menu_from([item["record"] for item in items])}))
    return directory


def test_items_join_to_a_demo_store_by_seed_and_index(tmp_path):
    store = tmp_path / "store"
    D.write(store, 2, seed=5, chunk=2)
    attempts = H.read_jsonl(store / "attempts.jsonl")
    drawn_tasks = H.drawn_items({"seed": 5, "chunk": 2}, len(attempts))
    assert [(x["record"]["family"], x["record"]["kind"]) for x in drawn_tasks] == [(a["family"], a["kind"]) for a in attempts]
    export_items = H.export(tmp_path / "joined", {"seed": 0, "chunk": 2}, None, store=store)
    assert [x["index"] for x in export_items] == [n for n, a in enumerate(attempts) if a["e"] >= 0]
    assert [x["e"] for x in export_items] == [a["e"] for a in attempts if a["e"] >= 0]
    labels = [json.loads(line) for line in (store / "labels.jsonl").read_text().splitlines()]
    assert [x["record"]["family"] for x in export_items] == [row["family"] for row in labels]
    with pytest.raises(ValueError):  # another chunk size draws other tasks
        H.export(tmp_path / "wrong", {"seed": 0, "chunk": 1}, None, store=store)


def test_records_are_json_with_one_subgoal_per_plan_segment(items):
    rng = np.random.default_rng(3)
    world, metas, tasks = S.sample_tasks(rng, 30, "train")
    for i, task in enumerate(tasks):
        record = H.canonical_record(task)
        assert set(record) == {"family", "kind", "plain", "spec", "subgoals"}
        assert len(record["subgoals"]) == len(task.plan)
        json.dumps(record)
        if task.spec["kind"] == "eat" and task.spec.get("order"):
            assert record["subgoals"] == [f"eat the {c} cheese" for c in task.spec["order"]]
    for item in items:
        assert H.same_spec(item["record"]["spec"], item["record"]["spec"])
        assert {c["colour"] for c in item["scene"]["cheeses"]} <= set(S.TRAIN_COLOURS)


def test_scene_summary_reads_terrain_and_cheeses():
    rng = np.random.default_rng(0)
    world, metas = S.scenes(rng, 6, layout="gap", cheeses=(2, 3))
    for i, meta in enumerate(metas):
        scene = H.scene_summary(world, i)
        is_transposed, start, width, top = meta["gap"]
        trench = scene["terrain"]["trench"]
        assert trench["width"] == width and trench["depth"] == top
        assert trench["runs"] == ("left to right" if is_transposed else "back to front")
        assert sorted(c["colour"] for c in scene["cheeses"]) == sorted(c for _, c in S.present(world, i))
    flat, _ = S.scenes(rng, 1, layout="flat")
    assert H.scene_summary(flat, 0)["terrain"] == {"layout": "flat floor"}


def test_same_spec_ignores_set_order_and_number_spelling():
    want = {"kind": "eat", "eat": ["blue", "red"], "any_of": [], "order": ["red", "blue"]}
    assert H.same_spec({"kind": "eat", "eat": ["Red", "blue"], "any_of": [], "order": ["red", "blue"]}, want)
    assert not H.same_spec({"kind": "eat", "eat": ["red", "blue"], "any_of": [], "order": ["blue", "red"]}, want)
    assert not H.same_spec({"kind": "eat", "eat": ["red"], "any_of": [], "order": []}, want)
    assert H.same_spec({"kind": "parallel", "jobs": [["tower", "4"], ["region", "back"]]}, {"kind": "parallel", "jobs": [["region", "back"], ["tower", 4]]})
    assert not H.same_spec("eat", want)


def eat_item(colours, order):
    scene = {"gravity": True, "terrain": {"layout": "flat floor"}, "robots": {"count": 50}, "cheeses": [{"colour": c, "area": "center", "on": "floor", "x": 1, "y": 1, "z": 0} for c in colours]}
    record = {"family": "logic", "kind": "eat:ordered", "plain": f"eat the {order[0]} cheese first, then the {order[1]} cheese", "spec": {"kind": "eat", "eat": sorted(order), "any_of": [], "order": order}, "subgoals": [f"eat the {c} cheese" for c in order]}
    return {"id": "s1-0000001", "seed": 1, "index": 1, "record": record, "scene": scene}


def test_checks_reject_invented_contents_held_out_words_and_wrong_subgoals():
    item = eat_item(["red", "blue", "green"], ["red", "blue"])
    good = {"text": "red one first and then the blue", "thought": "you asked: red one first and then the blue. i see red, blue and green cheeses. so red, then blue. first: the red one.", "switches": ["now the blue one"]}
    pair, reason = H.check_follow(good, item, "close")
    assert reason is None and pair["switches"] == ["now the blue one"] and pair["done"] == "done."
    cases = {
        "colour orange not in scene": {**good, "thought": "you asked: x. i see red, orange and blue cheeses. first: the red one."},
        "held-out words": {**good, "text": "eat the crimson one, then blue"},
        "switches for": {**good, "switches": []},
        "not named in order": {**good, "switches": ["now the green one"]},
        "trench not in scene": {**good, "thought": "you asked: x. i see red and blue by the trench. first: the red one."},
        "words besides": {**good, "thought": "you asked: x. " + "very " * 60 + "first: the red one."},
    }
    for expect, answer in cases.items():
        pair, reason = H.check_follow(answer, item, "close")
        assert pair is None and expect in reason, (expect, reason)
    assert H.leaks("eat the warm coloured ones") and H.leaks("吃红色") and not H.leaks("eat the cool coloured ones")
    refusal = {"kind": "refuse", "text": "eat the black cheese", "thought": "you asked: eat the black cheese. i see red and blue. there is no black one. so i stay.", "reply": "there is no black cheese here."}
    pair, reason = H.check_refusal(refusal, item)
    assert reason is None and pair["kind"] == "refuse"
    assert H.check_refusal({**refusal, "kind": "maybe"}, item)[0] is None


def test_generate_and_verify_end_to_end_resume_and_load(tmp_path, items):
    directory = build(tmp_path, items[:12])
    fake = H.FakeHaiku(malformed_ids={items[0]["id"]}, dropped_ids={items[1]["id"]})
    provider = H.SpecHints(fake, items)
    rows = H.generate(directory, provider, batch=4, workers=3, refusal_share=0.3, log_every=60, sleep=no_sleep([]))
    assert {row["id"] for row in rows} == {item["id"] for item in items[2:12]} | {items[0]["id"]}
    assert H.read_jsonl(directory / "generated.jsonl")[0]["id"] in {row["id"] for row in rows}
    malformed_row = [row for row in rows if row["id"] == items[0]["id"]][0]
    assert malformed_row["pairs"] == [] and malformed_row["rejects"][0]["reason"] == "malformed item"
    assert {row["id"] for row in H.read_jsonl(directory / "failures.jsonl")} == {items[1]["id"]}
    calls = fake.calls
    assert H.generate(directory, provider, batch=4, workers=3) == [] or fake.calls > calls  # the dropped item is retried, nothing else
    assert sum(1 for row in H.read_jsonl(directory / "generated.jsonl") if row["id"] == items[2]["id"]) == 1
    result = H.verify(directory, provider, batch=5, workers=2)
    assert result["kept"] > 0 and set(result["by_rung"]) <= set(H.RUNGS) | {H.REFUSAL, "*"}  # "*": a malformed item, every rung
    assert result["by_rung"]["close"]["verify_keep_rate"] == 1.0
    for row in H.read_jsonl(directory / "verified.jsonl"):
        assert H.same_spec(row["read_as"], row["spec"])
    assert any(row["rung"] == H.REFUSAL and row["kind"] == "refuse" for row in H.read_jsonl(directory / "verified.jsonl"))
    calls = fake.calls
    H.verify(directory, provider)
    assert fake.calls == calls  # every pair already has a verdict
    labels = H.Labels("train", tmp_path)
    some = H.read_jsonl(directory / "verified.jsonl")[0]
    choices = labels.choices(some["seed"], some["index"])
    assert choices and {"instruction", "thought", "switches", "done", "rung"} == set(choices[0])
    assert labels.choices(-1, -1) == []


def test_verify_prompts_hold_only_instruction_and_scene_and_mismatches_are_rejected(tmp_path, items):
    directory = build(tmp_path, items[:4])
    fake = H.FakeHaiku()
    H.generate(directory, H.SpecHints(fake, items), batch=4, workers=1)
    prompts = []

    def wrong(prompt):
        prompts.append(prompt)
        entries = [json.loads(line) for line in prompt.split("one JSON object per line):\n", 1)[1].splitlines() if line.strip()]
        return json.dumps({e["id"]: {"kind": "tower", "height": 99} for e in entries})

    H.verify(directory, wrong, batch=50, workers=1)
    for prompt in prompts:
        for line in prompt.split("one JSON object per line):\n", 1)[1].splitlines():
            assert set(json.loads(line)) == {"id", "instruction", "scene"}
    rejects = H.read_jsonl(directory / "rejected.jsonl")
    assert rejects and all(row["reason"] == "read as a different task" for row in rejects)
    assert not H.read_jsonl(directory / "verified.jsonl")
    pair_ids = [row["pair_id"] for row in rejects]
    assert len(pair_ids) == len(set(pair_ids))
    for prompt in prompts:  # an item's instructions never share a call
        ids = [json.loads(line)["id"].split(":")[0] for line in prompt.split("one JSON object per line):\n", 1)[1].splitlines()]
        assert len(ids) == len(set(ids))


def test_calls_back_off_and_retry_on_errors_and_replies_without_json(tmp_path, items):
    directory = build(tmp_path, items[:3])
    delays = []
    fake = H.FakeHaiku(fail_calls=2, garbage_calls=1)
    H.generate(directory, H.SpecHints(fake, items), batch=10, workers=1, sleep=no_sleep(delays))
    assert len(delays) == 3 and delays[0] < delays[2]
    assert len(H.read_jsonl(directory / "generated.jsonl")) == 3
    calls = H.read_jsonl(directory / "calls.jsonl")
    assert [row["error"] is None for row in calls] == [False, False, False, True]
    directory = build(tmp_path, items[:2], "broken")
    H.generate(directory, H.FakeHaiku(fail_calls=100), batch=10, workers=1, passes=1, sleep=no_sleep([]))
    assert not H.read_jsonl(directory / "generated.jsonl")
    assert len(H.read_jsonl(directory / "failures.jsonl")) == 2


def test_meter_writes_a_throughput_line_on_schedule(tmp_path):
    log = tmp_path / "throughput.jsonl"
    with H.Meter("generate", 10, log, every=0.05) as meter:
        meter.add(items=5, calls=1)
        import time

        time.sleep(0.2)
    lines = H.read_jsonl(log)
    assert len(lines) >= 3 and lines[-1]["items"] == 5 and {"items_per_min", "calls_per_min", "failures"} <= set(lines[-1])


def test_testset_is_held_out_frozen_and_refused_by_the_training_loader(tmp_path, items, monkeypatch):
    args = argparse.Namespace(seed=0, chunk=24, robots=None, counts="uniform", fit=False, rule=None, apart=False, cube=False, free_cube=False, dry_run=False, dry_items=4, items=6, batch=None, verify_batch=25, workers=2, log_every=60)
    fake = H.FakeHaiku()
    test = tmp_path / H.TEST_NAME
    H.build_testset(test, args, lambda test_items: H.SpecHints(fake, test_items))
    assert (test / H.MARKER).exists() and (test / H.FROZEN).exists()
    assert {row["seed"] for row in H.read_jsonl(test / "items.jsonl")} == {H.TEST_SEED}
    test_pairs = H.read_jsonl(test / "verified.jsonl")
    assert test_pairs and {row["rung"] for row in test_pairs} <= set(H.TEST_RUNGS)
    for row in H.read_jsonl(test / "generated.jsonl"):
        for pair in row["pairs"]:
            assert not H.mentions(pair["instruction"], ["red", "green", "blue", "yellow", "orange", "purple", "white", "cyan"]), pair
    with pytest.raises(FileExistsError):
        H.build_testset(test, args, lambda test_items: H.SpecHints(fake, test_items))
    with pytest.raises(AssertionError):
        H.Labels(H.TEST_NAME, tmp_path)
    with pytest.raises(ValueError):
        H.export(tmp_path / "train", {"seed": H.TEST_SEED}, 2)
    directory = build(tmp_path, items[:3])
    leak = {**test_pairs[0], "id": items[0]["id"], "seed": items[0]["seed"], "index": items[0]["index"], "pair_id": items[0]["id"] + ":close", "rung": "close", "thought": "t", "switches": [], "done": "done."}
    H.append(directory / "verified.jsonl", leak)
    with pytest.raises(AssertionError):
        H.Labels("train", tmp_path)


def test_dry_runs_print_prompts_and_write_nothing_under_root(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(H.tempfile, "tempdir", str(tmp_path / "scratch"))
    (tmp_path / "scratch").mkdir()
    root = tmp_path / "root"
    H.main(["pilot", "--dry-run", "--dry-items", "3", "--root", str(root), "--workers", "1"])
    out = capsys.readouterr().out
    assert "PROMPT" in out and "REPLY (fake)" in out and "PARSED" in out and "KEPT" in out
    assert "You write training text" in out and "Someone typed an instruction" in out
    H.main(["export", "--dry-run", "--name", "x", "--root", str(root)])
    assert not root.exists()
