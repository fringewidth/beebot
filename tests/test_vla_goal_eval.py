"""Requested-events scoring (goal_eval.requested_scores and run(requested=...))."""

import math

from beebot.vla import goal_eval
from beebot.vla.data import make_windows
from beebot.vla.facts import window_facts
from beebot.vla.train import generate_scripted


def _store_and_windows():
    store = generate_scripted(16, 8, seed=0, device="cpu", batch_size=8)
    return store, make_windows(store, 16)[:16]


def _atoms(store, window):
    e, w0, w1 = (int(x) for x in window)
    static = {k: store[k][e].numpy() for k in ("walls", "cheese_color", "box_rect", "box_color")}
    return window_facts(static, goal_eval.stored_snapshot(store, e, w0), goal_eval.stored_snapshot(store, e, w1)).atoms


def test_replay_meets_what_its_own_window_asks_and_nothing_asked_scores_nan():
    store, windows = _store_and_windows()
    every = [sorted(_atoms(store, w), key=str) for w in windows]
    result = goal_eval.run(None, store, windows, "cpu", mode="replay", requested=every)
    asked = [i for i, atoms in enumerate(every) if atoms]
    assert asked, "the scripted windows should hold some events"
    # Asked for every event of its window, the best fraction is at least the facts score (a
    # snapshot's recall of the target bounds its Jaccard from above), and replay gets most of it.
    for i in asked:
        assert result["requested_best"][i].item() >= result["fact_score"][i].item() - 1e-6
    assert result["requested_best"][asked].mean().item() > 0.7
    nothing = goal_eval.run(None, store, windows, "cpu", mode="replay", requested=[[] for _ in windows])
    assert all(math.isnan(x) for x in nothing["requested_best"].tolist())
    assert "requested_best" in goal_eval.summary(result) and "requested_best" not in goal_eval.summary(goal_eval.run(None, store, windows, "cpu", mode="zero"))


def test_disruptive_events_count_only_when_not_asked():
    """On the recorded end frame itself: everything asked is all met with nothing unasked; nothing
    asked flags exactly the windows whose events include a disruptive one."""
    store, windows = _store_and_windows()
    for window in windows:
        e, w0, w1 = (int(x) for x in window)
        static = {k: store[k][e].numpy() for k in ("walls", "cheese_color", "box_rect", "box_color")}
        start, end = goal_eval.stored_snapshot(store, e, w0), goal_eval.stored_snapshot(store, e, w1)
        atoms = _atoms(store, window)
        if atoms:
            assert goal_eval.requested_scores(static, start, [end], atoms) == (1.0, 1.0, False)
        best, last, unasked = goal_eval.requested_scores(static, start, [end], set())
        assert best is None and last is None
        assert unasked == any(a[0] in goal_eval.DISRUPTIVE for a in atoms)
