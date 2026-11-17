import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import kaggle_watch
from kaggle_watch import Watch, arguments, parse, read_replay, replay, timed_lines

ROOT = Path(__file__).parents[2]


def cfg(*extra):
    return arguments(["--replay", "x", *extra])


def progress(run, step, loss, arm=None):
    payload = {"step": step, "minutes": step / 10, "demo_loss": loss, **({"arm": arm} if arm else {})}
    return f"[{run}] {json.dumps(payload)}"


def run_lines(watch, lines, t=0.0):
    for line in lines:
        watch.add(t, line)
    return watch.check()


def kaggle_log(path, timed):
    """A log in Kaggle's downloaded form: a JSON list, one entry per line."""
    entries = [json.dumps({"stream_name": "stdout", "time": t, "data": line + "\n"}) for t, line in timed]
    path.write_text("[" + "\n,".join(entries) + "]")
    return path


def test_parse_prefixed_and_plain_lines():
    assert parse('[scale-curve-2b] {"step": 50, "demo_loss": 2.1}') == ("scale-curve-2b", '{"step": 50, "demo_loss": 2.1}', {"step": 50, "demo_loss": 2.1})
    assert parse("[run] [transformers] note")[:2] == ("run", "[transformers] note")
    assert parse("torch 2.11 gpus 2") == (None, "torch 2.11 gpus 2", None)


def test_timed_lines_hold_back_the_unfinished_line():
    entries = [{"time": 1.0, "data": "a\nb"}, {"time": 2.0, "data": "c\n"}, {"time": 3.0, "data": "d"}]
    assert timed_lines(entries, final=False) == [(1.0, "a"), (2.0, "bc")]
    assert timed_lines(entries, final=True)[-1] == (3.0, "d")


def test_a_traceback_names_the_exception():
    watch = Watch(cfg())
    alert = run_lines(watch, ["[r1] Traceback (most recent call last):", '[r1]   File "x.py", line 1', "[r2] unrelated", "[r1] torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB"])
    assert alert.check == "a" and alert.code == 10 and "CUDA OOM in r1" in alert.reason and "Tried to allocate" in alert.reason


def test_a_argparse_error_and_nonzero_exit():
    assert "argparse error" in run_lines(Watch(cfg()), ["[r] usage: expert_sft.py [-h]", "[r] expert_sft.py: error: the following arguments are required: command"]).reason
    assert run_lines(Watch(cfg()), ["exit r 0"]) is None
    assert run_lines(Watch(cfg()), ["exit r -9"]).reason == "r exited with code -9"


def test_a_ignores_runs_the_kill_switch_ended_and_the_final_tails():
    watch = Watch(cfg())
    lines = ['{"event": "kernel_control", "command": "kill", "runs": ["r"], "reason": "kill r received"}', "exit r -15", "==== /kaggle/working/logs/r.out", "Traceback (most recent call last):", "ValueError: old"]
    assert run_lines(watch, lines) is None


def test_b_no_step_within_the_limit():
    watch = Watch(cfg("--first-step-minutes", "30"))
    run_lines(watch, ['[r] {"arm": "x", "step": 0, "minutes": 0.0}'])
    assert watch.tick(29 * 60) is None
    alert = watch.tick(30 * 60)
    assert alert.check == "b" and alert.code == 11
    run_lines(watch, [progress("r", 50, 2.0)])
    assert watch.tick(31 * 60) is None


def test_c_nan_and_rise_after_warmup():
    nan = run_lines(Watch(cfg()), [progress("r", 50, 2.0), '[r] {"step": 100, "loss": NaN}'])
    assert nan.check == "c" and "nan" in nan.reason
    watch = Watch(cfg("--warmup", "100", "--rise-points", "3"))
    assert run_lines(watch, [progress("r", 50, 9.0), progress("r", 100, 1.5), progress("r", 150, 1.6), progress("r", 200, 1.4), progress("r", 250, 1.6), progress("r", 300, 1.7)]) is None  # warmup ignored; reset by a lower point
    alert = run_lines(watch, [progress("r", 350, 1.8)])
    assert alert.check == "c" and alert.code == 12 and "1.5" in alert.reason


def test_d_fires_once_one_arm_is_worse_at_the_last_points():
    watch = Watch(cfg("--compare", "0.8b", "2b", "--separate", "0.05", "--after", "300", "--points", "3"))
    small = [progress("scale-curve-0.8b", s, l, "0.8b") for s, l in [(250, 1.40), (300, 1.30), (350, 1.25), (400, 1.20)]]
    large = [progress("scale-curve-2b", s, l, "2b") for s, l in [(250, 1.70), (300, 1.33), (350, 1.40), (400, 1.40)]]
    assert run_lines(watch, small + large[:3]) is None  # 300 is within 5%
    alert = run_lines(watch, large[3:])
    assert alert is None  # 300 (2.3%), 350 (12%), 400 (17%): not all three
    alert = run_lines(watch, [progress("scale-curve-0.8b", 450, 1.1, "0.8b"), progress("scale-curve-2b", 450, 1.3, "2b")])
    assert alert.check == "d" and alert.code == 13 and alert.run == "scale-curve-2b"


def test_d_needs_both_arms():
    watch = Watch(cfg("--compare", "a", "b", "--after", "0", "--points", "1"))
    assert run_lines(watch, [progress("run-a", 10, 1.0), progress("run-a", 20, 1.0)]) is None


def test_a_fired_check_is_not_fired_again_after_a_restart():
    lines = ["[r] Traceback (most recent call last):", "[r] ValueError: x"]
    first = Watch(cfg())
    alert = run_lines(first, lines)
    first.remember(alert)
    again = Watch(cfg(), first.fired)
    assert run_lines(again, lines) is None
    assert run_lines(again, ["[r] Traceback (most recent call last):"]).check == "a"  # a new one still fires


def test_replay_with_simulated_time(tmp_path, capsys):
    timed = [(60.0, "torch 2.11"), *[(600.0 + 60 * k, progress("scale-curve-0.8b", 50 * k, 1.5 - 0.05 * k, "0.8b")) for k in range(1, 12)], *[(630.0 + 60 * k, progress("scale-curve-2b", 50 * k, 1.5, "2b")) for k in range(1, 12)]]
    log = kaggle_log(tmp_path / "x.log", sorted(timed))
    assert read_replay(log)[0] == (60.0, "torch 2.11")
    code = replay(arguments(["--replay", str(log), "--compare", "0.8b", "2b", "--after", "300", "--points", "4", "--every", "60"]))
    out = capsys.readouterr().out
    assert code == 13 and "replay d at" in out and "scale-curve-2b loss worse" in out


def test_replay_of_a_finished_run_is_a_summary(tmp_path, capsys):
    log = kaggle_log(tmp_path / "x.log", [(10.0, progress("r", 50, 1.0)), (20.0, progress("r", 100, 0.9)), (30.0, "exit r 0")])
    assert replay(arguments(["--replay", str(log)])) == 0
    assert "ended (COMPLETE)" in capsys.readouterr().out


def test_kill_on_in_replay_never_sends(monkeypatch):
    monkeypatch.setattr(kaggle_watch.run_control, "send", lambda *a: pytest.fail("sent in replay"))
    alert = kaggle_watch.Alert("d", "worse", 60.0, "d", run="scale-curve-2b")
    line = kaggle_watch.act(alert, cfg("--kill-on", "a,d"), "slug", replay=True)
    assert "would send 'kill scale-curve-2b'" in line


def test_kill_on_live_sends_kill(monkeypatch):
    sent = []
    monkeypatch.setattr(kaggle_watch.run_control, "send", lambda run, word: sent.append((run, word)))
    line = kaggle_watch.act(kaggle_watch.Alert("c", "nan", 60.0, "c:r"), cfg("--kill-on", "c"), "slug", replay=False)
    assert sent == [("slug", "kill")] and "[sent 'kill']" in line


@pytest.mark.parametrize(("slug", "code"), [("beebot-scale-curve-a", 13), ("beebot-scale-curve-b", 10)])
def test_the_real_logs(slug, code, capsys):
    log = ROOT / "runs" / "kaggle" / slug / f"{slug}.log"
    if not log.exists():
        pytest.skip("local Kaggle log not present")
    assert replay(arguments(["--replay", str(log), "--compare", "0.8b", "2b", "--separate", "0.05", "--after", "300"])) == code
