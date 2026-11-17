import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path


SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import kernel_control
import run_control
from kernel_control import KernelControl, relay

URL = "https://ntfy.sh/test-topic"
TEMPLATES = [SCRIPTS / "kaggle_lm/kernel_template.py", SCRIPTS / "kaggle_train/kernel_template.py", SCRIPTS / "kaggle_tpu/kernel_template.py", SCRIPTS.parents[1] / "v2/scripts/kaggle_voxel/kernel_template.py"]


def message(id, text):
    return json.dumps({"id": id, "time": 1, "event": "message", "message": text})


def sleeper(ignore_term=False):
    code = "import signal, time\n" + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "") + "print('ready', flush=True)\ntime.sleep(120)\n"
    process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
    process.stdout.readline()  # the handler is in place
    return process


def control(replies, **kw):
    """A KernelControl whose polls return the given texts in turn; exits recorded instead of taken."""
    replies = list(replies)
    exits, lines = [], []

    def fetch(url, since):
        reply = replies.pop(0) if replies else ""
        if isinstance(reply, Exception):
            raise reply
        return reply

    kw = {"kill_grace": 2, "fetch": fetch, "exit": exits.append, "log": lines.append, **kw}
    return KernelControl(URL, **kw), exits, lines


def test_off_with_the_placeholder(monkeypatch):
    monkeypatch.delenv("BEEBOT_CONTROL_URL", raising=False)
    off = KernelControl("__CONTROL_URL__", log=lambda text: None).start()
    assert not off.on and "BEEBOT_CONTROL_URL" not in os.environ
    assert not any(t.name == "kernel-control" for t in threading.enumerate())


def test_on_exports_the_url_for_trainers_and_never_logs_it(monkeypatch):
    monkeypatch.delenv("BEEBOT_CONTROL_URL", raising=False)
    lines = []
    on = KernelControl(URL + "/", every=3600, fetch=lambda url, since: "", log=lines.append).start()
    assert on.on and os.environ["BEEBOT_CONTROL_URL"] == URL
    assert not any("test-topic" in line for line in lines)
    thread = next(t for t in threading.enumerate() if t.name == "kernel-control")
    assert thread.daemon  # never keeps the kernel alive


def test_poll_reads_words_and_moves_since():
    seen = []

    def fetch(url, since):
        seen.append(since)
        return "\n".join([json.dumps({"event": "open"}), message("m1", "hello"), message("m2", " Kill  2B "), "not json"])

    kc = KernelControl(URL, fetch=fetch, log=lambda text: None)
    assert kc.poll() == ["hello", "kill 2b"]
    kc.poll()
    assert seen[1] == "m2"


def test_poll_failures_never_raise_and_change_nothing():
    kc, exits, lines = control([TimeoutError("down")] * 12)
    process = sleeper()
    kc.track("a", process)
    for _ in range(12):
        kc.step()
    assert process.poll() is None and exits == [] and kc.failures == 12
    assert len(lines) == 2  # the first failure and the tenth
    process.kill()


def test_kill_terminates_every_trainer_and_ends_the_kernel():
    kc, exits, lines = control([message("m1", "kill")])
    a, b = sleeper(), sleeper()
    kc.track("run-a", a)
    kc.track("run-b", b)
    kc.step()
    assert a.poll() is not None and b.poll() is not None
    assert exits == [0]
    event = json.loads(lines[0])
    assert event["event"] == "kernel_control" and event["command"] == "kill" and event["runs"] == ["run-a", "run-b"]


def test_kill_escalates_to_sigkill_after_the_grace():
    kc, exits, _ = control([message("m1", "kill")], kill_grace=1)
    stubborn = sleeper(ignore_term=True)
    kc.track("stubborn", stubborn)
    t0 = time.time()
    kc.step()
    assert stubborn.returncode == -9 and 1 <= time.time() - t0 < 10 and exits == [0]


def test_kill_one_run_leaves_the_others():
    kc, exits, _ = control([message("m1", "kill 2b")])
    small, large = sleeper(), sleeper()
    kc.track("scale-curve-0.8b", small)
    kc.track("scale-curve-2b", large)
    kc.step()
    assert large.poll() is not None and small.poll() is None and exits == []
    small.kill()


def test_stop_waits_for_control_trainers_and_terminates_the_rest(tmp_path):
    (tmp_path / "beebot").mkdir()
    (tmp_path / "beebot" / "graceful.py").write_text("from beebot.control import Control\n")
    (tmp_path / "beebot" / "plain.py").write_text("print('no control')\n")
    kc, exits, lines = control([message("m1", "stop"), ""], src=str(tmp_path), stop_grace=2)
    graceful, plain = sleeper(), sleeper()
    kc.track("g", graceful, "beebot.graceful")
    kc.track("p", plain, "beebot.plain")
    kc.step()
    assert kc.stopping and plain.poll() is not None and graceful.poll() is None and exits == []
    time.sleep(2.1)
    kc.step()  # past the stop grace: the late one is terminated too
    assert graceful.poll() is not None and exits == []


def test_stop_before_any_trainer_is_kill():
    kc, exits, _ = control([message("m1", "stop")])
    kc.step()
    assert exits == [0]


def test_relay_prefixes_lines_and_reports_exits(tmp_path, capsys):
    (tmp_path / "a.out").write_text('{"step": 1, "loss": 2.0}\nWarning: noise\nlast without newline')
    done = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    done.wait()
    relay([("a", done)], noise=("Warning",), folder=str(tmp_path), every=0)
    out = capsys.readouterr().out.splitlines()
    assert out == ['[a] {"step": 1, "loss": 2.0}', "[a] last without newline", "exit a 3"]


def test_prepare_inlines_the_switch_and_fills_the_url(tmp_path, monkeypatch):
    monkeypatch.setattr(run_control, "FOLDER", tmp_path)
    for template in TEMPLATES:
        text = template.read_text()
        assert "__CONTROL_URL__" in text and run_control.MARK in text, template
        pushed = run_control.prepare(text, "some-run")
        assert run_control.MARK not in pushed and "class KernelControl" in pushed and "__CONTROL_URL__" not in pushed
        url = run_control.load("some-run")
        assert pushed.count(url) == 1  # only in the CONTROL_URL line
        filled = re.sub(r"__(RUNS)__", "[]", pushed)
        for name in ("DDP", "PLAY", "FLA"):
            filled = filled.replace(f"__{name}__", "False")
        filled = filled.replace("__EPISODES__", "0").replace("__DEMO_SCENES__", "0")
        compile(filled, str(template), "exec")
    assert oct((tmp_path / "some-run.json").stat().st_mode)[-3:] == "600"


def test_list_never_shows_urls(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_control, "FOLDER", tmp_path)
    url = run_control.create("run-x")
    monkeypatch.setattr(sys, "argv", ["run_control.py", "list"])
    run_control.main()
    out = capsys.readouterr().out
    assert "run-x" in out and url not in out and "ntfy" not in out


def test_other_words_are_harmless():
    kc, exits, _ = control([message("m1", "hello"), message("m2", "save")])
    process = sleeper()
    kc.track("a", process)
    kc.step()
    kc.step()
    assert process.poll() is None and exits == []
    process.kill()
