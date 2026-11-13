import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from beebot.control import Control, resolve


def message(id, text):
    return json.dumps({"id": id, "time": 1, "event": "message", "message": text})


def run_fake_loop(control, steps=2000):
    """The trainer's shape without a model: poll when due, save on "save", stop on "stop"."""
    saved, step = [], 0
    while step < steps:
        command = control.poll() if control.due() else None
        if command == "save":
            saved.append(step)
        if command == "stop":
            saved.append(step)
            break
        step += 1
        time.sleep(0.001)
    return step, saved


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("BEEBOT_CONTROL_URL", raising=False)
    control = Control("")
    assert not control.on and not control.due() and control.poll() is None
    assert resolve("__CONTROL_URL__") == ""


def test_environment_fallback(monkeypatch):
    monkeypatch.setenv("BEEBOT_CONTROL_URL", "https://ntfy.sh/x/")
    assert resolve("") == "https://ntfy.sh/x"


def test_commands_and_since():
    seen = []

    def fetch(url, since, timeout):
        seen.append(since)
        return "\n".join([json.dumps({"event": "open"}), message("a1", "hello"), message("a2", "Save ")]) if len(seen) == 1 else message("a3", "stop")

    control = Control("https://ntfy.sh/t", fetcher=fetch, log=lambda text: None)
    assert control.poll() == "save"
    assert control.poll() == "stop"
    assert seen[1] == "a2"  # only newer messages are asked for


def test_network_failure_never_raises():
    lines = []

    def fetch(url, since, timeout):
        raise TimeoutError("down")

    control = Control("https://ntfy.sh/t", fetcher=fetch, log=lines.append)
    assert control.poll() is None and control.poll() is None
    assert control.failures == 2 and len(lines) == 2


def test_fake_training_loop_against_a_local_server():
    queue = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            since = self.path.split("since=")[1]
            ids = [json.loads(m)["id"] for m in queue]
            body = "\n".join(queue[ids.index(since) + 1 :] if since in ids else queue).encode()  # ntfy: messages after `since`
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    control = Control(f"http://127.0.0.1:{server.server_port}/topic", every_minutes=0.0005, timeout=2, log=lambda text: None)  # every 0.03 s
    threading.Timer(0.3, lambda: queue.append(message("m1", "save"))).start()
    threading.Timer(0.8, lambda: queue.append(message("m2", "stop"))).start()
    step, saved = run_fake_loop(control)
    server.shutdown()
    assert len(saved) == 2 and step < 2000  # one save, then the stop's final save, and the loop ended early


def test_unreachable_server_does_not_stop_the_loop():
    control = Control("http://127.0.0.1:9/topic", every_minutes=0.0005, timeout=0.5, log=lambda text: None)
    step, saved = run_fake_loop(control, steps=100)
    assert step == 100 and saved == []
