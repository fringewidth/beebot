"""Send a command to a running job that watches its ntfy control topic (a Kaggle kernel's kill switch,
v1/scripts/kernel_control.py, or a trainer through beebot.control).

    python v1/scripts/run_control.py new RUN        # make the run's random topic; stored in ~/.config/beebot/runs/RUN.json (0600)
    python v1/scripts/run_control.py url RUN        # print the control URL (a secret: for injecting into a kernel, never commit)
    python v1/scripts/run_control.py list           # the known runs (names and dates, never the URLs)
    python v1/scripts/run_control.py kill RUN       # Kaggle kernel: terminate every trainer and end the session (quota stops)
    python v1/scripts/run_control.py kill RUN ARM   # Kaggle kernel: terminate only that trainer (e.g. 2b or scale-curve-2b)
    python v1/scripts/run_control.py stop RUN       # graceful: beebot.control trainers save and exit, others are terminated
    python v1/scripts/run_control.py save RUN       # beebot.control trainers checkpoint now and go on

vla_kaggle.py makes the topic automatically, named after the kernel slug, so RUN is the slug. A Kaggle
kernel polls every 60 s, a beebot.control trainer every control_every_minutes (default 5).
"""

import json
import os
import secrets
import sys
import time
import urllib.request
from pathlib import Path

FOLDER = Path.home() / ".config" / "beebot" / "runs"
SERVER = "https://ntfy.sh"
KERNEL_CONTROL = Path(__file__).with_name("kernel_control.py")
MARK = "# __KERNEL_CONTROL__"


def path(run):
    return FOLDER / f"{run}.json"


def create(run):
    """The run's control URL, made once: the topic name is 32 random bytes."""
    if path(run).exists():
        return load(run)
    FOLDER.mkdir(parents=True, exist_ok=True)
    url = f"{SERVER}/beebot-{secrets.token_urlsafe(24)}"
    descriptor = os.open(path(run), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as f:
        json.dump({"run": run, "control_url": url}, f)
    return url


def load(run):
    return json.loads(path(run).read_text())["control_url"]


def inject(text, run):
    """A kernel's text with its __CONTROL_URL__ placeholder replaced by the run's URL (made if new).
    Only the pushed copy has it: kernels are private, the repo's files keep the placeholder."""
    assert "__CONTROL_URL__" in text, "the kernel has no __CONTROL_URL__ placeholder"
    return text.replace("__CONTROL_URL__", create(run))


def inline(text):
    """A kernel's text with the line '# __KERNEL_CONTROL__' replaced by kernel_control.py (the kill switch)."""
    return text.replace(MARK, KERNEL_CONTROL.read_text()) if MARK in text else text


def prepare(text, run=None):
    """A kernel ready to push: the kill switch inlined and, with a run name, its control URL filled in."""
    text = inline(text)
    return inject(text, run) if run else text


def send(run, word):
    request = urllib.request.Request(load(run), data=word.encode(), method="POST", headers={"User-Agent": "beebot-control"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read())["id"]


def runs():
    """[(run, made at)] of the known runs, newest first. Never the URLs."""
    found = [(p.stem, p.stat().st_mtime) for p in FOLDER.glob("*.json")] if FOLDER.exists() else []
    return sorted(found, key=lambda r: -r[1])


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "list":
        for run, made in runs():
            print(f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(made))}  {run}")
        return
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    command, run = sys.argv[1], sys.argv[2]
    if command == "new":
        create(run)
        print(f"topic stored in {path(run)}")
    elif command == "url":
        print(load(run))
    elif command in ("stop", "save", "kill"):
        word = " ".join([command, *sys.argv[3:4]]) if command == "kill" else command
        print(f"sent {word!r} to {run} (message {send(run, word)})")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
