"""Send a command to a running job that watches its ntfy control topic (beebot.control).

    python scripts/run_control.py new RUN        # make the run's random topic; stored in ~/.config/beebot/runs/RUN.json (0600)
    python scripts/run_control.py url RUN        # print the control URL (a secret: for injecting into a kernel, never commit)
    python scripts/run_control.py stop RUN       # the job saves a checkpoint and resume state, then exits
    python scripts/run_control.py save RUN       # the job checkpoints now and goes on

A command arrives within the job's control_every_minutes (default 5).
"""

import json
import os
import secrets
import sys
import urllib.request
from pathlib import Path

FOLDER = Path.home() / ".config" / "beebot" / "runs"
SERVER = "https://ntfy.sh"


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


def send(run, word):
    request = urllib.request.Request(load(run), data=word.encode(), method="POST", headers={"User-Agent": "beebot-control"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read())["id"]


def main():
    command, run = sys.argv[1], sys.argv[2]
    if command == "new":
        create(run)
        print(f"topic stored in {path(run)}")
    elif command == "url":
        print(load(run))
    elif command in ("stop", "save"):
        print(f"sent {command} to {run} (message {send(run, command)})")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
