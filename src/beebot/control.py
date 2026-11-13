"""Remote control of a long run through a ntfy.sh topic (no account: the random topic name is the secret).

The trainer asks `Control.poll()` every few minutes; it answers "stop" (save a checkpoint and the
resume state, then exit cleanly), "save" (checkpoint now, continue) or None. Network failures never
raise: the poll times out quickly, logs one line and returns None, so training goes on.

The URL is `https://ntfy.sh/<topic>`. It comes from the `--control_url` argument or, preferably,
the environment variable BEEBOT_CONTROL_URL (an argument shows up in logs and config.json; the
variable does not). Only messages published after the run started count, so a run resumed after a
"stop" is not stopped again by the old message. scripts/run_control.py publishes the commands.
"""

import json
import os
import time
import urllib.parse
import urllib.request

COMMANDS = ("stop", "save")
ENV = "BEEBOT_CONTROL_URL"


def resolve(url=""):
    """The control URL: the argument, else the environment, else "" (off). A kernel template
    placeholder that was never filled in ("__...") counts as off."""
    url = (url or os.environ.get(ENV, "")).strip()
    return "" if not url or url.startswith("__") else url.rstrip("/")


def fetch(url, since, timeout):
    """The text of ntfy's JSON lines for messages after `since` (a unix time or a message id)."""
    request = urllib.request.Request(f"{url}/json?poll=1&since={urllib.parse.quote(str(since))}", headers={"User-Agent": "beebot-control"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode(errors="replace")


class Control:
    def __init__(self, url="", every_minutes=5.0, timeout=10.0, log=print, fetcher=fetch):
        self.url = resolve(url)
        self.every = every_minutes * 60
        self.timeout = timeout
        self.log = log
        self.fetcher = fetcher
        self.since = int(time.time())  # a message id once one has been seen
        self.last = time.time()
        self.failures = 0

    @property
    def on(self):
        return bool(self.url)

    def due(self, now=None):
        return self.on and (now or time.time()) - self.last >= self.every

    def poll(self):
        """The newest command ("stop" wins over "save") sent since the last poll, or None."""
        self.last = time.time()
        if not self.on:
            return None
        try:
            text = self.fetcher(self.url, self.since, self.timeout)
        except Exception as error:  # any network trouble: keep training
            self.failures += 1
            self.log(f"control: poll failed ({type(error).__name__}), {self.failures} in a row; training continues")
            return None
        self.failures = 0
        found = []
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("event") != "message":
                continue
            self.since = row.get("id", self.since)
            word = str(row.get("message", "")).strip().lower()
            if word in COMMANDS:
                found.append(word)
        if found:
            command = "stop" if "stop" in found else "save"
            self.log(f"control: received {command}")
            return command
        return None
