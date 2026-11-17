"""Kernel-side kill switch and log relay for Kaggle kernels. Standard library only: v1/scripts/run_control.py
inlines this file into every kernel v1/scripts/vla_kaggle.py pushes, in place of its KERNEL_CONTROL marker line,
so it works from the first second of the session, before the code dataset is unpacked.

A daemon thread polls the run's private ntfy topic (CONTROL_URL, filled in at push time, never printed)
every 60 s for messages sent after the kernel started. The words:

kill        Every trainer subprocess gets SIGTERM, then SIGKILL after 30 s (with its own children, e.g.
            torchrun's workers); any other child of the kernel (pip, a data generator) is killed too. The
            kernel prints one {"event": "kernel_control", ...} line and ends at once (os._exit), so the
            session stops and stops using quota. Files already in /kaggle/working stay.
kill NAME   Only the trainer NAME (its --name, or the end of it: "kill 2b" matches "scale-curve-2b") is
            terminated as above; the other runs go on. For the losing arm of a comparison.
stop        A graceful stop. Trainers that use beebot.control (their module imports it; today only
            beebot.vla.voxel.think_train) read the same topic through BEEBOT_CONTROL_URL, save a checkpoint
            and resume state, and exit; the kernel waits up to 30 minutes for them, then terminates them.
            Trainers that do not use beebot.control are terminated at once (SIGTERM, then SIGKILL): their
            checkpoints already written stay, nothing new is saved. No further runs start. With no trainer
            running yet (setup), stop is kill.
save        Left to the trainers that use beebot.control (checkpoint now, go on).

Poll failures never stop anything: they are logged (the first, then every tenth) and the thread goes on.
The thread is a daemon, so it never keeps the kernel alive after the trainers finish normally.
"""

import json
import os
import signal
import sys
import threading
import time
import urllib.parse
import urllib.request

POLL_EVERY = 60  # seconds between polls of the control topic
KILL_GRACE = 30  # seconds between SIGTERM and SIGKILL
STOP_GRACE = 30 * 60  # seconds a beebot.control trainer gets to save and exit after "stop"


def fetch_messages(url, since, timeout=10):
    """ntfy's JSON lines for the messages after `since` (a unix time or a message id)."""
    request = urllib.request.Request(f"{url}/json?poll=1&since={urllib.parse.quote(str(since))}", headers={"User-Agent": "beebot-control"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode(errors="replace")


def uses_control(module, src="/kaggle/working/code/src"):
    """Whether the trainer module polls the control topic itself (imports beebot.control)."""
    try:
        with open(os.path.join(src, *module.split(".")) + ".py") as f:
            return "beebot.control" in f.read()
    except OSError:
        return False


def descendants(pid):
    """The process ids below pid, from /proc (Linux); [] where there is no /proc."""
    children = {}
    for entry in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                parent = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(parent, []).append(int(entry))
    found, todo = [], [pid]
    while todo:
        for child in children.get(todo.pop(), []):
            found.append(child)
            todo.append(child)
    return found


def signal_quietly(pid, number):
    try:
        os.kill(pid, number)
    except (ProcessLookupError, PermissionError):
        pass


class KernelControl:
    def __init__(self, url, every=POLL_EVERY, kill_grace=KILL_GRACE, stop_grace=STOP_GRACE, fetch=fetch_messages, exit=os._exit, log=None, src="/kaggle/working/code/src"):
        self.url = "" if not url or url.startswith("__") else url.rstrip("/")  # an unfilled placeholder means off
        self.every = every
        self.kill_grace = kill_grace
        self.stop_grace = stop_grace
        self.fetch = fetch
        self.exit = exit
        self.log = log or (lambda text: print(text, flush=True))
        self.src = src
        self.trainers = []  # [name, process, graceful]
        self.stopping = False  # set by "stop": the kernel starts no further runs
        self.stop_deadline = None
        self.since = int(time.time())  # then the id of the last message seen
        self.failures = 0
        self.started = time.time()
        if self.url:
            os.environ["BEEBOT_CONTROL_URL"] = self.url  # trainers that use beebot.control poll the same topic

    @property
    def on(self):
        return bool(self.url)

    def start(self):
        self.log(f"kernel control {'on: polls its topic every %d s (kill, kill NAME, stop)' % self.every if self.on else 'off'}")
        if self.on:
            threading.Thread(target=self.loop, daemon=True, name="kernel-control").start()
        return self

    def track(self, name, process, module=""):
        """Register a trainer subprocess; module (e.g. beebot.vla.lm_train) says whether it handles stop itself."""
        self.trainers.append([name, process, bool(module) and uses_control(module, self.src)])
        return process

    def alive(self):
        return [t for t in self.trainers if t[1].poll() is None]

    def loop(self):
        while True:
            time.sleep(self.every)
            self.step()

    def step(self):
        """One poll and whatever it asks for. Never raises."""
        try:
            for word in self.poll():
                self.handle(word)
            if self.stop_deadline and time.time() > self.stop_deadline:
                late = self.alive()
                self.say("stop", [t[0] for t in late], "trainers did not exit within the stop grace")
                self.terminate(late)
                self.stop_deadline = None
        except Exception as error:  # the switch must never take the kernel down
            self.log(f"kernel control: {type(error).__name__}: {error}; going on")

    def poll(self):
        """The command words sent since the last poll, oldest first; [] on any failure."""
        try:
            text = self.fetch(self.url, self.since)
        except Exception as error:
            self.failures += 1
            if self.failures == 1 or self.failures % 10 == 0:
                self.log(f"kernel control: poll failed ({type(error).__name__}), {self.failures} in a row; training goes on")
            return []
        self.failures = 0
        words = []
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or row.get("event") != "message":
                continue
            self.since = row.get("id", self.since)
            words.append(" ".join(str(row.get("message", "")).lower().split()))
        return words

    def handle(self, word):
        if word == "kill":
            self.kill("kill received")
        elif word.startswith("kill "):
            self.kill_run(word.split(" ", 1)[1])
        elif word == "stop" and not self.stopping:
            self.stop()

    def say(self, command, runs, reason):
        self.log(json.dumps({"event": "kernel_control", "command": command, "runs": runs, "reason": reason, "minutes": round((time.time() - self.started) / 60, 1)}))

    def terminate(self, trainers):
        """SIGTERM to each trainer, then SIGKILL (with its children) to those still alive after the grace."""
        for _, process, _ in trainers:
            if process.poll() is None:
                signal_quietly(process.pid, signal.SIGTERM)
        end = time.time() + self.kill_grace
        while time.time() < end and any(p.poll() is None for _, p, _ in trainers):
            time.sleep(0.2)
        for _, process, _ in trainers:
            if process.poll() is None:
                for pid in descendants(process.pid):
                    signal_quietly(pid, signal.SIGKILL)
                signal_quietly(process.pid, signal.SIGKILL)
                process.wait()

    def kill(self, reason):
        """Terminate every trainer and every other child, then end the kernel at once."""
        self.say("kill", [t[0] for t in self.trainers], reason)
        self.terminate(self.alive())
        for pid in descendants(os.getpid()):
            signal_quietly(pid, signal.SIGKILL)
        self.log("kernel control: kernel ends now")
        sys.stdout.flush()
        sys.stderr.flush()
        self.exit(0)

    def kill_run(self, target):
        chosen = [t for t in self.trainers if t[0].lower() == target or t[0].lower().endswith("-" + target)]
        self.say("kill", [t[0] for t in chosen], f"kill {target} received" + ("" if chosen else ": no such run"))
        self.terminate([t for t in chosen if t[1].poll() is None])

    def stop(self):
        self.stopping = True
        running = self.alive()
        if not running:
            self.kill("stop received before any trainer was running")
            return
        graceful = [t for t in running if t[2]]
        plain = [t for t in running if not t[2]]
        self.say("stop", [t[0] for t in running], f"{[t[0] for t in graceful]} save and exit themselves; {[t[0] for t in plain]} are terminated")
        if graceful:
            self.stop_deadline = time.time() + self.stop_grace
        self.terminate(plain)


def relay(started, noise=(), width=600, whole="", every=30, folder="/kaggle/working/logs"):
    """Kaggle shows only the kernel's own output while it runs: print every run's new log lines as
    "[name] line" (the form v1/scripts/kaggle_watch.py parses) until every run has exited, then
    "exit NAME CODE" per run. Lines containing a noise string are dropped; lines are cut at width
    unless they contain `whole`."""
    seen = {name: 0 for name, _ in started}
    while True:
        done = all(p.poll() is not None for _, p in started)  # before the read: no line written after it is lost
        for name, _ in started:
            with open(f"{folder}/{name}.out", errors="replace") as f:
                lines = f.read().split("\n")
            end = len(lines) if done else len(lines) - 1  # a last line without its newline waits, unless the run has exited
            for line in lines[seen[name] : end]:
                if line.strip() and not any(n in line for n in noise):
                    print(f"[{name}] {line if whole and whole in line else line[:width]}", flush=True)
            seen[name] = end
        if done:
            break
        time.sleep(every)
    for name, p in started:
        print("exit", name, p.returncode, flush=True)
