"""Watch a Kaggle session from the Mac and exit as soon as something needs a decision (network only, light).

    .venv/bin/python v1/scripts/kaggle_watch.py OWNER SLUG                                  # all checks, alert only
    .venv/bin/python v1/scripts/kaggle_watch.py OWNER SLUG --compare 0.8b 2b --separate 0.05 --after 300
    .venv/bin/python v1/scripts/kaggle_watch.py OWNER SLUG --kill-on a,c                    # also sends kill for these
    .venv/bin/python v1/scripts/kaggle_watch.py --replay runs/kaggle/SLUG/SLUG.log --compare 0.8b 2b   # offline

Run it with run_in_background after every launch: the exit wakes the session. OWNER may be OWNER=CONFIG_DIR
(a teammate's token folder, as in kaggle_rest.py). It follows the session's log into logs/SLUG.out (the same
file and ".follow" count as kaggle_follow.py, so `ops tail logs/SLUG.out --label run=SLUG` follows it) and
parses the progress lines the kernel relays per run ("[name] {"step": ..., "loss": ...}"). Checks:

  a  exit 10  a Traceback, an argparse "error:", a CUDA OOM, or a run exiting non-zero (not one the kill switch ended)
  b  exit 11  no training step (step > 0) logged within --first-step-minutes of the session starting
  c  exit 12  a run's loss is NaN, or above its first loss at or after --warmup steps for --rise-points points in a row
              (by more than --rise, a fraction)
  d  exit 13  --compare A B: one arm's loss worse than the other's by more than --separate (a fraction) at each of
              the last --points steps both arms logged, all at or after step --after (to kill the losing arm)
  e  exit 0   the session ended: a one-line summary

On a check it prints one line, posts it to the ntfy.sh topic in --ntfy (default $BEEBOT_NTFY, none if unset; never in replay) and exits.
Default is alert only; --kill-on a,c sends `kill` to the kernel's control topic (run_control.py) for those checks,
and for d it sends `kill <losing arm>`, which ends only that trainer. A check that fired is remembered in
logs/SLUG.out.watch, so a restarted watcher does not fire it again (--reset forgets).

--replay FILE replays a downloaded log (runs/kaggle/SLUG/SLUG.log, Kaggle's JSON list with session times) as
the live watcher would see it, polling every --every seconds of simulated time; a plain-text log has no
times, so check b is off and minutes are not reported.
"""

import argparse
import http.client
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import kaggle_follow
import kaggle_rest
import run_control

EXIT = {"e": 0, "a": 10, "b": 11, "c": 12, "d": 13}
PREFIX = re.compile(r"^\[([^\]\s]+)\] (.*)$")
ARGPARSE = re.compile(r"^[\w.\-/ ]+: error: ")
EXCEPTION = re.compile(r"^([\w.]+(Error|Exception|Interrupt|Exit)\b|[\w.]*OutOfMemoryError)")
OOM = ("CUDA out of memory", "OutOfMemoryError")
EXITED = re.compile(r"^exit (\S+) (-?\d+)$")
LOSS_KEYS = ("loss", "demo_loss", "train_loss")
END_MARK = "[NbConvertApp] Writing"  # Kaggle's last log line: the session is over


@dataclass
class Alert:
    check: str
    reason: str
    time: float | None  # session seconds when the watcher saw it
    key: str  # what a restarted watcher remembers
    run: str = ""  # d: the losing run

    @property
    def code(self):
        return EXIT[self.check]


@dataclass
class Run:
    name: str
    arm: str = ""
    last_step: int = -1
    losses: dict = field(default_factory=dict)  # step -> loss, steps increasing
    reference: float | None = None  # c: the first loss at or after warmup
    over: int = 0  # c: points in a row above the reference
    exit_code: int | None = None

    def matches(self, spec):
        return spec in (self.name, self.arm) or self.name.endswith("-" + spec)

    def last_loss(self):
        return self.losses[max(self.losses)] if self.losses else None


def parse(line):
    """(run name or None, text without the prefix, the JSON object or None)."""
    m = PREFIX.match(line)
    run, text = (m.group(1), m.group(2)) if m else (None, line)
    payload = None
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except ValueError:
            payload = None
    return run, text, payload if isinstance(payload, dict) else None


def loss_of(payload, key=None):
    """The training loss in a progress line: --loss-key, else the first of LOSS_KEYS, else a key ending in "loss"
    that is not a validation loss."""
    keys = [key] if key else [*LOSS_KEYS, *[k for k in payload if k.endswith("loss") and not k.startswith("val")]]
    for k in keys:
        value = payload.get(k)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if value in ("nan", "NaN", "inf", "-inf"):
            return float(value)
    return None


class Watch:
    """The checks over a session's log lines. add() lines with their session time, then check() the new ones;
    tick() checks the clock (b). Pure: no network, no files."""

    def __init__(self, cfg, fired=None):
        self.cfg = cfg
        self.fired = dict(fired or {})
        self.lines = []  # (session seconds or None, line)
        self.runs = {}
        self.checked = 0
        self.dump = None  # inside the kernel's final "==== logs/NAME.out" tails (repeats)
        self.controlled = set()  # runs the kill switch ended (their non-zero exit is expected)

    def add(self, t, line):
        self.lines.append((t, line))

    def run(self, name):
        if name not in self.runs:
            self.runs[name] = Run(name)
        return self.runs[name]

    def check(self):
        """Process the lines added since the last call; the first alert, or None."""
        while self.checked < len(self.lines):
            i = self.checked
            self.checked += 1
            alert = self.line(i)
            if alert and self.fresh(alert):
                return alert
        return None

    def fresh(self, alert):
        if alert.check == "a":
            return i_after(alert.key, self.fired.get("a"))
        return alert.key not in self.fired

    def remember(self, alert):
        self.fired["a" if alert.check == "a" else alert.key] = alert.key

    def line(self, i):
        t, line = self.lines[i]
        run, text, payload = parse(line)
        if line.startswith("==== ") and line.endswith(".out"):
            self.dump = line.rsplit("/", 1)[-1][: -len(".out")]
            return None
        if self.dump and run is None:
            run = self.dump
        if payload is not None:
            return self.progress(run or payload.get("arm") or "main", payload, t)
        exited = EXITED.match(line)
        if exited:
            code = int(exited.group(2))
            self.run(exited.group(1)).exit_code = code
            if code != 0 and exited.group(1) not in self.controlled:
                return Alert("a", f"{exited.group(1)} exited with code {code}", t, f"a:{i}")
            return None
        if self.dump:  # the tails repeat what was relayed
            return None
        who = f"{run}: " if run else ""
        if text.startswith("Traceback (most recent call last)"):
            last = self.exception_after(i, run)
            kind = "CUDA OOM" if last and any(o in last for o in OOM) else "Traceback"
            return Alert("a", f"{kind} in {run or 'the kernel'}" + (f": {last}" if last else ""), t, f"a:{i}")
        if any(o in text for o in OOM):
            return Alert("a", f"CUDA OOM: {who}{text[:200]}", t, f"a:{i}")
        if ARGPARSE.match(text):
            return Alert("a", f"argparse error: {who}{text[:200]}", t, f"a:{i}")
        return None

    def exception_after(self, i, run):
        """The exception line ending the traceback that starts at line i (same run), if it has arrived."""
        for _, later in self.lines[i + 1 : i + 200]:
            other, text, _ = parse(later)
            if other != run:
                continue
            if EXCEPTION.match(text):
                return text[:200]
        return ""

    def progress(self, name, payload, t):
        if payload.get("event") == "kernel_control":
            self.controlled.update(payload.get("runs", []))
            return None
        step = payload.get("step")
        if not isinstance(step, int):
            return None
        run = self.run(name)
        run.arm = str(payload.get("arm", run.arm))
        run.last_step = max(run.last_step, step)
        loss = loss_of(payload, self.cfg.loss_key)
        if loss is None or (run.losses and step <= max(run.losses)):
            return None
        run.losses[step] = loss
        return self.loss_check(run, step, loss, t) or self.compare(t)

    def loss_check(self, run, step, loss, t):
        if not math.isfinite(loss):
            return Alert("c", f"{run.name} loss is {loss} at step {step}", t, f"c:{run.name}")
        if step < self.cfg.warmup:
            return None
        if run.reference is None:
            run.reference = loss
            return None
        run.over = run.over + 1 if loss > run.reference * (1 + self.cfg.rise) else 0
        if run.over >= self.cfg.rise_points:
            return Alert("c", f"{run.name} loss {loss} at step {step} has been above its first loss after warmup ({run.reference}) for {run.over} points", t, f"c:{run.name}")
        return None

    def compare(self, t):
        if not self.cfg.compare:
            return None
        spec_a, spec_b = self.cfg.compare
        a = next((r for r in self.runs.values() if r.matches(spec_a)), None)
        b = next((r for r in self.runs.values() if r.matches(spec_b)), None)
        if not a or not b or a is b:
            return None
        common = sorted(s for s in set(a.losses) & set(b.losses) if s >= self.cfg.after)
        if len(common) < self.cfg.points:
            return None
        last = common[-self.cfg.points :]
        for worse, better in ((a, b), (b, a)):
            gaps = [worse.losses[s] / better.losses[s] - 1 for s in last]
            if all(g > self.cfg.separate for g in gaps):
                return Alert("d", f"{worse.name} loss worse than {better.name} by {min(gaps):.0%}-{max(gaps):.0%} at steps {last[0]}-{last[-1]} (last {len(last)} both logged; now {worse.name} {worse.losses[last[-1]]} vs {better.losses[last[-1]]})", t, "d", run=worse.name)
        return None

    def tick(self, t):
        """b: no training step within the first --first-step-minutes of the session (t: session seconds)."""
        if t is None or self.cfg.first_step_minutes <= 0 or t < self.cfg.first_step_minutes * 60:
            return None
        if any(r.last_step > 0 for r in self.runs.values()):
            return None
        alert = Alert("b", f"no training step logged within {self.cfg.first_step_minutes:g} min of the session start", t, "b")
        return alert if self.fresh(alert) else None

    def summary(self, status, t):
        runs = "; ".join(f"{r.name} step {r.last_step}" + (f" loss {r.last_loss()}" if r.losses else "") + ("" if r.exit_code is None else f" exit {r.exit_code}") for r in self.runs.values() if r.last_step >= 0 or r.exit_code is not None)
        return f"ended ({status})" + (f" after {t / 60:.0f} min" if t else "") + (f": {runs}" if runs else ": no progress lines")


def i_after(key, previous):
    """Whether an a-alert at line key ('a:<line>') comes after the one already fired (previous)."""
    return previous is None or int(key.split(":")[1]) > int(previous.split(":")[1])


def minutes(t):
    return "" if t is None else f" at {t / 60:.1f} min"


def notify(topic, slug, text):
    if not topic:
        return
    try:
        request = urllib.request.Request(f"https://ntfy.sh/{topic}", data=text.encode(), method="POST", headers={"Title": f"beebot {slug}", "User-Agent": "beebot-watch"})
        urllib.request.urlopen(request, timeout=15).read()
    except Exception as error:  # the alert still goes out through the exit
        print(f"ntfy post failed ({type(error).__name__})", flush=True)


def act(alert, cfg, slug, replay):
    """The one line for an alert, after sending kill when --kill-on asks for it."""
    line = f"{slug} {alert.check}{minutes(alert.time)}: {alert.reason}"
    if alert.check in cfg.kill_on:
        word = f"kill {alert.run}" if alert.check == "d" else "kill"
        if replay:
            line += f" [replay: would send '{word}']"
        else:
            try:
                run_control.send(cfg.control or slug, word)
                line += f" [sent '{word}']"
            except Exception as error:
                line += f" [could not send '{word}': {type(error).__name__}; run_control.py kill {cfg.control or slug}]"
    return line


def read_replay(path):
    """[(session seconds or None, line)] of a downloaded log: Kaggle's JSON list, or plain text."""
    text = Path(path).read_text(errors="replace")
    if text.lstrip().startswith("[{"):
        return timed_lines(kaggle_rest.parse_entries(text), final=True)
    return [(None, line) for line in text.split("\n") if line]


def timed_lines(entries, final):
    """[(time, line)] from log entries: a line takes the time of the entry that completed it; a last line
    without its newline is held back unless final."""
    out, buffer, t = [], "", None
    for entry in entries:
        t = entry.get("time", t)
        pieces = (buffer + entry.get("data", "")).split("\n")
        out += [(t, piece) for piece in pieces[:-1]]
        buffer = pieces[-1]
    if final and buffer:
        out.append((t, buffer))
    return out


def replay(cfg):
    lines = read_replay(cfg.replay)
    watch = Watch(cfg)
    timed = lines[0][0] is not None if lines else False
    if not timed:
        for t, line in lines:
            watch.add(t, line)
        alert = watch.check()
        return finish(alert, watch, cfg, "replay", "COMPLETE", None, replay=True)
    end = max(t for t, _ in lines)
    now, i = 0.0, 0
    while True:
        now += cfg.every
        while i < len(lines) and lines[i][0] <= now:
            watch.add(*lines[i])
            i += 1
        alert = watch.check() or watch.tick(min(now, end))
        if alert:
            line_time = alert.time  # when the line was logged; the alert counts from the poll that sees it
            alert.time = min(now, end)
            return finish(alert, watch, cfg, "replay", "", None, replay=True, line_time=line_time if alert.check != "b" else None)
        if i >= len(lines):
            return finish(None, watch, cfg, "replay", "COMPLETE", end, replay=True)


def finish(alert, watch, cfg, slug, status, t, replay=False, line_time=None):
    if alert is None:
        print(f"{slug} e: {watch.summary(status, t)}", flush=True)
        return 0
    line = act(alert, cfg, slug, replay)
    if replay and line_time is not None:
        line += f" (line logged at {line_time / 60:.1f} min)"
    print(line, flush=True)
    if not replay:
        notify(cfg.ntfy, slug, line)
    return alert.code


def session_status(owner, slug):
    """QUEUED, RUNNING, COMPLETE, ERROR, ...; UNKNOWN when neither the REST API nor the kaggle CLI answers."""
    now = kaggle_follow.status(owner, slug)
    if now not in ("UNKNOWN", "GONE"):
        return now
    uvx = shutil.which("uvx") or str(Path.home() / ".local/bin/uvx")
    try:
        token = kaggle_rest.token()
        out = subprocess.run([uvx, "--with", "six", "kaggle", "kernels", "status", f"{owner}/{slug}"], capture_output=True, text=True, timeout=120, env={**os.environ, "KAGGLE_API_TOKEN": token}).stdout
        found = re.search(r"KernelWorkerStatus\.(\w+)", out)
        return found.group(1) if found else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def live(cfg):
    owner = kaggle_rest.account(cfg.owner)
    out = Path(cfg.out or f"logs/{cfg.slug}.out")
    out.parent.mkdir(parents=True, exist_ok=True)
    follow = Path(str(out) + ".follow")
    state = Path(str(out) + ".watch")
    fired = {} if cfg.reset or not state.exists() else json.loads(state.read_text())
    watch = Watch(cfg, fired)
    start = None  # wall time of the session start (an upper bound: the earliest consistent with what was seen)
    while True:
        try:
            now_status = session_status(owner, cfg.slug)
            entries = kaggle_rest.log_entries(owner, cfg.slug, cfg.window) if now_status != "QUEUED" else []
        except (OSError, http.client.HTTPException) as e:  # a dropped connection or timeout: try again next poll
            print(f"{time.strftime('%H:%M:%S')} network error, retrying: {e!r}", flush=True)
            time.sleep(cfg.every)
            continue
        wall = time.time()
        text = "".join(e.get("data", "") for e in entries)
        ended = (now_status not in kaggle_follow.LIVE and now_status != "UNKNOWN") or END_MARK in text
        if now_status == "RUNNING" or entries:
            latest = max((e.get("time", 0) for e in entries), default=0)
            start = min(start or wall, wall - latest)
        new, seen = kaggle_follow.new_lines(text, int(follow.read_text()) if follow.exists() else 0, ended)
        if new:
            with open(out, "a") as f:
                f.write("\n".join(new) + "\n")
        follow.write_text(str(seen))
        lines = timed_lines(entries, ended)
        clock = None if start is None else wall - start
        for t, line in lines[len(watch.lines) :]:
            watch.add(t if t is not None else clock, line)
        alert = watch.check() or watch.tick(clock)
        print(f"{time.strftime('%H:%M:%S')} {now_status}{minutes(clock)}: {len(new)} new lines; " + ", ".join(f"{r.name} step {r.last_step}" for r in watch.runs.values()), flush=True)
        if alert:
            alert.time = clock
            watch.remember(alert)
            state.write_text(json.dumps(watch.fired))
            return finish(alert, watch, cfg, cfg.slug, now_status, clock)
        if ended:  # the status can lag the log's end mark by minutes
            return finish(None, watch, cfg, cfg.slug, now_status if now_status not in kaggle_follow.LIVE else "log ended", clock)
        time.sleep(cfg.every)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("owner", nargs="?", help="OWNER or OWNER=CONFIG_DIR")
    parser.add_argument("slug", nargs="?")
    parser.add_argument("--replay", help="a downloaded log to replay offline (simulated time)")
    parser.add_argument("--out", help="the followed log (default logs/SLUG.out)")
    parser.add_argument("--every", type=float, default=120, help="seconds between polls (live and simulated)")
    parser.add_argument("--window", type=float, default=25, help="seconds to read a running session's log stream")
    parser.add_argument("--first-step-minutes", type=float, default=30, help="b: minutes after the session start by which a training step must appear (0: off)")
    parser.add_argument("--warmup", type=int, default=100, help="c: the first loss at or after this step is the reference")
    parser.add_argument("--rise", type=float, default=0.05, help="c: fraction above the reference that counts as risen (batch-16 losses wander ~2-3%% on a plateau)")
    parser.add_argument("--rise-points", type=int, default=3, help="c: logged points in a row above the reference")
    parser.add_argument("--loss-key", help="the progress lines' loss key (default: loss, demo_loss, train_loss, else any *loss)")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="d: two arms (run names, their ends, or the lines' 'arm')")
    parser.add_argument("--separate", type=float, default=0.05, help="d: fraction by which the losing arm is worse")
    parser.add_argument("--after", type=int, default=300, help="d: only steps at or after this one")
    parser.add_argument("--points", type=int, default=4, help="d: the last K steps both arms logged")
    parser.add_argument("--kill-on", default="", help="checks that also send kill, comma-separated (e.g. a,c; d kills only the losing arm)")
    parser.add_argument("--control", help="the control topic's run name (default the slug)")
    parser.add_argument("--ntfy", default=os.environ.get("BEEBOT_NTFY", ""), help="ntfy.sh topic for the alert line (default $BEEBOT_NTFY; '' for none; never in replay)")
    parser.add_argument("--reset", action="store_true", help="forget the checks a previous watcher fired")
    cfg = parser.parse_args(argv)
    cfg.kill_on = {c.strip() for c in cfg.kill_on.split(",") if c.strip()}
    if not cfg.replay and not (cfg.owner and cfg.slug):
        parser.error("OWNER and SLUG, or --replay FILE")
    unknown = cfg.kill_on - set("abcd")
    if unknown:
        parser.error(f"--kill-on: unknown checks {sorted(unknown)}")
    return cfg


def main():
    cfg = arguments()
    sys.exit(replay(cfg) if cfg.replay else live(cfg))


if __name__ == "__main__":
    main()
