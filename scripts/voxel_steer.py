"""The 3D sibling of swarm_steer.py: each visitor gets their own 16^3 voxel world, types commands
or questions, and watches the swarm of cubes with the driver's thoughts next to it. The page
(voxel_steer.html) polls frames that carry robot coordinates, not grids.

There is no trained 3D model yet, so the swarm is driven by a stand-in behind the Driver
interface: ReplayDriver plays stored expert demonstrations (beebot.vla.voxel.demos) and words the
expert's sub-goals as thoughts. A model replaces Driver.start, Driver.step and Driver.ask.

    uv run python scripts/voxel_steer.py [--store data/vla/voxel_demos_cube_sample] [--port 8792]

Then open http://127.0.0.1:8792 (the page is read from voxel_steer.html on every load).
"""

import argparse
import json
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

from beebot.vla.voxel import play_store
from beebot.vla.voxel import world as W

PAGE = Path(__file__).with_name("voxel_steer.html")
STORE = Path(__file__).resolve().parents[1] / "data/vla/voxel_demos_cube_sample"
DIRS = W.DIR_TABLE.numpy()  # (27, 3) the move of each action code, STAY last
HEX = "0123456789abcdef"


class Driver:
    """What steers one visitor's swarm; the site makes one per visitor. A model overrides start,
    step and ask. scene and advance have defaults: a sampled scene and the world's own rules with
    the cube's move failures."""

    note = ""  # one line under the page

    def __init__(self):
        self.rates = W.failure_rates()

    def scene(self, rng):
        """A new visitor's world (VoxelWorld, batch of 1) and a command to suggest ("" for none)."""
        world, _ = W.sample_scenes(rng, 1)
        return world, ""

    def start(self, world, text):
        """The visitor sent the command `text`. Returns the thoughts written before acting."""
        raise NotImplementedError

    def step(self, world):
        """(action codes (1, D, H, W) long in 0..26, new thoughts) for the next step. Codes of
        None mean there is nothing left to do and the swarm rests."""
        raise NotImplementedError

    def advance(self, world, codes):
        """The world one step later under codes."""
        return W.step(world, W.fail_moves(codes, self.rates))[0]

    def suggestion(self, world):
        """A command to suggest once the last one has finished ("" for none)."""
        return ""

    def ask(self, world, text):
        """The answer to a question about the scene, as text."""
        raise NotImplementedError


def join(words):
    return " and ".join(words) if len(words) < 3 else ", ".join(words[:-1]) + " and " + words[-1]


def cheese_names(world):
    """Colour names of the cheeses still in the scene."""
    return [W.COLOUR_NAMES[int(c)] for c, a in zip(world.cheese_color[0].tolist(), world.cheese_alive[0].tolist()) if a and c >= 0]


def goal_words(goal):
    """An expert sub-goal ({"kind", ...}, suite.Expert.describe) as a short phrase."""
    kind = goal["kind"]
    words = {
        "eat": lambda: f"eat the {goal['colour']} cheese",
        "surround": lambda: f"surround the {goal['colour']} cheese",
        "push": lambda: f"push the {goal['colour']} cheese " + ("into the trench" if goal["to"] == "trench" else f"to the {goal['to']}"),
        "tower": lambda: f"build a tower {goal['height']} high",
        "region": lambda: f"gather at the {goal['region']}",
        "shape": lambda: f"form a {goal['shape']}",
        "letters": lambda: f"spell {goal['word']} " + ("standing up" if goal["standing"] else "flat on the ground"),
        "fill": lambda: "fill the trench",
        "bridge": lambda: "bridge the trench",
        "union": lambda: "two groups, " + "; ".join(goal_words(part) for part in goal["parts"]),
        "stay": lambda: "nothing here to act on, stay put",
        "done": lambda: "hold still",
    }
    return words.get(kind, lambda: kind)()


def scene_words(world):
    names = cheese_names(world)
    return f"I see {'a ' if len(names) == 1 else ''}{join(names)} {'cheese' if len(names) == 1 else 'cheeses'}." if names else "No cheese in sight."


class ReplayDriver(Driver):
    """The stand-in: plays one stored demonstration per field, whatever the command says, and
    words the expert's sub-goal at each switch as a thought."""

    note = "stand-in: each field replays a stored expert demonstration; the language model is still training"

    def __init__(self, store, texts, episode=None):
        super().__init__()
        self.store, self.texts, self.episode = store, texts, episode
        self.e, self.t, self.playing, self.aim = 0, 0, False, ""

    def state(self, t):
        s, e = self.store, self.e
        return W.VoxelWorld(
            mass=torch.from_numpy(np.array(s["mass"][e, t], dtype=np.float32)).unsqueeze(0),
            walls=s["walls"][e : e + 1],
            cheese_pos=s["cheese_pos"][e, t].long().unsqueeze(0),
            cheese_alive=s["cheese_alive"][e, t].unsqueeze(0),
            cheese_color=s["cheese_color"][e : e + 1],
            gravity=s["gravity"][e : e + 1],
        )

    def scene(self, rng):
        self.e = int(rng.integers(self.store["count"])) if self.episode is None else self.episode
        self.t, self.playing = 0, False
        return self.state(0), self.texts[self.e]

    def start(self, world, text):
        steps = self.store["actions"].shape[1]
        if self.playing:
            thoughts = ["This field's recording has ended. A new field has another." if self.t >= steps else "Still on it."]
        else:
            self.playing = True
            self.aim = goal_words(self.store["subgoals"][self.e][0]["goal"])
            thoughts = [scene_words(world), f"First: {self.aim}."]
        return thoughts

    def step(self, world):
        codes, thoughts = None, []
        if self.t < self.store["actions"].shape[1]:
            codes = torch.from_numpy(np.array(self.store["actions"][self.e, self.t], dtype=np.int64)).unsqueeze(0)
            aims = [goal_words(entry["goal"]) for entry in self.store["subgoals"][self.e] if entry["t"] == self.t and self.t > 0]
            fresh = [aim for aim in aims if aim != self.aim]  # a push that only turns reads the same
            thoughts = ["That is done." if aim == "hold still" else f"Next: {aim}." for aim in fresh]
            self.aim = (fresh or [self.aim])[-1]
        return codes, thoughts

    def advance(self, world, codes):
        self.t += 1
        return self.state(self.t)

    def ask(self, world, text):
        """Answers from the scene's facts by keyword; anything else gets the scene in a line."""
        q = text.lower()
        names, first = cheese_names(world), cheese_names(self.state(0))
        gone = [n for n in first if n not in names]
        heights = world.walls[0].sum(0)
        terrain = "flat" if int(heights.max()) == int(heights.min()) else "uneven, with raised ground"
        answers = [
            (("eaten", "ate", "gone"), f"{join(gone).capitalize()} {'has' if len(gone) == 1 else 'have'} been eaten." if gone else "Nothing has been eaten yet."),
            (("how many cheese", "cheeses are there", "number of cheese"), f"{len(names)}." if not names else f"{len(names)}: {join(names)}."),
            (("how many", "robot", "cube", "swarm", "unit"), f"{int(world.mass[0].sum())} cubes."),
            (("gravity", "fall", "fly", "float"), "Gravity is on: an unsupported cube falls." if bool(world.gravity[0]) else "No gravity here: the cubes can float."),
            (("ground", "terrain", "floor", "hill", "trench", "wall"), f"The ground is {terrain}."),
            (("colour", "color", "cheese", "see", "what", "which"), scene_words(world)),
        ]
        hits = [answer for keys, answer in answers if any(k in q for k in keys)]
        return hits[0] if hits else f"{scene_words(world)} {int(world.mass[0].sum())} cubes on {terrain} ground."


def track(robots, codes, mass):
    """Which robot went where in one step, so the page can slide each cube: robots (N, 3) int
    (z, y, x) before, codes (D, H, W) the action at each robot's voxel, mass (D, H, W) bool
    after. Returns (M, 3): row i is robot i's voxel after the step while the count holds.
    Chosen moves first (a voxel is open once its robot has left), then falls, then stays; what
    is left (shoves, a move and a fall in one step) pairs by distance."""
    before = {tuple(p): i for i, p in enumerate(robots.tolist())}
    free = {tuple(p) for p in np.argwhere(mass).tolist()}
    after = {}

    def claim(i, q):
        opens = i not in after and q in free and (q not in before or before[q] in after or before[q] == i)
        if opens:
            after[i] = q
            free.discard(q)
        return opens

    wants = {i: tuple((np.array(p) + DIRS[codes[p]]).tolist()) for p, i in before.items() if codes[p] != W.STAY}
    progress = True
    while progress:
        progress = any([claim(i, q) for i, q in wants.items()])
    for p, i in sorted(before.items()):  # lowest first, so a column falls together
        claim(i, (p[0] - 1, p[1], p[2]))
    for i, q in wants.items():  # a move, then a fall
        claim(i, (q[0] - 1, q[1], q[2]))
    for p, i in before.items():
        claim(i, p)
    rest = [i for i in range(len(robots)) if i not in after]
    while rest and free:
        spots = sorted(free)
        gaps = ((robots[rest][:, None] - np.array(spots)[None]) ** 2).sum(-1)
        r, s = np.unravel_index(gaps.argmin(), gaps.shape)
        after[rest.pop(int(r))] = spots[int(s)]
        free.discard(spots[int(s)])
    rows = [after[i] for i in range(len(robots)) if i in after] + sorted(free)
    return np.array(rows, dtype=np.int64).reshape(-1, 3)


def hexes(values):
    return "".join(HEX[int(v)] for v in values)


class Site:
    """Visitors' worlds, drivers and notes (thoughts, answers, events), and the stepping loop."""

    def __init__(self, make_driver, fps, max_visitors, max_steps):
        self.make_driver = make_driver
        self.fps, self.max_visitors, self.max_steps = fps, max_visitors, max_steps
        self.lock = threading.Lock()
        self.visitors = {}  # sid -> {seen, driver, world, robots, t, since, text, suggestion, first, notes, hold, pending}
        self.rng = np.random.default_rng()
        self.seq = 0  # notes are numbered across visitors, so a page never sees a number twice

    def write(self, v, kind, texts):
        for text in texts:
            self.seq += 1
            v["notes"].append({"n": self.seq, "kind": kind, "text": text})
        del v["notes"][:-40]
        return sum(len(text) for text in texts)

    def think(self, v, thoughts):
        """Log thoughts and hold the swarm while the page types them out."""
        chars = self.write(v, "thought", thoughts)
        if thoughts:
            v["hold"] = time.time() + min(3.5, 0.5 + 0.03 * chars)

    def new(self, sid):
        with self.lock:
            self.evict()
            body = None
            if sid in self.visitors or len(self.visitors) < self.max_visitors:
                driver = self.make_driver()
                world, text = driver.scene(self.rng)
                robots = np.argwhere(world.mass[0].numpy() >= 0.5)
                self.visitors[sid] = {"seen": time.time(), "driver": driver, "world": world, "robots": robots, "t": 0, "since": 0, "text": "", "suggestion": text, "first": True, "notes": [], "hold": 0.0, "pending": None}
                heights = world.walls[0].sum(0).flatten().tolist()
                body = {"heights": hexes(heights), "gravity": bool(world.gravity[0]), "size": world.mass.shape[-1], **self.frame(sid, self.seq)}
            return body

    def command(self, sid, text, after):
        with self.lock:
            body = None
            if sid in self.visitors:
                v = self.visitors[sid]
                v.update(text=text, since=v["t"], first=False, suggestion="", pending=None)
                self.think(v, v["driver"].start(v["world"], text))
                body = self.frame(sid, after)
            return body

    def ask(self, sid, text, after):
        with self.lock:
            body = None
            if sid in self.visitors:
                v = self.visitors[sid]
                self.write(v, "question", [text])
                self.write(v, "answer", [v["driver"].ask(v["world"], text)])
                body = self.frame(sid, after)
            return body

    def frame(self, sid, after):
        v = self.visitors[sid]
        v["seen"] = time.time()
        world = v["world"]
        cheeses = [[k, *p, c] for k, (p, a, c) in enumerate(zip(world.cheese_pos[0].tolist(), world.cheese_alive[0].tolist(), world.cheese_color[0].tolist())) if a]
        state = "idle" if not v["text"] else "thinking" if time.time() < v["hold"] else "executing"
        notes = [note for note in v["notes"] if note["n"] > after]
        return {"t": v["t"], "robots": hexes(v["robots"].flatten()), "cheeses": cheeses, "notes": notes, "n": self.seq, "state": state, "command": v["text"], "suggestion": v["suggestion"], "first": v["first"]}

    def evict(self, idle=180):
        now = time.time()
        for sid in [k for k, v in self.visitors.items() if now - v["seen"] > idle]:
            self.visitors.pop(sid)

    def rest(self, v):
        v.update(text="", pending=None, suggestion=v["driver"].suggestion(v["world"]))
        self.write(v, "done", ["done"])

    def advance(self, v):
        """One step of one visitor's swarm. Thoughts come before the move they belong to: the
        codes wait (pending) until the page has had time to show them."""
        codes, thoughts = v["pending"] or v["driver"].step(v["world"])
        if v["pending"] is None and thoughts:
            v["pending"] = (codes, [])
            self.think(v, thoughts)
        elif codes is None or v["t"] - v["since"] >= self.max_steps:
            self.rest(v)
        else:
            before = v["world"]
            v["pending"] = None
            v["world"] = v["driver"].advance(before, codes)
            v["robots"] = track(v["robots"], codes[0].numpy(), v["world"].mass[0].numpy() >= 0.5)
            v["t"] += 1
            eaten = (before.cheese_alive[0] & ~v["world"].cheese_alive[0]).tolist()
            self.write(v, "ate", [f"ate {W.COLOUR_NAMES[int(c)]}" for c, gone in zip(before.cheese_color[0].tolist(), eaten) if gone])

    def loop(self):
        while True:
            t0 = time.time()
            with self.lock:
                for v in self.visitors.values():
                    if v["text"] and t0 - v["seen"] < 10 and t0 >= v["hold"]:
                        self.advance(v)
            time.sleep(max(0.0, 1 / self.fps - (time.time() - t0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default=str(STORE), help="demonstration store the stand-in replays")
    parser.add_argument("--episode", type=int, default=None, help="always replay this stored episode (default: a random one per field)")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max_steps", type=int, default=72, help="steps a command runs before the swarm rests")
    parser.add_argument("--max_visitors", type=int, default=24)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8792)
    args = parser.parse_args()
    torch.set_num_threads(2)
    store = play_store.load(args.store)
    texts = {row["e"]: row["text"] for row in map(json.loads, (Path(args.store) / "labels.jsonl").read_text().splitlines())}
    print(f"replaying {store['count']} demonstrations from {args.store}", flush=True)
    site = Site(lambda: ReplayDriver(store, texts, args.episode), args.fps, args.max_visitors, args.max_steps)
    threading.Thread(target=site.loop, daemon=True).start()
    palette = {name: [round(float(v), 3) for v in rgb] for name, rgb in zip(W.COLOUR_NAMES, W.COLOUR_RGB.tolist())}

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, separators=(",", ":")).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path == "/":
                self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/frame":
                sid, after = query.get("sid", [""])[0], query.get("after", ["0"])[0]
                with site.lock:
                    body = site.frame(sid, int(after) if after.isdigit() else 0) if sid in site.visitors else None
                self.send(200 if body else 404, body or {"error": "unknown visitor"})
            elif url.path == "/api/palette":
                self.send(200, {"colours": W.COLOUR_NAMES, "rgb": palette, "size": W.SIZE, "note": ReplayDriver.note})
            else:
                self.send(404, {"error": "not found"})

        def do_POST(self):
            url = urlparse(self.path)
            try:
                payload = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 4096)) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self.send(400, {"error": "bad request"})
                return
            sid = str(payload.get("sid") or secrets.token_hex(8))[:32]
            text = " ".join(str(payload.get("text", "")).split())[:140]
            after = payload.get("after")
            after = after if isinstance(after, int) else 0
            if url.path == "/api/new":
                body = site.new(sid)
                self.send(200 if body else 503, {"sid": sid, **body} if body else {"error": "the swarm is busy, try again in a minute"})
            elif url.path in ("/api/command", "/api/ask"):
                act = site.command if url.path == "/api/command" else site.ask
                body = act(sid, text, after) if text else None
                self.send(200 if body else 404, body or {"error": "unknown visitor"})
            else:
                self.send(404, {"error": "not found"})

    print(f"open http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
