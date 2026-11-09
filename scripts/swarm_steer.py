"""A public, live-steerable swarm: each visitor gets their own world and types instructions to
the 2D LM policy, which re-reads the instruction at every step. One batched forward advances
every active visitor's swarm. Suggested commands come from beebot.vla.steer, limited to the
intents its validation found reliable (scripts/steer_kinds.json).

    uv run python scripts/swarm_steer.py --init_run RUN [--checkpoint policy_latest.pt] [--port 8790]

Then open http://127.0.0.1:8790 (the page is read from swarm_steer.html on every load).
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

from beebot.vla import steer
from beebot.vla.lm_rl import RLConfig, load_policy
from beebot.vla.world import COLOUR_NAMES, COLOUR_RGB

PAGE = Path(__file__).with_name("swarm_steer.html")
FALLBACK = {"place.region": 0.5, "place.box": 0.5, "logic.single": 0.5, "logic.all": 0.5, "formation.shape": 0.4}
BY_SCORE = {"place.region"}  # a few stragglers always miss the region, so judge by the share that arrives


def reliable_kinds(path, protocol, floor):
    """intent key -> success under protocol ("fresh" or "after"), for keys at or above floor."""
    if not Path(path).exists():
        return dict(FALLBACK)
    kinds = json.loads(Path(path).read_text())["kinds"]
    rate = {k: v[protocol]["score" if k in BY_SCORE else "success"] for k, v in kinds.items() if protocol in v and v[protocol]["n"] >= 3}
    out = {k: r for k, r in rate.items() if r >= floor}
    return out or dict(FALLBACK)


def bits(grid):
    return "".join("1" if v else "0" for v in grid.flatten().tolist())


class Site:
    """Visitors' swarms, their suggestions and events, and the stepping loop."""

    def __init__(self, swarms, fresh, after, fps, max_visitors, max_steps):
        self.swarms, self.fresh, self.after = swarms, fresh, after
        self.fps, self.max_visitors, self.max_steps = fps, max_visitors, max_steps
        self.lock = threading.Lock()
        self.visitors = {}  # sid -> {seen, suggestion, events, since}
        self.rng = np.random.default_rng()

    def new(self, sid):
        with self.lock:
            self.evict()
            if sid not in self.visitors and len(self.visitors) >= self.max_visitors:
                return None
            scene, text = steer.first_scene(self.rng, self.fresh)
            self.swarms.reset(sid, scene)
            self.visitors[sid] = {"seen": time.time(), "suggestion": text, "first": True, "events": [], "since": 0}
            world = self.swarms.state[sid]["world"]
            boxes = [[*r, c] for r, c in zip(world.box_rect[0].tolist(), world.box_color[0].tolist()) if c >= 0]
            return {"walls": bits(world.walls[0].cpu().numpy()), "boxes": boxes, **self.frame(sid)}

    def command(self, sid, text):
        with self.lock:
            if sid not in self.visitors:
                return None
            self.swarms.instruct(sid, text)
            v = self.visitors[sid]
            v["since"] = self.swarms.state[sid]["t"]
            v["first"] = False
            v["suggestion"] = self.next_suggestion(sid, avoid=(text,))
            return self.frame(sid)

    def next_suggestion(self, sid, avoid=()):
        s = self.swarms.state[sid]
        return steer.suggest(steer.scene_of(s["world"], s["runners"]), self.after, self.rng, avoid=avoid)

    def frame(self, sid):
        s, v = self.swarms.state[sid], self.visitors[sid]
        v["seen"] = time.time()
        world = s["world"]
        cheeses = [[*p, c] for p, a, c in zip(world.cheese_pos[0].tolist(), world.cheese_alive[0].tolist(), world.cheese_color[0].tolist()) if a]
        events, v["events"] = v["events"], []
        running = bool(s["text"])
        return {"t": s["t"], "mass": bits((world.mass[0] >= 0.5).cpu().numpy()), "cheeses": cheeses, "events": events, "running": running, "command": s["text"], "suggestion": v["suggestion"], "first": v["first"]}

    def evict(self, idle=180):
        now = time.time()
        for sid in [k for k, v in self.visitors.items() if now - v["seen"] > idle]:
            self.visitors.pop(sid)
            self.swarms.state.pop(sid, None)

    def loop(self):
        while True:
            t0 = time.time()
            with self.lock:
                now = time.time()
                active = [sid for sid, v in self.visitors.items() if now - v["seen"] < 10 and self.swarms.state[sid]["text"]]
                for sid in active:  # a command runs max_steps, then the swarm rests
                    if self.swarms.state[sid]["t"] - self.visitors[sid]["since"] >= self.max_steps:
                        self.swarms.instruct(sid, "")
                        self.visitors[sid]["events"].append("done")
                if active:
                    eaten = self.swarms.advance(active)
                    for sid, colours in eaten.items():
                        if colours:
                            self.visitors[sid]["events"] += [f"ate {c}" for c in colours]
                            self.visitors[sid]["suggestion"] = self.next_suggestion(sid, avoid=(self.swarms.state[sid]["text"],))
            time.sleep(max(0.0, 1 / self.fps - (time.time() - t0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_run", required=True)
    parser.add_argument("--checkpoint", default="policy_latest.pt")
    parser.add_argument("--kinds", default=str(Path(__file__).with_name("steer_kinds.json")))
    parser.add_argument("--min_success", type=float, default=0.6)
    parser.add_argument("--fps", type=float, default=7.0)
    parser.add_argument("--max_steps", type=int, default=140, help="steps a command runs before the swarm rests")
    parser.add_argument("--max_visitors", type=int, default=24)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)
    policy, tokenizer = load_policy(RLConfig(init_run=args.init_run, checkpoint=args.checkpoint), device)
    policy.eval()
    fresh = reliable_kinds(args.kinds, "fresh", args.min_success)
    after = reliable_kinds(args.kinds, "after", args.min_success)
    print(f"suggesting fresh: {sorted(fresh)}; after a command: {sorted(after)}", flush=True)
    site = Site(steer.Swarms(policy, tokenizer, device), fresh, after, args.fps, args.max_visitors, args.max_steps)
    threading.Thread(target=site.loop, daemon=True).start()
    palette = {name: [round(float(v), 3) for v in rgb] for name, rgb in zip(COLOUR_NAMES, COLOUR_RGB.tolist())}

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
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
            if url.path == "/":
                self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/frame":
                sid = parse_qs(url.query).get("sid", [""])[0]
                with site.lock:
                    body = site.frame(sid) if sid in site.visitors else None
                self.send(200 if body else 404, body or {"error": "unknown visitor"})
            elif url.path == "/api/palette":
                self.send(200, {"colours": COLOUR_NAMES, "rgb": palette, "size": 32})
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
            if url.path == "/api/new":
                body = site.new(sid)
                self.send(200 if body else 503, {"sid": sid, **body} if body else {"error": "the swarm is busy, try again in a minute"})
            elif url.path == "/api/command":
                text = " ".join(str(payload.get("text", "")).split())[:140]
                body = site.command(sid, text) if text else None
                self.send(200 if body else 404, body or {"error": "unknown visitor"})
            else:
                self.send(404, {"error": "not found"})

    print(f"open http://{args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
