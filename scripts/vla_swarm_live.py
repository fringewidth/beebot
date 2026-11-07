"""Steer the swarm live: the world keeps evolving under the current instruction, and a new one
takes over the moment it is typed. Cheese can be dropped in when the swarm runs out.

    uv run python scripts/vla_swarm_live.py [--qwen runs/lm-qwen-lora-c1]
        [--swarm runs/vla-rel-g32-swarm-text] [--port 8777]

Then open http://127.0.0.1:8777. The page starts on a held-out test scene. The model reads only
the current world and the instruction (it has no memory), so an instruction that names an end
state ("gather in the center") settles once met, and one that names a change ("move left")
keeps applying. "Since the instruction" lists what has happened since the instruction was
given, in the scorer's words. Only this machine can reach the page.
"""

import argparse
import json
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from vla_demo_media import describe  # noqa: E402
from vla_swarm_demo import QwenText, SwarmText  # noqa: E402

from beebot.vla import goal_eval  # noqa: E402
from beebot.vla.data import world_at  # noqa: E402
from beebot.vla.facts import window_facts  # noqa: E402
from beebot.vla.play_data import load_play_cache  # noqa: E402
from beebot.vla.train import Config, load_data  # noqa: E402
from beebot.vla.world import PALETTE_NAMES, World, box_masks, cheese_side, observe, step  # noqa: E402

PAGE = Path(__file__).parent / "vla_swarm_live.html"  # read on every load: edits show on refresh
MAX_ALIVE = 8  # cheeses on the board at once (play scenes have up to 4)


class Live:
    """One world, one instruction, one model at a time (a single viewer)."""

    def __init__(self, store, windows, specs, device):
        self.store, self.windows, self.specs, self.device = store, windows, specs, device
        self.models, self.lock = {}, threading.Lock()
        self.model_key, self.guidance, self.text, self.condition = next(iter(specs)), None, "", None
        self.reset(0)

    def policy(self):
        if self.model_key not in self.models:
            spec = self.specs[self.model_key]
            self.models[self.model_key] = spec["make"](spec["run"], self.device)
        return self.models[self.model_key]

    def reset(self, scene):
        self.scene = scene % len(self.windows)
        e, w0 = int(self.windows[self.scene][0]), int(self.windows[self.scene][1])
        self.world = world_at(self.store, torch.tensor([e]), torch.tensor([w0]), self.device)
        self.t, self.since = 0, goal_eval.snapshot(self.world.to("cpu"), 0)

    def static(self):
        w = self.world.to("cpu")
        g = w.mass.shape[-1]
        walls = w.walls[0].flatten().nonzero().flatten().tolist()
        boxes = [[*rect, PALETTE_NAMES[c]] for rect, c in zip(w.box_rect[0].tolist(), w.box_color[0].tolist()) if c >= 0 and rect[2] > 0 and rect[3] > 0]
        return {"grid": g, "walls": walls, "boxes": boxes, "side": cheese_side(g), "scene": self.scene, "scenes": len(self.windows)}

    def frame(self):
        w = self.world.to("cpu")
        robots = (w.mass[0] > 0.5).flatten().nonzero().flatten().tolist()
        cheeses = [[r, c, bool(a), PALETTE_NAMES[k]] for (r, c), a, k in zip(w.cheese_pos[0].tolist(), w.cheese_alive[0].tolist(), w.cheese_color[0].tolist()) if k >= 0]
        return {"t": self.t, "robots": robots, "cheeses": cheeses}

    def events(self):
        w = self.world.to("cpu")
        static = {"walls": w.walls[0].numpy(), "cheese_color": w.cheese_color[0].numpy(), "box_rect": w.box_rect[0].numpy(), "box_color": w.box_color[0].numpy()}
        return describe(window_facts(static, self.since, goal_eval.snapshot(w, 0)).atoms)

    def instruct(self, text, model_key, guidance):
        self.model_key, self.guidance = model_key, guidance
        self.text = text.strip()[:300]
        self.condition = self.policy().condition(self.text) if self.text else None
        self.since = goal_eval.snapshot(self.world.to("cpu"), 0)  # events count from here

    @torch.no_grad()
    def advance(self, n):
        frames = []
        if self.condition is None:
            return frames
        policy = self.policy()
        for _ in range(n):
            chunk = policy.sample(observe(self.world), self.condition, self.guidance)
            self.world, _ = step(self.world, chunk[:, 0])
            self.t += 1
            frames.append(self.frame())
        return frames

    def add_cheese(self, colour):
        """Revive an eaten cheese of that colour (else fill an empty slot, else add a slot) at a
        random free place: no wall, robot, box or other cheese under it. False if there is no
        room or MAX_ALIVE cheeses are already out."""
        w = self.world
        g, side = w.mass.shape[-1], cheese_side(w.mass.shape[-1])
        colours = w.cheese_color[0].tolist()
        index = PALETTE_NAMES.index(colour)
        dead = [k for k, (c, a) in enumerate(zip(colours, w.cheese_alive[0].tolist())) if c == index and not a]
        empty = [k for k, c in enumerate(colours) if c < 0]
        if not dead and not empty:
            if int(w.cheese_alive[0].sum()) >= MAX_ALIVE:
                return False
            # A new slot: the policy sees the rendered grid, whatever the number of slots.
            w = World(mass=w.mass, walls=w.walls, cheese_pos=torch.cat([w.cheese_pos, w.cheese_pos[:, :1]], 1), cheese_alive=torch.cat([w.cheese_alive, torch.zeros_like(w.cheese_alive[:, :1])], 1),
                      cheese_color=torch.cat([w.cheese_color, torch.full_like(w.cheese_color[:, :1], -1)], 1), box_rect=w.box_rect, box_color=w.box_color)
            empty = [w.cheese_pos.shape[1] - 1]
        slot = (dead or empty)[0]
        blocked = w.walls[0] | (w.mass[0] > 0) | box_masks(w)[0].any(0)
        for k, ((r, c), a) in enumerate(zip(w.cheese_pos[0].tolist(), w.cheese_alive[0].tolist())):
            if a:
                blocked[max(0, r - 1) : r + side + 1, max(0, c - 1) : c + side + 1] = True
        free = [(r, c) for r in range(g - side + 1) for c in range(g - side + 1) if not blocked[r : r + side, c : c + side].any()]
        if not free:
            return False
        r, c = random.choice(free)
        pos, alive, colour_index = w.cheese_pos.clone(), w.cheese_alive.clone(), w.cheese_color.clone()
        pos[0, slot] = torch.tensor([r, c], device=pos.device)
        alive[0, slot] = True
        colour_index[0, slot] = index
        self.world = World(mass=w.mass, walls=w.walls, cheese_pos=pos, cheese_alive=alive, cheese_color=colour_index, box_rect=w.box_rect, box_color=w.box_color)
        # The events baseline gets the new slot too, absent at the time of the instruction (as if
        # eaten), so the new cheese counts only once the swarm does something with it.
        k = pos.shape[1] - len(self.since["cheese_alive"])
        if k:
            self.since = {**self.since, "cheese_pos": np.concatenate([self.since["cheese_pos"], pos[0, -k:].cpu().numpy()]), "cheese_alive": np.concatenate([self.since["cheese_alive"], np.zeros(k, dtype=bool)])}
        else:
            self.since["cheese_alive"] = self.since["cheese_alive"].copy()
            self.since["cheese_pos"] = self.since["cheese_pos"].copy()
            self.since["cheese_pos"][slot] = pos[0, slot].cpu().numpy()
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", default="runs/lm-qwen-lora-c1")
    parser.add_argument("--swarm", default="runs/vla-rel-g32-swarm-text")
    parser.add_argument("--play_cache", default="data/vla/play_cache_g32")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    store = load_play_cache(args.play_cache)
    windows = load_data(Config(grid=32, data="play", play_cache=args.play_cache, val_windows=128, test_windows=512), "cpu")[5]
    specs = {
        "qwen": {"name": "Qwen LoRA", "detail": "0.6B, pretrained", "run": args.qwen, "make": QwenText, "guidance": 2.0},
        "swarm": {"name": "Swarm", "detail": "2.2M, from scratch", "run": args.swarm, "make": SwarmText, "guidance": 4.0},
    }
    live = Live(store, windows, specs, torch.device(args.device))

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, kind):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

        def do_GET(self):
            if urlparse(self.path).path == "/":
                self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            else:
                self.send(404, b"", "text/plain")

        def do_POST(self):
            route = urlparse(self.path).path
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                with live.lock:
                    if route == "/api/state":
                        out = {}
                    elif route == "/api/reset":
                        live.reset(int(body["scene"]))
                        live.instruct(live.text, live.model_key, live.guidance or specs[live.model_key]["guidance"])
                        out = {}
                    elif route == "/api/instruct":
                        live.instruct(body["text"], body["model"], float(body["guidance"]))
                        out = {}
                    elif route == "/api/step":
                        out = {"frames": live.advance(max(1, min(16, int(body.get("n", 4)))))}
                    elif route == "/api/cheese":
                        out = {"added": live.add_cheese(body["colour"])}
                    else:
                        return self.send(404, b"", "text/plain")
                    out.update({"static": live.static(), "frame": live.frame(), "events": live.events(), "text": live.text,
                                "models": [{"key": k, "name": v["name"], "detail": v["detail"], "guidance": v["guidance"]} for k, v in specs.items()],
                                "model": live.model_key})
                self.send(200, json.dumps(out).encode(), "application/json")
            except Exception as error:  # shown on the page
                self.send(500, json.dumps({"error": f"{type(error).__name__}: {error}"}).encode(), "application/json")

    print(f"open http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
