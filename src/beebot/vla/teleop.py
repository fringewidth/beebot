"""Steer the swarm by hand in a browser: a demo of the world and the attractor controller, and the
tool for recording human play.

    uv run vla-teleop [--port 8765] [--size 16] [--record]

Open http://localhost:8765. Click a square to place an attractor (up to 4); click it again to
remove it. Shift-click places a straight-mode attractor (pushes through cheeses). Keys: space
pauses, e toggles equal shares, c clears attractors, n starts a new scene. The side panel shows
the fact extractor's account of everything since the scene started.

With --record, every scene is an episode of 4 x size steps (64 at size 16); it is saved to data/vla/teleop/ in the same
format as program play, then a new scene starts.
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from beebot.vla.controller import Controller, commands_to_tensors
from beebot.vla.facts import window_facts
from beebot.vla.data import episode_steps
from beebot.vla.world import PALETTE_NAMES, PALETTE_RGB, cheese_side, sample_world, step

RECORD_DIR = Path("data/vla/teleop")


class Session:
    def __init__(self, record, size):
        self.record = record
        self.size = size
        self.episode_steps = episode_steps(size)
        self.lock = threading.Lock()
        self.rng = np.random.default_rng()
        self.new_scene()

    def new_scene(self):
        self.world = sample_world(self.rng, 1, size=self.size)
        self.controller = Controller()
        self.t = 0
        self.masses = [self.world.mass[0].clone()]
        self.positions = [self.world.cheese_pos[0].clone()]
        self.alive = [self.world.cheese_alive[0].clone()]
        self.fields = []
        self.start = self.snapshot()

    def snapshot(self):
        return {
            "mass": self.world.mass[0].numpy().copy(),
            "cheese_pos": self.world.cheese_pos[0].numpy().copy(),
            "cheese_alive": self.world.cheese_alive[0].numpy().copy(),
        }

    def static(self):
        return {
            "walls": self.world.walls[0].numpy(),
            "cheese_color": self.world.cheese_color[0].numpy(),
            "box_rect": self.world.box_rect[0].numpy(),
            "box_color": self.world.box_color[0].numpy(),
        }

    def advance(self, commands):
        points, valid, straight, share = commands_to_tensors([commands], 1)
        field = self.controller(self.world, points, valid, straight, share)
        self.world, _ = step(self.world, field)
        self.t += 1
        self.fields.append(field[0].clone())
        self.masses.append(self.world.mass[0].clone())
        self.positions.append(self.world.cheese_pos[0].clone())
        self.alive.append(self.world.cheese_alive[0].clone())
        if self.record and self.t >= self.episode_steps:
            self.save()
            self.new_scene()

    def save(self):
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        name = time.strftime("%Y%m%d-%H%M%S") + f"-human-{self.rng.integers(1 << 30):08x}"
        out = RECORD_DIR / name
        out.mkdir()
        s = self.static()
        np.savez_compressed(
            out / "episodes.npz",
            mass=torch.stack(self.masses).unsqueeze(0).numpy().astype(np.float16),
            field=torch.stack(self.fields).unsqueeze(0).numpy().astype(np.float16),
            cheese_pos=torch.stack(self.positions).unsqueeze(0).numpy().astype(np.int8),
            cheese_alive=torch.stack(self.alive).unsqueeze(0).numpy(),
            walls=s["walls"][None],
            cheese_color=s["cheese_color"][None],
            box_rect=s["box_rect"][None],
            box_color=s["box_color"][None],
        )
        (out / "meta.json").write_text(json.dumps({"program_id": name, "player": "human", "size": self.size, "steps": self.episode_steps, "phases": [[""] * self.episode_steps], "errors": [None]}))

    def state(self):
        w = self.world
        cheeses = [
            {"row": int(w.cheese_pos[0, k, 0]), "col": int(w.cheese_pos[0, k, 1]), "color": int(c), "alive": bool(w.cheese_alive[0, k])}
            for k, c in enumerate(w.cheese_color[0].tolist())
            if c >= 0
        ]
        boxes = [
            {"rect": w.box_rect[0, k].tolist(), "color": int(c)} for k, c in enumerate(w.box_color[0].tolist()) if c >= 0
        ]
        facts = window_facts(self.static(), self.start, self.snapshot())
        return {
            "t": self.t,
            "record": self.record,
            "episode_steps": self.episode_steps,
            "size": self.size,
            "cheese_side": cheese_side(self.size),
            "mass": np.round(w.mass[0].numpy(), 2).tolist(),
            "walls": w.walls[0].int().tolist(),
            "cheeses": cheeses,
            "boxes": boxes,
            "palette": [[int(255 * v) for v in rgb] for rgb in PALETTE_RGB.tolist()],
            "palette_names": list(PALETTE_NAMES),
            "facts": facts.lines,
        }


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Swarm Teleop</title>
<style>
:root { --bg: #111116; --panel: #1b1b22; --text: #e4e4ea; --muted: #9a9aa8; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 -apple-system, system-ui, sans-serif; }
main { display: flex; gap: 20px; padding: 16px; flex-wrap: wrap; }
canvas { background: #18181e; cursor: crosshair; border-radius: 6px; max-width: 100%; }
aside { background: var(--panel); padding: 12px 16px; border-radius: 6px; width: 360px; max-width: 100%; }
h1 { font-size: 16px; margin: 0 0 8px; }
.muted { color: var(--muted); }
kbd { background: #2a2a34; padding: 1px 5px; border-radius: 3px; }
ul { padding-left: 18px; margin: 6px 0; }
</style></head><body><main>
<canvas id="c" width="576" height="576"></canvas>
<aside>
<h1>Swarm teleop</h1>
<div id="status" class="muted"></div>
<p class="muted">Click: attractor (click again to remove). Shift-click: straight mode (push).
<kbd>space</kbd> pause, <kbd>e</kbd> equal shares, <kbd>c</kbd> clear, <kbd>n</kbd> new scene.</p>
<div><b>What happened since the scene started</b></div>
<ul id="facts"></ul>
</aside></main>
<script>
let N = 16, S = 36;
const canvas = document.getElementById('c'), ctx = canvas.getContext('2d');
let attractors = [], paused = false, equal = false, state = null, busy = false;
function rgb(a, k = 1) { return `rgb(${a[0]*k},${a[1]*k},${a[2]*k})`; }
function commands() {
  return attractors.map(a => ({row: a.row, col: a.col, straight: a.straight, share: equal ? 1 / attractors.length : null}));
}
canvas.addEventListener('click', e => {
  const r = canvas.getBoundingClientRect(), scale = canvas.width / r.width;
  const col = Math.floor((e.clientX - r.left) * scale / S), row = Math.floor((e.clientY - r.top) * scale / S);
  const i = attractors.findIndex(a => a.row === row && a.col === col);
  if (i >= 0) attractors.splice(i, 1);
  else if (attractors.length < 4) attractors.push({row, col, straight: e.shiftKey});
  draw();
});
document.addEventListener('keydown', e => {
  if (e.key === ' ') { paused = !paused; e.preventDefault(); }
  if (e.key === 'e') equal = !equal;
  if (e.key === 'c') attractors = [];
  if (e.key === 'n') { attractors = []; post('/new', {}); }
  draw();
});
async function post(path, body) {
  const res = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  state = await res.json();
  draw();
}
function draw() {
  if (!state) return;
  N = state.size; S = canvas.width / N;
  ctx.fillStyle = '#18181e'; ctx.fillRect(0, 0, N * S, N * S);
  for (const b of state.boxes) {
    const [y, x, h, w] = b.rect, c = state.palette[b.color];
    ctx.fillStyle = rgb(c, 0.3); ctx.fillRect(x * S, y * S, w * S, h * S);
    ctx.strokeStyle = rgb(c); ctx.lineWidth = 3; ctx.strokeRect(x * S + 1.5, y * S + 1.5, w * S - 3, h * S - 3);
  }
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    if (state.walls[y][x]) { ctx.fillStyle = '#605c68'; ctx.fillRect(x * S, y * S, S, S); continue; }
    const m = state.mass[y][x];
    if (m > 0.01) {
      ctx.fillStyle = `rgba(240,240,245,${Math.min(m, 1)})`; ctx.fillRect(x * S + 1, y * S + 1, S - 2, S - 2);
      if (m > 1.02) { ctx.fillStyle = 'rgba(120,180,255,0.9)'; ctx.fillRect(x * S + S/2 - S/9, y * S + S/2 - S/9, 2*S/9, 2*S/9); }
    }
  }
  for (const c of state.cheeses) if (c.alive) {
    ctx.fillStyle = rgb(state.palette[c.color]); ctx.fillRect(c.col * S + 2, c.row * S + 2, state.cheese_side * S - 4, state.cheese_side * S - 4);
  }
  for (const a of attractors) {
    const cx = a.col * S + S / 2, cy = a.row * S + S / 2;
    ctx.strokeStyle = a.straight ? '#ff6b6b' : '#7cd4ff'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(cx, cy, S / 2.6, 0, 2 * Math.PI); ctx.stroke();
    if (a.straight) { const d = S / 6; ctx.beginPath(); ctx.moveTo(cx - d, cy - d); ctx.lineTo(cx + d, cy + d); ctx.moveTo(cx + d, cy - d); ctx.lineTo(cx - d, cy + d); ctx.stroke(); }
  }
  const total = state.mass.flat().reduce((s, v) => s + v, 0);
  document.getElementById('status').textContent =
    `step ${state.t}${state.record ? ' / ' + state.episode_steps + ' (recording)' : ''}` +
    ` | mass ${total.toFixed(1)} | ${paused ? 'paused' : 'running'} | shares ${equal ? 'equal' : 'nearest'}`;
  const ul = document.getElementById('facts'); ul.innerHTML = '';
  for (const f of (state.facts.length ? state.facts : ['Nothing yet.'])) { const li = document.createElement('li'); li.textContent = f; ul.appendChild(li); }
}
async function loop() {
  if (!busy) { busy = true; try { await post('/step', {commands: commands(), paused}); } finally { busy = false; } }
  setTimeout(loop, 125);
}
loop();
</script></body></html>
"""


def make_handler(session):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, body, content_type):
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(PAGE, "text/html; charset=utf-8")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            with session.lock:
                if self.path == "/new":
                    session.new_scene()
                elif self.path == "/step" and not body.get("paused"):
                    session.advance(body.get("commands", []))
                state = session.state()
            self.reply(json.dumps(state), "application/json")

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--size", type=int, default=16)
    args = parser.parse_args()
    session = Session(args.record, args.size)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(session))
    print(f"open http://localhost:{args.port}{'  (recording to ' + str(RECORD_DIR) + ')' if args.record else ''}")
    server.serve_forever()


if __name__ == "__main__":
    main()
