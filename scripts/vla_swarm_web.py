"""A local web page for prompting the swarm: pick a held-out test scene, type an instruction, watch
the rollout, and read what happened in the scorer's words (the score-to-vibes check).

    uv run python scripts/vla_swarm_web.py [--qwen runs/lm-qwen-lora-c1]
        [--swarm runs/vla-rel-g32-swarm-text] [--port 8777]

Then open http://localhost:8777. Each model loads the first time it is used. Only this machine
can reach the page (it listens on 127.0.0.1).
"""

import argparse
import json
import random
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from vla_demo_media import describe, episode, gif  # noqa: E402
from vla_swarm_demo import QwenText, SwarmText, act, scene_words  # noqa: E402

from beebot.vla.data import world_at  # noqa: E402
from beebot.vla.play_data import load_play_cache  # noqa: E402
from beebot.vla.render import frame  # noqa: E402
from beebot.vla.train import Config, load_data  # noqa: E402

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Swarm Prompt</title>
<style>
:root { --bg: #f6f5f1; --panel: #ffffff; --ink: #1d1d1b; --muted: #6b6a64; --line: #e2e0d8; --accent: #2f5d8a; --chip: #eef1f5; }
@media (prefers-color-scheme: dark) { :root { --bg: #141416; --panel: #1d1d21; --ink: #ecebe6; --muted: #9c9a93; --line: #2e2e33; --accent: #7fb0e0; --chip: #262a31; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.45 -apple-system, system-ui, sans-serif; }
main { display: grid; grid-template-columns: 340px 1fr; gap: 20px; max-width: 1200px; margin: 0 auto; padding: 20px 16px; }
@media (max-width: 800px) { main { grid-template-columns: 1fr; } }
h1 { font-size: 18px; margin: 0 0 12px; }
section { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }
label { display: block; color: var(--muted); font-size: 13px; margin: 10px 0 4px; }
input, textarea, select, button { font: inherit; color: var(--ink); background: var(--bg); border: 1px solid var(--line); border-radius: 7px; padding: 7px 9px; }
textarea { width: 100%; resize: vertical; min-height: 64px; }
button { cursor: pointer; }
button.primary { background: var(--accent); color: #fff; border-color: var(--accent); width: 100%; margin-top: 10px; padding: 9px; }
button:disabled { opacity: .5; cursor: wait; }
.row { display: flex; gap: 6px; align-items: center; }
.row input[type=number] { width: 80px; }
.models label { display: inline-flex; gap: 6px; margin-right: 14px; color: var(--ink); font-size: 14px; }
#scene img { width: 100%; image-rendering: pixelated; border-radius: 6px; margin-top: 8px; }
#scene .words { color: var(--muted); font-size: 13px; margin-top: 6px; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.chip { background: var(--chip); border: 0; border-radius: 999px; padding: 4px 10px; font-size: 13px; }
#feed { display: flex; flex-direction: column; gap: 14px; }
.card { display: grid; grid-template-columns: auto 1fr; gap: 14px; }
@media (max-width: 800px) { .card { grid-template-columns: 1fr; } }
.card img { width: 260px; max-width: 100%; image-rendering: pixelated; border-radius: 6px; }
.card h2 { font-size: 15px; margin: 0 0 4px; }
.meta { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
.card ul { margin: 2px 0 8px 18px; padding: 0; }
.empty { color: var(--muted); }
</style></head>
<body><main>
<div>
  <h1>Swarm Prompt</h1>
  <section>
    <div class="models" id="models"></div>
    <div class="row">
      <div><label for="guidance">Guidance</label><input id="guidance" type="number" step="0.5" min="1" value="2"></div>
      <div><label for="steps">Steps (trained on 64)</label><input id="steps" type="number" step="16" min="16" max="256" value="96"></div>
    </div>
    <label for="sceneN">Test scene</label>
    <div class="row">
      <button id="prev" aria-label="previous scene">&lsaquo;</button>
      <input id="sceneN" type="number" min="0" value="0">
      <button id="next" aria-label="next scene">&rsaquo;</button>
      <button id="rand">Random</button>
    </div>
    <div id="scene"></div>
    <label for="text">Instruction</label>
    <textarea id="text" placeholder="eat the blue cheese, then gather in the top left"></textarea>
    <div class="chips" id="chips"></div>
    <button class="primary" id="go">Run</button>
  </section>
</div>
<div id="feed"><p class="empty">Runs appear here, newest first. A rollout takes a few seconds (swarm) to a minute (Qwen).</p></div>
</main>
<script>
const $ = id => document.getElementById(id);
let info = null;
async function boot() {
  info = await (await fetch('/api/info')).json();
  $('models').innerHTML = info.models.map((m, k) => `<label><input type="radio" name="model" value="${m.key}" ${k ? '' : 'checked'}> ${m.name}</label>`).join('');
  document.querySelectorAll('input[name=model]').forEach(r => r.onchange = () => { $('guidance').value = info.models.find(m => m.key === r.value).guidance; });
  $('guidance').value = info.models[0].guidance;
  $('sceneN').max = info.scenes - 1;
  await scene(0);
}
async function scene(i) {
  i = ((i % info.scenes) + info.scenes) % info.scenes;
  $('sceneN').value = i;
  const s = await (await fetch('/api/scene?i=' + i)).json();
  $('scene').innerHTML = `<img src="${s.image}" alt="scene ${i}"><div class="words">${s.words}</div>`;
  $('chips').innerHTML = s.suggestions.map(t => `<button class="chip">${t}</button>`).join('');
  document.querySelectorAll('.chip').forEach(c => c.onclick = () => { $('text').value = c.textContent; });
}
$('prev').onclick = () => scene(+$('sceneN').value - 1);
$('next').onclick = () => scene(+$('sceneN').value + 1);
$('rand').onclick = () => scene(Math.floor(Math.random() * info.scenes));
$('sceneN').onchange = () => scene(+$('sceneN').value);
$('text').onkeydown = e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('go').click(); } };
$('go').onclick = async () => {
  const text = $('text').value.trim();
  if (!text) return;
  const model = document.querySelector('input[name=model]:checked').value;
  $('go').disabled = true; $('go').textContent = 'Running...';
  try {
    const r = await (await fetch('/api/run', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model, text, scene: +$('sceneN').value, guidance: +$('guidance').value, steps: +$('steps').value})})).json();
    if (r.error) throw new Error(r.error);
    const empty = document.querySelector('#feed .empty'); if (empty) empty.remove();
    const card = document.createElement('section');
    card.className = 'card';
    const list = xs => xs.length ? '<ul>' + xs.map(x => `<li>${x}</li>`).join('') + '</ul>' : '<p class="meta">nothing</p>';
    card.innerHTML = `<img src="${r.gif}" alt="rollout"><div><h2></h2>
      <div class="meta">${r.model} &middot; guidance ${r.guidance} &middot; scene ${r.scene} &middot; ${r.steps} steps &middot; ${r.seconds}s</div>
      <div class="meta">At the end</div>${list(r.at_end)}<div class="meta">Also at some point</div>${list(r.at_some_point)}</div>`;
    card.querySelector('h2').textContent = text;
    $('feed').prepend(card);
  } catch (e) { alert(e.message); }
  $('go').disabled = false; $('go').textContent = 'Run';
};
boot();
</script></body></html>
"""


def suggestions(store, e, t):
    names = [f"{c} cheese" for c in re.findall(r"(\w+) cheese", scene_words(store, e, t))]
    boxes = re.findall(r"(\w+) box", scene_words(store, e, t))
    out = [f"eat the {names[0]}"] if names else []
    out += [f"surround the {names[-1]}"] if names else []
    out += [f"go into the {boxes[0]} box"] if boxes else []
    out += ["split into two groups", "gather in the top left", "move down and right"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", default="runs/lm-qwen-lora-c1")
    parser.add_argument("--swarm", default="runs/vla-rel-g32-swarm-text")
    parser.add_argument("--play_cache", default="data/vla/play_cache_g32")
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--out", default="runs/swarm_web")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    store = load_play_cache(args.play_cache)
    test_windows = load_data(Config(grid=32, data="play", play_cache=args.play_cache, val_windows=128, test_windows=512), "cpu")[5]
    specs = {
        "qwen": {"name": f"Qwen LoRA ({Path(args.qwen).name})", "run": args.qwen, "make": QwenText, "guidance": 2.0},
        "swarm": {"name": f"Swarm 2.2M ({Path(args.swarm).name})", "run": args.swarm, "make": SwarmText, "guidance": 4.0},
    }
    loaded, lock = {}, threading.Lock()

    def model(key):
        if key not in loaded:
            loaded[key] = specs[key]["make"](specs[key]["run"], device)
        return loaded[key]

    def scene_image(i):
        path = out / f"scene{i}.png"
        if not path.exists():
            e, w0 = int(test_windows[i][0]), int(test_windows[i][1])
            world = world_at(store, torch.tensor([e]), torch.tensor([w0]), "cpu")
            frame(episode([world], 0), 0, 0).resize((480, 480), Image.NEAREST).save(path)
        return f"/media/{path.name}"

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, kind):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value, code=200):
            self.send(code, json.dumps(value).encode(), "application/json")

        def log_message(self, *_):
            pass

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/info":
                self.send_json({"scenes": len(test_windows), "models": [{"key": k, "name": v["name"], "guidance": v["guidance"]} for k, v in specs.items()]})
            elif url.path == "/api/scene":
                i = int(parse_qs(url.query)["i"][0]) % len(test_windows)
                e, w0 = int(test_windows[i][0]), int(test_windows[i][1])
                self.send_json({"image": scene_image(i), "words": scene_words(store, e, w0), "suggestions": suggestions(store, e, w0)})
            elif url.path.startswith("/media/"):
                path = out / Path(url.path).name
                if path.exists():
                    self.send(200, path.read_bytes(), "image/gif" if path.suffix == ".gif" else "image/png")
                else:
                    self.send(404, b"", "text/plain")
            else:
                self.send(404, b"", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/api/run":
                return self.send(404, b"", "text/plain")
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                key, text = body["model"], body["text"].strip()[:300]
                i, guidance = int(body["scene"]) % len(test_windows), float(body["guidance"])
                steps = max(16, min(256, int(body.get("steps", args.steps))))
                with lock:
                    t0 = time.time()
                    policy = model(key)
                    worlds, at_end, at_any = act(policy, store, test_windows[i], policy.condition(text), steps, guidance, device)
                    name = f"run{int(t0 * 1000)}-{random.randrange(1000)}.gif"
                    gif(out / name, [(text[:34], episode(worlds, 0))], len(worlds) - 1)
                self.send_json({"gif": f"/media/{name}", "at_end": describe(at_end), "at_some_point": describe(at_any - at_end), "model": specs[key]["name"], "guidance": guidance, "scene": i, "steps": steps, "seconds": round(time.time() - t0, 1)})
            except Exception as error:  # shown on the page
                self.send_json({"error": f"{type(error).__name__}: {error}"}, 500)

    print(f"open http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
