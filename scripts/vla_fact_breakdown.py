"""Which facts the policy gets right: per atom type, recall (goal atoms the rollout also shows at
its best-matching snapshot) and how often the rollout adds atoms of that type the goal lacks.
Also for replay and do-nothing as references.

    .venv/bin/python scripts/vla_fact_breakdown.py runs/vla-r16-mean1-long
"""

import json
import sys
from collections import Counter
from pathlib import Path

import torch

from beebot.vla import goal_eval
from beebot.vla.data import make_windows, world_at
from beebot.vla.facts import window_facts
from beebot.vla.model import Policy
from beebot.vla.train import load_or_generate
from beebot.vla.world import observe, step

run_dir = Path(sys.argv[1])
cfg = json.loads((run_dir / "config.json").read_text())
device = torch.device("mps")
policy = Policy(cfg["grid"], width=cfg.get("width", 192), expert_width=cfg.get("expert_width", 128), layers=cfg.get("layers", 4), goal_tokens=cfg["goal_tokens"], goal_pool=cfg["goal_pool"], objective=cfg.get("objective", "flow")).to(device)
policy.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=device), strict=False)
policy.eval()
store = load_or_generate(cfg["grid"], cfg["val_episodes"], 10_000 + cfg["seed"], device)
windows = goal_eval.eventful_windows(store, make_windows(store, cfg["grid"]), device, 256, torch.Generator().manual_seed(1))
e, w0, w1 = windows[:, 0], windows[:, 1], windows[:, 2]
horizon = torch.ceil(goal_eval.SLACK * (w1 - w0).float()).long()

world, goal = world_at(store, e, w0, device), world_at(store, e, w1, device)
z = policy.goal_z(policy.goal_encoder(torch.cat([observe(world), observe(goal)], 1)))
frames = []
with torch.no_grad():
    for t in range(int(horizon.max())):
        if t % 4 == 0:
            chunk = policy.sample(observe(world), z)
        world, _ = step(world, chunk[:, t % 4])
        if (t + 1) % 4 == 0 or (t + 1) in set(horizon.tolist()):
            frames.append((t + 1, world))

hits, totals, extras = Counter(), Counter(), Counter()
for i in range(len(windows)):
    static = {k: store[k][e[i]].numpy() for k in ("walls", "cheese_color", "box_rect", "box_color")}
    start = goal_eval.stored_snapshot(store, e[i], w0[i])
    target = window_facts(static, start, goal_eval.stored_snapshot(store, e[i], w1[i])).atoms
    if not target:
        continue
    best, best_atoms = -1.0, set()
    for steps_taken, world_t in frames:
        if steps_taken > int(horizon[i]):
            break
        atoms = window_facts(static, start, goal_eval.snapshot(world_t, i)).atoms
        score = len(atoms & target) / len(atoms | target)
        if score > best:
            best, best_atoms = score, atoms
    for atom in target:
        totals[atom[0]] += 1
        hits[atom[0]] += atom in best_atoms
    for atom in best_atoms - target:
        extras[atom[0]] += 1
print(f"{'atom type':12s} {'in goals':>8s} {'recall':>7s} {'extra':>6s}")
for kind in sorted(totals, key=lambda k: -totals[k]):
    print(f"{kind:12s} {totals[kind]:8d} {hits[kind] / totals[kind]:7.2f} {extras[kind]:6d}")
