"""Why do eats fail: localisation (never gets to the right cheese) or execution (gets there, can't
close the ring)? Runs a finished goal-conditioned run on held-out test-program windows whose goal
eats a cheese, and also replays the recorded fields on the same windows for reference.

    uv run python scripts/vla_cheese_diagnostic.py runs/vla-d-g32-play-goal [--windows 160]

Closed loop, per target cheese (alive at the window start, eaten by its end): eaten or not; the
best ring coverage reached (share of the cheese's open ring squares holding a robot; eating takes
0.75); the closest any robot came (Chebyshev squares from the cheese's footprint); whether a
different cheese was eaten instead. Open loop, on recorded states inside the same windows: how
often the policy's first-step direction matches the recorded one, on robots next to a target
cheese versus all other robots. Ring closing: from the recorded state a few steps before the
recorded eat, do the recorded actions or the policy finish the eat? If the policy fails even from
there, it never learned to close a ring; if it succeeds there but not from the window start, its
own approach never reaches such states (compounding).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from beebot.vla.data import decode_field, encode_field, make_windows, world_at
from beebot.vla.facts import ring_share
from beebot.vla.goal_eval import SLACK, policy_obs
from beebot.vla.model import Policy
from beebot.vla.play_data import load_play, run_dirs
from beebot.vla.train import program_split
from beebot.vla.world import cheese_side, observe, step


def target_cheeses(store, e, w0, w1):
    alive0, alive1 = store["cheese_alive"][e, w0], store["cheese_alive"][e, w1]
    return [k for k in range(len(alive0)) if alive0[k] and not alive1[k]]


def closest(mass, pos, side):
    rows, cols = np.nonzero(mass > 0.5)
    if len(rows) == 0:
        return 99
    dy = np.maximum(np.maximum(pos[0] - rows, rows - (pos[0] + side - 1)), 0)
    dx = np.maximum(np.maximum(pos[1] - cols, cols - (pos[1] + side - 1)), 0)
    return int(np.maximum(dy, dx).min())


def rollout(policy, store, windows, device, mode):
    """Per target cheese: dict(eaten, best_cover, closest, other_eaten)."""
    out = []
    w = windows
    e, w0, w1 = w[:, 0], w[:, 1], w[:, 2]
    world = world_at(store, e, w0, device)
    goal = world_at(store, e, w1, device)
    horizon = torch.ceil(SLACK * (w1 - w0).float()).long()
    size = world.mass.shape[-1]
    side = cheese_side(size)
    if mode == "policy":
        z = policy.goal_z(policy.goal_encoder(torch.cat([observe(world), observe(goal)], dim=1)))
    targets = [target_cheeses(store, int(e[i]), int(w0[i]), int(w1[i])) for i in range(len(w))]
    best_cover = [{k: 0.0 for k in ks} for ks in targets]
    near = [{k: 99 for k in ks} for ks in targets]
    alive_start = world.cheese_alive.cpu().numpy().copy()
    walls = world.walls.cpu().numpy()
    eaten_any = np.zeros_like(alive_start)
    chunk = None
    for t in range(int(horizon.max())):
        if mode == "policy":
            if t % 4 == 0:
                chunk = policy.sample(policy_obs(policy, world, store, e, torch.minimum(w0 + t, w1 - 1)), z)
            field = chunk[:, t % 4]
        else:
            t_rec = (w0 + t).clamp(max=store["field"].shape[1] - 1)
            field = decode_field(store["field"][e, t_rec], device)
            field = torch.where(((w0 + t) < w1).to(device).view(-1, 1, 1, 1), field, torch.zeros_like(field))
        world, _ = step(world, field)
        mass = world.mass.cpu().numpy()
        pos = world.cheese_pos.cpu().numpy()
        alive = world.cheese_alive.cpu().numpy()
        for i in range(len(w)):
            if t >= int(horizon[i]):
                continue
            eaten_any[i] |= alive_start[i] & ~alive[i]
            for k in targets[i]:
                if alive[i, k]:
                    best_cover[i][k] = max(best_cover[i][k], float(ring_share(mass[i], tuple(pos[i, k]), walls[i])))
                    near[i][k] = min(near[i][k], closest(mass[i], pos[i, k], side))
    for i in range(len(w)):
        for k in targets[i]:
            others = [j for j in range(eaten_any.shape[1]) if j != k and eaten_any[i, j] and j not in targets[i]]
            out.append({"eaten": bool(eaten_any[i, k]), "best_cover": 1.0 if eaten_any[i, k] else best_cover[i][k], "closest": 0 if eaten_any[i, k] else near[i][k], "other_eaten": bool(others)})
    return out


def report(name, rows):
    n = len(rows)
    failed = [r for r in rows if not r["eaten"]]
    print(f"\n{name}: {n} target cheeses, eaten {sum(r['eaten'] for r in rows) / n:.0%}")
    if not failed:
        return
    covers = np.array([r["best_cover"] for r in failed])
    dist = np.array([r["closest"] for r in failed])
    print(f"  failed eats ({len(failed)}):")
    print(f"    never within 2 squares of the cheese: {np.mean(dist > 2):.0%}")
    print(f"    best ring cover <25%: {np.mean(covers < 0.25):.0%}, 25-50%: {np.mean((covers >= 0.25) & (covers < 0.5)):.0%}, 50-75%: {np.mean((covers >= 0.5) & (covers < 0.75)):.0%}, >=75% (covered but not eaten): {np.mean(covers >= 0.75):.0%}")
    print(f"    a different cheese eaten instead: {np.mean([r['other_eaten'] for r in failed]):.0%}")


@torch.no_grad()
def open_loop(policy, store, windows, device, samples_per_window=4, seed=0):
    """Direction agreement with the recorded field on robot squares next to a target cheese vs elsewhere."""
    rng = np.random.default_rng(seed)
    near_hits = near_n = far_hits = far_n = 0
    stay_counts = {"near": [0, 0, 0, 0], "far": [0, 0, 0, 0]}  # recorded stays, predicted stays, both, robots
    for e, w0, w1 in windows.tolist():
        targets = target_cheeses(store, e, w0, w1)
        times = rng.integers(w0, w1, size=samples_per_window)
        ee = torch.full((len(times),), e)
        tt = torch.tensor(times)
        world = world_at(store, ee, tt, device)
        goal = world_at(store, ee, torch.full((len(times),), w1), device)
        start = world_at(store, ee, torch.full((len(times),), w0), device)
        z = policy.goal_z(policy.goal_encoder(torch.cat([observe(start), observe(goal)], dim=1)))
        chunk = policy.sample(policy_obs(policy, world, store, ee, tt), z)
        predicted = encode_field(chunk[:, 0]).cpu().numpy()
        recorded = store["field"][ee, tt].numpy()
        mass = world.mass.cpu().numpy() > 0.5
        size = mass.shape[-1]
        side = cheese_side(size)
        for i in range(len(times)):
            close = np.zeros((size, size), dtype=bool)
            for k in targets:
                if store["cheese_alive"][e, times[i], k]:
                    y, x = store["cheese_pos"][e, times[i], k].tolist()
                    close[max(y - 2, 0) : y + side + 2, max(x - 2, 0) : x + side + 2] = True
            match = predicted[i] == recorded[i]
            for region, where in (("near", mass[i] & close), ("far", mass[i] & ~close)):
                stay_counts[region][0] += int((recorded[i][where] == 8).sum())
                stay_counts[region][1] += int((predicted[i][where] == 8).sum())
                stay_counts[region][2] += int(((recorded[i] == 8) & (predicted[i] == 8))[where].sum())
                stay_counts[region][3] += int(where.sum())
            near_hits += int((match & mass[i] & close).sum())
            near_n += int((mass[i] & close).sum())
            far_hits += int((match & mass[i] & ~close).sum())
            far_n += int((mass[i] & ~close).sum())
    for region, (rec, pred, both, n) in stay_counts.items():
        print(f"stay, robots {region} a target cheese: recorded {rec / max(n, 1):.0%} of robots, policy {pred / max(n, 1):.0%}; of the recorded stays the policy also stays on {both / max(rec, 1):.0%}")
    print(f"\nopen loop, first-step direction matches the recorded one: robots within 2 squares of a target cheese {near_hits / max(near_n, 1):.0%} (n={near_n}), other robots {far_hits / max(far_n, 1):.0%} (n={far_n})")


@torch.no_grad()
def closing_test(policy, store, windows, device, lead):
    """From the recorded state `lead` steps before the recorded eat, run the recorded fields or
    the policy (closed loop, same goal) for lead + 8 steps; returns (replay eaten, policy eaten)."""
    starts, targets = [], []
    for e, w0, w1 in windows.tolist():
        for k in target_cheeses(store, e, w0, w1):
            alive = store["cheese_alive"][e, w0 : w1 + 1, k].numpy()
            t_eat = w0 + int(np.argmin(alive))
            if t_eat - lead >= w0:
                starts.append((e, w0, w1, t_eat - lead))
                targets.append(k)
    if not starts:
        return None
    s = torch.tensor(starts)
    e, w0, w1, t0 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    k = torch.tensor(targets)
    horizon = lead + 8
    rates = []
    for mode in ("replay", "policy"):
        world = world_at(store, e, t0, device)
        if mode == "policy":
            z = policy.goal_z(policy.goal_encoder(torch.cat([observe(world_at(store, e, w0, device)), observe(world_at(store, e, w1, device))], dim=1)))
        chunk = None
        for t in range(horizon):
            if mode == "policy":
                if t % 4 == 0:
                    chunk = policy.sample(policy_obs(policy, world, store, e, torch.minimum(t0 + t, w1 - 1)), z)
                field = chunk[:, t % 4]
            else:
                field = decode_field(store["field"][e, (t0 + t).clamp(max=store["field"].shape[1] - 1)], device)
            world, _ = step(world, field)
        eaten = ~world.cheese_alive.cpu()[torch.arange(len(k)), k]
        rates.append(eaten)
    # Split by how far the window's goal frame lies after the recorded eat: when it is close, the
    # goal frame still shows the robots around where the cheese was.
    after = (w1 - (t0 + lead)).cpu()
    for name, mask in (("eat within 4 steps of the window end", after <= 4), ("eat more than 4 steps before the window end", after > 4)):
        if mask.any():
            print(f"    {name}: n={int(mask.sum())}, recorded {rates[0][mask].float().mean():.0%}, policy {rates[1][mask].float().mean():.0%}")
    return len(starts), rates[0].float().mean().item(), rates[1].float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--windows", type=int, default=160)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--closing-only", action="store_true")
    parser.add_argument("--open-loop-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((Path(args.run) / "config.json").read_text())
    dirs = [d for d in run_dirs(cfg["play_root"], size=cfg["grid"]) if program_split(d.name) == "test"]
    store = load_play(dirs)
    windows = make_windows(store, cfg["grid"])
    eat = [i for i, (e, w0, w1) in enumerate(windows.tolist()) if target_cheeses(store, e, w0, w1)]
    order = torch.randperm(len(eat), generator=torch.Generator().manual_seed(0))[: args.windows]
    windows = windows[torch.tensor(eat)[order]]
    print(f"{len(dirs)} test programs, {len(eat)} windows that eat a cheese; using {len(windows)}")
    policy = Policy(cfg["grid"], width=cfg["width"], expert_width=cfg["expert_width"], layers=cfg["layers"], goal_tokens=cfg["goal_tokens"], goal_pool=cfg["goal_pool"], objective=cfg["objective"], world_model=cfg.get("world_model", False), insulate=cfg.get("insulate", False)).to(args.device)
    policy.load_state_dict(torch.load(Path(args.run) / "model_best.pt", map_location=args.device))
    policy.eval()
    torch.manual_seed(0)
    if args.open_loop_only:
        open_loop(policy, store, windows, args.device)
        return
    if not args.closing_only:
        report("replay (recorded fields)", rollout(None, store, windows, args.device, "replay"))
        report("policy", rollout(policy, store, windows, args.device, "policy"))
        open_loop(policy, store, windows[:60], args.device)
    print("\nring closing from recorded states just before the recorded eat (lead + 8 steps):")
    for lead in (4, 12):
        result = closing_test(policy, store, windows, args.device, lead)
        if result:
            n, replay, policy_rate = result
            print(f"  {lead:2d} steps before: n={n}, recorded actions eat {replay:.0%}, policy eats {policy_rate:.0%}")


if __name__ == "__main__":
    main()
