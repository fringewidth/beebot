"""Closed-loop goal reaching: start a scene at a held-out window's first frame, condition on the
window's (start, end) frames, run the policy, and check whether the scene reaches the end frame.

Scored identically at every grid size:
- mass: overlap between the swarm's mass and the goal's, each pooled onto an 8x8 grid of cells
  (one cell is one cheese side), as 1 - total variation distance; at least MASS_OVERLAP;
- cheeses: the same ones alive, and every alive cheese within one cheese side of its goal spot.
"Reached" means the scene met both at any step within SLACK times the window's length; "at_end"
means it still meets them at the end of that time.

Facts score: the semantic version. The goal's facts (window_facts from the start frame to the
goal frame: direction and end region of the move, number of groups, cheeses eaten or pushed and
which way, boxes entered or left, ...) are compared with the rollout's facts (start frame to the
rollout's frame) as the Jaccard overlap of their atom sets, checked every FACT_EVERY steps; the
score is the best within the horizon ("fact_score") and the one at the horizon's end
("fact_score_end"). Windows whose goal has no facts are left out of it.

The replay baseline (re-applying the recorded fields) is the practical ceiling, slightly under
100%: mass is stored as float16, and pushes and eats are hard thresholds, so a rounding
difference occasionally flips one.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from beebot.vla.controller import MAX_ATTRACTORS, Controller
from beebot.vla.data import CHUNK, EXECUTE, command_map, decode_field, encode_field, world_at
from beebot.vla.facts import window_facts
from beebot.vla.world import N_CHANNELS, cheese_side, observe, step

MASS_OVERLAP = 0.75
SLACK = 1.5
FACT_EVERY = 4


def goal_blobs(mass, k=MAX_ATTRACTORS, iters=20):
    """(B, k, 2) long centers and (B, k) mass shares of each scene's mass, by weighted k-means
    started from the k heaviest squares. Used by the goal-reading oracle baseline."""
    b, g, _ = mass.shape
    rows, cols = torch.meshgrid(torch.arange(g, device=mass.device), torch.arange(g, device=mass.device), indexing="ij")
    coords = torch.stack([rows, cols], -1).view(1, -1, 2).float().expand(b, -1, -1)
    weight = mass.view(b, -1)
    centers = coords.gather(1, weight.topk(k, dim=1).indices.unsqueeze(-1).expand(-1, -1, 2)).clone()
    for _ in range(iters):
        nearest = torch.cdist(coords, centers).argmin(-1)
        members = F.one_hot(nearest, k).float() * weight.unsqueeze(-1)
        total = members.sum(1)
        centers = torch.where(total.unsqueeze(-1) > 1e-6, (members.transpose(1, 2) @ coords) / total.unsqueeze(-1).clamp(min=1e-6), centers)
    return centers.round().long(), total / total.sum(1, keepdim=True)


def policy_obs(policy, world, store, e, t):
    """The policy's input: the observation, plus the player's recorded commands at step t for a
    command-oracle diagnostic policy (one with extra input channels)."""
    obs = observe(world)
    if policy.obs_channels > N_CHANNELS:
        obs = torch.cat([obs, command_map(store, e, t, obs.shape[-1], obs.device)], dim=1)
    return obs


def snapshot(world, i):
    return {"mass": world.mass[i].float().cpu().numpy(), "cheese_pos": world.cheese_pos[i].cpu().numpy(), "cheese_alive": world.cheese_alive[i].cpu().numpy()}


def stored_snapshot(store, e, t):
    return {"mass": store["mass"][e, t].float().numpy(), "cheese_pos": store["cheese_pos"][e, t].numpy(), "cheese_alive": store["cheese_alive"][e, t].numpy()}


RECALL_TYPES = ("pushed", "ate", "around", "groups")  # the hard atom types, reported on their own


def fact_scores(static, start, goal, rollout):
    """(best Jaccard over the rollout snapshots, Jaccard at the last one, {atom type: (hits, total)}
    at the best snapshot, the number of goal atoms) between the goal's fact atoms and the
    rollout's, both from the window's start; None if the goal has no facts."""
    target = window_facts(static, start, goal).atoms
    if not target:
        return None
    found = [window_facts(static, start, snap).atoms for snap in rollout]
    scores = [len(atoms & target) / len(atoms | target) for atoms in found]
    best = found[max(range(len(scores)), key=scores.__getitem__)]
    recall = {kind: (sum(a in best for a in target if a[0] == kind), sum(a[0] == kind for a in target)) for kind in RECALL_TYPES}
    return max(scores), scores[-1], recall, len(target)


DISRUPTIVE = ("ate", "pushed")  # events that cannot be undone: done unasked, they count against a rollout


def requested_scores(static, start, rollout, requested):
    """For an instruction that asks for the atoms in requested (a set; empty: nothing asked):
    (the best fraction of them present at one snapshot, the fraction at the last snapshot, whether
    the last snapshot holds a disruptive atom (DISRUPTIVE) that was not asked for), atoms taken
    from the window's start as in fact_scores; None for the fractions when nothing was asked."""
    found = [window_facts(static, start, snap).atoms for snap in rollout]
    fractions = [len(atoms & requested) / len(requested) for atoms in found] if requested else [None]
    unasked = any(a[0] in DISRUPTIVE and a not in requested for a in found[-1])
    return max(fractions) if requested else None, fractions[-1], unasked


def match(world, goal):
    """(B,) bool success, the (B,) mass overlap, and (B,) bool for the cheese condition alone."""
    g = world.mass.shape[-1]
    cell = g // 8
    a = F.avg_pool2d(world.mass.unsqueeze(1), cell).flatten(1)
    b = F.avg_pool2d(goal.mass.unsqueeze(1), cell).flatten(1)
    a = a / a.sum(1, keepdim=True).clamp(min=1e-9)
    b = b / b.sum(1, keepdim=True).clamp(min=1e-9)
    overlap = 1 - 0.5 * (a - b).abs().sum(1)
    same_alive = (world.cheese_alive == goal.cheese_alive).all(1)
    near = (world.cheese_pos - goal.cheese_pos).abs().amax(-1) <= cheese_side(g)
    cheeses_ok = same_alive & (near | ~goal.cheese_alive).all(1)
    return (overlap >= MASS_OVERLAP) & cheeses_ok, overlap, cheeses_ok


@torch.no_grad()
def run(policy, store, windows, device, mode="policy", guidance=1.0, batch_size=128, text_emb=None, execute=EXECUTE, sample_steps=10, snap=False, requested=None):
    """mode: 'policy', 'zero' (no movement), 'replay' (the recorded fields) or 'oracle' (reads the
    goal frame: the scripted controller with one attractor per goal mass blob, shares by blob mass;
    it never pushes on purpose). Returns per-window tensors.

    With text_emb (N, D), one caption embedding per window, the policy is conditioned on the
    caption through the text head instead of on the window's frames. execute: chunk steps run
    before deciding again; sample_steps: Euler steps per decision; snap: round each generated
    vector to the nearest of the stored actions (8 unit directions or zero) before executing it.
    requested: one list of atoms per window (what its instruction asks for; requested_scores), or
    None; with it the result also holds requested_best, requested_end and unasked_disruptive."""
    reached, met_at_end, overlaps, cheeses, facts_best, facts_end, fact_counts = [], [], [], [], [], [], []
    asked_best, asked_end, unasked = [], [], []
    recall = {kind: [0, 0] for kind in RECALL_TYPES}
    for start in range(0, len(windows), batch_size):
        w = windows[start : start + batch_size]
        e, w0, w1 = w[:, 0], w[:, 1], w[:, 2]
        world = world_at(store, e, w0, device)
        goal = world_at(store, e, w1, device)
        horizon = torch.ceil(SLACK * (w1 - w0).float()).long().to(device)
        if mode == "policy" and text_emb is not None:
            z = policy.goal_z(policy.text_head(text_emb[start : start + batch_size].float().to(device)))
        elif mode == "policy":
            frames = torch.cat([observe(world), observe(goal)], dim=1)
            z = policy.goal_z(policy.goal_encoder(frames))
        ok_any = torch.zeros(len(w), dtype=torch.bool, device=device)
        ok_end = torch.zeros(len(w), dtype=torch.bool, device=device)
        overlap_end = torch.zeros(len(w), device=device)
        cheese_end = torch.zeros(len(w), dtype=torch.bool, device=device)
        chunk = None
        if mode == "oracle":
            controller = Controller()
            centers, share = goal_blobs(goal.mass)
            attract = share > 0.02
            share = torch.where(attract, share, torch.full_like(share, float("nan")))
        frames = []
        horizon_ends = set(horizon.tolist())
        for t in range(int(horizon.max())):
            if mode == "policy":
                if t % execute == 0:
                    # A command-oracle policy gets the commands of the recorded step t, held at the
                    # window's last step once the window is over.
                    obs = policy_obs(policy, world, store, e, torch.minimum(w0 + t, w1 - 1))
                    chunk = policy.sample(obs, z, steps=sample_steps, guidance=guidance)
                field = chunk[:, t % execute]
                if snap:
                    field = decode_field(encode_field(field), device)
            elif mode == "oracle":
                field = controller(world, centers, attract, torch.zeros_like(attract), share)
            elif mode == "replay":
                t_rec = (w0 + t).clamp(max=store["field"].shape[1] - 1)
                recorded = decode_field(store["field"][e, t_rec], device)
                field = torch.where(((w0 + t) < w1).to(device).view(-1, 1, 1, 1), recorded, torch.zeros_like(recorded))
            else:
                field = torch.zeros(len(w), 2, *world.mass.shape[-2:], device=device)
            world, _ = step(world, field)
            ok, overlap, cheese_ok = match(world, goal)
            live = t < horizon
            ok_any |= ok & live
            last = t == horizon - 1
            ok_end = torch.where(last, ok, ok_end)
            overlap_end = torch.where(last, overlap, overlap_end)
            cheese_end = torch.where(last, cheese_ok, cheese_end)
            # Every FACT_EVERY steps, and at every window's horizon end, so each window's last
            # snapshot is its state exactly at its horizon.
            if (t + 1) % FACT_EVERY == 0 or (t + 1) in horizon_ends:
                frames.append((t + 1, world))
        for i in range(len(w)):
            static = {k: store[k][e[i]].numpy() for k in ("walls", "cheese_color", "box_rect", "box_color")}
            h = int(horizon[i])
            rollout = [snapshot(world_t, i) for steps_taken, world_t in frames if steps_taken <= h]
            scores = fact_scores(static, stored_snapshot(store, e[i], w0[i]), stored_snapshot(store, e[i], w1[i]), rollout)
            facts_best.append(float("nan") if scores is None else scores[0])
            facts_end.append(float("nan") if scores is None else scores[1])
            fact_counts.append(0 if scores is None else scores[3])
            if requested is not None:
                asked = {tuple(a) for a in requested[start + i]}
                best_asked, end_asked, disrupted = requested_scores(static, stored_snapshot(store, e[i], w0[i]), rollout, asked)
                asked_best.append(float("nan") if best_asked is None else best_asked)
                asked_end.append(float("nan") if end_asked is None else end_asked)
                unasked.append(disrupted)
            if scores is not None:
                for kind, (hit, total) in scores[2].items():
                    recall[kind][0] += hit
                    recall[kind][1] += total
        reached.append(ok_any.cpu())
        met_at_end.append(ok_end.cpu())
        overlaps.append(overlap_end.cpu())
        cheeses.append(cheese_end.cpu())
    return {
        "reached": torch.cat(reached),
        "at_end": torch.cat(met_at_end),
        "overlap": torch.cat(overlaps),
        "cheese_ok": torch.cat(cheeses),
        "fact_score": torch.tensor(facts_best),
        "fact_score_end": torch.tensor(facts_end),
        "fact_count": torch.tensor(fact_counts),
        "recall": recall,
        **({"requested_best": torch.tensor(asked_best), "requested_end": torch.tensor(asked_end), "unasked_disruptive": torch.tensor(unasked)} if requested is not None else {}),
    }


def summary(result):
    n = len(result["reached"])
    p = result["reached"].float().mean().item()
    return {
        "n": n,
        "reached": round(p, 4),
        "reached_se": round(math.sqrt(max(p * (1 - p), 1e-9) / n), 4),
        "at_end": round(result["at_end"].float().mean().item(), 4),
        "overlap": round(result["overlap"].mean().item(), 4),
        "fact_score": round(np.nanmean(result["fact_score"].numpy()).item(), 4),
        "fact_score_end": round(np.nanmean(result["fact_score_end"].numpy()).item(), 4),
        **{f"recall_{kind}": round(hit / total, 4) if total else None for kind, (hit, total) in result.get("recall", {}).items()},
        **by_event_count(result),
        **({
            "requested_best": round(np.nanmean(result["requested_best"].numpy()).item(), 4),
            "requested_end": round(np.nanmean(result["requested_end"].numpy()).item(), 4),
            "unasked_disruptive": round(result["unasked_disruptive"].float().mean().item(), 4),
        } if "requested_best" in result else {}),
    }


EVENT_BUCKETS = ((1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 99, "4plus"))  # about equal shares of g32 play windows


def by_event_count(result):
    """Fact score sliced by how many events (fact atoms) the goal window holds: a task-complexity
    axis. fact_events_<k> is the mean over windows with k events, n_events_<k> their number."""
    if "fact_count" not in result:
        return {}
    count, score = result["fact_count"].numpy(), result["fact_score"].numpy()
    out = {}
    for low, high, name in EVENT_BUCKETS:
        mine = (count >= low) & (count <= high)
        out[f"fact_events_{name}"] = round(np.nanmean(score[mine]).item(), 4) if mine.any() else None
        out[f"n_events_{name}"] = int(mine.sum())
    return out


def eventful_windows(store, windows, device, n, generator):
    """n random windows the do-nothing baseline fails, so every scored window asks for change."""
    order = windows[torch.randperm(len(windows), generator=generator)]
    kept = []
    for start in range(0, len(order), 256):
        part = order[start : start + 256]
        zero = run(None, store, part, device, mode="zero")
        kept.append(part[~zero["reached"]])
        if sum(len(k) for k in kept) >= n:
            break
    return torch.cat(kept)[:n]


@torch.no_grad()
def direction_agreement(policy, store, windows, device, guidance=1.0):
    """Mass-weighted share of moving mass whose first sampled step points within 45 degrees of the
    recorded step, at window starts. Returns (true goal, shuffled goal)."""
    e, w0, w1 = windows[:, 0], windows[:, 1], windows[:, 2]
    world, goal = world_at(store, e, w0, device), world_at(store, e, w1, device)
    z = policy.goal_z(policy.goal_encoder(torch.cat([observe(world), observe(goal)], dim=1)))
    obs = policy_obs(policy, world, store, e, w0)
    true = decode_field(store["field"][e, w0], device)
    weight = world.mass * (true.abs().sum(1) > 0).float()
    scores = []
    for zz in (z, z.roll(1, dims=0)):
        field = policy.sample(obs, zz, guidance=guidance)[:, 0]
        cos = F.cosine_similarity(field, true, dim=1, eps=1e-6)
        scores.append(round((((cos > 0.7).float() * weight).sum() / weight.sum()).item(), 4))
    return scores
