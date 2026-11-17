"""Search for "move 37" runs of the 2D swarm policy: runs where the policy beats the scripted expert.

Each batch samples random scenes and one instruction per scene (some preceded by 1-2 other
instructions, as steer.py's "after" protocol), rolls the policy out (cap 140 steps), plays the
scripted expert (vla/experts.py) on the same scene and instruction, and judges both with the
suite's own check (experts.succeeded). A run is a candidate if the policy succeeded and the expert
did not, or the policy needed at most 0.7x the expert's steps.

    uv run python v1/scripts/move37_search.py --init_run RUN --checkpoint policy_latest.pt --hours 8
    uv run python v1/scripts/move37_search.py --policy dummy --device cpu --hours 0.05 --out runs/move37_test

Success of a run: the check passes for HOLD consecutive steps (so a robot swarm passing through a
region does not count); steps = the first step of that streak. The same rule and the same cap
apply to policy and expert.

Chained episodes: the first 1-2 instructions run for PREFIX steps each (the policy, no judging);
the last instruction is the judged one. The expert plays that last instruction from the same world
state the policy had when it was given (the scene rebuilt from the policy's state), so both see
the same scene and instruction.

Per candidate, runs/move37/cand_NNNNN/ holds episode.json (field_demo_video.py's input, the
policy's whole run including any earlier instructions), contact.png and meta.json; then one line
goes to candidates.jsonl.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import field_demo_record as FR  # noqa: E402  (Recorder, dummy_policy, sample_field, CHANNELS)

from beebot.vla import steer  # noqa: E402
from beebot.vla import tasks as T  # noqa: E402
from beebot.vla.controller import Controller, commands_to_tensors  # noqa: E402
from beebot.vla.data import CODE_VECTORS, encode_field  # noqa: E402
from beebot.vla.experts import Expert, mask_field, succeeded  # noqa: E402
from beebot.vla.goal_eval import snapshot  # noqa: E402
from beebot.vla.world import COLOUR_NAMES, COLOUR_RGB, DIRS, World, cheese_side, observe, step  # noqa: E402

INTENTS = ("logic.all", "logic.or", "logic.single", "logic.nearest_box", "place.box", "place.move", "place.region", "runner.catch")
CAP = 140  # steps per judged instruction
HOLD = 10  # consecutive steps the check must pass
PREFIX = 48  # steps of each earlier instruction in a chain (steer.validate's "after" protocol)
CHAIN_SHARE = 0.25
FASTER = 0.7
FRAMES = 6


# ---------------------------------------------------------------------------------------------
# Scenes and instructions


def sample_item(rng):
    """A random scene with varied settings, or None if placement failed."""
    low = int(rng.integers(1, 4))
    high = min(6, low + int(rng.integers(0, 4)))
    return T.sample_scene(
        rng,
        cheeses=(low, high),
        boxes=(0, int(rng.integers(1, 4))),
        wall_share=float(rng.integers(0, 2)),
        runners=int(rng.random() < 0.3),
    )


def pick_task(scene, rng):
    """(intent, Task) for the scene from INTENTS (keys uniformly, then a task), or None."""
    by_key = {}
    for key, task in steer.candidates(scene, rng, tries=24):
        if key in INTENTS:
            by_key.setdefault(key, []).append(task)
    if not by_key:
        return None
    key = str(rng.choice(sorted(by_key)))
    return key, by_key[key][int(rng.integers(len(by_key[key])))]


def merge(worlds):
    return steer.concat(worlds)


# ---------------------------------------------------------------------------------------------
# Actors: act(world, trackers, t, active) -> field (B, 2, H, W), probs (list or None), codes (B, H, W)


class Tracker:
    """One run of one instruction: its task, the eaten colours, and the success streak."""

    def __init__(self, task, scene, first):
        self.task, self.scene, self.first = task, scene, first
        self.eaten = []
        self.streak = 0
        self.met_at = None
        self.success = False
        self.done = False
        self.goal = None  # dummy policy only


class LMActor:
    def __init__(self, sample, tokenizer, device):
        from beebot.vla.lm_rl import tokens

        self.sample, self.tokenizer, self.device, self.tokens = sample, tokenizer, device, tokens

    @torch.no_grad()
    def __call__(self, world, trackers, t, active):
        obs = observe(world)
        cond = self.tokens(self.tokenizer, [tr.task.text for tr in trackers], self.device)
        _, field, probs = self.sample(obs, cond)
        return field, probs.float().cpu().numpy(), encode_field(field).cpu().numpy()


def dummy_goal(tracker, world, i):
    """A goal in field_demo_record.dummy_policy's format for the tracker's task (its attractor)."""
    spec = tracker.task.spec
    if "region" in spec:
        return {"kind": "region", "region": spec["region"]}
    if "box" in spec:
        return {"kind": "box", "box": list(spec["box"])}
    if "direction" in spec:
        if tracker.goal is None:
            rows, cols = np.nonzero(world.mass[i].numpy() >= 0.5)
            dy, dx = T.DIRECTIONS[spec["direction"]]
            y = int(np.clip(rows.mean() + 12 * dy, 2, T.SIZE - 3))
            x = int(np.clip(cols.mean() + 12 * dx, 2, T.SIZE - 3))
            open_squares = np.argwhere(~world.walls[i].numpy())
            y, x = open_squares[np.argmin(((open_squares - (y, x)) ** 2).sum(1))]
            tracker.goal = {"kind": "box", "box": [int(y), int(x), 1, 1]}
        return tracker.goal
    wanted = set(spec["eat"]) - set(tracker.eaten)
    if spec["any_of"] and not (set(spec["any_of"]) & set(tracker.eaten)):
        wanted = set(spec["any_of"])
    alive = world.cheese_alive[i].numpy()
    colours = world.cheese_color[i].numpy()
    slots = [k for k in range(len(alive)) if alive[k] and COLOUR_NAMES[colours[k]] in wanted]
    if not slots:  # nothing left to eat: point at a dead slot, so the dummy waits
        dead = [k for k in range(len(alive)) if colours[k] >= 0 and not alive[k]]
        return {"kind": "eat", "slot": dead[0] if dead else 0}
    rows, cols = np.nonzero(world.mass[i].numpy() >= 0.5)
    centre = np.array([rows.mean(), cols.mean()])
    slot = min(slots, key=lambda k: np.hypot(*(world.cheese_pos[i, k].numpy() - centre)))
    return {"kind": "eat", "slot": int(slot)}


class DummyActor:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, world, trackers, t, active):
        size = T.SIZE
        field = torch.zeros(len(trackers), 2, size, size)
        probs = [None] * len(trackers)
        codes = np.full((len(trackers), size, size), 8, np.int64)
        for i, tr in enumerate(trackers):
            if not active[i]:
                continue
            part = steer.part(world, i)
            p = FR.dummy_policy(part, dummy_goal(tr, world, i), self.rng)
            c, f = FR.sample_field(p, None)
            field[i] = f[0]
            probs[i] = p
            codes[i] = c.numpy()
        return field, probs, codes


class ExpertActor:
    def __init__(self, scenes, tasks):
        self.experts = [Expert(s, t) for s, t in zip(scenes, tasks)]
        self.controller = Controller()

    def __call__(self, world, trackers, t, active):
        orders = [e.commands(world, i, t) if active[i] else [] for i, e in enumerate(self.experts)]
        commands = [o if isinstance(o, list) else [] for o in orders]
        points, valid, straight, share = commands_to_tensors(commands, len(trackers), world.mass.device)
        field = self.controller(world, points, valid, straight, share)
        for i, o in enumerate(orders):
            if not isinstance(o, list):
                field[i] = mask_field(world, i, o)
        return field, [None] * len(trackers), encode_field(field).cpu().numpy()


# ---------------------------------------------------------------------------------------------
# Rolling a batch through one instruction each


def run_segment(world, runners, trackers, act, cap, judge):
    """Step the batch until every run is done (success held for HOLD steps, or cap steps; with
    judge False, exactly cap steps). Returns the world and per run the list of recorded states
    (the state at each step and the actor's output there), ending at the run's final state."""
    count = len(trackers)
    records = [[] for _ in range(count)]
    closed = [False] * count
    eaten_now = [[] for _ in range(count)]
    pushed_now = [[] for _ in range(count)]
    for t in range(cap + 1):
        active = [not c for c in closed]
        field, probs, codes = act(world, trackers, t, active)
        mass = world.mass.cpu().numpy()
        pos = world.cheese_pos.cpu().numpy()
        alive = world.cheese_alive.cpu().numpy()
        for i in range(count):
            if closed[i]:
                continue
            records[i].append(
                {
                    "mass": mass[i].copy(),
                    "cheese_pos": pos[i].copy(),
                    "cheese_alive": alive[i].copy(),
                    "probs": None if probs[i] is None else probs[i].astype(np.float16),
                    "codes": codes[i].astype(np.uint8),
                    "eaten": eaten_now[i],
                    "pushed": pushed_now[i],
                }
            )
            if trackers[i].done:
                closed[i] = True
        if all(closed) or t == cap:
            break
        world, info = step(world, field)
        if runners.any() and (t + 1) % T.RUNNER_PERIOD == 0:
            world = replace(world, cheese_pos=T.flee(world, runners))
        eaten = info.eaten.cpu()
        pushed = info.pushed.cpu()
        for i, tr in enumerate(trackers):
            eaten_now[i] = eaten[i].nonzero().flatten().tolist()
            pushed_now[i] = pushed[i].nonzero().flatten().tolist()
            if tr.done:
                continue
            tr.eaten += [COLOUR_NAMES[int(world.cheese_color[i, k])] for k in eaten_now[i]]
            if judge:
                result = T.score(tr.task, tr.scene, [tr.first, snapshot(world, i)], tr.eaten)
                if succeeded(tr.task, result):
                    if tr.streak == 0:
                        tr.met_at = t + 1
                    tr.streak += 1
                else:
                    tr.streak = 0
                if tr.streak >= HOLD:
                    tr.success, tr.done = True, True
            if t + 1 >= cap:
                tr.done = True
    return world, records


def split_worlds(world, runners, count):
    return [steer.part(world, i) for i in range(count)], [runners[i : i + 1] for i in range(count)]


def new_tracker(task, scene, world, i):
    return Tracker(task, scene, snapshot(world, i))


def rollout_batch(actor, rng_list, scenes, args, device):
    """Policy rollout of a batch: optional earlier instructions, then the judged one. Returns per
    item None (dropped) or a dict with the final task and intent, the run's records and the
    index in them where the judged instruction starts, its tracker, and the final scene."""
    count = len(scenes)
    world, runners = T.stack(scenes)
    worlds, run_masks = split_worlds(world.to(device), runners.to(device), count)
    records = [[] for _ in range(count)]
    commands = [[] for _ in range(count)]  # (text, first record index, intent)
    prefixes = [int(rng_list[i].integers(1, 3)) if rng_list[i].random() < CHAIN_SHARE else 0 for i in range(count)]
    for stage in range(2):
        idx = [i for i in range(count) if prefixes[i] > stage]
        if not idx:
            break
        trackers, keep, intents = [], [], []
        for i in idx:
            scene = steer.scene_of(worlds[i], run_masks[i])
            picked = pick_task(scene, rng_list[i])
            if picked is None:
                prefixes[i] = stage  # no earlier instruction fits; the chain is as long as it got
                continue
            keep.append(i)
            intents.append(picked[0])
            trackers.append(new_tracker(picked[1], scene, worlds[i], 0))
        if not keep:
            continue
        merged, merged_runners = merge([worlds[i] for i in keep]), torch.cat([run_masks[i] for i in keep])
        merged, segment = run_segment(merged, merged_runners, trackers, actor, PREFIX, False)
        for j, i in enumerate(keep):
            commands[i].append((trackers[j].task.text, len(records[i]), intents[j]))
            records[i] += segment[j]
            worlds[i] = steer.part(merged, j)
    out = [None] * count
    finals, trackers = [], []
    for i in range(count):
        scene = steer.scene_of(worlds[i], run_masks[i])
        picked = pick_task(scene, rng_list[i])
        if picked is None:
            continue
        finals.append(i)
        trackers.append(new_tracker(picked[1], scene, worlds[i], 0))
        out[i] = {"intent": picked[0], "task": picked[1], "scene": scene, "tracker": trackers[-1], "initial": scenes[i]}
    if not finals:
        return out
    merged, merged_runners = merge([worlds[i] for i in finals]), torch.cat([run_masks[i] for i in finals])
    merged, segment = run_segment(merged, merged_runners, trackers, actor, CAP, True)
    for j, i in enumerate(finals):
        commands[i].append((trackers[j].task.text, len(records[i]), out[i]["intent"]))
        out[i].update({"start": len(records[i]), "records": records[i] + segment[j], "commands": commands[i], "constants": worlds[i]})
    return out


def expert_batch(items, device):
    """The expert on the judged instruction of each item, from the policy's state when it was given."""
    scenes = [it["scene"] for it in items]
    tasks = [it["task"] for it in items]
    world, runners = T.stack(scenes)
    trackers = [new_tracker(t, s, world, i) for i, (t, s) in enumerate(zip(tasks, scenes))]
    _, segment = run_segment(world, runners, trackers, ExpertActor(scenes, tasks), CAP, True)
    return trackers, segment, world


def confirm_batch(actor, items, repeats):
    """Re-run the policy repeats times on each item's judged instruction from the same start state
    (fresh samples, one batch). Returns per item the list of (success, steps)."""
    scenes = [it["scene"] for it in items for _ in range(repeats)]
    tasks = [it["task"] for it in items for _ in range(repeats)]
    world, runners = T.stack(scenes)
    world = world.to(actor.device) if hasattr(actor, "device") else world
    runners = runners.to(world.mass.device)
    trackers = [new_tracker(t, s, world, i) for i, (t, s) in enumerate(zip(tasks, scenes))]
    run_segment(world, runners, trackers, actor, CAP, True)
    return [[(tr.success, tr.met_at) for tr in trackers[k * repeats : (k + 1) * repeats]] for k in range(len(items))]


# ---------------------------------------------------------------------------------------------
# Writing a candidate


def draw_state(ax, state, colours, walls, boxes, spec, start_centre):
    size = T.SIZE
    ax.imshow(np.where(walls, 0.25, 1.0), cmap="gray", vmin=0, vmax=1, extent=(0, size, size, 0), interpolation="nearest")
    for (y, x, h, w), colour in boxes:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=COLOUR_RGB[colour].tolist(), linewidth=2))
    if "box" in spec:
        y, x, h, w = spec["box"]
        ax.add_patch(Rectangle((x - 0.3, y - 0.3), w + 0.6, h + 0.6, fill=False, edgecolor="black", linestyle="--", linewidth=1))
    if "region" in spec:
        y, x = T.REGIONS[spec["region"]]
        ax.add_patch(Circle((x, y), 2.5, fill=False, edgecolor="black", linestyle="--", linewidth=1))
    if "direction" in spec:
        dy, dx = T.DIRECTIONS[spec["direction"]]
        ax.annotate("", xy=(start_centre[1] + 8 * dx, start_centre[0] + 8 * dy), xytext=(start_centre[1], start_centre[0]), arrowprops={"arrowstyle": "->", "linestyle": "--", "color": "black"})
    side = cheese_side(size)
    for (y, x), alive, colour in zip(state["cheese_pos"], state["cheese_alive"], colours):
        if alive and colour >= 0:
            ax.add_patch(Rectangle((x, y), side, side, facecolor=COLOUR_RGB[colour].tolist(), edgecolor="black", linewidth=0.8))
    rows, cols = np.nonzero(state["mass"] >= 0.5)
    ax.scatter(cols + 0.5, rows + 0.5, s=9, c="tab:blue", marker="s", linewidths=0)
    ax.set_xlim(0, size)
    ax.set_ylim(size, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def contact_sheet(path, title, spec, runs, constants):
    """runs: (label, states) pairs, one row each, FRAMES frames at evenly spaced steps."""
    walls = constants.walls[0].cpu().numpy()
    boxes = [(r, c) for r, c in zip(constants.box_rect[0].tolist(), constants.box_color[0].tolist()) if c >= 0]
    fig, axes = plt.subplots(len(runs), FRAMES, figsize=(2.3 * FRAMES, 2.5 * len(runs) + 0.9))
    axes = np.atleast_2d(axes)
    for row, (label, states, colours) in enumerate(runs):
        picks = np.linspace(0, len(states) - 1, FRAMES).round().astype(int)
        rows, cols = np.nonzero(states[0]["mass"] >= 0.5)
        start_centre = (rows.mean(), cols.mean())
        for col, k in enumerate(picks):
            ax = axes[row, col]
            draw_state(ax, states[k], colours, walls, boxes, spec, start_centre)
            ax.set_title(f"{label} step {k}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def final_states(records, start):
    return [{k: r[k] for k in ("mass", "cheese_pos", "cheese_alive")} for r in records[start:]]


def write_candidate(out_dir, number, item, expert_tracker, expert_segment, expert_world, reason, policy_steps, expert_steps, tokenizer, args, seed_info, confirm):
    name = f"cand_{number:05d}"
    directory = out_dir / name
    directory.mkdir(parents=True)
    texts = [c[0] for c in item["commands"]]
    intents = [c[2] for c in item["commands"]]
    constants = item["constants"]
    recorder = FR.Recorder(item["initial"], texts, tokenizer)
    command_at = [0] * len(item["records"])
    for j, (_, begin, _) in enumerate(item["commands"]):
        for k in range(begin, len(item["records"])):
            command_at[k] = j
    for k, rec in enumerate(item["records"]):
        world = World(
            mass=torch.from_numpy(rec["mass"])[None],
            walls=constants.walls.cpu(),
            cheese_pos=torch.from_numpy(rec["cheese_pos"])[None],
            cheese_alive=torch.from_numpy(rec["cheese_alive"])[None],
            cheese_color=constants.cheese_color.cpu(),
            box_rect=constants.box_rect.cpu(),
            box_color=constants.box_color.cpu(),
        )
        recorder.add(
            rec["mass"],
            rec["cheese_pos"],
            rec["cheese_alive"],
            constants.cheese_color[0].cpu().numpy(),
            observe(world)[0].numpy(),
            rec["probs"].astype(np.float32),
            rec["codes"].astype(np.int64),
            rec["eaten"],
            rec["pushed"],
            command_at[k],
        )
    tracker = item["tracker"]
    episode = recorder.episode(
        policy=args.policy,
        success=True,
        length=len(item["records"]) - 1,
        pushes=sum(len(r["pushed"]) for r in item["records"]),
        seed=seed_info,
        intents=intents,
        reason=reason,
    )
    data = {
        "note": f"move37 candidate: {reason}; policy {args.init_run}/{args.checkpoint} ({args.policy})",
        "policy": args.policy,
        "guidance": 2.0 if args.policy == "lm" else None,
        "size": T.SIZE,
        "cheese_side": 4,
        "codes": "0-7: world.DIRS (dy, dx) = " + json.dumps(DIRS) + "; 8: stay",
        "code_vectors": CODE_VECTORS.tolist(),
        "channels": FR.CHANNELS,
        "episodes": [],
        "instructions": [{"instruction": episode["instruction"], "intents": intents, "successes": 1, "rollouts": 1, "runs": [episode]}],
    }
    (directory / "episode.json").write_text(json.dumps(data, separators=(",", ":")))
    title = f"{episode['instruction']}   [{reason}: policy {policy_steps} steps, expert {expert_steps if expert_steps is not None else 'failed'}]"
    expert_label = "expert" if expert_tracker.success else "expert (failed)"
    contact_sheet(
        directory / "contact.png",
        title,
        item["task"].spec,
        [
            ("policy", final_states(item["records"], item["start"]), constants.cheese_color[0].tolist()),
            (expert_label, [{k: r[k] for k in ("mass", "cheese_pos", "cheese_alive")} for r in expert_segment], expert_world.cheese_color[0].tolist()),
        ],
        constants,
    )
    meta = {
        "id": name,
        "instructions": texts,
        "intents": intents,
        "instruction": item["task"].text,
        "intent": item["intent"],
        "scene_seed": seed_info,
        "policy_steps": policy_steps,
        "expert_steps": expert_steps,
        "reason": reason,
        "run": f"{args.init_run}/{args.checkpoint}",
        "confirm": confirm,
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=1))
    return name, directory


# ---------------------------------------------------------------------------------------------


def next_number(out_dir):
    numbers = [int(p.name[5:]) for p in out_dir.glob("cand_*") if p.name[5:].isdigit()]
    return max(numbers, default=0) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_run", default="")
    parser.add_argument("--checkpoint", default="policy_latest.pt")
    parser.add_argument("--policy", choices=("lm", "dummy"), default="lm")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", default="runs/move37")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confirm", type=int, default=2, help="re-runs of a would-be candidate; kept if the policy succeeds in at least this many of confirm+1 runs")
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    steer.SUGGEST_FAMILIES = ("logic", "place", "runner")  # steer.candidates reads this; only these makers give our intents
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.policy == "lm":
        from beebot.vla.lm_policy import PATCH
        from beebot.vla.lm_rl import RLConfig, lm_probs_sampler, load_policy

        policy, tokenizer = load_policy(RLConfig(init_run=args.init_run, checkpoint=args.checkpoint), device)
        policy.eval()
        actor = LMActor(lm_probs_sampler(policy, 2.0, T.SIZE // PATCH), tokenizer, device)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        actor = DummyActor(args.seed)
    totals = {"rollouts": 0, "policy_success": 0, "expert_success": 0, "move37_candidates": 0, "unconfirmed": 0}
    by_intent = {k: {"rollouts": 0, "policy_success": 0, "expert_success": 0, "candidates": 0} for k in INTENTS}
    number = next_number(out_dir)
    deadline = time.time() + args.hours * 3600
    batch_number = 0
    while time.time() < deadline:
        torch.manual_seed(args.seed * 100003 + batch_number)
        rngs, scenes, seeds = [], [], []
        index = batch_number * 1000  # scene seed [seed, index]: np.random.default_rng([seed, index]) draws the scene
        while len(scenes) < args.batch:
            rng = np.random.default_rng([args.seed, index])
            scene = sample_item(rng)
            if scene is not None:
                rngs.append(rng)
                scenes.append(scene)
                seeds.append([args.seed, index])
            index += 1
        items = rollout_batch(actor, rngs, scenes, args, device)
        kept = [(k, it) for k, it in enumerate(items) if it is not None]
        if kept:
            expert_trackers, expert_segments, expert_world = expert_batch([it for _, it in kept], device)
        qualified = []
        for j, (k, item) in enumerate(kept):
            tracker = item["tracker"]
            expert = expert_trackers[j]
            intent = item["intent"]
            totals["rollouts"] += 1
            by_intent[intent]["rollouts"] += 1
            totals["policy_success"] += tracker.success
            by_intent[intent]["policy_success"] += tracker.success
            totals["expert_success"] += expert.success
            by_intent[intent]["expert_success"] += expert.success
            if not tracker.success:
                continue
            reason = None
            if not expert.success:
                reason = "expert_failed"
            elif tracker.met_at <= FASTER * expert.met_at:
                reason = "faster"
            if reason is not None:
                qualified.append((j, k, reason))
        confirmations = confirm_batch(actor, [kept[j][1] for j, _, _ in qualified], args.confirm) if qualified and args.confirm > 0 else [[] for _ in qualified]
        for (j, k, reason), extra in zip(qualified, confirmations):
            item = kept[j][1]
            tracker = item["tracker"]
            expert = expert_trackers[j]
            intent = item["intent"]
            runs = [(True, tracker.met_at)] + extra
            wins = [steps for ok, steps in runs if ok]
            keep = len(wins) >= args.confirm if args.confirm else True
            if keep and reason == "faster":
                keep = float(np.median(wins)) <= FASTER * expert.met_at
            if not keep:
                totals["unconfirmed"] += 1
                continue
            confirm = {"successes": len(wins), "runs": len(runs), "steps": [steps if ok else None for ok, steps in runs]}
            name, directory = write_candidate(
                out_dir, number, item, expert, expert_segments[j], steer.part(expert_world, j), reason,
                tracker.met_at, expert.met_at if expert.success else None, tokenizer, args, seeds[k], confirm,
            )
            number += 1
            totals["move37_candidates"] += 1
            by_intent[intent]["candidates"] += 1
            line = {
                "id": name,
                "dir": str(directory),
                "instruction": " -> ".join(c[0] for c in item["commands"]),
                "intent": intent,
                "reason": reason,
                "policy_steps": tracker.met_at,
                "expert_steps": expert.met_at if expert.success else None,
                "policy_successes": f"{len(wins)}/{len(runs)}",
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(out_dir / "candidates.jsonl", "a") as f:
                f.write(json.dumps(line) + "\n")
        print(json.dumps({**totals, "batch": batch_number, "by_intent": by_intent}), flush=True)
        batch_number += 1


if __name__ == "__main__":
    main()
