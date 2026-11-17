"""Language-level plans: a planner turns an instruction into a sequence of simple commands the
policy already follows ("eat the red cheese", "push the red cheese left", ...), one at a time. A
command runs until its event is seen (the cheese eaten, the cheese pushed PUSH_SQUARES along) or
its step cap, then the planner is asked for the next one with the scene as it is now. The task is
still scored on its own instruction.

The planner here is an oracle written from the task spec: the ceiling of planning with this
executor. If the executor following perfect plans does not beat the executor given the raw
instruction, no learned planner will.

    uv run python -m beebot.vla.plans --init_run runs/lm-expert-sft-herd --tasks 48 --planners oracle,Qwen/Qwen3-4B

A pretrained chat LM as planner (LMPlanner) reads the instruction and the scene as text and
writes the next command, untrained: whether planning in language works before any training of it.
"""

import argparse
import json
import re
from dataclasses import dataclass

import numpy as np
import torch

from beebot.vla import tasks as T
from beebot.vla.facts import cheese_in_box
from beebot.vla.lm_policy import PATCH
from beebot.vla.lm_rl import RLConfig, lm_sampler, load_policy, tokens
from beebot.vla.world import COLOUR_NAMES, cheese_side

PUSH_SQUARES = 4  # a push command is done once its cheese moved this far along the direction
CAPS = {"eat": 40, "push": 24, "idle": 16}  # step caps per command kind
IDLE = "don't eat anything"
SPLITS = ("test_logic", "logic", "test_runner", "runner_logic", "test_herd", "herd_parallel")


@dataclass
class Command:
    text: str
    kind: str  # eat, push or idle
    colour: str = ""
    direction: str = ""


@dataclass
class Segment:
    command: Command
    start: int
    start_pos: np.ndarray  # the command's cheese position when it began (push)


def view(world, i, scene):
    """What a planner may know of episode i now: per colour, alive and centre; the swarm centre."""
    side = cheese_side(T.SIZE)
    cheeses = {}
    for k, c in enumerate(scene["cheese_colours"]):
        cheeses[COLOUR_NAMES[c]] = {"alive": bool(world.cheese_alive[i, k]), "pos": world.cheese_pos[i, k].cpu().numpy().astype(float), "centre": world.cheese_pos[i, k].cpu().numpy() + side / 2, "runner": k in scene["runners"]}
    mass = world.mass[i].cpu().numpy()
    rows, cols = np.nonzero(mass >= 0.5)
    swarm = np.array([rows.mean(), cols.mean()]) if len(rows) else np.array([T.SIZE / 2, T.SIZE / 2])
    return {"cheeses": cheeses, "swarm": swarm}


def eat(colour, cheese):
    return Command(("catch the {} cheese" if cheese["runner"] else "eat the {} cheese").format(colour), "eat", colour)


def push(colour, direction):
    return Command(f"push the {colour} cheese {direction}", "push", colour, direction)


def oracle(task, scene, state):
    """The next command for task given the current state, or None to run the instruction itself
    (families with nothing to decompose)."""
    spec, cheeses = task.spec, state["cheeses"]
    if task.family in ("logic", "runner"):
        alive = {c for c, v in cheeses.items() if v["alive"]}
        eaten = set(cheeses) - alive
        if spec["order"]:
            todo = [c for c in spec["order"] if c in alive]
            return eat(todo[0], cheeses[todo[0]]) if todo else Command(IDLE, "idle")
        if spec["any_of"]:
            todo = [] if eaten & spec["any_of"] else sorted(spec["any_of"] & alive)
        else:
            todo = sorted(spec["eat"] & alive)
        if not todo:
            return Command(IDLE, "idle")
        nearest = min(todo, key=lambda c: np.linalg.norm(cheeses[c]["centre"] - state["swarm"]))
        return eat(nearest, cheeses[nearest])
    if task.family == "herd":
        if "nudge" in spec:
            c, d = spec["nudge"]
            return push(c, d)
        targets, rect = spec["into"]
        side = cheese_side(T.SIZE)
        y, x, h, w = rect
        centre = np.array([y + h / 2, x + w / 2])
        for c in targets:
            v = cheeses[c]
            if not v["alive"] or cheese_in_box(v["pos"].astype(int), rect, side):
                continue
            dy, dx = centre - v["centre"]
            if abs(dy) >= abs(dx):
                return push(c, "down" if dy > 0 else "up")
            return push(c, "right" if dx > 0 else "left")
        return Command(IDLE, "idle")
    return None


def done(segment, state, t):
    command = segment.command
    if t - segment.start >= CAPS[command.kind]:
        return True
    if command.kind == "eat":
        return not state["cheeses"][command.colour]["alive"]
    if command.kind == "push":
        cheese = state["cheeses"][command.colour]
        dy, dx = T.DIRECTIONS[command.direction]
        moved = (cheese["pos"] - segment.start_pos) @ np.array([dy, dx])
        return not cheese["alive"] or moved >= PUSH_SQUARES
    return False


def planned(scenes, task_list, planner):
    """tasks.run's replan hook: per episode the current segment, replaced when it is done.
    planner(items) -> one Command (or None: run the instruction itself) per item, an item being
    (task, scene, state, the commands given so far): all episodes needing one at a step at once."""
    segments = [None] * len(task_list)
    texts = [t.text for t in task_list]
    trace = [[] for _ in task_list]

    def replan(t, world, eaten):
        states = [view(world, i, scene) for i, scene in enumerate(scenes)]
        due = [i for i in range(len(task_list)) if segments[i] is None or done(segments[i], states[i], t)]
        if not due:
            return None
        commands = planner([(task_list[i], scenes[i], states[i], [c for _, c in trace[i]]) for i in due])
        changed = t == 0
        for i, command in zip(due, commands):
            if command is None:
                continue
            pos = states[i]["cheeses"][command.colour]["pos"] if command.colour else np.zeros(2)
            segments[i] = Segment(command, t, pos)
            trace[i].append((t, command.text))
            changed |= command.text != texts[i]
            texts[i] = command.text
        return list(texts) if changed else None

    return replan, trace


def oracle_planner(items):
    return [oracle(task, scene, state) for task, scene, state, _ in items]


COMMANDS = """Commands (use colours that are in the scene):
- eat the <colour> cheese
- catch the <colour> cheese   (for a cheese that runs away)
- push the <colour> cheese <up|down|left|right>   (moves it about 4 squares without eating it)
- don't eat anything   (wait; use it when the instruction is already done)"""


def describe(scene, state, history):
    """The scene as text for a planner: cheeses (alive, position, running), boxes, the swarm."""
    lines = ["Grid: 32 x 32 squares, row 0 is the top, column 0 is the left."]
    for colour, v in state["cheeses"].items():
        if v["alive"]:
            y, x = v["centre"]
            lines.append(f"- {colour} cheese at row {y:.0f}, column {x:.0f}" + (" (runs away from the swarm)" if v["runner"] else ""))
        else:
            lines.append(f"- {colour} cheese: already eaten")
    for (y, x, h, w), c in zip(scene["boxes"], scene["box_colours"]):
        lines.append(f"- {COLOUR_NAMES[c]} box: rows {y} to {y + h - 1}, columns {x} to {x + w - 1}")
    sy, sx = state["swarm"]
    lines.append(f"- the swarm's centre: row {sy:.0f}, column {sx:.0f}")
    lines.append("Commands given so far: " + ("; ".join(history[-6:]) if history else "none"))
    return "\n".join(lines)


PATTERNS = [
    (re.compile(r"(eat|catch) the (\w+) cheese"), lambda m: (m.group(1), m.group(2), "")),
    (re.compile(r"push the (\w+) cheese (up|down|left|right)"), lambda m: ("push", m.group(1), m.group(2))),
]


def parse(reply, state):
    """The first command in reply, or idle when there is none or its cheese is not alive here."""
    reply = reply.lower()
    for pattern, read in PATTERNS:
        m = pattern.search(reply)
        if not m:
            continue
        verb, colour, direction = read(m)
        cheese = state["cheeses"].get(colour)
        if cheese is None or not cheese["alive"]:
            break
        if verb == "push":
            return push(colour, direction)
        return eat(colour, cheese)
    return Command(IDLE, "idle")


class LMPlanner:
    """A pretrained chat LM (no training) choosing the next command from the instruction and the
    scene described as text."""

    def __init__(self, model, device, max_new_tokens=16):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.tokenizer.padding_side = "left"
        self.lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float16 if device.type != "cpu" else torch.float32).to(device).eval()
        self.device, self.max_new_tokens = device, max_new_tokens
        self.replies = []

    def prompt(self, task, scene, state, history):
        content = f"You steer a swarm of robots on a grid by giving it one simple command at a time.\n{COMMANDS}\n\nScene now:\n{describe(scene, state, history)}\n\nThe instruction: \"{task.text}\"\nWhat is the next command? Answer with the command only."
        return self.tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False)

    @torch.no_grad()
    def __call__(self, items):
        enc = self.tokenizer([self.prompt(*item) for item in items], return_tensors="pt", padding=True).to(self.device)
        out = self.lm.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)
        replies = self.tokenizer.batch_decode(out[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
        self.replies += replies
        return [parse(reply, state) for reply, (_, _, state, _) in zip(replies, items)]


def compare(policy, tokenizer, split, n, steps, guidance, device, seed, planners, batch=24):
    """Per arm (direct: the instruction itself; then each planner) the summary on the same tasks,
    scenes and sampling seeds."""
    rng = np.random.default_rng(seed)
    pairs = [T.sample_task(rng, "test" if split.startswith("test_") else split, family=split[5:] if split.startswith("test_") else None) for _ in range(n)]
    sample = lm_sampler(policy, guidance, T.SIZE // PATCH)
    condition = lambda texts: tokens(tokenizer, texts, device)  # noqa: E731
    out = {"direct": [], **{name: [] for name in planners}}
    examples = {}
    for k in range(0, n, batch):
        chunk = pairs[k : k + batch]
        scenes, task_list = [s for s, _ in chunk], [t for _, t in chunk]
        torch.manual_seed(seed + k)
        out["direct"] += T.run(sample, condition, scenes, task_list, steps, device)[0]
        for name, planner in planners.items():
            replan, trace = planned(scenes, task_list, planner)
            torch.manual_seed(seed + k)
            out[name] += T.run(sample, condition, scenes, task_list, steps, device, replan=replan)[0]
            examples.setdefault(name, {"instruction": task_list[0].text, "plan": trace[0][:6]})
    return {arm: summary(r) for arm, r in out.items()}, examples


def summary(results):
    """Mean score, unasked-eating rate, and success: the full score with nothing unasked eaten."""
    return {"score": round(float(np.mean([s for s, _ in results])), 3), "unasked": round(float(np.mean([u for _, u in results])), 3), "success": round(float(np.mean([s >= 0.99 and not u for s, u in results])), 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_run", default="runs/lm-expert-sft-herd")
    parser.add_argument("--checkpoint", default="policy_latest.pt")
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--planners", default="oracle", help="comma-separated: oracle and/or Hugging Face chat models (e.g. Qwen/Qwen3-4B)")
    parser.add_argument("--tasks", type=int, default=48)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    device = torch.device(args.device)
    policy, tokenizer = load_policy(RLConfig(init_run=args.init_run, checkpoint=args.checkpoint), device)
    policy.eval()
    planners = {name: oracle_planner if name == "oracle" else LMPlanner(name, device) for name in args.planners.split(",")}
    for split in args.splits.split(","):
        card, examples = compare(policy, tokenizer, split, args.tasks, args.steps, args.guidance, device, args.seed, planners)
        print(json.dumps({"split": split, **card, "examples": examples}), flush=True)
    for name, planner in planners.items():
        if isinstance(planner, LMPlanner):
            print(name, "sample replies:", json.dumps(planner.replies[:12]), flush=True)


if __name__ == "__main__":
    main()
