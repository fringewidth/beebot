"""Closed-loop episodes of the think-then-act voxel policy (voxel/think.py), and of the scripted
expert on the same scenes.

Each step of a policy episode: render the scene's views, build the context (the instruction in
force, the user's place round the arena, memory = the latest first thought and the latest switch
line after it), call ThinkVoxelPolicy.think_then_act for every live episode of the batch at once,
turn each acting row's slot logits into action codes (argmax, or sampled with the episode's own
generator), and step the world under the run's pivot rule and move failure rates. Thoughts are
recorded when non-empty. A switch task's instruction changes at its step (Task.spec["steps"]);
the memory still holds the old instruction's thought on that step, as in training. A reply in
words (refuse or clarify: the row reached END without the act marker) ends the episode: no robot
moves again and the episode is scored on the scene as it is. A reply that runs out of tokens
before the act marker or END moves nothing that step (counted as unfinished).

Oracle thoughts (Episode.oracle): on the first step of each instruction the context is followed by
the canonical first thought (canonical_thought) in the reply's own form, "<think>\\n{thought}\\n
</think>\\n\\n", and the policy generates from there: the act marker (then its slots) or an answer.
Later steps generate freely, with the forced thought in memory.

Determinism: every episode carries a seed; its world noise (step's move order), move failures,
action sampling and view randomization come from generators of its own seeded from it, so an
episode plays the same whatever else is in its batch (up to the LM's batch numerics) and on every
run. The expert arm is the exception: the suite's experts keep one generator for their batch, so
expert episodes are deterministic for a given seed and batch.

Conditions (the training stores, data/vla/voxel_full/s*/small.pt config): pivot rule switch_cube,
gravity on, cube moves for the expert, failure 0.10 per move and 0.23 for up-and-over moves
(FAILURE), 72 steps (STEPS) and 144 for build tasks (BUILD_STEPS, data/vla/voxel_build), every
world with the object fields (world.with_objects, MAX_OBJECTS slots).

Stand-ins, one function each, until the real ones land:
  build_context      -> think_data.context(instruction, memory, standing=...) (voxel/think_data.py)
  canonical_thought  -> think_data.template_thought(instruction, record, summary)
  stand_in_views     -> views.scene_views(obs, rng, ViewSpec()) (voxel/views.py; pass views=...)
The user's place is always the one that makes the user's frame the world's (quarter_turns 0), so
instructions and thoughts need no turning; USER_POSITIONS and quarter_turns follow think_data.
"""

from dataclasses import dataclass, field, replace

import numpy as np
import torch

from beebot.vla.voxel import suite as S
from beebot.vla.voxel.lm import unpatchify3
from beebot.vla.voxel.world import CUBE_MOVES, MAX_OBJECTS, STAY, concat, failure_rates, observe, step, with_objects

RULE = "switch_cube"
FAILURE = failure_rates(0.10, up_over=0.23)
STEPS = 72
BUILD_STEPS = 144
MAX_NEW_TOKENS = 160
START, END = "<|im_start|>", "<|im_end|>"
IMAGES = "<IMAGES>"
FIRST_SHAPE = "you asked:"  # a first thought starts so (Haiku and template thoughts); any other thought is a switch or done line


# ---------------------------------------------------------------------------------------------
# Stand-ins for think_data and views (see the module docstring)

USER_POSITIONS = {  # think_data.USER_POSITIONS: (degrees added to the camera's azimuth, what the user turn says)
    "behind": (0, "(i'm standing behind the camera)"),
    "left": (-90, "(i'm standing to the left of the camera)"),
    "right": (90, "(i'm standing to the right of the camera)"),
    "opposite": (180, "(i'm standing opposite the camera, facing it)"),
}
STAND_IN_AZIMUTH = 45.0  # render.render_scene looks from the front right corner
STAY_LINE = "stay where you are"


def quarter_turns(degrees):
    """think_data.quarter_turns: the quarter turn (0 front, 1 right, 2 back, 3 left) of a place
    round the arena at this azimuth."""
    return int(((float(degrees) + 45.0) % 360.0) // 90.0)


def standing_for(azimuth):
    """The user-place sentence whose frame is the world's (quarter turn 0) for this camera."""
    for offset, sentence in USER_POSITIONS.values():
        if quarter_turns(azimuth + offset) == 0:
            return sentence
    raise AssertionError("four places cover the four quarter turns")


def build_context(instruction, memory=(), standing=None):
    """Stand-in for think_data.context: memory as an earlier assistant turn, then the user turn with
    the images, where the user stands and the instruction; it ends where the reply starts."""
    out = f"{START}assistant\n" + "\n".join(memory) + f"{END}\n" if memory else ""
    where = f"{standing}\n" if standing else ""
    return out + f"{START}user\n{IMAGES}\n{where}{instruction.strip()}{END}\n{START}assistant\n"


def bare(text):
    return text.strip().rstrip(".!?").strip()


def canonical_thought(instruction, record, summary):
    """Stand-in for think_data.template_thought: the first thought of an instruction from its
    canonical record and the scene summary (suite.scene_summary)."""
    terrain = summary.get("terrain", {}).get("layout", "flat floor")
    things = [f"a {c['colour']} cheese at the {c['area']}" for c in summary.get("cheeses", [])[:4]]
    sight = ", ".join([f"the swarm on {'a ' if terrain.startswith('trench') else ''}{terrain}"] + things)
    if record.get("spec", {}).get("kind") == "refuse":
        meaning = "nothing here matches it, so there is nothing to do"
    else:
        meaning = f"that means: {bare(record['plain'])}"
    subgoals = record.get("subgoals") or []
    first = bare(subgoals[0]) if subgoals else STAY_LINE
    return f"you asked: {bare(instruction)}. i see {sight}. {meaning}. first: {first}.".lower()


def stand_in_views(obs, rng=None, scale=8):
    """([one isometric image], {"azimuth": 45}) of an observation (channels, D, H, W): robots grey,
    terrain sandy, objects in their per-voxel colour, as render.render_scene draws (rng unused)."""
    from PIL import Image, ImageDraw

    from beebot.vla.voxel.render import BACKGROUND, ROBOT, SHADES, WALL

    grid = obs.numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs)
    robots, walls, things = grid[0] > 0.5, grid[1] > 0.5, grid[2] > 0.5
    solid = robots | walls | things
    covered = np.zeros_like(solid)  # solid on all three visible sides: never seen
    covered[:-1, :-1, :-1] = solid[1:, :-1, :-1] & solid[:-1, 1:, :-1] & solid[:-1, :-1, 1:]
    d, h, w = solid.shape
    s = scale
    image = Image.new("RGB", ((w + h) * s + 2 * s, (w + h) * s // 2 + d * s + 2 * s), BACKGROUND)
    draw = ImageDraw.Draw(image)
    ox, oy = h * s + s, d * s + s
    draw.polygon([(ox + (x - y) * s, oy + (x + y) * s / 2) for y, x in ((0, 0), (0, w), (h, w), (h, 0))], outline=(200, 200, 195), fill=(236, 236, 230))
    cells = np.argwhere(solid & ~covered)
    order = np.lexsort((cells[:, 0], cells.sum(axis=1)))
    for z, y, x in cells[order]:
        rgb = ROBOT if robots[z, y, x] else WALL if walls[z, y, x] else tuple(int(v * 255) for v in grid[3:6, z, y, x])
        cx, cy = ox + (x - y) * s, oy + (x + y) * s / 2 - z * s
        faces = ([(cx, cy - s), (cx + s, cy - s / 2), (cx, cy), (cx - s, cy - s / 2)], [(cx - s, cy - s / 2), (cx, cy), (cx, cy + s), (cx - s, cy + s / 2)], [(cx, cy), (cx + s, cy - s / 2), (cx + s, cy + s / 2), (cx, cy + s)])
        for face, k in zip(faces, SHADES):
            draw.polygon(face, fill=tuple(int(c * k) for c in rgb))
    return [image], {"azimuth": STAND_IN_AZIMUTH}


def views_and_pose(result):
    """(images, camera azimuth) from what a views function returns: images, or (images, pose)."""
    images, pose = result if isinstance(result, tuple) else (result, None)
    return list(images), float((pose or {}).get("azimuth", 0.0))


# ---------------------------------------------------------------------------------------------
# Episodes


@dataclass
class Episode:
    """One episode to play. world: the scene (a batch of one); task: what is scored (a switch task
    carries its parts); texts[j]: the instruction given from step starts[j] on; records[j]: the
    canonical record (suite.canonical_record) of the instruction in force then, for oracle thoughts
    and thought checks; oracle: feed canonical first thoughts; key: what identifies it in reports."""

    world: object
    meta: dict
    task: S.Task
    texts: list
    starts: list
    records: list
    steps: int
    seed: int
    oracle: bool = False
    key: dict = field(default_factory=dict)


@dataclass
class Played:
    """What happened in an episode. thoughts: [{"t", "part", "text", "forced"}] of every non-empty
    thought; spoke: {"t", "text"} of the reply in words that ended it, or None; unfinished: steps
    whose reply ran out of tokens; moves: per step, the share of robots whose chosen action was not
    STAY; actions: (steps, D, H, W) int8 chosen codes when recorded."""

    score: float
    unasked: bool
    passed: bool
    thoughts: list = field(default_factory=list)
    spoke: dict | None = None
    unfinished: int = 0
    moves: list = field(default_factory=list)
    actions: torch.Tensor | None = None


def part_at(starts, t):
    return max(j for j, start in enumerate(starts) if start <= t)


def episode_steps(task):
    return BUILD_STEPS if task.family == "build" else STEPS


def passed(task, value, unasked):
    """Whether an episode passes: build tasks by build.passed, every other by the suite's pass mark
    (suite.SUCCESS, else PASS_MARK) without eating anything forbidden."""
    from beebot.vla.voxel import build as B

    if task.family == B.FAMILY:
        ok = B.passed(task, value, unasked)
    else:
        ok = not unasked and value >= S.SUCCESS.get(task.family, S.PASS_MARK)
    return bool(ok)


def prepare_world(world):
    """A scene as the training stores have it: object fields filled in, gravity on (a rule)."""
    world = with_objects(world, max(MAX_OBJECTS, world.cheese_alive.shape[1]))
    return replace(world, gravity=torch.ones_like(world.gravity))


def play(episodes, actor, rule=RULE, failure=FAILURE, record=False):
    """Play a batch of episodes with actor(world, t) -> (B, D, H, W) codes; the world and its
    scoring as suite.run (switch tasks scored on their last instruction from the step it came,
    runners flee every RUNNER_PERIOD steps), but with each episode's own generators and length:
    an episode past its steps stays still. Returns a Played per episode (without the actor's
    thoughts, which the actor adds)."""
    world = concat([prepare_world(e.world) for e in episodes])
    b = len(episodes)
    steps = max(e.steps for e in episodes)
    noise = [torch.Generator().manual_seed(int(e.seed)) for e in episodes]
    fails = [torch.Generator().manual_seed(int(e.seed) + 7_919) for e in episodes]
    runners = torch.zeros(world.cheese_alive.shape, dtype=torch.bool)
    for i, e in enumerate(episodes):
        runners[i, list(e.task.runners)] = True
    start = world
    eaten = [[] for _ in episodes]
    since = [(S.snapshot(world, i), 0) for i in range(b)]
    early = [False] * b
    moves = [[] for _ in episodes]
    actions = []
    for t in range(steps):
        live = torch.tensor([t < e.steps for e in episodes])
        for i, e in enumerate(episodes):
            task = e.task
            if task.parts and t in task.spec["steps"][1:] and t < e.steps:
                part = task.parts[S.instruction_index(task, t) - 1]
                early[i] = early[i] or any(c in part.spec["forbid"] for c in eaten[i][since[i][1] :])
                since[i] = (S.snapshot(world, i), len(eaten[i]))
        codes = actor(world, t).to(torch.long).clone()
        codes[~live] = STAY
        robots = world.mass > 0.5
        for i in range(b):
            if t < episodes[i].steps:
                n = int(robots[i].sum())
                moves[i].append(round(float(((codes[i] != STAY) & robots[i]).sum()) / max(n, 1), 4))
        if record:
            actions.append(codes.to(torch.int8))
        effective = codes
        if failure is not None:
            dropped = torch.stack([torch.rand(codes.shape[1:], generator=g) for g in fails]) < failure[codes]
            effective = torch.where(dropped, torch.full_like(codes, STAY), codes)
        before = world.cheese_alive.clone()
        world, _ = step(world, effective, torch.stack([torch.rand(world.mass.shape[1:], generator=g) for g in noise]), rule)
        fleeing = runners & live.view(-1, 1)
        if fleeing.any() and (t + 1) % S.RUNNER_PERIOD == 0:
            world = replace(world, cheese_pos=S.flee(world, fleeing))
        for i, k in (before & ~world.cheese_alive).nonzero().tolist():
            eaten[i].append(S.COLOUR_NAMES[int(world.cheese_color[i, k])])
    out = []
    for i, e in enumerate(episodes):
        task = e.task
        static = {"walls": start.walls[i].numpy(), "cheese_color": start.cheese_color[i].tolist(), "gravity": bool(start.gravity[i]), "meta": e.meta}
        if task.parts:
            value, unasked = S.score(task.parts[-1], static, since[i][0], S.snapshot(world, i), eaten[i][since[i][1] :])
            unasked = unasked or early[i]
        else:
            value, unasked = S.score(task, static, S.snapshot(start, i), S.snapshot(world, i), eaten[i])
        value = float(value)
        kept = torch.stack([a[i] for a in actions])[: e.steps] if record else None
        out.append(Played(value, bool(unasked), bool(passed(task, value, unasked)), moves=moves[i], actions=kept))
    return out


class PolicyActor:
    """The policy as an actor for play(): one think_then_act call per step over the batch's live
    episodes. sample: draw each robot's action from its softmax (with the episode's generator)
    instead of the argmax. views(obs, rng) -> images or (images, pose)."""

    def __init__(self, policy, episodes, views=stand_in_views, sample=False, max_new_tokens=MAX_NEW_TOKENS):
        self.policy, self.episodes, self.views = policy, episodes, views
        self.sample, self.max_new_tokens = sample, max_new_tokens
        self.vision = getattr(policy, "visual", None) is not None
        n = len(episodes)
        self.memory = [[] for _ in range(n)]
        self.thoughts = [[] for _ in range(n)]
        self.spoke = [None] * n
        self.unfinished = [0] * n
        self.view_rng = [np.random.default_rng([int(e.seed), 3]) for e in episodes]
        self.draws = [torch.Generator().manual_seed(int(e.seed) + 104_729) for e in episodes]

    def note(self, i, t, part, text, forced):
        """Record a thought and carry it in memory: a first thought replaces the memory, any other
        line becomes its second line."""
        self.thoughts[i].append({"t": t, "part": part, "text": text, "forced": forced})
        if text.startswith(FIRST_SHAPE) or not self.memory[i]:
            self.memory[i] = [text]
        else:
            self.memory[i] = [self.memory[i][0], text]

    def __call__(self, world, t):
        obs = observe(world)
        codes = torch.full(world.mass.shape, STAY, dtype=torch.long)
        rows = [i for i, e in enumerate(self.episodes) if self.spoke[i] is None and t < e.steps]
        if not rows:
            return codes
        contexts, images, forced = [], [], []
        for i in rows:
            e = self.episodes[i]
            j = part_at(e.starts, t)
            pictures, azimuth = views_and_pose(self.views(obs[i], self.view_rng[i]))
            context = build_context(e.texts[j], self.memory[i], standing_for(azimuth))
            thought = None
            if e.oracle and t == e.starts[j]:
                part = S.instruction_at(e.task, t) if e.task.parts else e.task
                thought = canonical_thought(e.texts[j], e.records[j], S.scene_summary(world, i, part))
                context += f"<think>\n{thought}\n</think>\n\n"
            contexts.append(context)
            images.append(pictures if self.vision else [])
            forced.append(thought)
        result = self.policy.think_then_act(obs[rows], contexts, images if self.vision else None, max_new_tokens=self.max_new_tokens)
        side = world.mass.shape[-1]
        for k, i in enumerate(rows):
            e = self.episodes[i]
            j = part_at(e.starts, t)
            text = forced[k] if forced[k] is not None else result.thought[k]
            if text:
                self.note(i, t, j, text, forced[k] is not None)
            if bool(result.acts[k]):
                logits = result.logits[k].float().cpu()  # (n, PATCH^3, 27)
                if self.sample:
                    picks = torch.multinomial(logits.softmax(-1).reshape(-1, logits.shape[-1]), 1, generator=self.draws[i]).view(1, *logits.shape[:2])
                else:
                    picks = logits.argmax(-1).unsqueeze(0)
                codes[i] = unpatchify3(picks, side)[0]
            elif bool(result.finished[k]):
                self.spoke[i] = {"t": t, "text": result.answer[k]}
            else:
                self.unfinished[i] += 1
        return codes


class ExpertActor:
    """The scripted expert as demos.write plays the training stores (objects on): the object expert
    for every task but build, the build planner for build tasks (build.Mixed), cube moves, the rule."""

    def __init__(self, episodes, seed=0, rule=RULE):
        from beebot.vla.voxel import build as B
        from beebot.vla.voxel.objects import ObjectExpert

        tasks = [e.task for e in episodes]
        other = lambda some: ObjectExpert(some, moves=CUBE_MOVES, seed=seed, rule=rule)  # noqa: E731
        if any(task.family == B.FAMILY for task in tasks):
            self.player = B.Mixed(tasks, other, lambda some: B.make_expert(some, moves=CUBE_MOVES, seed=seed, rule=rule))
        else:
            self.player = other(tasks)

    def __call__(self, world, t):
        return self.player(world, t)


def play_policy(policy, episodes, views=stand_in_views, sample=False, max_new_tokens=MAX_NEW_TOKENS, rule=RULE, failure=FAILURE, record=False):
    """Play a batch with the policy: Played per episode with its thoughts, reply and unfinished steps."""
    actor = PolicyActor(policy, episodes, views, sample, max_new_tokens)
    out = play(episodes, actor, rule, failure, record)
    for i, played in enumerate(out):
        played.thoughts, played.spoke, played.unfinished = actor.thoughts[i], actor.spoke[i], actor.unfinished[i]
    return out


def play_expert(episodes, seed=0, rule=RULE, failure=FAILURE, record=False):
    return play(episodes, ExpertActor(episodes, seed, rule), rule, failure, record)


def batches(episodes, size):
    """Indices of episodes in batches of at most `size`, episodes of the same length together."""
    order = sorted(range(len(episodes)), key=lambda i: episodes[i].steps)
    return [order[k : k + size] for k in range(0, len(order), size)]
