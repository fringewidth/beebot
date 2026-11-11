"""Closed-loop episodes of the think-then-act voxel policy (voxel/think.py), and of the scripted
expert on the same scenes.

Each step of a policy episode: render the scene's views, build the context exactly as training
does (think_data.context: the memory turn = the latest first thought and the latest switch line
after it, then the user turn with the images and the instruction in force), call
ThinkVoxelPolicy.think_then_act for every live episode of the batch at once on the observation in
the camera's frame, turn each acting row's slot logits into action codes (argmax, or sampled with
the episode's own generator), turn those back into the world's frame, and step the world under the
run's pivot rule and move failure rates. Thoughts are recorded when non-empty. A switch task's
instruction changes at its step (Task.spec["steps"]); the memory still holds the old
instruction's thought on that step, as in training. A reply in words (refuse or clarify: the row
reached END without the act marker) ends the episode: no robot moves again and the episode is
scored on the scene as it is. A reply that runs out of tokens before the act marker or END moves
nothing that step (counted as unfinished).

Views: views.scene_views with one look (views.sample_look: cameras, lights, materials) per
episode, kept for all its steps (SceneViews); think_data.stand_in_views only when asked for.

The camera's frame, as think_data uses it (camera_frame.py): the user stands at the camera. The
camera's quarter turn k (camera_frame.quarter_turns of the views' pose azimuth on the episode's
first step; Episode.camera pins its board side) turns everything the policy reads: the voxel
observation (camera_frame.turn_grid) and every instruction's direction words
(think_data.in_user_frame); the policy's action codes are in that frame too and are turned back
(camera_frame.unturn_codes) before the world steps. An instruction whose direction words cannot be
turned (think_data.mappable: near side, north, toward me, ...) means the world's own sides, so its
episode puts the camera at the world's front (k = 0), as training does when no other label fits.
Texts already said in the camera's frame (Episode.as_said, basics eval) go as they are.

Oracle thoughts (Episode.oracle): on the first step of each instruction the context is followed by
the canonical first thought (oracle_thought: think_data.template_thought cut to what images show
and said in the camera's frame, the training template's thought) in the reply's own form,
"<think>\\n{thought}\\n</think>\\n\\n", and the policy generates from there: the act marker (then
its slots) or an answer. Later steps generate freely, with the forced thought in memory.

Determinism: every episode carries a seed; its world noise (step's move order), move failures,
action sampling and view randomization come from generators of its own seeded from it,
so an episode plays the same whatever else is in its batch (up to the LM's batch numerics) and on
every run. The expert arm is the exception: the suite's experts keep one generator for their
batch, so expert episodes are deterministic for a given seed and batch.

Conditions (the training stores, data/vla/voxel_full/s*/small.pt config): pivot rule switch_cube,
gravity on, cube moves for the expert, failure 0.10 per move and 0.23 for up-and-over moves
(FAILURE), 72 steps (STEPS) and 144 for build tasks (BUILD_STEPS, data/vla/voxel_build), every
world with the object fields (world.with_objects, MAX_OBJECTS slots).
"""

from dataclasses import dataclass, field, replace

import numpy as np
import torch

from beebot.vla.voxel import suite as S
from beebot.vla.voxel import think_data as T
from beebot.vla.voxel.camera_frame import quarter_turns, turn_grid, unturn_codes
from beebot.vla.voxel.lm import unpatchify3
from beebot.vla.voxel.world import CUBE_MOVES, MAX_OBJECTS, STAY, concat, failure_rates, observe, step, with_objects

RULE = "switch_cube"
FAILURE = failure_rates(0.10, up_over=0.23)
STEPS = 72
BUILD_STEPS = 144
MAX_NEW_TOKENS = 160
FIRST_SHAPE = "you asked:"  # a first thought starts so (Haiku and template thoughts); any other thought is a switch or done line


# ---------------------------------------------------------------------------------------------
# Training's text and views


def oracle_thought(instruction, record, summary, turns=0):
    """The training template's first thought of an instruction (think_data.template_thought, cut to
    what images show by think_data.visible, said in the frame of a camera at quarter turn `turns`)
    from its world-frame text, its canonical record and the scene summary (suite.scene_summary)
    where it starts."""
    raw = T.template_thought(instruction, record, summary)
    return T.in_user_frame(T.visible(raw, instruction, record), turns)


class SceneViews:
    """views.scene_views with one look (views.sample_look) per episode, kept for all its steps; the
    rollout gives each episode its own rng, which keys the look. spec: a views.ViewSpec (default
    ViewSpec())."""

    def __init__(self, spec=None):
        from beebot.vla.voxel import views as V

        self.V = V
        self.spec = V.ViewSpec() if spec is None else spec
        self.looks = {}

    def __call__(self, obs, rng, side=None):
        """side: the board side the camera looks from (views.side_azimuth), drawn otherwise."""
        if id(rng) not in self.looks:
            spec = self.spec if side is None else replace(self.spec, azimuth=self.V.side_azimuth(side, rng))
            self.looks[id(rng)] = (rng, self.V.sample_look(rng, spec))  # the rng is kept so that its id is not reused
        return self.V.scene_views(obs, rng, self.spec, look=self.looks[id(rng)][1])


def camera_side(episode):
    """The board side an episode's camera must look from, or None (drawn by the views): the pinned
    side (Episode.camera); the world's front when an instruction's direction words cannot be turned
    into another camera's frame (think_data.mappable), unless the texts are said as they are."""
    unturnable = not episode.as_said and not all(T.mappable(text) for text in episode.texts)
    return episode.camera or ("front" if unturnable else None)


# ---------------------------------------------------------------------------------------------
# Episodes


@dataclass
class Episode:
    """One episode to play. world: the scene (a batch of one); task: what is scored (a switch task
    carries its parts); texts[j]: the instruction given from step starts[j] on; records[j]: the
    canonical record (suite.canonical_record) of the instruction in force then, for oracle thoughts
    and thought checks; oracle: feed canonical first thoughts; key: what identifies it in reports;
    camera: the board side the camera (and the user at it) looks from (else drawn); as_said: the
    texts are already in the camera's frame (said as they are, never turned; basics eval)."""

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
    camera: str | None = None
    as_said: bool = False


@dataclass
class Played:
    """What happened in an episode. thoughts: [{"t", "part", "text", "forced"}] of every non-empty
    thought; spoke: {"t", "text"} of the reply in words that ended it, or None; unfinished: steps
    whose reply ran out of tokens; moves: per step, the share of robots whose chosen action was not
    STAY; actions: (steps, D, H, W) int8 chosen codes (world frame) when recorded; turns, camera:
    the camera's quarter turn (the frame the policy saw and spoke in) and its azimuth (policy
    episodes)."""

    score: float
    unasked: bool
    passed: bool
    thoughts: list = field(default_factory=list)
    spoke: dict | None = None
    unfinished: int = 0
    moves: list = field(default_factory=list)
    actions: torch.Tensor | None = None
    turns: int = 0
    camera: float | None = None


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


def play(episodes, actor, rule=RULE, failure=FAILURE, record=False, outcomes=None, advance=None):
    """Play a batch of episodes with actor(world, t) -> (B, D, H, W) codes; the world and its
    scoring as suite.run (switch tasks scored on their last instruction from the step it came,
    runners flee every RUNNER_PERIOD steps), but with each episode's own generators and length:
    an episode past its steps stays still. outcomes: a world.OutcomeModel in place of failure
    (pass failure=None with it): each pivot ends as MuJoCo shows for its class. Returns a Played
    per episode (without the actor's thoughts, which the actor adds).
    advance(world, codes, priority, t) -> world, if given, carries out a step in place of the
    grid's world.step (hardware/closed_loop.py: pivoting cubes in MuJoCo); codes are the actor's
    after the failure draw, priority the step's move-order scores (B, D, H, W). It replaces the
    grid's step, so it does not go with an outcome model."""
    if outcomes is not None and failure is not None:
        raise ValueError("give failure rates or an outcome model, not both")
    if outcomes is not None and advance is not None:
        raise ValueError("advance carries out the steps in place of the grid; it takes no outcome model")
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
        priority = torch.stack([torch.rand(world.mass.shape[1:], generator=g) for g in noise])
        if advance is not None:
            world = advance(world, effective, priority, t)
        elif outcomes is None:
            world, _ = step(world, effective, priority, rule)
        else:
            world, _ = step(world, effective, priority, rule, outcomes, torch.stack([torch.rand(world.mass.shape[1:], generator=g) for g in fails]))
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
    episodes, each on its observation in its camera's frame; the codes it returns are in the world's
    frame. sample: draw each robot's action from its softmax (with the episode's generator) instead
    of the argmax. views(obs, rng, side=None) -> images or (images, pose) (side: the board side the
    camera must look from); None: SceneViews()."""

    def __init__(self, policy, episodes, views=None, sample=False, max_new_tokens=MAX_NEW_TOKENS, system=None):
        self.policy, self.episodes = policy, episodes
        self.views = SceneViews() if views is None else views
        self.sample, self.max_new_tokens, self.system = sample, max_new_tokens, system
        self.vision = getattr(policy, "visual", None) is not None
        n = len(episodes)
        self.memory = [[] for _ in range(n)]
        self.thoughts = [[] for _ in range(n)]
        self.spoke = [None] * n
        self.unfinished = [0] * n
        self.frames = [None] * n  # (camera quarter turns, camera azimuth), set on the first step
        self.view_rng = [np.random.default_rng([int(e.seed), 3]) for e in episodes]
        self.draws = [torch.Generator().manual_seed(int(e.seed) + 104_729) for e in episodes]

    def note(self, i, t, part, text, forced):
        """Record a thought and carry it in memory: a first thought replaces the memory, any other
        line becomes its second line (think_data.memory_before)."""
        self.thoughts[i].append({"t": t, "part": part, "text": text, "forced": forced})
        if text.startswith(FIRST_SHAPE) or not self.memory[i]:
            self.memory[i] = [text]
        else:
            self.memory[i] = [self.memory[i][0], text]

    def context(self, i, t, world, azimuth):
        """(context text, forced thought or None) of episode i at step t, as think_data.example
        builds it, in the frame of the camera at `azimuth`, fixed on the episode's first step."""
        e = self.episodes[i]
        if self.frames[i] is None:
            self.frames[i] = (quarter_turns(azimuth), azimuth)
        turns, _ = self.frames[i]
        j = part_at(e.starts, t)
        text = T.context(e.texts[j] if e.as_said else T.in_user_frame(e.texts[j], turns), self.memory[i], self.system)
        thought = None
        if e.oracle and t == e.starts[j]:
            part = S.instruction_at(e.task, t) if e.task.parts else e.task
            thought = oracle_thought(e.texts[j], e.records[j], S.scene_summary(world, i, part), turns)
            text += f"<think>\n{thought}\n</think>\n\n"
        return text, thought

    def __call__(self, world, t):
        obs = observe(world)
        codes = torch.full(world.mass.shape, STAY, dtype=torch.long)
        rows = [i for i, e in enumerate(self.episodes) if self.spoke[i] is None and t < e.steps]
        if not rows:
            return codes
        contexts, images, forced = [], [], []
        for i in rows:
            side = camera_side(self.episodes[i])
            pictures, azimuth = T.views_and_pose(self.views(obs[i], self.view_rng[i], **({"side": side} if side else {})))
            text, thought = self.context(i, t, world, azimuth)
            contexts.append(text)
            images.append(pictures if self.vision else [])
            forced.append(thought)
        seen = torch.stack([turn_grid(obs[i], self.frames[i][0]) for i in rows])  # each row in its camera's frame
        result = self.policy.think_then_act(seen, contexts, images if self.vision else None, max_new_tokens=self.max_new_tokens)
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
                codes[i] = unturn_codes(unpatchify3(picks, side)[0], self.frames[i][0])  # the camera's frame back to the world's
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


def play_policy(policy, episodes, views=None, sample=False, max_new_tokens=MAX_NEW_TOKENS, rule=RULE, failure=FAILURE, record=False):
    """Play a batch with the policy: Played per episode with its thoughts, reply, unfinished steps
    and camera frame. views None: SceneViews (one look per episode)."""
    actor = PolicyActor(policy, episodes, views, sample, max_new_tokens)
    out = play(episodes, actor, rule, failure, record)
    for i, played in enumerate(out):
        played.thoughts, played.spoke, played.unfinished = actor.thoughts[i], actor.spoke[i], actor.unfinished[i]
        if actor.frames[i] is not None:
            played.turns, played.camera = actor.frames[i]
    return out


def play_expert(episodes, seed=0, rule=RULE, failure=FAILURE, record=False):
    return play(episodes, ExpertActor(episodes, seed, rule), rule, failure, record)


def batches(episodes, size):
    """Indices of episodes in batches of at most `size`, episodes of the same length together."""
    order = sorted(range(len(episodes)), key=lambda i: episodes[i].steps)
    return [order[k : k + size] for k in range(0, len(order), size)]
