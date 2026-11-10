"""Training examples for the think-then-act voxel policy: one example per sampled step of an expert
demonstration (demos.py stores), with the instruction in force, a thought where the policy should
think, the policy's own earlier thoughts as memory, and the expert's action codes.

    data = ThinkData(["data/vla/voxel_full/s0", ...], ["data/vla/haiku/run-s0", "data/vla/haiku/run-s0-later", ...], tokenizer, seed=0)
    example = data.sample(np.random.default_rng(1))
    input_ids, labels = tokenize(example, tokenizer)
    held = data.validation(512, seed=0)

Interface (the trainer codes against exactly this)
--------------------------------------------------
ThinkData(stores, haiku_dirs, tokenizer=None, seed=0, thought_share=1/3, haiku_share=None,
          refusal_share=0.05, held_share=0.05, channels=world.N_CHANNELS, system=None)
  stores        demo store directories (demos.write --compact or dense; small.pt must hold
                "instructions", i.e. written after the coverage commit)
  haiku_dirs    Haiku label directories (haiku.Labels: verified pairs only; the test set is refused).
                A directory joins the stores written with its seed (config.json); its draw options
                must equal the store's (ValueError otherwise), and every joined label's family must
                equal the store's attempt (ValueError otherwise).
  tokenizer     optional; when given, check_tokens(tokenizer) runs (the fixed tokens below)
  seed          not used for sampling (sample takes an rng); kept for the caller's bookkeeping
  thought_share share of act examples drawn at a thought step (first step, instruction change,
                sub-goal switch, done); the rest at a uniformly drawn step without a thought
  haiku_share   None: episodes drawn uniformly (the natural share of Haiku-labelled ones);
                a float: that share of act examples from Haiku-labelled episodes
  refusal_share share of examples that are Haiku refuse/clarify pairs (when any exist)
  held_share    episodes held out for validation by a stable hash of (seed, index)
  channels      observation channels (world.observe)
  system        optional system message text

.sample(rng: numpy.random.Generator) -> Example        (training episodes only)
.validation(n, seed=0) -> list[Example]                (fixed: held-out episodes, same sampler)
.counts() -> dict                                      (episodes, labelled, template, refusals, ...)
tokenize(example, tokenizer, scene_tokens=64, slots=False) -> (input_ids (L,) long, labels (L,) long)
  The SCENE token is expanded to scene_tokens copies of SCENE_ID (the model writes its scene
  embeddings over them); labels are -100 except on the reply's tokens. slots=True appends
  scene_tokens copies of SLOT_ID after the act marker (label -100) for a model that wants the
  action slots in the id sequence; by default the model appends its slots after the last token,
  which on an act example is the act marker.
still(obs) -> (D, H, W) long: STAY everywhere (the action of a refusal, if a trainer needs one)

Example (dataclass)
  obs          (channels, D, H, W) float, the scene at step t (world.observe of the store's frame)
  context_text the text before the reply, with exactly one SCENE in the user turn
  reply_text   act: "<think>\\n{thought}\\n</think>\\n\\n<tool_call>" or, on a step without a thought,
               "<think>\\n\\n</think>\\n\\n<tool_call>"; refusal: "<think>\\n{thought}\\n</think>\\n\\n{answer}<|im_end|>"
  act          True for an act example
  codes        (D, H, W) long, the expert's action codes at step t (act); None for a refusal
  seed, index  the episode's draw key (demos.draws: attempts.jsonl line index of seed's store)
  e, t         store row and step
  rung         "close", "indirect", "casual" (Haiku), "refusal", or "template"
  family       the task family (attempts.jsonl)
  kind         "first", "change" (instruction change), "switch" (sub-goal switch), "done",
               "plain" (no thought this step) or "refusal"
  instruction  the user instruction in the current turn
  store        the store directory

Fixed tokens (Qwen3, checked by check_tokens on Qwen/Qwen3-0.6B): "<think>" 151667, "</think>"
151668, "<tool_call>" 151657 (ACT), "<|im_start|>" 151644, "<|im_end|>" 151645, SCENE =
"<|image_pad|>" 151655, SLOT = "<|video_pad|>" 151656 are single tokens, so they split the text
before BPE: the empty thought "<think>\\n\\n</think>\\n\\n" is always [151667, 271, 151668, 271], a
thought is [151667, 198, <thought tokens>, 151668, 271] where the thought's own tokens end in
198 ("\\n") or a merged ".\\n" (624), and tokenizing context and reply separately gives the same
ids as tokenizing their concatenation. This is Qwen3's own chat template for an assistant turn
with reasoning (reasoning.strip() between "<think>\\n" and "\\n</think>\\n\\n", then content).

Text layout
-----------
    [<|im_start|>system\\n{system}<|im_end|>\\n]                                  only with system
    [<|im_start|>assistant\\n{first thought}[\\n{latest line}]<|im_end|>\\n]       memory, t > 0
    <|im_start|>user\\n{instruction}\\n<|image_pad|><|im_end|>\\n<|im_start|>assistant\\n
    reply_text

Memory is fixed in size: the latest first thought emitted before step t (of the instruction in
force before t: on the step an instruction changes it is still the old one's) and the latest
switch or done line emitted after that first thought and before t; none at an episode's first
step. Thoughts (the approved format, lower case):
  first   "you asked: <instruction>. i see <scene>. <what it means>. first: <first sub-goal>."
          at the episode's first step, and at an instruction change for the new instruction
  switch  a short line at each sub-goal switch ("now the green one", "<sub-goal> next.")
  done    "done." at the step the expert records the task as finished (eat and push tasks:
          Expert.subgoals {"kind": "done"}; other tasks hold their last sub-goal to the end and
          get no done line)
Sub-goal switches come from small.pt "subgoals" (Expert.subgoals: an entry whenever the plan
segment changes); a switch is a step where the segment index of the instruction in force rises,
and its line is the Haiku switches[index - 1] (or the template line for record subgoals[index]).

Labels: Haiku choices when verified ones exist for the episode (one rung drawn per sample; a
switch episode only when every part has a verified instruction, one rung per part), otherwise the
store's template instruction with a templated first thought from canonical_record and
scene_summary (at the part's first step) and "<sub-goal> next." switch lines.

Rendered examples (dev store: seed 0, the first 48 scenes of voxel_full/s0, with run-s0 labels)
---------------------------------------------------------------------------------------------
First step, Haiku (e 7, eat:except, rung close):
    context: <|im_start|>user\\nleave the white cheese alone and eat the others\\n<|image_pad|><|im_end|>\\n<|im_start|>assistant\\n
    reply:   <think>\\nyou asked: leave the white cheese alone and eat the others. i see a red cheese on the back left plateau and a white one on the front right plateau. not white leaves red. first: eat the red cheese.\\n</think>\\n\\n<tool_call>
Ordinary step (same episode, t 30):
    context: <|im_start|>assistant\\nyou asked: leave the white cheese ... first: eat the red cheese.<|im_end|>\\n<|im_start|>user\\nleave the white cheese alone and eat the others\\n<|image_pad|><|im_end|>\\n<|im_start|>assistant\\n
    reply:   <think>\\n\\n</think>\\n\\n<tool_call>
Done step (t 51): reply "<think>\\ndone.\\n</think>\\n\\n<tool_call>"; later steps carry "done." as
the memory's second line.
Template first step (no verified label):
    reply:   <think>\\nyou asked: bridge the gap. i see 39 robots, a trench running left to right between two plateaus. that means: build a bridge across the trench. first: build a bridge across the trench.\\n</think>\\n\\n<tool_call>
Instruction change (switch:goto>stay at t 21): the user turn holds "stay where you are", the
memory the first instruction's thought, the reply the second part's first thought.
Refusal: context with the refusal instruction at t 0, reply
    <think>\\nyou asked: make the blue cheese float in the air. ... i will not act.\\n</think>\\n\\ngravity holds the blue cheese on the floor, so i will do nothing.<|im_end|>
(The README section "Think-then-act training examples" has the exact output of render_examples.)
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel import haiku as H
from beebot.vla.voxel import suite as S
from beebot.vla.voxel.lm import world_at
from beebot.vla.voxel.play_store import load
from beebot.vla.voxel.world import N_CHANNELS, STAY, observe

SCENE = "<|image_pad|>"  # where the scene tokens go: exactly one in the user turn
SLOT = "<|video_pad|>"  # optional action-slot placeholder (tokenize(slots=True))
ACT = "<tool_call>"  # the act marker (stream.ACT)
THINK = "<think>"
UNTHINK = "</think>"
START = "<|im_start|>"
END = "<|im_end|>"
EMPTY_THOUGHT = f"{THINK}\n\n{UNTHINK}\n\n"
SINGLE_TOKENS = (THINK, UNTHINK, ACT, START, END, SCENE, SLOT)
DONE_LINE = H.DONE_LINE  # "done."
TEMPLATE = "template"
REFUSAL = H.REFUSAL  # "refusal"
THOUGHT_KINDS = ("first", "change", "switch", "done")
STAY_LINE = "stay where you are"
MAX_SEEN = 4  # things (objects, cheeses, zones) a templated thought names; those the task names first
OPTION_KEYS = ("robots", "counts", "rule", "apart", "coverage", "objects")  # draw options a label directory and a store must share


# ---------------------------------------------------------------------------------------------
# Text


def thought_block(thought):
    """The reply's thought in Qwen3's own form; the empty form when there is no thought."""
    return f"{THINK}\n{thought.strip()}\n{UNTHINK}\n\n" if thought else EMPTY_THOUGHT


def act_reply(thought):
    return thought_block(thought) + ACT


def text_reply(thought, answer):
    return thought_block(thought) + answer.strip() + END


def context(instruction, memory=(), system=None):
    """The text before the reply: system, memory (earlier assistant text) and the user turn with
    the instruction and SCENE."""
    out = f"{START}system\n{system}{END}\n" if system else ""
    if memory:
        out += f"{START}assistant\n" + "\n".join(memory) + f"{END}\n"
    return out + f"{START}user\n{instruction.strip()}\n{SCENE}{END}\n{START}assistant\n"


def article(phrase):
    return ("an " if phrase[:1] in "aeiou" or phrase.startswith("l-") else "a ") + phrase


def where(area):
    return "in the center" if area == "center" else f"on the {area}" if area in ("left", "right") else f"at the {area}"


def plainly_drawn(objects):
    """Whether every object is a plain small cube (the suite's cheeses): then colour and kind say it."""
    return all(o.get("shape", "cube") == "cube" and o.get("pattern", "plain") == "plain" and o.get("size", "small") == "small" for o in objects)


def object_phrase(entry, plain):
    from beebot.vla.voxel import objects as O

    if plain:
        words = [entry["colour"], entry.get("kind", "cheese")]
    else:
        words = [entry["size"], O.PATTERN_WORDS.get(entry["pattern"]), entry["colour"], O.SHAPE_WORDS[entry["shape"]], entry["kind"]]
    on = entry.get("on", "floor")
    return article(" ".join(w for w in words if w).lower()) + (f" on a {on}" if on != "floor" else "") + " " + where(entry["area"])


def zone_phrase(entry):
    return article(f"{entry['colour']} {entry['kind']}") + " " + where(entry["area"])


def named_in(spec):
    """(strings, object ids) a record spec names: its colours, regions, zones and target ids."""
    strings, ids = set(), set()

    def walk(value, key=None):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, k)
        elif isinstance(value, (list, tuple)):
            for v in value:
                walk(v, key)
        elif isinstance(value, str):
            strings.add(value.lower())
        elif isinstance(value, int) and not isinstance(value, bool) and key in ("target", "object"):
            ids.add(value)

    walk(spec)
    return strings, ids


def joined(parts):
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def sight(summary, record):
    """What a templated thought says it sees: the swarm and terrain, then up to MAX_SEEN things
    (objects or cheeses, and zones), those the task names first."""
    robots = summary["robots"]["count"]
    terrain = summary["terrain"]
    swarm = f"{robots} robot{'s' if robots != 1 else ''}"
    parts = []
    if terrain["layout"] == "flat floor":
        parts.append(swarm + " on a flat floor")
    elif terrain["layout"] == "low hills":
        parts.append(swarm + " on low hills")
    else:
        parts += [swarm, f"a trench running {terrain['trench']['runs']} between two plateaus"]
    strings, ids = named_in(record.get("spec", {}))
    things = summary.get("objects") if summary.get("objects") is not None else [dict(c, kind="cheese") for c in summary.get("cheeses", [])]
    plain = plainly_drawn(things)
    entries = [(o.get("id") in ids or o["colour"] in strings, object_phrase(o, plain)) for o in things]
    entries += [(z["colour"] in strings, zone_phrase(z)) for z in summary.get("zones", [])]
    ordered = [p for named, p in entries if named] + [p for named, p in entries if not named]
    parts += ordered[:MAX_SEEN]
    if len(ordered) > MAX_SEEN:
        parts.append(f"{len(ordered) - MAX_SEEN} more thing{'s' if len(ordered) - MAX_SEEN > 1 else ''}")
    return joined(parts)


def bare(text):
    return text.strip().rstrip(".!?").strip()


def template_thought(instruction, record, summary):
    """The first thought of a template instruction, from its canonical record and the scene."""
    subgoals = record.get("subgoals") or []
    if record.get("spec", {}).get("kind") == "refuse":
        meaning = "nothing here matches it, so there is nothing to do"
    else:
        meaning = f"that means: {bare(record['plain'])}"
    first = bare(subgoals[0]) if subgoals else STAY_LINE
    return f"you asked: {bare(instruction)}. i see {sight(summary, record)}. {meaning}. first: {first}.".lower()


def template_line(record, index):
    """The switch line of plan segment `index` of a record."""
    subgoals = record.get("subgoals") or []
    goal = bare(subgoals[index]) if 0 <= index < len(subgoals) else "the next part"
    return f"{goal} next.".lower()


# ---------------------------------------------------------------------------------------------
# Episodes


@dataclass(frozen=True)
class Event:
    """A step with a thought: kind first, change, switch or done; part: the instruction in force;
    index: the plan segment a switch starts."""

    t: int
    kind: str
    part: int
    index: int


def part_at(starts, t):
    return max(j for j, start in enumerate(starts) if start <= t)


def episode_events(instructions, subgoals, steps):
    """The thought steps of an episode from its instruction entries (small.pt "instructions") and
    sub-goal entries (small.pt "subgoals"), in step order."""
    starts = [int(entry["t"]) for entry in instructions]
    out = {}
    for j, start in enumerate(starts):
        if start < steps:
            out[start] = Event(start, "first" if j == 0 else "change", j, 0)
    highest = {}
    for entry in subgoals:
        t, index = int(entry["t"]), int(entry["index"])
        if t >= steps:
            continue
        j = part_at(starts, t)
        if t in out:
            highest[j] = max(highest.get(j, 0), index)
            continue
        if entry["goal"].get("kind") == "done":
            out[t] = Event(t, "done", j, index)
        elif index > highest.get(j, 0):
            out[t] = Event(t, "switch", j, index)
            highest[j] = index
    return [out[t] for t in sorted(out)]


def memory_before(events, t):
    """The events whose thoughts the context carries at step t: the latest first/change event
    before t and the latest switch/done event between it and t."""
    before = [ev for ev in events if ev.t < t]
    heads = [ev for ev in before if ev.kind in ("first", "change")]
    if not heads:
        return []
    head = heads[-1]
    tail = [ev for ev in before if ev.t > head.t and ev.kind in ("switch", "done")]
    return [head] + tail[-1:]


@dataclass
class Episode:
    store: int  # index into ThinkData.stores
    e: int
    seed: int
    index: int
    family: str
    held: bool
    choices: list  # Haiku choices (non-switch episodes)
    parts: list  # Haiku switch_parts (switch episodes; [] unless every part is verified)
    refusals: list

    @property
    def labelled(self):
        return bool(self.choices or self.parts)


class Pool:
    """Episodes to draw from, split once by what the sampler asks of them."""

    def __init__(self, episodes):
        self.episodes = episodes
        self.labelled = [ep for ep in episodes if ep.labelled]
        self.unlabelled = [ep for ep in episodes if not ep.labelled]
        self.refusing = [ep for ep in episodes if ep.refusals]


def held_out(seed, index, share):
    """Whether episode (seed, index) is held out: a stable hash, the same on every machine."""
    digest = hashlib.sha256(f"{seed}/{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < share


@dataclass
class Example:
    obs: torch.Tensor
    context_text: str
    reply_text: str
    act: bool
    codes: torch.Tensor
    seed: int
    index: int
    e: int
    t: int
    rung: str
    family: str
    kind: str
    instruction: str
    store: str


def still(obs):
    return torch.full(tuple(obs.shape[-3:]), STAY, dtype=torch.long)


def label_options(directory):
    path = Path(directory) / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


class ThinkData:
    def __init__(self, stores, haiku_dirs, tokenizer=None, seed=0, thought_share=1 / 3, haiku_share=None, refusal_share=0.05, held_share=0.05, channels=N_CHANNELS, system=None):
        if tokenizer is not None:
            check_tokens(tokenizer)
        self.seed = seed
        self.thought_share = thought_share
        self.haiku_share = haiku_share
        self.refusal_share = refusal_share
        self.held_share = held_share
        self.channels = channels
        self.system = system
        self.stores = [str(s) for s in stores]
        self.data = [load(s) for s in self.stores]
        self.labels = []
        for directory in haiku_dirs:
            directory = Path(directory)
            assert not H.is_test(directory), f"{directory} is the held-out test set; it never goes into training"
            self.labels.append((H.Labels(directory.name, root=directory.parent), label_options(directory)))
        self.episodes = []
        self.events = {}
        self.templates = {}
        for k, (path, store) in enumerate(zip(self.stores, self.data)):
            if "instructions" not in store:
                raise ValueError(f"{path} has no per-step instructions (small.pt 'instructions'); rewrite it with the current demos.py")
            self.episodes += self.join(k, path, store)
        self.train = [ep for ep in self.episodes if not ep.held]
        self.held = [ep for ep in self.episodes if ep.held]
        self.train_pool, self.held_pool = Pool(self.train), Pool(self.held)
        self.validation_cache = {}

    def join(self, k, path, store):
        """The kept episodes of store k with their Haiku labels (by (seed, index))."""
        config = store.get("config", {})
        seed = int(config["seed"])
        mine = []
        for labels, options in self.labels:
            if options.get("seed", seed) != seed:
                continue
            for key in OPTION_KEYS:
                if key in options and key in config and options[key] != config[key]:
                    raise ValueError(f"{labels.directory} was drawn with {key}={options[key]} but {path} with {config[key]}: the (seed, index) join would be wrong")
            mine.append(labels)
        out = []
        for n, row in enumerate(H.read_jsonl(Path(path) / "attempts.jsonl")):
            if row["e"] < 0:
                continue
            choices, parts, refusals = [], [], []
            for labels in mine:
                for label_row in labels.by_key.get((seed, n), []):
                    if label_row.get("family") != row["family"]:
                        raise ValueError(f"label {label_row.get('pair_id')} is family {label_row.get('family')} but {path} attempt {n} is {row['family']}: the (seed, index) join is broken")
                choices += labels.choices(seed, n)
                parts = parts or labels.switch_parts(seed, n)
                refusals += labels.refusals(seed, n)
            if parts:
                starts = [int(entry["t"]) for entry in store["instructions"][row["e"]]]
                if [p["t"] for p in parts[: len(starts)]] != starts:
                    parts = []  # the parts do not line up with the store's instruction changes
            out.append(Episode(k, int(row["e"]), seed, n, row["family"], held_out(seed, n, self.held_share), choices, parts, refusals))
        return out

    # -- per-episode pieces

    def steps(self, ep):
        return int(self.data[ep.store]["actions"].shape[1])

    def instructions(self, ep):
        return self.data[ep.store]["instructions"][ep.e]

    def episode_events(self, ep):
        key = (ep.store, ep.e)
        if key not in self.events:
            store = self.data[ep.store]
            self.events[key] = episode_events(store["instructions"][ep.e], store["subgoals"][ep.e], self.steps(ep))
        return self.events[key]

    def template_parts(self, ep):
        """Per instruction: {rung, instruction, thought, switches, done} from the store's template."""
        key = (ep.store, ep.e)
        if key not in self.templates:
            store = self.data[ep.store]
            parts = []
            for entry in self.instructions(ep):
                record = entry["record"]
                summary = S.scene_summary(world_at(store, [(ep.e, int(entry["t"]))]), 0)
                switches = [template_line(record, j) for j in range(1, len(record.get("subgoals") or []))]
                parts.append({"rung": TEMPLATE, "instruction": entry["text"], "thought": template_thought(entry["text"], record, summary), "switches": switches, "done": DONE_LINE})
            self.templates[key] = parts
        return self.templates[key]

    def labelling(self, ep, rng):
        """One choice per instruction: Haiku (a rung drawn per part) or the template."""
        if ep.parts:
            return [part["choices"][int(rng.integers(len(part["choices"])))] for part in ep.parts[: len(self.instructions(ep))]]
        if ep.choices:
            return [ep.choices[int(rng.integers(len(ep.choices)))]]
        return self.template_parts(ep)

    def thought_of(self, ep, parts, ev):
        part = parts[ev.part]
        if ev.kind in ("first", "change"):
            return part["thought"]
        if ev.kind == "done":
            return part.get("done") or DONE_LINE
        switches = part.get("switches") or []
        if 0 <= ev.index - 1 < len(switches) and switches[ev.index - 1]:
            return switches[ev.index - 1]
        return template_line(self.instructions(ep)[ev.part]["record"], ev.index)

    def observation(self, ep, t):
        return observe(world_at(self.data[ep.store], [(ep.e, t)]), self.channels)[0]

    # -- examples

    def example(self, ep, t, parts):
        """The act example of episode ep at step t under labelling parts (one per instruction)."""
        events = self.episode_events(ep)
        starts = [int(entry["t"]) for entry in self.instructions(ep)]
        j = part_at(starts, t)
        here = next((ev for ev in events if ev.t == t), None)
        memory = [self.thought_of(ep, parts, ev) for ev in memory_before(events, t)]
        thought = self.thought_of(ep, parts, here) if here is not None else ""
        codes = self.data[ep.store]["actions"][ep.e, t].long()
        return Example(
            obs=self.observation(ep, t),
            context_text=context(parts[j]["instruction"], memory, self.system),
            reply_text=act_reply(thought),
            act=True,
            codes=codes,
            seed=ep.seed,
            index=ep.index,
            e=ep.e,
            t=t,
            rung=parts[j]["rung"],
            family=ep.family,
            kind=here.kind if here is not None else "plain",
            instruction=parts[j]["instruction"],
            store=self.stores[ep.store],
        )

    def refusal(self, ep, pair):
        return Example(
            obs=self.observation(ep, 0),
            context_text=context(pair["instruction"], (), self.system),
            reply_text=text_reply(pair["thought"], pair["reply"]),
            act=False,
            codes=None,
            seed=ep.seed,
            index=ep.index,
            e=ep.e,
            t=0,
            rung=REFUSAL,
            family=ep.family,
            kind=REFUSAL,
            instruction=pair["instruction"],
            store=self.stores[ep.store],
        )

    def draw(self, pool, rng):
        """One example from a pool of episodes (Pool: the sampler of sample and validation)."""
        if pool.refusing and self.refusal_share and rng.random() < self.refusal_share:
            ep = pool.refusing[int(rng.integers(len(pool.refusing)))]
            return self.refusal(ep, ep.refusals[int(rng.integers(len(ep.refusals)))])
        episodes = pool.episodes
        if self.haiku_share is not None and pool.labelled and pool.unlabelled:
            episodes = pool.labelled if rng.random() < self.haiku_share else pool.unlabelled
        ep = episodes[int(rng.integers(len(episodes)))]
        events = self.episode_events(ep)
        thought_steps = [ev.t for ev in events]
        steps = self.steps(ep)
        if rng.random() < self.thought_share:
            t = thought_steps[int(rng.integers(len(thought_steps)))]
        else:
            others = sorted(set(range(steps)) - set(thought_steps))
            t = others[int(rng.integers(len(others)))] if others else int(rng.integers(steps))
        return self.example(ep, t, self.labelling(ep, rng))

    def sample(self, rng):
        return self.draw(self.train_pool, rng)

    def validation(self, n, seed=0):
        key = (n, seed)
        if key not in self.validation_cache:
            if not self.held:
                raise ValueError("no held-out episodes (too few episodes for held_share)")
            rng = np.random.default_rng(seed)
            self.validation_cache[key] = [self.draw(self.held_pool, rng) for _ in range(n)]
        return self.validation_cache[key]

    def counts(self):
        """How labels and templates mix: episode counts per source, and per family."""
        out = {"episodes": len(self.episodes), "train": len(self.train), "held": len(self.held)}
        out["haiku"] = sum(bool(ep.choices) for ep in self.episodes)
        out["haiku_switch"] = sum(bool(ep.parts) for ep in self.episodes)
        out["template"] = sum(not ep.labelled for ep in self.episodes)
        out["with_refusals"] = sum(bool(ep.refusals) for ep in self.episodes)
        out["refusal_pairs"] = sum(len(ep.refusals) for ep in self.episodes)
        out["haiku_choices"] = sum(len(ep.choices) for ep in self.episodes)
        families = {}
        for ep in self.episodes:
            row = families.setdefault(ep.family, {"episodes": 0, "haiku": 0, "template": 0})
            row["episodes"] += 1
            row["haiku" if ep.labelled else "template"] += 1
        out["families"] = dict(sorted(families.items()))
        out["per_store"] = {path: sum(ep.store == k for ep in self.episodes) for k, path in enumerate(self.stores)}
        out["per_store_haiku"] = {path: sum(ep.store == k and ep.labelled for ep in self.episodes) for k, path in enumerate(self.stores)}
        return out


# ---------------------------------------------------------------------------------------------
# Tokens


def token_id(tokenizer, text):
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"{text!r} is {len(ids)} tokens in this tokenizer, not one")
    return ids[0]


def check_tokens(tokenizer):
    """{token: id} of the fixed tokens; raises unless each is a single token."""
    return {text: token_id(tokenizer, text) for text in SINGLE_TOKENS}


def tokenize(example, tokenizer, scene_tokens=64, slots=False):
    """(input_ids, labels): the context with SCENE expanded to scene_tokens copies of its id, then
    the reply (labels: the reply's ids, -100 elsewhere); with slots, scene_tokens copies of SLOT's
    id after an act reply (label -100)."""
    scene_id = token_id(tokenizer, SCENE)
    head = tokenizer(example.context_text, add_special_tokens=False).input_ids
    where_scene = [i for i, v in enumerate(head) if v == scene_id]
    if len(where_scene) != 1:
        raise ValueError(f"the context holds {len(where_scene)} SCENE tokens, not one")
    i = where_scene[0]
    head = head[:i] + [scene_id] * scene_tokens + head[i + 1 :]
    reply = tokenizer(example.reply_text, add_special_tokens=False).input_ids
    tail = [token_id(tokenizer, SLOT)] * scene_tokens if slots and example.act else []
    ids = head + reply + tail
    labels = [-100] * len(head) + reply + [-100] * len(tail)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def render_examples(data, rng=None):
    """One rendered example of each kind found ({kind: Example}), for docs and checks."""
    rng = np.random.default_rng(0) if rng is None else rng
    out = {}
    for ep in data.episodes:
        parts = data.labelling(ep, rng)
        for ev in data.episode_events(ep):
            out.setdefault(ev.kind + (":haiku" if ep.labelled else ":template"), data.example(ep, ev.t, parts))
        events = data.episode_events(ep)
        plain = [t for t in range(data.steps(ep)) if t not in {ev.t for ev in events} and memory_before(events, t)]
        if plain:
            out.setdefault("plain" + (":haiku" if ep.labelled else ":template"), data.example(ep, plain[len(plain) // 2], parts))
        for pair in ep.refusals[:1]:
            out.setdefault(REFUSAL, data.refusal(ep, pair))
    return out


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description="counts, rendered examples and sampling speed of ThinkData")
    parser.add_argument("--stores", nargs="+", required=True)
    parser.add_argument("--haiku", nargs="*", default=[])
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--show", action="store_true", help="print one rendered example of each kind")
    args = parser.parse_args()
    torch.set_num_threads(1)
    t0 = time.time()
    data = ThinkData(args.stores, args.haiku)
    print(f"loaded in {time.time() - t0:.1f} s")
    print(json.dumps(data.counts(), indent=1))
    if args.show:
        for kind, ex in render_examples(data).items():
            print(f"\n== {kind} (e {ex.e}, t {ex.t}, {ex.family}, {ex.rung})\n{ex.context_text}{ex.reply_text}")
    rng = np.random.default_rng(0)
    for label, n in (("cold", args.samples), ("warm", args.samples)):
        t0 = time.time()
        kinds = {}
        for _ in range(n):
            ex = data.sample(rng)
            kinds[ex.kind] = kinds.get(ex.kind, 0) + 1
        rate = n / (time.time() - t0)
        print(f"{label}: {rate:.0f} examples/s ({rate * 60:.0f}/min) on one core; kinds {json.dumps(kinds)}")


if __name__ == "__main__":
    main()
