"""Training examples for the think-then-act voxel policy (Qwen3.5-4B, a vision-language model): one
example per sampled step of an expert demonstration (demos.py stores), with the scene as images,
the instruction in force, a thought where the policy should think, the policy's own earlier
thoughts as memory, the voxel observation (the action slots are built from it) and the expert's
action codes.

    data = ThinkData(["data/vla/voxel_full/s0", ...], ["data/vla/haiku/run-s0", "data/vla/haiku/run-s0-later", ...], tokenizer, seed=0)
    example = data.sample(np.random.default_rng(1))
    input_ids, labels = tokenize(example, tokenizer)
    held = data.validation(512, seed=0)

Interface (the trainer codes against exactly this)
--------------------------------------------------
ThinkData(stores, haiku_dirs, tokenizer=None, seed=0, thought_share=1/3, haiku_share=None,
          refusal_share=0.05, held_share=0.05, channels=world.N_CHANNELS, system=None,
          views=stand_in_views, validation_views=None, mismatch="raise")
  stores        demo store directories (demos.write --compact or dense; small.pt must hold
                "instructions", i.e. written after the coverage commit)
  haiku_dirs    Haiku label directories (haiku.Labels: verified pairs only; the test set is refused).
                A directory joins the stores written with its seed (config.json); its draw options
                must equal the store's (ValueError otherwise), and every joined label must belong to
                its episode (see "Join check" below).
  tokenizer     optional; when given, check_tokens(tokenizer) runs (the fixed tokens below)
  seed          not used for sampling (sample takes an rng); kept for the caller's bookkeeping
  thought_share share of act examples drawn at a thought step (first step, instruction change,
                sub-goal switch, done); the rest at a uniformly drawn step without a thought.
                With refusals (which always think) about 0.37 of all examples carry a thought.
  haiku_share   None: episodes drawn uniformly (the natural share of Haiku-labelled ones);
                a float: that share of act examples from Haiku-labelled episodes
  refusal_share share of examples that are Haiku refuse/clarify pairs (when any exist)
  held_share    episodes held out for validation by a stable hash of (seed, index)
  channels      observation channels (world.observe)
  system        optional system message text
  views         views(obs, rng) -> images, or (images, pose): PIL images of the observation,
                called at sample time with the sampler's rng (fresh randomization on every draw).
                pose["azimuth"] is the camera's place round the arena in degrees, measured from
                the world's front (+y) toward its right (+x): 0 front, 90 right, 180 back, 270
                left; the camera looks at the arena. Plain images count as azimuth 0. The stand-in
                is one isometric render (azimuth 45: the front right corner); the switch to the
                randomized views is one line:
                views=lambda obs, rng: scene_views(obs, rng, ViewSpec())  (voxel/views.py)
  validation_views  the views of validation examples (e.g. a clean mode); None: `views`, with the
                validation seed's rng, so validation examples are the same on every call and run
  mismatch      "raise": a label that does not belong to its episode raises ValueError; "drop":
                that episode's labels are dropped (it keeps its template) and counted

.sample(rng: numpy.random.Generator) -> Example        (training episodes only)
.validation(n, seed=0) -> list[Example]                (fixed: held-out episodes, same sampler)
.counts() -> dict                                      (episodes, labelled, template, refusals, mismatched, rewritten)
.mismatches: list of (store, attempt index, problem) of the episodes whose labels were dropped
tokenize(example, tokenizer, counts=None, slots=0) -> (input_ids (L,) long, labels (L,) long)
  IMAGES is expanded as the Qwen3-VL processor expands an image: <|vision_start|>, n x
  <|image_pad|>, <|vision_end|> per image, n = counts[i] or image_tokens(image) (Qwen3.5's resize:
  sides rounded to multiples of 32 pixels, one token per 32 x 32; the stand-in's 272 x 272 image
  is 256 x 256, 64 tokens). Labels are -100 except on the reply's tokens. slots=n appends n SLOT
  ids after an act reply (label -100) for a model that wants its action slots in the id sequence;
  otherwise the model appends its slots after the last token, which on an act example is the act
  marker.
image_tokens(image) -> int; still(obs) -> (D, H, W) long, STAY everywhere (a refusal's action,
  if a trainer needs one); stand_in_views(obs, rng=None) -> [PIL image]

Example (dataclass)
  obs          (channels, D, H, W) float, the scene at step t (world.observe of the store's frame)
  images       views(obs, rng): list of PIL images of that scene
  context_text the text before the reply, with "<IMAGES>" exactly once, at the start of the user turn
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
  instruction  the user instruction in the current turn (in the user's frame)
  store        the store directory
  user         where the user stands relative to the camera: "behind", "left", "right", "opposite"
  turns        the user's quarter turn round the arena (0: the world's own frame), see below
  camera       the camera's azimuth the views reported (degrees, as pose["azimuth"])

Fixed tokens (checked by check_tokens on Qwen/Qwen3.5-4B, local_files_only): "<think>" 248068,
"</think>" 248069, "<tool_call>" 248058 (ACT), "<|im_start|>" 248045, "<|im_end|>" 248046,
"<|vision_start|>" 248053, "<|image_pad|>" 248056, "<|vision_end|>" 248054, SLOT =
"<|video_pad|>" 248057 are single tokens (added tokens: they split the text before BPE). The
empty thought "<think>\\n\\n</think>\\n\\n<tool_call>" is always [248068, 271, 248069, 271, 248058]; a
thought is [248068, 198, <thought tokens>, 248069, 271] followed by 248058 (act) or the answer, the
thought's own tokens ending in 198 ("\\n") or a merged ".\\n"; tokenizing context and reply apart
gives the ids of their concatenation. This is the Qwen3.5 chat template's own assistant turn
(reasoning between "<think>\\n" and "\\n</think>\\n\\n", then the content; its generation prompt
with thinking off is "<think>\\n\\n</think>\\n\\n"). "<IMAGES>" itself is not a token (3 tokens):
it never reaches the model unexpanded. Qwen3-0.6B has the same tokens with ids 151667, 151668,
151657, 151644, 151645, 151652, 151655, 151653, 151656.

Text layout
-----------
    [<|im_start|>system\\n{system}<|im_end|>\\n]                                  only with system
    [<|im_start|>assistant\\n{first thought}[\\n{latest line}]<|im_end|>\\n]       memory, t > 0
    <|im_start|>user\\n<IMAGES>\\n{where the user stands}\\n{instruction}<|im_end|>\\n<|im_start|>assistant\\n
    reply_text                                                                   starts with "<think>\\n"

The context ends exactly at "<|im_start|>assistant\\n" and the reply starts with "<think>\\n" (Qwen3.5's
generation prompt ends in "<think>\\n", so the thought is always the reply's); an acting reply ends
with "<tool_call>".

The user's frame. The camera can be anywhere round the arena, and the user stands at one of four
places relative to it, drawn per example: "(i'm standing behind the camera)" (seeing what the
camera sees), "(i'm standing to the left of the camera)" and "(... to the right ...)" (a quarter
turn round the arena from the camera, facing the arena), "(i'm standing opposite the camera,
facing it)". The user's place round the arena is the camera's azimuth plus 0, -90, +90 or 180
degrees, rounded to the nearest quarter turn (half way goes to the later one): turns 0 is the
world's front (the world's own frame), 1 its right, 2 its back, 3 its left. Every direction word
of the instruction, the thoughts, the memory, the switch lines and the refusal's answer is said as
that user says it (in_user_frame): front/back/left/right and the corners (front/back first), -most
and -ward forms, "in front of", "behind" and "to the left/right of" a thing, forward(s),
backward(s), "back" after a motion verb, and "side by side" / "one behind the other" (swapped on a
quarter turn); idioms ("right now", "nothing left", "come back") stay. A label whose wording cannot
be turned reliably (UNMAPPABLE: near/far side or end, north/south/east/west, toward me, top left,
right-hand, clockwise, ...) is used only when turns is 0: otherwise another verified rung, then the
template, then the user's place that makes turns 0 (counts()["frames"] counts each). The voxel
observation and the action codes stay in the world's frame.

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

Rendered examples (data/vla/voxel_full/s0 with run-s0 and run-s0-later labels; render_examples, the CLI's --show)
--------------------------------------------------------------------------------------------------------------
These are in the world's frame, with "(i'm standing behind the camera)\\n" before each instruction left
out for width; a turned one follows them.
First step, Haiku (e 0, t 0, parallel, rung casual):
    context: <|im_start|>user\\n<IMAGES>\\nsplit up pls, half go front n the rest go front right<|im_end|>\\n<|im_start|>assistant\\n
    reply:   <think>\\nyou asked: split up pls, half go front n the rest go front right. i see the robots. two halves, two spots. first: split into two groups: one should go to the front, the other should go to the front right.\\n</think>\\n\\n<tool_call>
Ordinary step (same episode, t 36):
    context: <|im_start|>assistant\\nyou asked: split up pls, ... the other should go to the front right.<|im_end|>\\n<|im_start|>user\\n<IMAGES>\\nsplit up pls, half go front n the rest go front right<|im_end|>\\n<|im_start|>assistant\\n
    reply:   <think>\\n\\n</think>\\n\\n<tool_call>
Sub-goal switch, Haiku (e 195, t 20, logic):
    context: <|im_start|>assistant\\nyou asked: yo eat the cyan one first, then the yellow one, ty. i see cyan at left, yellow at back right, blue at front left. cyan then yellow. first: eat the cyan cheese.<|im_end|>\\n<|im_start|>user\\n<IMAGES>\\nyo eat the cyan one first, then the yellow one, ty<|im_end|>\\n<|im_start|>assistant\\n
    reply:   <think>\\nnow the yellow one\\n</think>\\n\\n<tool_call>
Sub-goal switch, template (e 754, t 10, blocks): reply "<think>\\npush the huge orange l-shaped block into the row running back to front next.\\n</think>\\n\\n<tool_call>"
Done (e 8, t 67, logic): reply "<think>\\ndone.\\n</think>\\n\\n<tool_call>"; later steps carry "done." as the memory's second line.
Template first step (e 13, t 0, runner):
    reply:   <think>\\nyou asked: hunt the yellow cheese. i see the swarm on low hills, a yellow cheese at the front right and a cyan cheese at the front. that means: catch the yellow cheese and eat it. first: eat the yellow cheese.\\n</think>\\n\\n<tool_call>
Instruction change, Haiku (e 49, t 21, switch:goto>stay):
    context: <|im_start|>assistant\\nyou asked: head to the front right of the floor. i see low hills and a red cheese in the center. front right is the floor corner in front and on the right. first: go to the front right of the floor.<|im_end|>\\n<|im_start|>user\\n<IMAGES>\\nremain where you are now<|im_end|>\\n<|im_start|>assistant\\n
    reply:   <think>\\nyou asked: remain where you are now. i see low hills and one red cheese. remain means every robot holds still. first: stay where you are.\\n</think>\\n\\n<tool_call>
Refusal (e 10, t 0, parallel):
    context: <|im_start|>user\\n<IMAGES>\\nsplit into two groups: one should surround the white cheese, the other should surround the cyan cheese<|im_end|>\\n<|im_start|>assistant\\n
    reply:   <think>\\nyou asked: split into two groups: ... i see white and red cheeses only, no cyan one. the second group has no cyan cheese to surround, so i will not split.\\n</think>\\n\\nthere is no cyan cheese in this scene, so i will not split or move.<|im_end|>

A turned example (user behind the stand-in camera at azimuth 45: turns 1, the user at the world's
right, so the world's "back to front" is the user's "right to left"):
    context: <|im_start|>user\\n<IMAGES>\\n(i'm standing behind the camera)\\nbuild a bridge across the trench<|im_end|>\\n<|im_start|>assistant\\n
    reply:   <think>\\nyou asked: build a bridge across the trench. i see the swarm, a trench running right to left between two plateaus and a yellow cheese on a plateau in the center. that means: build a bridge across the trench. first: build a bridge across the trench.\\n</think>\\n\\n<tool_call>

What images show (visible): the policy sees images, not the scene summary, so every thought (Haiku
and template, and switch lines) is cut to what a viewer can read off them: robot counts ("i see 54
robots" -> "i see the robots", "the other 26 hold" -> "the others hold"), coordinates ("at x 13",
"the row at x = 7" -> "the row running back to front") and measured heights, depths and widths go,
unless the instruction states the number (or, for a non-robot-count number such as a tower's height
or "3 of you", the task's record does); colours, kinds, sizes and coarse places stay. A sentence
still holding a number from nowhere is dropped, or in the "i see" or last sentence the number
becomes "some". Templates say "the swarm" and "other things", never a count.

Join check (mismatch="raise" by default): every label joined by (seed, index) must belong to its
episode: the label's family is the attempt's, its kind and spec are the store's record of the
instruction it labels (refusals: family only), and the exported item it was written from
(items.jsonl) has the attempt's family, kind and robot count. Any problem raises ValueError;
mismatch="drop" instead drops that episode's labels (it keeps its template), lists them in
.mismatches and counts them in counts()["mismatched_episodes"]. On the real stores of 2026-11-10
(voxel_full/s0..s7 with run-s*/run-s*-later) 894 of 30,905 episodes fail (s0 229, s1 305, s2 326,
s4 1, s5 1, s7 32): the stores' object-task draws part from the Haiku export at one index per seed
(s0 715, s1 816, s2 783, s7 2745; demos.draws alone reproduces the export, demos.write does not),
so for now load them with mismatch="drop".
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from beebot.vla.voxel import haiku as H
from beebot.vla.voxel import suite as S
from beebot.vla.voxel.lm import world_at
from beebot.vla.voxel.play_store import load
from beebot.vla.voxel.world import N_CHANNELS, STAY, observe

IMAGES = "<IMAGES>"  # where the scene's images go: exactly once in the user turn (plain text, not a token)
IMAGE_PAD = "<|image_pad|>"  # one per visual token of an image (Qwen3-VL processor layout)
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
SLOT = "<|video_pad|>"  # optional action-slot placeholder (tokenize(slots=n))
ACT = "<tool_call>"  # the act marker (stream.ACT)
THINK = "<think>"
UNTHINK = "</think>"
START = "<|im_start|>"
END = "<|im_end|>"
EMPTY_THOUGHT = f"{THINK}\n\n{UNTHINK}\n\n"
SINGLE_TOKENS = (THINK, UNTHINK, ACT, START, END, IMAGE_PAD, VISION_START, VISION_END, SLOT)
PATCH, MERGE = 16, 2  # Qwen3.5-4B's image processor: 16-pixel patches merged 2 x 2 into one visual token
MIN_PIXELS, MAX_PIXELS = 65536, 16777216  # its preprocessor_config.json size bounds
ISO_SCALE = 8  # the stand-in view: render.render_scene at 8 pixels a voxel, 272 x 272 (81 visual tokens)
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


def context(instruction, memory=(), system=None, standing=None):
    """The text before the reply: system, memory (earlier assistant text) and the user turn with
    IMAGES, where the user stands (standing, a USER_POSITIONS sentence), then the instruction. It
    ends exactly at the assistant's turn: the reply starts with "<think>\n"."""
    out = f"{START}system\n{system}{END}\n" if system else ""
    if memory:
        out += f"{START}assistant\n" + "\n".join(memory) + f"{END}\n"
    where = f"{standing}\n" if standing else ""
    return out + f"{START}user\n{IMAGES}\n{where}{instruction.strip()}{END}\n{START}assistant\n"


def stand_in_views(obs, rng=None, scale=ISO_SCALE):
    """[one isometric image] of an observation (channels, D, H, W), drawn as render.render_scene
    draws (robots grey, terrain sandy, objects in their per-voxel colour; no zones), ignoring rng.
    It stands in for views.scene_views(obs, rng, ViewSpec()) until that exists; ThinkData's
    `views` argument switches."""
    from PIL import Image, ImageDraw

    from beebot.vla.voxel.render import BACKGROUND, ROBOT, SHADES, WALL

    grid = obs.numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs)
    robots, walls, things = grid[0] > 0.5, grid[1] > 0.5, grid[2] > 0.5
    solid = robots | walls | things
    covered = np.zeros_like(solid)  # solid on all three visible sides (+z, +y, +x): never seen
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
        for face, k in zip(([(cx, cy - s), (cx + s, cy - s / 2), (cx, cy), (cx - s, cy - s / 2)], [(cx - s, cy - s / 2), (cx, cy), (cx, cy + s), (cx - s, cy + s / 2)], [(cx, cy), (cx + s, cy - s / 2), (cx + s, cy + s / 2), (cx, cy + s)]), SHADES):
            draw.polygon(face, fill=tuple(int(c * k) for c in rgb))
    return [image], {"azimuth": STAND_IN_AZIMUTH}


def views_and_pose(result):
    """(images, camera azimuth in degrees) from what a views function returns: a list of images
    (the camera then counts as the default viewer's, azimuth 0) or (images, pose) with
    pose["azimuth"]."""
    if isinstance(result, tuple):
        images, pose = result
        return list(images), float((pose or {}).get("azimuth", 0.0))
    return list(result), 0.0


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
    terrain = summary["terrain"]
    swarm = "the swarm"  # no count: the policy sees images, and a count cannot be read off them
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
        parts.append("other things")
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


COORDINATE = r"\(?\b[xyz]\s*=?\s*\d+(?:\s*(?:to|-)\s*\d+)?(?:\s*,\s*[xyz]\s*=?\s*\d+(?:\s*(?:to|-)\s*\d+)?)*\)?"
ROW_AT = re.compile(r"\b(the|a) row at ([xy])\s*=?\s*\d+")
NEAR_COORDINATE = re.compile(r"(?:,\s*)?\s*\b(?:at|from|to|running|near|around|on|along)?\s*" + COORDINATE)
MEASURE = re.compile(r"(?:,\s*)?\s*\b(?:at\s+)?(?:height|depth|width)\s+(\d+)\b")
COUNT_PATTERNS = (  # (pattern, replacement, numbers it keeps) for counts a viewer cannot read off an image
    (re.compile(r"\ba swarm of (\d+) (?:robots|bots)\b"), "the swarm", "instruction"),
    (re.compile(r"\ball (\d+) (?:robots|bots)\b(?! (?:tall|high))"), "all the robots", "instruction"),
    (re.compile(r"\b(\d+) (?:robots|bots)\b(?! (?:tall|high))"), "the robots", "instruction"),
    (re.compile(r"\ball (\d+) of (us|you)\b"), r"all of \2", "task"),
    (re.compile(r"\b(\d+) of (us|you)\b"), r"all of \2", "task"),
    (re.compile(r"\bthe other (\d+)\b"), "the others", "task"),
    (re.compile(r"\bthe (\d+)\b(?! (?:robots|bots|of|high|tall))"), "the robots", "task"),
    (re.compile(r"\ball (\d+)\b(?! (?:robots|bots|of|high|tall))"), "all of them", "task"),
    (re.compile(r"\bhalf of (\d+) is (?:about )?(\d+)"), "half of them", "task"),
)


def label_problem(label_row, attempt, entries, item=None):
    """Why a label does not belong to the episode it joined (None when it does). Its family must be
    the attempt's; but for a refusal (another instruction for the same scene) its kind and spec those
    of the store's record of the instruction it labels; and the exported item it was written from
    (items.jsonl) must have the attempt's family, kind and robot count, so a label that matches by
    chance after a misaligned draw (another scene with the same task) is caught too."""
    if label_row.get("family") != attempt["family"]:
        return f"family {label_row.get('family')} but the attempt is {attempt['family']}"
    if item is not None:
        record, robots = item["record"], item.get("scene", {}).get("robots", {}).get("count")
        if (record["family"], record["kind"]) != (attempt["family"], attempt["kind"]) or (robots is not None and robots != attempt.get("robots", robots)):
            return f"its item is {record['family']}/{record['kind']} with {robots} robots but the attempt is {attempt['family']}/{attempt['kind']} with {attempt.get('robots')}"
    if label_row.get("rung") == REFUSAL:
        return None
    j = int(label_row.get("part", 0))
    if j >= len(entries):
        return f"part {j} but the episode has {len(entries)} instructions"
    record = entries[j]["record"]
    if label_row.get("kind") != record["kind"] or canonical_json(label_row.get("spec")) != canonical_json(record["spec"]):
        return f"kind {label_row.get('kind')} spec {label_row.get('spec')} but the store's record is kind {record['kind']} spec {record['spec']}"
    return None


def canonical_json(value):
    return json.dumps(value, sort_keys=True)


def numbers_in(value):
    """The digit strings in a value (text, or a record's spec, plain and sub-goals, walked)."""
    if isinstance(value, dict):
        return set().union(*(numbers_in(v) for v in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(numbers_in(v) for v in value)) if value else set()
    if isinstance(value, bool) or value is None:
        return set()
    return set(re.findall(r"\d+", str(value)))


def visible(thought, instruction, record=None):
    """A thought cut down to what images of the scene show: robot counts, coordinates and measured
    heights, depths and widths go (numbers the instruction or the task's record states stay, e.g. a
    tower's height or "3 of you"); colours, kinds, sizes and coarse places are kept. The quoted
    instruction ("you asked: ...") is left as it is. A sentence that still holds a number from
    nowhere is dropped, or in the first or last sentence the number becomes "some"."""
    head = f"you asked: {bare(instruction).lower()}."
    quoted = thought.startswith(head)
    body = thought[len(head) :] if quoted else thought
    stated = numbers_in(instruction)
    keep = stated | numbers_in(record or {})

    def unless_kept(replacement, kept=None):
        kept = keep if kept is None else kept
        return lambda m: m.group(0) if any(g and g.isdigit() and g in kept for g in m.groups()) else m.expand(replacement)

    body = ROW_AT.sub(lambda m: f"{m.group(1)} row running {'back to front' if m.group(2) == 'x' else 'left to right'}", body)
    body = NEAR_COORDINATE.sub("", body)
    body = MEASURE.sub(unless_kept(""), body)
    for pattern, replacement, source in COUNT_PATTERNS:  # a robot count stays only when the instruction states it
        body = pattern.sub(unless_kept(replacement, stated if source == "instruction" else keep), body)
    sentences = [part for part in re.split(r"(?<=\.)\s+", body.strip()) if part]
    out = []
    for k, sentence in enumerate(sentences):
        stray = [n for n in re.findall(r"\b\d+\b", sentence) if n not in keep]
        if stray and 0 < k < len(sentences) - 1 and not sentence.startswith("i see"):
            continue
        for n in stray:
            sentence = re.sub(rf"\b{n}\b", "some", sentence)
        out.append(sentence)
    body = re.sub(r"\s+([,.;:])", r"\1", " ".join(out))
    body = re.sub(r",\s*,", ",", body).replace(" ,", ",")
    return (head + " " + body).strip() if quoted else body


def template_line(record, index):
    """The switch line of plan segment `index` of a record."""
    subgoals = record.get("subgoals") or []
    goal = bare(subgoals[index]) if 0 <= index < len(subgoals) else "the next part"
    return f"{goal} next.".lower()


# The user's frame: directions as the user standing around the arena says them

ORDER = ("front", "right", "back", "left")  # world directions counterclockwise seen from above, from the default viewer's side (+y)
USER_POSITIONS = {  # the user's place relative to the camera: (degrees added to the camera's azimuth, what the user turn says)
    "behind": (0, "(i'm standing behind the camera)"),
    "left": (-90, "(i'm standing to the left of the camera)"),
    "right": (90, "(i'm standing to the right of the camera)"),
    "opposite": (180, "(i'm standing opposite the camera, facing it)"),
}
STAND_IN_AZIMUTH = 45.0  # render.render_scene looks from the front right corner (x to the lower right, y to the lower left)
NOT_DIRECTIONS = (  # uses of the four words that are no direction: left as they are
    r"\b(?:all|alright|that's|thats|you're|youre) right\b",
    r"(?<!\bthe )(?<!\bto the )(?<!\bon the )(?<!\bat the )(?<!\bfront )(?<!\bback )(?<!\bfar )\bright (?:now|away|here|there|next to|beside|behind|in front|where|by|on top|above|below|against|into|onto|under|over)\b",
    r"\b(?:is|are|was|were|nothing|none|anything|what's|whats|still|ones|one|those|any|no|be) left\b(?! (?:side|of|corner|edge|half|to|and|or|plateau|zone|stripe|group|one|area|part))",
    r"\bleft (?:over|alone|behind|untouched|uneaten|out)\b",
    r"\b(?:come|came|comes|coming|go|goes|went|get|gets|got|bring|put|turn|hold|held|holding|fall|pull|give|sit|sitting|stand|standing|settle|settling|lie|lying|set) back\b(?! (?:left|right|of|side|corner|edge|half|row|plateau|area|part))",
    r"\bback (?:off|up|together|down)\b",
)
UNMAPPABLE = re.compile(  # direction words that cannot be turned reliably: used only when the user's frame is the world's
    r"\b(?:near|far|nearer|farther|further|this|that|other|opposite)\s+(?:side|end|edge|corner|wall|half)\b"
    r"|\b(?:north|south|east|west|northern|southern|eastern|western|northeast|northwest|southeast|southwest)\b"
    r"|\b(?:window|door|doorway|camera|viewer)\b"
    r"|\b(?:toward|towards|away from|closest to|nearest to|close to|near|nearest|facing)\s+(?:me|you|us)\b"
    r"|\b(?:top|bottom|upper|lower)\s+(?:left|right|edge|side|corner|half|row|part|end)\b"
    r"|\b(?:left|right)[- ]?hand\b"
    r"|\b(?:front|back)(?:left|right)\b|\bfar(?:left|right|back|front)\b"
    r"|\b(?:clockwise|counterclockwise|anticlockwise)\b"
)
MOTION_VERBS = r"(?:step|steps|move|moves|moving|head|heads|walk|walks|shift|shifts|slide|slides|push|pushes|roll|rolls|nudge|nudges|drift)"
SAID = {  # how a turned word is said in each kind of place: plain, after "in front of"/"behind", after a motion verb or as forward/backward
    "plain": {d: d for d in ORDER},
    "preposition": {"front": "in front of", "back": "behind", "left": "to the left of", "right": "to the right of"},
    "motion": {"front": "forward", "back": "back", "left": "left", "right": "right"},
    "adverb": {"front": "forward", "back": "backward", "left": "to the left", "right": "to the right"},
}


def quarter_turns(degrees):
    """The quarter turn (0 front, 1 right, 2 back, 3 left) of a place around the arena at this
    azimuth, nearest first (a place half way between two goes to the later one)."""
    return int(((float(degrees) + 45.0) % 360.0) // 90.0)


def user_turns(camera_azimuth, position):
    return quarter_turns(camera_azimuth + USER_POSITIONS[position][0])


def turn_word(word, turns):
    """A world direction word as said from the user's side `turns` quarter turns round."""
    return ORDER[(ORDER.index(word) - turns) % 4]


def mappable(text):
    return UNMAPPABLE.search(text) is None


USER_NAMES = tuple(USER_POSITIONS)


def turnable_choice(choice):
    """Whether every text of a label (instruction, thought, switch lines, reply) can be turned."""
    texts = [choice.get("instruction", ""), choice.get("thought", ""), choice.get("reply", "")] + list(choice.get("switches") or [])
    return all(mappable(text) for text in texts)


def turnable(parts):
    return all(turnable_choice(part) for part in parts)


def in_user_frame(text, turns):
    """Text with every direction word turned from the world frame (front = +y, toward the default
    viewer; left = -x) into the frame of a user `turns` quarter turns round the arena (1: at the
    world's right, looking across it). Corners keep front/back first, "in front of" and "behind"
    become the user's word for that side, forward(s) and backward(s) and "back" after a motion verb
    turn too, and "side by side" and "one behind the other" swap on a quarter turn. Every turned
    word is written through a marker, so nothing is turned twice. turns 0 returns the text as it is."""
    if not turns:
        return text
    kept = []

    def put(value):
        kept.append(value)
        return f"\x00{len(kept) - 1}\x00"

    def mark(word, style="plain"):
        return put(SAID[style][turn_word(word, turns)])

    for pattern in NOT_DIRECTIONS:
        text = re.sub(pattern, lambda m: put(m.group(0)), text)
    rows = {"side by side": "one behind the other"} if turns % 2 else {}
    text = re.sub(r"\bside by side\b|\bone behind (?:the other|another)\b", lambda m: put(rows.get(m.group(0), "side by side" if turns % 2 else m.group(0))), text)
    text = re.sub(r"\bin front of\b", lambda m: mark("front", "preposition"), text)
    text = re.sub(r"\bto the (left|right) of\b(?! the (?:floor|arena|area|grid|ground|field|world|map|room|space|board)\b)", lambda m: mark(m.group(1), "preposition"), text)  # beside a thing; of the floor it is a region
    text = re.sub(r"\bbehind\b", lambda m: mark("back", "preposition"), text)
    text = re.sub(r"\bforwards?\b", lambda m: mark("front", "adverb"), text)
    text = re.sub(r"\bbackwards?\b", lambda m: mark("back", "adverb"), text)
    text = re.sub(r"\b(" + MOTION_VERBS + r") back\b", lambda m: m.group(1) + " " + mark("back", "motion"), text)
    text = re.sub(r"\brear\b", "back", text)

    def corner(m):
        first, second = turn_word(m.group(1), turns), turn_word(m.group(3), turns)
        if first in ("left", "right"):
            first, second = second, first
        return put(first + m.group(2) + second)

    text = re.sub(r"\b(front|back)([ -])(left|right)\b", corner, text)
    text = re.sub(r"\b(front|back|left|right)(most|ward|wards)?\b", lambda m: mark(m.group(1)) + (m.group(2) or ""), text)
    text = re.sub("\x00(\\d+)\x00", lambda m: kept[int(m.group(1))], text)
    text = re.sub("\x00(\\d+)\x00", lambda m: kept[int(m.group(1))], text)  # a kept phrase may hold a mark
    return re.sub(r"\bon the (front|back)\b", r"at the \1", text)


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
    images: list
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
    user: str = "behind"  # where the user stands relative to the camera (USER_POSITIONS)
    turns: int = 0  # the user's quarter turn round the arena (0: the world's own frame); every direction word is in that frame
    camera: float = 0.0  # the camera's azimuth (degrees round the arena, 0 at the world's front, 90 at its right)


def still(obs):
    return torch.full(tuple(obs.shape[-3:]), STAY, dtype=torch.long)


def label_options(directory):
    path = Path(directory) / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


class ThinkData:
    def __init__(self, stores, haiku_dirs, tokenizer=None, seed=0, thought_share=1 / 3, haiku_share=None, refusal_share=0.05, held_share=0.05, channels=N_CHANNELS, system=None, views=stand_in_views, validation_views=None, mismatch="raise"):
        if tokenizer is not None:
            check_tokens(tokenizer)
        self.seed = seed
        self.thought_share = thought_share
        self.haiku_share = haiku_share
        self.refusal_share = refusal_share
        self.held_share = held_share
        self.channels = channels
        self.system = system
        self.views = views
        if mismatch not in ("raise", "drop"):
            raise ValueError(f"mismatch is raise or drop, not {mismatch}")
        self.mismatch = mismatch
        self.mismatches = []  # (store, attempt index, problem) of the episodes whose labels were dropped
        self.validation_views = validation_views
        self.stores = [str(s) for s in stores]
        self.data = [load(s) for s in self.stores]
        self.labels = []
        for directory in haiku_dirs:
            directory = Path(directory)
            assert not H.is_test(directory), f"{directory} is the held-out test set; it never goes into training"
            items = {(item["seed"], item["index"]): item for item in H.read_jsonl(directory / "items.jsonl")}
            self.labels.append((H.Labels(directory.name, root=directory.parent), label_options(directory), items))
        self.episodes = []
        self.frames = {"turned": 0, "world": 0, "rung_swapped": 0, "template_instead": 0, "world_frame_forced": 0}
        self.rewritten = {"haiku_thoughts": 0, "haiku_changed": 0, "haiku_lines_changed": 0, "template_thoughts": 0, "template_changed": 0}
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
        for labels, options, items in self.labels:
            if options.get("seed", seed) != seed:
                continue
            for key in OPTION_KEYS:
                if key in options and key in config and options[key] != config[key]:
                    raise ValueError(f"{labels.directory} was drawn with {key}={options[key]} but {path} with {config[key]}: the (seed, index) join would be wrong")
            mine.append((labels, items))
        out = []
        for n, row in enumerate(H.read_jsonl(Path(path) / "attempts.jsonl")):
            if row["e"] < 0:
                continue
            entries = store["instructions"][row["e"]]
            choices, parts, refusals = [], [], []
            problems = []
            for labels, items in mine:
                for label_row in labels.by_key.get((seed, n), []):
                    problem = label_problem(label_row, row, entries, items.get((seed, n)))
                    if problem:
                        problems.append(f"label {label_row.get('pair_id')}: {problem}")
                choices += labels.choices(seed, n)
                parts = parts or labels.switch_parts(seed, n)
                refusals += labels.refusals(seed, n)
            if parts and [p["t"] for p in parts[: len(entries)]] != [int(entry["t"]) for entry in entries]:
                problems.append(f"the Haiku parts start at {[p['t'] for p in parts]} but the store's instructions at {[entry['t'] for entry in entries]}")
            if problems:
                if self.mismatch == "raise":
                    raise ValueError(f"{path} attempt {n}: {problems[0]}; the (seed, index) join is broken ({len(problems)} problems on this episode; mismatch='drop' falls back to templates)")
                self.mismatches.append((path, n, problems[0]))
                choices, parts, refusals = [], [], []
            record = entries[0]["record"]
            choices = [self.visibly(choice, record) for choice in choices]
            parts = [dict(part, choices=[self.visibly(choice, entries[part["part"]]["record"]) for choice in part["choices"]]) for part in parts[: len(entries)]]
            refusals = [self.visibly(pair, None) for pair in refusals]
            out.append(Episode(k, int(row["e"]), seed, n, row["family"], held_out(seed, n, self.held_share), choices, parts, refusals))
        return out

    def visibly(self, choice, record):
        """A Haiku choice (or refusal pair) with its thought and lines cut to what images show
        (visible); counts the thoughts changed."""
        out = dict(choice, thought=visible(choice["thought"], choice["instruction"], record))
        if "switches" in choice:
            out["switches"] = [visible(line, choice["instruction"], record) for line in choice["switches"]]
        self.rewritten["haiku_thoughts"] += 1
        self.rewritten["haiku_changed"] += out["thought"] != choice["thought"]
        self.rewritten["haiku_lines_changed"] += out.get("switches") != choice.get("switches")
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
                raw = template_thought(entry["text"], record, summary)
                thought = visible(raw, entry["text"], record)
                self.rewritten["template_thoughts"] += 1
                self.rewritten["template_changed"] += thought != raw
                switches = [visible(line, entry["text"], record) for line in switches]
                parts.append({"rung": TEMPLATE, "instruction": entry["text"], "thought": thought, "switches": switches, "done": DONE_LINE})
            self.templates[key] = parts
        return self.templates[key]

    def labelling(self, ep, rng, usable=None):
        """One choice per instruction: Haiku (a rung drawn per part, among those `usable` accepts)
        or the template. None when a part has no usable choice."""
        def pick(choices):
            pool = [c for c in choices if usable is None or usable(c)]
            return pool[int(rng.integers(len(pool)))] if pool else None

        if ep.parts:
            parts = [pick(part["choices"]) for part in ep.parts[: len(self.instructions(ep))]]
        elif ep.choices:
            parts = [pick(ep.choices)]
        else:
            parts = [part if usable is None or usable(part) else None for part in self.template_parts(ep)]
        return None if any(part is None for part in parts) else parts

    def user_frame(self, ep, parts, rng, azimuth):
        """(parts, user position, quarter turns): the user's place drawn round the camera; labels
        whose direction words cannot be turned (UNMAPPABLE) are used only in the world's frame, so
        another rung, then the template, then the position that makes the frame the world's."""
        position = USER_NAMES[int(rng.integers(len(USER_NAMES)))] if rng is not None else "behind"
        turns = user_turns(azimuth, position)
        if turns == 0 or turnable(parts):
            self.frames["turned" if turns else "world"] += 1
            return parts, position, turns
        other = self.labelling(ep, rng, turnable_choice) if ep.labelled else None
        if other is not None:
            self.frames["rung_swapped"] += 1
            return other, position, turns
        template = self.template_parts(ep)
        if ep.labelled and turnable(template):
            self.frames["template_instead"] += 1
            return template, position, turns
        self.frames["world_frame_forced"] += 1
        position = next(name for name in USER_NAMES if user_turns(azimuth, name) == 0)
        return parts, position, 0

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

    def example(self, ep, t, parts, rng=None, views=None):
        """The act example of episode ep at step t under labelling parts (one per instruction);
        rng: the generator the views draw their randomization from."""
        events = self.episode_events(ep)
        starts = [int(entry["t"]) for entry in self.instructions(ep)]
        j = part_at(starts, t)
        here = next((ev for ev in events if ev.t == t), None)
        codes = self.data[ep.store]["actions"][ep.e, t].long()
        obs = self.observation(ep, t)
        images, azimuth = views_and_pose((views or self.views)(obs, rng))
        parts, position, turns = self.user_frame(ep, parts, rng, azimuth)
        memory = [in_user_frame(self.thought_of(ep, parts, ev), turns) for ev in memory_before(events, t)]
        thought = in_user_frame(self.thought_of(ep, parts, here), turns) if here is not None else ""
        return Example(
            obs=obs,
            images=images,
            context_text=context(in_user_frame(parts[j]["instruction"], turns), memory, self.system, USER_POSITIONS[position][1]),
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
            instruction=in_user_frame(parts[j]["instruction"], turns),
            store=self.stores[ep.store],
            user=position,
            turns=turns,
            camera=azimuth,
        )

    def refusal(self, ep, pair, rng=None, views=None):
        obs = self.observation(ep, 0)
        images, azimuth = views_and_pose((views or self.views)(obs, rng))
        position = USER_NAMES[int(rng.integers(len(USER_NAMES)))] if rng is not None else "behind"
        turns = user_turns(azimuth, position)
        if turns and not turnable_choice(pair):
            self.frames["world_frame_forced"] += 1
            position, turns = next(name for name in USER_NAMES if user_turns(azimuth, name) == 0), 0
        else:
            self.frames["turned" if turns else "world"] += 1
        return Example(
            obs=obs,
            images=images,
            context_text=context(in_user_frame(pair["instruction"], turns), (), self.system, USER_POSITIONS[position][1]),
            reply_text=text_reply(in_user_frame(pair["thought"], turns), in_user_frame(pair["reply"], turns)),
            act=False,
            codes=None,
            seed=ep.seed,
            index=ep.index,
            e=ep.e,
            t=0,
            rung=REFUSAL,
            family=ep.family,
            kind=REFUSAL,
            instruction=in_user_frame(pair["instruction"], turns),
            store=self.stores[ep.store],
            user=position,
            turns=turns,
            camera=azimuth,
        )

    def draw(self, pool, rng, views=None):
        """One example from a pool of episodes (Pool: the sampler of sample and validation)."""
        if pool.refusing and self.refusal_share and rng.random() < self.refusal_share:
            ep = pool.refusing[int(rng.integers(len(pool.refusing)))]
            return self.refusal(ep, ep.refusals[int(rng.integers(len(ep.refusals)))], rng, views)
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
        return self.example(ep, t, self.labelling(ep, rng), rng, views)

    def sample(self, rng):
        return self.draw(self.train_pool, rng)

    def validation(self, n, seed=0):
        key = (n, seed)
        if key not in self.validation_cache:
            if not self.held:
                raise ValueError("no held-out episodes (too few episodes for held_share)")
            rng = np.random.default_rng(seed)
            self.validation_cache[key] = [self.draw(self.held_pool, rng, self.validation_views) for _ in range(n)]
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
        out["mismatched_episodes"] = len(self.mismatches)  # labels dropped (mismatch="drop")
        out["mismatched_per_store"] = {path: sum(m[0] == path for m in self.mismatches) for path in self.stores}
        out["rewritten"] = dict(self.rewritten)
        out["frames"] = dict(self.frames)  # examples drawn so far: in a turned frame, in the world's, and how unturnable labels were handled  # thoughts visible() changed (templates: those built so far)
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


def image_tokens(image, patch=PATCH, merge=MERGE, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Visual tokens of a PIL image after Qwen3.5's resize (smart_resize to multiples of
    patch x merge pixels within the pixel bounds): one per merge x merge patches."""
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import smart_resize

    height, width = smart_resize(image.height, image.width, factor=patch * merge, min_pixels=min_pixels, max_pixels=max_pixels)
    return (height // (patch * merge)) * (width // (patch * merge))


def tokenize(example, tokenizer, counts=None, slots=0):
    """(input_ids, labels): the context with IMAGES expanded as the Qwen3-VL processor does,
    <|vision_start|> + <|image_pad|> x n + <|vision_end|> per image (n: counts, or image_tokens of
    each of example.images), then the reply (labels: the reply's ids, -100 elsewhere); slots: that
    many SLOT ids after an act reply (label -100), for a model that wants its action slots in the
    id sequence."""
    if example.context_text.count(IMAGES) != 1:
        raise ValueError(f"the context holds {example.context_text.count(IMAGES)} {IMAGES} placeholders, not one")
    before, after = example.context_text.split(IMAGES)
    counts = [image_tokens(image) for image in example.images] if counts is None else list(counts)
    pad, opening, closing = (token_id(tokenizer, text) for text in (IMAGE_PAD, VISION_START, VISION_END))
    vision = [i for n in counts for i in [opening] + [pad] * n + [closing]]
    encode = lambda text: tokenizer(text, add_special_tokens=False).input_ids  # noqa: E731
    head = encode(before) + vision + encode(after)
    reply = encode(example.reply_text)
    tail = [token_id(tokenizer, SLOT)] * slots if slots and example.act else []
    ids = head + reply + tail
    labels = [-100] * len(head) + reply + [-100] * len(tail)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def render_examples(data, rng=None):
    """One rendered example of each kind found ({kind: Example}), for docs and checks."""
    rng = np.random.default_rng(0) if rng is None else rng
    out = {}
    for ep in data.episodes:
        source = ":haiku" if ep.labelled else ":template"
        parts = data.labelling(ep, rng)
        events = data.episode_events(ep)
        for ev in events:
            if ev.kind + source not in out:
                out[ev.kind + source] = data.example(ep, ev.t, parts, rng)
        plain = [t for t in range(data.steps(ep)) if t not in {ev.t for ev in events} and memory_before(events, t)]
        if plain and "plain" + source not in out:
            out["plain" + source] = data.example(ep, plain[len(plain) // 2], parts, rng)
        if ep.refusals and REFUSAL not in out:
            out[REFUSAL] = data.refusal(ep, ep.refusals[0], rng)
    return out


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description="counts, rendered examples and sampling speed of ThinkData")
    parser.add_argument("--stores", nargs="+", required=True)
    parser.add_argument("--haiku", nargs="*", default=[])
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--show", action="store_true", help="print one rendered example of each kind")
    parser.add_argument("--mismatch", choices=("raise", "drop"), default="raise", help="labels that do not belong to their episode: raise, or drop them (the episode keeps its template)")
    args = parser.parse_args()
    torch.set_num_threads(1)
    t0 = time.time()
    data = ThinkData(args.stores, args.haiku, mismatch=args.mismatch)
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
