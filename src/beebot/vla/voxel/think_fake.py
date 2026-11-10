"""A stand-in for think_data.ThinkData with the same interface, for tests, smoke runs and throughput
measurements before the real training examples exist: random sparse scenes, and a learnable rule.
An acting example's instruction names one of the 27 action codes, its thought repeats it among
filler words, and every voxel's code is that one; a text example asks for something else and gets a
fixed refusal after an empty thought. With images > 0 each example carries that many random images
(field images, PIL) at an <IMAGES> place in its context; scene puts the voxel scene tokens (SCENE)
there too.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from beebot.vla.voxel.think import chat_context
from beebot.vla.voxel.world import N_ACTIONS, N_CHANNELS

FILLER = "the robots are near a wall and a red cube while the blue zone stays empty so we keep going slowly together".split()
ASKS = ("sing me a song", "what is the capital of France", "write a poem about cheese", "tell me a joke")
REFUSAL = "I can only move the robots in this scene."


@dataclass
class Example:
    obs: torch.Tensor  # (C, D, H, W)
    context_text: str  # with SCENE once
    reply_text: str
    act: bool
    codes: torch.Tensor | None  # (D, H, W) long, acting examples only
    metadata: dict = field(default_factory=dict)
    images: list = field(default_factory=list)  # PIL images


class FakeThinkData:
    """side: grid side; act_share: share of acting examples; instruction_words, thought_words:
    words of filler in the instruction and the thought (about 1.1 tokens each); images, image_size:
    random images per example and their side in pixels; scene: SCENE in the context."""

    def __init__(self, tokenizer=None, seed=0, side=16, channels=N_CHANNELS, act_share=0.8, instruction_words=18, thought_words=50, robot_share=0.03, images=0, image_size=256, scene=True):
        self.side, self.channels, self.act_share = side, channels, act_share
        self.images, self.image_size, self.scene = images, image_size, scene
        self.instruction_words, self.thought_words, self.robot_share = instruction_words, thought_words, robot_share

    def filler(self, rng, n):
        return " ".join(FILLER[i] for i in rng.integers(len(FILLER), size=n))

    def sample(self, rng):
        d = self.side
        obs = (torch.from_numpy(rng.random((self.channels, d, d, d))) < 0.05).float()
        obs[0] = torch.from_numpy(rng.random((d, d, d)) < self.robot_share).float()
        act = bool(rng.random() < self.act_share)
        code = int(rng.integers(N_ACTIONS))
        if act:
            instruction = f"take move {code} with every robot, {self.filler(rng, self.instruction_words)}"
            reply = f"<think>\nEvery robot takes move {code}. {self.filler(rng, self.thought_words)}\n</think>\n\n<tool_call>"
        else:
            instruction = f"{ASKS[int(rng.integers(len(ASKS)))]}, {self.filler(rng, self.instruction_words)}"
            reply = f"<think>\n\n</think>\n\n{REFUSAL}<|im_end|>"
        codes = torch.full((d, d, d), code, dtype=torch.long) if act else None
        pictures = [Image.fromarray(rng.integers(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8)) for _ in range(self.images)]
        return Example(obs, chat_context(instruction, images=self.images > 0, scene=self.scene), reply, act, codes, {"code": code}, pictures)

    def validation(self, n, seed):
        rng = np.random.default_rng(10_000 + seed)
        return [self.sample(rng) for _ in range(n)]
